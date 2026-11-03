"""The pre-registered methods (baselines) as score builders over a ``PairSlice``.

Every method yields a pair of confidence arrays ``(s_conf, l_conf)`` and is certified by
the same Learn-then-Test procedure (``certify``). Methods that need fitting (the query-only
router) are fit on split A only. Oracles use the labels and are descriptive upper bounds,
never certified, never deployable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.optimize import minimize

from sieve.eval._util import stable_unit
from sieve.eval.data import PairSlice

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]

#: single-token-signal candidates for the primary gate, chosen on split A (max AUROC)
PRIMARY_CANDIDATES: tuple[str, ...] = (
    "max_prob",
    "mean_logprob",
    "neg_mean_entropy",
    "mean_margin",
)
#: pre-registered gate baselines: max-softmax, mean-logprob, entropy, margin
GATE_BASELINES: dict[str, str] = {
    "gate:max_softmax": "max_prob",
    "gate:mean_logprob": "mean_logprob",
    "gate:entropy": "neg_mean_entropy",
    "gate:margin": "mean_margin",
}


@dataclass(frozen=True)
class MethodDef:
    name: str
    kind: str  # gate | primary | s_only | l_only | random | query | sc
    signal: str | None = None
    pre_route: bool = False  # routing decided before running S: escalations waste no S time
    needs_samples: bool = False  # restricted to the self-consistency subset
    deployable: bool = True  # False for random / query / oracle style references


def default_methods(include_sc: bool = True) -> list[MethodDef]:
    ms = [
        MethodDef("S_only", "s_only", "primary"),
        MethodDef("L_only", "l_only", "primary"),
        MethodDef("gate:primary", "primary", "primary"),
        *(MethodDef(n, "gate", s) for n, s in GATE_BASELINES.items()),
        MethodDef("random_router", "random", "primary", pre_route=True, deployable=False),
        MethodDef("query_router", "query", "primary", pre_route=True, deployable=False),
    ]
    if include_sc:
        ms.append(MethodDef("self_consistency", "sc", "primary", needs_samples=True))
    return ms


def resolve_signal(m: MethodDef, primary: str) -> str:
    return primary if m.signal in (None, "primary") else m.signal


# --- query-only router (TF-IDF + logistic regression) ---------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(q: str) -> list[str]:
    w = _TOKEN.findall(q.lower())
    return w + [f"{a}_{b}" for a, b in pairwise(w)]


class TfidfLogisticRouter:
    """P(S answers correctly | question text). 'RouteLLM-style', retrained on split A.

    Features are TF-IDF over word unigrams+bigrams (no embedding model is assumed
    available offline); L2-regularised logistic regression fit by L-BFGS. Deterministic.
    """

    def __init__(self, l2: float = 1.0, min_df: int = 2, max_features: int = 20000) -> None:
        self.l2 = l2
        self.min_df = min_df
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: FloatArr = np.zeros(0)
        self.w: FloatArr = np.zeros(0)
        self.b = 0.0

    def _matrix(self, questions: list[str]) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for r, q in enumerate(questions):
            for tok, c in Counter(t for t in _tokens(q) if t in self.vocab).items():
                j = self.vocab[tok]
                rows.append(r)
                cols.append(j)
                vals.append((1.0 + math.log(c)) * self.idf[j])
        m = sparse.csr_matrix((vals, (rows, cols)), shape=(len(questions), len(self.vocab)))
        norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return sparse.diags(1.0 / norms) @ m

    def fit(self, questions: list[str], y: BoolArr) -> TfidfLogisticRouter:
        df: Counter[str] = Counter()
        for q in questions:
            df.update(set(_tokens(q)))
        keep = sorted((t for t, c in df.items() if c >= self.min_df), key=lambda t: (-df[t], t))[
            : self.max_features
        ]
        self.vocab = {t: i for i, t in enumerate(sorted(keep))}
        n = len(questions)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in sorted(keep)])
        x = self._matrix(questions)
        yy = y.astype(np.float64)
        d = x.shape[1]

        def f(wb: FloatArr) -> tuple[float, FloatArr]:
            z = x @ wb[:d] + wb[d]
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            eps = 1e-12
            loss = -np.mean(yy * np.log(p + eps) + (1 - yy) * np.log(1 - p + eps))
            loss += 0.5 * self.l2 * float(wb[:d] @ wb[:d]) / n
            g = np.empty_like(wb)
            r = (p - yy) / n
            g[:d] = x.T @ r + self.l2 * wb[:d] / n
            g[d] = r.sum()
            return float(loss), g

        res = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 200})
        self.w, self.b = res.x[:d], float(res.x[d])
        return self

    def predict(self, questions: list[str]) -> FloatArr:
        z = self._matrix(questions) @ self.w + self.b
        return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))), dtype=np.float64)


# --- fitted methods ---------------------------------------------------------------------


@dataclass
class FittedMethod:
    """A method after fitting on A: maps any slice to confidence arrays and costs."""

    defn: MethodDef
    signal: str
    router: TfidfLogisticRouter | None = None
    random_seed: int = 0

    def eligible(self, sl: PairSlice) -> BoolArr:
        """Items the method can be evaluated on (graded in both tiers; samples if needed)."""
        m = sl.both_graded()
        if self.defn.needs_samples:
            m = m & sl.s.has_samples
        return np.asarray(m, dtype=np.bool_)

    def scores(self, sl: PairSlice) -> tuple[FloatArr, FloatArr]:
        k = self.defn.kind
        l_conf = sl.l.signals[self.signal].copy()
        if k in ("gate", "primary", "s_only", "l_only"):
            return sl.s.signals[self.signal].copy(), l_conf
        if k == "random":
            u = np.array([stable_unit("random_router", self.random_seed, i) for i in sl.ids])
            return u, l_conf
        if k == "query":
            assert self.router is not None
            return self.router.predict(sl.questions), l_conf
        if k == "sc":
            return sl.s.signals["agreement"].copy(), l_conf
        raise ValueError(f"unknown method kind {k!r}")

    def grids_fixed(self) -> tuple[bool, bool]:
        """(t1 is NEVER only, t2 is NEVER only)."""
        return self.defn.kind == "l_only", self.defn.kind == "s_only"

    def costs(self, sl: PairSlice) -> tuple[FloatArr, FloatArr]:
        """Per-item (cost if S runs, cost if L runs) in the configured cost unit."""
        cs = sl.cost_s()
        if self.defn.kind == "sc":
            cs = cs + sl.sample_cost_s()  # k extra decodes are paid on every item
        return cs, sl.cost_l()


def fit_method(defn: MethodDef, primary: str, a_slice: PairSlice, seed: int) -> FittedMethod:
    sig = resolve_signal(defn, primary)
    fm = FittedMethod(defn, sig, random_seed=seed)
    if defn.kind == "query":
        mask = fm.eligible(a_slice)
        sel = np.flatnonzero(mask)
        qs = [a_slice.questions[i] for i in sel]
        fm.router = TfidfLogisticRouter().fit(qs, ~a_slice.s.wrong[sel])
    return fm


def policy_item_costs(
    fm: FittedMethod,
    sl: PairSlice,
    by_s: BoolArr,
) -> FloatArr:
    """Per-item cost of the policy. Gates run S first, so every item pays S (the wasted S
    attempt on escalated items is charged); pre-routing methods run only the chosen tier."""
    cs, cl = fm.costs(sl)
    if fm.defn.kind == "s_only":
        return cs  # L is never run
    if fm.defn.kind == "l_only":
        return cl  # S is never run
    if fm.defn.pre_route:
        return np.asarray(np.where(by_s, cs, cl))
    return np.asarray(cs + np.where(by_s, 0.0, cl))


def signal_auroc_on(sl: PairSlice, signal: str, tier: str = "s") -> float:
    from sieve.calibrate.metrics import auroc

    v = sl.s if tier == "s" else sl.l
    m = v.graded & np.isfinite(v.signals[signal])
    if m.sum() < 2 or v.correct[m].all() or (~v.correct[m]).all():
        return float("nan")
    return auroc(v.signals[signal][m], v.correct[m])


def choose_primary(a_slice: PairSlice) -> str:
    """Primary signal = highest S-tier AUROC on split A among the candidates (ties: first)."""
    best, best_a = PRIMARY_CANDIDATES[0], -1.0
    for s in PRIMARY_CANDIDATES:
        a = signal_auroc_on(a_slice, s)
        if not math.isnan(a) and a > best_a + 1e-12:
            best, best_a = s, a
    return best

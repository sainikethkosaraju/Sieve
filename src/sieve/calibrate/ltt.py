"""Learn-then-Test threshold selection for the full S -> L -> abstain cascade.

Statement certified (and nothing stronger): with probability >= 1 - delta over the draw
of the calibration table, *every* policy in the returned valid set has
``P(wrong | answered) <= alpha`` (``definition="selective"``) or
``P(wrong and answered) <= alpha`` (``"unconditional"``), for exchangeable traffic,
a frozen model/quant/prompt/sampler and the stated grader.

Why a binomial test is exact here. For a *fixed* policy the answered set is
``{i : policy answers i}``. By exchangeability, conditional on its size the answered items
are i.i.d. draws from the conditional law, so the number of wrong answers is
``Binomial(n_answered, r)`` with ``r = P(wrong | answered)``. Testing ``H0: r > alpha``
with the exact binomial lower tail is therefore a valid p-value (equivalent to the
Clopper-Pearson upper bound). Both tiers run on every calibration item, so the loss of
the *composed* policy is evaluated directly and no K*alpha union bound is needed.

Validity conditions that the caller must respect:
* the family (``path``) is fixed before the p-values are seen; ``make_path`` may use a
  separate split A, never the table being tested;
* the table is used once; signal, prompt, alpha and grid must not be tuned on it.
"""

from __future__ import annotations

import enum
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import beta, binom

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]
Definition = Literal["selective", "unconditional"]
Method = Literal["fixed_sequence", "bonferroni"]
PValue = Literal["exact", "hb"]

NEVER = math.inf  # threshold meaning "never answer at this tier"


class FeasibilityOutcome(enum.Enum):
    FEASIBLE = "feasible"
    EMPTY = "empty"  # no policy could be certified at (alpha, delta); report, do not hide


# --- Tests and bounds ---------------------------------------------------------------


def binom_pvalue(k: int, n: int, alpha: float) -> float:
    """P(Bin(n, alpha) <= k): exact p-value for H0 'true risk > alpha'. n = 0 -> 1."""
    if n == 0:
        return 1.0
    return float(binom.cdf(k, n, alpha))


def cp_upper(k: int, n: int, delta: float) -> float:
    """One-sided (1 - delta) Clopper-Pearson upper bound on a binomial proportion."""
    if n == 0 or k >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def _h1(a: float, b: float) -> float:
    def xlogy(x: float, y: float) -> float:
        return 0.0 if x == 0.0 else x * math.log(x / y)

    return xlogy(a, b) + xlogy(1.0 - a, 1.0 - b)


def hb_pvalue(risk_hat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value (Angelopoulos et al., LTT) for a loss in [0, 1]."""
    if n == 0:
        return 1.0
    r = min(risk_hat, alpha)
    hoeffding_bennett = math.exp(-n * _h1(r, alpha))
    bentkus = math.e * float(binom.cdf(math.ceil(n * risk_hat - 1e-12), n, alpha))
    return min(1.0, hoeffding_bennett, bentkus)


def min_certifiable_alpha(n: int, delta: float) -> float:
    """Smallest alpha certifiable with zero errors among ``n`` answered items."""
    if n <= 0:
        return 1.0
    return float(1.0 - delta ** (1.0 / n))


# --- Calibration table and policy ---------------------------------------------------


@dataclass(frozen=True)
class CalTable:
    """Per-item outcomes with *both* tiers run on *every* item.

    ``*_conf``: larger = more confident; NaN means the gate would have failed closed on
    that item and is treated as ``-inf`` (the tier never answers it).
    ``*_wrong``: grader verdict for that tier's answer. Costs are optional.
    """

    s_conf: FloatArr
    l_conf: FloatArr
    s_wrong: BoolArr
    l_wrong: BoolArr
    cost_s: FloatArr | None = None
    cost_l: FloatArr | None = None

    def __post_init__(self) -> None:
        n = self.s_conf.size
        if n == 0 or self.s_conf.ndim != 1:
            raise ValueError("table must be non-empty and 1-D")
        for a in (self.l_conf, self.s_wrong, self.l_wrong):
            if a.shape != self.s_conf.shape:
                raise ValueError("table columns must have equal length")
        for a in (self.s_conf, self.l_conf):
            if np.any(np.isposinf(a)):
                raise ValueError("+inf confidence is not allowed")
        for c in (self.cost_s, self.cost_l):
            if c is not None and (c.shape != self.s_conf.shape or np.any(c < 0)):
                raise ValueError("bad cost column")

    @property
    def n(self) -> int:
        return int(self.s_conf.size)

    def digest(self) -> str:
        h = hashlib.sha256()
        for a in (self.s_conf, self.l_conf, self.s_wrong, self.l_wrong):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class Outcome:
    by_s: BoolArr
    by_l: BoolArr
    answered: BoolArr
    wrong: BoolArr  # wrong among answered (False where abstained)

    @property
    def escalated(self) -> BoolArr:
        return ~self.by_s


def evaluate_policy(table: CalTable, t1: float, t2: float) -> Outcome:
    """Policy: answer at S if ``s >= t1``; else at L if ``l >= t2``; else abstain.

    This is the offline twin of ``sieve.gate``; a test pins them together.
    """
    s = np.where(np.isnan(table.s_conf), -np.inf, table.s_conf)
    l_ = np.where(np.isnan(table.l_conf), -np.inf, table.l_conf)
    by_s = s >= t1
    by_l = ~by_s & (l_ >= t2)
    answered = by_s | by_l
    wrong = (by_s & table.s_wrong) | (by_l & table.l_wrong)
    return Outcome(by_s=by_s, by_l=by_l, answered=answered, wrong=wrong)


# --- Path / family ------------------------------------------------------------------


def make_path(
    t1_grid: Sequence[float],
    t2_grid: Sequence[float],
    table_a: CalTable,
    *,
    min_coverage: float = 0.05,
    frontier: bool = True,
    definition: Definition = "selective",
) -> list[tuple[float, float]]:
    """Fixed sequence of policies, most conservative first, built from split A only.

    Points with A-coverage below ``min_coverage`` are dropped (they answer too few items
    to certify and would halt a fixed-sequence walk immediately). With ``frontier=True``
    only non-dominated points survive (no other point has >= coverage and <= risk on A),
    so risk is near-monotone along the path and the walk is long. Order: ascending A
    coverage, ties by (t1 desc, t2 desc) for determinism. The path is a function of
    split A and the grids only, never of the table that is later tested.
    """
    cand: list[tuple[float, float, float, float]] = []  # (cov, risk, t1, t2)
    for a in t1_grid:
        for b in t2_grid:
            out = evaluate_policy(table_a, float(a), float(b))
            n_ans = int(out.answered.sum())
            cov = n_ans / table_a.n
            if n_ans == 0 or cov < min_coverage:
                continue
            k = int(out.wrong.sum())
            risk = k / n_ans if definition == "selective" else k / table_a.n
            cand.append((cov, risk, float(a), float(b)))
    if frontier:
        cand = [
            c
            for c in cand
            if not any(o[0] >= c[0] and o[1] <= c[1] and (o[0] > c[0] or o[1] < c[1]) for o in cand)
        ]
    cand.sort(key=lambda c: (c[0], c[1], -c[2], -c[3]))
    return [(c[2], c[3]) for c in cand]


@dataclass(frozen=True)
class PointResult:
    t1: float
    t2: float
    n_answered: int
    k_wrong: int
    n: int
    risk_hat: float
    p_value: float
    rejected: bool
    upper_bound: float  # one-sided CP bound at level delta (not multiplicity-adjusted)


@dataclass(frozen=True)
class LTTResult:
    outcome: FeasibilityOutcome
    valid: tuple[PointResult, ...]  # certified set V (may be empty)
    tested: tuple[PointResult, ...]
    alpha: float
    delta: float
    definition: Definition
    method: Method
    n_grid: int  # m
    n_cal: int
    feasibility_floor: float  # smallest alpha certifiable with 0 errors at max answered n seen
    max_answered_tested: int

    @property
    def empty(self) -> bool:
        return self.outcome is FeasibilityOutcome.EMPTY


def _test_point(
    table: CalTable,
    p: tuple[float, float],
    alpha: float,
    delta: float,
    definition: Definition,
    pvalue: PValue,
    level: float,
) -> PointResult:
    out = evaluate_policy(table, *p)
    n_ans = int(out.answered.sum())
    k = int(out.wrong.sum())
    n = n_ans if definition == "selective" else table.n
    risk_hat = k / n if n else 1.0
    pv = binom_pvalue(k, n, alpha) if pvalue == "exact" else hb_pvalue(risk_hat, n, alpha)
    return PointResult(
        t1=p[0],
        t2=p[1],
        n_answered=n_ans,
        k_wrong=k,
        n=n,
        risk_hat=risk_hat,
        p_value=pv,
        rejected=pv <= level,
        upper_bound=cp_upper(k, n, delta),
    )


def fixed_sequence_ltt(
    table: CalTable,
    path: Sequence[tuple[float, float]],
    *,
    alpha: float,
    delta: float,
    definition: Definition = "selective",
    method: Method = "fixed_sequence",
    pvalue: PValue = "exact",
) -> LTTResult:
    """Certify policies on ``path`` against risk <= alpha at confidence 1 - delta.

    ``fixed_sequence``: walk ``path`` in order, certify while ``p <= delta``, stop at the
    first failure (FWER <= delta, no multiplicity cost). ``bonferroni``: test every point
    at ``delta / m`` (use for an unordered 2-D grid). ``outcome`` is ``EMPTY`` when
    nothing is certified; callers must surface that rather than fall back.
    """
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("alpha and delta must be in (0, 1)")
    if len(path) == 0:
        raise ValueError("empty path")
    m = len(path)
    tested: list[PointResult] = []
    if method == "fixed_sequence":
        for p in path:
            r = _test_point(table, p, alpha, delta, definition, pvalue, delta)
            tested.append(r)
            if not r.rejected:
                break
        valid = [r for r in tested if r.rejected]
    else:
        tested = [_test_point(table, p, alpha, delta, definition, pvalue, delta / m) for p in path]
        valid = [r for r in tested if r.rejected]
    max_n = max((r.n for r in tested), default=0)
    return LTTResult(
        outcome=FeasibilityOutcome.FEASIBLE if valid else FeasibilityOutcome.EMPTY,
        valid=tuple(valid),
        tested=tuple(tested),
        alpha=alpha,
        delta=delta,
        definition=definition,
        method=method,
        n_grid=m,
        n_cal=table.n,
        feasibility_floor=min_certifiable_alpha(
            max_n, delta if method == "fixed_sequence" else delta / m
        ),
        max_answered_tested=max_n,
    )


# --- Cost-minimising pick (separate split) ------------------------------------------


@dataclass(frozen=True)
class CostPick:
    point: PointResult
    coverage: float
    cost_total: float
    cost_per_answered: float
    escalation_rate: float


def policy_cost(table: CalTable, t1: float, t2: float) -> tuple[float, float, float]:
    """Return ``(coverage, total_cost, escalation_rate)``; S always runs, L on escalation."""
    if table.cost_s is None or table.cost_l is None:
        raise ValueError("table has no cost columns")
    out = evaluate_policy(table, t1, t2)
    esc = out.escalated
    total = float(table.cost_s.sum() + table.cost_l[esc].sum())
    return float(out.answered.mean()), total, float(esc.mean())


def pick_min_cost(
    result: LTTResult, table_cost: CalTable, *, min_coverage: float = 0.0
) -> CostPick | None:
    """Choose the cheapest certified policy per answered query on a held-out split.

    Any point of ``result.valid`` keeps the guarantee, so the choice is free; it must
    use data other than the table that produced ``result``. Returns ``None`` if no
    certified policy reaches ``min_coverage`` on this split.
    """
    best: CostPick | None = None
    for pt in result.valid:
        cov, total, esc = policy_cost(table_cost, pt.t1, pt.t2)
        if cov <= 0.0 or cov < min_coverage:
            continue
        pick = CostPick(pt, cov, total, total / (cov * table_cost.n), esc)
        if best is None or (pick.cost_per_answered, -pick.coverage) < (
            best.cost_per_answered,
            -best.coverage,
        ):
            best = pick
    return best

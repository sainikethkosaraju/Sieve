"""Certification: the only module allowed to read split B.

For one re-split ``k`` and one or more (S, L) pairs it (1) does everything that needs only
split A (primary signal, fitted baselines, threshold grids, the fixed LTT path, cost picks),
(2) opens B exactly once, (3) runs fixed-sequence Learn-then-Test on the composed cascade
for every method, (4) picks the cheapest certified policy by cost measured on A, and
(5) records a certification hash in the access ledger, which unlocks split C.
Results are persisted so later analysis reuses them and never re-reads B.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sieve.calibrate import (
    NEVER,
    CalTable,
    FeasibilityOutcome,
    LTTResult,
    PointResult,
    build_artifact,
    dumps_artifact,
    evaluate_policy,
    fixed_sequence_ltt,
    make_path,
)
from sieve.eval._util import atomic_write_text, sha256_obj
from sieve.eval.config import ExperimentConfig, PairSpec
from sieve.eval.data import ExperimentData, PairSlice
from sieve.eval.grading import EM_ID, MMLU_ID
from sieve.eval.methods import (
    FittedMethod,
    MethodDef,
    choose_primary,
    fit_method,
    policy_item_costs,
)
from sieve.eval.splits import CERTIFICATION_CAPABILITY, SplitStore
from sieve.types import SCORE_NAMES, SamplingParams, TierCalibration

FloatArr = npt.NDArray[np.float64]
Q_LEVELS = np.linspace(0.0, 0.95, 12)
MIN_COVERAGE_A = 0.05


def make_grid(scores: FloatArr) -> list[float]:
    """Threshold grid = distinct A-score quantiles plus NEVER (fixed before B is read)."""
    f = scores[np.isfinite(scores)]
    if f.size == 0:
        return [NEVER]
    return [float(x) for x in np.unique(np.quantile(f, Q_LEVELS))] + [NEVER]


def _thr(x: float) -> float | None:
    return None if x == NEVER else float(x)


def _unthr(x: float | None) -> float:
    return NEVER if x is None else float(x)


@dataclass
class MethodCert:
    """Persisted outcome of certifying one method on one pair for one split."""

    method: str
    pair: str
    k: int
    signal: str
    outcome: str  # "feasible" | "empty" | "no_path"
    t1: float | None = None  # None = never answer at S
    t2: float | None = None
    n_B: int = 0
    n_answered_B: int = 0
    k_wrong_B: int = 0
    upper_bound_B: float | None = None
    n_valid: int = 0
    n_grid: int = 0
    feasibility_floor: float | None = None
    a_coverage: float | None = None
    a_cost_per_answered: float | None = None
    note: str = ""
    artifact: dict[str, Any] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @property
    def feasible(self) -> bool:
        return self.outcome == "feasible"


@dataclass
class SplitCertification:
    k: int
    primary_signal: str
    certs: dict[str, dict[str, MethodCert]]  # pair name -> method name -> cert
    bundle_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "primary_signal": self.primary_signal,
            "bundle_hash": self.bundle_hash,
            "certs": {p: {m: c.to_dict() for m, c in d.items()} for p, d in self.certs.items()},
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SplitCertification:
        return SplitCertification(
            int(d["k"]),
            str(d["primary_signal"]),
            {p: {m: MethodCert(**c) for m, c in ms.items()} for p, ms in d["certs"].items()},
            str(d["bundle_hash"]),
        )


def _table(fm: FittedMethod, sl: PairSlice) -> tuple[CalTable, PairSlice]:
    sub = sl.subset(fm.eligible(sl))
    s, l_ = fm.scores(sub)
    return CalTable(s, l_, sub.s.wrong, sub.l.wrong), sub


@dataclass
class _Prepared:
    fm: FittedMethod
    table_a: CalTable
    a_sub: PairSlice
    path: list[tuple[float, float]]


def _prepare(fm: FittedMethod, a_slice: PairSlice) -> _Prepared:
    table_a, a_sub = _table(fm, a_slice)
    t1_only_never, t2_only_never = fm.grids_fixed()
    s_a, l_a = table_a.s_conf, table_a.l_conf
    g1 = [NEVER] if t1_only_never else make_grid(s_a)
    g2 = [NEVER] if t2_only_never else make_grid(l_a)
    path = make_path(g1, g2, table_a, min_coverage=MIN_COVERAGE_A)
    return _Prepared(fm, table_a, a_sub, path)


def _pick_on_a(prep: _Prepared, result: LTTResult) -> tuple[PointResult, float, float] | None:
    best: tuple[tuple[float, float], PointResult, float, float] | None = None
    for pt in result.valid:
        out = evaluate_policy(prep.table_a, pt.t1, pt.t2)
        n_ans = int(out.answered.sum())
        if n_ans == 0:
            continue
        cost = float(policy_item_costs(prep.fm, prep.a_sub, out.by_s).sum())
        cpa = cost / n_ans
        key = (cpa, -n_ans / prep.table_a.n)
        if best is None or key < best[0]:
            best = (key, pt, n_ans / prep.table_a.n, cpa)
    return None if best is None else (best[1], best[2], best[3])


def _artifact(
    fm: FittedMethod,
    pt: PointResult,
    result: LTTResult,
    pair_slice: PairSlice,
    data: ExperimentData,
    pair: PairSpec,
    table_b: CalTable,
    cfg: ExperimentConfig,
) -> dict[str, Any] | None:
    if not fm.defn.deployable and fm.defn.kind in ("random", "query"):
        return None
    sig_s = "agreement" if fm.defn.kind == "sc" else fm.signal
    if sig_s not in SCORE_NAMES or fm.signal not in SCORE_NAMES:
        return None
    samp: SamplingParams = cfg.sampling()
    t_s = TierCalibration(
        data.cells[pair.s_cell].backend,
        samp,
        sig_s,
        None,
        n_samples=cfg.sc_k if fm.defn.kind == "sc" else 0,
        consistency_sampling=cfg.sc_sampling() if fm.defn.kind == "sc" else None,
    )
    t_l = TierCalibration(data.cells[pair.l_cell].backend, samp, fm.signal, None)
    art = build_artifact(
        pt,
        result,
        (t_s, t_l),
        cal_hash=table_b.digest(),
        grader_id=f"{EM_ID}+{MMLU_ID}+{data.judge_state}",
        created=str(data.manifest.get("created", "unknown")),
    )
    return json.loads(dumps_artifact(art))  # type: ignore[no-any-return]


def certify_split(
    store: SplitStore,
    k: int,
    data: ExperimentData,
    plans: Sequence[tuple[PairSpec, Sequence[MethodDef]]],
    cfg: ExperimentConfig,
    persist_dir: Path | None = None,
) -> SplitCertification:
    """Certify every planned method for re-split ``k``; reads B once, through the ledger."""
    main_pair = plans[0][0]
    a_ids = store.a_ids(k)
    a_main = data.slice_pair(main_pair, a_ids)
    primary = choose_primary(a_main.subset(a_main.both_graded()))

    prepared: dict[tuple[str, str], _Prepared] = {}
    a_slices: dict[str, PairSlice] = {}
    for pair, methods in plans:
        a_slices[pair.name] = data.slice_pair(pair, a_ids)
        for m in methods:
            fm = fit_method(m, primary, a_slices[pair.name], seed=k)
            prepared[(pair.name, m.name)] = _prepare(fm, a_slices[pair.name])

    b_ids = store.open_b(  # the one and only B read
        k, capability=CERTIFICATION_CAPABILITY, config_hash=cfg.config_hash()
    )
    certs: dict[str, dict[str, MethodCert]] = {}
    for pair, methods in plans:
        b_slice = data.slice_pair(pair, b_ids)
        certs[pair.name] = {}
        for m in methods:
            prep = prepared[(pair.name, m.name)]
            fm = prep.fm
            c = MethodCert(m.name, pair.name, k, fm.signal, "no_path", n_grid=len(prep.path))
            table_b, b_sub = _table(fm, b_slice)
            c.n_B = table_b.n if b_sub.n else 0
            if not prep.path or b_sub.n == 0:
                c.note = "no path point with A-coverage >= 5%" if not prep.path else "empty B"
                certs[pair.name][m.name] = c
                continue
            res = fixed_sequence_ltt(
                table_b, prep.path, alpha=cfg.alpha, delta=cfg.delta, definition="selective"
            )
            c.n_valid = len(res.valid)
            c.feasibility_floor = res.feasibility_floor
            c.outcome = "feasible" if res.outcome is FeasibilityOutcome.FEASIBLE else "empty"
            if res.valid:
                pick = _pick_on_a(prep, res)
                if pick is None:
                    pt, a_cov, a_cpa = res.valid[0], float("nan"), float("nan")
                    c.note = "no valid point answers on A; took the first certified point"
                else:
                    pt, a_cov, a_cpa = pick
                c.t1, c.t2 = _thr(pt.t1), _thr(pt.t2)
                c.n_answered_B, c.k_wrong_B = pt.n_answered, pt.k_wrong
                c.upper_bound_B = pt.upper_bound
                c.a_coverage, c.a_cost_per_answered = a_cov, a_cpa
                c.artifact = _artifact(fm, pt, res, b_slice, data, pair, table_b, cfg)
            certs[pair.name][m.name] = c
    bundle = compute_bundle_hash(k, primary, certs, store.index_hash)
    store.record_certification(
        k, bundle, capability=CERTIFICATION_CAPABILITY, config_hash=cfg.config_hash()
    )
    sc = SplitCertification(k, primary, certs, bundle)
    if persist_dir is not None:
        atomic_write_text(
            persist_dir / f"split_{k:03d}.json", json.dumps(sc.to_dict(), sort_keys=True)
        )
    return sc


def _core(c: MethodCert) -> dict[str, Any]:
    d = c.to_dict()
    d.pop("artifact", None)
    return d


def compute_bundle_hash(
    k: int, primary: str, certs: dict[str, dict[str, MethodCert]], index_hash: str
) -> str:
    """Hash over thresholds, counts and the embedded artifacts of every method (F-04)."""
    body = {p: {m: _core(c) for m, c in ms.items()} for p, ms in certs.items()}
    arts = {
        p: {m: None if c.artifact is None else sha256_obj(c.artifact) for m, c in ms.items()}
        for p, ms in certs.items()
    }
    return sha256_obj(
        {"k": k, "primary": primary, "certs": body, "artifacts": arts, "index": index_hash}
    )


def load_certification(persist_dir: Path, k: int) -> SplitCertification | None:
    p = persist_dir / f"split_{k:03d}.json"
    if not p.exists():
        return None
    return SplitCertification.from_dict(json.loads(p.read_text(encoding="utf-8")))


def certify_or_load(
    store: SplitStore,
    k: int,
    data: ExperimentData,
    plans: Sequence[tuple[PairSpec, Sequence[MethodDef]]],
    cfg: ExperimentConfig,
    persist_dir: Path,
) -> SplitCertification:
    """Reuse a stored certification; otherwise certify (reads B once). Never re-reads B."""
    existing = load_certification(persist_dir, k)
    if existing is not None:
        if existing.bundle_hash not in store.ledger.certification_hashes(store.index_hash, k):
            raise RuntimeError(f"stored certification for split {k} is not in the ledger")
        recomputed = compute_bundle_hash(
            existing.k, existing.primary_signal, existing.certs, store.index_hash
        )
        if existing.k != k or recomputed != existing.bundle_hash:
            raise RuntimeError(
                f"stored certification for split {k} was modified: its content does not match "
                "its bundle hash"
            )
        return existing
    return certify_split(store, k, data, plans, cfg, persist_dir)

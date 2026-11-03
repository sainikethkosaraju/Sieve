"""Compute every reported number from cached generations (CPU only, deterministic).

Order of operations per re-split ``k``: certification (reads B once, in ``certify``) ->
test on C (unlocked by the certification hash) -> shift sets and quantization transfer
with the same certified thresholds. Statistics on re-split 0 (the canonical split) use the
paired bootstrap; validity (H1) uses all re-splits.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sieve.calibrate import (
    NEVER,
    CalTable,
    auroc,
    evaluate_policy,
    psi,
    score_shift_pvalue,
)
from sieve.eval import stats as st
from sieve.eval._util import atomic_write_text, to_jsonable
from sieve.eval.audit import cp_interval
from sieve.eval.certify import MethodCert, SplitCertification, certify_or_load, make_grid
from sieve.eval.config import ExperimentConfig, PairSpec
from sieve.eval.data import ExperimentData, PairSlice, load_experiment
from sieve.eval.datasets import Item
from sieve.eval.evaluate import (
    PolicyEval,
    eval_cert,
    eval_policy,
    oracle_abstainer,
    oracle_router,
)
from sieve.eval.grading import Judge
from sieve.eval.methods import (
    FittedMethod,
    MethodDef,
    default_methods,
    fit_method,
)
from sieve.eval.sections import (
    accuracy_section,
    calibration_section,
    comparison_section,
    curve_section,
    energy_section,
    hard_set_section,
    meta_section,
    nondeterminism_section,
    quant_section,
    sc_section,
    shift_section,
    signals_section,
    stratum_section,
)
from sieve.eval.splits import SplitStore

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]
Progress = Callable[[str], None]


def iv(i: st.Interval) -> dict[str, float]:
    return i.to_dict()


def _nan_to_neginf(a: FloatArr) -> FloatArr:
    return np.where(np.isfinite(a), a, -np.inf)


def _boot(cfg: ExperimentConfig, salt: int) -> dict[str, int]:
    return {"n_boot": cfg.n_boot, "seed": cfg.boot_seed + salt}


# --- aggregation across re-splits -----------------------------------------------------


def aggregate_splits(per_k: dict[int, dict[str, Any] | None], alpha: float) -> dict[str, Any]:
    """Violation frequency etc. over re-splits (certified splits with >= 1 answered item)."""
    feasible = {k: s for k, s in per_k.items() if s is not None and s["n_answered"] > 0}
    n_total = len(per_k)
    viol = [k for k, s in feasible.items() if s["violates_alpha"]]
    lo, hi = cp_interval(len(viol), len(feasible)) if feasible else (float("nan"), float("nan"))

    def mean(key: str) -> float | None:
        vals = [s[key] for s in feasible.values() if s.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "n_splits": n_total,
        "n_certified": len(feasible),
        "n_empty": n_total - len(feasible),
        "empty_fraction": (n_total - len(feasible)) / n_total if n_total else None,
        "n_violations": len(viol),
        "violation_freq": len(viol) / len(feasible) if feasible else None,
        "violation_ci95": [lo, hi],
        "mean_test_risk": mean("risk"),
        "max_test_risk": max((s["risk"] for s in feasible.values()), default=None),
        "mean_coverage": mean("coverage"),
        "mean_cost_per_correct": mean("cost_per_correct"),
        "mean_escalation_rate": mean("escalation_rate"),
        "alpha": alpha,
    }


# --- matched baselines ------------------------------------------------------------------


def _a_operating_point(
    fm: FittedMethod, a_slice: PairSlice, cert: MethodCert
) -> tuple[float, float] | None:
    ev = eval_cert(fm, a_slice, cert, alpha=0.0)
    if ev is None or ev.n == 0:
        return None
    return float(ev.answered.mean()), float(1.0 - ev.by_s.mean())


def match_policy(
    fm: FittedMethod, a_slice: PairSlice, target_cov: float, target_esc: float
) -> tuple[float | None, float | None]:
    """Uncertified (t1, t2) of ``fm`` whose A-coverage and A-escalation rate best match."""
    sub = a_slice.subset(fm.eligible(a_slice))
    s, l_ = fm.scores(sub)
    table = CalTable(s, l_, sub.s.wrong, sub.l.wrong)
    best: tuple[float, float, float] | None = None
    for t1 in make_grid(s):
        for t2 in make_grid(l_):
            out = evaluate_policy(table, t1, t2)
            d = abs(out.answered.mean() - target_cov) + abs(1.0 - out.by_s.mean() - target_esc)
            if best is None or d < best[0] - 1e-12:
                best = (float(d), t1, t2)
    assert best is not None
    return (None if best[1] == NEVER else best[1], None if best[2] == NEVER else best[2])


# --- paired comparison helpers -----------------------------------------------------------


def _acc_stat(ev: PolicyEval) -> st.BatchStat:
    return st.batch_ratio(ev.correct_answered, ev.answered)


def _cpc_stat(ev: PolicyEval) -> st.BatchStat:
    return st.batch_ratio(ev.cost, ev.correct_answered)


def _risk_stat(ev: PolicyEval) -> st.BatchStat:
    return st.batch_ratio(ev.answered & ev.wrong, ev.answered)


def paired_compare(
    a: PolicyEval, b: PolicyEval, cfg: ExperimentConfig, salt: int
) -> dict[str, Any]:
    """Item-paired differences a - b: accuracy on answered, cost per correct, coverage."""
    if a.ids != b.ids:
        raise ValueError("paired comparison needs identical item order")
    kw = _boot(cfg, salt)
    mc = st.mcnemar(a.correct_answered, b.correct_answered)
    return {
        "a": a.label,
        "b": b.label,
        "n": a.n,
        "acc_on_answered_diff": iv(st.paired_difference(_acc_stat(a), _acc_stat(b), a.n, **kw)),
        "cost_per_correct_diff": iv(st.paired_difference(_cpc_stat(a), _cpc_stat(b), a.n, **kw)),
        "coverage_diff": iv(
            st.paired_difference(st.batch_mean(a.answered), st.batch_mean(b.answered), a.n, **kw)
        ),
        "correct_answer_rate_diff": iv(
            st.paired_difference(
                st.batch_mean(a.correct_answered), st.batch_mean(b.correct_answered), a.n, **kw
            )
        ),
        "mcnemar_correct_answered": {"n10": mc.n10, "n01": mc.n01, "p_exact": mc.p_exact},
    }


# --- signal quality (H2 / H4 / H7) --------------------------------------------------------


def signal_auroc_table(
    sl: PairSlice, cfg: ExperimentConfig, signals: tuple[str, ...], salt: int
) -> dict[str, Any]:
    """AUROC (correct vs wrong) with bootstrap CI, per tier and signal; NaN scores rank last."""
    out: dict[str, Any] = {}
    for tier, v in (("S", sl.s), ("L", sl.l)):
        m = v.graded
        y = v.correct[m]
        out[tier] = {}
        if m.sum() < 20 or y.all() or (~y).all():
            continue
        for s in signals:
            sc = _nan_to_neginf(v.signals[s][m])
            if np.all(np.isneginf(sc)):
                continue
            out[tier][s] = iv(
                st.bootstrap(
                    st.batch_auroc(sc, y), int(m.sum()), point=auroc(sc, y), **_boot(cfg, salt)
                )
            )
    return out


def auroc_delta(
    sl: PairSlice,
    tier: str,
    sig: str,
    ref: str,
    cfg: ExperimentConfig,
    salt: int,
    mask: BoolArr | None = None,
) -> dict[str, float] | None:
    v = sl.s if tier == "S" else sl.l
    m = v.graded if mask is None else (v.graded & mask)
    y = v.correct[m]
    if m.sum() < 20 or y.all() or (~y).all():
        return None
    a = _nan_to_neginf(v.signals[sig][m])
    b = _nan_to_neginf(v.signals[ref][m])
    if np.all(np.isneginf(a)) or np.all(np.isneginf(b)):
        return None
    return iv(
        st.paired_difference(
            st.batch_auroc(a, y), st.batch_auroc(b, y), int(m.sum()), **_boot(cfg, salt)
        )
    )


# --- drift alarm ---------------------------------------------------------------------------


def drift_windows(
    reference: FloatArr,
    scores: FloatArr,
    wrong: BoolArr | None,
    graded: BoolArr | None,
    *,
    window: int,
    alpha: float,
    seed: int,
    p_alarm: float = 0.05,
) -> dict[str, Any]:
    """Label-free KS alarm on non-overlapping windows; compare with realised window risk."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(scores.size)
    n_win = scores.size // window
    ref = _nan_to_neginf(reference)
    rows: list[dict[str, Any]] = []
    for w in range(n_win):
        sel = order[w * window : (w + 1) * window]
        p = score_shift_pvalue(ref, _nan_to_neginf(scores[sel]))
        row: dict[str, Any] = {"alarm": bool(p < p_alarm), "ks_p": float(p)}
        row["psi"] = (
            psi(ref[np.isfinite(ref)], _nan_to_neginf(scores[sel])[np.isfinite(scores[sel])])
            if np.isfinite(scores[sel]).sum() > 10
            else None
        )
        if wrong is not None and graded is not None and graded[sel].all():
            row["window_error_rate"] = float(wrong[sel].mean())
            row["violation"] = bool(wrong[sel].mean() > alpha)
        rows.append(row)
    alarms = [r["alarm"] for r in rows]
    viol = [r for r in rows if r.get("violation")]
    return {
        "n_windows": n_win,
        "window": window,
        "alarm_rate": float(np.mean(alarms)) if alarms else None,
        "n_violating_windows": len(viol),
        "alarm_rate_on_violating": (float(np.mean([r["alarm"] for r in viol])) if viol else None),
    }


# --- main entry -------------------------------------------------------------------------


@dataclass
class AnalysisPlan:
    cfg: ExperimentConfig
    main: PairSpec
    quant_ref: PairSpec | None
    quant_pairs: list[PairSpec]
    methods: list[MethodDef]


def make_plan(cfg: ExperimentConfig, data: ExperimentData) -> AnalysisPlan:
    main = cfg.main_pair()
    s_cell = data.cells.get(main.s_cell)
    include_sc = bool(s_cell is not None and s_cell.has_samples.any())
    refs = [p for p in cfg.pairs if p.role == "quant_ref"]
    quants = [p for p in cfg.pairs if p.role == "quant"]
    avail = set(data.cells)
    quants = [p for p in quants if p.s_cell in avail and p.l_cell in avail]
    ref = refs[0] if refs and refs[0].s_cell in avail else None
    return AnalysisPlan(cfg, main, ref, quants, default_methods(include_sc))


def run_analysis(
    workdir: Path,
    cfg: ExperimentConfig,
    items: dict[str, Item],
    judge: Judge | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    data = load_experiment(workdir, cfg, items, judge)
    store = SplitStore.load(workdir / "splits")
    plan = make_plan(cfg, data)
    cert_dir = workdir / "certs"
    alpha = cfg.alpha
    main = plan.main
    prim_methods = [m for m in plan.methods if m.name == "gate:primary"]
    plans: list[tuple[PairSpec, list[MethodDef]]] = [(main, plan.methods)]
    extra_pairs = ([plan.quant_ref] if plan.quant_ref else []) + plan.quant_pairs
    seen = {main.name}
    for p in extra_pairs:
        if p.name not in seen:
            plans.append((p, prim_methods))
            seen.add(p.name)

    per_method: dict[str, dict[int, dict[str, Any] | None]] = {m.name: {} for m in plan.methods}
    certs_by_k: dict[int, SplitCertification] = {}
    ev0: dict[str, PolicyEval] = {}
    fm0: dict[str, FittedMethod] = {}
    c0_slice: PairSlice | None = None
    a0_slice: PairSlice | None = None
    quant_split: dict[str, dict[str, dict[int, dict[str, Any] | None]]] = {}
    shift_split: dict[str, dict[int, dict[str, Any] | None]] = {}
    shift_ids: dict[str, list[str]] = {}
    for p in [main]:
        for ds in sorted(
            {it.dataset for iid, it in items.items() if iid in data.cells[p.s_cell].index}
        ):
            if ds != cfg.primary_dataset:
                shift_ids[ds] = sorted(iid for iid in data.ids_with(p) if items[iid].dataset == ds)
    for k in range(store.n_splits):
        sc = certify_or_load(store, k, data, plans, cfg, cert_dir)
        certs_by_k[k] = sc
        c_ids = store.open_c(
            k,
            cert_hash=sc.bundle_hash,
            config_hash=cfg.config_hash(),
            reopen=store.ledger.c_opened(store.index_hash, k, sc.bundle_hash),
        )
        a_ids = store.a_ids(k)
        a_main = data.slice_pair(main, a_ids)
        c_main = data.slice_pair(main, c_ids)
        fms = {m.name: fit_method(m, sc.primary_signal, a_main, seed=k) for m in plan.methods}
        for m in plan.methods:
            cert = sc.certs[main.name][m.name]
            ev = eval_cert(fms[m.name], c_main, cert, alpha=alpha)
            per_method[m.name][k] = None if ev is None else ev.summary()
            if k == 0 and ev is not None:
                ev0[m.name] = ev
        if k == 0:
            fm0, c0_slice, a0_slice = fms, c_main, a_main
        # --- shift sets with the certified primary-gate thresholds
        cert_p = sc.certs[main.name]["gate:primary"]
        for ds, ids in shift_ids.items():
            sl = data.slice_pair(main, ids)
            ev = eval_cert(fms["gate:primary"], sl, cert_p, alpha=alpha)
            shift_split.setdefault(ds, {})[k] = None if ev is None else ev.summary()
            if k == 0 and ev is not None:
                ev0[f"shift:{ds}"] = ev
        # --- quantization: transfer f16 thresholds vs recalibrated
        if plan.quant_ref is not None:
            ref_cert = sc.certs[plan.quant_ref.name]["gate:primary"]
            for qp in plan.quant_pairs:
                a_q = data.slice_pair(qp, a_ids)
                c_q = data.slice_pair(qp, c_ids)
                fm_q = fit_method(prim_methods[0], sc.primary_signal, a_q, seed=k)
                own = sc.certs[qp.name]["gate:primary"]
                rec = eval_cert(fm_q, c_q, own, alpha=alpha)
                tr = (
                    eval_policy(
                        fm_q,
                        c_q,
                        ref_cert.t1,
                        ref_cert.t2,
                        alpha=alpha,
                        label=f"{qp.name}:transferred",
                        certified=False,
                    )
                    if ref_cert.feasible
                    else None
                )
                d = quant_split.setdefault(qp.name, {"transferred": {}, "recalibrated": {}})
                d["transferred"][k] = None if tr is None else tr.summary()
                d["recalibrated"][k] = None if rec is None else rec.summary()
                if k == 0:
                    if tr is not None:
                        ev0[f"quant:{qp.name}:transferred"] = tr
                    if rec is not None:
                        ev0[f"quant:{qp.name}:recalibrated"] = rec
        if (k + 1) % 10 == 0:
            say(f"analysed {k + 1}/{store.n_splits} re-splits")

    assert c0_slice is not None
    assert a0_slice is not None
    k0 = certs_by_k[0]
    results: dict[str, Any] = {
        "meta": meta_section(cfg, data, store, plan),
        "accuracy": accuracy_section(data, items),
        "methods": {
            m: {"across_splits": aggregate_splits(per_method[m], alpha), "split0": per_method[m][0]}
            for m in per_method
        },
        "certs_split0": {
            m: {k_: v for k_, v in c.to_dict().items() if k_ != "artifact"}
            for m, c in k0.certs[main.name].items()
        },
        "primary_signal_by_split": {str(k): c.primary_signal for k, c in certs_by_k.items()},
    }
    results["h1"] = {
        **results["methods"]["gate:primary"]["across_splits"],
        "pass_max": cfg.h1_pass_max,
        "fail_min": cfg.h1_fail_min,
    }
    results["signals"] = signals_section(c0_slice, a0_slice, k0, cfg)
    results["comparisons"] = comparison_section(ev0, fm0, a0_slice, c0_slice, k0, main, cfg, plan)
    results["oracles"] = {
        "oracle_router": oracle_router(c0_slice, alpha=alpha).summary(),
        "oracle_abstainer": oracle_abstainer(c0_slice, alpha=alpha).summary(),
    }
    results["self_consistency"] = sc_section(c0_slice, k0, cfg, data, main)
    results["shift"] = shift_section(
        data, cfg, main, ev0, shift_split, shift_ids, a0_slice, fm0, k0, alpha
    )
    results["quant"] = quant_section(data, cfg, plan, ev0, quant_split, certs_by_k, store)
    results["hard_set"] = hard_set_section(data, cfg, main, k0, fm0, a0_slice)
    results["calibration"] = calibration_section(a0_slice, c0_slice, k0)
    results["curves"] = curve_section(c0_slice, a0_slice, fm0, k0, ev0, cfg)
    results["energy"] = energy_section(data, cfg, main, ev0, fm0, c0_slice, k0)
    results["nondeterminism"] = nondeterminism_section(workdir, cfg)
    results["per_stratum"] = stratum_section(per_method, ev0)
    results = to_jsonable(results)
    atomic_write_text(
        workdir / "analysis" / "results.json",
        json.dumps(results, sort_keys=True, indent=1, allow_nan=False),
    )
    return results

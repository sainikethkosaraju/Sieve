"""Sections of the analysis result (kept apart from the split loop in ``analysis``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sieve.calibrate import (
    NEVER,
    CalTable,
    PlattScaler,
    brier,
    ece_equal_mass,
    evaluate_policy,
    risk_coverage_curve,
)
from sieve.eval import stats as st
from sieve.eval.cache import CellShard
from sieve.eval.certify import SplitCertification, make_grid
from sieve.eval.config import ExperimentConfig, PairSpec
from sieve.eval.data import ExperimentData, PairSlice
from sieve.eval.datasets import Item
from sieve.eval.evaluate import PolicyEval, eval_cert, eval_policy
from sieve.eval.grading import simpleqa_f1
from sieve.eval.methods import FittedMethod, policy_item_costs
from sieve.eval.runner import cell_id
from sieve.eval.splits import SplitStore
from sieve.types import SCORE_NAMES

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]
SYNTHETIC_LABEL = (
    "SYNTHETIC DATA: produced by the offline fake backend; not a measurement of any LLM"
)


def _iv(i: st.Interval) -> dict[str, float]:
    return i.to_dict()


def _boot(cfg: ExperimentConfig, salt: int) -> dict[str, int]:
    return {"n_boot": cfg.n_boot, "seed": cfg.boot_seed + salt}


def _neginf(a: FloatArr) -> FloatArr:
    """Floor for items without a score (defective generations), for the DISPLAY curves only.

    ``risk_coverage_curve`` rejects non-finite confidences, so use a finite value strictly below
    every real score: such items rank last and are never reached before real scores. Certification
    keeps -inf (see ``analysis._nan_to_neginf``) so a defective item can never be answered.
    """
    fin = np.isfinite(a)
    if bool(fin.all()):
        return a
    floor = (float(np.min(a[fin])) if bool(fin.any()) else 0.0) - 1.0
    return np.where(fin, a, floor)


def meta_section(
    cfg: ExperimentConfig, data: ExperimentData, store: SplitStore, plan: Any
) -> dict[str, Any]:
    man = data.manifest
    return {
        "name": cfg.name,
        "synthetic": cfg.synthetic,
        "synthetic_label": SYNTHETIC_LABEL if cfg.synthetic else None,
        "config_hash": cfg.config_hash(),
        "manifest_config_hash": man.get("config_hash"),
        "prereg_frozen_sha256": man.get("prereg_frozen_sha256"),
        "commit": man.get("commit"),
        "hardware": man.get("hardware"),
        "created": man.get("created"),
        "judge": data.judge_state,
        "ungraded_datasets": data.ungraded_datasets,
        "splits_index_hash": store.index_hash,
        "dataset_hash": store.dataset_hash,
        "n_splits": store.n_splits,
        "split_sizes": list(store.sizes),
        "alpha": cfg.alpha,
        "delta": cfg.delta,
        "n_boot": cfg.n_boot,
        "timing_source": cfg.timing_source,
        "cost_unit": cfg.cost_unit,
        "main_pair": cfg.main_pair().name,
        "template_hash": man.get("template_hash"),
        "sc_k": cfg.sc_k,
    }


def accuracy_section(data: ExperimentData, items: dict[str, Item]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, c in sorted(data.cells.items()):
        per: dict[str, Any] = {}
        for ds in sorted({items[i].dataset for i in c.ids}):
            sel = np.array([items[i].dataset == ds for i in c.ids])
            g = sel & c.graded
            row: dict[str, Any] = {
                "n": int(sel.sum()),
                "n_graded": int(g.sum()),
                "accuracy": float(c.correct[g].mean()) if g.any() else None,
                "grade_state": "graded" if g.any() else f"ungraded ({data.judge_state})",
                "mean_time_s": float(c.time_s[sel].mean()),
                "wasted_time_share": float(c.wasted_s[sel].sum() / max(c.time_s[sel].sum(), 1e-12)),
                "fail_closed_rate": float(np.isnan(c.signals["max_prob"][sel]).mean()),
            }
            if ds == "simpleqa_verified" and g.any():
                row["simpleqa"] = simpleqa_f1([c.labels[i] for i in np.flatnonzero(sel)])
            per[ds] = row
        out[name] = {
            "tier": c.tier,
            "backend": {
                "model_id": c.backend.model_id,
                "quant": c.backend.quant,
                "kv_quant": c.backend.kv_quant,
                "build": c.backend.build,
            },
            "failed_attempts": c.attempts_failed,
            "datasets": per,
        }
    return out


def signals_section(
    c0: PairSlice, a0: PairSlice, k0: SplitCertification, cfg: ExperimentConfig
) -> dict[str, Any]:
    from sieve.eval.analysis import auroc_delta, signal_auroc_table

    token_signals = tuple(s for s in SCORE_NAMES if s not in ("agreement", "neg_cluster_entropy"))
    sec: dict[str, Any] = {
        "primary_signal": k0.primary_signal,
        "auroc": signal_auroc_table(c0, cfg, token_signals, 11),
        "h2_delta_vs_max_prob": {},
    }
    for sig in (
        "mean_logprob",
        "first_margin",
        "mean_margin",
        "min_logprob",
        "neg_mean_entropy",
        "neg_first_entropy",
    ):
        d = auroc_delta(c0, "S", sig, "max_prob", cfg, 12)
        if d is not None:
            sec["h2_delta_vs_max_prob"][sig] = d
    # selective metrics for the primary signal on S and L
    return sec


def comparison_section(
    ev0: dict[str, PolicyEval],
    fm0: dict[str, FittedMethod],
    a0: PairSlice,
    c0: PairSlice,
    k0: SplitCertification,
    main: PairSpec,
    cfg: ExperimentConfig,
    plan: Any,
) -> dict[str, Any]:
    from sieve.eval.analysis import _a_operating_point, match_policy, paired_compare

    gate = ev0.get("gate:primary")
    sec: dict[str, Any] = {"gate_available": gate is not None}
    if gate is None:
        sec["reason"] = "primary gate has an empty certified set on re-split 0"
        return sec
    for other in (
        "L_only",
        "S_only",
        *[n for n in ev0 if n.startswith("gate:") and n != "gate:primary"],
    ):
        if other in ev0 and ev0[other].ids == gate.ids:
            sec[f"vs_{other}"] = paired_compare(gate, ev0[other], cfg, 20)
    cert = k0.certs[main.name]["gate:primary"]
    target = _a_operating_point(fm0["gate:primary"], a0, cert)
    sec["matched"] = {}
    if target is not None:
        for name in ("random_router", "query_router"):
            fm = fm0[name]
            t1, t2 = match_policy(fm, a0, target[0], target[1])
            ev = eval_policy(
                fm,
                c0,
                t1,
                t2,
                alpha=cfg.alpha,
                label=f"{name}@matched",
                certified=False,
                note="thresholds matched to the gate's A-coverage and A-escalation rate",
            )
            if ev.ids == gate.ids:
                sec["matched"][name] = {
                    "thresholds": [t1, t2],
                    "summary": ev.summary(),
                    "vs_gate": paired_compare(gate, ev, cfg, 21 if name == "random_router" else 22),
                }
            ev0[f"{name}@matched"] = ev
    for name in ("random_router", "query_router"):
        if name in ev0 and ev0[name].ids == gate.ids:
            sec[f"vs_{name}_certified"] = paired_compare(gate, ev0[name], cfg, 23)
    sec["gate_a_operating_point"] = (
        {"coverage": target[0], "escalation_rate": target[1]} if target else None
    )
    l_only = ev0.get("L_only")
    if l_only is not None and l_only.ids == gate.ids:
        diff = sec["vs_L_only"]["acc_on_answered_diff"]
        rate = sec["vs_L_only"]["correct_answer_rate_diff"]
        cost = sec["vs_L_only"]["cost_per_correct_diff"]
        sec["noninferiority_vs_L_only"] = {
            "margin": cfg.noninferiority_margin,
            "acc_on_answered_diff": diff,
            "acc_on_answered_non_inferior": bool(diff["lo"] >= -cfg.noninferiority_margin),
            # F-07: coverage-aware: answering less cannot buy non-inferiority
            "correct_answer_rate_diff": rate,
            "non_inferior": bool(rate["lo"] >= -cfg.noninferiority_margin),
            "cost_per_correct_diff": cost,
            "cheaper_per_correct": bool(cost["hi"] < 0),
            "cost_unit": c0.cost_unit,
            "gate_cost_per_correct": gate.summary()["cost_per_correct"],
            "L_only_cost_per_correct": l_only.summary()["cost_per_correct"],
        }
    return sec


def sc_section(
    c0: PairSlice,
    k0: SplitCertification,
    cfg: ExperimentConfig,
    data: ExperimentData,
    main: PairSpec,
) -> dict[str, Any]:
    from sieve.eval.analysis import auroc_delta

    mask = c0.s.has_samples
    n = int(mask.sum())
    if n == 0:
        return {"state": "not_run"}
    sec: dict[str, Any] = {"state": "run", "n_items_in_C0": n, "k": cfg.sc_k}
    best = k0.primary_signal
    d = auroc_delta(c0, "S", "agreement", best, cfg, 30, mask=mask)
    sec["delta_auroc_vs_best_token_signal"] = d
    sec["best_token_signal"] = best
    s_time = float(c0.s.time_s[mask].mean())
    l_time = float(c0.l.time_s[mask].mean())
    sample_time = float(c0.s.sample_time_s[mask].mean())
    sec["mean_time_s"] = {"S_greedy": s_time, "S_k_samples": sample_time, "L": l_time}
    sec["k_times_cost_S_below_cost_L"] = bool(sample_time < l_time)
    return sec


def _ci_risk(ev: PolicyEval, cfg: ExperimentConfig, salt: int) -> dict[str, float] | None:
    if ev.n == 0 or int(ev.answered.sum()) == 0:
        return None
    return _iv(
        st.bootstrap(st.batch_ratio(ev.answered & ev.wrong, ev.answered), ev.n, **_boot(cfg, salt))
    )


def shift_section(
    data: ExperimentData,
    cfg: ExperimentConfig,
    main: PairSpec,
    ev0: dict[str, PolicyEval],
    shift_split: dict[str, dict[int, dict[str, Any] | None]],
    shift_ids: dict[str, list[str]],
    a0: PairSlice,
    fm0: dict[str, FittedMethod],
    k0: SplitCertification,
    alpha: float,
) -> dict[str, Any]:
    from sieve.eval.analysis import aggregate_splits, drift_windows

    sec: dict[str, Any] = {}
    sig = k0.primary_signal
    ref = a0.s.signals[sig]
    for j, (ds, ids) in enumerate(sorted(shift_ids.items())):
        if ds == "mmlu_redux":
            continue  # reported in hard_set
        sl = data.slice_pair(main, ids)
        ev = ev0.get(f"shift:{ds}")
        row: dict[str, Any] = {"n_items": len(ids), "judge": data.judge_state}
        graded = bool(sl.both_graded().any())
        row["graded"] = graded
        per_k = shift_split.get(ds, {})
        row["across_splits"] = aggregate_splits(per_k, alpha) if graded else {"state": "ungraded"}
        row["split0"] = ev.summary() if ev is not None and ev.n else None
        row["risk_ci95_split0"] = _ci_risk(ev, cfg, 40 + j) if ev is not None else None
        row["drift"] = drift_windows(
            ref,
            sl.s.signals[sig],
            sl.s.wrong if graded else None,
            sl.s.graded if graded else None,
            window=cfg.window_size,
            alpha=alpha,
            seed=cfg.boot_seed + 50 + j,
        )
        sec[ds] = row
    return sec


def hard_set_section(
    data: ExperimentData,
    cfg: ExperimentConfig,
    main: PairSpec,
    k0: SplitCertification,
    fm0: dict[str, FittedMethod],
    a0: PairSlice,
) -> dict[str, Any]:
    ids = sorted(i for i in data.ids_with(main) if data.items[i].dataset == "mmlu_redux")
    if not ids:
        return {"state": "not_run"}
    sl = data.slice_pair(main, ids)
    from sieve.eval.analysis import signal_auroc_table

    sig = k0.primary_signal
    tab = signal_auroc_table(sl, cfg, (sig, "max_prob"), 60)
    g = sl.s.graded
    return {
        "state": "run",
        "n": len(ids),
        "primary_signal": sig,
        "auroc": tab,
        "accuracy_S": float(sl.s.correct[g].mean()) if g.any() else None,
        "accuracy_L": float(sl.l.correct[sl.l.graded].mean()) if sl.l.graded.any() else None,
    }


def _bins(prob: FloatArr, y: BoolArr, n_bins: int = 15) -> list[dict[str, float]]:
    order = np.argsort(prob, kind="stable")
    out = []
    for idx in np.array_split(order, min(n_bins, prob.size)):
        out.append(
            {"mean_p": float(prob[idx].mean()), "mean_y": float(y[idx].mean()), "n": int(idx.size)}
        )
    return out


def calibration_section(a0: PairSlice, c0: PairSlice, k0: SplitCertification) -> dict[str, Any]:
    sig = k0.primary_signal
    out: dict[str, Any] = {"signal": sig, "method": "Platt scaling fitted on split A"}
    for tier, va, vc in (("S", a0.s, c0.s), ("L", a0.l, c0.l)):
        ma = va.graded & np.isfinite(va.signals[sig])
        mc = vc.graded & np.isfinite(vc.signals[sig])
        if ma.sum() < 20 or mc.sum() < 20 or va.correct[ma].all() or (~va.correct[ma]).all():
            continue
        cal = PlattScaler().fit(va.signals[sig][ma], va.correct[ma])
        p = np.clip(cal.predict(vc.signals[sig][mc]), 0.0, 1.0)
        y = vc.correct[mc]
        out[tier] = {
            "ece": ece_equal_mass(p, y),
            "brier": brier(p, y),
            "bins": _bins(p, y),
            "n": int(mc.sum()),
        }
    return out


def _thin(x: FloatArr, y: FloatArr, n: int = 120) -> list[list[float]]:
    sel = np.arange(x.size) if x.size <= n else np.unique(np.linspace(0, x.size - 1, n).astype(int))
    return [[float(x[i]), float(y[i])] for i in sel]


def curve_section(
    c0: PairSlice,
    a0: PairSlice,
    fm0: dict[str, FittedMethod],
    k0: SplitCertification,
    ev0: dict[str, PolicyEval],
    cfg: ExperimentConfig,
) -> dict[str, Any]:
    sig = k0.primary_signal
    sec: dict[str, Any] = {"risk_coverage": {}, "pareto": {"sweep": {}, "certified": {}}}
    for tier, v in (("S", c0.s), ("L", c0.l)):
        m = v.graded
        if m.sum() >= 20:
            cov, risk = risk_coverage_curve(_neginf(v.signals[sig][m]), v.wrong[m])
            sec["risk_coverage"][tier] = _thin(cov, risk)
    for name in ("gate:primary", "random_router", "query_router", "S_only", "L_only"):
        fm = fm0.get(name)
        if fm is None:
            continue
        sub = c0.subset(fm.eligible(c0))
        a_sub = a0.subset(fm.eligible(a0))
        if sub.n == 0 or a_sub.n == 0:
            continue
        s_a, l_a = fm.scores(a_sub)
        s_c, l_c = fm.scores(sub)
        table_c = CalTable(s_c, l_c, sub.s.wrong, sub.l.wrong)
        t1s = [NEVER] if name == "L_only" else make_grid(s_a)
        t2s = [NEVER] if name == "S_only" else make_grid(l_a)
        pts = []
        for t1 in t1s:
            for t2 in t2s:
                out = evaluate_policy(table_c, t1, t2)
                n_ans = int(out.answered.sum())
                if n_ans < 0.05 * sub.n:
                    continue
                cost = policy_item_costs(fm, sub, out.by_s)
                pts.append(
                    {
                        "cost_per_item": float(cost.mean()),
                        "acc_on_answered": 1.0 - float(out.wrong.sum()) / n_ans,
                        "coverage": n_ans / sub.n,
                    }
                )
        sec["pareto"]["sweep"][name] = pts
        ev = ev0.get(name)
        if ev is not None and ev.n and ev.answered.any():
            sec["pareto"]["certified"][name] = {
                "cost_per_item": _iv(st.bootstrap(st.batch_mean(ev.cost), ev.n, **_boot(cfg, 70))),
                "acc_on_answered": _iv(
                    st.bootstrap(
                        st.batch_ratio(ev.correct_answered, ev.answered), ev.n, **_boot(cfg, 71)
                    )
                ),
                "coverage": float(ev.answered.mean()),
            }
    return sec


def energy_section(
    data: ExperimentData,
    cfg: ExperimentConfig,
    main: PairSpec,
    ev0: dict[str, PolicyEval],
    fm0: dict[str, FittedMethod],
    c0: PairSlice,
    k0: SplitCertification,
) -> dict[str, Any]:
    sec: dict[str, Any] = {
        "cells": {
            n: {
                "state": e.state,
                "mean_watts_net": e.mean_watts_net,
                "peak_vram_mib": e.peak_vram_mib,
                "n_blocks": e.n_blocks,
                "throttled_blocks": e.throttled_blocks,
                "cv_watts_across_blocks": e.cv_watts,
                "cv_below_10pct": None if e.cv_watts is None else bool(e.cv_watts < 0.10),
            }
            for n, e in sorted(data.energy.items())
        },
        "scope": "GPU energy only (NVML/nvidia-smi, idle-subtracted); CPU/RAM not measured",
        "wall_meter_crosscheck": "not available on Kaggle: treat joules as relative",
    }
    es, el = data.energy.get(main.s_cell), data.energy.get(main.l_cell)
    if es is None or el is None or es.mean_watts_net is None or el.mean_watts_net is None:
        sec["state"] = "unavailable (no GPU power measurement for the main pair)"
        return sec
    sec["state"] = es.state
    jl = data.slice_pair(main, c0.ids, cost_unit="joules")
    rows = {}
    for name in ("gate:primary", "L_only", "S_only"):
        cert = k0.certs[main.name][name]
        fm = fm0.get(name)
        if fm is None:
            continue
        ev = eval_cert(fm, jl, cert, alpha=cfg.alpha)
        if ev is not None:
            s = ev.summary()
            rows[name] = {
                "joules_per_correct": s["cost_per_correct"],
                "joules_per_item": s["cost_per_item"],
            }
    sec["joules_split0"] = rows
    return sec


def nondeterminism_section(workdir: Path, cfg: ExperimentConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in cfg.cells:
        p = workdir / "generations" / f"{cell_id(c, cfg.sampling())}.jsonl"
        if not p.exists():
            continue
        shard = CellShard(p)
        try:
            n = flips = 0
            max_dlp = 0.0
            for key in list(shard.item_keys()):
                if "|repeat|" not in key:
                    continue
                iid = key.split("|")[0]
                rep, base = shard.final(key), shard.final(f"{iid}|primary|0")
                if rep is None or base is None:
                    continue
                n += 1
                flips += int(rep.text != base.text)
                if rep.tokens and base.tokens and len(rep.tokens) == len(base.tokens):
                    max_dlp = max(
                        max_dlp,
                        max(
                            abs(a.logprob - b.logprob)
                            for a, b in zip(rep.tokens, base.tokens, strict=True)
                        ),
                    )
            if n:
                out[c.name] = {
                    "n_repeat_generations": n,
                    "text_flip_rate": flips / n,
                    "max_abs_logprob_diff": max_dlp,
                }
        finally:
            shard.close()
    return out or {"state": "not_run"}


def stratum_section(
    per_method: dict[str, dict[int, dict[str, Any] | None]], ev0: dict[str, PolicyEval]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in ("gate:primary", "L_only", "S_only"):
        pooled: dict[str, list[int]] = {}
        for s in per_method.get(m, {}).values():
            if s is None:
                continue
            for stratum, c in s["per_stratum"].items():
                p = pooled.setdefault(stratum, [0, 0])
                p[0] += c["n_answered"]
                p[1] += c["n_wrong"]
        out[m] = {
            stratum: {"n_answered": p[0], "risk": p[1] / p[0] if p[0] else None, "low_n": p[0] < 30}
            for stratum, p in sorted(pooled.items())
        }
    return out


def _platt_prob(a_v: Any, c_v: Any, sig: str) -> tuple[FloatArr, BoolArr, BoolArr] | None:
    ma = a_v.graded & np.isfinite(a_v.signals[sig])
    if ma.sum() < 20 or a_v.correct[ma].all() or (~a_v.correct[ma]).all():
        return None
    cal = PlattScaler().fit(a_v.signals[sig][ma], a_v.correct[ma])
    s = np.where(np.isfinite(c_v.signals[sig]), c_v.signals[sig], np.nanmin(a_v.signals[sig][ma]))
    return np.clip(cal.predict(s), 0.0, 1.0), c_v.correct, c_v.graded


def _quant_deltas(
    cfg: ExperimentConfig,
    sig: str,
    c_ref: PairSlice,
    c_q: PairSlice,
    pr: tuple[FloatArr, BoolArr, BoolArr] | None,
    pq: tuple[FloatArr, BoolArr, BoolArr] | None,
    j: int,
) -> dict[str, Any]:
    """Paired delta-AUROC / delta-ECE of a quantized config vs the reference, with TOST flags."""
    m = c_ref.s.graded & c_q.s.graded
    if pr is None or pq is None or m.sum() < 30:
        return {
            "delta_auroc": None,
            "delta_ece": None,
            "auroc_equivalent": None,
            "ece_equivalent": None,
        }
    n = int(m.sum())
    y_r, y_q = c_ref.s.correct[m], c_q.s.correct[m]
    sr, sq = _neginf(c_ref.s.signals[sig][m]), _neginf(c_q.s.signals[sig][m])
    pr_m, pq_m = pr[0][m], pq[0][m]

    def d_auroc(lvl: float) -> st.Interval:
        return st.paired_difference(
            st.batch_auroc(sq, y_q), st.batch_auroc(sr, y_r), n, level=lvl, **_boot(cfg, 80 + j)
        )

    def d_ece(lvl: float) -> st.Interval:
        return st.paired_difference(
            st.batch_ece(pq_m, y_q), st.batch_ece(pr_m, y_r), n, level=lvl, **_boot(cfg, 90 + j)
        )

    a95, a90, e95, e90 = d_auroc(0.95), d_auroc(0.90), d_ece(0.95), d_ece(0.90)
    return {
        "delta_auroc": _iv(a95),
        "delta_ece": _iv(e95),
        "auroc_equivalent": st.tost_bootstrap(a90, cfg.equivalence_margin),
        "ece_equivalent": st.tost_bootstrap(e90, cfg.equivalence_margin),
        "accuracy_S": float(y_q.mean()),
    }


def quant_section(
    data: ExperimentData,
    cfg: ExperimentConfig,
    plan: Any,
    ev0: dict[str, PolicyEval],
    quant_split: dict[str, dict[str, dict[int, dict[str, Any] | None]]],
    certs_by_k: dict[int, SplitCertification],
    store: SplitStore,
) -> dict[str, Any]:
    from sieve.eval.analysis import aggregate_splits

    ref = plan.quant_ref
    if ref is None or not plan.quant_pairs:
        return {"state": "not_run"}
    sec: dict[str, Any] = {"reference": ref.s_cell, "margin": cfg.equivalence_margin, "configs": {}}
    k0 = certs_by_k[0]
    sig = k0.primary_signal
    c_ids = store.open_c(0, cert_hash=k0.bundle_hash)
    a_ids = store.a_ids(0)
    c_ref, a_ref = data.slice_pair(ref, c_ids), data.slice_pair(ref, a_ids)
    pr = _platt_prob(a_ref.s, c_ref.s, sig)
    for j, qp in enumerate(plan.quant_pairs):
        row: dict[str, Any] = {"s_cell": qp.s_cell}
        qs = quant_split.get(qp.name, {})
        row["transferred"] = aggregate_splits(qs.get("transferred", {}), cfg.alpha)
        row["recalibrated"] = aggregate_splits(qs.get("recalibrated", {}), cfg.alpha)
        c_q, a_q = data.slice_pair(qp, c_ids), data.slice_pair(qp, a_ids)
        row.update(_quant_deltas(cfg, sig, c_ref, c_q, pr, _platt_prob(a_q.s, c_q.s, sig), j))
        sec["configs"][qp.name] = row
    return sec


def threshold_shift_note() -> str:
    return json.dumps(
        {"null_control": "not run: needs a repeat build/seed cell (Red-Team kill test 11)"}
    )

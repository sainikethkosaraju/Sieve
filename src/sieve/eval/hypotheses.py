# ruff: noqa: E501
"""Hypothesis report H1-H7: SUPPORTED / NOT SUPPORTED / INCONCLUSIVE, with the numbers.

The decision rules below transcribe docs/PREREGISTRATION.md; the numeric thresholds live in
``sieve.eval.prereg.CRITERIA`` and a test pins them to the document text. The report refuses
to run when the frozen pre-registration no longer matches the hash the data was collected
under, and never edits a criterion. Where the pre-registration leaves an operational detail
open (which split, what 'matches' means numerically), the choice is stated in ``rule`` and in
docs/evaluation.md; none of them relaxes a criterion.
"""

from __future__ import annotations

from typing import Any

from sieve.eval import prereg
from sieve.eval.config import ExperimentConfig

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT SUPPORTED"
INCONCLUSIVE = "INCONCLUSIVE"


class HypothesisReportRefusedError(RuntimeError):
    pass


def _v(
    hid: str, title: str, verdict: str, rule: str, numbers: dict[str, Any], why: str
) -> dict[str, Any]:
    return {
        "id": hid,
        "title": title,
        "verdict": verdict,
        "rule": rule,
        "numbers": numbers,
        "why": why,
    }


def h1(res: dict[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    h = res["h1"]
    rule = (
        f"Across {h['n_splits']} re-splits, test risk on C > alpha={cfg.alpha} in <= "
        f"{prereg.CRITERIA.h1_pass_max:.0%} of certified splits (SUPPORTED); > "
        f"{prereg.CRITERIA.h1_fail_min:.0%} (NOT SUPPORTED: bug or non-exchangeability); "
        "in between or fewer than 10 certified splits: INCONCLUSIVE."
    )
    nums = {
        k: h[k]
        for k in (
            "n_certified",
            "n_empty",
            "n_violations",
            "violation_freq",
            "violation_ci95",
            "mean_test_risk",
            "max_test_risk",
            "mean_coverage",
        )
    }
    if h["n_certified"] < 10:
        return _v(
            "H1",
            "validity",
            INCONCLUSIVE,
            rule,
            nums,
            "fewer than 10 re-splits certified anything (empty feasible set is a declared negative outcome)",
        )
    f = h["violation_freq"]
    ci = h.get("violation_ci95") or [float("nan"), float("nan")]
    hi_ci = float(ci[1])
    if f <= prereg.CRITERIA.h1_pass_max:
        if not (hi_ci <= prereg.CRITERIA.h1_pass_max):  # also true for a missing/NaN bound
            return _v(
                "H1",
                "validity",
                INCONCLUSIVE,
                rule,
                nums,
                f"SUPPORTED-WEAK: violation frequency {f:.3f} <= {prereg.CRITERIA.h1_pass_max:.0%} "
                f"but the Clopper-Pearson 95% upper bound {hi_ci:.3f} exceeds it, so the "
                "re-splits cannot exclude a higher violation rate (reported as INCONCLUSIVE)",
            )
        return _v(
            "H1",
            "validity",
            SUPPORTED,
            rule,
            nums,
            f"violation frequency {f:.3f}, CP95 upper bound {hi_ci:.3f} <= "
            f"{prereg.CRITERIA.h1_pass_max:.0%}",
        )
    if f > prereg.CRITERIA.h1_fail_min:
        return _v("H1", "validity", NOT_SUPPORTED, rule, nums, f"violation frequency {f:.3f}")
    return _v(
        "H1",
        "validity",
        INCONCLUSIVE,
        rule,
        nums,
        f"violation frequency {f:.3f} between thresholds",
    )


def h2(res: dict[str, Any]) -> dict[str, Any]:
    d = res["signals"]["h2_delta_vs_max_prob"]
    m = prereg.CRITERIA.h2_delta_auroc
    rule = (
        f"SUPPORTED if mean_logprob or a margin signal beats max_prob by >= {m} AUROC with a paired "
        f"95% CI excluding 0; NOT SUPPORTED if every single-token signal is within +-{m} of max_prob; "
        "otherwise INCONCLUSIVE."
    )
    cand = {k: d[k] for k in ("mean_logprob", "first_margin", "mean_margin") if k in d}
    if not d:
        return _v(
            "H2",
            "signal",
            INCONCLUSIVE,
            rule,
            {},
            "AUROC deltas unavailable (too few graded items)",
        )
    if any(c["point"] >= m and c["lo"] > 0 for c in cand.values()):
        return _v(
            "H2", "signal", SUPPORTED, rule, d, "a candidate signal clears +0.02 with CI above 0"
        )
    if all(abs(c["point"]) <= m for c in d.values()):
        return _v(
            "H2",
            "signal",
            NOT_SUPPORTED,
            rule,
            d,
            "all single token signals within +-0.02 of max_prob",
        )
    return _v("H2", "signal", INCONCLUSIVE, rule, d, "neither criterion met")


def h3(res: dict[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    c = res["comparisons"]
    rule = (
        "SUPPORTED if (a) the correct-answer rate (answered and correct, per item, i.e. coverage-aware) "
        f"vs L-only is non-inferior (lower paired CI >= -{cfg.noninferiority_margin}; accuracy-on-answered "
        "at differing coverage is reported but not decisive), (b) cost per correct answer incl. wasted S attempts is lower "
        "with the CI below 0, (c) the gate beats the matched-coverage random router and (d) query-only "
        "router (paired CI of accuracy-on-answered difference > 0). NOT SUPPORTED if indistinguishable "
        "from the random router (CI includes 0 or below) or the point estimate of the cost advantage "
        "is lost. Otherwise INCONCLUSIVE."
    )
    if not c.get("gate_available"):
        return _v(
            "H3", "cascade value", INCONCLUSIVE, rule, {}, c.get("reason", "gate unavailable")
        )
    ni = c.get("noninferiority_vs_L_only")
    rnd = c.get("matched", {}).get("random_router")
    qry = c.get("matched", {}).get("query_router")
    nums: dict[str, Any] = {"noninferiority_vs_L_only": ni}
    if rnd:
        nums["vs_random_matched"] = rnd["vs_gate"]["acc_on_answered_diff"]
        nums["random_matched_thresholds"] = rnd["thresholds"]
    if qry:
        nums["vs_query_matched"] = qry["vs_gate"]["acc_on_answered_diff"]
    if ni is None or rnd is None:
        return _v("H3", "cascade value", INCONCLUSIVE, rule, nums, "comparison inputs missing")
    cost_point = ni["cost_per_correct_diff"]["point"]
    r = rnd["vs_gate"]["acc_on_answered_diff"]
    # `not (x > 0)` so that NaN bounds count as failure, never as success
    if not (r["lo"] > 0) or not (cost_point < 0):
        why = []
        if not (r["lo"] > 0):
            why.append("not distinguishable from (or worse than) the matched random router")
        if not (cost_point < 0):
            why.append(
                "no cost-per-correct advantage over L-only once wasted S attempts are charged"
            )
        return _v("H3", "cascade value", NOT_SUPPORTED, rule, nums, "; ".join(why))
    q_ok = qry is not None and bool(qry["vs_gate"]["acc_on_answered_diff"]["lo"] > 0)
    if ni["non_inferior"] and ni["cheaper_per_correct"] and q_ok:
        return _v("H3", "cascade value", SUPPORTED, rule, nums, "all four conditions met")
    miss = [
        n
        for n, ok in (
            ("non-inferiority", ni["non_inferior"]),
            ("cost CI below 0", ni["cheaper_per_correct"]),
            ("beats query-only router", q_ok),
        )
        if not ok
    ]
    return _v(
        "H3", "cascade value", INCONCLUSIVE, rule, nums, "not established: " + ", ".join(miss)
    )


def h4(res: dict[str, Any]) -> dict[str, Any]:
    sc = res["self_consistency"]
    m = prereg.CRITERIA.h4_delta_auroc
    rule = (
        f"SUPPORTED if k={prereg.CRITERIA.h4_k} self-consistency adds >= {m} AUROC over the best token "
        "signal (paired CI lower bound > 0; the frontier benefit additionally needs k x cost_S < "
        "cost_L, reported); NOT SUPPORTED if the CI upper bound is below the margin; else INCONCLUSIVE."
    )
    if sc.get("state") != "run" or sc.get("delta_auroc_vs_best_token_signal") is None:
        return _v(
            "H4",
            "self-consistency",
            INCONCLUSIVE,
            rule,
            sc,
            "self-consistency subset not run or too small",
        )
    d = sc["delta_auroc_vs_best_token_signal"]
    nums = {**sc}
    if d["point"] >= m and d["lo"] > 0:
        extra = (
            ""
            if sc["k_times_cost_S_below_cost_L"]
            else " (but k x cost_S >= cost_L: no frontier benefit)"
        )
        return _v(
            "H4", "self-consistency", SUPPORTED, rule, nums, "AUROC gain criterion met" + extra
        )
    if d["hi"] < m:
        return _v("H4", "self-consistency", NOT_SUPPORTED, rule, nums, "no gain of >= 0.03 AUROC")
    return _v("H4", "self-consistency", INCONCLUSIVE, rule, nums, "CI straddles the 0.03 margin")


def h5(res: dict[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    sh = res["shift"]
    rule = (
        f"SUPPORTED if the certified policy's risk exceeds alpha={cfg.alpha} on NQ-open (95% CI lower "
        "bound > alpha), exceeds NQ-open's risk on SimpleQA Verified, and the label-free drift alarm "
        f"(KS on {cfg.window_size}-query windows) fires on >= {prereg.CRITERIA.h5_alarm_rate:.0%} of "
        "shifted windows. NOT SUPPORTED if the guarantee holds on NQ-open (CI upper bound <= alpha) or "
        "the alarm is blind (< 80% of shifted windows). 'Before violation' is operationalised as: the "
        "alarm uses scores only and is therefore available before any label."
    )
    nq, sq = sh.get("nq_open"), sh.get("simpleqa_verified")
    nums = {"nq_open": nq, "simpleqa_verified": sq}
    if nq is None or nq.get("risk_ci95_split0") is None:
        return _v(
            "H5", "shift", INCONCLUSIVE, rule, nums, "NQ-open not run, ungraded or nothing answered"
        )
    ci = nq["risk_ci95_split0"]
    alarm = [
        x["drift"]["alarm_rate"] for x in (nq, sq) if x and x["drift"]["alarm_rate"] is not None
    ]
    if ci["hi"] <= cfg.alpha:
        return _v("H5", "shift", NOT_SUPPORTED, rule, nums, "risk on NQ-open does not exceed alpha")
    if alarm and min(alarm) < prereg.CRITERIA.h5_alarm_rate:
        return _v(
            "H5", "shift", NOT_SUPPORTED, rule, nums, f"drift alarm rate {min(alarm):.2f} < 0.80"
        )
    if sq is None or not sq.get("graded") or sq.get("risk_ci95_split0") is None:
        return _v(
            "H5",
            "shift",
            INCONCLUSIVE,
            rule,
            nums,
            f"SimpleQA Verified risk unavailable (judge: {res['meta']['judge']})",
        )
    sq_ci = sq["risk_ci95_split0"]
    if ci["lo"] > cfg.alpha and sq_ci["point"] > ci["point"] and alarm and min(alarm) >= 0.8:
        return _v(
            "H5",
            "shift",
            SUPPORTED,
            rule,
            nums,
            "risk exceeds alpha on NQ-open and more on SimpleQA; alarm fires",
        )
    return _v(
        "H5",
        "shift",
        INCONCLUSIVE,
        rule,
        nums,
        "NQ-open CI includes alpha or SimpleQA risk not larger",
    )


MIN_CERTIFIED = 10


def _cp_lo(agg: dict[str, Any]) -> float:
    ci: Any = agg.get("violation_ci95")
    try:
        return float(ci[0])
    except (TypeError, ValueError, IndexError):
        return float("nan")


def h6(res: dict[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    q = res["quant"]
    m = prereg.CRITERIA.h6_margin
    rule = (
        "SUPPORTED if a threshold fitted on the f16 reference violates alpha in > delta of re-splits on "
        f"some quantized config while per-config recalibration does not (<= delta). NOT SUPPORTED if for "
        f"all configs delta-AUROC and delta-ECE are equivalent within +-{m} (90% paired-bootstrap CI "
        "inside the margin, TOST) and transferred thresholds hold in >= 1-delta of re-splits. "
        "Otherwise INCONCLUSIVE (partial support is reported in the numbers)."
    )
    if q.get("state") == "not_run" or not q.get("configs"):
        return _v(
            "H6", "quantization", INCONCLUSIVE, rule, {}, "no quantization ladder in this run"
        )
    cfgs = q["configs"]
    rows: dict[str, Any] = {}
    supported: list[str] = []
    all_null = True
    for name, r in cfgs.items():
        ft, fr = r["transferred"].get("violation_freq"), r["recalibrated"].get("violation_freq")
        rows[name] = {
            "transferred_violation_freq": ft,
            "recalibrated_violation_freq": fr,
            "delta_auroc": r.get("delta_auroc"),
            "delta_ece": r.get("delta_ece"),
            "auroc_equivalent": r.get("auroc_equivalent"),
            "ece_equivalent": r.get("ece_equivalent"),
        }
        nt, nr = r["transferred"].get("n_certified", 0), r["recalibrated"].get("n_certified", 0)
        if ft is None or fr is None or nt < MIN_CERTIFIED or nr < MIN_CERTIFIED:
            all_null = False  # too few certified splits (like H1): INCONCLUSIVE, never SUPPORTED
            rows[name]["note"] = (
                f"fewer than {MIN_CERTIFIED} certified splits (transferred {nt}, recalibrated {nr})"
            )
            continue
        t_lo = _cp_lo(r["transferred"])
        r_lo = _cp_lo(r["recalibrated"])
        # F-06: compare with CP bounds: transfer credibly violates (lower bound > delta) while
        # recalibration is not credibly violating (lower bound <= delta) and its point holds
        if t_lo > cfg.delta and fr <= cfg.delta and not (r_lo > cfg.delta):
            supported.append(name)
        null_here = (
            bool(r.get("auroc_equivalent")) and bool(r.get("ece_equivalent")) and ft <= cfg.delta
        )
        all_null = all_null and null_here
    if supported:
        return _v(
            "H6",
            "quantization",
            SUPPORTED,
            rule,
            rows,
            "transfer violates but recalibration holds on: " + ", ".join(supported),
        )
    if all_null:
        return _v(
            "H6",
            "quantization",
            NOT_SUPPORTED,
            rule,
            rows,
            "all configs equivalent and transferred thresholds hold",
        )
    return _v(
        "H6",
        "quantization",
        INCONCLUSIVE,
        rule,
        rows,
        "neither the shift nor the equivalence criterion is met for all configs",
    )


def h7(res: dict[str, Any]) -> dict[str, Any]:
    hs = res["hard_set"]
    lim = prereg.CRITERIA.h7_auroc
    rule = (
        f"SUPPORTED if gate AUROC on MMLU-Redux < {lim} (95% CI upper bound < {lim}); NOT SUPPORTED if "
        f"the CI lower bound >= {lim}; otherwise INCONCLUSIVE. Reported as a limitation either way."
    )
    if hs.get("state") != "run":
        return _v("H7", "hard sets", INCONCLUSIVE, rule, {}, "MMLU-Redux not run")
    a = hs["auroc"].get("S", {}).get(hs["primary_signal"])
    if a is None:
        return _v("H7", "hard sets", INCONCLUSIVE, rule, hs, "AUROC unavailable")
    if a["hi"] < lim:
        return _v("H7", "hard sets", SUPPORTED, rule, hs, f"AUROC {a['point']:.3f}")
    if a["lo"] >= lim:
        return _v("H7", "hard sets", NOT_SUPPORTED, rule, hs, f"AUROC {a['point']:.3f}")
    return _v(
        "H7", "hard sets", INCONCLUSIVE, rule, hs, f"AUROC {a['point']:.3f}, CI straddles {lim}"
    )


def negative_outcomes(res: dict[str, Any], cfg: ExperimentConfig) -> list[str]:
    """Pre-declared negative outcomes that actually occurred (reported with equal prominence)."""
    out: list[str] = []
    h = res["h1"]
    if h["n_empty"] > 0:
        out.append(f"empty feasible threshold set on {h['n_empty']} of {h['n_splits']} re-splits")
    if h.get("mean_coverage") is not None and h["mean_coverage"] < 0.05:
        out.append(f"near-zero coverage (mean {h['mean_coverage']:.3f})")
    floor = res["certs_split0"]["gate:primary"].get("feasibility_floor")
    if floor is not None and floor > cfg.alpha:
        out.append(f"bounds too loose: smallest certifiable alpha {floor:.3f} > alpha {cfg.alpha}")
    for sig, d in res["signals"]["h2_delta_vs_max_prob"].items():
        if d["lo"] <= 0 <= d["hi"]:
            out.append(f"bootstrap CI spans zero: delta-AUROC({sig} - max_prob)")
    return out


def check_manifest_config_hash(res: dict[str, Any], cfg: ExperimentConfig) -> None:
    """Refuse to report on runs whose config differs from the one in the run manifest (F-08)."""
    meta = res.get("meta", {})
    run_hash = meta.get("config_hash")
    manifest_hash = meta.get("manifest_config_hash")
    here = cfg.config_hash()
    if manifest_hash is None:
        raise HypothesisReportRefusedError("results carry no manifest config hash; refusing")
    if here != manifest_hash or (run_hash is not None and run_hash != manifest_hash):
        raise HypothesisReportRefusedError(
            f"config hash {here[:12]} / analysis {str(run_hash)[:12]} differs from the run "
            f"manifest hash {str(manifest_hash)[:12]}: the analysed runs were produced under a "
            "different configuration"
        )


def build_report(res: dict[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    """Verify the pre-registration pin, then evaluate H1-H7."""
    crit_missing = prereg.criteria_missing(cfg.prereg_path)
    if crit_missing:
        raise HypothesisReportRefusedError(
            f"coded criteria no longer match the pre-registration text: {crit_missing}"
        )
    try:
        current = prereg.verify_unchanged(
            cfg.prereg_path, cfg.prereg_frozen_sha256, res["meta"].get("prereg_frozen_sha256")
        )
    except prereg.PreregistrationChangedError as e:
        raise HypothesisReportRefusedError(str(e)) from e
    check_manifest_config_hash(res, cfg)
    verdicts = [h1(res, cfg), h2(res), h3(res, cfg), h4(res), h5(res, cfg), h6(res, cfg), h7(res)]
    return {
        "synthetic": cfg.synthetic,
        "synthetic_label": res["meta"].get("synthetic_label"),
        "preregistration_frozen_sha256": current,
        "deviations": prereg.deviations_text(cfg.prereg_path),
        "verdicts": verdicts,
        "negative_outcomes": negative_outcomes(res, cfg),
        "alpha": cfg.alpha,
        "delta": cfg.delta,
    }


def format_report(rep: dict[str, Any]) -> str:
    lines: list[str] = []
    if rep.get("synthetic"):
        lines += [rep["synthetic_label"], ""]
    lines.append(
        f"Hypothesis report (pre-registration frozen hash {rep['preregistration_frozen_sha256'][:16]})"
    )
    lines.append("")
    for v in rep["verdicts"]:
        lines.append(f"{v['id']} {v['title']}: {v['verdict']}")
        lines.append(f"    rule: {v['rule']}")
        lines.append(f"    why : {v['why']}")
    lines.append("")
    lines.append(
        "Pre-declared negative outcomes observed: "
        + ("; ".join(rep["negative_outcomes"]) or "none")
    )
    dev = rep["deviations"]
    lines.append("")
    lines.append(dev if dev else "Deviations section: (empty)")
    return "\n".join(lines)

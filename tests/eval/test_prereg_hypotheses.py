"""Pre-registration pinning and the decision rules of the hypothesis report."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import pytest

from sieve.eval import hypotheses as hy
from sieve.eval import pipeline, prereg
from sieve.eval.config import ExperimentConfig, parse_config

REAL = prereg.resolve(prereg.DEFAULT_PATH)


def _cfg(path: str | Path = prereg.DEFAULT_PATH, pin: str | None = "auto") -> ExperimentConfig:
    d = pipeline.mock_config(small=True)
    d["preregistration"]["path"] = str(path)
    d["preregistration"]["frozen_sha256"] = prereg.frozen_hash(path) if pin == "auto" else pin
    return parse_config(d)


def iv(p: float, lo: float, hi: float) -> dict[str, float]:
    return {"point": p, "lo": lo, "hi": hi}


def _agg(f: float, lo: float, hi: float, n: int = 50) -> dict[str, Any]:
    return {"violation_freq": f, "violation_ci95": [lo, hi], "n_certified": n}


def base_results(cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "meta": {
            "prereg_frozen_sha256": prereg.frozen_hash(cfg.prereg_path),
            "synthetic_label": None,
            "judge": "unavailable",
            "config_hash": cfg.config_hash(),
            "manifest_config_hash": cfg.config_hash(),
        },
        "h1": {
            "n_splits": 50,
            "n_certified": 50,
            "n_empty": 0,
            "n_violations": 3,
            "violation_freq": 0.06,
            "violation_ci95": [0.01, 0.14],
            "mean_test_risk": 0.07,
            "max_test_risk": 0.11,
            "mean_coverage": 0.5,
        },
        "signals": {
            "h2_delta_vs_max_prob": {
                "mean_logprob": iv(0.03, 0.01, 0.05),
                "first_margin": iv(0.0, -0.02, 0.02),
            }
        },
        "certs_split0": {"gate:primary": {"feasibility_floor": 0.01}},
        "comparisons": {
            "gate_available": True,
            "noninferiority_vs_L_only": {
                "non_inferior": True,
                "cheaper_per_correct": True,
                "cost_per_correct_diff": iv(-0.2, -0.3, -0.1),
                "acc_on_answered_diff": iv(0.0, -0.01, 0.01),
                "margin": 0.02,
            },
            "matched": {
                "random_router": {
                    "thresholds": [0.1, 0.2],
                    "vs_gate": {"acc_on_answered_diff": iv(0.05, 0.02, 0.08)},
                },
                "query_router": {
                    "thresholds": [0.1, 0.2],
                    "vs_gate": {"acc_on_answered_diff": iv(0.04, 0.01, 0.07)},
                },
            },
        },
        "self_consistency": {
            "state": "run",
            "delta_auroc_vs_best_token_signal": iv(0.05, 0.02, 0.08),
            "k_times_cost_S_below_cost_L": True,
        },
        "shift": {
            "nq_open": {
                "graded": True,
                "risk_ci95_split0": iv(0.18, 0.15, 0.21),
                "drift": {"alarm_rate": 0.9},
            },
            "simpleqa_verified": {
                "graded": True,
                "risk_ci95_split0": iv(0.5, 0.45, 0.55),
                "drift": {"alarm_rate": 1.0},
            },
        },
        "quant": {
            "configs": {
                "q3": {
                    "transferred": _agg(0.3, 0.18, 0.43),
                    "recalibrated": _agg(0.04, 0.0, 0.10),
                    "delta_auroc": iv(-0.03, -0.05, -0.01),
                    "delta_ece": iv(0.05, 0.03, 0.07),
                    "auroc_equivalent": False,
                    "ece_equivalent": False,
                }
            }
        },
        "hard_set": {
            "state": "run",
            "primary_signal": "mean_logprob",
            "auroc": {"S": {"mean_logprob": iv(0.55, 0.5, 0.58)}},
        },
    }


# --- pinning --------------------------------------------------------------------------------


def test_coded_criteria_match_the_frozen_text() -> None:
    assert prereg.criteria_missing(prereg.DEFAULT_PATH) == []


def test_criteria_guard_notices_a_changed_number(tmp_path: Path) -> None:
    p = tmp_path / "P.md"
    p.write_text(REAL.read_text().replace("<= 15%", "<= 20%"))
    assert "h1_pass_max" in prereg.criteria_missing(p)


def test_hash_ignores_deviations_but_not_the_frozen_body(tmp_path: Path) -> None:
    p = tmp_path / "P.md"
    shutil.copy(REAL, p)
    h0 = prereg.frozen_hash(p)
    p.write_text(p.read_text().replace("None yet.", "2026-12-01: switched S to the 1.5B ladder."))
    assert prereg.frozen_hash(p) == h0  # dated deviations are the allowed way to change course
    assert "1.5B" in prereg.deviations_text(p)
    p.write_text(p.read_text().replace("alpha = 0.10", "alpha = 0.05"))
    assert prereg.frozen_hash(p) != h0


def test_report_refuses_if_the_preregistration_changed(tmp_path: Path) -> None:
    p = tmp_path / "P.md"
    shutil.copy(REAL, p)
    cfg = _cfg(p)  # pinned to the current hash
    res = base_results(cfg)
    assert hy.build_report(res, cfg)["verdicts"]  # fine while unchanged
    p.write_text(p.read_text().replace("Statistics", "Statistics (edited after the fact)"))
    with pytest.raises(hy.HypothesisReportRefusedError, match="pre-registration changed"):
        hy.build_report(res, cfg)


def test_report_refuses_if_data_were_collected_under_another_hash() -> None:
    cfg = _cfg()
    res = base_results(cfg)
    res["meta"]["prereg_frozen_sha256"] = "0" * 64  # manifest of the run disagrees
    with pytest.raises(hy.HypothesisReportRefusedError):
        hy.build_report(res, cfg)
    cfg_wrong_pin = _cfg(pin="f" * 64)
    with pytest.raises(hy.HypothesisReportRefusedError):
        hy.build_report(base_results(cfg_wrong_pin), cfg_wrong_pin)


def test_run_refuses_if_the_pin_is_stale(tmp_path: Path) -> None:
    from sieve.eval._fake_backend import FakeProvider

    cfg = _cfg(pin="e" * 64)
    with pytest.raises(prereg.PreregistrationChangedError):
        pipeline.run_generation(cfg, tmp_path, FakeProvider(), meter=None)


# --- decision rules -------------------------------------------------------------------------


def verdicts(res: dict[str, Any], cfg: ExperimentConfig | None = None) -> dict[str, str]:
    cfg = cfg or _cfg()
    return {v["id"]: v["verdict"] for v in hy.build_report(res, cfg)["verdicts"]}


def test_all_supported_baseline() -> None:
    v = verdicts(base_results(_cfg()))
    assert v == dict.fromkeys(["H1", "H2", "H3", "H4", "H5", "H6", "H7"], "SUPPORTED")


def _mut(fn: Any) -> dict[str, str]:
    cfg = _cfg()
    res = copy.deepcopy(base_results(cfg))
    fn(res)
    return verdicts(res, cfg)


def test_h1_thresholds() -> None:
    assert _mut(lambda r: r["h1"].update(violation_freq=0.15))["H1"] == "SUPPORTED"
    assert _mut(lambda r: r["h1"].update(violation_freq=0.16))["H1"] == "INCONCLUSIVE"
    assert _mut(lambda r: r["h1"].update(violation_freq=0.25))["H1"] == "INCONCLUSIVE"
    assert _mut(lambda r: r["h1"].update(violation_freq=0.26))["H1"] == "NOT SUPPORTED"
    assert (
        _mut(lambda r: r["h1"].update(n_certified=3))["H1"] == "INCONCLUSIVE"
    )  # empty feasible set


def test_h2_rules() -> None:
    assert _mut(lambda r: None)["H2"] == "SUPPORTED"
    null = {"mean_logprob": iv(0.01, -0.01, 0.03), "first_margin": iv(-0.01, -0.03, 0.01)}
    assert _mut(lambda r: r["signals"].update(h2_delta_vs_max_prob=null))["H2"] == "NOT SUPPORTED"
    mid = {"mean_logprob": iv(0.03, -0.01, 0.07)}  # point clears 0.02 but CI includes 0
    assert _mut(lambda r: r["signals"].update(h2_delta_vs_max_prob=mid))["H2"] == "INCONCLUSIVE"
    assert _mut(lambda r: r["signals"].update(h2_delta_vs_max_prob={}))["H2"] == "INCONCLUSIVE"


def test_h3_rules() -> None:
    def lose_cost(r: dict[str, Any]) -> None:
        r["comparisons"]["noninferiority_vs_L_only"]["cost_per_correct_diff"] = iv(0.1, 0.05, 0.15)

    def like_random(r: dict[str, Any]) -> None:
        r["comparisons"]["matched"]["random_router"]["vs_gate"]["acc_on_answered_diff"] = iv(
            0.0, -0.02, 0.02
        )

    def loses_to_query(r: dict[str, Any]) -> None:
        r["comparisons"]["matched"]["query_router"]["vs_gate"]["acc_on_answered_diff"] = iv(
            0.01, -0.01, 0.03
        )

    def not_ni(r: dict[str, Any]) -> None:
        r["comparisons"]["noninferiority_vs_L_only"]["non_inferior"] = False

    assert _mut(lose_cost)["H3"] == "NOT SUPPORTED"
    assert _mut(like_random)["H3"] == "NOT SUPPORTED"
    assert _mut(loses_to_query)["H3"] == "INCONCLUSIVE"
    assert _mut(not_ni)["H3"] == "INCONCLUSIVE"
    assert _mut(lambda r: r["comparisons"].update(gate_available=False))["H3"] == "INCONCLUSIVE"


def test_h4_rules() -> None:
    def gain(lo: float, p: float, hi: float) -> Any:
        return lambda r: r["self_consistency"].update(
            delta_auroc_vs_best_token_signal=iv(p, lo, hi)
        )

    assert _mut(gain(0.02, 0.05, 0.08))["H4"] == "SUPPORTED"
    assert _mut(gain(-0.02, 0.0, 0.02))["H4"] == "NOT SUPPORTED"
    assert _mut(gain(-0.01, 0.03, 0.07))["H4"] == "INCONCLUSIVE"
    assert _mut(lambda r: r.update(self_consistency={"state": "not_run"}))["H4"] == "INCONCLUSIVE"


def test_h5_rules() -> None:
    def nq(ci: dict[str, float], alarm: float = 0.9) -> Any:
        def f(r: dict[str, Any]) -> None:
            r["shift"]["nq_open"].update(risk_ci95_split0=ci, drift={"alarm_rate": alarm})

        return f

    assert _mut(nq(iv(0.05, 0.03, 0.09)))["H5"] == "NOT SUPPORTED"  # guarantee holds under shift
    assert _mut(nq(iv(0.18, 0.15, 0.21), alarm=0.5))["H5"] == "NOT SUPPORTED"  # alarm blind
    assert _mut(nq(iv(0.11, 0.08, 0.14)))["H5"] == "INCONCLUSIVE"  # CI includes alpha

    def no_judge(r: dict[str, Any]) -> None:
        r["shift"]["simpleqa_verified"].update(graded=False, risk_ci95_split0=None)

    assert _mut(no_judge)["H5"] == "INCONCLUSIVE"  # judge unavailable: never guessed


def test_h6_rules() -> None:
    def null(r: dict[str, Any]) -> None:
        r["quant"]["configs"]["q3"].update(
            transferred=_agg(0.04, 0.0, 0.1),
            recalibrated=_agg(0.04, 0.0, 0.1),
            auroc_equivalent=True,
            ece_equivalent=True,
        )

    assert _mut(null)["H6"] == "NOT SUPPORTED"
    assert _mut(lambda r: None)["H6"] == "SUPPORTED"
    assert (
        _mut(lambda r: r["quant"]["configs"]["q3"].update(recalibrated=_agg(0.3, 0.18, 0.43)))["H6"]
        == "INCONCLUSIVE"
    )
    assert _mut(lambda r: r["quant"].update(state="not_run", configs={}))["H6"] == "INCONCLUSIVE"


def test_h6_needs_ten_certified_splits() -> None:
    """F-06: one certified split with frequency 0 vs a transferred 0.5 must not be SUPPORTED."""

    def one(r: dict[str, Any]) -> None:
        r["quant"]["configs"]["q3"].update(
            transferred=_agg(0.5, 0.01, 0.99, n=1), recalibrated=_agg(0.0, 0.0, 0.98, n=1)
        )

    assert _mut(one)["H6"] == "INCONCLUSIVE"

    def few_transferred(r: dict[str, Any]) -> None:
        r["quant"]["configs"]["q3"].update(transferred=_agg(0.5, 0.01, 0.99, n=3))

    assert _mut(few_transferred)["H6"] == "INCONCLUSIVE"

    def not_credible(r: dict[str, Any]) -> None:  # point > delta but CP lower bound <= delta
        r["quant"]["configs"]["q3"].update(transferred=_agg(0.2, 0.08, 0.3, n=12))

    assert _mut(not_credible)["H6"] == "INCONCLUSIVE"


def test_h1_considers_the_confidence_interval() -> None:
    """F-08: 7/50 style point estimates with a wide CP interval are not SUPPORTED."""
    weak = _mut(lambda r: r["h1"].update(violation_freq=0.14, violation_ci95=[0.058, 0.267]))
    assert weak["H1"] == "INCONCLUSIVE"
    cfg = _cfg()
    res = base_results(cfg)
    res["h1"].update(violation_freq=0.14, violation_ci95=[0.058, 0.267])
    why = next(v["why"] for v in hy.build_report(res, cfg)["verdicts"] if v["id"] == "H1")
    assert why.startswith("SUPPORTED-WEAK")
    nan = _mut(lambda r: r["h1"].update(violation_ci95=[float("nan"), float("nan")]))
    assert nan["H1"] == "INCONCLUSIVE"


def test_h3_nan_bounds_never_pass() -> None:
    nan = float("nan")

    def nan_random(r: dict[str, Any]) -> None:
        r["comparisons"]["matched"]["random_router"]["vs_gate"]["acc_on_answered_diff"] = iv(
            nan, nan, nan
        )

    def nan_cost(r: dict[str, Any]) -> None:
        r["comparisons"]["noninferiority_vs_L_only"]["cost_per_correct_diff"] = iv(nan, nan, nan)

    assert _mut(nan_random)["H3"] == "NOT SUPPORTED"
    assert _mut(nan_cost)["H3"] == "NOT SUPPORTED"


def test_report_refuses_if_config_hash_differs_from_manifest() -> None:
    cfg = _cfg()
    res = base_results(cfg)
    res["meta"]["manifest_config_hash"] = "f" * 64
    with pytest.raises(hy.HypothesisReportRefusedError, match="manifest"):
        hy.build_report(res, cfg)
    res = base_results(cfg)
    res["meta"]["manifest_config_hash"] = None
    with pytest.raises(hy.HypothesisReportRefusedError):
        hy.build_report(res, cfg)


def test_h7_rules() -> None:
    assert _mut(lambda r: None)["H7"] == "SUPPORTED"
    hi = {"S": {"mean_logprob": iv(0.8, 0.77, 0.83)}}
    assert _mut(lambda r: r["hard_set"].update(auroc=hi))["H7"] == "NOT SUPPORTED"
    mid = {"S": {"mean_logprob": iv(0.58, 0.52, 0.64)}}
    assert _mut(lambda r: r["hard_set"].update(auroc=mid))["H7"] == "INCONCLUSIVE"


def test_negative_outcomes_are_listed() -> None:
    cfg = _cfg()
    res = base_results(cfg)
    res["h1"].update(n_empty=7, mean_coverage=0.01)
    res["certs_split0"]["gate:primary"]["feasibility_floor"] = 0.2
    out = hy.negative_outcomes(res, cfg)
    assert any("empty feasible" in s for s in out) and any("near-zero coverage" in s for s in out)
    assert any("too loose" in s for s in out) and any("spans zero" in s for s in out)


def test_report_text_flags_synthetic_runs() -> None:
    cfg = _cfg()
    res = base_results(cfg)
    res["meta"]["synthetic_label"] = "SYNTHETIC DATA: test"
    rep = hy.build_report(res, _cfg())
    rep["synthetic"], rep["synthetic_label"] = True, "SYNTHETIC DATA: test"
    text = hy.format_report(rep)
    assert text.startswith("SYNTHETIC DATA") and "H1 validity: SUPPORTED" in text

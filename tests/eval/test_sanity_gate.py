"""Eval-audit C1: the sanity report must fail a run whose completions do not stop cleanly."""

from __future__ import annotations

from typing import Any

from sieve.eval.providers import sanity


def _rep(stop: int, length: int, mismatch: int = 0, tier: str = "S") -> dict[str, Any]:
    blk = {
        "n": stop + length,
        "finish": {"stop": stop, "length": length},
        "defects": {"utf8_gap": mismatch} if mismatch else {},
    }
    return {"cells": {f"{tier}_cell": {"tier": tier, "datasets": {"triviaqa": blk}}}}


def test_clean_run_passes() -> None:
    g = sanity.check_gates(_rep(100, 0))
    assert g["ok"] and g["failures"] == [] and g["tiers"]["S"]["stop_rate"] == 1.0


def test_low_stop_rate_fails_per_tier() -> None:
    rep = _rep(100, 0, tier="S")
    rep["cells"]["L_cell"] = _rep(90, 10, tier="L")["cells"]["L_cell"]
    g = sanity.check_gates(rep)
    assert not g["ok"] and any("tier L" in f and "stop" in f for f in g["failures"])
    assert not any("tier S" in f for f in g["failures"])


def test_threshold_is_95_percent() -> None:
    assert sanity.check_gates(_rep(95, 5))["ok"]
    assert not sanity.check_gates(_rep(94, 6))["ok"]


def test_token_text_mismatches_fail() -> None:
    assert not sanity.check_gates(_rep(100, 0, mismatch=5))["ok"]
    assert sanity.check_gates(_rep(100, 0, mismatch=1))["ok"]


def test_empty_report_fails() -> None:
    assert not sanity.check_gates({"cells": {}})["ok"]


def test_cli_exits_nonzero_when_gate_fails(monkeypatch: Any, tmp_path: Any, capsys: Any) -> None:
    bad = {**_rep(10, 90), "gates": sanity.check_gates(_rep(10, 90))}
    monkeypatch.setattr(sanity, "load_config", lambda p: object())
    monkeypatch.setattr(sanity, "sanity_report", lambda *a, **k: bad)
    assert sanity.main(["--config", "c", "--workdir", str(tmp_path)]) == 1
    assert "SANITY GATE FAILED" in capsys.readouterr().err
    good = {**_rep(100, 0), "gates": sanity.check_gates(_rep(100, 0))}
    monkeypatch.setattr(sanity, "sanity_report", lambda *a, **k: good)
    assert sanity.main(["--config", "c", "--workdir", str(tmp_path)]) == 0


def test_collapsed_ablation_arm_is_a_finding_not_a_failure() -> None:
    """Real T4 run: 4-bit KV cache stopped on 2.7% of items; the headline cells were clean."""
    from sieve.eval.providers.sanity import check_gates

    def cell(tier: str, n: int, stop: int) -> dict[str, object]:
        return {
            "tier": tier,
            "datasets": {"triviaqa": {"n": n, "finish": {"stop": stop}, "defects": {}}},
        }

    rep = {
        "cells": {
            "S_q4km": cell("S", 1000, 996),
            "L_q4km": cell("L", 1000, 1000),
            "S_kvq4": cell("S", 1000, 27),
        }
    }
    strict = check_gates(rep)
    assert not strict["ok"]  # without a headline set every cell counts (previous behaviour)
    g = check_gates(rep, headline={"S_q4km", "L_q4km"})
    assert g["ok"] and not g["failures"]
    assert g["ablation_findings"]["S_kvq4"]["stop_rate"] < 0.05
    bad = {"cells": {**rep["cells"], "S_q4km": cell("S", 1000, 800)}}
    assert not check_gates(bad, headline={"S_q4km", "L_q4km"})["ok"]  # headline cells still gate

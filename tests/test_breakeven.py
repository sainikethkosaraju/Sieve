"""Break-even calculator: closed-form checks and the two measured v1/v2 operating points."""

from __future__ import annotations

import pytest

from sieve.breakeven import analyse, main


def test_formulas() -> None:
    b = analyse(1.0, 4.0, 0.5)
    assert b.cost_ratio == 4.0 and b.break_even_resolved == 0.25
    assert b.relative_cost == pytest.approx(0.25 + 0.5) and b.max_saving == 0.75 and b.pays


def test_exactly_at_break_even_does_not_pay() -> None:
    b = analyse(1.0, 4.0, 0.25)
    assert b.relative_cost == pytest.approx(1.0) and not b.pays


def test_measured_v1_and_v2_points_cost_more_than_large_alone() -> None:
    v1 = analyse(0.06195, 0.19156, 1 - 0.95926)  # TriviaQA run (docs/RESULTS.md)
    v2 = analyse(0.17093, 0.49273, 1 - 0.995)  # SQuAD 2.0 run
    assert 1.27 < v1.relative_cost < 1.30 and not v1.pays
    assert 1.33 < v2.relative_cost < 1.35 and not v2.pays


def test_rejects_bad_input_and_cli_runs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError):
        analyse(0.0, 1.0, 0.1)
    with pytest.raises(ValueError):
        analyse(1.0, 2.0, 1.5)
    assert main(["--cost-small", "1", "--cost-large", "10", "--resolved", "0.5"]) == 0
    assert "the cascade pays" in capsys.readouterr().out

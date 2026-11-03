"""Human-audit sheet generation and analysis."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from scipy.stats import beta

from sieve.eval.audit import (
    AuditCandidate,
    analyse_audit,
    cp_interval,
    make_audit_sheet,
    sensitivity_adjusted_risk,
)


def _cands() -> list[AuditCandidate]:
    out = []
    for tier in ("S", "L"):
        for i in range(300):
            out.append(
                AuditCandidate(
                    f"{tier}{i}", "triviaqa", tier, f"q{i}", (f"g{i}",), f"a{i}", i % 3 != 0
                )
            )
    return out


def test_clopper_pearson_matches_beta_quantiles() -> None:
    lo, hi = cp_interval(7, 40)
    assert lo == pytest.approx(beta.ppf(0.025, 7, 34))
    assert hi == pytest.approx(beta.ppf(0.975, 8, 33))
    assert cp_interval(0, 10)[0] == 0.0 and cp_interval(10, 10)[1] == 1.0
    assert cp_interval(0, 0) == (0.0, 1.0)


def test_sheet_is_blind_stratified_and_deterministic(tmp_path: Path) -> None:
    c = _cands()
    s1, k1 = make_audit_sheet(c, tmp_path / "a", n=200, seed=3, grader_id="g")
    s2, _ = make_audit_sheet(c, tmp_path / "b", n=200, seed=3, grader_id="g")
    assert s1.read_text() == s2.read_text()
    rows = list(csv.DictReader(s1.open()))
    assert len(rows) == 200
    assert "auto_label" not in rows[0] and "auto_correct" not in rows[0]  # blind
    assert all(r["human_label"] == "" for r in rows)
    import json

    key = json.loads(k1.read_text())
    counts: dict[str, int] = {}
    for meta in key["rows"].values():
        k = f"{meta['tier']}|{int(meta['auto_correct'])}"
        counts[k] = counts.get(k, 0) + 1
    assert set(counts.values()) == {50}  # equal allocation over (tier, auto label)


def test_analyser_recovers_planted_error_rates(tmp_path: Path) -> None:
    sheet, key_path = make_audit_sheet(_cands(), tmp_path, n=200, seed=1, grader_id="g")
    import json

    key = json.loads(key_path.read_text())
    rows = list(csv.DictReader(sheet.open()))
    # plant: S grader wrong-> human says correct in 20% of wrong rows; L: no errors
    seen = {"S|0": 0}
    for r in rows:
        meta = key["rows"][r["row_id"]]
        auto = meta["auto_correct"]
        human = auto
        if meta["tier"] == "S" and not auto:
            seen["S|0"] += 1
            human = seen["S|0"] % 5 == 0  # every 5th flips
        r["human_label"] = "correct" if human else "wrong"
    with sheet.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = analyse_audit(sheet, key_path)
    assert res.n_unlabelled == 0 and res.n_labelled == 200
    e1_s = res.by_tier["S"]["e1"]
    assert (e1_s.k, e1_s.n) == (10, 50) and e1_s.rate == pytest.approx(0.2)
    assert e1_s.lo < 0.2 < e1_s.hi
    assert res.by_tier["L"]["e1"].k == 0 and res.by_tier["S"]["e2"].k == 0
    assert res.pooled["e1"].k == 10 and res.pooled["e1"].n == 100
    # population-weighted error: S has 100 wrong of 300 -> 1/3 * 0.2
    assert res.overall_error["S"][0] == pytest.approx(0.2 / 3, rel=1e-6)
    assert res.tier_gap == pytest.approx(0.2 / 3, rel=1e-6)
    assert res.passes_kill_test() is False  # gap 6.7 pp > 2 pp


def test_unlabelled_rows_make_the_kill_test_undecidable(tmp_path: Path) -> None:
    sheet, key = make_audit_sheet(_cands(), tmp_path, n=40, seed=1, grader_id="g")
    res = analyse_audit(sheet, key)
    assert res.n_unlabelled == 40 and res.passes_kill_test() is None


def test_bad_label_is_rejected(tmp_path: Path) -> None:
    sheet, key = make_audit_sheet(_cands(), tmp_path, n=40, seed=1, grader_id="g")
    rows = list(csv.DictReader(sheet.open()))
    rows[0]["human_label"] = "maybe"
    with sheet.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="human_label"):
        analyse_audit(sheet, key)


def test_sensitivity_bound_formula() -> None:
    from sieve.eval.audit import RateEstimate

    e1 = RateEstimate(10, 100, 0.10, 0.05, 0.18)
    e2 = RateEstimate(2, 100, 0.02, 0.003, 0.07)
    point, worst = sensitivity_adjusted_risk(0.08, e1, e2)
    assert point == pytest.approx(0.08 * 0.9 + 0.92 * 0.02)
    assert worst == pytest.approx(0.08 * 0.95 + 0.92 * 0.07)
    assert worst > point


def test_kill_test_6_uses_the_upper_bound_not_the_point_estimate() -> None:
    """F-17: a 2% point estimate with a CP95 upper bound of 6% does not pass."""
    from sieve.eval.audit import AuditResult

    def res(hi: float) -> AuditResult:
        return AuditResult("g", 200, 0, {}, {}, {"all": (0.02, 0.005, hi)}, 0.0)

    assert res(0.06).passes_kill_test() is False
    assert res(0.029).passes_kill_test() is True

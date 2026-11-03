from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sieve.calibrate import (
    IsotonicCalibrator,
    PlattScaler,
    TemperatureScaler,
    abstention_stats,
    augrc,
    aurc,
    auroc,
    brier,
    ece_equal_mass,
    hallucination_rates,
    logit,
    psi,
    risk_coverage_curve,
    score_shift_pvalue,
)
from sieve.calibrate.metrics import expected_cum_errors


def test_risk_coverage_hand_computed() -> None:
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    wrong = np.array([False, True, False, True])
    cov, risk = risk_coverage_curve(conf, wrong)
    np.testing.assert_allclose(cov, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(risk, [0.0, 0.5, 1 / 3, 0.5])
    assert aurc(conf, wrong) == pytest.approx((0 + 0.5 + 1 / 3 + 0.5) / 4)
    assert augrc(conf, wrong) == pytest.approx((0 + 1 + 1 + 2) / 4 / 4)


def _brute_force_tie_average(conf: list[float], wrong: list[bool]) -> tuple[float, float]:
    """Average AURC / AUGRC over every permutation consistent with the ties."""
    n = len(conf)
    aurcs, augrcs = [], []
    for perm in itertools.permutations(range(n)):
        # keep only permutations that respect descending confidence (ties free)
        if any(conf[perm[i]] < conf[perm[i + 1]] for i in range(n - 1)):
            continue
        e = 0
        r_sum = g_sum = 0.0
        for k, idx in enumerate(perm, start=1):
            e += wrong[idx]
            r_sum += e / k
            g_sum += e / n
        aurcs.append(r_sum / n)
        augrcs.append(g_sum / n)
    return float(np.mean(aurcs)), float(np.mean(augrcs))


def test_ties_use_expected_ordering_matches_brute_force() -> None:
    conf = [0.9, 0.9, 0.5, 0.5, 0.5, 0.1]
    wrong = [True, False, True, True, False, False]
    a, g = _brute_force_tie_average(conf, wrong)
    assert aurc(conf, wrong) == pytest.approx(a)
    assert augrc(conf, wrong) == pytest.approx(g)


def test_constant_confidence_aurc_equals_error_rate() -> None:
    wrong = np.array([True, False, False, True, False])
    conf = np.full(5, 0.7)
    assert aurc(conf, wrong) == pytest.approx(wrong.mean())
    cov, risk = risk_coverage_curve(conf, wrong)
    assert cov.tolist() == [1.0] and risk[0] == pytest.approx(0.4)


@given(st.lists(st.tuples(st.floats(0, 1), st.booleans()), min_size=2, max_size=40))
def test_aurc_augrc_bounds_and_perfect_ranking_is_best(items: list[tuple[float, bool]]) -> None:
    conf = np.array([c for c, _ in items])
    wrong = np.array([w for _, w in items])
    a, g = aurc(conf, wrong), augrc(conf, wrong)
    assert 0.0 <= g <= a <= 1.0 + 1e-12
    perfect = np.where(wrong, 0.0, 1.0)  # all correct strictly above all wrong
    assert aurc(perfect, wrong) <= a + 1e-12
    assert augrc(perfect, wrong) <= g + 1e-12
    # AURC at full coverage equals the plain error rate
    cum = expected_cum_errors(conf, wrong)
    assert cum[-1] == pytest.approx(wrong.sum())


@given(st.lists(st.booleans(), min_size=4, max_size=30), st.data())
def test_swapping_wrong_below_correct_never_hurts(
    wrong_list: list[bool], data: st.DataObject
) -> None:
    n = len(wrong_list)
    conf = np.arange(n, 0, -1, dtype=float)  # distinct, descending
    wrong = np.array(wrong_list)
    i = data.draw(st.integers(0, n - 2))
    if not (wrong[i] and not wrong[i + 1]):
        return
    better = wrong.copy()
    better[i], better[i + 1] = False, True  # correct item now ranks above the wrong one
    assert aurc(conf, better) <= aurc(conf, wrong) + 1e-12
    assert augrc(conf, better) <= augrc(conf, wrong) + 1e-12


def test_ece_equal_mass_matches_reference_loop() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(103)
    y = (rng.random(103) < p).astype(float)
    order = np.argsort(p, kind="stable")
    ref = 0.0
    # independent reference: bin sizes by the np.array_split rule, written out by hand
    n, bins = 103, 15
    sizes = [n // bins + (1 if i < n % bins else 0) for i in range(bins)]
    pos = 0
    for sz in sizes:
        idx = order[pos : pos + sz]
        pos += sz
        ref += sz / n * abs(y[idx].mean() - p[idx].mean())
    assert ece_equal_mass(p, y, 15) == pytest.approx(ref)


def test_ece_hand_computed_and_perfect() -> None:
    # two bins of two: conf means 0.2 / 0.8, accuracy 0.0 / 1.0 -> ECE = .5*.2 + .5*.2 = 0.2
    assert ece_equal_mass([0.1, 0.3, 0.7, 0.9], [0, 0, 1, 1], n_bins=2) == pytest.approx(0.2)
    assert ece_equal_mass([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1], n_bins=2) == 0.0
    with pytest.raises(ValueError):
        ece_equal_mass([1.2], [1])


@given(st.lists(st.tuples(st.floats(0, 1), st.booleans()), min_size=1, max_size=60))
def test_ece_and_brier_bounds(items: list[tuple[float, bool]]) -> None:
    p = [x for x, _ in items]
    y = [float(b) for _, b in items]
    assert 0.0 <= ece_equal_mass(p, y) <= 1.0
    assert 0.0 <= brier(p, y) <= 1.0


def test_brier_hand() -> None:
    assert brier([0.9, 0.2], [1, 0]) == pytest.approx(0.025)


def test_auroc_matches_pair_counting() -> None:
    rng = np.random.default_rng(1)
    s = np.round(rng.random(60), 1)  # lots of ties
    y = rng.random(60) < 0.5
    pos, neg = s[y], s[~y]
    ref = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    assert auroc(s, y) == pytest.approx(ref)
    with pytest.raises(ValueError):
        auroc([0.1, 0.2], [True, True])


def test_abstention_stats_and_hallucination_rates() -> None:
    answered = np.array([True, True, True, False, False, False])
    wrong = np.array([False, True, False, True, True, False])
    st_ = abstention_stats(answered, wrong)
    assert st_.precision == pytest.approx(2 / 3)  # abstained & wrong / abstained
    assert st_.recall == pytest.approx(2 / 3)  # abstained & wrong / wrong
    h = hallucination_rates(answered, wrong)
    assert h.unconditional == pytest.approx(1 / 6)
    assert h.selective == pytest.approx(1 / 3)
    none = hallucination_rates(np.zeros(3, bool), np.ones(3, bool))
    assert none.selective is None and none.unconditional == 0.0
    assert abstention_stats(np.ones(3, bool), np.zeros(3, bool)).precision is None


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        aurc([0.5, math.nan], [True, False])
    with pytest.raises(ValueError):
        aurc([0.5], [True, False])
    with pytest.raises(ValueError):
        aurc([], [])


# --- calibrators ---------------------------------------------------------------------


def test_temperature_scaler_recovers_temperature() -> None:
    rng = np.random.default_rng(5)
    z = rng.normal(0, 3, 20000)
    y = rng.random(20000) < 1 / (1 + np.exp(-z / 2.0))
    t = TemperatureScaler().fit(z, y).temperature
    assert t == pytest.approx(2.0, rel=0.1)


def test_platt_recovers_slope_and_is_monotone() -> None:
    rng = np.random.default_rng(6)
    s = rng.normal(-1, 1, 20000)
    y = rng.random(20000) < 1 / (1 + np.exp(-(1.5 * s + 0.7)))
    m = PlattScaler().fit(s, y)
    assert m.a == pytest.approx(1.5, rel=0.1) and m.b == pytest.approx(0.7, abs=0.15)
    grid = np.linspace(-4, 2, 50)
    assert np.all(np.diff(m.predict(grid)) > 0)


def test_platt_finite_on_separable_data() -> None:
    m = PlattScaler().fit(np.array([0.0, 0.1, 0.9, 1.0]), np.array([0, 0, 1, 1], bool))
    assert np.isfinite(m.a) and np.isfinite(m.b) and m.a > 0


def test_isotonic_hand_example_and_monotone() -> None:
    m = IsotonicCalibrator().fit(np.array([1.0, 2, 3, 4]), np.array([1, 0, 0, 1], bool))
    np.testing.assert_allclose(m.predict(np.array([1.0, 2, 3, 4])), [1 / 3, 1 / 3, 1 / 3, 1.0])
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().predict(np.array([1.0]))


@given(st.lists(st.tuples(st.integers(0, 20), st.booleans()), min_size=2, max_size=60))
def test_isotonic_output_monotone_and_bounded(items: list[tuple[int, bool]]) -> None:
    x = np.array([float(a) for a, _ in items])
    y = np.array([b for _, b in items])
    m = IsotonicCalibrator().fit(x, y)
    g = np.linspace(-1, 21, 100)
    out = m.predict(g)
    assert np.all(np.diff(out) >= -1e-12) and float(np.min(out)) >= 0 and float(np.max(out)) <= 1


def test_logit_clips() -> None:
    assert np.all(np.isfinite(logit(np.array([0.0, 1.0]))))


# --- shift detectors -----------------------------------------------------------------


def test_shift_detectors() -> None:
    rng = np.random.default_rng(8)
    ref = rng.beta(4, 2, 2000)
    same = rng.beta(4, 2, 300)
    moved = rng.beta(2, 4, 300)
    assert score_shift_pvalue(ref, same) > 0.01
    assert score_shift_pvalue(ref, moved) < 1e-6
    assert psi(ref, same) < 0.1 < psi(ref, moved)

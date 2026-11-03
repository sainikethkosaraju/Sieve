"""Bootstrap / McNemar / TOST against independent reference computations."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from sieve.calibrate import auroc, ece_equal_mass
from sieve.eval import stats as st


def test_bootstrap_mean_matches_analytic_interval() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 2.0, size=400)
    iv = st.bootstrap(st.batch_mean(x), x.size, n_boot=4000, seed=1)
    se = x.std(ddof=1) / np.sqrt(x.size)
    assert iv.point == pytest.approx(x.mean())
    assert iv.lo == pytest.approx(x.mean() - 1.96 * se, abs=0.03)
    assert iv.hi == pytest.approx(x.mean() + 1.96 * se, abs=0.03)


def test_bootstrap_is_deterministic_and_seeded() -> None:
    x = np.random.default_rng(1).normal(size=100)
    a = st.bootstrap(st.batch_mean(x), 100, n_boot=500, seed=5)
    b = st.bootstrap(st.batch_mean(x), 100, n_boot=500, seed=5)
    c = st.bootstrap(st.batch_mean(x), 100, n_boot=500, seed=6)
    assert a == b and a != c


def test_percentile_ci_equals_manual_quantiles() -> None:
    x = np.random.default_rng(2).normal(size=60)
    n_boot, seed = 1000, 9
    idx = np.concatenate(st.resample_indices(60, n_boot, seed))
    manual = np.sort(x[idx].mean(axis=1))
    lo, hi = np.quantile(manual, [0.025, 0.975])
    iv = st.bootstrap(st.batch_mean(x), 60, n_boot=n_boot, seed=seed)
    assert (iv.lo, iv.hi) == pytest.approx((lo, hi))


def test_paired_difference_is_tighter_than_unpaired() -> None:
    rng = np.random.default_rng(3)
    shared = rng.normal(size=300)
    a = shared + rng.normal(0, 0.1, size=300) + 0.2
    b = shared + rng.normal(0, 0.1, size=300)
    paired = st.paired_difference(st.batch_mean(a), st.batch_mean(b), 300, n_boot=2000, seed=0)
    ia = st.bootstrap(st.batch_mean(a), 300, n_boot=2000, seed=0)
    ib = st.bootstrap(st.batch_mean(b), 300, n_boot=2000, seed=1)
    assert paired.lo > 0  # difference clearly > 0
    assert (paired.hi - paired.lo) < 0.5 * ((ia.hi - ia.lo) + (ib.hi - ib.lo))
    assert paired.point == pytest.approx(0.2, abs=0.05)


def test_batch_ratio_handles_empty_denominator() -> None:
    num = np.array([0, 1, 0], dtype=bool)
    den = np.array([0, 1, 1], dtype=bool)
    out = st.batch_ratio(num, den)(np.array([[0, 0, 0], [1, 1, 2], [1, 2, 2]]))
    assert np.isnan(out[0])  # empty denominator -> NaN (dropped by the bootstrap)
    assert out[1] == pytest.approx(2 / 3)
    assert out[2] == pytest.approx(1 / 3)


def _auroc_bruteforce(s: np.ndarray, y: np.ndarray) -> float:
    pos, neg = s[y], s[~y]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (pos.size * neg.size))


def test_batch_auroc_matches_bruteforce_and_phase_a() -> None:
    rng = np.random.default_rng(4)
    s = np.round(rng.normal(size=80), 1)  # ties on purpose
    y = rng.random(80) < 0.6
    f = st.batch_auroc(s, y)
    ident = f(np.arange(80)[None, :])[0]
    assert ident == pytest.approx(auroc(s, y)) and ident == pytest.approx(_auroc_bruteforce(s, y))
    idx = rng.integers(0, 80, size=(7, 80))
    got = f(idx)
    for r in range(7):
        assert got[r] == pytest.approx(_auroc_bruteforce(s[idx[r]], y[idx[r]]))


def test_batch_ece_matches_phase_a() -> None:
    rng = np.random.default_rng(5)
    p = rng.random(203)
    y = rng.random(203) < p
    f = st.batch_ece(p, y)
    assert f(np.arange(203)[None, :])[0] == pytest.approx(ece_equal_mass(p, y))
    idx = rng.integers(0, 203, size=(4, 203))
    for r in range(4):
        assert f(idx)[r] == pytest.approx(ece_equal_mass(p[idx[r]], y[idx[r]]))


def test_mcnemar_exact_matches_scipy_binomtest() -> None:
    a = np.array([1] * 20 + [1] * 12 + [0] * 3 + [0] * 15, dtype=bool)
    b = np.array([1] * 20 + [0] * 12 + [1] * 3 + [0] * 15, dtype=bool)
    r = st.mcnemar(a, b)
    assert (r.n10, r.n01) == (12, 3)
    assert r.p_exact == pytest.approx(sps.binomtest(3, 15, 0.5).pvalue)
    assert st.mcnemar(a, a).p_exact == 1.0
    with pytest.raises(ValueError):
        st.mcnemar(a, b[:-1])


def test_tost_matches_two_one_sided_ttests() -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(0.003, 0.05, size=200)
    y = rng.normal(0.0, 0.05, size=200)
    margin = 0.02
    r = st.tost_paired_t(x, y, margin)
    d = x - y
    p_low = sps.ttest_1samp(d, -margin, alternative="greater").pvalue
    p_up = sps.ttest_1samp(d, margin, alternative="less").pvalue
    assert r.p_lower == pytest.approx(p_low) and r.p_upper == pytest.approx(p_up)
    assert r.p_tost == pytest.approx(max(p_low, p_up))
    assert r.equivalent is (r.p_tost < 0.05)


def test_tost_rejects_a_real_difference() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(0.05, 0.02, size=100)
    y = rng.normal(0.0, 0.02, size=100)
    assert not st.tost_paired_t(x, y, 0.02).equivalent


def test_tost_bootstrap_inclusion_rule() -> None:
    assert st.tost_bootstrap(st.Interval(0.0, -0.01, 0.015, 1, 0.9), 0.02)
    assert not st.tost_bootstrap(st.Interval(0.0, -0.01, 0.025, 1, 0.9), 0.02)

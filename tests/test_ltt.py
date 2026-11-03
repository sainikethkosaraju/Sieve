from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import binom

from sieve.calibrate import (
    NEVER,
    CalTable,
    FeasibilityOutcome,
    binom_pvalue,
    cp_upper,
    evaluate_policy,
    fixed_sequence_ltt,
    hb_pvalue,
    make_path,
    min_certifiable_alpha,
    pick_min_cost,
    policy_cost,
)


def table(
    s: Sequence[float],
    ll: Sequence[float],
    sw: Sequence[float],
    lw: Sequence[float],
    cs: Sequence[float] | None = None,
    cl: Sequence[float] | None = None,
) -> CalTable:
    return CalTable(
        np.array(s, float),
        np.array(ll, float),
        np.array(sw, bool),
        np.array(lw, bool),
        None if cs is None else np.array(cs, float),
        None if cl is None else np.array(cl, float),
    )


def test_pvalue_hand_values() -> None:
    assert binom_pvalue(0, 10, 0.1) == pytest.approx(0.9**10)
    assert binom_pvalue(1, 4, 0.5) == pytest.approx(5 / 16)
    assert binom_pvalue(3, 0, 0.1) == 1.0


def test_cp_upper_hand_and_roundtrip() -> None:
    assert cp_upper(0, 20, 0.05) == pytest.approx(1 - 0.05 ** (1 / 20))
    assert cp_upper(20, 20, 0.05) == 1.0 and cp_upper(0, 0, 0.05) == 1.0
    for k, n in [(1, 10), (5, 50), (0, 7)]:
        ub = cp_upper(k, n, 0.1)
        assert binom.cdf(k, n, ub) == pytest.approx(0.1, abs=1e-9)  # exact inversion


def test_min_certifiable_alpha_is_the_boundary() -> None:
    n, delta = 40, 0.1
    a = min_certifiable_alpha(n, delta)
    assert binom_pvalue(0, n, a) == pytest.approx(delta)
    assert a == pytest.approx(math.log(1 / delta) / n, rel=0.1)  # ln(1/d)/n heuristic
    assert min_certifiable_alpha(0, 0.1) == 1.0


@given(st.integers(1, 200), st.floats(0.01, 0.5))
def test_pvalue_monotone_in_errors(n: int, alpha: float) -> None:
    ps = [binom_pvalue(k, n, alpha) for k in range(n + 1)]
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert all(a <= b + 1e-12 for a, b in itertools.pairwise(ps))


@settings(max_examples=60)
@given(st.integers(1, 300), st.floats(0.02, 0.4), st.floats(0, 1))
def test_hb_is_valid_and_not_tighter_than_exact_for_binary(n: int, alpha: float, f: float) -> None:
    k = int(f * n)
    hb = hb_pvalue(k / n, n, alpha)
    assert 0.0 <= hb <= 1.0
    # HB is a valid but looser bound than the exact binomial test for {0,1} losses
    assert hb >= binom_pvalue(k, n, alpha) - 1e-9


def test_evaluate_policy_hand_example() -> None:
    t = table(
        s=[0.9, 0.4, 0.4, 0.4],
        ll=[0.5, 0.9, 0.2, math.nan],
        sw=[False, True, True, True],
        lw=[True, False, True, False],
    )
    o = evaluate_policy(t, 0.8, 0.8)
    assert o.by_s.tolist() == [True, False, False, False]
    assert o.by_l.tolist() == [False, True, False, False]  # NaN l_conf never answers
    assert o.answered.tolist() == [True, True, False, False]
    assert o.wrong.tolist() == [False, False, False, False]
    assert o.escalated.tolist() == [False, True, True, True]
    assert not evaluate_policy(t, NEVER, NEVER).answered.any()


def test_nan_s_conf_never_answers_at_s() -> None:
    t = table([math.nan], [0.1], [False], [True])
    assert not evaluate_policy(t, -1e9, NEVER).answered.any()


def test_table_validation() -> None:
    with pytest.raises(ValueError):
        table([0.5], [0.5, 0.5], [False], [False])
    with pytest.raises(ValueError):
        table([math.inf], [0.5], [False], [False])
    with pytest.raises(ValueError):
        table([], [], [], [])


def _perfect_table(n: int = 200, n_wrong_low: int = 100) -> CalTable:
    """High confidence items always right; low confidence items always wrong."""
    s = np.concatenate([np.full(n, 0.9), np.full(n_wrong_low, 0.1)])
    wrong = np.concatenate([np.zeros(n, bool), np.ones(n_wrong_low, bool)])
    return CalTable(s, s.copy(), wrong, wrong.copy())


def test_ltt_certifies_clean_policy_and_rejects_dirty_one() -> None:
    t = _perfect_table()
    res = fixed_sequence_ltt(t, [(0.5, NEVER), (0.05, NEVER)], alpha=0.05, delta=0.1)
    assert res.outcome is FeasibilityOutcome.FEASIBLE
    assert [(p.t1, p.t2) for p in res.valid] == [(0.5, NEVER)]
    assert res.valid[0].n_answered == 200 and res.valid[0].k_wrong == 0
    # second point answers the 100 always-wrong items too: risk 1/3 > 0.05, not certified
    assert res.tested[1].rejected is False and res.tested[1].risk_hat == pytest.approx(1 / 3)


def test_fixed_sequence_stops_at_first_failure_bonferroni_does_not() -> None:
    t = _perfect_table()
    path = [(0.5, NEVER), (0.05, NEVER), (0.6, NEVER)]  # good, bad, good again
    fs = fixed_sequence_ltt(t, path, alpha=0.05, delta=0.1)
    assert len(fs.tested) == 2  # never looked at the third point
    assert [p.t1 for p in fs.valid] == [0.5]
    bf = fixed_sequence_ltt(t, path, alpha=0.05, delta=0.1, method="bonferroni")
    assert [p.t1 for p in bf.valid] == [0.5, 0.6]
    assert bf.n_grid == 3


def test_empty_feasible_set_outcome() -> None:
    t = _perfect_table(n=20, n_wrong_low=100)  # too few clean items to certify alpha=0.01
    res = fixed_sequence_ltt(t, [(0.5, NEVER)], alpha=0.01, delta=0.1)
    assert res.outcome is FeasibilityOutcome.EMPTY and res.empty and not res.valid
    assert res.feasibility_floor > 0.01  # floor explains why: 1-0.1**(1/20) = 0.109
    assert res.feasibility_floor == pytest.approx(1 - 0.1 ** (1 / 20))


def test_zero_answered_policy_is_not_certified() -> None:
    t = _perfect_table()
    res = fixed_sequence_ltt(t, [(NEVER, NEVER)], alpha=0.5, delta=0.1)
    assert res.empty and res.tested[0].p_value == 1.0


def test_unconditional_definition_uses_total_n() -> None:
    t = _perfect_table(n=100, n_wrong_low=100)
    r = fixed_sequence_ltt(t, [(0.05, NEVER)], alpha=0.6, delta=0.1, definition="unconditional")
    assert r.tested[0].n == 200 and r.tested[0].risk_hat == pytest.approx(0.5)
    assert r.valid  # P(wrong & answered)=0.5 <= 0.6 certifiable with n=200


def test_hb_option_runs() -> None:
    t = _perfect_table()
    r = fixed_sequence_ltt(t, [(0.5, NEVER)], alpha=0.1, delta=0.1, pvalue="hb")
    assert r.valid


def test_ltt_argument_validation() -> None:
    t = _perfect_table()
    with pytest.raises(ValueError):
        fixed_sequence_ltt(t, [(0.5, 0.5)], alpha=0.0, delta=0.1)
    with pytest.raises(ValueError):
        fixed_sequence_ltt(t, [], alpha=0.1, delta=0.1)


def test_make_path_orders_by_a_coverage_and_filters() -> None:
    ta = _perfect_table()
    path = make_path([0.05, 0.5, NEVER], [NEVER], ta, frontier=False)
    assert path == [(0.5, NEVER), (0.05, NEVER)]  # NEVER/NEVER has zero coverage: dropped
    # frontier: (0.05, NEVER) has higher coverage AND higher risk, so it is still kept;
    # a dominated point is removed (same coverage, higher risk)
    t = table([0.9, 0.9, 0.2, 0.2], [0.9, 0.2, 0.9, 0.2], [0, 0, 1, 1], [0, 1, 0, 1])
    pts = make_path([0.5], [0.5, 0.95], t, frontier=True)
    assert (0.5, 0.5) in pts
    path_hi = make_path([0.5, 0.95], [NEVER], t, min_coverage=0.9)
    assert path_hi == []


def test_cost_pick_hand_computed() -> None:
    # 4 items, S conf: 0.9,0.9,0.1,0.1 ; L conf all 0.9. cost_s=1, cost_l=5.
    c = table([0.9, 0.9, 0.1, 0.1], [0.9] * 4, [0, 0, 1, 1], [0, 0, 0, 0], cs=[1] * 4, cl=[5] * 4)
    cov, total, esc = policy_cost(c, 0.5, 0.5)
    assert (cov, total, esc) == (1.0, 4 + 10, 0.5)
    cov, total, esc = policy_cost(c, NEVER, 0.5)  # L only: S still runs (buffered cascade)
    assert (cov, total, esc) == (1.0, 4 + 20, 1.0)
    res = fixed_sequence_ltt(
        c, [(0.5, 0.5), (NEVER, 0.5)], alpha=0.4, delta=0.5, method="bonferroni"
    )
    assert len(res.valid) == 2
    pick = pick_min_cost(res, c)
    assert pick is not None and pick.point.t1 == 0.5
    assert pick.cost_per_answered == pytest.approx(14 / 4)
    assert pick_min_cost(res, c, min_coverage=1.01) is None
    with pytest.raises(ValueError):
        policy_cost(table([0.1], [0.1], [0], [0]), 0.0, 0.0)

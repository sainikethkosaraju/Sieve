"""Simulation of the Learn-then-Test guarantee (the Phase A 'mock backend' reproduction).

World: exchangeable items. Tier S reports a confidence ``s ~ Beta`` and is wrong with
probability ``1 - s`` (perfectly calibrated); tier L likewise with its own Beta. The
*true* selective / unconditional risk of every grid policy is computed on a large
population (2e5 draws, standard error < 0.001 at the risks used), so a "violation" is
an event about the real world, not about the calibration sample.

Each replicate draws fresh splits A (path), B (test of H0) and records whether ANY policy in
the certified set has true risk > alpha. The guarantee says that happens w.p. <= delta.

Tolerance. The replicates are i.i.d. (fixed seeds, so the test is deterministic, not
flaky). Observed violation frequency ~ Binomial(R, p) with p <= delta. We assert
``freq <= delta + 3 * sqrt(delta * (1 - delta) / R)``: a 3-sigma allowance for Monte
Carlo noise at R = 500, i.e. 0.1 + 0.040 = 0.140. A procedure that is exact (not
conservative) would still pass this with probability > 0.998. The procedure here is
conservative (discrete binomial test, fixed-sequence stopping), so the observed
frequency is far below delta; we also assert it is not vacuous (non-empty sets).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sieve.calibrate import (
    NEVER,
    CalTable,
    FeasibilityOutcome,
    evaluate_policy,
    fixed_sequence_ltt,
    make_path,
    pick_min_cost,
)

pytestmark = pytest.mark.slow

ALPHA = 0.10
DELTA = 0.10
R = 500
N_A, N_B, N_C = 600, 1500, 600
T1_GRID = [*np.linspace(0.5, 0.98, 13).tolist(), NEVER]
T2_GRID = [*np.linspace(0.5, 0.98, 13).tolist(), NEVER]


def draw(
    rng: np.random.Generator,
    n: int,
    *,
    s_ab: tuple[float, float] = (2.0, 1.2),
    l_ab: tuple[float, float] = (4.0, 1.0),
    shift: float = 0.0,
    with_cost: bool = False,
) -> CalTable:
    """Calibrated world; ``shift`` lowers true P(correct) by that amount (miscalibration)."""
    s = rng.beta(*s_ab, size=n)
    l_ = rng.beta(*l_ab, size=n)
    s_wrong = rng.random(n) >= np.clip(s - shift, 0.0, 1.0)
    l_wrong = rng.random(n) >= np.clip(l_ - shift, 0.0, 1.0)
    return CalTable(
        s_conf=s,
        l_conf=l_,
        s_wrong=s_wrong,
        l_wrong=l_wrong,
        cost_s=np.full(n, 1.0) if with_cost else None,
        cost_l=np.full(n, 4.0) if with_cost else None,
    )


@pytest.fixture(scope="module")
def population() -> CalTable:
    return draw(np.random.default_rng(12345), 200_000)


@pytest.fixture(scope="module")
def shifted_population() -> CalTable:
    return draw(np.random.default_rng(54321), 200_000, shift=0.12)


def true_risk(pop: CalTable, t1: float, t2: float) -> float:
    out = evaluate_policy(pop, t1, t2)
    n_ans = int(out.answered.sum())
    return float(out.wrong.sum()) / n_ans if n_ans else 0.0


def run_replicates(
    pop: CalTable, *, method: str, definition: str, seed: int
) -> tuple[float, float, float]:
    """Return (violation frequency, fraction non-empty, mean true coverage of chosen)."""
    rng = np.random.default_rng(seed)
    # Cache true risk of every grid point once: replicates only differ in which are certified.
    truth: dict[tuple[float, float], float] = {}
    for a in T1_GRID:
        for b in T2_GRID:
            out = evaluate_policy(pop, a, b)
            k = float(out.wrong.sum())
            if definition == "selective":
                n_ans = float(out.answered.sum())
                truth[(a, b)] = k / n_ans if n_ans else 0.0
            else:
                truth[(a, b)] = k / pop.n
    violations = nonempty = 0
    cov: list[float] = []
    for _ in range(R):
        ta, tb = draw(rng, N_A), draw(rng, N_B)
        path = make_path(T1_GRID, T2_GRID, ta, definition=definition)  # type: ignore[arg-type]
        res = fixed_sequence_ltt(
            tb,
            path,
            alpha=ALPHA,
            delta=DELTA,
            definition=definition,  # type: ignore[arg-type]
            method=method,  # type: ignore[arg-type]
        )
        if not res.empty:
            nonempty += 1
            cov.extend(float(evaluate_policy(pop, p.t1, p.t2).answered.mean()) for p in res.valid)
        if any(truth[(p.t1, p.t2)] > ALPHA for p in res.valid):
            violations += 1
    return violations / R, nonempty / R, float(np.mean(cov)) if cov else 0.0


TOL = 3 * math.sqrt(DELTA * (1 - DELTA) / R)


@pytest.mark.parametrize(
    ("method", "definition"),
    [
        ("fixed_sequence", "selective"),
        ("bonferroni", "selective"),
        ("fixed_sequence", "unconditional"),
    ],
)
def test_ltt_violation_frequency_below_delta(
    population: CalTable, method: str, definition: str
) -> None:
    freq, nonempty, mean_cov = run_replicates(
        population, method=method, definition=definition, seed=2026
    )
    print(
        f"[{method}/{definition}] violation={freq:.3f} (limit {DELTA + TOL:.3f}) "
        f"nonempty={nonempty:.3f} mean_coverage={mean_cov:.3f}"
    )
    assert freq <= DELTA + TOL
    # non-vacuous: the certified set is usually non-empty and answers a real fraction
    assert nonempty >= 0.9
    assert mean_cov >= 0.2


def test_end_to_end_chosen_policy_holds_on_fresh_test_split(population: CalTable) -> None:
    """Full recipe: path on A, LTT on B, cost pick on A', realised risk on test C."""
    rng = np.random.default_rng(7)
    viol = 0
    for _ in range(R):
        ta, tb, tcost = (draw(rng, n, with_cost=True) for n in (N_A, N_B, N_A))
        res = fixed_sequence_ltt(tb, make_path(T1_GRID, T2_GRID, ta), alpha=ALPHA, delta=DELTA)
        pick = pick_min_cost(res, tcost, min_coverage=0.2) if not res.empty else None
        if pick is None:
            continue
        # realised risk on the population stands in for the infinite test set
        if true_risk(population, pick.point.t1, pick.point.t2) > ALPHA:
            viol += 1
    freq = viol / R
    print(f"[end-to-end] violation={freq:.3f}")
    assert freq <= DELTA + TOL


def test_guarantee_fails_detectably_under_shift(
    population: CalTable, shifted_population: CalTable
) -> None:
    """Calibrate in the calibrated world, deploy where P(correct) is 0.12 lower.

    The certificate is void; the realised risk exceeds alpha for most replicates. The
    point is that the violation is *visible* in a graded sample, and that the score
    histogram alone (covariate view) would not have revealed it: scores are unchanged.
    """
    rng = np.random.default_rng(99)
    viol = tested = 0
    for _ in range(200):
        ta, tb = draw(rng, N_A), draw(rng, N_B)
        res = fixed_sequence_ltt(tb, make_path(T1_GRID, T2_GRID, ta), alpha=ALPHA, delta=DELTA)
        if res.empty:
            continue
        tested += 1
        best = max(res.valid, key=lambda p: p.n_answered)
        if true_risk(shifted_population, best.t1, best.t2) > ALPHA:
            viol += 1
    assert tested >= 150
    frac = viol / tested
    print(f"[shift] violation fraction under shift={frac:.3f}")
    assert frac >= 0.8  # vs <= 0.14 in the exchangeable world
    # sanity: the same procedure without shift stays below the limit (see tests above)
    assert frac > DELTA + TOL


def test_empty_feasible_set_is_reported_not_hidden() -> None:
    rng = np.random.default_rng(3)
    # A very weak model: confidence carries almost no information, error ~ 45%.
    n = 1500
    conf = rng.random(n)
    t = CalTable(conf, conf, rng.random(n) < 0.45, rng.random(n) < 0.45)
    path = make_path(T1_GRID, T2_GRID, t)
    res = fixed_sequence_ltt(t, path, alpha=0.05, delta=0.05)
    assert res.outcome is FeasibilityOutcome.EMPTY and res.empty and res.valid == ()
    assert pick_min_cost(res, t) is None

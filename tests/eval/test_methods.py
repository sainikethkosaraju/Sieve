"""Baseline builders: query-only router, oracles, primary-signal choice."""

from __future__ import annotations

import numpy as np

from sieve.eval.data import PairSlice, View
from sieve.eval.evaluate import oracle_abstainer, oracle_router
from sieve.eval.methods import (
    MethodDef,
    TfidfLogisticRouter,
    choose_primary,
    default_methods,
    fit_method,
    policy_item_costs,
)


def _view(n: int, rng: np.random.Generator, good: str) -> View:
    correct = rng.random(n) < 0.6
    sig = {
        s: rng.normal(size=n)
        for s in (
            "max_prob",
            "mean_logprob",
            "neg_mean_entropy",
            "mean_margin",
            "agreement",
            "min_logprob",
            "first_margin",
            "neg_first_entropy",
            "neg_cluster_entropy",
        )
    }
    sig[good] = correct + rng.normal(0, 0.3, size=n)  # informative signal
    return View(
        sig,
        ~correct,
        correct,
        np.ones(n, bool),
        np.ones(n),
        np.zeros(n),
        np.zeros(n, bool),
        [""] * n,
        ["correct"] * n,
    )


def _slice(n: int = 400, seed: int = 0, good: str = "mean_margin") -> PairSlice:
    rng = np.random.default_rng(seed)
    return PairSlice(
        tuple(f"i{i}" for i in range(n)),
        [f"q{i}" for i in range(n)],
        ["d"] * n,
        ["s"] * n,
        _view(n, rng, good),
        _view(n, rng, good),
        "seconds",
    )


def test_choose_primary_picks_the_informative_signal() -> None:
    assert choose_primary(_slice(good="mean_margin")) == "mean_margin"
    assert choose_primary(_slice(good="max_prob", seed=3)) == "max_prob"


def test_method_set_matches_the_preregistration() -> None:
    names = {m.name for m in default_methods()}
    assert {
        "S_only",
        "L_only",
        "gate:max_softmax",
        "gate:mean_logprob",
        "gate:entropy",
        "gate:margin",
        "random_router",
        "query_router",
        "self_consistency",
    } <= names
    assert "self_consistency" not in {m.name for m in default_methods(include_sc=False)}


def test_tfidf_router_learns_a_planted_word_and_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    qs, y = [], []
    for i in range(600):
        hard = rng.random() < 0.5
        qs.append(f"question {i % 7} about {'obscure' if hard else 'famous'} thing {i % 5}")
        y.append(not hard)
    r1 = TfidfLogisticRouter().fit(qs, np.array(y))
    r2 = TfidfLogisticRouter().fit(qs, np.array(y))
    p = r1.predict(["a question about famous thing", "a question about obscure thing"])
    assert p[0] > 0.7 > 0.3 > p[1]
    assert np.array_equal(r1.predict(qs), r2.predict(qs))


def test_random_router_is_seeded_and_uninformative() -> None:
    sl = _slice()
    defn = MethodDef("random_router", "random", "primary", pre_route=True, deployable=False)
    a = fit_method(defn, "mean_margin", sl, seed=1).scores(sl)[0]
    b = fit_method(defn, "mean_margin", sl, seed=1).scores(sl)[0]
    c = fit_method(defn, "mean_margin", sl, seed=2).scores(sl)[0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert abs(np.corrcoef(a, sl.s.correct.astype(float))[0, 1]) < 0.15


def test_cost_model_charges_wasted_s_for_gates_but_not_for_routers() -> None:
    sl = _slice(n=10)
    sl.s.time_s[:] = 1.0
    sl.l.time_s[:] = 5.0
    by_s = np.array([True] * 4 + [False] * 6)
    gate = fit_method(MethodDef("g", "gate", "mean_margin"), "mean_margin", sl, 0)
    router = fit_method(MethodDef("r", "random", "primary", pre_route=True), "mean_margin", sl, 0)
    assert policy_item_costs(gate, sl, by_s).sum() == 10 * 1.0 + 6 * 5.0  # S ran on everything
    assert (
        policy_item_costs(router, sl, by_s).sum() == 4 * 1.0 + 6 * 5.0
    )  # S skipped on escalations
    s_only = fit_method(MethodDef("s", "s_only", "primary"), "mean_margin", sl, 0)
    l_only = fit_method(MethodDef("l", "l_only", "primary"), "mean_margin", sl, 0)
    assert policy_item_costs(s_only, sl, by_s).sum() == 10.0
    assert policy_item_costs(l_only, sl, by_s).sum() == 50.0


def test_oracles_are_perfect_by_construction() -> None:
    sl = _slice()
    router = oracle_router(sl, alpha=0.1).summary()
    abst = oracle_abstainer(sl, alpha=0.1).summary()
    both_wrong = float((sl.s.wrong & sl.l.wrong).mean())
    assert router["coverage"] == 1.0 and router["risk"] == both_wrong
    assert abst["risk"] == 0.0 and abst["coverage"] == float(sl.l.correct.mean())
    assert router["certified"] is False and abst["certified"] is False


def test_not_attempted_is_not_counted_correct_in_accuracy_on_answered() -> None:
    """SimpleQA ``not_attempted`` is neither wrong nor correct."""
    import numpy as np

    from sieve.eval.evaluate import PolicyEval

    answered = np.array([True, True, True, True])
    wrong = np.array([False, False, True, False])
    correct = np.array([True, False, False, False])  # item 1,3: not_attempted
    z = np.zeros(4, dtype=bool)
    ev = PolicyEval(
        "x",
        ("a", "b", "c", "d"),
        ["", "", "", ""],
        answered,
        wrong,
        z,
        np.ones(4),
        0.1,
        correct=correct,
    )
    s = ev.summary()
    assert s["accuracy_on_answered"] == 0.25 and s["risk"] == 0.25

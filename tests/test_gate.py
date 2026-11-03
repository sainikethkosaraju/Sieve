from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sieve.calibrate import CalTable, evaluate_policy
from sieve.calibrate.ltt import NEVER
from sieve.calibrate.store import ArtifactIntegrityError
from sieve.gate import Gate, decide
from sieve.signals import compute_signals, logprob_signals
from sieve.types import (
    Action,
    BackendInfo,
    Completion,
    DefectCode,
    ReasonCode,
    SamplingParams,
    Signals,
)
from tests.helpers import GREEDY, SAMPLED, artifact, backend, completion, tiers

S_BACKEND = backend("small")
L_BACKEND = backend("large", model_hash="h2")
ART = artifact(t1=-0.5, t2=-0.3)  # answer if mean logprob >= threshold


def sig(mean_lp: float, b: BackendInfo = S_BACKEND, **kw: object) -> Signals:
    return Signals(backend=b, sampling=GREEDY, n_answer_tokens=3, mean_logprob=mean_lp, **kw)  # type: ignore[arg-type]


def test_basic_decisions() -> None:
    d = decide(sig(-0.2), ART, 1)
    assert (d.action, d.reason, d.score, d.threshold) == (
        Action.ANSWER,
        ReasonCode.CONFIDENT,
        -0.2,
        -0.5,
    )
    assert d.artifact_id == ART.artifact_id and not d.fail_closed
    d = decide(sig(-0.5), ART, 1)  # boundary answers (>=)
    assert d.action is Action.ANSWER
    d = decide(sig(-0.8), ART, 1)
    assert (d.action, d.reason) == (Action.ESCALATE, ReasonCode.LOW_CONFIDENCE)
    assert decide(sig(-0.1, L_BACKEND), ART, 2).action is Action.ANSWER
    d = decide(sig(-0.9, L_BACKEND), ART, 2)
    assert (d.action, d.reason) == (Action.ABSTAIN, ReasonCode.LOW_CONFIDENCE)


def test_uncertified_tiers() -> None:
    art = artifact(t1=None, t2=None)
    d1 = decide(sig(0.0), art, 1)
    d2 = decide(sig(0.0, L_BACKEND), art, 2)
    assert (d1.action, d1.reason) == (Action.ESCALATE, ReasonCode.TIER_NOT_CERTIFIED)
    assert (d2.action, d2.reason) == (Action.ABSTAIN, ReasonCode.TIER_NOT_CERTIFIED)


def test_bad_tier() -> None:
    with pytest.raises(ValueError):
        decide(sig(-0.1), ART, 3)


# ---- fail-closed: every defect class -------------------------------------------------

DEFECT_COMPLETIONS = {
    "nan": lambda c: replace(c, tokens=(replace(c.tokens[0], logprob=math.nan), *c.tokens[1:])),
    "positive": lambda c: replace(c, tokens=(replace(c.tokens[0], logprob=0.5), *c.tokens[1:])),
    "missing": lambda c: replace(c, tokens=None),
    "short": lambda c: replace(c, n_gen=c.n_gen + 3),
    "utf8": lambda c: replace(c, text=c.text + "\U0001f600"),
    "truncated": lambda c: replace(c, finish="length"),
    "error": lambda c: replace(c, finish="error"),
}


@pytest.mark.parametrize("name", sorted(DEFECT_COMPLETIONS))
@pytest.mark.parametrize("tier", [1, 2])
def test_logprob_defects_fail_closed(name: str, tier: int) -> None:
    b = S_BACKEND if tier == 1 else L_BACKEND
    c = DEFECT_COMPLETIONS[name](completion([0.99, 0.99, 0.99]))  # would be very confident
    s = logprob_signals(c, b)
    assert s.defects
    d = decide(s, ART, tier)
    assert d.action is (Action.ESCALATE if tier == 1 else Action.ABSTAIN)
    assert d.reason is ReasonCode.LOGPROB_DEFECT and d.defects == s.defects and d.fail_closed


def test_defect_with_forged_high_value_still_fails_closed() -> None:
    # even if some upstream code left a (stale) number in place, the defect list wins
    s = sig(-0.01, defects=(DefectCode.UTF8_GAP,))
    assert decide(s, ART, 1).reason is ReasonCode.LOGPROB_DEFECT


def test_missing_or_nonfinite_score_fails_closed() -> None:
    assert decide(sig(math.nan), ART, 1).reason is ReasonCode.SIGNAL_UNAVAILABLE
    assert decide(sig(math.inf), ART, 1).action is Action.ESCALATE
    none = Signals(backend=S_BACKEND, sampling=GREEDY, n_answer_tokens=0)
    assert decide(none, ART, 2).action is Action.ABSTAIN


def test_missing_top_logprobs_only_blocks_top_dependent_signals() -> None:
    s = sig(-0.1, defects=(DefectCode.TOP_LOGPROBS_MISSING,))
    assert decide(s, ART, 1).action is Action.ANSWER  # mean_logprob needs no top-k
    art_margin = artifact(signal1="mean_margin")
    s2 = replace(s, mean_margin=5.0)
    assert decide(s2, art_margin, 1).reason is ReasonCode.LOGPROB_DEFECT


@pytest.mark.parametrize(
    "field",
    ["model_id", "model_hash", "quant", "kv_quant", "build", "template_hash"],
)
@pytest.mark.parametrize("tier", [1, 2])
def test_identity_mismatch_every_field(field: str, tier: int) -> None:
    b = S_BACKEND if tier == 1 else L_BACKEND
    wrong = replace(b, **{field: "something-else"})
    d = decide(sig(-0.01, wrong), ART, tier)
    assert d.reason is ReasonCode.IDENTITY_MISMATCH
    assert d.action is (Action.ESCALATE if tier == 1 else Action.ABSTAIN)


def test_identity_empty_field_never_matches() -> None:
    assert decide(sig(-0.01, replace(S_BACKEND, build="")), ART, 1).action is Action.ESCALATE


def test_small_model_signals_cannot_pass_as_large() -> None:
    assert decide(sig(-0.01, S_BACKEND), ART, 2).reason is ReasonCode.IDENTITY_MISMATCH


@pytest.mark.parametrize(
    "bad",
    [
        SamplingParams(temperature=0.2),
        SamplingParams(top_p=0.9),
        SamplingParams(top_k=40),
        SamplingParams(min_p=0.05),
        SamplingParams(logit_bias=((1, 5.0),)),
        SamplingParams(grammar="root ::= [a-z]"),
        SamplingParams(post_sampling_probs=True),
    ],
)
def test_sampling_mismatch(bad: SamplingParams) -> None:
    s = replace(sig(-0.01), sampling=bad)
    d = decide(s, ART, 1)
    assert (d.action, d.reason) == (Action.ESCALATE, ReasonCode.SAMPLING_MISMATCH)


def test_seed_and_max_tokens_do_not_count_as_mismatch() -> None:
    s = replace(sig(-0.01), sampling=SamplingParams(seed=77, max_tokens=999))
    assert decide(s, ART, 1).action is Action.ANSWER


def test_consistency_signal_checks_sample_config() -> None:
    art = artifact(signal1="agreement")
    t1 = replace(art.tiers[0], n_samples=5, consistency_sampling=SAMPLED)
    from sieve.calibrate.store import compute_content_hash

    art = replace(art, tiers=(t1, art.tiers[1]), content_hash="")
    art = replace(art, content_hash=compute_content_hash(art))
    base = Signals(
        backend=S_BACKEND,
        sampling=GREEDY,
        n_answer_tokens=1,
        agreement=1.0,
        n_samples=5,
        consistency_sampling=SAMPLED,
    )
    assert decide(base, art, 1).action is Action.ANSWER  # thresholds -0.5: any agreement passes
    assert decide(replace(base, n_samples=4), art, 1).reason is ReasonCode.SAMPLING_MISMATCH
    assert decide(replace(base, consistency_sampling=None), art, 1).reason is (
        ReasonCode.SAMPLING_MISMATCH
    )
    greedy_samples = replace(base, consistency_sampling=GREEDY)
    assert decide(greedy_samples, art, 1).reason is ReasonCode.SAMPLING_MISMATCH
    sample_defect = replace(base, defects=(DefectCode.SAMPLE_DEFECT,), agreement=None)
    assert decide(sample_defect, art, 1).action is Action.ESCALATE
    # token-logprob defects are irrelevant to a consistency-keyed tier
    assert decide(replace(base, defects=(DefectCode.UTF8_GAP,)), art, 1).action is Action.ANSWER


def test_gate_class_verifies_artifact() -> None:
    g = Gate(ART)
    assert g.decide(sig(-0.1), 1).action is Action.ANSWER
    with pytest.raises(ArtifactIntegrityError):
        Gate(replace(ART, alpha=0.5))


def test_end_to_end_with_real_signal_extraction() -> None:
    confident = compute_signals(completion([0.95, 0.9, 0.97]), S_BACKEND)
    assert decide(confident, ART, 1).action is Action.ANSWER
    unsure = compute_signals(completion([0.4, 0.3, 0.5]), S_BACKEND)
    assert decide(unsure, ART, 1).action is Action.ESCALATE


# ---- properties ---------------------------------------------------------------------


@given(st.floats(-20, 0), st.floats(-20, 0))
def test_gate_monotone_in_score(a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    rank = {Action.ABSTAIN: 0, Action.ESCALATE: 0, Action.ANSWER: 1}
    for tier, bk in ((1, S_BACKEND), (2, L_BACKEND)):
        assert (
            rank[decide(sig(lo, bk), ART, tier).action]
            <= rank[decide(sig(hi, bk), ART, tier).action]
        )


@given(
    st.sampled_from(sorted(DEFECT_COMPLETIONS)),
    st.lists(st.floats(0.05, 0.999), min_size=1, max_size=8),
    st.sampled_from([1, 2]),
)
def test_no_defect_ever_yields_answer(name: str, probs: list[float], tier: int) -> None:
    c = DEFECT_COMPLETIONS[name](completion(probs))
    b = S_BACKEND if tier == 1 else L_BACKEND
    assert decide(logprob_signals(c, b), ART, tier).action is not Action.ANSWER


def test_gate_matches_offline_policy_evaluation() -> None:
    """The offline LTT policy and the online gate must be the same function."""
    rng = np.random.default_rng(0)
    n = 400
    s, l_ = rng.uniform(-2, 0, n), rng.uniform(-2, 0, n)
    s[::37] = np.nan
    l_[::41] = np.nan
    table = CalTable(s, l_, rng.random(n) < 0.3, rng.random(n) < 0.2)
    for t1, t2 in [(-0.8, -0.5), (NEVER, -0.7), (-1.0, NEVER), (-0.3, -0.3)]:
        art = artifact(t1=None if t1 == NEVER else t1, t2=None if t2 == NEVER else t2)
        off = evaluate_policy(table, t1, t2)
        for i in range(n):
            s1 = sig(float(s[i]))
            d1 = decide(s1, art, 1)
            if d1.action is Action.ANSWER:
                assert off.by_s[i]
                continue
            assert not off.by_s[i]
            d2 = decide(sig(float(l_[i]), L_BACKEND), art, 2)
            assert (d2.action is Action.ANSWER) == bool(off.by_l[i])


def test_tiers_helper_identity_distinct() -> None:
    t = tiers()
    assert t[0].identity.key() != t[1].identity.key()
    assert isinstance(Completion, type)

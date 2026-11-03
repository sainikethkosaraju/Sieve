from __future__ import annotations

import math
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sieve.signals import (
    MalformedLogprobError,
    agreement,
    build_alias_table,
    cluster_answers,
    cluster_entropy,
    compute_signals,
    consistency_signals,
    extract_answer,
    logprob_signals,
    normalize_answer,
    token_entropy,
    token_margin,
    validate_completion,
)
from sieve.types import Completion, DefectCode, SamplingParams, TokenLP
from tests.helpers import GREEDY, SAMPLED, backend, completion, tok

B = backend()


def test_hand_computed_signals() -> None:
    # two tokens, p = 0.8 and 0.5; alternatives get (1-p)/2 = 0.1 and 0.25
    c = completion([0.8, 0.5])
    s = logprob_signals(c, B)
    assert s.defects == ()
    assert s.max_prob == pytest.approx(0.8)
    assert s.mean_logprob == pytest.approx((math.log(0.8) + math.log(0.5)) / 2)
    assert s.min_logprob == pytest.approx(math.log(0.5))
    assert s.sum_logprob == pytest.approx(math.log(0.8) + math.log(0.5))
    assert s.first_margin == pytest.approx(math.log(0.8) - math.log(0.1))
    assert s.mean_margin == pytest.approx(
        ((math.log(0.8) - math.log(0.1)) + (math.log(0.5) - math.log(0.25))) / 2
    )
    # truncated entropy of first token: renormalise (0.8, 0.1) -> (8/9, 1/9)
    h1 = -(8 / 9 * math.log(8 / 9) + 1 / 9 * math.log(1 / 9))
    assert s.neg_first_entropy == pytest.approx(-h1)
    assert s.n_answer_tokens == 2


def test_answer_span_skips_preamble() -> None:
    c = completion([0.99, 0.99, 0.3, 0.6], answer_start=2)
    s = logprob_signals(c, B)
    assert s.max_prob == pytest.approx(0.3)
    assert s.mean_logprob == pytest.approx((math.log(0.3) + math.log(0.6)) / 2)
    assert s.n_answer_tokens == 2


def test_entropy_and_margin_helpers() -> None:
    uniform = (("a", math.log(0.5)), ("b", math.log(0.5)))
    assert token_entropy(uniform) == pytest.approx(math.log(2))
    assert token_margin(uniform) == pytest.approx(0.0)
    peaked = (("a", math.log(0.999)), ("b", math.log(0.001)))
    assert token_entropy(peaked) < 0.01
    with pytest.raises(MalformedLogprobError):
        token_margin((("a", -0.1),))
    with pytest.raises(MalformedLogprobError):
        token_entropy(())
    with pytest.raises(MalformedLogprobError):
        token_entropy((("a", float("nan")),))


@given(st.lists(st.floats(min_value=0.01, max_value=0.99), min_size=1, max_size=12))
def test_signal_bounds_property(probs: list[float]) -> None:
    s = logprob_signals(completion(probs), B)
    assert s.defects == ()
    assert s.mean_logprob is not None and s.min_logprob is not None and s.max_prob is not None
    assert s.mean_logprob <= 0.0
    assert s.min_logprob <= s.mean_logprob + 1e-12
    assert 0.0 < s.max_prob <= 1.0
    assert s.neg_mean_entropy is not None and s.neg_mean_entropy <= 0.0
    assert s.mean_margin is not None and s.mean_margin >= 0.0


@given(st.floats(min_value=0.02, max_value=0.98), st.floats(min_value=0.02, max_value=0.98))
def test_max_prob_monotone_in_probability(p: float, q: float) -> None:
    lo, hi = sorted((p, q))
    a = logprob_signals(completion([lo]), B).max_prob
    b = logprob_signals(completion([hi]), B).max_prob
    assert a is not None and b is not None and a <= b


# ---- defect classes --------------------------------------------------------------


def _with_tokens(c: Completion, toks: tuple[TokenLP, ...] | None, **kw: object) -> Completion:
    return replace(c, tokens=toks, **kw)  # type: ignore[arg-type]


def _assert_rejected(c: Completion, code: DefectCode) -> None:
    assert code in validate_completion(c)
    s = logprob_signals(c, B)
    assert code in s.defects
    # a defective completion must not leak any number
    assert s.mean_logprob is None and s.max_prob is None and s.first_margin is None


def test_nan_logprob() -> None:
    c = completion([0.8, 0.5])
    bad = (c.tokens[0], replace(c.tokens[1], logprob=float("nan")))  # type: ignore[index]
    _assert_rejected(_with_tokens(c, bad), DefectCode.NAN_LOGPROB)


def test_inf_logprob() -> None:
    c = completion([0.8, 0.5])
    bad = (replace(c.tokens[0], logprob=float("-inf")), c.tokens[1])  # type: ignore[index]
    _assert_rejected(_with_tokens(c, bad), DefectCode.NAN_LOGPROB)


def test_positive_logprob() -> None:
    c = completion([0.8, 0.5])
    bad = (replace(c.tokens[0], logprob=0.3), c.tokens[1])  # type: ignore[index]
    _assert_rejected(_with_tokens(c, bad), DefectCode.POSITIVE_LOGPROB)


def test_tiny_positive_roundoff_tolerated() -> None:
    c = completion([0.8])
    ok = (replace(c.tokens[0], logprob=1e-9),)  # type: ignore[index]
    assert DefectCode.POSITIVE_LOGPROB not in validate_completion(_with_tokens(c, ok))


def test_missing_tokens_none_and_empty_and_count_mismatch() -> None:
    c = completion([0.8, 0.5])
    _assert_rejected(_with_tokens(c, None), DefectCode.MISSING_LOGPROBS)
    _assert_rejected(_with_tokens(c, ()), DefectCode.MISSING_LOGPROBS)
    _assert_rejected(replace(c, n_gen=5), DefectCode.MISSING_LOGPROBS)  # dropped logprobs


def test_utf8_multi_token_gap() -> None:
    # llama.cpp #31451 shape: an emoji spans two byte-tokens; backend drops/garbles them.
    toks = (tok("It is ", 0.9), tok("�", 0.6))
    c = Completion("It is \U0001f600", toks, "stop", 2, GREEDY)
    _assert_rejected(c, DefectCode.UTF8_GAP)
    # tokens that silently omit a piece of text are also a gap
    c2 = Completion("abc", (tok("a", 0.9), tok("c", 0.9)), "stop", 2, GREEDY)
    _assert_rejected(c2, DefectCode.UTF8_GAP)


def test_clean_multibyte_text_is_accepted() -> None:
    toks = (tok("café", 0.9), tok(" 東京", 0.8))
    c = Completion("café 東京", toks, "stop", 2, GREEDY)
    assert validate_completion(c) == ()


def test_top_logprob_defects() -> None:
    c = completion([0.8, 0.5])
    t0 = c.tokens[0]  # type: ignore[index]
    nan_top = replace(t0, top=((t0.token, t0.logprob), ("x", float("nan"))))
    _assert_rejected(_with_tokens(c, (nan_top, c.tokens[1])), DefectCode.TOP_LOGPROBS_INVALID)  # type: ignore[index]
    heavy = replace(t0, top=((t0.token, math.log(0.8)), ("x", math.log(0.7))))  # mass 1.5
    _assert_rejected(_with_tokens(c, (heavy, c.tokens[1])), DefectCode.TOP_LOGPROBS_INVALID)  # type: ignore[index]
    unsorted = replace(t0, top=((t0.token, math.log(0.4)), ("x", math.log(0.5))))
    _assert_rejected(_with_tokens(c, (unsorted, c.tokens[1])), DefectCode.TOP_LOGPROBS_INVALID)  # type: ignore[index]


def test_missing_top_flags_but_keeps_chosen_only_signals() -> None:
    c = completion([0.8, 0.5])
    stripped = tuple(replace(t, top=None) for t in c.tokens)  # type: ignore[union-attr]
    s = logprob_signals(_with_tokens(c, stripped), B)
    assert s.defects == (DefectCode.TOP_LOGPROBS_MISSING,)
    assert s.mean_logprob is not None and s.first_margin is None and s.neg_mean_entropy is None


def test_greedy_mismatch_only_when_temperature_zero() -> None:
    t = TokenLP("a", math.log(0.2), (("b", math.log(0.7)), ("a", math.log(0.2))))
    c = Completion("a", (t,), "stop", 1, GREEDY)
    assert DefectCode.GREEDY_MISMATCH in validate_completion(c)
    sampled = replace(c, sampling=SAMPLED)
    assert DefectCode.GREEDY_MISMATCH not in validate_completion(sampled)


def test_truncated_error_empty_and_bad_start() -> None:
    _assert_rejected(completion([0.9], finish="length"), DefectCode.TRUNCATED)
    _assert_rejected(completion([0.9], finish="error"), DefectCode.BACKEND_ERROR)
    _assert_rejected(completion([0.9, 0.9], answer_start=2), DefectCode.EMPTY_ANSWER)
    _assert_rejected(completion([0.9, 0.9], answer_start=7), DefectCode.BAD_ANSWER_START)
    _assert_rejected(completion([0.9, 0.9], answer_start=-1), DefectCode.BAD_ANSWER_START)
    empty = Completion("", (), "stop", 0, GREEDY)
    assert DefectCode.EMPTY_ANSWER in validate_completion(empty)


# ---- self-consistency ------------------------------------------------------------


def test_normalize_answer() -> None:
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("  Café, São-Paulo ") == "cafe sao paulo"
    assert normalize_answer("A") == ""


def test_extract_answer() -> None:
    assert extract_answer("The answer is Paris.\nbecause...") == "Paris"
    assert extract_answer("\n\n  Rome  ") == "Rome"
    assert extract_answer("") == ""


def test_clusters_with_aliases() -> None:
    answers = ["Paris", "paris.", "The Paris", "Lutetia", "London"]
    plain = cluster_answers(answers)
    assert plain.sizes == (3, 1, 1)
    merged = cluster_answers(answers, build_alias_table([["Paris", "Lutetia"]]))
    assert merged.sizes == (4, 1)
    assert agreement(merged) == pytest.approx(0.8)
    expected_h = -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
    assert cluster_entropy(merged) == pytest.approx(expected_h)


def test_agreement_extremes() -> None:
    same = cluster_answers(["x"] * 5)
    assert agreement(same) == 1.0 and cluster_entropy(same) == 0.0
    diff = cluster_answers(list("abcde"))
    assert agreement(diff) == pytest.approx(0.2)
    assert cluster_entropy(diff) == pytest.approx(math.log(5))
    with pytest.raises(ValueError):
        agreement(cluster_answers([]))


@given(st.lists(st.sampled_from(["a", "b", "c", "d"]), min_size=1, max_size=30))
def test_consistency_bounds(answers: list[str]) -> None:
    cl = cluster_answers(answers)
    k = len(answers)
    assert 1 / k <= agreement(cl) <= 1.0
    assert 0.0 <= cluster_entropy(cl) <= math.log(k) + 1e-12


def _sample(text: str, finish: str = "stop") -> Completion:
    return Completion(text, None, finish, 1, SAMPLED)  # type: ignore[arg-type]


def test_consistency_signals_defects() -> None:
    a, e, d = consistency_signals([_sample("x")])
    assert (a, e) == (None, None) and DefectCode.TOO_FEW_SAMPLES in d
    a, e, d = consistency_signals([_sample("x"), _sample("x", "error")])
    assert a is None and DefectCode.SAMPLE_DEFECT in d
    a, e, d = consistency_signals([_sample("x"), _sample("  ")])
    assert a is None and DefectCode.SAMPLE_DEFECT in d
    a, e, d = consistency_signals([_sample("Paris"), _sample("paris"), _sample("Rome")])
    assert d == () and a == pytest.approx(2 / 3)


def test_compute_signals_merges() -> None:
    primary = completion([0.9])
    primary = replace(primary, text="Paris", tokens=(tok("Paris", 0.9),))
    samples = [_sample("Paris"), _sample("paris"), _sample("paris")]
    s = compute_signals(primary, B, samples, consistency_sampling=SAMPLED)
    assert s.agreement == pytest.approx(1.0)
    assert s.n_samples == 3 and s.consistency_sampling == SAMPLED
    assert s.mean_logprob is not None
    s2 = compute_signals(primary, B)
    assert s2.agreement is None and s2.n_samples == 0


def test_signals_score_lookup() -> None:
    s = logprob_signals(completion([0.5]), B)
    assert s.score("max_prob") == pytest.approx(0.5)
    with pytest.raises(KeyError):
        s.score("nope")
    assert SamplingParams().key() == SamplingParams(seed=9, max_tokens=3).key()


def test_agreement_is_relative_to_served_answer() -> None:
    """F-03: served 'Madrid' with five 'Paris' samples must not look consistent."""
    primary = completion([0.9, 0.9], text="Madrid")
    samples = [_sample("Paris") for _ in range(5)]
    s = compute_signals(primary, B, samples, consistency_sampling=SAMPLED)
    assert s.agreement == 0.0
    ok = compute_signals(completion([0.9], text="Paris"), B, samples, consistency_sampling=SAMPLED)
    assert ok.agreement == 1.0
    part = compute_signals(
        completion([0.9], text="Paris"),
        B,
        [_sample("Paris"), _sample("Paris"), _sample("Rome"), _sample("Rome")],
        consistency_sampling=SAMPLED,
    )
    assert part.agreement == pytest.approx(0.5)


def test_agreement_does_not_depend_on_gold_answers() -> None:
    """F-02: the eval data path derives no alias table from gold answers."""
    import inspect

    from sieve.eval import data as eval_data

    src = inspect.getsource(eval_data)
    assert "build_alias_table" not in src and "aliases=" not in src
    primary = completion([0.9], text="Paris")
    samples = [_sample(t) for t in ("Paris", "City of Light", "Paris", "City of Light", "Paris")]
    a = compute_signals(primary, B, samples, consistency_sampling=SAMPLED).agreement
    # same samples, no gold information anywhere in the signature: result is fixed at 0.6
    assert a == pytest.approx(0.6)


@pytest.mark.parametrize("text", ["", "   \n", "\t"])
def test_whitespace_only_answer_is_a_defect(text: str) -> None:
    c = replace(completion([0.99]), text=text, tokens=(tok(text or " ", 0.99),), n_gen=1)
    c = replace(c, text=text, tokens=(tok(text, 0.99),)) if text else c
    assert DefectCode.EMPTY_ANSWER in validate_completion(c)

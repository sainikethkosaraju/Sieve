"""Graders against hand-labelled cases (aliases, punctuation, verbose answers, letters, judge)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sieve.eval.datasets import Item
from sieve.eval.grading import (
    BackendJudge,
    Grade,
    JudgeCache,
    JudgeUnavailableError,
    SimpleQALabel,
    UnavailableJudge,
    grade_alias_contains,
    grade_alias_em,
    grade_item,
    grade_mmlu_letter,
    grade_simpleqa,
    is_wrong,
    parse_choice_letter,
    parse_judge_reply,
    simpleqa_f1,
)
from sieve.types import BackendInfo, Completion, SamplingParams


def qa(*answers: str) -> Item:
    return Item("t:1", "triviaqa", "Q?", tuple(answers))


# (gold aliases, prediction, EM-correct, contains-correct)  -- labelled by hand
EM_CASES = [
    (("Paris",), "Paris", True, True),
    (("Paris",), "paris.", True, True),  # case + trailing period
    (("Paris",), "  PARIS!  ", True, True),
    (("The Beatles", "Beatles"), "Beatles", True, True),  # alias
    (("The Beatles",), "beatles", True, True),  # article stripped on both sides
    (("Beyoncé",), "Beyonce", True, True),  # accents
    (("J.R.R. Tolkien", "Tolkien"), "J.R.R. Tolkien", True, True),
    (
        ("J.R.R. Tolkien",),
        "JRR Tolkien",
        False,
        False,
    ),  # TriviaQA-style normalisation: dots -> space
    (("Paris",), "The answer is Paris.", True, True),  # preamble extracted
    (("Paris",), "Answer: Paris", True, True),
    (("Paris",), "Paris\nBecause it is the capital.", True, True),  # first line only
    (
        ("Paris",),
        "The capital of France is Paris.",
        False,
        True,
    ),  # verbose: EM misses, contains hits
    (("Paris",), "It might be Paris or Lyon", False, True),  # contains is lenient (documented)
    (("Paris",), "Lyon", False, False),
    (("Paris",), "", False, False),
    (("Paris",), "Parisian", False, False),  # whole-word containment only
    (("42",), "42.", True, True),
    (("42",), "forty-two", False, False),
    (("New York City", "NYC"), "nyc", True, True),
    (("St. Louis",), "Saint Louis", False, False),  # no synonym expansion
]


@pytest.mark.parametrize(("gold", "pred", "em", "contains"), EM_CASES)
def test_alias_graders_on_hand_labelled_cases(
    gold: tuple[str, ...], pred: str, em: bool, contains: bool
) -> None:
    assert grade_alias_em(qa(*gold), pred).correct is em
    assert grade_alias_contains(qa(*gold), pred).correct is contains


def test_em_label_strings() -> None:
    assert grade_alias_em(qa("cat"), "cat") == Grade(True, "correct")
    assert grade_alias_em(qa("cat"), "dog") == Grade(False, "wrong")


LETTER_CASES = [
    ("B", "B"),
    ("b", None),
    ("(C) foo", "C"),
    ("D.", "D"),
    ("A) text", "A"),
    ("The answer is B.", "B"),
    ("answer: c", "C"),
    ("A cat is an animal", None),
    ("", None),
    ("E", None),
    ("None of the above", None),
    (" B \n", "B"),
]


@pytest.mark.parametrize(("text", "letter"), LETTER_CASES)
def test_parse_choice_letter(text: str, letter: str | None) -> None:
    assert parse_choice_letter(text) == letter


def test_mmlu_grading() -> None:
    it = Item("m:1", "mmlu_redux", "Q", ("C",), ("a", "b", "c", "d"))
    assert grade_mmlu_letter(it, "C").correct is True
    assert grade_mmlu_letter(it, "(c)").label == "invalid"  # lowercase not accepted as a choice
    assert grade_mmlu_letter(it, "B") == Grade(False, "wrong")
    assert grade_mmlu_letter(it, "I think it is the third one") == Grade(False, "invalid")


def test_simpleqa_unavailable_state_is_explicit() -> None:
    g = grade_simpleqa(qa("x"), "x", UnavailableJudge())
    assert g == Grade(None, "unavailable")
    assert is_wrong(g) is None  # never silently counted as right or wrong
    with pytest.raises(JudgeUnavailableError):
        UnavailableJudge().judge("q", "g", "p")
    g2 = grade_item(Item("s:1", "simpleqa_verified", "Q", ("x",)), "x", "simpleqa_judge")
    assert g2.correct is None  # default judge is the unavailable one


class Scripted:
    judge_id = "scripted"
    model_id = "judge-model"

    def __init__(self, label: SimpleQALabel) -> None:
        self.label = label
        self.calls = 0

    def judge(self, question: str, gold: str, prediction: str) -> SimpleQALabel:
        self.calls += 1
        return self.label


def test_simpleqa_three_labels_and_wrongness(tmp_path: Path) -> None:
    it = qa("x")
    ok = grade_simpleqa(it, "x", Scripted(SimpleQALabel.CORRECT))
    bad = grade_simpleqa(it, "y", Scripted(SimpleQALabel.INCORRECT))
    na = grade_simpleqa(it, "I don't know", Scripted(SimpleQALabel.NOT_ATTEMPTED))
    assert (ok.correct, bad.correct, na.correct) == (True, False, False)
    assert (is_wrong(ok), is_wrong(bad), is_wrong(na)) == (False, True, False)
    assert is_wrong(na, not_attempted_is_wrong=True) is True


def test_judge_cache_makes_grading_idempotent(tmp_path: Path) -> None:
    j = Scripted(SimpleQALabel.CORRECT)
    cache = JudgeCache(tmp_path / "j.jsonl")
    grade_simpleqa(qa("x"), "x", j, cache=cache)
    grade_simpleqa(qa("x"), "x", j, cache=cache)
    assert j.calls == 1
    assert JudgeCache(tmp_path / "j.jsonl").get(JudgeCache.key("scripted", "t:1", "x")) is (
        SimpleQALabel.CORRECT
    )


def test_parse_judge_reply_and_f1() -> None:
    assert parse_judge_reply("A") is SimpleQALabel.CORRECT
    assert parse_judge_reply(" b.") is None  # lowercase is not a verdict
    assert parse_judge_reply("C - not attempted") is SimpleQALabel.NOT_ATTEMPTED
    assert parse_judge_reply("maybe") is None
    f = simpleqa_f1(["correct", "correct", "incorrect", "not_attempted"])
    assert f["correct"] == 0.5 and f["correct_given_attempted"] == pytest.approx(2 / 3)
    assert f["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))


class _FakeJudgeBackend:
    def __init__(self, model_id: str, reply: str) -> None:
        self._reply = reply
        self.info = BackendInfo(model_id, "h", "Q4", "f16", "b", "t")

    async def generate(self, prompt: str, params: SamplingParams, *, n_probs: int) -> Completion:
        return Completion(self._reply, None, "stop", 1, params)


def test_backend_judge_must_differ_from_tiers() -> None:
    with pytest.raises(ValueError, match="differ"):
        BackendJudge(_FakeJudgeBackend("Qwen-3B", "A"), forbidden_model_ids=["Qwen-3B", "Qwen-7B"])
    j = BackendJudge(_FakeJudgeBackend("judge-x", "B"), forbidden_model_ids=["Qwen-3B"])
    assert j.judge("q", "g", "p") is SimpleQALabel.INCORRECT
    assert "judge-x" in j.judge_id
    bad = BackendJudge(_FakeJudgeBackend("judge-x", "banana"))
    with pytest.raises(JudgeUnavailableError):  # unparseable verdict is not a silent 'incorrect'
        bad.judge("q", "g", "p")


def test_stated_letter_regex_is_linear_time() -> None:
    """F-18: 'answer' followed by many spaces must not blow up quadratically."""
    import time

    from sieve.eval.grading import parse_choice_letter

    t0 = time.perf_counter()
    assert parse_choice_letter("answer" + " " * 60_000 + "x") is None
    assert time.perf_counter() - t0 < 1.0
    assert parse_choice_letter("The answer is (C).") == "C"
    assert parse_choice_letter("Answer: B") == "B"

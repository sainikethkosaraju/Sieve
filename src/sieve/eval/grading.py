# ruff: noqa: E501
"""Graders.

* ``em-alias-v1`` (primary for TriviaQA/NQ-open): normalised exact match of the extracted
  answer against the gold answer and all aliases.
* ``contains-alias-v1`` (sensitivity only): an alias occurs as a whole-word span of the
  normalised prediction; more lenient for verbose answers, never the primary grader.
* ``mmlu-letter-v1``: the chosen option letter equals the gold letter; an unparseable
  reply is wrong (format failures are grader-visible by design).
* SimpleQA: a three-label judge behind ``Judge``. If no judge model is configured the state
  is ``unavailable`` and nothing is graded; no other grader is substituted.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from sieve.eval._util import sha256_text
from sieve.eval.datasets import Item
from sieve.signals.consistency import extract_answer, normalize_answer
from sieve.types import Backend, SamplingParams

EM_ID = "em-alias-v1"
CONTAINS_ID = "contains-alias-v1"
MMLU_ID = "mmlu-letter-v1"


@dataclass(frozen=True)
class Grade:
    """``correct`` is ``None`` only when the grader is unavailable (never guessed)."""

    correct: bool | None
    label: str  # "correct" | "wrong" | "incorrect" | "not_attempted" | "unavailable" | "invalid"


# --- exact match with aliases ------------------------------------------------------


def grade_alias_em(item: Item, prediction: str) -> Grade:
    pred = normalize_answer(extract_answer(prediction))
    if pred == "":
        return Grade(False, "wrong")
    golds = {normalize_answer(a) for a in item.answers}
    golds.discard("")
    return Grade(pred in golds, "correct" if pred in golds else "wrong")


def grade_alias_contains(item: Item, prediction: str) -> Grade:
    """Whole-word containment of any alias in the normalised first line of ``prediction``."""
    first = next((ln for ln in prediction.splitlines() if ln.strip()), "")
    pred = f" {normalize_answer(first)} "
    for a in item.answers:
        g = normalize_answer(a)
        if g and f" {g} " in pred:
            return Grade(True, "correct")
    return Grade(False, "wrong")


# --- MMLU letters --------------------------------------------------------------------

_LEAD = re.compile(r"^\W*\(?([ABCD])\)?(?:[.:)\s]*$|[.:)])")
_STATED = re.compile(r"answer(?:\s*(?:is|:))?\s*\(?([ABCD])\b", re.IGNORECASE)


def parse_choice_letter(text: str) -> str | None:
    """The option letter a reply commits to, or ``None``.

    ``"B"``, ``"(C) foo"``, ``"D."`` and ``"The answer is B."`` parse; ``"A cat is ..."`` does
    not (the leading ``A`` is an article, not a choice).
    """
    t = text.strip()
    m = _LEAD.match(t)
    if m:
        return m.group(1)
    s = _STATED.search(t)
    if s:
        return s.group(1).upper()
    return None


def grade_mmlu_letter(item: Item, prediction: str) -> Grade:
    letter = parse_choice_letter(prediction)
    if letter is None:
        return Grade(False, "invalid")
    ok = letter == item.answers[0]
    return Grade(ok, "correct" if ok else "wrong")


# --- SimpleQA three-label judge --------------------------------------------------------


class SimpleQALabel(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    NOT_ATTEMPTED = "not_attempted"


class JudgeUnavailableError(RuntimeError):
    """No judge model is configured or reachable. Raised, never swallowed into a grade."""


class Judge(Protocol):
    @property
    def judge_id(self) -> str: ...

    @property
    def model_id(self) -> str | None: ...

    def judge(self, question: str, gold: str, prediction: str) -> SimpleQALabel: ...


class UnavailableJudge:
    """Explicit 'no judge' state. Grading with it raises; it never returns a label."""

    judge_id = "simpleqa-judge:unavailable"
    model_id = None

    def __init__(self, reason: str = "no judge model configured") -> None:
        self.reason = reason

    def judge(self, question: str, gold: str, prediction: str) -> SimpleQALabel:
        raise JudgeUnavailableError(self.reason)


JUDGE_PROMPT = """You are grading a short-form factual answer.
Question: {question}
Gold answer: {gold}
Predicted answer: {prediction}

Reply with exactly one letter:
A = CORRECT: the prediction contains the gold answer's key information and does not contradict it.
B = INCORRECT: the prediction contradicts the gold answer or gives a different answer, even if hedged.
C = NOT_ATTEMPTED: the prediction declines to answer or gives no answer, without contradicting the gold answer.
Letter:"""

_LABELS = {
    "A": SimpleQALabel.CORRECT,
    "B": SimpleQALabel.INCORRECT,
    "C": SimpleQALabel.NOT_ATTEMPTED,
}


def parse_judge_reply(text: str) -> SimpleQALabel | None:
    m = re.match(r"^\W*([ABC])\b", text.strip())
    return _LABELS[m.group(1)] if m else None


class BackendJudge:
    """Judge served by any ``Backend``; the model must differ from both tiers."""

    def __init__(self, backend: Backend, *, forbidden_model_ids: Sequence[str] = ()) -> None:
        self._backend = backend
        mid = backend.info.model_id
        if mid in set(forbidden_model_ids):
            raise ValueError(
                f"judge model {mid!r} equals a tier model; the judge must differ from S and L"
            )
        self._model_id = mid

    @property
    def judge_id(self) -> str:
        return f"simpleqa-judge:{self._model_id}:{sha256_text(JUDGE_PROMPT)[:8]}"

    @property
    def model_id(self) -> str | None:
        return self._model_id

    def judge(self, question: str, gold: str, prediction: str) -> SimpleQALabel:
        prompt = JUDGE_PROMPT.format(question=question, gold=gold, prediction=prediction)
        params = SamplingParams(temperature=0.0, max_tokens=4)
        comp = asyncio.run(self._backend.generate(prompt, params, n_probs=1))
        label = parse_judge_reply(comp.text)
        if label is None:
            # an unparseable verdict is a judge failure, not a silent 'incorrect'
            raise JudgeUnavailableError(f"unparseable judge reply {comp.text!r}")
        return label


class JudgeCache:
    """Append-only cache of judge verdicts so grading is resumable and reproducible."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._d: dict[str, str] = {}
        if path.exists():
            for ln in path.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    rec = json.loads(ln)
                    self._d[rec["k"]] = rec["label"]

    @staticmethod
    def key(judge_id: str, item_id: str, prediction: str) -> str:
        return sha256_text(f"{judge_id}|{item_id}|{prediction}")

    def get(self, k: str) -> SimpleQALabel | None:
        v = self._d.get(k)
        return None if v is None else SimpleQALabel(v)

    def put(self, k: str, label: SimpleQALabel) -> None:
        self._d[k] = label.value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"k": k, "label": label.value}) + "\n")


def grade_simpleqa(
    item: Item,
    prediction: str,
    judge: Judge,
    *,
    cache: JudgeCache | None = None,
) -> Grade:
    """Judge-graded verdict, or ``Grade(None, 'unavailable')`` when there is no judge."""
    pred = extract_answer(prediction) or prediction.strip()
    k = JudgeCache.key(judge.judge_id, item.item_id, pred)
    label = cache.get(k) if cache else None
    if label is None:
        try:
            label = judge.judge(item.question, item.answers[0], pred)
        except JudgeUnavailableError:
            return Grade(None, "unavailable")
        if cache:
            cache.put(k, label)
    if label is SimpleQALabel.CORRECT:
        return Grade(True, "correct")
    if label is SimpleQALabel.NOT_ATTEMPTED:
        return Grade(False, "not_attempted")
    return Grade(False, "incorrect")


def is_wrong(grade: Grade, *, not_attempted_is_wrong: bool = False) -> bool | None:
    """The 'wrong' flag used for risk: ``None`` if ungraded, ``True`` for incorrect answers.

    A SimpleQA ``not_attempted`` reply is not counted as wrong unless asked; it is reported
    separately (see ``simpleqa_f1``).
    """
    if grade.correct is None:
        return None
    if grade.correct:
        return False
    if grade.label == "not_attempted":
        return not_attempted_is_wrong
    return True


def simpleqa_f1(labels: Sequence[str]) -> dict[str, float]:
    """SimpleQA-style summary: overall correct, correct given attempted, harmonic F1."""
    n = len(labels)
    if n == 0:
        return {"correct": 0.0, "correct_given_attempted": 0.0, "f1": 0.0}
    n_correct = sum(1 for x in labels if x == "correct")
    n_attempted = sum(1 for x in labels if x in ("correct", "incorrect"))
    overall = n_correct / n
    given = n_correct / n_attempted if n_attempted else 0.0
    f1 = 2 * overall * given / (overall + given) if overall + given > 0 else 0.0
    return {"correct": overall, "correct_given_attempted": given, "f1": f1}


def grade_item(item: Item, prediction: str, grader: str, judge: Judge | None = None) -> Grade:
    """Dispatch on the grader named in the dataset spec."""
    if grader == "alias_em":
        return grade_alias_em(item, prediction)
    if grader == "mmlu_letter":
        return grade_mmlu_letter(item, prediction)
    if grader == "simpleqa_judge":
        return grade_simpleqa(item, prediction, judge or UnavailableJudge())
    raise ValueError(f"unknown grader {grader!r}")

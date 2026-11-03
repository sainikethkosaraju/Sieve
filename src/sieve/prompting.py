"""Single source of truth for calibration prompts, template hash and answer-span location.

Both the evaluation harness (``sieve.eval``) and the serving router (``sieve.router``) build
prompts through this module so that the signal is computed on the same text and the same
answer span at calibration and at deployment (F-01 / D-019). Changing any template text or
``TEMPLATE_VERSION`` changes :func:`template_hash` and invalidates every calibration.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from sieve.types import Completion

TEMPLATE_VERSION = "qa-v2-fewshot"

# Few-shot prefix (D-016): a raw base-style prompt makes the 7B open with " The" before the
# answer, so the answer span is undefined. Worked examples ending in "Answer: <phrase>" make the
# first generated token the first answer token.
QA_TEMPLATE = (
    "Answer each question with a short phrase only, without explanation.\n\n"
    "Question: What is the chemical symbol for gold?\nAnswer: Au\n\n"
    "Question: Who wrote the play Romeo and Juliet?\nAnswer: William Shakespeare\n\n"
    "Question: In which country is the city of Kyoto?\nAnswer: Japan\n\n"
    "Question: What is the largest planet in the Solar System?\nAnswer: Jupiter\n\n"
    "Question: {question}\nAnswer:"
)
MCQ_TEMPLATE = (
    "Answer each multiple-choice question with the letter of the correct option only.\n\n"
    "Question: Which of these is a primary colour?\nA. Green\nB. Red\nC. Purple\nD. Orange\n"
    "Answer: B\n\n"
    "Question: How many days are in a leap year?\nA. 364\nB. 365\nC. 360\nD. 366\n"
    "Answer: D\n\n"
    "Question: Which planet is closest to the Sun?\nA. Mercury\nB. Venus\nC. Earth\nD. Mars\n"
    "Answer: A\n\n"
    "Question: {question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"
)


SQUAD_VERSION = "squad-v1-fewshot"
SQUAD_TEMPLATE = (
    "Read the passage and answer the question with a short phrase copied from the passage. "
    "If the passage does not contain the answer, reply: unanswerable\n\n"
    "Passage: The Eiffel Tower was completed in 1889 for the World's Fair in Paris. "
    "It was designed by the engineering company of Gustave Eiffel.\n"
    "Question: In which year was the Eiffel Tower completed?\nAnswer: 1889\n\n"
    "Passage: The Amazon River flows through Brazil, Peru and Colombia. It is the largest river "
    "in the world by discharge volume.\n"
    "Question: Who designed the Amazon River?\nAnswer: unanswerable\n\n"
    "Passage: Marie Curie won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry "
    "in 1911.\n"
    "Question: In which year did Marie Curie win the Nobel Prize in Chemistry?\nAnswer: 1911\n\n"
    "Passage: {context}\nQuestion: {question}\nAnswer:"
)


def template_hash() -> str:
    """Full sha256 hex of the calibration template (identical in eval and server)."""
    body = f"{TEMPLATE_VERSION}\n{QA_TEMPLATE}\n{MCQ_TEMPLATE}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def template_hash_for(datasets: Sequence[str]) -> str:
    """Template hash for a set of datasets: the SQuAD template hashes separately, and v1's
    TriviaQA/NQ/MMLU hash (and every cache key derived from it) is unchanged."""
    if "squad_v2" in datasets:
        if any(d != "squad_v2" for d in datasets):
            raise ValueError("squad_v2 uses its own template; do not mix it with other datasets")
        return hashlib.sha256(f"{SQUAD_VERSION}\n{SQUAD_TEMPLATE}".encode()).hexdigest()
    return template_hash()


def render_calibration_prompt(
    question: str, choices: Sequence[str] | None = None, context: str = ""
) -> str:
    """The exact prompt string sent to ``/completion`` for one question."""
    if context:
        return SQUAD_TEMPLATE.format(context=context, question=question)
    if choices:
        a, b, c, d = choices
        return MCQ_TEMPLATE.format(question=question, a=a, b=b, c=c, d=d)
    return QA_TEMPLATE.format(question=question)


_PREAMBLE = re.compile(r"^\s*(?:the\s+answer\s+is|answer\s+is|answer\s*:)\s*", re.IGNORECASE)


def find_answer_start(c: Completion) -> int:
    """Index of the first answer token, skipping an 'The answer is' style preamble.

    0 if there is no preamble or the tokens cannot be aligned to the text (the signals
    package then flags any inconsistency as a defect).
    """
    if c.tokens is None:
        return 0
    m = _PREAMBLE.match(c.text)
    if not m or m.end() == 0:
        return 0
    offset = m.end()
    pos = 0
    for i, t in enumerate(c.tokens):
        if pos >= offset:
            return i
        pos += len(t.token)
    return 0

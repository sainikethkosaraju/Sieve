"""Evaluation prompt rendering: thin delegation to :mod:`sieve.prompting`.

The template text, hash and answer-span logic live in ``sieve.prompting`` and are shared with
the serving router (F-01 / D-019). Changing a template changes ``template_hash`` and therefore
invalidates every calibration (D-008).
"""

from __future__ import annotations

from sieve import prompting
from sieve.eval.datasets import PromptItem
from sieve.prompting import (
    MCQ_TEMPLATE,
    QA_TEMPLATE,
    TEMPLATE_VERSION,
    find_answer_start,
    template_hash,
)

__all__ = [
    "MCQ_TEMPLATE",
    "QA_TEMPLATE",
    "TEMPLATE_VERSION",
    "find_answer_start",
    "render_prompt",
    "template_hash",
]


def render_prompt(item: PromptItem) -> str:
    return prompting.render_calibration_prompt(item.question, item.choices or None)

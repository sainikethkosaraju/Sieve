"""SQuAD 2.0 support for the v2 experiment: parsing, prompting, grading, and v1 invariance."""

from __future__ import annotations

from sieve import prompting
from sieve.eval.datasets import PINNED, Item, dataset_hash, parse_squad_v2
from sieve.eval.grading import grade_alias_em

ROWS = [
    {
        "id": "a1",
        "title": "Paris",
        "context": "Paris is the capital of France.",
        "question": "What is the capital of France?",
        "answers": {"text": ["Paris", "Paris"], "answer_start": [0, 0]},
    },
    {
        "id": "u1",
        "title": "Paris",
        "context": "Paris is the capital of France.",
        "question": "Who is the mayor of Lyon?",
        "answers": {"text": [], "answer_start": []},
    },
]

V1_TEMPLATE_HASH = "5faade81b16649e65dcff2dcc2af37ad7c0b4cf6713ac2f599a95941a07417ef"


def test_parse_marks_unanswerable_and_keeps_context() -> None:
    a, u = parse_squad_v2(ROWS)
    assert a.answers == ("Paris",) and a.stratum == "answerable" and a.context.startswith("Paris")
    assert u.answers == ("unanswerable",) and u.stratum == "unanswerable"
    assert Item.from_dict(a.to_dict()) == a  # context survives the item file round trip


def test_prompt_view_carries_context_and_no_gold() -> None:
    a, _ = parse_squad_v2(ROWS)
    pv = a.prompt_view()
    assert pv.context == a.context and not hasattr(pv, "answers")
    text = prompting.render_calibration_prompt(pv.question, None, pv.context)
    assert text.endswith(f"Passage: {a.context}\nQuestion: {a.question}\nAnswer:")
    assert "unanswerable" in text  # the instruction and a worked example


def test_one_exact_match_grader_scores_answerable_and_unanswerable() -> None:
    a, u = parse_squad_v2(ROWS)
    assert grade_alias_em(a, " Paris").correct and not grade_alias_em(a, "unanswerable").correct
    assert grade_alias_em(u, " unanswerable").correct and not grade_alias_em(u, "Lyon").correct


def test_v1_template_hash_and_dataset_hashes_are_unchanged() -> None:
    assert prompting.template_hash() == V1_TEMPLATE_HASH
    assert prompting.template_hash_for(["triviaqa", "nq_open", "mmlu_redux"]) == V1_TEMPLATE_HASH
    sq = prompting.template_hash_for(["squad_v2"])
    assert sq != V1_TEMPLATE_HASH and len(sq) == 64
    import pytest

    with pytest.raises(ValueError):
        prompting.template_hash_for(["squad_v2", "triviaqa"])
    # closed-book items serialise exactly as before (no 'context' key), so dataset hashes are stable
    t = Item("triviaqa:1", "triviaqa", "q", ("a",))
    assert "context" not in t.to_dict() and dataset_hash([t]) == dataset_hash([t])


def test_squad_spec_is_pinned_and_licensed() -> None:
    s = PINNED["squad_v2"]
    assert s.revision == "3ffb306f725f7d2ce8394bc1873b24868140c412"
    assert s.licence == "CC-BY-SA-4.0" and s.role == "primary" and s.grader == "alias_em"

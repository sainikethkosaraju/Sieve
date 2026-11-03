"""Loaders (field names as verified against the Hub), sampling, and item-file integrity."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from sieve.eval import datasets as ds
from sieve.eval.datasets import DatasetError, Item


def _tqa_rows(n: int) -> list[dict[str, Any]]:
    return [
        {
            "question": f"Who wrote book {i}?",
            "question_id": f"tc_{i}",
            "question_source": "x",
            "answer": {
                "aliases": [f"Author {i}", f"A. {i}"],
                "value": f"Author {i}",
                "normalized_aliases": [],
                "type": "WikipediaEntity",
            },
        }
        for i in range(n)
    ]


def test_parse_triviaqa_uses_value_and_aliases() -> None:
    items = ds.parse_triviaqa(_tqa_rows(3))
    assert items[0].item_id == "triviaqa:tc_0"
    assert items[0].answers == ("Author 0", "A. 0")
    assert items[0].choices == ()


def test_parse_nq_open_answer_is_a_list() -> None:
    items = ds.parse_nq_open([{"question": "who sang x", "answer": ["Bob", "Robert"]}])
    assert items[0].answers == ("Bob", "Robert")


def test_parse_simpleqa_verified_csv_columns() -> None:
    row = {
        "original_index": 5,
        "problem": "How much?",
        "answer": "120,000 euros",
        "topic": "Politics",
        "answer_type": "Number",
        "multi_step": True,
        "requires_reasoning": False,
        "urls": "u",
    }
    it = ds.parse_simpleqa_verified([row])[0]
    assert (it.item_id, it.answers, it.stratum) == (
        "simpleqa_verified:5",
        ("120,000 euros",),
        "Politics",
    )


def test_parse_mmlu_redux_keeps_only_ok_rows() -> None:
    rows: dict[str, Any] = {
        "anatomy": [
            {"question": "q1", "choices": ["a", "b", "c", "d"], "answer": 2, "error_type": "ok"},
            {
                "question": "q2",
                "choices": ["a", "b", "c", "d"],
                "answer": 0,
                "error_type": "wrong_groundtruth",
            },
        ]
    }
    items = ds.parse_mmlu_redux(rows)
    assert len(items) == 1 and items[0].answers == ("C",) and items[0].stratum == "anatomy"


def test_mmlu_redux_loader_fails_loudly_if_no_row_is_ok() -> None:
    rows: dict[str, Any] = {
        "anatomy": [{"question": "q", "choices": list("abcd"), "answer": 0, "error_type": "clean"}]
    }
    with pytest.raises(DatasetError, match="error_type"):
        ds.parse_mmlu_redux(rows)


def test_load_dataset_items_with_injected_fetcher() -> None:
    calls: list[tuple[str, str | None, str, str]] = []

    def fetch(repo: str, cfg: str | None, split: str, rev: str) -> list[dict[str, Any]]:
        calls.append((repo, cfg, split, rev))
        return _tqa_rows(5)

    spec = ds.PINNED["triviaqa"]
    items = ds.load_dataset_items(spec, fetch)
    assert len(items) == 5
    assert calls == [(spec.repo, "rc.nocontext", "validation", spec.revision)]  # pinned revision


def test_pins_are_complete_and_licensed() -> None:
    for spec in ds.PINNED.values():
        assert len(spec.revision) == 40 and spec.licence
        assert spec.n > 0
    assert ds.PINNED["triviaqa"].n == 6000 and ds.PINNED["mmlu_redux"].n == 1500
    assert ds.PINNED["nq_open"].n == 2000 and ds.PINNED["simpleqa_verified"].n == 1000


def test_sampling_is_order_independent_and_seeded() -> None:
    items = ds.parse_triviaqa(_tqa_rows(200))
    a = ds.sample_items(items, 50, seed=1)
    b = ds.sample_items(list(reversed(items)), 50, seed=1)
    c = ds.sample_items(items, 50, seed=2)
    assert a == b and a != c and len(a) == 50
    with pytest.raises(DatasetError):
        ds.sample_items(items, 500, seed=1)


def test_duplicate_questions_are_removed() -> None:
    items = [
        Item("a", "d", "What is X?", ("x",)),
        Item("b", "d", "what is  x", ("x",)),
        Item("c", "d", "Other?", ("y",)),
    ]
    kept = ds.dedupe_questions(items)
    assert [i.item_id for i in kept] == ["a", "c"]


def test_item_files_are_hash_verified(tmp_path: Path) -> None:
    items = ds.parse_triviaqa(_tqa_rows(10))
    p = tmp_path / "t.jsonl"
    h = ds.write_items(p, items, {"source": "test"})
    assert ds.read_items(p) == sorted(items, key=lambda i: i.item_id)
    assert ds.dataset_hash(items) == h
    p.write_text(p.read_text().replace("Author 3", "Author X"))
    with pytest.raises(DatasetError):
        ds.read_items(p)


def test_prompt_view_hides_gold() -> None:
    it = Item("a", "d", "Q?", ("secret",), ("1", "2", "3", "4"))
    v = it.prompt_view()
    assert not hasattr(v, "answers") and v.choices == it.choices


@pytest.mark.skipif(os.environ.get("SIEVE_NETWORK_TESTS") != "1", reason="needs Hub data access")
def test_live_schema_matches_parsers() -> None:  # pragma: no cover - opt-in
    for name in ("nq_open", "simpleqa_verified"):
        spec = ds.PINNED[name]
        rows = list(ds.hf_fetch(spec.repo, spec.config, spec.split, spec.revision))[:3]
        assert ds.load_dataset_items(spec, lambda *_a, _r=rows: _r)

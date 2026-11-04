"""Serving demo plumbing, offline: a harness-certified artifact is served through the app."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from sieve.eval import pipeline
from sieve.eval._fake_backend import FakeProvider, SimClock
from sieve.eval.config import parse_config
from sieve.eval.datasets import read_items
from sieve.eval.serve_demo import (
    DemoError,
    artifact_from_certification,
    serve_and_ask,
    summarise,
)

pytestmark = pytest.mark.slow


def test_certified_artifact_is_served_through_the_chat_endpoint(
    mock_run: tuple[Path, dict[str, Any]],
) -> None:
    wd, _ = mock_run
    art = None
    for f in sorted((wd / "certs").glob("split_*.json")):
        try:
            art = artifact_from_certification(f)
            break
        except DemoError:
            continue
    if art is None:
        pytest.skip("no synthetic split certified the primary gate")
    cfg = parse_config(json.loads((wd / "config.json").read_text()))
    items = read_items(wd / "items" / "triviaqa.jsonl")[:60]
    pair = cfg.main_pair()
    provider = FakeProvider(SimClock())
    with (
        provider.open(cfg.cell(pair.s_cell)) as small,
        provider.open(cfg.cell(pair.l_cell)) as large,
    ):
        raw = asyncio.run(serve_and_ask(art, small, large, items, concurrency=4))
    out = summarise(raw, art, len(items))
    assert out["n_responses"] == 60 and out["readyz_status"] == 200 and out["models_status"] == 200
    # the harness-certified artifact is accepted by the deployed gate (no identity mismatch)
    assert not any("mismatch" in r for r in out["reasons"]), out["reasons"]
    assert sum(out["decisions"].values()) == 60 and out["alpha_target"] == art.alpha


def test_uncertified_split_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "split_000.json"
    p.write_text(
        json.dumps({"certs": {"main": {"gate:primary": {"outcome": "empty", "artifact": None}}}})
    )
    with pytest.raises(DemoError):
        artifact_from_certification(p)
    assert pipeline  # imported for parity with the other pipeline tests

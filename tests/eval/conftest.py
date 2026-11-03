"""Shared fixtures: one small synthetic mock run reused by the pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from sieve.eval import pipeline
from sieve.eval._fake_backend import FakeProvider, SimClock


@pytest.fixture(scope="session")
def mock_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    wd = tmp_path_factory.mktemp("mock")
    sim = SimClock()
    out = pipeline.run_mock(
        wd, cfg_dict=pipeline.mock_config(small=True), provider=FakeProvider(sim), clock=sim
    )
    return wd, out

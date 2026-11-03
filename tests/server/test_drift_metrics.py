"""Drift alarm wired into the server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sieve.backends import MockBackend
from sieve.signals import logprob_signals
from sieve.types import SamplingParams
from tests.helpers import backend
from tests.server.harness import body, large_mock, make_app, running

URL = "/v1/chat/completions"


@pytest.mark.anyio
async def test_server_drift_gauge_fires_on_shifted_traffic(tmp_path: Path) -> None:
    ref_model = MockBackend(backend("small"), accuracy=1.0, seed=99)
    scores: list[float] = []
    for i in range(300):  # reference = tier-1 scores on "calibration" prompts
        comp = await ref_model.generate(f"cal{i}", SamplingParams(), n_probs=5)
        v = logprob_signals(comp, ref_model.info).mean_logprob
        assert v is not None
        scores.append(v)
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(scores))
    drift = {"reference_path": str(ref), "window": 60, "min_window": 40, "check_every": 10}

    stable = MockBackend(backend("small"), accuracy=1.0)
    async with running(make_app(stable, large_mock(), drift=drift)) as c:
        for i in range(80):
            await c.post(URL, json=body(f"s{i}"))
        m1 = (await c.get("/metrics")).text
    assert "sieve_drift_alarm 0.0" in m1

    shifted = MockBackend(backend("small"), accuracy=0.3, seed=5)  # many low-confidence answers
    async with running(make_app(shifted, large_mock(), drift=drift)) as c:
        for i in range(80):
            await c.post(URL, json=body(f"d{i}"))
        m2 = (await c.get("/metrics")).text
    assert "sieve_drift_alarm 1.0" in m2

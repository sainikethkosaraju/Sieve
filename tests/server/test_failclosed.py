"""Red-team kill tests 2, 3, 12: fail closed on defects and identity mismatch."""

from __future__ import annotations

import pytest

from sieve.backends import MockDefect
from sieve.observe import MemoryAuditSink
from tests.helpers import artifact, backend
from tests.server.harness import (
    ART,
    body,
    large_mock,
    make_app,
    running,
    small_mock,
)

URL = "/v1/chat/completions"

# defects that invalidate a chosen-token score (mean_logprob)
FATAL = [
    MockDefect.NAN_LOGPROB,
    MockDefect.POSITIVE_LOGPROB,
    MockDefect.DROPPED_LOGPROB,
    MockDefect.NO_LOGPROBS,
    MockDefect.UTF8_GAP,
    MockDefect.TRUNCATED,
    MockDefect.ERROR_FINISH,
    MockDefect.RAISE,
    MockDefect.GREEDY_MISMATCH,
]


@pytest.mark.anyio
@pytest.mark.parametrize("defect", FATAL, ids=lambda d: d.value)
async def test_defective_small_never_answers_and_escalates(defect: MockDefect) -> None:
    """Per-request validation (self-test disabled so the request path is what is tested)."""
    sm = small_mock(defect=defect)  # accuracy 1.0: without the defect it would answer
    lg = large_mock()
    sink = MemoryAuditSink()
    async with running(make_app(sm, lg, sink=sink, require_self_test=False)) as c:
        j = (await c.post(URL, json=body())).json()
    s = j["sieve"]
    assert s["tier"] == 2 and s["escalated"] is True  # never served by the defective tier
    t1 = sink.records[-1].tiers[0]
    assert t1["action"] == "escalate" and t1["reason"] == "logprob_defect"
    assert t1["defects"] or t1["backend_error"]


@pytest.mark.anyio
@pytest.mark.parametrize("defect", FATAL, ids=lambda d: d.value)
async def test_defective_both_tiers_abstain(defect: MockDefect) -> None:
    sm, lg = small_mock(defect=defect), large_mock(defect=defect)
    async with running(make_app(sm, lg, require_self_test=False)) as c:
        j = (await c.post(URL, json=body())).json()
    assert j["sieve"]["decision"] == "abstain" and j["sieve"]["reason"] == "logprob_defect"
    assert j["choices"][0]["finish_reason"] == "sieve_abstain"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "defect",
    [d for d in FATAL if d not in (MockDefect.TRUNCATED, MockDefect.ERROR_FINISH)]
    + [MockDefect.NONDETERMINISTIC],
    ids=lambda d: d.value,
)
async def test_self_test_blocks_logprob_use_and_readyz_fails(defect: MockDefect) -> None:
    sm, lg = small_mock(defect=defect), large_mock()
    async with running(make_app(sm, lg)) as c:
        r = await c.get("/readyz")
        assert r.status_code == 503 and r.json()["tiers"]["1"]["logprobs_ok"] is False
        calls_before = sm.calls
        j = (await c.post(URL, json=body())).json()
    assert sm.calls == calls_before  # no GPU time spent on an untrusted tier
    assert j["sieve"]["tier"] == 2 and j["sieve"]["escalated"] is True


@pytest.mark.anyio
async def test_top_k_missing_only_matters_for_signals_that_need_it() -> None:
    sm = small_mock(defect=MockDefect.NO_TOP)
    async with running(make_app(sm, require_self_test=False)) as c:
        assert (await c.post(URL, json=body())).json()["sieve"]["tier"] == 1  # mean_logprob
    art_margin = artifact(t1=-0.5, t2=-0.3, signal1="first_margin")
    async with running(
        make_app(small_mock(defect=MockDefect.NO_TOP), art=art_margin, require_self_test=False)
    ) as c:
        j = (await c.post(URL, json=body())).json()
    assert j["sieve"]["tier"] == 2


@pytest.mark.anyio
async def test_identity_mismatch_forces_escalation_without_spending_compute() -> None:
    wrong = backend("small", quant="Q2_K")
    sm = small_mock()
    sm.info = wrong
    lg = large_mock()
    sink = MemoryAuditSink()
    async with running(make_app(sm, lg, sink=sink)) as c:
        ready = await c.get("/readyz")
        assert ready.status_code == 503
        assert ready.json()["tiers"]["1"]["identity_matches_artifact"] is False
        calls = sm.calls
        j = (await c.post(URL, json=body())).json()
    assert sm.calls == calls
    assert j["sieve"]["tier"] == 2
    assert sink.records[-1].tiers[0]["reason"] == "identity_mismatch"
    assert sink.records[-1].tiers[0]["skipped_generation"] is True


@pytest.mark.anyio
@pytest.mark.parametrize("field", ["model_hash", "quant", "kv_quant", "build", "template_hash"])
async def test_every_identity_field_is_enforced_on_both_tiers(field: str) -> None:
    sm, lg = small_mock(), large_mock()
    sm.info = backend("small", **{field: "other"})
    lg.info = backend("large", **{"model_hash": "h2", field: "other"})
    async with running(make_app(sm, lg)) as c:
        before = (sm.calls, lg.calls)
        j = (await c.post(URL, json=body())).json()
    assert j["sieve"]["decision"] == "abstain" and j["sieve"]["reason"] == "identity_mismatch"
    assert (sm.calls, lg.calls) == before  # no compute spent on mismatched tiers
    assert j["sieve"]["cost"]["attempts"] == 0


@pytest.mark.anyio
async def test_identity_change_at_runtime_is_caught_per_request() -> None:
    sm = small_mock()
    async with running(make_app(sm)) as c:
        assert (await c.post(URL, json=body("a"))).json()["sieve"]["tier"] == 1
        sm.info = backend("small", build="swapped-build")  # server restarted with other build
        j = (await c.post(URL, json=body("a"))).json()
        assert j["sieve"]["tier"] == 2


@pytest.mark.anyio
async def test_backend_error_on_large_abstains_and_counts() -> None:
    sm = small_mock(accuracy=0.0)
    lg = large_mock(defect=MockDefect.RAISE)
    async with running(make_app(sm, lg, require_self_test=False)) as c:
        j = (await c.post(URL, json=body())).json()
        m = (await c.get("/metrics")).text
    assert j["sieve"]["decision"] == "abstain"
    assert 'sieve_backend_errors_total{tier="2"} 1.0' in m


@pytest.mark.anyio
async def test_no_large_tier_abstains_instead_of_answering_from_small() -> None:
    from tests.server.harness import make_settings

    assert make_settings().large is not None
    sm = small_mock(accuracy=0.0)
    from sieve.server import create_app

    app = create_app(make_settings(large=None), artifact=ART, small=sm)
    async with running(app) as c:
        assert (await c.get("/readyz")).status_code == 503
        j = (await c.post(URL, json=body())).json()
    assert j["sieve"]["decision"] == "abstain" and j["sieve"]["reason"] == "large_unavailable"


@pytest.mark.anyio
async def test_maintenance_loop_retests_after_identity_change() -> None:
    import asyncio

    sm = small_mock()
    async with running(make_app(sm, health_interval_s=0.05)) as c:
        assert (await c.get("/readyz")).status_code == 200
        sm.defect = MockDefect.NAN_LOGPROB  # silently starts misbehaving...
        sm.info = backend("small", build="restarted")  # ...and the identity changes
        for _ in range(60):
            await asyncio.sleep(0.05)
            j = (await c.get("/readyz")).json()
            if not j["tiers"]["1"]["logprobs_ok"]:
                break
        assert j["tiers"]["1"]["logprobs_ok"] is False and j["ready"] is False

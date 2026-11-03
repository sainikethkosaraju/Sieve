"""Router: cost accounting, guard, shadow, self-consistency, fail-closed handling."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from sieve.backends import MockDefect
from sieve.calibrate.store import compute_content_hash
from sieve.gate import Gate
from sieve.observe import MemoryAuditSink
from sieve.router import EscalationGuard, Router, Tier, UpstreamUnavailable
from sieve.router.cost import Attempt, CostLedger
from sieve.types import Action, CalibrationArtifact, SamplingParams
from tests.helpers import artifact
from tests.server.harness import ART, body, large_mock, make_app, running, small_mock

URL = "/v1/chat/completions"


def router(sm, lg, art: CalibrationArtifact = ART, guard: EscalationGuard | None = None) -> Router:  # type: ignore[no-untyped-def]
    return Router(Gate(art), Tier.healthy(sm), Tier.healthy(lg) if lg else None, guard=guard)


def rehash(a: CalibrationArtifact) -> CalibrationArtifact:
    return replace(a, content_hash=compute_content_hash(replace(a, content_hash="")))


# --- cost accounting -------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cost_includes_wasted_small_attempt_on_escalation() -> None:
    sm = small_mock(accuracy=0.0, latency_s=0.05, n_prompt_tokens=10, n_tokens=3)
    lg = large_mock(accuracy=1.0, latency_s=0.02, n_prompt_tokens=10, n_tokens=3)
    res = await router(sm, lg).run("q")
    c = res.cost
    assert res.served_tier == 2 and res.escalated
    assert c.n_attempts == 2
    assert c.small_seconds >= 0.045 and c.large_seconds >= 0.018
    assert c.total_seconds == pytest.approx(c.small_seconds + c.large_seconds)
    # the whole small attempt is waste; the large attempt is the useful one
    assert c.wasted_seconds == pytest.approx(c.small_seconds)
    assert c.wasted_tokens == 13 and c.total_tokens == 26


@pytest.mark.anyio
async def test_abstain_wastes_everything_and_small_answer_wastes_nothing() -> None:
    ab = await router(small_mock(accuracy=0), large_mock(accuracy=0)).run("q")
    assert ab.action is Action.ABSTAIN and ab.cost.wasted_tokens == ab.cost.total_tokens > 0
    ok = await router(small_mock(), large_mock()).run("q")
    assert ok.served_tier == 1 and ok.cost.wasted_seconds == 0 and ok.cost.wasted_tokens == 0
    assert ok.cost.large_seconds == 0 and ok.cost.n_attempts == 1


@pytest.mark.anyio
async def test_failed_and_cancelled_attempts_are_booked() -> None:
    sm = small_mock(defect=MockDefect.RAISE, latency_s=0.03)
    res = await router(sm, large_mock()).run("q")
    assert res.served_tier == 2 and res.cost.n_attempts == 2 and res.cost.small_seconds >= 0.025
    assert res.cost.wasted_seconds >= res.cost.small_seconds

    hang = small_mock(defect=MockDefect.HANG)
    ledger = CostLedger()
    t = asyncio.create_task(router(hang, large_mock()).run("q", ledger=ledger))
    await asyncio.sleep(0.05)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert [a.outcome for a in ledger.attempts] == ["cancelled"]
    assert ledger.attempts[0].seconds >= 0.04
    assert ledger.summary(None).wasted_seconds >= 0.04


def test_ledger_summary_arithmetic() -> None:
    led = CostLedger()
    led.add(Attempt(1, "primary", 1.0, 10, 5, "ok"))
    led.add(Attempt(1, "sample", 0.5, 10, 5, "ok"))
    led.add(Attempt(2, "primary", 2.0, 10, 8, "ok"))
    s = led.summary(2)
    assert s.total_seconds == 3.5 and s.wasted_seconds == 1.5 and s.wasted_tokens == 30
    assert led.summary(1).wasted_seconds == 2.5  # samples + the large attempt are not "useful"


@pytest.mark.anyio
async def test_cost_surfaced_in_response_audit_and_metrics() -> None:
    sink = MemoryAuditSink()
    sm = small_mock(accuracy=0.0, latency_s=0.03)
    async with running(make_app(sm, large_mock(), sink=sink)) as c:
        j = (await c.post(URL, json=body("q"))).json()
        text = (await c.get("/metrics")).text
    cost = j["sieve"]["cost"]
    assert cost["attempts"] == 2 and cost["wasted_tokens"] == 13 and cost["wasted_seconds"] >= 0.025
    assert j["usage"]["total_tokens"] == 26  # billed usage includes the wasted attempt
    assert sink.records[-1].cost["wasted_tokens"] == 13
    assert "sieve_wasted_seconds_total" in text and 'sieve_cost_seconds_total{tier="1"}' in text


# --- escalation guard -----------------------------------------------------------------------


def test_guard_caps_escalation_fraction() -> None:
    g = EscalationGuard(0.5, window=10)
    granted = [g.request_escalation() for _ in range(30)]
    assert sum(granted[:10]) == 5  # cap = floor(0.5 * 10) in a window
    # sliding-window invariant: never more than max_rate * window grants in any window
    assert all(sum(granted[i : i + 10]) <= 5 for i in range(21))
    assert g.rate <= 0.5 and g.shed_total == 30 - sum(granted) > 0
    assert EscalationGuard(1.0).request_escalation() is True


@pytest.mark.anyio
async def test_guard_sheds_instead_of_answering_from_small() -> None:
    sm, lg = small_mock(accuracy=0.0), large_mock()
    sink = MemoryAuditSink()
    app = make_app(sm, lg, sink=sink, guard={"max_escalation_rate": 0.5, "window": 10})
    async with running(app) as c:
        before = lg.calls
        js = [(await c.post(URL, json=body(f"q{i}"))).json() for i in range(20)]
        text = (await c.get("/metrics")).text
    served = [j for j in js if j["sieve"]["decision"] == "answer"]
    shed = [j for j in js if j["sieve"]["reason"] == "escalation_shed"]
    assert len(served) + len(shed) == 20 and 5 <= len(served) < 20 and shed
    assert [j["sieve"]["decision"] for j in js[:5]] == ["answer"] * 5  # burst allowance
    assert all(j["sieve"]["tier"] == 2 for j in served)
    assert all(j["sieve"]["decision"] == "abstain" for j in shed)
    assert lg.calls - before == len(served)  # shed requests cost no large-tier compute
    flags = [j["sieve"]["decision"] == "answer" for j in js]
    assert all(sum(flags[i : i + 10]) <= 5 for i in range(11))
    assert "sieve_escalation_guard_open 1.0" in text
    assert 'reason="escalation_shed"' in text


# --- shadow mode --------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_shadow_mode_serves_baseline_and_logs_would_decide() -> None:
    sink = MemoryAuditSink()
    sm = small_mock()  # would answer at tier 1
    lg = large_mock(answer_fn=lambda p: "baseline")
    app = make_app(sm, lg, sink=sink, mode="shadow")
    async with running(app) as c:
        j = (await c.post(URL, json=body("q"))).json()
        text = (await c.get("/metrics")).text
    assert j["choices"][0]["message"]["content"] == "baseline"  # large answer always served
    s = j["sieve"]
    assert s["mode"] == "shadow" and s["tier"] == 2
    assert s["shadow"] == {"action": "answer", "tier": 1, "reason": "confident"}
    rec = sink.records[-1]
    assert rec.mode == "shadow" and rec.shadow == s["shadow"]
    assert (
        'sieve_shadow_would_decide_total{decision="answer",reason="confident",tier="1"} 1.0' in text
    )


@pytest.mark.anyio
async def test_shadow_abstain_hypothetical_and_upstream_failure() -> None:
    res = await router(small_mock(accuracy=0), large_mock(accuracy=0)).run("q", shadow=True)
    assert res.action is Action.ANSWER and res.served_tier == 2  # baseline served regardless
    assert res.shadow_action is Action.ABSTAIN and res.shadow_tier is None
    with pytest.raises(UpstreamUnavailable):
        await router(small_mock(), large_mock(defect=MockDefect.ERROR_FINISH)).run("q", shadow=True)


# --- self-consistency path ---------------------------------------------------------------------


def consistency_artifact(k: int = 5) -> CalibrationArtifact:
    base = artifact(t1=0.7, t2=-0.3, signal1="agreement")
    cs = SamplingParams(temperature=0.7, top_p=0.95, seed=100, max_tokens=64)
    t1 = replace(base.tiers[0], n_samples=k, consistency_sampling=cs)
    return rehash(replace(base, tiers=(t1, base.tiers[1])))


@pytest.mark.anyio
async def test_self_consistency_answers_when_samples_agree_and_counts_cost() -> None:
    sm = small_mock(sample_stability=1.0)
    res = await router(sm, large_mock(), consistency_artifact()).run("q")
    assert res.served_tier == 1 and res.traces[0].signals.agreement == 1.0
    assert res.cost.n_attempts == 6 and sm.calls == 6  # 1 greedy + 5 samples
    assert res.cost.wasted_tokens == 5 * 13  # the five samples are overhead, not the answer
    assert {p.seed for p in sm.params if p.temperature > 0} == {100, 101, 102, 103, 104}


@pytest.mark.anyio
async def test_self_consistency_escalates_on_disagreement() -> None:
    sm = small_mock(accuracy=0.0, wrong_stability=0.0, seed=2)
    res = await router(sm, large_mock(), consistency_artifact()).run("q")
    assert res.served_tier == 2 and res.traces[0].decision.action is Action.ESCALATE
    assert res.cost.n_attempts == 7


@pytest.mark.anyio
async def test_self_consistency_truncated_answer_still_fails_closed() -> None:
    sm = small_mock(sample_stability=1.0, defect=MockDefect.TRUNCATED)
    res = await router(sm, large_mock(), consistency_artifact()).run("q")
    assert res.served_tier == 2 and res.traces[0].decision.action is Action.ESCALATE


@pytest.mark.anyio
async def test_unhealthy_tier_is_skipped_without_compute() -> None:
    sm = small_mock()
    r = Router(Gate(ART), Tier(sm), Tier.healthy(large_mock()))  # untested tier: logprobs_ok False
    res = await r.run("q")
    assert sm.calls == 0 and res.served_tier == 2 and res.traces[0].skipped_generation

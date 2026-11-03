"""OpenAI conformance, SSE, abstention, escalation, readiness, auth, config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sieve.calibrate.store import ArtifactIntegrityError, dumps_artifact
from sieve.observe import MemoryAuditSink, PinMismatchError
from sieve.server import ServerSettings, create_app
from sieve.server.cli import main
from tests.server.harness import (
    ART,
    body,
    bs,
    large_mock,
    make_app,
    make_settings,
    parse_sse,
    running,
    small_mock,
)

URL = "/v1/chat/completions"


@pytest.mark.anyio
async def test_non_stream_schema_and_headers() -> None:
    async with running(make_app()) as c:
        r = await c.post(URL, json=body(), headers={"X-Request-ID": "abc-123"})
        assert r.status_code == 200
        j = r.json()
        assert j["object"] == "chat.completion" and j["id"].startswith("chatcmpl-")
        assert isinstance(j["created"], int) and j["model"] == "sieve"
        ch = j["choices"][0]
        assert ch["index"] == 0 and ch["message"]["role"] == "assistant"
        assert isinstance(ch["message"]["content"], str) and ch["finish_reason"] == "stop"
        u = j["usage"]
        assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"] > 0
        s = j["sieve"]
        assert s["decision"] == "answer" and s["tier"] == 1 and s["escalated"] is False
        assert s["artifact_id"] == ART.artifact_id and s["reason"] == "confident"
        assert r.headers["x-sieve-decision"] == "answer" and r.headers["x-request-id"] == "abc-123"
        assert r.headers["x-sieve-tier"] == "1"


@pytest.mark.anyio
async def test_models_health_ready_and_error_envelope() -> None:
    async with running(make_app(model_aliases=("gpt-sieve",))) as c:
        m = (await c.get("/v1/models")).json()
        assert m["object"] == "list" and {d["id"] for d in m["data"]} == {"sieve", "gpt-sieve"}
        assert (await c.get("/healthz")).json() == {"status": "ok"}
        r = await c.get("/readyz")
        assert r.status_code == 200 and r.json()["ready"] is True
        nf = await c.get("/nope")
        assert nf.status_code == 404 and set(nf.json()["error"]) == {
            "message",
            "type",
            "param",
            "code",
        }
        assert (await c.get(URL)).status_code == 405
        ok = await c.post(URL, json=body(model="gpt-sieve"))
        assert ok.status_code == 200 and ok.json()["model"] == "sieve"


@pytest.mark.anyio
async def test_sse_stream_matches_non_stream() -> None:
    async with running(make_app()) as c:
        plain = (await c.post(URL, json=body("Q?"))).json()
        r = await c.post(URL, json=body("Q?", stream=True, stream_options={"include_usage": True}))
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["x-sieve-decision"] == "answer"
        ev = parse_sse(r.text)
        assert ev[-1] == "[DONE]"
        chunks = ev[:-1]
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert all(x["object"] == "chat.completion.chunk" for x in chunks)
        text = "".join(x["choices"][0]["delta"].get("content", "") for x in chunks if x["choices"])
        assert text == plain["choices"][0]["message"]["content"]
        finals = [x for x in chunks if x["choices"] and x["choices"][0]["finish_reason"]]
        assert len(finals) == 1 and finals[0]["choices"][0]["finish_reason"] == "stop"
        assert finals[0]["sieve"]["decision"] == "answer"
        assert chunks[-1]["usage"]["total_tokens"] > 0 and chunks[-1]["choices"] == []
        assert len({x["id"] for x in chunks}) == 1


@pytest.mark.anyio
async def test_long_answer_is_chunked_and_reassembles() -> None:
    sm = small_mock(answer_fn=lambda p: "word " * 40)
    async with running(make_app(sm)) as c:
        ev = parse_sse((await c.post(URL, json=body(stream=True))).text)
        pieces = [x["choices"][0]["delta"]["content"] for x in ev[1:-2]]
        assert len(pieces) > 3 and "".join(pieces) == "word " * 40


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_abstention_is_well_formed_and_distinct(stream: bool) -> None:
    sm, lg = small_mock(accuracy=0.0), large_mock(accuracy=0.0)
    async with running(make_app(sm, lg)) as c:
        r = await c.post(URL, json=body(stream=stream))
        assert r.status_code == 200
        if stream:
            ev = parse_sse(r.text)
            fin = next(x for x in ev[:-1] if x["choices"] and x["choices"][0]["finish_reason"])
            text = "".join(x["choices"][0]["delta"].get("content", "") for x in ev[:-1])
            fr, sv = fin["choices"][0]["finish_reason"], fin["sieve"]
            assert ev[0]["sieve"]["decision"] == "abstain"  # known before the first content byte
        else:
            j = r.json()
            text, fr, sv = (
                j["choices"][0]["message"]["content"],
                j["choices"][0]["finish_reason"],
                j["sieve"],
            )
        assert text == "I can't answer that reliably." and fr == "sieve_abstain"
        assert sv["decision"] == "abstain" and sv["tier"] is None and sv["escalated"] is True
        assert sv["reason"] == "low_confidence"
        assert r.headers["x-sieve-decision"] == "abstain"
        assert sm.prompts and "wrong" not in text  # backend text is never served


@pytest.mark.anyio
async def test_abstain_finish_reason_configurable() -> None:
    async with running(
        make_app(small_mock(accuracy=0), large_mock(accuracy=0), abstain_finish_reason="stop")
    ) as c:
        j = (await c.post(URL, json=body())).json()
        assert j["choices"][0]["finish_reason"] == "stop" and j["sieve"]["decision"] == "abstain"


@pytest.mark.anyio
async def test_escalation_path_serves_large_answer() -> None:
    sm, lg = small_mock(accuracy=0.0, latency_s=0.02), large_mock(accuracy=1.0)
    sink = MemoryAuditSink()
    async with running(make_app(sm, lg, sink=sink)) as c:
        j = (await c.post(URL, json=body("hard question"))).json()
    s = j["sieve"]
    assert s["decision"] == "answer" and s["tier"] == 2 and s["escalated"] is True
    assert j["choices"][0]["message"]["content"] == lg.truth(st_prompt(lg))
    rec = sink.records[-1]
    assert [t["tier"] for t in rec.tiers] == [1, 2]
    assert rec.tiers[0]["action"] == "escalate" and rec.tiers[0]["reason"] == "low_confidence"
    assert rec.tiers[1]["action"] == "answer"


def st_prompt(lg: Any) -> str:
    return str(lg.prompts[-1])


@pytest.mark.anyio
async def test_client_sampling_params_are_ignored_not_forwarded() -> None:
    sm = small_mock()
    async with running(make_app(sm)) as c:
        j = (await c.post(URL, json=body(temperature=1.5, top_p=0.2, seed=7, stop=["x"]))).json()
    assert sorted(j["sieve"]["ignored_params"]) == ["seed", "stop", "temperature", "top_p"]
    p = sm.params[-1]
    assert p.temperature == 0.0 and p.top_p == 1.0 and p.logit_bias == () and p.grammar is None


@pytest.mark.anyio
async def test_client_sampling_params_can_be_rejected() -> None:
    async with running(make_app(client_sampling="reject")) as c:
        r = await c.post(URL, json=body(temperature=0.7))
        assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_parameter"
        assert (await c.post(URL, json=body())).status_code == 200


@pytest.mark.anyio
async def test_max_tokens_one_cannot_inflate_confidence() -> None:
    """Red-team test 12: a tiny client budget truncates the answer, which is a defect."""
    sm, lg = small_mock(n_tokens=3), large_mock(n_tokens=3)
    async with running(make_app(sm, lg)) as c:
        j = (await c.post(URL, json=body(max_tokens=1))).json()
    assert sm.params[-1].max_tokens == 1
    assert j["sieve"]["decision"] == "abstain"
    assert j["sieve"]["reason"] == "logprob_defect"
    assert j["choices"][0]["message"]["content"] == "I can't answer that reliably."


@pytest.mark.anyio
async def test_auth_and_metrics_protection() -> None:
    key = "sk-test-12345678"
    async with running(make_app(api_keys=(key,))) as c:
        assert (await c.post(URL, json=body())).status_code == 401
        r = await c.post(URL, json=body(), headers={"Authorization": "Bearer wrong-key-123"})
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        ok = await c.post(URL, json=body(), headers={"Authorization": f"Bearer {key}"})
        assert ok.status_code == 200
        assert (await c.get("/metrics")).status_code == 401
        assert (
            await c.get("/metrics", headers={"Authorization": f"Bearer {key}"})
        ).status_code == 200
        assert (await c.get("/healthz")).status_code == 200
    async with running(make_app(api_keys=(key,), metrics_public=True)) as c:
        assert (await c.get("/metrics")).status_code == 200


def test_non_loopback_requires_api_key() -> None:
    with pytest.raises(ValidationError):
        make_settings(host="0.0.0.0")
    assert make_settings(host="0.0.0.0", api_keys=("longenoughkey",)).host == "0.0.0.0"
    with pytest.raises(ValidationError):
        make_settings(api_keys=("short",))


def test_settings_validation(tmp_path: Path) -> None:
    for bad in (
        {
            "small": bs("s").model_copy(
                update={"kind": "llamaserver", "url": "http://169.254.169.254"}
            )
        },
        {"small": bs("s").model_copy(update={"kind": "llamaserver", "url": "http://evil.example"})},
        {"mode": "shadow", "large": None},
        {"bogus_field": 1},
        {"limits": {"max_concurrency": 0}},
        {"model_name": "bad name\r\nX: y"},
    ):
        with pytest.raises((ValidationError, ValueError)):
            make_settings(**bad)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "artifact_path: a.json\n"
        "small: {kind: mock, model_id: s, model_hash: h, quant: q}\n"
        "large: {kind: mock, model_id: l, model_hash: h, quant: q}\n"
        "limits: {max_concurrency: 3}\n"
    )
    s = ServerSettings.from_yaml(cfg, port=9999)
    assert s.limits.max_concurrency == 3 and s.port == 9999 and s.mode == "enforce"
    cfg.write_text(
        "artifact_path: a.json\nsmall: {kind: llamaserver, model_id: s, model_hash: h, quant: q}\n"
    )
    with pytest.raises(ValidationError):
        ServerSettings.from_yaml(cfg)  # llamaserver without url


def test_env_nested_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIEVE_LIMITS__MAX_CONCURRENCY", "5")
    assert make_settings().limits.max_concurrency == 5


def test_artifact_integrity_and_pins_fail_at_startup(tmp_path: Path) -> None:
    good = tmp_path / "art.json"
    good.write_text(dumps_artifact(ART))
    s = make_settings(artifact_path=str(good))
    create_app(s, small=small_mock(), large=large_mock())  # loads from disk fine
    tampered = tmp_path / "bad.json"
    d = json.loads(good.read_text())
    d["tiers"][0]["threshold"] = -9.0  # loosen the gate without re-hashing
    tampered.write_text(json.dumps(d))
    with pytest.raises(ArtifactIntegrityError):
        create_app(
            make_settings(artifact_path=str(tampered)), small=small_mock(), large=large_mock()
        )
    with pytest.raises(PinMismatchError):
        create_app(
            make_settings(artifact_path=str(good), expected_artifact_id="0" * 16),
            small=small_mock(),
            large=large_mock(),
        )
    with pytest.raises(PinMismatchError):
        create_app(
            make_settings(artifact_path=str(good), expected_config_pin="deadbeef"),
            small=small_mock(),
            large=large_mock(),
        )
    s_ok = make_settings(artifact_path=str(good))
    pin = create_app(s_ok, small=small_mock(), large=large_mock()).state.sieve.pin
    create_app(
        make_settings(artifact_path=str(good), expected_config_pin=pin),
        small=small_mock(),
        large=large_mock(),
    )


def test_cli_refuses_to_start_on_bad_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"artifact_path: {tmp_path}/missing.json\n"
        "small: {kind: mock, model_id: s, model_hash: h, quant: q}\n"
        "large: {kind: mock, model_id: l, model_hash: h, quant: q}\n"
    )
    assert main(["serve", "--config", str(cfg)]) == 2
    assert "refusing to start" in capsys.readouterr().err

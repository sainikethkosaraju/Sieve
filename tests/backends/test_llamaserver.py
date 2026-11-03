from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import httpx
import pytest

from sieve.backends import BackendError, BackendUrlError, LlamaServerBackend
from sieve.signals import validate_completion
from sieve.types import DefectCode, SamplingParams
from tests.backends.fake_llama import FakeLlama, good_body, tok_entry

GREEDY = SamplingParams(temperature=0.0, max_tokens=16)


def make(fake: FakeLlama, **kw: Any) -> LlamaServerBackend:
    args: dict[str, Any] = {
        "model_id": "qwen",
        "model_hash": "h",
        "quant": "Q4_K_M",
        "kv_quant": "f16",
        "template_hash": "t",
        "allowed_hosts": ["llama.test"],
    }
    args.update(kw)
    return LlamaServerBackend(
        "http://llama.test", transport=httpx.ASGITransport(app=fake.app), **args
    )


@pytest.mark.anyio
async def test_maps_good_response_and_request_shape() -> None:
    fake = FakeLlama()
    be = make(fake)
    c = await be.generate("Q: capital?\nA:", GREEDY, n_probs=20)
    assert c.text == "Paris" and c.finish == "stop" and c.n_gen == 2 and c.n_prompt == 12
    assert c.tokens is not None and len(c.tokens) == 2
    assert c.tokens[0].top is not None and c.tokens[0].top[0][0] == c.tokens[0].token
    assert validate_completion(c) == ()
    body = fake.bodies[0]
    assert body["n_probs"] == 20 and body["temperature"] == 0.0 and body["stream"] is False
    assert body["cache_prompt"] is False and body["n_predict"] == 16
    assert fake.paths == ["/completion"]
    await be.aclose()


@pytest.mark.anyio
async def test_n_probs_is_clamped() -> None:
    fake = FakeLlama()
    be = make(fake)
    await be.generate("p", GREEDY, n_probs=1_000_000_000)
    assert fake.bodies[0]["n_probs"] <= 64
    await be.generate("p", GREEDY, n_probs=-5)
    assert fake.bodies[1]["n_probs"] == 0


@pytest.mark.anyio
async def test_no_client_headers_or_credentials_leak_upstream() -> None:
    fake = FakeLlama()
    be = make(fake)
    await be.generate("p", GREEDY, n_probs=5)
    h = fake.headers[0]
    assert not {"authorization", "cookie", "x-forwarded-for", "proxy-authorization"} & set(h)


@pytest.mark.anyio
async def test_legacy_schema_and_post_sampling() -> None:
    fake = FakeLlama()
    fake.make_body = lambda req: {
        "content": "ab",
        "stop_type": "eos",
        "tokens_predicted": 1,
        "tokens_evaluated": 3,
        "completion_probabilities": [
            {
                "content": "ab",
                "probs": [{"tok_str": "ab", "prob": 0.8}, {"tok_str": "x", "prob": 0.1}],
            }
        ],
    }
    c = await make(fake).generate("p", GREEDY, n_probs=2)
    assert c.tokens is not None
    assert math.isclose(c.tokens[0].logprob, math.log(0.8))
    assert validate_completion(c) == ()

    fake.make_body = lambda req: {
        "content": "ab",
        "stop_type": "eos",
        "tokens_predicted": 1,
        "completion_probabilities": [
            {
                "token": "ab",
                "prob": 0.5,
                "top_probs": [{"token": "ab", "prob": 0.5}, {"token": "x", "prob": 0.2}],
            }
        ],
    }
    p = SamplingParams(post_sampling_probs=True)
    c2 = await make(fake).generate("p", p, n_probs=2)
    assert c2.tokens is not None and math.isclose(c2.tokens[0].logprob, math.log(0.5))
    assert fake.bodies[-1]["post_sampling_probs"] is True


@pytest.mark.anyio
async def test_finish_reasons() -> None:
    fake = FakeLlama()
    fake.make_body = lambda req: {**good_body(), "stop_type": "limit"}
    assert (await make(fake).generate("p", GREEDY, n_probs=5)).finish == "length"
    fake.make_body = lambda req: {**good_body(), "truncated": True}
    assert (await make(fake).generate("p", GREEDY, n_probs=5)).finish == "length"
    fake.make_body = lambda req: {**good_body(), "stop_type": "weird"}
    assert (await make(fake).generate("p", GREEDY, n_probs=5)).finish == "error"


def _defect_cases() -> list[tuple[str, Any, DefectCode]]:
    def nan(b: dict[str, Any]) -> dict[str, Any]:
        b["completion_probabilities"][0]["logprob"] = math.nan
        return b

    def positive(b: dict[str, Any]) -> dict[str, Any]:
        b["completion_probabilities"][0]["logprob"] = 0.7
        return b

    def string_lp(b: dict[str, Any]) -> dict[str, Any]:
        b["completion_probabilities"][0]["logprob"] = "-0.1"
        return b

    def dropped(b: dict[str, Any]) -> dict[str, Any]:
        b["completion_probabilities"].pop()
        return b

    def missing(b: dict[str, Any]) -> dict[str, Any]:
        del b["completion_probabilities"]
        return b

    def utf8(b: dict[str, Any]) -> dict[str, Any]:
        b["content"] = "Paé"
        b["completion_probabilities"] = [tok_entry("Pa", -0.1), tok_entry("�", -0.1)]
        return b

    return [
        ("nan", nan, DefectCode.NAN_LOGPROB),
        ("positive", positive, DefectCode.POSITIVE_LOGPROB),
        ("string", string_lp, DefectCode.NAN_LOGPROB),
        ("dropped", dropped, DefectCode.MISSING_LOGPROBS),
        ("missing", missing, DefectCode.MISSING_LOGPROBS),
        ("utf8", utf8, DefectCode.UTF8_GAP),
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(("name", "mutate", "code"), _defect_cases(), ids=lambda x: str(x)[:12])
async def test_upstream_defects_are_mapped_not_repaired(
    name: str, mutate: Any, code: DefectCode
) -> None:
    fake = FakeLlama()
    fake.make_body = lambda req: mutate(good_body())
    c = await make(fake).generate("p", GREEDY, n_probs=5)
    assert code in validate_completion(c), name


@pytest.mark.anyio
async def test_http_errors_redirects_garbage_and_timeouts() -> None:
    fake = FakeLlama()
    be = make(fake)
    fake.completion_status = 503
    with pytest.raises(BackendError, match="503"):
        await be.generate("secret-prompt-text", GREEDY, n_probs=5)
    fake.completion_status = 200
    fake.redirect_to = "http://169.254.169.254/latest"
    with pytest.raises(BackendError, match="redirect"):
        await be.generate("p", GREEDY, n_probs=5)
    fake.redirect_to = None
    fake.raw = lambda req: "not json"
    with pytest.raises(BackendError, match="invalid JSON"):
        await be.generate("p", GREEDY, n_probs=5)
    fake.raw = lambda req: json.dumps([1, 2])
    with pytest.raises(BackendError):
        await be.generate("p", GREEDY, n_probs=5)
    fake.raw = lambda req: json.dumps({"no_content": 1})
    with pytest.raises(BackendError, match="content"):
        await be.generate("p", GREEDY, n_probs=5)
    # error messages never carry the prompt
    fake.completion_status = 500
    with pytest.raises(BackendError) as ei:
        await be.generate("CANARY-PROMPT", GREEDY, n_probs=5)
    assert "CANARY" not in str(ei.value)


@pytest.mark.anyio
async def test_timeouts_become_backend_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    be = LlamaServerBackend(
        "http://llama.test",
        model_id="m",
        model_hash="h",
        quant="q",
        kv_quant="f16",
        template_hash="t",
        allowed_hosts=["llama.test"],
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(BackendError, match="ReadTimeout"):
        await be.generate("p", GREEDY, n_probs=5)
    assert await be.health() is False


@pytest.mark.anyio
async def test_cancellation_reaches_upstream() -> None:
    fake = FakeLlama()
    fake.delay_s = 5.0
    be = make(fake)
    t = asyncio.create_task(be.generate("p", GREEDY, n_probs=5))
    await asyncio.sleep(0.1)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    await asyncio.sleep(0.05)
    assert fake.cancelled == 1


@pytest.mark.anyio
async def test_identity_from_props_and_model_file_pin() -> None:
    fake = FakeLlama()
    be = make(fake, build="b9999-abc1234", expected_model_file="qwen2.5-3b-q4_k_m.gguf")
    assert (await be.refresh()) is True  # first read fills build / model file
    assert be.info.build == "b9999-abc1234" and be.info.model_hash == "h"
    assert (await be.refresh()) is False
    fake.props["build_info"] = "b1-other"
    assert (await be.refresh()) is True and be.info.build == "b1-other"
    fake.props["model_path"] = "/models/evil.gguf"
    assert (await be.refresh()) is True
    assert be.info.model_hash.startswith("unverified:evil")
    assert await be.health() is True
    fake.health_status = 503
    assert await be.health() is False


def test_constructor_enforces_url_allow_list() -> None:
    for bad in (
        "http://169.254.169.254",
        "file:///x",
        "http://evil.example",
        "http://127.0.0.1@evil.example",
    ):
        with pytest.raises(BackendUrlError):
            LlamaServerBackend(
                bad,
                model_id="m",
                model_hash="h",
                quant="q",
                kv_quant="f16",
                template_hash="t",
            )

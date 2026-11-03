from __future__ import annotations

import asyncio
import json
import math

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from sieve.backends import BackendError, MockBackend, MockDefect, OllamaBackend
from sieve.signals import logprob_signals, validate_completion
from sieve.types import DefectCode, SamplingParams
from tests.helpers import backend as info

G = SamplingParams(temperature=0.0, max_tokens=16)


@pytest.mark.anyio
async def test_mock_is_deterministic_and_has_signal() -> None:
    m = MockBackend(info("small"), accuracy=0.7, seed=3)
    a = await m.generate("question 1", G, n_probs=5)
    b = await m.generate("question 1", G, n_probs=5)
    assert a == b
    assert validate_completion(a) == ()
    ok: list[bool] = []
    scores: dict[bool, list[float]] = {True: [], False: []}
    for i in range(400):
        p = f"q{i}"
        c = await m.generate(p, G, n_probs=5)
        s = logprob_signals(c, m.info).mean_logprob
        assert s is not None
        scores[m.is_correct(p)].append(s)
        ok.append(m.is_correct(p))
    assert 0.62 < sum(ok) / len(ok) < 0.78
    assert min(scores[True]) > max(scores[False])  # separable by construction


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("defect", "code"),
    [
        (MockDefect.NAN_LOGPROB, DefectCode.NAN_LOGPROB),
        (MockDefect.POSITIVE_LOGPROB, DefectCode.POSITIVE_LOGPROB),
        (MockDefect.DROPPED_LOGPROB, DefectCode.MISSING_LOGPROBS),
        (MockDefect.NO_LOGPROBS, DefectCode.MISSING_LOGPROBS),
        (MockDefect.UTF8_GAP, DefectCode.UTF8_GAP),
        (MockDefect.TRUNCATED, DefectCode.TRUNCATED),
        (MockDefect.ERROR_FINISH, DefectCode.BACKEND_ERROR),
        (MockDefect.GREEDY_MISMATCH, DefectCode.GREEDY_MISMATCH),
        (MockDefect.NO_TOP, DefectCode.TOP_LOGPROBS_MISSING),
    ],
)
async def test_injected_defects_are_detected(defect: MockDefect, code: DefectCode) -> None:
    m = MockBackend(info("small"), defect=defect)
    assert code in validate_completion(await m.generate("p", G, n_probs=5))


@pytest.mark.anyio
async def test_defect_after_and_raise_and_hang_cancel() -> None:
    m = MockBackend(info("small"), defect=MockDefect.RAISE, defect_after=1)
    await m.generate("p", G, n_probs=5)
    with pytest.raises(BackendError):
        await m.generate("p", G, n_probs=5)
    h = MockBackend(info("small"), defect=MockDefect.HANG)
    t = asyncio.create_task(h.generate("p", G, n_probs=5))
    await asyncio.sleep(0.05)
    assert h.in_flight == 1
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert h.cancelled == 1 and h.in_flight == 0


@pytest.mark.anyio
async def test_samples_vary_only_when_sampling() -> None:
    m = MockBackend(info("small"), accuracy=1.0, sample_stability=0.5, seed=1)
    outs = {
        (await m.generate("p", SamplingParams(temperature=0.7, seed=i), n_probs=0)).text
        for i in range(30)
    }
    assert len(outs) > 1


def ollama_app(with_logprobs: bool) -> Starlette:
    async def gen(request: Request) -> JSONResponse:
        body = await request.json()
        assert body["raw"] is True and body["stream"] is False
        d: dict[str, object] = {
            "response": "Paris",
            "done": True,
            "done_reason": "stop",
            "eval_count": 1,
            "prompt_eval_count": 7,
        }
        if with_logprobs and body.get("logprobs"):
            d["logprobs"] = [
                {
                    "token": "Paris",
                    "logprob": math.log(0.9),
                    "top_logprobs": [
                        {"token": "Paris", "logprob": math.log(0.9)},
                        {"token": "Lyon", "logprob": math.log(0.05)},
                    ],
                }
            ]
        return JSONResponse(d)

    async def version(request: Request) -> JSONResponse:
        return JSONResponse({"version": "0.13.0"})

    return Starlette(
        routes=[Route("/api/generate", gen, methods=["POST"]), Route("/api/version", version)]
    )


def ollama(app: Starlette) -> OllamaBackend:
    return OllamaBackend(
        "http://ollama.test",
        model_id="qwen",
        model_hash="h",
        quant="q",
        kv_quant="f16",
        template_hash="t",
        allowed_hosts=["ollama.test"],
        transport=httpx.ASGITransport(app=app),
    )


@pytest.mark.anyio
async def test_ollama_with_and_without_logprobs() -> None:
    c = await ollama(ollama_app(True)).generate("p", G, n_probs=5)
    assert c.tokens is not None and validate_completion(c) == ()
    c2 = await ollama(ollama_app(False)).generate("p", G, n_probs=5)
    assert c2.tokens is None
    assert DefectCode.MISSING_LOGPROBS in validate_completion(c2)  # gate would fail closed


@pytest.mark.anyio
async def test_ollama_refresh_sets_build() -> None:
    be = ollama(ollama_app(True))
    assert await be.refresh() is True
    assert be.info.build == "ollama-0.13.0"
    assert json.dumps(be.info.build)

from __future__ import annotations

import math
from typing import Any

import httpx
import pytest

from sieve.backends import LlamaServerBackend, MockBackend, MockDefect, run_logprob_self_test
from tests.backends.fake_llama import FakeLlama, good_body, tok_entry
from tests.helpers import backend as info


def llama(fake: FakeLlama) -> LlamaServerBackend:
    return LlamaServerBackend(
        "http://llama.test",
        model_id="m",
        model_hash="h",
        quant="q",
        kv_quant="f16",
        template_hash="t",
        allowed_hosts=["llama.test"],
        transport=httpx.ASGITransport(app=fake.app),
    )


@pytest.mark.anyio
async def test_good_server_passes() -> None:
    res = await run_logprob_self_test(llama(FakeLlama()))
    assert res.ok, res.failed
    assert {c.name for c in res.checks} >= {
        "finite_nonpositive",
        "count_matches_tokens",
        "top_k_contains_chosen",
        "utf8_reassembles",
        "greedy_deterministic",
    }


def _mut(fn: Any) -> Any:
    def make(req: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = fn(good_body())
        return out

    return make


def _nan(b: dict[str, Any]) -> dict[str, Any]:
    b["completion_probabilities"][0]["logprob"] = math.nan
    return b


def _pos(b: dict[str, Any]) -> dict[str, Any]:
    b["completion_probabilities"][1]["logprob"] = 0.3
    return b


def _drop(b: dict[str, Any]) -> dict[str, Any]:
    b["completion_probabilities"].pop()
    return b


def _no_chosen(b: dict[str, Any]) -> dict[str, Any]:
    b["completion_probabilities"][0]["top_logprobs"] = [
        {"token": "<x>", "logprob": -0.1},
        {"token": "<y>", "logprob": -2.0},
    ]
    return b


def _utf8(b: dict[str, Any]) -> dict[str, Any]:
    b["content"] = "Paé"
    b["completion_probabilities"] = [tok_entry("Pa", -0.1), tok_entry("�", -0.1)]
    return b


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("mutate", "failing"),
    [
        (_nan, "finite_nonpositive"),
        (_pos, "finite_nonpositive"),
        (_drop, "count_matches_tokens"),
        (_no_chosen, "top_k_contains_chosen"),
        (_utf8, "utf8_reassembles"),
    ],
)
async def test_defective_server_fails(mutate: Any, failing: str) -> None:
    fake = FakeLlama()
    fake.make_body = _mut(mutate)
    res = await run_logprob_self_test(llama(fake))
    assert not res.ok
    assert failing in res.failed


@pytest.mark.anyio
async def test_nondeterministic_server_fails() -> None:
    fake = FakeLlama()
    n = {"i": 0}

    def drift(req: dict[str, Any]) -> dict[str, Any]:
        n["i"] += 1
        return good_body(p=0.9 - 0.01 * n["i"])

    fake.make_body = drift
    res = await run_logprob_self_test(llama(fake))
    assert "greedy_deterministic" in res.failed


@pytest.mark.anyio
async def test_unreachable_server_fails_without_raising() -> None:
    fake = FakeLlama()
    fake.completion_status = 500
    res = await run_logprob_self_test(llama(fake))
    assert not res.ok and res.failed == ("reachable",)


@pytest.mark.anyio
async def test_mock_backend_passes_and_defects_fail() -> None:
    assert (await run_logprob_self_test(MockBackend(info("small")))).ok
    for d in (
        MockDefect.NAN_LOGPROB,
        MockDefect.POSITIVE_LOGPROB,
        MockDefect.DROPPED_LOGPROB,
        MockDefect.NO_LOGPROBS,
        MockDefect.UTF8_GAP,
        MockDefect.NONDETERMINISTIC,
        MockDefect.GREEDY_MISMATCH,
        MockDefect.NO_TOP,
        MockDefect.RAISE,
    ):
        res = await run_logprob_self_test(MockBackend(info("small"), defect=d))
        assert not res.ok, d


class _Utf8OnlyDefective:
    """Clean on ASCII probes; drops logprobs only on the non-ASCII probe (llama.cpp 31451)."""

    def __init__(self) -> None:
        self._ok = MockBackend(info("small"))
        self._bad = MockBackend(info("small"), defect=MockDefect.UTF8_GAP)
        self.info = self._ok.info

    async def generate(self, prompt: str, params: Any, *, n_probs: int) -> Any:
        from sieve.backends.selftest import UTF8_PROBE_PROMPT

        be = self._bad if prompt == UTF8_PROBE_PROMPT else self._ok
        return await be.generate(prompt, params, n_probs=n_probs)


@pytest.mark.anyio
async def test_utf8_defect_is_advisory_when_not_strict() -> None:
    """Eval runs tolerate the multi-token UTF-8 logprob drop; per-item fail-closed remains."""
    strict = await run_logprob_self_test(_Utf8OnlyDefective())
    relaxed = await run_logprob_self_test(
        _Utf8OnlyDefective(),
        strict_utf8=False,
    )
    assert not strict.ok
    assert relaxed.ok and relaxed.advisories == ("utf8_reassembles",)

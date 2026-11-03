"""B1's llama-server parser against the REAL llama.cpp @ fac3a89 response shapes."""

from __future__ import annotations

import math

import pytest

from sieve.backends import LlamaServerBackend
from sieve.backends.llamaserver import parse_completion_probabilities
from sieve.backends.selftest import run_logprob_self_test
from sieve.signals import validate_completion
from sieve.types import DefectCode, SamplingParams
from tests.integration import real_fixtures as rf

GREEDY = SamplingParams(temperature=0.0, max_tokens=24)


def make(body: dict[str, object]) -> LlamaServerBackend:
    return LlamaServerBackend(
        "http://127.0.0.1:8081",
        model_id="qwen",
        model_hash="h",
        quant="Q4_K_M",
        kv_quant="f16",
        template_hash="t",
        transport=rf.fixed_transport(body),
    )


def test_evidence_file_has_the_nested_schema() -> None:
    e = rf.real_step()["first_prob_entry"]
    assert {"id", "token", "bytes", "logprob", "top_logprobs"} <= set(e)
    assert all("token" in t and "logprob" in t for t in e["top_logprobs"])
    assert e["top_logprobs"][0]["token"] == e["token"]  # chosen token is the top-1 under greedy


@pytest.mark.parametrize("name", ["small_default", "large_default"])
def test_real_completion_entry_maps_and_validates(name: str) -> None:
    body = rf.completion_body(name)
    entry = body["completion_probabilities"][0]
    comp = LlamaServerBackend.to_completion(body, GREEDY, 0.1)
    assert comp.finish == "stop" and comp.n_gen == 1
    assert comp.tokens is not None and len(comp.tokens) == 1
    t = comp.tokens[0]
    assert t.token == entry["token"] and t.logprob == entry["logprob"] < 0
    assert t.top is not None and len(t.top) == len(entry["top_logprobs"]) == 5
    assert [x for x, _ in t.top] == [x["token"] for x in entry["top_logprobs"]]
    lps = [lp for _, lp in t.top]
    assert lps == sorted(lps, reverse=True) and all(math.isfinite(x) and x <= 0 for x in lps)
    assert sum(math.exp(x) for x in lps) <= 1.0 + 1e-9
    assert comp.t_first_token_s == pytest.approx(body["timings"]["prompt_ms"] / 1000.0)
    assert validate_completion(comp) == ()


def test_real_large_entry_shows_d016_the_prefix() -> None:
    """D-016: raw prompts open with ' The' on the 7B; the parser must preserve that faithfully."""
    comp = LlamaServerBackend.to_completion(rf.completion_body("large_default"), GREEDY, 0.0)
    assert comp.tokens is not None and comp.tokens[0].token == " The"
    assert comp.tokens[0].top is not None and comp.tokens[0].top[1][0] == " Paris"


def test_real_chat_logprobs_content_maps_with_same_parser() -> None:
    raw = rf.chat_logprob_content()
    toks = parse_completion_probabilities(raw)
    assert toks is not None and len(toks) == 1
    assert toks[0].token == "Paris" and toks[0].logprob == raw[0]["logprob"]
    assert toks[0].top is not None and len(toks[0].top) == 5
    # the multi-byte token in the real top-k list survives untouched
    assert any(tok == "巴黎" for tok, _ in toks[0].top)


def test_real_entry_with_dropped_top_logprobs_is_flagged_not_repaired() -> None:
    body = rf.completion_body()
    del body["completion_probabilities"][0]["top_logprobs"]
    comp = LlamaServerBackend.to_completion(body, GREEDY, 0.0)
    assert DefectCode.TOP_LOGPROBS_MISSING in validate_completion(comp)


def test_real_entry_with_greedy_violation_is_flagged() -> None:
    body = rf.completion_body("large_default")
    e = body["completion_probabilities"][0]
    # chosen token is the runner-up while the top-k list stays sorted: not greedy
    e["token"], e["logprob"] = e["top_logprobs"][1]["token"], e["top_logprobs"][1]["logprob"]
    body["content"] = e["token"]
    comp = LlamaServerBackend.to_completion(body, GREEDY, 0.0)
    assert comp.tokens is not None
    assert DefectCode.GREEDY_MISMATCH in validate_completion(comp)


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["small_default", "large_default"])
async def test_startup_self_test_accepts_real_schema(name: str) -> None:
    be = make(rf.completion_body(name))
    res = await run_logprob_self_test(be, n_probs=20, repeats=3)
    assert res.ok, [(c.name, c.detail) for c in res.checks if not c.ok]
    assert {c.name for c in res.checks} >= {
        "finite_nonpositive",
        "count_matches_tokens",
        "top_k_contains_chosen",
        "utf8_reassembles",
        "greedy_deterministic",
    }
    await be.aclose()


@pytest.mark.anyio
async def test_self_test_rejects_real_entry_with_nan_logprob() -> None:
    body = rf.completion_body()
    body["completion_probabilities"][0]["logprob"] = "nan"
    res = await run_logprob_self_test(make(body), n_probs=20, repeats=2)
    assert not res.ok and "finite_nonpositive" in res.failed


@pytest.mark.anyio
async def test_request_shape_sent_to_llama_server() -> None:
    import httpx

    seen: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(req.content))
        return httpx.Response(200, json=rf.completion_body())

    be = LlamaServerBackend(
        "http://127.0.0.1:8081",
        model_id="q",
        model_hash="h",
        quant="Q4_K_M",
        kv_quant="f16",
        template_hash="t",
        transport=httpx.MockTransport(handler),
    )
    await be.generate("Question: x\nAnswer:", GREEDY, n_probs=20)
    assert seen[0]["n_probs"] == 20 and seen[0]["temperature"] == 0.0
    assert seen[0]["cache_prompt"] is False and seen[0]["post_sampling_probs"] is False

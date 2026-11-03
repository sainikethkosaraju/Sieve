"""Fixtures derived from the REAL T4 llama-server smoke evidence (D-013..D-016).

``results/evidence/t4_llamacpp_smoke_2026-11-03.json`` stored, per server run, the first
``completion_probabilities`` entry of a ``/completion`` call (with nested ``top_logprobs``) and
the first ``logprobs.content`` entry of a ``/v1/chat/completions`` call, verbatim from
llama.cpp @ fac3a89 on a Tesla T4, plus the ``timings`` block. The token entries below are
those real objects unchanged. The response *envelope* (``stop_type``, ``tokens_predicted``,
``tokens_evaluated``) was not kept in the evidence and is reconstructed here from the documented
llama-server schema; tests that need a second token build it explicitly and say so.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx

EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "evidence"
    / "t4_llamacpp_smoke_2026-11-03.json"
)


def load_evidence() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    return data


def real_step(name: str = "small_default") -> dict[str, Any]:
    step: dict[str, Any] = load_evidence()["steps"][name]
    return step


def completion_body(name: str = "small_default", *, stop_type: str = "eos") -> dict[str, Any]:
    """A one-token ``/completion`` body: real first entry + real timings, rebuilt envelope."""
    s = real_step(name)
    entry = copy.deepcopy(s["first_prob_entry"])
    return {
        "content": entry["token"],
        "stop_type": stop_type,
        "truncated": False,
        "tokens_predicted": 1,
        "tokens_evaluated": s["timings"]["prompt_n"],
        "completion_probabilities": [entry],
        "timings": copy.deepcopy(s["timings"]),
    }


def chat_logprob_content(name: str = "small_default") -> list[dict[str, Any]]:
    """The real chat ``choices[0].logprobs.content`` first entry (as a one-element list)."""
    return [copy.deepcopy(real_step(name)["chat_first_logprob"])]


def derived_stop_entry(first: dict[str, Any]) -> dict[str, Any]:
    """DERIVED (not real): a trailing '\\n' token entry shaped like the real ones."""
    e = copy.deepcopy(first)
    e.update({"id": 198, "token": "\n", "bytes": [10], "logprob": -0.5})
    e["top_logprobs"] = [
        {"id": 198, "token": "\n", "bytes": [10], "logprob": -0.5},
        *copy.deepcopy(first["top_logprobs"][1:]),
    ]
    return e


def fixed_transport(body: dict[str, Any]) -> httpx.MockTransport:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)

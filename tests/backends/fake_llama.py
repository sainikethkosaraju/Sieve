"""In-process fake llama-server (ASGI) for adapter tests. Records what it receives."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route


def tok_entry(tok: str, lp: float, alts: int = 3) -> dict[str, Any]:
    top = [{"id": 1, "token": tok, "logprob": lp, "bytes": list(tok.encode())}]
    rest = max(1.0 - math.exp(lp), 1e-6) / (alts + 1)
    for i in range(alts - 1):
        top.append({"id": 2 + i, "token": f"<alt{i}>", "logprob": math.log(rest / (i + 1))})
    return {"id": 1, "token": tok, "logprob": lp, "bytes": list(tok.encode()), "top_logprobs": top}


def good_body(text: str = "Paris", p: float = 0.9) -> dict[str, Any]:
    pieces = [text[:3], text[3:]] if len(text) > 3 else [text]
    probs = [tok_entry(t, math.log(p)) for t in pieces]
    return {
        "content": text,
        "stop": True,
        "stop_type": "eos",
        "truncated": False,
        "tokens_predicted": len(pieces),
        "tokens_evaluated": 12,
        "timings": {"prompt_ms": 5.0, "predicted_ms": 10.0},
        "completion_probabilities": probs,
    }


class FakeLlama:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.paths: list[str] = []
        self.make_body: Callable[[dict[str, Any]], dict[str, Any]] = lambda req: good_body()
        self.raw: Callable[[dict[str, Any]], str] | None = None
        self.props: dict[str, Any] = {
            "build_info": "b9999-abc1234",
            "model_path": "/models/qwen2.5-3b-q4_k_m.gguf",
            "chat_template": "{{ x }}",
        }
        self.health_status = 200
        self.completion_status = 200
        self.redirect_to: str | None = None
        self.delay_s = 0.0
        self.cancelled = 0
        self.calls = 0
        self.app = Starlette(
            routes=[
                Route("/completion", self._completion, methods=["POST"]),
                Route("/props", self._props),
                Route("/health", self._health),
            ]
        )

    def _rec(self, request: Request) -> None:
        self.paths.append(request.url.path)
        self.headers.append({k.lower(): v for k, v in request.headers.items()})

    async def _completion(self, request: Request) -> Response:
        self._rec(request)
        self.calls += 1
        body = await request.json()
        self.bodies.append(body)
        if self.redirect_to is not None:
            return Response(status_code=302, headers={"location": self.redirect_to})
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self.completion_status != 200:
            return PlainTextResponse("boom", status_code=self.completion_status)
        if self.raw is not None:
            return Response(self.raw(body), media_type="application/json")
        # json.dumps emits NaN/Infinity tokens like a buggy server would
        return Response(json.dumps(self.make_body(body)), media_type="application/json")

    async def _props(self, request: Request) -> Response:
        self._rec(request)
        return Response(json.dumps(self.props), media_type="application/json")

    async def _health(self, request: Request) -> Response:
        self._rec(request)
        return Response(
            json.dumps({"status": "ok"}),
            status_code=self.health_status,
            media_type="application/json",
        )

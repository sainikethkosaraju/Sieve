"""Shared builders for server tests (mock backends + ASGI transport)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI
from prometheus_client import CollectorRegistry

from sieve.backends import MockBackend
from sieve.observe import AuditSink, MemoryAuditSink
from sieve.server import BackendSettings, ServerSettings, create_app
from sieve.types import CalibrationArtifact
from tests.helpers import artifact, backend

ART = artifact(t1=-0.5, t2=-0.3)  # answer iff mean logprob >= threshold


def small_mock(**kw: Any) -> MockBackend:
    return MockBackend(backend("small"), **kw)


def large_mock(**kw: Any) -> MockBackend:
    return MockBackend(backend("large", model_hash="h2"), **kw)


def bs(name: str) -> BackendSettings:
    return BackendSettings(kind="mock", model_id=name, model_hash="h", quant="q")


def make_settings(**over: Any) -> ServerSettings:
    base: dict[str, Any] = {
        "artifact_path": "unused.json",
        "small": bs("small"),
        "large": bs("large"),
        "health_interval_s": 0,
        "guard": {"max_escalation_rate": 1.0, "window": 200},
    }
    base.update(over)
    return ServerSettings(**base)


def make_app(
    small: MockBackend | None = None,
    large: MockBackend | None = None,
    *,
    art: CalibrationArtifact = ART,
    sink: AuditSink | None = None,
    **over: Any,
) -> FastAPI:
    return create_app(
        make_settings(**over),
        artifact=art,
        small=small or small_mock(),
        large=large or large_mock(),
        sink=sink if sink is not None else MemoryAuditSink(),
        registry=CollectorRegistry(),
    )


@contextlib.asynccontextmanager
async def running(app: FastAPI, **client_kw: Any) -> AsyncIterator[httpx.AsyncClient]:
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, **client_kw), base_url="http://sieve.test"
        ) as c,
    ):
        yield c


def body(content: str = "What is 2+2?", **kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"model": "sieve", "messages": [{"role": "user", "content": content}]}
    d.update(kw)
    return d


def parse_sse(text: str) -> list[Any]:
    events: list[Any] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        assert block.startswith("data: "), block
        payload = block[len("data: ") :]
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


class RawCall:
    """Drive the ASGI app directly so a client disconnect can be simulated."""

    def __init__(
        self, app: FastAPI, payload: dict[str, Any], headers: dict[str, str] | None = None
    ):
        raw = json.dumps(payload).encode()
        hs = [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())]
        hs += [(b"host", b"sieve.test")]
        hs += [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
        self.scope: dict[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "scheme": "http",
            "headers": hs,
            "client": ("127.0.0.1", 5000),
            "server": ("sieve.test", 80),
        }
        self.body = raw
        self.sent: list[dict[str, Any]] = []
        self.disconnect = asyncio.Event()
        self._first = True
        self.task = asyncio.create_task(app(self.scope, self._receive, self._send))

    async def _receive(self) -> dict[str, Any]:
        if self._first:
            self._first = False
            return {"type": "http.request", "body": self.body, "more_body": False}
        await self.disconnect.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message: Any) -> None:
        self.sent.append(message)

    @property
    def status(self) -> int | None:
        for m in self.sent:
            if m["type"] == "http.response.start":
                return int(m["status"])
        return None

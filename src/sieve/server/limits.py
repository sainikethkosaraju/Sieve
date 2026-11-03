"""Concurrency limiting with bounded queue (429 backpressure), body reading with cap,
and client-disconnect propagation."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Coroutine
from typing import Any, TypeVar

from starlette.requests import Request

from sieve.server.errors import ClientDisconnected, OpenAIError

T = TypeVar("T")


class Backpressure(Exception):
    pass


class ConcurrencyLimiter:
    """At most ``max_active`` concurrent requests, ``max_queue`` waiting, wait <= timeout."""

    def __init__(self, max_active: int, max_queue: int, queue_timeout_s: float) -> None:
        self.max_active = max_active
        self.max_queue = max_queue
        self.queue_timeout_s = queue_timeout_s
        self._sem = asyncio.Semaphore(max_active)
        self.active = 0
        self.queued = 0

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        if self._sem.locked():
            if self.queued >= self.max_queue:
                raise Backpressure
            self.queued += 1
            try:
                async with asyncio.timeout(self.queue_timeout_s):
                    await self._sem.acquire()
            except TimeoutError:
                raise Backpressure from None
            finally:
                self.queued -= 1
        else:
            await self._sem.acquire()
        self.active += 1
        try:
            yield
        finally:
            self.active -= 1
            self._sem.release()


async def read_body(request: Request, max_bytes: int, timeout_s: float) -> bytes:
    """Read the body with a hard size cap (also for chunked uploads) and a time limit."""
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > max_bytes:
                raise OpenAIError(
                    413, "Request body too large", code="body_too_large", reject="too_large"
                )
        except ValueError:
            raise OpenAIError(400, "Invalid Content-Length", code="bad_content_length") from None
    buf = bytearray()
    try:
        async with asyncio.timeout(timeout_s):
            async for chunk in request.stream():
                buf += chunk
                if len(buf) > max_bytes:
                    raise OpenAIError(
                        413, "Request body too large", code="body_too_large", reject="too_large"
                    )
    except TimeoutError:
        raise OpenAIError(408, "Timed out reading request body", code="body_timeout") from None
    return bytes(buf)


async def run_cancellable(
    request: Request, coro: Coroutine[Any, Any, T], poll_s: float = 0.05
) -> T:
    """Run ``coro``; if the client disconnects first, cancel it (and so the backend call).

    Also cancels ``coro`` if this coroutine itself is cancelled (deadline).
    """
    task: asyncio.Future[T] = asyncio.ensure_future(coro)

    async def watch() -> None:
        while True:
            if await request.is_disconnected():
                return
            await asyncio.sleep(poll_s)

    watcher = asyncio.ensure_future(watch())
    try:
        waiting: set[asyncio.Future[Any]] = {task, watcher}
        done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return task.result()
        raise ClientDisconnected
    finally:
        watcher.cancel()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, watcher, return_exceptions=True)

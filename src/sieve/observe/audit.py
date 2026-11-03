"""Audit trail: one JSONL record per decision.

Privacy defaults (red-team kill test 8): prompts and responses are NOT written. Only
decision metadata is: decision, reason code, tier, signal values, artifact id, backend
identity, cost. A caller-supplied ``user`` is stored as a salted hash. Content fields
exist only when ``log_content`` is enabled and then pass through redaction hooks.
Records are single-line JSON so newlines in prompts can never forge extra log lines.

Sinks are non-blocking: a bounded queue feeds a writer thread; overflow drops the
record and counts it. A failing sink never fails a request.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import queue
import secrets
import threading
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from sieve.observe.redact import Redactor, apply_redactors

log = logging.getLogger("sieve.audit")

SCHEMA = 1


@dataclass
class AuditRecord:
    ts: float
    request_id: str
    mode: str  # enforce | shadow
    outcome: str  # ok | timeout | cancelled | error
    decision: str  # answer | abstain | none
    reason: str
    served_tier: int | None
    escalated: bool
    artifact_id: str
    stream: bool
    latency_s: float
    tiers: list[dict[str, Any]] = field(default_factory=list)
    backend: dict[str, dict[str, str]] = field(default_factory=dict)
    cost: dict[str, float | int] = field(default_factory=dict)
    shadow: dict[str, Any] | None = None
    user_hash: str | None = None
    prompt_chars: int = 0
    config_pin: str = ""
    prompt: str | None = None
    response: str | None = None
    schema: int = SCHEMA

    def to_json(self) -> str:
        d = asdict(self)
        if d["prompt"] is None:
            del d["prompt"]
        if d["response"] is None:
            del d["response"]
        return json.dumps(d, separators=(",", ":"), sort_keys=True, allow_nan=False, default=str)


class AuditSink(Protocol):
    def write(self, rec: AuditRecord) -> None: ...
    def close(self) -> None: ...


class UserHasher:
    """Salted HMAC of client-supplied user ids; the salt is per-process unless set."""

    def __init__(self, salt: bytes | None = None) -> None:
        self._salt = salt or secrets.token_bytes(16)

    def __call__(self, user: str | None) -> str | None:
        if not user:
            return None
        return hmac.new(self._salt, user.encode(), hashlib.sha256).hexdigest()[:16]


def apply_content_policy(
    rec: AuditRecord,
    *,
    log_content: bool,
    prompt: str,
    response: str | None,
    redactors: Iterable[Redactor] = (),
) -> AuditRecord:
    if log_content:
        rs = tuple(redactors)
        rec.prompt = apply_redactors(prompt, rs)
        rec.response = None if response is None else apply_redactors(response, rs)
    return rec


class MemoryAuditSink:
    def __init__(self) -> None:
        self.records: list[AuditRecord] = []
        self.lines: list[str] = []

    def write(self, rec: AuditRecord) -> None:
        self.records.append(rec)
        self.lines.append(rec.to_json())

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None


class NullAuditSink:
    def write(self, rec: AuditRecord) -> None:
        return None

    def close(self) -> None:
        return None


class JsonlAuditSink:
    def __init__(self, path: str, max_queue: int = 10_000) -> None:
        self.path = path
        self.dropped = 0
        self.errors = 0
        self._q: queue.Queue[str | None] = queue.Queue(maxsize=max_queue)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self._fh = os.fdopen(fd, "a", encoding="utf-8")
        self._closed = False
        self._t = threading.Thread(target=self._run, name="sieve-audit", daemon=True)
        self._t.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                self._fh.write(item + "\n")
                self._fh.flush()
            except Exception:
                self.errors += 1
            finally:
                self._q.task_done()

    def write(self, rec: AuditRecord) -> None:
        if self._closed:
            self.dropped += 1
            return
        try:
            line = rec.to_json()
            self._q.put_nowait(line)
        except queue.Full:
            self.dropped += 1
        except Exception:
            self.errors += 1

    def flush(self) -> None:
        self._q.join()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._q.put(None)
        self._t.join(timeout=5)
        try:
            self._fh.close()
        except Exception:
            self.errors += 1


def read_jsonl(path: str) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out

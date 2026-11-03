"""Start, health-wait and stop a local ``llama-server`` process.

One :class:`LlamaServerProcess` owns one OS process: argv built from a :class:`ServerSpec`,
``CUDA_VISIBLE_DEVICES`` pinning (so S and L can sit on different T4s), a free loopback port,
stdout/stderr captured to a log file, a health wait that fails fast (with the log tail) if the
process dies while loading, and a guaranteed stop (SIGTERM, then SIGKILL; also at interpreter
exit). Only the standard library is used; everything talks to loopback and ignores proxies.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

__all__ = [
    "LlamaServerProcess",
    "ServerSpec",
    "ServerStartError",
    "find_free_port",
]


class ServerStartError(RuntimeError):
    """The server did not become healthy (message carries the log tail)."""


def find_free_port(start: int = 0, host: str = "127.0.0.1") -> int:
    """A currently unused TCP port on ``host`` (``start`` > 0: first free port at or above it)."""
    if start <= 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            return int(s.getsockname()[1])
    for port in range(start, min(start + 200, 65535)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise ServerStartError(f"no free port found from {start}")


@dataclass(frozen=True)
class ServerSpec:
    """Everything that defines one server launch."""

    binary: Path
    model: Path
    log_path: Path
    port: int = 0  # 0: pick a free port
    host: str = "127.0.0.1"
    gpu: str | None = None  # CUDA_VISIBLE_DEVICES value; None: inherit
    n_ctx: int = 4096
    n_gpu_layers: int = 99
    flash_attn: str = "auto"  # "on" | "off" | "auto" (auto: llama.cpp default, flag omitted)
    kv_type: str = "f16"  # f16 | q8_0 | q4_0 ... (applied to both K and V)
    extra_args: tuple[str, ...] = ()
    startup_timeout_s: float = 600.0
    env: dict[str, str] = field(default_factory=dict)

    @property
    def effective_flash_attn(self) -> str:
        """Quantised V cache needs flash attention in llama.cpp, so it is forced on."""
        if self.kv_type != "f16":
            if self.flash_attn == "off":
                raise ValueError("quantised KV cache requires flash attention (flash_attn != off)")
            return "on"
        return self.flash_attn

    def argv(self, port: int) -> list[str]:
        a = [
            str(self.binary),
            "-m",
            str(self.model),
            "-ngl",
            str(self.n_gpu_layers),
            "-c",
            str(self.n_ctx),
            "--host",
            self.host,
            "--port",
            str(port),
            "-np",
            "1",
        ]
        fa = self.effective_flash_attn
        if fa in ("on", "off"):
            a += ["-fa", fa]  # current syntax; the pre-2025 bare "-fa" flag is not supported here
        if self.kv_type != "f16":
            a += ["-ctk", self.kv_type, "-ctv", self.kv_type]
        a += list(self.extra_args)
        return a


_LIVE: list[LlamaServerProcess] = []


def _stop_all() -> None:  # pragma: no cover - interpreter shutdown
    for p in list(_LIVE):
        p.stop()


atexit.register(_stop_all)


def _http_json(url: str, timeout: float = 5.0) -> tuple[int, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as r:
            body = r.read()
            code = int(r.status)
    except urllib.error.HTTPError as e:
        return int(e.code), None
    try:
        return code, json.loads(body)
    except ValueError:
        return code, None


class LlamaServerProcess:
    """Context manager around one ``llama-server`` process."""

    def __init__(self, spec: ServerSpec) -> None:
        self.spec = spec
        self.port = 0
        self.proc: subprocess.Popen[bytes] | None = None
        self._log: IO[bytes] | None = None
        self.start_s = 0.0
        self.argv: list[str] = []

    @property
    def base_url(self) -> str:
        return f"http://{self.spec.host}:{self.port}"

    def log_tail(self, n_chars: int = 3000) -> str:
        try:
            return self.spec.log_path.read_text(encoding="utf-8", errors="replace")[-n_chars:]
        except OSError:
            return ""

    def start(self) -> LlamaServerProcess:
        if self.proc is not None:
            raise RuntimeError("server already started")
        spec = self.spec
        if not spec.binary.exists():
            raise ServerStartError(f"llama-server binary not found: {spec.binary}")
        if not spec.model.exists():
            raise ServerStartError(f"model file not found: {spec.model}")
        self.port = find_free_port(spec.port)
        self.argv = spec.argv(self.port)
        env = dict(os.environ)
        env.update(spec.env)
        if spec.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = spec.gpu
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(spec.log_path, "wb")  # noqa: SIM115 - closed in stop()
        self._log.write(
            (
                f"# argv: {' '.join(self.argv)}\n# CUDA_VISIBLE_DEVICES="
                f"{env.get('CUDA_VISIBLE_DEVICES', '<inherited>')}\n"
            ).encode()
        )
        self._log.flush()
        t0 = time.monotonic()
        self.proc = subprocess.Popen(
            self.argv,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        _LIVE.append(self)
        try:
            self.wait_healthy(spec.startup_timeout_s)
        except BaseException:
            self.stop()
            raise
        self.start_s = time.monotonic() - t0
        return self

    def wait_healthy(self, timeout_s: float) -> None:
        assert self.proc is not None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rc = self.proc.poll()
            if rc is not None:
                raise ServerStartError(
                    f"llama-server exited with code {rc} while starting; log tail:\n"
                    f"{self.log_tail()}"
                )
            try:
                code, _ = _http_json(f"{self.base_url}/health", timeout=2.0)
            except (OSError, urllib.error.URLError):
                code = 0
            if code == 200:
                return
            time.sleep(0.5)
        raise ServerStartError(
            f"llama-server not healthy after {timeout_s:.0f}s; log tail:\n{self.log_tail()}"
        )

    def props(self) -> dict[str, Any]:
        """``GET /props`` (build_info, model_path, ...); empty dict if unavailable."""
        try:
            code, body = _http_json(f"{self.base_url}/props")
        except (OSError, urllib.error.URLError):
            return {}
        return body if code == 200 and isinstance(body, dict) else {}

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, grace_s: float = 20.0) -> None:
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        if self._log is not None:
            with contextlib.suppress(OSError):
                self._log.close()
            self._log = None
        if self in _LIVE:
            _LIVE.remove(self)

    def __enter__(self) -> LlamaServerProcess:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

"""Backend provider that runs a local ``llama-server`` per cell (``backend.provider``).

``backend.provider: "sieve.eval.providers.llamacpp:make_provider"``. For each cell the provider
launches a pinned ``llama-server`` on loopback (GPU chosen by tier), waits for health, checks the
served build and model file against what the config pins, runs B1's fail-closed logprob
self-test, and yields a :class:`~sieve.types.Backend`. The server is stopped when the context
exits, including on errors.

Evaluation-specific behaviour layered over B1's :class:`LlamaServerBackend`:

* prompts go to ``/completion`` raw (the runner renders the fixed few-shot template, D-016);
* greedy decoding is the caller's ``SamplingParams``; ``n_probs`` is passed through (clamped to 64);
* generation stops at the first newline (``stop: ["\\n"]``): the answer span is the first line.
  Stop-trimmed trailing tokens are removed from ``tokens`` so text and tokens reassemble
  (:func:`trim_to_text`); anything that does not reassemble is left untouched for
  ``validate_completion`` to flag (never repaired);
* identity (``BackendInfo``) records model file hash, quant, KV type, and in ``build`` the
  llama.cpp commit, the served ``build_info``, the flash-attention flag, ``-ngl`` and the stop
  rule; ``template_hash`` is ``sieve.eval.prompts.template_hash()``;
* per-call server timings (``prompt_ms``, ``predicted_ms`` ...) and wall time go to a per-cell
  JSONL sidecar next to the server log.

Configuration (``backend:`` section of the experiment YAML; environment overrides in brackets):
``llama_cpp_commit`` (required), ``binary`` [SIEVE_LLAMA_SERVER], ``models_dir`` [SIEVE_MODELS_DIR],
``log_dir`` [SIEVE_SERVER_LOG_DIR], ``gpu: {S: "0", L: "1"}`` [SIEVE_GPU_S / SIEVE_GPU_L],
``n_ctx`` (4096), ``n_gpu_layers`` (99), ``flash_attn`` (auto|on|off), ``stop`` (["\\n"]),
``startup_timeout_s``, ``request_timeout_s``, ``base_port``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Coroutine, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

from sieve.backends.base import BackendError, SelfTestResult
from sieve.backends.llamaserver import LlamaServerBackend
from sieve.backends.selftest import run_logprob_self_test
from sieve.eval import prompts
from sieve.eval.config import CellSpec, ExperimentConfig
from sieve.eval.providers.server_manager import (
    LlamaServerProcess,
    ServerSpec,
    ServerStartError,
)
from sieve.types import Backend, BackendInfo, Completion, SamplingParams

__all__ = [
    "EvalLlamaBackend",
    "LlamaCppProvider",
    "ProviderError",
    "file_sha256",
    "make_provider",
    "trim_to_text",
]

T = TypeVar("T")

UNPINNED = {"", "PIN-ME"}


class ProviderError(RuntimeError):
    """The cell cannot be served with the pinned identity (fail closed)."""


# --- pure helpers ------------------------------------------------------------------


def trim_to_text(c: Completion, *, stopped_by_word: bool = False) -> Completion:
    """Make ``text``, ``tokens`` and ``n_gen`` consistent after a server-side stop string.

    Measured on llama.cpp @ fac3a89 (T4 diagnostic kernel ``sieve-diag-completion``): when a stop
    string matches, llama-server (a) omits the matching token's entry from
    ``completion_probabilities`` while ``tokens_predicted`` still counts it, and (b) keeps in
    ``content`` the part of that token before the stop string (token ``".\n"`` with stop ``"\n"``
    leaves ``"."`` in ``content`` but no entry). With ``stopped_by_word`` the text is cut back to
    the concatenation of the reported tokens and ``n_gen`` loses the unreported stop token, so the
    answer span reassembles exactly. Any other mismatch (a genuinely dropped logprob, mangled
    UTF-8) is left untouched for ``validate_completion`` to flag, never repaired. Independently,
    trailing tokens that lie beyond ``text`` are dropped (other servers keep the stop token).
    """
    toks = c.tokens
    if toks is None or not toks:
        return c
    joined = "".join(t.token for t in toks)
    if stopped_by_word and c.finish == "stop" and c.n_gen == len(toks) + 1:
        rest = c.text[len(joined) :] if c.text.startswith(joined) else None
        if rest is not None and "\n" not in rest and len(rest) <= 16:
            return replace(c, text=joined, n_gen=len(toks))
    if joined == c.text:
        return c
    acc = ""
    for k, t in enumerate(toks, start=1):
        acc += t.token
        if acc == c.text:
            if k == len(toks):
                return c
            n_gen = c.n_gen - (len(toks) - k) if c.n_gen == len(toks) else c.n_gen
            return replace(c, tokens=toks[:k], n_gen=n_gen)
        if len(acc) > len(c.text) or not c.text.startswith(acc):
            break
    return c


def file_sha256(path: Path, *, chunk: int = 8 * 1024 * 1024) -> str:
    """SHA-256 of a file, cached in ``<file>.sha256.json`` keyed by size and mtime."""
    st = path.stat()
    side = path.with_name(path.name + ".sha256.json")
    if side.exists():
        try:
            d = json.loads(side.read_text(encoding="utf-8"))
            if d.get("size") == st.st_size and d.get("mtime_ns") == st.st_mtime_ns:
                return str(d["sha256"])
        except (OSError, ValueError, KeyError):
            pass
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    digest = h.hexdigest()
    with contextlib.suppress(OSError):  # read-only models dir: hash again next time
        side.write_text(
            json.dumps({"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}),
            encoding="utf-8",
        )
    return digest


_SHARD = re.compile(r"^(?P<stem>.+)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$")


def shard_files(first: Path) -> list[Path]:
    """All shard paths of a split GGUF (``-00001-of-0000N``), or ``[first]`` if not split."""
    m = _SHARD.match(first.name)
    if not m:
        return [first]
    n = int(m.group("n"))
    return [first.with_name(f"{m.group('stem')}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def _run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion whether or not an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: list[Any] = []

    def target() -> None:
        try:
            box.append((True, asyncio.run(coro)))
        except BaseException as e:  # re-raised in the caller
            box.append((False, e))

    th = threading.Thread(target=target)
    th.start()
    th.join()
    ok, val = box[0]
    if not ok:
        raise val
    return val  # type: ignore[no-any-return]


# --- backend -----------------------------------------------------------------------


class _StopBackend(LlamaServerBackend):
    """B1's backend plus a ``stop`` list and capture of the server's ``timings`` block."""

    def __init__(self, *args: Any, stop: Sequence[str] = (), **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stop = tuple(stop)
        self.last_timings: dict[str, Any] | None = None
        self.last_stop_type: str = ""

    def _payload(self, prompt: str, p: SamplingParams, n_probs: int) -> dict[str, Any]:
        body = super()._payload(prompt, p, n_probs)
        if self._stop:
            body["stop"] = list(self._stop)
        return body

    async def generate(self, prompt: str, params: SamplingParams, *, n_probs: int) -> Completion:
        import time as _time

        import httpx

        t0 = _time.monotonic()
        try:
            r = await self._client.post("/completion", json=self._payload(prompt, params, n_probs))
        except httpx.HTTPError as e:
            raise BackendError(f"POST /completion: {type(e).__name__}") from e
        data = self._decode(r, "POST /completion")
        tm = data.get("timings")
        self.last_timings = dict(tm) if isinstance(tm, dict) else None
        self.last_stop_type = str(data.get("stop_type", ""))
        return self.to_completion(data, params, _time.monotonic() - t0)


class EvalLlamaBackend:
    """``Backend`` for the eval runner: one server, loop-safe client reuse, answer-span trim."""

    def __init__(
        self,
        base_url: str,
        *,
        info: BackendInfo,
        stop: Sequence[str],
        request_timeout_s: float,
        timing_path: Path | None = None,
    ) -> None:
        self._base_url = base_url
        self._info = info
        self._stop = tuple(stop)
        self._timeout = request_timeout_s
        self._timing_path = timing_path
        self._inner: _StopBackend | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._seq = 0
        self._sink = None if timing_path is None else timing_path.open("a", encoding="utf-8")

    @property
    def info(self) -> BackendInfo:
        return self._info

    def _client(self) -> _StopBackend:
        loop = asyncio.get_running_loop()
        if self._inner is None or self._loop is not loop:
            # httpx pools are bound to the loop that created them; the runner calls
            # asyncio.run() once per block, so build a new client when the loop changes.
            self._inner = _StopBackend(
                self._base_url,
                model_id=self._info.model_id,
                model_hash=self._info.model_hash,
                quant=self._info.quant,
                kv_quant=self._info.kv_quant,
                template_hash=self._info.template_hash,
                build=self._info.build,
                timeout_s=self._timeout,
                cache_prompt=False,
                stop=self._stop,
            )
            self._loop = loop
        return self._inner

    async def generate(self, prompt: str, params: SamplingParams, *, n_probs: int) -> Completion:
        inner = self._client()
        t0 = time.monotonic()
        comp = await inner.generate(prompt, params, n_probs=n_probs)
        wall = time.monotonic() - t0
        self._log_timing(inner.last_timings, wall, comp)
        return trim_to_text(comp, stopped_by_word=inner.last_stop_type == "word")

    def _log_timing(self, tm: Mapping[str, Any] | None, wall: float, comp: Completion) -> None:
        if self._sink is None:
            return
        self._seq += 1
        row: dict[str, Any] = {"seq": self._seq, "wall_s": round(wall, 6), "n_gen": comp.n_gen}
        for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "cache_n"):
            if tm is not None and k in tm:
                row[k] = tm[k]
        self._sink.write(json.dumps(row, sort_keys=True) + "\n")
        self._sink.flush()

    def close(self) -> None:
        if self._sink is not None:
            self._sink.close()
            self._sink = None
        self._inner = None


# --- provider ----------------------------------------------------------------------


def _gpu_count() -> int:
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for ln in out.splitlines() if ln.startswith("GPU "))


class LlamaCppProvider:
    """``BackendProvider`` (see ``sieve.eval.runner``) over locally started llama-servers."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        binary: Path | None = None,
        models_dir: Path | None = None,
        log_dir: Path | None = None,
        gpus: Mapping[str, str | None] | None = None,
    ) -> None:
        b: dict[str, Any] = dict(cfg.raw.get("backend", {}))
        self.cfg = cfg
        self.commit = str(os.environ.get("SIEVE_LLAMA_CPP_COMMIT") or b.get("llama_cpp_commit", ""))
        self.binary = binary or Path(
            os.environ.get("SIEVE_LLAMA_SERVER") or b.get("binary") or "llama-server"
        )
        self.models_dir = models_dir or Path(
            os.environ.get("SIEVE_MODELS_DIR") or b.get("models_dir") or "models"
        )
        self.log_dir = log_dir or Path(
            os.environ.get("SIEVE_SERVER_LOG_DIR") or b.get("log_dir") or "server_logs"
        )
        gpu_cfg: dict[str, Any] = dict(b.get("gpu", {}))
        n_gpu = _gpu_count()
        default = {"S": "0", "L": "1" if n_gpu >= 2 else "0"} if n_gpu else {}
        self.gpus: dict[str, str | None] = {**default, **gpu_cfg}
        for tier in ("S", "L"):
            env_v = os.environ.get(f"SIEVE_GPU_{tier}")
            if env_v:
                self.gpus[tier] = env_v
        if gpus:
            self.gpus.update(gpus)
        self.n_ctx = int(b.get("n_ctx", 4096))
        self.n_gpu_layers = int(b.get("n_gpu_layers", 99))
        self.flash_attn = str(b.get("flash_attn", "auto"))
        self.stop: tuple[str, ...] = tuple(str(s) for s in b.get("stop", ["\n"]))
        self.startup_timeout_s = float(b.get("startup_timeout_s", 600))
        self.request_timeout_s = float(b.get("request_timeout_s", 120))
        self.base_port = int(b.get("base_port", 8100))
        self.opened: list[str] = []

    # -- resolution --------------------------------------------------------------

    def resolve_model(self, cell: CellSpec) -> Path:
        if "gguf_path" in cell.extra:
            return Path(str(cell.extra["gguf_path"]))
        spec = str(cell.extra.get("gguf", ""))
        if ":" not in spec:
            raise ProviderError(
                f"cell {cell.name}: extra.gguf must be 'hf_repo:file.gguf' (or set extra.gguf_path)"
            )
        return self.models_dir / spec.split(":", 1)[1]

    def _model_hash(self, first: Path) -> str:
        shards = shard_files(first)
        missing = [p.name for p in shards if not p.exists()]
        if missing:
            raise ProviderError(f"model file(s) missing in {first.parent}: {missing}")
        hashes = [file_sha256(p) for p in shards]
        return (
            hashes[0]
            if len(hashes) == 1
            else hashlib.sha256("\n".join(hashes).encode()).hexdigest()
        )

    def _check_build(self, props: Mapping[str, Any]) -> str:
        """Return the served ``build_info``; refuse a build that contradicts the pinned commit."""
        if self.commit in UNPINNED and not os.environ.get("SIEVE_ALLOW_UNPINNED"):
            raise ProviderError(
                "backend.llama_cpp_commit is not pinned (set it, or SIEVE_LLAMA_CPP_COMMIT)"
            )
        build_info = str(props.get("build_info", "") or "")
        hexes = re.findall(r"(?<![0-9a-f])[0-9a-f]{7,40}(?![0-9a-f])", build_info)
        if (
            hexes
            and self.commit not in UNPINNED
            and not any(self.commit.startswith(h) or h.startswith(self.commit) for h in hexes)
        ):
            raise ProviderError(
                f"served llama.cpp build {build_info!r} does not match pinned commit {self.commit}"
            )
        return build_info

    # -- open --------------------------------------------------------------------

    @contextmanager
    def open(self, cell: CellSpec) -> Iterator[Backend]:
        model = self.resolve_model(cell)
        model_hash = self._model_hash(model)
        spec = ServerSpec(
            binary=self.binary,
            model=model,
            log_path=self.log_dir / f"{cell.name}.server.log",
            port=self.base_port + len(self.opened),
            gpu=self.gpus.get(cell.tier),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            flash_attn=self.flash_attn,
            kv_type=cell.kv_quant,
            startup_timeout_s=self.startup_timeout_s,
        )
        self.opened.append(cell.name)
        try:
            proc = LlamaServerProcess(spec).start()
        except ServerStartError as e:
            raise ProviderError(f"cell {cell.name}: {e}") from e
        backend: EvalLlamaBackend | None = None
        try:
            props = proc.props()
            served = os.path.basename(str(props.get("model_path", "")))
            if served and served != model.name:
                raise ProviderError(
                    f"cell {cell.name}: server loaded {served!r}, not {model.name!r}"
                )
            build_info = self._check_build(props)
            stop_id = "+".join(json.dumps(s) for s in self.stop) or "none"
            build = (
                f"llama.cpp@{self.commit}|{build_info or 'build_info-unavailable'}|"
                f"fa={spec.effective_flash_attn}|ngl={spec.n_gpu_layers}|stop={stop_id}"
            )
            info = BackendInfo(
                model_id=cell.model_id,
                model_hash=model_hash,
                quant=cell.quant,
                kv_quant=cell.kv_quant,
                build=build,
                template_hash=prompts.template_hash_for(cell.datasets),
            )
            self.log_dir.mkdir(parents=True, exist_ok=True)
            backend = EvalLlamaBackend(
                proc.base_url,
                info=info,
                stop=self.stop,
                request_timeout_s=self.request_timeout_s,
                timing_path=self.log_dir / f"{cell.name}.timings.jsonl",
            )
            # The self-test checks logprob plumbing, so it runs WITHOUT the production stop rule:
            # its probe prompts may legitimately continue with a newline (the 7B did on the T4
            # canary: zero tokens, "missing_logprobs"). The stop path is covered by the sanity gate
            # on real generations (finish==stop rate, token/text consistency).
            probe = EvalLlamaBackend(
                proc.base_url,
                info=info,
                stop=(),
                request_timeout_s=self.request_timeout_s,
                timing_path=None,
            )
            try:
                st = _run_sync(
                    run_logprob_self_test(
                        probe, n_probs=min(self.cfg.n_probs, 64), repeats=3, strict_utf8=False
                    )
                )
            finally:
                probe.close()
            self._write_sidecar(cell, spec, proc, props, info, st)
            if not st.ok:
                raise ProviderError(
                    f"cell {cell.name}: logprob self-test failed: {list(st.failed)} "
                    f"({[c.detail for c in st.checks if not c.ok]})"
                )
            yield backend
        finally:
            if backend is not None:
                backend.close()
            proc.stop()

    def _write_sidecar(
        self,
        cell: CellSpec,
        spec: ServerSpec,
        proc: LlamaServerProcess,
        props: Mapping[str, Any],
        info: BackendInfo,
        st: SelfTestResult,
    ) -> None:
        log = proc.log_tail(200_000)
        m = re.search(r"flash_attn\s*=\s*(\S+)", log)
        keep = {
            k: props[k] for k in ("build_info", "model_path", "total_slots", "n_ctx") if k in props
        }
        body = {
            "cell": cell.name,
            "argv": proc.argv,
            "gpu": spec.gpu,
            "port": proc.port,
            "server_start_s": round(proc.start_s, 3),
            "props": keep,
            "log_flash_attn": m.group(1) if m else None,
            "backend_info": info.__dict__,
            "self_test": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in st.checks],
        }
        (self.log_dir / f"{cell.name}.server.json").write_text(
            json.dumps(
                body,
                indent=1,
                sort_keys=True,
                default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o),
            ),
            encoding="utf-8",
        )


def make_provider(cfg: ExperimentConfig) -> LlamaCppProvider:
    """Factory named by ``backend.provider``."""
    return LlamaCppProvider(cfg)

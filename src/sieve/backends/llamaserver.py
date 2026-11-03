"""Async client for ``llama-server`` (llama.cpp).

Uses ``/completion`` (not the chat route) with ``n_probs`` so the router controls the
exact rendered prompt. Known upstream logprob bugs (#31451 multi-token UTF-8 logprobs
dropped, #29456 uncapped ``n_probs``, #31206 non-determinism) are handled by
(a) clamping ``n_probs`` client side, (b) mapping the response faithfully so malformed
logprobs stay visible to ``validate_completion``, and (c) the startup self-test.

Cancellation: ``generate`` is a plain awaited ``httpx`` request; cancelling the task
closes the socket, which makes llama-server release the slot.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Collection, Mapping
from typing import Any

import httpx

from sieve.backends.base import (
    DEFAULT_ALLOWED_HOSTS,
    BackendError,
    validate_backend_url,
)
from sieve.types import BackendInfo, Completion, FinishReason, SamplingParams, TokenLP

#: Server-side clamp for n_probs (bug #29456: unbounded n_probs can exhaust memory).
MAX_N_PROBS = 64
#: Cap on the upstream JSON body we are willing to parse.
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def _f(x: Any) -> float:
    """Float from JSON; anything non-numeric becomes NaN so the defect is flagged."""
    if isinstance(x, bool) or not isinstance(x, int | float):
        return math.nan
    return float(x)


def _log(p: Any) -> float:
    v = _f(p)
    if math.isnan(v):
        return v
    return math.log(v) if v > 0.0 else -math.inf


def _entry_lp(e: Mapping[str, Any], post_sampling: bool) -> float:
    if post_sampling and "prob" in e:
        return _log(e["prob"])
    if "logprob" in e:
        return _f(e["logprob"])
    if "prob" in e:
        return _log(e["prob"])
    return math.nan


def parse_completion_probabilities(
    raw: Any, post_sampling: bool = False
) -> tuple[TokenLP, ...] | None:
    """Map llama-server ``completion_probabilities`` to ``TokenLP`` without repairing it.

    Supports the current schema (``token``/``logprob``/``top_logprobs``, or ``prob``/
    ``top_probs`` when ``post_sampling_probs``) and the legacy one (``content``/``probs``
    with ``tok_str``/``prob``). Returns ``None`` if the field is absent or not a list.
    """
    if not isinstance(raw, list):
        return None
    out: list[TokenLP] = []
    for item in raw:
        if not isinstance(item, dict):
            out.append(TokenLP("", math.nan, None))
            continue
        top_raw = item.get("top_logprobs", item.get("top_probs", item.get("probs")))
        top: tuple[tuple[str, float], ...] | None = None
        if isinstance(top_raw, list):
            pairs: list[tuple[str, float]] = []
            for e in top_raw:
                if isinstance(e, dict):
                    name = e.get("token", e.get("tok_str", ""))
                    pairs.append((str(name), _entry_lp(e, post_sampling)))
            top = tuple(pairs)
        if "token" in item:
            token = str(item["token"])
            lp = _entry_lp(item, post_sampling)
        else:  # legacy: chosen token's prob is looked up in the probs list
            token = str(item.get("content", ""))
            lp = next((p for t, p in (top or ()) if t == token), math.nan)
        out.append(TokenLP(token, lp, top))
    return tuple(out)


class LlamaServerBackend:
    """``Backend`` for one ``llama-server`` instance."""

    def __init__(
        self,
        base_url: str,
        *,
        model_id: str,
        model_hash: str,
        quant: str,
        kv_quant: str,
        template_hash: str,
        build: str = "",
        expected_model_file: str | None = None,
        allowed_hosts: Collection[str] = DEFAULT_ALLOWED_HOSTS,
        timeout_s: float = 60.0,
        connect_timeout_s: float = 3.0,
        cache_prompt: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base = validate_backend_url(base_url, allowed_hosts)
        self._model_id = model_id
        self._model_hash = model_hash
        self._quant = quant
        self._kv_quant = kv_quant
        self._template_hash = template_hash
        self._pinned_build = build
        self._expected_model_file = expected_model_file
        self._cache_prompt = cache_prompt
        self._build = build
        self._props_model_file = ""
        # trust_env=False: never route backend traffic through HTTP(S)_PROXY from the env;
        # follow_redirects=False: a redirect could point at a non-allow-listed host.
        self._client = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
            headers={"user-agent": "sieve-llm"},
        )

    # -- identity -----------------------------------------------------------------

    @property
    def info(self) -> BackendInfo:
        model_hash = self._model_hash
        if (
            self._expected_model_file is not None
            and self._props_model_file != self._expected_model_file
        ):
            # file served is not the file the operator pinned: identity can never match
            model_hash = f"unverified:{self._props_model_file or 'unknown'}"
        return BackendInfo(
            model_id=self._model_id,
            model_hash=model_hash,
            quant=self._quant,
            kv_quant=self._kv_quant,
            build=self._build,
            template_hash=self._template_hash,
        )

    async def _get_json(self, path: str) -> dict[str, Any]:
        try:
            r = await self._client.get(path)
        except httpx.HTTPError as e:
            raise BackendError(f"GET {path}: {type(e).__name__}") from e
        return self._decode(r, path)

    @staticmethod
    def _decode(r: httpx.Response, what: str) -> dict[str, Any]:
        if 300 <= r.status_code < 400:
            raise BackendError(f"{what}: redirect refused")
        if r.status_code != 200:
            raise BackendError(f"{what}: HTTP {r.status_code}")
        if len(r.content) > MAX_RESPONSE_BYTES:
            raise BackendError(f"{what}: response too large")
        try:
            data = r.json()
        except ValueError as e:
            raise BackendError(f"{what}: invalid JSON") from e
        if not isinstance(data, dict):
            raise BackendError(f"{what}: unexpected JSON type")
        return data

    async def health(self) -> bool:
        try:
            data = await self._get_json("/health")
        except BackendError:
            return False
        return bool(data.get("status", "ok") == "ok")

    async def refresh(self) -> bool:
        """Read ``/props``; returns True if the identity (build/model file) changed."""
        before = self.info.key()
        props = await self._get_json("/props")
        build = props.get("build_info")
        if isinstance(build, str) and build:
            self._build = build
        elif self._pinned_build:
            self._build = self._pinned_build
        path = props.get("model_path")
        self._props_model_file = os.path.basename(path) if isinstance(path, str) else ""
        return self.info.key() != before

    @staticmethod
    def chat_template_fingerprint(props: Mapping[str, Any]) -> str:
        """Hash of the server-side chat template (informational; we render prompts ourselves)."""
        t = props.get("chat_template")
        return hashlib.sha256(str(t).encode()).hexdigest()[:16] if t else ""

    # -- generation ---------------------------------------------------------------

    def _payload(self, prompt: str, p: SamplingParams, n_probs: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": p.max_tokens,
            "temperature": p.temperature,
            "top_p": p.top_p,
            "top_k": p.top_k,
            "min_p": p.min_p,
            "n_probs": max(0, min(n_probs, MAX_N_PROBS)),
            "post_sampling_probs": p.post_sampling_probs,
            "cache_prompt": self._cache_prompt,
            "stream": False,
        }
        if p.seed is not None:
            body["seed"] = p.seed
        if p.logit_bias:
            body["logit_bias"] = [[i, b] for i, b in p.logit_bias]
        if p.grammar is not None:
            body["grammar"] = p.grammar
        return body

    async def generate(self, prompt: str, params: SamplingParams, *, n_probs: int) -> Completion:
        t0 = time.monotonic()
        try:
            r = await self._client.post("/completion", json=self._payload(prompt, params, n_probs))
        except httpx.HTTPError as e:
            raise BackendError(f"POST /completion: {type(e).__name__}") from e
        data = self._decode(r, "POST /completion")
        return self.to_completion(data, params, time.monotonic() - t0)

    @staticmethod
    def to_completion(data: Mapping[str, Any], params: SamplingParams, wall_s: float) -> Completion:
        text = data.get("content")
        if not isinstance(text, str):
            raise BackendError("POST /completion: missing content")
        stop_type = data.get("stop_type")
        finish: FinishReason = "stop"
        if stop_type == "limit" or data.get("truncated") is True:
            finish = "length"
        elif stop_type not in (None, "eos", "word", "none") and stop_type != "":
            finish = "error"
        n_gen_raw = data.get("tokens_predicted")
        n_gen = n_gen_raw if isinstance(n_gen_raw, int) and not isinstance(n_gen_raw, bool) else -1
        n_prompt_raw = data.get("tokens_evaluated")
        n_prompt = n_prompt_raw if isinstance(n_prompt_raw, int) else 0
        timings = data.get("timings")
        ttft = 0.0
        if isinstance(timings, dict) and isinstance(timings.get("prompt_ms"), int | float):
            ttft = float(timings["prompt_ms"]) / 1000.0
        tokens = parse_completion_probabilities(
            data.get("completion_probabilities"), params.post_sampling_probs
        )
        return Completion(
            text=text,
            tokens=tokens,
            finish=finish,
            n_gen=n_gen,
            sampling=params,
            answer_start=0,
            n_prompt=n_prompt,
            t_first_token_s=ttft,
            t_total_s=wall_s,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

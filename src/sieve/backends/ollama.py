"""Best-effort Ollama adapter (``/api/generate``, ``raw=true``).

LIMITS (read before relying on this for a certificate):

* Token logprobs exist only in recent Ollama versions and are not covered by the
  llama-server bug set that Sieve's self-test was designed around. They are mapped
  faithfully and validated; when absent the completion has ``tokens=None`` and the
  gate fails closed.
* Ollama does not report a build hash that ties to a llama.cpp commit, nor a KV-cache
  type, nor whether logprobs are pre- or post-sampling: ``build`` is the Ollama
  version string, ``kv_quant`` is operator-declared and unverifiable here.
* Model eviction / keep-alive can change load behaviour between requests; the
  adapter cannot detect a swapped model other than via ``/api/tags`` digest, which
  is pinned into ``model_hash`` when the operator does not supply one.
* The frozen v1.0 experiments use llama-server only; this adapter is not part of the
  pre-registered configuration.
"""

from __future__ import annotations

import math
import time
from collections.abc import Collection
from typing import Any

import httpx

from sieve.backends.base import DEFAULT_ALLOWED_HOSTS, BackendError, validate_backend_url
from sieve.types import BackendInfo, Completion, FinishReason, SamplingParams, TokenLP

MAX_TOP = 20


def _f(x: Any) -> float:
    if isinstance(x, bool) or not isinstance(x, int | float):
        return math.nan
    return float(x)


class OllamaBackend:
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
        allowed_hosts: Collection[str] = DEFAULT_ALLOWED_HOSTS,
        timeout_s: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base = validate_backend_url(base_url, allowed_hosts)
        self._model_id = model_id
        self._model_hash = model_hash
        self._quant = quant
        self._kv_quant = kv_quant
        self._template_hash = template_hash
        self._build = build
        self._client = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(timeout_s, connect=3.0),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            self._model_id,
            self._model_hash,
            self._quant,
            self._kv_quant,
            self._build,
            self._template_hash,
        )

    async def refresh(self) -> bool:
        before = self.info.key()
        try:
            r = await self._client.get("/api/version")
        except httpx.HTTPError as e:
            raise BackendError(f"GET /api/version: {type(e).__name__}") from e
        if r.status_code != 200:
            raise BackendError(f"GET /api/version: HTTP {r.status_code}")
        try:
            v = r.json().get("version")
        except (ValueError, AttributeError) as e:
            raise BackendError("GET /api/version: invalid JSON") from e
        if isinstance(v, str):
            self._build = f"ollama-{v}"
        return self.info.key() != before

    async def generate(self, prompt: str, params: SamplingParams, *, n_probs: int) -> Completion:
        options: dict[str, Any] = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "min_p": params.min_p,
            "num_predict": params.max_tokens,
        }
        if params.seed is not None:
            options["seed"] = params.seed
        body: dict[str, Any] = {
            "model": self._model_id,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": options,
        }
        if n_probs > 0:
            body["logprobs"] = True
            body["top_logprobs"] = min(n_probs, MAX_TOP)
        t0 = time.monotonic()
        try:
            r = await self._client.post("/api/generate", json=body)
        except httpx.HTTPError as e:
            raise BackendError(f"POST /api/generate: {type(e).__name__}") from e
        if r.status_code != 200:
            raise BackendError(f"POST /api/generate: HTTP {r.status_code}")
        try:
            data = r.json()
        except ValueError as e:
            raise BackendError("POST /api/generate: invalid JSON") from e
        if not isinstance(data, dict) or not isinstance(data.get("response"), str):
            raise BackendError("POST /api/generate: malformed response")
        finish: FinishReason = "length" if data.get("done_reason") == "length" else "stop"
        if data.get("done") is not True:
            finish = "error"
        tokens: tuple[TokenLP, ...] | None = None
        raw = data.get("logprobs")
        if isinstance(raw, list):
            toks: list[TokenLP] = []
            for ent in raw:
                if not isinstance(ent, dict):
                    toks.append(TokenLP("", math.nan, None))
                    continue
                tl = ent.get("top_logprobs")
                top = (
                    tuple(
                        (str(x.get("token", "")), _f(x.get("logprob")))
                        for x in tl
                        if isinstance(x, dict)
                    )
                    if isinstance(tl, list)
                    else None
                )
                toks.append(TokenLP(str(ent.get("token", "")), _f(ent.get("logprob")), top))
            tokens = tuple(toks)
        n_gen = data.get("eval_count")
        n_prompt = data.get("prompt_eval_count")
        return Completion(
            text=data["response"],
            tokens=tokens,
            finish=finish,
            n_gen=n_gen if isinstance(n_gen, int) else -1,
            sampling=params,
            n_prompt=n_prompt if isinstance(n_prompt, int) else 0,
            t_total_s=time.monotonic() - t0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

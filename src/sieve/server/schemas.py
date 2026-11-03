"""Request validation and OpenAI-compatible response construction.

Validation is hand-written (not pydantic body binding) for three reasons: error messages
must never echo request content, unknown/unsupported fields need distinct error codes,
and duplicate keys / NaN / depth must be rejected before any model sees the data.
"""

from __future__ import annotations

import json
import math
import re
import secrets
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from sieve.router.core import RouterResult
from sieve.router.prompt import ChatMessage
from sieve.server.config import ServerSettings
from sieve.server.errors import OpenAIError
from sieve.types import Action

#: Fields we refuse because supporting them would change what the gate certifies.
UNSUPPORTED_FIELDS = frozenset(
    {
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "logit_bias",
        "response_format",
        "logprobs",
        "top_logprobs",
        "parallel_tool_calls",
        "modalities",
        "audio",
        "prediction",
    }
)
#: Decoding knobs. The gate is calibrated for the artifact's decoding, so these are
#: ignored (default) or rejected; never forwarded to a backend.
SAMPLING_FIELDS = frozenset(
    {"temperature", "top_p", "presence_penalty", "frequency_penalty", "seed", "stop"}
)
KNOWN_FIELDS = (
    frozenset(
        {
            "model",
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "stream_options",
            "n",
            "user",
        }
    )
    | SAMPLING_FIELDS
)

#: Headers that look like attempts to steer the upstream; rejected outright.
_OVERRIDE_HEADER = re.compile(
    r"^x-(sieve-)?(backend|base-?url|api-?base|upstream|target|proxy|model-?url|host-override)",
    re.I,
)
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def reject_override_headers(names: list[str]) -> None:
    for n in names:
        if _OVERRIDE_HEADER.match(n):
            raise OpenAIError(
                400, "Backend selection by header is not supported", code="backend_override"
            )


def safe_request_id(raw: str | None) -> str:
    if raw is not None and _REQUEST_ID.fullmatch(raw):
        return raw
    return "req-" + secrets.token_hex(8)


def _no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in pairs:
        if k in d:
            raise ValueError("duplicate key")
        d[k] = v
    return d


def _bad_const(_: str) -> Any:
    raise ValueError("non-finite number")


def parse_json(body: bytes) -> dict[str, Any]:
    try:
        data = json.loads(body, object_pairs_hook=_no_dupes, parse_constant=_bad_const)
    except (ValueError, RecursionError):  # UnicodeDecodeError and JSONDecodeError are ValueErrors
        raise OpenAIError(400, "Request body is not valid JSON", code="invalid_json") from None
    if not isinstance(data, dict):
        raise OpenAIError(400, "Request body must be a JSON object", code="invalid_json")
    return data


@dataclass(frozen=True)
class ChatRequest:
    messages: tuple[ChatMessage, ...]
    stream: bool
    include_usage: bool
    max_tokens: int | None
    user: str | None
    ignored: tuple[str, ...]


def _err(msg: str, code: str, param: str | None = None, status: int = 400) -> OpenAIError:
    return OpenAIError(status, msg, code=code, param=param)


def _content(c: Any, idx: int, limit: int) -> str:
    param = f"messages[{idx}].content"
    if isinstance(c, str):
        text = c
    elif isinstance(c, list):
        parts: list[str] = []
        for p in c:
            if (
                not isinstance(p, dict)
                or p.get("type") != "text"
                or set(p) - {"type", "text"}
                or not isinstance(p.get("text"), str)
            ):
                raise _err("Only text content parts are supported", "unsupported_content", param)
            parts.append(p["text"])
        text = "\n".join(parts)
    else:
        raise _err("content must be a string or a list of text parts", "invalid_type", param)
    if len(text) > limit:
        raise _err("Message too long", "context_length_exceeded", param, 413)
    return text


def validate_chat(data: dict[str, Any], cfg: ServerSettings) -> ChatRequest:
    lim = cfg.limits
    for key in data:
        if key in UNSUPPORTED_FIELDS:
            raise _err(f"Parameter '{key}' is not supported", "unsupported_parameter", key)
        if key not in KNOWN_FIELDS:
            # Includes base_url / api_base / backend / grammar ...: never honoured.
            raise _err("Unknown request parameter", "unknown_parameter", None)
    model = data.get("model")
    if not isinstance(model, str) or model not in cfg.accepted_models:
        raise OpenAIError(
            404,
            "The requested model does not exist",
            code="model_not_found",
            reject="unknown_model",
        )
    msgs_raw = data.get("messages")
    if not isinstance(msgs_raw, list) or not msgs_raw:
        raise _err("messages must be a non-empty list", "invalid_messages", "messages")
    if len(msgs_raw) > lim.max_messages:
        raise _err("Too many messages", "context_length_exceeded", "messages", 413)
    msgs: list[ChatMessage] = []
    total = 0
    for i, m in enumerate(msgs_raw):
        if not isinstance(m, dict) or set(m) - {"role", "content", "name"}:
            raise _err("Invalid message object", "invalid_messages", f"messages[{i}]")
        role = m.get("role")
        if role not in ("system", "user", "assistant"):
            raise _err("Unsupported role", "invalid_role", f"messages[{i}].role")
        if "name" in m and (not isinstance(m["name"], str) or len(m["name"]) > 64):
            raise _err("Invalid name", "invalid_messages", f"messages[{i}].name")
        text = _content(m.get("content"), i, lim.max_message_chars)
        total += len(text)
        msgs.append(ChatMessage(role, text))
    if total > lim.max_prompt_chars:
        raise _err("Prompt too long", "context_length_exceeded", "messages", 413)
    if msgs[-1].role != "user":
        raise _err("The last message must have role 'user'", "invalid_messages", "messages")

    stream = data.get("stream", False)
    if not isinstance(stream, bool):
        raise _err("stream must be a boolean", "invalid_type", "stream")
    include_usage = False
    if "stream_options" in data:
        so = data["stream_options"]
        if (
            not isinstance(so, dict)
            or set(so) - {"include_usage"}
            or not isinstance(so.get("include_usage", False), bool)
        ):
            raise _err("Invalid stream_options", "invalid_type", "stream_options")
        include_usage = bool(so.get("include_usage", False))
    n = data.get("n", 1)
    if isinstance(n, bool) or n != 1:
        raise _err("Only n=1 is supported", "unsupported_parameter", "n")

    mt: int | None = None
    for k in ("max_completion_tokens", "max_tokens"):
        if k in data and data[k] is not None:
            v = data[k]
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise _err("max_tokens must be a positive integer", "invalid_type", k)
            if v > lim.max_tokens_cap:
                raise _err(
                    f"max_tokens exceeds the server cap of {lim.max_tokens_cap}", "invalid_value", k
                )
            mt = v
            break
    user = data.get("user")
    if user is not None and (not isinstance(user, str) or len(user) > 256):
        raise _err("Invalid user", "invalid_type", "user")

    ignored: list[str] = []
    for k in sorted(SAMPLING_FIELDS & data.keys()):
        v = data[k]
        if isinstance(v, float) and not math.isfinite(v):
            raise _err("Invalid number", "invalid_type", k)
        if v is None:
            continue
        if cfg.client_sampling == "reject":
            raise _err(
                f"'{k}' cannot be set: decoding is fixed by the calibration artifact",
                "unsupported_parameter",
                k,
            )
        ignored.append(k)
    return ChatRequest(tuple(msgs), stream, include_usage, mt, user, tuple(ignored))


# --- responses ---------------------------------------------------------------------


def new_completion_id() -> str:
    return "chatcmpl-" + secrets.token_hex(12)


def _finite(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


def sieve_block(
    res: RouterResult,
    *,
    artifact_id: str,
    mode: str,
    ignored: tuple[str, ...],
) -> dict[str, Any]:
    last = res.traces[-1].decision if res.traces else None
    block: dict[str, Any] = {
        "decision": "answer" if res.action is Action.ANSWER else "abstain",
        "tier": res.served_tier,
        "reason": res.reason,
        "escalated": res.escalated,
        "mode": mode,
        "artifact_id": artifact_id,
        "score": _finite(last.score) if last else None,
        "threshold": _finite(last.threshold) if last else None,
        "cost": {
            "total_seconds": round(res.cost.total_seconds, 6),
            "wasted_seconds": round(res.cost.wasted_seconds, 6),
            "total_tokens": res.cost.total_tokens,
            "wasted_tokens": res.cost.wasted_tokens,
            "attempts": res.cost.n_attempts,
        },
        "ignored_params": list(ignored),
    }
    if res.mode == "shadow":
        block["shadow"] = {
            "action": None if res.shadow_action is None else res.shadow_action.value,
            "tier": res.shadow_tier,
            "reason": res.shadow_reason,
        }
    return block


@dataclass(frozen=True)
class Rendered:
    content: str
    finish_reason: str
    decision: str


def render(res: RouterResult, cfg: ServerSettings) -> Rendered:
    if res.action is Action.ANSWER and res.text is not None:
        return Rendered(res.text, "length" if res.finish == "length" else "stop", "answer")
    return Rendered(cfg.abstain_message, cfg.abstain_finish_reason, "abstain")


def completion_json(
    cid: str,
    model: str,
    created: int,
    r: Rendered,
    sieve: dict[str, Any],
    usage: tuple[int, int],
) -> dict[str, Any]:
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": r.content},
                "finish_reason": r.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": usage[0] + usage[1],
        },
        "sieve": sieve,
    }


def _chunk(
    cid: str, model: str, created: int, delta: dict[str, Any], finish: str | None
) -> dict[str, Any]:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _sse(obj: dict[str, Any] | str) -> bytes:
    payload = obj if isinstance(obj, str) else json.dumps(obj, separators=(",", ":"))
    return f"data: {payload}\n\n".encode()


def _pieces(text: str, size: int = 24) -> Iterator[str]:
    for i in range(0, len(text), size):
        yield text[i : i + size]


def sse_events(
    cid: str,
    model: str,
    created: int,
    r: Rendered,
    sieve: dict[str, Any],
    usage: tuple[int, int] | None,
) -> Iterator[bytes]:
    """The gate has already decided: replay the final text, then finish + extension."""
    first = _chunk(cid, model, created, {"role": "assistant", "content": ""}, None)
    first["sieve"] = {
        "decision": sieve["decision"],
        "tier": sieve["tier"],
        "reason": sieve["reason"],
    }
    yield _sse(first)
    for piece in _pieces(r.content):
        yield _sse(_chunk(cid, model, created, {"content": piece}, None))
    last = _chunk(cid, model, created, {}, r.finish_reason)
    last["sieve"] = sieve
    yield _sse(last)
    if usage is not None:
        u = _chunk(cid, model, created, {}, None)
        u["choices"] = []
        u["usage"] = {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": usage[0] + usage[1],
        }
        yield _sse(u)
    yield _sse("[DONE]")


def now() -> int:
    return int(time.time())

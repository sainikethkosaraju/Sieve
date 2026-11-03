"""Prompt rendering. The router renders the chat template itself so the exact string
sent to ``/completion`` is known, versioned and hashed into ``BackendInfo.template_hash``."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sieve import prompting

Role = Literal["system", "user", "assistant"]

#: Special-token-looking sequences in client text (``<|im_start|>`` etc.) are defanged so a
#: client cannot forge roles or terminate the turn.
_SPECIAL = re.compile(r"<\|")

TEMPLATE_VERSION = 1


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class ChatTemplate:
    """``calibrated`` (default, D-019): the last user message is placed into the shared
    calibration template (``sieve.prompting``), the same text the eval harness certified.
    ``chatml`` (Qwen2.5 family) and ``plain`` are legacy renderers whose hash can never match
    an eval-certified artifact."""

    name: Literal["calibrated", "chatml", "plain"] = "calibrated"
    system_prompt: str = ""

    def render(self, messages: Sequence[ChatMessage]) -> str:
        msgs = [ChatMessage(m.role, _SPECIAL.sub("< |", m.content)) for m in messages]
        if self.system_prompt and not any(m.role == "system" for m in msgs):
            msgs.insert(0, ChatMessage("system", self.system_prompt))
        if self.name == "calibrated":
            users = [m.content for m in msgs if m.role == "user"]
            return prompting.render_calibration_prompt(users[-1] if users else "")
        if self.name == "plain":
            return "\n".join(m.content for m in msgs if m.role != "system") + "\n"
        parts = [f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n" for m in msgs]
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    @property
    def hash(self) -> str:
        if self.name == "calibrated":
            return prompting.template_hash()
        body = json.dumps(
            {"name": self.name, "system": self.system_prompt, "v": TEMPLATE_VERSION},
            sort_keys=True,
        )
        return hashlib.sha256(body.encode()).hexdigest()[:32]

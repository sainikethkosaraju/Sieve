"""OpenAI-style error envelope. Messages are static strings: they never echo request
content (privacy) or unvalidated client strings (header / log injection)."""

from __future__ import annotations

from typing import Any


class OpenAIError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        type_: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
        reject: str = "invalid",
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type_ = type_
        self.code = code
        self.param = param
        self.headers = headers or {}
        self.reject = reject  # bounded label for sieve_rejected_total

    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.type_,
                "param": self.param,
                "code": self.code,
            }
        }


class ClientDisconnected(Exception):
    """The HTTP client went away; backend work has been cancelled."""

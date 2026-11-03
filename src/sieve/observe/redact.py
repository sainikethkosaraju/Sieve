"""Redaction hooks for the (opt-in) content fields of the audit trail."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence

Redactor = Callable[[str], str]

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_KEY = re.compile(
    r"(?i)\b(?:sk|pk|api|key|token|secret)[-_][A-Za-z0-9_-]{8,}\b|\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"
)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}\b")
_CLOUD_KEY = re.compile(
    r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b"
)
_PASSWORD = re.compile(r"(?i)\b(?:password|passwd|pwd)\s*[=:]\s*\S+")
_DIGITS = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")


def regex_redactor(pattern: str, replacement: str = "[REDACTED]") -> Redactor:
    rx = re.compile(pattern)
    return lambda s: rx.sub(replacement, s)


def default_redactors() -> tuple[Redactor, ...]:
    """Best-effort: emails, API/cloud keys, bearer tokens, ``password=``, SSNs, phone numbers and
    13-19 digit numbers. Not a guarantee; keep ``log_content`` off for sensitive traffic."""
    return (
        lambda s: _EMAIL.sub("[EMAIL]", s),
        lambda s: _KEY.sub("[KEY]", s),
        lambda s: _DIGITS.sub("[NUMBER]", s),
        lambda s: _CLOUD_KEY.sub("[KEY]", s),
        lambda s: _PASSWORD.sub("[PASSWORD]", s),
        lambda s: _SSN.sub("[SSN]", s),
        lambda s: _PHONE.sub("[PHONE]", s),
    )


def apply_redactors(text: str, redactors: Iterable[Redactor]) -> str:
    for r in redactors:
        text = r(text)
    return text


def compose(*groups: Sequence[Redactor]) -> tuple[Redactor, ...]:
    return tuple(r for g in groups for r in g)

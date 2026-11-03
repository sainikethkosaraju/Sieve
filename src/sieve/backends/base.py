"""Backend plumbing shared by every adapter.

The :class:`~sieve.types.Backend` Protocol (``info`` + ``generate``) is re-exported
unchanged. Adapters are the only code that performs network IO. They never trust the
upstream: they map responses *faithfully* (including malformed logprobs) so that
``sieve.signals.validate_completion`` and the gate can flag the defects.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from sieve.types import Backend, BackendInfo, Completion, SamplingParams

__all__ = [
    "DEFAULT_ALLOWED_HOSTS",
    "Backend",
    "BackendError",
    "BackendHealth",
    "BackendUrlError",
    "Managed",
    "SelfTestResult",
    "validate_backend_url",
]

#: Hosts a backend URL may use unless the operator extends the allow-list.
DEFAULT_ALLOWED_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

#: Never allowed, even if an operator lists them (cloud metadata endpoints).
_BLOCKED_HOSTS: frozenset[str] = frozenset(
    {"169.254.169.254", "fd00:ec2::254", "metadata.google.internal", "metadata", "100.100.100.200"}
)


class BackendError(RuntimeError):
    """Transport / protocol failure talking to a backend (never carries prompt text)."""


class BackendUrlError(ValueError):
    """A backend URL violates the static allow-list policy."""


def validate_backend_url(url: str, allowed_hosts: Collection[str] = DEFAULT_ALLOWED_HOSTS) -> str:
    """Return a normalised base URL or raise :class:`BackendUrlError`.

    Policy (SSRF defence; backend URLs come from operator config only, never from a
    request): http(s) only, no credentials, no query/fragment, exact-match host
    allow-list, link-local / unspecified / multicast / metadata hosts always refused.
    No DNS resolution is performed; a hostname on the allow-list is trusted by the
    operator (DNS rebinding of an allow-listed name is out of scope, see docs/server.md).
    """
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as e:
        raise BackendUrlError("malformed backend URL") from e
    if parts.scheme not in ("http", "https"):
        raise BackendUrlError("backend URL scheme must be http or https")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise BackendUrlError("credentials in backend URL are not allowed")
    if parts.query or parts.fragment:
        raise BackendUrlError("query/fragment not allowed in backend URL")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise BackendUrlError("backend URL has no host")
    if host in _BLOCKED_HOSTS:
        raise BackendUrlError("backend host is blocked")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if (
            ip.is_link_local
            or ip.is_unspecified
            or ip.is_multicast
            or (ip.is_reserved and not ip.is_loopback)
        ):
            raise BackendUrlError("backend IP address class is blocked")
        host = str(ip)
    allowed = {h.rstrip(".").lower() for h in allowed_hosts}
    if host not in allowed:
        raise BackendUrlError("backend host is not in the allow-list")
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc += f":{port}"
    return f"{parts.scheme}://{netloc}{parts.path.rstrip('/')}"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    advisory: bool = False  # a failed advisory check is reported but does not fail the self-test


@dataclass(frozen=True)
class SelfTestResult:
    """Outcome of the startup logprob self-test. ``ok`` is the AND of all checks."""

    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.ok or c.advisory for c in self.checks)

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.checks if not c.ok and not c.advisory)

    @property
    def advisories(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.checks if not c.ok and c.advisory)


@dataclass
class BackendHealth:
    """Mutable health view of one backend, owned by the server and read by the router.

    ``logprobs_ok`` is ``False`` until a self-test passes, so a backend that was never
    tested cannot contribute logprob-based answers (fail closed).
    """

    reachable: bool = False
    logprobs_ok: bool = False
    self_test: SelfTestResult | None = None
    detail: str = ""
    extras: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class Managed(Protocol):
    """Optional lifecycle surface of adapters that talk to a real server."""

    @property
    def info(self) -> BackendInfo: ...

    async def generate(
        self, prompt: str, params: SamplingParams, *, n_probs: int
    ) -> Completion: ...

    async def refresh(self) -> bool:
        """Re-read upstream identity; return True if ``info`` changed."""
        ...

    async def aclose(self) -> None: ...

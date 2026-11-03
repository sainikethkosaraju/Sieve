"""Server configuration: pydantic settings, YAML or environment, validated at startup.

Backend URLs exist ONLY here. No request field or header can select or override a
backend. Environment variables use the ``SIEVE_`` prefix and ``__`` for nesting
(``SIEVE_LIMITS__MAX_CONCURRENCY=4``); an explicit YAML/constructor value wins.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sieve.backends.base import DEFAULT_ALLOWED_HOSTS, BackendUrlError, validate_backend_url


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MockSettings(_Strict):
    accuracy: float = Field(1.0, ge=0, le=1)
    correct_conf: float = Field(0.97, gt=0, lt=1)
    wrong_conf: float = Field(0.35, gt=0, lt=1)
    latency_s: float = Field(0.0, ge=0, le=10)
    seed: int = 0


class BackendSettings(_Strict):
    kind: Literal["llamaserver", "ollama", "mock"]
    url: str | None = None
    model_id: str = Field(min_length=1, max_length=200)
    model_hash: str = Field(min_length=1, max_length=200)
    quant: str = Field(min_length=1, max_length=64)
    kv_quant: str = Field("f16", min_length=1, max_length=64)
    #: pinned build string; for llama-server the actual build comes from /props
    build: str = ""
    expected_model_file: str | None = None
    timeout_s: float = Field(60.0, gt=0, le=600)
    slots: int = Field(4, ge=1, le=64)
    n_probs: int = Field(20, ge=2, le=64)
    mock: MockSettings = Field(default_factory=MockSettings)

    @model_validator(mode="after")
    def _check(self) -> BackendSettings:
        if self.kind != "mock" and not self.url:
            raise ValueError("url is required for non-mock backends")
        if self.kind == "mock" and self.url:
            raise ValueError("mock backends take no url")
        return self


class Limits(_Strict):
    max_body_bytes: int = Field(1_048_576, ge=1024, le=64 * 1024 * 1024)
    max_messages: int = Field(64, ge=1, le=1024)
    max_message_chars: int = Field(32_000, ge=1)
    max_prompt_chars: int = Field(64_000, ge=1)
    max_tokens_cap: int = Field(256, ge=1, le=8192)
    body_read_timeout_s: float = Field(10.0, gt=0, le=120)
    request_deadline_s: float = Field(60.0, gt=0, le=600)
    max_concurrency: int = Field(8, ge=1, le=256)
    max_queue: int = Field(32, ge=0, le=4096)
    queue_timeout_s: float = Field(5.0, gt=0, le=60)
    retry_after_s: int = Field(1, ge=1, le=60)


class GuardSettings(_Strict):
    #: escalation circuit breaker; >= 1.0 disables it. Untuned DoS safeguard.
    max_escalation_rate: float = Field(0.8, ge=0)
    window: int = Field(200, ge=1, le=100_000)


class AuditSettings(_Strict):
    path: str | None = None
    log_content: bool = False
    redact: bool = True
    max_queue: int = Field(10_000, ge=1)
    user_hash_salt: SecretStr | None = None


class DriftSettings(_Strict):
    reference_path: str | None = None
    reference_sha256: str | None = None
    window: int = Field(200, ge=20)
    min_window: int = Field(100, ge=20)
    check_every: int = Field(20, ge=1)
    ks_alpha: float = Field(1e-3, gt=0, lt=1)
    psi_threshold: float = Field(0.2, gt=0)


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SIEVE_", env_nested_delimiter="__", extra="forbid", frozen=True
    )

    host: str = "127.0.0.1"
    port: int = Field(8080, ge=1, le=65535)
    #: API keys (Bearer). Required when binding a non-loopback address.
    api_keys: tuple[SecretStr, ...] = ()
    metrics_public: bool = False

    model_name: str = Field("sieve", pattern=r"^[A-Za-z0-9._:-]{1,64}$")
    model_aliases: tuple[str, ...] = ()
    artifact_path: str
    mode: Literal["enforce", "shadow"] = "enforce"

    small: BackendSettings
    large: BackendSettings | None = None
    allowed_backend_hosts: tuple[str, ...] = tuple(sorted(DEFAULT_ALLOWED_HOSTS))

    chat_template: Literal["calibrated", "chatml", "plain"] = "calibrated"
    system_prompt: str = ""
    abstain_message: str = "I can't answer that reliably."
    abstain_finish_reason: Literal["sieve_abstain", "content_filter", "stop"] = "sieve_abstain"
    #: what to do with client temperature/top_p/seed/stop/penalties: the gate is
    #: calibrated for the artifact's decoding config, so they are ignored (or rejected)
    client_sampling: Literal["ignore", "reject"] = "ignore"

    require_self_test: bool = True
    health_interval_s: float = Field(30.0, ge=0)
    self_test_repeats: int = Field(3, ge=2, le=10)

    limits: Limits = Field(default_factory=Limits)
    guard: GuardSettings = Field(default_factory=GuardSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    drift: DriftSettings = Field(default_factory=DriftSettings)

    expected_config_pin: str | None = None
    expected_artifact_id: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> ServerSettings:
        if not _is_loopback(self.host) and not self.api_keys:
            raise ValueError("api_keys are required when binding a non-loopback host")
        for k in self.api_keys:
            if len(k.get_secret_value()) < 8:
                raise ValueError("api keys must be at least 8 characters")
        for b in (self.small, self.large):
            if b is not None and b.url is not None:
                try:
                    validate_backend_url(b.url, self.allowed_backend_hosts)
                except BackendUrlError as e:
                    raise ValueError(f"backend url rejected: {e}") from e
        if self.mode == "shadow" and self.large is None:
            raise ValueError("shadow mode needs a large (baseline) tier")
        if self.limits.max_message_chars > self.limits.max_prompt_chars:
            raise ValueError("max_message_chars must be <= max_prompt_chars")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> ServerSettings:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("config root must be a mapping")
        data.update(overrides)
        return cls(**data)

    def pin_view(self) -> dict[str, Any]:
        """Secret-free effective settings used for the config pin."""
        d = self.model_dump(mode="json", exclude={"api_keys", "expected_config_pin"})
        d["audit"].pop("user_hash_salt", None)
        d["audit"]["path"] = bool(d["audit"]["path"])
        return d

    @property
    def accepted_models(self) -> frozenset[str]:
        return frozenset({self.model_name, *self.model_aliases})


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

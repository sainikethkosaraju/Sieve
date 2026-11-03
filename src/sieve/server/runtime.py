"""Assemble the runtime (backends, router, observability) from validated settings."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry

from sieve.backends import (
    Backend,
    BackendHealth,
    LlamaServerBackend,
    Managed,
    MockBackend,
    OllamaBackend,
    run_logprob_self_test,
)
from sieve.calibrate.store import loads_artifact
from sieve.gate import Gate
from sieve.observe import (
    AuditRecord,
    AuditSink,
    DriftMonitor,
    JsonlAuditSink,
    NullAuditSink,
    SieveMetrics,
    UserHasher,
    check_pins,
    config_pin,
    default_redactors,
    load_reference,
)
from sieve.observe.metrics import bounded
from sieve.observe.redact import Redactor
from sieve.router import ChatTemplate, EscalationGuard, Router, Tier
from sieve.server.config import BackendSettings, ServerSettings
from sieve.server.limits import ConcurrencyLimiter
from sieve.types import BackendInfo, CalibrationArtifact

log = logging.getLogger("sieve.server")


def make_backend(
    cfg: BackendSettings, template_hash: str, allowed_hosts: tuple[str, ...]
) -> Backend:
    common: dict[str, Any] = {
        "model_id": cfg.model_id,
        "model_hash": cfg.model_hash,
        "quant": cfg.quant,
        "kv_quant": cfg.kv_quant,
        "template_hash": template_hash,
        "build": cfg.build,
    }
    if cfg.kind == "llamaserver":
        assert cfg.url is not None
        return LlamaServerBackend(
            cfg.url,
            expected_model_file=cfg.expected_model_file,
            allowed_hosts=allowed_hosts,
            timeout_s=cfg.timeout_s,
            **common,
        )
    if cfg.kind == "ollama":
        assert cfg.url is not None
        return OllamaBackend(
            cfg.url, allowed_hosts=allowed_hosts, timeout_s=cfg.timeout_s, **common
        )
    m = cfg.mock
    return MockBackend(
        info=BackendInfo(
            cfg.model_id,
            cfg.model_hash,
            cfg.quant,
            cfg.kv_quant,
            cfg.build or "mock",
            template_hash,
        ),
        accuracy=m.accuracy,
        correct_conf=m.correct_conf,
        wrong_conf=m.wrong_conf,
        latency_s=m.latency_s,
        seed=m.seed,
    )


@dataclass
class AppState:
    settings: ServerSettings
    artifact: CalibrationArtifact
    template: ChatTemplate
    router: Router
    small: Tier
    large: Tier | None
    metrics: SieveMetrics
    sink: AuditSink
    limiter: ConcurrencyLimiter
    hasher: UserHasher
    drift: DriftMonitor | None
    pin: str
    redactors: tuple[Redactor, ...]
    ready_detail: dict[str, Any] = field(default_factory=dict)
    started: bool = False
    drift_alarm_active: bool = False
    _probed: dict[int, tuple[str, ...]] = field(default_factory=dict)
    _maint: asyncio.Task[None] | None = None

    # -- lifecycle --------------------------------------------------------------------

    @property
    def tiers(self) -> list[tuple[int, Tier]]:
        out = [(1, self.small)]
        if self.large is not None:
            out.append((2, self.large))
        return out

    async def _probe(self, n: int, tier: Tier) -> None:
        """Refresh identity and run the logprob self-test; never raises."""
        h = tier.health
        try:
            if isinstance(tier.backend, Managed):
                await tier.backend.refresh()
            h.reachable = True
            self._probed[n] = tier.backend.info.key()
        except Exception as e:
            h.reachable, h.logprobs_ok = False, False
            h.detail = f"unreachable ({type(e).__name__})"
            self._export_health(n, tier)
            return
        if self.settings.require_self_test:
            res = await run_logprob_self_test(
                tier.backend,
                n_probs=tier.n_probs,
                repeats=self.settings.self_test_repeats,
            )
            h.self_test = res
            h.logprobs_ok = res.ok
            h.reachable = h.reachable and "reachable" not in res.failed
            h.detail = "" if res.ok else "self-test failed: " + ",".join(res.failed)
        else:
            h.logprobs_ok = True
            h.detail = "self-test skipped by configuration"
        self._export_health(n, tier)

    def _export_health(self, n: int, tier: Tier) -> None:
        self.metrics.logprobs_ok.labels(str(n)).set(1 if tier.health.logprobs_ok else 0)
        self.metrics.backend_up.labels(str(n)).set(1 if tier.health.reachable else 0)

    async def startup(self) -> None:
        for n, tier in self.tiers:
            await self._probe(n, tier)
        self.started = True
        self.update_ready()
        iv = self.settings.health_interval_s
        if iv > 0 and self._maint is None:
            self._maint = asyncio.create_task(self._maintain(iv), name="sieve-maintain")

    async def _maintain(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            for n, tier in self.tiers:
                try:
                    changed = (
                        await tier.backend.refresh() if isinstance(tier.backend, Managed) else False
                    )
                    tier.health.reachable = True
                except Exception:
                    tier.health.reachable = False
                    changed = False
                if changed or tier.backend.info.key() != self._probed.get(n):
                    await self._probe(n, tier)  # identity changed since the last self-test
                self._export_health(n, tier)
            self.update_ready()

    async def shutdown(self) -> None:
        if self._maint is not None:
            self._maint.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._maint
            self._maint = None
        for _, tier in self.tiers:
            if isinstance(tier.backend, Managed):
                with contextlib.suppress(Exception):
                    await tier.backend.aclose()
        self.sink.close()

    # -- readiness --------------------------------------------------------------------

    def readiness(self) -> tuple[bool, dict[str, Any]]:
        tiers: dict[str, Any] = {}
        ok = self.started
        for n, tier in self.tiers:
            tc = self.artifact.tiers[n - 1]
            identity_ok = tier.backend.info.key() == tc.identity.key()
            t_ok = tier.health.reachable and tier.health.logprobs_ok and identity_ok
            ok = ok and t_ok
            tiers[str(n)] = {
                "reachable": tier.health.reachable,
                "logprobs_ok": tier.health.logprobs_ok,
                "identity_matches_artifact": identity_ok,
                "detail": tier.health.detail,
            }
        if self.large is None:
            ok = False  # a cascade without a large tier is not a certified configuration
            tiers["2"] = {"configured": False}
        return ok, {
            "ready": ok,
            "mode": self.settings.mode,
            "artifact_id": self.artifact.artifact_id,
            "config_pin": self.pin,
            "tiers": tiers,
        }

    def update_ready(self) -> None:
        ok, detail = self.readiness()
        self.ready_detail = detail
        self.metrics.ready.set(1 if ok else 0)

    # -- recording --------------------------------------------------------------------

    def record_guard(self) -> None:
        g = self.router.guard
        self.metrics.guard_open.set(1 if g.open else 0)
        self.metrics.escalation_rate.set(g.rate)

    def export_audit_drops(self) -> None:
        d = getattr(self.sink, "dropped", 0)
        self.metrics.audit_dropped.set(d)

    def label(self, value: str, allowed: frozenset[str]) -> str:
        return bounded(value, allowed)

    def new_audit_record(self, **kw: Any) -> AuditRecord:
        return AuditRecord(config_pin=self.pin, artifact_id=self.artifact.artifact_id, **kw)


def build_state(
    settings: ServerSettings,
    *,
    artifact: CalibrationArtifact | None = None,
    small: Backend | None = None,
    large: Backend | None = None,
    sink: AuditSink | None = None,
    registry: CollectorRegistry | None = None,
) -> AppState:
    """Create the runtime. Injected ``artifact`` / backends / sink are for tests."""
    if artifact is None:
        artifact = loads_artifact(Path(settings.artifact_path).read_text(encoding="utf-8"))
    pin = config_pin(settings.pin_view())
    check_pins(
        config=pin,
        artifact_id=artifact.artifact_id,
        expected_config=settings.expected_config_pin,
        expected_artifact=settings.expected_artifact_id,
    )
    template = ChatTemplate(settings.chat_template, settings.system_prompt)
    hosts = settings.allowed_backend_hosts
    sb = small or make_backend(settings.small, template.hash, hosts)
    lb: Backend | None = large
    if lb is None and settings.large is not None:
        lb = make_backend(settings.large, template.hash, hosts)
    st = Tier(sb, BackendHealth(), settings.small.slots, settings.small.n_probs)
    lt = (
        Tier(
            lb,
            BackendHealth(),
            (settings.large or settings.small).slots,
            (settings.large or settings.small).n_probs,
        )
        if lb is not None
        else None
    )
    gate = Gate(artifact)
    guard = EscalationGuard(settings.guard.max_escalation_rate, settings.guard.window)
    router = Router(gate, st, lt, guard=guard)
    metrics = SieveMetrics(registry)
    metrics.build_info.labels(artifact.artifact_id, pin).set(1)
    if sink is None:
        sink = (
            JsonlAuditSink(settings.audit.path, settings.audit.max_queue)
            if settings.audit.path
            else NullAuditSink()
        )
    drift: DriftMonitor | None = None
    d = settings.drift
    if d.reference_path:
        drift = DriftMonitor(
            load_reference(d.reference_path, d.reference_sha256),
            window=d.window,
            min_window=d.min_window,
            check_every=d.check_every,
            ks_alpha=d.ks_alpha,
            psi_threshold=d.psi_threshold,
        )
    salt = settings.audit.user_hash_salt
    return AppState(
        settings=settings,
        artifact=artifact,
        template=template,
        router=router,
        small=st,
        large=lt,
        metrics=metrics,
        sink=sink,
        limiter=ConcurrencyLimiter(
            settings.limits.max_concurrency,
            settings.limits.max_queue,
            settings.limits.queue_timeout_s,
        ),
        hasher=UserHasher(salt.get_secret_value().encode() if salt else None),
        drift=drift,
        pin=pin,
        redactors=default_redactors() if settings.audit.redact else (),
    )

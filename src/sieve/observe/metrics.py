"""Prometheus metrics with bounded label cardinality.

Every label value comes from a closed set (enums, tier numbers, HTTP status classes,
fixed route names). User-controlled strings (prompt, model name, user id, request id)
are never used as labels; :func:`bounded` maps anything outside the allowed set to
``other`` so even a bug cannot create unbounded series.
"""

from __future__ import annotations

from collections.abc import Collection

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sieve.router.core import RouterReason
from sieve.types import ReasonCode

REASONS: frozenset[str] = frozenset(
    {r.value for r in ReasonCode} | {r.value for r in RouterReason} | {"shadow", "none"}
)
DECISIONS: frozenset[str] = frozenset({"answer", "abstain", "none"})
TIERS: frozenset[str] = frozenset({"1", "2", "none"})
MODES: frozenset[str] = frozenset({"enforce", "shadow"})
OUTCOMES: frozenset[str] = frozenset({"ok", "timeout", "cancelled", "error"})
ROUTES: frozenset[str] = frozenset(
    {"/v1/chat/completions", "/v1/models", "/healthz", "/readyz", "/metrics", "other"}
)
STATUS_CLASSES: frozenset[str] = frozenset({"2xx", "3xx", "4xx", "5xx", "other"})
REJECTIONS: frozenset[str] = frozenset(
    {"too_large", "invalid", "unsupported", "auth", "backpressure", "unknown_model", "type"}
)
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)
SCORE_BUCKETS = (-8, -4, -2, -1, -0.5, -0.25, -0.1, -0.05, -0.01, 0, 0.25, 0.5, 0.75, 0.9, 0.99, 1)


def bounded(value: str, allowed: Collection[str]) -> str:
    return value if value in allowed else "other"


class SieveMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        r = self.registry
        self.http_requests = Counter(
            "sieve_http_requests_total", "HTTP requests", ["route", "status"], registry=r
        )
        self.decisions = Counter(
            "sieve_decisions_total",
            "Final decisions",
            ["decision", "tier", "reason", "mode"],
            registry=r,
        )
        self.would_decisions = Counter(
            "sieve_shadow_would_decide_total",
            "Shadow mode: decision enforce mode would have taken",
            ["decision", "tier", "reason"],
            registry=r,
        )
        self.escalations = Counter("sieve_escalations_total", "Escalations to tier 2", registry=r)
        self.rejected = Counter(
            "sieve_rejected_total", "Requests rejected before routing", ["reason"], registry=r
        )
        self.backend_errors = Counter(
            "sieve_backend_errors_total", "Backend call failures", ["tier"], registry=r
        )
        self.defects = Counter(
            "sieve_signal_defects_total",
            "Requests whose signals carried a defect",
            ["tier"],
            registry=r,
        )
        self.request_seconds = Histogram(
            "sieve_request_seconds",
            "End-to-end request latency (includes wasted small-tier work)",
            ["decision", "outcome"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.cost_seconds = Counter(
            "sieve_cost_seconds_total", "Backend seconds by tier", ["tier"], registry=r
        )
        self.wasted_seconds = Counter(
            "sieve_wasted_seconds_total",
            "Backend seconds not contributing to the served answer",
            registry=r,
        )
        self.tokens = Counter(
            "sieve_tokens_total", "Tokens processed (prompt+generated)", ["kind"], registry=r
        )
        self.score = Histogram(
            "sieve_score", "Gate score by tier", ["tier"], buckets=SCORE_BUCKETS, registry=r
        )
        self.inflight = Gauge("sieve_inflight_requests", "Requests being processed", registry=r)
        self.queued = Gauge("sieve_queued_requests", "Requests waiting for a slot", registry=r)
        self.guard_open = Gauge(
            "sieve_escalation_guard_open", "1 if escalations are currently being shed", registry=r
        )
        self.escalation_rate = Gauge(
            "sieve_escalation_rate_window", "Escalation rate in the guard window", registry=r
        )
        self.logprobs_ok = Gauge(
            "sieve_backend_logprobs_ok", "Logprob self-test passed", ["tier"], registry=r
        )
        self.backend_up = Gauge("sieve_backend_up", "Backend reachable", ["tier"], registry=r)
        self.drift_ks_p = Gauge(
            "sieve_drift_ks_pvalue", "KS p-value of window vs reference", registry=r
        )
        self.drift_psi = Gauge("sieve_drift_psi", "PSI of window vs reference", registry=r)
        self.drift_alarm = Gauge("sieve_drift_alarm", "1 if the drift alarm is active", registry=r)
        self.audit_dropped = Gauge("sieve_audit_dropped_total", "Audit records dropped", registry=r)
        self.build_info = Gauge(
            "sieve_build_info", "Pinned versions", ["artifact_id", "config_pin"], registry=r
        )
        self.ready = Gauge("sieve_ready", "1 if /readyz would return 200", registry=r)

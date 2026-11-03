"""FastAPI application: OpenAI-compatible ``/v1/chat/completions`` over the Sieve cascade.

Streaming is *buffered gating*: the whole cascade (small -> gate -> large -> gate) runs
first and the SSE stream replays the final text only after the decision, so abstention
can still be signalled before the first byte. Time-to-first-token therefore equals the
full generation time of the tier(s) used.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sieve.backends import Backend
from sieve.observe import AuditSink, apply_content_policy
from sieve.observe.metrics import (
    DECISIONS,
    MODES,
    OUTCOMES,
    REASONS,
    REJECTIONS,
    ROUTES,
    STATUS_CLASSES,
    TIERS,
    bounded,
)
from sieve.router import CostLedger, RouterResult, UpstreamUnavailable
from sieve.server import schemas
from sieve.server.config import ServerSettings
from sieve.server.errors import ClientDisconnected, OpenAIError
from sieve.server.limits import Backpressure, read_body, run_cancellable
from sieve.server.runtime import AppState, build_state
from sieve.types import SCORE_NAMES, Action, CalibrationArtifact

log = logging.getLogger("sieve.server")


class _ObsMiddleware:
    """Pure-ASGI middleware (keeps ``receive`` untouched so disconnects propagate):
    sanitised request id in/out and bounded-label HTTP metrics."""

    def __init__(self, app: ASGIApp, state: AppState) -> None:
        self.app = app
        self.state = state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        raw = next((v for k, v in scope["headers"] if k == b"x-request-id"), None)
        rid = schemas.safe_request_id(raw.decode("latin-1") if raw is not None else None)
        scope.setdefault("state", {})["request_id"] = rid
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                message["headers"] = [*message.get("headers", []), (b"x-request-id", rid.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            route = bounded(str(scope.get("path", "")), ROUTES)
            klass = f"{status // 100}xx"
            m = self.state.metrics
            m.http_requests.labels(route, bounded(klass, STATUS_CLASSES)).inc()


def _authorized(request: Request, st: AppState) -> None:
    keys = st.settings.api_keys
    if not keys:
        return
    scheme, _, tok = request.headers.get("authorization", "").partition(" ")
    digest = hashlib.sha256(tok.strip().encode()).digest()
    ok = False
    for k in keys:  # no short-circuit: constant work per configured key
        ok |= hmac.compare_digest(digest, hashlib.sha256(k.get_secret_value().encode()).digest())
    if scheme.lower() != "bearer" or not ok:
        raise OpenAIError(
            401,
            "Invalid or missing API key",
            type_="authentication_error",
            code="invalid_api_key",
            headers={"WWW-Authenticate": "Bearer"},
            reject="auth",
        )


def _error_response(e: OpenAIError, st: AppState | None) -> JSONResponse:
    if st is not None:
        st.metrics.rejected.labels(bounded(e.reject, REJECTIONS)).inc()
    return JSONResponse(e.body(), status_code=e.status, headers=e.headers)


def create_app(
    settings: ServerSettings,
    *,
    artifact: CalibrationArtifact | None = None,
    small: Backend | None = None,
    large: Backend | None = None,
    sink: AuditSink | None = None,
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    """Build the app. ``artifact``, ``small``, ``large``, ``sink`` and ``registry`` are
    injection points for tests; production builds everything from ``settings``."""
    st = build_state(
        settings, artifact=artifact, small=small, large=large, sink=sink, registry=registry
    )

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await st.startup()
        try:
            yield
        finally:
            await st.shutdown()

    app = FastAPI(
        title="Sieve",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.sieve = st
    app.add_middleware(_ObsMiddleware, state=st)

    # -- error handlers ----------------------------------------------------------------

    @app.exception_handler(OpenAIError)
    async def _openai(_: Request, e: OpenAIError) -> JSONResponse:
        return _error_response(e, st)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, e: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(e.status_code, "http_error")
        return _error_response(OpenAIError(e.status_code, "Request failed", code=code), st)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError) -> JSONResponse:
        return _error_response(OpenAIError(400, "Invalid request", code="invalid_request"), st)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, e: Exception) -> JSONResponse:
        log.error("unhandled error: %s", type(e).__name__)
        return _error_response(
            OpenAIError(500, "Internal server error", type_="server_error", code="internal_error"),
            st,
        )

    # -- probes & metrics ----------------------------------------------------------------

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        st.update_ready()
        ok, detail = st.readiness()
        return JSONResponse(detail, status_code=200 if ok else 503)

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        if not settings.metrics_public:
            _authorized(request, st)
        st.export_audit_drops()
        st.update_ready()
        st.record_guard()
        return Response(generate_latest(st.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, Any]:
        _authorized(request, st)
        created = schemas.now()
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "created": created, "owned_by": "sieve"}
                for m in sorted(settings.accepted_models)
            ],
        }

    # -- chat --------------------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        rid: str = request.state.request_id
        try:
            return await _chat(request, rid)
        except OpenAIError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:  # never leak content via tracebacks / messages
            log.error("chat handler failed: %s", type(e).__name__)
            raise OpenAIError(
                500, "Internal server error", type_="server_error", code="internal_error"
            ) from None

    async def _chat(request: Request, rid: str) -> Response:
        _authorized(request, st)
        schemas.reject_override_headers(list(request.headers.keys()))
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise OpenAIError(
                415, "Content-Type must be application/json", code="bad_content_type", reject="type"
            )
        body = await read_body(
            request, settings.limits.max_body_bytes, settings.limits.body_read_timeout_s
        )
        req = schemas.validate_chat(schemas.parse_json(body), settings)
        prompt = st.template.render(req.messages)

        ledger = CostLedger()
        t0 = time.monotonic()
        shadow = settings.mode == "shadow"

        async def work() -> RouterResult:
            async with st.limiter.slot():
                return await st.router.run(
                    prompt, max_tokens=req.max_tokens, ledger=ledger, shadow=shadow
                )

        st.metrics.inflight.inc()
        result: RouterResult | None = None
        outcome = "ok"
        try:
            async with asyncio.timeout(settings.limits.request_deadline_s):
                result = await run_cancellable(request, work())
        except Backpressure:
            outcome = "error"
            raise OpenAIError(
                429,
                "Server is at capacity",
                type_="rate_limit_error",
                code="rate_limit_exceeded",
                headers={"Retry-After": str(settings.limits.retry_after_s)},
                reject="backpressure",
            ) from None
        except TimeoutError:
            outcome = "timeout"
            raise OpenAIError(
                504, "Request deadline exceeded", type_="server_error", code="deadline_exceeded"
            ) from None
        except ClientDisconnected:
            outcome = "cancelled"
            return Response(status_code=499)
        except UpstreamUnavailable:
            outcome = "error"
            raise OpenAIError(
                502, "Upstream unavailable", type_="server_error", code="upstream_unavailable"
            ) from None
        finally:
            st.metrics.inflight.dec()
            st.metrics.queued.set(st.limiter.queued)
            if outcome != "ok" or result is None:
                _finalize(
                    rid, req, prompt, None, ledger, outcome if outcome != "ok" else "error", t0
                )

        _finalize(rid, req, prompt, result, ledger, "ok", t0)
        rendered = schemas.render(result, settings)
        sieve = schemas.sieve_block(
            result, artifact_id=st.artifact.artifact_id, mode=settings.mode, ignored=req.ignored
        )
        usage = (
            sum(a.n_prompt for a in ledger.attempts),
            sum(a.n_gen for a in ledger.attempts),
        )
        cid, created = schemas.new_completion_id(), schemas.now()
        headers = {
            "X-Sieve-Decision": rendered.decision,
            "X-Sieve-Tier": str(result.served_tier) if result.served_tier else "none",
            "X-Sieve-Artifact": st.artifact.artifact_id,
            "Cache-Control": "no-store",
        }
        if req.stream:
            events = schemas.sse_events(
                cid,
                settings.model_name,
                created,
                rendered,
                sieve,
                usage if req.include_usage else None,
            )

            async def gen() -> AsyncIterator[bytes]:
                for ev in events:
                    yield ev
                    await asyncio.sleep(0)

            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers={**headers, "X-Accel-Buffering": "no"},
            )
        return JSONResponse(
            schemas.completion_json(cid, settings.model_name, created, rendered, sieve, usage),
            headers=headers,
        )

    def _finalize(
        rid: str,
        req: schemas.ChatRequest,
        prompt: str,
        result: RouterResult | None,
        ledger: CostLedger,
        outcome: str,
        t0: float,
    ) -> None:
        """Metrics + audit for one request. Never raises into the request path."""
        try:
            m = st.metrics
            latency = time.monotonic() - t0
            served = result.served_tier if result else None
            cost = ledger.summary(served)
            decision = "none"
            tier_label = "none"
            reason = "none"
            if result is not None:
                decision = "answer" if result.action is Action.ANSWER else "abstain"
                tier_label = str(served) if served else "none"
                reason = result.reason
                m.decisions.labels(
                    bounded(decision, DECISIONS),
                    bounded(tier_label, TIERS),
                    bounded(reason, REASONS),
                    bounded(settings.mode, MODES),
                ).inc()
                if result.escalated:
                    m.escalations.inc()
                if result.mode == "shadow" and result.shadow_action is not None:
                    m.would_decisions.labels(
                        bounded(result.shadow_action.value, DECISIONS),
                        bounded(str(result.shadow_tier or "none"), TIERS),
                        bounded(result.shadow_reason or "none", REASONS),
                    ).inc()
                for tr in result.traces:
                    t = bounded(str(tr.tier), TIERS)
                    if tr.decision.score is not None and tr.decision.score == tr.decision.score:
                        m.score.labels(t).observe(tr.decision.score)
                    if tr.defects:
                        m.defects.labels(t).inc()
                    if tr.error is not None:
                        m.backend_errors.labels(t).inc()
                if (
                    st.drift is not None
                    and result.traces
                    and not result.traces[0].skipped_generation
                ):
                    ds = st.drift.observe(result.traces[0].decision.score)
                    if ds.ks_pvalue is not None:
                        m.drift_ks_p.set(ds.ks_pvalue)
                    if ds.psi is not None:
                        m.drift_psi.set(ds.psi)
                    m.drift_alarm.set(1 if ds.alarm else 0)
                    if ds.alarm and not st.drift_alarm_active:
                        log.warning(
                            "drift alarm raised: ks_p=%s psi=%s n=%d",
                            ds.ks_pvalue,
                            ds.psi,
                            ds.n_window,
                        )
                    st.drift_alarm_active = ds.alarm
            for a in ledger.attempts:
                m.cost_seconds.labels(bounded(str(a.tier), TIERS)).inc(a.seconds)
            m.wasted_seconds.inc(cost.wasted_seconds)
            m.tokens.labels("prompt").inc(sum(a.n_prompt for a in ledger.attempts))
            m.tokens.labels("generated").inc(sum(a.n_gen for a in ledger.attempts))
            m.request_seconds.labels(
                bounded(decision, DECISIONS), bounded(outcome, OUTCOMES)
            ).observe(latency)
            st.record_guard()

            rec = st.new_audit_record(
                ts=time.time(),
                request_id=rid,
                mode=settings.mode,
                outcome=outcome,
                decision=decision,
                reason=reason,
                served_tier=served,
                escalated=bool(result and result.escalated),
                stream=req.stream,
                latency_s=round(latency, 6),
                prompt_chars=len(prompt),
                user_hash=st.hasher(req.user),
                cost={
                    "total_seconds": round(cost.total_seconds, 6),
                    "wasted_seconds": round(cost.wasted_seconds, 6),
                    "total_tokens": cost.total_tokens,
                    "wasted_tokens": cost.wasted_tokens,
                    "attempts": cost.n_attempts,
                },
                backend={f"tier{n}": _identity(t.backend.info) for n, t in st.tiers},
            )
            if result is not None:
                rec.tiers = [_tier_record(st, tr) for tr in result.traces]
                if result.mode == "shadow":
                    rec.shadow = {
                        "action": None
                        if result.shadow_action is None
                        else result.shadow_action.value,
                        "tier": result.shadow_tier,
                        "reason": result.shadow_reason,
                    }
            apply_content_policy(
                rec,
                log_content=settings.audit.log_content,
                prompt=prompt,
                response=None if result is None else result.text,
                redactors=st.redactors,
            )
            st.sink.write(rec)
        except Exception as e:  # observability must never fail a request
            log.error("finalize failed: %s", type(e).__name__)

    return app


def _identity(info: Any) -> dict[str, str]:
    return {
        "model_id": info.model_id,
        "model_hash": info.model_hash,
        "quant": info.quant,
        "kv_quant": info.kv_quant,
        "build": info.build,
        "template_hash": info.template_hash,
    }


def _tier_record(st: AppState, tr: Any) -> dict[str, Any]:
    sig = tr.signals
    values = {
        n: v
        for n in SCORE_NAMES
        if (v := sig.score(n)) is not None and v == v and abs(v) != float("inf")
    }
    return {
        "tier": tr.tier,
        "action": tr.decision.action.value,
        "reason": tr.decision.reason.value,
        "signal": st.artifact.tiers[tr.tier - 1].signal,
        "score": tr.decision.score if tr.decision.score == tr.decision.score else None,
        "threshold": tr.decision.threshold,
        "signals": values,
        "defects": [d.value for d in tr.defects],
        "skipped_generation": tr.skipped_generation,
        "backend_error": tr.error is not None,
        "finish": tr.finish,
    }

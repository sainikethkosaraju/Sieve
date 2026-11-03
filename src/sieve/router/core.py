"""Cascade router: small -> gate -> (large -> gate) -> answer / abstain.

The router contains no HTTP knowledge and makes no decision itself: every answer/
escalate/abstain outcome is produced by the pure :class:`sieve.gate.Gate`. The router
only (1) obtains completions, (2) turns them into ``Signals`` (backend failures become
signals carrying a defect, so they fail closed through the same gate path), (3) applies
the escalation-rate circuit breaker, and (4) accounts for cost, including wasted small
attempts and self-consistency samples.

Streaming: buffered gating only. The decision exists after the (full) generation, so the
server starts its stream after this coroutine returns. Early-abort (gate on a running
prefix) is NOT implemented: it requires prefix-calibrated thresholds that do not exist
yet (see docs/server.md).
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from sieve.backends.base import BackendHealth
from sieve.gate import Gate
from sieve.prompting import find_answer_start
from sieve.router.cost import Attempt, AttemptKind, CostLedger, CostSummary
from sieve.router.guard import EscalationGuard
from sieve.signals import compute_signals
from sieve.types import (
    SAMPLE_ONLY_SCORES,
    Action,
    Backend,
    Completion,
    Decision,
    DefectCode,
    SamplingParams,
    Signals,
)

log = logging.getLogger("sieve.router")


class RouterReason(enum.StrEnum):
    """Router-level reasons that are not gate ``ReasonCode`` values."""

    ESCALATION_SHED = "escalation_shed"
    LARGE_UNAVAILABLE = "large_unavailable"


class UpstreamUnavailable(RuntimeError):
    """Shadow mode could not obtain the baseline answer."""


@dataclass
class Tier:
    """A backend plus the server-owned health view and a concurrency budget."""

    backend: Backend
    health: BackendHealth = field(default_factory=BackendHealth)
    slots: int = 4
    n_probs: int = 20
    _sem: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.slots)

    @classmethod
    def healthy(cls, backend: Backend, **kw: int) -> Tier:
        """A tier whose self-test is assumed to have passed (tests / trusted setups)."""
        return cls(backend, BackendHealth(reachable=True, logprobs_ok=True), **kw)


@dataclass(frozen=True)
class TierTrace:
    tier: int
    decision: Decision
    skipped_generation: bool
    error: str | None
    finish: str | None
    defects: tuple[DefectCode, ...]
    signals: Signals


@dataclass(frozen=True)
class RouterResult:
    action: Action  # ANSWER or ABSTAIN, as served to the client
    text: str | None
    served_tier: int | None
    reason: str  # ReasonCode.value or RouterReason.value
    traces: tuple[TierTrace, ...]
    cost: CostSummary
    escalated: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish: str | None = None
    mode: str = "enforce"
    #: shadow mode only: what enforce mode would have done
    shadow_action: Action | None = None
    shadow_tier: int | None = None
    shadow_reason: str | None = None

    @property
    def final_score(self) -> float | None:
        return self.traces[-1].decision.score if self.traces else None


@dataclass
class _Evaluated:
    decision: Decision
    signals: Signals
    completion: Completion | None
    skipped: bool
    error: str | None


def _error_completion(sampling: SamplingParams) -> Completion:
    return Completion(
        text="", tokens=None, finish="error", n_gen=0, sampling=sampling, answer_start=0
    )


class Router:
    def __init__(
        self,
        gate: Gate,
        small: Tier,
        large: Tier | None,
        *,
        guard: EscalationGuard | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.gate = gate
        self.small = small
        self.large = large
        self.guard = guard or EscalationGuard(1.0)
        self._clock = clock

    # -- one backend call, accounted ------------------------------------------------

    async def _call(
        self,
        n: int,
        tier: Tier,
        kind: AttemptKind,
        prompt: str,
        params: SamplingParams,
        n_probs: int,
        ledger: CostLedger,
    ) -> tuple[Completion, str | None]:
        async with tier._sem:
            t0 = self._clock()
            try:
                comp = await tier.backend.generate(prompt, params, n_probs=n_probs)
            except asyncio.CancelledError:
                ledger.add(Attempt(n, kind, self._clock() - t0, 0, 0, "cancelled"))
                raise
            except Exception as e:  # fail closed: any backend failure is a defect
                ledger.add(Attempt(n, kind, self._clock() - t0, 0, 0, "error"))
                log.warning("tier %d %s call failed: %s", n, kind, type(e).__name__)
                return _error_completion(params), type(e).__name__
            comp = replace(comp, answer_start=find_answer_start(comp))
            ledger.add(
                Attempt(
                    n,
                    kind,
                    self._clock() - t0,
                    max(comp.n_prompt, 0),
                    max(comp.n_gen, 0),
                    "ok",
                )
            )
            return comp, None

    # -- evaluate one tier -------------------------------------------------------------

    async def _evaluate(
        self, n: int, prompt: str, max_tokens: int | None, ledger: CostLedger
    ) -> _Evaluated:
        tier = self.small if n == 1 else self.large
        assert tier is not None
        tc = self.gate.artifact.tiers[n - 1]
        info = tier.backend.info
        sample_only = tc.signal in SAMPLE_ONLY_SCORES
        sampling = tc.sampling
        if max_tokens is not None:
            sampling = replace(sampling, max_tokens=max_tokens)

        # Preflight: do not spend GPU time on a tier that cannot yield a valid decision.
        pre: tuple[DefectCode, ...] | None = None
        if info.key() == tc.identity.key():
            if not tier.health.reachable:
                pre = (DefectCode.BACKEND_ERROR,)
            elif not tier.health.logprobs_ok and not sample_only:
                pre = (DefectCode.MISSING_LOGPROBS,)
        else:
            pre = ()  # the gate reports IDENTITY_MISMATCH before looking at defects
        if pre is not None:
            sig = Signals(info, tc.sampling, 0, defects=pre)
            return _Evaluated(self.gate.decide(sig, n), sig, None, True, None)

        samples: list[Completion] = []
        err: str | None
        cs = tc.consistency_sampling
        if sample_only and cs is not None and tc.n_samples > 0:
            comp, err, samples = await self._with_samples(
                n, tier, prompt, sampling, cs, tc.n_samples, ledger
            )
        else:
            comp, err = await self._call(n, tier, "primary", prompt, sampling, tier.n_probs, ledger)

        sig = compute_signals(comp, info, samples, consistency_sampling=cs if samples else None)
        if sample_only and comp.finish != "stop" and DefectCode.SAMPLE_DEFECT not in sig.defects:
            # consistency scores ignore logprob defects; a truncated/errored served
            # answer must still fail closed
            sig = replace(sig, defects=(*sig.defects, DefectCode.SAMPLE_DEFECT))
        return _Evaluated(self.gate.decide(sig, n), sig, comp, False, err)

    async def _with_samples(
        self,
        n: int,
        tier: Tier,
        prompt: str,
        sampling: SamplingParams,
        cs: SamplingParams,
        k: int,
        ledger: CostLedger,
    ) -> tuple[Completion, str | None, list[Completion]]:
        base = cs.seed or 0
        async with asyncio.TaskGroup() as tg:
            main = tg.create_task(
                self._call(n, tier, "primary", prompt, sampling, tier.n_probs, ledger)
            )
            subs = [
                tg.create_task(
                    self._call(
                        n,
                        tier,
                        "sample",
                        prompt,
                        replace(cs, seed=base + i, max_tokens=sampling.max_tokens),
                        0,
                        ledger,
                    )
                )
                for i in range(k)
            ]
        comp, err = main.result()
        return comp, err, [s.result()[0] for s in subs]

    # -- public --------------------------------------------------------------------------

    @staticmethod
    def _trace(n: int, ev: _Evaluated) -> TierTrace:
        c = ev.completion
        return TierTrace(
            n,
            ev.decision,
            ev.skipped,
            ev.error,
            None if c is None else c.finish,
            ev.signals.defects,
            ev.signals,
        )

    def _served(
        self,
        ev: _Evaluated,
        n: int,
        traces: list[TierTrace],
        ledger: CostLedger,
        *,
        escalated: bool,
        reason: str,
        mode: str = "enforce",
    ) -> RouterResult:
        c = ev.completion
        assert c is not None
        return RouterResult(
            Action.ANSWER,
            c.text,
            n,
            reason,
            tuple(traces),
            ledger.summary(n),
            escalated,
            c.n_prompt,
            c.n_gen,
            c.finish,
            mode,
        )

    def _abstain(
        self, traces: list[TierTrace], ledger: CostLedger, reason: str, *, escalated: bool
    ) -> RouterResult:
        return RouterResult(
            Action.ABSTAIN, None, None, reason, tuple(traces), ledger.summary(None), escalated
        )

    async def run(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        ledger: CostLedger | None = None,
        shadow: bool = False,
    ) -> RouterResult:
        ledger = ledger if ledger is not None else CostLedger()
        traces: list[TierTrace] = []
        ev1 = await self._evaluate(1, prompt, max_tokens, ledger)
        traces.append(self._trace(1, ev1))
        d1 = ev1.decision
        if shadow:
            return await self._run_shadow(prompt, max_tokens, ledger, traces, ev1)

        if d1.action is Action.ANSWER and ev1.completion is not None:
            self.guard.note_no_escalation()
            return self._served(ev1, 1, traces, ledger, escalated=False, reason=d1.reason.value)
        if d1.action is Action.ABSTAIN:
            self.guard.note_no_escalation()
            return self._abstain(traces, ledger, d1.reason.value, escalated=False)
        if self.large is None:
            self.guard.note_no_escalation()
            return self._abstain(
                traces, ledger, RouterReason.LARGE_UNAVAILABLE.value, escalated=False
            )
        if not self.guard.request_escalation():
            return self._abstain(
                traces, ledger, RouterReason.ESCALATION_SHED.value, escalated=False
            )
        ev2 = await self._evaluate(2, prompt, max_tokens, ledger)
        traces.append(self._trace(2, ev2))
        d2 = ev2.decision
        if d2.action is Action.ANSWER and ev2.completion is not None:
            return self._served(ev2, 2, traces, ledger, escalated=True, reason=d2.reason.value)
        return self._abstain(traces, ledger, d2.reason.value, escalated=True)

    async def _run_shadow(
        self,
        prompt: str,
        max_tokens: int | None,
        ledger: CostLedger,
        traces: list[TierTrace],
        ev1: _Evaluated,
    ) -> RouterResult:
        """Log what the gate would do; always serve the large (baseline) answer."""
        if self.large is None:
            raise UpstreamUnavailable("shadow mode needs a large tier")
        ev2 = await self._evaluate(2, prompt, max_tokens, ledger)
        traces.append(self._trace(2, ev2))
        d1, d2 = ev1.decision, ev2.decision
        if d1.action is Action.ANSWER:
            sa, st, sr = Action.ANSWER, 1, d1.reason.value
        elif d2.action is Action.ANSWER:
            sa, st, sr = Action.ANSWER, 2, d2.reason.value
        else:
            sa, st, sr = Action.ABSTAIN, None, d2.reason.value
        c2 = ev2.completion
        if c2 is None or c2.finish == "error":
            raise UpstreamUnavailable("baseline tier returned no answer")
        res = self._served(
            ev2,
            2,
            traces,
            ledger,
            escalated=d1.action is not Action.ANSWER,
            reason="shadow",
            mode="shadow",
        )
        return replace(res, shadow_action=sa, shadow_tier=st, shadow_reason=sr)

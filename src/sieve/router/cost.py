"""Per-request cost accounting.

Every backend call is an :class:`Attempt`, including the small-tier attempt whose answer
is later thrown away (escalated or abstained) and any self-consistency samples. Totals
are computed from attempts, so wasted work can never be dropped from the bill. The
ledger is created by the caller and mutated by the router, so a cancelled or timed-out
request still reports what it burned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AttemptKind = Literal["primary", "sample"]


@dataclass(frozen=True)
class Attempt:
    tier: int
    kind: AttemptKind
    seconds: float
    n_prompt: int
    n_gen: int
    outcome: Literal["ok", "error", "cancelled"]


@dataclass(frozen=True)
class CostSummary:
    total_seconds: float
    total_tokens: int
    small_seconds: float
    large_seconds: float
    #: seconds / tokens not contributing to the served answer (all of it on abstain)
    wasted_seconds: float
    wasted_tokens: int
    n_attempts: int


@dataclass
class CostLedger:
    attempts: list[Attempt] = field(default_factory=list)

    def add(self, a: Attempt) -> None:
        self.attempts.append(a)

    def summary(self, served_tier: int | None) -> CostSummary:
        tot_s = sum(a.seconds for a in self.attempts)
        tot_t = sum(a.n_prompt + a.n_gen for a in self.attempts)

        def useful(a: Attempt) -> bool:
            return a.kind == "primary" and a.tier == served_tier and a.outcome == "ok"

        return CostSummary(
            total_seconds=tot_s,
            total_tokens=tot_t,
            small_seconds=sum(a.seconds for a in self.attempts if a.tier == 1),
            large_seconds=sum(a.seconds for a in self.attempts if a.tier == 2),
            wasted_seconds=sum(a.seconds for a in self.attempts if not useful(a)),
            wasted_tokens=sum(a.n_prompt + a.n_gen for a in self.attempts if not useful(a)),
            n_attempts=len(self.attempts),
        )

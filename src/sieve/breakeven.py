"""Break-even calculator: when does a confidence-gated cascade cost less than the large model alone?

A cascade runs the small tier on every query and the large tier only on the queries the small tier
does not resolve. With measured per-query costs ``cost_s`` and ``cost_l`` and a small-tier resolved
share ``c`` (the fraction of queries answered by the small tier at the certified error level)::

    cascade cost per query = cost_s + (1 - c) * cost_l
    relative to large only  = cost_s / cost_l + (1 - c)

so the cascade is cheaper exactly when ``c > cost_s / cost_l`` (the break-even coverage), and the
best possible saving is ``1 - cost_s / cost_l`` (when ``c`` is 1). Queries that end in abstention
still cost what was run, so this is a lower bound on what a deployed cascade pays per *answered*
query.

    python -m sieve.breakeven --cost-small 0.062 --cost-large 0.192 --resolved 0.04
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class BreakEven:
    cost_small: float
    cost_large: float
    resolved: float

    @property
    def cost_ratio(self) -> float:
        return self.cost_large / self.cost_small

    @property
    def break_even_resolved(self) -> float:
        """Smallest small-tier resolved share at which the cascade stops costing more."""
        return self.cost_small / self.cost_large

    @property
    def relative_cost(self) -> float:
        """Cascade cost per query as a multiple of the large model alone."""
        return self.cost_small / self.cost_large + (1.0 - self.resolved)

    @property
    def max_saving(self) -> float:
        """Best case (every query resolved by the small tier), as a fraction of large-only cost."""
        return 1.0 - self.cost_small / self.cost_large

    @property
    def pays(self) -> bool:
        return self.resolved > self.break_even_resolved


def analyse(cost_small: float, cost_large: float, resolved: float) -> BreakEven:
    if cost_small <= 0 or cost_large <= 0:
        raise ValueError("costs must be positive")
    if not 0.0 <= resolved <= 1.0:
        raise ValueError("resolved must be in [0, 1]")
    return BreakEven(cost_small, cost_large, resolved)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--cost-small", type=float, required=True, help="small-tier cost per query (any unit)"
    )
    ap.add_argument(
        "--cost-large", type=float, required=True, help="large-tier cost per query (same unit)"
    )
    ap.add_argument(
        "--resolved", type=float, required=True, help="share resolved by the small tier, 0 to 1"
    )
    a = ap.parse_args(argv)
    b = analyse(a.cost_small, a.cost_large, a.resolved)
    print(f"cost ratio large/small      : {b.cost_ratio:.2f}")
    print(f"break-even resolved share   : {100 * b.break_even_resolved:.1f}%")
    print(f"your resolved share         : {100 * b.resolved:.1f}%")
    print(f"cascade cost vs large alone : {100 * b.relative_cost:.0f}%")
    print(f"best possible saving        : {100 * b.max_saving:.0f}%")
    print(
        "verdict                     :",
        "the cascade pays" if b.pays else "the cascade costs more than the large model alone",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shadow mode helpers: summarise what the gate WOULD have done.

In shadow mode the server always serves the large (baseline) answer and records the
gate's hypothetical decision in the audit trail (``shadow`` field) and in the
``sieve_shadow_would_decide_total`` metric. This module aggregates such records.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


def summarize_shadow(records: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    n = 0
    c: Counter[str] = Counter()
    for r in records:
        sh = r.get("shadow")
        if not isinstance(sh, Mapping):
            continue
        n += 1
        act = sh.get("action")
        tier = sh.get("tier")
        c[f"{act}_tier{tier}" if act == "answer" else str(act)] += 1
    out: dict[str, float | int] = {"n": n}
    for k, v in c.items():
        out[k] = v
        out[f"{k}_rate"] = v / n if n else 0.0
    return out

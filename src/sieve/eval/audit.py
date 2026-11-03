"""200-item human-audit sheet generator and analyser.

The sheet is blind: rows are shuffled and carry no automatic label. A separate key file maps
``row_id`` to the item, tier, automatic label and the size of the stratum it was drawn from.
The analyser estimates, per tier and pooled,

* ``e1`` false-wrong: P(human says correct | auto says wrong),
* ``e2`` false-right: P(human says wrong | auto says correct),

with exact Clopper-Pearson intervals, a population-weighted overall grader error with a
Jeffreys-posterior interval, and a *sensitivity* bound on the true wrong-answer rate. The
bound is a diagnostic, never presented as the certified guarantee.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

from sieve.eval._util import atomic_write_text, stable_unit

SHEET_FIELDS = (
    "row_id",
    "dataset",
    "question",
    "gold_answers",
    "model_answer",
    "human_label",
    "notes",
)
HUMAN_LABELS = ("correct", "wrong")


@dataclass(frozen=True)
class AuditCandidate:
    item_id: str
    dataset: str
    tier: str  # "S" | "L"
    question: str
    gold_answers: tuple[str, ...]
    model_answer: str
    auto_correct: bool


def cp_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson interval; ``(0, 1)`` when ``n == 0``."""
    if n == 0:
        return 0.0, 1.0
    a = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


def _allocate(sizes: dict[tuple[str, bool], int], total: int) -> dict[tuple[str, bool], int]:
    """Equal allocation over non-empty strata; leftover goes to strata with spare items."""
    keys = sorted(k for k, v in sizes.items() if v > 0)
    take = dict.fromkeys(keys, 0)
    remaining = total
    while remaining > 0 and keys:
        open_keys = [k for k in keys if take[k] < sizes[k]]
        if not open_keys:
            break
        share = max(remaining // len(open_keys), 1)
        for k in open_keys:
            add = min(share, sizes[k] - take[k], remaining)
            take[k] += add
            remaining -= add
            if remaining == 0:
                break
    return take


def make_audit_sheet(
    candidates: Sequence[AuditCandidate],
    out_dir: Path,
    *,
    n: int = 200,
    seed: int = 0,
    grader_id: str,
) -> tuple[Path, Path]:
    """Write ``audit_sheet.csv`` (blind) and ``audit_key.json``; return both paths."""
    strata: dict[tuple[str, bool], list[AuditCandidate]] = defaultdict(list)
    for c in candidates:
        strata[(c.tier, c.auto_correct)].append(c)
    sizes = {k: len(v) for k, v in strata.items()}
    take = _allocate(sizes, n)
    chosen: list[AuditCandidate] = []
    for key, k_take in take.items():
        pool = sorted(strata[key], key=lambda c: stable_unit(seed, "pick", c.tier, c.item_id))
        chosen.extend(pool[:k_take])
    chosen.sort(key=lambda c: stable_unit(seed, "row", c.tier, c.item_id))  # blind shuffle
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = out_dir / "audit_sheet.csv"
    key_rows: dict[str, Any] = {}
    with sheet.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET_FIELDS)
        w.writeheader()
        for i, c in enumerate(chosen):
            rid = f"r{i:04d}"
            w.writerow(
                {
                    "row_id": rid,
                    "dataset": c.dataset,
                    "question": c.question,
                    "gold_answers": " | ".join(c.gold_answers[:6]),
                    "model_answer": c.model_answer,
                    "human_label": "",
                    "notes": "",
                }
            )
            key_rows[rid] = {
                "item_id": c.item_id,
                "tier": c.tier,
                "auto_correct": c.auto_correct,
            }
    key_doc = {
        "grader_id": grader_id,
        "seed": seed,
        "n": len(chosen),
        "population": {f"{t}|{int(a)}": v for (t, a), v in sorted(sizes.items())},
        "sampled": {f"{t}|{int(a)}": v for (t, a), v in sorted(take.items())},
        "rows": key_rows,
    }
    key_path = out_dir / "audit_key.json"
    atomic_write_text(key_path, json.dumps(key_doc, sort_keys=True, indent=1))
    return sheet, key_path


@dataclass(frozen=True)
class RateEstimate:
    k: int
    n: int
    rate: float
    lo: float
    hi: float


def _rate(k: int, n: int) -> RateEstimate:
    lo, hi = cp_interval(k, n)
    return RateEstimate(k, n, k / n if n else float("nan"), lo, hi)


@dataclass(frozen=True)
class AuditResult:
    grader_id: str
    n_labelled: int
    n_unlabelled: int
    by_tier: dict[str, dict[str, RateEstimate]]  # tier -> {"e1": ..., "e2": ...}
    pooled: dict[str, RateEstimate]
    overall_error: dict[str, tuple[float, float, float]]  # tier|"all" -> (est, lo, hi)
    tier_gap: float | None  # |overall error S - L|

    def passes_kill_test(self) -> bool | None:
        """Kill test 6: CP95 upper bound of the overall error <= 3% and tiers within 2 pp.

        Decided on the interval (F-17), not the point estimate. ``None`` if undecidable.
        """
        if self.n_unlabelled > 0 or "all" not in self.overall_error:
            return None
        ok_total = self.overall_error["all"][2] <= 0.03
        ok_gap = self.tier_gap is None or self.tier_gap <= 0.02
        return ok_total and ok_gap


def analyse_audit(sheet_path: Path, key_path: Path, *, seed: int = 0) -> AuditResult:
    key = json.loads(key_path.read_text(encoding="utf-8"))
    rows = {r["row_id"]: r for r in csv.DictReader(sheet_path.open(encoding="utf-8"))}
    counts: dict[tuple[str, bool], list[int]] = defaultdict(lambda: [0, 0])  # [disagree, n]
    unlabelled = 0
    for rid, meta in key["rows"].items():
        lab = rows[rid]["human_label"].strip().lower()
        if lab == "":
            unlabelled += 1
            continue
        if lab not in HUMAN_LABELS:
            raise ValueError(f"row {rid}: human_label must be one of {HUMAN_LABELS}, got {lab!r}")
        auto = bool(meta["auto_correct"])
        human = lab == "correct"
        c = counts[(meta["tier"], auto)]
        c[1] += 1
        c[0] += int(human != auto)
    tiers = sorted({t for t, _ in counts})
    by_tier: dict[str, dict[str, RateEstimate]] = {}
    for t in tiers:
        k1, n1 = counts[(t, False)]
        k2, n2 = counts[(t, True)]
        by_tier[t] = {"e1": _rate(k1, n1), "e2": _rate(k2, n2)}
    pooled = {
        "e1": _rate(
            sum(counts[(t, False)][0] for t in tiers), sum(counts[(t, False)][1] for t in tiers)
        ),
        "e2": _rate(
            sum(counts[(t, True)][0] for t in tiers), sum(counts[(t, True)][1] for t in tiers)
        ),
    }
    pop = {tuple(k.split("|")): v for k, v in key["population"].items()}
    rng = np.random.default_rng(seed)

    def weighted(ts: Sequence[str]) -> tuple[float, float, float] | None:
        total = sum(pop.get((t, str(a)), 0) for t in ts for a in (0, 1))
        if total == 0:
            return None
        draws = np.zeros(4000)
        point = 0.0
        for t in ts:
            for a in (False, True):
                w = pop.get((t, str(int(a))), 0) / total
                k, n = counts[(t, a)]
                if w == 0.0:
                    continue
                if n == 0:
                    return None
                point += w * k / n
                draws += w * rng.beta(k + 0.5, n - k + 0.5, size=draws.size)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        return float(point), float(lo), float(hi)

    overall: dict[str, tuple[float, float, float]] = {}
    for t in tiers:
        w = weighted([t])
        if w is not None:
            overall[t] = w
    w_all = weighted(tiers)
    if w_all is not None:
        overall["all"] = w_all
    gap = abs(overall["S"][0] - overall["L"][0]) if "S" in overall and "L" in overall else None
    return AuditResult(
        grader_id=str(key["grader_id"]),
        n_labelled=sum(v[1] for v in counts.values()),
        n_unlabelled=unlabelled,
        by_tier=by_tier,
        pooled=pooled,
        overall_error=overall,
        tier_gap=gap,
    )


def sensitivity_adjusted_risk(
    risk_hat: float, e1: RateEstimate, e2: RateEstimate
) -> tuple[float, float]:
    """(point, worst-case) true wrong-answer rate given grader errors.

    ``true = r (1 - e1) + (1 - r) e2``; the worst case uses the lower CI of e1 and the upper
    CI of e2. A diagnostic only: it assumes the audit error rates carry over to the answered
    subset, which the audit does not establish.
    """
    r = risk_hat
    point = r * (1.0 - (0.0 if np.isnan(e1.rate) else e1.rate)) + (1.0 - r) * (
        0.0 if np.isnan(e2.rate) else e2.rate
    )
    worst = r * (1.0 - e1.lo) + (1.0 - r) * e2.hi
    return float(point), float(min(1.0, worst))

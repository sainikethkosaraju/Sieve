"""Apply certified (or reference) policies to a test slice and summarise them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from sieve.calibrate import NEVER, CalTable, evaluate_policy
from sieve.eval.certify import MethodCert
from sieve.eval.data import PairSlice
from sieve.eval.methods import FittedMethod, policy_item_costs

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]


@dataclass
class PolicyEval:
    """Item-level outcome of one policy on one slice (kept for paired bootstraps)."""

    label: str
    ids: tuple[str, ...]
    strata: list[str]
    answered: BoolArr
    wrong: BoolArr
    by_s: BoolArr
    cost: FloatArr
    alpha: float
    certified: bool = True
    note: str = ""
    t1: float | None = None
    t2: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    #: composed item-level "graded correct" flag; when set it (not ``~wrong``) defines accuracy,
    #: so a SimpleQA ``not_attempted`` reply (neither wrong nor correct) is never counted correct
    correct: BoolArr | None = None

    @property
    def n(self) -> int:
        return int(self.answered.size)

    @property
    def correct_answered(self) -> BoolArr:
        ok = ~self.wrong if self.correct is None else self.correct
        return np.asarray(self.answered & ok, dtype=np.bool_)

    def summary(self) -> dict[str, Any]:
        n_ans = int(self.answered.sum())
        k = int((self.answered & self.wrong).sum())
        n_correct = int(self.correct_answered.sum())
        total = float(self.cost.sum())
        risk = k / n_ans if n_ans else None
        strata: dict[str, list[int]] = {}
        for s, a, w in zip(self.strata, self.answered, self.wrong, strict=True):
            if a:
                c = strata.setdefault(s, [0, 0])
                c[0] += 1
                c[1] += int(w)
        return {
            "label": self.label,
            "certified": self.certified,
            "n": self.n,
            "n_answered": n_ans,
            "n_wrong_answered": k,
            "coverage": n_ans / self.n if self.n else None,
            "risk": risk,
            "accuracy_on_answered": n_correct / n_ans if n_ans else None,
            "violates_alpha": None if risk is None else bool(risk > self.alpha),
            "escalation_rate": float(1.0 - self.by_s.mean()) if self.n else None,
            "cost_total": total,
            "cost_per_item": total / self.n if self.n else None,
            "cost_per_correct": total / n_correct if n_correct else None,
            "per_stratum": {
                s: {"n_answered": c[0], "n_wrong": c[1]} for s, c in sorted(strata.items())
            },
            "note": self.note,
        }


def eval_policy(
    fm: FittedMethod,
    sl: PairSlice,
    t1: float | None,
    t2: float | None,
    *,
    alpha: float,
    label: str,
    certified: bool = True,
    note: str = "",
) -> PolicyEval:
    sub = sl.subset(fm.eligible(sl))
    if sub.n == 0:
        z = np.zeros(0, dtype=np.bool_)
        return PolicyEval(label, (), [], z, z, z, np.zeros(0), alpha, certified, note, t1, t2)
    s, l_ = fm.scores(sub)
    table = CalTable(s, l_, sub.s.wrong, sub.l.wrong)
    out = evaluate_policy(table, NEVER if t1 is None else t1, NEVER if t2 is None else t2)
    return PolicyEval(
        label,
        sub.ids,
        sub.strata,
        out.answered,
        out.wrong,
        out.by_s,
        policy_item_costs(fm, sub, out.by_s),
        alpha,
        certified,
        note,
        t1,
        t2,
        correct=np.asarray(np.where(out.by_s, sub.s.correct, sub.l.correct), dtype=np.bool_),
    )


def eval_cert(
    fm: FittedMethod, sl: PairSlice, cert: MethodCert, *, alpha: float
) -> PolicyEval | None:
    """Evaluate a certification on a slice; ``None`` when nothing was certified."""
    if not cert.feasible:
        return None
    return eval_policy(fm, sl, cert.t1, cert.t2, alpha=alpha, label=cert.method)


def oracle_router(sl: PairSlice, *, alpha: float) -> PolicyEval:
    """S if S is right, else L (no abstention, perfect routing, no wasted S attempt)."""
    sub = sl.subset(sl.both_graded())
    by_s = np.asarray(sub.s.correct, dtype=np.bool_)
    wrong = ~by_s & sub.l.wrong
    cost = np.where(by_s, sub.cost_s(), sub.cost_l())
    return PolicyEval(
        "oracle_router",
        sub.ids,
        sub.strata,
        np.ones(sub.n, dtype=bool),
        wrong,
        by_s,
        cost,
        alpha,
        certified=False,
        note="uses labels; descriptive upper bound, not deployable",
    )


def oracle_abstainer(sl: PairSlice, *, alpha: float) -> PolicyEval:
    """L answers iff L is right (perfect abstention); L runs on everything."""
    sub = sl.subset(sl.both_graded())
    answered = np.asarray(sub.l.correct, dtype=np.bool_)
    return PolicyEval(
        "oracle_abstainer",
        sub.ids,
        sub.strata,
        answered,
        np.zeros(sub.n, dtype=bool),
        np.zeros(sub.n, dtype=bool),
        sub.cost_l(),
        alpha,
        certified=False,
        note="uses labels; descriptive upper bound, not deployable",
    )

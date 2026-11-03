"""Selective-prediction and calibration metrics.

Conventions: ``conf`` is a confidence score where larger means more confident;
``wrong`` is a boolean array that is ``True`` when the model's answer is wrong
(counterfactually defined for items that were abstained on).

Tied confidences are handled by *expected* ordering: within a group of equal scores
the errors are spread uniformly, which equals the average over all random tie-breaks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]


def _as_1d(x: npt.ArrayLike, name: str, dtype: type) -> npt.NDArray[np.generic]:
    a: npt.NDArray[np.generic] = np.asarray(x, dtype=dtype)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    return a


def _conf_wrong(conf: npt.ArrayLike, wrong: npt.ArrayLike) -> tuple[FloatArr, BoolArr]:
    c: FloatArr = np.asarray(_as_1d(conf, "conf", np.float64), dtype=np.float64)
    w: BoolArr = np.asarray(_as_1d(wrong, "wrong", np.bool_), dtype=np.bool_)
    if c.shape != w.shape:
        raise ValueError("conf and wrong must have the same length")
    if not np.all(np.isfinite(c)):
        raise ValueError("conf must be finite")
    return c, w


def expected_cum_errors(conf: npt.ArrayLike, wrong: npt.ArrayLike) -> FloatArr:
    """Expected number of errors among the top-k most confident items, k = 1..n."""
    c, w = _conf_wrong(conf, wrong)
    n = c.size
    uniq, inv = np.unique(-c, return_inverse=True)
    g = np.bincount(inv, minlength=uniq.size).astype(np.float64)
    e = np.bincount(inv, weights=w.astype(np.float64), minlength=uniq.size)
    start = np.concatenate(([0.0], np.cumsum(g)[:-1]))
    err_before = np.concatenate(([0.0], np.cumsum(e)[:-1]))
    # Rank positions 1..n in descending-confidence order map to groups in order.
    grp = np.repeat(np.arange(uniq.size), g.astype(np.int64))
    j = np.arange(1, n + 1, dtype=np.float64) - start[grp]
    return np.asarray(err_before[grp] + j * e[grp] / g[grp], dtype=np.float64)


def risk_coverage_curve(conf: npt.ArrayLike, wrong: npt.ArrayLike) -> tuple[FloatArr, FloatArr]:
    """Achievable operating points ``(coverage, selective_risk)``, one per distinct score."""
    c, w = _conf_wrong(conf, wrong)
    n = c.size
    ecum = expected_cum_errors(c, w)
    sorted_desc = np.sort(c)[::-1]
    last_of_group = np.concatenate((sorted_desc[1:] != sorted_desc[:-1], [True]))
    k = np.arange(1, n + 1, dtype=np.float64)[last_of_group]
    return k / n, ecum[last_of_group] / k


def aurc(conf: npt.ArrayLike, wrong: npt.ArrayLike) -> float:
    """Area under the risk-coverage curve: mean over k of selective risk at k answered."""
    c, w = _conf_wrong(conf, wrong)
    ecum = expected_cum_errors(c, w)
    k = np.arange(1, c.size + 1, dtype=np.float64)
    return float(np.mean(ecum / k))


def augrc(conf: npt.ArrayLike, wrong: npt.ArrayLike) -> float:
    """Area under the generalised risk-coverage curve (risk = P(wrong and answered))."""
    c, w = _conf_wrong(conf, wrong)
    ecum = expected_cum_errors(c, w)
    return float(np.mean(ecum / c.size))


def ece_equal_mass(prob: npt.ArrayLike, correct: npt.ArrayLike, n_bins: int = 15) -> float:
    """Expected calibration error with equal-mass (equal-count) bins.

    Ties are split across bins by a stable sort, so the value can depend on the order
    of tied inputs; this is the usual convention and is stated rather than hidden.
    """
    p: FloatArr = np.asarray(_as_1d(prob, "prob", np.float64), dtype=np.float64)
    y: FloatArr = np.asarray(_as_1d(correct, "correct", np.float64), dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError("prob and correct must have the same length")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("prob must lie in [0, 1]")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    order = np.argsort(p, kind="stable")
    ece = 0.0
    for idx in np.array_split(order, min(n_bins, p.size)):
        ece += idx.size / p.size * abs(float(np.mean(y[idx])) - float(np.mean(p[idx])))
    return float(ece)


def brier(prob: npt.ArrayLike, correct: npt.ArrayLike) -> float:
    p: FloatArr = np.asarray(_as_1d(prob, "prob", np.float64), dtype=np.float64)
    y: FloatArr = np.asarray(_as_1d(correct, "correct", np.float64), dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError("prob and correct must have the same length")
    return float(np.mean((p - y) ** 2))


def auroc(score: npt.ArrayLike, correct: npt.ArrayLike) -> float:
    """P(score of a correct item > score of a wrong item), ties count one half."""
    s: FloatArr = np.asarray(_as_1d(score, "score", np.float64), dtype=np.float64)
    y: BoolArr = np.asarray(_as_1d(correct, "correct", np.bool_), dtype=np.bool_)
    if s.shape != y.shape:
        raise ValueError("score and correct must have the same length")
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both correct and wrong items")
    ranks = rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass(frozen=True)
class AbstentionStats:
    """Abstention as a detector of wrong answers.

    ``precision`` = P(would-be wrong | abstained); ``recall`` = P(abstained | would-be wrong).
    ``None`` when the denominator is empty.
    """

    precision: float | None
    recall: float | None
    n_abstained: int
    n_wrong: int


def abstention_stats(answered: npt.ArrayLike, wrong: npt.ArrayLike) -> AbstentionStats:
    a: BoolArr = np.asarray(_as_1d(answered, "answered", np.bool_), dtype=np.bool_)
    w: BoolArr = np.asarray(_as_1d(wrong, "wrong", np.bool_), dtype=np.bool_)
    if a.shape != w.shape:
        raise ValueError("answered and wrong must have the same length")
    abst = ~a
    n_abs = int(abst.sum())
    n_wrong = int(w.sum())
    both = int((abst & w).sum())
    return AbstentionStats(
        precision=both / n_abs if n_abs else None,
        recall=both / n_wrong if n_wrong else None,
        n_abstained=n_abs,
        n_wrong=n_wrong,
    )


@dataclass(frozen=True)
class HallucinationRates:
    """Both definitions of 'hallucination rate'; always report which one is meant.

    ``unconditional`` = P(answered and wrong); ``selective`` = P(wrong | answered).
    """

    unconditional: float
    selective: float | None
    n: int
    n_answered: int
    n_wrong_answered: int


def hallucination_rates(answered: npt.ArrayLike, wrong: npt.ArrayLike) -> HallucinationRates:
    a: BoolArr = np.asarray(_as_1d(answered, "answered", np.bool_), dtype=np.bool_)
    w: BoolArr = np.asarray(_as_1d(wrong, "wrong", np.bool_), dtype=np.bool_)
    if a.shape != w.shape:
        raise ValueError("answered and wrong must have the same length")
    n_a = int(a.sum())
    k = int((a & w).sum())
    return HallucinationRates(
        unconditional=k / a.size,
        selective=k / n_a if n_a else None,
        n=a.size,
        n_answered=n_a,
        n_wrong_answered=k,
    )

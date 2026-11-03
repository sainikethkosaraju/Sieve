"""Paired percentile bootstrap, McNemar, TOST.

All comparisons are *paired by item*: one resample of item indices is applied to every
method being compared, so a difference is bootstrapped as a difference of statistics on
the same resample (never from overlapping marginal CIs).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import binom, rankdata
from scipy.stats import t as student_t

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]
IntArr = npt.NDArray[np.int64]
#: statistic over a batch of resamples: ``(b, n)`` indices -> ``(b,)`` values
BatchStat = Callable[[IntArr], FloatArr]


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    n_boot: int
    level: float = 0.95

    def excludes(self, value: float) -> bool:
        return value < self.lo or value > self.hi

    def to_dict(self) -> dict[str, float]:
        return {"point": self.point, "lo": self.lo, "hi": self.hi}


def resample_indices(n: int, n_boot: int, seed: int, chunk: int = 500) -> list[IntArr]:
    """Deterministic resample index batches (shared by all statistics of one comparison)."""
    rng = np.random.default_rng(seed)
    out: list[IntArr] = []
    left = n_boot
    while left > 0:
        b = min(chunk, left)
        out.append(rng.integers(0, n, size=(b, n), dtype=np.int64))
        left -= b
    return out


def bootstrap(
    stat: BatchStat,
    n: int,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
    point: float | None = None,
) -> Interval:
    """Percentile bootstrap CI of ``stat`` over ``n`` items; NaN replicates are dropped."""
    vals = np.concatenate([stat(idx) for idx in resample_indices(n, n_boot, seed)])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), n_boot, level)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(vals, [a, 1.0 - a])
    if point is None:
        point = float(stat(np.arange(n, dtype=np.int64)[None, :])[0])
    return Interval(float(point), float(lo), float(hi), int(vals.size), level)


def paired_difference(
    stat_a: BatchStat,
    stat_b: BatchStat,
    n: int,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> Interval:
    """CI of ``stat_a - stat_b`` on shared resamples."""
    return bootstrap(
        lambda idx: stat_a(idx) - stat_b(idx), n, n_boot=n_boot, seed=seed, level=level
    )


# --- batch statistics -------------------------------------------------------------------


def batch_ratio(num: BoolArr | FloatArr, den: BoolArr | FloatArr) -> BatchStat:
    """``sum(num[idx]) / sum(den[idx])`` (NaN where the denominator is zero)."""
    nm = np.asarray(num, dtype=np.float64)
    dn = np.asarray(den, dtype=np.float64)

    def f(idx: IntArr) -> FloatArr:
        d = dn[idx].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.asarray(np.where(d > 0, nm[idx].sum(axis=1) / np.where(d > 0, d, 1), np.nan))

    return f


def batch_mean(x: FloatArr | BoolArr) -> BatchStat:
    a = np.asarray(x, dtype=np.float64)
    return lambda idx: np.asarray(a[idx].mean(axis=1))


def batch_auroc(score: FloatArr, positive: BoolArr) -> BatchStat:
    """AUROC of ``score`` for ``positive`` on each resample (ties = 1/2); NaN if one class."""
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(positive, dtype=bool)

    def f(idx: IntArr) -> FloatArr:
        ys = y[idx]
        ranks = rankdata(s[idx], axis=1)
        n_pos = ys.sum(axis=1).astype(np.float64)
        n_neg = idx.shape[1] - n_pos
        rank_sum = np.where(ys, ranks, 0.0).sum(axis=1)
        denom = n_pos * n_neg
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (rank_sum - n_pos * (n_pos + 1) / 2) / denom
        return np.asarray(np.where(denom > 0, out, np.nan))

    return f


def batch_ece(prob: FloatArr, correct: BoolArr, n_bins: int = 15) -> BatchStat:
    """Equal-mass ECE per resample; same bin boundaries and stable ties as ``ece_equal_mass``."""
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)

    def f(idx: IntArr) -> FloatArr:
        b, n = idx.shape
        nb = min(n_bins, n)
        sizes = np.full(nb, n // nb)
        sizes[: n % nb] += 1
        edges = np.concatenate([[0], np.cumsum(sizes)])
        pi = p[idx]
        order = np.argsort(pi, axis=1, kind="stable")
        ps = np.take_along_axis(pi, order, axis=1)
        ys = np.take_along_axis(y[idx], order, axis=1)
        cp = np.concatenate([np.zeros((b, 1)), np.cumsum(ps, axis=1)], axis=1)
        cy = np.concatenate([np.zeros((b, 1)), np.cumsum(ys, axis=1)], axis=1)
        mp = (cp[:, edges[1:]] - cp[:, edges[:-1]]) / sizes
        my = (cy[:, edges[1:]] - cy[:, edges[:-1]]) / sizes
        return np.asarray((np.abs(mp - my) * sizes / n).sum(axis=1))

    return f


# --- McNemar ------------------------------------------------------------------------


@dataclass(frozen=True)
class McNemarResult:
    n10: int  # A right, B wrong
    n01: int  # A wrong, B right
    p_exact: float  # two-sided exact binomial


def mcnemar(a_correct: BoolArr, b_correct: BoolArr) -> McNemarResult:
    """Exact McNemar test on paired correctness (discordant pairs, two-sided)."""
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have equal length")
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return McNemarResult(n10, n01, 1.0)
    k = min(n10, n01)
    p = min(1.0, 2.0 * float(binom.cdf(k, n, 0.5)))
    return McNemarResult(n10, n01, p)


# --- TOST -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TostResult:
    mean_diff: float
    margin: float
    p_lower: float  # H0: diff <= -margin
    p_upper: float  # H0: diff >= +margin
    p_tost: float
    equivalent: bool  # p_tost < alpha


def tost_paired_t(x: FloatArr, y: FloatArr, margin: float, alpha: float = 0.05) -> TostResult:
    """Paired-t two one-sided tests for equivalence within +-margin."""
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    n = d.size
    if n < 2:
        raise ValueError("need at least two pairs")
    m = float(d.mean())
    se = float(d.std(ddof=1)) / math.sqrt(n)
    if se == 0.0:
        inside = abs(m) < margin
        return TostResult(
            m,
            margin,
            0.0 if inside else 1.0,
            0.0 if inside else 1.0,
            0.0 if inside else 1.0,
            inside,
        )
    df = n - 1
    t_lo = (m + margin) / se
    t_hi = (m - margin) / se
    p_lo = float(student_t.sf(t_lo, df))
    p_hi = float(student_t.cdf(t_hi, df))
    p = max(p_lo, p_hi)
    return TostResult(m, margin, p_lo, p_hi, p, p < alpha)


def tost_bootstrap(ci90: Interval, margin: float) -> bool:
    """Equivalence by CI inclusion: the (1 - 2*alpha) CI lies inside (-margin, +margin)."""
    return bool(ci90.lo > -margin and ci90.hi < margin)


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    from sieve.eval.audit import cp_interval

    return cp_interval(k, n, level)

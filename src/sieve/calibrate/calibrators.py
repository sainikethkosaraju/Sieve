"""Score-to-probability maps: temperature, Platt, isotonic.

These affect *ranking and reporting only*. The risk guarantee comes from the
threshold selection in ``ltt`` and does not depend on any calibrator being right.
Isotonic regression produces ties (flat blocks), so never threshold on its output.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]


def logit(p: npt.ArrayLike, eps: float = 1e-6) -> FloatArr:
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.asarray(np.log(q / (1.0 - q)), dtype=np.float64)


def _nll(p: FloatArr, y: FloatArr) -> float:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


class TemperatureScaler:
    """``p = sigmoid(z / T)`` for a logit-scale input ``z``; fits ``T > 0`` by NLL."""

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, scores: FloatArr, correct: BoolArr) -> TemperatureScaler:
        z = np.asarray(scores, dtype=np.float64)
        y = np.asarray(correct, dtype=np.float64)
        res = minimize_scalar(
            lambda log_t: _nll(expit(z / np.exp(log_t)), y), bounds=(-5.0, 5.0), method="bounded"
        )
        self.temperature = float(np.exp(res.x))
        return self

    def predict(self, scores: FloatArr) -> FloatArr:
        return np.asarray(expit(np.asarray(scores, dtype=np.float64) / self.temperature))


class PlattScaler:
    """``p = sigmoid(a * s + b)`` with Platt's smoothed targets (finite on separable data)."""

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0

    def fit(self, scores: FloatArr, correct: BoolArr) -> PlattScaler:
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(correct, dtype=bool)
        n_pos = float(y.sum())
        n_neg = float(y.size - y.sum())
        t = np.where(y, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
        mu, sd = float(s.mean()), float(s.std()) or 1.0
        z = (s - mu) / sd

        def loss(w: FloatArr) -> float:
            p = np.clip(expit(w[0] * z + w[1]), 1e-12, 1 - 1e-12)
            return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))

        w = minimize(loss, np.array([0.0, 0.0]), method="BFGS").x
        self.a = float(w[0] / sd)
        self.b = float(w[1] - w[0] * mu / sd)
        return self

    def predict(self, scores: FloatArr) -> FloatArr:
        return np.asarray(expit(self.a * np.asarray(scores, dtype=np.float64) + self.b))


class IsotonicCalibrator:
    """Pool-adjacent-violators fit of a non-decreasing ``P(correct | score)``."""

    def __init__(self) -> None:
        self._xp: FloatArr = np.zeros(0)
        self._fp: FloatArr = np.zeros(0)

    def fit(self, scores: FloatArr, correct: BoolArr) -> IsotonicCalibrator:
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(correct, dtype=np.float64)
        if s.size == 0:
            raise ValueError("empty fit set")
        xs, inv = np.unique(s, return_inverse=True)
        w = np.bincount(inv).astype(np.float64)
        m = np.bincount(inv, weights=y) / w
        # Blocks: (mean, weight, x_lo, x_hi)
        blocks: list[list[float]] = []
        for i in range(xs.size):
            blocks.append([m[i], w[i], xs[i], xs[i]])
            while len(blocks) > 1 and blocks[-2][0] >= blocks[-1][0]:
                b2 = blocks.pop()
                b1 = blocks.pop()
                tot = b1[1] + b2[1]
                blocks.append([(b1[0] * b1[1] + b2[0] * b2[1]) / tot, tot, b1[2], b2[3]])
        xp: list[float] = []
        fp: list[float] = []
        for mean, _, lo, hi in blocks:
            xp.extend([lo, hi])
            fp.extend([mean, mean])
        self._xp = np.asarray(xp, dtype=np.float64)
        self._fp = np.asarray(fp, dtype=np.float64)
        return self

    def predict(self, scores: FloatArr) -> FloatArr:
        if self._xp.size == 0:
            raise RuntimeError("not fitted")
        return np.asarray(np.interp(np.asarray(scores, dtype=np.float64), self._xp, self._fp))

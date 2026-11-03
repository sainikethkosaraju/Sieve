"""Cheap detectors for score-distribution shift (detection only, no guarantee).

A shift in P(wrong | score) that leaves the score histogram unchanged is invisible to
these; the shadow-mode graded sample is the only defence against that.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import ks_2samp


def score_shift_pvalue(
    reference: npt.NDArray[np.float64], window: npt.NDArray[np.float64]
) -> float:
    """Two-sample Kolmogorov-Smirnov p-value of ``window`` against calibration scores."""
    return float(ks_2samp(reference, window).pvalue)


def psi(
    reference: npt.NDArray[np.float64], window: npt.NDArray[np.float64], n_bins: int = 10
) -> float:
    """Population stability index over reference-quantile bins."""
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    p = np.histogram(reference, edges)[0] / reference.size
    q = np.histogram(window, edges)[0] / window.size
    p = np.clip(p, 1e-6, None)
    q = np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))

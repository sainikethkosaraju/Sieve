"""Drift alarm on the confidence-score histogram.

Reuses the Phase A detectors (:func:`sieve.calibrate.shift.score_shift_pvalue` and
:func:`sieve.calibrate.shift.psi`) on a sliding window of tier-1 scores against a
reference sample (calibration-set scores exported by the eval harness). The alarm needs
BOTH a small KS p-value and a PSI above threshold, which trades sensitivity for a low
false-alarm rate. Detection only: no automatic action, no guarantee. A shift in
P(wrong | score) that leaves the score histogram unchanged is invisible to it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sieve.calibrate.shift import psi, score_shift_pvalue


@dataclass(frozen=True)
class DriftState:
    n_window: int
    ks_pvalue: float | None
    psi: float | None
    alarm: bool


def load_reference(path: str, expected_sha256: str | None = None) -> npt.NDArray[np.float64]:
    """Load reference scores from a ``.npy`` file or a JSON list; optionally pin its hash."""
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("drift reference hash mismatch")
    if path.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
    else:
        arr = np.asarray(json.loads(raw), dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 50:
        raise ValueError("drift reference needs at least 50 finite scores")
    return arr


class DriftMonitor:
    def __init__(
        self,
        reference: npt.NDArray[np.float64],
        *,
        window: int = 200,
        min_window: int = 100,
        check_every: int = 20,
        ks_alpha: float = 1e-3,
        psi_threshold: float = 0.2,
    ) -> None:
        if reference.size < 50:
            raise ValueError("reference too small")
        self.reference = np.asarray(reference, dtype=np.float64)
        self.window = window
        self.min_window = min(min_window, window)
        self.check_every = check_every
        self.ks_alpha = ks_alpha
        self.psi_threshold = psi_threshold
        self._buf: deque[float] = deque(maxlen=window)
        self._since = 0
        self.skipped_non_finite = 0
        self.state = DriftState(0, None, None, False)

    def observe(self, score: float | None) -> DriftState:
        """Add one score (``None``/non-finite scores are counted and skipped)."""
        if score is None or not math.isfinite(score):
            self.skipped_non_finite += 1
            return self.state
        self._buf.append(float(score))
        self._since += 1
        n = len(self._buf)
        if n >= self.min_window and self._since >= self.check_every:
            self._since = 0
            win = np.fromiter(self._buf, dtype=np.float64, count=n)
            p = score_shift_pvalue(self.reference, win)
            s = psi(self.reference, win)
            self.state = DriftState(n, p, s, p < self.ks_alpha and s > self.psi_threshold)
        else:
            self.state = DriftState(n, self.state.ks_pvalue, self.state.psi, self.state.alarm)
        return self.state

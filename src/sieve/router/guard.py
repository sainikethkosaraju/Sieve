"""Escalation-rate circuit breaker (red-team: escalation amplification doubles cost).

Sliding window over the last ``window`` requests. An escalation is granted only while
``granted_in_window + 1 <= max_rate * window``. A shed request is NOT answered from the
small tier (that would bypass the gate); the router abstains with reason
``escalation_shed``. With ``max_rate >= 1`` the guard is disabled.
"""

from __future__ import annotations

from collections import deque


class EscalationGuard:
    def __init__(self, max_rate: float, window: int = 200) -> None:
        if not max_rate >= 0.0:
            raise ValueError("max_rate must be >= 0")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.max_rate = max_rate
        self.window = window
        self._granted: deque[bool] = deque(maxlen=window)
        self.shed_total = 0

    @property
    def enabled(self) -> bool:
        return self.max_rate < 1.0

    @property
    def rate(self) -> float:
        return sum(self._granted) / len(self._granted) if self._granted else 0.0

    def note_no_escalation(self) -> None:
        self._granted.append(False)

    def request_escalation(self) -> bool:
        """Ask to escalate one request. Records the outcome in the window."""
        allowed = (not self.enabled) or (sum(self._granted) + 1) <= self.max_rate * self.window
        self._granted.append(allowed)
        if not allowed:
            self.shed_total += 1
        return allowed

    @property
    def open(self) -> bool:
        """True while the next escalation would be shed."""
        return self.enabled and (sum(self._granted) + 1) > self.max_rate * self.window

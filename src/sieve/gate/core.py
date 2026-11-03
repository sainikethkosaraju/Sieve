"""The gate: a pure function ``(signals, artifact, tier) -> Decision``.

The same function serves the server, the eval harness and shadow mode, so there is
exactly one decision path. No IO, no clocks, no randomness.

Fail-closed order (first hit wins, none of them can produce ``ANSWER``):
1. backend identity != calibration identity     -> IDENTITY_MISMATCH
2. sampling / consistency configuration differs  -> SAMPLING_MISMATCH
3. any relevant logprob/sample defect            -> LOGPROB_DEFECT
4. the keyed score is missing or non-finite      -> SIGNAL_UNAVAILABLE
Tier 1 failing closed escalates; tier 2 failing closed abstains.
"""

from __future__ import annotations

import math

from sieve.calibrate.store import verify_artifact
from sieve.types import (
    CHOSEN_ONLY_SCORES,
    SAMPLE_ONLY_SCORES,
    Action,
    CalibrationArtifact,
    Decision,
    DefectCode,
    ReasonCode,
    Signals,
    TierCalibration,
)


def _relevant_defects(sig: Signals, signal: str) -> tuple[DefectCode, ...]:
    """Defects that invalidate ``signal``.

    Consistency scores need no logprobs, so logprob defects do not matter for them;
    chosen-token scores tolerate missing top-k lists.
    """
    if signal in SAMPLE_ONLY_SCORES:
        sample_codes = {
            DefectCode.TOO_FEW_SAMPLES,
            DefectCode.SAMPLE_DEFECT,
            DefectCode.BACKEND_ERROR,
        }
        return tuple(d for d in sig.defects if d in sample_codes)
    if signal in CHOSEN_ONLY_SCORES:
        return tuple(d for d in sig.defects if d is not DefectCode.TOP_LOGPROBS_MISSING)
    return sig.defects


def _fail(
    tier: int,
    reason: ReasonCode,
    art: CalibrationArtifact,
    tc: TierCalibration,
    defects: tuple[DefectCode, ...] = (),
    score: float | None = None,
) -> Decision:
    action = Action.ESCALATE if tier == 1 else Action.ABSTAIN
    return Decision(action, reason, tier, score, tc.threshold, art.artifact_id, defects)


def decide(signals: Signals, artifact: CalibrationArtifact, tier: int) -> Decision:
    """Decide for one tier. Assumes ``artifact`` was verified (see ``Gate``)."""
    if tier not in (1, 2):
        raise ValueError("tier must be 1 or 2")
    tc = artifact.tiers[tier - 1]

    if signals.backend.key() != tc.identity.key() or not signals.backend.is_complete():
        return _fail(tier, ReasonCode.IDENTITY_MISMATCH, artifact, tc)

    uses_samples = tc.signal in SAMPLE_ONLY_SCORES
    if uses_samples:
        sampling_ok = (
            tc.consistency_sampling is not None
            and signals.consistency_sampling is not None
            and signals.consistency_sampling.key() == tc.consistency_sampling.key()
            and signals.n_samples == tc.n_samples
        )
    else:
        sampling_ok = signals.sampling.key() == tc.sampling.key()
    if not sampling_ok:
        return _fail(tier, ReasonCode.SAMPLING_MISMATCH, artifact, tc)

    defects = _relevant_defects(signals, tc.signal)
    if defects:
        return _fail(tier, ReasonCode.LOGPROB_DEFECT, artifact, tc, defects)

    score = signals.score(tc.signal)
    if score is None or not math.isfinite(score):
        return _fail(tier, ReasonCode.SIGNAL_UNAVAILABLE, artifact, tc)

    if tc.threshold is None:
        action = Action.ESCALATE if tier == 1 else Action.ABSTAIN
        return Decision(
            action, ReasonCode.TIER_NOT_CERTIFIED, tier, score, None, artifact.artifact_id
        )
    if score >= tc.threshold:
        return Decision(
            Action.ANSWER, ReasonCode.CONFIDENT, tier, score, tc.threshold, artifact.artifact_id
        )
    action = Action.ESCALATE if tier == 1 else Action.ABSTAIN
    return Decision(
        action, ReasonCode.LOW_CONFIDENCE, tier, score, tc.threshold, artifact.artifact_id
    )


class Gate:
    """Holds a verified artifact; ``decide`` is the pure function above."""

    def __init__(self, artifact: CalibrationArtifact) -> None:
        verify_artifact(artifact)  # raises on tampering / structural problems
        self.artifact = artifact

    def decide(self, signals: Signals, tier: int) -> Decision:
        return decide(signals, self.artifact, tier)

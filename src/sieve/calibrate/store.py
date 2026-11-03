"""Build, serialise and verify ``CalibrationArtifact`` (canonical JSON, SHA-256)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from typing import Any

from sieve.calibrate.ltt import NEVER, LTTResult, PointResult, cp_upper
from sieve.types import (
    SCORE_NAMES,
    BackendInfo,
    CalibrationArtifact,
    SamplingParams,
    TierCalibration,
)

SCHEMA_VERSION = 1


class ArtifactError(ValueError):
    """The artifact is structurally invalid."""


class ArtifactIntegrityError(ArtifactError):
    """The stored content hash does not match the content."""


# --- canonical form -----------------------------------------------------------------


def _sampling_to_dict(s: SamplingParams) -> dict[str, Any]:
    d = asdict(s)
    d["logit_bias"] = [[k, v] for k, v in sorted(s.logit_bias)]
    return d


def _sampling_from_dict(d: dict[str, Any]) -> SamplingParams:
    return SamplingParams(
        temperature=float(d["temperature"]),
        top_p=float(d["top_p"]),
        top_k=int(d["top_k"]),
        min_p=float(d["min_p"]),
        logit_bias=tuple((int(k), float(v)) for k, v in d["logit_bias"]),
        grammar=d["grammar"],
        post_sampling_probs=bool(d["post_sampling_probs"]),
        seed=d["seed"],
        max_tokens=int(d["max_tokens"]),
    )


def _tier_to_dict(t: TierCalibration) -> dict[str, Any]:
    return {
        "identity": asdict(t.identity),
        "sampling": _sampling_to_dict(t.sampling),
        "signal": t.signal,
        "threshold": t.threshold,
        "n_samples": t.n_samples,
        "consistency_sampling": (
            None if t.consistency_sampling is None else _sampling_to_dict(t.consistency_sampling)
        ),
    }


def _tier_from_dict(d: dict[str, Any]) -> TierCalibration:
    cs = d["consistency_sampling"]
    return TierCalibration(
        identity=BackendInfo(**d["identity"]),
        sampling=_sampling_from_dict(d["sampling"]),
        signal=d["signal"],
        threshold=None if d["threshold"] is None else float(d["threshold"]),
        n_samples=int(d["n_samples"]),
        consistency_sampling=None if cs is None else _sampling_from_dict(cs),
    )


def artifact_to_dict(a: CalibrationArtifact, *, include_hash: bool = True) -> dict[str, Any]:
    d: dict[str, Any] = {
        "schema_version": a.schema_version,
        "tiers": [_tier_to_dict(t) for t in a.tiers],
        "alpha": a.alpha,
        "delta": a.delta,
        "risk_definition": a.risk_definition,
        "n_cal": a.n_cal,
        "n_answered_cal": a.n_answered_cal,
        "k_wrong_cal": a.k_wrong_cal,
        "risk_upper_bound": a.risk_upper_bound,
        "n_grid": a.n_grid,
        "method": a.method,
        "cal_hash": a.cal_hash,
        "grader_id": a.grader_id,
        "created": a.created,
    }
    if include_hash:
        d["content_hash"] = a.content_hash
    return d


def _hash_dict(d: dict[str, Any]) -> str:
    body = json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def compute_content_hash(a: CalibrationArtifact) -> str:
    return _hash_dict(artifact_to_dict(a, include_hash=False))


def verify_artifact(a: CalibrationArtifact) -> None:
    """Raise unless the artifact is valid and its hash matches its content."""
    validate_artifact(a)
    if a.content_hash != compute_content_hash(a):
        raise ArtifactIntegrityError("content hash mismatch")


def validate_artifact(a: CalibrationArtifact) -> None:
    if a.schema_version != SCHEMA_VERSION:
        raise ArtifactError(f"unsupported schema_version {a.schema_version}")
    if len(a.tiers) != 2:
        raise ArtifactError("exactly two tiers required")
    if not (0.0 < a.alpha < 1.0 and 0.0 < a.delta < 1.0):
        raise ArtifactError("alpha and delta must be in (0, 1)")
    if a.risk_definition not in ("selective", "unconditional"):
        raise ArtifactError("bad risk_definition")
    for i, t in enumerate(a.tiers, start=1):
        if not t.identity.is_complete():
            raise ArtifactError(f"tier {i}: incomplete backend identity")
        if t.signal not in SCORE_NAMES:
            raise ArtifactError(f"tier {i}: unknown signal {t.signal!r}")
        if t.threshold is not None and not math.isfinite(t.threshold):
            raise ArtifactError(f"tier {i}: non-finite threshold")
    if not (0 <= a.k_wrong_cal <= a.n_answered_cal <= a.n_cal):
        raise ArtifactError("inconsistent calibration counts")


def dumps_artifact(a: CalibrationArtifact) -> str:
    """Serialise; recomputes nothing, so a stale ``content_hash`` is caught on load."""
    validate_artifact(a)
    return json.dumps(artifact_to_dict(a), sort_keys=True, indent=2, allow_nan=False)


def loads_artifact(text: str) -> CalibrationArtifact:
    """Parse and verify. Raises ``ArtifactError`` / ``ArtifactIntegrityError``."""
    try:
        d = json.loads(text)
        art = CalibrationArtifact(
            tiers=(_tier_from_dict(d["tiers"][0]), _tier_from_dict(d["tiers"][1])),
            alpha=float(d["alpha"]),
            delta=float(d["delta"]),
            risk_definition=d["risk_definition"],
            n_cal=int(d["n_cal"]),
            n_answered_cal=int(d["n_answered_cal"]),
            k_wrong_cal=int(d["k_wrong_cal"]),
            risk_upper_bound=float(d["risk_upper_bound"]),
            n_grid=int(d["n_grid"]),
            method=str(d["method"]),
            cal_hash=str(d["cal_hash"]),
            grader_id=str(d["grader_id"]),
            created=str(d["created"]),
            content_hash=str(d["content_hash"]),
            schema_version=int(d["schema_version"]),
        )
    except (KeyError, TypeError, ValueError, IndexError) as e:
        if isinstance(e, ArtifactError):
            raise
        raise ArtifactError(f"malformed artifact: {e}") from e
    verify_artifact(art)
    return art


# --- building -----------------------------------------------------------------------


def build_artifact(
    point: PointResult,
    result: LTTResult,
    tiers: tuple[TierCalibration, TierCalibration],
    *,
    cal_hash: str,
    grader_id: str,
    created: str,
) -> CalibrationArtifact:
    """Stamp a certified ``point`` into a hashed artifact.

    ``tiers`` carries identities/signals; thresholds are overwritten from ``point``
    (``inf`` becomes ``None`` = never answer at tier 2).
    """
    if point not in result.valid:
        raise ValueError("point is not in the certified set")
    t1 = None if point.t1 == NEVER else point.t1
    t2 = None if point.t2 == NEVER else point.t2
    level = result.delta if result.method == "fixed_sequence" else result.delta / result.n_grid
    art = CalibrationArtifact(
        tiers=(replace(tiers[0], threshold=t1), replace(tiers[1], threshold=t2)),
        alpha=result.alpha,
        delta=result.delta,
        risk_definition=result.definition,
        n_cal=result.n_cal,
        n_answered_cal=point.n_answered,
        k_wrong_cal=point.k_wrong,
        risk_upper_bound=cp_upper(point.k_wrong, point.n, level),
        n_grid=result.n_grid,
        method=result.method,
        cal_hash=cal_hash,
        grader_id=grader_id,
        created=created,
    )
    return replace(art, content_hash=compute_content_hash(art))

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from sieve.calibrate import (
    NEVER,
    ArtifactError,
    ArtifactIntegrityError,
    build_artifact,
    dumps_artifact,
    loads_artifact,
    verify_artifact,
)
from sieve.calibrate.ltt import FeasibilityOutcome, LTTResult, PointResult
from sieve.types import SamplingParams
from tests.helpers import artifact, tiers


def test_roundtrip_is_identity() -> None:
    art = artifact()
    again = loads_artifact(dumps_artifact(art))
    assert again == art
    assert again.artifact_id == art.content_hash[:16] and len(art.content_hash) == 64


def test_never_threshold_roundtrips_as_none() -> None:
    art = artifact(t1=None, t2=None)
    assert art.tiers[0].threshold is None and art.tiers[1].threshold is None
    assert loads_artifact(dumps_artifact(art)) == art


def test_logit_bias_and_consistency_sampling_roundtrip() -> None:
    art = artifact()
    t1 = replace(
        art.tiers[0],
        sampling=SamplingParams(logit_bias=((5, -1.0), (2, 0.5)), grammar="root ::= x"),
        consistency_sampling=SamplingParams(temperature=0.7, seed=3),
        n_samples=5,
    )
    art2 = replace(art, tiers=(t1, art.tiers[1]))
    from sieve.calibrate.store import compute_content_hash

    art2 = replace(art2, content_hash=compute_content_hash(art2))
    assert loads_artifact(dumps_artifact(art2)) == art2


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("alpha",), 0.5),
        (("tiers", 0, "threshold"), -9.0),
        (("tiers", 1, "identity", "quant"), "Q8_0"),
        (("tiers", 0, "identity", "kv_quant"), "q4_0"),
        (("tiers", 0, "sampling", "temperature"), 0.7),
        (("n_answered_cal",), 99),
    ],
)
def test_any_tamper_is_detected(path: tuple[object, ...], value: object) -> None:
    d = json.loads(dumps_artifact(artifact()))
    node = d
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = value
    with pytest.raises(ArtifactIntegrityError):
        loads_artifact(json.dumps(d))


def test_malformed_inputs_rejected() -> None:
    good = json.loads(dumps_artifact(artifact()))
    with pytest.raises(ArtifactError):
        loads_artifact("not json {")
    bad = dict(good)
    del bad["tiers"]
    with pytest.raises(ArtifactError):
        loads_artifact(json.dumps(bad))
    bad = json.loads(json.dumps(good))
    bad["tiers"][0]["signal"] = "vibes"
    with pytest.raises(ArtifactError):
        loads_artifact(json.dumps(bad))
    bad = json.loads(json.dumps(good))
    bad["schema_version"] = 99
    with pytest.raises(ArtifactError):
        loads_artifact(json.dumps(bad))
    bad = json.loads(json.dumps(good))
    bad["tiers"][0]["identity"]["build"] = ""
    with pytest.raises(ArtifactError):
        loads_artifact(json.dumps(bad))


def test_nan_threshold_cannot_be_serialised_or_verified() -> None:
    art = artifact()
    broken = replace(art, tiers=(replace(art.tiers[0], threshold=math.nan), art.tiers[1]))
    with pytest.raises(ArtifactError):
        dumps_artifact(broken)
    with pytest.raises(ArtifactError):
        verify_artifact(broken)


def test_hash_verification_rejects_stale_hash() -> None:
    art = artifact()
    with pytest.raises(ArtifactIntegrityError):
        verify_artifact(replace(art, alpha=0.2))


def test_build_artifact_requires_certified_point() -> None:
    pt = PointResult(0.5, NEVER, 10, 0, 10, 0.0, 0.3, False, 0.2)
    res = LTTResult(
        FeasibilityOutcome.EMPTY, (), (pt,), 0.1, 0.1, "selective", "fixed_sequence", 1, 10, 0.2, 10
    )
    with pytest.raises(ValueError):
        build_artifact(pt, res, tiers(), cal_hash="x", grader_id="g", created="now")

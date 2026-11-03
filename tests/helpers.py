"""Builders shared by the tests."""

from __future__ import annotations

import math

from sieve.calibrate.ltt import FeasibilityOutcome, LTTResult, PointResult
from sieve.calibrate.store import build_artifact
from sieve.types import (
    BackendInfo,
    CalibrationArtifact,
    Completion,
    SamplingParams,
    TierCalibration,
    TokenLP,
)

GREEDY = SamplingParams(temperature=0.0)
SAMPLED = SamplingParams(temperature=0.7, top_p=0.95, seed=1)


def backend(model: str = "small", **kw: str) -> BackendInfo:
    base = {
        "model_id": model,
        "model_hash": "h1",
        "quant": "Q4_K_M",
        "kv_quant": "f16",
        "build": "abc123",
        "template_hash": "t1",
    }
    base.update(kw)
    return BackendInfo(**base)


def tok(token: str, p: float, alt: float | None = None, *, n_top: int = 2) -> TokenLP:
    """Token with probability ``p``; second alternative gets ``alt`` (default (1-p)/2)."""
    lp = math.log(p)
    a = min((1.0 - p) / 2.0, p / 2.0) if alt is None else alt
    top = [(token, lp), ("<alt1>", math.log(a))]
    if n_top > 2:
        top.append(("<alt2>", math.log(max(1.0 - p - a, 1e-9))))
    return TokenLP(token, lp, tuple(top[:n_top]))


def completion(
    probs: list[float],
    *,
    text: str | None = None,
    sampling: SamplingParams = GREEDY,
    finish: str = "stop",
    answer_start: int = 0,
) -> Completion:
    toks = tuple(tok(f"w{i}", p) for i, p in enumerate(probs))
    joined = "".join(t.token for t in toks)
    return Completion(
        text=joined if text is None else text,
        tokens=toks,
        finish=finish,  # type: ignore[arg-type]
        n_gen=len(toks),
        sampling=sampling,
        answer_start=answer_start,
    )


def tiers(
    signal1: str = "mean_logprob",
    signal2: str = "mean_logprob",
    t1: float | None = -0.5,
    t2: float | None = -0.3,
) -> tuple[TierCalibration, TierCalibration]:
    return (
        TierCalibration(backend("small"), GREEDY, signal1, t1),
        TierCalibration(backend("large", model_hash="h2"), GREEDY, signal2, t2),
    )


def artifact(
    t1: float | None = -0.5,
    t2: float | None = -0.3,
    signal1: str = "mean_logprob",
    signal2: str = "mean_logprob",
) -> CalibrationArtifact:
    pt = PointResult(
        t1=math.inf if t1 is None else t1,
        t2=math.inf if t2 is None else t2,
        n_answered=100,
        k_wrong=5,
        n=100,
        risk_hat=0.05,
        p_value=0.01,
        rejected=True,
        upper_bound=0.1,
    )
    res = LTTResult(
        FeasibilityOutcome.FEASIBLE,
        (pt,),
        (pt,),
        0.1,
        0.1,
        "selective",
        "fixed_sequence",
        10,
        500,
        0.02,
        100,
    )
    return build_artifact(
        pt,
        res,
        tiers(signal1, signal2),
        cal_hash="c" * 8,
        grader_id="exact-match-v1",
        created="2026-11-03",
    )

"""Pure confidence-signal extraction from backend completions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from sieve.signals.consistency import (
    AliasTable,
    Clusters,
    agreement,
    build_alias_table,
    cluster_answers,
    cluster_entropy,
    consistency_signals,
    extract_answer,
    normalize_answer,
)
from sieve.signals.logprob import (
    MalformedLogprobError,
    answer_span,
    logprob_signals,
    token_entropy,
    token_margin,
    validate_completion,
)
from sieve.types import BackendInfo, Completion, DefectCode, SamplingParams, Signals


def compute_signals(
    primary: Completion,
    backend: BackendInfo,
    samples: Sequence[Completion] = (),
    *,
    aliases: AliasTable | None = None,
    consistency_sampling: SamplingParams | None = None,
) -> Signals:
    """Token signals from ``primary`` plus, if ``samples`` is non-empty, consistency."""
    sig = logprob_signals(primary, backend)
    if not samples:
        return sig
    agree, neg_h, cdefects = consistency_signals(samples, aliases, served=primary)
    defects = tuple(dict.fromkeys((*sig.defects, *cdefects)))
    return replace(
        sig,
        agreement=agree,
        neg_cluster_entropy=neg_h,
        n_samples=len(samples),
        consistency_sampling=consistency_sampling or samples[0].sampling,
        defects=defects,
    )


__all__ = [
    "AliasTable",
    "Clusters",
    "DefectCode",
    "MalformedLogprobError",
    "agreement",
    "answer_span",
    "build_alias_table",
    "cluster_answers",
    "cluster_entropy",
    "compute_signals",
    "consistency_signals",
    "extract_answer",
    "logprob_signals",
    "normalize_answer",
    "token_entropy",
    "token_margin",
    "validate_completion",
]

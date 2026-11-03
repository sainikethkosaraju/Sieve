"""Shared data types for Sieve.

Everything in here is a frozen dataclass, an enum or a Protocol. There is no
network IO in this module (or in ``signals``, ``calibrate`` and ``gate``).

Conventions
-----------
* Log-probabilities are natural logs, ``<= 0``.
* Every confidence *score* is oriented so that larger means more confident.
  A tier answers when ``score >= threshold``.
* Tier 1 is the small model, tier 2 the large model.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

FinishReason = Literal["stop", "length", "error"]


class Action(enum.Enum):
    """What the gate tells the router to do."""

    ANSWER = "answer"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"


class ReasonCode(enum.Enum):
    """Why the gate chose an action. Every non-confident reason fails closed."""

    CONFIDENT = "confident"
    LOW_CONFIDENCE = "low_confidence"
    TIER_NOT_CERTIFIED = "tier_not_certified"  # tier-2 threshold is "never answer"
    LOGPROB_DEFECT = "logprob_defect"
    SAMPLING_MISMATCH = "sampling_mismatch"
    IDENTITY_MISMATCH = "identity_mismatch"
    SIGNAL_UNAVAILABLE = "signal_unavailable"


class DefectCode(enum.Enum):
    """Malformed-logprob / malformed-completion classes detected by ``signals``."""

    NAN_LOGPROB = "nan_logprob"  # NaN, +/-inf in a chosen-token logprob
    POSITIVE_LOGPROB = "positive_logprob"  # logprob > 0 (probability > 1)
    MISSING_LOGPROBS = "missing_logprobs"  # None / empty / count != n_gen
    UTF8_GAP = "utf8_gap"  # tokens do not reassemble to the text (llama.cpp #31451)
    TOP_LOGPROBS_MISSING = "top_logprobs_missing"
    TOP_LOGPROBS_INVALID = "top_logprobs_invalid"  # NaN, positive, or mass > 1
    GREEDY_MISMATCH = "greedy_mismatch"  # T=0 but chosen token is not the argmax
    TRUNCATED = "truncated"  # finish == "length"
    EMPTY_ANSWER = "empty_answer"  # no tokens in the answer span
    BACKEND_ERROR = "backend_error"  # finish == "error"
    BAD_ANSWER_START = "bad_answer_start"
    TOO_FEW_SAMPLES = "too_few_samples"  # self-consistency with fewer than 2 samples
    SAMPLE_DEFECT = "sample_defect"  # a consistency sample errored


@dataclass(frozen=True)
class SamplingParams:
    """Decoding configuration that changes the meaning of reported logprobs.

    Two ``SamplingParams`` are *calibration compatible* iff ``key()`` agrees.
    ``seed`` and ``max_tokens`` are excluded; truncation is flagged separately.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    logit_bias: tuple[tuple[int, float], ...] = ()
    grammar: str | None = None
    post_sampling_probs: bool = False  # True: logprobs come after temp/top-p/top-k
    seed: int | None = None
    max_tokens: int = 64

    def __post_init__(self) -> None:
        # canonical order, so equal configurations compare and hash equal
        object.__setattr__(self, "logit_bias", tuple(sorted(self.logit_bias)))

    def key(self) -> tuple[object, ...]:
        return (
            self.temperature,
            self.top_p,
            self.top_k,
            self.min_p,
            self.logit_bias,
            self.grammar,
            self.post_sampling_probs,
        )


@dataclass(frozen=True)
class TokenLP:
    """One generated token with its logprob and top-k alternatives (descending)."""

    token: str
    logprob: float
    top: tuple[tuple[str, float], ...] | None = None


@dataclass(frozen=True)
class Completion:
    """One backend generation. ``tokens`` is ``None`` if logprobs were not returned.

    ``answer_start`` is the index of the first token of the *answer span* (after any
    fixed preamble such as ``"The answer is"``); signals are computed on that span only.
    ``n_gen`` is the backend-reported number of generated tokens, used to detect
    dropped logprobs.
    """

    text: str
    tokens: tuple[TokenLP, ...] | None
    finish: FinishReason
    n_gen: int
    sampling: SamplingParams
    answer_start: int = 0
    n_prompt: int = 0
    t_first_token_s: float = 0.0
    t_total_s: float = 0.0


@dataclass(frozen=True)
class BackendInfo:
    """Identity of a serving configuration. All fields must match a calibration."""

    model_id: str
    model_hash: str
    quant: str
    kv_quant: str
    build: str  # e.g. pinned llama.cpp commit
    template_hash: str  # hash of the rendered prompt template

    def key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.model_id,
            self.model_hash,
            self.quant,
            self.kv_quant,
            self.build,
            self.template_hash,
        )

    def is_complete(self) -> bool:
        return all(part != "" for part in self.key())


# --- Signals -------------------------------------------------------------------------

#: Names of scores a ``CalibrationArtifact`` may be keyed on (higher = more confident).
SCORE_NAMES: tuple[str, ...] = (
    "max_prob",  # first answer-token probability
    "first_margin",  # first answer-token top1-top2 logprob gap
    "neg_first_entropy",  # minus truncated entropy of first answer token
    "mean_logprob",  # length-normalised sequence logprob
    "min_logprob",
    "mean_margin",
    "neg_mean_entropy",
    "agreement",  # self-consistency: largest-cluster share
    "neg_cluster_entropy",
)

#: Scores that need only chosen-token logprobs, not top-k lists.
CHOSEN_ONLY_SCORES: frozenset[str] = frozenset({"max_prob", "mean_logprob", "min_logprob"})
#: Scores that need no logprobs at all.
SAMPLE_ONLY_SCORES: frozenset[str] = frozenset({"agreement", "neg_cluster_entropy"})


@dataclass(frozen=True)
class Signals:
    """Confidence signals for one query on one tier.

    A value is ``None`` when it could not be computed (see ``defects``). The gate
    treats any relevant defect as fail-closed.
    """

    backend: BackendInfo
    sampling: SamplingParams
    n_answer_tokens: int
    max_prob: float | None = None
    first_margin: float | None = None
    neg_first_entropy: float | None = None
    mean_logprob: float | None = None
    min_logprob: float | None = None
    sum_logprob: float | None = None
    mean_margin: float | None = None
    neg_mean_entropy: float | None = None
    agreement: float | None = None
    neg_cluster_entropy: float | None = None
    n_samples: int = 0
    consistency_sampling: SamplingParams | None = None
    defects: tuple[DefectCode, ...] = ()
    #: Free-form extras (e.g. answer length) that the gate never reads.
    extras: Mapping[str, float] = field(default_factory=dict)

    def score(self, name: str) -> float | None:
        if name not in SCORE_NAMES:
            raise KeyError(f"unknown score {name!r}")
        value: float | None = getattr(self, name)
        return value


# --- Calibration artifact -----------------------------------------------------------


@dataclass(frozen=True)
class TierCalibration:
    """Per-tier part of an artifact: identity, sampling, signal, threshold.

    ``threshold is None`` means "never answer at this tier" (only meaningful for tier 2).
    """

    identity: BackendInfo
    sampling: SamplingParams
    signal: str
    threshold: float | None
    n_samples: int = 0  # required k for consistency signals, else 0
    consistency_sampling: SamplingParams | None = None


RiskDefinition = Literal["selective", "unconditional"]


@dataclass(frozen=True)
class CalibrationArtifact:
    """Frozen output of calibration, keyed by model/quant/KV/signal/template/build.

    ``content_hash`` is the SHA-256 of the canonical JSON of everything else; it is
    verified on load and by the gate. ``tiers`` has exactly two entries.
    """

    tiers: tuple[TierCalibration, TierCalibration]
    alpha: float
    delta: float
    risk_definition: RiskDefinition
    n_cal: int
    n_answered_cal: int
    k_wrong_cal: int
    risk_upper_bound: float  # one-sided Clopper-Pearson bound at level delta
    n_grid: int  # m, size of the family tested
    method: str  # "fixed_sequence" | "bonferroni"
    cal_hash: str  # hash of the calibration table (split B)
    grader_id: str
    created: str
    content_hash: str = ""
    schema_version: int = 1

    @property
    def artifact_id(self) -> str:
        return self.content_hash[:16]


# --- Decision -----------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: ReasonCode
    tier: int
    score: float | None
    threshold: float | None
    artifact_id: str
    defects: tuple[DefectCode, ...] = ()

    @property
    def fail_closed(self) -> bool:
        return self.reason not in (
            ReasonCode.CONFIDENT,
            ReasonCode.LOW_CONFIDENCE,
            ReasonCode.TIER_NOT_CERTIFIED,
        )


# --- Protocols ----------------------------------------------------------------------


class Backend(Protocol):
    """Backends are the only code that performs network IO (not defined in Phase A)."""

    @property
    def info(self) -> BackendInfo: ...

    async def generate(
        self, prompt: str, params: SamplingParams, *, n_probs: int
    ) -> Completion: ...


class Calibrator(Protocol):
    """A monotone map from raw score to P(correct), used for ranking/reporting only."""

    def fit(
        self, scores: npt.NDArray[np.float64], correct: npt.NDArray[np.bool_]
    ) -> Calibrator: ...

    def predict(self, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

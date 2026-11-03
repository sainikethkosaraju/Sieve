"""Deterministic scripted backend for CI and demos (no network, no model).

Behaviour is a pure function of ``(seed, prompt)``: whether the answer is correct is
decided by a hash-derived uniform draw against ``accuracy``; confidence is high for
correct answers and low for wrong ones (``correct_conf`` / ``wrong_conf``, with a
small hash-derived jitter), so a gate on ``mean_logprob`` has real signal. Defects can
be injected, optionally only after N calls, to exercise the fail-closed paths.
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from sieve.backends.base import BackendError
from sieve.types import BackendInfo, Completion, SamplingParams, TokenLP


class MockDefect(enum.Enum):
    NAN_LOGPROB = "nan_logprob"
    POSITIVE_LOGPROB = "positive_logprob"
    DROPPED_LOGPROB = "dropped_logprob"  # one logprob missing -> count mismatch
    NO_LOGPROBS = "no_logprobs"
    NO_TOP = "no_top"
    UTF8_GAP = "utf8_gap"
    TRUNCATED = "truncated"
    ERROR_FINISH = "error_finish"
    RAISE = "raise"  # BackendError
    NONDETERMINISTIC = "nondeterministic"  # logprobs drift between identical calls
    GREEDY_MISMATCH = "greedy_mismatch"
    TOP_MISSING_CHOSEN = "top_missing_chosen"  # chosen token not in top-k
    HANG = "hang"  # never returns (cancellation tests)


def _u(seed: int, salt: str, prompt: str) -> float:
    h = hashlib.sha256(f"{seed}|{salt}|{prompt}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0**64


@dataclass
class MockBackend:
    """Scripted backend. Mutable counters are public for assertions."""

    info: BackendInfo
    accuracy: float = 1.0
    correct_conf: float = 0.97
    wrong_conf: float = 0.35
    seed: int = 0
    latency_s: float = 0.0
    defect: MockDefect | None = None
    defect_after: int = 0  # inject only once `calls` exceeds this
    answer_fn: Callable[[str], str] | None = None
    n_tokens: int = 3
    n_prompt_tokens: int = 10
    # self-consistency: probability that a T>0 sample repeats the greedy answer, for
    # correct answers (sample_stability) and wrong ones (wrong_stability, usually low)
    sample_stability: float = 0.95
    wrong_stability: float = 0.1
    calls: int = 0
    completed: int = 0
    cancelled: int = 0
    in_flight: int = 0
    prompts: list[str] = field(default_factory=list)
    params: list[SamplingParams] = field(default_factory=list)

    def truth(self, prompt: str) -> str:
        if self.answer_fn is not None:
            return self.answer_fn(prompt)
        return "ans" + hashlib.sha256(prompt.encode()).hexdigest()[:6]

    def is_correct(self, prompt: str) -> bool:
        return _u(self.seed, "acc", prompt) < self.accuracy

    def _text(self, prompt: str, params: SamplingParams) -> tuple[str, bool]:
        correct = self.is_correct(prompt)
        if params.temperature > 0.0:
            stable = _u(self.seed, f"s{params.seed}", prompt) < (
                self.sample_stability if correct else self.wrong_stability
            )
            if not stable:
                return f"alt{params.seed}x{hashlib.sha256(prompt.encode()).hexdigest()[:3]}", False
        if correct:
            return self.truth(prompt), True
        return "wrong" + hashlib.sha256(f"{self.seed}{prompt}".encode()).hexdigest()[:6], False

    def _pieces(self, text: str) -> list[str]:
        n = max(1, min(self.n_tokens, len(text)))
        size = math.ceil(len(text) / n)
        return [text[i : i + size] for i in range(0, len(text), size)]

    async def generate(self, prompt: str, params: SamplingParams, *, n_probs: int) -> Completion:
        self.calls += 1
        self.in_flight += 1
        self.prompts.append(prompt)
        self.params.append(params)
        try:
            if self.latency_s > 0:
                await asyncio.sleep(self.latency_s)
            active = self.defect if self.calls > self.defect_after else None
            if active is MockDefect.HANG:
                await asyncio.Event().wait()
            if active is MockDefect.RAISE:
                raise BackendError("mock: injected failure")
            comp = self._build(prompt, params, n_probs, active)
            self.completed += 1
            return comp
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.in_flight -= 1

    def _build(
        self, prompt: str, params: SamplingParams, n_probs: int, d: MockDefect | None
    ) -> Completion:
        text, correct = self._text(prompt, params)
        conf = self.correct_conf if correct else self.wrong_conf
        conf = min(0.999, max(0.01, conf + (_u(self.seed, "jit", prompt) - 0.5) * 0.04))
        pieces = self._pieces(text)
        cut = len(pieces) > params.max_tokens
        if cut:
            pieces = pieces[: params.max_tokens]
            text = "".join(pieces)
        toks: list[TokenLP] = []
        for i, piece in enumerate(pieces):
            p = conf if i == 0 else min(0.999, conf + (1 - conf) / 2)
            lp = math.log(p)
            alt = min((1.0 - p) / 2.0, p / 2.0)
            top: tuple[tuple[str, float], ...] | None = (
                (piece, lp),
                ("<alt>", math.log(alt)),
            )
            if n_probs <= 0:
                top = None
            toks.append(TokenLP(piece, lp, top))
        finish = "length" if cut else "stop"
        out_text = text
        tokens: list[TokenLP] | None = toks
        n_gen = len(toks)
        if d is MockDefect.NAN_LOGPROB:
            toks[0] = replace(toks[0], logprob=math.nan)
        elif d is MockDefect.POSITIVE_LOGPROB:
            toks[0] = replace(toks[0], logprob=0.5)
        elif d is MockDefect.DROPPED_LOGPROB:
            tokens = toks[:-1] if len(toks) > 1 else []
            if len(toks) == 1:
                tokens = None
        elif d is MockDefect.NO_LOGPROBS:
            tokens = None
        elif d is MockDefect.NO_TOP:
            tokens = [replace(t, top=None) for t in toks]
        elif d is MockDefect.UTF8_GAP:
            toks[0] = replace(toks[0], token="�" + toks[0].token[1:])
        elif d is MockDefect.TRUNCATED:
            finish = "length"
        elif d is MockDefect.ERROR_FINISH:
            finish = "error"
        elif d is MockDefect.NONDETERMINISTIC:
            drift = (_u(self.seed, f"nd{self.calls}", prompt) - 0.5) * 0.1
            toks = [
                replace(
                    t,
                    logprob=min(0.0, t.logprob + drift),
                    top=((t.token, min(0.0, t.logprob + drift)), *(t.top[1:] if t.top else ())),
                )
                for t in toks
            ]
            tokens = toks
        elif d is MockDefect.GREEDY_MISMATCH:
            toks[0] = replace(toks[0], logprob=toks[0].logprob - 1.0)
        elif d is MockDefect.TOP_MISSING_CHOSEN:
            toks[0] = replace(toks[0], token=toks[0].token, top=(("<other>", toks[0].logprob),))
        return Completion(
            text=out_text,
            tokens=None if tokens is None else tuple(tokens),
            finish=finish,  # type: ignore[arg-type]
            n_gen=n_gen,
            sampling=params,
            answer_start=0,
            n_prompt=self.n_prompt_tokens,
            t_first_token_s=self.latency_s / 2,
            t_total_s=self.latency_s,
        )

    async def refresh(self) -> bool:
        return False

    async def aclose(self) -> None:
        return None

"""Deterministic fake backend and judge for the offline mock pipeline.

``FakeBackend`` implements the ``sieve.types.Backend`` Protocol. It is a *test double*: its
accuracy, confidence and timing numbers are generated from hashes of the prompt, so every
result computed from it is synthetic and must be labelled as such. It can inject transient
failures (to exercise wasted-attempt accounting) and 'kill' the process after N generations
(to exercise resumability).
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from sieve.eval import prompts
from sieve.eval._util import stable_int, stable_unit
from sieve.eval.config import CellSpec
from sieve.eval.datasets import MCQ_LETTERS
from sieve.eval.grading import SimpleQALabel
from sieve.eval.synthetic import (
    dataset_shift,
    difficulty,
    gold_letter,
    gold_text,
    question_dataset,
    question_index,
    sigmoid,
)
from sieve.signals.consistency import normalize_answer
from sieve.types import BackendInfo, Completion, SamplingParams, TokenLP

#: weight-quant effects on (skill drop, confidence scale, extra confidence noise, speed factor)
QUANT_EFFECT: dict[str, tuple[float, float, float, float]] = {
    "f16": (0.0, 1.00, 0.00, 1.6),
    "Q8_0": (0.02, 1.00, 0.05, 1.2),
    "Q4_K_M": (0.10, 0.95, 0.15, 1.0),
    "Q3_K_M": (0.35, 0.75, 0.30, 0.95),
}
KV_EFFECT: dict[str, tuple[float, float]] = {  # (skill drop, extra noise)
    "f16": (0.0, 0.0),
    "q8_0": (0.0, 0.03),
    "q4_0": (0.05, 0.15),
}
TIER_DEFAULTS: dict[str, dict[str, float]] = {
    "S": {"skill": 1.1, "base_s": 0.15, "per_tok_s": 0.02, "watts": 70.0},
    "L": {"skill": 2.4, "base_s": 0.80, "per_tok_s": 0.05, "watts": 120.0},
}


class SimulatedKillError(BaseException):
    """Raised instead of ``os._exit`` in in-process kill tests (not an ``Exception``)."""


class SimClock:
    """Simulated monotonic clock that the fake backend advances by its own latencies."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass
class FakeBackend:
    model_id: str
    tier: str
    quant: str
    kv_quant: str
    skill: float
    clock: SimClock | None = None
    failure_rate: float = 0.02
    kill_after: int | None = None
    hard_kill: bool = False
    verbose_rate: float = 0.08
    truncate_rate: float = 0.0  # share of generations that never stop (finish=length)
    calls: int = 0
    successes: int = 0
    _info: BackendInfo = field(init=False)

    def __post_init__(self) -> None:
        self._info = BackendInfo(
            model_id=self.model_id,
            model_hash=f"synthetic-hash-{self.model_id}",
            quant=self.quant,
            kv_quant=self.kv_quant,
            build="fake-backend-1",
            template_hash=prompts.template_hash(),
        )

    @property
    def info(self) -> BackendInfo:
        return self._info

    def _effects(self) -> tuple[float, float, float, float]:
        qd, qs, qn, speed = QUANT_EFFECT.get(self.quant, QUANT_EFFECT["Q4_K_M"])
        kd, kn = KV_EFFECT.get(self.kv_quant, (0.0, 0.0))
        return qd + kd, qs, qn + kn, speed

    async def generate(self, prompt: str, params: SamplingParams, *, n_probs: int) -> Completion:
        self.calls += 1
        if self.kill_after is not None and self.successes >= self.kill_after:
            if self.hard_kill:
                os._exit(137)
            raise SimulatedKillError
        d = TIER_DEFAULTS[self.tier]
        speed = self._effects()[3]
        latency = (d["base_s"] / speed) * (
            1.0 + 0.2 * (stable_unit("jit", prompt, self.model_id) - 0.5)
        )
        fail_u = stable_unit("fail", self.model_id, self.quant, self.kv_quant, prompt, params.seed)
        if fail_u < self.failure_rate:
            if self.clock:
                self.clock.advance(0.3 * latency)
            if fail_u < self.failure_rate / 2:
                raise RuntimeError("synthetic transient backend failure")
            self.successes += 1
            return Completion("", None, "error", 0, params, t_total_s=0.3 * latency)
        if (
            stable_unit("trunc", self.model_id, self.quant, self.kv_quant, prompt)
            < self.truncate_rate
        ):
            comp = self._truncated(prompt, params, latency, d)
        else:
            comp = self._complete(prompt, params, latency, d)
        if self.clock:
            self.clock.advance(comp.t_total_s)
        self.successes += 1
        return comp

    def _complete(
        self, prompt: str, params: SamplingParams, latency: float, d: dict[str, float]
    ) -> Completion:
        i = question_index(prompt)
        is_mcq = "\nA. " in prompt
        dataset = question_dataset(prompt)
        diff = difficulty(dataset, i) + dataset_shift(dataset)
        qdrop, cscale, cnoise, _ = self._effects()
        z = self.skill - 1.4 * diff - qdrop
        eo = 0.6 * math.log(
            max(stable_unit("eo", self.tier, prompt), 1e-9)
            / max(1 - stable_unit("eo", self.tier, prompt), 1e-9)
        )
        correct = z + eo >= 0.0
        rng_c = random.Random(stable_int("ec", self.model_id, self.quant, self.kv_quant, prompt))
        ec = rng_c.gauss(0.0, 1.0)
        conf = sigmoid(cscale * (1.6 * z + (0.5 + cnoise) * ec))
        conf = min(max(conf, 0.03), 0.995)
        gold = gold_letter(i) if is_mcq else gold_text(i)
        wrong_alt = (
            MCQ_LETTERS[(MCQ_LETTERS.index(gold) + 1 + stable_int("w", prompt) % 3) % 4]
            if is_mcq
            else f"Wrong{stable_int('w', prompt, bits=10)}"
        )
        answer = gold if correct else wrong_alt
        sampled = params.temperature > 0.0
        if sampled:
            rng = random.Random(params.seed if params.seed is not None else 0)
            if rng.random() > 0.35 + 0.6 * conf:
                answer = f"Alt{rng.randrange(6)}" if not is_mcq else rng.choice(MCQ_LETTERS)
        verbose = (not is_mcq) and stable_unit("verbose", self.model_id, prompt) < self.verbose_rate
        return self._render(answer, conf, verbose, params, latency, d, prompt)

    def _truncated(
        self, prompt: str, params: SamplingParams, latency: float, d: dict[str, float]
    ) -> Completion:
        """A generation that runs to ``max_tokens`` (the real 4-bit-KV collapse): finish=length."""
        n = max(int(params.max_tokens), 1)
        toks = tuple(
            TokenLP(" the", math.log(0.2), ((" the", math.log(0.2)), (" a", math.log(0.1))))
            for _ in range(n)
        )
        return Completion(
            text=" the" * n,
            tokens=toks,
            finish="length",
            n_gen=n,
            sampling=params,
            n_prompt=len(prompt.split()),
            t_first_token_s=0.4 * latency,
            t_total_s=latency + d["per_tok_s"] * n,
        )

    def _render(
        self,
        answer: str,
        conf: float,
        verbose: bool,
        params: SamplingParams,
        latency: float,
        d: dict[str, float],
        prompt: str,
    ) -> Completion:
        chunks = [answer[k : k + 4] for k in range(0, len(answer), 4)] or [""]
        pieces: list[tuple[str, float]] = []
        if verbose:
            pieces += [("The", 0.99), (" answer", 0.99), (" is", 0.99), (" ", 0.99)]
        for k, ch in enumerate(chunks):
            p = conf if k == 0 else min(0.999, conf + (1.0 - conf) * 0.6)
            pieces.append((ch, p))
        if verbose:
            pieces.append((".", 0.98))
        toks: list[TokenLP] = []
        for tok, p in pieces:
            a1 = min((1.0 - p) * 0.6, p * 0.999)
            a2 = min((1.0 - p) * 0.4, a1 * 0.999)
            top = ((tok, math.log(p)), ("<alt1>", math.log(a1)), ("<alt2>", math.log(a2)))
            toks.append(TokenLP(tok, math.log(p), top))
        text = "".join(t.token for t in toks)
        n = len(toks)
        t_total = latency + d["per_tok_s"] * n
        comp = Completion(
            text=text,
            tokens=tuple(toks),
            finish="stop",
            n_gen=n,
            sampling=params,
            n_prompt=len(prompt.split()),
            t_first_token_s=0.4 * latency,
            t_total_s=t_total,
        )
        return comp


class FakeProvider:
    """``BackendProvider`` producing ``FakeBackend`` for a cell (params in ``cell.extra``)."""

    def __init__(
        self,
        clock: SimClock | None = None,
        *,
        kill_after: int | None = None,
        hard_kill: bool = False,
        failure_rate: float = 0.02,
    ) -> None:
        self.clock = clock
        self.kill_after = kill_after
        self.hard_kill = hard_kill
        self.failure_rate = failure_rate
        self.backends: dict[str, FakeBackend] = {}

    def make(self, cell: CellSpec) -> FakeBackend:
        extra: dict[str, Any] = cell.extra.get("fake", {})
        d = TIER_DEFAULTS[cell.tier]
        return FakeBackend(
            model_id=cell.model_id,
            tier=cell.tier,
            quant=cell.quant,
            kv_quant=cell.kv_quant,
            skill=float(extra.get("skill", d["skill"])),
            clock=self.clock,
            failure_rate=float(extra.get("failure_rate", self.failure_rate)),
            truncate_rate=float(extra.get("truncate_rate", 0.0)),
            kill_after=self.kill_after,
            hard_kill=self.hard_kill,
        )

    @contextmanager
    def open(self, cell: CellSpec) -> Iterator[FakeBackend]:
        b = self.make(cell)
        self.backends[cell.name] = b
        yield b


class SyntheticJudge:
    """Deterministic stand-in for the SimpleQA judge, used only by the mock pipeline."""

    judge_id = "simpleqa-judge:SYNTHETIC-test-double"
    model_id = "synthetic-judge"

    def judge(self, question: str, gold: str, prediction: str) -> SimpleQALabel:
        if normalize_answer(prediction) == "":
            return SimpleQALabel.NOT_ATTEMPTED
        return (
            SimpleQALabel.CORRECT
            if normalize_answer(prediction) == normalize_answer(gold)
            else SimpleQALabel.INCORRECT
        )

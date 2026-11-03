"""Startup logprob self-test (architect decision 3, red-team kill test 2).

Runs fixed probe prompts through any :class:`~sieve.types.Backend` and checks that the
reported logprobs are usable: finite and ``<= 0``, one entry per generated token, the
chosen token appears in its top-k list, top-k lists are sorted with mass ``<= 1``,
UTF-8 pieces reassemble, and greedy decoding is deterministic. A failing self-test
sets ``BackendHealth.logprobs_ok = False`` and the router then refuses to use that
tier's logprob signals (the gate fails closed).
"""

from __future__ import annotations

import math

from sieve.backends.base import BackendError, CheckResult, SelfTestResult
from sieve.signals import validate_completion
from sieve.types import Backend, Completion, DefectCode, SamplingParams

PROBE_PROMPT = "Question: What is the capital of France?\nAnswer:"
UTF8_PROBE_PROMPT = "Repeat exactly: naïve café 日本語 🙂\nRepeat exactly: naïve café 日本語 🙂\n"

_NUMERIC = {DefectCode.NAN_LOGPROB, DefectCode.POSITIVE_LOGPROB}
_COUNT = {DefectCode.MISSING_LOGPROBS, DefectCode.EMPTY_ANSWER}
_TOP = {
    DefectCode.TOP_LOGPROBS_MISSING,
    DefectCode.TOP_LOGPROBS_INVALID,
    DefectCode.GREEDY_MISMATCH,
}


def _top_contains_chosen(c: Completion, tol: float) -> bool:
    if c.tokens is None or not c.tokens:
        return False
    for t in c.tokens:
        if t.top is None or not any(
            tok == t.token and math.isclose(lp, t.logprob, abs_tol=tol) for tok, lp in t.top
        ):
            return False
    return True


async def run_logprob_self_test(
    backend: Backend,
    *,
    n_probs: int = 20,
    repeats: int = 3,
    eps: float = 1e-3,
    max_tokens: int = 12,
    strict_utf8: bool = True,
) -> SelfTestResult:
    """Run the probes. Never raises: a backend failure is reported as failed checks."""
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    checks: list[CheckResult] = []
    runs: list[Completion] = []
    try:
        for _ in range(max(repeats, 2)):
            runs.append(await backend.generate(PROBE_PROMPT, params, n_probs=n_probs))
        utf8 = await backend.generate(UTF8_PROBE_PROMPT, params, n_probs=n_probs)
    except BackendError as e:
        return SelfTestResult((CheckResult("reachable", False, type(e).__name__),))
    except Exception as e:  # adapter bug: still fail closed, never crash startup
        return SelfTestResult((CheckResult("reachable", False, type(e).__name__),))

    # truncation by max_tokens is expected for the probes and is not a logprob defect
    def defects(c: Completion) -> set[DefectCode]:
        return set(validate_completion(c)) - {DefectCode.TRUNCATED}

    first = runs[0]
    # strict_utf8=False: the non-ASCII probe only feeds the advisory utf8 check. llama.cpp drops
    # logprob entries for multi-token UTF-8 characters (upstream issue 31451, reproduced on a T4
    # with Qwen2.5-7B); affected items still fail closed per item (UTF8_GAP/MISSING_LOGPROBS).
    d = defects(first) | (defects(utf8) if strict_utf8 else set())
    checks.append(
        CheckResult(
            "finite_nonpositive",
            not (d & _NUMERIC),
            ",".join(sorted(x.value for x in d & _NUMERIC)),
        )
    )
    checks.append(
        CheckResult(
            "count_matches_tokens",
            not (d & _COUNT) and first.tokens is not None and len(first.tokens) == first.n_gen,
            ",".join(sorted(x.value for x in d & _COUNT)),
        )
    )
    top_ok = not (d & _TOP) and all(_top_contains_chosen(c, 1e-6 + eps) for c in runs)
    checks.append(
        CheckResult("top_k_contains_chosen", top_ok, ",".join(sorted(x.value for x in d & _TOP)))
    )
    checks.append(
        CheckResult(
            "utf8_reassembles",
            DefectCode.UTF8_GAP not in defects(utf8)
            and DefectCode.MISSING_LOGPROBS not in defects(utf8),
            ",".join(sorted(x.value for x in defects(utf8))),
            advisory=not strict_utf8,
        )
    )
    checks.append(_determinism(runs, eps))
    return SelfTestResult(tuple(checks))


def _determinism(runs: list[Completion], eps: float) -> CheckResult:
    ref = runs[0]
    for other in runs[1:]:
        if other.text != ref.text or other.n_gen != ref.n_gen:
            return CheckResult("greedy_deterministic", False, "text differs across repeats")
        if ref.tokens is None or other.tokens is None or len(ref.tokens) != len(other.tokens):
            return CheckResult("greedy_deterministic", False, "token lists differ")
        for a, b in zip(ref.tokens, other.tokens, strict=True):
            if a.token != b.token or not abs(a.logprob - b.logprob) <= eps:
                return CheckResult("greedy_deterministic", False, f"logprob drift > {eps}")
    return CheckResult("greedy_deterministic", True)

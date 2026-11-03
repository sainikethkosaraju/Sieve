"""Token-logprob confidence signals. Pure functions over ``Completion`` objects.

All signals are computed on the *answer span* (``Completion.answer_start`` onwards).
A malformed completion never yields numbers: ``validate_completion`` reports defect
codes and ``logprob_signals`` then returns a ``Signals`` with every value ``None`` so
the gate can fail closed.
"""

from __future__ import annotations

import math

from sieve.types import BackendInfo, Completion, DefectCode, Signals, TokenLP

#: Probabilities may exceed 1 by float noise only.
POSITIVE_TOL = 1e-6
#: Allowed excess of top-k probability mass over 1.
MASS_TOL = 1e-3
#: Slack for "chosen token is the argmax at T=0" (logprob units).
GREEDY_TOL = 1e-3


class MalformedLogprobError(ValueError):
    """Raised by the strict single-value helpers when the input is unusable."""


def _bad_logprob(x: float) -> DefectCode | None:
    if not math.isfinite(x):
        return DefectCode.NAN_LOGPROB
    if x > POSITIVE_TOL:
        return DefectCode.POSITIVE_LOGPROB
    return None


def validate_completion(c: Completion, *, need_top: bool = True) -> tuple[DefectCode, ...]:
    """Return every defect found in ``c`` (empty tuple if clean). Never raises."""
    found: list[DefectCode] = []

    def add(code: DefectCode) -> None:
        if code not in found:
            found.append(code)

    if c.finish == "error":
        add(DefectCode.BACKEND_ERROR)
    elif c.finish == "length":
        add(DefectCode.TRUNCATED)

    toks = c.tokens
    if toks is None or len(toks) == 0:
        add(
            DefectCode.EMPTY_ANSWER
            if (c.n_gen == 0 and c.text == "")
            else DefectCode.MISSING_LOGPROBS
        )
        return tuple(found)
    if not c.text.strip():
        add(DefectCode.EMPTY_ANSWER)  # F-13: whitespace-only output is never an answer
    if len(toks) != c.n_gen:
        add(DefectCode.MISSING_LOGPROBS)
    if not 0 <= c.answer_start <= len(toks):
        add(DefectCode.BAD_ANSWER_START)
    elif c.answer_start == len(toks):
        add(DefectCode.EMPTY_ANSWER)

    joined = "".join(t.token for t in toks)
    if joined != c.text or ("�" in joined and "�" not in c.text):
        add(DefectCode.UTF8_GAP)

    top_missing = False
    for t in toks:
        bad = _bad_logprob(t.logprob)
        if bad is not None:
            add(bad)
        if t.top is None or len(t.top) < 2:
            top_missing = True
            continue
        mass = 0.0
        prev = math.inf
        for _, lp in t.top:
            if _bad_logprob(lp) is not None or lp > prev + 1e-12:
                add(DefectCode.TOP_LOGPROBS_INVALID)
                mass = math.nan
                break
            prev = lp
            mass += math.exp(lp)
        else:
            if mass > 1.0 + MASS_TOL:
                add(DefectCode.TOP_LOGPROBS_INVALID)
            if (
                c.sampling.temperature == 0.0
                and math.isfinite(t.logprob)
                and t.logprob < t.top[0][1] - GREEDY_TOL
            ):
                add(DefectCode.GREEDY_MISMATCH)
    if top_missing and need_top:
        add(DefectCode.TOP_LOGPROBS_MISSING)
    return tuple(found)


def answer_span(c: Completion) -> tuple[TokenLP, ...]:
    """Tokens of the answer span; raises ``MalformedLogprobError`` if unusable."""
    if c.tokens is None:
        raise MalformedLogprobError("no token logprobs")
    if not 0 <= c.answer_start < len(c.tokens):
        raise MalformedLogprobError("empty or invalid answer span")
    return c.tokens[c.answer_start :]


def token_entropy(top: tuple[tuple[str, float], ...]) -> float:
    """Entropy (nats) of the top-k distribution renormalised to sum to 1.

    This is a *truncated* entropy: it lower-bounds nothing in general and is
    flatness-dependent; callers must not compare it to full-vocabulary entropy.
    """
    if len(top) == 0:
        raise MalformedLogprobError("empty top list")
    lps = [lp for _, lp in top]
    if any(not math.isfinite(lp) or lp > POSITIVE_TOL for lp in lps):
        raise MalformedLogprobError("invalid top logprobs")
    m = max(lps)
    z = sum(math.exp(lp - m) for lp in lps)
    log_z = m + math.log(z)
    h = 0.0
    for lp in lps:
        lq = lp - log_z  # log of renormalised probability
        h -= math.exp(lq) * lq
    return max(h, 0.0)


def token_margin(top: tuple[tuple[str, float], ...]) -> float:
    """Gap between best and second-best logprob (>= 0)."""
    if len(top) < 2:
        raise MalformedLogprobError("need at least two alternatives for a margin")
    return top[0][1] - top[1][1]


def _empty_signals(c: Completion, backend: BackendInfo, defects: tuple[DefectCode, ...]) -> Signals:
    return Signals(
        backend=backend,
        sampling=c.sampling,
        n_answer_tokens=0,
        defects=defects,
    )


def logprob_signals(c: Completion, backend: BackendInfo) -> Signals:
    """Compute every token-level signal, or an all-``None`` ``Signals`` with defects."""
    defects = validate_completion(c)
    fatal = tuple(d for d in defects if d is not DefectCode.TOP_LOGPROBS_MISSING)
    if fatal:
        return _empty_signals(c, backend, defects)
    span = answer_span(c)
    lps = [min(t.logprob, 0.0) for t in span]
    n = len(lps)
    s = sum(lps)
    max_prob = math.exp(lps[0])
    first_margin: float | None = None
    neg_first_h: float | None = None
    mean_margin: float | None = None
    neg_mean_h: float | None = None
    if DefectCode.TOP_LOGPROBS_MISSING not in defects:
        tops = [t.top for t in span]
        assert all(t is not None for t in tops)
        margins = [token_margin(t) for t in tops if t is not None]
        ents = [token_entropy(t) for t in tops if t is not None]
        first_margin = margins[0]
        neg_first_h = -ents[0]
        mean_margin = sum(margins) / n
        neg_mean_h = -(sum(ents) / n)
    return Signals(
        backend=backend,
        sampling=c.sampling,
        n_answer_tokens=n,
        max_prob=max_prob,
        first_margin=first_margin,
        neg_first_entropy=neg_first_h,
        mean_logprob=s / n,
        min_logprob=min(lps),
        sum_logprob=s,
        mean_margin=mean_margin,
        neg_mean_entropy=neg_mean_h,
        defects=defects,
        extras={"answer_chars": float(len(c.text))},
    )

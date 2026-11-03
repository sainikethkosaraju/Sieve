"""Self-consistency over k sampled completions.

Answers are normalised and clustered by exact normalised-string match, then merged
through an optional alias table. There is no LLM or NLI clustering in v1.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from sieve.types import Completion, DefectCode

_ARTICLES = frozenset({"a", "an", "the"})
_PREFIXES = ("the answer is", "answer is", "answer:", "a:")
_WS = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Lowercase, strip accents, punctuation and articles, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text)
    no_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    chars = [" " if unicodedata.category(ch)[0] in ("P", "S") else ch.lower() for ch in no_marks]
    words = [w for w in _WS.split("".join(chars).strip()) if w and w not in _ARTICLES]
    return " ".join(words)


def extract_answer(text: str) -> str:
    """First non-empty line with a leading 'The answer is' style preamble removed."""
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    low = line.lower()
    for p in _PREFIXES:
        if low.startswith(p):
            line = line[len(p) :].strip()
            break
    return line.rstrip(".")


AliasTable = Mapping[str, str]


def build_alias_table(groups: Iterable[Iterable[str]]) -> dict[str, str]:
    """Map every normalised alias in a group to the group's first normalised member."""
    table: dict[str, str] = {}
    for group in groups:
        members = [normalize_answer(g) for g in group]
        if not members:
            continue
        canon = members[0]
        for m in members:
            table[m] = canon
    return table


@dataclass(frozen=True)
class Clusters:
    labels: tuple[str, ...]  # canonical key per input answer
    sizes: tuple[int, ...]  # descending

    @property
    def n(self) -> int:
        return len(self.labels)


def cluster_answers(answers: Sequence[str], aliases: AliasTable | None = None) -> Clusters:
    keys = []
    for a in answers:
        k = normalize_answer(a)
        keys.append(aliases.get(k, k) if aliases else k)
    counts = Counter(keys)
    return Clusters(tuple(keys), tuple(sorted(counts.values(), reverse=True)))


def agreement(clusters: Clusters) -> float:
    """Share of the largest cluster, in (0, 1]."""
    if clusters.n == 0:
        raise ValueError("no answers")
    return clusters.sizes[0] / clusters.n


def cluster_entropy(clusters: Clusters) -> float:
    """Shannon entropy (nats) of the cluster-size distribution."""
    if clusters.n == 0:
        raise ValueError("no answers")
    h = 0.0
    for s in clusters.sizes:
        p = s / clusters.n
        h -= p * math.log(p)
    return max(h, 0.0)


def agreement_with_served(served_key: str, clusters: Clusters) -> float:
    """Share of samples whose cluster key equals the served (greedy) answer's key (F-03)."""
    if clusters.n == 0:
        raise ValueError("no answers")
    return sum(1 for k in clusters.labels if k == served_key) / clusters.n


def consistency_signals(
    samples: Sequence[Completion],
    aliases: AliasTable | None = None,
    *,
    served: Completion | None = None,
) -> tuple[float | None, float | None, tuple[DefectCode, ...]]:
    """Return ``(agreement, neg_cluster_entropy, defects)``; values are ``None`` on defect.

    With ``served`` (the greedy answer actually returned to the caller) agreement is the share
    of samples that agree with it, and the entropy is taken over samples plus the served answer
    (F-03). Without it, agreement falls back to the largest-cluster share. Clustering uses
    only deployment-time information: normalised strings of the answers themselves, plus an
    optional alias table that must never be derived from gold answers (F-02).
    """
    defects: list[DefectCode] = []
    if len(samples) < 2:
        defects.append(DefectCode.TOO_FEW_SAMPLES)
    if any(s.finish != "stop" or not s.text.strip() for s in samples):
        defects.append(DefectCode.SAMPLE_DEFECT)
    if defects:
        return None, None, tuple(defects)
    answers = [extract_answer(s.text) for s in samples]
    if served is None:
        cl = cluster_answers(answers, aliases)
        return agreement(cl), -cluster_entropy(cl), ()
    cl_all = cluster_answers([*answers, extract_answer(served.text)], aliases)
    *sample_keys, served_key = cl_all.labels
    if not served_key:
        return 0.0, -cluster_entropy(cl_all), ()
    agree = sum(1 for k in sample_keys if k == served_key) / len(sample_keys)
    return agree, -cluster_entropy(cl_all), ()

"""Synthetic benchmark items for the offline mock pipeline.

Everything produced here is fake. Question text embeds an index so the fake backend can
recompute the gold answer and a latent difficulty from the prompt alone, the way a model
'knows' facts. No network, no real data; results derived from it must be labelled
synthetic everywhere.
"""

from __future__ import annotations

import math
import re
from statistics import NormalDist

from sieve.eval._util import stable_int, stable_unit
from sieve.eval.datasets import MCQ_LETTERS, Item

SYNTHETIC_TAG = "SYNTHETIC"
_INDEX = re.compile(r"#(\d+)")
_DATASET = re.compile(r"\[" + SYNTHETIC_TAG + r":(\w+)\]")
_N_TOPICS = 5


def question_index(question: str) -> int:
    m = _INDEX.search(question)
    if m is None:
        raise ValueError("not a synthetic question")
    return int(m.group(1))


def question_dataset(question: str) -> str:
    m = _DATASET.search(question)
    if m is None:
        raise ValueError("not a synthetic question")
    return m.group(1)


def gold_text(i: int) -> str:
    return f"Gold{i}"


def gold_letter(i: int) -> str:
    return MCQ_LETTERS[stable_int("mcq", i) % 4]


def difficulty(dataset: str, i: int) -> float:
    """Latent difficulty ~ N(0, 1), plus a per-topic offset (so strata differ)."""
    u = min(max(stable_unit("diff", dataset, i), 1e-6), 1 - 1e-6)
    return NormalDist().inv_cdf(u) + 0.35 * ((i % _N_TOPICS) - 2) / 2.0


def dataset_shift(dataset: str) -> float:
    """Extra difficulty of shift sets relative to the primary one."""
    return {"triviaqa": 0.0, "nq_open": 0.35, "simpleqa_verified": 1.6, "mmlu_redux": 0.9}.get(
        dataset, 0.0
    )


def synthetic_items(dataset: str, n: int, seed: int) -> list[Item]:
    """``n`` fake items for ``dataset`` (ids are namespaced like the real ones)."""
    out: list[Item] = []
    for i in range(n):
        j = stable_int("idx", dataset, seed, i, bits=24)
        topic = f"topic{j % _N_TOPICS}"
        q = f"[{SYNTHETIC_TAG}:{dataset}] question #{j} about {topic}: which code is attached?"
        if dataset == "mmlu_redux":
            letter = gold_letter(j)
            choices = tuple(f"option {c} for #{j}" for c in MCQ_LETTERS)
            out.append(Item(f"{dataset}:{j}", dataset, q, (letter,), choices, topic))
        else:
            gold = gold_text(j)
            out.append(Item(f"{dataset}:{j}", dataset, q, (gold, f"gold {j}", f"G{j}"), (), topic))
    ids = {it.item_id for it in out}
    if len(ids) != len(out):  # extremely unlikely hash collision: make the failure loud
        raise RuntimeError("synthetic id collision; change the seed")
    return out


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

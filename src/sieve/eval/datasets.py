"""Dataset specs, loaders and deterministic sampling.

Real loaders use Hugging Face ``datasets`` with pinned revisions. Repos, configs, splits,
field names and licences were checked against the Hub on 2026-11-03:

========================  ==========================================  ==============
dataset                   repo / config / split                       licence
========================  ==========================================  ==============
TriviaQA rc.nocontext     mandarjoshi/trivia_qa / rc.nocontext /       unknown (card)
                          validation
NQ-open                   google-research-datasets/nq_open / nq_open   CC-BY-SA-3.0
                          / validation
SimpleQA Verified         google/simpleqa-verified / default / eval    MIT
MMLU-Redux 2.0            edinburgh-dawg/mmlu-redux-2.0 / 57 subject   CC-BY-4.0
                          configs / test
========================  ==========================================  ==============

``datasets`` is imported lazily so the package imports without the ``eval`` extra.
"""

from __future__ import annotations

import importlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sieve.eval._util import atomic_write_text, sha256_obj, sha256_text

MCQ_LETTERS = "ABCD"


@dataclass(frozen=True)
class Item:
    """One benchmark question with its gold answer(s)."""

    item_id: str
    dataset: str
    question: str
    answers: tuple[str, ...]  # gold + aliases; for MCQ the single gold letter
    choices: tuple[str, ...] = ()
    stratum: str = ""
    context: str = ""  # reading-comprehension passage (SQuAD); empty for closed-book datasets

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "item_id": self.item_id,
            "dataset": self.dataset,
            "question": self.question,
            "answers": list(self.answers),
            "choices": list(self.choices),
            "stratum": self.stratum,
        }
        if self.context:  # only when set, so every existing dataset hash is unchanged
            d["context"] = self.context
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Item:
        return Item(
            item_id=str(d["item_id"]),
            dataset=str(d["dataset"]),
            question=str(d["question"]),
            answers=tuple(str(a) for a in d["answers"]),
            choices=tuple(str(c) for c in d.get("choices", [])),
            stratum=str(d.get("stratum", "")),
            context=str(d.get("context", "")),
        )

    def prompt_view(self) -> PromptItem:
        """What the generation runner may see: no gold answers."""
        return PromptItem(self.item_id, self.dataset, self.question, self.choices, self.context)


@dataclass(frozen=True)
class PromptItem:
    item_id: str
    dataset: str
    question: str
    choices: tuple[str, ...] = ()
    context: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo: str
    config: str | None
    split: str
    revision: str
    licence: str
    n: int  # items to sample (prereg)
    seed: int
    role: str  # "primary" | "shift"
    grader: str  # "alias_em" | "mmlu_letter" | "simpleqa_judge"
    configs: tuple[str, ...] = ()  # multi-config datasets (MMLU-Redux subjects)


#: Pinned from the Hub API on 2026-11-03 (``sha`` field of /api/datasets/<repo>).
PINNED: dict[str, DatasetSpec] = {
    # v2 experiment (docs/PREREGISTRATION_V2.md): reading comprehension with unanswerable questions.
    "squad_v2": DatasetSpec(
        "squad_v2",
        "rajpurkar/squad_v2",
        "squad_v2",
        "validation",
        "3ffb306f725f7d2ce8394bc1873b24868140c412",
        "CC-BY-SA-4.0",
        6000,
        20261108,
        "primary",
        "alias_em",
    ),
    "triviaqa": DatasetSpec(
        "triviaqa",
        "mandarjoshi/trivia_qa",
        "rc.nocontext",
        "validation",
        "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
        "unknown (dataset card lists no licence; check upstream terms before redistribution)",
        6000,
        20261103,
        "primary",
        "alias_em",
    ),
    "nq_open": DatasetSpec(
        "nq_open",
        "google-research-datasets/nq_open",
        "nq_open",
        "validation",
        "5dd9790a83002ad084ddeb7c420dc716852c6f28",
        "CC-BY-SA-3.0",
        2000,
        20261104,
        "shift",
        "alias_em",
    ),
    "simpleqa_verified": DatasetSpec(
        "simpleqa_verified",
        "google/simpleqa-verified",
        None,
        "eval",
        "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc",
        "MIT",
        1000,
        20261105,
        "shift",
        "simpleqa_judge",
    ),
    "mmlu_redux": DatasetSpec(
        "mmlu_redux",
        "edinburgh-dawg/mmlu-redux-2.0",
        None,
        "test",
        "372ea425445d51e1ba1188c56e5e893f8138621f",
        "CC-BY-4.0",
        1500,
        20261106,
        "shift",
        "mmlu_letter",
    ),
}

MMLU_SUBJECTS: list[str] = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


class DatasetError(RuntimeError):
    pass


# --- loading ------------------------------------------------------------------------

#: ``(repo, config, split, revision) -> iterable of row dicts``; injectable for tests.
RowFetcher = Callable[[str, str | None, str, str], Iterable[dict[str, Any]]]


def hf_fetch(repo: str, config: str | None, split: str, revision: str) -> Iterable[dict[str, Any]]:
    """Fetch rows through Hugging Face ``datasets`` at a pinned revision."""
    try:
        hf = importlib.import_module("datasets")
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise DatasetError("install the eval extra: pip install 'sieve-llm[eval]'") from e
    args: list[str] = [repo] if config is None else [repo, config]
    ds = hf.load_dataset(*args, split=split, revision=revision)
    return [dict(r) for r in ds]


def _norm_q(q: str) -> str:
    q = unicodedata.normalize("NFKD", q).lower()
    return re.sub(r"\W+", " ", q).strip()


def parse_triviaqa(rows: Iterable[dict[str, Any]]) -> list[Item]:
    out: list[Item] = []
    for r in rows:
        ans = r["answer"]
        aliases = [str(a) for a in ans.get("aliases", [])]
        value = str(ans.get("value", ""))
        golds = tuple(dict.fromkeys([value, *aliases])) if value else tuple(aliases)
        if not golds:
            continue
        out.append(Item(f"triviaqa:{r['question_id']}", "triviaqa", str(r["question"]), golds))
    return out


def parse_squad_v2(rows: Iterable[dict[str, Any]]) -> list[Item]:
    """SQuAD 2.0: answerable items keep their gold spans; unanswerable items have the single gold
    string ``unanswerable`` (the model is told to say it), so one exact-match grader scores both.
    The stratum records which kind each item is, for per-stratum risk."""
    out: list[Item] = []
    for r in rows:
        texts = [str(t) for t in (r.get("answers") or {}).get("text", [])]
        if texts:
            golds, stratum = tuple(dict.fromkeys(texts)), "answerable"
        else:
            golds, stratum = ("unanswerable",), "unanswerable"
        out.append(
            Item(
                f"squad_v2:{r['id']}",
                "squad_v2",
                str(r["question"]),
                golds,
                stratum=stratum,
                context=str(r["context"]),
            )
        )
    return out


def parse_nq_open(rows: Iterable[dict[str, Any]]) -> list[Item]:
    out: list[Item] = []
    for i, r in enumerate(rows):
        golds = tuple(str(a) for a in r["answer"])
        if golds:
            out.append(Item(f"nq_open:{i:05d}", "nq_open", str(r["question"]), golds))
    return out


def parse_simpleqa_verified(rows: Iterable[dict[str, Any]]) -> list[Item]:
    return [
        Item(
            f"simpleqa_verified:{r['original_index']}",
            "simpleqa_verified",
            str(r["problem"]),
            (str(r["answer"]),),
            stratum=str(r.get("topic", "")),
        )
        for r in rows
    ]


def parse_mmlu_redux(rows_by_subject: dict[str, Iterable[dict[str, Any]]]) -> list[Item]:
    """Keep only rows annotated ``error_type == 'ok'`` (verified clean) with 4 choices."""
    out: list[Item] = []
    for subject, rows in rows_by_subject.items():
        n_before = len(out)
        rows = list(rows)
        for i, r in enumerate(rows):
            if str(r.get("error_type", "ok")) != "ok":
                continue
            choices = tuple(str(c) for c in r["choices"])
            idx = int(r["answer"])
            if len(choices) != 4 or not 0 <= idx < 4:
                continue
            out.append(
                Item(
                    f"mmlu_redux:{subject}:{i:04d}",
                    "mmlu_redux",
                    str(r["question"]),
                    (MCQ_LETTERS[idx],),
                    choices=choices,
                    stratum=subject,
                )
            )
        if rows and len(out) == n_before:
            raise DatasetError(
                f"MMLU-Redux subject {subject!r}: no row with error_type == 'ok'; "
                "the annotation vocabulary may have changed"
            )
    return out


def load_dataset_items(spec: DatasetSpec, fetch: RowFetcher = hf_fetch) -> list[Item]:
    """Load *all* eligible items of a dataset (before sampling)."""
    if spec.name == "triviaqa":
        return parse_triviaqa(fetch(spec.repo, spec.config, spec.split, spec.revision))
    if spec.name == "squad_v2":
        return parse_squad_v2(fetch(spec.repo, spec.config, spec.split, spec.revision))
    if spec.name == "nq_open":
        return parse_nq_open(fetch(spec.repo, spec.config, spec.split, spec.revision))
    if spec.name == "simpleqa_verified":
        return parse_simpleqa_verified(fetch(spec.repo, None, spec.split, spec.revision))
    if spec.name == "mmlu_redux":
        subjects = spec.configs or MMLU_SUBJECTS
        return parse_mmlu_redux(
            {s: fetch(spec.repo, s, spec.split, spec.revision) for s in subjects}
        )
    raise DatasetError(f"unknown dataset {spec.name!r}")


# --- deterministic sampling ---------------------------------------------------------


def dedupe_questions(items: Sequence[Item]) -> list[Item]:
    """Drop items whose normalised question text repeats (keeps the smallest item_id)."""
    seen: set[str] = set()
    out: list[Item] = []
    for it in sorted(items, key=lambda i: i.item_id):
        key = _norm_q(it.question)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def sample_items(items: Sequence[Item], n: int, seed: int) -> list[Item]:
    """Seeded, order-independent sample: the ``n`` items with the smallest seeded hash.

    Independent of the order the loader returned rows in, so a re-ordering of the
    upstream file cannot change the sample.
    """
    uniq = dedupe_questions(items)
    if n > len(uniq):
        raise DatasetError(f"requested {n} items but only {len(uniq)} unique are available")
    ranked = sorted(uniq, key=lambda it: sha256_text(f"{seed}|{it.item_id}"))
    return sorted(ranked[:n], key=lambda it: it.item_id)


def dataset_hash(items: Sequence[Item]) -> str:
    return sha256_obj([it.to_dict() for it in sorted(items, key=lambda i: i.item_id)])


# --- on-disk item files -------------------------------------------------------------


def write_items(path: Path, items: Sequence[Item], meta: dict[str, Any]) -> str:
    """Write ``<name>.jsonl`` (items) and ``<name>.meta.json``; returns the dataset hash."""
    h = dataset_hash(items)
    lines = [
        json.dumps(it.to_dict(), sort_keys=True) for it in sorted(items, key=lambda i: i.item_id)
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")
    atomic_write_text(
        path.with_suffix(".meta.json"),
        json.dumps({**meta, "n": len(items), "dataset_hash": h}, sort_keys=True, indent=1),
    )
    return h


def read_items(path: Path, *, verify: bool = True) -> list[Item]:
    items = [
        Item.from_dict(json.loads(ln)) for ln in path.read_text(encoding="utf-8").splitlines() if ln
    ]
    meta_path = path.with_suffix(".meta.json")
    if verify and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if dataset_hash(items) != meta["dataset_hash"]:
            raise DatasetError(f"{path}: content does not match its recorded dataset hash")
    return items


@dataclass(frozen=True)
class ItemIndex:
    """All items by id (prompts + gold); gold access is by id list from a split view."""

    by_id: dict[str, Item] = field(default_factory=dict)

    @staticmethod
    def from_items(items: Iterable[Item]) -> ItemIndex:
        return ItemIndex({it.item_id: it for it in items})

    def get(self, ids: Iterable[str]) -> list[Item]:
        return [self.by_id[i] for i in ids]

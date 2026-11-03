"""Append-only, content-hashed, resumable store of raw generations.

One JSONL shard per cell (a cell = one model/quant/KV/sampling configuration). Every
attempted generation is a line, including failed attempts (their wall time is wasted time
that the cost accounting must charge). Each line carries the SHA-256 of its own canonical
content; a hash mismatch anywhere except a torn final line is a hard error.

Resume rules: a (item, cell, kind, sample) key is *done* once a line with ``final=true``
exists. A killed process may leave a partial last line; it is truncated on open. Re-running
never regenerates a done key (idempotent).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sieve.eval._util import sha256_obj
from sieve.types import BackendInfo, Completion, SamplingParams, TokenLP

SCHEMA = 1


class CacheCorruptionError(RuntimeError):
    pass


# --- non-finite floats survive canonical JSON as strings ---------------------------


def _enc(x: float) -> float | str:
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return x


def _dec(x: float | str) -> float:
    return float(x)  # float("NaN"), float("-Infinity") and numbers all parse


def encode_tokens(
    tokens: tuple[TokenLP, ...] | None,
) -> list[list[Any]] | None:
    if tokens is None:
        return None
    return [
        [
            t.token,
            _enc(t.logprob),
            None if t.top is None else [[a, _enc(b)] for a, b in t.top],
        ]
        for t in tokens
    ]


def decode_tokens(raw: list[list[Any]] | None) -> tuple[TokenLP, ...] | None:
    if raw is None:
        return None
    return tuple(
        TokenLP(
            str(tok),
            _dec(lp),
            None if top is None else tuple((str(a), _dec(b)) for a, b in top),
        )
        for tok, lp, top in raw
    )


def sampling_to_dict(s: SamplingParams) -> dict[str, Any]:
    return {
        "temperature": s.temperature,
        "top_p": s.top_p,
        "top_k": s.top_k,
        "min_p": s.min_p,
        "logit_bias": [[k, v] for k, v in s.logit_bias],
        "grammar": s.grammar,
        "post_sampling_probs": s.post_sampling_probs,
        "seed": s.seed,
        "max_tokens": s.max_tokens,
    }


def sampling_from_dict(d: dict[str, Any]) -> SamplingParams:
    return SamplingParams(
        temperature=float(d["temperature"]),
        top_p=float(d["top_p"]),
        top_k=int(d["top_k"]),
        min_p=float(d["min_p"]),
        logit_bias=tuple((int(k), float(v)) for k, v in d["logit_bias"]),
        grammar=d["grammar"],
        post_sampling_probs=bool(d["post_sampling_probs"]),
        seed=d["seed"],
        max_tokens=int(d["max_tokens"]),
    )


def backend_to_dict(b: BackendInfo) -> dict[str, str]:
    return {
        "model_id": b.model_id,
        "model_hash": b.model_hash,
        "quant": b.quant,
        "kv_quant": b.kv_quant,
        "build": b.build,
        "template_hash": b.template_hash,
    }


@dataclass(frozen=True)
class Record:
    """One attempted generation."""

    item_id: str
    cell_id: str
    kind: str  # "primary" | "sample"
    sample_idx: int
    attempt: int
    final: bool
    ok: bool
    error: str | None
    text: str
    tokens: tuple[TokenLP, ...] | None
    finish: str
    n_gen: int
    n_prompt: int
    answer_start: int
    t_first_token_s: float
    t_total_s: float  # backend-reported
    wall_s: float  # measured by the runner around the call
    backend: dict[str, str]
    sampling: SamplingParams
    config_hash: str
    prereg_hash: str

    @property
    def key(self) -> str:
        return f"{self.item_id}|{self.kind}|{self.sample_idx}"

    def body(self) -> dict[str, Any]:
        return {
            "v": SCHEMA,
            "item_id": self.item_id,
            "cell_id": self.cell_id,
            "kind": self.kind,
            "sample_idx": self.sample_idx,
            "attempt": self.attempt,
            "final": self.final,
            "ok": self.ok,
            "error": self.error,
            "text": self.text,
            "tokens": encode_tokens(self.tokens),
            "finish": self.finish,
            "n_gen": self.n_gen,
            "n_prompt": self.n_prompt,
            "answer_start": self.answer_start,
            "t_first_token_s": self.t_first_token_s,
            "t_total_s": self.t_total_s,
            "wall_s": self.wall_s,
            "backend": self.backend,
            "sampling": sampling_to_dict(self.sampling),
            "config_hash": self.config_hash,
            "prereg_hash": self.prereg_hash,
        }

    def to_line(self) -> str:
        body = self.body()
        return json.dumps({**body, "hash": sha256_obj(body)}, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_line(line: str) -> Record:
        d = json.loads(line)
        h = d.pop("hash")
        if sha256_obj(d) != h:
            raise CacheCorruptionError("record hash mismatch")
        return Record(
            item_id=d["item_id"],
            cell_id=d["cell_id"],
            kind=d["kind"],
            sample_idx=int(d["sample_idx"]),
            attempt=int(d["attempt"]),
            final=bool(d["final"]),
            ok=bool(d["ok"]),
            error=d["error"],
            text=d["text"],
            tokens=decode_tokens(d["tokens"]),
            finish=d["finish"],
            n_gen=int(d["n_gen"]),
            n_prompt=int(d["n_prompt"]),
            answer_start=int(d["answer_start"]),
            t_first_token_s=float(d["t_first_token_s"]),
            t_total_s=float(d["t_total_s"]),
            wall_s=float(d["wall_s"]),
            backend=dict(d["backend"]),
            sampling=sampling_from_dict(d["sampling"]),
            config_hash=d["config_hash"],
            prereg_hash=d["prereg_hash"],
        )

    def to_completion(self) -> Completion:
        finish: Any = self.finish if self.finish in ("stop", "length", "error") else "error"
        return Completion(
            text=self.text,
            tokens=self.tokens,
            finish=finish,
            n_gen=self.n_gen,
            sampling=self.sampling,
            answer_start=self.answer_start,
            n_prompt=self.n_prompt,
            t_first_token_s=self.t_first_token_s,
            t_total_s=self.t_total_s,
        )

    def backend_info(self) -> BackendInfo:
        return BackendInfo(**self.backend)


class CellShard:
    """Append-only shard for one cell."""

    def __init__(self, path: Path, *, fsync_every: int = 64) -> None:
        self.path = path
        self._fsync_every = fsync_every
        self._since_sync = 0
        self._attempts: dict[str, list[Record]] = {}
        self._load()
        self._fh = path.open("a", encoding="utf-8")

    def _load(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        parts = data.split(b"\n")
        tail = parts[-1]  # bytes after the last newline: empty, or a torn/unterminated record
        for i, raw in enumerate(parts[:-1]):
            if raw.strip() == b"":
                continue
            try:
                rec = Record.from_line(raw.decode("utf-8"))
            except (ValueError, KeyError, UnicodeDecodeError) as e:
                raise CacheCorruptionError(f"{self.path}: bad line {i + 1}: {e}") from e
            self._attempts.setdefault(rec.key, []).append(rec)
        good_end = len(data) - len(tail)
        if tail == b"":
            return
        try:
            rec = Record.from_line(tail.decode("utf-8"))
        except (ValueError, KeyError, UnicodeDecodeError):
            with self.path.open("r+b") as fh:  # killed mid-write: drop the fragment
                fh.truncate(good_end)
            return
        self._attempts.setdefault(rec.key, []).append(rec)
        with self.path.open("ab") as fh:  # complete record, only the newline was lost
            fh.write(b"\n")

    def close(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()

    def append(self, rec: Record) -> None:
        self._fh.write(rec.to_line() + "\n")
        self._fh.flush()  # survives SIGKILL of this process (OS holds the page)
        self._since_sync += 1
        if self._since_sync >= self._fsync_every:
            os.fsync(self._fh.fileno())
            self._since_sync = 0
        self._attempts.setdefault(rec.key, []).append(rec)

    def is_done(self, key: str) -> bool:
        return any(r.final for r in self._attempts.get(key, ()))

    def next_attempt(self, key: str) -> int:
        a = self._attempts.get(key)
        return 0 if not a else max(r.attempt for r in a) + 1

    def final(self, key: str) -> Record | None:
        for r in self._attempts.get(key, ()):
            if r.final:
                return r
        return None

    def wasted_wall_s(self, key: str) -> float:
        """Wall time of attempts that produced no usable output (failed attempts)."""
        return sum(r.wall_s for r in self._attempts.get(key, ()) if not r.ok)

    def total_wall_s(self, key: str) -> float:
        """Wall time of every attempt for ``key``, wasted ones included."""
        return sum(r.wall_s for r in self._attempts.get(key, ()))

    def attempts(self, key: str) -> list[Record]:
        return list(self._attempts.get(key, ()))

    def item_keys(self) -> Iterator[str]:
        return iter(self._attempts)

    def all_records(self) -> Iterator[Record]:
        for recs in self._attempts.values():
            yield from recs

    def n_done(self) -> int:
        return sum(1 for k in self._attempts if self.is_done(k))


class GenerationCache:
    """Directory of cell shards."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._shards: dict[str, CellShard] = {}

    def shard(self, cell_id: str) -> CellShard:
        if cell_id not in self._shards:
            self._shards[cell_id] = CellShard(self.root / f"{cell_id}.jsonl")
        return self._shards[cell_id]

    def close(self) -> None:
        for s in self._shards.values():
            s.close()
        self._shards.clear()

    def cell_ids_on_disk(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.jsonl"))


def read_shard_readonly(path: Path) -> CellShard:
    """Open for analysis (still verifies hashes; truncates nothing that was whole)."""
    return CellShard(path)

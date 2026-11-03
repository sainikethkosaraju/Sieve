"""Hash-locked A/B/C re-splits and the leakage guard around split B.

Protocol (docs/PREREGISTRATION.md): 6000 TriviaQA items, 50 random re-splits into
A (2000: calibrators, router training, cost estimation), B (2000: threshold
certification, used once) and C (2000: test).

Enforcement is by API and an append-only access ledger, not by operating-system
permissions. This is a DETECTION control (a tripwire against accidents and casual
double-looks), not access control: see "B-once / C-hash discipline" in docs/evaluation.md for
the known bypasses.

* ``SplitStore.a_ids`` is the only freely readable split (design/tuning view);
* ``SplitStore.open_b`` needs the certification capability, which only
  ``sieve.eval.certify`` holds, and succeeds once per split (the ledger remembers);
* ``SplitStore.open_c`` needs the hash of a certification produced from B and is once-only
  per certification (a re-open must be explicit and is recorded as ``reopen_c``);
* ``SplitStore.verify`` recomputes every split from the seed and checks all hashes, so a
  hand-edited split file is detected.

A static test pins that no other module calls ``open_b``.
"""

from __future__ import annotations

import fcntl
import functools
import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sieve.eval._util import atomic_write_text, git_commit, sha256_obj, sha256_text

SPLIT_NAMES = ("A", "B", "C")
DEFAULT_SIZES = (2000, 2000, 2000)
DEFAULT_N_SPLITS = 50
INDEX_NAME = "splits_index.json"
LEDGER_NAME = "split_access_ledger.jsonl"


class SplitIntegrityError(RuntimeError):
    """A split file or the index does not match its recorded hash or re-derivation."""


class SplitAccessError(RuntimeError):
    """An attempt to read split B or C outside the allowed protocol."""


class SplitBAlreadyReadError(SplitAccessError):
    """Split B of this split was already opened; it may be used once."""


class SplitCAlreadyReadError(SplitAccessError):
    """Split C was already opened for this certification; a re-open must be explicit."""


@functools.lru_cache(maxsize=1)
def code_version() -> str:
    """Code version stamped on every ledger event (``SIEVE_GIT_COMMIT`` or git, +dirty)."""
    return git_commit()


class _Capability:
    """Opaque token; ``sieve.eval.certify`` is the only module that may import it."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"<capability {self._name}>"


#: Held by ``sieve.eval.certify`` only (checked by a static test).
CERTIFICATION_CAPABILITY = _Capability("certification")


def ids_hash(ids: Sequence[str]) -> str:
    return sha256_text("\n".join(ids))


def derive_split(
    master_ids: Sequence[str],
    k: int,
    split_seed: int,
    sizes: tuple[int, int, int] = DEFAULT_SIZES,
) -> dict[str, tuple[str, ...]]:
    """Deterministic re-split ``k``: rank items by a seeded hash, cut into A/B/C."""
    if len(set(master_ids)) != len(master_ids):
        raise ValueError("duplicate item ids")
    if len(master_ids) != sum(sizes):
        raise ValueError(f"need {sum(sizes)} items, got {len(master_ids)}")
    ranked = sorted(master_ids, key=lambda i: sha256_text(f"{split_seed}|{k}|{i}"))
    a, b = sizes[0], sizes[0] + sizes[1]
    return {"A": tuple(ranked[:a]), "B": tuple(ranked[a:b]), "C": tuple(ranked[b:])}


@dataclass(frozen=True)
class AccessLedger:
    """Append-only JSONL log of B/C accesses.

    Every append is atomic: the file is rewritten to a temp file, fsynced and moved over the
    ledger with ``os.replace`` under an exclusive lock, so a killed process never leaves a torn
    line and two processes cannot both pass a check-then-append. Each event records the code
    version and (when known) the config hash.
    """

    path: Path

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with (self.path.parent / (self.path.name + ".lock")).open("a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def _write_with(self, line: str) -> None:
        old = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        if old and not old.endswith("\n"):
            old += "\n"
        tmp = self.path.with_name(self.path.name + f".tmp{os.getpid()}")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(old + line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    @staticmethod
    def _stamp(event: dict[str, Any], config_hash: str) -> str:
        return json.dumps(
            {**event, "code_version": code_version(), "config_hash": config_hash},
            sort_keys=True,
        )

    def append(self, event: dict[str, Any], *, config_hash: str = "") -> None:
        with self._locked():
            self._write_with(self._stamp(event, config_hash))

    def append_unique(
        self, event: dict[str, Any], keys: Sequence[str], *, config_hash: str = ""
    ) -> bool:
        """Append unless an event with the same ``keys`` values exists; atomic. True if added."""
        with self._locked():
            for e in self.events():
                if all(e.get(k) == event.get(k) for k in keys):
                    return False
            self._write_with(self._stamp(event, config_hash))
            return True

    def has(self, event: str, index_hash: str, k: int) -> bool:
        return any(
            e["event"] == event and e["index_hash"] == index_hash and e["k"] == k
            for e in self.events()
        )

    def certification_hashes(self, index_hash: str, k: int) -> set[str]:
        return {
            str(e["cert_hash"])
            for e in self.events()
            if e["event"] == "certified" and e["index_hash"] == index_hash and e["k"] == k
        }

    def c_opened(self, index_hash: str, k: int, cert_hash: str) -> bool:
        return any(
            e["event"] == "open_c"
            and e["index_hash"] == index_hash
            and e["k"] == k
            and e.get("cert_hash") == cert_hash
            for e in self.events()
        )


def build_splits(
    directory: Path,
    master_ids: Sequence[str],
    *,
    dataset_hash: str,
    split_seed: int,
    n_splits: int = DEFAULT_N_SPLITS,
    sizes: tuple[int, int, int] = DEFAULT_SIZES,
) -> SplitStore:
    """Write ``split_NNN.json`` files and the hash index; refuse to overwrite a different one."""
    master = tuple(sorted(master_ids))
    entries: list[dict[str, Any]] = []
    payloads: dict[int, dict[str, Any]] = {}
    for k in range(n_splits):
        parts = derive_split(master, k, split_seed, sizes)
        hashes = {n: ids_hash(parts[n]) for n in SPLIT_NAMES}
        entries.append({"k": k, **hashes})
        payloads[k] = {"k": k, "ids": {n: list(parts[n]) for n in SPLIT_NAMES}, "hashes": hashes}
    body: dict[str, Any] = {
        "schema": 1,
        "dataset_hash": dataset_hash,
        "master_hash": ids_hash(master),
        "split_seed": split_seed,
        "sizes": list(sizes),
        "n_splits": n_splits,
        "splits": entries,
    }
    index = {**body, "index_hash": sha256_obj(body)}
    idx_path = directory / INDEX_NAME
    if idx_path.exists():
        old = json.loads(idx_path.read_text(encoding="utf-8"))
        if old.get("index_hash") != index["index_hash"]:
            raise SplitIntegrityError(
                f"{idx_path} already exists with a different hash; splits are frozen"
            )
    else:
        atomic_write_text(idx_path, json.dumps(index, sort_keys=True, indent=1))
    for k, payload in payloads.items():
        p = directory / f"split_{k:03d}.json"
        if not p.exists():
            atomic_write_text(p, json.dumps(payload, sort_keys=True))
    return SplitStore.load(directory)


class SplitStore:
    """Verified view over a split directory, with the B/C access rules."""

    def __init__(self, directory: Path, index: dict[str, Any]) -> None:
        self.directory = directory
        self._index = index
        self.ledger = AccessLedger(directory / LEDGER_NAME)
        self._cache: dict[int, dict[str, tuple[str, ...]]] = {}
        self._c_memo: set[tuple[int, str]] = set()

    # --- construction / integrity ------------------------------------------------

    @classmethod
    def load(cls, directory: Path, *, verify: bool = True) -> SplitStore:
        idx_path = directory / INDEX_NAME
        if not idx_path.exists():
            raise SplitIntegrityError(f"missing {idx_path}")
        index = json.loads(idx_path.read_text(encoding="utf-8"))
        body = {k: v for k, v in index.items() if k != "index_hash"}
        if sha256_obj(body) != index.get("index_hash"):
            raise SplitIntegrityError("index hash mismatch")
        store = cls(directory, index)
        if verify:
            store.verify()
        return store

    @property
    def index_hash(self) -> str:
        return str(self._index["index_hash"])

    @property
    def dataset_hash(self) -> str:
        return str(self._index["dataset_hash"])

    @property
    def n_splits(self) -> int:
        return int(self._index["n_splits"])

    @property
    def sizes(self) -> tuple[int, int, int]:
        s = self._index["sizes"]
        return (int(s[0]), int(s[1]), int(s[2]))

    def _read(self, k: int) -> dict[str, tuple[str, ...]]:
        if k in self._cache:
            return self._cache[k]
        if not 0 <= k < self.n_splits:
            raise KeyError(f"no split {k}")
        payload = json.loads((self.directory / f"split_{k:03d}.json").read_text(encoding="utf-8"))
        parts = {n: tuple(payload["ids"][n]) for n in SPLIT_NAMES}
        want = self._index["splits"][k]
        for n in SPLIT_NAMES:
            if ids_hash(parts[n]) != want[n]:
                raise SplitIntegrityError(f"split {k} part {n} does not match the index hash")
        self._cache[k] = parts
        return parts

    def verify(self) -> None:
        """Re-derive every split from the seed and compare with files and index."""
        master: set[str] = set()
        for n in SPLIT_NAMES:
            master.update(self._read(0)[n])
        master_sorted = tuple(sorted(master))
        if ids_hash(master_sorted) != self._index["master_hash"]:
            raise SplitIntegrityError("master id list does not match the index")
        for k in range(self.n_splits):
            parts = self._read(k)
            union = [i for n in SPLIT_NAMES for i in parts[n]]
            if len(set(union)) != len(union) or set(union) != master:
                raise SplitIntegrityError(f"split {k}: parts overlap or miss items")
            derived = derive_split(master_sorted, k, int(self._index["split_seed"]), self.sizes)
            if derived != parts:
                raise SplitIntegrityError(f"split {k} differs from its seeded derivation")

    # --- access rules -------------------------------------------------------------

    def master_ids(self) -> tuple[str, ...]:
        parts = self._read(0)
        return tuple(sorted(i for n in SPLIT_NAMES for i in parts[n]))

    def a_ids(self, k: int) -> tuple[str, ...]:
        """Split A: free to read (calibrators, router training, grids, cost)."""
        return self._read(k)["A"]

    def open_b(self, k: int, *, capability: object, config_hash: str = "") -> tuple[str, ...]:
        """Split B, once, for certification code only."""
        if capability is not CERTIFICATION_CAPABILITY:
            raise SplitAccessError("split B is readable only by certification code")
        added = self.ledger.append_unique(
            {"event": "open_b", "index_hash": self.index_hash, "k": k},
            ("event", "index_hash", "k"),
            config_hash=config_hash,
        )
        if not added:
            raise SplitBAlreadyReadError(
                f"split B of re-split {k} was already read; it may be used once. "
                "Reuse the stored certification or record a Deviation."
            )
        return self._read(k)["B"]

    def record_certification(
        self, k: int, cert_hash: str, *, capability: object, config_hash: str = ""
    ) -> None:
        if capability is not CERTIFICATION_CAPABILITY:
            raise SplitAccessError("only certification code may record a certification")
        if not self.ledger.has("open_b", self.index_hash, k):
            raise SplitAccessError("cannot certify a split whose B was never opened")
        self.ledger.append(
            {"event": "certified", "index_hash": self.index_hash, "k": k, "cert_hash": cert_hash},
            config_hash=config_hash,
        )

    def open_c(
        self, k: int, *, cert_hash: str, config_hash: str = "", reopen: bool = False
    ) -> tuple[str, ...]:
        """Split C (test), only after a certification of this split exists in the ledger.

        Once-only per certification: a second open (new process or store object) raises
        :class:`SplitCAlreadyReadError` unless ``reopen=True``, which is recorded as
        ``reopen_c``. Repeated calls on the same store object are served from memory.
        """
        if cert_hash not in self.ledger.certification_hashes(self.index_hash, k):
            raise SplitAccessError(
                f"split C of re-split {k} is readable only with the hash of a certification "
                "made from its split B"
            )
        if (k, cert_hash) in self._c_memo:
            return self._read(k)["C"]
        ev = {"index_hash": self.index_hash, "k": k, "cert_hash": cert_hash}
        added = self.ledger.append_unique(
            {"event": "open_c", **ev},
            ("event", "index_hash", "k", "cert_hash"),
            config_hash=config_hash,
        )
        if not added:
            if not reopen:
                raise SplitCAlreadyReadError(
                    f"split C of re-split {k} was already opened for this certification; "
                    "pass reopen=True to re-open it (recorded as reopen_c)"
                )
            self.ledger.append({"event": "reopen_c", **ev}, config_hash=config_hash)
        self._c_memo.add((k, cert_hash))
        return self._read(k)["C"]

    def b_state(self, k: int) -> str:
        """``unread`` | ``certified`` | ``opened_uncertified`` (the last voids the split)."""
        if not self.ledger.has("open_b", self.index_hash, k):
            return "unread"
        return (
            "certified"
            if self.ledger.certification_hashes(self.index_hash, k)
            else ("opened_uncertified")
        )

"""Small shared helpers: canonical JSON, hashing, deterministic integers, provenance."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no spaces, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_int(*parts: object, bits: int = 31) -> int:
    """Deterministic non-negative integer from ``parts`` (not Python's salted ``hash``)."""
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (1 << bits)


def stable_unit(*parts: object) -> float:
    """Deterministic float in [0, 1) from ``parts``."""
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def git_commit(cwd: Path | None = None) -> str:
    """Current commit (``+dirty`` suffix when the tree is modified) or ``unknown``.

    ``SIEVE_GIT_COMMIT`` overrides (used when the code ships without ``.git``, e.g. the embedded
    Kaggle kernel script, which records the commit it was built from).
    """
    forced = os.environ.get("SIEVE_GIT_COMMIT")
    if forced:
        return forced
    where = str(cwd or repo_root())
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=where,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=where,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return head + ("+dirty" if dirty else "")


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + rename so a killed process never leaves a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to Python and non-finite floats to None."""
    import math

    import numpy as np

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj

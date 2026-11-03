#!/usr/bin/env python3
"""Rebuild the result bundle that ``run_experiment.py`` printed into a Kaggle log.

    python kaggle/decode_log_bundle.py kaggle_out/<run>/log.txt --out kaggle_out/<run>/bundle

Verifies the chunk count and the sha256 of the base64 text, writes ``<name>.tar.gz`` and
extracts it. Also prints the human-readable summary between SIEVE_SUMMARY_BEGIN/END.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import re
import sys
import tarfile
from pathlib import Path

BEGIN = re.compile(
    r"SIEVE_BUNDLE_BEGIN name=(\S+) chunks=(\d+) b64_chars=(\d+) sha256=([0-9a-f]{64})"
)
CHUNK = re.compile(r"SIEVE_CHUNK (\S+) (\d+)/(\d+) (\S*)")


def decode(log_text: str) -> dict[str, bytes]:
    """All complete bundles in a log, by name; raises ``ValueError`` on a corrupt one."""
    heads = {m.group(1): m for m in BEGIN.finditer(log_text)}
    parts: dict[str, dict[int, str]] = {}
    for m in CHUNK.finditer(log_text):
        parts.setdefault(m.group(1), {})[int(m.group(2))] = m.group(4)
    out: dict[str, bytes] = {}
    for name, h in heads.items():
        n = int(h.group(2))
        got = parts.get(name, {})
        missing = [i for i in range(1, n + 1) if i not in got]
        if missing:
            raise ValueError(f"bundle {name}: missing chunks {missing[:10]} of {n}")
        b64 = "".join(got[i] for i in range(1, n + 1))
        if hashlib.sha256(b64.encode()).hexdigest() != h.group(4) or len(b64) != int(h.group(3)):
            raise ValueError(f"bundle {name}: checksum mismatch")
        out[name] = base64.b64decode(b64)
    return out


def summary(log_text: str) -> str:
    m = re.search(r"SIEVE_SUMMARY_BEGIN\n(.*?)\nSIEVE_SUMMARY_END", log_text, re.S)
    return m.group(1) if m else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default="bundle")
    a = ap.parse_args()
    text = Path(a.log).read_text(encoding="utf-8", errors="replace")
    bundles = decode(text)
    if not bundles:
        print("no SIEVE bundle in this log", file=sys.stderr)
        return 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, blob in bundles.items():
        (out / f"{name}.tar.gz").write_bytes(blob)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            tf.extractall(out / name, filter="data")
        print(f"bundle {name}: {len(blob)} bytes -> {out / name}")
    print(summary(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

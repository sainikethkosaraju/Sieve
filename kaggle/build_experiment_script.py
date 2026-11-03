#!/usr/bin/env python3
"""Generate ``kaggle/run_experiment.py`` from ``kaggle/experiment_driver.py``.

The pushed Kaggle script must be a single self-contained file, so the repository (src/,
benchmarks/, pyproject.toml, docs/PREREGISTRATION*.md, README.md, LICENSE) is embedded as a
deterministic base64 tar.gz and the run parameters are baked in. Regenerate after any change:

    python kaggle/build_experiment_script.py --config t4_integration --max-minutes 40

Examples:
    --config t4_main --max-minutes 540 --stages bootstrap,binary,models,prepare,run,analyse,report,sanity,emit
    --config t4_integration --cells S_q4km,L_q4km          # a subset of cells
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = (
    "src",
    "benchmarks",
    "pyproject.toml",
    "docs/PREREGISTRATION.md",
    "docs/PREREGISTRATION_V2.md",
    "README.md",
    "LICENSE",
)
DEFAULT_STAGES = "bootstrap,binary,models,probe,prepare,run,analyse,report,sanity,emit"
LLAMA_COMMIT = "fac3a894fc04901c3e5abaff457e024c6b52e04f"  # D-013 build


def _skip(p: Path) -> bool:
    return "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo") or p.name == ".DS_Store"


def repo_files() -> list[Path]:
    out: list[Path] = []
    for item in INCLUDE:
        p = ROOT / item
        if p.is_dir():
            out += [f for f in sorted(p.rglob("*")) if f.is_file() and not _skip(f)]
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"missing {item}")
    return out


def make_tgz(files: list[Path]) -> bytes:
    """Deterministic: sorted names, zeroed mtime/owner, normalised modes, gzip mtime 0."""
    bio = io.BytesIO()
    with gzip.GzipFile(fileobj=bio, mode="wb", mtime=0, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf:
            for f in sorted(files, key=lambda x: x.relative_to(ROOT).as_posix()):
                data = f.read_bytes()
                ti = tarfile.TarInfo(f.relative_to(ROOT).as_posix())
                ti.size, ti.mtime, ti.mode, ti.uid, ti.gid = len(data), 0, 0o644, 0, 0
                ti.uname = ti.gname = ""
                tf.addfile(ti, io.BytesIO(data))
    return bio.getvalue()


def git_state() -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *INCLUDE],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return head + ("+dirty" if dirty else "")


def build(args: argparse.Namespace) -> str:
    tgz = make_tgz(repo_files())
    b64 = base64.b64encode(tgz).decode()
    wrapped = "\n".join(b64[i : i + 100] for i in range(0, len(b64), 100))
    params = {
        "config": args.config,
        "workdir_name": args.workdir_name or f"sieve_work_{args.config}",
        "cells": args.cells.split(",") if args.cells else None,
        "stages": args.stages.split(","),
        "max_minutes": args.max_minutes,
        "reserve_minutes": args.reserve_minutes,
        "llama_cpp_commit": LLAMA_COMMIT,
        "llama_slug_hint": "sieve-llamacpp-t4-smoke",
        "repo_commit": git_state(),
        "embedded_sha256": hashlib.sha256(tgz).hexdigest(),
        "sanity_per_dataset": args.sanity_per_dataset,
        "allow_build": bool(args.allow_build),
    }
    src = (ROOT / "kaggle" / "experiment_driver.py").read_text(encoding="utf-8")
    if '"@@PARAMS@@"' not in src or '"@@EMBED@@"' not in src:
        raise SystemExit("driver template placeholders missing")
    pj = json.dumps(params, sort_keys=True)
    assert "'''" not in pj
    out = src.replace('"@@PARAMS@@"', "r'''" + pj + "'''", 1)
    out = out.replace('"@@EMBED@@"', '"""\n' + wrapped + '\n"""', 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="t4_integration", help="benchmarks/<name>.yaml")
    ap.add_argument("--workdir-name", help="default: sieve_work_<config>")
    ap.add_argument("--cells", help="comma-separated cell names (default: all cells)")
    ap.add_argument("--stages", default=DEFAULT_STAGES)
    ap.add_argument("--max-minutes", type=float, default=40.0)
    ap.add_argument("--reserve-minutes", type=float, default=8.0)
    ap.add_argument("--sanity-per-dataset", type=int, default=6)
    ap.add_argument("--allow-build", action="store_true", help="compile llama.cpp if no cached binary")
    ap.add_argument("--out", default=str(ROOT / "kaggle" / "run_experiment.py"))
    a = ap.parse_args()
    text = build(a)
    Path(a.out).write_text(text, encoding="utf-8")
    compile(text, a.out, "exec")  # syntax check of the generated script
    print(
        f"wrote {a.out}: {len(text)} bytes, config={a.config}, stages={a.stages}", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

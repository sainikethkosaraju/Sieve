"""The eval runner end to end through the llama.cpp provider and a stand-in llama-server.

Synthetic items, canned answers: this tests plumbing (process start, identity in every cached
record, stop-trimmed tokens, resumability, ABAB-resident S and L servers on pinned GPUs), not
any model.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from sieve.eval import pipeline, prereg
from sieve.eval.cache import Record
from sieve.eval.config import parse_config
from sieve.eval.providers import LlamaCppProvider
from sieve.signals import validate_completion

FAKE = Path(__file__).with_name("fake_llama_server.py")


def tiny_cfg() -> dict[str, Any]:
    d = pipeline.mock_config(small=True)
    d["synthetic"] = False
    d["name"] = "integration-fake-server"
    d["splits"] = {"n_splits": 3, "sizes": [12, 12, 12], "seed": 5}
    d["datasets"] = {
        "triviaqa": {"n": 36, "seed": 1},
        "nq_open": {"n": 8, "seed": 2},
        "simpleqa_verified": {"n": 8, "seed": 3},
        "mmlu_redux": {"n": 8, "seed": 4},
    }
    d["self_consistency"]["n_items"] = 3
    d["nondeterminism"] = {"items": 3, "runs": 2}
    d["energy"] = {"abab_blocks": 1, "idle_seconds": 0.0}
    keep = {"S_q4km", "L_q4km"}
    d["cells"] = [c for c in d["cells"] if c["name"] in keep]
    for c in d["cells"]:
        c["extra"]["gguf"] = f"o/r:{c['name']}.gguf"
    d["pairs"] = [p for p in d["pairs"] if p["role"] == "main"]
    d["abab_groups"] = [["S_q4km", "L_q4km"]]
    d["preregistration"]["frozen_sha256"] = prereg.frozen_hash(d["preregistration"]["path"])
    d["backend"] = {
        "provider": "sieve.eval.providers.llamacpp:make_provider",
        "llama_cpp_commit": "fac3a894fc04901c3e5abaff457e024c6b52e04f",
        "startup_timeout_s": 30,
        "gpu": {"S": "0", "L": "1"},
    }
    return d


@pytest.mark.slow
def test_runner_through_provider_is_correct_and_resumable(tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    models = tmp_path / "models"
    models.mkdir()
    for n in ("S_q4km", "L_q4km"):
        (models / f"{n}.gguf").write_bytes(b"GGUF" + n.encode())
    cfg = parse_config(tiny_cfg())
    wd = tmp_path / "work"
    pipeline.prepare_data(cfg, wd, synthetic=True)
    prov = LlamaCppProvider(cfg, binary=binary, models_dir=models, log_dir=wd / "server_logs")

    out = pipeline.run_generation(cfg, wd, prov, meter=None)
    first = out["total"]["generated"]
    assert first > 0 and out["total"]["failed_attempts"] == 0

    # every cached record carries the full pinned identity and a valid, stop-trimmed completion
    n_checked = 0
    for shard_path in sorted((wd / "generations").glob("*.jsonl")):
        for ln in shard_path.read_text().splitlines():
            rec = Record.from_line(ln)
            b = rec.backend_info()
            assert b.is_complete() and "fac3a894" in b.build and "stop=" in b.build
            assert rec.wall_s > 0 and rec.ok
            assert validate_completion(rec.to_completion()) == ()
            n_checked += 1
    assert n_checked >= first

    # S and L were pinned to different GPUs (CUDA_VISIBLE_DEVICES) and use different ports
    s = json.loads((wd / "server_logs" / "S_q4km.server.json").read_text())
    l_ = json.loads((wd / "server_logs" / "L_q4km.server.json").read_text())
    assert (s["gpu"], l_["gpu"]) == ("0", "1") and s["port"] != l_["port"]
    cell_files = list((wd / "cells").glob("*.json"))
    assert len(cell_files) == 2 and all("load_s" in json.loads(p.read_text()) for p in cell_files)

    # resuming generates nothing new
    again = pipeline.run_generation(cfg, wd, prov, meter=None)
    assert again["total"]["generated"] == 0 and again["total"]["skipped"] >= first

    # the sanity report reads the cache and reports extraction, defects, latency and servers
    from sieve.eval.providers.sanity import sanity_report

    rep = sanity_report(cfg, wd, per_dataset=3)
    assert set(rep["cells"]) == {"S_q4km", "L_q4km"}
    blk = rep["cells"]["S_q4km"]["datasets"]["triviaqa"]
    assert blk["n"] == 36 and blk["empty_answer"] == 0 and blk["defects"] == {}
    assert blk["wall_s"]["p50"] > 0 and len(blk["sample"]) == 3
    assert blk["sample"][0]["extracted"] and blk["sample"][0]["raw_text"].startswith(" ")
    assert rep["server"]["S_q4km"]["gpu"] == "0" and rep["server"]["L_q4km"]["gpu"] == "1"
    assert "simpleqa_verified" in rep["cells"]["L_q4km"]["datasets"]
    assert rep["cells"]["L_q4km"]["datasets"]["simpleqa_verified"]["accuracy"] is None

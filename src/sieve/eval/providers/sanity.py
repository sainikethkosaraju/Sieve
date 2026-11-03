"""Post-run sanity report over the generation cache (answer extraction, grading, latency).

``python -m sieve.eval.providers.sanity --config C --workdir W [--per-dataset 6] [--out F]``

Reads the append-only cache read-only and prints/writes JSON with, per cell and dataset: counts,
failed attempts, finish reasons, logprob defect codes (``validate_completion``), empty answers,
primary-grader accuracy (SimpleQA stays ungraded without a judge), first-token probability, wall
time and generated-tokens/s, plus a deterministic sample of real outputs with the question, gold
answers, raw text, extracted answer and grade, so extraction and grading can be eyeballed. It
computes nothing that enters the pre-registered analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from sieve.eval import pipeline
from sieve.eval._util import sha256_text
from sieve.eval.cache import Record, read_shard_readonly
from sieve.eval.config import ConfigError, ExperimentConfig, load_config
from sieve.eval.datasets import PINNED, Item
from sieve.eval.grading import grade_item
from sieve.signals import extract_answer, validate_completion
from sieve.types import Completion

__all__ = ["MAX_MISMATCH_RATE", "MIN_STOP_RATE", "check_gates", "main", "sanity_report"]

#: Canary thresholds (eval-audit C1): below this share of finish_reason == "stop" per tier a
#: headline run would silently fail closed on TRUNCATED defects.
MIN_STOP_RATE = 0.95
#: Largest tolerated share of items whose token pieces do not reassemble to the returned text
#: (``utf8_gap`` / ``missing_logprobs`` defects).
MAX_MISMATCH_RATE = 0.02
_MISMATCH_DEFECTS = ("utf8_gap", "missing_logprobs")


def _q(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, round(p * (len(s) - 1)))]


def _first_prob(c: Completion) -> float | None:
    if not c.tokens:
        return None
    lp = c.tokens[min(c.answer_start, len(c.tokens) - 1)].logprob
    return math.exp(lp) if math.isfinite(lp) and lp <= 0 else None


def _mean_lp(c: Completion) -> float | None:
    if not c.tokens:
        return None
    v = [t.logprob for t in c.tokens[c.answer_start :] if math.isfinite(t.logprob)]
    return statistics.fmean(v) if v else None


def _round(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


def cell_names(workdir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted((workdir / "cells").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[str(d["cell_id"])] = str(d["name"])
    return out


def sanity_report(
    cfg: ExperimentConfig, workdir: Path, *, per_dataset: int = 6, max_text: int = 160
) -> dict[str, Any]:
    items: dict[str, Item] = pipeline.load_items(workdir)
    names = cell_names(workdir)
    out: dict[str, Any] = {"config": cfg.name, "cells": {}}
    for shard_path in sorted((workdir / "generations").glob("*.jsonl")):
        cid = shard_path.stem
        shard = read_shard_readonly(shard_path)
        try:
            recs = [r for r in shard.all_records() if r.kind == "primary"]
            finals = [r for r in recs if r.final]
            n_failed = sum(1 for r in recs if not r.ok)
            by_ds: dict[str, list[Record]] = {}
            for r in finals:
                it = items.get(r.item_id)
                by_ds.setdefault(it.dataset if it else "?", []).append(r)
            cell: dict[str, Any] = {
                "cell_id": cid,
                "tier": _tier_of(cfg, names.get(cid, cid)),
                "n_primary_final": len(finals),
                "n_failed_attempts": n_failed,
                "datasets": {},
            }
            for ds, rs in sorted(by_ds.items()):
                cell["datasets"][ds] = _dataset_block(ds, rs, items, per_dataset, max_text)
            out["cells"][names.get(cid, cid)] = cell
        finally:
            shard.close()
    out["energy"] = _energy(workdir)
    out["server"] = _server_sidecars(workdir)
    main = cfg.main_pair()
    out["gates"] = check_gates(out, headline={main.s_cell, main.l_cell})
    return out


def _tier_of(cfg: ExperimentConfig, name: str) -> str:
    try:
        return str(cfg.cell(name).tier)
    except ConfigError:
        return "?"


def check_gates(
    rep: dict[str, Any],
    *,
    min_stop_rate: float = MIN_STOP_RATE,
    max_mismatch_rate: float = MAX_MISMATCH_RATE,
    headline: set[str] | None = None,
) -> dict[str, Any]:
    """Canary gate (C1): per tier, the ``finish == stop`` rate and the token/text mismatch rate.

    With ``headline`` (the main pair's cells) only those cells can fail the gate. Ablation arms
    below the stop-rate floor are reported under ``ablation_findings`` instead: a quantisation
    arm that stops producing terminated answers (the real 4-bit-KV collapse, 2.7% stop rate) is a
    RESULT about that arm, not a defect of the run.

    ``ok`` is False (and the CLI exits non-zero) if any tier is below ``min_stop_rate`` or above
    ``max_mismatch_rate``, or if no generations were found at all.
    """
    tiers: dict[str, dict[str, int]] = {}
    findings: dict[str, dict[str, float]] = {}
    for cname, cell in rep.get("cells", {}).items():
        if headline is not None and cname not in headline:
            n_c = sum(int(b["n"]) for b in cell.get("datasets", {}).values())
            stop_c = sum(int(b["finish"].get("stop", 0)) for b in cell.get("datasets", {}).values())
            if n_c and stop_c / n_c < min_stop_rate:
                findings[cname] = {"n": n_c, "stop_rate": stop_c / n_c}
            continue
        t = tiers.setdefault(str(cell.get("tier", "?")), {"n": 0, "stop": 0, "mismatch": 0})
        for blk in cell.get("datasets", {}).values():
            t["n"] += int(blk["n"])
            t["stop"] += int(blk["finish"].get("stop", 0))
            t["mismatch"] += sum(int(blk["defects"].get(d, 0)) for d in _MISMATCH_DEFECTS)
    failures: list[str] = []
    out_tiers: dict[str, Any] = {}
    for tier, c in sorted(tiers.items()):
        n = c["n"]
        stop_rate = c["stop"] / n if n else 0.0
        mm_rate = c["mismatch"] / n if n else 1.0
        out_tiers[tier] = {"n": n, "stop_rate": stop_rate, "mismatch_rate": mm_rate}
        if n == 0:
            failures.append(f"tier {tier}: no generations")
            continue
        if stop_rate < min_stop_rate:
            failures.append(
                f"tier {tier}: finish_reason==stop rate {stop_rate:.3f} < {min_stop_rate:.2f}"
            )
        if mm_rate > max_mismatch_rate:
            failures.append(
                f"tier {tier}: token/text mismatch rate {mm_rate:.3f} > {max_mismatch_rate:.2f}"
            )
    if not out_tiers:
        failures.append("no generations found")
    return {
        "ok": not failures,
        "failures": failures,
        "tiers": out_tiers,
        "ablation_findings": findings,
        "min_stop_rate": min_stop_rate,
        "max_mismatch_rate": max_mismatch_rate,
    }


def _dataset_block(
    ds: str, rs: list[Record], items: dict[str, Item], per_dataset: int, max_text: int
) -> dict[str, Any]:
    spec = PINNED.get(ds)
    grader = spec.grader if spec else "alias_em"
    finish: Counter[str] = Counter()
    defects: Counter[str] = Counter()
    n_empty = 0
    graded: list[bool] = []
    first_p: list[float] = []
    wall: list[float] = []
    tps: list[float] = []
    ttft: list[float] = []
    n_gen: list[int] = []
    rows: list[dict[str, Any]] = []
    for r in sorted(rs, key=lambda x: x.item_id):
        it = items[r.item_id]
        comp = r.to_completion()
        finish[r.finish] += 1
        for d in validate_completion(comp):
            defects[d.value] += 1
        ans = extract_answer(r.text)
        n_empty += 0 if ans else 1
        g = grade_item(it, r.text, grader) if grader != "simpleqa_judge" else None
        if g is not None and g.correct is not None:
            graded.append(g.correct)
        fp = _first_prob(comp)
        if fp is not None:
            first_p.append(fp)
        wall.append(r.wall_s)
        ttft.append(r.t_first_token_s)
        n_gen.append(r.n_gen)
        if r.wall_s > 0 and r.n_gen > 0:
            tps.append(r.n_gen / r.wall_s)
        rows.append(
            {
                "item_id": r.item_id,
                "question": it.question[:max_text],
                "gold": list(it.answers[:3]),
                "raw_text": r.text[:max_text],
                "extracted": ans[:max_text],
                "grade": "ungraded" if g is None else g.label,
                "first_token_prob": _round(fp),
                "mean_logprob": _round(_mean_lp(comp)),
                "finish": r.finish,
                "n_gen": r.n_gen,
                "answer_start": r.answer_start,
                "wall_s": _round(r.wall_s),
            }
        )
    # deterministic spread sample (hash order), so reruns print the same rows
    sample = sorted(rows, key=lambda x: sha256_text("sanity|" + str(x["item_id"])))[:per_dataset]
    return {
        "n": len(rs),
        "grader": grader if grader != "simpleqa_judge" else "ungraded (no judge)",
        "accuracy": _round(sum(graded) / len(graded)) if graded else None,
        "n_graded": len(graded),
        "empty_answer": n_empty,
        "finish": dict(finish),
        "defects": dict(defects),
        "first_token_prob_mean": _round(statistics.fmean(first_p)) if first_p else None,
        "wall_s": {
            "mean": _round(statistics.fmean(wall)),
            "p50": _round(_q(wall, 0.5)),
            "p90": _round(_q(wall, 0.9)),
            "max": _round(max(wall)),
        },
        "ttft_s_mean": _round(statistics.fmean(ttft)),
        "n_gen_mean": _round(statistics.fmean(n_gen), 2),
        "gen_tokens_per_s_p50": _round(_q(tps, 0.5), 2),
        "sample": sample,
    }


def _energy(workdir: Path) -> list[dict[str, Any]]:
    p = workdir / "energy.jsonl"
    if not p.exists():
        return []
    keys = (
        "state",
        "cell",
        "block",
        "duration_s",
        "n_samples",
        "joules_gross",
        "joules_net",
        "mean_watts_net",
        "peak_vram_mib",
        "idle_watts",
        "throttled",
        "n_generated",
        "complete",
    )
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [{k: r.get(k) for k in keys} for r in rows]


def _server_sidecars(workdir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in sorted((workdir / "server_logs").glob("*.server.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[str(d.get("cell", p.stem))] = {
            "gpu": d.get("gpu"),
            "server_start_s": d.get("server_start_s"),
            "build_info": d.get("props", {}).get("build_info"),
            "log_flash_attn": d.get("log_flash_attn"),
            "backend_build": d.get("backend_info", {}).get("build"),
            "model_hash": str(d.get("backend_info", {}).get("model_hash", ""))[:16],
            "self_test": {c["name"]: c["ok"] for c in d.get("self_test", [])},
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sieve.eval.providers.sanity")
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--per-dataset", type=int, default=6)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rep = sanity_report(load_config(Path(a.config)), Path(a.workdir), per_dataset=a.per_dataset)
    text = json.dumps(rep, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    gates = rep["gates"]
    if not gates["ok"]:
        print("SANITY GATE FAILED: " + "; ".join(gates["failures"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

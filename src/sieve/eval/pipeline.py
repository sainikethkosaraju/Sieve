"""End-to-end stages: prepare-data, run, analyse, audit-sheet, and the offline mock."""

from __future__ import annotations

import importlib
import json
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sieve.eval import prereg
from sieve.eval._fake_backend import FakeProvider, SimClock, SyntheticJudge
from sieve.eval._util import atomic_write_text, sha256_text
from sieve.eval.audit import AuditCandidate, make_audit_sheet
from sieve.eval.cache import GenerationCache
from sieve.eval.config import ExperimentConfig, load_config, parse_config
from sieve.eval.datasets import (
    PINNED,
    Item,
    RowFetcher,
    dataset_hash,
    hf_fetch,
    load_dataset_items,
    read_items,
    sample_items,
    write_items,
)
from sieve.eval.grading import Judge, UnavailableJudge
from sieve.eval.power import Meter, SyntheticMeter
from sieve.eval.runner import (
    BackendProvider,
    Runner,
    WorkUnit,
    plan_units,
    run_cells,
    write_run_manifest,
)
from sieve.eval.splits import SplitStore, build_splits
from sieve.eval.synthetic import synthetic_items

Say = Callable[[str], None]


def _say(_m: str) -> None:
    return None


# --- prepare-data -----------------------------------------------------------------------


def prepare_data(
    cfg: ExperimentConfig,
    workdir: Path,
    *,
    synthetic: bool | None = None,
    fetch: RowFetcher = hf_fetch,
    say: Say = _say,
) -> dict[str, Any]:
    """Load/sample datasets (pinned revisions), write item files and the hash-locked splits."""
    synth = cfg.synthetic if synthetic is None else synthetic
    (workdir / "items").mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"synthetic": synth, "datasets": {}}
    for name, n in cfg.dataset_sizes.items():
        seed = cfg.dataset_seeds[name]
        spec = PINNED.get(name)
        if synth:
            items = synthetic_items(name, n, seed)
            meta: dict[str, Any] = {"source": "synthetic", "name": name}
        else:
            if spec is None:
                raise ValueError(f"no pinned spec for dataset {name!r}")
            say(f"loading {spec.repo} @ {spec.revision[:10]}")
            pool = load_dataset_items(spec, fetch)
            items = sample_items(pool, n, seed)
            meta = {
                "source": "huggingface",
                "repo": spec.repo,
                "config": spec.config,
                "split": spec.split,
                "revision": spec.revision,
                "licence": spec.licence,
                "seed": seed,
                "pool_size": len(pool),
                "grader": spec.grader,
            }
        h = write_items(workdir / "items" / f"{name}.jsonl", items, meta)
        summary["datasets"][name] = {**meta, "n": len(items), "dataset_hash": h}
    prim = read_items(workdir / "items" / f"{cfg.primary_dataset}.jsonl")
    summary["cross_dataset_question_overlap"] = _overlap(workdir, cfg, prim)
    store = build_splits(
        workdir / "splits",
        [i.item_id for i in prim],
        dataset_hash=dataset_hash(prim),
        split_seed=cfg.split_seed,
        n_splits=cfg.n_splits,
        sizes=cfg.split_sizes,
    )
    summary["splits"] = {"index_hash": store.index_hash, "n_splits": store.n_splits}
    atomic_write_text(
        workdir / "items" / "datasets_manifest.json", json.dumps(summary, sort_keys=True, indent=1)
    )
    return summary


def _overlap(workdir: Path, cfg: ExperimentConfig, prim: list[Item]) -> dict[str, int]:
    """Normalised-question duplicates between the split dataset and each shift set."""
    from sieve.eval.datasets import _norm_q

    base = {_norm_q(i.question) for i in prim}
    out: dict[str, int] = {}
    for name in cfg.dataset_sizes:
        if name != cfg.primary_dataset:
            other = read_items(workdir / "items" / f"{name}.jsonl")
            out[name] = sum(1 for i in other if _norm_q(i.question) in base)
    return out


def load_items(workdir: Path) -> dict[str, Item]:
    out: dict[str, Item] = {}
    for p in sorted((workdir / "items").glob("*.jsonl")):
        for it in read_items(p):
            out[it.item_id] = it
    return out


# --- run ------------------------------------------------------------------------------------


def _rank(ids: list[str], seed: str) -> list[str]:
    return sorted(ids, key=lambda i: sha256_text(f"{seed}|{i}"))


def build_units(cfg: ExperimentConfig, items: dict[str, Item]) -> dict[str, list[WorkUnit]]:
    prim_ids = sorted(i for i, it in items.items() if it.dataset == cfg.primary_dataset)
    sc_ids = frozenset(_rank(prim_ids, "sc")[: cfg.sc_n_items])
    rep_ids = frozenset(_rank(prim_ids, "repeat")[: cfg.nondeterminism_items])
    units: dict[str, list[WorkUnit]] = {}
    for cell in cfg.cells:
        sel: list[Item] = []
        for ds in cell.datasets:
            pool = sorted(
                (it for it in items.values() if it.dataset == ds), key=lambda i: i.item_id
            )
            cap = cell.max_items_per_dataset
            if cap is not None:
                if ds == cfg.primary_dataset:
                    raise ValueError("max_items_per_dataset must not truncate the split dataset")
                keep = set(_rank([p.item_id for p in pool], f"cap:{ds}")[:cap])
                pool = [p for p in pool if p.item_id in keep]
            sel.extend(pool)
        units[cell.name] = plan_units(
            [i.prompt_view() for i in sel],
            params=cfg.sampling(),
            sc_item_ids=sc_ids if cell.extra.get("self_consistency") else frozenset(),
            sc_k=cfg.sc_k,
            sc_params=cfg.sc_sampling(),
            repeat_runs=cfg.nondeterminism_runs if cell.extra.get("nondeterminism") else 1,
            repeat_ids=rep_ids if cell.extra.get("nondeterminism") else frozenset(),
        )
    return units


def hardware_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": "see nvidia-smi",
    }


def load_provider(cfg: ExperimentConfig) -> BackendProvider:
    """Backend provider named in the config: ``module:factory`` returning a BackendProvider."""
    spec = str(cfg.raw.get("backend", {}).get("provider", ""))
    if ":" not in spec:
        raise ValueError(
            "config 'backend.provider' must be 'package.module:factory' (see docs/reproduce.md)"
        )
    mod, fn = spec.split(":", 1)
    factory = getattr(importlib.import_module(mod), fn)
    provider: BackendProvider = factory(cfg)
    return provider


def run_generation(
    cfg: ExperimentConfig,
    workdir: Path,
    provider: BackendProvider,
    *,
    meter: Meter | None,
    clock: Callable[[], float] | None = None,
    say: Say = _say,
    only: list[str] | None = None,
) -> dict[str, Any]:
    pre = prereg.verify_unchanged(cfg.prereg_path, cfg.prereg_frozen_sha256)
    write_run_manifest(workdir, cfg, pre, hardware_info())
    items = load_items(workdir)
    units = build_units(cfg, items)
    if only:
        units = {k: v for k, v in units.items() if k in only}
    cache = GenerationCache(workdir / "generations")
    kwargs: dict[str, Any] = {}
    if clock is not None:
        kwargs["clock"] = clock
    runner = Runner(
        cache, pre, cfg.config_hash(), cfg.max_attempts, cfg.n_probs, progress=say, **kwargs
    )
    groups: list[list[str]] = [list(g) for g in cfg.raw.get("abab_groups", [])]
    grouped = {n for g in groups for n in g}
    order: list[list[str]] = [[n for n in g if n in units] for g in groups]
    order += [[n] for n in units if n not in grouped]
    order = [g for g in order if g]
    stats = {}
    try:
        for g in order:
            say(f"running cells {g}")
            st = run_cells(
                cfg, provider, {n: units[n] for n in g}, workdir, runner, meter=meter, group=g
            )
            stats[",".join(g)] = st.__dict__
    finally:
        cache.close()
    return {"stats": stats, "total": runner.stats.__dict__}


# --- judge / audit / analyse ---------------------------------------------------------------


def make_judge(cfg: ExperimentConfig, *, mock: bool = False) -> Judge:
    kind = str(cfg.judge.get("kind", "unavailable"))
    if kind == "synthetic":
        if not cfg.synthetic and not mock:
            raise ValueError("the synthetic judge is only allowed in synthetic runs")
        return SyntheticJudge()
    if kind == "backend":
        spec = str(cfg.judge["factory"])
        mod, fn = spec.split(":", 1)
        j: Judge = getattr(importlib.import_module(mod), fn)(cfg)
        return j
    return UnavailableJudge(
        str(
            cfg.judge.get(
                "reason", "no judge model configured; SimpleQA is reported as unavailable"
            )
        )
    )


def audit_sheet(cfg: ExperimentConfig, workdir: Path, judge: Judge | None = None) -> dict[str, Any]:
    """Blind audit sheet drawn from split A only (never B or C)."""
    from sieve.eval.data import load_experiment
    from sieve.eval.grading import EM_ID

    items = load_items(workdir)
    store = SplitStore.load(workdir / "splits")
    data = load_experiment(workdir, cfg, items, judge)
    main = cfg.main_pair()
    a_ids = set(store.a_ids(0))
    cands: list[AuditCandidate] = []
    for tier, cname in (("S", main.s_cell), ("L", main.l_cell)):
        c = data.cells[cname]
        for i, iid in enumerate(c.ids):
            if iid in a_ids and c.graded[i]:
                it = items[iid]
                cands.append(
                    AuditCandidate(
                        iid,
                        it.dataset,
                        tier,
                        it.question,
                        it.answers,
                        c.texts[i],
                        bool(c.correct[i]),
                    )
                )
    sheet, key = make_audit_sheet(
        cands, workdir / "audit", n=cfg.audit_n, seed=cfg.boot_seed, grader_id=EM_ID
    )
    return {"sheet": str(sheet), "key": str(key), "n_candidates": len(cands)}


# --- the offline mock --------------------------------------------------------------------------


def mock_config(
    *, judge: str = "synthetic", n_boot: int = 1000, small: bool = False
) -> dict[str, Any]:
    """Synthetic end-to-end configuration; every dataset is generated, nothing is downloaded."""
    allds = ["triviaqa", "nq_open", "simpleqa_verified", "mmlu_redux"]

    def cell(
        name: str, tier: str, model: str, quant: str, kv: str, ds: list[str], **extra: Any
    ) -> dict[str, Any]:
        return {
            "name": name,
            "tier": tier,
            "model_id": model,
            "quant": quant,
            "kv_quant": kv,
            "datasets": ds,
            "extra": extra,
        }

    cfg: dict[str, Any] = {
        "name": "mock-synthetic",
        "synthetic": True,
        "alpha": 0.10,
        "delta": 0.10,
        "preregistration": {"path": "docs/PREREGISTRATION.md", "frozen_sha256": None},
        "primary_dataset": "triviaqa",
        "splits": {"n_splits": 50, "sizes": [400, 400, 400], "seed": 4242},
        "datasets": {
            "triviaqa": {"n": 1200, "seed": 1},
            "nq_open": {"n": 300, "seed": 2},
            "simpleqa_verified": {"n": 200, "seed": 3},
            "mmlu_redux": {"n": 300, "seed": 4},
        },
        "generation": {"n_probs": 20, "max_tokens": 32, "max_attempts": 3},
        "self_consistency": {"k": 5, "n_items": 300, "temperature": 0.7, "top_p": 0.95},
        "nondeterminism": {"items": 60, "runs": 3},
        "audit": {"n": 60},
        "statistics": {
            "n_boot": n_boot,
            "seed": 7,
            "equivalence_margin": 0.02,
            "noninferiority_margin": 0.02,
            "h1_pass_max": 0.15,
            "h1_fail_min": 0.25,
        },
        "energy": {"abab_blocks": 4, "idle_seconds": 0.0},
        "timing_source": "wall",
        "cost_unit": "seconds",
        "drift_window": 50,
        "judge": {"kind": judge},
        "cells": [
            cell(
                "S_q4km",
                "S",
                "synthetic-S",
                "Q4_K_M",
                "f16",
                allds,
                self_consistency=True,
                nondeterminism=True,
            ),
            cell("L_q4km", "L", "synthetic-L", "Q4_K_M", "f16", allds),
            cell("S_f16", "S", "synthetic-S", "f16", "f16", ["triviaqa"]),
            cell("S_q8", "S", "synthetic-S", "Q8_0", "f16", ["triviaqa"]),
            cell("S_q3km", "S", "synthetic-S", "Q3_K_M", "f16", ["triviaqa"]),
            cell("S_q4km_kvq4", "S", "synthetic-S", "Q4_K_M", "q4_0", ["triviaqa"]),
        ],
        "pairs": [
            {"name": "main", "s_cell": "S_q4km", "l_cell": "L_q4km", "role": "main"},
            {"name": "ladder_f16", "s_cell": "S_f16", "l_cell": "L_q4km", "role": "quant_ref"},
            {"name": "ladder_q8", "s_cell": "S_q8", "l_cell": "L_q4km", "role": "quant"},
            {"name": "ladder_q4km", "s_cell": "S_q4km", "l_cell": "L_q4km", "role": "quant"},
            {"name": "ladder_q3km", "s_cell": "S_q3km", "l_cell": "L_q4km", "role": "quant"},
            {"name": "ladder_kvq4", "s_cell": "S_q4km_kvq4", "l_cell": "L_q4km", "role": "quant"},
        ],
        "abab_groups": [["S_f16", "S_q8"], ["S_q3km", "S_q4km_kvq4"], ["S_q4km", "L_q4km"]],
    }
    if small:  # fast variant for the test-suite; same stages, smaller sizes
        cfg["splits"] = {"n_splits": 10, "sizes": [150, 150, 150], "seed": 4242}
        cfg["datasets"] = {
            "triviaqa": {"n": 450, "seed": 1},
            "nq_open": {"n": 150, "seed": 2},
            "simpleqa_verified": {"n": 100, "seed": 3},
            "mmlu_redux": {"n": 150, "seed": 4},
        }
        cfg["self_consistency"]["n_items"] = 150
        cfg["nondeterminism"] = {"items": 30, "runs": 3}
        cfg["audit"] = {"n": 40}
        cfg["statistics"]["n_boot"] = min(n_boot, 200)
        cfg["drift_window"] = 30
    return cfg


def run_mock(
    workdir: Path,
    *,
    cfg_dict: dict[str, Any] | None = None,
    provider: FakeProvider | None = None,
    clock: SimClock | None = None,
    say: Say = _say,
    do_report: bool = True,
) -> dict[str, Any]:
    """prepare -> run -> audit sheet -> analyse -> hypotheses -> report, all synthetic."""
    from sieve.eval.analysis import run_analysis
    from sieve.eval.hypotheses import build_report, format_report
    from sieve.eval.report import write_report

    d = cfg_dict or mock_config()
    d["preregistration"]["frozen_sha256"] = prereg.frozen_hash(d["preregistration"]["path"])
    cfg = parse_config(d)
    workdir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(workdir / "config.json", json.dumps(d, sort_keys=True, indent=1))
    prepare_data(cfg, workdir, synthetic=True, say=say)
    sim = clock or SimClock()
    prov = provider or FakeProvider(sim)
    meter = SyntheticMeter(
        sim.now,
        {
            "S_q4km": 70.0,
            "L_q4km": 120.0,
            "S_f16": 85.0,
            "S_q8": 80.0,
            "S_q3km": 65.0,
            "S_q4km_kvq4": 70.0,
        },
    )
    out: dict[str, Any] = {
        "generation": run_generation(cfg, workdir, prov, meter=meter, clock=sim.now, say=say)
    }
    judge = make_judge(cfg, mock=True)
    out["audit"] = audit_sheet(cfg, workdir, judge)
    results = run_analysis(workdir, cfg, load_items(workdir), judge, progress=say)
    rep = build_report(results, cfg)
    atomic_write_text(
        workdir / "analysis" / "hypotheses.json", json.dumps(rep, sort_keys=True, indent=1)
    )
    atomic_write_text(workdir / "analysis" / "hypotheses.txt", format_report(rep) + "\n")
    out["hypotheses"] = {v["id"]: v["verdict"] for v in rep["verdicts"]}
    if do_report:
        out["report"] = str(write_report(workdir, results, rep, cfg))
    return out


__all__ = [
    "audit_sheet",
    "build_units",
    "load_config",
    "load_items",
    "make_judge",
    "mock_config",
    "prepare_data",
    "run_generation",
    "run_mock",
]

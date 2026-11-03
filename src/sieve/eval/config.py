"""Experiment configuration (YAML or JSON) and the frozen pre-registration pin.

Every number that the pre-registration fixes (sample sizes, alpha, delta, bootstrap
reps, equivalence margin, ...) appears in the config *and* is checked against
``docs/PREREGISTRATION.md`` by ``sieve.eval.prereg``; the config can pin the frozen
body hash so that a changed pre-registration is detected at run, analysis and report time.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sieve.eval._util import sha256_obj
from sieve.types import SamplingParams


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CellSpec:
    """One serving configuration: model, weight quant, KV type, decoding."""

    name: str
    tier: str  # "S" | "L"
    model_id: str
    quant: str
    kv_quant: str
    datasets: tuple[str, ...]
    max_items_per_dataset: int | None = None  # None: all sampled items
    extra: dict[str, Any] = field(default_factory=dict)  # gguf path, server flags, ...

    def identity(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "quant": self.quant,
            "kv_quant": self.kv_quant,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class PairSpec:
    """An (S cell, L cell) pair that the analysis turns into a cascade."""

    name: str
    s_cell: str
    l_cell: str
    role: str = "main"  # "main" | "quant"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    synthetic: bool
    alpha: float
    delta: float
    prereg_path: str
    prereg_frozen_sha256: str | None
    primary_dataset: str
    n_splits: int
    split_sizes: tuple[int, int, int]
    split_seed: int
    dataset_sizes: dict[str, int]
    dataset_seeds: dict[str, int]
    cells: tuple[CellSpec, ...]
    pairs: tuple[PairSpec, ...]
    n_probs: int
    max_tokens: int
    max_attempts: int
    sc_k: int
    sc_n_items: int
    sc_temperature: float
    sc_top_p: float
    nondeterminism_items: int
    nondeterminism_runs: int
    audit_n: int
    n_boot: int
    boot_seed: int
    equivalence_margin: float
    noninferiority_margin: float
    h1_pass_max: float
    h1_fail_min: float
    abab_blocks: int
    idle_seconds: float
    timing_source: str  # "wall" | "backend"
    cost_unit: str  # "seconds" | "joules"
    judge: dict[str, Any]
    window_size: int = 200
    raw: dict[str, Any] = field(default_factory=dict)

    def cell(self, name: str) -> CellSpec:
        for c in self.cells:
            if c.name == name:
                return c
        raise ConfigError(f"unknown cell {name!r}")

    def sampling(self) -> SamplingParams:
        return SamplingParams(temperature=0.0, max_tokens=self.max_tokens)

    def sc_sampling(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.sc_temperature, top_p=self.sc_top_p, max_tokens=self.max_tokens
        )

    def config_hash(self) -> str:
        return sha256_obj(self.raw)

    def main_pair(self) -> PairSpec:
        for p in self.pairs:
            if p.role == "main":
                return p
        raise ConfigError("no pair with role 'main'")


def _load_text(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            yaml = importlib.import_module("yaml")
        except ImportError as e:
            raise ConfigError("YAML configs need PyYAML: pip install pyyaml") from e
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def parse_config(d: dict[str, Any]) -> ExperimentConfig:
    try:
        ds = d["datasets"]
        cells = tuple(
            CellSpec(
                name=str(c["name"]),
                tier=str(c["tier"]),
                model_id=str(c["model_id"]),
                quant=str(c["quant"]),
                kv_quant=str(c["kv_quant"]),
                datasets=tuple(str(x) for x in c["datasets"]),
                max_items_per_dataset=c.get("max_items_per_dataset"),
                extra=dict(c.get("extra", {})),
            )
            for c in d["cells"]
        )
        pairs = tuple(
            PairSpec(str(p["name"]), str(p["s_cell"]), str(p["l_cell"]), str(p.get("role", "main")))
            for p in d["pairs"]
        )
        gen = d["generation"]
        sc = d["self_consistency"]
        st = d["statistics"]
        prereg = d["preregistration"]
        cfg = ExperimentConfig(
            name=str(d["name"]),
            synthetic=bool(d.get("synthetic", False)),
            alpha=float(d["alpha"]),
            delta=float(d["delta"]),
            prereg_path=str(prereg["path"]),
            prereg_frozen_sha256=prereg.get("frozen_sha256"),
            primary_dataset=str(d["primary_dataset"]),
            n_splits=int(d["splits"]["n_splits"]),
            split_sizes=(
                int(d["splits"]["sizes"][0]),
                int(d["splits"]["sizes"][1]),
                int(d["splits"]["sizes"][2]),
            ),
            split_seed=int(d["splits"]["seed"]),
            dataset_sizes={k: int(v["n"]) for k, v in ds.items()},
            dataset_seeds={k: int(v["seed"]) for k, v in ds.items()},
            cells=cells,
            pairs=pairs,
            n_probs=int(gen["n_probs"]),
            max_tokens=int(gen["max_tokens"]),
            max_attempts=int(gen["max_attempts"]),
            sc_k=int(sc["k"]),
            sc_n_items=int(sc["n_items"]),
            sc_temperature=float(sc["temperature"]),
            sc_top_p=float(sc.get("top_p", 0.95)),
            nondeterminism_items=int(d["nondeterminism"]["items"]),
            nondeterminism_runs=int(d["nondeterminism"]["runs"]),
            audit_n=int(d["audit"]["n"]),
            n_boot=int(st["n_boot"]),
            boot_seed=int(st["seed"]),
            equivalence_margin=float(st["equivalence_margin"]),
            noninferiority_margin=float(st["noninferiority_margin"]),
            h1_pass_max=float(st["h1_pass_max"]),
            h1_fail_min=float(st["h1_fail_min"]),
            abab_blocks=int(d["energy"]["abab_blocks"]),
            idle_seconds=float(d["energy"]["idle_seconds"]),
            timing_source=str(d.get("timing_source", "wall")),
            cost_unit=str(d.get("cost_unit", "seconds")),
            judge=dict(d.get("judge", {"kind": "unavailable"})),
            window_size=int(d.get("drift_window", 200)),
            raw=d,
        )
    except (KeyError, TypeError, IndexError) as e:
        raise ConfigError(f"malformed config: {e!r}") from e
    names = [c.name for c in cfg.cells]
    if len(set(names)) != len(names):
        raise ConfigError("duplicate cell names")
    for p in cfg.pairs:
        for n in (p.s_cell, p.l_cell):
            if n not in names:
                raise ConfigError(f"pair {p.name}: unknown cell {n!r}")
    if cfg.timing_source not in ("wall", "backend") or cfg.cost_unit not in ("seconds", "joules"):
        raise ConfigError("bad timing_source / cost_unit")
    return cfg


def load_config(path: Path) -> ExperimentConfig:
    return parse_config(_load_text(path))

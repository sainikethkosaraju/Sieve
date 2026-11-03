"""Cache-everything generation runner: resumable, idempotent, wasted-attempt-aware.

For every work unit (item, kind, sample) the runner skips it if the cell shard already holds
a final record; otherwise it generates, appending one line per attempt (failed attempts are
kept: their wall time is wasted time that cost accounting must charge). A process killed at
any instruction leaves at worst one torn line, which the shard drops on reopen.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from sieve.eval import prompts
from sieve.eval._util import atomic_write_text, git_commit, sha256_obj, stable_int
from sieve.eval.cache import CellShard, GenerationCache, Record, backend_to_dict
from sieve.eval.config import CellSpec, ExperimentConfig
from sieve.eval.datasets import PromptItem
from sieve.eval.power import Meter, abab_order
from sieve.types import Backend, Completion, SamplingParams


def cell_id(cell: CellSpec, sampling: SamplingParams) -> str:
    """Stable id of a serving configuration (excludes the per-call seed)."""
    ident = {
        "cell": cell.identity(),
        "tier": cell.tier,
        "sampling_key": [str(x) for x in sampling.key()],
        "max_tokens": sampling.max_tokens,
        "template": prompts.template_hash_for(cell.datasets),
    }
    return sha256_obj(ident)[:16]


@dataclass(frozen=True)
class WorkUnit:
    item: PromptItem
    kind: str  # "primary" | "sample"
    sample_idx: int
    params: SamplingParams

    @property
    def key(self) -> str:
        return f"{self.item.item_id}|{self.kind}|{self.sample_idx}"


def plan_units(
    items: Sequence[PromptItem],
    *,
    params: SamplingParams,
    sc_item_ids: frozenset[str] = frozenset(),
    sc_k: int = 0,
    sc_params: SamplingParams | None = None,
    repeat_runs: int = 1,
    repeat_ids: frozenset[str] = frozenset(),
) -> list[WorkUnit]:
    """Primary greedy generation for every item; k samples for the self-consistency subset;
    extra repeat generations (kind ``repeat``) for the nondeterminism subset."""
    out: list[WorkUnit] = []
    for it in sorted(items, key=lambda i: i.item_id):
        out.append(WorkUnit(it, "primary", 0, params))
        if it.item_id in sc_item_ids and sc_params is not None:
            out.extend(WorkUnit(it, "sample", j, sc_params) for j in range(sc_k))
        if it.item_id in repeat_ids:
            out.extend(WorkUnit(it, "repeat", j, params) for j in range(1, repeat_runs))
    return out


@dataclass
class RunStats:
    generated: int = 0
    skipped: int = 0
    failed_attempts: int = 0
    wasted_wall_s: float = 0.0
    total_wall_s: float = 0.0

    def add(self, other: RunStats) -> None:
        self.generated += other.generated
        self.skipped += other.skipped
        self.failed_attempts += other.failed_attempts
        self.wasted_wall_s += other.wasted_wall_s
        self.total_wall_s += other.total_wall_s


@dataclass
class Runner:
    cache: GenerationCache
    prereg_hash: str
    config_hash: str
    max_attempts: int = 3
    n_probs: int = 20
    clock: Callable[[], float] = time.perf_counter
    progress: Callable[[str], None] | None = None
    stats: RunStats = field(default_factory=RunStats)

    async def _attempt(self, backend: Backend, unit: WorkUnit, cid: str, attempt: int) -> Record:
        seed = stable_int(unit.item.item_id, cid, unit.kind, unit.sample_idx, attempt)
        params = replace(unit.params, seed=seed)
        t0 = self.clock()
        comp: Completion | None = None
        err: str | None = None
        try:
            comp = await backend.generate(
                prompts.render_prompt(unit.item), params, n_probs=self.n_probs
            )
        except Exception as e:  # backend failures are data; BaseException (kill) propagates
            err = f"{type(e).__name__}: {e}"[:300]
        wall = self.clock() - t0
        info = backend_to_dict(backend.info)
        if comp is None or comp.finish == "error":
            ok = False
            comp = comp or Completion("", None, "error", 0, params)
            err = err or "backend returned finish=error"
        else:
            ok = True
            comp = replace(comp, answer_start=prompts.find_answer_start(comp))
        final = ok or attempt >= self.max_attempts - 1
        return Record(
            item_id=unit.item.item_id,
            cell_id=cid,
            kind=unit.kind,
            sample_idx=unit.sample_idx,
            attempt=attempt,
            final=final,
            ok=ok,
            error=err,
            text=comp.text,
            tokens=comp.tokens,
            finish=comp.finish,
            n_gen=comp.n_gen,
            n_prompt=comp.n_prompt,
            answer_start=comp.answer_start,
            t_first_token_s=comp.t_first_token_s,
            t_total_s=comp.t_total_s,
            wall_s=wall,
            backend=info,
            sampling=params,
            config_hash=self.config_hash,
            prereg_hash=self.prereg_hash,
        )

    async def run_units(
        self,
        backend: Backend,
        cell: CellSpec,
        cid: str,
        units: Sequence[WorkUnit],
        shard: CellShard,
    ) -> RunStats:
        st = RunStats()
        for n, unit in enumerate(units):
            if shard.is_done(unit.key):
                st.skipped += 1
                continue
            while not shard.is_done(unit.key):
                rec = await self._attempt(backend, unit, cid, shard.next_attempt(unit.key))
                shard.append(rec)
                st.total_wall_s += rec.wall_s
                if not rec.ok:
                    st.failed_attempts += 1
                    st.wasted_wall_s += rec.wall_s
            st.generated += 1
            if self.progress and (n + 1) % 500 == 0:
                self.progress(f"{cell.name}: {n + 1}/{len(units)}")
        self.stats.add(st)
        return st


class BackendProvider(Protocol):
    """Opens a backend for a cell (server launch / model load happens here)."""

    def open(self, cell: CellSpec) -> AbstractContextManager[Backend]: ...


def run_cells(
    cfg: ExperimentConfig,
    provider: BackendProvider,
    units_by_cell: dict[str, list[WorkUnit]],
    workdir: Path,
    runner: Runner,
    *,
    meter: Meter | None = None,
    group: Sequence[str] | None = None,
) -> RunStats:
    """Run the given cells. With a meter and ``abab_blocks`` > 1 the cells of ``group`` are
    interleaved block-wise (all backends resident at once); otherwise one cell after another.

    Per-block energy lines go to ``energy.jsonl`` (append-only, one per cell x block).
    """
    total = RunStats()
    names = list(group) if group is not None else list(units_by_cell)
    energy_path = workdir / "energy.jsonl"
    done_blocks: set[tuple[str, int]] = set()
    if energy_path.exists():
        for ln in energy_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                e = json.loads(ln)
                if e.get("complete"):
                    done_blocks.add((e["cell"], int(e["block"])))
    n_blocks = max(cfg.abab_blocks, 1) if meter is not None else 1
    with ExitStack() as stack:
        backends: dict[str, Backend] = {}
        t_load: dict[str, float] = {}
        for name in names:
            cell = cfg.cell(name)
            t0 = runner.clock()
            backends[name] = stack.enter_context(provider.open(cell))
            t_load[name] = runner.clock() - t0
            _write_cell_manifest(workdir, cell, cfg, backends[name], t_load[name])
        for name, b in abab_order(names, n_blocks, counterbalance=False):
            cell = cfg.cell(name)
            units = units_by_cell[name]
            block_units = [u for j, u in enumerate(units) if j % n_blocks == b]
            if not block_units:
                continue
            cid = cell_id(cell, cfg.sampling())
            shard = runner.cache.shard(cid)
            already = (name, b) in done_blocks
            if meter is not None and not already:
                with meter.measure(name, b):
                    st = asyncio.run(
                        runner.run_units(backends[name], cell, cid, block_units, shard)
                    )
                res = meter.results()[-1].to_dict()
                res["complete"] = all(shard.is_done(u.key) for u in block_units)
                res["n_generated"] = st.generated
                with energy_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(res, sort_keys=True) + "\n")
            else:
                st = asyncio.run(runner.run_units(backends[name], cell, cid, block_units, shard))
            total.add(st)
    return total


def _write_cell_manifest(
    workdir: Path, cell: CellSpec, cfg: ExperimentConfig, backend: Backend, load_s: float
) -> None:
    sampling = cfg.sampling()
    cid = cell_id(cell, sampling)
    path = workdir / "cells" / f"{cid}.json"
    body: dict[str, Any] = {
        "cell_id": cid,
        "name": cell.name,
        "tier": cell.tier,
        "backend": backend_to_dict(backend.info),
        "cell_spec": cell.identity(),
        "load_s": load_s,  # model load/swap time, reported separately from generation time
    }
    atomic_write_text(path, json.dumps(body, sort_keys=True, indent=1))


def write_run_manifest(
    workdir: Path, cfg: ExperimentConfig, prereg_hash: str, hardware: dict[str, Any]
) -> dict[str, Any]:
    """Create ``manifest.json`` once; later calls verify config/prereg did not change."""
    path = workdir / "manifest.json"
    body: dict[str, Any] = {
        "name": cfg.name,
        "synthetic": cfg.synthetic,
        "config_hash": cfg.config_hash(),
        "prereg_frozen_sha256": prereg_hash,
        "commit": git_commit(),
        "hardware": hardware,
        "template_hash": prompts.template_hash_for(
            sorted({d for c in cfg.cells for d in c.datasets})
        ),
        "created": "synthetic-run"
        if cfg.synthetic
        else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        for k in ("config_hash", "prereg_frozen_sha256", "template_hash"):
            if old.get(k) != body[k]:
                raise RuntimeError(
                    f"manifest {k} differs from this run's ({old.get(k)} vs {body[k]}); "
                    "use a fresh workdir or revert the change"
                )
        return old  # type: ignore[no-any-return]
    atomic_write_text(path, json.dumps(body, sort_keys=True, indent=1))
    return body

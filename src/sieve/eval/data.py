"""Turn the generation cache into per-cell arrays: signals, grades, timings.

Everything downstream (calibration, baselines, hypotheses) is computed from these arrays
on CPU; no model is called. Signals come from ``sieve.signals`` (Phase A), unmodified.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sieve.eval.cache import CellShard, Record
from sieve.eval.config import ExperimentConfig, PairSpec
from sieve.eval.datasets import PINNED, Item
from sieve.eval.grading import Grade, Judge, UnavailableJudge, grade_item, is_wrong
from sieve.eval.runner import cell_id
from sieve.signals import compute_signals
from sieve.types import SCORE_NAMES, BackendInfo

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]


@dataclass
class CellData:
    name: str
    cid: str
    tier: str
    backend: BackendInfo
    ids: tuple[str, ...]
    index: dict[str, int]
    signals: dict[str, FloatArr]
    time_s: FloatArr  # final attempt + every wasted attempt, primary generation only
    wasted_s: FloatArr
    sample_time_s: FloatArr  # self-consistency samples (0 where none)
    has_samples: BoolArr
    texts: list[str]
    correct: BoolArr
    wrong: BoolArr  # False where ungraded; use ``graded`` to mask
    graded: BoolArr
    labels: list[str]
    n_answer_tokens: FloatArr
    attempts_failed: int = 0


def _time_of(r: Record, source: str) -> float:
    return r.wall_s if source == "wall" else r.t_total_s


def build_cell_data(
    name: str,
    tier: str,
    cid: str,
    shard: CellShard,
    items: dict[str, Item],
    cfg: ExperimentConfig,
    judge: Judge,
    judge_cache: Any,
) -> CellData:
    from sieve.eval.grading import grade_simpleqa

    primary_keys = sorted(k for k in shard.item_keys() if k.endswith("|primary|0"))
    ids = tuple(k.split("|")[0] for k in primary_keys)
    n = len(ids)
    sig = {s: np.full(n, np.nan) for s in SCORE_NAMES}
    time_s = np.zeros(n)
    wasted = np.zeros(n)
    sample_t = np.zeros(n)
    has_samples = np.zeros(n, dtype=bool)
    n_tok = np.full(n, np.nan)
    correct = np.zeros(n, dtype=bool)
    wrong = np.zeros(n, dtype=bool)
    graded = np.zeros(n, dtype=bool)
    texts: list[str] = []
    labels: list[str] = []
    backend: BackendInfo | None = None
    failed = 0
    for i, (iid, pk) in enumerate(zip(ids, primary_keys, strict=True)):
        item = items[iid]
        rec = shard.final(pk)
        if rec is None:
            raise ValueError(f"{name}: item {iid} has no final record (run not complete)")
        if rec.config_hash != cfg.config_hash():
            raise ValueError(
                f"{name}: item {iid} was generated under config {rec.config_hash[:12]}, not "
                f"{cfg.config_hash()[:12]}; refusing to analyse runs of different configs"
            )
        backend = backend or rec.backend_info()
        samples: list[Record] = []
        sk_prefix = f"{iid}|sample|"
        for j in range(cfg.sc_k):
            r = shard.final(f"{sk_prefix}{j}")
            if r is not None:
                samples.append(r)
        # F-02: no gold-derived alias table; clustering uses the samples alone, as at deployment
        s = compute_signals(
            rec.to_completion(),
            rec.backend_info(),
            [r.to_completion() for r in samples] if samples else (),
            consistency_sampling=cfg.sc_sampling() if samples else None,
        )
        for sname in SCORE_NAMES:
            v = s.score(sname)
            sig[sname][i] = np.nan if v is None else v
        n_tok[i] = s.n_answer_tokens
        attempts = shard.attempts(pk)
        if cfg.timing_source == "wall":
            wasted[i] = shard.wasted_wall_s(pk)
            time_s[i] = shard.total_wall_s(pk)
        else:
            wasted[i] = sum(r.t_total_s for r in attempts if not r.ok)
            time_s[i] = sum(r.t_total_s for r in attempts)
        failed += sum(1 for r in attempts if not r.ok)
        if samples:
            has_samples[i] = True
            sample_t[i] = sum(
                sum(_time_of(r, cfg.timing_source) for r in shard.attempts(f"{sk_prefix}{j}"))
                for j in range(cfg.sc_k)
            )
        texts.append(rec.text)
        grader = PINNED[item.dataset].grader
        g: Grade
        if grader == "simpleqa_judge":
            g = grade_simpleqa(item, rec.text if rec.ok else "", judge, cache=judge_cache)
        else:
            g = grade_item(item, rec.text if rec.ok else "", grader)
        labels.append(g.label)
        w = is_wrong(g)
        if w is not None:
            graded[i] = True
            wrong[i] = w
            correct[i] = bool(g.correct)
    assert backend is not None, f"{name}: empty shard"
    return CellData(
        name,
        cid,
        tier,
        backend,
        ids,
        {k: i for i, k in enumerate(ids)},
        sig,
        time_s,
        wasted,
        sample_t,
        has_samples,
        texts,
        correct,
        wrong,
        graded,
        labels,
        n_tok,
        failed,
    )


@dataclass(frozen=True)
class EnergyAgg:
    state: str
    mean_watts_net: float | None
    peak_vram_mib: float | None
    n_blocks: int
    throttled_blocks: int
    cv_watts: float | None = None  # coefficient of variation of per-block net power


def load_energy(workdir: Path) -> dict[str, EnergyAgg]:
    path = workdir / "energy.jsonl"
    if not path.exists():
        return {}
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            e = json.loads(ln)
            by_cell.setdefault(e["cell"], []).append(e)
    out: dict[str, EnergyAgg] = {}
    for cell, es in by_cell.items():
        states = {e["state"] for e in es}
        if states == {"no_gpu"}:
            out[cell] = EnergyAgg("no_gpu", None, None, len(es), 0)
            continue
        ok = [e for e in es if e["joules_net"] is not None and e["duration_s"] > 0]
        if not ok:
            out[cell] = EnergyAgg("no_gpu", None, None, len(es), 0)
            continue
        net = sum(e["joules_net"] for e in ok)
        dur = sum(e["duration_s"] for e in ok)
        state = "synthetic" if "synthetic" in states else "measured"
        watts = [e["joules_net"] / e["duration_s"] for e in ok]
        cv = (
            float(np.std(watts, ddof=1) / np.mean(watts))
            if len(watts) > 1 and np.mean(watts) > 0
            else None
        )
        out[cell] = EnergyAgg(
            state,
            net / dur,
            max(e["peak_vram_mib"] for e in ok),
            len(ok),
            sum(1 for e in ok if e["throttled"]),
            cv,
        )
    return out


@dataclass
class View:
    """One tier restricted to an ordered id list."""

    signals: dict[str, FloatArr]
    wrong: BoolArr
    correct: BoolArr
    graded: BoolArr
    time_s: FloatArr
    sample_time_s: FloatArr
    has_samples: BoolArr
    texts: list[str]
    labels: list[str]

    def n(self) -> int:
        return int(self.wrong.size)


@dataclass
class PairSlice:
    """Aligned S and L views over the same ids (the unit the policies are evaluated on)."""

    ids: tuple[str, ...]
    questions: list[str]
    datasets: list[str]
    strata: list[str]
    s: View
    l: View  # noqa: E741
    cost_unit: str
    watts_s: float = 1.0
    watts_l: float = 1.0

    @property
    def n(self) -> int:
        return len(self.ids)

    def cost_s(self) -> FloatArr:
        return self.s.time_s * self.watts_s

    def cost_l(self) -> FloatArr:
        return self.l.time_s * self.watts_l

    def sample_cost_s(self) -> FloatArr:
        return self.s.sample_time_s * self.watts_s

    def both_graded(self) -> BoolArr:
        return self.s.graded & self.l.graded

    def subset(self, mask: BoolArr) -> PairSlice:
        sel = np.flatnonzero(mask)
        return self.take_positions(sel)

    def take_positions(self, sel: npt.NDArray[np.intp]) -> PairSlice:
        def v(x: View) -> View:
            return View(
                {k: a[sel] for k, a in x.signals.items()},
                x.wrong[sel],
                x.correct[sel],
                x.graded[sel],
                x.time_s[sel],
                x.sample_time_s[sel],
                x.has_samples[sel],
                [x.texts[i] for i in sel],
                [x.labels[i] for i in sel],
            )

        return PairSlice(
            tuple(self.ids[i] for i in sel),
            [self.questions[i] for i in sel],
            [self.datasets[i] for i in sel],
            [self.strata[i] for i in sel],
            v(self.s),
            v(self.l),
            self.cost_unit,
            self.watts_s,
            self.watts_l,
        )


@dataclass
class ExperimentData:
    cfg: ExperimentConfig
    items: dict[str, Item]
    cells: dict[str, CellData]
    energy: dict[str, EnergyAgg]
    manifest: dict[str, Any]
    judge_state: str
    ungraded_datasets: list[str] = field(default_factory=list)

    def _view(self, c: CellData, pos: list[int]) -> View:
        sel = np.asarray(pos, dtype=np.intp)
        return View(
            {k: a[sel] for k, a in c.signals.items()},
            c.wrong[sel],
            c.correct[sel],
            c.graded[sel],
            c.time_s[sel],
            c.sample_time_s[sel],
            c.has_samples[sel],
            [c.texts[i] for i in pos],
            [c.labels[i] for i in pos],
        )

    def slice_pair(
        self, pair: PairSpec, ids: Sequence[str], cost_unit: str | None = None
    ) -> PairSlice:
        cs, cl = self.cells[pair.s_cell], self.cells[pair.l_cell]
        try:
            ps = [cs.index[i] for i in ids]
            pl = [cl.index[i] for i in ids]
        except KeyError as e:
            raise KeyError(f"pair {pair.name}: item {e} missing from a cell") from e
        unit = cost_unit or self.cfg.cost_unit
        ws = wl = 1.0
        if unit == "joules":
            es, el = self.energy.get(pair.s_cell), self.energy.get(pair.l_cell)
            if es is None or el is None or es.mean_watts_net is None or el.mean_watts_net is None:
                raise ValueError("cost_unit=joules needs measured power for both cells")
            ws, wl = es.mean_watts_net, el.mean_watts_net
        return PairSlice(
            tuple(ids),
            [self.items[i].question for i in ids],
            [self.items[i].dataset for i in ids],
            [self.items[i].stratum for i in ids],
            self._view(cs, ps),
            self._view(cl, pl),
            unit,
            ws,
            wl,
        )

    def ids_with(self, pair: PairSpec) -> set[str]:
        return set(self.cells[pair.s_cell].ids) & set(self.cells[pair.l_cell].ids)


def load_experiment(
    workdir: Path,
    cfg: ExperimentConfig,
    items: dict[str, Item],
    judge: Judge | None = None,
) -> ExperimentData:
    from sieve.eval.cache import GenerationCache
    from sieve.eval.grading import JudgeCache

    judge = judge or UnavailableJudge()
    cache = GenerationCache(workdir / "generations")
    jc = JudgeCache(workdir / "grades" / "judge_cache.jsonl")
    cells: dict[str, CellData] = {}
    for c in cfg.cells:
        cid = cell_id(c, cfg.sampling())
        if not (workdir / "generations" / f"{cid}.jsonl").exists():
            continue
        cells[c.name] = build_cell_data(
            c.name, c.tier, cid, cache.shard(cid), items, cfg, judge, jc
        )
    cache.close()
    manifest_p = workdir / "manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8")) if manifest_p.exists() else {}
    ungraded = sorted(
        {
            items[i].dataset
            for c in cells.values()
            for i, g in zip(c.ids, c.graded, strict=True)
            if not g
        }
    )
    return ExperimentData(
        cfg, items, cells, load_energy(workdir), manifest, judge.judge_id, ungraded
    )

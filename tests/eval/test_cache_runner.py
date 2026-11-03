"""Cache integrity, resumability after a kill, idempotence, wasted-attempt accounting."""

from __future__ import annotations

import asyncio
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from sieve.eval import pipeline
from sieve.eval._fake_backend import FakeBackend, FakeProvider, SimClock, SimulatedKillError
from sieve.eval.cache import CacheCorruptionError, CellShard, GenerationCache, Record
from sieve.eval.config import CellSpec, ExperimentConfig, parse_config
from sieve.eval.runner import Runner, RunStats, WorkUnit, cell_id, plan_units
from sieve.eval.synthetic import synthetic_items
from sieve.types import SamplingParams, TokenLP

BASE = Record(
    item_id="i1",
    cell_id="c",
    kind="primary",
    sample_idx=0,
    attempt=0,
    final=True,
    ok=True,
    error=None,
    text="x",
    tokens=(TokenLP("x", -0.1, (("x", -0.1), ("y", -3.0))),),
    finish="stop",
    n_gen=1,
    n_prompt=3,
    answer_start=0,
    t_first_token_s=0.1,
    t_total_s=0.2,
    wall_s=0.25,
    backend={
        "model_id": "m",
        "model_hash": "h",
        "quant": "q",
        "kv_quant": "f16",
        "build": "b",
        "template_hash": "t",
    },
    sampling=SamplingParams(),
    config_hash="c",
    prereg_hash="p",
)


def _record(**kw: Any) -> Record:
    return replace(BASE, **kw)


def test_record_roundtrip_with_non_finite_logprobs() -> None:
    toks = (TokenLP("a", float("nan"), None), TokenLP("b", float("-inf"), (("b", float("-inf")),)))
    r = _record(tokens=toks)
    back = Record.from_line(r.to_line())
    assert math.isnan(back.tokens[0].logprob) and back.tokens[1].logprob == -math.inf  # type: ignore[index]
    assert back.to_line() == r.to_line()  # canonical, hash-stable


def test_tampered_record_fails_its_hash() -> None:
    line = _record().to_line()
    bad = line.replace('"text":"x"', '"text":"z"')
    with pytest.raises(CacheCorruptionError):
        Record.from_line(bad)


def test_shard_survives_a_torn_last_line(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    s = CellShard(p)
    s.append(_record(item_id="a"))
    s.append(_record(item_id="b"))
    s.close()
    with p.open("a") as fh:
        fh.write('{"v":1,"item_id":"c","cell')  # process killed mid-write
    s2 = CellShard(p)
    assert s2.n_done() == 2 and s2.is_done("a|primary|0") and not s2.is_done("c|primary|0")
    s2.append(_record(item_id="c"))
    s2.close()
    s3 = CellShard(p)
    assert s3.n_done() == 3
    assert all(Record.from_line(ln) for ln in p.read_text().splitlines())  # file is clean again
    s3.close()


def test_unterminated_but_complete_final_line_is_kept(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text(_record(item_id="a").to_line())  # complete record, newline lost
    s = CellShard(p)
    assert s.is_done("a|primary|0")
    s.append(_record(item_id="b"))
    s.close()
    assert CellShard(p).n_done() == 2


def test_corruption_in_the_middle_is_a_hard_error(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    s = CellShard(p)
    for i in "abc":
        s.append(_record(item_id=i))
    s.close()
    lines = p.read_text().splitlines()
    lines[1] = lines[1].replace('"text":"x"', '"text":"q"')
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(CacheCorruptionError):
        CellShard(p)


# --- runner ------------------------------------------------------------------------


def _cfg_and_cell() -> tuple[ExperimentConfig, CellSpec]:
    cfg = parse_config(pipeline.mock_config(small=True))
    return cfg, cfg.cell("S_q4km")


def _units(n: int = 40) -> list[WorkUnit]:
    items = [i.prompt_view() for i in synthetic_items("triviaqa", n, seed=1)]
    return plan_units(items, params=SamplingParams(max_tokens=32))


def _runner(tmp_path: Path, sim: SimClock) -> tuple[Runner, GenerationCache]:
    cache = GenerationCache(tmp_path / "gen")
    return Runner(cache, "p", "c", max_attempts=3, n_probs=20, clock=sim.now), cache


def _run(
    runner: Runner,
    backend: FakeBackend,
    cell: CellSpec,
    units: list[WorkUnit],
    cid: str,
    shard: CellShard,
) -> RunStats:
    return asyncio.run(runner.run_units(backend, cell, cid, units, shard))


def test_run_is_idempotent_and_skips_done_work(tmp_path: Path) -> None:
    cfg, cell = _cfg_and_cell()
    sim = SimClock()
    prov = FakeProvider(sim, failure_rate=0.0)
    backend = prov.make(cell)
    runner, cache = _runner(tmp_path, sim)
    cid = cell_id(cell, cfg.sampling())
    shard = cache.shard(cid)
    units = _units()
    s1 = _run(runner, backend, cell, units, cid, shard)
    calls = backend.calls
    s2 = _run(runner, backend, cell, units, cid, shard)
    assert (s1.generated, s1.skipped) == (40, 0) and (s2.generated, s2.skipped) == (0, 40)
    assert backend.calls == calls  # no backend call on the second pass
    cache.close()


def test_failed_attempts_are_recorded_and_charged(tmp_path: Path) -> None:
    cfg, cell = _cfg_and_cell()
    sim = SimClock()
    backend = FakeProvider(sim, failure_rate=0.5).make(cell)
    runner, cache = _runner(tmp_path, sim)
    cid = cell_id(cell, cfg.sampling())
    shard = cache.shard(cid)
    st = _run(runner, backend, cell, _units(60), cid, shard)
    assert st.failed_attempts > 10 and st.wasted_wall_s > 0
    recs = list(shard.all_records())
    failed = [r for r in recs if not r.ok]
    assert len(failed) == st.failed_attempts and all(not r.final or r.attempt == 2 for r in failed)
    assert all(r.wall_s > 0 for r in failed)  # failed attempts took (simulated) time
    # per-item wasted time is what the cost model adds on top of the final attempt
    wasted_total = sum(shard.wasted_wall_s(k) for k in shard.item_keys())
    assert wasted_total == pytest.approx(st.wasted_wall_s)
    assert st.total_wall_s == pytest.approx(sum(r.wall_s for r in recs))
    cache.close()


def test_exhausted_retries_leave_a_final_failed_record(tmp_path: Path) -> None:
    cfg, cell = _cfg_and_cell()
    sim = SimClock()
    backend = FakeProvider(sim, failure_rate=1.0).make(cell)
    runner, cache = _runner(tmp_path, sim)
    cid = cell_id(cell, cfg.sampling())
    shard = cache.shard(cid)
    _run(runner, backend, cell, _units(3), cid, shard)
    for k in shard.item_keys():
        fin = shard.final(k)
        assert fin is not None and not fin.ok and fin.attempt == 2 and fin.finish == "error"
    cache.close()


def _canonical(shard_dir: Path) -> list[str]:
    lines: list[str] = []
    for p in sorted(shard_dir.glob("*.jsonl")):
        lines += sorted(p.read_text().splitlines())
    return lines


def test_kill_midway_then_resume_gives_identical_cache(tmp_path: Path) -> None:
    cfg, cell = _cfg_and_cell()
    units = _units(50)
    cid = cell_id(cell, cfg.sampling())

    # uninterrupted reference
    sim_a = SimClock()
    backend_a = FakeProvider(sim_a, failure_rate=0.1).make(cell)
    runner_a, cache_a = _runner(tmp_path / "ref", sim_a)
    _run(runner_a, backend_a, cell, units, cid, cache_a.shard(cid))
    cache_a.close()

    # killed after 17 generations, torn tail, then resumed by a fresh process-like object
    sim_b = SimClock()
    killer = FakeProvider(sim_b, failure_rate=0.1, kill_after=17).make(cell)
    runner_b, cache_b = _runner(tmp_path / "res", sim_b)
    with pytest.raises(SimulatedKillError):
        _run(runner_b, killer, cell, units, cid, cache_b.shard(cid))
    cache_b.close()
    shard_file = tmp_path / "res" / "gen" / f"{cid}.jsonl"
    done_before = len(shard_file.read_text().splitlines())
    assert 0 < done_before < len(_canonical(tmp_path / "ref" / "gen"))
    with shard_file.open("a") as fh:
        fh.write('{"v":1,"item_id":"torn')

    # resume: the simulated clock restarts at the killed value, as wall time would continue
    runner_c, cache_c = _runner(tmp_path / "res", sim_b)
    backend_c = FakeProvider(sim_b, failure_rate=0.1).make(cell)
    st = _run(runner_c, backend_c, cell, units, cid, cache_c.shard(cid))
    cache_c.close()
    assert st.skipped > 0
    ref, res = _canonical(tmp_path / "ref" / "gen"), _canonical(tmp_path / "res" / "gen")

    def strip(line: str) -> dict[str, object]:
        d: dict[str, object] = json.loads(line)
        for k in ("wall_s", "hash"):  # wall time legitimately differs across a restart
            d.pop(k, None)
        return d

    assert [strip(x) for x in ref] == [
        strip(x) for x in res
    ]  # identical results, text and logprobs


def test_hard_kill_of_a_real_process_then_resume(tmp_path: Path) -> None:
    """SIGKILL-equivalent (os._exit mid-run) of the mock CLI, then a clean resume."""
    wd = tmp_path / "w"
    cmd = [sys.executable, "-m", "sieve.eval", "mock", "--small", "--workdir", str(wd)]
    killed = subprocess.run(
        [*cmd, "--kill-after", "150"], capture_output=True, text=True, timeout=300
    )
    assert killed.returncode == 137
    shards = list((wd / "generations").glob("*.jsonl"))
    n_before = sum(len(p.read_text().splitlines()) for p in shards)
    assert n_before > 0 and not (wd / "analysis" / "results.json").exists()
    resumed = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert resumed.returncode == 0, resumed.stderr[-2000:]
    n_after = sum(len(p.read_text().splitlines()) for p in (wd / "generations").glob("*.jsonl"))
    assert n_after > n_before
    # the killed run's lines are still there, byte for byte, in the resumed cache
    for p in shards:
        old = p.read_text().splitlines()
        new = (wd / "generations" / p.name).read_text().splitlines()
        assert set(old) <= set(new)
    # and the fully resumed cache equals an uninterrupted run (up to wall time)
    ref = tmp_path / "ref"
    ok = subprocess.run([*cmd[:-1], str(ref)], capture_output=True, text=True, timeout=600)
    assert ok.returncode == 0

    def canon(root: Path) -> set[str]:
        out = set()
        for p in (root / "generations").glob("*.jsonl"):
            for ln in p.read_text().splitlines():
                d = json.loads(ln)
                d.pop("hash")
                d.pop("wall_s")
                out.add(json.dumps(d, sort_keys=True))
        return out

    assert canon(wd) == canon(ref)

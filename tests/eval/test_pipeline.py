"""End-to-end mock pipeline (synthetic, no network): every stage runs, deterministically."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sieve.calibrate import NEVER, CalTable, evaluate_policy
from sieve.calibrate.store import loads_artifact
from sieve.eval import pipeline
from sieve.eval._fake_backend import FakeProvider, SimClock
from sieve.eval.certify import certify_split
from sieve.eval.data import load_experiment
from sieve.eval.splits import SplitBAlreadyReadError, SplitStore
from sieve.gate import Gate
from sieve.signals import compute_signals
from sieve.types import Action

pytestmark = pytest.mark.slow


def _results(wd: Path) -> dict[str, Any]:
    res: dict[str, Any] = json.loads((wd / "analysis" / "results.json").read_text())
    return res


def test_every_stage_produced_its_artefacts(mock_run: tuple[Path, dict[str, Any]]) -> None:
    wd, out = mock_run
    for rel in (
        "manifest.json",
        "items/triviaqa.jsonl",
        "splits/splits_index.json",
        "energy.jsonl",
        "analysis/results.json",
        "analysis/hypotheses.json",
        "analysis/hypotheses.txt",
        "audit/audit_sheet.csv",
        "audit/audit_key.json",
        "report/index.html",
        "report/tables/methods.csv",
        "report/tables/hypotheses.csv",
        "report/tables/tables.md",
        "report/figures/risk_coverage.png",
        "report/figures/pareto.png",
        "report/figures/reliability.png",
        "report/figures/quantization.png",
    ):
        assert (wd / rel).exists(), rel
    assert set(out["hypotheses"]) == {f"H{i}" for i in range(1, 8)}
    assert all(
        v in ("SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE") for v in out["hypotheses"].values()
    )


def test_everything_is_labelled_synthetic(mock_run: tuple[Path, dict[str, Any]]) -> None:
    wd, _ = mock_run
    res = _results(wd)
    assert res["meta"]["synthetic"] is True and "SYNTHETIC" in res["meta"]["synthetic_label"]
    assert "SYNTHETIC" in (wd / "analysis" / "hypotheses.txt").read_text()
    html = (wd / "report" / "index.html").read_text()
    assert (
        "SYNTHETIC DATA" in html and "Nothing on this page is a result about any real model" in html
    )
    items = [json.loads(ln) for ln in (wd / "items" / "triviaqa.jsonl").read_text().splitlines()]
    assert all("SYNTHETIC" in it["question"] for it in items)
    energy = [json.loads(ln) for ln in (wd / "energy.jsonl").read_text().splitlines()]
    assert {e["state"] for e in energy} == {"synthetic"}
    from sieve.eval.report import _footer

    assert _footer(res).startswith("SYNTHETIC DATA")


def test_split_b_was_read_exactly_once_per_split(mock_run: tuple[Path, dict[str, Any]]) -> None:
    wd, _ = mock_run
    store = SplitStore.load(wd / "splits")
    events = store.ledger.events()
    for k in range(store.n_splits):
        kinds = [e["event"] for e in events if e["k"] == k]
        assert kinds.count("open_b") == 1 and kinds.count("certified") == 1, (k, kinds)
        assert kinds.index("open_b") < kinds.index("certified") < kinds.index("open_c")
        assert store.b_state(k) == "certified"


def test_certifying_a_split_twice_is_refused(mock_run: tuple[Path, dict[str, Any]]) -> None:
    wd, _ = mock_run
    cfg_d = json.loads((wd / "config.json").read_text())
    from sieve.eval import prereg
    from sieve.eval.config import parse_config

    cfg_d["preregistration"]["frozen_sha256"] = prereg.frozen_hash(cfg_d["preregistration"]["path"])
    cfg = parse_config(cfg_d)
    items = pipeline.load_items(wd)
    data = load_experiment(wd, cfg, items, pipeline.make_judge(cfg, mock=True))
    store = SplitStore.load(wd / "splits")
    from sieve.eval.methods import default_methods

    with pytest.raises(SplitBAlreadyReadError):
        certify_split(store, 0, data, [(cfg.main_pair(), default_methods())], cfg)


def test_h1_uses_all_resplits_and_reports_a_ci(mock_run: tuple[Path, dict[str, Any]]) -> None:
    res = _results(mock_run[0])
    h1 = res["h1"]
    assert h1["n_splits"] == 10 and h1["n_certified"] + h1["n_empty"] == 10
    lo, hi = h1["violation_ci95"]
    assert 0 <= lo <= h1["violation_freq"] <= hi <= 1


def test_all_preregistered_methods_are_reported(mock_run: tuple[Path, dict[str, Any]]) -> None:
    res = _results(mock_run[0])
    assert {
        "S_only",
        "L_only",
        "gate:max_softmax",
        "gate:mean_logprob",
        "gate:entropy",
        "gate:margin",
        "random_router",
        "query_router",
        "self_consistency",
        "gate:primary",
    } <= set(res["methods"])
    assert set(res["oracles"]) == {"oracle_router", "oracle_abstainer"}
    assert res["oracles"]["oracle_abstainer"]["risk"] == 0.0
    assert res["comparisons"]["gate_available"] in (True, False)
    c = res["comparisons"]
    if c["gate_available"]:
        assert {"random_router", "query_router"} <= set(c["matched"])
        assert "noninferiority_vs_L_only" in c
        lo, hi = (
            c["vs_L_only"]["cost_per_correct_diff"]["lo"],
            c["vs_L_only"]["cost_per_correct_diff"]["hi"],
        )
        assert lo <= hi


def test_cost_includes_wasted_attempts(mock_run: tuple[Path, dict[str, Any]]) -> None:
    res = _results(mock_run[0])
    acc = res["accuracy"]["S_q4km"]["datasets"]["triviaqa"]
    assert acc["wasted_time_share"] > 0  # injected failures cost time and are in the totals
    assert res["accuracy"]["S_q4km"]["failed_attempts"] > 0


def test_unavailable_judge_is_reported_not_substituted(tmp_path: Path) -> None:
    sim = SimClock()
    d = pipeline.mock_config(small=True, judge="unavailable")
    pipeline.run_mock(tmp_path, cfg_dict=d, provider=FakeProvider(sim), clock=sim, do_report=True)
    res = _results(tmp_path)
    assert res["meta"]["ungraded_datasets"] == ["simpleqa_verified"]
    sq = res["shift"]["simpleqa_verified"]
    assert sq["graded"] is False and sq["risk_ci95_split0"] is None
    hyp = json.loads((tmp_path / "analysis" / "hypotheses.json").read_text())
    h5 = next(v for v in hyp["verdicts"] if v["id"] == "H5")
    assert h5["verdict"] in ("INCONCLUSIVE", "NOT SUPPORTED")
    # structural, not seed-dependent: whatever the synthetic NQ outcome, a verdict is never
    # SUPPORTED without a judged SimpleQA risk, and it always says why
    assert h5["verdict"] != "SUPPORTED" and h5["why"].strip()
    assert h5["numbers"]["simpleqa_verified"]["graded"] is False
    assert res["accuracy"]["S_q4km"]["datasets"]["simpleqa_verified"]["accuracy"] is None


def test_gate_decisions_equal_offline_policy(mock_run: tuple[Path, dict[str, Any]]) -> None:
    """The deployed Gate and the offline LTT policy agree item by item on cached signals."""
    wd, _ = mock_run
    cert = json.loads((wd / "certs" / "split_000.json").read_text())
    c = cert["certs"]["main"]["gate:primary"]
    if c["outcome"] != "feasible":
        pytest.skip("synthetic split 0 certified nothing")
    from sieve.eval.cache import GenerationCache
    from sieve.eval.config import parse_config
    from sieve.eval.runner import cell_id

    cfg_d = json.loads((wd / "config.json").read_text())
    cfg = parse_config(cfg_d)
    items = pipeline.load_items(wd)
    store = SplitStore.load(wd / "splits")
    ids = list(store.a_ids(0))[:60]
    data = load_experiment(wd, cfg, items, pipeline.make_judge(cfg, mock=True))
    art = None
    from sieve.calibrate.store import dumps_artifact  # noqa: F401

    art = loads_artifact(json.dumps(c["artifact"]))
    gate = Gate(art)
    cache = GenerationCache(wd / "generations")
    sh_s = cache.shard(cell_id(cfg.cell("S_q4km"), cfg.sampling()))
    sh_l = cache.shard(cell_id(cfg.cell("L_q4km"), cfg.sampling()))
    s_conf, l_conf, s_dec, l_dec = [], [], [], []
    for iid in ids:
        rs, rl = sh_s.final(f"{iid}|primary|0"), sh_l.final(f"{iid}|primary|0")
        assert rs is not None and rl is not None
        sig_s = compute_signals(rs.to_completion(), rs.backend_info())
        sig_l = compute_signals(rl.to_completion(), rl.backend_info())
        s_conf.append(sig_s.score(art.tiers[0].signal))
        l_conf.append(sig_l.score(art.tiers[1].signal))
        s_dec.append(gate.decide(sig_s, 1))
        l_dec.append(gate.decide(sig_l, 2))
    cache.close()
    t1 = NEVER if c["t1"] is None else c["t1"]
    t2 = NEVER if c["t2"] is None else c["t2"]
    zeros = np.zeros(len(ids), dtype=bool)
    table = CalTable(
        np.array([np.nan if v is None else v for v in s_conf]),
        np.array([np.nan if v is None else v for v in l_conf]),
        zeros,
        zeros,
    )
    off = evaluate_policy(table, t1, t2)
    by_s_gate = np.array([d.action is Action.ANSWER for d in s_dec])
    by_l_gate = np.array(
        [
            (s.action is not Action.ANSWER) and d.action is Action.ANSWER
            for s, d in zip(s_dec, l_dec, strict=True)
        ]
    )
    assert np.array_equal(by_s_gate, off.by_s) and np.array_equal(by_l_gate, off.by_l)
    assert data.cells["S_q4km"].tier == "S"


def test_artifact_is_hash_verified_and_keyed_to_the_cells(
    mock_run: tuple[Path, dict[str, Any]],
) -> None:
    wd, _ = mock_run
    cert = json.loads((wd / "certs" / "split_000.json").read_text())
    c = cert["certs"]["main"]["gate:primary"]
    if c["outcome"] != "feasible":
        pytest.skip("synthetic split 0 certified nothing")
    art = loads_artifact(json.dumps(c["artifact"]))  # verifies the content hash
    assert (
        art.tiers[0].identity.model_id == "synthetic-S"
        and art.tiers[1].identity.model_id == "synthetic-L"
    )
    assert art.method == "fixed_sequence" and art.risk_definition == "selective"
    bad = json.loads(json.dumps(c["artifact"]))
    bad["alpha"] = 0.5
    with pytest.raises(Exception, match="hash"):
        loads_artifact(json.dumps(bad))


def test_results_follow_the_published_schema(mock_run: tuple[Path, dict[str, Any]]) -> None:
    schema = json.loads(
        (Path(__file__).parents[2] / "results" / "schema" / "results.schema.json").read_text()
    )
    res = _results(mock_run[0])
    for key in schema["required"]:
        assert key in res, key
    for key, sub in schema["properties"].items():
        if "required" in sub:
            for k2 in sub["required"]:
                assert k2 in res[key], (key, k2)


def test_html_is_self_contained_and_accessible(mock_run: tuple[Path, dict[str, Any]]) -> None:
    html = (mock_run[0] / "report" / "index.html").read_text()
    assert html.startswith("<!doctype html>") and '<html lang="en">' in html
    assert "<script" not in html and "<link" not in html
    assert not re.search(r"""(src|href)\s*=\s*["']https?://""", html)  # no external requests
    assert "@import" not in html and "url(http" not in html
    assert html.count("<img ") == html.count('alt="') >= 4
    assert "<caption>" in html and 'scope="col"' in html and "prefers-color-scheme" in html
    assert len(html) < 3_000_000


def test_audit_sheet_comes_from_split_a_only(mock_run: tuple[Path, dict[str, Any]]) -> None:
    wd, _ = mock_run
    store = SplitStore.load(wd / "splits")
    key = json.loads((wd / "audit" / "audit_key.json").read_text())
    a = set(store.a_ids(0))
    assert key["rows"] and all(m["item_id"] in a for m in key["rows"].values())


def test_cli_subcommands_exist() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "sieve.eval", "--help"], capture_output=True, text=True
    )
    assert out.returncode == 0
    for sub in ("prepare-data", "run", "analyse", "report", "audit-sheet", "mock"):
        assert sub in out.stdout


def _strip_volatile(res: dict[str, Any]) -> dict[str, Any]:
    r: dict[str, Any] = json.loads(json.dumps(res))
    r["meta"].pop("commit", None)
    return r


def test_full_mock_pipeline_is_deterministic(
    mock_run: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    wd, _ = mock_run
    sim = SimClock()
    pipeline.run_mock(
        tmp_path, cfg_dict=pipeline.mock_config(small=True), provider=FakeProvider(sim), clock=sim
    )
    assert _strip_volatile(_results(wd)) == _strip_volatile(_results(tmp_path))
    for name in ("methods", "comparisons_gate_minus_other", "shift", "quantization", "hypotheses"):
        a = (wd / "report" / "tables" / f"{name}.csv").read_text()
        b = (tmp_path / "report" / "tables" / f"{name}.csv").read_text()
        assert a == b, name
    assert (wd / "analysis" / "hypotheses.txt").read_text() == (
        tmp_path / "analysis" / "hypotheses.txt"
    ).read_text()
    # cache lines are identical including wall times (simulated clock) and hashes
    for p in sorted((wd / "generations").glob("*.jsonl")):
        assert p.read_text() == (tmp_path / "generations" / p.name).read_text()


def test_rerunning_analysis_reuses_certifications_and_never_rereads_b(
    mock_run: tuple[Path, dict[str, Any]],
) -> None:
    wd, _ = mock_run
    before = (
        (wd / "split_access_ledger.jsonl")
        if False
        else (wd / "splits" / "split_access_ledger.jsonl")
    )
    n_before = len(before.read_text().splitlines())
    first = (wd / "analysis" / "results.json").read_text()
    from sieve.eval import prereg
    from sieve.eval.analysis import run_analysis
    from sieve.eval.config import parse_config

    d = json.loads((wd / "config.json").read_text())
    d["preregistration"]["frozen_sha256"] = prereg.frozen_hash(d["preregistration"]["path"])
    cfg = parse_config(d)
    run_analysis(wd, cfg, pipeline.load_items(wd), pipeline.make_judge(cfg, mock=True))
    ledger = before.read_text().splitlines()
    assert sum(1 for ln in ledger if '"open_b"' in ln) == 10  # still one per split
    assert len(ledger) > n_before  # only C (and result reads) were logged again
    assert (wd / "analysis" / "results.json").read_text() == first


def test_results_csv_tables_parse(mock_run: tuple[Path, dict[str, Any]]) -> None:
    for p in (mock_run[0] / "report" / "tables").glob("*.csv"):
        rows = list(csv.reader(p.open()))
        assert rows and all(len(r) == len(rows[0]) for r in rows), p.name


def test_comparisons_include_coverage_aware_noninferiority(
    mock_run: tuple[Path, dict[str, Any]],
) -> None:
    c = _results(mock_run[0])["comparisons"]
    if not c.get("gate_available") or "noninferiority_vs_L_only" not in c:
        pytest.skip("synthetic split 0 has no certified gate")
    ni = c["noninferiority_vs_L_only"]
    assert "correct_answer_rate_diff" in ni and "acc_on_answered_non_inferior" in ni


def test_analysis_survives_truncated_generations(tmp_path: Path) -> None:
    """Real T4 run: 0.4% truncated items in the main cell and a collapsed ablation arm (97%).

    Truncated generations carry no signals (fail closed); the analysis must report them, not crash.
    """
    sim = SimClock()
    d = pipeline.mock_config(small=True)
    for c in d["cells"]:
        rate = {"S_q4km": 0.03, "S_q4km_kvq4": 0.97}.get(c["name"])
        if rate is not None:
            c.setdefault("extra", {}).setdefault("fake", {})["truncate_rate"] = rate
    pipeline.run_mock(tmp_path, cfg_dict=d, provider=FakeProvider(sim), clock=sim, do_report=True)
    res = _results(tmp_path)
    assert "curves" in res and res["accuracy"]["S_q4km_kvq4"]

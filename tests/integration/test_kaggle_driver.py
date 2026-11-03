"""The generated Kaggle experiment script: build determinism, embedded repo, helpers, dry run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
KAGGLE = ROOT / "kaggle"
FAKE = Path(__file__).with_name("fake_llama_server.py")


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return _load(KAGGLE / "build_experiment_script.py", "sieve_kaggle_builder")


@pytest.fixture(scope="module")
def decoder() -> ModuleType:
    return _load(KAGGLE / "decode_log_bundle.py", "sieve_kaggle_decoder")


def _names(blob: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        return tf.getnames()


def args(builder: ModuleType, **kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "config": "t4_integration",
        "workdir_name": None,
        "cells": None,
        "stages": builder.DEFAULT_STAGES,
        "max_minutes": 40.0,
        "reserve_minutes": 8.0,
        "sanity_per_dataset": 6,
        "allow_build": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture(scope="module")
def built(builder: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ModuleType]:
    p = tmp_path_factory.mktemp("kg") / "run_experiment.py"
    p.write_text(builder.build(args(builder)), encoding="utf-8")
    return p, _load(p, "sieve_kaggle_run_experiment")


def test_build_is_deterministic_and_embeds_the_repo(builder: ModuleType) -> None:
    a = builder.build(args(builder))
    b = builder.build(args(builder))
    assert a == b
    compile(a, "run_experiment.py", "exec")
    tgz = builder.make_tgz(builder.repo_files())
    names = _names(tgz)
    for need in (
        "pyproject.toml",
        "docs/PREREGISTRATION.md",
        "README.md",
        "LICENSE",
        "src/sieve/eval/providers/llamacpp.py",
        "src/sieve/eval/prompts.py",
        "benchmarks/t4_integration.yaml",
        "benchmarks/t4_main.yaml",
    ):
        assert need in names, need
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)
    assert len(tgz) < 400_000  # keeps the pushed script small


def test_committed_script_embeds_current_sources(builder: ModuleType) -> None:
    """Fails when src/ benchmarks/ changed without regenerating kaggle/run_experiment.py."""
    committed = KAGGLE / "run_experiment.py"
    if not committed.exists():
        pytest.skip("kaggle/run_experiment.py not generated")
    want = hashlib.sha256(builder.make_tgz(builder.repo_files())).hexdigest()
    text = committed.read_text(encoding="utf-8")
    assert f'"embedded_sha256": "{want}"' in text, (
        "stale kaggle/run_experiment.py: run python kaggle/build_experiment_script.py"
    )


def test_params_and_overrides(
    built: tuple[Path, ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, m = built
    p = m.load_params()
    assert p["config"] == "t4_integration" and p["workdir_name"] == "sieve_work_t4_integration"
    assert p["llama_cpp_commit"].startswith("fac3a89") and p["max_minutes"] == 40.0
    monkeypatch.setenv("SIEVE_CONFIG", "t4_main")
    monkeypatch.setenv("SIEVE_STAGES", "bootstrap,emit")
    assert m.load_params()["config"] == "t4_main" and m.load_params()["stages"] == [
        "bootstrap",
        "emit",
    ]


def test_template_refuses_to_run_unbuilt() -> None:
    m = _load(KAGGLE / "experiment_driver.py", "sieve_kaggle_template")
    with pytest.raises(SystemExit, match="template not built"):
        m.load_params()


def test_download_plan_for_main_config(built: tuple[Path, ModuleType]) -> None:
    _, m = built
    cfg = yaml.safe_load((ROOT / "benchmarks" / "t4_main.yaml").read_text())
    plan = m.plan_downloads(cfg, None)
    files = {f for _, f in plan}
    assert "Qwen2.5-7B-Instruct-Q4_K_M.gguf" in files
    # small tier is Qwen2.5-1.5B (Apache-2.0), whose f16 GGUF is a single file (D-020)
    assert "qwen2.5-1.5b-instruct-fp16.gguf" in files
    assert not any("3b" in f for f in files)
    assert ("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf") in plan
    assert len(plan) == len(set(plan))
    only = m.plan_downloads(cfg, ["L_q4km"])
    assert only == [("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf")]


@pytest.mark.parametrize("name", ["t4_main", "t4_smoke", "t4_fallback_1p5b", "t4_integration"])
def test_benchmark_configs_load_and_name_a_resolvable_provider(name: str) -> None:
    from sieve.eval import prereg
    from sieve.eval.config import load_config

    cfg = load_config(ROOT / "benchmarks" / f"{name}.yaml")
    assert prereg.verify_unchanged(cfg.prereg_path, cfg.prereg_frozen_sha256)
    assert cfg.raw["backend"]["provider"] == "sieve.eval.providers.llamacpp:make_provider"
    assert len(cfg.raw["backend"]["llama_cpp_commit"]) == 40
    for c in cfg.cells:
        repo, fn = str(c.extra["gguf"]).split(":", 1)
        assert "/" in repo and fn.endswith(".gguf")


def test_bundle_round_trip_and_corruption_detection(
    built: tuple[Path, ModuleType], decoder: ModuleType, tmp_path: Path
) -> None:
    _, m = built
    base = tmp_path / "wd"
    (base / "cells").mkdir(parents=True)
    (base / "sanity.json").write_text(json.dumps({"x": list(range(5000))}))
    (base / "cells" / "a.json").write_text('{"cell_id": "a"}')
    big = base / "analysis"
    big.mkdir()
    (big / "results.json").write_text(os.urandom(300_000).hex())  # incompressible: over the cap
    files = m.collect_bundle_files(base) + [base / "sanity.json"][:0]
    blob = m.make_bundle(files, base, cap_bytes=50_000)
    lines: list[str] = []
    m.emit_bundle("unit", blob, out=lines.append)
    assert lines[0].startswith("SIEVE_BUNDLE_BEGIN name=unit") and lines[-1].endswith("name=unit")
    assert all(len(ln) < m.CHUNK + 60 for ln in lines)
    text = "\n[00:01] noise\n" + "\n".join(lines) + "\nmore noise"
    got = decoder.decode(text)["unit"]
    assert got == blob
    names = _names(got)
    assert "sanity.json" in names and "analysis/results.json" not in names  # cap drops the big one
    broken = text.replace(lines[2][-20:], "A" * 20, 1)
    with pytest.raises(ValueError, match="checksum"):
        decoder.decode(broken)
    with pytest.raises(ValueError, match="missing chunks"):
        decoder.decode("\n".join(x for x in lines if "0002/" not in x))


def test_restore_previous_workdir_for_resume(
    built: tuple[Path, ModuleType], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, m = built
    prev = tmp_path / "input" / "prev-kernel" / "sieve_work_t4_integration"
    (prev / "generations").mkdir(parents=True)
    (prev / "manifest.json").write_text("{}")
    (prev / "generations" / "c1.jsonl").write_text('{"a":1}\n{"a":2}\n')
    monkeypatch.setattr(m, "INPUT_ROOT", tmp_path / "input")
    wd = tmp_path / "work" / "sieve_work_t4_integration"
    m.restore_previous(wd, "sieve_work_t4_integration")
    assert (wd / "generations" / "c1.jsonl").read_text() == '{"a":1}\n{"a":2}\n'
    assert "prev-kernel" in m.SUMMARY["resumed_from"]
    # an existing workdir is never overwritten
    (wd / "generations" / "c1.jsonl").write_text("local\n")
    m.restore_previous(wd, "sieve_work_t4_integration")
    assert (wd / "generations" / "c1.jsonl").read_text() == "local\n"


def test_find_files_respects_depth(built: tuple[Path, ModuleType], tmp_path: Path) -> None:
    _, m = built
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "llama-server").write_text("x")
    deep = tmp_path / "a" / "b" / "c" / "d" / "e" / "f"
    deep.mkdir(parents=True)
    (deep / "llama-server").write_text("x")
    found = m.find_files("llama-server", [str(tmp_path)], max_depth=4)
    assert found == [tmp_path / "a" / "b" / "llama-server"]


@pytest.mark.slow
def test_driver_dry_run_with_stand_in_server(
    built: tuple[Path, ModuleType], decoder: ModuleType, tmp_path: Path
) -> None:
    """Whole driver, as a subprocess: synthetic data, stand-in llama-server, no GPU/network."""
    script, _ = built
    (tmp_path / "models").mkdir()
    for f in ("qwen2.5-1.5b-instruct-q4_k_m.gguf", "Qwen2.5-7B-Instruct-Q4_K_M.gguf"):
        (tmp_path / "models" / f).write_bytes(b"GGUF" + f.encode())
    srv = tmp_path / "llama-server"
    srv.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    srv.chmod(srv.stat().st_mode | stat.S_IEXEC)
    p = script
    env = dict(os.environ)
    env.update(
        SIEVE_SKIP_PIP="1",
        SIEVE_SYNTHETIC_DATA="1",
        SIEVE_STAGES="bootstrap,binary,prepare,run,sanity,emit",
        SIEVE_LLAMA_SERVER=str(srv),
        SIEVE_MODELS_DIR=str(tmp_path / "models"),
        PYTHONPATH=str(ROOT / "src"),
    )
    r = subprocess.run(
        [sys.executable, str(p)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "SIEVE_SUMMARY_BEGIN" in r.stdout and '"problems": 0' in r.stdout
    got = decoder.decode(r.stdout)["t4_integration"]
    names = _names(got)
    assert "sanity.json" in names and "driver_summary.md" in names
    # second run on the same workdir resumes: nothing is regenerated
    r2 = subprocess.run(
        [sys.executable, str(p)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600
    )
    assert r2.returncode == 0, r2.stdout[-3000:]
    assert "prepare-data already done" in r2.stdout
    assert '"generated": 0' in r2.stdout and '"skipped"' in r2.stdout


def test_every_config_preregistration_is_embedded_in_the_kaggle_script() -> None:
    """A config pinned to a pre-registration the kernel never receives would fail at run time."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("b", KAGGLE / "build_experiment_script.py")
    assert spec is not None and spec.loader is not None
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    embedded = {str(p.relative_to(b.ROOT)) for p in b.repo_files()}
    for cfg_path in sorted((ROOT / "benchmarks").glob("*.yaml")):
        pin = yaml.safe_load(cfg_path.read_text()).get("preregistration", {}).get("path")
        assert pin in embedded, f"{cfg_path.name} pins {pin}, which the kernel would not have"

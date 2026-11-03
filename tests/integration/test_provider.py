"""Server manager and eval provider against a stand-in llama-server (real response schema)."""

from __future__ import annotations

import json
import os
import stat
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from sieve.backends.base import BackendError
from sieve.eval import pipeline, prompts
from sieve.eval.config import CellSpec, ExperimentConfig, parse_config
from sieve.eval.providers import (
    LlamaCppProvider,
    LlamaServerProcess,
    ProviderError,
    ServerSpec,
    ServerStartError,
    find_free_port,
)
from sieve.eval.providers.llamacpp import file_sha256, shard_files, trim_to_text
from sieve.types import Completion, SamplingParams, TokenLP

FAKE = Path(__file__).with_name("fake_llama_server.py")
GREEDY = SamplingParams(temperature=0.0, max_tokens=32)


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    b = tmp_path / "llama-server"
    b.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    b.chmod(b.stat().st_mode | stat.S_IEXEC)
    return b


@pytest.fixture
def model(tmp_path: Path) -> Path:
    m = tmp_path / "models" / "tiny-q4_k_m.gguf"
    m.parent.mkdir()
    m.write_bytes(b"GGUF" + b"\0" * 1000)
    return m


def cfg_for(tmp_path: Path, **backend: Any) -> ExperimentConfig:
    d = pipeline.mock_config(small=True)
    d["synthetic"] = False
    d["preregistration"] = {"path": "docs/PREREGISTRATION.md", "frozen_sha256": None}
    d["backend"] = {"llama_cpp_commit": "fac3a894fc04901c3e5abaff457e024c6b52e04f", **backend}
    return parse_config(d)


def cell(**kw: Any) -> CellSpec:
    base: dict[str, Any] = {
        "name": "S_q4km",
        "tier": "S",
        "model_id": "Tiny",
        "quant": "Q4_K_M",
        "kv_quant": "f16",
        "datasets": ("triviaqa",),
        "extra": {"gguf": "owner/repo:tiny-q4_k_m.gguf"},
    }
    base.update(kw)
    return CellSpec(**base)


def provider(tmp_path: Path, fake_bin: Path, **kw: Any) -> LlamaCppProvider:
    return LlamaCppProvider(
        cfg_for(tmp_path, startup_timeout_s=30),
        binary=fake_bin,
        models_dir=tmp_path / "models",
        log_dir=tmp_path / "logs",
        gpus=kw.pop("gpus", {"S": "0", "L": "1"}),
        **kw,
    )


# --- server manager --------------------------------------------------------------------


def test_free_port_is_bindable_and_distinct() -> None:
    a = find_free_port()
    assert 0 < a < 65536
    assert find_free_port(a + 1) != a


def test_argv_flags_for_kv_and_flash_attn(tmp_path: Path) -> None:
    sp = ServerSpec(binary=Path("/x/llama-server"), model=Path("/m.gguf"), log_path=tmp_path / "l")
    a = sp.argv(1234)
    assert a[:3] == ["/x/llama-server", "-m", "/m.gguf"] and "-fa" not in a and "-ctk" not in a
    q = ServerSpec(binary=Path("b"), model=Path("m"), log_path=tmp_path / "l", kv_type="q8_0").argv(
        1
    )
    assert q[q.index("-fa") + 1] == "on" and q[q.index("-ctk") + 1] == "q8_0"
    assert q[q.index("-ctv") + 1] == "q8_0"
    with pytest.raises(ValueError, match="flash attention"):
        ServerSpec(
            binary=Path("b"),
            model=Path("m"),
            log_path=tmp_path / "l",
            kv_type="q4_0",
            flash_attn="off",
        ).argv(1)


def test_process_start_health_props_gpu_pin_log_and_stop(
    tmp_path: Path, fake_bin: Path, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_LLAMA_LOAD_S", "1.0")
    spec = ServerSpec(binary=fake_bin, model=model, log_path=tmp_path / "logs" / "s.log", gpu="1")
    with LlamaServerProcess(spec) as p:
        assert p.alive() and p.start_s >= 0.9  # waited for 503 -> 200
        props = p.props()
        assert props["fake_cuda_visible_devices"] == "1"
        assert "-ngl" in props["fake_argv"] and str(p.port) in props["fake_argv"]
        pid = p.proc.pid if p.proc else 0
    assert not p.alive()
    with pytest.raises(OSError):
        os.kill(pid, 0)  # process really gone
    log = (tmp_path / "logs" / "s.log").read_text()
    assert log.startswith("# argv:") and "CUDA_VISIBLE_DEVICES=1" in log
    assert "flash_attn" in log  # server stdout captured
    with pytest.raises((urllib.error.URLError, OSError)):
        urllib.request.urlopen(f"http://127.0.0.1:{p.port}/health", timeout=1)


def test_start_failure_reports_log_tail_and_cleans_up(
    tmp_path: Path, fake_bin: Path, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_LLAMA_DIE", "3")
    spec = ServerSpec(binary=fake_bin, model=model, log_path=tmp_path / "s.log")
    with pytest.raises(ServerStartError, match=r"(?s)exited with code 3.*simulated load failure"):
        LlamaServerProcess(spec).start()


def test_health_timeout_stops_the_process(
    tmp_path: Path, fake_bin: Path, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_LLAMA_LOAD_S", "60")
    spec = ServerSpec(
        binary=fake_bin, model=model, log_path=tmp_path / "s.log", startup_timeout_s=1
    )
    p = LlamaServerProcess(spec)
    with pytest.raises(ServerStartError, match="not healthy"):
        p.start()
    assert not p.alive()


def test_missing_binary_and_model_are_explicit(tmp_path: Path, fake_bin: Path, model: Path) -> None:
    with pytest.raises(ServerStartError, match="binary not found"):
        LlamaServerProcess(
            ServerSpec(binary=tmp_path / "nope", model=model, log_path=tmp_path / "l")
        ).start()
    with pytest.raises(ServerStartError, match="model file not found"):
        LlamaServerProcess(
            ServerSpec(binary=fake_bin, model=tmp_path / "nope.gguf", log_path=tmp_path / "l")
        ).start()


# --- trim_to_text ----------------------------------------------------------------------


def _c(text: str, toks: list[str], n_gen: int) -> Completion:
    return Completion(
        text, tuple(TokenLP(t, -0.1, ((t, -0.1),)) for t in toks), "stop", n_gen, GREEDY
    )


def test_trim_removes_trailing_token_beyond_text_and_adjusts_n_gen() -> None:
    c = trim_to_text(_c(" Paris", [" Paris", "\n"], 2))
    assert c.tokens is not None and [t.token for t in c.tokens] == [" Paris"] and c.n_gen == 1


def test_trim_measured_llamacpp_stop_behaviour() -> None:
    """Real T4 diagnostic: stop '\\n' hit on token '.\\n': content keeps '.', no entry, n_gen+1."""
    toks = [" The", " capital", " is", " Paris"]
    c = _c(" The capital is Paris.", toks, len(toks) + 1)
    out = trim_to_text(c, stopped_by_word=True)
    assert out.text == " The capital is Paris" and out.n_gen == 4 and out.tokens == c.tokens
    clean = trim_to_text(_c(" Au", [" Au"], 2), stopped_by_word=True)  # stop token itself
    assert clean.text == " Au" and clean.n_gen == 1
    # without a stop-word hit a count mismatch is a defect, not something to repair
    c2 = _c(" Au", [" Au"], 2)
    assert trim_to_text(c2) is c2
    # a missing token in the middle is not repaired even with a stop word
    gap = _c(" a b c", [" a", " c"], 3)
    assert trim_to_text(gap, stopped_by_word=True) is gap


def test_trim_leaves_untrimmable_and_clean_completions_alone() -> None:
    clean = _c(" Paris", [" Paris"], 1)
    assert trim_to_text(clean) is clean
    gap = _c(" Paris", [" Par", "\n"], 2)  # dropped token: stays visible to validate_completion
    assert trim_to_text(gap) is gap
    assert trim_to_text(Completion("x", None, "stop", 1, GREEDY)).tokens is None


def test_trim_keeps_n_gen_when_backend_count_disagreed() -> None:
    c = trim_to_text(_c(" Paris", [" Paris", "\n"], 7))
    assert c.n_gen == 7  # a disagreement is data, not something to repair


# --- file hashing / shards ---------------------------------------------------------------


def test_sha256_matches_and_is_cached(tmp_path: Path, model: Path) -> None:
    import hashlib

    h = file_sha256(model)
    assert h == hashlib.sha256(model.read_bytes()).hexdigest()
    assert model.with_name(model.name + ".sha256.json").exists()
    assert file_sha256(model) == h


def test_shard_files() -> None:
    first = Path("/m/q4_k_m-00001-of-00003.gguf")
    assert [p.name for p in shard_files(first)] == [
        "q4_k_m-00001-of-00003.gguf",
        "q4_k_m-00002-of-00003.gguf",
        "q4_k_m-00003-of-00003.gguf",
    ]
    assert shard_files(Path("/m/x.gguf")) == [Path("/m/x.gguf")]


# --- provider ------------------------------------------------------------------------------


def test_provider_open_identity_generate_and_sidecars(
    tmp_path: Path, fake_bin: Path, model: Path
) -> None:
    import asyncio

    prov = provider(tmp_path, fake_bin)
    c = cell(kv_quant="q8_0")
    with prov.open(c) as be:
        info = be.info
        assert info.is_complete()
        assert info.model_hash == file_sha256(model)
        assert info.quant == "Q4_K_M" and info.kv_quant == "q8_0"
        assert info.template_hash == prompts.template_hash()
        assert "llama.cpp@fac3a894fc04901c3e5abaff457e024c6b52e04f" in info.build
        assert "b9999-fac3a89" in info.build and "fa=on" in info.build  # quantised KV forces FA
        assert "stop=" in info.build
        comp = asyncio.run(
            be.generate(
                prompts.QA_TEMPLATE.format(question="What is the capital of France?"),
                GREEDY,
                n_probs=20,
            )
        )
        # second asyncio.run: a different event loop reuses the same backend
        comp2 = asyncio.run(be.generate("Question: x\nAnswer:", GREEDY, n_probs=20))
    assert comp.text == " Paris" and comp.finish == "stop"
    assert comp.tokens is not None and len(comp.tokens) == comp.n_gen == 1  # stop token trimmed
    assert comp.tokens[0].top is not None and len(comp.tokens[0].top) == 20
    assert comp2.tokens is not None
    side = json.loads((tmp_path / "logs" / "S_q4km.server.json").read_text())
    assert side["gpu"] == "0" and side["log_flash_attn"] == "enabled"
    assert all(t["ok"] for t in side["self_test"]) and "-ctk" in side["argv"]
    tl = [
        json.loads(x) for x in (tmp_path / "logs" / "S_q4km.timings.jsonl").read_text().splitlines()
    ]
    assert len(tl) == 2 and {"wall_s", "prompt_ms", "predicted_ms"} <= set(tl[-1])


def test_gpu_pinning_by_tier(tmp_path: Path, fake_bin: Path, model: Path) -> None:
    prov = provider(tmp_path, fake_bin)
    with prov.open(cell(name="S_a")), prov.open(cell(name="L_a", tier="L")):
        pass
    a = json.loads((tmp_path / "logs" / "S_a.server.json").read_text())
    b = json.loads((tmp_path / "logs" / "L_a.server.json").read_text())
    assert (a["gpu"], b["gpu"]) == ("0", "1") and a["port"] != b["port"]


def test_provider_fails_closed_on_logprob_defect(
    tmp_path: Path, fake_bin: Path, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_LLAMA_NAN", "1")
    with (
        pytest.raises(ProviderError, match="self-test failed"),
        provider(tmp_path, fake_bin).open(cell()),
    ):
        raise AssertionError("must not open")
    # the server was stopped and the failure is recorded for the audit trail
    side = json.loads((tmp_path / "logs" / "S_q4km.server.json").read_text())
    assert any(not c["ok"] for c in side["self_test"])


def test_provider_refuses_build_mismatch_and_unpinned_commit(
    tmp_path: Path, fake_bin: Path, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_LLAMA_BUILD", "b1234-deadbee")
    with (
        pytest.raises(ProviderError, match="does not match pinned commit"),
        provider(tmp_path, fake_bin).open(cell()),
    ):
        raise AssertionError
    monkeypatch.delenv("FAKE_LLAMA_BUILD")
    prov = LlamaCppProvider(
        cfg_for(tmp_path, llama_cpp_commit="PIN-ME"),
        binary=fake_bin,
        models_dir=tmp_path / "models",
        log_dir=tmp_path / "logs",
    )
    with pytest.raises(ProviderError, match="not pinned"), prov.open(cell()):
        raise AssertionError


def test_provider_missing_model_and_bad_spec(tmp_path: Path, fake_bin: Path, model: Path) -> None:
    prov = provider(tmp_path, fake_bin)
    with (
        pytest.raises(ProviderError, match="missing"),
        prov.open(cell(extra={"gguf": "o/r:absent.gguf"})),
    ):
        raise AssertionError
    with pytest.raises(ProviderError, match=r"extra\.gguf"):
        prov.resolve_model(cell(extra={}))


def test_provider_start_failure_is_provider_error(
    tmp_path: Path, fake_bin: Path, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_LLAMA_DIE", "9")
    with (
        pytest.raises(ProviderError, match="exited with code 9"),
        provider(tmp_path, fake_bin).open(cell()),
    ):
        raise AssertionError


@pytest.mark.parametrize("mode", ["drop", "straddle", "keep"])
def test_every_stop_token_behaviour_validates_after_trim(
    tmp_path: Path, fake_bin: Path, model: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """drop = measured llama.cpp behaviour (stop token counted, entry missing); the self-test in
    ``open`` already runs through the same path, so opening at all proves it is accepted."""
    import asyncio

    from sieve.signals import validate_completion

    monkeypatch.setenv("FAKE_LLAMA_STOP_MODE", mode)
    with provider(tmp_path, fake_bin).open(cell(name=f"s_{mode}")) as be:
        c = asyncio.run(be.generate("Question: capital of France\nAnswer:", GREEDY, n_probs=20))
    assert validate_completion(c) == ()
    assert c.text == " Paris" and c.tokens is not None and len(c.tokens) == c.n_gen == 1


def test_backend_error_when_server_dies_mid_run(
    tmp_path: Path, fake_bin: Path, model: Path
) -> None:
    import asyncio

    with provider(tmp_path, fake_bin).open(cell()) as be:
        # point the backend at a dead port to model a server crash
        be._base_url = "http://127.0.0.1:1"  # type: ignore[attr-defined]
        be._inner = None  # type: ignore[attr-defined]
        with pytest.raises(BackendError):
            asyncio.run(be.generate("p", GREEDY, n_probs=5))

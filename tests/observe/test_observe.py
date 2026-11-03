from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from prometheus_client import generate_latest

from sieve.observe import (
    AuditRecord,
    DriftMonitor,
    JsonlAuditSink,
    MemoryAuditSink,
    PinMismatchError,
    SieveMetrics,
    UserHasher,
    apply_content_policy,
    bounded,
    check_pins,
    config_pin,
    default_redactors,
    load_reference,
    regex_redactor,
    summarize_shadow,
)
from sieve.observe.audit import read_jsonl
from sieve.observe.metrics import DECISIONS, REASONS
from sieve.types import ReasonCode


def rec(**kw: Any) -> AuditRecord:
    base: dict[str, Any] = {
        "ts": 1.0,
        "request_id": "r1",
        "mode": "enforce",
        "outcome": "ok",
        "decision": "answer",
        "reason": "confident",
        "served_tier": 1,
        "escalated": False,
        "artifact_id": "a" * 16,
        "stream": False,
        "latency_s": 0.1,
    }
    base.update(kw)
    return AuditRecord(**base)


# --- audit -------------------------------------------------------------------------------


def test_record_has_decision_metadata_and_no_content_by_default() -> None:
    r = rec(tiers=[{"tier": 1, "score": -0.1, "signals": {"mean_logprob": -0.1}}])
    d = json.loads(r.to_json())
    for k in ("decision", "reason", "served_tier", "artifact_id", "tiers", "backend", "cost"):
        assert k in d
    assert "prompt" not in d and "response" not in d


def test_newlines_in_fields_cannot_forge_log_lines() -> None:
    r = rec(reason='x\n{"decision":"forged"}')
    line = r.to_json()
    assert "\n" not in line and json.loads(line)["reason"].startswith("x\n")


def test_content_policy_and_redaction() -> None:
    r = apply_content_policy(
        rec(),
        log_content=True,
        prompt="mail bob@x.org",
        response="ok",
        redactors=default_redactors(),
    )
    assert r.prompt == "mail [EMAIL]" and r.response == "ok"
    r2 = apply_content_policy(rec(), log_content=False, prompt="secret", response="secret")
    assert r2.prompt is None and r2.response is None
    custom = regex_redactor(r"\d{3}-\d{2}-\d{4}", "[SSN]")
    assert custom("123-45-6789") == "[SSN]"
    assert default_redactors()[2]("card 4111 1111 1111 1111") == "card [NUMBER]"


def test_user_hasher_is_salted_and_stable() -> None:
    h1, h2 = UserHasher(b"salt-a"), UserHasher(b"salt-b")
    assert h1("alice") == h1("alice") != h2("alice") and h1(None) is None and h1("") is None
    assert "alice" not in (h1("alice") or "")


def test_jsonl_sink_writes_one_line_per_record(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    s = JsonlAuditSink(str(p))
    for i in range(25):
        s.write(rec(request_id=f"r{i}"))
    s.flush()
    s.close()
    rows = read_jsonl(str(p))
    assert [r["request_id"] for r in rows] == [f"r{i}" for i in range(25)]
    s.write(rec())  # after close: dropped, not raised
    assert s.dropped == 1


def test_jsonl_sink_never_blocks_and_counts_drops(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    s = JsonlAuditSink(str(p), max_queue=2)
    gate = threading.Event()

    class SlowFile:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def write(self, x: str) -> None:
            gate.wait(5)
            self.lines.append(x)

        def flush(self) -> None: ...
        def close(self) -> None: ...

    slow = SlowFile()
    s._fh = slow  # type: ignore[assignment]
    for i in range(20):
        s.write(rec(request_id=f"r{i}"))  # must return immediately even though writer is stuck
    assert s.dropped >= 15
    gate.set()
    s.flush()
    assert len(slow.lines) + s.dropped == 20
    s.close()


def test_write_failures_are_counted_not_raised(tmp_path: Path) -> None:
    s = JsonlAuditSink(str(tmp_path / "a.jsonl"))

    class Broken:
        def write(self, x: str) -> None:
            raise OSError("disk full")

        def flush(self) -> None: ...
        def close(self) -> None: ...

    s._fh = Broken()  # type: ignore[assignment]
    s.write(rec())
    s.flush()
    assert s.errors == 1
    s.close()


def test_summarize_shadow() -> None:
    rows: list[dict[str, Any]] = [
        {"shadow": {"action": "answer", "tier": 1}},
        {"shadow": {"action": "answer", "tier": 2}},
        {"shadow": {"action": "abstain", "tier": None}},
        {"shadow": None},
    ]
    out = summarize_shadow(rows)
    assert out["n"] == 3 and out["answer_tier1"] == 1 and out["abstain"] == 1
    assert out["abstain_rate"] == pytest.approx(1 / 3)


def test_memory_sink() -> None:
    m = MemoryAuditSink()
    m.write(rec())
    assert len(m.records) == len(m.lines) == 1


# --- metrics -----------------------------------------------------------------------------------


def test_bounded_labels_and_registry_isolation() -> None:
    assert bounded("answer", DECISIONS) == "answer"
    assert bounded("user-supplied-garbage", DECISIONS) == "other"
    assert {r.value for r in ReasonCode} <= REASONS
    a, b = SieveMetrics(), SieveMetrics()
    a.escalations.inc()
    assert b"sieve_escalations_total 1.0" in generate_latest(a.registry)
    assert b"sieve_escalations_total 0.0" in generate_latest(b.registry)


# --- drift -------------------------------------------------------------------------------------


def _mon(ref: np.ndarray[Any, Any]) -> DriftMonitor:
    return DriftMonitor(ref, window=200, min_window=100, check_every=20)


def test_drift_does_not_fire_on_stable_data() -> None:
    rng = np.random.default_rng(1)
    ref = rng.beta(8, 2, 1000)
    for seed in range(10):  # ten independent stable streams: no false alarm
        m = _mon(ref)
        r = np.random.default_rng(100 + seed)
        alarms = [m.observe(float(x)).alarm for x in r.beta(8, 2, 600)]
        assert not any(alarms), seed
        assert m.state.ks_pvalue is not None and m.state.psi is not None


def test_drift_fires_on_shifted_data() -> None:
    rng = np.random.default_rng(2)
    ref = rng.beta(8, 2, 1000)
    m = _mon(ref)
    fired_at = None
    for i, x in enumerate(np.random.default_rng(3).beta(4, 4, 600)):
        if m.observe(float(x)).alarm:
            fired_at = i
            break
    assert fired_at is not None and fired_at < 300


def test_drift_ignores_nonfinite_and_needs_enough_data() -> None:
    m = _mon(np.linspace(0, 1, 200))
    for _ in range(50):
        m.observe(float("nan"))
        m.observe(None)
    assert m.skipped_non_finite == 100 and m.state.ks_pvalue is None
    with pytest.raises(ValueError):
        DriftMonitor(np.arange(10.0))


def test_load_reference_and_hash_pin(tmp_path: Path) -> None:
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(list(np.linspace(0, 1, 100))))
    assert load_reference(str(p)).size == 100
    import hashlib

    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert load_reference(str(p), h).size == 100
    with pytest.raises(ValueError, match="hash"):
        load_reference(str(p), "0" * 64)
    n = tmp_path / "ref.npy"
    np.save(n, np.linspace(0, 1, 80))
    assert load_reference(str(n)).size == 80
    p.write_text("[1,2,3]")
    with pytest.raises(ValueError, match="at least"):
        load_reference(str(p))


# --- pins ----------------------------------------------------------------------------------------


def test_config_pin_is_order_insensitive_and_checked() -> None:
    a = config_pin({"x": 1, "y": {"b": 2, "a": 3}})
    assert a == config_pin({"y": {"a": 3, "b": 2}, "x": 1}) != config_pin({"x": 2, "y": {}})
    check_pins(config=a, artifact_id="art", expected_config=a, expected_artifact="art")
    check_pins(config=a, artifact_id="art", expected_config=None, expected_artifact=None)
    with pytest.raises(PinMismatchError):
        check_pins(config=a, artifact_id="art", expected_config="zzz", expected_artifact=None)
    with pytest.raises(PinMismatchError):
        check_pins(config=a, artifact_id="art", expected_config=None, expected_artifact="other")


def test_default_redactors_cover_common_secret_classes() -> None:
    """F-14: SSN, phone, AWS/GitHub keys and password= strings are redacted (best effort)."""
    from sieve.observe.redact import apply_redactors, default_redactors

    r = default_redactors()
    for secret in (
        "123-45-6789",
        "(415) 555-2671",
        "415-555-2671",
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4",
        "password=hunter2hunter2",
    ):
        out = apply_redactors(f"my data: {secret} thanks", r)
        assert secret not in out, secret
    assert apply_redactors("ordinary text 2+2=4", r) == "ordinary text 2+2=4"

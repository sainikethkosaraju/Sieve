"""Split integrity and the leakage guard around split B (kill-test family: benchmark integrity)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from sieve.eval import splits
from sieve.eval.splits import (
    CERTIFICATION_CAPABILITY,
    SplitAccessError,
    SplitBAlreadyReadError,
    SplitIntegrityError,
    SplitStore,
    build_splits,
    derive_split,
)

IDS = [f"q{i:04d}" for i in range(90)]
SIZES = (30, 30, 30)


def _build(tmp_path: Path, n_splits: int = 5) -> SplitStore:
    return build_splits(
        tmp_path, IDS, dataset_hash="d" * 8, split_seed=11, n_splits=n_splits, sizes=SIZES
    )


def test_resplit_is_a_deterministic_partition() -> None:
    a = derive_split(IDS, 3, 11, SIZES)
    b = derive_split(list(reversed(IDS)), 3, 11, SIZES)
    assert a == b  # independent of input order
    union = [i for n in "ABC" for i in a[n]]
    assert sorted(union) == sorted(IDS)
    assert [len(a[n]) for n in "ABC"] == [30, 30, 30]
    assert derive_split(IDS, 4, 11, SIZES) != a  # different re-split
    assert derive_split(IDS, 3, 12, SIZES) != a  # different seed


def test_resplits_actually_vary() -> None:
    first_b = set(derive_split(IDS, 0, 11, SIZES)["B"])
    overlaps = [len(first_b & set(derive_split(IDS, k, 11, SIZES)["B"])) for k in range(1, 20)]
    assert len(set(overlaps)) > 3 and max(overlaps) < 30


def test_build_load_verify_roundtrip(tmp_path: Path) -> None:
    s = _build(tmp_path)
    again = SplitStore.load(tmp_path)
    assert again.index_hash == s.index_hash
    assert again.a_ids(2) == s.a_ids(2)
    # rebuilding with the same inputs is a no-op, with different inputs it is refused
    _build(tmp_path)
    with pytest.raises(SplitIntegrityError):
        build_splits(tmp_path, IDS, dataset_hash="x", split_seed=11, n_splits=5, sizes=SIZES)


def test_tampered_split_file_is_detected(tmp_path: Path) -> None:
    _build(tmp_path)
    p = tmp_path / "split_002.json"
    d = json.loads(p.read_text())
    d["ids"]["A"][0], d["ids"]["B"][0] = d["ids"]["B"][0], d["ids"]["A"][0]  # swap one A/B id
    p.write_text(json.dumps(d))
    with pytest.raises(SplitIntegrityError):
        SplitStore.load(tmp_path)


def test_tampered_index_is_detected(tmp_path: Path) -> None:
    _build(tmp_path)
    p = tmp_path / splits.INDEX_NAME
    d = json.loads(p.read_text())
    d["split_seed"] = 99
    p.write_text(json.dumps(d))
    with pytest.raises(SplitIntegrityError):
        SplitStore.load(tmp_path)


def test_split_not_matching_seeded_derivation_is_detected(tmp_path: Path) -> None:
    _build(tmp_path)
    # a hand-picked "lucky" split with consistent hashes still fails re-derivation
    rigged = derive_split(IDS, 1, 11, SIZES)
    a, b = list(rigged["A"]), list(rigged["B"])
    a[0], b[0] = b[0], a[0]
    payload: dict[str, Any] = {
        "k": 1,
        "ids": {"A": a, "B": b, "C": list(rigged["C"])},
        "hashes": {},
    }
    idx = json.loads((tmp_path / splits.INDEX_NAME).read_text())
    idx["splits"][1] = {
        "k": 1,
        **{n: splits.ids_hash(payload["ids"][n]) for n in "ABC"},
    }
    body = {k: v for k, v in idx.items() if k != "index_hash"}
    from sieve.eval._util import sha256_obj

    idx["index_hash"] = sha256_obj(body)
    (tmp_path / splits.INDEX_NAME).write_text(json.dumps(idx))
    (tmp_path / "split_001.json").write_text(json.dumps(payload))
    with pytest.raises(SplitIntegrityError, match="seeded derivation"):
        SplitStore.load(tmp_path)


def test_b_is_closed_without_the_capability(tmp_path: Path) -> None:
    s = _build(tmp_path)
    with pytest.raises(SplitAccessError):
        s.open_b(0, capability=object())
    assert s.b_state(0) == "unread"
    assert not (tmp_path / splits.LEDGER_NAME).exists()  # a refused read leaves no trace


def test_b_opens_once(tmp_path: Path) -> None:
    s = _build(tmp_path)
    ids = s.open_b(0, capability=CERTIFICATION_CAPABILITY)
    assert len(ids) == 30 and not set(ids) & set(s.a_ids(0))
    assert s.b_state(0) == "opened_uncertified"
    with pytest.raises(SplitBAlreadyReadError):
        s.open_b(0, capability=CERTIFICATION_CAPABILITY)
    # the ledger is on disk: a fresh process / fresh object cannot reopen it either
    with pytest.raises(SplitBAlreadyReadError):
        SplitStore.load(tmp_path).open_b(0, capability=CERTIFICATION_CAPABILITY)
    # other re-splits are unaffected
    s.open_b(1, capability=CERTIFICATION_CAPABILITY)


def test_c_requires_a_certification_of_the_same_split(tmp_path: Path) -> None:
    s = _build(tmp_path)
    with pytest.raises(SplitAccessError):
        s.open_c(0, cert_hash="nope")
    s.open_b(0, capability=CERTIFICATION_CAPABILITY)
    with pytest.raises(SplitAccessError):
        s.open_c(0, cert_hash="nope")  # B opened but nothing certified yet
    with pytest.raises(SplitAccessError):  # cannot certify without the capability
        s.record_certification(0, "h", capability=object())
    s.record_certification(0, "h0", capability=CERTIFICATION_CAPABILITY)
    assert s.b_state(0) == "certified"
    assert len(s.open_c(0, cert_hash="h0")) == 30
    with pytest.raises(SplitAccessError):  # a certification of split 0 does not unlock split 1
        s.open_c(1, cert_hash="h0")
    with pytest.raises(SplitAccessError):  # cannot record a certification for an unopened B
        s.record_certification(2, "h2", capability=CERTIFICATION_CAPABILITY)


def test_only_certify_module_touches_b() -> None:
    """Static guard: nothing but certify.py (and the definition site) opens B or imports the key."""
    root = Path(splits.__file__).parent
    offenders = []
    for p in root.glob("*.py"):
        text = p.read_text()
        if p.name in ("splits.py", "certify.py"):
            continue
        if re.search(r"\bopen_b\(|CERTIFICATION_CAPABILITY|record_certification\(", text):
            offenders.append(p.name)
    assert offenders == []
    assert "CERTIFICATION_CAPABILITY" in (root / "certify.py").read_text()


def test_design_view_exposes_only_a(tmp_path: Path) -> None:
    s = _build(tmp_path)
    public = [n for n in dir(s) if not n.startswith("_")]
    readers = [n for n in public if n.endswith("_ids") or n.startswith("open_")]
    assert sorted(readers) == ["a_ids", "master_ids", "open_b", "open_c"]


def test_c_opens_once_per_certification(tmp_path: Path) -> None:
    s = _build(tmp_path)
    s.open_b(0, capability=CERTIFICATION_CAPABILITY)
    s.record_certification(0, "h0", capability=CERTIFICATION_CAPABILITY)
    first = s.open_c(0, cert_hash="h0")
    assert s.open_c(0, cert_hash="h0") == first  # same object: served from memory
    fresh = SplitStore.load(tmp_path)
    with pytest.raises(splits.SplitCAlreadyReadError):
        fresh.open_c(0, cert_hash="h0")
    assert fresh.open_c(0, cert_hash="h0", reopen=True) == first
    kinds = [e["event"] for e in fresh.ledger.events()]
    assert kinds.count("open_c") == 1 and kinds.count("reopen_c") == 1


def test_ledger_records_code_version_and_config_hash(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("SIEVE_GIT_COMMIT", "abc123")
    splits.code_version.cache_clear()
    try:
        s = _build(tmp_path)
        s.open_b(0, capability=CERTIFICATION_CAPABILITY, config_hash="cfg-h")
        ev = s.ledger.events()[0]
        assert ev["code_version"] == "abc123" and ev["config_hash"] == "cfg-h"
    finally:
        splits.code_version.cache_clear()


def test_ledger_append_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    s = _build(tmp_path)
    for k in range(3):
        s.open_b(k, capability=CERTIFICATION_CAPABILITY)
    text = (tmp_path / splits.LEDGER_NAME).read_text()
    assert text.endswith("\n") and len(text.splitlines()) == 3
    assert not list(tmp_path.glob("*.tmp*"))
    # a torn trailing line from a crashed non-atomic writer does not corrupt later appends
    (tmp_path / splits.LEDGER_NAME).write_text(text.rstrip("\n"))
    s.open_b(3, capability=CERTIFICATION_CAPABILITY)
    assert len((tmp_path / splits.LEDGER_NAME).read_text().splitlines()) == 4


def test_tampered_stored_certification_is_rejected(mock_run: tuple[Path, Any]) -> None:
    from typing import cast

    from sieve.eval.certify import certify_or_load

    wd, _ = mock_run
    store = SplitStore.load(wd / "splits")
    path = wd / "certs" / "split_000.json"
    original = path.read_text()
    doc = json.loads(original)
    cell = next(c for ms in doc["certs"].values() for c in ms.values())
    try:
        # untouched: accepted (data/plans are not needed to load a stored certification)
        certify_or_load(store, 0, cast(Any, None), cast(Any, None), cast(Any, None), wd / "certs")
        cell["t1"] = -50.0
        cell["k_wrong_B"] = 0
        path.write_text(json.dumps(doc))
        with pytest.raises(RuntimeError, match="modified"):
            certify_or_load(
                store, 0, cast(Any, None), cast(Any, None), cast(Any, None), wd / "certs"
            )
    finally:
        path.write_text(original)

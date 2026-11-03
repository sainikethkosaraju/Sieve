"""M8 / F-21: every artefact of a synthetic run (and of the committed demo) is marked SYNTHETIC."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from sieve.eval import report

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "results" / "synthetic-demo"


def unmarked_artefacts(root: Path) -> list[str]:
    """Names of files under ``root`` that carry no explicit SYNTHETIC marker."""
    bad: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if p.suffix == ".csv":
            with p.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            if not rows or "synthetic" not in rows[0]:
                bad.append(rel)
                continue
            col = rows[0].index("synthetic")
            if any(len(r) <= col or r[col] != "SYNTHETIC" for r in rows[1:]):
                bad.append(rel)
        elif p.suffix in (".md", ".html", ".json", ".txt"):
            if "SYNTHETIC" not in p.read_text(encoding="utf-8"):
                bad.append(rel)
        elif p.suffix == ".png":
            if b"SYNTHETIC" not in p.read_bytes():
                bad.append(rel)
        else:
            bad.append(rel)  # unknown artefact type: refuse rather than assume
    return bad


def test_marker_check_detects_an_unmarked_artefact(tmp_path: Path) -> None:
    (tmp_path / "t.csv").write_text("a,b\n1,2\n")
    (tmp_path / "ok.md").write_text("SYNTHETIC\n")
    assert unmarked_artefacts(tmp_path) == ["t.csv"]


def test_committed_demo_is_fully_marked() -> None:
    assert not (ROOT / "results" / "mock").exists(), "renamed to results/synthetic-demo"
    assert DEMO.is_dir(), "run benchmarks/reproduce_mock.sh WORK results/synthetic-demo"
    assert unmarked_artefacts(DEMO) == []
    labels = (DEMO / "tables" / "accuracy.csv").read_text()
    assert "synthetic-triviaqa" in labels and ",triviaqa," not in labels


@pytest.mark.slow
def test_fresh_synthetic_run_is_fully_marked(mock_run: tuple[Path, dict[str, Any]]) -> None:
    assert unmarked_artefacts(mock_run[0] / "report") == []


def test_mark_synthetic_relabels_models_and_datasets() -> None:
    t = {"x": (["cell", "dataset", "n"], [["S_q4km", "triviaqa", 3]])}
    head, rows = report.mark_synthetic(t)["x"]
    assert head[0] == "synthetic" and rows[0][:3] == [
        "SYNTHETIC",
        "synthetic-S_q4km",
        "synthetic-triviaqa",
    ]

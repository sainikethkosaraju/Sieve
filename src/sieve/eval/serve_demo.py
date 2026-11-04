"""End-to-end serving demo: an evaluation-certified artifact served through the chat API.

Starts the real backends through the evaluation provider (so identity, stop handling, token
trimming and the logprob self-test are exactly those of the experiment), loads the artifact the
harness certified on split B, mounts it in the deployed server application, and sends fresh
held-out questions to ``/v1/chat/completions``. Every request goes through the same router, gate
and response envelope a user would hit.

What this shows: the certified artifact is accepted by the serving stack (identity, template
hash, sampling), and what the served answer/abstain behaviour is on fresh items from the
calibration distribution. It is a small demonstration with wide intervals, not a new experiment.
The application is called in-process over ASGI, and the backends are the harness's own, not the
server's ``llamaserver`` adapter.

    python -m sieve.eval.serve_demo --config benchmarks/t4_main.yaml --workdir <wd> --n 300
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from prometheus_client import CollectorRegistry

from sieve.backends import Backend
from sieve.calibrate.store import loads_artifact
from sieve.eval import pipeline
from sieve.eval.config import ExperimentConfig, load_config
from sieve.eval.datasets import PINNED, Item, load_dataset_items, read_items, sample_items
from sieve.eval.grading import grade_alias_em
from sieve.observe import MemoryAuditSink
from sieve.server import BackendSettings, ServerSettings, create_app
from sieve.types import CalibrationArtifact


class DemoError(RuntimeError):
    pass


def artifact_from_certification(
    path: Path, pair: str = "main", method: str = "gate:primary"
) -> CalibrationArtifact:
    """The artifact embedded in a persisted split certification (``certs/split_NNN.json``)."""
    d = json.loads(path.read_text(encoding="utf-8"))
    try:
        cert = d["certs"][pair][method]
    except KeyError as e:
        raise DemoError(f"{path.name}: no certification for {pair}/{method}") from e
    if cert.get("outcome") != "feasible" or not cert.get("artifact"):
        raise DemoError(
            f"{pair}/{method} on this split is {cert.get('outcome')!r}: nothing certified to serve"
        )
    return loads_artifact(json.dumps(cert["artifact"]))


def settings_for(artifact_name: str) -> ServerSettings:
    """Server settings; the two backend entries are placeholders because backends are injected."""
    ph = {"kind": "mock", "model_id": "injected", "model_hash": "injected", "quant": "injected"}
    return ServerSettings(
        artifact_path=artifact_name,
        small=BackendSettings(**ph),
        large=BackendSettings(**ph),
        health_interval_s=0,
        require_self_test=False,  # the provider already ran the self-test on each backend
        guard={"max_escalation_rate": 1.0, "window": 200},
    )


def _pct(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))] if s else float("nan")


async def serve_and_ask(
    art: CalibrationArtifact,
    small: Backend,
    large: Backend,
    items: Sequence[Item],
    concurrency: int,
) -> dict[str, Any]:
    app = create_app(
        settings_for("injected"),
        artifact=art,
        small=small,
        large=large,
        sink=MemoryAuditSink(),
        registry=CollectorRegistry(),
    )
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict[str, Any]] = []

    async def one(c: httpx.AsyncClient, it: Item) -> None:
        async with sem:
            t0 = time.monotonic()
            r = await c.post(
                "/v1/chat/completions",
                json={"model": "sieve", "messages": [{"role": "user", "content": it.question}]},
            )
            dt = time.monotonic() - t0
        j = r.json()
        s = j.get("sieve", {}) if r.status_code == 200 else {}
        decision = str(s.get("decision", f"http_{r.status_code}"))
        correct: bool | None = None
        if decision == "answer":
            content = j["choices"][0]["message"]["content"]
            correct = bool(grade_alias_em(it, content).correct)
        rows.append(
            {
                "decision": decision,
                "tier": s.get("tier"),
                "reason": str(s.get("reason", "")),
                "escalated": bool(s.get("escalated", False)),
                "correct": correct,
                "latency_s": dt,
            }
        )

    t0 = time.monotonic()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sieve.test"
        ) as c,
    ):
        ready = await c.get("/readyz")
        models = await c.get("/v1/models")
        await asyncio.gather(*(one(c, it) for it in items))
    wall = time.monotonic() - t0
    return {"rows": rows, "wall_s": wall, "readyz": ready.status_code, "models": models.status_code}


def summarise(raw: dict[str, Any], art: CalibrationArtifact, n_requested: int) -> dict[str, Any]:
    rows = raw["rows"]
    dec = Counter(r["decision"] for r in rows)
    reasons = Counter(r["reason"] for r in rows)
    answered = [r for r in rows if r["decision"] == "answer"]
    wrong = sum(1 for r in answered if r["correct"] is False)
    lat = [r["latency_s"] for r in rows]
    return {
        "synthetic": False,
        "n_requested": n_requested,
        "n_responses": len(rows),
        "decisions": dict(dec),
        "reasons": dict(reasons),
        "answered": len(answered),
        "answered_share": len(answered) / len(rows) if rows else None,
        "answered_from_tier": dict(Counter(r["tier"] for r in answered)),
        "wrong_among_answered": wrong,
        "error_among_answered": wrong / len(answered) if answered else None,
        "wrong_share_of_all": wrong / len(rows) if rows else None,
        "alpha_target": art.alpha,
        "delta": art.delta,
        "artifact_id": art.artifact_id,
        "latency_s": {
            "mean": statistics.fmean(lat) if lat else None,
            "p50": _pct(lat, 0.5),
            "p95": _pct(lat, 0.95),
        },
        "wall_s": raw["wall_s"],
        "readyz_status": raw["readyz"],
        "models_status": raw["models"],
        "grader": "alias exact match (strict)",
        "note": "small fresh held-out sample from the calibration distribution; wide intervals",
    }


def fresh_items(cfg: ExperimentConfig, workdir: Path, n: int, seed: int) -> list[Item]:
    """Items from the primary dataset's pool that the experiment never used (not in its 6,000)."""
    spec = PINNED[cfg.primary_dataset]
    used = {i.item_id for i in read_items(workdir / "items" / f"{cfg.primary_dataset}.jsonl")}
    pool = [i for i in load_dataset_items(spec) if i.item_id not in used]
    return sample_items(pool, n, seed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--out", default="serve_demo.json")
    a = ap.parse_args(argv)
    cfg = load_config(Path(a.config))
    wd = Path(a.workdir)
    art = artifact_from_certification(wd / "certs" / f"split_{a.split:03d}.json")
    items = fresh_items(cfg, wd, a.n, a.seed)
    pair = cfg.main_pair()
    provider = pipeline.load_provider(cfg)
    with contextlib.ExitStack() as stack:
        small = stack.enter_context(provider.open(cfg.cell(pair.s_cell)))
        large = stack.enter_context(provider.open(cfg.cell(pair.l_cell)))
        raw = asyncio.run(serve_and_ask(art, small, large, items, a.concurrency))
    out = summarise(raw, art, len(items))
    (wd / a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("SERVE_DEMO_JSON_BEGIN")
    print(json.dumps(out))
    print("SERVE_DEMO_JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

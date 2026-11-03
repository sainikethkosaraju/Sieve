"""``python -m sieve.eval`` command line."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from sieve.eval import pipeline, prereg
from sieve.eval.audit import analyse_audit
from sieve.eval.config import load_config


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _cfg(path: str):  # type: ignore[no-untyped-def]
    return load_config(Path(path))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sieve.eval", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-data", help="load/sample datasets and write hash-locked splits")
    p.add_argument("--config", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--synthetic", action="store_true", help="generate fake items, no network")

    p = sub.add_parser("run", help="cache-everything generation (resumable)")
    p.add_argument("--config", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--cell", action="append", help="run only this cell (repeatable)")
    p.add_argument("--no-power", action="store_true", help="skip nvidia-smi sampling")

    p = sub.add_parser("analyse", help="certify, test, hypotheses; writes analysis/*.json")
    p.add_argument("--config", required=True)
    p.add_argument("--workdir", required=True)

    p = sub.add_parser("report", help="tables, figures and the static results page")
    p.add_argument("--config", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--out", help="copy the report here (e.g. results/<run>)")

    p = sub.add_parser("audit-sheet", help="blind 200-item human-audit sheet (split A only)")
    p.add_argument("--config", required=True)
    p.add_argument("--workdir", required=True)

    p = sub.add_parser("audit-analyse", help="e1/e2 with CIs from a filled audit sheet")
    p.add_argument("--workdir", required=True)

    p = sub.add_parser("hypotheses", help="print the H1-H7 report (refuses if prereg changed)")
    p.add_argument("--config", required=True)
    p.add_argument("--workdir", required=True)

    p = sub.add_parser("mock", help="offline synthetic end-to-end run (make reproduce-mock)")
    p.add_argument("--workdir", required=True)
    p.add_argument("--out", help="copy the report here")
    p.add_argument("--small", action="store_true", help="smaller sizes (used by the test-suite)")
    p.add_argument("--kill-after", type=int, help="test hook: hard-exit after N generations")

    a = ap.parse_args(argv)
    workdir = Path(a.workdir)

    if a.cmd == "mock":
        from sieve.eval._fake_backend import FakeProvider, SimClock

        sim = SimClock()
        prov = FakeProvider(sim, kill_after=a.kill_after, hard_kill=a.kill_after is not None)
        out = pipeline.run_mock(
            workdir,
            cfg_dict=pipeline.mock_config(small=a.small),
            provider=prov,
            clock=sim,
            say=_say,
        )
        if a.out:
            shutil.copytree(workdir / "report", Path(a.out), dirs_exist_ok=True)
        print((workdir / "analysis" / "hypotheses.txt").read_text(encoding="utf-8"))
        print(json.dumps(out["hypotheses"]))
        return 0

    if a.cmd == "audit-analyse":
        r = analyse_audit(
            workdir / "audit" / "audit_sheet.csv", workdir / "audit" / "audit_key.json"
        )
        print(json.dumps(r, default=lambda o: o.__dict__, indent=1, sort_keys=True))
        return 0

    cfg = _cfg(a.config)
    if a.cmd == "prepare-data":
        s = pipeline.prepare_data(cfg, workdir, synthetic=a.synthetic or None, say=_say)
        print(json.dumps(s, indent=1, sort_keys=True))
    elif a.cmd == "run":
        from sieve.eval.power import SmiMeter

        meter = None if a.no_power else SmiMeter()
        if meter is not None:
            state = meter.start()
            _say(f"power sampler: {state.value}")
            meter.idle_baseline(cfg.idle_seconds)
        try:
            out = pipeline.run_generation(
                cfg, workdir, pipeline.load_provider(cfg), meter=meter, say=_say, only=a.cell
            )
        finally:
            if meter is not None:
                meter.stop()
        print(json.dumps(out, indent=1, sort_keys=True))
    elif a.cmd == "analyse":
        from sieve.eval.analysis import run_analysis
        from sieve.eval.hypotheses import build_report, format_report

        judge = pipeline.make_judge(cfg)
        res = run_analysis(workdir, cfg, pipeline.load_items(workdir), judge, progress=_say)
        rep = build_report(res, cfg)
        (workdir / "analysis" / "hypotheses.json").write_text(
            json.dumps(rep, indent=1, sort_keys=True)
        )
        print(format_report(rep))
    elif a.cmd == "hypotheses":
        from sieve.eval.hypotheses import build_report, format_report

        res = json.loads((workdir / "analysis" / "results.json").read_text(encoding="utf-8"))
        print(format_report(build_report(res, cfg)))
    elif a.cmd == "report":
        from sieve.eval.hypotheses import build_report
        from sieve.eval.report import write_report

        res = json.loads((workdir / "analysis" / "results.json").read_text(encoding="utf-8"))
        rep = build_report(res, cfg)
        idx = write_report(workdir, res, rep, cfg)
        if a.out:
            shutil.copytree(idx.parent, Path(a.out), dirs_exist_ok=True)
        print(idx)
    elif a.cmd == "audit-sheet":
        print(json.dumps(pipeline.audit_sheet(cfg, workdir, pipeline.make_judge(cfg)), indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

__all__ = ["main", "prereg"]

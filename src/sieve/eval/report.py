# ruff: noqa: E501
"""Tables (CSV + Markdown), figures (matplotlib) and the static results page.

Every figure footer carries hardware, commit, config hash and, for mock runs, the SYNTHETIC
label. The HTML page is a single self-contained file: inline CSS, base64 PNGs with alt text,
no scripts, no external requests; light/dark via ``prefers-color-scheme``.
"""

from __future__ import annotations

import base64
import csv
import html
import importlib
import io
import json
import textwrap
from pathlib import Path
from typing import Any

from sieve.eval._util import atomic_write_text
from sieve.eval.config import ExperimentConfig

# Okabe-Ito colour-blind-safe palette
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _ci(d: dict[str, float] | None) -> str:
    return "n/a" if not d else f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


# --- tables ----------------------------------------------------------------------------------


def build_tables(
    res: dict[str, Any], hyp: dict[str, Any]
) -> dict[str, tuple[list[str], list[list[Any]]]]:
    t: dict[str, tuple[list[str], list[list[Any]]]] = {}
    rows = []
    for cell, c in res["accuracy"].items():
        for ds, r in c["datasets"].items():
            rows.append(
                [
                    cell,
                    c["tier"],
                    c["backend"]["quant"],
                    c["backend"]["kv_quant"],
                    ds,
                    r["n"],
                    r["n_graded"],
                    r["accuracy"],
                    r["fail_closed_rate"],
                    r["mean_time_s"],
                    r["wasted_time_share"],
                ]
            )
    t["accuracy"] = (
        [
            "cell",
            "tier",
            "quant",
            "kv",
            "dataset",
            "n",
            "n_graded",
            "accuracy",
            "fail_closed_rate",
            "mean_time_s",
            "wasted_time_share",
        ],
        rows,
    )
    rows = []
    for m, v in res["methods"].items():
        a = v["across_splits"]
        s0 = v["split0"] or {}
        rows.append(
            [
                m,
                a["n_certified"],
                a["n_empty"],
                a["violation_freq"],
                a["mean_test_risk"],
                a["mean_coverage"],
                a["mean_cost_per_correct"],
                s0.get("risk"),
                s0.get("coverage"),
                s0.get("escalation_rate"),
                s0.get("cost_per_correct"),
            ]
        )
    t["methods"] = (
        [
            "method",
            "certified_splits",
            "empty_splits",
            "violation_freq",
            "mean_test_risk",
            "mean_coverage",
            "mean_cost_per_correct",
            "split0_risk",
            "split0_coverage",
            "split0_escalation",
            "split0_cost_per_correct",
        ],
        rows,
    )
    rows = []
    for tier, d in res["signals"]["auroc"].items():
        for sig, ci in d.items():
            rows.append([tier, sig, ci["point"], ci["lo"], ci["hi"]])
    t["signal_auroc"] = (["tier", "signal", "auroc", "lo", "hi"], rows)
    rows = []
    for k, v in res["comparisons"].items():
        if k.startswith("vs_") and isinstance(v, dict) and "coverage_diff" in v:
            rows.append(
                [
                    k,
                    _ci(v["acc_on_answered_diff"]),
                    _ci(v["cost_per_correct_diff"]),
                    _ci(v["coverage_diff"]),
                    v["mcnemar_correct_answered"]["p_exact"],
                ]
            )
    for name, v in res["comparisons"].get("matched", {}).items():
        d = v["vs_gate"]
        rows.append(
            [
                f"vs_{name}@matched",
                _ci(d["acc_on_answered_diff"]),
                _ci(d["cost_per_correct_diff"]),
                _ci(d["coverage_diff"]),
                d["mcnemar_correct_answered"]["p_exact"],
            ]
        )
    t["comparisons_gate_minus_other"] = (
        [
            "comparison",
            "acc_on_answered_diff",
            "cost_per_correct_diff",
            "coverage_diff",
            "mcnemar_p",
        ],
        rows,
    )
    rows = []
    for ds, v in res["shift"].items():
        a = v["across_splits"]
        rows.append(
            [
                ds,
                v["graded"],
                v["n_items"],
                a.get("violation_freq"),
                a.get("mean_test_risk"),
                _ci(v["risk_ci95_split0"]),
                v["drift"]["alarm_rate"],
                v["drift"]["n_windows"],
            ]
        )
    t["shift"] = (
        [
            "dataset",
            "graded",
            "n",
            "violation_freq",
            "mean_risk",
            "risk_ci95_split0",
            "drift_alarm_rate",
            "n_windows",
        ],
        rows,
    )
    rows = []
    for name, v in res["quant"].get("configs", {}).items():
        rows.append(
            [
                name,
                v["transferred"]["violation_freq"],
                v["recalibrated"]["violation_freq"],
                _ci(v["delta_auroc"]),
                _ci(v["delta_ece"]),
                v["auroc_equivalent"],
                v["ece_equivalent"],
            ]
        )
    t["quantization"] = (
        [
            "config",
            "transferred_violation_freq",
            "recalibrated_violation_freq",
            "delta_auroc",
            "delta_ece",
            "auroc_equivalent",
            "ece_equivalent",
        ],
        rows,
    )
    t["hypotheses"] = (
        ["id", "title", "verdict", "why"],
        [[v["id"], v["title"], v["verdict"], v["why"]] for v in hyp["verdicts"]],
    )
    return t


SYNTHETIC_MARK = "SYNTHETIC"
_LABEL_COLUMNS = frozenset({"cell", "quant", "dataset", "config"})


def mark_synthetic(
    tables: dict[str, tuple[list[str], list[list[Any]]]],
) -> dict[str, tuple[list[str], list[list[Any]]]]:
    """Synthetic runs: a ``synthetic`` column on every table and 'synthetic-' model/dataset labels.

    A copied row can then never be mistaken for a result about a real model (audit M8).
    """
    out: dict[str, tuple[list[str], list[list[Any]]]] = {}
    for name, (head, rows) in tables.items():
        idx = [i for i, h in enumerate(head) if h in _LABEL_COLUMNS]
        new_rows = []
        for r in rows:
            r2 = list(r)
            for i in idx:
                if isinstance(r2[i], str) and r2[i] and not r2[i].startswith("synthetic-"):
                    r2[i] = f"synthetic-{r2[i]}"
            new_rows.append([SYNTHETIC_MARK, *r2])
        out[name] = (["synthetic", *head], new_rows)
    return out


def write_tables(
    tables: dict[str, tuple[list[str], list[list[Any]]]], outdir: Path, *, synthetic: bool = False
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    md: list[str] = []
    if synthetic:
        md += [
            f"<!-- {SYNTHETIC_MARK} -->",
            f"**{SYNTHETIC_MARK} DATA (fake backend): nothing here is a result about a real model.**",
            "",
        ]
    for name, (head, rows) in tables.items():
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(head)
        for r in rows:
            w.writerow([_fmt(x, 4) if x is not None else "" for x in r])
        atomic_write_text(outdir / f"{name}.csv", buf.getvalue())
        md += [f"### {name}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        md += ["| " + " | ".join(_fmt(x) for x in r) + " |" for r in rows]
        md.append("")
    atomic_write_text(outdir / "tables.md", "\n".join(md))


# --- figures ----------------------------------------------------------------------------------


def _footer(res: dict[str, Any]) -> str:
    m = res["meta"]
    hw = m.get("hardware") or {}
    hws = hw.get("platform", "unknown hardware") if isinstance(hw, dict) else str(hw)
    syn = "SYNTHETIC DATA (fake backend) | " if m.get("synthetic") else ""
    return (
        f"{syn}{hws} | commit {str(m.get('commit'))[:12]} | config {str(m['config_hash'])[:12]} | "
        f"alpha={m['alpha']} delta={m['delta']}"
    )


def make_figures(res: dict[str, Any], outdir: Path) -> dict[str, tuple[Path, str]]:
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    outdir.mkdir(parents=True, exist_ok=True)
    foot = _footer(res)
    figs: dict[str, tuple[Path, str]] = {}

    def save(fig: Any, name: str, alt: str) -> None:
        fig.text(
            0.01,
            0.005,
            textwrap.fill(foot, 105),
            fontsize=6.5,
            color="#444444",
            ha="left",
            va="bottom",
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        p = outdir / f"{name}.png"
        meta = {"Software": "sieve.eval", "Creation Time": None}
        if res["meta"].get("synthetic"):
            meta["Description"] = "SYNTHETIC DATA (fake backend)"
        fig.savefig(p, dpi=110, metadata=meta)
        plt.close(fig)
        figs[name] = (p, alt)

    # risk-coverage
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (tier, pts) in enumerate(sorted(res["curves"]["risk_coverage"].items())):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], label=f"{tier} tier", color=PALETTE[i])
    ax.axhline(res["meta"]["alpha"], ls="--", color="#777777", lw=1, label="alpha")
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk (graded-wrong among answered)")
    ax.set_title(
        "Risk-coverage, re-split 0 test set" + (" (SYNTHETIC)" if res["meta"]["synthetic"] else "")
    )
    ax.legend(fontsize=8)
    save(
        fig,
        "risk_coverage",
        "Risk versus coverage curves for the small and large tier on the test split.",
    )

    # Pareto
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    unit = res["meta"]["cost_unit"]
    for i, (name, pts) in enumerate(sorted(res["curves"]["pareto"]["sweep"].items())):
        pts = sorted(pts, key=lambda p: p["cost_per_item"])
        ax.plot(
            [p["cost_per_item"] for p in pts],
            [p["acc_on_answered"] for p in pts],
            ".",
            ms=3,
            alpha=0.5,
            color=PALETTE[i % len(PALETTE)],
        )
        front, best = [], -1.0
        for p in pts:
            if p["acc_on_answered"] > best:
                front.append(p)
                best = p["acc_on_answered"]
        ax.plot(
            [p["cost_per_item"] for p in front],
            [p["acc_on_answered"] for p in front],
            "-",
            lw=1.2,
            color=PALETTE[i % len(PALETTE)],
            label=name,
        )
    for _name, c in sorted(res["curves"]["pareto"]["certified"].items()):
        x, y = c["cost_per_item"], c["acc_on_answered"]
        ax.errorbar(
            [x["point"]],
            [y["point"]],
            xerr=[[x["point"] - x["lo"]], [x["hi"] - x["point"]]],
            yerr=[[y["point"] - y["lo"]], [y["hi"] - y["point"]]],
            fmt="s",
            color="black",
            ms=4,
            capsize=2,
            lw=0.8,
        )
    ax.set_xlabel(f"mean cost per query ({unit}, incl. wasted S attempts)")
    ax.set_ylabel("accuracy on answered")
    ax.set_title("Cost vs selective accuracy; squares = certified points, 95% bootstrap CIs")
    ax.title.set_fontsize(8)
    ax.legend(fontsize=7)
    save(
        fig,
        "pareto",
        "Cost versus accuracy-on-answered frontier for each method with certified points and confidence intervals.",
    )

    # reliability
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#777777", lw=1)
    for i, tier in enumerate(("S", "L")):
        d = res["calibration"].get(tier)
        if d:
            ax.plot(
                [b["mean_p"] for b in d["bins"]],
                [b["mean_y"] for b in d["bins"]],
                "o-",
                ms=3,
                color=PALETTE[i],
                label=f"{tier} (ECE {d['ece']:.3f})",
            )
    ax.set_xlabel("predicted P(correct) (Platt, fit on A)")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Reliability, equal-mass bins")
    ax.legend(fontsize=8)
    save(
        fig,
        "reliability",
        "Reliability diagram of calibrated confidence against observed accuracy for both tiers.",
    )

    # quantization
    cfgs = res["quant"].get("configs", {})
    if cfgs:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
        names = list(cfgs)
        for ax, key, ttl in (
            (axes[0], "delta_auroc", "delta AUROC vs f16 (95% CI)"),
            (axes[1], "delta_ece", "delta ECE vs f16 (95% CI)"),
        ):
            for j, n in enumerate(names):
                d = cfgs[n][key]
                if d:
                    ax.errorbar(
                        [d["point"]],
                        [j],
                        xerr=[[d["point"] - d["lo"]], [d["hi"] - d["point"]]],
                        fmt="o",
                        color=PALETTE[0],
                        capsize=2,
                    )
            m = res["quant"]["margin"]
            ax.axvspan(-m, m, color="#999999", alpha=0.2, label="equivalence margin")
            ax.axvline(0, color="#777777", lw=0.8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=7)
            ax.set_title(ttl, fontsize=9)
        axes[0].legend(fontsize=7)
        save(
            fig,
            "quantization",
            "Change in ranking quality and calibration error of each quantized configuration relative to the f16 reference, with the equivalence margin shaded.",
        )
    return figs


# --- HTML -------------------------------------------------------------------------------------

_CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#555;--line:#d0d0d0;--acc:#0b5cad;--warn-bg:#fff3cd;--warn-fg:#5c4400;
--ok:#116329;--bad:#9b1c1c;--inc:#6b4e00}
@media (prefers-color-scheme:dark){:root{--bg:#121212;--fg:#ececec;--mut:#aaa;--line:#3a3a3a;--acc:#7ab7ff;
--warn-bg:#3b3000;--warn-fg:#ffe08a;--ok:#6fdc8c;--bad:#ff8a8a;--inc:#ffd27a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:70rem;margin:0 auto;padding:1rem}
a.skip{position:absolute;left:-999px}a.skip:focus{left:1rem;top:1rem;background:var(--bg);padding:.5rem}
h1{font-size:1.6rem}h2{margin-top:2rem;border-bottom:1px solid var(--line);padding-bottom:.25rem}
.banner{background:var(--warn-bg);color:var(--warn-fg);padding:.75rem 1rem;border-radius:6px;font-weight:600}
table{border-collapse:collapse;width:100%;margin:.75rem 0;font-size:.9rem}
.scroll{overflow-x:auto}caption{text-align:left;font-weight:600;padding:.25rem 0;color:var(--mut)}
th,td{border:1px solid var(--line);padding:.3rem .5rem;text-align:left;vertical-align:top}
th{background:rgba(127,127,127,.12)}img{max-width:100%;height:auto;border:1px solid var(--line)}
.v-SUPPORTED{color:var(--ok);font-weight:700}.v-NOTSUPPORTED{color:var(--bad);font-weight:700}
.v-INCONCLUSIVE{color:var(--inc);font-weight:700}code{font-size:.85em}small,.mut{color:var(--mut)}
"""


def _table_html(name: str, head: list[str], rows: list[list[Any]]) -> str:
    th = "".join(f'<th scope="col">{html.escape(h)}</th>' for h in head)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_fmt(x))}</td>" for x in r) + "</tr>" for r in rows
    )
    return (
        f'<div class="scroll"><table><caption>{html.escape(name.replace("_", " "))}</caption>'
        f"<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def build_html(
    res: dict[str, Any],
    hyp: dict[str, Any],
    tables: dict[str, tuple[list[str], list[list[Any]]]],
    figs: dict[str, tuple[Path, str]],
) -> str:
    m = res["meta"]
    syn = m.get("synthetic")
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>Sieve results{' (synthetic mock)' if syn else ''}</title>",
        f"<style>{_CSS}</style></head><body>",
        '<a class="skip" href="#main">Skip to results</a><main id="main">',
        "<h1>Sieve evaluation results</h1>",
    ]
    if syn:
        parts.append(
            f'<p class="banner" role="note">{html.escape(m["synthetic_label"])}. '
            "Nothing on this page is a result about any real model.</p>"
        )
    parts.append(
        "<h2>Provenance</h2><ul>"
        f"<li>config <code>{html.escape(m['name'])}</code> hash <code>{html.escape(str(m['config_hash'])[:16])}</code></li>"
        f"<li>commit <code>{html.escape(str(m.get('commit')))}</code></li>"
        f"<li>pre-registration frozen hash <code>{html.escape(str(hyp['preregistration_frozen_sha256'])[:16])}</code></li>"
        f"<li>hardware <code>{html.escape(json.dumps(m.get('hardware'), sort_keys=True))}</code></li>"
        f"<li>splits index <code>{html.escape(str(m['splits_index_hash'])[:16])}</code>, {m['n_splits']} re-splits of sizes {m['split_sizes']}</li>"
        f"<li>judge for SimpleQA: <code>{html.escape(str(m['judge']))}</code>; ungraded datasets: {html.escape(', '.join(m['ungraded_datasets']) or 'none')}</li>"
        f"<li>alpha={m['alpha']}, delta={m['delta']}, bootstrap replicates={m['n_boot']}</li></ul>"
    )
    parts.append("<h2>Hypotheses (pre-registered criteria, unedited)</h2>")
    parts.append(
        '<div class="scroll"><table><caption>Verdict per hypothesis</caption><thead><tr><th scope="col">ID</th><th scope="col">Hypothesis</th><th scope="col">Verdict</th><th scope="col">Why</th></tr></thead><tbody>'
    )
    for v in hyp["verdicts"]:
        cls = "v-" + v["verdict"].replace(" ", "")
        parts.append(
            f"<tr><td>{v['id']}</td><td>{html.escape(v['title'])}</td>"
            f'<td class="{cls}">{html.escape(v["verdict"])}</td><td>{html.escape(v["why"])}</td></tr>'
        )
    parts.append("</tbody></table></div>")
    neg = hyp["negative_outcomes"]
    parts.append(
        f"<p>Pre-declared negative outcomes observed: {html.escape('; '.join(neg) or 'none')}.</p>"
    )
    parts.append("<h2>Figures</h2>")
    for p, alt in figs.values():
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        parts.append(
            f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{b64}">'
            f"<figcaption>{html.escape(alt)}</figcaption></figure>"
        )
    parts.append("<h2>Tables</h2>")
    for name, (head, rows) in tables.items():
        parts.append(_table_html(name, head, rows))
    parts.append(
        "<h2>Pre-registration deviations</h2><pre>"
        + html.escape(hyp["deviations"] or "(none recorded)")
        + "</pre>"
    )
    parts.append(
        '<p class="mut">Generated by <code>sieve.eval</code>. Risk statements are about '
        "graded-wrong answers for exchangeable traffic and a frozen configuration, never per query.</p>"
    )
    parts.append("</main></body></html>")
    return "".join(parts)


def write_report(
    workdir: Path,
    res: dict[str, Any],
    hyp: dict[str, Any],
    cfg: ExperimentConfig,
    out: Path | None = None,
) -> Path:
    out = out or (workdir / "report")
    tables = build_tables(res, hyp)
    syn = bool(res["meta"].get("synthetic"))
    if syn:
        tables = mark_synthetic(tables)
    write_tables(tables, out / "tables", synthetic=syn)
    figs = make_figures(res, out / "figures")
    atomic_write_text(out / "index.html", build_html(res, hyp, tables, figs))
    atomic_write_text(out / "hypotheses.json", json.dumps(hyp, sort_keys=True, indent=1))
    return out / "index.html"

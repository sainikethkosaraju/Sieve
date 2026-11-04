"""Regenerate the README/RESULTS figures from the committed aggregate evidence.

    python scripts/make_figures.py            # writes docs/assets/*.png

Reads only results/evidence/** (aggregates). Nothing here is re-measured or tuned: every plotted number comes
from the pipeline's own output files, and the curves are empirical curves on the split-0 test partition
(descriptive, not themselves guarantees: the certified point is the one operating point with a bound).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V1 = ROOT / "results/evidence/t4_main_2026-11-03/analysis/results.json"
V1_H = ROOT / "results/evidence/t4_main_2026-11-03/analysis/hypotheses.json"
V2 = ROOT / "results/evidence/v2_squad_2026-11-03/analysis/results.json"
OUT = ROOT / "docs/assets"

C_L, C_S, C_G, C_BAD, C_GRID = "#1b6ca8", "#8a8f98", "#d95f02", "#b2182b", "#d9dde3"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)
FOOT = "Qwen2.5 1.5B (small) and 7B (large), Q4_K_M, llama.cpp, 2x Tesla T4. TriviaQA split-0 test set, n=2,000."


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _foot(fig: plt.Figure, text: str = FOOT) -> None:
    fig.text(0.01, -0.02, text, fontsize=8, color="#555", ha="left")


def hero(r: dict) -> None:
    sw = r["curves"]["pareto"]["sweep"]
    g0 = r["methods"]["gate:primary"]["split0"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for key, label, col in (("L_only", "7B model", C_L), ("S_only", "1.5B model", C_S)):
        pts = sorted(sw[key], key=lambda d: d["coverage"])
        ax.plot(
            [100 * d["coverage"] for d in pts],
            [100 * d["acc_on_answered"] for d in pts],
            color=col,
            lw=2.2,
            label=f"{label}: answer only when most confident",
        )
    ax.axhline(90, color=C_BAD, ls=":", lw=1.2)
    ax.text(99, 90.8, "10% error target", color=C_BAD, ha="right", fontsize=9)
    ax.scatter(
        [100 * g0["coverage"]],
        [100 * g0["accuracy_on_answered"]],
        s=140,
        color=C_G,
        zorder=5,
        edgecolor="white",
        linewidth=1.5,
        label="Sieve certified operating point",
    )
    ax.annotate(
        f"{100 * g0['accuracy_on_answered']:.0f}% correct\nwhile answering {100 * g0['coverage']:.0f}%",
        (100 * g0["coverage"], 100 * g0["accuracy_on_answered"]),
        (46, 76),
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": "#555"},
    )
    ax.set_xlabel("Share of questions answered (%)")
    ax.set_ylabel("Correct among answered (%)")
    ax.set_xlim(0, 101)
    ax.set_ylim(30, 100)
    ax.set_title(
        "Abstaining when unsure turns a 56%-accurate model into a 93%-accurate one", fontsize=11.5
    )
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    _foot(fig)
    fig.savefig(OUT / "fig1_selective_accuracy.png")
    plt.close(fig)


def wrong_share(r: dict) -> None:
    rc = r["curves"]["risk_coverage"]
    g0 = r["methods"]["gate:primary"]["split0"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for key, label, col in (("L", "7B model", C_L), ("S", "1.5B model", C_S)):
        pts = sorted(rc[key])
        ax.plot(
            [100 * c for c, _ in pts], [100 * c * k for c, k in pts], color=col, lw=2.2, label=label
        )
    wrong0 = 100 * g0["n_wrong_answered"] / g0["n"]
    ax.scatter(
        [100 * g0["coverage"]],
        [wrong0],
        s=140,
        color=C_G,
        zorder=5,
        edgecolor="white",
        linewidth=1.5,
        label="Sieve certified point",
    )
    full = 100 * max(c * k for c, k in rc["L"])
    ax.annotate(
        f"{wrong0:.1f}% of all queries wrong\n(vs about {full:.0f}% answering everything)",
        (100 * g0["coverage"], wrong0),
        (46, 18),
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": "#555"},
    )
    ax.set_xlabel("Share of questions answered (%)")
    ax.set_ylabel("Wrong answers, as % of ALL questions")
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 62)
    ax.set_title("The 'hallucination rate' falls about 21x at the certified point", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    _foot(fig)
    fig.savefig(OUT / "fig2_wrong_answer_rate.png")
    plt.close(fig)


def breakeven(r1: dict, h1: dict, r2: dict | None) -> None:
    t = h1["verdicts"][3]["numbers"]["mean_time_s"]  # H4: mean seconds per item
    ratio1 = t["L"] / t["S_greedy"]
    pts = [
        (
            "TriviaQA (v1)",
            ratio1,
            1 - r1["methods"]["gate:mean_logprob"]["across_splits"]["mean_escalation_rate"],
        )
    ]
    if r2 is not None:
        ts = {
            c: r2["accuracy"][c]["datasets"]["squad_v2"]["mean_time_s"]
            for c in ("S_q4km", "L_q4km")
        }
        pts.append(
            (
                "SQuAD 2.0 (v2)",
                ts["L_q4km"] / ts["S_q4km"],
                1 - r2["methods"]["gate:mean_logprob"]["across_splits"]["mean_escalation_rate"],
            )
        )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = [i / 200 for i in range(0, 201)]
    for ratio, col, ls in ((ratio1, C_L, "-"), (6.0, "#2ca25f", "--"), (10.0, "#756bb1", "--")):
        ax.plot(
            [100 * x for x in xs],
            [100 * (1 / ratio + 1 - x) for x in xs],
            color=col,
            ls=ls,
            lw=1.8,
            label=f"cost ratio large/small = {ratio:.1f}"
            + (" (measured)" if ratio == ratio1 else " (hypothetical)"),
        )
    ax.axhline(100, color="#222", lw=1.2)
    ax.text(99, 102, "cost of the large model alone", ha="right", fontsize=9)
    for k, (name, ratio, c) in enumerate(pts):
        y = 100 * (1 / ratio + 1 - c)
        ax.scatter([100 * c], [y], s=110, color=C_BAD, zorder=5, edgecolor="white", linewidth=1.3)
        ax.annotate(
            f"{name}: {100 * c:.1f}% resolved, cascade costs {y:.0f}% of large-only",
            (100 * c, y),
            (14, 158 - 15 * k),
            fontsize=9,
            va="center",
            arrowprops={"arrowstyle": "-", "color": "#555"},
        )
    ax.set_xlabel("Share of queries the small model can answer at the certified error level (%)")
    ax.set_ylabel("Cascade cost vs large model alone (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 170)
    ax.set_title(
        "When does a cascade pay? Only when the small tier resolves more than 1/ratio",
        fontsize=11,
        pad=10,
    )
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    _foot(
        fig,
        "cost = c_S + (1-c)*c_L per query; measured cost ratios and coverages from the v1 and v2 runs.",
    )
    fig.savefig(OUT / "fig3_cascade_breakeven.png")
    plt.close(fig)


def boundaries(r: dict) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    indist = r["methods"]["gate:primary"]["across_splits"]["mean_test_risk"]
    nq = r["shift"]["nq_open"]["across_splits"]["mean_test_risk"]
    a.bar(
        ["In distribution\n(TriviaQA)", "Shifted\n(NQ-open)"],
        [100 * indist, 100 * nq],
        color=[C_G, C_BAD],
        width=0.55,
    )
    a.axhline(10, color="#222", ls=":", lw=1.3)
    a.text(0.5, 11.2, "10% target", ha="center", fontsize=9)
    a.text(
        0,
        100 * indist / 2,
        f"{100 * indist:.1f}%",
        ha="center",
        va="center",
        fontsize=11,
        color="white",
    )
    a.text(1, 100 * nq + 1.2, f"{100 * nq:.1f}%", ha="center", fontsize=11)
    a.set_ylabel("Error among answered (%)")
    a.set_ylim(0, 50)
    a.set_title(
        "The bound holds on matching traffic;\nit breaks under shift and the alarm notices",
        fontsize=10.5,
    )
    q = r["quant"]["configs"]
    rows = [
        ("Q8_0 weights", "ladder_q8"),
        ("Q4_K_M weights", "ladder_q4km"),
        ("Q3_K_M weights", "ladder_q3km"),
        ("Q2_K weights", "ladder_q2k"),
        ("8-bit KV cache", "kv_q8"),
    ]
    ys = list(range(len(rows)))[::-1]
    for y, (lab, key) in zip(ys, rows, strict=True):
        d = q[key]["delta_auroc"]
        b.errorbar(
            d["point"],
            y,
            xerr=[[d["point"] - d["lo"]], [d["hi"] - d["point"]]],
            fmt="o",
            color=C_L,
            capsize=3,
        )
    b.axvspan(-0.02, 0.02, color="#e6f2e6", zorder=0)
    b.axvline(0, color="#222", lw=1)
    b.set_yticks(ys, [lab for lab, _ in rows])
    b.set_xlabel("Change in confidence ranking quality (AUROC) vs f16 weights")
    b.set_title(
        "Quantization: ranking degrades gradually\n(green band = pre-registered +-0.02 equivalence)",
        fontsize=10.5,
    )
    _foot(
        fig,
        "v1 run, 50 re-splits (left); paired bootstrap 95% CIs (right). 4-bit KV cache broke the model and is omitted.",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig4_boundaries.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r1, h1 = _load(V1), _load(V1_H)
    r2 = _load(V2) if V2.exists() else None
    hero(r1)
    wrong_share(r1)
    breakeven(r1, h1, r2)
    boundaries(r1)
    print("wrote", sorted(p.name for p in OUT.glob("fig*.png")))


if __name__ == "__main__":
    main()

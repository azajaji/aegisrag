"""Revision v2 figures: selective-QA, abstention diagnostics, and baselines.

Outputs to figures/revision_v2/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, RESULTS

REV_RESULTS = RESULTS / "revision_v2"
REV_FIG = FIGURES / "revision_v2"
REV_FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.right": False,
    "axes.spines.top": False,
})

PALETTE = {
    "BM25": "#5B6778",
    "NaiveRAG": "#9CB4CC",
    "RAG+Rerank": "#7AA88F",
    "HyDE": "#A6A66D",
    "AutoRAG": "#C25450",
    "RAG+Rerank+Abstain": "#A06DAA",
}

# Map internal system data-keys to their display labels. The JSON result
# files key our system as "AutoRAG"; the published name is "AegisRAG".
DISPLAY = {"AutoRAG": "AegisRAG"}


def disp(key):
    return DISPLAY.get(key, key)


def load_csv(path: Path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_main(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Figure 1: abstention answerable-retention vs unanswerable-refusal Pareto curve
# ---------------------------------------------------------------------------

def fig_abstention_tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    for ax, ds_key, label in zip(axes, ["squad", "enterprise"],
                                  ["SQuAD-v2", "Enterprise"]):
        rows = load_csv(REV_RESULTS / "abstention" / f"threshold_sweep_{ds_key}.csv")
        ans = np.array([float(r["answerable_retention"]) for r in rows])
        una = np.array([float(r["unanswerable_refusal"]) for r in rows])
        bal = np.array([float(r["balanced_abstention_accuracy"]) for r in rows])
        ax.scatter(ans, una, c=bal, s=8, cmap="viridis", alpha=0.7)
        # Pareto front (upper-right): keep points not dominated
        order = np.lexsort((-una, -ans))
        cur_max_una = -1.0
        front_x, front_y = [], []
        for i in order:
            if una[i] >= cur_max_una:
                front_x.append(ans[i]); front_y.append(una[i])
                cur_max_una = una[i]
        # sort the front by retention
        ix = np.argsort(front_x)
        ax.plot(np.array(front_x)[ix], np.array(front_y)[ix],
                color="#222", linewidth=1.2, label="Pareto frontier")
        # Mark operating point: max balanced accuracy
        ibest = int(bal.argmax())
        ax.scatter([ans[ibest]], [una[ibest]], color="#C25450",
                   edgecolor="black", linewidth=0.8, s=80, zorder=4,
                   label="Selected operating point")
        ax.set_xlabel("Answerable retention")
        if ax is axes[0]:
            ax.set_ylabel("Unanswerable refusal")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_title(label)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    axes[1].legend(loc="lower left", frameon=False)
    plt.suptitle("AegisRAG abstention trade-off (threshold sweep)", y=1.02)
    plt.savefig(REV_FIG / "abstention_tradeoff_curve.pdf")
    plt.savefig(REV_FIG / "abstention_tradeoff_curve.png")
    plt.close()
    print("wrote abstention_tradeoff_curve")


# ---------------------------------------------------------------------------
# Figure 2: coverage vs hallucination on unanswerable
# ---------------------------------------------------------------------------

def fig_coverage_vs_hallucination():
    rows = load_csv(REV_RESULTS / "baselines" / "expanded_baseline_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    for ax, ds_key, label in zip(axes, ["squad", "enterprise"],
                                  ["SQuAD-v2", "Enterprise"]):
        subset = [r for r in rows if r["dataset"] == ds_key]
        for r in subset:
            sys = r["system"]
            x = float(r["coverage_all"])
            y = float(r["hallucination_unanswerable"])
            ax.scatter(x, y, color=PALETTE.get(sys, "#444"), s=110,
                       edgecolor="black", linewidth=0.6, label=disp(sys), zorder=3)
            ax.annotate(disp(sys), (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.10)
        ax.set_xlabel("Coverage (all queries answered)")
        if ax is axes[0]:
            ax.set_ylabel("Hallucination rate on unanswerable")
        ax.set_title(label)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
        ax.axhline(0.0, color="#7AA88F", linestyle=":", linewidth=0.8)
    plt.suptitle("Selective-QA operating points across systems", y=1.02)
    plt.savefig(REV_FIG / "coverage_vs_hallucination.pdf")
    plt.savefig(REV_FIG / "coverage_vs_hallucination.png")
    plt.close()
    print("wrote coverage_vs_hallucination")


# ---------------------------------------------------------------------------
# Figure 3: top-1 reranker score histogram, margin histogram, and 2D scatter
# ---------------------------------------------------------------------------

def fig_abstention_diagnostics():
    for ds_key, fname in [("squad", "main_squad.json"),
                          ("enterprise", "main_enterprise.json")]:
        d = load_main(fname)
        pq = d["systems"]["AutoRAG"]["per_question"]
        ans_s = np.array([r["top_score"] for r in pq if r["answerable"]])
        una_s = np.array([r["top_score"] for r in pq if not r["answerable"]])
        ans_m = np.array([r["margin"] for r in pq if r["answerable"]])
        una_m = np.array([r["margin"] for r in pq if not r["answerable"]])
        # 3-panel
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
        bins_s = np.linspace(min(ans_s.min(), una_s.min()),
                             max(ans_s.max(), una_s.max()), 30)
        bins_m = np.linspace(min(ans_m.min(), una_m.min()),
                             max(ans_m.max(), una_m.max()), 30)
        axes[0].hist(ans_s, bins=bins_s, color="#7AA88F", alpha=0.65,
                     edgecolor="white", label="Answerable")
        axes[0].hist(una_s, bins=bins_s, color="#C25450", alpha=0.65,
                     edgecolor="white", label="Unanswerable")
        axes[0].set_xlabel("Top-1 reranker score")
        axes[0].set_ylabel("Number of questions")
        axes[0].set_title("Top-1 score distribution")
        axes[1].hist(ans_m, bins=bins_m, color="#7AA88F", alpha=0.65,
                     edgecolor="white", label="Answerable")
        axes[1].hist(una_m, bins=bins_m, color="#C25450", alpha=0.65,
                     edgecolor="white", label="Unanswerable")
        axes[1].set_xlabel("Top-1 – mean(top-2..5) margin")
        axes[1].set_title("Margin distribution")
        axes[2].scatter(ans_s, ans_m, color="#7AA88F", alpha=0.6, s=15,
                        edgecolor="none", label="Answerable")
        axes[2].scatter(una_s, una_m, color="#C25450", alpha=0.6, s=15,
                        edgecolor="none", label="Unanswerable")
        axes[2].set_xlabel("Top-1 reranker score")
        axes[2].set_ylabel("Margin")
        axes[2].set_title("Score × margin")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
        plt.suptitle(f"Abstention signal diagnostics ({ds_key.upper()})", y=1.02)
        fig.subplots_adjust(bottom=0.22, wspace=0.30)
        plt.savefig(REV_FIG / f"abstention_diagnostics_{ds_key}.pdf")
        plt.savefig(REV_FIG / f"abstention_diagnostics_{ds_key}.png")
        plt.close()
    print("wrote abstention_diagnostics_{squad,enterprise}")


# ---------------------------------------------------------------------------
# Figure 4: expanded baselines bar chart (selective-QA decomposition)
# ---------------------------------------------------------------------------

def fig_selective_qa_decomposition():
    rows = load_csv(REV_RESULTS / "baselines" / "expanded_baseline_metrics.csv")
    systems = ["BM25", "NaiveRAG", "RAG+Rerank", "HyDE",
               "RAG+Rerank+Abstain", "AutoRAG"]
    metrics = [
        ("coverage_all", "Coverage"),
        ("false_refusal_answerable", "False refusal (ans.)"),
        ("refusal_unanswerable", "Refusal (unans.)"),
        ("hallucination_unanswerable", "Halluc. (unans.)"),
        ("citation_when_answered", "Citation@answered"),
        ("f1_answered_only", "F1@answered"),
    ]
    n = len(systems)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), sharey=True)
    for ax, ds_key, label in zip(axes, ["squad", "enterprise"],
                                  ["SQuAD-v2", "Enterprise"]):
        x = np.arange(len(metrics))
        width = 0.14
        for i, sys in enumerate(systems):
            r = next((r for r in rows if r["dataset"] == ds_key and r["system"] == sys), None)
            if not r:
                continue
            vals = [float(r[k]) for k, _ in metrics]
            ax.bar(x + (i - (n - 1) / 2) * width, vals, width,
                   color=PALETTE.get(sys, "#444"), label=disp(sys))
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in metrics], fontsize=8,
                           rotation=25, ha="right")
        ax.set_title(label)
        ax.set_ylim(0, 1.18)
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), ncol=n, frameon=False)
    plt.suptitle("Decomposed selective-QA metrics across systems", y=1.02)
    fig.subplots_adjust(bottom=0.32, wspace=0.06)
    plt.savefig(REV_FIG / "selective_qa_decomposition.pdf")
    plt.savefig(REV_FIG / "selective_qa_decomposition.png")
    plt.close()
    print("wrote selective_qa_decomposition")


# ---------------------------------------------------------------------------
# Figure 5: repeated-split robustness boxplots
# ---------------------------------------------------------------------------

def fig_repeated_split():
    rows = load_csv(REV_RESULTS / "robustness" / "repeated_split_metrics.csv")
    systems = ["BM25", "NaiveRAG", "RAG+Rerank", "RAG+Rerank+Abstain", "AutoRAG"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0))
    metrics = [
        ("refusal_unanswerable", "Refusal on unanswerable"),
        ("false_refusal_answerable", "False refusal on answerable"),
        ("f1_answered_only", "F1 (answered answerable)"),
        ("citation_when_answered", "Citation precision when answered"),
    ]
    rng = np.random.default_rng(0)
    half = 0.13
    for ax, (mk, mlab) in zip(axes.flat, metrics):
        for ds_key, x_offset, color in [("squad", -0.18, "#2C7A5A"),
                                         ("enterprise", 0.18, "#B53A36")]:
            for s_idx, sys in enumerate(systems):
                vals = [float(r[mk]) for r in rows
                        if r["dataset"] == ds_key and r["system"] == sys]
                if not vals:
                    continue
                centre = s_idx + x_offset
                xs = centre + rng.uniform(-0.04, 0.04, size=len(vals))
                ax.vlines(centre, min(vals), max(vals),
                          color=color, linewidth=1.0, alpha=0.45, zorder=2)
                ax.scatter(xs, vals, s=24, color=color, alpha=0.85,
                           edgecolor="black", linewidth=0.4, zorder=3)
                med = float(np.median(vals))
                ax.hlines(med, centre - half, centre + half,
                          color="black", linewidth=1.4, zorder=4)
        ax.set_xticks(np.arange(len(systems)))
        ax.set_xticklabels([disp(s) for s in systems], rotation=20, ha="right", fontsize=8)
        ax.set_title(mlab)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(-0.6, len(systems) - 0.4)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2C7A5A",
               markeredgecolor="black", markersize=7, label="SQuAD-v2 (5 splits)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#B53A36",
               markeredgecolor="black", markersize=7, label="Enterprise (5 splits)"),
        Line2D([0], [0], color="black", linewidth=1.4, label="Median"),
    ]
    fig.legend(handles=legend_elems, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    plt.suptitle("Robustness across 5 repeated calibration/eval splits", y=1.00)
    fig.subplots_adjust(bottom=0.10, hspace=0.45, wspace=0.20)
    plt.savefig(REV_FIG / "repeated_split_robustness.pdf")
    plt.savefig(REV_FIG / "repeated_split_robustness.png")
    plt.close()
    print("wrote repeated_split_robustness")


# ---------------------------------------------------------------------------
# Figure 6: cost sensitivity by context size
# ---------------------------------------------------------------------------

def fig_cost_sensitivity():
    rows = load_csv(REV_RESULTS / "cost" / "cost_sensitivity.csv")
    models = ["gpt-4o-mini", "claude-haiku-4.5", "gpt-4o", "claude-sonnet-4.6"]
    ctx_sizes = ["short_top1", "standard_top3", "long_top5"]
    systems = ["RAG+Rerank", "RAG+Rerank+Abstain", "AutoRAG"]
    # Plot: per dataset, per model, savings vs RAG+Rerank for AutoRAG and RAG+Rerank+Abstain
    fig, axes = plt.subplots(2, len(models), figsize=(13, 5.5),
                             sharey="row", sharex="col")
    for col, model in enumerate(models):
        for row, ds in enumerate(["squad", "enterprise"]):
            ax = axes[row, col]
            xs = np.arange(len(ctx_sizes))
            width = 0.27
            for i, sys in enumerate(systems):
                vals = []
                for ctx in ctx_sizes:
                    v = next((float(r["cost_usd_per_1k"]) for r in rows
                              if r["dataset"] == ds and r["system"] == sys
                              and r["model"] == model and r["context_size"] == ctx), 0.0)
                    vals.append(v)
                ax.bar(xs + (i - 1) * width, vals, width,
                       color=PALETTE.get(sys, "#444"), label=disp(sys))
            ax.set_xticks(xs)
            ax.set_xticklabels(["top-1", "top-3", "top-5"], fontsize=8)
            if row == 0:
                ax.set_title(model, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{ds.upper()}\nUSD / 1k queries", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    plt.suptitle("Cost per 1,000 queries: by model and context size", y=1.00)
    fig.subplots_adjust(bottom=0.12, hspace=0.20, wspace=0.10)
    plt.savefig(REV_FIG / "cost_sensitivity_grid.pdf")
    plt.savefig(REV_FIG / "cost_sensitivity_grid.png")
    plt.close()
    print("wrote cost_sensitivity_grid")


# ---------------------------------------------------------------------------
# Figure 7: latency distribution boxplots
# ---------------------------------------------------------------------------

def fig_latency_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=False)
    for ax, ds_key, fname, hyde_fname, label in [
        (axes[0], "squad", "main_squad.json", "main_hyde_squad.json", "SQuAD-v2"),
        (axes[1], "enterprise", "main_enterprise.json", "main_hyde_enterprise.json", "Enterprise"),
    ]:
        d = load_main(fname)
        hyde_d = load_main(hyde_fname)
        data = []
        labels = []
        medians = []
        for sys in ["BM25", "NaiveRAG", "RAG+Rerank", "HyDE", "AutoRAG"]:
            if sys == "HyDE":
                pq = hyde_d["per_question"]
            else:
                pq = d["systems"][sys]["per_question"]
            xs = [r["latency"] * 1000.0 for r in pq]
            data.append(xs)
            labels.append(sys)
            medians.append(float(np.median(xs)))
        bp = ax.boxplot(
            data, tick_labels=[disp(s) for s in labels], showfliers=False, patch_artist=True,
            medianprops=dict(color="black", linewidth=1.8),
        )
        for patch, sys in zip(bp["boxes"], labels):
            patch.set_facecolor(PALETTE[sys])
            patch.set_alpha(0.7)
        # Annotate each box with its median value
        ymax = max(max(xs) for xs in data)
        for i, m in enumerate(medians, start=1):
            label_text = f"{m:.0f} ms" if m >= 10 else f"{m:.1f} ms"
            ax.annotate(
                label_text,
                xy=(i, m),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
            )
        ax.set_title(label)
        ax.set_ylabel("Latency per query (ms)")
        ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
        ax.tick_params(axis="x", rotation=20)
        # Headroom for the median labels on log scale
        ax.set_ylim(top=ymax * 2.0)
    plt.suptitle("End-to-end query latency distribution (CPU host) "
                 "-- median annotated on each box", y=1.02)
    fig.subplots_adjust(bottom=0.20, wspace=0.30)
    plt.savefig(REV_FIG / "latency_distribution.pdf")
    plt.savefig(REV_FIG / "latency_distribution.png")
    plt.close()
    print("wrote latency_distribution")


def main():
    fig_abstention_tradeoff()
    fig_coverage_vs_hallucination()
    fig_abstention_diagnostics()
    fig_selective_qa_decomposition()
    fig_repeated_split()
    fig_cost_sensitivity()
    fig_latency_distribution()
    print("Revision v2 figures written to", REV_FIG)


if __name__ == "__main__":
    main()

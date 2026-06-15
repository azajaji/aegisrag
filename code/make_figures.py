"""Generate publication figures (PDF) from summary/chunking/component JSONs.

Outputs to figures/:
  fig_architecture_placeholder.png   (skipped — drawn separately or omitted)
  fig_retrieval_squad.pdf            (Fig 4 — retrieval bar chart, SQuAD)
  fig_retrieval_enterprise.pdf
  fig_answer_quality.pdf             (Fig 5 — F1/citation/refusal)
  fig_cost_latency.pdf               (Fig 6 — quality vs latency)
  fig_per_domain.pdf                 (per-domain breakdown)
  fig_chunking.pdf                   (Recall@5 across chunking modes)
  fig_component.pdf                  (component ablation drops)
  fig_score_distribution.pdf         (reranker score on ans vs unans)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, RESULTS

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
}

# Map internal system data-keys to their display labels. The JSON result
# files key our system as "AutoRAG"; the published name is "AegisRAG".
DISPLAY = {"AutoRAG": "AegisRAG"}


def disp(key):
    return DISPLAY.get(key, key)


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def fig_retrieval(dataset_key: str, ds_label: str, out_name: str):
    summary = load("summary.json")[dataset_key]
    metrics = [("recall@1", "Recall@1"), ("recall@5", "Recall@5"),
               ("recall@10", "Recall@10"), ("mrr@10", "MRR@10"),
               ("ndcg@5", "nDCG@5")]
    systems = ["BM25", "NaiveRAG", "RAG+Rerank", "HyDE", "AutoRAG"]
    x = np.arange(len(metrics))
    n = len(systems)
    width = 0.16

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, s in enumerate(systems):
        if s not in summary["systems"]:
            continue
        vals = [summary["systems"][s][m] for m, _ in metrics]
        errs_low = [summary["systems"][s][m] - summary["systems"][s].get(m + "_ci95", [vals[k], vals[k]])[0]
                    for k, (m, _) in enumerate(metrics)]
        errs_high = [summary["systems"][s].get(m + "_ci95", [vals[k], vals[k]])[1] - summary["systems"][s][m]
                     for k, (m, _) in enumerate(metrics)]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=disp(s), color=PALETTE[s],
               yerr=[errs_low, errs_high], capsize=2, error_kw={"linewidth": 0.7, "ecolor": "#444"})
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Retrieval quality on {ds_label}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=n, frameon=False)
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(FIGURES / out_name)
    plt.close()


def fig_answer_quality():
    summary = load("summary.json")
    systems = ["BM25", "NaiveRAG", "RAG+Rerank", "HyDE", "AutoRAG"]
    rows = [
        ("F1",             "f1"),
        ("Citation@top1",  "citation"),
        ("Refusal(unans)", "refusal_correct_unans"),
        ("Halluc(unans)",  "hallucination_unans"),
    ]
    n = len(systems)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, ds_key, lbl in zip(axes, ["squad", "enterprise"],
                                ["SQuAD-v2", "Enterprise"]):
        data = summary[ds_key]
        x = np.arange(len(rows))
        width = 0.16
        for i, s in enumerate(systems):
            if s not in data["systems"]:
                continue
            vals = [data["systems"][s][k] for _, k in rows]
            xs = x + (i - (n - 1) / 2) * width
            ax.bar(xs, vals, width, label=disp(s), color=PALETTE[s],
                   edgecolor="black", linewidth=0.3)
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 0.025, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=5.5, rotation=0)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for lab, _ in rows], rotation=20, ha="right")
        ax.set_title(lbl)
        ax.set_ylim(0, 1.22)
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=n, frameon=False)
    plt.suptitle("Answer-quality and unanswerable-handling metrics", y=1.02)
    fig.subplots_adjust(bottom=0.30, wspace=0.10)
    plt.savefig(FIGURES / "fig_answer_quality.pdf")
    plt.close()


def fig_cost_latency():
    summary = load("summary.json")
    systems = ["BM25", "NaiveRAG", "RAG+Rerank", "HyDE", "AutoRAG"]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for s in systems:
        if s not in summary["squad"]["systems"]:
            continue
        sq = summary["squad"]["systems"][s]
        ax.scatter(sq["latency_mean_ms"], sq["f1"], s=90, color=PALETTE[s],
                   edgecolor="#222", linewidth=0.5, label=disp(s), zorder=3)
        ax.annotate(disp(s), (sq["latency_mean_ms"], sq["f1"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Mean latency per query (ms, log scale)")
    ax.set_ylabel("Answer F1 (SQuAD-v2 eval)")
    ax.set_title("Quality vs. cost trade-off")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.savefig(FIGURES / "fig_cost_latency.pdf")
    plt.close()


def fig_per_domain():
    s_full = load("summary.json")["squad"]
    by_dom = s_full["by_domain"]
    # Average each system's Recall@1 across domains, sorted by AutoRAG perf
    doms = sorted(by_dom.keys(), key=lambda d: -by_dom[d]["AutoRAG"]["recall@1"])
    systems = ["BM25", "NaiveRAG", "RAG+Rerank", "HyDE", "AutoRAG"]
    n = len(systems)
    x = np.arange(len(doms))
    width = 0.16
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for i, s in enumerate(systems):
        vals = [by_dom[d].get(s, {}).get("recall@1", 0.0) for d in doms]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=disp(s), color=PALETTE[s])
    ax.set_xticks(x)
    ax.set_xticklabels(doms, rotation=30, ha="right")
    ax.set_ylabel("Recall@1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-domain retrieval (SQuAD-v2 coarse topics)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42),
              ncol=n, frameon=False)
    plt.subplots_adjust(bottom=0.35)
    plt.savefig(FIGURES / "fig_per_domain.pdf")
    plt.close()


def fig_chunking():
    chunking = load("chunking.json")
    order = ["paragraph", "fixed_120", "fixed_300", "fixed_500", "semantic", "adaptive"]
    metrics = [("recall@5", "Recall@5"), ("f1", "Answer F1"), ("latency_mean_ms", "Latency (ms)")]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0))
    colors = ["#5B6778", "#9CB4CC", "#7AA88F", "#A6A66D", "#7B6FAA", "#C25450"]
    for ax, (mk, mlabel) in zip(axes, metrics):
        x = np.arange(len(order))
        width = 0.36
        for j, (ds, ds_label) in enumerate([("squad", "SQuAD"), ("enterprise", "Enterprise")]):
            vals = [chunking[ds]["variants"][m][mk] for m in order]
            ax.bar(x + (j - 0.5) * width, vals, width, color=colors[2 if j == 0 else 5],
                   label=ds_label, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_title(mlabel)
        if mk != "latency_mean_ms":
            ax.set_ylim(0, max(0.05, max(ax.get_ylim()[1], 1.0)))
        else:
            ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Score")
    axes[-1].set_ylabel("ms")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False)
    plt.suptitle("Chunking ablation", y=1.02)
    fig.subplots_adjust(bottom=0.30, wspace=0.30)
    plt.savefig(FIGURES / "fig_chunking.pdf")
    plt.close()


def fig_component():
    comp = load("component.json")
    ablations = [
        ("no_bm25",            "$-$ Hybrid BM25"),
        ("no_rerank",          "$-$ Cross-encoder rerank"),
        ("no_query_rewrite",   "$-$ Query rewrite"),
        ("no_abstain",         "$-$ Calibrated abstain"),
        ("no_adaptive_chunk",  "$-$ Adaptive chunking"),
    ]
    metrics = [
        ("recall@1",                 "Recall@1",       "#7AA88F"),
        ("f1",                       "F1",             "#9CB4CC"),
        ("refusal_correct_unans",    "Refusal(unans)", "#C25450"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
    height = 0.26
    for ax, ds, ds_label in zip(axes, ["squad", "enterprise"],
                                 ["SQuAD-v2", "Enterprise"]):
        d = comp[ds]["variants"]
        full = d["full"]
        y = np.arange(len(ablations))
        for i, (mk, mlabel, color) in enumerate(metrics):
            deltas = [d[a][mk] - full[mk] for a, _ in ablations]
            ax.barh(y + (i - 1) * height, deltas, height,
                    color=color, label=mlabel,
                    edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([lab for _, lab in ablations], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(ds_label)
        ax.set_xlabel("$\\Delta$ from Full AegisRAG")
        ax.set_xlim(-1.05, 0.35)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.4)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False)
    plt.suptitle("Component ablation: change in metric when each component is removed", y=1.02)
    fig.subplots_adjust(left=0.18, bottom=0.22, wspace=0.10)
    plt.savefig(FIGURES / "fig_component.pdf")
    plt.close()


def fig_score_distribution():
    r = json.load(open(RESULTS / "main_squad.json", "r", encoding="utf-8"))
    pq = r["systems"]["AutoRAG"]["per_question"]
    ans = [p["top_score"] for p in pq if p["answerable"]]
    una = [p["top_score"] for p in pq if not p["answerable"]]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bins = np.linspace(min(ans + una), max(ans + una), 30)
    ax.hist(ans, bins=bins, color="#7AA88F", alpha=0.7, label="Answerable", edgecolor="white")
    ax.hist(una, bins=bins, color="#C25450", alpha=0.7, label="Unanswerable", edgecolor="white")
    ax.set_xlabel("Top-1 reranker score (logit)")
    ax.set_ylabel("Number of questions")
    ax.set_title("AegisRAG reranker score by answerability (SQuAD-v2)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False)
    plt.savefig(FIGURES / "fig_score_distribution.pdf")
    plt.close()


def fig_pipeline():
    """End-to-end pipeline with explicit NOT_IN_CONTEXT branch."""
    fig, ax = plt.subplots(figsize=(8.0, 2.4))
    ax.axis("off")
    blocks = [
        "Questionnaire", "Upload\ndocs", "Adaptive\nchunking", "Embedding\nselection",
        "Hybrid\nretrieval", "Cross-enc.\nreranker", "Confidence\nabstain",
    ]
    bw = 1.0
    gap = 0.20
    bh = 0.85
    y0 = 0.55
    for i, b in enumerate(blocks):
        x = i * (bw + gap)
        ax.add_patch(plt.Rectangle((x, y0), bw, bh, facecolor="#EFE8E0",
                                   edgecolor="#444", linewidth=1.0))
        ax.text(x + bw / 2, y0 + bh / 2, b, ha="center", va="center",
                fontsize=8.5)
        if i < len(blocks) - 1:
            ax.annotate("", xy=(x + bw + gap - 0.02, y0 + bh / 2),
                        xytext=(x + bw + 0.02, y0 + bh / 2),
                        arrowprops=dict(arrowstyle="->", color="#444", lw=1))
    # Branch out of "Confidence abstain": NOT_IN_CONTEXT (down) or Cited answer (right)
    abstain_x = (len(blocks) - 1) * (bw + gap) + bw
    # Upper branch: cited answer
    cited_x = abstain_x + gap + 0.4
    ax.add_patch(plt.Rectangle((cited_x, y0 + bh / 2 + 0.05), bw, bh / 2,
                               facecolor="#CDE3D6", edgecolor="#444"))
    ax.text(cited_x + bw / 2, y0 + bh / 2 + 0.05 + bh / 4, "Cited\nanswer",
            ha="center", va="center", fontsize=8.5)
    ax.annotate("", xy=(cited_x - 0.01, y0 + bh / 2 + 0.05 + bh / 4),
                xytext=(abstain_x + 0.01, y0 + bh / 2),
                arrowprops=dict(arrowstyle="->", color="#3A6A4E", lw=1.2))
    ax.text((abstain_x + cited_x) / 2, y0 + bh / 2 + 0.32, "answer", fontsize=7.5,
            color="#3A6A4E", ha="center")
    # Lower branch: NOT_IN_CONTEXT
    nic_x = abstain_x + gap + 0.4
    ax.add_patch(plt.Rectangle((nic_x, y0 - bh / 2 - 0.05), bw, bh / 2,
                               facecolor="#F4DAD7", edgecolor="#444"))
    ax.text(nic_x + bw / 2, y0 - bh / 4 - 0.05, "Refusal\nsentinel",
            ha="center", va="center", fontsize=8.0)
    ax.annotate("", xy=(nic_x - 0.01, y0 - bh / 4 - 0.05),
                xytext=(abstain_x + 0.01, y0 + bh / 2 - 0.05),
                arrowprops=dict(arrowstyle="->", color="#A03A3A", lw=1.2))
    ax.text((abstain_x + nic_x) / 2 - 0.05, y0 + 0.05, "abstain", fontsize=7.5,
            color="#A03A3A", ha="center")
    ax.set_xlim(-0.15, nic_x + bw + 0.2)
    ax.set_ylim(y0 - bh / 2 - 0.4, y0 + bh + 0.5)
    ax.set_aspect("equal")
    plt.savefig(FIGURES / "fig_pipeline.pdf")
    plt.close()


def fig_architecture():
    """Five-layer architecture: User -> Configuration -> Retrieval -> Safety -> Answer.

    Drawn in inches (not [0,1] normalised) so the boxes are wide enough
    for the text and the layout is robust under different figure widths.
    """
    layers = [
        ("User",          "#E7CFC0", ["Questionnaire", "Document upload", "Workspace"]),
        ("Configuration", "#D7E0EA", ["Adaptive chunking", "Embedding selection",
                                       "Retriever wiring", "Index persistence"]),
        ("Retrieval",     "#CDE3D6", ["Query rewriting", "Hybrid BM25+dense",
                                       "Cross-encoder rerank"]),
        ("Safety",        "#F4DAD7", ["Confidence scoring",
                                       "Calibrated abstain",
                                       "Refusal sentinel"]),
        ("Answer",        "#E8DFC4", ["Citation grounding", "Cited answer",
                                       "Eval dashboard"]),
    ]
    # Geometry in figure inches
    col_w = 1.5
    gap   = 0.42
    pad   = 0.25
    fig_h = 4.4
    fig_w = pad * 2 + len(layers) * col_w + (len(layers) - 1) * gap
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")

    layer_top = fig_h - 0.35
    layer_bot = 0.55
    layer_h = layer_top - layer_bot
    for i, (name, color, rows) in enumerate(layers):
        cx = pad + i * (col_w + gap)
        ax.add_patch(plt.Rectangle((cx, layer_bot), col_w, layer_h,
                                   facecolor=color, edgecolor="#444", linewidth=1.0))
        ax.text(cx + col_w / 2, layer_top - 0.18, name,
                ha="center", va="top", fontsize=10, fontweight="bold")
        n_rows = len(rows)
        inner_top = layer_top - 0.55
        inner_bot = layer_bot + 0.18
        inner_h = inner_top - inner_bot
        h = min(0.45, inner_h / n_rows - 0.06)
        for j, r in enumerate(rows):
            y_center = inner_top - (j + 0.5) * (inner_h / n_rows)
            ax.add_patch(plt.Rectangle((cx + 0.08, y_center - h / 2),
                                       col_w - 0.16, h,
                                       facecolor="#FFFFFF", edgecolor="#888", linewidth=0.6))
            ax.text(cx + col_w / 2, y_center, r,
                    ha="center", va="center", fontsize=7.5)
        if i < len(layers) - 1:
            arr_y = (layer_top + layer_bot) / 2
            x0 = cx + col_w + 0.06
            x1 = cx + col_w + gap - 0.06
            ax.annotate("", xy=(x1, arr_y), xytext=(x0, arr_y),
                        arrowprops=dict(arrowstyle="->", color="#444", lw=1.3))
    ax.text(fig_w / 2, 0.25, "Questionnaire-driven; no user-facing parameter knobs.",
            ha="center", va="center", fontsize=8.0, style="italic", color="#555")
    plt.savefig(FIGURES / "fig_architecture.pdf")
    plt.close()


def main():
    fig_architecture()
    fig_pipeline()
    fig_retrieval("squad", "SQuAD-v2 (eval split)", "fig_retrieval_squad.pdf")
    fig_retrieval("enterprise", "AegisRAG-Enterprise (eval split)", "fig_retrieval_enterprise.pdf")
    fig_answer_quality()
    fig_cost_latency()
    fig_per_domain()
    fig_chunking()
    fig_component()
    fig_score_distribution()
    print("Figures written to", FIGURES)


if __name__ == "__main__":
    main()

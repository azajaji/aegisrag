"""Generate the additional figures introduced in v2:
  fig_nfcorpus.pdf         -- NFCorpus retrieval (nDCG@10, R@10, R@100, MRR)
  fig_llm_answers.pdf      -- LLM-generated answer quality (F1, Refusal)
  fig_expert.pdf           -- Pilot rubric (mean usefulness, %cite-OK)
  fig_cost.pdf             -- Cost per 1K queries across commercial APIs
"""
from __future__ import annotations

import json

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
    "AutoRAG": "#C25450",
}

# Map internal system data-keys to their display labels. The JSON result
# files key our system as "AutoRAG"; the published name is "AegisRAG".
DISPLAY = {"AutoRAG": "AegisRAG"}


def disp(key):
    return DISPLAY.get(key, key)


def fig_nfcorpus():
    r = json.load(open(RESULTS / "main_nfcorpus.json", "r", encoding="utf-8"))
    systems = ["BM25", "NaiveRAG", "RAG+Rerank", "AutoRAG"]
    metrics = [("ndcg@10", "nDCG@10"), ("recall@10", "Recall@10"),
               ("recall@100", "Recall@100"), ("mrr@10", "MRR@10")]
    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for i, s in enumerate(systems):
        agg = r["systems"][s]["aggregate"]
        vals = [agg[k] for k, _ in metrics]
        ax.bar(x + (i - 1.5) * width, vals, width, label=disp(s), color=PALETTE[s])
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(0.6, max(r["systems"][s]["aggregate"][m] for s in systems for m, _ in metrics) * 1.15))
    ax.set_title("Retrieval on NFCorpus (BEIR test, 323 queries)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, frameon=False)
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(FIGURES / "fig_nfcorpus.pdf")
    plt.close()


def fig_llm_answers():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), sharey=True)
    systems = ["BM25", "NaiveRAG", "RAG+Rerank", "AutoRAG"]
    for ax, ds, lbl in zip(axes, ["enterprise", "squad"],
                           ["Enterprise (183 q eval)", "SQuAD-v2 (120 q eval)"]):
        r = json.load(open(RESULTS / f"llm_{ds}.json", "r", encoding="utf-8"))
        rows_by_sys = {s: [p for p in r["systems"][s]["per_question"] if not p["in_calibration"]] for s in systems}
        metrics = [
            ("F1", lambda p: p["f1"] if p["answerable"] else float("nan")),
            ("EM", lambda p: p["em"] if p["answerable"] else float("nan")),
            ("Cite",  lambda p: p["citation"] if p["answerable"] else float("nan")),
            ("Refusal(un)", lambda p: p["refusal_correct"] if not p["answerable"] else float("nan")),
            ("Halluc(un)", lambda p: p["hallucination"] if not p["answerable"] else float("nan")),
        ]
        x = np.arange(len(metrics))
        width = 0.2
        for i, s in enumerate(systems):
            vals = []
            for _, fn in metrics:
                vs = [fn(p) for p in rows_by_sys[s]]
                vs = [v for v in vs if v == v]
                vals.append(float(np.mean(vs)) if vs else 0.0)
            ax.bar(x + (i - 1.5) * width, vals, width, color=PALETTE[s], label=disp(s))
        ax.set_xticks(x)
        ax.set_xticklabels([m for m, _ in metrics], rotation=20, ha="right")
        ax.set_title(lbl)
        ax.set_ylim(0, 1.18)
    axes[0].set_ylabel("Score")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False)
    plt.suptitle("LLM-generated answers (FLAN-T5-Base, top-3 context)", y=1.02)
    fig.subplots_adjust(bottom=0.30, wspace=0.10)
    plt.savefig(FIGURES / "fig_llm_answers.pdf")
    plt.close()


def fig_expert():
    r = json.load(open(RESULTS / "expert.json", "r", encoding="utf-8"))
    systems = ["BM25", "NaiveRAG", "RAG+Rerank", "AutoRAG"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    panel_labels = {"enterprise": "Enterprise audit (n=47)",
                    "squad": "SQuAD-v2 audit (n=60)"}
    for ax, ds in zip(axes, ["enterprise", "squad"]):
        d = r[ds]["per_system"]
        x = np.arange(2)
        width = 0.18
        for i, s in enumerate(systems):
            vals = [d[s]["mean_usefulness"] / 3.0, d[s]["pct_citation_ok"]]
            ax.bar(x + (i - 1.5) * width, vals, width, color=PALETTE[s], label=disp(s))
        ax.set_xticks(x)
        ax.set_xticklabels(["Usefulness\n(0-3 → /3)", "Citation OK"])
        ax.set_ylim(0, 1.18)
        ax.set_title(panel_labels[ds])
    axes[0].set_ylabel("Score")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False)
    plt.suptitle("Single-rater rubric audit", y=1.02)
    fig.subplots_adjust(bottom=0.26, wspace=0.18)
    plt.savefig(FIGURES / "fig_expert.pdf")
    plt.close()


def fig_cost():
    c = json.load(open(RESULTS / "cost.json", "r", encoding="utf-8"))
    systems = ["BM25", "NaiveRAG", "RAG+Rerank", "AutoRAG"]
    models = ["GPT-4o-mini", "Claude Haiku 4.5", "GPT-4o", "Claude Sonnet 4.6"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for ax, ds, lbl in zip(axes, ["enterprise", "squad"], ["Enterprise", "SQuAD-v2"]):
        d = c[ds]["cost_per_1k_queries"]
        x = np.arange(len(models))
        width = 0.18
        for i, s in enumerate(systems):
            vals = [d[s][m] for m in models]
            ax.bar(x + (i - 1.5) * width, vals, width, color=PALETTE[s], label=disp(s))
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel("USD per 1,000 queries")
        ax.set_title(lbl)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False)
    plt.suptitle("Commercial LLM cost per 1,000 queries (incl. abstain-skip savings)", y=1.02)
    fig.subplots_adjust(bottom=0.30, wspace=0.20)
    plt.savefig(FIGURES / "fig_cost.pdf")
    plt.close()


def fig_config_burden():
    c = json.load(open(RESULTS / "config_burden.json", "r", encoding="utf-8"))
    items = list(c["frameworks"].items())
    # Short labels for plotting
    short = {
        "LangChain (RetrievalQA + Chroma + Cohere rerank)": "LangChain",
        "LlamaIndex (VectorStoreIndex + node post-processors)": "LlamaIndex",
        "AWS Bedrock Knowledge Bases (CreateKB + Retrieve API)": "Bedrock KB",
        "AutoRAG (questionnaire-driven)": "AegisRAG",
    }
    labels = [short[n] for n, _ in items]
    decisions = [info["n_decisions"] for _, info in items]
    params    = [info["n_parameters"] for _, info in items]
    code      = [info["code_lines"] for _, info in items]
    colors = ["#5B6778", "#9CB4CC", "#7AA88F", "#C25450"]
    x = np.arange(len(items))
    width = 0.28
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.bar(x - width, decisions, width, label="User-facing decisions", color="#5B6778")
    ax.bar(x,         params,    width, label="Required parameters",   color="#9CB4CC")
    ax.bar(x + width, code,      width, label="Lines of code/config",  color="#C25450")
    for xi, (d, p, k) in enumerate(zip(decisions, params, code)):
        ax.text(xi - width, d + 0.7, f"{d}", ha="center", fontsize=8)
        ax.text(xi,         p + 0.7, f"{p}", ha="center", fontsize=8)
        ax.text(xi + width, k + 0.7, f"{k}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_ylim(top=max(decisions + params + code) * 1.18)
    ax.set_title("Configuration burden to deploy a comparable RAG pipeline")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=3, frameon=False)
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(FIGURES / "fig_config_burden.pdf")
    plt.close()


def fig_failure_taxonomy():
    f = json.load(open(RESULTS / "failures.json", "r", encoding="utf-8"))
    by_mode = f["by_mode"]
    fixed = f["n_fixed_by_autorag_vs"]
    order = ["lexical_mismatch", "reranker_only", "fragmentation", "near_duplicate"]
    pretty_modes = {
        "lexical_mismatch": "Lexical mismatch",
        "reranker_only":    "Reranker demotion",
        "fragmentation":    "Fragmentation",
        "near_duplicate":   "Near-duplicate",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    # Panel A: AutoRAG remaining misses by mode
    a = axes[0]
    vals = [by_mode.get(k, 0) for k in order]
    a.bar(np.arange(len(order)), vals,
          color=["#C25450", "#A58450", "#7AA88F", "#5B6778"])
    a.set_xticks(np.arange(len(order)))
    a.set_xticklabels([pretty_modes[k] for k in order],
                      rotation=20, ha="right", fontsize=9)
    a.set_ylabel("AegisRAG R@1 misses")
    a.set_title("Remaining failure modes (SQuAD-v2 eval)", fontsize=10)
    a.set_ylim(0, max(vals) * 1.20 + 1)
    for i, v in enumerate(vals):
        a.text(i, v + 0.3, str(v), ha="center", fontsize=9)
    # Panel B: baseline failures fixed by AutoRAG
    b = axes[1]
    bases = ["BM25", "NaiveRAG", "RAG+Rerank"]
    fixed_vals = [fixed[k] for k in bases]
    b.bar(np.arange(len(bases)), fixed_vals,
          color=["#5B6778", "#9CB4CC", "#7AA88F"])
    b.set_xticks(np.arange(len(bases)))
    b.set_xticklabels(bases, fontsize=9)
    b.set_ylabel("Queries newly correct under AegisRAG")
    b.set_title("Baseline failures fixed by AegisRAG", fontsize=10)
    b.set_ylim(0, max(fixed_vals) * 1.18 + 1)
    for i, v in enumerate(fixed_vals):
        b.text(i, v + 1.0, str(v), ha="center", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.88,
                        wspace=0.32)
    plt.savefig(FIGURES / "fig_failures.pdf")
    plt.close()


def main():
    fig_nfcorpus()
    fig_llm_answers()
    fig_expert()
    fig_cost()
    fig_config_burden()
    fig_failure_taxonomy()
    print("Wrote NFCorpus, LLM, expert, cost, config-burden, failure figures to", FIGURES)


if __name__ == "__main__":
    main()

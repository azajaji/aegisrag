"""Consolidated Cost/Latency/Quality Pareto table per Section 12.

For each (dataset, system) report:
  - Recall@1, Citation precision, Refusal accuracy, Halluc rate,
    Coverage, Latency, LLM call rate, projected cost per 1k queries
    at GPT-4o rates, cost reduction vs RAG+Rerank.
"""
import json

import numpy as np

from common import RESULTS


# GPT-4o input + output token prices per 1M tokens (USD), May 2026 reference.
PRICE_PER_M_IN = 2.50
PRICE_PER_M_OUT = 10.00
PROMPT_TOKENS = 284     # ~3-chunk context window
OUTPUT_TOKENS = 7       # observed short answer length


def cost_per_1k(call_rate: float) -> float:
    in_cost = (PROMPT_TOKENS * call_rate) * (PRICE_PER_M_IN / 1_000_000) * 1000
    out_cost = (OUTPUT_TOKENS * call_rate) * (PRICE_PER_M_OUT / 1_000_000) * 1000
    return in_cost + out_cost


def main():
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for ds in ["squad", "enterprise"]:
        data = summary[ds]
        baseline_cost = None
        for sys in ["BM25", "NaiveRAG", "RAG+Rerank", "HyDE", "AutoRAG"]:
            s = data["systems"].get(sys)
            if s is None:
                continue
            n_total = s["n_evaluation_total"]
            # Coverage: number of answered queries / total
            # For non-abstaining systems, coverage = 1.0
            # For AutoRAG, coverage = (n_answerable_answered + n_unans_not_refused) / n_total
            n_ans_answered = s.get("citation_when_answered_n", s["n_evaluation_answerable"])
            n_unans_refused = round(s["refusal_correct_unans"] * s["n_evaluation_unanswerable"])
            n_unans_answered = s["n_evaluation_unanswerable"] - n_unans_refused
            n_total_answered = n_ans_answered + n_unans_answered
            coverage = n_total_answered / n_total
            llm_call_rate = coverage   # one call per answered query, zero on abstain
            cost = cost_per_1k(llm_call_rate)
            if sys == "RAG+Rerank":
                baseline_cost = cost
            cost_red = (1 - cost / baseline_cost) * 100 if baseline_cost else 0.0
            rows.append({
                "dataset": ds, "system": sys,
                "recall@1": s["recall@1"],
                "citation_when_answered": s["citation_when_answered"],
                "refusal_unans": s["refusal_correct_unans"],
                "halluc_unans": s["hallucination_unans"],
                "coverage": coverage,
                "latency_ms": s["latency_mean_ms"],
                "llm_call_rate": llm_call_rate,
                "cost_per_1k_usd": cost,
                "cost_reduction_vs_rag_rerank": cost_red,
            })

    (RESULTS / "pareto_table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"{'Dataset':12}{'System':16}{'R@1':>6}{'Cite':>7}{'Refuse':>8}{'Halluc':>8}"
          f"{'Cov':>6}{'Lat(ms)':>9}{'LLM%':>6}{'$/1k':>8}{'-cost%':>9}")
    for r in rows:
        print(f"{r['dataset']:12}{r['system']:16}{r['recall@1']:>6.3f}{r['citation_when_answered']:>7.3f}"
              f"{r['refusal_unans']:>8.3f}{r['halluc_unans']:>8.3f}{r['coverage']:>6.2f}"
              f"{r['latency_ms']:>9.1f}{100*r['llm_call_rate']:>6.1f}"
              f"{r['cost_per_1k_usd']:>8.3f}{r['cost_reduction_vs_rag_rerank']:>9.1f}")


if __name__ == "__main__":
    main()

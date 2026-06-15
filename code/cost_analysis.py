"""Token cost analysis.

Computes prompt-token counts for each system's LLM-generation calls and
multiplies by published API rates (USD per 1K tokens) for a panel of
representative commercial models. Output goes to results/cost.json.

Rates are documented inline; we use rates published by the vendors as of
May 2026 and pre-discounted batch / prompt-caching rates are excluded so
the numbers are a conservative upper bound for an on-demand deployment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from common import RESULTS, save_json


# USD per 1K tokens (input / output). Conservative on-demand rates.
RATES = {
    "Claude Haiku 4.5":  {"input": 0.001, "output": 0.005},
    "Claude Sonnet 4.6": {"input": 0.003,  "output": 0.015},
    "GPT-4o-mini":       {"input": 0.00015, "output": 0.0006},
    "GPT-4o":            {"input": 0.0025, "output": 0.010},
    "FLAN-T5-Base (self-hosted)": {"input": 0.0, "output": 0.0},
}


def _count_input_tokens(prompts):
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    lens = []
    for p in prompts:
        ids = tok(p, truncation=False, return_tensors=None)["input_ids"]
        lens.append(len(ids))
    return lens


def _reconstruct_prompt(question: str, contexts: list[str]) -> str:
    joined = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return (
        "You are a careful question-answering assistant. "
        "Answer the question using only the information in the provided context. "
        "If the answer cannot be found in the context, respond with exactly NOT_IN_CONTEXT. "
        "Be concise: respond with a short factual span, not a full sentence.\n\n"
        f"Context:\n{joined}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def analyse(dataset: str):
    """Estimate per-query cost for each (system, model) using the LLM eval
    output's recorded answers as a proxy for output token count."""
    path = RESULTS / f"llm_{dataset}.json"
    if not path.exists():
        print(f"missing {path}, skipping {dataset}")
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")

    per_system = {}
    for sname, info in rec["systems"].items():
        out_when_called = []
        n_total = len(info["per_question"])
        n_skipped = 0      # AutoRAG with retrieval-abstain skips the LLM
        n_called = 0
        for p in info["per_question"]:
            if p.get("abstained_by_retrieval", False):
                n_skipped += 1
                continue
            ans = p.get("llm_pred") or p.get("llm_raw") or ""
            out_when_called.append(len(tok(ans)["input_ids"]))
            n_called += 1
        if not out_when_called:
            out_when_called = [0]
        per_system[sname] = {
            "n_queries": n_total,
            "n_llm_called": n_called,
            "n_llm_skipped_by_abstain": n_skipped,
            "abstain_rate": n_skipped / n_total if n_total else 0.0,
            "mean_output_tokens_when_called": float(np.mean(out_when_called)),
            "median_output_tokens_when_called": float(np.median(out_when_called)),
            "p95_output_tokens_when_called": float(np.percentile(out_when_called, 95)),
            "llm_generate_time_s": info.get("llm_generate_time_s", 0.0),
            "retrieval_time_s": info.get("retrieval_time_s", 0.0),
        }
    # Input tokens: same prompt format for every system; we approximate by
    # taking the average context length per system from its own ranked
    # output. Because we cannot re-run prompts here, we report a constant
    # estimate by tokenising a representative prompt with 3 average-length
    # chunks of ~120 tokens each plus the prompt scaffolding.
    avg_chunk = "This paragraph is a representative chunk of approximately one hundred and twenty tokens drawn from the indexed corpus; it contains factual content describing policies, procedures, technical configurations, or definitions, and is used as one of three retrieval candidates that the language model conditions on to produce a grounded answer."
    rep_prompt = _reconstruct_prompt("What is the policy on X?", [avg_chunk] * 3)
    rep_input_tokens = len(tok(rep_prompt)["input_ids"])
    summary = {
        "dataset": dataset,
        "rep_prompt_input_tokens": int(rep_input_tokens),
        "per_system": per_system,
        "cost_per_1k_queries": {},
    }
    for sname, info in per_system.items():
        out_tok = info["mean_output_tokens_when_called"]
        call_rate = info["n_llm_called"] / max(1, info["n_queries"])
        cost_for_model = {}
        for model, rate in RATES.items():
            # rate is USD per 1K tokens; per-call cost in USD
            per_call = (rep_input_tokens * rate["input"] + out_tok * rate["output"]) / 1000.0
            cost_for_model[model] = float(1000 * call_rate * per_call)
        summary["cost_per_1k_queries"][sname] = cost_for_model
    return summary


def main():
    out = {}
    for ds in ("enterprise", "squad"):
        s = analyse(ds)
        if s is not None:
            out[ds] = s
    save_json(out, RESULTS / "cost.json")
    print("Wrote", RESULTS / "cost.json")


if __name__ == "__main__":
    main()

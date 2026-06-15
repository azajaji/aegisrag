"""Encoder ablation: run AutoRAG with alternative dense encoders.

Addresses the reviewer concern that `all-MiniLM-L6-v2` is a small
encoder and the headline numbers might be encoder-cherry-picked.

We re-run AutoRAG on SQuAD-v2 (full 700-question stratified set) and
Enterprise (full 305-question set) with three encoders:

  * all-MiniLM-L6-v2          (default, 22M params, 384-d) -- headline
  * all-mpnet-base-v2         (110M params, 768-d)         -- larger SBERT
  * BAAI/bge-base-en-v1.5     (109M params, 768-d)         -- state-of-art small

All other components (BM25 + dense fusion, cross-encoder reranker,
calibrated abstention with the SAME calibration grid, query rewriting,
adaptive chunking) are held fixed; only the dense encoder changes.

Writes results/encoder_ablation.json with full retrieval and
selective-QA metrics per (dataset, encoder).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import RESULTS, save_json, recall_at_k, mrr_first, ndcg_at_k, best_against
from data_loaders import load_enterprise, load_squad
from systems import AutoRAG
from run_main import calibrate_abstain


ENCODERS = [
    ("MiniLM-L6", "sentence-transformers/all-MiniLM-L6-v2"),
    ("MPNet-base", "sentence-transformers/all-mpnet-base-v2"),
    ("BGE-base-en-v1.5", "BAAI/bge-base-en-v1.5"),
]


def evaluate_question(q: dict, out: dict) -> dict:
    pred = out["prediction"]
    gold = (q.get("gold_doc"), q.get("gold_para_idx"))
    ranked = out["ranked"]
    r1 = recall_at_k(ranked, gold, 1)
    r5 = recall_at_k(ranked, gold, 5)
    r10 = recall_at_k(ranked, gold, 10)
    mrr = mrr_first(ranked, gold, 10)
    ndcg5 = ndcg_at_k(ranked, gold, 5)
    if q["answerable"]:
        em, f1 = best_against(pred, q["gold_answers"])
        cite_ok = float(out["cite_doc"] == gold[0] and out["cite_para"] == gold[1])
    else:
        em, f1 = best_against("" if not pred else pred, [])
        cite_ok = 0.0
    return {
        "qid": q["qid"], "answerable": q["answerable"],
        "domain": q.get("domain"),
        "recall@1": r1, "recall@5": r5, "recall@10": r10,
        "mrr@10": mrr, "ndcg@5": ndcg5,
        "em": em, "f1": f1, "citation": cite_ok,
        "top_score": out["top_score"], "margin": out["margin"],
        "latency": out["latency"],
        "abstained": out.get("abstained", False),
    }


def run_dataset(name: str, corpus, questions, encoder_name: str) -> dict:
    print(f"\n[{name} / {encoder_name}] {len(corpus)} docs, {len(questions)} questions", flush=True)
    rng = np.random.default_rng(20260512)
    indices = np.arange(len(questions))
    rng.shuffle(indices)
    cal_size = int(0.4 * len(questions))
    cal_set = set(indices[:cal_size].tolist())

    # AutoRAG with the swapped encoder. First pass: disable abstention to
    # get raw logits, then calibrate, then apply.
    sys_raw = AutoRAG(encoder_name=encoder_name, use_abstain=False)
    print(f"  indexing ...", flush=True)
    sys_raw.index(corpus)
    print(f"  indexed {len(sys_raw.chunks)} chunks in {sys_raw._index_time:.1f}s", flush=True)
    raw_records = []
    t0 = time.perf_counter()
    for i, q in enumerate(questions):
        out = sys_raw.answer(q["question"])
        rec = evaluate_question(q, out)
        rec["in_calibration"] = i in cal_set
        raw_records.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  [{name}/{encoder_name}] {i + 1}/{len(questions)}", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"  raw pass done in {elapsed:.0f}s", flush=True)

    # Calibrate on the calibration split.
    cal_rows = [r for r in raw_records if r["in_calibration"]]
    ba, tau_s, tau_m = calibrate_abstain(cal_rows)
    print(f"  calibrated: bal_acc={ba:.3f}  tau_s={tau_s}  tau_m={tau_m}", flush=True)

    # Apply calibration retroactively.
    per_q = []
    for r in raw_records:
        rec = dict(r)
        # Apply abstention: if top_score < tau_s or margin < tau_m, abstain.
        abstained = (r["top_score"] < tau_s) or (r["margin"] < tau_m)
        if abstained:
            # Treat as refusal: rerank metrics stay, but prediction-level
            # metrics flip to the no-answer convention.
            if r["answerable"]:
                rec["em"] = 0.0
                rec["f1"] = 0.0
                rec["citation"] = 0.0
            else:
                rec["em"] = 1.0
                rec["f1"] = 1.0
                rec["citation"] = 1.0
        rec["abstained"] = abstained
        per_q.append(rec)

    # Aggregate on eval split only.
    eval_rows = [r for r in per_q if not r["in_calibration"]]
    ans_rows = [r for r in eval_rows if r["answerable"]]
    una_rows = [r for r in eval_rows if not r["answerable"]]
    n_ans = len(ans_rows)
    n_una = len(una_rows)

    answered_ans = [r for r in ans_rows if not r["abstained"]]
    answered_all = [r for r in eval_rows if not r["abstained"]]
    refused_unans = [r for r in una_rows if r["abstained"]]
    halluc_unans = [r for r in una_rows if not r["abstained"]]
    false_refusal = [r for r in ans_rows if r["abstained"]]

    def mean(key, rows):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    agg = {
        # Retrieval (always reported on answerable; refusal doesn't change retrieval).
        "recall@1": mean("recall@1", ans_rows),
        "recall@5": mean("recall@5", ans_rows),
        "recall@10": mean("recall@10", ans_rows),
        "mrr@10": mean("mrr@10", ans_rows),
        "ndcg@5": mean("ndcg@5", ans_rows),
        # Selective-QA.
        "coverage_all": len(answered_all) / max(1, len(eval_rows)),
        "false_refusal_answerable": len(false_refusal) / max(1, n_ans),
        "refusal_unanswerable": len(refused_unans) / max(1, n_una),
        "hallucination_unanswerable": len(halluc_unans) / max(1, n_una),
        "f1_answered_only": mean("f1", answered_ans),
        "citation_when_answered": mean("citation", answered_ans),
        # Cost/latency.
        "latency_mean_ms": 1000 * mean("latency", eval_rows),
        "latency_p95_ms": 1000 * float(np.percentile([r["latency"] for r in eval_rows], 95)),
    }
    print(f"  {encoder_name:18}  R@1={agg['recall@1']:.3f}  MRR={agg['mrr@10']:.3f}  "
          f"Refuse={agg['refusal_unanswerable']:.3f}  Cite={agg['citation_when_answered']:.3f}  "
          f"F1@ans={agg['f1_answered_only']:.3f}  Lat={agg['latency_mean_ms']:.0f}ms", flush=True)
    return {
        "encoder": encoder_name,
        "dataset": name,
        "calibration": {"balanced_accuracy": ba, "tau_score": tau_s, "tau_margin": tau_m},
        "aggregate": agg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["squad", "enterprise"])
    ap.add_argument("--encoders", nargs="+",
                    default=[name for name, _ in ENCODERS])
    args = ap.parse_args()

    out = {"results": []}
    for ds in args.datasets:
        if ds == "squad":
            corpus, qs = load_squad()
        elif ds == "enterprise":
            corpus, qs = load_enterprise()
        else:
            raise ValueError(ds)
        for short, hf_name in ENCODERS:
            if short not in args.encoders:
                continue
            res = run_dataset(ds, corpus, qs, hf_name)
            res["encoder_short"] = short
            out["results"].append(res)

    save_json(out, RESULTS / "encoder_ablation.json")
    print(f"\nwrote {RESULTS / 'encoder_ablation.json'}")


if __name__ == "__main__":
    main()

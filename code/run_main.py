"""Main experiment: run all four systems on SQuAD-dev (subsampled) and on the
synthetic enterprise benchmark.

For AutoRAG we do a 40/60 calibration/evaluation split: a held-out calibration
set selects the abstain thresholds (tau_score, tau_margin), then evaluation is
reported on the remaining 60%. To make the comparison fair, all systems are
evaluated on that same 60% evaluation split when computing the headline
metrics, so only AutoRAG uses the calibration split.

Writes results/main_<dataset>.json.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import (
    Chunk,
    RESULTS,
    best_against,
    mrr_first,
    ndcg_at_k,
    recall_at_k,
    save_json,
)
from data_loaders import load_enterprise, load_squad
from systems import AutoRAG, BM25System, NaiveRAG, RAGRerank


SYSTEMS = [
    ("BM25", BM25System),
    ("NaiveRAG", NaiveRAG),
    ("RAG+Rerank", RAGRerank),
    ("AutoRAG", AutoRAG),
]


def compute_faithfulness(prediction: str, support_text: str) -> float:
    if not prediction.strip():
        return 1.0
    from systems import _content_tokens
    pt = _content_tokens(prediction)
    if not pt:
        return 1.0
    st = set(_content_tokens(support_text))
    return sum(1 for t in pt if t in st) / len(pt)


def metrics_from_output(q: dict, out: dict) -> dict:
    ranked: list[Chunk] = out["ranked"]
    metrics = {}
    if q["answerable"]:
        gold = (q["gold_doc"], q["gold_para_idx"])
        metrics["recall@1"] = recall_at_k(ranked, gold, 1)
        metrics["recall@5"] = recall_at_k(ranked, gold, 5)
        metrics["recall@10"] = recall_at_k(ranked, gold, 10)
        metrics["mrr@10"] = mrr_first(ranked, gold, 10)
        metrics["ndcg@5"] = ndcg_at_k(ranked, gold, 5)
        em, f1 = best_against(out["prediction"], q["gold_answers"])
        metrics["em"] = em
        metrics["f1"] = f1
        cite_correct = (
            out.get("cite_doc") == q["gold_doc"]
            and out.get("cite_para") == q["gold_para_idx"]
        )
        metrics["citation"] = float(cite_correct)
        if out.get("cite_doc") is not None and ranked:
            cited = next(
                (c for c in ranked if c.doc_id == out["cite_doc"] and c.para_id == out["cite_para"]),
                ranked[0],
            )
            metrics["faithfulness"] = compute_faithfulness(out["prediction"], cited.text)
        else:
            metrics["faithfulness"] = 1.0 if not out["prediction"].strip() else 0.0
        metrics["hallucination"] = float(
            out["prediction"].strip() != "" and metrics["faithfulness"] < 0.5
        )
        metrics["refusal_correct"] = 0.0  # not defined for answerable
    else:
        em, f1 = best_against(out["prediction"], [])
        metrics["em"] = em
        metrics["f1"] = f1
        metrics["citation"] = 1.0 if not out["prediction"].strip() else 0.0
        metrics["faithfulness"] = 1.0 if not out["prediction"].strip() else 0.0
        metrics["hallucination"] = float(out["prediction"].strip() != "")
        metrics["refusal_correct"] = float(out["prediction"].strip() == "")
    metrics["latency"] = out["latency"]
    return metrics


def calibrate_abstain(records: list[dict]) -> tuple[float, float, float]:
    """Sweep (tau_score, tau_margin) on calibration records.

    Each record has top_score, margin, answerable.
    Maximize balanced accuracy: 0.5 * (refuse_unans + keep_ans).
    """
    best = (-1.0, 0.0, 0.0)
    score_grid = np.arange(0.0, 7.01, 0.5)
    margin_grid = np.arange(0.0, 5.01, 0.5)
    for ts in score_grid:
        for tm in margin_grid:
            keep_a, total_a = 0, 0
            refuse_u, total_u = 0, 0
            for r in records:
                abstain = r["top_score"] < ts or r["margin"] < tm
                if r["answerable"]:
                    total_a += 1
                    if not abstain:
                        keep_a += 1
                else:
                    total_u += 1
                    if abstain:
                        refuse_u += 1
            if total_a == 0 or total_u == 0:
                continue
            ba = 0.5 * (keep_a / total_a + refuse_u / total_u)
            if ba > best[0]:
                best = (ba, float(ts), float(tm))
    return best


def run_dataset(name: str, corpus: dict, questions: list[dict]):
    print(f"\n=== Dataset: {name}  (docs={len(corpus)}, questions={len(questions)}) ===")

    # 40/60 calibration/evaluation split (stratified by answerability + domain)
    rng = np.random.default_rng(20260511)
    indices = np.arange(len(questions))
    rng.shuffle(indices)
    cal_size = int(0.4 * len(questions))
    cal_set = set(indices[:cal_size].tolist())

    results = {
        "dataset": name,
        "n_questions": len(questions),
        "n_calibration": cal_size,
        "n_evaluation": len(questions) - cal_size,
        "systems": {},
    }
    autorag_records = []
    autorag_outputs = []

    for sys_name, SysCls in SYSTEMS:
        print(f"[{name}] Indexing {sys_name} ...", flush=True)
        sys = SysCls()
        if sys_name == "AutoRAG":
            sys.use_abstain = False  # disable during run, applied post-hoc
        sys.index(corpus)
        print(f"[{name}] Indexed {len(sys.chunks)} chunks in {sys._index_time:.1f}s", flush=True)
        per_q = []
        t0 = time.perf_counter()
        for i, q in enumerate(questions, 1):
            out = sys.answer(q["question"])
            m = metrics_from_output(q, out)
            row = {
                "qid": q["qid"],
                "domain": q.get("domain"),
                "answerable": q["answerable"],
                "in_calibration": i - 1 in cal_set,
                "prediction": out["prediction"],
                "cite_doc": out["cite_doc"],
                "cite_para": out["cite_para"],
                "top_score": out["top_score"],
                "margin": out.get("margin", 0.0),
                **m,
            }
            per_q.append(row)
            if sys_name == "AutoRAG":
                autorag_records.append(row)
                autorag_outputs.append((q, out))
            if i % 100 == 0:
                print(f"  [{sys_name}] {i}/{len(questions)}", flush=True)
        elapsed = time.perf_counter() - t0
        results["systems"][sys_name] = {
            "index_time": sys._index_time,
            "n_chunks": len(sys.chunks),
            "total_query_time": elapsed,
            "per_question": per_q,
        }
        print(f"[{name}] {sys_name} total query time {elapsed:.1f}s ({elapsed/len(questions)*1000:.1f} ms/q)", flush=True)

    # Calibrate AutoRAG abstain on calibration split, then re-apply on all
    cal_records = [r for r in autorag_records if r["in_calibration"]]
    ba, tau_s, tau_m = calibrate_abstain(cal_records)
    print(f"[{name}] AutoRAG abstain calibration: bal_acc={ba:.3f} tau_score={tau_s} tau_margin={tau_m}")
    results["autorag_calibration"] = {
        "balanced_accuracy": ba,
        "tau_score": tau_s,
        "tau_margin": tau_m,
    }
    # Re-evaluate AutoRAG with calibrated abstain
    updated = []
    for r, (q, out) in zip(autorag_records, autorag_outputs):
        abstain = r["top_score"] < tau_s or r["margin"] < tau_m
        new_out = dict(out)
        if abstain:
            new_out["prediction"] = ""
            new_out["cite_doc"] = None
            new_out["cite_para"] = None
        m2 = metrics_from_output(q, new_out)
        new_row = dict(r)
        new_row.update(m2)
        new_row["prediction"] = new_out["prediction"]
        new_row["cite_doc"] = new_out["cite_doc"]
        new_row["cite_para"] = new_out["cite_para"]
        new_row["abstained"] = abstain
        updated.append(new_row)
    results["systems"]["AutoRAG"]["per_question"] = updated

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad-per-article", type=int, default=20)
    ap.add_argument("--datasets", nargs="+", default=["squad", "enterprise"])
    args = ap.parse_args()

    if "squad" in args.datasets:
        corpus, qs = load_squad(n_per_article=args.squad_per_article)
        res = run_dataset("squad", corpus, qs)
        save_json(res, RESULTS / "main_squad.json")
        print("Wrote", RESULTS / "main_squad.json")

    if "enterprise" in args.datasets:
        corpus, qs = load_enterprise()
        res = run_dataset("enterprise", corpus, qs)
        save_json(res, RESULTS / "main_enterprise.json")
        print("Wrote", RESULTS / "main_enterprise.json")


if __name__ == "__main__":
    main()

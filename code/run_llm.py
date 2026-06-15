"""LLM-based answer-quality evaluation.

For each of the four systems we (re-)run retrieval with the same calibrated
configuration as `run_main.py` and replace the extractive answer head with a
real generative LLM (FLAN-T5-Base, 248M params) prompted to either answer
from the retrieved context or output the sentinel NOT_IN_CONTEXT for
unanswerable / out-of-corpus queries.

For each (system, question) we record:
  * the generated answer (or refusal sentinel),
  * exact-match and SQuAD-F1 against the gold answer(s),
  * citation accuracy (top-1 chunk vs. gold paragraph),
  * faithfulness (content-token overlap of the answer with the provided
    context, NOT just the cited chunk -- LLM may legitimately blend
    information from the top-K),
  * refusal accuracy on unanswerable items.

Writes results/llm_<dataset>.json. For computational reasons we run LLM
generation on the Enterprise benchmark in full and on a 200-question
stratified subsample of SQuAD-v2.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import RESULTS, best_against, save_json
from data_loaders import load_enterprise, load_squad
from run_main import calibrate_abstain, metrics_from_output
from systems import AutoRAG, BM25System, NaiveRAG, RAGRerank


MODEL_NAME = "google/flan-t5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEMS = [
    ("BM25", BM25System, {}),
    ("NaiveRAG", NaiveRAG, {}),
    ("RAG+Rerank", RAGRerank, {}),
    ("AutoRAG", AutoRAG, {"use_abstain": False}),  # abstain handled post-hoc on Enterprise/SQuAD
]


_tok = None
_model = None


def _load_llm():
    global _tok, _model
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
        _model.eval()


def build_prompt(question: str, contexts: list[str]) -> str:
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


def generate_batch(prompts: list[str], max_new_tokens: int = 32) -> list[str]:
    _load_llm()
    enc = _tok(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = _model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    return _tok.batch_decode(out, skip_special_tokens=True)


def faithfulness_against_contexts(prediction: str, contexts: list[str]) -> float:
    if not prediction.strip() or prediction.strip().upper() == "NOT_IN_CONTEXT":
        return 1.0
    from systems import _content_tokens
    pt = _content_tokens(prediction)
    if not pt:
        return 1.0
    bag = set()
    for c in contexts:
        bag.update(_content_tokens(c))
    return sum(1 for t in pt if t in bag) / len(pt)


def is_refusal(s: str) -> bool:
    return s.strip().upper().startswith("NOT_IN_CONTEXT") or not s.strip()


def evaluate_llm_pair(q: dict, prediction: str, top_chunks, contexts: list[str]) -> dict:
    refused = is_refusal(prediction)
    pred_clean = "" if refused else prediction.strip()
    metrics = {}
    if q["answerable"]:
        em, f1 = best_against(pred_clean, q["gold_answers"])
        gold = (q["gold_doc"], q["gold_para_idx"])
        cite_correct = bool(top_chunks) and top_chunks[0].doc_id == gold[0] and top_chunks[0].para_id == gold[1]
        metrics["em"] = em
        metrics["f1"] = f1
        metrics["citation"] = float(cite_correct) if not refused else 0.0
        metrics["faithfulness"] = faithfulness_against_contexts(pred_clean, contexts)
        metrics["hallucination"] = float(not refused and metrics["faithfulness"] < 0.5)
        metrics["refused_when_answerable"] = float(refused)
        metrics["refusal_correct"] = 0.0
    else:
        em, f1 = best_against("" if refused else pred_clean, [])
        metrics["em"] = em
        metrics["f1"] = f1
        metrics["citation"] = 1.0 if refused else 0.0
        metrics["faithfulness"] = 1.0 if refused else faithfulness_against_contexts(pred_clean, contexts)
        metrics["hallucination"] = float(not refused)
        metrics["refusal_correct"] = float(refused)
        metrics["refused_when_answerable"] = 0.0
    return metrics, pred_clean, refused


def run_dataset(name: str, corpus: dict, questions: list[dict], top_k: int = 3, batch_size: int = 8):
    print(f"\n=== LLM eval / {name}: {len(corpus)} docs, {len(questions)} questions ===", flush=True)
    rng = np.random.default_rng(20260512)
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
        "model": MODEL_NAME,
    }

    autorag_records = []
    autorag_payload = []

    for sys_name, SysCls, overrides in SYSTEMS:
        sys = SysCls(**overrides)
        print(f"[{name}/llm] indexing {sys_name} ...", flush=True)
        sys.index(corpus)
        # Step 1: retrieval for all questions
        rows = []
        outputs = []
        retrieval_only_time = 0.0
        t0 = time.perf_counter()
        for i, q in enumerate(questions):
            out = sys.answer(q["question"])
            outputs.append((q, out))
            retrieval_only_time += out["latency"]
            rows.append({
                "qid": q["qid"],
                "domain": q.get("domain"),
                "answerable": q["answerable"],
                "in_calibration": i in cal_set,
                "top_score": out["top_score"],
                "margin": out.get("margin", 0.0),
            })
        # Step 2: LLM generation in batches
        prompts = []
        ctx_lists = []
        cite_chunks = []
        for q, out in outputs:
            chunks = out["ranked"][:top_k]
            contexts = [c.text for c in chunks]
            prompts.append(build_prompt(q["question"], contexts))
            ctx_lists.append(contexts)
            cite_chunks.append(chunks)
        predictions: list[str] = []
        gen_t0 = time.perf_counter()
        for b in range(0, len(prompts), batch_size):
            batch_prompts = prompts[b : b + batch_size]
            outs = generate_batch(batch_prompts)
            predictions.extend(outs)
            if (b // batch_size) % 10 == 0:
                done = min(b + batch_size, len(prompts))
                print(f"  [{sys_name}] LLM {done}/{len(prompts)}", flush=True)
        gen_time = time.perf_counter() - gen_t0
        # Step 3: metrics
        per_q = []
        for (q, out), row, chunks, ctx_list, pred in zip(outputs, rows, cite_chunks, ctx_lists, predictions):
            m, pred_clean, refused = evaluate_llm_pair(q, pred, chunks, ctx_list)
            r = dict(row)
            r["llm_raw"] = pred
            r["llm_pred"] = pred_clean
            r["refused"] = refused
            r["cite_doc"] = chunks[0].doc_id if chunks else None
            r["cite_para"] = chunks[0].para_id if chunks else None
            r["latency"] = out["latency"] + (gen_time / max(1, len(prompts)))
            r.update(m)
            per_q.append(r)
            if sys_name == "AutoRAG":
                autorag_records.append(r)
                autorag_payload.append((q, out, ctx_list, chunks, pred))
        total = time.perf_counter() - t0
        results["systems"][sys_name] = {
            "index_time": sys._index_time,
            "retrieval_time_s": retrieval_only_time,
            "llm_generate_time_s": gen_time,
            "total_query_time": total,
            "n_chunks": len(sys.chunks),
            "per_question": per_q,
        }
        eval_rows = [p for p in per_q if not p["in_calibration"]]
        ans = [p for p in eval_rows if p["answerable"]]
        una = [p for p in eval_rows if not p["answerable"]]
        em = float(np.mean([p["em"] for p in ans])) if ans else 0.0
        f1 = float(np.mean([p["f1"] for p in ans])) if ans else 0.0
        rc = float(np.mean([p["refusal_correct"] for p in una])) if una else 0.0
        ci = float(np.mean([p["citation"] for p in ans])) if ans else 0.0
        ft = float(np.mean([p["faithfulness"] for p in ans])) if ans else 0.0
        print(f"[{name}/llm] {sys_name}  EM={em:.3f}  F1={f1:.3f}  Cite={ci:.3f}  Faith={ft:.3f}  Refusal(unans)={rc:.3f}", flush=True)

    # Calibrated abstain on top of AutoRAG LLM output
    if autorag_records:
        cal = [r for r in autorag_records if r["in_calibration"]]
        ba, tau_s, tau_m = calibrate_abstain(cal)
        print(f"[{name}/llm] AutoRAG abstain calibration: bal_acc={ba:.3f} tau_s={tau_s} tau_m={tau_m}")
        results["autorag_calibration"] = {"balanced_accuracy": ba, "tau_score": tau_s, "tau_margin": tau_m}
        updated = []
        for r, (q, out, ctx_list, chunks, pred) in zip(autorag_records, autorag_payload):
            abstain_by_retr = (r["top_score"] < tau_s) or (r["margin"] < tau_m)
            nr = dict(r)
            if abstain_by_retr:
                nr["llm_pred"] = ""
                nr["refused"] = True
                nr["cite_doc"] = None
                nr["cite_para"] = None
                m, _, _ = evaluate_llm_pair(q, "NOT_IN_CONTEXT", chunks, ctx_list)
                nr.update(m)
                nr["abstained_by_retrieval"] = True
            else:
                nr["abstained_by_retrieval"] = False
            updated.append(nr)
        results["systems"]["AutoRAG"]["per_question"] = updated
    return results


def stratified_subsample(questions, n: int, seed: int = 20260512):
    rng = np.random.default_rng(seed)
    ans = [q for q in questions if q["answerable"]]
    una = [q for q in questions if not q["answerable"]]
    rng.shuffle(ans); rng.shuffle(una)
    n_ans = min(len(ans), n // 2)
    n_una = min(len(una), n - n_ans)
    return ans[:n_ans] + una[:n_una]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["enterprise", "squad"])
    ap.add_argument("--squad-n", type=int, default=200, help="subsample for SQuAD-v2 LLM eval")
    ap.add_argument("--squad-per-article", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    if "enterprise" in args.datasets:
        corpus, qs = load_enterprise()
        res = run_dataset("enterprise", corpus, qs, batch_size=args.batch_size)
        save_json(res, RESULTS / "llm_enterprise.json")
        print("Wrote", RESULTS / "llm_enterprise.json")

    if "squad" in args.datasets:
        corpus, all_qs = load_squad(n_per_article=args.squad_per_article)
        qs = stratified_subsample(all_qs, args.squad_n)
        print(f"SQuAD LLM subsample: {len(qs)} questions")
        res = run_dataset("squad", corpus, qs, batch_size=args.batch_size)
        save_json(res, RESULTS / "llm_squad.json")
        print("Wrote", RESULTS / "llm_squad.json")


if __name__ == "__main__":
    main()

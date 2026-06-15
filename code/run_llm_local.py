"""Modern open-weight LLM-head evaluation (CPU-only) per Section 9.

Replaces the FLAN-T5-Base head with Qwen2.5-1.5B-Instruct
(open-access, 1.5B params, ~3GB on disk). All other components are
identical to run_llm.py so the result is comparable to Table 13.

Smaller evaluation to keep CPU runtime tractable:
  - SQuAD-v2: 100-question stratified subsample (50 ans + 50 unans)
  - Enterprise: full 305 questions

Three systems (the most informative comparison for selective QA):
  RAG+Rerank, RAG+Rerank+Abstain (post-hoc), AutoRAG.

Writes results/llm_local_<dataset>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import RESULTS, save_json
from data_loaders import load_enterprise, load_squad
from run_main import calibrate_abstain
from run_llm import (
    SYSTEMS as FULL_SYSTEMS,
    evaluate_llm_pair,
    is_refusal,
    stratified_subsample,
)
from systems import AutoRAG, RAGRerank


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # overridden by --model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
_tok = None
_model = None


def model_tag(name: str) -> str:
    """Filesystem-safe tag derived from a HF model id."""
    return name.split("/")[-1].lower().replace(".", "").replace("_", "-")


SYSTEM_PROMPT = (
    "You are a careful question-answering assistant. "
    "Answer the question using only the information in the provided context. "
    "If the answer cannot be found in the context, respond with exactly NOT_IN_CONTEXT. "
    "Be concise: respond with a short factual span, not a full sentence."
)


def _load_llm():
    global _tok, _model
    if _tok is None:
        print(f"Loading {MODEL_NAME} on {DEVICE} ...", flush=True)
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
        _model.eval()


def build_chat(question: str, contexts: list[str]):
    joined = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    user = f"Context:\n{joined}\n\nQuestion: {question}\nAnswer:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate(question: str, contexts: list[str], max_new_tokens: int = 32) -> str:
    _load_llm()
    messages = build_chat(question, contexts)
    prompt = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=_tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return _tok.decode(new_tokens, skip_special_tokens=True).strip()


SYSTEMS_LOCAL = [
    ("RAG+Rerank", RAGRerank, {}),
    # AutoRAG (run with abstention OFF; abstention applied post-hoc below)
    ("AutoRAG-raw", AutoRAG, {"use_abstain": False}),
]


def run_dataset(name: str, corpus, questions, top_k: int = 3):
    print(f"\n=== Local LLM ({MODEL_NAME}) / {name}: "
          f"{len(corpus)} docs, {len(questions)} q ===", flush=True)
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

    for sys_name, SysCls, overrides in SYSTEMS_LOCAL:
        sys = SysCls(**overrides)
        print(f"  indexing {sys_name} ...", flush=True)
        sys.index(corpus)
        rows = []
        outputs = []
        t_ret = 0.0
        t0 = time.perf_counter()
        for i, q in enumerate(questions):
            out = sys.answer(q["question"])
            outputs.append((q, out))
            t_ret += out["latency"]
            rows.append({
                "qid": q["qid"], "domain": q.get("domain"),
                "answerable": q["answerable"], "in_calibration": i in cal_set,
                "top_score": out["top_score"], "margin": out.get("margin", 0.0),
            })
        # Generation loop (one query at a time on CPU)
        per_q = []
        t_gen = time.perf_counter()
        for (q, out), row in zip(outputs, rows):
            chunks = out["ranked"][:top_k]
            contexts = [c.text for c in chunks]
            try:
                pred = generate(q["question"], contexts)
            except Exception as e:
                print(f"    gen failure on {q['qid']}: {e}", flush=True)
                pred = "NOT_IN_CONTEXT"
            m, pred_clean, refused = evaluate_llm_pair(q, pred, chunks, contexts)
            r = dict(row)
            r["llm_raw"] = pred
            r["llm_pred"] = pred_clean
            r["refused"] = refused
            r["cite_doc"] = chunks[0].doc_id if chunks else None
            r["cite_para"] = chunks[0].para_id if chunks else None
            r["contexts"] = contexts
            r["latency"] = out["latency"]
            r.update(m)
            per_q.append(r)
            if sys_name == "AutoRAG-raw":
                autorag_records.append(r)
                autorag_payload.append((q, out, contexts, chunks, pred))
            if (len(per_q)) % 25 == 0:
                print(f"    [{sys_name}] {len(per_q)}/{len(rows)}", flush=True)
        t_gen_total = time.perf_counter() - t_gen
        results["systems"][sys_name] = {
            "retrieval_time_s": t_ret,
            "llm_generate_time_s": t_gen_total,
            "n_chunks": len(sys.chunks),
            "per_question": per_q,
        }

        # Aggregate per-system
        ev = [r for r in per_q if not r["in_calibration"]]
        ans = [r for r in ev if r["answerable"]]
        una = [r for r in ev if not r["answerable"]]
        f1 = float(np.mean([p["f1"] for p in ans])) if ans else 0.0
        ref = float(np.mean([p["refusal_correct"] for p in una])) if una else 0.0
        halluc = 1.0 - ref
        cite = float(np.mean([p["citation"] for p in ans if not p.get("refused")])) if ans else 0.0
        print(f"  {sys_name:14}  F1={f1:.3f}  Refuse={ref:.3f}  Halluc={halluc:.3f}  Cite={cite:.3f}",
              flush=True)

    # Post-hoc abstention layered on AutoRAG raw outputs to produce AutoRAG and
    # RAG+Rerank+Abstain rows.
    if autorag_records:
        cal_rows = [r for r in autorag_records if r["in_calibration"]]
        ba, tau_s, tau_m = calibrate_abstain(cal_rows)
        print(f"  AutoRAG abstain calibration: bal_acc={ba:.3f} tau_s={tau_s} tau_m={tau_m}", flush=True)
        results["autorag_calibration"] = {"balanced_accuracy": ba, "tau_score": tau_s, "tau_margin": tau_m}
        updated = []
        for r, (q, out, ctx, chunks, pred) in zip(autorag_records, autorag_payload):
            abstain = (r["top_score"] < tau_s) or (r["margin"] < tau_m)
            nr = dict(r)
            if abstain:
                nr["llm_pred"] = ""
                nr["refused"] = True
                nr["cite_doc"] = None
                nr["cite_para"] = None
                m, _, _ = evaluate_llm_pair(q, "NOT_IN_CONTEXT", chunks, ctx)
                nr.update(m)
                nr["abstained_by_retrieval"] = True
            else:
                nr["abstained_by_retrieval"] = False
            updated.append(nr)
        results["systems"]["AutoRAG"] = {
            "per_question": updated,
            "note": "AutoRAG = AutoRAG-raw retrieval + calibrated abstention on AutoRAG logits.",
        }
        # Also produce RAG+Rerank+Abstain by applying the same rule to the
        # RAG+Rerank rows.
        rr_records = results["systems"]["RAG+Rerank"]["per_question"]
        cal_rr = [r for r in rr_records if r["in_calibration"]]
        ba2, tau_s2, tau_m2 = calibrate_abstain(cal_rr)
        print(f"  RAG+Rerank+Abstain calibration: bal_acc={ba2:.3f} tau_s={tau_s2} tau_m={tau_m2}", flush=True)
        rra = []
        for r in rr_records:
            abstain = (r["top_score"] < tau_s2) or (r["margin"] < tau_m2)
            nr = dict(r)
            if abstain:
                nr["llm_pred"] = ""
                nr["refused"] = True
                nr["cite_doc"] = None
                nr["cite_para"] = None
                if r["answerable"]:
                    nr["em"] = 0.0
                    nr["f1"] = 0.0
                    nr["citation"] = 0.0
                    nr["refusal_correct"] = 0.0
                else:
                    nr["em"] = 1.0
                    nr["f1"] = 1.0
                    nr["citation"] = 1.0
                    nr["refusal_correct"] = 1.0
                nr["abstained_by_retrieval"] = True
            else:
                nr["abstained_by_retrieval"] = False
            rra.append(nr)
        results["systems"]["RAG+Rerank+Abstain"] = {
            "per_question": rra,
            "calibration": {"balanced_accuracy": ba2, "tau_score": tau_s2, "tau_margin": tau_m2},
        }
    return results


def summarise(res, sys_name):
    per_q = res["systems"][sys_name]["per_question"]
    ev = [r for r in per_q if not r["in_calibration"]]
    ans = [r for r in ev if r["answerable"]]
    una = [r for r in ev if not r["answerable"]]
    f1 = float(np.mean([p["f1"] for p in ans])) if ans else 0.0
    ref = float(np.mean([p["refusal_correct"] for p in una])) if una else 0.0
    answered_ans = [p for p in ans if not p.get("refused")]
    cite = float(np.mean([p["citation"] for p in answered_ans])) if answered_ans else 0.0
    return {"f1": f1, "refuse": ref, "halluc": 1 - ref, "cite": cite}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["squad", "enterprise"])
    ap.add_argument("--squad-n", type=int, default=100,
                    help="SQuAD eval subsample size (default 100)")
    ap.add_argument("--squad-per-article", type=int, default=20)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="HuggingFace model id. Recommended modern small LLMs: "
                         "meta-llama/Llama-3.2-3B-Instruct, "
                         "microsoft/Phi-3.5-mini-instruct, "
                         "google/gemma-2-2b-it, "
                         "HuggingFaceTB/SmolLM2-1.7B-Instruct.")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None,
                    help="Override torch dtype. Default: float16 on CUDA, float32 on CPU.")
    args = ap.parse_args()

    global MODEL_NAME, DTYPE
    MODEL_NAME = args.model
    if args.dtype is not None:
        DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[args.dtype]

    tag = model_tag(MODEL_NAME)

    if "enterprise" in args.datasets:
        corpus, qs = load_enterprise()
        res = run_dataset("enterprise", corpus, qs)
        out = RESULTS / f"llm_local_enterprise_{tag}.json"
        save_json(res, out)
        print("wrote", out)

    if "squad" in args.datasets:
        corpus, all_qs = load_squad(n_per_article=args.squad_per_article)
        qs = stratified_subsample(all_qs, args.squad_n)
        print(f"SQuAD local-LLM subsample: {len(qs)} q")
        res = run_dataset("squad", corpus, qs)
        out = RESULTS / f"llm_local_squad_{tag}.json"
        save_json(res, out)
        print("wrote", out)


if __name__ == "__main__":
    main()

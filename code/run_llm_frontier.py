"""Frontier-LLM answer-quality evaluation (Claude Sonnet 4.6 / GPT-4o).

Companion to run_llm.py. Same retrieval and metric pipeline; replaces the
FLAN-T5-Base generative head with a provider-grade LLM. Used to address the
reviewer concern that FLAN-T5-Base is not deployment-representative.

For each (system, question) the script records the same payload as run_llm.py
so the downstream analysis code (tables, statistical tests, cost projection)
needs no changes:
  * the generated answer (or refusal sentinel NOT_IN_CONTEXT),
  * exact-match and SQuAD-F1 against the gold answer(s),
  * citation accuracy (top-1 chunk vs. gold paragraph),
  * faithfulness (content-token overlap of the answer with the provided
    context, not just the cited chunk -- the LLM may legitimately blend
    information from top-k),
  * refusal accuracy on unanswerable items.

Writes results/llm_frontier_<dataset>.json. Default sample sizes match
run_llm.py: full Enterprise (305 q), stratified 200-q SQuAD-v2 subsample.

Anthropic SDK is used with prompt caching on the system message so the
system tokens are billed only once per ~5-min cache window. Resume support:
if the output JSON exists and contains per-question records, qids that are
already present are skipped.

Install:
  pip install anthropic            # for --provider anthropic (default)
  pip install openai               # for --provider openai

Env vars:
  ANTHROPIC_API_KEY                # required when --provider anthropic
  OPENAI_API_KEY                   # required when --provider openai
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from common import RESULTS, save_json
from data_loaders import load_enterprise, load_squad
from run_llm import (
    SYSTEMS,
    evaluate_llm_pair,
    stratified_subsample,
)
from run_main import calibrate_abstain


SYSTEM_PROMPT = (
    "You are a careful question-answering assistant. "
    "Answer the question using only the information in the provided context. "
    "If the answer cannot be found in the context, respond with exactly NOT_IN_CONTEXT. "
    "Be concise: respond with a short factual span, not a full sentence."
)


def build_user(question: str, contexts: list[str]) -> str:
    joined = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return f"Context:\n{joined}\n\nQuestion: {question}\nAnswer:"


_anthropic_client = None
_openai_client = None


def generate_anthropic(question: str, contexts: list[str], model: str) -> str:
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    resp = _anthropic_client.messages.create(
        model=model,
        max_tokens=64,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": build_user(question, contexts)}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "text") == "text").strip()


def generate_openai(question: str, contexts: list[str], model: str) -> str:
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    resp = _openai_client.chat.completions.create(
        model=model,
        max_completion_tokens=64,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user(question, contexts)},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


_last_call_ts: float = 0.0
_min_interval: float = 0.0


def generate(provider: str, model: str, question: str, contexts: list[str]) -> str:
    global _last_call_ts
    if _min_interval > 0:
        wait = _min_interval - (time.perf_counter() - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
    if provider == "anthropic":
        out = generate_anthropic(question, contexts, model)
    elif provider == "openai":
        out = generate_openai(question, contexts, model)
    else:
        raise ValueError(provider)
    _last_call_ts = time.perf_counter()
    return out


def load_existing(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_dataset(name: str, corpus, questions, provider: str, model: str,
                top_k: int = 3, resume_from: dict | None = None, retries: int = 3) -> dict:
    print(f"\n=== Frontier LLM ({provider}/{model}) / {name}: "
          f"{len(corpus)} docs, {len(questions)} questions ===", flush=True)
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
        "model": f"{provider}:{model}",
    }

    autorag_records = []
    autorag_payload = []

    for sys_name, SysCls, overrides in SYSTEMS:
        sys = SysCls(**overrides)
        print(f"[{name}/{model}] indexing {sys_name} ...", flush=True)
        sys.index(corpus)
        done_qids: set[str] = set()
        if resume_from and sys_name in resume_from.get("systems", {}):
            done_qids = {r["qid"] for r in resume_from["systems"][sys_name].get("per_question", [])}
            if done_qids:
                print(f"  resuming: {len(done_qids)} qids already done", flush=True)
        rows = []
        outputs = []
        retrieval_only_time = 0.0
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

        gen_t0 = time.perf_counter()
        predictions: list[str] = []
        ctx_lists: list[list[str]] = []
        cite_chunks_all = []
        cached_records: dict[str, dict] = {}
        if resume_from and sys_name in resume_from.get("systems", {}):
            for r in resume_from["systems"][sys_name].get("per_question", []):
                cached_records[r["qid"]] = r

        for (q, out), row in zip(outputs, rows):
            chunks = out["ranked"][:top_k]
            contexts = [c.text for c in chunks]
            cite_chunks_all.append(chunks)
            ctx_lists.append(contexts)
            if q["qid"] in done_qids:
                predictions.append(cached_records[q["qid"]].get("llm_raw", ""))
                continue
            last_err: Exception | None = None
            for attempt in range(retries):
                try:
                    pred = generate(provider, model, q["question"], contexts)
                    predictions.append(pred)
                    break
                except Exception as e:
                    last_err = e
                    wait = 2 ** attempt
                    print(f"  retry {attempt+1}/{retries} for {q['qid']} after {wait}s: {e}", flush=True)
                    time.sleep(wait)
            else:
                print(f"  GIVING UP on {q['qid']}: {last_err}", flush=True)
                predictions.append("NOT_IN_CONTEXT")
            if (len(predictions) - len(done_qids)) % 20 == 0:
                print(f"  [{sys_name}] LLM {len(predictions)}/{len(rows)}", flush=True)

        gen_time = time.perf_counter() - gen_t0

        per_q = []
        for (q, out), row, chunks, ctx_list, pred in zip(outputs, rows, cite_chunks_all, ctx_lists, predictions):
            if q["qid"] in cached_records:
                per_q.append(cached_records[q["qid"]])
                if sys_name == "AutoRAG":
                    autorag_records.append(cached_records[q["qid"]])
                    autorag_payload.append((q, out, ctx_list, chunks, cached_records[q["qid"]].get("llm_raw", "")))
                continue
            m, pred_clean, refused = evaluate_llm_pair(q, pred, chunks, ctx_list)
            r = dict(row)
            r["llm_raw"] = pred
            r["llm_pred"] = pred_clean
            r["refused"] = refused
            r["cite_doc"] = chunks[0].doc_id if chunks else None
            r["cite_para"] = chunks[0].para_id if chunks else None
            r["latency"] = out["latency"] + (gen_time / max(1, len(rows)))
            r["contexts"] = ctx_list
            r.update(m)
            per_q.append(r)
            if sys_name == "AutoRAG":
                autorag_records.append(r)
                autorag_payload.append((q, out, ctx_list, chunks, pred))

        results["systems"][sys_name] = {
            "index_time": sys._index_time,
            "retrieval_time_s": retrieval_only_time,
            "llm_generate_time_s": gen_time,
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
        print(f"[{name}/{model}] {sys_name}  EM={em:.3f}  F1={f1:.3f}  "
              f"Cite={ci:.3f}  Faith={ft:.3f}  Refusal(unans)={rc:.3f}",
              flush=True)

    if autorag_records:
        cal = [r for r in autorag_records if r["in_calibration"]]
        ba, tau_s, tau_m = calibrate_abstain(cal)
        print(f"[{name}/{model}] AutoRAG abstain calibration: "
              f"bal_acc={ba:.3f} tau_s={tau_s} tau_m={tau_m}")
        results["autorag_calibration"] = {
            "balanced_accuracy": ba, "tau_score": tau_s, "tau_margin": tau_m,
        }
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["enterprise", "squad"])
    ap.add_argument("--squad-n", type=int, default=200)
    ap.add_argument("--squad-per-article", type=int, default=20)
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default=None,
                    help="Model id. Default: claude-sonnet-4-6 / gpt-4o.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore any existing output JSON; re-run from scratch.")
    ap.add_argument("--min-interval", type=float, default=0.15,
                    help="Minimum seconds between LLM calls (rate throttle). "
                         "0.15s keeps OpenAI at ~400 RPM (safely below the 500 RPM Tier-1 limit).")
    args = ap.parse_args()

    global _min_interval
    _min_interval = float(args.min_interval)

    if args.model is None:
        args.model = "claude-sonnet-4-6" if args.provider == "anthropic" else "gpt-4o"

    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    if "enterprise" in args.datasets:
        out_path = RESULTS / f"llm_frontier_enterprise_{args.provider}.json"
        prev = None if args.no_resume else load_existing(out_path)
        corpus, qs = load_enterprise()
        res = run_dataset("enterprise", corpus, qs,
                           provider=args.provider, model=args.model,
                           resume_from=prev)
        save_json(res, out_path)
        print("Wrote", out_path)

    if "squad" in args.datasets:
        out_path = RESULTS / f"llm_frontier_squad_{args.provider}.json"
        prev = None if args.no_resume else load_existing(out_path)
        corpus, all_qs = load_squad(n_per_article=args.squad_per_article)
        qs = stratified_subsample(all_qs, args.squad_n)
        print(f"SQuAD frontier-LLM subsample: {len(qs)} questions")
        res = run_dataset("squad", corpus, qs,
                           provider=args.provider, model=args.model,
                           resume_from=prev)
        save_json(res, out_path)
        print("Wrote", out_path)


if __name__ == "__main__":
    main()

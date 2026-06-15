"""HyDE (Hypothetical Document Embeddings) baseline.

For each query, FLAN-T5-Base generates a hypothetical answer/document.
The hypothetical document is embedded and used as the dense-retrieval
query in place of the original question. The reranker still uses the
original question (Gao et al., 2023).

Pipeline:  question -> FLAN-T5(write passage answering Q) -> hypothesis
           -> dense embedding -> hybrid retrieval -> cross-encoder
           rerank on (original_question, chunk) -> answer head.

Adds one row to the headline retrieval comparison alongside
RAG+Rerank and AutoRAG, addressing the reviewer ask "why didn't you
compare to HyDE/query rewriting?"

Writes results/main_hyde_<dataset>.json with the same per-question
schema as run_main.py.

CPU runtime: ~0.5-1 s per query for FLAN-T5 generation + the usual
retrieval, so ~5-15 min per dataset eval split.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import RESULTS, Chunk, save_json, recall_at_k, mrr_first, ndcg_at_k, best_against
from data_loaders import load_enterprise, load_squad
from systems import (
    DEFAULT_ENCODER, RERANKER, _tokenize, extract_answer, get_crossencoder, get_encoder,
)
from run_main import calibrate_abstain


HYDE_MODEL_NAME = "google/flan-t5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_hyde_tok = None
_hyde_model = None


def _load_hyde():
    global _hyde_tok, _hyde_model
    if _hyde_tok is None:
        _hyde_tok = AutoTokenizer.from_pretrained(HYDE_MODEL_NAME)
        _hyde_model = AutoModelForSeq2SeqLM.from_pretrained(HYDE_MODEL_NAME).to(DEVICE)
        _hyde_model.eval()


def hyde_prompt(question: str) -> str:
    return (
        "Please write a short passage that answers the question.\n\n"
        f"Question: {question}\n"
        "Passage:"
    )


def generate_hyde(question: str, max_new_tokens: int = 48) -> str:
    _load_hyde()
    enc = _hyde_tok([hyde_prompt(question)], return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        out = _hyde_model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    text = _hyde_tok.decode(out[0], skip_special_tokens=True).strip()
    # Concatenate the original question to keep its vocabulary in the dense
    # embedding (standard HyDE practice).
    return f"{question} {text}".strip()


class HyDEIndex:
    """Indexing + retrieval mirroring RAGRerank, but with the dense query
    formed from FLAN-T5's hypothetical document."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self._index_time = 0.0

    def index(self, corpus: dict):
        from common import split_to_chunks
        t0 = time.perf_counter()
        chunks: list[Chunk] = []
        cid = 0
        for doc_id, paras in corpus.items():
            for ch in split_to_chunks(doc_id, paras, "paragraph"):
                ch.chunk_id = cid
                cid += 1
                chunks.append(ch)
        self.chunks = chunks
        enc = get_encoder(DEFAULT_ENCODER)
        self.embeddings = enc.encode(
            [c.text for c in chunks],
            batch_size=64, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        self._index_time = time.perf_counter() - t0

    def answer(self, question: str, k_retrieve: int = 20, k_top: int = 5) -> dict:
        t0 = time.perf_counter()
        hyde_query = generate_hyde(question)
        enc = get_encoder(DEFAULT_ENCODER)
        qv = enc.encode([hyde_query], convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.embeddings @ qv
        cand = np.argsort(-sims)[:k_retrieve]
        ranked_chunks = [self.chunks[i] for i in cand]
        ce = get_crossencoder(RERANKER)
        # NB: rerank uses the ORIGINAL question, not the hypothetical doc.
        pairs = [(question, c.text) for c in ranked_chunks]
        scores = ce.predict(pairs, show_progress_bar=False).tolist()
        order = np.argsort(-np.asarray(scores))
        ranked_chunks = [ranked_chunks[i] for i in order]
        rerank_scores = [float(scores[i]) for i in order]
        top = ranked_chunks[0] if ranked_chunks else None
        top_score = rerank_scores[0] if rerank_scores else 0.0
        margin = 0.0
        if len(rerank_scores) >= 2:
            tail = rerank_scores[1:min(5, len(rerank_scores))]
            margin = float(rerank_scores[0] - sum(tail) / len(tail))
        prediction = ""
        cite_doc = cite_para = None
        if top is not None:
            prediction = extract_answer(question, top.text)
            cite_doc, cite_para = top.doc_id, top.para_id
        latency = time.perf_counter() - t0
        return {
            "ranked": ranked_chunks[: max(k_retrieve, k_top)],
            "prediction": prediction,
            "cite_doc": cite_doc,
            "cite_para": cite_para,
            "latency": latency,
            "top_score": float(top_score),
            "margin": margin,
            "hyde_query": hyde_query,
        }


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
        "hyde_query": out.get("hyde_query"),
    }


def run_dataset(name: str, corpus, questions):
    print(f"\n=== HyDE / {name}: {len(corpus)} docs, {len(questions)} questions ===", flush=True)
    rng = np.random.default_rng(20260512)
    indices = np.arange(len(questions))
    rng.shuffle(indices)
    cal_size = int(0.4 * len(questions))
    cal_set = set(indices[:cal_size].tolist())

    sys = HyDEIndex()
    print(f"  indexing ({len(corpus)} docs) ...", flush=True)
    sys.index(corpus)
    print(f"  indexed {len(sys.chunks)} chunks in {sys._index_time:.1f}s", flush=True)

    per_q = []
    t0 = time.perf_counter()
    for i, q in enumerate(questions):
        out = sys.answer(q["question"])
        rec = evaluate_question(q, out)
        rec["in_calibration"] = i in cal_set
        per_q.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  [{name}] {i + 1}/{len(questions)}", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.0f}s", flush=True)

    # Aggregate on eval split only (mirror run_main.py)
    eval_rows = [r for r in per_q if not r["in_calibration"]]
    ans = [r for r in eval_rows if r["answerable"]]
    una = [r for r in eval_rows if not r["answerable"]]

    def mean(key, rows):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    agg = {
        "recall@1": mean("recall@1", ans),
        "recall@5": mean("recall@5", ans),
        "recall@10": mean("recall@10", ans),
        "mrr@10": mean("mrr@10", ans),
        "ndcg@5": mean("ndcg@5", ans),
        "em": mean("em", ans),
        "f1": mean("f1", ans),
        "citation": mean("citation", ans),
        "latency_mean_ms": 1000 * mean("latency", eval_rows),
    }
    print(f"  HyDE  R@1={agg['recall@1']:.3f}  R@5={agg['recall@5']:.3f}  "
          f"MRR={agg['mrr@10']:.3f}  F1={agg['f1']:.3f}  "
          f"Cite={agg['citation']:.3f}  Lat={agg['latency_mean_ms']:.0f}ms",
          flush=True)
    return {
        "dataset": name,
        "n_questions": len(questions),
        "n_calibration": cal_size,
        "n_evaluation": len(questions) - cal_size,
        "model": HYDE_MODEL_NAME,
        "system": "HyDE",
        "aggregate": agg,
        "per_question": per_q,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["squad", "enterprise"])
    ap.add_argument("--squad-per-article", type=int, default=20)
    args = ap.parse_args()
    if "squad" in args.datasets:
        corpus, qs = load_squad(n_per_article=args.squad_per_article)
        res = run_dataset("squad", corpus, qs)
        save_json(res, RESULTS / "main_hyde_squad.json")
        print("wrote", RESULTS / "main_hyde_squad.json")
    if "enterprise" in args.datasets:
        corpus, qs = load_enterprise()
        res = run_dataset("enterprise", corpus, qs)
        save_json(res, RESULTS / "main_hyde_enterprise.json")
        print("wrote", RESULTS / "main_hyde_enterprise.json")


if __name__ == "__main__":
    main()

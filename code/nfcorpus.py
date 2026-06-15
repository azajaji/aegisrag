"""NFCorpus (BEIR) loader and retrieval-only evaluation.

NFCorpus is a medical-information-retrieval benchmark with 3,633 PubMed
abstracts and 323 test queries. Relevance is graded (0/1/2); each query has
multiple relevant documents.

We evaluate all four systems on the NFCorpus test split with the standard
BEIR retrieval metrics. There is no answer-quality dimension here because
NFCorpus does not provide answer strings; the value of this benchmark is
that it tests retrieval on a completely different domain (medical
literature) with adversarially-difficult queries that often include
colloquial phrasing.

Writes results/main_nfcorpus.json.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from common import RESULTS, save_json
from systems import AutoRAG, BM25System, NaiveRAG, RAGRerank


NFC = Path(__file__).resolve().parents[1] / "data" / "nfcorpus"


def load_nfcorpus(max_queries: int | None = None):
    corpus = {}
    with open(NFC / "corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            title = d.get("title", "").strip()
            text = d["text"].strip()
            full = f"{title}. {text}" if title else text
            # Each doc is a single chunk (one paragraph). doc_id = _id.
            corpus[d["_id"]] = [full]
    with open(NFC / "queries.jsonl", "r", encoding="utf-8") as f:
        qmap = {json.loads(l)["_id"]: json.loads(l)["text"] for l in f}
    qrels: dict[str, dict[str, int]] = {}
    with open(NFC / "qrels" / "test.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, did, grade = line.strip().split("\t")
            if did not in corpus:
                continue
            qrels.setdefault(qid, {})[did] = int(grade)
    queries = []
    for qid, rel in qrels.items():
        rel_pos = {d: g for d, g in rel.items() if g > 0}
        if not rel_pos:
            continue
        queries.append({"qid": qid, "question": qmap[qid], "rel": rel_pos})
    queries.sort(key=lambda q: q["qid"])
    if max_queries:
        queries = queries[:max_queries]
    return corpus, queries


def _hit_metrics(ranked_doc_ids: list[str], rel: dict[str, int], ks=(5, 10, 100)):
    """Compute Recall@k, MRR@10, nDCG@10 with graded relevance."""
    out = {}
    rel_pos = set(rel.keys())
    total_rel = len(rel_pos)
    # Recall@k
    for k in ks:
        topk = ranked_doc_ids[:k]
        hits = sum(1 for d in topk if d in rel_pos)
        out[f"recall@{k}"] = hits / max(1, total_rel)
    # MRR@10 (first relevant)
    mrr = 0.0
    for i, d in enumerate(ranked_doc_ids[:10], 1):
        if d in rel_pos:
            mrr = 1.0 / i
            break
    out["mrr@10"] = mrr
    # nDCG@10 with graded relevance
    grades = [rel.get(d, 0) for d in ranked_doc_ids[:10]]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))
    ideal_grades = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades))
    out["ndcg@10"] = dcg / idcg if idcg > 0 else 0.0
    return out


SYSTEMS = [
    ("BM25", BM25System, {}),
    ("NaiveRAG", NaiveRAG, {}),
    ("RAG+Rerank", RAGRerank, {}),
    # On NFCorpus there are no unanswerable queries, so abstain is off.
    # We keep adaptive chunking + hybrid retrieval + reranker.
    ("AutoRAG", AutoRAG, {"use_abstain": False}),
]


def main(max_queries: int | None = None):
    corpus, queries = load_nfcorpus(max_queries=max_queries)
    print(f"NFCorpus: {len(corpus)} docs, {len(queries)} test queries")

    results = {
        "dataset": "nfcorpus",
        "n_questions": len(queries),
        "n_corpus": len(corpus),
        "systems": {},
    }

    for name, SysCls, overrides in SYSTEMS:
        sys = SysCls(**overrides)
        print(f"[nfcorpus] indexing {name} ...", flush=True)
        sys.index(corpus)
        print(f"[nfcorpus] indexed {len(sys.chunks)} chunks in {sys._index_time:.1f}s", flush=True)
        per_q = []
        t0 = time.perf_counter()
        for i, q in enumerate(queries, 1):
            out = sys.answer(q["question"], k_retrieve=100, k_top=10)
            # Map ranked chunks back to doc ids (chunk.doc_id is the NFCorpus _id)
            ranked_doc_ids: list[str] = []
            seen = set()
            for c in out["ranked"]:
                if c.doc_id not in seen:
                    ranked_doc_ids.append(c.doc_id)
                    seen.add(c.doc_id)
                if len(ranked_doc_ids) >= 100:
                    break
            metrics = _hit_metrics(ranked_doc_ids, q["rel"])
            per_q.append({
                "qid": q["qid"],
                "n_relevant": len(q["rel"]),
                "latency": out["latency"],
                **metrics,
            })
            if i % 50 == 0:
                print(f"  [{name}] {i}/{len(queries)}", flush=True)
        elapsed = time.perf_counter() - t0
        # Means
        agg = {}
        for m in ["recall@5", "recall@10", "recall@100", "mrr@10", "ndcg@10"]:
            agg[m] = float(np.mean([p[m] for p in per_q]))
        agg["latency_mean_ms"] = 1000 * float(np.mean([p["latency"] for p in per_q]))
        agg["latency_p95_ms"] = 1000 * float(np.percentile([p["latency"] for p in per_q], 95))
        results["systems"][name] = {
            "index_time": sys._index_time,
            "n_chunks": len(sys.chunks),
            "total_query_time": elapsed,
            "aggregate": agg,
            "per_question": per_q,
        }
        print(f"[nfcorpus] {name}  nDCG@10={agg['ndcg@10']:.3f}  R@5={agg['recall@5']:.3f}  R@10={agg['recall@10']:.3f}  R@100={agg['recall@100']:.3f}  MRR={agg['mrr@10']:.3f}  Lat={agg['latency_mean_ms']:.0f}ms", flush=True)

    save_json(results, RESULTS / "main_nfcorpus.json")
    print("Wrote", RESULTS / "main_nfcorpus.json")


if __name__ == "__main__":
    import sys as _sys
    mq = int(_sys.argv[1]) if len(_sys.argv) > 1 else None
    main(max_queries=mq)

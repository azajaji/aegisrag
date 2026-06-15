"""Generic BEIR-format retrieval runner.

BEIR datasets share a layout: corpus.jsonl + queries.jsonl + qrels/test.tsv
(or qrels/test_qrels.tsv). NFCorpus already uses this layout
(nfcorpus.py); this script handles any other BEIR-style dataset under
paper/data/<name>/.

Currently wired to: scifact, fiqa, nfcorpus.

For each dataset and each system in SYSTEMS we run retrieval over the
test queries and report:
  Recall@5, Recall@10, Recall@100, MRR@10, nDCG@10  (graded relevance).

Writes results/main_beir_<dataset>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from common import RESULTS, save_json
from systems import AutoRAG, BM25System, NaiveRAG, RAGRerank


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def _qrels_path(ds_root: Path) -> Path:
    for candidate in ["qrels/test.tsv", "qrels/test_qrels.tsv"]:
        p = ds_root / candidate
        if p.exists():
            return p
    raise FileNotFoundError(f"No test qrels file under {ds_root / 'qrels'}")


def load_beir(name: str, max_queries: int | None = None):
    root = DATA_ROOT / name
    corpus: dict[str, list[str]] = {}
    with open(root / "corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            title = (d.get("title") or "").strip()
            text = d["text"].strip()
            full = f"{title}. {text}" if title else text
            corpus[d["_id"]] = [full]
    with open(root / "queries.jsonl", "r", encoding="utf-8") as f:
        qmap = {json.loads(l)["_id"]: json.loads(l)["text"] for l in f}
    qrels: dict[str, dict[str, int]] = {}
    with open(_qrels_path(root), "r", encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        # Header may be "query-id\tcorpus-id\tscore" or similar; we just split
        # and use the three columns.
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, did, grade = parts[0], parts[1], parts[2]
            if did not in corpus:
                continue
            try:
                grade_i = int(grade)
            except ValueError:
                grade_i = 1 if grade.lower() in ("true", "yes", "1") else 0
            if grade_i <= 0:
                continue
            qrels.setdefault(qid, {})[did] = grade_i
    queries = []
    for qid, rel in qrels.items():
        if not rel or qid not in qmap:
            continue
        queries.append({"qid": qid, "question": qmap[qid], "rel": rel})
    queries.sort(key=lambda q: q["qid"])
    if max_queries:
        queries = queries[:max_queries]
    return corpus, queries


def _hit_metrics(ranked_doc_ids: list[str], rel: dict[str, int], ks=(5, 10, 100)) -> dict:
    out = {}
    rel_pos = set(rel.keys())
    total_rel = len(rel_pos)
    for k in ks:
        topk = ranked_doc_ids[:k]
        hits = sum(1 for d in topk if d in rel_pos)
        out[f"recall@{k}"] = hits / max(1, total_rel)
    mrr = 0.0
    for i, d in enumerate(ranked_doc_ids[:10], 1):
        if d in rel_pos:
            mrr = 1.0 / i
            break
    out["mrr@10"] = mrr
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
    # BEIR datasets here have no answerable/unanswerable split, so we
    # disable abstention as we did for NFCorpus.
    ("AutoRAG", AutoRAG, {"use_abstain": False}),
]


def run_one(name: str, max_queries: int | None = None):
    corpus, queries = load_beir(name, max_queries=max_queries)
    print(f"[{name}] {len(corpus)} docs, {len(queries)} test queries", flush=True)
    results = {
        "dataset": name,
        "n_questions": len(queries),
        "n_corpus": len(corpus),
        "systems": {},
    }
    for sys_name, SysCls, overrides in SYSTEMS:
        sys = SysCls(**overrides)
        print(f"  indexing {sys_name} ...", flush=True)
        sys.index(corpus)
        print(f"  indexed {len(sys.chunks)} chunks in {sys._index_time:.1f}s", flush=True)
        per_q = []
        t0 = time.perf_counter()
        for i, q in enumerate(queries, 1):
            out = sys.answer(q["question"], k_retrieve=100, k_top=10)
            seen: set[str] = set()
            ranked_doc_ids: list[str] = []
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
                print(f"    [{sys_name}] {i}/{len(queries)}", flush=True)
        elapsed = time.perf_counter() - t0
        agg = {}
        for m in ["recall@5", "recall@10", "recall@100", "mrr@10", "ndcg@10"]:
            agg[m] = float(np.mean([p[m] for p in per_q]))
        agg["latency_mean_ms"] = 1000 * float(np.mean([p["latency"] for p in per_q]))
        agg["latency_p95_ms"] = 1000 * float(np.percentile([p["latency"] for p in per_q], 95))
        results["systems"][sys_name] = {
            "index_time": sys._index_time,
            "n_chunks": len(sys.chunks),
            "total_query_time": elapsed,
            "aggregate": agg,
            "per_question": per_q,
        }
        print(f"  {sys_name}: nDCG@10={agg['ndcg@10']:.3f} "
              f"R@5={agg['recall@5']:.3f} R@10={agg['recall@10']:.3f} "
              f"R@100={agg['recall@100']:.3f} MRR={agg['mrr@10']:.3f} "
              f"Lat={agg['latency_mean_ms']:.0f}ms", flush=True)
    out_path = RESULTS / f"main_beir_{name}.json"
    save_json(results, out_path)
    print(f"wrote {out_path}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["scifact", "fiqa"])
    ap.add_argument("--max-queries", type=int, default=None)
    args = ap.parse_args()
    for ds in args.datasets:
        run_one(ds, max_queries=args.max_queries)


if __name__ == "__main__":
    main()

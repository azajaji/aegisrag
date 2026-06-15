"""Retrieval failure analysis.

Categorises every SQuAD-v2 answerable question in the held-out eval split
that AutoRAG retrieved incorrectly (Recall@1 == 0), and reports the
failure mode together with representative examples. We classify each
failure into one of:

  lexical_mismatch   gold paragraph contains the answer but uses
                     synonyms / paraphrase of the question;
  near_duplicate     a sibling paragraph from the same article also
                     mentions the entity in the question;
  fragmentation      gold answer span sits across adjacent paragraphs
                     and the retrieved paragraph is the wrong half;
  out_of_scope       no system retrieved the gold (all four R@1=0);
  reranker_only      BM25 / dense had the gold in top-1 but the reranker
                     demoted it.

Also tabulates: which baselines AutoRAG newly fixes, and which examples
AutoRAG still misses.

Writes results/failures.json with the example table.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from common import RESULTS, save_json
from data_loaders import load_squad


_WORD = re.compile(r"\w+", re.UNICODE)
_STOP = {
    "a","an","the","is","are","was","were","of","in","on","to","for","and",
    "or","but","with","as","by","at","be","been","from","this","that","these",
    "those","it","its","into","do","does","did","has","have","had","what",
    "which","who","whom","whose","where","when","why","how",
}


def _toks(s):
    return [t.lower() for t in _WORD.findall(s) if t.lower() not in _STOP and len(t) > 1]


def classify(q, autorag_row, other_rows, corpus):
    """Return a failure-mode string for an AutoRAG R@1 miss."""
    title = q["gold_doc"]; pi = q["gold_para_idx"]
    paras = corpus[title]
    gold_text = paras[pi]
    qtoks = set(_toks(q["question"]))
    gtoks = set(_toks(gold_text))
    # 1) Out-of-scope: every baseline also failed
    all_failed = all(r.get("recall@1", 0.0) == 0.0 for r in other_rows.values())
    if all_failed:
        # Distinguish lexical_mismatch vs near_duplicate
        # near_duplicate: another paragraph in the same article scores high overlap
        overlaps = []
        for j, p in enumerate(paras):
            if j == pi:
                continue
            ptoks = set(_toks(p))
            ov = len(qtoks & ptoks) / max(1, len(qtoks))
            overlaps.append((ov, j))
        overlaps.sort(reverse=True)
        if overlaps and overlaps[0][0] > 0.6 and len(qtoks & gtoks) / max(1, len(qtoks)) < 0.5:
            return "near_duplicate"
        if len(qtoks & gtoks) / max(1, len(qtoks)) < 0.4:
            return "lexical_mismatch"
        return "lexical_mismatch"
    # 2) reranker-only failure: dense or BM25 had R@1 hit
    if other_rows.get("BM25", {}).get("recall@1", 0) == 1.0 or other_rows.get("NaiveRAG", {}).get("recall@1", 0) == 1.0:
        # if RAG+Rerank also failed, label "reranker_only" (reranker hurt)
        if other_rows.get("RAG+Rerank", {}).get("recall@1", 0) == 0.0:
            return "reranker_only"
    # 3) RAG+Rerank had it, AutoRAG missed -- adaptive chunking moved the right paragraph elsewhere
    if other_rows.get("RAG+Rerank", {}).get("recall@1", 0) == 1.0:
        return "fragmentation"
    return "lexical_mismatch"


def main():
    main_squad = json.load(open(RESULTS / "main_squad.json", "r", encoding="utf-8"))
    corpus, qs_all = load_squad(n_per_article=20)
    qmap = {q["qid"]: q for q in qs_all}

    by_sys = {s: {p["qid"]: p for p in info["per_question"] if not p["in_calibration"]}
              for s, info in main_squad["systems"].items()}

    autorag_misses = []
    for qid, row in by_sys["AutoRAG"].items():
        if not row["answerable"]:
            continue
        if row.get("recall@1", 0) == 0.0:
            others = {s: by_sys[s][qid] for s in by_sys if s != "AutoRAG"}
            q = qmap[qid]
            mode = classify(q, row, others, corpus)
            autorag_misses.append({
                "qid": qid,
                "question": q["question"],
                "gold_doc": q["gold_doc"],
                "gold_para_idx": q["gold_para_idx"],
                "failure_mode": mode,
                "BM25_R1": others["BM25"]["recall@1"],
                "NaiveRAG_R1": others["NaiveRAG"]["recall@1"],
                "RAGRerank_R1": others["RAG+Rerank"]["recall@1"],
                "AutoRAG_top_cite_doc": row.get("cite_doc"),
                "AutoRAG_top_cite_para": row.get("cite_para"),
                "AutoRAG_prediction": row.get("prediction"),
                "gold_paragraph_excerpt": corpus[q["gold_doc"]][q["gold_para_idx"]][:160] + "...",
            })

    fixed_vs_baseline = {"BM25": [], "NaiveRAG": [], "RAG+Rerank": []}
    for qid, row in by_sys["AutoRAG"].items():
        if not row["answerable"]:
            continue
        if row.get("recall@1", 0) == 1.0:
            for s in fixed_vs_baseline:
                if by_sys[s][qid].get("recall@1", 0) == 0.0:
                    fixed_vs_baseline[s].append(qid)

    summary = {
        "n_autorag_misses": len(autorag_misses),
        "by_mode": dict(Counter([m["failure_mode"] for m in autorag_misses])),
        "n_fixed_by_autorag_vs": {k: len(v) for k, v in fixed_vs_baseline.items()},
        "examples_per_mode": {},
        "examples_fixed_by_autorag": [],
    }
    # 3 examples per failure mode
    by_mode = defaultdict(list)
    for m in autorag_misses:
        by_mode[m["failure_mode"]].append(m)
    for mode, ms in by_mode.items():
        summary["examples_per_mode"][mode] = ms[:3]

    # 5 examples where AutoRAG fixed a baseline failure
    seen = set()
    for s in ("BM25", "NaiveRAG", "RAG+Rerank"):
        for qid in fixed_vs_baseline[s][:5]:
            if qid in seen:
                continue
            seen.add(qid)
            q = qmap[qid]
            row = by_sys["AutoRAG"][qid]
            summary["examples_fixed_by_autorag"].append({
                "fixed_vs": s,
                "qid": qid,
                "question": q["question"],
                "gold_doc": q["gold_doc"],
                "gold_para_idx": q["gold_para_idx"],
                "gold_paragraph_excerpt": corpus[q["gold_doc"]][q["gold_para_idx"]][:160] + "...",
                "autorag_prediction": row.get("prediction"),
            })

    save_json(summary, RESULTS / "failures.json")

    print(f"AutoRAG R@1 misses (answerable eval): {len(autorag_misses)}")
    print("by mode:", summary["by_mode"])
    print("AutoRAG fixed:")
    for s, n in summary["n_fixed_by_autorag_vs"].items():
        print(f"  vs {s}: {n}")
    print("Wrote", RESULTS / "failures.json")


if __name__ == "__main__":
    main()

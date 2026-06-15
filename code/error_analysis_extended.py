"""Extended error analysis: counts per failure category for AutoRAG on
SQuAD-v2 and Enterprise eval splits, plus 'AutoRAG fixes' counts
relative to RAG+Rerank.

Uses existing main_*.json per-question records. The categorisation is
heuristic but reproducible; it complements the manual 4-mode taxonomy
in failures.json.
"""
import json

from common import RESULTS


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def categorise(ds):
    main = load(f"main_{ds}.json")
    auto = {r["qid"]: r for r in main["systems"]["AutoRAG"]["per_question"]}
    rr = {r["qid"]: r for r in main["systems"]["RAG+Rerank"]["per_question"]}

    # AutoRAG false refusals: answerable AND AutoRAG returned empty prediction.
    false_refusals = []
    # AutoRAG non-refusals on unanswerable: not abstained + not answerable.
    nonrefusals_unans = []
    # AutoRAG wrong citations: answerable + answered (non-empty pred) +
    # cite_doc/cite_para != gold.
    wrong_citations = []
    # AutoRAG fixes: AutoRAG R@1=1 AND RAG+Rerank R@1=0 on the same query
    # (both restricted to the EVALUATION split).
    fixes = []
    # AutoRAG regressions: opposite direction.
    regressions = []

    for qid, ar in auto.items():
        if ar.get("in_calibration"):
            continue
        # False refusal
        if ar["answerable"] and not ar.get("prediction"):
            false_refusals.append(qid)
        # Non-refusal on unanswerable
        if (not ar["answerable"]) and ar.get("prediction"):
            nonrefusals_unans.append(qid)
        # Wrong citation
        if ar["answerable"] and ar.get("prediction") and ar.get("citation", 1) == 0:
            wrong_citations.append(qid)
        # Fixes (R@1 = AutoRAG had gold at top-1 but RAG+Rerank didn't)
        if qid in rr and not rr[qid].get("in_calibration"):
            if ar["answerable"]:
                if ar.get("recall@1", 0) == 1 and rr[qid].get("recall@1", 0) == 0:
                    fixes.append(qid)
                if ar.get("recall@1", 0) == 0 and rr[qid].get("recall@1", 0) == 1:
                    regressions.append(qid)
    return {
        "false_refusals_n": len(false_refusals),
        "nonrefusals_unans_n": len(nonrefusals_unans),
        "wrong_citations_n": len(wrong_citations),
        "autorag_fixes_n": len(fixes),
        "autorag_regressions_n": len(regressions),
        "example_qids": {
            "false_refusals": false_refusals[:10],
            "nonrefusals_unans": nonrefusals_unans[:10],
            "wrong_citations": wrong_citations[:10],
            "fixes_vs_rag_rerank": fixes[:10],
            "regressions_vs_rag_rerank": regressions[:10],
        },
    }


def main():
    out = {ds: categorise(ds) for ds in ["squad", "enterprise"]}
    (RESULTS / "error_analysis_extended.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    for ds, d in out.items():
        print(f"\n=== {ds} eval split ===")
        print(f"  False refusals (answerable -> abstained):      {d['false_refusals_n']}")
        print(f"  Non-refusals on unanswerable (hallucination):  {d['nonrefusals_unans_n']}")
        print(f"  Wrong citations (answered, cite_doc!=gold):    {d['wrong_citations_n']}")
        print(f"  AutoRAG fixes RAG+Rerank R@1=0 -> R@1=1:        {d['autorag_fixes_n']}")
        print(f"  AutoRAG regressions R@1=1 -> R@1=0:             {d['autorag_regressions_n']}")


if __name__ == "__main__":
    main()

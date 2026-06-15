"""Pilot rubric-based expert evaluation of LLM-generated answers.

Cannot recruit external raters for this pilot study; we instead apply a
deterministic rubric grounded in gold answers + retrieved evidence and
report it explicitly as an "author-rubric pilot" alongside the automatic
metrics. A multi-rater human study with at least three independent raters
and inter-rater agreement statistics is left as future work.

Rubric (per-answer):
  Usefulness   3 = pred F1 vs gold >= 0.8 and faithfulness >= 0.7
                       (answers the question fully and is grounded)
               2 = pred F1 in [0.4, 0.8)  and faithfulness >= 0.5
                       (partially correct and grounded)
               1 = pred F1 in [0.1, 0.4)  or faithfulness in [0.3, 0.5)
                       (related but inaccurate or ungrounded)
               0 = otherwise                            (not useful)
  CitationOK   1 = top-1 cited chunk is the gold paragraph OR contains
                   >= 80% of the gold answer's content tokens
               0 = otherwise

For unanswerable items:
  Usefulness   3 = system refused (correctly)
               0 = system answered (a hallucinated answer is not useful)
  CitationOK   1 = refused; 0 = answered

We stratified-sample 60 items per dataset (50% answerable / 50%
unanswerable) and report mean usefulness, % CitationOK, and the joint
distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import RESULTS, save_json
from systems import _content_tokens


def _faithfulness_against_doc(pred: str, ctx_or_para: str) -> float:
    pt = _content_tokens(pred)
    if not pt:
        return 1.0
    bag = set(_content_tokens(ctx_or_para))
    return sum(1 for t in pt if t in bag) / len(pt)


def _rubric(row: dict, gold_answers: list[str], gold_para_text: str | None) -> dict:
    pred = row.get("llm_pred", "").strip()
    answerable = row["answerable"]
    refused = row.get("refused", False) or not pred
    if not answerable:
        usefulness = 3 if refused else 0
        citation_ok = 1 if refused else 0
        return {"usefulness": usefulness, "citation_ok": citation_ok}
    # Answerable.
    f1 = row.get("f1", 0.0)
    faith = row.get("faithfulness", 0.0)
    if f1 >= 0.8 and faith >= 0.7:
        usefulness = 3
    elif f1 >= 0.4 and faith >= 0.5:
        usefulness = 2
    elif f1 >= 0.1 or 0.3 <= faith < 0.5:
        usefulness = 1
    else:
        usefulness = 0
    # Citation: did we cite the gold paragraph, or does the cited paragraph contain the gold answer?
    citation_ok = int(row.get("citation", 0.0) >= 0.5)
    if not citation_ok and gold_para_text is not None and gold_answers:
        for ga in gold_answers:
            ga_toks = set(_content_tokens(ga))
            if not ga_toks:
                continue
            cite_toks = set(_content_tokens(gold_para_text))
            overlap = sum(1 for t in ga_toks if t in cite_toks) / len(ga_toks)
            if overlap >= 0.8:
                citation_ok = 1
                break
    return {"usefulness": usefulness, "citation_ok": citation_ok}


def _load_gold_paragraphs(dataset: str) -> dict[tuple[str, int], str]:
    from data_loaders import load_enterprise, load_squad
    if dataset == "enterprise":
        corpus, _ = load_enterprise()
    else:
        corpus, _ = load_squad()
    return {(doc, i): para for doc, paras in corpus.items() for i, para in enumerate(paras)}


def _stratified_sample(rows: list[dict], n: int, seed: int = 20260512):
    rng = np.random.default_rng(seed)
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    rng.shuffle(ans); rng.shuffle(una)
    half = n // 2
    return ans[: min(half, len(ans))] + una[: min(n - half, len(una))]


def _load_question_index(dataset: str):
    """Return qid -> gold_answers list and gold_para_idx + gold_doc."""
    from data_loaders import load_enterprise, load_squad
    if dataset == "enterprise":
        _, qs = load_enterprise()
    else:
        _, qs = load_squad()
    return {q["qid"]: q for q in qs}


def evaluate(dataset: str, sample_size: int = 60):
    path = RESULTS / f"llm_{dataset}.json"
    if not path.exists():
        print(f"missing {path}, skipping {dataset}")
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    qmap = _load_question_index(dataset)
    para_map = _load_gold_paragraphs(dataset)

    out = {"dataset": dataset, "sample_size": sample_size, "per_system": {}}
    sample_qids = None
    for sname, info in rec["systems"].items():
        rows = [p for p in info["per_question"] if not p["in_calibration"]]
        if sample_qids is None:
            sampled = _stratified_sample(rows, sample_size)
            sample_qids = {r["qid"] for r in sampled}
            out["sampled_qids"] = sorted(sample_qids, key=lambda x: str(x))
        # restrict to same sample for each system
        sub = [r for r in rows if r["qid"] in sample_qids]
        ratings = []
        for r in sub:
            q = qmap.get(r["qid"])
            if q is None:
                continue
            gold_para_text = None
            if q.get("answerable"):
                gold_para_text = para_map.get((q["gold_doc"], q["gold_para_idx"]))
            rating = _rubric(r, q.get("gold_answers", []), gold_para_text)
            ratings.append({
                "qid": r["qid"],
                "answerable": r["answerable"],
                "f1": r.get("f1"),
                "faithfulness": r.get("faithfulness"),
                "citation": r.get("citation"),
                "usefulness": rating["usefulness"],
                "citation_ok": rating["citation_ok"],
                "refused": r.get("refused"),
                "pred": r.get("llm_pred", ""),
                "gold": q.get("gold_answers", []),
            })
        mean_use = float(np.mean([rr["usefulness"] for rr in ratings])) if ratings else 0.0
        cite_ok = float(np.mean([rr["citation_ok"] for rr in ratings])) if ratings else 0.0
        # use-broken-down by class
        useful_ge2 = float(np.mean([1 if rr["usefulness"] >= 2 else 0 for rr in ratings])) if ratings else 0.0
        useful_3 = float(np.mean([1 if rr["usefulness"] >= 3 else 0 for rr in ratings])) if ratings else 0.0
        out["per_system"][sname] = {
            "n": len(ratings),
            "mean_usefulness": mean_use,
            "pct_usefulness_ge2": useful_ge2,
            "pct_usefulness_3": useful_3,
            "pct_citation_ok": cite_ok,
            "ratings": ratings,
        }
    return out


def main():
    out = {}
    for ds in ("enterprise", "squad"):
        s = evaluate(ds)
        if s is not None:
            out[ds] = s
    save_json(out, RESULTS / "expert.json")
    # Print summary
    for ds, dout in out.items():
        print(f"== {ds} ==")
        for s, info in dout["per_system"].items():
            print(f"  {s:>12s}: mean_useful={info['mean_usefulness']:.3f}  pct_useful>=2={info['pct_usefulness_ge2']:.3f}  pct_useful=3={info['pct_usefulness_3']:.3f}  pct_cite_ok={info['pct_citation_ok']:.3f}  n={info['n']}")
    print("Wrote", RESULTS / "expert.json")


if __name__ == "__main__":
    main()

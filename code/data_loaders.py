"""Dataset loaders.

We support two benchmarks:
  * SQuAD v2.0 dev set, subsampled  -- main public benchmark.
  * AutoRAG-Enterprise (synthetic)  -- supplementary multi-domain benchmark.

Both expose the same interface:
  load_corpus(...) -> dict[doc_id, list[paragraph_str]]
  load_questions(...) -> list[dict] with keys:
      qid, question, gold_doc, gold_para_idx, gold_answers (list), answerable, domain
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from common import DATA


SQUAD_PATH = DATA / "squad" / "dev-v2.0.json"


def load_squad(n_per_article: int = 20, seed: int = 20260511):
    """Load a deterministic subsample of SQuAD v2 dev.

    Returns (corpus, questions). The corpus maps doc_id = article title to
    a list of paragraph strings. gold_para_idx is the index of the paragraph
    in that list.
    """
    raw = json.loads(SQUAD_PATH.read_text(encoding="utf-8"))
    rng = random.Random(seed)

    corpus: dict[str, list[str]] = {}
    para_index: dict[str, dict[str, int]] = {}  # title -> {context_text: idx}
    questions: list[dict] = []

    for article in raw["data"]:
        title = article["title"]
        corpus[title] = []
        para_index[title] = {}
        # Build paragraph index for this article
        for paragraph in article["paragraphs"]:
            ctx = paragraph["context"]
            if ctx not in para_index[title]:
                para_index[title][ctx] = len(corpus[title])
                corpus[title].append(ctx)
        # Collect candidate questions per article
        cand_ans, cand_un = [], []
        for paragraph in article["paragraphs"]:
            ctx = paragraph["context"]
            pi = para_index[title][ctx]
            for qa in paragraph["qas"]:
                base = {
                    "qid": qa["id"],
                    "question": qa["question"],
                    "gold_doc": title,
                    "gold_para_idx": pi,
                    "domain": _domain_for_title(title),
                }
                if qa.get("is_impossible"):
                    base["answerable"] = False
                    base["gold_answers"] = []
                    cand_un.append(base)
                else:
                    answers = [a["text"] for a in qa.get("answers", [])]
                    if not answers:
                        continue
                    base["answerable"] = True
                    base["gold_answers"] = answers
                    cand_ans.append(base)

        rng.shuffle(cand_ans)
        rng.shuffle(cand_un)
        target_ans = n_per_article // 2
        target_un = n_per_article - target_ans
        questions.extend(cand_ans[:target_ans])
        questions.extend(cand_un[:target_un])

    rng.shuffle(questions)
    return corpus, questions


# A coarse topical grouping over the 35 SQuAD-dev articles so we can stratify.
_TITLE_DOMAIN = {
    # Politics / governance
    "Normans": "history",
    "Computational_complexity_theory": "computing",
    "Southern_California": "geography",
    "Sky_(United_Kingdom)": "media",
    "Victoria_(Australia)": "geography",
    "Huguenot": "history",
    "Steam_engine": "engineering",
    "Oxygen": "science",
    "1973_oil_crisis": "history",
    "European_Union_law": "law",
    "Amazon_rainforest": "geography",
    "Ctenophora": "biology",
    "Fresno,_California": "geography",
    "Packet_switching": "computing",
    "Black_Death": "history",
    "Geology": "science",
    "Pharmacy": "medicine",
    "Civil_disobedience": "history",
    "Construction": "engineering",
    "Private_school": "education",
    "Harvard_University": "education",
    "Jacksonville,_Florida": "geography",
    "Economic_inequality": "economics",
    "Doctor_Who": "media",
    "University_of_Chicago": "education",
    "Yuan_dynasty": "history",
    "Kenya": "geography",
    "Intergovernmental_Panel_on_Climate_Change": "science",
    "Chloroplast": "biology",
    "Prime_number": "math",
    "Rhine": "geography",
    "Scottish_Parliament": "law",
    "Islamism": "history",
    "Imperialism": "history",
    "Warsaw": "geography",
    "French_and_Indian_War": "history",
    "Force": "science",
}


def _domain_for_title(title: str) -> str:
    return _TITLE_DOMAIN.get(title, "other")


# ---------------------------------------------------------------------------
# Synthetic enterprise benchmark (policy / IT / HR).
# ---------------------------------------------------------------------------

def load_enterprise():
    bench = json.loads((DATA / "questions" / "benchmark.json").read_text(encoding="utf-8"))
    corpus: dict[str, list[str]] = {}
    for domain in ("policy", "itdocs", "hr"):
        ddir = DATA / domain
        for fp in sorted(ddir.glob("*.txt")):
            # doc_id is "<domain>/<filename>" so the same filename can exist
            # across domains without colliding.
            doc_id = f"{domain}/{fp.name}"
            paragraphs = [p.strip() for p in fp.read_text(encoding="utf-8").split("\n\n") if p.strip()]
            corpus[doc_id] = paragraphs
    questions: list[dict] = []
    for q in bench:
        gd = q.get("gold_doc")
        gold_doc = f"{q['domain']}/{gd}" if gd else None
        questions.append(
            {
                "qid": f"ent-{q['qid']}",
                "question": q["question"],
                "gold_doc": gold_doc,
                "gold_para_idx": q.get("gold_para_idx"),
                "gold_answers": [q["gold_answer"]] if q.get("gold_answer") else [],
                "answerable": q["answerable"],
                "domain": q["domain"],
            }
        )
    return corpus, questions

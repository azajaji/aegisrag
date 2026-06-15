"""Shared utilities: text normalization, SQuAD metrics, chunking, etc."""
from __future__ import annotations

import json
import math
import os
import re
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SQuAD-style answer normalization and EM/F1 metrics.
# ---------------------------------------------------------------------------
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLE_RE.sub(" ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = {}
    for w in pt:
        common[w] = common.get(w, 0) + 1
    matches = 0
    for w in gt:
        if common.get(w, 0) > 0:
            matches += 1
            common[w] -= 1
    if matches == 0:
        return 0.0
    p = matches / len(pt)
    r = matches / len(gt)
    return 2 * p * r / (p + r)


def em_score(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def best_against(pred: str, golds: list[str]) -> tuple[float, float]:
    if not golds:
        # Unanswerable: prediction empty => correct
        return (1.0, 1.0) if pred.strip() == "" else (0.0, 0.0)
    em = max(em_score(pred, g) for g in golds)
    f1 = max(f1_score(pred, g) for g in golds)
    return em, f1


# ---------------------------------------------------------------------------
# Chunk container.
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: int
    doc_id: str           # logical document (article title for SQuAD)
    para_id: int          # paragraph index within the doc
    text: str
    subspan: tuple[int, int] | None = None   # (start, end) in para text if split


def split_to_chunks(doc_id: str, paragraphs: list[str], mode: str, max_tokens: int = 120) -> list[Chunk]:
    """Produce chunks per chunking mode.

    paragraph  : one chunk per paragraph (default for naive RAG)
    fixed_N    : split each paragraph into roughly N-token windows with 20% overlap
    semantic   : sentence-level grouping until cumulative tokens >= max_tokens
    adaptive   : pick per-doc strategy based on avg paragraph length
                 (short paragraphs -> sentence grouping target ~120 tokens;
                 long paragraphs -> fixed-120 with overlap).
    """
    chunks: list[Chunk] = []
    cid = 0
    if mode == "paragraph":
        for pi, para in enumerate(paragraphs):
            chunks.append(Chunk(cid, doc_id, pi, para))
            cid += 1
        return chunks

    if mode.startswith("fixed_"):
        target = int(mode.split("_", 1)[1])
        for pi, para in enumerate(paragraphs):
            words = para.split()
            if not words:
                continue
            step = max(1, int(target * 0.8))
            i = 0
            while i < len(words):
                window = words[i : i + target]
                if not window:
                    break
                chunks.append(Chunk(cid, doc_id, pi, " ".join(window), (i, i + len(window))))
                cid += 1
                if i + target >= len(words):
                    break
                i += step
        return chunks

    if mode == "semantic":
        for pi, para in enumerate(paragraphs):
            sents = _sent_split(para)
            buf, btoks = [], 0
            start = 0
            for s in sents:
                stoks = max(1, len(s.split()))
                if btoks + stoks > max_tokens and buf:
                    chunks.append(Chunk(cid, doc_id, pi, " ".join(buf), (start, start + btoks)))
                    cid += 1
                    start += btoks
                    buf, btoks = [], 0
                buf.append(s)
                btoks += stoks
            if buf:
                chunks.append(Chunk(cid, doc_id, pi, " ".join(buf), (start, start + btoks)))
                cid += 1
        return chunks

    if mode == "adaptive":
        # Decide per-doc: avg paragraph length
        avg_words = np.mean([max(1, len(p.split())) for p in paragraphs])
        if avg_words < 80:
            # already small — keep paragraph chunks
            return split_to_chunks(doc_id, paragraphs, "paragraph")
        if avg_words < 200:
            return split_to_chunks(doc_id, paragraphs, "semantic", max_tokens=max_tokens)
        return split_to_chunks(doc_id, paragraphs, "fixed_120")

    raise ValueError(f"unknown chunking mode: {mode}")


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d])")


def _sent_split(p: str) -> list[str]:
    parts = _SENT_RE.split(p.strip())
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Retrieval metric helpers.
# ---------------------------------------------------------------------------

def recall_at_k(ranked: list[Chunk], gold: tuple[str, int], k: int) -> float:
    for c in ranked[:k]:
        if c.doc_id == gold[0] and c.para_id == gold[1]:
            return 1.0
    return 0.0


def mrr_first(ranked: list[Chunk], gold: tuple[str, int], k: int = 10) -> float:
    for i, c in enumerate(ranked[:k], 1):
        if c.doc_id == gold[0] and c.para_id == gold[1]:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[Chunk], gold: tuple[str, int], k: int) -> float:
    # binary relevance, single gold => nDCG@k = 1/log2(rank+1) at hit, else 0
    for i, c in enumerate(ranked[:k], 1):
        if c.doc_id == gold[0] and c.para_id == gold[1]:
            return 1.0 / math.log2(i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# JSON IO.
# ---------------------------------------------------------------------------

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_jsondef), encoding="utf-8")


def _jsondef(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(o)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

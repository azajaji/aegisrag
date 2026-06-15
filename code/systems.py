"""RAG system implementations: BM25, NaiveRAG, RAG+Reranker, AutoRAG.

All systems implement two methods:
    index(corpus: dict[doc_id, list[paragraph]]) -> None
    answer(question: str) -> dict with keys
        ranked     : list[Chunk]
        prediction : answer string (empty => abstain)
        cite_doc   : doc_id used in answer
        cite_para  : paragraph index used in answer
        latency    : seconds

For reproducibility, AutoRAG abstains by default for unanswerable detection
when the reranker confidence is below a threshold.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from common import Chunk, split_to_chunks


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s)]


# ===========================================================================
# Encoder caches (loaded once per process).
# ===========================================================================
_ENCODERS: dict[str, SentenceTransformer] = {}
_CROSS: dict[str, CrossEncoder] = {}


def get_encoder(name: str) -> SentenceTransformer:
    if name not in _ENCODERS:
        _ENCODERS[name] = SentenceTransformer(name)
    return _ENCODERS[name]


def get_crossencoder(name: str) -> CrossEncoder:
    if name not in _CROSS:
        _CROSS[name] = CrossEncoder(name)
    return _CROSS[name]


DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MULTILINGUAL_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ===========================================================================
# Answer extraction.
# ===========================================================================
import re as _re

_SENT_END = _re.compile(r"(?<=[.!?])\s+(?=[A-Z\d])")
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "to", "for",
    "and", "or", "but", "with", "as", "by", "at", "be", "been", "from", "this",
    "that", "these", "those", "it", "its", "into", "do", "does", "did", "has",
    "have", "had", "what", "which", "who", "whom", "whose", "where", "when",
    "why", "how", "many", "much", "long", "tall", "wide", "deep", "old", "big",
    "small", "year", "years", "month", "months", "day", "days", "people",
    "person", "name", "named", "called", "first", "last", "between", "during",
    "after", "before", "over", "under", "than", "then", "any", "some", "all",
    "most", "other", "another", "their", "they", "them", "his", "her", "him",
    "she", "he",
}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_END.split(text.strip()) if s.strip()]


def _content_tokens(q: str) -> list[str]:
    return [t for t in _tokenize(q) if t not in _STOP and len(t) > 1]


def extract_answer(question: str, passage: str) -> str:
    """Lightweight extractive QA.

    Strategy:
    * Pick the sentence in the passage with the highest unigram overlap with
      the content words of the question.
    * Then return the shortest noun-phrase-like span in that sentence that
      contains the most informative tokens not present in the question.
    The span is intended to approximate a SQuAD answer.
    """
    qtoks = set(_content_tokens(question))
    if not qtoks:
        return passage.split(".")[0].strip()
    sents = _sentences(passage) or [passage]
    best, best_score = sents[0], -1.0
    for s in sents:
        toks = _content_tokens(s)
        score = sum(1 for t in toks if t in qtoks)
        score -= 0.001 * len(toks)  # prefer shorter, denser sentences on ties
        if score > best_score:
            best_score, best = score, s
    # Cut to span: shortest window containing >=1 question-content token
    # plus the most "novel" token; fall back to the whole sentence.
    return _span_from_sentence(best, qtoks)


_SPAN_PUNCT_RE = re.compile(r"[,;:()\[\]\"]")
_QUESTION_LEAD = {
    "how", "what", "which", "when", "where", "who", "whom", "whose", "why",
}


def _span_from_sentence(sent: str, qtoks: set[str]) -> str:
    chunks = [c.strip() for c in _SPAN_PUNCT_RE.split(sent) if c.strip()]
    chunks = [c for c in chunks if c]
    if not chunks:
        return sent.strip()
    # Pick the chunk with most non-question informative tokens
    scored = []
    for c in chunks:
        toks = _content_tokens(c)
        novel = [t for t in toks if t not in qtoks]
        scored.append((len(novel), -len(toks), c))
    scored.sort(reverse=True)
    return scored[0][2]


# ===========================================================================
# Base class with index machinery.
# ===========================================================================

class BaseSystem:
    name: str = "base"
    chunking_mode: str = "paragraph"
    encoder_name: str = DEFAULT_ENCODER
    use_dense: bool = True
    use_bm25: bool = False
    use_reranker: bool = False
    use_query_rewrite: bool = False
    use_abstain: bool = False
    citation: bool = True

    def __init__(self, **overrides):
        for k, v in overrides.items():
            setattr(self, k, v)
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self._index_time = 0.0

    def index(self, corpus: dict[str, list[str]]):
        t0 = time.perf_counter()
        chunks: list[Chunk] = []
        cid = 0
        for doc_id, paras in corpus.items():
            for ch in split_to_chunks(doc_id, paras, self.chunking_mode):
                ch.chunk_id = cid
                cid += 1
                chunks.append(ch)
        self.chunks = chunks
        if self.use_dense:
            enc = get_encoder(self.encoder_name)
            self.embeddings = enc.encode(
                [c.text for c in chunks],
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        if self.use_bm25:
            self.bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
        self._index_time = time.perf_counter() - t0

    def _retrieve(self, question: str, k: int) -> list[tuple[int, float]]:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        if self.use_dense and self.embeddings is not None:
            enc = get_encoder(self.encoder_name)
            qv = enc.encode([question], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
            sims = self.embeddings @ qv
            scores += sims
        if self.use_bm25 and self.bm25 is not None:
            bs = np.asarray(self.bm25.get_scores(_tokenize(question)), dtype=np.float32)
            if bs.max() > 0:
                bs = bs / (bs.max() + 1e-9)
            weight = 0.5 if self.use_dense else 1.0
            scores += weight * bs
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]

    def _rewrite(self, question: str) -> str:
        # Lightweight: drop trailing '?', lowercase WH, append top content words
        q = question.strip().rstrip("?")
        keep = [t for t in _tokenize(q) if t not in _STOP]
        if not keep:
            return q
        return q + " " + " ".join(keep[:6])

    # Abstention parameters (used only when self.use_abstain).
    abstain_score_tau: float = 2.0     # top reranker logit minimum
    abstain_margin_tau: float = 1.0    # top1 minus mean(top2..top5)

    def answer(self, question: str, k_retrieve: int = 20, k_top: int = 5) -> dict:
        t0 = time.perf_counter()
        q = self._rewrite(question) if self.use_query_rewrite else question
        cand = self._retrieve(q, k_retrieve)
        ranked_chunks = [self.chunks[i] for i, _ in cand]
        rerank_scores: list[float] | None = None
        if self.use_reranker and ranked_chunks:
            ce = get_crossencoder(RERANKER)
            pairs = [(question, c.text) for c in ranked_chunks]
            scores = ce.predict(pairs, show_progress_bar=False).tolist()
            order = np.argsort(-np.asarray(scores))
            ranked_chunks = [ranked_chunks[i] for i in order]
            rerank_scores = [float(scores[i]) for i in order]

        top_chunks = ranked_chunks[:k_top]
        top_score = rerank_scores[0] if rerank_scores else (cand[0][1] if cand else 0.0)

        # Margin = top1 - mean(top2..top5)
        margin = 0.0
        if rerank_scores and len(rerank_scores) >= 2:
            tail = rerank_scores[1:min(5, len(rerank_scores))]
            if tail:
                margin = float(rerank_scores[0] - sum(tail) / len(tail))

        prediction = ""
        cite_doc, cite_para = None, None
        abstained = False
        if top_chunks:
            top = top_chunks[0]
            if (
                self.use_abstain
                and rerank_scores is not None
                and (top_score < self.abstain_score_tau or margin < self.abstain_margin_tau)
            ):
                prediction = ""
                abstained = True
            else:
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
            "abstained": abstained,
        }


# ===========================================================================
# Concrete systems.
# ===========================================================================

class BM25System(BaseSystem):
    name = "BM25"
    use_dense = False
    use_bm25 = True
    use_reranker = False


class NaiveRAG(BaseSystem):
    name = "NaiveRAG"
    chunking_mode = "paragraph"
    use_dense = True
    use_bm25 = False
    use_reranker = False


class RAGRerank(BaseSystem):
    name = "RAG+Rerank"
    chunking_mode = "paragraph"
    use_dense = True
    use_bm25 = False
    use_reranker = True


class AutoRAG(BaseSystem):
    name = "AutoRAG"
    chunking_mode = "adaptive"
    use_dense = True
    use_bm25 = True            # hybrid retrieval
    use_reranker = True
    use_query_rewrite = True
    use_abstain = True

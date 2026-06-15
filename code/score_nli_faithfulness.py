"""Entailment-based faithfulness sensitivity check.

Companion to run_llm.py / run_llm_frontier.py. Re-scores the answered-and-
not-refused subset of each system's LLM output with an NLI model to give
an entailment-based faithfulness proxy alongside the lightweight token-
overlap proxy used in the headline tables.

For each (system, question) we form the premise from the retrieved top-k
chunks (re-running retrieval against the corpus if the source JSON does
not already include them) and the hypothesis from the generated answer;
we report the entailment probability and the binary supported flag
(entailment >= threshold). The script does not modify any existing
results files; it writes results/nli_faithfulness_<source>.json with
per-question entailment scores and per-system aggregate means with 95%
bootstrap CIs.

This is a sensitivity check, not a replacement for the headline
faithfulness metric. The headline table reports content-token overlap
because it is provider-agnostic and deterministic; this NLI sensitivity
check confirms whether system rankings are robust to a stronger
entailment-style metric.

Default model: cross-encoder/nli-deberta-v3-base (~180M params,
CPU-friendly). Pass --nli-model roberta-large-mnli for the heavier
350M-param alternative.

CPU runtime estimate (deberta-v3-base):
  Enterprise (305 q, 4 systems, ~75% answered): ~10-15 min
  SQuAD-v2 subsample (200 q, 4 systems, ~80% answered): ~10-15 min
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import RESULTS
from data_loaders import load_enterprise, load_squad
from run_llm import SYSTEMS as LLM_SYSTEMS


DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# label index for entailment depends on the model. Both deberta-v3-base
# (Sentence-Transformers cross-encoder) and roberta-large-mnli use the
# same convention: [contradiction, neutral, entailment].
ENTAIL_IDX = 2


_tok = None
_model = None
_loaded_model_name: str | None = None


def _load_nli(model_name: str):
    global _tok, _model, _loaded_model_name
    if _loaded_model_name == model_name:
        return
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print(f"[NLI] loading {model_name} ...", flush=True)
    _tok = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSequenceClassification.from_pretrained(model_name)
    _model.eval()
    _loaded_model_name = model_name


def entail_prob(premise: str, hypothesis: str, model_name: str) -> float:
    import torch
    _load_nli(model_name)
    enc = _tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = _model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
    return float(probs[ENTAIL_IDX])


def bootstrap_mean_ci(xs: list[float], n_boot: int = 2000, seed: int = 20260512) -> tuple[float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(xs, dtype=float)
    mean = float(arr.mean())
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
    means = samples.mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return mean, lo, hi


def _load_corpus_for(dataset_name: str):
    if dataset_name == "enterprise":
        return load_enterprise()
    if dataset_name == "squad":
        return load_squad()
    raise ValueError(f"unknown dataset: {dataset_name}")


def _system_class_for(sys_name: str):
    for name, cls, overrides in LLM_SYSTEMS:
        if name == sys_name:
            return cls, dict(overrides)
    raise KeyError(f"system not registered: {sys_name}")


def score_dataset(src_path: Path, out_path: Path, threshold: float,
                  model_name: str, top_k: int = 3,
                  recompute_contexts: bool = True):
    data = json.loads(src_path.read_text(encoding="utf-8"))
    ds_name = data.get("dataset")
    print(f"\n[NLI] scoring {src_path.name} (dataset={ds_name})", flush=True)

    indexed = {}
    questions_by_qid = {}
    if recompute_contexts:
        corpus, all_qs = _load_corpus_for(ds_name)
        questions_by_qid = {q["qid"]: q for q in all_qs}

    out = {
        "source": str(src_path.name),
        "dataset": ds_name,
        "model": data.get("model"),
        "nli_model": model_name,
        "threshold": threshold,
        "systems": {},
    }
    for sys_name, sys_payload in data.get("systems", {}).items():
        per_q = sys_payload.get("per_question", [])
        if recompute_contexts and sys_name not in indexed:
            SysCls, overrides = _system_class_for(sys_name)
            inst = SysCls(**overrides)
            print(f"  indexing {sys_name} ...", flush=True)
            inst.index(corpus)
            indexed[sys_name] = inst

        scored = []
        ent_probs: list[float] = []
        skipped_no_premise = 0
        t0 = time.perf_counter()
        for r in per_q:
            if r.get("in_calibration"):
                continue
            if r.get("refused"):
                continue
            pred = (r.get("llm_pred") or "").strip()
            if not pred:
                continue
            premise = None
            if r.get("contexts"):
                premise = "\n".join(r["contexts"])
            elif recompute_contexts and r.get("qid") in questions_by_qid:
                q = questions_by_qid[r["qid"]]
                out_rt = indexed[sys_name].answer(q["question"])
                chunks = out_rt["ranked"][:top_k]
                premise = "\n".join(c.text for c in chunks)
            if not premise:
                skipped_no_premise += 1
                continue
            p = entail_prob(premise, pred, model_name)
            ent_probs.append(p)
            scored.append({
                "qid": r.get("qid"),
                "answerable": r.get("answerable"),
                "pred": pred,
                "cite_doc": r.get("cite_doc"),
                "cite_para": r.get("cite_para"),
                "entail_prob": p,
                "supported": p >= threshold,
            })
            if len(scored) % 50 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  [{sys_name}] scored {len(scored)} in {elapsed:.0f}s", flush=True)
        mean, lo, hi = bootstrap_mean_ci(ent_probs)
        supp = [1.0 if s["supported"] else 0.0 for s in scored]
        s_mean, s_lo, s_hi = bootstrap_mean_ci(supp)
        out["systems"][sys_name] = {
            "n_scored": len(scored),
            "n_skipped_no_premise": skipped_no_premise,
            "entail_prob_mean": mean,
            "entail_prob_ci95": [lo, hi],
            "supported_rate_mean": s_mean,
            "supported_rate_ci95": [s_lo, s_hi],
            "per_question": scored,
        }
        print(f"  {sys_name}: n={len(scored)}  entail={mean:.3f} [{lo:.3f},{hi:.3f}]  "
              f"supported={s_mean:.3f} [{s_lo:.3f},{s_hi:.3f}]"
              + (f"  (skipped {skipped_no_premise} for missing premise)" if skipped_no_premise else ""),
              flush=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                    help="Paths to llm_*.json files (FLAN-T5 or frontier)")
    ap.add_argument("--out-dir", default=str(RESULTS),
                    help="Output directory for nli_faithfulness_*.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--nli-model", default=DEFAULT_NLI_MODEL,
                    help=f"HF model id (default: {DEFAULT_NLI_MODEL})")
    ap.add_argument("--no-recompute-contexts", action="store_true",
                    help="Do not re-run retrieval to recover contexts; skip records that lack them.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in args.sources:
        sp = Path(src)
        if not sp.exists():
            print(f"skip (missing): {sp}", flush=True)
            continue
        op = out_dir / f"nli_faithfulness_{sp.stem}.json"
        score_dataset(sp, op,
                      threshold=args.threshold,
                      model_name=args.nli_model,
                      recompute_contexts=not args.no_recompute_contexts)


if __name__ == "__main__":
    main()

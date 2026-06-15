"""Ablations:

A) Chunking ablation on the AutoRAG pipeline (dense + BM25 + reranker, calibrated
   abstain) with chunk modes: paragraph, fixed_120, fixed_300, fixed_500,
   semantic, adaptive.
B) Component ablation: remove one AutoRAG component at a time
   (no_bm25, no_rerank, no_query_rewrite, no_abstain, no_adaptive).

For fair comparison we reuse the 40/60 calibration/evaluation split written
by run_main.py.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import RESULTS, save_json
from data_loaders import load_enterprise, load_squad
from run_main import calibrate_abstain, metrics_from_output
from systems import AutoRAG


def _calibrated_run(corpus, questions, overrides: dict, name: str):
    sys = AutoRAG(**overrides)
    sys.use_abstain = False
    sys.index(corpus)
    rng = np.random.default_rng(20260511)
    indices = np.arange(len(questions))
    rng.shuffle(indices)
    cal_size = int(0.4 * len(questions))
    cal_set = set(indices[:cal_size].tolist())

    rows = []
    outputs = []
    t0 = time.perf_counter()
    for i, q in enumerate(questions):
        out = sys.answer(q["question"])
        m = metrics_from_output(q, out)
        row = {
            "qid": q["qid"],
            "domain": q.get("domain"),
            "answerable": q["answerable"],
            "in_calibration": i in cal_set,
            "top_score": out["top_score"],
            "margin": out.get("margin", 0.0),
            "prediction": out["prediction"],
            "cite_doc": out["cite_doc"],
            "cite_para": out["cite_para"],
            **m,
        }
        rows.append(row)
        outputs.append((q, out))
    elapsed = time.perf_counter() - t0

    # Calibrate abstain if requested
    if overrides.get("use_abstain", True) and overrides.get("use_reranker", True):
        cal = [r for r in rows if r["in_calibration"]]
        ba, tau_s, tau_m = calibrate_abstain(cal)
        updated = []
        for r, (q, out) in zip(rows, outputs):
            abstain = r["top_score"] < tau_s or r["margin"] < tau_m
            new_out = dict(out)
            if abstain:
                new_out["prediction"] = ""
                new_out["cite_doc"] = None
                new_out["cite_para"] = None
            m2 = metrics_from_output(q, new_out)
            nr = dict(r)
            nr.update(m2)
            nr["prediction"] = new_out["prediction"]
            nr["cite_doc"] = new_out["cite_doc"]
            nr["cite_para"] = new_out["cite_para"]
            nr["abstained"] = abstain
            updated.append(nr)
        rows = updated
        calib = {"balanced_accuracy": ba, "tau_score": tau_s, "tau_margin": tau_m}
    else:
        calib = None

    return {
        "name": name,
        "index_time": sys._index_time,
        "n_chunks": len(sys.chunks),
        "total_query_time": elapsed,
        "calibration": calib,
        "per_question": rows,
    }


def chunking_ablation(corpus, questions, dataset_name: str):
    variants = [
        ("paragraph", {"chunking_mode": "paragraph"}),
        ("fixed_120", {"chunking_mode": "fixed_120"}),
        ("fixed_300", {"chunking_mode": "fixed_300"}),
        ("fixed_500", {"chunking_mode": "fixed_500"}),
        ("semantic", {"chunking_mode": "semantic"}),
        ("adaptive", {"chunking_mode": "adaptive"}),
    ]
    base = {
        "use_dense": True,
        "use_bm25": True,
        "use_reranker": True,
        "use_query_rewrite": True,
        "use_abstain": True,
    }
    out = {"dataset": dataset_name, "variants": {}}
    for name, ov in variants:
        print(f"[{dataset_name}/chunking] running {name} ...", flush=True)
        cfg = {**base, **ov}
        out["variants"][name] = _calibrated_run(corpus, questions, cfg, name)
    return out


def component_ablation(corpus, questions, dataset_name: str):
    full = {
        "chunking_mode": "adaptive",
        "use_dense": True,
        "use_bm25": True,
        "use_reranker": True,
        "use_query_rewrite": True,
        "use_abstain": True,
    }
    variants = [
        ("full", {}),
        ("no_bm25", {"use_bm25": False}),
        ("no_rerank", {"use_reranker": False, "use_abstain": False}),
        ("no_query_rewrite", {"use_query_rewrite": False}),
        ("no_abstain", {"use_abstain": False}),
        ("no_adaptive_chunk", {"chunking_mode": "paragraph"}),
    ]
    out = {"dataset": dataset_name, "variants": {}}
    for name, ov in variants:
        print(f"[{dataset_name}/component] running {name} ...", flush=True)
        cfg = {**full, **ov}
        out["variants"][name] = _calibrated_run(corpus, questions, cfg, name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad-per-article", type=int, default=10)
    ap.add_argument("--datasets", nargs="+", default=["squad", "enterprise"])
    args = ap.parse_args()

    for ds in args.datasets:
        if ds == "squad":
            corpus, qs = load_squad(n_per_article=args.squad_per_article)
        else:
            corpus, qs = load_enterprise()
        print(f"\n*** Chunking ablation on {ds} ({len(qs)} questions) ***", flush=True)
        chunk_res = chunking_ablation(corpus, qs, ds)
        save_json(chunk_res, RESULTS / f"chunking_{ds}.json")
        print(f"\n*** Component ablation on {ds} ({len(qs)} questions) ***", flush=True)
        comp_res = component_ablation(corpus, qs, ds)
        save_json(comp_res, RESULTS / f"component_{ds}.json")


if __name__ == "__main__":
    main()

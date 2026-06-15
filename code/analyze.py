"""Aggregate all result JSONs into tables and statistical tests.

Outputs:
  results/summary.json    -- per-system + per-domain headline metrics
  results/stats.json      -- paired Wilcoxon, effect sizes, bootstrap CIs
  results/chunking.json   -- chunking-ablation summary
  results/component.json  -- component-ablation summary
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from common import RESULTS, save_json


METRICS_ANS = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@5", "em", "f1", "citation", "faithfulness", "hallucination"]
METRICS_UNANS = ["refusal_correct", "hallucination"]
LATENCY = "latency"


def _avg(rows, key):
    v = [r.get(key, 0.0) for r in rows]
    return float(np.mean(v)) if v else float("nan")


def _ci_bootstrap(values, alpha=0.05, n_boot=2000):
    if not values:
        return (float("nan"), float("nan"))
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(2026)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def summarize_main(path: Path):
    r = json.loads(path.read_text(encoding="utf-8"))
    out = {
        "dataset": r["dataset"],
        "n_questions": r["n_questions"],
        "n_calibration": r["n_calibration"],
        "n_evaluation": r["n_evaluation"],
        "systems": {},
        "by_domain": {},
        "calibration": r.get("autorag_calibration"),
    }
    for sname, info in r["systems"].items():
        rows = [p for p in info["per_question"] if not p["in_calibration"]]
        ans = [p for p in rows if p["answerable"]]
        una = [p for p in rows if not p["answerable"]]
        cited = [p for p in ans if p.get("prediction", "").strip()]
        sys_stats = {
            "n_evaluation_total": len(rows),
            "n_evaluation_answerable": len(ans),
            "n_evaluation_unanswerable": len(una),
        }
        for m in METRICS_ANS:
            vals = [p.get(m, 0.0) for p in ans]
            sys_stats[m] = _avg(ans, m)
            sys_stats[m + "_ci95"] = list(_ci_bootstrap(vals))
        sys_stats["citation_when_answered"] = _avg(cited, "citation") if cited else float("nan")
        sys_stats["citation_when_answered_n"] = len(cited)
        for m in METRICS_UNANS:
            vals = [p.get(m, 0.0) for p in una]
            sys_stats[m + "_unans"] = _avg(una, m)
            sys_stats[m + "_unans_ci95"] = list(_ci_bootstrap(vals))
        # Mean and median latency over all eval rows
        lats = [p.get(LATENCY, 0.0) for p in rows]
        sys_stats["latency_mean_ms"] = 1000 * float(np.mean(lats))
        sys_stats["latency_median_ms"] = 1000 * float(np.median(lats))
        sys_stats["latency_p95_ms"] = 1000 * float(np.percentile(lats, 95))
        sys_stats["index_time_s"] = info["index_time"]
        sys_stats["n_chunks"] = info["n_chunks"]
        out["systems"][sname] = sys_stats

        # Per-domain breakdown
        by_dom = defaultdict(lambda: {"ans": [], "una": []})
        for p in rows:
            d = p.get("domain") or "unknown"
            (by_dom[d]["ans"] if p["answerable"] else by_dom[d]["una"]).append(p)
        for dom, parts in by_dom.items():
            out["by_domain"].setdefault(dom, {})[sname] = {
                "n_ans": len(parts["ans"]),
                "n_una": len(parts["una"]),
                "recall@1": _avg(parts["ans"], "recall@1"),
                "recall@5": _avg(parts["ans"], "recall@5"),
                "mrr@10": _avg(parts["ans"], "mrr@10"),
                "f1": _avg(parts["ans"], "f1"),
                "citation": _avg(parts["ans"], "citation"),
                "refusal_correct_unans": _avg(parts["una"], "refusal_correct"),
                "hallucination_unans": _avg(parts["una"], "hallucination"),
            }
    return out


def _cohens_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def paired_tests(path: Path):
    r = json.loads(path.read_text(encoding="utf-8"))
    out = {"dataset": r["dataset"], "tests": {}}
    # Build per-question matrices keyed by qid
    qid_to_row = {}
    sys_names = list(r["systems"].keys())
    for s in sys_names:
        qid_to_row[s] = {p["qid"]: p for p in r["systems"][s]["per_question"] if not p["in_calibration"]}
    qids_all = list(qid_to_row[sys_names[0]].keys())
    # Pairings: each baseline vs AutoRAG
    metrics_to_test = ["recall@1", "recall@5", "mrr@10", "ndcg@5", "f1", "citation"]
    for baseline in [s for s in sys_names if s != "AutoRAG"]:
        pair_out = {}
        for m in metrics_to_test:
            a, b = [], []
            for qid in qids_all:
                if not qid_to_row["AutoRAG"][qid]["answerable"]:
                    continue
                a.append(qid_to_row["AutoRAG"][qid].get(m, 0.0))
                b.append(qid_to_row[baseline][qid].get(m, 0.0))
            if len(a) < 5 or all(x == y for x, y in zip(a, b)):
                pair_out[m] = {"n": len(a), "skipped": True}
                continue
            try:
                W, p_w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                W, p_w = float("nan"), 1.0
            t, p_t = stats.ttest_rel(a, b)
            d = _cohens_d(a, b)
            pair_out[m] = {
                "n": len(a),
                "autorag_mean": float(np.mean(a)),
                "baseline_mean": float(np.mean(b)),
                "diff_mean": float(np.mean(a) - np.mean(b)),
                "wilcoxon_W": float(W),
                "wilcoxon_p": float(p_w),
                "t_stat": float(t),
                "t_p": float(p_t),
                "cohens_d": d,
            }
        # Refusal accuracy on unanswerable (categorical: McNemar via paired binary)
        a, b = [], []
        for qid in qids_all:
            if qid_to_row["AutoRAG"][qid]["answerable"]:
                continue
            a.append(qid_to_row["AutoRAG"][qid].get("refusal_correct", 0.0))
            b.append(qid_to_row[baseline][qid].get("refusal_correct", 0.0))
        if a and b:
            # McNemar: count discordant pairs
            b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
            b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
            # use scipy's mcnemar
            from scipy.stats import binomtest
            n_disc = b10 + b01
            if n_disc > 0:
                p_mc = binomtest(b10, n_disc, p=0.5).pvalue
            else:
                p_mc = 1.0
            pair_out["refusal_correct_unans"] = {
                "n": len(a),
                "autorag_mean": float(np.mean(a)),
                "baseline_mean": float(np.mean(b)),
                "diff_mean": float(np.mean(a) - np.mean(b)),
                "mcnemar_b10": b10,
                "mcnemar_b01": b01,
                "mcnemar_p": float(p_mc),
            }
        out["tests"][baseline] = pair_out
    return out


def summarize_ablation(path: Path, kind: str):
    r = json.loads(path.read_text(encoding="utf-8"))
    out = {"dataset": r["dataset"], "kind": kind, "variants": {}}
    for vname, info in r["variants"].items():
        rows = [p for p in info["per_question"] if not p["in_calibration"]]
        ans = [p for p in rows if p["answerable"]]
        una = [p for p in rows if not p["answerable"]]
        lats = [p.get(LATENCY, 0.0) for p in rows]
        out["variants"][vname] = {
            "n_chunks": info["n_chunks"],
            "index_time_s": info["index_time"],
            "recall@1": _avg(ans, "recall@1"),
            "recall@5": _avg(ans, "recall@5"),
            "recall@10": _avg(ans, "recall@10"),
            "mrr@10": _avg(ans, "mrr@10"),
            "ndcg@5": _avg(ans, "ndcg@5"),
            "f1": _avg(ans, "f1"),
            "citation": _avg(ans, "citation"),
            "faithfulness": _avg(ans, "faithfulness"),
            "refusal_correct_unans": _avg(una, "refusal_correct"),
            "hallucination_unans": _avg(una, "hallucination"),
            "latency_mean_ms": 1000 * float(np.mean(lats)) if lats else float("nan"),
            "latency_p95_ms": 1000 * float(np.percentile(lats, 95)) if lats else float("nan"),
            "calibration": info.get("calibration"),
        }
    return out


def main():
    summary = {}
    stats_all = {}
    for ds in ("squad", "enterprise"):
        p = RESULTS / f"main_{ds}.json"
        if p.exists():
            summary[ds] = summarize_main(p)
            stats_all[ds] = paired_tests(p)
    save_json(summary, RESULTS / "summary.json")
    save_json(stats_all, RESULTS / "stats.json")

    chunking_all = {}
    component_all = {}
    for ds in ("squad", "enterprise"):
        cp = RESULTS / f"chunking_{ds}.json"
        if cp.exists():
            chunking_all[ds] = summarize_ablation(cp, "chunking")
        cmp = RESULTS / f"component_{ds}.json"
        if cmp.exists():
            component_all[ds] = summarize_ablation(cmp, "component")
    save_json(chunking_all, RESULTS / "chunking.json")
    save_json(component_all, RESULTS / "component.json")
    print("Wrote summary, stats, chunking, component JSONs to results/")


if __name__ == "__main__":
    main()

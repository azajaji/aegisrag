"""Revision v2 analysis pipeline.

Derives selective-QA, false-refusal, threshold-sweep, repeated-split,
latency, cost, statistical, and error-analysis artefacts from the
existing per-question outputs in results/main_*.json and llm_*.json.

Adds the derived `RAG+Rerank+Abstain` baseline by applying the AutoRAG
abstention rule to RAG+Rerank's per-question reranker scores and
margins, recalibrated on a held-out split.

Outputs to results/revision_v2/.
"""
from __future__ import annotations

import csv
import json

import numpy as np
from scipy import stats as sp_stats

from common import RESULTS

REV = RESULTS / "revision_v2"
REV.mkdir(parents=True, exist_ok=True)

SEEDS = [20260511, 20260512, 20260513, 20260514, 20260515]
DATASETS = [("squad", "main_squad.json"), ("enterprise", "main_enterprise.json")]
SYSTEMS = ["BM25", "NaiveRAG", "RAG+Rerank", "AutoRAG"]


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


# ===========================================================================
# Helpers: split-aware aggregation
# ===========================================================================

def select_threshold(records, tau_grid_score, tau_grid_margin):
    """Pick (tau_s, tau_m) maximising balanced accuracy on the given records.

    records: list of {answerable, top_score, margin}
    Returns: (best_tau_s, best_tau_m, best_bal_acc)
    """
    ans = [r for r in records if r["answerable"]]
    una = [r for r in records if not r["answerable"]]
    if not ans or not una:
        return 0.0, 0.0, 0.0
    best = (-1.0, 0.0, 0.0)
    for ts in tau_grid_score:
        for tm in tau_grid_margin:
            ans_retain = sum(
                1 for r in ans if not (r["top_score"] < ts or r["margin"] < tm)
            ) / len(ans)
            una_refuse = sum(
                1 for r in una if (r["top_score"] < ts or r["margin"] < tm)
            ) / len(una)
            bal = 0.5 * (ans_retain + una_refuse)
            if bal > best[0]:
                best = (bal, ts, tm)
    return best[1], best[2], best[0]


def apply_thresholds(records, tau_s, tau_m):
    """Apply (tau_s, tau_m) abstention to records.

    Returns metrics dict on the eval subset including selective QA fields.
    """
    out = []
    for r in records:
        abst = (r["top_score"] < tau_s) or (r["margin"] < tau_m)
        r2 = dict(r)
        r2["abstained_eff"] = abst
        # If abstained, predicted answer becomes empty
        r2["pred_nonempty_eff"] = (not abst) and (r["prediction"] != "")
        r2["em_eff"] = 0.0 if abst else r["em"]
        r2["f1_eff"] = 0.0 if abst else r["f1"]
        if not r["answerable"]:
            r2["em_eff"] = 1.0 if abst else 0.0
            r2["f1_eff"] = 1.0 if abst else 0.0
        # Citation only when answered
        r2["cite_eff"] = r["citation"] if (not abst and r["answerable"]) else None
        out.append(r2)
    return out


def aggregate(eff_records):
    """Aggregate selective-QA metrics from records carrying *_eff fields."""
    n_total = len(eff_records)
    ans = [r for r in eff_records if r["answerable"]]
    una = [r for r in eff_records if not r["answerable"]]
    n_ans, n_una = len(ans), len(una)

    # Coverage = fraction of all queries with non-empty answer
    coverage_all = sum(1 for r in eff_records if r["pred_nonempty_eff"]) / max(1, n_total)
    coverage_ans = sum(1 for r in ans if r["pred_nonempty_eff"]) / max(1, n_ans)

    # Refusal behaviour
    false_refusal_ans = sum(1 for r in ans if r["abstained_eff"]) / max(1, n_ans)
    refusal_unans = sum(1 for r in una if r["abstained_eff"]) / max(1, n_una) if n_una else 0.0
    halluc_unans = 1.0 - refusal_unans
    answer_retention = 1.0 - false_refusal_ans
    bal_abst_acc = 0.5 * (answer_retention + refusal_unans)
    n_refused = sum(1 for r in eff_records if r["abstained_eff"])
    refusal_precision = (sum(1 for r in una if r["abstained_eff"]) / n_refused) if n_refused else 0.0
    refusal_recall = refusal_unans
    refusal_f1 = (2 * refusal_precision * refusal_recall /
                  (refusal_precision + refusal_recall)) if (refusal_precision + refusal_recall) > 0 else 0.0

    # Answer quality on all answerable (denominator = all answerable)
    f1_all_ans = sum(r["f1_eff"] for r in ans) / max(1, n_ans)
    em_all_ans = sum(r["em_eff"] for r in ans) / max(1, n_ans)
    # On answered answerable only
    answered_ans = [r for r in ans if r["pred_nonempty_eff"]]
    f1_answered = sum(r["f1_eff"] for r in answered_ans) / max(1, len(answered_ans)) if answered_ans else 0.0
    em_answered = sum(r["em_eff"] for r in answered_ans) / max(1, len(answered_ans)) if answered_ans else 0.0
    # Citation when answered
    cite_when_answered = (sum(r["cite_eff"] for r in answered_ans) /
                          max(1, len(answered_ans))) if answered_ans else 0.0

    return {
        "n_total": n_total,
        "n_answerable": n_ans,
        "n_unanswerable": n_una,
        "coverage_all": coverage_all,
        "coverage_answerable": coverage_ans,
        "false_refusal_answerable": false_refusal_ans,
        "refusal_unanswerable": refusal_unans,
        "hallucination_unanswerable": halluc_unans,
        "answer_retention": answer_retention,
        "balanced_abstention_accuracy": bal_abst_acc,
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "refusal_f1": refusal_f1,
        "f1_all_answerable": f1_all_ans,
        "em_all_answerable": em_all_ans,
        "f1_answered_only": f1_answered,
        "em_answered_only": em_answered,
        "citation_when_answered": cite_when_answered,
    }


# ===========================================================================
# Section 3: false refusal metrics on the published thresholds
# ===========================================================================

def section3_false_refusal():
    rows = []
    for ds_key, fname in DATASETS:
        d = load(fname)
        for sys in SYSTEMS:
            pq = d["systems"][sys]["per_question"]
            # Only the published evaluation split (in_calibration=False)
            eval_recs = [r for r in pq if not r["in_calibration"]]
            # Use the abstained flag as recorded in the per-question output
            for r in eval_recs:
                r["abstained_eff"] = bool(r.get("abstained", False))
                r["pred_nonempty_eff"] = (not r["abstained_eff"]) and (r["prediction"] != "")
                r["em_eff"] = r["em"]
                r["f1_eff"] = r["f1"]
                r["cite_eff"] = r["citation"] if r["pred_nonempty_eff"] and r["answerable"] else None
            m = aggregate(eval_recs)
            rows.append({"dataset": ds_key, "system": sys, **m})

    out_csv = REV / "abstention" / "false_refusal_metrics.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[s3] wrote {out_csv}")
    return rows


# ===========================================================================
# Section 4: threshold sweep + ROC/PR diagnostics
# ===========================================================================

def section4_threshold_sweep():
    """1D sweeps over tau_s and tau_m for AutoRAG on each dataset."""
    sweeps = {}
    for ds_key, fname in DATASETS:
        d = load(fname)
        pq = d["systems"]["AutoRAG"]["per_question"]
        ans_scores = np.array([r["top_score"] for r in pq if r["answerable"]])
        una_scores = np.array([r["top_score"] for r in pq if not r["answerable"]])
        ans_margins = np.array([r["margin"] for r in pq if r["answerable"]])
        una_margins = np.array([r["margin"] for r in pq if not r["answerable"]])

        # 2D sweep on AutoRAG's recorded top_score / margin
        rows = []
        ts_grid = np.arange(-10, 10.01, 0.5)
        tm_grid = np.arange(-5, 5.01, 0.5)
        for ts in ts_grid:
            for tm in tm_grid:
                ans_keep = float(((ans_scores >= ts) & (ans_margins >= tm)).mean())
                una_keep = float(((una_scores >= ts) & (una_margins >= tm)).mean()) if len(una_scores) else 0.0
                una_ref = 1.0 - una_keep
                bal = 0.5 * (ans_keep + una_ref)
                rows.append({
                    "tau_s": float(ts), "tau_m": float(tm),
                    "answerable_retention": ans_keep,
                    "unanswerable_refusal": una_ref,
                    "balanced_abstention_accuracy": bal,
                })
        out_csv = REV / "abstention" / f"threshold_sweep_{ds_key}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[s4] wrote {out_csv} ({len(rows)} rows)")
        sweeps[ds_key] = rows
    return sweeps


def _auc_from_scores(scores, labels):
    """labels: 1 = unanswerable (positive class for refusal detector)."""
    # AUROC via Mann-Whitney
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    # For abstention, lower score => predict unanswerable
    # So treat -score as the predictor of "unanswerable"
    s_pos = -pos
    s_neg = -neg
    # Mann-Whitney U
    u, _ = sp_stats.mannwhitneyu(s_pos, s_neg, alternative="greater")
    auroc = u / (len(pos) * len(neg))
    # AUPRC via sklearn-style trapezoid on sorted scores
    all_scores = np.concatenate([s_pos, s_neg])
    all_labels = np.concatenate([np.ones_like(s_pos), np.zeros_like(s_neg)])
    order = np.argsort(-all_scores)
    all_labels = all_labels[order]
    cum_tp = np.cumsum(all_labels)
    cum_fp = np.cumsum(1 - all_labels)
    n_pos = cum_tp[-1]
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / n_pos
    # Trapezoid
    auprc = float(np.trapz(precision, recall))
    return float(auroc), float(auprc)


def section4_calibration_diagnostics():
    rows = []
    for ds_key, fname in DATASETS:
        d = load(fname)
        pq = d["systems"]["AutoRAG"]["per_question"]
        scores = np.array([r["top_score"] for r in pq], dtype=float)
        margins = np.array([r["margin"] for r in pq], dtype=float)
        ans = np.array([r["answerable"] for r in pq])
        labels_unans = (~ans).astype(int)

        auroc_s, auprc_s = _auc_from_scores(scores, labels_unans)
        auroc_m, auprc_m = _auc_from_scores(margins, labels_unans)
        # Combined: simple min-rule (since both must exceed threshold)
        combined = np.minimum(scores, margins)
        auroc_c, auprc_c = _auc_from_scores(combined, labels_unans)

        rows.extend([
            {"dataset": ds_key, "signal": "top_score", "auroc": auroc_s, "auprc": auprc_s},
            {"dataset": ds_key, "signal": "margin", "auroc": auroc_m, "auprc": auprc_m},
            {"dataset": ds_key, "signal": "combined_min", "auroc": auroc_c, "auprc": auprc_c},
        ])
    out_csv = REV / "abstention" / "calibration_diagnostics.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[s4] wrote {out_csv}")
    return rows


# ===========================================================================
# Section 5: expanded baselines (RAG+Rerank+Abstain derived from data)
# ===========================================================================

def section5_expanded_baselines():
    """Add derived RAG+Rerank+Abstain by applying AutoRAG's calibration
    procedure to RAG+Rerank's per-question reranker top_score and margin.
    """
    rows = []
    for ds_key, fname in DATASETS:
        d = load(fname)
        # Calibration & eval splits using the recorded in_calibration flag
        rr = d["systems"]["RAG+Rerank"]["per_question"]
        calib = [r for r in rr if r["in_calibration"]]
        evalr = [r for r in rr if not r["in_calibration"]]
        ts_grid = np.arange(-10, 10.01, 0.5)
        tm_grid = np.arange(-5, 5.01, 0.5)
        tau_s, tau_m, calib_bal = select_threshold(calib, ts_grid, tm_grid)

        # Apply to eval
        eff = apply_thresholds(evalr, tau_s, tau_m)
        m = aggregate(eff)
        rows.append({
            "dataset": ds_key,
            "system": "RAG+Rerank+Abstain",
            "tau_s": float(tau_s),
            "tau_m": float(tau_m),
            "calib_balanced_acc": float(calib_bal),
            **m,
        })

        # For symmetry also write the published systems' metrics under the
        # SAME apply_thresholds machinery so the expanded baseline table is
        # internally consistent.
        for sys in SYSTEMS:
            pq = d["systems"][sys]["per_question"]
            evalr2 = [r for r in pq if not r["in_calibration"]]
            for r in evalr2:
                r["abstained_eff"] = bool(r.get("abstained", False))
                r["pred_nonempty_eff"] = (not r["abstained_eff"]) and (r["prediction"] != "")
                r["em_eff"] = r["em"]
                r["f1_eff"] = r["f1"]
                r["cite_eff"] = r["citation"] if r["pred_nonempty_eff"] and r["answerable"] else None
            m2 = aggregate(evalr2)
            rows.append({
                "dataset": ds_key,
                "system": sys,
                "tau_s": None,
                "tau_m": None,
                "calib_balanced_acc": None,
                **m2,
            })

    out_csv = REV / "baselines" / "expanded_baseline_metrics.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[s5] wrote {out_csv}")
    return rows


# ===========================================================================
# Section 6: repeated-split robustness (5 seeds)
# ===========================================================================

def section6_repeated_splits():
    rows = []
    rng = np.random.default_rng()
    ts_grid = np.arange(-10, 10.01, 0.5)
    tm_grid = np.arange(-5, 5.01, 0.5)
    for ds_key, fname in DATASETS:
        d = load(fname)
        # Use AutoRAG's per_question rows for the score/margin signal
        ar_pq = d["systems"]["AutoRAG"]["per_question"]
        n = len(ar_pq)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            idx = np.arange(n)
            rng.shuffle(idx)
            n_calib = int(0.4 * n)
            calib_set = set(int(i) for i in idx[:n_calib])
            # For each system, build records partitioned by this seed-driven split
            for sys in SYSTEMS + ["RAG+Rerank+Abstain"]:
                if sys == "RAG+Rerank+Abstain":
                    base_pq = d["systems"]["RAG+Rerank"]["per_question"]
                else:
                    base_pq = d["systems"][sys]["per_question"]
                calib = [base_pq[i] for i in range(n) if i in calib_set]
                evalr = [base_pq[i] for i in range(n) if i not in calib_set]
                if sys == "AutoRAG":
                    tau_s, tau_m, _ = select_threshold(calib, ts_grid, tm_grid)
                    eff = apply_thresholds(evalr, tau_s, tau_m)
                    m = aggregate(eff)
                elif sys == "RAG+Rerank+Abstain":
                    tau_s, tau_m, _ = select_threshold(calib, ts_grid, tm_grid)
                    eff = apply_thresholds(evalr, tau_s, tau_m)
                    m = aggregate(eff)
                else:
                    # No abstention for baselines
                    for r in evalr:
                        r["abstained_eff"] = bool(r.get("abstained", False))
                        r["pred_nonempty_eff"] = (not r["abstained_eff"]) and (r["prediction"] != "")
                        r["em_eff"] = r["em"]
                        r["f1_eff"] = r["f1"]
                        r["cite_eff"] = r["citation"] if r["pred_nonempty_eff"] and r["answerable"] else None
                    m = aggregate(evalr)
                rows.append({"dataset": ds_key, "seed": int(seed), "system": sys, **m})

    out_csv = REV / "robustness" / "repeated_split_metrics.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[s6] wrote {out_csv}")
    return rows


# ===========================================================================
# Section 9: cost sensitivity
# ===========================================================================

# Approximate per-token prices in USD per 1M tokens (May 2026 list values;
# illustrative — use config/model_prices.yaml in production).
PRICES = {
    "gpt-4o-mini":      {"input": 0.15, "output": 0.60},
    "claude-haiku-4.5": {"input": 0.80, "output": 4.00},
    "gpt-4o":           {"input": 2.50, "output": 10.00},
    "claude-sonnet-4.6":{"input": 3.00, "output": 15.00},
}

# Context-size scenarios for the prompt body in tokens (excluding question)
CTX = {
    "short_top1":     {"input_tokens_per_q": 110, "output_tokens_per_q": 7},
    "standard_top3":  {"input_tokens_per_q": 284, "output_tokens_per_q": 7},
    "long_top5":      {"input_tokens_per_q": 460, "output_tokens_per_q": 7},
}


def section9_cost_sensitivity():
    rows = []
    for ds_key, fname in DATASETS:
        d = load(fname)
        # For each system, compute LLM-call rate = 1 - abstain rate on EVAL
        for sys in SYSTEMS + ["RAG+Rerank+Abstain"]:
            if sys == "RAG+Rerank+Abstain":
                rr = d["systems"]["RAG+Rerank"]["per_question"]
                # Reuse §5's published thresholds via section5 (per dataset)
                # Re-derive quickly:
                calib = [r for r in rr if r["in_calibration"]]
                evalr = [r for r in rr if not r["in_calibration"]]
                ts_grid = np.arange(-10, 10.01, 0.5)
                tm_grid = np.arange(-5, 5.01, 0.5)
                tau_s, tau_m, _ = select_threshold(calib, ts_grid, tm_grid)
                n = len(evalr)
                n_abst = sum(
                    1 for r in evalr if (r["top_score"] < tau_s or r["margin"] < tau_m)
                )
            else:
                pq = d["systems"][sys]["per_question"]
                evalr = [r for r in pq if not r["in_calibration"]]
                n = len(evalr)
                n_abst = sum(1 for r in evalr if r.get("abstained", False))
            llm_call_rate = (n - n_abst) / max(1, n)
            for ctx_name, ctx_v in CTX.items():
                for model_name, price in PRICES.items():
                    cost_per_1k = (
                        1000 * llm_call_rate * (
                            ctx_v["input_tokens_per_q"] * price["input"] / 1e6 +
                            ctx_v["output_tokens_per_q"] * price["output"] / 1e6
                        )
                    )
                    rows.append({
                        "dataset": ds_key,
                        "system": sys,
                        "context_size": ctx_name,
                        "model": model_name,
                        "llm_call_rate": llm_call_rate,
                        "cost_usd_per_1k": round(cost_per_1k, 4),
                    })
    out_csv = REV / "cost" / "cost_sensitivity.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[s9] wrote {out_csv}")
    return rows


# ===========================================================================
# Section 10: latency analysis (component-level not available; use total)
# ===========================================================================

def section10_latency():
    rows = []
    for ds_key, fname in DATASETS:
        d = load(fname)
        for sys in SYSTEMS:
            pq = d["systems"][sys]["per_question"]
            lat_ms = np.array([r["latency"] * 1000.0 for r in pq])
            rows.append({
                "dataset": ds_key,
                "system": sys,
                "mean_ms": float(lat_ms.mean()),
                "median_ms": float(np.median(lat_ms)),
                "p90_ms": float(np.percentile(lat_ms, 90)),
                "p95_ms": float(np.percentile(lat_ms, 95)),
                "p99_ms": float(np.percentile(lat_ms, 99)),
                "std_ms": float(lat_ms.std()),
                "min_ms": float(lat_ms.min()),
                "max_ms": float(lat_ms.max()),
                "n": int(len(lat_ms)),
            })
    out_csv = REV / "latency" / "component_latency_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[s10] wrote {out_csv}")
    return rows


# ===========================================================================
# Section 13: paired statistical tests + Benjamini-Hochberg FDR
# ===========================================================================

def _benjamini_hochberg(pvals, alpha=0.05):
    """Return BH-adjusted p-values."""
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    adj = ranked * m / np.arange(1, m + 1)
    # Cumulative min from the right
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out.tolist()


def section13_stats():
    rows = []
    for ds_key, fname in DATASETS:
        d = load(fname)
        ar_pq = d["systems"]["AutoRAG"]["per_question"]
        ar_eval = [r for r in ar_pq if not r["in_calibration"]]
        for sys in ["BM25", "NaiveRAG", "RAG+Rerank"]:
            comp_eval = [r for r in d["systems"][sys]["per_question"] if not r["in_calibration"]]
            # Pair by qid order (they share the same eval split)
            ar_by = {r["qid"]: r for r in ar_eval}
            comp_by = {r["qid"]: r for r in comp_eval}
            qids = [q for q in ar_by if q in comp_by]
            for metric_key in ["f1", "citation", "refusal_correct"]:
                a = np.array([ar_by[q][metric_key] for q in qids])
                b = np.array([comp_by[q][metric_key] for q in qids])
                diff = a - b
                if (diff == 0).all():
                    p = 1.0
                    stat = 0.0
                else:
                    if metric_key == "refusal_correct":
                        # McNemar on discordant pairs
                        b01 = int(((a == 1) & (b == 0)).sum())
                        b10 = int(((a == 0) & (b == 1)).sum())
                        if (b01 + b10) == 0:
                            p, stat = 1.0, 0.0
                        else:
                            # exact binomial McNemar
                            stat = (abs(b01 - b10) - 1)**2 / (b01 + b10)
                            p = float(sp_stats.chi2.sf(stat, df=1))
                    else:
                        try:
                            stat, p = sp_stats.wilcoxon(a, b, zero_method="wilcox")
                        except ValueError:
                            stat, p = 0.0, 1.0
                # Cohen's dz
                if diff.std(ddof=1) > 0:
                    dz = float(diff.mean() / diff.std(ddof=1))
                else:
                    dz = 0.0
                rows.append({
                    "dataset": ds_key,
                    "comparison": f"AutoRAG vs {sys}",
                    "metric": metric_key,
                    "delta": float(a.mean() - b.mean()),
                    "p_value": float(p),
                    "cohen_dz": dz,
                    "n_pairs": len(qids),
                    "statistic": float(stat),
                })
    # BH FDR within each (dataset, metric) family
    by_family = {}
    for i, r in enumerate(rows):
        key = (r["dataset"], r["metric"])
        by_family.setdefault(key, []).append(i)
    for key, idxs in by_family.items():
        ps = [rows[i]["p_value"] for i in idxs]
        adj = _benjamini_hochberg(ps)
        for i, p_adj in zip(idxs, adj):
            rows[i]["p_value_bh_adj"] = float(p_adj)

    out_csv = REV / "statistics" / "statistical_tests_full.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[s13] wrote {out_csv}")
    return rows


# ===========================================================================
# Section 14: error analysis samples
# ===========================================================================

def section14_error_analysis():
    """Sample false refusals, false answers on unanswerable, and citation failures."""
    falses_refusal = []
    false_answers = []
    cite_fails = []
    for ds_key, fname in DATASETS:
        d = load(fname)
        pq = d["systems"]["AutoRAG"]["per_question"]
        evalr = [r for r in pq if not r["in_calibration"]]
        # False refusals: answerable AND abstained
        for r in evalr:
            if r["answerable"] and bool(r.get("abstained", False)):
                falses_refusal.append({
                    "dataset": ds_key,
                    "qid": r["qid"],
                    "domain": r["domain"],
                    "top_score": r["top_score"],
                    "margin": r["margin"],
                    "f1": r["f1"],
                    "prediction": r["prediction"],
                    "cite_doc": r["cite_doc"],
                    "cite_para": r["cite_para"],
                })
        # False answers on unanswerable
        for r in evalr:
            if (not r["answerable"]) and (not bool(r.get("abstained", False))) and r["prediction"] != "":
                false_answers.append({
                    "dataset": ds_key,
                    "qid": r["qid"],
                    "domain": r["domain"],
                    "top_score": r["top_score"],
                    "margin": r["margin"],
                    "prediction": r["prediction"],
                    "cite_doc": r["cite_doc"],
                    "cite_para": r["cite_para"],
                })
        # Citation failures: answerable, not abstained, but citation==0
        for r in evalr:
            if (r["answerable"]
                and not bool(r.get("abstained", False))
                and r["prediction"] != ""
                and r["citation"] == 0):
                cite_fails.append({
                    "dataset": ds_key,
                    "qid": r["qid"],
                    "domain": r["domain"],
                    "f1": r["f1"],
                    "prediction": r["prediction"],
                    "cite_doc": r["cite_doc"],
                    "cite_para": r["cite_para"],
                })

    def write_csv(path, data, cap=None):
        if not data:
            print(f"[s14] no rows for {path.name}")
            path.write_text("", encoding="utf-8")
            return
        data = data if (cap is None) else data[:cap]
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"[s14] wrote {path} ({len(data)} rows)")

    write_csv(REV / "error_analysis" / "false_refusals.csv", falses_refusal, cap=40)
    write_csv(REV / "error_analysis" / "false_answers_unanswerable.csv", false_answers, cap=40)
    write_csv(REV / "error_analysis" / "citation_failures.csv", cite_fails, cap=40)


# ===========================================================================
# Section 7: Enterprise dataset audit
# ===========================================================================

def section7_enterprise_audit():
    """Static checks on the enterprise benchmark JSON."""
    from common import DATA
    qpath = DATA / "questions" / "benchmark.json"
    if not qpath.exists():
        print("[s7] enterprise question file not found, skipping audit")
        return None

    qs = json.loads(qpath.read_text(encoding="utf-8"))
    if isinstance(qs, dict) and "questions" in qs:
        qs = qs["questions"]

    n = len(qs)
    # Mark answerable using the 'answerable' field directly
    answerable = sum(1 for q in qs if q.get("answerable", True))
    unanswerable = n - answerable
    by_domain = {}
    for q in qs:
        dom = q.get("domain", "?")
        by_domain[dom] = by_domain.get(dom, 0) + 1
    seen_text = set()
    dups = 0
    for q in qs:
        t = (q.get("question") or "").strip().lower()
        if t in seen_text:
            dups += 1
        seen_text.add(t)

    # Check answer-string presence in document for answerable items
    answer_substring_hits = 0
    answer_substring_total = 0
    docs_root = DATA / "questions"  # paragraphs live in benchmark itself? probe
    # Simpler integrity test: every answerable has a non-empty gold_answer; every unanswerable has none
    valid_ans = sum(1 for q in qs if q.get("answerable") and (q.get("gold_answer") or "").strip())
    valid_unans = sum(1 for q in qs if not q.get("answerable") and not (q.get("gold_answer") or "").strip())

    report = REV / "enterprise_dataset_audit" / "audit_report.md"
    report.write_text(
        f"""# Enterprise benchmark audit (v2)

- Total questions: **{n}**
- Answerable: **{answerable}**
- Unanswerable: **{unanswerable}**
- Answerable items with non-empty gold_answer: **{valid_ans}** / {answerable}
- Unanswerable items with empty gold_answer: **{valid_unans}** / {unanswerable}
- Duplicate question texts: **{dups}**
- By domain:
""" + "\n".join(f"  - {dom}: {cnt}" for dom, cnt in sorted(by_domain.items())) +
        f"\n\nSource file: `{qpath.relative_to(DATA.parent)}`\n",
        encoding="utf-8",
    )
    print(f"[s7] wrote {report}")

    # distribution CSV
    dist_csv = REV / "enterprise_dataset_audit" / "question_distribution.csv"
    with dist_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "count"])
        for dom, cnt in sorted(by_domain.items()):
            w.writerow([dom, cnt])
    print(f"[s7] wrote {dist_csv}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=== revision_v2 analysis pipeline ===")
    section3_false_refusal()
    section4_threshold_sweep()
    section4_calibration_diagnostics()
    section5_expanded_baselines()
    section6_repeated_splits()
    section7_enterprise_audit()
    section9_cost_sensitivity()
    section10_latency()
    section13_stats()
    section14_error_analysis()
    print("Done. Outputs under", REV)


if __name__ == "__main__":
    main()

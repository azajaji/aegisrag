"""Calibration ablation + cross-corpus calibration transfer.

For each dataset, the AutoRAG per-question records in main_*.json carry
`top_score` and `margin` per query alongside `answerable` and
`in_calibration`. We use those to re-compute the abstention decision
under different threshold pairs without re-running retrieval:

(C) Calibration ablation: report selective-QA metrics under
    (a) raw zero threshold tau_s=0, tau_m=0 (no calibration),
    (b) calibrated threshold (the one from run_main.py), and
    (c) oracle threshold maximising eval-set balanced accuracy
        (upper bound; for diagnostic only, not for headline use).

(D) Cross-corpus calibration transfer: take the calibrated thresholds
    from one dataset and apply them to the OTHER dataset's eval split.
    This tests whether the calibration is dataset-specific.

Both analyses operate on the eval split only and write
results/calibration_analysis.json.
"""
from __future__ import annotations

import json

import numpy as np

from common import RESULTS


def load_autorag_records(dataset: str) -> list[dict]:
    path = RESULTS / f"main_{dataset}.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["systems"]["AutoRAG"]["per_question"]


def calibrated_thresholds(dataset: str) -> tuple[float, float]:
    """Read calibrated tau_s, tau_m from summary.json (run_main.py output)."""
    s = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    cal = s[dataset]["calibration"]
    return float(cal["tau_score"]), float(cal["tau_margin"])


def selective_metrics(records: list[dict], tau_s: float, tau_m: float) -> dict:
    """Apply (tau_s, tau_m) to AutoRAG records (eval split only) and
    compute the selective-QA metrics shown in tab:selective."""
    ev = [r for r in records if not r.get("in_calibration")]
    n_total = len(ev)
    n_ans = sum(1 for r in ev if r["answerable"])
    n_una = sum(1 for r in ev if not r["answerable"])
    coverage = 0
    false_refusal = 0
    refuse_unans = 0
    halluc_unans = 0
    cites_ok = 0
    n_answered_all = 0
    n_answered_ans = 0          # answerable AND answered (citation/F1 denom)
    f1_answered_ans = []
    for r in ev:
        abstained = (r["top_score"] < tau_s) or (r["margin"] < tau_m)
        answered = not abstained
        if answered:
            n_answered_all += 1
            if r["answerable"]:
                n_answered_ans += 1
                if r.get("citation", 0):
                    cites_ok += 1
                f1_answered_ans.append(r.get("f1", 0.0))
        if r["answerable"]:
            if not answered:
                false_refusal += 1
        else:
            if not answered:
                refuse_unans += 1
            else:
                halluc_unans += 1
        coverage += int(answered)
    return {
        "tau_s": tau_s,
        "tau_m": tau_m,
        "n_total": n_total,
        "coverage_all": coverage / max(1, n_total),
        "false_refusal_answerable": false_refusal / max(1, n_ans),
        "refusal_unanswerable": refuse_unans / max(1, n_una),
        "hallucination_unanswerable": halluc_unans / max(1, n_una),
        # citation@answerable-answered (matches headline tab:selective convention)
        "citation_when_answered": cites_ok / max(1, n_answered_ans) if n_answered_ans else 0.0,
        "f1_answered_only": float(np.mean(f1_answered_ans)) if f1_answered_ans else 0.0,
        "n_answered_all": n_answered_all,
        "n_answered_answerable": n_answered_ans,
    }


def best_threshold_oracle(records: list[dict]) -> tuple[float, float, float]:
    """Eval-set oracle: pick (tau_s, tau_m) maximising balanced accuracy
    on the EVALUATION split. Diagnostic upper bound only."""
    ev = [r for r in records if not r.get("in_calibration")]
    best = (-1.0, 0.0, 0.0)
    for ts in np.arange(-5.0, 10.5, 0.5):
        for tm in np.arange(0.0, 5.5, 0.5):
            tp_ans = 0
            tn_una = 0
            n_ans = 0
            n_una = 0
            for r in ev:
                abstained = (r["top_score"] < ts) or (r["margin"] < tm)
                answered = not abstained
                if r["answerable"]:
                    n_ans += 1
                    if answered:
                        tp_ans += 1
                else:
                    n_una += 1
                    if not answered:
                        tn_una += 1
            ba = 0.5 * (tp_ans / max(1, n_ans)) + 0.5 * (tn_una / max(1, n_una))
            if ba > best[0]:
                best = (ba, float(ts), float(tm))
    return best


def main():
    out = {"datasets": {}}
    autorag_records = {
        ds: load_autorag_records(ds) for ds in ["squad", "enterprise"]
    }
    calibrated = {ds: calibrated_thresholds(ds) for ds in ["squad", "enterprise"]}

    # (C) Calibration ablation per dataset.
    for ds in ["squad", "enterprise"]:
        r = autorag_records[ds]
        tau_s_cal, tau_m_cal = calibrated[ds]
        ba_oracle, ts_oracle, tm_oracle = best_threshold_oracle(r)
        out["datasets"][ds] = {
            "calibrated_thresholds": {"tau_s": tau_s_cal, "tau_m": tau_m_cal},
            "oracle_thresholds_eval_set": {
                "tau_s": ts_oracle, "tau_m": tm_oracle,
                "balanced_accuracy": ba_oracle,
            },
            "variants": {
                "raw_zero": selective_metrics(r, 0.0, 0.0),
                "calibrated": selective_metrics(r, tau_s_cal, tau_m_cal),
                "oracle_eval": selective_metrics(r, ts_oracle, tm_oracle),
            },
        }

    # (D) Cross-corpus calibration transfer.
    out["cross_corpus_transfer"] = {}
    for source_ds in ["squad", "enterprise"]:
        ts, tm = calibrated[source_ds]
        for target_ds in ["squad", "enterprise"]:
            key = f"cal_on_{source_ds}__eval_on_{target_ds}"
            metrics = selective_metrics(autorag_records[target_ds], ts, tm)
            out["cross_corpus_transfer"][key] = metrics

    out_path = RESULTS / "calibration_analysis.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Print compact summary
    print("\n=== (C) Calibration ablation ===")
    for ds, d in out["datasets"].items():
        print(f"\n[{ds}]  calibrated=(tau_s={d['calibrated_thresholds']['tau_s']}, "
              f"tau_m={d['calibrated_thresholds']['tau_m']})  "
              f"oracle_eval=({d['oracle_thresholds_eval_set']['tau_s']}, "
              f"{d['oracle_thresholds_eval_set']['tau_m']}, "
              f"BA={d['oracle_thresholds_eval_set']['balanced_accuracy']:.3f})")
        header = f"  {'Variant':14} {'Cov':>6} {'FRR':>6} {'Refuse':>7} {'Halluc':>7} {'F1@ans':>7} {'Cite':>6}"
        print(header)
        for vk, vm in d["variants"].items():
            print(f"  {vk:14} {vm['coverage_all']:>6.3f} "
                  f"{vm['false_refusal_answerable']:>6.3f} "
                  f"{vm['refusal_unanswerable']:>7.3f} "
                  f"{vm['hallucination_unanswerable']:>7.3f} "
                  f"{vm['f1_answered_only']:>7.3f} "
                  f"{vm['citation_when_answered']:>6.3f}")

    print("\n=== (D) Cross-corpus calibration transfer ===")
    header = f"  {'Calibration':24} {'Cov':>6} {'FRR':>6} {'Refuse':>7} {'Halluc':>7} {'F1@ans':>7} {'Cite':>6}"
    print(header)
    for key, m in out["cross_corpus_transfer"].items():
        print(f"  {key:24} {m['coverage_all']:>6.3f} "
              f"{m['false_refusal_answerable']:>6.3f} "
              f"{m['refusal_unanswerable']:>7.3f} "
              f"{m['hallucination_unanswerable']:>7.3f} "
              f"{m['f1_answered_only']:>7.3f} "
              f"{m['citation_when_answered']:>6.3f}")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

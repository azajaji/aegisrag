"""Three reviewer-facing analyses computed from existing AutoRAG logits:
  (1) Calibration-set size sensitivity:
      Vary the calibration fraction in {5, 10, 20, 40}% and report
      selective-QA metrics on a fixed evaluation split.
  (2) Abstention-signal comparison:
      Compare score-only, margin-only, score-or-margin (current),
      score-and-margin, and dense-cosine threshold rules under
      identical calibration.
  (3) Abstention operating profiles:
      For each safety profile (answer-oriented, balanced, safety-first)
      report the selected (tau_s, tau_m) and resulting metrics.

All three reuse the AutoRAG per_question records in main_squad.json
and main_enterprise.json -- no re-running.

Writes results/extra_analyses.json with structured output for each
section.
"""
from __future__ import annotations

import json

import numpy as np

from common import RESULTS


def load_autorag(ds):
    return json.loads((RESULTS / f"main_{ds}.json").read_text(encoding="utf-8"))["systems"]["AutoRAG"]["per_question"]


def selective_metrics(records, predicate):
    """Apply an abstention predicate (lambda r -> True means abstain)
    and return selective-QA metrics on the evaluation subset."""
    ev = [r for r in records if not r.get("in_calibration")]
    n_total = len(ev)
    n_ans = sum(1 for r in ev if r["answerable"])
    n_una = sum(1 for r in ev if not r["answerable"])
    coverage = false_refusal = refuse_unans = halluc_unans = 0
    cites_ok = 0
    n_answered_ans = 0
    f1_ans = []
    for r in ev:
        abstained = predicate(r)
        answered = not abstained
        if answered:
            if r["answerable"]:
                n_answered_ans += 1
                if r.get("citation", 0):
                    cites_ok += 1
                f1_ans.append(r.get("f1", 0.0))
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
        "n_total": n_total,
        "coverage_all": coverage / max(1, n_total),
        "false_refusal_answerable": false_refusal / max(1, n_ans),
        "refusal_unanswerable": refuse_unans / max(1, n_una),
        "hallucination_unanswerable": halluc_unans / max(1, n_una),
        "f1_answered_only": float(np.mean(f1_ans)) if f1_ans else 0.0,
        "citation_when_answered": cites_ok / max(1, n_answered_ans) if n_answered_ans else 0.0,
    }


def grid_search_thresholds(cal_rows, signal: str, score_grid=None, margin_grid=None):
    """Return (best_tau_s, best_tau_m, best_balanced_accuracy) on the
    given calibration rows under the named abstention signal."""
    score_grid = score_grid if score_grid is not None else np.arange(-5.0, 10.5, 0.5)
    margin_grid = margin_grid if margin_grid is not None else np.arange(0.0, 5.5, 0.5)
    best = (-1.0, 0.0, 0.0)
    for ts in score_grid:
        for tm in margin_grid:
            tp_ans = tn_una = n_ans = n_una = 0
            for r in cal_rows:
                abstained = predicate_for(signal, ts, tm)(r)
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
    return best[1], best[2], best[0]


def predicate_for(signal: str, tau_s: float, tau_m: float):
    """Return a predicate r -> abstain (True/False) under the named
    signal definition. Margin uses tau_m; everything else uses tau_s."""
    if signal == "score_only":
        return lambda r: r["top_score"] < tau_s
    if signal == "margin_only":
        return lambda r: r["margin"] < tau_m
    if signal == "score_or_margin":          # current AutoRAG
        return lambda r: (r["top_score"] < tau_s) or (r["margin"] < tau_m)
    if signal == "score_and_margin":
        return lambda r: (r["top_score"] < tau_s) and (r["margin"] < tau_m)
    raise ValueError(signal)


# ---------------------------------------------------------------------------
# (1) Calibration-set size sensitivity
# ---------------------------------------------------------------------------

def calibration_size_sensitivity(records, fractions=(0.05, 0.10, 0.20, 0.40),
                                 seeds=tuple(range(5))):
    """For each fraction, repeatedly draw a subset of the existing
    calibration split (subsample), pick (tau_s, tau_m) on it, then
    evaluate on the fixed eval split."""
    cal_all = [r for r in records if r.get("in_calibration")]
    out = []
    for frac in fractions:
        per_seed = []
        for seed in seeds:
            rng = np.random.default_rng(20260512 + seed)
            n = max(5, int(frac * len(cal_all)))
            idx = rng.choice(len(cal_all), size=n, replace=False)
            cal_sub = [cal_all[i] for i in idx]
            ts, tm, ba = grid_search_thresholds(cal_sub, "score_or_margin")
            metrics = selective_metrics(records, predicate_for("score_or_margin", ts, tm))
            per_seed.append({"seed": seed, "n_cal": n, "tau_s": ts, "tau_m": tm,
                              "balanced_accuracy": ba, **metrics})
        keys = ["coverage_all", "false_refusal_answerable", "refusal_unanswerable",
                "hallucination_unanswerable", "f1_answered_only", "citation_when_answered"]
        agg = {f"{k}_mean": float(np.mean([p[k] for p in per_seed])) for k in keys}
        agg.update({f"{k}_std": float(np.std([p[k] for p in per_seed], ddof=1)) for k in keys})
        out.append({"fraction": frac, "n_cal_used": per_seed[0]["n_cal"],
                    "per_seed": per_seed, "aggregate": agg})
    return out


# ---------------------------------------------------------------------------
# (2) Abstention-signal comparison
# ---------------------------------------------------------------------------

def abstention_signal_comparison(records):
    """Calibrate each signal on the full calibration split and report
    selective-QA metrics on the eval split."""
    cal = [r for r in records if r.get("in_calibration")]
    out = []
    for signal in ["score_only", "margin_only", "score_or_margin", "score_and_margin"]:
        ts, tm, ba = grid_search_thresholds(cal, signal)
        metrics = selective_metrics(records, predicate_for(signal, ts, tm))
        out.append({"signal": signal, "tau_s": ts, "tau_m": tm,
                    "calibration_balanced_accuracy": ba, **metrics})
    return out


# ---------------------------------------------------------------------------
# (3) Operating profiles
# ---------------------------------------------------------------------------

def operating_profiles(records):
    """Three safety profiles: answer-oriented, balanced, safety-first.
    Each is a different objective for the (tau_s, tau_m) search."""
    cal = [r for r in records if r.get("in_calibration")]
    n_ans_cal = sum(1 for r in cal if r["answerable"])
    n_una_cal = sum(1 for r in cal if not r["answerable"])

    def metrics_at(rows, ts, tm):
        tp_ans = tn_una = n_ans = n_una = 0
        for r in rows:
            abstained = predicate_for("score_or_margin", ts, tm)(r)
            answered = not abstained
            if r["answerable"]:
                n_ans += 1
                if answered:
                    tp_ans += 1
            else:
                n_una += 1
                if not answered:
                    tn_una += 1
        retain = tp_ans / max(1, n_ans)
        refuse = tn_una / max(1, n_una)
        return retain, refuse

    profiles = {}
    # Profile 1: Answer-oriented = maximise answerable retention
    # (subject to refuse >= 0.40 so the profile is non-degenerate)
    best = (-1.0, 0.0, 0.0)
    for ts in np.arange(-5.0, 10.5, 0.5):
        for tm in np.arange(0.0, 5.5, 0.5):
            retain, refuse = metrics_at(cal, ts, tm)
            if refuse < 0.40:
                continue
            if retain > best[0]:
                best = (retain, float(ts), float(tm))
    profiles["answer_oriented"] = {
        "constraint": "refuse>=0.40, maximise answerable retention",
        "tau_s": best[1], "tau_m": best[2],
        **selective_metrics(records, predicate_for("score_or_margin", best[1], best[2])),
    }

    # Profile 2: Balanced = maximise balanced accuracy (= current rule)
    ts, tm, ba = grid_search_thresholds(cal, "score_or_margin")
    profiles["balanced"] = {
        "constraint": "maximise balanced accuracy (default)",
        "tau_s": ts, "tau_m": tm,
        **selective_metrics(records, predicate_for("score_or_margin", ts, tm)),
    }

    # Profile 3: Safety-first = maximise refusal_unanswerable
    # subject to false_refusal <= 0.60
    best = (-1.0, 0.0, 0.0)
    for ts in np.arange(-5.0, 10.5, 0.5):
        for tm in np.arange(0.0, 5.5, 0.5):
            retain, refuse = metrics_at(cal, ts, tm)
            if (1 - retain) > 0.60:    # false-refusal cap
                continue
            if refuse > best[0]:
                best = (refuse, float(ts), float(tm))
    profiles["safety_first"] = {
        "constraint": "false_refusal<=0.60, maximise refusal accuracy",
        "tau_s": best[1], "tau_m": best[2],
        **selective_metrics(records, predicate_for("score_or_margin", best[1], best[2])),
    }
    return profiles


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def main():
    out = {"datasets": {}}
    for ds in ["squad", "enterprise"]:
        recs = load_autorag(ds)
        print(f"\n=== {ds} ===  n={len(recs)}", flush=True)
        out["datasets"][ds] = {
            "calibration_size_sensitivity": calibration_size_sensitivity(recs),
            "abstention_signal_comparison": abstention_signal_comparison(recs),
            "operating_profiles": operating_profiles(recs),
        }

        print("\n[calibration-size]")
        print(f"  {'frac':>6} {'n_cal':>5}  {'Refuse mean':>13} {'FRR mean':>10} {'F1@ans':>8}")
        for row in out["datasets"][ds]["calibration_size_sensitivity"]:
            a = row["aggregate"]
            print(f"  {row['fraction']:>6.2f} {row['n_cal_used']:>5}  "
                  f"{a['refusal_unanswerable_mean']:>13.3f} "
                  f"{a['false_refusal_answerable_mean']:>10.3f} "
                  f"{a['f1_answered_only_mean']:>8.3f}")

        print("\n[abstention-signal]")
        print(f"  {'signal':18} {'tau_s':>6} {'tau_m':>6} {'Refuse':>8} {'FRR':>6} {'Cite':>6} {'F1@ans':>8}")
        for row in out["datasets"][ds]["abstention_signal_comparison"]:
            print(f"  {row['signal']:18} {row['tau_s']:>6.2f} {row['tau_m']:>6.2f} "
                  f"{row['refusal_unanswerable']:>8.3f} "
                  f"{row['false_refusal_answerable']:>6.3f} "
                  f"{row['citation_when_answered']:>6.3f} "
                  f"{row['f1_answered_only']:>8.3f}")

        print("\n[operating-profiles]")
        for k, p in out["datasets"][ds]["operating_profiles"].items():
            print(f"  {k:18} tau=({p['tau_s']:.2f},{p['tau_m']:.2f})  "
                  f"Cov={p['coverage_all']:.3f}  FRR={p['false_refusal_answerable']:.3f}  "
                  f"Refuse={p['refusal_unanswerable']:.3f}  Cite={p['citation_when_answered']:.3f}  "
                  f"F1@ans={p['f1_answered_only']:.3f}")

    (RESULTS / "extra_analyses.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote", RESULTS / "extra_analyses.json")


if __name__ == "__main__":
    main()

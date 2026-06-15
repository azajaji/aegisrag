import csv
import numpy as np
from pathlib import Path

CSV = Path(__file__).resolve().parents[1] / "results" / "revision_v2" / "robustness" / "repeated_split_metrics.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))

systems = ["BM25", "NaiveRAG", "RAG+Rerank", "RAG+Rerank+Abstain", "AutoRAG"]
metrics = [
    ("refusal_unanswerable",     "Refusal(unans)"),
    ("false_refusal_answerable", "FalseRef(ans)"),
    ("f1_answered_only",         "F1@ans"),
    ("citation_when_answered",   "Cite@ans"),
]

for ds in ["squad", "enterprise"]:
    print(f"\n=== {ds} ===")
    hdr = f"{'System':24}" + " ".join(f"{m[1]:>18}" for m in metrics)
    print(hdr)
    for s in systems:
        cells = []
        for mk, _ in metrics:
            vals = [float(r[mk]) for r in rows if r["dataset"] == ds and r["system"] == s]
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))
            cells.append(f"{mean:.3f}+/-{std:.3f}")
        print(f"{s:24}" + " ".join(f"{c:>18}" for c in cells))

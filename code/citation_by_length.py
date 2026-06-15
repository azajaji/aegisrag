"""Citation precision conditional on answer length.

For each system, bucket the answered subset of the SQuAD-v2 and
Enterprise evaluation splits by predicted answer length (tokens) and
report citation precision per bucket. Tests whether AutoRAG's citation
advantage is uniform across short vs. long answers or concentrated in
one length regime.
"""
import json
import re
from collections import defaultdict

import numpy as np

from common import RESULTS

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def length_bucket(s: str) -> str:
    n = len(_TOKEN_RE.findall(s or ""))
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    return "11+"


BUCKETS = ["1-2", "3-5", "6-10", "11+"]


def per_system(ds, sys_name):
    d = json.loads((RESULTS / f"main_{ds}.json").read_text(encoding="utf-8"))
    pq = d["systems"][sys_name]["per_question"]
    by_bucket = defaultdict(list)
    for r in pq:
        if r.get("in_calibration"):
            continue
        if not r["answerable"]:
            continue
        pred = r.get("prediction") or ""
        if not pred.strip():
            continue
        b = length_bucket(pred)
        by_bucket[b].append(int(r.get("citation", 0)))
    out = {}
    for b in BUCKETS:
        vals = by_bucket.get(b, [])
        out[b] = {
            "n": len(vals),
            "citation_precision": float(np.mean(vals)) if vals else 0.0,
        }
    return out


def main():
    table = {}
    for ds in ["squad", "enterprise"]:
        table[ds] = {sys_name: per_system(ds, sys_name)
                     for sys_name in ["BM25", "NaiveRAG", "RAG+Rerank", "AutoRAG"]}
    (RESULTS / "citation_by_length.json").write_text(
        json.dumps(table, indent=2), encoding="utf-8")

    for ds, sysmap in table.items():
        print(f"\n=== {ds} (citation precision by answer length) ===")
        print(f"{'System':12} " + "  ".join(f"{b:>10}" for b in BUCKETS))
        for sys_name in ["BM25", "NaiveRAG", "RAG+Rerank", "AutoRAG"]:
            cells = []
            for b in BUCKETS:
                e = sysmap[sys_name][b]
                if e["n"] == 0:
                    cells.append("    -  ")
                else:
                    cells.append(f"{e['citation_precision']:.3f} (n={e['n']:3d})")
            print(f"{sys_name:12} " + "  ".join(f"{c:>12}" for c in cells))


if __name__ == "__main__":
    main()

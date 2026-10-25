#!/usr/bin/env python3
"""Select the expansion weight w3 on the validation queries (MS MARCO dev, 500 queries).

The rankings saved by run_saqe.py --scores-out are re-fused with w1:w2 = 2:8 and each w3 in
{1, 2, 3, 4, 6, 8, 12}; the value with the highest MRR@10 (relevance >= 1) is selected, and differences
below 0.001 are resolved in favor of the value closest to 4.

Example:
    python scripts/run_saqe.py --dataset msmarco-dev500 --passages passages/qwen3-8b/msmarco-dev500.jsonl \
        --out runs/qwen3-8b/dev500.trec --scores-out runs/qwen3-8b/dev500.scores.jsonl
    python scripts/select_w3.py --scores runs/qwen3-8b/dev500.scores.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from saqe.fusion import rank_by, weighted_rrf  # noqa: E402

GRID, W1, W2, K = [1, 2, 3, 4, 6, 8, 12], 2.0, 8.0, 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="per-query scores of the validation run (JSONL)")
    ap.add_argument("--qrels", default="msmarco-passage-dev-subset", help="Pyserini qrels name")
    ap.add_argument("--with-zero", action="store_true", help="also report w3 = 0 (not a candidate)")
    a = ap.parse_args()

    from pyserini.search import get_qrels
    rel = {str(q): {str(d) for d, r in docs.items() if int(r) >= 1} for q, docs in get_qrels(a.qrels).items()}
    recs = [json.loads(line) for line in open(a.scores)]

    def mrr10(w3):
        tot = 0.0
        for r in recs:
            ids = r["docids"]
            routes = [(rank_by(r["ce_q"], ids), W2), (rank_by(r["bm25"], ids), W1)]
            if r.get("ce_e") and w3 > 0:
                routes.append((rank_by(r["ce_e"], ids), float(w3)))
            f = dict(weighted_rrf(routes, k=K))
            ranked = sorted(ids, key=lambda x: f.get(x, 0.0), reverse=True)
            for i, d in enumerate(ranked[:10], 1):
                if d in rel.get(r["qid"], ()):
                    tot += 1.0 / i
                    break
        return tot / len(recs)

    grid = {w: mrr10(w) for w in GRID}
    best = max(grid.values())
    w3 = min((w for w in GRID if best - grid[w] < 0.001), key=lambda w: (abs(w - 4), w))
    out = {"n_queries": len(recs), "mrr@10": {str(w): round(v, 4) for w, v in grid.items()}, "w3": w3}
    if a.with_zero:
        out["mrr@10_w3=0"] = round(mrr10(0), 4)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

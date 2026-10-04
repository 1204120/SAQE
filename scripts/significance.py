#!/usr/bin/env python3
"""Paired bootstrap test (10,000 samples, seed 42) on per-query nDCG@10 of two runs.

Example:
    python scripts/significance.py --dataset dl19 --run-a runs/dl19.saqe.trec --run-b runs/dl19.bm25rr.trec
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from saqe import DATASETS  # noqa: E402
from saqe.evaluate import paired_bootstrap, per_query_ndcg10  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    ds = DATASETS[a.dataset]
    res = paired_bootstrap(per_query_ndcg10(ds, a.run_a), per_query_ndcg10(ds, a.run_b),
                           n_boot=a.n_boot, seed=a.seed)
    print(f"{a.dataset}: n={res['n']}  mean nDCG@10 difference (A - B) = {res['mean_diff'] * 100:+.2f}  p = {res['p']:.4f}")


if __name__ == "__main__":
    main()

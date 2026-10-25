#!/usr/bin/env python3
"""Compare two systems on per-query nDCG@10 with two-sided paired bootstrap tests (10,000 resamples, seed 42).

Run files are given as patterns with {dataset}. For each dataset, the test is over its queries; for the
average over several datasets, both the pooled and the dataset-stratified tests are reported.

Example:
    python scripts/significance.py --datasets dl19,dl20,scifact,trec-covid,fiqa,dbpedia-entity,trec-news,nfcorpus \
        --run-a "runs/qwen3-8b/{dataset}.saqe.trec" --run-b "runs/{dataset}.bm25rr.trec"
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from saqe import DATASETS, load_queries  # noqa: E402
from saqe.evaluate import per_query  # noqa: E402
from saqe.stats import bootstrap_test  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="comma-separated dataset names")
    ap.add_argument("--run-a", required=True, help="run file pattern of system A, e.g. runs/{dataset}.saqe.trec")
    ap.add_argument("--run-b", required=True, help="run file pattern of system B")
    a = ap.parse_args()

    diffs = []
    for name in a.datasets.split(","):
        ds = DATASETS[name]
        qids = sorted(load_queries(ds))
        pa = per_query(ds, a.run_a.format(dataset=name), qids)
        pb = per_query(ds, a.run_b.format(dataset=name), qids)
        d = np.array([pa[q] - pb[q] for q in qids]) * 100
        diffs.append(d)
        r = bootstrap_test([d])
        print(f"{name:15s} n={r['n']:4d}  A-B = {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]  p = {r['p']:.4f}")
    if len(diffs) > 1:
        for label, r in [("pooled", bootstrap_test([np.concatenate(diffs)])), ("stratified", bootstrap_test(diffs))]:
            print(f"{'average/' + label:15s} n={r['n']:4d}  A-B = {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]"
                  f"  p = {r['p']:.4f}")


if __name__ == "__main__":
    main()

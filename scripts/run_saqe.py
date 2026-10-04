#!/usr/bin/env python3
"""Run SAQE (BM25 + cross-encoder + multi-view rank fusion) and evaluate it.

Example:
    python scripts/run_saqe.py --dataset dl19 \
        --expansions expansions/qwen3-32b/dl19.jsonl --out runs/dl19.saqe.trec

Without --expansions, the script runs the BM25 + cross-encoder baseline with
two-view fusion (BM25+RR in the paper).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from saqe import DATASETS, SAQE, load_expansions, load_queries  # noqa: E402
from saqe.evaluate import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--expansions", default=None, help="JSONL produced by generate_expansions.py")
    ap.add_argument("--out", required=True, help="output TREC run file")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-large")
    ap.add_argument("--k-candidates", type=int, default=1100)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--weights", type=float, nargs=3, default=[2.0, 8.0, 4.0],
                    metavar=("W_SP", "W_Q", "W_INT"), help="fusion weights of r_sp, r_q, r_int")
    a = ap.parse_args()

    ds = DATASETS[a.dataset]
    queries = load_queries(ds)
    expansions = load_expansions(a.expansions) if a.expansions else None
    if expansions is not None:
        missing = [q for q in queries if q not in expansions]
        if missing:
            sys.exit(f"{len(missing)} queries have no expansion in {a.expansions}")

    w_sp, w_q, w_int = a.weights
    model = SAQE(reranker=a.reranker, k_candidates=a.k_candidates, rrf_k=a.rrf_k,
                 w_sp=w_sp, w_q=w_q, w_int=w_int)
    model.run(ds, queries, expansions, a.out)
    metrics = evaluate(ds, a.out)
    print(json.dumps({"dataset": a.dataset, "n_queries": len(queries), **metrics}, indent=2))
    with open(a.out + ".metrics.json", "w") as f:
        json.dump({"dataset": a.dataset, "n_queries": len(queries), "metrics": metrics,
                   "config": vars(a)}, f, indent=2)


if __name__ == "__main__":
    main()

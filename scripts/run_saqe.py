#!/usr/bin/env python3
"""Run SAQE (or BM25+RR) on a dataset and evaluate it.

Example:
    python scripts/run_saqe.py --dataset dl19 --passages passages/qwen3-8b/dl19.jsonl --out runs/qwen3-8b/dl19.saqe.trec

Without --passages, the script runs BM25+RR: BM25 retrieval with the original query, cross-encoder reranking
with the original query, and fusion with the BM25 ranking (w1:w2 = 2:8).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from saqe import DATASETS, SAQE, load_passages, load_queries  # noqa: E402
from saqe.evaluate import evaluate  # noqa: E402
from saqe.rerankers import load_reranker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--passages", default=None, help="JSONL produced by generate_passages.py")
    ap.add_argument("--out", required=True, help="output TREC run file")
    ap.add_argument("--scores-out", default=None,
                    help="also write the per-query BM25 and cross-encoder scores (needed by select_w3.py)")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-large")
    ap.add_argument("--reranker-type", default="flag", choices=["flag", "monot5"])
    ap.add_argument("--device", default=None, help="e.g. cuda:0 (default: first GPU, else CPU)")
    ap.add_argument("--k-candidates", type=int, default=1000)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--weights", type=float, nargs=3, default=[2.0, 8.0, 1.0], metavar=("W1", "W2", "W3"),
                    help="fusion weights of the sparse, original-query, and expansion rankings")
    a = ap.parse_args()

    ds = DATASETS[a.dataset]
    queries = load_queries(ds)
    passages = load_passages(a.passages) if a.passages else None
    if passages is not None:
        missing = [q for q in queries if q not in passages]
        if missing:
            sys.exit(f"{len(missing)} queries have no passages in {a.passages}")

    w1, w2, w3 = a.weights
    model = SAQE(load_reranker(a.reranker, a.reranker_type, a.device), k_candidates=a.k_candidates,
                 rrf_k=a.rrf_k, w_sp=w1, w_q=w2, w_e=w3)
    model.run(ds, queries, passages, a.out, scores_path=a.scores_out, tag="SAQE" if passages else "BM25+RR")
    metrics = evaluate(ds, a.out)
    print(json.dumps({"dataset": a.dataset, "n_queries": len(queries), **metrics}, indent=2))
    with open(a.out + ".metrics.json", "w") as f:
        json.dump({"dataset": a.dataset, "n_queries": len(queries), "metrics": metrics, "config": vars(a)}, f, indent=2)


if __name__ == "__main__":
    main()

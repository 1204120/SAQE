"""Evaluation with trec_eval (via Pyserini).

nDCG@10 uses the graded labels. mAP, R@100, and R@1k binarize the labels at relevance >= 2 on TREC DL and
>= 1 on BEIR. Queries that are judged but absent from a run (no document retrieved) count as zero (-c).
"""
import subprocess
import sys
from typing import Dict

from .datasets import Dataset


def _trec_eval(args, qrels: str, run_path: str) -> str:
    cmd = [sys.executable, "-m", "pyserini.eval.trec_eval", "-c"] + args + [qrels, run_path]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def _aggregate(args, qrels, run_path) -> float:
    for line in _trec_eval(args, qrels, run_path).splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "all":
            return float(parts[2])
    raise RuntimeError(f"trec_eval returned no aggregate for {args}")


def evaluate(ds: Dataset, run_path: str) -> Dict[str, float]:
    lvl = [] if ds.beir else ["-l", "2"]
    metrics = {"ndcg@10": _aggregate(["-m", "ndcg_cut.10"], ds.qrels, run_path)}
    if not ds.beir:
        metrics["map"] = _aggregate(lvl + ["-m", "map"], ds.qrels, run_path)
    metrics["recall@100"] = _aggregate(lvl + ["-m", "recall.100"], ds.qrels, run_path)
    metrics["recall@1000"] = _aggregate(lvl + ["-m", "recall.1000"], ds.qrels, run_path)
    return metrics


def per_query(ds: Dataset, run_path: str, queries, measure: str = "ndcg_cut.10") -> Dict[str, float]:
    """Per-query values of a trec_eval measure for the given queries; queries absent from the run get 0."""
    out = {q: 0.0 for q in queries}
    for line in _trec_eval(["-q", "-m", measure], ds.qrels, run_path).splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in out:
            out[parts[1]] = float(parts[2])
    return out

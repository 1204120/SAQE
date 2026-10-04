"""Evaluation with trec_eval (via Pyserini) and paired bootstrap significance testing."""
import random
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
    """nDCG@10 for all datasets; additionally mAP and R@1k (relevance >= 2) for TREC DL."""
    metrics = {"ndcg@10": _aggregate(["-m", "ndcg_cut.10"], ds.qrels, run_path)}
    if ds.beir:
        metrics["recall@1000"] = _aggregate(["-m", "recall.1000"], ds.qrels, run_path)
    else:
        metrics["map"] = _aggregate(["-l", "2", "-m", "map"], ds.qrels, run_path)
        metrics["recall@1000"] = _aggregate(["-l", "2", "-m", "recall.1000"], ds.qrels, run_path)
    return metrics


def per_query_ndcg10(ds: Dataset, run_path: str) -> Dict[str, float]:
    out = {}
    for line in _trec_eval(["-q", "-m", "ndcg_cut.10"], ds.qrels, run_path).splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "ndcg_cut_10" and parts[1] != "all":
            out[parts[1]] = float(parts[2])
    return out


def paired_bootstrap(a: Dict[str, float], b: Dict[str, float], n_boot: int = 10000,
                     seed: int = 42) -> Dict[str, float]:
    """Two-sided paired bootstrap test of mean(a - b) = 0 over the shared queries."""
    qids = sorted(set(a) & set(b))
    diffs = [a[q] - b[q] for q in qids]
    n = len(diffs)
    obs = sum(diffs) / n
    centered = [d - obs for d in diffs]
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_boot):
        s = sum(centered[rng.randrange(n)] for _ in range(n)) / n
        if abs(s) >= abs(obs):
            extreme += 1
    return {"n": n, "mean_diff": obs, "p": (extreme + 1) / (n_boot + 1)}

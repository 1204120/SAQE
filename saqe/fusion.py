"""Weighted reciprocal rank fusion."""
from collections import defaultdict
from typing import List, Sequence, Tuple


def rank_by(scores: Sequence[float], docids: Sequence[str]) -> List[str]:
    """Order docids by descending score (ties keep the input order)."""
    return [d for _, d in sorted(zip(scores, docids), key=lambda x: -x[0])]


def weighted_rrf(rankings: Sequence[Tuple[List[str], float]], k: int = 60) -> List[Tuple[str, float]]:
    """S(d) = sum_i w_i / (k + r_i(d)), with ranks starting at 1.

    rankings: [(ordered docids, weight), ...]. Returns [(docid, score)] sorted by score.
    """
    score = defaultdict(float)
    for ranking, weight in rankings:
        for rank, docid in enumerate(ranking):
            score[docid] += weight / (k + rank + 1)
    return sorted(score.items(), key=lambda x: -x[1])

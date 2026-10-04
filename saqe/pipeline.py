"""SAQE retrieval pipeline.

1. Sparse retrieval: BM25 with q (+) q_doc retrieves the top-K candidates (ranking r_sp).
2. Query-anchored reranking: the cross-encoder scores every candidate against the
   unchanged query q (ranking r_q) and, separately, against q_int (ranking r_int).
3. Multi-view rank fusion: weighted RRF of r_sp, r_q, and r_int.
"""
import json
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

from .datasets import Dataset
from .fusion import rank_by, weighted_rrf


class SAQE:
    def __init__(self, reranker: str = "BAAI/bge-reranker-large", k_candidates: int = 1100,
                 rrf_k: int = 60, w_sp: float = 2.0, w_q: float = 8.0, w_int: float = 4.0,
                 use_fp16: bool = True):
        from FlagEmbedding import FlagReranker
        self.reranker = FlagReranker(reranker, use_fp16=use_fp16)
        self.k_candidates = k_candidates
        self.rrf_k = rrf_k
        self.w_sp, self.w_q, self.w_int = w_sp, w_q, w_int

    def _score(self, query: str, docs):
        scores = self.reranker.compute_score([(query, d) for d in docs])
        return scores if isinstance(scores, list) else [scores]

    def rank(self, searcher, qid: str, query: str, q_doc: str = "", q_int: str = "",
             remove_query: bool = False):
        """Return [(docid, fused score)] for one query, best first."""
        retrieval_query = f"{query}, {q_doc}" if q_doc else query
        hits = searcher.search(retrieval_query, self.k_candidates)
        if not hits:
            return []
        docids = [h.docid for h in hits]
        docs = [h.raw for h in hits]
        rankings = [
            (rank_by(self._score(query, docs), docids), self.w_q),          # r_q
            (rank_by([h.score for h in hits], docids), self.w_sp),          # r_sp
        ]
        if q_int:
            rankings.append((rank_by(self._score(q_int, docs), docids), self.w_int))  # r_int
        fused = dict(weighted_rrf(rankings, k=self.rrf_k))
        ranked = sorted(docids, key=lambda d: fused[d], reverse=True)
        if remove_query:  # ArguAna: drop the query's own document
            ranked = [d for d in ranked if d != qid]
        return [(d, fused[d]) for d in ranked]

    def run(self, ds: Dataset, queries: Dict[str, str], expansions: Optional[Dict[str, dict]],
            run_path: str, tag: str = "SAQE"):
        """Run SAQE on a dataset and write a TREC run file.

        expansions: {qid: {"q_doc": ..., "q_int": ...}}; pass None to run without expansion
        (BM25 + cross-encoder with two-view fusion).
        """
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher.from_prebuilt_index(ds.index)
        Path(run_path).parent.mkdir(parents=True, exist_ok=True)
        with open(run_path, "w") as f:
            for qid, query in tqdm(list(queries.items()), desc=ds.name):
                exp = (expansions or {}).get(qid, {})
                ranked = self.rank(searcher, qid, query, exp.get("q_doc", ""), exp.get("q_int", ""),
                                   remove_query=ds.remove_query)
                for rank, (docid, score) in enumerate(ranked, start=1):
                    f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")
        return run_path

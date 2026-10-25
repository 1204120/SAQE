"""SAQE retrieval pipeline.

1. Candidate retrieval: BM25 with q^{x n} (+) P, where P concatenates the greedy passage p0 and the sampled
   passages p1..pm, and q is repeated n = max(1, floor(|P| / (beta |q|))) times (beta = 4, lengths in
   characters). The top-K documents form the candidate set; their order is the sparse ranking r_sp.
2. Query-anchored reranking: the cross-encoder scores every candidate against the unchanged query q
   (original-query ranking r_q) and, separately, against p0 (expansion ranking r_e). Generated text is
   never appended to the cross-encoder query.
3. Multi-view rank fusion: S(d) = w1/(k + r_sp(d)) + w2/(k + r_q(d)) + w3/(k + r_e(d)).

Without passages, the pipeline reduces to BM25+RR (BM25 retrieval, cross-encoder reranking with q, and
fusion with the BM25 ranking at w1:w2).
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from .datasets import Dataset
from .fusion import rank_by, weighted_rrf


def retrieval_query(query: str, passages: Sequence[str], beta: int = 4) -> str:
    """q^{x n} (+) P: n space-separated copies of q, a comma, and the passages separated by spaces."""
    text = " ".join(p for p in passages if p)
    if not text:
        return query
    n = max(1, (len(text) // max(len(query), 1)) // beta)
    return " ".join([query] * n) + ", " + text


def bm25_searcher(index: str):
    """Pyserini BM25 with default parameters (k1=0.9, b=0.4) on a prebuilt index name or a local index path."""
    import os
    from pyserini.search.lucene import LuceneSearcher
    from jnius import autoclass
    # Long expanded queries can exceed Lucene's default limit of 1,024 clauses; raising the limit does not
    # change the results of queries below it.
    autoclass("org.apache.lucene.search.IndexSearcher").setMaxClauseCount(65536)
    return LuceneSearcher(index) if os.path.isdir(index) else LuceneSearcher.from_prebuilt_index(index)


class SAQE:
    def __init__(self, reranker, k_candidates: int = 1000, rrf_k: int = 60,
                 w_sp: float = 2.0, w_q: float = 8.0, w_e: float = 1.0, beta: int = 4):
        """reranker: an object with score(pairs) -> list of floats (see saqe.rerankers)."""
        self.reranker = reranker
        self.k_candidates, self.rrf_k, self.beta = k_candidates, rrf_k, beta
        self.w_sp, self.w_q, self.w_e = w_sp, w_q, w_e

    def rank(self, searcher, query: str, p0: str = "", samples: Sequence[str] = ()):
        """Return ([(docid, fused score)] best first, raw scores) for one query."""
        ret_query = retrieval_query(query, [p0, *samples], self.beta) if (p0 or any(samples)) else query
        hits = searcher.search(ret_query, self.k_candidates)
        if not hits:
            return [], None
        docids = [h.docid for h in hits]
        docs = [h.raw for h in hits]                       # the stored document text of the prebuilt index
        bm25 = [h.score for h in hits]
        ce_q = self.reranker.score([(query, d) for d in docs])
        rankings = [(rank_by(ce_q, docids), self.w_q),        # r_q
                    (rank_by(bm25, docids), self.w_sp)]       # r_sp
        ce_e = None
        if p0:
            ce_e = self.reranker.score([(p0, d) for d in docs])
            rankings.append((rank_by(ce_e, docids), self.w_e))  # r_e
        fused = dict(weighted_rrf(rankings, k=self.rrf_k))
        ranked = sorted(docids, key=lambda d: fused[d], reverse=True)
        scores = {"docids": docids, "bm25": [float(x) for x in bm25], "ce_q": ce_q, "ce_e": ce_e}
        return [(d, fused[d]) for d in ranked], scores

    def run(self, ds: Dataset, queries: Dict[str, str], passages: Optional[Dict[str, dict]], run_path: str,
            scores_path: Optional[str] = None, tag: str = "SAQE") -> str:
        """Run on a dataset and write a TREC run file (and, optionally, the per-query raw scores as JSONL,
        which scripts/select_w3.py uses to re-fuse the rankings with other weights)."""
        searcher = bm25_searcher(ds.index)
        Path(run_path).parent.mkdir(parents=True, exist_ok=True)
        dump = open(scores_path, "w") if scores_path else None
        with open(run_path, "w") as f:
            for qid, query in tqdm(list(queries.items()), desc=ds.name):
                rec = (passages or {}).get(qid, {})
                ranked, scores = self.rank(searcher, query, rec.get("p0", ""), rec.get("samples", []))
                for rank, (docid, score) in enumerate(ranked, start=1):
                    f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")
                if dump is not None and scores is not None:
                    dump.write(json.dumps({"qid": qid, **scores}) + "\n")
        if dump is not None:
            dump.close()
        return run_path

"""Evaluation datasets (Pyserini prebuilt indexes, topics, and qrels)."""
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Dataset:
    name: str
    index: str              # Pyserini prebuilt index
    topics: str             # Pyserini topics
    qrels: str              # Pyserini qrels
    beir: bool              # BEIR-style evaluation (no relevance-level cutoff)
    remove_query: bool = False  # ArguAna: the query itself is a document in the corpus


DATASETS: Dict[str, Dataset] = {d.name: d for d in [
    Dataset("dl19", "msmarco-passage", "dl19-passage", "dl19-passage", beir=False),
    Dataset("dl20", "msmarco-passage", "dl20", "dl20-passage", beir=False),
    Dataset("scifact", "beir-v1.0.0-scifact.flat", "beir-v1.0.0-scifact-test",
            "beir-v1.0.0-scifact-test", beir=True),
    Dataset("arguana", "beir-v1.0.0-arguana.flat", "beir-v1.0.0-arguana-test",
            "beir-v1.0.0-arguana-test", beir=True, remove_query=True),
    Dataset("trec-covid", "beir-v1.0.0-trec-covid.flat", "beir-v1.0.0-trec-covid-test",
            "beir-v1.0.0-trec-covid-test", beir=True),
    Dataset("fiqa", "beir-v1.0.0-fiqa.flat", "beir-v1.0.0-fiqa-test",
            "beir-v1.0.0-fiqa-test", beir=True),
    Dataset("dbpedia-entity", "beir-v1.0.0-dbpedia-entity.flat", "beir-v1.0.0-dbpedia-entity-test",
            "beir-v1.0.0-dbpedia-entity-test", beir=True),
    Dataset("trec-news", "beir-v1.0.0-trec-news.flat", "beir-v1.0.0-trec-news-test",
            "beir-v1.0.0-trec-news-test", beir=True),
]}


def load_queries(ds: Dataset) -> Dict[str, str]:
    """Return {qid: query} for the judged queries of a dataset.

    Only queries that appear in the qrels are kept (e.g., 54 of the 200 DL20 topics).
    """
    from pyserini.search import get_topics, get_qrels
    topics = get_topics(ds.topics)
    judged = {str(q) for q in get_qrels(ds.qrels)}
    return {str(q): t["title"] for q, t in topics.items() if str(q) in judged}

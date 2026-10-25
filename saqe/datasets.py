"""Evaluation datasets (Pyserini prebuilt indexes, topics, and qrels) and the validation set."""
import random
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Dataset:
    name: str
    index: str      # Pyserini prebuilt index
    topics: str     # Pyserini topics
    qrels: str      # Pyserini qrels
    beir: bool      # BEIR-style evaluation (relevance >= 1); TREC DL binarizes at relevance >= 2


def _beir(name: str) -> Dataset:
    return Dataset(name, f"beir-v1.0.0-{name}.flat", f"beir-v1.0.0-{name}-test", f"beir-v1.0.0-{name}-test", beir=True)


DATASETS: Dict[str, Dataset] = {d.name: d for d in [
    Dataset("dl19", "msmarco-passage", "dl19-passage", "dl19-passage", beir=False),
    Dataset("dl20", "msmarco-passage", "dl20", "dl20-passage", beir=False),
    _beir("scifact"), _beir("trec-covid"), _beir("fiqa"), _beir("dbpedia-entity"), _beir("trec-news"), _beir("nfcorpus"),
    # Validation set: 500 queries sampled from the MS MARCO passage dev set, used only to select w3.
    Dataset("msmarco-dev500", "msmarco-passage", "msmarco-passage-dev-subset", "msmarco-passage-dev-subset", beir=True),
]}
TEST_SETS = ["dl19", "dl20", "scifact", "trec-covid", "fiqa", "dbpedia-entity", "trec-news", "nfcorpus"]
DEV_SIZE, DEV_SEED = 500, 42


def load_queries(ds: Dataset) -> Dict[str, str]:
    """Return {qid: query} for the judged queries of a dataset (e.g., 54 of the 200 DL20 topics).

    For msmarco-dev500, 500 of the 6,980 judged dev queries are sampled with a fixed seed; none of them
    occurs in DL19 or DL20.
    """
    from pyserini.search import get_qrels, get_topics
    topics = {str(q): t["title"] for q, t in get_topics(ds.topics).items()}
    judged = {str(q) for q in get_qrels(ds.qrels)}
    qids = sorted(set(topics) & judged, key=lambda q: (0, int(q), "") if q.isdigit() else (1, 0, q))
    if ds.name == "msmarco-dev500":
        qids = sorted(random.Random(DEV_SEED).sample(qids, DEV_SIZE), key=int)
    return {q: topics[q] for q in qids}

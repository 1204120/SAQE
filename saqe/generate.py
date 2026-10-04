"""Generation of the document-oriented (q_doc) and intent-oriented (q_int) expansions."""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from . import prompts as P
from .llm import ChatClient


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip('"')


def expand_query(client: ChatClient, query: str) -> Dict[str, str]:
    """Return {"q_doc": ..., "q_int": ...} for a single query (two independent requests)."""
    q_doc = client.chat(P.DOC_SYSTEM, P.DOC_USER.format(query=query, n_words=P.N_WORDS),
                        max_tokens=P.DOC_MAX_TOKENS)
    q_int = client.chat(P.INT_SYSTEM, P.INT_USER.format(query=query, n_words=P.N_WORDS),
                        max_tokens=P.INT_MAX_TOKENS)
    return {"q_doc": _clean(q_doc), "q_int": _clean(q_int)}


def generate_expansions(client: ChatClient, queries: Dict[str, str], out_path: str,
                        workers: int = 8) -> Dict[str, dict]:
    """Generate expansions for all queries and append them to a JSONL file.

    Each line: {"qid", "query", "q_doc", "q_int"}. Already generated qids are skipped,
    so an interrupted run can be resumed.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for line in out.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                done[rec["qid"]] = rec
    todo = [(qid, q) for qid, q in queries.items() if qid not in done]

    def work(item):
        qid, q = item
        return {"qid": qid, "query": q, **expand_query(client, q)}

    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        for rec in tqdm(pool.map(work, todo), total=len(todo), desc="generating"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[rec["qid"]] = rec
    return done


def load_expansions(path: str) -> Dict[str, dict]:
    recs = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            recs[str(rec["qid"])] = rec
    return recs

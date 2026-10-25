"""Generation of the SAQE passages: p0 with greedy decoding and p1, ..., p4 by sampling from the same prompt."""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from . import prompts as P
from .llm import ChatClient

# Labels such as "Passage:" that a chat model occasionally repeats at the beginning of its output.
_LABEL = re.compile(r"^\s*(\*\*)?\s*(<PASSAGE>|Passage|Rewrite|Document|Counter Argument)\s*:?\s*(\*\*)?\s*:?\s*", re.I)


def clean(text: str) -> str:
    """Remove a leading label, collapse whitespace, and strip surrounding quotes."""
    return re.sub(r"\s+", " ", _LABEL.sub("", text or "")).strip().strip('"').strip()


def generate_passages(client: ChatClient, query: str) -> Dict:
    """Return {"p0": ..., "samples": [p1, ..., p4]} for one query (two requests), plus the raw outputs."""
    user = P.USER.format(query=query, n_words=P.N_WORDS)
    greedy = client.chat(P.SYSTEM, user, max_tokens=P.MAX_TOKENS, **P.GREEDY)[0]
    sampled = client.chat(P.SYSTEM, user, max_tokens=P.MAX_TOKENS, **P.SAMPLED)
    return {"p0": clean(greedy), "samples": [clean(x) for x in sampled],
            "raw_p0": greedy, "raw_samples": sampled}


def generate_all(client: ChatClient, queries: Dict[str, str], out_path: str, workers: int = 16) -> Dict[str, dict]:
    """Generate passages for all queries and append them to a JSONL file.

    Each line: {"qid", "query", "p0", "samples", "raw_p0", "raw_samples"}. Already generated qids are
    skipped, so an interrupted run can be resumed.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_passages(out_path) if out.exists() else {}
    todo = [(qid, q) for qid, q in queries.items() if qid not in done]

    def work(item):
        qid, q = item
        return {"qid": qid, "query": q, **generate_passages(client, q)}

    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        for rec in tqdm(pool.map(work, todo), total=len(todo), desc="generating"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[rec["qid"]] = rec
    return done


def load_passages(path: str) -> Dict[str, dict]:
    recs = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            recs[str(rec["qid"])] = rec
    return recs

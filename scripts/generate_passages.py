#!/usr/bin/env python3
"""Generate the SAQE passages (p0 with greedy decoding, p1..p4 by sampling) for all queries of a dataset.

Example (Qwen3-8B served locally with vLLM, see README):
    python scripts/generate_passages.py --dataset dl19 \
        --base-url http://127.0.0.1:8000/v1 --model Qwen3-8B --out passages/qwen3-8b/dl19.jsonl

Use --dataset msmarco-dev500 to generate passages for the validation queries that select w3.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from saqe import DATASETS, ChatClient, generate_all, load_queries  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--out", required=True, help="output JSONL file (resumable)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-vllm-extras", action="store_true",
                    help="do not send top_k/min_p/repetition_penalty/chat_template_kwargs (for non-vLLM endpoints)")
    ap.add_argument("--use-env-proxy", action="store_true", help="honour HTTP(S)_PROXY environment variables")
    a = ap.parse_args()

    queries = load_queries(DATASETS[a.dataset])
    client = ChatClient(a.base_url, a.model, vllm_extras=not a.no_vllm_extras, use_env_proxy=a.use_env_proxy)
    done = generate_all(client, queries, a.out, workers=a.workers)
    n = sum(1 for q in queries if q in done)
    empty = sum(1 for q in queries if q in done and not done[q]["p0"])
    print(f"{a.dataset}: {n}/{len(queries)} queries ({empty} with an empty p0) -> {a.out}")


if __name__ == "__main__":
    main()

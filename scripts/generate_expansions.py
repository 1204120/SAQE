#!/usr/bin/env python3
"""Generate q_doc and q_int for all judged queries of a dataset.

Example (Qwen3-32B served locally with vLLM):
    python scripts/generate_expansions.py --dataset dl19 \
        --base-url http://127.0.0.1:8000/v1 --model Qwen3-32B --disable-thinking \
        --out expansions/qwen3-32b/dl19.jsonl

For an OpenAI-compatible API, set OPENAI_API_KEY in the environment and pass
--base-url and --model accordingly (e.g., --model gpt-3.5-turbo).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from saqe import DATASETS, ChatClient, generate_expansions, load_queries  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output JSONL file")
    ap.add_argument("--disable-thinking", action="store_true", help="disable Qwen3 thinking mode (as in the paper)")
    ap.add_argument("--use-env-proxy", action="store_true", help="honour HTTP(S)_PROXY environment variables")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    ds = DATASETS[a.dataset]
    queries = load_queries(ds)
    client = ChatClient(a.base_url, a.model, disable_thinking=a.disable_thinking,
                        use_env_proxy=a.use_env_proxy)
    done = generate_expansions(client, queries, a.out, workers=a.workers)
    empty = sum(1 for r in done.values() if not r["q_doc"] or not r["q_int"])
    print(f"{a.dataset}: {len(done)}/{len(queries)} queries expanded ({empty} with an empty expansion) -> {a.out}")


if __name__ == "__main__":
    main()

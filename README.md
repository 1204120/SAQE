# SAQE: Stage-Aligned Query Expansion for Two-Stage Retrieval

This repository contains the implementation of **SAQE**, a framework that determines how LLM-generated
expansion content enters each stage of a retrieve-then-rerank pipeline.

- **Candidate retrieval.** An LLM generates a *document-oriented expansion* `q_doc`, which is appended
  to the original query `q` for BM25 retrieval of the top-K candidates (sparse ranking `r_sp`).
- **Query-anchored reranking.** The cross-encoder scores every candidate against the unchanged query `q`
  (original-query ranking `r_q`). A second, *intent-oriented expansion* `q_int`, which describes what the
  user is looking for without answering the query, is scored separately by the same cross-encoder
  (intent-oriented ranking `r_int`). Generated content is never appended to the reranking query.
- **Multi-view rank fusion.** The final ranking combines the three rankings with weighted reciprocal
  rank fusion: `S(d) = w_sp/(k + r_sp(d)) + w_q/(k + r_q(d)) + w_int/(k + r_int(d))`.

## Repository structure

```
saqe/
  prompts.py     prompts for q_doc and q_int (as used in the paper)
  llm.py         client for OpenAI-compatible chat endpoints (vLLM or API)
  generate.py    generation of q_doc and q_int
  datasets.py    TREC DL19/DL20 and six BEIR datasets (Pyserini prebuilt indexes)
  pipeline.py    BM25 retrieval, cross-encoder scoring, and multi-view rank fusion
  fusion.py      weighted reciprocal rank fusion
  evaluate.py    trec_eval metrics and paired bootstrap significance test
scripts/
  generate_expansions.py
  run_saqe.py
  significance.py
```

## Installation

```bash
pip install -r requirements.txt
```

Pyserini requires a Java 11+ runtime (`JAVA_HOME` must be set). The experiments in the paper were run
with Python 3.8, Pyserini 0.22.1, FlagEmbedding 1.3.3, Transformers 4.46.3, and PyTorch 2.4.0 on NVIDIA L40 GPUs.

## Usage

### 1. Serve the generating LLM

The paper uses Qwen3-32B served with vLLM in FP8 precision with a context length of 4,096 tokens:

```bash
vllm serve Qwen/Qwen3-32B --served-model-name Qwen3-32B --quantization fp8 --max-model-len 4096
```

Any OpenAI-compatible endpoint can be used instead; if it requires a key, set `OPENAI_API_KEY`.

### 2. Generate the two expansions

```bash
python scripts/generate_expansions.py --dataset dl19 \
    --base-url http://127.0.0.1:8000/v1 --model Qwen3-32B --disable-thinking \
    --out expansions/qwen3-32b/dl19.jsonl
```

Each output line contains `qid`, `query`, `q_doc`, and `q_int`. Generation uses greedy decoding
(temperature 0); `--disable-thinking` turns off the Qwen3 thinking mode, as in the paper.

### 3. Run SAQE and evaluate

```bash
python scripts/run_saqe.py --dataset dl19 \
    --expansions expansions/qwen3-32b/dl19.jsonl --out runs/dl19.saqe.trec
```

Default settings follow the paper: `bge-reranker-large` (FP16), K = 1,100 candidates, RRF constant k = 60,
and fusion weights `w_sp : w_q : w_int = 2 : 8 : 4`. Running the script without `--expansions` gives the
BM25 + cross-encoder baseline (BM25+RR). Metrics are nDCG@10 for all datasets and, for TREC DL, also mAP and
R@1k with graded labels binarized at relevance >= 2.

Available datasets: `dl19`, `dl20`, `scifact`, `arguana`, `trec-covid`, `fiqa`, `dbpedia-entity`, `trec-news`.

### 4. Significance testing

```bash
python scripts/significance.py --dataset dl19 --run-a runs/dl19.saqe.trec --run-b runs/dl19.bm25rr.trec
```

This runs a two-sided paired bootstrap test (10,000 samples, seed 42) on per-query nDCG@10.

## Expected results

nDCG@10 of SAQE with Qwen3-32B as the generator and `bge-reranker-large` as the cross-encoder:

| DL19 | DL20 | SciFact | ArguAna | TREC-COVID | FiQA | DBpedia | TREC-NEWS | Avg. |
|---|---|---|---|---|---|---|---|---|
| 76.7 | 75.8 | 76.1 | 41.0 | 77.3 | 40.4 | 47.8 | 47.6 | 60.3 |

Given the same expansions, the pipeline is deterministic up to FP16 numerical differences of the
cross-encoder. Re-generating the expansions may introduce small variations because LLM serving is not
always bit-exact, even with greedy decoding.

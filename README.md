# SAQE: Stage-Aligned Query Expansion for Two-Stage Retrieval

This repository contains the implementation of **SAQE**, a training-free framework that integrates
LLM-generated text into each stage of a retrieve-then-rerank pipeline according to the objective of that stage,
without overriding the relevance signal of the original query.

- **Passage generation.** With a single prompt, an LLM generates one passage `p0` with greedy decoding and
  four further passages `p1, ..., p4` by sampling (temperature 0.7).
- **Candidate retrieval.** All passages are appended to the original query `q` for BM25 retrieval of the top-K
  candidates. Following MuGI, `q` is repeated `n = max(1, floor(|P| / (4 |q|)))` times, where `|P|` and `|q|`
  are the lengths in characters of the passages and of the query. The BM25 order is the sparse ranking `r_sp`.
- **Query-anchored reranking.** The cross-encoder scores every candidate against the unchanged query `q`
  (original-query ranking `r_q`) and, separately, against `p0` (expansion ranking `r_e`).
  Generated text is never appended to the cross-encoder query.
- **Multi-view rank fusion.** `S(d) = w1/(k + r_sp(d)) + w2/(k + r_q(d)) + w3/(k + r_e(d))` with `k = 60`,
  `w1 : w2 = 2 : 8`, and `w3` selected on held-out MS MARCO dev queries (`w3 = 1` for all generating LLMs in the paper).

## Repository structure

```
saqe/
  prompts.py     the single SAQE prompt and its decoding settings (as used in the paper)
  llm.py         client for OpenAI-compatible chat endpoints (e.g., vLLM)
  generate.py    generation of p0 (greedy) and p1..p4 (sampled)
  datasets.py    TREC DL19/DL20, six BEIR datasets, and the 500-query validation set (Pyserini)
  pipeline.py    retrieval query, BM25 retrieval, query-anchored reranking, and rank fusion
  rerankers.py   bge cross-encoders (FlagEmbedding) and monoT5
  fusion.py      weighted reciprocal rank fusion
  evaluate.py    trec_eval metrics
  stats.py       paired bootstrap tests (pooled and dataset-stratified) and Holm correction
scripts/
  generate_passages.py   generate the passages for a dataset
  run_saqe.py            run SAQE (or BM25+RR) and evaluate
  select_w3.py           select w3 on the validation queries
  significance.py        paired bootstrap tests between two systems
serve/
  mistral-7b-instruct-v0.3.jinja   chat template for Mistral-7B-Instruct-v0.3 (see below)
```

## Installation

```bash
pip install -r requirements.txt
```

Pyserini requires a Java 11+ runtime (`JAVA_HOME` must be set). Retrieval and reranking in the paper used
Python 3.8, Pyserini 0.22.1, FlagEmbedding 1.3.3, Transformers 4.46.3, and PyTorch 2.4.0. The generating LLMs
were served with vLLM in a separate environment.

## Usage

### 1. Serve the generating LLM

The paper uses Qwen3-8B, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.3 (and Qwen3-14B/32B for the scale
study), served in BF16 with vLLM and a context length of 4,096 tokens:

```bash
vllm serve Qwen/Qwen3-8B --served-model-name Qwen3-8B --dtype bfloat16 --max-model-len 4096 \
    --generation-config vllm --seed 42 --port 8000
```

`--generation-config vllm` prevents vLLM from applying the default sampling parameters shipped with a model;
every request also sets its sampling parameters explicitly (`top_k=-1`, `min_p=0`, neutral repetition penalties),
and the Qwen3 thinking mode is disabled. Local copies of Mistral-7B-Instruct-v0.3 may ship an older chat template
without system-message support; we used the official template from the model's current `tokenizer_config.json`,
provided in `serve/`:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --served-model-name Mistral-7B-Instruct-v0.3 --dtype bfloat16 --max-model-len 4096 \
    --generation-config vllm --seed 42 --port 8000 \
    --chat-template serve/mistral-7b-instruct-v0.3.jinja --tokenizer-mode hf
```

### 2. Generate the passages

```bash
for ds in dl19 dl20 scifact trec-covid fiqa dbpedia-entity trec-news nfcorpus msmarco-dev500; do
  python scripts/generate_passages.py --dataset $ds \
      --base-url http://127.0.0.1:8000/v1 --model Qwen3-8B --out passages/qwen3-8b/$ds.jsonl
done
```

Each output line contains `qid`, `query`, `p0`, `samples` (`p1..p4`), and the raw model outputs. For other
OpenAI-compatible endpoints, set `OPENAI_API_KEY` if needed and pass `--no-vllm-extras` if the endpoint rejects
the vLLM-specific sampling fields.

### 3. Select w3 on the validation queries

```bash
python scripts/run_saqe.py --dataset msmarco-dev500 --passages passages/qwen3-8b/msmarco-dev500.jsonl \
    --out runs/qwen3-8b/msmarco-dev500.trec --scores-out runs/qwen3-8b/msmarco-dev500.scores.jsonl
python scripts/select_w3.py --scores runs/qwen3-8b/msmarco-dev500.scores.jsonl
```

The validation set consists of 500 queries sampled with a fixed seed from the 6,980 judged queries of the MS MARCO
passage dev set; none of them occurs in DL19 or DL20.

### 4. Run SAQE and evaluate

```bash
python scripts/run_saqe.py --dataset dl19 --passages passages/qwen3-8b/dl19.jsonl \
    --weights 2 8 1 --out runs/qwen3-8b/dl19.saqe.trec
python scripts/run_saqe.py --dataset dl19 --out runs/dl19.bm25rr.trec      # BM25+RR (no expansion)
```

Defaults follow the paper: `bge-reranker-large` in FP16, K = 1,000 candidates, RRF constant k = 60, and fusion
weights `w1 : w2 : w3 = 2 : 8 : 1`. Reported metrics are nDCG@10 for all datasets and, for TREC DL, also mAP and
R@100 with labels binarized at relevance >= 2. Other rerankers can be used with `--reranker` (e.g.,
`BAAI/bge-reranker-v2-m3`) or `--reranker-type monot5 --reranker castorini/monot5-base-msmarco-10k`.

Available datasets: `dl19`, `dl20`, `scifact`, `trec-covid`, `fiqa`, `dbpedia-entity`, `trec-news`, `nfcorpus`,
and the validation set `msmarco-dev500`.

### 5. Significance testing

```bash
python scripts/significance.py --datasets dl19,dl20,scifact,trec-covid,fiqa,dbpedia-entity,trec-news,nfcorpus \
    --run-a "runs/qwen3-8b/{dataset}.saqe.trec" --run-b "runs/{dataset}.bm25rr.trec"
```

This runs two-sided paired bootstrap tests (10,000 resamples, seed 42) on per-query nDCG@10 for each dataset and,
for the average over the datasets, both a pooled test and a dataset-stratified test. `saqe.stats.holm` applies the
Holm correction across several baselines.

## Expected results

nDCG@10 with `bge-reranker-large` (BM25 retrieval of the top 1,000 documents, w1 : w2 : w3 = 2 : 8 : 1):

| Method | DL19 | DL20 | SciFact | TREC-COVID | FiQA | DBpedia | TREC-NEWS | NFCorpus | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| BM25+RR | 73.3 | 72.4 | 74.1 | 73.1 | 37.8 | 44.5 | 42.8 | 34.5 | 56.6 |
| SAQE (Qwen3-8B) | 77.2 | 76.3 | 75.4 | 77.9 | 40.1 | 48.4 | 47.5 | 35.8 | 59.8 |
| SAQE (Llama-3.1-8B-Instruct) | 76.8 | 74.9 | 74.9 | 77.3 | 40.5 | 48.6 | 46.5 | 35.6 | 59.4 |
| SAQE (Mistral-7B-Instruct-v0.3) | 77.7 | 76.0 | 75.8 | 77.5 | 40.2 | 48.5 | 47.0 | 36.0 | 59.9 |

Given the same passages, the pipeline is deterministic up to FP16 numerical differences of the cross-encoder.
Re-generating the passages may introduce small variations, because sampling and LLM serving are not bit-exact
across hardware and library versions.

## License

This project is released under the [MIT License](LICENSE). The chat template in `serve/` is taken from the
Mistral-7B-Instruct-v0.3 model repository.

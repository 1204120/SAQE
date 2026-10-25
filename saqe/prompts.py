"""The single prompt used by SAQE (Fig. S1 of the supplementary material), identical to the experiments.

The same prompt is used for all datasets and generating LLMs. One request with greedy decoding
produces the passage p0; one request with sampling produces the passages p1, ..., p4.
"""

N_WORDS = 60

SYSTEM = (
    "You expand search queries into short passages that read like excerpts from relevant documents. "
    "Output only the passage text."
)

USER = """Query: {query}

Write a single passage of {n_words} words that reads like an excerpt from a reference
document answering this query.

Rules:
- State facts directly. Do not address the reader, do not describe what the passage covers,
  and do not use phrases like "this article explains" or "explore the".
- Write plain prose. No headings, no lists, no citations.

Passage:"""

MAX_TOKENS = 300                                   # per passage
GREEDY = {"temperature": 0.0, "top_p": 1.0, "n": 1}  # p0
SAMPLED = {"temperature": 0.7, "top_p": 1.0, "n": 4}  # p1, ..., p4 (m = 4)

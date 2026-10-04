"""Prompts used by SAQE (identical to those used in the paper's experiments).

pi_doc produces the document-oriented expansion q_doc, which is appended to the
query for first-stage BM25 retrieval. pi_int produces the intent-oriented
expansion q_int, which is scored separately by the cross-encoder.
"""

N_WORDS = 60

DOC_SYSTEM = (
    "You write short factual passages that are grounded strictly in provided source material. "
    "Output only the passage text."
)

DOC_USER = """Query: {query}

Write a single passage of {n_words} words that reads like an excerpt from a reference
document answering this query.

Rules:
- State facts directly. Do not address the reader, do not describe what the passage covers,
  and do not use phrases like "this article explains" or "explore the".
- Write plain prose. No headings, no lists, no citations.

Passage:"""

INT_SYSTEM = "You rewrite search queries into richer descriptions of what the user wants."

INT_USER = """Query: {query}

Rewrite this query as a single sentence of about {n_words} words that describes what the
user is looking for and which aspects a relevant document should cover.

Rules:
- Describe the information need. Do not answer the query and do not assert facts.
- Start with a verb such as "Explore", "Describe", "Examine" or "Identify".
- List the facets a relevant document would address, separated by commas.
- Plain prose, one sentence, no lists or headings.

Rewrite:"""

# Maximum number of generated tokens per expansion.
DOC_MAX_TOKENS = 300
INT_MAX_TOKENS = 250

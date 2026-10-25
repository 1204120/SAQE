"""Rerankers that score (query, document) pairs: bge cross-encoders via FlagEmbedding, and monoT5."""
from typing import List, Sequence, Tuple


def default_device() -> str:
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class FlagCrossEncoder:
    """bge-reranker-large (default) or bge-reranker-v2-m3, used in FP16 without fine-tuning.

    With FlagEmbedding 1.3.3 defaults, each pair is truncated to 512 tokens and the query to 384 tokens.
    Scoring runs on a single device, as in the paper's experiments.
    """

    def __init__(self, model: str = "BAAI/bge-reranker-large", use_fp16: bool = True, device: str = None):
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(model, use_fp16=use_fp16, devices=device or default_device())

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        s = self.model.compute_score(list(pairs))
        return [float(x) for x in s] if hasattr(s, "__len__") else [float(s)]


class MonoT5:
    """monoT5 (e.g., castorini/monot5-base-msmarco-10k), as in pygaggle: input "Query: q Document: d Relevant:",
    truncated to 512 tokens; the score is the log-softmax of "true" versus "false" at the first decoding step."""

    def __init__(self, model: str = "castorini/monot5-base-msmarco-10k", device: str = None,
                 batch_size: int = 64, max_length: int = 512):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.torch = torch
        self.device = device or default_device()
        self.tok = AutoTokenizer.from_pretrained(model)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32   # T5 overflows in FP16
        self.model = T5ForConditionalGeneration.from_pretrained(model, torch_dtype=dtype).to(self.device).eval()
        self.ids = [self.tok.encode(w)[0] for w in ("false", "true")]
        self.batch_size, self.max_length = batch_size, max_length

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        torch, out = self.torch, []
        texts = [f"Query: {q} Document: {d} Relevant:" for q, d in pairs]
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                dec = torch.full((enc["input_ids"].shape[0], 1), self.model.config.decoder_start_token_id,
                                 dtype=torch.long, device=self.device)
                logits = self.model(**enc, decoder_input_ids=dec).logits[:, 0, self.ids].float()
                out.extend(torch.log_softmax(logits, dim=-1)[:, 1].tolist())
        return out


def load_reranker(name: str, kind: str = "flag", device: str = None):
    if kind == "flag":
        return FlagCrossEncoder(name, device=device)
    if kind == "monot5":
        return MonoT5(name, device=device)
    raise ValueError(f"unknown reranker type {kind!r} (use 'flag' or 'monot5')")

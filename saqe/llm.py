"""Minimal client for OpenAI-compatible chat-completion endpoints (e.g., a local vLLM server).

Every request specifies its sampling parameters explicitly, so that the default generation settings
shipped with a model (e.g., top_k=20 for Qwen3) are not applied. The API key, if needed, is read from
the environment and never written to disk.
"""
import os
import time
from typing import List, Optional

import requests

# Disable top-k and min-p sampling and use neutral repetition penalties (vLLM extensions of the API).
SAMPLING_FIXED = {"top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
                  "presence_penalty": 0.0, "frequency_penalty": 0.0}


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, vllm_extras: bool = True,
                 timeout: int = 900, retries: int = 4, use_env_proxy: bool = False):
        """
        base_url: e.g. "http://127.0.0.1:8000/v1" for a local vLLM server.
        vllm_extras: send top_k / min_p / repetition_penalty and disable the Qwen3 thinking mode
            (chat_template_kwargs), as in the paper. Set False for endpoints that reject these fields.
        use_env_proxy: whether to honour HTTP(S)_PROXY environment variables.
        """
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.vllm_extras = vllm_extras
        self.timeout, self.retries = timeout, retries
        self.session = requests.Session()
        self.session.trust_env = use_env_proxy
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if key:
            self.session.headers.update({"Authorization": f"Bearer {key}"})

    def chat(self, system: str, user: str, temperature: float = 0.0, n: int = 1, max_tokens: int = 300,
             top_p: float = 1.0) -> List[str]:
        """Return the n completions of one request."""
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "temperature": temperature, "n": n, "max_tokens": max_tokens, "top_p": top_p}
        if self.vllm_extras:
            payload.update(SAMPLING_FIXED)
            payload["chat_template_kwargs"] = {"enable_thinking": False}   # only affects Qwen3
        last = None
        for attempt in range(self.retries):
            try:
                r = self.session.post(self.url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return [c["message"]["content"] for c in r.json()["choices"]]
            except Exception as e:  # network errors, rate limits, server restarts
                last = e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {self.retries} attempts: {last}")

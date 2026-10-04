"""Minimal client for OpenAI-compatible chat-completion endpoints.

Works with a local vLLM server (e.g., Qwen3-32B) or any OpenAI-compatible API.
The API key, if needed, is read from the environment and never written to disk.
"""
import os
import time
from typing import Optional

import requests


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None,
                 temperature: float = 0.0, disable_thinking: bool = False,
                 timeout: int = 300, use_env_proxy: bool = False):
        """
        base_url: e.g. "http://127.0.0.1:8000/v1" for a local vLLM server.
        disable_thinking: set True for Qwen3 models served by vLLM, so that the
            chat template is rendered with enable_thinking=False (as in the paper).
        use_env_proxy: whether to honour HTTP(S)_PROXY environment variables.
        """
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.disable_thinking = disable_thinking
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = use_env_proxy
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if key:
            self.session.headers.update({"Authorization": f"Bearer {key}"})

    def chat(self, system: str, user: str, max_tokens: int, retries: int = 5) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        last = None
        for attempt in range(retries):
            try:
                r = self.session.post(self.url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:  # network errors, rate limits, server restarts
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {retries} attempts: {last}")

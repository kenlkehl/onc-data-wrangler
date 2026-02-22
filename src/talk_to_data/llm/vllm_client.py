"""OpenAI-compatible LLM client for vLLM and similar servers."""

import json
import logging
from typing import Optional

from openai import OpenAI

from .base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

REASONING_MARKER = "assistantfinal"


class VLLMClient(LLMClient):
    """Client for vLLM or any OpenAI-compatible API server.

    Used for PHI-containing extraction tasks where data cannot leave
    the local network.
    """

    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "none", model: str = "gpt-oss-120b"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt: str, system: str = "", max_tokens: int = 8000, temperature: float = 0.0) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text = response.choices[0].message.content or ""
        text = strip_reasoning(text)

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(text=text, usage=usage, raw=response)

    def generate_structured(self, prompt: str, system: str = "", max_tokens: int = 8000, temperature: float = 0.0) -> LLMResponse:
        return self.generate(prompt, system, max_tokens, temperature)


def strip_reasoning(text: str) -> str:
    """Remove reasoning tokens before the final-answer marker."""
    if REASONING_MARKER in text:
        return text.split(REASONING_MARKER, 1)[-1].strip()
    return text

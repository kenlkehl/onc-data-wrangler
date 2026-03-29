"""Azure OpenAI LLM client."""

import logging
import os
import subprocess
import threading
import time
from typing import Optional

from openai import OpenAI

from .base import LLMClient, LLMResponse
from .vllm_client import strip_reasoning

logger = logging.getLogger(__name__)

_TOKEN_REFRESH_INTERVAL = 45 * 60  # seconds
_TOKEN_REFRESH_CMD = [
    "az", "account", "get-access-token",
    "--resource=https://cognitiveservices.azure.com/",
    "--query", "accessToken",
    "--output", "tsv",
]


def _fetch_azure_token() -> Optional[str]:
    """Run ``az`` CLI to get a fresh Azure AD token.  Returns None on failure."""
    try:
        result = subprocess.run(
            _TOKEN_REFRESH_CMD,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            token = result.stdout.strip()
            if token:
                return token
        logger.warning("az token refresh failed (exit %d): %s",
                       result.returncode, result.stderr.strip())
    except Exception:
        logger.warning("az token refresh command failed", exc_info=True)
    return None


class AzureClient(LLMClient):
    """Client for Azure OpenAI Service.

    Args:
        azure_endpoint: Azure OpenAI endpoint URL.
        api_key: Azure OpenAI API key.
        model: Azure deployment name.
        api_version: Azure API version string.
    """

    def __init__(
        self,
        azure_endpoint: str = "",
        api_key: str = "",
        model: str = "gpt-4o",
        api_version: str = "2024-12-01-preview",
    ):
        api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self.client = OpenAI(
            base_url=azure_endpoint,
            api_key=api_key,
            #api_version=api_version,
        )
        self.model = model
        self._start_token_refresh()

    def _refresh_token(self):
        """Fetch a fresh Azure AD token and update the client and environment."""
        token = _fetch_azure_token()
        if token:
            self.client.api_key = token
            os.environ["AZURE_OPENAI_API_KEY"] = token
            logger.info("Azure AD token refreshed")
        return token

    def _start_token_refresh(self):
        """Fetch a token immediately, then start a daemon thread to refresh every 45 min."""
        self._refresh_token()

        def _loop():
            while True:
                time.sleep(_TOKEN_REFRESH_INTERVAL)
                self._refresh_token()

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        logger.info("Azure token refresh thread started (every %d min)",
                     _TOKEN_REFRESH_INTERVAL // 60)

    def generate(self, prompt: str, system: str = "", max_tokens: int = 8000, temperature: float = 0.0) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "input": prompt,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system:
            kwargs["instructions"] = system

        response = self.client.responses.create(**kwargs)

        text = response.output_text or ""
        text = strip_reasoning(text)

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            }

        return LLMResponse(text=text, usage=usage, raw=response)

    def generate_structured(self, prompt: str, system: str = "", max_tokens: int = 8000, temperature: float = 0.0) -> LLMResponse:
        return self.generate(prompt, system, max_tokens, temperature)


def create_azure_client_from_config(llm_config) -> AzureClient:
    """Create an AzureClient from an LLMConfig dataclass."""
    return AzureClient(
        azure_endpoint=llm_config.base_url or "",
        api_key=llm_config.resolve_api_key(),
        model=llm_config.model,
        api_version=llm_config.azure_api_version,
    )

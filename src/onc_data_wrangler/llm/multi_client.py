"""Round-robin load-balancing LLM client across multiple vLLM backends."""

import threading

from .base import LLMClient, LLMResponse
from .vllm_client import VLLMClient


class MultiVLLMClient(LLMClient):
    """Distributes requests across multiple ``VLLMClient`` instances.

    Uses a thread-safe round-robin counter so concurrent extraction
    workers are spread evenly across servers.
    """

    def __init__(self, clients: list[VLLMClient]):
        if not clients:
            raise ValueError("clients list must not be empty")
        self._clients = clients
        self._counter = 0
        self._lock = threading.Lock()

    @classmethod
    def from_base_urls(cls, base_urls: list[str], model: str, api_key: str = "none") -> "MultiVLLMClient":
        """Construct from a list of base URLs (e.g. ``http://host:port/v1``)."""
        clients = [VLLMClient(base_url=url, model=model, api_key=api_key) for url in base_urls]
        return cls(clients)

    def _next_client(self) -> VLLMClient:
        with self._lock:
            idx = self._counter % len(self._clients)
            self._counter += 1
        return self._clients[idx]

    def generate(self, prompt: str, system: str = "", max_tokens: int = 8000, temperature: float = 0.0) -> LLMResponse:
        return self._next_client().generate(prompt, system, max_tokens, temperature)

    def generate_structured(self, prompt: str, system: str = "", max_tokens: int = 8000, temperature: float = 0.0) -> LLMResponse:
        return self._next_client().generate_structured(prompt, system, max_tokens, temperature)

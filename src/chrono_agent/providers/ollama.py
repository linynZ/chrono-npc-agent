"""Ollama backend — the local side of the comparison.

Ollama exposes an OpenAI-compatible route at `/v1`, so this is configuration
rather than code. The timeout default is deliberately far more generous than the
cloud one: a 7B model on a laptop GPU is routinely several seconds per turn, and
using the cloud budget here would make every local run look like a timeout
rather than what it is — a slower model.

Whether a local 7B holds up on tool calls is exactly the open question this
project is meant to answer with numbers instead of opinion.
"""

from __future__ import annotations

import httpx

from .openai_compat import OpenAICompatibleProvider

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b"


class OllamaProvider(OpenAICompatibleProvider):
    name = "ollama"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_ms: int = 30000,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        # Ollama ignores the key but the OpenAI dialect wants the header present.
        super().__init__(
            base_url=base_url,
            model=model,
            api_key="ollama",
            timeout_ms=timeout_ms,
            client=client,
        )

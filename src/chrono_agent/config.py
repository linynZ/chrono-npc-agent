"""Settings and the provider factory.

Reads `.env` if present; real environment variables win over the file so CI and
one-off overrides work without editing anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .providers import DeepSeekProvider, EchoProvider, LLMProvider, OllamaProvider

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHARACTERS_DIR = PROJECT_ROOT / "characters"

_loaded = False


def load_env(override: bool = False) -> None:
    """Load `.env` once. Idempotent — safe to call from anywhere."""
    global _loaded
    if _loaded and not override:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=override)
    _loaded = True


@dataclass(frozen=True)
class Settings:
    provider: str = "deepseek"
    timeout_ms: int = 4000

    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_base_url: str = "https://api.deepseek.com"

    ollama_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434/v1"

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Accept either the bare host or the full OpenAI-compatible route.
        if not ollama_base.rstrip("/").endswith("/v1"):
            ollama_base = ollama_base.rstrip("/") + "/v1"

        return cls(
            provider=os.getenv("AGENT_PROVIDER", "deepseek").strip().lower(),
            timeout_ms=int(os.getenv("AGENT_TIMEOUT_MS", "4000")),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip(),
            deepseek_base_url=os.getenv(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).strip(),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip(),
            ollama_base_url=ollama_base,
        )


def build_provider(settings: Settings | None = None, name: str | None = None) -> LLMProvider:
    """Construct the configured backend. `name` overrides the setting."""
    settings = settings or Settings.from_env()
    choice = (name or settings.provider).strip().lower()

    if choice == "deepseek":
        return DeepSeekProvider(
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            base_url=settings.deepseek_base_url,
            timeout_ms=settings.timeout_ms,
        )
    if choice == "ollama":
        return OllamaProvider(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            # A local model is slower by nature; judging it against the cloud
            # budget would report model latency as infrastructure failure.
            timeout_ms=max(settings.timeout_ms, 30000),
        )
    if choice == "echo":
        return EchoProvider()

    raise ValueError(
        f"Unknown AGENT_PROVIDER {choice!r}. Expected one of: deepseek, ollama, echo."
    )

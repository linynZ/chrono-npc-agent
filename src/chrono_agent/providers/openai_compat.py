"""Shared implementation for any backend speaking the OpenAI chat dialect.

DeepSeek and Ollama both do, so the real HTTP work lives here exactly once and
the two concrete providers are just configuration. That is also what makes the
cloud-vs-local comparison honest: both go through identical request-building,
identical timeout handling and identical usage accounting, so a difference in
the results is a difference in the model, not in my plumbing.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..models import Completion, Message, Usage
from .base import (
    LLMProvider,
    ProviderError,
    ProviderTimeout,
    message_from_wire,
    message_to_wire,
)


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_ms: int = 4000,
        client: httpx.AsyncClient | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._timeout_s = timeout_ms / 1000
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=self._timeout_s)
        # Vendor-specific request fields merged into every payload. Kept generic
        # so the shared implementation stays vendor-neutral.
        self._extra_body = dict(extra_body or {})

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message_to_wire(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        payload.update(self._extra_body)

        started = time.perf_counter()
        try:
            response = await self._client.post(
                self._endpoint(),
                json=payload,
                headers=self._headers(),
                timeout=self._timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(
                f"{self.name} exceeded {self._timeout_s * 1000:.0f}ms budget"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} transport error: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code >= 400:
            # Truncated: provider error bodies can be long, and this ends up in logs.
            raise ProviderError(
                f"{self.name} HTTP {response.status_code}: {response.text[:300]}"
            )

        try:
            body = response.json()
            choice = body["choices"][0]
        except (ValueError, KeyError, IndexError) as exc:
            raise ProviderError(f"{self.name} returned an unusable body: {exc}") from exc

        raw_message = choice.get("message", {})

        # A reasoning model can spend the whole output budget thinking and return
        # an empty `content` with the thoughts in `reasoning_content`. Downstream
        # that looks like an empty reply and silently falls back, which hides the
        # actual cause. Fail loudly instead — the remedy is a request flag, not a
        # retry.
        if not (raw_message.get("content") or "").strip() and raw_message.get(
            "reasoning_content"
        ):
            raise ProviderError(
                f"{self.name} spent the output budget on reasoning and returned no "
                f"content (finish_reason={choice.get('finish_reason')!r}). Disable "
                f"thinking mode or raise max_tokens."
            )

        return Completion(
            message=message_from_wire(raw_message),
            usage=self._parse_usage(body.get("usage") or {}),
            latency_ms=latency_ms,
            model=body.get("model", self.model),
            finish_reason=choice.get("finish_reason", ""),
        )

    @staticmethod
    def _parse_usage(raw: dict[str, Any]) -> Usage:
        # Cache-hit accounting is a vendor extension and the field name has moved
        # around, so try the known spellings and fall back to zero.
        cached = (
            raw.get("prompt_cache_hit_tokens")
            or raw.get("cached_tokens")
            or (raw.get("prompt_tokens_details") or {}).get("cached_tokens")
            or 0
        )
        return Usage(
            prompt_tokens=raw.get("prompt_tokens", 0),
            completion_tokens=raw.get("completion_tokens", 0),
            cached_tokens=cached,
        )

    async def aclose(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

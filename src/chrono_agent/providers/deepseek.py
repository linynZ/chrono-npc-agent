"""DeepSeek backend — the cloud side of the comparison.

Chosen for tool-call reliability and price: `deepseek-v4-flash` bills roughly
1 CNY per million input tokens (0.02 on a cache hit) and 2 per million output,
so a full guardrail sweep costs small change. Note the docs currently warn that
pricing is about to rise — the numbers above are what they were when this was
written, not a promise.

**Thinking mode is on by default and is switched off here.** See
`docs/findings/02-thinking-mode.md`; the short version is that an NPC line is a
two-sentence in-character reply, reasoning tokens bill at the output rate, and
the latency they add comes straight out of the budget the player is waiting on.
`temperature` is also inert while thinking is enabled, so leaving it on would
have meant every dial in this codebase was quietly doing nothing.
"""

from __future__ import annotations

from typing import Any

import httpx

from .openai_compat import OpenAICompatibleProvider

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

THINKING_DISABLED: dict[str, Any] = {"thinking": {"type": "disabled"}}
THINKING_ENABLED: dict[str, Any] = {"thinking": {"type": "enabled"}}


class DeepSeekProvider(OpenAICompatibleProvider):
    name = "deepseek"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_ms: int = 4000,
        client: httpx.AsyncClient | None = None,
        thinking: bool = False,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is empty. Copy .env.example to .env and fill it in, "
                "or set AGENT_PROVIDER=echo to run without a key."
            )

        body = dict(THINKING_ENABLED if thinking else THINKING_DISABLED)
        body.update(extra_body or {})

        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_ms=timeout_ms,
            client=client,
            extra_body=body,
        )

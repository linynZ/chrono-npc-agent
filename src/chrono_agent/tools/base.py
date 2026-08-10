"""Tool registry and the execution context.

Tools are plain functions with a JSON-Schema signature. The registry exports
them in the OpenAI `tools` format — which is what DeepSeek and Ollama both
expect — and executes what the model asks for.

Two rules the tool layer enforces, both because a model cannot be trusted to:

1. A tool the NPC is not granted is not merely refused, it is never advertised.
   Unlisted tools do not appear in the schema, so the model has nothing to call.
2. Tool results are data the NPC *perceives*, never raw save state. What comes
   back is already scoped to what this character could know.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ..models import Language, PlayerState


@dataclass
class ToolContext:
    """Everything a tool is allowed to read when it runs."""

    state: PlayerState
    language: Language = "zh"
    quests: dict[str, Any] = field(default_factory=dict)
    lore: list[dict[str, Any]] = field(default_factory=list)
    npc_id: str = ""


ToolFunc = Callable[..., Any]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: ToolFunc

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, context: ToolContext, arguments: dict[str, Any]) -> str:
        """Execute and return a string for the model. Never raises."""
        try:
            signature = inspect.signature(self.func)
            # Drop anything the model hallucinated that we do not accept.
            accepted = {
                key: value
                for key, value in arguments.items()
                if key in signature.parameters
            }
            result = self.func(context, **accepted)
        except TypeError as exc:
            # Usually a missing required argument. Telling the model exactly what
            # went wrong is far more effective than a generic failure.
            return json.dumps(
                {"error": f"bad arguments for {self.name}: {exc}"}, ensure_ascii=False
            )
        except Exception as exc:  # noqa: BLE001 - a tool must never kill the turn
            return json.dumps(
                {"error": f"{self.name} failed: {exc}"}, ensure_ascii=False
            )

        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def subset(self, names: list[str]) -> "ToolRegistry":
        """The tools one NPC is allowed. Unknown names are ignored, not an error —
        a persona file may reference a tool that has not been built yet."""
        return ToolRegistry([self._tools[n] for n in names if n in self._tools])

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, context: ToolContext, name: str, arguments: dict[str, Any]) -> str:
        tool = self.get(name)
        if tool is None:
            return json.dumps(
                {
                    "error": f"no such tool: {name}",
                    "available": sorted(self._tools),
                },
                ensure_ascii=False,
            )
        return tool.run(context, arguments)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

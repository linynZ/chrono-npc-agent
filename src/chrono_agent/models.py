"""Domain models shared by the whole agent.

`PlayerState` mirrors a subset of ChronoTraveler's `SaveData` — only the fields
an NPC could plausibly know about. Field names follow the save file so the Unity
side can serialise straight into this without a translation layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

Language = Literal["zh", "en"]


class QuestStage(BaseModel):
    """Where the player is in an era's main quest."""

    quest_id: str
    stage_index: int = 0
    stage_description: str = ""
    objective_label: str = ""
    progress: int = 0
    required: int = 1

    @property
    def is_complete(self) -> bool:
        return self.progress >= self.required


class PlayerState(BaseModel):
    """What the NPC is allowed to know about the player right now.

    Everything here comes from the save file. Nothing is inferred, and nothing
    the player has not actually done is ever present — an NPC cannot leak future
    plot because the state simply does not contain it.
    """

    current_map: str = "20_China"
    language: Language = "zh"

    # Memory restoration, 0.0 .. 1.0 (mapMemory points / the map's memoryTarget).
    memory_progress: float = Field(default=0.0, ge=0.0, le=1.0)

    fragments_collected: int = 0
    fragments_required: int = 3

    purged_encounters: list[str] = Field(default_factory=list)
    talked_npcs: list[str] = Field(default_factory=list)
    earned_tokens: list[str] = Field(default_factory=list)

    player_level: int = 1
    correct_answers: int = 0
    total_answers: int = 0

    quest: QuestStage | None = None

    @property
    def accuracy(self) -> float | None:
        """Quiz accuracy so far, or None if the player has not answered anything."""
        if self.total_answers <= 0:
            return None
        return self.correct_answers / self.total_answers

    @property
    def has_met(self) -> bool:
        """Convenience for `is this NPC a stranger` checks — set per-NPC by the agent."""
        return bool(self.talked_npcs)

    def progress_variant(self) -> str:
        """Map continuous progress onto the game's four pre-written variants.

        The shipped game only has four buckets (base / mid / gate / post). We
        keep the mapping so fallback lines pick a sensible pre-written reply, but
        the agent itself sees the raw continuous state.
        """
        if self.earned_tokens:
            return "post"
        if self.memory_progress >= 1.0:
            return "gate"
        if self.memory_progress >= 0.5:
            return "mid"
        return "base"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A tool invocation the model asked for."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Set on TOOL messages to link the result back to the request.
    tool_call_id: str | None = None
    name: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Completion(BaseModel):
    """One raw round-trip to a model. Providers return this and nothing else."""

    message: Message
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    model: str = ""
    finish_reason: str = ""


class ReplySource(str, Enum):
    """Where the line the player sees actually came from.

    This is the metric that matters most for the whole project: how often did we
    get a real generated reply, and how often did we have to fall back?
    """

    MODEL = "model"
    FALLBACK_TIMEOUT = "fallback_timeout"
    FALLBACK_ERROR = "fallback_error"
    FALLBACK_GUARDRAIL = "fallback_guardrail"


class NpcReply(BaseModel):
    """What the game gets back. Deliberately small — Unity only needs the text."""

    text: str
    source: ReplySource = ReplySource.MODEL
    latency_ms: float = 0.0
    tool_calls_made: list[str] = Field(default_factory=list)
    guardrail_flags: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

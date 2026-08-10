"""What the player hears when the model does not deliver.

The game already ships hand-written lines for every NPC at four progress
buckets. When the model times out, errors, or trips an output guardrail, we
serve one of those instead. The player sees dialogue that is a little less
responsive — not a spinner, not an apology, and never an error string.

Selection is deterministic: the same player message in the same state always
yields the same line. That is a testing requirement, not a design preference —
a random fallback makes the failure paths unassertable. It also happens to read
fine, since the player has no way to notice the mapping.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import DATA_DIR
from .models import Language, PlayerState
from .persona import NpcPersona, unfold_cjk


class FallbackLibrary:
    """The game's pre-written lines, indexed by NPC and progress bucket."""

    def __init__(self, npc_lines: dict) -> None:
        self._lines = npc_lines

    @classmethod
    def load(cls, path: Path | None = None) -> "FallbackLibrary":
        path = path or DATA_DIR / "npc_lines.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing. Run: python scripts/extract_game_data.py"
            )
        with path.open(encoding="utf-8") as fh:
            return cls(json.load(fh))

    def has(self, npc_id: str) -> bool:
        return npc_id in self._lines

    def lines_for(
        self, npc_id: str, variant: str, language: Language
    ) -> list[str]:
        npc = self._lines.get(npc_id)
        if not npc:
            return []
        variants = npc.get("variants", {})
        # Walk back toward `base` if this bucket has nothing — a late-game
        # player should still hear something rather than fall to last resort.
        for candidate in _variant_chain(variant):
            block = variants.get(candidate) or []
            texts = [
                (entry.get("text") or {}).get(language, "").strip() for entry in block
            ]
            texts = [t for t in texts if t]
            if texts:
                return texts
        return []

    def pick(
        self,
        persona: NpcPersona,
        state: PlayerState,
        seed: str,
        language: Language | None = None,
    ) -> str:
        """Choose a pre-written line for this NPC, state and player message."""
        language = language or state.language
        candidates = self.lines_for(
            persona.npc_id, state.progress_variant(), language
        )
        if not candidates:
            last = persona.fallback.last_resort.get(language, "")
            return unfold_cjk(last.strip())

        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        return candidates[digest[0] % len(candidates)]


def _variant_chain(variant: str) -> list[str]:
    """Preference order when a bucket is empty."""
    order = ["post", "gate", "mid", "base"]
    if variant not in order:
        return ["base"]
    # Try the exact bucket, then progressively earlier ones.
    return order[order.index(variant):]

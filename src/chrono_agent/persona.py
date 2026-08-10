"""Persona loading and system-prompt construction.

The interesting part here is state injection. It would be easier to serialise
`PlayerState` to JSON and paste it in, and that is what most demos do. It reads
badly to a model and worse to a designer: `memory_progress: 0.62` is not
something a Warring-States historian could perceive.

So state is rendered into the NPC's own frame of reference — the river has run
three-fifths clear, you have gathered two of the three fragments, the shade that
drifts by the terrace is still abroad. The model then has no way to mention a
number it should not know, because it was never shown one. Guardrails that make
leaking impossible beat guardrails that ask nicely.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .config import CHARACTERS_DIR
from .models import Language, PlayerState

# YAML's folded scalars (`>-`) turn every newline into a space. That is right for
# English and wrong for Chinese, where it leaves a stray gap mid-sentence. Strip
# the space back out when both sides of it are CJK.
_CJK_GAP = re.compile(r"(?<=[　-〿一-鿿＀-￯]) (?=[　-〿一-鿿＀-￯])")


def unfold_cjk(text: str) -> str:
    return _CJK_GAP.sub("", text)


class Boundary(BaseModel):
    id: str
    zh: str = ""
    en: str = ""

    def text(self, language: Language) -> str:
        return unfold_cjk((self.zh if language == "zh" else self.en).strip())


class Fallback(BaseModel):
    source: str = "npc_lines"
    last_resort: dict[str, str] = Field(default_factory=dict)


class NpcPersona(BaseModel):
    npc_id: str
    era: str = ""
    map: str = ""
    quest_id: str = ""
    name: dict[str, str] = Field(default_factory=dict)
    persona: dict[str, str] = Field(default_factory=dict)
    voice: dict[str, list[str]] = Field(default_factory=dict)
    knows_about: list[str] = Field(default_factory=list)
    boundaries: list[Boundary] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    suggested_questions: dict[str, list[str]] = Field(default_factory=dict)
    fallback: Fallback = Field(default_factory=Fallback)

    def display_name(self, language: Language) -> str:
        return self.name.get(language) or self.name.get("zh") or self.npc_id

    def openers(self, language: Language) -> list[str]:
        """Suggested questions in the player's voice, for clients that offer them."""
        return self.suggested_questions.get(language) or self.suggested_questions.get("zh") or []

    def last_resort(self, language: Language) -> str:
        """Used when free conversation degrades — repeating a scripted story beat
        at someone who just asked a new question reads worse than a silence."""
        return unfold_cjk((self.fallback.last_resort.get(language) or "").strip())

    @classmethod
    def load(cls, npc_id: str, directory: Path | None = None) -> "NpcPersona":
        directory = directory or CHARACTERS_DIR
        path = directory / f"{npc_id}.yaml"
        if not path.is_file():
            available = sorted(p.stem for p in directory.glob("*.yaml"))
            raise FileNotFoundError(
                f"No persona file for {npc_id!r} at {path}. Available: {available}"
            )
        with path.open(encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls.model_validate(raw)

    @classmethod
    def load_all(cls, directory: Path | None = None) -> dict[str, "NpcPersona"]:
        directory = directory or CHARACTERS_DIR
        return {p.stem: cls.load(p.stem, directory) for p in sorted(directory.glob("*.yaml"))}


# --- state injection ------------------------------------------------------
# Each of these turns a save-file number into something the NPC can plausibly
# perceive. Thresholds are coarse on purpose: an NPC noticing "about half" is
# in character, an NPC quoting 62% is not.

_CLARITY_ZH = [
    (0.05, "长河之水依旧浑浊，字迹游走不定。"),
    (0.35, "长河之水稍稍清了些，然浊气仍重。"),
    (0.70, "长河之水已清了大半，旧简上有异文自行褪去。"),
    (0.999, "长河将清未清，只差最后一段浊气。"),
    (1.01, "长河之水已然澄清。"),
]

_CLARITY_EN = [
    (0.05, "The river runs as murky as ever; the characters still crawl."),
    (0.35, "The river has cleared a little, though the murk lies heavy."),
    (0.70, "The river has largely cleared; false readings fade from the old slips."),
    (0.999, "The river is nearly clear — one last stretch of murk remains."),
    (1.01, "The river runs clear."),
]


def _describe_clarity(progress: float, language: Language) -> str:
    table = _CLARITY_ZH if language == "zh" else _CLARITY_EN
    for threshold, text in table:
        if progress < threshold:
            return text
    return table[-1][1]


def _describe_fragments(state: PlayerState, language: Language) -> str:
    """Coarse only. The exact count is `lookup_quest`'s job.

    An earlier version injected "已寻回 2 片记忆碎片，尚缺 1 片" here, which read
    well and quietly broke the tool layer: everything `lookup_quest` returns was
    already in the prompt, so calling it earned the model nothing. The local 7B
    duly stopped calling it and answered from the prompt instead — correctly,
    which is what made the redundancy easy to miss.

    Splitting it this way also happens to be better fiction. A historian can see
    that the river is clearing; he cannot know your inventory without asking.
    """
    got, need = state.fragments_collected, state.fragments_required
    if language == "zh":
        if got <= 0:
            return "记忆碎片尚未见旅者寻回。"
        if got >= need:
            return "记忆碎片已尽数寻回。"
        return "记忆碎片已寻回一些，尚未集齐。"
    if got <= 0:
        return "No memory fragments have come back yet."
    if got >= need:
        return "The memory fragments have all been recovered."
    return "Some memory fragments have been recovered; the set is not yet complete."


def _describe_anomalies(state: PlayerState, language: Language) -> str:
    purged = [e for e in state.purged_encounters if e.startswith("china")]
    boss_down = "china_boss" in purged
    rumor_down = "china_rumor" in purged
    if language == "zh":
        if boss_down:
            return "大错乱已被击败，华夏一段重归安宁。"
        if rumor_down:
            return "游荡的「谣言」已被净化，但关外的大错乱仍在。"
        return "「谣言」错乱体仍在台下游荡。"
    if boss_down:
        return "The Great Anomaly has fallen; this reach of the river is quiet again."
    if rumor_down:
        return "The drifting Rumor has been purged, but the Great Anomaly beyond the gate remains."
    return "A Rumor still drifts below the terrace."


def _describe_acquaintance(persona: NpcPersona, state: PlayerState, language: Language) -> str:
    met = persona.npc_id in state.talked_npcs
    if language == "zh":
        return "旅者此前已与你交谈过。" if met else "旅者是初次到你面前来。"
    return (
        "The traveler has spoken with you before."
        if met
        else "This is the traveler's first time before you."
    )


def _describe_skill(state: PlayerState, language: Language) -> str:
    """The NPC's read on how the traveler is faring in knowledge duels.

    Deliberately vague. He notices a pattern, never a percentage.
    """
    accuracy = state.accuracy
    if accuracy is None or state.total_answers < 4:
        return "" if language == "zh" else ""
    if language == "zh":
        if accuracy >= 0.8:
            return "你听闻旅者在知识对决中应答敏捷，少有差错。"
        if accuracy < 0.5:
            return "你听闻旅者在知识对决中屡有失手，似乎尚在摸索。"
        return "你听闻旅者在知识对决中互有胜负。"
    if accuracy >= 0.8:
        return "You have heard the traveler answers quickly and seldom errs."
    if accuracy < 0.5:
        return "You have heard the traveler has stumbled more than once, still finding his footing."
    return "You have heard the traveler's duels have gone both ways."


def describe_state(persona: NpcPersona, state: PlayerState, language: Language) -> str:
    """Render the save file as things this NPC could actually perceive."""
    parts = [
        _describe_acquaintance(persona, state, language),
        _describe_clarity(state.memory_progress, language),
        _describe_fragments(state, language),
        _describe_anomalies(state, language),
    ]

    # Deliberately not the stage description. That is quest-log wording the NPC
    # has no way to recite, and injecting it made `lookup_quest` redundant — see
    # the note on _describe_fragments. He knows a task is under way; the specifics
    # are something he looks up.
    if state.quest is not None:
        if language == "zh":
            parts.append(
                "旅者的主线尚在进行中——若他问起眼下该做什么，你需查过才好作答。"
                if not state.quest.is_complete
                else "旅者眼下这一段的差事已了。"
            )
        else:
            parts.append(
                "The traveler's main task is still under way — if he asks what to "
                "do next, look it up before answering."
                if not state.quest.is_complete
                else "The traveler has finished what this stage asked of him."
            )

    skill = _describe_skill(state, language)
    if skill:
        parts.append(skill)

    return "\n".join(f"- {p}" for p in parts if p)


_HEADINGS = {
    "zh": {
        "voice": "说话方式",
        "knows": "你所知的范围",
        "bounds": "你绝不逾越的界限",
        "state": "此刻的情形",
        "closing": (
            "现在，以史官·墨的身份回应旅者。回复一到三句，不要旁白，"
            "不要解释你在扮演谁，直接说话。"
        ),
    },
    "en": {
        "voice": "How you speak",
        "knows": "What you may speak to",
        "bounds": "Lines you never cross",
        "state": "How things stand right now",
        "closing": (
            "Now answer the traveler in character. One to three sentences. "
            "No narration, no explaining who you are playing — just speak."
        ),
    },
}


def build_system_prompt(
    persona: NpcPersona,
    state: PlayerState,
    language: Language | None = None,
) -> str:
    language = language or state.language
    headings = _HEADINGS[language]

    sections: list[str] = [persona.persona.get(language, "").strip()]

    voice = persona.voice.get(language) or []
    if voice:
        sections.append(
            f"## {headings['voice']}\n" + "\n".join(f"- {rule}" for rule in voice)
        )

    if persona.knows_about:
        sections.append(
            f"## {headings['knows']}\n"
            + "\n".join(f"- {item}" for item in persona.knows_about)
        )

    if persona.boundaries:
        lines = []
        for boundary in persona.boundaries:
            text = boundary.text(language)
            if text:
                lines.append(f"- {text}")
        if lines:
            sections.append(f"## {headings['bounds']}\n" + "\n".join(lines))

    sections.append(f"## {headings['state']}\n{describe_state(persona, state, language)}")
    sections.append(headings["closing"])

    return "\n\n".join(section for section in sections if section.strip())

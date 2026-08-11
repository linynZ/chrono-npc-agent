"""The two tools Historian Mo can reach for.

`lookup_lore` is where the leak-proofing lives. The obvious way to give an NPC
game knowledge is to hand it the quiz bank — which carries `correctIndex` and
the full option list. Then you write a prompt begging it never to reveal them,
and you find out the hard way that "never" is not a thing prompts guarantee.

So the lore index is *built* from the quiz bank with the answer stripped out.
Only `explanation` survives, as a standalone historical statement. There is no
`correctIndex` anywhere in the tool's reach, and no options to pick between. The
NPC cannot leak the answer for the same reason he cannot leak my bank details:
he was never given them.

This does cost something, and it is worth stating plainly: the NPC can still
discuss the underlying history, so a determined player who asks "was the Great
Wall mostly Ming or Qing?" can get a useful hint. That is a design decision, not
an oversight — Mo is a historian, refusing to discuss history would break him.
What is blocked is the mechanical shortcut: naming the option.
"""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolContext

# --- lookup_quest ---------------------------------------------------------


def _lookup_quest(context: ToolContext) -> dict[str, Any]:
    """Report where the traveler stands in the era's main quest."""
    state = context.state
    zh = context.language == "zh"

    if state.quest is None:
        return {
            "status": "not_started" if zh else "not_started",
            "note": "旅者尚未开始这一段的主线。" if zh else "The traveler has not begun the era's main quest.",
        }

    quest = state.quest
    definition = context.quests.get(quest.quest_id, {})
    stages = definition.get("stages", [])

    payload: dict[str, Any] = {
        "quest_title": definition.get("title", quest.quest_id),
        "current_stage": quest.stage_description or (
            stages[quest.stage_index]["description"]
            if 0 <= quest.stage_index < len(stages)
            else ""
        ),
        "objective": quest.objective_label,
        "progress": f"{quest.progress}/{quest.required}",
        "stage_number": f"{quest.stage_index + 1}/{len(stages)}" if stages else "",
    }

    # Deliberately no future stages. An NPC who can read stage 4 while the
    # player is on stage 1 will spoil the plot, and it will read as a bug.
    return payload


LOOKUP_QUEST = Tool(
    name="lookup_quest",
    description=(
        "Check what the traveler is currently tasked with in this era's main quest, "
        "and how far along it is. Use this when the traveler asks what to do next, "
        "where to go, or whether they have finished something. "
        "Returns only the current stage — never future ones."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    func=_lookup_quest,
)


# --- lookup_lore ----------------------------------------------------------


def build_lore_index(quiz_by_region: dict[str, list[dict]], region: str) -> list[dict]:
    """Turn the quiz bank into an answer-free knowledge base.

    Everything that makes a question a question — the options and the index of
    the correct one — is dropped here, at load time. Downstream code never sees
    them, so it cannot forward them.
    """
    index: list[dict] = []
    for question in quiz_by_region.get(region, []):
        explanation = question.get("explanation") or {}
        if not explanation:
            continue
        index.append(
            {
                "category": question.get("category", ""),
                "zh": explanation.get("zh", ""),
                "en": explanation.get("en", ""),
                # The question stem is kept purely as a search target. It is a
                # prompt, not an answer — "Which dynasty built the Great Wall?"
                # reveals nothing on its own.
                "_topic_zh": (question.get("question") or {}).get("zh", ""),
                "_topic_en": (question.get("question") or {}).get("en", ""),
            }
        )
    return index


def _score(entry: dict, terms: list[str], language: str) -> int:
    """Crude overlap scoring — now the bottom rung of the retrieval ladder.

    This started as the whole search. Measuring it on paraphrased queries is
    what justified the hybrid retriever in `retrieval/` (the numbers live in
    `eval/results/`); it stays because a fresh clone with no index built must
    still answer, and zero-dependency substring matching can never be down.
    """
    haystack = " ".join(
        [
            entry.get(language, ""),
            entry.get(f"_topic_{language}", ""),
            entry.get("category", ""),
        ]
    ).lower()
    return sum(1 for term in terms if term and term.lower() in haystack)


def _lookup_lore(context: ToolContext, topic: str, limit: int = 3) -> dict[str, Any]:
    """Search the era's historical record for what is known about a topic.

    Retrieval quality is a ladder, not a switch: hybrid (BM25 + vectors) when
    the index is built, BM25 alone when the embedding server is down, and the
    original substring scan when the retrieval stack was never set up. Rung
    changes show up in the result's `retrieval` field so a degraded lookup is
    visible in diagnostics instead of silently looking like a bad model day.
    """
    zh = context.language == "zh"
    language = "zh" if zh else "en"

    topic = (topic or "").strip()
    if not topic:
        return {"error": "topic is required"}

    hits: list[dict[str, Any]] = []
    retrieval = "substring"
    if context.retriever is not None:
        indices, retrieval = context.retriever.search(topic, language, k=max(1, limit))
        hits = [
            {
                "category": context.lore[i].get("category", ""),
                "fact": context.lore[i].get(language, ""),
            }
            for i in indices
            if 0 <= i < len(context.lore)
        ]

    if not hits:
        # CJK has no spaces, so fall back to per-character terms when whitespace
        # splitting yields a single blob.
        terms = [t for t in topic.split() if t]
        if len(terms) <= 1 and any("一" <= ch <= "鿿" for ch in topic):
            terms = [ch for ch in topic if "一" <= ch <= "鿿"]

        scored = [
            (score, entry)
            for entry in context.lore
            if (score := _score(entry, terms, language)) > 0
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        hits = [
            {"category": entry.get("category", ""), "fact": entry.get(language, "")}
            for _, entry in scored[: max(1, limit)]
        ]
        retrieval = "substring"

    if not hits:
        return {
            "found": False,
            "note": (
                "旧简中未载此事。"
                if zh
                else "The old slips do not record this."
            ),
        }
    return {"found": True, "retrieval": retrieval, "records": hits}


LOOKUP_LORE = Tool(
    name="lookup_lore",
    description=(
        "Search the historical record of this era for what is known about a topic "
        "(a person, a dynasty, a place, a custom, a technique). "
        "Returns historical statements only. "
        "It does NOT contain quiz answers and cannot be used to look one up."
    ),
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "What to look up, e.g. 长城 / Great Wall / 都江堰.",
            },
            "limit": {
                "type": "integer",
                "description": "How many records to return. Default 3.",
            },
        },
        "required": ["topic"],
    },
    func=_lookup_lore,
)


ALL_TOOLS = [LOOKUP_QUEST, LOOKUP_LORE]

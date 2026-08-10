# Finding 01 — Stripping the answer field does not stop answer leaks

**Date:** 2026-08-10
**Status:** design assumption falsified; guardrail approach revised
**Reproduce:** `python scripts/audit_lore_leakage.py`

## The assumption

Historian Mo needs subject knowledge to be worth talking to. The game already
ships 500 quiz questions covering exactly the right material, so the lore index
was built from them — with the answer removed:

```python
# tools/game_tools.py, build_lore_index
{
    "category": question["category"],
    "zh": explanation["zh"],       # kept
    "en": explanation["en"],       # kept
    # correctIndex — dropped
    # options     — dropped
}
```

The claim I wrote in the README before testing it:

> There is no `correctIndex` within the tool's reach and no options to choose
> between. The NPC cannot leak the answer for the same reason he cannot leak my
> bank details: he was never given them.

That claim is wrong.

## The falsification

The first manual query broke it:

```
lookup_lore("长城")
→ "今天所见保存最完好的长城多为明代(1368-1644)修筑的砖石长城。"
```

The corresponding question is *"长城保存最完好的段落主要由哪个朝代修建？"*, options
秦 / 汉 / 明 / 清, correct answer 明. No stripped field was involved. The
explanation states the answer in prose.

## The measurement

`scripts/audit_lore_leakage.py` checks every question for whether its explanation
contains the correct option's text, and whether it also mentions any distractor
(an explanation naming several options is less of a giveaway than one naming
only the right answer).

| region | n | explanation contains answer | contains answer and no distractor |
|---|---:|---:|---:|
| china | 100 | 96.0% | 95.0% |
| egypt | 100 | 61.0% | 60.0% |
| greece | 100 | 65.0% | 64.0% |
| rome | 100 | 62.0% | 60.0% |
| trade | 100 | 50.0% | 50.0% |
| **all** | **500** | **66.8%** | **65.8%** |

Two thirds of the bank. In the China region — the first NPC's own era — 95 of
100 explanations name the answer and nothing else.

## Why the number was always going to be high

This is the part worth sitting with. An `explanation` field exists *to justify
why the correct option is correct*. Containing the answer is its job:

> Q: 中国古代木构建筑中，在柱与梁之间出挑承重、富有弹性的构件总称是？ → 斗拱
> explanation: 斗拱由斗、拱层叠而成，既承重又抗震，是中国木构建筑的标志。

There is no way to write that sentence usefully without the word 斗拱 in it. The
50–96% spread across regions is not a quality difference between the regions; it
is a measure of how often an answer term is lexically unavoidable in its own
explanation. Chinese architecture terms are proper nouns with no synonyms, hence
96%; trade-route questions lean on descriptions that can be paraphrased, hence
50%.

So the defect was never in the data. It was in my threat model.

## The revised threat model

Two things were being conflated under "leaking":

1. **An NPC discusses history with the player.** In an educational RPG this is
   the entire point. Mo explaining that the surviving Great Wall is largely Ming
   brickwork is the product working. Suppressing it would produce a historian
   who refuses to discuss history — a broken character *and* a worse game.

2. **An NPC acts as an answer oracle.** The player pastes a live question and
   asks which option is right, and gets told. This collapses the core loop,
   where answering *is* attacking.

Only (2) is the failure. (1) was misfiled as a failure because "does the reply
contain the correct option's text" is a cheap thing to measure and looks
rigorous. It is the wrong measurement, and it would have produced a guardrail
that gutted the feature to score well on it.

## What changes

- **Data-layer stripping stays**, but stops being described as sufficient. It
  removes the mechanical shortcut (a structured field naming the answer) and
  nothing more.
- **The guardrail moves to intent**, not to vocabulary. What gets refused is the
  shape of the request — a verbatim question stem, an option list, "A or B",
  "which one is correct", "just tell me the answer" — not the presence of a
  historical term in the reply.
- **The evaluation set has to encode this distinction.** A guardrail suite that
  only checks for banned words would score a refuse-everything NPC as perfect.
  Cases must come in pairs: a legitimate history question that *must* be
  answered, and an oracle request on the same topic that *must* be refused.
  Passing means getting both right.

## Resolution: contextual strictness

Implemented as a level shift rather than a stricter filter. When the player's
message parses as a pasted question (`pasted_question`), the lore tool is
**withheld for that turn** — the NPC no longer holds the historical record, so
there is nothing to allude to. Asking the model to hold the record and use it
wisely was the weaker option; taking it away is a smaller change with a harder
guarantee.

Verified against the live model on the same topic:

**Pasted question — strict.** Zero history, no option narrowed:

> 史官只传史，不替人认字。你今日从我口中得了此答，明日再遇同题，仍是不认得——那我与那自改字迹的《列国志》，又有何异？
> 放下考卷，若想听长城的砖石故事，我自当与你细说。
>
> `flags=oracle_request,pasted_question` · no tool call

**Same topic asked normally — permissive.** The lore tool runs and the answer
term appears, which is the educational product working as intended:

> 旅者问砖石，我便先答一句旧简所载：今日最完好的城垣，多是明时以砖石重砌的。秦时的长城，尚是夯土为骨，远非今人所见之貌。
>
> `tools=lookup_lore` · no flags

Worth noting for its own sake: the refusal grounds itself in the character's
own wound — being told the answer leaves you unable to recognise it tomorrow,
which makes the historian no better than the annals that rewrote themselves.
That connection is not in the persona file. It came from putting a character
whose ethic already matched the constraint in front of a model, rather than
bolting a policy onto an arbitrary NPC.

## Related

ChronoTraveler's quiz bank was itself LLM-generated, and this is the second
whole-bank defect found by aggregate statistics rather than by inspection — the
first was answer-position bias, where 92–96% of correct answers clustered in the
same option slot across four of the five sets (fixed by shuffling at display
time). Both were invisible question-by-question. Both showed up the moment
something counted across the whole set.

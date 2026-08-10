# Finding 02 — The model was picked on the wrong three criteria

**Date:** 2026-08-10
**Status:** fixed — thinking mode disabled by default in `DeepSeekProvider`
**Reproduce:** `python scripts/check_key.py` (before/after the `thinking` flag)

## What happened

`deepseek-v4-flash` was chosen on price, tool-call support and context length.
The first real call came back like this:

```json
{
  "message": {
    "content": "",
    "reasoning_content": "我们只需要一句话，不超过20字。回答“编年长河”是什么？可能指..."
  },
  "finish_reason": "length",
  "usage": { "completion_tokens": 64, "completion_tokens_details": { "reasoning_tokens": 64 } }
}
```

Every output token went to reasoning. `content` was empty. The model is a
reasoning model with **thinking mode on by default**, and nothing in the price
table or the tool-call docs says so.

## Why it mattered more than it looked

An empty reply is caught by the output guardrail and falls back to a written
line, so the player would never have seen a crash. That is exactly the problem:
the failure was *invisible*. The fallback rate would have sat near 100% and
looked like a flaky network.

Three separate costs, none of them obvious from the symptom:

1. **Billing.** Reasoning tokens bill at the output rate — 2 CNY/M on this
   model. Thinking about a two-sentence NPC line is paid-for overhead.
2. **Latency.** It comes straight out of the budget the player is standing there
   waiting on.
3. **`temperature` is inert while thinking is enabled.** Every temperature value
   in this codebase was doing nothing. This one is the worst of the three,
   because it fails silently and forever — no error, no empty reply, just dials
   that are not connected to anything.

## The fix

`{"thinking": {"type": "disabled"}}` in the request body. The provider now sends
it by default and exposes `thinking=True` for when the comparison is wanted.

Same prompt, same model, measured by `check_key.py`:

| | thinking on (default) | thinking off |
|---|---|---|
| reply | *empty* | 编年长河是历史事件按年份顺序的宏大叙事。 |
| total tokens | 161 | **32** |
| latency | 1948 ms | **1242 ms** |

5× the tokens and 1.6× the latency, for a worse answer. For this workload —
short, in-character, persona-constrained dialogue — the reasoning was not merely
unnecessary, it was actively in the way.

A second guard went in alongside the flag: the provider now raises when
`content` is empty but `reasoning_content` is not, instead of returning an empty
string for the guardrail to swallow. A silent fallback hid the root cause once;
it should not get to do it twice.

## What I would do differently

The selection criteria were price, tool calls, context length. The missing
fourth is **default inference behaviour** — is thinking on, what does it cost,
which sampling parameters does it disable. That question is not in the pricing
table or the quick-start; it surfaced only from reading the raw response body.

Generalised: for a latency-bound conversational workload, "is this a reasoning
model and can I turn it off" belongs in the same breath as "what does it cost".

## Related

Both findings so far were invisible from the layer above. Finding 01 needed
aggregate statistics over the whole quiz bank; this one needed the raw HTTP
response instead of the SDK's parsed `.content`. The convenience layer is where
defects go to hide.

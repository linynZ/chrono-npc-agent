# Finding 05 — Streaming helps less than expected, caching helps somewhere else, and I nearly published a cold start

**Date:** 2026-08-10
**Status:** streaming shipped; the expectation attached to it was wrong
**Reproduce:** `POST /api/chat/stream`, and the timing harness in this document

## What I expected

p95 sat just under 3 s and the README called it "honest but not good enough for
someone standing in front of an NPC". Streaming was the obvious fix, and I said
so out loud: get the first words out in a few hundred milliseconds and the rest
can arrive while the player reads.

## What actually happened

Streaming works — 49 to 103 deltas per reply, the client can replay them exactly,
and guardrail retraction is handled. But the number it was supposed to move
barely moved:

| | first token | complete | ratio |
|---|---:|---:|---:|
| cloud, no tool call | 1391 ms | 1981 ms | 1.4× |
| cloud, with tool call | 2906 ms | 4270 ms | 1.5× |
| local, warm | 750 ms | 1153 ms | 1.5× |

A consistent 1.4–1.6×, not the order of magnitude implied by "a few hundred
milliseconds". Time-to-first-token is a floor, and the floor is most of the wait.

## Why the floor is where it is

The suspect was prefill: the system prompt carries persona, voice, knowledge
scope, four boundaries and injected state — 1047 characters, 741 prompt tokens.
Plausible, and wrong. Four identical calls:

| call | latency | prompt tokens | cache hit | cache miss |
|---|---:|---:|---:|---:|
| 1 | 1725 ms | 741 | 640 | 101 |
| 2 | 1520 ms | 741 | 640 | 101 |
| 3 | 1685 ms | 741 | 640 | 101 |
| 4 | 2375 ms | 741 | 640 | 101 |

**86% of the prompt was already being served from cache, and latency did not
care.** Prefill was not the bottleneck; the round trip and the provider's own
queueing are. Shortening the prompt — which would have cost persona quality —
would have bought nothing.

The caching is not worthless, it just pays out somewhere else. Cached input
bills at 0.02 CNY/M against 1 CNY/M, so 640 of every 741 prompt tokens cost 2% of
list. That is a cost optimisation that arrived for free and was invisible until
measured. **Cache saves money, not time.**

## The cold start I nearly reported

First streaming measurement of the local model: **4114 ms to first token**. Next
to the cloud's 1391 ms, the obvious story was "local streams badly."

That number contradicted an earlier measurement — the same model had completed a
whole non-streamed turn in 752 ms. Two numbers 5× apart for the same work is not
a result, it is a bug in the measurement. Re-running three times in a row:

```
本地 qwen2.5:7b
  1st: first token 1060 ms   complete 1709 ms
  2nd: first token  750 ms   complete 1332 ms
  3rd: first token  758 ms   complete 1153 ms
```

Ollama evicts an idle model from VRAM and reloads on demand — 41.9 s cold, and a
partial penalty for a while after. The 4114 ms was a reload, not inference.

Warm, the local model reaches first token in **~750 ms, and does it consistently**.
The cloud path ranges 1109–2649 ms across three consecutive identical calls. So
local is not merely faster on average; it is *steadier*, which for dialogue
matters more than the mean — a player notices variance long before they compute
an average.

The near-miss is the lesson, and it is the same one as Findings 01, 03 and 04:
the first plausible number was the misleading one. What caught it was a
contradiction with an earlier measurement, not care taken at the time.

## What shipped

- SSE endpoint `POST /api/chat/stream`, events `delta` / `replace` / `done`
- Sentence-boundary guardrail checks during the stream, because a guardrail
  wants the finished reply and the player wants the first words now. Exposure is
  bounded to one sentence rather than eliminated, and `replace` retracts what is
  already on screen. **A client that ignores `replace` is broken** — it would
  leave a leaked answer visible.
- Optimistic streaming with retraction when a tool call turns up after a
  preamble. Models emit nothing alongside tool calls in practice, so it rarely
  fires, but leaving the preamble stranded above the real answer is worse than
  a flicker.
- `first_token_ms` is reported separately from `latency_ms` everywhere. Reporting
  only the total would hide the only thing streaming improves.

## The honest summary

Streaming was worth doing — 1.5× on perceived wait, and it matches the game's
existing typewriter dialogue box exactly, so the Unity side needs no new
animation. But it did not solve the latency problem, and the README no longer
claims a target it does not hit. The remaining lever is the round trip itself,
which points at the local backend rather than at any further prompt work.

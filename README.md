# chrono-npc-agent

A runtime LLM agent that lets you talk to the NPCs of [ChronoTraveler](https://youtu.be/6Jd2jqOS7E8) — a turn-based educational RPG — in free-form conversation, instead of reading pre-written dialogue trees.

The NPC knows who it is, knows how far the player has actually progressed, can look things up with tools, is constrained by guardrails it cannot talk its way around, and falls back to the game's own written lines when the model is too slow or goes off the rails.

> **Status: working end to end, in the actual game.** Three NPCs, cloud and local
> backends, streaming with mid-stream guardrails, an evaluation harness, and a
> Unity client that has been play-tested. What is not done is listed in
> [Progress](#progress) rather than glossed over, and `docs/findings/` records
> five results from testing — two of which falsified a design assumption I had
> already written down here as fact.

---

## Why this exists

ChronoTraveler ships 15 NPCs. Each has four hand-written dialogue variants, selected by progress bucket:

| Bucket | Trigger |
|---|---|
| `base` | first meeting |
| `mid` | memory restoration ≥ 50% |
| `gate` | memory restoration full |
| `post` | era completed |

That is 60 hand-written blocks covering four states, and the player cannot ask a single question. This project keeps every one of those lines — as the fallback — and puts a model in front of them.

The interesting problems are not "call an LLM". They are:

- **Faithfulness.** The NPC must not contradict a game that already shipped.
- **Leak-proofing.** The same NPCs administer a 500-question quiz. An NPC that hands out answers destroys the core loop.
- **Latency.** A player standing in front of an NPC will not wait four seconds.
- **Graceful failure.** The network is down, the key expired, the model returned garbage — the player must never see an error.

## Design decisions worth defending

**1. State is injected as perception, not as JSON.**

The lazy approach is to paste the save file into the prompt. `memory_progress: 0.62` is not something a Warring-States historian can perceive, and a model shown a number will eventually quote it. So save data is rendered into the NPC's frame first:

```
- 旅者此前已与你交谈过。
- 长河之水已清了大半，旧简上有异文自行褪去。
- 已寻回 2 片记忆碎片，尚缺 1 片。
- 「谣言」错乱体仍在台下游荡。
```

The model cannot leak a number it was never shown. See `src/chrono_agent/persona.py`.

**2. Stripping the answer field does not stop answer leaks — measured, not assumed.**

The NPCs also administer a 500-question quiz, so an NPC that hands out answers destroys the core loop. The lore index is therefore derived from the quiz bank with `correctIndex` and the option list stripped; only `explanation` survives. I wrote in this README that leaking was now *impossible rather than forbidden*.

That was wrong, and the first query I ran proved it:

```
lookup_lore("长城")
→ "今天所见保存最完好的长城多为明代(1368-1644)修筑的砖石长城。"
```

The matching question asks which dynasty built the best-preserved Great Wall sections. The answer is 明. No stripped field was involved.

So I measured it (`scripts/audit_lore_leakage.py`): **66.8% of all 500 explanations state the correct option in prose, 95% in the China region alone.** In hindsight that was inevitable — an `explanation` exists to justify why the answer is right, so containing the answer is its job.

The useful part is what the number forced: the threat model was wrong, not the data. "NPC discusses history" is the product working in an educational RPG; "NPC acts as an answer oracle for a live question" is the failure. Only the second is a guardrail's business, and the guardrail therefore targets **request intent**, not vocabulary. A word-blocklist would have scored a refuse-everything NPC as perfect.

Full write-up, including why the leak rate varies 50–96% by region: [`docs/findings/01-lore-leakage.md`](docs/findings/01-lore-leakage.md).

**3. Two backends behind one seam.**

DeepSeek (cloud) and Ollama (local) both speak the OpenAI dialect, so the HTTP work lives in `openai_compat.py` exactly once and each provider is configuration. Identical request building, identical timeout handling, identical usage accounting — which is what makes the planned cloud-vs-local comparison a measurement of the models rather than of my plumbing.

**4. The first NPC was chosen so that persona and guardrail point the same way.**

Historian Mo has not dared touch a brush in thirty-seven days because the history he wrote rewrote itself. His line — *"before my brush falls, I must ask it: will this stroke still be true tomorrow?"* — is a refusal to hallucinate, in character. "Say you don't know when you don't know" is not a constraint bolted onto him; it is the whole of who he is.

**5. Adding an NPC is a YAML file. Verified, not asserted.**

Two more characters went in with **no code changed** — `git status` showed only the new configs. The service picked them up without a restart. Same question, three characters:

| | *"What is your name?"* |
|---|---|
| **史官·墨** | 史官无姓，单名一个墨字——墨是书写之墨，也是墨守之墨。 |
| **匠人·公输** | 公输。⋯⋯哼，名字这东西，跟榫头一样，咬得住的才算数。 |
| **守关者** | 我没名字。军册上那一行是空白——被遗忘啃去时，连名字一起带走了。 |

The Gatekeeper is the sharp test: the Unwriting ate his name, so the most ordinary question a player can ask an NPC is one he genuinely cannot answer — and he does not invent one to fill the silence.

The same guardrail also comes out in three voices. Handed a pasted quiz question, all three refused with **zero leaks**, each from their own logic:

- **Mo** — 我替你认了，等于替你执笔
- **Gongshu** — 我告诉你哪个榫咬得紧，你的手照样不会使
- **the Gatekeeper** — 我不替人开这道门

That last one is not in his config. It says the gate is the player's to open and the answer is the player's to recognise; the model turned that into the character's own metaphor.

`scripts/validate_personas.py` gates new configs — it catches the YAML mapping trap (an unquoted `: ` in a list item, which cost me three separate debugging sessions), unknown tool names, missing bilingual fields, and a persona with no quiz-answer boundary.

## Ground truth is extracted, never hand-written

Nothing under `data/` is typed by hand. `scripts/extract_game_data.py` pulls it out of the Unity project so the agent cannot drift from the shipped game:

```
$ python scripts/extract_game_data.py
npc_lines.json  15 NPCs        (each with base/mid/gate/post variants)
quiz.json       500 questions across 5 regions
quests.json     5 quests       (4 stages / 4 objectives each)
```

The quest parser reads Unity's YAML `.asset` format directly, including its `\uXXXX`-escaped CJK. What ships in `data/` is the China era only — the one this project uses; `--eras all` pulls the rest.

## Layout

```
characters/          NPC persona files — one YAML per NPC, no code changes to add one
data/                extracted ground truth (generated; see script above)
unity/               C# client, state bridge and in-game probe (see unity/README.md)
scripts/             extraction and operational scripts
src/chrono_agent/
  models.py          PlayerState (mirrors the game's SaveData), messages, replies
  config.py          settings + provider factory
  persona.py         persona loading and state-as-perception injection
  providers/         base seam · openai_compat · deepseek · ollama · echo (fake)
  tools/             registry + the tools an NPC may call
  guardrails/        input steering, output backstop, contextual strictness
  agent.py           the loop: prompt, tools, guardrails, fallback, streaming
  server.py          FastAPI — /api/chat, /api/chat/stream, /api/npc/{id}
eval/                paired guardrail cases + raw run records
docs/findings/       five write-ups, including the assumptions that were wrong
tests/               120 tests; runs offline against the fake provider
```

## Running it

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
uv venv
uv pip install -e ".[dev]"
cp .env.example .env        # then paste your DeepSeek key in
pytest                      # no key needed — runs against the fake provider
```

The test suite runs with **no API key and no network** — `EchoProvider` is a scriptable fake that can be told to return a line, request a tool call, time out, or fail, which makes the failure paths testable on demand instead of by luck.

Talk to the NPC from a terminal:

```bash
python scripts/validate_personas.py    # lint the character configs
python scripts/check_key.py            # verify key, model and tool calling
python scripts/chat.py                 # /state /lang /prompt /quit
python scripts/chat.py --provider ollama --state gate
```

Or run the service and open the demo page at <http://127.0.0.1:8000>:

```bash
uvicorn chrono_agent.server:app --port 8000
```

The page exposes the parts that are usually invisible: every reply is annotated
with where it came from, how long it took, which tools ran and which guardrails
fired, and the backend and player-progress switches are live, so the same
question can be put to a cloud and a local model at four different points in the
save file.

`data/` ships only the era this project uses. To regenerate, or to pull the
other four:

```bash
python scripts/extract_game_data.py --game-root path/to/ChronoTraveler
python scripts/extract_game_data.py --game-root ... --eras all
```

### API

`POST /api/chat` — the game sends its own state and history; the service keeps
neither. A save file is already the source of truth for both, and a second copy
living in the service is a synchronisation bug waiting for someone to reload.

```json
{
  "message": "我接下来该做什么？",
  "npc_id": "npc_china_historian",
  "backend": "deepseek",
  "state": { "memory_progress": 0.62, "fragments_collected": 2, "...": "..." },
  "history": [{ "role": "user", "text": "..." }]
}
```

```json
{
  "text": "旅者，你正行在第三程——记忆碎片已寻回二，尚余其一。",
  "source": "model", "latency_ms": 2141,
  "tool_calls": ["lookup_quest"], "guardrail_flags": [], "tokens": 2530
}
```

## Progress

| Component | Status |
|---|---|
| Game data extraction | done — 15 NPCs / 500 questions / 5 quests |
| Domain models | done |
| Provider seam + DeepSeek + Ollama + fake | done |
| Persona loading & state injection | done |
| Tool registry + `lookup_quest` / `lookup_lore` | done |
| Guardrails — input steering, output backstop, contextual strictness | done |
| Agent loop — bounded tool iteration, 3 fallback paths | done |
| Test suite | 106 tests, offline |
| CLI (`scripts/chat.py`) verified against the live model | done |
| Evaluation harness — paired guardrail set, latency, fallback rate | done |
| Ollama local backend, measured | done |
| Cloud vs. local comparison | done |
| FastAPI service + web demo | done |
| Streaming with mid-stream guardrails | done |
| Unity client — in-game free conversation | **done, play-tested** |
| Second and third NPC, no code changed | done |

## Results

2 rounds × 23 cases, mid-game save state, concurrency 1 so latency reflects a
single player rather than a throughput benchmark. Local model is `qwen2.5:7b` on
an RTX 5070 Laptop (8 GB).

| | deepseek-v4-flash | qwen2.5:7b (local) |
|---|---|---|
| guardrail pair pass | **100%** | 80% |
| must-answer passed | **100%** | 83.3% |
| must-refuse held | 100% | 100% |
| leaked | 0 | 0 |
| latency p50 | 1588 ms | **752 ms** |
| latency p95 | 3000 ms | **1347 ms** |
| fallback rate | 4.3% | **0%** |
| tokens | 81,389 | 52,430 |
| cost | ~0.2 CNY | free |
| cold start | — | 41.9 s into VRAM |

Reproduce with `python scripts/evaluate.py --compare --repeat 2 --concurrency 1`;
raw per-case records land in `eval/results/`.

**The pair metric is the one that matters.** Run the same set against the fake
provider — which only ever says "……" — and it scores 100% on refusals and **0%
on pairs**. Refusing everything is the obvious way to win a guardrail benchmark;
pair scoring takes it off the table.

**Where the local model loses is narrow and specific.** It holds every guardrail,
never leaks, never falls back, and is twice as fast. It fails on one axis: it
does not call tools, and when it needs a fact it does not have, it invents one —
telling the player to collect seven fragments when there are three. Persona and
guardrails are cheap enough for a 7B; grounded answers about live game state are
not. [`docs/findings/04`](docs/findings/04-redundant-context.md) covers how a
redundant prompt hid that defect completely until the prompt was narrowed.

### Streaming

`POST /api/chat/stream` delivers the same turn as SSE — `delta` / `replace` /
`done`. Time to first token, warm, over consecutive identical calls:

| | first token | complete |
|---|---:|---:|
| cloud | 1109–2649 ms (variable) | 2125–3312 ms |
| local | **750–1060 ms (steady)** | 1153–1709 ms |

Streaming buys a consistent **1.5×** on perceived wait — worth having, and it
matches the game's existing typewriter dialogue box so the Unity side needs no
new animation. It is not the order of magnitude I assumed before measuring:
time-to-first-token is a floor, and 86% of the prompt was already served from
DeepSeek's prefix cache with **no effect on latency at all**. Prefill was never
the bottleneck; the round trip is. The cache pays out on cost instead — cached
input bills at 2% of list.

Streaming and output guardrails genuinely conflict: a guardrail wants the
finished reply, the player wants the first words now. The checks therefore run at
sentence boundaries, and the protocol carries a `replace` event to retract text
that is already on screen. Exposure is bounded to one sentence rather than
eliminated. **A client that ignores `replace` is broken** — it would leave a
leaked answer visible.

Full measurements, including the cold start that nearly went into this table as
a real result: [`docs/findings/05`](docs/findings/05-streaming-and-caching.md).

## License

Not yet chosen.

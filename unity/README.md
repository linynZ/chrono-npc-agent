# Unity client

Four files. Drop them into `Assets/Scripts/Agent/` in a ChronoTraveler checkout.

| file | depends on | job |
|---|---|---|
| `NpcAgentClient.cs` | Unity only | HTTP + SSE. No game references, so it lifts into any project. |
| `ChronoAgentBridge.cs` | the game | Reads live save state into the wire format. |
| `NpcFreeTalk.cs` | the game | The real feature — free conversation layered on top of scripted dialogue. |
| `NpcAgentProbe.cs` | both | Console probe. F8 cycles six questions; useful when the UI is not the thing being debugged. |

The split is the point: `NpcAgentClient` knows nothing about the game. Porting
this elsewhere means rewriting the bridge and the UI, not the transport.

## How free conversation fits in

Scripted dialogue runs first and is **not touched**. It still fires
`OnDialogueCompletedId`, which is what advances a quest's TalkTo objective — an
NPC whose dialogue was handed wholesale to a model would quietly break the quest
chain. When the written lines finish, a prompt offers to keep talking:

```
E → scripted lines (quest fires, narrative intact)
  → "按 T 与史官·墨继续交谈"
  → T → free conversation panel
        suggested openers (from the persona YAML) + a text field
        ESC closes
```

The prompt appears **only** if the service answered `/api/npc/{id}` for that NPC.
Service down, or NPC not configured: no prompt, and the game behaves exactly as
it always did. An entry point that does nothing when pressed is worse than none.

Streaming lands naturally here — the game already reads dialogue a character at
a time, so `onDelta` drives the text directly rather than waiting for the whole
reply and then animating text that has already arrived.

Free conversation also degrades differently. The scripted path substitutes a
written line, which is right there and wrong here: the player just asked
something specific, and answering with an unrelated story beat reads worse than
the character having nothing to say. `free_form: true` switches the fallback to
the persona's last-resort line — *「……」史官·墨垂下眼，指尖在案上停了一停，终究没有开口。*

## Verified

Compiled against the real project with the game's own offline check
(`tools/verify_compile.sh` — Roslyn csc against the Unity engine DLLs and package
assemblies, runs while the editor is open):

```
OK: 135 runtime scripts compile clean (Unity 6000.3.15f1)
```

131 existing scripts plus these four. Three namespace assumptions were wrong
across two passes and the compiler caught all three — `CollectionManager` lives
in `ChronoTraveler.Collectibles` (plural, unlike its folder), the localisation
entry point is `LocalizationManager` rather than `Localization`, and `UITheme`
sits in `ChronoTraveler.Battle` despite being used project-wide.

**Not yet run in play mode.** Compiling is not the same as working, and this
section will not claim otherwise until someone has actually talked to him.

## What the bridge sends

Only what the NPC being spoken to could plausibly know:

- current map, language, memory restoration, fragments held
- whether **this** NPC has been met — not the whole acquaintance list
- encounters purged **from the current quest** — `SaveManager` exposes membership
  tests rather than the set, and that turns out to be the better shape: an NPC
  has no business knowing about eras they have never mentioned
- current quest stage and its first objective

The service narrows this further before it reaches the model — exact counts are
withheld so the tools are not redundant, see `docs/findings/04`.

## Two things a caller must get right

**`onReplace` is not optional.** It is how the service retracts text already on
screen when a guardrail trips mid-reply. Whatever it hands you replaces the
entire visible line; an empty string clears it. A client that ignores it will
leave a leaked quiz answer in front of the player, which is the exact failure the
guardrails exist to prevent.

**The decoder is stateful for a reason.** A UTF-8 Chinese character is three
bytes and lands across a chunk boundary regularly. `SseHandler` keeps a
`Decoder` across calls rather than running `Encoding.UTF8.GetString` per chunk,
which would corrupt precisely the characters this game is made of.

## Wiring it into real dialogue

The probe writes to the Console instead of the dialogue box, deliberately.
Routing this into the shipped `DialogueUI` means editing a finished game, and
that is a decision to make on purpose rather than as a side effect of a probe.

When it is made, streaming lands naturally: the game already renders dialogue
with a typewriter, so `onDelta` drives it directly instead of waiting for the
whole reply and then animating text that has already arrived.

## Field naming

The DTOs use `snake_case`. `JsonUtility` cannot rename fields and the service
speaks snake_case, so the structs match the wire rather than C# convention.
A hand-written mapping layer for six structs would be the worse trade.

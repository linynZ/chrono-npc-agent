# Unity client

Three files. Drop them into `Assets/Scripts/Agent/` in a ChronoTraveler checkout
and press **F8** in play mode.

| file | depends on | job |
|---|---|---|
| `NpcAgentClient.cs` | Unity only | HTTP + SSE. No game references, so it lifts into any project. |
| `ChronoAgentBridge.cs` | the game | Reads live save state into the wire format. |
| `NpcAgentProbe.cs` | both | Self-booting probe. F8 cycles through six questions, Console shows the result. |

The split is the point: everything game-specific is in one file. Porting this to
another project means rewriting `ChronoAgentBridge` and nothing else.

## Verified

Compiled against the real project with the game's own offline check
(`tools/verify_compile.sh` — Roslyn csc against the Unity engine DLLs and package
assemblies, runs while the editor is open):

```
OK: 134 runtime scripts compile clean (Unity 6000.3.15f1)
```

131 existing scripts plus these three. Two namespace assumptions were wrong on
the first pass and the compiler caught both — `CollectionManager` lives in
`ChronoTraveler.Collectibles` (plural, unlike its folder) and the localisation
entry point is `LocalizationManager`, not `Localization`.

**Not yet run in play mode.** Compiling is not the same as working, and this
section will not claim otherwise until someone presses F8.

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

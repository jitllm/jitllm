# The server orchestrator: placement driven by the server itself

jitllm's blocks, their KV pages and their recurrent state move between the
host and any device at runtime (the relocation principle). A user can move
them (`PlacementService.RelocateBlocks`). This design is the server moving
them itself: making room for a conversation that needs a card, giving the
cards to the conversations that are generating, and taking them back from the
ones that went quiet. Every decision is one a caller can force or forbid.

## What exists

- Preemption (`server/preempt.go`): a per-model KV budget; a generate that
  would exceed it parks other sessions of the same model (their history home
  and into a store), lowest priority then least recently used, and the parked
  session resumes byte for byte. It moves history, never weights, and never
  across models.
- Plasticity in the engine: `gpu.GPU.AddDevices` (a running model takes a
  card), `State.MoveBlocks` (any block range to any device, the CPU ("cpu")
  among them, with its history; `State.PlaceBlocks` is it from the CPU with
  the tier choosing the card), binary16 histories that travel converted.

## v1: promote on demand, demote the idle to make room

**Promote.** When a session starts a generate (where preemption admits its
KV), the orchestrator offers the session's host blocks to the model's
devices (`PlaceBlocks`), so a conversation that is talking runs on a card
whenever a card can hold it -- also after it was demoted, and for a model
loaded on the CPU when the server has a device.

**Make room.** When the card declines for lack of room, the orchestrator
demotes the least deserving sessions on that card -- not generating (their
lock is free), lowest priority, then idle longest, sessions of other models
first -- by bringing their blocks home with their history (`MoveBlocks` to
"cpu"), then offers again. A block's weights leave the card when no session of its
model holds it there (the tier's rule). It never demotes a running session,
and it stops when the promotion fits or no victim is left: then the session
runs where it is, as before (nothing is refused).

**Leave alone.** A placement the caller chose is pinned: the orchestrator
neither promotes nor demotes it. A session is pinned by RelocateBlocks or
SetPlacement (before anything moves, a move between devices included), and
at creation by a positive `max_device_blocks` or named devices. A model is
left alone when its load named the host alone or capped its device blocks.
`Session.Unpin` hands a session back. The engine's own per-session relocation
on a full card (SetRelocation) keeps working underneath.

**One budget per card.** A server holds a tier per model, and each tier used
to size its own pool for a card from what was free when it opened, so the room
one model gave back stayed out of another's reach. Every tier the engine opens
takes its cards' pools from one `gpu.PoolRegistry`, keyed by the card's
identity: a card's memory is spent once across models.

**Visible.** Every orchestrated move is recorded -- session, model, blocks,
from, to, why ("promote: generating", "make room for s-…"), how long it took
-- in a bounded log the stats stream carries, and counted (promotions,
demotions, blocks moved, bytes moved). The web UI's Placement tab and the TUI
show it.

**Forcible.** `jitllmd -orchestrate=false` (and an Engine option) turns it
off; per session, pinning as above; per model, `max_device_blocks` caps it.
Defaults on: the server reaches the best placement without user tuning.

## Gates

- `TestTheTalkingModelGetsTheCard`: two models on 1200 MiB of one card. The
  model generating gets the card every turn and the idle one goes home (9
  promotions and 5 demotions over three turns on a V100), each turn's first
  token the unmoved controls' (host and card). The stats carry the same
  counts. Without the registry both models fit their own budget, nothing
  moves, and the gate fails.
- `TestAPinnedSessionIsNotMoved`: a pinned session keeps its blocks while
  another model talks; with the orchestrator off nothing moves.
- `TestBatchASessionNotWhollyOnTheDeviceRunsAlone`: a session held to two
  device blocks stays at two.
- `gpu.TestTwoModelsShareACardThroughTheRegistry` (engine/model): the
  second model takes the room the first gives back.
- The engine's gates hold a moved history to the unmoved one exactly
  (`TestAnyRangeMovesWithItsHistory`, `TestABinary16HistoryMovesToTheDevice`).

## Later (v2)

Rebalancing across cards as load shifts (MoveBlocks between devices),
predictive promotion (a session about to generate: queued, typing), host-RAM
pressure (paging weights out of host for idle models), and a cost model that
weighs a move's milliseconds against the tokens it buys.

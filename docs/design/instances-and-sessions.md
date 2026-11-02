# Instances and sessions

What AGENTS.md's "The shape: model, instance, session, scheduler" rests on.
This file is evidence; the rule is AGENTS.md's.

## What it replaced

`model.State` was one type holding four things: the placement (which block
runs where, the device route), the composed kernels and their scratch, the
old engine path's forward buffers, and one conversation's history. Each
State built its own JIT and composed its CPU forms when it was made, so every
session opened composed, and a second session's first decision question
composed again. A server session owned a State for its whole life; an idle
pool of placed States stood in for "an instance ready to take a request".

## Choices, and why

- **A lease on a shared instance, not an exclusive instance.** Batching is
  many sessions' rows in one pass over the weights; an instance held by one
  session at a time cannot batch. A session holds its lease for a phase;
  which row of the step it occupies changes every step.
- **Session-owned pages, not a batch's interleaved ones.** A multi-row
  batch's page holds every row's positions side by side, so moving one
  sequence to another instance or retiring it means cutting its slots out.
  Separate sessions under `StepRuns` already batched without that coupling.
- **Moving history: copy, then free.** `migrateKV` and `migrateRec` copy; the
  move frees the source once the copy has landed, and on failure the source
  stays the owner. A fenced transfer is not a second holder.
- **Park moves the session's history, not the instance's blocks.** The old
  Park took every block off the card (`SetGPULayers(0)`), which with a shared
  instance would relocate every other session leasing it.
- **Binding is allocation-free; a move is eager.** An instance runs for one
  session at a time between steps (`Instance.use`). A move of the instance's
  blocks (`SetGPULayers`, `SetDeviceLayers`, `MoveBlocks`, a committed
  `PendingMove`) moves the history of every session that ran on it, and its
  own session's view, before it returns (`settleHist`): a lazy move left the
  sessions' views holding the blocks, so a seam emptied of blocks kept their
  weights and recurrent state on the card. A session whose history would not
  move stays away and moves at its next run; one short of room on the card
  moves blocks to the CPU, every session's history of them with them
  (`reserveMoving`, `roomFor`).

## Measured on the way

- Two sessions stepped together on one instance decode what each decodes
  alone on stories260K, SmolLM2-360M and Qwen3-MOE-4x0.6B under the pinned
  reduction order (no tuner, an f32 cache, GEMMExact). With the binary16
  cache the joint step reads NMSE 2.4e-3 to 8.8e-3 against each alone
  (SmolLM2, qwen3moe on the Xeon E5-2680 v4): a session's rows read their
  history through the rows' form and alone through the one-row form.
- A mixture's greedy tokens parted at the sixth token between the joint
  step and alone with the cache's width unpinned; at HEAD, before this
  change, two States stepped jointly parted the same way, and with the f32
  cache both agree.
- Qwen3.5-0.8B parts at token 1 between the batched path and the sequence
  alone, identically before and after this change ("seq 1 token 1:
  scheduled 13, solo 220"): the rows path against the one-row path, the same
  class as `TestPrefillMatchesForward`'s exact-GEMM arm.
- After an instance is made and served, sessions' prompts, tokens alone and
  together and every sampler ask the CPU composer for nothing (four models).

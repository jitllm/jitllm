# A prompt chunk in a ragged step runs as a prefill

A batch step (`model.Scheduler`, `tier.GPU.LayersRows`) carries rows of several
sequences: decoding rows, and prompt chunks admitted beside them. On a device,
the step is built for decode. A prompt chunk in it pays for that, and a step
made only of prompt chunks (a decision batch, a server burst of new requests)
pays the most.

## What was measured

M4 mini, Metal, every block placed, `TestDecisionDeviceAB` batch-dev arm, 20
rounds after 3 soak: the two decision requests as one `DecideBatch` took
**19.84x** (d1-3B Q8_0) and **19.12x** (lev Q8_0) each request alone. #165
routes a decision batch on a device to the requests alone; the general
Scheduler still takes this path.

The device counters over the measured rounds (batch arm against alone, d1):

| counter | batch | alone | what it is |
|---|---|---|---|
| TSubRead | 747 s | 37 s | waiting for the device: its execution |
| TSubStage | 197 s | 0.5 s | staging uploads before a submission |
| RopeTables | 554400 | 18480 | rotary tables built on the host |
| PagedLaunches / PagedPrefillLaunches | 320 / 0 | 0 / 1600 | attention through the decode plan, or the tiled prefill |
| RecChainRows | 401720 | 0 | rows stepping a recurrent state after another of their run |

On the laptop's CUDA card with 12 of 30 blocks placed the batch and the
requests alone read the same (1.02x). The cost is the step's form, not
batching.

## Where it comes from

1. **Attention.** `pagedprefill.go`: "A ragged batch keeps FlashDecodeKV: its
   rows are different sequences." Every row of a chunk reads its sequence's
   whole history on its own; a prefill's query tile reads each K and V once for
   the tile. For a 500-row chunk at 500 keys of history that is the
   difference between ~500 and ~8 reads of each key.
2. **Rotary tables.** A ragged step's rows are at scattered positions, and the
   tables are built per row on the host and staged (`RopeTables`,
   `TSubStage`), where a prefill turns its rows on the device from one table
   (`ropeDev`).
3. **Recurrence.** Each run steps its rows in order in one launch (the scan's
   run form, `deltaScanOf` with `Runs`), the same sequential algorithm as a
   prefill chunk's. It is not expected to be a cost beyond the prefill's; the
   per-kernel split below says whether it is.

## The change

**First, measure the split.** A per-kernel-family device time for one ragged
step and for the same rows as per-sequence prefills (Metal's command-buffer
GPU times per family, CUDA events), so each item below has its share before
it is built.

**Attention: a run's rows attend as a prefill.** `groupRuns` already gives
each sequence's rows in position order with the sequence's table. Two forms:

- *Per-run launches*: each run of more than one row launches the paged
  prefill kernels over its rows, with q and the output addressed at the run's
  first row and the run's own descriptor (`PRowTab`, keyStart, keyEnd); the
  decoding rows (runs of one) keep FlashDecodeKV. Launches grow with the
  runs (at most the batch's sequences), and nothing in the kernels changes.
- *One launch, tiles per run*: the prefill kernels' partition base and table
  read per query tile rather than from row 0, and the tiling never puts two
  runs in one tile (runs padded to the tile). One launch per layer, a kernel
  change gated against the per-sequence prefill.

Per-run launches first: no kernel changes, and the gate (a ragged step's
logits against each sequence's prefill, every row) holds the tiled form when
it comes. Both are basic generated code on every backend (RULE 8); the
Volta/Ampere MMA forms follow the prefill's own selection.

**Rotary: on the device.** The step uploads each row's position, already
staged for the KV writes, and the rotary kernel reads the device table at it,
as a prefill does; the host tables remain for positions past
`kernels.RopeTabMaxPos`.

**Recurrence: unchanged unless the split says otherwise.**

## Gates

- A ragged step whose rows are prompt chunks of several sequences, every row's
  logits against each sequence's own prefill on the same device, in the
  reduction band (RULE 11c), on CUDA, Vulkan and Metal, a hybrid and a dense
  model; a violation (a run's tile reading another run's table) must part
  them.
- Selection: a counter of attention launches taken in the prefill form inside
  ragged steps, asserted non-zero where runs exceed one row, zero where every
  run is one row.
- `TestDecodeDoesNotAllocate` and `TestSchedulerMixesPromptsIntoDecodeSteps`
  unchanged; a warm ragged step stays allocation-free.
- Perf: `dev/bench` `TestDecideBatchIsNotSlowerThanAlone` with the decision
  batch routed through the Scheduler on the device again (#165's routing
  turned off by its violation), on the M4, n >= 20, twice, A/A first. When it
  holds, #165's device routing comes out.

## The run table's forms

The one-launch form was built (`jit/gpu/tier/raggedprefill.go`): each
sequence with more than one row in the step gets its rows into a run table,
padded to whole query tiles with its last row, and the prompt attention's own
kernels read it -- slot i is row `pMap[i]` of q, the descriptors and the
output. Every form a prompt alone takes has it:

| form | where | the table |
|---|---|---|
| FlashPrefill80 (and its m16n8k8 twin) | sm_75 on | `FlashPrefill70Shape.Map`; with sinks the one split's partials at the rows and a mapped `FlashAttentionMergeWide` |
| FlashPrefill70 | sm_70, heads of 64 and 128 | as FlashPrefill80 |
| FlashPrefillTile | Metal, and `PreferTiles` | as FlashPrefill80 |
| the staged scores, softmax and accumulate | Vulkan, and every shape a flash form refuses | `FlashShape.Map`: `pMap` and each slot's partition base `pBase` (a run's first key, 64-aligned); the score and weight planes stay in slots |

The staged form's table runs on the instruction the staged form chose: the
FMA tiles, m16n8k16 scores with the FMA or the m16n8k8 accumulate, sm_70's
m8n8k4 pair. A run whose span passes the pass width runs in passes as the
sequence alone does: pass j of each slot attends its row's keys from its
run's base plus j pass widths (`ragPassStage`), the softmax and the
accumulate write each pass's partials at the rows, `FlashAttentionMergeRun`
folds them over the scratch's rows, and a mapped `FlashAttentionMergeWide`
(with the sink) writes the table's rows. A run shorter than the step's
longest has empty passes at the end, which fold to the same bits.

Gates: `backend.TestStagedPrefillRunTable` (every staged form, direct and in
passes, bit for bit against each sequence alone; the violation -- one run's
base a block past its first key -- must part that run and leave the other
alone) and `TestFlashPrefillRunTable`; `model.TestRaggedStepAttendsAsAPrefill`
with arms for the device's choice, the tiles, the staged form and the staged
form in passes, each asserting its own counter (`Stats.RagPrefills`,
`RagStaged`, `RagStagedMMA`, `RagPasses`).

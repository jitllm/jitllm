# The composer: model in, kernels and a plan out

`jit/` is the composer and nothing else. It takes a model and a placement and
returns, for every device, the kernels the model runs there and the plan that
launches them. `engine/` runs the plan: paging, KV pools, sessions, transfers.
The old JIT, its tiers and its CPU emitters are in `jit_old/`, frozen, and
`jit/` imports none of it.

The pipeline is a plugin surface. A new op is one registered decomposition, a
new optimization one registered pass, a new backend one registered lowering
and its profile. Research at the op level (a new recurrence, a new attention,
a factorization trick) lands as one of those and reuses everything else.

## The pipeline

    model (.jlm) + placement (given, or suggested in plan mode)
        │
        ▼
     FRONT END        the model's graph: every block, the head, the step's tables,
        │             each value's shape, each weight's storage format
        ▼
     DECOMPOSE        one registered decomposition per graph op kind
        │             op ──► primitives: map, reduce, contract, gather/scatter, scan, layout
        │             facts (ranges, head geometry, ...) ride on the primitives as data
        ▼
     GENERATE         each primitive ──► one naive kernel in the composer's IR
        │             no fusion, no schedule; the IR knows its device, nothing is lowered
        ▼
     PASSES           IR in, IR out, one optimization each, registered, switchable,
        │             parameterised; a per-device profile orders them and sets main's choices
        ▼
     LOWER            after the last pass only: one registered lowering per backend
        │             cuda-sm70 ──► jit/ir     host CPU ──► x86 bytes through jit/x86
        ▼
     kernels + plan   per device: kernels, launches, buffers, transfers at device seams

### Front end

The input is the model, read from its container: configuration, weights by
role and their storage formats, the tower and head where present. The front end
builds the model's graph from it in `engine/graph` ops: every block, the head
(final norm, vocabulary projection, sampling's argmax) and the step's tables
(the rotary angles). Nothing outside the composer assembles the composer's
input.

Decision for the user: today the block graphs are built by `engine/model`
(`buildGraphs`, `BlockGraph`), which `jit/` may not import. Either the graph
builders move into `engine/graph`, which `jit/` imports, or the composer builds
the graph from `format/jlm` itself.

### Placement and plan mode

A placement says which device runs each part of the model. Each part is formed
for its device; passes see the runs of consecutive parts placed on one device,
so a cross-block optimization (a block's last residual add folded into the next
block's norm, one launch over several blocks) is a pass over that run. Where
two consecutive parts sit on different devices, the plan carries the transfer.

In plan mode the composer suggests a placement from what its passes know --
bytes moved, kernel costs, fusable runs, memory needed -- and the engine accepts
it or forces its own (every decision forcible, the engine's `With...()` rule).

Where plan mode cannot decide, the engine proposes several placements and the
composer estimates each: it takes every candidate through the front end, the
decompositions and the passes to final IR, and estimates its efficiency from
that IR alone -- bytes each kernel reads and writes, launches, transfers at the
device seams, memory per device, the work a kernel's binding leaves each
thread -- without lowering and without touching a backend. The estimate is a
function of the IR and the device's description, so ranking candidates costs
no compilation and no device time; the engine picks, and only the chosen
placement is lowered.

### Decomposition

`decompose` is the only code that names a graph op kind. Each op kind has one
registered decomposition into primitives:

| primitive | meaning |
|---|---|
| map | an element of the output from elements of the inputs |
| reduce | an element of the output from a combine over an axis |
| contract | a sum of products over an axis |
| gather / scatter | reads and writes through an index (a page table, an expert id) |
| scan | a recurrence carried along an axis |
| layout | copies, slices, transposes |

A primitive is a scalar function over coordinates of a shape. Facts a pass
needs -- an axis of attended positions and its live end, head geometry, a
quantization window -- are data on the primitive and are carried onto its IR,
never re-derived from op names.

### Generate: the naive IR

Each primitive becomes one kernel in the composer's IR, written in the most
direct way: no fusion, no schedule, no binding beyond "the elements of the
output are the iteration space". The IR knows its device (it may use what the
device has), but this step lowers nothing. The naive kernels are the stable
starting point every pass and every plugin transforms.

The IR: an iteration space, loops (static or run-time trip counts), scalar
nodes, loads and stores addressing buffers by coordinates of their shapes, and
the facts. It has no backend instruction in it.

### Passes

The interface:

    type Pass interface {
        Name() string
        Apply(ir []*Kernel, t Target) ([]*Kernel, error)   // IR in, IR out
    }

The rules, every one a gate where it can be:

- **One optimization a pass.** Fusion, constant folding, simplification, CSE,
  tiling, split reductions, binding to threads / lanes / warps / workgroups,
  unrolling, reassociation, workgroup-memory staging, prefetch, run-time
  lengths, vectorization, matrix instructions: each its own pass.
- **IR in, IR out.** A pass never reads a primitive's definition, never
  regenerates IR, never adds primitives, never produces backend code.
- **By structure.** A pass decides from shape, structure and target, never from
  an op, family or model name, so it applies to any kernel with that structure.
- **Switchable and parameterised.** Every pass can be turned off and takes
  parameters; a per-device profile orders the passes and sets main's choices.
  Research adds or reorders passes in a profile.
- **Accountable.** Each kernel records the passes that formed it, and the status
  reports them.

### Lower and backends

Only after the last pass is the IR lowered, by the backend's registered
lowering, the only code that produces backend code. cuda-sm70 lowers to the
GPU IR (`jit/ir`, a copy); the host CPU lowers to x86 through `jit/x86` (a
copy of the encoder). A new backend is a new lowering and its device's profile;
decompositions and passes do not change.

### Parity with main

Main is the reference the composer converges to, per device:

- **cuda-sm70.** Every kernel is dataflow-identical to main's GPU IR kernel
  for the same op and shape: `paritytest.Compare` and `Frontier` on canonical
  dataflow (index arithmetic as affine forms; values, loads and stores exact),
  after converting `jit/ir` to `jit_old/gpu/ir`.
- **host CPU (AVX2).** Every kernel's bytes are identical to what jit_old/cpu's
  emitter produces for the same op and shape, for the kernels a host decode
  calls, recorded by the board's call recorder (`jitllmtest` build).

The board, `paritytest/board`, composes the model it opens and compares every
kernel main's step ran on that device: blocks, head and step tables alike.

## Rules

- `jit/` imports nothing from `jit_old/` or `paritytest`. Beyond its own copies
  of the GPU IR and the x86 encoder it imports only the standard library,
  `engine/graph`, `format/jlm` and `format/quant`.
- No op, family or model name appears in `jit/` outside the decompositions.
- No pass, and not the lowering, reads a primitive's definition or generates
  from one (`srcgate.TestPassesRewriteNests`, an empty list).
- No panics, no mutable globals, no `_ =`, minimal comments.
- A kernel reached any other way is stated plainly in the status, never counted
  as formed by the design.

## What main does: the passes to reach

The evidence from matching synth-ernie45's decode once already, with the shape
each difference has. It is a work list for passes, not code to keep.

### cuda-sm70

| main's kernel | from the naive IR, main's form is |
|---|---|
| `matvec_*_sN` + `reduce` | the contraction over K split into S segments (S doubles while below 32, E·S below twice the device's slots, and T a multiple of 32·S·2), each segment's terms in blocks of 32 unrolled as one chain from 0, the segments' planes summed by a combine; a residual add fused into the combine as its last term |
| `rmsnormquantw` | the norm's sum of squares and its scaling map in one workgroup of 1024 threads a row: the sum spread over the threads, a group reduction (a butterfly 16 down to 1, a workgroup slot, a barrier, a second butterfly); the row's int8 form after a barrier as a phase of the same workgroup, four elements a thread, a window's amax exchanged through workgroup slots, the exact scale amax/127, its inverse with a four-candidate correction, codes rounded half away from zero, four packed a word, the window's sums |
| `quantizeact` | the gated activation fused into the same int8 chain, flat threads, four elements a thread |
| `ropeqkvpaged` | the query's rotary, the key's rotary fused at its stores into the paged key pool's write, and the value's paged write, as parts of one launch |
| `flashkvpaged` | the key range split into partitions (each the live span's share rounded up to 32, at most the chunk a workgroup holds); one workgroup a (row, group of query heads sharing a key head, partition), 4 warps; the dots with lanes over consecutive keys and warps over runs, the head's width within a thread in four interleaved chains summed pairwise; the scores in place in a plane of workgroup memory, scaled, masked, their maximum folded and then butterflied (1 up to 16) and exchanged across warps through 32 copies a warp; the weights the same with their sum; the weighted values with lanes over the head's width clamped to it and warps over keys four a trip, the staged weight masked past the live span, the warps' sums through the plane; each run's page found once (the run's start held to the range's last aligned run, the place its run's plus the lane where lanes run over keys) and carried from trip to trip; trip counts at run time; the partial padded to 32 with 32 copies of the maximum and the sum |
| `flashmergewide` | the split's combine: the partitions weighted by exp of their maximum less the greatest |

Outside the blocks: `ropetable` (the step's angles), `normpartrows` and
`normapplyrows` (the head's norm with its sum split one term a thread and the
partials summed in the apply), `matvec_f32_103424x64_s1` (the vocabulary
projection, f32 activations, one segment).

### host CPU (AVX2)

A host decode calls twelve kernels, each a generic routine taking its length at
run time through `cpu.Args` (`K` whole vectors of 8, `Rows` the tail): in the
blocks `RMSNorm(64)`, the f32 row-major matvec, `RoPE(16, 16, NORM)`,
`NarrowF16` (the f16 history write), `AttnScores2`, `Scale`, `Softmax` and
`AttnAcc2` over the f16 history (two query heads a key head), `Axpy` (the
residual add, `x += alpha·y` with alpha read from scratch) and `ActMul(SiLU)`;
outside them `RopeTable(8)` and `Argmax`. Each has a vector body of eight
floats a trip and a scalar tail; the run-time length is a pass on the IR, the
x86 lowering takes what the passes leave.

## Order of work

Each step ends with its gate green and the board measured.

1. **Skeleton.** The registries (decompositions, passes, lowerings), the IR,
   the model front end and the placement types; the gates (imports, names, no
   definition reads, no op names outside decompositions); the board opening the
   model and composing it.
2. **Decompositions** for every op synth-ernie45 runs: blocks, head, tables.
3. **Naive IR** for every part, lowered on cuda-sm70 once its elements are
   bound to threads. The x86 lowering grows with the CPU passes in step 5:
   the design lowers after the last pass, and a scalar lowering of naive IR
   would be thrown away against main's vector kernels.
4. **cuda-sm70 passes**, one difference at a time, in the table's order:
   matvecs, residual fusion, norm and int8 chain, activation chain, rotary and
   history, attention, then the head and the tables.
5. **CPU passes and the x86 lowering**, kernel by kernel to byte identity.
6. **Plan mode and estimates**: a suggested placement from the passes' costs,
   and the estimate of an engine-proposed placement from final IR, unlowered.
7. **Plan execution.** The engine runs what the composer hands it, and a gate
   proves the hand-off before the engine depends on it.
   1. *The buffer plan*: for every storage a step binds, its size and its
      source -- a weight or norm vector of the container, in the layout the
      container stores; a table the engine builds (the KV pages and the page
      table, the row descriptors, the rotary position and planes, the quant
      offsets); the step's own scratch; or a value one kernel writes and
      another reads -- beside the launches in order. `Plan.Storages` is it:
      each storage's key, `Source` and bytes, every binding naming its
      storage. A buffer reached through a flat index states its extent where
      it is made (`Buf.Extent`, `Facts.Extents`, `PrimStore.Extent`);
      `TestEveryStorageStatesWhatItsKernelsReach` holds every stated size on
      the GPU to the estimate's reach. The engine sizes what it owns: the
      history's pools and page table and the model's rotary planes. The host
      is not held yet: the estimate's reach over a host kernel's flat index
      is a loose bound.
   2. *The execution gate*: synth-ernie45's composed decode steps on CUDA from
      that plan alone, every source real, from an empty history to a few
      positions, held to the engine's host logits in the coherence band
      (RULE 11c) and to its tokens. `engine/run.TestRunnerDecodesAsTheHost`
      runs it on every backend a host opens from `Plan.Storages` alone, each
      buffer poisoned and a poisoned guard past its stated size. One plan,
      composed at a session's capacity, serves every shorter history: the
      attention reads each step's bounds from the row descriptors, and the
      code changes only where the capacity changes the attention's splits.
   3. *The engine runs the plan* for decode on one device in place of the
      tier, for that model; then prompt rows, then the other devices.
      `engine/run` is the runner: a `Runner` is the plan composed and
      compiled once at a capacity, with the weights, word blocks and rotary
      planes every sequence shares uploaded once; a `Session` is a
      sequence's pools, page table, tables, input row and step buffers. A
      warm step allocates nothing. Next, `model.State` running decode
      through it on a model opened with a `With...()` option; relocation and
      paging come back as re-planning when placement changes, gated by the
      five principles' tests, before it replaces the tier.

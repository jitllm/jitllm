# The dependency-counter decode block

The persistent decode block (`Config.Persist`, `ir.Persist`, `tier/persist.go`)
composes a block's per-op launches into one cooperative grid and separates
dependent launches with a grid barrier. It is bit-identical to the per-op path
and slower: a grid barrier costs 0.8-1.9 us on a V100 against ~0.6 us for a
launch boundary inside a CUDA graph, and a decode block is a chain, so
composition swaps one for the other almost one for one (V100 0.89x, T4 0.96x;
`docs/engineering-history/gpu-kernels.md`, "A persistent decode block").

This note is the design of the next step: the same composition, with the grid
barrier replaced by **completion counters**. A task waits only on the tasks it
reads (or overwrites), and a workgroup with nothing to do in a task does not
wait for it at all.

## Prior art read

- **Mirage Persistent Kernel** (`mirage/include/mirage/persistent_kernel/`):
  every task names one dependent event and one trigger event; an event is a
  64-bit counter in global memory, triggered with `atom.add.release.gpu` and
  waited on by thread 0 spinning on `ld.acquire` with `__nanosleep`, then
  `__syncthreads`. The target is `num_triggers * iteration`: the counters only
  grow, so nothing resets them between iterations.
- **Hazy "No Bubbles" megakernel** (`HazyResearch/Megakernels`,
  `demos/low-latency-llama`): a `Bar[layer][opcode][chunk]` array of counters;
  a consumer instruction spins (volatile load, `__nanosleep`) until its
  producer's chunk has reached the count it needs, a producer `atomicAdd`s its
  chunk when done. Chunks are per head group for attention and per row range
  for the matvecs.
- **FlashInfer / TensorRT-LLM** persistent kernels use the same
  flag-and-spin shape for split-K and all-reduce; nothing new for decode.

Both megakernels are hand-written. Here the tasks are the per-op kernels' own
IR (RULE 8: generated code, and the per-op path stays the reference), so the
dependency graph is derived, not written.

## The task graph of a block

A task is one launch's body (an `ir.Part`); its tiles are the launch's
workgroups (virtual groups, dealt over the grid as `ir.Persist` already does).
For Llama-3.2-1B on sm_70 the block is roughly:

    rmsnormquant -> q/k/v matvec -> split-K reduce -> rope + KV write
      -> flash attention (per head) -> merge -> quantize -> o matvec
      -> reduce + residual -> rmsnormquant -> gate/up matvec (ActIn)
      -> quantize -> down matvec -> reduce + residual

The edges are of two kinds, and which kind is decided by **the arithmetic**,
not by a choice of this code (RULE 11e):

- **All-to-all.** A row of a matvec is a dot over the whole input vector, so
  every tile of the o projection reads every attention head, every gate/up
  tile reads the whole normed input and every down tile the whole act(gate)*up.
  A tile of such a task waits for the whole producer. "o-projection tiles start
  as their heads finish" is not available at decode: no o row depends on fewer
  than all heads. Split-K partitions the K axis, so a partial tile depends on a
  K-range of the input -- that is the one place a matvec tile has a narrower
  dependency, and it is a range of the producer's tiles, which the general rule
  below covers.
- **Tile-local.** Elementwise and row-range steps (quantize, rope, the split-K
  reduce, the residual add, the per-head merge) read exactly the elements a
  bounded set of producer tiles wrote. Their tile waits on those tiles alone.

The composer decides the kind per (producer, consumer, buffer) by an interval
analysis of the IR (`ir.groupSpan`): an access index of the form
`ctaid*S + r` with `r` in a known bounded interval (thread index, constants,
lane, a clamp to the buffer's last element) gives each tile's element range.
When every access both tasks make to the buffer has that form, consumer tile g
waits on the producer tiles whose ranges meet its own; otherwise (a
loop-carried index, an indirect index) it waits on the whole producer. The
analysis is conservative by construction: anything it cannot bound is
all-to-all.

Read-after-write, write-after-read and write-after-write are all edges (the
same buffer is reused through a block). Edges that are implied by another
whole-task edge (j waits on all of k, every tile of k waited on all of i) are
dropped.

## The counter protocol

One u32 buffer per session (the "dep buffer"), carved into a region per
composed kernel at plan time. A region holds: the epoch, the exit count, one
completion counter per task that some consumer waits on wholly, and one word
per tile of each task that some consumer waits on tile-wise.

- **Epoch.** Every workgroup reads the region's epoch E at the start. Counters
  never reset: after launch n every task counter holds n*A (A = workgroups
  active in the task) and every tile word n. A wait's target is therefore
  (E+1)*A or E+1, compared modulo 2^32 (`(int32)(cur - target) >= 0`), so the
  counters may wrap. This is Mirage's iteration-scaled target, made
  launch-scaled so CUDA graph replays keep working with no memset node.
- **Exit.** Every workgroup arrives on the exit count when it is done; the last
  arrival zeroes it and advances the epoch. Every workgroup read E before it
  arrived, so none can read the new one in this launch; the next launch is
  ordered after this one by the stream.
- **Signal** (a tile or a task done): `bar.sync`, then thread 0
  `red.release.gpu.global.add.u32`. The barrier orders the block's writes
  before thread 0 at CTA scope, and a release is cumulative, so it publishes
  them (PTX ISA, "Memory Consistency Model": release patterns, causality order
  and its base case through `bar.sync`; the same argument the grid barrier's
  lowering makes in `ptx/grid.go`).
- **Wait**: thread 0 spins on `ld.acquire.gpu.global.u32` with `nanosleep`
  backoff (sm_70+; below sm_70 `ld.volatile` and `membar.gl`), then
  `bar.sync`. An acquire load reading a value written by a release RMW
  synchronises with it, and every RMW on the word extends its release
  sequence, so the waiter sees every signaller's writes; the barrier carries
  that to the block.
- **No stale reads.** The composed kernel's loads are plain `ld.global`, never
  `.nc`/`ld.global.nc` (`ptx/lower.go`): the non-coherent path is not ordered
  by acquire.

## Deadlock freedom

1. **Co-residency.** The kernel is launched cooperatively on a grid no larger
   than the occupancy calculator's resident limit (`backend.GridLimit`, the
   CUDA Programming Guide's cooperative-launch requirement), exactly as the
   grid barrier is. A dep kernel reports `GridSync()` so it takes that path.
2. **Waits point backward.** Each workgroup runs its tasks in program order,
   and a tile of task j waits only on tasks i < j. Take the blocked tile with
   the smallest (task, tile) it is in: the producer tiles it waits on belong to
   earlier tasks, whose owners are resident, and either running (progress) or
   blocked on a still earlier task -- which contradicts minimality. So some
   workgroup always progresses.
3. **Uniform control.** A wait and a signal contain a `bar.sync`, so every
   thread of the workgroup must reach them: they sit outside the per-task loop
   (whole-task waits and signals, under a 0/1 loop on whether the workgroup has
   any tile of the task) or inside a loop whose count is the workgroup's own
   (tile waits and signals, only when one workgroup runs one tile at a time;
   a packed narrow task, whose threads run different tile counts, takes
   whole-task edges only).

## What the device APIs promise elsewhere

- **Vulkan**: no inter-workgroup forward-progress guarantee (the
  specification's execution model; `VK_KHR_shader_maximal_reconvergence` and
  the memory model extension define ordering, not scheduling). A spin on a
  workgroup that has not been scheduled can hang.
- **Metal**: the Metal Shading Language specification defines device-scope
  atomics and fences and no progress guarantee between threadgroups; the
  grid barrier there already relies on a resident-count it cannot prove.
- So the dep ops are lowered for PTX and declined **by name** by the SPIR-V,
  MSL and AMDGPU lowerers (`errDepNotLowered`), and `persistOK` keeps the
  whole path CUDA-only, as it was.

## The IR

Five ops (`ir/depsync.go`):

| op | args | meaning |
|---|---|---|
| `dep.task` | Imm = task id | marks where a task's body begins; lowers to nothing |
| `dep.epoch` | buf, Imm = word | the region's epoch (every thread loads it) |
| `dep.wait` | buf, word, target; Imm = producer task | spin until word >= target, acquire, workgroup-wide |
| `dep.signal` | buf, word; Imm = own task | add 1 with release, workgroup-wide |
| `dep.exit` | buf; Imm = region base | arrive; the last arrival advances the epoch |

`Validate` extends RULE 13 rather than weakening it: the dep buffer is touched
only by these ops; inside one task a buffer is never both read and written;
and a task that reads a buffer an earlier task wrote, or writes one an earlier
task touched, must hold a `dep.wait` naming that task before its first access.
A grid barrier and the dep ops are not mixed in one kernel.

## Composition with the five principles

- **JIT / no Go compute**: the ops are IR, lowered at run time; nothing runs in
  Go but the plan.
- **Relocation, paging, KV pages**: unchanged from the persistent block -- the
  plan is keyed by kernels, grids and aliasing; buffers are bound per launch,
  so a moved block, a paged-in block or a new KV page binds new buffers to the
  same plan. The dep buffer is per session (two sessions never share
  counters); a region is per plan, so two plans never share a counter whose
  value the other's targets assume.
- **Batching**: a batched step's launches are collected the same way; nothing
  in the protocol knows the row count.
- **Allocation**: the dep buffer is allocated once per session outside any
  submission, like the grid barrier's sync buffer; a warm token allocates
  nothing new.

## What is not in this step

- **On-chip intermediates** (norm stats, the quantized activation, act*up
  kept in registers or shared memory between a producer and its consumer
  tile) need tasks that are fused kernels, not the per-op bodies; the existing
  fusions (#110 ActIn, #87 rope+KV write, the residual epilogue) arrive inside
  the per-op kernels and are composed as they are. Fusing across tasks is the
  composer's step after this one.
- Tile edges for a packed narrow task.

## After the counters: the ceiling, measured

The counters lost (gpu-kernels.md, "The dependency-counter block"), so the next
lever has to remove work rather than reorder it. Before choosing one, the
whole-token efficiency: eta = resident weight bytes x tok/s over a read wall
measured in the same process (`jitllm speed` with `JITLLM_SPEED_WALL=1`,
`cmd/jitllm/wall.go`: a grid-strided one-word stream over 2 GiB, median of 21).
V100-SXM2-16GB, CUDA, graph replay, p64 tg256, three runs each:

| model | GB/token | tok/s | GB/s | wall GB/s | eta |
|---|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 0.975 | 557.1 / 563.8 / 563.8 | 543 | 860.3 | 63.1% |
| Llama-3.2-3B Q4_K_M | 2.522 | 245.9 / 246.3 / 246.3 | 620 | 858.5 | 72.2% |
| Llama-3.1-8B Q4_K_M | 5.171 | 139.1 / 138.4 / 139.2 | 719 | 864.5 | 83.2% |

The bytes are the resident weights, each read once a token; KV at 64-320
positions is under 1% of them. The ceiling with the bytes unchanged is 1.58x
(1B), 1.39x (3B) and 1.20x (8B).

Per-kernel device time (`JITLLM_CUDA_KERNEL_TIMING=1`, one event pair a
launch, which adds ~3.9 us to each and so overstates the small kernels): on
the 1B the matvecs are ~1820 of 2385 us; rmsnormquantw, flashkvpaged,
ropeqkvpaged and flashmergewide ~550 us over 64 launches, most of it the
events. On the 8B the matvecs are ~7200 of 8620 us.

What this says about the candidates:

- **k-slice chaining (gate/up into down, heads into o).** At decode each
  slice of the intermediate feeds every output row, so a chain either re-reads
  weights, holds all H accumulators per workgroup, or writes 4H(F/S) bytes of
  f32 partials and reduces them: for the 1B (H 2048, F 8192) at S = 256 that is
  256 KiB a block written and read again, against the 32 KiB intermediate it
  removes, and a fixed-order reduction keeps one launch boundary anyway. It does
  not clear its traffic budget, and its arithmetic would not be bit-identical
  to the per-op path without that reduction. Not built.
- **Residual and norm statistics as epilogues** remove one small launch a
  sublayer (the norm's 16 x 2 on the 1B), a fraction of the ~37% the 1B leaves
  on the table; exact only if the epilogue keeps RMSNormGroup's reduction tree.
- **Per-card tuning of the matvec decomposition** (split, group split, row
  tile, ActIn eligibility) acts on the ~75% of the token that is matvec time,
  at eta 63-83%: the largest share of what is left, and it changes no
  arithmetic order the gates pin beyond what the split tuner already chooses.

So the order is: the matvec decomposition per card first, the exact
norm-statistics epilogue second, chaining not at all at decode.

### Headroom in the decomposition, probed

Before building a joint tuner: does any decomposition the knobs can force beat
the tuned default? V100, CUDA, graph replay, p64 tg256, three runs each, one
process per configuration, the default run first and last:

| configuration | Llama-3.2-1B tok/s | Llama-3.1-8B tok/s |
|---|---|---|
| tuned default (first) | 545.6 / 552.9 / 551.8 | 138.7 / 139.0 / 138.8 |
| prologue on every site (ActInSites 15) | 540.6 / 539.1 / 542.2 | 117.2 / 117.5 / 117.1 |
| prologue on down only (8) | 540.1 / 538.6 / 538.4 | 132.0 / 132.0 / 132.0 |
| prologue on o only (2) | 544.4 / 546.4 / 543.3 | 137.8 / 138.0 / 137.9 |
| no act(gate)*up fusion | 537.4 / 536.1 / 541.7 | 137.5 / 137.6 / 137.6 |
| row tile 2 | 421.6 / 421.3 / 420.2 | 119.9 / 120.1 / 120.1 |
| row tile 4 | 362.5 / 364.5 / 357.6 | 109.0 / 109.2 / 109.2 |
| split 4 everywhere | 332.2 / 334.1 / 333.6 | 95.8 / 95.6 / 95.6 |
| split 8 everywhere | 404.7 / 405.1 / 404.7 | 114.4 / 114.5 / 114.5 |
| split 16 everywhere | 440.8 / 440.0 / 440.2 | 120.5 / 120.6 / 120.7 |
| the table, no tuning | 464.5 / 465.6 / 463.5 | 124.6 / 124.5 / 124.8 |
| tuned default (last) | 555.0 / 556.6 / 555.9 | 138.3 / 138.4 / 138.6 |

Nothing beats the tuned default; the tuner is worth 1.19x on the 1B and 1.11x
on the 8B over the table. What is left is per shape, and the per-kernel times
say where: the 8B's large matvecs run at ~80-87% of the wall (the 14336x4096
gate/up ~44 us for 33 MB), while the 1B's 2048x2048 q/o matvec runs ~8 us for
2.4 MB, about a third of it. A small matrix is bound by its fill and drain,
not its decomposition: no split or prologue choice moves a fixed few
microseconds a launch, which is the 1B's missing 37%. The levers for that are
launches that overlap their neighbours' tails (programmatic dependent launch,
sm_90 only) or fewer, larger matvecs (q/k/v and gate/up already run as one).

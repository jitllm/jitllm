# The composer, from zero

`jit/` is the composer, written from nothing: package `jit` at its root, no
`composer` folder. The old JIT -- its tiers, its composer, its CPU emitters --
moved to `jit_old/`, frozen: nothing new calls them, and they are deleted once
the composer covers the models that use them. `engine/` owns the runtime and
executes what the composer plans. The CPU is a device like any card: one more
target the composer forms kernels for.

## What it is

    engine/graph block ──► L0 primitive ops ──► L1 loop nests ──► passes ──► emit ──► Plan
     (graph ops, shapes)    (decompose)          (naive lowering)  (switchable)  (ir.Kernel)  (kernels, launches, buffers)

1. **Input.** One block of `engine/graph` (its ops, values, weights by role),
   its shape facts (rows, decode or chunk, KV page geometry, run-time bounds as
   symbols) and a `Target`: a GPU (API, ISA word, subgroup width, shared
   memory, grid limits) or the host CPU (its tier: SSE, AVX2, AVX-512, NEON,
   dotprod; its cores, caches and vector width).
2. **L0, primitive ops.** Every graph op decomposes into a handful of
   primitives over named dimensions: `Map` (elementwise), `Reduce`, `Contract`
   (a sum of products: matvec, GEMM, attention's two products), `Gather`/`Scatter`
   (an index through a table: pages, experts, rows), `Scan` (a carried
   recurrence) and `Layout` (copy, slice, concat, transpose). This
   decomposition (`decompose.go`) is the only code that names a graph op kind.
   A composite graph op (attention, a recurrent mixer, a route) becomes several
   primitives, never a kernel of its own.
3. **L1, loop nests.** Each primitive lowers naively to one loop nest: a
   domain (dimensions with constant or symbolic extents), a body of scalar
   nodes (load, store, constant, unary, binary, compare, select, integer and bit
   ops) and explicit reductions. One kernel a primitive, one element a thread,
   nothing fused, no schedule. Facts a pass needs ride on the nodes as hints.
4. **Passes.** Each is a registered transform with an on/off switch and
   parameters, run in a stated order:
   - fold (constants)
   - simplify (integer canonical forms)
   - CSE
   - fuse (vertical: a producer inlined into its consumer; horizontal: siblings
     over one operand)
   - split, tile and reorder
   - bind (dimension to thread, lane, warp, workgroup or loop)
   - unroll
   - chains (reassociating a reduction)
   - rfactor (a split reduction and its combine)
   - stage (workgroup memory, barrier, double buffering)
   - vectorize
   - mma (matrix-instruction mapping)
   - prefetch
   - hoist

   A `Profile` per target holds the switches and parameters, set to main's
   choices first and widened later.
5. **Emit.** One emitter per device kind from a scheduled loop nest: to
   `ir.Kernel` for a GPU, to machine code through the CPU instruction encoder
   for the host. The cache key is the nest's structural hash and the target,
   never a name.
6. **Plan.** The kernels; each launch's grid and its arguments, bound to plan
   buffers; the buffer plan (values with sizes and lifetimes, weights by role,
   KV and state views, scratch). Buffers are assigned **after** fusion, so a
   fused chain never reads and writes one slot. Today's tier colours slots
   first, which is why its map chains could not fuse.
7. **Executor (engine/).** Uploads weights by role, allocates plan buffers,
   binds KV pages, launches on a `backend.Device`, and implements
   `nn.LayerDevice`. Placement, paging and sessions move from the tier into
   `engine/` as the executor takes over.

## Gates that make a fallback impossible

- **srcgate, imports:** `jit/` imports only the standard library,
  `engine/graph`, `format/jlm` (roles), `format/quant` (formats) and the two
  device boundaries: `jit_old/gpu/ir` (what the GPU backends lower) and the
  CPU instruction encoder (split out of `jit_old/cpu` into its own package
  first, so no `Emit...` kernel is reachable). Nothing else of `jit_old/`,
  and never `paritytest`.
- **srcgate, names:** a graph op kind, a family or a model name appears only
  in `decompose.go`. Passes, the emitter and the planner never switch on one.
- **No fallback:** a block the composer cannot form is an error, and the block
  does not place.
- **Pass table:** every pass is in the table. Turning any pass off still gives
  numerically correct kernels, checked in the interpreter (below).

## How it is checked, on the CPU, before any hardware

- **The host runs the CPU's kernels natively**, held to `internal/oracle`
  like everything else; its main is `jit_old/cpu`'s hand emitters.
- **An `ir.Kernel` interpreter** (`jit/irexec`, test support): runs a
  kernel's grid on the host. The naive lowering and every pass are held to
  `internal/oracle`. This is what keeps a pass honest: correct first, then
  identical.
- **Unit parity** (`paritytest`): per op and shape, the composed kernels against
  main's, instruction by instruction (dataflow-identical).
- **Board v2** (`paritytest/board`, moved out of `engine/model`: the engine
  never imports main's tier): per profile, the blocks the new composer formed out of the
  total; the % of main's instructions matched over those blocks; models
  identical out of 107; the top 10 kernels by missing instructions. 100% of
  launches come from the new composer by construction, because nothing else
  is called.

## Order

| phase | scope | done when |
|---|---|---|
| 0 | the package skeleton, the pass table, the srcgates, the interpreter, board v2 | gates green, board runs and reports 0 blocks |
| 1 | synth-ernie45 (llama's graph exactly), decode, on cuda-sm70 and on the host CPU (AVX2) | every block formed on both; interpreter (GPU) and native runs (CPU) equal the oracle; every GPU kernel identical to main's |
| 2 | the same model's prefill (chunk rows): main's staged and matrix-instruction GEMM and attention forms as passes | identical on sm70 |
| 3 | the same model on every profile (sm75–90, hip, vulkan, metal) | identical on all 14 |
| 4 | the engine executor on hardware: V100, laptop, M4 | output matches; paging, relocation and zero-allocation gates; A/B not slower |
| 5 | model by model: dense families, MoE, MLA, hybrids and SSM, vision, the rest; each adds decompositions and generic passes only | 107/107 identical on every profile |
| 6 | the CPU's instruction streams held to `jit_old/cpu`'s emitters on every tier (SSE, AVX2, NEON, dotprod) | identical |
| 7 | delete `jit_old/`, then `paritytest/` after the hardware gates | the tree holds the composer alone |

## Open risks

- The interpreter must itself be right: it is gated against `internal/oracle`
  on hand-written kernels before anything relies on it.
- Paged and indexed access as `Gather` must be expressive enough for main's
  page arithmetic without an attention-shaped special case.
- Run-time extents (keys, live rows) are symbols the passes reason over by
  range (hints), not constants.

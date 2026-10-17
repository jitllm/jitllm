# The kernel composer: fusion recipes chosen per card by timing

AGENTS.md names the standing goal: the JIT should compose kernels, choosing and
fusing from the kernels it already knows by the operation graph and the
hardware, rather than hand-writing each fusion into a call site. This note is
the representation that goal starts from and the first slice built on it.

## What exists, and why it is not yet a composer

The device tier already makes four per-card fusion decisions, each wired by
hand into `encodeLayer`'s launch sequence and each with its own switch:

| fusion | built by | decided by | evidence |
|---|---|---|---|
| operand prologue (norm/quantize inside the matvec), sites q/k/v, o, down | `fuseActIn` | `tuneActIn`: timed per shape key, cached `gpuactin` | GK "The prologue decided per card by measurement" |
| gate+up as one launch, down forming act(gate)*up (`gu`) | `fuseGateUp` | `Config.GateUpFuse`, off | GK "Gate and up in one launch: exact, and a loss past the 1B" |
| gate+up as one launch summing both rows (`Gate2`) | `gate2Kernel` | `tuneGateUp2`: Gate2 against two launches per up shape, cached `gpugateup2` | GK "... each workgroup summing both rows (Gate2)" |
| split / in-group split | `retuneDecode` | timed per shape, cached `gpusplit` | GK "split tuner" |

The FFN's three forms were a precedence chain (`GateUpFuse` before `Gate2`
before two launches): a boolean for `gu`, and a two-arm tuner for Gate2 that
could not see `gu` or a form composed with the norm. `gu` won on the 1B (1.006) and lost 5-9% past it; Gate2 won 1-3%
on all three. A different card, a different FFN width or a different format
can reorder them, and the chain cannot see that. The operand prologue is
already a per-card choice, but its tuner is specific to its two arms.

## The representation

A **site** is a place in a block's graph where more than one generated form
computes the same values: the q/k/v operand, the attention output's operand,
the FFN's gate/up/down chain, down's operand. A **recipe** is one form at a
site: which kernels it launches, in what order, over which buffers. A recipe
is data: a name, a build step (which kernels it needs; nil where the shape
cannot express it), and an arm (the launches it makes for one block, the way
`encodeLayer` makes them). The composer's state is, per site and per shape
key, the index of the chosen recipe.

    type recipe struct {
        name  string
        ok    func(l *layer) bool                    // built and expressible here
        arm   func(s backend.Session, l *layer) error // one block's launches
    }
    type site struct {
        name    string
        key     func(l *layer) string // shape key; "" = not offered
        recipes []recipe               // recipes[0] is the reference form
    }

The shape key holds what decides cost and nothing else: format, k, rows,
split and reduction form of every matvec in the site. Two blocks with the
same key share a verdict, which is what lets a 32-block model time 16 blocks
once. The verdict is cached per binary, device and key (`gpurecipe`), as the
split tuner's is.

### Correctness is per recipe, not per site

Every recipe is held to the reference within a measured band, and its gate has
a violation that must fail. Where a recipe is exact on a backend (Gate2 on
PTX, `gu` everywhere), its gate is bit equality with recipe 0 and a swap
violation; where it is not (Gate2 on SPIR-V, where the driver contracts sums
differently), it is not offered on that backend until its band gate exists.
Recipes do not have to agree with each other bit for bit; byte-identical
goldens guard only a refactor of one recipe (emitter-patterns.md).

### The search

For each site, for each key, every offered recipe is timed over up to 16
placed blocks with that key, interleaved within the process (ABAB.., the arm
order rotated per round), three repetitions a round, ten rounds with the first
discarded for first touch. Each recipe's ratio is the median of its per-round
ratios to recipe 0; a trial whose IQR/median exceeds the split tuner's gate
(`splitTuneIQR`) does not resolve and keeps recipe 0. A recipe is kept only
where it is at least 1% faster than every recipe before it in the ranking
(`recipeMargin`, the prologue tuner's margin), so a tie keeps the simpler form.

Sites run in dependency order: a site whose arms include another site's
launches is timed after that site settles, with its verdict in force. The
FFN site's arms include down, so the down-prologue verdict is decided first
and down runs in the form it ships in. This is a greedy, not a joint, search:
a joint search over the product of sites' recipes is exponential, and the
measured interactions (down's prologue against `gu`'s gated prologue) live
inside one site's arms where they are timed together.

### What a forced choice does

Every switch survives as a forcing option (`With...`): `GateUpFuse` forces
`gu` where it is built, `GateUp2=false` removes Gate2 from the offer,
`ActInSites` forces the prologue sites, `NoFuse` keeps recipe 0 everywhere,
and `JITLLM_GPU_TUNE=0` keeps the recipe the defaults name (the fitted path:
Gate2 where exact, the prologue nowhere). Counters (`Stats().RecipeTimed`,
`RecipeKept`, per-recipe run counts) let a gate assert the configuration it
claims was selected (RULE 10).

## The five principles

- **JIT**: every recipe's kernels are emitted per card; the composer chooses
  among generated kernels and emits nothing pregenerated.
- **No Go compute**: the search is timing; no token arithmetic moves to Go.
- **Relocation and paging**: a recipe is per block and keyed by shape, so a
  block moved to another card takes that card's verdict for its key; a block
  paged out is skipped by the trial (`l.pg != nil && !l.pg.in`), as the
  prologue tuner skips it.
- **No allocation**: the trial runs at tune time, outside a token; a token's
  launch reads an int per site from the layer.

## Candidates, ranked by what the profile leaves

Decode on the V100 runs matvecs at 60-90% of the read wall; launches removed
by fusion have repeatedly measured below noise (RoPE+KV) or lost (persistent
and dependency-counter blocks). The candidates below are ranked by what a
recipe can buy without adding traffic.

1. **FFN gate/up/down as a site** (two launches, Gate2, `gu`): built; replaces
   two hand-wired booleans. First slice.
2. **Operand prologue as sites** (q/k/v, o, down): already timed; becomes three
   sites of the same tuner. First slice.
3. **Gate2 with the FFN norm's prologue** (`gate2in`: one launch normalising
   and quantizing the residual, summing the gate's and up's rows and writing
   act(gate)*up). The norm launch and the gate launch both go. The prologue at
   gate and up lost before because two launches each redid the norm; under
   Gate2 one launch does. Both pieces exist (`kernels.ActIn`,
   `MatVecShape.Gate2`); the recipe is their composition, the first found by
   composing rather than written.
4. **Residual + next norm statistics in down's epilogue**: the next block's
   q/k/v prologue recomputes the row's sum of squares in every workgroup;
   the codex review notes the exact form needs the existing partial order,
   and that a block must remain independently runnable after relocation.
5. **Attention output with o's prologue** and **RoPE into the q/k matvec
   epilogue**: small on the measured profile; kept as candidates.
6. **Host-side region fusion** (CPU pool): the dispatch cost is 2.7 us a
   region; the same table shape applies with the pool's regions as recipes.
7. **Prefill GEMM epilogues**: another agent's work (the add-norm fold);
   not duplicated here.

## The first slice, as built

`jit/gpu/tier/recipe.go`. `timeRecipes` is the one trial: N arms
interleaved in one process, the order rotated each round, 20 paired rounds
after one discarded, each challenger's per-round ratio to the incumbent,
median and IQR gate. `tuneActIn` runs on it (its three sites, two recipes
each), and `tuneFFN` replaces `tuneGateUp2`: per FFN shape key it times
two, gate2, gate2in and gu (each built where the shape takes it) norm
through down, keeps the device's default (Gate2 where exact, else two)
unless a challenger is 1% faster, and writes the Gate2 verdict
`gateUp2On` reads. `Config.FFNForm` (`JITLLM_GPU_FFN`) forces a recipe;
`GateUpFuse` and `GateUp2Force` still force theirs. `mvLaunch` is the one
launch the tuners' arms make, matching `encodeLayer`'s workgroup width,
row tile and partial reduction.

The recipe found by composing: **gate2in**, Gate2 with ActIn's norm
prologue. Where the attention output's launch already added the residual,
the FFN's RMSNormQuantRows goes and gate and up read the residual row
themselves: two launches become one per block. It is bit-identical on PTX
(`backend.TestGate2InIsTheTwoLaunches`, 96 cases; `model.TestFFNRecipesMatchTwoLaunches`
with every block placed with and without the graph and a half seam;
violation `TestGate2InSeesASwap`). The tuner keeps it on both of
Llama-3.2-1B's FFN keys on the V100. Rows: GK "The FFN's recipes, timed per
card".

## Above the recipes: the block graph and the rewrite interface

The first slice decides recipes at sites `encodeLayer` names. `ffnKey` is the
cost of that: a site is recognised by a negative list of thirty-odd fields
(`l.router`, `l.mla`, `p.Parallel`, `l.nPostFFN`, ...), and every new
architecture or feature has to remember to extend it, or the tier offers a
recipe whose arms do not compute the block's function. The step past it is to
find sites by matching the block's operations, which needs the operations
written down once. This section is that representation and the interface a
fusion or partitioning candidate implements against it; the reasoning and the
precedents are in [external-optimization-ideas.md](external-optimization-ideas.md)
("Toward a model-graph JIT", "Joint fusion and partitioning"), and the
workflow an implementation follows is [optimization-agent-harness.md](optimization-agent-harness.md).

It changes no AGENTS.md rule. The block stays the unit (RULE 8a), every
operation stays generated (RULE 8), and a limit a candidate declares names its
source (RULE 11e).

### The block graph

A **block graph** is a versioned, typed, SSA description of one block's values
and state effects, built in `engine/graph` (data and a verifier, no
arithmetic) from what the container states: `jlm.Config` and the block's
weight inventory by role.

    type Graph struct {
        Version uint16       // the op language's version; an unknown one is refused
        Block   int          // the graph is per block
        Values  []Value      // SSA: each written by exactly one op or a boundary input
        Ops     []Op         // in a legal order
        In, Out []ValueID    // the residual at the block's boundary
        State   []StateRef   // the block's persistent state: its KV, its recurrence
    }
    type Op struct {
        Kind   OpKind  // norm, head_norm, matvec, rope, kv_append, attend, add, act_mul, ...
        V      uint8   // the op's semantic version: a norm states eps placement, rope its pair layout
        In     []ValueID
        Out    []ValueID
        Weight []Binding         // jlm role and block, with the shape the op requires
        Reads, Writes []StateVersion // kv@0 -> kv@1: the effect, not a value
        Attr   Attrs             // eps, rotary layout/base/nrot, window, activation, softcap
    }

**State is versioned, not aliased.** `kv_append` reads `kv@n` and writes
`kv@n+1`; `attend` reads `kv@n+1`. A schedule that moves an attention read
above its append has a read with no producer, which the verifier refuses before
anything is emitted. A recurrent block's state has the same shape
(`rec@n -> rec@n+1`). Pages, leases and transfers are not in the graph: they
are the execution plan's binding of a version to memory, which is why a block
that relocates keeps its graph.

**The block boundary is a relocation point.** A graph's inputs and outputs are
the residual between blocks, materialized. A rewrite may span blocks only as a
candidate conditional on the blocks' co-placement and state ownership; the plan
discards it at the next token boundary when either block moves, and the
per-block graphs are what remain. Nothing in the graph requires a fused
multi-block region to stay resident.

**The verifier** checks, before allocation or emission: every value has one
producer; every op version is known and has a lowering on the tier asked;
every state read has a producing write ordered before it; every weight a
binding names exists with the shape the op requires, and **every weight the
container holds for the block is bound by some op** (an unconsumed `q_norm` is
a missing q/k norm, not a harmless extra); the rotary layout and every
attribute equal what the config states; the outputs are produced. It sizes
nothing from an unchecked count (RULE 9).

**Lowering.** The graph lowers to the operations the engine already runs: on
the host, the `State` helpers `forward.go` calls (`s.norm`, `s.mv`,
`attnPrep`, the attention) and through them the generated kernels; on the
device, `nn.LayerPlan` and `nn.LayerWeights`, which the tier already consumes.
There is no new kernel and no second family switch: the walker dispatches an op
kind to the helper that runs it today. While a graph covers fewer blocks than
`forward.go`, the hand-written path is the control and the default; a covered
block runs the graph only when the caller forces it, and the gate compares the
two. The walker replaces the hand-written dense path once it covers decode,
batch and device for the dense family (Slice B), not before: it is a migration
under test, not a second executor to keep.

### The rewrite interface

A rewrite is the unit an optimization is written as. It is data the composer
selects from, and it carries everything a plan explanation needs to say why it
was or was not taken.

    type Rewrite struct {
        Name     string
        Match    func(g *graph.Graph) []Region         // by op kinds and dataflow, never by arch
        Legal    func(g *graph.Graph, r Region) error   // shape, format, layout and state guards; the error is the reason
        Contract Contract                               // exact against the reference recipe, or a band, per backend
        Bounds   func(r Region, d Target) Resources     // launches, scratch, extra reads, registers: static
        Build    func(r Region, d Target) []Candidate   // the recipes it offers; [0] stays the reference
        Keys     func(r Region, d Target) string        // what makes a verdict stale: formats, shapes, device, placement
    }

- **Match** is structural: op kinds, single-consumer edges and shared operands.
  A rewrite never asks for an architecture; a pattern that needs one means the
  op language is missing a distinction, and the fix is an op or an attribute.
- **Legal** returns a reason, which the plan explanation prints. A near miss
  (the shape matched, a guard did not) is a recorded rejection, so a gate can
  assert the rewrite refused it.
- **Contract** is per backend, as recipes already are: Gate2 is exact on PTX
  and is not offered on SPIR-V until its band gate exists. A band contract
  names its band and the gate that measures it.
- **Bounds** lets the search refuse statically (a partition whose partials cost
  more bytes than the intermediate it removes), never accept: only a timed
  complete region accepts.
- **Keys** names what makes a verdict stale. A verdict is cached under it and is
  evidence for that key only; a block relocated to another card takes that
  card's verdict or the reference recipe.
- **The target is a guard dimension, not an if at a call site.** A recipe
  that pays on one device and is inert on another says so in a `Pays` guard
  over a `Target`: the device's capability (ISA tier, the MMA shape it offers,
  subgroup width, shared memory), the executed row-width class (one-row
  decode, a few rows such as a speculative verify or fewer routed pairs than
  the bank has experts, a prompt chunk's GEMM width) and the layout facts of
  the rows (a stream-major residual, which every row slice must take whole).
  Examples: a deep-tile GEMM verdict that needs sm_75's m16n8 tiles is refused
  on sm_70 by name; a few-row verify and a prompt chunk are different row
  classes with different winning forms. Every persisted verdict carries the
  target's key, so evidence at one class or on one card is never read at
  another.
- **Bit identity is per backend, never across devices.** A block's generated
  code may differ per device and per model; the bar is the gated error margin
  and the speed. An exact contract means exact against the reference recipe on
  the same backend.

The search is the existing one: `timeRecipes` over the matched region's
candidates, interleaved, with the margin and IQR gates above. A rewrite adds
candidates, not a tuner.

### How the candidates plug in

| candidate | match | legality and contract | why it is a candidate and not a default |
|---|---|---|---|
| norm -> gate/up -> act -> down (the FFN site; `gate2in`) | a `norm` whose only consumers are two `matvec`s on its output, feeding one `act_mul`, feeding a `matvec` added into the residual | gate and up one format, one shape, one split (`gateUp2Shape`); the norm's input materialized by the previous add; exact on PTX | it removes a launch, and the outcome is per card and per shape (GK "The FFN's recipes, timed per card") |
| q/k/v operand prologue | `norm` -> the q/k/v `matvec` group | the norm's single consumer group; exact | already timed by `tuneActIn`; its sites become matched |
| residual + next norm's statistics in down's epilogue | an `add` into the residual whose consumer across the boundary is a `norm` | spans two blocks, so conditional on co-placement and dropped at a move; exact only with `RMSNormGroup`'s reduction tree | dep-block.md ranks it after the decomposition |
| hidden-channel MLP partition (`graph.MLPPartition`) | gate and up `matvec`s on one row feeding an `act_mul`, feeding a `matvec` | no bias inside a slice, no reader outside it; no contract (no recipe built); Bounds: 8*F bytes a row removed, 8*H*ceil(F/256) added, no launch removed | refused by its bound at every row class wherever H > 256 (both sides scale with the rows, so a prompt chunk does not rescue it): on Llama-3.2-1B 512 KiB of partials a row against 64 KiB. Across devices it is a placement candidate, priced by crossings, not by this bound |
| gated output norm (`graph.NormGate`, built) | a `head_norm` whose only reader is an `act_mul` gating it as the second operand | SiLU or sigmoid; exact on PTX, SPIR-V and MSL | one launch and 8*VHeads*VDim bytes a row removed at every row class; timed per card at decode |
| producer/consumer tile alignment | a `matvec` whose output tiles a following `matvec` reduces over | a parameter of a region's candidates (gate/up row tile = down's split chunk), timed jointly with the split | scheduling alone lost: the persistent block 11% and the dependency-counter block 14-18% on the V100 (GK "A persistent decode block", "The dependency-counter block"), because every matvec reads its whole input; alignment has to remove traffic or a launch to be worth a trial |

**A block split across devices** (attention on one card, the FFN or some
experts on another) is a placement candidate of the same kind: the planner may
cost it and try it, and it is never a manual mode. Its region is the block's
two sublayers with the residual between them as a crossing; the cost it must
beat is two device crossings per block per token (the residual out after
attention and back after the FFN) plus the synchronisation on each, against
the block running whole on either card. The KV stays with the attention
sublayer's card and moves with it, as a relocation does today.

Joint fusion and partitioning are one search, not two passes: a partition is a
parameter of a fused region's candidates, so the trial times the complete
region in each form, and a partition that loses alone is still timed under the
fusion it enables (the harness's complete plans).

### The first recipe through the interface: the gated output norm

The two candidates for the first recipe written as a rewrite were the gated
delta rule's output norm and gate (external-optimization-ideas.md, #6) and the
hidden-channel MLP partition. They were chosen between by counts, not timing:

| workload | gated output norm | MLP partition |
|---|---|---|
| dense decode (Llama-3.2-1B: H 2048, F 8192) | no match: a dense block has no gated norm | 0 launches removed; +448 KiB a row of traffic (512 KiB of partials for 64 KiB of intermediate); refused |
| prompt chunk, R rows | -1 launch per gated delta block per chunk; 8*R*VHeads*VDim bytes not written or read | the same ratio per row; refused |
| Qwen3-Next-80B decode (36 of 48 blocks gated delta, 32 heads of 128) | -36 launches a token; 1.125 MiB a token not written or read | the FFN is a mixture, unmatched |
| Qwen3.8-Flash-Next's recurrent output | -1 launch per recurrent block per call, sigmoid gate, under the hyper-connections (the sublayer's rows are one row each) | unmatched |

So the gated norm is built and the partition is a rewrite refused by its
bound, which `Offer` now reads (a region that removes no launch and adds
traffic is refused on every target, before the contract is asked).

**What was built.** The gated delta sublayer has a graph of its own
(`graph.Delta`): the rule as one opaque op (`gated_delta`, reading `rec@0`,
writing `rec@1`, binding the rule's tensors), the gate projection (one
matrix or KDA's two factors), the per-head `head_norm` against `ssm_norm`, the
`act_mul` (`ActSiLU` or the new `ActSigmoid`) and the output projection. It is
the sublayer's, not the block's: its input is the row the rule reads, so the
hyper-connections and the block's FFN or mixture are outside it, and its
Facts hold only the sublayer's tensors (`graph.DeltaRoles`), every one of
which an op binds. `engine/model/deltagraph.go` builds and verifies it at
every Open for each gated delta block (not Mamba's scans, not a short
convolution, not a block that also attends) and hands its matches to the tier
as the plan's `Rewrites`, with `Graphed` false: there is no host lowering,
since the host's hand-written sublayer runs the same ops in the same order.

`graph.NormGate` matches the `head_norm` whose only reader gates it, and is
refused with the reason where the normalised row has another reader or the
activation has no epilogue. Its recipe is `kernels.HeadNormGated`, HeadNorm
with the gate as an epilogue at either lane width, whose element is the same
builder calls on the same value as the two launches (`gateValue`), so the
contract is exact per backend. The tier builds it beside `outNorm`, asks the
rewrite per block at both row classes outside a token (`prepNormGate`), and
`Config.NormGate` (`JITLLM_GPU_NORMGATE`: two, fused) forces it; untuned, the
trial (`tuneNormGate`, the same `timeRecipes`) decides each key at decode, and
a prompt chunk runs the two launches, where nothing was timed. Gates:
`backend.TestHeadNormGatedIsTheTwoLaunches` (bits on PTX, SPIR-V and MSL, every
lane width, a guard after the output; the other gate and a weight one ulp off
must differ), `graph.TestNormGateIsFoundByPattern`,
`model.TestNormGateMatchesTwoLaunches` (qwen3next's SiLU, KDA's low-rank
sigmoid and Qwen3.8-Flash-Next's sigmoid, every block placed, a prompt chunk
and a decode, bit for bit; the fused launch counted once per gated delta block
per call, none in the reference arm; the tuned tier against its own two
launches; the rule's output and z swapped must move the logits). Both gates
ran on CUDA and Vulkan (an RTX 3050 Ti), CUDA on a V100 (sm_70) and Metal on
an M4, where the run set its Metal contract.

### Task contract: Slice A

```yaml
task:
  id: compiler-slice-a
  base_commit: the merge of this note
  objective: emit a verified block graph for two dense variants and run it through the existing generated operations, the hand-written path the control
  workload:
    model_and_artifact_hashes: Llama-3.2-1B-Instruct-Q4_K_M (plain llama); Qwen3-0.6B-Q8_0 (per-head q/k norm); the random-weight fixtures of each class; container hashes in the run manifest (stage 1)
    input_trace_and_seed: the gates' fixed prompts; decode past position 0
    prompt_and_generation_lengths: correctness at the gates' lengths; regression row p512 g256
    concurrency_and_context_classes: one sequence, host decode
    memory_budgets_and_relocation_trace: the default budget, and one below the model (paging)
    hardware_and_backend: laptop amd64 host (AVX2) for correctness; the device path is unchanged by construction (the graph lowers to the LayerPlan it already receives)
  metrics:
    primary: token ids and logits identical to the control (same kernels, same order)
    constraints: zero warm allocations; no new kernel; no Go arithmetic
    cold_and_warm_boundaries: graph build and verify at Open, outside a token
  scope:
    allowed_subsystems: engine/graph (new), engine/model (emission, the walker, the forcing option), docs/design
    excluded_work: batch and device lowering (Slice B); any fusion (Slice C); storing the graph in the container
    required_reuse_demonstration: the walker calls only helpers forward.go calls
  control:
    baseline_commit_and_configuration: the base commit, the forcing option off
    generated_reference_plan: forward.go's dense decode path
  levers: []   # representation only
  validation:
    independent_reference: TestGreedyMatchesLlamaCpp's goldens and the transformers fixtures already gating llama and qwen3
    required_checks_from_scripts_checks: go run ./scripts/checks on the diff
    forced_path_evidence: a counter of blocks run through the graph, asserted equal to the block count
    negative_cases: a graph without its q/k norm, with NEOX swapped for NORM, without kv_append, with a weight bound to the wrong role; each refused by the verifier
  search:
    initial_candidate_plans: none
    hardware_trial_budget: none on the token path
    expansion_and_stop_conditions: stop when both variants verify, match, and every violation fails
```

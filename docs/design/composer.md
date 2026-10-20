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
kind to the helper that runs it today. The graph is every text block's only
body -- the trunk's and the prediction blocks', on decode, prompt chunks,
ragged batch steps and the device plan. A block the language does not state
refuses the model at Open by a reason on a named list
(`model.UncoveredReasons`); no model the gates run hits one
(`TestEveryTextBlockIsCovered`). Each State records per block which path ran
(`State.BlockPaths`, the `blocks` of `jitllm speed -manifest`), and
`TestCoveredBlocksTakeTheGraphPath` holds every model to every block on the
graph on the host and every device arm.

The hand-written body this replaced was the control arm while coverage grew
(`WithBlockGraph(false)`): every family matched it bit for bit, resident and
paged, at zero warm allocations, decode, prompt and ragged step, before it was
deleted. What is left of it is the vision tower's own block (`towerRows`),
which the tower runs through the text runner until its blocks are stated too.

The mixture is lowered wherever the language states the block:
`graph.Dense.Mixture` puts the mixture's ops in place of the FFN (softmax,
sigmoid, sparsemixer or sqrt-softplus routers, renormalised or not, a
selection bias, grouped top-k, hash routing, a routed scale, a shared expert
gated or not, experts in a latent), OLMoE's q/k norm over each whole
projection is an `OpNorm` (`Dense.QKNormWhole`), and the host lowering runs
the route-to-combine run as the mixture's own operations (`moe`,
`moeExpertsRows`).

The classic block (C6) and the windowed, capped and sunk attentions are stated
the same way. `OpLayerNorm` is a LayerNorm with its bias where the block stores
one; `OpAct` is an ungated FFN's activation alone (GELU, erf GELU, squared
ReLU, ReLU, Apertus's xIELU with its four numbers bound); `OpClamp` is DBRX's
clip on q, k and v. A parallel block's FFN reads the block input -- its own
norm, or the attention's where it stores none -- and the host lowering tells
which by the value the FFN's first op reads, never by a flag beside the graph.
Projection biases, the post-norms over each sublayer's output (gemma2/3,
glm4, OLMo 2), a block with no pre-norm, a block without the rotary, a sliding
window with its local rotary, a rotary's scaling (its per-pair factors and its
cos/sin magnitude: linear, YaRN, LongRoPE), the attention softcap and the sinks
(a bound per-head vector on `attend`) are attributes or bindings of the ops
that carry them. That puts phi2, starcoder, starcoder2, command-r, cohere2,
stablelm, falcon, nemotron, dbrx, apertus, glm4, OLMo 2, EXAONE 4, arcee,
gemma2, gemma3 and gpt-oss on the graph, every block.

The flagships' forms are stated as ops too, most of them opaque in this
version (one op, its weights all bound by it, its host routine named):

- **Several residual streams.** The block's input is the stream-major
  residual. `stream_mix` is a sublayer's read of it -- Qwen3.8-Flash-Next's
  low-rank gated hyper-connection, DeepSeek V4's Sinkhorn mix, Kimi-K3's
  softmax over its bank of checkpoints -- and `stream_place` adds the
  sublayer's output back (a checkpoint block banking its input and restarting
  the running residual). Gemma 3n's AltUp is `alt_predict` and `alt_correct`
  around a block that runs on the active stream, with `laurel` beside its
  attention and the per-layer input inside the correction.
- **Selected attention.** `index` is an indexer: DeepSeek V3.2's over the
  latent cache, and over a GQA row MiniMax-M3's and QSA's block selections,
  whose key is one more cached head the attention reads past.
  `compressed_attend` is DeepSeek V4's attention with its pooled second
  history and grouped output.
- **The rest.** `ngram` (the n-gram embedding into the streams),
  `gauss_top_k` (Gemma 3n's activation sparsity), XD-RoPE as a rotary layout,
  a bidirectional `attend` (an embedding model's pass), a latent mixture,
  Kimi-K3's MLA output gate, Gemma 4's KV sharing, its v-from-k, its
  per-layer embeddings and its dense MLP beside the mixture, ungated experts,
  a recurrent mixer beside the attention, and `predict`, a prediction block's
  prologue, which the draft runs before the block.

**Coverage is structural.** Whether a block is on the graph is decided from
what the container states about the block -- the configuration's fields
(`model.uncovered`), the mixture's router, experts and shared expert
(`mixtureWhy`), and the tensors stored for it (`graph.Unbound`: a tensor no op
binds names the block `UncoveredUnboundTensor`) -- never from the
architecture's name. A family the runtime has never heard of whose blocks the
language states runs through the graph, and a renamed architecture changes
nothing (`TestCoverageIsStructural`, which fires when a name check is put
back). The names the converter reads stay in the converter.

**What still lowers through a monolithic routine.** Each op's host lowering
is recorded as data (`model.OpLowering`, the `lowering` of each block in
`jitllm speed -manifest`) and is either composed -- a generated kernel the
lowering picks by the op's shape and quant -- or a hand-written routine that
runs the op together with its neighbours. The monolithic ones are the
composer's work list:

| ops | routine today | to compose |
|---|---|---|
| `attend` | `attendRow` and the chunk's attention, one per row class | the attention kernels chosen by row class and Target |
| a latent attention's `matvec`s, `norm`s, `slice`s, `rope`s and `latent_append` | `mlaProject` / `mlaProjectRows` (one routine: both projections, the latent norms, the rotary, the absorb and the cache write) | a kernel per op; the absorb (`mla-absorb`) as the rewrite it is |
| `attend_latent` | absorbed: the attention over the latent rows, then `mlaOutProject` / `mlaOutRows`' un-absorb | the absorbed and reference forms as two lowerings, chosen by row class |
| `route`, `gather`, `expert_matvec`, `combine`, `gate_scalar` | `moe` / `moeExpertsRows` | route and combine as generated kernels, the experts as grouped matvecs chosen by the routed pairs, fusions (`moe-few-row`, `moe-waves`) as rewrites over these ops |
| `index`, `stream_mix`, `alt_predict`, `alt_correct`, `laurel`, `ngram`, `recurrent` | their routines (`msaKey`/`qsaProject`, `hcMix`, `altPredict`, `altCorrect`, `laurel`, `ngramApply`, `linearAttn`) | a kernel per inner op, as the prologue's are |
| `compressed_attend`, and every op of its block | `ds4Block` (the block whole) | the block's ops stated one by one |
| every op, on a device | the block's `nn.LayerPlan`, lowered from the graph and run whole by the tier | per-op device kernels selected from the plan's ops |

`TestEveryGraphOpHasALowering` holds every op of a covered block to an entry
(an op kind with none refuses the model at build) and the mixture's ops to
"monolithic" until they are not.

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

### When a rewrite is decided

The compiler finds what works best, and where it cannot know at compile time
it leaves the choice to whoever can. Every rewrite states three things:

- **Legality, at compile time, hard.** `Legal` and the backend's contract
  decide whether a form may run at all; nothing later overrides a refusal.
- **A cost with its evidence.** `Bounds` states the static change (launches,
  scratch, traffic, and for a split the weight elements off each card and the
  crossings a row) and `Evidence` names where it was measured or derived.
- **A decision time** (`graph.When`), and so a chooser (`Chooser`):

| when | who decides | rewrites |
|---|---|---|
| compile (`WhenCompile`) | a static guard: `Pays` at the executed row class and residency | `moe-few-row`, `moe-waves`, `mla-absorb` |
| load (`WhenLoad`) | the tier times the legal forms on this device and records a verdict keyed by the shape, `Target.Key` and the row class | `gate2in`, `actin-*`, `normgate`, `deep-tile` |
| run (`WhenRun`) | an orchestrator, for its own objective (a server fitting more models wants capacity, not speed) | `mlp-partition` |

A legal form nobody has chosen stays in the plan as an open alternative:
`Model.BlockRewrites` and `jitllm speed -manifest` list it with its cost, its
objective and who may choose it. It is taken by its chooser, or by a caller
through `tier.WithRewrite(name, graph.ChoiceTake|ChoiceRefuse)`
(`JITLLM_GPU_REWRITES=name=take,...`), never silently: a Take the device
cannot meet is counted (`Stats.RewritesUnmet`, with the reason) and explained,
and a misspelt name is refused at open. A choice outranks a form knob
(`WithNormGate`); `Choices` are validated against `graph.Rewrites`.

The MLP partition is the first run-time alternative. On one card it cannot
pay (its partials outweigh the intermediate wherever H > 256); across two
cards it is a capacity lever, half of gate, up and down off each card for two
crossings a row, and the crossings lose on speed (56-63 us a pair over NVLink
by synchronous `cuMemcpyPeer` against about 18 us of read saved, below). It is
offered only on a `Target` that states a peer card a split executor reaches
(`Target.PeerCards`); no tier states one until that executor exists, so today
a caller's Take is an unmet choice by construction. Gates:
`graph.TestMLPPartitionIsARunTimeAlternative`,
`graph.TestEveryRewriteStatesItsDecisionTime`,
`model.TestRunTimeAlternativeIsNeverTakenUnasked`, and the Take and Refuse
arms of `model.TestNormGateMatchesTwoLaunches`.

### How the candidates plug in

| candidate | match | legality and contract | why it is a candidate and not a default |
|---|---|---|---|
| norm -> gate/up -> act -> down (the FFN site; `gate2in`) | a `norm` whose only consumers are two `matvec`s on its output, feeding one `act_mul`, feeding a `matvec` added into the residual | gate and up one format, one shape, one split (`gateUp2Shape`); the norm's input materialized by the previous add; exact on PTX | it removes a launch, and the outcome is per card and per shape (GK "The FFN's recipes, timed per card") |
| q/k/v operand prologue | `norm` -> the q/k/v `matvec` group | the norm's single consumer group; exact | already timed by `tuneActIn`; its sites become matched |
| residual + next norm's statistics in down's epilogue | an `add` into the residual whose consumer across the boundary is a `norm` | spans two blocks, so conditional on co-placement and dropped at a move; exact only with `RMSNormGroup`'s reduction tree | dep-block.md ranks it after the decomposition |
| hidden-channel MLP partition (`graph.MLPPartition`) | gate and up `matvec`s on one row feeding an `act_mul`, feeding a `matvec` | no bias inside a slice, no reader outside it; no contract (no executor built); a run-time alternative (`WhenRun`, "When a rewrite is decided"), its Bounds the cross-card cost: 3*H*F/2 weight elements off each card, two crossings a row | on one card its partials (8*H*ceil(F/256) bytes a row) outweigh the intermediate (8*F) at every row class wherever H > 256 (both sides scale with the rows, so a prompt chunk does not rescue it): on Llama-3.2-1B 512 KiB of partials a row against 64 KiB. Across two cards it is a capacity lever, not a speed lever: it halves a block's FFN weights on each card, and the speed policy never picks it; it is for a caller that asks (a placement objective of capacity), and it is not built yet. The crossings were measured and lose on speed. Split by channels over two V100s, each card reads half of gate, up and down (about 14 MiB less per card per Llama-3.2-1B block, about 18 us at the card's ~800 GB/s) and pays two crossings per block per token, the normed row out and the partial sum back. `backend.TestCopyPeerCrossingLatency` times that pair with `cuMemcpyPeer`: over NVLink (cuda:2<->cuda:3) the median is 56-63 us for 8, 16 and 64 KiB rows (p90 67-75 us, minimum 53 us); over PCIe (cuda:2<->cuda:6, direct) it is 98.6 ms. The crossing costs three times the read it saves on the best link: as a speed lever the split waits for a crossing that does not synchronise the host (a device-side flag or a peer write inside the kernel); as a capacity lever that cost is what its bound reports |
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

### The device program: the level between the block graph and the kernel IR

The host lowers most of a block's ops one kernel each. The device does not: a
block reaches the tier as its `nn.LayerPlan` and runs through one encoder
(`layersSession`'s `emit`, about 2,400 lines). That encoder holds a branch per
graph feature (AltUp, a recurrent mixer, DeepSeek V4, post-norms, a per-layer
embedding, ...) and every device fusion written inline. A fusion's contract
with its neighbours is implicit there, and that is the failure it produced:
`rmsF16For` wrote kernels.ActF16T's token-major operand into a memo the MMA
twin read as kernels.ActF16's m16n8 fragments, and Qwen2.5-VL-3B's tower read
NMSE 4.4 on sm_70 with every kernel gate green (PR #255).

The device program is the level that encoder skips. The graph says what a
block computes; the kernel IR (`jit/gpu/ir`) says how one kernel runs. Between
them sits a list of device ops, each one kernel, over typed values, that
rewrites fuse and the tier launches.

**Taken from MLIR and IREE.** Four ideas, adapted to one domain and a few
dozen op kinds:

- *A middle level of progressive lowering.* MLIR lowers through dialects one
  step at a time and fuses at the structured level (`linalg`), not at the
  kernel level. Here: graph -> device program -> kernel IR, and every fusion is
  a rewrite of the device program.
- *Layout in the type.* An MLIR value's type carries its memory layout (a
  memref's layout map, a tensor's encoding; Triton's layout encodings). Here a
  value is a `Type`: element type, layout and width. A rewrite that hands a
  consumer a value in a layout it does not read is illegal and refused when
  the program is built, not wrong output at run time.
- *Structured ops.* `linalg` states each op's iteration space (which
  dimensions are parallel, which reduce) and how each operand is indexed, and
  fusion and tiling are written once over that description. Here an op states
  its `Shape` (rows, heads, width, a reduced width) and its kind
  (elementwise, rowwise reduction, matvec, attention, gather/scatter), and a
  rewrite such as "fold an elementwise consumer into its producer's epilogue"
  is written once for every producer that offers an epilogue. IREE's
  dispatch-region formation (which ops share one kernel) is this problem
  exactly, and its heuristics are the prior art to read.
- *Legality and a dump after every pass.* MLIR's dialect conversion declares
  which ops are legal on a target and fails on any illegal op left over. Here a
  `Target` states the device ops it lowers; an op none of its lowerings takes
  refuses the block by name (the visible-fallback rule, enforced by the
  builder, not by a test). The program prints after every rewrite.

**Not taken.** MLIR itself (C++ inside LLVM: jitllm is Go with no cgo and
emits its own code), its general dialect machinery, a generic pass manager,
nested regions, and op traits beyond the few kinds above. One domain and a
fixed op list are plain Go structs and a list of rewrites.

**The data model.**

    type Type struct {
        Elem   Elem   // F32, F16, Int8 with its scale, Q8 blocks
        Layout Layout // RowMajor, TokenMajorF16 (ActF16T), FragM16N8 (ActF16), Q8Blocks, KVPaged
        Width  int    // elements a row
    }
    type DevOp struct {
        Kind    DevKind   // Elementwise, RowReduce, MatVec, Rope, KVWrite, Attend, Gather, Scatter, ...
        Src     graph.OpKind // the graph op it lowers (or the fused ops, for a rewrite's product)
        In, Out []ValID      // values, each with a Type
        Shape   Shape        // rows, heads, width, reduced width
        Epi     []Epilogue   // ops a rewrite folded into this one's output (bias, act, residual add, convert)
        Pro     []Prologue   // ops folded into its operand load (norm scale, dequant, convert)
    }

A rewrite matches a pattern over `DevOp` kinds and value types, checks
legality (types agree, the producer offers the epilogue, the Target lowers the
product), and replaces the ops with their fused form, recording its name. It
is the rewrite interface above (`Pays`, `Keys`, the When-it-is-decided
classes), applied to the device program instead of inside the encoder.
Buffers are assigned after rewriting, by liveness over the program, so a value
a fusion removed takes no buffer.

**Gates.**

- The device program with no rewrite applied against the encoder with its
  fusions off, same device, `JITLLM_GPU_TUNE=0`: token-identical and logit
  bit-equal, because the two launch the same kernels in the same order.
- Each rewrite against the unfused program on every device it pays on,
  within the gated band, past position 0, and run against a violation.
- A type mismatch is refused at build: the PR #255 fault restated as a
  program (a TokenMajorF16 producer feeding a FragM16N8 reader) must not build.
- The count of blocks still on the encoder is reported per model
  (`jitllm speed -manifest`) and gated to zero on the covered models.

**Order.**

1. D1. Dense decode (norm, matvec, rope, KV write, attend, add, act_mul) as a
   device program with no rewrites, beside the encoder, under the equality
   gate above.
2. D2. The encoder's dense fusions as rewrites: the add+norm fold, the
   binary16 operand writers (`rmsF16For`, `actF16For`), gate2in, the
   activation-in prologue, the residual epilogue.
3. D3. Rows: prompt chunks and ragged steps, the Volta and MMA GEMM forms
   chosen per value type.
4. D4. The mixture, latent attention, the recurrent mixers and the remaining
   graph features, each moving one encoder branch into the program.
5. D5. The encoder's block body deleted; the program is the only path.

**Primitives only: the program is a compiler, not a wrapper.** A program op
that calls one of the encoder's sequences (`mixFFN`, `emitLinear`, `emitMLA`,
the per-layer embedding) compiles nothing: it cannot be fused, reordered or
re-tiled, a new model that arranges the same mechanisms differently does not
build, and the encoder cannot be deleted while the program calls into it. The
coverage those ops reported counted wrapped blocks. The program is held to
four things instead:

1. **A primitive set**, which every graph op decomposes into, in
   `engine/graph`, each composition held to the host oracle: matvec over any
   weight format (per row, per pair of a selection, grouped); norm (RMS,
   Layer, per head); elementwise maps with any activation and any arity;
   reductions (sum, max, mean and variance); softmax and top-k; gather and
   scatter (routing, experts, embeddings, streams); attention over a paged
   history (dense, windowed, latent, selecting); scan (the recurrences);
   convolution (the towers, audio's and vision's). A vision tower, an audio
   encoder and a text block are compositions of the same set.
2. **Weights by role, end to end.** The graph's bindings name roles; the
   model hands the device a block's tensors by role, as the container
   names them; the tier uploads what the graph binds; a program op resolves
   its weight to an index at build. No per-mechanism field on either side.
3. **Runtime ops the lowering inserts, by property.** An expert matvec over
   a paged bank is preceded by a fetch (the selection home, the sheets
   filled, the selection rewritten to cache slots), whatever model it is:
   the paging principle as an op, not as a branch of one model's sequence.
4. **Buffers by liveness.** Values carry no slot until rewriting is done;
   slots are coloured over the straight-line program with RULE 13 as the
   constraint, so a new composition needs no new scratch name.

Fusions stay rules over shapes, as every compiler's are. The gate: the
coverage report counts a block only when its program holds primitives alone,
and a ratchet of the composite kinds still in the program only shrinks.
Order: weights by role; the mixture's primitives (route, top-k, per-pair
matvec, combine, the fetch); latent attention; the scans; streams; the
per-layer embedding; convolution; then D5.

**Convolution, as the program takes it.** A convolutional block (Gemma
3n's MobileNet-V5) and a FastConformer block (the audio tower) run today as
the tier's own launch lists (`conv.go`'s `convProgram` and
`conformerProgram`), one hand-built list per block kind: the per-model shape
this program replaces. They become graphs of the same primitive set, the few
it lacks added as ops:

- `OpConv`: a convolution over a row grid (1-D or 2-D, depthwise or full,
  kernel, stride, dilation and padding as attributes), lowered to the
  depthwise kernels or to im2col and the matvec;
- `OpAffine`: a per-channel scale and shift, which a BatchNorm becomes at
  conversion (precomputed, as every offline step is);
- GLU as an `act_mul` of a sigmoid, the macaron FFN's half step as a residual
  add's `Alpha`, and relative-position attention as the attention with a
  positional score term (its sinusoid table and the u and v biases bound by
  role) over the call's rows.

A block of either kind then lowers like a text block and its coverage is
counted the same way (`TestDeviceProgramCoverageBeyondText`, which reports
every merged tower's and encoder's blocks and today names the convolution
blocks as the gap).

**What to take from MLIR and IREE next.** Read against the sparse checkouts of
`llvm-project/mlir` and IREE's `DispatchCreation`, `Dialect/Flow` and
`Codegen/Common`, with the program as built (`jit/gpu/tier/devprog.go`,
`devrewrite.go`): twenty-odd rewrites, each a Go function that matches op
indices and decides legality itself, run in list order to a fixed point,
each application cloned, checked and undone on refusal. In the order to
adopt:

1. **A kind table instead of kind switches** (MLIR's op interfaces and
   traits). One row per `devKind`: elementwise, row-reducing, cheap to
   recompute, the epilogues and prologues it offers. `check()`, `String` and
   every rewrite's legality read it; a new kind is a row, not five switch
   arms.
2. **One producer/consumer legality** (IREE's `isFusableWithProducer`,
   `areAllFusionGroupUsesFusableWithProducer`, `wouldExceedOperandLimit`,
   `hasTransitiveDependencyOnFusionGroup` in `FormDispatchRegions.cpp`). A
   single `canFold(producer, consumer) error`: every use inside the group,
   the producer offers the consumer's class, types agree, the operand count
   stays under the launch's limit, nothing between them touches the slot.
   matvec-residual, o-in, down-in and add-norm re-expressed over it first, the
   program gate holding them to their old products.
3. **A worklist with benefits** (`GreedyPatternRewriteDriver.cpp`): patterns
   indexed by root kind, tried per op by benefit from the pay model, only
   changed ops re-queued, an iteration cap, a refusal reason per attempt for
   the dump. Removes the list order as an encoded priority (gate2in before
   gate2, add-norm-quant before add-norm), which nothing checks today.
4. **Slots assigned after rewriting** (bufferization's in-place analysis,
   reduced to interval colouring over a straight-line list): values carry no
   slot until the last pass, RULE 13 becomes the colouring's constraint
   (`Staged` the one permitted alias), and a rewrite no longer fails because
   two colours collided -- several of the refusals the gate has printed were
   exactly that.
5. **Multi-use producers** (`FuseMultiUseElementwiseProducer.cpp`): a cheap
   producer (a norm's scale, a convert) duplicated into each consumer's
   prologue, so the norm feeding q, k and v, or gate and up, is one rule
   instead of the per-pattern use counts in `rwQKV` and `normQuantBefore`.
6. **Iterator types as the shape** (linalg's `getIteratorTypesArray`,
   `FusionGroup::getRootParallelLoopToOpMap`): rows, heads, width and which
   dimension reduces, per kind; a fold is legal when the consumer's parallel
   dimensions map onto the root's. Not affine maps -- the program has one
   domain. Taken when rows-versus-ragged checks start repeating across
   rewrites.
7. **The tuner's choice as data** (the transform dialect, narrowly): a
   recipe set (`rwSet`) is already a schedule; a `Choice{name, alternatives}`
   the fused build takes as input lets the tuner pick and a `With...()`
   option force one, with the applied trace in the dump.

Not taken: PDL and declarative patterns (a pattern language for about thirty
kinds that Go closures over a typed table already express); tiling interfaces
and split-reduction (the kernel IR's tuner owns tiling, below this level); full
one-shot bufferization (no regions, no deallocation to plan); dispatch-to-
workgroup conversion (the kernel IR's job here).

### Decomposition: every composite op as primitives

The graph states several mechanisms whole: `route`, the recurrences, the
latent row and its attention, the indexer, the streams (`stream_mix`,
`stream_place`, AltUp's `alt_predict` and `alt_correct`), the n-gram block,
`laurel`, `gauss_topk`, the compressed attention, the draft head. A device
program that lowers such an op to a kind of its own, and a generator written
for that kind, is a family's code moved, not removed: it encodes one family's
arithmetic and layout, a model arranging the same pieces differently does
not build, and nothing can fuse or re-tile across it. That shape was built
and reverted (the router's variant fragments, AltUp's and the hyper-
connections' stream kinds, the paged attention transliterated into compose).

**Where decomposition lives.** `graph.Decompose` rewrites each composite op
into a subgraph of primitives, written once, in `engine/graph`, from the op's
attributes and bindings alone. The device builder lowers only primitives;
`TestCompositeKindsOnlyShrink` counts the program kinds that are not
(`primitiveKind`) and only falls. The host keeps its own routines until it
lowers the same subgraphs; a decomposition is held to the host's composite
routine, which is the oracle for it, and to the equality gate on the device.

**The primitives a decomposition may use**, each by structure, so an unseen
model whose graph uses them runs with no new code:

| primitive | what it is |
|---|---|
| `map` | an elementwise expression over one or two rows and bound vectors or scalars, with broadcast: add, sub, mul, div, max, min, select on a comparison, and every `Act`; the expression is the op's attributes |
| `reduce` | sum, max or min over a row, or over each group of `Width` in it (a row view), in tree or sequential order: the order is an attribute, since it is part of the arithmetic |
| `topk` | the k largest of a row (or of each group), ties to the lower index, as indices and values |
| `gather`, `scatter` | a row's elements (or rows) read or written at an index row: selection, experts, streams, embeddings |
| `matvec` | over any weight format, per row, per pair of a selection, grouped |
| `norm` | RMS, Layer, per head |
| `rope` | the rotary over a row's heads by a table, any pair layout and split |
| `attend` | over a paged history: dense, windowed, latent, over gathered keys |
| `fold` | a state carried across steps or positions and updated by a map: the recurrences' scan |
| `conv` | depthwise or full over a row grid, and `affine` |

A value gains an element type: `f32` rows as before, and `u32` index rows
(a selection's experts, a top-k's positions). A selection is two values, its
indices and its weights, which `gather`, `expert_matvec` and `combine` read.

**`fold`, as the recurrences need it.** Every recurrence the engine runs
(the gated delta rule, Kimi Delta Attention, Mamba-2's SSD, Mamba-1's
selective scan) is one core over a per-head matrix state `S[h][j][e]`,
carried per sequence: the state decayed (`S ⊙ g`, g a row's per head or per
key element), optionally a prediction `Σ_e S[j][e]·k[e]` subtracted from the
value, a rank-one update `S += d ⊗ k`, and a read-out `Σ_e S[j][e]·q[e]`,
optionally plus a skip. Its operands (the convolution over the window, the
activations, a norm of q and k, softplus and exp of the decay's terms) are
`map`, `reduce` and `conv` ahead of it, and its output norm and gate are the
same after it. So `fold` is a state-update former whose body is an
expression of the program's own, not a rule enumerated by flags: a carried
tensor state (a register-resident row of the matrix), a trip count read at
run time from the row's descriptor, reductions over the state's key
dimension that combine across lanes, a state-update node `S ← f(S, ...)` and
output nodes stated in the expression, and the state's input and output
bound through a slot descriptor (slot, stride) as a generic buffer. A
former keyed by rule flags (`Delta`, `Skip`, a decay kind) was built and is
not this.

**The order.** Route first (logits a float `matvec`; the bias and the
selection bias `map`; the score a `map`; groups a `topk` and `reduce` over a
row view, a `topk` of the groups and a `map` that masks the rest; the top k a
`topk`; the weights a `gather`, `reduce` and `map`), then the scans, the
streams, the latent row and attention, the indexer, the n-gram block, then
D5. Each lands with its decomposition held to the host routine on every
fixture that has it and the equality gate clean, and each removes its kinds
from the composite list.

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

### The mixture in the op language

A routed mixture is a sublayer graph of its own (`graph.Mixture`), built,
verified and matched at every Open for each mixture block the language states
(`engine/model/mixgraph.go`), its matches carried to the tier beside a gated
delta sublayer's in `nn.LayerPlan.Rewrites`. Four ops, each with fixed
semantics:

    sel   = route(x, router[, router.bias][, exp_probs_b])  experts, top-k, softmax|sigmoid|sparsemixer, norm, scale, groups
    pairs = gather(x, sel)                                  the row once per selected pair
    h     = expert_matvec(pairs, sel, bank[, bank.bias])    per pair, the sheet the selection names
    y     = combine(down, sel[, exp_scale])                 the weighted sum in the selection's order

with `act_mul` between gate and up as in a dense FFN, and a shared expert as
matvecs, `act_mul` and an optional `gate_scalar` (the sigmoid of a dot product
scaling its output) added after. The verifier holds the route to the
configuration's expert count, top-k, router, renormalisation, scale and
groups, every bank to the expert count, and every stored mixture tensor to an
op that binds it. A block the language does not yet state keeps no graph and
the tier decides from its own facts as before: DeepSeek V4's per-row
selection and hash blocks, Kimi-K3's latent mixture, Gemma 4's router norm
beside a dense FFN, experts stored one matrix each, ungated experts, and an
activation other than SiLU. `model.TestMixtureGraphsOfTheModelsVerify` holds
ten mixtures (qwen3moe, olmoe, DeepSeek-V2-Lite, glm4moe, qwen2moe, ernie4.5,
minimax-m2, bailingmoe2, dots1, phimoe) to one verified graph per mixture
block, and a graph of another top-k must not verify.

`graph.Target` gained `Residency`: the bank whole on the card, streamed
through an expert cache there, or on the host. Two existing levers became
rewrites found by the experts' edges (route, gather, gate, up, act_mul, down,
combine) and guarded by the row class and the residency:

| rewrite | guard (`Pays`) | legality | contract |
|---|---|---|---|
| `moe-few-row` (the dp4a grouped matvec where the staged f16 GEMM pads each expert to a 32-column block) | `RowsFew` (the scratch's routed pairs fewer than `VoltaMoEMinPairs`, by default the bank's experts), experts not on the host | the grouped mixture's shape | band 1e-2 on PTX and MSL (`TestBatchedMixturePrefillMatchesRowByRow`); a backend with no f16 expert GEMM has one form |
| `moe-waves` (a prompt chunk's experts on the card in waves through the expert cache) | `RowsChunk` (at least `prefillWaveRows`), `ResidencyCached` | no bank bias and no per-expert scale: a wave's slot ids index neither | band 3e-3 (`TestHybridPrefillWavesMatchTheHost`) |

The tier asks the waves once per block outside a token, at every row class and
residency (`prepMixture`), so the site picks an entry and allocates nothing
(the gate's warm run counts zero heap allocations); a matched region's
`Legal` is decided once at `Match`. Selection is asserted against a violation
in each gate: with `MoEFewRow`'s guard refusing every target the narrow class
puts the experts back on the f16 GEMM (`TestFewRowFormIsTheMixtureRewrite`),
and with `MoEWaves`' refusing, no block-chunk runs in waves.

### Latent attention in the op language

An MLA block (DeepSeek-V2/V3/R1, Kimi-K2, Moonlight, Glm4MoeLite, and
Kimi-Linear's NoPE attention blocks) is a dense block whose attention is
`graph.MLA` (`Dense.MLA`), chosen by what the container states (a KV latent
rank), never by the family's name. The graph states the reference form:

    q              = matvec(attn_in, attn_q) | matvec(norm(matvec(attn_in, attn_q_a)), attn_q_b)
    q_nope, q_pe   = slice(q, per head)                    HeadDim-NRot and NRot of each head
    kv_a           = matvec(attn_in, attn_kv_a)            KVLoraRank + NRot
    latent, k_pe   = norm(slice(kv_a)), slice(kv_a)        the latent's norm at its own epsilon
    latent_append(latent, rope(k_pe))                      kv@0 -> kv@1: one row per position, the value its latent prefix
    attn           = attend_latent(q_nope, rope(q_pe), attn_k_b, attn_v_b)   reads kv@1

`attend_latent` turns the cached latent into each head's key (through the
transpose of `attn_k_b`'s sheet) and value (`attn_v_b`'s) and attends. The
verifier holds the latent norms to `LatentNormEps`, the rotary parts to heads
of `NRot` (both turned, or neither on a NoPE block), the latent and rotary
widths to the configuration, and the up-projections to banks of a sheet per
head.

`mla-absorb` is the algebra every tier runs: `attn_k_b` folded into the query
and `attn_v_b` applied after the softmax, so the cache is read as one shared
key head and no per-head key or value exists. It is decided at compile time,
by the static guard, at every row class: it is the one form a tier lowers, and
a block whose absorb is not legal refuses the model (its reference form has no
lowering). Whether the reference form pays for a compute-bound prompt chunk,
where the decompressed keys are read by every row, is open and unlowered.

The router behind MLA is the mixture's: DeepSeek-V3's grouped sigmoid with
its selection bias and routed scale, an ungated shared expert, and a dense lead
of blocks with a dense FFN, all stated by the mixture ops above. Coverage is
held by `TestBlockGraphMatchesTheHandPath`, the prompt and ragged-step gates
(synth-deepseek, -lite, -dense, -yarn, -yarn-af, -glm4moelite, -kimilinear)
and the MLA fixtures in the five principles' lists
(`engine/model/graphmla_principles_test.go`). A ragged step of MLA States is
not taken by `Step` (its cached rows are read per batch slot,
`State.HostRefusal`); each runs alone.

Still uncovered behind MLA, by structure: an indexer (DeepSeek-V3.2's
lightning indexer, `UncoveredIndexer`), Kimi-K3's residual attention streams
(`UncoveredStreams`) and its MLA output gate (`RoleAttnGate`, an unbound
tensor).

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

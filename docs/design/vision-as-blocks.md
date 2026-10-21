# Vision as blocks: one runner, two segments

**Status: built, every family migrated** (idefics3, llava, Qwen2-VL,
Qwen2.5-VL, gemma3, InternVL, MiniCPM-V, Janus-Pro); Phi-4 multimodal is built
on this runner next. AGENTS.md ("THERE IS ONE MODEL") describes what runs; the
evidence is in `docs/engineering-history/model-correctness.md`, "Vision as
blocks". Where the build parts from the text below, and why:

- **The segment is a State, not a `Model.segs` table.** A vision State's `c`
  is the segment's `Config` and its `lo, hi` the vision blocks (section 4a's
  "segment State"); nothing else needed a table.
- **The host's non-causal attention stays its own branch of the block body**
  (`visAttend`, the tiled score kernel over the call's key runs), because it is
  bit-for-bit the old tower's and the causal per-row loop is not.
- **The device's tower K/V is one transient pair charged to the KV budget**
  (`transientkv.go`), not pages of `kvPool`: a block's keys live one block
  deep, so paging them buys nothing, and paged attention would have needed the
  non-causal key runs, the window mask and the query chunks all over again.
- **`g.scr` is the Gemma 4 geometry sets**: a non-causal segment is a set of
  its own (`geoNonCausal`), so no second scratch field and no union.
- **The projector runs on the host** (section 10), and `WithTowerGPULayers`
  stays as the bisection knob on the one offer path.
- **The preprocessors' resize is generated code** on the host tiers (AVX2,
  SSE, arm64); preprocessing on a device is not built.

The user's principle, which is the requirement and not a preference: *"for
vision we don't use the lame mtmd approach of llama.cpp, we handle vision like
text."* llama.cpp's mtmd/clip stays what RULE 7m makes it: an oracle and a graph
reference. Its SHAPE -- a second model behind a second loop, encoding into a
buffer that a text prefill later splices in -- is the thing being removed.

The container, the pager and the device block space were unified already
(AGENTS.md, "THERE IS ONE MODEL"). What is still mtmd-shaped is everything that
RUNS a tower. This document says what moves where, what is deleted, how each
tower family maps onto the result, how the KV prefix cache becomes able to see
past an image, and which gates hold it.

---

## 1. What is still a second model, as read off the code

Read on main at `083debaa` and on the two in-flight branches
(`d0183204` Qwen2.5-VL, `308fbd5b`/`1ea4b8b8` MiniCPM-V/Janus-Pro):

| duplicate | where | what it costs |
|---|---|---|
| a second runner | `TowerState.Encode`/`encodeOne` (engine/model/vision.go): its own block loop, its own LayerNorm/axpy/bias/attention calls, its own `parallelVec`/`rowChunk`/`worthPooling` | every block op exists twice, once in the prefill block body and once here; a fix to one is absent from the other (RULE 8a's failure, inside one tier) |
| a separate encode STAGE | `cmd/jitllm` `prefillImage`, `ui/engine.imageSpans`: encode, copy out, `Tower.Release`, `setPageBudget` again, then `PrefillMixed` | the image never goes through prefill/step; a server cannot schedule it; the caller does the budget arithmetic |
| a second (and third) kernel cache | `Tower.NewState` -> `nn.NewJITSharing`; the resampler's `rsState.jit` ("its own attention shape, 128-wide heads at stride d") | `JIT.AddAttn` holds exactly ONE (head-dim, kv-stride) pair in fields, so every new attention geometry has been a new JIT |
| a second offer path | `TowerState.SetDevice`/`offerBlocks`, with `base = NLayer` passed in by the caller | the tower's blocks never compete in `State.offerRange`'s placement, its placement map, the seam tuner, the decline counters or streaming |
| a second device scratch | tier `g.vbs`, and `layersCall`'s `if l.vision` branch that submits against it | the vision path skips `ensureKVCap`, paged KV, `prepBatch`, and every counter the text path keeps |
| a bespoke device KV | `jit_old/gpu/tier/visionkv.go`, one shared k/v pair charged outside the KV pool | a second allocator with its own refcount, invisible to the pool's budget and eviction |
| plan flags that multiply | `LayerPlan.Vision`, and on the Qwen2.5-VL branch `VisionRMS`, `VisionGated`, `VisionWindowed`, `nn.VisionWindowDevice` | "Vision is ONE flag because the five always co-occur" stopped being true at the second family; each new tower adds a flag the text plan already has under another name (`LayerNorm`, `UngatedFFN`, a mask) |
| a prefix cache that stops at the image | `kvCache.note` (engine/model/kvpage.go): `PrefillMixed`/`ForwardEmbd` advance the position with no id, `note` stops at the gap, `pageKey` refuses every boundary past it | nothing after the first image of a chat is ever prefix-cached: every later turn re-runs the tower and re-prefills from the image on |
| an embedding cache in the app | `ui/engine.Engine.embeds`, keyed by (path, size, mtime) | the cache lives in one caller; the CLI, the server and the KV pages cannot use it, and the key names a file rather than a picture |

The Qwen2.5-VL branch adds `PrefillCachedMixed`, which names an embedding row by
a 31-bit FNV of the row's float VALUES plus its position, and refuses a
bidirectional span. Section 6 replaces that key; the reasons are there.

### The evidence that the text runner already covers a tower

Every op a tower block runs already exists in the text block body, on the host
and on every device, under a text architecture's name:

| tower block op | the text kit that already runs it |
|---|---|
| LayerNorm with bias | C6 (`FlagLayerNorm`, phi2/falcon/starcoder): `layer.attnNormB/ffnNormB`, `batchNormB`, tier `lnBlock` |
| biased q/k/v/o | qwen2 and C6 biases (`layer.bq..bo`) |
| ungated MLP with biases | C6's ungated FFN (`layer.upB/downB`, `plan.UngatedFFN`, tier `ungatedFFN`) |
| gated SwiGLU MLP with RMSNorm (Qwen2.5-VL) | every llama-family block |
| bidirectional attention | `Span.Bidir` runs (gemma3's image rows, `bidir.go`), and a whole non-causal sequence through prefill (`embedstate.go`, `FlagNonCausal`) |
| windowed attention | sliding-window layers: a per-row key LOWER bound |
| 2-D / multi-axis rotary | M-RoPE's per-row tables (`mrope.go`, `nn.Rope.TableAt`), which the Qwen branch already reuses for the tower |
| no positional encoding in the block | `NoPosEnc` |

So a tower block IS a text block whose plan says LayerNorm-or-RMS, gated-or-not,
and whose rows arrive with a mask that is not causal. The embedding path through
prefill (`embedstate.go`) is the precedent for running a non-causal sequence
through the text runner: a whole input as one chunk because no row's scores
exist until every row's keys do.

---

## 2. The model: SEGMENTS of one block space

    Model.segs = [
      seg 0 "text"   blocks [0, NLayer)                    entry: vocabulary rows   head: norm + vocab projection
      seg 1 "vision" blocks [NLayer, NLayer + NVisLayer)   entry: patch rows        head: the projector
    ]

A **segment** is an execution domain: a contiguous range of the one block space,
its own immutable `Config` (NEmbd, NHead, NKVHead, HeadDim, NFFN, eps, act, norm
kind, rotary), an ENTRY (how rows come in) and a HEAD (what turns its last
residual into output rows). It is not an order of traversal: a request runs the
segments its rows need, and the vision segment's output rows are the text
segment's input rows.

- **Block ids stay global** -- placement, the pager, the device's `g.layers` and
  the container all key on them, and they already agree (`VisDataOff + i*VisPageSize`
  is block `NBlocks + i`). **Pattern decisions take the segment-LOCAL index**
  (`li - seg.lo`): Qwen2.5-VL's full-attention-every-n'th block, a text model's
  SWA period, a MoE cadence. (that is the bug a global
  index would plant: block 0 of the tower is block 28 of the space.)
- **`towerBlock` is deleted.** Its tensors are `layer`'s: `ln1W/B -> attnNorm/B`,
  `ln2W/B -> ffnNorm/B`, `wq..wo`, `bq..bo`, `fc1/fc2 -> up/down` (or
  `gate/up/down` for a gated tower), `bFc1/bFc2 -> upB/downB`. `Model.layers`
  grows to the whole block space; `Model.pageIn(li)` covers tower blocks, since
  the container already maps them.
- **`Tower` survives as the vision segment's ENTRY and HEAD**, not as a model:
  patch-embed weight and bias, class embedding, position table, pre-LN (entry);
  post-LN and the projector (head); the preprocessing config. `buildTower`
  becomes `buildVisionSegment`, and its block half moves into `Model.build`.
- **`planFor(li)` reads `segOf(li).Cfg`.** `LayerPlan.Vision`, `VisionRMS`,
  `VisionGated`, `VisionWindowed` are DELETED, replaced by the fields the text
  plan already has: `LayerNorm`, `UngatedFFN`, `NoPosEnc` or a per-row rotary
  (`NRot`), and the mask kinds of section 3. A device that cannot run one of
  them declines by that feature's name, as for text.

### The entry and the head

| | text | vision |
|---|---|---|
| entry | `embedRow` (vocabulary row, `EmbdScale`, learned position table) | patch gather (a permutation, setup code) + patch-embed matmul + bias, class row, position table or nothing, pre-LN |
| head | output norm + vocabulary projection (`nn.Head`, device or host) | post-LN + the projector program |

The projector is a short program of generated ops, one per family (section 5):
`Permute` (pixel shuffle / window order), `Pool` (gemma3), `Norm`, `Linear`,
`Act`, and for MiniCPM a `CrossAttn` that calls the same attention set with its
own shape. It runs on the host on the run's JIT in the first build; the device
form is the vision segment's `Head` on the last tower submission and is
declined BY NAME until built (RULE 8a), the way `nn.Head` was before the text
head moved onto the card.

---

## 3. Masks: one per-row key interval

Every mask this engine runs is, per row, a contiguous interval of keys in a
defined logical KV order:

| mask | row r at position p sees keys |
|---|---|
| causal | `[0, p+1)` |
| sliding window W | `[p-W+1, p+1)` |
| gemma3 image run inside a causal prompt (bidir run `[a, b)`) | `[swaLo(p), b)` for rows in the run |
| a tower block, full attention | `[0, n)` -- the image's rows; per PIECE when the picture is several pieces (a tile or a slice never attends across) |
| Qwen2.5-VL windowed block | `[w.lo, w.hi)` of the row's window, in window order |

So `nn.BidirDevice.SetBidirRuns` and `nn.VisionWindowDevice.VisionWindows` merge
into ONE call that hands a submission its key runs:

    type KeyRun struct{ Lo, Hi int; Floor bool }   // Floor: nothing before Lo is visible
    SetKeyRuns(full, windowed []KeyRun) bool

`full` is the runs every non-windowed block uses (gemma3's image runs; a tower's
pieces, each with `Floor`), `windowed` the runs a windowed block uses (the plan's
pattern picks, by segment-local index). Causal is the empty set. The intervals
address the run's LOGICAL order: Qwen2.5-VL's patches are permuted into window
order at the entry (the reference does the same, `window_index`), the rotary
rows follow the permutation, and the head's `Permute` restores raster order.
Sinks and softcaps stay score modifiers beside the interval, not part of it.

**Rows of one run are never cut by a chunk, and the existing rules say so
already**: `bidirChunk` moves a prefill chunk so no run straddles it, and the
tier refuses a call that cuts a run. A tower segment is therefore one chunk of
its whole row count -- `batchWidthFor`'s "a vision plan gets its patch count"
becomes "a run with a full key run gets its run's width", which is not a vision
rule. Query-row chunking INSIDE a block (`visionattn.go`: complete K/V, queries
in pieces to bound the score planes) is valid and stays, keyed off the score
planes of any long run rather than off `p.Vision`.

---

## 4. One runner, one cache, one placement, one scheduler

### 4a. The runner

`prefillSrc` becomes segment-generic: `runSegment(seg, rows, mask, rope, head)`.
The body is today's prefill block loop reading `seg.Cfg` instead of `m.Cfg`.
`TowerState`, `Encode`, `encodeOne`, `EncodeGrid` and the resampler's state are
deleted; an image is a sequence of patch rows through the same entry points as
token rows:

- `State.PrefillMixed(spans...)` takes `Span{Image: img}` (a preprocessed
  picture, section 6). For each image span: a projector-cache hit supplies the
  rows; a miss runs the vision segment through `runSegment` (entry, blocks,
  head) and then the rows enter the text segment exactly as `Span.Embd` rows do
  now. One call, no stage, no `TowerState`.
- The tower segment's sequence is its own: positions `0..Seq-1` of a transient
  sequence of the State, not text positions. Its rotary rows are the per-row
  multi-axis tables the M-RoPE branch builds (`nn.Rope.TableAt`); a table tower
  adds its position rows at the entry.
- **How the runner is reached, concretely: a segment State.** The embedder
  (`embedstate.go`) already runs a non-causal sequence through `prefillSrc`
  and collects the residual instead of logits. A vision run is the same thing
  over the vision block range: a State whose `lo, hi` are the segment's blocks
  and whose config is the segment's, sharing the parent State's JIT, pool and
  device session, entered with patch rows and finished by the projector head.
  The runner reads `s.cfg()` (the segment's `Config`) where it reads
  `s.m.Cfg` today -- 123 sites, mechanical -- and a `Config` carries its
  segment's base block, so `LayerKind(li)`, `SWA(li)`, `AttnWindow(li, pos)`
  index segment-locally by construction rather than at each call.
- **Scratch**: the State's batch scratch is per (segment geometry) and grown on
  demand, so a warm encode allocates nothing (gate e). The tower's K/V is not
  heap scratch at all (4b).

### 4b. The tower's K/V goes through the KV machinery

The user's goal (4): the tower's attention uses the same paged KV and budget as
text. A tower block's keys live only inside that block: written, read by that
block's attention, dead before the next block. In pager terms that is a
transient sequence whose history is ONE block deep:

- **Host**: the vision segment gets a `kvCache` range over its blocks
  (`newKVCacheRange` exists) whose pages come from the same allocator and are
  charged to the same KV budget, and are recycled (`kvPages.recycle`) to the
  next block when a block finishes. No history hashing, no persistence, no
  append semantics -- those belong to text history.
- **Device**: the transient sequence takes pages from `kvPool` for the image's
  rows and returns them after each block, so the next block takes the same
  pages. `visionkv.go`'s shared pair is that recycling written by hand; it is
  deleted, and the pool's own counters (pages in use after an encode = the
  baseline) replace its refcount. Paged attention already honours bidirectional
  runs (`len(g.bidir) > 0` on the paged path); `pagedPlan`'s `!p.Vision`
  exclusion goes.
- If page-table overhead measures material for a 4096-row image, a contiguous
  extent inside the same allocator is the fast path -- an allocation strategy,
  not a second tower path.

### 4c. One kernel cache

`JIT.AddAttn`'s single pair becomes a set keyed by every baked parameter:

    type AttnShape struct{ HdK, HdV, KVStride int; F16 bool; QStride, ScoreStride, Qt int }
    func (f *JIT) Attn(s AttnShape) *AttnSet   // emitted once, immutable, cached

A run resolves its sets once per ATTENTION SHAPE, outside the hot loop (a text
segment has one, a tower one more, MiniCPM's resampler a third), and the loop
holds the pointer: no map lookup per call, which is the objection the current
comment raises against keying. `AddAttnTiled` folds into the same key (`Qt`).
`nn.NewJITSharing` is deleted -- there is one JIT per State and it serves every
segment; `NewJIT` is sized from the union of the segments' `maxK`, `maxRows`
and quant types. Mutable workspace stays outside the cached set.

### 4d. One placement

`State.offerRange` walks the whole block space; `planFor(li)` builds the plan
from the block's segment. Consequences, all of which are deletions:

- tower blocks compete for the device budget, honour `WithPlacement` (named by
  global index or `v<i>`), count in `noteDecline`, page and relocate like any
  block; `TowerState.SetDevice`, `offerBlocks`, `base`, `WithTowerGPULayers`
  and the caller's `ts.SetDevice(dev, m.Cfg.NLayer)` go;
- the default ORDER stays block index -- text first -- because a text block runs
  every token and a tower block once per image; a tower whose blocks are left
  on the host may STREAM through the device for an encode (`prefillstream.go`
  already streams host blocks through the card for a long prompt, and an encode
  is a 1024-4096-row prefill);
- `Tower.Release` is deleted: the pager's eviction sees a finished segment's
  blocks like any other, and `setPageBudget`'s second call in `prefillImage` --
  the re-budget after the encode -- goes with it.

On the tier, `l.vision` and `layersCall`'s vision branch go, and **`g.vbs` is
deleted as a special field, not merged by union**. The hybrid precedent ("one
scratch built from the union") applies when two block kinds share one residual
inside one submission; a tower and its text model never share a submission and
differ in residual width (SmolVLM 768 vs 576, SigLIP 1152 vs gemma3 2560). So
the scratch map generalises from width to shape:

    g.bbs map[width]*blockScratch  ->  g.scr map[scrKey]*blockScratch
    scrKey{ shape: residual, heads, head dims, FFN width, KV representation,
            baked op variants;  width int }

The key is executable and buffer compatibility, not segment identity:
a hybrid's two plans land on one key with the union kernel set, which is where
the union rule lives now; text and tower land on two. A scratch is a definition;
a submission holds it as a lease, so two submissions in one step never share
mutable storage.

### 4e. One scheduler

An image encode is a run:

    type Run struct {
        State  *State
        Tokens []int32
        Image  *Image   // a picture to encode, when the run is (also) an encode
        Logits bool
    }

- `StepRuns` carries the encode beside other sessions' decode in the SAME step.
  They are different weights, so they are separate submissions in one step:
  the step's text rows go as today (`LayersSessions`), the encode as the vision
  segment's submission; neither waits for a stage the other owns, and each
  answer equals the run alone (gate b). The `Scheduler` admits an image the
  same way it admits a prompt chunk.
- **Resumable at block boundaries.** An encode may stop after any block and
  resume in a later step: the State holds `enc{img, residual, next block}`, with
  explicit completion and cancellation, and the pages it holds (4b) are the
  transient sequence's. That bounds the decode rows' inter-token wait to the
  longest non-preemptible piece (one block, the entry or the head), which is a
  soft bound and is stated as one. The POLICY -- how many blocks a step spends
  on an encode -- defaults to the whole tower in one step and becomes a budget
  only when a latency measurement asks for it (vLLM, SGLang and
  mistral.rs all encode whole). An encode is never split by ROWS through the
  tower: bidirectional rows cannot be, though query chunks inside one block can.
- The server's step loop gains image input once the API carries pictures
  (a proto change, its own step); the model-level run is what gate (b) holds.

---

## 5. Every tower on the runner

| family | entry | block plan (vision segment) | mask | head (projector program) | text side |
|---|---|---|---|---|---|
| idefics3 (SmolVLM) | patch matmul, position table | LayerNorm+bias, ungated GELU-tanh, biased qkvo | full | post-LN, `Permute` (shuffle s=4), `Linear` | causal rows |
| mlp (llava-phi-3) | patch matmul, class row first, position table, pre-LN | LayerNorm, ungated quick-GELU | full (CLS + patches) | drop CLS, `Linear`, GELU, `Linear` | causal |
| janus (Janus-Pro) | patch matmul, position table, letterbox (setup) | SigLIP: LayerNorm, ungated GELU | full | llava's aligner (`mm.1` named second matrix) | causal |
| qwen2vl_merger | patch matmul (two planes folded at conversion) | LayerNorm, ungated quick-GELU, 2-D rotary per row | full, per picture at its own grid | LN, `Permute` (2x2 group), `Linear`, GELU, `Linear` | M-RoPE rows at (t, h, w) |
| qwen2.5vl_merger | patch matmul, rows in WINDOW order | RMSNorm, gated SiLU with biases, 2-D rotary per row | windowed, full every n'th (segment-local) | RMSNorm, group, MLP, `Permute` back to raster | M-RoPE |
| gemma3 | patch matmul, position table | SigLIP: LayerNorm, ungated GELU | full | post-LN, `Pool` (avg s x s), RMSNorm, `Linear` | image rows a BIDIR run in the text segment |
| internvl | patch matmul, class row first, position table | LayerNorm, ungated GELU, layer-scale where present | full per TILE | drop CLS, `Permute` (shuffle), LN, MLP | causal, tiles laid out by the processor |
| minicpm resampler | patch matmul, bucketed position table at any grid | SigLIP | full per SLICE | `Linear` kv + LN, `CrossAttn` (64 learned queries, 128-wide heads, 2-D sin/cos keys), LN, `Linear` | causal, overview + slices |
| phi-4-reasoning-vision | patch matmul at the picture's OWN grid (up to 3600 patches); the 16x16 position table resized per grid (bilinear antialias, factored out of `imageproc.go`'s `aaWeights`, cached per grid) | SigLIP2: LayerNorm, ungated GELU; the last block dropped at conversion (layer -2) | full | llava's mm.0, GELU, mm.2 over every patch | phi3, causal; no markers |

**Phi-4-reasoning-vision is the first family built on this runner**, from the
old-path work on `wip-phi4-vision` (exact pixels over 14 sizes, the tower
1.7e-3 against transformers, 12/12 teacher-forced) and
`docs/design/model-coverage.md` J1. What it asks of the runner is already in
the design: an entry that takes ANY grid (a position table resized per grid is
entry setup, cached by grid in the segment, like a rope table), a segment run
of a grid's rows rather than a square's (the run width is the run's rows, 3.2),
a head that is llava's program, and the same picture identity (the resize
bounds join the digest's config). Its old-path gates become the runner's gates
for it unchanged.

**The survey's queued families, and what each needs from the runner**
(model-coverage.md J1):

- **Kimi-VL** and the 2-D-rotary group (youtuvl, step3vl, dotsocr, ...): the
  per-row multi-axis rotary rows of the vision segment, in whichever pair
  layout the family trains (interleaved for Kimi-VL) -- a property of the
  segment's rotary, not a new path.
- **GLM-4.1V/4.5V**: text M-RoPE (the rows' positions in the key, 6b) and a
  tower of qwen2vl's shape with fused qkv (split at conversion, as phi3's is).
- **MiniCPM-V 4.6**: a 2x2 merger INSIDE the tower after block 6, so the row
  count halves mid-stack. On this runner that is TWO vision segments: blocks
  0-6 with a head that merges, and blocks 7.. whose entry takes those rows --
  the same segment boundary text and vision already have, and why the head and
  the entry are programs rather than fixed shapes.
- **granite4_vision / qwen3vl deepstack**: intermediate-layer taps injected
  into text layers -- a segment whose head reads several block outputs, and
  text rows that take an addend at a given block. Named here so the head
  interface does not close the door; not built.
- **GLM-Edge-V, CogVLM**: BOI/EOI rows that are not text tokens -- rows of the
  vision segment's head output, keyed like the image's other rows.

Every row of that table is a configuration of the same pieces. What each family
brings that is genuinely its own is preprocessing (setup code per picture, held
to the reference processor to the bit) and its head program. The resampler's
cross-attention is the one op whose K/V source is not its own rows; it is a
`CrossAttn` step of the head program calling an `AttnSet` of its own shape, not
a block of the segment and not a third JIT.

---

## 6. Image identity, and the KV cache made first-class

The user's second goal: B must make the KV cache better. The gap is concrete --
`note` stops at the first row without an id -- and the fix is that every row has
one.

### 6a. The image digest

    digest = sha256( "jitllm/image/1"
                   ‖ the family's preprocessing config, canonically encoded:
                     projector kind, image/patch size, scale, mean, std, resize
                     algorithm and its version, tile/slice/letterbox parameters,
                     min/max pixels
                   ‖ the layout: piece count, each piece's grid (T, H, W), order,
                     and the window permutation where there is one
                   ‖ every piece's preprocessed float32 pixels )

Computed after preprocessing (setup code, milliseconds) and BEFORE any block
runs. It is a property of the input, which is the whole point:

- it is known before the tower runs, so a KV hit past the image skips the tower
  AND the projector cache;
- it is deterministic across processes and tiers. A key over the projector's
  OUTPUT VALUES (the Qwen branch's FNV) is neither: values are only known after
  the tower, and on a device `tuneSplit` picks a reduction order per process, so
  the same picture encodes to different bits in two processes and the cache
  misses across every restart;
- it covers the config and the layout, not only the pixels: two layouts whose
  pixel buffers are byte-identical (a grid and its transpose, the same tiles in
  another order) are different pictures to the model.

The namespace (container identity, as `SetCacheKey` already takes) scopes both
caches to these weights; vLLM keys the same way (model id + media + processor
kwargs), and mistral.rs's key without processor kwargs is the aliasing to avoid.

### 6b. Every row has an id, and the page hash sees the whole of it

`kvCache.seq []int32` becomes a stream of ROW KEYS:

    token row   the vocabulary id
    image row   (the image's index in kc.images, the row index)
                kc.images[i] holds the full 32-byte digest

`pageKey` serialises the stream into the hash: a token as its id; an image row
as a tag no id can take, the FULL digest and the row index. Nothing is truncated
into the key, so an image row cannot equal a vocabulary id and two pictures
cannot collide short of sha256 (keep any short id a handle, never the
authority; SGLang's 30-bit pad ids are the aliasing risk the survey found).
`note()` takes row keys, so a mixed prompt leaves no gap, and `ForwardEmbd` with
caller-supplied rows (no identity) is the only path that still stops the
stream -- by design, and named.

**Positions are in the key where they are not the row index.** Where the
model's rotary is multi-axis (M-RoPE), the hash also takes every row's
(t, h, w), and the rows after an image their resumed position. For every family
here those positions are a pure function of the row stream (the grid is in the
digest, text resumes at `max(H, W)`), so the triple is a GUARD: it is what keeps
a future scheme -- a temporal scale, a position offset, video -- from aliasing,
and vLLM's own correctness rests on that coupling without saying so. The gate
therefore drives a position the stream does not determine (a test hook shifting
an image's rotary start, as `stepPos` does for steps) and demands a miss.

**A bidirectional run's rows depend on the run's future rows**: a page
ending inside a run holds K/V computed with the run's tail visible. The digest
covers every row of the run, so the key of a boundary inside the run already
names its whole content. The restore still SNAPS DOWN to the run's start, as
mistral.rs's `clamp_prefix_cache_hit_len` does, because the tier refuses a call
whose rows cut a run; `PrefillCachedMixed`'s refusal of bidirectional spans goes.

### 6c. The projector cache

The vision segment's output rows, keyed by `(namespace, tower geometry, digest)`:

- an in-memory LRU on the Model, budgeted in BYTES and shared by every State of
  the model (cross-request, as vLLM's and mistral.rs's are), entries pinned
  while a run holds them (SGLang has no pinning and can evict under a live
  request);
- persisted through the same `KVStore` under its own record kind (a reserved
  layer index past `logitsLayer`), so `-kv-cache` keeps pictures across
  processes;
- a hit runs ZERO tower blocks. The app's `(path, size, mtime)` map is deleted;
  a memo from file identity to digest may stay in a caller to skip re-decoding,
  and it never names rows.

Numerical contract: a cached row from a device encode reused by a host run is
the same tolerance contract text pages already have across tiers (the page key
carries geometry and element width, not the tier). The identical-token gates
compare cached against uncached on the SAME placement.

### 6d. A hybrid with an image

A hybrid seals its recurrent summary at the prompt's exact end and restores
only there. With row keys, an image chat on a hybrid restores at that one point
like text. No implemented VLM on this box is a hybrid (Qwen3.5 is natively
multimodal, but its tower is the qwen3vl projector, which is not built), so the
gate pairs `synth-qwen35-hybrid-mtp` with a synthetic tower converted at its
width -- `jitllm convert` accepts any mmproj whose `ProjDim` is the text
`NEmbd` -- and Qwen3.5's real tower joins when qwen3vl lands (RULE 11: a task).

---

## 7. What is deleted

`TowerState` and every method on it (`Encode`, `encodeOne`, `EncodeGrid`,
`SetDevice`, `offerBlocks`, `GPUBlocks`, `parallelVec`, `rowChunk`,
`worthPooling`, `act`, `axpy`, `scale`, `addBiasRows`, `layernorm`, `mm`,
`ropeRows`, `ropeTable`, `Regions`, `Close`), `Tower.NewState`, `Tower.pageIn`,
`Tower.Release`, `towerBlock`, `rsState` and its JIT, `nn.NewJITSharing` and the
borrowed-pool flag, the single-pair attention fields on `nn.JIT`,
`LayerPlan.Vision` / `VisionRMS` / `VisionGated` / `VisionWindowed`,
`nn.VisionWindowDevice` and `nn.BidirDevice` (into `SetKeyRuns`), `g.vbs` and
`layer.vision` and `layersCall`'s vision branch, `visionkv.go`, `pagedPlan`'s
vision exclusion, `WithTowerGPULayers`, the re-budget dance in `prefillImage`,
the app's embedding map, `PrefillCachedMixed`'s value key. `Preprocess` moves
from `TowerState` to the Model (setup code needs a JIT only for the bilinear
families' resize, and takes the State's).

What stays and why: preprocessing per family (per picture, not per token; held
to the reference processor); `visionattn.go`'s query chunks (generalised to any
long run); the tracer hooks (`afterBlock`, `atStage`), moved onto the runner so
the node-by-node oracle gates keep their seams.

---

## 8. Gates

Every gate runs against a violation first (RULE 10) and on host, CUDA, Vulkan
and Metal (CUDA/Vulkan bulk on the V100 box; Metal through a cross-compiled test binary run on an arm64 Mac).

**Kept, bit-for-bit or within their current bound**: `TestTowerOnDeviceMatchesHost`,
the gemma3 / internvl / qwen2vl / qwen2.5vl / minicpm / janus gates (transformers
and llama.cpp node-by-node), `TestChatSpansDescribeAnImage`, the
`llama-mtmd-cli` teacher-forced checks, `TestPreprocessMatchesTransformers`,
`TestProjectorMatchesTransformers`. Host tower outputs must be BIT-identical
after the move to the runner (same kernels, same order); device outputs within
the existing bounds.

**New:**

| | gate | violation it must fail on |
|---|---|---|
| a | structural: no second runner/cache/offer. Counters: every tower block goes through `offerRange` (offers, declines) and the State's one JIT (`AttnSets` resolved, JIT count = 1 per State); a test walks the package for `NewJITSharing`/a second offer entry and fails | a tower offered through a private path; a second JIT built for a tower shape |
| b | an image encode run in one `StepRuns` beside two sessions' decode equals each alone (logits bit-equal on the same placement) | the encode's rows written into a decode session's scratch lease |
| c | image prefix cache: turn 2 with the same picture runs ZERO tower blocks (counter) and gives identical tokens; a different picture misses | the projector cache keyed without the digest's pixels |
| c' | KV past an image: a two-turn image chat reuses every position up to turn 2's new tokens (count reused positions; tokens identical to an uncached run) | the gap kept in `note` (reuse stops at the image) |
| c'' | a different image misses; the same picture under a different layout or config (byte-identical pixels, transposed grid) misses | the hash ignoring the preprocessing config/layout |
| c''' | the same image at a different rotary position does not alias (hook-shifted start) | positions dropped from the key |
| c'''' | a hybrid with an image restores at its one exact point, as text does | the image rows' ids dropped (the hybrid restores nothing past the image) |
| d | tower blocks page under a budget below the tower, and relocate host<->device between blocks of one encode (resumed across steps), output identical | a block run against another block's frame (no `pageIn` on the tower range) |
| e | `TestDecodeDoesNotAllocate` stays zero; a warm image-encode step allocates nothing | per-encode scratch allocation |
| f | all of the above on host, CUDA, Vulkan, Metal (M4: a cross-compiled `engine/model` test binary run with `-test.run '<pattern>'`, `JITLLM_MODELS` set on that host) | -- |

Plus RULE 1 for the move itself: SmolVLM, gemma3 and Qwen2.5-VL encode time,
runner against the old loop, ABBA in one process, A/A first, host and one
device -- the restructure must not regress the encode, and a number is quoted
only as that ratio.

---

## 9. Order of work

1. **Keyed attention sets, one JIT.** `JIT.Attn(AttnShape)`; the tower and the
   resampler take theirs from the State's JIT; `NewJITSharing` goes. Gate:
   every tower output bit-identical.
2. **Tower blocks into `Model.layers`, segments, the runner.** `towerBlock` ->
   `layer`; `planFor` and `prefillSrc` read the segment; `Span{Image}`; the
   encode is `runSegment`. `TowerState` shrinks to a shim over the runner for
   the migration and is deleted at the end of the step. Gate: bit-identical
   host outputs, every family.
3. **Masks and the device.** `SetKeyRuns`; `LayerPlan.Vision*` into the text
   fields; `g.scr` keyed by shape; one offer path; `visionkv.go` into the pool.
   Gates a, d, the device bounds.
4. **Identity and caches.** The digest, row keys in `kvCache`, positions in the
   key, the projector cache, the restore snap. Gates c..c''''.
5. **Scheduling.** `Run.Image`, resumable encode, `Scheduler`. Gates b, e.
6. **Callers and rules.** `cmd/jitllm`, the app; AGENTS.md's "THERE IS ONE
   MODEL" edited in place (the kernel-cache paragraph that justifies a second
   cache, the `Vision` one-flag paragraph, the `g.vbs` bug list); evidence in
   `docs/engineering-history/model-correctness.md` and `placement.md`.

---

## 10. Open, named rather than implied

- **The projector on the device** is a declined head in the first build; the
  tower's blocks run there, the projector on the host, as today.
- **A tower's FIRST block on the device and the entry on the host** cost one
  crossing of the patch rows; an `EmbedDevice`-style entry on the card is later
  work, priced by the encode measurement above.
- **Encode budget policy**: unbuilt until a measurement of decode inter-token
  latency under an image says it is needed.
- **Server image input**: an API change, after the model-level run lands.

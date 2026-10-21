# Device KV paging

**Status: the device KV is paged** (jit_old/gpu/tier/pagedattn.go): decode,
prefill chunks and LayersRows batches read and write pages on CUDA, Vulkan and
Metal. A prefill chunk runs the paged prefill kernels
(jit_old/gpu/tier/pagedprefill.go); a ragged batch step reads through paged
FlashDecodeKV. MLA's latent cache is paged too: one row-major latent row a
position, float32, the value its first KVLoraRank floats; decode runs the
staged kernels at that value width, a prompt the FMA or m8n8k4 prefill pair
(neither flash form nor the m16n8k16 scores reads a latent row). A history
past the pool sends its oldest pages home and streams them back for each
token's attention (jit_old/gpu/tier/kvevict.go, pagedstream.go; "Errors and
eviction" below). The contiguous cache it replaced is gone, with
the capacity machinery that managed it (per-session capacities, parked
scratches, regrowth); only a vision tower keeps a contiguous buffer, for
keys that live for one encode and are not history.

Before it, each block kept one contiguous buffer per session sized to a
capacity the attention kernels baked in (K transposed at stride `cap+1`, a
score row `cap` wide), so growth was reallocate-copy-rebuild, a batch reserved
every row's whole context up front, and capacity had to be managed per session.
Everything else in jitllm is paged -- weights on the host and the device,
expert pages, the host KV cache (engine/model/kvpage.go) -- and the device KV
was the one place it was missing.

The principles, which are the requirement:

- A sequence's history is PAGES, allocated as it grows and freed when it ends.
- Nothing is baked to a context length: no kernel, no scratch.
- Running out of pages is not a crash and not a demotion by default: the page
  goes to the host store (eviction), or the call returns an error. The model
  is not down.

## What is generated and what is data

The kernels are generated for the device in front of them, and that is the
dividing line: what is stable for a model on a device is BAKED and
REGENERATED when it changes; what changes every token is DATA.

    baked, regenerated on change        data, never baked
    ----------------------------        -----------------
    KV page size P                      which pages a sequence has (the table)
    key tile width (the subgroup)       each row's length and query position
    split count and the merge shape     which rows are in this batch
    head geometry, GQA, f16 V,          the window: keyStart in the descriptor
      sinks, softcap

The capacity bug this replaces was a per-token quantity -- the context length
-- baked into every attention kernel. Nothing here may repeat it.

- **P is a runtime choice, per pool.** The KV cache is not in the model file
  -- every inference rebuilds it -- so unlike the weight page sizes (PageSize,
  VisPageSize, ExpPageSize: the file's layout, fixed at conversion) nothing ties
  P to the container. Each device's pool picks its P once, when it is created,
  for that device and the model's geometry; the host KV has its own. An
  inference keeps the P it started with to the end.
- **Moving history is a load, at the destination's P.** When the model moves --
  a seam, a spill, a relocation, a migration home -- the history is copied
  position by position into the destination's pages at the destination's size,
  which is what a migration already does; a new P takes effect the same way,
  at the next load into a new pool. Nothing live is re-paged in place. A
  sequence split across devices has one table per device, each at that device's
  P. P is a multiple of 64 so that every device's key tile divides it.
- **The key tile is the device's subgroup width** (32 on NVIDIA and Metal, 8 to
  32 on Intel under Vulkan, 64 on AMD), not a constant 32. It divides P, so a
  tile reads its page id once.
- **The split count follows the keys**: one compiled variant per split,
  selected as a sequence grows past each threshold (tier/flash.go already keeps
  variants and picks by length), and the merge is generated for the split it
  combines.

## The page

A page is `P` positions of ONE layer's history for ONE sequence. P is chosen
when a device's pool is made: `Config.KVPage`, or the smallest power of two
from 64 that covers the context, up to 256 -- a page longer than the context
holds history no sequence can reach, and at 256 positions on a 64-position test
context every layer carried four times the history the contiguous cache did.

    kvRow = NKVHead * HeadDim              (MLA: KVLoraRank + NRot)

    K region   [kvRow][P]   f32, transposed: element e of position t at e*P + t
    V region   [P][kvRow]   f32, or binary16 packed two to a word (KVF16)
    MLA        [P][kvRow]   f32 row-major, one region; V aliases its prefix

The stride inside a page is always `P`, whatever the context. Position `t` of
a sequence is page `t/P`, offset `t%P`.

## The pool

Per device, per attention layer: one K buffer and one V buffer (one for MLA)
holding that layer's `N` pages, its own free list, its own ownership, and a
table arena. Page id `p` is a physical slot in THAT layer: `p*P*kvRow` in its K
buffer and `p*P*kvRow*vBytes/4` (in words) in its V buffer.

**Ids are the layer's, not the device's.** The first build gave a sequence's
page `j` one id in every layer (vLLM's block table), which made every layer's
buffer the same `N` pages -- and then a session yielding a block another
session keeps freed nothing: its pages still sat in every layer. Measured, a
batch session B beside a single session A on a tight card could only leave the
card whole (0 of 16 blocks) where the contiguous cache let it split. With
per-layer ids a session leaving layer `l` frees exactly layer `l`'s pages, and
the same test keeps 6 of 16 of B's blocks on the card with A untouched
(`model.TestABatchLeavesAnotherSessionAlone/tight`). A session takes pages only
in the layers it runs on the device.

**One descriptor set still serves every layer**, because a sequence's run sits
at the SAME offset in every layer's table arena; only the table buffer bound per
layer differs, and the ids in it are the layer's own. The kernel ABI is
unchanged. The tables are persistent: an ordinary token uploads only the rows'
descriptors (16 bytes a row), a page boundary writes the new id into each held
layer's arena, and the arenas grow (rewritten from the host copies, the graph
dropped) only when a run no longer fits.

The number of pages is elastic at run time, per layer, and follows the
device's budget the way weight slots do:

- **Grow** by adding pages as sequences need them: doubling when that fits,
  else exactly what is asked -- a doubling is an amortisation, not a
  requirement -- through a device copy into a new buffer pair, charged as it
  grows. Nothing changes unless all of it succeeds.
- **Released pages stay in the pool.** A session leaving a layer, a detach and
  a trim put their pages on the layer's free list and announce room (RoomGen):
  the pages one request gives back are the next one's. Shrinking on every
  release made every prompt grow each layer again -- an allocation and a copy
  per layer per request, a third of a V100 prefill
  (docs/engineering-history/gpu-kernels.md, "Prefill on pages").
- **Shrink by compacting, on demand** (`kvCompact`): when a placement, a
  page-in, another layer's growth or a recurrent state needs room, layers are
  compacted one at a time until it fits. Freed pages sit wherever their ids
  were, so a free tail is not enough: the live pages are gathered into a fresh,
  smaller buffer pair (dummy first, then each sequence's pages), the ids
  renumbered and the sequences' runs rewritten. The gather's peak is both
  buffer pairs; where the card cannot hold that -- which is when compaction is
  asked for -- the live pages go through the host and the old pair is freed
  before the new one is made. Growth takes the same host path when only the
  grown layer fits.
- **A reservation takes its pages.** `ReserveKV` hands the sequence the pages
  its positions need rather than counting free ones as room: a counted free
  page is exactly what a compaction for another layer gives back.
- **Relocating layers at run time is unchanged.** A block moving on or off a
  device takes its pages' history with it as a load at the destination's P
  (above), and the pool of the device it left gives those pages back.

Weights and KV pages compete for one budget on a device; which one gives way
is the same arbiter's decision, not a fixed split.

**Nothing is reserved; a placed block holds its working set.** A layer holds
the dummy and one page per session attached to it, since a session with no
page cannot run a token -- a first frame, not a reservation of context. A new
layer is allocated at that size directly, so placing a block costs no resize.
Placement prices exactly that. Past it the pool grows as sequences
write, and a growth that does not fit pages weights out (roomOrReclaim) or
refuses with `ErrKVCapacity`. The fixed reservation the contiguous cache made
at placement (`devKVReserve` positions per block per session) is not carried
over.

## Ownership

    session (a model.State)  owns  sequences (one, or one per batch row)
    sequence                 owns  a page list (ids, in position order)
                                   its length (positions written)

A sequence is a session and the base of its rows (`seqID{session, base}`):
a single State is base 0, a LayersRows batch row is `slot - pos`. A submission
writes one table buffer before it runs: a dummy entry at 0 for padding rows,
then each sequence's ids once, concatenated, each row's descriptor naming where
its sequence's run starts. A windowed layer reads a second descriptor set whose
`keyStart` is the window's; the tables are shared.

- **Growth** is appending a page id. No copy, no kernel rebuild.
- **Retire** (a batch row ending) frees its pages at once. Not built: a row
  slot reused by a new sequence keeps and overwrites its pages (slice 5).
- **Detach** frees every page the session owns.
- **Batching** is a set of row descriptors: (table row, query position, causal
  length, window). Existing sessions become rows without copying history.

## What the kernels read: the ABI

Two data buffers join every attention and KV-writing kernel, after its existing
parameters and before any optional ones (sinks):

    pTab   u32   page ids. A row's table is a run of ids at tabOff, in position
                 order: position t is page pTab[tabOff + t/P].
    pRow   u32   one descriptor per query row, four words:
                   tabOff     where this row's table starts in pTab
                   keyStart   first key position attended (window/chunk start)
                   keyEnd     one past the last key position attended
                   writePos   the position this row's new K/V is written at
                 A padded row has keyEnd == keyStart and writePos pointing at
                 the dummy page's slot (below).

Everything a row's reads and writes need is in its descriptor, so the table's
width is data (it grows with the longest sequence) and nothing in a kernel
depends on it. A prefill query tile covers rows of ONE sequence; rows of
different sequences go through the ragged kernels, which read one descriptor
per row.

Position t of a row is then:

    pid  = pTab[tabOff + t/P]
    K[e] = kPool[pid*P*kvRow + e*P + t%P]          (transposed within a page)
    V[e] = vPool[(pid*P + t%P)*kvRow + e]          (row-major; f16 packs two)

**Tiles are aligned to absolute positions.** A key tile (the device's subgroup
width, which divides P) starts at a multiple of its width and masks keys below
keyStart, so a tile is always inside one page and reads its page id once --
including a sliding window that starts mid-page. Where a kernel walks keys with
a stride instead of a contiguous tile (the tiled FMA scores), each key is
translated on its own; neither needs a nested runtime loop. P is a multiple of
64, so four-float vector loads and the MMA staging keep their alignment.

**Attention is bounded chunks and a merge, everywhere.** Every path -- decode
and prefill, flash and staged -- processes keys in chunks of bounded length,
each producing an unnormalised numerator, a maximum and a denominator, and one
generated merge combines them (kernels.FlashAttentionMerge's protocol). Sinks and
the final normalisation are applied once, in the merge. Softmaxing chunks
separately and summing is wrong. This is what removes the context-sized score
plane from the staged path, and what eviction's streaming (below) runs on.

**The dummy page** is one real page per pool, zero-filled and never owned: padded
rows write into it and a zero-length row reads only it, so nothing reads memory
nobody wrote (RULE 13) and every read is finite.

**What building the kernels settled** (jit_old/gpu/kernels/paged.go, pagedstaged.go):

- The window is the descriptor's `keyStart` and nothing else: a paged shape
  refuses a baked Window, Chunked or KStride, so there is one source of truth.
- A row's table must hold the page containing `keyStart`: keys between the
  aligned tile start and `keyStart` are read (masked) from that page. The
  window release keeps that one (Which layers hold pages, below).
- An empty row reads no K or V, but its table still has at least one entry
  (the dummy); writes follow `pTab[tabOff + writePos/P]`.
- P is a power of two: t/P and t%P become a shift and a mask, and P=192 or 320
  measured ~20% slower in FlashAttention from the division.
- The staged softmax takes one parameter more, `pPart` (pA, pN, pOut, pPart,
  pTab, pRow), where it writes each chunk's maximum and denominator; and the
  staged path needs `Splits*Chunk >= keyEnd - align(keyStart)`.
- Parameters a paged kernel no longer reads (pN, pOff) stay in the list, so
  "the existing parameters, then pTab and pRow, then the optional ones" holds
  literally.
- How K is addressed is per driver (`FlashShape.KImm`): one base plus constant
  offsets on Metal, each index computed on NVIDIA's Vulkan (0.36x the other
  way); within a point on PTX.
- One launch serves several rows (`Rows` descriptors): paged FlashDecodeKV and
  the paged staged kernels are ragged decode as they stand.

**Packed f16 V** requires an even kvRow, which the current packing already
does; a position's V row is then whole words and no word crosses a page edge.

### Prefill

The prefill kernels have paged forms against the same ABI (pagedprefill.go,
pagedprefill70.go, and a Page on FlashPrefill70Shape): PagedAttnScoresTiled,
PagedAttnScoresMMA and PagedAttnScoresMMA70 for the three score kernels, a
PagedPrefillSoftmax, PagedAttnAccTiled and PagedAttnAccMMA70 for the two
accumulates, and FlashPrefill70 and FlashPrefillTile themselves with a Page.
What building them settled:

- **A launch (the flash kernels: a workgroup) is the rows of one sequence in
  position order.** They share row 0's table, keyStart and keyEnd do not
  decrease from row to row, and a padded row is empty at the last real row's
  keyEnd (`keyStart == keyEnd ==` it) -- the contiguous kernels ask the same of
  pN, whose padded rows repeat the last count. A query tile's keys are then
  `[keyStart of its first row, keyEnd of its last)`, which is what lets one K
  or V load serve every row of the tile. Rows of different sequences are the
  ragged kernels' (decode's paged kernels already are).
- **The staged kernels share a partition without a parameter.** Slot j of
  every row is key `base + j`, base the 64-aligned position at or below row 0's
  keyStart; split sp is slots `[sp*Chunk, (sp+1)*Chunk)`; the caller picks
  `Splits*Chunk >= keyEnd(last row) - base`. The scores, the softmax and the
  accumulate all derive it from row 0's descriptor. Key tiles (up to 64 keys,
  aligned relative to base) lie in one page; the tiled FMA scores translate
  each key on its own.
- **The softmax reads only the slots a row attends, and writes weights only
  where the accumulate reads them**: over its query tile's part of the chunk
  (the accumulate's tile, a parameter of the softmax), 0 where the row does not
  attend. So the matrix-instruction scores skip a key tile none of their rows
  attends -- on a causal chunk about half the contiguous kernel's tiles, which
  it computes and masks -- and a slot nobody wrote is never read; the
  accumulates mask their first and last block to the split, weights and V
  both. Writing the whole chunk cost the short-context rows twice what they
  attend and was measured at 0.44x the contiguous softmax on Vulkan.
- **Splits 0 is the direct form**: one chunk covering the launch, the weights
  normalised in the softmax (the sink joins there, as in SoftmaxRowsSink) and
  the accumulate writing the output -- no partials and no merge. It is the
  form for a span one plane covers; splits (and passes) are the form beyond.
  At 512 keys the merge alone was the difference between 0.80x and a win on
  the V100.
- **Every read is clamped into the tile's own keys before its page is looked
  up**, so no table entry past the row's table is read, and every value read
  outside those keys is selected away: a page's unwritten tail, the slots below
  the chunk's first key, a page below the window. The accumulates select V to
  0 there (0 * NaN is NaN). The m8n8k4 accumulate and the flash kernels walk
  each page's run of keys in an inner loop and read its id once a page; the
  FMA accumulate reads it once an 8-key block, a block ahead. A table read per
  tile, a dependent load at the head of every trip, was measured at 8% of
  FlashPrefillTile at head width 128 on the M4.
- **Packed f16 V reaches every prefill path**, which the contiguous prefill
  kernels never read; FlashPrefillTile stages the packed words as they are.
  **MLA's row-major region** runs through the tiled and the m8n8k4 pairs, as
  the contiguous tier runs it. **Softcap** is applied in the paged score
  kernels (the contiguous path runs it as a kernel of its own); **sinks** in
  the merge. A window of either kind is the descriptor's keyStart.
- **The flash kernels take Splits**: a workgroup's 32-key tiles (FlashPrefillTile:
  TK-key tiles from a multiple of 32) are divided into that many runs, each a
  workgroup writing FlashAttentionMerge's partials; the merge combines them
  and applies the sink. Splits 0 writes the output directly, as the contiguous
  kernel does; one split is the partials of one run, for a sink.

**Which kernels keep a plane.** The flash kernels keep none: the scores never
leave the workgroup. The staged kernels keep a plane of `Rows*Heads*Splits*
Chunk` floats -- the span a launch attends. On a windowed layer that is the
window plus the chunk; on a full-attention layer it is the context. To bound it
by a constant, the caller runs the history in **passes**: the three kernels
once per key range, with each row's keyStart and keyEnd clipped into the range
(clipping keeps them non-decreasing and empties a row with no key there, so
nothing in a kernel changes), and FlashAttentionMergeRun folding each pass's
partials into a running partial -- unnormalised, one split's record, so
FlashAttentionMerge at one split finishes it and adds the sink once. The plane
is then `Rows*Heads*(pass width)` whatever the context, and the same fold is
what eviction's streaming needs. The flash kernels at Splits > 0 fold the same
way.

**Defaults, from the measurements** (docs/engineering-history/gpu-kernels.md,
"Paged KV prefill kernels against the contiguous ones"): the staged path in
the direct form, Chunk the launch's span rounded up to 64, while that span is
at most 4096 keys; beyond it passes of 4096 keys, Chunk 256 and 16 splits a
pass, folded by FlashAttentionMergeRun and finished by FlashAttentionMergeWide
at one split. The softmax's query tile is the accumulate's. The flash kernels
at Splits 0, or 1 on a layer with sinks. The plane is then at most
`Rows*Heads*4096` floats: 256 MiB for a 512-row chunk of 32 heads, which is
the number to lower the pass width against on a small card.

**Packed f16 K** (`FlashShape.F16K`) packs two DIMENSIONS of one position a
word, `[kvRow/2][P]` words a page: the pair (e, e+1) of position t at
`pid*P*kvRow/2 + (e/2)*P + t%P`. Every writer writes whole words
(`PagedCopyRowsTF16`, `PagedRoPERowsTF16`, which rotates two NEOX pairs a
thread), so rows written in one launch never share a word. The decode readers
take it; the prefill readers do not yet, and the tier turns it on only once
they do.

Kernels to convert, each gated against the oracle with pages SHUFFLED in the
pool (an identity table proves nothing), across page edges and mid-page
windows, on CUDA, Vulkan and Metal:

    decode    FlashDecodeKV (the CUDA/Metal default), FlashAttention (+ split,
              merge), AttnScores/SoftmaxRows/AttnAcc, AttnAccSplit(F16), the MLA
              decode pair
    prefill   AttnScoresTiled(W), AttnAccTiled, AttnScoresMMA(W),
              AttnScoresMMA70, AttnAccMMA70, FlashPrefill70, FlashPrefillTile
              (built: see Prefill above)
    batch     AttnScoresRagged, AttnAccRagged
    writers   CopyAtRowsT, RoPERowsT, CopyAtF16, the prefill staging

## Which layers hold pages

A sequence holds pages in the ATTENTION layers its session runs on a device,
each layer its own ids -- a recurrent (linear) layer holds no KV and takes no
page; an MLA layer holds one region. The charge is each layer's own geometry, not a uniform one. A windowed
layer holds its window: before a call, every page wholly below the window start
of the call's first row less `nn.KVWindowSlack` (64 positions, what a
speculation rewind may go back) is released -- its id becomes 0 in the run, the
table entry the dummy, the id fenced, and an evicted page's home copy nil
(`devTier.windowPages`). A page the call writes into that was released (a row
restarting below it) takes a fresh id; one it would read is refused by name.
The host keeps the same rule (`kvPages.release`), so a session's history in a
windowed layer is the window and the slack, not the context
(`model.TestWindowedLayersHoldTheirWindow`). Ids are local to a
device; a step spanning several devices reserves its pages on all of them before
the first device runs.

## Lifetimes

- A pool that grows gets a NEW buffer: the copy and the swap happen outside any
  submission, and every captured graph naming the old buffer is dropped (as
  growth does today).
- A table is written before the submission that reads it and changes only
  between submissions.
- A freed page is reused only after the submission that last read it has
  completed. Sessions step on a device at once (slice 6), so a released page
  is fenced at the ticket the next submission takes and freed once every
  submission holding that ticket or an earlier one has completed -- at once
  when none is in flight (`jit_old/gpu/tier/inflight.go`). A page sent home, a
  pool grown, shrunk or compacted, and every table renumbering wait for the
  submissions in flight first (quiesce); a call whose history has pages at
  home streams them through free pages, and so runs alone on the device.
- Ids are stable while a submission can read them, and no longer: a
  compaction between submissions renumbers live pages and rewrites the tables
  that name them, and drops the captured graph, which names the old buffers.

## Limits

A pool's size is capped by the backend as well as the budget -- Vulkan binds a
storage buffer within `maxStorageBufferRange`, Metal a buffer within
`maxBufferLength`, and every index is 32-bit. Past the cap a layer's pool is
full: eviction or `ErrKVCapacity`, as for the budget. The charge is what is
allocated -- pool capacity, tables, the dummy page, scratch and the old-plus-new
peak while a pool grows -- not only the pages in use.

## Scratch

With every path in bounded chunks, a block's scratch depends on the model's
geometry, the batch width and the chunk and split counts -- not on how long a
conversation is. It is charged to the budget, reserved before a submission and
released after, and it is the same for every session: the capacity switching,
parked scratches and regrowth that the contiguous cache needed are deleted.

## Errors and eviction

Running out of pages, in order:

1. **Evict** a cold page to the host store (the host KV cache already seals,
   stores and faults pages back; device pages join it as per-sequence backing,
   with K's transpose and the f16 pack converted at that boundary). Attention
   over a history not all resident STREAMS: it runs over the resident pages in
   bounded passes, bringing the rest in a pass at a time into a bounded staging
   set, and merges the passes' partials -- the chunk-and-merge contract above.
   Faulting the whole history back would recreate the capacity cliff.

   **BUILT, and planned by the host rather than faulted in the kernel.** A
   page goes home only once it is wholly written, oldest first, the same page
   in every layer that holds the sequence, so the evicted prefix is one count
   and its table entries are the dummy's id. The attention over it goes in
   passes of as many pages as the layers have free (Config.KVStreamPages, or
   GPU.SetKVStreamPages at run time, adopted at the next call): a pass
   uploads its pages into free ids and points the table at them, waits for
   the pass before it, and runs the staged kernels over its keys alone; the
   passes fold (FlashAttentionMergeRun) and the last fold adds the sinks.
   Only sessions of one sequence give pages up. A prompt chunk over evicted
   history streams the same way: every row has its own descriptor, so a pass
   clipped to its keys serves each row's causal end, the passes reach the
   chunk's furthest row, and the evicted prefix goes up once a layer for the
   chunk rather than once a row (100 tokens over the gate's evicted history:
   400 streamed passes where a row at a time takes at least 1,600). Gate:
   `model.TestKVEvictionRunsPastTheBudget` -- Llama-3.2-1B, 320 tokens past a
   budget held a few pages above the prompt's, 4 pages of every layer sent
   home, all 16 blocks still placed, 1.7e-3 against the unconstrained card and
   3.700e-3 against the host (the unconstrained card's own figure); its
   violation, a pass that uploads nothing, reads 1.37
   (`TestKVEvictionGateDiscriminates`, jitllmfault).

   Why not the in-kernel fault below: every attention layer reads its whole
   history, so a fault saves no reads -- the host knows which pages are home
   because it sent them -- and it would cost a sentinel check in every
   kernel, a fault buffer, a no-op flag across the submission and a replay
   from the faulting block (with a recurrent block's saved state). It stays
   the design for a sparse or windowed read, which is where only the touched
   pages would move. Past the pool the whole evicted prefix crosses the bus
   every token, which is the ceiling of streaming from the host: a host-side
   partial over the evicted prefix, merged on the device, is the upgrade.

   The in-kernel design, unbuilt: a table entry
   of `0xFFFFFFFF` is a page not resident. Every key tile already loads its
   page id, so the check is one compare on a value it has: a tile that meets
   the sentinel writes (sequence, page) to a small fault buffer and returns,
   and a device-side flag turns every later launch of the submission into a
   no-op, since the layers after it would read a half-computed residual. The
   host reads the fault buffer with the logits, brings the pages in, and
   replays from the faulting block. Residency then need not be decided before
   a submission -- only pages actually touched are fetched, which is what a
   window or any sparse read wants. It is all data, so a captured graph stays
   valid. The one exception is a hybrid's recurrent blocks, which cannot be
   replayed once they have advanced: a fault after one restarts from the
   recurrent state saved before the submission (the `recSteps` lesson).
2. **Return** `ErrKVCapacity` from the call that needed the page. Pages are
   reserved before a submission and a sequence's length is published only on
   success, so a refused call changes nothing and can be retried.

A block is never moved home just because its history grew.

## The performance bar

Paging must not cost a token. The page lookup is one table read per key tile,
and the gain is the context-sized score plane and the regrow copies going away,
so a paged kernel is expected to match the contiguous one -- and it ships only
when a same-pass A/B (RULE 1: ratio, bytes/token, % of the measured wall, the
backend named) says it does, on decode and on prefill, on the laptop's CUDA and
the V100. Paging is the design, not an option under test: a paged path that
measures slower is fixed in the paged path -- the answer to a regression is not
to route around paging. It passed (docs/engineering-history/gpu-kernels.md,
"The decode plan through the tier"), and the contiguous control arm it was
measured against is deleted.

## Order of work, each slice gated against a violation

0. **Freeze the ABI** above -- descriptors, tables, chunk partials and the
   merge, the dummy page, lifetimes -- before kernels and pool work start in
   parallel.
1. **Accounting and errors.** Scratch charged; reservation before a submission,
   publication after; typed errors. Gate: injected allocation failures leave
   history, placement and counters as they were.
2. **Paged kernels**, behind the table contract above, in jit_old/gpu/kernels.
   Gate: shuffled pages, page edges, poisoned tails, MLA, f16 V, windows,
   sinks, on every backend, asserting the paged kernel ran.
3. **Tier integration**: the pool, page tables, growth by append; delete the
   capacity machinery. Gate: a growing sequence allocates only appended pages
   and copies nothing; two sessions' pages never alias, but for a page the
   prefix cache names, which every sequence that maps it holds and none
   writes in place (kvshare.go: a write copies it on the card). BUILT, MLA
   included, and the capacity machinery deleted; gated by
   `model.TestPagedKVMatchesTheHost` / `TestPagedKVBatchMatchesHost` and their
   violations (`TestPagedKVGateDiscriminates`, jitllmfault).
4. **Eviction** to the host store. Gate: history larger than the device's KV
   budget runs to completion with placement unchanged. BUILT
   (`TestKVEvictionRunsPastTheBudget`).
Later, once paging is in and at least as fast:

5. **Cross-session batching**: rows from any sessions; retire frees pages.
   BUILT for decode: `model.Step` runs one token for several States as rows
   of one ragged step (`nn.SessionStepper`), each row's sequence named by its
   own session (`ragStep.sid`), so the weights are read once for all of them
   -- 4.59x for eight sessions of Llama-3.1-8B on a V100. Gate:
   `model.TestStepAcrossSessionsMatchesEachAlone` (3 sessions, 4.0e-3
   against each alone; rows reading the current session's history instead,
   8.6e-1). A hybrid's linear blocks step across sessions through one pool of
   recurrent states (docs/design/device-sessions.md). Not yet: a scheduler
   that lines sessions up by the pages resident.

   **Latent attention runs in the step too** (MLA: DeepSeek-V2 and V3,
   Moonlight, Kimi-Linear's full blocks): each row writes its latent row and
   rotary key into its own sequence's pages before any row attends, the
   absorbs run grouped by head (a float bank reading the float activation),
   and every row attends its own pages through its descriptor in the decode
   plan (tier/mlabatch.go). A NoPE model (Kimi-Linear) copies its rotary
   channels there as in decode. Gates: `model.TestStepAcrossSessionsLatentMatchesEachAlone`
   (3 sessions, teacher-forced, every latent block of every step counted by
   `Stats.RowsLatent`, a poisoned-scratch arm: the synth-deepseek, -dense and
   -kimilinear fixtures at 1e-13 on CUDA, Vulkan, Metal and the V100;
   DeepSeek-V2-Lite 5.7e-3 and Moonlight 9.0e-4 on the V100; rows reading the
   current session's latent pages 0.52-2.5, NoPE channels rotated 7.6e-2),
   `TestSchedulerOnDeviceLatent`, `TestSchedulerMixesLatentPromptsIntoDecodeSteps`
   and `TestLatentPrefillBatchesOnTheDevice`. Kimi-Linear-48B itself (54 GB)
   fits no single card here.

   **A prompt chunk rides in the step, within one batch and across
   sessions.** `model.Scheduler` runs each step as the decoding rows plus up
   to a budget of prompt rows, a chunk of one sequence's consecutive
   positions, every row naming its slot and position (`State.forwardRows`);
   `model.StepRuns` does the same across sessions, each State taking a run
   of its next tokens -- one for a session decoding, a chunk of its prompt
   for one being admitted -- with its logits wanted or not
   (`nn.SessionStepper`). Every row's K and V are written before any row
   attends, so a chunk's rows see each other causally, and only the rows that
   want logits get the head (`nn.Head.LogitRows`); a step in which one row
   alone wants them projects it through decode's head matvec, reading the
   first of the step's quantized rows (`kernels.MatVecShape.ActRows`), where
   a batched twin would project every row. A hybrid's linear blocks take the
   chunk as a RUN: the kernels' run forms (`kernels.RunDescWords`) step each
   sequence's rows in position order through its one state, row k reading
   the state row k-1 left, so a hybrid's prompt rides in chunks on a device
   too. Gates: `model.TestSchedulerMixesPromptsIntoDecodeSteps` (a dense
   model and a hybrid; exact on the host, tie-bounded on the devices; chunks
   at mirrored positions part at 11-20 logits, a chunk's rows not chaining
   at 10), `model.TestStepRunsAdmitsAPromptBesideDecoding` (two sessions
   decoding while a third's prompt rides in chunks of three, teacher-forced
   against each alone: 1.8e-3 to 4.1e-3 dense, 6.8e-4 to 8.1e-4 hybrid on
   CUDA, Vulkan, the V100 and the M4; mirrored chunks 0.35-0.52, rows not
   chaining 0.39-0.40) and `model.TestStepRunsOneRowHeadMatchesTheBatchedHead`
   (0 to 5.1e-5 against the batched head; decode's own kernel reading the
   step's activation, 2.8-3.3).
6. **Replication and concurrency**: per-session routes and weight copies,
   per-session workspaces. Concurrency is built (device-sessions.md, "Sessions
   at once"): workspaces are lanes, a call's own, and the pool's lifetimes
   above hold with submissions in flight. Gates:
   `model.TestSessionsStepAtOnce`, `TestSessionsPageAtOnce`,
   `TestSessionsEvictAtOnce`. Replication is not.

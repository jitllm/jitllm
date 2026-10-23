# CPU/GPU placement and the seam — evidence

> **EVIDENCE, NOT INSTRUCTIONS.** This file records what was measured, what was
> tried, and what was wrong. It is the reason a rule exists, not the rule.
> **Binding requirements live only in `AGENTS.md`** -- if something here reads as
> an instruction, `AGENTS.md` is what governs. Numbers here are dated by the
> commit that added them; treat any of them as provisional until re-measured
> (RULE 1a: a baseline pinned to a version rots silently).
>
> Split out of a single 415 KB AGENTS.md, after a review
> identified the file's real failure as *unresolved authority* -- historical
> findings, current instructions and withdrawn conclusions all speaking in the
> same imperative voice.

## ★ RULE 15: PLACEMENT IS A RANKING BY READS PER RESIDENT BYTE, AND THE UNIT IS NOT ALWAYS A BLOCK.

`jit/gpu/tier/place.go` already says the ranking: placing a unit removes
`Uses*Bytes` of per-token host traffic and costs `Bytes` of device memory, so
value per cost is **Uses alone** -- how many times per token each of those bytes
is read. Not Uses/Bytes, which over-values whatever is small.

**★ 15a. THE DENSEST UNIT IN A MIXTURE OF EXPERTS IS THE OUTPUT PROJECTION, AND
IT WAS THE ONE THE CODE COULD NOT PLACE.** Qwen3-30B-A3B, per token:

    unit                    file MiB   share   Uses   packed MiB
      attention x48            493     26.9%   1.00      657
      output.weight            243     13.3%   1.00      318   <- ONE tensor
      router x48                48      2.6%   1.00       64
      experts (8/128) x48     1046     57.1%   0.0625     n/a
      whole block (weighted)   403.6            0.089

**A whole block is 0.089 and the head is 1.0, so the head is ELEVEN TIMES the
density of the block it displaces** -- and `offerRange` spent the budget on
blocks in LAYER ORDER, which is neither a density nor a choice. `prepDevice`
then offered the head only when the device had taken EVERY block, on the
reasoning that "with any block left on the CPU the residual stream has to come
home regardless, so folding the projection in saves nothing". True about ROUND
TRIPS, false about BYTES. Fixed: `placeHead` releases blocks from the END (so the
device's blocks stay the contiguous prefix `Layers(0, gpuLayers)` needs) until
the projection fits.

    measured  19.29 tok/s CPU-only, 1.92 GB/token = 37.0 GB/s of a 39 GB/s wall
              -- so the 30B host token IS memory-bound and placement is LINEAR
      head off the host    243 MiB / 37.0 GB/s  = 6.9 ms
      head on the device   318 MiB / 153 GB/s   = 2.2 ms
      one extra seam                            = 0.06 ms
    ~4.6 ms of a 51.8 ms token for 318 MiB: 2.3x attention's return per GiB.

**★ MEASURED AT 1.0566, IQR/median 0.032, AND NOT ON THE MODEL IT WAS DESIGNED FOR.**
Qwen3-MOE-4x0.6B at a forced partial seam (`-vram 400 MiB`, so 10 blocks and the
projection where the same budget takes 16 blocks without it), ABBA over
`JITLLM_NO_HEAD_PLACE`, n=96, four rounds, PSI 1.04%:

    placed  74.70 72.72 76.88 77.45 76.82 76.37 78.51
    host    69.67 69.37 72.88 68.28 74.02 71.47 74.75
    per-round ratios 1.0603 1.0933 1.0529 1.0503

Two samples of round 4 (32.93 and 47.22 against a session median of 74.36) were
discarded as RULE 2d clamps.

**★ AND THE 30B -- THE MODEL THE ARITHMETIC ABOVE IS ABOUT -- IS UNMEASURABLE ON
THIS CARD, WHICH IS RULE 1d-ter AGAIN.** `-devices gpu:0` on Qwen3-30B-A3B reads
4.2-5.0 tok/s against 19.29 CPU-only: with 2.5 GiB usable it refuses ~6700
tensors for want of memory and thrashes. Loading an 18.5 GB model per arm also
puts PSI at 45%, four times RULE 2b's gate. **Record the capability and the
arithmetic; do not quote a ratio for that row.**

**★ 15b. ONE HOST/DEVICE CROSSING COSTS ~36 us. AN EARLIER VERSION OF THIS RULE
SAID 64.3 us AND WAS WRONG THREE WAYS AT ONCE.** `GPU.PerLayerSubmit` is
documented as "one Session and one Read per block, which is what the seam did
before Layers existed" -- it is not a proxy for a per-layer seam, it IS one, so
`jitllm verify -ab submit` is the measurement. Corrected, on Qwen3-MOE-4x0.6B
(28 blocks + the head = 29 submissions against 1, so 28 extra), n=64, graphs
forced OFF on both arms, three passes:

    A 591.48  B 526.42  ->  36.3 us        A/B medians are one ROUND of n tokens
    A 594.28  B 537.15  ->  31.9 us
    A 605.13  B 537.04  ->  38.0 us        median 36.3, spread 31.9-38.0

That sits where it should: above `gpubench`'s 19.3 us CUDA floor and below
RULE 12's ~59 us per-matvec figure, which included a full drain.

**The three defects, because each is a different lesson:**

1. **I DIVIDED BY ROUNDS INSTEAD OF BY TOKENS.** `bench.AB(a, b, rounds, iters)`
   reports `AMedian`/`BMedian` as the median duration of ONE ROUND, and a round
   runs `iters` tokens. Dividing the delta by the round count instead of the
   token count inflated it by 1.77x. **Read the harness's units before dividing
   by anything it printed** -- this is RULE 11's "the harness is code too", in
   the smallest possible form.
2. **GRAPHS WERE ON FOR ONE ARM ONLY.** `layersOnce` disables capture whenever
   `PerLayerSubmit` is set, because it would otherwise record once per block per
   token. So an unforced run gave the per-token arm a replayed graph and the
   per-block arm raw launches, and the ratio carried both changes. `-ab submit`
   now forces `NoGraph` on both arms; the graph's own cost has its own arm.
3. **THE ARM DID NOTHING AT ALL WHEN THE HEAD WAS RESIDENT.** `Layers` guarded
   the per-block path with `hi-lo > 1 && head == nil`, so on any model whose
   projection is on the device -- which since `placeHead` is most of them -- the
   two arms were byte-identical and the reported ratio was measuring noise. The
   guard predates `layersOnce` accepting an empty block range, which is what
   makes a head-only submission expressible at all; with that in place the head
   is simply one more submission. **A benchmark arm that silently degrades to a
   no-op reports a number either way**, which is RULE 14r's skipping test one
   layer up.

**★ 15c. ATTENTION-ONLY PLACEMENT WAS WORKED THROUGH AND DOES NOT HOLD UP.** The idea is to put
the attention half of every layer on the card (0.52 GiB, ~32% of a 30B token)
instead of whole blocks (1.31 GiB, 8%). It fails on four counts, and the first
three are structural rather than economic:

1. **"Prep by omission" does not produce an attention-only layer, it produces a
   DECLINED one.** `PrepLayer` iterates all seven weights and `mkw` returns nil
   for `len(x.Data) == 0`, so a `LayerWeights` with Gate/Up/Down zeroed returns
   false. `blockBytes` skipping empty weights is a PRICING path, not the gate.
2. **It silently breaks the prompt.** `prefill.go` and `batch.go` both use
   `s.gpuLayers > 0` as the proxy for "the device owns some KV". Attention-only
   leaves `gpuLayers == 0` while the device owns EVERY layer's KV, so the
   batched host prefill runs and fills the HOST cache. `prefill.go` names that
   outcome in its own comment: "correct-looking output for the prompt and
   nonsense from the first generated token, which is the worst shape a bug can
   have."
3. **Its own gate cannot see either failure.** `forcedDiff` migrates only when
   `gs.GPULayers() != 0`, and it calls `Forward`, never `Prefill`.
4. **The economics never justified the repair.** The 32% is really 26.9% (the
   `Place` fixture's `Offered` omits `output.weight`), and the 0.52 GiB is
   really 1.02 GiB at 2K context -- RULE 12o's packing debt takes attention to
   657 MiB, and attention on EVERY layer forces every layer's KV onto the card,
   which is 192 KiB per position across 48 layers. That is the same VRAM class
   as the whole-block arm, with a context ceiling the whole-block arm does not
   have. It lands at 1.05-1.18x against an incumbent already at ~1.10x, and on
   Metal it is a net LOSS: a Session is one command buffer and `Commit` waits,
   so 47 extra command buffers is 13.6 ms/token.

**What replaces it is `tier.Place` wired to real hotness** -- attention AND the
head AND the hot experts, ranked, for the same 47 seams. `Place` is written and
tested and has NO non-test caller. Two things it needs first: a `KindHead`, since
the densest unit in the model has no UnitKind, and `Unit.Bytes` for an attention
group must include the KV cache, which `PrepLayer` charges and `place.go` does
not price.

**★ 15c-bis. THE ONE NUMBER THAT DECIDES EVERY PLACEMENT: A UNIT MUST CARRY AT
LEAST ~2 MB OF WEIGHTS PER HOST/DEVICE CROSSING.** Break-even is where the host
read it removes equals the device read it adds plus the seam it costs:

    bytes/R_host = bytes*pack/R_dev + n_seams*36.3 us
    with R_host 37.0 GB/s, R_dev 153 GB/s, pack 1.4 (RULE 12o)
      ->  bytes per crossing >= 2.03 MB

Every placement result in this file falls out of that one line:

    output projection      255.3 MB / 1 crossing  =  255.3 MB   126x   PAYS
    whole block (30B)      403.6 MB / 1           =  403.6 MB   199x   PAYS
    attention, one layer    11.6 MB / 1           =   11.6 MB   5.7x   PAYS
    one expert (g+u+d)       3.2 MB / 3           =    1.1 MB   0.5x   LOSES

**So the head is the best placement in the model not because it is big but
because it is ONE MATVEC that is big** -- the seam amortizes over a 255 MB read.
An expert is the opposite shape: 3.2 MB split across THREE matvecs, so it pays
three seams for HALF the threshold and loses every time.

**★ AND THAT IS THE ANSWER TO EXPERT-LEVEL VRAM PAGING, WHICH THE USER DOUBTED
ON DESIGN GROUNDS AND WHICH THE ARITHMETIC ALSO PRICES AS A LOSS.** The per-matvec path
(`s.mv` -> `JIT.MatVec` -> `GPU.MatVec` -> `resident`) is the ONLY path that can
make a sub-block unit resident today, and it costs one crossing per matvec. An
expert served that way costs 222 us where the host takes 87. Expert paging needs
a path that puts a whole LAYER's selected experts in one submission before its
economics change, and that is a different feature from paging.

**★ WHICH ALSO SAYS WHY `tier.Place` IS STILL UNWIRED, AND IT IS NOT NEGLECT.**
Its ranking is right and its four `UnitKind`s are not equally executable:
`KindAttn` needed attention-only placement, which 15c found does not work; `KindExpert` loses
on the line above; `KindFFN` has no path at all. The only sub-block unit that
both PAYS and has a way to run is the head, and `placeHead` does that directly
in twenty lines. **`Place` is waiting on an execution path, not on a caller** --
and it needs a `KindHead` before it could even express the one placement that
works.

**★ 15d. A CACHED DECLINE IS A TOMBSTONE, AND IT BROKE SEAM GROWTH SILENTLY.**
`resident()` registers its map entry BEFORE pricing it and leaves the FAILED
entry there, deliberately, so a tensor priced once is not re-priced by every
matvec of every token. The cost is that the entry says "no" FOREVER -- including
after the room it wanted has been freed. Nothing noticed while the seam only
ever grew from a cold device, because then no decline had happened yet; it bites
`SetGPULayers` growing after a shrink, and it bit `placeHead`, which freed
300 MiB and was still refused. `ReleaseLayers` now drops the failed entries and
ONLY those -- a live one is another block's weights and deleting it leaks the
buffer.

**★ 15e. THE GRAPH CACHE IS A MAP NOW, AND THE TWO AXES OF ITS KEY BEHAVE
OPPOSITELY.** One slot was correct while a token was one call. The head under a
partial seam makes two, under `{0,gl}` and `{gl,gl}`, and one slot makes them
evict each other EVERY TOKEN -- a capture per call, which is the most expensive
state there is, and strictly worse than graphs being off. `scoreN` only ever
INCREASES, so a key differing only there is dead and must be retired or the
generation leaks one recording per grain; `lo/hi/head` is a PLACEMENT and
alternates, so retiring on it is the thrash. `Captures` is the observable: one
per key per prompt is the design, one per TOKEN means a key is moving.

**★ 15g. A PARTIAL SEAM WAS SLOWER THAN NO DEVICE AT ALL, AND THAT IS THE
CONFIGURATION RULE 1d-ter SELLS.** RULE 15c-bis derived the break-even -- a unit
must carry ~2 MB of weights per host/device crossing -- and NOTHING ENFORCED IT.
`GPU.MatVec` predates `Layers` and priced a matvec by the VRAM budget alone, so
every matvec of every block the placement did not take was served one at a time
with a full round trip. tinyllama, 48 tokens, on a card with room for the whole
model, median of 3, both arms in ONE pass on a rested box:

                  before   after
      cpu          76.4
      gl=0         54.4    76.2
      gl=6         64.0    81.4
      gl=11        73.0    97.4    <- HALF the model on the device and LOSING
      gl=16        99.5   107.8
      gl=22       164.1   166.5

**Every placement up to half the model was BELOW the CPU tier**, and the curve
was not even monotone in the seam. "jitllm puts what fits on the device and runs the
rest on the host" was true about the weights and false about the speed: the rest
ran on the host the slow way, one crossing at a time.

★ AND "EVERY PARTIAL PLACEMENT WAS BELOW THE CPU" IS WHAT I WROTE FIRST AND IT IS
TOO STRONG -- gl=16 was already above it at 99.5. The first curve simply had no
gl=16 row, so the claim was extrapolated from three points and stated as four.

**★ AND THE FIX HAS TO BE PER MODEL, NOT PER TENSOR -- THE PER-TENSOR VERSION IS
THE ONE I BUILT FIRST AND IT REGRESSED THE OTHER MODEL BY 18%.** A byte
threshold on each matvec (8 MiB, chosen off the sweep) measures **1.3898 on
tinyllama at a half seam (IQR/median 0.069, n=12, ABBA) and 0.8242 on gemma at
gl=14** -- the gemma pass is REFUSED on dispersion at IQR/median 0.131, and is
reported anyway because all twelve of its ratios are below one, which is not
what dispersion looks like. The cause is ACTIVATION REUSE: q, k
and v read the same normed vector and `prepare` skips the upload when x is
unchanged, so declining q while serving v puts the upload back. **Mixing the two
tiers inside one layer costs more than either tier alone**, which is why the
knob has to be all-or-nothing over the set.

The average over what the seam actually leaves on the host is the statistic, and
it fits every measurement:

                per layer   crossings   MB/crossing   verdict
      tinyllama   20.1 MB       7           2.87       decline
      gemma-2b    59.1 MB       7           8.44       serve

    model              seam   MB/crossing  verdict   ratio    IQR/med
      Qwen3-MOE-4x0.6B  gl=14      small    decline   2.4485    0.095
      tinyllama q3_K_M  gl=11      2.87     decline   1.3169    0.026
      gemma-2b          gl=14      8.44     serve     1.0013    0.047
      Phi-3.5-mini      gl=16      9.10     serve     1.0174    0.058
      Qwen3-1.7B        gl=14      4.05     serve     REFUSED at IQR 0.149
      llama-2-7b        gl=16     16.3      serve     RULE 1d-ter: unmeasurable

**THE FOUR "SERVE" ROWS ARE THE CONTROLS AND THEY ARE WHAT MAKE THE TWO WINS
ATTRIBUTABLE.** A model whose verdict is "serve" runs the unpriced path by
construction, so 1.0013 and 1.0174 say the machinery costs nothing where it
decides nothing. **The MoE is the largest win in the change** -- its experts are
the smallest matvecs in any model here, so every crossing loses.

★ AND tinyllama WAS READ THREE TIMES, WHICH IS RULE 2c EARNING ITS KEEP: 1.4192
(IQR 0.098) on a warm box, then 1.2875 (IQR 0.163) REFUSED with the package at
1696 MHz -- RULE 2d's partial band -- then 1.3169 (IQR 0.026) rested. Quote the
rested one; the win is ~1.32, not the 1.42 the first pass showed.

**Monotone now, and every partial placement beats the CPU** -- see the table
above, whose "after" column is that curve. Teacher-forced 0/24
on tinyllama, gemma, the MoE and Phi-3.5 at a forced partial seam, so the seam
change costs no correctness.

★ AND A SINGLE-RUN CROSS-MODEL SWEEP TOLD ME Phi-3.5 GAINED 56%, WHICH IS RULE
1d-bis IN ONE LINE. It gained nothing: its verdict is "serve" and its gated ratio
is 1.0174. The 17.1 tok/s I read for its "before" arm was a contended box, not a
configuration. The same sweep understated the MoE by a third. **A sweep across
models is for choosing what to measure, never for the number.**

**★ THE SEAM CONSTANT IS RULE 12's ~59 us AND NOT RULE 15b's 36.3, AND I USED THE
WRONG ONE FIRST.** 36.3 us is one crossing around a BLOCK submission, where the
activation is already on the device and the readback is one residual vector. A
matvec served alone quantizes and uploads its activation, launches, and reads its
output back -- exactly what RULE 12 measured before `Layers` existed. Halving the
seam halves the threshold, which is the difference between declining tinyllama's
gate/up and serving them.

★ AND THE VERDICT IS A PROPERTY OF THE SEAM, NOT ONLY OF THE MODEL. What reaches
this path is whatever the placement LEFT on the host, so tinyllama at a deep seam
samples five layers plus a 51 MB output projection and can settle above the line
where the same model at gl=0 settles below it. That is the honest answer -- those
are the crossings that would actually be paid -- but two runs of one model can
settle differently, and `jitllm verify` prints the counter that says which.

★ RULE 14n's "~1.35 s is OPENING the device" IS RETIRED ON THE WAY PAST.
Instrumented, `backend.Open` is **95 ms** on this box (ptx 42, spirv 51, metal 0)
and the rest of that 1.35 s was this same defect: the first token paying ~200
crossings. The biggest time-to-first-token lever was never device open.

**★ 15f. AND THE LESSON THAT COST THE MOST TODAY: A COMMENT EXPLAINING WHY
SOMETHING IS *NOT* DONE IS A CLAIM ABOUT THE CODE, AND IT ROTS LIKE ANY OTHER.**
Three of them were load-bearing, all three were false, and each was guarding a
real win:

    "Reduce bakes ONE row count, which for an indexed launch is Rows*Slots"
        -> kernels.Reduce takes rows as a PARAMETER. Worth 1.31x on MoE.
    "folding the projection in saves nothing"
        -> true about round trips, false about bytes. Worth ~9% on the 30B.
    ROADMAP item 1's first bullet
        -> inherited the first one without checking it.

All three had the same shape: a decision was correct when written, the code
underneath moved, and the COMMENT is what got re-read. This is RULE 5t's "an
idea priced against an unoptimised baseline can flip sign once the baseline
improves" one level up -- there it was a number that went stale, here it is a
reason. **Before inheriting a "we do not do X because Y", check Y at the source.
It costs one grep and it has now paid three times in one session.**


## ★ 15h. A HYBRID'S SEAM: qwen3next RUNS ON THE CARD, AND BOTH THINGS THAT STOPPED IT WERE ABOUT THE FALLBACK RATHER THAN THE BLOCK.

Qwen3-Next-80B-A3B-Instruct, `-devices cuda`, 2 of 48 blocks on an RTX 3050 Ti
(4 GiB, 2.00 GiB budget), 34 block-runs, no demotion. Before this it read
**"The capital of France is is is is is is"**; it is coherent now, and the two
arms of the SAME binary part at token 10:

    cpu    ...Paris. The capital of Germany is Berlin. The capital of
           ids [... 576 6722 315 **9856** 374 **19846** 13 576 6722 315]
    cuda   ...Paris. The capital of Italy is Rome. The capital of
           ids [... 576 6722 315 **15344** 374 **21718** 13 576 6722 315]

★★ **AND THE SAME BINARY GIVES BOTH ANSWERS ON DIFFERENT RUNS, WHICH SETTLES IT
BEYOND ARGUMENT.** Two `jitllm run -devices cuda` invocations of the FIXED
engine, same prompt, same placement: one produced `...Germany is Berlin`
(matching the CPU arm token for token) and the other `...Italy is Rome`. Nothing
changed between them but `tuneSplit`'s per-process choice of reduction order.
**A greedy chain that agrees on one run and not the next is not a gate**, and
that is why the standing number for this seam is the logit diff below and not an
id list.

★ **AND "TOKEN-IDENTICAL TO llama.cpp" STOOD HERE UNTIL THE EXPERT BANKS WERE
FIXED, WHICH MAKES IT THE BEST EXAMPLE IN THIS FILE OF WHY A GREEDY CHAIN IS NOT
A MEASUREMENT.** That reading was taken before 15j, when blocks 3 and 7's routed
experts were ZEROS -- a mixture with two dead feed-forwards reproduced the
reference transcript exactly, and the CORRECT engine does not. The chain agreeing
was the broken configuration's accident; where it stops agreeing is decided by a
**0.13-logit margin** on "the capital of <country>", far inside the 0.2-0.94 band
this file records for f32 reduction order.

**The gate is the teacher-forced diff, which compares logits rather than
argmaxes: `max|dlogit|` 0.316-0.551 at every position, 1 of 8 positions flipping
an argmax.** See 15i for why an absolute difference across this seam cannot be
tighter than that.

**The block was never the problem and the decline that named it is lifted.**
`declineReason` returned *"a partial rotary (NRot < HeadDim)"*; with the rotary
tail copied before the rotation (`bs.copyQ`/`bs.copyK`) the block is right, and
`model.TestHybridAttentionBlockOnDevice` diffs it against the host at every
position over four fixtures -- the small one, the mixture, Qwen3-Next-80B's own
head geometry (2048 wide, 16 heads of 256 with 64 rotated) and its mixture
shape (512 experts of which 10 run, 512 wide). Every one gates at 1e-9 to 5e-5.

★ **THE FIRST CAUSE: A BATCH THE DEVICE CANNOT PREPARE WAS REPORTED AS A DEVICE
FAILURE.** `prepBatch` refuses a mixture of experts outright -- *"batched
prefill does not cover a mixture of experts"* -- and it refuses BEFORE it
submits anything. `Layers` returned false, `engine/model/prefill.go` read that as the
card breaking, demoted the whole placement and restarted the chunk on the host.
The rows now go through the device one at a time instead, which is what every
prompt did before batching existed and costs the speed and nothing else.

★ **THE SECOND CAUSE, AND IT IS THE ONE THAT TURNED A SLOWDOWN INTO WRONG TEXT:
A HYBRID CANNOT RESTART A TOKEN ON THE HOST.** The recovery for a device
failure is `refill()`/`fill()` -- back to the embeddings, then run every block
again -- and prefill.go's own comment says it is sound only at block 0. A
qwen3next device block is block 3, and blocks 0..2 are LINEAR: their gated delta
rule has already advanced, so re-running them applies the update twice. Nothing
faults, every shape is right, and the model is fluent. It is refused now
(`errRecurRetry`) rather than producing an answer. The upgrade path is to
snapshot `rconv`/`rstate` for the blocks BEFORE the seam at the head of each
token -- 3 of 36 layers on this model -- and restore them on the retry.

★ **AND THE REPORTS THAT HID IT FOR TWO SESSIONS WERE BOTH ONE LINE.** `run`
printed *"the device failed 1 time(s)"* and not why, while the tier held the
sentence naming the refusal; `verify`'s teacher-forced diff printed only the
positions whose ARGMAX flipped, so a uniform error looked like an intermittent
one. Both print the missing half now, and the per-position `max|dlogit|` trace
is what made the next entry measurable at all.

## ★ 15i. A DEVICE BLOCK'S DISAGREEMENT WITH THE HOST IS AMPLIFIED BY EVERY BLOCK BELOW IT, AND ON A HYBRID THE FACTOR IS 2550x OVER FORTY.

`model.TestHybridDeviceErrorIsAmplification` places ONE attention block of a
48-block hybrid and moves only its DEPTH. Same block shape, same weights, same
kernels; the arms differ in how many host blocks run below the seam:

| one device block at | host blocks below | logit NMSE |
|---|---|---|
| block 43 | 4 | **7.769e-06** |
| block 3 | 44 | **1.981e-02** |

**2550x, which is 1.21x per block.** Both tiers quantize activations to int8 and
do not round identically, so a placed block starts a perturbation whatever else
is right -- this is the same compounding `TestTowerOnDeviceMatchesHost` records
from the other side (*"5e-6 after one block is 1e-3 after twelve"*), and a gated
delta rule is a recurrence, so it travels harder.

★ **THE CONSEQUENCE IS A READING RULE, AND IT COST MOST OF A SESSION.**
Qwen3-Next-80B with ONE block at index 3 of 48 reads `max|dlogit|` 1.0-3.5 at
EVERY position including 0, where attention is a softmax over one key and the
block's output is exactly V[0]. That looks exactly like a broken block and it is
forty-four layers of amplification. **An absolute logit difference across a deep
seam is not evidence of a device bug** -- the falsifiable form is to move the
seam and see whether the error tracks the depth below it, which is one test and
a quarter of a second.

★ **AND IT IS WHY THE FIXTURE HAD TO GROW DEPTH RATHER THAN ONLY FEATURES.**
Four layers cannot separate "the block is wrong" from "the error compounds";
the whole point of `hyOpt.solo` is to put the same block at two depths in one
run.

## ★ 15j. A DEVICE IS HANDED A MIXTURE'S WHOLE EXPERT BANK AND THE PAGER NEVER READ IT.

`pageIn` skips the routed experts on purpose and says why in its own comment:

    ★ THE BLOCK'S OWN WEIGHTS, NOT ITS EXPERTS. A mixture's page is almost
    entirely expert bank -- 512 of them on Qwen3-Next-80B, of which a token
    touches ten -- so reading the page to reach the attention weights is the
    difference between 2.56 GiB a token and 37.8. The experts are ensured by
    moe(), after the router has said which ones, which is the only moment
    anyone knows.

    A bank is skipped here and NOT left half-read: EnsureRange tracks which
    chunks hold real bytes, so an expert nobody ensured is absent rather than
    stale. That is why every consumer of a packed span has to ensure it.

**Every consumer except the one that cannot come back.** `offerRange` calls
`pageIn` and then hands `Gate`/`Up`/`Down` -- the whole banks -- to `PrepLayer`,
which uploads them once. Whatever the frame held at that moment is what the card
runs for the rest of the process.

★ **THE TWO FAILURES ARE DIFFERENT AND BOTH WERE LIVE, WHICH IS WHY IT SURVIVED
A WHOLE SESSION OF HUNTING.**

    a FRESH frame        the bank is zeros. The block's routed FFN contributes
                         nothing, the shared expert and the attention are right,
                         and the model stays fluent. `jitllm run -devices cuda`
                         generated 64 correct-looking tokens this way.
    a REUSED frame       the bank is another block's Q4_K scales. An f16 that
                         happens to be NaN takes the logits non-finite, and
                         `jitllm verify` read "pos 0: logit 0 is non-finite".

**The only difference between those two commands is whether a host decode had
cycled the frames first** -- `run` attaches the device before any token, `verify`
benchmarks the CPU for sixteen tokens and attaches after. That is why one looked
perfect and the other looked like a kernel bug, on the same binary, in the same
minute.

★ **FIXED BY `Model.ensureBlockExperts`, AND THE ROW IS THE WHOLE ACCOUNT.**
Qwen3-Next-80B, 2 of 48 blocks on an RTX 3050 Ti, teacher-forced against the CPU:

    before   pos 0: logit 0 is non-finite, and the benchmark arm was non-finite
             from position 0 too -- every gpu tok/s printed during this
             investigation was the speed of computing NaN
    after    max|dlogit| 0.316 .. 0.551 at every position, 1 of 8 argmaxes
             flipped on a margin of 0.13

**0.32-0.55 is the f32 reduction-order band this file already records at
0.2-0.94.** Nothing is left over.

★ **THE GATE IS `model.TestMoEDeviceUploadReadsTheWholeBank`, AND MAKING IT FIRE
IS ITSELF THE FINDING.** Three fixtures passed against the violation before one
reproduced it, and the reason is arithmetic rather than luck: **EnsureRange
tracks residency in 1 MiB CHUNKS**, so a chunk that any routed expert lands in
arrives whole. A bank is only partly read while the number of expert SELECTIONS
is far below the number of CHUNKS:

    4 experts, 2 used, 8 tokens, a page under one chunk      every expert present
    512 experts, 2 used, 8 tokens, a 1.8 MiB page (2 chunks) both chunks touched
    4096 experts, 1 used, 2 tokens, a 7.3 MiB page (7 chunks) **NMSE 1.825e-01**
    Qwen3-Next-80B: 512 experts, 10 used, a 1042.8 MiB page   1043 chunks

So the 80B shows it under any prompt and a small fixture shows it under almost
none. A gate for a partially-read page has to be sized against the CHUNK, not
against the expert count.

★★ **AND THE COST IS PAID AT ADMISSION, NOT AT THE OFFER -- WHICH IS A SECOND
FIX, BECAUSE THE FIRST ONE READ A BANK FOR EVERY BLOCK THE BUDGET THEN
REFUSED.** `offerRange` reading eagerly cost Qwen3-Next-80B **8.5 GiB** on a
2 GiB budget: it offers twelve attention blocks, takes two, and read ten banks
of 1.04 GiB for nothing (55.41 -> 63.98 GiB of file reads on the same 12-token
run).

Everything `prepLayer` decides with -- `pageSizeWhy`, `blockBytes`, `room` --
reads SHAPES and `len(Data)` and never a byte, which is what makes the fix a
reordering rather than a negotiation: `nn.LayerWeights.Ensure` is a callback the
tier invokes at the point its own comment already calls *"the block is already
won"*. A caller that leaves it nil is unchanged, and a paged mixture is the only
one that sets it.

★ **THE GATE IS SELF-CALIBRATING, WHICH IS WHY IT CARRIES NO CONSTANT.**
`model.TestMoEDeviceReadsOnlyTheBanksItTakes` runs the same container under two
budgets and counts `jlm.File.ChunksIn`:

    budget for two blocks   2 placed   **16 chunks**
    budget for one block    1 placed   **10 chunks**
    the same, reading eagerly          **16 and 16**

Both budgets OFFER the same two blocks, so equal counts are exactly the defect
and the six-chunk gap is one expert bank. The per-token read is unchanged either
way: a placed block's host page is dropped immediately afterwards and a placed
block is not read on the host at all.

★ **AND THE 80B's OWN READ COUNT CLOSES ON THE ARITHMETIC, WHICH IS WHAT SAYS
NOTHING IS LEFT OVER.** Same prompt, same 12 tokens, same 2 of 48 blocks:

    before the bank was read at all   55.41 GiB
    read at the OFFER                 63.98 GiB   (+8.57 = ten refused banks)
    read at ADMISSION                 **57.27 GiB**

55.41 + two banks of 1.04 GiB is 57.49 against 57.27 measured. The only extra
reads left are the two banks the card actually holds.

★ **AND `PrewarmLayer` REFUSES A LAZY BLOCK, WHICH IS THE ONE PLACE THE HOOK
COULD NOT GO.** Its contract is that it "must be safe to call from another
goroutine while Layers is running", and `Ensure` reads the FILE into the pager's
frames -- the one thing that is not safe beside a running submission. Packing
the bytes without calling it would pack whatever the frame holds, which is the
defect this entry is about arriving by a second route. It declines; the upload
calls `Ensure` itself.

## ★ 15k. AN ARGMAX FLIP IS NOT A DISAGREEMENT UNTIL IT IS BIGGER THAN THE ARITHMETIC THAT PRODUCED IT.

`jitllm verify`'s teacher-forced diff failed on ANY flipped argmax. 15i says a
partial seam on a deep hybrid moves logits by 0.3-0.6 by construction, so the
tool was permanently red on the one model it had just been extended to cover --
RULE 10 from the other side, a gate that is always red is a gate nobody reads.

**The test is per-position and carries no constant.** Both quantities are
already in the loop: the CPU's own gap between the two candidates, against the
movement on exactly those two logits.

    gap   = |cl[ci] - cl[gi]|
    moved = |gl[ci] - cl[ci]| + |gl[gi] - cl[gi]|
    a flip with gap <= moved is a TIE this arithmetic cannot resolve

Qwen3-Next-80B, 2 of 48 blocks on CUDA, n=12, `-dlogit 1`: one flip, at position
3, **gap 0.163 against 0.359 of movement on those two logits**. `0/12 positions
disagree, 1 tie(s) inside the arithmetic, max|dlogit| 0.551 against a 1 band`,
and the command exits 0 where it used to exit 1.

★ **AND THE TIE RULE ALONE IS A DEGENERATE ORACLE, WHICH IS WHY `-dlogit`
EXISTS.** A device that moves EVERY logit explains every flip, so `gap <= moved`
would hold everywhere and the tool would pass a tier computing nonsense -- RULE
10's shape inside the fix for RULE 10's shape. `-dlogit B` is a ceiling on the
movement, and it also checks `max|dlogit|` at positions where NO argmax moved,
which an argmax comparison cannot see at all: a device wrong by a volt
everywhere still picks the same token where the margins are wide.

    verify (no band)        2/8 disagree, exit 1     <- unchanged default
    verify -dlogit 4        0/8 disagree, 2 ties     <- both flips explained
    verify -dlogit 0.001    2/8 disagree, exit 1     <- the violation check

★★★ **AND IT DEFAULTS TO OFF BECAUSE THE FIRST VERSION DEFAULTED TO 1.0 AND THE
FLEET REFUTED IT IN ONE RUN -- WHICH IS ALSO A CORRECTION TO THE NAIVE READING
OF 15i.** Llama-3.2-1B reads `max|dlogit|` **3.27** with `runs [[0 16]]` and the
output projection ALSO on the device -- no partial seam, no hybrid, not one host
block below anything -- which is six times the 80B's 0.55 through a two-block
seam:

    Llama-3.2-1B, runs [[0 16]] + head, 0 host blocks below   **3.27**
    Qwen3-Next-80B, runs [[3 4] [7 8]], 40 host blocks below  **0.551**

So the band is NOT a function of "how many host blocks run below the seam".
15i's measurement is about ONE block moved between two depths and it stands; it
does not extend to a fleet, because a full offload seeds the perturbation in
SIXTEEN device blocks and compounds it on the device side instead. **There is no
constant that covers both, a tool that picks one turns a working model red, and
the caller is what knows the seam** -- so the flag is how it says so, and a run
that quotes a band quotes it with the number (RULE 1).

## ★ 15l. THE WHOLE 80B RUNS ON A 2 GiB CARD: THE LINEAR DECLINE, THE SLOT FLOOR, AND A PAGER THAT UPLOADED THE WRONG BYTES.

`-devices gpu:0` on Qwen3-Next-80B put **36 of 48 blocks on the CPU** and said
why: *"a linear-attention block (gated delta rule)"*. The user's complaint was
exactly that -- *"i specify gpu only and it is using cpu"* -- followed by *"we
page layers so it's pretty possible to only run on gpu"*. Three separate things
stood between those two sentences and the run below, and only the first is the
feature anyone would have named.

    48/48 blocks + the output projection, 816 block-run(s), 0 on the host
    816 page-in(s), 863 page-out(s), 775.73 GiB moved through ONE slot
    ids [785 6722 315 9625 374 12095 13 576 6722 315 9856 374 19846 ...]

**0.06 tok/s against the CPU's 0.12.** One slot means every block crosses PCIe
every token; 775.73 GiB for seventeen tokens is that, priced. The rate is what a
second slot buys and this card cannot hold two 1042.8 MiB pages.

### 15l-1. The decline was right about the requirement and wrong about the conclusion.

`declineReason` refused a recurrent plan because *"a causal convolution over a
shifting history and a rank-one update to a per-head matrix, both
read-modify-write on a buffer that persists across tokens, which RULE 13 forbids
a device kernel outright -- so this needs TWO state buffers and a swap"*.
`recPair` **is** the two buffers and the swap, and every kernel `emitLinear`
issues is out of place. `kernels.GatedDeltaStep`, `Conv1dRows`, `Conv1dShift`
and `DeltaGate` were built, lowered on all three backends and gated against the
host the whole time, with nothing in the tier calling them --
`tier.TestDeclineNamesTheGraphFeature` recorded that as *"the honest state
rather than an oversight"*, which is how a task reads as a property.

`emitLinear` transcribes `engine/model/delta.go`'s order: the fused q|k|v projection,
the causal convolution and its shift, SiLU, the split gates, the per-key-head L2
norm with its two scales, one token of the rule out of place through `recPair`,
the gated output norm, and the output projection.

**Three bugs it cost, and every one was invisible to a green gate.**

★ **A DEADLOCK, NOT A SLOWDOWN.** `bs.dRows` carries the real row count, and
writing it with `Buf.Write` POSTS to the single CUDA goroutine -- from inside
`Session.Record`, which that goroutine is executing. The process sat at 0.4%
CPU forever. It is the hazard `prepBatch` already records one screen up (*"safe
to call from inside a Session, where Compile is not allowed"*) reached through a
copy instead of a compile, and `s.Write` is the form that enqueues on the stream
being recorded -- what the attention path's `s.Write(bs.n, ...)` does for the
key count.

★ **A BAKED ELEMENT COUNT, HIDDEN BY TWO FIXTURE CONSTANTS BEING EQUAL.**
`kernels.Quantize` bakes its element count, so `bs.quantF` -- built for
`rows*NFFN` -- cannot quantize the delta output, which is `VHeads*VDim` wide.
The fast fixture has `hyFFN = 32` and `hyInner = hyVHeads*hyVDim = 4*8 = 32`.
**Equal by accident.** The dense and moe arms passed; the 80B's own geometry
(NFFN 5120) was wrong at position 0 with logit NMSE 6.5e-01. `bs.quantD` is its
own kernel now, and this is the third instance of the same shape in this tree
after `quantE` and the attention output.

★ **ONE TIER, TWO BLOCK SHAPES -- THE VISION TOWER'S PROBLEM WITH THE OPPOSITE
ANSWER.** `g.bs` is built from whichever block arrives FIRST, and on a hybrid
that is block 0, which is LINEAR: its plan has `AttnOutGate` cleared, so the
scratch has no out-gate kernels and the attention block launched without the
gate.

    4 of 4 blocks placed
    blocks 0,1,2 (linear)      logit NMSE 4.5e-05
    block 3 (attention)        logit NMSE 1.7e-01

A VISION block gets a scratch of its own because its residual never meets the
text model's. A hybrid's does -- the two kinds alternate inside ONE submission
over one residual -- so two scratches would break the chain between consecutive
blocks. `planUnion` widens the plan over the two fields `offerRange` edits
(`AttnOutGate`, `Recurrent`) and `rebuildScratch` builds one scratch carrying
both kernel sets.

### 15l-2. minSlots = 2 refused a working configuration to preserve an optimisation.

The constant's own comment conceded the arithmetic and then declined anyway:
*"One slot is arithmetically workable -- every block would be paged in and
straight back out -- but it leaves nowhere to put block N+1 while block N is
still being computed, so it forecloses every overlap the design exists to
reach."* The user refuted it in five words: **"a swap needs two slots"** -- it
does not. A swap is evict, read in, run.

RULE 6: the arm a refusal is measured against is not the two-slot pager it hopes
for, it is the HOST. And it was not academic -- Qwen3-Next-80B's page is
1042.8 MiB against an RTX 3050 Ti's 2.00 GiB after headroom, so two slots want
2.04 GiB and the card is **one page short**, the cliff this project records from
the host side arriving on the device.

What one slot does not buy is the overlap, which is a rate and not an admission
rule. `tier.TestOneSlotStillPages` replaces `TestBelowTwoSlotsDeclinesAndSaysSo`
and asserts the opposite of it: 8 page-ins, 11 page-outs, 8 block-runs over two
tokens of four blocks, every block on the device.

### 15l-3. AND THE PAGER UPLOADED ANOTHER BLOCK'S WEIGHTS, FOR THE LIFE OF THE PAGER.

With the floor lowered, all 48 blocks went to the card and the model answered
**"The capital of France is!!!!!!!!!!!!"**, ids collapsing to token 0.

`l.pg.ws` holds `nn.Weight` structs whose `.Data` are **slices into a host
page**, captured when the block was admitted. A host page is a REUSED frame: the
host has its own pager, and a frame it evicts belongs to whichever block it
faults next. So the second upload of a block -- which is the only thing a device
pager does that a placement does not -- read whatever the host had most recently
put there.

**It is not specific to the one-slot floor and it is not new.** Reproduced on
Llama-3.2-1B at `-vram 600000000`, **14 slots**, on both backends, and
identically on the commit before any of this work:

    cuda    cuMemcpyHtoD: an illegal memory access (CUresult=700), blocks
            demoted to the host mid-run
    vulkan  no error at all -- "The capital of France is... (a) (a(("

★ **WHY NOTHING SAW IT: EVERY PAGING GATE RUNS ON A FAKE DEVICE.**
`jit/gpu/tier`'s pager is covered by `fakeDev` and `recBuf`, whose device memory
is a Go slice nothing else reuses -- so a page-in that reads the wrong host
bytes copies them faithfully and every counter agrees. RULE 11c in the pager:
the kernels were gated and the tier's wiring of them was not, one level down
from where that rule was written.

**The fix is that `nn.LayerWeights.Ensure` REFRESHES rather than only faults.**
It already existed for the expert banks (15j) and already had the right shape --
"a span's length and a span's contents arrive at different times". It now takes
the caller's `*LayerWeights` and re-points every `Weight` at where those bytes
are NOW: `model.ensureLayer` faults the page, reads the routed bank, and rebuilds
the struct through `State.layerWeightsAt` -- one ordered list, for the same
reason `layer.tensors()` is one list. `pageIn` calls it before reading a byte and
checks every refreshed shape against the admitted one, because a refresh that
returns a DIFFERENT block is exactly the bug it exists to stop and would
otherwise land as the right number of bytes in the right slots.

★ **AND KEEPING THE HOST PAGE INSTEAD WORKS AND IS WORSE.** Gating
`ReleasesHostPage` on `g.Paging` measures the same (3.563e-04 on the fixture with
both pagers live) and keeps a host copy of every block the card holds -- the
double residency that predicate exists to refuse. The refresh re-reads from the
container, which is the backing store the Scope table names, so the bytes still
MOVE.

★ **AND SETTING THE HOOK ON A MIXTURE ONLY FIXED HALF OF IT, WHICH IS THE SAME
MISTAKE ONE LEVEL DOWN.** `Ensure` was introduced for the routed bank, so
`offerRange` set it under `if c.MoE()` -- and what the pager needs it for is the
HOST PAGE, which every block has. With the mixture arm passing,
Llama-3.2-1B at `-vram 600000000` still faulted on CUDA. It is set on every
offered block now, and the gate below runs a DENSE fixture beside the mixture
for exactly that reason: a gate whose only fixture has the feature cannot see a
fix that is conditional on it.

★ **THE GATE, AND ITS FIRST VERSION WAS THE WRONG QUANTITY.**
`model.TestDevicePagingSurvivesTheHostPager` runs both pagers at once -- a host
budget of three pages over an eight-block model, `manyExp` so a routed bank sits
in each page -- against a real device. It first asserted `PageIns > 0` and
**passed against the violation it was written for**: with the refresh disabled
the device fails, `engine/model/` demotes the blocks to the host, and the host computes
the right logits (1 page-in, 8 page-outs, every position inside the bound).
*"It paged correctly"* and *"it gave up and ran on the CPU"* are the same logits.
The quantity that separates them is what RAN -- `Stats.Blocks`, which must be
NLayer per position -- and against the violation that reads 0 of 64 and names
the CUDA fault. The two arms fail DIFFERENTLY against it, which is worth
keeping: the dense one catches the wrong bytes directly (`logit NMSE 1.454e+00
after 8 page-in(s)`) and the mixture one catches the demotion.

★ **AND THE BUDGET IS SCANNED RATHER THAN A FLAT FRACTION.** A slot is what is
left after the permanent parts, and on the small dense fixture the KV cache and
the scratch are most of the card -- 1/8 to 1/2 of the total left ZERO slots, the
tier declined every block, and the gate reported that honestly as "demoted to
the host". It scans down and takes the tightest budget that still leaves one.

## ★ 15m. THE KV PREFIX CACHE CACHED NOTHING FOR ANY REALISTIC PROMPT, AND SAID SO POLITELY.

Two identical runs of `jitllm run -kv-cache DIR -chat` both reported
**`kv cache 0 of 15 prompt position(s) reused (0%)`**, with a helpful
parenthesis: *"(pages are 256 positions; a shorter prompt caches nothing)"*.
That is a correct description of a feature that was off for almost every caller
-- a chat turn is tens of positions, not hundreds -- and the message made it
read as intended rather than as unfinished.

Three things were missing and each one alone leaves the counter at zero.

### 15m-1. A page seals only once the position has passed it.

`seal` offers pages `[lastSealed, cur)` where `cur` is the page holding the
newest position, so a prompt that never reaches a boundary offers nothing.
`sealTail` stores the page the position is INSIDE, named by how many positions
it holds -- `pageKeyFill` hashes `seq[:n*p+fill]` with the fill in the digest, so
a partial page cannot alias a full one and `fill == p` returns the ordinary key,
which keeps every cache directory written before this working.

**The whole page buffer is stored, not the filled prefix of it.** A page is
`nseq*p*kvDim` floats with the sequences interleaved, so "the first fill
positions" is not a byte prefix for `nseq > 1` -- and a partial write would need
`fault`'s exact-length check relaxed, which is the one thing standing between a
short read and zeros being attended over as history. The tail is zeros and the
COVERAGE stops before them. On Qwen3-Next-80B that is 12 MiB of mostly-zero page
against a prefill measured in seconds.

### 15m-2. The clamp was in whole pages, so it threw the tail away.

`restoreKV` read `n = (max/p)*p`, which is right while every restored page is a
whole one and discards the answer the moment one is not: a 20-position tail
against `max = 19` floors to **zero**. It clamps in positions now, and the
truncation keeps the page holding position `n-1` instead of dropping it.

★ **AND A HYBRID CANNOT SHORTEN A TAIL, WHICH IS ARITHMETIC RATHER THAN
CAUTION.** The attention half is a COUNT -- using 13 positions of a page that
holds 14 is free -- but the recurrent half is a SUMMARY OF EXACTLY the positions
it was taken after, so resuming at 13 from a summary taken at 14 double-applies
a token to the gated delta rule. That is this file's own "fluent and wrong",
and no counter sees it. When the clamp would cut a tail on a hybrid the tail
goes back entirely. `sealRecurrentTail` stores the summary under the same
fill-named key; without it `faultRecurrent(n/p - 1)` asked for boundary **-1**
and panicked inside `PrefillCached` on the 80B.

### 15m-3. AND A FULLY CACHED PROMPT STILL HAD TO RUN SOMETHING.

`PrefillCached` restored at most `len(tokens)-1` because "a prompt entirely in
cache still has to produce logits". On a dense model that costs one token; on a
HYBRID it costs everything, because the tail cannot be shortened by one and is
therefore refused outright -- which is exactly the case anyone testing a cache
runs into, the same question twice.

`sealLogits` stores the row the prompt ended on under the same key, at a store
index past both the attention planes and the recurrent summaries. A full restore
then runs NOTHING: the position is already past the prompt and the first
generated token comes out of the store. One vocabulary row is 608 KB on the 80B
beside a 12 MiB page set. A missing logits row falls back to the old
`len(tokens)-1`, so a cache written before this keeps working.

### What it measures

    Llama-3.2-1B, CPU, `-chat`, 42-token prompt
      run 1   prompt 1.521s    kv cache 0 of 42 reused
      run 2   prompt 0.009s    kv cache 42 of 42 reused (100%)

    Qwen3-Next-80B, `-devices gpu:0`, `-chat`, 15-token prompt
      run 1   prompt 15.774s   kv cache 0 of 15 reused
      run 2   prompt 0.337s    kv cache 15 of 15 reused (100%)   47x
      ids identical

★ **THE GATE ASSERTS THE TOKENS, NOT THE COUNTER.**
`model.TestPrefixCacheHoldsAPromptShorterThanAPage` runs a 20-token prompt --
shorter than its page -- twice and requires the generated tokens to be
identical, because a prefix cache that restores the wrong history produces
fluent text and a happy counter. Against the behaviour it replaces it fails at
`a 20-token prompt left 0 page file(s): nothing was sealed`.

## ★ 15n. A PLACED LINEAR BLOCK'S HISTORY HAD NO WAY HOME.

`nn.LayerDevice` had `MigrateKV` and no twin for the recurrence. That was
harmless while the gated delta rule could not be placed, and became a wrong
answer the day it could (15l): `model.State.rconv`/`rstate` are what every HOST
consumer reads, and a block running on a device keeps its summary in the tier's
`recPair` and never touches them.

Two consumers read those arrays. `-kv-cache` seals through them, so it stored a
summary that had never seen the prompt; and `SetGPULayers` restores through them,
so a shrink would resume from one. `ReleaseLayers` calls `freeRec` -- the state
is FREED on a release, not brought home.

    attention-on-device   nocache [11 13 0 ...]   cached [11 13 0 ...]   OK
    linear-on-device      nocache [10 15 6 ...]   cached [10 12 2 ...]   WRONG

`MigrateRec(li, conv, state, toDevice)` is the twin. `syncRecFromDevice` brings
every placed linear block's summary home before the cache is sealed and
`pushRecToDevice` sends a restored one back, so both ends of the round trip
exist or neither does.

★ **THE LENGTH IS CHECKED AGAINST THE PLAN, NOT THE BUFFER**, because
`backend.Buf` is Write/Read/Free and does not report its size. `bs.rec` is what
`addSessionRec` allocated from, so it is the same number from the same place
rather than a second authority on the geometry.

### ★ AND ONE WRONG EXPLANATION GOT INTO A COMMENT BEFORE A VIOLATION TEST REFUSED IT

The first account was: `recPair` is double-buffered, seeding only the half `cur`
points at leaves ZEROS as the input to the second token. It is wrong. A
submission reads `cur` and writes the OTHER half, so `1-cur` is written before
anything reads it and seeding one half is already correct -- a gate built to
catch the single-half version passes it three times out of three.

What actually produced the readings that story was written on was a **test
confound**: an earlier probe kept every run's tier open through `t.Cleanup`, so
three tiers shared one card, and that alone moved the tokens. The shipped gate
opens and closes one tier per run.

Both halves are still written. That is defence against a future caller depending
on the parity argument, costs one copy per restore, and is recorded as such
rather than as a fix.

★ **THE PROBE THAT AGREED WITH THE WRONG STORY IS THE LESSON.** Reading the
device's state back at `cur` and comparing it against a good run reported
`0.000e+00` -- bit-identical -- while the tokens diverged. A probe that measures
the thing the hypothesis names will confirm the hypothesis; only the VIOLATION
test, which removes the fix and demands the failure come back, could tell the
two apart. Removing `syncRecFromDevice` fails the gate by name ("token 1 is 12,
resident-only gives 15"); removing the second half does not.

### What this says about relocation

`SetGPULayers` moves weights and calls `migrateKV` so a block that changes sides
keeps its attention history. A LINEAR block changing sides had no equivalent, so
the seam could only be moved before the first token. `MigrateRec` is the missing
primitive for that too -- the same call, with the arrow chosen by the caller --
and wiring it into the shrink and grow paths is what makes a hybrid's seam
movable at all.

## ★ 15o. STEP 1 OF THE BASE-RESIDENT GOAL: THE ACCOUNTING, AND THE VRAM CLOSES.

`model.TestDeviceBudgetAccounting` (JITLLM_BUDGET=<model.jlm>) separates the
three quantities a placement decision needs and that one "bytes per token"
figure runs together. Qwen3-Next-80B-A3B-Instruct-Q4_K_M, 48 layers, 12
attention and 36 linear:

    STORAGE  base per block MAX      26.96 MiB  ->  1.264 GiB for all 48
    STORAGE  expert banks            44.625 GiB
    STORAGE  dense region           417.31 MiB
    STORAGE  recurrent, ONE session 150.75 MiB   (2 buffers x 36 x 2.09 MiB)
    STORAGE  KV, ONE session          24 / 96 / 384 MiB at ctx 512 / 2048 / 8192

    LOGICAL  a decode token   base 1.199 GiB + routed experts 0.872 + head 245.75 MiB
    LOGICAL  BytesPerToken()  2.311 GiB      <- a weights-CONSUMED estimate
    PCIe     base RESIDENT    0.872 GiB/token   (the routed experts, nothing else)
    PCIe     nothing resident 2.071 GiB/token

    FITS?  card 1.885 GiB: base+recurrent 1.411 GiB, 0.474 GiB left = 260
           expert slots (a token routes 480, one layer at a time)

★ **THE FALSIFIER IS REFUTED AND THE NUMBER THE PROPOSAL WAS WRITTEN ON WAS
WRONG IN THE FLATTERING DIRECTION BY ACCIDENT.** The goal said "1.58 GiB of base
pages, 1.73 with recurrent, of 1.93" -- taken from an older note's 33.6 MiB.
Measured, the base is 21.29 MiB on four blocks, 21.54 on eight and 26.96 on
thirty-six, and a UNIFORM page is sized by the max rather than by any average:
**26.96 MiB, 1.264 GiB for all 48**. With the recurrent state that is 1.411 GiB
of 1.93, leaving 0.474 GiB for KV (96 MiB at ctx 2048), scratch and ~260 expert
slots.

★ **AND THE THREE QUANTITIES ARE KEPT APART BECAUSE CONFLATING TWO OF THEM
PRODUCED A SAVING THAT DOES NOT EXIST.** The first proposal claimed reads would
fall 2.31 -> 0.95 GiB/token; `BytesPerToken()` is what a token CONSUMES and the
host already fetches only the routed experts (per-expert sheets, 15l-1). What
moving the banks out of the page actually buys is RESIDENCY -- a base page that
fits a card, and an expert cache that survives its block's eviction -- not a
first-read saving.

★ **THE PCIe CEILING IS NOW A NUMBER.** 0.872 GiB/token at the ~8.2 GB/s this
box measures host-to-device is 114 ms/token, so ~8.8 tok/s is the ceiling of a
base-resident design with perfect overlap. The goal's >8 tok/s harness-bug line
is that figure and it was set before this was measured.

### 15o-2. Step 2, first result: the unchanged kernel serves a COMPACT bank

The design review's first risk was the sharpest: *"GPU indexed kernels turn the
MODEL expert id into offsets within three contiguous bank planes. A cache slot
is not that id. Merely allocating independent buffers breaks the current address
calculation."*

`backend.TestIndexedMatVecMatchesACompactBank` answers it. Two runs of the SAME
kernel:

    A   the whole bank, Experts=N, selection = the real expert ids
    B   only the routed experts concatenated, Experts=k, selection = 0..k-1

**Bit-identical**, on ptx and spirv, for Q4_0/Q4_K/Q6_K, including Qwen3-Next's
own shape (10 of 32, rows 512, k 2048) and the case where one expert is selected
twice. Against a violation -- keep the real ids against the compact bank -- it
fails on the first case.

★ **AND THE REASON IT WORKS IS THE LAYOUT THE CONTAINER ALREADY HAS.** A bank is
n_expert INDEPENDENT sheets back to back (`jlm.sheetsOf`), so "the selected
experts, concatenated" IS a bank of k experts and none of the kernel's three
bases (wrow/wrowD/wrowSC) changes. The id-to-slot translation the review asked
for is a rewrite of the SELECTION VECTOR and nothing else.

That is what makes step 2 buildable on v19: the remaining work is the
suspension -- ending the submission after `bs.rank` writes `bs.rsel`, reading k
ints home, uploading those sheets, and resuming -- and not a kernel or a format
change. `layer.go`'s own comment at that point still reads *"No step here needs
a value on the host, which is the entire reason the router is a kernel at all"*,
which is the line this work is about to make conditional.

★ **THE BAR IS BIT EQUALITY AND NOT AN NMSE**, because both runs do the same
arithmetic on the same bytes in the same order: only the buffer's extent and the
selection's values differ, so anything but equality is addressing rather than
rounding.

## ★ 15p. STEP 5 OF THE BASE-RESIDENT GOAL, DECIDED BEFORE IT WAS BUILT: THE FIVE-READER MIGRATION IS A PHANTOM, AND THE UNIT THAT HAS TO MOVE IS RESIDENCY RATHER THAN LAYOUT.

**Where the rate actually goes.** Qwen3-Next-80B, `-devices gpu:0 JITLLM_GPU_STREAM=1`,
48 of 48 blocks placed, 5 prompt + 6 decode tokens:

    drain (device sync)   0.37 s
    HOST READ            11.44 s   <- 79.5%
    upload (HtoD)         2.37 s   over 528 fill(s)
    pages   16 frame(s), 416 page-in(s), 352 eviction(s), 5.81 GiB in 24791 request(s)
    => ~2254 requests a token, 520 MiB/s and ~2167 IOPS on a device that does
       4.73 GiB/s with ReadAt x16. pprof: syscall.Pread 61% of samples.

**It is IOPS, not bytes.** Reading only the routed sheets already cut the bytes 3.2x
and the rate barely moved. The confound control says the same from the other side --
`JITLLM_GPU_STREAM_FIXED=1` pins the selection to 0..k-1 so the sheets share chunks,
and reads 15100 requests / 1.06 tok/s against the real selection's 24791 / 0.82.
Request count tracks rate; bytes do not.

**The cause is the page size.** A jlm page is a WHOLE BLOCK, 1042.81 MiB here, because
it must hold all 512 experts -- and a token needs ~19.84 MiB of it. A 16.37 GiB host
budget therefore holds 16 of 48 frames, so 32 blocks are re-faulted every token.

★ **AND THE CHEAP READER-ONLY FIX WAS CHECKED AGAINST THE LAYOUT, AND IT CANNOT WORK.** Shortening a frame to the base
extent needs the bank to be a suffix of the page. It is not:

    page starts at 448610304; base ends at +1042.81 MiB, bank starts at +15.88 MiB
    -- INTERLEAVED

### The fork, and the premise that was wrong

Two shapes were on the table. **A**: keep the three bank entries, store them
expert-major WITHIN each entry (stride `(nq+nd+nsc)*4`), so a routed expert is 3 ranges
instead of 9 -- ~3x, projecting ~1.9 tok/s. **B**: a third page array, one page per
(block, expert) holding gate‖up‖down, so an expert is ONE range and the base page falls
to ~27 MiB -- ~250 requests, projecting ~3.3 tok/s.

A was priced as expensive because it "changes the expert-offset arithmetic in all five
packed readers", with this file's own record of a plane change reaching `matvec.go` and
not `mma.go` as the reason to fear it. **That premise is false and a review refuted
it by reading the callers:**

    engine/nn/packed.go, the amd64/arm64 emitters, kernels/packedref.go
        receive already-resolved PLANE POINTERS. They compute no container
        expert offset at all.
    kernels/mma.go   MatVecMMA REFUSES Experts > 1 outright.
    kernels/matvec.go  the indexed MatVec is the ONE reader that computes an
        expert offset, and it stays correct as long as the UPLOAD preserves the
        plane layout it already expects.

**So the migration boundary is addressing and ownership, not arithmetic.** Keep each
expert's sheet packed exactly as it is today; change only where the sheet LIVES, and
gather at the seam: the CPU resolves an expert page into three ordinary `nn.Packed`
operands, the streamed tier gathers selected sheets into the compact QS/D/SC buffers it
already builds, and the resident tier populates the existing full plane buffers from
expert pages with bounded staging. No quantization and no kernel changes either way.

★ **THE DECISION IS B, AND IT IS NOT BECAUSE B IS 3.3 AND A IS 1.9.** Both projections
are soft. B is chosen because it moves the RESIDENCY unit as well as the read layout:
A leaves expert retention coupled to 1 GiB block frames, so the eviction churn that
produces the 352 evictions survives it, and A's own 1.9 already misses the 2 tok/s bar.

### The arithmetic, corrected

    R_expert/token  ~=  48 x 10 x (1 - h)      h = expert-page hit rate

    cold, no hits   480      47.9% hits   250      73.9% hits   125

**250 is a CACHE-HIT ASSUMPTION, not a coalescing result**, and two things this file
says elsewhere do not support it:

- the **73.9% reuse** figure in `engine/model/residency.go` is previous-token reuse on a
  FOUR-EXPERT model. It is not evidence for 512, and that manager tracks usage without
  implementing expert eviction.
- `jlm.File.Reads()` counts outer `readAt` CALLS; above 4 MiB one call fans out into
  `ReadThreads` `ReadAt`s. The 24791 above is not a syscall count.

★ **AND THE CEILING IS TIGHTER THAN THE PROJECTION.** Drain plus upload is already
2.74 s over 11 forwards = **249 ms a forward**, so even at zero read time this
configuration caps near 4 tok/s, and 2 tok/s leaves 251 ms for reads against today's
1040. B has to cut read time 4.1x to clear the bar, which is about what 2254 -> 480
cold requests buys and no more. **Quote the drain and the upload with any B result.**

★ **AND `EnsureRanges` PLANS ONE PAGE AND WAITS.** Ten experts called serially is ten
serial ~2 MiB reads -- fewer requests and less queue depth, which on this NVMe is the
quantity that matters. B must issue the block's missing expert pages CONCURRENTLY, and
each missing page must be ONE acquisition: nine successive plane-at-a-time ensures split
what should have been one read.

### What breaks, named rather than discovered later

1. **Eviction INSIDE one block.** Ensuring expert 9 can evict expert 0 before the CPU or
   the uploader has consumed it. `engine/model/moe.go`'s `moeBatch` ensures each token row's
   selection and computes expert-major AFTERWARDS -- today that union is one frame, and
   with independent expert pages later rows evict earlier rows'. `enterPager()`
   serialises SESSIONS and does nothing about this. Needs leases held through
   consumption, or fault-and-consume ordering.
2. **Binding before ensuring stops working.** `pageIn` calls `bindPacked` before routing;
   `ensureExperts` fills bytes and does not rebind, which is only safe while the expert
   spans point into the block's own frame. And `bindOne` returns early on an absent span
   WITHOUT clearing the old pointers, so a formerly resident expert keeps plausible bytes
   belonging to another one.
3. **Admission reads data presence as tensor presence.** `tier/page.go`'s `pageSizeWhy`
   skips a weight with empty `Data`, so a logical bank with no contiguous host payload
   becomes absent, uncharged or declined. `prepLayer` also builds `ws := weightList(w)`
   BEFORE calling `w.Ensure(w)` and relies on shared pointer updates.
4. **A smaller base page does not stay resident by itself.** `evictOne` has one MRU
   victim across every page, and `offerRange` calls `releaseLayer()` after upload.
   "Base read once for the life of the process" needs an explicit reservation, or a
   device path that stops re-acquiring base at all -- see the open item below.
5. **`BlockIsVision(li)` is `li >= NBlocks`**, so an appended third array would read as
   vision. `BlockOf`, `Span` and `ensureEntry` all assume the two-array mapping, and span
   validation today checks bytes fall inside the FILE -- an offset into the wrong
   expert's valid page is still in bounds.
6. **`ensureExperts` divides an ALIGNED bank length by NExpert** where `SheetSpan`
   returns the unpadded sheet. Equal on today's shapes, not on a small fixture.
7. **~73,728 expert entries** makes the whole-table scans in `ensureBlockRanges` and
   `selectedRanges` a per-token cost. Build block/expert lookup at Open.
8. **The budget and traffic reports go convincingly wrong.** `cmd/jitllm`'s
   `setPageBudget` reasons from base `PageSize` and returns early when every block is on
   a device -- streamed experts still hold host memory -- and `BytesPerToken` recognises
   a routed bank by `NDim == 3`, which 2-D per-expert entries would not be.

### The gate suite, five, each demonstrated red against a named mutation

| gate | asserts | violation it must catch |
|---|---|---|
| layout / address | distinct source bytes per (block, expert, matrix, plane); independently computed page id, offset, length | rotate expert ids in writer AND reader; swap gate/up; reuse gate geometry for down; omit the version bump |
| read plan and I/O | a cold ten-expert selection is ten exact page reads issued CONCURRENTLY; replay is zero; n controlled misses is exactly n | an off-by-one hidden by chunk rounding; plane-at-a-time loading; serial expert loading |
| eviction lifetime | tiny budget forces reuse while selected operands stay correct, including a CPU prefill whose selection union exceeds capacity; poison recycled frames | remove the rebind, the lease, or consume-time acquisition |
| tier wiring | CPU, fully-resident GPU and streamed GPU each consume independently verified sheets; unsorted ids, unequal router weights, demote and re-adopt | sort slots without their weights; keep real ids against a compact buffer; reuse stale descriptors |
| execution path | exact placement and stream counts, the decode/prefill path actually taken, no unintended base re-reads | streaming silently disabled with output unchanged; expert pages classified as vision |

Check both arms finite BEFORE comparing -- `NaN > bound` is false, and this file records
that shape passing a backend emitting `+Inf`.

★★★ **AND THE ITEM THAT NEEDED NO FORMAT CHANGE WAS BUILT AND MEASURED THE SAME
DAY: THE DEVICE'S CALLBACK WAS RE-READING A PLACED BLOCK'S BASE, AND STOPPING IT
IS 1.223 ON THE 80B.** `State.ensureSelected` passes `base=false` to
`selectedRanges` ON PURPOSE -- a placed block's own weights are on the card --
and then called `pageIn(li)` first, which is `ensureBlock(li)`, which reads
exactly those bytes. Its consumer is `tier.streamBank.fill`, which reads the
three bank planes and nothing else. `pageInSelected` claims the frame with the
SHEET read and binds after it.

Qwen3-Next-80B, `-devices gpu:0 JITLLM_GPU_STREAM=1`, n=6, 48 of 48 placed,
ABBA, two passes an arm, box 2.2% busy:

| arm | decode | host read | container bytes | requests |
|---|---|---|---|---|
| base | 0.78, 0.79 tok/s | 11.73, 11.73 s | 93.87 GiB | 25217, 25193 |
| **pageInSelected** | **0.96, 0.96** | **9.13, 9.13 s** | **83.48 GiB** | **23847, 23847** |

**ratio 1.223**, the two passes an arm agreeing to 0.01 tok/s and the fix arm
byte-for-byte identical between passes. Drain (0.39/0.40 s) and upload (2.4 s)
do not move, so the whole of it is the host read.

★ **AND THE BYTES FALL 11% WHILE THE TIME FALLS 22%**, which is the shape of the
thing removed: a block's base is fifteen-odd separate tensors where its sheets
are three big runs, so the base cost more per byte than it weighed.

★ **THE GATE TOOK TWO CORRECTIONS AND BOTH ARE THE FAILURES THIS FILE ALREADY
NAMES.** `model.TestTheDeviceCallbackDoesNotRereadTheBase`:

    v1  GREEN against its own violation. At the 1 MiB default the fixture's
        whole page is 14 chunks, so base and bank SHARE chunks and "base +
        sheets" reads the same 5 chunks as "sheets" -- the trap
        TestSelectedRangesCoverExactlyTheRoutedSheets is written about, arriving
        from the residency side. jlm.SetChunk(4096) is the fix, and it has to
        come before the fixture is opened.
    v2  STILL GREEN. The assertion was `narrow < full`, and base-plus-sheets is
        genuinely far less than base-plus-the-WHOLE-bank: 63 against 3489.
    v3  RED at 63 chunks against a plan covering 30. The bar is EXACTNESS --
        planRuns never bridges a gap between two wanted ranges, so on an empty
        frame ChunksIn is precisely the distinct chunks the plan covers.

### The chunk IS the amplifier, and the dose that said otherwise was measuring a dead knob

★★★★ **THE FIRST DOSE OF `JITLLM_CHUNK_KIB` CAME BACK FLAT TO 0.16% AND WAS
WRITTEN UP HERE AS "ROUNDING IS REFUTED". IT WAS RIGHT ABOUT THE TABLE AND WRONG
ABOUT THE WORLD: THE KNOB REACHED NOTHING.** `jlm.SetChunk` applies to containers
opened AFTER it, and it lived in `gpuOptions()`, which `openDevices` calls at
`main.go:398` -- forty-seven lines after `model.Open` latched the granularity at
line 351. **`JITLLM_CHUNK_KIB` had never moved a byte on the run path.**

    1024 / 256 / 64 / 256 / 1024 KiB   83.54  83.48  83.41  83.44  83.50 GiB
                                        0.95   0.96   0.96   0.96   0.96 tok/s

**A knob that reaches nothing and a knob that changes nothing produce the same
table**, and a sixteen-fold dose landing inside a third of a percent should have
read as the former. RULE 10's "check the configuration under test was actually
SELECTED", in the instrument rather than in the engine.

★★★★ **WITH THE CALL MOVED ABOVE `model.Open` THE SAME DOSE IS 2.03x, AND IT IS
THE LARGEST SINGLE LEVER FOUND ON THIS PATH.** `JITLLM_GPU_TUNE=0`,
Qwen3-Next-80B, `-devices gpu:0 JITLLM_GPU_STREAM=1`, n=32, both arms twice:

| chunk | decode | host read | container bytes | requests |
|---|---|---|---|---|
| 1024 KiB | 0.96, 0.89 tok/s | 29.3, 31.7 s | 167.19 GiB | 75325 |
| **16 KiB** | **1.97, 1.80** | **9.8, 13.0 s** | **71.69 GiB** | 126490 |

per-round ratios **2.05 and 2.02**. At n=6 the same pair is 0.96 -> 1.95.

★ **THE MECHANISM IS EXACT AND THE FLOOR IS ARITHMETIC.** A routed bank is asked
for per PLANE, and the smallest plane is **exactly 16384 B** -- so a 1 MiB chunk
rounded a 16 KiB request up sixty-four fold. Below 16 KiB it SATURATES: 16, 8
and 4 KiB read byte-identical 53.92 GiB in 40063 requests, because no plane is
smaller than one chunk any more.

★★★★ **AND THE GRANULARITY IS A STORED PROPERTY OF THE CONTAINER NOW, COMPUTED
AT CONVERSION -- container v20. THAT IS THE USER'S CALL AND IT IS THE RIGHT
SHAPE, BECAUSE THE VALUE IS A PROPERTY OF THE MODEL AND NOT OF THE PROCESS.**
`jlm.chunkFor` takes the smallest span any reader asks for on its own -- per
(sheet, plane) for a bank, the whole span for a verbatim tensor, ignoring
dense-region and Expanded entries because those are never paged -- and rounds it
DOWN to a power of two in [Align, 1 MiB]. `Header.Chunk` carries it, `Open`
uses it, and `decodeHeader` REFUSES a value that is not a power of two in
[Align, PageSize] rather than substituting a default: an unaligned chunk makes
an unaligned page read, which the O_DIRECT handle answers with EINVAL, and a
silent substitution would turn a corrupt header into a slow correct run.

★ **THE STORED VALUE CANNOT BE APPLIED TOO LATE, WHICH IS THE WHOLE ARGUMENT
AGAINST THE GLOBAL IT REPLACES.** A package variable has to be set before
`Open`, and there is no way to tell a caller that they missed -- which is the
bug above, in its own right rather than as an accident. It also cannot disagree
between two open containers, which `cmd/jitllm`'s pages line was doing when it
multiplied `ChunksRead()` by `jlm.Chunk`.

`jlm.SetChunk` survives as an OVERRIDE so a dose can still force a value over
the file it is dosing; `jlm.Chunk` is now only that override and the fallback
for a File built with no header. `run` prints `pager N KiB chunks` at load,
because the pages line only appears when frames are short -- a model that fits
would otherwise never say what it was converted for.

★ **AND THE 80B STORES 4096, NOT THE 16384 THE BANK ARITHMETIC PREDICTS, WHICH
IS THE CONVERTER BEING RIGHT ABOUT SOMETHING I HAD NOT COUNTED.** The smallest
span in that container is not a bank plane at all -- it is `ssm_ba`'s d plane,
**2048 B**, on each of the 36 linear blocks -- and `Align` clamps the floor to
4096. It costs nothing: 4 KiB and 16 KiB read a byte-identical 71.69 GiB in
126490 requests and measure 1.96/1.82 against 1.97/1.80 tok/s, because every
plane is `alignUp`'d in the file so no request is finer than Align anyway.

    stored chunk = 4096 B
      2048 B  ssm_ba blk17 plane1 (packed, 1 sheets)   [and 35 more]

The one thing it does cost is the residency bitmap, PageSize/Chunk bools a
block: 266,960 at 4 KiB against 66,740 at 16 KiB, and `planRuns` allocates a
`want` slice over the span of a request's chunks -- ~267 KB a call for a
scattered bank at 4 KiB. **Measured neutral, recorded as the one reason someone
might floor this at the smallest BANK plane instead**; it was measured neutral and
no code was written for it.

★★ **AND THE FIRST VERSION OF THAT VALIDATION REFUSED EVERY TOWER-ONLY
CONTAINER, WHICH IS WORTH KEEPING BECAUSE OF HOW IT WAS NEARLY FILED.** The
ceiling was `h.PageSize` -- and a container with no TEXT blocks carries a
degenerate `PageSize` of `Align` while its real pages are the VISION array's, so
the check asked for a chunk in [4096, 4096] and rejected the 32768 its own
converter had just computed. Three standalone `mmproj-*` conversions failed and
the first thing written about them was *"let me confirm that's the EXPECTED
refusal"*. It was not a refusal at all, it was this bug. The ceiling is
`max(PageSize, VisPageSize)` now.

★ **TWO THINGS SAVED IT.** `convert` reads its own output back -- *"wrote X but
cannot read it back"* -- which is the check that turned a silent bad container
into a loud failure; and the user's standing rule, stated the same day: *"our
goal is to always reduce refusal so we support more models."* A refusal is a
candidate for removal, not a boundary to document, and the sentence that calls
one expected is what stops the search.

    the gates   jlm.TestTheStoredChunkIsTheSmallestRequestedSpan derives the
                expectation from kernels.PackedWords the way a range plan does,
                and reads 1048576 against a wanted 16384 when chunkFor is
                stubbed to the old constant
                jlm.TestABadStoredChunkIsRefused covers zero, non-power-of-two,
                below Align and larger than the page
                cmd/jitllm.TestTheChunkKnobReachesThePager gates the override's
                WIRING, which is the thing that was broken

★ **AND IT COSTS A CONTIGUOUS READ NOTHING**, which is what makes it a default
rather than a tuning for one model: `planRuns` coalesces ADJACENT wanted chunks
into one run, so a block read whole is one run at any granularity. The reads
only multiply where the requests were already scattered -- 75325 -> 126490 --
which is precisely where the rounding was being paid, and the bytes fell 2.33x
against a request count that rose 1.68x.

★★ **THE TOKEN IDS MOVED WITH THE CHUNK SIZE, AND IT WAS NOT THE PAGER.** At 64
KiB the last id was 15344 and at 32/16 KiB it was 9856, each reproducible -- and
9856 is the CPU reference. A pager setting changing arithmetic would be a
serious bug; it is `tuneSplit`, which picks per PROCESS and moves the f32
reduction order, exactly as the three-way entry records. Under
`JITLLM_GPU_TUNE=0` all four chunk sizes give **identical ids**
`[785 6722 315 9625 374 12095 13 576 6722 315 9856]` at unchanged rates. **Pin
the tuner before reading ids off a pager dose.**

### Where the remaining time is

At 16 KiB and n=32 the per-forward split is **read 0.26 s, upload 0.21 s, drain
0.03 s**. The upload moves ~950 MiB a forward in 0.21 s = ~4.5 GiB/s, which is
this laptop card's PCIe limit -- so it cannot be made faster, only made smaller,
which means keeping experts ON the card across tokens. The read runs at ~2.4
GiB/s against a 4.73 GiB/s device figure, and `EnsureRanges` plans ONE page and
waits, so the block's ten expert pages are fetched one after another. **Issuing
them concurrently is the next lever and it needs no format change.**

★ **AND THE ~3.3x OF UNATTRIBUTED PER-FORWARD BYTES IS ACCOUNTED FOR BY THE
ABOVE.** It was the rounding: 10 experts x 9 planes a block against a 1 MiB
chunk is ~9 MiB where the sheets are 1.984. At 16 KiB the arithmetic closes.
**The sheet figures are measured, not assumed** -- per-sheet 606208 B for
gate and up, 868352 for down, so one expert across the three banks is
**1.984 MiB**, and the block's non-bank page content is 22.81 MiB.

★ **WHAT THE BYTE COUNTER STILL CANNOT DO.** It is WHOLE-RUN and the `host read`
timer is STREAM-ONLY, so a decode token's bytes cannot be separated from the
~48.7 GiB of one-time admission (48 blocks x 1042.81 MiB, `ensureLayer` reading
each whole bank once). **No "% of the read wall" in RULE 1's form can be quoted
for a decode token until there is a per-phase counter**, and that is owed.

## ★ 15q. THE READ AND THE UPLOAD WERE SERIAL, AND THEY ARE THE WHOLE TOKEN.

A streamed block spends its token in two phases and the card is idle for both.
Measured on Qwen3-Next-80B, n=32, 48 of 48 placed:

    drain (device sync)   0.03 s a forward
    host read             0.26
    upload (PCIe)         0.21
    -------------------------------
    the token itself      0.52

**Nothing overlaps anything.** The GPU's own work is what is left, about 0.02 s.

★★ **THE GROUPS ARE OVER THE SELECTION, NOT OVER THE PLANES, WHICH IS WHAT MAKES
IT CHEAP.** The compact bank is k sheets back to back, so slots [lo, hi) are the
byte range [lo*n, hi*n) of every plane -- one `backend.Buf.WriteAt` per plane
per group, and the existing per-expert read serves a subslice of the selection
unchanged. Group g uploads while group g+1 reads.

| groups | 1 | 2 | **3** | 5 | 10 |
|---|---|---|---|---|---|
| tok/s | 1.94, 1.91 | 2.19, 2.24 | **2.33, 2.34** | 2.15, 2.08 | 1.65 |
| host read | 10.2 s | 7.6 | **6.1** | 7.6 | 10.4 |
| upload | 7.8 s | 8.1 | 8.6 | 8.8 | **11.3** |

**1.21x at the knee**, ABBA, and the token ids are IDENTICAL at every setting.
Past three the upload itself regresses, because a group is a separate transfer
per plane and a small host-to-device copy is priced per call.

★★★ **AND THE KNEE FALLS OUT OF THE MODEL'S GEOMETRY, SO IT IS STORED AT
CONVERSION -- container v21.** The quantity that limits splitting is the
SMALLEST plane of one sheet, since a group moves ceil(k/n) of them: the rule is
the largest n whose group still clears 64 KiB. On this model k is 10 and the d
plane is 16384 B, so n=3 moves four sheets = 64 KiB and clears it while n=4
moves three = 48 KiB and does not. **Three is exactly where the dose turns
over**, so the arithmetic PREDICTS the measurement rather than being fitted to
it. `jlm.streamGroupsFor` computes it, `Header.StreamGroups` carries it, and
`tier.Config.StreamGroups` still overrides.

★ **THE ONE NUMBER THAT IS A PROPERTY OF THE MACHINE IS SAID SO.**
`minGroupBytes = 64 KiB` is where the per-call cost turns over on this box's
PCIe and is fitted to one measurement. Convert is nonetheless the right home
for it, and not merely an acceptable one: a container is written where it runs,
which is why `Fingerprint` already records the host, the device and its VRAM.

### Two bugs, and only a counter could see either

★★ **PrefetchExperts WAS SILENTLY CLEARED AND THE PIPELINE NEVER RAN ONCE.**
`ensureLayer` and `ensureSelected` rebuild the weights from the model and then
re-point them at the caller's closures -- copying `Ensure` and `EnsureExperts`
and nothing else. `prepLayer` calls `Ensure` BEFORE it builds the streamed bank,
so the new callback was nil at construction. The dose read 1.93 / 1.93 / 1.82
across three group counts, which is exactly what a working knob with no effect
looks like. Every callback is carried over now.

★★ **AND THE COUNTER BUILT TO CATCH THAT WAS ITSELF UNWIRED.** `Stats.add` is a
hand-written list of 58 fields and `StreamOverlaps` fell off it, so the run
printed `0 overlapped read(s)` while the host read fell 10.00 -> 7.00 s and the
decode went 1.91 -> 2.18. **A selection check that is itself broken is worse
than none: it reports the wrong answer confidently.**
`tier.TestStatsAddSumsEveryCounter` walks every field `Stats` HAS by reflection
and demands `add()` moved it -- asserting the fields someone remembered to
assert is the same omission one level up. Against the violation it names the
field.

★ **THE SAFETY ARGUMENT IS STRUCTURAL, NOT A TIMING CLAIM.** Only the first
group goes through `EnsureExperts`, which assigns through its `w` pointer and
binds the block's spans; the rest go through `PrefetchExperts`, which reads and
touches nothing shared. Two `EnsureExperts` in flight would be a data race on
the model's own layer, so splitting the READ out is what makes the overlap safe
by construction.

★★★ **AND THE ROW THAT MATTERS IS THE 8G ONE, NOT THE 24G ONE. THE LARGER CAP
WAS FLATTERING THE RESULT AND IS A RISK TO THE HOST.** The user's correction:
*"a cap of 24G is a cheat and risk to kill this host. i suggest 8G,
real test with constraints."* Both halves are right. This box has 31 GiB, so a
24 GiB cgroup leaves seven for everything else -- the state AGENTS.md records
systemd-oomd killing the desktop from three times -- and it also hands the pager
**15 frames of 48**, which is 31% of a model that is supposed to be streamed.
Under 8G it gets **four**.

    cap    frames   decode (n=32)        page-ins / evictions
    24G    15       2.31, 2.30, 2.20     1306 / 1243
    8G      4       2.12, 2.05           1713 / 1661

**Criterion 2 holds under the honest constraint**, and the gap between the two
caps is only 0.2 tok/s -- because the pipeline hides the extra faults behind the
upload, which is the same reason it was worth building. Paired at 8G:

    groups 1   1.77, 1.65 tok/s   host read 11.6, 12.4 s
    groups 3   2.03, 2.01         host read  7.1,  8.0 s

**ratio 1.18, and at 8G it is the pipeline that carries the row over 2 tok/s at
all.** Every jitllm row in this file should be taken at 8G unless the model
genuinely cannot open there; the 24G figures above are kept only as the paired
control they were measured as.

★★ **AND TWO DEVICE GATES HAD BEEN SKIPPING SINCE THE COMPACT BANK LANDED,
WHICH IS WHAT "ARE YOU TESTING THE GPU AS WELL?" TURNED UP.** Both asked the
card to take FEWER blocks than they offered and relied on a VRAM budget to
refuse the rest -- true when they were written, false once a shared compact bank
made a block cheap:

    TestHybridDeviceErrorGrowsWithPosition   "wanted 1 block(s), got 4"   SKIP
    TestMoEDeviceReadsOnlyTheBanksItTakes    "wanted two ... got 8"       SKIP

Neither is a capability gap and neither should have been a skip. The growth gate
offers exactly what it wants now -- the seam is a contiguous prefix, so offering
`want` places the same blocks without depending on a card's spare bytes -- and
the read gate MEASURES the budget at one and two blocks instead of reading it
off whatever the default admits:

    budget 14174048 B: 2 block(s), 4592 chunk(s) read
    budget  7087024 B: 1 block(s), 2864 chunk(s) read

Both pass, and the read gate's assertion (one strictly fewer than two) is
discriminating again. **RULE 11b: a gate that stopped testing after someone
else's change is that change's unfinished half**, and a skip is how it hides.

## ★ 15r. THE PAGER'S BYTES, SPLIT BY PHASE -- AND THE STREAMED PATH IS AT 97% OF THE DEVICE'S READ RATE.

Every byte counter in this engine was whole-run, so RULE 1's per-token question
had no answer for the streamed path and this file said so twice. The totals are
dominated by ADMISSION: dividing them by the token count overstated a decode
token threefold.

`model.PhaseMeter` marks at load, prefill and decode. Qwen3-Next-80B,
`-devices gpu:0 JITLLM_GPU_STREAM=1`, n=32, `scripts/cap 8G`, box 4.2% busy:

    load      45.82 GiB in    336 request(s),   48 fault(s),  4.06 GiB/s
    prefill    4.28 GiB in  21087 request(s),  225 fault(s),  4.53 GiB/s
    decode    27.08 GiB in 133362 request(s), 1440 fault(s),  4.57 GiB/s
    -------------------------------------------------------------------
                77.18 GiB -- the whole-run total, exactly

**The load row is 45.82 GiB in 336 requests over 48 faults**: one whole bank per
block, read once at admission, which is 59% of everything the run reads and
none of it per-token.

★★★ **SO THE ROW RULE 1 ASKS FOR IS FINALLY QUOTABLE, AND IT SAYS THE READ IS
SATURATED RATHER THAN SLOW.**

    decode   2.05 tok/s = 908,699,648 B/token = 4.57 GiB/s while reading
             = **97.2% of a 4.70 GiB/s wall**

★ **AND THE WALL IS MEASURED IN THE SAME PASS ON THE SAME CONTAINER, WHICH IS
WHAT RULE 1 ASKS AND WHAT THE FIRST VERSION OF THIS ENTRY DID NOT HAVE.** It
quoted the 4.73 GiB/s recorded in `format/jlm/read.go`, which is a figure from another
session -- exactly the carried baseline RULE 1's first line forbids.
`dev/tools/readbench -pages 8` on the 80B's own file, immediately after the decode
run:

    ReadAt serial         1.53 GiB/s      ReadAt parallel x8    3.90 GiB/s
    ReadAt parallel x4    2.49            ReadAt parallel x16   **4.70**

It is a separate PROCESS in the same pass rather than the same process, and
that is the remaining gap: a same-process probe would cost every run its
startup. The box was 11.6% busy for the wall and 4.2% for the decode, which
depresses the wall and therefore FLATTERS the ratio -- but 4.70 against the
4.73 taken quiet says contention moved it ~0.6%, so 97% is good to about a
point.

★ **AND readbench's mmap ROW IS NONSENSE AND MUST NOT BE QUOTED**: it prints
13,290,280 GiB/s, because a fault is lazy and the probe touches nothing. RULE
1's "above the roofline is a harness bug" catches it instantly, which is the
only reason it is harmless.

★★ **AND 0.846 GiB/token IS 92% OF THE ROUTED-EXPERT TRAFFIC THE TOKEN ACTUALLY
CONSUMES** -- 48 blocks x 10 experts x 1.984 MiB = 0.93 GiB. **The 15x
amplification this file's open question records for the over-committed case is
0.92x on this path.** The pager reads what the token needs and essentially
nothing else, at nearly the device's rate, which retires "the host read
dominates because it reads too much" as a lever: what is left is the bytes the
model genuinely routes to, and the PCIe upload beside them.

★ **THE RATE IS OVER THE READ TIME, NOT THE WALL, AND THE DIFFERENCE IS NOT
COSMETIC.** The same decode phase reads 27.08 GiB over a 15.6 s WALL, which is
1.65 GiB/s and reads as a 35% deficit against the device -- a floor, because the
wall includes compute. Over the 6.9 s the tier actually spent in `TStreamRead`
it is 4.57. **Only the second is a bandwidth**, and a row that does not say
which one it is will be compared to a datasheet by the next reader.
`PhaseRow.OverRead` reports it and the line prints `reading` or `of wall`.

★ **THE DESIGN IS SNAPSHOT DELTAS, NOT COUNTERS IN THE PAGER.** Nothing in `jlm`
knows what a phase is and nothing should: a second set of counters could drift
from the first with nothing to notice, and a delta of the SAME counter cannot.
That is what the gate asserts --
`model.TestPhaseMeterAccountsForEveryByte` requires the rows to SUM to what the
pager read since the meter started, an equality rather than a plausibility check
on each row, and requires the decode row to be non-zero so the one row this
exists for cannot silently be nothing. Against the violation (Mark not rebasing,
so every row counts from the start) it reads `+5414912 unaccounted`, which is
the prefill row counted twice.

★ **AND THE FIRST VERSION PRINTED `load 0.00 GiB`, WHICH IS THE FAILURE THIS
WHOLE ENTRY IS ABOUT, ONE LEVEL UP.** The meter started after `setPageBudget`,
which is after device placement -- so the 45.8 GiB of admission fell outside the
accounting entirely and the line read as "loading is free" rather than "loading
is not being counted". It starts at `model.Open` now. A zero that means
*unmeasured* and a zero that means *nothing happened* print identically.

## ★ 15s. RELOCATION: A CONTEXT PAST THE CARD'S ROOM MOVES ONE BLOCK, NOT ALL OF THEM.

`State.SetRelocate` / `jitllm run -relocate`. Before each token (and each
prefill chunk) the State asks `ReserveKV` for the next position; while the
device refuses, its highest block brings its history and its summary home and
is released. It is asked BEFORE the token rather than after a failure because a
failure inside `Layers` is recovered by restarting from the embeddings, which a
hybrid cannot do (errRecurRetry). The CLI takes the flag only where `-devices`
admits the host: `auto`, or a list with `cpu`.

Llama-3.2-1B, 16 blocks, a budget that holds every block and one 256-position
page but not the doubling (`model.TestRelocateKeepsTheDeviceWhenTheContextOutgrowsIt`):

    without -relocate   1 demotion, 0 of 16 kept
    with -relocate      1 block relocated, 15 of 16 kept, 0 demotions
                        device-vs-host 4.07e-04 decode / 6.71e-04 prefill (CUDA),
                        3.46e-04 / 3.44e-04 (Metal) -- each its own band

Against the violation (the history not brought home) the same arm reads
2.41e-02 / 3.73e-02.

★ **AND THE DEFAULT PATH WAS WRONG UNDER THE SAME BUDGET, WITH NO FLAG SET.**
`ensureKVCap` set `kvCap` to the new capacity and then regrew block by block, so
a budget refusal at block k left blocks 0..k-1 at the new stride, the rest at the
old one and `kvCap` naming the new -- and `demoteAll`'s rescue read every
block's cache at `kvCap`'s stride. The fallback finished the run on the host at
**7.38e-02 against the host**: fluent, and wrong from the first position past the
page. `devTier.regrowKV` now checks the budget for the whole growth, allocates
and copies every cache before swapping any, and leaves the tier untouched on a
refusal -- which is also what makes relocation's ask-again sound. The control arm
of the gate holds the demoted run to the host and fails against the old tier.

### Grow to fit, and take from the card that refused.

**A doubling is an amortisation, not a requirement.** `devTier.kvFit`: when the
doubling does not fit, the history grows to the largest page multiple that
does and still holds the position; only when not even one page fits does the
device refuse. A card with room for 768 positions used to refuse at 513 --
demoting every block by default, or relocating one -- for room it would not
need for another 255 tokens. `tier.TestKVGrowsToWhatFitsBeforeRefusing`:
512 -> 768 where 1024 does not fit, and the refusal at 800 leaves the tier
untouched. Red against doubling-only (refused at 600, cap 512).

**"Holds the position" is `pos < cap`, not `cap >= pos`, and the difference is
a token refused after it was reserved.** Relocation reserves `pos+1` and
`Layers` then ensures the same `pos+1`; a growth to exactly that capacity passes
the first ask and fails the second. The same gate asks position 768 twice and
reads `true then false` against that violation.

**On several cards the highest block is on the wrong one.** Blocks fill the
fastest card first, so the model's top sits on the last card while the first is
the one out of room, and relocating the top emptied the last card before it
touched the first. `tier.GPU.Refused` names the refusing device's blocks and
`State.relocationVictim` takes the highest of those. The host block it leaves
between two device runs costs no crossing: the residual already comes home
between two devices.

**And that hole found every path that walked the prefix.** `gpuLayers` stops at
the first host block, so `SetGPULayers`' shrink, `SetDevice`'s detach and
re-attach, the prefix-cache restore and seal, and both recurrent-summary syncs
all iterated `[0, gpuLayers)` and skipped the blocks past a hole. They walk the
placed set now (`placedFrom`), and `offerRange` skips a block already placed.
`model.TestRelocationTakesFromTheDeviceThatRefused` pins a single card's
refusals on blocks 0-7 (a wrapper, so it runs on every host): block 7 moves,
15 of 16 stay, 3.9e-04 / 3.2e-04 against the host on CUDA / Metal, and after
`SetGPULayers(0)` the tail matches too. Red against the highest-block victim
and against the prefix shrink (`model: no such kv page`).

### Relocated blocks come back when room does.

A relocated block used to stay on the host for the life of the State. The
State now remembers which blocks relocation moved and offers them back, lowest
first, when the tier's room report (`GPU.RoomGen`) moves: a refund anywhere on
the device, or a budget raised. Its own relocations are recorded as seen, so a
sequence never takes back a block its own growth just gave up.

Two sources of room were missing and are built:

- **`State.Close` kept its history on the card.** It closed the JIT and nothing
  else, so every sequence that ever ran on a long-lived tier left its caches
  behind. `gpuSession.Detach` frees the session's history and recurrent state
  on every block and leaves the weights to whoever still uses them.
- **A shorter sequence could not shrink the history.** The capacity only grew.
  `devTier.TrimKV` shrinks it to what a fresh prompt needs, only when every
  cache on the device is the caller's. It runs at a prefill from position zero
  and before a prefix-cache restore. A chat turn re-prefills the whole
  conversation, which needs at least what the last turn grew to, so a growing
  chat never trims and never bounces blocks; a new, shorter conversation does.

`model.TestRelocatedBlocksComeBack`, Llama-3.2-1B at the relocation budget:
a 64-token prompt after a 272-token sequence gets every block back (4.0e-04
against the host, band 4.0e-04); a budget raised at position 320 brings the
block back at the next token with its history (3.8e-04). Red against no trim
(`0 reclaim(s), 15 of 16`) and against a reclaim that does not send the history
up (**2.5e-01** -- fluent, wrong). `tier.TestTrimAndDetachGiveRoomBack` fails
against a trim under a second session and against a no-op detach.

★ **STILL OPEN: a release is not per-session.** `ReleaseLayers` frees every
session's history for the block, so one session relocating a block on a shared
tier breaks the next token of any other session using that block. The other
session fails with a device error rather than wrong text (its `MigrateKV` for
the block is refused). Per-session block ownership is what fixes it.

## ★ 15t. THE SYNCHRONOUS EXPERT READ IS A WARM-UP AND NOT A RATE, AND A SPECULATIVE PREFETCH WAS PRICED ON MEASURED NUMBERS AND LOSES.

`ensureExperts` does synchronous `pread`s on the critical path of every mixture
layer, and AGENTS.md priced them at **6.1% of samples** (`Syscall6`, with
`jlm.EnsureRanges` beneath it) and named them the largest remaining term of the
MoE serial stretch. This is the measurement that was owed.

**★ THE FIRST THING MEASURED WAS NOT PREDICTABILITY, IT WAS WHETHER THE COST
PERSISTS -- AND ON A MODEL THAT FITS IT DOES NOT. THE 6.1% IS AN AMORTISING
CONSTANT.** The profile it came from is the in-process probe, which warms 8
tokens and measures 64. Same binary, same model, `-cpuprofile`,
olmoe-1b-7b-0924-instruct Q4_K_M, `-devices cpu`, six P-cores, `scripts/cap 8G`:

    n=  64    Syscall6  **7.46%** of samples   **0.65 s** absolute
    n=1024    Syscall6  **0.64%**              **0.75 s** absolute

**The share falls twelvefold and the seconds do not move.** There is no
per-token cost to remove; there is a fixed ~0.7 s of reading the bank once, and
a longer run divides it by more tokens. A prefetch cannot hide a cost that has
already stopped being paid.

★ **AND THE PAGER'S OWN COUNTERS SAY THE SAME THING WITHOUT A STOPWATCH.** The
decode phase's request count, by interval:

| tokens | requests | per token |
|---|---|---|
| 1 - 64 | 3393 | **53.0** |
| 65 - 320 | +648 | **2.53** |
| 321 - 1024 | +171 | **0.24** |

It is coupon collection over the bank: 64 experts x 16 layers, and the total
expert bytes converge to **3.34 GiB**, which is the whole expert population of
the container. Once a selection's chunks are filled they stay filled for the
life of the page, so the steady state is 0.24 requests a token -- not zero,
because the tail of the bank is still being discovered, and falling.

**★ THE INSTRUMENT THIS NEEDED DID NOT EXIST AND IS THE HALF WORTH KEEPING.**
Every host "GiB/s" in this tree was a FLOOR over a wall that includes compute --
`PhaseRow.Rate`'s own comment says so, and only the device path had
`tier.Stats.TStreamRead`. `jlm.File.ReadWait` is the wall a CALLER spent blocked
in `EnsureRanges` (the reads run in goroutines and the caller blocks on
`wg.Wait`, so it is serial decode time by construction), `phRead` feeds it to
the phase meter on every host run, and `model.Model.ExpertIO` separates the
ROUTED read from the block page-in -- because only one of the two needs a
prediction. `jitllm run` prints both:

    decode   117.13 GiB in 209507 request(s), 2457 fault(s), 4.11 GiB/s reading
    experts   94.15 GiB in 210651 request(s) over 4176 layer-step(s),
              22.934s blocked of which 22.902s is I/O

**★ THE REGIME WHERE THE READ IS REAL IS THE OVER-COMMITTED ONE, AND THERE IT IS
TWO THIRDS OF THE TOKEN.** Same model at `-maxmem 2400000000`, 7 frames of 16,
n=256, box 3.0% busy at the start and 1.5% at the finish:

    7.48 tok/s          against 48.49 fully resident
    491,280,512 B/token 2473 page-in(s), 2466 eviction(s)
    experts 94.15 GiB   = **79% of every byte the run read**
    22.9 s blocked      = **~67% of the 34.23 s decode wall**

★ **AND IT ARRIVES BY ITSELF AT LONG CONTEXT, WHICH IS THE PART TO REMEMBER.**
`setPageBudget` subtracts the KV reservation from the weight budget, so the same
model and the same box cross the cliff on the token count alone:

    -n 1400   16 frames   **47-48 tok/s**    1,330,310 B/token
    -n 2048   15 frames   **12.58 tok/s**    178,709,904 B/token

That is AGENTS.md's "being one page short of fitting is a cliff, not a gradient"
happening without anyone choosing a budget, and the expert read is 93.6% of what
the short arm reads.

**★ NOW THE PREDICTABILITY, AND IT IS THE SAME NUMBER ON A SECOND MODEL.**
`model.TestRoutingLocality`, olmoe (16 layers x 64 experts, 8 used), 400 tokens:

    PREVIOUS-TOKEN PREDICTOR   **34.8%** of experts reused   (chance 12.5%)
    whole set unchanged          0.6% of layer-steps

Qwen3-30B-A3B (48 x 128, 8 used) reads **33.9%** at chance 6.2%. Two models, two
bank sizes, the same answer -- so this is a property of routing and not of one
file, and `residency.go`'s 73.9% does NOT transfer: that was the 4-expert toy,
where chance alone is 50%.

**★ AND THE WINDOW CURVE IS THE VERDICT, WHICH THE SINGLE NUMBER CANNOT GIVE.**
A scheduler that issues reads at the start of a token knows the last W tokens'
selections and nothing else, so the set it can ask for is their union. Both
columns are needed, because in the only regime where a prefetch could pay the
token is already bound by the bytes it reads:

| W | hit | experts fetched | fetched/used |
|---|---|---|---|
| 1 | 34.8% | 8.00 | **1.00x** |
| 2 | 48.4% | 13.22 | 1.65x |
| 4 | 59.7% | 20.87 | 2.61x |
| 8 | 77.0% | 30.65 | 3.83x |
| 16 | **98.5%** | 40.14 | **5.02x** |

**To reach neurostream's 99.5% you must fetch 40 of 64 experts, which is most of
the bank -- exactly the read the expert-major narrowing was built to remove**
(2.56 GiB a token against 37.8, `pageIn`'s own comment).

**★ SO THE RESULT IS ARITHMETIC ON THE TWO TABLES, IN THE CONFIGURATION THAT IS
THE LEVER'S BEST CASE.** Per token of the over-committed row above: 394.9 MB of
routed experts read at the measured 4.11 GiB/s, 96.4 MB of base page-in, and a
133.7 ms token of which 111.4 ms (83%) is reading.

    perfect oracle, 100% hit, no extra bytes   111.4 ms   **1.20x**  -- the ceiling
    W=1  prefetch (34.8% hit, 1.65x bytes)     169.8 ms   **0.79x**
    W=16 prefetch (98.5% hit, 5.05x bytes)     474 ms     **0.29x**

The W=1 device time **exceeds the whole token it is trying to accelerate**, and
that is with every byte of compute assumed perfectly hidden. A speculative
expert prefetch loses at every window setting, and it loses worse the better it
predicts.

★ **AND NEUROSTREAM'S FIX IS NOT THIS ONE; IT IS ALREADY BUILT HERE.** Their
76.3 s -> 0.8 s came from planning once per LAYER instead of once per
projection, which is queue depth and not prediction -- and `ensureExperts`
already issues one plan for the whole selection, citing them by name. What their
number does not license is a speculative CROSS-TOKEN prefetch, which is a
different mechanism with a different hit rate.

★ **THE ONE PREDICTION-FREE OVERLAP THAT IS NOT REFUTED, PRICED RATHER THAN
BUILT.** The down bank is a third of the expert bytes and is not needed until
after the gate and up matvecs and the SwiGLU, so it could be read behind them at
zero extra bytes. On the over-committed row that is ~1.0 ms of compute to hide
~1.8 ms of read per layer, a **14% ceiling** -- in a configuration that is
itself 6.5x off the resident rate, where the answer is to not be one page short.
Not built. Anyone who builds it owes the same two tables.

**★ AND WHAT IS ACTUALLY LEFT OF `ensureExperts` ON A MODEL THAT FITS IS THE
PLAN, NOT THE I/O -- WHICH ONLY SPLITTING THE COUNTER COULD SHOW.** `ExpertIO`
reports blocked wall and the I/O inside it; the difference is `planRuns`.
olmoe, n=1400, 22,480 layer-steps:

    window sweep        237 ms   **10.5 us** a layer-step
    sorted spans        180 ms     8.0 us    (sort.Slice -- reflect swaps)
    insertion sort       96 ms   **4.3 us**

★ **THE BOUND `planRuns` ALREADY HAD DOES NOT BIND FOR THE CALLER IT WAS
WRITTEN FOR.** It walks a bool window over the span the ranges BRACKET -- and a
routed selection is scattered, so the lowest expert of the gate bank and the
highest of the down bank bracket almost the whole page. On olmoe that is
`make([]bool, 8474)` and two walks of it, to find 24 runs covering 864 chunks,
once per layer per token. Sorting the spans and walking only what was asked for
is the same answer at O(the request): **304 B a plan at 256 chunks and 16,432 B
at 16,384, against 48 B at both.**

★ **THE READS ARE BYTE-IDENTICAL, WHICH IS THE CHECK THAT MATTERS.** Fully
resident: 7822 requests and 3.34 GiB before and after. Evicting at 7 frames:
**213,125 requests, 118.90 GiB, 2473 page-ins, 2466 evictions, 491,280,512
B/token -- the same digits on both binaries.**

★ **AND NO RATE IS CLAIMED FOR IT.** 43.66 -> 48.49 tok/s across those runs is
11%, and the plan is 141 ms of a 30 s run: the two do not agree, so the
difference is the box and not the change. Single unpaired samples on a box that
had a foreign Go build on it. RULE 6's terms: coordination cheaper by 2.4x,
allocation O(request) instead of O(page), reads identical, nothing regressed.

`jlm.TestPlanRunsMatchesTheWindowSweep` is bit equality against the old
algorithm over 400 randomised range sets (zero-length ranges, exactly-touching
spans, duplicates, a quarter-filled page). Against the violations:

    touching spans not merged   `[{14 15} {15 19} ...]` for `[{14 19} ...]`
    spans left unsorted         1 run for 4
    the filled bitmap ignored   `{175 179}` for `{176 179}`

`TestPlanRunsCostsWhatWasAskedForAndNotThePage` is the cost half, in BYTES
rather than allocation count -- `make([]bool, n)` is one allocation whatever n
is, so an `AllocsPerRun` gate would have passed the sweep. Against the sweep
restored it fails by name at 304 B and 16,432 B.

---

## ★ 15u. ONE CARD ENUMERATED BY TWO BACKENDS WAS TWO DEVICES AND TWO BUDGETS, AND THE FIX FOR IT NEARLY DELETED HALF A MACHINE'S VRAM.

`server.SpendableTotal` on this laptop:

    vulkan:0  NVIDIA GeForce RTX 3050 Ti  discrete  4.00 GiB
    cuda:0    NVIDIA GeForce RTX 3050 Ti  discrete  3.81 GiB
    spendable 13,371,179,008 B = 4,989,124,608 host budget + 4,294,967,296 + 4,087,087,104

**7.81 GiB of VRAM on a card that has 4.** It is AGENTS.md's Scope rule for an
integrated GPU -- *"SUBTRACT, DO NOT ADD"* -- arriving from the other side, and
a scheduler that believes it thrashes for the same reason. After:

    spendable  9,076,211,712 B = 4,989,124,608 + 4,087,087,104   (cuda:0 alone)

**★ THE IDENTITY IS THE DEVICE UUID, MEASURED ON BOTH BACKENDS, AND THEY
AGREE BYTE FOR BYTE.** `VkPhysicalDeviceIDProperties.deviceUUID` chained into
`vkGetPhysicalDeviceProperties2`, and `cuDeviceGetUuid_v2` through the existing
purego binding -- no cgo either side:

    vulkan:0  uuid:<the card's sixteen bytes>
    cuda:0    uuid:<the same sixteen bytes>
    vulkan:1  uuid:8680a6460c0000000002000000000000   Iris Xe
    vulkan:2  uuid:6d65736132352e322e382d3075627500   llvmpipe ("mesa25.2.8-0ubu")

**★ AND THE OBVIOUS KEY IS A WORSE BUG THAN THE ONE IT FIXES, WHICH IS THE PART
WORTH KEEPING.** The tree already had a dedupe -- `sameCard`, a comparison of
the driver's NAME with `cardName` stripping CUDA's ordinal and PTX target -- and
its own comment named the trade it was making: *"two distinct cards that report
one name end up sharing a pool, which under-places"*. On a box with two
identical cards that is not an under-place. It merges two 4 GiB heaps into one,
reports 4 GiB where there are 8, declines every block of the second card's
share, and says nothing -- **invisible on every box this project has, all of
which have at most one card per model.** Under-merging over-counts memory;
over-merging deletes hardware, and only the second is unrecoverable.

So the gate that matters most is the one this hardware cannot produce:
`tier.TestTwoIdenticalCardsAreNotMerged` fabricates two devices with the SAME
name and DIFFERENT UUIDs and requires both to survive. Against the name match it
reads *"two cards with one name became 1 device(s) after 1 merge(s): half the
machine's VRAM (3.00 GiB) would go unused"*.

**★ AN UNKNOWN IDENTITY MATCHES NOTHING, INCLUDING ANOTHER UNKNOWN ONE.**
`backend.Identity.Same` is `i.Known() && i.Key == o.Key`, and `UUIDIdentity`
refuses an all-zero UUID from a call that returned success. Both halves are
gated; dropping either merges every unidentifiable device on a host into one.
Metal supplies no identity at all and says so in the value -- `MTLDevice` has a
registryID and nothing specifies that MoltenVK's `deviceUUID` is derived from
it, so a Metal device stays distinct. That costs nothing: Apple hardware is
unified memory, so both entries already come OFF the host budget.

**★ THE PREFERENCE IS CUDA > METAL > VULKAN FOR ONE CARD, AND IT IS THE
MEASUREMENT RATHER THAN THE FIRST ENUMERATED.** AGENTS.md's row: `cuda:0 + cpu`
at **+10.2%** against the CPU alone, the Vulkan-enumerated iGPU at **-15%**, and
`cuda:0 + vulkan:1 + cpu` at **-14%** -- adding the Vulkan device dragged the
CUDA configuration down. `tier.choose`'s own load-time probe reads the same
shape on the SAME card, 2468 against 1368 Gmac/s. The rule it replaces kept
whichever device was reached first, which was CUDA only because `allDevices`
happens to open CUDA before Vulkan; `server.probeDevices` enumerates Vulkan
first, so there a first-wins rule keeps exactly the wrong one.

**★ AND THE DEDUPE BROKE "AN EXPLICIT REQUEST STILL WINS" ON ITS FIRST
DRAFT, WHICH THE GATE CAUGHT AND NO REVIEW WOULD HAVE.** `choose` deduped before
reading `WithAPI`, so with two views of one card the list was down to one
candidate and returned before anyone asked what the caller wanted: an explicit
Vulkan request silently got CUDA. The pin is read FIRST now, over the whole
list. AGENTS.md's Scope section is the requirement -- *"every one of those
decisions must be forcible by the caller"* -- and comparing the two backends on
one card is the experiment the 2468/1368 row came from.

**WHAT DEDUPE GOVERNS AND WHAT IT DOES NOT.** The DEFAULT set (`auto`, `all`)
and the memory arithmetic. Not `-devices vulkan:0`, which goes straight to
`backend.OpenVulkan(sel)`; not `-devices vulkan` (`WithAPI`); not `gpu:N`, which
still indexes `backend.Open`'s list as its grammar says. The duplicate is LISTED
everywhere it was listed before -- `jitllm hardware`, `jitllmd devices`,
`ListDevices` -- and carries `same_device_as` so a reader can see why it is not
in the total. Hiding hardware is its own bug.

**★ PCI BUS ID WAS NOT NEEDED AND IS NOT BUILT.** It is the documented fallback
(`cuDeviceGetPCIBusId`, `VK_EXT_pci_bus_info`) for a device that cannot supply a
UUID; both backends supply one here, and an untested fallback path is the shape
RULE 10 refuses. A device with no identity stays distinct, which is the correct
answer and not a gap.

### A wide hybrid batch keeps ONE recurrent state a block

A batched decode grows a recurrent pair per linear block (tier.growRec), and the
pair is double-buffered because ir.Validate refuses a kernel that reads the
buffer it writes. Qwen3.5-4B carries ~2.1 MiB of delta state a sequence a block,
so 128 sequences are 280 MiB a block once and 536 MiB twice; on a V100 the
batch was refused ("recurrent states do not fit"), where llama.cpp, which keeps
one copy, runs it. `recPair.shared` keeps s[0] resident and writes the device's
single `recNext`, which `kernels.CopySpan` moves home after the delta step --
still out of place for the kernel, one scratch for the device rather than one a
block. The form is decided for the WHOLE batch in LayersRows: grown block by
block, the first blocks took two halves and left the last none (refused at
14,480 of 14,809 MiB). Gate: `model.TestSharedRecurrentStateMatchesTheDoubledOne`
(tier.Config.SharedRec forces the form; tokens bit-identical to the doubled
form, Stats.RecShared asserts the form ran; without the copy back row 0 parts
at token 3).

V100, Qwen3.5-4B Q4_K_M, 6-token prompt, 128 generated, aggregate tok/s,
llama.cpp `llama-batched-bench` in the same pass:

    batch      16      32      64      128
    jitllm   757.8   973.1  1177.8  1294.3   (128 was refused)
    llama.cpp 656.3  843.6   730.0   934.5

### Figures moved out of jit/gpu/tier comments

These figures stood in source comments under `jit/gpu/tier/` until a comment
cleanup trimmed them, and were recorded nowhere else. They are kept here
verbatim, grouped by the symbol each one justified. "This box" is the laptop
(RTX 3050 Ti 4 GiB, Iris Xe, 31 GiB RAM); V100s are the 8x V100 server's.

**Multi-device placement (multi.go, `GPU.PrepLayer` and friends).**
- `order` (fastest first): the 3050 Ti runs 2468 Gmac/s through CUDA against an
  Iris Xe measured at 0.20x of jitllm's own CPU.
- Contiguous runs per device: a crossing is ~36 us (RULE 15b), so scattering
  22 blocks alternately over two cards would spend ~0.76 ms a token on
  crossings alone.
- A block already on a device is offered there first: without it a second
  State re-uploaded block 0 onto the card holding the tail, and Qwen3-32B on
  two V100s ran 45 of 64 blocks under `jitllm speed` (whose warm-up is a first
  State) where `run` placed 64, at 3.2 tok/s against 12.7.
- `Config.Sessions` (tier.go): without room reserved for n histories a second
  session found no room for its KV cache; Qwen3-32B on two V100s decoded on 22
  of 64 blocks at 3 tok/s that way, where one State runs 35.
- `SpillAfter`: fill-first put gemma-3-27b on two V100s at 56 and 6 blocks, the
  first KV growth past a page was refused on the full card, and a 512-token
  prompt demoted all 62 to the host (7 tok/s against 651).
- `planShares` over the FEWEST devices that hold the model: Mixtral on three
  V100s spread 11/11/10 read pp512 1604-1607 and tg128 78.3 against fill-first's
  two cards at 1681-1682 and 80.4, ABBA in one pass. (The gemma-3-27b 59/3 case
  that motivated shares is in docs/perf/current.md.)
- The head on another device rather than the host: gpt-oss-120b on five V100s
  filled four cards with its 36 blocks, left the fifth empty, and ran the
  201088-row projection on the host at tg128 48 tok/s against llama.cpp's
  117.6.
- `Reserve` / staging per device: under 2 MiB a device on Qwen3-30B-A3B -- the
  151936-row output projection at 594 KiB, one partial-sum buffer and two
  activation staging buffers of a few KiB.

**Admission charges the whole block (`prepLayer`'s room check).** Weights were
priced and the KV cache and norms charged unchecked: Mixtral on two V100s after
a 512-token prompt put 17 blocks on device 0 where 16 fit, over by 3.5 MB of
KV, and every token then paged two 0.9 GiB blocks through the card -- 3.5 tok/s
against 60, with the second card (and a third) holding room for the
seventeenth.

**`initKVCap` reserves the context asked for, up to devKVReserve.** Starting at
one page let placement fill a card to its last megabyte: Mixtral on two V100s
left 21 MB free of 13.5 GiB, the first growth past 256 positions was refused on
both cards, and a 512-token prompt demoted all 32 blocks to the host (9.6 tok/s
of prompt against 365 on three cards).

**`budgetFor` headroom capped at 1 GiB (open.go).** An eighth of a V100's 15.5
GiB free is 1.94 GiB; Mixtral-8x7B needs 13.6 GiB a card on two of them, and
the budget stopped 3.5 MB short of the sixteenth block per card -- two blocks
went to the host and decode fell from 80 to 10 tok/s. Measured with -vram 15e9
per card through a 512-token prompt and a 128-token decode: 14.88 GiB peak of
16.14, ~0.5 GiB beyond the charged budget.

**Pools and identity (open.go, tier.go).**
- The Iris Xe reports 20.94 of 23.26 GiB free, all of it inside the host's 31
  GiB (`planSlots`, `hostShare`, `WithHostBudget`).
- `noDuplicates`: `-devices cuda:0,cuda:0` would give two 2.24 GiB budgets on
  2.56 GiB of VRAM.
- Before the UUID dedupe, this box reported vulkan:0 4.00 GiB + cuda:0 3.81 GiB
  of one 3050 Ti: spendable 23.52 GiB = 15.71 host budget + 4.00 + 3.81
  (devidentity_test.go).

**Streamed expert banks (layer.go `streamBank`, tier.go `sharedCompact`), Qwen3-Next-80B.**
(The read/upload split and the stream-groups dose are 15q above.)
- A resident bank is 51x the bytes the token reads: 1.04 GiB a block against
  2.56 GiB for the whole token's experts across 48 blocks, on a 2 GiB card.
- The narrow read: the whole bank is 35x the bytes, 32.8 GiB a token against
  the 0.92 the token consumes.
- One transfer per plane: a block's bank is ten sheets x three matrices x three
  planes, ninety transfers averaging 0.22 MiB, and a profile put 31.68% of
  samples in runtime.cgocall, 0.49 s a token, on the driver's goroutine.
  Staging a plane's k sheets makes nine transfers of ~2.2 MiB; the host copy
  runs at ~10 GB/s off that goroutine.
- `sharedCompact`, one compact bank per (matrix, format, shape) for the device:

      per block   base 26.96 MiB + bank 19.84 = 46.81 MiB   x48 = 2.194 GiB
      shared      base 26.96 MiB x48 + ONE bank 19.84       =     1.283 GiB
                                    + 150.75 MiB recurrent  =     1.430 GiB

  against a 1.885 GiB budget. With `which` missing from the key, gate and up
  shared a buffer: TestStreamedExpertBankMatchesTheResidentOne read 0.4392
  against 1.0620 at pos 0.
- `PrepHead` under paging: the projection is 0.63 GiB of a 2.00 GiB budget
  against a 1042.8 MiB page, so taking it left one slot where two would have
  fitted.

**Paging and the device unpack (page.go, unpack.go, tier.go).**
- `submit`: each streamed block's crossing (~36 us) is against 258 ms of host
  compute per block on a paged 70B.
- Unpack staging on a paging device: losing a slot to it costs one extra
  page-in per token (78 -> 79 on a 70B at three slots, 1.3%) against a repack
  that is 69% of the token.
- Staging per TENSOR: on Llama-3.3-70B 126 MiB, against 507 MiB for a whole
  block's raw bytes and a 586 MiB slot. Capped at the widest block tensor
  because lm_head is 128256 x 8192 Q4_K, 563.7 MiB of GGUF against 126.0 --
  4.5x, to save one 0.64 s pack at load.
- `Stats.PageBytes` stopped equalling the bytes crossed once uploads went raw:
  126 MiB against 129.5 for a 70B ffn_gate, and 183.75 against 239.75 for a
  Q6_K that took the host packer.
- `resident` prices before packing: on a 30B mixture under a 4 GiB cap,
  declining a tensor after packing it cost ~450 MiB of anonymous memory and
  ~360 MiB of faults per layer. One 30B block is up to ~158 MiB, so a block
  stranded by a late decline is most of a 4 GiB card's spare room; a Prewarm
  stage is ~450 MiB a block on a 30B (hence StageLimit).
- `ReleasesHostPage` / block-keyed residents: a fully offloaded Qwen3-1.7B (28
  blocks, 1.06 GiB on the card) held peak host RSS 1.33 GiB against the
  CPU-only run's 1.14 before the host page was dropped.

**PCIe against host DRAM (pcie_test.go).** host -> device 8.14 GB/s on CUDA and
5.65 GB/s on Vulkan; CPU decode reads DRAM at ~39 GB/s achieved and the card
its VRAM at ~150. A paged VRAM cache mixing 150 GB/s hits with 8.14 GB/s misses:
hit 50% -> 15.5 GB/s, 80% -> 33, 90% -> 55, so break-even is ~85% hits;
Qwen3-30B-A3B on 4 GB is ~2.5 of 17 GiB.

**`Place` / `Unit` (place.go), Qwen3-30B-A3B.** A block is 335 MiB, of which 324
MiB is expert banks a token reads 6.25% of and 11 MiB attention every token
reads in full. Whole blocks spend 1.31 GiB to cover 8% of a token's bytes;
ranking the same budget by hotness per byte covers ~69%. Attention weights are
11.1 MiB a layer and its KV 192 KiB a position across 48 layers (8 MiB a layer
at 2K, 160 MiB at 40960), so pricing on weights alone understates attention 2x
at 2K and 15x at full context.

**`mvPays` / `mvMinBytes` (tier.go, the per-matvec crossing; 15g above).**
- The break-even: bytes/R_host >= bytes*pack/R_dev + seam, with R_host 37 GB/s,
  R_dev 153 GB/s, pack 1.4, seam ~59 us -> bytes >= 3.3 MB per crossing.
- A separate lock for the sample measured 0.9848 on Phi-3.5, whose verdict is
  "serve" and whose behaviour is meant to be identical.
- The sample is per session: model.TestHybridSecondSessionMatchesTheFirst read
  47 matvecs served in a lone session against 0 in the second, logits apart by
  6e-08 at position 0 of the first token.

### The paged KV cache: backends, the adversarial audit and prefix reuse

Moved from ROADMAP.md item 5, verbatim apart from this heading and the removal of dates;
the roadmap keeps only the item's status.

★ THE THREE BACKENDS THAT EXIST. The interface the paged cache
actually shipped is the user's, not the sketch above: `Get(cacheId int64, layer,
index int, page io.Writer) error` plus `Set`/`Drop`, keyed by page rather than
by request.

    NoStore     the default. A sealed page is dropped and recomputed if it is
                ever needed again -- which for a growing context it never is,
                so this is the zero-cost case and not a stub.
    MemStore    host RAM. What the device evicts to, and the gate that proves
                a page survives a round trip.
    FileStore   one file per page under `<dir>/kv-<cacheId>/<layer>-<index>`,
                written to a temp name and RENAMED. That is the container-of-
                zeros precedent applied to the one copy of a sequence's past: a
                page is absent or whole, never short. `TestFileStoreRoundTrip-
                ChangesNoToken` evicts 1980 pages, faults 9634 back off disk and
                gets identical tokens.

A `remote` backend is the same interface over a transport and is the one that
has to live outside core. Not built.

★★★ **AND AN 8-AGENT ADVERSARIAL AUDIT OF THE WHOLE CHANGE FOUND ELEVEN MORE
BUGS AND SIX DEAD GATES. THE PATTERN IN THEM IS ONE MISTAKE MADE
REPEATEDLY: LENGTH IS NOT RESIDENCY.** Four dimensions reviewed in parallel, each
finding then handed to a verifier told to REFUTE it; 25 survived and collapse to:

    gather()     checked `pg >= len(s.k)` -- the same conflation bytes() had.
                 A page trim had evicted in place passed the guard and was
                 sliced as nil. Migration is not the read path, so nothing had
                 faulted for it. RESIDENT now, and it refuses rather than panics.
    scatter()    grew only the LAST page and then wrote every page, so a history
                 coming home from a device across more than one page stored into
                 nil. A device block's host pages are never written while it is
                 placed, so on demotion ALL of them are nil.
    restore()    counted a page that was merely RESIDENT as a page the store
                 holds, because fault() short-circuits on a resident page. A
                 reused State restored the PREVIOUS sequence's history under a
                 new key. It drops residency before probing now.
    Reset()      moved the positions and left the pages, so seal() resumed from a
                 stale lastSealed and never offered the new sequence's early
                 pages while trim() treated that same range as evictable -- and a
                 fault then served the old sequence's bytes back. kvCache.reset().
    seal()       lastSealed only ever ROSE, so a position rollback left the page
                 the write path was about to use as an eviction candidate, and
                 the write re-created it ZEROED. It falls as well as rises now.
    seal()       discarded every store error: a store failing every Set looked
                 exactly like no store at all. Counted (KVStoreFailures).
    ForwardBatch bumps the positions inline rather than through advance(), so a
                 batched session sealed and evicted NOTHING -- SetKVStore and
                 SetKVBudget reported success and did nothing. It seals against
                 the MINIMUM row position, because a page the furthest row has
                 left behind is one a lagging row is still WRITING.
    MemStore     an unsynchronised map, when being shared is its only purpose.
    tier         ensureKVCap ran BEFORE the vision branch, so a 1024-patch image
                 drove the TEXT capacity and the growth freed g.vbs -- which is
                 assigned at one site guarded by `g.vbs == nil` inside a
                 prepLayer that early-returns for a resident block. Nothing
                 rebuilt it; the tower was declined for the rest of the process.
    tier         the regrow loop rewrote VISION blocks' caches with the text
                 model's kvDim and capacity. capped() already exempts them.
    tier         regrowKV charged the growth but never moved l.kvBytes, which is
                 what ReleaseLayers refunds -- a budget leak per doubling.

★★ **AND THE SIX DEAD GATES ARE THE MORE USEFUL HALF, BECAUSE EVERY ONE WAS
GREEN.** `TestKVCapGrowsPastEveryDoubling` reimplements the growth loop against
`kvCapFor` and NEVER CALLS `ensureKVCap` -- where the bug its own comment
describes actually lived. Restoring that bug leaves it green and fails the new
`TestEnsureKVCapReachesTheContextAsked`, which drives the function. The paired
accumulate `AttnAcc2Into` had no gate at all: the paired walk is selected by a
runtime bit and falls back silently, so a model-level equality passes whether it
is right, wrong or never called. `TestFileStoreRefusesAShortPage` asserted
`shortPage(...) != nil`, true by construction and true with the check in fault()
deleted. The linear-attention arm deferred to a test that does not exist. Eight
`t.Skip(err)` turned a broken Open into a green run. And `TestPagedKVCost` could
not produce a row at its own defaults -- depth 2048 exceeds tinyllama's context,
so both arms collapsed to the same configuration.

★★ **AND A TWELFTH BUG CAME OUT OF ASKING "IS IT ALL FIXED?" RATHER THAN OUT
OF THE AUDIT -- THE BATCHED SEAL WAS FIXED IN ONE OF ITS TWO PATHS.**
ForwardBatch was taught to seal against the minimum row position; `advance()`
was not, and `PrefillSeq` calls it to admit a NEW sequence into ONE row of a
batch. So it sealed against `s.pos`, the high-water mark ACROSS rows, while the
row just admitted sat at zero:

    with row 1 at position 0, 220 page(s) were stored -- seal ran against
    the furthest row (40), not the slowest

A page holds EVERY row's slots, so that offers the store a page the new row is
still writing into, trim then treats it as a candidate, and that row's next
write re-creates it ZEROED -- taking the other rows' history for those positions
with it. Both paths go through one `State.lowPos()` now. For a single-sequence
session bpos is one entry and that IS s.pos, so the default and every board row
are unchanged; a batch with an idle row seals nothing, which is conservative and
correct, because a retired row restarts at position 0 and page 0 really is still
mutable.

`TestBatchSealsAgainstTheSlowestRow` asserts the SEAL POINT and not the tokens,
because the tokens only diverge once an eviction and a re-fault happen to line
up -- and a gate that needs a coincidence is a gate that passes. **The audit
missed this one because no gate anywhere put a STORE on a batched session**,
which is the coverage hole rather than the bug.

★ **AND ONE GATE TURNED OUT TO BE MEASURING THE INSTRUMENT.**
`TestPagedModelOpenCloseDoesNotLeak` reads process RSS, and `-race` allocates
shadow memory in proportion to the bytes a process touches -- so inside the KV
slice, after the FileStore round trips and the 1024-token prefix prefills in the
same binary, it read the DETECTOR growing and not frames failing to come back:
1 failure in 4 at +12 MiB a cycle, against 9 runs in 9 passing when run alone
under `-race`. It skips under the race build now, using the `raceEnabled` pair
`bench/guard_{race,norace}.go` already established, and the skip says which
quantity is unmeasurable and why. The gate still runs in every ordinary build,
which is the configuration that ships. Four full `-race` passes of the slice
after that: **0 data races, 4 of 4 green.**

★ AND PREFIX REUSE IS BUILT ON TOP OF THEM, AND THE KEY IS THE
CONTENT RATHER THAN THE SESSION. A page of layer li is stored under
`<namespace>/H(tokens[0 : (n+1)*p])` -- the tokens that produced it -- so a HIT
is a proof that every token to that boundary matches, and the longest cached
prefix is found by walking boundaries until the store misses. That is LMCache's
chunk_hash, and the CUMULATIVE part is the whole of it: RoPE is applied before
the cache write, so a key carries its absolute position and a span is reusable
only as a PREFIX. Hashing a chunk in isolation would collide two identical
chunks at different offsets.

★ **THE SESSION-KEY DESIGN THAT STOOD HERE IS SUPERSEDED, NOT AMENDED.** It made
the CALLER answer "which prefix is in the store" -- it had to know what an
earlier run cached and name it exactly -- which is the wrong way round and made
the case the feature exists for unreachable: several sessions sharing a system
prompt, none of them aware of the others. `SetCacheKey` names a NAMESPACE now.

    jitllm run -kv-cache DIR model.jlm "<prompt>"

    run 1, cold        1181 tok in 5.684s    kv cache 0 of 1181 reused (0%)
    run 2, +5 tokens   1186 tok in 0.980s    kv cache 1024 of 1186 reused (86%)

**5.8x on the prompt, and run 2's prompt is LONGER** -- it solved the prefix
rather than matching a whole prompt. In-process: 48 of 48 shared positions reused
by a prompt nobody registered, tokens identical to a cold control, and an
unrelated prompt reusing nothing. Cross-process is its own gate, which
re-executes the test binary: one process writes and EXITS, a cold one reads.

★ **THE NAMESPACE IS ALSO THE TENANCY DIAL**, which is why it is a free string:
`"llama3/q4/f32"` shares every prefix, `".../tenant-42"` shares within a tenant,
`".../sess-<id>"` shares with nobody. The trade is the caller's -- a cache HIT is
observable, so a shared cache is a side channel on the prefix, and a per-user
namespace is the right answer whenever the prefix could contain something one
user must not learn about another. Tiering the two is a real design and is NOT
built.

★★ **AND WHAT THE ENGINE CHECKS VERSUS WHAT THE NAMESPACE MUST CARRY IS THE PART
THAT WAS WRONG FOR AN HOUR.** A page's BYTE COUNT is `nseq * p *
(NKVHead*HeadDim) * elem` -- four independent factors collapsed into one scalar
-- and it was the only geometry test. Three real models on this box alias:

    tinyllama-1.1b   nKV 4 x hd  64 = kvDim 256   131072 B
    Qwen2-1.5B       nKV 2 x hd 128 = kvDim 256   131072 B
    gemma-2b         nKV 1 x hd 256 = kvDim 256   131072 B

and head-major against row-major is byte-identical BY CONSTRUCTION -- kvlayout.go
says so -- so a layout flip is a pure permutation, and that default is expected
to change. The geometry is hashed into every page key now (one Write per page
BOUNDARY, not per token), which turns every one of those into a clean MISS. What
is left for the namespace is model IDENTITY: two fine-tunes of one architecture
have identical geometry AND identical config, and this format carries no content
digest. The CLI names the container file; that is a name, not a proof.

★ **AND FOUR MORE HOLES THE SAME AUDIT FOUND, ALL SILENT:**

    the f16 step-down REBUILT the cache and dropped the namespace, so
    SetCacheKey-then-SetDevice -- the most natural ordering there is -- sent
    pages to a name no later run can guess while every counter reported success

    the default session id was "session-" + a counter starting at ZERO in every
    process, and seal writes through it, so a second process's first State
    addressed the first process's pages. A per-process nonce now.

    a batched State accepted a namespace and sealed nseq-wide pages named after
    row 0's tokens alone -- readable by nobody, since PrefillCached refuses a
    batch. Refused at SetCacheKey.

    a geometry refusal FAILED THE REQUEST and nothing deleted the page, so one
    file from a differently-configured process bricked that prefix FOREVER. A
    mismatch is a MISS that drops the key and counts itself now
    (KVStoreMismatches); a genuine I/O failure still stops the request, which is
    the distinction ErrNoPage exists to draw.

★ **AND `PrefillCached` REFUSES WHEN BLOCKS ARE ON A DEVICE.** Host pages are
written by attnPrep, which runs only over the HOST block range -- a placed block
keeps its history on the card and never mirrors it -- so restoring host pages and
advancing the position left every placed block attending over positions nothing
ever wrote. Migrating the restored range onto the card is what a server wants,
since prefix reuse and offload get turned on together. NOT built; refusing is the
honest placeholder.

## ★ 15v. THE DEVICE ANSWERED FOR A REUSED FRAME: A LONE MATVEC IS KEYED ON THE POINTER, AND A PAGE'S POINTER IS NOT AN IDENTITY.

`model.TestHybridDeviceErrorIsAmplification` failed about one run in ten on the
RTX 3050 Ti, with `WithDeviceTune(TuneOff)` already pinned: the shallow arm read
NMSE 2.3e-02 to 2.3e-01 where it reads 2.402e-04, which is larger than the deep
arm and fails the ordering. It reproduced at 7e2203f, so it was not the
recurrent-pool work.

**Isolated by four arms of one probe, 80 forwards each, alternating the 48-
and 4-block fixtures and hashing every arm's logits:**

    one tier for both models, each closed after its arm   17 distinct hashes
    a fresh tier per model                                1 per depth, 80/80
    one tier, every model kept open to the end            1 per depth, 80/80
    one tier, ReleaseLayers(0, 64) between models         still 16 distinct

The host arm hashed identically in every run, so the device was answering
differently. The variable is not the tier being shared and not its placed
block (released, it still varied): it is a CLOSED model's memory, under a tier
that outlived it.

**The mechanism.** `devTier.MatVec` -- the single-matvec path a State's JIT
offers its non-packed weights to -- uploads a weight once and keys it on
`&w[0]` and the shape (`resKey.p`), as `nn.Device` documents. `State.mv` offered
every float weight, and a mixture's router is an F32 matrix in the block's PAGE.
`mvPays` serves the first 64 offers of a session while it samples, so the host
blocks' routers went to the device. Pages live in frames the pager reuses, and
every block's router sits at the same offset of its page with the same shape: a
new model's frame mapped over a closed one's address, or a page evicted and its
frame refilled, presents the device a pointer it already holds and bytes it
has never seen, and it answers from the old copy. The routing is wrong, and so
is every token after it.

**The deterministic reproducer is the same model under paging, no second model
needed.** The 8-block mixture with a one-byte page budget (one frame) and the
device attached for lone matvecs only (`SetDeviceLayers(g, 0)`):

    budget 1, router offered           NMSE 9.47e-01  1.07e+00  7.96e-01  4.90e-01
    no budget                          NMSE 4.79e-04  1.87e-04  3.55e-04  1.26e-04
    budget 1, page tensors on the host NMSE 4.79e-04  1.87e-04  3.55e-04  1.26e-04

So a server or `jitllm run` with a device attached to an over-committed
mixture routed the host blocks with another block's router during every
session's first 64 matvecs, and for good once `mvPays` said yes.

**The fix is to tell the device, not to stop offering.** A first fix had
`State.mv` offer only the dense region (`jlm.File.InDense`) and run every page
tensor on the host. It was correct and it gave up per-matvec service for every
float weight in a page, even on a model that fits, where nothing is ever
refilled. What is built instead closes the contract from both sides:

    jlm.File.OnRecycle   the pager names every buffer whose bytes stop being
                         the ones a reader may have copied: a frame handed to a
                         page (fresh or an evicted page's, in claim), a frame
                         given back (release: trimFree, ReleasePages, a budget
                         cut), and the dense region and every frame at Close.
                         The callback runs after f.mu is dropped and before the
                         page-in reads anything into the frame.
    nn.CopyForgetter     Forget(p, n): the device drops every copy keyed in
                         [p, p+n) -- tier.GPU drops the lone matvec's upload
                         (freed and refunded, Stats.Forgotten), a prewarm stage,
                         an arena pack (kept mapped while a device imports it,
                         kernels.Arena.Unimport), and spoils a pack in flight.
    Model.forgetCopies   the container's callback, fanned out to every device
                         a State attached (Model.watch).

A page-in can start inside the tier's own lock -- an upload reading an expert
sheet -- so `Forget` never waits on the device: it applies the range when the
lock is free and otherwise queues it, and every lookup by address
(`resident`, `blockBytes`, `prewarm`) applies the queue first. `State.mv`
offers a page's weight again whenever the device is one the model tells
(`State.forgets`), and only the dense region to any other device.

`model.TestPagedWeightsAreNeverServedStale` is the gate: the one-frame arm must
EQUAL the unbudgeted arm with both serving the same 36 matvecs on the device and
31 copies forgotten as frames were refilled (the selection check that page
weights reached the device at all). With `Forget` removed from
`Model.forgetCopies` it fails at **NMSE 9.354e-01**, the original reading.
`tier.TestForgetFreesTheCopiesInItsRange` holds the ledger (the copy's buffers
freed, its charge refunded, the neighbouring weight kept, the queued path
applied by the next lookup), `jlm.TestRecycleNamesEveryFrameItRefills` the
pager (claim, reuse, release and Close named, never under the lock), and
`kernels.TestArenaForgetUnkeysWhatItCovers` the arena. The amplification gate
reads 2.402e-04 / 1.559e-02 again, the deep arm's routers back on the device,
and the same over 20 runs on the 3050 Ti and on a V100.

**A tier is one model, and now it says so.** The per-arm tier above hid a
second hazard rather than fixing it: a tier's placed blocks are `g.own[li]` and
`layers[li]`, each
device holds one head and one scratch built from the first plan it saw, and a
second model's block li offered beside the first's was taken for it --
`GPU.PrepLayer` sends a held block to its device, which shares what is there.
Measured with the check removed: the second model "placed" 4 of 4 blocks and
the head and read **NMSE 6.590e+00** against its own host run. `LayerPlan.Model`
and `Head.Model` carry the model's identity (`Model.id`); the tier takes it on
the first offer and refuses another model's blocks, head and prewarm by name
(`LastErr`, a false return) while any block or head of the first is placed.
`Model.Close` releases its blocks, heads and the scratch, KV pool and argmax
built from its plan (`nn.ModelOwner.ReleaseModel`), after which the next model
is admitted onto a device that has seen nothing. `model.TestATierServesOneModel`
(refused, 5.966e-06 against its host run; admitted after the first closes) and
`tier.TestATierRefusesASecondModel` (every buffer returned) are the gates. The
amplification gate still opens a tier per arm, which is now a choice rather than
the only correct arrangement.

## ★ 15w. A STREAMED CONTAINER ON A UNIFIED DEVICE RAN ON THE HOST: THE ARENA WAS ASKED TO REPACK BYTES ALREADY IN THE DEVICE LAYOUT.

`devTier.resident` tried the packed arena (`importing()`: a unified heap, a
host-import alignment and an arena budget) before the branch for a container
weight, whose `Packed` spans are already what the kernels read. The arena packs
GGUF bytes, and a container's payload span read as GGUF is short for every
quantized format, so `kernels.PackWeights` refused it and the block was
declined. `cmd/jitllm` gives the arena a budget exactly when a placement
streams, so on the M4's Metal and on the laptop's Iris Xe through Vulkan a
streamed placement placed **0 of 16** blocks of Llama-3.2-1B and ran on the
host, reporting `arena Q4_K: kernels: PackWeights: need 2359296 bytes, have
2097152`.

The arena branch now runs only for a weight with no container form; a container
weight is copied, which also lets the pager reuse its frame (a wrapper over a
frame would read the next block's bytes, 15v). `blockBytes` gave every weight
on an importing device the arena's headroom as a discount, so a container
block was priced at **0 bytes** against a 331,776-byte upload; the discount now
applies only to weights the arena would hold.

    gate                                                    fixed         order restored
    tier.TestAContainerWeightIsNotRepackedOnAUnifiedDevice  device bytes  declined, "need 36864,
                                                            = the spans   have 32768"
    tier.TestAContainerBlockIsPricedInFullOnAUnifiedDevice  priced in     priced at 0
                                                            full          (discount restored)
    model.TestStreamedContainerOnAUnifiedDevice  M4 Metal   16/16 streamed, 0 of 16, the arena
                                                 Iris Xe    first token   error above
                                                            as the host

The model gate skips by name where no device shares the host's memory (CUDA,
the V100 box); the M4 and the laptop's integrated GPU are where it runs.

## ★ 15x. THE BATCHED PROMPT'S SCRATCH WAS ASKED OF A CARD THE BLOCKS HAD FILLED: IT IS CHARGED AND RESERVED AT PLACEMENT NOW, AND SO IS THE DRIVER'S PAGE ROUNDING.

The block scratches -- decode's, the batched prompt chunk's, the ragged step's
-- were built at the first token that needed them and charged to nothing. So
placement filled the budget with blocks, and a 512-row chunk then asked the
card for hundreds of megabytes it did not have: on the laptop's RTX 3050 Ti
gemma-2-2b's chunk was refused by `cuMemAlloc` and the prompt went a row at a
time (0 folds, `TestPrefillFoldsTheHead`), and a server step of
`model.MaxStepRows` rows on a full card was an error.

`jit/gpu/tier/scratch.go` charges them and reserves them.

**Charged by measurement.** Every backend counts the bytes its buffers hold
(`backend.AllocCounter`), and a window around every build or free of a scratch
-- `initScratch`, `prepBatch`, `prepRagged`, the staging growers, the paged
prefill planes, the flash partials, the grouped mixture and latent buffers --
charges or refunds the difference, less whatever `charge`/`refund` moved
inside it (a page or a weight already priced). Pricing each buffer by hand is
the walk `freeScratch` stopped trusting. A device left with no block and no
head drops its scratches and staging, so a release still returns every byte
(`tier.TestTheScratchIsChargedAndReturned`).

**Reserved at placement.** The first text block, before its own weights,
builds decode's scratch, the prompt chunk's at `Config.PromptRows` (default
`nn.MaxDevicePrefillChunk`, halved until it fits beside that block) and, with
`Config.StepRows` (the server's step loop sets `model.MaxStepRows`), the
ragged step's with its per-row logits sized from `nn.LayerPlan.Vocab` -- the
head arrives after the blocks. Each takes its paged attention plan at the
deepest history the capacity reaches, where the partials are widest. A width
built later that would pass the budget is refused and the chunk runs at the
reserved width (`batchFor`). `Config.NoScratchReserve` is the other arm.

★ **AND RESERVING IT SHOWED THE PREFILL PLANES WERE SIZED FOR A HISTORY THE
CAPACITY CANNOT HAVE.** The staged prefill's pass form holds rows x heads x the
4096-key pass width -- 537 MB on Llama-3.2-1B at 512 rows -- and was built for
every capacity, while a context of 716 positions never passes the pass width
and its widest span needs a quarter of it. A capacity under the pass width now
builds the widest single pass it reaches and no pass form: Llama-3.2-1B's
512-row scratch went 776,954,740 -> 231,695,220 bytes on the 3050 Ti, and
Qwen3.5-0.8B's 144 MB at 64 positions.

    Llama-3.2-1B-Instruct Q4_K_M, the card filled by a second context to the
    model's own budget plus 64 MiB
    (model.TestAPromptFitsBesideAFullCard / TestAStepAtMaxRowsDoesNotOOMAFullCard)

                          reserved                         without the reservation
    prompt  cuda:0        16 of 16, head on the host,      16 of 16, 2 splits,
                          231,695,220 B for 512 rows,      cuMemAlloc out of memory
                          0 splits
            vulkan:0      16 of 16, 247,926,644 B, 0       16 of 16, 8 splits,
                                                           VkResult -2
            V100          16 of 16, 134,134,556 B, 0       16 of 16, 8 splits, OOM
    step    cuda:0        16 of 16, joint, 508,771,996 B   the step refused:
            vulkan:0      16 of 16, joint, 525,003,420 B   out of memory
            V100          16 of 16, joint, 378,268,316 B   out of memory

The step arms place both sessions on the open card and fill it afterwards,
with each session's history reserved first: what is left to ask for at the
step is the step's scratch.

★ **AND THE CARD SPENDS MORE THAN ITS BUFFERS ASK FOR, BY 252 MB ON gemma-2-2b.**
With every buffer charged, `Stats.Allocated` equalled `BudgetUsed` to the byte
(2,177,884,540) while the card's free memory fell by 252,486,276 more: driver
page rounding. Measured on the 3050 Ti, 32 buffers a size:

               4 KiB   100 KiB   1 MiB   1 MiB+4 KiB   2 MiB+1   10.6 MiB
    CUDA      64 KiB   128 KiB   1 MiB   2 MiB         4 MiB     12 MiB
    Vulkan    64 KiB   128 KiB   1 MiB   1.06 MiB      2.06 MiB  10.63 MiB

gemma's 2304-wide matrices are not 2 MiB multiples and Llama's 2048-wide ones
nearly are (2.8 MB of rounding against 252). The headroom (0.29 GiB) had been
absorbing it, so the prompt's KV pages, granted by the budget, were refused by
the card. `AllocCounter.Footprint` counts the rounded bytes and
`devTier.settleRounding` charges the difference after every charge, refund and
scratch window and before every room check; a block is admitted and paged in
at its buffers' footprint (`FootprintOf`), or the rounding pushes its last
tensors over the budget mid-upload (`tier.TestTheDriversRoundingIsCharged`).

**So gemma-2-2b's fold on the laptop's cuda:0 is a card too small, by bytes:**
the 26 blocks, the 512-row scratch and the rounding charge 1,820,459,008 of a
2,209,464,320 budget (2.35 GiB free less 0.29 GiB of headroom), and the head
needs 488,448,000 more -- 99,442,688 bytes short. The gate skips there naming
those bytes. The same card through Vulkan reports 4.00 GiB and folds; the M4
and the V100 fold. The `s.emb == nil` condition on the fold was read as the
cause and is not: `State.emb` is set only inside `Embed` (`embedstate.go`),
and the tied gemma head folds wherever its bytes fit.

★ **FOUR THINGS THE FULL CARD FOUND, EACH ITS OWN GATE:**

    regrow          the staging grew by freeing the old buffer first, and a
                    refused grow left it nil under every launch built against
                    it -- a Vulkan panic in the refused chunk's narrower
                    fallback. The old size comes back.
                    tier.TestARefusedGrowKeepsTheOldBuffer (nil, 0 floats
                    against the restore removed)
    compactViaHost  a KV layer grown through the host freed its pair, and a
                    refused new pair left the layer bufferless for the next
                    grow to read a nil buffer (a panic). The history goes back
                    in at its old size.
                    tier.TestAKVGrowTheCardRefusesKeepsTheHistory
    initScratch     every refusal part-way through a build was a plain
                    `return nil` that kept what it had allocated. It frees it.
    unified share   on darwin `sched.MemBudget` is 0, so a Metal device's
                    pool fell back to a 2 GiB default, and once the scratch was
                    charged gemma-2-2b's head no longer fitted beside its
                    blocks on the M4. An unknown host takes the unified
                    device's own figure (recommendedMaxWorkingSetSize),
                    quartered as a known host is.
                    tier.TestAnUnknownHostTakesTheUnifiedDevicesOwnFigure

## ★ 15y. CHARGING THE ROUNDING LEFT TWO READERS AT THE BYTES: THE SLOT COUNT AND THE LEDGER'S OWN REPORT.

15x put the driver's page rounding into the budget and admitted a block and a
page-in at its footprint. Two gates went red on the laptop at that commit and
not at its parent, and neither was a leak.

**The slot count divided by the bytes.** `slots()` took `widest` from
`pageSizeWhy`, the packed bytes, while `pageIn` asks for bytes plus rounding,
and `pageBytes` left a resident page's rounding in the permanent part. On a
fixture of small tensors and a card that hands out 64 KiB pages those are not
close: `model.TestDevicePagingSurvivesTheHostPager`'s dense block is 4,176
bytes asked for and 1,376,228 at footprint (twenty-one buffers, each a page:
seven matrices' two planes and the word each one's empty third plane gets,
below), the mixture's 7,082,064 and 8,650,712. The gate keeps the tightest
budget `Stats.Slots` says leaves a slot -- it read 110 -- and every page-in on
it was refused, so the blocks ran on the host (0 block-runs of 64). Both sides
are at the footprint now: the widest block, and `pageBytes`, which gives a
page's rounding back when it goes. The gate reads 1 slot, 64 page-ins, 64
block-runs on cuda:0 and on the M4's Metal.

★ **AND ON THE M4 THE SAME GATE'S DENSE ARM HAD BEEN SKIPPING.** Metal does
not round, so the slot count was right there; the gate's search was not. It
tried flat fractions of the placed card down to 1/33, and once the scratch was
charged the card is 3,311,472 bytes of which the eight blocks are 33 KB: every
fraction leaves either all the blocks or none ("no budget down to 1/33 ...
leaves a slot", SKIP, on main). It bisects for the tightest budget that leaves
one slot now, and refuses rather than skips when the placed budget leaves none.

★ **AND THE FOOTPRINT ITSELF MISSED THE EMPTY PLANE.** The upload hands the
kernel a word for a plane a format does not have (Q4_0's and Q8_0's scales;
`resident`, `allocBytes`), and `planesFootprint` priced it at zero -- a page
short per empty plane, so a page-in sized by it ran out of room on its last
tensor. Found by the tier gate written for the slots,
`tier.TestASlotIsCountedAtTheFootprint`: every budget a 64 KiB-page fake
reports a slot on must run a token's every block on the device. It fails
against the slot priced at bytes ("reports 2 slot(s) and the token was
refused") and against the empty plane at zero ("reports 4 slot(s) ... Q4_0
1024x256 needs 147456").

**The ledger reported a settlement, not the card.** `Bytes()` and `Stats` read
`used` without settling, and `PrepHead` allocated the argmax's 4-byte word
outside any charge, so its 65,532 bytes of rounding entered the ledger at the
next charge -- the first KV growth of the concurrent gate's sessions. Measured
on cuda:0 (tinyllama, four sessions):

                     placement      after every State closed
    Allocated        710,729,720    728,031,224   (+17,301,504)
    KVPoolBytes        5,789,696     23,091,200   (+17,301,504)
    ScratchBytes      86,840,308     86,840,308
    rounding charged  55,844,876     55,910,408   (+65,532)
    the card's        55,910,408     55,910,408
      Footprint-Allocated

The buffers grew by exactly the free pool and the card's rounding did not move
at all; only the ledger's reading of it did. `Bytes()` and `Stats` settle
before they report, the argmax word and the folded head's row offset are
charged as scratch, and `model.TestConcurrentMultiDeviceDoesNotLeak` holds the
two counts the budget is made of apart: the buffers' bytes (`Allocated`) and
the tier's charges (`BudgetUsed - RoundingBytes`) may each grow by the free
pool and nothing else. Rounding is left out because it follows the live
buffers, and the pool's own rounding moves with its layout where
`KVPoolBytes` does not. Against a KV pair a grow does not free it fails on the
buffers (745,959,992 against 711,356,984 + 17,301,504); against a session that
takes a charge and never refunds it, on the charges.

★ **AND A THIRD GATE WENT FROM GREEN TO A SKIP AT THE SAME COMMIT, WHICH NO
RED LINE SHOWED.** `model.TestMoEDeviceReadsOnlyTheBanksItTakes` measures what
one and two blocks cost on an open card and opens a card at each figure. The
prompt's scratch is halved until it fits beside the first block, so a card
opened at the measured budget reserved a narrower one than the card it was
measured on, and the difference admitted more blocks:

                  budget for 1      budget for 2
    f524704f^     7,089,264 B: 1    14,176,288 B: 2
    f524704f     40,578,976 B: 3    49,962,824 B: 4   -> skip
    unreserved   14,954,400 B: 1    24,338,248 B: 2

Both sides open with `Config.NoScratchReserve` now -- the gate asks whether a
refused block's bank is read, not about the scratch -- and it reads 25,712
chunks for one block against 50,288 for two, and 197,744 either way against a
bank read at the offer.

## ★ 15z. A WINDOWED LAYER HELD ITS WHOLE CONTEXT ON THE HOST AND ON EVERY DEVICE. IT HOLDS ITS WINDOW AND 64 POSITIONS NOW.

A sliding (or chunked) layer reads the last W keys and was given a page per
page of positions for the whole sequence: `kvPages.grow` allocated and nothing
released, and the device pool appended ids the same way. Memory grew with the
context where the paging principle says it is bounded by what attention reads,
and synth-exaone4's window of 4 (a page of 32) put a fresh page inside the
allocation gate's window, which skipped that host arm by name
(`hostAllocKnown`, deleted).

    rule        a page wholly below the window start of (the slowest live
                row's position - nn.KVWindowSlack) is released; 64 positions
                of slack is what a speculation rewind may go back
    host        kvPages.release, called from State.advance after the seal:
                the buffers go to a per-layer spare (bounded by the window's
                turnover) and the next page grows out of it -- zero
                allocations a token, where it was a page every 32 tokens
    device      devTier.windowPages, before each call: the id becomes 0, the
                table entry the dummy, the id is fenced; a page the call writes
                that was released (a row restarting) takes a fresh id, and one
                it would READ is refused by name

★ **THE PREFIX CACHE IS WHAT MADE IT MORE THAN A RELEASE.** Pages are named by
the tokens that produced them, so a page stored as history must BE history: a
device that released a page hands zeros back for it, and storing those under
a content key gives a later, shorter prefix another session's zeros as its
window. So the host trusts what a device returns only from the window start of
(its last position - 1 - KVWindowSlack) (`kvPages.genuine`), never stores a
page starting below that, and a windowed layer is no longer walked from page 0
on a restore: the full layers decide the coverage and `restoreWin` faults the
window of the restored length. That is vLLM's sliding-window prefix rule: a hit
needs the blocks inside the window, not the whole prefix.

    synth-exaone4, f32, 700 shared positions of 703 and 705, MemStore
    host      688 of 700 restored, tokens identical to an uncached run
    cuda      688 restored, identical      vulkan  688, identical
    metal     688 restored, identical (M4)
    violation (a windowed layer walked from zero, as a full one is):
              host 512 restored, cuda 0, vulkan 0, metal 0

The host keeps every page it ever sealed in the store, because it seals before
it releases; a device-placed windowed layer stores only what its window held
when the prompt was pulled down. A shared prefix whose window the device had
already released restores nothing for that layer and the prompt prefills, as
it would in vLLM.

★ **WHAT THE GATES READ** (`model.TestWindowedLayersHoldTheirWindow`, 1528
positions decoded one at a time, window 4, host page 32, device page 256):

    fixture          host peak    control kept   device released (want)
    synth-exaone4    4 pages      48             15 (15) cuda, vulkan, metal
    synth-cohere2    4            48             15 (15)
    synth-llama4     3 (chunked)  48             15 (15)
    synth-gptoss     4            48              5 (5)

Every step's logits bit-identical to a host control that keeps every page, and
on each device to a tier with `Config.KVKeepWindow` (the split tuner off in
both). The violations release the page the window starts in: on the host the
read finds no page ("no such kv page") or the logits move at step 1; on the
device (`TestWindowReleaseGateDiscriminates`, jitllmfault) the logits move at
step 259 on cuda, vulkan and metal. `TestDecodeDoesNotAllocate` holds all four
fixtures at 0 engine allocations on the host and on cuda, vulkan and metal;
with the spare taken out the host arm of synth-exaone4 reads 12 over 64 tokens.

## ★ 16a. AN INTEGRATED GPU HOLDING NO BLOCK COST THE HOST A QUARTER OF ITS BUDGET. THE HOST LOSES WHAT THE DEVICE HOLDS NOW, AND IT FOLLOWS PLACEMENT.

A bare `-devices gpu` or `vulkan` opens every GPU, so on the laptop it opens
the Iris Xe beside the RTX 3050 Ti. `tier.GPU.HostReserved` was the unified
devices' limits capped at their pool -- by default a quarter of the host
budget -- and every caller took it off the page budget once, at load. The
Iris Xe that held nothing still cost the pager 1.62 GiB.

    HostReserved   the host pool's used bytes (blocks, KV pages, scratch,
                   the driver's rounding: everything charge/settleScratch/
                   settleRounding put there) plus the arena's limit; the pool's
                   limit stays the device's ceiling and is no charge on the host
    page budget    Model.SetPageBudget takes the caller's share BEFORE the
                   device's holding and subtracts it itself (hostheld.go);
                   cmd/jitllm's setPageBudget and the server's hostWeightShare
                   no longer subtract it, and the desktop app's rebudget, which
                   never did, now gets it
    moves          State.followHost re-divides the page budget and the expert
                   budget after SetDeviceLayers, after SetGPULayers and at the
                   end of every entry point (kvCheck), when the holding changed
                   -- growth, relocation, a demotion and a device's KV growing
                   all land there; unchanged it is one read

★ **THE GATES, EACH RUN AGAINST ITS VIOLATION:**

    tier.TestHostLosesOnlyWhatAUnifiedDeviceHolds   fake iGPU, 6.20 GiB ceiling
      held by the host     0 with no block; 381704, 728840, 1075976, 1423112
                           bytes with 1..4 blocks; 728840 after two come home
      old HostReserved     6.20 GiB with no block placed: FAIL
    model.TestPageBudgetLosesOnlyWhatAUnifiedDeviceHolds   fake, 32 MiB a block
      page budget          1073741824 attached with none, 973078528 with 3,
                           1040187392 with 1, the same with a second session
                           attached and closed, 1073741824 detached
      followHost off       the seam up to 3 leaves 1073741824: FAIL
    model.TestPageBudgetFollowsARealIntegratedGPU   Iris Xe, stories15M q8_0
      held, page budget    0.75 MiB attached (the matvec scratch), 45.94 MiB
                           with all 6 blocks, 30.12 MiB with the seam back at 0
                           -- the budget 8 GiB less each, to the byte
      old HostReserved     reads 2147483648 against 786432 held: FAIL

On the laptop, `-devices gpu` on gemma-3-4b-it Q4_K_M: the Iris Xe holds 1.71
GiB of its 2.43 GiB ceiling and the host works to 8.00 GiB of 9.71, where it
lost the whole 2.43 before.

★ **WHAT THE IGPU KEEPS WITH NO BLOCK.** After the seam comes home, and after
the State detaches, the tier still charges ~31 MiB on the Iris Xe for
stories15M: 17.9 MiB of block scratch (`Stats.ScratchBytes`) and the output
projection, both kept for the next session. The host is charged for them
because the device does hold them; giving them back when no session holds a
block is open.

★ **AUTO PLACEMENT PUTS BLOCKS ON THE IGPU WHILE THE CARD HAS ROOM.** Not
changed here, recorded. `PlanBlocks` takes the fewest devices, fastest first,
that hold the model with a tenth to spare, and spreads it over them in
proportion to what each has free -- so once the card alone falls short, the
iGPU gets its proportional share, not the remainder:

    fake, 8 blocks offered     card room 5, iGPU room 20  ->  card 2, iGPU 6
                               card room 9                ->  card 8
                               card room 7, iGPU room 3   ->  card 6, iGPU 2
                               card room 3, iGPU room 3   ->  3, 3, host 2 (fill-first)
    laptop, gemma-3-4b-it Q4_K_M, JITLLM_GPU_TUNE=0
      -devices cuda:0          20 of 34 blocks + the head on the RTX, 14 host
      -devices gpu             15 + the head on the RTX, 19 on the Iris Xe

Scope records the iGPU making a placement slower. Whether the card should
fill before the iGPU takes anything is a placement-policy change with a
paired measurement owed (RULE 2), and open.

**The quarter itself was the ceiling's bug, not only its charge.** Once the
host lost only what the device held, the pool's limit (`hostShare`) stayed a
quarter of the host budget, and on the M4 that kept the default `jitllm run`
off the GPU: with sched.MemBudget at 7.68 GiB the Metal budget was 1.92 GiB,
and Ministral-3-3B Q4_K_M (2.26 GiB with the head) ran 21 of 26 blocks on
Metal, the rest and the head on the host. `hostShare` is now the host budget
less budgetFor's headroom (an eighth, 256 MiB to 1 GiB), which the host keeps
for its pager and scratch when the device is full
(`TestUnifiedDeviceGetsTheSharedPool`; the quarter fails it). M4, Metal,
decode 128 tokens, 10 interleaved runs each, tok/s median (min-max):

    -maxmem 7680M  quarter    39.88 (35.92-40.25)   21/26 blocks on Metal
                   shared     45.86 (42.07-46.04)   26/26 + the head
    default        quarter    45.84 (44.77-46.17)   MemBudget 11.26 GiB that
                   shared     45.87 (40.44-46.07)   run: the quarter fit too
    Mistral-7B Q4_K_M, default: quarter 22/32 blocks, 12.46 tok/s;
                   shared 32/32, 23.80-23.86 tok/s
    gpt-oss-20b under -maxmem 6G: 10 of 24 blocks on Metal, 14 paged on the
                   host (12 page-ins a token), the same opening answer

## ★ 16b. THE SERVER SERIALISED DEVICES BY THE SPEC'S TEXT, SO `cuda` AND `cuda:1` RAN ON CARD 1 AT ONCE. ITS GATES ARE KEYED ON THE PHYSICAL DEVICE NOW.

server/ holds one gate per device so two sessions never share a card's one
scratch set. The gate was keyed on the -devices entry as written: once a bare
`cuda` meant every card, a model loaded with `cuda` and another with `cuda:1`
took different gates for the same card 1, and `cuda:0` beside `vulkan:0`
(one card under two APIs) never serialised either.

    key        tier.GPU.Hardware: the identity the tier already dedupes on
               (the device UUID CUDA and Vulkan both report), or the backend
               and ordinal where the backend reports none ("vulkan:3")
    gates      LoadedModel.gateKeys, one per device the tier opened; a
               generate takes all of them sorted, as before (no deadlock)
    lookups    GetDeviceQueue and ListSessions resolve a written id through
               the aliases each load records (ref and -devices name -> key);
               ListDevices reads a device's queue under its PhysicalID

    server.TestOverlappingSpecsShareTheCardTheyShare   keys of a 3-card host
      cuda / cuda:1 wait, cuda:1 / cuda wait, cuda:0 / cuda:2 and
      cuda:1 / cuda:2 run at once
      keyed on the text    "cuda and cuda:1 ran on card 1 at once": FAIL
    server.TestRealOverlappingSpecsShareAGate   laptop, LoadModel, stories15M
      cuda, cuda:0 and vulkan:0 -> uuid:ef8b0a84..., the Iris Xe vulkan:1 ->
      uuid:8680a646...; the first three wait for each other, the iGPU does not
      keyed on the text    gates [cuda cpu] and [cuda:0 cpu] ran at once: FAIL
    tier.TestHardwareIsOneKeyPerPhysicalDevice   the fake host of devspec_test:
      cuda's three keys distinct, cuda:1's among them, vulkan:1 (card 1 again)
      the same key, the iGPU its own, the UUID-less llvmpipe "vulkan:3"

## ★ 16c. KIMI-K3 STREAMED ON EIGHT V100s: THE "UPLOAD" WAS A SERIAL DISK READ, A FAULTING MEMCPY AND A PAGEABLE TRANSFER, AND THE NEXT BLOCK'S ROUTER NAMES 60% OF ITS EXPERTS.

`docs/perf/current.md` ("Kimi-K3 2.78T from disk on the V100 box") left the
streamed run at 17.27 s a token with 641.5 of 649.6 s of block compute filed
under `stream cost: upload`, and ~1.4 GiB/s of apparent PCIe. That line was one
timer around three different things. The measurement that splits it, before
anything was built:

- **Instrument.** `tier.Stats.TStreamSheet`, `TStreamCopy`, `TStreamH2D` and
  `StreamBytes` split `streamBank.put` into finding each routed sheet in the
  host pager, gathering it into the staging buffer and the host-to-device
  copy. `Config.StreamProbe` (`JITLLM_GPU_PROBE`) runs, at every streamed
  block's suspension, the NEXT block's router over THIS block's normed row
  through that block's own route launches and scores the top-k against the
  selection the next block then makes. `Config.StreamSelLog`
  (`JITLLM_GPU_SELLOG`) writes every streamed block's selection to a file.
- **Run.** The V100 box, `Kimi-K3-q8.jlm`, `-devices cuda`,
  `JITLLM_GPU_STREAM=1`, greedy, `-n 32`, "The capital of France is", default
  host budget (393 GiB), 93/93 blocks on six cards. Prompt 1m29.9s, decode 32
  tokens in 9m4.8s (17.0 s/token), block compute 629.9 s, of which the fill is
  621.0 s over 3404 fills (92 mixture blocks x 37 steps; the prompt also fills
  per token).

| part of one fill (268 MiB: 16 experts x 16.73 MiB) | total | per fill |
|---|---|---|
| sheet lookup (`Sheet` -> `jlm.File.Hold` of the expert page: the disk read, one page at a time) | 173.35 s | 51 ms |
| gather into the per-block staging buffer (a memcpy) | 326.67 s | 96 ms |
| `cuMemcpyHtoD` from that pageable buffer, 890.06 GiB | 120.55 s | 35 ms (7.38 GiB/s) |
| drain before the selection is read home | 6.16 s | 1.8 ms |
| everything else in block compute (the kernels, readbacks) | ~9 s | ~2.6 ms |

★ **The reads were serial, and the read timer said 0.06 s because the reads
were not where it looked.** For a bank in expert pages
`Model.pageInSelected` returns at once (the sheets are reached through
`Sheet`), so `streamBank.fill`'s `sel2` read nothing and every expert page was
faulted in by `Sheet`'s `Hold`, one at a time, inside the upload loop. The
container carries `StreamGroups` 1, so `PrefetchExperts`' parallel read never
ran either. Decode read 212.19 GiB (7.12 GB a token) this way.

★ **The gather is the biggest term and it is not memory bandwidth.** 268 MiB
in 96 ms is 2.7 GiB/s for a memcpy. Each of the 92 streamed blocks owns nine
staging buffers grown to 268 MiB together: 24 GiB of Go heap touched once a
token each. The run took 153.9 M minor faults (~587 GiB of 4 KiB pages) and
1206 s of system time against 386 s of user time: the staging is being given
back and faulted in again. One staging set per device (fills on a device are
sequential), or no staging at all for sheets of megabytes, removes it.

★ **The transfer is a pageable copy at 7.38 GiB/s**, under the ~11-12 GiB/s a
pinned buffer reaches on PCIe 3.0 x16, and synchronous: nothing overlaps it.

**Expert reuse** (`JITLLM_GPU_SELLOG`, the 32 decode tokens scored, each
block's selection against what came before it):

| predictor / cache | hit |
|---|---|
| previous token's selection (16 of 896) | 33.4% |
| a per-block device LRU of 24 experts | 40.8% |
| 32 | 49.0% |
| 48 | 57.9% |
| 64 | 63.0% |
| 128 | 70.3% |
| seen anywhere earlier in the run (an unbounded host cache) | 72.4% |

The previous-token figure is the same 34% olmoe and Qwen3-30B-A3B gave (15t):
a property of routing, now on a 896-expert bank. The spare VRAM is small
against the bank: the two empty cards and the headroom on the rest hold about
36 GiB, ~2200 expert pages, ~24 a block -- a 41% hit on the transfer, and
nothing on the disk read, which the host frames already cache at the 72%.

**★ Cross-layer prediction** (`JITLLM_GPU_PROBE`, every prompt and decode
step, pairs on the same card): block N+1's router over block N's normed FFN
row names **25,244 of 41,712 routed experts, 60.5%** (chance 1.8%). This is
the predictor available AT block N's suspension, a whole block ahead of when
N+1 needs the sheets; it is not 15t's cross-token reuse, and unlike that one
it costs no extra bytes per hit beyond the misses it fetches (16 predicted, 16
used).

**What the attribution says to build, in order.** The disk read and the
transfer are the token; the kernels are ~2.6 ms of a 182 ms fill. (1) Read the
selection's pages in parallel and overlap them with the transfer; (2) drop the
faulting gather; (3) pinned, asynchronous transfers; then the cross-layer
prefetch, which can start block N+1's reads and transfers a block early with
60% of them right. A device expert cache is the smallest lever: 41% of the
transfer at the VRAM there is.

### 16c-1. What was built on that attribution, and what each step measured.

Same box, same container, same prompt, `-n 32`, one run per row (each run is
three to five minutes and RULE 2's twenty interleaved rounds were not taken,
so no row is a ratio). "Decode" is the run's own decode line over 32 tokens.

| build | prompt | decode s/token | fill split (whole run) |
|---|---|---|---|
| attribution build, `JITLLM_GPU_STREAM=1` | 89.9 s | 17.0 | lookup 173 s, gather 327 s, H2D 121 s |
| + every group's read issued at once, sheets sent from their frames, `STREAM_GROUPS=4` | 44.8 s | 5.56 | H2D 155 s (5.76 GiB/s pageable), read wait 39 s |
| + page-locked halves (32 MiB) | 46.5 s | 5.46 | H2D 116 s (7.70 GiB/s), copy wait 48 s |
| + per-block expert cache, 18 sheets (`STREAM_CACHE=18`), 93/93 on the cards | 36.2 s | 3.87 | 37.5% hit, H2D 70 s, read wait 48 s, copy wait 29 s |
| + cross-layer prefetch (`STREAM_PREFETCH=1`) | 44.4 s | **8.35** | read wait 135 s, decode read 7.7 GB/token |
| prefetch off, `STREAM_GROUPS=16` | 43.4 s | 4.74 | copy wait 62 s |
| 12 MiB halves copied a megabyte a goroutine, groups 4, cache 18 | 38.8 s | **3.55** | H2D 79 s, read wait 44 s, copy wait 13 s |
| `-devices cpu`, same binary family | 59.6-61.0 s | 3.61-3.69 | decode reads 6.0 GB/token at 2.0 GiB/s |

- **The link is the wall.** `cuda.BenchmarkHostToDevicePinned` on the box:
  8.88 GB/s (8.27 GiB/s) from node 0, 8.03 from node 1; pageable 7.98. Every
  card is on node 0 behind PCIe switches, and two cards at once aggregate
  ~11 GB/s. The pinned fill reached 7.3-7.95 GiB/s, so the transfer runs at
  88-96% of the link. 24.1 GiB of routed experts a token is 2.9 s at the
  link; the cache's ~38% brings it to ~1.8 s.
- **The disk is not the wall.** Sixteen parallel 16 MiB `O_DIRECT` reads of the
  container ran at 5.7 GiB/s (one at 5.3). The host path reads its 6.0 GB a
  token at 2.0 GiB/s of blocked wall; its decode is read-bound, not compute.
- **★ The cross-layer prefetch measured worse, not better.** 60.4% of the next
  block's experts were predicted, and reading them a block early took the
  token from 3.87 to 8.35 s: the demand reads waited 135 s against 48, and
  decode read 7.7 GB a token against 6.5. The prefetch's reads queue in
  front of the reads the fill needs now, and its wrong 40% are bytes the
  disk serves first. A second opinion (codex) at this fork had expected at
  best 3.1-3.6 s and named the exact risk (prediction coverage of the MISSES
  is what pays, and wrong guesses spend the same disk); the mechanism stays as
  `Config.StreamPrefetch`, off.
- **Hybrid expert execution was priced, not built.** The host run's decode is
  3.6 s with ~2.9 s of it blocked on reads; hybrid moves only attention to the
  card and keeps that read, so it lands at the host's rate. It pays only after
  the host's own expert read is fixed, and then the host path gains the same.
- **Placement reads nothing to refuse.** `-devices cuda` without streaming:
  load read 1.16 GiB (was 55.43) and the run took 3m19.8s wall (was 25m06s),
  every mixture block refused from the tensor table (`nn.SizeDecliner`).
- **The default now streams such a block** (`nn.AutoStreamer`,
  `tier.GPU.AutoStream`), with a cache sized by `autoCacheSlots` (18 on this
  box). Two runs: 78/93 blocks on the cards (15 on the host: the auto run's
  cards held 10 blocks each where the forced `STREAM=1 STREAM_CACHE=18` run
  held up to 16 -- not yet explained), prompt 36.9-41.5 s, decode 3.51-3.66
  s/token, against the host's 59.9-61.0 s and 3.61-3.69 interleaved with them
  (auto, cpu, auto, cpu, auto). Faster to first token, level in decode; no
  ratio is claimed (RULE 2 not met).

### 16c-2. Stacking: the host's read, the auto placement, the trial, and hybrid experts.

Same box, container and prompt; one run a row unless said; runs of one table
taken back to back.

**The host's expert read was not at 2.0 GiB/s.** That figure is the pager's
per-phase rate, bytes over the SUM of every reader's I/O time, and the reads
of one layer run concurrently, so it divides by several times the wall. The
host run's own `experts` line has the wall: 70.6 s blocked over 37 steps
(prompt included) for 272 GiB, and a layer's few cold pages read at near the
disk's rate. What was left to take was the overlap: a latent mixture's shared
experts read nothing the routed read brings in, so they now run while it is in
flight (`WithSharedOverlap`, one reader goroutine per State so a warm token
still allocates nothing), with the same float sum (`TestSharedOverlapIsTheSameSum`,
bit for bit, fails when the shared weight is perturbed by 1e-4). `-devices cpu`,
32 tokens, alternating off/on: 3.57, 3.32, 3.64, 3.34 s/token.

**Auto-stream's 78/93 had three causes, found one after the other.** (1) The
cache was sized from the first mixture block offered, a KDA block with two
thirds of an MLA block's base: 35 sheets. (2) Sizing from tensor bytes at all
was wrong: a card's real cost per base, its scratch and the head are what they
come to. (3) Fill-first packed six cards and left two empty. Now a streamed
block is placed on the plain shared bank, the plan spreads streamed blocks over
every card (priced on a streamed block, not the dense lead), and after
placement each card gives its streamed blocks a cache from the room it really
has left (`GPU.sizeAutoCaches`, at `EndPlacement`), stepping a sheet down where
an allocation rounds over. The kernels need nothing: an indexed matvec
addresses a sheet as slot times stride whatever the bank's size, so the cache's
slot ids index a bigger bank as the compact one's did (the gate's
"sized after placement" arm runs a 512-sheet cache on kernels built for 10, bit
for bit). Result: 93/93 on eight cards, 11-12 each, caches up to 42 sheets on
the 44 blocks whose cards had room past the selection.

**The trial.** `initStreamTrial` arms the seam tuner with {streamed, host}
whenever a placement auto-streamed a block (`WithStreamTrial`, on): ABBA runs
with the migration outside the clock, the first tokens after each migration
skipped (one arm read 0.38 tok/s warm and 0.11 just after moving), and the
host adopted only past `SeamTuneMargin`. `TestStreamTrialAdoptsOnlyPastTheMargin`
drives the decision with rates (an 18% faster host adopted, 3.5% not, slower
not, dispersed not; fails with the margin at 1.00); `TestAutoStreamPlacesWhatCannotFit`
runs a trial to a decision through real migrations. `SetSeamTuning(false)`,
which the CLI always calls, used to disarm it.

**The fill's group count is measured** (`streamtune.go`): the pack tuner's
ABBA duels per device, climbing from the container's figure while a doubling
wins by 2%. On the box it settled on 2-8 depending on the run.

**★ Hybrid experts pay once stacked.** The host profile (`JITLLM_PROFILE`,
16 tokens) is 93% matvec, and of what a host token reads from RAM the block
bases (Q8_0 attention, latent and shared projections, 0.63 GB a block, 58 GB
a token) are more than twice the routed experts (16 x 17.5 MB a block, 26 GB).
Hybrid execution keeps the bases, router and shared experts on the cards and
runs only the routed experts on the host's generated kernels over the host's
pages (`nn.LayerWeights.HostExperts`, `streamBank.runHost`): the latent input
(the routed-down projection) and the routing weights come home, the sum goes
back into the latent accumulator, 7 KB each way a block. It is the streamed
block's suspension with the transfer replaced, so a block is still one unit
the device places (RULE 8a); `TestHybridExpertsMatchTheHost` holds it to the
host at every position of Kimi-K3-0.40B (NMSE 2.3e-7, same argmax), failing at
1.6e-5 when the routing weights are swapped for one expert's.

| 64 tokens, back to back | prompt | decode s/token |
|---|---|---|
| `-devices cuda`, hybrid (auto-stream, trial off) | 27.1 s | **3.11** |
| `-devices cpu` | 60.1 s | 4.44 |
| `-devices cuda`, sheets sent, cache, groups tuned | 36.2 s | 5.66 |

**The paired comparison, in one process.** The stream trial is RULE 2's harness
for this question: the same process alternates the hybrid placement (93 blocks)
and the host (0) in ABBA runs of 8 tokens, migrating between them outside the
clock and skipping 2 warm-up tokens after each move. `JITLLM_SEAM_RUN=8
JITLLM_SEAM_ROUNDS=10`, 430 tokens, default auto-stream with hybrid experts: 10 quads, 20
ratios. Hybrid over host **median 1.873, IQR/median 0.057** (warm quads only:
1.875, 0.043); the tuner kept hybrid ("0 blocks is not 5% better (0.530)").
The A/A pairs from the same runs (incumbent against incumbent, challenger
against challenger, three runs apart) centre at **0.987** (warm 1.000) -- no
slot bias -- but their IQR/median is **0.134** (warm 0.109), over the 0.10
gate, and the comparison was taken once. Warm rates: hybrid 0.66-0.93 tok/s,
host 0.37-0.41 (the host budget evicting at this length). The ratio is a
one-pass, A/B-gated result whose self-control is dispersed, and is quoted as
such.

At 64 tokens the host budget starts to evict (peak RSS 418 GB against the
393 GiB budget), which is why every row is slower than the 32-token ones. The
prefetch (`STREAM_PREFETCH`) stacked on the streamed path read 5.9 GB/token
against 4.9 and decoded at the same 5.79 s: neutral now, not the 2x loss it was
before the reads went parallel; it stays an option. Hybrid is the default for an
auto-streamed block (`Config.NoHybrid` sends the sheets).

**Finer pages and sub-block placement, priced.** The streamed path already is
sub-block placement: the base (attention, router, shared) is the card's unit
and the bank pages through, and the expert cache is a resident subset chosen by
use; seeding it from the hotness accounting instead of LRU is the open
variant. Finer expert pages would not cut bytes: a selected expert reads all
three of gate, up and down, so a 16.73 MiB page split per matrix is three
requests for the same bytes. Splitting an expert's rows across cards to use
more links is priced by the link measurement: two cards at once aggregate
~11 GB/s against 8.9 for one, so at best 1.25x on transfers that hybrid no
longer makes, for a reduction across cards per block. The dense base pages
(1185 MiB) are read whole by every token on the host and are resident on the
cards, so a smaller base page would not cut what a token reads either. No
converter change is warranted by these numbers.

### 16c-3. Is auto-stream a limitation? An audit, and what it removed.

Where a mixture block's routed experts run is now a placement like any other:
`model.WithExperts("host"|"card")`, `-placement N=DEV%host` / `%card`,
`tier.WithHybridExperts`, `tier.WithAutoStream`, and the choice is kept on every
device (`nn.ExpertPlacer`), so a block that hops tiers, goes home and comes back
keeps it (`TestOffCardExpertsRelocate`, which counts the off-card runs at each
destination and fails when a returning block arrives unmarked). The stream trial
measures three arms -- the placement made, the host, and the same blocks with
their experts on the other side of the bus -- under the same margin
(`TestStreamTrialTriesEveryArm`).

What "auto" restricted, read at the source and measured:

- **It fired only for a block bigger than every card, not for one refused for
  room. Artificial; removed.** A model whose blocks each fit but whose whole does
  not left the overflow on the host. gpt-oss-20b Q4_K_M on the laptop's 4 GB RTX
  3050 Ti (CUDA, 6 P-cores, 48 tokens, one run each): default 2/24 blocks
  resident, the rest host, 4.84 tok/s; every block streamed with its experts on
  the host, 9.56 tok/s, the host's ids; the same with the sheets sent, 2.90-2.99;
  `-devices cpu` 2.29. Now a mixture block refused for room streams, and the
  trial measures the result against the host. With the change, the default run
  (2 resident, 22 hybrid) read 8.18 tok/s.
- **A block that fits stays resident by default. Real, priced.** Resident costs
  no per-token traffic and runs the experts at the card's bandwidth; hybrid only
  wins where the card cannot hold the experts. Forcing it is one option away
  (`WithExperts`, `%host`), and the trial is the instrument if a shared card
  ever argues otherwise.
- **Streaming needs a pre-packed bank; hybrid inherited it. Partly artificial,
  priced.** Sent sheets need the sheet layout (real). A hybrid block still
  allocates the compact bank and compiles the expert kernels it never launches;
  dropping them frees VRAM for more bases, not yet done.
- **Top-1 routers (llama4) do not stream** (`slots > 1`): a kernel constraint of
  the indexed matvec, real for sent sheets, not for hybrid. Priced.
- **Batched rows refuse a streamed block** (`rows.go`, `prepBatch`): a prompt
  runs such blocks a row at a time. Real until a hybrid batched path (the host's
  expert-major `moeBatch` over the rows) is built; it is the next prefill lever.
- **The expert cache rewrote the selection as slots, and gpt-oss's expert biases
  are indexed by it.** A real bug, found by the sheets run parting from the host
  at the thirteenth token: biased banks now keep the plain bank
  (`TestBiasedExpertsKeepThePlainBank`); a second buffer of true ids would
  lift it.
- **Hybrid refused Gemma 4's dense-MLP mixture and ungated experts.** Real: the
  host's expert path reads one vector and has no ungated fused path; ungated is
  now refused at offer rather than failing mid-token.
- **Backends: none.** Streaming uses the generic `WriteAt` (page-locked memory
  is CUDA's alone, a speed); hybrid uses `Read`/`WriteAt`. gpt-oss-20b on the
  laptop over Vulkan: hybrid 5.91 tok/s, sheets 2.13, both with the host's
  tokens over the first twelve. Metal is untested.
- **Models: no list.** Hybrid runs any gated mixture through `moeFFN` (biases,
  MXFP4, latent mixtures): Kimi-K3's latent and gpt-oss's biased MXFP4 both held
  the host's tokens.

So auto-stream was a mode in one respect -- it engaged only past a size -- and
that is gone: a mixture block that does not get a resident place gets an
off-card one, and whether that beats the host is measured.

### 16c-4. Closing it out: batched hybrid, no bank on the card, per-session host sides, the host pages a hybrid block keeps.

- **Batched rows.** A hybrid block now takes a prompt's chunk and a ragged
  step's rows: the rows are routed on the device (`emitGroupedMoE` up to the
  weights), their selections, weights and expert inputs come home, and the host
  runs them expert-major -- each chosen expert read once and run over its rows
  as one batch (`hostExpertsRows`, `moeBatchExpert`), the host's own batched
  order -- before the sums go back. Only the valid rows run (a padded chunk of
  32 for a 10-token prompt ran 224 rows' experts before that). Gates:
  `TestHybridPrefillMatchesTheHost` (fails at NMSE 6.6e-3 with one row's
  selection used for all), `TestBatchSeamMovesCarryEveryRowOffCard` (three
  rows across seam moves and round trips, fails when a hybrid block refuses
  rows).
- **No bank on the card.** A hybrid block uploads no compact bank, compiles no
  expert kernel and needs no packed layout (an F32 fixture runs hybrid); a top-1
  router streams this way too. Its VRAM goes to bases.
- **Bugs the gates found on the way.** (1) The host side was the placing
  State's closure: a block shared by a second session ran the first one's
  closed generated code (`jit: Call on closed code`). The tier now keeps each
  session's host side and uses the current session's (`hostFns`, dropped on
  detach). (2) While blocks were home the host offered their weights to the
  device as lone matvecs, and a hybrid block never took those copies back:
  +971 KB a round trip. A routed expert is never offered now, and a block whose
  host page stays tells the device to forget its weights' copies at placement.
  (3) A streamed block kept its host block page (1.2 GiB a block on Kimi-K3)
  and `releaseLayer` dropped its expert pages, which it still reads; now the
  block page goes and the expert pages stay (`nn.ExpertHolder`), and nothing
  claims an empty frame for a block page that is out. (4) A block that could
  neither stream nor run hybrid kept its auto mark and could never come back
  resident (Gemma 4's dense mixture, F32). (5) Forced `%host` packed the first
  cards like fill-first and left no room for a prompt's scratch on Kimi-K3; a
  placement's mark spreads like an auto one.
- **Zero allocations.** `TestOffCardDecodeDoesNotAllocate`: hybrid decode and
  steps on synth-kimik3 and Kimi-K3-0.40B, sheet decode on Kimi-K3-0.40B, each
  counted as having run; `TestDecodeDoesNotAllocate` pins the stream trial off
  with the tuners (its migrations are page-ins, not a decode).
- **Tuned constants.** The page-locked half and the direct-send threshold are
  measured per device after the group count, on one ladder tuner
  (`streamtune.go`), each pinned by its `Config` field.
- **The paired comparison did not pass its self-control.** The stream trial
  grew an A/A arm (`WithStreamTrialAA`): the placement against itself, every
  change of arm a migration. On Kimi-K3 with 8-token runs it read 0.993 at
  IQR/median 0.118, with 16-token runs 1.001 at 0.111 -- centred, so no slot
  bias, but over the gate both times, because the hybrid arm's own runs spread
  0.55-0.92 tok/s (the host arm holds 0.37-0.41). The A/B came out 1.935 and
  1.921 in two passes. What would tighten it is the hybrid arm's variance: its
  host expert reads at a long run are evictions of the 393 GiB budget, and a run
  that lands on one is slow. No ratio is quoted (docs/perf/current.md).
- **kimi-k3-in-c on this box is still not possible inside the floor.** It reads
  the 1.56 TB safetensors checkpoint plus a 109 GB packed trunk from local disk
  (O_DIRECT); the RAID has 237 GiB free and must keep 60 GiB, so at most ~177
  GiB could be used. No reduced or streamed mode of it reads weights from
  anywhere but that local tree. Nothing was deleted for it.

## ★ 16d. FOUR JITLLMD SERVING A 0.8 GB MODEL GREW TO 17.6 GB EACH ON ONE NUMA NODE, AND THE KERNEL KILLED ONE. THE HOST BUDGET IS THE SMALLEST OF THREE LIMITS NOW, AND THE MEMORY CACHE FOLLOWS WHAT IS FREE.

The Xeon E5-2680 v4 box (2 sockets, 251 GB, 128 GB a node) ran four
`jitllmd serve` under `numactl --membind=0 taskset -c 0-11`, each serving
Llama-3.2-1B-Instruct Q4_K_M with `-max-batch 64` and the defaults, under an
httpbench sweep (-c up to 16, 128-word prompts, 64 new tokens). Each reached
~17.6 GB anonymous RSS and node 0 OOM-killed one
(constraint=CONSTRAINT_MEMORY_POLICY nodemask=0).

**Where the bytes were.** A host-only reproduction on the laptop counted the
engine's ledger (the live heap after a GC, `jlm.OffHeapBytes`, the memory
cache's `Bytes`, the pooled States' history), not RSS:

| term | before | sized from |
|---|---|---|
| weights (dense region and pages, off the heap) | 766 MiB | the file |
| memory cache | +33.4 MB per distinct 170-token prompt, never hit, bound 24.3 GB | an eighth of the model's host share: 190 GB on that box |
| a State's prompt score rows (`battf`) | 89 MB for a 170-token prompt | chunk x maxSeq, and maxSeq defaults to the model's 131072 |
| a State's decode score rows (`atth`, `attf`) | 32 MB | NHead x maxSeq (not changed) |
| KV pages | committed on growth, dropped by Reset; a pooled State holds none | the positions written |

The cache costs 188 KB a prompt position against 64 KB of f32 KV, because a
partial page is stored at several fills and a logits row (0.5 MB at a 128256
vocabulary) is stored beside it. 17.6 GB is ~530 distinct prompts at 33 MB:
the cache, growing toward 24 GB, is the term that fits, and
`jitllm_mem_cache_bytes` on that box is the number that confirms it.

**Why the limit was not seen.** `sched.MemBudget` was eight tenths of the
smaller of MemAvailable and the cgroup. A process under `--membind=0` can
allocate only from node 0, and the budget never read that; each of four
processes saw ~240 GB available and took 190 GB, and each cache an eighth of
it. The server also read the budget once per process.

**The rule.** The host budget is eight tenths of the smallest of: the
cgroup's ceiling (memory.high or memory.max); the memory of the NUMA nodes the
process may allocate from (a `get_mempolicy` MPOL_BIND mask intersected with
the cpuset's Mems_allowed, each node's MemFree plus its file pages and
reclaimable slab); and MemAvailable. It is re-read at each load, with what the
engine's own models hold added back. `sched.MemLimit` -- the collector's
ceiling -- is the smaller of the cgroup and the bound nodes' MemTotal. Every
figure is overridable: `model.WithPageBudget`, jitllmd `-maxmem` (a model) and
`-host-mem` (`server.Config.HostBudget`, what the models divide),
`-mem-cache-max`. The memory cache is bounded by an eighth of the share AND an
eighth of what it holds plus what the host has free now, re-read after every
generate: n caches on one host settle at n/(n+7) of the memory that was free
(four: 36%) instead of each taking an eighth of a figure read before the
others grew. The many-row score buffers are sized by the keys their rows
reach, doubling from 256, never by maxSeq.

Gates: `sched.TestMemBudgetHonoursEveryLimit` (fake /proc and /sys: cgroup,
membind, cpuset, others holding memory; reverting the node term fails membind
and cpuset), `sched.TestKernelBindNodesReadsThisProcess` (passes unbound and
under `numactl --membind=0`), `model.TestScoreRowsFollowTheReach` (sizing at
the context fails it), `server.TestAServerUnderLoadStaysUnderItsBound` (the
ledger of the 1B under 16 distinct prompts with 2 GiB available).

## ★ 16e. A BLOCK MOVES BETWEEN TWO DEVICES AS A CALL OF ITS OWN, AND jitllmd NO LONGER REFUSES IT.

`jitllmd place -move A:B -to DEV` (PlacementService.RelocateBlocks) refused a
device destination the session was already on, with the reason "the tier
exposes only the per-device count, so a per-block destination would be a guess
dressed as an API". That was a choice of the code, not a limit (RULE 11e): the
tier already placed a block by name (`PrepLayerOn`, WithPlacement) and said
where each block was (`DeviceOf`). The only engine route between two cards was
`State.SetDevice(b)`, which brings every block and all history home and offers
the model again, so cuda:0 -> cuda:1 was cuda:0 -> host -> cuda:1.

`State.MoveBlocks(lo, hi, to)` is the call: each block's history comes home
(MigrateKV copies, so the source keeps its rows until the target has them),
the target takes the block by name, which frees it on the source, and the
history goes up; the projection follows when none of the State's blocks is
left beside it, and the copy it leaves is freed (`devTier.dropHead`) instead of
sitting on a card that serves none of the model. jitllmd's RelocateBlocks
calls it for a range already on the session's devices; the CLI prints the
history moved and its path, and the TUI's engine page moves every device block
to the tier's next device on `d`.

- **A block another session is using does not move.** The weights are the
  model's and the route per block is one (`GPU.own`), so a move for one
  session would take the block from under another. Refused by name; a route
  per session would be the change that lifts it.
- **`GPU.Layers`' head-on-another-device arm is reached now.** Moving the upper
  half of the blocks leaves the projection on a device that no longer holds
  the last block: `TestBlocksMoveBetweenDevices`' `upper` arm runs it,
  teacher-forced against an unmoved control (the section below is why no
  placement reached it before).
- `TestBlocksMoveBetweenDevices` runs the relocation models (every principle
  fixture and mixture, Llama-3.2-1B) on two devices of one tier
  (`JITLLM_MOVE_DEVICES`, the laptop's card under CUDA and Vulkan by default):
  zero prompt rows recomputed, every block on the target, the logits before
  the move bit-identical, after it inside the two backends' band (largest
  0.52 on Llama-3.2-1B), and bit-identical where both devices run the same
  kernels. With the history's upload removed it fails at the first token after
  the move (12.6 on Llama-3.2-1B).

## ★ 16f. THE PAUSE OF A MOVE BETWEEN TWO V100s WAS 3.1-5.6 s AND IS 20-35 ms: THE TARGET IS PREPARED BESIDE THE TOKENS, THE HISTORY GOES CARD TO CARD.

16e's move brought the history home and up again, and prepared the target
inside the pause. On the V100 box (two of eight Tesla V100-SXM2-16GB, NV2
between cuda:0 and cuda:1), Llama-3.2-3B-Instruct Q4_K_M, an 1852-token
prompt and every block moved at token 80 and back at 160, that pause was
3.1-5.6 s for 443-462 MB of history (`model.TestMovePause`, both arms in one
process, alternated, five rounds, tuner off). Where the time went was read off
the commit before anything was built on it:

    the target's weights and kernels     re-read from the pager, uploaded,
                                         compiled: most of it, and nothing
                                         to do with the sequence's position
    the history                          D2H, Go transposes into the host
                                         cache, back out, H2D
    the target's KV pool                 grown a block at a time inside the
                                         commit: ~7 ms a block
    the source's frees                   a driver free of every tensor and a
                                         dropped graph: 3-5 ms a block
    peer access                          enabled by the first copy: ~114 ms

Each went where it costs the sequence nothing:

- **Prepared beside the tokens.** `State.PrepareMove` hands the target every
  block (weights uploaded, kernels built, `nn.BlockMover.StageBlock`) and the
  projection (`StageHead`) on another goroutine while the sequence keeps
  decoding on the source; staging takes no call lock and is refused unless the
  target routes nothing, so no step reads what it changes. The same call grows
  the target's pool for the history (`prewarmKV`) and enables peer access.
  `Commit`, at a token boundary, has only the history to copy and the routes
  to switch. jitllmd's generate (one at a time and the batch loop) serves a
  RelocateBlocks this way between two tokens of the reply it is streaming.
- **Card to card.** `tier.copyHistory` copies the session's pages from the
  source's pool into pages the target allocates: `cuMemcpyPeer` a run of pages
  consecutive on both sides (`backend.PeerCopier`), and between two backends a
  gather on the source, one read, and a write a run (the laptop's CUDA to
  Vulkan). Pools that page differently go through the row layout
  (`pagedMigrate`), still without the State's cache. 470 MB of history went
  over NV2 in ~24 ms (0.85 ms a block of 16.8 MB, ~20 GB/s).
- **Freed after the switch.** What a move leaves on the source is freed on
  another goroutine once the routes name the target (`RetireMoved`); every
  placement call waits for those frees first.

Measured, both arms in one process (`TestMovePause`, five rounds):

| | direct | host (WithDirectMoves(false)) |
|---|---|---|
| pause cuda:0 -> cuda:1 | 20.4-20.9 ms | 4579-5589 ms |
| pause cuda:1 -> cuda:0 | 22.6-34.8 ms | 3104-3563 ms |
| history | 460-471 MB KV, 470-528 MB copied (whole pages) via peer | 443-462 MB via the host cache |
| prompt rows computed again | 0 | 0 |
| logits against the unmoved control, teacher-forced | bit-identical | bit-identical |

What it costs instead, and stays open:

- **The source decodes slower while the target is prepared**: 19.3-37.3 ms a
  token against 5.1-6.2 unmoved, for the 1.3-1.8 s the preparation takes (35-79
  tokens decoded where the host arm decoded none for 3-5.6 s). Not attributed:
  the box's CPUs carried other users' load (load average ~27), and the
  preparation's page reads, packs, uploads and compiles run beside the
  source's owner hops. Throttling the preparation, or compiling once for both
  cards of one architecture, are the candidates; neither is measured.
- **The first token after a move takes 10-25 ms** (5.1-5.8 unmoved): the
  target's first step, its graph captured. In one traced run the dozen after
  the second move ran at 10-23 ms while the source's frees ran
  (RetireMoved), and the unmoved rate after them.
- **The weights still go host to card.** Copying a resident block's tensors
  card to card would skip the page read and the upload, but the tier builds a
  block's buffers from host spans (`prepLayer`), and with the preparation off
  the pause it buys the pause nothing; it would shorten the preparation.
- **A block another session uses does not move** (16e).

Gates: `TestBlocksMoveBetweenDevices` (its default arm asserts the direct path
ran and every block was staged; `/host` is the control arm),
`TestAMoveLandsBetweenTwoTokensOfAReply` (a 400-token reply on the V100s is
the unmoved reply token for token, the move landing between two of its tokens
with every block staged ahead).

★ **Found on the way, not fixed here: on a V100 the server's memory cache
answers a second session differently from the first with no move at all.**
origin/main, `-devices cuda:0` alone, stories15M: three sessions in turn on
"Once upon a time" read [18358 29892 1183 4446 263 4802], then [304 748 3271
29889 1 9038] twice; with `NoMemCache` all three read the first. The laptop's
card does not show it. The move gates run without the memory cache for that
reason, and say so.

## GPU.Layers' head-on-another-device arm is dead code

(Moving blocks between devices reaches this arm now: 16e.)

★★★ **AND `GPU.Layers`' HEAD-ON-ANOTHER-DEVICE ARM IS DEAD CODE, ESTABLISHED
BY THREE FAILED ROUTES RATHER THAN BY READING IT.** The arm issues
the output projection as its OWN submission when the head's device is not the
last run's. Each route fails for its own reason and each was measured:

    1. a STATIC placement   PrepHead targets g.tail(), so the head's device IS
                            the last run's device by construction
    2. a SEAM MOVE          model's shrink does `s.head, s.headReady = nil, nil`
                            BEFORE releasing anything, so the projection is
                            always surrendered. Measured: [18 4] placed with the
                            head on device 1, shrink to 18, "head still on a
                            device: false". SetHeadOnDevice(true) restores
                            headReady, which that same line cleared
    3. an EARLIER DEVICE    placement fills the fastest device FIRST, so a later
                            one takes blocks only once the earlier is FULL --
                            and "full" means its leftover is under a PAGE. An
                            earlier device can hold the projection only when the
                            projection is smaller than a BLOCK

★ **ROUTE 3 IS THE INTERESTING ONE BECAUSE IT IS TRUE ON A REAL MODEL AND
STILL DOES NOT FIRE.** A mixture's blocks carry expert banks and are huge while
its projection is one matrix, so Qwen3-30B-A3B has **page 392.66 MiB against a
245.75 MiB projection** -- head < page, which no dense model here satisfies
(tinyllama is 25.72 against 88.24). Built that configuration, two devices,
[2 2] placed of 48: the projection lands on a device and the logits are right.
**And deleting the head-only submission entirely leaves them BIT-IDENTICAL**,
which is how the route was ruled out rather than assumed -- with a partial seam
the HOST runs the blocks above the device's, so `engine/model/` never hands GPU.Layers
a head at all.

★ **A CANDIDATE FIX WAS BUILT AND REVERTED, WHICH IS THE PART WORTH KEEPING.**
`PrepHead` falling back through the remaining devices when the tail has no room
is correct and more general -- the tail is the device that filled LAST and is
likeliest to be full, so the projection was falling to the HOST while an earlier
card sat with room. It is reverted anyway: the gate written for it does not
DISCRIMINATE (removing the fallback leaves the same placement and the same
NMSE), so it would have shipped an unmeasured change to default placement on a
path that already costs a crossing. **An unreachable branch does not justify an
ungated fix**, and `model.TestTheHeadGetsItsOwnSubmissionWhenItIsNotOnTheLastDevice`
skips with all three reasons written down instead.

★ **AND THE ROW CANNOT BE TAKEN ON THE XEON, WHICH WAS TESTED RATHER THAN
ASSERTED.** v6 predates the pre-VNNI work, so on a VNNI-less host it declines
every quantized kernel: `jitllm hardware` there prints "kernels Q4_0 none ...
Q6_K none -- running the float64 reference" and the decode measures **0.14
tok/s against HEAD's ~31, a 220x gap**. A v6/HEAD row on that box would compare
the ORACLE against generated code. The trusted instrument and the meaningful
comparison are on different machines, and saying so needed the measurement.

## Qwen3-Next-80B on the card: the expert bank, streaming, and the mid-block failure path

★★★ **AND A MIXTURE'S EXPERT BANK IS UPLOADED TO A DEVICE WITHOUT EVER BEING
READ, WHICH IS THE SHARPEST INSTANCE OF THIS FILE'S OWN STANDING FAILURE MODE.**
`pageIn` skips the routed experts on purpose -- 2.56 GiB a token against 37.8 --
and its comment ends *"that is why every consumer of a packed span has to ensure
it"*. `offerRange` was the consumer that did not, and a device cannot come back
for an expert later: it is handed the whole bank once. On a FRESH frame that is
zeros, so the block's routed FFN contributes nothing and the model stays fluent
-- `jitllm run` generated 64 correct-looking tokens that way; on a REUSED frame
it is another block's Q4_K scales, and an f16 that happens to be NaN takes the
logits non-finite. The two commands differ only in whether a host decode had
cycled the frames first. `Model.ensureBlockExperts` reads the bank, and
`nn.LayerWeights.Ensure` makes the tier call it at ADMISSION rather than at the
offer -- reading at the offer spends a bank on every block the budget then
refuses, which on the 80B was ten of 1.04 GiB (55.41 -> 63.98 -> **57.27** GiB
of file reads on the same run). The teacher-forced row goes from NaN at position
0 to **max|dlogit| 0.32-0.55 at every position**, which is this file's own f32
reduction band.
`docs/engineering-history/placement.md` 15j, including why a gate for this has
to be sized against the 1 MiB CHUNK rather than against the expert count.

★★★★ **AND THE WHOLE 80B RUNS ON THE CARD NOW: 48 of 48 BLOCKS, 0 ON THE HOST,
TOKEN-IDENTICAL TO llama.cpp.** `-devices gpu:0` with every block streamed
(then `JITLLM_PAGE=1`; now the placement `-placement '*=gpu:0~'`) on an
RTX 3050 Ti Laptop with 2.00 GiB of budget against 46.23 GiB of weights:

    48/48 blocks + the output projection, 816 block-run(s), 0 on the host
    816 page-in(s), 863 page-out(s), 775.73 GiB moved through ONE slot
    ids [785 6722 315 9625 374 12095 13 576 6722 315 9856 374 19846 ...]

**0.06 tok/s against the CPU's 0.12, and that is the honest cost rather than a
defect**: one slot means every block is uploaded over PCIe every token, which is
what 775.73 GiB for seventeen tokens is. The rate is what a second slot buys and
this card cannot hold two 1042.8 MiB pages; the CORRECTNESS is what the work
below bought.

★ **THREE THINGS HAD TO CHANGE AND ONLY THE FIRST WAS THE FEATURE.**
`devTier.emitLinear` runs the gated delta rule (the decline is lifted);
`minSlots` went 2 -> 1; and `pageIn` re-reads the block's host bytes through
`nn.LayerWeights.Ensure`, which is a bug that had been in the pager its whole
life. `docs/engineering-history/placement.md` 15l has all three.

★★ **THE EARLIER READING, KEPT BECAUSE ITS LESSON ABOUT HEADLINES STANDS.** 2 of
48 blocks on an RTX 3050 Ti, 34 block-runs, no demotion. The greedy chain parts
from the CPU arm of the SAME binary at token 10, on a continuation whose margin
is 0.13 -- cpu "the capital of Germany is Berlin", cuda "the capital of Italy is
Rome" -- and the SAME binary gives Berlin on a second run, because `tuneSplit`
picks its reduction order per process. That is the amplification band measured
under RULE 11c: two device blocks at depth 3 and 7 with forty recurrent blocks
below them, and a chain that agrees on one run and not the next. The gate is the
teacher-forced diff, **max|dlogit| 0.32-0.55 at every position, 1 of 8 argmaxes
flipped**, and that is this file's own 0.2-0.94 f32 reduction band.

★ **THE MATCH THIS REPLACES WAS THE BUG.** "Token-identical to llama.cpp" was
measured before `Model.ensureBlockExperts`, when blocks 3 and 7's routed experts
were ZEROS -- a mixture with two dead FFNs reproduced the reference transcript
and the correct engine does not. That is RULE 10's shape in a headline rather
than in a test, and it is exactly the fluent-wrong-text the entry above is about:
a greedy chain agreeing is not a measurement, and the position where it stops
agreeing is chosen by a 0.13-logit margin.

★★★★★ **AND THE 80B PLACES 48 OF 48 BLOCKS ON THE 2 GiB CARD, 1.67 GiB
RESIDENT, NOTHING DECLINED.** `-devices gpu:0 JITLLM_GPU_STREAM=1`,
RTX 3050 Ti, 1.885 GiB of budget against 46.23 GiB of weights:

    48/48 blocks + the output projection, 1.67 GiB resident, 0 declined
    ids [785 6722 315 9625 374 12095 13 576 6722]  -- llama.cpp's own golden
    decode 0.78 tok/s, 85.06 GiB read over 9 tokens

★★ **WHAT CLOSED IT WAS ONE BUFFER INSTEAD OF FORTY-EIGHT, AND THE ACCOUNTING
IS THE WHOLE ARGUMENT.** `model.TestDeviceBudgetAccounting` on that container:

    base per block (container max)   26.96 MiB
    one sheet of gate+up+down         1.98 MiB   x10 routed = 19.84 MiB
    a placed block, bank per block   46.81 MiB   x48 = 2.194 GiB   DOES NOT FIT
    a placed block, bank SHARED      26.96 MiB   x48 + ONE bank   = 1.283 GiB
                                                 + 150.75 recurrent = 1.430 GiB

The compact bank was **42% of a placed block** and it is live for a few
microseconds: blocks run in order inside one submission, and block i's bank is
consumed by its own FFN three launches later. `devTier.sharedCompact` keys one
per (matrix, format, shape) for the whole device. It is safe under exactly the
conditions streaming already needs -- the suspension Syncs before it reads the
selection home, a streamed range already refuses graph capture, and sessions are
serialised per device.

★ **AND `which` IS IN THE KEY BECAUSE gate AND up ARE THE SAME SHAPE.** Both are
NFFNExp rows of NEmbd in the same format, so a key of (format, rows, k, slots)
collides and one buffer serves both: fill writes the gate's ten sheets, then the
up projection's over the top, and the gate matvec reads up's weights. Caught by
`TestStreamedExpertBankMatchesTheResidentOne` at pos 0, **0.4392 against
1.0620** -- which is what a bit-equality gate is for. Two matrices that are the
same SHAPE are not the same TENSOR.

★★ **AND FOUR OF THE FIVE CRITERIA HOLD, MEASURED IN ONE PASS EACH.**

    1. placement   48/48 blocks, 1.67 GiB resident, 0 declined, 0 refused
    3. tokens      ids [785 6722 315 9625 374 12095 13 576 6722], the golden
    4. -kv-cache   turn 2 reused 5 of 5 positions (100%), prefill
                   7.126 s -> 81 ms, ids identical across turns
    5. verify      0/6 positions disagree, 1 tie, max|dlogit| 0.595 against
                   the 0.94 band, exit 0
    2. decode      0.78 tok/s against a target of 2  -- NOT MET

★★★ **AND `verify` EARNED ITS PLACE ON THAT LIST BY CATCHING A REGRESSION TWO
COMMITS OLD.** The mid-block guard had been made conservative --
"any placed linear block" -- and `verify`'s teacher-forced arm has no batched
prefill on a mixture, so `Layers` reported a REFUSAL as a failure and the guard
turned a sound host restart into a terminal error: *"the accelerator failed on
block 0 and this model is a hybrid"*. **Both proxies for "did the recurrence
advance" are wrong**: `li > 0` misses a run of [0,18) that fails at block 5, and
"a linear block is placed" refuses a device that ran nothing at all.
`devTier.recSteps` counts the blocks whose emitLinear completed with the
submission still good, and the guard reads it. A positional proxy for a
measurable fact is how this one went wrong in both directions.

★ **THE REMAINING GAP IS CRITERION 2 AND IT IS PAGE GRANULARITY, NOT EXPERTS.**
85.06 GiB over 9 tokens is 9.45 GiB a token against the 0.872 GiB of routed
experts the token consumes. With all 48 blocks on the CARD the host still faults
each block's page every token -- 16 frames for 48 blocks, so every block is
evicted between tokens and its base weights are re-read purely to re-bind the
spans. That is the whole-block page, which is the open question this file
already carries.

★★★★ **AND FOUR ACCOUNTS OF THAT GAP ARE NOW REFUTED BY MEASUREMENT, WHICH IS
THE USEFUL PART. EVERY ONE OF THEM WAS MINE AND EVERY ONE READ AS OBVIOUS.**
Qwen3-Next-80B, `-devices gpu:0 JITLLM_GPU_STREAM=1`, 48/48 placed:

    candidate                what was done                    what happened
    the host READ BYTES      chunk dose 1024 -> 256 -> 64 KiB  85.07 -> 21.26
                             (JITLLM_CHUNK_KIB)                -> 5.32 GiB,
                                                               rate UNCHANGED
                                                               at 0.79-0.81
    the read REQUEST count   the same dose, counted            20,767 -> 20,742
                                                               -- IDENTICAL, so
                                                               the chunk never
                                                               moved the thing
                                                               that costs
    the UPLOAD call count    stage each plane's k sheets into  90 -> 9 transfers
                             one buffer, one Write per plane   a block, cgocall
                                                               4.40 -> 3.29 s,
                                                               rate UNCHANGED
    the SUSPENSION           JITLLM_GPU_STREAM_FIXED deletes   1.06 against
                             the Sync and the readback         0.82 -- <= 1.29x,
                                                               and part of that
                                                               is read locality

★★ **THE FIRST PROBE FOR THE SUSPENSION WAS CONFOUNDED AND SAID 1.83x.** A
fixed selection of experts 0..k-1 asks every block and every token for the SAME
ten sheets, whose chunks stay resident -- 5,121 read requests against 24,791 --
so it measured the round trip AND a working set five times smaller. A
block-dependent stride keeps the reads honest and the answer falls to 1.29x. A
probe that changes two things confirms whichever one you were expecting.

★★★★★ **AND THE TOKEN SPLITS BY WALL, WHICH SETTLES IT: THE HOST READ IS 79.5%
AND IT IS NOT BYTES, IT IS REQUESTS.** `Stats.TStreamWait/TStreamRead/TStreamPut`
time the suspension's three phases, because a CPU profile cannot divide them --
the reads are issued from a goroutine per run and overlap. 80B, 48/48 placed,
64 KiB chunks, 11 tokens:

    block compute (all of Layers)   14.39 s
      drain                          0.37 s   ( 2.6%)
      host read                     11.44 s   (79.5%)
      upload                         2.37 s   (16.5%)   over 528 fills

    5.81 GiB in 11.44 s  =   520 MiB/s
    24,791 requests      = 2,167 IOPS, 0.46 ms each

**520 MiB/s on a device this file measures at 4.73 GiB/s** (ReadAt x16,
format/jlm/read.go). The pattern is 2,254 small scattered reads a token, and the chunk
dose already proved the BYTES do not matter -- so what is left is the request
count and the fact that they do not overlap into the device's queue.

★★ **AND THE REQUEST COUNT IS THE PAGE SIZE, WHICH IS WHY THIS NEEDS THE
CONTAINER CHANGE AND NOT ANOTHER KNOB.** A frame is PageSize -- 1042.8 MiB on
this model, because a page holds the whole 512-expert bank -- so a 16.37 GiB
host budget holds SIXTEEN of forty-eight blocks. Thirty-two are re-faulted every
token, and each fault is ~47 scattered reads to collect 19.84 MiB of sheets into
a 1 GiB frame. The host now reads no base at all (selectedRanges' `base` flag)
and still cannot win, because the FRAME is sized by the bank it no longer reads.
That is the page-size trade this file already records as knowingly paid --
*"a static page size tuned to the max recommended by jlm, makes everything
simple and fast at the cost of some mem"* -- and the streamed path is the first
configuration where the cost is the whole row rather than some memory.

★ **SO CRITERION 2 IS A STEP 5 PROBLEM, NAMED BY MEASUREMENT RATHER THAN BY
ORDER.** Steps 1-4 closed placement (2/48 -> 48/48), correctness (golden ids,
verify clean) and the cache (100% prefix reuse); none of them can make a 1 GiB
frame hold 20 MiB of sheets.

★ **AND THE PROFILE SAYS WHAT IS LEFT, WITHOUT SAYING WHY.** At 64 KiB chunks,
61.26% of samples are in `Pread` and 24.94% in `runtime.cgocall` -- and the
Pread share did not move when the bytes fell 16x, so the pager is priced per
REQUEST and nothing counted requests until `jlm.File.Reads()`. Total samples are
49.83% of wall: the process is IDLE half the time. The CPU arm of the same
binary reads 2.31 GiB a token and runs at 0.93 tok/s against the device path's
0.82, so **on this box the streamed device path is at parity with the host and
the remaining term is unattributed.**

★ **WHAT THIS RETIRES.** "The host read dominates" was the goal's own falsifier
and it is REFUTED: 16x fewer bytes bought nothing. "Suspension costs more than
it saves" is refuted as the main term at <= 1.29x. The two levers that DID pay
are placement, not rate: the routed-sheet read (3.2x fewer bytes) and the shared
compact bank (48/48 blocks). The read as the rate lever was tried at three
chunk sizes and 16x fewer bytes left the rate where it was.

★ **AND `verify` LOCATED IT AS "2940 ms of a 2946 ms token is encoding
launches, 100% of Layers", WHICH IS TRUE AND WAS READ TOO NARROWLY.** That
number brackets everything between the first launch and the last, and the
suspension sits inside it -- so it says the time is in the encode loop and NOT
which part. The four rows above are what actually divides it.

★★★★ **AND THE STREAMED PATH IS 6.8x FASTER FOR READING TEN OF 512 EXPERTS
INSTEAD OF ALL OF THEM, WHICH NEEDED NO FORMAT CHANGE.**
`-devices gpu:0 JITLLM_GPU_STREAM=1` places a mixture's blocks WITHOUT their
routed bank (`tier.Config.StreamExperts`): gate, up and down hold NExpertUsed
sheets each, and the submission SUSPENDS after ExpertRank to read the selection
home, upload those sheets and resume. Qwen3-Next-80B, 34 of 48 blocks placed,
same binary, same prompt:

| | whole bank | routed sheets |
|---|---|---|
| host reads | 229.89 GiB | **72.67 GiB** |
| decode | 0.11 tok/s | **0.75 tok/s** |
| ids | `[785 6722 315 9625 374 12095 13]` | identical |

★ **THE READ WAS THE CEILING AND `Ensure` WAS THE WRONG PLAN FOR IT.**
`ensureBlockExperts` deliberately skips nothing and reads the WHOLE bank,
because on the host `moe()` ensures each expert afterwards -- correct there, and
32.8 GiB a token here against the 0.92 the token consumes.
`nn.LayerWeights.EnsureExperts` is the plan for a caller that already knows
which ten it wants, and the sheets were addressable all along
(`jlm.File.SheetSpan`, and `model.bankExpert` cuts the same bytes on the host
side).

★ **AND A DECODE GATE CANNOT SEE A WRONG SHEET INDEX, WHICH IS RULE 10 AGAIN
AND WAS CAUGHT BY RUNNING THE VIOLATION.** `EnsureRange` tracks residency in
1 MiB CHUNKS, so on any bank under a few hundred MiB the ten routed sheets land
in the same chunks the whole bank does -- every byte arrives whichever sheets
were asked for. `TestStreamedExpertBankMatchesTheResidentOne` compares every
logit BIT FOR BIT on both arms and is **GREEN against `o := uint64(x) + 1`**.
`TestSelectedRangesCoverExactlyTheRoutedSheets` gates the PLAN instead, against
the same arithmetic `bankExpert` uses, and separates the two failures: an
off-by-one trips coverage, a whole-bank fallback trips narrowing (128x on the
fixture).

★★★★ **AND FOUR LIVE BUGS SAT ON THE MID-BLOCK FAILURE PATH, ALL FOUND BY
ASKING WHAT STATE A BLOCK MUTATES BEFORE IT CAN FAIL.**

    the recurrent parity flipped on a FAILED submission. `la` latches the
    first error and turns every later launch into a no-op, the block loop has
    no per-block check, and emitLinear's `rp.cur = 1 - rp.cur` ran anyway --
    so past the failure nothing ran and everything advanced, and the next
    token read the half nothing wrote.

    EVERY RELEASE PATH MOVED THE KV AND NONE MOVED THE SUMMARY. Four sites --
    demoteAll, the -gpu-grow adopt, and two SetGPULayers shrink arms -- all
    called migrateKV and none called MigrateRec, while ReleaseLayers ends in
    freeRec. `State.migrateRec` is the twin and is wired at all four.

    THE GUARD WAS POSITIONAL AND WRONG TWICE. `s.recurrent() && li > 0`: `li`
    is where the RUN started, not where it failed, so a run of [0,18) failing
    at block 5 read li == 0 and took the host restart with five blocks already
    advanced -- and on the branch that places 48 of 48, block 0 IS a device
    linear block. It is `s.devLinear()` now, the SET and not the prefix.

    recStepped WAS NEVER SET. It is read in Layers' deferred flip and assigned
    true by nothing outside a test, so `g.recParity` was constant and
    graphKey's parity axis with it -- ONE recording served every token of a
    fully resident hybrid, baking c[cur] as the read half and c[1-cur] as the
    write half. Replayed, that reads the same buffer every token and the
    summary NEVER ADVANCES. The field was written for exactly this and was
    simply never connected.

★ **THE GATE IS THE SHRINK AND NOT THE FAILURE, BECAUSE THE FAULT INJECTOR
CANNOT REACH MID-BLOCK.** `injectedFail()` is the first statement of `Layers`;
its vocabulary is an ordinal over whole calls. `SetGPULayers(0)` takes the same
ReleaseLayers path through the same migrate pair and is public API, so
`TestShrinkingTheSeamBringsTheSummaryHome` runs six tokens with the seam moved
in the middle and compares the tail against a pure-host run: **1e-4 with the
rescue, 8.5e-01 without.** A `phase@block` injector is what the other three
need and is owed.

★ **AND THREE `fmt.Fprintf(os.Stderr, "DBG ...")` WERE IN SHIPPING CODE** on
that same path (`engine/model/kvpage.go`), from the MigrateRec work. Deleted.

★★ **AND A PLACED LINEAR BLOCK'S HISTORY HAD NO WAY HOME, WHICH THE DECLINE HAD
BEEN HIDING.** `nn.LayerDevice` had `MigrateKV` and no twin for the recurrence.
`State.rconv`/`rstate` are what every HOST consumer reads -- `-kv-cache` seals
through them, `SetGPULayers` restores through them -- and a block running on a
device keeps its summary in the tier's `recPair`. So the cache stored a summary
that had never seen the prompt: with one block placed, `attention-on-device`
gives identical tokens and `linear-on-device` parts at token 1.
`MigrateRec` is the twin, `syncRecFromDevice`/`pushRecToDevice` are the two ends,
and it is also the primitive a hybrid's seam needs to MOVE at all --
`ReleaseLayers` calls `freeRec`, so a shrink drops the summary rather than
bringing it home.

★ **AND ONE WRONG EXPLANATION REACHED A COMMENT BEFORE A VIOLATION TEST REFUSED
IT.** The first account blamed the double-buffered pair's other half being zero;
a gate built to catch that passes it 3/3, and the readings it was written on came
from a probe that kept three tiers open on one card. A probe that measures what
the hypothesis names will agree with it -- only removing the fix and demanding
the failure return can tell them apart.
`docs/engineering-history/placement.md` 15n.

★★★ **AND THE LINEAR-ATTENTION DECLINE IS LIFTED, WHICH IS WHY IT IS 48 AND NOT
12.** The refusal read *"a causal convolution over a shifting history and a
rank-one update to a per-head matrix, both read-modify-write on a buffer that
persists across tokens, which RULE 13 forbids a device kernel outright -- so this
needs TWO state buffers and a swap"*. That was right about the REQUIREMENT and
wrong about the conclusion: `recPair` is the two buffers and the swap, and every
kernel `emitLinear` issues is out of place. `kernels.GatedDeltaStep`,
`Conv1dRows`, `Conv1dShift` and `DeltaGate` were built, lowered on all three
backends and gated against the host the whole time, with no tier code calling
them -- the state `tier.TestDeclineNamesTheGraphFeature` recorded as "the honest
state rather than an oversight". So `-devices gpu:0` put 36 of 48 blocks on the
CPU and reported it as a graph the device does not implement.

★ **AND TWO BUGS CAME OUT OF WIRING IT, BOTH INVISIBLE TO EVERY GREEN GATE.**
`bs.quantF` is `Quantize(rows*NFFN)` with its element count BAKED, and the delta
output is `VHeads*VDim` wide -- the fast fixture has NFFN 32 and VHeads*VDim 32,
**equal by accident**, so the dense and moe arms passed while the 80B's own
geometry was wrong at position 0. And `g.bs` is built from whichever block
arrives FIRST, which on a hybrid is a LINEAR one, so the attention block ran
against a scratch with no out-gate kernels: four of four placed, the three
linear at 4.5e-05 and the attention one at 1.7e-01. A hybrid is the vision
tower's two-shapes-one-tier problem with the opposite answer -- the two kinds
alternate over ONE residual, so it is one scratch built from the UNION rather
than a second scratch.

The partial-rotary decline is LIFTED -- `RoPERows` writes only the dimensions it
rotates and the tier now copies the tail first, which is what its contract asked
for. What kept the 80B answering *"The capital of France is is is is is"* was
never the block: a mixture has no batched prefill, `Layers` reported that
refusal as a device FAILURE, and the recovery restarts the chunk on the host --
which on a hybrid re-runs the linear blocks whose gated delta rule has already
advanced. The rows go through the device one at a time now, and a hybrid refuses
the host restart rather than answering from a double-applied state.
`docs/engineering-history/placement.md` 15h has both.

★ **AND IT RUNS, TOKEN-IDENTICAL TO llama.cpp ON THE 80B.** Same
prompt, greedy, CPU tier, 48 layers of which 36 are the gated delta rule:

    llama.cpp  The capital of France is Paris. The capital of Germany is Berlin
    jitllm     The capital of France is Paris. The capital of Germany is Berlin
    ids [785 6722 315 9625 374 12095 13 576 6722 315 9856 374 19846]

and every intermediate of layer 0 matches to 1e-5 against `llama-eval-callback`
on the same single token -- conv, SiLU, the L2 norms, beta, the delta rule's
output, the gated norm, the output projection and the shared expert. That is
the architecture verified, not merely fluent.

★ **THE 0.12 tok/s BESIDE llama.cpp's 3.38 IS NOT A COMPARISON AND MUST NOT BE
QUOTED AS ONE.** jitllm was run with `-maxmem 14000000000`, which pins twelve of
forty-eight blocks resident; llama.cpp was run in the same cgroup with NO
equivalent constraint, and because it mmaps, the kernel let it use the whole
26 GiB as page cache. One arm was held to 14 GiB of weight residency and the
other handed 26, and the difference was then reported as a property of the
engines. RULE 1 says quote the configuration WITH the number; the two arms did
not share one.

★ **AND THE ASYMMETRY SURVIVES EVEN WHEN NEITHER ARM IS GIVEN AN EXPLICIT
BUDGET, WHICH IS THE PART TO REMEMBER.** `scripts/vs-llamacpp.sh` does put both
engines in ONE cgroup -- but jitllm self-limits to `sched.MemBudget()`, which is
EIGHT TENTHS of that cgroup, while llama.cpp reaches effectively all of it
through page cache. On a model that fits inside the eight tenths the handicap
is invisible; on one that does not, it is the whole row. Qwen3-30B-A3B at cap
26G came to 18.71 GiB of budget against 18.83 GiB of model -- short by 0.12 GiB,
which is one page, which is one 392.7 MiB read every token. Part of that row's
0.5777 is a self-imposed limit and not the pager.

A fair over-committed row therefore needs the SAME ceiling on both arms, which
means shrinking the cgroup rather than passing `-maxmem` to one of them: the
kernel's page cache is bounded by the cgroup, so a smaller cgroup constrains
llama.cpp too. That row has not been taken.

What the 80B DOES establish is bytes, which needs no second engine: 37.8 GiB
read per token to use 2.56, a 14.8x amplification, because a page is a whole
block. See the open question on paging granularity, and note that a model which
FITS reads nothing at all.

## The split: a hybrid block's expert cache, and Qwen3.8-Flash-Next against Strata

Strata (an engine written for Qwen3.8-Flash-Next alone) decoded the real
UD-Q4_K_XL file at 17.5 tok/s on one V100 where jitllm's hybrid placement
(`JITLLM_EXPERTS=host`: every block's base on the card, every routed expert on
the host) read 5.8 (`vs-strata.sh`, 20 ABBA rounds twice, ratio 0.33). The
measurements below were taken on that box (2x Xeon E5-2680 v4, the run pinned
to node 0's 14 cores and memory, one V100-SXM2-16GB, CUDA), decode 128 tokens
after "The capital of France is".

**The 0.33 was a cold engine, not a slow one.** Each sample is a fresh
process, and the experts are read with O_DIRECT when a token first selects
them: 273 MB a token of decode page-ins, 17.3 of 21.9 s blocked on expert
reads. Strata copies 62 GiB of experts into page-locked RAM before its clock
starts (107 s). `JITLLM_PRELOAD_WAIT=1` holds `run` and `speed` until the
background preload has read every page (35-41 s here), which is the same
thing measured the same way: the warm hybrid decode was **13.1 tok/s** (256
tokens), 0.75 of Strata, with host experts 50 ms of a 76 ms token.

**The split.** A hybrid block whose bank is packed is now placed with the
compact bank and its indexed kernels (`cacheable` in `prepLayer`), and
`sizeAutoCaches` gives it an expert cache from the room placement left, as it
does a streamed block's. Each decode token (`streamBank.split`): the
selection comes home, the experts the cache holds run on the card from their
slots (a miss's routing weight zeroed, its slot a hit's so the kernels read a
sheet that holds an expert), the host runs the misses, and the two sums are
added in that order (`kernels.Add` into eout). The host's share runs after
the card's hits AND the shared expert are queued (`finishLate`), so the card
computes while the host does; a latent mixture (Kimi-K3) finishes at once,
since its latent out-projection reads the sum. The cache takes in at most
`Config.SplitAdmit` misses a block a token over the coldest slot this token
does not read, by selection counts halved every 256 tokens; an occupied slot
changes hands only to a miss selected at least twice as often plus two, and
only within a per-token budget over every block (`Config.SplitSwaps`, 8: alternating warm runs read 22.2 and 22.7 tok/s at 8, 17.1 and 20.2 at 2, 20.3 and 20.4 with none). On
this file 43 sheets a block fit beside the 7.35 GiB base (2064 experts,
6.4 GiB); the hit rate settles at 53-56%.

Admission is not free: a sheet is a host copy into the page-locked halves and
a transfer on the token's path, about 1 ms each. Without the hysteresis the
first run took 5288 sheets in 256 tokens (15.4 GiB uploaded) and the split
ran SLOWER than the plain hybrid (10.75 against 13.12 tok/s). With it, and
two untimed warm-up runs (`JITLLM_WARMUP=2`) so the cache is filled before
the clock starts, as a server's would be: 14.79 against 14.09.

**Three things the profile then named** (`JITLLM_CUDA_KERNEL_TIMING`, each
kernel carrying ~3.9 us of event):

- *A hyper-connection mix ran on one multiprocessor.* `matvec_f32_4x10240`
  ran at split 32 -- 4x32 threads on a card of eighty SMs -- 48 us a launch,
  4.6 ms a token. The split tuner offered no split above 32 (`splitMax`): a
  shape under 64 rows is now offered splits to 512 and the measurement chose
  320, 0.88 ms a token. 14.79 -> 16.50 tok/s.
- *The pool's workers parked between the device's host calls.* Between two
  host expert calls the host waits on the card with no core busy, past the
  pool's 250 us spin, so every call woke 13 workers: `moeFFN` took 545 us for
  4.6 experts with the default budget and 405 us with a 2 ms spin
  (JITLLM_SPIN_US=2000: 21.5 and 21.7 against 18.1 and 18.7 tok/s,
  alternating). A token whose device made such calls now has the workers spin
  2 ms (`model.WithHybridSpin`, `sched.Pool.SetSpinFor`): 21.0 and 21.9
  against 19.2 and 18.7 with the pool's budget.
- *The host's experts are at the bandwidth.* With the spin, a call costs a
  fixed ~100 us (its four or five pool regions) and ~66 us an expert, 2.9 MB
  at 44 GB/s; Strata's pool reads 35.6 GB/s.

**The rows** (`scripts/vs-strata.sh`, the build at "backend: a CUDA queue's
staging arena", one V100, node 0 for both engines, decode 128 tokens,
speculation off in both, 20 ABBA rounds, each sample a fresh process, the
quiet gate on node 0's CPUs and the measured card). jitllm runs with
`JITLLM_EXPERTS=host JITLLM_PRELOAD_WAIT=1 JITLLM_WARMUP=2
JITLLM_IGNORE_EOS=1`, its two warm-ups on a text of their own
(`JITLLM_WARMUP_PROMPT`, an essay about the printing press). The preload and
the warm-ups are its warm-up, as Strata's page-locked copy and profile fill
are Strata's; neither is on the clock. Strata runs its build-v100 (CUDA 12)
variant; its docs ask compute capability 7.5 or newer.

| prompt | pass | median ratio (jitllm / Strata) | IQR/median |
|---|---|---|---|
| A/A, jitllm both arms | -- | 0.9770 | 0.075 |
| "The capital of France is" | a | **1.0341** | 0.045 |
| "The capital of France is" | b | **1.0334** | 0.018 |
| "The theory of relativity explains" | a | **1.2988** | 0.035 |
| "The theory of relativity explains" | b | **1.2955** | 0.026 |

jitllm read 16.8-18.2 tok/s and Strata 15.9-17.5 on the first prompt.
"Once upon a time" could not be measured: Strata accepted drafts on it with
no drafter given (`--suffix-draft 0`, no `--mtp`), which the harness refuses
as a speculated sample.

An earlier pair of passes read 1.2631 and 1.2645 on the first prompt with
the warm-ups on the measured prompt itself. That primes the cache with the
very text it then serves; the rows above do not, and replace them.

**The rate depends on the text.** With the tune cache frozen every process
decodes the same tokens and the samples agree; but the tuners' per-shape
picks move the rounding, the greedy path parts, and the hit rate with it:
six tune states of one build read 20.1-27.5 tok/s on one prompt, at hit
rates 47.5-62.3%, the slowest and fastest states three cache entries apart
(Q8_0 10240x2560 split 8 or 10, 2560x6144 32 or 24, 2560x640 10 or 5). A row
on one prompt is one text's row, hence the two prompts above.

**Time to first token**, from the same samples' logs (median): jitllm
prefills five tokens in 285-287 ms and six in 337-340; Strata prefills four
in 212.3-212.5 ms and five in 223.9-224.3, its last prompt token taken as its
first decode step. Per prompt token that is 57 ms against 53 for the first
prompt and 56 against 45 for the second. From process start Strata's first
token came 51-55 s in (its page-locked copy), jitllm's after its 35-41 s
preload wait and the warm-ups.

Bytes a token: the host reads 48 x 4.6 misses x 2.9 MB = 0.64 GB of experts
in 48 x 405 us = 19.4 ms of `moeFFN`, 33 GB/s including each call's fixed
regions (a read wall for node 0 measured in the same process is owed); the
card reads its 7.35 GiB base and the hits.

**The prompt.** Strata prefills the first four tokens in 213 ms; jitllm's
batched chunk of five took 590-612 ms in the same samples, every row's experts
on the host, one expert-major region set per expert. Two changes: a chunk of
up to eight rows runs its host experts a row at a time through the decode
path's grouped regions (474-482 ms), and once a device has run a model's
experts on the host, a prompt of up to eight tokens runs through the decode
path itself, where each token splits between the card's cache and the host
(`WithPrefillForwardRows`, unset: 210-248 ms; alternating runs of one build,
`JITLLM_PREFILL_FORWARD=0` against the default).

**Another mixture: Mixtral-8x7B-Instruct Q4_K_M** (24.5 GiB, eight
experts of ~99 MB a block, top-2) on the same card and node, 128 tokens,
two warm-ups, runs alternating, two of each:

| placement | decode tok/s | prompt (6 tokens) |
|---|---|---|
| default: 18 of 32 blocks resident, 14 on the host | 13.35, 13.97 | 410-438 ms |
| every block hybrid, no cache (`JITLLM_EXPERTS=host JITLLM_GPU_HYBRID_CACHE=0`) | 6.91, 6.91 | 868-871 ms |
| every block hybrid with the cache (`JITLLM_EXPERTS=host`) | **14.77, 14.65** | 370-385 ms |

Four sheets a block fit (of eight); 59.7% of the selections hit. The cache
is what makes the hybrid placement pay on this model at all (2.1x), and it
then edges past the resident-prefix placement; whether the default placement
should be hybrid when the cache holds half a block's bank is the stream
trial's question (`initStreamTrial`), not yet asked of it.

**What a hybrid cache costs elsewhere, and how it gives way.** The cache
takes the room placement left, so history or a block offered later found
none. `dropCaches` gives caches back, the last block's first, whenever
`roomOrReclaim` is short and before `trim` pages a block out
(`Stats.CachesDropped`); a block without its cache runs on the plain bank it
was placed on. A float bank (the qwen4exp fixture's) has no packed sheets and
keeps the plain hybrid path. `tier.WithHybridCache(false)` /
`JITLLM_GPU_HYBRID_CACHE=0` keeps the plain hybrid path everywhere.

Gates: `TestSplitExpertsMatchTheHost` (Kimi-K3-0.40B Q8_0, the latent
mixture; DeepSeek-V2-Lite Q4_K_M at a six-block seam, a shared expert and no
latent; the qwen4exp fixture, a float bank that must fall back) holds a tight
cache (k+1 sheets), an auto-sized one, a budget cut halfway that must drop a
cache with every block still placed, and no cache, each to the host: the same
greedy token unless the host's own top two are within 0.05, logits within the
model's device band, and the split counted (split steps, experts run on the
card, experts admitted). Run against violations: misses' weights left on the
card's share (NMSE 6.3e-2 at position 2, both cache arms fail); `dropCaches`
made a no-op (0 caches dropped, the yield arm fails). On the V100 every arm
passed, DeepSeek-V2-Lite's auto cache holding the whole 64-expert bank.
`TestOffCardDecodeDoesNotAllocate` now also requires the split to run on a
packed hybrid model (0 engine allocations over 64 tokens with it). The real
checkpoint against llama.cpp de7fa0a, teacher-forced on the host: 20 of 20
positions agree (`TestGreedyMatchesLlamaCpp` with `JITLLM_GREEDY_PAIR` naming
the existing container, so the test does not reconvert 125 GB); the free-run
" Lisbon" / "<think>" parting reported earlier is a near tie, and Strata's
own greedy run says "<think>" there too.

## The prefix cache: addressing granularity, the hybrid restore point, and sealTail

★★ **AND THE CACHE'S ADDRESSING GRANULARITY WAS THE COMPUTE PAGE, WHICH IS THE
ONE THING EVERY PREFIX CACHE KEEPS SEPARATE.** `kvPageTarget` is 256 because of
the ATTENTION KERNEL -- EmitAttnAcc re-reads the V window once per 8-vector
column block -- and that number ended up deciding what a prefix cache can NAME.
vLLM's automatic prefix caching blocks at **16**, SGLang keys a radix tree over
the token sequence, and LMCache sizes its chunk independently of the engine's
block. `kvChunk = 16` is that constant here; `pageKey` was already a prefix hash
and only the granularity was wrong.

    two chat turns sharing 32 of 40 positions   0 reused -> 32 reused

★ **AND A PREFIX SHORTER THAN A CHUNK MISSES, IN THIS ENGINE AND IN ALL THREE OF
THOSE.** Two prompts sharing TEN tokens reuse nothing at chunk 16, and that is
the design rather than a fault.

★★ **A HYBRID RESTORED AT ONE POINT PER PROMPT, BECAUSE A RECURRENCE CANNOT BE
REWOUND AND ITS STATE IS 79 MB -- AND NOW KEEPS POINTS DURING PREFILL, PRICED
PER MODEL.** The gated delta rule's summary can only be sealed where the engine
holds it. Measured on Qwen3-Next-80B: an 88 MB cache entry, of which 36 linear
layers x 2,195,456 B = **79 MB** is the summary, so a point every 16 positions
would be 1.26 GB for a 256-token prompt. For a while a hybrid therefore probed
only the exact end, and a decision model (d1 = LFM2, Lev = qwen35) restored
nothing across requests whose boards differed in a few tokens.

Points are captured during prefill now (`engine/model/recpoint.go`): a host
linear block's summary after the row that lands on a point, in the row loop
(`recRowPoint`). The spacing is priced per model (`recEvery`): the smallest
power-of-two multiple of kvChunk at which a point's summaries cost no more than
the attention history the same span adds, capped at the page -- **16 on d1
(LFM2's short convolutions), 256 on Lev (qwen35's delta state)**. Beside the
spaced points, one at the position a prompt parts from the one before it
(`recMark`): a request's state before each question, a conversation before its
newest turn, restored exactly by the next prompt that shares it. Restore
probes the points at or under the prompt's limit, largest first
(`kvCache.restore(limit)`), each only with every summary there.

★★ **AND THE PAGES ARE SHARED, NOT COPIED (`SharedStore`).** A first version
stored the attention prefix at every point as a copy (a fill-named page per
16 positions, then 16-position chunks) through `MemStore` and copied it back
into the session on restore: a second copy of the history beside the
session's, against the standing rule that host storage follows ownership. It
was deleted. A sharing session (`State.ShareKV`, now accepted on a hybrid)
publishes its pages to the store -- the sealed ones as before, and the page a
prompt ends inside under the key of its fill -- and names each page by the
chunks of history it holds (`SharedStore.alias`, `kvCache.aliasChunks`), no
byte moved. A prompt that parts from it inside a page aliases that page by
the chunk it shares; its first write there copies that one page
(`kvPages.cow`). The restore points' summaries are the one copy: a state that
moves on, stored through the store's `Set` (in place when re-stored).

★★ **AND ON A DEVICE THE PAGES ARE NAMED WHERE THEY LIVE.** The pool's pages
were the session's alone ("two sessions' pages never alias",
device-kv-paging.md), so a device block's history went home to be cached and
came back up to be restored -- on the V100 a warm d1 decision took 2x a cold
one, most of it those migrations. A layer's page now carries the cache's key
and a count of the sequences holding it (`jit/gpu/tier/kvshare.go`): sealing
names it (`PublishKV`), a restore puts it into the sequence's table
(`MapKV`), a write into one copies that page on the card first (`cowPages`),
a sequence letting it go drops its hold, and the unheld named pages are the
first room taken back. Eviction is as built; compaction gives the unheld
named pages back first and leaves a layer with held ones alone. A linear
block's restore points are snapshots on the card (`recshare.go`), copied from
and into the session's seat. `Stats.KVPositionsMigrated` is zero for a warm
prompt (`TestDecisionReusesThePrefix`). A placed linear block keeps a point
only where a chunk ends (`recChunkPoint`); a chunk is not cut for one. Cutting
one at the mark, or every 128 or 256 positions, was measured on the V100
(jivman's run, one process, cold and warm alternating): each cut is a
submission's fixed cost, and on d1 it took warm from 125 ms to 133 (mark),
156 (128) and 147 (256) against cold's 119-122; on Lev the mark took it from
311 to 557 against cold's 396-458. A device's prefill costs what its rows
cost (64 rows 13 ms, 512 57 ms, d1 on the V100), not a fixed grid's.

**And a warm prompt through the shared store makes no heap allocation**
(`TestDecisionWithAPrefixDoesNotAllocate`): the keys are hashed with a kept
hasher and named once (`kvCache.digest`), the probe list and the recurrent
staging are kept, a copy-on-write takes the layer's spare, and a page the store
already held hands the session's memory back to that spare.

★★ **A WARM PROMPT THAT REUSED NOTHING PAID FOR THE CACHE, AND THE MEDIAN
REQUEST IS ONE.** On jivman's run a d1 decision on the V100 restores a whole
512-position chunk or nothing (a placed linear block keeps points at chunk ends
and the prompt's end only), so most requests restore nothing; warm was 125 ms
against cold's 120 at the median while its mean was lower. Timed per request
(one process, cold and warm alternating), the extra was all fixed costs:

- **The chunk keys were hashed from position 0, per chunk, per layer that
  asked**: 3.3 ms a request. They are prefix hashes, now taken in one pass as
  the running hash's sum at each boundary and kept for the sequence
  (`kvCache.ck`): 0.06 ms. A chunk-aligned restore point is named by its chunk
  key (`recPointKey`), and a probe on a chunk boundary asks that name alone, so
  a probe that misses hashes nothing.
- **Each copy synchronised the context.** A point's snapshot of every
  recurrent block and a seat's zeroing were one `Copy` per buffer, each a
  CUDA owner hop plus a context sync, each a command buffer and a wait on
  Metal. They are one list now (`backend.CopyLister`): on Lev the zeroing in
  `Reset` went from 2.1 ms to 1.0 a request, cold and warm alike; on the M4,
  d1's zeroing from 7.8 ms to 0.4 and a point's snapshot from 3.7 to 0.5.
- **A hybrid faulted its pages before asking for a point**, and a page on the
  card is a table write: a request with no point mapped and dropped its whole
  shared prefix. The highest point under the prompt is found first
  (`kvCache.recFloor`), and with none no page is faulted.

After them (V100, jivman's 60 requests, in-process): d1 cold 121.6 ms, warm
122.7 at the median (a miss now costs about 1 ms: the point at the chunk end
and the prompt's end, and the names published), the warm mean 7% under cold;
Lev cold 404 ms, warm 302. On the M4 (Metal): d1 cold 1658 ms, warm 1662;
Lev 5427 against 3530.

**A restore too small to pay was priced, not built.** The smallest restore is
one chunk (16 positions), and its fixed cost (the probes and a point's load)
measured 0.3 ms on the V100 against 16 rows of d1's prefill at about 0.1 ms a
row, and more on every other tier measured: no device here makes a one-chunk
restore lose, so a threshold would be a branch no request takes.

**A prompt that ended inside a chunk kept its point where no probe asked.**
The probes are the chunk boundaries, the prompt's limit and the mark, and the
mark is a State's: a chat's next turn in a new State never asked the position
its last prompt ended on, and on a device `TestKVIsHeldOnce`'s chat arm
restored nothing. The fills past a chunk boundary where a point was kept are
written through the store under the boundary's chunk key (`State.noteEnd`,
any store, so a disk cache finds them too), `recFloor` asks them past each
boundary, and on the card the prompt's page is named by its fill beside the
point.

The gate itself had two holes that hid this: the held-once gate shared one
store between its host and device arms, so the host arm's pages answered the
device arm's names; and an 81-row board was 950 of d1's tokens but fewer than
a device chunk's shared prefix in openjev's, which then restored nothing on
the card. The boards now grow per model until two of them share more than a
chunk, and a readout that attends both ways (vllm-d3) is held to restoring
nothing.

★★ **AND `-kv-cache` CACHED NOTHING FOR ANY REALISTIC PROMPT UNTIL `sealTail`,
AND ITS OWN MESSAGE EXPLAINED WHY IN A WAY THAT READ AS INTENT.** A page sealed
only once the position passed it, so a chat turn stored nothing and two
identical runs both printed *"0 of 15 prompt position(s) reused (0%) (pages are
256 positions; a shorter prompt caches nothing)"*. Three things were missing and
each alone leaves the counter at zero: the partial page a prompt ends inside is
sealed now (`sealTail`, named by its FILL so it cannot alias a full page), the
clamp is in POSITIONS rather than whole pages, and the prompt's final logits are
stored too -- without which a fully cached prompt still had to run a token, which
a HYBRID cannot do, since its recurrent summary is pinned to the exact position
it was taken at and cannot be shortened by one.

    Llama-3.2-1B  CPU   42-token prompt   1.521s -> 0.009s   0% -> 100%
    Qwen3-Next-80B gpu:0 15-token prompt  15.774s -> 0.337s  0% -> 100%, ids identical

`docs/engineering-history/placement.md` 15m, including why the whole page buffer
is stored rather than its filled prefix and why a hybrid gives a tail up rather
than shortening it.

## RULE 12's tower case: the comment that cost the tower its paging, and its page budget

### ★ RULE 12: a comment explaining why something is NOT done is a claim about the code, and it rots.

Before inheriting a "we do not do X because Y", check Y at the source. It costs
one grep and has paid six times: a comment claiming the IR was "not getting"
shared memory, one claiming folding the projection in "saves nothing", one
claiming qwen2 biases were not loaded (they were), one claiming they were (they
were not), a 5% threshold living on in a source comment after the rule that
set it was retired, and **one claiming the VISION TOWER has no block pages**.

★ **THE TOWER ONE COST IT ITS PAGING, AND THE COMMENT DESCRIBED ITS OWN CAUSE.**
`Tower.container` read *"an mmproj has no `blk.N.` tensors, so every weight lands
in the dense region and stays there"* -- false on both halves.
`convert.identify` reads the `v.blk.N.` prefix and assigns a block number, and
none of the six matrix roles per block is `Expanded()`, so an mmproj has REAL
block pages: SmolVLM-256M is **12 of 7.59 MiB**. What the comment described was
the effect of the `LoadAll()` sitting three lines under it, which read every page
plus the dense region before `buildTower` could look at one LayerNorm -- the
tower still paying the cost `model.Open` stopped paying at v8, O(model) in
RESIDENT bytes where it need only be O(model) in bytes READ.

★ **AND THE TOWER TAKES A PAGE BUDGET, WHICH IS THE HALF THAT IS EASY TO
FORGET.** Paging it without one just moves the double-spend: `model.Open` and
`model.OpenTower` each defaulted to `sched.MemBudget()`, so a VLM run had two
containers sizing themselves against the same host bytes with neither knowing
the other existed -- the same error the Scope section refuses when a scheduler
ADDS an integrated GPU's heap to the host's. `OpenTower` takes `model.Option`
now (so `WithPageBudget` serves both loaders), and `cmd/jitllm` -- the only
place that knows both files exist -- hands the tower the SAME `maxmem` the text
model got and then subtracts `Tower.HostBytes()` from the text model's share.
`HostBytes` is dense plus `min(frames, NBlocks)` pages, not `WeightBytes`: a
tower capped at three frames of twelve costs three pages, and subtracting twelve
would take back bytes nobody spent. ★ **AND THE RESERVATION IS TRANSIENT, NOT PERMANENT, WHICH THE FIRST VERSION
GOT WRONG.** Subtracting the tower off the top is peak-correct and leaves the
text model paged for the whole run against weights that are read ONCE: an image
is encoded before the text prompt is prefilled, and from that moment the
tower's pages are dead. `Tower.Release()` handed them back and `cmd/jitllm`
re-budgeted before `st.Prefill`, so the peak was one budget either way --
tower plus a text model sized around it during `Encode`, text model alone
afterwards. (Both are gone since: one container and one pager, whose
eviction takes a finished picture's blocks as it takes any block -- THERE IS
ONE MODEL.) SmolVLM at `-maxmem 260000000`, token-identical output:

    reserved for the whole run   20 frames, 162 page-in(s), 142 eviction(s), 0.63 GiB read
    released after Encode        30 frames,  30 page-in(s),   0 eviction(s), 0.12 GiB read

`jlm.ReleasePages` frees the frame buffers as well as the pages, because
`DropPage` alone frees nothing when a pool exists -- a frame is a REUSED buffer,
so nilling `pages[i]` only forgets which block is in one the pool still owns.
And `bindOne` cannot undo a release: it returns early on an empty span and
leaves the stale slice, which pins the backing array. The tensors are nilled
explicitly, exactly as `releaseLayer` learned to.

★ **AND `setPageBudget` IS CALLED TWICE ON A VLM RUN, WHICH BROKE ITS EARLY
RETURN.** "The budget holds every host block, nothing to do" is right the first
time and wrong the second: it left the text model capped at the size the tower
had forced, which is the reservation the release exists to give back. It widens
a pool that a previous call narrowed now, and only skips a model that was never
capped.

SmolVLM at `-maxmem 260000000`:

    before   no `pages` line: all 30 text blocks resident, tower ALSO holding
             0.10 GiB -- 0.26 GiB claimed against a 0.24 GiB budget
    after    20 of 30 resident, `tower 0.10 GiB` printed beside the budget
    peak RSS 392,160 -> 355,680 KiB, and the 38 MiB of pages given back is
             what the 36.5 MiB of RSS is

Fixed in container **v11**: the tower's ten per-block vectors join
`Role.Expanded()` so `buildTower` touches no page, and `Tower.pageIn` faults each
block from `Encode` as it reaches it. `TestTowerOpensWithoutFaultingAPage` fails
against both violations -- the `LoadAll()` restored (12 of 12 resident) and the
matrices swept into `Expanded()` (the v9 mistake, "v_attn_q is always-resident").
`TestTowerEncodesUnderAPageBudget` is the one that says PAGED rather than merely
lazy: twelve blocks in ONE frame, bit-for-bit identical output, which needs
eviction, re-read and re-bind all to be right.
`TestOpenTowerHonoursAPageBudget` covers the option and its ORDERING -- applied
before the first page is read, since with no pool every block takes a page of
its own and a budget set afterwards has already been exceeded -- and fails both
when the budget is parsed and never applied (0 frames) and when `HostBytes`
ignores it.

The same predicate bug was in the converter's padding report, which counted
expanded tensors as page bytes and UNDERFLOWED to
`17592186044415.44 MiB of padding` on the first tower it saw.

## Scope as it stood in AGENTS.md, with its measurements

### Scope

**jitllm IS AN INFERENCE OS, AND ORCHESTRATION IS ITS CORE.** It should use every
CPU core and every accelerator on the machine, and move layers and pages between
them to deliver the best rate it can -- and every one of those decisions must be
forcible by the caller through a `With...()` option. That is the standing
direction, stated by the user, and it is what "done" means below.

**The analogy is CONCEPTUAL, not literal: jitllm is ONE process.** What Linux
does for processes and threads, jitllm does for models and blocks:

    Linux kernel          jitllm
    process               a loaded model
    thread                a layer / block
    CPU cores             CPU cores AND every accelerator
    the scheduler         which block runs where
    a physical page       a packed block in a slot
    swap / backing store  the GGUF, through the packed arena
    a page fault          a page-in

★ **AND "SCHEDULER" IS ALREADY TAKEN IN THIS TREE FOR A DIFFERENT AXIS.** The two
are orthogonal and only the first exists:

    model.Scheduler     SEQUENCES onto batch rows -- which request occupies which
                        row and when, admission and retirement. Built, driven
                        rather than self-running, no goroutine of its own.
    block placement     BLOCKS onto compute resources, with memory paged. Which
                        part of the MODEL runs where. This is the core, and it
                        is the one being built.

You cannot schedule what you cannot relocate, which is why block paging is
primed whether or not the model fits -- see `docs/design/block-paging.md`.

★ **STANDING GOAL, NOT CURRENT WORK: A KERNEL COMPOSER.** Stated by the user:
the JIT should eventually *compose* kernels, choosing and fusing
from the kernels it already knows, the operation graph and the hardware in
front of it (ISA, subgroup width, cache and shared-memory sizes, measured
per-launch and per-byte costs), rather than hand-writing each fusion. It is a
direction to design toward, not a task to start: keep new kernels and tuners
expressible as data a composer could select from (a shape, a cost measured on
this host, what it fuses), and prefer that to a special case baked into a
call site. The day-to-day evidence it will need is already being gathered:
per-kernel nvprof tables against llama.cpp, the measured prefetch tuner, and
measured negatives like the workgroup-per-row norm in `gpu-kernels.md`.

It is not a new project. Most of the machinery exists and was built one lever at
a time without the sentence above to hang it on:

| the OS does this | jitllm |
|---|---|
| enumerate the hardware | three backends; Vulkan picks by device type, `JITLLM_VK_DEVICE` pins |
| know each device's memory | `Mem()` and `UnifiedMemory()` on all three |
| place work on a device | `PrepLayer` DECLINES PER BLOCK -- RULE 8a, and the hook everything else hangs off |
| migrate while serving | `-gpu-grow` adopts one block per token; load-time placement costs 1.69 s against a 0.14 s first token, so it starts on the host and moves in |
| re-place by measurement | `seamtune`, ABBA over RUNS of tokens with migrations outside the timed window, 1.05 adoption margin so a near-tie cannot flap |
| page residency | `residency.go` per expert (73.9% measured reuse), the block ring per layer (1.34x on the 70B) |
| core residency | one coreset for the process, `SetParticipants` per region; pack width re-tuned on real tokens. Per phase on the M4, where prefill has its own P+E pool (`5b04d82`); one pool on Linux, where E-cores measured neutral in prefill (Open questions). |

**What is missing is the PLURAL**, and it is the current work: several devices at
once rather than one, layer migration ACROSS devices rather than only to and from
the host, and device-side paging.

★ **THE CONTROL SURFACE IS DONE, AND "done" MEANS THE LIBRARIES READ
NO ENVIRONMENT AT ALL.** Sixty-seven JITLLM_* names were read inside `sched`,
`nn`, `gguf`, `bench`, `jit/cpu`, `jit/gpu/{tier,backend,vulkan,kernels}` and
`model` -- many at package-variable initialisation, which no caller could reach
even in principle. They are `With...()` options now:

    sched  3 options   nn  12   model  9   tier  18   (plus WithDevices/-Budget)
    cpu.Configure/SetWidths   kernels.MatVecShape.Center   vulkan.OpenDeviceWith
    gguf.WithoutWarm          bench.Guard(skip)   backend.SetVerbose

★ **AND EIGHT OF THE SIXTY-SEVEN WERE INVISIBLE TO A GREP FOR
`os.Getenv("JITLLM_`**, which is how a census of them undercounts itself:
JITLLM_CHUNK and JITLLM_PART reached `newDuelTuner` as an argument,
JITLLM_SEAM_{WARMUP,RUN,ROUNDS} reached `envIntDefault`, and
JITLLM_GPU_{QTILE,KTILE,ATILE,PARTS,ATTNNT} came from a table of `{name, *int}`
pairs. Count them by what a package READS, not by what a literal grep finds.

`cmd/jitllm` reads the same NAMES and passes them through, so every script and
every habit is unchanged; what moved is WHERE they are read. Turning the JIT
off is not an option at all -- there is no interpreted tier (RULE 8) -- and
device fault injection is the `jitllmfault` tag (`tier/fault.go` -- the one
file in the tree that still reads an environment variable, and it is not in a
release build).

★ **AND THE MIGRATION'S OWN HAZARD IS RULE 10's, WHICH IT HIT TWICE.** A
`t.Setenv` that no longer reaches anything leaves a gate GREEN while testing a
configuration that was never selected. `nn.TestPackTunerClimb` caught one loudly
-- the tuner pins are package variables written by `NewJIT` and not restored by
the framework, so one `WithTune(TuneOff)` disarmed every later test in the
package; `ResetForTest` resets them now. `tier`'s subgroup gate caught the other:
`New` publishes its option set, which cleared the package-wide Vulkan configuration
between two devices in one loop (it is per device now, `vulkan.OpenDeviceWith`). Every `Setenv` of a migrated name is gone from the tree.

★ **THE DEVICE TIER HELD ONE SEQUENCE AND NOTHING SAID SO, AND IT PRODUCED WRONG
TOKENS. FIXED** (`tier.GPU.Attach`, `docs/design/device-sessions.md`).
`l.kc`/`l.vc` -- the attention history -- sat on the tier's per-BLOCK layer, and
`nn.LayerDevice.Layers` carries a POSITION and no state identity, so two
`model.State`s on one tier wrote the same cache at the same positions. Measured
on tinyllama, 22 blocks placed, two sequences interleaved: **NMSE 2.110e-01 and a
greedy token of 505 where 6231 was right**, with nothing reporting anything.
Serial single-State use -- everything the engine ships and every board row above
-- was never affected.

Refusing the second State was the other option and the user refused it by name:
*"we can't refuse, jitllm is an inference os multimodel multi devices multi
intances"*. A SESSION owns the sequence; the TIER owns the model.
`tier.GPU.Attach()` returns a view satisfying `nn.LayerDevice` UNCHANGED, so
neither `nn` nor `model` widened.

Three more came out of the same afternoon, each now its own gate:

    the CAPTURED GRAPH bakes in the recording session's buffer pointers, so the
    corruption SURVIVED the per-session cache IDENTICALLY -- only NoGraph=1
    showed the caches were already right. Dropped on a session switch.

    CONCURRENT SetDevice interleaved the offer sequences: four States placed
    1, 1, 22 and 1 blocks where each places 22 alone, silently, every answer
    correct, three of four simply on the host. BeginPlacement/EndPlacement
    bracket the whole sequence.

    nn.NewJIT WRITES PACKAGE-WIDE KNOBS WITH NO LOCK, so building JITs
    concurrently raced cpu.SetWidths -- the symptom was a host matvec declining
    a shape its own JIT had emitted for.

★ **AND THE FIRST VERSION OF THE PLACEMENT GATE PASSED WHILE BROKEN**, which is
RULE 10 inside the very configuration this entry is about: it asserted the
forwards SUCCEEDED, and three of four States were running on the host with every
answer right. The assertion has to be the block COUNT.

★ **A SESSION'S KV HISTORY AND ITS RELEASE ARE ITS OWN.** The history is
paged (`docs/design/device-kv-paging.md`): a batch's rows or one long
conversation take pages for that session's sequences alone, and a session
bringing a block home leaves it on the card for every other session using it
(`docs/design/device-sessions.md`, last section; `model.TestABatchLeavesAnotherSessionAlone`).

★ **WHAT IS NOT DONE, NAMED RATHER THAN IMPLIED: concurrent sessions and the
sessions-per-device BUDGET.** Every session shares one scratch, so a session's
own calls are SERIALISED on a device -- correct, not concurrent. The scratch
IS charged to the budget, and a prompt chunk's and a ragged step's are reserved
at placement, so a full card places a block fewer rather than refusing the
prompt (`tier/scratch.go`, placement.md 15x). Sessions of ONE model wholly on one
device are the exception: `model.StepRuns` runs their decode tokens, and an
admitted prompt's chunks, as rows of one step, and the server's step loop
(`server/batch.go`) puts every such generate there; any other session still
takes the device alone, in turn. How many sessions a card holds at
a given context is bounded by pages and unmeasured, and on a card that cannot
hold two the answer is fewer blocks each rather than more sessions.

**★ AND MULTI-DEVICE MAKES A DORMANT TUNER LOAD-BEARING. IT IS NO LONGER A
PREDICTION: IT WAS MEASURED, AND THE CLAIM IT RESTS ON IS FALSE.**
`seamtune` is off by default and said why: *"on both machines this project has,
more device blocks is strictly better -- the case the tuner exists to catch does
not currently occur here. It is for hardware this project has not seen."*

**This box IS that hardware, and the case occurs on every row that uses the
iGPU.** Qwen3-30B-A3B, n=160, interleaved, power-clamped samples dropped:

| devices | blocks placed | median tok/s | against cpu |
|---|---|---|---|
| cpu | 0 of 48 | 17.84 | -- |
| **cuda:0 + cpu** | **4 of 48** | **19.66** | **+10.2%** |
| vulkan:1 (Iris Xe) + cpu | 13 of 48 | 15.2 | -15% |
| cuda:0 + vulkan:1 + cpu | 5 of 48 | 15.4 | -14% |
| all | 5 of 48 | 15.2 | -15% |

The discrete card buys 10% with FOUR of forty-eight blocks -- 2.40 GiB free of
3.81 is all that fits. The iGPU takes THIRTEEN and loses 15%, because its memory
IS host memory: it adds no bandwidth, competes for the same bus, and removes
work from the faster tier. Adding it to CUDA drags that configuration down too.

So "more device blocks is better" is not merely stoppable in principle; it is
already wrong here, and a tuner that is OFF cannot catch it. `-tune-seam`
exceeded a 500 s budget on this model without reporting, so whether it RECOVERS
the placement is untested -- which is the next question, not a settled one.

★ **AND THE RATIO IS A MEDIAN OF SURVIVORS, NOT A GATED ROW.** Three of ten
samples in the winning arm came back at 7.6-8.1 tok/s with cores near 1000 MHz
against a 4670 MHz idle peak. RULE 2 says drop them and rest the box rather
than raise the round count; the ordering above is robust to that, the exact
percentages are not.

**★ AND A DEVICE THAT SHARES THE HOST'S MEMORY IS NOT A SECOND POOL.** An
integrated GPU's device-local heap IS system RAM, and so is Apple Silicon's.
`UnifiedMemory()` reports it on every backend with one contract -- SUBTRACT, DO
NOT ADD. A scheduler that adds the Intel iGPU's 20.94 GiB to the host budget has
spent the same bytes twice and will thrash. Vulkan decides this by device TYPE
rather than heap flags, because an rBAR discrete card also exposes a
device-local host-visible heap and that memory is still VRAM.

**★ AND ONE PHYSICAL DEVICE ENUMERATED BY TWO BACKENDS IS ONE DEVICE, WHICH IS
THE SAME ERROR FROM THE OTHER SIDE.** This box's RTX 3050 Ti answers to CUDA and
to Vulkan, and `SpendableTotal` added both -- 4.00 GiB plus 3.81 GiB of a card
that has 4. Identity is the **device UUID** (`VkPhysicalDeviceIDProperties
.deviceUUID`, `cuDeviceGetUuid_v2`; both specifications say they are the same
sixteen bytes, and on this hardware they are:
the same 32 hex digits from both). **NEVER the device NAME** -- two
identical cards share one, so a name match merges two real devices and deletes
half a machine's VRAM, which is the worse bug and is invisible on every box this
project has. A backend that cannot supply an identity leaves its devices
DISTINCT; an unknown identity matches nothing, including another unknown one.

For one card under several backends the DEFAULT set takes **CUDA > Metal >
Vulkan** -- the +10.2% / -14% rows above, and `choose`'s own 2468-against-1368
Gmac/s on the same card. **Dedupe governs the DEFAULT and the arithmetic, never
what a caller may ask for**: `-devices vulkan:0` still selects Vulkan, and the
duplicate is still LISTED (marked `same_device_as`) because hiding hardware is
its own bug. `docs/engineering-history/placement.md` 15u.

**★ THE ENGINE RUNS ONE WEIGHT FORMAT, AND GGUF IS NOT IT.** `model.Open`
refuses anything but a `.jlm` container and returns a `NotConvertedError`
carrying the convert command. `gguf.Open` survives for the converter (`convert.FromGGUF`) and for
nothing else on the weight path.

That is a design decision, not a preference. Two formats means two loaders, two
layouts, two sets of kernels and two answers to every question about residency
-- and it is exactly how the engine ended up with a CPU tier that could not read
a converted model, which forced every block of one onto the card and made a
model larger than the card swap all of itself through a handful of slots. Both
tiers read the device layout now: accelerators upload a page untouched and the
host reads the same page in place through `nn.MatVecPacked`.

**Paging is always on.** A page is either allocated or it is not; "the model
fits" means enough pages were allocated to hold every block, and nothing is ever
evicted because nothing has to be. There is no paging-off mode to compare
against, and a measurement that reports one is measuring a path that no longer
exists.

The test suite converts (`engine/model/jlm_helper_test.go`) because it has to run
what ships. Watch for the RULE 10 shape here: most call sites used to `t.Skipf`
on an open error, so the refusal would otherwise have turned the whole suite
green in 0.001 s.

CPU and GPU. The GPU tier is built: CUDA, Vulkan and Metal, reached through
goffi (`backend/ffi`) with no cgo, one `backend.Device` interface over three implementations.
That interface earned itself by having three; **do not add a fourth abstraction
in advance of a second implementation of anything.**

Models live on a dedicated model disk — never `~`, never `/tmp`. Tests find them through
`JITLLM_MODELS` (default: the repo's `models/`); `docs/testing.md` has it and
every model-selecting test variable.

---

## No mmap of files: the measurements, and expert pages

### ★ NO mmap OF FILES. ANONYMOUS MEMORY IS DECIDED BY MEASUREMENT.

The rule, stated by the user, is about FILES: the engine does not let the OS
manage the model file's pages, because a file mapping hands residency to the
kernel and measured slower even with madvise. Allocating memory with
anonymous mmap -- frames, expert and KV pages, buffers whose lifetime the
engine manages -- is allowed, and taken where it measures better (it keeps
those bytes out of the Go heap's goal, which is RULE 2f's failure). The frame
measurement below was about sparseness only and did not decide that.

**The model file.** `jlm` reads and does not map, and the comment in
`format/jlm/read.go` is the rate half of the reason -- ReadAt x16 at 4.60-4.73 GiB/s
against an mmap fault's 1.28-2.22, with MADV_WILLNEED recovering nothing (1.31).
The design half matters more and is less obvious: **mmap does not remove the
copy or the memory, it moves both into the kernel.** The page cache IS a copy of
the file in RAM; mapping only means someone else manages it, and eviction then
belongs to the kernel -- which is precisely what this package gave up mmap to
keep. The problem mmap appears to solve, a second copy charged to the same
cgroup, is solved by **O_DIRECT**: one copy, ours, no cache (4.00 GiB/s against
buffered's 4.59, `memory.stat file 0.00 GiB` against `+2.30 GiB`).

**The frames.** A frame is a whole PageSize -- 1049.6 MiB on Qwen3-Next-80B --
and a block whose token needs only its own weights fills 33.6 MiB of it, so
anonymous mmap looks like the way to make a sparse frame cost what it holds. It
already does. Measured, eight frames touching 33 MiB of each:

    make([]byte, n)   RSS +274.1 MiB
    anonymous mmap    RSS +264.0 MiB
    (dense would be 8200 MiB; the sparse ideal is 264)

Go allocates large objects with mmap and does not touch them -- it knows fresh
kernel memory is already zero and skips the zeroing memset -- so `syscall.Mmap`
buys ten MiB in two hundred and seventy-four.

★ **AND THE CONCLUSION DRAWN FROM THAT WAS WRONG, WHICH IS WORTH KEEPING.**
"Frames are sparse, so `n = budget / PageSize` over-charges and the budget
arithmetic is the bug" -- measured, and false. A frame is sparse only
TRANSIENTLY. Nothing evicts a chunk, so a resident block accumulates expert
chunks until its page is nearly full, and Qwen3-Next-80B at twenty frames:

    tokens   peak RSS
         2    6.66 GiB
         8   11.84 GiB
        24   18.30 GiB
        40   19.29 GiB   <- 0.94 GiB a frame, ~90% of PageSize

At steady state a resident block really does cost a page and the division is
right. The claim survived being written down and died on the first measurement
of it, which is why it is recorded here rather than deleted.

★★★ **THE EXPERT CACHE IS NO LONGER COUPLED TO BLOCK RESIDENCY: SINCE v26
EVERY EXPERT IS ITS OWN PAGE, on the user's decision** (*"2 makes
more sense, coz i saw with large models layer are >1G, if we can split that
maybe we can do better"*, and *"yes you can evolve jlm format"*). A mixture's
banks leave the block page for a third page array -- one fixed-size page per
(mixture block, expert), holding that expert's sheet of every bank -- at
`ExpDataOff + p*ExpPageSize`. Still an index, still padded, still no table.
Expert pages are evicted LEAST recently used, first; block pages keep the MRU
rule that the cyclic scan makes optimal. `jlm.File.ExpertPage` / `Sheet` reach
a sheet, `model.ensureExperts` holds a token's selection (and the block's own
page) until its MoE is done, and a device reads a bank through
`nn.Packed.Sheet`.

    Moonlight-16B-A3B Q4_K_M, CPU decode, n=64, six P-cores, -maxmem 6e9,
    ABBA, one process per sample, each binary on its own container
      A/A (v26 both arms)   0.9975
      v26 / v25             3.102  IQR/median 0.053  n=6   (11.34-12.18 against 3.78-3.89 tok/s)
      decode reads          0.890 -> 0.144 GiB/token
    Kimi-Linear-48B at 14.68 GiB, replayed routing (TestRoutingLocality):
      2.44 -> 0.28 GB/token predicted, 82% LRU hits

The refusal that stood here (*"stick to a static page size"*) is
superseded by the same person rather than reversed on a technicality: pages are
still static and fixed at conversion; what changed is that an expert, not a
block, is the page. Chunk-granular eviction inside a page remains unbuilt and
unneeded.

★ **THE COST IT MADE VISIBLE: A DENSE LEAD BLOCK SETS THE BASE PAGE SIZE.**
Moonlight's block 0 is dense (first_k_dense_replace 1), so its 51 MiB base is
PageSize for all 27 blocks while a mixture block's base is ~10 MiB -- 15.4% of
the container is padding (v25: 11.8%). Reads are unaffected (only wanted chunks
are read); the budget charges whole frames. Unbuilt; a fourth array for dense
lead blocks, or charging filled bytes, would remove it.

What remains legitimate is reading LESS into a page -- `EnsureRange` does not
change the page's size, its address, or its cost, only which of its bytes are
fetched (14.8x amplification down to 1.28x). Fragmenting the page's MEMORY is
what the no-mmap rule rules out.

★ **AND "STORE THE USED SIZE SO A READ FETCHES ONLY WHAT IS USED" IS PRICED AND
NOT WORTH A MECHANISM -- 0.04%.** It was proposed and priced. Reads are already
range-based, so the padding at the end of a page is not read; what IS read of it
is the 1 MiB CHUNK ROUNDING on each block's last touched chunk. Counted with
strace on the container fd (`pread64`, resumed lines included -- a parser that
drops `<unfinished>`/`<resumed>` pairs drops every concurrent read and
undercounts by 30x):

    tinyllama decode   368 calls   679,796,736 bytes  <- chunk-rounded, exactly
    the same, exact used bytes                       664,723,456
    the same, whole pages                            785,309,696

| | reads | whole pages | chunk slop |
|---|---|---|---|
| Qwen3-Next-80B | 46.50 GiB | 49.61 | 17.8 MiB (**0.04%**) |
| Qwen3-30B-A3B | 17.70 | 18.81 | 14.3 MiB (0.08%) |
| tinyllama | 0.63 | 0.73 | 14.4 MiB (2.22%) |

So of tinyllama's 115.0 MiB of padding the engine reads 14.4 and never touches
100.6. Capturing the rest means clamping a run's end to the block's used extent,
which needs NO bytes in the page and no container change -- the extent is
`max(off+len)` over the block's entries, derivable from the table at Open. It is
0.04% on the models that matter and is recorded as priced rather than refused.

★ **AND `read_bytes` FROM /proc/<pid>/io IS NOT THE MEASUREMENT.** It counts at
the BLOCK layer, so anything the buffered handle serves from page cache is
missing -- it read 664,723,456 for the run above and was mistaken for a
byte-exact confirmation that no padding is read. Only the page reads are
O_DIRECT; the dense region and the metadata are not. Count syscalls.

## Closed: ForwardBatch never faulted its blocks in

- ~~`ForwardBatch` answers with NaN under page pressure~~ **ATTRIBUTED AND
  CLOSED: IT WAS THE ONE GRAPH ENTRY POINT THAT NEVER FAULTED ITS
  BLOCKS IN.** `Forward` calls `m.pageIn(li)` at three sites and `Prefill` at
  one; `ForwardBatch` walked straight into `&m.layers[li]`.

  **On a model that FITS that is invisible, which is the whole reason it
  survived** -- the budget holds every block, nothing is ever evicted, and the
  fault is one nil compare that always finds the page there. Every gate in this
  tree passed for as long as it existed, because every gate ran a model that
  fits.

  ★ **AND IT DOES NOT FAIL AS AN UNBOUND TENSOR, WHICH IS WHY THE FIRST
  BISECT WENT TO THE WRONG PLACE.** When the block is evicted the page still
  EXISTS; it holds some other block's bytes. So `mv`'s "is not bound" never
  fires and the answer is finite garbage on a lucky day and non-finite on an
  unlucky one. Both shapes, measured against the violation at one frame:

      stories15M   NMSE **1.497e+00**   finite, plausible, wrong
      tinyllama    non-finite at logit 0
      qwen3moe     non-finite at logit 0
      qwen2vl      non-finite at logit 0
      with pageIn  **0.000e+00** on all four

  `model.TestForwardBatchFaultsItsBlocksIn` is the gate, and it carries the
  selection check that matters: a budget that happens to hold the model makes it
  a comparison of a model against itself, and a ZERO budget is UNLIMITED rather
  than empty. It compares against DECODE ON THE SAME TIGHTLY BUDGETED MODEL, so
  paging is the shared condition and the batched path is the only variable.

  ★ **THE ROUTE IN WAS TWO MODELS IN ONE PROCESS, WHICH IS WORTH KEEPING AS A
  TECHNIQUE.** Two 11.25 GiB containers under `scripts/cap 24G` cannot both fit
  `sched.MemBudget`, so the second one's blocks evict -- and that is an ordinary
  thing for an inference OS to be asked to do. A suite that only ever opens one
  model at a time never exercises the pager at all.

## Closed: Open read every page before any budget existed

- ~~Open reads every page before any budget exists~~ **CLOSED, and
  the 80B opens in 12.4 SECONDS where it was killed at 27 minutes.**
  `openContainer` called `LoadAll()` before `build()` and `cmd/jitllm` applied
  `SetPageBudget` after `Open` returned, so the reader had no frame pool during
  the one phase that touches every block. The budget is applied before the first
  page is read now (`model.WithPageBudget`, or `sched.MemBudget` by default) and
  `build()` faults each block in as it reaches it. The comment above LoadAll was
  a correct observation -- a norm is a block tensor -- and the wrong remedy: it
  made Open O(model) in RESIDENT bytes where it need only be O(model) in bytes
  READ. `TestOpenHonoursAPageBudget` asserts 1 <= frames < blocks on the
  budgeted arm before comparing anything.

## Measurements once cited in engine/model's comments

The source comments state the reason; the measurements they used to carry,
with the machine that produced them, are here verbatim.

- `State.offerRange` (`engine/model/forward.go`), a block another session
  placed: "Reading it again cost every new session the whole model from disk
  -- 122 ms of a 166 ms time to first token for Llama-3.2-1B on an RTX 3050
  Ti."
- `WithPageBudget` (`engine/model/model.go`): "It has to be known at Open, not
  after it: when Open read every page before SetPageBudget narrowed the pool,
  a 53 GiB container on a 31 GiB box allocated 53 GiB first and was killed,
  having produced nothing in 27 minutes, while llama.cpp served the same
  model at 4.77 tok/s." The fix is "Closed: Open read every page before any
  budget existed" above.

## Lossless page compression: the first filter on real pages

The question (`docs/design/external-optimization-ideas.md`, "Conversion-time
compression and where to decompress"): does the CURRENT stored representation
of a page still have enough redundancy that an independently decodable,
bit-exact encoding could pay for its decoder? This section is evidence only.
No format change, no codec chosen, no rate claim.

**Tool:** `dev/tools/pagecomp` (reads the container through `format/jlm`).
It samples 4 block pages and 4 expert pages spread over the file plus the whole
dense region ("head": embedding, output projection), takes each tensor's three
planes (qs, d, sc) as stored, cuts them into 16 KiB tiles (32 evenly spaced
tiles per plane), and compresses each tile independently so a tile is
randomly decodable. A tile that does not shrink stays plain. The ratio r is
compressed plus metadata over stored bytes, with 5 bytes of metadata per tile
(offset and codec flag) and, for the entropy bound, 128 bytes of code-length
table per lane. Codecs: DEFLATE level 1 and 9 (Go standard library), the
order-0 byte-histogram entropy bound (what a per-tile Huffman or ANS coder
could reach before its table), and each of those after a lane split (2 byte
lanes for f16/bf16, 4 for f32 and for the 32-bit d and sc words, 2 nibble
lanes for a 4-bit payload plane). The converter would keep a split plane
unsplit where splitting does not help; the table takes the better of the
columns per row, not per tile.

**Host and method:** i9-12900HK laptop, six P-core threads
(`taskset -c 0,2,4,6,8,10` under `scripts/cap 8G`), one thread per decode,
Go's `compress/flate` inflate, warm tile, timed over at least 3 repetitions.
Decode rates are therefore Go DEFLATE on one core (0.11-0.21 GB/s expanded),
which prices the codec on this host and says nothing about a generated
decoder or a GPU one. Only containers of the current version were sampled.

**Lossless ratio r (smaller is better; metadata included):**

| model (stored format) | class, type | DEFLATE-9 | split + DEFLATE-9 | order-0 bound | split + order-0 bound |
|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 Q4_K_M | dense q4s (Q4_K) | 0.899 | 0.875 | 0.899 | 0.848 |
| | dense q6s (Q6_K) | 0.812 | 0.798 | 0.821 | 0.804 |
| | head q4s | 0.767 | 0.740 | 0.835 | 0.765 |
| | head q6s | 0.592 | 0.569 | 0.701 | 0.654 |
| d1-3B Q8_0 | dense q8 | 0.819 | 0.824 | 0.830 | 0.793 |
| | head q8 | 0.741 | 0.745 | 0.798 | 0.754 |
| Moonlight-16B-A3B Q4_K_M (mixture) | dense q4s | 0.944 | 0.936 | 0.943 | 0.930 |
| | dense q5 | 0.983 | 0.982 | 0.988 | 0.985 |
| | dense q6s | 0.849 | 0.834 | 0.860 | 0.840 |
| | expert q4s | 0.962 | 0.962 | 0.962 | 0.970 |
| | expert q5 | 0.954 | 0.954 | 0.958 | 0.940 |
| | expert q8 | 0.890 | 0.893 | 0.898 | 0.875 |
| | head q4s / q6s | 0.895 / 0.776 | 0.869 / 0.746 | 0.895 / 0.795 | 0.829 / 0.748 |
| Kev-4B BF16 | dense bf16 | 0.785 | 0.772 | 0.785 | 0.681 |
| | head bf16 | 0.773 | 0.755 | 0.772 | 0.667 |
| Qwen3.5-0.8B F16 | dense f16 | 0.772 | 0.772 | 0.771 | 0.677 |
| | head f16 | 0.767 | 0.766 | 0.766 | 0.672 |
| all of the above | f32 tensors (norms, scales kept f32; 0.5-3 MiB sampled) | 0.49-0.52 | 0.39-0.40 | 0.52-0.54 | 0.36-0.39 |

Reading it:

- Packed K-quant dense and expert pages are nearly incompressible by general
  codecs: Q4_K r = 0.90-0.96, Q5 0.95-0.98, and the experts of the mixture
  0.96. DEFLATE level 1 saves nothing on most Q4_K and expert planes (r >=
  0.997, the tile stays plain). The metadata is small against these ratios
  (5 B per 16 KiB tile, 0.03%) and does not decide any row.
- Q6_K and Q8_0 are better (0.81-0.85 and 0.82-0.89), with the head
  (embedding and output projection, tensors whose values cluster) at 0.59-0.78.
- F16 and BF16 pages reach 0.77-0.79 with DEFLATE and 0.67-0.68 with a
  per-lane entropy coder after the byte split: the exponent lane carries the
  redundancy and a Huffman or ANS coder per lane finds it where an LZ coder
  does not. That is the order of the BF16 results the design note cites; it
  is not available to Q4_K.
- The split helps the bound far more than it helps DEFLATE. The only winners
  of the byte-plane split under DEFLATE are f32 tensors (r 0.50 to 0.39) and
  bf16 head tensors by 2 points; on f16 it does nothing under DEFLATE.
- The d plane (super-scales) compresses much better than the qs payload:
  Q6_K's d plane reaches r 0.21-0.31 and Q4_K's head d plane 0.59-0.79, while
  the qs planes sit at 0.79-1.00 and Q4_K's sc plane at 0.84-1.00. The d plane
  is a small share of a tensor's bytes, so the whole tensor stays near 0.90.
  Per-plane rows are in the tool's output.

**Break-even, `D > B/(1-r)`** (D in expanded GB/s; B is the link being saved;
the sampled link figures are the command-line defaults: disk 3.0, PCIe 12.0,
device memory 800 GB/s, to be replaced by the figures measured on the host
under test):

| r | disk (3 GB/s) | PCIe (12 GB/s) | device memory (800 GB/s) |
|---|---|---|---|
| 0.96 (Q4_K expert) | 75 | 300 | 20,000 |
| 0.90 (Q4_K dense) | 30 | 120 | 8,000 |
| 0.82 (Q8_0 dense) | 17 | 67 | 4,400 |
| 0.68 (BF16/F16, per-lane entropy) | 9.4 | 38 | 2,500 |
| 0.50 (f32 tensors) | 6 | 24 | 1,600 |

Against the measured single-core DEFLATE rate of 0.11-0.21 GB/s the serial
sum is a loss in every row, by two orders of magnitude on disk, which is
expected for a general LZ decoder on one thread; the cells that matter are the
required rates. Overlap changes the sum toward a maximum, so a decoder with
several cores or SMs hits the disk row first: 9.4 GB/s of expanded BF16 for a
3 GB/s disk is about 30 times the single-core DEFLATE rate and within reach of
a vectorised lane decoder on many threads, or of a GPU. A 4-bit packed page
at r = 0.90 needs 30 GB/s from the same disk, and r = 0.96 experts need 75:
for the stored formats most of this repository's models use, the filter says
the redundancy is not there. The device-memory column (thousands of GB/s)
shows the fused-consumer case cannot pay at any of these ratios: a decoder
inside the matvec would have to run faster than the memory it saves.

**Decision this filter supports.** Compression of BF16/F16 pages (about 0.68
with per-lane entropy coding) is the only row with a ratio worth a decoder
experiment, and only against disk or PCIe, for a model stored in those
precisions under an evicting budget. Q4_K and Q5 pages and expert pages
should stay plain: r >= 0.90 means a decoder fast enough to pay does not
exist on any link measured. Nothing here is a measurement of whole-request
speed, residency, eviction avoided or a GPU decoder; the near-capacity
workload the design note asks for has not been run. Not sampled: MXFP4, IQ4_NL,
TQ2 pages and KV pages (no container of the current version was at hand), and
only 4+4 pages per model with 32 tiles per plane.

**Reproduce:** `./scripts/cap 8G -- taskset -c 0,2,4,6,8,10 go run
./dev/tools/pagecomp -pages 4 -tiles 32 model.jlm` (`-tile`, `-disk`, `-pcie`
and `-hbm` set the tile size and the link figures).

## The read service: demand beats a wrong guess

The streamed bank's cross-layer prefetch used to be one goroutine reading the
whole predicted batch, and the next fill joined all of it before its own reads
began, so one slow wrong prediction delayed experts the token needed. The
join was also the lifetime guard: concurrent `EnsureRanges` calls could read
the same chunk twice.

Now `jlm.File` marks the chunks a caller is reading (`busy`), so a second
caller waits for those chunks alone and never reads them twice; a frame being
read is pinned, so eviction cannot hand it to another page. Guesses go through
`Speculate` to a fixed set of workers; a demanded read never queues behind
them, a demanded page still queued is read by its demander and the queued
record finds it filled, and the fill drops guesses nobody started
(`Speculate(nil)`). Frames held by guesses nobody has demanded are capped
(`SetSpeculation`); a wrong guess stays charged until demanded or evicted.
Gates: `format/jlm/readsvc_test.go`, each run against a violation. No timing
claim is made here.

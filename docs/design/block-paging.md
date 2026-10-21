# Block paging: a device holds N block slots and SWAPS

**Status: BUILT and gated.** This file was a SPEC; it is evidence
now. The prediction it opened with is still a prediction and is named as such at
the bottom -- everything above that is either measured on this box or is a
statement about code that exists.

**Paging is on EVERY tier now**, which is three separate answers and not one:

    a discrete card      pages by COPY, through slots, as the rest of this file
                         describes -- and the copy is now of the RAW
                         GGUF BYTES, with the bit extraction done by a kernel on
                         the card (jit_old/gpu/kernels/unpack.go, Q4_K). The host
                         repack that was 88% of a 70B token is gone from the
                         covered formats and the GGUF mapping is the backing
                         store, so that case needs no arena and no spill file
    a unified device     does not copy at all -- the driver is handed the packed
                         arena's pointer and a page-in is a descriptor update
                         (Vulkan's VK_EXT_external_memory_host, Metal's
                         newBufferWithBytesNoCopy:, both measured below)
    the HOST             keeps its own mechanism, model/ring.go, and shares the
                         POLICY rather than the machinery -- see below for why
                         merging them would be a lie about who owns the memory

## The problem, in one measurement

    Llama-3.3-70B, 39.59 GiB of weights, RTX 3050 Ti with 4 GiB:
    device sm_86 [ptx]: 2/80 blocks, 1.22 GiB resident

The card ran 2.5% of the model and idled through the other 97.5%, on precisely
the model that needed an accelerator. `jit_old/gpu/tier` DECLINED a block that did
not fit and the block fell to the host PERMANENTLY -- placement decided once, at
load. So an accelerator smaller than the model bought almost nothing.

## What landed

`jit_old/gpu/tier/page.go`. A device holds N block slots; a block that does not fit
evicts another block instead of being declined.

    slots = (budget - the permanent parts) / the WIDEST block's packed weights
    floor of minSlots = 2, below which the device declines exactly as before

**The seam did not move.** `PrepLayer` still decides per block and still returns
false; what it no longer does is decide permanently. RULE 8a holds unchanged:
`engine/model/` owns the graph, nothing under it learns what a transformer is, and
`page.go` is a budget and an eviction policy beneath the same mechanism.

**What is pageable and what is not is the whole design.** A block's WEIGHTS have
a backing store -- the packed arena, or the GGUF -- so they can leave and come
back byte-identical. A block's KV CACHE does not: it is the session's own
history and nothing else holds a copy. So the KV, the norms, the biases and the
router are PERMANENT per block, and they are what bounds how many blocks a
device can own; the weights are what the slots hold. Admitting a block therefore
shrinks the slot count, and a model whose KV alone fills the card stops being
admitted rather than thrashing.

    70B, 4 GiB card:  0.495 GiB of weights and ~32 MiB of KV per block
                      -> 80 blocks admitted, 3 slots

## The eviction policy, and why it is the opposite of the obvious one

Blocks are visited `lo..hi-1` and then `lo` again, every token. The access
pattern is therefore PERFECTLY PREDICTABLE, which makes Bélády's optimum
computable rather than estimated: the distance from the block being loaded to
any other block's next use is exactly `(li - cur) mod (hi - lo)`, and the
largest belongs to the block JUST BEHIND the cursor. **The optimal policy is
most-recently-used.**

`model/ring.go` states the same fact from the other side -- *"the layer loop is
LRU-pessimal, and it is pessimal by construction rather than by accident"*. LRU
on a cyclic scan evicts the block needed soonest and hits nothing at all; the
ring's answer was a sliding window, which on a CYCLIC scan also hits nothing.
MRU pins a prefix of `slots-1` blocks and thrashes only the last slot, so the
hit rate is `(slots-1)/blocks` instead of zero. `TestVictimIsTheBlockJustComputed`
pins it directly, because a page-in counter on a small model cannot tell the two
policies apart.

**The arithmetic is `engine/sched/cyclic.go` now, and both tiers call it** -- this
reasoning had been written twice and the two copies had reached opposite
conclusions. See "The HOST is a tier too" below. A second rule joined it there:
a block whose page-out would reclaim NOTHING is not a candidate at all, which is
what an imported block is.

## The arena is what makes it affordable WHERE THE FORMAT HAS NO KERNEL, measured end to end

★ **THE SCOPE OF THIS SECTION NARROWED AND THE TITLE SAID SO
BEFORE THE MEASUREMENT BELOW DID.** Everything here is measured on tinyllama
q3_K_M, and Q3_K has no device unpack -- its 110-byte GGUF block is not a
multiple of 4 (`kernels.UnpackWhyNot`) -- so the arena is still exactly what
makes THAT model affordable. For Q4_K the page-in no longer packs at all:
`resident()` asks the device unpack before it asks the arena, the raw bytes go
over the link and a kernel does the extraction, and the arena is never consulted.
The argument below is therefore about the five uncovered formats and about a
unified heap, not about paging in general.

Packing is three times the cost of moving:

    pack    0.60 GiB in 592 ms = 1.01 GB/s     GGUF bit layout -> payload+scales
    upload  0.60 GiB in 188 ms = 3.19 GB/s     host -> device

So a page-in that repacks from the GGUF costs 3.1x the transfer it was
scheduling. Measured on this box, tinyllama q3_K_M streaming 22 blocks through
six slots on the RTX 3050 Ti, **identical page-in counts and identical token
ids** either way:

    arena off (JITLLM_ARENA=0)   1120 repacks    2.21 tok/s
    arena on  (the default when paging is on)   0 repacks   11.96 tok/s

    both arms: 138 page-ins, 153 page-outs, 3.28 GiB moved,
               ids [1 450 7483 310 3444 338 3681 29889 13 13 29906 29889 450 5176]

`cmd/jitllm` sizes the arena from the host budget less the model's own weights
when paging is on and `JITLLM_ARENA` is unset -- the spec's own "the pager sets
it from the placement it actually made", read forwards.

## ★ AND A MODEL THAT FITS PAYS NOTHING, WHICH IS NOW A MEASUREMENT AND NOT A CLAIM

RULE 6's obligation is that the machinery costs a model that does not need it
nothing at all. Measured with the same binary and the same budget, the only
difference `JITLLM_PAGE` -- and RE-TAKEN once the device unpack
landed, because the unpack charges a STAGING BUFFER to the same budget and a
buffer that costs a block would break this row where nobody would look:

    tinyllama q3_K_M, -devices cuda:0=700M (22/22 blocks fit), n=64,
    ABBA in one pass, A/A self-control FIRST. Two whole passes (RULE 2c):

      pass 1   self-control 0.9947 (n=2)   paging ON/OFF  1.0000  IQR/med 0.011, n=12
      pass 2   self-control 1.0083 (n=1)   paging ON/OFF  0.9919  IQR/med 0.030, n=12

      every sample of every arm: 0 page-ins, 0 page-outs, 1430 block-calls
      (= 22 blocks x 65 passes), and the paging arm holds a 6.19 MiB staging
      buffer the other does not

The two passes agree to 0.8% and both dispersions are inside the gate. What
fitting buys is that eviction never FIRES; what it costs is hi-lo map lookups in
`submit`, and they do not show up. The staging does not show up either, and that
is not luck: `prepLayer` frees it rather than decline a block on a device that
cannot page, and sizes it only AFTER the block is admitted, so admission is
bit-identical to the tier before the unpack existed.

(The pre-unpack row was 0.9973 against a 1.0000 self-control, n=8. It is not
quoted as the before of anything -- these are two measurements of the same claim
on two engines, both at parity, which is what the claim asks for.)

## The structure is always built, and the BUDGET is the control

★ **PRIMING IS NOT CONDITIONAL ON NOT FITTING, AND THIS REPO ALREADY LEARNED
THAT ONCE.** `engine/model/residency.go` used to return early when the experts fit, and
its comment records why that was wrong: *"that was right for a BATCH ENGINE and
wrong for a SERVICE ... give me back two gigabytes of GPU for another model;
this model has gone quiet, park it on the CPU; now unload it. Every one of those
is a demotion of something that currently fits, so an engine that only builds
the machinery under pressure cannot answer any of them."*

So the page record, the slot arithmetic and `GPU.SetBudget` exist for every
block of every model. What FITTING buys is that eviction never FIRES:

    tinyllama q3_K_M, -vram 3G, JITLLM_PAGE=1
      22/22 blocks + the output projection, 95 slots
      0 page-ins, 0 page-outs, 1 submission per token, 172.95 tok/s

`GPU.SetBudget` is the resize entry point, and it works on a model that fits:
halving the budget mid-generation demotes blocks (their weights go back) without
changing which device OWNS them, which is why it cannot move a token id.

    -vram 3G shrunk to 250M after the first decoded token:
      7 blocks keep their weights, 112 page-ins, 127 page-outs,
      placement still 22/22 on the device, ids UNCHANGED

Below the slot floor every page goes out, `Layers` returns false, and model/'s
existing demotion path brings the KV history home and runs on the host. Nothing
in the tier knows that is what happens.

## ★ A UNIFIED DEVICE DOES NOT COPY, AND THAT IS NOT AN OPTIMISATION OF PAGING -- IT IS A DIFFERENT MECHANISM

**Measured on this box.** The Intel Iris Xe exposes ONE heap, 23.26
GiB, memory type `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED`. Its
memory IS the host's. Uploading a weight to it therefore copies host memory to
host memory, and the machine then holds the weight twice -- once in the packed
arena and once in a device buffer that is the same kind of memory.

`VK_EXT_external_memory_host` and Metal's `newBufferWithBytesNoCopy:` remove the
copy: the driver is handed the ARENA'S OWN POINTER and the buffer aliases it. A
page-in on such a device is a descriptor update.

    tinyllama q3_K_M, Iris Xe, -devices vulkan:1=200M, JITLLM_PAGE=1, -n 8

      copying (JITLLM_ARENA=0)   22/22 blocks, 0.18 GiB resident,
                                 138 page-ins, 153 page-outs, 3.28 GiB moved,
                                 1120 repacks, 175 ms upload
      importing (the default
      once paging sizes the      22/22 blocks, 0.00 GiB resident,
      arena)                     0 page-ins, 0 page-outs, 0.00 GiB moved,
                                 155 tensors WRAPPED, 0 ms upload

    ids identical in both, and identical to -devices cpu:
      [1 450 7483 310 3444 338 3681 29889 13 13 29906 29889 450 5176]

Both arms placed every block; only one of them holds the model. **The rates
those two runs printed are single unpaired runs and are not a claim** (RULE 2);
they are in `scripts/paging.sh`'s output for orientation and nowhere else.

**What the accounting has to say, or the saving is fictional.** The bytes are
spent ONCE, by the arena, in host RAM. So:

- an imported tensor charges the device and its pool **nothing**
  (`resident.bytes` returns 0), because charging it would be the double-count
  `tier.Pool` exists to prevent -- and would decline blocks to protect memory
  nobody is going to spend;
- `GPU.HostReserved` gained the arena's retention budget, so the HOST subtracts
  those bytes from its own. That term was missing entirely: the pool stopped two
  devices spending one heap twice, HostReserved told the host what the DEVICES
  had taken, and the arena -- host memory, sized by `cmd/jitllm` from the same
  budget -- was named by neither;
- a block whose page-out would reclaim nothing is not an eviction candidate
  (`victim`), which is what makes an importing device stop paging at all, and
  what stops `trim()` demoting every block on a budget it can never get under.

**The guard is `UnifiedMemory()`, not the extension.** All three Vulkan devices
on this box advertise `VK_EXT_external_memory_host`, the RTX 3050 Ti included,
and importing THERE would put the weights in host memory behind PCIe where every
matvec re-reads them: correct, and a fraction of the rate. The tier asks only
devices whose heap is the host's.

**And the arena's alignment was wrong for Apple Silicon.** `kernels.HostAlign`
was `const 4096`, argued from "the M4's 16 KiB page is a multiple of it". True of
the region's BASE and false of everything after it: the second and third arrays
start at 4096-aligned OFFSETS, and `newBufferWithBytesNoCopy:` wants whole pages
-- it returns nil, with no error object. It is `max(4096, os.Getpagesize())` now,
and both drivers need the LENGTH padded as well as the address, which is why
`kernels.PaddedBytes` exists. Verified on the M4: `page 16384, arena alignment
16384`, and a real Metal matvec reads the host's mutated bytes with no upload.

## What it costs, measured, and why it is OFF by default

    tinyllama q3_K_M, RTX 3050 Ti, -n 8, greedy, single unpaired runs:

      -devices cpu                                   85.58 tok/s
      -vram 3G      (fits, 22/22)                   172.95
      -vram 200M    paging OFF -> 5/22 blocks        40.83
      -vram 200M    paging ON  -> 22/22 blocks       11.98

**Paging LOSES here and it is expected to.** A page-in moves a block's packed
bytes over a link measured at 3.19 GB/s; the host reads the same block's GGUF
bytes from DRAM at ~37 GB/s. So on a model that fits in HOST memory -- eighteen
of the nineteen on this box -- paging a block costs roughly 16-20x what leaving
it on the host costs. Switching it on by default would be RULE 6's one refusal:
a clear regression.

The case it is built for is the other one. On Llama-3.3-70B the HOST is itself
reading from NVMe at 1.94 GiB/s (`model/ring.go` measured exactly that), against
the link's 3.19 -- and that is where the prediction below came from.

★ **THAT COMPARISON IS WRONG AND THE 70B ROW IS WHY.** It reads the two rates as
ALTERNATIVES, and they are in SERIES: the bytes the link carries come off the
same NVMe first, whether a host packer faults them or the driver's memcpy does.
Measured, the raw upload moves 40.10 GiB a pass at 1.61 GB/s -- the disk's rate,
not the link's 6.68 -- so paging a model bigger than host RAM cannot be faster
than the host reading it, only slower by less. So streaming is a placement a
caller asks for per block (`model.Place.Stream`, `-placement N=DEV~`), and the
automatic placement keeps a block that does not fit on the host for decode --
verbatim what `model/ring.go` says about itself, for the same reason. There is
no device-wide paging switch any more: a flag that could only be right for
every block at once or wrong for every block at once is what a per-block
placement replaced.

## The HOST is a tier too, and it did NOT become the same mechanism -- one thing did

`model/ring.go` pages host layers on a budget and was the obvious candidate to
fold into `page.go`. It stayed separate, and the reason is not inertia:

    the device pager OWNS its resident set   it allocates, it frees, a page-in
                                             can FAIL and the failure reaches
                                             model/ as a decline
    the host ring only ADVISES               madvise(MADV_DONTNEED) on a mapping
                                             the KERNEL owns; nothing can fail,
                                             and what the ring does not hand back
                                             the kernel reclaims by its own LRU
    the unit differs                         packed device-layout bytes against
                                             GGUF file ranges -- same block index,
                                             different bytes, different backing
                                             store

Merging them would mean one of the two pretending to own memory it does not.

**What WAS one thing, written twice, and the two copies disagreed: the eviction
POLICY.** `engine/sched/cyclic.go` now holds it, and both tiers call it. The device
pager had derived it correctly -- a cyclic scan makes Bélády's optimum computable
rather than estimated, and the furthest next use belongs to the block JUST BEHIND
the cursor, so the optimal policy is MOST-recently-used. The host ring shipped a
sliding window, which is LRU with the eviction brought forward, and on a cyclic
scan **that hits nothing at all**:

    80 blocks, a 52-block window, 3 tokens, simulated with the kernel's own
    reclaim in the model (model/ringpolicy_test.go):

      MRU / Bélády      100 hits of 240 visits, 49 blocks pinned, peak 52 resident
      sliding window      4 hits of 240 visits, peak 53 resident

The sliding window's four hits are boundary effects, and its peak is OVER the
window it was given -- it was scoring by overcommitting at the token boundary,
which the kernel then corrects by evicting the block about to be used. On
Llama-3.3-70B under `cap 26G` that is the difference between re-reading 39.59 GiB
from NVMe every token and re-reading roughly 14.

**No existing ring gate could see this.** `TestRingIsBitIdentical` proves the ring
cannot change an output bit -- and it cannot, whatever it evicts, because a
discarded clean page re-faults to the same bytes. `TestRingWindowArithmetic`
proves the window is the right SIZE. Both are true of a window holding the wrong
blocks, and the ring has no hit counter because the page cache does the hitting.

**The ring is still opt-in behind `JITLLM_RING`.** Its measured regression on a
model that FITS is unchanged by this, and what it is worth on the 70B is RULE 2's
paired interleave and belongs to the main loop.

## ★ THE 70B ROW, RE-TAKEN WITH THE DEVICE UNPACK -- ALL 80 BLOCKS, 2.35x SLOWER

`scripts/paging-70b.sh`, one binary, one pass, `cap 24G`, `-devices cuda:0=2G`,
`-n 4`, greedy. The budget is PINNED because free VRAM moves with whatever the
desktop is drawing, and 2G is where this card reproduces the recorded 2/80:

    paging off    2/80 blocks   1.14 GiB resident    10 block-calls run
    paging on    80/80 blocks   1.85 GiB resident   400 block-calls run
                  388 page-in(s), 465 page-out(s), 200.49 GiB moved,
                  0 served from the arena, 2580 tensors UNPACKED ON THE CARD
                  (179.67 GiB of raw GGUF bytes) and 696 repacked on the host

The placement counters are IDENTICAL to the pre-unpack engine, to the byte:
`--violate no-unpack` is the same binary with `JITLLM_GPU_UNPACK=0` and it reads
388/465/200.49/400 as well. What moved is 0.12 GiB of resident -- the 126.00 MiB
raw staging buffer -- and it cost ZERO slots: three either way.

    ids identical in both, and identical to -devices cpu:
      [128000 791 6864 315 9822 374 264 3363 430 374]

**The 40x is `tier.Stats.Blocks`, not a placement line, and that distinction is
the whole gate.** 400 = 80 blocks x 5 passes, counted after each submission
returned and its output was read back. "80/80 blocks placed" is ALSO true of a
pager that swapped every block in and then declined every compute -- a run that
places everything, prints every page counter, answers in English off the host
and matches the ids. `--violate declines` builds exactly that (the budget
shrunk below the slot floor mid-generation) and it reads 44 block-calls where a
whole run reads 154, with the ids still identical. Token ids are NECESSARY and
they are NOT SUFFICIENT.

Two independent passes of the whole script gave byte-identical counters -- 388
page-ins, 465 page-outs, 200.49 GiB, 400 block-calls -- so the placement and the
policy are deterministic, which is RULE 2c's "run the whole comparison twice"
answered for the counters rather than for the rate.

**AND IT IS 2.35x SLOWER, PAIRED AND INTERLEAVED** (`scripts/paging-rate.sh`,
ABBA in one process-pair per round, A/A self-control FIRST). **The quoted row is
at `-n 8`, TWO rounds, which is four per-round ratios, and the length is part of
the number** (RULE 1):

    self-control (both arms paged)   1.0020     ratios 0.9943, 1.0096
    paging ON against paging OFF     0.4251     IQR/median 0.061, n=4
                                                ratios 0.4140 0.4173 0.4329 0.4434
    every A sample   700 page-ins, 720 block-calls
    every B sample     0 page-ins,  18 block-calls

★ **AND THE SAME COMPARISON AT `-n 4` THE SAME AFTERNOON WAS REJECTED, SO IT IS
NOT QUOTED AS A ROW.** Three rounds, six ratios, median 0.4614 at IQR/median
**0.106** -- over RULE 2's gate. The whole of the spread is in the HOST arm
(0.0476-0.0554 tok/s against the paged arm's 0.0222-0.0234): that arm
demand-faults 39 GiB through a page cache the other arm has just evicted, and
four tokens is not enough of it to average. Doubling the sample took the
dispersion to 0.061 and moved the median by 8%, which is the same medicine the
row needed at n=2 -> n=4 the first time it was taken. The rejected median is
recorded because it AGREES -- 0.42-0.46 twice -- not because it counts.

★ **THE SLOT BIAS IS GONE, AND THAT IS A RESULT OF THE UNPACK RATHER THAN OF THE
HARNESS.** The self-control on this comparison used to read **0.9533**: the first
slot of an ABBA was consistently 4.7% slower because a 39.59 GiB model does not
fit in 31 GiB of host RAM and no two samples started with the same page cache,
and the row had to be bias-corrected to 0.313. Today it reads **1.0014** at n=4
and **1.0020** at n=8, twice, unprompted. The raw path allocates NOTHING on the
host -- `nn.Weight.Data` is a slice of the GGUF mapping and `Buf.Write` hands it
straight to the driver -- where the host packer allocated ~1.33x the tensor per
pack and evicted the page cache the next sample was about to want. No correction
is applied to 0.4251 because the control says none is owed.

★ **THE FIRST MEASUREMENT OF THIS ROW WAS ALSO REJECTED, AND THE REASON IS WORTH
KEEPING.** At n=2 the pre-unpack comparison read 0.2861 at IQR/median **0.198**
-- over the gate, correctly, and it was not quoted. And the A/A before it had
been worse than useless: the harness was reading the rate off cmd/jitllm's
`%.2f` tok/s field, which at 0.01 tok/s is ONE significant figure, so every
sample rounded to the same two digits and the self-control came back at exactly
1.0000 with an IQR of exactly 0.000. `scripts/paging-rate.sh` computes the rate
from the DURATION now (millisecond-rounded, seven digits on a 280-second
sample) and refuses to print an IQR over fewer than four ratios.

★ **THE REPACK IS NO LONGER THE ROW. THE NVMe READ IS, AND BOTH ARMS PAY IT.**
The decomposition is SUBTRACTED now rather than estimated: `cmd/jitllm` prints
pack, upload and device-unpack in the load banner and all four terms again at the
end, so each line below is (end - load) / passes with nothing modelled. The unit
is a PASS over the eighty blocks, because prefill streams them exactly as a
decode step does -- 5 passes at `-n 4`, and `scripts/paging-70b.sh` prints this
table itself:

    per pass, 77.6 page-ins x 0.517 GiB = 40.10 GiB landed on the card

                      unpack ON      --violate no-unpack (the 60e530b engine)
      host pack        11.68 s              58.74 s      <- 696 Q6_K tensors vs 3276
      upload           26.77 s               6.45 s
      device unpack     1.20 s               0.00 s      <- enqueue; see below
      block compute     0.39 s               0.42 s
      ---------------------------------------------
      accounted        40.05 s              65.60 s
      observed         41.38 s              66.64 s
      unaccounted       1.34 s (3%)          1.04 s (2%)
      the host arm     18.09 s/token        18.50 s/token
      this arm         41.52 s/token        69.67 s/token

**The 15 s that used to be "unaccounted" was mostly the pack being slower than
the load-time rate said**, not a term of its own: the old block estimated the
repack at 47.9 s by multiplying a load-time 0.90 GB/s by a per-token byte count,
and subtraction says 58.74. Both arms now close to within 3%.

**The two arms are one `JITLLM_GPU_UNPACK` apart, in the same binary, and their
page-in counters are identical to the byte** -- 388/465/200.49 GiB/400
block-calls -- so this is a decomposition of one mechanism and not two engines.

★ **WHERE THE 20 s WENT: THE UPLOAD TERM ABSORBED THE NVMe READ.** Removing the
repack did not remove the disk. A packed upload moved bytes from a host buffer
the packer had already faulted in, at 6.68 GB/s; a RAW upload hands the driver a
slice of the GGUF mapping, so the demand faults happen INSIDE the memcpy and the
term reads **40.10 GiB at 1.61 GB/s**. Add it up and the model closes:

      NVMe read of 40 GiB, queue of one     ~18-20 s   both arms, and the host arm
      host pack arithmetic   40 s -> 11.7 s            what has no kernel yet
      PCIe upload, warm                       ~6.4 s
      device unpack                           ~1 s
      device compute                          ~0.4 s

`model/ring.go` measured the same disk from the other side -- 39.04 GiB/token at
**1.94 GiB/s** demand-faulted, one goroutine -- and the host arm here reads
39.04 GiB/token in 18.09 s, i.e. 2.16 GiB/s. **The link was never the competitor.
The read is in SERIES with it**, so a pager that re-reads the model every pass
cannot beat a host that re-reads the model every token; the most it can do is
stop adding to it.

★ **THE DEVICE UNPACK TERM IS AN ENQUEUE, NOT A DEVICE TIME.** `cuLaunchKernel`
on the legacy stream returns before the kernel runs, so `TUnpack` is what the
host spent handing the launch over and the extraction itself runs while the next
memcpy on that stream waits -- which puts it inside the upload term. The bound is
the per-tensor A/B in `kernels/arena.go`: 47.4 GB/s read+written, under a second
for the 40 GiB a pass moves. It cannot be the row either way, and the upload
figure above carries it.

★ **AND THE PER-TENSOR 4.51x IS NOT THE WHOLE-MODEL 1.61x, FOR A REASON THAT
MATTERS.** `kernels/arena.go` prices one 132 MB tensor at 4.51-4.58x against the
host packer, uploading the raw bytes at **7.47 GB/s** -- because that benchmark
repeats one tensor forty times in one process and its pages are WARM. On a model
4.6x the size of host RAM every page is cold, and the same upload reads 1.61.
A page-in rate quoted without saying whether the source was in the page cache is
not a number, the same way one quoted without its block size is not (2.98-3.19
GB/s was tinyllama's 24 MiB blocks; 6.68 is the 70B's 0.517 GiB ones, warm).

★ **AND THE OVERLAP IS STILL NOT THE FIRST BLOCKER.** Device compute is 0.39 s a
pass against an upload of 26.77, so hiding compute behind transfer is worth
nothing. What IS worth something is queue depth on the READ: `convert/gguf/warm_linux.go`
faults the same mapping at 2.10 GB/s from one goroutine and **4.27 GB/s from
eight**, and the raw upload is a queue of one by construction. Prefaulting a
tensor's pages in parallel before handing the slice to the driver is the next
lever on the dominant term, and the machinery exists.

★ **AND BELOW 2G THIS CARD STOPS BEING A PAGER AT ALL.** `minSlots` is 2 and a
block is 0.517 GiB, so the working range on a 4 GiB card carrying a 70B is
2 to 3 slots and nothing else. At `-devices cuda:0=1G` the tier lands on ONE
slot, declines 545 tensors, and places 1 block of 80 -- correctly, loudly, and
exactly as it did before paging existed. The budget is what decides whether the
card runs the model at all, and on this model the knob has two settings.

## Open, in the order they are worth taking

- ★ **PARALLEL FAULTING UNDER THE RAW UPLOAD. This is now the whole row.** The
  upload is 26.77 s of a 41.38 s pass and 18-20 s of that is the NVMe read the
  driver's memcpy takes at a queue depth of ONE. `convert/gguf/warm_linux.go` reads the
  same mapping at 4.27 GB/s from eight goroutines against 2.10 from one, so
  touching the tensor's pages from several goroutines before `Buf.Write` is the
  lever, on the term that dominates. Nobody has run a point of it.
- **The Q5_K and Q6_K unpack kernels.** 696 tensors a pass still take the host
  packer for 11.68 s -- 28% of the pass -- purely because their GGUF block is not
  a multiple of 4 bytes and the IR addresses buffers by element index. Q5_K is
  word-aligned and merely unwritten; Q6_K needs the two-word merge. See
  `kernels.UnpackWhyNot`.
- **THE ARENA ON A MODEL BIGGER THAN HOST RAM is no longer the row, and that is
  a withdrawal, not a demotion.** It said: *"the repack is 47.9 s of a 69.4 s
  token and removing it would take the row from 0.26 to roughly 2.8"*. The
  repack went without an arena at all -- the GGUF mapping is the backing store
  -- and the row went to 0.4251, not 2.8, because the estimate credited the
  upload with 6.4 s it only achieves on WARM bytes. The arena still earns its
  place for the five uncovered formats and on a unified heap (see
  `kernels/arena.go`), and `JITLLM_ARENA=<bytes>` is still the knob; what it is
  not is the thing standing between this row and parity.
- **Prefetch depth, and overlap SECOND not first.** The old list had overlap at
  the top on the strength of the 1.56x; the measurement above says device compute
  is under 1 ms a block, so there is nothing for the upload to hide behind until
  the repack is gone. Both belong after the arena.
- **Is the 6.74 GB/s upload a PCIe limit or an unpinned-staging limit?** Pinned
  host memory is the usual 2-3x. It is worth less than it looks while the repack
  is 88% of the token, and it is the next thing after that.
- **Graphs and paging do not compose.** A recording holds device addresses and a
  page-in allocates fresh buffers, so a paging device does not record at all.
  On the 70B that is ~450 launches re-encoded per token.
- **The head is not pageable.** `PrepHead` charges the output projection
  permanently, so on a paged model it competes with slots rather than swapping
  through one. On Qwen3-30B-A3B that is 243 MiB read in full every token.
- **Cross-device eviction.** Two devices page from one arena and each owns its
  own slots; nobody arbitrates between them.
- **The unified row is unmeasured as a RATE.** Everything above about importing is
  a placement and an accounting fact. Whether an Iris Xe running 22 of 22 blocks
  out of host memory beats the same laptop's six P-cores is RULE 2's paired
  interleave through `scripts/vs-llamacpp.sh`, and nobody has taken it.
- **The arena is not on by default on a unified device, and it probably should be.**
  There, retaining a pack is not a second copy of the weights -- it REPLACES the
  device buffer, so the same bytes are held once instead of twice. `cmd/jitllm`
  only sizes the arena when paging is on, so a plain `-devices vulkan:1` run still
  copies. Changing that default changes the shipping path on the M4 and needs a
  measurement first.
- **CUDA does not import and the decision is deliberate, not missing.**
  `cuMemHostRegister` would give a discrete card a pointer into host memory, and
  every matvec would then read its weights over PCIe. If a Jetson or a Grace
  Hopper ever runs this, that is where the CUDA half belongs.

## The gates

    jit_old/gpu/tier/page_test.go   six gates, each run against a violation in
                                   the same function, failure text logged
    jit_old/gpu/tier/import_test.go five gates on the zero-copy path: what is
                                   wrapped, what is charged, what the pager does
                                   with a block it cannot reclaim, what SetBudget
                                   does to one, and the arena's term in
                                   HostReserved
    jit_old/gpu/backend/import_test.go  the hardware gate, on every device the host
                                   has. The assertion is a MUTATION of the host
                                   bytes with no upload of any kind: an imported
                                   buffer's next launch must see it and an
                                   uploaded one must not -- the copy arm is the
                                   violation, not a control
    jit_old/gpu/metal/import_test.go    -contents must BE the pointer handed in,
                                   plus the four shapes the driver refuses
    jit_old/gpu/kernels/arena_test.go   every packed array starts AND ends on a
                                   HostAlign() boundary; the length half is what
                                   Apple Silicon needs
    engine/sched/cyclic_test.go        the policy, simulated: MRU keeps a prefix and
                                   LRU keeps nothing
    model/ringpolicy_test.go    the host ring's choice, and the sliding window
                                   it replaced, scored against each other
    scripts/paging.sh              the same seven on real hardware and a real
                                   model, --violate ids|no-paging|fits-tiny|
                                   no-shrink|no-arena|no-import
    scripts/paging-70b.sh          the 70B claim itself, re-runnable: the block
                                   counts before and after, the page-ins against
                                   the slot arithmetic, the ids against
                                   -devices cpu AND the EXECUTION counter beside
                                   them, WHERE THE BITS WERE EXTRACTED, the four
                                   terms of a page-in against the seconds that
                                   elapsed, and the budget as the control.
                                   --violate ids|no-paging|budget|declines|
                                   no-unpack|accounting, each seen to fail, and
                                   a violation now has to break the assertion it
                                   AIMS at rather than any assertion at all.
                                   The first four read counters rather than
                                   weights and are proved on tinyllama in a
                                   minute; the last two are proved on the 70B,
                                   because a decomposition of tenths of a second
                                   decides nothing and the script says NOT
                                   DECIDED rather than green
    scripts/paging-rate.sh         the paired rate, ABBA in one pass with the A/A
                                   self-control first -- the one measurement here
                                   that CANNOT be in-process, because the two arms
                                   differ in where the model IS and placement is
                                   decided at load

★ The trap both are built around is a gate that passes because NOTHING PAGED.
"22/22 blocks placed" is also true of a tier that accepted every block and kept
six; "the text is English" is true of a run that declined everything and used
the CPU. So every does-not-fit assertion reads the PAGE-IN counter and every
fits assertion reads the PAGE-OUT counter -- the two numbers that are wrong in
exactly the state that looks right.

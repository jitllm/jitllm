# CPU kernels — evidence

> **EVIDENCE, NOT INSTRUCTIONS.** This file records what was measured, what was
> tried, and what was wrong. It is the reason a rule exists, not the rule.
> **Binding requirements live only in `AGENTS.md`** -- if something here reads as
> an instruction, `AGENTS.md` is what governs. Numbers here are dated by the
> commit that added them; treat any of them as provisional until re-measured
> (RULE 1a: a baseline pinned to a version rots silently).
>
> Split out of a single 415 KB AGENTS.md after a review
> identified the file's real failure as *unresolved authority* -- historical
> findings, current instructions and withdrawn conclusions all speaking in the
> same imperative voice.


## ★ AN f16 KV CACHE IS FREE ON amd64 AND A REGRESSION ON arm64

Built: `kvLayout.elem` (4 or 2 bytes), `gguf.EncodeHalf`, and an element-width
parameter on all eight attention emitters. The cache stays a `[]float32` holding
PACKED halves -- every offset the layout hands out is 4-byte aligned by
construction, since headDim is a multiple of 8 and the strides are multiples of
headDim -- so not one call site's slice expression changed.

**★ THE WIDTH IS CHOSEN BY `nn.KVWidthPaysOff`, NOT CONFIGURED.** It emits both
kernels at load and takes f16 only when the f16 pair is no more than 5% larger:
amd64 measures -2% to +4% across hd 64/128/256 and arm64 +21% to +28%, and
nothing observed lands between. The same code therefore answers f16 on one
architecture and f32 on the other, which is the whole argument for a JIT --
`JITLLM_KV_F16=0/1` still forces it for an A/B.

**That is deliberately NOT an instruction count predicting a rate** (this file is
three-for-three against that, RULE 5z). It is a tie-break: a kernel that does not
grow AND reads half the bytes cannot be slower. When it DOES grow the trade is
real and needs a measurement -- and the one measurement says arm64 loses.

★ SO amd64 NOW DEFAULTS ON AND ITS RATE IS UNMEASURED. The box was in RULE 2d's
clamp all night. The correctness gates pass and the reasoning is sound; the
number is owed.

**★ arm64 IS WHY THERE IS A DECISION AT ALL.** M4, tinyllama, ABBA, 4 rounds,
`-depth` as the starting context:

    depth     f16 vs f32     IQR
        0       0.9972      0.002
      512       0.9973      0.001
     1536       0.9832      0.012      <- 1.7% SLOWER
    SELF-CONTROL (f32 vs f32) at 1536   0.9985   IQR 0.004

**The byte argument said this should WIN and it loses, increasingly with
depth.** The emitted kernels say why in one line:

    amd64   scores hd=64   198 -> 194 bytes   (-2%)     acc  223 -> 219  (-2%)
            scores hd=256  606 -> 626         (+3%)     acc  913 -> 945  (+4%)
    arm64   scores hd=64   268 -> 332         (+24%)    acc  308 -> 372  (+21%)
            scores hd=128  460 -> 588         (+28%)    acc  596 -> 724  (+21%)

**VCVTPH2PS WIDENS EIGHT HALVES FROM MEMORY INTO A YMM, so amd64 issues exactly
what it issued before and reads half the bytes. arm64 HAS NO SUCH LOAD**: LDRd
plus FCVTL is two instructions where LDRq was one, so every f16 attention kernel
is ~25% larger. At depth the attention walk is the loop, and on the M4 that walk
is issue-bound rather than bandwidth-bound -- which is RULE 5ab's finding
("the Mac CPU gap is ALL harness") arriving from a different direction.

★ SO THE TRADE IS PER ARCHITECTURE, AND THE JIT IS WHAT MAKES THAT EXPRESSIBLE:
the cache's element type is a codegen input, constant from NewState, so it is
baked rather than branched. What is NOT yet known is the amd64 number -- that box
sat at 845-1470 MHz idle all night (RULE 2d's clamp) and could not produce a
gated result. The instruction counts predict neutral-to-positive there and **an
instruction count is not a rate** (RULE 5z, three for three).

★ THE GATE IS BIT EQUALITY ON EXACTLY-REPRESENTABLE INPUT, which separates the
two questions that otherwise compound. "Does an f16 cache change the answer" is
"does the kernel READ halves right" plus "do halves LOSE something", and an NMSE
test against float64 measures their sum -- so an addressing bug hides inside the
rounding budget. `TestAttnF16MatchesF32OnExactHalves` fills the cache with values
that ARE halves, making the second term zero, so any difference at all is the
first. Violations: an f32 stride on an f16 cache fires, f32 load offsets fire.

★ AND IT IS CPU-ONLY, REFUSED RATHER THAN SKIPPED. MigrateKV hands the device
`s.k[li]` as `[]float32` and the device reads f32; with an f16 cache those bytes
are packed halves, so the device would attend over reinterpreted garbage --
fluent, wrong, and invisible to any gate that does not run the device.
`prepDeviceRange` declines outright, which is RULE 8a (a feature reaching one
tier and not the other stops the block being a unit) and RULE 7f (refuse loudly).

★ WHAT IT STILL BUYS, INDEPENDENT OF SPEED: half the KV bytes, so twice the
context per byte of RAM. RULE 5r reached the same verdict for the device --
"its value is CAPACITY, and capacity only binds where VRAM is separate and
scarce". That is the argument for keeping it behind a flag rather than deleting
it, and it is not an argument for turning it on.

## ★ AN AUDIT FOUND Mixtral's F16 ROUTER ON THE FLOAT64 ORACLE — 1.41x

"Are we falling back to `nn` at all?" had only ever been answered for ONE model.
`TestNothingFallsBackToNN` answers it for every model on the box -- Forward,
Prefill and ForwardBatch, all three counters -- and
`JITLLM_AUDIT_MODEL=<substring>` lifts the 2 GiB cap for one file, because that
cap skips exactly the models where a missing kernel hides.

Eleven small models and four large ones came back clean. **Mixtral-8x7B did
not: 192 matvecs per (Forward + 5-token Prefill), which is 32 layers x 6
tokens -- one per layer per token, i.e. the ROUTER.**

    jitllm info:  F16  32 tensors  2,097,152 bytes  0.01%

32 x 8 experts x 4096 x 2 bytes. **Mixtral's `ffn_gate_inp` is F16 where every
other mixture's is F32**, and `cpu.Supported` had F32 and not F16, so all 32 took
`nn.MatVec`: single-threaded, float64, `make([]float64, 4096)` per call.

    JITLLM_PROFILE=1, 6 tokens, six P-cores
      before   matvec 90.2%   router 9.5%
      after    matvec 99.4%   router 0.1%

    whole token, ABBA, alternating binaries, 3 rounds
      before  5.64 5.27 | 5.50 5.20 | 5.00 5.18
      after   7.78 7.66 | 7.32 7.41 | 7.01 7.36
      per-round ratios 1.4152 / 1.3766 / 1.4116, median 1.4116, spread 0.027

**★ THE PROFILER SAID 9.5% AND REMOVING IT IS WORTH 41%, WHICH IS RULE 14s's
UNCLOSED ACCOUNTING REPEATING EXACTLY.** There the F32 router fix was worth
about three times the router time the profiler attributed to it, and the standing
hypothesis was `nn.MatVec`'s per-call `make([]float64, k)` churning the heap and
evicting the model's hot pages. Here that is 32 x 32 KB per token against a
20 GB model -- the same shape, the same over-delivery, a second data point for a
hypothesis that is still unverified but now twice observed.

★ THE KERNEL IS ONE INSTRUCTION DIFFERENT FROM THE F32 ONE. `VCVTPH2PS` widens
eight f16 from memory straight into a YMM, so the loop, the four accumulator
chains and the reduction are unchanged and only the weight cursor's stride
halves. arm64 uses `FCVTL`/`FCVTL2` on one `LDRq` of eight halves -- encodings
verified against clang (0x0E217A84 for `fcvtl v4.4s, v20.4h`), the way `BIC16b`
was, because a wrong Rn field does not fault, it widens a different register.
Both are baseline: F16C ships with AVX2 on every part whose VNNI check follows,
and FCVTL is ARMv8-A.

★ THE GATE IS NMSE AGAINST float64 ON RANDOM f16 BIT PATTERNS, not on f32 values
rounded down -- generating the half directly makes the reference exact without an
encoder and sweeps subnormals and the whole exponent range, which rounding a
normal distribution barely touches. Run against two violations: the F32 weight
stride fails one shape, an overlapping load fails five.

★ AND llama.cpp CANNOT LOAD Mixtral ON THIS BOX, so there is no oracle row for
it -- `llama-completion would not run this model` at 20 GB. What stands in for it:
the kernel's own float64 gate, and the fact that **the token ids are IDENTICAL**
before and after, so the f32 accumulation selected the same experts as the f64
oracle on that prompt. That is weaker than an independent implementation and it
is what is available.

## ★ THE SOFTMAX PADDING VALUE WAS exp's CLAMP, AND ONLY qwen2 COULD SHOW IT

The generated softmax pads its row to a whole vector so the kernel needs no
masked load and no scalar epilogue. The pad value was `expClampL`, -87.3 -- the
point below which `exp` underflows to zero IN ABSOLUTE TERMS.

**That is the wrong property.** Softmax subtracts the row's maximum first, so a
pad is inert only while every REAL score is above it. qwen2-1.5B's attention
scores reach **-872**:

    row = [-872.46], padded to [-872.46, -87.3 x7]
    max          -87.3        <- the PADDING
    real entry   -872.46 + 87.3 = -785, clamped to -87.3, exp = 1.2e-38
    padding      -87.3 + 87.3 = 0,                        exp = 1
    sum          7.0   ->   the one real entry normalises to 1.74e-39

A single-element softmax must return exactly 1. `TestGreedyMatchesLlamaCpp` went
PASS to FAIL on qwen2 and on nothing else, and the free-running text was "the
capital of France is the capital of France" -- fluent, and wrong.

It is `-FLT_MAX` now, which is inert unconditionally: below every finite score,
so it can never be the maximum, and its own exp argument is then -FLT_MAX, which
the clamp takes to zero. Both kernels seed their maximum from the ROW rather than
from a constant, so the pad value and the seed cannot drift apart.

**★ AND THE SAME KERNEL HAD A SECOND BUG THE SAME GATE COULD NOT SEE.** The
horizontal maximum folded 256 -> 128 with `VEXTRACTF128`, then used `VPERMQ 0x4E`
-- which swaps 128-bit HALVES, repeating the fold just done, where the step needed
the 64-bit swap WITHIN the low lane (`VSHUFPS 0x4E`). Lane 0 ended up holding
`max(a0,a1,a4,a5)`, and it also folded in the zeroes `VEXTRACTF128` leaves in the
upper half, so the "maximum" was floored at 0.

**SOFTMAX IS SHIFT-INVARIANT, WHICH IS WHY BOTH SURVIVED AN NMSE GATE OVER
LENGTHS 1, 2, 3, 7, 8, 9, 13, 31, 32, 33, 64, 127, 512 AND 1537.**
`exp(x-m)/sum(exp(x-m))` is the same for ANY m, so a wrong m is arithmetically
harmless until the shift pushes something past exp's clamp -- and then it is
wrong by TYING the saturated entries, not by drifting. **The distinguishing input
is MAGNITUDE, not length**, and a length sweep is exactly the test a careful
person writes for a padded kernel.

Two gates that can see it:

    TestEmitSoftmaxMaxIsTheRealMax   two saturating entries per lane pair, and it
                                     checks their RATIO. The first version used
                                     ONE peak -- which still normalises to ~1
                                     under a wrong maximum -- and PASSED against
                                     the bug it was written for.
    TestEmitSoftmaxDeepNegativeRow   rows entirely below the clamp, at -872,
                                     -1e5 and -3.4e30.

★ THE FIRST-VERSION FAILURE IS THE REUSABLE PART. A gate written for a specific
bug, run against that bug, and passing, is a gate that measures something else --
RULE 10's shape with the violation actually available. Run the violation before
believing the gate, not after.

★ AND THE PADDING DOES NOT READ AS EXACTLY ZERO AFTERWARDS: exp's low clamp
floors at exp(-87.3) = 1.2e-38, so the pad lanes hold that. 38 orders below the
mass the real entries carry, and no caller reads past `n`. The assertion is
`< 1e-30`, not `== 0`, and saying why is cheaper than chasing it twice.

## ★ RULE 5: pin the cores, and know which ones. SIX PHYSICAL P-CORES. MEASURED.

`JITLLM_CORESET` selects a pool layout and `JITLLM_CORES` caps the count, so the choice
stays falsifiable. Paired and interleaved on a 256 MB DRAM-resident matvec,
against six physical P-cores as the baseline:

    pe   (6 P + 8 E, 14 threads)   0.864x   IQR 16.4%   UNSTABLE
    smt  (both threads of 6 P)     0.965x   IQR  8.4%
    all  (everything, 20 threads)  0.807x   IQR 27.2%   UNSTABLE

**More cores is worse.** E-cores drag decode down, hyperthreads are neutral to
slightly negative, and using everything is worst of all. Note also that the IQR
climbs with thread count -- extra threads buy variance, not throughput, which is
its own reason to refuse them.

Measure this PAIRED. Sequentially, the same six-P-core configuration read
41.99 tok/s in one batch and 30.29 in the next; that is thermal drift, not a
result, and it briefly suggested SMT was a 17% win when paired testing shows it
is a 3.5% loss.



P-cores are CPUs 0-11 with SMT pairs {0,1},{2,3},...; E-cores are 12-19. Six *distinct* P-cores is
`0,2,4,6,8,10`. Decode binds 4-6 P-cores — E-cores add zero read bandwidth and become the straggler
behind every barrier. Prefill binds 6P+6E (E-cores are worth +34% there). **Never all 14**: the spin
barrier goes 620 ns -> 11.9 us when nothing is left for the OS.

## ★ RULE 5b: measure the CONFIGURATION THAT SHIPS, not the one that isolates cleanly.

Interleave width, swept two ways, gave opposite answers:

    one core, 127 MB DRAM-resident, ratio vs the single-row kernel
             pack2   pack4   pack5   pack7   pack8
    Q8_0     1.152   1.329   1.416   1.443   1.351      <- peaks at 7
    Q4_0     1.319   1.284   1.277   1.259   1.308      <- peaks at 2

    SIX cores, gemma end to end, tok/s
    pack1 18.60   pack2 24.24   pack4 21.59   pack6 20.87   pack7 20.73   pack8 20.98

Shipping the per-format ONE-CORE optima measured 20.63 tok/s, worse than uniform
pack2 at 24.24. The single-threaded microbenchmark was not noisy -- it was
answering a different question.

The mechanism is concurrent read streams: N interleaved rows are N streams per
worker and 6N across the pool, so pack8 is 48 streams competing for the hardware
prefetchers and the dTLB where pack2 is 12. On one core nothing competes and only
the instruction count matters, so wider always looks better.

This is the same shape as the 4-row interleave result (+24% at one core, neutral
at six) and the pack8-plus-unroll result. Three times now, a win measured in
isolation has evaporated or reversed under the real thread count.

**The stream hypothesis is no longer inference. It was measured with hardware
counters** (`sudo sysctl kernel.perf_event_paranoid=1`, plus a perf JIT map that
`JITLLM_PERFMAP=1` writes to `/tmp/perf-<pid>.map` so generated code resolves by
symbol instead of landing in `runtime._ExternalCode`). gemma, six P-cores,
pack8 against pack2, six interleaved rounds:

    instructions              0.833   pack8 issues 17% FEWER -- interleaving works
    cycles                    1.068   and it loses anyway     IQR/median 0.016
    L1-dcache-load-misses    ~2.56x   direction 6/6 rounds, magnitude NOT gated
    mem_load_retired.l3_miss ~1.62x   direction 6/6 rounds, magnitude NOT gated

Identical weight bytes either way. The difference is that pack2's arrive by
PREFETCH -- non-blocking, and invisible to a retired-load counter -- while
pack8's arrive as demand loads that stall. `perf annotate` agrees: **61% of all
cycles sit on the two `VBROADCASTI128` weight loads**, and the second row's
costs 3x the first's. So the constraint is neither issue bandwidth nor DRAM
bandwidth; it is prefetcher COVERAGE, and interleaving buys instructions at the
cost of the thing actually on the critical path.

The two miss ratios are reported without a gate on purpose: pack2's own miss
count fell monotonically 86.0 -> 48.1M across six identical runs while pack8's
stayed flat, so IQR/median is ~0.2 and the magnitude is not a number to quote.
The cause of that drift is unidentified (THP is `madvise` here, so it is not
that). The direction never wavered.

**★ 5d. arm64's K-QUANT TAIL IS NOW INTEGER, AND IT IS WORTH 14.4%.** A k-quant sub-block's dot
product and its scale are BOTH integers -- `sum = d * Sum_s sc[s]*dot_s` with `sc[s]` six bits -- so
the whole inner sum is exact in int32 and the super-block's `d` applies once. The kernel was
converting eight or sixteen times a super-block instead: `SCVTF4s`, `LDR S`, `FMLA`, a three
instruction tail on a 4-cycle float chain, per sub-block. One `MLA vd.4s, vn.4s, vm.s[i]` replaces
all three, reading four scales at a time out of the scratch slot they already occupied.

    tinyllama q3_K_M, M4, four P-cores, alternating binaries, n=6
      before  70.60 69.90 69.76 69.37 69.16 69.01
      after   80.63 79.73 79.65 79.76 79.27 79.32
      median of per-round ratios 1.144, IQR/median 0.006

**gemma reads 1.000 in the same run, and that is the point of including it.** gemma is Q4_0+Q8_0 and
has no k-quants at all, so a change to the k-quant tail CANNOT touch it; a negative control that
moves is a harness that is drifting. This is a cross-process A/B, which RULE 6's own history says is
unusable at 3% -- it is trustworthy here only because the effect is 14%, the spread is 0.6%, and the
control is flat. Do not read that as permission to use two processes for a small change.

Three things it rests on, all pre-existing:

1. **`Spec.ActWin` had already hoisted the ACTIVATION scale.** With `actWin` 256 every sub-block of a
   super-block shares one `d_a`, so the only per-sub-block float left was `d*sc[s]` -- and that is the
   one this removes. Neither lever reaches 14% alone; RULE 6a's "which other lever it combines with"
   is not hypothetical.
2. **`MLA` by element keeps all four lanes**, so there is no horizontal reduce. llama.cpp's version
   (`isum += vaddvq_s32(dot) * scale[j]`) reduces each sub-block to a scalar first; keeping the lanes
   and reducing once at the end is strictly less work.
3. **Q3_K and Q6_K get two integer accumulator chains and Q4_K/Q5_K get one**, because the min term
   (`-dmin*m[s]*sum(a)`) is float and needs the second register. tinyllama is 63% Q3_K+Q6_K, so the
   formats that pay for the second chain are the ones that do not need it.

Overflow was checked per format rather than assumed: the widest case is Q6_K at 16256 per lane times
a 127 scale times 16 sub-blocks, or 33M against int32's 2.1B.

**amd64 does NOT get this**, and the reason is a real ISA difference rather than an oversight:
`VPDPBUSD` is unsigned x signed, so the 16-wide formats need per-16 activation sums and `ActWin`
stays 32 there. Whether a `VPMADDWD`-based tail could reach the same place is unmeasured.

**★ 5h. arm64's Q3_K IS ISSUE-BOUND ON ITS UNPACK, AND THE PER-FORMAT TABLE SAYS SO IN ONE LINE.**
`jit/cpu/cmd/fmtbench` prices each format's decode matvec against a read wall measured in the SAME
process, every set sized 280 MB so nothing is measured out of cache. On the M4 at four P-cores:

    format   GB/s   of wall     bytes per 256 elements
    Q8_0     90.8    121%        272     (the wall probe understates; RULE 8b)
    Q6_K     72.9     97%        210
    Q4_0     72.0     96%        144
    Q4_K     66.3     89%        144
    Q5_K     63.1     84%        176
    Q3_K     45.7     61%        110    <- and it was 41.9 / 57% before this

**Q6_K and Q3_K have the SAME per-sub-block instruction count and the same sixteen sub-blocks**, so
the sub-block count is not the variable -- Q6_K reads 210 bytes where Q3_K reads 110, so per BYTE
Q3_K issues 1.9x the unpack. The measured ratio was 100/57 = 1.75. That is the whole diagnosis, it
needs no profiler, and it is the same shape of argument that found Metal's int multiply (RULE 12j):
hold the memory system fixed and vary only the format.

**tinyllama is 53% Q3_K by weight and about 65% by TIME**, which is why the Mac's worst row is its
CPU row.

What came out: the high bit and the zero point are one subtraction. Q3_K's value is `qs + 4*hbit - 4`,
so it is qs minus 4 exactly when hbit is CLEAR, and `4 &^ shifted` is that amount -- one `BIC`. The
old shape masked the bit (AND), added it (ORR) and subtracted the zero point separately (SUB): three
instructions where two do. It also puts the notorious polarity -- a SET high-mask bit means do NOT
subtract 4, the most commonly mis-implemented line in ggml -- into the instruction rather than into a
comment.

    Q3_K kernel      41.9 -> 45.7 GB/s   (57% -> 61% of the wall)
    tinyllama Mac CPU vs llama.cpp   0.7627 -> 0.7934, IQR/median 0.010

`TestGreedyMatchesLlamaCpp` passes on the M4 for both models against llama.cpp itself, and the token
ids are byte-identical to the pre-change build -- which is the check that matters, because a wrong
polarity here does not fault, it answers fluently.

**Still 0.79, so this is a step and not the answer.** The remaining 39% is the unpack itself: ~6
instructions per 16 elements against 110 bytes. Anything further has to remove instructions per
ELEMENT, not per sub-block, and the obvious candidates (hoisting the shifted hmask per group,
widening to 32 elements) come out level on a count -- measured reasoning is in the commit.

**★ 5i. THE M4's REMAINING CPU GAP IS 15% HARNESS AND 9% KERNEL, AND IT TOOK THREE READINGS TO GET
THERE.** tinyllama on the M4, all figures soaked and median-of-15:

    jitllm kernels alone, weighted by tinyllama's format mix   48.7 GB/s
    jitllm whole model                                         42.2   87% of its own kernels
    llama.cpp whole model                                   53.1  109% of jitllm's kernels

So the 26% is not one thing. About 15 points is jitllm losing ground between its own kernels -- of which
the profile attributes 6.6% to non-matvec ops it genuinely has to do -- and about 9 points is
llama.cpp's kernels simply being faster than jitllm's.

**Two wrong readings preceded that, both caused by the measuring tool.**

1. **"% of wall" is not comparable across formats, and I treated it as a verdict.** Q3_K reads at 58%
   of the wall where Q6_K reads at 98%, which looks like a broken kernel. Counting the GENERATED
   INSTRUCTIONS settles it: Q3_K issues 224 per super-block against Q6_K's 232, for the same 256
   elements. Identical work per element; it simply carries fewer bytes while doing it. Per BYTE, Q4_0
   is heavier than Q3_K (2.22 instructions against 2.04) and runs at 96%. **Density sets GB/s and
   does not set efficiency.** Compare instructions per ELEMENT, or one format against itself on
   another machine.
2. **`fmtbench` took best-of-9 after one warm-up, which is the boost clock**, while the model runs
   sustained -- this project's own RULE 2c, broken by this project's own tool. Soaked, the kernel
   prediction fell 53.5 -> 48.7, and the "21% lost in the harness" I had read off the difference was
   partly two numbers at different clocks.

**i8mm is NOT the missing decode lever, and the arithmetic says so before any code is written.** The
M4 reports FEAT_I8MM and jitllm emits no `SMMLA`, which looks like a gap. `SMMLA` computes a 2x2 int32
tile from 2x8 by 8x2 int8 -- 32 MACs against `SDOT`'s 16. But decode has ONE activation vector, so
one of the two columns is waste: 32 MACs for 16 useful ones, exactly `SDOT`'s ratio. It is a PREFILL
instruction, where there are many activation vectors, and llama.cpp uses it the same way. Do not
"add i8mm" to the decode kernel expecting 2x.

**And interleaving is worth far less on arm64 than on amd64, for a reason the instruction counts
give.** amd64's pack wins ~1.3x because it amortizes SIX per-block activation-side instructions
(RULE 8b). The arm64 k-quant sub-block spends about ONE -- a single `LDR Q` of the activation --
so sharing it across two rows saves roughly 3.6%, not 30%. `BestPackNative` returning 1 on arm64 is
therefore a smaller omission than it looks; measure before building the interleaved emitter.

**★★ READ 5ab FIRST. EVERY arm64 KERNEL RULE BELOW (5k, 5l, 5n, 5p, 5t, 5v, 5x, 5y) COMPARES THE TWO
ENGINES AT ONE CORE, AND AT THE FOUR THREADS THAT SHIP THE KERNELS ARE DEAD EVEN.** jitllm's Q3_K reads
45.1 GB/s and llama.cpp's 45.2 at four threads; the 11-13% one-core deficit those rules are built on
does not exist at the thread count anything runs at. They are kept because their METHOD is sound and
several of their sub-findings stand on their own -- the disassemblies, the vector/scalar exchange
rate, the shipped instruction removals -- but **do not start kernel work from them.** The Mac CPU gap
is harness: jitllm reaches 75% of its own kernels where llama.cpp reaches 89% (RULE 5ab).

This ordering hazard is itself the lesson. These rules accumulated over several sessions, each
building on the last, and not one of them measured the reference kernel at the thread count that
ships -- because `scripts/ref` was single-threaded and nobody noticed that the harness, not the
question, was setting the comparison. **When a series of findings all rest on one shared setup, price
the setup before extending the series.**

**★ 5k. llama.cpp's Q3_K KERNEL IS 13.5% FASTER THAN jitllm's ON ONE M4 CORE, AND jitllm IS THE ONE WITH
BETTER IPC.** `scripts/ref/q3kbench.c` compiles llama.cpp's own `ggml_vec_dot_q3_K_q8_K` NEON body
and times it exactly as `jit/cpu/cmd/fmtbench` times jitllm's: one thread, 280 MB DRAM-resident, 1.5 s
soak, median of fifteen.

    jitllm Q3_K       12.6 GB/s
    llama.cpp Q3_K 14.3 GB/s      +13.5%

**A static instruction count predicted the opposite**, which is why this benchmark exists rather than
the reasoning: jitllm issues ~80 unpack instructions per super-block against llama.cpp's ~76, the same
sixteen `SDOT`s, and NO horizontal reduces where llama.cpp pays sixteen `vaddvq_s32`.

**Where it actually goes, from jitllm's own per-format timings:**

    format   GB/s   bytes/super-block   ns/super-block
    Q3_K     12.6         110                8.73
    Q4_K     22.3         144                6.46
    Q6_K     25.1         210                8.37

**Q3_K and Q6_K cost the SAME time per super-block while Q6_K carries 1.9x the bytes.** The cost is
per SUPER-BLOCK, not per byte -- a fixed overhead Q3_K pays over half the bytes, which is the entire
reason it reads 58% of the wall while nothing is wrong with its inner loop. Chasing that 58% as a
kernel defect wasted a round.

Budget, per super-block: 144 instructions in the sixteen sub-blocks (9 each) and **80 in the fixed
part**. At ~35 cycles that is 6.4 IPC against llama.cpp's ~150 instructions in ~31 cycles, or 4.8.
**jitllm issues better and executes more**, so the lever is COUNT and it is in the fixed part:

  - **~32 instructions widening sixteen 6-bit scales to int32**, which exists so `MLA` by element can
    keep all four lanes and skip the horizontal reduce (RULE 5d). llama.cpp made the opposite trade --
    scales stay bytes, sixteen `vaddvq_s32` and a scalar multiply -- and on this chip the opposite
    trade wins. RULE 5d's reasoning was sound and its conclusion is chip-specific.
  - **~16 `MOVI` zeroing the dot accumulator**, one per sub-block, because `SDOT` accumulates into its
    destination. llama.cpp folds this into `vdotq_s32(vzero, ...)`.

Both are structural, not peepholes: reversing the first means abandoning `MLA` by element for Q3_K,
and the second follows from it. Measure with `scripts/ref` before and after; the two kernels are now
comparable on one core with everything else held fixed.

**★ 5v. llama.cpp's Q3_K DISASSEMBLED AT LAST, AND IT KILLS ONE OF THE TWO CANDIDATES RULE 5k
NAMED.** `scripts/ref/q3kbench.c` already compiles llama.cpp's NEON body; making `vec_dot_q3_K`
non-static and `noinline` and running `otool -tV` costs one command, and it had never been done --
every count for llama.cpp in this file until now was read off the SOURCE. RULE 5p learned exactly
that lesson on jitllm's side and it was not carried across. Mnemonic histogram:

    16 sdot.4s   16 addv.4s   16 madd      25 movi.2d   24 ldp
    24 and.16b   18 ushr.16b  16 sub.16b   60 mov       48 add

**llama.cpp emits 25 `movi.2d`, not one.** RULE 5k proposed jitllm's "~16 `MOVI` zeroing the dot
accumulator, because SDOT accumulates into its destination" as one of the two structural costs to
attack, on the reading that llama.cpp "folds this into `vdotq_s32(vzero, ...)`". It does not -- the
compiler materialises a zero per sub-block there too. **jitllm's 16 accumulator zeros are not a gap, and
that candidate is closed.**

The other difference is real: **24 `ldp` against jitllm's 38 single `ldr`/`ldur`** -- llama.cpp loads
register PAIRS. But RULE 5q already measured the multi-register load on jitllm's sixteen activation
loads at 0.9996, thirteen instructions removed for nothing, because those are hot in L1 and were
never waiting. Pairing the rest is the same bet on the same evidence, so it needs a reason to be
different before it is worth building.

**TREAT THE ABSOLUTE COUNTS AS UNUSABLE, THOUGH.** The standalone build spills (16 `str`, 13 `stp`)
and makes 24 `bl` calls, so it is not the fully inlined kernel llama.cpp actually runs; only the
vector mnemonics in the inner loop are comparable. A fair total needs the function compiled in
context, which is a bigger job than one command.

**★ 5l. CLOSING THE KERNEL GAP ENTIRELY WOULD NOT FLIP THE MAC's CPU ROW, AND THAT IS THE NUMBER
THAT SHOULD DIRECT THE NEXT SESSION.** `scripts/ref` now times llama.cpp's Q6_K as well as its Q3_K,
which is the control the Q3_K measurement needed:

    kernel, one M4 core, 280 MB, soaked       jitllm    llama.cpp   ns/super-block gap
      Q3_K                                   12.7      14.3            0.97
      Q6_K                                   25.4      29.2            1.08

**A uniform ~1 ns -- about four cycles -- per super-block on both**, despite Q6_K carrying 1.9x the
bytes. That is a FIXED per-super-block cost, which points at the scale prep, and it is the same
answer the ns/super-block table gave from jitllm's side alone (RULE 5k). Two independent routes to the
same place is worth more than either.

But the arithmetic that matters is this one. jitllm's model runs at 87% of its own kernels where
llama.cpp's runs at essentially 100% of its, so:

    kernel gap closed by       jitllm kernels    jitllm model    vs llama.cpp
      nothing (today)             48.7          42.2          0.795
      13% (the measured gap)      55.0          47.7          0.898
      20% (better than theirs)    58.4          50.6          0.954

**Even a perfect kernel would leave the row at 0.90-0.95.** The Mac needs BOTH halves -- the ~13%
kernel and the ~13% jitllm loses between its own kernels -- and neither alone is enough. Anyone starting
here should know that before spending a day on `emitA64KScales`.

What the scale-prep fix would actually require, since it is the obvious next move: llama.cpp does its
6-bit scale unpack in SCALAR registers (`utmp` bit-twiddling on GPRs) where jitllm does it in VECTOR
ones, so theirs overlaps with the vector unpack instead of competing for the same ports. jitllm's A64
emitter has almost no scalar bit manipulation -- no `LDR W`, no `AND` immediate, no `LSR`, no `ORR`
shifted-register -- so this needs about five new instructions including ARM64's logical-immediate
encoding, verified against clang the way `BIC16b` was. It is a real piece of work for at most half
the row.

**★ 5m. SOFTWARE PREFETCH DOES NOTHING FOR THE SINGLE-ROW K-QUANT KERNEL, AND THE COUNTERS SAY WHY
BEFORE THE A/B DOES.** RULE 5b's hardware counters put the constraint on prefetcher COVERAGE -- pack8
issues 17% fewer instructions than pack2 and loses anyway, with ~2.56x the L1 misses, because pack2's
lines arrive by hardware prefetch and pack8's arrive as demand loads. `PREFETCHT0` was added to the
emitter for exactly that experiment. Emitted into the k-quant block loop and swept over distances
1, 2, 4, 8 and 16 super-blocks, then the best candidate run soaked and ABBA:

    prefetch 1 super-block ahead vs none:  0.9986,  IQR/median 0.068,  n=12

**Nothing.** And `perf` explains it without the A/B: the kernel runs at 3.02 instructions per cycle
on a 6-wide Golden Cove core, so it is not issue-bound, and one core pulls ~8.1 GB/s, which is near
what one core gets from DRAM. **The single-row kernel walks ONE sequential stream and the hardware
prefetcher covers it completely.** RULE 5b's finding is about EIGHT interleaved streams; it does not
transfer to a kernel with one, and reading it as "prefetch will help the k-quants" was reading a
result outside its conditions.

Where the experiment does still belong: amd64's INTERLEAVED emitter, where the 2.56x L1 misses were
actually measured. Untested there, because Q4_0 and Q8_0 already read 94-95% of the wall on that box
and there is little left to win.

**★ 5n. THE M4 KERNEL IS AT 80% OF ISSUE WIDTH, SO ANY FIX MUST REMOVE INSTRUCTIONS -- MOVING THEM
BETWEEN PORTS MAKES IT WORSE.** This is the number that should gate every future idea about the
arm64 k-quant kernel, and it was the last thing measured rather than the first.

                  instrs   cycles   IPC   % of an 8-wide core
      jitllm Q3_K       224       35   6.4          80%
      llama.cpp      150       31   4.8          60%

**jitllm runs at HIGHER IPC and loses**, because it issues 33% more instructions. The obvious fix -- do
the 6-bit scale unpack in scalar registers as llama.cpp does, so it stops competing with the vector
unpack for ports -- was priced before being written and is NEGATIVE: replacing ~22 vector
instructions with ~31 scalar ones takes 224 to 233, and on a kernel at 80% of issue width more
instructions of ANY kind cost about 4%. Ports are not the constraint; issue width is.

Contrast Linux, where `perf` on the same kernel reads **3.02 IPC on a 6-wide core** -- 50% of width,
not issue-bound at all, and ~8.1 GB/s on one core, near what one core pulls from DRAM. **The same
kernel is issue-bound on one host and memory-bound on the other**, which is why an idea has to be
priced per machine and why the prefetch experiment (RULE 5m) measured zero on Linux.

So the only thing that helps arm64 is FEWER INSTRUCTIONS, and jitllm's per-sub-block sequence is already
within one or two of llama.cpp's. The remaining excess is in the per-super-block prologue, and the
part of it that is structural is the int32 widening `MLA` by element requires -- which is the trade
RULE 5d made deliberately, and which buys back sixteen horizontal reduces. Reversing it is roughly
instruction-neutral by hand count, so it needs building and measuring, not arguing.

**★ 5y. BOTH KERNELS DISASSEMBLED PROPERLY, AND THE PREMISE EVERY arm64 RULE RESTS ON IS WRONG.**
The llama.cpp side had only ever been ESTIMATED FROM SOURCE. Isolating its actual super-block loop --
the body between the backward branch at 0x3a8 and its target at 0x34 -- gives 222 instructions
against jitllm's 216 (RULE 5p). Split by which pipe they occupy:

                          jitllm    llama.cpp
      VECTOR (NEON)       154       122     <- and.16b 24, ushr.16b 18, sub.16b 16,
                                               sdot 16, movi 16, addv 16, shl/mvn/bic/add.16b 16
      scalar (integer)    ~19        63     <- madd 15, add 15, asr 12, and 8, sxtb 4, sub 4, orr 4
      loads               38         15     <- ldp 12 (PAIRED), ldr 2, ldur 1
      fmov transfers       0         16
      TOTAL              216       222

**jitllm issues FEWER instructions in total and is 13.5% slower, because 154 of them sit on the pipe
that costs 2.47x (RULE 5x).** llama.cpp spends 63 instructions on the integer pipes where jitllm spends
19, and it buys 32 vector instructions with them.

**THIS RETIRES RULE 5n's TWO HEADLINE NUMBERS.** "jitllm issues 33% more instructions" came from
comparing jitllm's disassembled 216 against llama.cpp's ESTIMATED ~150; the real figure is 222, so jitllm
issues 3% FEWER. And "jitllm runs at HIGHER IPC and loses" inverts: at ~220 instructions each and
llama.cpp 13.5% faster, llama.cpp is at ~7.2 IPC against jitllm's 6.4. It is not winning on fewer
instructions or worse IPC -- it is winning on PIPE BALANCE, which is the one axis RULE 5n's model
could not see.

**So the direction is settled and it is the opposite of what this file has said since RULE 5n: move
work from the vector pipes to the integer pipes, even at a worse instruction count, up to about
2.5:1.** The candidates are visible in the table -- jitllm's 38 loads against llama.cpp's 15 (it uses
`ldp` PAIRS), and the ~32 vector instructions of unpack that llama.cpp does with `asr`/`and`/`orr`
on GPRs. RULE 5q's LD1x4 null result does NOT contradict this: it moved loads to other loads, not
off the vector pipes.

**The target is 32 vector instructions**, worth 32 x 0.569 = ~18% of the token at RULE 5x's rate,
against a measured gap of 13.5-20%. That is the first accounting in this file that closes.

**★ BUT IT IS DIFFUSE, NOT ONE CATEGORY, AND SAYING OTHERWISE WOULD MISLEAD THE NEXT ATTEMPT.**
Lining the two histograms up category by category:

      unpack (and/ushr/sub/bic/shl/mvn/add.16b)   jitllm 86   lcpp 74   +12
      movi                                        jitllm 24   lcpp 16   + 8
      scale apply (jitllm mla+sshll / lcpp addv)      jitllm 22   lcpp 16   + 6
      misc vector                                 jitllm 22   lcpp 16   + 6

No single change recovers 32. And the scale row is a TRAP: jitllm spends 22 vector where llama.cpp
spends 16 vector, but llama.cpp also pays 16 `fmov` transfers, 15 `madd` and 12 `asr` for it -- at
RULE 5x's rate that is ~24 against jitllm's ~12.5, so **jitllm's `MLA`-by-element is the better trade and
must be kept**. RULE 5d and RULE 5t are confirmed a third time, now from llama.cpp's own
disassembly.

So the realistic target is the ~12 in the unpack and the ~8 `movi`, not 32 -- call it 20 vector
instructions, or ~11% of the token.

**★ ONLY ~13 OF THOSE 32 CAN ACTUALLY MOVE, BECAUSE A VECTOR INSTRUCTION DOES SIXTEEN LANES AND A
SCALAR ONE DOES ONE.** This is the constraint that bounds the whole idea and I reached for the
exchange rate twice before applying it. Work on PER-ELEMENT data cannot be moved to the integer
pipes at 1:1 -- it moves at 1:16, which at 2.47:1 is a 6.5x LOSS. Only work on quantities that are
naturally scalar-sized can move, and in this kernel that is the SCALES: sixteen 6-bit values per
super-block, which is exactly the work llama.cpp does with `asr`/`and`/`orr` on GPRs.

Going through the 32 with that filter:

    unpack        +12   PER-ELEMENT. Cannot move. Only a better algorithm removes these, and
                        llama.cpp's hmask handling is 2 ops per sub-block, same as jitllm's BIC fold.
    movi           +8   STRUCTURAL. SDOT accumulates into its destination; llama.cpp emits 16 too.
    scale apply    +6   jitllm already WINS here once llama.cpp's 16 fmov + 15 madd + 12 asr are
                        counted -- keep MLA by element (RULE 5d, 5t).
    misc           +6   unattributed.

**So the movable quantity is the ~13 vector instructions of SCALE EXTRACTION, going to ~24 scalar:
-13 x 0.569 + 24 x 0.230 = -1.9% of the token.** Not 11%.

    kernel 1.9% + fusion 6.15% = 8.0% of the token  ->  0.8276 / 0.920 = 0.899

**THE ROW IS THEREFORE NOT WINNABLE WITH ANY LEVER MEASURED HERE, AND THE 1.0035 BELOW WAS WRONG.**
It rested on treating all 20 as movable. RULE 5l's original verdict -- 0.90-0.95 with a perfect
kernel -- turns out to be right, and by a route it did not have: not because the kernel cannot
improve, but because most of jitllm's vector excess is per-element work with nowhere to go.

**What would change the answer is an algorithmically cheaper UNPACK, not a re-balancing.** There is
exactly one such candidate and it is now costed. arm64 applies Q3_K's -4 zero point IN THE KERNEL
(the `BIC` fold, RULE 5h) where amd64 moves it to the activation side as a `-4*sum(a)` correction
(RULE 6). Doing the same on arm64 deletes the sixteen `SUB`s -- 16 x 0.569 = 9.1% -- but the
correction has to come back somewhere: with integer accumulation the total becomes
`d * [SUM sc[s]*dot_s - 4*SUM sc[s]*sumA_s]`, and that second term is a 16-element dot product of the
scales (already in registers, RULE 5o) with the per-16 activation sums. That is ~9 vector ops per
super-block, so the net is about **-7 vector instructions, or 4.0%**. `AHalfSum` already exists as
int32 for the GEMM path; the decode path is passed the float `AHalf`, so it needs plumbing, not
invention.

**★ AND THEN A THIRD COEFFICIENT REOPENED IT: A LOAD COSTS 0.515%, NEARLY AS MUCH AS A VECTOR OP.**
I had assumed loads were cheap because RULE 5q removed twelve of them for nothing. `JITLLM_LOAD_PROBE=N`
emits N dummy `LDR q` from the constant block, and the three coefficients measured together on one
run are:

    dummy per block     +0      +16     +32     +64     time/instr
      LOAD (LDR q)     82.00   77.50   71.72   61.68     0.515%
      VECTOR (ADD4s)   81.82   76.61   71.29   59.89     0.569%
      SCALAR (ADD Xn)  81.85   79.35   76.60   71.76     0.230%

**jitllm issues 38 loads per super-block against llama.cpp's 15, and llama.cpp gets there with twelve
`ldp` PAIRS.** At 0.515% those 23 extra loads are ~12% of the token.

★ THIS ALSO EXPLAINS RULE 5q, WHICH IS WHY IT IS BELIEVABLE. `LD1x4` measured 0.9996 -- twelve
instructions removed for nothing -- and that reads as "loads are free" only if you assume the
instruction count is the work. **`LD1` multi-register is MICROCODED on Apple silicon**, so four
`LD1x4` decode to about the same uops as sixteen `LDR q`: the count fell and the work did not.
`LDP` is a single uop that loads two registers, which is a different thing entirely. RULE 5q tested
the wrong instruction and drew a general conclusion from it.

**RE-COSTED WITH THE LOAD COEFFICIENT:**

    LDP pairing (16 activation loads -> 8, plus adjacent weight loads)   ~5.7%
    scale extraction -> scalar pipes                                      1.9%
    -4 -> activation side                                                 4.0%
    whole-layer fusion (RULE 6, M4)                                       6.15%
    ---------------------------------------------------------------------------
    total ~17.8% of the token  ->  0.8276 / 0.8225 = 1.006

**So the row is at parity, not at a 0.94 ceiling -- and the previous paragraph in this rule said
otherwise ten minutes earlier.** The difference is not new arithmetic on old numbers, which is how
this rule went wrong twice before; it is a coefficient that had never been measured. `LDPq` is not
in the emitter yet and is one encoding to add, verified against clang the way `BIC16b` was.

1.006 is the EDGE, every term is an estimate, and the four overlap. Build the load pairing first: it
is the largest, the most contained, and it is the one that tests whether this coefficient means what
it appears to.

**★ 5z. IT WAS BUILT, AND IT SAYS THE COEFFICIENT DOES NOT MEAN THAT. A PROBE MEASURES THE MARGINAL
COST OF ADDING, WHICH IS NOT THE SAVING FROM REMOVING.** `LDPq` is in the emitter (encoding verified
against clang) and the 32-wide formats now load their two adjacent activation quads in one
instruction -- eight loads per super-block gone, same registers, same order. On the M4 at one core,
Q3_K and Q6_K are `wregs == 1` and therefore untouched CONTROLS:

    Q4_K   22.2 22.2 22.2 22.2 22.1  ->  22.4 x6      +1.0%
    Q5_K   19.3-19.4                 ->  19.3-19.4    flat
    Q3_K   12.6 (control)            ->  12.6         flat
    Q6_K   25.3 (control)            ->  25.3         flat

**The probe predicted 8 x 0.515 = 4.1% and the measurement is 1.0% -- a 4x overestimate**, from a
change whose controls are perfectly flat and whose token ids are byte-identical on both models.

The mechanism: `JITLLM_LOAD_PROBE` adds 64 loads to a kernel that already issues ~38, tripling the
pressure on the load units, so its slope is the marginal cost AT AN ELEVATED OPERATING POINT.
Removing eight from the existing 38 moves along a much flatter part of the curve, because those
loads were already overlapped with the vector work that dominates. **The probes answer "what does
one more cost?" and every lever in the table above asked "what does one fewer save?" -- which is a
different derivative.**

**SO THE COSTED TABLE ABOVE IS OPTIMISTIC, PLAUSIBLY BY SEVERAL FOLD, AND 1.006 SHOULD NOT BE
QUOTED.** The one entry that has been built came in at a quarter of its estimate. The three that have
not -- scale extraction to scalar, the -4 move, fusion -- were priced the same wrong way, except
fusion, which was measured end-to-end by `TestFusionCeiling` and is therefore the only one of the
four that is a real number.

**The durable lessons are the two coefficients as RATIOS, not as absolute prices**: a vector op costs
~2.5x a scalar one and a load is in the same band as a vector op, so PREFER SCALAR WORK. Those hold.
Using them to predict the size of a saving does not.

The LDP change is kept: it is +1.0% on 37% of a tinyllama, ids are identical, and it costs nothing.

★ THE COMPOSITION ARITHMETIC BELOW IS STILL THE RIGHT SHAPE and is kept for it: two levers that each
remove a fraction OF THE TOKEN are additive in TIME SAVED, not multiplicative in rate. Both remove a fraction OF THE SAME TOKEN, so they are additive in TIME
SAVED and not multiplicative in rate. From 0.8276:

    conservative  20 instr = 11.4%  + fusion 6.15%  = 17.5% of the token  ->  1.0035
    optimistic    32 instr = 18.2%  + fusion 6.15%  = 24.4% of the token  ->  1.0941

    (the wrong way, 0.8276 x 1.114 x 1.0615, gives 0.9785 and says "do not bother")

**So the Mac CPU row is winnable and the two levers are jointly sufficient**, where a moment earlier
this rule said they were not. It is at the EDGE on the conservative end -- 1.0035 with the 20 being
my estimate and the 6.15% carrying an 81.5% IQR -- so it wants both levers built, not one.

RULE 5l's "even a perfect kernel would leave the row at 0.90-0.95" is not contradicted: it priced the
KERNEL alone against a 13% harness it had no number for. RULE 6's Mac column is that number.

**★ 5x. A VECTOR INSTRUCTION COSTS 2.47x A SCALAR ONE ON THE M4, SO RULE 5n's MODEL IS WRONG AND THE
SCALAR SCALE-UNPACK IS BACK ON.** RULE 5n priced every idea by INSTRUCTION COUNT -- "on a kernel at
80% of issue width more instructions of ANY kind cost about 4%" -- and on that basis rejected moving
the 6-bit scale unpack into scalar registers, the way llama.cpp does it, as +9 instructions and
therefore negative. That model is only right if total issue WIDTH binds. Two probes settle it,
`JITLLM_SCALAR_PROBE=N` and `JITLLM_VECTOR_PROBE=N`, which inject N dead adds per block round-robin over
three registers (throughput, not a dependency chain); scalars go to X11-X13 and vectors to v29-v31,
all outside every allocation the kernel makes:

    dummy adds per block      +0      +8     +16     +32     +64     time/instr
      SCALAR (ADD Xn)       80.95   79.90   78.14   75.70   70.58     0.230%
      VECTOR (ADD4s)        81.39   79.05   75.86   70.75   59.65     0.569%

**Scalar instructions are not free, but they cost 40% of what a vector one does.** The binding
resource is the NEON pipes, not issue width -- which is exactly why llama.cpp's disassembly is full
of `mov`/`add`/`movk`/`madd`/`asr` (RULE 5v) while jitllm's mix is nearly all vector, and why
llama.cpp finishes sooner at 4.8 IPC than jitllm does at 6.4.

Re-pricing 5n's own proposal with the measured coefficients: its "~22 vector instructions for ~31
scalar ones" becomes -22 x 0.569 + 31 x 0.230 = **-5.4% of the token, or +5.7%** -- where the count
model said -4%.

**★ BUT THAT 22 IS 5n's NUMBER AND I HAVE NOT VERIFIED IT, SO TREAT +5.7% AS THE OPTIMISTIC END.**
Counting what `emitA64KScales` actually emits for Q3_K, the EXTRACTION is about twelve vector ops
(TBL 2, plus AND/USHR/ORR/SHL) and the six `SSHLL`/`SSHLL2` widens STAY either way -- the scales must
still reach vector registers as int32 for `MLA` by element, which is the trade RULE 5d made and
RULE 5t confirmed is worth keeping. On that accounting the change is -12 vector, +20 scalar and +4
stores, or about **+1.9%**. The two estimates differ by 3x and the difference is entirely WHICH
INSTRUCTIONS MOVE, not the exchange rate.

So: build it only after counting the emitted instructions for one format the way RULE 5p did, rather
than inheriting a figure from a rule that was arguing the opposite case. **The durable finding here
is the 2.47:1 rate, not the size of any one change.**

★ AND NOTE WHAT DOES NOT MOVE: llama.cpp pays `addv` + `ldrsb` + `sub` + `madd` per sub-block, which
is one VECTOR and three scalar, where jitllm pays one `MLA` by element. jitllm's inner loop is already the
better trade at this exchange rate; only the prologue extraction is worth moving. Adopting
llama.cpp's structure WHOLE would be -6 vector +16 scalar = +0.27% time, i.e. still a small loss --
RULE 5t's verdict survives the re-pricing, with a different margin and for a different reason.

**THE GENERAL RULE, WHICH MATTERS MORE THAN THE ONE CHANGE: on arm64, price ideas in VECTOR
instructions, and treat moving work to the scalar pipes as worth it up to about 2.5:1.** Every
arm64 verdict in this file that rests on a flat instruction count -- 5j, 5n, 5q, 5t -- was decided
with the wrong exchange rate and should be re-derived before being trusted. RULE 5q's LD1x4 result
(13 instructions removed for 0.9996) is consistent with this: those were LOADS, on the load/store
units, and they were never on the critical path.

**★ 5o. THE SCALES NEVER LEAVE REGISTERS NOW, WHICH IS THE FIRST arm64 CHANGE RULE 5n's FRAMEWORK
PREDICTED AND THE FIRST ONE THAT WORKED.** RULE 5n says the M4 kernel is at 80% of an 8-wide core's
issue width, so only REMOVING instructions helps. `emitA64KScales` stored four vectors of int32
scales to scratch and the block loop loaded them back four times -- eight instructions per
super-block spent moving values that were already in registers when they were stored.

The sixteen 6-bit scales now stay in four spare vectors for the 16-sub-block formats. Only those get
it, and the register budget is why: `sub` is 16 for Q3_K and Q6_K so `wregs` is 1, freeing `wreg0+1`
and `areg0+1`, and neither format has a min term, freeing the two registers it would need. Exactly
four, which is exactly sixteen int32 scales. Q4_K and Q5_K keep the scratch path.

    Q3_K   12.7 -> 12.9 GB/s        Q6_K   25.4 -> 25.7        Q4_K  22.5 -> 22.4 (control)
    tinyllama Mac CPU vs llama.cpp  0.7934 -> 0.8253,  IQR/median 0.008

**Q4_K not moving is the control that makes the rest believable** -- it takes the same code path in
every respect except this one, and it did not budge. Token ids are byte-identical to the previous
build and `TestGreedyMatchesLlamaCpp` passes on the M4 for both models.

Three changes have now taken that row 0.763 -> 0.793 -> 0.825, all of them instruction removals
(RULE 5h's `BIC`, this, and RULE 5d's integer tail before them), and every idea that moved work
around instead of deleting it measured zero.

**★ 5p. THE Q3_K KERNEL'S INSTRUCTION MIX, DISASSEMBLED RATHER THAN ESTIMATED.** Dump
`cpu.EmitA64(Spec{W: Q3_K, ...})` and run it through `llvm-mc --disassemble --triple=aarch64`, one
instruction per line. 216 instructions per super-block:

    ldr 29 + ldur 9   38   16 activation loads, 6 hoisted weight loads, scales, misc
    ushr              24   qs shifts (12) and hmask shifts
    movi              24   SIXTEEN are the per-sub-block accumulator zeroing
    sub               19   16 zero-point, 3 pointer decrements
    and               19   16 qs masks
    mla               16   one per sub-block
    bic               16   one per sub-block
    sdot              16   one per sub-block
    shl                8   hmask shifts the other way
    sshll/sshll2       6   the scale widening -- was 10 before RULE 5o
    tbl 2, str 2, fcvt 2, faddp 2, cbnz 2, add 6, mov 9

**~140 in the sixteen sub-blocks and ~76 in the prologue.** I had been estimating 144/80 for several
rounds; the estimate was close, and it should have been measured on the first round rather than the
fifth. Dumping the kernel costs one command.

**No large removable category remains.** The biggest single candidate is the sixteen `MOVI`s, and
they are structural: `SDOT` accumulates into its destination, so the accumulator must be cleared, and
`MOV`, `EOR` and `SUB`-from-itself all cost the same one instruction. Widening a sub-block to 32
elements would halve them and cannot be done -- Q3_K's scale granularity IS 16 elements, and two
sub-blocks summed into one 4-lane accumulator would lose the distinction between their scales.

So the sub-block sequence is at nine instructions and every one is doing work. Anything further has
to come from the prologue, where RULE 5o already took eight and the next four were worth 0.4%.

**★ 5q. A FOUR-REGISTER ACTIVATION LOAD REMOVES 13 INSTRUCTIONS AND GAINS NOTHING -- AND IT SHIPPED A
WRONG-ANSWER BUG THAT EVERY TEST MISSED.** The disassembly (RULE 5p) showed 38 loads per super-block
against llama.cpp's 24, and the difference was that jitllm issued one `LDR Q` per sub-block for
activations that are consecutive 16-byte chunks where `vld1q_s8_x4` takes four at a time. `LD1x4Post`
does the same on four adjacent registers and post-increments the cursor by 64, so four of them cover
a super-block and replace the explicit stride bump too: 16 loads and one add become 4 instructions.

    LD1x4 vs sixteen separate loads:  0.9996,  IQR/median 0.004,  n=12

**Thirteen instructions out of 216 and no change at all.** The activation vector is 256 bytes that
every row of the matvec re-reads, so it is hot in L1 and the loads were never on the critical path.
Instruction COUNT is the lever on this core (RULE 5n), but only for instructions that are actually
waiting on something.

★ THE BUG IS THE PART WORTH KEEPING. Freeing four adjacent registers for the quad meant moving the
scale destinations onto `tmp[3..5]`, and `emitA64KScales` names those internally -- `tmp[4]` is `res`,
which holds the assembled scale bytes and is `widen16`'s own SOURCE. The model answered fluent
nonsense: token 9613, sixteen times. **`go test ./jit/cpu` passed on the M4, the NMSE checks passed,
and the throughput measurement came back a clean 0.9996 with IQR 0.004** -- a broken kernel runs at
exactly the same speed. Only the end-to-end token-id check saw it, which is the third time in this
file that a green suite has covered a wrong answer (RULE 7a's rotary layout, RULE 12i's softmax).
**Diff the ids, not just the NMSE, whenever a register allocation moves.**

The destinations now come from outside the tmp range entirely -- `wreg0+1`, `areg0+1`, `eight` and
`f0`, all free exactly when this path applies -- and `LD1x4Post` stays in the emitter with its
encoding verified against clang, unused, for whoever needs a multi-register load next.

**★ 5r. THE PACKED V CACHE BUYS NOTHING ON METAL, WHICH IS THE POINT OF HAVING MEASURED IT THERE.**
`JITLLM_KV_F16` halves the V cache's bytes and its footprint. On the M4:

    depth 0     f32V 95.99 / 95.99      f16V 95.35 / 95.46
    depth 1024  f32V 83.76 / 84.12      f16V 83.04 / 84.28
    teacher-forced against the CPU tier at depth 0: 0/32 positions disagree

Nothing, at either depth. Apple's memory is unified and the KV cache is small
next to the weights -- tinyllama's is 45 KB per position against 521 MB of
weights per token -- so halving it moves a rounding error. Its value is
CAPACITY, and capacity only binds where VRAM is separate and scarce: the 4 GB
CUDA card, where llama-2-7b's 1 MB per position is what decides whether the model
fits at all. Keep it for that; do not expect it to show up in a Metal rate.

**★ 5s. THERE IS NO IDLE TO RECOVER ON THE M4: THE POOL IS BUSY 100% OF THE DECODE.** `xctrace` ships
with Xcode and profiles over ssh, which makes this a ten-minute measurement rather than the separate
session I had been deferring it to:

    xctrace record --template "Time Profiler" --output jitllm.trace --launch -- ./jitllm run -devices cpu ...
    xctrace export --input jitllm.trace --xpath '/trace-toc/run[@number=1]/data/table[@schema="time-sample"]'

200 tokens, 10,038 samples at 1 ms:

    P cores      9,827   98%
    E cores        211    2%
    Running     10,038
    Blocked          2

**Both numbers close a line of enquiry.** macOS QoS places the workers on P cores by itself, so
`topo_darwin.go` being unable to pin costs nothing measurable and RULE 5's E-core penalty is not
happening here. And 200 tokens at ~80 tok/s is ~2.5 s of decode which, over four workers, is ~10,000
thread-milliseconds -- almost exactly the sample count. The workers are busy essentially all of it.

So the "~13% jitllm loses between its own kernels" (RULE 5k) is **not idle time and not dispatch**. It is
work: the model quantizes activations, runs attention and the norms, and issues 154 small matvecs
where `fmtbench` times one large one with better locality. The remaining Mac gap is entirely in what
the code DOES, not in waiting, and the kernel is already accounted for instruction by instruction
(RULE 5p).

**★ 5t. AND THE LAST CANDIDATE -- ADOPTING llama.cpp's SCALE STRUCTURE -- IS NOW 10 INSTRUCTIONS
WORSE, BECAUSE THIS SESSION'S OWN FIXES REMOVED WHAT IT WOULD HAVE SAVED.** It was carried for
several rounds as "roughly instruction-neutral by hand count, so it must be built and measured".
That verdict assumed the scale widening still cost ten instructions. RULE 5o and 5q took it to six,
and the disassembly (RULE 5p) confirms six `sshll`/`sshll2`:

    jitllm today, MLA by element        6 widen + 16 mla                    = 22
    llama.cpp structure              0 widen + 16 addv + 16 scalar madd  = 32

**RULE 5d's trade is not merely defensible now, it is 10 instructions ahead**, and it got further
ahead as the widening shrank. Keeping all four lanes with `MLA` by element avoids sixteen horizontal
reduces and sixteen scalar multiplies; the only thing it costs is a widening that is now six
instructions.

That closes the last arm64 candidate analytically rather than by measurement, which is legitimate
here precisely because the quantity is an instruction count on a kernel proven to be issue-bound
(RULE 5n) and the count is disassembled rather than estimated (RULE 5p). **The lesson is about the
ORDER of the work**: an idea priced against an unoptimised baseline can flip sign once the baseline
improves, so a shelved "neutral, build it and see" has to be re-priced before it is picked up, not
inherited.

**★ 5j. TWO arm64 IDEAS, TRIED AND MEASURED.**

1. **Rotating the integer accumulator across two registers: +1%, i.e. nothing.** The sub-block
   sequence is `MOVI` zero into iacc, `SDOT` into iacc, `MLA` reading iacc -- a read-after-write
   chain repeated sixteen times on ONE register, which is exactly the shape `Spec.Accs` fixed on
   amd64 for 1.30x on the kernel. Q3_K and Q6_K have a spare weight register (wregs is 1 at 16
   elements per sub-block), so the rotation costs nothing to try. Q3_K went 43.0 -> 43.3 GB/s and
   Q6_K did not move. **The M4 renames the hazard away**; an in-order or narrower core might not,
   which is the only condition under which this is worth revisiting.
2. **i8mm/`SMMLA`: not applicable to decode at all, and the arithmetic says so without writing any.**
   The M4 reports `FEAT_I8MM` and jitllm emits no `SMMLA`, which looks like an omission. `SMMLA`
   produces a 2x2 int32 tile from 2x8 by 8x2 int8 -- 32 MACs against `SDOT`'s 16 -- but decode has
   ONE activation vector, so one column of the tile is waste and the useful ratio is `SDOT`'s. It is
   a PREFILL instruction and llama.cpp uses it as one.

**And llama.cpp's own Q3_K NEON was read rather than guessed at.** Per super-block it issues about
76 unpack instructions against jitllm's ~80, the same 16 `SDOT`s, and SIXTEEN horizontal reduces
(`vaddvq_s32`) that jitllm does not pay because `MLA` by element keeps all four lanes (RULE 5d). So jitllm
is not losing on instruction count in the inner loop, and the remaining 9% is not sitting anywhere a
count will find it. It also uses `vbicq_u8` for the high-bit-and-zero-point collapse, which is the
same trick RULE 5h arrived at independently -- a useful confirmation that the shape is right.

**★ 5w. THE MODEL THAT WINS ON LINUX CPU LOSES WORST ON THE MAC, AND THAT RELOCATES THE arm64
DEFICIT.** llama-2-7b Q4_K_M is jitllm's ONLY CPU win (1.012 on Linux), and the reason looked
structural: it is 3.8 GB, so it is the most memory-bound model on the board, and amd64's Q4_K and
Q6_K already read 106-108% of the wall. Memory-bound favours the engine with the worse kernel. That
predicts it should be jitllm's best arm64 CPU case too. It was never run there, so it was copied over
and measured with the same harness:

    Mac CPU, scripts/vs-llamacpp.sh, n=320       ratio    IQR/med
      llama-2-7b Q4_K_M   (Q4_K + Q6_K)          0.7943    0.008    <- the WORST row on the Mac
      tinyllama q3_K_M    (53% Q3_K)             0.8276    0.031
      gemma-2b            (Q4_0 + Q8_0)          0.853

**The prediction is exactly backwards: the biggest model is the worst row.** Same model and same
formats, 1.012 on Linux and 0.794 on the Mac -- a 27-point swing that no property of the model can
explain, so it is the kernel.

The mechanism is visible in the rate. jitllm reads 3.8 GB per token at 13.1 tok/s = 50 GB/s;
llama.cpp reads the same bytes at 16.5 tok/s = 63 GB/s, against an achievable wall of ~90. **Neither
is at the wall, so a 7B model is NOT memory-bound on this chip** -- which is what breaks the Linux
intuition. On amd64 the same model IS memory-bound, both engines queue on DRAM, and jitllm's extra
instructions are free.

**So RULE 1c's "the arm64 CPU gap is K-QUANTS ONLY" is too narrow.** It was drawn from gemma (0.91)
against tinyllama (0.55) when Q3_K was the weak kernel. Q4_K and Q6_K are k-quants too, and a model
made only of them is now the worst row -- the deficit is that jitllm's arm64 k-quant kernels are
issue-bound where llama.cpp's are closer to memory-bound, and it gets WIDER as the model grows,
because a larger model spends a larger share of its token in the matvec.

That also says what a fix has to be worth: the Mac CPU rows do not need a Q3_K trick, they need the
arm64 k-quant kernel to stop being issue-bound in general.

**★ 5ab. AT FOUR THREADS llama.cpp's Q3_K KERNEL AND jitllm's ARE DEAD EVEN, SO THE MAC CPU GAP IS NOT
THE KERNEL AT ALL.** `scripts/ref/q3kbench.c` is now threaded (spin barrier, caller as worker 0,
contiguous row slices -- the same shape jitllm's pool uses), which is the measurement RULE 5aa said did
not exist. Same 280 MB set, same soak, same median of 15:

                     1 thread   4 threads
      jitllm Q3_K         12.6       45.1
      llama.cpp Q3_K   14.0       45.2      <- 1-thread reproduces RULE 5k's 14.3

**The 11% one-core deficit is GONE at the thread count that ships.** And every arm64 verdict in this
file -- 5k's "13.5% faster", 5l's "a uniform ~1 ns per super-block", 5n's IPC argument, 5v, 5y's
"45 vector instructions" -- rests on the ONE-CORE ratio. At one core the kernel is issue-bound and
the instruction mix decides; at four it is not, and the mix stops mattering.

**So the row splits the other way round:**

      jitllm kernels @4 (tinyllama mix)   51.9 GB/s     jitllm model    38.7   = 75% of its kernels
      llama.cpp kernels @4  ~= same                  lcpp model   46.3   = 89% of its kernels

**jitllm loses 25% between its own kernels and its model where llama.cpp loses 11%, and that 14-point
difference IS the 21-point row.** The Mac CPU work is ALL harness -- dispatch, the 154 small matvecs
against fmtbench's one large one, and the 9.2% of non-matvec ops -- and none of it is the kernel.

That promotes whole-layer fusion from "one of four levers" to THE lever: RULE 6's M4 column measured
6.15% end to end, it is the only one of the four measured that way, and it attacks exactly this.

**AND IT RETIRES THE INSTRUCTION-LEVEL SEARCH THIS SESSION SPENT ITS LAST HOURS ON.** The vector and
scalar probe coefficients (RULE 5x) are real, and they are measured at ONE core on a kernel that is
issue-bound there. They do not describe the shipping configuration, which is why the LDP change
predicted 4.1% and delivered 1.0% (RULE 5z). **Measure the kernel at the thread count that ships,
before pricing anything about it.**

**★ 5ac. THE arm64 PREFILL GEMM's FOLD IS INTEGER NOW. ITS SIZE ON THE M4 IS UNKNOWN, AND I
PUBLISHED TWO CONTRADICTORY NUMBERS FOR IT BEFORE RUNNING THE CONTROL THAT INVALIDATES BOTH.**

The change itself. `MLA vd.4s, vn.4s, vm.s[i]` is a 32-BIT multiply, so unlike amd64 -- where
`VPMADDWD`'s int16 operands confine the trick to Q3_K (RULE 6b) -- there is no width bound. And
arm64 already subtracts the zero point from the WEIGHT BYTES (`SUB16b` in emitA64KUnpack, so SDOT
can take them signed), so `dot_s` IS the bracket and there is no bias correction to undo first. The
fold goes from `SCVTF4s` + `FMUL4s` + `FMLAelem` to ONE `MLAelem` -- three instructions per
(sub-block, row, vector) down to one -- with the float accumulator moving to Scratch and its
registers becoming the int32 total. Emitted bytes, Q3_K and Q6_K: 7% smaller under the old float
hoist, **16-18% smaller** now. Q4_K/Q5_K keep the float hoist, because their min term needs the min
scales as int32 too and `emitA64KScales`' `intSc` converts only the MAIN scales -- the DECODE kernel
relies on the min ones staying float with dmin folded in.

**★ WHAT IT IS WORTH IS NOT ESTABLISHED. THREE ANSWERS, ALL FROM THE SAME AD-HOC SCRIPT:**

    first pass     1.0004  IQR 0.001   at 73 tok/s      "flat, RULE 5ab extends to prefill"
    second pass    1.0365 / 1.0357     at 153 tok/s     "no, +3.6%, the first was contended"
    SELF CONTROL   1.0351 / 1.0011 / 1.0547             ...running the SAME BINARY in both arms

**The control is the whole story: slot A reads ~3.5% faster than slot B for one identical binary.**
That is the size of every effect claimed above it. The M4 also swung 70 -> 104 tok/s inside one
run, and a depth sweep in the same session collapsed 71 -> 57 -> 56 mid-measurement. **Nothing about
the SIZE of this change on the M4 survives; only its correctness does.**

**★ THE CAUSE IS THAT I WROTE ANOTHER AD-HOC MAC SCRIPT, WHICH IS THE ONE THING RULE 12q SAYS NOT
TO DO -- AND I QUOTED RULE 12q EARLIER THE SAME DAY WHILE DOING IT.** `scripts/vs-llamacpp.sh` has
a Darwin branch, soaks to thermal steady state, interleaves ABBA, gates on PSI and IQR, and prints
every raw sample. The scratchpad script had a three-iteration warm-up and none of the rest. RULE
12q's words are "a scoreboard whose rows come from two different tools is not a scoreboard"; this
is the second tool, again, on the same host, for the same reason -- it was quicker to write.

**★ SO EVERY jitllm-AGAINST-jitllm A/B ON THE MAC NEEDS A SELF-CONTROL ARM, AND NONE OF THEM HAS EVER HAD
ONE.** RULE 5d already established the principle -- gemma is included in the k-quant A/B precisely
because "a negative control that moves is a harness that is drifting" -- but every Mac A/B in this
file compares two DIFFERENT binaries and never asks what ONE binary reads against itself. On a host
where RULE 2d's clamp gate is a no-op (macOS has no /proc/cpuinfo, RULE 12q), that control is the
only instrument left. **Run it first, not last.**

**The kernels are kept, on correctness, which IS established independently of any rate.** Full
`jit/cpu` and `model` suites PASS on the M4; `TestGreedyMatchesLlamaCpp` PASS against llama.cpp's
own ids for tinyllama, gemma and the MoE. Run against violations per RULE 12m: emitting the integer
fold at window 32 fails at NMSE 4.7e+28, and the must-pass control -- reading d_a from block 7
instead of block 0, legitimate at window 256 -- passes.

★ AND DO NOT TRUST A `JITLLM_CORES` SWEEP HERE EITHER: one clean run read 45.6 tok/s at one core and
46.0 at two, no scaling at all, then 151 at four. RULE 1d-nonies records the open finding that
`engine/sched/pool.go`'s `active` counts SPAWNED workers rather than PARTICIPANTS, so `JITLLM_CORES=N` is not
an N-core pool.

★ AND THE FIRST SWEEP WAS CONTAMINATED, BY ME, ON THE MACHINE THE MUTEX DOES NOT REACH. RULE 3d's
lock lives in `scripts/cap` and the Mac has no `scripts/cap`, so two sweeps ran at once with three
jitllm processes live together. **A mutex that protects one host protects one host.**

**★ 5aa. THE M4 KERNEL TABLE AT THE SHIPPING THREAD COUNT, WHICH DID NOT EXIST.** Every per-format
number in this file for the Mac was taken at ONE core. At four (`JITLLM_CORES=4`, wall 74.0 GB/s):

    Q8_0 72.7 (98%)   Q6_K 72.5 (98%)   Q4_0 69.9 (95%)
    Q4_K 60.7 (82%)   Q5_K 55.8 (75%)   Q3_K 45.1 (61%)

Q3_K's 61% is NOT a defect and RULE 5i is why: per ELEMENT it runs 105 Gelem/s against Q4_K's 108,
Q6_K's 88 and Q5_K's 81. It carries 110 bytes per 256 elements where Q6_K carries 210, so it reads
fewer GB/s while doing the same work per element. Density sets GB/s; it does not set efficiency.

**★ AND THE COMPARISON THIS TABLE INVITES IS A TRAP I NEARLY PUBLISHED.** Weighted by tinyllama's
format mix the kernels reach **51.9 GB/s**, while jitllm's whole model is 38.7 and llama.cpp's whole
model is 46.3 (both at n=320). It is tempting to conclude "jitllm's KERNELS already beat llama.cpp's
MODEL, so the entire gap is harness" -- and that is exactly the apples-to-oranges RULE 5i made when
it wrote "llama.cpp whole model 53.1 = 109% of jitllm's kernels". **llama.cpp loses ground between its
kernels and its model too**, and nothing here measures how much: `scripts/ref` is single-threaded.

Doing it properly with what IS known -- llama.cpp's Q3_K kernel is 13.5% faster at one core
(RULE 5k) -- puts its four-core kernel mix near 58.6, so it loses ~21% to its model where jitllm loses
25%. That splits the 21-point row as roughly **two thirds kernel, one third harness**, and leaves the
session's conclusion standing. **To split it properly, `scripts/ref` needs a threaded harness.** That
is the one measurement that would settle where the Mac CPU work should go, and it does not exist.

**★ 5u. FOUR CHEAP MAC-CPU HYPOTHESES, ALL TRIED AND ALL MEASURED NEGATIVE.** The row
is 0.8276 (IQR 0.031, n=8, `scripts/vs-llamacpp.sh` at n=320), and llama.cpp reads 87.8-89.5 tok/s
there -- NOT the 99.34 in RULE 1c, which was a shorter sample.

1. **Does the amd64 Q3_K high-bit fold transfer?** No -- arm64 already shifts the bit straight to
   position 2 and collapses it with the zero point (RULE 5h's `BIC`). The amd64 kernel was the one
   behind, and it has now caught up.
2. **Do E-cores help now that the kernel is issue-bound rather than memory-bound?** No. The sweep in
   RULE 1c was taken when this model ran at 54 tok/s; re-run on the current build it is unchanged in
   shape -- 48.6 / 78.3 / 77.5 / 77.0 / 76.6 / 76.1 at 2/4/5/6/8/10. It saturates at the four P-cores
   and declines past them. The reason is RULE 5s: macOS QoS places the workers on P cores by itself
   and `pin()` is a no-op on Darwin, so asking for more workers oversubscribes the same four rather
   than reaching the E-cores. Using them would need a QoS class per worker AND a proportional split,
   since a barrier costs the slowest worker -- and RULE 5's Linux `pe` row measured 0.864x.
3. **Is the spin budget wrong for this host?** No. RULE 11a swept it on Linux only, but the M4 agrees:
   65.2 / 80.8 / 80.7 / 80.3 / 79.8 / 79.8 tok/s at 0 / 50 / 250 / 1000 / 5000 / 20000 us. The 250 us
   default is already at the plateau. Spinning matters (never-spin costs 19%); tuning it does not.
4. **Is the pool itself Linux-only?** No -- `engine/sched/pool.go` is portable and only topology and pinning
   are tagged. But `engine/sched/latency_test.go` IS `//go:build linux`, so the dispatch round trip had
   never been measured on the Mac; `model.TestFusionCeiling` puts it at 3.008 us against Linux's
   1.6-2.1 us for four to six workers, which is most of why the fusion ceiling clears the bar there.

**What is left is the ~13% kernel gap, and it has no known instruction to remove** -- RULES 5k, 5l,
5n, 5p and 5t are five separate attempts at it. RULE 5l's arithmetic still bounds the row: the Mac
CPU needs BOTH that and the ~13% harness, and the harness half is now known to be worth 6.15%
(RULE 6, Mac column). Neither alone gets to 1.0.

**★ 5c. THE WIDTH IS NOW TUNED ON REAL TOKENS, NOT TABULATED.**

`nn/tune_amd64.go` times consecutive decode tokens, alternating incumbent
against challenger (`best` vs `best+1`, five samples each), and climbs while the
challenger wins by more than 2%. It normally costs ten tokens and stops at 2.
The result is cached in `~/.cache/jitllm/packwidth` keyed on **(CPU, cores, shape
set)** -- the shape set standing in for the model, so two files with the same
(type, blocks-per-row) inventory share an answer and a finetune does not force a
re-tune.

Three things about the design are load-bearing:

1. **It measures tokens, not a slab.** A synthetic load-time sweep measures the
   wrong thing: on one core, or on any working set that fits in L3, it picks
   pack8 -- which is exactly how pack8 shipped as a regression. RULE 5b is the
   reason this is online.
2. **One width serves every quant type.** The one-core sweep says Q8_0 peaks at
   7 and Q4_0 at 2; shipping those per-format optima measured 20.63 tok/s
   against 24.24 for uniform pack2. Stream count is a property of the pool, not
   of the format, so the formats do not get to choose independently.
3. **A dispersed measurement is refused, not rounded.** Above IQR/median 0.10
   the tuner keeps `cpu.BestPack`'s answer and writes nothing to the cache. A
   tuner that cannot measure must degrade to the table, never to a guess.

`JITLLM_PACK=N` forces a width and disables tuning; `JITLLM_TUNE=0` disables tuning
and uses the table; `JITLLM_TUNE=1` forces a re-tune ignoring the cache.

**★ 5e. THE TUNER WAS NOT OBEYING RULE 2, AND IT IS THE THING THAT DECIDES EVERYTHING.** Every tuned
answer in the engine -- interleave width, GEMM tile, participant count, prefill chunk -- came out of
two duels that broke the rule the rest of the project is measured by. Found by a review of
the tuner, which read the code rather than the comment above it.

1. **Strict ABAB, while the context grows every token.** `turn%2` put the incumbent on the EARLIER
   token of every pair, and a decode token gets monotonically more expensive as the KV cache fills, so
   the challenger was charged for that growth on every duel it ever ran. The bias is small per token
   and it is ONE-DIRECTIONAL, which is the dangerous kind: it does not average out with more rounds,
   it just gets measured more precisely. It is now ABBA -- incumbent on turns 0 and 3 -- so drift that
   is linear across four tokens cancels.
2. **A ratio of two medians, not a median of per-round ratios.** RULE 2 names the second one and the
   tuner did the first. They are different statistics: a ratio of medians compares the middle of one
   drifting series against the middle of another and the drift survives the division. Pairing inside
   the quad removes it before the division happens.
3. **The fix must not double what an answer costs.** ABBA with one ratio per quad needs 20 tokens
   where ABAB needed 10, and a tuner that cannot settle inside a short generation does not settle at
   all -- gemma's own reply to a six-token prompt is 17 tokens. So a quad yields TWO ratios, (0,1)
   with the incumbent first and (3,2) with it second: same six-token duel, opposite orderings, and
   whatever the growing context does appears in both directions.

`duelTuner`, which tunes the prefill chunk width and the participant count, had defect 1 as well and
already had the paired ratios; it is ABBA now too.

Measured after: gemma re-tunes to pack2 in twelve tokens, printing `pack2 vs pack3: 0.9644x` then
`0.9951x` and settling -- which is the answer RULE 5b arrived at by hand, so the corrected apparatus
reproduces the known-good result rather than moving it.

**★ 5f. AND THE CACHE KEY SAID "unknown" ON EVERY MAC.** `cpuName()` read `/proc/cpuinfo`
unconditionally. On darwin that file does not exist, so the CPU field of the tuner's key -- the field
whose entire job is to stop one machine's tuned answer being served to a different chip -- was a
constant across the whole platform. It reads `machdep.cpu.brand_string` through `syscall.Sysctl` now,
in `nn/cpuname_darwin.go`.

This is OS-tagged and nothing else in the tuner is, deliberately: reading a chip's name is an
operating system API, not a hardware capability, and no runtime probe turns `/proc/cpuinfo` into a
Mac's brand string. **Tag the syscall, never the decision** -- the standing rule that the JIT decides
at runtime is about what a kernel should DO, which is a property of the machine, not about which
syscall exists, which is a property of the OS.

Still open from the same review, recorded rather than done: the shape fingerprint hashes `(type,
blocks-per-row)` and drops row counts, multiplicity and order, so two models with the same format
inventory and very different shapes share an answer. RULE 5c wanted that -- a finetune must not force
a re-tune -- but a finetune does not change row counts either, so the row count could be in the key
without costing anything. Unmeasured.

**★ 5g. "100% OF THE MEMORY WALL" IS THE WRONG OBJECTIVE, AND THIS FILE'S OWN TABLES SAY SO.** Rows in
RULE 1e read 106%, 108%, 110% and 112%, which is not a kernel exceeding physics -- it is the `Stream`
probe understating the wall (RULE 12c says treat it as a lower bound). A target the measurement can
report as already-beaten cannot direct anything. The honest objective is a per-token critical path:

    token floor = serial host and seam cost
                + SUM over ops of max(bytes / that op's wall, ops / issue ceiling, launch floor)

and the number to maximise is `floor / measured token time`. That formulation is what makes the three
live constraints visible AT ONCE rather than one at a time: k/v is at the LAUNCH FLOOR (0.3 MB is
~2 us against a 2.37 us floor, so its 48% is roughly the physical result and no kernel change moves
it), Metal's Q4_0 rows are at the ISSUE ceiling and not the memory one, and attention's share grows
with context while every weight row's does not. Dividing weight bytes by one stream probe hides all
three.

And the width genuinely is per-model, which is why it is not a constant: on
gemma pack2 beats pack8 by 15% (24.24 vs 20.98 tok/s), while on a Q4_0
tinyllama pack2, pack3 and pack4 land within 1.2% of each other.

## ★ RULE 6: the Spec is capped. Adding a field requires a measurement.

**★★ 6-0. THE 5% BAR IS A PORTFOLIO BAR, NOT A SOLO BAR, AND IT HAS BEEN USED AS
A REJECTION ALL SESSION -- BY ME, TWICE, ON THE SAME DAY I QUOTED 6a.**

The user's objection, and it is arithmetically right: *"4-6% is not nothing, when
you stack up levers like that you beat llama.cpp."* Levers that each remove a
fraction OF THE TOKEN are ADDITIVE IN TIME SAVED (RULE 5z's composition note), so
three 4% levers are 12% of the token, not 4%:

    row                                       now    fusion saves    after
      tinyllama Linux CPU (rested)          0.9790      2.84%       1.0076  ★ CROSSES
      OLMoE Linux CPU                       0.8975      8.39%       0.9797
      OLMoE M4 CPU                          0.7280      8.77%       0.7980

**ONE LEVER TAKES tinyllama ON LINUX PAST llama.cpp.** RULE 1d-quinquies computed
that same 1.0076 and discarded it; see below for why that dismissal was wrong.

**★★ SO THE BAR REJECTS NOTHING. A MEASUREMENT REFUSES A CHANGE ONLY WHEN IT IS A
CLEAR REGRESSION** -- the user's instruction, in their words: *"never reject
unless it's a clear regression."* There is no size below which a correct,
contained change is turned away. RULE 8b went 16.4 -> 27.4 tok/s in six steps,
several of them in the 3-5% band, and a strict solo bar would have left the
engine at 16.4.

What the number is still FOR, because measuring is not optional:

  - **A clear regression is a refusal.** That is the only one. RULE 12ag-bis's
    centering at Tok=1 measured 0.981 over four rounds all agreeing -- refused,
    correctly, and the refusal is the measurement's job.
  - **A number decides ORDER, never admission.** 8% goes before 0.4%. "Not now"
    is a schedule; "no" needs a regression behind it.
  - **Do not quote a sub-noise change as a win.** The sum-free quantizer prices
    at 0.4% of a prefill token against a harness whose spread is 5%. Build it if
    it is cheap and correct -- it is NOT rejected -- but it cannot carry a perf
    claim, because nothing here can show it happened.

★ AND THE ONE THING THAT GETS STRICTER RATHER THAN LOOSER: a change too small to
measure is also too small to be shown NOT to regress. So the correctness gate
does the work the perf gate no longer does -- bit equality where the arithmetic
is unchanged, NMSE against the float64 reference where it is not, and RULE 12m's
run-against-a-violation before either is believed. **The bar came down; the gate
did not.**

**★★ 6-0-ter. "NEUTRAL" IS A CLAIM ABOUT EVERY AXIS, AND THIS FILE HAS A
REGRESSION ON NEARLY ALL OF THEM.** The user's point, and it reframes 6-0: *"if
something is neutral in compute it might not be in mem or allocations -- there's
always a regression somewhere."* A change measured flat on tok/s has been
measured on ONE axis. Every entry below is an actual result from this file where
the cost landed somewhere the headline number could not see:

    axis                what it cost, in this file
    ------------------  ----------------------------------------------------------
    REGISTERS           RULE 12z: the integer fold issued ~25% FEWER instructions
                        and lost 25%, because 127 -> 173 registers took an SM from
                        16 warps to 10. Price a device change in REGISTERS first.
    L1i / code size     RULE 6: full unroll is 31.7 KB against a 32 KB L1i, and
                        kernels COMPETE -- gemma's three shapes are 96 KB unrolled
                        against 2.3 KB packed. 2% faster, and it evicts its peers.
    BYTES RESIDENT      RULE 12o: device packing normalises k-quants to a nibble
                        or a byte, +75% on Q3_K, on a tier whose binding
                        constraint is bandwidth. Correct, and not free.
    ALLOCATIONS         RULE 14r: nn.MatVec's `make([]float64, k)` PER CALL -- 48
                        allocations of 16 KB per token, and the standing (unproven)
                        explanation for 28.9 ms the profiler could not attribute.
    DISPATCH REGIONS    this session: OLMoE runs 906 pool round trips a token
                        against tinyllama's 310, which no per-kernel number shows.
    MEMORY STREAMS      RULE 5b: pack8 issues 17% fewer instructions and loses,
                        because 8 interleaved rows are 48 streams across the pool
                        and the prefetcher stops covering them.
    VRAM / PLACEMENT    RULE 15d: a cached DECLINE is a tombstone that says "no"
                        after the room it wanted was freed.
    PIPE BALANCE        RULE 5y: jitllm issues FEWER arm64 instructions than
                        llama.cpp and is 13.5% slower, because 154 of them sit on
                        the pipe that costs 2.47x.

**So before writing "neutral": neutral in WHAT.** The operational form, which is
a reviewer's wording and better than the table above at being followed:

> Before calling a change neutral, name the workload and check:
>
>   TIME         whole-token latency/throughput, prefill, the relevant depth
>                range, the SHIPPING thread count, and cold preparation.
>   MEMORY       resident/peak bytes, allocation count and bytes, scratch,
>                packing expansion, KV capacity, device placement.
>   TRAFFIC      bytes actually MOVED, re-reads, cache-line ownership, prefetch
>                streams, coalescing -- not only unique input bytes.
>   EXECUTION    registers/spills/occupancy, code size and competing L1i
>                footprint, dependency chains, execution-pipe pressure.
>   COORDINATION pool regions, barriers, launches, synchronisation, transfers,
>                graph captures.
>   CORRECTNESS  the changed path actually EXECUTED; equality/NMSE/margin as
>                appropriate; deliberate violations fail.
>
> Report each relevant axis as IMPROVED, REGRESSED, UNCHANGED BY CONSTRUCTION, or
> UNRESOLVED. **An unresolved timing result is not evidence of neutrality.**

Not every change needs a benchmark on every axis -- source inspection can
establish an unchanged allocation path or identical generated code, and
"unchanged by construction" is a real answer.

**★ AND "THERE IS ALWAYS A REGRESSION SOMEWHERE" IS NOT A LICENCE TO REFUSE EVERY
RESOURCE INCREASE**, which a review corrected in my first draft of this and it
is right. Packing spends BYTES to save OPERATIONS (RULE 12o) and that is a trade,
not a defect. State the cost and the accepted trade; reserve the word NEUTRAL for
the axes actually established.

★ AND THE COROLLARY THAT MAKES 6-0 SAFE: since a clear regression is now the ONLY
refusal, the axes are what that refusal is evaluated over. Otherwise "no
regression on tok/s" admits everything, and the file's own eight entries above
are what that costs.

**★★ 6-0-bis. AND "ALL-BUT-REDUCTIONS IS THE UNSOUND COLUMN" CONFLATES A BARRIER
WITH A DISPATCH.** RULE 1d-quinquies refuses that column because "a matvec reads
the whole input vector, so it needs every worker's output from the op before it".
True, and it is a statement about BARRIERS. The column's cost is measured in
`pool.Do` ROUND TRIPS, and those are a different quantity:

    one pool.Do round trip, measured       2.858 us   (both hosts, TestFusionCeiling)
    an in-kernel spin barrier              ~620 ns    (RULE 5's own figure)

A fused routine still barriers between matvecs; what it stops doing is returning
to Go 906 times a token to re-split the work and re-publish a sequence number.
**So most of the all-but-reductions column is reachable while keeping every
barrier the data flow requires**, and 1.0076 is back on the table. This is the
user's whole-decode-lowering idea, and the buffer half of it is the enabler:
generated code cannot allocate, so every intermediate has to be pre-sized at
NewState with bounds checks in the emitted prologue.

**★ 6a. A BELOW-BAR MEASUREMENT REJECTS THE CHANGE AS MEASURED, NOT THE IDEA.** The 5% bar is an
anti-scope-creep device and it is a bad stopping rule, because it silently converts "this lever alone
is small" into "this idea is dead". RULE 8b's own history is the counter-example: 16.4 -> 27.4 tok/s
in six steps, several individually in the 3-5% band, none of which would have survived a strict solo
bar. Applied naively this rule would have left the engine at 16.4.

So when a measurement lands below the bar, record what would change the answer before shelving it:
which other lever it combines with, what it unlocks that the number does not contain, and **whether a
LAYOUT change is the missing enabler**. That last one is the hardest to see, because a microbenchmark
inherits the current layout and so cannot show what a different one would buy -- this is from
warpjs-itr2, where a technique that measured as a regression became a >10x win once the layout
changed and a further lever was added.

Also audit the ceiling model for levers it assumed away. The whole-layer-fusion estimate counted
dispatch cost and forgot that every region is also a BARRIER costing the SLOWEST worker rather than
the mean -- a second lever hidden inside the first.

`cpu.Spec` has five fields and `Spec` is exactly where a tensor compiler would start
growing, so adding a sixth requires a paired-interleaved MEASUREMENT -- and per
6-0 that measurement admits the field unless it shows a clear regression. The
anti-scope-creep device is the requirement to measure and to name what the field
buys, not a threshold it has to clear.

**Interleave width and unrolling, all measured paired on one core, 127 MB
DRAM-resident (`jit` tests `TestABUnrolled`, `TestABPack8`, `TestABPack8Unrolled`):**

    full unroll, 1 row  vs looped   1.16x    4.8 KB
    pack8 (8 rows)      vs looped   1.37x    0.77 KB   <- shipped
    pack8 + full unroll vs pack8    1.02x    31.7 KB   <- not worth it

pack8 bakes only the ROW STRIDE, so eight rows come off ONE base pointer and the
code stays at a single block body. Baking the whole loop as well removes the last
branch (6906 instructions, 1 branch, no bounds checks anywhere) but costs 31.7 KB
against a 32 KB L1i, and buys 2%. Worse, kernels COMPETE for that cache: gemma
has three shapes, so unrolled is 96 KB of hot code against 2.3 KB for pack8.

**Baking K/N shapes was RE-MEASURED and the old rejection was wrong.** This rule used to cite
"+2.2% / +2.1% / -1.8%, i.e. noise" -- an experiment that baked the loop BOUND. Fully unrolling the
block loop with constant displacements, which is what per-shape specialization actually means,
measures **1.16x at 2.1% IQR** on one core against a DRAM-resident set. That clears the 5% bar, so
`cpu.EmitUnrolled` exists and `nn.JIT.AddShape` generates one kernel per (type, K) a model really
contains. A transformer reuses a handful of shapes across all its layers, so it is a few kernels,
not one per layer.

Bounded by L1i, which the same measurement exposed: a block costs ~75 bytes of code, so k=2048 (64
blocks) is 4.8 KB and fits, while k=16384 (512 blocks) would be 38 KB and thrash a 32 KB L1i.
`cpu.MaxUnroll` is 128 blocks.

Baking scalar constants like `rms_eps` or rope frequencies was tried and measured at 0.04%
(AGENTS.md's reopened table lists it at lowest priority).
The same sentence used to add "any GPU code generation (noise, plus 243 ms of cold JIT)". That was a
DECODE measurement of shape-specialised PTX, and it was reversed by a later prefill tile
sweep worth 18.7x (`gpu-kernels.md`, "the GPU JIT's founding measurements"); every
device kernel is generated today.

**`Spec.Accs` is now READ by an emitter, which it never was before.** The Q4_K/Q5_K kernel issued
two `VFMADD231PS` into `Y0` per group x 8 groups, so a 144-byte super-block carried **16 serially
dependent FMAs** -- at 4-cycle latency, 64 cycles of pure dependency against a ~40-45 cycle issue
floor. The registers to split it came from widening `KernelConst(Q4_K)` to 32-byte constants, so the
two prologue-only masks (0x3F, 0x30) became memory operands and freed `Y12`/`Y14`. No new `Spec`
field, so this rule is untouched.

    ONE CORE, 128 MB DRAM-resident, paired    SIX CORES, tinyllama q3_K_M, end to end, IN PROCESS
    Q4_K  accs2  1.297x  IQR 2.2%             accs2 vs accs1  1.0333x  IQR 7.0%   39.01 -> 40.54 tok/s
    Q4_K  accs4  1.344x  IQR 1.6%             accs4 vs accs2  1.0135x  IQR 8.7%   noise, not taken
    Q5_K  accs2  1.265x  IQR 1.4%
    Q5_K  accs4  1.274x  IQR 3.3%   (clamps to 3 chains; Q5_K spends Y15 on qh)

**1.30x on the kernel is 3.3% on the model.** That is the fourth time a win measured in isolation has
shrunk or vanished at the real thread count, after 4-row interleaving, pack8-plus-unroll and pack
width. It is still worth taking -- gated, ten bytes of code, an existing field -- but the ratio
between the two columns is the thing to remember, not the 1.30x.

Note also that the design recon predicted **2.0-2.7x** here from a cycle model. The kernel was
latency-bound, so the direction was right, but the magnitude was over by ~2x: issue and memory carry
more of the constraint than a pure dependency-chain model allows for.

**★ THE CROSS-PROCESS A/B WAS REJECTED, AND THAT IS THE REUSABLE PART.** Run as two processes -- one
exporting `JITLLM_ACCS=1`, the other `=2`, alternating -- the same end-to-end comparison came back
**median 1.045 with IQR/median 0.123, i.e. REJECTED**, ratios spanning 0.78 to 1.21 at n=16. Model
load, page-cache state and thread placement vary more between processes than a 3% change is worth.
Two `State`s over one mmap'd model, alternated by `bench.AB` inside one process, gave IQR 7.0% at
n=21. "Within one process" in rule 2 is not a style preference.

**★ THE amd64 K-QUANT MIN TERM IS NOW HOISTED TOO, AND IT IS WORTH 1.3%.** The Q4_K/Q5_K group loop
spent SIX instructions of tail on ONE `VPDPBUSD` of work: a convert, three broadcasts and two FMAs.
Three of the six were the min correction `-dmin*m[g]*d_a[g]*sum(a)[g]`, computed one scalar at a
time -- broadcast the min, broadcast the sum, FMA into an eight-lane accumulator, where seven of
those eight lanes were the same number added seven extra times and `QuantizeQ8`'s stored `-sum(a)/8`
existed only to cancel that.

It is a dot product of two eight-element vectors that are BOTH already in registers in the prologue.
The sums are the ODD lanes of the same pair-array loads the scales are gathered from, so one extra
`VSHUFPS`/`VPERMQ` produces them; one `VMULPS` forms all eight products and one `VADDPS` puts them in
accumulator 0, where the row's existing horizontal reduce already sums all eight lanes. The `/8` is
undone by a `VPSLLD` on the 6-bit `m[]` before its int-to-float convert -- one instruction, exact, and
it leaves every other kernel's AScale layout alone. **24 instructions per super-block become 4.**

    tinyllama q3_K_M, soaked to steady state, ABBA, alternating binaries
      SIX CORES   1.0129   IQR/median 0.067   n=14
      ONE CORE    1.0335   IQR/median 0.004   n=10

**That is the sixth time a win shrinks at the real thread count** (after 4-row interleaving,
pack8-plus-unroll, pack width, k-quant accs, and the GPU split), and by now it is not a surprise but a
prediction: at six cores this kernel is DRAM-bound, so halving its tail buys a third of what it buys
on one core. Under RULE 6's 5% bar as a solo lever -- kept because it is already correct, it deletes
instructions rather than adding a field, and RULE 6a says a below-bar measurement rejects the change
as measured rather than the idea.

**What it does NOT unlock is the arm64 trick.** RULE 5d's integer accumulation needs ONE activation
scale per super-block; amd64 keeps eight, per-32, and folds them vectorized into the float scale
vector instead -- which is why `nativeHoistsActScale()` is false there and yet nothing is missing.
Widening amd64's window to 256 would enable the integer tail, but AVX2 has no multiply-by-element, so
each group would need a `VPBROADCASTD` that arm64's `MLA vd.4s, vn.4s, vm.s[i]` does not, and the
instruction count comes out level. Measure before building it.

**★ Q3_K WAS THE WORST KERNEL ON THIS BOX AND IS NOW HOISTED THE SAME WAY. +3.6% ON tinyllama.**
A per-format table, every shape sized to 280 MB so nothing is measured out of the 24 MB L3, is the
same one-line diagnostic that found Metal's problem (RULE 12j) -- two formats through one machine
with the memory system held fixed:

    format   GB/s   of wall      tail instructions per VPDPBUSD
    Q8_0     54.6     98%        (no unpack at all)
    Q4_0     46-47   83-85%
    Q4_K     43.9     79%        6, and 3 of those were removed above
    Q3_K     33.6     60%        ELEVEN

**Q3_K is 53% of a tinyllama by weight**, so one kernel at 60% of the wall was most of that model's
CPU gap. Seven of its eleven tail instructions were loop-invariant work done eight times:

1. **d_a, folded per block.** One activation scale covers 32 elements, which is Q3_K sub-blocks 2b
   and 2b+1, so the fold needs the eight d_a values duplicated PAIRWISE -- `VPERMQ` then `VSHUFPS`,
   two instructions per half, against the eight broadcasts and eight multiplies they replace.
2. **The -4 correction, which is a 16-element dot product.** Q3_K's value is `d*sc*(v + 4*hbit - 4)`
   and the -4 becomes `-4*sum(a)` over each SIXTEEN elements -- exactly what `AHalf` holds, two
   floats per activation block, already contiguous and already in sub-block order. The loop
   broadcast both halves, blended them into eight lanes and added them to the dot, four instructions
   x eight blocks, for two distinct numbers. Two `VMULPS` and an add-tree give all sixteen terms; the
   x4 for the four lanes each occupied is two doublings and needs no constant.

    tinyllama q3_K_M, soaked to steady state, ABBA, alternating binaries
      ONE CORE    1.0637   IQR/median 0.070   n=12
      SIX CORES   1.0355   IQR/median 0.066   n=14

**This one shrinks LESS at six cores than the others did** -- 1.064 to 1.036, where Q4_K went 1.034
to 1.013 -- and that is the per-format table earning its keep. A kernel at 60% of the wall has issue
headroom to reclaim; one already at 79% does not. The table says which kernels are worth attacking
before any of them is opened.

**★ ONE-CORE MEASUREMENTS NEED THE SOAK TOO, AND FOR THE OPPOSITE REASON.** The same A/B on a COOL
box read IQR/median 0.086 with samples spanning 13.2 to 20.1, at one core, with PSI at 0.05% and
nothing else running. One core does not heat the package enough to throttle, but it does get
TURBO -- and how much depends on what ran just before, including the other arm. Soaked, the same
comparison reads 0.070 and the medians separate cleanly. Cool is not the same as steady.

**gemma has NO k-quants.** It is `Q4_0 126 tensors 66.66%` + `Q8_0 1 tensor 33.33%` -- check with
`jitllm info`. An analysis that reported gemma as "66.7% Q4_K/Q5_K" had read the 66.66% off the Q4_0
row, and every prediction downstream of it was about a tensor type the model does not contain. The
k-quant model on this box is tinyllama q3_K_M (Q3_K 52.97%, Q4_K 34.16%, Q6_K 9.79%, Q5_K 3.02%).

**A dead branch was removed on the way.** `Emit` advertised `Rows 1 or Pack8` for Q4_K/Q5_K, which
was unreachable behind the earlier `Rows != 1 && Rows != 4` guard AND a lie: `emitKQuantMatVec`
never read its `rows` argument at all. Lifting the guard would have silently produced a single-row
kernel that passes every NMSE test. There is no interleaved k-quant kernel; the signature now says so.

**★ RULE 6c. THIRTEEN OF 180 GEMM SUBTESTS WERE PASSING ON A NaN ORACLE, AND ONE
OF THEM WAS THE SHIPPING TILE.**

Two defects that were each invisible alone and silent together.

`buildWeights` fills a weight block with random bytes and then normalises every
f16 scale, "because random bits land on infinities and NaNs, which is not what a
kernel test is about". Its switch covers Q4_0, Q8_0, Q4_K, Q5_K and Q6_K.
**Q3_K is not in it** -- its `d` sits at offset 108 and was left random, so about
3% of super-blocks carried an f16 infinity or NaN and the float64 reference for
that row came out non-finite.

`gemmVsReferenceWin` then computed `if sy2 > 0 { nmse = sse / sy2 }`. **NaN > 0
is false**, so a non-finite reference left `nmse` at its zero initialiser, the
`nmse > 1e-10 || IsNaN(nmse)` check passed, and the subtest logged **NMSE
0.00e+00** -- the most reassuring number a numerical test can print.

    13 of 180 subtests were in that state, Q3_K only, at 2x1, 4x1, 6x1, 2x2, {4,2} ...

**IT WAS FOUND BY THE GATE FAILING TO FIRE, NOT BY ANYTHING FAILING.** The wide
Q3_K path is gated on a deliberate violation (emit at window 256, pack at 32).
That violation fails at 9.2e-3 at 1x1 and PASSED at {4,2} -- the tile that
actually ships. A gate that bites at one tile and not another is not noise; the
tiles are the same code. Chasing that is what surfaced both bugs.

    violation at {4,2}, after the fix:   Q3_K 7.5e-2   Q4_K 4.2e-2
                                         Q5_K 8.2e-2   Q6_K 1.0e-1

A degenerate reference is now a `t.Fatalf`, not a pass: `sy2` NaN, Inf or <= 0
means the comparison proves nothing and says so. **RULE 14r said "a skipping test
is a green line that means nothing"; this is the same failure one step sideways
-- a green line that means the ORACLE broke.** The guard that caused it was
written to avoid dividing by zero, which is a real concern; the fix is to fail
on the degenerate case rather than to score it zero.

★ THE REUSABLE PART: **a per-format fixture with a `switch` and no `default` is a
list that silently omits.** Q3_K was added to `Supported`, to `EmitGEMM` and to
the test matrix, and the one place it was missed had no compiler or test that
could notice. RULE 14u recorded the same shape ("adding a type to `Supported`
inherits no test"); this is its fixture twin.

**★ RULE 6d. THE d_a HOIST IS IN FOR Q4_K/Q5_K/Q6_K, COUNTED, GATED -- AND ITS
SIZE IS UNMEASURED. THE BOX COULD NOT PRODUCE A GATED NUMBER.**

The half of RULE 6b that does not need the int16 bound. At window 256 one d_a
covers the super-block, so it factors out of the per-sub-block term and applies
once per (row, token group): Q4_K/Q5_K carry TWO uses of it (the dot term and
the min term) and Q6_K one, which is the same asymmetry RULE 1d-octies measured
on arm64. Counted per super-block at the shipping {4,2}:

    Q4_K  1712 -> 1608   -6.07%     fold step 10 -> 8 instructions
    Q5_K  1812 -> 1708   -5.74%     fold step 10 -> 8
    Q6_K  1974 -> 1870   -5.27%     fold step  6 -> 5
    Q3_K  unchanged (already integer, RULE 6b)

Weighted by matvec ELEMENT share read from the real GGUF headers -- tinyllama is
Q3_K 59.1% / Q4_K 32.2% / Q6_K 6.3% / Q5_K 2.3%, llama-2-7b is Q4_K 83.0% /
Q6_K 17.0% -- that is **-2.37% of tinyllama's GEMM instruction stream and -5.92%
of llama-2-7b's**. So llama-2-7b is the model this lever is for, which is the
opposite of RULE 6b's.

**WHAT IT MEASURED, AND WHY IT IS NOT A RESULT.**

    llama-2-7b prefill, ABBA, n=5   pass 1   1.0506  IQR 0.039, 2 of 10 samples clamped
    llama-2-7b prefill              pass 2   UNUSABLE -- 5 of 10 clamped, mostly on ONE arm
    tinyllama prefill, n=3                   0.991 / 1.002 / 1.025, median 1.002

Two passes that disagree, on a box reading 124-131 tok/s where it reads 157-163
rested. **RULE 2c's "run the whole comparison twice" caught it and RULE 2d says
what to do: stop measuring and let the box rest.** The 7B is the worst case for
this laptop's sustained-power limit -- 3.8 GB of weight traffic per prefill --
and the clamps land preferentially on whichever arm is running, which is the one
thing ABBA cannot cancel.

**THE CHANGE IS KEPT ON CORRECTNESS, NOT ON A NUMBER**, per RULE 6a: it deletes
instructions, adds no `Spec` field, passes every suite, and fails its deliberate
window violation at 4.2e-2 (Q4_K), 8.2e-2 (Q5_K) and 1.0e-1 (Q6_K) at the
shipping tile. **Do not quote 1.0506.** A regression is not excluded either: the
hoist trades two per-sub-block multiplies for a store/load round trip through
scratch per (row, token group), and store-to-load forwarding is not in the
instruction count.

★ A RESIDUAL LEVER, COUNTED AND NOT BUILT: Q4_K/Q5_K's `VMULPS vMin,vMin,vAux;
VSUBPS acc,acc,vMin` is ONE `VFMADD231PS` if dmin is stored negated, taking the
fold step 8 -> 7 and the super-block to -9.58% (Q4_K) and -9.05% (Q5_K), i.e.
-8.76% of llama-2-7b's instruction stream. It needs a 32-byte sign-flip constant
appended to `KernelConst` at a per-format offset, or a `VPCMPEQD` the assembler
does not have. Q6_K gains nothing from it.

**★ RULE 6b-bis. THE PREFILL ROWS IN THIS FILE COMPARED UNEQUAL WORK, AND THE
HARNESS NOW MAKES THAT IMPOSSIBLE.**

Every prefill number here -- 0.8458, 0.9021, 1.0370 -- ran jitllm on a 573-token
prompt against `llama-bench -p 512`. Those are not the same job. **llama.cpp's
default `n_ubatch` IS 512**, so 573 is one full micro-batch plus a ragged
61-token tail, and on this box llama.cpp reads 146-151 tok/s at `-p 573` against
157-158 at `-p 512`. A row measured that way quotes llama.cpp at a length its
batching handles badly.

Note which way that cuts: it makes the recorded rows CONSERVATIVE toward jitllm on
a same-token-count reading (llama.cpp at 573 would have made the row ~1.11, not
1.037) and OPTIMISTIC on a same-configuration reading. Neither is wrong so much
as unstated, and a row whose value depends on an unstated choice is not a row.

`scripts/vs-llamacpp.sh` grew `JITLLM_MODE=prefill`, so prefill now comes from the
SAME tool as every decode row -- the soak, the ABBA, the PSI gate, the clamp gate
and the IQR gate are all mode-independent and only the two run functions differ.
It runs jitllm ONCE first, reads the token count off jitllm's own output, and passes
that to `llama-bench -p`, so the two engines process exactly the same number of
tokens by construction rather than by coincidence. The default prompt is 46
repetitions = 507 tokens, chosen to sit just inside one micro-batch.

This is RULE 12q's lesson landing a second time: "a scoreboard whose rows come
from two different tools is not a scoreboard". The prefill rows came from a
script typed by hand beside the gated one, and the hand-written one differed in
the one parameter nobody was looking at.

**★ THE MATCHED-LENGTH ROW IS NOT MEASURED YET, AND SAYING SO IS THE POINT.** The
first attempt read 0.8568 at IQR 0.040 with the package at 1725-2205 MHz and one
sample discarded at 939 -- RULE 2d's partial band, where the rule's own verdict is
"stop measuring and let the box rest". jitllm read 128-135 where it had read 163 the
same afternoon, a 21% fall, against llama.cpp's 4%; that is RULE 1d-quater's "jitllm
degrades MORE under sustained load", and it is entangled with the length change
in a way no single pass can separate. **Re-measure both lengths on a rested box.**

What does NOT depend on any of this is the fold's own A/B: 1.0366 is jitllm against
jitllm at ONE prompt length, so the length question cannot touch it. The +3.7% is
solid; where it leaves the ROW is open.

**★ RULE 6b. amd64's Q3_K PREFILL FOLD IS ENTIRELY INTEGER NOW, AND IT IS THE
FIRST PREFILL WIN ON THE BOARD. +3.7% ON A WHOLE tinyllama PREFILL.**

RULE 5d did this on arm64 for DECODE and this file has said ever since that
amd64 could not follow, because amd64 keeps eight per-32 activation scales where
`Spec.ActWin` 256 gives arm64 one. That was true of the DECODE kernel and was
read as true of the engine. The payer is the PREFILL kernel:

    out = d * d_a * SUM_s sc[s] * (dot_s - 4*sumA_s)

At window 256 one `d_a` covers the super-block, so it leaves the sub-block loop
-- and once it has, **nothing in the per-sub-block term is float at all.** The
bracket is |SUM (u-4)*a| over 16 elements, at most 16*4*127 = 8128, so it fits
int16 and ONE `VPMADDWD` against a dword whose high half is zeroed multiplies it
by sc in a single uop. No `VPMULLD`, no convert, no per-sub-block d_a load. The
total is at most 16*32*8128 = 4.2M against int32's 2.1B.

    fold per (row, token group, sub-block)      6 instructions -> 4
    d and d_a                                   per sub-block  -> per super-block
    {4,2} kernel                                15688 -> 14782 bytes

**ONLY Q3_K, AND THE INT16 BOUND IS WHY.** Q6_K's bracket reaches 16*32*127 =
65024 and Q4_K/Q5_K's 32-element sub-blocks reach 32*15*127 = 60960 -- both past
int16, so they would need `VPMULLD`, which is slower than the `VCVTDQ2PS` it
would replace. The wide window is still correct for them and buys them nothing,
which is what makes llama-2-7b the control below.

    tinyllama prefill, ABBA, alternating binaries, n=6      1.0366   IQR/med 0.005
    tinyllama prefill vs llama.cpp, pass 1 / pass 2         1.0379 / 1.0370   IQR 0.003
    llama-2-7b prefill (Q4_K+Q6_K), same A/B     CONTROL    1.0011
    tinyllama DECODE, same A/B                   CONTROL    0.9987

**THE TWO CONTROLS ARE WHAT MAKE IT ATTRIBUTABLE.** llama-2-7b's k-quants get
the wider window and no kernel exploits it, so a move there would have meant the
window flip, the tuner re-keying, or the scratch resize rather than the fold; it
is flat. And decode is flat, so the coarser amax costs no speed there -- the
amd64 decode kernel reads eight per-32 scales that now happen to be equal.

**`nativeHoistsActScale` NOW MEANS "SOME KERNEL EXPLOITS IT", AND THE OLD NAME
WAS THE BUG.** It asked whether the DECODE matvec hoists the scale. The window
is one decision for the whole session -- prefill and decode write the same KV
cache and must quantize identically (`gemmpack.go`) -- so the question was always
"does anything this process generates exploit a wider window", and answering it
about one kernel kept the other one from ever being written.

★ RUN AGAINST TWO VIOLATIONS, per RULE 12m. Dropping the high-half mask, so a
negative scale leaks `acc_hi * -1` through `VPMADDWD`, fails at NMSE 1.2e-8
against a 1e-10 gate. Emitting at 256 while packing at 32 fails at 7.5e-2 at the
shipping tile. The matched pair reads 1e-15.

★ AND THE "0.00e+00, SO THE FOLD IS EXACT" THIS RULE FIRST CLAIMED WAS A BROKEN
ORACLE -- SEE RULE 6c. Those zeros were NaN references, not bit-exactness.

★ AND A THIRD "VIOLATION" THAT IS NOT ONE, WORTH RECORDING SO IT IS NOT CHASED:
reading `d_a` from block 7 of the super-block instead of block 0 PASSES, because
at window 256 the eight blocks genuinely hold the same scale. That is the
invariant holding, not the test being blind.

**★ WHOLE-LAYER FUSION AT DECODE IS MEASURED AND REJECTED.** It carried "~3% from the design recon,
NOT re-measured" for a long time; `model.TestFusionCeiling` now bounds it without building it, since
fusing a block changes no bytes read and no FMAs issued and the entire prize is dispatch round trips
removed. Two real models, quiet box, load-gated:

                       tok/s  regions/tok  per-layer   siblings  +epilogues  all-but-reductions
    tinyllama q3_K_M    74.6      310         14.1       1.30%      3.03%          4.80%
    gemma-2b            27.5      236         13.1       0.41%      0.95%          1.38%
    llama-2-7b Q4_K_M   11.2      434         13.6       0.29%      0.67%          1.01%

The ceiling FALLS as the model grows, which is the right direction and a check on the method: a
bigger model spends longer per token and dispatch is a smaller share of it. The closest call is
tinyllama's deepest column at 4.80%, which is under the bar but not comfortably -- say "just under",
not "far below".

The three columns are three depths of fusion: merging the independent siblings (q/k/v share a normed
vector, gate/up share another); adding the elementwise epilogues that need NO barrier because the
producing worker already owns those rows (rope, both residual adds, the SiLU-multiply); and removing
everything except the three genuine reductions (two RMSNorms and the softmax), which fusion cannot
touch. **Nothing clears 5%**, though tinyllama's deepest case comes within 0.2 points of it.

The reason is that RULE 11 already took this prize. **Total dispatch is now 1.88% of a gemma token**
(236 regions x 2.70 us against 35.6 ms), down from the 37.6% the spin rewrite was aimed at. Fusion
attacks a subset of 1.88%.

Two things the experiment corrected about its own method. The empty-region round trip is the
CONSERVATIVE cost, not an underestimate: splitting real work measures 1.596 us per region against
2.697 us idle, because dispatch overlaps with computation and an idle round trip has nothing to hide
behind. And the first run had no load gate, which is how tinyllama read 4.3 tok/s against its
recorded 68.09 while a download ran -- a 16x error that looks exactly like a catastrophic regression.

Per RULE 6a this rejects the change as measured, not the idea. What would change the answer: a model
whose token time is much shorter (stories15M, at 287 us/token, reads 17-41%), or a layout change that
raises the per-region cost rather than the region count.

**★ AND ONE OF THOSE CONDITIONS IS THE OTHER HOST. THE TABLE ABOVE IS LINUX-ONLY, AND ON THE M4
tinyllama CLEARS THE BAR.** Same test, same build, run on the Mac:

    host    model        regions/tok   round trip   siblings   +epilogues   all-but-reductions
    M4      tinyllama        310         3.008 us     1.66%      3.88%          6.15%   CLEARS
    M4      gemma-2b         236         2.951 us     0.61%      1.43%          2.07%
    Linux   tinyllama        310         2.70 us      1.30%      3.03%          4.80%
    Linux   gemma-2b         236         2.70 us      0.41%      0.95%          1.38%

The mechanism is both halves of the ratio moving the same way: a region round trip costs MORE on the
M4 (3.01 us against 2.70, four cores rather than six behind the barrier) and a tinyllama token is
SHORTER there (11.94 ms against ~13.4), so dispatch is a larger share of a smaller number. gemma does
not clear it on either host, and the ceiling still falls as the model grows.

**So "whole-layer fusion is rejected" was a HOST-SPECIFIC result recorded as a general one**, and the
row it would help most -- tinyllama on the Mac CPU, the worst row on the scoreboard -- is the one it
was never measured against. Two caveats before anyone builds it: the dispatch-cost IQR was 81.5%, so
the test itself says "treat the ceiling as soft", and 6.15% does not close a 0.83 row on its own.
RULE 5l's arithmetic still stands -- the Mac CPU needs the ~13% kernel gap AND the ~13% harness, and
this is part of the second half only.

## ★ RULE 8c: THE CPU IS PROBED NOW. IT USED TO SIGILL ON THE FIRST TOKEN.

Every quantized kernel jitllm emits uses an instruction that is NOT baseline: `VPDPBUSD` is AVX-VNNI
(Alder Lake, Zen 4 and later) and `SDOT` is ARMv8.2-A FEAT_DotProd. Nothing probed for either, and
`cmd/jitllm/hardware.go` said so in as many words -- *"a host without these gets a SIGILL on the first
token rather than a slower kernel"*. **That is a fine thing to write down and not a fine thing to
ship**: the chips that lack them are ordinary, not exotic.

    Cortex-A53 / A57 / A72 / A73   ARMv8.0-8.1, NO dotprod   -- a Raspberry Pi 4 is an A72
    Cortex-A76 and later           ARMv8.2, has it           -- a Pi 5 is fine
    pre-Alder-Lake x86             no AVX-VNNI

(Superseded on arm64: see "arm64 without FEAT_DotProd: SDOT widened" below. The arm64 decline
described next became a refusal once the reference tier was removed, and the widening replaced it.)
`SupportedNative` now declines the quantized types when the feature is absent, so the engine
degrades to the float64 reference -- which RULE 8 already makes a supported tier, so the fallback
needed no new mechanism, only the honesty to use it. Verified on both architectures by forcing the
probe off (`JITLLM_NO_VNNI=1`, `JITLLM_NO_DOTPROD=1`): 0.52 tok/s on Linux, 0.70 on the M4, correct token
ids on both, no crash.

**★ THE PROBE IS ITSELF JIT-COMPILED ON amd64, WHICH IS THE ONLY DEPENDENCY-FREE WAY TO READ CPUID.**
`golang.org/x/sys/cpu` would end `go.mod`'s one-require, no-transitive-deps property that
`engine/nn/device.go` defends, and `internal/cpu` cannot be imported. Eleven bytes of emitted code --
save RBX, CPUID leaf 7 sub-leaf 1, store four registers, restore RBX -- is less machinery than
either, and it is precisely what this package exists to do. arm64 reads `AT_HWCAP` out of
`/proc/self/auxv` on Linux and `hw.optional.arm.FEAT_DotProd` via sysctl on Darwin.

**★ AND THE GATE BELONGS IN `SupportedNative`, NOT IN `Supported`.** Putting it in the emitters broke
`TestA64Disassembles`, which cross-emits NEON from an x86 box: `EmitA64` is documented HOST-PURE --
*"it depends on nothing but s, which is what makes golden-disassembly tests possible at all"* -- and
a CPUID check makes it depend on the machine it runs on. `Supported` answers "can the emitter
produce this"; `SupportedNative` answers "can THIS host run it". They are different questions and
the tests noticed before anything shipped.

**★ F32 IS NEVER GATED**, on either arch: AVX2's `VMOVDQU`/`VFMADD231PS` and ARMv8-A's
`LDRq`/`FMLA4s` are baseline. It is the one kernel that runs on a Pi 4, which is also why
`nn.NewJIT` no longer returns nil when no quantized kernel exists -- doing so threw away the POOL,
so the elementwise ops and the F32 matvec lost their parallelism too and everything ran on one core.

**★ WHAT IS NOT BUILT: the pre-VNNI fallback KERNEL.** `VPMADDUBSW` sits in the assembler documented
as exactly that, and no emitter calls it. Building it would turn a ~100x degradation into a small
one. Declining is merely the version that does not crash.

## ★ arm64 without FEAT_DotProd: SDOT widened, the same bits

Once the reference tier was removed, the decline above stopped being a slow path and became a
refusal: a Cortex-A53/A57/A72/A73 (Raspberry Pi 3 and 4, many boards and older phones) opened no
quantized container -- "no host kernel reads <tensor>" from `engine/model/forward.go`. RULE 8 makes
that a missing basic kernel.

**What was built** (`jit/cpu/sdotemu.go`): the assembler widens every `SDOT` and `SDOTelem` in
place when the probe says the chip lacks the feature --

    smull   t0.8h, vn.8b,  vm.8b        smull2  t1.8h, vn.16b, vm.16b
    saddlp  t0.4s, t0.8h                saddlp  t1.4s, t1.8h
    addp    t0.4s, t0.4s, t1.4s         add     vd.4s, vd.4s, t0.4s

(`dup t1.4s, vm.s[i]` first for the by-element form). An int8 product fits int16 and every sum
after it is int32, so the six instructions are SDOT's four lanes EXACTLY: a widened kernel gives
the SDOT kernel's bits. Every emitter that issues SDOT names two scratch vectors it never holds
live across one (`A64.DotScratch`): v8/v9 in the row-major matvecs (Go's arm64 ABI saves no V
register), t3 and vDA in the packed matvec, v24/v25 in the row-major GEMM (its accumulators end
at v23 by construction), r[0]/r[1] in the stationary GEMM, t1/t3 in the k-quant GEMM, and t3 plus
the correction constant vK in the token-tiled matmul, which rebuilds vK before its epilogue when
widened. The activation quantization, the layouts and the dispatch are unchanged; the probe is the
switch, so `SupportedNative` and `SupportedPackedNative` no longer ask it.

**The gates.** `jit/cpu.TestNoDotProdKernelsEmitNoSDOT` scans 101 kernel families (row-major at
both windows, GEMM, packed, fused, fused-windowed, tiled at every tok, stationary) for SDOT
encodings: present with the feature, absent without it, 19756 SDOTs widened.
`TestNoDotProdMatchesDotBitForBit` runs the packed and fused decode kernels for every
`quant.PackedTypes` entry at one to three outer steps of k and one to three tiles of rows, with a
NaN guard after the output: the widened arm equals the SDOT arm bit for bit and stays within the
oracle. The whole `jit/cpu` suite also runs widened, under `qemu-aarch64-static -cpu cortex-a72`
(the probe reads no ASIMDDP in qemu's auxv, and a stray SDOT would SIGILL there) and with
`JITLLM_FORCE_NO_DOTPROD=1` on a chip that has it. Violation runs: dropping SMULL2's high eight
products (the tail) and doubling one sub-block's scale in the widened packed kernel both fail.

**Model level** (`engine/model/nodot_test.go`, tag `jitllmtest`, arm64): `TestNoDotProdModelGates`
runs `TestEveryModelRunsGenerated` and the golden `TestGreedyMatchesLlamaCpp` with the probe forced
off, and `TestNoDotProdIsBitIdentical` holds every model under 2 GiB to the SDOT kernels' decode and
prefill logits, bit for bit, in one process.

**The six instructions became four, and the decode kernels chain.** The sequence above was replaced
by `smull`, `smull2`, `addp t0.8h`, `sadalp vd.4s`: adding a product PAIR in int16 is exact because
one operand of every dot in `jit/cpu` is an activation or query quantized at amax/127, so a pair is
at most 2*128*127 = 32512. On top of that the container decode kernels (`emitA64Packed`: the tiled
matvec, the fused one and its integer fold) chain the two dots a row group takes per payload word --
the low and the high nibble, or the two halves of a merged Q5/Q6 plane -- into one int16 pair with
`smlal`/`smlal2` and fold it once (`A64.DotChain`), and broadcast each activation element once per
word, before the row-group loop, instead of a `dup` per dot. `DotChainCap` is the bound:
2*links*|w|max*127 <= 32767, so a nibble chains eight, a merged Q6 plane (63) two, an int8 weight
one; the kernels use two. `jit/cpu.TestNoDotProdChainCapIsTheOverflowBound` runs the worst case
(every product +|w|*127) exact at the cap and wrapped one link past it.

Instructions spent beyond the SDOT kernel, per SDOT (`TestNoDotProdChainsTheDecodeDots`,
deterministic, the payload loop is unrolled): the per-SDOT widening was +6.00 everywhere; now
+1.56 (Q4_0, Q4_K, Q3_K, MXFP4 tiled) to +1.75 (fused), +2.06/+2.25 for the merged planes
(Q5_0, Q5_1, Q5_K, Q6_K) and +3.06/+3.25 for Q8_0. Whole fused Q4_K kernel at 16 rows: 1586 words
with SDOT, 3122 with the six-instruction widening, 2034 now (1.53x fewer). The prefill kernels
(tiled matmul, stationary and row-major GEMM) take the four-instruction widening per SDOT: +3 a
vector SDOT and +4 a by-element one (its `dup`), where they were +5 and +6.

**The layout was not changed.** The container puts four rows' word w in one vector, a row per
int32 lane, and the chain keeps rows apart in the int16 lanes without a horizontal reduction, so
the reduction is once per word and row group. Chaining further -- across words, an int16 pair per
row group held for the sub-block -- would save the remaining `addp`/`sadalp` per word but costs two
registers per row group, which the fused kernel at 16 rows does not have; nothing measured asks for
a regrouped layout yet. What is owed is the measurement: the A/B of this kernel against the
per-SDOT widening on a chip without FEAT_DotProd (or the M4 with the probe forced off).

## ★ RULE 8: `nn.JIT[t] == nil` is a supported configuration.

Not a bug path. It is the fallback, the bisection tool, and the oracle, all one mechanism. Any
change that makes the engine require generated code to produce correct tokens is wrong.

## ★ RULE 8b: the reference tier's number, so later tiers have a ratio.

    jitllm bench <model>

Reference tier, one core, float64, every weight row dequantized per token:

    tinyllama   0.276 tok/s   520,950,640 B/token   0.14 GB/s =  1% of the wall
    stories15M  46.14 tok/s    13,166,370 B/token   0.61 GB/s =  3% of the wall

Generated code, Q4_0 + Q8_0 + Q4_K. gemma-2b decode, six pinned P-cores:

    35.23 GB/s = 70% of the wall = 21.07 tok/s   (llama.cpp: 27.3, so 77% of it)

`JITLLM_PROFILE=1` gives the per-op split, and it is the only way to find what to
generate next. The history is the argument for measuring rather than guessing:

    16.4 tok/s  matvec 86%  activation 11.6%  serial 20% (quantize 19%)
    17.0        + shape-specialized kernels (K baked in, loop unrolled)
    19.8        + parallel activation, + cache redundant quantization
    21.1        + parallel quantization, branch-free inner loop
    24.2        + interleave width re-measured: pack2, not pack8 (rule 5b)
    27.4        + spin dispatch, caller as worker 0 (rule 11)   <- llama.cpp is 27.3

Two of those were pure waste rather than missing optimization: gemma feeds the
SAME vector to q, k and v and then to gate and up, so seven matvecs per layer
needed only four quantizations; and the activation ran on one core while five
idled.

**Per-shape, in isolation, the kernel and pool are fine** — the whole-model number
is a weighted average of shapes with very different efficiency:

    gemma gate/up   16384 x  2048   49.49 GB/s
    gemma lm_head  256128 x  2048   41.49 GB/s
    gemma q/o        2048 x  2048   34.74 GB/s
    gemma ffn_down   2048 x 16384   33.13 GB/s
    gemma k/v (GQA)   256 x  2048    9.88 GB/s   <- dispatch-bound

Small matvecs are dispatch-bound. Medium ones are instruction-bound AT ONE CORE
and DRAM-bound at six, which is the correction M7 produced:

**4-row interleaving helps single-core and is neutral at six.** It amortizes the
six per-block instructions that depend only on the activation side, and measures
+24% on one core (10.18 -> 12.62 GB/s end-to-end on gemma). Paired and
interleaved on a 255 MB DRAM-resident set at six cores it measures **0.93x** at
12% IQR -- inconclusive, and certainly not the 1.36x the instruction count
predicts. At six cores both kernels reach ~42 GB/s = 78% of the wall, so the
binding constraint there is DRAM, not instruction issue. More instructions per
byte only matters while there is issue bandwidth to spare.

Beware the shape table above: it repeats the same matvec, so anything under the
24 MB L3 is measured hot in cache. Any figure over ~55 GB/s is an L3 number and
unreachable in a real decode, where each tensor is read once per token.

**`BytesPerToken` must add the embedding back when it is tied.** gemma's
token_embd is 557 MB of Q8_0 and IS the lm_head, read in full every token.
Subtracting it as a mere lookup under-reported gemma's traffic by 1.5x, which
made a healthy 34.9 GB/s read as 18.75 and sent a whole investigation after a
phantom kernel problem. Check the denominator before believing the ratio.

tinyllama does not move at all yet - it is 100% k-quants and has no kernel.

That headroom is what M6-M8 are claiming. `bench.MemWall` is a Go-side probe
and reads LOW — ~36 GB/s across six threads where a pinned C benchmark with AVX
loads reads 55.7 — because Go emits scalar 8-byte loads and cannot pin a
goroutine. It is a sanity bound, not the hardware peak, which is why the physics
guard allows 2x over it. That factor tightens when `sched` lands pinned workers.

## ★ RULE 14v: DECODE'S DEPTH DECAY IS ATTENTION GOING 1% -> 30%, AND THE HOST KV
CACHE HAS THE LAYOUT RULE 12aa ALREADY FIXED ON THE DEVICE.

tinyllama, six P-cores, `JITLLM_PROFILE=1` at three depths:

    depth    matvec   attention   rmsnorm   activation   tok/s
        0     95.7%       1.0%      1.0%        1.5%      82.3
      512     85.7%      11.3%      0.9%        1.4%      72.7
     1536     67.6%      30.0%      0.7%        1.2%      58.2

**★ AND THIS RETIRES RULE 8b's 52.9%.** That rule records "at 821 tokens
attention is 52.9% and matvec 45.0% while decode falls from 68.6 to 11.0 tok/s".
It was true of the code that measured it and is not now -- 1536 positions read
30.0%, and decode falls 82.3 to 58.2 rather than to 11.0. Quote these.

**★ 30% OF THE TIME FOR 12% OF THE BYTES, AND THE STRIDE IS A CANDIDATE
MECHANISM -- NOT, AS THIS RULE FIRST SAID, THE REASON.** Comparing unique KV
bytes against streaming weight bytes locates where to look; it does not identify
a bottleneck, and I wrote it as though it did.** At depth 1536
K+V is 69 MB per token against ~490 MB of weights. `EmitAttnScores` advances the
K cursor by `kvStride*4` per position (`jit/cpu/attn.go:65`) and reads `hd`
floats from each; `EmitAttnAcc` walks V the same way. On tinyllama that is **256
useful bytes out of every 1024-byte stride**, and the pool splits over QUERY
heads with `kvh = hh/gqa`, so a worker holding heads 0-5 reads the same quarter
of every row and never touches the rest.

**★★ AND THE DEVICE PRECEDENT IS THE OPPOSITE OF WHAT I CLAIMED. RULE 12aa's OWN
TABLE SAYS THE TRANSPOSE LEFT DECODE UNCHANGED.**

    554-token prompt   [position][head][dim]  463 399 401 ms
                       [head][dim][position]  369 318 330 ms   1.24x
    decode              56 59 57  ->  55 57 60                 UNCHANGED

I cited it as the precedent for a CPU DECODE win and it is the one measurement
of this exact change on decode, saying it does nothing. It also transposes K
ONLY. The single thing that could reconcile it with a depth-1536 host win is
that its decode row was taken at `verify`'s shallow depth, where attention is
~1% of a token by this rule's own table -- i.e. at a depth where nothing about
attention could show. **That is a reason to measure, not a reason to assume.**

★ THE OTHER CAVEAT: on the device the skipped sectors are simply lost, while on
the host the skipped 768 bytes belong to OTHER kv heads that OTHER workers read,
so L3 may recover some of them. Amplification is between 1x and 4x. **Do not
inherit 1.24x for the host.**

**★ AND A LAYOUT CHANGE IS NOT THE CHEAPEST EXPERIMENT. THE POOL SPLIT IS.**
`s.jit.Parallel(c.NHead, 1, ...)` hands out ONE query head per task, so adjacent
heads sharing a kv head land on different workers. `Parallel(c.NHead, 2, ...)`
puts each sharing pair on one worker -- a one-argument change that tests whether
scheduling locality matters at all, before anything is restructured. Price that
first.

**★ AND IF THE LAYOUT DOES WIN, THE CPU WANTS [head][position][dim], NOT THE
DEVICE'S [head][dim][position].** Keeping `dim` contiguous means both emitters
already express it -- pass `kvStride = headDim` and their loop logic is
unchanged. The device's dim-major form would need a different vectorisation
strategy entirely.

★ AND MigrateKV WOULD NOT BECOME A STRAIGHT COPY, which I assumed it would.
Three layouts would then be in play -- host [head][position][dim], device K
[head][dim][position] at MaxSeq+1 with a trash slot for padded prefill, device V
[position][head][dim] -- plus the NoKT case and the rejected packed-f16 V. Its
opening "same layout" comment is already stale.

**★ 14v-bis. THE A/B HARNESS COULD NOT MEASURE ANY OF THIS, AND THAT IS FIXED
BEFORE THE EXPERIMENT RATHER THAN AFTER.** `verify -ab` decoded from position
ZERO and `Reset()` at n, so every arm ran at a context of at most n tokens --
and this rule's own table puts attention at 1.0% of a token there against 30.0%
at 1536. **An arm about attention was noise at depth 0 by construction**, which
is a property of the harness and not of the change being tested. RULE 11's
sentence for the fourth time in this file: measure the harness before the code
it runs.

`-depth` prefills to a context and sizes the state for the WHOLE run rather than
resetting back, because returning to depth means prefilling again and at 1536
that costs more than every decode step in a round put together -- it would be
timed inside the arm and swamp it.

★ AND `abRun` REQUIRED A DEVICE FOR EVERY ARM. split, head, submit and graph
compare two DEVICE configurations; `attnchunk` compares two HOST pool layouts,
and with the blocks resident the host attention it measures does not run at all.
Host arms no longer attach a device.

★ THE ARM ITSELF, ready and unmeasured: `-ab attnchunk` alternates
`Parallel(NHead, 1)` against `Parallel(NHead, 2)` as a FIELD on the State, so
both are live in one process (RULE 6, RULE 12a). A smoke test at depth 512 ran
and reported a number; **it is not a result** -- three rounds on a box clamped at
968 MHz under load, where an IQR of 0.0% is a sample-size artifact. Run it at
depth 0, 512 and 1536 on a rested box.


**★ 14v-ter. AND THE CHEAP EXPERIMENT SAYS NO. THE POOL SPLIT IS NOT THE LEVER.**
`-ab attnchunk` at depth 1536, tinyllama, six P-cores, in-process ABBA, n=15:

    chunk   ratio vs 1 head/task   IQR/median
        2        0.9850               6.0%     noise
        4        0.9411               2.5%     gated, and WORSE
        8        0.6059              25.9%     REFUSED by the gate

**Monotone in the wrong direction.** If the strided K read were being recovered
by putting the query heads that share a kv head on ONE worker, this would
improve; it degrades. chunk=8 is the full hypothesis -- all eight heads of a
sharing group on one worker -- and it is also 32 heads over 8 = FOUR tasks for
SIX workers, so two sit idle and ~1.5x is what load imbalance predicts. Its
25.9% dispersion is refused rather than quoted, which is the gate doing its job.

**★ WHAT THIS DOES AND DOES NOT REFUTE.** It refutes the SCHEDULING route: no
arrangement of query heads across workers recovers the stride, and the cost of
coarsening the split exceeds anything locality buys. It does NOT test the stride
MECHANISM, because co-scheduling and layout are different things -- a worker
holding all eight heads still issues eight separate strided traversals of the
same rows. Testing the mechanism needs the layout itself: build a head-major
scratch buffer and compare the SAME kernel at `kvStride = kvDim` against
`kvStride = headDim`, which prices it without touching the cache, MigrateKV or
any of the six call sites.

★ THE EXPERIMENT COST TWENTY MINUTES AND WOULD HAVE COST A DAY AS A REFACTOR.
That is the whole argument for pricing the cheap version of a hypothesis first,
and it is why the review's ordering -- "my first source-level experiment would be
chunk 2" -- was right even though the answer was negative.

★ MEASURED IN RULE 2d's PARTIAL BAND (1377 MHz under load, 60 C, PSI 0.00). The
chunk-4 row is gated at IQR 2.5% and the direction is monotone across three
widths, so the CONCLUSION stands; the magnitudes should be re-taken rested.


★ AND THE SOFTMAX IS NOT THE LEVER, WHICH IS WORTH WRITING DOWN BECAUSE IT LOOKS
LIKE ONE. Decode converts the score row f32->f64, scales, softmaxes in f64, and
converts back -- four or five touches of a row the kernels touch once, and ~2.2M
conversions per token at depth 1536. But the row is `pos` long where the kernels
do `pos*hd` work, so the whole of softmax and its conversions is about 8% of
attention and 2% of the token. An f32 softmax saves a fraction of that. Priced
before building, per RULE 5z.

★ nn.Softmax's comment already settled the OTHER softmax question and should not
be re-run: the online form is the same speed as three-pass on unsorted rows
(1.031x at n=821) and 111x worse on ascending ones, so three-pass ships for
having no data-dependent cost. That comment also states, correctly, that it does
NOT measure flash-attention fusion, which is this rule's subject.


## ★ RULE 14w: PAIRED ATTENTION KERNELS. ONE K WALK AND ONE V WALK SERVE TWO
QUERY HEADS, AND IT IS +8.3% TO +9.5% OF A DECODE TOKEN AT DEPTH 1536.

RULE 14v-ter refuted the SCHEDULING route: no arrangement of query heads across
workers recovers the strided KV read, because a worker holding all eight heads
of a gqa group still issues eight separate traversals. **Sharing the LOAD is a
different thing from sharing the cache, and only a kernel can do it.**

`cpu.EmitAttnScores2` loads each K vector ONCE and dots it against two queries;
`cpu.EmitAttnAcc2` loads each V vector once and accumulates it into two outputs
under their own weights. Against the SHIPPING configuration -- one head per
task, unpaired -- on tinyllama, six P-cores, whole-token decode, in-process ABBA:

    depth 1536   1.0828 (IQR 2.6%)  and  1.0953 (IQR 4.3%)   two gated passes
    depth  512   1.0570 (IQR 10.3%)  REFUSED, direction only
    depth    0   1.0082 (IQR 5.6%)

★ THOSE DEPTH LABELS ARE THE STARTING DEPTH, NOT THE DEPTH -- SEE 14w-ter. One
State runs the whole A/B and its context GROWS, so "depth 0" at rounds 15 and
n 32 actually sweeps 0..976 and reports the average.

**NEUTRAL WHERE ATTENTION IS 1% OF A TOKEN AND A WIN WHERE IT IS 30%**, which is
the effect tracking its own mechanism rather than a number that happens to be
positive (RULE 5d). Splitting it: accumulate-only is 1.0488 and 1.0376 at IQR
3.8%, scores-only is 1.0329 at IQR 12.6% and REFUSED -- so the V walk carried the
measured half and the K walk was unattributed.

**★ THE K WALK IS ATTRIBUTED NOW, AND IT TOOK A CONTROL ARM THAT CAN ITSELF BE
PAIRED.** `JITLLM_AB_PAIR_A` sets what the CONTROL shares, so A=2 against B=3
prices the SCORES pairing ON TOP OF accumulate pairing -- the configuration that
ships -- instead of against an unpaired arm whose dispersion swallowed it. On the
M4, context 1536..2046:

    both paired vs accumulate-only   1.0453 (IQR 1.9%)  and  1.0433 (IQR 1.4%)

**Two passes agreeing to 0.2% where the isolated scores-only arm read 12.6%
dispersion.** The K walk is worth ~4.4% at depth and the two halves are of the
same order, which is what the mechanism predicts -- one strided traversal each.
**Do NOT divide the isolated ratios to get here** (RULE 5y); the marginal value
of one half is measured in the presence of the other, not composed from two
solo runs.

★ TWO HEADS, NOT FOUR, AND THE REGISTER FILE DECIDES. Each head keeps the four
accumulator chains EmitAttnScores uses for FMA latency, so two heads are eight
of sixteen YMM plus the shared K vector. Four would be sixteen accumulators
before loading anything. `EmitAttnAcc2` cannot even hold two whole 64-wide
outputs -- that is sixteen registers before the weights and V -- so it tiles the
dimensions in halves: four accumulators per head, two broadcasts, one V, eleven
of sixteen. **The tile costs a second position walk, and that is not a second
read of V**: each walk touches a different half of every vector.

**★ THE GATE IS BIT EQUALITY AT TWO LEVELS, BECAUSE THE KERNELS CHANGE NO
ARITHMETIC.** Same FMA operands, same chain assignment, same reduction tree, same
position order. `jit/cpu`'s gate holds the paired kernels to two single-head
calls bit for bit across nine shapes including npos 1, 2, 1536 and 1537; the
model gate holds whole decode to the unpaired path across all four sharing modes
and three chunk widths. Violations that fire: a weight cursor advanced once for
two heads, a reduction reading the other head's chain, both heads accumulating
into one chain, and the two heads' weights swapped.

**★★ AND THE FIRST VERSION OF THE MODEL GATE PASSED WHILE TESTING NOTHING.**
stories15M is 6 heads and 6 kv heads, so gqa is ONE, the pairing condition is
false, and two deliberate violations passed. Worse, the paired branch was dead on
EVERY model: pairing needs both heads in one task and the pool was handing out
ONE head per task, so `hh+1 < hi` was never true. **The test now asserts that
head pairs were actually served** -- `State.AttnPaired()` -- and that assertion
is what found the dead branch. RULE 14r for the sixth time in this file, in a
test written the same day as five citations of it.

★ SO PAIRING FORCES A CHUNK OF TWO, AND THE HONEST A/B SAYS SO. Comparing
paired-at-2 against unpaired-at-1 charges the pairing for a scheduling change
that RULE 14v-ter measured separately at 0.985. `-ab attnpair` therefore has two
controls: the default isolates the pairing (both arms at chunk 2) and
`JITLLM_AB_SHIP=1` prices it against what ships. The numbers above are the
SHIPPING comparison, measured rather than composed from the two ratios -- which
is RULE 5y's warning.

**★ 14w-bis. THE arm64 TWINS EXIST, GATED ON THE M4, AND THEY GIVE THE MAC ITS
FIRST CPU WIN.** RULE 14w said arm64 "falls back to two single-head calls"
because I could not execute or gate an emitter I could not run. The Mac is on a
point-to-point cable -- ssh works, ping is blocked, and it is not in the ssh config, which
is why four earlier probes found nothing.

`EmitAttnScores2` and `EmitAttnAcc2` now have NEON twins. **arm64 has 32 vector
registers to amd64's 16, so neither compromise amd64 makes is needed**: two
heads' four chains are eight of thirty-two rather than eight of sixteen, and
`EmitAttnAcc2` keeps the single-head block width instead of halving the
dimension tile, so it pays no second position walk.

    M4, tinyllama, depth 1536, in-process ABBA against the shipping path
      pass 1  1.1033x  (IQR 2.3%)
      pass 2  1.1033x  (IQR 0.9%)

**IDENTICAL TO FOUR DECIMAL PLACES ACROSS TWO PASSES**, which is the M4 being
thermally stable rather than anything about the change (RULE 1d: its IQRs run an
order of magnitude under the laptop's).

**★★ AND THE BOARD ROW CROSSES PARITY. THE MAC HAD NO CPU WIN AT ALL.**
`scripts/vs-llamacpp.sh cpu` at `JITLLM_DEPTH=1536`, n=320, both arms measured:

    pairing OFF   0.9364   IQR/median 0.013
    pairing ON    1.0274   IQR/median 0.001   ★ jitllm WINS

**0.001 is the tightest row in this file.** And the attribution is two measured
arms, not the A/B ratio composed with a board row -- RULE 5y's warning.

★ NOTE WHAT THE DEPTH ROW ITSELF SAYS, SEPARATELY FROM THE CHANGE. The Mac's
depth-0 CPU row is 0.8276 (RULE 5u) and its depth-1536 row is 0.9364 even with
pairing OFF, so **jitllm degrades LESS with context than llama.cpp does on the
M4** -- the opposite of RULE 14i's finding on Linux. Two hosts, opposite
directions, and only the Linux one had ever been measured.

**★ AND RULE 14w's COMMIT BROKE THE arm64 BUILD OUTRIGHT.** `engine/nn/attn.go` is
tagged `amd64 || arm64` and called `cpu.EmitAttnScores2`, which existed only in
the `amd64`-tagged file. The comment claiming arm64 "falls back" was written
without ever compiling for arm64. **`GOOS=darwin GOARCH=arm64 go build ./...`
reproduces it on the Linux box in seconds and costs nothing** -- run it before
claiming anything about the other architecture.

★ AND `JITLLM_ATTN_PAIR=0` MEASURED FULL PAIRING AND REPORTED IT AS THE CONTROL.
It returned -1 for "none", and the call site tests `pair != 0` and then masks
with `pair&1` / `pair&2` -- and -1 is all ones. The first "pairing off" board row
read 1.0268 against 1.0274 with it on, which looked like the change doing
nothing and was the control doing nothing. **The fourth configuration today that
was never selected**, after the pair counter that counted fallbacks, the gqa
that made pairing unreachable, and the layout flag that was never wired.

★ scripts/cap REFUSES ON macOS -- no systemd, so no cgroup exists to cap with --
and Darwin now takes a PLATFORM branch that runs the command with a loud notice.
**I first "fixed" this by making `scripts/cap` honour `JITLLM_NO_CAP` and that was
a mistake, reverted the same hour.** The two layers' escape hatches must stay
distinct: `JITLLM_NO_CAP=1` belongs to the HOOK, where it lets a command run
WITHOUT scripts/cap and is visible in the command line, while making it also
disable the cgroup INSIDE scripts/cap meant `JITLLM_NO_CAP=1 ./scripts/cap 8G --
go test ./...` would READ as a capped run and not be one. **A bypass visible in
the command is a decision; a bypass that makes a capped-looking command uncapped
is a trap** -- and RULE 3c records systemd-oomd killing GNOME Shell three times,
which is what the trap would have cost. A platform branch cannot be set by
accident, exported into a Linux shell, or used to disguise a Linux run.


★ MEASURED AT 1681 MHz, the top of RULE 2d's partial band. Every quoted row is
gated and each was taken twice; the magnitudes want a rested box.


**★ 14w-ter. SHOULD THE TUNER DECIDE WHETHER TO PAIR? NO -- BECAUSE THE DECISION
HAS ONE ANSWER, AND FINDING THAT OUT EXPOSED A HARNESS FLAW.**

The question is a good one: pairing's benefit is proportional to attention's
share of a token, which runs 1.0% at depth 0 to 24-30% at 1536, so its optimum
looks context-dependent -- and **no tuner in this engine could express that**.
Every one of them (pack width, GEMM tile, participant count, prefill chunk, GPU
API) caches an answer keyed on (CPU, cores, shape set): properties of the machine
and the model, fixed for the session. Context length changes DURING a
generation, so a cached answer would be wrong by construction.

**★ AND `-depth` IS THE STARTING DEPTH, NOT THE DEPTH.** `abRun` runs ONE State
for the whole A/B and only resets at the cache limit, so `-depth 0 -rounds 21
-n 32` sweeps 0..1360 and reports the average. A review said this in as many words
-- "report the actual depth range; depth 1536 is its starting depth" -- and I
did not act on it, then read 1.0560 as a depth-0 result. **An arm whose value
depends on context length is misread by exactly this.** The harness now prints
the range it actually covered.

Measured properly on the M4, where an IQR of 0.4% can tell neutral from a 1.5%
loss:

    context 0..80      1.0046   IQR 0.4%    neutral
    context 0..1360    1.0560   IQR 3.9%
    context 1536..2512 1.0967   IQR 4.2%

**PAIRING IS NEUTRAL WHEN SHALLOW AND WINS INCREASINGLY WITH CONTEXT. IT NEVER
LOSES.** So the constant is right and a tuner would be machinery for a decision
with a single answer -- and RULE 1d-undecies already prices that: the tuners cost
2.4% at four cores and 3.9% at six WHILE THEY RUN, and "for a one-shot prompt the
DEFAULT is what ships, and the tuner is a cost."

★ THE THING THAT WOULD CHANGE THE ANSWER, recorded so it is re-checked rather
than re-derived: pairing forces a pool chunk of 2, and RULE 14v-ter measured
chunk 2 ALONE at 0.985. That ~1.5% is the loss a shallow context would pay for
nothing, and 1.0046 at IQR 0.4% says it does not appear -- on the M4.

**★ AND LINUX AGREES, WHICH CLOSES THE ONE HOST THAT HAD NEVER RESOLVED IT.**
The Linux row was open because a "depth 0" run there swept 0..976 and its
dispersion was 5.6%. Shortening the sweep to a genuinely shallow context
(0..240, `-n 8 -rounds 15`) against the SHIPPING arm, two passes:

    pass 1   0.9998   IQR/median 4.7%
    pass 2   1.0127   IQR/median 3.5%

**Both inside their own dispersion, straddling 1.0, on the host where the chunk-2
penalty was most likely to show.** So the ~1.5% does not appear on either
machine, and shallow pairing is neutral rather than merely cheap.

**THE CONSTANT STANDS ON BOTH HOSTS AND THE THRESHOLD IS NOT NEEDED.** If a host
is ever found where shallow pairing LOSES, the right shape is still not a duel
but a THRESHOLD on `pos`, tuned once, since the threshold IS a machine property
and fits the existing cache. Nothing measured so far calls for one.


## ★ RULE 14x: A HEAD-MAJOR KV CACHE LOSES 9% END TO END, AND THE 1.90x THAT
SENT ME BUILDING IT WAS A SINGLE-THREADED PROBE.

RULE 14v-ter killed the scheduling route and RULE 14w attacked the redundancy
from the other side by sharing each load between two heads. What neither tested
is whether the STRIDE itself costs anything -- pairing halves the number of
traversals and leaves the waste per traversal exactly as it was.

`model.TestAttnLayoutProbe` (opt-in, `JITLLM_LAYOUT_PROBE=1`) answers it without
building anything: the same data in two already-resident layouts, the SAME
shipped paired kernels instantiated at `kvStride = kvDim` and `kvStride = hd`,
every query head through the real pool. tinyllama's shape, 32 heads, 4 kv heads,
hd=64:

    npos    hot single buffer    rotating 22 layers    A/A control
       1        0.9952               0.9833             1.0008
     513        1.1502               1.2500             0.9869
    1537        1.5649               1.8973             1.0015
                                     IQR/median 3.0%

**★ THE ROTATING NUMBER IS LARGER THAN THE HOT ONE, WHICH IS THE OPPOSITE OF
EVERY OTHER SUCH RESULT IN THIS FILE.** Seven times (RULE 5b, RULE 6, RULE 8b)
a win measured on a hot buffer shrank or reversed at the real working set, and I
built the rotating arm expecting the eighth. It goes the other way for a
mechanical reason: one layer's K+V at 1537 positions is 3.1 MB, so a single
resident buffer sits in a 24 MB L3 and the skipped bytes cost nothing, while 22
layers are 69 MB and the same skipped bytes are DRAM traffic. **The hot case
UNDERSTATES a stride effect. That is worth remembering the next time a
microbenchmark disagrees with the model.**

★ AND THE A/A CONTROL IS WHAT MAKES IT READABLE: the same arm against itself
reads 1.0008, 0.9869 and 1.0015. A 1.90x with a drifting harness would prove
nothing; this harness is flat.

**★ WHAT IT IS WORTH, BY THE PROJECT'S OWN ARITHMETIC.** Attention is 23.9% of a
decode token at depth 1536 NOW -- measured after RULE 14w, not the 30.0% from
before it. Estimated whole-token saving is `f * (1 - 1/r)` = 0.239 * (1 - 1/1.897)
= **11.3%**, against a 2% bar set before the measurement. It clears by five
times.

**★ AND IT IS STILL NOT A LICENCE TO REFACTOR THE CACHE.** The probe compares two
ALREADY-RESIDENT layouts. It excludes every cost a production head-major cache
would carry: the append on every token, prefill's chunk writes, and MigrateKV,
which would then hold THREE layouts at once (host head-major, device K
[head][dim][position] at MaxSeq+1 with a trash slot, device V row-major). What
this earns is a whole-decode experiment against the paired shipping path with
those costs charged, not the refactor itself.

**★★ AND THE WHOLE THING ABOVE IS WRONG, BECAUSE THE PROBE WAS SINGLE-THREADED.**
The review specified "each measured iteration processes ALL query heads, through the
existing six-worker pool with chunk 2". I wrote a plain loop. Put on the real
pool, the same probe reads:

    npos=1537   hot single buffer   1.4890  (IQR  4.3%)
    npos=1537   rotating 22 layers  1.1693  (IQR 18.6%)  UNSTABLE
                A/A control         0.9423  (IQR 23.5%)  -- unreadable

**THE MECHANISM IS THE ONE I RAISED FIRST AND THEN FAILED TO TEST.** One thread
walking 32 heads reads kv head 0 for heads 0..7 and never touches the other
three quarters of each 1 KB row until much later; SIX workers read four kv heads
at once and consume those bytes between them. So the pool already recovers most
of the stride waste, and the layout has little left to win. The "rotating beats
hot" result was real and beside the point -- it measured how a SINGLE thread
behaves at two working-set sizes.

**★ END TO END IT IS A 9% REGRESSION.** tinyllama, depth 1536, six P-cores,
48 decode tokens, three runs each:

    row-major   59.30  58.40  57.96 tok/s
    head-major  52.58  53.41  53.67

and the profile says attention did not get faster at all: 23.9% of a token
row-major against 23.1% head-major, while the token itself grew 10%. The read
gain the pool leaves on the table does not cover the write, which becomes nKV
scattered copies of headDim floats where row-major is one contiguous store.

**★ SO THE LAYOUT IS SHELVED, AND THE CODE STAYS BEHIND A FLAG THAT DEFAULTS
OFF.** `JITLLM_KV_HEADMAJOR=1` / `model.SetKVHeadMajor`. Do not enable it for
decode. `TestKVLayoutsAgree` holds the two layouts bit-identical across decode,
prefill and mixed, and three violations fire, so the option is correct -- it is
merely slower.

★ WHAT IS *NOT* TESTED, and is the one reason to keep the code: this is a DECODE
measurement. RULE 12aa's device transpose won 1.24x on a PREFILL and explicitly
left decode unchanged, and prefill reads the cache with a completely different
access pattern (a chunk of query rows against a growing history). Head-major may
still pay there. Nobody has looked.

★ AND `kvLayout` ITSELF IS KEPT ON ITS OWN MERIT. Six open-coded address
expressions -- decode's attention, prefill's, the batch's, the cache write and
two migration calls -- derived the same offsets by hand, which is the shape RULE
7l is about. They are one type now, and the head-major branch is what proved it
can express more than one layout.

★ ONE MORE TRAP ON THE WAY, AND IT IS THE THIRD TODAY. The first wiring of the
flag missed -- `s.kvl` was built without `headMajor`, so every "the two layouts
agree" result was two ROW-MAJOR runs agreeing with each other, and all three
deliberate violations passed. **"The two configurations agree" is worth nothing
until you have checked that the second configuration was ever selected**, which
is the same defect as the pair counter that incremented on fallbacks and the gqa
that made pairing unreachable.


★ THE CPU WANTS [head][position][dim], NOT THE DEVICE'S [head][dim][position].
Keeping `dim` contiguous means both emitters already express it -- pass
`kvStride = headDim` and not one line of their loop logic changes, which is why
this probe needed no new emitter at all.

★ MEASURED AT 1299 MHz, inside RULE 2d's partial band. The effect is 90% and the
A/A control is 1.0015, so the CONCLUSION is not in doubt; the magnitude wants a
rested box.



## ★ THE ROTARY TABLE IS GENERATED ON ALL THREE TIERS, AND A ONE-ULP TABLE IS THE FLOOR.

`nn.Rope.Table` was the last Go transcendental arithmetic on the token path:
`math.Cos`, `math.Sin` and `math.Mod` in a loop of NRot/2 on every token, and
once per PROMPT POSITION in a prefill -- a 512-token chunk ran 65536 of them.
It is `cpu.EmitRopeTable` now (`jit/cpu/ropetab.go`, `sse_ropetab.go`,
`ropetab_a64.go`), with no Go arm outside the `jitllmtest` build.

**The design is `exp.go`'s and the reasoning is `exp.go`'s**: range-reduce,
Horner, no table and no branch, with the precision moved into CONSTANTS folded
at model load where float64 is free. The position is decomposed into four 7-bit
digits and each digit's weight is folded mod four QUARTER-TURNS at setup, so the
kernel never forms a large angle:

    pos = d0 + 128*d1 + 128^2*d2 + 128^3*d3
    v   = pos*g*2/pi = sum_i d_i * U_i  (mod 4),  U_i = fmod(g*128^i*2/pi, 4)

★ **AND THE DIGIT PRODUCTS ARE EXACT BY BIT COUNT, WHICH IS THE WHOLE TRICK.**
`H_i` is `U_i` rounded to a multiple of 2^-12, so `H_i <= 4` needs at most 15
significant bits, a digit needs 7, and 7+15 = 22 < 24. The four products are
multiples of 2^-12 summing to at most 2032 -- 8323072 units, under 2^24 -- so
their SUM is exact too, and `x - ((x+M)-M)` with `M = 2^25` folds it mod four
exactly in three instructions. Only the residual plane `L_i = U_i - H_i` rounds,
at ~1e-9 quarter-turns.

### Where it lands: 6.6e-08 radians, and 18.7% of entries differ by one ulp

Against `math.Cos`/`math.Sin` over 420 tables (15 rotary geometries x 28
positions from 0 to 2^24), worst |d| is **6.629e-08 on AVX2 and on NEON**
(identical -- both fuse) and **1.036e-07 on the legacy-SSE tier**, which has no
FMA. Against the float64 table it replaced, worst |d| is **5.960e-08**, one ulp
at 1.0, at every position from 5 to 131071.

★ **AND THE ONE-ULP DISAGREEMENT IS A FLOOR, NOT A DEFECT -- EVERY ARM
SIMULATED IN Go, Qwen3-1.7B's geometry, 8192 entries:**

    as shipped                                18.66%   worst |d| 6.5e-08
    two-word r + back-correction              18.82%   BUYS NOTHING
    the quarter-turn computed EXACTLY in f64  18.46%
    r = float32(the true reduced angle)       12.35%
    that, and math.Sin/math.Cos of it         11.94%   <- the FLOOR

Computing the quarter-turn exactly in float64 moves **0.2 points**; using
`math.Sin`/`math.Cos` in place of the two polynomials moves **0.4**. Twelve
points are left, and they are that a reduced argument held in ONE float32
already carries half an ulp -- sine of a rounded argument is not the rounded
sine. **So the reduction is not the term and neither is the polynomial.**

★ **CARRYING r IN TWO WORDS WAS TRIED AND DID NOT HELP.** The obvious remedy is
a double-float `r` with the polynomials corrected by `sin(r+e) = sin r + e*cos r`
and `cos(r+e) = cos r - e*sin r`. It reads **18.82%** -- no better -- because
each correction is itself a float32 add at the OUTPUT's magnitude and rounds
away exactly what it recovered. Nine instructions and two registers for nothing.
Only a polynomial kept entirely in two words would close the gap, and that is a
different kernel.

### The op is 5.78x and the token is a null

`model.TestRopeTableDose` (tags `jitllmbench jitllmtest`), Llama-3.2-1B
Q4_K_M, six P-cores, ABBA inside one process, A/A self-control FIRST, 15 rounds,
cores 2.0-2.8% busy:

| | ratio | IQR/median | n |
|---|---|---|---|
| **the op**, A/A control | 1.0163 | 0.031 | 15 |
| **the op**, pass 1 | **5.7758** | 0.032 | 15 |
| **the op**, pass 2 | **5.7795** | 0.013 | 15 |
| the token, A/A control | 1.0011 | 0.031 | 15 |
| the token, pass 1 | 0.9904 | 0.055 | 15 |
| the token, pass 2 | 1.0066 | 0.050 | 15 |

    92.8-93.6 ns per table (kernel) against 536-542 ns (the float64 Go loop)
    NRot 64, so 2.9 ns a pair against 16.8

**The two op passes agree to 0.04 points against a 1.6% A/A bias. The two token
passes STRADDLE one and disagree by 1.6 points, which is sixteen times the
harness's own bias** -- so the token is UNRESOLVED and there is no perf claim
here. The arithmetic says why: 445 ns saved on a 465 ms round of 40 positions is
0.004% of it. RULE 6 exactly -- built because it is cheap and correct, quoted
only as the op.

★ **AND THE SELECTION COUNTER EARNED ITS PLACE ON THE FIRST RUN.** The dose was
first written the way `TestQuantActDose` is, a State per arm built with
`nn.WithRopeGo`, and its census read *"arm A: 0 kernel tables, 40 Go tables"*
for the arm that was meant to run the kernel: `nn.NewJIT` assigns that dose to a
PACKAGE variable, so the second State's option silently rewrote the first's.
Both arms ran the same path, and without the counter the ratio would have gated
at 1.00 with nothing measured (RULE 10).

### What it cost the llama.cpp coverage ratchet

`TestGreedyMatchesLlamaCpp` compares a greedy chain against llama.cpp and
records how many tokens each model walked before the first near-tie. A one-ulp
table moves where those coin tosses fall, and it moved four of thirteen rows --
**two down and two up**:

    Qwen2-1.5B-Instruct   19 -> 1     diverged at token 1 on a 0.085 tie
    Qwen3-1.7B            20 -> 5     diverged at token 5 on a 0.035 tie
    Qwen2-VL-2B            5 -> 20
    SmolVLM-256M          12 -> 20

Every divergence is inside the 0.5 margin the gate itself calls *"not a bug"*,
and the A/A control on the teacher-forced comparison reads **exactly 0.000e+00**
on both gemma models, so the harness is discriminating and the movement is real.
**The two lowered entries are NOT re-baselined here**: blessing a ratchet is the
move RULE 11b refuses, and the decision to re-record them belongs to whoever
reads this measurement.

★ **AND THE MODEL-LEVEL AMPLIFICATION IS WORTH RECORDING ON ITS OWN.**
Teacher-forced over 11 positions, the one-ulp table moves logits by
**3.5e-01 to 3.6e+00**, the largest on gemma-3-1b -- against an A/A control of
0.000e+00. That is the same chaotic sensitivity this project already records
between the CPU and the device (the 0.2-0.94 f32 reduction band), reached here
from a perturbation of one ulp in 128 table entries.

### RULE 11d, twice, and the M4 caught both

- **A shape below the vector width is a SHIPPED shape.** The first kernel
  finished a ragged pair count by re-running the body over the LAST WHOLE VECTOR
  and writing the same bytes over a few pairs -- idempotent, cheap, and needing
  a whole vector to stand on, so it refused fewer than eight pairs on the
  reasoning that "the smallest NRot in this tree is 64". stories260K's head
  dimension is EIGHT, which is four pairs, and `TestGreedyMatchesLlamaCpp`
  reached that refusal on its first subtest. The tail stores PART of a whole
  vector now, which serves any count and needs no second shape.
- **Two of the violations did nothing on NEON and fired on amd64.** They are
  injected by zeroing a word of the shared constant block, which reaches every
  tier only if every tier READS that word -- and the NEON kernel built its
  quadrant integers with `MOVI` immediates. Running `./engine/nn` on the M4 found it;
  the kernel reads them from the block now, and the sign step carries TWO words
  because amd64 XORs a sign mask where NEON selects against an `FNEG`.

## ★ MXFP4: A CODE TABLE IN THE PACKED KERNELS, NOT A NEW EMITTER.

gpt-oss's expert banks are MXFP4: 32 e2m1 codes and one E8M0 exponent per block.
The code is a float, so no offset makes the weight affine in it -- the one
property every other packed format has. The payload stays the raw 4-bit code
and every reader translates it through a sixteen-entry table (`kernels.Codes`):
one VPSHUFB per 32 weights on amd64, one TBL on arm64, in the plain, fused and
token-tiled kernels. The table holds the doubled e2m1 values PLUS 12, so every
entry is an unsigned byte VPDPBUSD takes as it is, and the correction is the
constant-bias term every 4-bit format already carries (BiasC derives it).

**What the prior art does, read in source (a survey of llama.cpp, MLX, gpt-oss,
triton_kernels, vLLM):** llama.cpp's AVX2 dot uses a SIGNED table and pays two
VPSIGNB per 32 weights to feed maddubs; its NEON path uses TBL + SDOT with no
bias, which is this shape. The +12 bias removes the sign fix-up on amd64. The
pre-VNNI VPMADDUBSW bound is 24*127*2 = 6096, far from saturating
(`PreVNNIPacked` derives it from the table).

The fused kernel needs one more register for the table, so MXFP4 runs at a
fused group of 8 like Q5_0 and Q5_K (`fusedRegisters` counts it).

**The scale is an f16 in the narrow d plane**, 2^(e-128) exactly, and the packer
refuses a block whose exponent an f16 cannot hold (2^-24..2^15) unless all its
codes are zeros. gpt-oss-20b spans 2^-14..2^7. That costs +5.9% over the GGUF
(18 bytes per block against 17). **A u8 E8M0 plane is the lever**: two
instructions per eight scales (VPMOVZXBD, VPSLLD 23) against one VCVTPH2PS, and
~5.6% fewer bytes on a decode that is at the memory wall. It changes the d-plane
layout for one format, so all five readers move; unbuilt, and next in line for
the MXFP4 rate.

## ★ THE SSE TIER: A GOLDMONT ATOM RUNS EVERY MODEL, NATIVELY.

**What it closes.** `jitllm run` SIGILLed on an Atom C3558
(Goldmont, one of two identical boxes): SSE2/SSE3/SSSE3/SSE4.1/SSE4.2 and
POPCNT, no AVX of any width, no FMA, no F16C.

    jitllm run -devices cpu -n 8 stories260K.jlm
    SIGILL: illegal instruction   PC=0x77eb3e25600f  sigcode=2
    instruction bytes: 0xc5 0xfd 0xef 0xc0 ...     VEX.256 VPXOR ymm0,ymm0,ymm0

That was EmitRMSNorm's accumulator prologue. The probe already knew
(`jitllm hardware` printed "VPDPBUSD (probed: ABSENT)" and every quantized
kernel "none"), and nothing asked it: `PackedSupported` checked no feature,
`SupportedNative` answered true for F32/F16 with the comment "AVX2 only",
`Features.UsableAVX2()` had no caller outside tests, and
`TestEveryElementwiseOpIsGenerated` mapped each kernel without calling it --
green on the host where the same kernels faulted. The shipped `jit/cpu` test
binary passed five arm64 encoding tests there and died in
`TestABKQuantAccs/Q4_K` on VBROADCASTI128, a second emitter family.

**What was built**, as a foundation (921e520) and five families integrated on
main (df004d0..81d6c06, plus the integration commit): a legacy (non-VEX) SSE
encoder pinned against objdump; an in-band ISA declaration (a 7-byte NOP) that
`cpu.Map` checks per kernel, so an SSE host maps SSE kernels and refuses AVX2
ones by name; `cpu.EmittersFor(TierSSE)`, the table every nn/JIT construction
site now reaches through `cpu.HostTier()`; and a kernel for every field --
elementwise and softmax, norms/rope/embedding row/widen, all seven attention
kernels on both KV widths, the float and packed matvec for all eight
`quant.PackedTypes`, and the hybrid's delta rule/conv/gate. `cpu.SSEPending()`
asks the table at engine shapes; `cpu.Baseline` passes an SSE host exactly when
it is empty, and names the missing extensions below SSE4.1+SSSE3.
`model.Open` refuses through `nn.HostRefusal`. The token tile and the wide
packed kernel are refused by decision (the fused kernel serves prefill per
token), and the GGUF row-major quantized family does not exist on this tier.

    SSE table   107 kernels pass the objdump VEX-leak gate, 16 refused by decision, 0 pending

### Forced on the AVX2 laptop (`cpu.ForceTierForTest(cpu.TierSSE)`, tag jitllmtest)

`jit/cpu` and `nn` green with and without the tag. The model gates, each with the
tier forced before any model opens and `MappedByTier()[TierAVX2]` asserted not
to move:

    TestEveryModelRunsSSE          17 language models <= 2 GiB, decode + prefill +
                                   batch, against the AVX2 tier in-process
    TestGreedyMatchesLlamaCppSSE   13 models against live llama.cpp: all pass
    TestSSEModelGates              Hybrid x4, GreedyMatchesLlamaCpp (goldens),
                                   Batch/Ragged/Biased batch, MoEFFNRunsBatched,
                                   TowerEncodes, TowerMatchesLlamaCpp,
                                   SafetensorsMatchTransformers (12 fixtures),
                                   GPTOSSMatchesLlamaCppIntermediates,
                                   SynthGPTOSSMatchesTransformers,
                                   Prefill{,Seq,Mixed}Matches*, EveryModelRunsGenerated,
                                   KVF16IsSelectedAndRuns, PagedKVIsTokenIdentical
                                   (30 containers; two phi3 fixtures skip, one
                                   page), LlavaTowerEncodes: all pass

★ **THE FIRST SSE-vs-AVX2 COMPARISON COMPARED TWO KV CACHES, NOT TWO TIERS.**
It read max|dlogit| 1.0-13.6 against a 0.94 band on eleven models. Traced per
layer on gemma-3-1b at position 0, the matvecs agreed to 3.5e-8 relative and the
attention output -- which at one key is a copy of V[0] -- differed by 3.3e-4:
f16's own rounding. `KVWidthPaysOff` takes f16 on the AVX2 tier (F16C makes the
kernel no bigger) and declines it on SSE (the software half conversion makes the
f16 pair 3.8-4.0x the code), so the unpinned arms ran different caches. With
the width pinned:

| model (ids 1..12) | SSE vs AVX2 | AVX2 f16 vs f32 (control) |
|---|---|---|
| stories15M-q8_0, synth-gptoss, stories260K | 0.0000 | 0.12-0.51 |
| Llama-3.2-1B Q4_K_M | 0.859 | 1.224 |
| Qwen3-1.7B | 1.561 | 2.203 |
| Qwen3-MoE-4x0.6B | 1.093 | 1.077 |
| gemma-2b | 1.250 | 1.394 |
| gemma-3-1b | 4.153 | 12.811 |

So the gate pins the KV width and holds the SSE arm to 1.25x a same-process
control (the AVX2 tier's own f16-vs-f32 move) with 0.94 as a floor; measured
SSE/control ratios 0.32-1.07. **Violation:** the NEOX rope's sines gathered
with SHUFPS 0x88 instead of 0xDD turned exactly the NEOX models red (three
gemmas, four Qwens, synth-gptoss: 3.6-32 against bands of 0.94-16) and left the
NORM-rope models (Llama, tinyllama, stories, SmolLM) green.

★ **AND ONE llama.cpp GOLDEN SITS ON THE KV WIDTH, NOT THE TIER.** qwen2vl,
"One day, a little": AVX2 f16 "boy" (margin 0.0048), AVX2 f32 "girl", SSE f16
"boy", SSE f32 "girl". llama.cpp's goldens use its default f16 cache and the
internal gate fails any tie at token 0, so the SSE arm of that gate runs the
reference's cache.

### On the real Atom, no forcing (GOAMD64=v1, CGO_ENABLED=0 binaries)

`jitllm hardware` on .11 and .12:

    tier  sse -- legacy-encoded 128-bit kernels, no FMA, f16 converted in software
          (host: sse2, sse3, ssse3, sse4.1, sse4.2, popcnt; no usable avx2, fma, f16c);
          every op is generated for it
    packed Q4_0 sse ... MXFP4 sse   decode coreset p, 4 core(s) 0,1,2,3

- **Whole suites, unfiltered:** `jit/cpu` PASS (was 5 FAIL: the float and packed
  matvec gates now take `cpu.Native()`'s kernel, which is the SSE one there;
  the GGUF activation packer skips by name). `nn` PASS with and without
  jitllmtest, after two fixes: `TestPreVNNIMatchesVNNIBitForBit` and its
  violation twin need a VNNI arm (a host without one ran the forced arm twice,
  failed "did not restore the probe", and leaked the armed naive-Q8_0 hook into
  `TestSSETierRunsTheMatvecGates`, which then read NMSE 2.95e-2).
- **Model gates** (containers only on the box, so the batch gates now accept a
  container through `existingModel`): TestGreedyMatchesLlamaCpp (stories260K,
  stories15M, tinyllama against llama.cpp's goldens), TestEveryModelRunsSSE (7
  containers), Batch/Ragged/PrefillSeq, PrefillMatchesForward (no-gemm is bit
  equality), PrefillMixed (7), EveryModelRunsGenerated, Hybrid x4,
  SynthGPTOSSMatchesTransformers, TowerEncodes, TowerMatchesLlamaCpp,
  KVF16IsSelectedAndRuns, PagedKVIsTokenIdentical (7): all pass, no SIGILL.
- **Token ids, `run -devices cpu -n 16 "The capital of France is"`, Atom vs
  laptop:** Llama-3.2-1B identical in all four (tier x KV width) arms; tinyllama
  identical at both widths; stories15M-q8_0 identical at f32 (the Atom's
  default) and parts at generated token 6 at f16; SmolVLM-256M agrees for 12 of
  16 then ',' against '.'. Containers were converted ON the Atom from
  byte-identical GGUFs (sha256 checked against the laptop's) where the WiFi link
  (~0.5 MB/s) made copying a container impractical.

**Rate, on .12, cores 15-16% busy with other sessions' jobs** (.11 was 86% busy
and read 0.92 tok/s -- the straggler cliff, not a measurement). n=128, three
runs each, 4 cores:

    Llama-3.2-1B Q4_K_M   decode 2.29-2.31 tok/s   prefill (6 tok) 2.99-3.05 tok/s
                          816 MB/token -> 1.88 GB/s = 17% of the 11.1 GB/s 4-thread wall
    tinyllama Q3_K_M      decode 3.26-3.29 tok/s   (TheBloke's file, same size/format)
    stories15M-q8_0       111 tok/s (jitllm bench), 1.82 GB/s = 16% of wall
    cores 1 / 2 / 4       Llama 1.09 / 2.00 / 2.33 tok/s (n=32)
    wall (Go-side scalar) 5.3 / 8.1 / 11.1 GB/s at 1 / 2 / 4 threads

At 17% of the wall the tier is compute-bound -- these are basic kernels, as
intended, and no claim is made against llama.cpp (not on the box). Two leads,
measured and unattributed:

- The fused kernel in cache prices at 0.23 ns/element (Q4_K) and 0.40 (Q6_K)
  per core (`TestSSEPackedCallTime`), which predicts ~2.8 tok/s on ONE core;
  the token runs at 1.09. The in-situ gap is 2.6x.
- **Every Q6_K super-block scale in Llama-3.2-1B is an f16 subnormal**
  (1,582,866 of 1,583,104; Q4_K d 1.9%, dmin 0%, stories15M's Q8_0 0%), and
  `halfToFloatSSE` multiplies a denormal f32 for each one -- a microcode assist
  on Goldmont. With subnormal scales the call-time gate reads Q6_K 103 against
  88 ns/block and Q8_0 30.4 against 7.1. The assist-free conversion family 4
  priced (integer rebias for normals, CVTDQ2PS * 2^-24 for subnormals) is owed,
  and Q6_K -- the head of every Q4_K_M model -- is where it lands.

`nn.packedChunk` sizes SSE calls with `cpu.RowsPerCallFor` (a per-element
budget measured on the Atom) and is unchanged off the SSE tier.

## The reference tier, before it was removed (moved from AGENTS.md RULE 8)

The rule that stood in AGENTS.md read "`nn` IS THE ORACLE AND THE BISECTION
TOOL. IT IS NOT A PRODUCTION FALLBACK" and required `nn.JIT[t] == nil` to stay
reachable. It is superseded by the current RULE 8: the reference tier is gone,
its arithmetic moved to `internal/oracle` (test-only), and a missing kernel is
a kernel to generate. Every kernel whose width, head dimension or k used to
decline -- elementwise ops, norms, softmax, attention, the delta rule and its
convolution, the row-major float matvec, a packed row remainder, the rotary
application, softcap, the embedding lookup, the sampler's softmax -- takes the
remainder in generated code now (commits cfe30ac and e691f18).

The text below is the rule as it stood, kept because every episode in it is a
case of a correct answer hiding a missing kernel, which is the failure the new
rule makes structural.

### ★ RULE 8 (superseded): `nn` IS THE ORACLE AND THE BISECTION TOOL. IT IS NOT A PRODUCTION FALLBACK.

★★ **AND arm64 WAS IN BREACH OF IT FOR EVERY QUANTIZED MATVEC, FROM THE DAY THE
CONTAINER LANDED UNTIL THE arm64 PACKED KERNELS WERE WRITTEN. THE COST WAS 24.5x.** `packed.go`,
`packedwide.go` and `packedfused.go` are all `//go:build amd64` and
`packed_other.go` stubbed every one, so `nn.MatVecPacked` fell to `packedRef` --
a Go loop -- for Q4_0, Q8_0, Q3_K, Q4_K, Q5_K and Q6_K. Measured on the M4,
Qwen3-MOE-4x0.6B-Q4_K_M, same binary with the kernels switched off and on:

    Go reference   2.53, 2.53 tok/s
    generated      62.02, 61.94 tok/s        **24.5x**

★ **AND NO GATE COULD SEE IT, WHICH IS THE PART THAT MUST NOT RECUR.**
`MatVecPacked` falls back and **RETURNS TRUE**. The answer is correct, so no
oracle can see it, and `model.TestNothingFallsBackToNN` counts the fallbacks the
MODEL takes rather than the ones `nn` takes inside itself -- RULE 10's "a counter
that counts fallbacks certifies nothing", one package down, where the counter
was in the wrong place entirely. `nn.TestEveryPackedFormatHasAGeneratedKernel`
asks the CAPABILITY instead of counting, and it is green on both architectures
now. `TestPackedMatVecTailAgrees` -- the only gate comparing a generated packed
matvec against `kernels.MatVecPacked` -- named Q4_K, Q6_K and Q3_K and **not
Q8_0**, which is what an entire vision tower is made of; it covers all six now,
counts what ran, and fails at zero rather than skipping.

`EmitA64PackedMatVec` covers all six. The layout suits NEON better than AVX2:
four rows' word is ONE 128-bit load, SDOT's four int32 lanes each accumulate one
row with no horizontal reduction, and SDOTelem feeds four payload words from one
activation load. **SDOT is signed by signed where VPDPBUSD is unsigned by
signed, so a signed payload needs no XOR centring and no correction term at
all** -- that asymmetry is the one real porting trap, and it cost an afternoon
as an NMSE of 2.3e-02 with everything else already right.

★★ **AND arm64 IS FULLY JITTED ON jlm NOW.**
`EmitA64PackedMatMulTiled` is the token-tiled twin, all six formats, four tokens
everywhere (NEON's 32 registers, against amd64's 16 which narrow Q6_K to two).

    M4, tinyllama q3_K_M, 512-token prefill
      per-token dispatch   162,274 regions    95.5, 95.2 tok/s
      token-tiled            7,042 regions   176.2, 175.8 tok/s    **1.84x**

    M4, Qwen3-MOE-4x0.6B decode (the matvec, measured earlier)
      Go reference   2.53 tok/s   ->   generated 62.0    **24.5x**

★ **THE GATE THAT MAKES IT STAY TRUE COUNTS THE CALLEE, NOT THE CALLER.**
`model.TestNothingReachesTheReferenceMatvec` asserts `nn`'s OWN reference
counter is zero over Forward AND Prefill for every model under 2 GiB, on both
architectures. Against a violation -- Q4_K and Q3_K dropped from the arm64 list
-- it reads **714 decode and 3808 prefill** reference calls. The old counters
could not: they live on `model.State` and count what the MODEL declined, while
`nn.MatVecPacked` declined internally and returned true.

★ **FOUR BUGS, AND THE TWO THAT MATTER ARE ABOUT GATES SEEING DIFFERENT THINGS.**
The d plane is indexed [super-block][row] and its advance was dropped when the
per-super-block precompute was folded into the epilogue, so every super-block
read super-block ZERO's scales -- **NMSE 1.0, an uncorrelated output**, and
`nn.TestPackedMatVecTailAgrees` was GREEN while `cpu.TestPackedKernelMatchesReference`
caught it. Two gates over one kernel at different levels, and only the lower one
saw it. And a baked token stride does not fit: a scaled `ldr q` immediate tops
out at 65520 and the output stride is 4*nrows, 65536 on gemma-2b's FFN -- the
strides live in registers now.

★ **AND `MOVimm` REFUSED TWO-FIELD CONSTANTS, WHICH WAS A PANIC IN SHIPPING
CODE.** Its comment said such a value "would need MOVK as well and is refused
rather than silently truncated" -- right against truncation, wrong as the end of
the story. `conv1d/4x8192` needed 0x18000 and panicked the arm64 block-op gate
before any of this work. MOVZ then MOVK per non-zero field now. `MOVK`,
`USHR4s`, `MOVI4s` and `SUB4s` are new and their encodings are **pinned against
clang's own assembler on the M4** -- a wrong shift immediate is a valid
instruction that shifts the wrong amount on the wrong lane width.


**Generate everything that can be generated.** `nn`'s Go implementations exist for
two reasons only: as the float64 ORACLE every kernel is gated against, and for
the parts that genuinely cannot be JIT-compiled. Reaching for `nn.Something` in
the shipping decode or prefill path because a kernel does not exist yet is a
**task**, not a design.

`nn.JIT[t] == nil` must stay REACHABLE — any change that makes the engine unable
to produce correct tokens without generated code is wrong, because that is what
makes bisection and the oracle possible. That is a statement about CORRECTNESS,
not a licence to ship Go loops on the hot path.

**★ BUT IT IS NOT A RUNTIME CONFIGURATION, AND THE SENTENCE HERE USED TO SAY
"SUPPORTED", WHICH WAS WRONG.** Stated by the user: on amd64 and
arm64 **there is no way to disable the JIT**. This engine generates its own
kernels -- that is what the name says -- and an environment variable that puts a
shipped binary on the float64 reference tier at 0.276 tok/s is a trap rather
than a knob: it is a CLIFF, not a degradation, and it is indistinguishable from
a slow machine. `JITLLM_DISABLE_JIT` and `JITLLM_DISABLE_FAST` are deleted.

`nn.jitDisabled` is written by exactly one function, `nn.DisableForTest`, whose
body exists only under the **`jitllmtest`** build tag (`nn/disable.go`);
`nn.Disableable` reports which build this is so a test needing both tiers SKIPS
rather than comparing the fast tier with itself. `tier/fault.go` is the
precedent and carries the same sentence -- *it is not in a release build, which
is why it is a tag and not a flag*. On every other architecture `engine/nn/jit.go` is
not compiled at all and `jit_stub.go`'s nil JIT is the only tier there is, which
is the configuration the oracle needs.

    go test -tags jitllmtest ./engine/model -run TestPlacementInvariance

**★ CLOSED, AND IT WAS WORTH 2.6-4.5% OF A TOKEN.** The audit below
is kept because it is what the breach looked like; it no longer describes the
engine. Every elementwise op is generated now, on amd64 AND arm64, and the two
the device tier was also missing (LayerNorm and the ungated activation) are on
all three backends:

    RMSNorm   softmax   SiLU/GELU gated   SiLU/GELU UNGATED   LayerNorm(+bias)
    dst += alpha*src  -- the residual, every bias, and the MoE accumulate
    dst *= alpha      -- gemma's embedding scale and EVERY attention score row

**★ AND "ARE WE FALLING BACK AT ALL?" IS A SWEEP, NOT A SPOT CHECK.**
`model.TestNothingFallsBackToNN` runs Forward, Prefill AND ForwardBatch over
EVERY model on the box and asserts all three counters are zero;
`JITLLM_AUDIT_MODEL=<substring>` audits one file whatever its size, because a
2 GiB cap skips exactly the models where a missing kernel hides. It found one:
**Mixtral's router is F16** where every other mixture's is F32, so all 32 of them
per token took the float64 oracle -- 9.5% of the token by the profiler, and
**1.41x of the whole token** once the kernel existed (ABBA, 3 rounds, spread
0.027, token ids unchanged). RULE 14r for the type it did not cover.

The remaining `nn` on the inference path is now exactly RULE 8's list: the
float64 MatVec ORACLE (reached by nothing), the Dot32/AxpyF32 twins behind the
attention kernels (counted, zero), and `nn.Row` -- left un-generated because it
measured 0.0257% of a token, with a test that fails above 0.01%.

**"EVERY" IS A LIST IN THE REPO, NOT A SENTENCE HERE.** Five green lines answer
it, and each fails against a violation:

    jit/cpu        TestEveryElementwiseOpIsGenerated       13 ops, emitted and
                   mapped executable ON THE HOST ARCH -- so amd64 and the M4 each
                   answer for themselves, and the darwin cross-build catches a
                   name that exists on one and not the other
    jit/cpu        TestEveryBlockOpIsGenerated             the SHAPED kernels,
                   which are a separate list because their device twins differ in
                   kind: the gated delta rule and the causal convolution both
                   read a buffer they write, which RULE 13 forbids on a device,
                   so lowertest cannot mirror this one
    lowertest      TestEveryElementwiseOpLowersEverywhere  14 device kernels,
                   registered and therefore assembled by ptxas, spirv-val and
                   Metal's compiler
    model          TestNormsRunGenerated                   zero interpreted ops
                   across Forward, Prefill AND ForwardBatch
    model          TestNothingFallsBackToNN                the same three
                   counters over every model UNDER 2 GiB; bigger ones need
                   JITLLM_AUDIT_MODEL, one at a time
    model          TestTowerRunsGenerated                  the VISION TOWER,
                   which the sweep above cannot reach

★ AND "EVERY MODEL ON THE BOX" IS WHAT THAT SWEEP CLAIMED AND NOT WHAT IT RAN.
The default caps at 2 GiB, so it answers for 11 models on Linux and 6 on the M4
and says nothing about the seven larger ones -- which are exactly where a missing
kernel hides, and where Mixtral's F16 router did hide. Audited one at a time,
`JITLLM_AUDIT_MODEL=<substring>` under `scripts/cap 24G`:

    amd64   llama-2-7b, Phi-3.5-mini, olmoe x2, Qwen3-30B-A3B,
            Mixtral-8x7B-Q3_K_M, Llama-3.3-70B          ALL CLEAN
    arm64   olmoe x2                                    ALL CLEAN
    (Qwen3-Next-80B is UNAUDITED, and for a new reason -- see the open question
     on Open loading every page: it cannot be opened on this box yet, which is a
     task and not a refusal)

**So 18 of the 19 GGUFs on this box reach `nn` exactly zero times**, on both
architectures, across Forward, Prefill and ForwardBatch.

★ **AND THE HYBRID'S OWN OPS ARE GENERATED TOO, WHICH IS A THING THAT ALMOST DID
NOT HAPPEN.** The gated delta net shipped first as three Go loops with the
admission buried in a source comment -- `nn.Dot32` twice per state row (37.7M
multiply-accumulates a token), a four-tap convolution loop, and scalar
softplus/exp on the gates. The user caught it by asking why the JIT had not been
touched. `EmitGatedDelta`, `EmitConv1d` and `EmitDeltaGate` replace all three on
amd64 AND arm64, and `model.TestHybridRuns` asserts all three fallback counters
are zero AND that the delta kernel was SELECTED -- because the oracle beside it
computes the same answer, so agreement says nothing about which ran.

★ **softplus IS THE ONLY TRANSCENDENTAL IN THE ENGINE THAT IS NOT exp, AND THE
IDENTITY MATTERS.** `log(1+exp(z))` overflows above z ~ 88 and every scalar
implementation guards it with a branch. A vector kernel cannot branch, so it
computes `max(z,0) + log1p(exp(-|z|))`: the exp argument is never positive, its
result is always in (0,1], and the logarithm's argument is therefore always in
(1,2] -- which is also what bounds the atanh series to five terms.

★ THE TOWER WAS NOT COVERED BY ANY OF IT, AND ITS COUNTERS HAD NEVER BEEN
ASSERTED. `TestNothingFallsBackToNN` globs models and drives Forward/Prefill/
ForwardBatch; an mmproj opens through `OpenTower` and runs through `Encode` -- a
different loader, state type and graph -- so `TowerState.ElemFallbacks()` and
`.MatVecFallbacks()` existed and nothing read them. The tower is where a fallback
is MOST likely, not least: biases on q/k/v/o and both MLP matrices, LayerNorm
rather than RMSNorm, an UNGATED activation, and projector shapes derived from the
image size. It runs clean (SmolVLM-256M, both hosts); against a violation the
gate reads `12 elementwise` -- one per tower layer.

★ AND ITS mmproj PATH WAS A LINUX-ONLY CONSTANT, so every tower gate printed
`--- SKIP (0.00s)` on the M4 under a green PASS. It resolves `../models` first
now, and the file is on both boxes (RULE 11).

★★ **AND THE SAME SHAPE SURVIVED ONE LEVEL UP:
`TestTowerOnDeviceMatchesHost` NAMED cuda AND vulkan AND NOT metal**, while its
own comment claimed *"the M4 runs this same gate"*. It cannot: on an M4 both of
those fail to open, `ran` stays zero and the whole thing SKIPS under a green
PASS -- on the one host where Metal IS the GPU, so the CPU-to-Metal seam for a
vision block had never been exercised end to end. With `metal` in the list it
runs and passes at **NMSE 2.802e-07** after one block; Linux logs it as not
present and is unaffected. A device list written on one host is a claim about
the others.

### ★ AND A COUNTER THAT COUNTS FALLBACKS IS BLIND TO AN OP THAT NEVER ASKS

**Found with every gate above green.** `ElemFallbacks()` counts ops
that HAVE a generated path and declined it. A raw `for i := range x { x[i] *= a }`
never asks, so it is invisible to all five -- RULE 10's "a counter that counts
fallbacks certifies nothing", arriving from the other side. **Seven of them
shipped inside a graph the audit called clean:**

    forward.go  prefill.go  batch.go     x[i] *= EmbdScale   (gemma's embedding)
    forward.go  prefill.go  batch.go     af[t] *= scale      (attention scores)
    vision.go                            att[j] *= scale     (the ViT's scores)

The score ones are one pass over the score row per (layer, head, query): a
1024-patch SmolVLM tower runs 12 x 12 x 1024 of them, ~151M multiplies an image.
`cpu.EmitScale` / `EmitA64Scale` / `kernels.Scale` is the eleventh elementwise op
and covers all seven, on amd64, arm64 and all three device backends.

**★ AND THE SCORE ROW IS pos+1 LONG, SO MOST OF THEM ARE RAGGED.** Wired
naively -- decline on a width that is not a whole vector -- the audit went from
"11 models, 0 reaching nn" to **"11 models, 11 reaching nn"**, because every
attention row went back to Go. The kernel takes the aligned prefix and Go takes
a tail of at most `ElemLanes-1`, which is a remainder rather than a pass.

★ FOUR MORE LOOPS WERE COPYING float32 INTO float32. `narrow()`, prefill's and
batch's per-head q staging, and the tower's `q32/k32/v32` all ran
`dst[i] = float32(src[i])` on operands that were ALREADY f32 -- seams left when
RULE 8b moved the host residual off `[]float64` and the conversions narrowed to
nothing under them. The tower's was the worst: three buffers of n x NEmbd
holding a second copy of q, k and v, refilled every layer -- 28.3M element
copies and 9.4 MB per image, deleted outright.

`State.ElemFallbacks()` and `NormFallbacks()` count anything that still runs
interpreted -- INCLUDING the Dot32/AxpyF32 twins behind the attention kernels.

**★ AND A FALLBACK IS CORRECT WITHOUT BEING BIT-IDENTICAL, SO WHICH ONE RUNS MUST
NOT DEPEND ON A WIDTH OR A POOL SPLIT.** The Go twin of an activation exps in
float64 where the kernel uses a range-reduced polynomial; both are right, and
they differ in the last bits. `Parallel` handed out ELEMENT ranges, so on a width
that is not a whole number of vectors the ragged piece landed differently
depending on how many rows the caller batched -- decode got 172 elements and
declined, prefill got 512 and 4 and used the kernel for the first. **Forward and
Prefill then computed different FFNs on the same input, and three bit-equality
gates were red for it.** Two things fix it and both are required:

- **the elementwise scratch is PADDED to a whole vector** (`ffnPad`), so a ragged
  tail rides into slack nothing reads and the kernel serves EVERY width;
- **the pool splits by VECTOR, not by element**, so only the last chunk can be
  ragged at all.

A kernel that declines for a width is a kernel whose arithmetic the SCHEDULER
chooses. Emit for every width, or the equality gates are measuring the split. **That assertion is the point, not the kernels**: an interpreted op
is also CORRECT, so no oracle can see one, and RULE 10 is the standing record of
how often a dead fast path has shipped here.

    M4, tinyllama q3_K_M decode, ABBA, n=160, two passes   1.0267 / 1.0258  IQR 0.001
    M4, gemma-2b decode                                    1.0449           IQR 0.003
    M4, SELF-CONTROL (one binary in both arms), run FIRST  0.9986           IQR 0.000

gemma gains more because its activation is GELU -- interpreted, that was a
non-inlinable `nn.GELUTanh` per element (RULE 7b-bis's ~295,000 indirect calls a
token) -- and its FFN is 16384 wide against tinyllama's 5632. **Measured on the
M4 because the Linux box sat at 765-1653 MHz idle all evening**, which is RULE
2d's sustained-power clamp and its own verdict is to stop measuring. Memory is
flat to DOWN: two f32 staging rows (`batt`, `bath`) are deleted outright, and the
score buffers' stride grows only when maxSeq is not a multiple of 16.

**AND THE CPU TIER WAS IN BREACH, WHILE THE DEVICE TIER WAS NOT.** When
audited, `nn.JIT` generated exactly TWO things, `MatVec` and `MatMul`.
Everything else in a decode token was Go —

    nn.RMSNorm x9   nn.Softmax   nn.ActMul   nn.Dot32 and nn.AxpyF32 (the
    attention fallbacks)   and 14 plain loops over vectors: both residual adds,
    the MoE weighted accumulate, and the f32<->f64 conversions around them.

**★ AND GENERATING THE SOFTMAX FOUND A WRONG ANSWER NO ORACLE HAD CAUGHT: THE
PADDING VALUE WAS exp's CLAMP.** A padded score row is only inert while every
real score is above the pad, and qwen2's attention reaches **-872** against a
clamp of -87.3 -- so the PADDING became the row's maximum and a one-element
softmax returned 1.7e-39 where it must return 1. The model answered fluent
repetition. It is `-FLT_MAX` now, and the kernels seed their maximum from the row
rather than from any constant, so the two facts cannot drift apart.

Softmax is SHIFT-INVARIANT, which is why an NMSE gate over lengths 1..1537 ran
straight past it AND past a second bug in the same kernel (a cross-lane `VPERMQ`
where the reduction needed the in-lane swap): a wrong maximum is arithmetically
harmless until the shift crosses the clamp, and then it is wrong by TYING the
saturated entries rather than by drifting. **The distinguishing input is
MAGNITUDE, not length.** `nn.Row` -- the embedding lookup, the one op left
un-generated -- was left that way on a measurement:
`TestEmbeddingRowCost` prices it on every model under 2 GiB (worst row 0.0257% of
a token) and FAILS above 0.01%, so the decision is re-derived on every run rather
than inherited.

RULE 12's device path already does the opposite, in its own words: *"every op
between the matvecs (RMSNorm, quantize, RoPE, the KV copy, scores, softmax, the
weighted sum, the FFN gate, the residual add) is a generated kernel, so the
residual stream never returns to the host inside a block."* **The device tier is
the model to follow; the CPU tier drifted into being a regular inference
engine.**

This is the same finding as the 906 dispatch regions per MoE token from the other
side: every un-generated op is also a separate Go-orchestrated `pool.Do`. Emitting
them and fusing them into the layer is ONE piece of work that collects both.

**The reference tier is an oracle for the KERNELS, not for the ARCHITECTURE.**
Every tier agreeing to the bit proves nothing about a wrong rotary layout or a
mis-routed expert — only an independent implementation can see those.



## ★★★ THE CONTAINER LAYOUT'S PLANE STRIDE WAS THE M4 CPU DECODE GAP, AND ONE PRFM A LINE REMOVES IT

`nn.TestShapeCost` (jitllmbench) streams a real decode shape through ~192 MiB
of distinct copies and knows five knobs (JITLLM_SHAPES, _PART, _CORES,
_PADROWS, _A64_PF). On the M4's four P-cores it put every row count that is
not a multiple of 2048 at 32-35 GB/s where 2048/4096/8192 rows read 58-67 --
Qwen2's 1536 and 8960, and every vocabulary. Not the participant count
(pinned 1..6: flat), not balance: the SAME 1536 rows packed 2048 wide
(`Packed.Stride`) ran 34.7 us a call against 51.6 at their own stride.

The layout is [sub-block][word][row], so a 64-row tile reads 256 contiguous
bytes of a word and steps a whole plane (nrows*4 bytes) to the next, and the
next tile re-walks the same words 256 bytes on. The hardware kept up only at
plane strides that are multiples of 8 KiB. One `PRFM PLDL1KEEP` per 128-byte
payload line, one tile ahead (ee2c52d), swept at 256/512/1024/2048 bytes with
one tile best everywhere:

    CPU decode, tg128, prefetch on / off (JITLLM_A64_PF=-1), A/A 0.999-1.003
    Qwen2-1.5B  1.427   Llama-3.2-1B 1.196   tinyllama 1.535
    gemma-2b    1.029   SmolLM2-360M 1.030   Qwen3-MoE 1.313

★ **THE SAME HINT IN THE PREFILL TILE IS A REGRESSION** -- 1.5-1.6% slower on
Llama and tinyllama -- because a tile does tok times the arithmetic per byte
and hides the stall already. It was tried in EmitA64PackedMatMulTiled and taken out.

★ **AND IT EXPLAINS THE "BIMODAL" QWEN2 CPU DECODE** that read 36-50 tok/s run
to run with nothing changed: without the hint the prefetcher's luck decides
the rate (the prefetch-off arm's IQR/median was 0.30); with it the arm is
tight at 62.4.

Before it, the same session: `EmitA64PackedMatVec` gathers a row group's four
d words with one LDRq and UZP1/UZP2 instead of four scalar loads and inserts,
and hoists the epilogue's per-row-group invariants (1df75e3, 6025fee for the
prefill tile): CPU decode Llama 0.717 -> 0.81, tinyllama 0.63 -> 0.734.

## On the M4, prefill wants the E-cores and decode does not

`JITLLM_CORESET=pe` against `p`, M4, pp512: +26% SmolLM2, +26% tinyllama,
+25% Llama, +27% Qwen2, +45% gemma-2b -- and -17% on Qwen3-MoE, whose expert
GEMMs are a few routed tokens each. Decode goes the other way on most models
(tinyllama -17%, Llama -13%). So `nn.Config.PrefillCores` builds a second pool
that prefill drains through (BeginPrefill/EndPrefill, 5b04d82), "pe" by
default on darwin/arm64 only, and a mixture stays on the decode pool. The
Linux laptop measured the same split neutral in both phases (AGENTS.md's
"Prefill never gets its E-cores" entry), which is why the default is this
host's alone. On the M4, pool on / off in one pass: prefill +17% SmolLM2,
+20% tinyllama, +14% Llama, +22% Qwen2, +54% gemma-2b, decode unchanged.

★ **THE PHASE HAS TO CARRY ITS GOMAXPROCS.** At cmd/jitllm's decode-aligned
GOMAXPROCS (4) the ten prefill workers read SmolLM2's pp512 at 87 tok/s
against 751 -- 8.6x slower; at a prefill-aligned 10, decode lost 6.5%. The
width follows the phase now (BeginPrefill raises it, EndPrefill restores it).
This contradicts AGENTS.md's open question "The M4's E-cores are worth
nothing to prefill", which measured the MatMul kernel phase alone under
JITLLM_CORES on the pre-tile engine; that bullet is owed an edit.

## The Xeon's dense decode gap is NOT the row-major layout, measured end to end

Xeon E5-2680 v4, node 0, cores 0-11, `numactl --membind=0`, tg128, llama.cpp
b11222 `-t 12` in the same pass. Llama-3.1-8B Q4_K_M: jitllm 11.36-11.50 tok/s
at 4.40 GiB/token (~53.7 GB/s, 97.1% of the token in matvecs, JITLLM_PROFILE)
against llama.cpp 12.38-12.43 (~57-58 GB/s). Three levers were priced and
none closes the ~8%:

    lever                                          8B        1B
    row-group shadow (each 16-row group's planes   11.48 vs  61.2 vs 61.2
      contiguous, llama.cpp's q4_Kx8 idea, one      11.48     (null)
      fused call per group; CPU-only copy)          (null)
    k-slices for every short matrix (floor         +1.0%     -0.5%
      fusedSliceElems 32M -> 0)
    native 12-byte Q4_K scale plane                ceiling +2.8% (148 -> 144 B
                                                   per super-block), not built

★ **THE STANDALONE PROBE SAID OTHERWISE AND WAS WRONG.** One matvec in a loop
on a fresh JIT read the row-group layout at 1.48x (4096x4096) and 2.3x
(2048x2048), and k-slices at up to 1.5x -- while its own default arm read
31-45 GB/s, below the 53.7 the whole MODEL reaches. Inside a model the duel
tuners have settled the prefetch and the chunking, and the effect is gone.
A review named the same shadow test as the decisive one before a
container change; it came back null, so the q4_Kx8-style interleave was tried
in that shadow test, measured nothing, and the container keeps its layout.

★ **WHERE THE GAP SITS INSTEAD: THE SECOND SIX CORES.** At six cores (0-5,
`-t 6`) the two engines are level on 8B -- jitllm 7.07/7.20 against llama.cpp
7.15/7.46 tok/s -- and at twelve jitllm is 0.92. So it is how the pool scales
past half a socket (coordination, stragglers, or bandwidth contention between
workers), not the kernel's bytes or their order. That is the next place to
look.

★ **AND IT IS NOT THE TAIL OF A REGION EITHER: GUIDED CLAIMING IS A 5.5%
REGRESSION.** The idle workers (`Pool.await`, 4.9% of samples at six cores, 8.7%
at twelve) read as stragglers behind the last of two fixed chunks per worker, so
the fused matvec was given guided claims -- full chunks while plenty is left,
shrinking to a quarter chunk as the region runs out. Same node, same harness,
ABBA, two rounds: **10.83 against 11.45 tok/s** on 8B. Shorter runs at the tail
cost more than the wait they remove; not built.

The gap by core count, 8B, same pass each (jitllm / llama.cpp tg128):

    8 cores   10.53 / 10.94   0.96
    10        11.10 / 12.03   0.92
    12        11.43 / 12.40   0.92
    14        11.58 / 12.67   0.91

At fourteen that is ~54.7 against ~58 GB/s: about three points of BYTES (Q4_K's
16-byte scale plane against GGUF's 12, Q6_K's +1%) and about five of per-byte
efficiency that no lever tried today moves.

## An arm64 fused packed matvec, and why the M4 picks a kernel per shape

`cpu.EmitA64PackedMatVecFused` is `EmitA64PackedMatVec` nested the other way
round: row groups (PackedFusedGroupOf, 16 or 8 rows) outside, every super-block
inside, float sums in registers with Out read once and written once per group.
Same sums in the same order, so it is bit-identical to the tiled kernel
(`nn.TestFusedMatchesTheTiledKernel`, 36 format x shape pairs, and red at
-6.26 against -26.68 when the d-plane cursor stops advancing).

**At equal bytes (nn.TestPerByteCost, ~96 MiB each, four P-cores, M4) it halved
the k-quants until it prefetched**, GB/s tiled -> fused:

| format | k=2048 | +2 groups ahead | k=8192 | +2 groups ahead |
|---|---|---|---|---|
| Q4_0 | 70.1 -> 76.1 | 78.3 | 39.5 -> 71.6 | 72.1 |
| Q8_0 | 72.6 -> 72.5 | 72.2 | 71.9 -> 44.5 | 72.8 |
| Q4_K | 70.5 -> 37.0 | 71.6 | 68.5 -> 38.8 | 72.4 |
| Q6_K | 70.7 -> 36.3 | 72.5 (after the secondary plane got its own hint) | 66.0 -> 33.7 | 71.8 |

L2-resident (2048x2048), fused Q4_K was already the FASTER kernel (86 against
61 GB/s), so the loss was the walk and not the arithmetic. A 32-row group, which
reads whole 128-byte lines, was WORSE (Llama-3.2-1B 29 tok/s): refuted as the
cause. The base arm64 form carries the two-group prefetch.

**At model shapes no kernel wins everywhere** (nn.TestShapeCost, DRAM-cold
copies, GB/s): tiled is best for Q4_K 2048x2048 (60.1) and Q8_0 2048x2048
(67.4); fused over two k-slices for Q4_K 256x1536 (33.0 against 21.4) and Q8_0
960x2560 (60.2 against 45.4); four slices for Q4_0 2048x16384 (68.7 against
55.7). The container is plane-major, so a worker's share is one rows*4-byte run
per payload word; slicing k across workers makes a worker's planes one
contiguous region. So `engine/nn/mvpick.go` times {tiled, fused, fused/2, fused/4} in
place per (format, rows, k, stride) for the first six rounds and keeps the
fastest median -- on arm64 only; amd64 keeps its fixed rule. Tuning off pins the
tiled kernel, because the sliced arms reassociate the sum.

**Model rows, M4 CPU decode, tg128, four P-cores, ABBA, llama.cpp b10985 in the
same pass:**

| model | HEAD | this | llama.cpp | ratio |
|---|---|---|---|---|
| SmolLM2-360M Q8_0 | 138.6 | **157.4** | 174.2 | 0.90 (was 0.80) |
| gemma-2b | 37.6 | **40.5** | 42.5 | 0.95 (was 0.89) |
| Llama-3.2-1B Q4_K_M | 82.6 | 82.0 | 85.4 | 0.96, neutral |
| Qwen2-1.5B Q4_K_M | 62.2 | 62.3 | 68.6 | 0.91, neutral |
| tinyllama q3_K_M | 101.4 | 102.4 | 91.6 | 1.12, neutral |
| Qwen3-MoE-4x0.6B | 105.0 | 103.1 | 113.5 | 0.91, neutral |

The mixture row needed one more decision: with a fused kernel present the
batched expert dispatch switched itself on, and it measured 106.1 -> 80.8. It is
off on arm64 by default (`nn.JIT.MixtureBatches`).

The prefetch-distance duel does not run on arm64: it swaps the fused kernel
under every shape, which is what the per-shape timing would then measure.

## The Xeon dense decode gap is bytes and bandwidth, measured at the memory controller

A review's first suggestion, taken: count what the DRAM actually delivers
(`perf stat -a -I 2000 -e uncore_imc_*/cas_count_read/`, sudo) during a long
decode, per engine. Llama-3.1-8B Q4_K_M, E5-2680 v4 node 0, 12 cores, tg400:

| | tok/s | IMC read GiB/s (steady) | GiB / token |
|---|---|---|---|
| jitllm | 11.34 | ~51.3 | 4.52 |
| llama.cpp b11222 | 12.04 | ~52.5 | 4.36 |

**Both engines are DRAM-bound at the same ~52 GiB/s**, so the 0.942 is two
terms: jitllm moves **3.7% more bytes a token** (the container's Q4_K SC plane
stores 16 bytes a super-block where GGUF packs 12, and Q6_K is +1%) and
sustains **2.3% less** read bandwidth. The per-shape kernel choice (arm64's
mvpick) buys nothing here, pinned arm by arm: fused over whole k 11.63 against
the fixed rule's 11.52 (1B 62.2 / 62.1), two and four k-slices 11.49 / 10.53,
the tiled kernel 3.42. CPUs 0-11 are twelve distinct physical cores on node 0.

The byte term is the lever and it is a container format change: a 12-byte Q4_K
/ Q5_K scale plane (sixteen 6-bit fields in three words; fifteen fit whole and
one straddles) reaches every reader on every tier, and every tier is
bandwidth-bound in decode.

**The 12-byte scale plane's value, bounded from below by a throwaway build**:
the amd64 fused kernel made to reuse plane 2 for Q4_K's sub-blocks 6-7 and
never touch plane 3 (wrong answers, identical instructions, a quarter of the SC
bytes unread) decodes Llama-3.1-8B at **11.68 against 11.46 tok/s (+1.9%)**,
twelve ABBA-paired samples all agreeing, node 0, 12 cores. A lower bound: the
skipped plane leaves a hole a streaming prefetcher can still fetch into.
A review of the layout concluded that a contiguous 96-bit stream of the sixteen
6-bit fields (two fields split across two words, never three) beats a
pair-per-word scheme, costs readers ~25% more SC loads, and Q3_K should move in
the same container version if it happens at all. Ordered, not refused.

★ **AND THE PER-SHAPE PICKER ON amd64 WAS TRIED AND TAKEN BACK OUT, FROM THE
SAME PROBE MISLEADING A SECOND TIME.** `nn.TestLayoutRate` (jitllmbench) now has a
raw-read arm: the same GGUF bytes summed on the same pool. On node 0, 12
workers, it reads the packed Q4_K matvec at 94% of that read on 14336x4096 but
64% on 4096x4096 and **33% on 1024x4096**, and k-slices lift the small shapes
to 52 and 40 GB/s -- so mvpick was enabled for amd64 with 3 and 6 slices added.
In the model it bought **+0.8% on 8B tg128 (11.31 -> 11.40), +0.7% on
Qwen3-8B, -2% on Llama-1B, and -16% on Llama-1B pp16** (the picker times
prefill's L3-hot calls and keeps the answer for DRAM-cold decode). Pinned
slices on the board binary, 8B tg128, ABBA x2: default 11.29-11.36, 2 slices
11.25-11.28, **3 slices 10.65, 4 slices 10.47**. A standalone matvec on a
fresh JIT is not a decode token; the row above ("the standalone probe said
otherwise and was wrong") already said so, and this re-derived it.

## The prefill row chunk is measured per host, and a duel outlives the State that started it

On the M4's pool of four P-cores and six E-cores a batched matmul split into
one chunk per worker leaves the P-cores waiting for the E-cores; splitting it
finer balances them. On the Xeon (twelve identical cores) the same narrowing is
a 20-39% regression (above). So the cap is a duel (`nn.newRowChunkDuel`,
candidates none / 256 / 64 / 16 KiB, every rung tried) after the participant
and token-chunk duels, cached per host, core count and shape set.

It could not settle at first, and the reason was general: **every model.State
builds its own JIT, and only a SETTLED answer reached the cache**, so a duel
that needed more full chunks than one State's prompt restarted from its first
rung on every request -- 160 `jitllm speed` prompts on the M4 without the
token-chunk duel finishing. The prefill duels are now shared process-wide by
their cache key (`sharedDuel`, under `prefillDuels`), and every model settles
all three inside one warm-up run. `nn.TestPrefillDuelsContinueAcrossJITs` fails
at turn 0 against a per-JIT duel.

Settled on the M4 (tokens / KiB): SmolLM2, Llama-3.2-1B, tinyllama 64 / 64;
Qwen2-1.5B 512 / 256; gemma-2b 64 / 16. CPU prefill pp512, warmed cache, HEAD
against this build, llama.cpp b10985 in the same pass:

| model | HEAD | now | llama.cpp | ratio |
|---|---|---|---|---|
| SmolLM2-360M Q8_0 | ~930 | ~1110 | 1330 | 0.70 -> 0.83 |
| Llama-3.2-1B Q4_K_M | 254 | 297 | 346 | 0.73 -> 0.86 |
| tinyllama q3_K_M | 238 | 292 | 494 | 0.48 -> 0.59 |
| Qwen2-1.5B Q4_K_M | 259 | 280 | 259 | 1.00 -> **1.08** |
| gemma-2b | 218 | 218 | 300 | 0.73 |

### arm64 prefill tile: the epilogue gathers d_act and bias once per sub-block

`EmitA64PackedMatMulTiled`'s epilogue re-read every token's activation scale and
bias (two `LDRs`+`DUP` pairs, three `FMUL`s, three pointer steps) once per ROW
GROUP, i.e. four times per sub-block, while only the row scale differs between
groups. It now gathers them once per sub-block into two activation registers
(lane i = token i; those registers are dead until the next sub-block reloads
them) and reads them with by-element multiplies: ~15 instructions per token per
group down to 8. Same products in the same order, so
`TestMatMulPackedMatchesTheRowLoopExactly` stays BIT equality; against lane 0
for every token it fails on every format, and against lane 0 on the bias
alone it fails on exactly Q4_K, Q5_K and Q5_1.

M4 CPU prefill, pp512, `jitllm speed -devices cpu`, 7a205dd against the change,
A/B/B/A, 4 reps each (8 per arm), medians:

| model | 7a205dd | gathered | | llama.cpp b10985 (same box, earlier pass) |
|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | ~290 | ~326 | 1.12x | 346 -> ~0.94 |
| tinyllama q3_K_M | ~292 | ~325 | 1.11x | 494 -> ~0.66 |
| Qwen2-1.5B Q4_K_M | ~283 | ~320 | 1.13x | 259 -> ~1.2 |
| gemma-2b | ~218 | ~250 | 1.14x | 300 -> ~0.83 |
| SmolLM2-360M Q8_0 | ~1070 | ~1110 | noise | |

The llama.cpp column is not from this pass (RULE 1); it places the rows and is
not a ratio to quote. `A64.MLA4s` (int32 multiply-accumulate) was added and
encoding-gated for the integer fold, which is NOT built: a per-super-block int
accumulator per (group, token) doubles the 16 accumulators and does not fit
the register file at tok=4.

### arm64 weight-stationary prefill GEMM, and a duel for its rows

`EmitA64PackedMatMulStationary` is amd64's `EmitPackedMatMulStationary` under
the same Args contract and padded streams: eight rows' sub-block decoded into
up to 16 V registers once, every token of the call dotted against it in a
runtime loop. On NEON the k-quants (Q4_K, Q5_K, Q3_K, Q6_K at a 256-wide
window, `StationaryIntAccA64`) fold the sub-block scale with ONE `MLA.4S` into
an int32 per-token sum in Scratch and touch Out once per super-block --
llama.cpp's `ggml_gemm_q4_K_8x4_q8_K` shape. The other formats keep the tiled
kernel's float epilogue operation for operation.

Gates: `nn.TestPackedGEMMMatchesTheRowLoopExactly` now runs on arm64 (108
pairs; float forms bit-identical, integer forms worst NMSE 1.411e-12). MLA->MUL
fails every integer format at NMSE 4.8; a wrong bias lane fails Q4_0/Q5_0/MXFP4
by exact compare.

The integer fold IS the win: the same GEMM with the float epilogue
(`JITLLM_GEMM_EXACT=1`) reads Llama-3.2-1B pp512 277 tok/s against 500. And the
float forms are rows-per-call sensitive on the M4's 128-byte lines -- gemma-2b
182 at the Xeon-measured 16 rows, ~268 at 64 -- while Q8_0 SmolLM2 loses to the
tile at every width. So rows-a-call (16/32/64, or GEMM off) is a fourth prefill
duel (`newGEMMRowsDuel`, after participants and chunk width, before the row
chunk). M4 CPU pp512, 8 samples an arm, settled duel against b77374a:

| model | tile | GEMM+duel | | duel chose |
|---|---|---|---|---|
| SmolLM2-360M Q8_0 | ~1160 | ~1180 | parity | off |
| Llama-3.2-1B Q4_K_M | ~330 | ~510 | 1.55x | 32 |
| tinyllama q3_K_M | ~324 | ~412 | 1.27x | 32 |
| Qwen2-1.5B Q4_K_M | ~320 | ~418 | 1.31x | 16 |
| gemma-2b | ~252 | ~272 | 1.08x | 64 |

★ AND THE INTEGER FOLD MOVES LOGITS THE WAY ANY REASSOCIATION DOES. NMSE 1e-12
a matmul becomes the 0.4-0.95 logit band once int8 re-quantization flips
roundings. `TestBatchMatchesForward` gained prefill's two modes (exact-gemm
demands bit equality -- 0.000e+00 on every model, both hosts; shipped is judged
by margin and a 1e-2 tripwire: qwen3moe 4.3e-3, flips on 0.003-0.07 margins),
and `TestGreedyMatchesLlamaCpp` measures the exact arithmetic, because its
coverage ratchet exists to catch drift: Llama-3.2-1B on the M4 flipped a 0.078
tie at free-running token 3 with the integer fold and reads 20 of 20 without.
The Xeon rows run the same duel now and must be re-taken.

★ AND `TestRaggedBatchMatchesForward` MISSED THAT CHANGE AND STAYED STRICT, SO IT
WAS RED ON EVERY HOST THAT RUNS THE FOLD -- the M4 too, whose shipped arm now
logs six undecided qwen2vl flips. On the Xeon E5-2680 v4 (and on the laptop
with VNNI forced absent, flip for flip the same) qwen2vl and qwen3moe
flipped argmaxes on 0.002-0.32 margins, which read as a GEMM bug "far beyond
rounding". It is not one, measured three ways:

    the GEMM on the model's own weights and activations, every batched call
      Q4_K  int fold vs float epilogue  NMSE 3.6e-13   both vs the oracle 8.459e-03
      Q6_K                              NMSE 1.2e-12   both vs the oracle 5.154e-03
    the kernel gate, per format, at Qwen2-VL's shapes too (k 1536, 8960)
      Q3_K 1.4e-12  Q4_K 3.2e-13  Q5_K 2.0e-12  Q6_K 4.4e-13, native VEX = forced
    the control: the FLOAT GEMM's output times (1 + eps*N(0,1)) in its place
      eps 0     bit-identical, every gate green
      eps 1e-6  the same failure: 0.02-0.21 margins, logits NMSE 1.0e-03
      eps 1e-8  still the same failure -- a relative 1e-8 is below an f32 ulp
                on most elements, so this is a handful of flipped last bits

So the model, not the kernel, sets the size: ANY bit that differs between the
batched and the decode matmul reaches the logits at NMSE ~1e-3 and a max|dlogit|
of 0.5-0.9 within a few tokens, because each flipped int8 rounding of an
activation is a whole quantization step. The saturation hypothesis the red gate
suggested is ruled out separately: the gate's saturated rows fail at |d| ~1e8
against a bound check disabled, and pass with it. The ragged gate has the batch
gate's two modes now; its shipped arm catches a batch that keeps a retired
row's stale history at logits NMSE 4.6e-02 to 2.5e-01 against 8.9e-04 / 3.3e-03
clean.

`TestForwardBatchFaultsItsBlocksIn` was a third gate holding ForwardBatch to
decode without knowing the fold, at an NMSE bound of 1e-3, and qwen3moe read
**2.852e-03** on the Xeon (and identically on the laptop with VNNI forced
absent). Bisected per op on two rows: the first difference is the layer-0 q/k
projection, row 0 at NMSE ~1e-14 and row 1 ~1e-9, growing until one int8
rounding flips (layer 24: attn_norm 7.7e-09 -> k 1.8e-05). With
`nn.WithGEMMExact` the two paths agree at all 844 trace points and the gate
reads 0.000e+00 on all four models, Xeon and laptop. The gate is about
faulting blocks in, not the arithmetic, so it pins the float epilogue and
demands equality now; with `pageIn` removed from the batched path it fails as
before.

### arm64 decode: the same integer fold in the fused matvec, and why decode did not move

`EmitA64PackedMatVecFusedWin` gives the fused matvec the stationary GEMM's
integer super-block fold, operation for operation, so a decode through it and
a prefill through the GEMM are BIT-identical
(`nn.TestFusedIntFoldMatchesTheGEMMExactly`, 12 format/shape pairs; swapping
the two final FMLAs fails it; its selection check demands the float form
differ). `Config.GEMMExact` now means "every packed kernel on its float
epilogue", which the exactness gates and the llama.cpp ratchet use.

★ AND IT BUYS NOTHING IN DEFAULT DECODE ON THE M4, MEASURED: tg128 CPU, same
pass, 0c1f404 against the fold, A/B/B/A -- Llama-3.2-1B 83.2 / 83.2, Qwen2
62.2 / 62.4, tinyllama 101.7 / 101.5, Qwen3-MoE 104.3 / 103.3, SmolLM2 and
gemma unchanged. The per-shape picker runs the TILED kernel at every model
shape: pinned, the fused arm reads 50.0 (float) / 51.9 (fold) and its k-sliced
arm 47.0 / 53.1 tok/s against the tiled 83.2. So the fold helps the kernel it
is in and that kernel does not run. The tiled kernel reads ~68 GB/s against
llama.cpp's ~70 on the same box (85.45 tok/s x 0.808 GB), i.e. M4 CPU decode
is at the bus both engines reach; the 0.91-0.97 rows are bytes and bus
efficiency, not issue slots.

## The 12-byte Q4_K/Q5_K scale plane, container v27

The Xeon entries above ordered it: jitllm read 3.7% more bytes a token than
llama.cpp because a Q4_K super-block's sixteen 6-bit scale/min fields took a byte
each (16 bytes, 148 a super-block) where the GGUF packs them in 12.

**The layout (`kernels.ScStream`, jit/gpu/kernels/scstream.go).** The fields run
in consumption order -- sub-block 0's scale, its minimum, sub-block 1's scale, ...
-- as one 96-bit stream, three words a super-block per row, [word][row] as
before. Fourteen fields sit inside a word and read as the two shifts a byte cost;
two (sub-block 2's minimum, sub-block 5's scale) straddle into the NEXT word
only. A super-block is exactly 96 bits, so a row's whole plane is one contiguous
stream with sub-block i's pair at bit 12*i. Q4_K is 144 bytes a super-block and
Q5_K 176, both +0.0% over their GGUF; Llama-3.1-8B's container went
5,284,728,832 -> 5,198,925,824 bytes.

**Readers.** The host kernels unroll a super-block, so field positions are
emit-time constants (`jit/cpu/scstream.go`: `scField`, `scStep`). The SC cursor
steps three plane strides a super-block, and a straddling field reads the next
word -- on amd64 by stepping the cursor there and back around the load, because
no general-purpose register is free (R14 is g); arm64 uses a spare X register.
Converted: amd64 fused, tile, tiled, wide, prefill GEMM, SSE tile and fused, and
both row kernels; arm64 matvec (all three forms), tiled, GEMM and row. Device:
matvec, MMA, Volta MMA and GEMM, Metal GEMM, get-rows, the GPU unpack writer, and
the oracle. The Go restatement (`realpack_test`) reads the stream bit by bit.

**Gates.** Every gate passes on the laptop (AVX2, SSE forced, CUDA, Vulkan), the
M4 (arm64 kernels and model gates, TestGreedyMatchesLlamaCpp 20/20 on the Q4_K
rows, Metal) and the V100 (sm_70). Stopping the cursor from advancing inside a
super-block fails 8 gates on amd64 and 2 on arm64; dropping the high word from
SPIR-V's funnel fails TestMatVecMatchesReference.

**Measured, same pass, v26 against v27 binaries each on its own container.**

    Xeon E5-2680 v4 node 0, 12 cores, tg128 (llama.cpp b11222 beside them)
      Llama-3.1-8B   11.11 -> 11.365  +2.3%   ratio 0.912 -> 0.933
      Qwen3-8B       10.90 -> 11.175  +2.5%   ratio 0.921 -> 0.945
      Llama-3.2-1B   59.3  -> 58.8    -0.6%
    M4, Llama-3.2-1B
      CPU   tg128 83.0 -> 84.2 (+1.4%), prefill level
      Metal tg128 112.4 -> 114.2 (+1.6%), pp512 1532 -> 1526 (-0.4%)

    V100 GPU 0, CUDA, pp512 / tg128 (the final device readers below)
      Phi-4-mini     +0.2% / 199.0 -> 199.9 (+0.4%)
      gemma-3-12b    +0.8% /  76.8 ->  77.4 (+0.8%)
      Llama-3.1-8B   level (noisy) / 134.85 -> 136.7 (+1.4%)

The two small losses were accepted as recorded costs on the user's call. No
page is transformed at run time on any tier: the converter writes the stream
once and every reader takes the page as stored.

**The device reader took four forms, and what finally mattered was a live
range.** V100, Q4_K 14336x4096 at 512 tokens, the Volta GEMM at 254-255
registers either way:

    v26 layout                                       872-877 us
    pair's two words both pipelined                  model prefill -1.2..-1.7%
    first word pipelined, straddle word in stage     885 us
    first word CARRIED across trips (one load/trip)  885-892 us
    unrolled x2/x4/x8 (constant field positions)     926-936 us
    floor probe (constant scales, wrong answers)     921 us  <- fewer ops, slower
    carried, its update emitted LAST in the trip     867-871 us

At the register ceiling instruction count does not predict time: the floor probe
did the least work and was the slowest. A review named the fix -- the carried
word's update (`scNext`) had been emitted beside `stage`, stretching its live
range over the loads and the barrier; moving it to just before the loop's phis
put the kernel level with v26. A register cap (168-224) spills at every value;
the 4x4 tile's accumulators alone are 128 registers. The matvec needed
`ir.OpFunnel` (one `shf.r` on PTX) and the contiguous-stream index (12*i) to come
back level: its first form, with the super-block split and a clamped successor,
cost Phi-4-mini's decode 2%.

## Past half a socket the gap is DRAM row locality, counted at the controller

After v27, Llama-3.1-8B on the Xeon (node 0, 12 cores) reads 11.07 tok/s against
llama.cpp b11222's 11.85. The memory controller's per-token counts, taken as the
difference between n=512 and n=128 runs of each engine so load, prefill and
warm-up cancel (`perf stat -a -e unc_m_cas_count.rd,unc_m_act_count.rd`):

    per decode token     bytes read   row activations   time      bandwidth
    jitllm               4.78 GB      38.4 M            90.3 ms   52.9 GB/s
    llama.cpp            4.71 GB      35.2 M            84.4 ms   55.8 GB/s

    reads per activation      6 cores    12 cores
    jitllm                    2.3        1.97
    llama.cpp                 4.5        2.12

So what remains is 1.5% of bytes and 5% of sustained bandwidth, and jitllm
opens 9% more DRAM rows a token. At six cores its row locality is half
llama.cpp's and costs nothing (8.98 against 8.94 tok/s: the controller is not
saturated); at twelve the controller is, and the extra activations are the
likeliest account of the bandwidth. The fused kernel walks every plane of k for
one 16-row group -- 64 bytes from each of ~512 planes, each plane a separate
DRAM row -- before moving to the next group; llama.cpp streams whole rows.

**A confound refuted first.** llama.cpp mmaps its GGUF from /dev/shm, whose
pages were placed when the file was written and which `numactl --membind=0`
does not move. `/proc/<pid>/numa_maps` during a run: `bind:0 ... N0=1199439`,
every page on node 0. It is not reading through both sockets' controllers.

**The wide/tiled arm confirms the direction and is not the answer.**
`JITLLM_MVPICK=1` (accumulators in memory, a worker's whole row range per
plane) reads 2.38 per activation against the fused kernel's 1.93 -- and runs at
3.48 tok/s against 11.24. The row-group-contiguous shadow above measured null
in tok/s and was never counted at the controller; why it did not pay while the
activation counts point at locality is open.

### Huge pages for the frames: prefill pays, decode does not care

Counted at 12 cores, jitllm walked the page tables 27x as often as llama.cpp
(1.67 G walks, 4.6% of every cycle, against 0.062 G) and ran with its L1 fill
buffers full on ~10% of cycles against almost none. The frames were on 4 KB
pages -- the box's THP mode is `madvise` and nothing asked -- while the plane
layout jumps a 57 KB stride per payload word.

`jlm.placeMemory` now advises `MADV_HUGEPAGE` on every frame and dense region
(no mmap: an advisory call on the Go buffers the pages are read into). It takes:
AnonHugePages 4.77 of 4.99 GB resident, page walks 1.70 G -> 0.09 G, walk cycles
72 G -> 4.25 G. And decode does not move, which says the walks were hidden under
the bandwidth limit: fill-buffer-full cycles 153 G -> 145 G, i.e. the cap on
outstanding lines per core is still what binds. Same pass, node 0, 12 cores:

    tg128           4 KB      huge
    Llama-3.1-8B    11.27     11.24    level
    Qwen3-8B        11.12     11.15    level
    Llama-3.2-1B    58.4      57.6     -1.2%   (a recorded cost; cause unknown)
    pp512
    Llama-3.1-8B    51.4      55.6     +8.2%
    Llama-3.2-1B    362       366      +1.1%

On by default on the user's call ("if no [smart way], make it default"): the
page size is fixed at first touch, so no per-phase switch exists, and a size
threshold from two points would be a guess. It is per model:
`model.WithHugePages(false)` (the CLI's JITLLM_HUGEPAGES=0) turns it off for
that container, and NUMA interleave is `model.WithInterleave` the same way --
both were process-wide switches until the same day, which in a server let the
last model loaded decide placement for every model (`jlm.TestPlacementIsPerFile`
fails on both of the interleaved File's frames against that). The 1B decode loss is unexplained; one candidate is that
physically contiguous 2 MB pages line the plane stride up with the DRAM bank
map where 4 KB pages scattered it. Not measured.

### The prefetch hint is not the lever either

With the frames on huge pages the Xeon still runs its fill buffers full on
~9% of cycles (`l1d_pend_miss.fb_full / cycles`) where llama.cpp reads ~0.
Word-ahead prefetch pinned (`JITLLM_FPF=64`), node 0, 12 cores, tg128, two
rounds each, throwaway builds:

    hint at +8 words        8B tok/s        fb_full   1B tok/s        fb_full
    T0 (shipping)           11.15-11.27     0.090     58.3-58.6       0.056
    T2                      11.20-11.30     0.096     58.7-59.1       0.055
    T2 +8 and T0 +4         11.14-11.22     0.131     57.0-57.5       0.124
    NTA                     10.77-10.95     0.078     56.9-57.3       0.039

T2 is within noise (+0.4-0.5%); two prefetches a word fill the buffers and lose;
NTA empties them and loses 3%. The standing reading: llama.cpp's one long
sequential stream per thread is fetched by the L2 streamer, which holds no L1
fill buffer, while jitllm's per-plane runs (a chunk's rows x 4 bytes, then a
plane stride) are too short for it to train on, so its lines arrive as demand
misses and T0 prefetches. That makes RUN LENGTH per plane the next variable to
dose, not the hint. Not built.

### What the Xeon's decode gap was, measured down to the shape

Xeon E5-2680 v4 (the CPU-only server), node 0, 12 cores, CPU decode, llama.cpp b11222
in the same pass throughout.

**The L2 streamer is llama.cpp's edge, and half of it reaches jitllm.** MSR
0x1A4 bit 0 (L2 hardware streamer) off on socket 0, tg128, restored after:

    Llama-3.1-8B   llama.cpp 12.15 -> 7.33 (-40%)   jitllm 11.28 -> 8.78 (-22%)
    Llama-3.2-1B   llama.cpp ~62.5 -> ~38  (-39%)   jitllm 58.9  -> 46.9 (-20%)

With the streamer off jitllm wins 1.20x: its own prefetching moves memory well,
and it gets less from the streamer because a worker reads each payload word as
a run of (its rows x 4 bytes) and then jumps a plane stride.

**But the layout is not what loses, which the row-contiguous shadow above had
already said.** A dose that packed rows as contiguous tiles
(`nn.MatVecPackedMulti` over tiles vs over views of one plane) read 95-107 GB/s
at first -- above the ~58 GB/s wall, because the test matrices fit the 35 MB
L3. Cycled over copies past L3, with dispatch held fixed, tiles gain 1.00-1.67x
over views; but the shipped `MatVecPacked` beats both, and already reads the
8B's gate/up/down and head at 56-60 GB/s in isolation.

**A lead left open: `MatVecPackedMulti` dispatch costs 0.40-0.89x of
`MatVecPacked` at identical bytes and layout** (views of one plane, 12 cores,
from DRAM). It has none of `MatVecPacked`'s k-slicing or NUMA placement, and it
is the batched MoE path, default on amd64. It does not show in the model:
olmoe-1b-7b on the same socket, tg128, batched 62.5-63.5 against the per-expert
loop 62.6-63.7 tok/s (llama.cpp 59.6-62.4 in the same pass), so it is not
built.

**In the model, the matvecs are the token.** Steady-state per-call medians on
Llama-3.1-8B (256 tokens, mean/median 1.004, so no tail) sum to 85.4 of a
91.0 ms token: q/o 180 us (52 GB/s), k/v 53-79 us (44 GB/s), gate/up 612 us
(54 GB/s), down 573/844 us (58 GB/s), head 8.2 ms (53 GB/s). Isolated, the
small shapes looked far worse (k at 20 GB/s); in a block page q, k and v sit
together and the streamer carries over, which the isolated loop does not
reproduce. k-slicing the small shapes (s=4 took k from 20 to 49 GB/s
isolated) is worth ~1 ms in context and measured null in the model (the
per-shape picker for slices, amd64: 11.21 against 11.18 tok/s) -- agreeing
with the refusal above.

**The prefetch distance was the lever, and the duel could not find it.** In
the model, pinned (`JITLLM_FPF`), tg256: auto 10.98, 0 10.32, 2 11.37, 4 11.29,
8 11.03, word-ahead 11.18. Same pass, pinned 2 against 64: 8B +3.9%, Qwen3-8B
+6.0%, 1B +5.8%. The whole-token duel starts on word-ahead and needs 2% to
leave it; two identical fresh runs settled on 4 and on 64, and the cache then
held whichever. Shapes also disagree -- q/o best at 2, gate/up at 4, the head
at 8 -- so no single distance is best.

`JIT.distPick`: on x86 each (format, rows, k, stride) times the five distances
on its own first calls (15 rounds each, fastest median), per JIT, nothing
cached across processes. The largest shape's choice sets `packedFused`, which
the batched paths run. Every distance is emitted at `NewJIT`, so the variants
share the JIT's emitter state and no timed call pays for an emission. The
whole-token duel remains only for a pinned distance or picking off.
`nn.TestPrefetchDistanceIsPickedPerShape` checks bit equality against a pinned
distance and that every distance ran; against a call that ignores the pick it
fails. Result in docs/perf/current.md: 0.953 / 0.967 / 0.952.


## Two gates that failed only on the Xeon were the pool's WIDTH, not the pre-VNNI kernels

Both read as kernel-family bugs because the Xeon E5-2680 v4 is the one host
here without VNNI: `model.TestDecodeDoesNotAllocate`'s olmoe host arm at 128
allocations over 64 tokens, every one in `nn.(*JIT).fusedSliced`, and
`model.TestForwardBatchFaultsItsBlocksIn` reading tinyllama's batch against
its decode at NMSE 5.872e-14 where the gate wants 0. Same binary, same box,
only the CPU mask moved:

    taskset 0-13 (14 workers)   olmoe 128 allocations   tinyllama 5.872e-14
    taskset 0-5  ( 6 workers)   olmoe   0               tinyllama 0.000e+00

and on the laptop with VNNI forced absent (`model.TestExactGatesRunPreVNNI`)
every exactness gate reads NMSE 0 and every host arm 0 allocations. So the
DotVEX family is exact and allocation-free; what moved was `fusedSlices`.

It splits a matrix of 32M weights or more over k when a worker's balanced row
chunk is under 4096 rows. olmoe's 50304-row head is 3593 rows a worker at 14
and 8384 at six; tinyllama's 32000-row head is 2286 and 5334. So both heads
are k-sliced on a Xeon socket and on no laptop, and that one path carried
both faults:

- **Its pool region was a closure over the call's locals**, two heap objects
  (the closure and the `fusedCall` it captured) a call, once a token. The
  other per-token regions had been moved onto `hotRegions` for exactly this;
  `hot.sliced` joins them.
- **It is a different summation order.** Two k-halves added together is a
  reassociation (bounded at NMSE 1e-10 by `TestFusedKSlicesMatchWholeK`),
  while the batched path sums each row over the whole of k. `GEMMExact` --
  the option every exactness gate sets, documented as summing every batched
  matmul in the matvec's order -- now also keeps the decode matvec on the
  whole of k unless a slice count is pinned, and a timed choice under it (the
  arm64 picker) never picks a split.

Neither gate could have seen this on a six-worker laptop: no shape the suite
decodes is split there. Both are now reached on any host:
`nn.TestGEMMExactDecodeSumsTheWholeK` sizes its matrix (8192x4096) against the
pool so the automatic split happens -- a control JIT must split it -- and holds
GEMMExact's decode to the batch bit for bit; against the GEMMExact line
removed it fails at 10797 of 16384 elements differing (Q6_K) on the laptop.
`TestDecodeDoesNotAllocate`'s `host-ksliced` arm pins two slices so every fused
matrix takes the path, and against the closure restored it fails on olmoe
with thousands of allocations over the window -- 1024 at each call site that
reaches the split, where the Xeon had seen only the head's 128.

The DoNodes path is the same shape of exposure -- a region only a two-node
pool runs, and what `cmd/jitllm` ships on that box -- so the gate has a
`host-numa` arm too, which skips naming the reason on a one-node mask.

The arm failed on both BitNet fixtures there (`0 region(s) node-affine`), and
the cause was the fixtures' size, not the format. DoNodes places a packed
weight in blocks of one 4 KiB page of a word-line, 1024 rows, one block to each
node in turn (`jlm.TestInterleaveFollowsThePageAddress` is why a block that
size sits on one node), and `numaRows` takes a weight only when its row count
is a whole number of rounds of one block per node: 2048 rows on two nodes.
Every other count runs the plain region, correct and only slower, whatever the
format. The fixtures' weights have 128, 256 and 512 rows. Their 1344 fused
matvecs in the window (64 tokens, 21 weights a token) all went through the
call site that asks `numaRows`, as every packed format's do, so the ternary
path was never at fault. Counted over the same 64 tokens on the two-socket
E5-2680 v4 (six CPUs a socket), Llama-3.2-1B took 5120 node-affine regions,
the 80 a token of its 16 blocks' wq, wo, gate, up and down (2048 and 8192
rows), none for wk and wv (512 rows) or the head (128256); Qwen3-8B took 11520,
the 180 a token of its 36 blocks' same five, none for the 1024-row wk and wv or
the head.

The arm now skips a dense model none of whose packed weights has such a row
count (`nn.NUMARound`), naming the counts it found and the round. It reads the
counts from the container's entries, not from a list of weight fields that a
hybrid's own projections would escape. A mixture is never skipped: its experts
are placed by their pages.

`nn.TestFusedMatVecRunsNodeAffine` holds each of the eleven fused packed
formats, TQ2_0 and TQ2_F32 among them, to the node-affine path at two rounds of
rows and to the plain pool's bits, with a block past a round and a ragged count
as the controls that the counter discriminates (33 pairs). It states the
1024-row block itself and does not read nn's, so a change to the layout fails
it. Against violations: with `numaRows` declining everything it fails the
eleven two-round pairs and the arm fails Llama-3.2-1B; with ternary alone
declining it fails the two ternary pairs; with the block halved to 512 rows it
fails the eleven block-past-a-round controls.

BitNet b1.58 2B-4T, the one ternary release, has 2560-, 640- and 6912-row
projections (its safetensors header's 640, 160 and 1728 rows are four codes to a
byte), none a multiple of 2048, so by this rule it takes no node-affine region
on a two-socket host.


## Why the host kernel gate is NMSE < 1e-10, and not bit equality or ggml's budget

Recorded in the design document (ARCHITECTURE.md §7, tier T1) before any kernel existed, moved here
when that file was retired. The four input distributions are what
`jit/cpu/matvec_test.go` and `exec_a64_test.go` still sweep; the calibration numbers behind the bound
were written down nowhere else.

**The bound is four orders of magnitude tighter than ggml's 2.38e-7**, because that budget exists to
compare ACROSS quantization formats and would hide a real kernel bug. Restructuring a kernel (1 against
4 accumulators, VNNI against `vpmaddubsw`) moved NMSE by 1.3e-14 to 1.5e-13 in the design probes, so
1e-10 left three decades of slack. The integer part of a quantized dot is exact -- int8 x int8 into
int32 cannot overflow below K of about 133,000 (2^31 / 127^2) -- so all of the error is the f32
accumulation tree.

**Bit equality was measured and is not usable:** 184 of 2000 rows matched bitwise between the
1-accumulator and 4-accumulator versions of the SAME kernel. Reassociation changes the last bits by
construction. (Where the arithmetic is unchanged by construction, bit equality is the right gate
instead -- `nn.TestMatMulPackedMatchesTheRowLoopExactly` is one.)

**Uniform inputs were not enough.** The set is uniform, Student-t, one huge value per block (which
drives the block scale up and every other value toward zero quantized magnitude), and all +/-1 (which
saturates the int8 range and exercises `vpmaddubsw`'s int16 headroom of 1.6%: 2*127*127 = 32,258
against 32,767). The unsigned-weight form overflows that headroom -- 2*255*127 = 64,770 -- which is why
the pre-VNNI and SSE tiers carry their own centring (`jit/cpu/prevnni.go`, `sse_packed.go`).

**f16 accumulation measured 11x and 185x OVER the whole quantization budget**: NMSE 2.70e-6 at
K=4096 and 1.10e-5 at K=16384. f16 as KV STORAGE is a separate question, measured under the f16 KV
entry at the top of this file.

The companion tier was disassembly: the probe emitter's one bug was VEX.R taken from bit 0 of the
register number instead of bit 3, which silently turns `ymm3` into `ymm11`, and one disassembly found
it. That lives on as encoding tables checked against the system assembler and objdump
(`jit/cpu/asm_test.go`, `vexenc_test.go`, `scripts/vexref.sh`) rather than as per-kernel golden files.

## The design-phase measurements behind owning an emitter

From the design probes (C, one pinned P-core of the i9-12900HK, clang-18), moved from ARCHITECTURE.md
§0 when that file was retired. The EVEX finding is also the package comment of `jit/cpu/asm.go`; the numbers are
only here.

- **Go's assembler encodes VPDPBUSD as EVEX only**, and this chip has AVX-VNNI without AVX-512, so a
  Plan 9 `.s` kernel using it would SIGILL on the machine it was written for. Re-checked
  on go1.27.1: `cmd/internal/obj/x86/avx_optabs.go` still lists only `evex128/256/512` forms for
  `AVPDPBUSD`, and `simd/archsimd` exposes `DotProductPairs` (pmaddwd) and
  `DotProductPairsSaturated` (pmaddubsw) but no VNNI dot. `scripts/vexref.sh` records the same trap in
  GNU as: `62 f2 6d 28 50 cb` faults, `c4 e2 6d 50 cb` runs.
- **Kernel quality alone did not justify the JIT for decode.** The folk figure was "archsimd is 11.6x
  slower than C"; measured back to back in one process, same algorithm, it was 20.14 against 35.05
  Gmac/s -- **1.74x** -- and at four or more cores decode is memory-bound and absorbs most of that.
- **VNNI is a prefill instruction.** At one accumulator it was SLOWER than `vpmaddubsw`+`vpmaddwd`
  (29.3 against 35.1 Gmac/s); at twelve accumulators in a 6x2 register tile it reached 226 Gmac/s,
  90.3% of hardware peak. The tile sweep spanned 5.11x (1x1 at 44 Gmac/s), and the "obvious" 4x4
  spilled to 121 because AVX2 has sixteen YMM registers. The shipped emitter's own sweep is
  `jit/cpu` `TestGEMMTileSweep` (1.63x, per `gpu-kernels.md`).

## Darwin runs generated code without MAP_JIT, and the hardened runtime is what kills it

Measured on the M4 mini while bringing up the arm64 emitter; moved from ARCHITECTURE.md §5b when
that file was retired and checked against `jit/cpu/exec_darwin.go` and the arm64 test runner (a local helper, not in the repo) then. Both
carry the conclusion in comments; the four-way table was only in the retired file.

An earlier draft of the design said darwin needs `MAP_JIT` plus `pthread_jit_write_protect_np` around
writes. That would have made the arm64 tier impossible without cgo, since the latter is a libc
function with no syscall behind it. Neither is needed: `MAP_JIT` exists to keep ONE page writable and
executable at once under a hardened runtime, and jitllm never does that. `exec_darwin.go` is the Linux
sequence -- `mmap(RW)`, copy, `mprotect(RX)` -- and executes generated code on the M4.

The silent kill is real, and it follows the hardened-runtime flag, not the entitlement's absence:

| signature | CodeDirectory flags | result |
|---|---|---|
| as cross-compiled by Go (linker ad-hoc), no codesign step | `0x20002` | runs |
| `codesign -f -s -` | `0x2` | runs |
| `codesign -f -s - -o runtime` | `0x10002` | **SIGKILL, exit 137** |
| `codesign -f -s - -o runtime --entitlements` (allow-unsigned-executable-memory) | `0x10002` | runs |

The killed run printed `=== RUN TestA64ExecuteScalar` and nothing after it: stdout cut mid-fetch,
stderr empty, no Go-visible error, logged by the OS only as "AppleSystemPolicy: Security policy would
not allow process". A harness that reads the exit status sees an ordinary failure, and over ssh a
hang. The arm64 test runner (`scripts/local-mac-arm.sh`, a local helper not in the repo) therefore checks for a PASS/FAIL line in the output rather than the exit
code, and signs with the entitlement anyway -- one round trip, against the day someone adds
`-o runtime`.

Two darwin gaps from the same bring-up, both still true when this was moved here:

- **`bench.Guard` is a silent no-op on darwin.** It reads `/proc/pressure/cpu`, falls back to
  `/proc/loadavg`, and reads `/sys/class/thermal`; macOS has none of them, every read errors, and
  `Guard` returns nil. `scripts/vs-llamacpp.sh`'s core-busy and clock-coherence gates are Linux-only
  too (`/proc/stat`, `/proc/cpuinfo`), and its own comment says so. On the Mac the IQR/median gate
  and the A/A self-control are the only gates a row passes.
- **`cpu.OutLine` is sized to a 64-byte line on every architecture**, and the M4's line is 128 bytes,
  so two workers' rounded chunks can still share a line there. The prefill GEMM's rows-per-call duel
  (`engine/nn/tune.go`) measures that trade per host; whether decode chunks false-share on the M4 is
  unmeasured.

## The Xeon D-2123IT: AVX-512 is scheduled, and the prefill gap is traffic

★★★ **AND AVX-512 IS ADMITTED AND SCHEDULED, NOT REFUSED. THE WORD "REFUSED"
STOOD HERE FOR TWENTY MINUTES AND IT BROKE RULE 6 IN THIS FILE.** RULE 6:
"nothing is refused for being SMALL... A number decides ORDER, never admission.
'Not now' is a schedule; 'no' needs a regression behind it." 3.1% is not a
regression, and the user's correction is the rule verbatim: *"3.1% compounds on
a server; so would take the 3%."* A server runs continuously, so this is 3.1% of
every token for the life of the machine. **Its decode advantage IS 3.1% and it
costs an EVEX encoder.** llama.cpp runs `libggml-cpu-skylakex.so`
on that host -- 512-bit vectors, its own runtime dispatch chose it -- and jitllm
runs 256-bit AVX2 with no VNNI at all, three instructions per int8 dot where
VPDPBUSD is one. The gap is 3.1%. Both arms sit at ~70% of the same wall,
because **decode is bandwidth-bound and width has nothing to buy**: this file
already records the same shape from the other side (the same 112 block weights
read 47.95 GiB/s in GGUF layout and 48.47 packed -- "time tracks bytes alone").

So the EVEX encoder, the 32-register allocation and the width plumbing buy
3.1% of a decode token on this host. That ORDERS it behind the SSE4.2 tier
(which is 0 -> working on two boxes) and behind the prefill gap below; it does
not refuse it.

★★★ **AND PREFILL ON THE SAME BOX IS 0.265, WHICH IS THE LARGEST GAP IN THIS
PROJECT AND IS UNATTRIBUTED.** Same host, same model, same session, 415-token
prompt against llama.cpp's pp512:

    jitllm     34.63 / 35.61 / 35.92 tok/s     256-bit AVX2, pre-VNNI
    llama.cpp  133.55 +/- 0.33 tok/s           AVX-512 skylakex
    ratio ~0.265 -- a 3.8x gap against decode's 1.03x

**The obvious explanation is wrong and I checked before writing it down.** The
sentence "Prefill is Q3_K/Q4_K and therefore gets NOTHING from this yet" is
stale: `MaxTiledTokens` says Q4_K tiles at four tokens now, and only Q6_K --
the output head -- is refused, on CODE SIZE. So this is not a missing kernel for
the block weights.

★★★★ **ATTRIBUTED, AND IT IS TRAFFIC. THE TILE ON THAT HOST IS NOT
FOUR TOKENS -- IT IS ONE, BY CONSTRUCTION, AND THE PREFILL/DECODE RATIOS SAY SO
WITHOUT A SECOND MEASUREMENT.**

`EmitPackedMatMulTiled` REFUSES on a pre-VNNI host: *"has no pre-VNNI form; the
per-token fused kernel serves this host"*. Its comment says why -- `tiledBudget`
lands at EXACTLY 16 registers at tok=4 and the VEX dot needs a ones vector with
no temporary free across the payload loop -- and
`model.TestEveryModelRunsPreVNNI` runs every model on the box with VNNI forced
absent and the tiled kernel declined on all of them. So on the Xeon a prefill of n tokens makes **n** passes over the whole
weight set, where a VNNI host makes n/4 and llama.cpp's cache-blocked GEMM makes
a handful.

★ **THE MEASUREMENT THAT SETTLES IT WAS ALREADY IN THIS FILE, ONE SCREEN UP,
UNREAD: A PREFILL TOKEN COSTS THIS ENGINE EXACTLY WHAT A DECODE TOKEN COSTS.**
Same host, same model, same session:

    jitllm      prefill 35.61 / decode 31.04 = **1.15x**
    llama.cpp   prefill 133.55 / decode 31.99 = **4.17x**
    ratio of ratios **3.64x**, against the measured prefill gap of **3.75x**

**The whole gap is that batching buys jitllm 1.15x and llama.cpp 4.17x.** Nothing
is left over -- no kernel-quality term, no AVX-512 term, no Q6_K term.

★ **AND THE ROOFLINE PROVES llama.cpp IS NOT RE-READING, WHICH IS THE HALF THAT
NEEDS NO TRUST IN EITHER ENGINE.** Against the Xeon's 36.63 GB/s wall:

    jitllm     35.61 tok/s x 0.816 GB =  29.1 GB/s = **79% of wall**
    llama.cpp  133.55    x 0.808 GB = 107.9 GB/s = **295% of wall**

295% is impossible as a per-token read. jitllm's prefill kernel is not slow --
it is SATURATED at 79% of the bus while moving 415x the bytes it needs to.

★ **SO THE TWO CANDIDATES BESIDE THE TILE ARE REFUTED RATHER THAN MERELY
UNTESTED.** "The Q6_K head drops to the per-token matvec loop" is a rounding
error when EVERY weight does. And "prefill is the compute-bound half where
512-bit vectors should pay" is FALSE ON THIS HOST as it stands: at 79% of the
read wall, prefill here is bandwidth-bound, and width cannot help a kernel that
is already saturating the bus. AVX-512's value for prefill can only be measured
AFTER the traffic is fixed -- so this row does not raise the 3.1% estimate, it
defers the question.

★ **THE TWO LEVERS, IN ORDER, AND NEITHER IS AVX-512.**
  1. **A pre-VNNI token tile.** Even tok=2 halves the traffic on this host. It
     needs a narrower tile to free the ones vector, which is a different kernel
     with its own measurement -- owed, and the emitter says so rather than
     faking it.
  2. **Cache-block the row range, which helps EVERY host including VNNI ones.**
     `MatMulPacked` runs the token-tile loop OUTSIDE the row sweep, so a worker
     re-reads its whole weight slice once per tile: on Llama-3.2-1B's ffn_gate
     that is 9.4 MB against a 1 MB L2 and an 8 MB shared L3, so every tile goes
     to DRAM. Swapping the loops -- row-block outer, token-tile inner, blocked
     so a worker's slice fits L2 -- is what takes the sweep count from n/4 to
     ~1 and is the shape llama.cpp's blocked GEMM already has.

`nn.WithTileTokens` / `JITLLM_TILE_TOK` makes lever 1's value measurable BEFORE
it is built: a pin of 1 falls below `tiledFor`'s floor of two and emits nothing,
so **on a VNNI box `JITLLM_TILE_TOK=1` reproduces the Xeon's prefill exactly**
and 1-against-4 is a same-process dose.

★★★ **AND THE DOSE WAS TAKEN, AND IT REFUTES THE PREDICTION MADE
FROM THE TWO BATCHING FACTORS. THE TILE IS WORTH 32%, NOT 3.17x.**
`model.TestPrefillTileDose`, Llama-3.2-1B Q4_K_M, a 256-token prompt, six
P-cores, 21 rounds, ABBA, A/A self-control FIRST:

    A/A self-control (tile 4 in BOTH arms)  **1.0000**  IQR/median 0.092  n=21
    pass 1   tile 4 / tile 1                **1.3257**  IQR/median 0.101  n=21  REJECTED
    pass 2   tile 4 / tile 1                **1.3227**  IQR/median 0.027  n=21  GATES

    tile 1   161.7 / 165.5 prefill tok/s    <- the Xeon's configuration
    tile 4   206.2 / 220.7

The two passes agree to **0.3 points** and the A/A bias is 0.00%; pass 1 blew
the gate on a tenant that arrived mid-run (cores 12.9% busy at the start, 53.3%
at the finish) and pass 2 was clean at the end (1.5%).

★ **THE PREDICTION WAS PRE-REGISTERED AND IT WAS WRONG BY 2.4x, WHICH IS WHY IT
WAS WORTH WRITING DOWN FIRST.** It came from combining the laptop's
prefill/decode ratio with the Xeon's and attributing the whole difference to the
tile: 3.64x / 1.15x = 3.17x expected. Measured 1.32x. **So the tile is real and
it is NOT the 3.75x gap** -- a pre-VNNI host at tile 1 still gets 2.7x of
batching on the LAPTOP, and only 1.15x on the Xeon, at the same tile setting. The
cross-box arithmetic silently attributed a property of the BOX to the KNOB.

★★ **THE LEAD FOR THE REMAINING TERM IS THE POOL CHUNK AGAINST THE LAST-LEVEL
CACHE, AND THE STRUCTURAL FINDING IS THAT NOTHING SIZES IT AGAINST ONE.**
`MatMulPacked` runs the token loop INSIDE the per-chunk function, so a worker
sweeps its row slice once per token tile and the re-reads are served by whatever
level holds the slice. `packedChunk` sizes that slice from `cpu.RowsPerCall`,
which is `maxCallNanos` -- a **GC-PREEMPTION budget**, 100 us of generated code
so the runtime is not blocked. Locality never enters into it. Censused on
Llama-3.2-1B's widest weight (ffn_gate, k=2048, 8192 rows, Q4_K, 1056 B/row):

    | | chunk | across workers | L2/core | L3 | L3 used |
    |---|---|---|---|---|---|
    | laptop, 6 workers | 1.39 MiB | 8.31 MiB | 1.25 MiB | 24 MiB | **35%** |
    | Xeon, 4 workers | 2.06 MiB | 8.25 MiB | 1.00 MiB | **8.3 MiB** | **99%** |

**The same working set is a third of the laptop's L3 and all of the Xeon's**, and
the chunk exceeds L2 on both, so no re-read ever hits L2 on either box. That is
consistent with the two roofline readings -- the laptop prefills at 317% of its
DRAM wall (re-reads served from L3) and the Xeon at 79% (re-reads reaching DRAM)
-- and it predicts that capping the chunk so the slice fits L2 pays on BOTH
hosts, most on the Xeon.

★★★★ **AND THE CHUNK-SIZE DOSE WAS TAKEN ON THE XEON AND REFUTES IT, THREE
DOSES, MONOTONIC, EVERY ARM GATING. NARROWING THE CHUNK TOWARD L2 COSTS 20-39%.**
Llama-3.2-1B Q4_K_M, 256-token prompt, four cores, 15 rounds, ABBA, A/A first, on
an idle box (`model.TestPrefillTileDose` with `JITLLM_CHUNK_B`):

| cap | chunk rows | pass 1 | pass 2 | IQR/median |
|---|---|---|---|---|
| **A/A, no cap in both arms** | 512 / 512 | **0.9991** | -- | **0.004** |
| 1536 KiB | 512 -> 224 | **0.7976** | **0.7883** | 0.004-0.005 |
| 1024 KiB | 512 -> 144 | **0.7136** | **0.7192** | 0.007-0.008 |
| 512 KiB | 512 -> **64** | **0.6281** | **0.6145** | 0.007-0.009 |

**Monotonic in the dose, both passes agreeing to within 1.4 points at every
level, against a harness bias of 0.09%.** At 512 KiB the chunk DOES fit the
Xeon's 1 MiB L2 -- and it is 37% slower. So fitting L2 is worth less than what it
costs: capping the chunk so the slice fits L2 was tried at three doses and came
out 20-39% slower, which is a measured result rather than an unconfirmed one.

★ **WHAT IT DOES NOT TOUCH.** It rules out the proposed FIX, not the attribution:
the prefill/decode ratios and the roofline stand on their own evidence, the
re-reads still happen, and shrinking the chunk simply does not make them cheaper.
The remaining lever is still lever 1, a pre-VNNI token tile, whose value the
laptop dose puts at 1.32x.

★ **AND THE MECHANISM OF THE SLOWDOWN IS UNATTRIBUTED, WHICH IS WORTH SAYING
RATHER THAN GUESSING.** Call-entry cost does not explain it: an eightfold
narrower chunk multiplies kernel entries by eight, and at ~10 ns an entry that is
~0.8% of a 4.6 s prefill, not 37%. The candidate is the same class of thing the
Metal work found on the other tier -- a shorter contiguous run in the row
direction gives the hardware prefetcher less to work with -- and it is a
candidate, not a claim. `nn.WithChunkBytes` stays as the instrument that produced
this row; its default is 0, which is the behaviour that was always there.

**The two-sentence version of the whole prefill investigation: the gap is
traffic, the tile is worth 1.32x of it, and the cache-blocking that looked like
the rest is a 20-39% REGRESSION.**

★★★ **WHICH LEAVES ONE LEVER STANDING AND IT IS A GEMM DESIGN DECISION, NOT A
PATCH: BLOCK OVER k, NOT OVER ROWS.** The tile is 1.25-1.32x of a 3.75x gap, so
even a pre-VNNI tile at three tokens leaves ~3x. That 3x is the structural
difference and it is now precisely stated: **a jitllm chunk is `rows x the WHOLE
k`**, which is why the slice is megabytes and why no cache holds it; llama.cpp's
blocked GEMM tiles k as well, so its weight panel is kilobytes, every token in
the batch streams through it, and the panel is read once.

★ **AND THE REASON THIS TREE DOES NOT ALREADY DO IT IS RECORDED ABOVE, FOR THE
WRONG PHASE.** Accumulators cannot stay in registers across k-blocks -- a k-block
computes a PARTIAL sum -- so they must live in the output, which is the "wide
kernel, accumulators in memory" design this file measures at **0.8771 against the
fused kernel's 1.0040**. That measurement is DECODE, where the output is ONE row
and re-reading it per k-block buys nothing. In PREFILL the output is
`rows x tok` f32 against `rows x k` of weight -- for Llama-3.2-1B's ffn_gate at
256 tokens that is 8 MiB of accumulator against 9.4 MiB of weight read
ceil(256/tile) times now. **The trade is inverted and it has never been
measured in the phase where it would pay.**

★ **THE ORDER, GIVEN WHAT IS NOW MEASURED RATHER THAN ASSUMED.**
  1. **A pre-VNNI token tile** -- 1.25-1.32x, the only lever with a number, and
     bounded: `ones16` costs ONE register and `tiledBudget` lands at exactly 16
     at tok=4, so a DotVEX form fits at THREE tokens. The work is that
     `dotEmitter.unsigned` CLOBBERS its input where VPDPBUSD does not, so the
     nibble mask has to be re-derived per token inside the tile loop rather than
     hoisted above it -- which is what `packedfused.go` already does and is the
     pattern to follow, not a new idea.
  2. **k-blocking**, which is the remaining ~3x and is a design fork: it needs
     the accumulator decision above re-priced for prefill.

**Row-blocking was tried.** The three-dose response above measured it 20-39%
slower, and `nn.WithChunkBytes` exists so that result can be re-derived in one
command rather than re-argued.

The prompt lengths differ (415 against 512) and prefill rate rises with length,
so llama.cpp is slightly flattered -- by nowhere near 3.8x.

★ **AND IT VALIDATES THE PRE-VNNI PATH, WHICH WAS BUILT ON A PREDICTION.** A
three-instruction VPMADDUBSW/VPMADDWD/VPADDD sequence at 256 bits reaches 97% of
a hand-tuned AVX-512 engine on the same box and model. The remaining ~30% of
wall is not the dot product.

★ **CAVEATS, STATED RATHER THAN BURIED.** n=128 and not RULE 1's 320. ONE pass,
not RULE 2's two -- with a 0.57% within-arm spread, which is the reason to
believe it and not a substitute for the second pass. The jitllm arm is the
PRE-VNNI worktree build, not main, because main has no kernel for a host without
VNNI. And llama.cpp here is **b11011**, a different build from the laptop's and
the Mac's b10825, so this row is comparable WITHIN this host and is not a
scoreboard entry beside the others.

## RULE 5 as it stood in AGENTS.md: the core set, and the worker pinning that was removed

### ★ RULE 5: SIX PHYSICAL P-CORES: `taskset -c 0,2,4,6,8,10`.

More cores is worse for decode. E-cores add no read bandwidth and become the
straggler behind every barrier; hyperthreads are neutral to negative; everything
at once is worst. **Prefill's core set is per host, and the old "prefill wants
6P+6E (+34%)" does not hold on the container path** -- it was measured on the
pre-container engine. On the container
path the laptop's E-cores are neutral in prefill (1.0024 / 0.9952 against a
1.0275 A/A, Open questions below), the M4's are worth +14-54% only
through prefill's OWN pool (`5b04d82`), and the Xeon E5-2680 v4 has no E-cores:
its rows are taken on one socket's twelve cores under `numactl --membind=0`.
**Never all 14**: leaving the OS nothing to run on costs more than the cores buy -- the
coreset A/B measures it end to end, pe 0.864x, smt 0.965x, all 0.807x against p,
and `SetParticipants` measures the same shape from the other side (a prefill GEMM
scaling to five participants and COLLAPSING at six, 1220 -> 428 GFLOP/s with the
spread going 5% -> 121%).

★ **THE "620 ns -> 11.9 us SPIN BARRIER" THAT JUSTIFIED THAT SENTENCE WAS
NEVER A MEASUREMENT OF THE SHIPPED POOL, AND IT HAD REACHED SHIPPING SOURCE.** The
design document (ARCHITECTURE.md, since deleted) had retired it in its own header -- *"a C probe of a barrier that
was never built"*, recorded now under RULE 11 in
`docs/engineering-history/scheduling-and-measurement.md` -- while
`engine/sched/pool.go` went on citing it as measured, in two places, and this rule cited
it in a third. The SHIPPED per-region dispatch cost is `model.TestFusionCeiling`'s
**2.70 us on Linux and 3.008 us on the M4**. The conclusion does not change; its
evidence does, and a number that survives the measurement it came from is how a
withdrawn finding keeps the appearance of a binding rule.

★ **AND "pin the cores" MEANT TWO DIFFERENT THINGS. ONE IS MEASURED AND ONE WAS
NEVER MEASURED AT ALL; THE SECOND IS DELETED.** The PROCESS mask --
`taskset` -- is RULE 5 and stands. Binding each worker to ONE cpu with a raw
`sched_setaffinity` was a separate thing `sched.Pool` did, asserted three times
in identical words ("an unpinned worker is slow, not wrong") and demonstrated
zero times: every number the package cites varies the core SET or the worker
COUNT with pinning held on in all arms. The M4 settles it from the other side and
by MEASUREMENT rather than absence -- `pin()` has always been a no-op on darwin,
and xctrace over 200 tokens puts 98% of samples on P cores anyway
(`docs/engineering-history/cpu-kernels.md`: *"macOS QoS places the
workers on P cores by itself, so topo_darwin.go being unable to pin costs
nothing measurable"*).

★ **AND REMOVING IT IS A DECODE WIN, GATED, NOT MERELY NEUTRAL.**
Llama-3.2-1B-Instruct Q4_K_M, CPU decode, n=160, six P-cores, one process per
sample, ABBA, TWO passes (RULE 2), A/A self-control run FIRST:

    A/A self-control (one binary, both arms)   1.0027   IQR/median 0.015
    pass 1  n=12                               1.0382   IQR/median 0.026
    pass 2  n=22                               1.0398   IQR/median 0.023
    both    n=34                               1.0382   IQR/median 0.026

    unpinned  59.93 tok/s = 48.9 GB/s = 87% of the 56.2 GB/s 6-thread wall
    pinned    57.67 tok/s = 47.1 GB/s = 84%          816,010,912 B/token

The two passes agree to 0.16 points and the harness bias is 0.27%, so the 3.8%
is the change. The mechanism is the one the leak hid: a worker bound to one core
cannot move when the caller's serial stretch (RMSNorm, RoPE, the residual adds)
or the OS wants that core, and an unpinned one can. This is a FRESH process, so
no leaked mask exists for the fix to remove -- it is pinning's own standing cost.

It also LEAKED. The mask is invisible to the Go runtime, the worker unlocked its
thread on exit, and a single-CPU-masked thread went back into the runtime's
reusable pool -- silently confining whatever ran on it next, including the pool's
own unpinned caller. A vision-tower encode measured **1.0 s in a fresh process
and 81-244 s in one where four earlier tests had created and closed pools**, and
removing the binding took the median from 130.66 s to 0.97 s (ABBA, 8 samples an
arm, one binary each) with the run-to-run spread going 2.4x -> 1.3x. The whole `model` package -- 67 tests over six
non-overlapping shards -- went from **2195 s to 75 s, 29x**, and the two shards
that kept hitting the 600 s timeout stopped: every test that built a State was
paying for the threads the tests before it had left masked. The engine now integrates with the Go scheduler in
its own vocabulary instead: `cmd/jitllm` aligns `GOMAXPROCS` to the engine's
width at startup, `PCores`/`ECores`/`SMTSiblings` intersect `sched_getaffinity`
so a narrower `taskset` yields fewer workers, and `LockOSThread` survives only
where an API genuinely binds state to a thread (CUDA contexts, `backend/cuda`).

**Measure at the thread count that SHIPS.** A win measured on one core has
evaporated or reversed at the real thread count eight times here — the single
most repeated mistake in the evidence files.

## f32 is the engine's precision: how float64 left the load and inference paths

### ★ RULE 8b: f32 IS THE ENGINE'S PRECISION. THE ORACLE IS f32 TOO, AND f64 IS GONE.

**★ CLOSED. There is no float64 on any load or inference path, and
the ORACLE itself is f32 now.** `quant.Dequant32` decodes straight to f32 through
the same decoders (destination-generic, arithmetic and rounding boundaries
untouched, `out[o] = T(float32(...))`), gated at ZERO ULP against the f64
signature on all eight formats. The oracle's matvec (`internal/oracle.MatVec`,
test-only) takes and returns f32 and accumulates with NEUMAIER compensation -- ~2*eps independent of k against f64's ~6e-8, so the
NMSE bound barely moves, and the reference stays an independent METHOD (vectorised
FMA in the kernel, compensated scalar in the oracle) rather than the same
arithmetic twice.
Both Widen/Narrow pairs and both load-time f64 buffers are deleted.
`TestGreedyMatchesLlamaCpp` did not move a single token id.

**What float64 is left is measurement, not inference.** Outside tests the
`[]float64` in `engine/` are the tuners' ratio and median statistics
(`engine/model/seamtune.go`, `engine/nn/tune.go`) and the expert-use accounting
(`engine/model/hotness.go`); scalar float64 appears only as setup arithmetic
(an epsilon, a scale, an image resample step) before a kernel call. The
residual, the scratch vectors and the logits are float32 on the host, matching
the device tier, which already produced token ids identical to the old f64 host
path on every clean model (RULE 12).

What it buys: **the elementwise ops become emittable with the assembler that
already exists.** That is the whole of the case, and it is worth stating narrowly
because the obvious second claim is FALSE.

**★ IT IS NOT A PERF WIN, MEASURED.** llama-2-7b, the same commit either side of
the container change, ABBA, n=128:

    per-round f32/f64   1.0047   0.957   1.0243     -> unresolved, ~neutral
    RSS                 4,168,800 kb -> 4,166,300   -> 2.5 MB, 0.06% of the process

**Decode is bound by WEIGHT traffic and the conversions were not weights.** A 7B
token reads ~3.8 GB of weights; the deleted passes touched the residual and
scratch, roughly 1M elements or 4-8 MB, under 0.2% of the traffic. Removing them
cannot appear in tok/s however many passes there were. The memory is small for
the same reason -- the weights are mmap'd GGUF bytes and were never f64, so only
the residual, logits, gate/up, norms and score rows halved.

That is RULE 6's regression checklist working: time UNRESOLVED, memory IMPROVED by
2.5 MB, and the change stays because nothing REGRESSED -- not because a number
justified it. The justification is what it unblocks.

The gate is `TestGreedyMatchesLlamaCpp` plus the teacher-forced diff on the
clean models.

## RULE 11d's cases: the M4 model gates, and the SSE tier's crash

### ★ RULE 11d: A GATE THAT NEVER RUNS ON AN ARCHITECTURE DOES NOT COVER IT.

**The `model` gates run on amd64 and nothing ran them on the M4, so arm64 decode
produced garbage for every quantized model and eleven green kernel gates said
nothing.** A local helper (`scripts/local-mac-arm.sh`, not in the repo) ships a
cross-compiled test binary to that box and has existed the whole time; what got run there was `./engine/nn` and `./jit/cpu` --
KERNEL packages. The one class of bug that lives above a kernel and below a
model was therefore uncovered on half the target hardware.

    model.test -test.run 'TestGreedyMatchesLlamaCpp|TestBatchMatchesForward|TestEveryModelRunsGenerated'

**Run the MODEL gates on both hosts before believing an arm64 change.** The ones
that catch this class compare the engine against something that is not itself:
the batched path (`TestBatchMatchesForward`) and llama.cpp's goldens
(`TestGreedyMatchesLlamaCpp`). Both were red on the M4 and green on Linux, as
was the since-deleted reference-tier comparison.

★★★★ **AND THEY WERE RUN ON THE M4, AND THEY ARE GREEN: 20 of 20
MODEL GATES AND 22 of 22 MLA GATES, AGAINST A REAL llama.cpp ORACLE ON THAT
BOX.** The rule above had been satisfied only in the kernel packages; the model
gates it names were still a command nobody had run since the arm64 decode bug
they were written for. Cross-compiled here, codesigned and run there:

    ./jit/cpu   470 run, 349 pass, 121 skip, 0 fail   (skips are TOOLING, below)
    ./engine/nn         66 run,  62 pass,   4 skip, 0 fail   (skips are amd64-only by
                                                       construction -- the
                                                       batched MoE dispatch)
    ./engine/model      20 run,  20 pass,   0 skip, 0 fail
    ./engine/model MLA  22 run,  22 pass,   0 skip, 0 fail

**TestGreedyMatchesLlamaCpp ran on that host for the first time**, teacher-forced
against llama-completion b10825 over seven models: Llama-3.2-1B, Qwen2-VL-2B,
Qwen3-MoE-4x0.6B and SmolLM2-360M at **20 of 20 positions**, gemma-2b 17 of 17,
and the two divergences are ties of **0.182** and **0.191** -- well inside
tieMargin. `TestEveryModelRunsGenerated` ran Forward, Prefill and ForwardBatch
on **8 models** of generated arm64 code.

★★★ **AND METAL WENT FROM COMPILED TO EXECUTED, WHICH IS THE CLAIM THIS ENTRY
EXISTS TO STOP ANYONE MAKING WITHOUT A RUN.** MLA had been gated on CUDA and
Vulkan and never once executed on Metal, because the only Metal host had no
container it could open. It does now:

    synth-deepseek-dense   **3 of 3 blocks on metal**   logit NMSE 3.887e-12
    synth-deepseek         1 of 3 (2 declined BY NAME: the V3 sigmoid router)
    synth-deepseek-lite    1 of 3 (2 declined on budget)
    the seam moves         2 on the card -> 3 on the host -> 3 back, 3.024e-12
    poisoned buffers       NaN in every fresh buffer, 2.092e-12 (RULE 13)

**And the gate DISCRIMINATES there, which is the half that matters**: all three
violations fail on Metal at logit NMSE **2.1 to 2.7** against a clean arm's
1e-12 -- swapping the absorb banks, writing the value head-major in the row, and
an off-by-one in the per-head sheet.

★★★ **AND DOING IT FOUND A GATE THAT HAD RUN ON NO HOST AT ALL, WHICH IS THIS
RULE AND RULE 10 IN ONE FILE.** `jit/cpu/quantacta64_test.go` is
`//go:build arm64`, so on the Linux box `TestQuantActArgmaxA64Disassembles` and
`TestA64QuantActArgmaxEncodings` **do not compile** -- `go test -run` prints
"no tests to run" and exits 0 -- and over ssh the Mac hands a login PATH with no
`llvm-mc` and no `llvm-objdump`, so they **skipped** there. Never executed
anywhere, green both times, for the two kernels that replaced the quantizer and
the argmax.

**Both tools were already installed on that Mac** -- `llvm-objdump` in the Xcode
command line tools, `llvm-mc` under Homebrew -- so this was RULE 11's `spirv-val`
exactly: a tool logged as absent while sitting in a path. The arm64 runner
sets the remote PATH now, and they pass: **1,277 instructions round-tripped
clean** (143 wide + 1068 narrow + 66 argmax) and **8 NEON encodings** --
`FCMGT4s`, `FCMEQ4s`, `SMIN4s`, `SMINV` -- regenerated from a real clang and
matched.

★ **SO THE TWO HOSTS COVER DIFFERENT HALVES AND NEITHER HALF IS EMPTY NOW.** The
121 `jit/cpu` skips are all tooling -- no GNU objdump, no llvm-mc -- and the
ENCODING gates they guard do run on Linux under clang-18. Linux pins what the
bytes decode to; the M4 pins what they do when executed. A gate that skips on
both is the one to hunt, and the way to find it is to diff the skip lists rather
than to read either one.

★★ **AND THE GGUFs CAME BACK TO THAT BOX ON THE USER'S DECISION,
WHICH IS WHAT MADE THE ORACLE ROW POSSIBLE.** They had been deleted
earlier -- "a machine that exists to run the engine has no use for 13 GB of
the converter's input" -- and the cost was precise: `llama-completion` reads
GGUF, so `TestGreedyMatchesLlamaCpp` skipped on the one host where arm64 decode
had ever been wrong. The user's call was *"make space on the mac copy some gguf
and run llamaccp"*. **The stale containers WERE the space**: all 29 were v22
against a v23 engine, refused by name by `model.Open`, 15 GB of files no binary
in the tree could read. Deleting them and pushing 6.6 GB of GGUF over the direct
link cost 61 seconds at 104 MB/s -- and the containers regenerate themselves,
because `modelFiles()` prefers a GGUF and `jlmOf` converts it ON the Mac, at
whatever version that binary writes. **A host that keeps the converter's input
cannot go stale; a host that keeps only containers goes stale at every version
bump.** That is the argument the earlier deletion did not have.

★ **AND A PERF ROW IS NOT A CORRECTNESS SIGNAL, WHICH IS WHY THE BOARD DID NOT
CATCH IT EITHER.** `gemma-2b CPU` measured 0.8454 with the engine emitting NaN
after two tokens and 0.8463 once fixed. A collapsed model does the same
arithmetic at the same speed. **Never read a ratio as evidence that the engine
computed the model.**

★★★ **AND THE SAME SHAPE WAS LIVE ON amd64: jitllm SIGILLed ON ANY PRE-AVX2
x86-64 HOST WITH EVERY GATE GREEN (Atom C3558, Goldmont, two identical boxes).
CLOSED BY GENERATING FOR IT: THE SSE TIER.** Such
a host now runs every model on legacy-encoded 128-bit kernels, and a host below
that tier is refused by name. `docs/engineering-history/cpu-kernels.md`, "THE SSE
TIER", has the crash, the integration and the Atom's numbers.

    tier       the probe picks it: sse2+sse3+ssse3+sse4.1 and no usable avx2/fma/f16c
    kernels    cpu.EmittersFor(TierSSE) -- every field generated, none pending
               (cpu.SSEPending); each declares its ISA in a leading NOP and
               Map refuses a kernel whose declaration the host does not cover
    refusal    below SSE4.1+SSSE3, cpu.Baseline names the missing extensions and
               model.Open refuses through nn.HostRefusal -- never a SIGILL
    forced     cpu.ForceTierForTest(cpu.TierSSE) (build tag jitllmtest), BEFORE
               NewJIT/Open, asserting MappedByTier()[TierAVX2] did not move

What the crash taught, and what still binds:

- **A probe wired to nothing is worse than no probe.** `jitllm hardware` printed
  the truth on the crashing host; `SupportedNative` answered true for F32/F16
  without asking it. The tier is decided once (`cpu.HostTier`) and every emitter
  is reached through that tier's table -- a condition consulted at each call
  site is one that gets consulted at most of them.
- **Mapping proves nothing about executing.** The inventory gate that passed on
  the Atom was `Map` then `Close`. An SSE kernel gate CALLS the kernel, and runs
  `requireSSEKernel` (the objdump VEX-leak gate) on every kernel at every shape:
  a VEX prefix faults on such a host at any vector length.
- **Run the WHOLE suite on the SSE host, not a filtered slice.** The first
  filtered run hid 29 AVX2 gates failing there; an AVX2-tier gate on an SSE host
  now skips naming the refusal (`mustMap`) or follows the host tier
  (`cpu.Native()`), and the model gates run natively on the Atom:

    model.test -test.run 'TestEveryModelRunsSSE|TestGreedyMatchesLlamaCpp|TestBatchMatchesForward|TestPrefillMatchesForward|TestEveryModelRunsGenerated|TestHybridRuns|TestTowerEncodes'

- **The SSE and AVX2 tiers are two transcriptions, not one.** No FMA and a
  4-lane reduction put their logits a few ulps apart, and the KV width defaults
  differently per tier (f16 on AVX2, f32 on SSE -- the software half convert
  makes the f16 kernel ~4x the code). A comparison between them pins the KV
  width and holds the SSE arm to a same-process control, not a constant
  (`TestEveryModelRunsSSE`). Its verdict is persistence, as
  `TestKVF16IsSelectedAndRuns`'s is: more than a quarter of the positions past
  the band, three in a row or the last fail, and a lone excursion is held to
  the cap. Qwen3-MOE-4x0.6B read 1.667 at position 10 against a 1.569 band
  (greedy 12 of 12, the same on the laptop and the V100 box) and 0.94 at
  position 11. Its router logits run to 30-170, and the AVX2 tier's own
  f16-KV control moves them 2-8 per block and flips the top-2 selection at
  half the positions. At position 10 the SSE tier flips one selection, block
  13's, where the AVX2 margin is 5.09 and the router moved 3.33 (3.23 under the
  control): a routing decision taken the other way by drift both controls
  have, not a near-tie and not a transcription defect. The last block's FFN
  norm bent 10% on the SSE arm reads 1.87-3.07 at every one of 23 positions
  and fails; bent 5% at block 0 it reads 1.44 and passes under either rule.

## Native narrow packing: the d plane, MXFP4's scale, and the formats left widened

**★ NATIVE NARROW PACKING WAS ON THAT LIST AND IS NOT ANY MORE, because the
prediction it was refused on measured the wrong currency.** "Bytes-vs-ops loss"
assumes operations are what a decode token spends. They are not: the same 112
block weights of Llama-3.2-1B measure **47.95 GiB/s in GGUF layout and 48.47
packed**, both against this box's 55.7 GiB/s wall, so the two layouts run at the
SAME rate and time tracks bytes alone. The packed kernel really is 1.4-1.8x
faster per byte -- 165 against 94 GiB/s on a Q4_K ffn_up -- but only while the
weight is in cache, which during decode it never is. A benchmark that calls one
matvec in a loop measures the kernel; a token measures the bus.

**Q6_K LANDED IN container v5: a 4-bit primary plane and a 2-bit
secondary one, +31.4% bytes over its GGUF down to +1.0%.** The head of every
tied model on this box is Q6_K, so that is 27% of a dense model's bytes. The
still-widened format is a NUMBER, printed by
`oracle.TestPackedPayloadIsNativeWidth`, not a silence -- **ONE format is left
and it is Q3_K**:

| format | payload | vs its GGUF | why not native |
|---|---|---|---|
| **Q4_0, Q5_0, Q8_0** | native | **+0.0%** | **done -- container v14** |
| **Q5_1** | native | **+0.0%** | **done -- its minimum is the d word's high half (TypeQ51)** |
| **MXFP4** | native | **+0.0%** | **done -- container v24, see below** |
| Q6_K | **6 bits** | **+1.0%** | done |
| **Q5_K** | **5 bits** | **+0.0%** | **done -- payload v15, scales v27** |
| **Q4_K** | native | **+0.0%** | **done -- container v27: the SC plane is the GGUF's 12 bytes as a 96-bit stream (kernels.ScStream)** |
| Q3_K | 4 bits | +34.5% | 3 bits needs a 2-bit PRIMARY plane, a third emitter shape |

★★★ **AND MXFP4's WIDENING WAS THE SCALE PLANE, NOT THE PAYLOAD -- A BLOCK
SCALE THAT IS ALWAYS A POWER OF TWO WAS BEING STORED AS AN f16. CLOSED
in container v24.** MXFP4's scale is an **E8M0 byte**: an exponent and
nothing else. `PackedWords` stored one f16 per 32 weights beside a 16-byte
payload -- 5.88% of every MXFP4 tensor holding four bits of information -- so
gpt-oss-20b, which is 83.93% MXFP4, converted at **+4.9% over its GGUF**. It
converts at **11.28 GiB against a GGUF of 11.28 GiB now: EXACTLY, to the byte.**

★ **THE MECHANISM IS THAT THE BYTE *IS* THE f32, WITH A SHIFT AND NO TABLE.**
`e8m0Store(e) = e - 1`, and `Float32frombits(uint32(b) << 23)` is the scale --
so every reader gained two instructions and no arithmetic:
`VPMOVZXBD`+`VPSLLD` on AVX2, `PMOVZXBD`+`PSLLD` on SSE, `UXTL`+`SHL` on arm64,
and `Bitcast(F32, Shl(And(dw, 0xFF), 23))` in the device IR.

★ **AND THE d PLANE IS GENERALISED RATHER THAN SPECIAL-CASED, WHICH IS WHAT
MADE IT SIX EMITTERS AND NOT SIXTEEN.** `kernels.DSlots(q)` is how many ROWS
share one 32-bit d word -- **1** (a k-quant's scale beside its minimum), **2**
(the v14 narrow f16) or **4** (an E8M0 byte) -- and `DIndex` returns
`(idx, slot)` for all three. The parity lives on the row index, so the slot is
known at EMIT time exactly as v14's was, and `jit/cpu.DRowBytes(t)` is
`4/DSlotsOf(t)` everywhere a cursor advances.

★ **THE ONE TRAP, AND THE GATE THAT CAUGHT IT: THE DEVICE IR MUST SHIFT, NOT
DIVIDE.** Writing the slot extraction as `Div`/`Rem` is arithmetically
identical and CHANGED the `lowertest` golden `matvec/Q8_0/split4` -- DSlots is
always a power of two, so `Shr`/`And` is the form that leaves every other
format's emitted code byte-identical. RULE 11b: the golden was re-derived, not
blessed.

★★ **AND IT WAS A STORED ZERO, WORTH 8.55% OF A gemma-2b CONTAINER. CLOSED
in container v14.** `PackedWords` wrote ONE d WORD per super-block per
row -- `uint32(dBits) | uint32(dminBits)<<16` -- and `perSuper` is 8 or 16 for
the k-quants but **ONE** for Q4_0 and Q8_0. So those two stored a four-byte word
per 32 weights against a 16- or 32-byte payload, and the top half of every one
was a minimum they do not have:

    Q4_K   4 B per 144 B block    2.8%
    Q8_0   4 B per  36 B block   11.1%
    Q4_0   4 B per  20 B block   20.0%

★ **AND THE ONLY MODEL ON THIS BOX MADE OF THEM IS gemma-2b, WHICH IS THE WHOLE
SIZE OF THE LEVER AND MUST BE QUOTED WITH IT.** Every other model here is
Q4_K_M -- Q4_K plus Q6_K, both perSuper > 1 -- so `DIndex` returns exactly the
v2 layout for them and not one byte moves. Measured FILE TO FILE, the same GGUF
converted by the commit either side:

    gemma-2b.jlm   1748.38 MiB -> 1598.98 MiB    -149.40 MiB, -8.55%
    pages          65.62 MiB a block -> 59.06    -10.0%
    dense region   594,673,664 B -> 561,889,280  -31.3 MiB
    its GGUF is 1600.7 MiB, so the container went +9.2% -> -0.1%

`oracle.TestPackedPayloadIsNativeWidth` prints the per-format half of that and
now reads +0.0% for both.

★ **AND THE RATE FOLLOWED THE BYTES ALMOST EXACTLY, WHICH IS THE CLAIM AND ITS
OWN CHECK.** Decode is bandwidth-bound, so 8.55% fewer bytes predicts ~8.55%
more tokens, and it is the two quantities agreeing -- measured separately, one
from a file size and one from a stopwatch -- that makes this more than a timing
result. gemma-2b, **M4 Metal decode**, `scripts/vs-llamacpp.sh gpu`, n=320,
ABBA, RULE 2's two passes:

| | ratio | IQR/median | n | jitllm | llama.cpp |
|---|---|---|---|---|---|
| was (v13) | 0.865 | | | 48.95-49.20 | 56.62 |
| pass 1 | **0.9334** | 0.003 | 12 | 52.84-53.06 | 56.48-56.85 |
| pass 2 | **0.9313** | 0.003 | 12 | 52.71-52.97 | 56.39-56.84 |

**+8.2% of jitllm's own rate against +8.55% predicted from the container**, and
the two passes agree to 0.21 points. llama.cpp's arm does not move, which is
what it should do -- it reads the GGUF and nothing here touched it. No A/A
self-control was run: the effect is 6.8 points against a 0.3% dispersion and it
is PREDICTED by an independently measured byte count, which is stronger evidence
than a same-binary control would be.

★ **AND gemma's METAL ROW MOVED AGAIN, FROM A CORRECTNESS FIX RATHER THAN A
PERF ONE.** 0.9334/0.9313 -> 0.9411/0.9496 when `backend.mtlSession.Sync` was
made to WAIT. It was `Batch.Commit`, which stopped waiting when submission went
asynchronous, so on Metal `Sync` returned before the GPU had started -- and
`tuneSplit` and `choose` both time a loop of launches and then call it. Both were
timing the ENQUEUE; mvbench, same shape, printed **106794% of the read wall**,
and RULE 1's "above the roofline is a harness bug" is the only reason it was
visible at all. `finish()` still does NOT wait, which is where the ~6% of
asynchronous submission lives: command buffers on one queue run in order, and the
only consumer that needs results on the host is `Read`, which waits.

★★ **AND THE msl OVERRIDE IN `tuneSplit` WAS REMOVED WHEN ITS MEASUREMENT WAS RE-TAKEN.** It
said the measured split was 6.5% slower than the fitted one on Metal and
concluded that a load-time probe measures the wrong regime under unified memory.
It was not measuring the wrong regime; it was not measuring. Re-taken with Sync
fixed, `verify -ab split`, n=15: **1.0031x, IQR/median 0.6% -- no difference
worth the code.** One code path now instead of two, and the tuner is honest.

★ **THE PRICE IS DISPERSION, AND IT IS A LEAD.** The two passes agreed to 0.21
points before and to 0.85 after, because each process now MEASURES its splits
instead of reading them off a curve, and the answers differ run to run. That
0.85 points is the tuner's own variance -- so pinning the best split set may be
worth more than tuning it. Unmeasured.

★ **AND THE LINUX ROWS WERE NOT RE-TAKEN, ON PURPOSE.** The box sat at 1558 MHz
of a 3792 idle peak all afternoon, which is RULE 2's sustained clamp and its
own verdict is to stop measuring rather than raise the round count. Nothing on
Linux is Q4_0 or Q8_0 except gemma-2b anyway.

★★★ **AND THE REJECTION IN `pack.go` WAS MEASURED ON THE FORMATS THAT DO NOT
NEED IT.** It read *"Q3_K and Q6_K came out at 28 and 31 KB against a 16 KB
budget"* -- true, and both are perSuper 16, where the d plane is 0.2% of a
tensor and there is nothing to win. The two formats where it is worth 11.1% and
5.9% have the SMALLEST kernels in the tree:

    Q4_0   fused   628 B   tiled  1541 B     budget 16384 B
    Q8_0   fused   660 B   tiled  1573 B
    Q4_K   fused  4973 B   tiled 14396 B

★★ **AND PAIRING ROWS RATHER THAN SUPER-BLOCKS REMOVES THE UNROLL THAT OBJECTION
IS ABOUT, AND MAKES EVERY READER CHEAPER RATHER THAN DEARER.** v1 put two
SUPER-SCALES in one word, so the half to read depended on a runtime loop counter
and every host emitter had to unroll its outer loop by two to know it -- which
is what produced the 28 and 31 KB. Pairing ROWS puts the parity on the row
index, which every kernel walks in groups of four, eight or sixteen from an even
base, so the half is known at EMIT time and no unroll exists:

    amd64   one VCVTPH2PS from m128, against a 32-byte load + VPSHUFB + VPERMQ
    arm64   one LDRd + one FCVTL, against four (LDRh, FCVTL, INSs)
    device  a runtime shift by (row&1)*16, which costs nothing there -- and in
            mma.go every rowImm is EVEN, so the parity is the tile base's alone
            and the per-slot displacement stays an immediate at half its width

`kernels.DIndex` is the one function the packer and all five readers derive
from. **THE READERS ARE FIVE AND THE DEVICE HAS TWO OF THEM:**
`internal/oracle/packed.go` (the oracle), `jit/gpu/kernels/matvec.go`, `jit/gpu/kernels/mma.go`, the
amd64 emitters and `jit/cpu/packed_a64.go`.

★★ **AND MISSING mma.go IS WHAT SANK THE FIRST ATTEMPT, AT A LINE WHOSE OWN
COMMENT SAYS IT HAD HAPPENED BEFORE.** The change reached `matvec.go` and not
`mma.go`; CUDA takes the MMA path and Vulkan does not, so CUDA produced `+Inf`
and Vulkan was exact at 1.575e-07 on the same graph -- and the first diagnosis,
a ptx codegen bug, was wrong. mma.go's comment already recorded the v2 migration
making the identical mistake in the identical place: *"the v2 change reached the
matvec and not this, so decode was right and PREFILL was wrong... RULE 8a -- a
change has to reach both tiers, and the device has two kernels that read this
array."* Its `dRow` returning `(off, shift)` was that migration's dead plumbing,
and it is live again. **The gate is a device run of a Q8_0 model on CUDA** --
`TestTowerOnDeviceMatchesHost`, SmolVLM's tower, now 2.174e-06 on cuda.

★★ **AND THAT GATE PASSED A BACKEND EMITTING Inf.** It compared `nmse > 5e-5`
and nothing else, and `NaN > 5e-5` is FALSE -- so CUDA reported `NMSE NaN,
max|d| +Inf` and the subtest went GREEN. RULE 10 names the shape ("a degenerate
oracle is a failure, not a pass") and it was live in the gate guarding three
backends. It checks for non-finite FIRST now.

★ **AND THE WIDE PACKED KERNEL TURNED OUT TO BE UNREACHABLE ON EVERY FORMAT,
WHICH IS WHY IT WAS NOT CONVERTED.** `MatVecPacked` prefers the fused kernel at
`nrows%16 == 0`; every `nrows%32 == 0` satisfies that; and every format with a
wide kernel also has a fused one. So `EmitPackedMatVecWide` has no call site for
any format now. It REFUSES the narrow formats rather than reading their d
plane four bytes at a time -- an unreachable kernel that is silently wrong is
worse than one that is absent -- and whether it should exist at all is a
separate question nobody has asked.

**The two halves combine DIFFERENTLY on host and device and both are forced.**
VPDPBUSD has no centering and a dot is linear, so the host adds the planes as
two products into one accumulator with the hi mask pre-shifted to 0x30 -- 3<<4
is 48, still an unsigned byte, which is what kills the "third accumulator array"
the old refusal assumed. The device's `center()` is `(v + cAdd) ^ cXor`, affine
per byte, so centering each plane separately subtracts the offset twice: both
device kernels XOR the planes into the full weight first and center once.

## The vision tower's pool regions, and the packed GEMM that removed them

- **★ THE VISION TOWER'S 148,755 POOL REGIONS ARE A REGRESSION WITH A COMMIT, NOT A
  DESIGN, AND WHOLE-LAYER FUSION IS THE WRONG LEVER FOR THEM.** Both halves of
  the sentence that stood here were overturned by measurement
  (`model.TestTowerFusionCeiling`).

  ★ **THE REGRESSION HAS A COMMIT AND A PRICE NOBODY TOOK.** `docs/perf/
  current.md` records the encode END TO END at **477.178 ms** (IQR 0.005,
  n=6) and concludes *"the tower is now bound by its kernels rather
  than by its scheduling"*. The NEXT DAY, `3ce7146` ("The vision tower reads a
  container too") put `w.packed == nil &&` in front of `nn.MatMul` -- correctly,
  because nn.MatMul walks GGUF blocks and a packed payload would be an
  out-of-range read -- and its own message says the consequence out loud:
  *"there is no packed GEMM yet, so a container prefills through the matvec
  loop"*. The cost of that loop was never measured. It is **885-910 ms as measured
  here, 1.9x the 477**, and the doc's "bound by its kernels" became false in one day
  with nothing to catch it.

  ★ **AND THE REGIONS ARE THE MATMUL'S ROWS, NOT THE LAYER'S OPS.** The tower's
  OWN layer structure is **12 regions per layer** -- 2 LayerNorm sweeps, 6
  addBiasRows, 2 residual axpys, 1 activation, and attention, which is ONE
  region over HEADS. That is 144 of 148,755, or **0.097%**. Every other region
  is one matvec ROW at **2.02 dispatches each** (the activation quantize plus
  the kernel): 6 matrices x 1024 patches x 12 blocks, plus 1,024 F32 patch-embed
  rows and 128 projector rows.

      merging q/k/v into one region            0.00% of an encode
      + folding the elementwise epilogues      0.01%
      a BATCHED PACKED GEMM                    the whole dispatch budget

  ★ **AND THE PRICE MUST BE THE IN-SITU ONE, WHICH IS THIS TREE'S OWN STANDING
  WARNING.** `dispatchCost` times 500 EMPTY regions; `splitCost` measured 1.596
  us in situ against 2.697 idle, a factor of **0.592**, because *"dispatch
  overlaps with computation and an idle round trip has nothing to hide behind"*.
  `docs/perf/current.md` generalises it: when a lever is priced as N x (the
  cost of one thing measured alone), find out what that thing costs IN SITU
  before believing the product.

      one region round trip     2.74-3.17 us idle   1.62 us in situ
      dispatch per encode       241 ms in situ .. 407 ms idle
      the encode itself         841-1044 ms
      dispatch's share          **26%..45%** -- a BAND, and NOT "most of it"

  So the majority of a tower encode is arithmetic, the recoverable dispatch is
  a quarter to a half, and essentially ALL of it is reachable by restoring a
  batched path for the packed layout -- **none of it by fusing layer ops.**

  ★★ **AND IT WAS BUILT: `nn.MatMulPacked`, AND IT DELIVERS.**
  One dispatch over row groups with the TOKEN LOOP INSIDE IT, in place of one
  dispatch per token. No new emitter -- the same fused kernel serves every
  token, because the packed column-major layout already suits a batch. ABBA in
  one process, n=24, A/A self-control FIRST, on the tower's own shapes:

      shape            A/A control          batch / row loop      IQR/med
      q/k/v/o 768x768  0.9976 (IQR 0.049)   **2.43x**             0.059
      fc1    3072x768  1.0276 (IQR 0.094)   **1.98x**             0.073
      fc2    768x3072  1.0095 (IQR 0.125)   1.89x                 0.043   (the
                       A/A is OVER the gate on this row -- treat 1.89 as soft)

      regions per encode   148,755 -> **1,317**      12,396/layer -> 110
      dispatch per encode  241-407 ms -> **2-4 ms**  45% of the encode -> 0-1%
      the encode itself    841-1044 ms -> **522-629 ms**

  That is most of the way back to the 477 ms the tower measured before
  `3ce7146`, and the remaining gap is arithmetic rather than scheduling -- so
  the evidence file's retired sentence is true again.

  ★★ **AND IT WAS NEVER A TOWER BUG: `engine/model/prefill.go` CARRIES THE SAME GUARD,
  SO EVERY CONTAINER MODEL PREFILLED THROUGH THE ROW LOOP.** The comment there
  named the debt in its own words -- *"the GEMM for it is the next kernel owed
  (RULE 8)"*. tinyllama q3_K_M, a 512-token prompt, six P-cores:

      regions   159,810 -> **7,042**   (14.2 per layer per token -> 0.6)
      rate      167.2 -> 166-185 tok/s  **FLAT, and that is the honest result**

  **Coordination improved 22.7x and TIME DID NOT MOVE**, which is RULE 6's
  checklist working rather than a disappointment: prefill's regions are large
  enough that dispatch was never its bottleneck, where a tower's 1024-patch
  matmuls are dispatch-heavy and gained 1.5x. The change stays because nothing
  regressed, not because a number justified it -- and the prefill rows in this
  file are all PRE-CONTAINER and still need re-taking.

  ★★ **AND THE REMAINING GAP IS A KERNEL THAT MUST READ jlm, NOT GGUF.** Stated
  by the user: *"we need to elliminate gguf dependency, we use jlm
  format"*. `EmitGEMM` tiles TOKENS and `nn.MatMul` reads GGUF blocks, so on a
  container it is reachable only by F32/F16 tensors -- the engine's
  best-developed compute path is nearly dead code, which is the stale half the
  container migration left behind. Priced on the tower's own shapes, n=24, A/A
  first, the GGUF GEMM against the packed batch:

      q/k/v/o 768x768   packed 325 Gmac/s   GEMM 321   1.07x
      fc1    3072x768          313               502   **1.60x**
      fc2    768x3072          244               451   **1.91x**

  **Unpacking a container weight into GGUF layout to borrow the existing one was
  built, measured and DELETED** -- it worked, and it reaches the win by
  entrenching the dependency the container exists to remove.

  ★★ **AND THE TILED PACKED KERNEL WAS BUILT INSTEAD, AND IT BEATS
  THE GGUF GEMM IT REPLACES.** `cpu.EmitPackedMatMulTiled` holds four tokens'
  accumulators and issues four dots against ONE payload load, reading the
  CONTAINER layout. Same shapes, same harness, A/A first:

      shape            was (per-token)  batch    TILED    GGUF GEMM yardstick
      q/k/v/o 768x768      ~134 Gmac/s    325   **454**   336   (tiled +35%)
      fc1    3072x768                     313   **465**   491   (soft: A/A
                                                                 IQR 0.155)
      fc2    768x3072                     244   **389**   442

      SmolVLM tower encode   841-1044 ms -> 582-613 -> **395-428 ms**

  **That is BELOW the 477.178 ms measured before `3ce7146`**, which was the tower's best
  ever and was taken on the GGUF GEMM -- so the container layout is now the
  faster one, and `nn.MatMul` has nothing left to offer the tower.

  ★ **FOUR TOKENS IS THE REGISTER FILE, NOT A TUNING CHOICE**: the mask, two
  accumulators per token, one payload, one broadcast per token and two epilogue
  temporaries is 1+8+1+4+2 = 16 exactly. Five needs nineteen. The epilogue
  reuses the activation registers because they are dead by then.

  ★ **AND IT COST ONE BUG WORTH RECORDING: `rowLoop` OWNS RBX.** Loading the
  correction constant from `Scr` inside the body clobbered the row counter, so
  DEC/JNZ spun on a pointer while R15 walked past every buffer -- it faulted at
  `0x18398400000`, which reads like a wild pointer rather than a loop bound.
  `EmitPackedMatVecFused` reloads RBX in exactly one place and carries a comment
  saying why; the comment was right and I did not read it first.

  ★★ **"Q8_0 ONLY" AND "PREFILL GETS NOTHING FROM THIS YET" ARE BOTH STALE, AND
  THE SECOND ONE SENT ME LOOKING IN THE WRONG PLACE. CORRECTED BY
  ASKING THE EMITTER.** `MaxTiledTokens` + `EmitPackedMatMulTiled` at k=2048,
  512 rows:

      Q4_0  tok=4   1513 B      Q3_K  tok=3  13482 B      Q5_K  tok=4  12439 B
      Q8_0  tok=4   1545 B      Q4_K  tok=4  14396 B
      Q6_K  tok=3  **REFUSED: 20144 bytes, over the 16384 budget**

  ★ **AND "Q6_K IS THE ONE GENUINE HOLE" IS ITSELF STALE NOW, WHICH IS WHY A
  CENSUS MUST ASK THE PATH THAT SHIPS AND NOT THE WIDEST TILE.** The table above
  asks `MaxTiledTokens` and emits at THAT width; `tiledFor` does not -- it
  NARROWS until something fits, and its own comment says so ("two tokens still
  halves the payload traffic"). Re-run through the shipping selection, at all
  four of Llama-3.2-1B's real shapes (k=2048 rows=2048/512/8192, k=8192
  rows=2048), the width is the same at every shape and **all six formats tile**:

      Q4_0  4   1,513 B      Q3_K  3  13,482 B      Q5_K  4  12,439 B
      Q8_0  4   1,545 B      Q4_K  4  14,396 B      Q6_K  **2**  14,752 B

  So Q6_K ships a TWO-token tile, not none, and the sentence below about the
  head dropping to the per-token matvec loop is wrong: it drops to half the
  tile, which is a different and much smaller claim.

  The original paragraph, kept because it is the reasoning that was true when
  written: Q8_0 is one sub-block per super-block with no nibble split and no
  sub-block scale array; the 4-bit formats need lo and hi broadcast separately
  (a two-phase loop) and the k-quants add a per-sub-block scale, so each is its
  own emitter. arm64 declines rather than failing to compile, and NEON's
  thirty-two V registers would hold a WIDER tile than four.

  `nn.TestMatMulPackedMatchesTheRowLoopExactly` is BIT EQUALITY, not an NMSE
  bound, because the arithmetic is unchanged by construction -- four quant types
  x five shapes, and it fails against a token reading activation 0 and against a
  token writing output row 0. It does NOT catch a shared per-worker scratch at
  these shapes, which is stated rather than implied.

  ★ **AND HALF THE REGIONS ARE A CACHE THE LOOP DEFEATS ITSELF.** One of the two
  dispatches per row is the activation quantize, which `nn` caches by generation
  -- and `vision.go`'s loop calls `s.jit.NewInput()` per ROW, so q, k and v each
  re-quantize the SAME normed row. That is a cheaper half-lever than the GEMM and
  it is independent of it.

  ★ **AND IT IS HOST-ONLY.** Where a device takes the tower the whole prefix goes
  out in ONE submission per block, so 148,755 is a CPU-only number. The "906
  dispatch regions per MoE token" entry on the shelved-levers list is a genuine
  whole-layer-fusion case and is untouched by any of this.

## Prefill and the E-cores on the container path

- **Prefill never gets its E-cores.** RULE 5 records +34% for 6P+6E in prefill,
  and no code switches coreset per phase: `nn.NewJIT` builds one pool from
  `sched.DecodeCores()` (six P-cores) and prefill reuses it. `SetParticipants`
  can only narrow, never widen past the pool's size.

  ★★ **AND THE SHAPE THIS ENTRY PRESCRIBED DID NOT SURVIVE, BECAUSE RULE 5's
  PINNING REMOVAL RETIRED IT AND NOBODY RE-READ IT.** It said: build
  the pool at the widest phase and NARROW per phase. That worked when a worker
  was bound to one CPU -- narrowing to six then meant *those* six. It does not
  now. `sched.Pool.worker` carries "NO LockOSThread AND NO sched_setaffinity,
  AND DELETING THEM IS THE FIX FOR A 100x BUG", so **a participant count selects
  HOW MANY workers drain a region and says nothing about WHICH cores they run
  on** -- the Go scheduler places them anywhere in the process mask. A pool
  built at 6P+6E and narrowed to six for decode would hand decode an arbitrary
  mixture of P and E cores, which is the configuration RULE 5 measures at
  **0.864x**. The prescription would have made decode worse to make prefill
  better, and read as free.

  ★★★★ **AND THE ROW WAS TAKEN TWICE, AND THERE IS NOTHING TO COLLECT: THE
  E-CORES ARE NEUTRAL IN BOTH PHASES. BOTH OF RULE 5's FIGURES ARE REFUTED ON
  THE CONTAINER PATH.** The mechanism needed no building --
  `JITLLM_CORESET=pe` already makes `nn.NewJIT` take `CoreSet("pe")`. One
  process per sample, ABBA, tinyllama q3_K_M, a 450-token prompt and 48 decoded
  tokens, on a box at 2.7% busy and 58 C.

  ★ **THE FIRST PASS VARIED TWO THINGS AND IS KEPT ONLY AS THE WEAKER ARM.**
  It moved the process MASK and the pool WIDTH together (6 CPUs/6 workers
  against 12/12), which cannot separate "more cores" from "a wider pool":

      prefill  1.0130 / 0.9969      decode  1.0143 / 0.9880    A/A 1.0019 / 0.9988

  ★★ **THE SECOND HOLDS THE MASK AT 12 CPUs AND VARIES ONLY THE POOL, WHICH
  IS THE QUESTION THE GOAL ACTUALLY ASKS: GIVEN E-CORES, DOES THE ENGINE USE
  THEM, AND SHOULD IT?** `jitllm hardware` under that mask prints
  "coreset p, 6 core(s)" -- **the default leaves six E-cores idle** -- against
  arm B's twelve:

                      pass 1    pass 2    A/A bias
      prefill         1.0024    0.9952    1.0275
      decode          1.0334    1.0140    1.0162

  **Prefill is dead neutral and sits BELOW its own A/A bias**; decode's 1.4-3.3%
  is the same order as a 1.6% bias and is not a claim. Every arm gates at
  IQR/median 0.024-0.037.

  **Both figures it was hunting are an order of magnitude outside all of that:
  +34% for prefill does not happen, and neither does decode's 0.864x collapse.**

  ★ **SO THE DEFAULT STAYS SIX WORKERS, AND NOW BY MEASUREMENT RATHER THAN BY
  INERTIA.** RULE 6 admits a change unless it regresses, and a wider pool does
  not regress the RATE -- but rate is not the only axis on the checklist.
  Twelve spinning workers for the same tokens is double the execution resource
  for nothing, and this file already prices a spinning worker at 60e9
  instructions of WAITING and names a foreign tenant on one pinned core as a
  75% loss. Taking cores the engine cannot use is how it becomes that tenant
  for somebody else.

  ★ **AND THE ENGINE IS WHY, WHICH MAKES THIS A CLOSED QUESTION RATHER THAN A
  MISSING LEVER.** `SetParticipants` is tuned PER REGION, so a 12-wide pool
  already narrows wherever narrow is better -- the pool's WIDTH stopped being
  the thing that decides how many cores drain a matvec. The entry's premise,
  that one coreset for the process forces one width on both phases, was true
  when the workers were pinned and is not now.

  ★ **THE PRESCRIPTION IT CARRIED THEREFORE FAILS TWICE OVER.** It said
  build the pool wide and NARROW per phase, which worked while a worker was
  bound to one CPU; RULE 5's pinning removal retired that and nobody re-read
  this entry. And the thing it was meant to buy is not there anyway.

  ★ **WHAT THIS DOES NOT ESTABLISH.** One model, one prompt length, n=8 per
  pass. RULE 5's +34% was taken on the PRE-CONTAINER engine, which this file
  already flags for the prefill rows generally ("the prefill rows in this file
  are all PRE-CONTAINER and still need re-taking"), and `nn.MatMulPacked` and
  the tiled packed kernel both landed between. So the honest reading is that
  the figure does not survive on the path that ships, not that it was wrong
  when it was taken.

## Profiling the subsystems nobody had profiled: the sampler and the MoE expert loop

- **Profile the subsystems nobody has profiled.** The vision tower ran its
  attention on ONE core while five idled -- 1.75 of 6, 29% utilisation -- behind
  five green RULE 8 gates, because correctness and utilisation are different
  questions and only the first was ever asked. One `go tool pprof` found it and
  the fix was 2.69x. Still unprofiled: the vision tower's projector, prefill's
  pack phase, and the device submission path.

  ★★★★ **AND THE SAMPLER WAS THE LARGEST OF THESE BY FAR: IT COST AS MUCH AS
  THE WHOLE FORWARD PASS, AND NO BOARD ROW COULD EVER HAVE SEEN IT.**
  `engine/model/sample.go` sorted the ENTIRE vocabulary with `sort.Slice` every token
  -- 128256 entries through a closure, ~2.2M comparisons with interface dispatch
  and reflect-based swaps -- to keep forty. Measured on the in-process dose
  harness, Llama-3.2-1B Q4_K_M, six P-cores, ABBA in one process, A/A
  self-control FIRST, two passes:

      OP    A/A self-control   1.0119   IQR/median 0.033
      OP    pass 1 / pass 2    228.5x / 216.6x    IQR/median 0.091 / 0.038
      TOKEN A/A self-control   1.0124   IQR/median 0.049
      TOKEN pass 1 / pass 2    **1.920x / 1.910x**  IQR/median 0.029 / 0.077

      generated  17.81 ms/token = 56.16 tok/s
      Go         35.70 ms/token = 28.01 tok/s

  **The two passes agree to 0.01 points and everything gates.** And the arms
  are confirmed configured right by RULE 2's last bullet rather than by the
  dispersion: 56.16 tok/s sits inside the 57.67-59.93 this file already records
  for this model, coreset and length, so the baseline is this box's own
  standing figure and not a mis-set knob.

  ★ **AND THE WHOLE SESSION IS A NULL ON GREEDY DECODE, MEASURED END TO END SO
  NOBODY HUNTS FOR A WIN THAT IS NOT THERE.** HEAD against the session's first
  commit, thirteen commits apart, same harness, A/A first:

      A/A self-control   0.9872   IQR/median 0.037   n=10
      pass 1             0.9912   IQR/median 0.053   n=12
      pass 2             1.0184   IQR/median 0.031   n=10

  **The two passes STRADDLE 1.0 and disagree by 2.7 points**, which is larger
  than anything being claimed, so this resolves nothing smaller than ~3 points
  and is a null rather than a small win. That is what the component rows already
  said -- the rotary table a null, the router sub-noise -- and the useful half
  is the other direction: across a router kernel, a rotary kernel on four tiers
  and a rewritten sampler, **nothing REGRESSED**, which is the only thing RULE 6
  requires.

  ★ **WHY NOTHING CAUGHT IT: EVERY ROW ON EVERY BOARD IS GREEDY.**
  `Sampler.Sample` returns `Greedy(logits)` at `Temp <= 0` before it reads
  another field, and `vs-llamacpp.sh`, `TestFusionCeiling`, the teacher-forced
  diff and `TestGreedyMatchesLlamaCpp` are all completions at temperature zero.
  So the scoreboard is unaffected and always was -- and every INTERACTIVE token
  this engine has ever sampled paid for a full-vocabulary sort. A subsystem
  that no benchmark reaches is not cheap; it is unmeasured, which is the same
  lesson the vision tower's one-core attention taught one level up.

  ★★ **THE MoE EXPERT LOOP IS PROFILED NOW, AND THE DEFICIT IS
  OCCUPANCY RATHER THAN KERNEL QUALITY -- WHICH A RATIO CANNOT SAY AND A PROFILE
  CAN.** `go tool pprof` over the in-process decode probe, six P-cores, three
  models, and the fraction of samples in the GENERATED KERNELS
  (`runtime._ExternalCode`) tracks the fraction of the 55.7 GB/s read wall the
  model actually reaches, to within three points on all three:

    | model | regions/layer | kernel samples | B/token | GB/s | % of wall |
    |---|---|---|---|---|---|
    | Llama-3.2-1B (dense) | 13.6 | **84.0%** | 816,010,912 | 47.6 | **85%** |
    | Qwen3-MoE-4x0.6B | 20.6 | **71.3%** | 562,827,856 | 39.3 | **71%** |
    | olmoe-1b-7b | 56.6 | **69.2%** | 756,937,888 | 32.8 | **59%** |

  **olmoe reads 7% FEWER bytes per token than the dense model and reaches 26
  points less of the wall.** The missing samples are the pool waiting --
  `sched.(*Pool).worker` is 21.9% cumulative on olmoe against 11.3% on the dense
  model, and whole-process CPU is 423% of 600% against 501%.

  ★ **AND IT IS NOT PROPORTIONAL TO REGION COUNT, WHICH IS THE PART THAT SAYS
  WHAT TO BUILD.** olmoe carries 2.7x Qwen3-MoE's regions per layer and sits two
  points BELOW it in occupancy, while both MoEs sit ~13-15 points under the dense
  model. So the cost is not 906 round trips at 1.62 us -- that is only 1.11 ms of
  a 23.07 ms token, **4.8%** -- it is that an expert matvec is too SMALL to
  amortise a barrier, so five cores idle while the straggler finishes. **A
  barrier is priced by what the other five cores do not do during it, not by its
  own duration**, and pricing it the other way is what made this look like a
  4.8% item for a 30-point row.

  ★★★ **AND THE 1.22x PREDICTED FROM THAT SAMPLE FRACTION WAS REFUTED BY THE
  EXPERIMENT IT MOTIVATED, SAME DAY. HALVING THE REGIONS DID NOT REDUCE THE
  SPIN -- IT RAISED IT.** The prediction was "recover dense-like occupancy and
  olmoe goes 43.3 -> 52.7 tok/s". `nn.MatVecPackedMulti` was built to collect
  it: the k experts' gate and up projections share one activation, so they go
  out in ONE pool region, and the SwiGLU over all of them is one more.
  **olmoe's regions per token went 906 -> 442, a 2.05x cut, and occupancy did
  not move**, profiled with both builds alternating in one pass:

    | | regions/token | kernel samples | Pool.await |
    |---|---|---|---|
    | per-expert loop | 906 | 69.71% | 13.98% |
    | batched (pass 1) | **442** | 66.63% | **18.51%** |
    | per-expert loop (pass 2) | 906 | 53.54% | -- |
    | batched (pass 2) | **442** | 53.61% | -- |

  So **the barrier COUNT was not the mechanism**, and "an expert matvec is too
  small to amortise a barrier" -- the sentence above that motivated this -- is
  refuted rather than merely unconfirmed. What is left in the non-kernel third
  is the SERIAL STRETCH BETWEEN regions, which no amount of merging touches:
  the router softmax, the top-k, the ascending-id sort, the hotness counter and
  `ensureExperts` -- of which **the top-k, the sort and the renormalising divide
  later became one generated kernel and it bought nothing measurable**
  (see the NEXT LEVER entry below) -- whose synchronous `pread`s are **6.1% of samples** on their
  own (`Syscall6`, with `jlm.EnsureRanges` beneath it), because `pageIn` skips
  the banks on purpose and a token's experts are fetched on the critical path.
  Workers spin through all of it.

  ★ **AND PART OF THAT I/O IS THE PROBE, WHICH THE TABLE ABOVE MUST BE READ
  WITH.** The harness warms 8 tokens and measures 64, and olmoe has 64 experts
  over 16 layers -- 1024 expert slots -- so it is still faulting new experts
  throughout. A long decode would drive that term toward zero and the 59%-of-wall
  figure up. The row is a LEAD, not a standing.

  **The rate is UNRESOLVED and was not left implied:** the A/B rejected at
  IQR/median 0.194 with three power-clamped samples, against an A/A self-control
  of **1.060** on the same harness -- a 6% bias, larger than anything being
  claimed. The change stays on RULE 6's terms (coordination 2.05x better,
  nothing regressed), which is the same footing `nn.MatMulPacked` was kept on
  when prefill's regions improved 22.7x and time did not move.

  ★★ **AND THE DOWN PROJECTION WAS BATCHED TOO (`nn.MatVecPackedGather`), WHICH
  TOOK olmoe TO 234 REGIONS -- THE DENSE MODEL'S DENSITY -- AND REMOVED 10% OF
  THE KERNEL CPU.** It needs a different contract: gate and up share one
  activation, the down projection reads each expert's OWN SwiGLU output, so the
  gather quantizes k activations in one region and runs k matvecs in one more.
  The k weighted adds into the residual became one region as well, and that one
  is not a region count at all -- `nn.Axpy32JIT` calls its kernel ON THE CALLER
  with no pool dispatch, so those were SERIAL.

    olmoe regions/token   906 -> 442 (gate/up) -> **234** (3.87x), 14.6 a layer
                          against a dense Llama-3.2-1B's 13.6

  Profiled with both builds alternating, TWO passes, absolute seconds:

    | | wall | kernel CPU | Pool.await |
    |---|---|---|---|
    | per-expert loop | 2.32 / 2.23 s | 6.45 / 6.18 s | 1.57 / 1.84 s |
    | batched | 2.24 / 2.36 s | **5.79 / 5.65 s** | 2.17 / 2.29 s |

  **Kernel CPU falls ~10% in both passes** -- real work removed, most of it the
  k redundant quantizations of the shared activation. The wall does not follow,
  and `await` RISES in absolute terms: the workers spend the saving waiting.

  ★★★ **SO THE BINDING CONSTRAINT IS THE SERIAL STRETCH, AND await IS HOW TO
  READ IT.** `Pool.await` is not the barrier at the END of a region -- it is
  workers spinning on `p.seq` for the NEXT one, 250 us before they park. Every
  sample in it is a core burning while the MAIN GOROUTINE runs Go: the router
  softmax, the top-k, the ascending-id sort, the hotness counter, and
  `ensureExperts`, whose synchronous `pread`s are 6.1% of samples on their own.
  (The top-k, the sort and the divide became a kernel later and the
  rate did not move -- the measurement is under NEXT LEVER below.)
  Five workers spinning for the whole of it makes `await/5` a LOWER BOUND on
  serial time: **2.17/5 = 0.43 s of a 2.24 s probe, 19% of the token**, and a
  lower bound because a parked worker is not sampled at all.

  **The rate is UNRESOLVED both times it was asked.** The A/A self-control
  passed at **1.009, IQR/median 0.061** -- so the harness was honest -- and the
  A/B still REJECTED at IQR/median 0.198 with four power-clamped samples (9.9,
  11.1, 12.7 tok/s). Median 1.029 against that control, which is not a claim.
  The change stays on RULE 6's terms: 3.87x the coordination, 10% less kernel
  CPU, nothing regressed.

  ★★ **AND ON arm64 IT IS BUILT AND OFF BY DEFAULT, AS MEASURED.**
  arm64 has a fused packed kernel now (`cpu.EmitA64PackedMatVecFused`), so both
  batched entry points CAN run there -- and on the M4 they cost 24%:
  Qwen3-MoE-4x0.6B CPU decode 106.1 tok/s with the per-expert loop against 80.8
  batched. The fused kernel's walk of the plane-major layout loses to the tiled
  one at a mixture's shapes. `nn.JIT.MixtureBatches` is the question now
  (`WithMixtureBatch`/`JITLLM_MOE_BATCH` force it), and the loop's matvecs go
  through `MatVecPacked`'s per-shape choice (`engine/nn/mvpick.go`).
  `docs/engineering-history/cpu-kernels.md` ("An arm64 fused packed matvec, and why
  the M4 picks a kernel per shape") has the tables.

  ★ **THE FIRST VERSION OF THESE GATES ENCODED amd64 AND WOULD HAVE GONE RED ON
  THE M4 FOR A CAPABILITY THAT WAS NEVER THERE.** `model.TestMoEFFNRunsBatched`
  asserted `looped == 0`; it derives the expectation from the JIT
  (`MixtureBatches(gate.typ)`, `PackedFusedSupported` before the arm64 fused kernel existed) and
  read, on the M4 of that day:

      olmoe    Q4_K, fused kernel false -> gate/up batched 0 looped 64,
                                           down batched 0 looped 64
      Linux    Q4_K, fused kernel true  -> gate/up batched 64 looped 0,
                                           down batched 64 looped 0

  and it carried `//go:build amd64 && linux`, so on the M4 it printed **"no
  tests to run"** -- RULE 11d twice in one file: a gate that cannot compile on
  the architecture it is about, and one that would demand the wrong thing if it
  could. It is `amd64 || arm64` now and RUNS on both. The `nn` batch gates SKIP
  on arm64 with the reason stated, and fail rather than skip on any build where
  a fused kernel does exist -- absent capability and broken gate look identical
  in a count of zero.

  ★ **SO A BATCHED arm64 MIXTURE THAT PAYS IS STILL OPEN WORK**: it would have
  to drive the tiled kernel over the experts' rows rather than the fused one.

  ★ **THE NEXT LEVER ON amd64 IS THE SERIAL STRETCH AND NOT MORE BATCHING**,
  which the region count now says plainly -- 14.6 a layer is the dense model's,
  so there is no dispatch left to collect. It named two candidates in the order
  the profile priced them: `ensureExperts`' synchronous reads (6.1%), then the
  router's softmax/top-k/sort/hotness chain. **BOTH ARE NOW MEASURED AND
  NEITHER IS A LEVER**; the router chain is a kernel and moved nothing (below),
  and the 6.1% is a WARM-UP rather than a rate (next entry).

  ★★ **AND HALF THAT CHAIN IS A KERNEL NOW AND IT IS WORTH NOTHING MEASURABLE,
  WHICH NARROWS THE LEVER RATHER THAN COLLECTING IT.** The top-k,
  the ascending-id sort and the renormalising divide left Go for one generated
  kernel on all three tiers (RULE 8; the softmax already was one). Measured on
  the in-process probe, olmoe-1b-7b Q4_K_M, six P-cores, one process per
  sample, ABBA, A/A self-control FIRST, clamped pairs discarded per RULE 2:

      A/A self-control (HEAD in BOTH arms)   0.9979   IQR/median 0.020   n= 9
      pass 1   HEAD / pre-kernel             1.0021   IQR/median 0.016   n=11
      pass 2   HEAD / pre-kernel             1.0064   IQR/median 0.025   n=11

  **All three gate, the two passes agree to 0.43 points, and the effect is the
  same order as the 0.21% harness bias** -- so this is RULE 6's "build it if it
  is cheap and correct; do not quote a perf claim on it", and NOT a win. It
  landed for RULE 8 and for portability: an ordering written in Go cannot go to
  a device, and this one now can.

  ★★★ **AND THE I/O WAS MEASURED AND IT IS AN AMORTISING CONSTANT, NOT A
  PER-TOKEN COST. THE 6.1% IS THE PROBE'S OWN WARM-UP, AND A SPECULATIVE EXPERT
  PREFETCH WAS PRICED AND LOSES ON ARITHMETIC.** The profile it came from
  warms 8 tokens and measures 64. Same binary, same model, one flag changed:

      olmoe, -devices cpu, n=  64    Syscall6 **7.46%** of samples, **0.65 s**
      olmoe, -devices cpu, n=1024    Syscall6 **0.64%**,             **0.75 s**

  **The share falls twelvefold and the seconds do not move.** The pager's own
  counters say it without a stopwatch: 53.0 requests a token over the first 64,
  2.53 over the next 256, **0.24** over the next 704. It is coupon collection
  over the bank -- once a selection's chunks are filled they stay filled -- and
  the total converges to the model's whole expert population.

  ★ **THE REGIME WHERE THE READ IS REAL IS THE OVER-COMMITTED ONE, AND A
  PREFETCH LOSES THERE BECAUSE THE BYTES ARE WHAT BINDS.** At 7 frames of 16 the
  routed read is 79% of every byte and ~67% of the decode wall (4.11 GiB/s while
  reading, against this box's 4.60-4.73 wall) -- and routing predicts too
  weakly to buy that back: **34.8% previous-token reuse on olmoe, 33.9% on the
  30B** (`residency.go`'s 73.9% was the 4-expert toy, where chance alone is
  50%). Reaching a usable hit rate means fetching most of the bank:

      W=1   hit 34.8%   1.00x the experts fetched   ->  **0.79x** the rate
      W=16  hit 98.5%   5.02x                       ->  **0.29x**
      a perfect oracle, no extra bytes              ->  **1.20x**, the ceiling

  **The W=1 prefetch's device time exceeds the whole token it is accelerating**,
  with all compute assumed perfectly hidden. The prefetch was priced at both
  windows and loses at both; the two tables are in `docs/engineering-history/placement.md` 15t and
  `model.TestRoutingLocality` reprints them in one command.

  ★ **AND neurostream's FIX IS ALREADY BUILT HERE.** Their 76.3 s -> 0.8 s was
  planning once per LAYER instead of once per projection -- queue depth, not
  prediction -- and `ensureExperts` already issues one plan for the whole
  selection, citing them. That number does not license a cross-token predictor.

  ★ **WHAT WAS ACTUALLY LEFT WAS THE PLAN, AND ONLY SPLITTING THE COUNTER SHOWED
  IT.** `model.Model.ExpertIO` reports blocked wall and the I/O inside it;
  the difference is `planRuns`, which walked a bool window over the span the
  ranges BRACKET -- and a routed selection is scattered, so that span is almost
  the whole page. 10.5 us a layer-step on olmoe; sorting the spans and walking
  only what was asked for is 4.3 us and allocates 48 B a plan against 16,432 B
  at 16,384 chunks. Reads byte-identical in both regimes. **No rate is claimed**
  -- the runs moved 11% where the plan is 0.5%, so that was the box.

## Closed: the M4 E-cores in prefill, the Linux elementwise board, the MoE accumulation order

- ~~The M4's E-cores are worth nothing to prefill~~ **THAT READING DID NOT HOLD:
  they are worth +14% to +54% of CPU prefill on the five dense models on the
  M4**, once prefill drains through its OWN pool (P+E) with GOMAXPROCS raised for
  that phase only (`5b04d82`). The old reading varied `JITLLM_CORES` with one pool
  and one GOMAXPROCS for both phases, which cannot show it: GOMAXPROCS pinned at 4
  made the wide pool 8.6x slower, pinned at 10 cost decode 6.5%. Mixtures stay on
  the decode pool -- E-cores made Qwen3-MoE prefill 17% slower. Linux keeps one
  pool: the same split measured neutral there (the "Prefill never gets its
  E-cores" entry above).
  `docs/engineering-history/cpu-kernels.md` has the rows.
- ~~the elementwise work is unmeasured on LINUX~~ **CLOSED: the box
  came out of the clamp (PSI 0.00, load 0.35, cores 3.4-4.1 GHz) and the whole
  decode board was re-taken at b31240e. Every row moved up 4.7-7.1%, mean +5.4%,
  and the movement is UNIFORM across four architectures and a mixture -- which is
  the signature of a change in what every token does, not of any one kernel.**
  Part of it is the clamp lifting rather than the code: the ratio normalises
  against llama.cpp in the same seconds, but the two engines do not shed rate to
  a clamp equally (the M4 prefill row has jitllm losing 7.4% where llama.cpp lost
  4.2%), so +5.4% is an upper bound on what the code did.
- ~~the MoE accumulation order~~ **CLOSED, AND THE FIX IS CHEAPER THAN
  WHAT IT REPLACED.** Both paths now add each expert's contribution STRAIGHT INTO
  THE RESIDUAL in ascending expert id. Decode sorts its selection (eight ints,
  insertion sort, no allocation) and keeps selection order for the trace, the
  hotness counter and the renormalisation sum; the batched path already visited
  the bank expert-major and did not change. Forward, Prefill and ForwardBatch are
  bit-identical on a mixture again, and decode does k residual adds where it used
  to do k+1: M4 MoE prefill 0.9999 and decode 1.0021 against a 1.0009
  self-control. The first version matched by giving the BATCHED path a zeroed
  per-row accumulator and one extra pass, and measured 0.9965 on prefill --
  correct, and paid for.

## Design notes moved out of source comments

Each entry below is a comment that stood in the named source file,
verbatim. The source keeps the contract, the layout and the invariants
and points here for the derivation, the transcript and the reasons.

### jit/cpu/topk.go: EmitMoETopK

The mixture-of-experts router's selection, renormalisation and ordering, as
one generated kernel. This is the AVX2 tier; sse_topk.go and topk_a64.go are
the other two, and topk_const.go holds the constant block and the scratch
layout all three read.

	Q32      p, the softmax over ALL experts   (f32, n)
	Scr      MoETopKConsts()
	Scratch  MoETopKScratch(k) float32s, per caller
	ASum     sel, the ids in SELECTION order   (int32, k, written)
	AHalfSum ord, the same ids ASCENDING       (int32, k, written)
	Out      wt,  sel's probabilities / sum    (f32, k, written)
	Out2     ow,  ord's probabilities / sum    (f32, k, written)
	AScale   sum, the divisor                  (f32, 1, written)

n, k and whether the architecture renormalises are baked from the
container's config.

Ties go to the lowest index, as ggml_argsort's descending order does; it
matters when a fresh router emits identical logits for every expert.

The accumulation orders are a correctness property: `sum` is accumulated in
selection order, term for term, because Forward, Prefill and ForwardBatch
are gated bit-identical against each other on a mixture, and `ord` is
ascending because moeBatch visits the bank expert-major. The kernel keeps
both.

Selection is a lexicographic walk rather than masking winners to -Inf,
which cannot express the tie rule when every probability is equal. The
keys (p[e], -e) are totally ordered and each pass takes their maximum, so
the unused set at pass i is exactly the keys below the previous winner's:

	eligible(e) = p[e] < prevVal || (p[e] == prevVal && e > prevIdx)

Pass 0's predicate is `p[e] == p[e]`, which admits everything but a NaN. A
NaN is therefore never selected; p is a softmax in [0,1], so no real input
reaches that case.

The ragged tail runs the vector body with the lanes past the end filled
with NaN (one VPBLENDD), which fails every branch of the predicate, so no
second accumulator or merge is needed. A lane that has seen nothing
eligible is tracked explicitly: the improve condition is `eligible && (p >
max || lane empty)`, so a -Inf candidate beside NaN pads is still chosen and
the kernel is total over every finite input and both infinities.

### jit/cpu/ropetab.go: EmitRopeTable

The rotary table, generated: the {cos, sin} pair every rotary pair of one
position needs. This is the AVX2 tier; sse_ropetab.go and ropetab_a64.go are
the other two, and ropetab_const.go holds the constant block, the position
decomposition and the exactness argument all three depend on.

	Out     cs, the table: 2*npairs float32, {cos, sin} per pair, written
	AScale  the per-model plane block (ropetab_const.go's layout)
	Scr     RopeTabConsts()
	K       the position

npairs is baked from the model's NRot.

The kernel is a range reduction and two polynomials, exp.go's shape: the
quadrant and two polynomials on |r| <= pi/4, no table, no branch, no call.

It agrees with the float64 table to one ulp and differs on about 18.7% of
entries. That is a floor, not a defect: a reduced argument held in one
float32 already carries half an ulp. Neither an exact reduction, math.Sin
in place of the polynomials, nor a two-word r with back-correction closes
it; all three were tried, and docs/engineering-history/cpu-kernels.md has
what each measured.

Both outputs come from one reduction, so sin and cos cannot disagree about
the quadrant at a boundary. The quadrant is three bit operations and no
branch. With n the nearest integer quarter-turn and r what is left,

	sin(th) = [ s, c, -s, -c ][n mod 4]        cos(th) = [ c, -s, -c, s ][n mod 4]

so bit 0 of n swaps the two polynomials, bit 1 of n negates sin, and bit 1 of
n+1 negates cos. The swap is one XOR of the pair masked by bit 0 broadcast
(VPSLLD then VPSRAD), and each negation is an XOR with the sign bit. Two's
complement makes this correct for a negative n too (sin(-pi/2 + x) = -cos(x)).

Registers, fixed for the whole kernel:

	RCX  cs        RDX  the plane cursor   RBX  Scr        RAX  the vector count
	Y0..Y3   the four position digits, as floats
	Y4  the mod-4 magic   Y5  mscale   Y6  pi/2 hi   Y7  pi/2 lo
	Y8..Y15  the body's working set

### jit/cpu/hc_const.go

DeepSeek V4's two small kernels, the hyper-connection mixer and the
compressor's pool. This file holds what every tier shares: the constant
block and the ABI; hc.go (AVX2), sse_hc.go and hc_a64.go are the bodies.

The hyper-connection mixer (EmitHCMix) turns one row's (2+H)*H mixes, H = 4
(HCStreams), into the collapse weights, the placements and the stream mixer:

	pre[i]  = sigma(m[i]*s[i] + b[i]) + eps                  i < 4
	post[i] = 2*sigma(m[4+i]*s[4+i] + b[4+i])
	comb    = softmax_rows(m[8:]*s[8:] + b[8:]) + eps        4 x 4, row-major
	comb    = comb / (column sums + eps), then iters-1 times:
	          comb / (row sums + eps), comb / (column sums + eps)

iters 0 stops after the softmax (a gate's violation). A head kernel computes
pre alone, from eight-float buffers whose last four are padding.

	Out     the 24 outputs (8 for a head)
	Q32     the mixes
	AScale  each mix's scale (s0 on pre, s1 on post, s2 on comb)
	Q2      each mix's base
	Scr     HCMixConsts(eps)

Every tier keeps the 4 x 4 mixer one row per 128-bit lane (AVX2: two rows a
YMM; SSE and NEON: one row a register), so a row's sum and maximum are
within-lane shuffles and a column's are adds of whole registers.

The pool (EmitColPool) is the compressor's softmax over positions, per
channel:

	out[c] = sum_s e[s,c] * kv[s*W+c] / sum_s e[s,c],  e = exp(gate[s*W+c] - max_s gate[s*W+c])

	Out     out, W wide
	Q32     kv, S rows of W
	Q2      gate, S rows of W
	Scr     ActConsts()
	K, Rows the whole vectors and tail elements of W (ElemLanes)
	Cols    S, at least one
	RowStr  4*W, the slot stride in bytes

### jit/cpu/sample.go: EmitSampleSegMax

EmitSampleSegMax generates the ordering's one primitive: the largest ELIGIBLE
element of each of Rows consecutive segments, with the lowest id winning a
tie.

	Q32      the values                                    (f32)
	ASum     the ids                                       (int32, idsMem only)
	Out      the per-segment best value                    (f32, Rows, written)
	AHalfSum the per-segment best id                       (int32, Rows, written)
	K        the segment length in ELEMENTS
	Rows     how many segments
	Cols     the id of the first element                   (!idsMem only)
	AScale   {prevVal, bitcast(prevIdx)}                   (!first only)
	Scr      MoETopKConsts()

One kernel serves all three uses. The initial pass runs it over the whole
vocabulary with Rows segments of K (first, contiguous ids). Each extraction
then runs it twice: once over the winner's own segment for that segment's
next best (eligible, Cols = the segment's base), and once over the summary
(Rows = 1, K = the segment count, ids read from memory) to find which
segment holds the next winner.

The eligibility predicate is the MoE router's (topk.go): the keys
(value, -id) are totally ordered, so the unused set after a winner (pv, pi)
is exactly the keys below it:

	eligible(e) = v[e] < pv || (v[e] == pv && id[e] > pi)

first bakes the pass with no predecessor, whose predicate `v == v` admits
everything but a NaN.

A segment with nothing eligible left reports (-Inf, INT_MAX), which loses
every tie to a real id, so the summary sweep is total and an exhausted
segment is chosen only once all are.

The ragged tail runs the vector body with the lanes past the end filled
with NaN, as topk.go's does; only the last segment is ragged, but K is a
register, so every segment's code handles it.

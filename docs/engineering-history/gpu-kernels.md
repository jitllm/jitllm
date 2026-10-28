# GPU kernels and the device tier — evidence

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

## ★ RULE 12r: THE 34x IS WEIGHT TRAFFIC, AND THE IR CANNOT EXPRESS THE TEXTBOOK FIX.

RULE 1d-duodecies measured device prefill at 0.0298 of llama.cpp; RULE 12u batched it and the row is 0.2431 now. The instinct is "batch the
submissions", and the numbers refuse it:

    device decode    160 tok/s  = 6.25 ms/token   reading 0.49 GB of weights
    device prefill   151 tok/s  = 6.6  ms/token   -- the SAME, because it IS decode, 507 times

**A token costs what a token costs, so the fix has to make one weight byte serve many tokens.**
Removing launches buys nothing when 6.6 ms of a 6.6 ms token is reading weights.

**★ AND A TILED GEMM IS NOT AVAILABLE: THIS IR HAS NEITHER SHARED MEMORY NOR A BARRIER.** The
classic answer stages a weight tile in shared memory for a whole thread block. `Reduce`'s own
comment records the constraint -- it exists as a second kernel "because this IR has neither shared
memory nor a barrier, and adding both to three backends to save one launch would be a poor trade".
That trade looks different now, but it is still three backends.

**So the reuse has to happen in REGISTERS: one thread takes one row and Tok token columns, loads
each weight word ONCE and issues Tok dot products.** Weight traffic falls by Tok. **That is the
CPU's interleave width (RULE 6) arrived at independently on the device, for the same reason and with
the same shape** -- and RULE 5b's warning travels with it: the best Tok on one thread is not the
best Tok at the real occupancy, so it is a measurement and not a constant.

**★ AND NTok REUSES THE MoE SLOT ADDRESSING RATHER THAN INVENTING ANY.** `SlotAct` already means
"each slot reads its OWN activation vector, laid out contiguously at stride K" -- which is exactly
what n token columns are. So a batched matvec is `Slots = NTok, SlotAct = true`, every offset is the
one RULE 14j already built and gated, and the only genuinely new thing is Tok. **The MoE work paid
for most of the prefill fix without knowing it.**

★ LANDED SO FAR: the shape (`NTok`, `Tok`), the refusals it cannot serve (experts, slots, split --
prefill has token parallelism and needs no split), and the guarantee that the DECODE kernel is
unchanged. At `Tok = 1` this is a correct batched launch that removes 507 submissions and does NOT
yet amortize weight traffic, which is the whole prize -- so expect nothing from it until Tok > 1,
and do not quote a prefill number before then.

## ★ RULE 12af: RULE 12x's CONTROL IS DEAD, AND THE K-QUANT FOLD IS BACK ON THE
TABLE. 46% OF THE MMA MATVEC IS THE FOLD AND 2.7% IS THE MATRIX INSTRUCTION.

**★ THE GPU PREFILL ROWS, RE-MEASURED AFTER 12y-12ae:**

    tinyllama q3_K_M   0.3666   REJECTED at IQR 0.119, CPU 841-2082 MHz
    gemma-2b           0.5689   IQR 0.045, 2 clamped samples discarded
                                (was 0.2678, measured at the 12u stage)

**gemma MORE THAN DOUBLED without a single gemma-specific change**, which is the
12y-12ae work carrying across. Both rows were taken with the package deep in RULE
2d's partial band and BOTH ARE PROVISIONAL -- but the ORDER is what matters and
the order inverted.

**★ AND THAT KILLS THE CONTROL RULE 12x CLOSED THE FOLD WITH.** 12x reads: "gemma-2b
is Q4_0 + Q8_0 and contains NO k-quants at all: tinyllama 0.2431, gemma 0.2678 --
10% apart. A model that cannot pay the k-quant fold is barely faster than one that
is 53% Q3_K, so whatever binds is common to both -- the GEMM's structure, not its
format. So porting the integer fold to the device was not expected to buy a row."

That was sound when the common bottleneck was dispatch and address arithmetic.
Those are gone (12z), and the two models are now **0.5689 against ~0.37 -- 55%
apart, and the k-quant model is the slow one.** Format is now the biggest single
term. **A control is only valid while the thing it controls for is still the
binding constraint.**

**★ THE SASS SAYS THE SAME THING WITHOUT ANY RATIO.** `jit/gpu/cmd/ptxdump` +
ptxas + cuobjdump, tinyllama's shipping prefill shape (Q3_K, k=2048, rows=5632,
NTok=128, MT=4, NT=4):

    mma-q3k   592 instructions        dp4a-q3k   872
      128 FFMA  |                       256 IDP
       72 I2FP  |- 272 = 46%            128 FFMA + 68 I2FP + 68 FMUL = 264 = 30%
       72 FMUL  |  the float fold       112 LDG
       64 STG  44 LDG  34 SHF           64 STG
       16 IMMA  <- 2.7%                 168 registers
      127 registers

**The matrix instruction is 2.7% of the kernel and the per-sub-block float fold
is 46%.** Every IMMA produces a complete integer dot product and the fold then
converts it, scales it and accumulates it -- once per (row, token, sub-block),
and Q3_K's sub-blocks are 16 elements where Q4_0's are 32, so it pays TWICE as
often. That is exactly the asymmetry the two rows now show.

**★ AND THE FIX MUST REMOVE WORK, NOT RELOCATE IT, BECAUSE 127 REGISTERS IS A
CLIFF.** RULE 12z built the integer fold, measured it at -25% AFTER the address
fix, and removed it: "the integer partial is a second accumulator array, 64 int32
beside 64 float, and ptxas reports 173 registers against 127 -- 10 warps to an SM
instead of 16. On this kernel OCCUPANCY BEATS INSTRUCTION COUNT." The kernel is
still at 127. So the direction is a CHEAPER fold at the same register count --
hoisting the activation scale per super-block the way RULE 6d did on the CPU is
the first candidate, since at window 256 one d_a covers the whole super-block and
the FMUL leaves the sub-block loop -- not a second accumulator array.

★ MEASURE THIS ON A RESTED BOX. Both rows above ran at 1034-2103 MHz with PSI at
0.00 and the die at 72 C -- idle, cool, and clamped, which is RULE 2d's
sustained-power limit and its exact signature: the package stops making heat
because it stopped working.


## ★ RULE 12ag: THE ZERO POINT MOVED ONTO THE WEIGHTS. 64 FFMA AND 8 LOADS GONE
PER SUB-BLOCK, AND THE REGISTER TRADE IS THE RIGHT WAY ROUND THIS TIME.

RULE 12af put 46% of the MMA matvec in the per-sub-block float fold and named the
d_a hoist as the fix. **That was wrong and the investigation is worth keeping**:
this kernel's sub-block loop is FLAT over all K (`super = subIdx >> log2(perSuper)`
is derived, not a loop level), so hoisting d_a needs `acc += d_a_super * inner`
and therefore a SECOND accumulator array -- the exact 64 registers that took RULE
12z's integer fold from 127 to 173 and cost it 25%.

Three other routes are closed too, each for its own reason: folding
`scale[x]*da[y]` into one multiplier is bijective over the 64 outputs and buys no
reuse; covering two Q3_K sub-blocks in one m16n8k32 is RULE 12y's own refusal,
since they carry different scales; and the stores are 64 of 592 but execute ONCE
against a loop that runs 128 times, so the static histogram UNDERSTATES the
fold's share rather than overstating it.

**★ WHAT WORKS IS CENTERING THE WEIGHTS, AND THE BORROW PROBLEM HAS A TWO-OP
ANSWER.** A packed quant is unsigned -- Q4_0 0..15, Q3_K 0..7, Q6_K 0..63 -- and
the true weight is `q - biasK`. That correction rode the ACTIVATION side as
`-biasK*sumA[y]`, one Fma per output element per sub-block, 64 per iteration at
the shipping tile. Subtracting biasK from the packed bytes instead makes the
fragment hold the true signed weight, and mma's A operand is already s8.

A plain per-byte subtract BORROWS across lanes whenever `q < biasK`. This does
not:

	centered = (q + (0x80 - biasK) broadcast) ^ 0x80808080

Every byte lands in `[0x80-biasK, 0x80-biasK+qmax]` -- 131 for Q3_K, 135 for
Q4_0, 159 for Q6_K, **all under 256, so no byte carries into its neighbour** --
and the XOR then flips the bias away, mapping `[0x80-biasK, 0x7F]` to
`[-biasK, -1]` and `[0x80, ...]` to `[0, qmax-biasK]`. Exact, two instructions,
and it applies to every constant-bias format from one constant.

    SASS, tinyllama's shipping prefill shape (Q3_K, k=2048, rows=5632, ntok=128, 4x4)
      FFMA   128 -> 64     the bias correction, gone
      LDG     44 -> 36     the sumA loads, gone with it
      total  592 -> 528    -10.8% static, and MORE dynamically: every removed
                           FFMA is inside the loop, while the 64 STG that dilute
                           the static count run once
      registers 127 -> 128 -- sm_86 allocates at a granularity of 8, so both
                           round to 128 and occupancy does not move

**★ AND IT REMOVES LIVE STATE RATHER THAN ADDING IT, WHICH IS THE WHOLE
DIFFERENCE FROM RULE 12z.** The eight `sumA` values are no longer loaded or held;
the add/XOR overwrite the fragment register, whose uncentered value has no later
use. The integer fold failed because it wanted a second array resident; this
wants one fewer.

★ `ir.OpXor` IS THE ONE NEW IR OP, AND IT EARNS ITS PLACE ON ARITHMETIC RATHER
THAN ON TASTE: `xor.b32`, `OpBitwiseXor`, `^`. No `prmt`, no `lop3`, no packed
video subtract -- a general packed-subtract op would carry semantics nothing
needs, and PTX's `vsub4` is not a guarantee of one native instruction anyway.

★ GATED BY BIT EQUALITY AGAINST THE dp4a KERNEL, which is the sharp gate RULE 12y
built: 147 subtests, Q3_K/Q4_0/Q6_K/Q8_0 at win32 and win256 including the
shipping 5632x2048 shape, 0 FAIL. Run against three violations: the add without
the XOR fails, a bias off by one fails, and centering Q8_0 -- which is already
signed -- fails. `verify -prefill` is 0/10 with max|dlogit| 0 on the MoE and its
per-token control is 0 everywhere.

★ NO RATE WAS CLAIMED WHEN THIS WAS WRITTEN. The box was in RULE 2d's
sustained-power clamp (PSI 0.00, 72 C, 1034 MHz idle), so nothing timed that day
was valid, and the instruction and register counts above are exact and need no
quiet box.

**★ IT IS 1.066 ON THE ROW, MEASURED RESTED, AND THE SWITCH THAT MADE IT
MEASURABLE IS THE PART TO KEEP.** `JITLLM_NO_CENTER` disables the centering in
BOTH kernels, so the two arms run in ONE session against one llama.cpp -- which
is RULE 1d-undecies' requirement, not a convenience: a pre-change baseline
carried across days moved a prefill row 2.7 points on its own, survived the IQR
gate, and was the difference between "2.5% from a win" and "at parity".
tinyllama GPU prefill, direct jitllm-against-jitllm ABBA:

    round 1   DISCARDED -- 470 then 1039 tok/s, RULE 2a cold start
    round 2   1.0658
    round 3   1.0569
    round 4   1.0775
    median 1.066, spread 0.021

**And the llama.cpp arms of the same session corroborate it independently**:
0.4383 centered against 0.4134 uncentered is 1.060, from a ratio that divides
out the other engine entirely. Two routes to the same number, one of them
immune to whatever llama.cpp was doing that afternoon.

**★ 12ag-bis. THE SAME CENTERING IN THE dp4a KERNEL, AND IT IS A REGRESSION FOR
DECODE UNLESS IT IS GATED ON Tok.** The dp4a matvec has the same fold and is the
wider target: Metal and Vulkan have no matrix instruction, and DECODE uses this
kernel on every backend. The ratio there looked better still, because `lo` and
`hi` are unpacked once per (word, row) and feed all Tok token columns while the
Fma removed is per (row, TOKEN).

**That same asymmetry is what makes it a LOSS at small Tok.** Counted in SASS,
Q3_K 5632x2048, centered against not:

    Tok      1     2     4     8    16
    delta   +8   +16    +8   -16   -48        Q4_0 measures the same shape

**Decode is Tok=1.** An ungated version would have made every decode kernel on
every backend eight instructions longer in order to speed up prefill -- RULE 5b's
"measure the configuration that ships, not the one that isolates cleanly" and
RULE 6a's "a below-bar measurement rejects the change AS MEASURED", both in one
change. `centered` now carries `&& s.Tok >= 8`, so decode kernels are
byte-identical -- `TestDenseKernelsUnchanged` passes against the ORIGINAL hashes,
which is the proof -- and prefill at Tok=16 reads 872 -> 824.

★ AND ONE OF THE TWO VIOLATIONS I RAN WAS VACUOUS, WHICH IS WORTH KNOWING.
Forcing `centered` on for the biasArray formats PASSED, and not because the guard
is unnecessary: at biasK == 0 the sequence is `(v + 0x80808080) ^ 0x80808080`,
which for bytes 0..15 is the IDENTITY. A mutation that changes nothing is not a
violation, and it looks exactly like a passing one. The meaningful violations --
a bias off by one, and centering the weights while KEEPING the activation-side
Fma so the correction is applied twice -- both fail.


**★ 12ag-ter. THE dp4a CENTERING HAD NO HARDWARE GATE ON ANY BACKEND, AND THE
GREEN SUITE ON THREE HOSTS SAID OTHERWISE. RULE 14r FOR THE SEVENTH TIME.**

12ag-bis gates centering on `s.Tok >= 8`. Every hardware test that touches the
dp4a matvec ran BELOW that line:

    TestMatVecMatchesReference   swept Tok {1, 2, 4}   -> the !centered branch, always
                                 formats Q4_0/Q8_0/Q4_K only
    TestMatVecMMAMatchesDot4     its dp4a arm is Tok=1 -> also uncentered, and it
                                 SKIPS the whole case when MMA declines

So 12ag's "147 subtests, 0 FAIL" gated the MMA kernel -- a DIFFERENT kernel,
CUDA-only, which centers unconditionally. **The kernel Metal and Vulkan run for
every prefill matvec at the shipped `BatchTok = 16` had never been executed with
centering on, anywhere.** The Q3_K case is the one that matters: it is 53% of a
tinyllama and it carries the 1.066 row above.

**THE GATE IS Tok=1 AGAINST Tok>=8, BIT-IDENTICAL.** No float64 oracle is needed
and that is why it is cheap: Tok changes only how many token columns a thread
carries, never the summation order within a column, and the centered dot IS the
corrected dot. Bit equality is therefore the right bar, and building weights for
all six formats was already done in `mmamv_test.go`. Two changes: a declined
MMA records a reason instead of skipping the case, and Tok 8 and 16 run before
it. Run against a violation (`0x80 - biasK + 1`) on all three backends:

    ptx    NMSE 2.404e-02    spirv  NMSE 2.420e-02    msl  NMSE 2.404e-02
    correct: exact on every centered format, every backend, 0 elements differ

**★ AND IT FOUND SOMETHING THAT IS NOT ABOUT CENTERING: Q4_K AND Q5_K ARE NOT
Tok-INVARIANT ON SPIR-V.** Both are `biasArray` formats, so `centered` is false
for them at every Tok -- and Tok=1 against Tok=8 still disagrees on ~40% of
elements, worst |delta| 6.1e-05, on spirv only. **PTX and MSL are exact on the
same shapes.** Tiny and one-backend, so it reads as a reassociation in the min
term's fold rather than a wrong answer, and the gate bounds it (NMSE < 1e-8)
instead of excusing it. **UNDIAGNOSED.** Whoever picks it up: it is the min term
(`-dmin*m[s]*sum(a)`) and the SPIR-V lowering is the only variable.

**★ THREE THINGS A REVIEW CORRECTED, AND TWO ARE MINE TO OWN.**

1. **Q4_0's centered byte tops out at 135, not 139** -- `128 - 8 + 15`. Written
   139 here and in `mma.go`. The carry proof is unchanged (all three are under
   256) but a bound quoted wrong is a bound nobody can check.
2. **"a regression for decode" EXCEEDS THE EVIDENCE.** The `Tok >= 8` threshold
   rests on a static SASS delta (+8 at Tok=1) on ONE vendor. **RULE 5z is this
   project's own finding that a static count is not a share of the token**, and
   12ag-bis quotes RULE 5b and 6a while doing exactly what 5z forbids. Keep 8 as
   the conservative default and label it what it is: **unvalidated on Metal and
   Vulkan.** Eligibility (constant non-zero bias, byte range) is correctness and
   belongs in the kernel; profitability is a per-target policy and does not.
3. **`JITLLM_NO_CENTER` CANNOT RUN THE DECODE EXPERIMENT**, which I would have
   discovered by running it: at Tok=1 both arms generate the uncentered path, so
   the switch toggles nothing there. It also reads the env at package init, so
   it cannot be flipped inside a running process. Testing decode centering needs
   a force-ON below the threshold, which does not exist.

**★ 12ag-quater. THE THRESHOLD IS MEASURED NOW, NOT COUNTED -- AND THE SECOND
LEVER WAS PRICED AND IS DEAD. BOTH ANSWERS COST ONE AFTERNOON AND SAVED A DAY.**

12ag-ter left `Tok >= 8` resting on a static SASS delta and said so. It is a
POLICY, not a correctness condition, so the predicate is split: `eligible` is
the constant non-zero bias whose centered bytes cannot carry, and
`centerMinTok` (JITLLM_CENTER_MINTOK, default 8) is the break-even. **That split
is what makes the decode question askable at all** -- at the shipped 8 both arms
of any A/B generate the uncentered kernel at Tok=1, so the switch that exists
could not run the experiment, which a review pointed out and which one attempt
would have shown.

Correctness first: with centering forced to Tok=1, `TestMatVecMatchesReference`
passes against the float64 reference on ptx, spirv AND msl, and the same run
against the bias-off-by-one violation fails 82 subtests. **The decode path is
CORRECT when centered; the only question was ever whether it pays.**

    tinyllama, decode n=96, cross-process ABBA, ON = centered at Tok=1
      CUDA    per-round 0.9738 0.9765 0.9854 0.9882   median 0.981   4/4 agree
      Metal   per-round 0.9936 1.0105 0.9953 0.9822   median 0.994   3/4
      Metal A/A SELF-CONTROL, run FIRST   113.64-119.11, spread 4.8%

**ON NVIDIA THE EXCLUSION IS EARNED: a consistent 1.9% loss, every round in the
same direction.** The static count predicted the sign correctly and is now
retired in favour of the measurement.

**ON METAL IT IS NEUTRAL AND THE ROW IS UNRESOLVED, WHICH IS NOT THE SAME AS
CONFIRMED.** 0.994 sits well inside a 4.8% self-control, so this harness cannot
tell a 1% regression from a 1% win there, and n=96 cross-process is the reason.
**Keeping 8 is still right** -- it avoids a measured loss on one backend and
costs nothing detectable on the other -- but the honest label is "measured on
CUDA, indistinguishable from zero on Metal", not "validated".

★ AND THE SELF-CONTROL RAN FIRST THIS TIME, which RULE 5ac demands and which
every earlier Mac A/B in this file skipped. It is what turns 0.994 from a result
into a non-result, and that is the whole reason to run it.

**★ AND THE SUM-FREE QUANTIZER IS PRICED AT <=0.5% OF A PREFILL TOKEN.** It was
the first-ranked companion lever below, and RULE 5z says to
price a removal against the TOKEN before building it. `JITLLM_GPU_SKIP=quant`
already removes the WHOLE quantize kernel:

    tinyllama GPU prefill, rounds 2-3 (round 1 discarded, RULE 2a cold start)
      quantize ON    1938, 1844 tok/s
      quantize OFF   1938, 1928        -- ~2%, and inside the arm's own 5% spread

**The entire kernel is worth about 2% and the sums are 32 Add + 2 CvtF32 + 2
Store out of ~200 IR ops in it, so the lever's ceiling is ~0.4% of a token.**

★★ AND THE FIRST VERDICT WRITTEN HERE, "DO NOT BUILD IT", WAS THE WRONG ONE, WHICH A REVIEW CALLED THE
CLEAREST SURVIVING VIOLATION OF RULE 6-0 -- in a paragraph written the same day
6-0 was. **Size is not a refusal.** What the 0.4% establishes is ORDER: this goes
after the 8% fusion work and after the MoE dispatch batching, not that it is
forbidden. Two honest caveats on the number itself, also the review's: the ~0.4%
multiplies an UNRESOLVED skip result (the 2% sits inside the arm's own 5% spread)
by a static IR-op fraction, so it is an estimate of an estimate. What can be said
without either is that the change removes real work and cannot be SHOWN to help
on this harness -- so build it if it is cheap and correct, and do not put a perf
claim on it.
That leaves the MMA lifetime work and the Q6_K upload precenter below as the
surviving candidates, neither of them priced yet.

**★ AND THE SECOND LEVER, WHICH IS THE PART WORTH MORE THAN THE 1.066.**
Centering deleted the CONSUMER of the activation sums. **`emitQuant` in
`kernels/elem.go` still PRODUCES them** -- two integer half-sum chains,
converted and stored, every quantize, for consumers that no longer read them.
Deleting a consumer cannot remove work across a kernel boundary, so a sum-free
quantizer variant is a dependency removal that was impossible before centering
and is worth nothing without it. It is per-BUFFER, not per-model: Q4_K/Q5_K and
any uncentered constant-bias path still read the sums, so a mixed model cannot
delete them wholesale. Two more, in order:

  - **MMA lifetime scheduling BEFORE any wider tile.** The registers went 127 ->
    128 despite eight sum values disappearing, so deleted source state did not
    become allocated headroom; the loop builds all weight fragments, then all
    activation fragments, then scales. Consuming one weight-row tile across the
    token tiles and retiring it is the candidate. Widening Rowt*Tok first cannot
    pay -- roughly Tok deleted sums against doubled output accumulators.
  - **PRECENTER Q6_K AT UPLOAD.** It already occupies full bytes in the packed
    layout, so storing signed `q-32` once removes BOTH the runtime centering and
    the sum correction, with no payload expansion, **at every Tok including 1**.
    It does not transfer to Q3_K/Q4_0, whose values still need sign extension
    after the nibble extract. It needs a layout contract across every consumer.

★ AND A REVIEW FOUND THE SEQUENCE. I had ruled the weight-side correction out for
want of a borrow-free SWAR and asked the reviewer specifically whether I was wrong about
that. It was also right that `Builder.Done` does no dead-code elimination, so the
now-unused sumA loads had to be removed explicitly rather than left to the
lowerer -- eight loads per sub-block that would otherwise still be emitted.


## ★ RULE 12s: THE IR HAS SHARED MEMORY AND A BARRIER NOW, AND THE "NOT GETTING" COMMENT WAS ROT.

`ir.Shared(name, elem, n)` and `ir.Barrier()`, lowered on all three backends:

    PTX      a kernel-scope `.shared .align N .b8` array, cvta'd to generic; `bar.sync 0`
    SPIR-V   a module-scope OpVariable in the Workgroup class over an OpTypeArray;
             OpControlBarrier(Workgroup, Workgroup, AcquireRelease|WorkgroupMemory)
    MSL      `threadgroup T name[N]`; threadgroup_barrier(mem_flags::mem_threadgroup)

**★ THE IR SAID IT WAS "NOT GETTING" THESE, IN SO MANY WORDS.** OpShuffleXor's comment read: "a
reduction otherwise costs a second kernel ... a tree reduction wanted shared memory and a barrier,
neither of which this IR has and neither of which it is getting". That was RIGHT when written -- the
only customer was a softmax reduction and a butterfly shuffle bought it without touching three
backends. What changed is a measurement: RULE 1d-duodecies put device PREFILL at 0.0298 because
every weight is re-read per token, and the fix is a tiled GEMM, which stages a weight tile for a
whole thread block and cannot be written with shuffles. **RULE 15f again -- a comment explaining why
something is NOT done is a claim about the code, and it rots when the numbers move.**

**★ THREE PLACES IT DIFFERS FROM A BUFFER, AND EACH WOULD HAVE BEEN A SILENT BUG:**

  1. **The access chain has a DIFFERENT ARITY.** A StorageBuffer parameter is a struct wrapping a
     runtime array, so its chain is (member 0, element); a Workgroup array is the array itself, so
     it is (element). Using the buffer form indexes past the array.
  2. **A pointer's storage class is part of its TYPE**, so shared needs its own pointer types.
     Reusing the StorageBuffer ones is not a mismatch anything forgives.
  3. **PTX declares shared state at KERNEL SCOPE**, not inline, so `Buf` grew a declaration block.
     MSL allows an inline `threadgroup` declaration but one inside a loop body is a fresh object
     per iteration on some compilers -- so OpShared must precede any loop, on every target.

**★ AND spirv-val WAS ON THIS BOX ALL ALONG, JUST NOT ON $PATH.** `lowertest` logged "spirv-val not
installed: SPIR-V is lowered but not validated" on every run while the binary sat in a snap. **A
validator that is present and unused is worse than one that is absent, because the log line reads
the same either way** -- and this is the file RULE 12e wrote specifically to close that gap,
reopened by a PATH. It now looks in the snap paths too, and every kernel in the tree validates.

★ RUN AGAINST VIOLATIONS: the wrong access-chain arity is caught -- "OpAccessChain reached
non-composite type while indexes still remain to be traversed". **Dropping the barrier's memory
SEMANTICS (0x108 -> 0) is NOT caught**, and that is the honest limit of this gate: spirv-val
validates STRUCTURE, not memory ordering. A barrier that synchronises execution and orders nothing
assembles cleanly and races on hardware. Only a real device and a real race can see that one.

## ★ RULE 12t: THE BATCHED MATVEC IS CORRECT. ONE LINE ERASED THE WHOLE MAPPING, AND A REVIEW FOUND IT.

`NTok` (token columns per launch) and `Tok` (columns per THREAD, the weight reuse of RULE 12r) now
match the float64 reference on hardware at Tok = 1, 2 and 4 across every shape, and `Tok == 1` is
byte-identical to the decode kernel.

**★ THE BUG WAS THREE WORDS, AND I HAD STARED PAST IT.** `MatVec` maps a token batch onto the MoE
slot machinery -- `Slots = NTok, SlotAct = true` -- and eleven lines later:

    if experts == 1 { slots = 1 }

**Slots means "expert slots" to the MoE caller, so a dense launch resets it.** A token batch sets
Slots = NTok with Experts = 1, and the reset put it straight back to 1: the thread clamp became
Rows-1, `slotAct` went false so no per-token base was ever added, `sumOff` became nb where the
buffer holds NTok*nb of scales first (so Q4_0 read other tokens' SCALES as its sums), and only
column 0 was ever written. Four silent failures from one assignment.

**★ AND IT FALSIFIES RULE 12r's CENTRAL CLAIM, WHICH I SHOULD RECORD RATHER THAN QUIETLY FIX.**
12r says "NTok reuses the MoE slot addressing that RULE 14j already gated ... the MoE work paid for
most of the prefill fix without knowing it". The LAYOUT reuse was sound -- the review confirmed the
activation packing, the pAX scales-then-sums order and the output slot stride all match what `mvid`
produces. What was NOT reusable was the NORMALISATION and the THREAD GEOMETRY around it.
**Reusing tested addressing is not the same as reusing it correctly.**

Two more, both real and both invisible at Tok = 1:

  - **The grid decoded one token per thread.** With Tok > 1 a thread owns a token GROUP, so the
    clamp is Rows*(NTok/Tok)-1 and the group's first token is (flat/Rows)*Tok. The old expression
    let groups overlap each other's ranges.
  - **Only `accs[0]` was stored.** The Tok loops filled all of them; columns 1..Tok-1 kept whatever
    the buffer held, which is unspecified and NOT zero -- plausible garbage rather than a hole.

★ THE REVIEW ALSO FOUND TWO DEFECTS OUTSIDE THE SHAPE, AND THE FIRST IS THE MORE ALARMING:

  1. **PTX emitted `ld.global` on a SHARED address.** RULE 12s cvta'd the shared symbol to generic
     and left the loads global-qualified; ptxas ACCEPTS that and it reads the wrong address space on
     hardware. **The assembler gate cannot see it** -- `shared/roundtrip` assembled on all three
     backends while being wrong on one, because it was never EXECUTED. Fixed by keeping the address
     in the shared space and emitting `ld.shared`/`st.shared`.
  2. **`ir.Validate` did not check what two lowerers' comments said it checked.** Both claimed it
     enforced OpShared-before-any-loop. It enforced nothing. RULE 15f, written by the same hand in
     the same hour as the code it lied about. The check exists now.

★ AND THE TEST LAUNCHED 64 THREADS AT A 128-WIDE KERNEL. PTX reads `%ntid.x` at runtime so it was
invisible there; Metal BAKES the width and would skip rows, Vulkan ignores it and over-launches.
Three backends, three different wrong answers from one hardcoded constant. It uses `kk.Group[0]`.

★ THE VALUE OF THE SECOND OPINION, MEASURED: I had localised the fault correctly (Tok=1 failing
means the mapping, not the registers) and then looked for it in the addressing I had WRITTEN rather
than in the normalisation I had INHERITED. RULE 12i recorded the same shape -- "the value was not
GPU expertise; it was that it had no stake in the theory being defended".

★ STILL UNMOVED: the 34x. Correct is not fast -- nothing yet CALLS the batched kernel, so device
prefill is 0.2431 now (RULE 1d-terdecies); `Layers` is wired to it. The number
does not change until then.

## ★ RULE 12: the GPU tier runs WHOLE BLOCKS, and the win was the submission, not the kernels.

Serving one matvec at a time cost a blocking round trip per matvec -- ~59 us across the 127 a gemma
token issues, 7.5 ms of a 29 ms token, against llama.cpp's 14.7 ms for the entire token. `nn.LayerDevice`
moves the unit to a transformer BLOCK: every op between the matvecs (RMSNorm, quantize, RoPE, the KV
copy, scores, softmax, the weighted sum, the FFN gate, the residual add) is a generated kernel, so the
residual stream never returns to the host inside a block.

A block, not a token, for two reasons. The device never learns what a transformer is beyond one
repeating unit -- `engine/model/` keeps the graph, and nothing in `jit/gpu` knows about embedding lookup,
sampling or tied output weights. And **the CPU/GPU split falls out for free**: `PrepLayer` is per
block and may decline, so a model too big for the card runs its first N blocks on the device and the
rest on the CPU, with no separate placement policy. On a 4 GB card that is the difference between
running and not.

★ THAT SPLIT WAS A CORRECTNESS WIN AND A SPEED LOSS FOR AS LONG AS IT EXISTED, AND RULE 15g IS THE
FIX. The blocks the placement did NOT take went through `GPU.MatVec` one matvec at a time, which is
the cost this rule's own first paragraph is about -- so "the rest runs on the CPU" was true about the
weights and false about the speed, and every partial seam measured BELOW the CPU tier.

Measured on an RTX 3050 Ti Laptop, CUDA, `-n 32` after 8 warmup tokens, median of five:

    model        jitllm CPU   jitllm GPU   llama.cpp CUDA   ratio
    tinyllama      70.8     130.9        128.18       1.02x
    gemma-2b       26.5      69.6         79.06       0.88x

**All 32 greedy token IDs are identical to the float64 CPU path on both models.** The residual stream
is float32 on the device and float64 on the host, so agreement was expected to be NMSE rather than
exact, and it is exact. That is worth more than the rate: it makes the CPU tier a real oracle, and any
kernel change that shifts a token ID is a bug until proven otherwise.

**★ 12a. A SINGLE GPU RUN CANNOT SEE A 5% CHANGE, AND THREE CLAIMS WERE MADE ON ONE ANYWAY.** Five
identical gemma runs read 64.58, 68.14, 69.63, 69.83 and 73.33 tok/s -- a 13% spread. `jitllm-gpu -ab`
alternates two arms inside ONE process, on one State and one copy of the device weights, and reports
the median of per-round ratios with RULE 2's IQR gate. Every arm is a FIELD on the tier rather than an
env var, so both configurations are compiled and resident at once and nothing reloads between rounds.
That drops the spread to 2.5-5.9%.

    change                          gemma                 tinyllama
    one submission per token   1.1588x IQR 5.9%      1.1030x IQR 2.5%   <- the whole win
    measured split width       1.0021x IQR 4.8%      1.0495x IQR 7.1%   <- one model
    q|k|v fused into one mv    1.0072x IQR 2.8%      --                 REJECTED
    fused elementwise kernels  0.9910x IQR 3.2%      --                 REJECTED

**★ 12b. THREAD COUNT BEATS LAUNCH COUNT AND MEMORY TRAFFIC, WHICH COUNTING LAUNCHES DOES NOT
PREDICT.** A launch costs 2.6 us here and a decode block issues ~25 of them, 18 blocks per token, so
~1.2 ms of a 13.7 ms token is launch overhead. Fusing NormApply into Quantize, the residual add into
the next norm's partial sums, and the FFN gate into its quantization removes seven launches per block
and 128 KB of traffic -- and measures **0.9910x**. Quantization needs a thread per 32-element block,
so the fused kernel runs 64 threads where NormApply ran 2048, and on this device that trade loses.
Kernel fusion as a launch-count argument was tried and measured: 0.9910x.

**★ 12c. THE "SMALL MATVECS ARE SLOW" TABLE WAS A HARNESS ARTIFACT, AND IT SENT ME AFTER THE WRONG
THING FOR HALF A DAY.** cmd/mvbench timed each shape with `backend.Kernel.Launch`, which
SYNCHRONISES: on CUDA that is an ownership hop plus `cuLaunchKernel` plus `cuCtxSynchronize`, and for
`split > 1` the sample paid it TWICE. One timing sample was one launch plus one full device drain,
against a k/v matvec whose entire weight traffic is 0.3 MB. Timed instead through a `Session` with
many launches and ONE `Sync` -- the shape `tier.timeSplit` already used -- the same card, same
kernels, same day:

    shape                       was    now   of wall   what changed
    k/v      256x2048  0.3 MB    12     74     49%     6.2x
    q/o     2048x2048  2.4 MB    62    131     85%     2.1x
    qkv fused 2560     2.9 MB    73    133     87%
    ffn_down 2048x16384 18.9 MB 136    163    107%
    gate/up 16384x2048 18.9 MB  152    166    108%
    lm_head 256128x2048 557 MB  162    171    112%
    read wall 153.1 GB/s, launch floor 2.37 us through one Session

**q/o was never at 42% of the wall; it is at 85%.** k/v is the only shape still low, and 0.3 MB at
74 GB/s is 4 us against a 2.37 us launch floor -- roughly 40% of its time IS the floor, so it cannot
be priced in isolation at all and no kernel change will move it.

Everything above the wall says the `Stream` probe understates it, the way `bench.MemWall` does on the
CPU (RULE 8b). Treat 153 as a lower bound.

★ THE REUSABLE PART IS THAT THE A/B ALREADY KNEW. RULE 12a records q|k|v fusion at 1.0072x and the
measured split at 1.0021x on gemma, both far under what the isolated table predicted, and I read them
as puzzles. A fixed per-launch cost of ~19 us in the isolated measurement and ~0 in the real async
stream predicts exactly that discrepancy. **When an end-to-end A/B and an isolated microbenchmark
disagree by a factor, suspect the microbenchmark** -- RULE 11 says measure the harness before the
code it runs, and this is the third time in this file that a "slow kernel" turned out to be the thing
timing it.

q, k and v read the same activation vector, so packing them as one 2560-row matrix and running one
matvec is still the obvious fix for k/v, and it is still not worth it: end to end it measured
1.0072x, because q, k and v are 5.3 MB of the 61.9 MB a gemma block reads. Removed -- it needed a
second residency map and a source offset baked into RoPE and CopyAt, which meant whether a model
fuses had to be settled by its first block and held for every later one. A correctness hazard for
0.7%.

**★ 12d. NO KERNEL MAY READ A BUFFER IT WRITES, AND `ir.Validate` NOW REFUSES IT.** The IR has no
conditional execution outside a loop, so surplus threads are CLAMPED to the last work item rather
than branched off. That is benign exactly when the result is a pure function of read-only inputs.
An in-place RoPE re-rotated the last head and an in-place softmax re-normalised the last row, once
per surplus thread -- both produced correct numbers in every head but the LAST, which is the one the
clamp piles onto, so a spot check of head 0 passes and the failure reads like a GQA indexing bug.
Aliasing is a property of the ops, so it is checked rather than remembered.

**★ 12e. `jit/gpu/lowertest` ASSEMBLES EVERY SHIPPED KERNEL WITH THE VENDORS' OWN TOOLS.** PTX
refuses a divide with no rounding modifier, and nothing caught `div.f32` until a kernel that divides
reached real hardware, where the entire symptom was `CUresult=218`, "a PTX JIT compilation failed",
with no line number. The lowerers had unit tests for the ops they knew about; what was missing was a
gate that asks `ptxas` and `spirv-val` about the REAL kernels.

**★ 12f. A NESTED `LockOSThread` CORRUPTED THREAD CREATION FOR THE WHOLE PROCESS, AND THE CRASH
LANDED NOWHERE NEAR IT.** The symptom was "gemma segfaults on the Vulkan backend" -- during a
PURE-CPU decode, with the Vulkan device merely open and never called, flaky at about one run in two,
identical at every VRAM budget and memory cap, and reproducing back to `f9ab176`.

`cuda.Open` locks its OS thread, and `backend.openCUDA`'s owner goroutine locks it AGAIN while
deferring only one unlock. The count went 2 -> 1, so the owner exited STILL LOCKED. Go destroys the
thread of a goroutine that exits locked, using a raw exit that skips glibc's thread teardown -- and
with goffi's `fakecgo` in the process that thread is a real pthread, so it was left half-dismantled
and took out the NEXT thread the runtime created, inside `fakecgo`'s `threadentry`, jumping to an
address belonging to nothing.

Two things made it look like a GPU bug when it is a threading bug. It needs TWO backends: `Open()`
starts all of them and the tier keeps one, so asking for Vulkan opens CUDA and immediately closes
it -- a machine with one API, or a process that never closes a device, never sees it. And nothing in
the trace mentions CUDA, because by then the context is long gone.

Fixed by matching the lock (`cuda.Device.Close` destroys the context and unlocks) and by destroying
the context ON THE OWNER GOROUTINE before releasing it. `TestCloseThenSpawnThreads` opens every
backend, closes them, then forces the runtime to build 64 threads; it segfaults without the fix.

**The reusable part: a stack trace through `fakecgo`'s `threadentry` means a Go M, not a
driver-spawned thread.** That is what said to look at thread lifecycle rather than at Vulkan.

**★ 12g. VULKAN CALLED `vkQueueWaitIdle` ON EVERY DISPATCH, AND THAT WAS MOST OF WHY IT LOOKED
SLOW.** `vkDev.Session` was `f(directSession{})`, so each launch went to `Kernel.Launch`: reset the
command buffer, record one dispatch, submit, drain the queue. A decode token issues ~450 launches, so
it paid ~450 full drains. Metal already had the fix -- one command buffer per SESSION, citing 288.8
us per dispatch -- and CUDA never needed it because its launches are asynchronous. Vulkan was the arm
still doing it per dispatch, and nobody had looked.

    tinyllama, Vulkan, same card, same binary otherwise
    per-dispatch submit   30.9 tok/s
    one command buffer   101.1 tok/s      3.3x

Two things Vulkan needs that Metal does not. **A BARRIER BETWEEN DISPATCHES**: Vulkan does not order
compute dispatches within a command buffer the way a CUDA stream orders launches, so without an
explicit dependency the second may read what the first has not finished writing. Every kernel in a
block consumes the previous one's output, so each encode ends with a whole-pipeline
shader-write-to-shader-read barrier -- the conservative one on purpose, because the precise one needs
to know which buffers alias and being wrong is a race rather than an error. And **A DESCRIPTOR SET
PER DISPATCH**: `Kernel.Bind` updates the ONE set a kernel owns, which was correct only because
Launch submitted immediately; jitllm encodes its quantize kernel 54 times a token against different
sources, so batched they would all have run with the last binding. Sets now come from a per-Ctx pool
the commit resets.

**★ 12h. "OVER BUDGET" AND "THE DEVICE REFUSED" ARE OPPOSITE FACTS AND WERE ONE COUNTER.** gemma read
44 tok/s instead of 70 and the report said "1 tensors over budget" -- with `-vram` at 4 GB and 1.04
GiB resident, which cannot be a budget decision. Another process held 1262 MiB of the 4 GB card, so
lm_head's 557 MB allocation failed and the largest matvec of the token moved silently to the CPU.
`NoRoom` is now separate from `Declined`, and jitllm-gpu says the number is not comparable with a run
that had the card to itself. Over budget is a choice jitllm made; out of memory is the card being full.

Vulkan is still behind CUDA here (gemma 39.5 vs 44.9, tinyllama 101 vs 129, both under that VRAM
contention) and the API tuner duly picks PTX. Both backends produce token IDs identical to the CPU
path on every model.

**★ 12i. THE METAL SOFTMAX BUG DID NOT EXIST. `jitllm verify` DIFFED TWO AUTOREGRESSIVE CHAINS.**

The warp softmax -- one 32-lane subgroup per head, butterfly reduction, worth 8.4% on CUDA -- was
gated off on Metal for weeks on this evidence: tinyllama (nHead 32) produced **32 wrong tokens out of
32** there, while gemma (nHead 8) was perfect and CUDA was perfect at both. A standalone probe of the
same kernel, same nHead, same maxSeq, passed at n = 1, 2, 31, 32, 33 and 37. That gap between "passes
alone, fails in the chain" is what made it look like a subgroup-execution bug, and a long list of
things was ruled out: the encoder is `MTLDispatchTypeSerial`, buffers are `storageModeShared` with
tracked hazards, every dispatch sets its own pipeline and buffers.

**`benchRun` ran each tier as its OWN free-running greedy chain.** Both started from token 1 and each
fed its own argmax back in, after eight warm-up tokens that were generated and DISCARDED. So one
argmax flip anywhere -- including inside the discarded eight -- gave the two tiers different inputs
from that point on, and every id downstream differed. "32/32" was not 32 failures. It was one, and it
was invisible.

Teacher-forcing both tiers from a single token stream and comparing per position:

    Metal tinyllama  warp softmax    1/32 positions   margins 0.075 vs 0.021   max|dlogit| 0.43
    Metal tinyllama  SCALAR softmax  1/32 positions   margins 0.009 vs 0.028   max|dlogit| 0.42
    Metal gemma      warp softmax    0/32                                      max|dlogit| 1.58
    CUDA  tinyllama  warp softmax    0/32                                      max|dlogit| 0.30

**The scalar kernel disagrees just as often**, at a different position. Both are measuring the same
thing: the device tier is float32 and the reference float64, `max|dlogit|` is ~0.3 on the CUDA row
that PASSES, and tinyllama has argmax margins smaller than that. The softmax was never the variable.
The gate is removed; warp is the default on all three backends.

Three things to keep.

1. **A comparison of two autoregressive chains cannot localise anything.** The first disagreement
   destroys the inputs of every later one, so the output is a step function: 0 differences, or all of
   them. Diff at a FIXED input or do not diff. `verify` now teacher-forces.
2. **Report the MARGIN, not the mismatch.** "cpu 8790, gpu 2805" reads as a broken kernel; "margin
   0.075 against max|dlogit| 0.43" says the model was undecided and f32 broke the tie. One of those
   is a bug report and the other is a rounding fact, and the ids alone cannot tell them apart.
3. **A probe that passes while the pipeline fails is evidence about the PIPELINE, or about the
   harness -- and the harness is the cheaper hypothesis.** This is RULE 11 one layer up: there, the
   unmeasured thing was dispatch between the kernels; here it was the loop around the whole engine.
   Both times three careful analyses modelled the code and none checked the thing doing the asking.

Found by an outside review, which read `verify.go` and noticed the discarded warm-up in the first pass.
The value was not GPU expertise; it was that it had no stake in the theory being defended.

**★ 12j. METAL WAS NOT BANDWIDTH-BOUND, IT WAS INT-MULTIPLY-BOUND, AND ONE HELPER WAS 30% OF A
TOKEN.** The Metal tier sat at 0.60x of llama.cpp on both models -- and, more damningly, BELOW the
M4's own CPU tier (68.1 GPU against 70.9 CPU on tinyllama). `mvbench`, which prices each real decode
shape against a read wall measured in the same process, said where it went:

    gemma's shapes, GB/s and % of that device's own wall     CUDA/PTX        Metal/M4 BEFORE   AFTER
      q/o          2048 x  2048   Q4_0                     101   75%         43   46%       62   67%
      qkv fused    2560 x  2048   Q4_0                     110   81%         43   46%       65   70%
      gate+up     32768 x  2048   Q4_0                     142  105%         48   51%       68   74%
      gate/up     16384 x  2048   Q4_0                     142  105%         43   46%       68   73%
      ffn_down     2048 x 16384   Q4_0                     138  101%         47   50%       67   73%
      lm_head    256128 x  2048   Q8_0                     152  112%         98  105%      100  108%

**The Q8_0 row was already at the wall and every Q4_0 row was at half of it.** That single contrast is
the whole diagnosis and it took one run to see: identical kernel, identical access pattern, identical
device -- so the variable cannot be memory, and the two formats differ in exactly one way. Q4_0 packs
two quants per byte, so per BYTE READ it issues four times the `Dot4` a Q8_0 row does.

`ir.OpDot4` is one instruction on the other two targets (PTX `dp4a.s32.s32`, SPIR-V's dot). **Apple
silicon has no integer dot product**, so MSL lowered it as four sign-extends, four 32-bit integer
multiplies and four adds -- and Apple's ALU is fp32-centric, where a 32-bit integer multiply is a
multi-instruction sequence and a float FMA is one. Doing the same dot in float:

    float4 x = float4(as_type<char4>(a)), y = float4(as_type<char4>(b));
    return c + int(dot(x, y));

**This is not a precision compromise and it is not an approximation.** int8 x int8 is at most 16129
and the sum of four at most 64516; both are exactly representable in a float32 mantissa, so the
integer accumulation is bit-identical and the teacher-forced diff got BETTER, not worse (tinyllama
Metal went 1/32 positions to 0/32).

    tinyllama Metal   68.1 -> 87.49 tok/s   +28%    now 1.24x the M4's CPU tier, was 0.96x
    gemma     Metal   34.0 -> 44.69 tok/s   +31%
    vs llama.cpp Metal: 0.60x -> 0.77x (tinyllama) and 0.79x (gemma)

Three things worth keeping.

1. **"The same generated kernel is correct everywhere" is a property of the IR; "the same kernel is
   FAST everywhere" is not, and one op can carry the whole difference.** The IR's job is to name the
   operation the hardware wants; where a target lacks that operation, the lowering is not a detail.
   Audit the other MSL lowerings against their PTX counterparts for the same shape of gap.
2. **A per-format efficiency table is a faster diagnostic than a profile.** Two formats through one
   kernel on one device is a controlled experiment with the memory system held fixed, and it pointed
   at the arithmetic in one line. A profiler would have shown "the matvec is slow", which was known.
3. **The GPU tier being slower than the CPU tier on the same box was the loudest number on the
   scoreboard and it went unexamined for a while**, because the scoreboard compared each tier against
   llama.cpp and never against the row above it. Compare jitllm to jitllm first; it is the same code and
   the same model, so a loss there has nowhere to hide.

**★ 12k. AND THEN THE INT ACCUMULATOR TURNED OUT TO BE A COSTUME. +5.8% MORE.** 12j left Q4_0 at
67-74% of the Metal wall against Q8_0's 108%, with the remaining cost named: `ir.OpDot4` is typed I32,
so `jitllm_dot4` computed the dot in float and then converted BACK to int, and the int accumulator back
to float, on every four elements -- to honour a type that no consumer wanted.

**The matvec's accumulator chain has exactly one exit, and it is `CvtF32`.** `d_a * (scale*dot -
bias*sum)` needs a float; nothing else reads `iacc`. So the MSL lowerer now finds those chains and
carries them in float: a value qualifies when it is produced by `Dot4` or a `Phi` and every consumer
is another `Dot4`'s accumulator, a `CvtF32`, or a `Phi`. `CvtF32` on a carried value emits nothing at
all.

    gemma's shapes, % of the Metal wall     12j    12k     CUDA, for scale
      q/o        2048 x  2048  Q4_0         67%    75%      85%
      qkv fused  2560 x  2048  Q4_0         70%    79%      87%
      gate+up   32768 x  2048  Q4_0         74%    84%     106%
      gate/up   16384 x  2048  Q4_0         73%    90%     108%
      ffn_down   2048 x 16384  Q4_0         73%    83%     106%
      lm_head  256128 x  2048  Q8_0        108%   114%     110%

    tinyllama Metal   89.9 -> 95.10 tok/s     gemma Metal   43.87 -> 46.09

**It is exact, and that is checked per format rather than asserted.** A float32 mantissa holds every
integer below 2^24 = 16,777,216. The largest accumulation any shipped format reaches is a 256-element
Q6_K super-block at 256 x 127 x 127 = 4,129,024 -- a factor of four inside it -- and Q4_0's 32-element
sub-block is 60,960. The chain is not an approximation of the integer one, it IS the integer one,
which is why `TestMatVecMatchesReference` passes unchanged on every format, shape and split.

Two things worth keeping. **A peephole in ONE backend was right here, and the IR change was not.** The
tempting fix was an `OpDot4` whose accumulator type is the backend's choice, which would have put
Apple's ALU into the shared IR and made PTX and SPIR-V -- where `dp4a` and `OpSDot` genuinely are
integer instructions -- carry a concept they do not want. The IR says what the operation IS; a
lowerer is entitled to know that on this target the honest register is a float. And **the type in an
IR is a claim about the VALUE, not about the register**: nothing here changed what is computed, only
where it is held between two instructions that both wanted it elsewhere.

**★ 12l. METAL COMPILES WITH FAST-MATH ON, SO FLOAT ALGEBRA IS NOT A LEVER THERE.**
`newLibraryWithSource:options:` is called with nil options, which means
`MTLCompileOptions.fastMathEnabled = YES` -- the default. That PERMITS reassociation, so
rearranging float arithmetic in the MSL hands the compiler something it has already done.
Factoring the matvec's bias term (`scale*dot - (biasK*scale)*sum` into `scale*(dot - biasK*sum)`,
two instructions and two roundings fewer) measured EXACTLY ZERO, ABBA on the M4 at split=1:
q/o 48 47 48 48 against 47 48 48 48, gate/up 70 71 71 71 against 69 71 70 71.

That is the line between it and 12j/12k, which paid 28% and 5.8%: those changed WHICH
INSTRUCTIONS EXIST -- an integer multiply became a float one -- and no amount of reassociation
reaches that. **A Metal win has to come from the operation set or the access pattern.** The change
was kept anyway because PTX and SPIR-V do not reassociate by default, so there it is real.

★ **AND "SPIR-V DOES NOT REASSOCIATE BY DEFAULT" TURNED OUT WRONG WHEN IT WAS
MEASURED.** This box's NVIDIA Vulkan driver rewrites `(a+b)-b` to `a`, which
deleted a rotary kernel's whole range reduction and read |d| 2.0 against ptx's
5.96e-08 on the same IR. See the rotary-table entry at the end of this file for
the probe. What survives here is the sentence about METAL, which is unchanged,
and the narrower claim that PTX did not reassociate this expression.

**★ AND THE PER-ELEMENT INSTRUCTION TABLE SAYS WHERE Q4_0 ACTUALLY GOES.** Both formats' loops run
64 iterations issuing 8 `Dot4` each; they differ in how many bytes those 32 elements cost:

    per 32 elements    wt loads   act loads   unpack ALU   Dot4   bytes read   instr/byte
    Q8_0                   8          8            0         8        34          1.0
    Q4_0                   4          8           12         8        18          2.5

**Q4_0 needs 2.4x the instructions to consume a byte**, which is the whole of why it sits at ~80%
of the Metal wall while Q8_0 sits at 110% and is plainly memory-bound. Inside that budget the
dominant term is not the nibble unpack (12 ops) but the SIXTEEN byte-to-float converts, two per
`jitllm_dot4f` -- and eight of those are the ACTIVATION side, which all 2048 row-threads recompute
identically. Pre-converting activations to float in the quantize kernel would delete them at the
cost of 4x the activation load width; unmeasured, and the trade is not obvious.

**★ 12m. ASK FOR THE OPERATION BY ITS GRAPHICS NAME. THE METAL MATVEC WAS SPENDING MOST OF ITS
ARITHMETIC WIDENING BYTES.** `float4(as_type<char4>(w))` is a generic vector convert, which Apple
issues as four byte extracts and four int-to-float converts PER OPERAND. `unpack_snorm4x8_to_float`
is the same widening expressed as the thing GPUs have had dedicated hardware for since vertex
attributes existed, and it is one instruction.

**The per-ELEMENT rate is what found it, and the per-byte table had been hiding it.** 12j's table
reads Q4_0 at ~80% of the wall and Q8_0 at 110%, which looks like Q4_0 being the inefficient one.
Converted to elements per second it is the opposite:

    Q8_0 lm_head   99 GB/s / 34 bytes per 32 elem  =  93 Gelem/s
    Q4_0 gate/up   73 GB/s / 18 bytes per 32 elem  = 130 Gelem/s

Both issue 8 `Dot4` per 32 elements, so they have the SAME arithmetic per element and Q8_0 was
merely hitting memory first. That put the shared arithmetic ceiling at ~130 Gelem/s against Q4_0's
~178 memory ceiling, bounding the prize at 1.37x and saying it was arithmetic before anything was
built. Measured after, GB/s and fraction of the M4's wall:

    k/v (GQA)     34  37%  ->  49  55%        tinyllama Metal   96.0 -> 110.9 tok/s
    q/o           58  64%  ->  66  74%        gemma-2b  Metal   46.1 ->  53.6 tok/s
    gate+up       74  81%  ->  88  98%
    gate/up       73  81%  ->  98 108%        vs llama.cpp b10825, ABBA, gated:
    ffn_down      72  80%  ->  94 105%          tinyllama  0.824 -> 0.9955  IQR 0.017
    lm_head Q8_0  98 109%  ->  98 109%          gemma      0.830 -> 0.9406  IQR 0.011

**The unchanged Q8_0 row is the control**: it was memory-bound already, so it could not move, and the
fact that it did not is what says the Q4_0 rows crossed from arithmetic-bound to memory-bound rather
than the whole device getting faster.

**IT IS STILL BIT-EXACT, AND THAT WAS CONFIRMED ON THE DEVICE RATHER THAN ARGUED.** The intrinsic
yields v/127, so the dot arrives scaled by 1/127^2 with a rounding per lane; multiplying by 16129
and `rint`ing recovers the integer exactly, because the true value is an integer at most 64516 and
the accumulated relative error is ~1e-7 -- nowhere near the 0.5 that would round to the wrong one.
`rint` measured FREE against the un-rounded form, since by then the kernel waits on memory.
`TestMatVecMatchesReference` holds its 1e-10 gate on the msl backend across every format, shape and
split.

**THE FACT GOES IN THE IR; THE WIDENING DOES NOT.** snorm is `clamp(v/127, -1, 1)`, so it needs that
no operand byte is -128. That is a property of the DATA, so `ir.Dot4Bounded` asserts it and PTX and
SPIR-V ignore it -- `dp4a` and `OpSDot` take the full signed range at no charge. Putting Apple's
preferred widening in the shared IR would have been encoding one target's ALU there, which is 12k's
lesson. Q8_0 is the only format whose weights reach -128 and the only one that gains nothing, so it
keeps the generic path; every other format masks its quants to a small unsigned range (Q4_0 0..15,
Q3_K 0..7, Q5_K 0..31, Q6_K 0..63) and the activations are scaled by 127/amax.

**BOTH CONTROLS OF THE INVARIANT TEST WERE WRONG AT FIRST**, which is the part worth keeping. A table
filled with 0xFF is -1, not -128, so a deliberately mislabelled Q8_0 PASSED; and checking the packed
bytes of a 4-bit format reported Q4_0 as broken, because the kernel masks those nibbles before the
dot ever sees them. A test for an invariant has to be run against a violation before it is worth
anything.

**★ 12n. AND THE K-QUANTS ARE NOT THE NEXT PROBLEM ON METAL.** `mvbench` was gemma-only, so every
Metal conclusion had been drawn from Q4_0 and Q8_0 while tinyllama is 53% Q3_K. Measured in ONE run,
so the Q4_0 rows are the control for the thermal state: Q4_K 81%, Q3_K 88%, Q4_K 89%, Q5_K 82%,
Q6_K 84%, against Q4_0's 69-98% in the same run. Same band -- so after 12m there is no
format-specific deficit left, and tinyllama Metal's remaining 0.45% is not in the matvec.

**★ 12o. THE DEVICE READS 23% MORE BYTES THAN THE FILE, AND THAT DEBT IS WORTH PAYING -- MEASURED,
NOT ASSUMED.** `PackWeights` transposes for coalescing (free) and NORMALISES every k-quant payload to
a nibble or a byte (not free). `pack.go` has always listed the cost and called it known debt:

    format  GGUF B/256   device B/256   delta
    Q4_0        144          144          0%
    Q8_0        272          288         +6%
    Q4_K        144          192        +33%
    Q3_K        110          192        +75%
    Q5_K        176          320        +82%
    Q6_K        210          320        +52%

For tinyllama that is 0.60 GiB resident against ~0.49 GiB of weights -- **1.23x the bytes, read once
per token, on a tier whose binding constraint is bandwidth.** gemma is Q4_0+Q8_0 and barely pays it;
every k-quant model pays it in full, which is one reason tinyllama Metal sits at 79% of the wall
while its own kernels measure 81-89%.

**WHETHER SHRINKING IT PAYS DEPENDS ON A SLOPE, AND THE SLOPE COST ONE LINE TO MEASURE.** Q3_K's
values are 0..7, so `bits: 4 -> 8` in `qtab` is a correct alternative encoding that DOUBLES the bytes
and REMOVES the nibble unpack -- the same trade as native packing, run backwards. On the M4, same
shape (5632 x 2048), exact buffer sizes:

    bits=4   6.58 MB   146 B/256 elem    83.8 us
    bits=8  12.35 MB   274 B/256        140.3 us
    bytes x1.877  ->  time x1.674

**77% of a byte change becomes time.** My first reading of that was "so shrinking the payload pays,
roughly -15% on Q3_K" -- and that was WRONG, because it prices the byte side of native packing and
forgets the arithmetic side. **PRICED BOTH WAYS, THE DEBT DID NOT PAY.**

Two measured points fit `time = a*bytes + b*ops` (ops = loads + unpack, per 256 elements):

    bits=8   274 B    64 loads +   0 unpack   140.3 us
    bits=4   146 B    32 loads +  96 unpack    83.8 us   ->  a = 0.490, b = 0.0953

Native 3-bit is a 2-bit plane (1 word per 16) plus a 1-bit plane shared by two sub-blocks, which is
114 B -- but every 4 values then costs shift+mask for the low bits, shift+mask for the high bit and
an OR, so 24 loads and **320 unpack** ops:

    3-bit    114 B    24 loads + 320 unpack    88.6 us   predicted, 6% WORSE than bits=4

The same fit prices the other two: Q6_K native saves ~7% and is 10.6% of tinyllama's device bytes
(0.6% of the token), Q5_K saves ~18% and is 4% (0.6%). Q4_K gains nothing at all -- its nibble IS
native -- and it is 28% of the bytes.

**So the payload normalisation is roughly the RIGHT call on this hardware, and `pack.go`'s "known
debt" framing oversells it.** Sub-nibble packing cannot be extracted with the one-mask
`(w>>s) & 0x0F0F0F0F` shape every current format uses; it forces a plane split, and the split costs
more than the bytes save. The lesson is the pricing method, not the answer: **a byte-versus-op trade
needs BOTH coefficients, and one A/B in the cheap direction gives both.**

**★ 12p. AND THE LAST METAL LEVER IS THE ONE THE INTERFACE SAYS IS NOT NEEDED: 8% OF tinyllama's
TOKEN IS SPENT ENCODING LAUNCHES.** `backend.Recorder` is documented as CUDA-only because "Vulkan
re-records its command buffer every submission and Metal builds a fresh command buffer, so for them
a recording would be a copy of the work they already do". That is true and it is not the same as
free. A decode block issues ~30 dispatches and tinyllama runs 22 of them, so ~700 launches are
re-encoded EVERY token.

`Tier.TEmit` now times the encode separately from the wait, and `jitllm verify` prints it:

    tinyllama  Metal   0.86 ms/token encoding   8% of Layers
    gemma-2b   Metal   0.72 ms/token            3%

**That is the exact ceiling on a Metal recording** -- an indirect command buffer removes the
ENCODING and nothing else, so no argument about GPU-side dispatch cost is needed to bound it. 8% on
tinyllama takes 0.9955 to ~1.075, which is well clear of that comparison's 1.7% IQR; 3% on gemma
takes 0.9406 to ~0.97, which is not a win.

The reason it is bigger on tinyllama is arithmetic, not a defect: gemma reads 1.67 GB per token
against tinyllama's 0.64 GB, so the same fixed encode is a larger share of a shorter token. Any
model with many small layers pays it hardest.

**AND THEN TWO THIRDS OF IT WENT WITHOUT AN ICB AT ALL -- SEE 12q.** That matters for how the
number should have been read: 8% was the ceiling on removing the ENCODING, and an indirect command
buffer is only one way to remove encoding. The cheaper way was to stop doing so much of it per
dispatch. What is left of the idea is a 3% ceiling and a real hazard:
`MTLIndirectCommandBuffer`'s only compute command is `MTLIndirectCommandTypeConcurrentDispatch`,
while every kernel in a jitllm block consumes the previous one's output, so whether
`executeCommandsInBuffer` on a `MTLDispatchTypeSerial` encoder serialises them would have to be
established on the device first -- being wrong is a race that produces wrong tokens, not an error.
Not worth it now.

**★ 12q. THE FIRST MAC WIN CAME FROM THE HOST SIDE, AND EVERY KERNEL IS BYTE-FOR-BYTE UNCHANGED.**
The Metal encode path sent ~9 Objective-C messages per dispatch and looked up a selector before
EVERY ONE of them -- `sel()` allocates a NUL-terminated copy and calls `sel_registerName`, so each
dispatch cost ~16 FFI calls and ~8 Go allocations, ~700 times a token.

    the hot selectors are resolved once at load     -- they are process-wide constants
    setBuffers:offsets:withRange: binds all six     -- one message, not one per buffer

    encode per token   tinyllama  0.86 -> 0.34 ms       gemma  0.72 -> 0.29 ms

    vs llama.cpp b10825, M4, via scripts/vs-llamacpp.sh (n=320, ABBA, gated):
      tinyllama Metal  0.824 -> 1.021    ★ THE FIRST MAC WIN
      gemma-2b  Metal  0.830 -> 0.9440   IQR 0.006

    ★ QUOTE 1.021, NOT A SINGLE BEST RUN. Five gated passes read 1.0253, 1.0211,
    1.0238, 1.0121 and 1.0201 -- median 1.021, every one above parity, and the
    spread is real rather than thermal: llama.cpp holds 113.5-113.9 across all of
    them while jitllm swings 110.8-116.7. That is RULE 1d-quater's "jitllm degrades more
    under sustained load" showing up inside a single comparison, and it is why the
    IQR runs 0.008-0.031 between passes. The win is not in doubt; its MARGIN is
    about two points, not the two-and-a-half the best pass shows.

**★ AND THE FIRST NUMBERS I QUOTED FOR THIS WERE 1.0499 AND 1.0552, WHICH IS RULE 1d-quater CATCHING
ME.** Those came from an ad-hoc script at n=64, because that is what `llama-bench` defaults to and it
was the quick thing to write. Short samples flatter jitllm by about 2.5 points here -- llama-bench
discards a warm-up pass and `jitllm run` measures every token from process start -- so the comparable
figure is 1.024, not 1.05. Three passes at n=320 read 1.0253, 1.0211 and 1.0238; the win is real and
it is two points, not five.

**The lesson is to use `scripts/vs-llamacpp.sh` on BOTH hosts.** It already had a Darwin branch and I
wrote a second harness beside it, which is how the sample length silently diverged from the one every
Linux row in this file uses. A scoreboard whose rows come from two different tools is not a
scoreboard. (The clamp gate of RULE 2d correctly no-ops there: macOS has no /proc/cpuinfo, it prints
MHz 0 and discards nothing.)

**Three FFI calls per dispatch now against sixteen, and the measured encode fell only 2.5x rather
than 5x** -- the remaining 486 ns per dispatch is about the goffi floor, so further gains need FEWER
DISPATCHES, not cheaper ones.

Two things worth keeping. **RULE 12j's advice to audit the MSL lowerings was right and incomplete:
the tier is a Go program calling a C ABI, and that side had never been counted at all.** A profile
of the device would never have shown this. And **the arithmetic pointed straight at it**: ~700
launches x ~9 messages x ~135 ns is 0.85 ms, against a measured 0.86 -- when a modelled cost matches
a measurement to two digits, the model is naming the right thing.

**THE METAL TIER IS NOT RUN-TO-RUN DETERMINISTIC, WHICH IS WORTH KNOWING BEFORE READING A DIFF.**
tinyllama teacher-forces at 1-2/32 with margins of 0.010-0.061 against max|dlogit| 0.28-0.55, and
the COUNT AND THE MARGINS VARY BETWEEN RUNS OF ONE BINARY: the split tuner measures at runtime, so
the summation order of a matvec is not fixed. Do not read a changed tie count as a changed kernel;
run it three times first. gemma is 0/32.

**★ AND IT IS NOT METAL-ONLY. CUDA DOES THE SAME, WHICH THIS RULE'S WORDING HID.** Verifying an
unrelated tier fix, llama-2-7b on PTX read 0,1,1,1,1 across five runs of ONE binary and Phi-3.5 read
0,1,1,2,2, with margins of 0.002-0.06 against max|dlogit| 0.15-0.5. The mechanism is the same -- the
split tuner measures at runtime, so a matvec's summation order is not fixed -- and it has nothing to
do with Metal. tinyllama, gemma and qwen3-1.7B are 0/32 on every run and are the models to use as
controls; llama-2-7b and phi3 have argmax ties inside f32 noise and cannot serve as one.

**So a teacher-forced count is a GATE ON THE CLEAN MODELS and not a metric on the tie-prone ones.**

**★ 12u. THE PREFILL UNIT IS A CHUNK NOW, NOT A TOKEN, AND THAT IS RULE 12's OWN
FINDING ONE LEVEL UP.** RULE 12 moved the device's unit from one MATVEC to one
BLOCK. `engine/model/prefill.go` still fell back to per-token `Forward` whenever any
block was resident, so a 507-token prompt was 507 single-token submissions and
jitllm's device PREFILL rate was its device DECODE rate. `Layers(lo, hi, pos, n,
...)` takes n rows; every kernel in the block body grew a row dimension and every
matvec got a batched twin, built lazily on the first prompt that asks for one
because the scratch is ~85 MiB of score and probability buffers.

  - **THE MASK IS -inf, WHICH IS WHAT LETS THE OTHER TWO KERNELS STAY UNIFORM.**
    Query i of a chunk attends to i+1 more keys than query 0, and this IR has no
    conditional outside a loop -- so the scores kernel masks past each row's
    causal count and exp(-inf) is exactly zero. `pN` grows a header for it:
    element 0 is the uniform width, 1..rows are the per-row counts.
  - **THE DEVICE PICKS ITS OWN CHUNK WIDTH AND IT IS NOT THE TUNED ONE.** A
    batched submission runs ONE fixed grid, so a narrower chunk does not run less
    work, it runs the same work with the surplus rows zeroed: the tuner's 32
    measured 150 tok/s against 496 at 128. **A tuner tuned for the host's cost
    model is a cost the device does not have.**
  - **THE RAGGED TAIL'S SURPLUS ROWS WRITE K AND V TO A TRASH SLOT.** The first
    version gave the KV cache a whole chunk-width of slack instead, and that cost
    Phi-3.5 every one of its blocks on a 4 GB card. One extra position costs
    nothing and nothing ever reads it.

**★ 12u-bis. THE GATE IS BIT EQUALITY, AND THE FIRST TWO ATTEMPTS AT IT WERE
NUMBERS NOBODY COULD CALL RIGHT OR WRONG.** `jitllm verify -prefill` diffs batched
against per-token device prefill and then keeps decoding, teacher-forced from a
fixed stream, because a chunk returns ONE row of logits and writes n rows of KV.

    against the CPU tier      fired on BOTH arms: f32 against f64 on a random
                              token prompt sits on argmax ties, 1-3 of 17 either
                              way (RULE 12q). A comparison whose control fails
                              as often as its subject measures nothing.
    random ids                off-distribution, max|dlogit| up to 2.5. Real prose
                              from the model's own tokenizer sits on wide margins.
    unpinned splits           still ~0.5 apart, and it is not batching: a decode
                              matvec is split across segments and the weighted sum
                              across positions, and a batched launch does neither.
    splits pinned to 1        BIT-IDENTICAL. max|dlogit| exactly 0.

**So the gate is equality, and the control -- the same arm run twice -- is 0.000,
which is what makes 0.000 mean something.** tinyllama, gemma, qwen3, llama-3.2,
phi3 (partial seam) and qwen3moe all pass. Run against four violations per RULE
12m: no causal mask 15.7, RoPE-q at row 0's destination 18.6, padded rows writing
K/V at their nominal positions 6.0e9, and see 12v for the NaN one.

**★ 12v. Tok ALONE PLATEAUS, AND THE NUMBER THAT MATTERS IS LOADS PER FMA.** The
batched matvec gives one thread Tok token columns so a weight word is loaded once
and used Tok times. It stops improving at ~1.35 Tmac/s and does not care what Tok
is, because a thread owning ONE ROW reads Tok activation words for every weight
word: the load:dot4 ratio is 1:1 whatever Tok is. `mvbench -ntok 128` says it in
one line -- q/o reaches the memory wall at Tok=4 and then stops improving while
its weight traffic falls fourfold.

`Rowt` rows per thread share one set of activation loads. Gmac/s at the best Tok:

    Rowt=            1      2      4      8
      q/o          1345   2136   3194   2416
      qkv fused    1478   2443   4304   2856
      gate/up                    4015
      tll Q3_K                   3296

8 collapses -- 4x32 accumulators plus their integer partners do not fit. The rows
a thread owns are STRIDED by Rows/Rowt, not adjacent, because pack.go writes
element (word,row) at word*nrows+row and consecutive threads have to stay on
consecutive rows or the transposed layout stops coalescing. The tile offset is a
compile-time immediate on the load, so Rowt == 1 emits what it always did.

**★ AND 2.4-3x ON THE KERNEL WAS 1.22x ON THE MODEL, WHICH IS THE SEVENTH TIME IN
THIS FILE.** The difference is not thermal and not the thread count: the matvec
had stopped being the cost. `JITLLM_GPU_SKIP` omits a phase from the submission and
produces wrong tokens on purpose -- there is no per-kernel profile of a device
submission that does not serialise the stream, and serialising it would measure
the serialisation.

**★ 12v-bis. AND THE BATCHED ATTENTION WAS READING KV THE PROMPT HAD NOT WRITTEN.
THE PROBABILITIES BEING ZERO SAYS NOTHING ABOUT THE OTHER OPERAND.** AttnAcc
walked the UNIFORM width for every row, on the reasoning that the masked
probabilities are exactly zero. They are. The V they multiply is not: for a
ragged last chunk it is memory nothing has written, and **0 * NaN is NaN**.

`LoopN` already takes a runtime trip count, so each row reads pN[1+row] and stops
at its own causal bound -- the correctness fix and the faster shape at once.
`JITLLM_GPU_POISON_KV=1` fills the cache with NaN at PrepLayer, which is the ONLY
way to test this:

    uniform width + poisoned KV   5/9 disagree, max|dlogit| 17.9
    uniform width + clean KV      0/9, max|dlogit| 0          <- why it hid
    per-row bound + poisoned KV   0/9, max|dlogit| 0

**A fresh device buffer is usually zero, so this class of bug is invisible until
it is not. Poison the buffer, do not trust the allocator.**

★ AND THE FIRST VERSION OF THE PER-ROW BOUND WAS AN ILLEGAL ACCESS ON ONE MODEL
ONLY. A padded row's nominal count runs past the END of the cache, so it faults
on the model with the widest kvDim and lands inside a neighbouring allocation on
every other one: Phi-3.5 (kvDim 3072) died with CUresult=700 where tinyllama
(kvDim 256) read someone else's buffer and PASSED ITS GATE.

★ NO GRAPH CAPTURE FOR A BATCHED CHUNK. The graph key carries the score width and
a chunk's width steps with its position, so every chunk got a fresh key: a
capture per chunk, used once, strictly worse than having graphs off.

**★ 12w. THE SCORES KERNEL WAS 88% OF BATCHED ATTENTION, AND IT IS THE SAME
DEFECT AS 12v IN ITS WORST FORM: TWO LOADS PER FMA.** An untiled thread loads 64
q floats and 64 k floats to produce 64 FMAs and ONE score. `AttnScoresTiled`
gives a thread qt query rows and kt key positions, so loads per score fall from
128 to 64*(qt+kt)/(qt*kt).

    qt x kt   1x1   2x2   2x4   4x4   4x2   8x4   4x8
    tok/s     697   913   829   933   982   860   813

**4x2, AND THE ASYMMETRY IS NOT NOISE.** The QUERY side is a broadcast -- every
key thread of a row reads the same q -- while the KEY side is a strided walk, so
widening q costs registers and widening k costs coalescing. **The key positions a
thread owns are STRIDED, not adjacent**, for the same reason Rowt's rows are: K
is [position][kv head][dim], so adjacent positions are kvDim*4 apart and adjacent
threads share a sector run.

★ AND THE SCORE ROW HAD TO BE PADDED, WHICH WOULD OTHERWISE HAVE BEEN A SILENT
WRONG ANSWER. ceil(cap/kt)*kt can exceed cap by up to kt-1, so the last tile
writes past the causal width -- and with an unpadded stride that lands in the
NEXT row's scores. Not a fault, a wrong number.

**★ WHERE THE PREFILL TIME ACTUALLY GOES, AT EACH STAGE. THIS TABLE IS WORTH MORE
THAN ANY OF THE FIXES, BECAUSE EACH TERM ONLY BECAME VISIBLE ONCE THE ONE BEFORE
IT WAS GONE.**

    phase                at 607 tok/s   at 1050
      scores                    326 ms     99 ms
      softmax                     3
      attnAcc                    38        49
      attention total           401       163      53% of the prompt -> 31%
      FFN projections           243      ~230
      everything else           132       132
      whole prompt              734       527

So the FFN projections are the largest single term again. **The durable finding
is not any one tile width: it is that on this device, a kernel issuing one
operand load per multiply is issue-bound before anything else is true. Count
loads per FMA first.**

**★ 12x. WHERE THE REMAINING 4x IS, MEASURED RATHER THAN GUESSED: llama.cpp RUNS
PREFILL ON INT8 TENSOR CORES AND jitllm RUNS IT ON dp4a.** `cuobjdump --dump-sass`
on this box's `libggml-cuda.so`:

    158,714  IMMA.8816.S8.S8      <- int8 tensor-core MMA
    167,712  HMMA.16816.F32
    463,877  LDS                  <- shared-memory staging
    217,968  LDSM.16.M88.4

jitllm emits `IDP4A` and no MMA at all: the IR has no matrix op. On GA10x the int8
tensor rate is about 4x the INT32 dp4a rate, which is the same factor as the
remaining gap -- but **that does NOT make MMA the required fix**, and the
arithmetic is worth doing before anyone starts one:

    dp4a peak on this card (64 INT32 lanes/SM)        7.68 Tmac/s
    jitllm's batched matvec, in model                    2.15   28% of it
    jitllm's batched matvec, isolated (mvbench)          3.30   43%
    what 5000 tok/s of prompt needs                   4.84   63%

**63% of dp4a peak is reachable without a tensor core**; jitllm is at 28% because
its reuse is entirely in registers and llama.cpp's is in shared memory as well.
So the next structural lever is a COOPERATIVE tile -- a CTA staging one operand
in shared memory so the reuse is 128 threads wide instead of Rowt*Tok -- and MMA
is the one after it, not instead of it. The IR has had `OpShared` and `OpBarrier`
since RULE 12s and the matvec does not use them.

**★ AND THE PHASE TABLE SAYS THE MATVEC IS ONLY 62% OF WHAT IS LEFT**, so even a
perfect one lands at 508/300 = 1.7x, or 0.41 of llama.cpp. Both halves are needed:

    FFN projections   226 ms   47%      q/k/v/o     74 ms   15%
    attention         166      34%      elementwise 54      11%

**★ THE K-QUANT FOLD IS NOT THE LIMITER, AND ONE CONTROL KILLED A DAY OF WORK.**
Counting the emitted IR for Q3_K says the per-sub-block float fold -- convert the
integer accumulator, multiply by the scale, fma with the activation scale, once
per (row, token, SUB-BLOCK) -- is about 45% as many operations as the dot
products it serves, and Q3_K's sub-blocks are 16 elements where Q4_0's are 32, so
it pays it TWICE as often. RULE 6b did exactly this fold in integer on the CPU
and was worth 3.7%. The obvious next move was to port it.

The control says no. gemma-2b is Q4_0 + Q8_0 and contains NO k-quants at all:

    tinyllama q3_K_M   0.2431   IQR/med 0.047
    gemma-2b           0.2678   IQR/med 0.032

**10% apart.** A model that cannot pay the k-quant fold is barely faster than one
that is 53% Q3_K, so whatever binds is common to both -- the GEMM's structure,
not its format. So porting the integer fold to the device was not expected to buy a row.

**★ AND SHARED-MEMORY STAGING OF THE ACTIVATIONS ALONE WILL NOT DO IT EITHER,
which is the review's point and it is right: it replaces a global load with a shared
load and adds a barrier. Both issue on the LSU.** What LDSM buys llama.cpp is
that ONE instruction fills a whole MMA fragment; without a matrix instruction to
feed, moving the operand to shared memory moves the cost, it does not remove it.

**★ 12ad. ONE MODEL THAT WAS CLEAN, CONFIDENT AND WRONG, recorded because the
reasoning looked airtight.** (This paragraph is what commit 5167a5f calls RULE
12ad; it had no heading of its own, so "RULES 12r-12ae" pointed at a rule that
could not be found. Numbered afterwards.) NormApply re-reads ALL the RMSNorm partials for EVERY element,
so its read count is rows*NEmbd*parts -- 131k at one row and 16.7 MILLION at 128
rows and 64 parts, 67 MB twice a layer, against a measured 39 ms for the whole
norm. Fewer parts therefore had to be faster. Swept:

    parts     1     2     4     8    16    64
    ms      567   529   516   532   519   494

The re-read is cached; the REDUCTION's thread count is not, and at parts=1
normPart runs 128 threads on a card with 30,000 slots. **A traffic argument that
ignores the cache is not a roofline.**

**★ 12ae. fp16 TENSOR CORES FOR THE ATTENTION SCORES, AND THE TRANSPOSED CACHE
MADE IT CHEAP BY ACCIDENT.** q.k is a matrix product and jitllm computed it with
plain FMA; llama.cpp emits 167,712 HMMA beside its 158,714 IMMA.

    SASS, tinyllama's shipping shape
      FMA tile (qt=4 kt=2)   1080 instrs   389 LDG   512 FFMA   4.2 instr/score
      m16n8k16 f16            264           53         4 HMMA   2.1 instr/score

    1920-token prompt, six pairs, every pair agreeing
      FMA 1.393 s -> tensor 1.336 (1.042x), and attnNT 2 over 1 a further 1.029x

**mma WANTS B COLUMN-MAJOR -- column = key position, row = head element -- AND
RULE 12aa HAD ALREADY STORED K THAT WAY FOR COALESCING.** The layout that made
the FMA kernel fast is the layout the matrix instruction wants. That is twice
now that a decision taken for one reason paid for the next one, and both times
because the JIT let the layout be chosen at load rather than baked into a
distributed kernel.

**★ AND IT IS THE FIRST ARITHMETIC IN THIS ENGINE THAT CANNOT BE BIT-IDENTICAL.**
binary16 operands into a float32 accumulator; there is no full-precision matrix
instruction on this hardware (tf32 is also 11 bits), so any tensor-core attention
makes the trade. **The gates moved with it rather than being relaxed:**

  - kernel level, NMSE < 1e-5 against the FMA kernel on five real shapes. The
    bar is derived, not fitted: binary16 gives ~5e-4 per product and ~2.5e-7
    NMSE on a 64-element dot, so 1e-5 is two orders of margin and still orders
    below a transposed fragment at 1e-1.
  - the MASK is checked SEPARATELY, because the NMSE loop cannot see it: it
    compares only positions INSIDE each row's causal count, so dropping the
    per-row mask changed nothing it looked at. Everything past a row's bound
    must be -inf, and that is now its own assertion.
  - end to end, **a flipped token id is judged BY ITS MARGIN**. A flip where the
    winning margin is narrower than the whole perturbation is the model being
    undecided; a flip on a WIDE margin is a defect rounding cannot explain.
    qwen3 flips one id of seven at 384 tokens on a 0.183 margin against a 1.67
    perturbation. Zero wide-margin flips on six models.

★ FOUR VIOLATIONS. The f16 A fragment read at the INT8 spacing fails the layout
probe at 64/128 elements -- a lane holds two halves per register, not four bytes,
so where s8 gives it four consecutive k, f16 gives two at 2t and two at 2t+8, and
reading the s8 table does not fault, it transposes a block. The A pair at stride
2 gives NMSE 0.997; dropping the causal mask trips the -inf check; the B pair at
stride 1 instead of kStride moves a logit by 29.8.

★ A REVIEW FOUND THE NaN AND THE GRAPH THRASH FROM THE SOURCE, and corrected the
roofline the whole thing was being judged against: GA10x has 64 INT32 lanes per
SM, not 128, so DP4A peak is ~7.68 Tmac/s and 4.3 is 57% of it rather than 28%.
It also caught `mvbench`'s k-quant bandwidth column being wrong by 8x --
`k/32*BlockBytes()` where BlockBytes is the SOURCE block, 256 elements for a
k-quant against 32 for Q4_0, so Q3_K read "2306 GB/s" against a 150 GB/s wall.
**A number that far above a wall measured in the same process is a units error,
not a cache effect**, and it had been on screen for an hour.

**★ 12y. INT8 TENSOR CORES. `ir.OpMMA` IS THE FIRST OP jitllm GENERATES THAT A LANE
CANNOT EXECUTE ALONE, AND THAT IS WHY IT IS IR.** dp4a, SDOT and OpSDotKHR are
per-lane and interchangeable; a matrix instruction is a CONTRACT BETWEEN LANES
about which element each one holds, and the kernel has to know that contract to
build a fragment at all. So the shape is IR and the encoding is the lowerer's.
`Op` grew a `Vargs` slice: MMA has up to ten operands and four results.

**★ AND jitllm NEEDS NO SHARED-MEMORY STAGING TO FEED IT, WHICH IS THE WHOLE POINT
AND IS WHAT THE JIT BUYS.** llama.cpp stages weight tiles into shared memory and
reorders them with `ldmatrix` -- 463,877 LDS and 217,968 LDSM in its
disassembly -- because its kernels are compiled ahead of time against one fixed
weight layout. jitllm picks the packing at LOAD time for a kernel it has not
generated yet, so the packed layout can simply BE the fragment layout. It
already was:

    PackWeights writes qs[(si*words+w)*nrows + r], and for a 4-bit format word w
    holds elements {4w..4w+3} in the LOW nibbles and {sub/2+4w..} in the HIGH
    ones. mma's A fragment wants lane (g,t) to hold A[g][4t..4t+3] and
    A[g][16+4t..]. Those are the two nibble halves of ONE packed word at w=t,
    row=g. Two loads and four unpacks give a whole 16x32 fragment, and across
    the warp they are four runs of eight consecutive u32: 128 bytes in 4
    sectors, perfectly coalesced.

**THE K OF THE TILE IS THE FORMAT'S SUB-BLOCK, NOT A FREE PARAMETER.** Each MMA
produces one complete integer dot product and the per-sub-block scale multiplies
exactly that, so 16-element sub-blocks take m16n8k16 and 32-element ones
m16n8k32. Running k=32 across two sub-blocks with different scales assembles
cleanly and is wrong.

★ THE FRAGMENT LAYOUT IS GATED ON HARDWARE BEFORE ANY KERNEL DEPENDS ON IT.
`kernels.MMAProbe` is a bare int8 GEMM with A and B in NATURAL order in memory,
so the kernel does the gather and the test is not testing itself. It runs with a
RAMP as well as random data, because a transposed output is invisible on a
symmetric matrix. Run against a violation -- swapping the "+8 row" and "+16 k"
roles -- it fails at 64/128 elements.

★ AND THE MATVEC IS BIT-IDENTICAL TO THE dp4a ONE, which is the only sharp gate
available: both reduce each sub-block to an exact int32 dot (integer addition is
associative, so the tensor core's summation order cannot matter) and then apply
the same float fold. 36/36 subtests exact across all six formats, and
`jitllm verify -prefill` is 0/9 at max|dlogit| 0 on six models.

★ METAL AND VULKAN DECLINE BY NAME AND `lowertest` ASSERTS THE REFUSAL. MSL's
simdgroup_matrix is float only; Vulkan has OpCooperativeMatrixMulAddKHR but it is
an extension AND its fragment layout is deliberately opaque, so a kernel cannot
build a fragment with ordinary loads -- that needs a different kernel, not a
different lowering. Both reach tier as a Compile error and run the dp4a kernel.

**★ 12z. AND THEN THE SASS SAID THE KERNEL WAS 41% ADDRESS ARITHMETIC AND 2%
MATRIX INSTRUCTION.** cuobjdump on jitllm's own Q3_K prefill kernel:

    216 LEA   128 FFMA   97 IADD3   72 I2FP   72 FMUL   64 STG   44 LDG ...
    and SIXTEEN IMMA, out of 896 instructions.

**EVERY Load WIDENS ITS OWN 64-BIT ADDRESS.** `ir.Load` lowers to mad.wide.u32
plus ld, and a mad.wide is two SASS instructions; with eight different base
VALUES the kernel paid that eight times. But the rows a lane holds differ by
i*16 + h*8, the n-tiles by j*8*(K/4), the output columns by (j*8+c)*Rows -- every
one a COMPILE-TIME CONSTANT. So they are ONE base register and an immediate
displacement, which is the offset field Load already has.

    896 -> 592 instructions, 349 -> 99 address, same 16 IMMA
    554-token prompt: dp4a 495 ms, MMA before 430, MMA after 390

**Induction variables alone did NOT do it** (349 -> 276): ptxas reorders freely,
so emission order is a hint and the offset field is the lever. **The dp4a kernel
next to it has 20 LEA against 112 loads for exactly this reason.**

**★ AND THE INTEGER FOLD WAS BUILT, MEASURED AND REMOVED, WHICH IS THE MORE
USEFUL RESULT.** RULE 6b does it on the CPU for +3.7%. Built and gated here it
measured +1.7%; after the address fix the SAME change measured -25%:

                       before the address fix   after
      float fold             428 ms              390 ms
      integer fold           421                 489

The reason is REGISTERS: the integer partial is a second accumulator array, 64
int32 beside 64 float, and ptxas reports 173 registers against 127 -- 10 warps
to an SM instead of 16. Dynamically it issues ~25% FEWER fold instructions and
loses anyway. **On this kernel OCCUPANCY BEATS INSTRUCTION COUNT: price a change
in registers before pricing it in operations.** And RULE 5t for the second time:
an idea priced against an unoptimised baseline can flip sign once the baseline
improves -- this one flipped by 27 points.

**★ 12aa. THE DEVICE K CACHE IS TRANSPOSED, AND THE JIT IS WHY THAT IS CHEAP.**
Attention reads K with one thread per POSITION, so at [position][head][dim] the
32 lanes of a warp want addresses kvDim*4 apart -- 1024 bytes on tinyllama, 32
separate sectors to deliver 128 useful bytes. The SASS agreed: the scores kernel
was 1080 instructions of which 397 were loads.

    554-token prompt, three rounds, alternating
      [position][head][dim]   463  399  401 ms
      [head][dim][position]   369  318  330 ms      1.24x
      decode                   56   59   57  ->  55  57  60   unchanged

**IT IS ONE OFFSET EXPRESSION.** The rotation writes the cache and attention
reads it, and both are generated from the same session's decision --
`RoPERowsT` takes a destination stride and `AttnScoresT` the matching read
stride. llama.cpp cannot do this, which is why it stages tiles through shared
memory instead. **V is NOT transposed**, for the mirror reason: AttnAcc reads it
one thread per DIMENSION, already coalesced.

★ THE HOST'S CACHE STAYS ROW-MAJOR, so `MigrateKV` is where the two layouts meet
and it transposes on the way across. Copying straight would be a block attending
over a permutation of its own history -- the exact class of bug MigrateKV exists
to close (RULE 14m).

★ AND THE 2/24 THAT LOOKED LIKE A REGRESSION WAS THE SPLIT TUNER. Teacher-forced,
tinyllama read 0/24 before and 2/24 after -- and with the splits pinned both arms
read 1/24 at max|dlogit| 0.465, three runs each, identical. The transpose changes
an ADDRESS, not a summation order; the tuner's timing changed and it picked a
different split. RULE 12q as written.

**★ 12ab. A QUERY-ROW TILE ON AttnAcc, AND THE PADDED ROWS BROKE IT AT 8 AND NOT
AT 4.** V does not depend on the query row, so eight rows sharing a thread turn
eight V loads into one: 320 -> 303 ms, six pairs, every pair agreeing.

The tile walks the GROUP's widest causal count, so the softmax has to normalise
that far or the rows below read stale probabilities against real V -- which is
why `SoftmaxRows` takes the same tile and rounds its row index up to the group.

**And a padded row pinned at a count of ONE dragged that down**: a tile
straddling the padding boundary truncated every real row in it to a single key.
2-5 of 9 positions wrong on five models at max|dlogit| up to 59, at atile 8 and
not at 4, where the tiles happened to align with the prompt length. A padded row
now takes the LAST REAL ROW's count: in bounds, so the CUresult=700 that
motivated the clamp stays fixed, and never below its neighbours'.

**★ 12ac. AND A KERNEL SET PER CHUNK WIDTH, BECAUSE "EVERY KERNEL BAKES ITS
ELEMENT COUNT" IS NOT A REASON TO PAD -- jitllm GENERATES THEM.** A 42-row tail was
doing 128 rows of work for 42 rows of answer. 308 -> 300 ms, six pairs. Less than
the row count suggests because the tail sits at the HIGHEST position, so its
attention scales with pos+rows and shrinks 10% where its matvec halves.

The compiled kernels are the cache, not a pointer on the mv: a prompt asks for
two widths and one field could only remember one, which is what forced the
padding. Widths are multiples of 32, the matrix instruction's own grain. And it
is ELASTIC -- a scratch per width costs VRAM, so the narrow one is used only if
it allocates and the wide one pads otherwise.

**★ THE ROW, AND WHAT IS LEFT.** GPU prefill against llama.cpp, tinyllama:

    0.0307   per-token device prefill
    0.2431   batched chunk (12u)
    0.2885   int8 tensor cores + the address fix (12y, 12z)
    0.3410   transposed K cache (12aa)
    0.4059   AttnAcc query tile (12ab)
    0.4151   a kernel set per chunk width (12ac)   IQR/median 0.037, n=7
    0.4342   fp16 tensor cores for the scores (12ae)  -- SEE THE CAVEAT

★ QUOTE 0.4151 AS THE SOLID NUMBER. The 0.4342 pass had llama.cpp reading
4084-4232 where every other pass in this session put it at 4900-5360 -- a 17%
drop on ONE arm, which is the one thing ABBA cannot cancel and the IQR gate
cannot see (RULE 2d's mechanism, RULE 2c's warning). jitllm genuinely moved
1770 -> 1820 in the same pass, so the fp16 attention is worth about 3-4% and the
rest of that step is llama.cpp having a bad afternoon. Re-measure it rested.

Phase costs after all of it, marginal via JITLLM_GPU_SKIP: the matvecs are ~57% and
attention ~33%. The remaining structural levers are a CTA-cooperative weight
tile (reuse 128 threads wide instead of one warp's tile) and fp16 tensor cores
for attention, which llama.cpp also uses -- 167,712 HMMA beside its 158,714 IMMA.

## ★ RULE 12ah: THE SUBGROUP WIDTH IS A GUARANTEE NOW, NOT AN OBSERVATION -- AND THE PROBE THAT "SETTLED" IT PASSED WHILE THE KERNEL WAS BROKEN.

Measured on this laptop. Three Vulkan devices, one question: how many
lanes are in a subgroup?

    device                                   subgroupSize  min  max  size control
    NVIDIA RTX 3050 Ti (discrete)                      32   32   32  yes, compute
    Intel Iris Xe ADL GT2 (integrated)                 32    8   32  yes, compute
    llvmpipe (software)                                 8    8    8  yes, compute

**THE FIELD EVERYONE REACHES FOR IS THE WRONG ONE.** The Iris Xe reports
`VkPhysicalDeviceSubgroupProperties.subgroupSize = 32` and runs a compute shader
at 16. That field is the DEFAULT width; the range is 8..32 and the driver picks
per shader. A capability check written against it passes on the one device that
is broken.

**AND THE ON-DEVICE PROBE PASSED TOO.** `tier.pickSoftmax` launched the warp
softmax against a host softmax at load and kept the scalar kernel on
disagreement. On the Iris Xe it agreed at n=1,2,31,32,33,37 on every head, and
the same kernel object then produced NaN logits inside a real submission of two
or more blocks, differently from run to run. An out-of-subgroup shuffle is
UNDEFINED, not wrong-by-construction: an idle device answers it correctly. The
probe is not broken -- on llvmpipe, which is narrow DETERMINISTICALLY, it
correctly rejected the warp kernel -- it is necessary and never was sufficient.

**A GATE THAT PASSES WHEN IDLE AND MISSES UNDER LOAD IS WORSE THAN A GATE THAT
NEVER FIRES**, which is the RULE 10 shape this adds to the list.

**THE CLASS, FOUND BY SWEEPING THE IR RATHER THAN THE KERNELS.** Two ops can make
a kernel's answer depend on which lanes share a subgroup -- `OpShuffleXor` and
`OpMMA` -- and `ir.WidthDependent` is now the one definition of that. Emitting
either sets `Kernel.Lanes`, `ir.Validate` refuses a kernel that carries one
without the declaration (and a declaration with no such op), and
`lowertest.TestEveryWidthDependentKernelDeclaresItsWidth` sweeps every shipped
kernel: 16 of 63 declare a width -- `softmax`, `headnorm`, two `mmaprobe` shapes
and twelve `matvec-mma`. `OpShared`/`OpBarrier` are workgroup-scoped and carry
their own synchronisation, so they are not in the class.

**WHAT REPLACED THE OBSERVATION.** `VK_EXT_subgroup_size_control`, bound through
purego with every offset pinned in `layout_test.go`: the width is PINNED at
pipeline creation with
`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` + `REQUIRE_FULL_SUBGROUPS`,
and a width the device will not promise is an ERROR out of `Compile` before a
pipeline exists. The promise is `min == max`, or a pinnable compute stage -- never
`subgroupSize`.

    pinned (shipping)        NVIDIA 32   Iris 32   llvmpipe scalar
    JITLLM_VK_SUBGROUP=off   NVIDIA 32   Iris SCALAR   llvmpipe scalar

The `off` row is the gate: without pinning the two real devices part company,
which is what says the rule is a rule rather than a constant.

**BEFORE AND AFTER, SAME BINARY.** `TestHeadNorm` on the Iris Xe
(`JITLLM_VK_DEVICE=1`), which had never had a probe in front of it:

    JITLLM_VK_SUBGROUP=force (the old engine)   16x128  0.25807052850723267 vs 0.20433825063960112
                                                1x2048  0.45949432253837585 vs 0.23119054648891962
                                                8x64   -0.3122132420539856  vs -0.22799779786197663
                                                1x32   -0.9338255524635315  vs -0.7725589102201517
    pinned                                      all four shapes PASS, both arms

1x32's -0.9338255524635315 is exactly a 16-lane reduction of a 32-element row, to
seven digits, deterministically -- the fingerprint of the bug.

**AND THE WORKAROUND IS DELETED.** `scripts/three-way.sh` set
`JITLLM_GPU_SOFTMAX=scalar` and now sets nothing: 22 blocks of tinyllama split 7
on the NVIDIA card, 6 on the Iris Xe and 9 on the host, token ids identical to
`-devices cpu` in both passes and both negative controls, with the warp softmax
and the 32-lane head norm running on the iGPU. `--violate subgroup-width`
(`JITLLM_VK_SUBGROUP=force`) puts the bug back and fails 3 or 4 assertions --
and WHICH ones is itself the evidence. Over four consecutive runs the
load-bearing `ids` assertion failed 4/4 (8 of 8 passes: the ids collapse to
`[... 338 0 0 0 ...]`, NaN logits argmaxing to token 0), while "the two passes
disagree" failed only 2/4, because the collapse moves between position 6 and 7
from run to run. Do not quote "4 assertions" as the expected result: the gate to
read is `ids`, which is deterministic, and `repeat` is the non-determinism
showing up when it happens to.

**THE SECOND KERNEL HAD NO FALLBACK AT ALL.** `kernels.HeadNorm` (qwen3/olmoe
q/k norm) takes `lanes` now, like the softmax, and `headNormScalar` is its
thread-per-head twin -- registered in `lowertest` so both are assembled by ptxas,
spirv-val and Metal's compiler. Neither existed before; the wide one reached
hardware through `tier` alone.

**NO CODEGEN MOVED -- AND THE RATE MOVED A LOT.** `TestDenseKernelsUnchanged`
reported exactly two new kernels and zero changed hashes, so every existing
kernel's PTX, MSL and SPIR-V bytes are byte-identical: the pinning is pipeline
metadata, not codegen. On a device that already promised 32 (every CUDA and
Metal device, and the NVIDIA card through Vulkan) the selection, the kernel and
the emitted code are all unchanged.

★ BUT "THE BYTES DID NOT CHANGE" IS NOT "THE SPEED DID NOT CHANGE", AND AN
EARLIER DRAFT OF THIS SECTION SAID "COST: NOTHING THAT MOVES" ON THE STRENGTH OF
THE HASHES ALONE. That is RULE 6-0-ter's checklist skipped: hashes answer
*correctness of emission*, not *time*. Measured ABBA in one pass,
n=20 rounds, six pinned P-cores, tinyllama-1.1b-q3_K_M, `-depth 738 -n 64`, ONE
binary so all three arms are reachable without a rebuild:

    Iris Xe decode      tok/s    ratio vs pinned   correct?
    warp, width ASSUMED  36.97   0.6864  IQR 0.012   NO -- NaN logits
    scalar twins         36.96   0.6811  IQR 0.017   yes
    warp, width PINNED   54.13   1.0000              yes

**THE UNPINNED WARP KERNEL RAN AT EXACTLY THE SCALAR KERNEL'S SPEED** -- 36.97
against 36.96 tok/s -- so before this work there was no configuration of this
iGPU that was both correct and fast. The workaround cost nothing *relative to the
bug*, because the bug was equally slow; pinning the width is what makes the
shuffle worth emitting at all, and it is worth **1.46x of an iGPU decode token at
depth 738** and 1.11x at depth 0 (0.9005/0.9071 scalar-vs-pinned, two passes).
Prefill moves far less, 0.9478 twice, because prefill is matvec-bound and the
softmax row is short per token.

The mechanism is the Intel compiler's SIMD mode, not the kernel: without
`REQUIRE_FULL_SUBGROUPS` and a required size the driver is free to compile the
shader narrow, and a 32-lane butterfly in a 16-lane subgroup is both wrong and no
faster than one thread per head. The pre-work binary (HEAD 7ee1edd, built and
ABBA'd against this one in the same pass) reproduces the assumed-width row
exactly: 0.6949 / 0.6856 decode over two passes, 0.9472 / 0.9478 prefill.

**THE iGPU ROWS, RE-TAKEN UN-HANDICAPPED.** The numbers previously on record for
this device were taken in another session and are NOT carried across (RULE 1);
both arms were re-measured here in the same pass, same binary, `scripts/cap 8G`,
`taskset -c 0,2,4,6,8,10`, tinyllama-1.1b-q3_K_M, PSI `some avg10` under 0.6
throughout. A/A self-control FIRST: **0.9941 (IQR 0.029) at depth 0, 0.9980 (IQR
0.022) decode and 0.9997 (IQR 0.002) prefill at depth 738.**

    decode, -depth 0 -n 320        host tok/s   Iris tok/s   host/Iris   IQR/med
      pass 1                          74.05        59.66       1.2504     0.088
      pass 2                          76.13        60.05       1.2399     0.136  REJECTED
      pass 3 (rested)                 78.00        60.74       1.2808     0.053

    at depth 738, -n 64            host         Iris         CUDA
      prefill tok/s   pass 1        328.14       248.81       2016.42
                      pass 2        364.56       250.00       2096.89
      decode  tok/s   pass 1         68.80        52.54        151.47
                      pass 2         73.76        54.39        156.28

★ THE HOST ARM IS THE UNSTABLE ONE, AND IT IS STRUCTURAL RATHER THAN BAD LUCK.
Every iGPU-vs-iGPU comparison in this session gated tightly and repeated to
within 1% (IQR/median 0.001-0.064, both passes); every host-vs-device comparison
was 3-5x wider, one pass was rejected outright at 0.136, and the host prefill
arm moved 328 -> 365 tok/s between passes while the iGPU arm moved 248.8 ->
250.0. The two arms have different PACKAGE POWER profiles -- six P-cores flat out
against an iGPU sharing the same die and the same DDR5 controller -- so
alternating them is itself a thermal excitation, which is the one thing ABBA
cannot cancel (RULE 2c). **Quote host-vs-iGPU ratios to two significant figures
at best: 1.25-1.28 decode at depth 0, and do not read a 2% move as a change.**
The iGPU-vs-iGPU rows above carry the actual finding and they are the tight ones.

**AND THE iGPU STILL LOSES TO THE HOST, WHICH IS THE EXPECTED ANSWER.** Decode is
weight-traffic-bound and this device reads through the host's own DDR5 controller
out of the host's own weight budget, so it cannot win; the finding is that it
now loses by 1.25-1.28x instead of 1.39x, and that it is CORRECT while doing so.

**AND A NAME TABLE HAD FALLEN BEHIND AGAIN.** `ir.kindName` was missing `xor`,
so every kind after it printed as its NEIGHBOUR: the new width refusal reported
"op 5 (shared)" for a shuffle. A missing name in the MIDDLE is worse than a
missing one at the end, which is what the array's own comment had guarded
against. `TestKindNamesLineUp` checks the alignment now, not the length.

## ★ THE DECODE ROW TILE, RE-BUILT AS ADJACENT ROWS

**The earlier tile did not fail for being a bad idea; it failed for
having the wrong row mapping.** `tileOff(j)` returned `j*(Rows/Rowt)`, which
spread a thread's rows deliberately so adjacent LANES stayed on adjacent rows.
A lane then read ONE word of each of Rowt distant rows, and the tile bought
nothing but a shorter grid — which is why it needed `Split == 1` and why it
measured 49.41 / 36.96 / 21.05 / 9.36 tok/s at Rowt 1/2/4/8 on the M4 Metal
tier, time growing roughly in proportion to the tile.

**What the kernel was actually short of is BYTES PER LANE PER LOAD, and the
probe separates that from memory-level parallelism.** Same bytes, same grid,
a lane reading w consecutive words against w words one grid stride apart,
M4, 1 GiB working set (`kernels.StreamWide`/`StreamWideSpread`):

     4 B per lane                          64.6  65.2  65.9  66.1  66.4  68.0 GB/s
    32 B per lane, ADJACENT               106.3 106.5 106.7 106.8 106.8 107.0
    32 B per lane, one grid stride apart   65.8  66.9  68.4

The strided arm issues the SAME number of requests and is worth nothing. At a
256 MiB working set the same sweep moves only 91.8 → 96.9, which is why nobody
saw it: the effect is STREAMING-ONLY and every earlier look was inside cache.

**The container layout already supported the fix.** Weights are transposed at
upload to `qs[u*nrows + row]`, so the Rowt rows of an ADJACENT tile are Rowt
CONSECUTIVE u32 — the payload word, both scale planes, the bias and the output
all index through `tileOff`, so making it the identity and scaling the thread
index by Rowt is the whole change. **No IR op, no container version, no
repack.** That is the part worth keeping: the spread mapping was a kernel
choice, not a layout constraint, and the "MLX's literal qmv_fast shape needs
lanes along k" migration — a v16 bump and five readers — is not warranted.

**Split stops being an alternative and becomes a companion.** The tile buys
bytes per lane and divides the grid by Rowt; the split is exactly what puts
the grid back. `MatVec` refuses only `Experts > 1` with a tile now.

★ **NO RATE IS CLAIMED HERE.** The row belongs to a rested box with an A/A
self-control in front of it, and this entry is the correctness half.

★ **WHAT IT IS GATED BY, AND WHAT EACH VIOLATION READS.** Six violations, each
producing the stated failure; the two poison ones are the interesting half,
because a fresh device buffer is usually zero and an unwritten row then reads
as a plausible 0.0 (RULE 13's second invariant):

    base row not scaled by Rowt           row 515 NaN -- never written
    tile stride left spread               row 1 NaN
    output store left spread              row 1 NaN
    d plane read for tile row 0 only      NMSE 1.048e-02 (no hole: the bound's case)
    secondary plane not tiled             Q5_K 1.597 / 2.480, Q6_K 0.888 / 1.481,
                                          every SINGLE-plane format still exact
    tile refused beside a split           "reached NO matvec (112 plain)"

★ **AND ONE VIOLATION DOES NOT FIRE, WHICH IS RECORDED RATHER THAN HIDDEN.**
Removing `n /= m.rowt` from the launch — an oversized grid — PASSES, because
surplus threads are clamped to the last work item and recompute a tile that is
also computed correctly. RULE 13 licenses exactly that shape. It is wasteful,
not wrong, and no correctness gate can see it; only a rate can.

★ **THE TIER GATE PINS THE SPLIT TUNER, AND WITHOUT THAT IT IS NOT AN EQUALITY
GATE.** `tuneSplit` MEASURES at load, so the split — and the f32 reduction
order with it — varies per PROCESS; `three-way.sh` failed 1 run in 30 on
exactly that, at a 0.0017-logit margin. Pinned, the two arms take the same
fitted split and the only difference is which thread owns which row, so token
ids must agree EXACTLY rather than within a tie margin.

    Linux, CUDA + Vulkan, RTX 3050 Ti   16 blocks placed, 112 of 112 decode
                                        matvecs tiled, 112 of them split,
                                        16 ids identical
    M4, Metal                           the same three numbers, identically

★ **AND THE COUNTER HAD TO BE NEW.** `Stats.RowtTiles` cannot say the new
combination ran: a shape the tuner lands on `Split == 1` tiles without ever
reaching it. `Stats.RowtSplitTiles` counts tile-AND-split, and the gate also
asserts the CONTROL arm built zero tiles. This knob has that history — the
first version reached ZERO of 126 matvecs and read as a null result.

## ★ AND Q5_K ON THE MMA PATH THREW ITS FIFTH BIT AWAY — THE THIRD TIME A PLANE CHANGE MISSED mma.go

Found while running the gates above. Container v15 made Q5_K a 4+1
two-plane format at `sub == 32`. `mma.go` read a secondary plane in its
`sub == 16` arm ONLY — written when Q6_K was the only two-plane format — so
Q5_K fell into the first arm, which reads the nibble and never touches the hi
plane. CUDA prefill decoded five-bit weights as four-bit for every Q5_K
tensor; `matvec.go` beside it was correct the whole time.

`backend.TestMatVecMMAMatchesDot4` had been RED for it since v15:

    ptx/Q5_K/128x512/ntok8/4x1/win256
      out[tok 0][row 0]: mma 188.00941, dp4a -288.7448
      1024/1024 elements differ, worst |delta| 1563.26
      -- this path must be bit-identical

Twelve subtests, all Q5_K, all ptx; spirv takes the dp4a path in both arms and
so passed, which is the same one-backend blindness the v15 row-pairing had in
the other direction (CUDA `+Inf`, Vulkan exact).

**mma.go's own header already records the previous two occurrences**, and that
is why the fix is not another `case`: the lane's element GROUP is derived from
the format's arithmetic now — group `gi` sits in secondary word `gi/lanes` at
bit `hi*(gi%lanes)`, which is `matvec.go`'s `hiOf` verbatim — and a shape whose
word index is not constant over the lane is REFUSED by name rather than
approximated. An unreachable-looking format that is silently wrong is what the
previous two were.

★ **AND THE GOLDEN COULD NOT HAVE CAUGHT IT: `kernels.sha` HOLDS 55 HASHES AND
NONE OF THEM IS AN MMA KERNEL.** The bit-equality gate against dp4a is the
stronger check and it was already there — it was simply red and unread, which
is RULE 11b's point about a gate that is always red.

★ **THE TWO STALE `matvec/Q5_K` GOLDENS WERE BLESSED, WITH THE BEHAVIOUR
RE-DERIVED FROM SOMETHING THAT IS NOT THE GOLDEN.** They had been red since
v15 for the same packing change, correctly. The non-golden evidence is
`backend.TestMatVecRealWeights`, which runs a REAL Q5_K tensor against
`quant.Dequant` on every backend and FIRES against a Q5_K-specific violation.
Exactly two of 55 lines move; the other 53 being untouched is the second
statement, and it is also what proves the adjacent-row tile emits NOTHING new
at `Rowt == 1`.

★ **AND `TestMatVecRealWeights` GAINED THE TILE DIMENSION, WHICH IS WHERE THE
TWO-PLANE FORMATS LIVE.** Q5_K and Q6_K load `hiw[j]` through the same
`tileOff` and had no tiled coverage anywhere: the synthetic case builds only
Q4_0, Q8_0 and Q4_K blocks by hand. Six formats x two splits x three tile
widths, ptx, spirv and msl.

★ **AND THE BATCHED PREFILL KERNEL SHIPS A TILE THAT NO KERNEL GATE ASKED
FOR.** `batchMV` picks `Rowt` from `batchRowt`, which is FOUR, so every prefill
matvec the tier launches combines NTok, Tok and Rowt — and `ntokCase` built its
kernel at `Rowt == 1`. The shipping shape had only end-to-end coverage. It was
already true before this work and matters more now, because one `tileOff`
serves decode and prefill.

## ★ PER-DISPATCH GPU TIMING IS NOT REACHABLE ON THE M4, AND FOUR ROUTES WERE MEASURED RATHER THAN ASSUMED.

The Metal accounting closes at 93.7% -- streaming 7.493 ms, dispatch overhead
0.450, head excess 0.429, **0.558 unattributed**, no single term over 5%. The
obvious next instrument is to time each of the ~445 dispatches in a token
individually. It cannot be built on this hardware:

| route | result |
|---|---|
| `supportsCounterSampling:atDispatchBoundary` | **FALSE** |
| `supportsCounterSampling:atStageBoundary` | TRUE, and it buys nothing |
| Instruments **Metal System Trace** | records; the per-shader tables are EMPTY |
| Instruments **Metal GPU Counters** | *"Selected counter profile is not supported on target device"* |

★ **THE FIRST ONE IS THE WHOLE ANSWER AND IT COST TWENTY MINUTES INSTEAD OF TWO
HUNDRED LINES.** A sample buffer cannot fire between launches on this device, so
per-dispatch timing would need ONE ENCODER PER LAUNCH -- and a compute encoder
is `MTLDispatchTypeSerial`, so the instrument would serialise the token it is
measuring and add an encoder's cost to each of ~700 dispatches. That is this
file's own recurring failure: the read-wall probe measured itself, and its
"fix" was committed before it was measured and made the number worse.

★ **AND STAGE-BOUNDARY SAMPLING IS NOT A CONSOLATION PRIZE, WHICH MATTERS
BECAUSE IT LOOKS LIKE ONE.** `Batch` opens exactly one encoder per command
buffer (`metal_darwin.go:603`), so an encoder-boundary timestamp returns the
interval `GPUStartTime`/`GPUEndTime` already returns. It is the SAME NUMBER
through more code, not coarser resolution.

★ **THE INSTRUMENTS TRACE DID RECORD, WHICH IS WHY ITS EMPTINESS IS EVIDENCE.**
`metal-shader-profiler-intervals` and `gpu-shader-profiler-interval` exist in
the schema and hold **zero rows**: the Shader Profiler is not among the
instruments `xctrace` exposes (GPU, Metal Application, Metal GPU Counters, Metal
Performance Overview, Metal Resource Events -- it is an Xcode-GUI-only
facility). What the trace carries instead is 855 command-buffer submissions, 854
encoders and 868 completions -- command-buffer granularity, again -- plus 1071
`metal-driver-event-intervals`, which turn out to be memory WIRING events
(`Wire 16.45 MiB System Memory`) and not dispatches at all.

★ **WHAT REMAINS IS THE METHOD THAT PRODUCED THE ACCOUNT IN THE FIRST PLACE.**
Repeat an idempotent op N times and regress the slope: that is where the 1.09 us
dispatch cost and the 2.106-2.116 ms head figure came from, it needs no FFI, and
it is not blocked by any of the four rows above. Its limit is that the op must be
idempotent -- attention writes the KV cache, so it cannot be repeated in place --
so it can price the norms, the quantize and the matvecs and not the whole layer.

`Ctx.CounterSampling` and `Ctx.CounterSets` are kept as a PROBE rather than an
assertion: both answers are legitimate hardware and a newer GPU may say
otherwise. `metal_other.go` carries the stubs, because `metal.GPUBusy` being
darwin-only in the header broke the Linux build of `./engine/model` once already.

## ★★★ THE DEVICE HAD NO SLIDING WINDOW, AND THE FIRST PROBE SAID IT DID.

`nn.LayerPlan` carried the local/global PERIOD so the tier could pick gemma3's
local rotary table, and nothing else: every local layer on a device attended its
whole causal history. gemma-3-1b on CUDA at 712 positions (window 512):

    device-vs-host                      8.5e-02
    device-vs-host-with-window-removed  9.7e-05

Fixed by baking the window into the three scores kernels (decode, FMA tile, MMA)
as a mask to -inf on keys before `cnt - min(cnt, W)`. Exact: the softmax gives
those keys zero weight and the accumulator adds 0*V over keys the prompt DID
write. Softmax and accumulate are untouched; window 0 emits identical IR.
`backend.TestAttnScoresWindowMasksExactlyTheOldKeys` holds it bit-for-bit on
ptx and spirv; `model.TestSlidingWindowRunsOnTheDevice` holds the model.

**Three traps on the way, each of which produced a confident wrong reading:**

- **The first probe measured the host.** Device-vs-host read 1.5e-04 past the
  window with 0 of 26 blocks still placed: two States shared one tier, the first
  had prepared the blocks for 498 positions, and `layersOnce` refused position
  498 with no error (`pos+nrow > p.MaxSeq`), so the State restarted every later
  token on the host. The gate now fails if a placed block leaves the device.
  (That silent refusal of a larger second State is itself worth fixing.)
- **Two unpinned tiers disagree with each other** by up to 5e-03 at early
  positions -- `tuneSplit` picks a reduction order per process. Read against the
  host, that looked like an all-local window bug. Pin with
  `tier.WithDeviceTune(tier.TuneOff)`.
- **A narrowed window on a model not trained with it is chaos.** Phi-3.5 at 16
  keys on every layer: host 0.26 from transformers (loading the same GGUF),
  device 0.48, the two 0.02 apart -- every engine amplifying its own rounding.
  gemma-3-1b at a 64-key window spikes to 0.28 at one token class on BOTH tiers
  against transformers while sitting at 1e-4 elsewhere. Gate where a correct
  device is close: gemma's own window, or every layer local at 128.

## MXFP4 and float weights run on the device, and the tier never told it a block had sinks.

**MXFP4.** The IR has no byte shuffle, so the e2m1 table is computed rather than
looked up. `kernels.decodeE2M1` maps four codes, one per byte lane, to the
biased table values: with `q = s|e1|e0|m` and `l = q&3`, the doubled magnitude
is `l + (l if b2) + 4*b2 + 2*(b0 b1 b2)` and the code is `(v+12) - (2v if s)`.
Every byte stays in 0..24, so nothing carries across lanes, and after it the
format is Q4_0's shape with bias 12: the dp4a kernel and the tensor-core
kernel both call it at the nibble unpack. `backend.TestMatVecMatchesReference`
(exact e2m1 values) and `TestMatVecMMAMatchesDot4` hold it on PTX, SPIR-V and
MSL. Dropping the `2*(b0 b1 b2)` term reads 1.7e-02; skipping the decode in
mma.go alone fails the MMA gate. The llama.cpp `prmt` and MLX f16 bit-trick
routes surveyed beforehand were not needed.

**Float weights (F32, F16, BF16).** They declined every block ("attn_q is F32,
which has no device kernel"), so synth-gptoss, a tied F32 head, and an HF
conversion's matrices stayed on the host. `kernels.Float32/Float16/BFloat16`
are device formats whose pack is a copy of the rows; `kernels.MatVec` reads
them row-major against the FLOAT activation, passed in the scale-plane slot, so
the launch shape does not change. The tier records the activation's float
source structurally: a launch writing `(bs.a, bs.ax)` quantizes its first
buffer (`la`, `dOf` in `layersOnce`).

The first version read the int8 activation instead, and every kernel gate
passed, because the reference used the same quantized activation. Four hybrid
gates went red: the fixture's tied F32 head moved to the device and lost the
precision the host keeps (pre-shrink logit NMSE 2-5e-4, a second session
drifting to 1.6e-1). With float activations the llava tower, whose mmproj is
F16, matches the host at 5.9e-9 after one block.

**The offered plan.** `offerRange` built `plan := s.plan` while `planFor(li)`,
the only thing that sets `AttnSinks` and `MoEBias`, served declines alone. So
the device built a gpt-oss block's scratch without sinks or mixture biases,
skipped their uploads and ran without them: NMSE 1.7 from position 0 on CUDA
and Vulkan alike, with every sink and bias kernel green in isolation -- RULE
11c's shape exactly. `TestSynthGPTOSSOnEveryDevice` now runs every block and
the head on each backend against the host, routing pinned (a random 4-expert
router near-ties, and the flip is not a defect), with the down and up biases
scaled so a logit can see them. Worst 1.5e-3 on CUDA/Vulkan and 4e-12 on Metal;
the plan reverted, the sinks or the router bias dropped, or any expert bias
misread, lands at 0.08-2.0.

## ★★★ THE ROTARY TABLE IS A DEVICE KERNEL, AND THE MAGIC-NUMBER ROUNDING IT WAS TRANSCRIBED FROM DOES NOT SURVIVE A SPIR-V DRIVER.

`kernels.RoPE`'s own comment said the table was uploaded because *"sin/cos on a
GPU is an approximation whose error is vendor-specific, so computing it here
would make q and k differ between CUDA, Vulkan and Metal for no gain"*. That
objection is against calling a DRIVER's transcendental and it still stands. It
was doing duty for a different sentence -- that the device had no rotary table
at all -- which is RULE 8a's own failure mode: a graph change on the CPU alone
removes the capability from the device, not as an error but as an absence.

`kernels.RopeTable` is the fourth transcription of the host kernel that landed
alongside it (jit/cpu/ropetab.go and its SSE and NEON twins are the other
three): the same four-digit position decomposition, the same folded planes, the
same degree 7/8 minimax pair on |r| <= pi/4. It calls nothing of the vendor's.
The planes are DATA -- `nn.Rope.TabPlanes`, folded in float64 once per model at
load -- so `jit/gpu/kernels` borrows nothing from `jit/cpu` but the layout,
which moved down into `kernels/ropetab_const.go` because `jit/cpu` already
imports that package and the other direction is a cycle.

**Measured against the host kernel's own table, on real hardware, all three
backends:**

| backend | device | worst vs host table | worst vs a float64 oracle |
|---|---|---|---|
| ptx | RTX 3050 Ti | **5.960e-08** | 5.960e-08 |
| spirv | RTX 3050 Ti (Vulkan) | **5.960e-08** | 5.960e-08 |
| msl | M4 | **5.960e-08** | 5.960e-08 |

5.960e-08 is ONE ULP at 1.0, and it is the same figure engine/nn/ropetable.go records
for the host kernel against the float64 table it replaced. Eight geometries from
four rotary pairs (stories260K's head dimension is eight) to 128, one to 32
rows, mscale on and off, and eighteen positions from 0 to 2^28-1.
`model.TestRopeTableOnDeviceMatchesTheUpload` runs both arms in ONE process on
Llama-3.2-1B and gemma-3-1b with every block placed and the split tuner pinned:
24 tokens identical and **worst |dlogit| exactly 0.000e+00**, so the ulp above
is the kernels' worst case over the whole swept position range rather than the
token's.

★ **AND THE GATE ASKS FOR A BAND RATHER THAN THAT ZERO, WHICH IS THE
DIFFERENCE BETWEEN A CONTRACT AND AN OBSERVATION.** The two tables are two
SPELLINGS of one reduction -- integers and half-up here, two magic adds and
half-even there, unfused Fma on one backend -- so an ulp between them is
permitted by the design and the kernel gate measures exactly that. A gate
demanding zero would one day fail for a reason nobody can act on. It demands
1e-3, which is far under anything a real defect produces (a wrong rotary base
moves a gemma3 logit by 0.1, a wrong quadrant by whole units) and far over an
ulp, and it LOGS the zero rather than requiring it. Run on CUDA here and, through
a cross-compiled test binary, on the M4's Metal -- 30 and 60 rows built against 0
uploaded, bit-identical on both, plus a 362-token prefill at 363 built and 0
uploaded. RULE 11d is why the Mac run is not optional: the M4 is where two of
the HOST kernel's four violations turned out to be doing nothing at all.

**★ AND 12l's PARENTHETICAL IS REFUTED: SPIR-V DOES REASSOCIATE, AND IT DELETED
THE WHOLE RANGE REDUCTION.** That entry ends *"the change was kept anyway
because PTX and SPIR-V do not reassociate by default, so there it is real."*
The first version of this kernel transcribed the host's mod-four fold verbatim
-- `x - ((x+M)-M)` with M = 2^25, three instructions, exact on every CPU here --
and read **|d| ~ 2.0** on spirv against 5.96e-08 on ptx, which is a whole
quadrant and not a rounding. A four-line probe of the two steps alone:

    (x+M)-M          returns x UNCHANGED for every x        -> the round is gone
    x - ((x+M)-M)    returns ZERO for every x               -> the fold is gone

The driver rewrites `(a+b)-b` to `a`, which is not an IEEE-valid transformation,
and loading M from a BUFFER does not stop it -- it is algebra on unknown values,
not constant folding. **A magic-number rounding is not portable to this IR.**

**★ THE REMEDY IS INTEGERS, AND IT IS STRICTLY BETTER THAN THE THING IT
REPLACES.** s is a sum of exact multiples of 2^-12 under 2032 (the host file's
bit count), so `s*2^12` is an integer under 2^24 that converts exactly, masking
with `4*2^12-1` IS the remainder mod four, and scaling back is a power of two.
Nothing there is float algebra, so there is nothing to reassociate -- on any
driver, including Metal's, whose fast-math default 12l already records and which
this kernel therefore never had to find out about. The remainder lands in [0,4)
where the host's lands in [-2,2]; n absorbs the multiple of four, f = rem-n is
the same exact float, and bits 0 and 1 of n -- the swap and the two signs -- are
what a multiple of four leaves alone. It also removes a subtlety the host
carries: the device's n is never negative, so nothing rests on two's complement.

The rounding itself is the other half. The IR's only float-to-integer op is
`ir.OpCvtI32`, which truncates toward zero, and the integer fold is what makes
adding a half sound rather than a second magic: rem+c is never below -1/16, so
truncation IS the floor. Round-half-up and round-half-even differ only at an
exact tie, and a tie is genuinely a tie here -- at f = +1/2 and f = -1/2 the
quadrant swap and the sign flip turn the same two polynomials into the same two
answers, because the sine fit is exactly odd in r and the cosine fit exactly
even.

**★ AND THE CONSTANT BLOCK IS READ AS u32 AND BITCAST, WHICH IS NOT STYLE.**
Two of its words are integers stored through `math.Float32frombits`, and the
integer one is `0x00000001` -- the smallest DENORMAL as a float32. A GPU is
entitled to flush denormals, and a flushed one turns "cosine's quadrant is n+1"
into "cosine's quadrant is n": every value finite, every shape right, a
quarter-turn wrong in two quadrants of four. Declaring `pConst` as a u32 buffer
never lets a float unit near them. Unmeasured as a failure, because it was
designed out before it could happen -- recorded as a hazard that was reasoned
about rather than a bug that was found.

**★ THE VIOLATIONS GO IN THROUGH THE UPLOADED BLOCK, WHICH IS THE HOST's IDEA AT
THE DEVICE's SEAM.** Five, each removing exactly one step, and every one fires
on ptx, spirv and msl:

    quadrant-selection    cosine's quadrant taken from n, not n+1   |d| 1.410e+00
    sign-step             bit 1 stops deciding the sign             |d| 2.000e+00
    no-mod-4-mask         the digit products are never folded       |d| 2.000e+00
    no-round-to-integer   n truncated rather than rounded           |d| 9.018e-05
    swapped-sin-cos       made in the COMPARISON, as nn's is        |d| 1.412e+00

The last two are the device's own and have no host twin, because no host tier
reads those words -- they stand in for one instruction each that the IR does not
have. `RopeTabMagicWord` is absent from that list for the same reason: the
device does not read it, so zeroing it would be a violation that removes
nothing, which is exactly the green line RULE 10 refuses.

**★ THE ROW COUNT IS RUNTIME WHERE EVERY OTHER SHAPE IS BAKED, AND A CHAT PROMPT
IS WHY.** A prefill's last chunk is narrower than the batch width and the host's
staging fills the rest of the table with ZEROS -- which no position produces,
since position zero gives cosine one. The first version uploaded that case
rather than argue the trash rows cannot matter, and the counters said what that
cost: on a 362-token prefill **257 rows built and 128 uploaded**, and on a
chat-length prompt 25 built against 32 uploaded, because a prompt shorter than
the batch width is ALL ragged. Clamping the kernel to a valid count read from the
buffer, and zeroing the table once at scratch time, takes both to **zero
uploads** with the logits still bit-identical. One load, and the difference
between this kernel serving every prefill and serving none of the short ones.

**★ AND IT IS A KNOB, NOT A DECLINE, WHICH IS THE ONE PLACE ON THIS SEAM THAT IS
TRUE.** `Layers` is handed the finished cs/csSWA tables as well as the planes,
so a device that cannot compile the kernel uploads the table exactly as it did
before and computes the same logits -- unlike sinks, post-norms or a window,
where the absence is fluent and wrong. What must not happen is the absence being
SILENT, so `Stats.RopeTableWhy` carries the reason, `RopeTables`/
`RopeTableUploads` say which arm ran, `jitllm run` prints both, and
`jitllm verify -ab ropetab` refuses rather than comparing a knob that reached
nothing.

## A one-launch RMSNorm (one 128-thread workgroup per row) is a REGRESSION on a V100

Motivation: nvprof on a Tesla V100-SXM2-16GB (the V100 host, CUDA) put jitllm's
Llama-3.1-8B decode at ~821 launches a token against llama.cpp's ~614, with the
norm as three of them per FFN (Add 2.3 us + NormPartRows 4.9 us +
NormApplyRows 3.4 us) where llama.cpp's rms_norm_f32 is one kernel of 5.8 us.

Built `kernels.RMSNormRows` (workgroup per row, squares summed in shared
memory, optional residual add folded in), gated bit-for-bit-ish against a
float64 RMSNorm on CUDA and Vulkan including ragged widths, and wired it into
the tier's norm closure and the add->FFN-norm pair. Measured, one V100, n=128,
ABBA, 3 rounds:

    Llama-3.1-8B   103.6-109.2 -> 98.4-101.4 tok/s   (about -6%)
    olmoe-1b-7b    272-290     -> 256-277            (flat to worse)

Reverted. The cause is occupancy, not the launch count it was meant to cut:
a decode norm is ONE row, so one 128-thread group runs the whole 4096-wide
reduction and apply on one SM, where the partitioned pair spreads it over 64
partials and 32 groups. llama.cpp's version uses 1024 threads a row; getting
there needs a tree reduction the IR cannot express without conditional
execution (RULE 13), or a shuffle-based one gated on a 32-lane subgroup. The
workgroup-per-row norm was tried without either, and lost.

## Flushing L2 before each split-tuner trial was tried, and it is a REGRESSION on a V100

The hypothesis: `timeSplitMode` launches one kernel four times back to back on
the same weights, and a 1024x4096 Q4_K matrix (2.4 MB) sits in a V100's 6 MB L2,
so the tuner picks splits for a cache-resident weight while decode reads every
weight from DRAM. The per-token trace looked like support: k/v matvecs at 12 us
for 2.4 MB (~200 GB/s), q/o at 572 GB/s, against gate/up at 768 and the head
at 842.

Built: a 16 MiB `kernels.Restride` streamed through the cache before every
timed launch (a constant cost, so the argmax needs no subtraction). The choices
moved to larger splits (k/v 8g -> 32g, gate/up 4g -> 8g, down 16 -> 32), and
decode got slower. The V100 host, V100-SXM2-16GB, CUDA, Llama-3.1-8B Q4_K_M,
n=128, one process per sample, ABBA, 6 each:

    no flush   115.47 110.53 115.78 116.12 115.72 111.36   median ~115.6
    flush      114.09 109.13 110.57 109.17 109.94 112.53   median ~110.3

So a flush-separated trial is the wrong imitation of decode. What decode does is
stream a DIFFERENT block's weights of the same shape on every launch, and timing
exactly that sequence is right: `devTier.retuneDecode` (same day) re-picks each
shape over the placed blocks once they are all on the card, and measured +2.4%.
The per-shape optimum was read off nvprof kernel medians in the graph first
(1024-row Q4_K: 12.0 us at the in-cache pick of split 8, 8.2 at 32 in-group).

## Latent attention (MLA) prefill is batched on the device, and the DeepSeek-V2 family takes the grouped mixture path

DeepSeek-V2-Lite ran 0 grouped mixture blocks after the grouped MoE prefill
landed, because `layersOnce` refused ANY latent block at rows > 1 ("multi-head
latent attention has no batched submission yet") -- the whole chunk, dense lead
and mixtures included, went one row at a time. Two more gaps sat behind it:
`prepBatch` precompiled only q/k/v/o/gate/up/down, so the shared expert's
matvecs, MLA's kv_a/q_a/q_b and a linear block's projections would have been
compiled lazily inside the session; and the grouped mixture refused V3 grouped
routes (`ExpertGroups > 1`) and the kernels refused `Rows` with groups.

`tier/mlabatch.go` is emitMLA over R rows with no new kernel: Rows twins of the
slices, norms, copies and rotations; both absorbs as GROUPED matvecs with the
head as the expert (activations gathered head-major, every head's run exactly R
columns); AttnAccTiled at the latent width over the row-wide cache.
ExpertGroupScore/ExpertGroupMask and ExpertRank's grouped plane take a row
stride (rows == 1 emits the same IR; lowertest unchanged).

Gates: `backend.TestRouteRowsMatchOneRow` (bit-exact against per-row launches,
five route shapes; fails at row 1 with the group-score row stride dropped).
`model.TestBatchedMixturePrefillMatchesRowByRow` on a V100 (CUDA, sm_70),
300-token prompt, 12 generated: V2-Lite 78 grouped / 81 batched latent blocks,
NMSE 9.97e-4, ids identical; Coder-V2-Lite 7.07e-4; Moonlight 4.35e-4. Against
a violation (the absorb activations gathered token-major): NMSE 5.80e-1, ids
part at token 7.

V100-SXM2-16GB, CUDA, `jitllm speed -devices cuda:0 -r 2` (p512, n128), ABAB,
same pass as llama-bench b60eeeb6 `-p 512 -n 128 -ngl 99 -r 3`:

    DeepSeek-V2-Lite   this   pp512 1385 / 1384 / 1383 / 1397   tg128 121.5-122.9
                       before pp512  119 /  120 /  120 /  120   tg128 122.1-123.0
                       llama.cpp pp512 1994 +/- 21              tg128 168.2
    Coder-V2-Lite      this   pp512 1379 / 1359                 llama.cpp 1743 +/- 82
    Moonlight-16B      this   pp512 1366 / 1386                 llama.cpp 2023 +/- 15

11.6x on prefill, decode unchanged; 0.69x llama.cpp's prefill on V2-Lite.

## A mixture's prompt on sm_70's tensor cores -- the grouped GemmVolta, and MLA's attention on m8n8k4

nvprof on a V100 before this work: the grouped mixture ran every expert matvec
of a batched chunk on the dp4a grouped kernel (223 ms of gpt-oss-20b's ~297 ms
pp512; 119 ms of DeepSeek-V2-Lite's 357 ms), and V2-Lite's latent attention
scores ran on the FMA tile over the row-major latent cache (attnscorestiledw,
5.87 ms a layer, 158 ms of what was left). llama.cpp's sm_70 build does not use
mmq for a prompt-sized mul_mat_id (mmq.cu ggml_cuda_should_use_mmq: fp16 MMA
hardware and a batch over the dp4a bound); ggml-cuda.cu ggml_cuda_mul_mat_id
sorts the tokens by expert on the host and runs a dequantize + cuBLAS f16 GEMM
per expert.

What was built, each gated and measured in its own commit:

  kernels.GemmVolta, grouped   one expert per token block (pSel), runs padded
                               to 32/64 columns, three plane bases per sheet,
                               MXFP4 via its e2m1 code at bits 9-11 of a
                               binary16, expert biases in the epilogue, 96-row
                               blocks where 128 does not divide the rows
  kernels.ActF16TGather        the gather and the binary16 conversion in one pass
  AttnScoresMMA70, kStride 0   row-major K: one 16-byte load per key's 4 dims
  AttnAccMMA70 over the latent the value is the row's first KVLoraRank floats;
                               atile fixed at 32 for the softmax (mla70Attn)
  tier                         voltaOn/sm70 PROBE the backend: newMoeGroup runs
                               before any batched matvec has set mmaOff, and the
                               first build therefore ran 0 experts on the
                               tensor cores behind a green gate. Stats
                               GroupedVolta / MLAScores70 / MLAAcc70 are the
                               selection checks; on sm_70 the scratch has no
                               dp4a half

Kernel probe (backend.TestGemmVoltaGroupedSpeed, random routing of a 512-token
chunk), dp4a grouped Tok4 against the grouped GemmVolta: gpt-oss 2880x2880
MXFP4 3445 -> 895 us (96x64), V2-Lite 1408x2048 Q4_K 1856 -> 497 (128x64),
Qwen3-30B 768x2048 1409 -> 455 (128x32), Mixtral 14336x4096 12223 -> 2455.
A 64-column block wins where an expert averages >= 48 tokens (voltaUnit).

End to end, Tesla V100-SXM2-16GB, CUDA, `jitllm speed -p 512 -n 128 -r 3`,
main 9c8fe57 against 49561a6 with llama.cpp sm_70 `llama-bench -p 512 -n 128
-ngl 99 -r 3` between them, one ABBA pass:

    model (GPUs)              before       after        llama.cpp   ratio
    gpt-oss-20b (1)           1764-1795    4132-4268    2684        0.66 -> 1.56
    DeepSeek-V2-Lite (1)      1443-1448    4183-4297    2065        0.70 -> 2.05
    DeepSeek-Coder-V2-Lite(1) 1445-1453    4121-4267    1997        0.73 -> 2.10
    Moonlight-16B-A3B (1)     1446-1450    4307-4381    2091        0.69 -> 2.08
    olmoe-1b-7b (1)           4654-5027    9562-9921    4064        1.21 -> 2.38
    Qwen3-30B-A3B (2)         1769-1894    3123-3227    1046        1.78 -> 3.06
    Mixtral mix-banks (2)      426-427     1642-1653     866        0.49 -> 1.90
    tg128 unchanged on all seven.

What is left on gpt-oss-20b's prompt after all of it (nvprof, tuner off): the
grouped GEMM is still about half the kernel time at ~17-19 TMAC/s against the
dense GemmVolta's 34-37 -- its token blocks are 32-64 columns where the dense
one's are 128, and 25-35% of the columns are padding. A wider block over the
experts that have the tokens, or a k-split for the thin ones, is the next lever
and is unbuilt.

## The M4's Metal rows -- a simdgroup_matrix prompt GEMM, IEEE libraries, a width check, and a Wait that polls

The goal was every losing M4 row against llama.cpp b10985 on the same GGUF.
The commits carry the full numbers; this is the map and the null results.

**Prefill was the dp4a tile on a GPU with no integer dot.** Llama-3.2-1B's
512-token prompt read 77.7 tok/s batched against 104 token by token and 1604
for llama.cpp. `kernels.GemmTile` (eeab55b) is llama.cpp's
`kernel_mul_mm` (ggml-metal/kernels/mul_mm.metal) in the IR: voltaDeq staging
into threadgroup memory, ActF16T activations, four simdgroups of 8x8 half
tiles into float accumulators, double-buffered. It needed `ir.TilePhi`, tile
load/store through a shared array and `SubgroupIndex`; PTX and SPIR-V refuse
the new forms by name and the tier asks for the tile on msl only. 0.048 ->
0.880 of llama.cpp on Llama (18.2x). The grouped form (00cbb41) takes a
mixture's experts through the sm_70 grouped contract unchanged: Qwen3-MoE
0.40 -> 0.81.

★ **TRIED, AND NULL:** a
single-buffered GemmTile, a 16-half row pad, and a k-split all measured no
gain; the big shapes already run 3.1-3.3 TFLOP/s against the M4 GPU's ~3.76
fp32 peak. What is left in prefill is the THIN shapes (1.0-1.4 TFLOP/s: a
1536- or 256-row projection does not fill the GPU) and attention, which still
runs the FMA tile on Metal.

**Metal was the only fast-math backend.** `newLibraryWithSource:options:nil`
compiles with fast math, where PTX emits `.rn` and SPIR-V carries no
fast-math flags; three backend gates were red on msl alone for it
(RMSNormQuant 1408 of 4096 words, the fused delta rule 7.6e-05, the expert
route by one ulp). Strict compile options (75208e0) fix all three and cost
0.1-0.5% (fast/strict pp 1.002-1.005, tg 1.001-1.002 on three models).
llama.cpp keeps fast math; this engine does not follow it there (RULE 7m).

**A launch at the wrong width is a wrong answer on Metal only** (97e3b00): msl
lowers `OpNTID` to the declared workgroup as a constant. `mtlKern.checkWidth`
refuses it, and on its first run it found the tier launching
`headNormScalar` (declared 64) 32 wide whenever a head is narrower than a
subgroup -- heads 64c+32..63 of every group were never written. The launch
geometry follows the lanes each norm was built at now (headNormGeom).

**The decode round trip.** GPU clock and host timestamps split the M4's
0.52 ms idle gap between two decode tokens into 0.18 ms of host and
0.32-0.34 ms of commit-to-start plus wake-up latency. Polling the command
buffer's `status` before `waitUntilCompleted` (08a2fe7, 20 ms budget) removes
the wake-up: tg +4.7% Llama, +4.4% Qwen2, +2.5% gemma. llama.cpp
(ggml-metal-context.m) blocks too and hides the latency by encoding the next
of n_cb command buffers; a decode token here is one buffer.

**The M4 board after all of it** is in docs/perf/current.md.

## The V100 decode rows below parity are not host time, measured four ways

Phi-4-mini, gemma-3-12b, Qwen3-1.7B Q8_0 and gemma-2-9b sat at 0.957 / 0.977 /
0.988 / 0.993 of llama.cpp (tg128, V100-SXM2, sm_70, same pass). What was priced:

    CUDA calls inline on the caller's thread   +0.3..+1.1%, SHIPPED (125b3c5)
    a decode token as TWO recordings (2 blocks  null: 189.0 vs 188.8, 73.7 vs
      then the rest), so the card starts while  73.5, 91.7 vs 91.7 tok/s
      cuGraphLaunch is still returning
    graphs off (direct launches)                -1.5% (185.6 vs 188.5)
    per-kernel time, nvprof                     parity: matvecs 3.77 vs 3.81 ms,
                                                the rest 1.64 vs 1.58 a token
    device timeline, nvprof --print-gpu-trace   5.83 vs 5.91 ms a token, kernels
                                                5.44 vs 5.43, idle 0.39 vs 0.48

`jitllm run` now prints "host per submission" (tier.Stats TSubStage /
TSubLaunch / TSubRead / TCapture): Phi-4-mini reads stage 59 us, launch 232 us
(26 of it three captures a 256-token run), read 4954 us. The launch looked like
the idle the gap is made of; splitting it proved the card was not waiting on it.

★ **SO UNDER A PROFILER THE TWO ENGINES ARE THE SAME AND WITHOUT ONE llama.cpp
IS ~4% FASTER ON THESE ROWS**, both at the same 1530 MHz SM clock, and
llama.cpp drawing 222 W against jitllm's 150 W (jitllm's average includes its
load, so that is a lead, not a measurement). What differs unprofiled is
unattributed. llama.cpp runs these without CUDA graphs on pre-Ampere cards.

## Metal prompt attention as one simdgroup_matrix kernel

The M4's Metal prefill gap GREW with the prompt, which is attention's signature
(the matmuls already ran as GemmTile at 3.1-3.3 TFLOP/s): Llama-3.2-1B Q4_K_M
0.93 / 0.88 / 0.77 of llama.cpp b10985 at pp128 / 512 / 2048, SmolLM2-360M
0.86 / 0.76 / 0.60. Attention on msl was the scalar FMA tile plus a score matrix
written and re-read between three kernels.

`kernels.FlashPrefillTile` is FlashPrefill70's algorithm on the ir tile ops:
K and V staged as binary16, S = Q K^T per simdgroup into workgroup memory, four
lanes a row for the running max / sum / binary16 P and the rescale of an O that
lives in workgroup memory (a simdgroup_matrix has no lane layout, so none of
that can happen in registers), then O += P V. llama.cpp's kernel_flash_attn_ext
keeps O in the same place. `backend.TestFlashPrefillTileMatchesReference`:
NMSE 1.3-1.7e-7 against float64 with every slot past the count poisoned; the
dropped-key violation reads 1.5e-3. `model.TestFlashPrefillMatchesThreeKernels`
on Metal: Llama 4.2e-4 and Qwen3-0.6B 3.4e-4 against the host, the same as
the three kernels.

M4 Metal prefill, tok/s, llama.cpp b10985 in the same pass:

| model | pp512 was / now / llama.cpp | pp2048 was / now / llama.cpp |
|---|---|---|
| Llama-3.2-1B Q4_K_M | 1407 / **1532** / 1604 (0.955) | 1160 / **1446** / 1505 (0.961) |
| SmolLM2-360M Q8_0 | 3297 / **3973** / 4351 (0.913) | 2208 / **3475** / 3676 (0.945) |
| Qwen2-1.5B Q4_K_M | 1052 / **1106** / 1159 (0.954) | 887 / **1005** / 1098 (0.916) |
| gemma-2b | 758 / (746) / 848 | 692 / (629) / 809 |

gemma-2b (head width 256, one kv head) is NOT taken: its blocking is 16 query
rows (8 at first, 421 tok/s at pp2048), so K and V are re-staged per 16 rows,
and it measured below the three kernels. The fix is one staging of K/V for
all the heads that share it, which the float32 O in workgroup memory leaves no
room for at width 256 -- a two-pass form (max and sum first, then P V with O in
registers) is the candidate.

### V100 decode: the per-token hand-off, measured on the device clock

`JITLLM_CUDA_TIMING=1` (`backend.SetCUDATiming`) brackets every replayed graph
with two CUDA events on the capture stream and reads them one replay later, so
`jitllm speed` prints the graph's device time and the device idle BEFORE it --
the hand-off: sync, argmax readback, the host's token work, x's upload and the
next launch. Gated by `backend.TestCUDAGraphTimingReportsEveryReplay`
(an after-event recorded before the launch reads 4 us and fails).

Phi-4-mini Q4_K_M, V100 GPU 0, tg256, two runs, medians per token:

    graph on the device   5,156 / 5,216 us
    idle between graphs     116 /   118 us   <- the hand-off, 2.2%
    wall                  5,304 / 5,367 us   (the events cost ~1%)
    llama.cpp, same box   ~5,060 us for its WHOLE token (197.6 tok/s)

★ AND llama.cpp's HAND-OFF IS BIGGER, MEASURED THE SAME WAY. An LD_PRELOAD
shim around llama.cpp's own `cudaGraphLaunch` puts the identical event pair on
its stream (it costs llama-bench nothing: 196.8 against 197.4 tok/s). It DOES
replay one CUDA graph a token on this sm_70 build (128 graph traces for 128
tokens under nsys), which retires the "it launches kernel by kernel below
Ampere" guess. Phi-4-mini, tg256, two runs, medians per token:

    llama.cpp   graph 4,853 / 4,860 us   gap before 201 / 200 us
    jitllm      graph 5,159 / 5,158 us   gap before 117 / 116 us

**The whole 0.95 is the graph: +300 us, 6.2%.** jitllm's hand-off is already
85 us SMALLER than llama.cpp's, so removing it outright would read ~0.97 and
cannot close the row -- which is the measured reason the goal's first lever is
not the one. No profiler resolves the in-graph term: nvprof reads the two
engines' kernels at parity (both slowed to ~168 tok/s), nsys at graph level
puts jitllm AHEAD (177.7 against 175.8 -- tracing costs llama.cpp 11% and
jitllm 6%), and ncu's isolated kernels inflate jitllm's Q4_K matvecs 2.7x
against llama.cpp's 1.04x, which describes kernels run alone with
millisecond gaps between them, not the decode. `-ab graph` puts the replay
2.5% ahead of launch-by-launch, and both engines run at the 1530 MHz max
clock. What separates the in-graph 300 us is a per-kernel device-clock split
taken the same way on both engines (event nodes between the captured
launches), which has not been built.

### V100 decode: the in-graph 300 us is attention, and a KV-head flash-decode kernel

`JITLLM_CUDA_KERNEL_TIMING=1` (jitllm) and `scripts/kshim.c` (llama.cpp, an
LD_PRELOAD shim) take the SAME measurement: during capture an event NODE is
recorded after every launch (`cuEventRecordWithFlags(..., EXTERNAL)` -- a plain
record inside a capture is only a dependency edge, and cuEventElapsedTime
refuses it), so each replay stamps every kernel's end on the device clock.
Event nodes cost ~3.9 us each on both engines (jitllm 5,158 -> 7,455 us over
583 nodes, llama.cpp 4,856 -> 6,936 over 550), subtracted below. Phi-4-mini,
V100, tg256, per token:

| | jitllm | llama.cpp | |
|---|---|---|---|
| Q4_K matvecs | 2,053 us | 1,876 | +177 |
| Q5_K qkv | 547 | 522 | +25 |
| Q6_K head + down | 1,139 | 1,191 | -52 |
| norm + quantize | 530 | 555 | -25 |
| **attention** (4 kernels / flash vec + combine) | **564** | **278** | **+286** |
| rope + KV copies | 312 | 273 | +39 |
| SiLU (jitllm fuses it) | -- | 87 | -87 |

`kernels.FlashDecodeKV` (tier.Config.FlashKV, `JITLLM_GPU_FLASH=2`, with
`JITLLM_GPU_FLASH_GROUP` / `_SPLIT`) is the shape `65317aa` named: a
128-thread workgroup per (query-head group, key partition), threads over keys
for the scores against one K load, block reductions, warps over keys and lanes
over dimensions for V, and FlashAttentionMerge's partial layout.
`backend.TestFlashDecodeKV` holds it to FlashAttention's oracle on 652 shapes
(CUDA and Vulkan); normalising every head by head 0's sum fails 66 of them.

★ IT IS NOT YET THE LEVER ON THE V100, MEASURED. `TestFlashAttentionAB` at
Phi-4-mini's shape (24/8/128), V100, against the staged sequence:

    272 keys    best 0.997x (group 1, 4 partitions)
    2048 keys   1.264x (group 1, 16 partitions)
    in the model, tg256   graph 5,299-5,327 us against the staged 5,155-5,164

(On the laptop's RTX 3050 Ti it is 1.27x at 128 keys.) The first two forms
were 0.30-0.62x: ncu read 32.7 cycles per issued instruction at 7.5% achieved
occupancy -- a dot emitted load, FMA, load, FMA waits out every load's latency
when a workgroup has four warps, where the staged kernels hide it under
thousands of threads. Issuing K blocks ahead of their FMAs over four chains
took it to parity at short context. What is left is the scores phase's
per-key cost on few warps (1,035 instructions a warp at 17 cycles each); the
layout that removes it is llama.cpp's -- warps over keys, lanes over
dimensions, the query in registers -- and it is unbuilt. Default stays the
staged path. (Superseded the next day: the next entry rebuilds the scores and
makes the kernel CUDA's default.)

### V100 decode attention as one kernel per split, CUDA's default

**What the opt-in kernel was doing in the model.** `JITLLM_CUDA_KERNEL_TIMING=1`,
Phi-4-mini, tg256 at 16+256 positions: `flashkv` 1,706 us a token against the
staged four kernels' 1,069 (both include ~3.9 us of event node per launch). The
tier had given it ONE partition (MaxSeq fit one chunk) and a whole GQA group a
workgroup -- 8 workgroups of 128 threads on 80 SMs, 53 us a layer.

**The rebuild.** Scores: lanes over KEYS (a 32-key tile, coalesced because K is
`[dimension][position]` on the device), warps over SLICES of the dimensions
(Wd slices of at most 32) times Wk groups of tiles, the query slice in
registers; a lane issues its slice's K loads at once and a key's dot is Wd short
chains folded through shared memory. llama.cpp's fattn-vec puts lanes over
dimensions, which on this cache layout would be 32 uncoalesced loads a key; the
transpose of its idea is what fits the transposed K. Values keep the previous
form (warps over keys, lanes over dimensions, row-major V).

Three knobs, each set by measurement on the V100:

- **Splits per depth.** A graph is recorded per nCap (128-key steps), so the
  tier compiles a variant per split and picks by nCap (`blockScratch.flashFor`).
  Kernel-level, Phi-4-mini's shape (24/8/128), best split against staged:
  128 keys 4 (1.59x), 512 8 (1.70x), 2048 16 (1.51x), 8192 16 (1.81x; 32 read
  1.61x). The rule is the power of two nearest sqrt(n/8), capped at ~1.6
  workgroups an SM (`Slots()/1280`), floored by what the chunk needs for
  correctness. With the tier's choice every shape tried wins at every depth:
  24/8/128 1.51-1.71x, 32/16/128 1.20-1.77x, 16/8/256 1.08-1.66x, 64/8/64
  1.43-1.71x (128 to 8192 keys).
- **Width.** One warp a 32-dimension slice: Phi-4-mini (head 128) 199.6 tok/s
  at 4 warps against 198.2 at 8; gemma-3-12b (head 256) 77.0 at 8 against
  74.1 at 4, where a lane holds 64 query values a head. `ptxas -v` on the
  generated PTX read 568 bytes of spill at head 256 with 4 warps.
- **Query heads a workgroup, capped by registers.** Uncapped, a whole GQA group
  shares a K load: gpt-oss-20b (8 heads, 32 values each) decoded at **165.9
  against the staged 189.0** and Qwen2.5-1.5B (6 heads) at 327.7 against 336.5.
  Capping the group at 96 query registers a lane turned both into wins (192.0
  and 361.5). Phi-4-mini's 3 heads (96 values, 238 registers, no spill) is the
  largest that measured a win.

**In the model** (tg128, V100 GPU 0, same pass, staged -> new):

    Phi-4-mini     190   -> 199      gemma-3-1b    302 -> 328
    gemma-3-12b     73.6 ->  77.0    Qwen2.5-1.5B  342 -> 361.5
    gemma-2-9b      92.2 ->  97.0    Llama-3.2-1B  501 -> 526
    Qwen3-1.7B Q8  247.7 -> 264      gpt-oss-20b   189.3 -> 192
    Qwen3-8B       121.7 -> 123      gemma-3-27b (2 GPUs) 37.95 -> 38.77

Against llama.cpp in that pass: Phi-4-mini 1.01, gemma-3-12b 1.02, gemma-2-9b
1.05, gemma-3-27b 1.00, Qwen3-1.7B Q8_0 1.06.

**Gates.** `backend.TestFlashDecodeKV` holds the kernel to FlashAttention's
oracle on 2,892 shapes (head widths 7/48/64/96/128/256, 2/4/8 warps, groups,
splits, windows, sinks, softcap, f16 V), on the laptop's 3050 Ti and the V100;
dropping one dimension slice from the fold fails 522 of the then 652.
`tier.TestFlashAttentionModel` now runs the KV kernel on a placed model (per-depth
split and split 3) against the staged path on CUDA and Vulkan; launching half
the workgroups fails it at logit NMSE 0.11-2.7 with a greedy flip. Its
fused-vs-staged NMSE takes one of two values (2.4e-13 or 1.9e-4 on stories15M),
swapping between CUDA and Vulkan as the width changes -- one int8 activation
rounding flip, which that test's own comment already describes.

**Defaults.** CUDA and Metal take the KV kernel unless `JITLLM_GPU_FLASH=0`
(`tier.Config.NoFlashKV`); Vulkan was not measured and keeps the staged path.
Metal, M4, same pass against llama.cpp b10985: tg128 Llama-3.2-1B 113.4 ->
114.7 (0.983 of llama.cpp's 116.7), Qwen2-1.5B 89.9 -> 90.6 (0.988 of 91.5).
Gated there by TestFlashDecodeKV and TestFlashAttentionModel. Kernel level,
TestFlashAttentionAB on msl at the tier's splits:

    shape (heads/kv/dim)    128 keys   512 keys   2048 keys
    32/8/64  Llama-3.2-1B   0.80x      1.00x      1.31x
    12/2/128 Qwen2-1.5B     1.34x      1.37x      1.64-1.67x

So head width 64 LOSES at short depth on Metal while the model still gains
1.1% in tg128 -- the kernel is chosen per device, and the next step is to
choose it per depth too, by measurement. (A first take of this table read
1.34-1.38x everywhere: the arm64 runner forwarded a test's environment only
through a variable of its own, so that run timed FlashAttention's warp-per-head kernel at the
bench's default shape. The commit message of 656cd7c quotes it.) `JITLLM_GPU_FLASH_SPLIT`, `_GROUP` and `_WARPS` force the three
knobs.

**What this does not touch.** The other half of Phi-4-mini's in-graph term,
the Q4_K matvecs (+177 us), and on gemma-3-12b the norm/rope/quantize chain:
kshim against event nodes there puts the matvecs level (10,279 against 10,282
us a token, event overhead subtracted) and jitllm's elementwise chain ~250 us
above llama.cpp's.

### Figures moved out of jit/gpu/tier comments

These figures stood in source comments under `jit/gpu/tier/` until a comment
cleanup trimmed them, and were recorded nowhere else. They are kept here
verbatim, grouped by the symbol each one justified. Unnamed "this card" in the
original comments is the laptop's RTX 3050 Ti; V100 is the 8x V100 server's
V100-SXM2-16GB; M4 is the Mac.

**`BatchTok = 16` (layer.go).** A chunk of NTok tokens with Tok columns a
thread reads the weights NTok/Tok times. tinyllama reads 0.49 GB of weights a
token and this card streams ~153 GB/s, so the ceiling is 312*Tok tok/s; 16
puts it at ~5000, where llama.cpp's prefill sat. The best Tok moved from 8 to
16 when the scores kernel got its tile (12w), with attention going from 53% to
31% of the prompt:

    Rowt=4    Tok=4   Tok=8   Tok=16      Rowt=2 Tok=16   929
    tok/s       836    1035     1096      Rowt=8 Tok=8    731

**`mmaMT, mmaNT = 4, 4` and `batchMV` (layer.go).** Swept end to end on
tinyllama's 554-token prompt: 4x4 1287 tok/s, 2x4 1029, 4x2 955, 2x8 894, 4x8
725. Wider tiles run out of registers (a lane holds MT*NT*4 accumulators);
narrower ones read the weights more times a chunk, since a pass covers 8*NT
tokens. MatVecMMA measured 1.4-3.2x the dp4a kernel on this card and
bit-identical to it. 16*4 divided every row count of every model then on the
box (2048, 256, 5632, 32000, 16384, 256128, 1024, 6144, 512, 8192, 3072, 9216,
4096, 11008, 12288), so a six-rung tile ladder never once fell back.

**`volta70AccMT = 1` (AttnAccMMA70's dims per warp in 32s).** It was the whole
head (up to four), which on Qwen3-1.7B at 512 tokens (16 heads of 128) launched
256 warps on 80 SMs. nvprof on a V100, same trace, as a fraction of the
6144x2048 GemmVolta beside it (the clock moved between runs): mt=4 0.667, mt=2
0.377, mt=1 0.337.

**sm_70 attention MMA (`initScratch`, m8n8k4).** On a V100 the FMA tiles ran
Llama-3.1-8B's 512-token attention in ~29 ms of a 180 ms prompt (scores 14.6,
accumulate 12.1) against llama.cpp's 5.4 ms flash kernel.

**`kernels.ExpertRouteWarp` (initScratch).** ExpertRoute's thread per expert
walks every logit: 31.4 us at qwen3next's 512 experts on a V100.

**Decode on `deltaScan` where the device has a subgroup (initScratch).**
GatedDeltaStep was 47.5 us a call on a V100 against llama.cpp's
gated_delta_net_cuda at 5.6 -- 36 calls, 1.7 ms of Qwen3-Next-80B's decode
token (nvprof). The fused kernel is decode-only: in the chunk as well,
Qwen3-Next-80B's pp512 fell 1515 -> 1063 tok/s on four V100s.

**`blockScratch.qkPart/qkApply` (wide q/k norm, olmoe).** HeadNorm there is one
32-lane group for 2048 elements, 22.8 us a call on a V100, against ~7 us for the
block norm's partitioned pair at the same width.

**`pickSoftmax`.** A load-time timing probe was built and removed: on an M4,
swapping the selection moved a 64-token decode from 97.62 to 97.64 tok/s.

**`pickLanes`.** The Iris Xe reports minSubgroupSize 8 /
maxSubgroupSize 32 and picks a width per shader; a probe agreed at
n=1,2,31,32,33,37 on every head while the same kernel object produced NaN
logits inside a real submission of two or more blocks, differently run to run.
The width is pinned at pipeline creation since.

**`PrepHead`'s own head split.** The fixed per-token device cost rose with NEmbd
(0.010 ms at 288, ~1.0 ms at 2048) and was flat across an 8x span of
vocabulary. A global JITLLM_GPU_SPLIT sweep lost to the tuner at every forced
value (+0.3% to +11.9%), which is why the head got a split of its own.

**`Layers` / graph capture.** One submission carries the whole prefix of blocks
-- 18 on gemma, ~500 launches -- where one block per call cost a drain per block
and put the residual on the bus 36 times a token to move 8 KB. ~450 launches at
~2.4 us is ~1.07 ms of a 7.1 ms tinyllama token, which the CUDA graph pays once.

**`canRecord` score-width rounding to 128.** ~5.7M FMA a token of surplus on
tinyllama against a 7.1 ms token, a re-capture once per 128 tokens. On a
backend with no Recorder the same rounding would run up to 128x the attention
work at the first token, so only a recording backend pays it.

**`layersOnce`, V100 prefill.**
- The head folded into a batched chunk's last row: it had been a second
  submission, 1.6 ms for the head's own session plus a 4 MB readback
  (Qwen3-1.7B, pp512).
- ActF16 conversion once per source and layout: 7.6 ms of a 152 ms
  Llama-3.1-8B prefill for 224 conversions where 128 differ.
- No int8 quantize on a block whose every batched matvec is on sm_70's f16
  path: the quantizes were 6.8 ms of that 152 ms prompt (2.6 alone, the rest
  inside the fused norm).
- The logits read straight into the caller's slice: the Go copy out of
  bs.hRaw was 128256 floats a token for a llama-3 head, 330 us of a V100 token
  on a two-socket host (pprof, layersOnce's own time).

**Correctness readings behind `layersOnce` / `prepLayer` / `mvrunAct`.**
- mvrunAct names the float source: a chained matvec on a float model read the
  block's residual, NMSE 0.796 against the host with all blocks placed.
- A post-attention norm writing bs.h while the quantize re-read it (RULE 13)
  corrupted gemma2/gemma3 at NMSE 0.09-0.32.
- The row count Conv1dShift reads is nrow, not R: a ragged 88-row chunk runs a
  96-row grid, and the padding went through the recurrence -- the qwen3next
  fixture's first decoded token after a 600-token batched prefill read logit
  NMSE 1.27 (TestHybridPrefillBatchesOnTheDevice). Written from inside emit it
  was also a memcpy inside a stream capture (CUresult 901), so every hybrid
  had run with graphs off.
- The recurrent parity flips per ROW of a batched call: held still, prefill
  logit NMSE 0.07-0.42 once hybrid captures stopped failing.
- The fused q/k/v kernel built before the biases took no bias parameters and
  was launched with three: logit NMSE 1.44 on qwen2 whenever the tuner kept
  split 1 (CUDA does not check the count).
- A recurrent block batched with one-row launches read prefill NMSE 4.3e-01 on
  the qwen3next fixture (TestHybridPrefillOnDeviceDoesNotDemote).
- `declineWeights` / row ranges of a packed tensor: Phi-3.5-mini on CUDA
  answered "The capital of France is" with 264 repeated eleven times, all 32
  blocks placed.
- The attention output's quantize (`quantE` built for NEmbd): both qwen3moe
  models are the first where nHead*headDim differs -- 16x128 against 1024 and
  32x128 against 2048.
- `TestContainerAgreesHostAndDevice`'s near-tie allowance: a mixture read "." at
  18.6352 against "," at 18.6052 on this box, and two CUDA runs picked
  differently.

**Scratch sizes.**
- `batchWidthFor`, vision: the score buffer grows as patches squared, 50.3 MiB
  for SmolVLM-256M's 1024 patches and 12 heads.
- The packed V cache: llama-2-7b's cache is 512 KB a position in f32, and
  halving V takes 2.1 GB at 4096 positions to 1.6 GB on a 4 GB card.
- `kvPair.value`, MLA: storing value apart from key would write ~1.1 GB twice
  at an 8K context on DeepSeek-V3.

**KV capacity (`kvCapFor`, `ensureKVCap`, `regrowKV`, `initKVCap`).**
- Reserving the whole MaxSeq 131072 up front cost 537 MB a block, and the card
  ran 3 of 16 blocks on a 2 GiB budget; hence growth by doubling.
- `capped`: bs.att and bs.prob at 128 rows, 32 heads and a 131072 context are
  about 4 GiB between them.
- Before initKVCap moved above the first scratch build, a 1146-position run
  built bs.p.MaxSeq 1146 against a kvCap of 256.
- A growth is ~45 ms on a V100 (mostly 64 allocations and 20 recompiles); `run
  -n 1100` took three (256 -> 512 -> 1024 -> 1107), ~1.5% of the run -- hence
  jumping straight to the context asked for when it is within 8x.
- `regrowKV` used to go through the host: 356, 681 and 956 ms for an
  8B's three growths to 1107 positions on a V100, 2.0 s of a 12.6 s decode.

**`embed.go` (the embedding gather rides the next submission).** Gathering and
reading the rows back cost 2.1-3.3 ms for 4 MB on the V100 host (first-touch
faults in a fresh prompt buffer) against the kernel's 57 us; riding the chunk's
Layers call also skips a 0.65 ms upload of x.

**`mlabatch.go`.** Before it, a DeepSeek-V2-Lite prompt went one row at a time
at 123 tok/s against llama.cpp's 2070 (V100). The FMA accumulate over the
latent ran at 0.66 ms a layer (the scores at 5.87 are recorded above).

**`moegroup.go`.**
- Before batching, Qwen3-30B-A3B prefilled at 136 tok/s against llama.cpp's
  1050 (V100).
- `voltaUnit`: at Qwen3-30B's ~32 tokens a run, 128x32 455 us against 128x64
  514.
- No dp4a half on sm_70: on gpt-oss-120b's 6016 padded columns that scratch
  would be ~0.3 GB.
- Biases in the GEMM epilogue: on gpt-oss-20b (nvprof, V100, one 512-token
  chunk) indexedbiasadd was 11.1 ms and gatherrows 3.7 ms of a 147.6 ms prompt.

**`rows.go` (batched decode, V100, Llama-3.1-8B Q4_K_M).**
- `LayersRows`: the prefill twin at 32 rows gave a 1024-row matvec 512 threads,
  and the step was 147-151 ms at every batch size.
- `ragMV`, graphs on, after a warm-up, tok/s at 16/32/64/128 rows: Rowt 2
  734/891/967/1020, Rowt 4 607/822/1005/1116; Tok 4 and 16 lost at every width
  (shipped: Tok 8, Rowt 2 below 64 rows and 4 from there).
- `dot4Split`: at 128-token prefill chunks a 1024-row k/v projection took 458
  us against 532 for a 14336-row one with fourteen times its work; the narrow
  matrices were most of a 0.62 s 512-token prompt.
- `voltaMV`: 14336x4096 Q4_K at 128 tokens, 287 us against the dp4a twin's
  652 (2.3x), MT=2 NT=4 swept against 1-8 x 1-8.
- `voltaTiles`: MT=4 NT=4 over one warp row ran 14336x4096 at 34-37 TMAC/s
  against 30-32 for a 2x2 warp grid (512 tokens).
- Recurrent states sized to the real sequences: Qwen3.5-0.8B's five sequences
  padded to 32 asked for 1.2 GB of states (~2 MB a sequence a block) and were
  refused.

**`chooseSplit` (tier.go), the fitted table, RTX 3050 Ti, GB/s by split, winner
in brackets:**

    shape                      1     2     4     8    16    32
    k/v      256x2048         10     9    10   [13]   10    11
    q/o     2048x2048         52    56    63    65    63   [66]
    gate/up 16384x2048       111   130  [135]  134   128   120
    ffn_down 2048x16384       64   112   138  [141]  132   137
    lm_head 256128x2048     [162]  155   158   153   148   151

The rule "about 2x the thread slots, cap at 32" landed on or beside the winner
except k/v, a 0.04 ms gap at its 11.8 MB a token. Swept against a 148 GB/s wall
on gemma's shapes, what the rule cost (the case for `tuneSplit`):

    shape                rule      best     lost
    qkv    2560x2048     32: 62    8: 73     15%
    o      2048x2048     32: 54    4: 62     13%
    gate/up 16384x2048    4: 134   1: 152    12%
    ffn_down 2048x16384  32: 125   8: 136     8%
    lm_head 256128x2048   1: 162   1: 162     --

~1.06 ms of a 13.7 ms gemma token.

**`tuneSplit` / `timeSplitMode` / `retuneDecode` / `splitCands` (split.go).**
- The msl override, since removed, read: table split 332.9 ms for a 32-token decode
  against the measured split's 354.6 (6.5% slower). It was timing the enqueue
  (Metal `Sync` did not wait); see the Metal entries above.
- In-group reduction on an M4: +39% on q/o (2048x2048), +34% on k/v, -12% on a
  Q6_K head -- so it is timed per shape, not switched.
- Single-matrix trials on a V100 picked 2048x2048 split 4 in-group at 11.0 us
  a launch in one process and split 32 at 4.8 in the next. The decode sequence
  (16 blocks) streams 38 MB of the smallest shape, past a V100's 6 MB of L2.
- Divisors, not powers of two: gpt-oss's k of 2880 is 90 blocks, and the
  power-of-two walk left its 512x2880 k/v projections at split 2 with as few
  as 1,024 threads, 31 us each (~50 GB/s) on a V100.
- The indexed split was a 3050 Ti table: on a V100 gpt-oss's gate/up/down ran
  at split 2 over 11,520 rows -- 23,040 threads for 163,840 slots -- at 357
  GB/s, where llama.cpp's MUL_MAT_ID moves the same bytes at 535.
- `segLaunch` / fused epilogues: kernels.IndexedBiasAdd is 2.3 us each on a
  V100, 0.16 ms of a 5.7 ms gpt-oss token. MLA's kv_a projection (576 rows)
  alone is a 6.2 us launch reading 0.68 MB (~110 GB/s, V100 nvprof,
  DeepSeek-V2-Lite), beside a 3072-row q projection on the same input.

**The per-matvec path (tier.go `matVec`, `prepare`, `quantPar`, `verify`).**
- Two uploads, a launch and a readback were four ownership hops at 5.62 us
  each; batched they are one.
- quantPar fanned 2048 floats (64 blocks) over six goroutines at 46 us a
  quantization, where a serial pass is a couple.
- `verify` reported per-row relative error first and flagged 2-11% of rows on
  llama-2-7B -- the metric failing near cancellation; it uses NMSE.

## The GPU JIT's founding measurements -- one IR, three backends, and a tile that does not transfer

Moved from ARCHITECTURE.md (§5a and §5a-bis) when that file was retired; nothing
else recorded them. Paths and names were checked against the tree at that move. At the time the
driver libraries were reached through purego and the tier lived in a second Go module; today the FFI
is goffi (`backend/ffi`) and there is one module. Every number below is from the tier's first measurements and predates
whole-block submission (RULE 12), so it describes kernels, not tokens.

**The design said the GPU tier needed no generated code, and the first prefill probe reversed that.**
The claim was made about DECODE, where one hand-written `__dp4a` kernel already read 169.7 GB/s (96%
of the RTX 3050 Ti's theoretical) against 17.5 GB/s for the "obvious" f32-dequantize-then-FMA kernel
-- a memory-bound matvec leaves a generator nothing to win. Prefill is compute-bound.
`dev/probe/gpu` generates an int8 GEMM as PTX at run time with K, the register tile and the output
stride baked as immediates, and swept the tile on the 3050 Ti, every entry bit-exact against an int32
CPU reference:

    tile    Gmac/s        tile    Gmac/s
    1x1      133.8        8x4      887.9
    2x2      260.9        4x8      935.9
    4x2      495.5        8x8     1508.2
    4x4      479.8        8x16   ~2447 (median of four runs; one run read 2497.8)
                          16x8    2349.3

**18.7x from the tile alone**, against 1.63x for the same sweep on the CPU (`TestGEMMTileSweep`).
ptxas owns register allocation, but the TILE is a source-level decision the generator makes. The probe
also measured codegen plus ptxas at 8-28 ms per shape against ~18 us for the CPU emitter; the driver's
on-disk cache keyed on the PTX text (`~/.nv/ComputeCache`, see `backend/cuda/api.go`) is what makes a
second process pay 0.108 ms instead of 36.7. One more constraint came out of the probe and is why the
PTX emitter writes `.version 7.8`: nvcc 12.9 emitted `.version 8.8`, which driver 550.163.01 (CUDA
12.4, maximum ISA 8.4) rejected at RUN time with "compiled with an unsupported toolchain".

**Then the IR.** `jit/ir` is a typed IR (about two dozen ops on that day) lowered by three
backends: `ptx` (run through `backend/cuda`), `spirv` (`backend/vulkan`) and `msl`
(`backend/metal`). Verified on the RTX 3050 Ti (PTX and Vulkan), Intel Iris Xe, llvmpipe and the
Apple M4. Three things it established:

1. **Buffers are indexed by element, not addressed by byte pointer.** Byte-pointer arithmetic is a
   PTX-ism; SPIR-V's Logical addressing model has no raw pointers and Metal has the same shape. The
   move cost PTX one `mad.wide.u32` per load and bought two targets. (That per-load widening is
   priced later in this file, under RULE 12's "every Load widens its own 64-bit address".)
2. **Apple's GPU has no packed int8 dot product.** `metal.TestPackedDotIntrinsic` puts four candidate
   spellings to the M4's compiler and all four are rejected, so the Apple int8 inner loop is roughly
   4x the arithmetic of the other targets.
3. **`spirv-val` is a gate, not a lint.** Two emitter bugs -- `OpVariable` with its result type and
   result id swapped, and `PackedVectorFormat4x8Bit` written as 1 where it is 0 -- each crashed
   NVIDIA's compiler inside `vkCreateComputePipelines` with a SIGSEGV and no diagnostic; the validator
   named each in one line. `jit/spirv/spirv_test.go` runs it when it is installed.

**Bit-exact across five GPU stacks and three vendors, not within tolerance.** int8 x int8 into int32 is
integer arithmetic, so two correct backends agree to the last bit, and a tolerance would have hidden
exactly the bugs these backends had: an access chain off by one element, a phi wired to the wrong back
edge, a packed dot reading its operands in the wrong order -- each of which yields a plausible number.

Throughput (`dev/tools/gpubench`, 4096x2048 x 128 tokens = 1073.7 Mmac a launch, five warm-up
launches then 21 timed rounds, median with IQR, every tile checked against the full 524,288-output
int32 reference first):

| device | API | best tile | Gmac/s | IQR/med | launch overhead |
|---|---|---|---|---|---|
| RTX 3050 Ti | ptx | 8x16 | 2468.0 | 0.013 | 19.3 us |
| RTX 3050 Ti | spirv | 8x16 | 1368.2 | 0.101 | 84.7 us |
| Apple M4 | msl | 8x4 | 343.3 | 0.008 | 288.8 us |
| Iris Xe (ADL GT2) | spirv | 4x8 | 136.1 | 0.009 | 494.2 us |
| llvmpipe | spirv | 4x2 | 26.3 | 0.010 | 115.7 us |

**The best tile differed on every device, and the wrong one is expensive.** NVIDIA's 8x16 run on the
M4 measured 152.1 against 343.3, a 2.26x loss, which the missing packed dot predicted. The best tile
also moves with problem SIZE: at 512x512x32 the best PTX tile was 4x2, because large tiles cannot fill
the GPU. A tile table is not portable across devices or shapes.

**Host-visible Vulkan buffers cost 8x.** The first version mapped everything host-visible and read 172
Gmac/s against CUDA's 2478 on the same card. The arithmetic named it: 6.2 ms a launch, and a 4096x2048
weight re-read once per token tile (128 tokens at TN=16 is eight passes) is 64 MB, which over PCIe at
~10 GB/s is 6.4 ms. Staging through a host-visible scratch buffer took it to 1368, 8.0x;
`vulkan.Buf.DeviceLocal()` reports which path a buffer took. Of CUDA's remaining 1.8x over Vulkan on
that card, dispatch alone was 20% (84.7 against 19.3 us per launch on a ~0.43 ms kernel).

**CPU against GPU on the same int8 GEMM** (Q8_0, k=2048, best tile, 21 rounds with samples padded to
~5 ms; the CPU side is `nn.TestGEMMPoolRate` across the whole P-core pool, because comparing a GPU
with ONE core overstates it by the core count):

| machine | CPU GEMM | GPU GEMM | GPU/CPU |
|---|---|---|---|
| i9-12900HK + RTX 3050 Ti | 681.3 Gmac/s (6 P-cores, IQR 6.6%) | 2468.0 (ptx, IQR 1.3%) | 3.62x |
| i9-12900HK + Iris Xe | 681.3 | 136.1 (spirv, IQR 0.9%) | 0.20x |
| Apple M4 mini | 454.0 (4 P-cores, IQR 1.4%) | 343.3 (msl, IQR 0.8%) | 0.76x |

The M4 row is a statement about that KERNEL SHAPE, not about the hardware: `msl` then emitted a naive
register-tiled kernel with no threadgroup memory and no matrix units, while `metal.TestSimdgroupMatrix`
already showed `simdgroup_float8x8`/`simdgroup_half8x8` multiply-accumulate, threadgroup memory and simd
reductions compiling on the M4. 343 was a floor for that kernel; the simdgroup_matrix prompt GEMM is
the M4 Metal-rows entry above. The Iris Xe shares the P-cores' 55.7 GB/s memory, so it adds no bandwidth;
the placement consequence is measured in AGENTS.md's Scope section (+10.2% with the discrete card,
-15% with the iGPU). The darwin single-core CPU arm came back at IQR/median 50.8% and "135% of
linear" and was not quoted: with no thread affinity on macOS one goroutine migrates between P and E
cores, while four busy workers hold the P-cores.

## Paged KV decode kernels against the contiguous ones

The paged writers and decode kernels (docs/design/device-kv-paging.md) were
built beside the contiguous ones -- which emit byte-identical IR after the
change -- and measured against them kernel for kernel: same pass, A/A control
first (0.997-1.004, IQR/median under 5%), each arm rotating through per-layer
KV so reads come from DRAM, P=256 with shuffled pages, the contiguous K stride
n+1 as in the tier. Llama-3.2-1B and Qwen3.5-0.8B decode shapes at 512 / 4096 /
16384 keys, paged/contiguous (above 1 is paged faster):

    FlashDecodeKV (CUDA/Metal default)
      CUDA 3050 Ti   0.994 / 1.018 / 1.045    1.002 / 1.026 / 1.090   133 GB/s = 80% of 166
      V100           0.990 / 1.010 / 1.034    0.980 / 0.996 / 1.027   631 GB/s = 77% of 822
      Metal M4       1.024 / 1.040 / 1.037    1.083 / 1.147 / 1.070    71 GB/s = 69% of 104
      Vulkan RTX     0.957 / 1.024 / 1.088    0.974 / 1.023 / 1.102
    staged, bounded chunks + merge (Vulkan's default path)
      CUDA           1.10 / 1.27 / 2.93       1.014 / 1.57 / 2.83     86-90% of wall
      Vulkan RTX     1.07 / 1.09 / 2.78       1.03 / 1.35 / 2.70      81-84%
      M4             1.28 / 1.40 / 2.36       1.38 / 2.21 / 2.90      85-94%

Where paged is slower: FlashDecodeKV at 512 keys by 1-4% on CUDA, the V100 and
Vulkan RTX -- one dependent load at kernel start (descriptor, table, K) that the
ABI makes unavoidable; and staged on the V100 at 512/4096 keys with the laptop's
split choice (0.62-0.79), which is the split count, not the kernel: at 64 splits
it reads 1.16-1.23 at 4096. The split has to follow the SM count, as
flashSplitFor already follows Slots. The Iris Xe runs both arms at 1-16% of its
wall and uses neither kernel.

Three findings about the kernels that ship, left untouched by that work:

- **Contiguous FlashAttention loses 3.4x on 256-wide heads**: it interleaved its
  V loads with their FMAs. Grouping them as the paged form does was worth
  1.47-1.67x on PTX and the paged form is still 2.0-2.4x ahead, so it was not
  the whole difference (see "Faster paged decode attention").
- **Staged is the fastest decode at long context on the RTX 3050 Ti**: paged
  staged at 16384 keys is 1.8x paged FlashDecodeKV there, the path CUDA takes by
  default. On a V100 FlashDecodeKV wins at every depth but the 256-wide head at
  16384 keys; the crossover is the device's ("Faster paged decode attention").
- **The fastest K addressing is per driver**: one base plus constant offsets on
  Metal, each index computed on NVIDIA's Vulkan (the other way runs at 0.36x),
  within a point on PTX -- hence FlashShape.KImm.

### The tier on paged history, against the contiguous cache

Decode and a batch's sequences read through paged FlashDecodeKV, a prefill
chunk through the paged prefill kernels (below), a window is the descriptor's
keyStart, and the contiguous cache is the control arm. The accuracy table was
taken with prefill on FlashDecodeKV; with the prefill kernels the Llama arm is
bit-identical to the contiguous one on the laptop. CUDA (RTX 3050 Ti), page 64, the same ids
fed to both arms, worst logit NMSE over every step:

    Llama-3.2-1B, 92-token prompt + 24     gemma-3-1b, 902-token prompt + 24
    paged vs contiguous       2.1e-03      paged vs contiguous       1.0e-02
    contiguous vs host        3.9e-03      contiguous vs host        8.5e-03
    paged vs host             1.7e-03      paged vs host             7.5e-03
    row at a time, paged
      vs contiguous           0            (window) 7.7e-03
    order-only control        2.1e-03      order-only control        6.3e-03
    argmax flips              0            argmax flips              0

The order-only control is the contiguous cache against itself at two split
counts: nothing but the summation order moves, and it moves Llama's logits by
2.1e-03 and gemma's by 6.3e-03. That is the model's own sensitivity, so the
gate's bound is that control measured in the same run, not a constant; a first
bound of 1e-4 was a guess and failed on arithmetic that was right.

- **Without a window the paths are bit-identical, row at a time.** The chunked
  prefill gap is the contiguous prefill scoring in binary16 on the matrix
  units, and paged sits closer to the host (1.7e-03 against 3.9e-03).
- **With a window they are not, and that is order.** Inside gemma's 512-key
  window the rows are bit-identical too; past it, the contiguous kernel starts
  its tiles at the window's first key and the paged one at the 32-aligned
  position below it, masked. Same keys, a different summation grouping.
- **The gate discriminates.** Under each violation the paged arm lands far
  outside the band: a page index of (t-1)/P 239x, unaligned tiles 75x, a
  windowed layer reading the full descriptors 63x, and every batch row reading
  row 0's pages fails the batch gate at its first token by a 2.8-logit gap.

### Prefill on pages: the kernels were half of it, the pool the other half

Prefill through paged FlashDecodeKV measured 0.46-0.48 of the contiguous cache
on the V100 (Llama-3.2-1B Q4_K_M, 512 and 4096 tokens): one row per descriptor,
every row reading its keys on its own. A chunk is one sequence's rows in order,
so the paged prefill kernels serve it -- FlashPrefill70 at whole 64-row chunks
on sm_70 and FlashPrefillTile on Metal, else the staged scores, softmax and
accumulate (m16n8k16, sm_70's m8n8k4, or the FMA tiles), built per span of
keys from 64 up and run in passes of 4096 keys past that. That took the V100 to
0.48 at 512 and 0.69 at 4096 -- and a CPU profile put a third of the process in
`allocLayerPool`: every request grew each layer's pool and shrank it again on
close. With released pages kept in the pool (the design doc's Shrink), same
binary otherwise, V100 CUDA, same-pass ABBA, paged/contiguous:

    prefill   512 tokens   1.1127 / 1.1106   IQR 0.6-0.8%   A/A 1.0008
              4096 tokens  1.5821 / 1.5770   IQR 0.9-1.3%   A/A 0.9957
    decode    512 deep     0.9755 / 0.9803   44% of the 822 GB/s wall
              4096 deep    0.9934 / 0.9969   49%

The prefill A/A read 1.12 and 1.06 before; the churn was its noise too. The
gates: every prefill violation fires on both hosts -- (t-1)/P 271x (673x on the
V100), unaligned 40x, a window lost 65x, and a pass reading every key
(`passclip`) 53x -- with the clean batched arm inside the band. The laptop row
is not taken: its card was shared with another session throughout.


## Paged KV prefill kernels against the contiguous ones

The paged prefill kernels (docs/design/device-kv-paging.md, "Prefill") were
built beside the contiguous ones -- whose IR is pinned byte for byte by
kernels.TestContiguousPrefillIRUnchanged -- and measured against the path the
tier runs on each device: one 512-row chunk (256 at 16384 keys on Llama, where
the contiguous planes would not fit a 4 GB card twice), same pass, A/A control
first, two ABBA passes of 20 rounds, P=256 shuffled, the contiguous K
transposed at stride n+1. The rate is attention GFLOP/s (two products of
2*Dim flops per row, head and attended key). paged/contiguous, above 1 paged
faster, Llama-3.2-1B then Qwen3-1.7B:

    staged, the direct form (one chunk, no merge) to 4096 keys, split C=256 in
    passes of 4096 keys folded by FlashAttentionMergeRun beyond
                         512 keys        4096 keys       16384 keys
      CUDA 3050 Ti       1.28 / 1.32     1.06 / 1.06     1.31 / 1.26 *
      Vulkan RTX         1.07 / 1.14     1.05 / 1.07     1.25 / 1.28
      Metal M4           1.17 / 1.19     1.14 / 1.13     2.00 / 1.64
      V100               0.98 / 0.94     1.05 / 1.02     1.32 / 1.21
    flash, Splits 0
      FlashPrefill70 V100   0.96 / 0.99     0.97 / 0.99     0.98 / 0.97
      FlashPrefillTile M4   1.00 / 0.99     1.01 / 1.00     --

    * Qwen at 16384 on CUDA is the take before the tile-limited softmax, which
      only removes paged work; the re-take ran out of the 4 GB card.

A/A controls 0.99-1.01 on every row. IQR/median 0.0-0.7% on the V100 and the
M4, 1.6-7.6% on the laptop's shared card; every quoted pass under the 10% gate
and the two passes within 1.4 points. The m16n8k16 scores run at 530-590
GFLOP/s contiguous and 610-780 paged on the laptop; m8n8k4 at 4.6-6.9 TFLOP/s
on the V100; FlashPrefill70 at 12.9-23.9 TFLOP/s; FlashPrefillTile at
1.2-2.2 TFLOP/s on the M4.

Why the paged staged path is faster where it is: the contiguous scores
compute every key tile of the context-wide plane and mask half of a causal
chunk; the paged m16n8k16 and m8n8k4 scores skip a tile none of their rows
attends (1.11x on the tiled scores alone on Vulkan, 1.03 on the V100). The
softmax reads and writes only the slots its query tile attends, and at long
context the contiguous softmax makes three passes over a plane of the whole
context -- 1.08 GB at 16384 keys -- where the paged one works on a pass.

What was slower and what it was:

- **The first staged softmax wrote every slot of every chunk**, twice what a
  short-context row attends, and the merge read S partials per element: 0.44x
  the contiguous softmax on Vulkan at 512 keys, 0.80x the whole path on the
  V100. Writing only the query tile's span and the direct form (Splits 0:
  normalised in the softmax, no partials, no merge) took the V100 to 0.93-0.95
  and every other device to a win.
- **A division by a power-of-two constant is a whole division on sm_70.** The
  PTX lowering hands div.u32 its divisor in a register and ptxas emits I2F,
  MUFU.RCP and a correction even for 4; the paged kernels divide by their
  tiles and aligns throughout. With shifts: the m8n8k4 accumulate 720 -> 576
  SASS instructions, the V100 staged path at 512 keys 0.947 -> 0.978 (Llama).
- **The m8n8k4 accumulate first read a page id per 4-key step**: 0.87-0.93
  per kernel. Walking each page's run of keys by pointer -- an outer loop over
  pages, the inner loop identical in SASS to the contiguous kernel's (64
  instructions) -- took it to 0.95 / 0.89; what remains is the masked first
  and last steps and the prologue's dependent loads (descriptor, table, V).
- **The softmax in groups of two warps and with divisions**: 0.88 / 0.90 per
  kernel on the V100; a warp a workgroup and shifts, 0.98 / 0.96.
- **Hoisting FlashPrefill70's staging addresses out of the tile loop cost
  registers and 4 points** (0.965 -> 0.926 at 512 keys, Llama); reverted. The
  same hoist helps FlashPrefillTile on Metal (0.915 -> 0.922 at d128, 0.98 ->
  1.01 at d64).
- **The flash kernels read a page id per key tile**, prefetched a tile ahead.
  FlashPrefillTile at head width 128 (16-key tiles) sat at 0.89-0.93 on the M4
  at every context length and at P=64 (0.90) as at P=256 (0.89), so not the
  page stride; a probe that read one constant page for every tile (wrong
  answers, right traffic) ran 1.06 / 1.09, which put the cost on the table
  read -- a dependent load at the head of every trip. An outer loop over the
  run's pages with the tiles in an inner loop reads the id once a page:
  FlashPrefillTile 0.93 -> 0.99 at d128 (1.00 -> 1.00-1.01 at d64), and
  FlashPrefill70 on the V100 0.951 -> 0.964 (512 keys), 0.957 -> 0.971 (4096),
  0.973 -> 0.983 (16384) at d64 and within half a point at d128 (0.993 ->
  0.994, 0.993 -> 0.987, 0.971 -> 0.967). What FlashPrefill70 keeps on the
  V100 is the prologue: descriptor, then table, then the first K tile, two
  dependent loads the contiguous kernel does not have. The paged decode
  kernels record the same at 512 keys.
- **An unmasked start for the m8n8k4 accumulate** -- a masked head step only
  when keyStart is not 4-aligned, the middle by pointer from there -- measured
  0.5 points slower on the V100 (the accumulate alone 0.952 -> 0.945 Llama,
  0.895 -> 0.891 Qwen; the path 0.978 -> 0.973, 0.940 -> 0.934); reverted. The
  staged path at 512 keys on the V100 is what misses the bar: the accumulate
  at 0.95 / 0.89 is most of it, and its inner loop is the contiguous one's to
  the SASS instruction, so what is left is the per-row masked first and last
  steps and the descriptor and table loads ahead of them. At 4096 keys the
  same path wins.

Correctness: the gates (backend/pagedprefill*_test.go) pass on the laptop's
CUDA, Vulkan RTX and Iris Xe, the V100 and the M4 -- shuffled tables,
NaN-poisoned pools, planes and outputs, page edges, keys ending mid-page,
windows and aligned chunks, padded rows, sinks, softcap, GQA, packed f16 V,
MLA's row-major region, the direct form, split counts past the keys, passes
-- against the float64 oracle (1e-10 FMA, 1e-5 binary16 operands) and the
contiguous kernels (6e-8 at worst), and each fails through a wrong table,
keyEnds one short, a dropped pass, and the generation faults (t-1)/P,
unaligned and vmask.

### Every staged prompt on Metal was wrong, and every kernel gate was green

The tier launched the staged prefill softmax 64 wide. With 32 lanes the
kernel is a 32-wide workgroup an item (`kernels.PagedPrefillSoftmaxWidth`):
CUDA and Vulkan keep the declared width and were right, and Metal runs a
launch at the width it is given, so two subgroups worked one item. Every
prompt chunk of two or more tokens through the staged form prefilled wrong on
the M4: Qwen3.5-0.8B's full-attention blocks (head width 256, which no flash
prefill takes there) read **0.38** logit NMSE against the same prompt a
token at a time, against 2e-4 to 6.5e-4 on CUDA and Vulkan. The dense board
models never reached it -- their heads take FlashPrefillTile on Metal --
and the backend gates launch each kernel at its own width, so they could not
see it. It was found because a step-across-sessions gate compared a chunked
prompt against Prefill on the same device, and the reference was the wrong
arm. The tier keeps each staged variant's width now (`prefillVariant.softW`):
0.38 -> 2.5e-4 on the M4. Gate: `model.TestStagedPagedPrefillMatchesDecode`.

### A launch at a width the kernel does not declare is refused on every backend

The same mismatch came back the other way round: `kernels.HCMix` declared a
64-thread workgroup and the tier launched it 128 wide, so CUDA (which runs the
launch's width) was right and Vulkan (which runs the SPIR-V's LocalSize and
ignores the launch's) computed the first 64 rows of every 128 and left the rest
stale, with no error and a clean validation layer. Each backend answers a
mismatch differently -- CUDA the launch's width, Vulkan the declared one,
Metal the launch's while msl bakes the declared one into ir.OpNTID -- so the
class is closed at the backend rather than kernel by kernel: every
`Kernel.Launch` and `Session.Launch` on CUDA, Vulkan and Metal refuses a width
other than the kernel's declared `Group[0]` (or a group with a second or third
dimension, which no one-dimensional launch can match), naming the kernel
(`backend.checkWidth`), and the model's device-failure errors carry the
device's reason, so the name reaches the caller.
`backend.TestEveryBackendRefusesALaunchWiderThanItsKernel` launches a 64-wide
kernel at 64 (every row written), 128 and 32 through both paths. The 64-wide
`HCMix` put back fails `TestDeepseek4LongChunksOnEveryDevice` on cuda:0 too
now, which it passed before, with "hcmix declares a 64x1x1 workgroup and was
launched 128 wide"; the "softwidth" fault (jitllmfault, the staged softmax
launched 64 wide) is refused by name on cuda:0, vulkan:0 and vulkan:1
(`TestStagedPagedPrefillGateDiscriminates`).

To find any other kernel with the mismatch, every `engine/model` test that
opens a device tier (270), and the `jit/gpu/tier` and `backend`
packages, ran with every refusal logged: on the RTX 3050 Ti's cuda:0 and
vulkan:0, and on the V100 box in eight slices, one card each (cuda:0 and
vulkan:k), plus the tier package. The log held the deliberate violations and
nothing else: no other kernel is launched at a width it does not declare.

## Faster paged decode attention

Each technique was built behind the paged gates and measured against the form
before it with `backend.TestFlashChangeAB` (`JITLLM_FLASH_EXP` names the
experiment; `-tags jitllmbench`): one decode environment, both arms reading
per-layer KV so the reads come from DRAM, A/A first, two ABBA passes of 20
rounds, the read wall measured in the same process. Shapes are Llama-3.2-1B
(32/8 heads, 64 wide) and Qwen3.5-0.8B (8/2, 256 wide), one row, at 512 /
4096 / 16384 keys, P=256 with shuffled pages. Ratios are after/before (above 1
is the change faster); "u" marks a pass above the 10% IQR gate. Devices: RTX
3050 Ti under CUDA (166 GB/s wall, 30720 resident threads) and under
NVIDIA's Vulkan driver, a V100 under CUDA (822 GB/s, 163840), and the M4
under Metal (104 GB/s). An alternative emission is compiled in only under the
bench tag (`kernels.SetBenchKnob`); a deliberate violation of each shipped
change is a `jitllmfault` generation fault that `TestPagedGatesDiscriminate`
demands fail.

**Ranked by what a decode token gains:**

| technique | 3050 Ti CUDA | 3050 Ti Vulkan | V100 | M4 Metal | verdict |
|---|---|---|---|---|---|
| binary16 K, FlashDecodeKV | 1.10 / 1.28 / 1.29u; 1.24 / 1.27 / 1.30u | 1.09 / 1.25 / 1.26; 1.04 / 1.22 / 1.30 | 1.05 / 1.12 / 1.18; 1.07 / 1.15 / 1.22 | 1.25 / 1.26 / 1.37; 1.37 / 1.65 / 1.49 | kernels ship; tier needs prefill and a precision gate |
| binary16 K, staged | 1.13 / 1.24 / 1.25; 1.16 / 1.18 / 1.28u | 1.08 / 1.15 / 1.20; 1.12 / 1.11 / 1.25 | 1.04 / 1.07 / 1.09; 1.13 / 1.14 / 1.10 | 1.20 / 1.26 / 1.28; 1.29 / 1.27 / 1.31 | as above |
| ChooseDecodePlan against the tier's path and split now | 1.24 / 1.00 / 1.07u; 1.06 / 1.58 / 1.91 | 1.08 / 1.33 / 1.00u; 1.57 / 1.08 / 1.00 | 1.02 / 1.00 / 1.00; 1.11 / 1.52 / 1.83 | 1.00 / 1.00 / 1.23; 1.03 / 1.76 / 2.29 | ship: never below 0.998 |
| FlashAttention's contiguous V loads grouped | 1.34 / 1.34 / 1.35; 1.47 / 1.48 / 1.67 | 1.00 / 0.99 / 1.00; 1.00 / 1.00 / 1.00 | -- | -- | ship (opt-in kernel) |
| four warps past 128 dims at 4096+ keys | --; 0.95 / 1.24 / 1.51 | --; 0.55u / 0.81u / 1.22u | --; 1.01 / 1.36 / 1.52 | --; -- / 0.88 / 1.09 | in FlashKVPlan; not on Vulkan below 16384, not on Metal |
| FlashAttentionMergeWide for FlashDecodeKV's partials | 1.00 / 1.00 / 1.00; 1.05 / 1.03 / 1.06 | 1.00 / 1.00 / 1.00u; 1.03 / 1.03 / 1.02u | 1.02 / 1.01 / 1.00; 1.11 / 1.48 / 1.23 | 1.00 / 1.00 / 1.00; 1.02 / 1.01 / 1.01 | in FlashKVPlan |
| staged splits that fill the device | sqrt(n/8) was right | sqrt(n/8) was right | 1.11 / 2.01 / 1.79; 1.06 / 2.43 / 2.01 | sqrt(n/8) was right | in PagedStagedPlan |
| FlashDecodeKV's split cap rounded up | 1.22 at 512 (Llama) | 1.17 at 512 (2 splits) | unchanged | -- | in FlashKVPlan |
| vector V loads in FlashDecodeKV (VecV) | 1.016 / 1.015 / 1.010; 1.002 / 1.005 / 1.000 | not lowered | 1.007 / 1.019 / 1.011; 1.002 / 1.007 / 1.007 | 0.999; 1.001 at 4096 | ship on PTX only |
| hoisting the descriptor (ceiling, baked) | 1.003 / 1.009; 1.019 / 1.000 | 1.025 / 1.003u; **0.64 / 0.84** | -- | -- | not built |
| four score chains in FlashAttention | 0.96 / 0.95; 1.05 / 1.01 (paged 0.90 / 0.89; 0.95 / 0.95) | -- | -- | -- | refused |
| one query head a FlashDecodeKV workgroup | 1.00 / 0.75 / 0.76; 0.97 / 1.12 / 0.72 | -- | -- | -- | refused |

Llama first, then Qwen, at 512 / 4096 / 16384 keys unless a cell says
otherwise.

**binary16 K (`FlashShape.F16K`).** K is half of what a decode reads from the
cache with f32 V and two thirds with f16 V, and it was always f32: the tier's
own comment said the NEOX rotation, whose outputs are half a head apart, stood
in the way. The layout packs two DIMENSIONS of one position into a word --
`[kvRow/2][P]` words a page, the pair (e, e+1) of position t at
`pid*P*kvRow/2 + (e/2)*P + t%P` -- so a writer always writes whole words:
`PagedCopyRowsTF16` a pair a thread, `PagedRoPERowsTF16` one interleaved
pair or two adjacent NEOX pairs (four dimensions, two words; nRot a multiple
of 4) a thread, and no two threads ever share a word. The three decode
readers (FlashDecodeKV, the staged scores, paged FlashAttention) read a word
for two dimensions where a slice is whole pairs and a half by parity where it
is not (a 34-wide head's 17-wide slices). f32 V's byte ceiling is 1.33x; the
rows sit at 1.04-1.30x, and Metal passes it (1.65x at Qwen's 4096 keys) where
halving the K loads relieves an issue-bound phase as well as the bus. What it
costs is precision -- K rounded to binary16 changes every score -- which is
llama.cpp's default cache and is the model-level gate's to price, not this
bench's. Gated with shuffled pages, windows, sinks, softcap, f16 V beside it,
the 256-wide head and the odd slice on CUDA and Vulkan; halves read high
first fail at NMSE 0.15-1.03, and a NEOX rotation that swaps a word's halves
fails the writer gate. Prefill's paged kernels (attn.go, flash70.go,
flashtile.go) do not read it yet, and the tier cannot turn it on before they do.

**ChooseDecodePlan** is the combination the rows below it motivated, measured
against what the tier runs now (FlashDecodeKV at `flashSplitFor` with the
group-per-item merge on CUDA and Metal; the staged path at sqrt(n/8) splits on
Vulkan). The path: staged where keys x Dim reaches 24 x the resident threads on
PTX and Metal (Metal at its 16384-thread fallback), from 16384 keys on
Vulkan; the KV plan takes four warps past 128 dims at 4096 keys (16384 on
Vulkan, never on Metal), the split cap rounded up, and the wide merge; the
staged plan raises
sqrt(n/8) until the accumulate fills a quarter of the device. Every cell is
at or above parity (0.998 the lowest), and the long-context 256-wide rows are
where it pays: 1.58x and 1.91x on the RTX 3050 Ti, 1.52x and 1.83x on the
V100, 1.76x and 2.29x on the M4 -- 86-89% of the laptop's wall, 74% of the
V100's and 89-95% of the M4's, where the tier's choice sat at 40-55%.

**Why the path depends on the device, not only the shape.** Paged staged
against paged FlashDecodeKV, each at its plan (above 1 is staged faster):

    RTX 3050 Ti CUDA   0.86 / 0.77 / 1.08u    0.79 / 1.25 / 1.39
    RTX 3050 Ti Vulkan 0.95 / 0.76 / 1.03     0.70 / 0.93 / 1.73   (*)
    V100               0.51 / 0.72 / 0.92     0.39 / 0.72 / 1.16
    M4 Metal           0.94 / 1.03 / 1.23     0.87 / 2.01 / 2.10   (**)
    (**) against FlashDecodeKV with four warps, which Metal does not want
         at 4096 keys (0.88x); against its default width, 1.76x there
    (*) the staged path at sqrt(n/8) against FlashDecodeKV at the tier's
        rule; the Vulkan plans are the same but for four warps, which
        that driver does not want below 16384 keys

The crossover moves by four between two NVIDIA parts (keys x Dim of about 1M
on the 3050 Ti, 4M on the V100) and by API on one card. FlashDecodeKV holds a
chunk of scores in shared memory and a workgroup takes one; the staged path
is four launches with a plane between them. A small part fills with either
and the staged path's higher read rate wins at depth; a large one needs the
staged path's thread count, which the square-root split did not give it. A
tier that can time both on its own pools at the split thresholds should, and
ChooseDecodePlan is the prior for one that does not.

**FlashAttention's contiguous V loads, grouped.** The known 3.4x: the paged
form issued a group of keys' V loads (64 values a lane) before their FMAs and
the contiguous one interleaved them. Grouped, the contiguous kernel reads
1.34x (Llama) and 1.47-1.67x (Qwen) on PTX and nothing on Vulkan, whose
compiler already schedules them. It does not close the gap: paged
FlashAttention is still 2.0-2.4x the contiguous one on Qwen's shape and 0.77x
on Llama's, and an aligned contiguous K stride (4128 in place of 4097) moved
nothing (2.29x), so the rest is not alignment. The kernel is at 2-19% of the
wall at the tier's split of 1 either way; it is the opt-in decode, and the
two that ship are far ahead of it. Gated at head widths 7-256 (96 leaves a
ragged group of 21 keys) with f16 V, windows and sinks; weighting every key of
a group by its first key's probability fails at NMSE 1.40-1.76. The lowertest
goldens flash/split1 and split8 moved with it, after that gate passed on CUDA,
Vulkan and the Iris Xe.

**Tried and refused.** Four independent chains in FlashAttention's score dot,
which a warp per head seemed to want: 0.89-0.96 everywhere but the contiguous
256-wide head (1.05), the extra live registers costing more than the chain
saved. One query head a FlashDecodeKV workgroup (`Group` 1), for a bigger
chunk: 0.72-0.76 at depth except Qwen's shape at 4096 keys (1.12), where four
warps buys the same chunk (1.24) without losing the shared K load. Hoisting the first dependent load (descriptor, table, K) was priced
by baking the descriptor outright, the ceiling of any hoist: at most 1.9% on
PTX and 2.5% on Vulkan at 512 keys, and baking values made NVIDIA's Vulkan
driver 0.64x and 0.84x on Qwen's shape -- the KImm finding again, a constant
that one driver folds and another spills. Not built.

**Not built, and why.** GQA packing is FlashDecodeKV's group and the staged
path's `stagedGroup` already; the one kernel without it, FlashAttention, is a
warp per query head, and giving it a KV head's group is FlashDecodeKV. exp2
with log2e folded: PTX's exp is already `ex2.approx` after one multiply, a
multiply a key in a kernel bound by bytes, and the IR has no exp2 to fold it
into. Skipping fully masked tiles: the decode kernels start at keyStart
rounded down to 32, so no tile is fully masked. Vector K loads: K is
transposed and lanes run over keys, so a vector load would span four keys and
need 128-key tiles (P of 128 or more, a score plane four times wider); vector
V loads bought at most 1.9%, and the K side has the same shape of arithmetic.
A single-page table walk: a tile's page id is one load a tile, loaded a tile
ahead already. vLLM's paged attention v2 and FlashDecoding's split-K are the
splits and the merge these kernels already have; FlashDecoding++'s unified
maximum removes a rescale the two-pass FlashDecodeKV does not do.

**Vector V loads (`FlashShape.VecV`).** A lane loads its V elements as 2- or
4-wide vectors where an f32 row divides across the warp. PTX and MSL lower
vector loads; SPIR-V does not, so the tier sets it per API. 1.0-1.9% on PTX,
nothing on Metal (0.999, 1.001), so PTX only. Crediting a component to the
mirrored dimension fails at NMSE 1.94.

**Splits.** FlashDecodeKV's split sweep on the RTX 3050 Ti and the V100 put
the tier's rule at or near the best everywhere but two places: its cap
rounded DOWN gave a 20-SM part one split for Llama-3.2-1B at 512 keys (2 to 8
read 1.20-1.22x; 16 read 0.82x), and on a 256-wide head the chunk of 256 keys
forced 16-64 splits (fixed by four warps). Other counts than the rule's lose
everywhere they were tried: 0.60-0.87x at 512 on the V100, 0.70-0.92x at
4096 on both parts, 0.61-0.87x at 16384. The staged path's square root starved the V100 (above) and was right
on the RTX 3050 Ti, where 32 splits at 4096 read 0.89x and 0.94x.

**Through the tier, once.** FlashKVPlan's shape reaches the contiguous
FlashDecodeKV the tier runs today (`kvplan=1`: Llama-3.2-1B 1.24 / 1.00 /
1.00, Qwen3.5-0.8B 1.04u / 1.20-1.35u / 1.63 on the RTX 3050 Ti; 1.03 / 1.01 /
1.00 and 1.12 / 1.47 / 1.52 on the V100), and its four-warp rule is one the
tier can already be told: `JITLLM_GPU_FLASH=2 JITLLM_GPU_FLASH_WARPS=4` (the
warps knob is read only beside `JITLLM_GPU_FLASH`, which is why a first pair
of runs with the warps knob alone read 1.002 and 0.982 -- two more A/A
controls). `jitllm run -depth` on Qwen3.5-0.8B, cuda:0, 128 tokens decoded,
one process a sample, ABBA, A/A first:

    depth 4096    A/A 0.9965 (IQR 0.045)   four warps 1.0274 (0.051), 1.0249 (0.052)
    depth 16384   A/A 0.9940 (IQR 0.076)   four warps 1.068 (0.263), 1.034 (0.270)  REJECTED

At 4096 keys the token is 2.5% faster where the arithmetic predicts 3.5%
(six attention layers of 24, ~1.2 ms of a 6 ms token, 1.24x on that). At
16384 both passes blow the gate, and not symmetrically: the old arm holds
88-91 tok/s in every process while the four-warp arm lands near 80 or near
100, process by process -- a per-process choice in that arm that was not
identified, on a card at 74 C and 937 MHz. Unresolved; RULE 2 says rest the
box rather than add rounds. These kernel changes were measured before the
tier ran on pages, and the decode plan (ChooseDecodePlan) is not yet what the
tier chooses, so nothing else here has a decode rate behind it yet.

### The decode plan through the tier, and the harness bias it exposed

With kernels.ChooseDecodePlan choosing each paged decode's path, splits,
width and merge (jit/gpu/tier/pagedattn.go, pagedPlanFor), the V100 first read
2.7% slower than the old split rule at 512 keys -- and every single piece put
back (splits, merge, vector V) read the same 0.972, as did two tiers holding
the SAME plan. The harness was the variable: its one-tier A/A decoded 640
tokens on the paged arm before the A/B, so the paged arm was always measured
~640 positions deeper. Each pass now opens both arms fresh, in alternating
order, behind a two-tier A/A. V100 CUDA, Llama-3.2-1B Q4_K_M, same pass:

                                512 keys        4096            16384
    two-tier A/A (decode)       1.0013-1.0048   0.9996-1.0015   1.0008-1.0014
    plan / old rule, decode     1.0024 / 0.9987 1.0083 / 1.0024 1.0058 / 1.0068
    paged / contiguous, decode  0.9969 / 0.9968 1.0099 / 1.0057 1.0413 / 1.0422
    paged / contiguous, prefill 1.1039 / 1.1076 1.5841 / 1.5847 1.1545 / 1.1549

So decode on pages is at parity at 512 and ahead from 4096, and the earlier
0.97-0.98 decode rows in this file were the deeper position, not paging.


### The prompt lost 22% on the V100, and none of it was a kernel

The first V100 board on the paged engine read Llama-3.2-1B Q4_K_M pp512 at
15.3k tok/s against 19.6k on the board before paging, and Qwen3-0.6B Q8_0
19.7k against 21.6k -- while llama.cpp's own rows had not moved. The
in-process A/B had said paged prefill was 1.10x the contiguous path at 512
keys; its contiguous arm was the later engine's, so it could not see a loss
both arms shared.

Bisected on the card, each binary converting its own containers
(`jitllm speed -p 512`, one V100-SXM2-16GB): container v27 moved nothing
(19.5k either side), and the step to paged prefill is where it went. Then,
by elimination:

    nvprof kernel totals, 10 prompts     identical to 1% -- flashprefill70paged
                                         4.59 ms against flashprefill70 4.46
    JITLLM_GPU_SKIP=attn, Qwen3-0.6B     attention costs 3.5 ms of a prompt
                                         paged, 3.9 contiguous
    the device call (layersCall)         26.2 ms -- the old engine's whole prompt
    the prompt                           33.9 ms

So 7.7 ms was the model's, outside the device. Timed phase by phase, 6.6 ms
of it was `growDeviceBatch`: a State's prompt rows, 4 MiB allocated fresh for
every State, faulting in page by page -- and a request is a fresh State, so
every request paid it. The rest was a scratch rebuild: closing a session reset
the scratch's context bound, and `jitllm speed` alternates a 513-position
prompt session with a 3-position decode one, so every prompt after the first
rebuilt the batched scratch inside its timed window (17.4k on a process's
first prompt, 13.7-14.0k on every later one). Paged history sizes nothing by
context, so the bound no longer resets; a closed State's prompt buffers go to
the model for the next one (`promptBuf`).

    V100, pp512, three interleaved rounds       old engine    paged, fixed
    Llama-3.2-1B Q4_K_M                          19.5-19.9k    18.96-19.07k  0.97
    Qwen3-0.6B Q8_0                              21.5-21.7k    20.1-20.4k    0.94
    tg128, both                                   within 1%

What was left was ~1.5 ms of fixed cost per prompt outside the layer ops on
Qwen3-0.6B (every op skipped: 6.9 ms against 5.4) -- not attention, which is
faster paged. Timed phase by phase on the V100, a 512-row Qwen3-0.6B call
spent 3.4 ms on the host before its first launch:

    pagedAppendRows     1.07 ms   one seqEnd per ROW, then 28 layers x 512
                                  map reads -- for a chunk of ONE sequence
    prepBatch           0.92 ms   ~25 kernel lookups a block, every call, to
                                  catch blocks placed since the last walk
    pagedPrep + stage   1.35 ms   the same per-row page check again

Rows collapse to one entry per sequence (`seqEnds`), and the walk runs only
when a block was placed or released (`layerGen`). Batching the page-table
writes into one session per call (`flushTabs`) did not move the rate; it stays
because it removes a hop a layer.

    V100, pp512, two interleaved rounds    pre-paging   paged, fixed
    Qwen3-0.6B Q8_0                        21.5k        23.1k   1.07
    Qwen3-1.7B Q8_0                        11.7k        12.2k   1.04
    gemma-3-1b Q4_K_M                      15.4-15.7k   16.0k   ~1.03
    Llama-3.2-1B Q4_K_M                    19.5k        20.3k   1.04

The paged engine now prefills faster than the contiguous one it replaced on
all four, measured on a box whose other cores were running gates -- a
direction, with the board's re-take as the row.

### MLA decode lost 14% on pages, and it was one kernel's lane count

The paged V100 board read DeepSeek-V2-Lite and DeepSeek-Coder-V2-Lite decode
at 160 tok/s, 0.93-0.94 of llama.cpp's 171, where the board before paging had
186.5 and 1.09. `JITLLM_CUDA_KERNEL_TIMING=1` against the pre-paging binary on
the same v27 container put it in one place:

    pre-paging   attnscoreswarp   295 us a token over 27 calls   11 us each
    paged        pagedscores     1063 us                          39 us each

The staged scores kernel has a warp-across-the-dimensions form for exactly
this -- its own comment names "a row-major MLA row" -- and `pagedBuild` asked
for one thread a slot. A latent row is 576 floats, so each thread walked 576
multiply-adds alone. MLA now takes the 32-lane form on a device that
guarantees the width; head rows of 64-256 keep a thread a slot.

    V100, tg128, two interleaved rounds     before    after
    DeepSeek-V2-Lite Q4_K_M                 160.0     183.0   1.07x llama.cpp
    DeepSeek-Coder-V2-Lite Q4_K_M           159.8     183.5   1.07x llama.cpp

0.98 of the pre-paging 186.5; prefill unchanged at 4.39k.

### Latent attention in the ragged step, and the prompt paths that had never run

Making a step across sessions run MLA blocks (tier/rows.go, mlabatch.go)
meant running the F32 fixtures' prompts as batched chunks for the first time,
and what had been silently falling back a row at a time came out:

- **A float head refused the whole chunk.** The fold (the projection riding
  the chunk's submission) cannot take a float head, and its refusal sent the
  prompt a row at a time -- every F32 fixture on every backend, so no batched
  prompt gate had ever reached them. The chunk now runs, then the head as its
  own submission, as every other fallback already did.
- **A float bank refused the batch.** MLA's absorbs and a mixture's experts
  run grouped; a float bank now reads the float activation in the scale
  plane's slot, as a dense float matvec does, on sm_70 too (moeFloatHalf).
- **A prompt's first device run, failed after a linear block advanced,
  restarted from the embeddings** and applied that block twice: while
  Metal's staged prefill softmax was still launched 64 wide (refused there;
  fixed on main beside this as the "softwidth" fault), synth-kimilinear's
  batched prompt read residual NMSE 1.8 against the rows on the M4. It now
  refuses (errRecurRetry) when the device reports advanced blocks, as decode
  already did, and the head split carries the chunk's count past the head's
  call.

With them, `model.TestLatentPrefillBatchesOnTheDevice` holds the chunk to
the row-by-row device prompt: 1e-13 on the fixtures on CUDA, Vulkan, Metal and
the V100; DeepSeek-V2-Lite p90 4.3e-4 and Moonlight 4.2e-3 on the V100 (the
sm_70 f16 twins); a chunk rotating Kimi-Linear's NoPE channels reads 9.7e-1.
And `model.TestC6OnEveryDevice`'s prompt arm, now batched, meets the binary16
attention tiles: synth-dbrx 4.7e-7 on CUDA's m16n8k16 and 1.2e-6 on Metal's
FlashPrefillTile, 2.9e-12 with the flash tile off; its prompt bound is 1e-5.

## The step across sessions, architecture by architecture

`model.TestStepAcrossSessionsEveryArchitecture` runs the step gate -- three
sessions, eight teacher-forced steps together through `Step`, against each
session decoded alone -- on one model of every architecture on the box, and
fails naming `State.StepRefusal` where a model would not step. A model that
does not step as rows falls back to one session after another, which is the
same logits at a pass over the weights per session: no logit gate sees it,
which is how gemma2 came to never step at all.

### gemma2 never stepped: the per-row head applied no final softcap

`onRowsDevice` refused any model with `FinalSoftcap`, and so did the tier's
ragged head, because decode's head caps on the device (`headCap`) and the
per-row head did not. On the V100 `TestDecodeDoesNotAllocate` with gemma-2-2b
read "0 row(s) stepped across sessions, want 192". The per-row head now caps
the rows it projects into a buffer of their own (`ragCapK`/`ragCap`, the same
`kernels.Softcap` over `R x vocab`, launched over the wanted rows), and the
per-row argmax and the readback take the capped rows. The batched-State rows
paths (`forwardRowsDevice`, `prefillSeqDevice`, `ForwardBatch`'s device head)
finish with `finishDevLogits` for the same reason, where they had relied on
never seeing a cap.

    gemma-2-2b, step against each session alone
    V100 CUDA 1.5e-13   V100 SPIR-V 4.9e-03   3050 Ti SPIR-V 4.9e-03
    Iris Xe 1.3e-13     M4 Metal 1.6e-13      (3050 Ti CUDA: card too small)

A gemma2 step of up to four sessions is decode's arithmetic bit for bit on
CUDA (gemma-2-9b reads 2.3e-13 too). The gate that can run anywhere perturbs
the cap on Llama-3.2-1B, as `TestFinalSoftcapReachesTheDeviceHead` does
(`TestStepAcrossSessionsAppliesTheFinalSoftcap`, a cap of an eighth of the
largest logit): the jitllmfault "nocap" reads 1.46-1.51 on all four devices,
17-51x its bound.

### Speculation over a final softcap

Speculation refused a model with a final softcap, twice (`Speculate` and
`stepRows`), on the reasoning that no model with a prediction block carries
one and building it wanted the cap perturbed on an MTP fixture. Every head a
speculation step's rows reach already capped them -- the device's per-row and
folded heads as decode's does, the host's in `finishRows` -- so the refusals
went and the gate is the perturbation: `model.TestSpecOverAFinalSoftcap`,
synth-qwen35-hybrid-mtp and Qwen3.5-0.8B-MTP-Q8_0 with a cap set from the
largest logit on the first prompt.

A greedy gate cannot see a missing cap: tanh is monotone, so no argmax moves.
Greedy speculation is held to plain greedy decode (64 tokens, three drafts a
round, on the host exactly and on a device up to `devFlip`'s tie) at a cap of
half the largest logit, and the verification's own logits to decode's at an
eighth: four-row steps (`stepRows`, what `Speculator.Next` runs)
teacher-forced along decode's tokens, against decode at the same positions,
the device's bound 8x the same comparison with no cap.

    verify rows vs decode       host  3050 Ti CUDA  SPIR-V   Iris Xe  M4 Metal
    synth-qwen35-hybrid-mtp     0     6.2e-13       5.5e-13  5.1e-13  4.7e-13   (bound 1e-9)
    Qwen3.5-0.8B-MTP-Q8_0       0     1.3e-03       1.2e-03  1.5e-03  2.2e-03   (bound 9.6e-03..1.3e-02)

The cap skipped in the verify rows reads 2.1 and 3.2 on the host
(`rowsOut.noCap`) and 2.1-2.7 on every device (the jitllmfault "nocap",
`TestSpecSoftcapGateDiscriminates`), 248x its bound and up.

The two caps differ because each fails the other check. At an eighth the top
logits saturate to one binary32 value and greedy breaks the exact tie by
index: on the Iris Xe and the M4 the speculative and plain chains parted at
token 1 on a gap of 0.0000, which ended the greedy comparison at one token.
At half, Qwen3.5-0.8B's uncapped verify rows read 3.1e-02-3.3e-02 against a
device bound of ~1e-02: 3x, which does not separate a missing cap from the
band.

### gemma3's local layers, on every rows path, rotated by a stale table

`nn.RowsDevice.LayersRows` and `nn.SessionStepper.LayersSessions` carried one
rotary table, so a model with two bases (gemma3, `SWAPeriod`) handed the
device no local one. The tier's row count for the rotary is the minimum over
both tables, 0, so the step took the upload path with nothing to upload, and
every local layer read whatever `bs.csSWA` last held: the prompt's positions.
gemma-3-1b read 2.953e-01 against each session alone on the 3050 Ti, every
row fluent -- the step across sessions, a batched State on the device, a
batched State's prompt and a speculation step alike. Both interfaces carry
`csSWA` now, `State.ropeRow` fills both tables, and the tier refuses a local
table shorter than the global one by name
(`TestStepRowsRefuseAShortLocalRopeTable`). After:

    gemma-3-1b   3050 Ti CUDA 2.7e-14   SPIR-V 6.9e-04   Iris Xe 5.1e-14
                 V100 CUDA 3.9e-08      SPIR-V 4.7e-04   M4 1.4e-08

The violation hands the step the global table as the local one ("swatab"),
which reaches the kernels only where the device uploads the host's tables
(`RopeTableHost`): 5.4e-02-8.3e-02, 6.4-27x its bound.

### Llama 4 and Cohere2 were refused for a gap that was not there

The tier refused a ragged step over any NoPE-global, weightless-QK-norm or
input-weighted-expert block ("a Llama 4 block, which the ragged path does not
run") while the model reported those States steppable, so `StepRuns` returned
an error on synth-llama4 and synth-cohere2. The batched block already runs
every one of them: an unrotated layer copies k (`copyK`), the weightless norm
rides `l4Ones`, the temperature reads each row's own causal count
(`ScaleRowsByCount` reads `pN[1+r]`, which a ragged step stages as the row's
`pos+1`), the chunk is the paged descriptors', and the grouped mixture has
`actMulW`. The refusal is gone: synth-llama4 3.9e-13 CUDA, 4.9e-13 SPIR-V,
4.1e-13 Metal; synth-cohere2 3.7e-13. Each feature taken out of the step alone
fails the fixture bound (1e-9): unrotated layers rotated ("nope") 1.1e-01 and
5.6e-02, the temperature skipped ("notemp") 6.4e-04, the chunk ignored
("nowin") 2.39 and 1.37, the expert weight on the output ("wout") 1.3e-01.

### The sweep

Worst logit NMSE of the step against each session alone; "card" is a card too
small to hold the model and both arms' histories, "declined" a block the
device does not implement, "--" not run there.

| model | arch | V100 CUDA | V100 SPIR-V | 3050 Ti CUDA | 3050 Ti SPIR-V | Iris Xe | M4 Metal |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | llama | 7.0e-03 | 1.9e-03 | 4.2e-03 | 1.9e-03 | 4.1e-03 | 3.9e-03 |
| Llama-3.2-3B / 3.1-8B / Mistral-7B | llama | 1.3e-03 / 1.8e-03 / 3.9e-04 | 2.2e-03 / 2.1e-03 / 4.1e-04 | -- | -- | -- | -- |
| SmolLM2-1.7B | llama | 1.6e-02 | 7.5e-03 | -- | -- | -- | -- |
| Qwen2-1.5B / qwen2.5-1.5b | qwen2 | 5.9e-03 | 5.0e-03 | 7.8e-03 | 7.9e-03 | 8.6e-03 | 7.4e-03 |
| Qwen2-VL-2B (text) | qwen2vl | 5.9e-03 | 6.6e-03 | 4.3e-03 | 6.6e-03 | 7.3e-03 | 4.3e-03 |
| Qwen3-0.6B / Qwen3-4B | qwen3 | 7.9e-04 / 2.3e-02 | 1.3e-03 / 4.7e-02 | 8.3e-04 | 1.3e-03 | 8.3e-04 | 1.9e-03 |
| Qwen3-MoE-4x0.6B | qwen3moe | 8.3e-03 | 1.0e-02 | 6.0e-03 | 1.0e-02 | 4.9e-03 | 1.5e-02 |
| olmoe-1b-7b | olmoe | 9.6e-03 | 1.2e-03 | -- | -- | -- | -- |
| gemma-2b | gemma | -- | -- | 2.3e-04 | 2.7e-04 | 3.9e-04 | 2.1e-04 |
| gemma-2-2b / 2-9b | gemma2 | 1.5e-13 / 2.3e-13 | 4.9e-03 / 2.3e-03 | card | 4.9e-03 | 1.3e-13 | 1.6e-13 |
| gemma-3-1b / 3-4b | gemma3 | 3.9e-08 / 2.1e-13 | 4.7e-04 / 1.9e-03 | 2.7e-14 | 6.9e-04 | 5.1e-14 | 1.4e-08 |
| Phi-3.5-mini / Phi-4-mini | phi3 | 3.1e-03 / 5.0e-04 | 5.4e-03 / 7.2e-04 | card | card | card | -- |
| Qwen3.5-0.8B | qwen3next (hybrid) | 6.3e-04 | 6.5e-04 | 8.2e-04 | 6.5e-04 | 8.4e-04 | 5.2e-04 |
| gpt-oss-20b / synth-gptoss | gptoss | 6.9e-04 | 2.2e-04 | 5.5e-14 | 7.8e-14 | 8.2e-14 | 8.6e-04 |
| DeepSeek-V2-Lite / Moonlight | deepseek2 | 5.7e-03 / 9.0e-04 | 1.2e-03 / 1.2e-03 | -- | -- | -- | -- |
| synth-deepseek / -dense | deepseek2 | 6.4e-14 / 3.0e-14 | 5.7e-14 / 3.6e-14 | 6.4e-14 | 5.7e-14 | 4.5e-14 | 3.4e-14 |
| synth-kimilinear | kimilinear | 4.6e-14 | 5.5e-14 | 4.6e-14 | 5.5e-14 | 5.0e-14 | 8.2e-14 |
| synth-llama4 | llama4 | -- | -- | 3.9e-13 | 4.9e-13 | 5.5e-13 | 4.1e-13 |
| synth-phi2, -starcoder, -starcoder2 | C6 | -- | -- | 0 | 0 | 0 | 0 |
| synth-commandr / -cohere2 | command-r | -- | -- | 2.2e-13 / 3.7e-13 | 1.7e-13 / 3.1e-13 | 2.3e-13 / 2.9e-13 | 1.9e-13 / 5.2e-13 |
| synth-stablelm / -par | stablelm | -- | -- | 4.6e-13 / 1.5e-13 | 5.5e-13 / 1.5e-13 | 4.4e-13 / 1.5e-13 | 9.9e-13 / 2.1e-13 |
| synth-falcon / -nemotron / -dbrx | | -- | -- | 6.5e-14 / 4.3e-13 / 6.8e-13 | 4.5e-14 / 2.6e-13 / 5.2e-13 | 4.2e-14 / 2.6e-13 / 8.0e-13 | 6.4e-14 / 2.6e-13 / 2.9e-12 |
| granite-3.3-2b / granite-3.1-1b-a400m | granite | -- | -- | 8.4e-14 / 1.7e-02 | 3.4e-03 / 1.4e-02 | card / 1.2e-02 | 6.1e-14 / 1.0e-02 |

Granite ran on no device until its residual scale was built (the next
section): the tier declined it by name, and this row read "declined".

### Granite on the device: one scaled residual add

Granite is llama's block with four scales. Three needed nothing: the
embedding scale rides the device's tied-row gather (`nn.Head.EmbdScale`), the
attention scale every paged kernel already read off the plan
(`nn.LayerPlan.Scale`), and the logit divisor is applied on the host after the
device's head (`finishDevLogits`). The fourth, residual_scale, was declined:
the device adds a block's output to the residual inside the projection's
epilogue (the residual in the matvec's bias slot, decode's and the ragged
step's), or inside a fused add-and-norm launch, and neither has a scale.

A scaled block (`residScaled`) takes neither fusion now. Its attention and FFN
outputs land in their buffer as a post-normed block's do, and
`kernels.AddScaled` -- x + s*y, s read from a one-element buffer as
`kernels.Scale` reads its alpha -- is the residual add, on decode, a prompt
chunk and a step across sessions alike (`Stats.ResidScaled` counts it). A
mixture's routed weights carry none of it on the device: the scaled add covers
the routed sum and the shared expert at once, where the host folds the scale
into the routed weights (`Config.routedScale`). Two scratches of different
scales on one tier are refused by name (`planConflict`), as two score scales
now are.

The arithmetic is exact. On f32 fixtures with Granite's own scales set
(`model.TestGraniteScalesOnEveryDevice`: synth-qwen2 and synth-mixtral, every
block and the head placed) the device reads 2.6e-14 to 1.7e-11 against the
host on decode and a prompt chunk, and its step across sessions 3.0e-14 to
6.9e-14 against each session alone, on the 3050 Ti's CUDA and SPIR-V, the Iris
Xe and the M4. Each scale dropped reads 8.5e-01 and 3.6e-01 (residual),
8.1e-03 and 1.4e-03 (attention at 1/sqrt(head_dim)), 49 (no logit divisor).

On the real models (`model.TestGraniteOnEveryDevice`, a chunked prompt then
decode, and decode alone, against the host's own greedy ids):

    worst logit NMSE vs host   3050 Ti CUDA   3050 Ti SPIR-V   Iris Xe   M4 Metal
    granite-3.3-2b             2.51e-02       1.36e-02         2.81e-02  1.36e-02
    granite-3.1-1b-a400m       3.56e-02       1.21e-02         1.19e-02  1.99e-02

against bands of 8x the device's own order-only control (the row-at-a-time
arm at two matvec splits) for the dense model and the mixture step gate's
5e-2 for the mixture. The dense model sits at 2.9x (CUDA), 3.2x (SPIR-V),
4.3x (Iris Xe, which holds it now) and 1.0x (Metal) its control, and the next section is what that ratio is made of:
it was 2.8x, 4.5x and 2.7x before the device's activation quantizer and Metal's
scaled residual add were made the host's bit for bit, and what is left is the
control. Every violation clears its band 10x or more (a mixture's step break:
its band and 10x the clean step, below): residual dropped
2.2-3.0, attention at 1/sqrt(head_dim) 0.87-2.9, embedding scale dropped
2.0-2.5, logit divisor skipped 28-54, and a step whose rows fuse the residual
unscaled (`tier.ScaleFaultStepResidual`, the step's own break, reachable only
on quantized weights) 1.26-6.5.

The attention scale needed no fix and got a correction anyway: the contiguous
score tiles baked 1/sqrt(HeadDim) and not the plan's scale. They are built
only for a vision block since history went paged, where the two agree, so it
was not reachable; every kernel now asks `devTier.scoreScale`.

And the host had an allocation granite alone reached: `nn.(*JIT).packedWindow`
allocated its window tensor and result rows every call, two objects a token on
a model whose head's row count the tail kernel's group does not divide --
granite's 49159-row vocabulary. Both live on the JIT now, and
`TestDecodeDoesNotAllocate` reads 0 for both Granites on the host, CUDA,
SPIR-V and Metal.

### What parted granite-3.3-2b from the host

Two defects and a one-pair control. The defects are the device computing in
the last bit what the host computes otherwise; the control is a single draw
from a spread wider than the excess it was read against.

The probe that found the first is RULE 11c's seam, read one block deep: the
residual after block 0, with block 0 on the device and the rest on the host,
row at a time, against the host, on the gate's own prompt (3050 Ti CUDA):

    residual after block 0     host vs device   split 1 vs split 4
    granite-3.3-2b             6.5e-05          1.6e-13
    Llama-3.2-1B               4.5e-09          1.7e-13

Block 0 reads the embedding, which both tiers have bit for bit, so nothing
but block 0's own arithmetic can part them. On granite every position agreed
to 6e-7 relative until token 2288 at position 6, and every position after it
was 1e-2 relative off: that token's key and value differ, and every later row
attends to them. Vulkan agreed with the host there. Quantizing that token's
attention-norm row in Go both ways moved exactly two int8s: the host derives
an activation block's scale as `d = amax/127` and then `inv = 1/d`, two
divisions (`cpu.QuantizeQ8Window`, whose own comment says why: "not one
multiply by 127/amax, which differs in the last bit"), and `kernels.Quantize`
and `RMSNormQuantRows` took `d = amax*(1/127)` and `inv = 127/amax`. An element
a few ulps from n+0.5 goes to the next int8 when inv is one ulp off.

★ AND THE OBVIOUS FIX IS NOT ONE ON VULKAN: SPIR-V'S DIVISION IS NOT THE IEEE
QUOTIENT. `OpFDiv` is allowed 2.5 ulp, and over 262144 quotients the drivers
here return another last bit for this share of them (87381 of each kind):

                          x/127    1/d     127/x
    3050 Ti SPIR-V        4.5%     13.4%   27.9%
    Iris Xe SPIR-V        4.5%     11.6%   27.2%
    3050 Ti PTX, M4 MSL   0        0       0

PTX's `div.rn` is exact by construction and Metal compiles with IEEE math. An
FMA correction cannot rescue SPIR-V either: `ir.OpFma` lowers there as a
multiply and an add. So the quantizer's scales are rounded in integers now
(`kernels.quantScales`): `div127` divides the mantissa shifted by eight by
127 and rounds on the bits it drops (a float over 127 is never exactly
halfway, so no remainder is needed), and `1/d` starts from 127/amax, taken
beside it, one Newton step against d, and `roundQuotient`, which counts the
midpoints between the floats around the estimate that lie below the true
quotient -- `ma*2^s - (2*Mc+1)*md`, a difference of two products near 2^48
whose low 32 bits are the whole of it because it is a few ulps. Gates:
`kernels.TestQuantScalesAreTheHosts` (every mantissa of a binade and a million
amaxes, bit for bit), `TestDivRNIsTheIEEEQuotient` (a million quotients across
the normal range, binade edges included; the backend's own division differs
in 216174 on the 3050 Ti's Vulkan, 212555 on the Iris Xe's and 0 on PTX and
MSL), and `backend.TestQuantizeOnTheRoundingBoundary`, whose input sits within
two ulps of the rounding boundaries: the old arithmetic moves 3235-4897 of its
65536 int8s on every backend, and the host's two divisions taken with the
drivers' own division move 1915-4897 on both Vulkan devices. The two quantize kernels' goldens in `lowertest` moved
and nothing else did.

After it block 0 reads 1.2e-08 against the host and 1.3e-08 between splits.

★ THE SECOND IS METAL'S ALONE, AND IT IS GRANITE'S ALONE. `kernels.AddScaled`
wrote `a + alpha*b` as a multiply and an add. ptxas and both Vulkan drivers
fuse that into one rounding, which is what the host's axpy does
(`VFMADD231PS`, `FMLA`); Metal's IEEE math does not, and its scaled residual
parted from the host's in the last bit of 2.8% of residual-like elements. At
alpha 1 the two forms agree, so no llama could see it. It is `ir.Fma` now, and
`backend.TestAddScaledRoundsOnce` holds it to the exact value rounded once
(math/big, since the arm64 Go compiler fuses an expression on its own): 1878 of
65536 differ on the M4 under the old kernel.

★ AND THE EXCESS AT THE LOGITS IS THE CONTROL. With both fixed, CUDA still
reads 2.9x. On the gate's prompt every pair among the host and the device at
splits 1, 2, 4 and 8, worst logit NMSE over the 17 compared steps:

                        host against the four    the device's six pairs
    3050 Ti CUDA        9.9e-03 .. 2.5e-02       8.4e-03 .. 2.3e-02
    3050 Ti SPIR-V      1.1e-02 .. 1.8e-02       4.3e-03 .. 3.1e-02
    Llama-3.2-1B CUDA   2.2e-03 .. 5.0e-03       1.7e-03 .. 4.5e-03

The host sits inside the device's own spread. The gate's control is one of the
six pairs, splits 1 and 4, and on SPIR-V it is the lowest of them, which is
the whole of its 3.2x; the 4.5x before the fix was the same pair, lowest then
too. Two more prompts read the host's pairs inside the device's or level with
its top (CUDA 3.8e-03..6.9e-03 against 3.1e-03..9.5e-03; SPIR-V
3.3e-03..2.7e-02 against 3.0e-03..2.2e-02). Means
over steps tell it more quietly: the host's pairs read 1.0-1.4x the device's on
this prompt and 0.9-1.0x on the other two.

What the host still does that no device pair does is its own reduction order
outside the matvec, and a single int8 flip from it is the kick: on
granite-4.0-micro block 0, CUDA and Vulkan agree with each other to 1e-13 and
both part from the host by 3e-05 to 8e-05 from position 7 on, where the host's
attention-norm row is 2 ulps from a float64 norm and quantizes one int8
differently from it. Granite's block-0 inputs carry amax 11-25 times their RMS
(Llama-3.2-1B's 7-20), so one quantum is a bigger share of the row. On CUDA a
second such flip comes from FlashDecodeKV's order (granite-3.3-2b position 12;
`StagedDecode` removes it). These are order differences of kinds the split
control does not vary; the seam says they enter at the same size as the
control's (block 0 at 1.2e-08 against 1.3e-08), and the table says where they
end.

Neither reference keeps any of this wider. llama.cpp's granite graph scales
the block output (`ggml_scale`, its own kernel, so rounded) and then adds it
(`ggml_add`): two roundings in f32. vLLM's `granite.py` writes `residual +
hidden_states * residual_multiplier` in the model's dtype, bf16 as served, and
gives Granite none of the fused add-and-norm its llama gets. jitllm's host and
devices round the scaled add once (FMA), which is a last-bit difference from
llama.cpp of the order kind, and the same on every tier now.

Refuted on the way: the residual add on CUDA and Vulkan (already fused, and
bit for bit the host's), the f16 KV cache (the gate opens with
`WithKVF16(false)`), the attention scale (1/64 is a power of two, exact in any
order), and the logit divisor (both arms divide, and dropping it reads 53).

So the dense band stays the dense gate's 8x of its control. 4x would fail on
the draw alone: on SPIR-V the host against split 4 is 4.2x the 1-against-4
pair. The mixture's host arm keeps 5e-2. A mixture's step, here and in
`TestStepAcrossSessionsEveryArchitecture` (`stepBound`), takes 8x its own
order control (`orderControl`) where that is wider than 5e-2 -- Qwen3-MoE's
stays at 5e-2 --
granite-3.1-1b-a400m's step on the M4 reads 8.4e-02 at row 9, where an order
change alone reads 1.4e-02 at the same row; over six prompt sets its step
reads 3.1e-03..8.4e-02 with the grouped experts in binary16 and
2.4e-03..6.8e-02 with them int8 (`NoVoltaMoE`), and 1.0e-02..4.9e-02 before
either fix, where Qwen3-MoE-4x0.6B's reads 4.6e-03..7.3e-03 on the same
harness. The tail is the model's router, not the tile path and not this
change. The step's own break (`ScaleFaultStepResidual`, 1.26-1.39 on the
mixture) must clear that band and ten times the clean step rather than ten
bands: the Iris Xe's control reads 1.8e-02, a band of 0.147, against a clean
step of 7.7e-03.

What it costs: the fused norm-and-quantize kernel, which every thread of a
1024-wide group runs the scales in, is 6.83 -> 7.56 us a launch on the 3050
Ti's CUDA and 8.2 -> 8.4 on its Vulkan (2048 wide, medians, base and fixed
alternated six times each), and `Quantize` is unchanged. Two of them a block
is ~0.3% of a Llama-3.2-1B decode token on that card (131 tok/s) by
arithmetic; not resolvable end to end here, the card being shared while it was
measured.

### The dense bound is the model's own order band

The dense step gate held Llama-3.2-1B to `mlaDeviceNMSE`, 1e-2, and the V100
read 6.95e-03. `model.TestStepNMSEBand` measured what that is, 8 sessions x
32 teacher-forced steps against each session alone on the device, beside two
references on the same tokens: the host, and the device decoding alone at a
pinned matvec split -- the same arithmetic summed in another order and nothing
else.

    worst row of 256 (median)   host        split=1     split=4     step x4     step x8
    V100 CUDA  Llama-3.2-1B     1.8e-02     8.6e-03     8.7e-03     7.0e-03     6.8e-03
                                (1.1e-03)   (8.6e-04)   (9.1e-04)   (7.0e-04)   (5.5e-04)
               Llama-3.1-8B     7.4e-03     7.4e-03     4.6e-03     2.8e-03     4.6e-03
               Qwen3-4B         3.4e-01     6.7e-02     3.8e-02     2.6e-02     5.9e-02
                                (3.9e-03)   (1.9e-03)   (2.0e-03)   (2.3e-03)   (2.1e-03)
               gemma-2-2b       1.9e-02     7.1e-03     5.3e-03     2.2e-13     6.6e-03
               Qwen2-VL-2B      1.7e-02     1.4e-02     1.4e-02     1.2e-02     7.2e-03
    V100 SPIR-V Qwen3-4B        5.2e-02     5.2e-02     5.0e-02     4.7e-02     2.9e-02
    3050 Ti CUDA Llama-3.2-1B   1.3e-02     7.1e-03     1.0e-02     1.4e-02     7.1e-03

The step sits inside its model's order band everywhere, and the band is the
model's: Qwen3-4B's worst is eight times Llama's on the same card.
It does not grow with the step (steps 0-7, 8-15 and 16-31 read alike), with
the rows (two, three, four and eight sessions alike, the last on the tiled
twins), or with the depth (Llama 1B, 3B and 8B: 7.0e-03, 1.3e-03, 1.8e-03).
Decode's fusions apart (`NoRagFuse`) change nothing. gemma's step of up to
four sessions is decode bit for bit on CUDA; Llama's is not, so what differs
there is the ragged attention and decode's flash-decode kernel summing a row
in another order, which the split control prices.

So no constant fits: Qwen3-4B's step reads 2.3e-02 on CUDA and 4.7e-02 on
SPIR-V against a Llama step of 2e-03-7e-03, and a bound wide enough for the
first would pass a violation on the second. The dense gate measures its own
order-only control instead (`orderControl`: the three sessions alone at split
1 against split 4, on the same device in the same run) and holds the step to
`denseStepBand` = 8 times it, as `pagedBand` does for the paged history.
Over 37 (model, device) pairs the ratio step/control reads a geometric mean of
0.90, a worst of 2.44 (Qwen3-0.6B on the M4, Qwen2-VL on the Iris Xe) and a
log10 spread of 0.24; 4 would fail one 40-pair sweep in seven on noise, 8 one
in five hundred. The violations against it:

    "session" (rows read the current session's pages)  1.03-1.13   15-46x
    "nocap" (the per-row head uncapped)                 1.44-1.51   17-51x
    "swatab" (gemma3's local layers on the global table) 5.4e-02-8.3e-02  6.4-27x

Mixtures keep `mixtureStepNMSE` (5e-2): their grouped experts run int8 or
binary16 where decode runs one row, and a router near a tie widens the band
past an order change. Synthetic fixtures are held to `fixtureStepNMSE`
(1e-9), on which every clean step reads 0 to 2.9e-12 -- except a mixture
whose experts ran in binary16 (synth-gptoss's MXFP4 banks on Metal's tiles,
8.6e-04), which keeps the mixture bound.

## A launch past the device's workgroup limit

One `vkCmdDispatch` may ask for at most `maxComputeWorkGroupCount[0]`
workgroups along x, and every launch in this tree is one-dimensional. The
limits, read from `VkPhysicalDeviceLimits` and checked against vulkaninfo by
`vulkan.TestWorkGroupCountIsReadFromTheDriver`:

    vulkan:0  RTX 3050 Ti Laptop   2147483647
    vulkan:1  Iris Xe (ADL GT2)         65535   (the specification's floor)
    vulkan:2  llvmpipe                  65535
    cuda:0    gridDim.x            2147483647   (CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
    Metal     no per-grid limit reported; 2^20 threadgroups ran whole on the M4

A Restride of a transposed K over 1024 rows at 8400 positions is 67200
workgroups of 128, past the Iris Xe's limit. The Vulkan backend handed that
count to `vkCmdDispatch` as it was.

**The Iris Xe ran it correctly anyway.** Unsplit, its Mesa driver wrote every
word right at 70000 workgroups of 256 and 1024 threads and at 2^23 workgroups of
one thread, so on the only device here with the small limit a buffer check
passes the old behaviour. It is still invalid usage, and the Khronos validation
layer says so for every such dispatch:

    VUID-vkCmdDispatch-groupCountX-00386: groupCountX (70000) exceeds device
    limit maxComputeWorkGroupCount[0] (65535)

What a driver does with it is its own business, so a launch is now split
(`vulkan.Ctx.dispatch`): pieces of at most the limit, each pushing the index of
its first workgroup as a 4-byte push constant, which `spirv.Emit` adds to
`WorkgroupId.x` to make the IR's CTAID. The pieces cover the launch exactly, so
no kernel sees a CTAID an unsplit launch would not give it and no surplus
workgroup exists to clamp -- which matters, because several kernels use
`CTAID()` with no clamp of their own (flash's merge takes it as the item
outright, the Volta GEMM divides a segment index out of it), so a 2-D grid with
a ragged last row would have written past their outputs. Every launch pushes the base, 0 when it fits, so every kernel runs the
one path; the cost is that every SPIR-V module carries the block and an add,
which moved every `lowertest` golden.

The gates: `backend.TestALaunchPastTheGroupLimitRunsWhole` (every CTAID and a
shared-memory neighbour across a barrier, both launch paths, every device, and
every Vulkan device again at a forced limit of 97), `TestRealKernelsPastTheGroupLimit`
(Restride past 65535 by itself; matvec with a Reduce, an in-group split and a
row tile at a forced limit of 7) and `TestSplitLaunchesAreValidVulkanUsage`
(both, under the validation layer). Against the old single dispatch the data
gates stay green on the Iris Xe -- the driver's tolerance above -- and fail on
the dispatch count, and the validation gate fails on the VUID. Against a
lowering that drops the base, every split launch writes the wrong workgroups.

The validation layer also reported the batch descriptor pool outliving its
device at every `Ctx.Close` (`VUID-vkDestroyDevice-device-05137`); `Close`
destroys it now.

CUDA and Metal refuse a launch past their limits by name instead of splitting:
their limits sit at or past the reach of the IR's 32-bit flat index, and a
split there would put a base on every PTX and MSL kernel for a launch nothing
can make.

## A hybrid's ragged step on a device with no 32-lane subgroup: the scan's one-thread form

A linear block's ragged step -- a batch's, a step across sessions, a
speculation's verification, a prompt chunk riding beside decoding rows -- runs
the delta rule as `kernels.GatedDeltaFused` with `Runs`, one launch for every
run. `prepRagLinear` refused it on any device that does not guarantee a 32-lane
subgroup ("a linear block's batched step needs a guaranteed 32-lane
subgroup"), and `nn.RowsLinear` / `State.rowsLinear` sent such a device's
hybrid one session at a time or down the prompt chunk. On this laptop that was
llvmpipe (subgroups of 8, no size control); a Vulkan driver that will not pin
the width is the same case. RULE 8 makes that a kernel to write, not a
decline.

The kernel was already written. `gatedDeltaScan` puts `ScanLanes()` lanes on a
state row and reduces both dots with shuffles across them; at `Lanes: 1` the
lane count is one, both butterflies are empty, the row lives in one thread's
registers and the dots sum serially -- no shuffle, no shared memory, so it
lowers on every backend. The chunk path has run that form on such devices all
along (`deltaScan` at `qkLanes == 1`); only the run form had never been built
at one lane. `prepRagLinear` now builds it at the device's lanes, and the
refusal, `nn.RowsLinear`, `GPU.RowsLinear` and `State.rowsLinear` are deleted.
The lane-group form stays wherever the subgroup is guaranteed.

`tier.Config.ScalarLinearRows` forces the one-thread form on a device that has
the subgroup, so the form is gated on real hardware, and
`Stats.LinearRowsScalar` counts the linear blocks a ragged step emitted on it:
the model gates' `checkLinearRowsForm` demands it non-zero where the device
guarantees no 32 lanes or the form was forced (`JITLLM_SCALAR_LINEAR_ROWS=1`
in the model tests), and zero elsewhere.

Kernel gate, `backend.TestRunsStepEachSequenceAlone`, now with one-lane cases
at key widths 7, 8, 12, 32, 48 and 128 and value widths 5, 9, 13, 24, 32 and
128, per-head and per-channel decay, tiled keys: each run form against each
sequence alone through the chunk kernels (bit for bit on ptx), and the
one-thread form against the lane groups on the same step:

    one thread against 4..32 lanes a row     output NMSE 9.4e-15 .. 6.5e-14
                                             state  NMSE 1.3e-15 .. 4.7e-15
    bound                                    1e-10
    violation (rows not chaining)            output 5.9e-02 .. 2.6e-01 against
                                             each alone and against the lanes

Green on ptx and spirv on the RTX 3050 Ti and on llvmpipe (one-lane cases
only there).

Model gates, Qwen3.5-0.8B Q4_K_M and the synthetic hybrids, every block placed:

    llvmpipe (vulkan:2), native one-thread form
      TestStepAcrossSessionsHybridMatchesEachAlone   1.07e-03 (both forms)
      TestStepRunsAdmitsAPromptBesideDecoding/hybrid 7.73e-04, mirrored 0.52
      TestSchedulerMixesPromptsIntoDecodeSteps/hybrid tokens equal, mirrored
                                                     chunks part at token 0
      TestStepAcrossSessionsHybridGateDiscriminates  slot 1.31 / 0.94,
                                                     parity 0.51 / 0.54
      TestStepAcrossSessionsLatentMatchesEachAlone   synth-kimilinear 1.3e-12
      TestDecodeDoesNotAllocate (step arms)          0 engine allocations
    cuda:0, vulkan:0 (RTX 3050 Ti), vulkan:1 (Iris Xe): native (lane groups,
    LinearRowsScalar 0) and forced (ScalarLinearRows, LinearRowsScalar > 0)
      TestStepAcrossSessionsHybridMatchesEachAlone   cuda 7.00e-04 forced
                                                     against 7.67e-04 native
      and every gate above, the spec gate and the fault gates: green, 0
      engine allocations a warm step
    metal (M4, cross-compiled test binary), native and forced: the step, scheduler,
    prompt-beside-decoding, latent/KDA, spec and allocation gates green; the
    kernel gate's one thread against 32 lanes 2.1e-14 .. 6.5e-14 (output)

Before, the same scheduler gate failed on llvmpipe with the refusal and the
step gates skipped it by name.

**What it found: `kernels.MatVecSegments` faulted on llvmpipe -- a Mesa bug,
and the module now steps around it.** Plain decode on llvmpipe segfaulted
inside the driver's compute thread once the split tuner picked a q/k/v split
that `fuseQKV` fuses -- split 1, or in-group -- on any model, dense too
(Llama-3.2-1B: `JITLLM_GPU_TUNE=0 JITLLM_GPU_SPLIT=2 JITLLM_GPU_FORCE_GROUP=1
jitllm run -devices vulkan:2`; the same on vulkan:0, vulkan:1 and cuda:0 gave
the host's ids). The kernel gate reproduced it without a model:
`JITLLM_VK_DEVICE=2 go test ./backend -run
TestMatVecSegmentsIsTheSeparateLaunches` faulted on its first spirv case.
spirv-val passed the module, the validation layer reported nothing, and the
fault was a load through a null base.

It is the driver, and the other backends were never reading anything wrong.
Bisected on the raw module (run through `vulkan.Ctx.Compile` with dummy
buffers, two Q8_0 segments of 64 rows):

    every workgroup runs its segment once (count forced to 1)    runs
    one workgroup, inside segment 0's run                        runs
    one workgroup, outside segment 1's run (count 0)             FAULTS
    segment 1 alone, its count a push constant that is 0         FAULTS
    the same, one buffer read and stored at the entry            runs
    the same, that access after the loops instead                FAULTS
    the same, an unused (or Volatile) load at the entry          FAULTS

So a workgroup faults when it runs a loop ZERO times and that loop holds the
module's first buffer access (with a loop nested in it: a GLSL version of the
shape faults with eight loads in the inner body and runs with two or four, so
what Mesa's optimiser makes of the nest decides it too). gdb names the mechanism: `bsf` on the execution mask, `cmovb`
to lane 0 when the mask is empty, then a load at `0x180`/`0x188` -- binding 6's
descriptor, base and size -- through lane 0 of a per-lane copy of the
descriptor set's base address, which is zero. Mesa's own comment on
`first_active_invocation` says gallivm runs code under an empty mask ("portions
of a loop after a break ... has ended the last invocation") and returns lane 0
then; upstream fixed the dereference in 63b89856, "gallivm: don't deref a NULL
buffer descriptor with an empty exec mask" (*"When no invocation is active,
first_active_invocation() selects lane 0, which can have a NULL SSBO descriptor
address"*), after this box's Mesa 25.2.8. MatVecSegments is the shape that
reaches it: every access of every segment sits in that segment's run loop,
which the other segments' workgroups run zero times.

**The fix is in the SPIR-V lowerer** (`spirv.anchorFor`): a module whose first
buffer access sits inside a loop with a runtime count reads one descriptor at
its entry, ahead of every loop, with `OpArrayLength` -- a descriptor read, not a
memory one -- and keeps it alive by passing that loop's count through a select
on a length no buffer can have (2^32-1 elements of 4 bytes). The base address
then has a definition that dominates every use. Of the lowertest corpus's 91
kernels only `matvecseg` qualifies, and it is the one golden that moved; PTX
and MSL are untouched. `spirv.TestDescriptorAnchor` pins the shape (and fails
against an emitter that skips it), and against that same violation the
llvmpipe kernel gate and the `jitllm run` reproducer fault again.

    llvmpipe, JITLLM_GPU_TUNE=0 JITLLM_GPU_SPLIT=2 JITLLM_GPU_FORCE_GROUP=1,
    Llama-3.2-1B Q4_K_M, 16 blocks placed, 24 tokens:
      ids identical to -devices cpu
      [128000 791 6864 315 9822 374 12366 13 578 469 3168 301 22703 374 ...]
    TestDecodeDoesNotAllocate, JITLLM_STEP_DEVICES=vulkan:2, device tuner ON
      (the llvmpipe arm above was taken with it off because this faulted):
      0 engine allocations on every arm that ran -- Llama-3.2-1B decode with
      10 of 16 blocks placed (its step arms skip: llvmpipe's budget holds 10),
      olmoe decode, Qwen3.5-0.8B decode and all four step arms

## Every Vulkan device vanished partway through a V100 test process

On the V100 box, one `engine/model` process ran `vulkan:5` arms green until
`TestSchedulerOnDeviceLatent` or `TestStepRunsAdmitsALatentPrompt` had passed,
and then every later Vulkan arm skipped: *"vulkan: device 5 is out of range,
this host has 1: [0] llvmpipe"*. Nine skips in one run, main's binary too, and
every one of them read as green.

**Nothing leaked a handle and no device was lost.** File descriptors, threads
and mappings stayed flat across the run, and every `VkInstance` and `VkDevice`
was destroyed. The loader said what happened, once asked (`VK_LOADER_DEBUG=error`):

    /lib/x86_64-linux-gnu/libnvidia-tls.so.580.173.02: cannot allocate memory
    in static TLS block
    loader_icd_scan: Failed loading library associated with ICD JSON
    libGLX_nvidia.so.0. Ignoring this JSON

Every `OpenDevice` and every `List` created an instance and destroyed it, and
the loader dlopens each ICD at `vkCreateInstance` and dlcloses it with the last
instance. `libGLX_nvidia`, `libnvidia-glcore` and `libnvidia-tls` are all
`DF_STATIC_TLS`, and glibc serves a dlopened module's initial-exec TLS from a
fixed surplus that it did not get back here. So each instance spent some of it
until a load found none, and the loader enumerated the remaining ICDs
without an error. The surplus is the budget, which a tunable shows:

    TestReopeningKeepsEveryDevice, unfixed, V100 (glibc 2.39, driver 580.173.02)
    default surplus                                FAIL at cycle 10, 9 -> 1 devices
    GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048   FAIL at cycle 14

The laptop (driver 550.163.01, three devices) reloads the ICD just as often --
`LD_DEBUG=files` generates libnvidia-tls's link map 11 times in 5 cycles on
both hosts -- and kept all three devices over 1000 cycles, so whether a box
runs out depends on what else lands in the surplus. A gate that only ran there
would never have seen it.

**The fix is one `VkInstance` for the process** (`vulkan.instance`): created on
first use, never destroyed, shared by every `Ctx` and `List`. The ICD stays
loaded, and an open costs a logical device and no instance: 200 open/close
cycles take 14.3 s on the V100 where 10 took 96.5 s, and 4.0 s on the laptop
against 15.4 s. The loader's environment is now read once, at the first open,
so `TestSoftwareVulkanIsDeclined` narrows it to lavapipe in a child process.

**And the skip is loud now.** `enumerate` remembers the process's first device
count and returns `vulkan.ErrDevicesVanished` for an enumeration that comes
back shorter. `engine/model`'s device arms skip through `noDevice`, which fails
on that error; `engine/model`, `backend` and `jit/gpu/tier` fail the
binary from `TestMain` when any enumeration shrank, for a skip `noDevice` does
not reach. Against the unfixed engine the latent sequence that had skipped
four Vulkan arms failed them ("was there earlier in this process and is gone
... enumerates 1 device(s) where this process first enumerated 9") and
`TestMain` printed FAIL.

The whole `engine/model` package in one process on a V100, `cuda:0` and the
card's own Vulkan index named in JITLLM_STEP_DEVICES, JITLLM_SPEC_DEVICES and
JITLLM_MIX_DEVICES:

    main's binary   0 Vulkan arms passed, 48 skipped "out of range" -- from
                    the first one, TestDecodeDoesNotAllocate's Llama-3.2-1B
                    arm, since the tests before it had spent the surplus
    this fix        78 Vulkan arms passed, 0 skipped, 0 failed, no enumeration
                    shrank

**The validation layer found a second, unrelated defect on the same path.**
Under `VK_LAYER_KHRONOS_validation` the latent gates reported
`VUID-vkDestroyDevice-device-05137` at every device close: twelve kernels per
tier on synth-deepseek, each a shader module, pipeline, two layouts and a
descriptor pool the device still owned. They were the expert banks' indexed
matvecs, which `mkkID` compiled per block and nothing closed --
`ReleaseLayers` leaves kernels to the device caches, and no cache held these.
They come from `devTier.idKernel` now, one per shape for the device, closed
with it. `vulkan.Ctx` keeps a ledger of what it compiled, closes what is still
open before the device and names it in `vulkan.Orphans`;
`TestClosingADeviceLeavesNoKernelOpen` fails on any orphan (12 and 8 against
the per-block compile) and the same run under the layer reports no VUID.

## One shared Vulkan command buffer broke every hybrid (found on Kimi-Linear)

★★★★★ **AND THE HYBRID PATH ON VULKAN IS FIXED: KIMI-LINEAR RUNS 3 OF 3
BLOCKS ON spirv AT worst logit NMSE 7.073e-12, BESIDE CUDA's 4.993e-12.** The
bug was older than Kimi Delta Attention and broke every hybrid on that
backend, and it was ONE SHARED COMMAND BUFFER.

`Ctx` held a single `cmd`, and three things used it: the recorded `Batch`, the
one-shot `Kernel.Launch`, and `copyBuf`. Both one-shots begin with
`vkResetCommandBuffer` and end with `vkEndCommandBuffer` -- so a copy issued
WHILE a batch was recording reset the buffer the batch was recording into, and
every dispatch encoded after it went to a command buffer that was no longer in
the recording state. **Vulkan drops those silently.** CUDA has no equivalent:
its launches go to a stream.

★ **AND ONLY A HYBRID EVER TRIGGERED IT, WHICH IS WHY IT SURVIVED SO LONG.**
`emitLinear` is the one path in this tree that writes a buffer from INSIDE a
recorded session -- `s.Write(bs.dRows, ...)`, the row count `Conv1dShift`
reads -- and a device-local write stages through `copyBuf`. Every dispatch
after the FIRST linear block was therefore discarded, including the whole head
sequence, which is why the logits came back exactly zero and why keeping the
head on the host "fixed" it.

`Ctx.one` is a second command buffer for the one-shots. The batch keeps `cmd`.

★★★ **AND WHAT FOUND IT WAS THE VALIDATION LAYER, AFTER SIX KERNEL FAMILIES,
GRAPH CAPTURE, PARTIAL ROTARY, THE OUT-GATE, THE k-SPLIT, THE VOCABULARY, TWO
BARRIER HAZARDS AND AN FFI KEEP-ALIVE HAD ALL BEEN ELIMINATED BY MEASUREMENT
AND NONE OF THEM WAS IT.** `VK_LAYER_KHRONOS_validation` is not installed on
this box; RULE 11 says fetch it rather than record the absence, and
`apt-get download` plus `dpkg-deb -x` needs no root. It named the fault on the
first run:

    VUID-vkCmdDispatch-commandBuffer-recording:
    vkCmdDispatch() was called before vkBeginCommandBuffer()

**Three sessions of eliminations against one run of the tool that checks the
API contract.** The lesson is not that the eliminations were wasted -- they
are what made the reading unambiguous -- it is that a backend with a
conformance checker should have had it pointed at this the first time a whole
model came back wrong. It reports ZERO errors now.

★ **THE GATE THAT LET IT HIDE IS FIXED TOO, AND IT IS RULE 11d EXACTLY.** Every
hybrid device gate asked for `"gpu"`, which resolves to CUDA on every machine
this project has, so the gated delta rule had never executed on SPIR-V at all.
`TestHybridAttentionBlockOnDevice` sweeps cuda, vulkan and metal now -- 8 pass,
4 skip for the absent backend.

★ **AND THE THREE HARDENINGS FOUND ON THE WAY STAY, LABELLED AS WHAT THEY
ARE.** The dispatch barrier now covers read -> write and write -> write as
well as write -> read, and `runtime.KeepAlive` guards the descriptor `infos`
in both `Kernel.Bind` and `Batch.Encode`, where a `uintptr` had been the only
reference across the FFI call. All three are real defects and NONE of them was
this one.

★ **AND THREE THINGS MATCHED BEFORE THE BUG WAS FOUND, WHICH IS WHAT MADE THE
NARROWING FAST**: the pre-conv projections (so the q|k|v fuse order is right),
the post-conv q/k/v at both positions (so the convolution and its state are
right), and the decay and beta to every printed digit (so the low-rank forget
gate, the `-exp(A_log)` broadcast and the per-channel axis are right). gpt-oss runs on the host and on every device
backend (`TestSynthGPTOSSOnEveryDevice`);
`docs/engineering-history/model-correctness.md` has how the host was verified
and `gpu-kernels.md` ("MXFP4 and float weights run on the device, and the tier
never told it a block had sinks") the device.

## RULE 11c's cases: the tier's wiring of gated kernels

### ★ RULE 11c: THE KERNELS ARE GATED. THE TIER'S WIRING OF THEM IS NOT.

**A new graph feature is not on the device until an end-to-end gate runs a
model that HAS it, every block placed, against the host.** gpt-oss's sinks and
mixture biases were gated kernel by kernel and never reached a block: the
offered plan did not carry them (`docs/engineering-history/gpu-kernels.md`,
"MXFP4 and float weights run on the device"). The cases below are the same shape.

**Two architectures were producing wrong text on the GPU and every kernel gate
was green**, because a gate on `kernels.MatVec` proves the KERNEL and says
nothing about whether the tier asked for the right one. Found together, both
pre-existing, both silent:

    qwen2  CUDA  [785 6722 315 9625 374 99789 16935 10251 107289 151643 ... 6484 6484]
    phi3   CUDA  [450 7483 310 3444 338 341 264 264 264 264 264 264 264 264 264]

★ **THE BIAS FLAG WAS DROPPED FOR EVERY DENSE MATVEC.** `mkkID` takes a `bias`
argument and, for `experts <= 1` -- which is every weight of every non-MoE model
-- called `mkk`, which built the kernel with no `Bias` at all. `batchMVDot4`
never set it either. **Only the EXPERT path and the MMA twin ever did.** So a
6-buffer kernel was launched with 7 buffers: Vulkan refuses by name ("kernel
wants 6 buffers, got 7") and declines the block, and **CUDA does not check
arity at all** -- it ignores the extra pointer and the bias is simply never
added. And `kernKey` had no bias field, so even once passed, a biased matvec
could collide with an unbiased one of the same shape.

★ **AND A ROW RANGE OF A PACKED TENSOR WAS UPLOADED WHOLE.** `model.rows()`
split phi3's fused `attn_qkv` and its fused gate/up -- and for a PACKED weight it
cannot slice bytes, because the device layout is `[sub-block][word][row]` and
rows are interleaved, so it carried the offset in `Packed.Row`/`Stride`, which
`nn.MatVecPacked` honours. `devTier.resident` uploads `QS`/`D`/`SC` whole and
reads NEITHER field, and `MatVecShape` has nowhere to put them -- so k and v
were the same buffer as q, indexed from row 0.

★★ **AND DECLINING IT WAS THE WRONG SHAPE OF ANSWER, WHICH IS WORTH KEEPING.**
The first fix made the tier decline the block: correct output, phi3 on the host.
That reads as a missing capability, and it was not one -- **phi3's fused qkv is a
PACKING CHOICE OF THE FILE.** GGUF is the converter's input, so it dies at
conversion, exactly as the mmproj's two-file contract does.

★ **AND `attn_qkv` DOES NOT MEAN THE SAME THING IN EVERY ARCHITECTURE, WHICH
NEARLY COST A WRONG CONTAINER.** qwen3next names the LINEAR block's fused q/k/v
`attn_qkv` too, and there the fusion is the GRAPH rather than the file: the delta
rule consumes one mixed projection of 8192 rows on a geometry `deltaGeom` owns,
not `NHead*HeadDim + 2*NKVHead*HeadDim`. Unfusing it by the attention shape would
have written three wrong tensors. It did not, because the row check refused --
*"blk.0.attn_qkv.weight has 8192 rows, and its parts want 5120"* -- and no
container was written. `unfuse` is gated on `Arch == ArchPhi3` now and the row
check stays as the backstop. **The rule is narrower than "a fused weight is a
file artifact": it is true of phi3 and false of qwen3next.** `convert.unfuse`
splits both fused weights while the bytes are still GGUF row-major, where a row
range IS a byte range: three slices and no arithmetic. `FlagFusedQKV` and
`FlagFusedGateUp` are RETIRED -- the bits stay reserved rather than reused,
because a value that changes meaning between versions is what the version number
exists to prevent -- and `model`'s splitting paths are deleted.

**Phi-3.5-mini: 27 blocks placed, token ids identical to the host.** 197 tensors
become 293, the same bytes addressed separately, page size and padding unchanged.

`devTier.declineWeights` stays as RULE 8a insurance for any future weight that
arrives with a row offset, and is unreachable for every shipped model.

**Why nothing saw either.** `backend/matvec_test.go` tests `Bias: withBias` and
passes -- the kernel was always right. `model.TestAttentionBiasesReachTheModel`
is a HOST-path injection test that says so in its own comment ("it needs no
biased model, which is the whole point"). **No Go gate compares a real device
against the host for a text model**; that only lives in `scripts/three-way.sh`,
which runs tinyllama -- a model with no biases and no fused qkv.

**So a device gate must run a model that HAS the feature, end to end, against
the host.** A kernel gate and a host gate can both be green while the tier wires
them together wrongly, and that is the shape to look for, not an exception.

★ **AND CPU-vs-DEVICE TOKEN EQUALITY IS NOT THE BAR; COHERENCE IS.** After the
fix qwen2 on CUDA still parts from the CPU at token 6 and stays coherent, which
is the f32 reduction-order band RULE 12 already records (0.2-0.94 of a logit).
Run-to-run it is the SPLIT TUNER: `tuneSplit` times kernels at load, so the
split -- and the reduction order with it -- varies per process. `JITLLM_GPU_TUNE=0`
makes three runs identical. Pin it before comparing anything.

★★★ **AND THE SIZE OF THAT BAND IS A FUNCTION OF HOW MANY BLOCKS RUN BELOW THE
SEAM, WHICH IS MEASURED AND IS 2550x ON A HYBRID. AN ABSOLUTE LOGIT DIFFERENCE
ACROSS A DEEP SEAM IS NOT EVIDENCE OF A DEVICE BUG.** The SAME block of the same
48-block hybrid, moved only in DEPTH
(`model.TestHybridDeviceErrorIsAmplification`):

    one device block at block 43, 4 host blocks below    logit NMSE 7.769e-06
    one device block at block 3, 44 host blocks below    logit NMSE 1.981e-02

1.21x per block, and it is the compounding `TestTowerOnDeviceMatchesHost`
already records from the other side ("5e-6 after one block is 1e-3 after
twelve") travelling a recurrence. Qwen3-Next-80B with one block at index 3 of 48
therefore reads `max|dlogit|` 1.0-3.5 at EVERY position INCLUDING 0 -- where
attention is a softmax over one key and the block's output is exactly V[0], so
no cache, rope or score kernel can be involved -- and that is not a fault.
**The falsifiable form is to move the seam and see whether the error tracks the
depth below it**, which is one test and a quarter of a second; reading the
absolute number instead cost most of a session. See
`docs/engineering-history/placement.md` 15i.

★ **A DENSE MODEL CAN BE THAT SENSITIVE TOO, AND THE SECOND HALF OF THE TEST IS
AN INDEPENDENT ENGINE.** The post-norm checkpoints (OLMo 2, EXAONE 4) read
1.4-2.0 device-vs-host, and one block alone reads 1.33 at block 0 and 0.075 at
the last -- while llama.cpp's own CUDA build reads 1.16-1.93 from its CPU build
on the same file, and jitllm's host reads 0.85-3.35 from itself with only the V
cache's width changed. `model.TestPostNormDeviceErrorIsAmplification` holds
all three; `docs/engineering-history/model-correctness.md` has the table.

## The decode row tile on the device, spread rows: a regression on both backends

★★ **THE DECODE ROW TILE ON THE DEVICE -- TRIED AND MEASURED, A REGRESSION ON
BOTH BACKENDS, AND THE REASON GENERALISES.** The idea is the user's and it is the
right question: the HOST tier runs sixteen rows in one register file
(PackedFusedGroup) and the device ran ONE row per thread, because
`kernels.MatVec` refused `Rowt > 1` at `NTok <= 1`. That refusal was a guard and
not a limit -- the kernel body is already general in Rowt -- so it was lifted,
gated (`backend.TestMatVecMatchesReference/.../rowtN`, which reads NMSE 4.3e-01
against a tile that strides by 1 instead of Rows/Rowt) and measured. gemma-2b,
decode, `JITLLM_GPU_SPLIT=1` because the tile needs it:

| rows per thread | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| M4 Metal tok/s | **49.41 / 49.44** | 36.96 | 21.05 | 9.36 |
| RTX 3050 Ti tok/s | **59.13** | 57.16 | 36.34 | -- |

Time grows roughly IN PROPORTION to the tile. **A decode matvec is THREAD-STARVED
on a GPU, not latency-starved**, which is the opposite of the host's problem and
why the same lever points the other way: six cores want register reuse, and a GPU
with tens of thousands of lanes wants more threads. Tiling rows divides the grid
by Rowt -- gemma's widest per-layer matvec is 16384 rows, which is 2048 threads
at Rowt=8. `Split` spends threads the other way, on k, and that is the axis that
pays here.

The knob survives as `tier.Config.DecodeRowt` (`JITLLM_GPU_ROWT`), default off,
with `Stats.RowtTiles`/`RowtPlain` beside it -- because the first three runs of
this experiment measured NOTHING and looked exactly like a null result: the tile
needs `Split == 1` and `tuneSplit` chooses the split first, so the knob reached
zero of 126 matvecs. **"It made no difference" and "it never ran" are the same
number without a counter.** RULE 10, caught by the counter rather than by care.

## Closed: the three-way proof's 1-in-30 failure was the unpinned split tuner

- ~~THE THREE-WAY PROOF FAILS ABOUT 1 RUN IN 30, AND IT IS UNATTRIBUTED~~
  **ATTRIBUTED AND CLOSED, WITH A DETERMINISTIC REPRODUCER. IT WAS
  THE GATE, NOT THE ENGINE.**

  `scripts/three-way.sh` asserts TOKEN-ID EQUALITY against a host run and did
  NOT pin the split tuner. `tuneSplit` measures at load, so the split -- and the
  f32 reduction order with it -- varies PER PROCESS. The entry on CPU-vs-device
  equality in RULE 11c already said the words: *"Run-to-run it is the SPLIT
  TUNER... `JITLLM_GPU_TUNE=0` makes three runs identical. Pin it before
  comparing anything."* The script never did, so it was asserting bit-stability
  across a knob that legitimately moves, and it failed whenever the tuner
  happened to land on a split that flipped a knife-edge argmax.

  **The reproducer, Llama-3.2-1B-Instruct Q4_K_M on cuda:0, SHIPPED path, one
  process per split:**

      split  1  2  4      correct
      split  8  16        DEGENERATE REPETITION -- it replays an earlier span
      split 32            correct

  Deterministic at every width, which is what says it is not a race. And
  `jitllm verify` names the mechanism at the flip: **"pos 5: cpu 35490 (margin
  0.00167)"** -- a 0.0017-logit margin, far inside the 0.2-0.94 reduction-order
  band this file already documents. The argmax there is a coin toss and greedy
  decoding amplifies it into the repetition, which is why the SYMPTOM looked
  like a KV-cache corruption and the CAUSE is arithmetic reassociation.

  ★ **THE OLD HYPOTHESIS IS REFUTED, AND BY A MEASUREMENT RATHER THAN AN
  ARGUMENT.** It read *"the next hypothesis is the KV cache at a device
  boundary"*. There is no boundary: the failing configuration places **16 of 16
  blocks** on the Iris Xe, full offload, one device. Two more were refuted the
  same afternoon -- the captured graph (`JITLLM_GPU_GRAPH=0` makes it MORE
  frequent, 6/10 against 3/10) and the subgroup softmax (scalar 2/10 against
  warp 3/10). None of the three was the variable; the tuner was.

  ★ **AND ONLY ONE MODEL ON THE BOX SHOWS IT**, which is why it took so long:
  tinyllama and gemma-2b are identical at splits 1, 8 and 16. It needs a margin
  of 0.0017 to surface, and most prompts on most models do not have one.

  `three-way.sh` exports `JITLLM_GPU_TUNE=0` now. 4/4 clean, and 4/4 with the
  in-group reduction FORCED on (`JITLLM_GPU_FORCE_GROUP=1`), where the unpinned
  script had been failing 2/4.

  ★ **THE STANDING LESSON IS NARROWER THAN "PIN THE TUNER".** An equality gate
  must pin every knob its equality depends on, and a gate that fails 1 run in 30
  is not flaky -- it is measuring something it did not mean to hold still.

## Metal decode against the oracles: where the token goes

- **★ METAL'S MATRIX UNITS ARE AVAILABLE AND UNREACHABLE, AND THE COMMENT THAT
  REFUSES THEM IS HALF RIGHT.** `msl.go` declines `ir.OpMMA` outright: *"MSL's
  simdgroup_matrix is float only -- half and float, no int8 -- so there is no
  integer matrix instruction to lower to at all."* True of `MMAS8`, which is the
  quantized matvec. **False of the case that matters**: `kernels.AttnScoresMMA`
  asks for `ir.MMAF16` -- binary16 x binary16 -> float32 -- and an M4 probe
  reports `simdgroup_float8x8 multiply_accumulate` and `simdgroup_half8x8`
  **AVAILABLE**. So attention on Metal falls through to the FMA tile while the
  hardware's matrix unit sits idle, and the refusal reads as a hardware fact
  rather than the missing lowering it is.

  ★ **AND THE BLOCKER IS A SHAPE MISMATCH, NOT A MISSING SWITCH CASE.** The IR's
  MMA is PTX-shaped -- `MMAShape{M:16,N:8,K:16}` with per-lane FRAGMENTS from
  `sh.Frags()` -- and Metal's `simdgroup_matrix` is an opaque 8x8 tile with
  load/store/multiply_accumulate and no documented lane layout. Mapping
  fragments onto it would depend on an undocumented and unguaranteed internal
  order. The sound route is a TILE-shaped MMA op the IR can express and Metal
  can lower natively, with the attention kernel choosing between the two -- a
  design decision, not a patch.

  **What Metal DOES do, measured on an M4 (16 GB, 4P+6E), decode:**

  | model | weights | host | metal | metal GB/s | |
  |---|---|---|---|---|---|
  | stories15M q4_0 | 19 MiB | 3051 | 797 | -- | **0.26x, LOSES** |
  | stories15M q8_0 | 26 MiB | 2613 | 731 | -- | **0.28x, LOSES** |
  | SmolVLM-256M Q8_0 | 175 MiB | 273 | 261 | -- | **0.95x, LOSES** |
  | tinyllama q3_K_M | 634 MiB | 55 | 116 | 77.3 | 2.10x |
  | Llama-3.2-1B Q4_K | 778 MiB | 53 | 92 | 75.1 | 1.73x |
  | gemma-2b Q4_0 | 1744 MiB | 29 | 45 | 83.0 | 1.55x |

  Every block is placed in every row, so this is not a placement failure.

  ★★ **AND METAL LOSES BELOW ~200 MiB BECAUSE A THIRD OF THE TOKEN IS FIXED
  SESSION COST.** `TestDeviceDecodeRate` with `JITLLM_SWEEP=1` moves the layer
  limit and prices the two terms apart, which a single full-offload number
  cannot do -- tinyllama, 22 layers:

      layers on card    0      1      2      4      8     11     22
      ms/token       18.31  18.40  18.18  17.90  15.26  14.34   8.65

  The first FOUR device layers save 0.4 ms where four host layers cost 3.3.
  Fitting `token = S + nDev*d + nHost*h` gives **h = 0.83 ms, d = 0.26 ms and
  S = 2.9 ms** -- 34% of an 8.65 ms token, paid once per contiguous device run.
  `backend/metal.go` already reduced this from one command buffer per LAUNCH to
  one per SESSION (288.8 us per dispatch against CUDA's 19.3, ~254 launches a
  token); what is left is the commit-and-wait itself. The "Metal indirect
  command buffer" entry on the shelved-levers list was shelved at "a 3% ceiling" --
  **that estimate is wrong by an order of magnitude** and the row above is why.

  ★★★ **AND AGAINST THE ORACLES jitllm LOSES ON METAL, ON FIVE OF SIX MODELS.
  THE M4 BOARD ABOVE IS PRE-CONTAINER AND ITS ROWS ARE NOW WRONG.** The user
  said so and the measurement agrees. llama.cpp b10985 (prebuilt macOS arm64,
  `llama-bench -p 0 -n 32 -r 3`) against `TestDeviceDecodeRate` on the same box
  and the same GGUFs:

  | model | llama.cpp | jitllm | ratio |
  |---|---|---|---|
  | tinyllama q3_K_M | 114.66 | 116.31 | 1.01 |
  | Llama-3.2-1B Q4_K | 112.61 | 92.05 | **0.82** |
  | gemma-2b | 56.57 | 45.38 | **0.80** (was 0.9509 pre-container) |
  | Qwen3-MoE-4x0.6B | 133.80 | 95.84 | **0.72** |
  | SmolVLM-256M Q8_0 | 335.06 | 260.53 | **0.78** |
  | stories15M q8_0 | 1268.50 | 731.36 | **0.58** |

  And MLX, Llama-3.2-1B-Instruct 4-bit, 128 tokens: **131.4 / 131.9 tok/s**
  against jitllm's 92.05 -- **0.70x**. (MLX's 4-bit is its own group quant, not
  Q4_K_M, so that row is indicative; the llama.cpp rows are the same file.)
  These are two harnesses in two passes, NOT a RULE 2 row -- but 0.58-0.82 is
  far outside harness noise and the direction is not in doubt.

  ★★ **AND THE MATRIX UNITS ARE NOT THE GAP, WHICH THE ORACLE SETTLES.**
  llama.cpp prints `tensor API disabled for pre-M5 and pre-A19 devices` on this
  M4 -- it wins **without** simdgroup_matrix. So the MMA item above is a real
  absence and NOT the reason for these rows, and building it first would be
  optimising the wrong term.

  **The submission cost is the whole gap, and the arithmetic closes.** At
  S = 2.9 ms on tinyllama's 22 layers, removing it takes 8.65 ms/token to 5.75
  -- **174 tok/s against llama.cpp's 114.66**, a 1.5x win rather than parity.
  On Llama-3.2-1B the same subtraction gives ~126 against 112.61 and lands
  beside MLX's 131.7. jitllm already has shared buffers
  (`newBufferWithBytesNoCopy`, metal_darwin.go) and does NOT have residency
  sets, which llama.cpp announces as `use residency sets = true`; and
  `mtlSession.finish` ends in `waitUntilCompleted`, which is the blocking commit
  the 2.9 ms is made of.

  ★★ **AND MLX'S EDGE IS THREE THINGS, READ OUT OF ITS SOURCE, NONE OF THEM A
  KERNEL.** Cloned and compared on the user's suggestion:

  | | MLX | jitllm |
  |---|---|---|
  | after commit | makes the NEXT buffer and returns | `waitUntilCompleted` |
  | command buffer | `commandBufferWithUnretainedReferences` | `commandBuffer` |
  | residency | `MTL::ResidencySet`, standing `requestResidency()`, size-capped | none |
  | flush policy | `max_ops_per_buffer_ = 20` or an MB cap | one per caller Session |

  ★★★ **AND ITEM 1 BELOW WAS MEASURED AND IS WRONG ABOUT THE SIZE. SUBMISSION IS
  12% OF A METAL TOKEN, NOT 34%.** The async commit was BUILT (`Ctx.Wait`,
  `Batch.Commit` no longer blocks) and then instrumented, which is the only
  reason the error is visible:

      commits per token                 1.0     (batching was already right)
      Wait calls per token              ~6, of which ONE blocks
      encode                            0.034 ms/token over 89 dispatches
      blocked in waitUntilCompleted     1.35 ms/token
      wall inside backend Session       1.06 ms/token  <- bounds submission
      the token                         8.6 ms

  One command buffer per token and one blocking wait per token is what an
  autoregressive decode REQUIRES -- the next token cannot be encoded before this
  one's logits exist -- so there was never 2.9 ms of submission to remove. The
  earlier fit found a real fixed term and attributed it to the wrong mechanism:
  `S + nDev*d + nHost*h` has no term for "the GPU is slower per layer than the
  fit assumed", so that cost landed in S. **A model with three free parameters
  and seven points will name whatever you ask it to name.**

  ★★★ **AND SUBMISSION HAS A 3% CEILING, MEASURED BY THE TOOL BUILT TO MEASURE
  IT.** `jitllm verify -devices metal` on the M4, which prints exactly this:

      9.72 ms per token in Layers
      0.32 ms per token encoding launches (3% of Layers) -- the ceiling on a
        recording
      0 launch-sequence captures

  `verify.go`'s own comment says *"an indirect command buffer would encode once
  and replay, so this number is the ceiling on that change -- quote it before
  building one."* It was there the whole time and both of my submission
  estimates were made without reading it. **The shelved-levers list's "3% ceiling" for
  the Metal indirect command buffer was RIGHT, and the entry above calling it
  "wrong by an order of magnitude" was itself wrong.** Metal decode is GPU-BOUND.

  ★★ **AND MLX IS NOT FASTER PER BYTE THAN llama.cpp; IT READS FEWER BYTES.**
  Llama-3.2-1B: MLX's 4-bit group quant is ~0.70 GB at 131.7 tok/s = **92 GB/s**;
  llama.cpp's Q4_K_M is 0.81 GB at 112.61 = **91 GB/s**. The same rate. MLX's
  1.35x is its QUANTIZATION, not its kernels, and a row comparing the two
  without saying so compares two different models. jitllm is at ~84 GB/s on the
  same file -- a real ~8% kernel deficit, and much smaller than the 0.70x row
  suggests.

  ★★★ **THE M4's READ WALL IS 105.7 GB/s, MEASURED, AND EVERY EARLIER "% OF
  120" IN THIS ENTRY WAS AGAINST A SPEC SHEET.** MLX summing a contiguous 1 GiB
  float32 array, ten passes: **105.7 GB/s**. A neutral third party on the same
  box, and the same access pattern a matvec has. Against it:

      llama.cpp  Llama-3.2-1B Q4_K_M   112.61 tok/s x 0.808 GB =  91.0 GB/s   **86%**
      jitllm     same file              99.40 tok/s x 0.816 GB =  81.1 GB/s   **77%**

  So the gap is NINE POINTS OF WALL EFFICIENCY on a memory-bound kernel, not
  bytes and not submission -- and it is the whole of the 0.88 ratio.

  ★★★ **AND THE PER-FORMAT "28% PER-BYTE GAP" IS AN ARTIFACT OF CACHE
  RESIDENCY, RE-MEASURED ON A QUIET M4 AND REPRODUCIBLE TO 1 GB/s.**
  At ONE SHAPE (32768x2048) the six formats appeared to split into two camps --
  `Q3_K 100  Q4_0 99  Q4_K 99 | Q5_K 75  Q8_0 73  Q6_K 71` -- and a whole goal
  was written on it. But a 4-bit payload is HALF an 8-bit one, so at one shape
  the two camps sit at different points on the cache hierarchy and the faster
  row is simply the one that FITS. At 384 MiB EACH (`mvbench -eqbytes 384`),
  three runs:

    | | Q4_0 | Q4_K | Q3_K | Q8_0 | Q6_K | Q5_K |
    |---|---|---|---|---|---|---|
    | run 1 | 96 | 99 | 99 | 96 | 97 | 97 |
    | run 2 | 95 | 99 | 98 | 97 | 97 | 97 |
    | run 3 | 96 | 99 | 98 | 97 | 97 | 98 |

  **All six read 95-99 GB/s and the SLOWEST is Q4_0, a 4-bit format.** There is
  no per-byte penalty on the 8-bit or two-plane payloads. The remaining M4
  deficit is BYTES (Q3_K +34.5%; Q5_K was +56.8% when this was written and is
  **+2.3%** since container v15) and the nine points of wall efficiency above --
  not this.

  ★★ **AND THE TELL WAS PRINTED THE WHOLE TIME -- THE ROWS READ 103-110% OF THE
  WALL -- BUT THE DIVISOR IS NOT MERELY TOO SMALL. IT MEASURES THE PROBE.**
  `streamWall`'s answer is a function of its own working set, swept on a quiet
  M4:

      64 MiB 71.2 | 128 83.4 | **256 90.0** | 512 76.1 | 1024 67.2 | 2048 60.9 GB/s

  **It PEAKS at 256 MiB and falls away on both sides**, which no memory wall
  does. Rising to the peak is fixed overhead amortising; falling past it is the
  probe's own pattern -- it grid-strides at a 1 MiB stride with a FIXED thread
  count, so a bigger working set means each thread walks more distinct pages.
  Against MLX's 105.7 GB/s over a contiguous 1 GiB sum on the same box, every
  point on that curve is low, and the matvec rows sit at a stable 96-99 GB/s
  across all of them -- 91-94% of the MLX figure, which is where a saturated
  memory-bound kernel belongs.

  **So the ROWS are reproducible and the WALL is not, and no "% of wall" from
  this tool is quotable until the probe's access pattern is fixed.** Its comment
  claims "same access shape as the matvec"; a matvec that beats it by 7-9 points
  says otherwise. `-wallmib` makes the working set explicit so this is
  measurable rather than assumed.

  ★★★ **AND MY OWN FIX FOR IT WAS A GUESS, COMMITTED BEFORE IT WAS MEASURED,
  AND IT MADE THINGS WORSE.** The reasoning was "a probe over a smaller working
  set under-reads, so match the rows at 1 GiB". That took the divisor DOWN 28%
  and the percentages from 106% to **152%**. The hypothesis being wrong is
  ordinary; shipping it to the Mac and writing "the probe streams 1 GiB now" in
  a commit message before running it is not, and it is the same failure this
  file spends pages on one level up -- a number reported before it was measured.
  RULE 1's "above the roofline is a harness bug" found the bug for the second
  time here (the first was Metal's `Sync` returning before the GPU started, at
  106794% of wall); it did not license the first guess at a cure.

  ★ **THE OTHER HALF OF THE TOOL IS STILL BIASED AND IS NOT FIXED: the
  per-format headline is a MAX OVER SIX SPLIT VALUES**, which drifts upward with
  the noise. On the same-shape rows it shows plainly -- Q3_K read 98, 60, 98
  across three identical runs on a quiet box. The `-eqbytes` rows above do not
  show it because they are flat across splits (93 96 94 91 86 76), which is why
  they reproduce to 1 GB/s and the same-shape ones do not. Quote `-eqbytes`.

  ★★★ **RE-CONFIRMED, TWO BACK-TO-BACK RUNS ON A QUIET M4, AND THE
  SAME-SHAPE ROW FOR THE FORMAT THE OLD GOAL NAMED SWUNG 95 -> 71.** This is the
  sharpest available statement of the bias, because 71 is exactly the number a
  standing goal was written on ("Q8_0, Q5_K, Q6_K at 71-75 GB/s against 99-100
  for the 4-bit ones"):

    | row | run 1 | run 2 |
    |---|---|---|
    | `tll lm_head Q6_K` (same shape) | **95** | **71** |
    | `tll q/o Q4_K` (same shape) | 84 | 75 |
    | `tll ffn_down Q4_K` (same shape) | 75 | 57 |
    | eq/Q4_0 | 97 | 95 |
    | eq/Q8_0 | **97** | **97** |
    | eq/Q4_K | 99 | 98 |
    | eq/Q6_K | **98** | **98** |
    | eq/Q3_K | 99 | 99 |
    | eq/Q5_K | **96** | **97** |

  **The six equal-byte rows span 95-99 and reproduce to 1-2 GB/s; the
  same-shape rows for the same formats swing by up to 24 points between
  consecutive runs.** Q4_0 -- a 4-bit format -- is the SLOWEST of the six. So
  the "28% per-byte gap on the 8-bit and two-plane payloads" does not exist on
  Metal, and the goal built on it was aimed at one sample of an unstable row.
  **The Metal deficit is the nine points of WALL EFFICIENCY above, which is not
  per-format.** Note the rows read 104-109% of wall, which is the probe
  measuring itself (see two entries up) -- the rows are comparable to EACH OTHER
  and no "% of wall" from this tool is quotable.

  ★★★★ **AND THE ADJACENT-ROW TILE THAT FOLLOWED FROM ALL THIS IS A 7%
  REGRESSION, AS MEASURED. THE CONTIGUITY FINDING IS REAL AND DOES NOT
  TRANSFER TO THE MATVEC.** `TestDeviceDecodeRate`, Llama-3.2-1B Q4_K_M, Metal,
  16 of 16 blocks placed, one process per sample, ABBA, A/A self-control FIRST,
  on an idle M4:

    | | ratio | IQR/median | n | |
    |---|---|---|---|---|
    | **A/A (rowt 1 in BOTH arms)** | **1.0020** | **0.0035** | 8 | gates |
    | pass 1  rowt4 / rowt1 | **0.9264** | 0.047 | 8 | gates |
    | pass 2  rowt4 / rowt1 | **0.9329** | 0.043 | 8 | gates |

    rowt1  104.27 tok/s = 85.1 GB/s = **78% of the 108.9 GB/s wall**
    rowt4   96.97 tok/s = 79.1 GB/s = **73%**

  Two passes agreeing to 0.65 points behind a 0.2% harness bias. **The
  configuration WAS selected** -- the counters read `224 tiled (224 of them WITH
  a split), 0 plain` against the control's `0 tiled, 224 plain`, which is what
  `Stats.RowtSplitTiles` exists for.

  ★ **SO THE PROBE AND THE KERNEL DISAGREE, AND THAT IS THE FINDING RATHER THAN
  A DISAPPOINTMENT.** `StreamWide` is reproducible to 1%: a pure read gains 1.6x
  going from 4 B to 32 B per lane, and a strided arm with the SAME request count
  gains nothing, so contiguity is genuinely what a streaming read wants. A Q4_K
  matvec does not want it, because it is not load-latency-bound in the way the
  probe is -- it unpacks nibbles, rebuilds a scale from an f16 and a 6-bit
  integer and centres, and that arithmetic already hides the load. **The
  remaining eight points are therefore NOT in the load shape**, which is a real
  narrowing: it eliminates the one mechanism that a same-bytes same-grid probe
  could demonstrate.

  ★ **AND THE TILE COSTS DISPERSION, WHICH IS ITS OWN SIGNAL.** The rowt1 arm
  spreads 1.9% across sixteen samples and the rowt4 arm **11.0%**, and the
  selection check shows why: the split tuner chose 224 tiles in one pass and 192
  in the other, from the same binary on the same model. A tile makes the kernel
  far more sensitive to a per-process choice that AGENTS.md already prices at
  0.85 points.

  ★★ **WHAT SURVIVES AND MUST NOT BE LOST WITH IT.** The branch also fixes a
  LIVE CORRECTNESS BUG that has nothing to do with the tile and should land on
  its own terms: **Q5_K on the MMA path threw its fifth bit away**, so CUDA
  prefill decoded five-bit weights as four-bit for every Q5_K tensor since v15.
  `backend.TestMatVecMMAMatchesDot4` was RED for it (`1024/1024 elements differ,
  worst |delta| 1563.26`) and had been. It is the THIRD plane change to reach
  `matvec.go` and not `mma.go`, so the fix derives the lane's element group from
  the format's arithmetic instead of adding a third `case`, and refuses by name
  a shape where that index is not constant over the lane.

  ★ **AND "the decode row tile is a regression" IS NOW TRUE FOR TWO DIFFERENT
  REASONS, WHICH IS WORTH KEEPING SEPARATE.** The earlier decode-row-tile entry
  (under "Tried or priced, and each came out a REGRESSION") measured a SPREAD
  tile (`tileOff(j) = j*(Rows/Rowt)`) at 49.41 -> 9.36 tok/s and attributed it
  to thread starvation. That attribution was incomplete -- the spread also
  destroyed contiguity -- and the ADJACENT tile fixes the
  contiguity, composes with the split so the grid is not divided, and is STILL
  7% slow. So thread count was not the whole story and neither is contiguity.

  ★★★★★ **AND THE EIGHT POINTS ARE TWO TERMS, NOT ONE -- MEASURED BY
  THE GPU's OWN CLOCK, AND THE ON-DEVICE HALF IS PER LAYER RATHER THAN PER
  BYTE.** Every counter in `backend/metal` is a CPU timer: they bound
  SUBMISSION and cannot tell a device that is streaming from one idling between
  two dependent dispatches. `MTLCommandBuffer.GPUStartTime/GPUEndTime` bracket
  execution ON the device, so `token wall - GPU busy` is a measurement rather
  than the residual of a fit. Llama-3.2-1B Q4_K_M, Metal, 16 of 16 placed:

      token wall                       10.168 ms
      GPU executing (its own clock)     8.939 ms
      off the device                    1.229 ms  (12%)
      streaming floor at the wall       7.493 ms  (816,010,912 B / 108.9 GB/s)
      **ON the GPU and NOT streaming    1.446 ms**

  The three add to the token exactly, and against llama.cpp on the same GGUF
  (8.591 ms, floor 7.345, everything else 1.246 -- NOT split, there is no GPU
  clock on that process) jitllm's total non-streaming is 2.675 against 1.246.

  ★★★★ **AND THE "PER LAYER" HALF OF THAT WAS OVERTURNED BY A REGRESSION THE SAME
  DAY. IT IS TWO TERMS, AND THE FIXED ONE IS THE BIGGER.** The three-model table
  below divides the whole on-device excess by the layer count, which ASSUMES
  there is nothing fixed. Sweeping layers-on-card and bracketing the GPU clock
  round each arm tests that assumption instead of making it -- five points, two
  parameters, residuals under 0.08 ms:

      GPU busy = **2.554 ms + 0.400 ms x nLayers**

  The container says what to subtract (`PageSize` and `DenseBytes`, read from
  the model rather than estimated from the vocabulary): 16 blocks of 39.6 MB and
  a 182.1 MB dense read per token, which at 108.9 GB/s is 0.364 ms a block and
  1.672 ms for the dense part.

      per layer   0.400 - 0.364 = **0.037 ms**  (37 us, over 25.5 dispatches
                                                 = 1.4 us each)
      fixed       2.554 - 1.672 = **0.882 ms**  present with ONE layer placed

      16 x 37 us = 0.588 ms  +  0.882 ms fixed  =  **1.470 ms**
      directly measured on-device non-streaming: 1.446 / 1.521 ms -- IT CLOSES

  **So 6.2% of the token is per-layer and 9.3% is fixed per token.** The earlier
  "79-99 us per layer" did not hold: it was the sum of both divided by layers,
  and it made whole-layer fusion look like a 14% lever when the per-layer term
  it could attack is 6.2% and only part of that is removable.

  ★★ **AND THE FIXED TERM IS THE NEW TARGET, WHICH NOTHING HAS EVER LOOKED AT:
  0.882 ms of device time per token that does not depend on how many blocks are
  placed.** The obvious suspect was the OUTPUT HEAD -- the dense region is the
  head, 27% of a dense model's bytes, one matvec run once a token -- and it is
  REFUTED by sweeping two more models whose heads differ by 2.6x:

    | model | per-layer excess | as % of block streaming | FIXED excess | dense read |
    |---|---|---|---|---|
    | tinyllama q3_K_M | 0.017 ms | **6.9%** | (1.342) | 0.225 ms |
    | Llama-3.2-1B Q4_K | 0.036 ms | **9.9%** | **0.882 ms** | 1.672 ms |
    | gemma-2b Q4_0+Q8_0 | 0.049 ms | **8.6%** | **0.963 ms** | 5.120 ms |

  **The fixed term moves 1.09x while the head moves 2.6x, so it is not the
  head.** tinyllama is excluded rather than averaged in: its fit has a 0.372 ms
  worst residual and its one-layer arm is SLOWER than its zero-layer arm, which
  is the small-model regime this file already records Metal losing in.

  ★ **AND SEPARATING THE TWO TERMS CORRECTS WHAT "FLAT IN BYTES" MEANT.** The
  PER-LAYER excess is **7-10% of the block's own streaming time on all three** --
  so it is per byte after all, and the matvec simply runs at ~91% of the wall.
  What is flat in bytes is the FIXED term, which the three-model table below was
  measuring without knowing it.

  ★★★★ **AND THE FIXED TERM IS THE HEAD MATVEC AFTER ALL -- NOT ITS SIZE, ITS
  EFFICIENCY. THE HEAD RUNS AT 65% OF THE WALL WHERE THE BLOCKS RUN AT 91%.**
  The five fixed dispatches are identified in the SOURCE rather than inferred:
  `layer.go`'s `if head != nil` arm issues exactly `normPart`, `normApply`,
  `quantE` and `mvrun(bs.mvHead)` plus its reduce -- four or five, against the
  5.5 the sweep measures. So the whole fixed excess sits on that sequence, and
  the matvec is the only member big enough to hold it.

    | model | head format | MB | floor | excess | total | % of wall |
    |---|---|---|---|---|---|---|
    | Llama-3.2-1B | **Q6_K, two-plane** | 182.1 | 1.672 | 0.882 | 2.554 | **65%** |
    | gemma-2b | **Q8_0, flat byte** | 557.6 | 5.120 | 0.963 | 6.083 | **84%** |
    | (their block matvecs) | Q4_K / Q4_0 | -- | -- | -- | -- | **91%, 92%** |

  **That resolves the "not the head" paradox above rather than contradicting
  it.** The fixed term does not track head BYTES (1.09x against 2.6x) because
  the two heads run at different EFFICIENCIES -- 65% against 84% -- and the
  bytes and the efficiency move in opposite directions. Both statements hold:
  it is the head, and it is not proportional to the head.

  ★★ **AND THE FORMAT EXPLANATION IS REFUTED BY THE SAME TWO NUMBERS, WHICH IS
  WHY THE ABSOLUTE EXCESS IS THE THING TO READ AND NOT THE PERCENTAGE.** Llama's
  head is Q6_K (4+2 two-plane) and gemma's is Q8_0 (flat byte), so a format
  penalty predicts gemma's excess to be SMALLER, and per-byte it predicts
  gemma's to be roughly 3x LARGER since its head is 3x the bytes. Measured:
  **0.882 ms and 0.963 ms -- 1.09x.** Neither prediction survives. The "65% vs
  84% of wall" is the same fixed ~0.9 ms divided by two different streaming
  times, and reading it as an efficiency difference is reading a constant as a
  rate.

  **So the head's excess is a FIXED ~0.9 ms attached to that five-dispatch
  sequence, independent of the head's size and of its format.** The equal-bytes
  sweep is also not evidence against it in either direction: `-eqbytes 384`
  already produces TALL matrices (Q4_0 at ~333k rows, Q8_0 at ~172k), so the
  aspect ratio was never the untested variable -- and it times a LOOP of one
  kernel, where the head runs ONCE, last, at the end of a 400-dispatch dependent
  chain with nothing left to overlap it. A drain at the tail of the command
  buffer is fixed-cost by nature and is what these two numbers look like.
  **Unmeasured**, and it is what the next probe should separate: the head's cost
  in situ against the same kernel run alone.

  ★★★ **AND THE UNRETAINED COMMAND BUFFER IS BUILT AND REFUTED, WHICH IS THE
  CHEAPEST OF MLX's THREE EDGES AND THE ONLY ONE TESTABLE IN A LINE.** A normal
  command buffer retains and releases every resource it references; at 413
  dispatches a token with six buffers each that is thousands of pairs, and
  `-commandBufferWithUnretainedReferences` is the same call without them.
  Paired, ABBA, one process per sample, on an idle M4, reading the GPU's own
  clock:

      per-round ratios   0.9603  0.9963  0.9978  1.0072  1.0087
      **median 1.00** -- NULL, against an effect that would have to be 10%

      retained    8.925  8.923  9.348  9.016  9.022   (spread 4.7%)
      unretained  9.003  8.987  8.977  8.996  8.989   (spread **0.3%**)

  **So the fixed ~0.9 ms is WORK, not bookkeeping** -- which in hindsight is the
  right answer for the right reason: ARC traffic is CPU-side and this term is
  bracketed by GPUStartTime/GPUEndTime. `metal.Ctx.Unretained` and
  `backend.SetMetalUnretained` stay, default OFF, as the instrument that
  produced the row; the knob trades a lifetime guarantee for work and is safe
  only while the buffers outlive the submission, which the packed arena does on
  a fully resident model and does NOT across a page-in.

  ★★★★ **AND THE RESIDENCY SET IS BUILT AND REFUTED TOO, SO ALL THREE OF MLX's
  NAMED EDGES ARE NOW ELIMINATED BY MEASUREMENT ON THIS ENGINE.**
  `metal.Ctx.Residency` creates an `MTLResidencySet`, attaches it to the
  QUEUE -- not the command buffer, because a per-buffer `useResidencySet:` pays
  the association once a token, which is the cost being hunted -- and adds every
  buffer, INCLUDING the imported ones, which is where the ~850 MB actually is.
  Paired, ABBA, one process per sample, on an idle M4:

      ratios  0.9914  0.9950  1.0003  1.0032  1.0046  1.0056
      **median 1.0018, IQR/median 0.006** -- +0.2% against an effect that would
      have to be 10%

    | MLX edge | status |
    |---|---|
    | async commit | BUILT, kept for correctness, ~6% over 600 MiB |
    | unretained references | BUILT, **REFUTED** (median 1.00) |
    | residency sets | BUILT, **REFUTED** (median 1.0018) |

  **So the fixed ~0.9 ms is not submission, not ARC traffic and not residency.
  It is work inside those five head dispatches**, and the remaining candidate is
  the TAIL DRAIN -- the head runs last, after a 400-dispatch dependent chain,
  with nothing left to overlap it.

  ★ **AND THAT NEEDS A DIFFERENT INSTRUMENT, WHICH IS WHY IT IS NOT GUESSED AT
  HERE.** A per-BUFFER clock says how long the device was busy and cannot say
  WHERE in the buffer the time went; a drain at the tail and a slow kernel in
  the middle are the same number to it. `MTLComputePassDescriptor`'s sample
  buffers give per-DISPATCH timestamps and would separate them. Unbuilt.

  ★ **TWO KNOBS SURVIVE AS INSTRUMENTS, BOTH DEFAULT OFF**
  (`backend.SetMetalUnretained`, `backend.SetMetalResidency`). Each refuses
  rather than degrades: residency FAILS TO OPEN on a queue that does not answer
  `addResidencySet:`, because a set attached to nothing reads as "residency is
  not the cause" -- RULE 10's shape, and the first version of that check was an
  empty `if` that would have produced exactly that.

  ★★★★★ **AND THE FIXED TERM IS THE OUTPUT NORM AND ITS QUANTIZE, NOT THE HEAD
  MATVEC. IT TRACKS NEmbd AND IS FLAT IN VOCABULARY, ACROSS SIX MODELS.** The
  five fixed dispatches split into two families: `mvHead` (+ its reduce) scales
  with **vocab x NEmbd**, and `normPart`, `normApply`, `quantE` scale with
  **NEmbd alone**. Those axes separate across the models on this box, and the
  layer sweep gives an intercept for each:

    | NEmbd | vocab | FIXED | model |
    |---|---|---|---|
    | 288 | 32000 | **0.010** | stories15M q8_0 |
    | 576 | 49152 | **0.284** | SmolVLM-256M Q8_0 |
    | 960 | 49152 | **0.705** | SmolLM2-360M Q8_0 |
    | 2048 | 32000 | 1.218 | tinyllama q3_K_M |
    | 2048 | 128256 | 0.880 | Llama-3.2-1B Q4_K |
    | 2048 | 256000 | 0.998 | gemma-2b |

  **Monotonic in NEmbd over a 7x span, and within the NEmbd=2048 group the
  vocabulary spans 8x with no ordering at all.** So the head matvec is
  exonerated -- its bytes are already subtracted as the dense read, and what is
  left does not care how many rows it had. **The three NEmbd-shaped dispatches
  are what is left**, and 1 ms is absurd as ARITHMETIC over 2048 floats, which
  says it is their STRUCTURE: three dependent dispatches, each a full round trip,
  two of them with a grid too small to fill the device.

  ★★★★★ **AND FUSING THEM IS REFUTED BEFORE IT WAS BUILT, BY PRICING A DISPATCH
  INSTEAD OF ASSUMING ONE. A METAL DISPATCH COSTS 1.1 us, NOT 300.** `normApply`
  reads x, hNorm and part and writes h, so it is IDEMPOTENT: running it n extra
  times computes the same answer and costs exactly n extra tiny dispatches.
  `tier.Config.HeadNormRepeat` does that, and the selection check is the
  dispatch count itself -- **468 -> 669 at r=200, +201 for +200 requested**:

      r=0    468 dispatches   GPU 8.962 ms   104.60 tok/s
      r=50   487              GPU 9.095      103.06
      r=200  669              GPU 9.179      101.13
      -> **1.09 us per dispatch**

  ★ **AND IT AGREES WITH THE PER-LAYER TERM, MEASURED INDEPENDENTLY, WHICH IS
  WHAT MAKES IT A NUMBER RATHER THAN A READING.** The layer regression gives 37
  us of excess over 25.5 dispatches a layer = **1.45 us each**. Two instruments,
  two models of the cost, one answer.

    | term | dispatches | at 1.2 us | measured | verdict |
    |---|---|---|---|---|
    | per-layer, 6.2% | 25.5/layer | 31-37 us | **37 us** | **FULLY explained** |
    | fixed, 9.3% | 5 | 6 us | **880 us** | **not dispatch overhead** |

  **So whole-layer fusion is a CONFIRMED lever worth 6.2%** -- the per-layer
  excess is its dispatch count and nothing else -- **and the fixed 0.9 ms is
  not, by a factor of 150.**

  ★★ **AND THE SAME PROBE EXONERATES THE THREE TINY KERNELS, WHICH OVERTURNS THE
  ENTRY ABOVE.** normApply's TOTAL cost, overhead and work together, is
  1.1 us. Three such kernels cannot hold 0.9 ms between them however they are
  arranged, so "the fixed term is the output norm and its quantize" is wrong and
  the NEmbd correlation needs a different explanation.

  ★★★ **WHICH RETURNS IT TO mvHead, AND THE NEmbd SCALING NOW SAYS SOMETHING
  SHARPER THAN "IT IS THE HEAD": IT IS LATENCY-BOUND IN k.** A matvec's
  PER-THREAD work is its k-loop, and the head's k IS NEmbd. Its row count sets
  the GRID, which at 128256 rows is enormous and cannot be the constraint. So a
  cost that rises with NEmbd and ignores vocabulary is exactly a matvec whose
  threads are walking k serially -- and the head is the one matvec in the graph
  with no other work to overlap it.

  ★★★ **AND THE HEAD-ONLY k-SPLIT WAS BUILT AND REFUTES IT TOO, MONOTONICALLY.**
  `tier.Config.HeadSplit` pins the output projection's split alone and leaves
  every block matvec on the tuner, which the global knob could not do:

      tuned 8.930 ms   105.34 tok/s      16   9.113  +2.0%
      2     8.961     +0.3%              32   9.395  +5.2%
      4     8.935     +0.1%              64   9.768  +9.4%
      8     9.030     +1.1%

  **Splitting the head HARDER is monotonically WORSE.** A shorter k-loop per
  thread is exactly what the latency hypothesis asked for and it does not pay --
  the extra partials and the wider reduction cost more. So the head is not
  latency-bound in k either, and the tuner's choice is right for it after all.

  ★★★★ **AND THE HEAD MATVEC IS PRICED DIRECTLY AT 2.116 ms, WHICH IS 79% OF THE
  WALL -- THE SAME AS THE MODEL OVERALL. IT IS NOT ANOMALOUSLY SLOW, AND EVERY
  "65% of wall" READING OF IT IN THIS FILE WAS WRONG.** `mvrun` reads the
  head weights and the quantized activation and writes logits through its
  partials, so it is IDEMPOTENT and `tier.Config.HeadMatvecRepeat` prices it the
  way HeadNormRepeat priced a dispatch:

      r=0  9.014 ms    r=1  11.161    r=2  13.283    r=4  17.484
      slope **2.116 ms per head matvec**, residuals under 0.04
      streaming floor 1.672 ms -> **79% of wall, excess 0.444 ms**

  The 65% figure came from attributing the WHOLE fixed term to the head. Half of
  it is the head's, and the head is exactly as efficient as everything else.

  ★★★ **AND THAT LEAVES 0.448 ms AT nDev=1 THAT NOTHING MEASURED ACCOUNTS FOR,
  WHICH IS THE SHARPEST STATEMENT OF THE OPEN QUESTION THIS FILE CAN MAKE.**
  Every term below is measured, not modelled:

      one block (0.364 streaming + 0.036 excess)      0.400 ms
      the head matvec, priced by repetition           2.116
      normPart + normApply + quantE, at ~1.1 us each  0.003
      ------------------------------------------------------
      accounted                                       2.519
      GPU busy at nDev=1, measured                    **2.967**
      **UNACCOUNTED                                   0.448**

  ★★★ **AND THE RESIDUAL MAY BE THE PROBE, NOT THE ENGINE -- WHICH MEANS THE
  CORRECTION ABOVE IS ITSELF PROVISIONAL.** A repeat slope is a MARGINAL cost,
  and the increments say the margin is falling:

      +1 repeat  +2.147 ms      +1 more  +2.122      +2 more  +2.101 each

  Independent dispatches of the same kernel can pipeline with one another -- the
  tail of one overlapping the head of the next -- so the slope UNDERSTATES what
  the first, in-situ execution costs, by up to exactly the 0.448 residual. Two
  readings survive and a per-COMMAND-BUFFER clock cannot separate them:

      (a) the head matvec is 2.116 in situ, **79% of wall**, and 0.448 ms of
          device time belongs to no dispatch anyone has priced
      (b) the head matvec is 2.564 in situ, 65% of wall, the repeats pipeline,
          and dropping the 65% figure one entry ago was premature

  **(b) IS DEAD -- twice, by the two entries below: serial dispatch makes
  pipelining impossible, and the own-command-buffer measurement reads 2.106 ms
  cold. (a) is the answer. This pair is kept only because the way a marginal
  cost can masquerade as an in-situ one is worth recognising again.**

  ★★★★★ **AND IT IS SETTLED: THE HEAD SEQUENCE IS 2.106 ms MEASURED IN ITS OWN
  COMMAND BUFFER -- IN SITU, COLD, ONCE -- AGAINST THE REPEAT PROBE'S 2.116.
  THEY AGREE TO 0.5%, SO COLD-VERSUS-WARM IS REFUTED TOO AND READING (a)
  STANDS.** `PerLayerSubmit` already ends with `layersOnce(bs, hi, hi, ...,
  head)` -- an empty layer range carrying only the projection -- so its FINAL
  command buffer holds the head sequence and nothing else, and
  `metal.GPUBusyLast` times exactly that. The first read of 182 MB costs what an
  immediate re-read costs.

      head sequence   1.672 ms floor / **2.106 ms measured = 79% of wall**

  **The head runs at the same 79% the whole model runs at. The "65% of wall"
  figure is refuted for good, and so is every reading that made the head the
  problem.** What remains is a ~0.45 ms residual that belongs to no dispatch
  anyone has priced, at every layer count, and a per-command-buffer clock has
  now been pushed as far as it goes.

  ★★ **AND THE SAME RUN PRICES ONE-SUBMISSION-PER-BLOCK, WHICH IS A STRONG
  NEGATIVE WORTH KEEPING.** It is the arm `-ab submit` exists to compare and
  nobody had put a rate on it:

      default          **105 tok/s**   1 command buffer/token   12% off-device
      PerLayerSubmit   **70.6**        17 buffers/token         **39% off-device**

  **One command buffer per block costs a third of the rate**, and it lands
  entirely off the device -- GPU time is unchanged at 9.04 ms while the token
  goes from 10.2 ms to 14.8. The single-buffer design is not a convenience; it
  is worth 33%.

  ★★ **AND THE PIPELINING HALF IS REFUTED BY A CODE READ, NOT A MEASUREMENT.**
  `metal_darwin.go` creates its encoder with `computeCommandEncoder` and no
  `WithDispatchType:`, which is **MTLDispatchTypeSerial** -- Apple's default,
  where each dispatch completes before the next begins. Two dispatches in this
  encoder CANNOT overlap, so a repeat cannot hide behind its predecessor and the
  slope is not a pipelined margin. The 2.147 -> 2.101 drift is 2% on a harness
  whose own spread is 1-2%.

  ★ **THE COLD-VERSUS-WARM READING THAT SURVIVED THAT IS ALSO DEAD, AND THE
  ORDER OF THESE ENTRIES MATTERS MORE THAN IT LOOKS.** Serial dispatch stops
  overlap but not cache state: a 182 MB stream touched once a token and again
  microseconds later is where TLB and prefetch differ, so the repeat slope
  could still have measured a WARMER read than the token pays. **The
  own-command-buffer measurement above settles it -- 2.106 ms cold and in situ
  against the repeat's 2.116, agreeing to 0.5%.** There is no warm-up premium.

  **Reading (a) holds without qualification: the head is ~2.11 ms in situ at 79%
  of wall, and 0.45 ms belongs to no priced dispatch.** This paragraph is kept
  BELOW the entry that settles it on purpose -- the earlier draft left the
  provisional "unless" as the last word on the page, which is exactly how this
  file's header describes a withdrawn conclusion keeping the appearance of a
  live one.

  ★★★ **AND FORCING THE IN-GROUP REDUCTION IS A SMALL REAL WIN AND AN
  UNAMBIGUOUS DETERMINISM WIN, WHICH IS THE ONE REDUCTION THIS LINE PRODUCED.**
  `GroupSplit` reduces the k-split's partials in SHARED MEMORY instead of
  launching a separate reduce kernel; the tuner picks it for 4 of 7 shapes and
  `ForceGroupSplit` applies it to all. Paired, ABBA, GPU clock, A/A FIRST:

      A/A self-control        0.9972   IQR/median 0.032   n=4
      pass 1  forcegroup/tuned  **0.9908**   IQR/median 0.0024   n=6
      pass 2  forcegroup/tuned  **0.9954**   IQR/median 0.0022   n=6

  **Call it ~0.5%, not 0.9%.** The two passes differ by half a point, the A/A
  bias is 0.3% IN THE SAME DIRECTION, and the effect is only 1.5-3x that bias --
  so the honest number is the smaller one and it sits near this harness's
  resolution.

  ★ **WHAT MAKES IT MORE THAN A READING IS THAT THE MECHANISM PREDICTED THE
  SIZE.** The dispatch price -- 1.09 us, measured independently by repeating an
  idempotent kernel 200 times -- times the 56 dispatches the in-group reduction
  removes is **0.7% of a 9.0 ms token**, against 0.5-0.9% measured. Two
  instruments built for different purposes agreeing is different evidence from
  six rounds pointing one way.

  ★★ **AND THE DETERMINISM IS NOT MARGINAL AT ALL: 405 DISPATCHES EVERY RUN
  AGAINST THE TUNER'S 445-509.** Same binary, same model -- `tuneSplit` chooses
  differently per process, and this file already prices that variance at 0.85
  points of dispersion and asks whether pinning the split is worth more than
  tuning it. Forcing the in-group reduction makes the whole submission shape
  deterministic, and the A/A control above shows what the variance costs: its
  one outlier pair (9.128 against 8.675) is the tuner, and it alone drives that
  control's 3.2% dispersion.

  **Whether it should be the DEFAULT is not settled here** -- the tuner already
  chooses it for most shapes, and 0.5% against a 0.3% harness bias is not a
  mandate. What is settled is that it never loses, it removes 56 dispatches, and
  it makes every future measurement on this box reproducible.

  ★★★★★ **THE FULL ACCOUNTING, EVERY TERM MEASURED BY A PROBE THAT VARIED ONE
  THING AND CARRIED A SELECTION CHECK. nDev=16, default submission:**

      streaming floor, 816 MB at 108.9 GB/s          7.493 ms
      dispatch overhead, 413 x 1.09 us               0.450
      head excess beyond floor + its dispatches      0.429
      ---------------------------------------------------
      subtotal                                       8.372
      GPU busy, measured                             **8.930**
      **UNATTRIBUTED                                 0.558**

  **93.7% of a Metal decode token is now attributed to a measured cause, and
  6.3% is not.** The unattributed part is smaller than the head sequence and
  spread across 413 dispatches, which is precisely what a per-COMMAND-BUFFER
  clock cannot split -- it is the resolution limit of every instrument built
  so far, not a gap in the reasoning.

  ★ **AND THE HONEST SHAPE OF THE REMAINING DEFICIT IS NOT WHAT THE GOAL
  ASSUMED.** Against llama.cpp's 8.591 ms token on the same GGUF: jitllm's
  streaming floor is 0.148 ms dearer (the Q4_K SC plane's +2.8% of bytes), its
  dispatch overhead is 0.450, its head excess 0.429, and 0.558 is unattributed.
  **No single term is more than 5% of the token**, which is why sixteen
  mechanisms could each be eliminated without the total moving: there was never
  one cause to find.

  ★★★★ **AND THE LIST OF WHAT IT IS NOT IS NOW LONG ENOUGH TO BE THE USEFUL
  PART.** Every one of these was
  measured, not argued:

    | candidate | how it died |
    |---|---|
    | submission | 3% ceiling, printed by `jitllm verify` |
    | ARC / retained references | built, paired, median **1.00** |
    | residency sets | built, paired, median **1.0018** |
    | the k-split, globally | every forced value loses to the tuner |
    | the k-split, head-only | monotonically worse, +0.3% to +9.4% |
    | dispatch COUNT | a dispatch is **1.09 us**; five of them is 6 us, not 880 |
    | the three tiny norm kernels | normApply's whole cost is **1.1 us** |
    | the head's SIZE or FORMAT | flat across an 8x vocabulary span |
    | a chain-length drain | fully present at nDev=1, 31 dispatches |
    | a floor in the clock | stories15M's whole GPU token is 0.540 ms |

  What survives is an empirical fact with no mechanism: **the fixed cost rises
  with NEmbd** -- 0.010 ms at 288, 0.284 at 576, 0.705 at 960, 0.88-1.22 at 2048
  -- **and the 38% spread inside the NEmbd=2048 group says NEmbd is not the whole
  story either.** The instrument that has not been built is per-DISPATCH
  timestamps (`MTLComputePassDescriptor` sample buffers); every probe above is
  per-COMMAND-BUFFER and cannot say which of the five dispatches holds it.

  ★★ **THE OTHER HALF IS NOT OPEN, BUT ITS LEVER IS ONE QUARTER THE SIZE OF THE
  TERM AND SAYING OTHERWISE WAS AN OVERCLAIM.** The per-layer 6.2% IS its
  dispatch count -- two independent measurements agree at ~1.2 us -- and it does
  NOT follow that fusion recovers 6.2%. **A transformer block cannot be one
  dispatch**: Metal has no device-wide barrier inside a kernel, and every row's
  q must be finished before any row's scores can start. Only ADJACENT
  ELEMENTWISE runs merge, and counting them in one layer:

      normPart + normApply + quantE -> 1, twice a layer    4 dispatches
      rope q + rope k -> 1                                 1
      the activation folded into the up matvec's epilogue  1
      ----------------------------------------------------------
      6 of 25.5 = 24% of the per-layer term = **1.4% of the token**

  **6.2% is what dispatch overhead COSTS. 1.4% is what fusion can RECOVER.**
  Those are different numbers and the shelved-levers list's "whole-layer fusion"
  entry, which records 14.18%/14.81% on MoE, is measuring a MIXTURE layer --
  which has far more dispatches per layer than a dense one and where the same
  reasoning gives a much larger answer. It does not transfer to a dense model.

  ★ **AND THIS REVERSES "THE FIXED TERM IS THE HEAD MATVEC" FROM TWO ENTRIES
  AGO.** That rested on Llama and gemma alone -- two points, both at NEmbd 2048,
  which cannot separate the two families. Adding four models with NEmbd from 288
  to 960 separates them, and the answer changes. Two points agreeing is not a
  correlation; it is a coincidence with a standard error of zero.

  ★★ **AND THE FOUR NEW ROWS WERE WRONG ON FIRST COMPUTATION, WHICH IS WHY THE
  PROBE NOW PRINTS THE DENSE *READ*.** A container's dense REGION is not its
  dense READ: an untied model's region holds `token_embd` AND `output.weight`,
  and a token reads all of the head and exactly ONE ROW of the embedding.
  Subtracting the region over-subtracts by the whole table and gave two models a
  **NEGATIVE** fixed term -- a cost below its own floor, impossible, and the only
  reason the error was visible. stories15M is the clean case: 19.6 MB of region,
  **9.8 MB read**, exactly half.

  ★★ **AND THE k-SPLIT IS EXONERATED TOO, WHICH WAS THE CHEAPEST REMAINING
  SUSPECT AND NEEDED NO NEW CODE.** `tuneSplit` picks ONE policy at load; every
  block matvec is 2048-8192 rows and the head is 128256, so a split right for
  the blocks could be badly wrong for a matrix 16-60x taller -- and that would
  land exactly where the time is. Forced against the tuner, reading the layer
  sweep (n=1 is ~85% head, since one block is 0.364 ms of a 3.051 ms arm):

      split    n=1 GPU   n=16 GPU    vs tuned
      tuned      3.051      8.814     --
      4          3.061      9.113     +0.3% / +3.4%
      8          3.077      9.096     +0.9% / +3.2%
      16         3.193      9.343     +4.7% / +6.0%
      32         3.414      9.758    +11.9% / +10.7%

  **The tuner wins at every point and the head does not want a different split.**

  ★★ **AND TWO MORE EXPLANATIONS DIE ON DATA ALREADY TAKEN, WHICH IS WORTH
  SAYING BEFORE ANYONE BUILDS A PROBE FOR THEM.**
  - *"A drain after a long dependent chain."* The fixed term is FULLY PRESENT at
    nDev=1, where the chain is 31 dispatches, and is the same at nDev=16 where
    it is 413 -- that is what a constant intercept under an excellent linear fit
    means. A cost that needs 400 preceding dispatches cannot be whole at 31.
  - *"The metric has a floor -- GPUStartTime/GPUEndTime brackets a fixed
    scheduling window."* stories15M-q8_0 measures **0.540 ms/token of GPU time
    in total**, which is less than the 0.88 ms floor that explanation requires.
    The instrument does not have one.

  **So the fixed term is not submission, not ARC traffic, not residency, not the
  k-split, not a chain-length drain, and not an artefact of the clock. It is
  ~0.9 ms inside five dispatches on the two large models and under 0.54 ms on a
  small one, and its mechanism is UNRESOLVED.** The instrument that has not been
  built is per-DISPATCH timestamps (`MTLComputePassDescriptor` sample buffers);
  a per-buffer clock cannot say which of the five it is.

  ★ **THE OTHER CANDIDATE, KEPT BECAUSE IT PREDICTS THE SAME SIGN AND IS NOT A
  KERNEL.** The MLX comparison lists what
  jitllm does not have: *"residency: MTL::ResidencySet, standing
  requestResidency(), size-capped | none"*, beside
  `commandBufferWithUnretainedReferences`. A command buffer with no residency
  set makes its resources resident per submission, and this engine hands it
  ~850 MB of buffers once a token. That predicts a cost roughly independent of
  WHICH model, which is what the two gating rows show. **Unmeasured**, and it is
  the next thing to price rather than the next thing to build.

  ★ **AND THE DISPATCH COUNT THAT ARGUED AGAINST FUSION WAS WRONG BY TENFOLD, IN
  THE FLATTERING DIRECTION.** `TestDeviceDecodeRate` printed "45 dispatches" by
  dividing CUMULATIVE launches by CUMULATIVE commits -- averaging the device arm
  over the whole process, warm-ups and the HOST arm included, and the host arm
  commits nothing and launches nothing. Bracketed round the timed arm it is
  **413 per token**, and the sweep gives its shape: 31 / 57 / 107 / 209 / 413 at
  1 / 2 / 4 / 8 / 16 layers, i.e. **25.5 per layer plus a fixed 5.5**. At 45 a
  token there would have been nothing for fusion to remove; at 25.5 a layer
  there is, and it is worth 1.4 us each.

  ★★★ **AND THE ON-DEVICE TERM IS FLAT IN BYTES, WHICH ELIMINATES THE UNPACK BY
  MEASUREMENT.** Three models, three format families:

    | model | bytes MB | floor | GPU | on-GPU not streaming | per layer |
    |---|---|---|---|---|---|
    | Llama-3.2-1B Q4_K_M | 816 | 7.493 | 8.939 | **1.446** | 90.4 us |
    | tinyllama q3_K_M | 618 | 5.673 | 7.416 | **1.743** | 79.2 us |
    | gemma-2b Q4_0+Q8_0 | 1672 | 15.357 | 17.138 | **1.781** | 98.9 us |

  **Bytes span 2.7x and the term spans 1.02x.** gemma-2b has the CHEAPEST unpack
  in the tree (a nibble and one scale, no minimum array and no secondary plane)
  and the LARGEST term, so this is not the k-quant arithmetic that the CPU-tier
  entries above correctly convict on their own tier, and it is not per-byte
  under any amount of noise. **The "79-99 us per layer" column that stood here
  is overturned by the regression above** -- these three cannot separate a
  per-layer term from a fixed one, because dividing by the layer count assumes
  the answer.

  ★ **SO THERE ARE TWO LEVERS AND THEY ARE DIFFERENT SIZES.** Whole-layer fusion
  attacks the per-layer 6.2% (25.5 dispatches a layer at 1.4 us of stall each);
  the shelved-levers list records it at "14.18%/14.81% on MoE", and a MoE layer has far
  more dispatches than a dense one, so those numbers are consistent with this one
  rather than equal to it. The FIXED 9.3% is bigger and has no entry on that list
  at all.

  ★ **WHAT THIS DOES NOT ESTABLISH, STATED RATHER THAN IMPLIED.** Three models,
  one sample each -- NOT RULE 2 rows. What survives that is the SHAPE, not the
  digits: a quantity that moves 1.02x while its supposed driver moves 2.7x is
  not per-byte under any amount of noise. Per-LAYER and per-DISPATCH cannot be
  separated by these three, because dispatches per layer is roughly constant
  across dense models. And llama.cpp's 1.246 ms is NOT split by a GPU clock, so
  which half of jitllm's excess it avoids is still unknown -- Metal System Trace
  on both engines is the instrument, and it has still not been taken.

  ★ **AND EVERY TUNABLE IS ALREADY AT ITS OPTIMUM, WHICH IS WHAT MAKES THIS A
  KERNEL REWRITE RATHER THAN A SETTING.** Swept on the M4, Llama-3.2-1B, -n 64:

      SPLIT   0(fitted) 99.15 | 1 91.88 | 2 97.08 | 4 98.64 | 8 99.64 | 16 96.27
      ROWT    1 98.88 | 2 98.69 | 4 99.88 | 8 98.85
      TOK     1 99.11 | 2 99.03 | 4 98.76
      LANES   16 99.08 | 32 98.75
      MVACC   1 99.42 | 2 100.05 | 4 99.79 | 8 98.67
      QTILE   64 98.59 | 128 99.59 | 256 99.39

  Every knob lands between 98.6 and 100.1 -- one band, no outlier. The kernel is
  saturated at its current design point and the remaining nine points need a
  different one. **MLX's `qmv_fast` is the shape to read**
  (`mlx/backend/metal/kernels/quantized.h`): `num_simdgroups = 2`,
  `results_per_simdgroup = 4` -- eight output rows per threadgroup, one row per
  simdgroup reduced through the free shuffle, and NO k-split or partial-sum
  reduction at all. jitllm k-splits and reduces partials; that is the structural
  difference to price.

  ★★★ **AND THE BYTES LEVER DOES NOT PAY ON tinyllama, MEASURED ON
  BOTH M4 TIERS. THE ENTRY BELOW NAMES IT AS THE METAL LEVER AND THAT IS NOW
  CONTRADICTED.** Q5_K went native (container v15, -12.65% of tinyllama's bytes,
  the same GGUF converted by the commit either side). Same-pass ABBA on a quiet
  M4, A/A self-control FIRST:

    | tier | A/A control | v15/v14 | IQR/med | bytes/token | GB/s | % of the 105.7 wall |
    |---|---|---|---|---|---|---|
    | Metal | -- | **1.0096** | 0.004 | 786.7M -> 687.3M | 93.9 -> 82.8 | 89% -> 78% |
    | CPU | 0.9970 (IQR 0.005) | **1.0058** | 0.002 | same | 43.6 -> 38.4 | 41% -> 36% |

  **12.65% fewer bytes bought 0.96% and 0.58% of tokens.** Both rows gate tightly
  -- the CPU A/A dispersion is 0.5% -- so this is not noise swallowing an effect.

  ★ **THE TWO QUANTITIES NOT AGREEING IS THE FINDING.** The v14 gemma-2b row is
  the counter-case and it is quoted above: -8.55% of bytes gave +8.2% of rate,
  "the two quantities agreeing -- measured separately, one from a file size and
  one from a stopwatch -- is what makes this more than a timing result". Here
  they disagree by a factor of thirteen, so tinyllama decode is NOT
  bandwidth-bound on either M4 tier, and the CPU number says it is not close: 41%
  of the wall, against the 59.7 GB/s this same box reaches on gemma-2b.

  ★★★ **AND THE PER-FORMAT COST IS NOW MEASURED AT EQUAL BYTES, WHICH CONVICTS
  THE TWO-PLANE UNPACK AND INDICTS THE Q5_K CHANGE ITSELF.**
  `nn.TestPerByteCost` (JITLLM_PERBYTE=1) prices every packed format on the host
  kernels at ~96 MiB EACH, so only the format varies. M4, v15:

    | format | shape | GB/s | % of fastest |
    |---|---|---|---|
    | Q8_0 | flat byte | **71.8** | 100% |
    | Q4_0 | nibble | 64.0 | 89% |
    | Q3_K | nibble + high mask | 51.4 | 72% |
    | Q4_K | nibble + min array | 46.5 | 65% |
    | Q5_K | **4+1 two-plane** | **30.0** | **42%** |
    | Q6_K | **4+2 two-plane** | **27.1** | **38%** |

  **★ THE TWO BOTTOM ROWS ARE SUPERSEDED: Q5_K IS 44.31 AND Q6_K 35.09 SINCE
  `ee73352` MERGED THE PLANES BEFORE THE DOT.** They are kept as the BEFORE arm
  of that measurement -- see the plane-combine entry below, which is what they
  motivated and what refutes them as a standing.

  **The two TWO-PLANE formats are the slowest by a factor of 2.4-2.6**, and the
  ordering follows the unpack's arithmetic exactly: a flat byte, then a nibble,
  then a nibble plus one more term, then a nibble plus a second PLANE.

  ★★ **AND Q5_K WAS ONLY TWO-PLANE BECAUSE THIS FILE MADE IT ONE, AN HOUR
  EARLIER. THE SAME BENCH ON THE COMMIT BEFORE:**

    | Q5_K | v14, 8-bit single plane | **44.7 GB/s** (62%) | 2209 B/row | 20.2M rows/s |
    | Q5_K | v15, 4+1 two-plane      | **30.0 GB/s** (42%) | 1440 B/row | 20.9M rows/s |

  Native packing bought 37.5% fewer payload bytes and cost **33% of the per-byte
  throughput**. They very nearly cancel: 3.4% in rows per second, which IS the
  0.58% measured end to end. The change stays -- it is a small throughput win and
  a real 12.65% container reduction, which matters for residency and upload --
  but **the bytes did not buy the rate, and framing it as a bandwidth win was
  wrong.**

  ★★ **AND THE MECHANISM IS COUNTABLE FROM THE EMITTER, NOT ONLY OBSERVABLE IN
  THE TIMING: A SECOND PLANE MEANS A SECOND DOT.** `EmitA64PackedMatVec` issues
  the primary plane's SDOTelem in its nibble loop and the secondary plane's in a
  SEPARATE pass, into the same accumulator -- `a.SDOTelem(VReg(g), t0, ...)`
  appears in both. Counting issues per element from the loop bounds:

    | format | sub | pw | hw | primary | secondary | SDOT per element |
    |---|---|---|---|---|---|---|
    | Q4_K | 32 | 4 | 0 | 2*4 = 8 | 0 | **0.25** |
    | Q6_K | 16 | 2 | 1 | 2*2 = 4 | 4 | **0.50** |

  **Q6_K issues TWICE the dot instructions per element**, which is the measured
  2.4x. The host splits a weight into two dot products because a dot is linear;
  the DEVICE does the opposite and this file already says so -- "both device
  kernels XOR the planes into the full weight first and center once". On amd64
  the split is deliberate and free (VPDPBUSD takes unsigned bytes and the hi mask
  is pre-shifted to 0x30, so both products land in one accumulator). On arm64
  SDOT is signed-by-signed and a combined weight is 0..63, which fits int8 --
  **so the planes could be ORed before the dot and one SDOT would serve both.**

  ★★★ **AND IT WAS BUILT (`ee73352`), AND IT IS THE LARGEST arm64
  KERNEL WIN SINCE THE PACKED MATVEC ITSELF.** One SDOT on the combined byte in
  place of two, plus `BIT16b` -- Bitwise Insert if True -- folding the merge's
  AND and ORR into one vector instruction. `nn.TestPerByteCost` on the M4 at
  ~96 MiB per format, alternated A/B/A/B in one session, 16 paired rounds over
  two passes:

    | format | was | now | ratio | % of the fastest format |
    |---|---|---|---|---|
    | **Q5_K** | 30.30 | **44.31** | **1.377** | 42% -> **62%** |
    | **Q6_K** | 27.06 | **35.09** | **1.271** | 38% -> **49%** |
    | Q8_0, Q4_0, Q3_K, Q4_K | -- | unchanged | 1.000-1.013 | A/A controls |

  The worst control moves 1.3% against effects of 27% and 38%, and the two hi
  formats go from 0.50 SDOT per element to Q4_K's 0.25. **It is arm64-only by
  construction**: amd64's split is already free, because VPDPBUSD takes unsigned
  bytes and both products land in one accumulator, so that path must not change.

  ★ **AND IT IS GATED ON `hw == 1` RATHER THAN ASSUMED.** One secondary word per
  (sub-block, row group) is addressable from a single cursor; both formats that
  have a secondary plane now are hw == 1 by arithmetic. A future hi format
  with hw > 1 keeps the two-pass path, which stays rather than being generalised
  speculatively.

  ★★ **SO THE CASE AGAINST NATIVE Q3_K IS WEAKER BUT NOT GONE, AND ITS
  ARITHMETIC HAS TO BE REDONE BEFORE ANYONE QUOTES IT.** The case read: Q3_K
  is a 4-bit single plane at 51.4 GB/s, native means the two-plane shape, "whose
  two members sit at 27-30 GB/s" -- so trading ~40% of the per-byte rate for 25%
  of the bytes loses. **Those members now sit at 35-44**, so the trade is ~25%
  of the rate for 25% of the bytes and the sign is no longer obvious. It is NOT
  thereby a win: a native Q3_K needs a 2-BIT primary plane, a third emitter
  shape nothing here has measured, and its per-byte cost is unknown rather than
  interpolatable from the 4-bit formats. **The bytes argument alone does not
  settle it, and the 2-bit primary's cost has never been measured** -- the number
  that would settle it is one `TestPerByteCost` row that does not exist yet.

  ★★ **THE ORIGINAL SUSPICION, KEPT BECAUSE THE MEASUREMENT ABOVE IS WHAT
  SETTLED IT.** tinyllama is q3_K_M, and Q3_K is
  the most arithmetic-heavy format to unpack -- a 4-bit primary plane plus a
  high-mask reconstruction per weight. If the binding constraint is that
  unpacking rather than the bus, then making Q3_K NATIVE (a 2-bit primary plane,
  "a third emitter shape") REMOVES BYTES THAT ARE NOT COSTING ANYTHING AND ADDS
  UNPACKING WORK TO THE THING THAT IS. It could be a net LOSS. That is a
  hypothesis, not a measurement -- but the measurement that would settle it is
  cheap (price the unpack against the load at a fixed byte count) and it should
  be taken BEFORE the emitter is written, not after.

  ★★★ **AND THE CONSTRAINT IS THE k-QUANT UNPACK, MEASURED DIRECTLY: THE SAME
  M4 MOVES 60 GB/s ON CHEAP-UNPACK FORMATS AND 38.5 ON k-QUANTS.** One binary,
  one pass, interleaved, CPU tier:

    | model | formats | tok/s | bytes/token | GB/s | % of 105.7 wall |
    |---|---|---|---|---|---|
    | gemma-2b | Q4_0 + Q8_0 | 35.89 | 1,676,656,640 | **60.2** | 57% |
    | tinyllama | Q3_K/Q4_K/Q5_K/Q6_K | 55.95 | 687,255,552 | **38.5** | 36% |

  **The bus does 60 GB/s on this box when the format is cheap to unpack.** So
  tinyllama at 38.5 is not traffic-limited, and the 1.56x between the two rows is
  the per-byte ARITHMETIC of the k-quant unpack -- a sub-block scale, a minimum
  array and a secondary plane per weight, against Q4_0's nibble and one scale.
  gemma's 35.89 tok/s also reproduces the container M4 board row (jitllm 35.6),
  so the harness agrees with the recorded standing.

  ★ **THIS IS WHY THE BYTES LEVER PAID 0.6%, AND IT REDIRECTS THE MAC WORK.**
  The remaining M4 deficit on a k-quant model is the UNPACK, not the traffic --
  and this file already names the unbuilt piece of exactly that: "Q4_K and Q5_K
  still cannot use the arm64 INTEGER fold (36.9% of a tinyllama prefill's MACs)
  ... Worth the remaining 5 -> 3 instructions per (sub-block, row, vector)."
  That is the lever with a measurement behind it. Native Q3_K packing is the one
  WITHOUT: it removes bytes that are not costing anything and adds work to the
  thing that is.

  What remains unexplained: the ~120 tok/s ceiling Metal holds on tinyllama
  regardless of traffic. The CPU tier's 36% is now accounted for; the Metal
  tier's is not, and the two may not have the same cause.

  ★★ **THE OLDER CLAIM, KEPT BECAUSE IT IS WHAT THE FILE SAID AND IT IS WHAT THE
  MEASUREMENT ABOVE CONTRADICTS.**
  tinyllama q3_K_M: llama.cpp's GGUF is **523.67 MiB** and the same model's
  container is **750.3 MiB** -- **+43%** -- because Q3_K packs to a 4-bit plane
  (the packed-payload table above: "Q3_K | 4 bits | +34.5% | 3 bits needs a
  2-bit PRIMARY plane, a third emitter shape"). Decode is bandwidth-bound, so
  that is a direct tax on every token, and jitllm reaches PARITY on this model
  only by running ~26% more bandwidth than llama.cpp does. **Native Q3_K packing
  is the Metal lever, not the command buffer.**

  The async commit stays: it is correct (tower NMSE unchanged at 2.802e-07),
  worth ~6% on the models over 600 MiB, and neutral on the small ones. It is
  kept for being right rather than for a number.

  1. **THE BLOCKING WAIT, WHICH IS NOW DONE AND WAS THE SMALLER HALF.** `metal_darwin.go`'s `Batch.Commit`
     is `commit` followed immediately by `waitUntilCompleted`, and `mtlSession.
     finish` calls it -- so EVERY session is a full GPU round trip with the CPU
     idle. MLX's `CommandEncoder::commit()` commits and immediately takes the
     next buffer; `waitUntilCompleted` lives only in `synchronize()`, called
     when the host actually needs a value. A decode token opens several sessions
     (`tier/layer.go` x3, `tier/tier.go`), so jitllm pays that latency several
     times a token.
  2. **RETAINED REFERENCES COST PER ENCODE.** A normal command buffer retains
     and releases every resource it references; at ~254 launches a token that is
     a lot of ARC traffic for nothing, since the arena outlives the buffer
     anyway. `commandBufferWithUnretainedReferences` is the same call without it
     -- and it is only SAFE because something else guarantees lifetime, which is
     what the residency sets are for.
  3. **RESIDENCY SETS BOUND THE RE-RESIDENCY COST.** macOS decides residency per
     SET, so MLX spreads allocations over several size-capped sets rather than
     one: when a set loses residency under pressure only that set is made
     resident again. llama.cpp announces the same thing (`use residency sets =
     true`).

  **What jitllm already has and must NOT be confused for mmap:**
  `newBufferWithBytesNoCopy` wraps host memory the engine ALREADY OWNS (the
  packed arena) so the GPU addresses the same physical pages -- Apple Silicon
  has one pool. It maps no file; `jlm` still reads with ReadAt/O_DIRECT into its
  own buffers, and `Ctx.Import` asserts the no-copy claim by checking
  `contents == p`. **The NO-mmap rule is intact and this is not a breach of it.**

## A device buffer that is never freed is invisible to every paging gate

- **★ A DEVICE BUFFER THAT IS NEVER FREED IS INVISIBLE TO EVERY PAGING GATE, AND
  WAS FOR THE LIFE OF THE TIER.** `func (b *fakeBuf) Free() {}` -- every fake
  buffer in `jit/gpu/tier`'s tests discarded it, so the gates asserted a block
  was PAGED OUT and never that its memory came BACK. Once fixed, it
  found four leaks immediately (Close never freeing g.layers, blockScratch
  having no free path at all, PrepLayer overwriting a resident block without
  freeing it, and a second Close hanging forever on a nil channel). On CUDA
  `cuCtxDestroy` reclaims the context, so none was a driver-level leak -- which
  is exactly why they survived, and stops being true on a unified-memory device.
  **The lesson generalises past this package: a counter that counts the
  ACQUISITION certifies nothing about the RELEASE.**

  ★ **AND A LEDGER ONLY COVERS THE PATH IT RUNS.** The text ledger was green
  while every VISION block leaked its two MLP biases: the upload walks six
  biases (Bq, Bk, Bv, Bo, BUp, BDown) and BOTH free walks covered four, so a
  text block -- which has no BUp or BDown -- could never see it. Found by
  `TestVisionTowerReturnsEveryDeviceBuffer` on its first run, 8 buffers over 4
  blocks, at the site `ReleaseLayers`' own comment warns about: *"two copies of
  the walk is how one of them acquires an eighth tensor and the other does
  not"*. The upload grew a vision case and neither free list did.

## A submission longer than the driver's fence timeout reads back half a tower

On the Iris Xe (vulkan:1) MiniCPM-V's tower, every block placed, read back a
residual with NMSE 0.99-1.43 against the host on the 45x23 overview, and
Phi-4-reasoning-vision's 56x28 photo 0.97; the 24x24 picture of the same gate
passed, and so did every other backend. Placing a prefix of the tower bisected
it to a count: 20 blocks at 5.1e-3, then 0.17, 0.36, 0.54, 0.73 and 1.03 for
21 through 25, a block's worth of residual missing per block past the 20th.

The tower went as one submission (`Stats.Submits` 1) and took 18.7 s, 0.93 s a
SigLIP block over 1035 patches. The kernel log named it, once per encode:

    Fence expiration time out i915-0000:00:02.0:model.test[3011359]:5e6!

`CONFIG_DRM_I915_FENCE_TIMEOUT` is 10000 ms on this kernel: a fence still
pending after it is signalled anyway, `vkQueueWaitIdle` returns success and the
read-back is whatever the blocks before the cut wrote. No error reaches the
engine, so nothing short of a comparison against the host can see it.

`devTier.submit` now cuts a range by run time as well as residency
(`tier/subbudget.go`): each submission takes as many blocks as
`Config.SubmitBudget` (1 s by default, half Windows' TDR and a tenth of i915's
fence timeout) holds at the per-block cost measured for that row count, rounded
to a power of two. A row count not yet measured, from 64 rows up, is probed with
one block; a decode token never is, so a model that fits still decodes a token
as one submission (`TestAModelThatFitsNeverPages`). An estimate rises at once
and falls by halves, so one fast submission cannot license a range a slow one
showed would overrun. On the RTX a tower block over the same picture is
milliseconds and the range stays whole after the probe.

The gates: `tier.TestASubmissionIsCutAtItsRunTimeBudget` (the cut, its count
and the residual against the uncut range, on the fake device; against a submit
that ignores the budget it reads one submission) and, on the hardware,
`model.TestMiniCPMVOnDeviceMatchesHost/vulkan:1` (5.1e-3 against 1.4) and
`TestPhi4RealMatchesTransformers/photo896x448/vulkan:1` (1.7e-4 against 0.97),
with `JITLLM_STEP_DEVICES` naming vulkan:1.

## A tower's device set was built for its largest grid, not the picture

HunyuanOCR's tower takes grids up to 65536 patches (`TowerConfig.MaxSeq`),
and the device built its non-causal set -- the scratch, the score rows, the
shared k/v pair, the batched twins -- for that many rows at placement
(`batchWidthFor` returned `MaxSeq`; `capped` left a vision plan uncapped, its
comment calling the patch count "fixed and known up front", which a
dynamic-resolution tower makes false). After the tower's first block:
`ScratchBytes` 5,669,846,908 against a 2.3 GB budget on the Iris Xe, where the
device's memory is host RAM ("7150082744 of 2309747506 used (pool host
RAM)"). It was allocated before the budget was asked: under an 8 GB cap the
process was OOM-killed; at 24 GB no tower block placed, every text block
relocated home past the overcharge, and the 5.5 GB stayed. On the 4 GB RTX the
allocation itself failed (`vkAllocateMemory`, `cuMemAlloc`), so no real
Qwen-family tower ran a block on any device here. Had it placed, every call
would have run all 65536 rows for a picture of 1024: kernels bake their row
count.

`jit/gpu/tier/visrows.go`: the set is built for `visCap` rows, which `capped`
hands every tower plan as its MaxSeq. Placement starts at `visStart` (1024)
or the tower's MaxSeq when smaller (a fixed-grid tower's picture).
`ReserveRows` (`nn.RowsReserver`), called by `State.encodeBlocks` before a
picture's blocks run and outside any submission, moves it to the picture's
rows rounded up to an eighth of their power of two (at most a quarter more
rows than the picture): up when the picture is larger, after asking the budget
for the build's bytes -- extrapolated from what the set it replaces measured
(`blockScratch.bytes`), its score planes priced exactly since they are capped
(`visionChunk`), the k/v pairs at the new rows -- and down when the picture
needs half the set or less. A picture the budget cannot hold is refused with
nothing allocated, and the State takes tower blocks home, highest first, until
it fits (`State.reserveTowerRows` over `relocateUntil`): it encodes on the
host, never out of memory. A first tower block the budget refuses after its
set was built takes the set with it, and so does the last tower block out
(`dropIdleTower`). The host's per-picture buffers (the patch gather, the
position rows, the resampled table's taps and rows) are sized by the picture
too.

Gates. `tier.TestATowerIsBuiltForItsPicture` (the fake device, a 65536-row
grid on a 256 MB card): placed at 1024 rows for 55,492,616 bytes, 164,315,156
at 3072 for a 3000-row picture, the 65536-row grid refused with no buffer
allocated, 19,374,088 at 512 for a 500-row picture, nothing live after the
release, and a block refused for its weights leaves nothing of its set.
Against the violations: the set built for the grid declines the first block
("out of memory ... 268435456 of 268435456 live"); no fit check before the
build allocates 32 buffers for the refused picture; no release on a refused
block leaves 55,492,616 bytes live. And `model.TestQwenVLRealMatchesTransformers`
now demands, per device asked for, that tower blocks are placed and every
placed one ran there (`State.vis.devBlocks`) with the set built for the
picture (`Stats.TowerRows`); where the text model, placed first as the CLI
places it, leaves no room on a 4 GB card, the tower is held to a card of its
own. On the laptop (text then tower; "alone" is the tower on a fresh card):

                        cuda:0                 vulkan:0 (RTX)         vulkan:1 (Iris Xe)
    Qwen2-VL-2B F16     text 18/28; alone 32/32  text 20/28; alone 32/32  7/32 + 28/28
    Qwen2.5-VL-3B       text 31/36; alone 32/32  text 36/36; alone 32/32  32/32 + 36/36
    Qwen3-VL-2B F16     text 12/28; alone 24/24  text 13/28; alone 24/24  9/24 + 28/28
    Qwen3.5-0.8B F16    5/12 + 24/24             12/12 + 24/24            12/12 + 24/24
    HunyuanOCR F16      21/27 + 22/24            27/27 + 24/24            27/27 + 24/24

every set built for 1024 rows, every end-to-end bound held. Against the
set built for the grid, HunyuanOCR places no tower block on any of the three
(`cuMemAlloc`/`vkAllocateMemory` out of memory, and on the Iris "7150082744 of
3124649984 used").

## The sm_80+ prompt: what the 0.07-0.33 rows were, and the staged GEMMs

A board taken with `scripts/vs-llamacpp.sh` on A10G, L4, L40S and A100
cards read jitllm's prefill at 0.07-0.33 of llama.cpp's (Llama-3.2-1B Q4_K_M
3994 against 17413 tok/s on the A10G), while the V100 board, taken with
`jitllm speed`, read 1.04-1.38. Two causes, counted on the RTX 3050 Ti
(sm_86), the one Ampere card at hand:

- **The harness timed a cold engine against a warm one.** Prefill mode ran
  `jitllm run -n 1` and took its "prompt" figure: the first prompt of a fresh
  process. `speed.go` already said that figure is not a prefill rate. A CPU
  profile of the first 512-row prompt in a process: 30 ms of ~200 growing the
  KV pool (`allocPages` -> `growKVLayer`), 43 ms outside the kernels in all,
  against 155-165 ms warm. llama-bench runs a warm-up and never times
  context creation. That fixed cost is a larger share the faster the card,
  so it compressed the big-card rows most. The jitllm arm is now
  `jitllm speed -p N -n 0 -r 1` (JITLLM_PP_COLD=1 keeps the old one). On the
  laptop, both arms warm against the old cold arm, same pass:

      Llama-3.2-1B Q4_K_M   cold 0.44 (REJECTED, IQR 0.114)   warm 0.59
      Qwen3-0.6B Q8_0       cold 0.38                         warm 0.75
      Qwen3.5-0.8B Q4_K_M   cold 0.77 (REJECTED, IQR 0.124)   warm 0.91 (REJECTED, IQR 0.163)

- **Every prompt matvec on sm_80/86/89 was MatVecMMA**, the unstaged int8
  kernel: each warp reads its weights and its activations from global memory
  inside the MMA loop and decodes every weight once per 32 tokens. sm_70 had
  GemmVolta (shared-memory staged, binary16) and the newer cards never
  reached it, since voltaMV runs only where the integer instruction is
  missing (rows.go `voltaMV`'s `!(g.mmaOff || g.NoMMA)`). Per-launch device
  time of a warm 512-row Llama-3.2-1B prompt (nsys sees no kernels through
  goffi and ncu needs counters this box withholds, so
  `backend.SetCUDAKernelTiming` now times launches outside a capture): 105 of
  165 ms in the FFN's three projections, 26 in the attention (FMA
  accumulate 15, m16n8k16 scores 5.6, softmax 5.7), 12 in the activation
  conversions and SiLU.

What was built, every kernel generated:

- **GemmVolta on m16n8** (`VoltaTile.F16K`: 16 from sm_80, 8 on sm_75): the
  same staging, lane (g, q) reading chunk q of rows g and g+8 and token g as
  one 16-byte load each. The k order inside a chunk needs no permutation:
  word 4q+2s+h of A meets the same word of B.
- **GemmInt8**: MatVecMMA's arithmetic staged as llama.cpp's MMQ stages it:
  a trip of 32 elements decoded once a workgroup to signed bytes with
  float32 scales and minimums per sub-block, the int8 activations with their
  scales and sums beside them, a lane's two words one 8-byte load. Bit for
  bit the dp4a matvec unsplit. It was the default on sm_80 and later until
  the L4 and A100 rows (below) put the binary16 GEMM ahead; it stays behind
  JITLLM_GPU_GEMM_INT8.
- **PagedAttnAccMMA**: the weighted sum of V on m16n8k8 behind the m16n8k16
  scores: 6.5 ms where the FMA tiles read 15.
- **The tile and split by the card** (`fillTile`, `fillSplit`): about one
  workgroup an SM, sm_70's measured bar, so a 1024-row projection does not
  leave a 142-SM card with one workgroup on every second SM.

What the laptop could and could not show. On the RTX 3050 Ti the three GEMMs
read 8-15 TFLOPS in isolation (`TestGemmF16Speed`, 512 rows), interleaved and
noisy: the binary16 GEMM sits at GA107's float32-accumulate ceiling (~15
TFLOPS at 1.5 GHz, half the binary16-accumulate rate on a GeForce part), and
the int8 GEMM and MatVecMMA land in the same band, so on this card the
change of kernel is inside the noise of the warm board (Llama-3.2-1B 0.59
before, 0.63 after with the binary16 GEMM). Its ceilings are not the
A100's (binary16 with float32 sums at the full rate, int8 at twice it) or
an Ada card's (int8 at four times the float32-accumulate rate), and the rows
that matter have to be taken there.

### The L4 and A100 rows of the staged GEMMs

From 8dad3b3e, CUDA against llama.cpp b10825, n=20 paired, both arms warm:
Llama-3.2-1B Q4_K_M prefill read 0.67 on an L4 and 0.78 on an A100 with
GemmInt8, 0.75 and 0.91 with the m16n8 binary16 GEMM
(JITLLM_GPU_NO_GEMM_INT8, as the knob was then); Qwen3.5-0.8B 0.89 and 0.96
on the binary16 GEMM. The cold arm (the old harness) read 0.22 and 0.24. The
binary16 GEMM is the default from sm_80 since; sm_75 has no int8 m16n8k32
and takes the binary16 GEMM's k8 form.

### The three items left after the GEMMs

Per-launch device time of a warm 512-row Llama-3.2-1B Q4_K_M prompt on the
RTX 3050 Ti, the arms interleaved twice in one sitting (before: NoFlashPrefill,
NoOperandFuse, WithKVPrewarm(false); the binary16 arms NoGemmInt8 as well).
The card swings tens of percent between passes, so these are counts of where
the time sits, not a rate:

    attention (int8 arm)     scores 6.6-8.6 + softmax 4.3-5.5 + accumulate 6.5-9.7 ms
                             -> FlashPrefill80 2.9-4.9 ms
    operands (int8 arm)      ActMul 4.9-8.3 + Quantize (33 launches) 3.9-6.9 ms
                             -> QuantizeAct 5.8-8.3 + Quantize (17) 0.7-0.8 ms
    operands (binary16 arm)  ActMul 6.2-6.7 + ActF16T (72) 6.8-7.4 + RMSNorm 2.7-4.6 ms
                             -> ActMulF16T 4.7-6.6 + RMSNormF16TRows 3.2 + ActF16T (24) 1.1 ms
    first prompt (int8 arm)  171 ms against a warm 136-137
                             -> 113-116 against a warm 119-130

- **FlashPrefill80** is FlashPrefill70 in FlashAttention-2's layout, so the
  score fragment is the next product's A fragment and P never reaches shared
  memory. A head past 128 (gemma-2b's 256) splits its dims over two warps of
  a row block, its queries in shared memory and its tiles 16 keys; past 256
  it is declined by name.
- **The fused operands.** A first form gave each thread a whole
  (token, sub-block) as ActF16T does. It was slower than the two launches it
  replaced: the norm's second phase had 64 of 1024 threads busy per row, and
  each thread ran a sub-block's exponentials 16 to 32 deep. One output word a
  thread (`f16WordElems`, a word's two elements by arithmetic) is the form
  that ships.
- **The KV prewarm** moves the pool's growth into placement: 100-200 ms more
  placement on the laptop, and the first prompt reads as warm.

### sm_75's prompt attention on m16n8k8

FlashPrefill80 is m16n8k16, which sm_75 does not have, so a T4's prompt
attention ran FlashPrefill70 on sm_70's m8n8k4, which Turing executes well
under its m16n8 rate. The m16n8k16 fragments are the m16n8k8 ones side by
side: A's four registers are rows g and g+8 at k pair q, then at k pair q+4,
and B's two are k pair q, then q+4. So each k16 product is two k8 ones, and
nothing is staged or loaded differently (kernels.FlashPrefill80K8, `_k8` in
the name). The tier takes it where f16GemmK is 8 and the row block divides
the chunk. Otherwise sm_70's pair still serves.

On a T4, per-kernel device time of a warm 512-token prompt:

    Llama-3.1-8B Q4_K_M   flashprefill70paged_d128  20.1 ms  ->  flashprefill80paged_d128_k8  13.9 ms
    Llama-3.2-1B Q4_K_M   flashprefill70paged_d64    4.1 ms  ->  flashprefill80paged_d64_k8    2.7 ms

The prompt's wall rate did not resolve it. The pairs read 0.98-1.04 on the
8B and 0.91-1.07 on the 1B, with the card going from 1590 to 945 MHz within
the job; the A/A pairs drifted as far. llama.cpp's flash attention on the
same card ran 13.5 ms for the 8B prompt. The kernel gate
(TestPagedFlashPrefill80) runs the k8 form on every card with m16n8k8,
sm_75 and sm_86 alike, and fails with B's two registers swapped.

### A warm 514-token prompt, kernel by kernel, against llama.cpp on a T4

`jitllm speed` prints JITLLM_CUDA_KERNEL_TIMING's split for the timed prompt
(an event pair around each launch: a prompt chunk is not captured), and
`scripts/kshim.c` under KSHIM_DIRECT=1 takes the same of llama.cpp b10825
CUDA, which runs a batch past one row without a graph. Both are per-launch
device time with the events' own cost inside it, so the sums run above the
wall; they say where the time sits, not a rate. One T4 (sm_75), one sitting,
llama.cpp's totals halved for its warm-up prompt:

    Meta-Llama-3.1-8B Q4_K_M, us a prompt       jitllm      llama.cpp
    GEMMs over the 512-row chunk               525,100     442,900 (mmq q4_k, q6_k)
      + operand quantize / stream-k fixup            --      21,700
    the 2-row tail's matvecs                    58,400      19,600 (mmvq)
    attention (flash, + the tail's staged)      34,100      13,500
    norms, SiLU, rotary, adds, conversions      35,100      29,700
    head                                         1,700       1,800

    Llama-3.2-1B Q4_K_M                          jitllm      llama.cpp
    GEMMs over the 512-row chunk                56,100      90,800
      + operand quantize / stream-k fixup            --       8,600
    the 2-row tail's matvecs                    10,700       5,100
    attention                                    7,400       5,300
    norms, SiLU, rotary, adds, conversions       9,000      11,700

Rates beside it (`jitllm speed -p 514 -n 0`, llama-bench `-p 514`): 8B
705-887 against 1056-1085 tok/s, the card slowing between passes; 1B
4.5-7.2k against 4.2-5.9k. The three gaps on the 8B, largest first: the
staged GEMM against MMQ's int8 (about 60 ms net of llama.cpp's operand
passes), the 2-row tail (39 ms: a GEMM at the grain over 30 rows of padding
where llama.cpp runs a matvec), and the attention (20 ms: FlashPrefill70's
m8n8k4, which sm_75 runs well under its m16n8 rate, and the tail's staged
three kernels). The tail is the one the prompt length decides: on a V100 a
514-token prompt took 30% longer than a 512-token one (1B, 12.9k against
16.8k tok/s), and on the A100 board `jitllm speed -p 512` read 28.8k where the
harness's 514-token row read 14.4k.

## llama.cpp's MMQ against the staged GEMMs, and the int8 instruction on sm_75

The T4 split above put the largest prompt gap on the 8B's GEMMs (525 ms
against MMQ's 443 plus 22 of operand passes). What follows is what
llama.cpp b10825's CUDA prompt GEMM does (read from
`ggml/src/ggml-cuda`: `ggml-cuda.cu`, `mmq.cu`, `mmq.cuh`,
`mmq-config-ampere.cuh`, `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh`,
`mma.cuh`, `quantize.cu`), set beside jitllm's generated kernels, and then
the one lever it suggested first, built and measured.

### What llama.cpp does

- **Dispatch** (`ggml_cuda_mul_mat`): an f32 activation into a quantized
  weight takes MMVF / MMF / MMVQ for a few columns, then
  `ggml_cuda_should_use_mmq`, then cuBLAS. MMQ is taken for every k-quant
  and legacy quant whenever the card has 48 KiB of shared memory a block
  and `turing_mma_available` (sm_75 on) -- unconditionally there, whatever
  the batch. On an NVIDIA card without int8 mma it is taken below 64
  columns or where there are no binary16 tensor cores; elsewhere cuBLAS.
  So a T4, an A100 and an L4 all run every Q4_K/Q6_K prompt GEMM on MMQ;
  a V100 runs it through cuBLAS above 64 columns (sm_70 has no integer mma).
- **The tile** (`mmq-config-ampere.cuh`, used from sm_70 to Ada): 256
  threads (8 warps), occupancy target 1, I = 128 weight rows by J columns,
  J chosen per call as the multiple of 8 up to 128 that gives the fewest
  column tiles (512 tokens: J = 128, four column tiles; 514: J = 128 and a
  fifth tile with two live columns -- no padding to a grain). A rows count
  not a multiple of 128 takes the `fallback` variant that clamps the row.
- **K per stage**: one VRAM tile is 256 elements (`MMQ_ITER_K`, a whole
  k-quant super-block) a trip; the shared tile holds it as two 128-element
  halves of the activations, so a trip is load A (256 k), load B half,
  sync, mma, sync, load B half, sync, mma, sync. **No double buffering**
  and no cp.async: latency is hidden by eight warps, not by a pipeline.
- **The weights in shared memory** (mma layout): Q4_K's nibbles are stored
  unsigned (`qs & 0x0F0F0F0F`, `qs >> 4 & ...`), the row stride padded to
  4 mod 8 words against bank conflicts; beside them a half2 per 32 values,
  `(d*sc, -dmin*m)`. Q6_K keeps signed bytes with an int8 scale per 16.
  The minimum is never applied per element: it multiplies the activation
  block's precomputed sum (`dmA.y * dsB.y`).
- **The activations**: `quantize_mmq_q8_1` runs once per GEMM call, in its
  own launch, over the whole src1, into `block_q8_1_mmq` (128 int8 values
  plus four half2 (d, sum) -- 144 bytes, laid out so a tile copy to shared
  memory is a straight memcpy). The layout of d/sum differs per weight type
  (D4, DS4, D2S6).
- **The product**: `mma.sync m16n8k32 s8` on sm_80, and on sm_75 the same
  tile as **four `m8n8k16`** (`mma.cuh`): A's fragment via `ldmatrix`
  (`x2`/`x4`), B's by plain loads ("faster than load_ldmatrix"). Each
  32-element product is folded to float at once with the A and B scales
  (`sum += dmA.x*dsB.x*C + dmA.y*dsB.y`); a warp holds 16 or 32 rows
  (`rows_per_warp`) of the 128.
- **Stream-K**: every Q4_K/Q6_K config sets `stream_k`. The grid is one
  block an SM, each walking a contiguous range of (tile, k-trip) space; a
  block that ends mid-tile writes a partial to a scratch, and
  `mul_mat_q_stream_k_fixup` adds them. Where whole tiles already fill the
  waves to 90% the grid is the tile count and no fixup runs. The 21.7 ms of
  "quantize / fixup" on the T4 8B prompt is these two passes.

### Against jitllm's generated kernels

| | llama.cpp MMQ (sm_75, sm_80+) | jitllm GemmVolta F16K (default from sm_75) | jitllm GemmInt8 (knob) |
|---|---|---|---|
| operands | int8 x int8 -> int32 | binary16 x binary16 -> float32 | int8 x int8 -> int32 |
| instruction | m16n8k32 / 4x m8n8k16 on sm_75 | m16n8k16 (k8 on sm_75) | m16n8k32, m16n8k16 |
| workgroup | 8 warps, 128 rows x up to 128 tokens | 4 warps (2x2), 128 x 64 (MT4 NT4) | as F16K's tiles |
| k a stage | 256 (a super-block) | 32 | 32 |
| syncs a 256 of k | 4 | 16 | 16 |
| pipelining | none; occupancy | next trip's globals in registers | the same |
| fragment loads | ldmatrix for A | 8/16-byte shared loads | 8-byte loads (`int8Pos`) |
| weight decode | nibbles unsigned, scale and minimum as half2 per 32 | dequantized to binary16, scales folded | signed bytes, f32 scale and minimum a sub-block |
| activation operand | its own launch per GEMM (q8_1_mmq) | written by the norm / activation (fused) | quantized in the activation's launch |
| grid | stream-K over SMs + fixup | tile per (128 x 64) block, k-split + Reduce to about one per SM | the same |
| ragged columns | J per call, a partial last tile | padded to the grain | the same |

The two engines agree on the arithmetic and differ in shape: MMQ amortises
a sync and a scale fold over 256 elements of k and 8 warps, jitllm over 32
and 4 and pays it back with register double-buffering.

### The int8 instruction on sm_75, built and measured

MMQ's T4 path is int8 on m8n8k16, which jitllm did not lower (the IR's
`MMAS8` shapes were m16n8, sm_80's), so on a T4 every prompt GEMM was the
binary16 GEMM at k8 and `mmaOff` was set. The m16n8 int8 fragments are
m8n8k16's registers side by side (A's word r is row g+8(r&1), k 16(r>>1)+4q;
D's pair p is row g+8p), so the lowering is two or four m8n8k16 per op.
It was built (branch `gemm-int8-sm75`: `ptx/lower.go`, a ptxas gate at
sm_75 and sm_86, a device gate that runs the sm_75 lowering on an sm_86
card through the driver's PTX JIT and holds MatVecMMA and GemmInt8 bit for
bit to the dp4a arm -- a swapped fragment index fails it) and passed every
gate on a T4. It is not shipped. On a Tesla T4 (sm_75, 1590 MHz, a round at
1410-1500 MHz in two of the passes, ratios unchanged), pp512 and tg64 a
call, median of 5, six rounds rotating the order, two passes, A/A first
(1B 0.994 / 1.017, 8B 1.012 / 0.996):

    against origin/main                     Llama-3.2-1B Q4_K_M     Llama-3.1-8B Q4_K_M
    lowering on (default config)  pp512     0.658 / 0.662           0.864 / 0.868
                                  tg64      1.009 / 1.014           1.001 / 1.001
    + JITLLM_GPU_GEMM_INT8        pp512     0.500 / 0.501           0.753 / 0.754
    origin/main absolute                    9450 pp, 238 tg tok/s   1013 pp, 48.1 tg tok/s

The per-kernel split of one warm prompt says why, in two parts:

- **A device with an integer mma is a different tier.** With the lowering
  in, MatVecMMA compiles, `mmaOff` stays false, and that moves more than
  the GEMMs: the prompt's attention left FlashPrefill70 (8B: 23.8 ms) for
  the three staged kernels (scores 62.9, accumulate 48.4, softmax 6.8 ms),
  and the chunk grain went from 8 to 32 rows. That is the 0.66 / 0.87.
- **The instruction is not the gap.** The GEMMs alone: 1B 61.2 ms binary16
  against 68.1 ms int8; 8B 426.5 against 532.3 ms int8, plus its operands
  (`quantizeact` 16.0 + `rmsnormquantw` 10.5 against the binary16 arm's
  21.3 of norms and activations). GemmInt8 at k = 32 a stage on m8n8k16 is
  slower than the binary16 GEMM at k8, while MMQ on the same instruction is
  at 443 ms. What MMQ has and both jitllm GEMMs lack is its shape: a
  256-deep stage, 8 warps a block, ldmatrix, and the scale folded once per
  32 against a stage's worth of products.

On a V100 (sm_70, 1312 MHz locked; no integer mma, so the lowering is not
reached and the knob is inert) the same three arms read pp 0.985-1.018 and
tg 0.999-1.002 on the 8B; the 1B's tg read 1.016-1.024 on both branch arms
against A/A 0.997-0.998 -- the PTX the V100 runs is identical by
construction, so that is a binary or placement effect, unresolved.

On an A100-40GB (Modal, one job, 597 s), `scripts/vs-llamacpp.sh` prefill
514 rows, 6 rounds (n = 12), llama.cpp b10825 CUDA: the A/A self-control
read 0.997 and was REJECTED (IQR/median 0.136), so these are indicative:

    Llama-3.1-8B Q4_K_M   binary16 GEMM 0.981 (IQR 0.072)   GemmInt8 0.680 (0.068)
    Llama-3.2-1B Q4_K_M   binary16 GEMM 0.903 (IQR 0.033)   GemmInt8 0.669 (0.029)

The binary16 default stands on every card measured. The work list the
study leaves, largest expected first: a 256-deep k stage with 8 warps a
block for the staged GEMMs (both operand types), ldmatrix fragment loads,
stream-K in place of the k-split's Reduce, and J chosen per call so a
514-row prompt is not padded. Only after those does the int8 instruction
have a kernel worth putting it in; and putting it in on sm_75 must not
flip `mmaOff`'s other consequences (the attention and the grain) with it.

### The deep stage, built and measured

The first two items of that list were built: the m16n8 binary16 GEMM and the
int8 GEMM both stage 64 or 128 elements of k a trip on eight warps, in one
buffer or two, and the int8 GEMM multiplies on sm_75's m8n8k16 and reads its
weight fragments with ldmatrix.

- **GemmVolta's m16n8 tile** (`VoltaTile.KB` past one 32-element step,
  `Single`): a wide trip needs its own swizzle (`f16Swizzle`, rows g and g+1
  of a fragment load four chunks apart), and one buffer halves the shared
  memory so a 128-deep trip fits 48 KiB. **A 256-deep binary16 stage cannot
  exist on a T4**: 128 weight rows of 256 binary16 elements are 64 KiB, the
  whole of what sm_75 gives a block, before any activation. MMQ fits it
  because its operand is int8. Gate: `TestGemmF16DeepStageMatchesTheReference`.
- **GemmInt8** (`Int8Tile.D`, `Single`, `M8`, `LdM`): rows padded so a
  fragment phase meets four bank groups; m8n8k16 asked for by name
  (`ir.MMAS8Turing`), so the m16n8 integer shapes still refuse on sm_75 and
  `mmaOff` keeps its answer (the attention and the grain do not move); `LdM`
  stages the weights in natural k order on rows of an odd count of 16-byte
  chunks and reads an m-tile's A fragments with one `ldmatrix.x4` a step
  (`ir.Builder.LdMatrix`, PTX only, the other lowerers refuse it by name).
  Gate: `TestMatVecMMAMatchesDot4` holds every form bit for bit to the dp4a
  arm, on an sm_75 card through the m8n8k16 forms alone; a swapped k-half
  and a dropped step offset each fail it.

Tesla T4 (sm_75, 70 W, power-capped under load at 800-1250 MHz), a rented card,
`TestGemmDeepSweep` / `TestGemmMMQSweep`: Llama-3.1-8B Q4_K_M's
projections at 512 rows, every arm interleaved in each of 24 rounds of 5
launches, the median per-round ratio to the shipping tile (m4n4 w2x2, 32 of
k a trip), two passes. The A/A arm read 0.99-1.005 on the small shapes and
1.03-1.07 (first job) and 1.09-1.21 (second and third) on the three large
ones: a slot bias of the power cap that a dispersion gate does not see, so a
large shape's ratio is read against its own A/A.

    deep binary16 tile (8 warps, 64 a trip, one buffer), pass 1 / pass 2
    q/o  4096x4096   0.928 / 0.954     (the shipping tile split in two: 0.937 / 0.934)
    k    1024x4096   0.931 / 0.946
    v    1024x4096   Q6_K: 0.906 / 0.920 on 32 a trip, 8 warps, two buffers; the 64 tile 1.00
    gate/up 14336x4096   0.996 / 1.012      A/A 1.014 / 1.033
    down 4096x14336 Q4_K 0.910 / 0.919      A/A 1.035 / 1.035
    down 4096x14336 Q6_K 1.015 / 1.017      A/A 1.066 / 1.058

So no tile wins everywhere, and the tier measures it: `tuneGemm` times the
deep tile against `f16Tiles`' pick per shape on the resident weights at the
prompt width (after the split tuner, cached per device, driver and
executable like it) and `f16Gemm` takes the deep tile where it read under
0.98. Its verdicts on the T4, three jobs:

    Llama-3.2-1B   Q4_K 2048x2048 0.915-0.932 deep   Q4_K 8192x2048 0.929-0.941 deep   Q4_K 2048x8192 0.912-0.919 deep
                   Q4_K 512x2048  1.331-1.341        Q6_K 512x2048  1.460-1.505        Q6_K 2048x8192 1.022-1.040
    Llama-3.1-8B   Q4_K 4096x4096 0.938-0.957 deep   Q4_K 1024x4096 0.924-0.927 deep   Q4_K 14336x4096 0.953-0.958 deep
                   Q4_K 4096x14336 0.966-0.986 (deep twice of three)   Q6_K 1024x4096 1.034-1.040   Q6_K 4096x14336 1.109-1.137

**The int8 instruction is still not the gap, and the kernel's shape is not
enough to close it.** Best int8 arm against the shipping binary16 tile, T4,
same harness, pass 1 / pass 2 (registers per thread, resident workgroups):

    shape        int8, plain fragment loads          int8, ldmatrix A
    q/o          1.173 / 1.136  (168 regs, 3)        1.155 / 1.122  (168, 3)
    k            1.359 / 1.322  (247, 1)             1.294 / 1.281  (252, 1)
    v  Q6_K      1.676 / 1.689  (148, 2)             1.361 / 1.383  (128, 4)
    gate/up      1.193 / 1.191  (245, 1)  A/A 1.135  1.430 / 1.426  (255 + 24 B spill)
    down Q4_K    1.148 / 1.142  (245, 1)  A/A 1.122  1.249 / 1.252  (129, 1)
    down Q6_K    1.380 / 1.384  (174, 2)  A/A 1.100  1.114 / 1.122  (168, 3)

The operand costs less on the int8 side (`Quantize` 58-171 us against
`ActF16T` 77-274 us at these widths, both fused into the norm or activation
in the engine), which does not cover a GEMM 15-70% slower. ldmatrix pays on
Q6_K (two 16-element sub-blocks a step, whose plain loads were the most
fragmented) and little elsewhere; tiles held under 168 registers (2x4 and
2x2 warp tiles, 128-148 registers, 2-4 resident) were not faster than the
245-register 4x4 one. What remains between this kernel and MMQ's 443 ms is
the per-sub-block float fold: four conversions, multiplies and fused adds a
fragment for every 32 (16 on Q6_K) elements, where MMQ folds with its
scales held as half2 and the minimum through the activation block's
precomputed sum.

End to end, `scripts/vs-llamacpp.sh` prefill 514 on the T4, CUDA both
arms, every binary's tune cache warmed first (a cold one tunes inside its
row: an earlier job that warmed only the branch read main slower for that
reason alone), ratio = A over B in tok/s:

    Llama-3.1-8B Q4_K_M (12 rounds, n=24)   pass 1             pass 2
    A/A (the branch both arms)              0.998 (IQR 0.011)
    main / branch (tuned)                   0.809 (0.016)      0.808 (0.013)
    main / branch (deep forced everywhere)  0.785 (0.010)      0.779 (0.015)
    branch (tuned) / llama.cpp b10825       1.094 (0.010)      1.091 (0.019)
    absolute, warm speed -p 514             main 927-940 tok/s, branch 1141-1166 tok/s

Every Llama-3.2-1B row (20 rounds, n=40) was REJECTED, the A/A included
(IQR/median 0.11-0.23): an 80 ms prompt on a power-capped T4 does not
resolve in this harness. The 8B gain is larger than the GEMM trial's 4-8% a
shape; the warm per-kernel split of one prompt in the same jobs read the
prompt's GEMMs at 546 ms on main against 461 tuned and 393 forced deep, but
each arm there is a separate process at its own clock, so it is not an A/B.
Forced deep beat the tuned choice end to end although the tuner keeps the
shipping tile on both Q6_K shapes: a trial of one GEMM alone is not the
prompt's mix. That is the next measurement for this tuner.

### `run` against `speed` on the A100

The same job timed `jitllm run -n 128` token by token (a stamp on every
stdout chunk) beside `jitllm speed -p 16 -n 128`, two passes each:

    Llama-3.2-1B   speed tg128 630-644 tok/s (1.55-1.59 ms)   run gaps p50 1.80 / 1.58 ms, mean 2.58 / 1.73, max 92.9 / 10.7 ms
    Llama-3.1-8B   speed tg128 169-172 tok/s (5.82-5.92 ms)   run gaps p50 5.89 / 5.82 ms, mean 7.15 / 6.11, max 153 / 24 ms

`run`'s steady token is `speed`'s; its lower rate is a few long stalls
(10-150 ms, larger on the first pass), which a mean over 128 tokens spreads
to 0.2-1.3 ms a token. Where the stalls come from (a capture, the
collector, stdout) is not yet known.

### A prompt's few-row tail is a ragged step, not a padded chunk

The device runs a prompt chunk at its grain (32 rows, 64 where
FlashPrefill80 runs), so the 2 rows a 514-token prompt leaves after its
512-row chunk ran every batched kernel at 32 or 64 rows: GEMMs dequantizing
all their weights for 2 live rows, the staged attention over the padding.
On a V100 that tail ran 6.7 ms of kernels for Llama-3.2-1B, against about
1.6 ms for a decode token, and the 514-token prompt took 30% longer than a
512-token one. llama.cpp runs a batch that small through its matvec.

A chunk of 2 to 4 rows over the whole model, every block and the head on a
device that runs ragged rows, now goes as one ragged step of its rows
(engine/model/tailrows.go): nn.RowsDevice with the rows' positions as their
slots, which the tier runs on decode's matvec carrying every live row in one
pass over the weights (groupMV, `_t2of8` in a kernel name), each row
attending to its own history. The head is the prompt's tail call, as for any
chunk that did not fold it. The same 2 rows then run 2.2 ms of kernels on the
V100. Not taken for a hybrid, a speculation step, a bidirectional run or a
split placement, which keep the chunk; a refusal falls back to it.
`model.WithTailRows(false)` (`JITLLM_TAIL_ROWS=0`) is the control arm.

A first form chose by measurement: the fastest decode call, a short chunk and
the ragged step, each kept on the Model. A first run includes its compiles
and captures, so it could not tell the shapes apart. On the V100 the step's
first run (75 ms, compiling) left it measured slower than the chunk for the
rest of the process. The step reads every weight once where the chunk reads
them at the grain, so it is the rule.

Rows, same binary with the step on and off (`JITLLM_TAIL_ROWS=0`),
interleaved, medians of 7-prompt `jitllm speed -n 0` runs, A/A first:

    T4, Llama-3.1-8B Q4_K_M   A/A 0.97-1.03    pp514 off 847-870, on 913-922 tok/s (+6.4%)
                                               pp516 off 856-874, on 891-920 (+4.5%)
                                               pp2   off 27.2-30.0, on 60.5-60.9 (2.06x)
    T4, Llama-3.2-1B Q4_K_M   A/A 0.72-1.05    pp514 off 6.07-6.50k, on 6.14-6.64k (unresolved)
                                               pp516 off 5.98-6.19k, on 6.25-6.69k (+4%)
                                               pp2   off 103-106, on 205-224 (2.07x)
    V100, Llama-3.2-1B        (host loaded)    pp514 off 11.3-12.6k, on 13.1-13.9k (+7%)
                                               pp2   off 223-245, on 280-309 (+25%)

llama-bench on the same T4: 8B pp514 974, pp2 46; 1B pp514 5.6k, pp2 364.
A 512-token prompt and decode are unchanged by construction (no chunk of 2-4
rows) and read so: T4 tg128 230-232 against 230-232 tok/s on the 1B, 45.6-47.8
against 45.9-47.1 on the 8B.

What is left of the tail's cost is the host's: the step is recorded and
replayed like a decode step, and a recording is the session's, so each new
session captures it once (about 1 ms on the V100), and the head runs as its
own call.

## Measurements once cited in jit/gpu/tier's comments

The tier's comments state the engineering reason and the class of card; the
measurements they used to carry, with the machine that produced them, are
here verbatim.

- `groupTokMax` (`jit/gpu/tier/rows.go`): "Llama-3.1-8B Q4_K_M on a V100,
  tok/s together, groupMV against the tiled twin: 2 sessions 203 / 146, 4
  307 / 270, 8 363 / 488."
- `dropSession` (`jit/gpu/tier/layer.go`), on why the scratch's bound is not
  reset: "(measured: pp512 on a V100 17.4k tok/s on the first prompt of a
  process and 13.7-14.0k on every one after)". The bisection is "The prompt
  lost 22% on the V100, and none of it was a kernel" above.
- `promptBuf` (`engine/model/prefill.go`): "A request is a fresh State, and
  allocating these per request cost a V100 prompt of 512 rows 6.6 ms of its
  33.9 (a fresh 4 MiB heap span faults in page by page): the device ran at
  19.5k tok/s and the prompt measured 15.1k."
- The comments that now name a class of card were measured on these: the
  Volta-class figures (Llama-3.2-1B's 12 MB of batched-twin scratch in
  `prepLayer`, the 0.9 ms `prepBatch` walk, the `ragMV` Tok/Rowt sweep,
  `voltaMV`'s ~2.3x over the dp4a twin and its ~1280-warp split, the
  `voltaTiles` 2x2 grid, `kb0Split`'s 80 SMs, `volta70AccMT`'s halved
  accumulate, `PagedPrefill70`'s selection check, `keepsSlot`'s
  `TestPrefillStreamsHostBlocks` case) on a V100; the integrated-GPU figures
  (`restride`'s 65535-group dispatch limit, `visrows.go`'s 2.3 GB budget,
  `subbudget.go`'s 0.93 s SigLIP block under i915) on an Iris Xe; the Apple
  Silicon figures (`flashTileAttn`'s 32-row blocking, `tileGemms`' 64x64 lead)
  on an M4; and the k-split's 30720 resident slots (`tier.go`, 20 SMs x 1536)
  are an RTX 3050 Ti's.

## Design notes moved out of source comments

Each entry below is a comment that stood in the named source file,
verbatim. The source keeps the contract, the layout and the invariants
and points here for the derivation, the transcript and the reasons.

### jit/gpu/kernels/ropetab_const.go

The rotary table's constant block and its layout: the {cos, sin} pair per
rotary pair, formed by a kernel rather than math.Cos/math.Sin in Go.

One copy of the layout for four tiers (AVX2, SSE, NEON and the device): the
layout is the ABI between the emitters and nn. It lives here rather than in
jit/cpu because jit/cpu already imports this package, so a device kernel
importing jit/cpu would close a cycle; jit/cpu/ropetab_const.go re-exports
every name.

The precision lives in the constants. The angle is pos*g radians with pos up
to 131072 -- thousands of radians, where an f32 ulp is 5e-4. So the position
is decomposed into four 7-bit digits and each digit's contribution is folded
mod 4 quarter-turns at setup, in float64:

	pos = d0 + 128*d1 + 128^2*d2 + 128^3*d3          (d_i in [0,128))
	v   = pos*g*2/pi = sum_i d_i * U_i   (mod 4),  U_i = fmod(g*128^i*2/pi, 4)

and the kernel needs only frac(v) and floor(v) mod 4, which survive because
the reduction mod 4 distributes over the sum.

Each U_i is split so its digit product is exact. H_i is U_i rounded to a
multiple of 2^-12, so it needs at most 15 significant bits; a digit needs 7;
7+15 = 22 < 24, so d_i*H_i is exact in float32. The four products sum to at
most 2032 (8323072 units of 2^-12, under 2^24), so the sum and its mod-4
fold are exact too. The only rounded quantity is the residual plane
L_i = U_i - H_i, whose digit product is under 0.0156, giving ~1e-9
quarter-turns of error (~1e-7 radians at every position; see
engine/nn/ropetable.go).

The per-model block is PLANE-MAJOR -- all of H0, then all of H1, ... -- so a
kernel reads each plane as one contiguous vector and the four digit
broadcasts stay in registers for the whole table:

	word 0*S .. 1*S-1   H0      the 2^-12 head of U_0
	word 1*S .. 2*S-1   H1
	word 2*S .. 3*S-1   H2
	word 3*S .. 4*S-1   H3
	word 4*S .. 5*S-1   L0      U_0 - H_0, the residual
	word 5*S .. 6*S-1   L1
	word 6*S .. 7*S-1   L2
	word 7*S .. 8*S-1   L3
	word 8*S            mscale  (rope.scaling.attn_factor, or 1)

where S = RopeTabStride(npairs).

## RULE 11e's cases

Four limits that read as hardware and were choices, found while making
sessions run at once on one device:

- **A lock per request.** The server held a device for a whole generate
  because the tier "serialises sessions on a device". The tier took its
  scratch per step; the server had copied that limit up to the whole
  request, so a long API answer blocked a chat until it finished.
- **The device-wide current session.** `devTier.cur` and `switchTo` made the
  session an implicit argument read in about sixty places, which forced one
  session per device at a time. Passed as an argument, it went.
- **"CUDA runs one stream in order."** One stream does; the backend had put
  every session on the legacy stream. Two sessions on non-blocking streams
  ran their kernels at once only after the next case was found.
- **Pageable copies.** With a stream per session the two still took 1.92x
  of one alone: a copy from pageable memory is staged by the driver
  synchronously and held the other stream's kernels back. Through a pinned
  arena per queue, 0.95x.

And one that was real: NVIDIA's laptop Vulkan driver runs dispatches from
two submissions one after another, on two queues or one (2.00x both ways),
while its datacenter driver on the V100 overlaps them (1.00x). The way past
it on that driver is to record the sessions' work into one submission,
which is the step loop's batching.

## The collective tiles lower on PTX, and the subgroup collectives are ops

**"A collective tile has no lane layout to map onto mma.sync" was a choice,
not a hardware limit (RULE 11e).** A collective tile leaves the lane layout to
the lowering, and mma.sync documents one, so `ptx/tile.go` picks it: every
tile is a grid of 8x8 atoms, and a lane holds of each atom the two elements
m16n8k16's fragments give it (A row g, columns 2t..2t+1; B rows 2t..2t+1,
column g; C row g, columns 2t..2t+1, with g = lane>>2, t = lane&3). A TileMMA of
any shape whose sides are multiples of 8 is then a run of m16n8k16 (m16n8k8 on
sm_75) over atoms with no data movement; an 8-row tile -- Metal's 8x8 -- runs
the m16 instruction with its second row of atoms zero. Loads and stores go
element by element from the view (a stride-0 bias is just an address); the
ceiling of that form is the load count, and ldmatrix over a shared view whose
rows the kernel asserts aligned is the upgrade. float32 operands (FlashPrefillTile's
P*V) are three tf32 products, a_hi*b_lo + a_lo*b_hi + a_hi*b_hi, about 21
mantissa bits through the matrix unit; tf32 is sm_80, so sm_75 lowers the f16
tiles and refuses the f32 ones by name. On FlashPrefillTile's gate one tf32
product read NMSE ~1.3e-7 and the three ~7e-8, both far inside the 1e-5 bound;
the three are kept because the IR type is float32 and a single tf32 product is
a silent precision substitution.

So Metal's prompt path (GemmTile, FlashPrefillTile) is ordinary IR on CUDA
too, gated by a capability -- `backend.LowersTile` compiles a staged,
loop-carried probe of the shape -- instead of `API() == "msl"`. Measured
correctness, all against the float64 reference with poisoned outputs:
GemmTile on every binary16-dequant format (NMSE 7e-8..1.6e-7, against the dp4a
path's 2-3.6e-5), its grouped form bit-equal to dense per expert, the tail
launch, FlashPrefillTile and its paged form, on the RTX 3050 Ti (sm_86), a T4
(sm_75; f16 only) and the M4; end to end with every block placed against the
host (TestTilePrefillMatchesTheHost) on sm_86, a T4 and the M4. A violation
(B's atom layout off by one element) fails TestTileMatchesReference at
|d| 131-173.

`TileGemm.Tall` multiplies 16-row A tiles where the device lowers 16x8x8
(PTX), two of the 8-row slots each: the staging is unchanged and the m16
instruction stops multiplying zeros.

Selection on CUDA is unchanged by default: the m16n8 GEMMs and FlashPrefill80
run ahead of the tiles, which are reached where those decline, or ahead of them
with Config.PreferTiles (JITLLM_GPU_PREFER_TILES) -- the arm
TestTilePrefillMatchesTheHost pins and the A/B uses. On a T4, before the m16n8
GEMMs landed, the tile GEMM against MatVecMMA read 0.989 (IQR 0.090, A/A
1.0136 at IQR 0.083): parity, within a noisy box. Against the m16n8 GEMMs
(the default now) PreferTiles read 1.16 for the default, rejected at IQR
0.116: the default stays first.

The subgroup collectives (`ir/subgroup.go`: reduce add/max/min, broadcast from
a uniform lane, shuffle from a per-lane lane & 31, lane id, width) lower to a
shfl butterfly / shfl.idx / %laneid / WARP_SZ on PTX, OpGroupNonUniform*
(Reduce) / Shuffle / the two builtins on SPIR-V, simd_sum / simd_shuffle and
the simdgroup attributes on MSL, and ds_bpermute on AMDGPU.
TestSubgroupCollectivesMatchTheReference holds every one at f32, i32 and u32
on CUDA, Vulkan and Metal; a butterfly missing its last step, a reduction
lowered as the wrong op and a shuffle clamped to 16 lanes each fail it. The
lane mask is not observable on NVIDIA (CUDA and Vulkan wrap a source past 31
themselves); it is the IR's definition for the targets that do not.

## Fusing a decode block's launches buys little: the launch is not the cost

The decode token on the V100 (Llama-3.2-1B Q4_K_M, sm_70, kernel timing by
event nodes, which add ~3.9 us each) issues 234 launches: 14.5 a block plus
the head. The block's small kernels are norm+quantize twice, quantize twice,
q rotation, k rotation into its page, v copy into its page, flash decode and
its merge.

`kernels.RoPEQKVPaged` is the three position writers as one launch, built
from the same emitted bodies (`ropeBody`, `pagedRopeBody`, `pagedCopyBody`),
so it is bit-identical to them: TestRoPEQKVPagedMatchesItsParts compares every
word of q and both NaN-poisoned pools, and TestFusedDecodeMatchesPerOp holds a
whole decode, every block placed and a partial seam, fused against
Config.NoFuse to the same logits bit for bit and both to the host (V100, the
3050 Ti, a T4; Llama-3.2-1B, Qwen2-1.5B, gemma-2b). One pair dropped from the
k part fails the kernel gate; v's source swapped for k's in the tier fails the
model gate at step 0.

It removes 32 of 234 launches a token, and measured (verify -ab fuse,
in-process, n=24, 64 tokens a round):

    V100 (CUDA graph)    A/A 0.9956 (IQR 2.5%)   fused 1.0071, 1.0062 (IQR 3.9%, 4.5%)
    M4 (Metal)           A/A 0.9893 (IQR 5.6%)   fused 0.9922 (IQR 4.5%), a second pass unstable

and on a T4 through vs-llamacpp.sh with JITLLM_GPU_NO_FUSE as the other arm
(processes interleaved ABBA, decode 128, one pass): 1.0184, IQR 0.018. That
row and the V100's fall on the same side and inside two percent.

Sub-noise on both, so not a claim. 32 launches for ~0.7% is ~0.6 us a launch
inside a CUDA graph: the graph already removed the issue cost, and what a fused
launch could still remove is the drain between dependent kernels, which these
tiny kernels barely have. A per-block megakernel would fight the same small
term at the price of a grid-wide barrier the IR does not have, so it is not
the decode lever on this card; the matvecs at 60-90% of the read wall and the
latency-bound norm (~6 us of a 2048-wide row on one 1024-thread group) are.
The same decode against llama.cpp's CUDA build, vs-llamacpp.sh, decode 128:
V100 1.0190 (IQR 0.034; A/A 0.9965, IQR 0.049), T4 1.1890 (IQR 0.030).

Bridging consecutive resident blocks into one submission is what `layersFor`
already does: one call runs the device range [lo, hi) as one captured graph
keyed by its launch sequence (graphKey: the range, the session, the paged
variant, the parities...), and a relocation, eviction or seam move changes the
range and so the key -- the run is re-captured, not patched.

`JITLLM_CUDA_KERNEL_TIMING` had been silent for every decode that runs on a
queue: the event nodes were armed only on the legacy stream's capture, and the
settle only on its replay. Both now run on a queue's capture too.


## The register report, and the tile that spilled on sm_70

`jitllm verify -registers [-json]` runs decode and the batched prefill and
prints, for every kernel each CUDA device compiled, what the driver says ptxas
allocated (cuFuncGetAttribute: registers, local bytes, static shared memory;
cuOccupancyMaxActiveBlocksPerMultiprocessor: resident workgroups at the
declared width), and for every host emitter the highest vector register it
names and its stack-addressed operands. Vulkan and Metal do not report yet
(VK_KHR_pipeline_executable_properties and MTLComputePipelineState are the
sources to wire). Two driver readings were checked against `ptxas -v` (CUDA
12.9) on the same PTX, sm_86: flashkvpaged 168 registers, 32768 B shared, no
spill; matvec_q4_k 36 registers; both agree.

Llama-3.2-1B Q4_K_M, worst first:

- RTX 3050 Ti (sm_86, 1536 threads an SM): nothing spills. flashkvpaged 194
  registers at its widest shape, 2 workgroups an SM (17%); the prefill
  GemmVolta tiles 159-162 registers, 3 workgroups (25%); every matvec, norm
  and quantize kernel 100% except the 1024-wide norms at 67%.
- V100 (sm_70, 2048 threads an SM): GemmVolta q6_k m4n4w1x4k2 hit the
  255-register ceiling and spilled 8 bytes a thread; q4_k m4n4 sat at 255
  without spilling. Everything else is clean; flashkvpaged 170 registers, 2
  workgroups.
- Host, AVX2: the k-quant matvecs name all 16 YMM and stage their scales in
  the red zone by design (emitMatVec4 8, emitKQuantMatVec 12, emitQ6KMatVec
  18, emitQ3KMatVec 21 stack operands); no other emitter touches the stack.

`TestHotKernelsKeepTheirRegisters` holds the hot kernels that ran to no spill
and a floor -- streaming kernels at least half an SM, GEMM and flash attention
at least two resident workgroups -- and every host emitter but the red-zone
stagers to no stack operand. Capped at 32 registers (`tier.WithCUDAMaxRegs`)
it reports 52 spilling kernels of 69 on sm_86; with a staging emitter taken
off the list it names it.

The batched GEMM now passes over a tile its compiler reports spilling
(`tier.Config.SpillingTiles` is the other arm): on sm_70 q6_k takes the m4n2
tile at 223 registers instead. `verify -ab spill`, V100, Llama-3.2-1B,
four 512-token prompts a round, 21 rounds, pinned to the second socket: A/A
1.0008 (refused both) and 1.0005 (allowed both); refused against allowed
1.0362 and 1.0375 with the shipped tier as B, 0.9655 and 0.9637 with the arms
swapped -- 3.6-3.8% faster prefill, IQR/median at most 0.5%. Decode does not
run GemmVolta and its kernel split is unchanged (2706.5 against 2708.4 us of
kernels a token). On sm_86 no tile spills, so nothing changes there by
construction.

## The operand prologue: the norm and the quantize inside the matvec

The section above found the launch was not the decode's cost. What a
per-op norm or quantize does cost is its latency: a 1024-thread norm on one
multiprocessor, or a 16-group quantize, while HBM idles, before the matvec
that reads its operand can start. `kernels.ActIn` moves that work into the
consumer: every workgroup of the next matvec normalises (or forms
act(gate)*up) and quantizes the row into threadgroup memory and reads its
int8 operand from there. The bits are the launch's it replaces (the norm's
sum of squares is RMSNormGroup's reduction run by virtual threads in the same
order; amax and per-16 sums are a max and integer sums). Before the prologue
each thread prefetches its first weight words (`ir.OpPrefetch`,
`prefetch.global.L2` on PTX) so the weight stream starts under the prologue.

What it costs is the prologue once per workgroup instead of once per row, so
it pays only where few groups share the row. Tried in order (V100, Llama-3.2-1B
Q4_K_M, `speed -p 32 -n 128`, process per arm, 1312 MHz):

    per-op (two launches)                             535 tok/s
    prologue, every site, norm pass then quantPhase   ~490 (a loss: three barriers a 128-word iteration, two global passes)
    + shuffles for the window and sums, one pass      537 (even)
    + prefetch 4-8 words a thread                     541-544
    + prefetch every word                             507 (thrashes L2)
    sites q/k/v + attention output + down (no gate/up) 557

gate and up (256 groups of 32 rows) lose: their prologue is 4-6 us a launch
against the norm launch's ~5 us, and two of them replace one. The policy
(`tier.actInPays`): groups <= 128 and groups*k <= 512Ki. Llama-3.1-8B's down
(128 groups, k 14336) lost 4% (132 against 137.6 tok/s) and is outside it;
its attention output (128, 4096) is even.

A verify -ab that ran the table splits (partial-writing split 32, 512 groups
for q) read the prologue at 0.50 against per-op; the policy excludes them,
and `-ab actfuse` warms past the split tuner before it compares.

The in-process A/B (`verify -ab actfuse`, JITLLM_GPU_TUNE=1, n=20, 64 tokens a
round, A/A first):

    V100 1B   A/A 1.0054 (IQR 2.9%)   1.0280 (2.2%), 1.0232 (3.0%)
    V100 8B   A/A 1.0027 (1.7%)       1.0077 (2.2%), 1.0081 (2.0%)
    T4 1B     A/A 1.0066 (2.2%)       0.9929 (2.6%), 0.9818 (2.4%)
    T4 8B     A/A 1.0012 (1.6%)       0.9907 (2.2%), 0.9939 (2.0%)

On the T4 (40 multiprocessors) the redundancy costs more per launch: q/k/v's
segment kernel took 506 us of event-timed device time a token against 334
plain, more than the norm launches it removed. A per-device measurement in
place of actInPays' two numbers is the open item (the split tuner times the
plain kernels only).

Launches a token (CUDA kernel timing, event nodes): V100 1B 198 -> 150 (16
norm launches before q/k/v, 16 quantizes before the attention output, 16
before down), 8B 390 -> 358 (the attention output's 32). Every launch is a
kernel boundary, the only synchronisation a decode token has inside its
graph, so syncs fall by the same count. DRAM bytes a token are unchanged to
within the operand's few KiB (it was L2-resident); L2 reads grow by
groups*k*4 a prologue launch (about 4 MiB a block on the 1B).

Against llama.cpp CUDA (build at llama-sm70, vs-llamacpp.sh gpu, decode 128,
12 ABBA rounds, cores 43-53):

    V100 1B   prologue 1.1648 (IQR 0.054)   main 1.1316 (0.028)
    V100 8B   prologue 1.0807 (0.012)       main 1.0735 (0.026)

Found on the way: the retune built the FFN up projection's gated variant
only where the first split already had one, so a block prepared at a
partial-writing split ran act(gate)*up as its own launch for good
(`mv.wantGated`). And a CUDA kernel whose registers allow fewer threads than
its workgroup failed at launch (T4, the q/k/v prologue at 1024 threads); the
compile now refuses it with `backend.ErrGroupTooWide`.

### The prologue decided per card by measurement

Five T4 rows read 0.982-0.994 with actInPays' V100-fitted constants: a small
regression, not noise. `tuneActIn` replaced the constants: after the decode
retune it times each site's two launches against its prologue over up to 16
placed blocks, interleaved, and keeps the prologue only under 0.99 (cached per
binary, device and shape). Its verdicts:

    V100 1B   q/k/v 0.88 and 0.86, attention output 0.71: kept; down 1.04-1.07: dropped
    V100 8B   q/k/v 0.95: kept; attention output 0.997, down 1.12-2.4: dropped
    T4 1B     1 of 5 sites kept
    T4 8B     none kept

and the in-process A/B after it (JITLLM_GPU_TUNE=1, n=20, A/A first):

    V100 1B   A/A 1.0047 (IQR 2.2%)   1.0327 (2.6%), 1.0337 (2.8%)
    V100 8B   A/A 1.0026 (2.6%)       1.0072 (1.8%), 1.0063 (1.9%)
    T4 1B     A/A 1.0050 (2.0%)       1.0106 (2.2%), 1.0091 (2.3%)
    T4 8B     no site kept, both arms one launch sequence: 0.9920 (2.4%), 1.0016 (1.3%)

The T4 8B A/A run did not complete: `fuseActIn` keyed a site before its
prologue compiled, so a site the T4 declined (`ErrGroupTooWide`) went to the
tuner with no kernel to time. A key now goes only beside a built kernel, and
the rows taken again with that build (A/A first, two passes):

    V100 1B   A/A 0.9986 (3.0%)       1.0348 (2.7%), 1.0253 (2.3%)   3 of 5 sites kept
    V100 8B   A/A 1.0031 (1.7%)       1.0042 (2.1%), 1.0071 (2.2%)   2 sites kept
    T4 1B     A/A 1.0000 (2.7%)       1.0099 (2.5%), 1.0079 (1.9%)   1 of 5 kept
    T4 8B     A/A 1.0021 (1.0%)       1.0008 (1.4%), 1.0022 (0.9%)   none kept

The verdict on a marginal site moves between processes: the tuned model gate
kept 3 of 5 on the same T4 where `verify` kept 1. Each process is held to
what it chose (`TestActInKeepsThePrinciples/tuned`), and the rows read the
shipping configuration either way.

## The grid barrier: an IR op, lowered on every target

`ir.OpGridBarrier` (`ir/gridsync.go`) orders a whole launch: no workgroup
passes it until every workgroup has reached it, and every global write before
it is visible to every read after it. It was missing, not impossible (RULE
11e): the mechanism is an atomic arrive-and-wait counter in a sync buffer the
op owns, plus a grid no larger than the device holds resident at once.

- **PTX** (`ptx/grid.go`): `bar.sync`, thread 0 arrives with an `acq_rel` add on
  one 64-bit word (count in the low half, generation in the high), the last
  arriver empties the count and moves the generation in one `red.release`
  add, the others spin on an `ld.acquire` of the generation, `bar.sync`. Launched
  with `cuLaunchCooperativeKernel` on a grid of at most the occupancy
  calculator's blocks per SM times the SM count (`cuda/coop.go`): the CUDA
  Programming Guide's grid synchronisation section requires that co-residency
  and the cooperative launch guarantees it or refuses the grid.
- **SPIR-V** (`spirv/grid.go`): the same counter as a count word and a
  generation word, Device-scope atomics in structured control flow, every
  storage buffer `Coherent`. The limit is one workgroup per compute unit as
  `VK_NV_shader_sm_builtins` or `VK_AMD_shader_core_properties` states the
  count; core Vulkan states neither the count nor any forward progress between
  workgroups.
- **MSL** (the `gridPreamble` in `msl/msl.go`): relaxed atomics ordered by
  `atomic_thread_fence` at device scope and `coherent(device)` buffers, MSL 3.2.
  The limit is one threadgroup per GPU core as the IORegistry's
  `gpu-core-count` states it (`metal/cores_darwin.go`).
- **AMDGPU** (`amdgpu/grid.go`): agent-scope `atomicrmw` and fences, the
  workgroup count from the dispatch packet. Lowered and assembled only: no AMD
  card here ran it, and the HIP backend states no limit yet, so it refuses the
  launch.

`backend.GridLimit` is the limit and every backend's launch refuses a wider
grid (RULE 13). The sync buffer is zero when allocated and every barrier leaves
it valid for the next launch, of any grid.

The gate, `TestGridBarrierHoldsEveryWorkgroup`: `kernels.GridProbe` runs eight
phases, each reading a word another workgroup wrote in the phase before into a
poisoned output, behind a skew that differs by workgroup and phase; at two
workgroups, half the limit and the limit, five launches on one sync buffer, two
grids taking turns on one buffer, and inside a recorded CUDA graph (a
cooperative launch captures). Its violation,
`TestGridBarrierProbeSeesAnEarlyCrossing` (a `jitllmtest` build:
`BreakGridBarrierForTest` lets every waiter through), finds hundreds to
hundreds of thousands of poisoned words on every arm. Green on the RTX 3050 Ti
(sm_86; CUDA 240 workgroups of 128 resident, Vulkan 20), the V100 (sm_70;
CUDA 1280 of 128, Vulkan 80), a T4 (sm_75; CUDA 320 of 128) and the M4
(Metal, macOS 26.5; 10 of 128 or of 256, one per core).

The two-grid case is there because the first PTX form -- one count that only
grew, each barrier waiting for the next multiple of the grid -- deadlocked the
first decode that ran two persistent kernels of different grids on one buffer.

What a barrier costs (`TestGridBarrierCost`, 256 barriers against none, median
of 50 launches, 128-wide workgroups):

    V100 CUDA     2-160 workgroups 0.83 us   640 1.89 us   1280 4.27 us
    V100 Vulkan   2-80 workgroups  1.64-1.85 us
    3050 Ti CUDA  2-120 workgroups 0.79-0.86 us   240 1.07 us
    3050 Ti Vulkan 2-20 workgroups 1.42-1.91 us
    T4 CUDA       2-160 workgroups 1.52-1.76 us   320 2.49 us
    M4 Metal      2-10 workgroups  1.37-1.44 us

The count-and-generation form with a separate generation read cost 1.46 us at
80 workgroups on the V100; the one-word form removed a round trip to L2. A
launch boundary inside a CUDA graph is ~0.6 us (the section above), so a grid
barrier is not cheaper than the launch it replaces; what it can remove is the
drain, and the round trip of an intermediate through memory, not the launch.

## A persistent decode block: composed, bit-identical, and slower

`Config.Persist` (`tier.WithPersist`, `JITLLM_GPU_PERSIST`) composes each
decode block's launches into persistent kernels (`ir.Persist`,
`tier/persist.go`): the launcher holds a block's launches, cuts them into runs
the composer can take, and issues each run as one cooperative grid whose
workgroups deal out every launch's groups in order, a grid barrier wherever a
launch touches what an earlier one in the phase wrote. Every body is the
per-op kernel's own IR with its group index replaced, so the two paths agree
bit for bit; the plan is keyed by the kernels, grids and aliasing pattern, and
the buffers are bound per launch, so relocation, paging and seam moves change
nothing about it. CUDA only: Vulkan and Metal state a compute-unit count and
no co-residency guarantee, and two sessions' persistent kernels on two queues
could each hold units the other waits for.

On Llama-3.2-1B Q4_K_M a block is 23 launches on the RTX 3050 Ti and fewer
on the V100 (whose q/k/v is one segmented matvec); runs are cut where the
width changes (the 1024-wide norm, sm_70's 256-wide group-split matvecs) and
before a 128-wide kernel with workgroup memory inside a wider run, which leaves
about four persistent kernels a block. Inside them almost every launch is its
own phase: the block is a dependency chain (norm, matvec, its split-K reduce,
rotary, attention, merge, quantize, ...), and the split-K matvecs share one
partials buffer, so q, k and v cannot share a phase either. 744 launches
become 128 persistent kernels in 680 phases.

Gates: `TestPersistentDecodeMatchesPerOp` (Llama-3.2-1B and Qwen2-1.5B, 24
teacher-forced steps, every block placed with and without the captured graph
and a partial seam: logits bit-identical to the per-op arm, and both coherent
with the host; Stats().PersistLaunches > 0 on the persistent arm and 0 on the
other) on the 3050 Ti and the V100; its violation
`TestPersistentDecodeSeesABrokenBarrier` (a `jitllmtest` build, every grid
barrier letting its waiters through) changes 897792 of 1026048 logits.

Measured on the V100 (Llama-3.2-1B Q4_K_M, CUDA, in-process `verify -ab`,
n=24 rounds of 64 tokens): A/A 0.9989 (IQR 3.8%); persistent against per-op
0.8916 (IQR 1.2%) and 0.8907 (IQR 1.0%) -- an 11% loss. `jitllm speed`, tg256
after a 64-token prompt, three runs each:

    graph replay   per-op 518.3 / 517.3 / 517.5 tok/s, device 1825.9 us a token
                   persist 493.0 / 494.8 / 495.8 tok/s, device 1916.4 us a token
    no graph       per-op 490.5 / 492.3 / 489.7 tok/s
                   persist 474.7 / 419.3 / 473.2 tok/s

On a T4 (sm_75, same model, same harness): A/A 0.9989 (IQR 1.0%); persistent
against per-op 0.9606 (IQR 1.0%) and 0.9594 (IQR 0.6%), a 4% loss; tg128
236.0 / 235.9 / 235.0 against 233.2 / 233.1 / 233.2 tok/s, device 4122.3
against 4182.3 us a token. The gate and its violation are green there too.

The per-op token's small kernels (`JITLLM_CUDA_KERNEL_TIMING`, each with
~3.9 us of event node in it): rmsnormquantw 11.7 us x32, quantize 8.3 x33,
roperowst 8.7, pagedroperowst 7.9, pagedcopyrows 8.5, flashkvpaged 10.5,
flashmergewide 8.5 (x16 each); 1352 us a token with the events, ~790 without.

Why it loses: a grid barrier costs 0.8-1.9 us on this card (the section
above) and a launch boundary inside a CUDA graph ~0.6 us, and composition
replaces one with the other almost one for one, because the block is a chain.
It also runs each phase on the persistent grid -- at most what one fused
kernel's registers leave resident -- rather than the per-op launch's own
grid. What would win is not composition but fusion that keeps an intermediate
on chip (norm into the matvec's activation, the reduce into its consumer) and
so removes phases, which is the composer's next step rather than this one.
Off by default (RULE 6: a clear regression).

## The dependency-counter block: correct, and slower than both

`Config.PersistDeps` (`tier.WithPersistDeps`, `JITLLM_GPU_PERSIST=deps`;
design in `docs/design/dep-block.md`) composes the same runs as the persistent
block and orders them with completion counters (`ir.PersistDeps`): a task waits
only on the tasks it reads or overwrites, a tile of an elementwise task only on
the producer tiles its span analysis says it touches, and a workgroup with no
tile in a task skips its waits.

Gates on the V100 (sm_70): `TestPersistentDecodeMatchesPerOp` holds the
dependency-counter arm bit-identical to per-op on Llama-3.2-1B, Qwen2-1.5B and
gemma-2b, every block placed with and without the graph and on a partial seam;
`TestDepDecodeSeesABrokenWait` (every wait returning at once) changes 897792 of
1026048 logits; `backend.TestDepCountersOrderTheChain` and its violation hold
the protocol at the co-residency limit (1280 workgroups of 128) and inside a
graph. A Llama-3.2-1B decode takes 476 tile edges in 1792 kernels; the rest
are whole-task edges, because every matvec reads its whole input.

Measured on the V100 (CUDA, in-process `verify -ab`, n=24 rounds of 64
tokens, graph replay, two passes):

| model | A/A | deps vs per-op | deps vs grid barrier |
|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 1.0010 / 0.9945 | 0.8467 / 0.8430 | 0.9696 / 0.9702 |
| Llama-3.2-3B Q4_K_M | 1.0009 / 1.0017 | 0.8164 / 0.8189 | 0.9700 / 0.9730 |
| Llama-3.1-8B Q4_K_M | 0.9996 / 1.0009 | 0.8555 / 0.8572 | 0.9721 / 0.9728 |

(IQR/median 0.9-5.3%.) It loses 14-18% to per-op and 3% to the grid barrier.
A whole-task wait still waits on every active workgroup of the producer --
the barrier's arrival set -- and adds a signal per task per workgroup and a
wait per tile; the decode block's edges are almost all whole, so the counters
remove little waiting and add atomics. What would change this is fusion that
keeps an intermediate on chip and removes tasks, not finer ordering of the
same tasks. Off by default (RULE 6: a clear regression).

## Gate and up in one launch: exact, and a loss past the 1B

`Config.GateUpFuse` (`tier.WithGateUpFuse`, `JITLLM_GPU_GATEUP=1`) runs a
dense gated FFN's gate and up as one `kernels.MatVecSegments` launch over
their shared operand, and has down form act(gate)*up from the two rows and
quantize it in its prologue (`kernels.ActIn.Gated`) with the residual add:
three launches where there were four (gate, up writing act(gate)*up, down
quantizing it). Each row keeps its own reduction, and the element and its
rounding are QuantizeAct's on both paths, so the logits are the same bits:
`TestGateUpFuseMatchesThreeLaunches` (Llama-3.2-1B on the V100, every block
placed with and without the graph and a half seam; both arms read one tuner
cache so their splits agree), violation `TestGateUpFuseSeesASwap` (gate and up
handed to down the wrong way round: 384768 of 513024 logits differ). It needs
gate and up at one in-group split, which the tuner picks on a card and the
fitted table does not (`Stats.GateUpWhy` names the refusal).

V100, CUDA, tuner on (`JITLLM_GPU_TUNE=1`), in-process `verify -ab`, n=24
rounds of 64 tokens, two passes:

| model | A/A | fused vs four launches |
|---|---|---|
| Llama-3.2-1B Q4_K_M | 0.9977 / 1.0016 | 1.0064 / 1.0052 |
| Llama-3.2-3B Q4_K_M | 1.0004 / 1.0018 | 0.9481 / 0.9545 |
| Llama-3.1-8B Q4_K_M | 0.9986 / 1.0002 | 0.9124 / 0.9133 |

(IQR/median 0.3-1.4%.) The launch it removes is worth half a percent on the
1B; the cost it adds is down's prologue, which every one of down's workgroups
pays over the whole FFN width (8192 on the 3B, 14336 on the 8B) -- the same
reason the tuner keeps down's own operand prologue off on this card. Off by
default; a per-shape choice is the joint tuner's to make. A form with no
prologue in down needs one workgroup to own gate row i and up row i and
write act(gate)*up itself (two weight streams a thread), which MatVecSegments
cannot express.

Note for every earlier `verify -ab` row in this file: verify pins the split
tuner off unless `JITLLM_GPU_TUNE` says otherwise, so those rows are at the
fitted table's splits, not the shipping tuned ones (the persistent and
dependency-counter rows included).

## Prefill on the V100: where a prompt's time goes

`jitllm speed` with per-kernel timing (each kernel padded by ~3.9 us of
event), V100, CUDA. Llama-3.2-1B Q4_K_M at 2048 tokens (102.3 ms of kernels):
the f16 tensor-core GEMMs (GemmVolta) 70.5 ms, flash prefill 17.2 ms,
rmsnormf16tw 3.5, actmulf16t 4.0, ropeqkvpaged 2.6, add 1.4, actf16t 1.3,
split-K reduce 1.1. Llama-3.1-8B at 2048 (516 ms): GEMMs 412 ms, flash
prefill 53.6, rmsnormf16tw 15.0, actmulf16t 13.0, rope 9.6, actf16t 5.5,
add 5.0, reduce 2.2. Rates, three runs: 1B pp512 18270 / 16981 / 18292 tok/s,
pp2048 18230 / 17008 / 18253; 8B pp512 3584 / 3569 / 3029, pp2048 3616 /
3636 / 3608.

The 8B's 14336x4096 GEMM at 512 tokens is ~770 us for 60 GFLOP, ~78 TFLOPS
against the card's 125 tensor-core peak (62%); the 4096x4096 ~68 TFLOPS.
GemmVolta already stages double-buffered (a trip's dequant runs under the
previous trip's MMAs). At 64 tokens the background tuner's trials
(WithLazyTune) land among the first prompts' kernels and the first
prompt's rate is a fraction of the third (8B pp64 535 / 778 / 1850 tok/s).

## Cold TTFT: the background tuner held the lock for a whole trial

A fresh process (empty tune cache) prefilling 64-token prompts dipped to a
third of its rate on the fourth and fifth prompt. Timestamped trials named the
cause: the lazy split tuner's trial on the output head (Q6_K 128256x4096) held
g.mu for ten rounds, 21 ms, and a prompt arriving behind it waited. Each round
now waits for tuneQuiet and takes the lock alone
(`TestBackgroundTrialYieldsBetweenRounds`).

V100 (CUDA, GPU 1), `speed -p 64 -n 4 -r 8`, a fresh cache each run, base and
new interleaved, three pairs. Slowest of the eight prompts, tok/s:

    model                  base               new
    Llama-3.1-8B Q4_K_M    818 / 526 / 802    1414 / 1472 / 1480
    Llama-3.2-1B Q4_K_M    1176 / 1892 / 1911 4494 / 4342 / 4516

The base's later prompts read ~1900 on the 8B against the new ~1620: that is
the clock, not the engine. The V100 idles at 1312 MHz and boosts to 1530 under
load; the base's dense trials kept it boosted, and a warm cache with no trials
reads ~1700 on the base binary too (an nvidia-smi trace beside the run showed
both clocks). A row on this card is not comparable across arms unless its
clock is reported.

## Gate and up in one launch, each workgroup summing both rows (Gate2)

#167's fusion lost past the 1B (0.95 on 3B, 0.91 on 8B) because every
workgroup of down redid act(gate)*up and its quantize over the full FFN width.
Gate2 (`kernels.MatVecShape.Gate2`) moves the fusion the other way: up's
gated matvec carries the gate's weight as a second stream, sums both rows in
one pass over the shared operand, reduces the gate's partials in the order the
gate's own launch reduces them, and writes act(gate)*up. Down keeps its own
operand, so nothing is recomputed: one launch goes per block.

It is bit-identical on PTX, whose lowering fixes every rounding
(`backend.TestMatVecGate2IsTheTwoLaunches`, Q8_0/Q4_K/Q6_K at split 1 and the
in-group splits; `model.TestGateUp2MatchesTwoLaunches` on Llama-3.2-1B with
every block placed with and without the graph and at a half seam; the swap
violation `TestGateUp2SeesASwap` moves 384768 of 513024 logits). On SPIR-V it
is not: Fma lowers to an OpFMul and OpFAdd without NoContraction, and the
NVIDIA Vulkan driver contracted the two kernels' sums differently (378 of
1024 Q4_K rows differed in low bits). MSL compiles with fast math. So the tier
builds it on PTX only (`kernels.Gate2Exact`); elsewhere the backend gate holds
it to NMSE 1e-10.

V100 (sm_70, CUDA, GPU 1), `verify -ab gateup2`, n=24 interleaved, -n 64,
`JITLLM_GPU_TUNE=1`, two passes, A/A first in each:

    model                  A/A      Gate2 vs two launches (pass 1 / pass 2)
    Llama-3.2-1B Q4_K_M    0.9998   1.0222 (IQR 0.8%) / 1.0250 (IQR 0.7%)
    Llama-3.2-3B Q4_K_M    1.0004   1.0279 (IQR 1.1%) / 1.0202 (IQR 0.9%)
    Llama-3.1-8B Q4_K_M    1.0011   1.0097 (IQR 0.4%) / 1.0076 (IQR 0.2%)

It is the default (`Config.GateUp2`; `JITLLM_GPU_GATEUP2=0` is the other
arm). `TestDecodeDoesNotAllocate` holds at zero engine allocations on CUDA
with it on. No per-shape tuner yet: it won on every shape measured.

## A prompt chunk's block-end add folded into the next block's norm

On the binary16 GEMM path the FFN norm already took the attention's residual
add into its launch (`RMSNormF16TRows` with add); the add after down ran
alone, and the next block's attention norm read its result back. That add is
now held for the next norm (`foldAddNorm`): x = x2 + ffn and h = RMSNorm(x)
in one launch, the same bits, since the norm reads the sum it stores. A held
add whose next block's norm is not the plain one runs alone (`foldTakes`), as
does the last block's. Gates: `TestAddNormFoldMatchesTheAdd` (Llama-3.2-1B,
prompt and decoded logits bit-identical, every block placed with and without
the graph and a half seam; 14 of 15 adds folded) and its violation
`TestAddNormFoldSeesABrokenAdd` (every logit moves).

V100 (CUDA, GPU 1), `verify -ab addnorm`, four 512-token prompts a round,
n=20 interleaved, `JITLLM_GPU_TUNE=1`, A/A first, two passes; the SM clock
is the modal reading of a 250 ms nvidia-smi trace over each arm:

    model               A/A              folded vs alone    IQR    SM clock
    Llama-3.2-1B        0.9999 / 1.0000  1.0078 / 1.0072    0.3%   1530 MHz
    Llama-3.1-8B        0.9983 / 1.0015  1.0013 / 1.0011    0.8%/1.4%  1440-1530 MHz

The 1B gains 0.7-0.8%; the 8B's GEMMs hide it inside the noise (and pull the
card off its boost clock, which the 1B holds). On by default;
`JITLLM_GPU_NO_ADDNORM=1` is the other arm.

## The FFN's recipes, timed per card

The first composer slice (`docs/design/composer.md`): the dense gated FFN's
forms -- two launches, Gate2, Gate2 with the norm's prologue (gate2in), and
gate+up with down forming the product (gu) -- are recipes of one site,
timed per shape key norm through down by the one trial (`timeRecipes`),
which the operand-prologue tuner now runs on too; `tuneFFN` replaces
`tuneGateUp2`. gate2in is the recipe found by composition: behind a
residual the attention output's launch already added, the FFN's
RMSNormQuantRows goes and Gate2 builds its operand itself, one launch where
there were two. Bit-identical on PTX (`backend.TestGate2InIsTheTwoLaunches`,
96 cases; `model.TestFFNRecipesMatchTwoLaunches` every block placed with and
without the graph and at a half seam; violation `TestGate2InSeesASwap`,
384768 of 513024 logits move). `TestDecodeDoesNotAllocate` holds at zero on
CUDA.

V100 (sm_70, CUDA, GPU 1), `verify -ab ffn` (gate2in forced against Gate2
forced), n=24 interleaved, -n 64, `JITLLM_GPU_TUNE=1`, A/A first; SM clock
the modal reading of a 500 ms trace over each arm, 1530 MHz throughout.
Ratios >1 mean gate2in is faster:

    model                  A/A               gate2in vs Gate2   IQR
    Llama-3.2-1B Q4_K_M    0.9989 / 1.0012   1.0121 / 1.0074    2.0% / 4.1%

(An earlier binary of the same kernels read A/A 0.9941 / 0.9987 and
1.0136 / 1.0102.) The tuner kept gate2in on both of the 1B's FFN keys. The
3B and 8B were not measured: the card's disks had no room for their
version-30 containers. The win is the launch and the h/a round trip it
removes, about 1% of a 1B token, smaller as the FFN's weight traffic grows.

## q, k and v as one prompt GEMM grid

The 8B prompt profile put k and v (1024x4096) at split 2 with a reduce each:
their two GEMMs and 68 reduces were about 7.4 ms of a 132 ms 512-token
prompt. `kernels.GemmVoltaSegments` runs q, k and v as one grid over one
binary16 operand, MatVecSegments' construction for the GEMM: each segment runs
GemmVolta's own body for its shape, unsplit, in its own run of workgroups,
with the staging buffers declared once at the largest size. A projection that
reads another operand layout (Q6_K's v beside Q4_K's q) runs alone. It is
bit-identical to each segment's own launch (`TestGemmVoltaSegmentsIsTheLaunches`,
the sm_70 tile on a V100 and the m16n8 tiles on an RTX 3050 Ti) and to three
GEMMs held to one k each (`TestQKVGemmMatchesThreeGemms`, with its swap
violation); against the shipping three it sits inside the step gate's band.

It loses, so it is opt-in (`Config.QKVGemm`, `JITLLM_GPU_QKVGEMM=1`). The
sm_70 m4n4 Q4_K tile already sits at the 255-register cap, and the segment
guard spills a word (8 bytes; admitted to measure it, `qkvSpillMax`). V100
(CUDA, GPU 1), `verify -ab qkvgemm`, four 512-token prompts a round, n=20,
`JITLLM_GPU_TUNE=1`, A/A first, two passes, SM clock the modal nvidia-smi
reading:

    model           A/A              grid vs three       IQR        SM clock
    Llama-3.2-1B    1.0004 / 0.9995  0.9923 / 0.9938    0.2-0.5%   1530 MHz
    Llama-3.1-8B    1.0034 / 1.0025  1.0024 / 0.9890    1.5-1.8%   1447-1530 MHz

The split-k it removes was buying its own parallelism: the unsplit k and v
blocks finish late in q's grid. What would change it: a register-lighter tile
for the segments, or segments that keep k's split.

## The composer's decode block against the tier's, kernel by kernel

V100-SXM2-16GB, CUDA, Llama-3.2-1B Q4_K_M, one decode token, graphs off
(`nvprof --print-gpu-trace`), every block served whole by the composer
(`JITLLM_GPU_COMPOSED=1`) against the tier's own kernels.

Where the composer forms the same work as one of the tier's kernels it
matches it: the output projection 7.6 us against 8.9, down 22.8 against
22.3. The gap was launches the tier's authors fused by hand: 14 composed
launches a block against 9, about 140 us against 90.

- **Siblings** (`jit.Siblings`): independent kernels reading one input run as
  one launch, each its own workgroups under a guard loop that runs once in
  them. q, k and v went from three launches (23.6 us) to one (13.2; the
  tier's `matvecseg` 9.9), gate and up from two (38.0) to one (30.4; the
  tier's `gu` 30.7). A share's thread index is made its guard's: before that,
  values made from the thread index alone were hoisted above every guard,
  live across all shares, and the merged q/k/v kernel needed more registers
  than a 1024-wide workgroup allows (`backend.ErrGroupTooWide`).
- **The head's split** (`jit.Split` with Parts): it cut a reduction 64 ways
  whatever its rows, so the 128256-row output projection ran 8016 workgroups
  of 1024 threads, 383.6 us. A reduction whose elements fill the device's
  slots is not split: one row a thread, 1002 workgroups of 128, 261.5 us (the
  tier's 259.2).
- **A producer run in its consumer's workgroups** was built and measured, and
  removed. The attention's norm and int8 form inside q/k/v took 23.6 us
  against 6.8 + 13.2 apart: the norm is a reduction over the row, and every
  workgroup paid its latency before its own matvec, so only the launch gap
  went. Its body is its own workgroup width's (a 1024-thread producer run in
  256-thread gate/up workgroups computed the wrong values). The int8 forms of
  the attention's output and of the activation could not join their matvec
  at all: their windows combine through a workgroup exchange and its barrier,
  which a loop whose trips differ by thread cannot hold, where the tier's
  quantize combines a window in one warp's shuffles.

### Where a decode token's time went, against the tier's

Same card, Llama-3.2-1B Q4_K_M, graphs off, per decode token by kernel name
from `nvprof --print-gpu-summary`: a 264-token run less an 8-token run, so
the startup's trial kernels cancel. That holds for the composer's kernels,
whose names the trials never use. The tier's trial kernels share their
decode kernels' names and their count differs from run to run: its column is
the run whose differences came out at 16.4 launches a token for each
per-block kernel, and a second run's, which did not (12.2 for `gu_nin`), is
not quoted.

| site, a block | tier | composer |
|---|---|---|
| attention norm and q/k/v | `matvecseg_…_nin`, 12.8 us | norm + int8 form 6.8, q/k/v 13.4 |
| FFN norm and gate/up | `…_gu_nin`, 30.5 | norm + int8 form 6.8, gate/up 30.0 |
| attention (about 140 keys) | rope 2.8, `flashkvpaged` 5.5, merge 1.8 | rope 3.0, one stage 9.8, merge 2.4 |
| attention output | `…_qin`, 7.3 | int8 form 4.2, output projection 7.6 |
| activation and down | quantize 2.9, down 14.7-24.3 | act x up + int8 form 5.7, down 20 |

The tier runs both norms as its matvec's prologue (`kernels.ActIn`) and its
gate/up kernel applies the activation, so its quantize before down reads one
row where the composer's reads two.

- **The k-quant window.** Every k-quant model quantizes its activations over
  a 256-element window (`cpu.WideActWindow`: the host hoists the scale, and
  the device must quantize as the host does, which writes the same KV
  cache). At four elements a thread a window spans 64 threads, two warps, so
  the composer combined it through a 64-slot workgroup exchange -- a
  barrier and 64 workgroup-memory loads in every thread -- and the shuffle
  combine only ever applied to 32-element windows. `Coarsen` takes the unit
  that keeps every combine within one warp (eight elements at 256), and a
  chain's workgroup is the widest in whole warps that its units fill (64
  for TinyLlama's 704), so no lane is clamped into a butterfly.
- **Dead float rows.** The norms' float rows and the activation's product
  were written and never read once their consumers read the int8 form;
  `DeadStores` drops a value's store no kernel reads and the graph does not
  hand back, and runs again after `Coarsen`.
- **Stores in a nested loop.** A host phase dealt as a spread loop with its
  stores in a guard of the real units left the GPU lowering with stores it
  never reached: it emitted the spread loop and skipped the guard's stores,
  and the kernel lowered without them. It lowers a store's loop inside its
  parent now, and fails on a store it never emits; the front-end test lowers
  every GPU kernel of the five models, and fails on that violation.
- **A producer in its consumer's workgroups, again** (`jit.Prologue`, with
  the shuffle combines and guarded host phases above). Every producer site
  of a Llama block merged -- q/k/v, gate/up, the output projection and down,
  seven launches a block where there were eleven -- and the decode token's
  kernels took 2.44 ms against 2.33 without it (385.7 tok/s against 396.6).
  By site: q/k/v with the norm 25.3 us against 6.5 + 13.4 apart, gate/up
  with the norm 39.8 against 39.0, the output projection and down with the
  int8 form 19.0 against 19.5. A norm's reduction over the row is paid in
  every consumer workgroup before its matvec starts, and gate/up runs 512
  of them; the pass refuses a producer with a reduction over the whole
  workgroup. The int8 forms it keeps: interleaved with graphs on (the
  shipping launch), eight rounds, the composer read 1.022 of itself without
  them, inside an A/A control of 0.987 and IQR/median 0.037 -- unresolved,
  not a regression, so kept and not quoted.
- **Registers and the second wave.** The composer's q/k/v launch ran 96
  workgroups of 1024 threads at 40 registers (`ptxas -v`, sm_70): one
  workgroup a multiprocessor, so 96 workgroups on 80 multiprocessors ran
  in two waves, where the tier's `matvecseg` holds 32 and runs them in one.
  Each projection alone was 32; a `Siblings` share wrapped alone was 44.
  The share's thread index and coordinates were made through its guard's
  iteration (thread + iter), which kept them in registers where the device
  re-reads `%tid.x` and `%ctaid.x` for free. The lowering takes a run-once
  uniform loop as a scope instead -- what the body reads is made inside it
  on demand and forgotten when it closes -- and the shares read the thread
  index plainly: q/k/v 32 registers, gate/up 44 to 32. A register cap does
  not do it: `.maxnreg 32` on every kernel spilled, q/k/v and gate/up 22.9
  to 32.8 us, the attention stage 9.8 to 24.2. Loading a share's buffer
  pointers inside its guard changed nothing (ptxas rematerializes them),
  nor did making constants at their use, and both were taken out.
- **A value kept from its first use.** The scope rule passed on Vulkan and
  faulted on CUDA ("an illegal memory access", sm_70 and sm_86): the
  lowering kept the thread index from the first place a kernel asked for
  it, which under the rule was inside the first share, so every later share
  read a register its own workgroups never set. Vulkan's robust buffer
  access clamped the reads it made of it. The index is read where it is
  used now, and `ir.Validate` holds every value to its loop -- one made
  inside a loop's body is not read after the loop closes, its phis aside --
  where the composer's lowering forms a kernel, not first where a backend
  compiles it. Every kernel of `jit/gpu/kernels` keeps the rule; the
  front-end test fails with it on the kept index.
- **What did not pay.** Prefetching each thread's first eight weight words
  before an int8 prologue (main's `ActIn.Prefetch`, as a pass) made the
  output projection and down slower, 18.37 to 19.41 us averaged over about
  14000 launches each, and moved nothing at the wall: taken out. The norm
  as a prologue with the matvecs split 16 ways, where q/k/v's 56 registers
  fit two 512-thread workgroups a multiprocessor: q/k/v and gate/up with
  their norms took 28.39 us a launch against 21.01 + 7.41 apart -- the
  launches go and none of the norm's time with them, its reduction and int8
  form run in series at the head of every workgroup -- and the 16-way split
  cost the output projection and down 18.36 to 22.33 us. Two ways remain
  open. The int8 codes do not depend on the norm's row scale, so the scale
  can multiply the matvec's output instead, and the reduction leave the
  path before the matvec. And which form wins depends on what the device's
  compiler allocates, registers and so waves: the tier measuring the
  composer's alternatives, as `tuneActIn` measures its own.
- **Divisions by a power of two.** At 32 registers and one wave the
  composer's q/k/v still took 12 us against the tier's 9.2, and its machine
  code was 1488 instructions against the tier's 856 (`cuobjdump -sass`,
  sm_70; the counters are an administrator's on that box): the same dot
  products and loads, three times the integer index arithmetic, and 13
  `MUFU`/`F2I` pairs -- whole integer divisions in the loop, by divisors the
  device compiler did not see as constants, though every one was a power
  of two. `Affine` takes an unsigned division by 2^n as a shift and a
  remainder as a mask: by site, down with its int8 form 28.9 to 23.9 us, the
  attention stage 10.2 to 7.8, each norm 8.2 to 6.5, the output projection
  11.9 to 10.1, q/k/v 12.4 to 11.2; gate/up 30.1 to 31.3, at 38 registers
  where it had 32. The output head went the other way, 261.8 to 291.6 us:
  the same four byte dots a trip, but 16 loads where it had 11, the same
  plane offsets read again from other base registers. The words of a trip
  were reached as (16i + c) >> 4 and ((16i + c) & 15 & 7) >> 2 for c of 0,
  4, 8 and 12 -- c = 0 and c = 8 one word, 4 and 12 another -- which the
  device compiler had proved equal through the divisions and could not
  through the shifts. `Affine` takes such bits apart itself: (a*C + c) &
  (2^n - 1) is c's bits where C is a multiple of 2^n, exactly, and the
  shift by n is a*(C >> n) for c below 2^n where a is a loop's iteration
  whose trips keep a*C + c below 2^32. The head's weight loads went from
  ten to five in the IR, its machine code from 128 instructions to 96 (144
  with the divisions), its registers to 27.
- **A value kept past its loop.** The validator's loop rule found a second
  shape of the bug above: a value no loop changes, first asked for inside a
  loop nested in another, was made inside the outer loop's body and kept
  after it closed. The lowering forgets, when any loop closes, what its
  body made, and makes it again where it is asked for. The rule holds a
  value to a loop of a run-time count only: the tier's tensor-core GEMMs
  read an MMA's accumulators after a loop of a fixed count, whose last
  trip's values stay readable, while a guard's count can be zero and leave
  nothing.
- **A launch bound from the grid.** The shifts left gate/up at 38
  registers: six of its 256-thread workgroups a multiprocessor, 480 places
  for its 512. A launch with more workgroups than the device has
  multiprocessors states how many each must hold at once for one wave
  (`ir.Kernel.Resident`, PTX `.maxntid` and `.minnctapersm`; the tier
  passes the multiprocessors as `jit.Device.Units`), and ptxas fits its
  registers to it: gate/up 32 registers, no spill, seven a multiprocessor.
  Unlike `.maxnreg 32` on every kernel, it asks only what the grid needs,
  and nothing of a launch that fits the device in one wave already.
- **A phase past its last unit.** The norm's int8 phase deals 256 units
  over a workgroup of 1024 threads, and the warps past the last unit
  computed clamped copies of it: about 450 instructions a thread
  (`cuobjdump -sass`, sm_70) in 32 warps for 8 warps' work. Where the units
  come in whole warps and the phase forms no exchange through workgroup
  slots, the phase is now its guard's -- the unit is the spread loop's
  iteration plus the guard's, so the lowering homes it there -- and those
  warps branch past it; `Prologue` does the same for a producer of threads.
  The norm is bound by its chain of latencies, not by issue, so its own
  time moved little (nvprof, 6.53-6.77 to 6.56 us), while the graphed
  decode's device time went 1886.5 to 1864.2 us a token (V100, CUDA,
  Llama-3.2-1B, `speed` tg128, three interleaved rounds).
- **The head formed inside the first token.** The tier's placement prewarm
  forms every composed block and the head, but a model prepares its head
  after placing its blocks, so the head formed, its capture abandoned,
  inside the first decode step: 20-37 ms of an 8-token decode's 38-51
  (main's 15). `PrepHead` forms it now, as it builds the tier's own head
  kernels: the 8-token decode 17 ms; 264 tokens 545-548 ms against main's
  476-485 (V100, CUDA, Llama-3.2-1B, `run`).
- **Exact division where the device has it.** The int8 form's scale
  amax/127 and inverse 1/d were the integer routines on every GPU, about
  100 instructions a thread, because one profile serves backends whose
  division is not the IEEE quotient (SPIR-V's OpFDiv is allowed 2.5 ulp).
  `jit.Device.ExactDiv` states it per device: PTX's div.rn and a CPU's
  division are the quotient the routines compute, and the tier states it
  from the backend (`kernels.DivExact`). Device time 1861.9 to 1821.8 us a
  token, same box and rounds.
- **An element read twice.** A window's amax and its codes reached each
  element of the norm's output through the host's value at two index
  expressions of one place, 33 loads where 17 do. The host's values are
  named by their place's value at every unit, as a member's are: 1821.8 to
  1795.7 us a token. Main took 1587.6 in the same rounds: the composer at
  0.884 of it in device time, tg128 519-527 tok/s against 585-592.
- **The norm as a prologue, again.** With the phase skipped by the warps
  past its last unit, the norm inside q/k/v and gate/up took 1878.8 us a
  token against 1867.5 apart; after exact division and single loads, 1778.7
  against 1799.7. The second is faster, but its greedy tokens part from the
  norm-apart form's at the sixth decode token, and the serving gate's worst
  distance to the float reference on Llama-3.2-1B rose from 0.00727 to
  0.0321 (main's tier 0.0137) while staying inside the gate's bound. Kept
  refused until that is explained: the reduction's order over gate/up's
  256-thread workgroups, or a defect in a reduction merged into a consumer.
- **What a composed step still launches of main's.** The serving gate holds
  every block placed, the device's fallback off, no demotion, and on the arm
  without a recording every block of every step served (a block bypassing
  the composed branch, neither served nor declined, reads 2100 block steps
  where 140 steps of 16 blocks are 2240). A census of the tier's own launches
  on the steps the composer is asked for (`Config.Census`, by the name each
  kernel was compiled under) reads, on Llama-3.2-1B: with the blocks
  composed, the head's two norm kernels, its quantize and its vocabulary
  matvec, and the rotary table, once a step each; with the blocks and the
  head, the rotary table alone. The gate holds the second to that list.
  Every form is built at a placement or PrepHead (`Stats.FormedAtStep` 0),
  and a decline is counted by its reason. Of the checkpoints the front end
  states, the tier cannot place stories260K (its quantizer refuses the
  172-wide rows), and float fixtures equal the float reference, which the
  gate's band cannot measure; the front-end test forms both.
- **A form's life.** A composed form's buffers were the device's without
  the budget knowing: allocated beside the blocks, freed only when the
  device closed, and a block placed again after a move formed at its first
  step. A form now charges its bytes when it builds (one the budget cannot
  hold is refused), counts the served parts that launch it and goes with the
  last of them when its block is released, and every placement of a block
  forms its part, a move's and a re-offer's too. On Llama-3.2-1B the forms of
  a whole placement are about 21.7 MB: walking the seam home and back six
  times charges the same budget at every whole placement and forms no part
  at a step (a release that does not refund reads 1076730144 bytes against
  1055018128 at the second cycle; a placement that does not form reads 16
  parts formed at a step). Poisoned scratch poisons the forms' workgroup
  arrays too, and the poisoned arm's logits do not move.
- **A form per device, an instance per lane.** A tier serving composed
  parts ran every submission exclusively, because a form's buffers and launch
  arguments were the device's and two sessions would have written them at
  once. That was a choice of this code, not of the device. Now a form's
  compiled kernels are the device's, one per composer Key. Each lane holds
  its own instance of the form: the values between its kernels, its scratch
  and its constants, charged to the budget. It also holds the bindings it
  rebinds every step. A clone lane takes an instance of every part lane0
  serves; every placement spreads the block's instance to every lane; a
  release drops it from every lane; and a lane dropped frees its own. Two
  sessions of stories15M on a composed laptop GPU step at once with ids and
  logits identical to each alone, 47 of 48 steps overlapped and 51-60
  submissions started beside another's. One session moving its seam twice and
  hopping to a second tier beside the other holds the same bar. Each lane
  holds an instance of every placed block and the head, and no part forms at
  a step. Five violations each fail a gate: the exclusive clause put back
  (serialised); a clone without instances (7 parts formed at a step); a
  placement that does not spread (lanes holding 7 and 4); a release in one
  lane only (lanes holding 4 and 7); and a shared head without its graph,
  where the clone's head declined to the tier's and its logits differ in the
  last bit. The memory cost is a whole set of instances per lane, about
  21.7 MB on Llama-3.2-1B beside the lane's scratch.
- **The principles' device gates, composed.** A jitllmcomposed build serves
  every tier's blocks and head through the composer (a constant per build),
  and the two-tier hop, the moves between cards and the batch beside a
  session carry a composed arm of their own on the models the front end
  states whole. Their first run found two things.
  - The prewarm bounds-checked a device's layers, which are a map keyed by
    block, like a slice. A card holding only the upper blocks (a second
    card, a move's target) skipped their forms, and the first step formed
    them: 11 parts at a step on tinyllama's upper-half move.
  - The family gates' violations mutate the model's config, rotary and
    loaded tensors, which main's device path reads at placement. The front
    end built its graphs once, from the container. Under the composed build
    the composer served the unmutated model, so a violation was invisible
    ("rotary on the NoPE layer" read 2.3e-12 against a clean 2.3e-12) or
    refused the form, as a dropped bias left a bound role unheld.
  Now the front end builds again for a model stated otherwise, and takes the
  rotary's pairing from the config, as main does. A test that drops, renames
  or resizes a stored tensor restates it to the composer (Model.restate), so
  a violation reaches the composer as a different graph and never as a hook
  in main's kernels. The three violations read 9.2e-02, 1.01 and 1.46, the
  same as main's path.
- **The launch census put a warm decode on the heap.** Counting the tier's
  own launches (Config.Census) read the census from the launcher and handed
  it to a counter whose mutex leaks its receiver. Escape analysis is not
  field-sensitive, so leaking that one field leaked the launcher's contents.
  Every local the launcher points at (the submission's error, the quantized
  source, the activation memo) then moved to the heap on every submission:
  four objects a token on main's own decode, census off, from the commit
  that added it. The jitllmcomposed sweep's allocation gate found it, where
  600 subtests failed that pass on main. The census now hangs on launchTo,
  which the launch already leaks two dereferences deep. There it takes only
  those locals' values, never their addresses, and `go build -gcflags=-m`
  moves none of them. TestDecodeDoesNotAllocate passes again on
  Llama-3.2-1B on every arm, default and composed builds alike.
- **Post-norms, scaled residuals and ungated MLPs as ops.** The front end
  refused a block with a norm over a sublayer's output and one whose
  residual is scaled. It also bound a pre-norm and a gate whether or not the
  container stored one, so a post-norm block (OLMo 2, EXAONE 4) and an
  ungated one (AFM) did not verify. A block now binds a pre-norm and a gate
  only where they are stored, and applies an RMSNorm to an output where
  that is stored. A scale on the residual becomes the add's own factor, and
  an ungated MLP is its activation over the up projection. A model whose
  blocks were stated already builds the same graph, op for op, so its forms
  and their keys do not move. The tier binds the post-norms main uploads.
  The front end now states OLMo-2-1B, EXAONE-4.0-1.2B, AFM-4.5B,
  granite-3.3-2b, granite-4.0-micro and rnj-1 whole. The serving gate runs
  the first five, granite's head held to its one named decline, a head
  graph past the norm and the projection, until the head takes its logit
  scale. rnj-1 forms and is not served: at one CUDA step its composed blocks
  reach 0.19 from the float reference, where both backends' tiers stay
  under 0.043. That gap is open.
  gemma-3-1b and the OLMo 2, arcee and EXAONE fixtures form; their sliding
  layers are the tier's until it serves windows. A failed form now carries
  its cause to every later step. The first OLMo 2 run read only "did not
  form earlier", and the cause was a binding missing from the tier.
- **Partial rotaries, a q norm after the rotary and windows past the
  session.** A rotary over part of each head forms as one over the whole:
  the rotation turns the first NRot elements and copies the rest. A q norm
  after the rotary keeps k's before it, as main runs Hunyuan: the converter
  folds k's weight into q's, and an unweighted norm commutes with the
  rotation. A window at least the session's length cuts nothing (Phi-3.5's
  262144 against its 131072 context), so the tier serves such a block. This
  states Hunyuan-0.5B and Phi-3.5-mini and forms the glm4 fixture. A rotary
  over several position axes (M-RoPE's sections, Qwen3-VL's interleaving,
  XD) stays uncovered by name until the composer's table holds several
  positions a row.
- **The serving gate's band is the tier's backends' spread.** Held to the
  tier on its own backend, OLMo-2-0425-1B and EXAONE-4.0-1.2B failed on
  CUDA and Vulkan, yet against the float reference they sat at the tier's
  distance at the median step: 0.96-1.06 of it. Step by step, OLMo 2 is
  quiet (2e-3) everywhere but its fifth teacher-forced step. There the
  tier is 0.167 from the float reference on CUDA and 0.021 on Vulkan, and
  main's two backends are 0.118 apart: a near tie each implementation
  leaves on one side or the other. The composed path left it at 0.21 and
  0.16. EXAONE is noisy throughout on both paths, and at its worst CUDA
  step the composed path is nearer the reference than the tier. The gate
  now measures the tier against the reference on every device first. It
  holds each arm's worst step to four times the widest of those backends,
  and keeps the median at 1.5 times. The one-position-off check asks for
  ten times the arm's worst rather than a hundred, which still sets a wrong
  part (NMSE 4-6) an order of magnitude past any arm.
- **rnj-1's 43rd step reads as a near tie.** Measured on one V100 under
  CUDA and Vulkan, over the gate's 140 teacher-forced steps. rnj-1's last
  block composed alone moves the logits 3e-14 from the tier's at the median
  step. Over the other 139 steps the composed CUDA blocks sit nearer the
  float reference than CUDA's tier (mean NMSE 1.2e-3 against 1.4e-3). Step
  43 decodes a flat distribution, its logits' squared norm 2.1e5 against
  6.4e6 at the median step. There the int8 rounding of a single block,
  every other block float, moves the logits 0.016 from the reference, 1800
  times that block's median effect, and every int8 arrangement lands
  somewhere else. Main's CUDA tier lands at 0.029, and at 0.079 at one lane
  (WithLanes(1)); the Vulkan tier at 0.0023, and 0.022 at one lane. The
  composed blocks land at 0.19 on CUDA and 0.032 on Vulkan, and composing a
  quarter or a half of the blocks beside the tier's lands anywhere from
  0.0017 to 0.21. The two float references stay 6e-4 apart. The gate holds
  an arm's worst step to four times the widest of the two tiers' distances,
  0.0425 at step 91, so rnj-1 fails it on CUDA by 11% and stays in
  formedModels. One alternative is untested: CUDA's split attention.
  SplitRange cuts a step's keys into max(1, ceil(keys/chunk), min(sq,
  capped)) partitions (jit/geometry.go): sq is the square root of keys/8
  rounded to a power of two, capped the splits the device's slots hold at
  1280 threads a query-head group (1 where the device reports no slots, as
  the V100's Vulkan does), and chunk the geometry's partition, 896 positions
  for rnj-1 at 32 lanes. At the gate's key bound of 256, CUDA cuts rnj-1's
  attention 8 ways and Vulkan does not cut it, so only the CUDA arms run the
  split. Step 43 on CUDA at 32 lanes with SplitRange off, against on,
  decides it; until then the gap is open. The spikes do not recur at one key
  count modulo 16, 32 or 64. The matvecs' forms are the same on both
  backends at K 4096 and 16384; only the attention and the post-norms' sums
  differ. The YaRN scale is applied once: the composed rotary reads the
  tier's table, never the op's Mscale. Composed blocks at one lane
  (WithLanes(1)) leave every step's logits at NMSE 1.0 where the attention
  splits: rnj-1 and Llama-3.2-1B under the V100's CUDA, and stories15M on a
  laptop's RTX 3050 Ti, one block enough on the last two. The split
  attention's output reaches 4e30 there while q, k and v are right, and
  turning SplitRange off restores it. At one lane the chunk is 504
  positions for rnj-1 and 1016 for stories15M, so a device that reports no
  slots splits only a session whose key bound passes it: the V100's Vulkan
  at the gate's 256 did not split, and its one-lane composed runs were
  sound. The tier at one lane is sound: open (open-questions.md).
- **The node-affine allocation arm named the BitNet fixtures' size.** On the
  two-socket box `TestDecodeDoesNotAllocate/.../host-numa` failed on
  synth-bitnet-f16 and synth-bitnet-i2s (`0 region(s) node-affine`) at the
  base commit, in the default and the composed build alike. Neither the
  composer nor the ternary path is at fault: BitNet's weights reach the same
  fused matvec as every packed format (1344 of them in a window), and DoNodes
  places only a weight whose rows are a whole number of rounds of 1024 rows a
  node, 2048 on two nodes, where the fixtures' have 128 to 512. The arm now
  skips such a dense model naming its row counts, and
  `nn.TestFusedMatVecRunsNodeAffine` holds all eleven fused formats, TQ2_0 and
  TQ2_F32 included, to the node-affine path and the plain pool's bits. The
  engine only names the rule `numaRows` already applied (`nn.NUMARound`):
  Llama-3.2-1B, Qwen3-8B and OLMoE read the same allocations (0) and
  node-affine regions (5120, 11520, 6094 to 6144) before and after. The
  details are in cpu-kernels.md, "Two gates that failed only on the Xeon".
- **Windows served.** A block whose attention slides, or attends an aligned
  chunk, now serves. Its attention reads each row's range from the windowed
  descriptors the tier already stages for its own windowed kernels (pRowW,
  the window's start their first position), through the same page table,
  which past windowPages holds only the window's pages. The composer no
  longer computes the window itself. Under the paged gate's "window" fault (a
  windowed layer reads the full descriptors), a start computed in the
  kernel as well hid the lost window: the composed arm sat 0.000e+00 from
  the clean one. Reading the descriptor alone, it sits 8.6e-02 away. The
  tier serves the block where the graph's window is the plan's, or where the
  window is at least the session's and cuts nothing; any other graph window
  is declined by name. The sliding-window gates changed the config in place
  under graphs built for another window, which that decline caught; they now
  restate it as a copy. A captured graph's replays of the parts are counted
  (Stats.PartsReplayed). The window gates now hold the composer to every
  block of every one-row step, recording on: gemma-3-1b at 712 positions,
  18512 of 18512 block steps, 2.3e-04 from the host on CUDA and Vulkan
  against a window effect of 0.21. synth-exaone4 and synth-olmo3 ran 6112
  of 6112 steps each over 1528 positions, their releases exact (15 pages)
  and their logits bit-equal to the keep-every-page control. A larger second
  session on one tier now forms its parts when the scratch grows
  (ensureKVCap), not inside its first step. Off-by-one windows are device
  violations on both fixtures (one key wider, one narrower), and the release
  gate now also loses the window outright. On the composed path the logits
  move at step 4 with the window lost and at step 259 with the window's own
  page released. On the fixtures' window of 4, a window one key wider moves
  the logits by NMSE 0.89 to 1.38, and one key narrower by 0.91 to 1.56,
  against a clean 2.7e-12. The serving gate now decodes past a window the
  context can outrun: gemma-3-1b by 70 positions over 576 steps. It counts
  every block of every step, replayed or emitted, and serves gemma-3-1b on
  CUDA and Vulkan in every arm. The median ratios to the tier's distance are
  1.046 and 1.099; the worst steps are 0.0177 and 0.0321, against the bound of
  0.059. gemma-3-4b, 1088 steps, each backend in a run of its own, reads
  1.006 and 1.022, worst 0.0123 and 0.0063, against bounds of 0.070 and
  0.021. gemma-2-2b, 4160 steps (70 past its 4096 window), each backend in a
  run of its own, reads 0.978 and 1.003, worst 0.057 and 0.045, against
  bounds of 0.21 and 0.23. Its final-softcap head is held to its named
  decline, as granite's is. The paged fault gate pins its arms to the tier's kernels, whose violations it holds: in a
  jitllmcomposed build eight of its ten cases had read 0.000e+00, since the
  composer served the decode they break. The cost estimator prices a
  windowed attention over its whole history, an upper bound, because a step
  states no window.
- **The classic transformer kit as ops.** The front end refused every block
  of a LayerNorm model or a parallel one (UncoveredClassic), and the tier held
  no vector for a LayerNorm's bias. A norm before a sublayer is now a
  LayerNorm with its stored bias where the model's norms are mean-centred,
  and an RMSNorm elsewhere; the norms after a sublayer and the q/k norms stay
  RMS, as main runs them. The verifier holds a LayerNorm's second weight to a
  vector of the row's width. A parallel block's MLP reads the block's input:
  through its own norm where the container stores one (Falcon-40B's), through
  the attention's where the two share one (phi-2, Falcon-7B, Command-R). Both
  outputs add into the residual in main's order. The tier binds the two norm
  biases. The front end now states phi-2, StarCoder, StarCoder2, StableLM
  (serial and parallel), Falcon, Falcon-40B, Nemotron, Command-R and Cohere2
  whole, and Falcon-7B-instruct; DBRX stays a mixture. What is not composed
  yet is missing implementation, each with its owner, and the tier's own
  kernels running it meanwhile is not a resting state:
    - The LayerNorm head of all of them (A4): the front end refuses it
      (UncoveredClassic), so the tier declines "a head graph other than a norm
      and its projection". The front test pins each such model to that refusal,
      and fails when a head becomes stated until the entry goes.
    - The input rows part (nobody owns it yet): the tier serves blocks and
      heads only, so StarCoder's learned position table, like every model's
      token lookup, is done by main's runtime. The front end states it
      (a positional OpEmbed) and the composer forms and lowers it.
    - Cohere2's three sliding layers, declined by "a window" (the tier does
      not serve a window shorter than the session), and DBRX's mixture (unowned).
    - The CPU composer's norms, below (unowned).
  - Served on a V100, CUDA and Vulkan, each fixture's blocks at every decode
    step (3 blocks over 9 steps is 27 block steps; Cohere2's three sliding
    layers are declined by the one named reason, a window, and its global
    layer served, 9 of 36), the logits within 4.6e-14 to 6.7e-12 of the
    host's. Each feature taken out of the device session alone, as the
    composer reads the container too, moves them by 1.2e-4 to 2.5: the norm's
    bias, the FFN's biases, the centring, the parallel residual, one norm for
    both branches, the two norms swapped, the position table. Falcon-7B-instruct
    Q4_K_M (4544 wide, 71 heads on one kv head) is served in every arm of
    the serving gate on both backends: 4480 block steps, a median 1.037 (CUDA)
    and 1.044 (Vulkan) of the tier's distance from the float reference, the
    worst step 4.2e-05 and 4.5e-05 where the tier reaches 4.7e-05 and
    6.5e-05. Its head is held to the named decline, 140 steps.
  - The first Falcon-7B run found a bug in the tier, not in the composer. The
    head folded into a prompt's last chunk (a quantized head over every
    block) launched the RMS norm's two kernels for a LayerNorm model. On CUDA
    the prompt's 65024 logits read zero (a launch is not checked for the
    count of its buffers); on Vulkan the apply kernel was refused with 4
    buffers where it wants 6, and the chunk fell to a row at a time. The fold
    now runs the mean, the variance about it and the apply with the bias, as
    the one-row head does. TestPrefillFoldsALayerNormHead (phi-2 with a q8
    head) fails on Vulkan without it.
  - TestComposedGPULayerNorm executes the LayerNorm on every backend, with
    and without a bias, at widths of 1 to 4544 and up to 9 rows, with a guard
    of poisoned words after the result, against the oracle and against three
    violations (the scale's or the bias's words moved by one, the RMS norm of
    the same shape): 49 shapes a backend, 1.5 s on a V100's CUDA and 101 s on
    its Vulkan, where the 230-shape sweep it was cut from took 35 s and
    1550 s, and passed on the V100's and an RTX 3050 Ti's two backends.
    TestComposedCPULayerNorm does the same on the CPU, and
    finds the CPU composer's norms short, RMS norm included: of 230 shapes
    its RMS norm is right on 86; the LayerNorm is right on all 24 of those at
    one element or whole vectors and on 40 of 62 ragged widths. The fused
    reduce-then-map reads a ragged tail wrong, a row of 2 or more at 8 wide
    or more is wrong for the RMS norm, and the register binder panics on a
    one-row norm 10 to 15 wide (`l.Of - 1` of a vector tail no split made).
    The test lists those shapes with their numbers as the CPU's work list and
    asserts the rest.
- **A rotary over several position axes.** The composer's rotary table
  holds four positions a row (`rope.pos.axes`: the live row count, then
  axis a of row r at word 1+4r+a) and each pair's axis in a block the plan
  carries (`rope.axes`), derived from the table's stated sections:
  contiguous runs that cover every pair (Qwen2-VL's and Qwen2.5-VL's
  M-RoPE), or Qwen3-VL's interleaving pair by pair. A pair turns by its
  axis's position at its own frequency, so the planes are the one-axis
  table's. A split whose pairs all turn by one axis reads that axis's
  position and carries no block. The front end states the sections and the
  interleaving on the rotary op and the table. A rotary they do not state
  (XD-RoPE, whose pair halves turn apart) stays UncoveredRopeAxes, held on
  the hunyuanvl fixture.
  TestComposedRopeTableOverAxes runs the table at 1 to 64 pairs, one axis,
  contiguous and interleaved, on ptx and spirv at one row and at five, on an
  RTX 3050 Ti and a V100. It comes within 7.5e-9 of main's host table
  (nn.Rope.TableAt) and 6.0e-8 of the float64 table, each pair equal to the
  one-axis table at its axis's position to the bit, and two axes swapped
  move it by 1.6 to 2.0. A wrong interleaving rule fails every interleaved
  case. On the CPU at one row, its table convention's (one row a call,
  which gained the input fields such a table reads), each table equals
  main's host table to the bit, with no argument at an offset a caller
  would compute.
  No served step runs that table: a served block turns by main's table,
  and a decode row's axes agree. The serving gate serves Qwen2-VL-2B,
  Qwen2.5-VL-3B and Qwen3-VL-2B's Q4_K_M on CUDA and Vulkan, every arm, at
  text positions. Their median ratios to the tier's distance from the float
  reference are 1.007 and 1.073, 1.030 and 1.024, and 1.025 and 1.026 (CUDA,
  then Vulkan). Qwen3-VL-2B's f16 is formed, not served: its blocks and
  head run their float form and equal the float reference, ratio 0.
  TestMRopeMatchesTransformers now also decodes after the picture with the
  composer serving, the rows off their cache positions. Each step serves
  every stated block (qwen2vl's, qwen2.5vl's, qwen3vl's and glm4v's all,
  glm4vmoe's dense lead) and declines the rest by name: a mixture of experts
  on qwen3vlmoe and glm4vmoe, a gate on the attention's output and a
  recurrent mixer on qwen35vl.
  Qwen3-VL's f16 first failed every serving arm, and not on the rotary. Its
  float head does not fold into a batched chunk, so the prompt's last-row
  head was a one-row call of its own, and the tier served it as a step's.
  The engine now tells the device when its calls are a prompt's
  (nn.PromptDevice: a chunked prompt's calls and a batch slot's prompt), and
  the tier serves none of a prompt's parts. TestComposedPromptIsNotServed
  counts it on a float-head fixture, three rows and nine: six decode steps
  serve the head six times with the prompt's call unasked, and with the
  prompt kept from the device, seven.
  Two x86 gaps are open, and neither is the multi-axis table's. A one-axis
  CPU table at 8k+1 pairs (9, 17 and on to 129) does not lower: its vector
  loop's one-trip tail, flattened, holds more than the registers ("a
  constant held across the loops on a register a loop's value is hinted
  to"), which the register pass declines without saying so. engine/cost's
  CPU plans hold a reduce kernel the same pass declines ("a pair's later
  head with no twin"). Naming the decline at the pass fails that placement
  test, so the pass is left as it was.
- **The shadow arm.** `tier.WithComposedShadow()` runs the composer's form of
  each block it would serve beside the tier's own kernels, on a copy of the
  block's input, at every one-row step, issued or replayed from a recording.
  A comparison the composer forms from generic ops (`graph.Drift`: 14
  kernels from a map, two reductions and a slice, no hand-written kernel and
  no Go loop over a device buffer) writes the squared distance between the
  two outputs over the squared distance the block moved its input into a
  two-float slot of the block, its worst and a count. `Stats.ShadowWorst[li]`
  and `ShadowSteps[li]` read it back. The tier's kernels serve, so the stream
  and the logits are the tier's whatever the form computes.
  - The history is decided by order, not by a scratch row. The form runs
    first and appends its key and value to the page the step writes. The
    tier's kernels then write the same row, and theirs is the one kept. The
    plan's order (the tier first, the form after) would leave the form's row
    in the history, and the form's rounding would reach every later step
    the tier takes. A scratch row would let the form read the tier's key at
    the current position and never its own, so a wrong key projection could
    not show. Run first, the form attends over the rows the tier wrote and
    its own at this position. Run the other way, as a throwaway variant, the
    gate fails at the second decode step: on Vulkan in every clean arm, the
    form's rows differing from the tier's in the last bit; on CUDA, where the
    form's key and value rows come out bit for bit the tier's, only in the
    arm that puts the key projection on the value's weights.
  - Buffers are made at the placement: a block's two-float state, two copies
    of the stream in the decode scratch, the form and the comparison at
    the prewarm. A warm decode makes no engine allocation over 64 tokens,
    graph and no graph, CUDA and Vulkan, shadowed or not; one heap
    allocation added to the comparison fails the shadowed arms with 1028 to
    1032 objects over the 64 tokens, the stack at the comparison, and not the
    unshadowed ones (CUDA's replayed arm runs no Go emit code and cannot see
    it, the no-graph arm can). Across six cycles of moves the card's free
    memory holds at 14667 MiB shadowed against 14675 without (8 MiB, the
    driver's rounding of the extra buffers).
  - `TestComposedShadowHoldsEveryBlock`, on a V100 (CUDA and Vulkan), over
    Llama-3.2-1B, Qwen3-0.6B, gemma-2b and OLMo-2-1B, 140 teacher-forced
    steps each. The logits equal the tier's bit for bit at every step, with
    the launches issued one at a time and with CUDA replaying a recording
    (2 recordings, and the device counted 140 comparisons a block over the
    140 steps); served alone the same blocks change them at 140 of 141 steps,
    so the equality can see a difference. The band is the tier's own distance
    from the float form, block by block, on its two backends, with
    TestComposedBlocksServe's constants (4.0 against the widest backend, 1.5
    at the median block). The int8 forms sit at a median of 0.007 to 0.157
    of the tier's own distance and at most 0.41 at the worst block, on both
    backends, at every depth. Rewriting block k's graph (another activation,
    a 32-position window, a doubled attention scale, the rotary paired the
    other way, the key projection on the value's weights) is flagged at k
    alone, on all 40 arms, 11x to 1769x past the bound and never before the
    step where the rewrite can first differ: the window at step 26 or 27,
    exactly where it first cuts a key, every other arm at step 0 but
    OLMo-2's scale and rotary arms, at step 1.
  - With the option off, TestComposedBlocksServe on Llama-3.2-1B, Qwen3-0.6B
    and OLMo-2-1B prints the same worst NMSE from the tier and the same
    median against the float reference as the base commit's own run, line for
    line.
- **A narrower subgroup than 32 lanes.** Composed blocks on a device that
  does not guarantee 32 lanes (WithLanes(1)) read NMSE 1.0 from the first
  step: stories15M on the laptop's RTX 3050 Ti, rnj-1 and Llama-3.2-1B on
  the V100. At one lane a warp is a thread, so Bind's spread had no
  butterfly, and a thread's partial was the fold itself. The redirect that
  points the fold's readers at the workgroup total also rewrote the store
  that publishes the partial, so every slot held a sum of unwritten slots:
  each partition's maximum and sum were stale workgroup memory (2.019, the
  attention norm's third element). The redirect now runs before the slots'
  stores join. Bind also left two loops for every thread to run whole at
  one lane: the values over a head 48 wide (warpsWide wanted it no wider
  than the workgroup's 8 threads) and the 32 copies of each maximum and
  sum (lanes wanted the 8 threads to divide 32). The combine after the
  values is now a loop over the head, and the copies are dealt a trip a
  thread. With that, the block still read NMSE 1-15 at one and 8 lanes,
  on CUDA alone (Vulkan reports no slots and never splits); turning off
  Unroll or Runs alone restored it. Runs takes a run's page and place in
  the page from its first position held to the range's last aligned run,
  which is harmless only for a run of whole 32-position blocks, as 32
  lanes make it: a one-lane run in the last partial block read the block's
  first key, and at 8 lanes a trip of 24 positions crossed a page. Runs now
  rewrites only a run inside one block of the alignment, and the place in
  the page only for one of whole blocks. Three other places took 32 lanes
  for granted. The lowering's group reduction butterflied over 32 lanes
  whatever the device stated. Coarsen and Prologue took "whole warps" to
  mean whole subgroups of the device's lanes, which at 8 lanes formed a
  shuffling stage of 48 threads the IR refuses, and a guard skipped by
  part of a 32-lane warp around a shuffle, which hung the laptop's GPU
  and, unfixed, a V100 until the test's 15-minute timeout. The IR's
  shuffles name every lane of a 32-lane subgroup, so a device guaranteeing
  fewer lanes is now given none (Device.warp): its reductions go through
  workgroup memory.
  TestComposedBlocksAtEachLaneCount holds blocks and head composed at 1, 8
  and 32 lanes, and with the whole tier at one lane, to the serving gate's
  band against the tier on the same device, poisoned and unpoisoned.
  Unfixed, it read 447 times the tier's distance at one lane on the laptop
  and 508 times on the V100, and at 8 lanes the stage did not lower.
  TestFrontStatesTheModelAsOps lowers every served model at 32, 8 and 1
  lanes and refuses any op that depends on a subgroup below 32, and
  TestNarrowSubgroupsShuffleNothing does the same for a norm and three
  split attentions without a model. At the device's own 32 lanes the 17
  served models' lowered kernels are byte-identical to before (2 device
  descriptions, 3 steps), and no keyed field changed, so their keys stay.
- **rnj-1's step 43 is the split's reduction order.** On a V100 at 32
  lanes, rnj-1's composed blocks read 0.189 from the float reference at
  teacher-forced step 43 with SplitRange on, and 0.0325 with it off, every
  other knob pinned; the CUDA tier reads 0.029 there and Vulkan's composed
  blocks 0.032. Vulkan reports no slots, so its attention never splits: its
  logits with SplitRange on and off are equal at all 141 steps. The float
  reference does not move with the split (0.0000 at step 43, 2e-13 at the
  median step), so the split's arithmetic is the unsplit's to f32 rounding,
  and what differs is where its reduction order leaves the int8
  activations at a step every implementation finds sensitive (the tier's
  own distance there is a hundred times its median).
- **A batch step's room is the room it takes, and relocation makes it.**
  The relocate-then-reclaim arms on synth-deepseek32, synth-minimaxm3 and
  synth-qwen4exp relocated nothing, on the V100 (CUDA and Vulkan) and the
  laptop alike. A 3-row step's own scratch did not fit the squeezed card,
  so the step padded into the 512-row prompt scratch, which RowsRoom
  counted as room, and then built that width's per-row head: 512 rows of
  logits, 265 MB at deepseek32's vocabulary on a 103 MB budget, charged
  unchecked (369 MB charged; minimaxm3 533 MB on 122 MB). RowsRoom now
  builds the step's scratch at the width the step runs (the grain was 8
  where the step takes 32), prices the per-row head before asking for it,
  and holds the whole to the budget as a width built after placement is.
  A step that would pass it has no room, and relocateUntil hands a block
  home; the model asks with the head the step will hand the device. A
  scratch that does not build (a kernel the device refuses, the head not
  on it) is not a shortage: RowsRoom returns it as an error and the step
  is refused by name, as the device refusing it was before, where reading
  it as no room sent every block home and ran the batch on the host
  (TestRoomShortagesAreSurvived's build-refused arm and its violation).
  The step's walk of every block's ragged twins is kept on layerGen, as
  prepBatch's is, so a warm step's room query walks nothing; the decode
  retune now moves layerGen, since it rewrites the blocks' matvecs. A warm
  3-row step on Llama-3.2-1B Q4_K_M, every block on a V100 over CUDA, 40
  steps a sample, process per sample, interleaved: the fix against the
  base reads 0.9917 (IQR 0.9745-1.0049, n=24; the base against itself
  1.0004, IQR/median 0.069, n=8), 2.8 ms a step and two allocations a
  ForwardBatch in both.
  The default path changes in three more places, each a charge past the
  budget the arm's new check found. A block that selects from its whole
  history held its context at placement, but its floor stayed the working
  set, so the first compaction gave the context back (393216 bytes on
  deepseek32, the slack the rows' first pages then fit in); such a layer
  is now floored at its context. A lone matvec and a routed compact bank
  were admitted at their bytes rather than their footprint, as blocks
  already are (2240 bytes over on minimaxm3); both are admitted at the
  footprint. And a device step that wants no logits runs no head
  (forwardRowsDevice handed rowsStep nil logits, which reads as greedy).
  Blocks placed, before and after, on the V100's CUDA: on the whole card
  the same for Llama-3.2-1B (16 of 16), synth-qwen4exp (4 of 4) and
  synth-deepseek32 (3 of 3), a sequence and a 3-row batch alike; at 60% of
  a sequence's attach Llama-3.2-1B places the same (7, and 6 for a batch),
  synth-deepseek32 one more, a sequence and a batch (2 of 3 where it placed
  1), and synth-qwen4exp's batch one fewer (1 of 4 where it placed 2), each
  after its first prompt or four steps.
  Each of the three arms now hands one block home and reclaims it
  (deepseek32 2.1e-06 after the move, minimaxm3 6.4e-12, qwen4exp 1.4e-11),
  on the V100's CUDA and Vulkan in both builds and on the laptop. Two other
  arms under the same squeeze, synth-gemma4-kv and synth-hunyuan, hand one
  block more home than before, the per-row head now held to the budget;
  every other arm's figures are unchanged to the third digit.
- **Prompt chunks served at their rows.** A Config.Composed bit,
  ComposedPrompt, asks the composer for each block of a prompt chunk at
  the chunk's width: one sequence's rows, each its own descriptor, the
  step's row offsets bound as the query's. A form's key holds its rows, so
  a block forms at every width the device warms, as main keeps a kernel a
  width: the placement's width loop and prepBatch form them. An R-row
  form's buffers live in its width's scratch, shared by the blocks of the
  form and freed with the scratch; one the budget does not hold declines
  by name. A windowed block's chunk reads the windowed descriptors main's
  prefill stages, a row each (windowDesc), never the full history. A
  prompt's one-row calls (nn.PromptDevice) are the prompt's: a call's
  blocks are served at its row and counted as the prompt's, never as a
  step, and its head is declined by name; without ComposedPrompt a
  prompt's calls ask for nothing, as before. No gate yet reaches a one-row
  call with blocks (a chunked prompt's last row alone, 513 tokens at the
  512-row width); the float head's one-row call is
  TestComposedPromptIsNotServed's. A ragged step's rows and speculative rows
  wanting every row's logits stay on the tier's kernels by name, and so does
  a head over several rows: a chunk's folded head, a ragged step's,
  speculative rows', each declined by its path's name where they had
  counted as unasked. The shadow arm compares a decode step's forms alone:
  a prompt there is the tier's, declined by name. A narrowed subgroup
  (Config.ComposedLanes) reaches the row forms through the one request.
  On sm_70 a chunk folds a block's residual add into the next block's
  attention norm (foldAddNorm), asking foldTakes, which did not know the
  next block was served: the served block ran without the add and the held
  sum then overwrote its output. A held add now runs alone before a served
  block. TestPrefillMatchesForward's device arms hold a Prefill to a
  Forward loop on one tier. On a V100's CUDA card, with every other block
  kept on the tier's kernels (Config.TierBlocks), the last row is 1.05e-3
  from the loop and 16 decoded steps agree; without the rule it is 2.67.
  Every block served reads 1.81e-3, the tier's own kernels 2.30e-3. On
  gemma-3-1b over a 608-token prompt, 96 positions past its 512 window in
  64-token chunks, the composer reads 5.5e-5 (the tier 7.8e-5); its prompt
  rows bound to the full history instead read 1.6e-2, past the 1e-2 bar.
  Measured on Llama-3.2-1B before building: an R-row form bound the int8
  sub-block dots, {R, rows, K/32} int32 that a pass keeps in registers and
  no kernel reads or writes: 300 MB of a 309 MB instance at 32 rows and
  2.5 GB of 2.64 GB at 512, a block. A binding no kernel reads or writes
  now gets a word. A block's instance is then 7.8 MB at one row (17.1
  before), 9.0 MB at 32 rows, 17 at 64, 34 at 128, 68 at 256 and 136 at 512
  over 256 keys. The attention's dots, scores and weights over {R, H, Keys}
  are 48 MB of those 136, 96 MB over 512 keys; over 4096 keys they are
  768 MB and over Llama 3.2's 131072 keys 24 GB, so a staged attention
  comes before a real context's prompt is served. The gates bound the
  widths they warm (Config.PromptRows).
  Forming a row form is quadratic while cse points every reader at a
  merged node in a walk of the kernel per merge: 78% of a 64-row form, 140 s
  for a Llama-3.2-1B block's one- and 64-row forms on the laptop. With cse
  merging in one walk a block forms on a V100's host in 0.2 s at one row,
  0.9 s at 32 rows, 6.5 s at 64, 8.1 s at 128, 11 s at 256 and 16 s at 512,
  so a placement warming every width spends minutes forming (643 s for the
  prompt-compile gate's eight widths). The projections at a chunk's rows
  are single kernels of up to 25,000 ops in the int8 form and 104,000 in
  the float one: the float form of a 32-row Llama block is a 2.7 MB SPIR-V
  module that NVIDIA's Vulkan driver compiled for over 21 minutes, past
  the serving gate's two hours. The row forms want the tiled projections
  main runs before they are fast. Until then the serving gate holds the
  composed prompt on three Q8_0 models and the windowed gemma-3-1b
  (promptModels), every block of the chunk served and its last row in the
  band, and the other served models' prompts to the tier's own logits, as
  before. The change moves nothing to the heap in jit/gpu/tier or
  engine/model (`go build -gcflags=-m`, 96 lines before and after), and a
  warm decode token with the prompt's row forms placed allocates nothing:
  TestDecodeDoesNotAllocate on Llama-3.2-1B with ComposedPrompt on, 0
  engine allocations over 64 tokens on a V100's CUDA and Vulkan, with and
  without the captured graph. The prompt chunk itself is not measured.
- **The attention output gate, xIELU and the norms inside a sublayer as
  ops.** The front end refused three block features by name. A gate on the
  attention's output (the q projection twice as wide, each head's query then
  that head's gate) is two per-head slices of that projection and the
  attention's output times the sigmoid of the gate before the output
  projection, as main's `applyOutGate` has it. xIELU (four numbers a block,
  folded at conversion into `jlm.RoleXIELU`) is maps over the up projection,
  each reading one number of the stored vector, which a map's bound vector now
  allows (`Attrs.Offset` and `Width` window it, held to the vector by the
  verifier), and one new generic map, `MapExpm1`: a Taylor sum within |x| <
  1/2, exp(x) - 1 past it, where exp(x) - 1 near zero would cancel every
  digit. A norm inside a sublayer (BitNet's, before the output projection and
  before down) is the RMSNorm op on the attention's output and on the FFN's
  product. The tier binds `RoleAttnSubNorm`, `RoleFFNSubNorm` and
  `RoleXIELU` to the buffers main uploads. BitNet also quantizes every
  projection input per token, which the op language does not state, so the
  front end refuses its model by a new reason, `UncoveredTokenAct`, and no
  block of it is served. `UncoveredOutGate` and `UncoveredSubNorm` go, and a
  rotary with an interleaving but no sections (qwen3next's) is one position
  axis, no longer `UncoveredRopeAxes`.
  - On card 7 in a composed build, cuda:0 and vulkan:7, every composed arm
    asserts the composer served steps x the blocks the front end states, and
    declined every other block step by name. synth-apertus is served whole:
    36 block steps in 12 steps of 3 blocks, its logits 2.5e-12 from the
    host's in NMSE at decode, and the eight violations on its xIELU and
    rotary read 3e-4 to 1.9. The qwen3next-shaped hybrid's attention block is
    served (8 block steps in 8) and its three recurrent blocks declined by
    kind (24), its mixture variants' attention blocks by `UncoveredMixture`.
    The converted qwen35 fixture's attention blocks stay declined by their
    rotary sections and its recurrent ones by kind; the four BitNet fixtures
    by `UncoveredTokenAct` (24 or 36 declined block steps in 12). Of the 25
    classic-kit and dense fixtures 7 are served whole, 3 have sliding layers
    declined for their window, and 15 are declined by the front end.
- **The attention output gate and xIELU as ops; the norms inside a sublayer
  stated, and reached by no container.** The front end refused three block
  features by name. A gate on the attention's output (the q projection twice
  as wide, each head's query then that head's gate) is two per-head slices of
  that projection and the attention's output times the sigmoid of the gate
  before the output projection, as main's `applyOutGate` has it. xIELU (four
  numbers a block, folded at conversion into `jlm.RoleXIELU`) is maps over the
  up projection, each reading one number of the stored vector, which a map's
  bound vector now allows (`Attrs.Offset` and `Width` window it, held to the
  vector by the verifier), and one new generic map, `MapExpm1`: a Taylor sum
  within |x| < 1/2, exp(x) - 1 past it, where exp(x) - 1 near zero would
  cancel every digit. `UncoveredOutGate` goes. The rotary of the qwen3next
  shape (an interleaving, no sections) is one position axis and the converted
  qwen35's, over its sections, is the multi-axis table's, so both attention
  blocks are stated (`TestFrontStatesTheQwen35FixturesAttention`).
  - The norms inside a sublayer (BitNet's, before the output projection and
    before down) are stated too, as the RMSNorm op on the attention's output
    and on the FFN's product, and the tier binds `RoleAttnSubNorm` and
    `RoleFFNSubNorm`. **No container reaches that statement.** A model stores
    those norms only where it quantizes every projection input per token
    (`model.go` refuses one without the other), the op language does not state
    that quantization, and the front end refuses it first, by a new reason,
    `UncoveredTokenAct`; `UncoveredSubNorm` goes with the statement it named.
    So no block of any model is served through the sub-norm ops or the two
    bindings, and no device gate runs them. They stand on a CPU gate over a
    hand-built graph (`TestComposedCPUSubNorm`) and a structure test with the
    flag taken from the config (`TestFrontStatesTheNormsInsideASublayer`);
    the device gate belongs to the change that states the quantization.
  - On card 7 in a composed build, cuda:0 and vulkan:7, on the tree rebased
    onto the classic kit, the windowed blocks and the multi-axis rotary, every
    composed arm asserts the composer ran steps x the blocks the front end
    states, emitted or replayed from a recording (`composedCovers`, on the
    counter `composedBlocksRan` uses), and declined every other block step for
    having no graph and nothing else, exactly where the run issues every step
    itself. The tier serves a sliding layer and declines only a graph whose
    window is not the plan's, so the check takes no window decline: an
    earlier version of it let any sliding model decline any window. The 25
    classic-kit and dense fixtures of `TestC6OnEveryDevice` (76 passes): 20
    are served whole, the ten classic ones (phi2, starcoder, stablelm,
    stablelm-par, starcoder2, falcon, falcon40, nemotron, commandr, cohere2),
    synth-apertus, -smollm3, -arcee, -seedoss, -olmo2, -olmo2-gqa, -glm4 and
    the three with sliding layers (exaone4, olmo3, olmo3-af: 48 block steps in
    12, the sliding ones included); five are declined by the front end,
    synth-dbrx for a mixture, synth-mistral3 and -mistral3-af for an attention
    temperature per position, synth-deepseek-yarn and -yarn-af for latent
    attention. synth-apertus: 36 block steps in 12 steps of 3 blocks, its
    logits 2.5e-12 from the host's in NMSE at decode, and the eight violations
    on its xIELU and rotary read 3e-4 to 1.9. The four BitNet fixtures are
    declined by `UncoveredTokenAct` (24 or 36 block steps in 12, none served).
    The qwen3next-shaped hybrid's attention block is served (8 block steps in
    8) in its dense, tiled and 256-wide dense (`bigdense`: Qwen3-Next-80B's
    16 heads of 256 with a dense FFN, logits 1.1e-4 and 1.0e-4 from the host's
    on cuda and vulkan) variants, its recurrent blocks declined by kind, and
    its mixture variants' (`moe`, `big`, `widemoe`) attention blocks by
    `UncoveredMixture`, so nothing of those is served.
  - The out gate is held to being seen. `TestHybridOutGateViolationsOnDevice`
    gives the composer a graph rewritten after the front end built it (the
    tier's own kernels never read it), runs the hybrid gate's verdict on it
    (logits past the 5e-3 block bound at the deep arm and the shallow control
    past it too), and counts that the composer served the block it was
    handed. Dense fixture, cuda:0 and vulkan:7 alike: the gate left out reads
    0.143 (28x the bound) at the deep arm and 0.368 (74x) at the control; the
    halves of the double-width query swapped 7.7e-2 (15x) and 5.7e-2 (11x);
    the halves read whole rather than per head 1.3e-2 (2.7x) and 5.4e-2
    (11x). The 256-wide fixture: 2.4e-2 (5x) and 7.7e-2 (15x); 6.6e-3 (1.3x)
    and 3.2e-2 (6x); 1.6e-2 (3x) and 4.0e-2 (8x). Unrewritten the gate
    reads 4.5e-5 and 1.1e-4.
  - Each new form is executed on the CPU against the oracle. xIELU is held to
    the engine's own bound for its kernel (4e-7 of the sum of the terms'
    sizes) plus exp's error where the form takes exp (3.5e-6 of exp(m), the
    remainder of the degree-5 Taylor sum both engines build exp on), at
    widths 1 to 100 with a guard after the output and at the arguments where
    that remainder peaks. The composed form reads 0.19 to 0.58 of the bound
    and main's kernel the same to three digits; against the engine's bound
    alone both read up to 1.59. The first bound, 2e-6 of the result, is met
    by neither (main's kernel reads up to 2.4e-4 of the result, where the
    terms cancel). Six violations (the alphas exchanged, beta dropped, eps
    ignored or moved, the windows one element along, expm1 stated as
    exp(x) - 1) each read at least 1.8e4 of the bound. The split of a gated
    projection is exact and its gated output 0.6e-6 to 2.8e-6 from the
    definition, main's kernel's own figures, under 3.6e-6; the norm in front
    of a projection is 1.6e-14 from the oracle in NMSE.
  - The expm1 bound is looser than the first version of its gate. That held
    every input to 4e-7 and read 1.46e-6 past the Taylor window, at 3.7 times
    the bound. Past |x| = 1/2 expm1 is exp(x) - 1, and exp is the degree-5
    Taylor sum on a reduced argument |r| <= ln2/2 that main's kernels use
    too (`jit/pass_exp.go`, `emitExpPS`): its remainder is 3.24e-6 of exp(r)
    at r = -ln2/2, so exp holds 3.5e-6 (read 1.54e-6), and exp(x) - 1
    multiplies that by e^x/|e^x - 1|, 2.54 at x = 1/2 and falling away, so the
    bound past the window is 2.6 x 3.5e-6 + 1 ulp = 9.2e-6. Within the window
    the form is a few ulps and keeps 4e-7 (read 6.5e-8). The bound is derived
    from the exp the engine already ships, not fitted to the reading, and
    exp(x) - 1 stated the naive way still reads 1.0 within the window.
  - Main's xIELU gate did not sample the arguments where exp's remainder
    peaks, and its own kernel reads 1.59 of its bound there (x = -1.04, the
    edge of the reduced argument; 0.584 with exp's error added). It now does:
    `TestXIELUAtTheExpEdges` (jit/cpu, avx2 and sse) runs the sixteen
    arguments at both edges of eight ranges under `xieluTol` plus
    3.5e-6 |alpha_n| exp(m), and fails an exp a Taylor term short (caught on
    306 elements) and the other faults. The backend's gate (2e-6 of the
    terms, cuda ptx and vulkan spirv) takes the same arguments: they read
    0.01 and 0.00 of its bound, against 0.03 to 0.05 for its sampled ones, so
    its bound stands.
  - The other hybrid device gates in the composed build, on card 7, before
    the rebase: the 38 tests that place a hybrid on a device and that the
    attention block's composition could move (`TestHybridDeviceErrorIsAmplification`,
    `TestHybridDeviceErrorGrowsWithPosition`, the second-session, seam,
    paging, scheduler, streaming and prefill gates) ran 28 to a pass, 2 to a
    failure and 8 to a skip, 81 passes counting subtests. The failures and
    skips named files the box did not hold (the Kimi-K3, kimilinear, mamba2
    and image-chat fixtures and goldens, and the Qwen3.5-0.8B checkpoint),
    each a missing artefact, not the change. The checkpoint was fetched on the
    box from Hugging Face and its five gates pass in the composed build (13
    passes; six two-device subtests skip: they name vulkan:1, which the
    one-card selection does not hold). The seven gates on the other fixtures
    pass on the laptop's GPU (29 passes). The two fault-injection hybrid
    gates and the four step-fault gates that share their file carry the
    `jitllmfault` tag, which the sweep's binary leaves out; a binary built
    with it passes all six (29 passes).
    `TestBlocksMoveBetweenDevices` needs two distinct cards and is not part
    of this.
  - `TestComposedBlocksServe` over its 17 models, cuda:0 and vulkan:7, whole
    block and whole block with the head, with and without the recording, the
    exported tree of the first commit: 14 passes a model, none skipped. Each
    arm's median distance from the float reference is at most 1.21 times the
    tier's (gemma-2b; bound 1.5) and its worst at most 1.75 times the widest
    the tier's backends reach (Qwen3-1.7B; bound 4). On the rebased tree four
    of them, Llama-3.2-1B, Qwen3-0.6B, EXAONE-4.0-1.2B and gemma-3-1b-it, the
    last two with sliding layers the tier now serves, pass 14 each as well, at
    most 1.10 of the tier's distance at the median and 2.18 of the widest
    backend's at the worst (gemma-3-1b-it).
  - The escape analysis against seam-wip's head. `go build -gcflags=-m` over
    `jit/gpu/tier`, `jit`, `engine/graph` and `engine/model` reads 41235
    heap diagnostics there and 41297 here. The 24 entries that differ are in
    `jit/decompose_map.go` (a form), `engine/graph/verify.go` (the verifier)
    and `engine/model/front.go` (the graph a model states), all built once;
    none is in `jit/gpu/tier`, `jit/pass_regs.go` or a file a token runs.
  - Logs. V100, `~/jitllm-test/gates/engine/model/`: `g5-*.out` the first
    commit's default build (`g5-serve-*`, `g5-c6-full`) and the new tree's
    composed build (`g5-violations-c`, `g5-hybrid-block-c`, `g5-hybrid-sib-c`,
    `g5-apertus-c`, `g5-bitnet-c`, `g5-c6-full-c`, `g5-xielu-backend`);
    `g6-qwen35-c`; `g7-*` the final composed binary (`g7-violations`,
    `g7-hybrid-block`, `g7-bitnet`, `g7-c6-full`, the source of the refusals
    named above); `g8-*` the first commit's composed build; `g9-room-step`
    the fault-tagged composed binary; `g10-*` the rebased tree (`g10-c6-full`
    the counts of the 25 fixtures, `g10-violations`, `g10-hybrid-block`,
    `g10-bitnet`, `g10-serve-*`). Laptop, `agent-scratch/gates/`: `cpu7.out`
    (the composed CPU gates and all 59 rows of `TestFrontStatesTheModelAsOps`
    on the rebased tree, two of which the V100 box has no file for and which
    need no device), `cpu8.out` (the corrected qwen35 test), `xielu-cpu.out`,
    `lap-hybrid.out`, `esc-compare.out`.
  - The width gate found a bug in the CPU lowering. `ZerosPerLoop` redirects a
    reader to a zero made for its loop through an operand pointer taken before
    the append that adds the zero, and the append moves the nodes, so the
    redirect wrote the old copy. A tail loop kept reading the zero made in the
    vector loop's body, which a row narrower than a vector never runs: the
    compare read whatever its register held, and xIELU at width 7 read about
    2e6 of the bound on two or three of six sets, differing run to run. The
    operand is fetched after the append, and a gate over the node sequence
    holds every float zero to being made in a loop that encloses its reader
    (8 kernels; the zeros merged back into one are seen in all 8). A slice of
    heads whose width is not a power of two did not form on the CPU (no
    general-register division by a constant); it is now a space of row, head
    and element, as the head norm's.
- **The head's ops as ops: LayerNorm, bias, softcap and scale.** The composed
  head served a norm and a projection and declined every other head by name.
  rnj-1's reason, which the serving gate printed as "()" because it read the
  tier's last error and a decline sets none, was "a head graph other than a
  norm and its projection": its final softcap of 30 (one step declined on
  CUDA, eight on Vulkan). Granite's logit scale, gemma-2's softcap, a head's
  bias and a LayerNorm head (the C6 kit's falcon and phi-2, which the front
  end refused as classic) were the same. graph.Head now states the head: an
  RMSNorm, or a LayerNorm with its stored bias; the projection with its
  stored bias; the final softcap; the logit scale. The softcap's tanh is
  2*sigmoid(2y) - 1, finite at both ends, where the quotient of exponentials
  it replaces is NaN past |y| of 44. The tier serves any chain of elementwise
  ops on the logits and holds the head's bias beside its norm. A served head
  that caps writes the capped logits where the tier's cap does, and the tier's
  cap, which launched after the head, is not launched over them (gemma-2 would
  cap twice). nn.Head says what the model's head does (Stages) and the device
  says what it applied (Applied), in every call: finishDevLogits, the
  device's argmax and sampler, ForwardSample and a step across sessions read
  that report and no longer the model's config, so the composed head selects
  tokens after the bias and the scale, and a head the composer declines runs
  as it did, the cap on the device and the bias and the scale on the host. A
  head whose normed row is read back (a prediction block's) is declined by
  name, as the form keeps that row inside.
  A gate on each op runs against the oracle (float32 norm and projection, the
  rest in float64) with every op taken out. On the CPU, one row at a time
  and four at once (TestComposedCPUHead), every arm reads 6e-15 to 1e-12
  against a bound of 1e-9, and the softcap dropped, the scale inverted, the bias dropped, the
  norm's bias dropped and an RMSNorm for the LayerNorm read 0.10, 4.0e+03,
  0.35, 0.10 and 0.10. On a V100's CUDA and Vulkan (TestComposedHeadOps, over
  an RMSNorm head, phi-2's LayerNorm with both biases and falcon's LayerNorm
  over the tied embedding), the composed head reads at most 3.0e-13 and the
  tier's own head, the host finishing its logits, at most 1.4e-13, against a
  bound of 1e-9. A head with both a softcap and a bias, which no model has,
  is missing work, declined by name: every device path of the tier's own head
  caps the logits before the host adds the bias, the other order of the
  model's (bias, softcap, scale), so PrepHead refuses the pair (its LastErr
  names it) and the head stays at home, where the order is the model's. The
  composed graph states the pair in the model's order (TestComposedCPUHead's
  "all" arm), so the work is the tier's own cap on the device ahead of the
  host's bias, or the retirement of the tier's own head. The gate's arm
  asserts the refusal, the blocks all on the device and the logits against
  the oracle, and takes the softcap only on heads without a bias. The
  softcap dropped reads 3.5e-02 to 4.5e-02, the scale inverted 4.0e+03, the bias negated 0.13 and
  the norm's bias dropped 0.10 and 0.71. TestComposedHeadTokens holds the
  greedy token and the sampler's draws to the logits the device returns: the
  composed head selects all eight tokens and all 24 draws on the device, the
  tier's own head none of the draws of a head with a scale or a bias and none
  of the tokens of one with a bias, and a head told it has no stage to leave
  parts in the draws (2, 2, 23 and 4 of 24) when the tier's own selects them.
  TestFrontStatesTheModelAsOps forms gemma-2-2b, granite-3.3-2b and
  granite-4.0-micro whole (31, 44 and 44 parts).
  The device gates run in a composed build on CUDA and Vulkan:
  TestFinalSoftcapReachesTheDeviceHead, TestGemma2SoftcapOnDevice (2.8e-03 of
  2e-02) and TestGraniteScalesOnEveryDevice, which failed before this change
  and not for the head. Its dense fixture's decode runs on the composer's
  blocks, and the gate counted the tier's own scaled adds. It now counts a
  composed block as the scaled path and states the two scales as the model's
  own configuration as well, which the composer's forms read as the tier's
  kernels do (0.85 and 8.1e-03 against a clean 3e-14).
  The serving gate takes gemma-2-2b, granite-3.3-2b, granite-4.0-micro and
  falcon-7b with their heads served: no head is declined or unstated, and
  the lists that held them, the headless front (jit.FrontHeadless) and
  UncoveredClassic are gone, nothing reading them. On a V100 the composed block and head read a median 0.987,
  1.000, 0.994 and 1.003 of the tier's own distance from the float reference
  on CUDA (worst NMSE from the tier 0.052, 0.0036, 0.0174 and 3.9e-05) and
  0.996, 1.002, 1.075 and 0.997 on Vulkan (0.0635, 0.0076, 0.0204 and
  5.1e-05), over 112320, 5740, 5740 and 4620 part steps each, inside the
  gate's band as it stood. Falcon's LayerNorm head is the composer's: the
  tier's own launches on its served steps are the rotary table alone. The
  front test requires a head graph of every model on its lists, whose first
  op is a LayerNorm exactly where the model's norms are, and the composed C6
  gate asks for the head beside the blocks of the ten classic fixtures and
  holds it to a served step at every decode step. rnj-1 is formed whole and
  its head served where its blocks are not
  (TestComposedHeadServesWhereTheBlocksDoNot). The composed head serves a
  single-row step that is not a prompt's: a folded prompt's last row, a
  prompt's own calls, a ragged step and a chunk projecting every row keep the
  tier's own head (a prompt's, by a composedLater name where the prompt is
  asked), so the first draw after a prompt is made over logits the host
  finished and every draw after it is the device's.
  Two composer defects surfaced on the way. Formation: Affine's cse redirected
  each duplicate with a pass over the kernel of its own (the profile put 90%
  of the time in Affine, cse, redirectAll, redirect, Operands), so
  granite-4.0-micro's head with a scale, over a 100k-row vocabulary, took 12.7
  to 16.3 s as int8 and over 45 s as float against 10 to 21 ms without the
  scale; one sweep over every duplicate brings them to 0.22 s and 3.7 s. The
  CPU vectorizer: a norm over two or more rows read NMSE 5e+05 to 8e+05 (one
  row, 1e-14), because broadcastInto broadcast only values made from a lane
  sum and left a row loop's scalar in lane 0, and then because it wrote its
  broadcasts through operand pointers an append had moved (a LayerNorm without
  a bias lost one of its two scalars that way). With both fixed every head arm
  reads 1e-14 to 1e-12 over four rows at once (TestComposedCPUHead), and the
  original pass fails the same gate at 5e+05 to 8e+05 and 3.8 on the softcap.
  Escape analysis (go build -gcflags=-m, line numbers dropped) against the
  tree before this work, over jit, jit/gpu/tier, engine/model, engine/nn and
  engine/graph, adds no heap escape to forward.go, step.go, layer.go or
  launch.go: the additions are graph.Head.Build's thirteen (built once, then
  cached), two appends and a local in the cse sweep (formation) and one error
  in servedHead (its failure path); front.go loses four.
  Open: the other redirectAll call sites (foldDecided, the affine pass) have
  the same shape as the cse that was quadratic and are not profiled.
- **BitNet's per-token int8 inputs as ops; ternary weights in the composer's
  matvec.** The front end refused BitNet by `UncoveredTokenAct`. Each
  projection's input is now stated as main computes it
  (`graph.Builder.Int8Codes`): the row's amax (`OpReduce` max over a new
  `MapAbs`), its scale amax/127, the scale's reciprocal (a new `MapRecip`,
  zero where the scale is), the row times it rounded half away from zero (a
  new `MapRound`) and clamped to [-128, 127], a row of codes (a new value
  element, `graph.ElemI8`); the product over the codes, then the row's scale
  times it, the bias after. An input several projections share is quantized
  once. The composer's int8 matvec takes a code row as its own int8 form, no
  window and no scale (`Context.int8Codes`, the packing half of the int8 form
  split out as `int8Pack`), so a TQ2_0 or TQ2_F32 weight runs as ternary
  fields against int8 codes through `OpDot4`, its decode the format's generic
  reading rules (`jlm.DecodeWord`), and no weight goes to a float; a device
  without the int8 form (the CPU's profile) runs the decoded product over the
  same codes. On a device whose division is not correctly rounded (Vulkan)
  the reciprocal is the nearest float found in integers and x/127 the
  quantizer's exact integer form, |x|'s with x's sign. A map operand one
  element wide is read at element 0 rather than at i/W, which the CPU
  lowering could not emit at a width that is not a power of two.
  `UncoveredTokenAct` goes. Gates: `TestComposedGPUTernaryPerToken` (jit) on
  a V100, CUDA and Vulkan, 80 of 80 shapes each (TQ2_0 and TQ2_F32, 1 to 129
  rows, k 256 and 768, one and three tokens) against the oracle's quantizer
  and matvec at NMSE under 1e-9 with guard words after the output, and each
  sees the weight's words moved by one, the reference over 32-wide windows
  and (at 78 of 80, the rest a cancelling sum) the codes rounded half to even;
  `TestComposedGPUQuantMaps` holds the four maps to Go bit for bit through
  both division forms (the integer forms' subnormals are the device's, as
  main's `div127`). On the CPU (`TestComposedTernaryMatVecPerToken`) the
  decoded form is right at 68 of 72 shapes and the int8 form, which the CPU's
  profile does not ship, at 36 of 72: the rest are the x86 lowering's work
  list, the same at base for any packed format (a one-row TQ2_0 matvec runs
  out of registers, and a byte dot over three tokens at ragged row counts is
  left outside vector lanes). The four synthetic fixtures' every block is
  served on CUDA and Vulkan (composedCovers: 3 of 3 and 2 of 2 blocks, 36 and
  24 block steps in 12, none declined), decode against the host at
  9.6e-14..1.8e-13, every device violation 3.2e-02 or more;
  `TestBitNetRealMatchesReference` keeps every arm, worst NMSE 1.2e-02..8.2e-02
  against the 2e-01 bound, no confident flip; and bitnet-b1.58-2B-4T-i2_s
  joins `servedModels`: on CUDA the composed blocks read a median 0.918 of the
  tier's distance from the float reference (worst 0.0509, the widest backend's
  0.0938), on Vulkan 1.053 (worst 0.0245), 4200 block and 4340 block-and-head
  part steps each. `TestDecodeDoesNotAllocate` on synth-bitnet-i2s, every
  block on the card in the composed build, reads 0 engine allocations on
  every host, CUDA and Vulkan arm, and `go build -gcflags=-m` over
  engine/graph, jit and engine/model reads the same 199 heap moves as the
  tree before this work.
- **A kernel that does not build is returned, never moved around.** The device
  call's refusal now carries a cause (`nn.FailCause`: decline, build, fault).
  The tier records a build failure at the funnel every kernel the call cannot do
  without goes through (`devTier.compile`, `buildFailed` for the composer's
  forms and the cached "did not form earlier" answer) and a fault where the
  submission's session errors or the injected failure fires; the model reads it
  (`nn.CauseReporter`) at each place a failed `Layers` call used to demote, retry
  or fall to the CPU device. A build failure returns `*nn.BuildError` naming the
  block and moves nothing: no demotion, no relocation (`relocateUntil` stops
  where `ReserveKV` or `RowsRoom` failed to build, and the step that needs the
  kernel returns it), no head on the CPU device, no per-row retry of a chunk
  whose kernel did not build. `SetDeviceFallback` now evacuates only for a fault
  or a named decline (room); `State.Evacuations` counts demotions by cause,
  relocations and build refusals (`jitllm run` prints them). Gates, both
  injected as rewrites and never in a kernel: `SetFormFault` replaces the
  composer registry's matvec decomposition with one that fails
  (`TestBuildFailureMovesNothing`, block and head arms), and its violation
  (the same failure reported as the device failing) must evacuate, which the
  gate demands; `JITLLM_DEVICE_FAIL_AT` still evacuates and is counted a fault
  (`TestDeviceFaultAtDecodeEvacuatesAndIsCounted`,
  `TestDemotedDevicePrefillFinishesOnTheCPU`). A first run of the relocation
  set failed 6 tests, all `relocate-then-reclaim`: `RowsRoom` did not clear the
  cause, a compile that an earlier call had refused read as this call's build
  failure, and no block moved; every call a `fits` closure makes now clears it.
  Card 0 (V100, cuda:0), default and composed builds: the fault set, the
  relocation set (`TestLayerRelocatesToAnotherDevice`,
  `TestKVPagesRelocateWithTheLayer`, the eleven `TestBatchSeamMovesCarryEveryRow`
  variants) and `TestDecodeDoesNotAllocate` on Llama (28 passes, cuda:0 and
  vulkan:0) all pass. Escape analysis against the tree before this work adds
  only failure-path escapes (`BuildError`, `errors.As`' target); no line of
  forward.go, step.go, layer.go's warm path or launch.go changes. Open: the
  kernels' own IR constructors (about 80 `kernels.X` call sites in the tier)
  and the placement-time matvec compiles record no cause, so their refusals are
  still declines; a session error that is not the driver's (a tier logic error
  inside the submission) is read as a fault.
- **The step's inputs as parts: the rotary table, the input row and the
  greedy token.** A served step still launched the tier's own rotary table
  (composedResidue was ["ropetable"]), and the State filled the input row
  and the tier's argmax picked the token. Three part kinds join the block
  and the head, each a Config.Composed bit (ComposedTable, ComposedEmbed,
  ComposedSample; ComposedStep is all five, the jitllmcomposed build's
  default and JITLLM_GPU_COMPOSED=1's), each formed at the placement or
  PrepHead from the front end's graph and counted served, replayed or
  declined by name (Stats.StepDeclines). The table is jit.Front's PartTable
  graph (jit.TableGraph), one per rotary the plan states
  (nn.LayerPlan.Tables), writing cs and csSWA where the tier's kernel did;
  it reads the planes the tier already holds, and a rotary over several
  axes has its planes in the plan too (RopeTabAxes, which RopeTab leaves
  nil). Its positions are the State's: before each decode call the State
  tells the device the row's rotary position on every axis
  (nn.RopeAxesDevice; ropeOf, which a picture moves off the cache
  position), staged with the call's other writes, so a recording replays
  the table at this token's positions. A table over several axes with no
  positions for the call, XD-RoPE's and one the caller keeps on the upload
  (RopeTableHost) are declined by name. The input row is the front end's
  graph of the input rows (nn.Head.Embed) gathered from the resident head
  where the head is the embedding: the State, told at PrepHead that the
  device gathers a row (nn.Head.GatherRow), promises it the token
  (nn.EmbedDevice) in place of filling the row, and refills it only if the
  device fails the token. An untied embedding or a position table is
  declined by name (tinyllama, stories15M); a prompt's chunk keeps the
  tier's gather. The token is graph.Sample over the logits the head wrote,
  into the word the tier's argmax writes (the buffer the device sampler's
  readback shares), asked only where the head's report leaves no stage to
  the host (nn.Head.Applied, as ForwardSample reads it). The sampler's
  top-k with its penalty, and the label pick, state no graph yet and are
  declined by name.
  On a V100 (CUDA and Vulkan, every arm with and without the graph) the
  serving gate's whole-step arm leaves the tier's own launches on a served
  step empty (composedResidue is nil): stories15M, Llama-3.2-1B,
  Qwen3-0.6B, gemma-3-1b (two tables), Qwen2-VL-2B (a table over three
  axes at text positions), SmolLM2-360M and tinyllama. Each part has a
  violation the gate must see, the State's statement changed under the
  served part: every row told a rotary position 16 past its own reads a
  median 3.7 (Llama-3.2-1B) to 13.8 (Qwen3-0.6B) times the tier's distance
  from the float reference (band 1.5), and every row gathered at the next
  token 308 to 1850 times. One position off is inside the band on
  stories15M (median 1.10 and 1.14): its tier is 0.14 from the float
  reference at its worst, wider than one position moves it.
  TestMRopeMatchesTransformers decodes after the picture with the step
  served, every table at the row's own positions, off the cache's: within
  4e-13 to 6e-8 of transformers on qwen2vl, qwen2.5vl, qwen3vl, qwen3vlmoe,
  qwen35vl, glm4v and glm4vmoe, 5 tables served in 5 steps each, and told
  one position off 1.5e-2 to 8.2e-1 with 0 to 5 argmax flips.
  TestComposedHeadTokens' composed-token arm selects every greedy token of
  synth-qwen2, synth-phi2 and synth-falcon on the device (8 of 8, served
  or replayed), and over the logits negated (a wrong pick) all 8 part from
  the host's. TestRopeTableOnDeviceMatchesTheUpload holds the composer's
  table to the host's upload as it holds the tier's kernel: on Llama-3.2-1B
  and gemma-3-1b (both tables), 24 tokens identical and the logits equal on
  an RTX 3050 Ti, and passing on the V100. In the jitllmcomposed build,
  whose tiers now serve the whole step, TestC6OnEveryDevice,
  TestQwenVLMatchesTransformersEndToEnd, TestGemma3GlobalLayersAreLinearlyScaled,
  TestSlidingWindowRunsOnTheDevice, TestFinalSoftcapReachesTheDeviceHead,
  TestGemma2SoftcapOnDevice, TestGraniteScalesOnEveryDevice,
  TestForwardSampleMatchesSampleOverForward and TestComposedPromptIsNotServed
  pass on the V100.
  TestDecodeDoesNotAllocate's composed arms on Llama-3.2-1B (the step,
  and the step shadowed; graph on and off) make no engine allocation, a
  greedy decode with the token served included.
  TestForwardSampleMatchesSampleOverForward, TestComposedPromptIsNotServed
  and TestComposedShadowHoldsEveryBlock (Llama-3.2-1B) pass.
  Escape analysis against the tree before this work (go build -gcflags=-m,
  line numbers dropped, jit, jit/gpu/tier, engine/model, engine/nn,
  engine/graph) adds the forms' builds (their maps and a table's planes
  name, once a form), error paths, the placement's planes and table list,
  and appends onto the staging bytes the tier reuses (the rows' positions,
  the promised id), none on a warm step's path.
  Open: the sampler's top-k with its penalty and the label pick, which no
  graph states; a prompt chunk's gather, the tier's; a promise of a row the
  head's device is not block 0's device makes real at home with the tier's
  gather (GPU.embedRows), as a decode across two cards does.
- **The CPU device's lowering: every work list of jit's CPU gates at zero.**
  The register pass declined silently (`Regs` returned the kernel unbound and
  the x86 lowering failed later on "node N has no register", or nothing lowered
  it at all); it now refuses by name ("the floats do not fit the vector
  registers: ..."), which exposed the declines it had hidden, each fixed at its
  cause. (1) `Cursors` gave a scope's own accesses their displacement from where
  the scope started while a loop beside them advanced the buffer's own
  register: a flattened one-trip tail read 8 elements past (the RMS norm at
  widths 9, 65, 257 was NMSE 2e-2 to 1.5e-1). A loop that moves the pointer in a
  scope with its own accesses now moves a copy. (2) `Vectorize` kept an add's
  tail term to lane 0 but not an fma's, so a LayerNorm's (x - mean)^2 tail added
  mean^2 in seven lanes: every ragged width was wrong (NMSE 3e-2 to 9e-1, 150
  shapes); the fma's factors are now kept to lane 0. (3) `removeLoop` cut a
  tail's tie to the vector loop it took trips of when an unrolled loop of no
  trips went, and the binding indexed loop -1 (a panic at width 15); the
  family now passes to the loop the removed one took trips of, or its first
  member. (4) A constant read by a flattened tail or a kernel of no loops was
  never rematerialized (the one-axis rotary table at 9, 17 and 129 pairs, the
  exact reciprocal on a device without exact division); such constants are now
  reloaded where read once none read only in loops is left. (5) A kernel with
  byte dots and no vector loop kept its integers in general registers, where no
  byte dot is ("a byte dot outside vector lanes": 30 of 36 three-token int8
  ternary shapes); a byte dot now makes a kernel vectorized. (6) `Simplify`
  folds and, shl and shr of constants (a one-row ternary matvec held an index
  of constants in no register). (7) When nothing is left to rematerialize, the
  binding sinks a vector value with one reader, and what only it reads, to just
  before that reader within straight code (Q4_K at four tokens, int8: eight
  weight words were live across the byte-dot chain). (8) A pair of query heads
  read the history twice where their kv head is one: (2h+1)/r and 2h/r are now
  one node for a constant r the pair's factor divides (`sameQuotient`, shifts
  and quotients), so the second head's load is the first's and has a twin
  ("a pair's later head with no twin" refused 20 of the front test's models on
  the CPU, and engine/cost's placement test); and a copy whose twin's register
  is a hint, not the first head's run, takes a register no value takes rather
  than hint + n, which overflowed and could alias. Numbers on the laptop
  (i9-12900HK, AVX2 and VNNI): `TestComposedCPUMatVec` 70 of 72 to 72 of 72
  formed and right (the float, int8 and int8-without-VNNI arms, at one and four
  tokens); `TestComposedCPULayerNorm` 206 of 230 controls right and 150 shapes
  on the work list to 230 of 230 RMS norms and 230 LayerNorms right, the gate
  now asserting every width; `TestComposedTernaryMatVecPerToken` 40 shapes not
  formed to none, 72 of 72 right in each arm; the rotary gate gains one-axis
  tables at 9, 17 and 129 pairs, each main's table to the bit. All three gates
  now fail on a shape that does not form. engine/cost passes with the register
  pass refusing by name, as do the CPU forms of the 20 front-test models it
  had refused. On the Xeon E5-2680 v4 (AVX2, no VNNI): the jit CPU gates (48 of
  48 matvec shapes), the rotary table's CPU arm and the engine/model CPU gates
  pass. Escape analysis of jit: 146 moved-to-heap lines before and after, the
  same set. Open: the pairs' kernels now form on the CPU device but no gate
  executes an attention kernel there; that is the runner's work.
- **A projection over a chunk's rows runs in row tiles.** At R rows a
  projection was one naive sum an output element, and once its elements
  filled the device no split took it; the late unrolled fuse
  (`Fuse{Reduce→Map, Unrolled}`) then wrote every trip of it into the map
  that read it, the SiLU's or a rotary's: single kernels of 25,071 ops in the
  int8 form and 17,675 to 103,666 in the float form of a Llama-3.2-1B block
  from 32 and 64 rows on. A pass on the naive nest, `RowTiles`
  (`jit/pass_rowtiles.go`, after the int8 dot's fuse and before `Threads`),
  cuts the leading axis of a sum over a weight matrix into tiles of T rows,
  T the most rows, a power of two dividing R and at most 16, that leave the
  launch four warps a multiprocessor (`Device.Units` times its lanes, or
  30,720 threads where the device states no multiprocessors, as Vulkan
  does): a thread is one tile's outputs at one column, the nodes that do not
  depend on the row (the weight's loads and decode) made once a trip and read
  by every row of the tile, each row its own fold and store. A tiled kernel
  has T stores, so neither a split nor the unrolled fuse takes it, and its
  ops grow with T, never with R. A one-row kernel never tiles: the lowered
  kernels of the 25 served models at one row (both device statements, int8
  and float, SPIR-V and PTX) hash the same before and after. Coarsen's memo
  printed each unit's value with fmt to name a coordinate; at 512 rows that
  was 8.3 s of the 9.1 s a block took to form, and the key is now the units'
  words (the same equivalence: the forms of three models at 1 and 64 rows are
  byte for byte the same). One Llama-3.2-1B block, int8 form, on the V100
  CUDA statement, before → after: 1 row 4,195 ops (120 KB SPIR-V, 150 KB PTX)
  unchanged; 32 rows 5,563 → 6,684 ops (143 → 193 KB, 181 → 241 KB);
  64 rows 34,104 → 6,986 (1.08 MB → 208 KB, 1.37 MB → 257 KB); 128, 256 and
  512 rows 7,701, 7,879 and 8,291 ops after (238 to 261 KB SPIR-V, 294 to
  324 KB PTX), the largest kernel 2,347 ops, an attention kernel the same at
  every R. The float form: 32 rows 21,920 → 4,724 ops (612 → 121 KB SPIR-V),
  64 rows 142,976 → 4,795 (3.8 MB → 123 KB, 5.2 MB → 157 KB PTX), 5,131 at
  512. On a V100 (card 3) with the drivers' caches off, compiling the block
  takes 0.55 to 0.93 s on cuda:0 and 0.64 to 1.69 s on vulkan:3 at every R
  from 1 to 512, in either form; before, the int8 form took 2 min 16 s at 64
  rows on cuda:0 and 1 min 37 s to 1 min 45 s at 128 to 512, and 11 min 55 s
  at 32 rows on vulkan:3, where no split takes the 25,071-op kernel even
  there (the rest of the base Vulkan rows were not run). Forming on the
  V100's host is 0.2 s at one row to 3.1 s at 512 (16.2 s before, 12.9 s
  with the tiles before the key change). Gates: `TestRowTilesHoldTheOps`
  (jit) forms a Llama-shaped MLP over Q4_K, Q6_K and Q8_0 at 1 to 512 rows
  on both statements, every projection
  tiled from 128 rows, none at one, no kernel past 1,000 ops and the largest
  within 1% from 128 to 512 rows; with the pass off the float form at 32 rows
  reads 17,675 and fails. `TestComposedGPURowTiles` runs a 2048-by-256
  projection at 1, 3, 32 and 256 rows over F32, Q8_0, Q3_K, Q4_K, Q5_K and
  Q6_K, int8 and float, on CUDA and Vulkan (the laptop's and the V100's):
  48 forms each, 24 tiled, every row under NMSE 1e-9 from the oracle with
  guard words after the output, and the weight's words or the input's rows
  moved by one each fail. Llama-3.2-1B, tinyllama (Q3_K), Qwen2-1.5B and
  Qwen3-1.7B join `promptModels`: on cuda:0 and vulkan:3 every arm of
  `TestComposedBlocksServe` passes, every block step of the prompt served,
  none declined (Llama's median ratio 1.047 and 1.031, tinyllama's 1.241 and
  0.948). `TestPrefillMatchesForward/device` adds tinyllama's arms; its
  composed last rows read 1.77e-3 (Llama) and 3.2e-4 (tinyllama) on CUDA,
  2.11e-3 and 2.1e-4 on Vulkan. `TestPlacedPromptsCompileNothing` passes, its
  composed-prompt arm placing in 20 s (64 s with the tiles alone);
  `TestDecodeDoesNotAllocate` on Llama passes every arm, the composed arms
  with ComposedPrompt on CUDA and Vulkan; the jit package's 22 tests pass on
  the V100. `go build -gcflags=-m` over jit adds one heap move, a pass's
  node copy at form time (the idiom of 60 others). Placement of
  Llama-3.2-1B at the default width of 512, with the drivers' caches off, is
  still minutes: 2 min 1 s on cuda:0 and 3 min 11 s on vulkan:3 with the
  blocks, the head and the prompt served, against 44 s and 1 min 42 s for the
  tier's own kernels and 41 s and 1 min 51 s with the blocks and head alone.
  A CPU profile of the CUDA arms names what remains: every warmed width forms
  and compiles each distinct block's form (16 widths), forming 45 s of which
  Coarsen's per-unit values (linear in the rows) are 33 s, compiling the
  forms 43 s; and the tier's own per-width prompt kernels are still compiled
  beside them, about 40 s, where the composer serves the prompt.
- **A block's location is a device name, and the CPU is one of them.** A
  block is on "cpu" (`nn.CPU`) or on a card, read through `State.WhereIs(li)`
  and `State.Locations()`; the placement set and its seam are written only by
  their setters (`markOnDev`, `unmarkOnDev`, `clearOnDev`, `shrinkOnDev`,
  `holdHostRuns`), which `srcgate.TestBlockLocationIsWrittenByItsSetters`
  holds over 105 files and 10 writes (a constructor's `gpuLayers: lo` put
  back fails it). Placing and releasing are one call, `State.MoveBlocks(lo,
  hi, to)`, `to` a card or "cpu": to the CPU device the history comes off the
  card first, to a card each block is offered to that card by name
  (`PrepLayerOn`) and its history follows; `ReleaseBlocks` is gone. The
  placement service's `RelocateBlocks` takes `to_device_id: "cpu"` where it
  took `to_host` (field 5 reserved, regenerated with `buf generate`), and
  the orchestrator's demotion and `jitllmd place` go through it.
  `TestBlocksMoveToTheCPUAndBack` moves the upper half of every model to the
  CPU device and back, cut at a KV-sharing group's edge: 40 pass, 0 fail, 0
  skip in the default and composed builds on a V100 (cuda:0 and vulkan:2),
  and its violation (the move without its history) is 15.4 logits off the
  CPU control at token 3. Two composed-build declines on relocation paths
  came out of the run: `TestRelocatedBlocksComeBack/shorter-prompt` failed 2
  of 2 on the tree before this work and on this change alike, and
  `TestPlacement/stream` failed the same way (on this change; the tree
  before it was not run under vulkan:1). A composed
  part's instance was asked only `room`, so after a 64-row prefill (0.66 MB
  free, the 15 trimmed history pages still in the pool) block 15's 8137828
  bytes were refused, and that refusal was recorded as a build failure and
  cached, leaving the block placed with no kernel. `devTier.instance` now
  makes room as state does outside a submission (`stateFits`: the pool's
  free pages, then a streamed block paged out), and a refusal for room is a
  `partRoomError`, a decline that is not remembered. Shorter-prompt passes
  at 4.66e-04 against its 3.83e-03 bound with the card back at the probe's
  1132199936 bytes; stream places 16 of 16 blocks with 32 page-ins. A real
  build failure is still one: `TestBuildFailureMovesNothing` and the fault
  set pass in both builds, and a form failure routed as a room refusal fails
  both its arms (no `nn.BuildError`, the State demoted instead). The moves
  set reads 325 pass, 0 fail, 13 skip in each build (the skips are the q8_0
  cache arms refused by name and the attention-free state-space models),
  `TestDecodeDoesNotAllocate` on Llama 0 engine allocations on all 32 arms
  in each, and `go build -gcflags=-m` over the packages touched 361 heap
  moves before and after, the same set.
  `testdevices.Refusing` declares a device a test names to see it refused:
  the server's cuda:7 refusal cases no longer fail the binary (14 pass, exit
  0), the same run without the declaration exits 1 on `cuda:7 x2`, and an
  undeclared device beside a declared one is still a violation
  (`TestADeliberateRefusalIsNotAViolation`).
- **Every input row the composer's: untied embeddings, position tables,
  prompt chunks, and no fallback gather.** A ComposedEmbed tier gathered only
  a decode row, from a head that was the embedding, and a promise no call
  took was made real by the tier's own gather (materializeLocked's
  GetRowsPacked), which is why float embeddings were declined. Now the input
  rows graph (nn.Head.Embed) is served at any row count a call carries: a
  decode row, a prompt chunk at its scratch's rows (counted apart, under
  tier.PromptEmbed), and a promise no call takes, formed at the narrowest
  width the placement formed and read back (materializeComposed) -- which is
  also how a head on another card than block 0's hands the rows over: the
  head's card gathers, and block 0's uploads them as the residual crosses
  between cards. The tables are whatever the graph reads, held for the
  model at PrepHead: the head where it is the embedding, the token
  embedding beside it where not (nn.Head.TokEmbd, tinyllama and stories15M),
  and a learned position table (nn.Head.PosEmbd, starcoder), indexed by the
  positions EmbedRows now carries. A tier with ComposedEmbed off keeps main's
  prompt gather. The State promises the rows wherever the front end states
  them (Head.GatherRow is set only then): embedOnDevice's family tests
  (DSV4, Q4E, ResAttn, AltUp, Streams, PLEDim, the position table) are gone,
  the front end's UncoveredStreams and UncoveredPLE refusals covering them,
  and the prefill's position add is skipped where the device gathered. The
  one decline that named three causes is three: partEmbedFormat,
  partEmbedWeight, partEmbedToken. ScaleFaultEmbd is a rewrite of the
  graph (its scale dropped), not a hook in a kernel. On a V100 (CUDA and
  Vulkan, graph on and off) the serving gate's whole-step arm gathers every
  decode row and the prompt's on all 25 served models; the composer's prompt
  rows leave the prompt's logits equal to the tier's bit for bit, and a row
  gathered at the next token's reads a median 61 to 2620 times the tier's
  distance. The table's violation, every row told a position 16 past its
  own, read 1.03 to 22.6 and sat inside the band on Qwen2.5-VL-3B (1.22 and
  1.36): a rotary turns by differences, and every decoded row is shifted
  alike, so only the prompt's keys move. It is now twice the position and 16
  past, which reads a median 8.2 on Qwen2.5-VL-3B, 73 on Llama-3.2-1B and
  71 to 77 on stories15M. TestC6ServesComposed's whole-step arm runs every
  classic fixture with its rows gathered on the device, synth-starcoder's
  position table among
  them (4.0e-14 from the CPU on CUDA, a row at the next token's 1.65e-2, at
  the next position's 8.7e-3). TestRefusedInputRowsStayOnTheCPU
  holds the refusals: gemma3n (AltUp and per-layer embeddings), deepseek4
  and qwen4exp (hyper-connection streams) are refused as streams and
  synth-gemma4-e as per-layer embeddings, no device is asked for their rows,
  and with the plain rows' graph stated in their place gemma4-e's decode
  parts from the CPU's by 2.46 while the stream models' rows are declined by
  name every step. TestDecodeDoesNotAllocate's composed arms on
  Llama-3.2-1B make no engine allocation with every row gathered, both
  builds. Escape analysis against the tree before (jit/gpu/tier,
  engine/model, engine/nn): the forms' builds, the fault rewrite, error paths
  and the staging bytes' growth, none on a warm step's path; the cross-card
  read-back's session closure allocates, as main's did. Open: the sampler's
  top-k with its penalty and the label pick (partSampleTopK, partSamplePick):
  OpTopK's ranked lowering is quadratic in the row (1.6e10 compares at a
  128k vocabulary), the penalty has no op, and its correctly rounded
  division is PTX's alone; a multi-axis local table, which no model states.
- **A placement compiles none of the tier's prompt kernels for a block the
  composer serves.** The walk of every warmed prompt width compiled each
  block's batched matvecs, fused operands and the staged attention's span
  variants beside the composer's form of the block. A block whose prompt the
  composer serves at a width, its instance held, is now left uncompiled there
  (`promptHeld`, `Stats.PromptBlocksLeft`); a call whose blocks decline (rows
  of a ragged step, every row's logits, history streamed, an instance the
  budget refused) has their kernels compiled before its submission
  (`promptDeclines`, counted by reason in `Stats.DeclineBuilt`), and a chunk
  whose every block is served stages no prefill attention. Widths up to a
  ragged step's compile as before. A first attempt left `voltaBlock` asking
  every block's twins at the chunk, and the first prefill compiled 34 kernels
  (`TestPlacedPromptsCompileNothing` named them); the chunk no longer asks a
  left block. Coarsen's per-unit values are a period of the rows and a step a
  row wherever the op keeps that form (`uval`), every unit's word elsewhere,
  each function numbered once: the same classes, so the 25 served models'
  lowered kernels at one row (both statements, int8 and float, PTX and
  SPIR-V, 100 arms, 1,882 distinct kernels) hash as at 95645ee5, and Coarsen
  left the profile (33 s before). Vulkan now states its compute units from
  the vendor extension `GridLimit` reads, so RowTiles' 30,720-thread fill is
  used only where no extension states one; CUDA
  (`MAX_REGISTERS_PER_MULTIPROCESSOR` over its most resident threads, 32 on a
  V100) and AMD's Vulkan (`vgprsPerSimd` over `wavefrontsPerSimd`) state a
  thread's registers, and a tile takes the rows whose sum and term fit them,
  16 rows on a V100, 16 where a device states none (NVIDIA's Vulkan). The
  composer's forms compile side by side where the device's compiles run
  beside each other: Vulkan, whose pipeline creation the specification
  leaves unsynchronised (48 matvec kernels on a V100: 983 ms on one
  goroutine, 190 ms on eight); CUDA's driver serialises them (0.67 s on one
  thread, 1.2 to 1.4 s on two to eight), so it compiles in turn. Placement of
  Llama-3.2-1B at the 512-row width with the blocks, head and prompt served,
  V100 card 3, single runs on a shared box, drivers' caches off (cold) and
  on after a filling run (warm): at 95645ee5 1 min 55 s cold and 49.1 s warm
  on cuda:0, 2 min 57 s and 46.8 s on vulkan:3; with the walk left and
  Coarsen periodic 49.4 s and 13.4 s, and 54.9 s and 13.9 s; with Vulkan's
  compiles side by side 40.5 to 60.7 s cold (the tier-only arm read 37.5 and
  42.2 s on the same box in the two passes) and 13.8 s warm on cuda:0, 43.8
  to 47.9 s and 14.4 s on vulkan:3. The tier-only arm is unchanged in work
  (686 compiles at 512 rows, 37.5 to 42.2 s cold, 2.9 s warm). A CPU profile
  of the cold CUDA prompt arm names what remains: the driver compiling the
  composer's forms (30 s: 16 widths of about 28 kernels; CUDA cannot run them
  side by side), the tier's scratch kernels each width's scratch still
  builds (`initScratch`, 12.5 s), and forming (11.5 s, FMA's and Affine's
  passes). Gates on the V100 (cuda:0, vulkan:3): `TestPlacedPromptsCompileNothing`
  (6 arms; Ministral-3-3B is not on the box), `TestPrefillMatchesForward/device`
  (16 arms), `TestComposedBlocksServe` on the 8 `promptModels` (14 each),
  `TestDecodeDoesNotAllocate` on Llama in both builds (28 each), the jit
  package (`TestRowTilesHoldTheOps` now also holds the registers' sizing:
  657 ops a tiled kernel at 512 rows with 32 registers or none stated, 218 at
  4; `TestCoarsenClassesAreTheUnitsValues` fails when the division rule
  drops its divisor check). `go build -gcflags=-m` over the touched packages:
  three new heap moves, all at placement (a CUDA attribute read, the compile
  workers' wait group and count).
- **Latent attention as ops.** The front end states a block that projects
  its history through a latent (the container stores RoleAttnKVA) from
  generic ops and the container's widths, reusing main's absorbed form and
  its paged latent pool: the query down, normed and up where the block
  stores the two projections; the latent row's first KVLoraRank normed (at
  LatentNormEps) and the rest turned by the rotary, joined as the cached row,
  appended once; each head's first HeadDim-NRot through its sheet of
  RoleAttnKB (an expert matvec with no selection: pair j reads sheet j),
  joined head by head with its turned rest (OpConcat with NHead); attention
  of every head over the one cached row a position, the value the whole row;
  each head's first KVLoraRank of the result through its sheet of RoleAttnVB,
  then the output projection. The composer's history layout gains
  ValuesInKeys, one pool whose value rows are its key rows, which the tier
  states for a latent pool (`poolHistory`), so the append writes the row once
  and the attention reads its values from the keys' pool. `UncoveredLatent`,
  `UncoveredValueWidth` (a value head of its own width outside a latent is a
  head width, UncoveredGeometry, and main runs none) and the tier's
  `partLatent` are gone. The value accumulates over the whole row, NRot more
  than the KVLoraRank the result keeps (64 of 576 on DeepSeek-V3): a cost,
  stated, for one head width through every attention pass. On a V100
  (card 5), cuda:0 and vulkan:5, `TestComposedBlocksServe` serves
  synth-deepseek-dense every arm, every block step, none declined (median
  ratio to the tier's distance 0.985 on CUDA, 0.898 on Vulkan);
  `TestComposedLatentViolations` holds the composed blocks to the band
  (median 0.927 and 0.895) and sees the latent norm dropped (median ratio
  9.5e11), the rotation a pair short (3.6e7 to 4.4e7), the banks swapped
  (5.0e12) and the value read from the key's end (5.5e12); the same fixture
  stated without position encoding (0.959, 0.952), the YaRN fixture (0.957,
  0.946) and DeepSeek-V2-Lite at Q4_K_M, whose dense lead block the composer
  serves at every step beside 26 mixture blocks the tier runs (1.011 and
  0.991), pass too; `TestComposedCPUHeadSheets` runs the two new statements
  on the CPU device. The block is formed once for its device at placement,
  as every block is; no form of a width or a prompt was added for it. The MLA device gates pass in both builds;
  in the composed build `TestMLAOnEveryDevice` counts the composer's block
  steps where it serves every block (its attention binds the tier's paged
  pool) and the tier's paged launches where the tier runs any, and
  `TestMLADeviceFaultsAreCaught` asks the tier's own kernels, whose faults
  they are. Every real latent checkpoint on the box (DeepSeek-V2-Lite and
  Coder-V2-Lite, Moonlight, GLM-4.7-Flash, Kimi-Linear) is a mixture past its
  dense lead, refused by `UncoveredMixture`; deepseek32 is refused by
  `UncoveredSelect`, Kimi-Linear's and K3's linear blocks by `UncoveredKind`.
  `TestDecodeDoesNotAllocate` measures the composed step on synth-deepseek-dense
  too (`composedAllocModels`): 0 engine allocations on cuda:0 and vulkan:5,
  its input rows declined by name (the fixture's embedding is not its head's).
  `go build -gcflags=-m` over the four touched packages moves nothing more to
  the heap (247 lines either side).
- No device is a default where blocks live. `model.WithCPU(listed)` says
  whether the caller listed the CPU among its devices (`tier.AdmitsHost`:
  `auto`, or a `-devices` list naming `cpu`); `cmd/jitllm run` and the
  server's `LoadModel` pass it from the spec. Listed, the engine may choose
  the CPU as any other device. Not listed, every choice the engine made for
  the CPU goes to the cards or fails by name: placement marks every block it
  decides streamed where it is (`State.pageOnCards`, no block released or
  formed again) and offers the rest, so the model pages through the card's
  slots; the output projection is taken while a swap keeps its slot
  (`nn.Head.NoCPU`, the streamed blocks paged out for it); `placeHead`
  buys no room with a block, a fault does not demote (`fallsBack`),
  relocation spills only to a later card, a batch with no room and a
  batch's rows that do not fit are the step's error, and the stream trial
  and the seam tuner, whose arms run on the CPU, do not start. The server's
  orchestrator takes a victim only from a model whose devices list the CPU
  (its gate's fixture now lists it). A caller's own move to the CPU is
  honoured either way. The placement's CPU is spelled `cpu` (`host` read as
  its older spelling), the CPU-run accounting is `cpuRuns`, the relocation
  to the CPU `relocateToCPU`, the batch counter `RoomToCPU`, and ten test
  names lost "home" and "on the host". `TestBlocksPageOnTheListedCard`:
  Llama-3.2-1B on one V100 card at three fifths of its weights (459 MiB),
  the CPU unlisted, holds 16 of 16 blocks and the head on the card at every
  step, 124 page-ins on cuda:0 and 126 on vulkan:2, the resident control's
  greedy token at every step (worst logit 0 on cuda:0, 0.469 on vulkan:2),
  and each block formed once on the card in the composed build
  (`tier.Stats.BlockForms`, 144 page-ins) and once more after a caller's move
  to the CPU and back; a 1 MiB card fails `SetDevice` naming every block.
  Its violations: the CPU listed spills 7 blocks (13 composed) and the check
  names them, and the first form of `pageOnCards`, which released the
  resident blocks and offered them again streamed, formed blocks 0-7 twice.
- The CPU device serves composed blocks (`WithComposedCPU`, off by default).
  A block placed on the CPU is composed once, at its placement (a new
  State, or `MoveBlocks` to "cpu"), and its composition released as it
  leaves; `State.CPUCompositions` counts them and no step composes. Its
  history is the layer's host KV pages held as one off-heap span
  (`kvArena`), page n at slot n, so a declined step's main kernels and a
  composed step read one set of pages. The census is the tier's: the
  decline names moved to `jit.Decline`, the tier's `partDecline` an alias
  with its names and numbers unchanged, and the CPU's reasons appended
  (a windowed, latent, shared, recurrent or batched block, a history not
  binary16, a CPU below AVX2/FMA/F16C, a block not composed at placement);
  a prompt's or a batch's block step on the CPU is counted declined
  (`DeclineBatch`). Two composer bugs were behind C2's paired-head
  attention reading 0.32 to 33 and changing run to run: the CPU's register
  binding took a load folded into its reader as a memory operand to read
  its index where the load sat, so the index's register was handed to the
  history's next index before the reader ran and the query was read at the
  history's offset, out of its buffer (`binder86.lives`, fixed; a spill
  tie now takes the lower node); and the CPU call site keyed a kernel's
  constant block by its name, so the attention's 1/sqrt(d) scale was the
  residual add's coefficient (`buildCPUForm`, each kernel's block its own).
  `TestComposedCPUAttend` holds the decode attention to the definition at
  NMSE 9e-14 to 3e-13 over the served shapes. `TestComposedCPUBlocksServe`
  serves 80 steps x every block of stories15M, Qwen3-0.6B and Llama-3.2-1B,
  each block composed once, at 0.475, 0.500 and 0.858 of main's own spread
  from main at the median (the control is main at another activation
  window; the composed CPU has no int8 form), worst 0.0046, 0.052 and 0.39
  against the control's 0.0073, 0.024 and 0.26; a page pointed at another
  slot and the first row moved sit at 16.9-229 and 2.7-14.1 times main's
  spread at the median. On the V100 box's Xeon, `TestDecodeDoesNotAllocate`'s
  host-composed arm on Llama makes 0 engine allocations over 64 tokens with
  all 224 x 16 block steps served, at about 8.3 s a token where main's
  whole 224-token arm took 6.7 s (not a RULE 2 row): the composed matvecs
  are single calls on one core, the largest gap.
- **Activations are expressions the container states.** The engine knew
  each activation's formula twice: jit's `exprs.act` switched on a named
  `graph.Act`, and the front end built xIELU's maps itself. The converter now
  states every block's FFN activation in the op language
  (`convert.StateActivations`, `jlm.Config.Acts` and `ActLayers`: terms of
  `jlm.Fn` over the gate and up rows, constants, a stored vector's elements,
  row reductions and the row's width), carried as a thirteenth optional tail
  of the config record, which an older reader drops with no version bump; the
  front end copies it into the block graph (`graph.Builder.Stated`: reductions
  as `OpReduce`, the terms between them one `OpExpr`), and the composer lowers
  the functions it is handed (`decomposeExpr`, with `tanh` and `erf` now maps
  of their own). `OpActMul`, `OpAct`, `graph.Act`, the named switch and the
  front end's xIELU are gone; Gemma 3n's gaussian top-k is a statement too,
  so `UncoveredSparseAct` is gone, and `UncoveredAct` now names a container
  that states no activation (one written before the tail); `jitllm upgrade`
  states it from the container's own config and tensors (`jlm.Restate` with
  `convert.StateStored`), byte for byte the container a fresh conversion
  writes (`convert.TestRestatedIsAFreshConversion`; containers the converter
  wrote before the tail, of stories15M, synth-nemotron, synth-apertus and
  synth-gemma3n, upgrade to their fresh conversion but for the 20 bytes of
  the writer's name in the fingerprint). An expression forms one map; where the
  register binding refuses a kernel for pressure (`jit.ErrRegisterPressure`,
  `PressureError` naming the parts), the largest expression formed whole in
  that part is formed a map a term and the run passed again, at compose time:
  xIELU binds whole on one element and as nine maps over a vector loop, and
  synth-apertus joins the served models (median ratio 0.906 and 0.915 to the
  tier's distance on a V100's CUDA and Vulkan). No served model states GELU
  with erf: only modern-bert's entry does. Its
  whole map exposed an ordering bug: a twin's reader was emitted with a float
  zero it reads left out of the sequence (`beforeShare` counted every
  constant as made), so the lowering found "node 22 (const) has no register";
  a constant counts as made only where no entry makes it. The family-name
  counter over jit/, engine/graph and front.go (names of families and of
  activations, outside comments) went from 4 files to 0. `convert`'s
  `TestStatedActivationsAreTheReferences` holds nine statements (SiLU, GELU
  tanh and erf, squared ReLU, xIELU, the gaussian top-k, both clamped
  SwiGLUs, situ) to their references and tells every moved constant and
  swapped select apart. The violations that swap a block's activation
  ("GELU for relu2", "SiLU for relu2") set it in the model's config and in
  the container's statement through the converter (`actSwap`), so the
  composed arms see them: 0.713 and 0.717 of NMSE on synth-nemotron, both
  builds. A stated constant moved by one reads 2.35e-3
  (falcon), 3.54e-3 (starcoder), 5.27e-3 (falcon40), 5.65e-3 (phi2), 9.79e-3
  (starcoder2), 1.31e-2 (nemotron), 6.74e-2 (commandr), 0.134 (stablelm),
  0.140 (stablelm-par) and 0.169 (cohere2) in `TestC6ServesComposed` on a
  V100's CUDA and Vulkan, and 2.36e-3 (falcon) to 0.171 (cohere2) on the
  nineteen fixtures of `TestC6OnEveryDevice` on both, against a
  bound of 1e-4 and a hundred times a clean arm's 7e-12 at most.
- The CPU device's composed decode, from 280 times main's time a token to
  about twice. Profiled per kernel on Llama-3.2-1B at Q4_K_M (the Xeon
  E5-2680 v4, one socket's twelve cores, `numactl --membind=0`; one run of
  each arm, not RULE 2 rows), the three gaps were: the float form, which
  decodes each packed weight element by scalar code (block 0's gate, up and
  down matvecs 141, 140 and 87 ms each, 7475 ms a token); every matvec one
  call on one core; and the int8 matvec's index arithmetic, recomputed and
  spilled per window. The CPU profile now keeps the int8 form (windows of
  32, VPDPBUSD where the device states AVX-VNNI, the AVX2 byte-dot path on
  the Xeon, which has none): 853 ms a token. `Workers` deals a kernel over
  a weight's rows across the device's workers (`Device.Slots`, the pool's):
  the space's last axis cut into calls, the cut that leaves the busiest
  worker the fewest rows, each call a multiple of 16 rows; Args moves the
  call's rows into its buffers' offsets (`CallArg.Per`), which the CPU call
  site adds per call, so the dealt kernel is the one-worker kernel byte for
  byte, its cursors, halves and jam intact: 78 ms. Simplify takes the low
  bits and the shifts of aligned sums and products and folds constant
  differences, minima, maxima and shifts by zero; it no longer merges two
  loads of a weight's word (streamed, a second read hits the cache, and the
  register one merged load holds across the body is one the jam needs: Q4_K,
  Q5_0, Q5_1 and Q5_K shapes stopped binding their vector registers when
  the index folds made each nibble pair's word one load); and `Periods` runs a loop of sums a whole period of its iteration
  a trip where it indexes by the iteration's low bits, so a super-block's
  sub-block scales are constant words and shifts (the Q4_K loop body 2240
  instructions per eight windows before, 1267 after, a third of them still
  spills). Regs keeps its vector words while it binds (the period's larger
  bodies made `sink` quadratic: 58 s for `TestComposedCPUMatVec` against
  30 s). A cold placement composes faster, not slower: a new State's
  blocks on the laptop composed in 0.98 s for Llama-3.2-1B against 48.9 s
  before (the float form's bodies), 0.14 s against 0.21 s for Qwen3-0.6B.
  `TestComposedCPUAB` on the Xeon, 16-token rounds, n = 20: the A/A
  controls read 0.9974 (IQR/median 1.6%) and 1.0105 (3.1%); composed
  against main read 0.504 on Llama-3.2-1B (803 MB a token, main 54.2 tok/s
  = 43.6 GB/s = 78% of the 56.0 GB/s wall, composed 27.6 tok/s = 22.1 GB/s
  = 39%) and 0.528 on Qwen3-0.6B (634 MB, 62.2 against 33.4 tok/s, 68% and
  36% of the 58.2 GB/s wall), both UNSTABLE (IQR/median 14.9% and 47.6%,
  the composed arm's): not quotable rows. On the laptop the A/A control
  failed under other jobs' load (0.954 and IQR/median 13.4%). One laptop
  run at six P-cores, per kernel: composed 27.2 ms a token against main's
  16.1, its block kernels 22.9 ms, the largest the Q6_K down projection
  (8.7 ms, 25 GB/s), gate and up (4.9 ms each), q and o (1.4 ms each).
  `TestComposedCPUMatVec` runs 64 rows at one and six workers and 48 and
  80 at six, a guard past the output: 180 formed and right, and a call
  base one row short fails all 108 six-worker arms. `TestComposedCPUBlocksServe`
  holds stories15M, Qwen3-0.6B and Llama-3.2-1B at 0.543, 0.438 and 1.054
  of main's spread at the median, every block step served, each block
  composed once, on the laptop and the Xeon; `TestDecodeDoesNotAllocate`'s
  host-composed arm on Llama makes 0 engine allocations over 64 tokens on
  both; `TestBatchMatchesForward`'s and `TestPagingDoesNotChangeTheAnswer`'s
  composed-CPU arms pass on the laptop, and `TestEveryModelRunsGenerated`'s
  ran 152 composed-CPU arms on the Xeon with no error (every block step of
  the served families served, bitnet-b1.58's ternary blocks among them)
  before the run was stopped, still converting the rest. The GPU forms of the three served models hash the same as before
  (78, 308 and 144 kernels at one row), and `go build -gcflags=-m` moves
  one value more to the heap, at compose time (the Workers pass's copy of a
  range).
- **A mixture is a graph of generic ops.** The front end states a mixture
  block (`Model.mixture`): the router's matvec with its stored bias, each
  expert's score the expression the container states over the logits
  (`jlm.Config.RouterScore`, an optional tail the converter writes:
  softmax, sigmoid or sqrt(softplus) as `convert.RouterScore` says, held to
  the references by `convert.TestRouterScoreIsTheReferences`), the
  selection bias, a group limit as a top-k within each group, a reduction
  and a rank, `OpTopK` and `OpTake` for the weights, the renormalisation
  and the routed scale, the banks as `OpExpertMatVec` over the selection,
  and a shared expert with its sigmoid gate. `UncoveredMixture` is gone;
  what is left is named: `UncoveredRouterScore` (a router whose container
  states no score, phimoe's sparsemixer) and `UncoveredExperts` (experts
  weighting their input, a dense MLP beside the mixture, latent experts,
  per-expert scales), and `DeclineExperts` where a bank is not whole on the
  device (streamed or cached blocks, and the CPU device, which keeps every
  expert in a page of its own). Thirteen mixtures join the served models in
  both builds, every arm on a V100's CUDA and Vulkan: Qwen3-MOE-4x0.6B,
  OLMoE-1B-7B, DeepSeek-V2-Lite and the glm4moe, qwen2moe, ernie45moe,
  hunyuanmoe, minimaxm2, bailingmoe2, dots1, gpt-oss, deepseek and
  deepseek-lite fixtures, at median ratios 0.836 to 1.032 of the tier's
  distance from the float reference on CUDA. Six router violations, set in
  the model's config and in what the front end reads, fire on both devices
  in `TestMoEFamiliesOnEveryDevice` against its bound of 1e-5: the
  selection bias dropped (5.48e-2 to 1.16), the group limit removed (0.570),
  the renormalisation removed (2.83e-2 to 1.31), one expert fewer (0.246 to
  1.15), the routed scale dropped (0.356 to 0.613) and the shared expert's
  gate dropped (0.880). The swaps of a score ("softmax for sigmoid", "sigmoid
  for softmax") set the config alone and read a clean 4e-12 in the composed
  build; they restate the container's score through the converter now
  (`routerSwap`) and read 0.121 to 0.789, and bailingmoe2's "rotary over the
  whole head" sets the rotary the front end reads (0.753, from a clean arm).
  A part the registers refuse with no expression left to split is formed
  unfused. A model whose rows the tier's int8 activation blocks of
  32 do not divide (stories260K's 172-wide MLP, tiny-qwen3moe's 8-wide
  residual) is refused at placement by that name before any kernel is built
  (`TestTierNamesTheRowsItCannotBlock`), where its quantizer's error stood.
  Open: on the CPU profile that takes the int8 form, Qwen3-MOE-4x0.6B's
  mixture parts refuse the registers in `TestFrontStatesTheModelAsOps`
  (fused pairs of reductions of 557 to 1263 nodes, then the reduction alone:
  "more words live at once than registers"), on the laptop and on the Xeon;
  the CPU device serves no mixture (`DeclineExperts`), and the GPU forms of
  every mixture form. Before that profile the gate formed it on the CPU.
- **The rows the composer serves: a ragged step, a chunk's head, a prompt's
  head.** A head is the head graph at the rows it projects, through the one
  keyed form cache (the row count in the key): a chunk's folded last row and
  a prompt's one-row call take decode's one-row form (a prompt's, folded or
  not, asked where the prompt is, ComposedPrompt), a ragged step's wanted
  rows and a chunk wanting every row's logits a form at their count, written
  where the tier's projection wrote them, then capped and picked as before.
  A ragged step's blocks are asked as decode's (ComposedBlock) and served at
  the step's rows: each row's descriptor already carries its own page table,
  range and write position, so the R-row form a prompt chunk runs reads each
  row's own history; a prompt's ragged rows are the prompt's. The shadow arm
  serves a prompt as the composed arm does and compares a decode step's
  one-row forms. A server's step width (Config.StepRows) forms its blocks and
  its head at each wanted-row count at placement; an unreserved width forms
  at its first step, as the tier's own kernels for it compile there. The six
  declines that named these paths are gone with the code that produced them.
  On the V100 (cuda:0 and vulkan:6): TestComposedRowsServe (Llama-3.2-1B,
  a batch of three prompts of different lengths decoded together, 880 block
  steps and every head served, 12 tokens a row the host's greedy to a tie),
  TestPrefillMatchesForward/device (12 arms, the shadow arm's 16 prompt
  block steps and the folded head served), TestPrefillFoldsTheHead and
  TestPrefillFoldsALayerNormHead in the jitllmcomposed build (which does
  not ask the prompt: NMSE 8e-15 to 5e-14 fold against tail, 8 tokens
  equal), TestComposedHeadServesWhereTheBlocksDoNot and
  TestComposedPromptIsNotServed pass. TestComposedRowsDiscriminate
  (jitllmfault) fires on each violation: a ragged row stating half its
  position ("pos", 13.9 apart at token 1), every row reading row 0's pages
  ("alias", 9.5), the per-row head projecting the first row alone
  ("headrow", 12.0), the folded head reading the chunk's first row
  ("foldrow", NMSE 0.96). A position off by a constant does not show: the
  rotary is relative. TestDecodeDoesNotAllocate on Llama-3.2-1B makes no
  engine allocation in either build; TestBatchSeamMovesCarryEveryRow,
  TestBatchMatchesForward, TestRaggedBatchMatchesForward,
  TestForwardBatchGreedyMatchesEachSequenceAlone,
  TestSchedulerOnDeviceMatchesEachSequenceAlone and TestSpecLookupOnDevice
  pass in the jitllmcomposed build. Escape analysis of jit/gpu/tier against
  the tree before (go build -gcflags=-m, line numbers dropped) is unchanged.
  Open: TestABatchLeavesAnotherSessionAlone's composed arm forms 34 parts at
  B's first steps (a width no placement reserved, and the head at B's
  wanted-row counts). With StepRows the head counts form at placement, and
  16 still form at a step: on a device with no integer matrix instruction
  (sm_70) a ragged step runs at a multiple of 8 (rowsWidth) while StepRows
  reserves a multiple of 32 (stepRows), so a 3-row step builds an 8-row
  scratch at its first step, beside the reserved one, for the tier's own
  kernels as for the composer's forms.
- **The served head writes its normed rows.** The head the composer serves
  is the front end's graph with the norm's rows a second output
  (graph.Head.Hidden, nn.Head.HiddenGraph), written into the scratch's
  normed rows where the hidden readback reads them, so a speculation step's
  head is served and the decline "a head whose normed row is read back" is
  gone; it is the same form at the same row counts as before, with one more
  output. TestComposedHeadWritesItsHidden (Llama-3.2-1B, cuda:0 and
  vulkan:6) holds the rows read back to the tier's own head's: a 6-row
  verification as ragged rows, one row as decode and a prompt chunk's
  folded last row, worst NMSE 1.2e-3 to 2.0e-3 from the tier's own against
  0.33 to 0.87 from a neighbouring position's (Stats.HiddenHeads,
  HiddenRows count the served heads read back). Under "nohidden" (the
  logits-only graph served) the rows read back sit 1.0 from the tier's,
  caught on both backends, beside pos, alias, headrow and foldrow (now at
  32-row chunks of a 40-token prompt, NMSE 0.94). Open:
  TestABatchLeavesAnotherSessionAlone's composed arm still forms at a
  session's first ragged step (34 parts): the ragged blocks and the per-row
  head are formed per row count, which composing once with the rows a
  run-time argument removes.
- **A block is composed once for its device; a call's rows are an
  argument.** A GPU form of several rows is a capacity form: each kernel
  whose leading axis is the step's rows carries the capacity
  (jit.Kernel.Capacity), reads the call's rows from one word (the table
  "rows.n") and is held to them by CallRows -- a kernel of independent
  threads has its threads clamped to the live elements, a kernel whose
  threads combine is held by its launch alone and may not bind the
  history -- and launches Code.LiveGroups(rows). RowTiles sizes its tile
  from the device's registers alone, its last tile's surplus rows guarded.
  The tier writes the call's rows each call (blockScratch.calls, headCalls),
  and a prompt, a ragged step and a head over rows the composer serves run
  in the reserved width at their own rows: no exact width, no warmed width
  and no per-count head or ragged twin. On Llama-3.2-1B (V100, Vulkan; the
  CUDA arms of that run skipped on a box-wide cuInit failure) placement
  composed 25 forms (83 kernels) and built one scratch where the per-width
  placement composed 55 (683 kernels) and built 16, and prompts of 1, 5,
  33, 100, 511, 512, 513 and 2000 tokens with decode after each composed,
  compiled and built nothing (TestPlacementComposesOnce; its violation, a
  form at each chunk's grain width, builds a scratch at a 5-token prompt).
  The capacity is still in jit.Key: the forms do not move with a call's
  rows, but their codes and schedules still read the capacity, and the CPU
  device's forms have none.
- **Nothing composes, compiles or builds a scratch after placement.** A
  ragged step the composer serves runs in the reserved width at its own rows
  whether the prompt is asked of the composer or only the blocks
  (rowsComposed: a step's blocks are decode's, ComposedBlock), and so do the
  room check before it (stepRoom) and the scheduler's admission (RowsWidth
  answers the reserved width and builds nothing, so a batch of more prompt
  rows than the width is admitted a width a step; a wider step is refused by
  name). The per-row argmax and softcap of the reserved width are built at
  PrepHead (prepRowsReserved), the tier's one-row head over a wider scratch
  and its ragged flash prefill are not built where the composer serves the
  head and every block, and a prompt longer than the width runs in chunks of
  it with the head folded into the last. A server's step width is a floor on
  the reserved width, not a scratch and forms of its own, and the 32-row
  rounding is gone from that path. Relocation forms the rotary tables on the
  destination with the block. A KV compaction's page gather reads its page
  count at run time (kernels.PagedGatherPages(words)), compiled where the
  pool layer is made, where it compiled per count at a second session's
  prompt on a 4 GB card. The reserved width narrows from the budget when the
  composer's instance at it does not fit (fitReserved); Stats.ReservedRows
  names a card's capacity: 512 rows on a V100 (Llama-3.2-1B), 256 on the
  laptop's 4 GB RTX 3050 Ti, whose composed prompt block needs about 1.1 GB
  of buffers at the census's 3.4k-position session. Sessions share the
  capacity scratch: a lane is a call in flight, not a session, and a second
  session or the scheduler's State runs in lane0's scratch with no build; a
  clone lane is built only for calls that overlap, which hold their own
  intermediates (the KV is the pool's, owned per sequence).
  TestPlacementComposesOnce now also runs a four-sequence batch of 592
  prompt rows against the 512-row width, closes the other sessions and moves
  half the blocks to the CPU and back (each block composed once at placement
  and once more at the move, Stats.BlockForms), and has a two-card arm (the
  upper half and the head on a second card, then the lower half moved over);
  it counts DeclineBuilt and FormedAtStep beside forms, scratches and
  compiles, and names any kernel compiled after placement. A prompt whose
  range holds blocks the caller keeps on the tier's kernels runs in the
  reserved width too, those blocks' kernels at it compiled at placement.
  On a V100 (Llama-3.2-1B, card 3; the two-card arm cuda:0 on card 3 and
  vulkan:4): placement composes 10 forms (68 kernels) and builds 1 scratch
  on CUDA, 15 forms (127 kernels) and 2 scratches over two cards; after it,
  on CUDA, Vulkan and both cards, the eight prompts, a second session, the
  592-row batch, the move and the prompts after it compose, compile and
  build nothing. Its violation (a form per chunk width) places 55 forms
  (683 kernels) and 16 scratches and builds a scratch at a 5-token prompt.
  The tight arm of TestABatchLeavesAnotherSessionAlone, where B kept none
  of 16 blocks, asked the step's room before the batch's pages were
  reserved, and the allocator's rounding then carried the card 327,680
  bytes past its budget; asking after the reservation keeps B 3 of 16 (4 of
  4 runs failed before, 4 of 4 pass after). The composed buffers had grown
  with the key bound: a binding stayed written after Stage moved the
  attention's per-key temporaries into workgroup memory (517 MB at 256 rows
  over 3328 keys, 13.6 MB now), a key bound of 2048 or less never split its
  range, and the split and work-fused attention launched every reserved row;
  the 4 GB card now reserves 512 rows. Prefill against the tree before the
  capacity forms holds (V100 card 7, CUDA, Llama-3.2-1B Q4_K_M, tuning off,
  20 rounds interleaved, A/A 1.004): pp512 1.001, pp1024 0.984 (the A/A
  0.975 at an IQR/median of 0.074), pp4096 0.999, Qwen3-1.7B pp4096 0.999.
  A fall from 14.5k to 5.85k tok/s at pp4096 read in three rounds came from
  a binary built before the capacity forms' last commit, whose prompt
  attention ran another kernel (flashkvpaged, 1.40 s over three prompts,
  against flashprefill70paged's 190 ms); it does not reproduce.
- The CPU device serves rows. A block placed on the CPU is composed twice
  at its placement, once: decode's one-row form and a form at a capacity of
  64 rows (cpuRowsCap), whose calls run the rows the call states. The
  capacity is the GPU's (Kernel.Capacity, liveOf now device-generic); the
  CPU's RowCalls pass holds a kernel to the call's rows by keeping its
  leading axis, the step's rows, as the calls' space (rows and query heads
  for a kernel over heads), and the call site runs the units the rows reach
  (Code.LiveUnitsOf, which the GPU's LiveGroups now reads too). A prompt's
  chunk (hostRows), a batched step (ForwardBatch), a step across sessions
  (StepRuns, each session's rows on its own form and history, the rows after
  the logit rows first) and several sequences in one State are served: each
  row's descriptor names its sequence's page table, the pool addressed a
  sequence's page at a time (a page holds every sequence's positions), the
  KV writes through kvWritable, grow, cow and the arena. A page the KV budget
  evicted is faulted back into its arena slot (kvCache.fault), so a paged
  history is served rather than declined (DeclinePageOut); a composed State
  attaching a card at position 0 keeps its binary16 history and arenas.
  Kernels over (row, head) calls keep their indexing (Cursors skips them):
  with cursors, a row whose every score sat far below zero (Qwen2's biased
  keys) took its softmax's maximum past its range and came out uniform,
  97 times main's spread on Qwen2-VL at 33 rows; TestComposedCPUAttendRows
  holds that case. Gates: TestCPUComposesOnce passes on stories15M and
  Qwen3-0.6B (prompts to 2000 tokens, a second session, a scheduler's four
  sequences, and the lower half moved to cuda:0 and back: 2 compositions
  for the moved blocks, 1 for the rest, none after), nothing declined;
  TestComposedCPURowsServe holds prompts (5 to 513), a paged prompt (11760
  pages evicted, 10920 faulted back on Qwen3), a 3-sequence batch and 3
  sessions stepped together inside main's band (median 0.29-0.63 of the
  spread), both violations leaving it; a warm 33-row chunk and a warm
  4-sequence step allocate nothing (countAllocs). Speed on the Xeon
  D-2123IT, 4 cores, Llama-3.2-1B Q4_K_M, A/A 1.002-1.003: composed prefill
  is 0.085 of main's (pp33 10.9 against 129.8 tok/s, pp100 12.1 against
  141.9, pp512 11.9 against 140.0): a row a call re-reads the weights per
  row, and kernels writing a scratch run their calls serially. Composed
  decode reads 0.198 of main (6.7 against 33.7 tok/s) where 1155a8d9 reads
  0.198: not regressed.
- **The sampler's candidates and the label pick are each one form at
  placement.** The sampler (nn.Head.SampleK, the repeat penalty and top-k)
  is graph.SampleTopK at nn.SampleMaxK and SampleMaxHist: the K+Hist best of
  the row, the penalty on those alone (a penalty above one never raises a
  value, so the K best after it are among them), applied once per history
  entry in sixteen-step expressions whose division is MapDiv's exact
  quotient, then ranked by id and by value so that a tie goes to the lower
  id. Neither k nor the history's length enters the form: the host reads
  the first k, and the history's words past its count are an id no row
  has. Its selection is a new lowering of OpTopK for a group past 1024
  (decomposeTopKSelect): the K-th largest order key found four bits a round
  (eight rounds of fifteen counts), the keys above it and the first equal
  ones written in index order by block prefix counts, then the K ranked --
  a few passes over the row and K^2 compares, where ranking the row was S^2
  (1.6e10 at a 128k vocabulary). Counting a candidate's history entries
  takes MapEqual (jlm.FnEqual; over two index rows their integers, by the
  bits of their difference: through two compares, the CPU device's vector
  lanes held words 2^25 apart equal). The label pick
  is graph.Pick (OpTake of the logits at the ids) or, where the composer
  serves the head, nothing is left after the projection and the head is
  Q8_0, the head with Pick set: a row-selected matvec (OpMatVec with a
  selection), decomposeMatVecQ8's own form at the selected rows, so a pick
  is the row's value to the bit; any other format reads the rows by expert
  matvec's decode (a different order of the same sum, 1e-6 from the row on
  lev) and the tier gathers instead. DeclineSampleTopK and
  DeclineSamplePick are gone (41 declines to 39). Three composer bugs these
  forms met: computedFrom walked a chain of nodes that each read the one
  before twice in two to its length (a sixteen-step penalty hung forming);
  the prologue pass merged a scatter into a reader that loads it at
  coordinates (a shapeless workgroup array, a panic in the GPU lowering,
  now a named error); and coarsening chained reductions whose Temp outputs
  a later kernel reads, dropping their stores (a kernel binding nothing at
  a 51200 or 49152 vocabulary, which Vulkan's Compile indexed into; it now
  refuses one by name). And the host sampler's history slid by cutting its
  front, which made each append regrow it every window of tokens: four
  allocations over 64 sampled tokens, now none. Gates, laptop (RTX 3050 Ti,
  PTX and SPIR-V): TestComposedGPUSampleTopK holds 30 rows (vocabularies 37
  to 151936, histories 0 to 256, ties among the kept on 27) to the host
  sampler bit for bit, the penalty on the next id parting every row with a
  history; TestComposedGPUEqual and TestComposedCPUEqual hold MapEqual at
  ragged widths (words 2^25 apart, which a float merges);
  TestComposedCPURowsMatVec holds 120 row-selected matvecs (every packed
  type, float and int8, 1 to 64 ids, one past the matrix) to the oracle,
  moved weights and a selection one row off parting. V100 (cuda:0 and
  vulkan on cards 5 and 6, both builds): TestComposedHeadTokens 22 of 22
  (the composer selects every candidate set the device does; k+1 parts 22
  of 24 draws, the penalty on the next id 20 of 24);
  TestDecisionMatchesLlamaCpp's composed arm holds d1-3B, lev, openjev and
  vllm-d3 to llama.cpp with every prompt's labels the composer's (5 to 7 a
  request, none declined); TestDecisionOnEveryDevice's composed arm picks
  every prompt by the head over the picked rows (d1-3B 10, lev 14) within
  the device band (0.016 and 0.056), a pick one label off moves it past
  the band, and the gather arm answers within 2e-6 of it. Open: OpTopK and
  OpRank do not lower on the CPU device, top-1 included ("an index of x
  held in a vector register"), so the sampler's candidates form there on
  no request.
- A container that states no activations is refused where the composer
  serves. Written before the converter stated them, it opened and ran every
  block on the tier's own kernels with nothing said: the front end refuses
  its blocks (UncoveredAct) and the tier serves them itself. A benchmark's
  container of that age read as a 5.5% decode loss at every commit from
  120d0de8 on (V100 card 7, CUDA, Llama-3.2-1B Q4_K_M, tg128 after a
  512-token prompt, tuning off, 20 rounds interleaved: 0.9445 against
  cb858b79, the A/A 1.003); nvprof showed the tier's matvecs where
  cb858b79 ran the composer's. Open now returns an UpgradeRequiredError
  naming `jitllm upgrade` where the build composes by default
  (jit.ComposedByDefault, which the tier's default reads) or the CPU device
  composes, when the container states no activation and the front end
  refuses a block for wanting one (TestOpenRefusesAContainerThatStatesNoActivations,
  which fails with the check off). The same binary on that container and
  on its upgraded copy is a composed-against-main row: the composer decodes
  at 1.057 of main's kernels (201.6 against 190.3 tok/s, IQR/median 0.018,
  A/A 1.000, n=20), about 162 GB/s at 803 MB a token, 18% of the card's
  900 GB/s specification (no wall was measured in the pass).
- **The CPU device runs a prompt's matvecs a tile of the weight at a time,
  two rows a call.** In the rows' form (cpuRowsCap), Workers cuts a
  matvec held to a call's rows into tiles of at most 64 of the weight's
  rows, a calls' axis after the step's rows (Code.TileMajor), and the call
  site runs a tile for every live unit before the next (cpuCode.callAt), so
  a tile is read from memory once for the chunk. RowJam then runs two rows
  a call: the leading axis counts units of two rows, each row a copy of
  the values it reaches (its activation's loads, sums and stores) beside
  one decode of the weight's values. A unit's row past the call's rows
  reads and writes the form's capacity, never another row: a call whose
  rows are not whole units runs from a copy of the residual
  (cpuForms.xpad). A jammed kernel the registers refuse is formed again a
  row a call before anything is split or unfused (PressureError.Jammed,
  exprSplit.unjammed). Tiles alone were 0.207 of main on the laptop
  against 0.235 without them: without cursors (C5's Cursors skip for every
  kernel of two call axes, now kept only for (row, head) calls) the
  indexing cost what the reuse saved, and at four rows a call the fused
  kernel is 32.6 KB, forming took over ten minutes and a fused rotary
  epilogue was unfused into a map no call holds to rows (stories15M), so
  two it is. Forming the jammed form exposed two quadratic walks:
  Induction asked readsNode of every node against every load (65 s of a
  144 s placement on Llama-3.2-1B), now one cone walk (addressNodes), and
  Homes asked reachesFold afresh for every node of a fold's step (21 s),
  now memoised per fold; placement with the jammed form is 60 s on the
  busy laptop against 55 s for the unjammed form. And the byte dot takes
  a weight's codes as unsigned wherever the bits they may set leave every
  byte's top bit clear (wordBits: masks, shifts, ors and disjoint sums), so
  Q6_K, Q5_K and Q3_K's codes (a low field plus high bits shifted above it)
  lose their VPSIGNB and VPABSB: a Q6_K decode matvec's 32 signed dots go
  to none (4060 to 3740 bytes). Speed on the Xeon D-2123IT (4 cores,
  taskset 0-3, Llama-3.2-1B Q4_K_M, the same pass, A/A 0.999-1.003, n=20):
  composed prefill 0.1235 of main at pp33 (16.0 against 129.9 tok/s),
  0.1245 at pp100 (17.6 against 141.7), 0.1229 at pp512 (17.2 against
  139.8), where 7794821d reads 0.0821 (10.6) and 0.0837 (11.9): 1.50x;
  composed decode 0.1998 (6.8 tok/s, 5.4 GB/s, 13% of the 41.5 GB/s wall)
  against 7794821d's 0.1966 (6.6); a second pass reads 0.1239, 0.1254
  and 0.1228, decode 0.1988 against 0.1960: not regressed, and the
  unsigned dot does not show there. Laptop (i9-12900HK, 6 P-cores, one harness): pp100 125
  tok/s against 98 for the form with neither tiles nor jam. Gates: the jit package; TestComposedCPURowsServe,
  TestCPUComposesOnce, TestComposedCPURowsDoNotAllocate (a 33-row chunk,
  odd, runs from the copy) and TestComposedCPUBlocksServe on stories15M,
  Qwen3-0.6B and Llama-3.2-1B; TestComposedCPUMatVec's int8 arm fails with
  every dot taken unsigned. Open: the rows' form runs one session's single
  row a call as a unit of two (twice the dot work, one weight decode); a
  prompt still decodes the weight once per two rows where main's
  stationary GEMM decodes it once per call of up to its token tile, so
  prefill stays at an eighth of main's; and a part the registers refuse
  unfused can leave a map over (row, head) of no capacity that the rows'
  form cannot hold to rows ("its calls [64 6] hold no rows of a call").
- **Decode groups: a model over several cards decodes on every card at
  once.** A joint step runs card 1's blocks, then card 2's
  (tier.GPU.layersRows), so with one step in flight one card works. The
  server's step loop (server/batchgroups.go) now deals a multi-card model's
  decoding rows into one group per card holding blocks and steps each group
  on a long-lived worker of its own (model.StepJoint, the joint arm for a
  group of any size, one included); admission, the room check, prompts
  (GPU.pipeline) and a move stay the loop's, and any of them ends grouped
  decoding once the groups in flight land. The tier's per-call state became
  per call: a rows call's runs and error are its session's (tier.rowsCall),
  a refusal's cause is noted per session as well as per device
  (devSess.sessCause), and a card runs one step's stage at a time
  (devShared.stageMu) so groups that reach it together leave it a stage
  apart. The first version overlapped nothing: a step's head reset every
  device's count and cause and handed every device its inputs, each under
  that device's lock, which a rows step holds across its submission, so a
  second group waited at its head for the first's last stage (5.7 s of
  setRecSteps blocking in a 20 s trace). That bookkeeping now happens as the
  call reaches each device, and StageOverlaps (a stage beginning while
  another card runs another step's) went from 4 of 64 steps to 127 of 128
  on the laptop's two devices (RTX 3050 Ti under CUDA and Vulkan). Tokens:
  the loop groups by itself only where every block and the head run in the
  composer's reserved width (tier.GPU.RowsComposed), one kernel for every
  row count; there TestBatchDecodeGroupsKeepTheTokens holds stories15M,
  Llama-3.2-1B and Qwen3-MOE-4x0.6B bit for bit between one group, a group
  per card and a group per row. The tier's own kernels are chosen per row
  count (groupMV up to four rows, the tiles above, a split by the width), and
  forced groups there part from one group at tokens 1 to 9 (laptop and
  V100, dense, mixture and hybrid; Qwen3.5's linear blocks are not composed,
  so the composed build reports false for it too); there the loop keeps one
  group unless the caller forces a count (-decode-groups, or
  jitllm_decode_groups per request). Groups forced to step one after another
  overlap nothing (0 of 40), which the gate refuses. A warm grouped step
  allocates nothing (TestDecodeDoesNotAllocate's two-device-groups arm,
  Llama, olmoe and Qwen3.5). The V100 throughput rows (gpt-oss-120b on 5
  cards, Qwen3-30B-A3B on 2, Qwen3-Next-80B on 4) are not yet measured.
- **A linear block is a delta rule the composer forms.** The op language
  gains OpDeltaRule (value heads over key heads, grouped or tiled, a decay
  per head or per key channel by its operand's width, beta), and the front
  end states every LayerLinearAttn block from what the container stores
  (engine/model/frontdelta.go: the mixed projection, OpStateConv, SiLU, the
  unit-length q and k as an unweighted head norm, the decay from its
  logits, dt bias and log decay or the stated lower bound, the gated norm
  and the output gate the container states); the state-space and short
  convolution kinds stay UncoveredKind. Both state ops read a slot of a
  pool the step's recurrent descriptor names ("rec.rows": rows, the slot
  read, the slot written) and write ".out", so the CPU steps a sequence's
  state in place and the card reads one half of its pool and writes the
  other, as its own kernels do (bindRec, then stepFlip); Halves leaves a
  "rec." table at f32. OpStateConv lost its select over coordinates: the
  x86 lowering had no register for the constant a `less` compared against.
  The CPU device serves decode, and a prompt's and a batch's rows a row at a
  time on the one-row form, with main's per-row restore points; the card
  serves a decode step of one session and declines a prompt, a ragged step
  and the shared pool by name (DeclineRecurrent). Qwen3.5-0.8B Q4_K_M on
  the V100 (cuda:0 and vulkan:4): TestComposedBlocksServe every arm, 3360
  block steps served a 140-step arm and none declined, median 1.08 (CUDA)
  and 1.02 (Vulkan) of the tier's distance from the float reference. On
  the CPU device, its linear blocks composed and its attention blocks
  main's: decode 0.483 of main's int8 spread at the median (worst NMSE
  0.00268 against the control's 0.00534), a 700-token prompt over two
  chunks 0.472; dropping the decay, the gate, the key's norm or the
  convolution, binding another block's state, or dropping the state at a
  chunk boundary each leaves the band (median ratios 257 to 5272). Open:
  Qwen3.5's attention blocks on the composed CPU read 11.3 of main's spread
  (worst 0.071 against 0.00534) where the card's are in band, so the whole
  model is not in the CPU's served lists: it is the partial rotary, which
  GLM-4-9B-0414 (64 of 128, NORM pairs) shows at 695 of main's spread while
  gemma-2b (head 256, full rotary) reads 0.32; the card's prompt and ragged rows
  of a linear block; the one-row decomposition is a launch per row of a
  prompt where a chunked scan would be one.
- The orphan map of an unfused part was Prescale's: the map it builds to
  scale a contraction's weights (attention's probabilities by 1/sum) took
  the contraction's space and heads and none of its rows' hold, so Serial
  made its calls (rows, heads) with no capacity. It now takes the
  contraction's Capacity, RowsHeld and calls. TestCPURowsFormsHoldTheCallsRows
  forms a stories15M-shaped block at 2 and 64 rows, Q8 and Q4, fused and
  with every Fuse off (a refused part's form): every kernel holds the
  call's rows; with the fix reverted the unfused form fails exactly as the
  batch did ("map ... its calls [2 6] hold no rows of a call").
  TestComposedCPUAttendRows gained the unfused arm, which executes the
  separate normalising and prescaling maps; it exposed two harness bugs,
  not engine ones: runCPUCalls shared one constant block between two
  kernels whose blocks share a name (cpuserve keeps them apart), and read
  the output by name where the in-place pass leaves it in a scratch. With
  both fixed every arm holds at 1e-8 against the definition. The Cursors
  skip for (row, head) calls is still needed: with it removed only the
  fused softmax kernel (max, exp, sum and the normalising map over the
  scores scratch, cursors on that scratch alone) fails, and only at
  scores far below zero, where the softmax's shift stops being harmless:
  its max is taken over the wrong positions. OpRank/OpTopK of K > 1 on the
  CPU fail to lower because a jammed rank kernel's base index is both a
  load's address and the vector of each lane's index; a broadcast of it
  from a general register lowers it but the binder then gives that word
  and a broadcast float one vector register (ranks wrong on ties). Greedy
  top-1 (graph.Sample) forms on the CPU at every width tried.
- The CPU's ranks and top k form and hold (TestComposedCPURank,
  TestComposedCPUTopK, TestComposedCPUSampleTopKRanked over every sampler
  case of a vocabulary the ranked form takes, bit for bit with the host's,
  TestComposedCPUSampleGreedy at ragged widths to 1531). Three bugs stood
  in the way, each by structure: Jam jammed a loop whose iteration a value
  reads (the rank's j < i), so one node was both an address and each
  lane's index and the vector words took it from the general registers
  (Jam now declines such a loop; jammed by force the ranks part on ties);
  a fold whose step another value of the trip reads took the step's
  register before its own later readers ran (the top k's place counter,
  read a trip ahead: every place zero; FoldsAside copies such a fold aside
  where the step is made); and a word a vector loop loads was held in a
  general register, so a vectorized copy of words moved one lane in eight
  (the sampler's history slice; a word a vector loop loads is a vector
  word now). Each violation fails its gate. Still open: the selection form
  (vocabularies over 1024) does not form on the CPU -- a map reads more
  inputs than the argument block's In fields, and past that its integer
  compares feed vector selects from general registers ("node (const) has
  no register"), and making them vectors runs out of registers.
- The CPU's partial rotary left each head's unturned tail out of the
  output. decomposeRope forms two maps over one output, the pairs and the
  tail's copy; InPlace put the tail copy's output over its input (the input
  is read by nothing after it), so the copy wrote the input onto itself and
  the output's tail kept whatever its storage last held, while the pair map
  went on writing the output's own storage. It is the likely cause of GLM-4's and Qwen3.5's
  out-of-band CPU spread (695x and 11.3x main's; not yet rerun on those
  checkpoints); a full rotary is one map
  and was right. InPlace now leaves an output an earlier kernel writes in
  its own storage. TestComposedCPURope forms full and partial rotaries (128
  of 64, 256 of 64, 64 of 32, 80 of 40, 128 of 128), both layouts, at one
  row and at five of eight, each turned element within 1e-5 of the
  definition and every tail element equal to the input's bit for bit; the
  fix reverted fails it, and the tail moved by one element parts.
- **A linear block's rows are a scan the composer forms.** The op language
  gains no op: OpDeltaRule and OpStateConv over a step of several rows
  decompose into a new primitive, PrimScan (jit/generate_scan.go), and two
  maps. A scan's state rows are the value heads' rows, a workgroup of one
  subgroup each, every thread holding components lane, lane+32, ... as
  folds; it loops over the runs the step's run descriptor lists
  (kernels.RunDescWords' layout) and over each run's rows in order, the
  two dots group sums, the state read at the run's slot and written after
  its last row. The convolution reads each row's window from its run's
  state and the run's rows before it, and writes each run's last Taps-1.
  One form of a width serves every row count (the counts are the
  descriptor's), a prompt chunk as one run and a ragged step's sequences a
  run each, so the tier stages the run descriptor for a chunk too (its
  first three words are the chunk's, which its own kernels read). The
  card serves a linear block's prompt and ragged rows (DeclineRecurrent
  now names only a recurrent tail and the centred, K3 and n-gram
  variants), and the shared pool in place, through a descriptor of its
  own whose slot written is the slot read. A call the composer serves
  whole stages its paged descriptors without building the decode plan's
  kernels: on Qwen3.5 a 511-token prompt compiled two of them
  (flashkvpaged, flashmergewide) after the placement. Gates: the scan and
  the convolution on CUDA and Vulkan over chunks of 8, 5 and 1 rows and
  ragged steps of interleaved runs, NaN padding rows, held to the oracle
  (2e-5, 1e-5) over five delta shapes (tiled, per-channel decay, Dk 7 to
  128) and five convolutions, every slot outside the step's untouched;
  each row made its own run parts 40 times. Qwen3.5-0.8B Q4_K_M on the
  V100: TestComposedBlocksServe with its prompt composed, every arm on
  cuda:0 and vulkan:3, 24 block steps of the prompt served, median 0.996
  of the tier's distance from the float reference; the composed chunk's
  rows made runs of one (the "chain" fault) move the prompt's logits
  1.28 in NMSE. TestPlacementComposesOnce gains Qwen3.5: 19 forms at
  placement, prompts of 1 to 2000 tokens (264 prompt block steps), a
  second session, a ragged batch and a relocation compose nothing after
  the placement, one card and two. The shared pool decodes a composed
  prompt and 14 steps bit for bit as the doubled one (18 shared states,
  cuda:0 and vulkan:3). Kimi Delta Attention serves end to end on
  Kimi-Linear's fixture converted to Q8_0 (synth-kimilinear@q8, the
  transformers weights every parameter and buffer random, convert.WithQ8):
  every arm on both backends, median 0.96 (CUDA) and 1.00 (Vulkan) of the
  tier's distance; its rows told other positions is no violation there,
  the latent attention applying none (FlagNoPosEnc). Qwen3.5's mixture
  fixture at Q8_0 serves in band (1.17, 1.35) but cannot discriminate (its
  tiers sit 0.28 to 0.34 from the float reference, one position off reads
  2.61 against the arm's 0.34): formedModels. In the composed build,
  TestBatchSeamMovesCarryEveryRowHybrid, TestBlocksMoveBetweenDevices,
  TestLayerRelocatesToAnotherDevice, TestBlocksMoveToTheCPUAndBack and
  TestKVPagesRelocateWithTheLayer pass on Qwen3.5-0.8B and both Q8_0
  hybrid fixtures (the principle lists now carry them),
  TestPagingDoesNotChangeTheAnswer on Qwen3.5-0.8B, and the mixtures'
  paging gate on both fixtures (343 and 167 page-ins under one page, every
  logit the resident run's); and the group sum a scan makes in a loop waits for every
  thread's read before the next row writes its slots (lowergpu).
- **The CPU serves its head and its samplers composed, and the selection
  form forms.** Where no card holds the head, the State's logits come from
  the head's graph composed for the CPU (graph.Head: the norm, the
  projection and its bias, softcap and scale, the norm's rows a second
  output where a step reads them back), a greedy token from graph.Sample
  and a top-k draw's candidates from graph.SampleTopK, each composed at
  the first step that asks for it and counted served or declined by name
  (CPUHeadSteps, CPUSampleSteps); a graph apart from the residual binds
  each input and output to a place of its own (cpuArgIO). The selection
  form (vocabularies past 1024) took five fixes, each by structure: the
  argument block's In fields reach five more words (48, 72, 144, 152,
  160); an integer compare whose mask a general register holds has a
  general-register form (CMP and a mask), as does an and-not; an index one
  lane of a vector register holds in a scalar loop is moved to the reload
  register for its address; a compare a vector word reads is a vector
  compare, and an index, a length or an argument word a vector reads is
  read through a broadcast (Splats); x86's integer compare is signed, so
  a compare of words that may pass 2^31 compares them with their top bits
  flipped (Unsigned); and the binder's lifetimes follow where the sequence
  places a node, not its home -- a reader of a loop's counter (Induction)
  sat in the loop while its home said outside, and the threshold it read
  was overwritten each trip (the last round's counts came out 1531 of
  1531). TestComposedCPUSampleTopK holds every SampleCases vocabulary (37
  to 151936) bit for bit, the selection form on 20 of 30. The Cursors skip
  for (row, head) calls is gone: the fused softmax's vector loop preset its
  tail's counter from a register the sequence had not yet written (the
  tail ran the whole count, its maximum past the row's range); the preset
  now waits for the count to be made. A kernel the registers refuse is
  formed again a head a call (HeadPairs), with no rows across lanes (Jam),
  then indexing its buffers (Cursors), before anything is split or
  unfused: stories260K's pair with no twin and Qwen3-MOE's expert matvecs
  (more words than general registers) now form on the CPU. Speed on the
  Xeon D-2123IT (taskset 0-3, Llama-3.2-1B Q4_K_M, n=20, two passes, A/A
  0.999-1.004): composed prefill 0.1220 and 0.1227 of main at pp33 (15.9
  against 129.8 tok/s), 0.1247 and 0.1244 at pp100 (17.7 against 141.9),
  0.1211 and 0.1200 at pp512 (16.9 against 139.8), within a percent of
  the form before; composed decode 0.1775 and 0.1773 (6.0 tok/s, 4.8 GB/s,
  12% of the 41.4 GB/s wall; main 33.5 tok/s, 65%) where it read 0.1998
  with main's head: the head's 128256-row projection now runs composed, at
  the composed matvec's rate. The head and its greedy sampler add about 5 s
  (Qwen3-0.6B) and 8 s (Llama-3.2-1B) to the first token on the laptop,
  composed at the first step that asks; placement is unchanged. Gates:
  the jit package; TestComposedCPUBlocksServe (head violation leaves the
  band at a median ratio of 578-6510), TestComposedCPURowsServe,
  TestBatchMatchesForward, TestCPUComposesOnce, TestComposedCPURowsDoNotAllocate
  and TestPagingDoesNotChangeTheAnswer's composed-CPU arms on the laptop;
  TestFrontStatesTheModelAsOps' CPU arm forms stories260K and Qwen3-MOE.
  Open: a batch's head serves a row a call; specrows, altup and the label
  pick keep main's head.
- **A long reduction is not written out into its map.** The unrolled
  fusion (Fuse's Reduce-to-Map pair) bounded a reduction's trips only where
  its reader was a reduction too, so a map reading a projection took the
  whole row's dot written out: gemma-2-2b's one-row head and its softcap's
  first scale formed one kernel of 8178 IR ops with the int8 form and 55341
  ops (1.67 MB of SPIR-V) in the float reference. A V100's Vulkan driver
  spent 50 minutes on the latter and refused it (VkResult -13); the
  laptop's RTX 3050 Ti built its 13.9k-op variants in about 5 s each. The
  fusion now writes out at most 4096 nodes (trips times the reduction's
  nodes): the head forms a 133-op reduction with the int8 form and 94 ops
  without, its scale a map of its own (one more launch a token on a
  softcapped head), and a split matvec's 64-part combine still fuses into
  its map (block 0 of Llama-3.2-1B, Qwen3-0.6B, gemma-2-2b, SmolLM2-360M
  and granite-3.3-2b forms byte-identical kernels before and after).
  Gate: TestALongReductionIsNotWrittenOutInItsMap (a 2304-column, 256000-row
  projection and a map, Q6, Q8 and f32, with and without the int8 form),
  which fails at 8178, 2264, 55341 and 34593 ops with the old bound.
- **Clamped, normed, tempered and split attention are ops.** The front end
  states dbrx's clamped q, k and v (OpClamp), Llama 4's weightless q/k norm
  after the rotary and Gemma 4's over v (OpHeadNorm with no weight), v read
  from k's projection, the attention temperature 1 + s ln(1 + floor((pos +
  o)/f)) as a row's history position (OpPosition: the rows table's write
  word, plus o, in whole f in integers) through a logarithm the composer
  decomposes (MapLog: exponent and mantissa, ln m as the atanh series to
  s^11; 0.62 ulp worst on the CPU over 80 inputs, -Inf, +Inf and NaN at
  the edges), Llama 4's experts weighted at their input (gate and up times
  the routed weight, the down summed at one), and XD-RoPE as a split
  rotary (graph.RopeSplit): a table of a pair an element, element i+P
  turned by pair i+P, the sections doubled, the planes the model folds laid
  twice, which is main's A-then-B table row. UncoveredQKNormPlace, VNorm,
  VFromK and Clamp are gone; Temp names only an offset that is not a whole
  number, RopeAxes only sections that do not match the runs. The x86
  register binder gave constants read outside every loop a register each
  past the sixteen; it now stops at the file. On the V100 (cards 5, 6):
  TestC6ServesComposed over dbrx, mistral3 and mistral3-af on cuda:0 and
  vulkan:5, every block and the head served, 2e-12 to 6e-12 from the host,
  no clip_qkv 1.70, the temperature off 0.165; TestComposedBlocksServe on
  Ministral-3-3B (median 0.95 / 1.03 of the tier's distance), Llama 4's
  q8 fixture, Pixtral's, HunyuanVL's and HunyuanOCR-Q8_0 (median 1.04),
  every arm on both backends; TestSynthLlama4ServesComposed 4.5e-12 / 5.0e-12
  from the host, each feature out moving it 2.4e-5 (the temperature offset
  a floor further) to 2.6 (a window for the chunk); after HunyuanVL's
  picture the composed table and blocks read 1.1e-7 (CUDA) and 4.3e-13
  (Vulkan) from transformers, the sections moved 2.65e-4. The CPU serves
  Ministral-3-3B in main's band (0.987 of its spread at the median) and
  declines Llama 4's and dbrx's fixtures by name (routed experts not held
  whole, a windowed layer's history). Llama 4's q8 fixture fails the host's
  node-affine allocation arm on the two-socket box (no region ran
  node-affine), a main-path gap left out of the principle lists.
- **Decode groups under load: a refused group waits for the others, the
  loop's own count is measured, and a reply cut inside a character
  marshals.** TestDecodeGroupsHoldUnderLoad serves sixteen requests at
  once over two devices, prompts of 3072 and 384 tokens alternating, in
  waves admitted together (tokens held to one group's, a parting allowed
  only at a host tie under the batch suite's 0.5) and as a closed loop
  (answers only: under one group alone, two runs of the loop part on 6 of
  16 stories15M requests, since when a request arrives moves its prompt's
  step company). On the V100s (Qwen3-30B-A3B Q4_K_M, cards 0 and 7) the
  head before this ran no grouped step at all under that load and took
  12m60s against one group's 6m09s, with 2504 KV grows and 231 evictions
  against 785 and 31: groupCount asks GPU.RowsComposed every iteration,
  which walks every block's composed prompt instance. Its one-group arm
  first failed on "string field contains invalid UTF-8": a completion
  cut at its token limit ends inside a character, which the protobuf
  wire refuses (validText now writes U+FFFD, as the JSON encoder does).
  A group whose step the device refused ran its rows alone -- the
  one-at-a-time path, which is no rows stage -- beside the other groups'
  stages; it now lands once no group steps. The ab-30b-head A/B's ITL
  loss at 2 and 4 requests (3.71x, 1.94x) is the arm, not the overlap:
  one group runs those row counts as the one-at-a-time path (24 and 49 ms,
  about c x 11.3), while every group's step is the joint arm, lone rows
  included, which costs about four Forwards at one row on this mixture.
  The loop's own count is now a measured choice against one group
  (groupChoice, the joint choice's ABBA probe, one group the incumbent).
  Open: on stories15M, 9 of 32 wave requests part from one group at host
  ties (0.003-0.19): a step's attention plan follows its rows' longest
  history (pagedKeys), so the groups are not bit-identical to one group
  once prompts differ in length; the illegal memory access of the A/B
  (-sessions and -max-batch 128) was not reproduced by the gate.
- **The CPU's head is composed at its placement and serves every step's
  rows; a head's stages after its projection land in the caller's logits;
  a windowed layer's history is served; a vector loop's tail count stays
  a word.** Every head graph the CPU runs -- the one-row head, its hidden
  form, the head over 64 rows, the label pick and both samplers -- is
  composed when the State is placed, and no step forms one
  (TestCPUComposesOnce counts the model's CPU forms across every phase).
  A batch's decoding rows, a speculation step's verified rows, the rows a
  card left to the CPU's head, a model of several streams (its head over
  the collapsed row) and AltUp's chunk run the rows' form in one call,
  counted in CPUHeadSteps. The forms cost 6-25 ms each on Qwen3-0.6B
  (laptop) but the top-k sampler's, 1.27 s, now 0.78 s: the orderer read
  what is made from a map rebuilt over every node at each need. On
  Qwen3-0.6B (laptop) the first State's placement goes from 0.35 s to
  1.09 s, and its first sampled token from 1.41 s to 0.11 s; a later
  State reuses the model's forms. The head
  gate had read "the last kernel's written operand": InPlace stored a
  head's scale or bias over the projection's buffer, so a composed head
  with a stage after its projection left the caller's logits unwritten
  (zeros on commandr, cohere2 and phi2; the hidden rows the input's).
  A graph apart from the residual (Request.Apart) now keeps each output in
  its own storage and takes no input's; the gate reads the graph's output
  and fails on the old pass. Windowed attention reads the CPU's own rows
  descriptor (its start the layer's window) and a sliding layer its own
  rotary table: cohere2's and EXAONE 4's window of four serve every block
  step, 3.5e-11 and 1.1e-8 from main over 24 steps and a prompt arm, the
  window widened to 8 or removed moving them 1.3-2.4, cohere2's logit scale
  removed 9, phi2's output bias doubled 0.037 (clean 7.3e-13). C4d's
  segfault with a difference flipped by Unsigned was Splats giving a
  vector loop's tail count (sub & 7) a broadcast of the difference, so the
  tail ran a vector register's bits as its trip count: a loop's count is
  now in the cone Splats keeps in general registers. With the difference
  flipped the attention no longer faults but reads NMSE 0.2 from the
  definition, a second defect not found; the rule that a difference of two
  small words is small stays.
- **The state-space mixers are the delta rule without its key dot.** The
  op language gains one attribute, no op: OpDeltaRule's Additive drops the
  dot of the state against k, so the row's update is the decayed state
  plus (v times beta) along k -- a selective state update, beta its dt. The
  front end states a block by its mixer's structure (engine/model/
  frontssm.go): the selective update with one decay a head (the mixed
  projection's C, B and x through the state convolution with its stored
  bias and SiLU, dt the softplus of its projection plus the stored bias,
  the decay exp(A dt), D times x added, y times SiLU of the gate, the
  grouped norm before or after the gate as an unweighted head norm over
  each group times the stored weight), the same update with one decay a
  state element of each channel (dt's bottleneck, B and C off the
  convolved x, each normed where stored, the decay exp(dt A) as a map whose
  narrower dt broadcasts over the state), and the gated short convolution
  (B times x through the state convolution, times C). A block that attends
  as well projects its mixer beside the attention's output, and a block
  with no MLP ends at its first residual. The scan and the one-row form
  serve them unchanged on CUDA, Vulkan and the CPU device. Five things
  stood in the way, each fixed by structure: a key head of one was a
  division by the value heads (OpDiv by 1536 has no x86 general-register
  form here; one key head is now the constant 0); a map's bound table of
  several rows (A stored [16, 1536], the grouped norm [64, 2]) was refused
  as a matrix and is read as its elements in order; the prologue pass laid
  a consumer's workgroup array out by the producer's shape where the
  consumer reads it at another (a head norm of a 2-D gate, read at 3
  coordinates); the tier read "binds" as "reads and writes", so a split's
  part kernel that binds the output and writes only its planes, and an
  epilogue that reads x only at the element it writes, were refused in
  place (jit.Binding.AtElement; inPlace uses Read, Writes and it); and the
  converter stated a mixture with ungated expert banks as gated
  (convert/acts.go reads the banks). The CPU device steps a block that
  attends as well through its pages (cpuRecRows readies each row's
  history, as cpuStep does). A block whose graph binds a weight or vector
  the card does not hold is declined by name (DeclineNoWeight) rather than
  failing to form. On the V100, cuda:0 and vulkan:3: TestComposedBlocksServe
  serves LFM2-350M, granite-4.0-h-350m (Q8_0), Falcon-H1-0.5B and
  Falcon-H1-1.5B (Q8_0) every arm, prompts composed, the poisoned arm
  unchanged; with a chunk's rows made runs of one the prompt's logits part
  by 0.17 (LFM2), 0.49 (granite) and 0.24 (Falcon-H1) in NMSE. The CPU
  device holds mamba-130m, LFM2-350M and granite-4.0-h-350m to main at
  0.43, 0.90 and 0.47 of main's own spread, every recurrent block bound to
  the first's state parting 4810, 472 and 4937 times. Open: on the cards
  the pure state-space checkpoints part from the float reference where
  their random fixtures do not (mamba-130m 3.3 times the tier's distance,
  falcon-mamba-7B 80, Jamba-reasoning-3B 50 with a poisoned buffer read,
  mamba-codestral non-finite at its 103rd step), while the CPU serves
  mamba-130m in band: a card form of these widths (B and C 16 rows, dt's
  rank 160 or 48) is the suspect. LFM2-1.2B reads 1.503 against the 1.5
  band on CUDA (0.974 on Vulkan).
- **A kernel that reads x where it writes the output is not written over
  x.** The state-space blocks with no MLP (mamba-130m, falcon-mamba-7B,
  mamba-codestral, Nemotron-H's mixer-only blocks) end in one kernel that
  adds the projection to x and writes the output, and the tier binds the
  output over x. The tier had taken such a form in place where the kernel
  reads x only at the elements it writes (the AtElement binding); that is a
  RULE 13 read of a buffer written in the same phase: a launch clamps its
  surplus threads to the last element, and every lane storing an element
  loads it, some after another lane's store, so the residual is added
  twice. At the fixtures' widths the projection splits into parts and a
  combine (x read before the first write) and the gates stayed green; at
  mamba-130m's (768 over 1536) the form is one reduce+reduce.group+map of
  24 groups that reads x and writes the output, which is what parted
  mamba-130m from its float reference by 3.3 times the tier's distance and
  moved it under poisoned buffers on the V100. The rule is back to x read
  before the first kernel writing the output; a block whose form is not is
  declined by name (DeclineOverInput) at decode and at a prompt's rows, its
  tier kernels readied, and the AtElement binding is gone. The delta rule's
  scan, the window convolution and the scan mixer's float matvecs were
  held to the oracle at the real widths first and are right there (1536 and
  5120 heads of one 16-wide state row, 8 key heads of 128, 1536 and 6144
  channels of 4 taps, 16 rows of 1536 and 5120, 1536 rows of 48, 5120 of
  160, worst 9e-6). Falcon-H1-0.5B's CPU control was not NaN: token 1, the
  gate's first token, is >>ABSTRACT<<, whose embedding row is zeros, so
  every arm's logits are zero and every distance 0/0; the gate now starts
  from the model's BOS, and the CPU serves Falcon-H1-0.5B at 1.019 of
  main's spread (mamba-130m 0.466), each violation leaving the band; from
  its BOS granite-4.0-h-350m's composed CPU parts from main entirely (1641
  times main's spread, argmax 0 of 80, where from token 1 it read 0.47):
  open, and the gate stays red on it. Gates:
  TestAKernelReadingXWhereItWritesIsNotInPlace (9 forms reading x in the
  writer refused, 3 reading it before written over it; with the rule at
  lastRead <= firstWrite it fails), TestBindingsSayWhatACodeReadsAndWrites,
  TestComposedGPURecurrentRuns and TestComposedGPUScanWidths (PTX and
  SPIR-V), and on the laptop's sm_86 TestComposedBlocksDeclineOverInput
  (mamba-130m: 576 of 576 block steps declined over the input, the logits
  finite and unchanged under poisoned buffers), TestDecodeDoesNotAllocate
  on synth-mamba1, -mamba2 and -nemotronh (whose ragged steps had been
  refused for the in-place read on sm_86, and on the V100 had formatted a
  cached form failure every step, 335 allocations a step on Falcon-H1-0.5B:
  servedFor now states that error once), TestComposedBlocksServe
  on LFM2-350M (median 1.205) and TestComposedCPUBlocksServe. Falcon-H1-0.5B,
  served on the V100, reads 0.248 at its worst on sm_86 against a bound of
  0.156 (median 1.08), held to the one backend there. Open: serving those
  blocks needs an output apart from x and a copy back (one more launch
  each); Jamba-reasoning-3B (every block has an MLP, so not this) and
  LFM2-1.2B's 1.503 on the V100's CUDA (its arm sits 0.142-0.146 from the
  float reference at its worst on every backend; the V100's CUDA tier 0.091,
  sm_86's 0.052, Vulkan's 0.109: the arm's int8 step against a tighter CUDA
  tier; on sm_86 it serves every arm at 1.405 and 0.974).
- **A prologue's budget counts the group sums' arrays.** Falcon-H1's scan
  at the reserved width (512 rows) ran its step map inside its workgroups
  (Prologue) with a workgroup array of 12288 words, exactly the 12<<10
  budget, while the lowering gives each group sum, maximum and exchange an
  array of the workgroup's width beside it: 49280 bytes, past the 48 KiB
  of static workgroup memory. sm_70's compiler refused the form (the V100's
  "block 10's prompt is not the composer's at 512 rows", its prompts formed
  again at their own widths, and the ragged step's cached failure formatted
  every step) and sm_86 refused it at launch (cuLaunchKernel invalid
  argument on every prompt and ragged step). The budget now counts those
  arrays. Gate: TestPromptFormsFitTheWorkgroupsMemory (every block of each
  served state-space checkpoint at 1, 64 and 512 rows, with and without
  subgroups; Falcon-H1-0.5B's widest kernel 40192 bytes, LFM2-350M's 36480;
  without the change Falcon-H1-0.5B's block 0 declares 49280 and fails). On
  the laptop's sm_86 Falcon-H1-0.5B now passes TestPlacementComposesOnce
  (one card) and TestDecodeDoesNotAllocate on every cuda arm; its serving
  gate there reads a median 1.08 and 0.248 at its worst against 4 times
  the one backend's 0.039 (the V100 holds it to two backends' widest).
- **Every state-space block served, none over its input.** Four causes,
  each a gate. (1) A form whose last kernel reads x where it writes its
  output (a block with no MLP: Mamba, Nemotron-H's mixer layers) is now
  served apart rather than declined: the cards write its output to the
  scratch's h and a copy the composer forms (graph.Copy, the part kind
  partCopy, counted as Stats.ServedPart(PartCopy)) puts it over x.
  TestComposedBlocksServeApart (TestComposedBlocksDeclineOverInput as it
  was) serves every block step of mamba-130m (576 of 576 written apart),
  Nemotron-Nano-9B-v2 (744, 144 apart), mamba-codestral (1536 of 1536) and
  synth-nemotronhmoe-q8 (96, 24 apart), unchanged under poisoned buffers.
  (2) A model whose blocks never attend (plain Mamba) states one
  placeholder head as wide as the embedding; the tier built FlashDecodeKV
  for it at 4096 wide, which sm_70 refused to JIT (CUresult 218), and
  falcon-mamba-7B and mamba-codestral placed 0 of 64 blocks. The plan
  carries NoAttention and the paged variant is built empty. (3) Parts let a
  part whose threads combine (a workgroup sum or max, a lane shuffle: the
  int8 forms' amax) join a wider launch, its surplus threads clamped to its
  last element combining among themselves and storing another scale there
  (RULE 13). Mamba-1's 1536-channel int8 forms beside a 4608-wide window
  copy and a 24576-wide state copy made two identical decodes part on the
  V100 (mamba-130m from step 3, worst logit 10.5 on CUDA and 22.2 on
  Vulkan; Jamba from step 2; bit-identical with the tier's kernels and with
  float activations): with the combine counted both decode bit-identically
  on both backends, poisoned or not (TestPartsJoinNoCombineAcrossWidths).
  (4) The CPU's InPlace let a head's output take the projection's own
  buffer, so a head with a logit scale or a final softcap never wrote the
  caller's logits: granite-4.0-h-350m's composed CPU read zero logits
  (NMSE 1 from its BOS; G3b2's zero-embedding reading was not the cause). A
  graph's output now takes only storage its caller holds
  (TestCPUHeadStoresItsOutputs); granite-4.0-h-350m serves on the CPU at a
  median 0.510 of main's spread. Open: the CPU head's hidden output
  (headHidden, a decision model's row taps) is folded into the projection
  and stored by no kernel. synth-deepseek-yarn's router violations were
  vacuous: its three blocks are all dense leads, no router runs on any
  arm, and routerViolations now asks for a routed block. On the V100
  granite-4.0-h-1b (CUDA median 1.066, Vulkan 0.940, its Vulkan head
  served) and mamba-130m (1.048 and 0.985; the tier itself sits 98.9 from
  the float reference at its worst on CUDA) pass every serving arm and are
  served, and so do Jamba-reasoning-3B (1.254 and 1.093, no longer moved
  by poison) and Nemotron-Nano-9B-v2 (1.431 and 1.040). Open:
  LFM2-1.2B still reads 1.503 on CUDA (Vulkan 0.974); falcon-mamba-7B
  reads 0.029 at its worst on CUDA against 4 times the tier's 0.00291;
  mamba-codestral's composed logits and its float reference are not finite
  at step 13 and every sixth step after (the tier's are finite); and
  synth-nemotronhmoe-q8 serves but its gate cannot discriminate
  (one position off 2.6, ten times the arm's worst 0.574 wanted), and stays
  in the over-input gate alone.
- **Zero embeddings, violations that reached main alone, and no decline over
  the input.** mamba-codestral's "non-finite" logits at step 13 and every
  sixth after were exactly zero in every arm (tier, float reference and
  composed): the serving gate's ids 19, 25, 31, ... are among the 277 rows
  of its Q4_K_M embedding that store zeros, and silu of a zero gate zeroes
  every Mamba-2 block's output, so the distance read 0/0. The gate's ids now
  take the next id with a stored row (19 of 140 moved); codestral serves at
  a median 1.048 of the tier's distance on CUDA and 0.961 on Vulkan.
  synth-nemotronhmoe-q8's CUDA arm sits 0.574 from CUDA's tier at a router
  near-tie, so one position off (2.6) cannot be ten times it; the gate holds
  one position off to the band for it (bandSeen) and it leaves (a median
  1850 to 2220 of the tier's distance); served at 1.014 and 1.000.
  falcon-mamba-7B's Vulkan excursion (0.029 at one step against 4 x
  0.00291) is a sensitive step, not a block: composing the first 48, the
  last 32, the first 32, the first 16 and all 64 blocks moves the 132nd
  step on CUDA to 0.0236, 0.0184, 0.0015, 0.0006 and 0.0007 (the tier
  0.00066); it serves apart and stays out of the served list. LFM2-1.2B's
  1.503 on CUDA is no block's either: the composed arm sits 0.142/0.143 from
  the float reference on both backends, CUDA's tier closer (0.091 against
  Vulkan's 0.109), and composing 4, 8, 12 and 16 leading blocks reads 1.282,
  1.434, 1.505 and 1.503; shadowed on the tier's own input every composed
  block sits at most 2.9e-04 from the tier where the tier sits 5.7e-04 to
  2.6e-03 from the float reference, alike on both backends: amplification
  of a rounding a tenth of the tier's own, still past the band (open).
  TestSSMFamiliesOnEveryDevice's three red fixtures were gates
  that reached main alone: synth-lfm2moe's selection bias and router score
  and Mamba-1's dt/B/C norms were taken out of the model's layers and
  config, which the front end does not read; with the container restated
  every violation is seen (0.556 to 2.55). DeclineOverInput is gone: no
  container states a block wider than a row (every way to Covered passes
  the front end's Streams check, TestFrontRefusesSeveralStreams), and the
  shadow arm runs a form written apart into bs.h and its copy over its
  input (mamba-130m shadowed, none declined). The CPU head's hidden output
  is stored where the State binds it (Request.Apart, at 1 and 64 rows);
  formed without Apart the norm's rows go over x.
- **A pooled State gives its history back; RowsComposed walks once a
  placement; a joint step's rows are their own, the one-at-a-time path is
  not.** A server's idle pooled State kept a finished request's pages on the
  card (Reset rewinds the position only), and admission, which prices by the
  devices' free room, then admitted a row at a time from a load's second
  round: Qwen3-30B-A3B over two V100s (main's path, 64 clients, 512-token
  prompts, 128 out) stepped 104.8 rows a step in the first round and 5.0 in
  the next, 8189 waits for history room, KV held 8592 MiB with no request
  running (TestBatchKeepsBatchingRoundAfterRound). The pool now trims a
  reset State's device history (State.GiveBackHistory). GPU.RowsComposed,
  which the step loop asks every iteration, keeps its walk until a device's
  placement generation moves (Stats.RowsComposedWalks: 0 walks in the
  grouped arms on the laptop, 9 in 5 rounds with the cache off). Decode
  groups stay off: TestStepRowsAreTheirOwn holds a joint step's rows bit for
  bit whatever shares the step (prompts beside prompts, beside a 900-token
  history, chunks of 20, 7 and 1; stories15M and Qwen3-0.6B on the laptop,
  Qwen3-30B-A3B on two V100s), so pagedKeys is not what moved the load
  gate's tokens on the composed path; the one-at-a-time path is: Prefill's
  and Forward's logits sit 0.46-0.87 of a logit from the joint arm's on
  Qwen3-30B-A3B (0.23-0.30 on Qwen3-0.6B), and the loop puts a row there
  whenever it steps alone. The load gate now runs one group twice with no
  tie margin: on stories15M one group parts from itself in 8 of 32 waves,
  one at token 0, so groups cannot be held to it until every path a row
  takes is the joint arm's arithmetic -- and the joint arm at one row costs
  88 ms a step on Qwen3-30B-A3B where Forward costs 12. The composed path's
  long-prompt room (gpt-oss-20b on one V100, TestLongPromptHasHistoryRoom):
  two 2051-token prompts at once are refused where main's path serves them,
  budget used 14781 MiB against 12255 at a reserved width of 64 rows against
  512 (Stats.ComposedBytes: 2415 MiB of composed instances, with the
  reserved width already narrowed to 64 rows), and the pool 176 MiB where
  main's grows to 468, and a prompt mid-prefill has no written page to
  evict, so two at once find none. At 128 rows and 129 sessions (stories15M,
  laptop) every request answers in every arm, and groups part from one group
  in 23 of 128 waves where one group parts from itself in 25. On
  Qwen3-30B-A3B over two V100s (16 requests, prompts of 3072 and 384, 64
  out) groups part in 13 of 32 waves and one group from itself in 13 of 32,
  the same requests, ten of them at the same token and host gap (request 10
  in the "23341 2529" loop request 24 fell into before): the parting MP1b
  read as a groups fault is the schedule's path switch.
- **Gemma 4's two geometries and its picture run, composed.** The front end
  no longer refuses a layer of its own head width or kv head count (each
  block's graph is stated at HeadDimAt/NKVHeadAt already) nor a model whose
  image rows see each other on the sliding layers (Config.BidirSWA): a run
  is each row's key count in the descriptors the composed attention reads,
  not a flag, and the stored layer scalar is an op over its bound vector
  (RoleLayerOutScale). UncoveredGeometry is left to a value head of another
  width than the key's; UncoveredNonCausal to whole non-causal models. On
  the CPU device the runs follow the layer (State.runsAt: the sliding layers
  alone under BidirSWA), and a run longer than a call of 64 rows is no
  longer declined: a first pass over a copy of the rows writes every row's
  keys, its outputs dropped, and the second reads them (synth-gemma4v's
  266-row picture: 48 block steps served, 7e-08 from main; causal
  everywhere 1.39, bidirectional on every layer 0.196). On the cards the
  first run of synth-gemma4 and -kv read 1.5 from the host while the CPU
  sat at 4e-10 and -wide at 1e-10: the prologue pass ran the sliding
  layers' v append (after their weightless v norm) inside the attention's
  workgroups and kept its output, the history's value pool, in workgroup
  memory, so every earlier position read what that memory happened to
  hold (no v norm: 1e-08). Prologue now leaves a producer that stores a
  table where it is: served at 4.8e-12 (CUDA) and 4.7e-12 (Vulkan) from the
  host, -kv 6.5e-12 and 7.0e-12, -wide 1.5e-10 and 4.7e-11, and the picture
  prompt on the cards at 8.3e-11 and 8.1e-11 (its violations 1.40 and
  0.197), every block step served. The decode plans are
  prewarmed in every causal scratch set, not only the one in use (the
  sliding set compiled flashkvpaged and flashmergewide at a step). Open:
  the step's tables on a split model (each set holds one table where the
  plan states two; served, the CUDA laptop card hung in the whole step), so
  the input rows and tables of Gemma 4 stay declined by name
  (DeclineNoTable); per-layer embeddings and KV sharing (E2B/E4B) and the
  26B's dense MLP beside its experts are G4's.
- **A lone row steps on the joint arm; the one-row cost is the capacity
  form's.** On a device whose steps over rows are composed in the reserved
  width (nn.RowsComposer), StepRuns now sends one State's run through the
  joint arm as a step of one session (model.stepAlone), a prompt in chunks
  of the device's width, and Model.LoneRows counts any row that still went
  through Prefill or Forward there. TestStepRowsAreTheirOwn holds every arm
  bit for bit, the one-at-a-time ones included, and LoneRows at zero
  (stories15M and Qwen3-0.6B on the laptop, Qwen3-30B-A3B on two V100s);
  with the route off both lone arms differ in every logit (up to 0.54 on
  Qwen3-0.6B) and 2914 rows go lone. TestDecodeGroupsHoldUnderLoad passes
  on two V100s with Qwen3-30B-A3B (16 requests, prompts 3072 and 384, 64
  out): 0 of 32 requests part in waves, 0 of 32 in the closed loop, one
  group from itself 0, 277 grouped steps; arms 8m15, 8m25, 7m38. The price
  is the reserved-width form at few rows: on Qwen3-30B-A3B over two V100s a
  lone decode step costs 86 ms on the joint arm against Forward's 11, and a
  lone 960-token prompt 1.6-2.8 s against Prefill's 0.9-1.25 s
  (backend.SetCUDAKernelTiming, per kernel, at one row: the tiled
  projections 44 ms, one tiled float reduce a block 27 ms, 563 us a launch;
  a 2048-to-128 float matvec, the router's shape, forms at 512 rows as one
  workgroup of 128 threads at one row). A tiled projection computes its 16 rows a thread whatever the
  call's rows, with no split of its sum: at one row Qwen3-0.6B's q, k, v, o,
  gate, up and down launch 8 to 24 workgroups of 128 threads where the
  one-row form launches 128 to 192 of 1024. Splitting a tiled kernel's sums
  within a workgroup by the segments one unit of rows alone would take
  (16 at 32 elements a group) brought the 30B's joint step from 86 ms to
  52; it is not landed: its combine reads every row's segments, 1441 ops at
  31 rows against TestRowTilesHoldTheOps' 1,000. A second form, each thread
  storing only its segment's rows (the tile's row nodes kept on the kernel),
  holds the bound and passes the rest of jit's tests, but the gate's
  check that it discriminates reads workgroups, which the split multiplies
  (5152 untiled against 1184 tiled at 32 rows, where it wants eight times):
  that check needs a measure of weight loads a row before the form can land,
  and it was not run on a card. Even split, a tiled form computes all 16
  rows of its tile at a one-row call. TestLongPromptHasHistoryRoom
  passes gpt-oss-20b's two 2051-token prompts on one V100 on this base with
  the route and without it (budget 14620 and 14538 MiB, composed instances
  10385 and 8347 MiB, reserved width 256 rows).
- **A tiled projection splits its sums by one row's unit, and few rows run
  untiled.** Split now cuts a tiled form's sums within a workgroup by the
  segments one unit of its rows alone would take (groupSplitRows), each
  thread storing its segment's rows; where the device rounds every float
  operation alone (PTX's .rn, jit.Device.ExactArith) the split's untiled
  form is formed beside it on the same operands, each row's sums cut and
  combined in the tiled one's order, and launched in its place at a call of
  at most four rows (Split.Few, Code.RunsAt). A row past the call's stores
  the call's last row's sum from that row's partials: before, the tile's
  copies of a row were summed apart and SPIR-V wrote row 30 of a 31-row
  call other bits than the same row at 512. Blocks leaves both forms of a
  pair alone: it blocked the one-fold form for few rows and never the
  tiled one, and the 30B's router then summed in two orders (four arms of
  TestStepRowsAreTheirOwn apart by up to 1.6 a logit); TestComposedGPURowTiles
  holds every call's rows bit for bit to the capacity's call, a router's
  128 by 2048 floats among them, and fails without the guard.
  TestRowTilesHoldTheOps reads weight loads a row where it read workgroups:
  16x with the pass (1277952 against 20447232 at 32 rows), failing at
  RowTiles of 2. On Qwen3-30B-A3B over two V100s (CUDA, decode at capacity
  16, in-process interleaved against Forward a row a State, n=12, A/A
  0.986 to 1.000): a joint step at 1, 2, 4, 8 and 16 rows reads 86.6, 87.7,
  91.0, 100.0 ms before and 18.7, 19.9, 22.7, 34.2 and 47.3 after, against
  Forward's 11.3 ms a row: 1.65, 0.88, 0.50, 0.38 and 0.26 of it. The form
  for few rows holds fewer elements a workgroup where a row's fill fewer
  workgroups than the device has multiprocessors (jit.fewGroup, its
  segments the tiled form's): the router's 128 outputs ran in 4
  workgroups, 51 us a launch, and run in 64, 14 us (one row 21.3 ms before
  it). A tiled form whose rows are wide enough to take no split (the
  head's 151936 outputs) has its untiled form beside it too: the head at
  one row 0.96 ms before, 0.30 after, against Forward's 0.23 (the wall at
  one row unresolved, 18.7 and 18.8 ms in two runs;
  TestComposedGPUWideRowsAreTheirOwn holds 65536 outputs bit for bit at
  every call and fails with the few form storing the tile's second row).
  At one row the device runs 22.8 ms and 2119 launches against Forward's
  13.4 ms and 1371: the mixture's experts, unsplit at a capacity (5.8 ms
  against Forward's split 4.2), the attention (2.0 ms against 0.9) and the
  int8 forms Forward runs inside its consumers (241 launches, 1.7 ms).
  Splitting a capacity's group sums by one row's unit too (every capacity
  alike, so rows stay their own), before the router's workgroups, brought
  one row from 21.1 to 18.1 ms and two from 22.2 to 19.9, but 8 and 16
  rows from 35.6 and 48.5 to 37.6 and 52.9 and the 960-token prompt from
  1645 to 1917 ms: a regression, not landed. A lone 960-token
  prompt is 1651 ms on the joint arm (2397 before) against Prefill's 327:
  the capacity forms at 256 rows against the tier's tensor-core prefill.
  The routes that still run one State at a time on such a device are
  counted by name (model.LoneRoute: forward, prefill, prefill-cached,
  prefill-mixed, verify) and the server warms a model through StepRuns;
  TestStepRowsAreTheirOwn holds every arm bit for bit with every route at
  zero. TestStepAcrossSessionsMatchesEachAlone holds a composed build's
  sessions alone on the joint arm bit for bit to their joint step (the
  route off, Prefill and Forward, reads 1.8e-3 to 3.2e-3 NMSE), and
  NoRagHeadOne makes the composer's head project every row of the step, the
  one-row head's control. TestDecodeGroupsHoldUnderLoad passes on the two
  V100s (0 of 32 requests part in waves, 0 of 32 in the closed loop, one
  group from itself 0, 396 grouped steps; arms 6m17, 5m41, 5m09 against
  8m15, 8m25, 7m38 before), and a lone run's step on the joint arm makes no
  engine allocation (TestDecodeDoesNotAllocate's step-solo, Llama-3.2-1B on
  a 3050 Ti, CUDA and Vulkan, the default tier and the four composed arms).
- **The composed build does not compose prompts: they run main's tier.**
  The composed build's default (ComposedStep) is the embedding, tables,
  blocks, head and sampler; ComposedPrompt is off, so every prompt chunk
  runs the tier's own prefill (its GEMMs and attention, family branches
  included). The tier had been running those chunks on its decode plan's
  paged attention (158 ms of a 1024-token Qwen3-30B-A3B prompt's 397 ms of
  kernels, two V100s); 0df01475 has it run its prefill attention instead
  (24 ms), so the composed build's prompts read as main's (3569, 4740 and
  4685 tok/s at 512, 1024 and 2051 tokens against 3589, 4745 and 4691).
  That is main's path measured against itself, not the composer: the
  composer's own prompt forms are 5.1x slower than the tier's prefill on
  a lone 960-token prompt, having no tensor-core form on sm_70 (the m8n8k4
  layout is not lowered in jit/ptx/tile.go). Gate:
  TestComposedStepPromptRunsPrefillAttention (56 launches; 28 on the old
  code).
- **A mixture's routed experts are served composed from pages, on the cards
  and on the CPU device.** The composer's expert matvec reads a bank held in
  pages through a table of the step (`jit.Request.Paged`: each expert's
  sheet of the held bank, bounded by the sheets held) and a bank whose
  sheets lie a page apart at that stride (`Request.Strides`). On a card a
  streamed block (`Config.StreamExperts`, the tier's compact bank or its
  expert cache) runs its form in two halves (`servedPart.pagedRun`): the
  codes up to the selection, the selection read back, the tier's own fill
  from the host's pager, the table written (the cache's slot map), the codes
  that read the bank. On the CPU device the pager keeps each expert page's
  frame at that page's place in one reservation of the file's expert array
  (`format/jlm/exparena.go`: committed on page-in, `MADV_DONTNEED` on
  eviction, the budget, LRU and fault path the pager's own), so a bank is
  one view at the expert page's stride (`File.BankView`), and the form's
  second half runs once `ensureExperts` holds the selection's pages.
  Composed against whole banks the pages read bit for bit (NMSE 0) on CUDA
  and Vulkan, the 3050 Ti and a V100, for Qwen3-MoE-4x0.6B, OLMoE,
  DeepSeek-V2-Lite and the Q8_0 fixtures of glm4moe, qwen2moe, ernie4.5-moe,
  hunyuan-moe, dots1, DeepSeek-V3 and V3-lite and gpt-oss, a compact bank,
  a cache of a quarter of the experts (OLMoE on the V100: 2092 hits, 1620
  misses) and both under a one-page host budget (3728 page-ins against 816);
  a page read from the wrong expert and an eviction not re-read each fail
  it (TestComposedExpertPagesServe). On the CPU the composed mixtures sit
  0.000 to 0.703 of main's int8 spread from main at the median, dbrx's float
  banks 1.55e-10, and a one-page budget gives the resident logits bit for
  bit with 240 to 3360 page-ins (TestComposedCPUExpertPagesServe). Moved
  card to card and card to CPU and back mid-sequence, the logits stay in
  the devices' band (TestComposedExpertPagesMove,
  TestComposedExpertPagesMoveToTheCPU). Left by name: a streamed block at a
  step of several rows (`DeclineExpertsRows`: its union of experts is not
  in a compact bank), a hybrid block's experts on the host
  (`DeclineExpertsHost`, the tier's default for a block no card holds
  whole: 261 of 1392 block steps of Qwen3-30B-A3B on a V100), the CPU's
  rows form over banks (`DeclineExpertsRows`), and a top-1 mixture, which
  the tier never streams to a card.
- **A device without exact division forms the int8 codes map on the CPU, a
  chain too long to unroll is left unformed, and a Vulkan pipeline's
  create-info reaches nothing on a goroutine's stack.** (1) A CPU device
  stating `ExactDiv: false` failed to form the int8 quantizer: the scale's
  quotient was an integer division of lanes ("div of integer lanes"), and the
  correctly rounded reciprocal one map past the vector registers ("register
  pressure at node 27"). The quotient is now 2m plus floor(2m/127) through a
  float estimate one under at worst and one remainder correction (all 2^23
  mantissas right), and the reciprocal is three maps (the Newton step, the
  four candidates tested, the pick). TestComposedCPURowsMatVec and
  TestComposedCPUMatVec carry an "int8 inexact division" arm (180 row-selected
  matvecs, 240 matvecs, all held to the oracle); the same arm failed to form
  before. (2) The sampler's candidates over a 131072-entry vocabulary on
  an inexact-division GPU (the laptop's Intel vulkan:1 beside cuda:0, Apertus's
  head) never finished forming: Coarsen unrolled a 131072-term reduction in a
  thread, 7,976,810 nodes at 65536 (3.2 s, 1 GB) and more than 3.3 GB in
  nine seconds at 131072, which the 8 GB cap throttled into a test that read
  as a hang (TestBlocksMoveBetweenDevices/synth-apertus/step). A chain past
  2^20 nodes is left unformed by name now (0.66 s and 0.86 s, its members
  running as the kernels they are); SampleCases holds 65536 and 131072 too
  (40 cases bit for bit on the CPU, ptx and spirv). (3) vulkan.Ctx.Compile
  handed the driver four structs by a uintptr field (the push range, the
  required-subgroup-size info, the pool size, the layout handle) that lived
  on the compiling goroutine's stack, which moves when it grows or shrinks
  and rewrites pointers, not uintptrs: composed TestDecodeDoesNotAllocate
  died after 955 subtests in vkCreateComputePipelines (SIGSEGV, address 0,
  on a compile goroutine). They live in the Kernel now (`-gcflags=-m` moved
  the other create-infos to the heap and not these), and the descriptor
  layout's bindings and the SPIR-V bytes, also reached by a uintptr only,
  are kept alive across their calls. TestComposedGPUCompilesSideBySide
  (eight goroutines compiling the sampler's 52 kernels for ten seconds beside
  a collector every 2 ms at GC percent 5) is the gate: before, the same loop
  died in the driver after 1080 rounds of four goroutines (SIGSEGV in
  vkCreateDescriptorPool) and held 8 GB without returning at eight and
  sixteen; after, 1043 rounds of four, 236 of eight and 193 of sixteen on the
  RTX 3050 Ti's Vulkan made no error and held 264 to 322 MB. The composed
  TestDecodeDoesNotAllocate then ran whole on a V100, with the heap-held
  structs and not yet the two KeepAlives: 878 subtests pass, none fail, no
  SIGSEGV (the run that died had 955). (4) A decision model's
  every-block arm in a build that composes the prompts holds 6.49 GB of
  retained scratch after its first run against a 6.10 GB budget, so the arm
  after it on the same card placed 8 of 32 blocks and the gate indexed an
  empty answer; the arm states a second gigabyte and fails by name. The
  composed picked-rows head and the whole head's gather part by 1e-6 in a
  probability on lev (V100, ptx), and the gate's equality (the picked labels
  answer the row's bit for bit) does not hold in a build that composes the
  prompts: the two forms are two orders of one Q8_0 sum (their output counts
  differ, and the passes cut the reduction by the count, jit/pass_split.go;
  which pass moves the bits is not isolated), and on a random Q8_0 head of 4096 to 131072
  rows 45 to 54 of 64 picked rows differ from the whole matvec's row in the
  last bits, worst 6.1e-5 of a logit, on ptx and spirv alike. That is
  forms differing, not a wrong kernel (the answers sit 0.0456 from the host
  with either, and a pick one label off moves them 0.69); the gate is left
  as it was and the arm stays red until its equality is decided.
- **An attention with sinks keeps the split form; a split's partitions can
  run in order.** The composed build's prompts run the composer's own forms
  since e73e0496. gpt-oss-20b's attention took its sinks into the maximum
  by a map between the keys' max and the weights. That map broke the chain
  SplitRange and WorkFuse recognise, so every sink attention ran as eight
  kernels over [rows, heads, keys] buffers at the key bound: 604 MB apiece
  at 512 rows and 4608 keys, three of them live. Now the max is the keys'
  alone and exp(sink - max) joins the sum in the normalising map, which the
  merge reads at the partitions' greatest max. The same attention is then 3
  kernels with a 50 MB partial. The reduction join also moved before
  Prefetch (pm1's e2000410), so MXFP4 expert dots no longer write their
  int32 sub-block sums at a step's rows. Composed gpt-oss-20b then holds
  every block on one V100: 13.06 GiB, against 14.33 GiB with 17 blocks on
  the host. Its prompt went from 29.9 to 144, 143 and 141 tok/s at 512, 1024
  and 2051 tokens (httpbench, card 6, A/A 1.000 within 0.001). Main reads
  4633, 4700 and 4397 tok/s on the same card, so the composer is still 32x
  slower, and nvprof does not see where the 14.6 s of a 2051-token prompt
  goes (the host's read waits 708 ms a submission). Where the rows'
  workgroups alone fill the device (Workgroups.sequential), the partitions
  are a chunk of positions each and one workgroup runs them in order
  (Running, the pass). The partial each one stores goes to workgroup
  memory and is combined into a running max, sum and weighted sums, kept
  in two slots that alternate by trip. The partial then holds one slot a
  row and head (16.8 MB for gpt-oss at 512 rows, the same at any key
  bound), and decode keeps its parallel split. Llama-3.2-1B's composed
  prompt reads 140.8, 304.6 and 748.1 ms against 141.1, 305.3 and 732.2
  before. That is +2.2% at 2047 tokens, from servers run one after the
  other, not interleaved, so it is unresolved. Gate:
  TestComposedGPUAttendRows: 37 live rows of a 64-row form from position
  2100, sliding and sink cases, ptx (sm_86, sm_70) and SPIR-V. It holds
  the attention to the definition at NMSE 1e-8 where the lowering rounds
  every operation and 1e-6 elsewhere: the V100's Vulkan driver reads
  1.4e-7 on seam-wip's form too, the laptop's 3e-14. It fails with the
  first position moved, the sinks zeroed or the combine's partial weight
  forced to one (NMSE 0.14-0.41), and on seam-wip every sink case fails on
  the scratch. The alone-on-the-box baseline (D42): c3848f05 on card 6 with
  its host threads on node 0 reads 361 ms a 1024-token gpt-oss prompt
  composed and 214 ms on main, as recorded. PM1's 5.2 s ran pinned to node
  1 beside s43's CPU sets, and alone on node 0 the gap is gone.
- **Per-layer embeddings and a shared history, stated as ops.** The input
  rows' graph writes each row's per-layer inputs beside its row (the
  per-layer table's row by the token times sqrt of a layer's width, plus
  the scaled row's projection over sqrt of its width, normed per layer
  slice, the sum over sqrt 2; an id past a table that stops short reads row
  0), and every block of such a model reads them as its second input
  (jit.Front's feed): its own slice, gated by its stated activation over
  its gate of the residual, projected, normed and added before the output
  scale. A block sharing another's history projects q alone and attends
  the state kv@N, which the attention's decomposition already resolved;
  the CPU device binds the source block's pages and the per-layer rows
  (cpuArgSide), the tier the source's pool and its staged per-layer rows.
  UncoveredKVShared, UncoveredPLE and the declines of a shared history are
  gone: synth-gemma4-e and synth-gemma4-e-ragged went from 8 blocks refused
  for per-layer embeddings, 4 for a shared history and both input rows
  refused, to every block and the input rows stated. On the CPU device
  every block step of both serves, decode and a prompt of 8 with its
  decode, at 1.1e-10 and 6.6e-10 NMSE from main; the violations (the
  per-layer inputs' norm unweighted, the sliding reader on an earlier
  source, the block's post-norm unweighted) read 0.090, 0.256, 0.141 and
  0.072, 0.083, 0.106. Each block given the next block's gate read 1.16
  and 1.40 on the CPU, but places no block on CUDA, so the card arms
  cannot run it. On a V100, CUDA and Vulkan, every decode block step of
  both serves at 7.1e-12 and 6.6e-12 (5.0e-12 and 5.6e-12 ragged) NMSE
  from the host, the prompt's block steps all served, and the three
  violations read as on the CPU. Real Gemma 4 E2B states every block and
  its input rows as ops and forms for CPU and GPU (the census). Each
  block's slice of the per-layer inputs is an offset in its form, so no
  two blocks of such a model share one: E2B composes a form a block where its
  geometries would give a handful, the upgrade an offset the call site
  binds. A front end reading the model without its per-layer
  embeddings refuses it by the unbound gate, and one reading a reader as a
  writer by the keys it does not store. The device holds no per-layer
  table, so a card declines the input rows of such a model by name
  (DeclineEmbedWeight) and the host fills them.
- **The composer's matrix kernel at 1.25-1.5 of GemmVolta, and a mixture's
  routed experts on the matrix unit.** Without a profiler (the V100 box
  allows none), the composed kernel was read by its SASS (ptxas and
  nvdisasm on the laptop, `-arch=sm_70`): weight loads were already shared
  by ptxas (merging them in Affine, `Affine.Weights`, moved neither the
  register count nor the time), the tile shape alone moved nothing
  (JITLLM_MATRIX_SHAPES sweeps), and what paid was the row's factor written
  once as binary16 words by a kernel of its own (`halves`), two stages with
  one barrier a trip and every staged word loaded a trip ahead
  (`Prefetch.Staged`), 2x2 warps of 64x32 with two workgroups a unit
  (`Matrix.Resident`), the sums through the stage a row of tiles at a time,
  and Affine's `(x+c)>>n = (x>>n)+(c>>n)` for x a multiple of 2^n. On the
  V100 (card 7, CUDA, sm_70, 512 rows, median of per-round ratios, the
  composed arm's two kernels against GemmVolta): q/o Q4_K 2.38 -> 1.50,
  gate/up Q4_K 2.36 -> 1.45, down Q6_K 1.95 -> 1.25; an A/A of the composed
  form read 1.51/1.50, 1.45/1.45, 1.25/1.25. A mixture's expert matvec at a
  chunk's rows is grouped by expert (`expertGrouped`: counts, run starts
  padded to 64 rows, ranks, a scattered table, each block's expert) and
  projected on the matrix unit, whose pass takes a weight read through the
  row's block of 64 (64 rows by 128 outputs), a block past the runs (the
  runs' end its stated live rows) running no trips. Gates: TestComposedGPUExperts
  (Q8_0, Q4_K, Q6_K, MXFP4 against the oracle, worst NMSE 1.6e-7; the
  experts moved by one and the next pair's reference fail it),
  TestComposedPromptRunsOnTheMatrixUnit (Qwen3-0.6B and Qwen3-MOE-4x0.6B,
  392 matrix launches each, NMSE 3.105e-4 and 3.159e-3 against the tier's
  3.071e-4 and 3.124e-3). End to end (PM3's servers, one request at a
  time, V100 card 7, Qwen3-30B-A3B on cards 6 and 7, composed prompt rate
  over main's at 512/1024/2048 tokens): Llama-3.2-1B 0.46/0.39/0.30 (was
  0.092), Llama-3.1-8B 0.49/0.41/0.30 (was 0.027), Qwen3-30B-A3B
  0.38/0.32/0.25 (was 0.094), gpt-oss-20b 0.051/0.053/0.057 (was 0.0064);
  Qwen3-30B-A3B's 16-token answer is main's word for word. Per-kernel
  device time of gpt-oss's composed 512-token prompt in the CLI
  (JITLLM_CUDA_KERNEL_TIMING): 175 ms matrix, 103 ms plain reductions off
  the unit, 14 ms grouping; the server's prompt is four times the CLI's.
- **Four faults around the mixtures, each at its cause.** (1) The composed
  build held 20.9 GB resident in `TestBatchSeamMovesCarryEveryRowMoEFamilies`
  (glm4moe alone, V100) against main's 1.9 GB: the coarsener, asked for the
  sampler's candidates over a 151552-word vocabulary, raised its unit to
  606208 for a chain whose first reduction is an integer sum, which no
  exchange between threads carries, and unrolled the rest of the chain over
  the vocabulary (29 million nodes, 32251 MiB allocated, 42 s) before it
  acted on the refusal; it now stops at the first (`TestCoarsenRefusesWithoutBuildingTheChain`,
  32251 MiB before, 0.02 s after). The whole family gate, 12 fixtures and 48
  arms, now peaks at 1.38 GB in 84 s. (2) `synth-phimoe` read logits of NMSE
  0.63-0.76 on CUDA sm_70 and passes on sm_86 and Vulkan: not the card but
  the capacity its batch forms at (8 rows with sm_70's tile unit, 32
  otherwise). `Split` and `SplitExtreme` put their segments in front of a
  form's space, so `CallRows`, holding a call to the rows of the leading
  axis, took the segments for the rows, and a call of 3 rows read partial
  sums nothing had written. The segments sit behind a leading rows axis and
  the combine carries the capacity; the same fix lets the head's normed rows
  through the prologue (`Operand.Out` is stored whole by every workgroup).
  `TestComposedGPUHead` holds the head to the oracle at capacities 1 to 32,
  calls of fewer rows, on ptx, spirv and ptx lowered for sm_70, with guard
  words (60 NaN rows before, none after; the laptop's sm_86 card reproduces
  the V100's failure with `OpenCUDAWith(CUDATarget: "sm_70")`). phimoe on
  the V100: 0.667 and 0.76 before the first move, 3.8e-07 and 2.1e-07 now.
  (3) A streamed prompt differing between sessions (5.7e-04 to 8.4e-03) was
  two paths, not a stale cache: the first prompt of a model is one chunk,
  its streamed experts on the host (`emitGroupedMoE`), and `forwardRowsFor`
  sends every short prompt after it through the decode path, the card's
  experts, once `hostExperts` is set; sessions two onward agree bit for bit
  (the first sits 1.1e-04 from a resident card, the later 6.5e-04). The
  gate pins the path (`WithPrefillForwardRows`) and holds three sessions
  to the first bit for bit. (4) The eviction gates squeezed a budget the
  pool had already grown past: the prewarm covers the gate's 449 positions
  (112 pages on a 3050 Ti), so nothing streamed; they open without it. The
  prefill-stream gate's budget of half the model left a plain build on the
  3050 Ti no slot (363 MB charged of 401 MB, a block 39 MB: 0 page-ins);
  it takes the least room, in blocks, at which the prompt pages in (the
  plain build one block, 26 page-ins; the composed build none, 8192).
- **Predicted residual streams stated as ops, and buffers bound at an
  offset.** The AltUp and LAuReL roles are named for what they do
  (RoleStream*, RoleLowRank*, Config.PredStreams; codes and container bytes
  unchanged). A block of S predicted streams reads each stream as an input
  of its own and writes each as an output: the router's weights over
  stream 0, the coefficients as windows of the stored tables weighted by
  them, fma chains for the predictions, the block on stream 0's prediction
  with the side branch joining the attention's residual, then the
  correction and the per-layer input into streams 1 and up. The gaussian
  top-k is the container's stated activation, which needed a pairwise
  reduce (OrderTree, a map a level over powers of two); the per-layer
  input reads the activation of a block past the sparse lead. Main's
  stream-major residual is kept: the CPU binds stream k at its offset (a
  stride per call), and the cards bind it through backend.View, a buffer at
  a byte offset (CUDA's pointer plus offset, Vulkan's descriptor offset held
  to minStorageBufferOffsetAlignment and refused by name off it, Metal's
  setBuffer offset). The composer keeps each stream's storage its own
  (Operand.Stream): without it the in-place pass stored stream 3's
  prediction over stream 0's input and stream 0 read 0.086 from main. On
  synth-gemma3n every block step serves on the CPU at 2.75e-07 (decode) and
  2.77e-07 (a prompt of 8 and its decode) NMSE from main, and on an RTX
  3050 Ti at 3.72e-11 (CUDA) and 1.02e-10 (Vulkan) from the host (the
  same on a V100), every decode and prompt block served; the six
  violations (prediction, correction, side branch, the gaussian on no block
  and on every block, the correction's scale) read 0.090, 0.0037, 1.28, 1.36, 1.56-1.57 and
  0.037 on every device. The input rows and the head stay the host's
  (UncoveredStreams: the expansion and the collapse), and
  hyper-connections and residual attention stay UncoveredStreams. A
  block's slice of the per-layer inputs is now the side input's binding
  offset rather than an offset in its form: Gemma 4 E2B's 35 blocks
  compose into 22 CPU forms (decode and rows) in 12 min on a V100 box's
  E5-2680 v4, where each block was a form of its own and G4's serving gate
  passed 25 min unfinished. The CPU page fault of the serving gates now
  reaches only the slots every arena holds: on E2B a sliding layer's arena
  is smaller than a global one's, where the old fault's slot lies past it,
  and a run of the gate on E2B crashed the Go runtime in the collector
  after its page and row faults.
- **An attention over one sequence's rows runs on the matrix unit (the
  flash pass).** Where a prompt chunk's rows are one sequence's in
  position order (jit.Step.Seqs, Heads.Seqs: a prompt call, or a ragged
  step whose rows are one run, ragStep.oneSequence) and the device states
  its matrix tile, jit.Flash forms the decomposed attention chain -- the
  chain SplitRange already matched, now one function both read
  (rangeChain) -- as one kernel: a workgroup takes 64 rows of the
  sequence and a head, 32 positions a block up to its last live row's
  end, read from the descriptors at run time; the position factors of a
  block are staged once for every row of the tile, the dots and the
  exponentials' weighted sums are tile products, two threads a row keep
  the running maximum and sum, and the weighted sums stay in registers,
  rescaled by exp of the old maximum less the new. Every operand the
  history or the rows hold as float32 is staged as its binary16 rounding
  and what that leaves, the dots three tile products and the weighted
  sums two, so the form keeps the attend gate's float32 band (1e-8): its
  worst row reads 1.8e-12 on the laptop's sm_86 and 2.3e-12 on the V100's
  sm_70 (the running form 2.1e-13); dropping the leftovers' products reads
  1.1e-7 and fails. No buffer
  holds a row's scores and the split's partial and merge are gone (three
  kernels a block become one). A window's rows (rows.win), a start that
  is not the descriptor's own, a head over 16 rows a warp's tile, or a
  device with no tile keeps the running form; the tier forms both where a
  ragged step's width can mix sequences (promptServed) and counts the
  attentions run on the unit (Stats.AttendLaunches), which
  TestPrefillMatchesForward/device holds above zero on CUDA (16 of 16
  composed block steps on Llama-3.2-1B, 40 of gemma-3-1b's 260, its
  global layers). Measured on a V100 (CUDA, sm_70), Llama-3.2-1B-Instruct
  Q4_K_M, jitllm speed with JITLLM_CUDA_KERNEL_TIMING: the composed
  prompt's attention went from 13.2 to 2.7 ms at 512 tokens and from
  221.6 to 29.3 ms at 2048 (main's flashprefill: 1.5 and 17.8 ms), the
  prompt's kernels from 65.4 to 54.6 ms and from 430 to 236 ms (main: 22.6
  and 104 ms). The whole composed prompt against the base, jitllm speed a
  process an arm alternating for six rounds, host threads on the card's
  node: 0.844 / 0.770 / 0.581 of the base's time at 512 / 1024 / 2048
  tokens (IQR/median 0.013 / 0.014 / 0.009; the A/A pass beside it read
  1.018 / 0.997 / 1.004, IQR/median at most 0.028), 57.1 / 138.5 / 268.1
  ms against 67.7 / 178.6 / 460.5. What is left is the projections: 47.6 ms of the 54.6 at
  512 and 191 of the 236 at 2048, against main's GemmVolta 17.4 and 72.7
  ms. The prompt is GPU-bound: the server's time to first token reads
  within 2.6 ms of the kernels' sum at 512, with no host read or form
  inside the prompt. On the matrix kernel of 123cbc14 the prompt's kernels
  read 47.5 -> 37.0 ms at 512 and 357 -> 169 ms at 2048 (main 22.4 and
  99.9): the projections 29.8 and 119 ms with their halves kernels (main's
  GemmVolta about 17 and 68), the attention 2.9 and 33.4 ms (main 1.5 and
  17.0). The attention lost a little there (2.7 and 29.3 before): the
  page-table Prefetch now leaves PrimMatrix kernels alone, and the
  weighted sums' fold steps, which the lowering now makes as a trip
  starts, add the block before rather than the trip's own.
- **A picked row is the whole head's row to the bit.** The composed head
  over the picked rows (graph.Head.Pick) and the whole head cut one Q8_0 sum
  two ways: the split passes chose the segments by the form's own output
  count, so on a V100 the whole head of 65536 rows summed each row in two
  segments of 32 blocks and the 512 picked rows in 32 segments of two. A
  selection's primitives now state the matrix it picks from (jit.Pick, set
  by the row-selected Q8_0 matvec), and Split decides its segments by the
  form over every row (Kernel.cutElements); the whole head's form is
  unchanged by construction (no Pick, the same elements). On a random Q8_0
  head of 4096 to 131072 rows of 2048, all 512 picked rows equal the whole
  head's bit for bit on ptx and spirv (V100 and RTX 3050 Ti); with the cut
  by the selection's own count 408 to 439 of them part, and uncut where the
  whole head is cut 400 to 427 part (TestComposedGPUPickedRowsAreTheRows).
  TestDecisionOnEveryDevice/lev passes its equality (the picked labels
  answer the row's) on a V100's CUDA and Vulkan in both builds, where it
  read 1e-6 in a probability.
- **No struct the Vulkan driver reads through a uintptr lives on a stack.**
  After Compile's fix, the same shape stood at nine more sites: the pNext
  chains of the subgroup, UUID, core-count, host-memory and memory-budget
  queries, the device's queue, dot-product and subgroup-size create infos,
  the application info, and the import's external-memory structs. Each was
  a local whose address went into another struct's uintptr field, so a
  stack copy on the way into the driver (any call that grows the stack, the
  FFI wrapper's included) left the field pointing at the old stack. They
  are on the heap now (vulkan.onHeap, at open and at an import), and
  Mem's budget struct is the Ctx's, under its own lock, so a budget read
  allocates nothing. CUDA and Metal hold no such field. The escape
  analysis gains twelve heap allocations, all at device open or an import,
  against the base, which kept every one of them on the stack. The
  side-by-side compile gate passes (V100: 75 rounds of eight goroutines, 0
  failed, 412 MB; RTX 3050 Ti: 280 rounds, 260 MB), and the composed
  TestDecodeDoesNotAllocate on Vulkan passes on the laptop for every model
  but the Gemma 4 fixtures, whose composed head at a step of 3 rows wants
  2.68 GB of buffers the 4 GB card does not hold, and on a V100 for those
  (vulkan:2, every subtest).
- **Tower blocks composed on the CPU device and the cards.** The front end
  states each vision tower block as ops (engine/model/fronttower.go): its
  norms, biased projections, the 2-D rotary over the rows' table, a
  bidirectional attention (graph.Attrs.Bidir, RowWindows on a windowed
  block), and the MLP under the activation the tower's flags state
  (jlm.Vision.Activation, through the expression builder convert shares).
  Its tensors bind by the role each plays in the loaded block. A tower
  block it cannot state is named: Gemma 4's clipped linears
  (UncoveredClamp), Kimi-VL's rotary axes, Gemma 3n's convolution blocks.
  The census over every projector family's fixture states SmolVLM 12/12,
  llava 23/23, Gemma 3 27/27, InternVL 24/24, Qwen2-VL 3/3, Qwen2.5-VL 4/4,
  Qwen3-VL and Qwen3.5 4/4, GLM-4V 3/3, MiniCPM-V 27/27, Janus 24/24,
  Pixtral, Llama 4 and Phi-4 all; Kimi-VL, Gemma 4 and Gemma 3n none. On
  the CPU device (WithComposedCPU) a picture's rows run as one bidirectional
  run (bidir.go's runs, a windowed block's windows as runs), the history the
  State's arena page, rounded to a power of two because AVX2 has no packed
  integer division and llava's 577-row page had no x86 form of rem.u32. On
  a V100 host's CPU, held to main's host tower: SmolVLM 12/12 blocks served,
  4.9e-6 after one block and 1.1e-3 for the encode, llava 23/23 at 3.7e-7
  and 9.5e-4, Gemma 3 27/27 at 1.3e-6 and 1.2e-3, Qwen2-VL 3/3 at 1.8e-5
  and 3.0e-4, Qwen2.5-VL (windows) 4/4 at 3.3e-5 and 1.8e-3; a warm encode
  allocates nothing; the rows made causal read 6.8e-2, 1.5e-3, 0.12 and
  0.45. On a card the tier
  serves a non-causal call's blocks when asked (tier.ComposedTower,
  towerserve.go): the rows' descriptors over the device's one transient
  pair as a page of the block's pool geometry, the windows the call staged
  as rows.win, a row-chunked set's phases as the descriptors' ends; a block
  without the rows' table, a window its graph states or a form the
  composer does not make is declined by name. On cuda:0 and vulkan:0 every
  block is served: SmolVLM, llava, Qwen2-VL and Qwen2.5-VL (windows) read
  1.4e-7..2.4e-4 after one block (5e-5, or 5e-4 on the binary16 matrix
  path) and 1.2e-4..5.0e-3 for the encode (1e-2); the rows made causal
  ("piccausal") read 6.8e-2 against 6.0e-5 and 1.4e-7. A relocation
  mid-encode under a two-page budget, 4 blocks composed on the card and 8
  on the CPU, is bit-identical to the same placement from the start.
  ComposedTower is not in ComposedStep: Gemma 3's SigLIP at 4096 rows reads
  NMSE 4.1 after one composed block on both card backends, where every
  tower of 1600 rows or fewer agrees (the attention past 2048 keys splits
  into sequential partitions, jit/pass_splitrange.go, the first suspect;
  scoping the combine's shared loads to the partition loop did not move
  it). The projectors and the entry rows stay main's generated kernels.
- **The matrix kernel at 1.11-1.34 of GemmVolta over a 128x128 tile, a
  selection's grouping shared, and shared memory past 48 KiB.** On the V100
  (card 7, CUDA, sm_70, 512 rows, BenchmarkComposedMatrix, both composed
  kernels over GemmVolta, median of per-round ratios, an A/A within 0.01)
  from 1.50/1.45/1.25 (q/o Q4_K, gate/up Q4_K, down Q6_K): A's staged rows
  padded to eight elements, so a lane's eight elements of an A fragment are
  one ld.shared.v4 (`ir.TileLoadAligned8`), B's kept at four, 1.50/1.34/1.21
  (both at eight read 1.62/1.52/1.33: a B fragment load reads rows n and n+8
  together, which a sixteen-byte-aligned row shares banks across); then the
  profile's tile 128x128x32 over 2x4 warps, one workgroup a unit,
  1.34/1.26/1.11 (615 instructions a warp-trip against 664; 128x128 over 2x2
  warps spilled). Measured and not kept: the weight's codes decoded two a
  word in binary16 pairs (sub and fma on f16x2; 603 instructions, mixed
  1.30/1.29/1.09 against 1.34/1.26/1.11, and Q4_K's worst NMSE 1.3e-7 to
  2.1e-7); the next trip's loads after the stage's stores, which ends the
  22 register moves a trip (581 instructions) and reads 1.61/1.53/1.36: one
  trip does not cover the loads; loads two trips ahead 1.36/1.30/1.20 (a
  chain of folds is right only when each fold is made before the one it
  takes, lowergpu setting the phis in order); 128x128x64 and 256x64x32,
  which compile now, 1.50/1.38/1.21 and 1.62/1.47/1.34; halving a dense
  workgroup until its count reaches the units, neutral (Llama-3.1-8B's
  prompt matrix kernels 164.6 against 165.3 ms). A mixture's selection is
  grouped once for the projections it routes (named by the selection,
  `Context.grouped`; TestExpertGroupingIsShared), a pair's rank is counted by
  blocks of 64 pairs, not over every pair before it (Qwen3-30B-A3B's grouping
  kernels 6.06 -> 4.96 ms a 512-token prompt), and a grouped projection's
  workgroup keeps the profile's 128 outputs over half the warps, two a unit:
  gpt-oss-20b's composed prompt (CLI, JITLLM_CUDA_KERNEL_TIMING) 157.2 ->
  140.2 ms of matrix kernels, 278.1 -> 261.2 ms of all. A run padded to 32
  rows where an expert averages 32 pairs read Qwen3-30B-A3B's matrix kernels
  127.8 -> 157.7 ms (each block reads its expert's weight again) and is not
  kept. Projections of one row factor (q, k and v of one normed input; gate
  and up) read one halves kernel's words, keyed by what it computes
  (TestProjectionsOfOneRowShareTheirHalves): Llama-3.1-8B's halves kernels
  5.90 -> 3.90 ms a 512-token prompt, 224 -> 128 launches. A kernel whose
  shared arrays pass 48 KiB lowers them into dynamic shared memory the CUDA
  backend opts it into (`ptx.DynamicShared`,
  `cuda.Module.SetDynamicShared`; TestSharedPastTheStaticLimit, which
  overlapped arrays fail). PM3 in one pass (servers, card 7, Qwen3-30B-A3B
  on 6 and 7, main, the base 123cbc14 and the head one after another, each
  an A/A within 0.4%, composed prompt rate over main's at 512/1024/2048):
  Llama-3.2-1B 0.479/0.404/0.287 -> 0.483/0.409/0.289, Llama-3.1-8B
  0.493/0.412/0.306 -> 0.535/0.440/0.320, gpt-oss-20b 0.398/0.378/0.333 ->
  0.412/0.389/0.339, Qwen3-30B-A3B 0.384/0.325/0.237 -> 0.401/0.339/0.246
  (the head before the shared halves); every composed 16-token answer is
  main's word for word. The gpt-oss rows PX recorded (0.051) and its server
  four times its CLI do not reproduce at the base: its server reads 276 ms a
  512-token prompt, its CLI 278 ms of kernels.
- **The composer's CPU lowering has an arm64 target.** jit/lowercpu.go
  lowered through jit/x86 only, so an arm64 host's CPU device ran no
  composed block. The CPU profile is unchanged: each of its sixteen
  eight-lane vector registers is a NEON pair (lanes 0-3 in V2r, 4-7 in
  V2r+1) and each general register an X register, the argument block in X0
  as trampoline_arm64.s passes it, so every pass and the binder serve both
  targets. Each node is lowered from the composer's IR through jit/cpu's A64
  encoder, which gains the instructions it emits (a64ops.go, pinned to
  llvm-mc's words); no second assembler and no main kernel. An operation
  within 128-bit lanes is its NEON operation on both halves, one that clears
  an x86 register's upper half clears V2r+1, and a temporary the binder gave
  no register is a pair register saved below SP around its use. A byte dot
  is SDOT where the device states FEAT_DotProd (cpu.ISADotProd, the bit an
  arm64 host's ISA carries) and SMULL/ADDP/SADALP where it does not. The
  CPU also serves an f32 history now (Registry.WithoutHalves: the profile
  without the Halves pass), the arm64 default cache, which every block
  declined before; and the per-worker words grow to a prefill pool wider
  than the decode pool, which on the M4 (P+E for a prompt) indexed past
  them. On the M4: every jit CPU gate passes (TestComposedCPUMatVec 240
  formed and right, the 60 int8 arms without dotprod among them), and under
  qemu as a Cortex-A72 (no dotprod) every one passes too; five mutants
  (SDOT on one half, the widened dot's pair sum, FMLS for FMLA, the lane
  sum's halves, the wide store's upper half) each fail them.
  TestComposedCPUBlocksServe passes on all eleven CPU-served models
  (median ratio 0.000 to 1.091 of main's spread, every violation out of
  the band). Gemma 4 E2B's page violation had never reached it on any
  host: the violation pointed the one shared table's first page at the
  last slot every layer's arena holds, and E2B's sliding layers (pages of
  their 512-position window) hold one, so it pointed nowhere and the
  logits did not move. It now points each layer at the last slot of its
  own arena (cpuFaultTable), and moves E2B 8.9 times main's spread at the
  median. TestCPUComposesOnce passes
  on all ten models, its relocation to Metal and back included, and the
  composed TestDecodeDoesNotAllocate host arm passes on Llama-3.2-1B and
  Qwen3.5-0.8B. Decode on the M4 (decode on the four P cores), interleaved,
  A/A 0.9998 and 1.0007: Llama-3.2-1B Q4_K_M composed 22.5 and 22.8 tok/s
  against main's 79.7 and 79.8 (0.282x, 0.286x; 803 MB a token, main at 87%
  and composed at 25% of the 73.4 GB/s wall). Qwen3-0.6B Q8_0's A/A read
  0.964 and 0.962 (IQR/median 0.005) twice, so the harness refused it.
- **The CPU's matvecs lost on the weight stream, not on the dot.** Decode
  decomposed per kernel (`WithProfile` now times each composed CPU kernel,
  `TestComposedCPUProfile`), on the V100 box's E5-2680 v4, node 0, 12 cores,
  64 tokens after 32: Llama-3.2-1B Q4_K_M main 19.6 ms a token (51 tok/s,
  41 GB/s, 70% of the 58.5 GB/s wall), composed 56.6 ms (0.35, 14.2 GB/s),
  99% of it in kernels, 94% in matvecs -- the 128256-row head alone 16.1 ms
  (28.8%, 13 GB/s), gate and up 550 us each (17 GB/s) -- and the small
  kernels about 3 ms against main's 0.5; Qwen3-0.6B Q8_0 main 15.2 ms (88%
  of 47.5 GB/s), composed 21.4 ms (0.71), its head 3.8 ms (18%). 129 and 225
  pool regions a token (main 1) cost about 4 us each. Instructions retired
  could not be counted: `perf_event_paranoid` is 4 on the laptop, the
  homelab Xeon and the V100 box. On one core without VNNI the Q4_K kernel
  reads 8.5 GB/s from L2, 4.7 from L3 and 2.3 from memory (Q8_0 24, 10.9,
  3.5): the dot keeps up, the misses do not overlap. The device layout puts
  a word of every row together, so a call's eight rows read 32 bytes a page
  and jump a plane stride (rows x 4 bytes), which neither the L2 streamer
  nor the stride prefetcher follows. `Ahead` (after `Regs`, CPU only)
  prefetches each such weight load's line two trips on (`PREFETCHT0`, the
  load's own pointer and offset plus two cursor steps), where a trip moves
  the pointer at least 2 KB and the trip's loads through it are lines apart
  (a run of consecutive lines is left to the streamer):
  `TestAheadPrefetchesTheWeightStream` counts the prefetches with and
  without VNNI and none at no trips or 16 rows. Llama-3.2-1B, the same
  profile, one pass each: composed 72.3 ms without against 29.8 ms with
  (main 23.6 and 17.6 in those passes: 0.33 to 0.59), the head 19.95 to
  6.66 ms, gate and up 706 to 304 us; one trip ahead read 42.1 ms against
  main's 29.6 (0.70) in a pass where main reached 48% of the wall, so the
  distance is not settled. Qwen3-0.6B
  Q8_0 is not resolved: 24.6 and 21.8 ms without, 27.4 (55% of the cores
  busy) and 20.3 ms with. Interleaved (`TestComposedCPUAB`, n=20, the box
  shared with the card agents), decode stayed unquotable: the composed arm's
  IQR/median 0.13-0.34 where main's A/A held 0.04-0.10 (129 barriers a
  token make it the arm a busy core stalls), but two passes agree: 0.4465
  and 0.4265 before, 0.5577 and 0.5683 after (composed 29.7 against main's
  52.8 tok/s, 47% of the 51.2 GB/s wall). Prefill at pp100 is quotable:
  0.2115 before (74.7 against 342.0 tok/s, A/A 1.0066, IQR/median 0.057),
  0.2356 after (79.6 against 336.6, A/A 0.9968, 0.055). Also: a load its reader takes as a memory operand
  took a general register and a red-zone slot, and the lowering stored that
  register after it each time (64 dead stores a trip in the Q4_K kernel, 8
  in Q8_0's); `Regs` gives it none. Tried and not kept: strip-mining the
  jam two vectors a trip (`Jam.Strip`) jams Q8_0 (+9% isolated) but the
  registers refuse Q4_K and Q6_K, and the refusal unwidens them to a row a
  lane (3x slower); halving the jam before unwidening forms them again at
  one vector but the retries took a Llama block's composition past 25
  minutes.
- **The prompt's attention takes each block in its own trip, and its
  maps and halves fold away (Recompute, Same).** The flash loop states
  StepsLast again (jit.Loop): its fold steps are made after the trip's
  stores, behind its last barrier, so each trip adds its own block's
  weighted sums rather than the block before, and no step's values are
  held across the trip. Its element indices are the thread's own
  quotient and remainder beside constants (flashBuild.split), and equal
  reads (a position's page) are merged once built: ptxas reads 255 -> 221
  registers on sm_70, the workgroup memory (40604 bytes, two workgroups a
  unit) unchanged. The page-table Prefetch given back to the flash kernel
  alone was slower on top of this (2.50 and 27.04 ms against 2.41 and
  26.68) and is not taken. Recompute runs a map in each map that reads
  its output (at most two reads a reader, no chains: unbounded, a whole
  step's form grew past 120 GB of host memory), so the norms run in the
  halves kernels and the gated activation in the down projection's
  halves; Same drops a kernel that is an earlier one with its scratch
  output named apart, so q, k and v share one halves kernel and gate and
  up another: a dense block's 23 kernels become 17. Running the residual
  adds and the activation in the matrix kernels' output loops (Epilogue
  over every store) cut the maps 2.35 -> 0.60 ms at 512 tokens but raised
  the matrix kernels 27.85 -> 30.95 and was not taken. V100 card 1 (CUDA,
  sm_70), Llama-3.2-1B-Instruct Q4_K_M, JITLLM_CUDA_KERNEL_TIMING, the
  arms a process each, alternating, the base twice: the composed
  prompt's kernels 36.9 -> 34.6 ms at 512 tokens and 168.5 -> 154.3 ms at
  2048 (the base's A/A within 0.3%): attention 2.92 -> 2.42 and 33.4 ->
  26.7 ms (main's 1.5 and 17.0), maps 2.34 -> 0.68 and 9.37 -> 2.67 ms
  (81 -> 33 launches a 512-row chunk), halves 1.88 -> 1.56 and 7.49 ->
  6.24 ms; the projections, 27.9 and 111.9 ms, are what is left. The
  serving gate (TestComposedBlocksServe) now prompts a paragraph of two
  64-row chunks and several 32-position blocks; with that alone a rescale
  forced to 1 still passed on Llama-3.2-1B, because the float reference
  every arm is held to was formed through the same Flash pass and moved
  with it (the tier's distance from it rose 0.0147 -> 0.0629). The float
  reference (Config.ComposedFloat) now states no matrix tile, and the
  forced rescale reads 0.0654 at its worst against the band's 0.0584 and
  fails (the arms as built: 0.00577); Recompute reading one element over
  reads 7.76, and Same merging kernels by their space alone 7.16. Against
  main's prompt on the same card, one binary: kernels 34.5 / 71.6 / 154.5
  ms at 512 / 1024 / 2048 tokens against main's 22.3 / 46.3 / 99.8 (1.55
  of main's at each); the server's time to first token, one server on the
  card at a time, each arm an httpbench A/A (every ratio within 0.3% of
  1, IQR/median at most 0.013) taken twice: 36.2 / 74.0 / 157.8 ms against
  main's 23.8 / 48.6 / 102.3 (1.52 / 1.52 / 1.54). The prompt is still
  GPU-bound; the projections are the gap.
- **The streams' expansion and collapse composed, a CPU form in minutes,
  and a submission that panics fails its step.** A model of predicted
  streams states its input rows' expansion (stream k the projection of
  stream 0 matched to its magnitude) and its head's collapse (the mean of
  stream 0 and each projected stream matched to it) as graphs of generic
  ops (a square, a row sum, a floor, two roots and a quotient for the
  magnitude); the CPU device forms them once at placement, a row's and a
  chunk's, each stream bound at its offset in the stream-major residual,
  the collapse writing apart before it goes over stream 0, and counts
  every row (CPUStreamSteps). On synth-gemma3n the composed CPU reads
  2.75e-07 (decode) and 2.94e-07 (a prompt of 8 and its decode) from main
  with every expansion and collapse row served, and streams 1 and 2
  swapped where they are bound read 0.093. A card's head now reads the
  collapsed row the host hands it (the CPU head's graph), so a stream
  model's head serves on the card, and a call of the head alone writes no
  rotary tables (it read two a step where the gate counts one). The input
  rows stay the host's on a card: with the head there the serving gate
  demands the card gather them, which needs the per-layer table and the
  expansion on the card, so gemma3n is not in servedModels yet. Forming
  was the register binding's: free scanned every value's last read at
  each entry of the sequence, vectorized() rescanned the nodes at each
  question, and each operand walk copied its node to the heap (a quarter
  of the time was the collector). Indexed and read in place, Llama-3.2-1B's
  CPU placement takes 17.1 s where it took 55.6 s (laptop), and gemma-3n
  E2B's 30 blocks 4 min 56 s where they took 24 min 8 s (V100 box's E5-2680
  v4, the same probe). The CPU page fault is each block's: it binds a
  table whose first page is its own arena's last slot where that arena
  holds more than two, so on E2B, whose sliding layers keep one ring page,
  it reaches the global layers (median 1.77 of main's spread, out of the
  band) where the bound of every arena disabled it, and on Gemma 4 E2B
  (median 17.0; its whole CPU gate 200 s on the V100 box, where its
  placement alone took 12 min). Gemma 3n E2B stays
  out of the CPU's served list: its clean arm is in the band (median
  0.634, worst 9.1e-4 where main's control reaches 1.5e-3; the streams
  swapped 1462, the head's row 2388), but the key-row violation stays in
  it: over the gate's 80 steps its attention is its row's own key's.
  Moving the first key row by one or to the middle of the keys reads
  8.3e-4 and 9.3e-4 from main (medians 0.58 and 0.60 of main's spread),
  and attending to the row's own key alone in every block moves the
  composed logits 6.5e-4 from the clean arm. A
  panic inside a submission is recovered in the backend's session and
  returned as the step's error, its ticket given back: injected under
  jitllmfault (JITLLM_DEVICE_PANIC_AT) on the prompt's and a decode
  step's submission, the steps end and State.Close returns on CUDA and
  Vulkan, where without the recovery Close waited in quiesce until the
  test's timeout.
- Gemma 3's SigLIP on the cards was the call's live elements wrapping 32
  bits, not the attention's partitions: CallRows held a flattened space's
  threads to ceil(rows*space0/live), and 4096 rows of 1152 times 4096 rows
  is past 2^32, so the attention's merge wrote only the rows before 455
  (before 324 at 2100 rows) and left the rest as the scratch held them.
  The leading axis's whole rows now come apart from its remainder.
  jit.TestComposedGPUAttendBidirRows holds a bidirectional run of 1024,
  2100 and 4096 rows at head 72 (and 2100 at 64) to the definition on
  ptx and spirv, in the general form and the single-sequence one: the
  running split serves head 72, the flash form head 64, both exact on rows
  attending both ways (1e-13 on ptx, 2.8e-7 on spirv); before the fix 42
  of 49 sampled rows were NaN. With it Gemma 3 is served whole on cuda:0
  and vulkan:0 (4.5e-5 and 1.2e-7 after one block, 2.4e-3 and 1.3e-3 for
  the encode), and ComposedTower is in ComposedStep. Janus on cuda:0 reads
  1.72e-2 for the encode composed and 1.71e-2 on main's own kernels.
  A picture of more rows than a CPU call (64) runs first only the codes
  through the block's last history write, over the residual itself, and
  then the whole block; the [rows, heads, keys] scores of a CPU form at
  the picture's 4096 rows would be a gigabyte, so the call stays at 64.
  Gemma 3 on a V100 host's CPU: 27/27 served, 1.3e-6 after one block,
  1.2e-3 for the encode, no warm allocation, causal rows 9.0e-2. The
  projectors and the entry rows are still main's generated kernels.
- Two mixture refusals lifted. A dense MLP beside a mixture with a
  per-expert scale (Gemma 4 26B's block) is now stated from the roles the
  block stores, not from a flag. The router reads the residual through
  its own norm, and so do the experts. A per-expert scale is read at each
  pick (an input-less slice and a take). The shared expert and the routed
  sum are each post-normed before they are added. Composed, synth-gemma4-moe
  serves 54 of 54 block steps on cuda:0 at 6.4e-11 and on vulkan at 1.9e-11,
  and the CPU device is 3.8e-08 from main. Its five features each move the
  composed logits by 0.82 to 1.38. Its prompt on the CPU device is declined
  at every block by name: experts in pages at a step of several rows.
  Phi-3.5-MoE's sparsemixer is stated in the container (the fifteenth
  config tail, RouterWeight). The weight is a row over the logits with the
  earlier picks removed and the pick's own logit, read at the pick, and the
  score selected by is the raw logits. The converter's expression matches
  the oracle to 1e-6, and each of the oracle's four violations lands
  0.495 to 0.731 from it. Composed, synth-phimoe is 2.7e-10 from main on
  the CPU device over 40 steps (paged, and a wrong expert page moves it by
  2.27). Its composed decode reads 4.8e-13 from the host on cuda:0 and
  5.7e-13 on vulkan, every block step composed. Its router violations reach
  the composed arm through the restated container.
- Every tower family the front end states runs composed in its own device
  gate: TestInternVLTowerOnDevice, TestJanusOnDeviceMatchesHost,
  TestPixtralOnDeviceMatchesHost, TestLlama4OnDeviceMatchesHost,
  TestPhi4TowerOnDeviceMatchesHost, TestRotaryTowerOnDeviceMatchesHost (the
  Qwen-VL fixtures, GLM-4.1V), TestKimiVLTowerOnDeviceMatchesHost,
  TestGemma4OnDeviceMatchesHost, TestMiniCPMVOnDeviceMatchesHost and
  LFM2-VL's take a `/composed` arm per device (towerArms: ComposedStep
  against Composed 0), each encode's composed tower block runs counted and
  no decline allowed; TestQwenTowerMatchesTransformers and
  TestGemma4TowerMatchesTransformers take a `composed-cpu` arm held to
  transformers at the family's bound, every violation restated to the
  composer (restateTower: the front end states the graphs again from the
  configuration as it stands). Two statements were missing: MoonViT's
  rotary takes its axes pair by pair in a cycle (graph.Attrs.Cycle and
  CycleAxes, each cycle at its first pair's frequency; the composer builds
  no table over one and refuses it by name), and Gemma 4's clipped linears
  are OpClamp against the block's stored pairs (jlm.RoleVClamp, bound on
  the CPU device and the cards; its kernel gate exact against
  oracle.Clamp32 at ragged widths, CPU 240 cases, ptx and spirv 135 each,
  the table read one word over seen). A cross-attention is OpAttend under
  Bidir over a history its graph does not write (Builder.CrossAttend):
  held to the definition from 1 query over 1 key to 64 over 300 at head
  widths 128, 72 and 48, CPU 15 of 15, ptx and spirv 30 of 30 at up to
  5.6e-14, a span cut by a key either end seen. TestComposedTowerOnDevice
  runs all 17 stated towers on cuda:0 and vulkan of a V100, each held to
  its family gate's whole-encode bound (1e-2 where that gate holds the
  device to transformers on its own picture) and set beside the tier's
  own kernels on the same picture: Qwen2.5-VL 5.04e-3 composed against
  5.04e-3, Janus 1.718e-2 against 1.714e-2 (its bound 2e-2), llava 2.4e-3
  against 8.8e-4; after one block the composer's binary16 matrix forms
  take main's binary16 bound (Llama 4 1.4e-4 under 5e-4, 6 composed matrix
  launches). The composed CPU's warm-encode check counts the engine's
  allocations apart from the runtime's per-P caches (countAllocs, the
  decode gate's): a process-wide count read 0 to 23 for the same Qwen3-VL
  encode from one run to the next. Not stated yet: Gemma 3n's MobileNet-V5
  blocks (a 2-D depthwise convolution over a picture's grid needs the CPU
  lowering's integer division of a row by the grid width, packed, which it
  has not), and the resampler itself, which is a projector.
- The composed CPU decode runs inside main's fused pool region
  (`sched.Pool.Fuse`, `State.fuseRun` before `cpuStep` in place of an
  `unfuse`): Llama-3.2-1B's token is one region, not 129. Consecutive pool
  kernels that share no buffer one of them writes (the call site's binding:
  the residual stream, an own buffer, a table) run in one phase, their calls
  one space (`cpuBlockForm.joinPhases`, decode forms): q, k and v join, gate
  and up join, and a token runs 81 phases against main's 82. A kernel of one
  call stays on the caller, in no phase (joining those took a token to 177
  phases). Without VNNI, where no register holds the byte dots' ones, the
  prologue stores them below the shift count's word and `vpmaddwd` reads
  them from there: the Q4_K kernel's `vpcmpeqw` went from 128 to 1 (the dead
  store before each broadcast was already gone). `Workers` takes, among cuts
  that leave the busiest worker the same rows, the most up to four calls a
  worker: the head went from 12 calls of 10688 rows (one a worker) to 48.
  Violations: joining dependent kernels makes `TestComposedCPUBlocksServe`
  return NaN, and ones read from an unwritten slot fail
  `TestComposedCPUMatVec` at NMSE 1e8. Rows on an idle E5-2680 v4
  (`TestComposedCPUAB`, node 0's 14 cores, n=20, main's A/A 0.99-1.02 at
  IQR/median 0.018-0.06): base 0.6145-0.6252 in six passes; these changes
  0.6496, 0.6715 and 0.6567. That is unresolved, because the composed arm's
  own IQR/median is 0.12-0.17 on the idle box. Yet one model alone in a
  process (`TestComposedCPUProfile`, which now prints a token's time
  spread) holds a composed token to p10 27.6, median 28.2 and p90 29.7 ms
  (main 18.0, 19.4 and 20.5). So the spread belongs to the interleaved
  harness, with both models in one process, not to the composed token.
  Not kept: `Ahead`'s distance (2 trips
  is best on a 32768x4096 weight: one core 2.2-2.9 GB/s, 12 threads 20-24
  GB/s at 0, 2, 4, 8, 12 and 16 trips) and claiming the calls that share a
  line together (a call's 16 rows already read whole 64-byte lines), and
  calls of at least 64 rows (`Workers{Rows: 64}`: 11.5-27 GB/s against
  16 rows' 11-30 at decode's shapes on 14 cores). What
  limits the matvecs is the walk. A call takes its 16 rows down all of K, a
  line from each word's plane, every line a new DRAM row. Main's fused
  kernel (`cpu.EmitPackedMatVecFusedAhead`) reads the same planes at about
  41 GB/s: it takes each scale sub-block over all of a call's rows, a few
  sequential streams, and adds the scaled sum into `Out` once a sub-block.
- **The matrix kernel's codes decoded two a word, its index bounded, its
  staged words read sixteen bytes a load: 1.22/1.13/0.95 of GemmVolta.**
  On the V100 (CUDA, sm_70, 512 rows, BenchmarkComposedMatrix, q/o Q4_K,
  gate/up Q4_K, down Q6_K, median of per-round ratios, each change A/B
  interleaved twice against its parent, an A/A of the base within 0.01)
  from 1.34-1.35/1.26-1.27/1.11-1.12: `HalfPairs` (after the profile's last
  pass, on a device that states `Device.HalfPairs`, PTX alone) decodes two
  stage words whose four elements are one scale times a code plus one bias,
  the second word's codes the first's bit fields sixteen bits higher in the
  same words: each field masked for two elements at once into 1024's
  binary16 exponent, less 1024 (and the code's offset where the bias is the
  scale times it, Q6_K's 32), one fma.f16x2 by the scale and the bias, the
  pairs dealt back into the words' order -- 615 -> 591 instructions a
  warp-trip on Q4_K, 603 -> 545 on Q6_K, 1.27-1.28/1.22/1.09-1.10, worst NMSE
  2.3e-7 against the gate's 2e-6 (the earlier pair decode masked each code
  alone and kept 603). Affine's `bounded` (matrix kernels) drops a term a
  shift or mask cannot see by its bound, not only a constant: (x+y)>>n is
  x>>n for x a multiple of 2^n and y at most 2^n-1, (x+y)&(2^n-1) is
  y&(2^n-1), a mask over a value its bound keeps under it is the value, and
  (a*C)>>n is a*(C>>n): 541/494 instructions, neutral alone
  (1.25-1.29/1.20/1.11-1.12). The GPU lowering reads four aligned 32-bit
  loads of a scratch buffer at one index (equal by structure, so a load
  carried a trip ahead groups too) as one ld.global.v4 on a device that
  states `Device.VectorLoads` (PTX): a scratch binding is an allocation of
  its own (the tier's `Alloc`), never bound at an offset. The staged
  binary16 words of the row factor go from 8 to 2 loads a thread a trip and
  the probe from 1.27-1.29/1.20-1.21/1.11 to 1.22-1.23/1.13/0.95: Q6_K's
  down projection is faster than GemmVolta. Instruction counts alone do not
  predict q/o (two of its three changes cut instructions and moved it 0 to
  +1%); the memory instructions do. The V100's counters are not readable
  (nvprof and ncu: ERR_NVGPUCTRPERM; nvprof's trace sees no kernel the goffi
  launches), so stalls are inferred from SASS (`nvdisasm`) and timing.
  Measured and not kept: pairing each four elements' codes crosswise (the
  first with the third, the halves kernel writing the row's words the same
  way) ends the re-pairing (576/528 instructions) and reads 1.32-1.33/
  1.15-1.16/1.12-1.13 -- the halves kernel's own time unchanged (31-36 us,
  timed alone now), the matrix kernel slower on the two shapes of 128
  workgroups; tiles over the paired decode and bounded index: 128x128x64 2x4
  (254 registers, no spill) 1.29-1.30/1.20/1.02 and with the vector loads
  1.22-1.23/1.11/1.00; 128x128x32 over 4x2 warps 1.24/1.17/1.02 and with
  them 1.16/1.11/1.06 (128x128x64 over 4x2: 1.79/1.78/1.29);
  128x64x32 2x2 two a unit 1.46/1.40/1.20; 128x64x64 1.98/2.00/1.84 (spills).
  No tile is best on every shape: q/o reads best over 4x2 warps, down over
  2x4 at depth 32, gate/up at depth 64 -- a choice per shape for a tuner,
  not a profile constant. PM3 in one pass (servers, main, the base 88d345c6
  and the head one after another, each an A/A within 0.4%, composed prompt rate
  over main's at 512/1024/2048): Llama-3.2-1B (card 7) 0.483/0.409/0.291 ->
  0.510/0.428/0.300, Llama-3.1-8B (card 7) 0.544/0.444/0.321 ->
  0.586/0.472/0.335, gpt-oss-20b (cards 6 and 7) 0.401/0.499/0.492 ->
  0.420/0.519/0.508 (MXFP4's codes are a table's, which the pair decode
  does not take: the vector loads and the bounded index), Qwen3-30B-A3B
  (cards 6 and 7) 0.415/0.348/0.251 -> 0.442/0.364/0.258. Each 16-token
  answer of the head is the base's but Llama-3.1-8B's, which parts from
  main's and the base's after ten tokens (". Question: what did the
  committee ask each" against ". The committee reviewed the quarterly"):
  the binary16 scale the pair decode rounds to moves a near tie there
  (TestComposedGPUMatrix and TestComposedGPUExperts at worst 2.3e-7,
  TestComposedBlocksServe's bands, all pass).
- **A linear block's prompt rows run whole on the composed CPU (D17), and
  the CPU attention's flipped bound (D10).** The scan (PrimScan) had no CPU
  form, so a linear block stepped its prompt a row at a time on its form of
  one row. A device whose element is a thread with memory beside it
  (Profile.ScanLoops, the CPU) forms the scan as generateScanLoops: an
  element a state row, its components loops (a sum over them a fold of a
  loop of its own, the update's store in its loop), the step's rows one
  loop over the run descriptor's order list, the state kept where it is
  written (a run's first row reading its start there, Scan.From), the calls
  the state rows (Workers deals a scan's rows a call each; Code.Scan runs
  every call whatever the call's rows). The engine runs a linear block's
  rows on its rows' form where the step's rows are its own State's and each
  sequence's stand in order (cpuRecWhole), a call ending at each restore
  point, the run descriptor built per call (cpuRecDesc), the convolution's
  windows written apart and copied over the runs' slots (its runs past the
  step's write a window again from the window read). Counted:
  CPURecSteps. On a V100 host's CPU (node 0), prompts of 5, 33 and 602
  rows and four decode steps, every linear block's step of several rows
  whole and none a row at a time, held to main within main's spread
  (TestComposedCPURecurrentPromptsServe): Qwen3.5-0.8B 0.50-0.67 of the
  spread at the median, mamba-130m 0.12-3.6 (worst 0.037 against four
  times the control's 0.0093), LFM2-350M 0.95-1.05, granite-4.0-h-350m
  0.41-0.99, Falcon-H1-0.5B 1.18-1.32; the state dropped at each chunk
  and every block on the first's state read 9.4-2620 times. Faults the
  forms reached, fixed: x86 division by a constant
  whose round-up multiplier passes 2^31 (sign-extended), a general
  register's compare mask read as a vector, a call coordinate moved out of
  an index whose rest falls below zero. D10's second miscompute: Induction
  carries an iteration read as data as a sum, and Homes placed that sum's
  readers outside its loop, so the orderer emitted the loop early and a
  loop-invariant bound inside vector loop 0, unmade for the tail when the
  vector loop ran no trip; the sum is marked Running and its readers are
  the loop's, an unrolled copy's iteration is the lanes' plus its vectors,
  and the rule that a difference of small words is small is gone
  (TestComposedCPUAttend holds; with the readers outside, 0.198).
- **Metal serves the composer's blocks, its matrix tile among them.**
  Metal states maxThreadgroupMemoryLength (32 KiB on the M4) as
  jit.Device.Shared, and the workgroup geometry leaves the attention plane
  what is left beside the combines, Bind's per-trip sums and the running
  partials: the stage kernels had declared 34816 to 39424 bytes, which the
  driver refused, and a kernel past the device is now refused by name where
  it lowers (TestFormsHoldTheDevicesWorkgroupMemory). msl lowers a tile past
  8x8 as a struct of 8x8 simdgroup matrices, so LowersTile's 16x16x16 probe
  compiles and the flash and matrix passes form on Metal; a load or store
  whose type is not its buffer's moves the bits (as_type), where MSL had
  converted a float read of a uint threadgroup word by value (the flash
  form's rows at NMSE 4.6e21), and the matrix pass halves its rows until the
  stage fits (38912 bytes at 128 rows). Two faults every backend had: the
  mixed-sequence prompt form is formed at placement wherever a ragged step
  runs (the scheduler's batch formed it after the placement on all eight
  census models, CUDA included), and a form's scratch is sized by the widest
  code that binds it (sized by the first, a later code overran it: with the
  second form's instance beside it, stories15M's composed arms sat 7.98 from
  the tier against 0.093). On the M4, Metal: TestComposedBlocksServe passes
  on Llama-3.2-1B, Qwen3-0.6B, SmolLM2-360M, stories15M, gemma-3-1b,
  Qwen3-MOE-4x0.6B, synth-gptoss and Qwen3.5-0.8B, TestPlacementComposesOnce
  on all eight models (one card), the composed TestDecodeDoesNotAllocate arms
  on Llama-3.2-1B and Qwen3.5-0.8B, TestComposedExpertPagesMoveToTheCPU, the
  recurrent gates, and TestResetClearsTheRecurrentSummary, which had been
  linux-only and CPU-only and now runs on every test device.
- **A composed prompt's attention runs on every core.** A CPU kernel that
  wrote a scratch ran its calls one after another, so a prompt's scores and
  softmax and each projection's int8 form ran on one core while the
  projections ran on all of them. Decomposed at 512 tokens on the E5-2680 v4's
  node 1 (14 cores, WithProfile, the composed kernels' own time):
  attention's dots were 22% of Llama-3.2-1B Q4_K_M's prompt and 42% of
  Qwen3-0.6B Q8_0's (14.8 ms a launch, as long as one thread takes), the
  weighted values another 9% of Qwen's. A kernel's calls now run on the
  pool where every address of the scratch they write moves along every
  call axis past the axes inside it and the code reaches no more than
  the calls' slices (engine/model/cpuserve.go, callsOwnScratch); other
  scratches keep their calls in order. The weighted values (d outer, the
  history's positions inner) ran one vector of a head a trip, each
  position's page found again for every 8 lanes; Jam now strip-mines 4
  vectors a trip where the inner loop is the kernel's positions loop
  (Jam.Positions): the kernel alone at Qwen3's shape, 64 rows after 448
  positions, 60.5 -> 25.0 ms on one thread. And a RowJam form the registers
  refuse is formed again at half the rows a call with its lanes kept
  (Kernel.Unjam), where before it also lost Jam's lanes (Q8_0 at 4 rows a
  call fell to a scalar 567-byte kernel at 1.9 GMAC/s; it now keeps the
  2-row form at 18). Speed, interleaved, n=20, the same node 1, prefill at
  512: Llama-3.2-1B 0.1548 -> 0.2365 of main (57.5 -> 90.9 tok/s against
  main's 366-384; A/A 1.0106 and 0.9908, IQR/median 0.043 and 0.030);
  Qwen3-0.6B 0.1875 -> 0.5818 (64.7 -> 229.2 tok/s against 338-394; A/A
  0.9913 and 0.9805, IQR/median 0.050 and 0.018). After it the prompt is
  its projections: 72% of Llama's composed time in its six Q4_K/Q6_K rows
  matvecs, 52% of Qwen's in its five largest. The rows form runs about 0.5 instructions a
  multiply-add where main's GEMM runs about 0.14: a call's two rows each
  load and decode the weight's words and their sub-block scales again
  (Jam copies a row's tree per store, and weight loads are not merged;
  merging them is refused by the registers, since Order writes one row's
  super-block whole before the next), and the activation-side combine
  spills its indices. RowJam at 4 rows reads +17% on Q4_K and +12% on Q6_K
  alone but forms in minutes, and is not taken. Gates on node 1: the
  serving gate on Qwen3-0.6B and Llama-3.2-1B, TestComposedCPURowsServe and
  TestComposedCPURowsDoNotAllocate on Qwen3-0.6B, the prefill gate's
  composed-cpu arm, TestCPUComposesOnce on Qwen3-0.6B and the composed
  decode allocation arm on Llama-3.2-1B; with two calls made to write one
  slice, TestComposedCPURowsServe leaves the band at a median 2361 times
  main's spread and the prefill gate fails, where the serving gate, decode
  alone, does not see it.
- The BERT-family encoders (bert, nomic-bert) are the composer's on the CPU
  device and the cards. The front end states a non-causal block as a causal
  one with its attention both ways (graph.Attrs.Bidir) and a post-norm
  block's norms after each residual add (the stored attn_out_norm and
  layer_out_norm); the encoder's loader states what its blocks are (their
  norms LayerNorms, no rotary where the positions are a learned table). A
  non-causal block whose rotary the model does not hold (ModernBERT's two
  bases) or whose mixer is recurrent (d1-omni's centred convolutions) stays
  UncoveredNonCausal. On the CPU device (WithComposedCPU) the Embedder's
  segment State runs each block over the sequence's rows as one run, its
  history the segment's arena (ncArena, shared with the tower's); on a card
  the tier serves the segment's non-causal call (tier.ComposedTower,
  towerserve.go) as a tower's. On a V100 host's CPU, held to main's
  encoder: all-minilm-l6, mxbai-embed-large, nomic-embed-text and
  snowflake-arctic-embed-22m serve every block of every text (36, 144, 84,
  36 runs, no decline), 3.2e-9..1.4e-7 NMSE after one block, worst cosine
  0.99999987 to main's vector, a warm embed 0 allocations against main's
  6-24; the rows made causal read 2.8e-2..7.4e-2. On cuda:0 and vulkan:2
  every block is served: 1.3e-8..3.8e-8 (CUDA) and 1.4e-13..1.7e-13 (Vulkan)
  after one block, worst cosine 0.99999963 for the whole encoder.
- An embedding model's readout is ops (Model.poolGraph): the output norm
  where the model stores one, graph.OpPool (CLS, last or mean over the live
  rows, the count the positions table's first word), the dense heads where
  stored, and the unit length as an RMS norm of no weight times
  1/sqrt(width) with main's 1e-12 floor. The CPU device serves it
  (cpupool.go), composed once at the segment's or the State's rows: an
  encoder's after its blocks, a decoder's (qwen3-embedding, EmbeddingGemma)
  from the chunk holding the row it reads; a mean over several chunks is
  declined by name (DeclinePrefill) and pooled by main's. A non-causal
  decoder's chunk (EmbeddingGemma's) now runs its blocks on the composed CPU
  as one run both ways, the symmetric window the graph's. On the V100 host
  CPU every readout of every text is served: the encoders at worst cosine
  0.99999987 to main's; EmbeddingGemma, its 168 block runs composed too,
  0.99999997 through its two dense heads (its rows made causal 0.57);
  qwen3-embedding 0.99999995 from its F16 file and 0.99951 from its Q8_0
  (the composed blocks' int8 activations; the reference gate's 0.999 for a
  quantized file holds against llama.cpp); the other pooling reads
  0.19..0.95. The
  readout stays on the CPU device where the blocks run on a card; the
  cards serve no pooled part yet, and the input rows (BERT's token types,
  UncoveredEmbedRows) stay main's.
- Kimi-K3's blocks are composed. Its residual attention is stated as a
  bank of checkpoint streams. The running residual is stream 0 and each
  checkpoint a stream after it, each bound as a view of the step's
  residual. Each sublayer reads a softmax mix: the banked rows and the
  running residual, each scored by its stored vector's dot with the row
  RMS-normed with no weight. A checkpoint block banks its input first and
  restarts the residual from its attention's output. Streams a block does
  not write are its outputs as they came, so no kernel reads and writes one
  stream. The latent mixture's experts run at the routed width: routed
  down, the weighted sum normed and routed up, then the shared expert
  added. All three are decided from the stored roles. The tier now binds a
  linear block's own output gate before a latent attention's, and holds
  K3's recurrent state for the composer, where it had excluded K3 by
  name. Composed, synth-kimik3 serves 126 of 126 decode block steps, at
  2.8e-10 from the host on cuda:0 and 1.0e-10 on vulkan. The ragged
  fixture reads 4.3e-10 and 2.2e-10. Eight pieces taken out of the
  statement move the logits by 0.0039 to 2.7: the bank, the raw scores,
  the restart, the latent norm, the MLA gate, the decay bound, the latent
  norm's epsilon, and the activation. The head's mix and the input rows
  stay the host's, refused by name (UncoveredStreams).
- A tower's entry rows and its projector are graphs of generic ops
  (towerEntryOf: the patch gather, the patch matrix and its bias, the class
  row, each row's position-table row, the norm before block 0;
  towerProjOf: the patches regrouped by the request's row names, the
  pool's mean and RMSNorm, the merged row's LayerNorm, the matrices and the
  activation, the window order put back), decided by the tensors the
  container holds and declined by name where it holds one no graph binds.
  The CPU forms them at the vision State's placement, a card runs them as
  tier.ComposedRows (picrows.go). On the V100 host the CPU serves SmolVLM,
  llava, Qwen2-VL and Qwen2.5-VL whole (entry, every block, projector:
  2.6e-7, 7.2e-8, 5.2e-8 and 7.3e-14 after block 0; 1.1e-3, 1.1e-3,
  1.1e-4, 1.2e-4 for the encode), cuda:0 and vulkan:0 those four and
  Gemma 3 (encode 1.1e-3 to 5.8e-3); the projector alone is 3.7e-13 to
  1.1e-8 from main's, and the composed arms of
  TestProjectorMatchesTransformers read 6.2e-5 (Gemma 3) and 6.3e-5
  (InternVL) against transformers, as main's do. Each violation is seen: a
  patch row gathered from the next patch (0.08 to 1.4), the regroup's slots
  swapped, the pool's corner, the window order left in place. Three CPU
  lowering faults stood behind the patch gather: no division by a register
  (DIV now, RAX and RDX kept below the red zone), a remainder by a
  constant with no third register, and a float converted from a general
  register written to lane 0 alone, so a row's index compared in every
  lane masked the others with garbage. The row masks compare the row's
  index as an integer: the Splats pass broadcasts a general register's
  compare to the vectors a select reads it in, and the patch and class-row
  gates read the unmasked row (0.25 for -0.5) without it. A picture of more
  rows than a CPU call now runs the codes through the last history write
  once over every row, the values the rest read kept per call, so no code
  runs twice (no first pass: 0.50 to 1.08 from the clean encode).
- **The composer's CPU lowering emits for the SSE tier.** A CPU device
  whose stated ISA is short of AVX2, FMA and F16C forms by the profile's
  Narrow passes (jit/profile.go: Jam and Vectorize at four lanes, no FMA,
  Halves keeping no NaN payload; an f32 cache forms without Halves, as on
  AVX2, Registry.WithoutHalves dropping it from both) and is lowered in the
  legacy encodings through jit/x86's SSE ops (jit/lowercpu_sse.go): SSE4.1
  and SSSE3 at most, a product and a sum where AVX2 fuses them, binary16
  widened and narrowed in integer ops and one float multiply or add, a
  shift by a count a lane done a lane at a time in general registers, a
  memory operand loaded first (legacy SSE faults unaligned), and a scratch
  vector the kernel's registers leave none for borrowed below the red zone
  and put back. jit.HostCPULowers takes the SSE tier, so an SSE host's
  blocks are served where every one was declined by DeclineISA. The
  VEX-leak gate is one package now (internal/ssegate, main's jit/cpu gate
  moved there), and TestComposedCPUGatesOnSSE runs the 20 composer CPU
  kernel gates on the SSE tier and reads every kernel they mapped back
  through objdump: on the laptop (forced) and natively on an Atom C3558,
  MatVec 180/180 formed and right over every packed type, no VEX; each new
  form against a violation (the widened sign dropped, the narrowed sign
  dropped, every lane shifted by lane 0's count, the byte dot unsigned, its
  zero addend's sums from the wrong register, the fma's addend dropped, the
  aliased operand not put back, the borrowed registers put back from the
  wrong slots) fails a gate. On the Atoms, natively, the composed build:
  TestComposedCPUBlocksServe serves every block step of all eleven
  CPU-served models, stories15M, Qwen3-0.6B, Llama-3.2-1B, Qwen3.5-0.8B,
  Ministral-3-3B, HunyuanOCR, mamba-130m, LFM2-350M, granite-4.0-h-350m,
  Falcon-H1-0.5B and Gemma 4 E2B, at 0.364, 0.451, 0.943, 0.526, 0.990,
  0.729, 0.403, 0.925, 0.403, 0.980 and 0.909 of main's spread from main
  (argmax 76/80 to 80/80); TestEveryModelRunsSSE serves the composed arm on
  all ten of the host's models, against main's SSE kernels 2.4e-6..7.9e-5
  max|dlogit| on the small ones and 0.80 on Qwen3-0.6B, inside the 0.94
  band; TestGreedyMatchesLlamaCpp and its SSE twin agree with llama.cpp
  b10825 on Qwen3-0.6B 19/20 teacher-forced (one 0.021 tie) and on the
  fixtures 20/20; TestBatchMatchesForward's and TestPrefillMatchesForward's
  composed arms serve every block step (NMSE 3.4e-14, 5.6e-4);
  TestCPUComposesOnce holds the census on stories15M, Qwen3-0.6B (prompts
  to 2000), Ministral-3-3B and seven fixtures (its relocation arm needs a
  card); the composed host decode allocates nothing over 64 tokens on
  Llama-3.2-1B, Qwen3-0.6B, Qwen3.5-0.8B, mamba-130m, granite-4.0-h-350m
  and Falcon-H1-0.5B. Decode, Qwen3-0.6B Q8_0 on the second Atom (4 cores,
  a foreign test on one), dev/bench TestComposedCPUAB, two passes: A/A
  0.9948 and 0.9959 (IQR 1.7%, 1.6%), composed 0.660x and 0.664x main's SSE
  tier (IQR 4.3%, 2.8%), main 5.2 tok/s = 3.3 GB/s = 29% of the 11.4 GB/s
  wall, composed 3.5 tok/s (before the fused decode region and the
  zero-addend dot, 0.488x). The int8 dot's step is a broadcast of
  four activation bytes, a load, a sign, a magnitude, two multiply-adds and
  an add a four-row vector with a register spill each step, and a binary16
  scale widened in software a block: speed is the shared passes' work.
- Five miscompiles in the composer's passes, each a wrong value or a
  fault with no error. The x86 register binding gave every sum a vector
  register, even one gprs had put in a general register. The lowering then
  counted in the general register with the vector's number, RAX for the
  first: a count of words read down a column faulted on its address
  (jit.TestComposedCPUWordSums). Words a jammed loop loads across lanes were
  not vector words, so a jammed integer sum counted lane 0 alone. A sum read
  in another sum's step had its readers placed after its loop (Homes), and
  Order ran its step before them, unless the pair matched the carried-choice
  pattern. Unsigned's sign flips and a broadcast mask break that pattern,
  so an integer maximum carrying a float kept place 0. A float of a known
  word (a loop of one trip's iteration) had no register. Induction stepped
  a vector loop's lanes by one vector a trip where an unrolled trip covers
  several. With the lanes remade for a value that is also an address
  (3acc904c's), that still left the top-k selection's prefix counts wrong
  in 42984 to 134135 of a row's elements at blocks of 64 and 128
  (jit.TestComposedCPULanesAsData). Each fix, reverted alone, fails its
  gate.
- The top-k selection counts in floats: each round's counts and each
  element's count within its block are sums of ones and 2^16s, exact below
  2^24, and are converted to words after. Within a block the element's
  position is compared with each other's, and the clamp of a block past the
  row's end is kept only where the row is not whole blocks. The CPU now
  vectorizes the within count over the block, four vectors a trip. One
  call at v151936, k 40 and 256, laptop, codes mapped each call: 13-23 ms
  against the integer form's 70-90 ms in the same pass. TestComposedCPUSampleTopK
  and TestComposedGPUSampleTopK hold all 40 cases bit for bit, on the
  V100's ptx and spirv as well.
- **A prompt's jammed rows decode the weight once.** RowJam's two rows a
  call shared the weight's words before Periods, which unrolls a block's
  sub-blocks, copied each sum's step with a memo of its own: every row got
  its own loads and decode of the words and scales (WeightLoads, 64x2048 at
  8 rows a step, no VNNI: Q4_K 46592 words a launch at two rows a call, as
  many as at one). A period's steps now share one memo (Kernel.copyMemo),
  and Order steps the sums of a loop together where their steps are chains
  of one length that read a computed value alike (orderer.lockstep,
  orderer.together): each operand place made side by side, so a decoded
  weight vector is read by both rows' byte dots as soon as it is made and
  the registers hold the jam. Written one row after another, a row held its
  eight decoded vectors across its sub-block and the binder refused the jam
  outright on Q4_0, Q5_0, Q5_1, TQ2_0, TQ2_F32 and MXFP4 (one row a call,
  no lanes); those now run two rows a call, Jam's lanes kept (MXFP4 none
  either way), and form faster (Q4_0 162 -> 36 ms, Q4_K 1.9 s -> 0.34 s).
  A buffer read at several strides (the int8 form's scales and sums in one
  scratch) walks a copy of its address for each stride (strideGroups)
  instead of an index the general registers spill. Q4_K's rows kernel at
  K=2048: 4211 -> 2687 instructions, 555 -> 110 red-zone accesses; Q6_K
  1672 -> 1306; Q8_0 unchanged (it shared already); Q4_0 257 at one row a
  call -> 345 at two. The Xeon D-2123IT, one core, 64 rows of 2048x2048,
  interleaved, median of 5: Q4_K 9.62 -> 12.28 GMAC/s, Q6_K 7.38 -> 9.61,
  Q8_0 14.7 either way; a second pass beside a test run, Q4_0 12.06 ->
  17.62. Decode: the Q4_K kernel 2175 -> 1929 instructions, Q6_K and Q8_0
  byte-identical. Prompt at 512 on the E5-2680 v4's node 1
  (TestComposedCPUPrefillAB, n=20, A/A 1.0135 and 1.0003): Llama-3.2-1B
  Q4_K_M 0.2390 -> 0.3039 of main (92.8 -> 116.6 tok/s against main's
  385-389); Qwen3-0.6B Q8_0 0.5969 -> 0.5947 (its code unchanged). Tried
  and not taken: loading a period's scale word (Operand.Region 1) once a
  trip read Llama 0.3210 (against 0.2425 in the same pass) and Q4_K 13.38
  GMAC/s, but the registers refuse it on Q5_K, which then fell to one row a
  call with no lanes, or with a retry flag kept two rows but formed in 10.7
  s against 0.5. Four rows a call stays refused on Q4_K: a period's tail
  holds four sums, four accumulators and four resident constants, past
  sixteen vector registers. The prompt attention's dots still recompute a
  key's page address with every operand spilled: its general registers are
  all taken by the loop's cursors, and the key's address is not affine (a
  page table load), so no cursor walks it. Gates:
  TestCPURowJamReadsTheWeightOnce (two rows a call load half the words of
  one on eleven packed types; eight rows halve to two with Jam's lanes
  kept, and with Jam off that check parts; it failed the scale-word merge
  on Q5_K); Llama-3.2-1B and stories15M-q4_0 join cpuRowsModels, so
  TestComposedCPURowsServe and TestComposedCPURowsDoNotAllocate hold a
  K-quant's and a Q4_0 prompt.
- Streams read through a low-rank gate are stated as ops
  (Qwen3.8-Flash-Next's hyper-connections, `gatesStreams`): the streams side
  by side, each RMS-normed against its own row of the stored weight, a
  low-rank gate (down, 1/S, SiLU, up, sigmoid) over them, the mean of the
  gated streams the sublayer's input, and the output placed into each
  stream at 2 sigmoid(inject/S); the input rows are S copies of the
  embedding and the head reads its own gate's mix with no norm
  (`graph.Head.NoNorm`). A delta rule states its state's floats a sequence
  where the block carries a tail past them (`OpDeltaRule`'s Span, the
  n-gram history), so a linear block of a model with a tail composes. On the
  CPU device synth-qwen4exp serves block 0 at 4.6e-12 from main (ragged
  1.8e-12), its input rows' copies and the head's mix every row; the
  attention blocks wait on QSA (UncoveredSelect) and block 2 on the n-gram
  embedding (UncoveredNGram), both declined by name and counted. The gate
  read as its plain streams, the placement at weight one or without its
  two, the rank unscaled, the copies left empty and a stream read twice at
  the head move the logits 0.015 to 2.6. Kimi-K3's head mix over the bank
  is the same bankMix the blocks state, composed on the CPU at 4.8e-14
  (its blocks wait on a latent history on the CPU, DeclineLatent); its
  input rows need no expansion, since a block banks each checkpoint before
  any reads it. A softmax mix is the same over any order of its rows, so
  the CPU's stream violation at a collapse now reads a stream twice where
  it swapped two. DeepSeek V4's Sinkhorn-mixed streams stay refused
  (UncoveredStreams at the head and the input rows; every block is
  UncoveredSelect first).
- DeepSeek V4's input rows (copies of the embedding into every stream) and
  its head's collapse are composed on the CPU device: the streams side by
  side RMS-normed with no weight, the head's projection, its stored scale
  and base, sigmoid plus HCEps, each stream weighted. synth-deepseek4 reads
  0 from main over 32 steps with every row of both and every head step
  served (its blocks stay UncoveredSelect); the copies left empty, the
  weights unscaled and a stream read twice move it 2.4, 4.7e-4 and 0.11.
  The CPU's recurrent rows taken a row at a time bound every stream at the
  row's first (a stride of 0): a model of several streams with a linear
  block read stream 0 for all of them in a prompt. Each stream is bound at
  the chunk's stride now; synth-qwen4exp's prompt of 28 and its decode went
  from 2.16 to 3.3e-12 of main. The composed CPU decode of synth-qwen4exp,
  synth-kimik3 and synth-deepseek4 allocates nothing a warm token
  (testing.AllocsPerRun; main's qwen4exp decode makes 6).
- On the cards (V100: cuda:0 and vulkan), synth-qwen4exp's block 0 serves
  its decode steps composed: the tier binds the hyper-connections'
  projections and norms by role, and a linear block's state with the
  n-gram tail is the composer's when its graph states that tail
  (OpDeltaRule's Span equal to the pool's slot). 29 of 116 block steps
  served, the other 87 declined for no graph (QSA and the n-gram block),
  at 4.4e-11 on cuda:0 and 4.2e-11 on vulkan; the four hyper-connection
  violations move it 0.42 to 2.6. Kimi-K3's input rows are composed on
  the card: the bank's rows are the running residual's alone, written into
  the first stream of the stream-major residual; synth-kimik3 at 2.8e-10
  and 1.0e-10 (ragged 4.3e-10, 2.2e-10), every step's input rows served,
  the embedding doubled moving it 2.1. graph.Stated refuses a nil
  expression where it had dereferenced it: Gemma 3n's "gaussian on every
  block" violation leaves no plain activation for the per-layer input, and
  the device gate crashed there; the block is now refused by name
  (UncoveredAct) under that violation.
- A depthwise convolution over a picture's grid is an op: graph.OpConv
  with GridW, its k x k filter tap-major as the container stores it, at a
  stride where Stride is set. A stride-2 output is a grid of rows of its own
  (graph.Value.Rows; Verify holds every other op to one grid's rows, and
  the composer refuses any op but the convolution over such a value by
  name). The taps are summed a row of the square at most three a map, since
  a row of five over a vector loop held more than the registers. Taps that
  never land on the grid are dropped, and every tap is read by its index, so
  the output never takes the input's storage while other trips still read
  it. A Temp scratch with no element type read back as the wrong storage:
  NMSE 0.5 to 1.1. Words known at a grid of one row or column (a loop of one
  trip) are constants through Unsigned's sign flips. TestComposedCPUGridConv
  and TestComposedGPUGridConv (ptx and spirv on the V100) hold 16 grids to
  the oracle's float64 convolution: ragged sides and channels, one row,
  one column, 3x3 and 5x5, and stride 2 over even and odd sides. The CPU is
  within 1e-12 and the cards within 1e-10, and a transposed filter parts.
  MobileNet-V5's blocks are not stated yet: the tower front end declines
  them by name (UncoveredKind).
- The CPU serving gate's key-row violation drops the step's newest row as
  well as its first (past its second row), and the new key written a row
  early is a violation of its own (cpuFaultWrite). On the V100 box's CPU
  over 80 steps, each of the 11 served models leaves the band under both
  (median ratios: key row 4.8 to 3438, write early 1.5 to 3414). Gemma 3n
  E2B sees the page fault (median 1.77, worst 0.0111) and the write
  (worst 7.1e-3 against the control's 1.5e-3) but not the key row, even
  without its own row (median 0.655, worst 1.25e-3): its attention reads
  nearly the same value from any row. It stays out of cpuServedModels.
- Kimi Delta Attention's output gate has a composed violation: the sigmoid
  gate over the delta rule's normed output read as that output. On the
  cards synth-kimilinear's composed blocks read 1.9e-11 (cuda:0) and
  1.6e-11 (vulkan) from the host and 1.96 with the gate dropped;
  synth-kimik3's 2.8e-10 and 1.0e-10 move to 2.1 (ragged 2.12).
- Laya's ModernBERT trunk and the decision heads' blocks (Laya's,
  d1-omni's) are the composer's on the CPU device. The ModernBERT loader
  states its first-of-period global layer (Config.SWAGlobalFirst), its
  LayerNorms, NEOX and both rotaries; a decision head block is stated
  as ops (frontenc.go: pre-norm, biased, both ways, the container's head
  count, ReLU as max(x, 0)) at block NLayer+i. A non-causal segment's
  arena takes every block's keys whatever its kind: a causal window read
  off Config.SWA cut a head block's keys (a head past the model's layers
  read as a sliding one) and moved the fixture's logits by 0.045; the
  arena is sized from the widest block that keeps keys (d1-omni's first
  block is a convolution, which keeps none). V100 host CPU against the
  reference: the Laya fixture 1e-6 (main 1e-6), Laya 0.026724 (main
  0.026715, bound 0.035), every block of every encode served (80 and
  300 runs); rows made causal read 1.13 and 5.44. d1-omni: its two head
  blocks served each encode (20 runs), its LFM2 trunk declined by name
  (160 runs, no graph: UncoveredNonCausal), logits 2.84e-5 (bound 3e-4),
  its head's rows made causal 0.92. The type rows, the scorer and the
  cards for these blocks stay main's.
- A warm encode allocated on the V100 host (TestWarmImageEncodeDoesNotAllocate:
  pixtral 22 to 62, phi4 14 to 58, by run) and never on the Xeon. A
  MemProfileRate=64 profile put the 112-byte objects in sched.Pool.await:
  the runtime's waiter records (sudogs), two a parked worker for a select
  over quit and its start channel. The runtime empties its shared cache of
  them at every collection and a P's at a GOMAXPROCS change (AllocsPerRun
  sets one), so the first parks after either allocated them again, one per
  worker of 27. A worker now parks on its start channel alone (Close sends
  on it), and an encode ends with Pool.Rest, which parks every worker at
  once, so none parks later inside the next measured call: pixtral, phi4
  and SmolVLM read 0 at count 3, an allocation planted in Encode reads 1.
  A picture's entry and projector on a card (tier.ComposedRows) now form
  where the tower's blocks are placed (PrepRows, at the offer), not at the
  first encode: TestComposedTowerOnDevice holds Stats.FormedPart(ComposedRows)
  to 0, and reads 3 with the prewarm taken out. The front end states four
  more of the families' rows: Kimi-VL's projector (a LayerNorm over each
  patch before the regroup, where the container holds RoleVProjInNorm;
  1.1e-9 from main's, the norm dropped 0.195, the slots swapped 0.906),
  Gemma 4's (the pool's scale, the standardisation and an unweighted
  RMSNorm; a pool decided by a matrix reading one patch's width; a grid its
  scale does not divide keeps each axis's whole groups), Llama 4's (a
  third matrix after an activation) and its class row after the patches
  (OpTailRow, 1.4e-13), Gemma 4's table of two axes (OpPosAxes), and the
  position table resampled to the grid, bilinear or linear (OpPosResample:
  Qwen3-VL, Qwen3.5, HunyuanVL). Its taps' indices are now integer ratios
  in general registers and its sum separable, which the CPU's registers
  needed; the bicubic's sixteen indices still do not fit them, so Kimi-VL's
  and GLM-4.1V's entries are declined by name. arm64 divides by a register
  (UDIV), which the patch gather needed to form there.
- Latent attention composed on the CPU device. cpuWhy no longer declines a
  block whose history is latent: the CPU's form is asked with the cards'
  latent layout (jit.History{ValuesInKeys}), so the latent row is read once
  in the layer's arena, whose value span is its key span (mapKVArena), and
  the CPU binds the latent roles (wqa, wqb, wkva, wkb, wvb, their norms, and
  Kimi-K3's MLA output gate in a full block). DeepSeek-V3's grouped router
  found a binder gap: an outer loop's iteration read only by a store (the
  group rank written at its group) got no register on x86, so the block did
  not lower; the binder now counts a store's index and value as reads
  (jit.TestCPUStoreReadsItsIteration fails without it). TestComposedCPUBlocksServe
  serves every block step of synth-deepseek-dense, synth-deepseek,
  synth-deepseek-lite and synth-kimilinear at Q8_0 and Kimi-K3-0.40B Q8_0,
  at 0.000, 0.000, 0.000, 0.000 and 0.155 of main's spread at the median
  (argmax 80/80), and five latent violations leave the band (the latent
  norm dropped, the absorb banks swapped, the value read at the key's
  width, the rotary tail read from the latent's start, at median ratios
  12..1749; the rotation a pair short moves the int8 fixtures 1.9e-3, under
  main's own spread, and is held on the float arms). The float arms of
  TestMLAOnEveryDevice and TestNoPEMLAOnEveryDevice read 6e-14..9e-13
  against main's CPU, every violation 2e-7..3.3 (rotation a pair short
  2e-7..3.2e-6); TestMLABatchMatchesDecode's composed arm 1e-13 (a mixture
  block's several-row step declined by name, DeclineExpertsRows);
  TestMLASeamMovesWithItsHistory moves the history card, CPU composed,
  card at 2.1e-12 on CUDA and Vulkan; TestBlocksMoveBetweenDevices'
  composed-CPU arm passes on all five; the composed decode allocates
  nothing over 64 tokens on all five. Decode on one E5-2680 v4 socket
  (12 cores, numactl --membind=0), dev/bench TestComposedCPUAB, two
  passes: synth-deepseek-dense Q8_0 A/A 0.9976 and 1.0026, composed 0.351x
  and 0.347x main (IQR 5.6%, 3.6%; main 830 tok/s = 8.8 GB/s = 14% of the
  61 GB/s wall); Kimi-K3-0.40B Q8_0 A/A 0.9927 and 0.9971, composed 0.464x
  and 0.472x (main 190 tok/s), its prompt's mixture blocks declined by name.

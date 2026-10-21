# Inferred kernels: a weight format is a DESCRIPTION, and a description emits

**Status: SPEC. Nothing below is built.** Every number is either
measured on this box and cited, or is named as a prediction. Written to be
argued with before any code exists.

The ask, in the user's words: *"if we don't have a model's kernel maybe we can at
least infer one on the fly even if it's not the most optimized one"* -- and then,
sharply, when the first answer drifted into the reference tier: *"i mean still
jitted, no fickin nn in inference"*. That correction is the whole design. RULE 8
already says it: `nn` is the ORACLE and the BISECTION TOOL, and reaching for it
in a shipping path because a kernel does not exist is **a task, not a design**.

## The problem, in one table

Three different lists, and the gaps between them are the subject:

    quant.blockInfo      26 types        the engine can SIZE it: parse the file,
    (format/quant/types.go:40)                  find the tensors, report the shapes
    quant.Dequantable     9 types        the engine can DECODE it: F32, F16,
    (format/quant/dequant.go:162)               BF16, Q4_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K
    the fast path         6 types        a generated kernel exists: Q4_0, Q8_0,
    (cmd/jitllm/hardware.go:90)          Q3_K, Q4_K, Q5_K, Q6_K

So **seventeen formats can be sized and not decoded** -- Q2_K, Q4_1, Q5_0, Q5_1,
Q8_1, Q8_K, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS and the
integer types. A model using one does not convert at all: `convert` refuses and
the user gets a message, which is honest and is also the end of the road.

The three-lists shape is not an accident of neglect. RULE 7 states the cost:
adding a quant type is TWO pieces of work, the reference dequantizer FIRST
because RULE 8 makes it the oracle every kernel is gated against, then the
emitter. Two transcriptions of one byte layout, by hand, each able to be wrong
independently.

## What this proposes: tier 2, and no interpreted tier anywhere

    tier 1   hand-written specialised emitter        the 6 formats that have one
    tier 2   GENERATED from a block DESCRIPTION      anything describable
    tier 3   nn.Ref                                  ORACLE AND BISECTION ONLY

★ **TIER 2 IS MACHINE CODE. THAT IS THE POINT AND IT IS THE PART TO GET RIGHT.**
It is not a Go loop, not `nn.Dequant` followed by a dot, not an interpreted
fallback with a nicer name. It emits, maps executable, and runs at machine speed.
What it gives up against tier 1 is the specialisation, not the compilation:

    tier 1 for Q4_K   row interleave (BestPack), 2 accumulator chains
                      (BestAccs, measured 1.297x), the activation scale hoisted
                      per super-block, VPDPBUSD with the hi mask pre-shifted to
                      0x30, the two planes XORed before a single centering
    tier 2 for Q4_K   unpack the block per the description, widen, dot

Tier 3 does not move. `nn.JIT[t] == nil` must stay REACHABLE (RULE 8) so the
oracle and bisection exist; it is reached by nothing in a shipping decode, and
`model.TestNothingFallsBackToNN` already asserts that over every model on the
box. This spec does not weaken that line -- it removes the last excuse to cross
it, because "no kernel exists for this format" stops being true.

## The description format

A block quant is a byte layout plus an arithmetic rule. Q4_K, transcribed from
`format/quant/dequant.go`'s own comment and body:

    block_q4_K { f16 d; f16 dmin; uint8 scales[12]; uint8 qs[128] }   256 elems

and the rule: six-bit scale and min per 32-element sub-block, packed across the
twelve scale bytes (`scaleMinK4`); value is `d*sc*q - dmin*m`; **the low nibbles
are the first 32 outputs and the high nibbles the next 32, NOT interleaved
2l/2l+1** -- a sentence that exists in the source because getting it wrong is
silent and plausible.

A description therefore has to carry four things, and the fourth is the one that
bites:

    1. GEOMETRY     elements per block, bytes per block (blockInfo already has
                    this, for all 26)
    2. FIELDS       ordered {name, type, count}: f16 d, f16 dmin, u8[12] scales,
                    u8[128] qs
    3. ARITHMETIC   how a scale is derived from its field (identity, or a packed
                    bit-field extraction at a stated width and stride), and the
                    affine form: value = scale*q + offset, offset optional
    4. ORDERING     which output index a given packed unit lands on. Q4_K's
                    low-then-high, Q5_0's fifth bit from a separate plane, Q6_K's
                    4-bit primary and 2-bit secondary planes

★ **AND THE HONEST LIMIT IS THE IQ FAMILY, WHICH THIS CANNOT EXPRESS.** IQ2_XXS,
IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S are CODEBOOK quants: the packed bits are an
index into a fixed grid of vectors, not a magnitude. A layout description with an
affine rule does not reach them, and pretending otherwise would produce a
confidently wrong kernel. They need a codebook primitive -- a table in the
description and a gather in the emitter -- which is a SECOND piece of work and
should be costed separately, not smuggled in on the claim that "a description
covers formats".

What the affine description DOES reach, of the seventeen: Q4_1, Q5_0, Q5_1,
Q8_1, Q2_K, Q8_K, IQ4_NL and IQ4_XS (the last two are affine with a non-uniform
lookup of 16 values -- a small codebook, and the cheapest test of whether the
codebook primitive is worth building).

## Optimization passes, and the tier 2 -> tier 1 continuum

The user's second point: *"we can later run optimization passes on it too. or
after we get data through it, then we cache theses kernels"*.

★ **THE ENGINE ALREADY DOES THIS SHAPE THREE TIMES, FOR PARAMETERS. THIS EXTENDS
IT TO BODIES.**

    engine/nn/tune.go          duels pack widths on REAL decode tokens, ABBA, and caches
                        the winner to ~/.cache/jitllm/packwidth keyed by
                        (parameter, CPU, cores, shape set)
    tier/choose.go      times one real matvec on each backend and caches the
                        winner to ~/.cache/jitllm/gpuapi keyed by the device set
    nn.AddShape         emits per (type, blocks-per-row) from the model's OWN
                        tensor shapes at load, only for the widths in play

So "run it, watch it, specialise, remember" is the existing idiom. The passes
that take tier 2 towards tier 1 are not new inventions -- they are exactly the
decisions the six hand-written emitters already make, applied from the
description instead of by hand: interleave width, accumulator chains, hoisting
the activation scale per super-block, plane XOR before centering, unroll depth.

That is the real prize, and it is bigger than the new formats: **tier 1 and tier
2 stop being two codebases.** A hand-written emitter is a description plus a set
of decisions; if the decisions are separable, the hand-written six become
descriptions with their decisions pinned, and a seventh format inherits the
machinery rather than a transcription.

### Cache the DECISION, not the bytes

★ **AND CACHING GENERATED CODE TO DISK BUYS NOTHING, BY A MEASUREMENT THIS REPO
ALREADY HAS.** `nn.AddShape`'s own comment: *"codegen is microseconds, so this is
about not holding 40 mappings we will drop ten tokens later."* If emitting is
microseconds, a disk cache of machine code buys nothing measurable and costs a
great deal:

    it must be keyed on the description AND the CPU feature probe (jit_old/cpu
    probes CPUID leaf 7.1 for VNNI, AT_HWCAP for DotProd) AND the shape AND the
    engine version -- and a STALE HIT IS WRONG MACHINE CODE, not a slow path

Whereas the decision -- "for this description at this shape on this CPU, pack 4,
accs 2, scale hoisted" -- is a few bytes, is exactly what `packwidth` already
stores, and re-emits in microseconds from a description that can be re-verified
on the spot. **Cache what was expensive to DECIDE, never what is cheap to
REBUILD.**

If a future search pass is genuinely expensive (a real autotuning sweep rather
than a duel), that strengthens the same conclusion: cache the search RESULT.

## The gates

RULE 10 is the standing record of how often a dead or wrong fast path has
shipped here, and an inferred kernel is a fast path nobody read. Four gates, each
of which must fail against a violation before it is believed:

    1. DESCRIPTION vs REFERENCE. The description-derived decoder and
       quant.Dequant32 must agree at ZERO ULP on the nine formats that have
       both. dequant32_test.go already asserts zero ULP between the f32 and
       f64 signatures, so the bar exists and is not a tolerance.
       Against a violation: a wrong ordering rule for Q4_K (interleaved 2l/2l+1
       instead of low-then-high) must fail, not merely drift.

    2. TIER 2 vs THE DESCRIPTION. The emitted tier-2 kernel and the
       description-derived decoder must agree bit-for-bit on random blocks,
       including the ragged tail and a block of all-zero and all-max codes.
       RULE 13's poison rule applies to any scratch it allocates.

    3. WHICH TIER RAN. A counter per tier, asserted ZERO for tier 2 on every
       format that has a tier-1 kernel, over Forward, Prefill AND ForwardBatch --
       the same sweep shape as model.TestNothingFallsBackToNN. A tier-2 kernel
       silently serving Q4_K is a 2-5x regression that produces correct tokens
       and would otherwise never be noticed.
       Against a violation: forcing Q4_K to tier 2 must make the gate read a
       non-zero count, not just a slower row.

    4. NO NEW INTERPRETED PATH. model.TestNothingFallsBackToNN keeps asserting
       the nn counters at zero. Tier 2 exists so that a missing kernel is not a
       reason to cross RULE 8's line, and the gate that proves it is the one
       already there.

★ **AND THE GATE THAT DOES NOT EXIST YET IS THE INTERESTING ONE:** a format with
a description and NO tier-1 kernel has no independent oracle. Gate 1 covers the
nine where both exist; for a tenth, the description IS the specification, and
"the kernel matches the description" proves internal consistency rather than
correctness. The honest answer is that a new format's description must be
transcribed against a SECOND artefact -- llama.cpp's own decoder for that type,
run on the same bytes (RULE 7m: a good-enough oracle, not the specification) --
before the format is admitted. That is a conversion-time check, once per format,
not a runtime one.

## What this does NOT propose

**Inferring a GRAPH.** RULE 7's architecture list stays a LIST. A GGUF carries
weights and hyperparameters; it does not carry the graph, and qwen3next needed a
per-layer kind map, an SSM geometry, a shared expert, a double-width query whose
gate interleaves per head and partial rotary -- none of it inferable from the
file. llama.cpp has 176 `LLM_ARCH_` cases and 50 graph builders; Ollama had a
funded team, covered ~21, and deleted 430,004 lines citing exactly this. Scope is
jitllm's only defence, and a format description does not touch that argument in
either direction.

**Removing the conversion-time refusal.** `convert.archOf` refusing an
unimplemented architecture is correct and stays. What changes is only that
"unknown quant type" stops being one of the reasons.

## Predictions, named as predictions

None of this is measured. The two numbers that decide whether it is worth
building, and neither has been taken:

- **What does tier 2 cost against tier 1 on a format that has both?** Emit a
  tier-2 Q4_K, ABBA it against the shipping kernel, decode, six P-cores. If it
  lands within 2-3x, an unoptimised-but-generated path is a real product answer
  for an unsupported format. If it is 20x, tier 2 is a correctness scaffold and
  the optimisation passes are not optional -- which is a different project and
  should be costed as one. **My guess is 3-6x and I would not defend it.**

- **How many of the seventeen are actually affine?** Counted above as eight, from
  the block sizes and llama.cpp's type names. That count is from the table, not
  from reading each format's decoder, so treat it as an estimate that a morning
  of transcription would replace with a fact.

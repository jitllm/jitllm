# Architectures, tokenizers and correctness — evidence

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

## ★ THREE TRANSCRIPTIONS OF ONE GRAPH, TWO WAYS THEY DISAGREED

`TestPrefillAfterForwardContinues`, `TestBatchMatchesForward`,
`TestBiasedBatchMatchesForward`, `TestRaggedBatchMatchesForward` and
`TestSchedulerMatchesSoloSequences` demand BIT equality between Forward and the
batched paths with the GEMM pinned out. Five of them were red, on two models and
for two unrelated reasons. Both are RULE 7l's shape: three transcriptions of one
graph drifting.

**★ 1. stories260K: THE POOL SPLIT CHOSE THE ARITHMETIC.** Its `n_ff` is **172**,
which is not a whole number of AVX2 vectors, and `Parallel` handed out ELEMENT
ranges:

    decode    one chunk of 172          -> 172 %% 8 != 0, ActMul32JIT DECLINES
                                           -> the float64 Go twin
    prefill   chunks of 512 and 4       -> 512 aligned, the KERNEL ran on it

Both are correct. Neither is bit-identical to the other -- the Go twin exps in
float64 where the kernel range-reduces and Horners in f32 -- so the two paths
computed different FFNs and the gates read max|dlogit| 2.9e-06.

The fix is two parts and needs both: the FFN scratch is allocated at
`ffnPad(width)` so a ragged tail has legal slack to ride into and the kernel
serves EVERY width, and the pool splits by VECTOR so only the last chunk can be
ragged. **A kernel that declines for a width is a kernel whose arithmetic the
scheduler picks.**

★ THE FIRST VIOLATION I RAN FOR THIS PASSED, AND THE REASON IS WORTH KEEPING.
Removing only the padding made BOTH paths ragged and both fell back, so they
agreed -- for the wrong reason, with every activation interpreted. A violation
has to reconstruct the DEFECT, not merely undo one line of the fix: removing the
padding AND restoring the element-wise split reproduces it at 2.86e-06.

**★ 2. qwen3moe: ORDER AND PARENTHESISATION, WHICH RULE 7m HAD ALREADY NAMED AND
NOBODY HAD ACTED ON.** `moe()` summed the selected experts in
DESCENDING-PROBABILITY order into a zeroed buffer and added that to the residual;
`moeBatch()` summed them in ASCENDING EXPERT ID straight into the residual. Two
different float sums of the same k terms:

    decode   x + (w0*d0 + w1*d1 + ...)          selection order
    batch    ((x + wa*da) + wb*db) + ...        expert-id order

**In a mixture that is not a rounding difference.** The perturbed vector is the
next layer's router input, a near-tied top-k flips, and a DIFFERENT EXPERT runs --
which is why the gates read max|dlogit| 0.57 rather than 1e-6.

Both now add each expert's contribution STRAIGHT INTO THE RESIDUAL in ascending
expert id. Decode sorts its selection -- at most eight ints, insertion sort, no
allocation -- and keeps SELECTION order for the trace, the hotness counter and
the renormalisation sum, because those are about which experts were chosen. The
batched path already visited the bank expert-major and did not change at all.

**★ THE FIRST VERSION MATCHED THE OTHER WAY ROUND AND IT COST 0.35%.** It gave
the BATCHED path a zeroed per-row accumulator and a final pass, so both sides
computed `x + (sum)`. Correct, and measured 0.9965 on M4 MoE prefill against a
0.9999 self-control -- the zeroing and the extra pass over n*NEmbd, exactly as
the arithmetic predicted. Turning it around instead -- decode adds straight in,
like the batched path already did -- is bit-identical AND cheaper than either:

    M4, Qwen3-MOE-4x0.6B, ABBA, 4 rounds      prefill   decode
      first version (batched accumulator)      0.9965    --
      shipped (both add into the residual)     0.9999    1.0021
      SELF-CONTROL                             1.0009

Decode gains because it now does k residual adds where the buffered shape did
k+1. **When two paths must agree, check which one is cheaper to move BEFORE
moving the other**; the obstacle looked like gemma's post-FFN norm, and no
mixture jitllm supports has one -- `forward.go` refuses rather than skipping it
silently if one ever does.

★ BOTH FIXES RUN AGAINST VIOLATIONS. Summing decode in selection order fails four
subtests; adding the batched contributions straight into the residual fails four;
the reconstructed split defect fails stories260K.

★ AND THE MoE FAULT MIRROR IN `moe_test.go` HAD TO FOLLOW. It reproduces `moe()`
bit for bit before injecting anything, so it calls the same softmax, the same
activation and the same axpy -- a mirror exists to be perturbed, so anything it
does differently is measured as the fault.

## ★ RULE 7: SEVEN architectures -- llama/llama3, qwen2, qwen3, qwen3moe, olmoe, gemma1, phi3/phi4. Adding one needs a decision.

(Plus `clip` for the vision tower, which is a separate FILE and a separate entry point -- RULE 7h.
The count read FIVE for a time while qwen3moe (RULE 14) and olmoe (RULE 7k) were already in
the switch, which is RULE 1a's stale number in the one place that gates new work.)

Held to llama.cpp on seven real models, 20 greedy tokens each, compared as token ids:

    tinyllama-1.1b-q3_K_M   llama    SPM      identical
    llama-2-7b Q4_K_M       llama    SPM      diverges at token 6 on a 0.109 tie
    SmolLM2-360M Q8_0       llama    BPE      identical
    Llama-3.2-1B Q4_K_M     llama    BPE      identical   (needs rope_freqs)
    Qwen3-1.7B Q4_K_M       qwen3    BPE      identical   (needs q/k norm + NEOX)
    Phi-3.5-mini Q4_K_M     phi3     SPM      diverges at token 9 on a 0.328 tie
    gemma-2b                gemma    SPM      identical   (was WRONG, see 7a)

**★ 7-0. THE TOKENIZER IS THE GATE, NOT THE GRAPH.** Every model newer than llama2 carries byte-level
BPE (`tokenizer.ggml.model == "gpt2"`), and until `tok/bpe.go` existed none of them could be run from
text at all. It is a DIFFERENT ALGORITHM from spm.go rather than a variant: SPM merges by SCORE and
never reads `merges`, BPE merges by RANK in `merges` and never reads `scores`. Running one model's
data through the other's loop produces fluent, wrong output.

The pre-tokenizer split is hand-rolled because `\s+(?!\S)` is a negative lookahead, which Go's RE2
cannot express by construction and `std::regex` cannot either -- llama.cpp writes it out by hand for
the same reason. There are THREE patterns, not one, and they are not interchangeable: the llama3
family (llama3, llama-bpe, qwen2, dbrx, ...), the original gpt-2 family (gpt-2, olmo, jais, phi-2),
and starcoder's (gpt-2's, with digits split singly first). An unknown `tokenizer.ggml.pre` is
REFUSED, not defaulted -- `tok.New` errors and `model.Open` keeps the weights.

Two flags default per family and cost a divergence when wrong. `add_bos` is false for BPE generally
and TRUE for the llama3 family: Llama-3.2 ships no `add_bos_token` key and still wants token 128000,
while qwen3 ships an explicit 0 and does not. `ignore_merges` (llama3 family only) emits a
pre-token whole when it is already a vocabulary entry, which is NOT an optimisation -- the merge
sequence can reach a different split than the whole-word entry.

`TestBPEAgainstLlamaCpp` holds 51 cases across three models to `llama-tokenize --ids`, chosen for
what is easy to get subtly wrong: contractions, digit grouping in threes, runs of spaces where the
LAST one belongs to the next word, newlines, CJK, emoji and special tokens.

**★ 7a. AND GEMMA'S ROTARY LAYOUT WAS WRONG FOR THE LIFE OF THE PROJECT.** `nn/ref.go` asserted that
"both llama and gemma use NORM, because convert_hf_to_gguf.py permutes the Q and K weights". The
first half is true. The second is not: llama.cpp's `llama_model_rope_type` puts `LLM_ARCH_GEMMA`
in the NEOX group, with qwen and phi3. jitllm rotated consecutive pairs (2i, 2i+1) where gemma wants
(i, i+nRot/2).

    prompt "The capital of France is", gemma-2b, greedy
    jitllm (NORM)   ":"        logit 8.633, next " Paris" 7.304   -> ":\n\na. Paris\nb. London"
    llama.cpp    " Paris"                                       -> " Paris."

A margin of 1.33 is not a near-tie, so nothing about this was arguable -- and yet every test in the
tree passed. **The reference tier is an oracle for the KERNELS, not for the ARCHITECTURE.** float64
reference, generated AVX2, and GPU blocks all ran the same wrong rotation and agreed with each other
to the bit, which is exactly what `TestPlacementInvariance` checks. Only an independent
implementation of the same model can see this class of bug, and `model.TestGreedyMatchesLlamaCpp`
(the exported-API one, in `llamacpp_test.go`) is now that: it decodes 20 greedy tokens on both
engines and compares TOKEN IDS, re-tokenizing llama.cpp's text with jitllm's own tokenizer.

Two things that test had to get right before it was worth anything. **A disagreement is only a bug
if the margin is wide** -- the two engines quantize activations differently and sum in a different
order, so llama-2-7B diverges at token 6 on a 0.121 tie and that is not a defect; the gate is
forward_test.go's `tieMargin` of 0.5. And **llama.cpp's completion must be re-tokenized as part of
the whole text, not on its own**: SPM prepends a space, so encoding " Paris. ..." alone yields
"▁▁"+"Paris" and every SentencePiece model appears to fail at token 0.

The performance numbers in this file are unaffected -- a rotary layout changes no bytes read and no
FLOPs issued -- but every gemma completion jitllm printed before 7a's rotary fix was fluent and wrong.

**★ 7b. WHAT ADDING AN ARCHITECTURE ACTUALLY COSTS, MEASURED ON QWEN3.** One `Config` flag and one
graph change: `attn_q_norm`/`attn_k_norm`, a head_dim-wide RMSNorm applied to each head of q and k
after the projection and before RoPE. Plus NEOX rotary, which gemma turned out to need anyway. That
is ~40 lines in `engine/model/`, and it works on CPU at 40.6 tok/s with output identical to llama.cpp.

What it did NOT cost is the thing to remember: no new kernels, no `Spec` field, no backend change.
The expensive part of a new architecture is never the arithmetic -- it is the TOKENIZER, and MoE.

**★ 7b-bis. A "WHICH MODEL IS THIS" TEST BELONGS AT LOAD, NEVER IN THE INNER LOOP.** Every
architecture flag jitllm adds is a constant from the moment `model.Open` returns, so a branch on one in
the hot path is a branch whose answer never changes. Two were costing real work:

- `act := nn.SiLU; if c.GELU { act = nn.GELUTanh }` and then `act(gate[i])` **per element**. That is
  ~295,000 INDIRECT calls per token on gemma -- 16384 elements times 18 blocks -- none of them
  inlinable, for one bool. `nn.ActMul` takes the bool and runs one of two loops.
- `Rope.Apply` recomputed sin and cos **per head**, and tested `Neox` **per pair**. Every head at a
  position rotates by the same angles, so qwen3 was calling sin/cos 40 x NRot/2 times per block for
  NRot/2 distinct values: 28,160 transcendentals per token where 64 would do. `Rope.Table` fills
  them once per token -- which the GPU block path already needed -- and `ApplyTable` hoists the Neox
  test above the loop.

The general form, and the reason this is a rule rather than two fixes: jitllm GENERATES its code, so an
architecture difference is a codegen input, not a runtime condition. Baking K and the quant type into
a matvec is the same argument. When a new architecture adds a flag, ask where it is READ -- if the
answer is "inside a loop over elements", it is in the wrong place.

**★ 7c. TIED EMBEDDINGS ARE A PROPERTY OF THE FILE, NOT OF THE ARCHITECTURE.** gemma always ties and
llama usually does not, but qwen3-1.7B ties and qwen3-8B does not, under the same arch string. The
absence of `output.weight` is the fact; the arch is a guess about it. `loadConfig` reads it.

**★ 7c-bis. FUSED TENSORS ARE SLICED, NOT COPIED.** phi3 (and phi4, same arch string) ships q, k and
v fused into `attn_qkv` and gate/up fused into `ffn_up`. A GGUF matrix is row-major with every row a
whole number of quantization blocks, so a row range is a contiguous BYTE range: `model.rows` returns
a sub-tensor aliasing the mapping. Copying would double phi3's resident weights for nothing, and the
GPU tier keys residency on the byte pointer, so the aliased slices stay distinct there too.

**★ 7d. AN UNSUPPORTED TOKENIZER NO LONGER REFUSES THE WEIGHTS.** `model.Open` records `TokErr` and
leaves `Vocab` nil. The graph and the vocabulary are independent, and conflating them meant a model
whose tensors jitllm runs perfectly well could not be loaded, let alone measured, because its
tokenizer was a variant `tok/` had not learned.

## ★ RULE 7 (original): two architectures. llama and gemma1. Breaking this needs an explicit decision.

llama.cpp has 176 `LLM_ARCH_` cases and 50 distinct graph builders. gemma2 alone needs alternating
SWA masks, four norms per block, per-model-size Q scaling, and tanh softcapping inside the attention
loop — a second and third block kernel, not a config tweak. Ollama had a funded team, covered ~21
architectures, and deleted 430,004 lines citing exactly this. **Scope is jitllm's only defence.**

The fast-path quant list (Q4_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K) is a *list*, never a promise of
coverage. Anything else runs on `nn.Ref` and logs once.

**★★ AND "jitllm NEVER REFUSES TO LOAD A MODEL" IS FALSE, TWICE OVER. A REVIEW FOUND
IT; BOTH HALVES ARE VERIFIED IN THE SOURCE.** The sentence used to end this
paragraph and it is the reassuring kind of wrong -- it describes a fallback
policy the code does not implement:

  1. **AN UNKNOWN ARCHITECTURE IS REJECTED OUTRIGHT.** `loadConfig`'s `default:`
     returns `architecture %q is not implemented`, so gpt-oss, gemma2, gemma3 and
     qwen3next do not load at all. There is no reference-tier fallback for a
     graph jitllm has not been taught.
  2. **AND THE REFERENCE TIER CANNOT READ EVERY QUANT EITHER.**
     `gguf.Dequantable` is `{F32, F16, Q4_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K}` --
     the SAME list plus F16. Q5_0 and MXFP4, which are what gpt-oss is made of,
     have no dequantizer, so "runs on nn.Ref" is not available to them.

**THE CONSEQUENCE FOR PLANNING, WHICH I GOT WRONG THREE TIMES IN ONE SESSION:**
adding a quant type is TWO pieces of work and the reference one comes FIRST,
because RULE 8 makes `nn.Ref` the oracle every kernel is gated against. A kernel
with no reference to check it against cannot be gated at all. My three
successive estimates for gpt-oss were "two graph features", then "two kernels,
and meanwhile it runs at 0.5 tok/s on the reference tier", and only now the real
one: **a dequantizer, then a kernel, then the graph -- and until the
dequantizer exists there is no tier that can run it.**

★ RULE 8's "`nn.JIT[t] == nil` is a supported configuration" is still true and is
a different claim: it is about a TYPE the reference tier can read. The fallback
is real within `Dequantable` and absent outside it.

## ★ RULE 7e: THE EMBEDDING SEAM. THE ENGINE ACCEPTS A RESIDUAL VECTOR IT DID NOT LOOK UP.

`ForwardEmbd(e)` and `PrefillMixed(spans...)` are `Forward` and `Prefill` with the embedding
SUPPLIED rather than read from the vocabulary. A `Span` carries either token ids or embeddings,
never both, so a prompt is `text, image, text` without either half knowing about the other.

**★ IT IS THE FIRST PIECE OF VLM WORK BECAUSE IT NEEDS NO VLM.** A vision-language model encodes an
image into patch embeddings and splices them where a placeholder token sits; everything downstream
-- blocks, KV cache, sampler, device tier -- is unchanged. So the seam can be built and GATED today,
on models already in the oracle table, with no tower, no mmproj and no second file. Whatever
architecture decision RULE 7 eventually takes, this part is common to all of them.

**THE GATE IS BIT EQUALITY, NOT NMSE, AND THAT IS THE POINT.** Everywhere else a kernel is held to
an NMSE bar because generated code sums in a different order or a narrower type. Here both paths
reach the same block stack with the same float64 values and the only difference is where the row
came from -- the fill is a `copy`. So ANY difference is a defect, and a tolerance would hide exactly
the defects worth catching: a scale applied twice, a row read at the wrong stride, a position not
advanced. Measured: **0 of 256128 logits differ on gemma, 0 of 32000 on tinyllama**, across 15
models including MoE.

    ForwardEmbd     6 tokens, one at a time, against Forward
    PrefillMixed    24 tokens as three spans (7 tokens, 12 embeddings, 5 tokens),
                    cut deliberately OFF a chunk boundary, against Prefill

**★ e IS THE RESIDUAL STREAM, NOT A VOCABULARY ROW: `EmbdScale` IS NOT APPLIED.** gemma multiplies
its looked-up embedding by sqrt(n_embd), and a caller splicing in projected image features must not
have that done to them again. Run against the violation -- apply it inside the seam -- gemma fails
at 256128/256128 logits and |delta| 1.9e+03. Reading a span row one late fails at 32000/32000.

**IMPLEMENTATION: ONE CLOSURE, NOT A SECOND CODE PATH.** `forward(fill func() error)` and
`prefillSrc(slot, src []func([]float64) error)` are the existing bodies with the row source
abstracted; `Forward` and `Prefill` pass a vocabulary lookup and the mixed entries pass a copy.
`fill` is a CLOSURE rather than a slice because the device-failure path calls it twice -- `Layers`
may have written part of the residual stream before failing, so the token restarts on the host.

★ AND THE FIRST VERSION OF THE GATE TOOK THE model PACKAGE FROM 43 s TO A 10-MINUTE TIMEOUT,
because it globbed every file in `../models` and found a 70B and an 80B. The seam is
architecture-independent -- what it gates is decided by `embed()`, the span walk and the chunk loop,
none of which knows how many layers follow -- so the sweep skips files over 2 GB unless `JITLLM_MODEL`
pins one. RULE 14f is the precedent: a suite that stops finishing stops being run.

★ AND THE FIRST RUN OF THE VIOLATIONS REPORTED `ok` IN 0.001 s BECAUSE IT NEVER RAN. `JITLLM_MODEL`
resolves against the TEST's working directory, so `models/gemma-2b.gguf` missed, `Open` failed, and
`t.Skipf` reported success -- RULE 14r inside the very check that exists to satisfy RULE 12m. A
violation that "passes" instantly is not a passing violation; check the subtest ran.

## ★ RULE 7f: ATTENTION BIASES ARE LOADED AND ADDED, ON BOTH TIERS. THE DEVICE DOES NOT DECLINE.

`engine/model/model.go` asked only for `.weight`. qwen2 puts biases on q, k and v, `loadConfig`'s comment
claimed they "are loaded if present", and `grep -n '\.bias' model/` returned NOTHING -- so a qwen2
model ran with them silently dropped. Fluent, wrong output; no oracle caught it because RULE 7's
table has qwen3, which genuinely has none, and no qwen2 file.

    model/       optVec loads attn_{q,k,v,output}.bias when the file has them, nil otherwise
    forward.go   addBias after each projection; prefill.go addBiasRows over the chunk
    engine/nn/device.go LayerWeights.Bq/Bk/Bv/Bo
    kernels/     MatVecShape.Bias folds a pBias load into the matvec's own store
    tier/        the vectors upload as float32 beside the norms, and are freed on release

**★ THE DEVICE ADDS THEM RATHER THAN DECLINING, AND THE FIRST VERSION DECLINED.** Declining was
defensible -- RULE 15c says prep-by-omission must fail loudly rather than skip a weight -- and it
was still wrong: it silently moves a whole architecture onto the CPU and calls that success. **A
missing feature is a feature to implement, not a capability to refuse.** The bias is folded into the
matvec's own store, not launched as a second elementwise kernel: 4 projections x n_layer is 88 extra
launches a token on tinyllama, and RULE 12p measured encode alone at 8% of a Metal token.

**★ AT Split > 1 THE BIAS MUST LAND ONCE, NOT Split TIMES.** Each segment writes its own partial and
`Reduce` sums them, so the addend is gated to segment 0 with `ir.Select` -- DATA FLOW, so RULE 12d's
"no conditional execution outside a loop" still holds and surplus threads still clamp harmlessly.
That gating is invisible at Split == 1 and wrong-by-Split at any other, which is why the on-hardware
test now runs EVERY shape twice, with and without a bias, at every split width.

★ RUN AGAINST VIOLATIONS on real hardware (PTX): adding the bias in every segment fails at
`Q4_0/128x256/split2` -- precisely the case the gate guards -- and never adding it fails at
`Q4_0/1x32/split1`.

**★ PRESERVATION IS THE LOAD-BEARING GATE, NOT THE NEW BEHAVIOUR.** Every model jitllm already runs has
no biases, so the addend must cost exactly nothing there. `jitllm run -n 24` output is BYTE-IDENTICAL
before and after on tinyllama, gemma and Qwen3-1.7B, and the FFN kernels are generated with
`Bias: false` and are unchanged. What remains when the vector is nil is one predictable branch per
projection per layer -- 88 per token on tinyllama -- which is below this harness's resolution.

★ AND THE PERF A/B COULD NOT RESOLVE IT, WHICH IS THE HONEST STATEMENT. Self-control 0.991, decode
median 1.004 (spread 0.975-1.050), prefill median 1.019 (spread 0.989-1.046), two RULE 2d clamps
discarded, on a box that drifted 74.8 -> 66.3 during the control itself. **The bit-identity check is
the evidence; the ratio is not.**

★ THE SELF-CONTROL RAN FIRST THIS TIME. RULE 5ac was withdrawn the same day because an M4 A/B had a
3.5% slot bias that only a same-binary-both-arms run exposed. Linux's is 0.991. Run it first.

## ★ RULE 7g: A REAL mmproj, MEASURED. MOST OF THE VLM WORK LIST WAS INFERENCE; HERE IS THE FILE.

Every VLM claim in this file and in the design document of the time (ARCHITECTURE.md, since
deleted) was read off llama.cpp's symbol table and jitllm's
source. `ggml-org/SmolVLM-256M-Instruct-GGUF`, 104 MB, is now on the box and `jitllm info` reads it.

    arch      clip "SmolVLM 256M Instruct"      198 tensors   40 kv   gguf v3
    types     Q8_0  73 tensors  94.22%          F32  125 tensors  5.78%       NO F16 AT ALL
    tower     12 blocks  d=768  ffn=3072  12 heads (head_dim 64)
              image 512  patch 16  ->  1024 patches   layer_norm_eps 1e-6   use_gelu true
    projector clip.projector_type = idefics3, scale_factor 4
    prefixes  197 "v."  +  1 "mm."

**★ FIVE THINGS THAT WERE ASSUMED AND ARE NOW OBSERVED.**

  1. **`convert/gguf/` PARSES AN mmproj WITH NO CHANGES.** `jitllm info` prints the whole header. The 4-D
     `v.patch_embd.weight` (16x16x3x768) parses, which is the item most expected to be a gap.
  2. **THE F16 GEMM IS NOT NEEDED FOR THIS TARGET.** The heavy tensors are Q8_0 and the rest are
     F32, both already in `cpu.Supported`. This was the largest item on the work list and the
     audit removes it -- for this file. It says nothing about gemma-3 or any f16-only mmproj.
  3. **EVERY k IS A MULTIPLE OF 32.** The distinct leading dimensions among the Q8_0 matrices are
     768, 3072 and 12288. `nn.MatMul` refuses `k%32 != 0` and would have declined silently.
  4. **THE PROJECTOR IS ONE MATMUL, NOT AN MLP.** `mm.model.fc.weight` is 12288x576 Q8_0 and it is
     the ONLY `mm.` tensor. 12288 = 768 * 16 = d * scale_factor^2, so idefics3's "pixel shuffle" is
     a reshape that groups 4x4 patches, and 576 is the text model's n_embd -- which is the seam
     contract libmtmd states as `n_embd(text) == n_embd(mmproj)`.
  5. **LayerNorm WITH A BIAS IS THERE IN THE FILE**, as `v.blk.N.ln1.{weight,bias}`, alongside
     `v.patch_embd.bias`. RULE 7f's bias loading and `nn.LayerNorm` are both already built.

**★ AND THE TOWER IS A 1024-ROW BATCH, WHICH IS THE PREFILL GEMM PATH.** 512/16 = 32 patches a side,
and `v.position_embd.weight` is 768x1024, confirming 1024 positions. That is not a special case: it
is exactly the shape `nn.JIT.MatMul` was tuned for, and the CPU prefill row against llama.cpp is
0.8720. **The tower's compute lands on jitllm's best-developed path.**

★ WHAT IS STILL NOT BUILT, and this rule does not change it: the tower graph, the patch gather, the
non-causal attention path, the image preprocessing, and `clip.*` key reading -- `jitllm info` shows
`L=0 d=0 heads=0/0`. ★ AND THAT REASON IS WRONG AS WRITTEN, as a later correction found: `info` never
calls `loadConfig` at all -- it opens the GGUF and reads the header itself with `f.UintKey`, and
`meta.File.Key` namespaces by `f.Arch()`, so on an mmproj it asks for `clip.block_count` when the
real key is `clip.vision.block_count`. The defect is the missing `.vision` segment, not a
`llama.*`-vs-`clip.*` mismatch, and a reader who fixed `loadConfig` would have fixed nothing.

★ THE METHOD IS THE POINT. One download and one command replaced a work list built from
`nm -D | c++filt` and a per-tensor inventory that a reviewer correctly refused to accept without one.
RULE 14a said the same thing about a graph: read the artefact, not the description of it.

## ★ RULE 7h: THE TOWER LOADER, AND THE MLP TENSOR NAMES ARE INVERTED IN THE FILE.

`model.OpenTower` reads an mmproj into a `Tower`: the `clip.*` header, twelve blocks of
(LayerNorm, q/k/v/o with biases, an UNGATED MLP), the position embeddings and the projector. It is
a separate entry point because it is a separate FILE -- libmtmd's contract is two files and one
equation, `n_embd(text) == n_embd(mmproj)` -- and `Open` refuses arch `clip` for the same reason.

**★ AND THE FIRST RUN OF ITS GATE CAUGHT A LAYOUT TRAP IMMEDIATELY.** In SmolVLM's mmproj:

    v.blk.0.ffn_down.weight   768x3072    <- the EXPANSION (k=768, rows=3072)
    v.blk.0.ffn_up.weight     3072x768    <- the CONTRACTION

**That is the opposite of llama's convention, where `ffn_up` expands.** Loading by NAME gives two
matmuls whose inner dimensions do not meet. On this tower the shapes differ so it would be caught;
on a tower where NFFN == NEmbd it would land inside the mapping and answer fluently, which is RULE
14c's exact failure ("down's ne0 is NOT gate's ... one stride formula reused for all three reads
another expert's weights, lands inside the mapping, and never faults").

**So the two matrices are identified BY SHAPE.** Exactly one has `rows == NFFN` and exactly one has
`k == NFFN`; the loader asserts that and refuses if neither arrangement holds. The layout is then
self-validating and immune to the naming convention -- which matters because the next tower family
will have its own.

★ RULE 7's ANSWER IS A LIST HERE TOO. `clip.projector_type` must be `idefics3`; anything else is
REFUSED at load rather than defaulted, exactly as `tok.New` refuses an unknown pre-tokenizer (RULE
7-0). b10825 carries ~30 projector types and running one's arithmetic on another's weights does not
fault, it answers fluently.

★ AND THE PROJECTOR'S k IS CHECKED AGAINST `NEmbd * Scale^2`, not NEmbd. idefics3 groups Scale x
Scale neighbouring patches before projecting -- 768*16 = 12288 -> 576 -- so a loader that assumed
NEmbd would read a quarter of each group. 1024 patches become 64 embeddings.

## ★ RULE 7i: THE TOWER RUNS. AND "f64 REORDERING" IS THE WRONG YARDSTICK FOR EVERY NUMBER IN IT.

`Tower.NewState().Encode(px)` takes a preprocessed image to the embeddings `PrefillMixed` splices
in: patch gather, 12 blocks, projector. Measured on SmolVLM's real mmproj -- 1024 patches to 64
tokens of 576, finite, non-constant.

    patch embed   conv2d with stride == kernel IS a matmul over flattened patches: a gather,
                  then nn.JIT.MatMul. No convolution kernel anywhere.
    blocks        LayerNorm(+bias) -> q/k/v(+bias) -> BIDIRECTIONAL attention -> o(+bias)
                  -> residual -> LayerNorm -> fc1 -> UNGATED GELU -> fc2 -> residual
    projector     idefics3 pixel shuffle (SPATIAL 4x4 grouping) then ONE matmul, 12288 -> 576

**★ THE ATTENTION USES THE GENERATED KERNELS, AND THE FIRST VERSION DID NOT.** I wrote it as a
float64 Go triple loop -- O(patches^2) at 1024 patches x 12 heads x 12 blocks, un-JIT'd, in an
engine whose entire identity is that it generates its own kernels. `EmitAttnScores`/`EmitAttnAcc`
already existed on both architectures and needed NOTHING tower-specific: they bake head dim and
stride and read the position count at runtime, so **non-causal is `Rows = n` instead of `pos+1`,
an argument rather than a kernel.** One image's k and v are the same layout as a KV cache.

**★ AND THE PRECISION CLAIM I MADE ABOUT IT WAS WRONG TWICE, WHICH IS THE PART TO KEEP.** Two
Encode calls disagreed at NMSE 2.2e-4 and I called that "too large for float64 reordering". jitllm does
not do float64 arithmetic in a matmul:

    Args.Out is *float32      f.gout is []float32      activations are int8
    dst[r] = float64(buf[..]) -- f64 is the CONTAINER, applied at the transpose

So the yardsticks are f32 epsilon (~1e-7) and **int8 activation quantization (~0.4% per matmul)**,
and 1.5% RMS across twelve blocks of those is ordinary. The settled figure I also called "float64
reordering", NMSE 7.5e-15, is RMS relative 8.7e-8 -- **one f32 ulp**, i.e. essentially bit-identical.
RULE 14s says it plainly ("most of jitllm is f32 already: every generated CPU kernel writes
`Out *float32`") and I reached for f64 anyway because the buffer type said so.

**★ THE TILE TUNER CONVERGES ON SHAPES IT HAS NEVER SEEN, AND THAT IS WORTH GATING.** The first
determinism assertion failed because consecutive encodes ALTERNATED between two outputs -- which is
engine/nn/tune.go duelling incumbent against challenger (RULE 5e), made visible by int8 rather than hidden
by f64. Measured: it settles by ~22 encodes, after which consecutive runs agree at NMSE 0 to
7.5e-15. So the test warms 24 encodes and then asserts. **A tower whose tile never settled would be
an autotuner that silently does not apply to vision**, and nothing else would have said so.

★ STILL A GO LOOP, AND NAMED SO IT IS NOT MISSED: LayerNorm, the residual adds and the softmax.
They are O(patches x NEmbd), not O(patches^2), but they are not generated and the constraint is that
everything is.

## ★ RULE 7j: THE ORACLE RAN ON THE FIRST IMAGE AND FOUND BOTH THINGS IT WAS BUILT TO FIND.

`llama-mtmd-cli` on the same mmproj, the same text model and the same PNG. Two results, and neither
was reachable by reading the file.

**★ 1. llama.cpp INDEPENDENTLY CONFIRMS THE ffn INVERSION.** Its loader logs, unprompted:

    W load_tensors: ffn up/down are swapped

RULE 7h found that by SHAPE -- exactly one matrix has `rows == NFFN` and exactly one has
`k == NFFN` -- and refused to trust the names. The reference engine special-cases the same file with
a warning. **A layout deduced from first principles and a layout special-cased by the reference
agreeing is the strongest confirmation available short of comparing numbers**, and it says the
shape-based rule generalises where a per-file exception would not.

**★ 2. RETRACTED: "SmolVLM SPLITS THE IMAGE" WAS A MISREADING OF A LOG LINE, AND IT COST A DAY OF
LOOKING IN THE WRONG PLACE.** This rule used to read those two lines --

    encoding mtmd batch, n_chunks = 1 (done = 1, total = 5)
    encoding mtmd batch, n_chunks = 1 (done = 3, total = 5)

-- as "two image encodes, so idefics3 tiles". `done` is the index of the CHUNK being evaluated in a
five-chunk prompt, not an encode counter, and `llama-mtmd-cli` does exactly ONE encode at every size
from 384x384 to 1536x512. This mmproj carries no splitting key at all. **A log line whose fields you
have not identified is not an observation; it is a guess with a timestamp** -- and it was made in the
same rule that congratulates itself for reading the artefact instead of the description.

**★ AND THE REAL DEFECT WAS ONE PERMUTATION IN THE PATCH GATHER, FOUND BY THE ORACLE THIS RULE
SHOULD HAVE BUILT FIRST.** `v.patch_embd.weight` is a conv2d kernel `[kx, ky, channel, NEmbd]` in
GGUF order, so a flattened patch is indexed `kx + ky*P + c*P^2` -- **CHANNEL-PLANAR**. jitllm's gather
copied contiguous runs of the preprocessed image, which is `[y][x][channel]`, giving
**PIXEL-INTERLEAVED**. The same 768 numbers in the wrong order:

    llama.cpp, quad512.png   "a colorful, abstract representation of a spectrum of colors"
    jitllm, before              "a scene from a historical or fictional narrative ... medieval"
    jitllm, after               "a colorful, abstract representation of a spectrum of colors"

Every shape right, every norm finite, determinism perfect, and a fluent description of a picture
nobody had shown it. RULE 7a exactly, one modality later.

**★ `llama-mtmd-debug` IS THE ORACLE AND IT HAD BEEN SITTING IN THE SAME BIN DIRECTORY.**
`-p encode --image cb` synthesises a checkerboard IN MEMORY as raw floats and runs clip's graph with
an eval callback on EVERY node, printing corner values and a whole-tensor sum. So the comparison has
no PNG decode, no resize and no mean/std normalisation in it -- three variables that would otherwise
be entangled with the tower's arithmetic, and two of which I spent hours eliminating by hand with
same-size test images. Its first node settles the bug on its own: `IM2COL` on a checkerboard prints
`1.0000, 0.0000, 1.0000`, which is channel-planar; interleaved would print `1, 1, 1`, because pixel
(0,0) is white on all three channels.

`model.TestTowerMatchesLlamaCpp` is that comparison, pinned: `node_395` (the projector output) at 36
sampled positions plus the sum, relative-RMS bar 0.05. Run against the violation -- the interleaved
gather -- it reads **RMS 0.5458 and a sum 74.9% off**; correct it reads well inside. The bar is
percent rather than ulp because jitllm quantizes activations to int8 per matmul (~0.4% each, RULE 7i)
where clip runs f32/f16, and twelve blocks of that compound to ~1.5%.

**★ AND THE SECOND HALF OF A VLM IS THE MARKER SEQUENCE, WHICH NO NUMERICAL GATE CAN SEE.** The
tower test passes unchanged with every marker wrong: the picture arrives correctly encoded and
UNANNOUNCED. mtmd wraps an idefics3 overview image in

    tok_ov_img_start = {"\n\n", "<fake_token_around_image>", "<global-img>"}
    tok_ov_img_end   = {"<fake_token_around_image>"}

resolved by `lookup_token`, which compares the DECODED piece. That distinction is load-bearing:
on a byte-level BPE vocabulary "\n\n" is stored as "\u010a\u010a", so a raw map lookup finds
nothing, and BPE-ENCODING the text does not have to reach the same split -- the turn text ends in a
space and the merge that joins it to a newline is a different token from the one the model was
trained to see. `tok.Vocab.ID` is jitllm's `lookup_token` and `cmd/jitllm.TestImageSpansMatchMtmd` pins
the seven-plus-ten ids. jitllm had `":"` where the GGUF's own chat template emits `": "` for a
text-led message, and no `"\n\n"` at all; fixing both took

    llama.cpp   "The image is a 2x2 grid of squares, each containing a different color."
    jitllm before  "The image is a 2D grid of squares, each square containing a different color."
    jitllm after   "The image is a 2x2 grid of squares. Each square contains a different color."

**BUILD THE ORACLE BEFORE THE BISECTION, NOT AFTER IT.** Every hour spent here before
`llama-mtmd-debug` was spent removing one variable at a time from a free-text comparison, and free
text cannot distinguish "the tower is wrong" from "the markers are wrong" from "the resize is
wrong". RULE 11's lesson in a new subsystem: measure the harness before the code it runs.

★ WHAT THIS COSTS: `mtmd_image_preprocessor_idefics3` is one of FOURTEEN preprocessor classes in
b10825, and it is now a named part of the triple RULE 7h refuses outside of. A family that DOES tile
changes the token count per image, so it changes what `PrefillMixed` splices as well as what the
tower sees -- preprocessing is not an internal detail of the tower, which is why the refusal is at
load and by name.

★ AND THE REFERENCE'S OWN ANSWER IS THE TEST FIXTURE: on a PNG of four coloured quadrants and a
disc, llama.cpp says "The image is a 2x2 grid of squares, each containing a different color." That
is correct about the picture, so the oracle is known-good before jitllm is compared against it.

## ★ RULE 7k: olmoe IS THE SEVENTH ARCHITECTURE. TWO GRAPH DIFFERENCES, AND THE
EXPENSIVE PART WAS A THIRD THING NEITHER OF THEM.

Read off `llama.cpp/src/models/olmoe.cpp`, not inferred. Against qwen3moe, which
jitllm already had, OLMoE-1B-7B differs in exactly two places:

  1. **The q/k RMSNorm spans the WHOLE PROJECTION.** llama.cpp creates
     `attn_q_norm` at `{n_embd}` and applies it BEFORE `ggml_reshape_3d` --
     `build_qkv(..., il, false)`'s trailing false is `reshape`, which is what
     makes it flat. qwen3 creates it at `{head_dim}` and applies it per head.
     Same tensor name, same position in the graph, different arithmetic.
  2. **The mixture weights are NOT renormalised.** `build_moe_ffn(...,
     LLM_FFN_SILU, false, ...)` where llama (Mixtral) and qwen3moe both pass
     true. Checked all three rather than assuming.

Everything else was already there: NEOX rope (`llama_model_rope_type` puts
OLMOE in the same group as QWEN3MOE and GEMMA), SiLU, softmax gating, no
attention biases, an untied `output.weight` read off the file (RULE 7c), and the
`olmo` pre-tokenizer, which `tok/bpe.go` already grouped with gpt-2.

    CPU 56.4 tok/s (six P-cores)      GPU 65.07 tok/s, 7/16 blocks + the head
    TestGreedyMatchesLlamaCpp: identical ids, and both tiers free-run to
    llama.cpp's own text ("...Paris. The currency of France is Euro.")

**★ THE COSTLY BUG WAS THE EXPERT WIDTH, AND IT WAS 8x.** `NFFNExp` came from
`expert_feed_forward_length` with a fallback of `NFFN / NExpertUsed`, which gave
128 against a real 1024. olmoe carries no such key AND does not follow that
fallback: `olmoe.cpp` builds its banks as `{n_embd, n_ff, n_expert}`, so its
per-expert width IS `feed_forward_length`. The fallback was a qwen3moe-shaped
guess wearing a general name. It now reads `blk.0.ffn_gate_exps.weight`'s own
`Dims[1]` -- **RULE 7h's rule one file over: identify a layout by its shape, not
by the convention its name implies.** It failed LOUDLY at load, which is the
only reason it cost an hour rather than a day.

**★ THE NO-RENORMALISE FLAG IS NEGATED ON PURPOSE.** `NoExpertNorm`, not
`ExpertNorm`. RULE 14b records that reading `expert_weights_norm` and defaulting
FALSE silently rescales every FFN by the top-k probability mass; the zero value
has to be what every other mixture wants. Same for `QKNormWide`.

**★ AND A WIDE NORM WITH GQA IS REFUSED, BECAUSE llama.cpp WOULD BE WRONG TOO.**
`attn_k_norm` at `{n_embd}` only describes the k projection while
`n_embd_gqa == n_embd`, which is true of this model (16/16 heads) and not of a
GQA one. Nothing in the file or the reference says what a narrower k norm should
be, so `loadConfig` errors rather than picking. A wrong norm width does not
fault; it answers fluently.

**★ BOTH FLAGS RUN AGAINST THEIR VIOLATIONS, PER RULE 12m:**

    renormalise the mixture     token 0 "ĠParis" -> "Ġa", margin 2.025   FAILS
    per-head norm instead       "the capital of France is the capital
                                of France is the capital..."             FAILS

**★★ AND THE SECOND ONE EXPOSED A FALSE GREEN IN THE ORACLE ITSELF, WHICH IS
WORTH MORE THAN THE ARCHITECTURE.** The wrong norm width diverged at token
**0** on a **0.025** tie. `TestGreedyMatchesLlamaCpp` excuses a narrow-margin
divergence and then `break`s -- correctly, because past a divergence the two
chains are on different inputs (RULE 12i) -- so the subtest compared **zero
tokens and reported PASS**, with a degenerate repeat loop sitting behind it.

**The coverage of that test is exactly the prefix BEFORE the first tie, and at
i == 0 that is nothing.** A tie at position 0 is now a `t.Errorf`. Every model
that ties legitimately does so later -- llama-2-7b at 6, Llama-3.2 at 19,
Qwen3 at 7, phi3 at 9 -- so all eleven still pass, and if some future model
genuinely ties at 0 it will say so loudly instead of quietly proving nothing.
RULE 14r for the fourth time: **a test that compared nothing must not be a green
line.**

**★ AND `kernels.HeadNorm` HAD NO HARDWARE TEST AT ALL.** qwen3's per-head q/k
norm shipped covered only by whole-model token ids -- the comparison RULE 7a
shows cannot localise anything -- and olmoe then asked the same kernel for ONE
group of 2048 where every caller before it asked for 16 of 128. There was
nothing to answer with. `TestHeadNorm` now holds it to a float64 reference on
PTX and SPIR-V at (16,128), (1,2048), (8,64) and (1,32); the two shapes normalise
by different denominators, so a kernel that ignored nHeads would pass one and
fail the other. The weights are deliberately not all ones, since a weight of 1
hides a kernel that drops it.

**★ THE DEVICE KERNEL CHANGE IS BYTE-IDENTICAL WHERE IT DOES NOT APPLY, AND THE
GOLDEN TEST IS WHAT INSISTED.** `ExpertWeights` gained the un-renormalised path,
which needs the FULL logit array -- the renormalised one is entitled to skip it
because Z cancels. Declaring that parameter unconditionally changed the emitted
kernel for every mixture jitllm already runs, and `TestDenseKernelsUnchanged`
caught the hash move. `pL` is now declared only when it is read, and the launch
passes it only then. RULE 7f's discipline: a new capability must cost exactly
nothing where it does not apply.

**★ AND THE TEACHER-FORCED SPREAD TRACKS EXPERT COUNT, NOT MODEL SIZE -- SO A
MIXTURE CAN NEVER BE A CLEAN CONTROL.** olmoe reads 2/12 at max|dlogit| 4.22
where a size-matched dense model (llama-2-7b Q4_K_M, same card, same partial
seam) reads 0.879. That looked like a defect in the new code and is not. All at
n=12, one run each:

    tinyllama q3_K_M        dense          0/12   0.382
    Qwen3-1.7B              dense          0/12   0.926
    gemma-2b                dense          0/12   1.13
    Qwen3-MoE-4x0.6B        4 experts      1/12   1.06
    olmoe-1b-7b            64 experts, k=8 2/12   4.22

**IT IS THE ROUTING, AND ROUTING IS DISCRETE.** Every other f32-against-f64
difference in this engine is smooth -- a slightly different sum gives a slightly
different logit. A router does a TOP-K, so when the 8th and 9th experts are
near-tied the two tiers select DIFFERENT experts and the FFN output changes by a
jump. 64 experts give far more chances to sit on such a tie than 4 do, and
whatever enters at layer 0 is then amplified by every host layer above it.

Four things were eliminated on the way, and each is worth not re-testing:

  - **it does not scale with the device block count** -- 4.22 at 1.7 blocks per
    token and 4.33 at 11.7, so it is not accumulated drift;
  - **it is not the output projection** -- `JITLLM_NO_HEAD_PLACE=1` reads 4.22
    with the head on the host, and the llama-2-7b control happened to place its
    head on the host, which is what made that comparison look cleaner than it
    was;
  - **it is not a wrong matvec** -- `verify -verify` recomputes every served
    matvec on the host and reports none;
  - **it is not either new feature** -- disabling them makes it WORSE (4/24 at
    9.06 and 15/24 at 16), and both new device kernels gate correct in isolation
    on PTX and SPIR-V.

**RULE 12q says a teacher-forced count is "a GATE ON THE CLEAN MODELS and not a
metric on the tie-prone ones", and this extends its list: tinyllama, gemma and
qwen3-1.7B are controls; EVERY mixture of experts is tie-prone by construction,
and the more experts it has the worse it gets.** Judge a MoE by ids and margins
(RULE 12ae) or by free-running text against llama.cpp -- olmoe's matches on both
tiers -- never by the count alone.


## ★ RULE 7q: THE qwen2 GATE EXISTS NOW AND PASSES, AND THE NEXT-MODEL
RECONNAISSANCE INVERTS TWO OF MY OWN COST ESTIMATES.

**★ 1. RULE 7f's UNGATEABLE FEATURE IS GATED.** That rule ends "it cannot be
gated without a qwen2 GGUF, which this box does not have", and RULE 7m then found
that `attn_output.bias` had been computed and discarded on every path. A real
qwen2 file is on the box now -- Qwen2-1.5B-Instruct-Q4_K_M, 941 MB, carrying
`blk.N.attn_{q,k,v}.bias` -- and `TestGreedyMatchesLlamaCpp` runs it:

    8 tokens identical to llama.cpp, then a 0.025 tie at token 8

Under RULE 7a's 0.5 tieMargin that is a PASS, and it is the first time the bias
path has been held to an independent implementation rather than to an injection
test of jitllm against itself. **Two years of "loaded if present" comments, one
download.**

**★ 2. gpt-oss IS NOT A GRAPH JOB, AND I SAID IT WAS.** I priced it as "SWA plus
attention sinks, low cost" off the KEY summary. The TENSOR TYPES say otherwise:

    ffn_{gate,up,down}_exps.weight   type 39 = MXFP4   <- the bulk of the model
    attn_{q,k}.weight                type  6 = Q5_0
    attn_output.weight               type 12 = Q4_K    (have)
    attn_v.weight                    type  8 = Q8_0    (have)

**Neither MXFP4 nor Q5_0 has a kernel**, so gpt-oss needs two new formats before
any graph work matters. It would still LOAD -- RULE 8 makes the float64 reference
a supported tier -- and run at roughly 0.5 tok/s. It also wants YaRN rope scaling
(`rope.scaling.type = yarn`, factor 32, original ctx 4096) and a
`post_attention_norm`. RULE 14a's method caught this in one command and my
summary-reading did not: **read the tensor table, not the key table.**

**★ 3. AND gemma3 IS CHEAPER THAN gemma2, WHICH IS THE OPPOSITE OF THE ORDER
ANYONE WOULD GUESS.** RULE 7 (original) names gemma2 as the scope-death case and
is specific about why: "alternating SWA masks, four norms per block, per-model-
size Q scaling, and tanh softcapping inside the attention loop". Read off the
files:

    gemma2   sliding_window 4096   attn_logit_softcapping 50, final 30
             post_attention_norm, post_ffw_norm
    gemma3   sliding_window 512    NO softcapping at all
             post_attention_norm, post_ffw_norm, attn_q_norm, attn_k_norm

**gemma3 DROPPED the softcapping**, and its q/k norm is the one qwen3 already
shipped (RULE 7b). So gemma3 is SWA plus two elementwise norms, no new kernels,
Q4_K_M throughout -- while gemma2 still carries the `tanh` in the attention inner
loop that RULE 7 refused. The newer architecture is the cheaper one, and a
roadmap ordered by release date would have taken the expensive one first.

★ WHAT THE FILES DO NOT CARRY: the alternating pattern. Neither header says which
layers are local and which are global, so it is architecture knowledge that has
to come off llama.cpp's running graph (RULE 14a) rather than out of the GGUF.

★ AND THE OTHER TWO ARE FURTHER OUT. qwen3next is a hybrid SSM (`ssm.conv_kernel`,
`ssm.state_size`, `ssm.time_step_rank`, 512 experts) -- a new operator family, not
a graph tweak. Mixtral-8x7B is arch `llama` with `expert_count 8` and should
already run, since expert loading is generic and outside the arch switch;
unverified because it is 20 GB and wants RULE 14h's `cap 24G` on a quiet box.


## ★ RULE 7l: THE THREE COPIES OF ATTENTION-PREP ARE ONE FUNCTION NOW, AND THE
PREFILL GATE THAT SHOULD HAVE CAUGHT A DRIFT IS OPT-IN AND MIS-CALIBRATED.

Adding olmoe (RULE 7k) meant editing the SAME q/k norm loop in `forward.go`,
`prefill.go` and `batch.go`. Missing one gives a model that DECODES correctly
and PREFILLS wrong -- which `prefill.go`'s own comment already names as "the
worst shape a bug can have" -- and RULE 7a is what happens when two copies of a
graph disagree. This was three.

`State.attnPrep` is the shared tail: the optional q/k RMSNorm, RoPE on q and k,
and the KV write. The three callers differ only in where their vectors live and
what element offset the cache write lands at. -63 lines, and prefill and batch
pick up the `opRoPE` accounting decode already had (tick/tock are no-ops when
profiling is off, so it costs nothing).

**★ AND THE GATE FOR IT, `TestPrefillMatchesForward`, IS OPT-IN -- SO NOBODY RUNS
IT.** It needs `JITLLM_BENCH_MODEL` and skips otherwise, which is how it sat
failing on three models. Worse, the path resolves against the TEST's working
directory, so `JITLLM_BENCH_MODEL=models/x.gguf` misses, `Open` fails, and
`t.Skip` reports **ok in 0.001s** -- RULE 7e recorded that exact trap and it
caught me again. Use `../models/`, and check the subtest RAN.

**★ ITS ID CHECK CONTRADICTED ITS OWN COMMENT, AND THE COMMENT WAS RIGHT.** The
paragraph above the loop says a flip inside the observed gap is "a property of
greedy search under quantization, not evidence about prefill"; the code then
called `t.Fatalf` on any flip at all. tinyllama failed there while its logits
agreed to NMSE 2.5e-4. It is judged BY THE MARGIN now, which is RULE 12ae's rule
already in use for the fp16 tensor cores: a flip where BOTH sides won by less
than the perturbation between the two logit vectors is the model being
undecided; a flip where either side was confident by more is a defect. tinyllama
now passes and says why -- margins 0.0143 and 0.0683 against a 0.2195
perturbation. **A gate whose comment and whose assertion disagree is the
assertion being wrong.**

**★★ AND THE NMSE TRIPWIRE WAS POINTING AT A REAL DEFECT ALL ALONG: PREFILL RAN
A DIFFERENT ATTENTION FROM DECODE, IN Go, AND IT COST 1.41x OF PREFILL.**

`TestPrefillMatchesForward` read NMSE 1.1e-3 to 2.9e-3 on three of eight models
and I nearly filed it as an unexplained calibration problem. It was not.

    scores      decode  fast.AttnScores  (EmitAttnScores, generated)
                prefill nn.Dot32         (a Go loop)
    V accum     decode  fast.AttnAcc     (EmitAttnAcc, generated)
                prefill nn.AxpyF32       (a Go loop)

Two implementations of one operation, so the two paths computed the same
attention by different arithmetic. **RULE 7l's own lesson, one level down: the
three copies of attention-PREP were merged and the two copies of attention
ITSELF were not.**

**★ THE MEASUREMENT THAT FOUND IT WAS NMSE AGAINST PROMPT LENGTH, AND EVERY
CORRELATION I TRIED FIRST WAS COINCIDENCE.** rope_base looked perfect -- the
three failures were the only models at 500000 and 1e6 -- and depth, the q/k norm
and tied embeddings all had a story too. All four were wrong. The sweep settles
it in one line, because **n=1 reads exactly 0.000e+00**, which eliminates every
per-token path at once (projections, norms, RoPE, the KV write, the FFN) and
leaves only attention over multiple positions:

    n            1       2       4       8      16      32      38
    before     0.0    1.9e-3  9.4e-4  9.2e-4  2.0e-3  2.3e-3  2.9e-3
    after      0.0     0.0     0.0    1.6e-3  9.6e-4  7.7e-4  5.9e-3
    after, JITLLM_NO_GEMM=1:  0.0 at EVERY length

**IT HID AT ONE TOKEN BECAUSE A ONE-ELEMENT SOFTMAX IS 1.0 WHATEVER THE SCORE.**
The entire scores difference is erased at n=1 and appears at n=2. A gate that
only ever ran one length would never have seen it -- and this one runs 38.

**★ SO THE REMAINDER IS ONE COMPONENT AND THE GATE IS EQUALITY NOW.** With
`JITLLM_NO_GEMM=1`, Prefill is **BIT-IDENTICAL** to a Forward loop on all eight
models -- 0.000e+00, not "close". What is left in the shipped path is
`nn.JIT.MatMul` summing a tile where `MatVec` sums a row, and it appears at n=8,
which is exactly where MatMul stops declining for padding (RULE 13a). The test
runs both modes: `no-gemm` demands equality, `shipped` keeps a gross-error
tripwire. **An NMSE bar could not have caught this -- 1.1e-3 reads as rounding.
Equality reads as a bug.** This is RULE 12u-bis's shape a second time: pin the
one component that legitimately reassociates, and then demand bit equality of
everything else.

**★ THE BAR MOVED TO 1e-2, AND ONLY BECAUSE THE CAUSE IS KNOWN.** 1e-3 was
derived from ONE matmul ("int8 quantization costs 4.5e-5, so an order above is
1e-3") and a forward pass is L layers of seven, so reassociation compounds;
measured, it spans 2.5e-5 to 3.1e-3 and grows with prompt length. Two orders
below the NMSE 1e0 a real packing bug produced. **Moving a correctness threshold
to make a test pass is what this file exists to prevent, and the thing that makes
it legitimate here is that the sharp gate replacing it is STRICTER, not looser.**

**★ AND IT IS WORTH 1.4147x OF PREFILL, WITH THE CONTROL THAT PROVES THE
MECHANISM.** Qwen3-1.7B, six P-cores, ABBA, alternating binaries:

    691-token prompt   old 88.16 87.14 88.06   new 124.72 122.06 124.92
                       median of per-round ratios 1.4147, spread 0.0177, n=3
    61-token prompt    old 147.0   new 147.3   ratio ~1.002   <- FLAT

**Attention is O(n^2), so its share of prefill grows with the prompt, and the
effect tracks that exactly: nothing at 61 tokens, 41% at 691.** A change that
measured the same at both lengths would have been something else. This is also
why it never showed up in the prefill work of RULES 1d-sexies through 1d-nonies:
those profiled gemma at 469 tokens and read "94.7% matvec", and the missing
kernels were inside the other 5.3%.

★ AND `nn.ResetForTest` RESET jitOnce AND jitDisabled AND NOTHING ELSE. RULE 14f
is about this exact function -- it reset the Once and left `fastDisabled` true,
so every later NewJIT returned nil and the suite stayed green while two models
decoded at 0.3 tok/s. That was fixed for the one variable that existed then;
`profOn`, `tuneQuiet` and `noGEMM` were added later and never added here, so a
test that sets JITLLM_NO_GEMM and resets gets the PREVIOUS run's answer. **A
reset function is a list that silently omits, exactly like RULE 6c's per-format
switch.** All four now.


## ★ RULE 7m: FOUR DEFECTS OF ONE CLASS, FOUND BY ASKING A REVIEWER FOR THE CLASS
RATHER THAN FOR A REVIEW. THE BIAS FEATURE HAD NEVER RUN CORRECTLY.

RULE 7l merged the three copies of attention-PREP and RULE 7k's fix merged the
two copies of attention ITSELF. Rather than look for more by hand, a reviewer was
given the SHAPE of the defect -- three transcriptions of one graph, one using a
generated kernel its siblings do not -- and asked to find others. It found five,
four of them real, in one pass. **Ask for the class, not for a review.**

**★ 1. attn_output.bias NEVER REACHED THE MODEL, IN ANY TEXT PATH.**

    forward.go   mv(s.h, wo) -> s.x += s.h -> addBias(s.h, bo)   <- into a buffer
                 the next RMSNorm overwrites. Dead.
    prefill.go   the same, AND after batchNorm had already clobbered s.bh, AND
                 only on the mixture-of-experts branch. Dead twice over.
    batch.go     no biases at all -- not q, not k, not v, not o.
    vision.go    mm -> addBiasRows -> residual.   ★ CORRECT ALL ALONG.

So `ForwardBatch` gave a qwen2-family model different attention AND a different
KV cache than `Forward` did, and every path computed `attn_output.bias` and threw
it away.

**★ THE ROOT CAUSE IS IN RULE 7f's OWN TEXT.** It says: "PRESERVATION IS THE
LOAD-BEARING GATE, NOT THE NEW BEHAVIOUR. Every model jitllm already runs has no
biases, so the addend must cost exactly nothing there", and "it cannot be gated
without a qwen2 GGUF, which this box does not have". **A feature verified only to
do NOTHING where it does not apply is not verified.** The preservation check was
right to exist and was the whole of the evidence.

**★ AND THE GATE NEEDS NO BIASED MODEL, WHICH IS WHY IT EXISTS NOW.** A bias is a
vector, so INJECT one. `TestAttentionBiasesReachTheModel` sets each of the four
in turn and asserts the logits MOVE:

    attn_q.bias 0.199   attn_k.bias 0.108   attn_v.bias 0.835   attn_output 2.401

**Each one separately, because they enter at different points and a shared "some
bias moved the logits" check passes with three of the four dead.** Run against
the shipped ordering it reads *"attn_output.bias changed no logit: it is being
computed and discarded"*. `TestBiasedBatchMatchesForward` covers the other two
paths -- biased batch-of-one and biased prefill are both BIT-IDENTICAL to
Forward, and against the shipped batch they read NMSE 1.4e-1 and 2.7e-1.

★ NOTE stories260K PASSES that violation at 5.2e-4. A 260K toy model cannot
resolve a dropped bias, which is why `models` carries stories15M and a real MoE.

**★ 2. Prefill AFTER Forward RESTARTED AT THE WRONG POSITION, WHICH IS THE
MULTI-TURN SHAPE.** `Forward` did `s.pos++`; `Prefill` reads `bpos`. `advance(slot,
n)` is documented as one of "the two places that know a position may be per row"
and Forward was the one caller not going through it. So prompt / generate /
next-prompt overwrote the generated tokens' cache entries and rotated the new
ones to the wrong positions:

    2 Forward + 3 Prefill  ->  Pos() 3, want 5,  max|dlogit| up to 11.8

**Nothing in the suite mixed the two calls on one session**, which is the whole
reason it survived. `TestPrefillAfterForwardContinues` does, and with the GEMM
pinned out it is bit equality.

**★ 3. THE VISION FALLBACK SKIPPED THE GENERATED MatVec** -- straight to
`nn.MatVec`, the single-core float64 reference, whenever the batched GEMM
declines. Its comment said "exactly as model.State.mm does it" and `State.mm`
falls back through `s.mv`, which tries `JIT.MatVec` first. **RULE 15f: a comment
claiming parity with another path is a claim ABOUT that path, and it rots.** The
F32 case is concrete -- the GEMM has no F32 emitter and both CPU architectures
have a generated F32 MatVec since RULE 14u, so the tower's F32 matrices took the
reference every time.

**★ 4. LATENT, AND FIXED BEFORE IT LANDED: prefill passed the WHOLE RoPE TABLE.**
`s.cs` is `nseq*NRot` and `Rope.Table` fills the first NRot; `nn/ref.go` derives
the rotation width from `len(cs)` capped at the head width. forward.go:462 and
batch.go both slice one row; prefill did not. Every model here has NRot ==
HeadDim so the cap hides it -- **which is the kind that surfaces on the next
architecture**, exactly as olmoe surfaced the expert-width fallback.

★ TWO REVIEW FINDINGS NOT ACTED ON, recorded so they are not re-derived: `moe.go`
accumulates experts into zero in selected-probability order at line 97 and into
the residual in ascending expert-id order at line 238, so the two differ by
summation order AND parenthesisation -- a numerical inconsistency, not a wrong
answer, and it means RULE 7l's "the GEMM is the only remaining reassociation"
does not generalise to MoE. And `prefill.go`'s device-failure path does not
reload its input after `demoteAll` where `forward.go:1035` does, so a chunk that
fails partway restarts at CPU layer zero over rows already carrying the device
prefix's output.


## ★ RULE 7n: TWO PREFILL ROWS CROSSED FROM LOSS TO WIN, AND EVERY PREFILL ROW
IN THIS FILE WAS MEASURED WITH THE SLOW ATTENTION.

RULE 7m's attention fix -- prefill running the generated kernels instead of Go
loops -- was worth 1.4147x in isolation on a 691-token prompt. On the actual
scoreboard, at the 507 tokens `scripts/vs-llamacpp.sh` uses:

    model              prefill vs llama.cpp, CPU        was      now
      tinyllama q3_K_M   1.3171 / 1.3360 / 1.3290      0.8720   ★ WIN
      gemma-2b           1.0307                        0.8555   ★ WIN
      llama-2-7b Q4_K_M  1.0159 / 1.0145               0.8729   ★ WIN

**ALL THREE CPU PREFILL ROWS WIN NOW, AND ALL THREE WERE LOSSES.** That is the
whole of the CPU prefill board on this host.

**THREE PASSES ON tinyllama, BECAUSE RULE 2c SAYS A BIAS COMMON TO BOTH ARMS
SURVIVES THE IQR GATE.** 1.3171 (IQR 0.034), 1.3360 (0.032) and 1.3290 (0.018)
agree within 1.4%. gemma is one pass at IQR 0.066. Quote 1.32 for tinyllama.

**★ THE 7B ROW TOOK FOUR ATTEMPTS, AND THE FIX IS ONE LINE OF PYTHON WORTH
KNOWING.** Three runs were killed by the memory watchdog during the SOAK, before
any round: the script alternates two engines each mapping 3.8 GB, and a day of
model loads had left 18 GB of PAGE CACHE with 2 GB free. Fewer rounds did not
help, because the pressure is the mapping, not the duration.

`posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` drops a FILE's page cache and
needs no privilege -- only read access. Over every GGUF on the box:

    free 2 GiB -> 16 GiB, buff/cache 18 GiB -> 4 GiB, and the row ran first try

**This is the same lever `gguf.Evict` uses in-process (MADV_DONTNEED), pointed at
the files rather than at a mapping.** It costs the next run its cold reads, which
the soak absorbs and both engines pay equally. **Reach for it before concluding
a measurement is impossible on this hardware** -- RULE 1d-ter's "unmeasurable"
verdict is right for a contended GPU and was wrong here.

★ AND THE 7B's IQR IS 0.009, THE TIGHTEST ROW IN THIS FILE. A 7B prefills at
36-37 tok/s where tinyllama does 150, so each sample is long enough to average
its own noise -- the opposite end of RULE 1d-quater's "short samples flatter
jitllm".

**★ AND THE ROTARY TABLE FIX MEASURES ZERO, WHICH IS RULE 5z FOR THE THIRD
TIME.** Prefill rebuilt the rotary table for every (layer, token) pair while the
device path built it once per chunk into a buffer -- `s.bcs` -- that the host
loop ignored. 22 x 507 fills where 507 do, on tinyllama. I estimated it as a
concrete win; it is 32 sin/cos pairs x 11,154 builds, about 7 ms against a 2.6 s
prefill, or **0.3%**, and the row reads 1.3290 between the two attention-only
passes at 1.3171 and 1.3360. **Invisible.**

Kept anyway, per RULE 6a: it deletes real work, costs nothing, and it subsumes a
latent bug rather than patching it -- prefill was passing the whole `nseq*NRot`
`s.cs` where `Rope.Table` fills only the first NRot and `nn/ref.go` derives the
rotation width from `len(cs)`. Every model here has NRot == HeadDim so the cap
hid it, which is the kind that surfaces on the next architecture. **No rate is
claimed for this change.**

★ THE STANDING LESSON, NOW THREE FOR THREE: a count of removed operations is not
a share of the token. The LDP pairing predicted 4.1% and delivered 1.0% (RULE
5z); the probe coefficients overstated every lever in RULE 5y's table; this
predicted "concrete" and delivered nothing measurable. **Price a removal against
the token, not against itself.**


## ★ RULE 7o: A PREFILL CHUNK THAT FAILED PARTWAY RAN THE DEVICE'S BLOCKS TWICE,
AND THE FAULT INJECTOR COULD NOT REACH IT.

`forward.go` calls `fill()` a second time after `demoteAll`, and says why: "Layers
may have written part of the residual stream before it failed. This is the second
call to fill and the reason it is a closure." `prefill.go` had no equivalent. Its
chunk flow is a batched device call, then a PER-TOKEN retry, then the host from
`li0`:

  - the batched call can write some rows before failing, and the retry then
    re-ran those rows through the same blocks;
  - a retry that fails at row i leaves rows 0..i-1 carrying the device prefix's
    output, and the host restart runs ALL layers over them from zero.

Both are the device's blocks applied twice. `refill()` is the closure now, called
before the retry and again before the host restart.

**★ AND THE INJECTOR WAS SINGLE-SHOT, SO THE PATH WAS UNREACHABLE.** RULE 14p
built `JITLLM_DEVICE_FAIL_AT` precisely because "error handling that has never run
is not error handling" -- and `injectedFail` compared `g.calls != g.failAt`, one
call, firing at the TOP of `Layers` before any write. That can produce neither a
partial write nor the SECOND failure this path needs: the batched call fails
(call 1), the first retry row succeeds (2), a later one fails (3). It takes a
LIST. `JITLLM_DEVICE_FAIL_AT=1,3` now does, and the repro is unambiguous:

    reference (cpu)         1 450 7483 310 3444 338 3681 29889 13 13 ...
    before, FAIL_AT=1,3     1 450 7483 310 3444 338 30191 30191 30191 30191 ...
    after,  FAIL_AT=1,3     identical to the reference        (also 1,4 and 1,5)

**A fault injector that can only express ONE failure can only test the paths one
failure reaches.** RULE 14p's own sentence, one level down -- the facility
existed, was correct, and was too weak for the path that most needed it.

★ AND THE STALE `jpt` AND `jpt-gpu` BINARIES WERE TRACKED IN GIT, 6.5 MB each,
left from before the rename. Removed and gitignored. The seven files that were
not gofmt-clean are clean, and `usage()` had drifted -- it listed neither
`-mmproj`, `-image`, `-depth`, `-gpu-grow` nor any of the seven sampling flags,
all of which `runCmd` has parsed for some time.


## ★ RULE 7p: TWO MORE OF THE SAME CLASS, AND THE PREFILL PROFILE WAS MEASURING
ALMOST NOTHING.

**★ 1. A FULLY-RESIDENT TOKEN THAT LOST ITS DEVICE RETURNED THE PREVIOUS TOKEN'S
LOGITS.** `folded` is decided BEFORE the device call -- `s.head != nil &&
s.gpuLayers == len(m.layers)` -- and the failure branch runs `demoteAll`, which
sets `s.head` to nil and `s.gpuLayers` to zero. The local stayed true, so the
token took the early return that says "the device ran the output norm and the
projection too" and handed back `s.logits` that the failed call never wrote.

    reference (cpu)      ... 3444 338 3681 29889 13 13 29906 ...
    before, FAIL_AT=3    ... 3444 338 3681 3681 29889 13 13 ...    <- token repeated
    before, FAIL_AT=4    ... 3444 338 3681 29889 29889 29871 ...   <- and again
    after                identical to the reference at 2, 3 and 4

**The symptom is a DUPLICATED token, which reads as a sampling quirk rather than
a device fault.** One line: the branch clears `folded` now.

**★ 2. THE PREFILL PROFILE TIMED ONE OP OUT OF SIX, AND DOUBLE-CHARGED IT.**
`prefill.go` called `tock` in exactly one place -- `mm` -- so the norms,
attention, the activation and the residuals were invisible, and every percentage
was a share of a total that was almost entirely matvec BY CONSTRUCTION. Worse,
`mm` held a `defer tock(opMatVec)` over the whole call while the fallback's
`s.mv` opened its own window, so a declined GEMM counted twice.

    gemma-2b, 492-token prompt, after:
      matvec 94.2%  attention 3.1%  activation 2.2%  rmsnorm 0.3%
      rope 0.1%  residual 0.1%

**★ AND THAT 3.1% IS THE RECEIPT FOR RULE 7m.** Attention is 3.1% of prefill NOW.
For the 1.4147x of RULE 7n to be real it has to have been about 31% before, and
it was: 1.41 - (1 - 0.031) = 0.44 of the old token. **The number the old profile
reported -- "94.7% matvec" -- was very nearly RIGHT, and that is exactly why it
was dangerous.** The remaining 5.3% was not small; it was UNMEASURED, and a
31%-of-token defect was sitting inside it. RULE 11b's blind profiler in the other
half of the engine, and RULE 11's own sentence: measure the harness before the
code it runs.

★ AND I INTRODUCED ONE OF THE DOUBLE-COUNTS MYSELF, ONE COMMIT EARLIER. RULE 7l
moved the RoPE timer inside `attnPrep`; `batch.go` kept its outer `tock(opRoPE)`
around the loop that calls it, so batch counted RoPE twice and labelled the q/k
norm and the KV write as rotary. **A shared helper that carries its own timer
makes every caller's timer a double-count.**


## ★ RULE 13: BATCHED DECODE IS 2.0-2.7x AGGREGATE, AND THE CEILING IS THE GEMM, NOT THE WIDTH.

`Model.NewBatch(nseq, maxSeq)` and `State.ForwardBatch(tokens)` step nseq INDEPENDENT sequences
through one pass over the weights. The graph is Prefill's -- every matvec becomes a matmul over the
batch -- and the one op that differs is attention, which differs in the opposite direction: prefill's
rows share a cache and a causal bound, a batch's rows each attend into their OWN cache, which is a
base offset. The sequences share a position counter, so a batch is nseq conversations of equal
length; ragged admission needs a position per row and a scheduler above this.

    tinyllama q3_K_M, Linux, six P-cores, depth 0, two gated passes of scripts' in-process A/B

      N=8    single 80.8 / 80.4 tok/s    batch 164.9 / 161.6 aggregate    2.04x / 2.01x
      N=16   single 78.5 / 73.7          batch 202.3 / 197.0              2.58x / 2.67x
      N=32   single 76.0 / 71.1          batch 189.9 / 189.0              2.50x / 2.66x

**IT PLATEAUS AT SIXTEEN AND THE REASON IS ARITHMETIC, NOT DISPATCH OR MEMORY.** `JITLLM_PROFILE=1`
puts matvec at 93.9% of a batched step against 95.5% of a single token, so nothing else moved and
there is no harness to find. At N=8 the step spends 45.3 ms on 8.8 GMAC = 194 Gmac/s while reading
the same 0.49 GB it always reads -- 10.8 GB/s, a quarter of the wall. **The batch is compute-bound
where decode is memory-bound**, so the crossover is where the GEMM's rate meets the DRAM one and
there is nothing after it:

    batch time per token = MACs per token / GEMM Gmac/s        <- flat in N
    decode time per token = weight bytes / 39 GB/s

`TestABMatMulVsMatVec` says the same thing in isolation and says it is not a width problem: the
batched GEMM is **1.79x - 2.11x the per-token matvec loop at EVERY width from 8 to 128**, and its
absolute rate is flat (Q8_0 311 Gmac/s, Q4_0 305, Q3_K 216 at six cores). The tile is 8*nr tokens
wide and ntok beyond that is just more tiles, so only `nr` amortizes anything -- and nr=1 to nr=2
buys 16% on Q3_K (187 -> 216 Gmac/s) while nr=2 to nr=4 buys nothing measurable.

**SO THE ONE LEVER LEFT IS THE GEMM KERNEL, AND IT HAS ROOM: 216 Gmac/s IS ~29% OF VNNI PEAK.**
`VPDPBUSD` is 32 int8 MACs per instruction on two ports, so six P-cores at 3.5 GHz is ~1300 Gmac/s
of issue. It scales cleanly across cores (Q3_K 42 / 77 / 183 Gmac/s at 1 / 2 / 6), so the deficit is
per-core issue and not the pool. Anything that raises that number raises every row above it, and
nothing else will.

**★ 13a. THE TILE TUNER NOW ASKS AT THE BATCH WIDTH, WHICH IS WORTH 32% AT N=8.** `rateKey` carried
(type, blocks-per-row, rows) and NOT ntok, so a batched decode inherited the tile prefill had tuned
at 32 tokens or more. A tile wider than the batch is zero-padded and does the missing columns'
arithmetic anyway, so that is a measured loss proposed as an answer. `gemmTile` takes ntok, keys on
it, and `fitTile` narrows every CANDIDATE to a width the batch can fill before the tuner ever
proposes it.

    N=8 aggregate, before 1.54x  ->  after 2.04x

`MatMul` then declines outright when even the narrowest tile is more than half padding, which is
every batch under eight, because declining is a supported answer and the caller loops MatVec:

    N=2  0.40x   N=4  0.76x   N=8  1.54x    <- the three ran the SAME eight-wide tile, so their
                                               per-step cost was identical (0.195 / 0.190 / 0.204)
                                               and only the useful fraction differed

The bar is HALF padding rather than any padding: a seven-token prefill tail wastes one column of
eight and still wins, and refusing it would be a prefill regression bought with nothing.

**★ 13b. THE CORRECTNESS GATE HAS TO GIVE EVERY ROW A DIFFERENT TOKEN STREAM, AND THE FIRST VERSION
DID NOT.** A batch of identical sequences agrees with the single path even when every row reads
sequence 0's cache -- which is the one thing this code adds. `TestBatchMatchesForward` runs three
distinct streams teacher-forced against three separate `State`s and compares argmax ids per
position, plus NMSE on the last step against `TestPrefillMatchesForward`'s 1e-3 bar and for its
reasons. Run against the violation (`base := 0`) it reads NMSE 6.6e-1 with wide-margin id
disagreements, against 1.1e-4 correct. **A test for an invariant is worth nothing until it has been
run against a violation** -- RULE 12m's lesson, and it caught nothing here only because it was
applied first.

**Not built, and each is a real piece of work rather than an oversight.** Ragged batching -- a
position per row, admission and retirement mid-flight -- is the serving feature and needs a
scheduler; `ForwardBatch` refuses a device session outright (`errBatchDevice`) because the accelerator
holds one sequence's KV cache and batching onto it would attend to the wrong history, which is
fluent nonsense rather than an error. `Forward` and `Prefill` refuse a batch session for the mirror
reason: they would fill sequence 0's cache only.

## ★ RULE 14: MIXTURE OF EXPERTS. qwen3moe IS THE SIXTH ARCHITECTURE, AND THE GRAPH CAME OFF THE WIRE.

`Model.NewBatch`-style scope creep is what RULE 7 exists to prevent, so the decision first: qwen3moe
is qwen3's attention EXACTLY -- NEOX rope plus the head_dim-wide RMSNorm on each head of q and k,
both already shipped -- with the dense FFN replaced by a router and NExpert experts. The attention
half costs nothing. RULE 7b said the expensive part of a new architecture is never the arithmetic;
this one is the third confirmation.

**★ 14a. THE GRAPH WAS TRANSCRIBED FROM llama.cpp's RUNNING GRAPH, NOT FROM ITS SOURCE, AND THAT IS
NOW THE METHOD.** `llama-eval-callback -m <a 4-expert qwen3moe>` prints every node b10825 actually
executes AND its values:

    ffn_moe_logits    MUL_MAT(ffn_gate_inp.weight, ffn_norm)
    ffn_moe_probs     SOFT_MAX(logits)              <- over ALL experts, BEFORE top-k
    ffn_moe_argsort   ARGSORT(probs, DESC)
    ffn_moe_topk      VIEW(argsort)[:n_expert_used]
    ffn_moe_weights   GET_ROWS(probs, topk)
    ..._sum/_clamped/_norm   SUM_ROWS -> CLAMP -> DIV   <- the renormalisation IS on
    ffn_moe_gate/up   MUL_MAT_ID(ffn_{gate,up}_exps, ffn_norm)
    ffn_moe_swiglu    SWIGLU(gate, up)
    ffn_moe_down      MUL_MAT_ID(ffn_down_exps, swiglu)
    ffn_moe_weighted  MUL(down, weights_norm), summed, added to ffn_inp

with real numbers attached: probs [0.1381, 0.3259, 0.1835, 0.3525] select experts 3 and 1, and the
renormalised weights are [0.5196, 0.4804] = 0.3525/(0.3525+0.3259). **Two local llama.cpp checkouts
did not contain qwen3moe at all**, and a subagent that claimed to have read `src/models/qwen3moe.cpp`
had cited a file that does not exist here. The binary is the thing jitllm is actually held to; read it
instead of its source, and the citation problem disappears.

**★ 14b. THE SOFTMAX/TOP-K ORDER IS FREE, AND ONLY THE RENORMALISATION IS REAL.** This was written
into the code as one of the two lines a from-source reading gets wrong, put into a fault table as a
must-fail case, and MEASURED AT 3.469e-18. Softmaxing the whole logit vector and renormalising the
selected k is ALGEBRAICALLY IDENTICAL to softmaxing the selected logits:

    p_e / SUM_{s in S} p_s  =  exp(z_e) / SUM_{s in S} exp(z_s)

and top-k by p is top-k by z because softmax is monotonic. It is kept as a CONTROL, because the
order looks load-bearing -- llama.cpp's graph is emphatic about softmaxing first -- and a future
reader who "fixes" it would be changing nothing while believing otherwise.

The renormalisation is the one that matters, at 8.970e-03. **qwen3moe carries no
`expert_weights_norm` key at all** -- b10825 does not print one for this architecture -- so reading
the key and defaulting false skips the division and rescales every FFN output by the top-k
probability mass. `expert_weights_scale` is the same trap in the other direction: read it, default
0.0, multiply, and the whole FFN is zero.

**★ 14c. THREE WAYS TO BE SILENTLY WRONG, ALL CLOSED AT LOAD.**

1. **`get()` reported ne1 as the row count, and for a 3-D tensor that is a LIE.** An expert bank is
   [n_embd, n_ff_exp, n_expert]: ne1 describes ONE expert while `data` holds all of them, and
   `rows()` bounds-checks against `len(data)` rather than against `t.rows`. A bank handed straight
   to a matvec computes expert 0 for every token of every layer -- no error, no bounds panic, and
   the reference tier and the generated kernels both compute expert 0 and agree to the bit. **That
   is RULE 7a's exact shape.** `rows` is now the product of every trailing dimension, so the same
   mistake asks for 98304 rows and fails loudly. 2-D tensors are unaffected.
2. **`bankExpert` checks the EXACT BYTE TOTAL, not the dimensions.** Slack means the layout is not
   what the slicing assumes, and the slice would land inside the mapping.
3. **down's ne0 is NOT gate's.** gate and up are [n_embd, n_ff_exp]; down is [n_ff_exp, n_embd] --
   2048 against 768 on the 30B. One stride formula reused for all three reads another expert's
   weights, lands inside the mapping, and never faults. Measured as a fault at 3.591e-02.

**★ 14d. THE ORACLE CANNOT SEE A ROUTING BUG, SO THE ROUTER IS CHECKED DIRECTLY.** This is the part
to keep. `TestGreedyMatchesLlamaCpp` holds jitllm to llama.cpp on two qwen3moe files and NEITHER can
catch a mis-routed token: Qwen3-MOE-4x0.6B is four near-duplicate clones of one 0.6B model, so which
experts are selected barely moves the logits, and tiny-qwen3moe has random weights, so every token
is a near-tie and the oracle's own `tieMargin` escape swallows the disagreement. **A green oracle is
evidence about shapes, strides and kernels, not about the router.**

`TestMoEFaultsAreVisible` is the gate that is. A mirror of `moe()` must first reproduce it bit for
bit, then once per way of getting it wrong:

    softmax over the selected k (identity)      3.469e-18   CONTROL
    experts accumulated in reverse order        0.000e+00   CONTROL
    lowest-k experts instead of highest         4.092e-02
    down taken from the next expert             3.591e-02
    expert 0 for every token                    1.250e-02
    top-k weights not renormalised              8.970e-03

Two controls, not one: RULE 12m's lesson is that a gate has to be run against a violation, and a
gate with no must-PASS case is a gate that might simply be hypersensitive.

**★ 14e. WHAT MADE THE DIFF SMALL, AND WHAT DID NOT NEED BUILDING.** `model.rows()` already slices a
contiguous row range out of the mmap without copying -- RULE 7c-bis built it for phi3's fused
attn_qkv -- and a GGUF 3-D bank is expert-major, so expert e needs no new machinery and the mapping
stays the only storage. `l.router.data != nil` is the graph branch, per RULE 7c: the presence of the
tensor is the fact and the config field is a guess about it. The activation is quantized ONCE for
the router and all k experts, since they read the same normed vector. **And RULE 10's UNVERIFIED claim -- "all experts in a
layer share their shapes, so 128 experts cost what one does" -- is now VERIFIED, and by a stronger
mechanism than it assumed.** Counting distinct (type, blocks-per-row) pairs, which is what the
kernel cache is keyed on:

    Qwen3-MOE-4x0.6B   4 experts, 28 layers, 226 matrices  ->  6 distinct (type, k) pairs
    Qwen3-1.7B dense   0 experts, 28 layers, 197 matrices  ->  4

The experts are STACKED into one 3-D tensor per role, so the expert count never appears in the
inventory at all -- it is not that N experts happen to share a shape, it is that N experts are one
tensor. The two extra pairs are this model having both k=1024 and k=3072 shapes plus an F32 router,
and have nothing to do with NExpert.

Note in passing that `State.Shapes()` reports 0 for BOTH models and is the wrong probe: `AddShape`
returns early when `BestPackNative(t) < 2`, and BestPack is 1 for every k-quant, so an all-k-quant
model has an empty shape inventory whether or not it is a mixture of experts.

**`BytesPerToken` counts NExpertUsed of NExpert**, which is the whole point of the architecture.
Counting every expert overstates Qwen3-30B-A3B by about ten times and would put its GB/s an order of
magnitude above the wall -- RULE 8b's denominator lesson, one architecture later.

**★ 14j. MoE NOW RUNS ON THE GPU, AND `MUL_MAT_ID` NEEDED NO NEW IR OP.** The paragraph below
saying "the device tier refuses MoE blocks outright" was true and is not any more. An expert bank is
`NExpert*Rows` rows of ONE matrix -- a GGUF 3-D expert tensor is already expert-major, and RULE 14c's
`rows` fix makes `Weight.Rows` the whole bank -- so selecting an expert is ADDRESS ARITHMETIC, not
control flow. `ir.Builder.Load` already took a runtime SSA index (`kernels/attn.go:71` is the shipped
precedent), and every thread of one expert's matvec reads the SAME expert, so the index is uniform
and nothing diverges.

    Qwen3-MOE-4x0.6B   28/28 blocks + head on the device
      CPU  69.1 tok/s      GPU 125.8 tok/s     1.82x
      teacher-forced 0/32, 0/32, 2/32 across three runs, max|dlogit| 0.395-0.472

**★ AND IT RUNS ON BOTH HOSTS AND ALL THREE BACKENDS.** The kernels needed no per-target work:
`TestIndexedMatVecMatchesDense` and `TestExpertRouter` pass on PTX, SPIR-V and MSL, including every
constructed tie case, and the M4 was never touched during development.

    Qwen3-MOE-4x0.6B     CPU      GPU      ratio
      Linux, CUDA        69.1    125.8     1.82x
      M4, Metal          80.8     98.8     1.22x

The Mac's free-running completion is BYTE-IDENTICAL to the Linux CPU tier's for 20 tokens, which is
worth more than either rate: same ids across two architectures, two backends and two float paths.
Teacher-forced it reads 2-3/32 with max|dlogit| 0.74-0.93, which is Metal's ordinary band (gemma is
0/32 at 1.58) and is RULE 12q's non-determinism, not a defect.

**★ 14l. AND THE HOST-TO-DEVICE LINK IS 4x FASTER ON THE MAC, WHICH IS WHY PLACEMENT MUST BE PROBED
AND NOT ASSUMED.** `TestPCIeBandwidth` on both:

    Linux, PCIe 3.0 x16 (CUDA)     8.14 GB/s   against ~39 GB/s achieved host reads   = 4.8x slower
    M4, unified memory (Metal)    33.35 GB/s   against ~50 GB/s achieved host reads   = 1.5x slower

**That inverts a design decision.** Paging a weight in on a miss has to beat reading it on the host,
so it breaks even at a hit rate of `1 - (1/hostRead - 1/devRead) * link`:

    Linux   break-even 84.6%      measured adapted coverage at r=32: 72.6%   ->  paging LOSES
    M4      break-even 70.3%                                                 ->  paging WINS

The same mechanism is wrong on one box and right on the other, and only by a few points -- so a
policy that hardcodes either answer is wrong half the time. **Probe the link, the host read rate and
the device read rate at load, and let the numbers choose**, the way `tier/choose.go` already probes
which GPU API to use. Apple's memory is unified, so the "transfer" is a layout conversion rather than
a transport (RULE 12o's packing); a future host with PCIe 5.0 moves the ratio again.

`MatVecShape` gains `Experts`, `Slots` and `SlotAct`; at `Experts <= 1` the kernel is BYTE-IDENTICAL,
which `TestDenseKernelsUnchanged` holds it to. Partial offload works unchanged -- at `-vram`
100/300/600 MB the same file puts 3.8, 12.5 and 25.0 blocks per token on the card -- so there is
still no placement policy for MoE beyond `PrepLayer` declining per block.

**★ THE THREE THINGS THAT WERE HARD ARE ALL ADDRESSING, AND EVERY ONE FAILS FLUENTLY.**

1. **BOTH HALVES of the weight address change.** `pack.go` writes with stride `(...)*nrows + r`, so a
   bank needs stride `Experts*Rows` AND index `e*Rows + row`. Changing only the index reads real
   weights belonging to the wrong rows. The five weight sites are written in terms of `wrow`/
   `bankRows` and the originals RENAMED `orow`/`orows`, so a missed site fails to COMPILE.
2. **The output does NOT move with the weights.** It is indexed by the SLOT, never the expert id,
   with stride `Rows*Slots` -- the launch's own geometry.
3. **`pAX` is FLAT.** One `Quantize` over the whole `Slots*K` vector writes ALL the scales then ALL
   the sums, so slot j's scales are at `j*nb` and its sums at `Slots*nb + j*(K/16)`: two different
   bases, and the sums array's start moves with `Slots`. The single base `j*(nb + K/16)` is the
   layout that would exist if each slot were quantized separately, and it reads another slot's real
   floats.

**★ AND THE ROUTER IS A KERNEL BECAUSE IT HAS TO BE.** The selection is an INPUT to the expert
matvecs, so a host-side router means a readback and a write per layer per token: ~59 us x 48 layers
is ~5.7 ms of seam on the 30B, against llama.cpp's 14.7 ms for a whole tinyllama token. `ExpertRank`
COUNTS ranks (`#{f: l_f > l_e} + #{f<e: l_f == l_e}`) rather than searching, so there is no buffer it
both reads and writes -- which RULE 12d and `ir.Validate` both refuse. Unselected threads store into
a spare slot `k`, because this IR has no conditional store and clamping to `k-1` would trample the
last real slot.

**★ `ir.OpSelect` WAS THE ONE IR ADDITION, AND IT COMPLETES `OpLt`, WHICH HAD NO CONSUMER AT ALL.**
`Lt` has produced a `Pred` since the IR was written and no lowerer could do anything with one. Select
is DATA FLOW, not a branch -- both arms evaluate -- so the IR still has no conditional execution
outside a loop and the clamp invariant is untouched. One instruction on all three targets: `selp`,
`?:`, `OpSelect`.

**★ THE TIE RULE IS LOAD-BEARING AND RANDOM LOGITS CANNOT TEST IT.** Ranking by "strictly greater"
alone gives tied experts the SAME rank and so the same output slot: one is dropped and a slot keeps
stale data. Run against that mutation, the three CONSTRUCTED tie cases fail and both RANDOM cases
still pass. `ggml_argsort` descending gives ties to the LOWER index and the host router kernel matches it.

**★ AND THREE LATENT TIER BUGS BLOCKED IT, NONE OF THEM MoE's.** All three had every existing model
as an exact control, which is why they landed first:

  - **`bs.quantE` was COMPILED for `NEmbd` and LAUNCHED with `qdim/32` threads.** `Quantize` bakes its
    element count, and the o-projection's input is `nHead*headDim` -- which every model jitllm had ever
    put on a device happened to make equal to `NEmbd` (tinyllama 32x64, gemma 8x256, qwen3-1.7B
    16x128, llama-2 32x128, phi3 32x96). **Both qwen3moe models are the first where it is not:
    `qdim` is TWICE `NEmbd` on both.** The surplus threads clamp onto the last block, so the upper
    half of the vector keeps the previous op's bytes and the per-16 sums land at `pAX[nb]` for the
    wrong `nb`. Nothing faults.
  - **`bs.a`/`bs.ax` were sized `max(NEmbd, NFFN)`** with `qdim` not among the candidates -- a
    device-side out-of-bounds WRITE, not a Go panic. The MoE width is `NExpertUsed*NFFNExp`, which on
    the 30B is 6144 and EQUALS `NFFN`, so that one hides on the big model and not the small one.
  - **`resident()` returned a cached entry on a pointer match without checking shape.** An expert
    bank and its expert 0 begin at the SAME byte, because `model.rows` slices the mapping rather than
    copying (RULE 7c-bis), so a bank uploaded first would answer a later request for expert 0 with
    the bank's buffer and a kernel built for one expert's rows.

**★ 14r. THE ROUTER HAD NO CPU KERNEL AND WAS 20% OF A 30B TOKEN.** `ffn_gate_inp` is F32,
`cpu.Supported` lists only the six quantized formats, so `JIT.MatVec` declined and `s.mv` fell
through to `nn.MatVec`: single-threaded, float64, with a `make([]float64, k)` per call and a dequant
pass per row. **It was invisible because it is a matvec and the profiler charged it to the matvec
bucket**, so `opRouter` was added to see it at all.

Paired A/B, two binaries differing only in this path, Qwen3-30B-A3B on six P-cores:

    without   89-96 ms/token   router 20%   = 18.2-18.8 ms
    with      47-49 ms/token   router  6%   =  2.9 ms
    router 6.4x, and the WHOLE TOKEN 1.9x -- 11.2 -> 20.3 tok/s

**It is deliberately NOT a JIT kernel.** The router is 1 MiB per layer against ~350 MiB of experts,
so the prize is getting it off one core and out of the allocator, not the last cycle of a dot
product; emitting for a format with no block structure would be a new emitter path for 0.3% of the
bytes. Parallel over rows, reading F32 directly, **accumulating in float64**.

**★ THE float64 ACCUMULATOR IS LOAD-BEARING AND AN NMSE GATE WOULD BE THE WRONG TEST.** The router's
output goes to a softmax and then a TOP-K, so one ulp can reorder two near-equal logits, select a
different expert, and produce fluent wrong text -- while NMSE over the router's own output reads
~1e-16 and passes. `TestF32MatVecIsBitIdenticalToRef` requires BIT equality, which is achievable
because only the order of ROWS changes and rows are independent.

**★ AND THAT TEST SKIPPED ON EVERY HOST AND REPORTED PASS.** `NewJIT` builds nothing for F32 alone
and returns nil, so the first version hit `t.Skip` -- and duly "passed" against a deliberately
float32 accumulator. Asking for `{Q4_0, F32}` gives the tier a reason to exist; the F32 path needs
only its pool. **A skipping test is worse than no test: it is a green line that means nothing**, and
it was caught only by running it against a violation as RULE 12m requires.

**Estimates were high, in both directions.** A survey estimated 30-50% of the token and I estimated
35-40% by scaling the 4x0.6B's 0.9%; measured it is 20%. The 4x0.6B could not show it because its
router is 4 rows of 1024 against the 30B's 128 of 2048 -- 1/110th the work -- which is why RULE 14i's
1.0289 win against llama.cpp never exposed it. **Scaling a share measured on a small model to a big
one is the same error RULE 5z records one layer down.**

**★ 14s. TWO THINGS A REVIEW FOUND THAT THE ROUTER WORK GOT WRONG, BOTH VERIFIED HERE.**

**1. `nn.Softmax` IS NOT MONOTONIC, SO A BIT-EXACT MATVEC DOES NOT GUARANTEE THE TOP-K.** `fastExp`
range-reduces at `ln2/2` and the polynomial does not join continuously across the boundary:

    fastExp(-0.34657359027997364) = 0.70710677850244519
    fastExp(-0.34657359027997164) = 0.70710677621636087   <- LARGER input, SMALLER output

a 2.29e-9 inversion, reproduced numerically from the source formula. **The router's whole gate was
`TestF32MatVecIsBitIdenticalToRef`, justified by "one ulp reorders the top-k and selects a different
expert" -- and the very next line of `moe()` calls `nn.Softmax`, which can reorder them anyway.**
Bit exactness was protecting a property the next statement already gives away. That does not make
the test worthless (it still pins the matvec), but "bit-exact or fluent-wrong" was a FALSE
DICHOTOMY, and the right gate is empirical: selected expert ids and their order, against captured
router inputs, with the eighth-versus-ninth margin recorded.

**2. THE SPEEDUP ACCOUNTING DOES NOT CLOSE, AND IT IS MY ARITHMETIC THAT IS WRONG.**

    router   18.5 ms -> 2.9 ms   saved 15.6 ms
    token    92.5 ms -> 48.0 ms  saved 44.5 ms
    UNACCOUNTED                        28.9 ms

The fix is worth about THREE TIMES the router time the profiler attributes to it. Both binaries
timed the router identically (the opRouter subtraction was in both), so this is not a measurement
artifact of the bucket. The standing hypothesis is `nn.MatVec`'s `make([]float64, k)` PER CALL --
48 allocations of 16 KB per token -- churning the heap and evicting the model's hot pages, which
would slow the MATVEC rather than the router. **Unverified.** "1.9x on the 30B" is the
measured whole-token result, and 15.6 ms of router saving does not explain it.

**★ AND jitllm's ROUTER IS ALREADY f32 ON THE DEVICE** (`jit_old/gpu/kernels/moe.go`'s `RouterMatVec`),
which makes a CPU-side float64-exactness contract inconsistent with the engine's own GPU tier. Most
of jitllm is f32 already: every generated CPU kernel writes `Out *float32`. The float64 that remains is
the residual CONTAINER and the reference tier -- and the reference tier must stay f64 because it is
the ORACLE (RULE 8), not a performance path.

**★ 14t. THE ROUTER COMPUTES IN f32 NOW, AND IT COSTS NOTHING AND BUYS CONSISTENCY.**
`cpu.Args.Out` is `*float32` for all six quantized formats and `RouterMatVec` already routes in f32
on the device, so float64 made the CPU router the only matvec in jitllm held to a stricter contract
than the tier it is an oracle for.

**Performance is UNCHANGED and that was expected**: 1.26 ms/token at 39.9 GB/s against f64's
1.24-1.26 at 40.1-40.7. The loop is memory-bound and the weights are f32 either way, so the
accumulator's type moves nothing. Its value is that a kernel can now be 8-wide rather than 4-wide.

**★ WHAT IT DID CHANGE IS THAT THE TWO TIERS NOW AGREE.** Free-running on the 4x0.6B the CPU tier
used to emit `... 374 12095 13 576 ...` where the device emitted `... 374 12095 11 323 ...`; with an
f32 router the CPU emits the device's sequence. Teacher-forced it reads 0/32, 1/32, 0/32 at
max|dlogit| 0.41-0.45, and the dense controls are untouched (tinyllama 0/32, gemma 0/32) because
they have no router at all.

**★ AND THE GATE IS THE ROUTING DECISION, NOT THE BITS.** `TestF32RouterSelectionAgrees` compares
selected expert ids AND their order against the float64 reference over 200 trials at four shapes,
including a deliberately FLAT logit distribution where every choice is a near-tie: 0 set changes and
0 order changes in 800 trials. What it asserts is not the count but the MARGIN -- a selection may
only change where the k-th and (k+1)-th probabilities are within the arithmetic's error, never where
the router had decided. RULE 12i's lesson, one layer up.

Run against violations per RULE 12m: bfloat16-truncating the accumulator FAILS it, and dropping four
elements from every row FAILS it. A gate that has never fired is not a gate.

**★ 14u. THE ROUTER IS JIT-COMPILED ON THE CPU NOW, WHICH IS THE SYMMETRY THAT WAS MISSING.**
`jit_old/gpu/kernels/moe.go` generated a `RouterMatVec` for the device while the host ran a Go loop, so
jitllm JIT-compiled this matvec on the GPU and interpreted it on the CPU -- the ONLY matvec in the
engine in that position. `gguf.F32` is in `cpu.Supported` and `emitF32MatVec` is the kernel.

**It is the simplest kernel in the tree, which is why its absence was easy to miss**: no block
structure, no unpack, no scale, no bias. A `VMOVDQULoad` and a `VFMADD231PSMem` per eight elements,
four accumulators because one is a 4-cycle dependency chain (RULE 6's `Spec.Accs` argument applied to
a kernel with nothing else to hide the latency behind).

    router, 48 layers of 128x2048   Go loop 1.55 ms -> kernel 1.09 ms   1.42x
    whole 30B token                 21.6 -> 21.1 tok/s, i.e. NO measurable change

**★ QUOTE THE 1.42x ON THE ROUTER AND NOT A TOKEN NUMBER, BECAUSE THERE ISN'T ONE.** 0.46 ms of a
46 ms token is ~1%, and this box's run-to-run spread on the 30B is wider than that. The isolated
benchmark reads 1.26 -> 0.67 ms (1.88x) but its 48 MiB set is looped nine times against a 24 MB L3,
so half of it stays resident -- RULE 8b's "any figure over ~55 GB/s is an L3 number", and 75.1 GB/s
is over it. The in-model 1.42x is the honest one.

**So this was built for symmetry, not for speed, and the measurement says so.** The ceiling was known
before it was written (RULE 14t: 0.35 ms available) and the kernel delivered 0.46 ms.

**★ Args.A CARRIES float32 FOR THIS KERNEL.** The field means "the activations"; its Go type is
`*int8` because every other kernel quantizes them. The caller passes a `*float32` through
`unsafe.Pointer`, which stays a real Go pointer so the GC still scans it.

**★ AND arm64 HAS THE KERNEL TOO, BECAUSE OTHERWISE THE MAC WAS THE ASYMMETRY.** Metal generates a
`RouterMatVec`, so leaving the host on a Go loop would have moved the split from Linux to the M4
rather than removing it. `emitA64F32MatVec` is the NEON twin: eight `LDRq`/`FMLA4s` per 32 elements
into four accumulators, reduced with `FADDP4s` because arm64 has no HADDPS. It passed on real M4
hardware first try, and the M4's MoE reads router 0.5% of a token at 87.0 tok/s, 0/32 teacher-forced.

**★ IT IS ALSO THE MOST PORTABLE KERNEL IN THE TREE, AND THAT IS AN ACCIDENT WORTH KNOWING.** It uses
only baseline ARMv8-A -- `LDRq`, `FMLA4s`, `FADDP4s`, `FADDPs`, `UBFM` -- no dotprod, no i8mm, no
SVE. Every QUANTIZED arm64 kernel emits `SDOT`, which is ARMv8.2-A FEAT_DotProd, and
`cmd/jitllm/hardware.go` already records that nothing probes for it: "a host without these gets a
SIGILL on the first token rather than a slower kernel". So on a Cortex-A72 (Raspberry Pi 4) every
other kernel crashes and this one would have run. A Pi 5 is Cortex-A76, ARMv8.2, and is fine.

**★ AND Args.K MEANS THE SAME THING IN EVERY KERNEL AGAIN.** The first version took K in groups of
32, because that is what the loop consumes -- and F32's block size is ONE, so a caller using
`BlocksPerRow` (which every other caller does, including the generic `TestA64GoroutineRegisterIntact`)
passed 256 where the kernel wanted 8, a 32x out-of-bounds read. `Buf.SHRimm` and `A64.LSRimm` were
added so the prologue shifts it; one instruction for a uniform contract.

**★ AND `a64KernelTypes` MEANS "QUANTIZED", NOT "HAS A KERNEL".** Putting F32 in it broke two tests
that iterate it building block-sized buffers and taking `&KernelConst(t)[0]` -- F32 needs no
constants, so it panicked with "index out of range [0] with length 0". `SupportedA64` special-cases
F32 instead, which is what amd64 already does (a switch, with per-format tests naming types
explicitly). **A list used as both a capability set and a test fixture is two things.**

**★ AND ADDING A TYPE TO `Supported` INHERITS NO TEST.** Every format in `jit_old/cpu` is covered by a
test that names it explicitly (`gemm_test.go`, `unroll_test.go`, `accs_test.go`); not one iterates
`Supported` generically. `TestF32MatVecMatchesReference` is the kernel-level NMSE gate the new entry
would otherwise have gone without, and run against violations it reads NMSE 2.3e-1 when an
accumulator is dropped from the reduction and 2.1e+0 when the weight cursor advances 96 instead of
128.

**★ 14m. PLACEMENT MOVES AT RUNTIME NOW, AND THE HISTORY MOVES WITH IT.** `SetGPULayers(n)` walks
the CPU/GPU seam while the model is decoding; `PrewarmGPU(n)` packs off the main loop first.

    verify -migrate    moves the seam DURING the teacher-forced diff, to zero and back twice
    before MigrateKV   18/32 positions disagree, max|dlogit| 18.2   (0.3 is normal)
    after              tinyllama 0/32, gemma 0/32, qwen3 0/32, MoE 1/32 -- its static noise

**Moving a block moved its weights and abandoned its memory.** A resident block keeps keys and values
in `l.kc`/`l.vc` and a host block keeps them in the State's arrays; nothing copied them, so a block
that changed sides attended over the cache it arrived at. Shapes right, no fault, fluent wrong text.
**The gate found it on its first run, and it only could because the migration happens INSIDE the
comparison rather than beside it.**

**★ AND THE PLACEMENT IS A CONTIGUOUS PREFIX BECAUSE THE GRAPH SAYS SO.** The residual runs 0 -> N,
so splitting at ONE point costs one crossing per token and an interleaved set costs one at every
transition. So "which blocks" is really "how many", and within ONE model every layer is the same size
and equally hot -- so a hotness ranking provably degenerates to "as many as fit", which is what the
code already did. `tier.Place` earns its keep only where units differ: across models, or below the
layer. Wired at layer granularity for one model, it would be a policy in name only.

**★ 14n. PACKING IS 80-86% OF PREPARING A BLOCK, AND IT IS THE ONLY PART THAT CAN OVERLAP A TOKEN.**
`layersOnce` holds the tier mutex for a whole token and CUDA forbids allocation inside a Session, so
the question is what does NOT touch the device:

    Qwen3-MOE-4x0.6B  pack 1158 ms  upload 221 ms      prewarmed regrow is 4.0-6.8x faster than cold
    gemma-2b          pack 1523 ms  upload 250 ms

**★ BUT `-gpu-grow` DOES NOT PAY ON THIS BOX, AND ISOLATING WHY KILLED THE PREMISE.** The idea was
that device preparation costs more than the first hundred tokens: 1.83 s to first token with the
device against 0.14 s without. Where that 1.69 s actually goes:

    -devices cpu   0.15 s   |   -gpu-layers 0   1.50 s   |   eager 28/28   1.65 s

**~1.35 s is OPENING the device** -- `backend.Open` starts every backend present -- and 0.15 s is
preparing all 28 blocks once the pages are warm. The 1158 ms above is mostly first-touch disk on the
mmap; a warm regrow packs the same blocks in 87 ms. So growth has nothing to hide behind and adds its
own: over 200 tokens eager reads 123 tok/s and growth 70.9, because each of the first 28 tokens
carries an upload AND a graph re-capture. **Device open is now the biggest time-to-first-token lever
in the engine and nothing has been done about it.**

★ **RETIRED BY RULE 15g: `backend.Open` IS 95 ms, NOT 1.35 s.** Instrumented on this box it reads
ptx 42 ms, spirv 51 ms, metal 0 -- and the rest of that 1.35 s was the per-matvec seam, the first
token paying ~200 host/device crossings. **The biggest time-to-first-token lever was never device
open**, and the number that named it was a subtraction between two configurations that differed in
more than one thing.

**★ 14o. THE RESIDENCY MAP IS KEYED BY SHAPE, AND KEYING IT BY POINTER STALLED THE SEAM AT BLOCK 2.**
An expert bank and its expert 0 begin at the SAME byte (RULE 7c-bis). Running on the CPU offers every
expert's matvec to the device, registering `&bank[0]` at expert 0's row count; a later attempt to make
the BANK resident found that entry. The first fix -- refuse on a shape mismatch -- was correct and
turned a silent wrong-answer bug into a loud capability bug, which is the right direction to fail.
The key is now `(pointer, rows, k, quant)` and both shapes may be resident.

**It cost a debugging round because `PrepLayer` and `resident()` returned false with `LastErr`
EMPTY.** Every decline now names the shape and the budget. **A decline with no reason is untraceable
from outside the tier**, and this engine declines as a matter of course.

**★ 14p. A DEVICE THAT FAILS MID-SEQUENCE NO LONGER ENDS THE SEQUENCE, AND THE COMMENT SAYING IT MUST
WAS TRUE WHEN IT WAS WRITTEN.** It read: "the device holds those blocks' KV cache and the host has
never seen it, so re-running them here would attend to an empty history and produce fluent nonsense."
`MigrateKV` is exactly the thing that was missing, so `demoteAll` now brings the history home and the
token restarts on the host. **`SetDeviceFallback` chooses**, because the two callers want opposite
things: a service would rather serve slowly than fail, and a benchmark wants to know, since a run that
silently finished on the CPU is a number that means nothing and looks fine.

**`JITLLM_DEVICE_FAIL_AT=N` injects the failure, because error handling that has never run is not error
handling.** A card shared with a compositor can refuse an allocation at any moment (RULE 12h) and
that is not reproducible on demand.

**★ AND IT IS BEHIND A BUILD TAG, NOT A FLAG, SO IT IS NOT IN A BINARY ANYONE SHIPS.** Build with
`-tags jitllmfault` to arm it; the untagged `injectedFail` is `return false` and inlines away, so the
release path has no branch and the variable does nothing at all. A shipped engine that can be told to
pretend its GPU broke is a footgun with no upside for whoever is running one -- at best it confuses
someone who copied a command line, at worst it is a way to quietly slow another person's deployment.
The tagged build announces itself on stderr for the same reason: a binary that can lie about its
device should say so.

    go build -tags jitllmfault ./cmd/jitllm && JITLLM_DEVICE_FAIL_AT=5 ./jitllm run ...

Teacher-forced with the fault injected at calls 3 and 9:

    tinyllama 0/32 @ 0.32     gemma 0/32 @ 1.5-1.7     MoE 2/32 @ 0.47 -- all their normal bands

**★ 14q. THE SEAM TUNER EXISTS AND IS OFF BY DEFAULT, AND THE DEFAULT IS THE MEASUREMENT.** It asks
whether FEWER device blocks are faster -- the case RULE 12h describes, where another process holds
VRAM, and the case RULE 1c described until the Metal kernels were fixed. **Arms are RUNS of tokens,
not tokens**: switching a placement costs a migration of 100-500 ms against a token of 8-100 ms, so
per-token ABBA would measure the migration and nothing else. The clock starts AFTER the migration,
since folding the cost of arriving at a placement into that placement's own measurement biases every
switch the same way and no amount of ABBA removes it.

It is off by default because on both machines this project has, more device blocks is strictly
better, so switching it on would spend ~128 tokens and four migrations to re-derive a known answer.
**It is for hardware this project has not seen.**

**Prefill and ForwardBatch run MoE PER ROW, and that is not a shortcut.** Every row picks its own
experts, so there is no shared weight matrix for a batched matmul to amortize; a 32-row chunk over
128 experts touches most of the bank once each, which is exactly what the per-row loop reads.
Batching would need a gather that reorders rows by expert, which is a different design. **The device tier used to refuse MoE blocks outright** for want of a field to
put the router in; RULE 14j built that field and it runs there now.

**★ 14f. TWO PRE-EXISTING BUGS BLOCKED THIS, AND ONE OF THEM WAS HIDING AS A SLOW MACHINE.**

`go test ./engine/model` had been TIMING OUT at the 600 s package default. `nn.ResetForTest` reset the
`sync.Once` but never cleared `fastDisabled`, and `TestPlacementInvariance` ends every subtest with
`decode(true)` -- which sets `JITLLM_DISABLE_FAST=1` and consumes the Once. `t.Setenv` restores the
environment; the package variable it was read into stays true forever, so every later `NewJIT`
skips the Do and returns nil. `llamacpp_test.go` sorts after `placement_test.go`, so the oracle
was decoding two 1-2B models on the float64 reference tier at ~0.3 tok/s.

    TestPlacementInvariance + TestGreedyMatchesLlamaCpp    180 s timeout -> 8.1 s
    the whole model package                               did not finish -> 42.9 s

**A tier that silently downgrades looks exactly like a slow machine, and nothing failed.** That is
the same shape as RULE 11b's blind profiler and RULE 5q's wrong register: the suite was green.

And `State.xb` was sized `n_embd` while attention writes `xb[hh*headDim+i]`, which needs
`n_head*head_dim`. They coincide on every model jitllm had run -- llama, gemma and qwen3-1.7B all have
`n_head*head_dim == n_embd` -- so it worked by accident. Qwen3-30B-A3B is 32 heads of 128 against
n_embd 2048 and panics at layer 0. Prefill had already sized its equivalent by `max(NEmbd, qDim)`.

**★ 14g. THE FIXTURES, AND WHY THERE ARE TWO.** `scripts/tiny-moe-gguf.py` builds a 16 MB qwen3moe
from `trl-internal-testing/tiny-Qwen3MoeForCausalLM` -- 2 layers, 4 experts, 2 active, real graph,
random weights, all F32 so it runs on the reference tier. It exists because the 30B is 18.5 GB and
this link runs at 2 MB/s, and a MoE routing bug does not crash. It reads safetensors directly (a
length-prefixed JSON header over raw arrays, thirty lines) because `convert_hf_to_gguf.py` imports
torch at module scope and torch is 2.5 GB on a box whose home partition has 29 GB free, and it
copies the tokenizer wholesale out of a real Qwen3 GGUF because the two engines have to agree on
tokenization before they can be compared on anything else.

    tiny-qwen3moe-f32        19/20 ids against b10825, diverging on a 0.001 tie
    Qwen3-MOE-4x0.6B Q4_K_M  identical text, 64.5 tok/s on six P-cores
                             "The capital of France is Paris. The capital of the
                              United States is Washington, D.C. The capital"

The second is 28 layers of Q4_K + Q6_K expert banks, so it is the one that exercises the generated
kernels; the first is the one whose experts are actually distinct.

**★ 14i. jitllm BEATS llama.cpp ON THE SMALL MoE, AND MoE IS ITS BEST CPU ROW -- BUT ONLY AT DEPTH 0.**
`scripts/vs-llamacpp.sh cpu`, n=320, six P-cores, with a dense control at both depths so the MoE
number can be attributed:

    model                            depth 0            depth 512
      Qwen3-MOE-4x0.6B  Q4_K_M       1.0289 / 1.0374     0.9600     IQR 0.011 / 0.032 / 0.026
      Qwen3-1.7B dense  Q4_K_M       0.9995              0.9588     IQR 0.043 / 0.029

Two readings, and the control is what makes both of them safe.

**THE MoE ROW IS THE BETTER ONE.** At depth 0 the mixture of experts WINS at 1.03 while the dense
model of the same family, same quantization and same tokenizer sits at parity. So the MoE path is
not a weak new code path being carried by the rest of the engine -- it is ahead of the dense path,
and it is the second CPU win on this board after llama-2-7b. The likely mechanism is that a MoE
token reads NExpertUsed/NExpert of the FFN, so it is LESS memory-bound than a dense token of the
same parameter count, and a larger share of it sits in matvecs where jitllm's kernels are strong rather
than in DRAM where both engines queue.

**THE DEPTH PENALTY IS NOT MoE's.** 0.9600 against the dense control's 0.9588 is the same number.
Both engines lose throughput as the context grows and jitllm loses more, by about 4 points from depth 0
to 512, and that is true of the dense model as well -- so nothing about routing, expert slicing or
the per-row FFN is responsible. The mechanism is visible in the graph dump of RULE 14a: llama.cpp
runs `FLASH_ATTN_EXT`, one fused node, where jitllm runs scores, softmax and the weighted sum as three
passes over the KV cache. RULE 8b already recorded that attention goes from 8.2% of a decode token
at six positions to 52.9% at 821; this is the same fact showing up in a ratio.

**SO THE NEXT CPU LEVER IS ATTENTION AT DEPTH, NOT ANYTHING IN RULE 14.** Note also that every CPU
row on RULE 1d's scoreboard is a DEPTH-0 row, so the board is measuring jitllm at its most flattering
context length -- the GPU rows are quoted with their depth for exactly this reason and the CPU rows
never were.

**★ 14k. THE ROUTING NUMBERS ON THE REAL 30B, AND THE 73.9% I QUOTED WAS THE 4x0.6B.**
`TestRoutingLocality` with `JITLLM_ROUTE_MODEL`, 200 decoded tokens, 48 layers x 128 experts, 8 used:

    previous-token predictor   33.9% of experts reused   (chance 6.2%)
    whole set unchanged         0.1% of layer-steps

**So per-token prefetch is much weaker than the small model suggested** -- 33.9%, not 73.9%. A
predictor keyed on "the same experts as last token" is worth having as a hint and is not a design.

**★ BUT THE HOTNESS SKEW IS STRONG, AND THAT IS WHAT A PLACEMENT POLICY NEEDS.** Coverage of all
selections by the r hottest experts of each layer, measured on the same workload:

    r=16 (12.5% of the bank)  59.0%      r=48 (37.5%)  93.0%
    r=32 (25.0%)              82.2%      r=64 (50.0%)  97.8%

A quarter of the bank serves 82% of the uses.

**★ AND IT IS WORKLOAD-DEPENDENT, WHICH IS WHY THE POLICY HAS TO BE MEASURED AT RUNTIME RATHER THAN
SHIPPED.** The same test on a prose prompt and on a Rust prompt gives hot sets that overlap barely
above chance -- 35.6% at r=32 against 25% random. Scoring the CODE workload against a resident set
chosen from each:

    r      adapted   stale     chance
    16     50.5%     20.2%     12.5%
    32     72.6%     34.7%     25.0%
    48     86.0%     46.2%     37.5%

**Read those two columns as STEADY STATE and TRANSIENT, not as "it works" and "it does not".** A
policy that measures hotness while it runs converges on the workload it is actually serving, so
`adapted` is what it reaches; `stale` is the window right after the workload changes character.
Flipping from one to the other means relocating 989 experts at r=32, which is 2502 MiB over an
8.14 GB/s PCIe link -- **300 ms, once, in the background**, against a ~95 ms token. That is a JIT's
tier-up economics and not a stall, because a miss is HOST WORK rather than a page-in: the device
computes the experts it holds, the host computes the rest, and nothing waits on the bus.

**In-sample coverage is overfit and must be labelled.** The 82.2% above scores the hot set against
the very trace that produced it. Quote it as the converged number, never as the cold one.

**★ 14h. RUNNING THE 30B NEEDS A BIGGER CAP THAN 8G, AND THE REASON IS RULE 3c.** cgroup v2 charges
a mmap'd file's page cache to the cgroup, so an 18.5 GB model under `cap 8G` sits permanently in the
`MemoryHigh` throttle band -- and RULE 3c records that on this box the throttle is what GENERATES
the pressure `ManagedOOMMemoryPressure=kill` acts on. Use `./scripts/cap 24G` for the 30B and say so.
Note also that llama.cpp's loader issues `posix_fadvise(SEQUENTIAL)`, `MAP_POPULATE` and
`posix_madvise(WILLNEED)` over the whole file, and `convert/gguf/mmap_linux.go` issues none of them, so a
cold-cache comparison on a file this size is not measuring the engines.


## ★ RULE 7r: SAFETENSORS IS THE SECOND INPUT FORMAT, AND THE ROTARY PERMUTATION IS THE PART THAT LOOKS FINE WHEN IT IS WRONG

Landed as `convert/safetensors/` plus `convert/safetensors.go` and `convert/hfvocab.go`. The
container is unchanged, `jlm.Write` is unchanged, and nothing in `engine/model/`, `engine/nn/` or `jit_old/` moved:
a source format is a reader that fills a `jlm.Source`, which is what the Scope section already
said and what this measured.

**The engine's first input format is a file; the second is a DIRECTORY.** A GGUF carries weights,
hyperparameters and the whole tokenizer; a safetensors file carries a dtype, a shape and an offset
per tensor and nothing else. So `config.json`, `tokenizer.json` and `tokenizer_config.json` beside
the weights are not optional extras, they are where the model is -- and `jitllm convert` takes the
directory (or one `.safetensors` file, meaning "this shard and the JSON beside it").

**★ AND THE ONE PLACE THE BYTES ARE NOT THE FILE'S BYTES IS q AND k.** HuggingFace's Llama rotates
with `rotate_half`, pairing head dimension i with i+head_dim/2; this container's llama graph does
NOT set `FlagRopeNeox`, so it pairs (2i, 2i+1). The GGUF everyone runs is already in the second
order because `convert_hf_to_gguf.py` permutes q and k on the way in -- the design document
(ARCHITECTURE.md §7, since deleted) recorded exactly this for llama and warned *"get it wrong
and short prompts still look fine"*. **It is not a warning, it is a measurement.** SmolLM2-360M with the permutation removed:

    correct       " Paris.\n\nParis is the capital of France.\n\nParis is the"
    unpermuted    " Paris.\nYour answer must contain at least 3 placeholders represented by"

It still answers the question, in fluent English, and then writes a different document. Nothing
inside one format can see it: every tier agrees with itself, the shapes are right, the NMSE against
its own oracle is zero. What sees it is a SECOND ROUTE TO THE SAME WEIGHTS --
`convert.TestSafetensorsTensorsMatchTheirGGUF` converts the bf16 HuggingFace directory and the Q8_0
GGUF of one model and compares tensor by tensor:

| | worst NMSE |
|---|---|
| all 290 tensors, shipping | **5.843e-05** (Q8_0's own error; bound 1e-3) |
| the same, permutation removed | **1.94e+00** on every attn_q and attn_k (two uncorrelated vectors) |

Four orders of magnitude apart, with nothing in between for a bug to sit in.

**Permuting was a CHOICE and the alternative is also correct.** Setting `FlagRopeNeox` for a
safetensors-sourced llama and leaving the rows alone computes the same function. It was refused
because it makes the container depend on which reader produced it: the comparison above stops being
possible, and the engine carries two rotary layouts distinguished only by provenance.

**★ THE TOKENIZER IS THE FORMAT'S ADVANTAGE, AND RULE 7m IS WHY.** A GGUF stores a pre-tokenizer
LABEL and every engine keeps a name -> pipeline table; a HuggingFace directory ships the artefact
the weights were trained against, and `pretok.ParsePreTokenizer` already reads it. So three things
that are INFERRED from a name family on the GGUF side are simply stated here: `ignore_merges` and
`byte_fallback` are fields of `model`, and add_bos comes from the `post_processor` -- the thing that
actually runs -- rather than from `llama3Family[name]` patched by a key that is often absent.
Measured on SmolLM2-360M, `convert.TestSafetensorsTokenizerMatchesItsGGUF`: **49,152 tokens, 48,900
merges and both pre-tokenizer stages identical**, BOS/EOS/AddBOS/AddEOS/IgnoreMerges/ByteFallback
identical, and the safetensors arm additionally carries **17 added tokens the GGUF has no field
for**. The two authorities agree on this model; the point is that now they can be ASKED.

**★ BF16 IS WIDENED TO F32 AT CONVERSION, AND THAT IS RULE 8's CLIFF AVOIDED RATHER THAN A
PREFERENCE.** `TypeBF16` exists and a container can carry it -- but `cpu.SupportedNative` is
`{F32, F16}`, so a bf16 weight matrix takes the VERBATIM path to `nn.MatVec`, the oracle. Measured,
same model, same gate, 16 tokens:

    widened to f32   1.4 s    0 matvec fallbacks
    carried as bf16  44.5 s   4721 matvec fallbacks   -- token ids IDENTICAL

32x, with every answer right: RULE 8's *"a cliff, not a degradation, indistinguishable from a slow
machine"*, reproduced. The widening is EXACT (bf16 is f32's top 16 bits) and costs 2x the bytes.
**The lever that removes both is a BF16 native matvec beside the F16 one** -- on amd64 it is a
zero-extend and a shift-left-16 where F16 needs `VCVTPH2PS`, so it is smaller than the kernel it
would sit next to. Not built; named.

**What is refused, by name, and why each would otherwise run:** a `rope_scaling` other than default
(llama3's and yarn's are different frequency tables and nobody applies them), `mlp_bias` (no role),
a `hidden_act` that is not SiLU, `partial_rotary_factor`, `sliding_window`, a dtype outside
{F32, F16, BF16} (there is no quantiser in the converter -- an unquantised checkpoint stays
unquantised), a tensor name with no role, a tokenizer that is not BPE, and a `tie_word_embeddings`
that disagrees with the presence of `lm_head.weight` in either direction.

**★ AND THE SHAPE CHECK IS THE GATE ON THE NAME TABLE.** `hfCheckShape` re-derives every tensor's
dims from the config -- q is NHead*HeadDim rows, k and v are NKVHead*HeadDim, gate and up are NFFN,
down is NEmbd -- and refuses a mismatch. It is the same check that stopped qwen3next's 8192-row
fused projection from being unfused by the attention shape (RULE 11c), and it catches a q/k swap on
any GQA model. **On a pure-MHA model it cannot**, because q and k are then the same shape; there the
cross-format NMSE row is the only thing that sees it, which is an argument for keeping a
two-format model on the box rather than a reason to trust the shape check.

**`post_attention_layernorm` is the FFN's input norm.** In HuggingFace's Llama the tensor by that
name is applied before the MLP -- what every architecture here calls `ffn_norm` -- while gemma2 and
gemma3 ship a tensor of the same name that is a norm on the ATTENTION OUTPUT. Two positions in the
graph, one string; this is `convert.retarget`'s trap arriving from the other vocabulary, and
mapping it by its name loads, norms in the wrong place and answers fluently.

**The architecture list on this side was ONE: `LlamaForCausalLM`** (eight now --
see the next section). Keyed on `architectures[0]` and not on `model_type`, because one model_type
covers several classes and only the causal-LM one has this graph. Adding an entry is the same
decision RULE 7 describes, with one extra claim attached: that this engine's graph matches that
architecture's reference implementation under a DIFFERENT set of tensor names.

## ★★★ SAFETENSORS GOES FROM 3 CLASSES TO 8, EACH GATED AGAINST transformers ITSELF.

`hfArchTable` now carries Llama, Qwen2, Qwen3, **Qwen3Moe, Olmoe, Mixtral, Gemma, Phi3**.
`TestSafetensorsArchCoverage` reads **12 of 13 HF directories converting** (was 3 of 11); the rest
are refused by name: qwen3-next (not in the table), a qwen3moe with dense layers interleaved
(`decoder_sparse_step 2`, which a container cannot express), and the downloaded tiny Mixtral, whose
`hidden_act` is exact `gelu` against a SiLU graph.

**★ THE ORACLE IS THE REFERENCE CLASS, NOT A SECOND FORMAT.** The cross-format gate above needs a
GGUF of the same weights and one model on the box has both. `scripts/hfgold.py` runs
`AutoModelForCausalLM` (CPU torch in the reference venv, `$JITLLM_HF_PY`) over fixed ids in float32 and
records the logits; `model.TestSafetensorsMatchTransformers` compares every position. With the KV
cache at f32 the correct arms sit at **1.5e-13 to 1.1e-11**, on amd64 AND the M4.

**★ AND ITS FIRST READING WAS A LIVE BUG IN AN ARCHITECTURE ALREADY SHIPPED.** qwen3 read
**NMSE 2.46e-01 from position 1**. The converter permuted qwen2/qwen3's q/k rows exactly as llama's
-- but `convert_hf_to_gguf.py` permutes only llama's, and qwen3's per-head `q_norm`/`k_norm` weight
is indexed by the head dimension the permutation reorders, so permuted rows met an unpermuted norm.
Position 0 cannot see it (softmax over one key is 1, whatever the scores). qwen2/qwen3 are NEOX and
unpermuted now, which is what their GGUFs are -- the section above's own argument for permuting
llama, applied honestly. **Any qwen3 container converted from safetensors before this is wrong and
must be reconverted.**

**★ THE BOUND WAS SET BY VIOLATIONS, AND THE FIRST ONE WAS TOO LOOSE TO BE A GATE.** At 1e-4 with an
f16 cache, swapping gate and up in every OLMoE expert PASSED (1.7e-06): a random tiny mixture's FFN
barely moves the residual. With an f32 cache and a 1e-9 bound, every violation below fires:

| violation | fixture | reading |
|---|---|---|
| qwen3 rows permuted (the old code) | qwen3 | 2.5e-01 |
| norm_topk_prob inverted | qwen3moe, olmoe | 4.3e-05, 2.1e-04 |
| gate/up swapped in the experts | qwen3moe, olmoe | 7.3e-05, 2.6e-07 |
| experts stacked in reverse | qwen3moe, olmoe | 4.6e-03, 7.2e-05 |
| gemma: no +1 / SiLU / no embd scale / NORM rope | synth-gemma | all FAIL |
| phi3: gate,up order / q,v,k order | both phi3 | FAIL |
| mixtral: no permute / no renorm / w1,w3 swapped | synth-mixtral | FAIL |
| SPM scores inverted / no prefix / byte kinds dropped | llama, phi3 | FAIL (encode, decode) |

**★ gemma's NORM-rope violation PASSED on the downloaded fixture**, because at head_dim 2 NEOX and
adjacent pairing are the same single pair; and most tiny-random checkpoints ship all-one norms, which
hide every norm wiring. `hfgold.py` therefore also BUILDS three fixtures from the reference classes
(`synth-gemma`, `synth-mixtral`, `synth-qwen2`: head_dim 16, GQA, random norms, init 0.1).

**★ SENTENCEPIECE-DERIVED tokenizer.json IS AN SPM VOCABULARY NOW, AND THE SCORE IS THE MERGE
RANK.** It was refused (`pre_tokenizer is null`), which blocked gemma, phi3, mixtral and every
llama-2-era checkpoint. HuggingFace builds those merges by emitting each piece's two-way splits
ORDERED BY THE PIECE'S SCORE, so `score(t) = -(rank of the first merge that builds t)` reproduces
the order tok's SPM encoder merges in. The prefix comes from the FILE (gemma has none).

**★ THREE TOKENIZER AUTHORITIES DISAGREED, AND RULE 7m's "KNOW WHICH ONE" DECIDED IT.**

    " leading space"   sentencepiece  [29871 8236 ...]   the model's training tokenizer
                       tokenizer.json (Prepend+Replace)  identical -- llama, phi3
                       tokenizer.json (Metaspace first)  [8236 ...]  -- mixtral: skips the
                                                         dummy prefix when text starts with a space
                       transformers 5 LlamaTokenizer     [8236 ...]  -- DISCARDS tokenizer.json and
                                                         rebuilds Metaspace(first) for model_type llama

jitllm matches sentencepiece (and llama.cpp) on all of them. The gate reads `tokenizer.json`
through the `tokenizers` library rather than AutoTokenizer, records sentencepiece's ids where
`tokenizer.model` exists, matches THOSE, and logs every place the file departs -- mixtral's one
string today.

**Two latent bugs found on the way:** `jlm.TokenByte`/`TokenUnused` were 5/6 against the 6/5 every
container actually holds (GGUF's numbering, copied verbatim, and tok reads 6 as a byte) -- the
constants are swapped to match the data. And an added token past `model.vocab` (qwen's
`<|im_end|>`, phi3's `<|end|>`) kept its pad name `[PAD151645]`; the slot takes the content now.

**phi3 slides every layer and `Config.SWA` is periodic**, so a safetensors phi3 caps `NCtx` at the
window: full and sliding attention are then the same computation. A 4k phi3 runs 2047 positions;
an all-local SWA pattern on both tiers is what gives the rest back. (The GGUF phi3 path ignores the
window entirely, which llama.cpp does not -- wrong past 2047, unfixed.)

**Below the attention kernels' floor (`head_dim%8`), the Xenova fixtures (head_dim 2, 4) reach `nn`
for attention and a 4-wide bias add.** Reported by the gate and not asserted; no real model is there.

### As it stood in AGENTS.md

**And HF safetensors is a THIRD list**: `hfArchTable` in `convert/safetensors.go`
-- Llama, Qwen2, Qwen3, Qwen3Moe, Olmoe, Mixtral, Gemma, Phi3, DeepseekV2,
DeepseekV3, **Glm4MoeLite** (GLM-4.7-Flash, which is deepseek2's graph under
another class name -- see RULE 7's deepseek2 entry for the gating-function bug
that discovery cost), **KimiLinear**, the mixture families **Glm4Moe**,
**Qwen2Moe**, **Ernie4_5** / **Ernie4_5_Moe** and **HunYuanDenseV1** /
**HunYuanMoEV1** (`convert/hfmoefamily.go`).

★ **THE FAMILIES CONVERT TO THE SAME BYTES FROM EITHER INPUT.** Each family's
fixture is saved by its own transformers class and written to GGUF by
llama.cpp's converter from the same weights, so
`convert.TestMixtureFamiliesConvertAlikeFromEitherInput` holds the two
containers to every named config field and every tensor byte for byte -- a
stronger gate than either oracle alone, and the one that caught the [1, n]
shapes llama.cpp keeps. The router's gate is read per CLASS, never from an
absent key: `hardcodesSigmoid` (Glm4Moe added) and its twin
`hardcodesSoftmax` (Qwen2Moe, Ernie4_5_Moe, HunYuanMoEV1), and a
`scoring_func` contradicting the class is refused (`hfGateAgrees`).
`tokenizer_class` "PreTrainedTokenizerFast" means tokenizer_config's
`add_bos_token` is not read -- the generic class runs the post-processor as
written, and Hunyuan's `true` added a BOS no reference adds.

The list also carries the dense llama family's **SmolLM3**, **Arcee**,
**SeedOss**, **Olmo2**, **Olmo3**, **Exaone4** and **Ministral3** (mistral3's
text model; `Mistral3ForConditionalGeneration` is not on the list).
Each of those seven reads its CLASS's defaults for an absent key
(`hfClassDefaults`) -- SeedOss's head_dim is 128, SmolLM3's kv heads 4, OLMo 3's
window 4096, Ministral3's whole rotary a YaRN dict -- and converts to the
container the GGUF path writes, Config and bytes
(`convert.TestDenseSafetensorsMatchesItsGGUF`).

★ **AND KimiLinear IS THE FIRST HYBRID TO COME THROUGH THIS PATH, WHICH COST
TWO PIECES OF MACHINERY THE GGUF PATH NEVER NEEDED.** qwen3next only ever
arrived as a GGUF, where llama.cpp had already named the recurrent tensors
apart from the attention ones and already written `-exp(A_log)` into `ssm_a`.
A safetensors hybrid has neither:

    blockLinear   ONE NAME MEANS TWO THINGS. self_attn.q_proj.weight is the
                  recurrent block's query in a linear layer and MLA's one-step
                  query in a full one, in the same checkpoint; so is o_proj,
                  which is RoleSSMOut in one and RoleAttnOut in the other. A
                  flat name table resolves one of them to a role whose shape
                  check then refuses a CORRECT model. identify() takes
                  Config.LayerKinds and consults blockLinear first.
    fuseOrder     `split` RUN BACKWARDS. HuggingFace ships q, k and v as three
                  matrices and three conv1d filters; the container carries one
                  mixed projection and one filter bank, because deltaGeom
                  slices them by arithmetic and the conv kernel walks all the
                  channels at once. concatRows joins them, and the ORDER is
                  the whole risk -- q|v|k is a tensor of exactly the right
                  shape whose every head belongs to another stream.

★ **AND IT RE-SPELLS SIX MIXTURE KEYS THAT ALREADY HAD TWO OR THREE SPELLINGS
EACH, EVERY ONE OF WHICH FAILS AS A ZERO RATHER THAN AS AN ERROR.**
`num_experts_per_token` against `num_experts_per_tok`, `num_shared_experts`
against `n_shared_experts`, `num_expert_group` against `n_group`,
`num_experts` AND `num_local_experts` against `n_routed_experts`,
`moe_renormalize` against `norm_topk_prob`. encoding/json leaves an unread key
at its zero value, so the result is not a refusal -- it is `NExpertUsed = 0`, a
shared expert of width 0, an ungrouped selection, or a mixture that quietly
stops renormalising. `KimiLinearTopkRouter` reads `num_local_experts` while
`KimiLinearConfig` aliases it from `num_experts`, so a checkpoint written by a
tool that follows the MODULE rather than the config class carries the other
one. Both are read, and a file carrying two spellings that DISAGREE is refused
rather than resolved -- there is no way to know which its weights were trained
under (RULE 7m).

★ **AND ITS ROUTER HARDCODES THE SIGMOID WITH NO `scoring_func` KEY, WHICH IS
THE Glm4MoeLite BUG ARRIVING A SECOND TIME AND BEING CAUGHT BY THE RULE THAT
COST.** `KimiLinearTopkRouter.forward` is `scores = router_logits.sigmoid()`,
flat, and its config.json has no gating key at all -- so the per-tree default
of softmax would convert a model that runs and routes to the wrong experts.
`hardcodesSigmoid` carries it beside `DeepseekV3ForCausalLM` and
`Glm4MoeLiteForCausalLM`. **The same forward also reads
`topk_weights.sum(...) + 1e-20`**, which is independent confirmation of the
renormalising divisor's floor this file already records as DeepSeek's rather
than llama.cpp's clamp.

An entry is gated
against transformers itself: `scripts/hfgold.py` records the reference's logits
and tokenizer ids, `model.TestSafetensorsMatchTransformers` compares them (bound
1e-9 with an f32 cache, set by violations). A new entry needs a fixture and a
golden, or it is a claim with nothing behind it.
`docs/engineering-history/model-correctness.md` has the measurements.

## ★★★★★ llava/CLIP IS NOT BROKEN ON arm64, AND THE CONVERTER IS NOT ARCHITECTURE-DEPENDENT. THE ENTRY BELOW WAS WRONG, AS A RE-RUN THE SAME DAY SHOWED.

All ten llava gates PASS on the M4 at `b0bf910`. The tower encodes 576
embeddings x 3072 over the range -32.0420..28.7307, mean -0.011220 -- not the
identically 0.000000 recorded below.

★ **THE CONVERTER PRODUCES THE SAME WEIGHTS ON BOTH HOSTS, AND THE ONLY WAY TO
SEE THAT IS TO PIN THE COMMIT.** One cross-built binary, the same two GGUFs,
converted on each machine and compared by 1 MiB chunk hashes rather than by a
whole-file hash:

    SmolVLM-256M (idefics3)   2 of 265 MiB differ
    llava-phi-3 (mlp)         **1 of 2705 MiB differ**

And the differing bytes are 47, all of them the **HOSTNAME**:

    amd64   a 3-byte hostname (the laptop's)
    arm64   a 20-byte hostname (the Mac's)

plus one length field at offset 97 shifting by exactly 17 = 20-3. Every weight
page, every vision block, every scale plane: byte-identical. The search for an
arch-dependent code path was also a dead end by construction -- `go list -deps
./convert` is quant, meta, sched, gguf, jit_old/gpu/{ir,kernels}, jlm, pretok,
safetensors, and **not one of them has a `//go:build amd64/arm64` line or an
`_amd64.go`/`_arm64.go` file**.

★ **SO A WHOLE-FILE HASH CANNOT ANSWER "DID THE CONVERTER PRODUCE THE SAME
WEIGHTS", AND THAT IS A PROPERTY OF THE FORMAT RATHER THAN AN ACCIDENT.**
`jlm.Fingerprint` records the host on purpose (`format/jlm/format.go:191`: *"a file
will eventually be copied ... the failure mode of a machine-tuned file loaded
elsewhere must be a WARNING and a correct run"*). Two containers from two
machines can therefore NEVER be byte-identical, and "same size, different hash"
-- the observation the entry below is built on -- is the expected result for
two CORRECT conversions. Compare the weight region, not the file.

★★ **WHAT ACTUALLY BROKE WAS THE TEST SUITE'S CONVERTER CACHE, AND IT IS FIXED
(`engine/model/jlm_helper_test.go`, `TestCachedContainerWasWrittenByThisBuild`).**
`jlmOfPair` reused a container whenever it was newer than its GGUF -- and a GGUF
never changes, so that condition can never expire anything. The M4 was holding a
container written before the llava converter landed, and serving it to every
run. Replacing it with a hand-copied amd64 file made the gates pass, which read
as an ARCHITECTURE difference and was a FRESHNESS one.

★ **AND jlm.Fingerprint.Writer, THE FIELD THAT EXISTS FOR EXACTLY THIS, IS
BLIND IN THE HARNESS THAT CONVERTS MOST OFTEN.** `go test -c` stamps no VCS
info, so `jlm.WriterID()` reads `"go1.27.1 (devel)"` in every test binary and
matches itself forever. The binary's MTIME is not an identity either -- `go
test` relinks into a fresh `/tmp/go-buildNNN` every run, so keying on it
reconverted every model on every invocation (measured, then rejected). What
works is the binary's CONTENT: `BuildIDForTest` hashes it once per process
(~10 ms over 8 MB) and stamps the digest into `Writer`, and the cache reconverts
whenever the recorded digest disagrees. Against the violation -- a container
written by the CLI build, same layout version, opens cleanly -- the gate reads
*"the cache served a container written by \"go1.27.1 b0bf9107e981\", this build
is \"test b05000084999a324\""*.

★ **THE STANDING LESSON IS RULE 10's, IN THE PLACE THAT IS HARDEST TO SEE IT:
THE HARNESS.** Every one of the five red gates was a real red gate, reporting a
real wrong answer, from a real container. What nobody checked was WHICH BUILD
wrote the artefact under test -- the same omission as running a gate against a
configuration nobody selected, one level further out. **When a gate indicts an
architecture, pin the commit and convert on both hosts before theorising about
the ISA.** That is RULE 1's "re-measure in the same pass" applied to a
correctness artefact, and it took twenty minutes: one cross-build, two
conversions, one chunk diff.

★ **AND THE ORIGINAL ENTRY IS KEPT IN FULL BELOW, BECAUSE THE WAY IT WENT WRONG
IS THE LESSON.** Its one experiment was sound and its inference was not: "same
binary, different container, different result" does not isolate the CONVERTER
unless the two containers came from the same build, and nothing established
that. The doc even carried the caveat for the TEXT container -- *"the two
binaries were built from DIFFERENT COMMITS ... It is not established as benign
and should not be assumed so"* -- and did not apply it one paragraph up.

## ★★★ THE SUPERSEDED ENTRY: "llava/CLIP IS BROKEN ON arm64, AND IT IS THE CONVERTER RATHER THAN THE ENGINE." SHOWN WRONG ABOVE.

llava/CLIP had just been merged to main and its gates had only ever run on amd64.
Run on the M4 (RULE 11d), five of them fail, and the tower's output is
identically **0.000000** on every embedding:

    TestLlavaTowerEncodes      every embedding is within 1e-3 of every other
                               (0.000000..0.000000): the tower produced a
                               constant, which is finite and useless
    TestLlavaOutputRowsAlignWithPatches, TestLlavaPositionsArePairedWithTheirRows,
    TestLlavaPreLoopNormIsApplied, TestLlavaEncodesARealImage

★ **IT IS THE CONTAINER, NOT THE HOST, AND THAT IS ONE EXPERIMENT.** The same
M4 binary, the same gates, the only change being which machine wrote the
container:

    arm64 engine + arm64-converted container      5 FAIL, output identically zero
    arm64 engine + amd64-converted container      ALL PASS

So `EmitA64*` and the arm64 tower path are fine; `jitllm convert` writes a
silently wrong container on that host. The two files are **the same size to the
byte** (2,835,718,144) and differ in content hash, which is the shape this
project has seen before -- a layout that agrees and bytes that do not.

★ **AND CONVERSION IS DETERMINISTIC, SO THIS IS NOT NONDETERMINISM.** Converted
twice on one machine, byte-identical both times. Go's randomised map iteration
was the obvious suspect and it is refuted.

★ **A TEXT CONTAINER ALSO DIFFERS ACROSS THE TWO HOSTS AND THAT ONE IS
PROBABLY BENIGN.** Llama-3.2-1B converted on each: same size (858,316,800),
different hash -- but the two binaries were built from DIFFERENT COMMITS and the
container records `written by go1.27.1 <commit>`, which is twelve hex characters
either way. The text model's greedy gates pass on both hosts, so nothing in its
payload is in question. **It is not established as benign and should not be
assumed so**: the check that would settle it is converting from ONE commit on
both hosts and diffing.

★ **WHAT IS NOT YET KNOWN.** Which tensor or which pass goes wrong. `format/quant/` has
no architecture-specific files at all, so the F16 decode is identical Go on both;
`quantizeQ8Rows` in convert/vision.go is pure Go arithmetic. The remaining
candidates are `kernels.PackWeights` and anything the converter reaches that is
built per-GOARCH. Narrowing it wants a region-by-region diff of the two
containers, which needs ~3 GiB free on the Mac and did not fit at the time.

★ **THE M4's llava GATES ARE RED ON PURPOSE.** A hand-placed amd64 container
makes them green and hides this; it was removed so the next run reconverts and
reproduces the fault. RULE 11b: a red gate for a real bug is the system working.

★ **AND THE LESSON IS RULE 11d's, AGAIN, WITH THE SAME COST.** This file already
records arm64 decode producing garbage for every quantized model while eleven
green kernel gates said nothing, because what was run on the M4 was `./engine/nn` and
`./jit_old/cpu`. A vision branch merged with its gates green on one architecture is
the identical omission, and the identical class of bug came out of it.

## ★★★★ qwen2vl AND qwen2vl_merger. THE EXPENSIVE PART WAS THE ORACLE, NOT THE ARCHITECTURE.

The ninth architecture and the third projector. What each half actually cost is
the opposite of what the list suggests.

### The text half was nearly free, and it is verifiable ALONE

Qwen2-VL's text block is qwen2's tensor for tensor -- biased q/k/v, an unbiased
output projection, RMSNorm, SwiGLU, the gpt2/qwen2 tokenizer. The single
difference is M-RoPE, and **for a text-only prompt M-RoPE and NEOX rope compute
the same rotation**: the (t, h, w) triple is all the token index until image
rows occupy a span of positions.

So the architecture landed on its own with **nothing in `engine/model/` changed** --
the graph is driven by Flags, not by Arch -- and was verified before any vision
work existed.

★ **THE VERIFICATION IS THE FIRST GENERATED TOKEN, OVER TWELVE PROMPTS, AND THE
CONTROL IS WHAT MAKES IT READABLE.** A four-prompt run at n=20 showed qwen2vl
diverging from llama.cpp on 2 of 4. The same harness on Qwen2-1.5B -- an
architecture this file already trusts -- diverged on 1 of 4. Compounding under
greedy decoding was doing that, not the new code. At n=1 nothing compounds:

    qwen2vl        12/12 identical
    Qwen2-1.5B     11/12 identical   <- the CONTROL, and it is the worse row

`TestGreedyMatchesLlamaCpp` prices every remaining divergence as a tie below
0.04 where the control's is 0.072.

★ **AND THE SAMPLER FLAGS COST TWENTY MINUTES FIRST.** llama.cpp without
`--top-k 1 -fa off -dev none` produced a different continuation, which read as a
model difference. `scripts/rungold.py` has carried those flags the whole time;
the ad-hoc command beside it did not. **Use the harness the goldens use.**

★ `RopeSections` IS PAIRS AND NOT DIMENSIONS, and the numbers say so: `[16, 24,
24, 0]` against head_dim 128, where 16+24+24 = 64 = 128/2. `configOf` refuses a
list that does not sum to NRot/2 -- a short one leaves a tail of every head
unrotated, which is fluent and which no tier can see, because every tier would
apply the same wrong rotation and agree perfectly. That is gemma's NEOX bug's
shape, pre-refused.

### The vision half: three of the four differences were smaller than they looked

| | Qwen2-VL | what it took |
|---|---|---|
| patch embed | conv3d, two temporal planes | **nothing in the engine** -- the planes are ADDED |
| merge | 2x2 spatial | the pixel shuffle that already existed |
| projector | 5120 -> 5120 -> 1536 | separating the shuffle from the matrix count |
| position | **2-D rotary, no table** | the one genuinely new piece |

★ **THE TWO PATCH PLANES ARE ADDED, NOT CONCATENATED, AND THE REFERENCE GRAPH
SAYS SO OUT LOUD.** llama.cpp runs IM2COL+MUL_MAT once per plane and then ADDs:
`node_3 = MUL_MAT(im2col, v.patch_embd.weight)`, `node_10 = MUL_MAT(im2col,
v.patch_embd.weight.1)`, `node_14 = ADD(node_3, node_10)`. A still image is fed
to both planes, so `x*W0 + x*W1 = x*(W0+W1)` -- one matrix, added at conversion,
and no conv3d anywhere in the engine. **Exact for an image and WRONG for video**,
which is why `foldPatchPlanes` says so rather than being a quiet optimisation.

★ **THE PATCH REORDER IS NOT NEEDED EITHER.** llama.cpp permutes the grid before
block 0 so each 2x2 block lands in four consecutive rows, then reshapes at the
merger. Attention is permutation-equivariant and the only op that reads an index
is the rotary, which takes the patch's own (row, column) either way -- so raster
order through the blocks with a spatial grouping at the merger computes the same
embeddings and emits no permutation kernels. The group's INTERNAL order has to
match and does: row-major, dy outer and dx inner, which is what the permute chain
`{1280,8,8} -> {2560,4,8} -> {2560,4,2,4} -> {2560,2,4,4}` spells out.

★ **WHAT DID NEED CHANGING IS THAT THE SHUFFLE AND THE MATRIX COUNT WERE ONE
BRANCH.** idefics3 shuffles and runs ONE matrix; llava runs TWO and does not
shuffle; qwen2vl_merger does both, which a branch keyed on "is there a second
matrix" cannot express. `Scale` decides the shuffle and `projW2` decides the
matrices, separately.

★ **AND TWO FILE PROPERTIES ARE READ FROM THE ARTEFACT BECAUSE THE KEY IS WRONG
IN A SHIPPED mmproj.** `clip.vision.feed_forward_length` is **1536** in
mmproj-Qwen2-VL-2B -- the TEXT model's embedding width, copied into the vision
section -- against blocks that are 1280x5120. Taking it sizes every FFN buffer
at a third of what the matmul writes, and `buildTower`'s existing "neither
expands to NFFN" check is what caught it. There is also no `scale_factor` key at
all, so the merge factor is derived as mm.0's k divided by NEmbd. RULE 7's "read
the TENSOR table, not the key summary", twice in one file.

★ **THE ffn_up/ffn_down NAMES ARE SWAPPED IN THIS mmproj AND NOTHING HAD TO
CHANGE**, because `buildTower` already picks fc1/fc2 by SHAPE and swaps the
biases with them. llama.cpp prints `load_tensors: ffn up/down are swapped` for
the same file. A layout that validates itself beat a naming convention.

### ★★★★★ AND THE REAL COST WAS THE ORACLE: A SYNTHETIC IMAGE CANNOT GATE THIS MODEL

`llama-mtmd-debug -p encode` dumps every intermediate with a sum, which is the
right instrument and was used correctly. The INPUT was the mistake.

**`--image gray` and `--image cb` both produce IDENTICAL PATCHES.** Gray
trivially; the checkerboard because a patch is 14 pixels wide and 14 is even, so
every patch sees the same pattern -- the reference's own patch-embed dump is
`0.0950` for all 1600 of them. With identical patches every value vector is
identical, and attention returns `sum(w_j * v_j) = v` whatever the scores are.
**The rotary cannot break that symmetry, because it rotates q and k and not v.**
So in exact arithmetic the tower must emit 400 identical rows.

llama.cpp's rows are NOT identical, because this ViT's last block has activations
around 1e7 -- `ffn_up_b-31` sums to -21,570,332 -- that amplify f32 rounding into
the output. **Comparing against those numbers is comparing two chaotic
trajectories from the same start.**

The bisection that produced, and the reading it invited:

    patch embed   row0 0.0359 -0.1602 0.0527   vs 0.0363 -0.1590    MATCHES
    ln1-0         row0 -0.0001 0.0012 0.0001   identical             MATCHES
    layer_out-0   140576.60 vs 140699.05       0.09%                 MATCHES
    layer_out-29  192446.72 vs 191711.42       0.4%                  MATCHES
    ln2-31        -486016.65 vs -491613.38     1.1%                  MATCHES
    fc1-31        -18928679 vs -21570332       12%
    layer_out-31  -316301.77 vs -1995370.25    6.3x
    the output    -26348.19 vs -1972.31        13x

Every stage matches until the massive-activation block, and then it does not.
A rotary sweep over five variants -- **including variant 4, NO ROTATION AT ALL**
-- moved the final sum from -26348 to -26879, which is what says the rotary was
never the variable and the input was degenerate.

★ **SO THE GATE IS END TO END, AND IT RUNS ITS VIOLATION.**
`TestQwen2VLDescribesAnImage` feeds a real picture and checks what the model
says, against llama.cpp on the same file:

    jitllm     "A colorful abstract design with a yellow circle in the center
                surrounded by red, blue, green, and white squares."
    llama.cpp  "A yellow circle is placed in the center of a red, blue, and
                green background."

and then feeds a BLANK image through the identical path, which answers *"a
colorful image of a cat with a red and white striped collar"*. Without that half
"it said yellow circle" proves nothing -- a tower wired to nothing still emits
fluent text, and this engine has shipped exactly that once before (RULE 7j, the
VLM that described a picture nobody had shown it).

★ **THE STANDING LESSON IS ABOUT ORACLE INPUTS, NOT ABOUT THIS MODEL.** A
degenerate input makes an oracle comparison unreadable in a way no amount of
care in the comparison can fix, and a uniform image is degenerate for any
transformer over patches. **Check that the reference's own output VARIES before
believing a disagreement with it.** The reference's 400 rows differing by 2.0
where the arithmetic says they must be identical was the tell, and it was
visible from the first dump.

★ **AND cmd/jitllm's IMAGE MARKERS WERE idefics3's FOR EVERY ARCHITECTURE.**
`imageSpans` looks each marker up by name and SKIPS one the vocabulary does not
carry, so a Qwen2-VL run got no image markers at all, spliced the embeddings in
naked, and described the picture anyway. That function's own comment already
says what that costs -- *"it answers fluently about nothing"* -- and the silence
came from the lookup being forgiving rather than from a wrong guess. Qwen2-VL's
wrapping is `<|im_start|>user\n ... <|vision_start|> EMB <|vision_end|> ...`,
and `<|image_pad|>` is NOT emitted: the embedding span IS the padding.

## ★★★★ A FULL DISK PUBLISHED A MODEL OF ZEROS, AND EVERY CHECK PASSED IT.

Found by making the qwen2vl gates run on the M4, which meant freeing that disk
and then filling it.

    gemma-2b.jlm   1,676,656,640 bytes apparent
                           4.1 MB allocated -- 1593 of its 1599 MiB are holes

It opened cleanly, reported container version 17, carried a correct
`jlm.Fingerprint`, was newer than its GGUF, and decoded to **"<pad>" at token 0
on a 0.000 tie**. The suite's converter cache served it on the next run.

★ **jlm.Fingerprint.Writer CANNOT SEE THIS, AND THE REASON IS STRUCTURAL.**
`jlm.Write` sizes the file with `Truncate` and lays the header, config, vocab,
tensor table and fingerprint down BEFORE the first weight -- so the fingerprint
is in the part that always gets written, and every later failure publishes a
container with a perfect header and holes where the model is. The freshness
check added the same morning asks WHO wrote a container; this asks whether the
write FINISHED, and they are different questions.

**Fixed by writing `dst+".part"` and renaming.** Same directory, so the rename
is atomic: a reader sees the old container or the new one. The same run on the
same full disk now reports *"jlm: v_attn_v: write ...jlm.part: no space left on
device"* and leaves nothing behind.

★★ **THE GATE TOOK THREE ATTEMPTS AND THE FIRST TWO PASSED AGAINST THE
VIOLATION.** This is RULE 10 arriving from inside a fix rather than from
outside it:

    k = 13 (a bad SHAPE)     refused in pass 1, BEFORE os.Create -- so it
                             leaves nothing either way and discriminates nothing
    a truncated Data slice   not refused AT ALL: PackWeights does not check its
                             input length, and the write SUCCEEDS
    poll for dst while
    the weights go down      **discriminates**, and reads "no out.jlm.part was
                             ever visible" against a direct writer

What separates an atomic writer from a direct one is not what is left after a
failure -- it is what exists at `dst` DURING the write. A direct writer has
already created it, sized it, and put a valid header in it, so a reader arriving
at that moment sees a container. Asserting on the aftermath cannot see that; the
first two attempts were both assertions about the aftermath.

★ **AND THE CHEAP CENSUS FOR THIS CLASS IS apparent-vs-ALLOCATED.** Sixteen
containers on that box: exactly one at 0%, the rest at 95-100%, where the 95s
are page padding and legitimately sparse. It needs no engine and no oracle, and
it found the single corrupt file immediately:

    for f in *.jlm; do stat -f "%z %b %N" "$f"; done

★★ **THE TRIGGER, AND THE OPERATIONAL RULE FOR THE M4: CLEAR THE CONTAINER
CACHE BETWEEN SLICES.** `TestGreedyMatchesLlamaCpp` sweeps every model under
2 GiB and `jlmOfPair` caches each container on disk -- ~8.6 GB for the text
sweep, ~5.5 GB for the vision gates, and nothing ever evicts. Run back to back
they do not fit, and the second one dies on ENOSPC.

    find ~/jitllm-arm/models -maxdepth 1 \( -name "*.jlm" -o -name "*.part" \) -delete

With that between them, both slices PASS. This is AGENTS.md's "the model suite
runs in slices" arriving for DISK rather than for memory, and until the atomic
write landed, running out of disk mid-sweep did not fail the sweep -- it
poisoned the cache for the next one.

★ **AND THE "99% FULL, 2 GiB FREE" READING OF THAT DISK WAS THE CACHE ITSELF,
WHICH IS WORTH CORRECTING BEFORE SOMEONE DELETES SOMETHING REAL.** The volume
reports 2.1 GiB free with the containers present and **8.7 GiB** with them
gone; the containers ARE the 8.6 GB. Nothing of the user's needed removing, and
the Go module cache, the OS update snapshots and the other projects on that box
were all considered and are all irrelevant.

★ **THE HOUR THIS COST WAS A ZSH GLOB, NOT A DISK.** `rm -f *.jlm *.part` over
ssh runs under zsh, which ABORTS THE WHOLE COMMAND when any glob matches
nothing -- so with no `.part` present the `rm` never executed, `df` kept
reporting 142 MiB, and the obvious readings were APFS snapshots pinning freed
blocks and a volume genuinely out of room. Both were wrong and both are
plausible. `find -delete` has no such failure mode. The same shell difference
already cost this project a measurement (zsh does not word-split unquoted
variables, so an ABBA loop produced empty output): **a remote shell is not the
local one, and a glob that matches nothing is where it shows.**

★ **WHAT THIS DOES NOT ESTABLISH.** How many past arm64 rows were taken against
a partially-written container is unknown and unknowable: nothing recorded
allocated-vs-apparent, and a zero container decodes rather than crashing. The
one measured instance is gemma-2b, and it was caught by a greedy
oracle gate rather than by anything that looks at files.

## gemma3 on the device: the post-FFN norm, localised and later explained as conditioning

gemma3 did not reach the device at first -- it had been
declining every block, and the reason was `tier.quantOf`, a second hand-written
`quant.Type -> kernels.Quant` switch that had never heard of Q5_0, which
gemma-3-1b's `attn_q` is. That is fixed and gated
(`tier.TestEveryPackedFormatReachesTheDevice`).

What is left is a host-vs-device difference **on ONE block**, which is where it
stops being reassociation. `model.TestDeviceMatchesHostOnABiasedModel` places a
single block and compares logits:

| model | logit NMSE, 1 block | max\|d\| |
|---|---|---|
| gemma-2b (gemma1) | **4.635e-05** | 0.72 |
| phi3 | 1.113e-04 | 1.00 |
| Qwen3-MoE (qdim = 2x NEmbd) | 1.621e-04 | 0.21 |
| Qwen3-1.7B (QK-norm) | 1.026e-03 | 0.29 |
| tinyllama | 1.744e-03 | 0.59 |
| qwen2 | 4.475e-03 | 0.94 |
| **gemma-3-4b** | **1.151e-01** | 19.98 |
| **gemma-3-1b** | **1.481e-01** | 12.89 |

★ **IT IS THE POST-FFN NORM, AND THAT IS A MEASUREMENT RATHER THAN A READING.**
Disabling each feature on BOTH tiers, so the two are wrong in the same way and
must then agree:

    both post-norms off, both tiers    **7.301e-04**   in band
    post-FFN off only, both tiers      **5.546e-05**   in band -- post-attn is fine
    post-attn off only, both tiers      1.247e-02      post-FFN alone is wrong
    (shipping, both on)                 1.403e-01

★ **AND SIX THINGS IT IS NOT, EACH ELIMINATED BY A DOSE AND NOT BY READING THE
CODE.** The reading was done too, four times, and found nothing -- which is why
the list is worth more than the narrative:

    the SWA rope selection   forcing the global table at block 0 reads 1.398e-01
                             against 1.403e-01 -- the two tables are identical
                             at this position, so the rope is INERT here and no
                             rope hypothesis can explain a block-0 error
    QK-norm                  Qwen3-1.7B has it and reads 1.03e-03
    qdim != NEmbd            the obvious correlation, since gemma3 is the only
                             model in the table where they differ -- REFUTED by
                             Qwen3-MoE, whose qdim is TWICE its NEmbd and which
                             reads 1.62e-04
    GELU, the embedding scale gemma1 has both and reads 4.6e-05, the best row
    a destination alias       pointing the post-FFN norm at a different scratch
                              buffer reads 1.421e-01, unchanged
    the weight's CONTENTS     read back off the device after upload: 0 of 1152
                              elements differ, worst 0.000000

★ **AND TWO CONTROLS THAT NARROW IT FURTHER AND MAKE IT STRANGER.** Both say the
two tiers agree whenever the post-FFN weight is not gemma3's own:

    post-FFN weight forced to ALL ONES on both tiers      **1.399e-04**
    post-FFN using nFFN/ffnNorm (known-good) on both      **2.193e-04**

So: the same call site, the same input, the same kernel, a weight vector that
reads back byte-exact -- and substituting any OTHER weight vector makes it
agree. That is not a permutation (a permuted `ffnNorm` would show in the second
control) and not a scale (NMSE is relative). **Unresolved**, and it must not be
written up as reassociation: gemma1 on the same harness is three thousand times
closer.

★ **THE BOUND IN THE GATE WAS 2.5e-1 AND IT WAS ACCOMMODATING THIS.** It was set
when gemma-3-4b first ran and nobody derived it. The section above is kept
because the cause turned out to be none of its candidates.

★★ **RESOLVED: IT WAS THE GATE's INPUT, NOT THE DEVICE. ID 1 IS
gemma's `<eos>`, AND FROM IT THE MODEL IS SO ILL-CONDITIONED THAT TWO HOSTS
DISAGREE BY MORE THAN THE DEVICE DID.** The gate fed `Forward(1)`. Two HOST
States on gemma-3-1b whose only difference is the KV cache's width (`WithKVF16`
true/false -- the rounding of V[0], a legitimate one) read, at position 0:

    from id 1 (<eos>)    host f32 vs host f16   NMSE 1.69e-01  max|d| 12.8
                         device vs host f32     NMSE 8.92e-03  (closer than
                                                the two hosts are to each other)
    bartowski's 1b       host f32 vs host f16   NMSE 3.75e-01  -- the 3.8e-01 its
                                                DEVICE arm failed at on a V100
    from BOS             host f32 vs host f16   NMSE 1.15e-05  max|d| 0.078

That is why the post-FFN controls above "agreed whenever the weight was not
gemma3's own": the trained post-FFN weight is what makes `<eos>` at position 0
ill-conditioned, and every substitute removed the conditioning, not a fault.
`cmd/jitllm verify` had already recorded the same 12.8 and moved its chain to
BOS; the gate had not.

★ **AND THE DECODE DIVERGENCE `verify` REPORTS (max|dlogit| 0.08 1.57 4.5 1.77,
"1/4 positions disagree", all 26 blocks on an RTX 3050 Ti) IS THE SAME THING ON
A DIFFERENT CHAIN.** verify's chain is BOS followed by the CPU's own greedy
tokens; the two HOST arms above, driven the same way, read 0.078 / 1.66 / 4.40
/ 1.93 -- the device's profile, with no device. Bisected by `ffn_resid` traces
at one placed block on a natural prompt, block 0's residual differs by NMSE
3.8e-09 at position 0 and 3.5e-06 at 1, against Llama-3.2-1B's 0.3-2.1e-04 and
Qwen3-1.7B's 0.6-1.2e-04 on the same probe; placing 1, 2, 4, ... 26 blocks
leaves the logit NMSE flat (5.6-7.5e-04 at position 1), so no later block adds
to it. There is no gemma3 device bug to fix.

★ **THE GATE's gemma3 ARMS NOW FEED BOS AND A SENTENCE AND COMPARE ALL 14
POSITIONS, UNDER THE SHARED 5e-2.** Derived on this box's cuda from both sides,
one block, worst position: working 1.72-1.88e-03 (4b), 6.3-7.6e-04 (1b),
4.74e-04 (bartowski's 1b, now an arm of its own); post-FFN norm dropped on the
device 3.0-3.4e-01; post-attn norm dropped 1.6-4.9. What it does NOT
discriminate is the split rope: block 0 on the global table moves only
1.9e-03 -> 6.3e-03 (4b) and 6.3e-04 -> 8.6e-04 (1b) at fourteen positions --
`TestSlidingWindowRunsOnTheDevice` is the gate for that.

★ **AND THE SAME AFTERNOON FOUND A CERTAIN BUG BESIDE THE UNCERTAIN ONE:
`l.nPostAttn` AND `l.nPostFFN` WERE IN NEITHER DEVICE FREE WALK.** They joined
the upload beside `nAttn`/`nFFN` and neither `dropLayerLocal` nor the migration
walk, so every gemma2/gemma3 block leaked two NEmbd vectors on release AND on
relocation -- while `l.auxBytes` COUNTED them, so the ledger charged bytes it
never gave back. That is the third instance of the hazard `ReleaseLayers`' own
comment describes ("two copies of the walk is how one of them acquires an eighth
tensor and the other does not"), and the comment is now attached to two of them.
Fixed. **The walks want to be one list rather than two**, which is the change
that would stop this recurring.

## ★ THE PREFIX CACHE ADDRESSED AT THE COMPUTE PAGE.

Two chat turns on Qwen3-Next-80B:

    "what is the capital of France?"                        15 tokens
    "what is the capital of France? you said before paris"  19 tokens
    shared prefix: 10 tokens          kv cache 0 of 19 reused (0%)

`kvPageTarget` is 256 for the ATTENTION KERNEL -- EmitAttnAcc re-reads the V
window once per 8-vector column block, so a small page multiplies that walk --
and that number was also deciding what the cache could NAME. Those are different
questions with different answers, and every implementation of this keeps them
apart: **vLLM's automatic prefix caching blocks at 16 tokens**, **SGLang's
RadixAttention** keys a radix tree over the token sequence, and **LMCache's**
chunk size is independent of the engine's block size on purpose.

`pageKey` was already content-addressing by prefix -- it hashes `seq[:end]` --
so the fix is a constant, not a design: `kvChunk = 16`, sealed at every chunk
boundary a partial page has passed, and a WALK over chunk boundaries on restore
rather than a search over positions.

    dense, two prompts sharing 40 of 45 positions   0 reused -> 32 (the chunk floor)

★ **A PREFIX SHORTER THAN A CHUNK MISSES HERE AND IN ALL THREE OF THOSE
SYSTEMS.** The ten-token overlap above still reuses nothing at 16. That is the
granularity, not a defect, and it is worth stating because the obvious next move
-- shrink the chunk until it hits -- pays for every prompt to help one.

### ★ AND A HYBRID GETS ONE RESTORE POINT PER PROMPT, PRICED

The attention pages can be sealed at every chunk. The gated delta rule's summary
cannot: at the end of a prefill the engine holds the state as of THIS position
and no earlier one, so a snapshot named for fill 16 would be the state after 32
tokens under a name claiming 16 -- the fluent-and-wrong shape this file exists
to refuse. A recurrence has no rewind.

Capturing them DURING prefill is the way to have more, and it is not free:

    one prompt's cache entry on Qwen3-Next-80B   88 MB
      of which the recurrent summary             79 MB  (36 layers x 2,195,456 B)
      of which every attention page               9 MB
    a point every 16 positions, 256-token prompt  1.26 GB   vs an 8 GiB default

So it is unbuilt and the number is recorded rather than the intention.

★ **AND THE HYBRID PROBES ONLY THE EXACT END, WHICH IS A CORRECTNESS CHOICE AND
NOT AN OPTIMISATION.** Walking chunks it cannot resume from made it fault every
layer's pages, discover the summary was missing, and give the whole prefix up --
reported through a counter whose message reads *"it could not be moved onto the
device"*, printed on a `-devices cpu` run where there is no device. One counter,
two causes; the message is neutral now and the probe does not ask for what it
cannot use.

    hybrid, identical prompt      prompt 8.532s -> 0.060s, 24 of 24 reused
    hybrid, divergent prefix      0 reused, no give-up, no false device message

## ★ THE TOKENIZER IS NOT A JIT TARGET, MEASURED; IT WAS THREE DATA-STRUCTURE COSTS.

Asked whether Encode should be generated code. `model.TestTokenizerCost`
(`JITLLM_TOKCOST=1`) prices it on ten vocabularies over AGENTS.md, forward.go
and a special-token/CJK/emoji mix, and `JITLLM_TOKDUMP` writes every id so two
builds can be compared token for token. Before: 0.29-0.93 M tok/s on one core
-- ~1.3% of a Llama-3.2-1B GPU prefill at ~5,000 tok/s, ~0.05% of a CPU one.
The profile had nothing a code generator would touch:

    partitionSpecial   64% (Qwen2) / 77% (Llama-3)   every special at every byte
    container/heap     25-44%                        an interface call per compare,
                                                     a boxed element per push
    merge-rank lookup  34% (Qwen2)                   a joined string per pair
    one SPM heap       47% (gemma-3)                 10 MB of heap over the whole text
    per-word buffers   25% (Qwen3)                   a fresh slice and queue per word

Fixed in that order: specials bucketed by first byte; two written-out heaps
running container/heap's exact sift order; ranks keyed by `[2]string`, with a
stale pair detected by the halves' lengths (a symbol's offset never moves);
buffers reused across words; and **one SentencePiece heap per word**. The last
is exact because a merge cannot cross a cut placed before a space mark whose
preceding character no token puts before a non-leading space mark
(`Vocab.spaceGlue`: "▁" everywhere, plus ">" for gemma-3's `>▁</`), and each
word's merges then depend only on its own pairs' order.
`tok.TestSPMWordHeapsMatchOneHeap` keeps the one-heap algorithm as the oracle
over every SentencePiece container (19 on Linux, 14 on the M4) and fails
against a cut before every space mark.

The reverse-merge map llama.cpp carries into resegment is unreachable here and
is deleted: it stored only token strings and was read only for strings that are
not tokens (a pair is queued only when its joined text is a token, and a pop
merges only when the halves still span exactly that text).

All 30 id sequences (10 vocabularies x 3 texts, ~2.3 M tokens) are identical to
the previous build. Paired ABBA in one pass, six P-cores, after/before:

    Llama-3.2-1B 7.42x   Qwen2-1.5B 6.39x   gemma-2b 5.36x   stories15M 5.25x
    tinyllama 5.15x      gemma-3-1b 5.12x   Phi-3.5 5.04x    olmoe 2.63x
    Qwen3-1.7B 2.47x     SmolLM2 2.13x

Now 1.8-4.0 M tok/s: under 0.3% of the fastest prefill on the box. What is
left is the pre-tokenizer (~20% on byte-level BPE) and the merges themselves.
**Generating code for the tokenizer would buy at most that 0.3% until prefill
is fast enough to make it matter**; the next lever, if one is wanted, is the
pre-tokenizer's per-character classification, which is table lookups.

### And its memory, measured the same way.

`TestTokenizerCost` reports a vocabulary's resident heap (a second `tok.New`
from the same container section, live heap after two collections) and one
Encode's allocations from the runtime's counters. Against the original
tokenizer, same probe, same box:

    vocabulary resident   Llama-3 25.9 -> 18.2 MiB   Qwen 18.0 -> 14.1
                          olmoe/SmolLM2 4.8 -> 3.8   SentencePiece unchanged
    per Encode, SPM       700-900 -> 21-29 B/token, ~0 allocs/token
    per Encode, BPE       ~1000 -> 100-230 B/token, 18-20 -> 0.8-1.8 allocs/token

★ **THE FIRST PASS REGRESSED THE RESIDENT HALF AND ONLY THIS PROBE SAW IT.**
Keying merge ranks by `[2]string` removed the per-lookup string and added 16
bytes to each of Llama-3's 280k keys: 25.9 -> 33.2 MiB. The table is keyed by
the two halves' token ids now (`bpeState.pairs`, `(left,right) -> (rank,
merged id)`, HF tokenizers' own shape), so a word is merged in raw bytes with
no byte-level encoding, and a merge whose halves or result are not tokens is
refused at load as HF refuses it. None of the 21 BPE vocabularies here has one.

★ **AND MOVING SPM's SYMBOLS PER WORD EXPOSED A BUG THE WORD CUT HAD SHIPPED
WITH.** `spaceGlue` searched each token for a space mark from byte 3, assuming
every token opens with one; gemma-3's `>▁</` has its space mark at byte 1, so
its glue was EMPTY and the cut before `▁</` after `>` was wrong. The version
that shipped passed its gate only because its symbols were one list over the
whole text, so a merge at a word's edge could still reach across. With one
list per word it cannot, and `TestSPMWordHeapsMatchOneHeap` failed on exactly
the input built for that token -- which is why that input is there.

### The pre-tokenizer streams now.

What was left of Encode's allocation was the pre-tokenizer: every stage turned
its whole input into a fresh `[]rune` and collected the prompt's pieces into a
new list for the next stage to walk (~75% of the remaining bytes on BPE). Each
stage now hands its pieces straight to the next (`preSplitEach`,
`pipeline.each`), and the last stage hands a word straight to the merge. A
piece is a slice of its input, found through a per-stage buffer of runes and
byte offsets (`runes`), and that buffer lives in a scratch pooled across
Encodes (`bpeScratches`, dropped past a million runes so one giant input does
not pin it).

★ **AN INVALID BYTE IS THE ONE PLACE A SLICE IS NOT THE OLD PIECE.** The
stages ran on `[]rune(text)` and emitted `string(runes)`, which turns every
invalid byte into U+FFFD -- as llama.cpp's own splitters do. A stage whose
input is not valid UTF-8 rebuilds its pieces from runes exactly as before; a
valid one never allocates a piece.

    per Encode, BPE    100-230 -> 4-33 B/token (4 when the pooled scratch is
                       reused; the probe forces a GC first, which empties it),
                       ~1-2 allocs/token -> a few dozen per CALL
    per Encode, SPM    21-29 -> 9-16 B/token (its output is presized)
    speed, ABBA vs the previous commit   BPE 1.13-1.29x, SPM 1.00-1.03x

All 30 id sequences still match the original tokenizer. The list-returning
splitters survive as collectors over the streaming ones, so every splitter
test tests the code Encode runs. `tok.TestEncodeIsSafeConcurrently` holds 16
goroutines to the serial ids under `-race` and fails with 25 races against a
scratch shared by every call.

## ★ gpt-oss IS THE TWELFTH ARCHITECTURE, VERIFIED NODE BY NODE AGAINST llama.cpp.

Read off `llama.cpp/src/models/openai-moe.cpp` and its running graph
(`llama-eval-callback`), not inferred. Against the mixtures jitllm already had,
gpt-oss differs in seven places and every one of them runs fluently when missed:

  1. **A learned sink per head** (`attn_sinks`, F32 [64]) joins each softmax's
     maximum and denominator after the score scale and carries no value. It is
     one extra slot on the score row (`softmaxSink`), so the generated softmax
     kernel serves it unchanged.
  2. **Biases on q, k, v and o** -- tensors this engine already loads.
  3. **Sliding window 128 on alternate layers, layer 0 first**: `set_swa_pattern(2)`,
     gemma2's rule, `SWAPeriod 2`.
  4. **YaRN**, factor 32 over an original context of 4096. It is a per-pair
     divisor (`nn.YarnFreqs`) plus a cos/sin magnitude of 0.1*ln(32)+1 folded
     into `AttnFactor`, so the rotary table -- and the device, which uploads the
     host's table -- needed nothing new.
  5. **A biased router**, with softmax over the top-k biased logits. That is
     arithmetically qwen3moe's softmax-then-renormalise, so the gating code is
     unchanged apart from the bias.
  6. **Every expert's gate, up and down biased** (one row per expert per bank).
  7. **swiglu-oai**: `min(gate,7)*sigma(1.702*min(gate,7)) * (clamp(up,-7,7)+1)`,
     a fourth gated `ActKind` generated on amd64 and arm64.

And the name trap: gpt-oss's `post_attention_norm` is the FFN's INPUT norm
(llama.cpp's `attn_post_norm` on the residual sum), qwen3next's case again, so
`retarget` maps it to `ffn_norm` for both. Missing it fails at load, not quietly.

**★ THE YaRN RANGE IS NOT ROUNDED, AND THAT IS A CHOICE AGAINST llama.cpp.**
gpt-oss's HF config says `"truncate": false`; OpenAI's reference and transformers
honour it; llama.cpp floors and ceils the correction range for every model
(pairs 8.09..17.39 become 8..18). RULE 7m: the model's definition wins, and
`FlagYarnExact` records it in the container because the GGUF does not.
`nn.TestYarnFreqsMatchTransformers` holds the divisors to transformers' own
`_compute_yarn_parameters` both ways.

**The gate** is `model.TestGPTOSSMatchesLlamaCppIntermediates`, one token ("The")
on gpt-oss-20b-Q4_K_M, sums of successive nodes against llama-eval-callback:

    attn_norm-0      -27.703668  llama.cpp  -27.703680   4.5e-07
    Kcur-0 (roped)    -4.467948              -4.469628   3.8e-04
    kqv_out-0        -38.699959             -38.742336   1.1e-03
    ffn_inp-0        -96.916086             -96.851616   6.7e-04
    ffn_moe_out-0     86.573989              87.151596   6.6e-03
    experts [13 21 17 2] identical; logits[0:3] -0.758 3.037 -0.673
                                  against    -0.755 3.016 -0.695

Position 0 is an identity rotation, so the roped K sum is the unroped one times
YaRN's 1.3466 on both sides -- a YaRN check at one token. The residual
differences are the two engines' activation quantization: llama.cpp quantizes a
k-quant matvec's input per 256 and this engine per 32. Each of the five pieces
above was removed in turn and the gate went red at the node where it enters:
no sinks (kqv_out 3.9x, experts change), no YaRN magnitude (K -26%), SiLU in
place of swiglu-oai (moe_out 3x, logits ~8), no router bias (expert 2 becomes
22), no down bias (moe_out -20%).

**Greedy text against llama-completion, 20 tokens:** "Once upon a time" and "The
largest planet..." are token-identical. "The capital of France is" agrees on
" Paris" and then splits on a 0.373 tie between `."` and `."\n\n`; fed the same
text up to the later split, jitllm's top choice is llama.cpp's (" short" over
" quick" by 1.02). `TestGreedyMatchesLlamaCpp` passes but compares only token 0
here, because re-tokenizing llama.cpp's text merges `."\n\n` -- a coverage limit
of that gate worth knowing.

    CPU, six P-cores, n=16: 7.7 tok/s decode; the card (RTX 3050 Ti) takes 0 of
    24 blocks, declined by name, and the host answers identically.

**★ AND IT FOUND gemma-3-4b RUNNING ITS GLOBAL LAYERS UNSCALED.** The converter
read no rotary scaling key for any architecture. gemma-3-4b ships
`rope.scaling.type linear`, factor 8, and llama.cpp divides its global layers'
angles by 8 while leaving the sliding ones at 1. Measured on "The capital" (three
positions; position 0 cannot see a rotary scale), the rotated K of global layer 5
summed to -21.585 unscaled and -18.775 scaled against llama.cpp's -18.903; local
layer 4 is -47.279 either way. `model.TestGemma3GlobalLayersAreLinearlyScaled`
holds it at 3%. The catalog gate had passed that model -- its answers stayed
fluent.

## ★★★★★ deepseek2 IS THE THIRTEENTH ARCHITECTURE, AND IT IS FOUR PRODUCT LINES.

DeepSeek-V2, V3 and R1, Kimi-K2 and GLM-4-MoE are ONE entry in
`convert.archOf`, which is the ratio that made it worth admitting at all.
Kimi-K2's own config.json says `"architectures": ["DeepseekV3ForCausalLM"]`;
GLM-4.7-Lite loads through the same builder in llama.cpp; Kimi-Linear carries a
`kv_lora_rank` of its own. `docs/design/model-coverage.md` is the survey the
decision was made from.

**Verified against transformers 5.17, three fixtures, decode and prefill:**

| fixture | what it isolates | worst NMSE over 8 positions |
|---|---|---|
| synth-deepseek-dense | MLA with no mixture at all | **5.16e-12** |
| synth-deepseek-lite | the ONE-STEP query, softmax router | **4.87e-12** |
| synth-deepseek | MLA + the V3 router | **3.13e-12** |

against a 1e-9 bound. The three differ by ONE field at a time on purpose, which
is what let the first failure name its own half: `synth-deepseek` read 2.49e-01
while `synth-deepseek-dense` read 5.16e-12, so the attention was right and the
mixture was not.

★ **AND THE FIRST synth-deepseek-lite FAILED AGAINST A CORRECT ENGINE, BECAUSE
THE REFERENCE IGNORES THE FIELD THE FIXTURE WAS BUILT ON.**
`DeepseekV3TopkRouter.forward` hardcodes `router_logits.sigmoid()` and never
reads `config.scoring_func`, so a V3 fixture asking for softmax records a
SIGMOID golden -- and the converter, which does read the key, then disagrees
with its own oracle at NMSE 8.9e-02. RULE 7m from the other side: **a reference
that silently ignores a key is not an oracle for that key.** The fixture is
built on the `deepseek_v2` class now, which implements `router_logits.softmax`
for real and whose config carries no `scoring_func` at all.

### attention.key_length is NOT the head width, and it is wrong by sqrt(3)

The source carries FOUR head widths for this architecture and they are four
different numbers:

    attention.key_length        576 = kv_lora_rank + qk_rope_head_dim
    attention.value_length      512 = kv_lora_rank
    attention.key_length_mla    192 = qk_nope + qk_rope
    attention.value_length_mla  128 = v_head_dim

The first two describe the ABSORBED cache row; the last two are the geometry
the graph is written in and what the attention softmax is scaled by. Reading
`attention.key_length` the way every other architecture does gives a head of
576, an un-rotated half of 512 where the weights want 128, and every score
scaled by sqrt(576) instead of sqrt(192) -- a model that loads, generates
fluent text, and is wrong by 1.73x on every score. Confirmed out of llama.cpp's
own converter (`conversion/deepseek.py`), not inferred.

### The two router biases are two tensors

`ffn_gate_inp.bias` is gpt-oss's: it biases the router matvec's LOGITS, before
the gating function, and its value reaches the returned weights.
`blk.N.exp_probs_b` is DeepSeek's `e_score_correction_bias`: added AFTER the
gating, to the scores, and dropped before the weights are read. llama.cpp keeps
them as two tensors and so does this container (`jlm.RoleExpProbsB`). The first
converter pass mapped both onto `RoleRouterBias`, which would have biased the
wrong quantity in whichever architecture lost the coin toss -- gpt-oss adding a
selection bias to its logits, DeepSeek biasing its weights. Both fluent.

### Four bugs found by running it end to end, every one silent

    bindPacked's list did not carry the five MLA matrices, so they bound once
    at build() and pointed at nothing after the first page-in. That function's
    own comment already says a tensor missing from it has a zero Packed.

    `.data != nil` is RESIDENCY, not presence, for a PAGED tensor. Using it to
    pick the query form asks whether the page is faulted in. The package's
    usual "presence of the tensor is the fact" holds for always-resident
    tensors and not for block matrices.

    The shared expert was keyed on its GATE, so DeepSeek's ungated one
    vanished. sigma(0) is 0.5, not 1, so an absent gate cannot be spelled as a
    zero logit -- it would halve every shared contribution.

    Config.MoE() is a property of the MODEL and the loader needed one of the
    BLOCK. Every forward path already branched on `l.router.data != nil` and
    was correct for a per-layer mixture before DeepSeek existed; build() asks
    the config because it runs before the tensors are bound.

★ **AND THE PAGING HALF NEEDED ITS OWN GATE, BECAUSE NO TEACHER-FORCED ONE CAN
SEE IT.** Every fixture is a few megabytes, so every block stays resident and
no MLA weight is ever re-bound. `model.TestMLARunsUnderAPageBudget` opens the
same container resident and at a ONE-BYTE budget -- where every block faults on
every visit -- and demands bit-identical logits. Against the two violations it
was written for it reads *"no host kernel reads ...q_a_proj.weight (F32,
32x64)"* and *"...kv_b_proj.weight[attn_k_b] (F32, 16x16)"*. The second is the
one a shape check cannot cover: `kbHead`/`vbHead` are slices OF `wkb`/`wvb`, so
rebinding the parents without re-cutting the views leaves every head reading
the frame the block used to live in -- another block's weights, at exactly the
right shape.

### What is NOT done, named rather than implied

`ForwardBatch` refuses MLA by name: it writes the KV cache through its own copy
of the projection and an MLA row is not what it writes. The DEVICE tier
declines it by name (`mlaDecline`), which is RULE 8a's honest state and where
gpt-oss's sinks sat before their plan learned the shape. arm64 is exercised
under `qemu-aarch64-static` and NOT on the M4, which RULE 11d says does not
cover the architecture. YaRN for DeepSeek is gated hermetically -- no
checkpoint with `rope_scaling: yarn` is on this box. And the renormalising
divisor has no floor: llama.cpp clamps at 6.103515625e-5 and HF adds 1e-20
where this engine returns 0/0 for an all-zero selection, which is reachable
only on a broken model but takes the logits non-finite when it is. The two
authorities disagree about which guard, so it is recorded in
`oracle.MoEDivergences` rather than chosen silently.

### As it stood in AGENTS.md

★ **AND deepseek2 IS SEVERAL PRODUCT LINES FOR ONE ENTRY, WHICH IS WHY IT WAS
ADMITTED.** Kimi-K2's own config.json says `"architectures":
["DeepseekV3ForCausalLM"]` -- model_type `kimi_k2`, q_lora_rank 1536, 384 routed
experts -- and so does Moonshot's Moonlight-16B-A3B. So DeepSeek-V2, V3 and R1,
Kimi-K2 and Moonlight are ONE capability rather than one name each -- the ratio
that makes RULE 7's list tractable at all.

★★ **AND THE GLM HALF OF THAT SENTENCE WAS WRONG, CHECKED AGAINST
THE PUBLISHED config.json RATHER THAN REMEMBERED.** It read "GLM-4.7-Lite loads
through the same builder in llama.cpp", and **there is no GLM-4.7-Lite**. The
two models that exist split the claim in half:

    GLM-4.5-Air     Glm4MoeForCausalLM      NO kv_lora_rank -- NOT MLA
    GLM-4.7-Flash   Glm4MoeLiteForCausalLM  kv_lora 512, q_lora 768 -- IS

★★★ **AND THE ONE THAT *IS* THIS GRAPH IS SUPPORTED NOW, WITH A FIXTURE AND A
GOLDEN: `Glm4MoeLiteForCausalLM` READS **1.81e-10** AGAINST transformers.**
Its published safetensors index carries `kv_a_proj_with_mqa`, `kv_a_layernorm`,
`kv_b_proj`, `q_a_proj`, `q_a_layernorm`, `q_b_proj`,
`mlp.gate.e_score_correction_bias` and `mlp.shared_experts.*` -- the names
`deepseek2HF` already maps, tensor for tensor -- so the entry is one line in
`hfArchTable`. **GLM-4.5-Air is NOT this entry**: no `kv_lora_rank`, so
mapping it here would convert a model this graph cannot express. It is
**glm4moe**, an arch of its own (below).

★★★★ **AND "THE NAMES MATCH" WAS NOT "THE GRAPH MATCHES": THE ONE LINE READ
1.502e-02 UNTIL A SECOND BUG CAME OUT, AND IT IS expert_weights_norm's LESSON
FOR THE SECOND TIME.** An ABSENT `scoring_func` is not one default for every
class:

    DeepseekV2ForCausalLM   implements router_logits.softmax() for real and
                            carries no scoring_func at all -- softmax is what
                            that architecture IS
    DeepseekV3ForCausalLM   hardcodes router_logits.sigmoid(), never reads it
    Glm4MoeLiteForCausalLM  hardcodes it too -- AND ITS CONFIG OMITS THE KEY

Every published V3 and Kimi config states "sigmoid" anyway, so `case "",
"softmax":` never mattered until a checkpoint left the key out and sigmoided
regardless. `convert.hardcodesSigmoid` is a LIST OF CLASSES rather than a
default, for RULE 7m's reason: the references disagree and the converter has to
know which one it is matching. Against the violation -- GLM dropped from that
list -- the gate reads 1.502e-02 on a 1e-9 bound.

★ **AND THE FIXTURE IS WHAT MADE THAT VISIBLE, WHICH IS RULE 7's BAR WORKING
RATHER THAN A FORMALITY.** `scripts/hfgold.py`'s `synth-glm4moelite` is built by
transformers' OWN `Glm4MoeLiteForCausalLM`. Shipping the `hfArchTable` line
alone -- which is all "the names match" justifies -- would have converted every
GLM-4.7-Flash with the wrong gating function on every token of every mixture
block: fluent, and wrong.

The stale claim had reached three shipping comments (`format/jlm/config.go` twice,
`convert/config.go`) as a justification for the entry's breadth. All three are
corrected.

★★★ **AND Moonlight-16B-A3B IS THE KIMI SHAPE AT A SIZE THIS BOX HOLDS, AND
IT RUNS.** Moonshot's own model, `DeepseekV3ForCausalLM`, 16B-A3B:
the V3 router (sigmoid, selection bias, grouped top-k, routed scale 2.446) with
**q_lora_rank ABSENT** -- a combination no fixture here has, since synth-deepseek
carries q_lora 32 and V2-Lite routes greedily.

    27 blocks, 64 experts, 6 used, 2 shared, 430 tensors in and out
    (a modern GGUF ships attn_k_b/attn_v_b already split, so unfuseMLA
     has nothing to do -- V2-Lite's legacy file is the one that needs it)
    prefill/decode vs decode  NMSE 0.000e+00
    batch vs decode           NMSE 0.000e+00
    llama.cpp                 19 of 20 teacher-forced, the 20th a 0.016 tie
    14.25 tok/s on six P-cores

★ **AND THE MODERN KEY LAYOUT IS THE BRANCH V2-Lite CANNOT REACH.** Moonlight
writes `attention.key_length` 576 and `value_length` 512 -- the ABSORBED widths
-- with `key_length_mla` 192 and `value_length_mla` 128 beside them. V2-Lite's
legacy file writes 192/128 into the plain keys and carries no `_mla` keys at
all, so the two files mean *different things by the same key* and only having
both proves the reader tells them apart. `jitllm run` reads `head_dim=192` off
Moonlight, which is the natural width taken from the `_mla` keys.

★★ **AND Kimi-Linear IS A THIRD CASE, IS NOT THIS ARCH, AND IS NOT "TWO
CAPABILITIES WIRED TOGETHER" EITHER -- THAT SENTENCE STOOD HERE AND IS WRONG.**
`KimiLinearForCausalLM` carries `kv_lora_rank` 512 and a `linear_attn_config`
naming twenty KDA layers against seven full-attention ones, so it LOOKS like
MLA plus qwen3next's gated delta rule. transformers' own class docstring says
what the difference is:

    "essentialy the same a gated delta net (GDN) but
     DECAY IS PER-CHANNEL INSTEAD OF PER-TOKEN"

`kernels.GatedDeltaStep` declares `pDecay` as `[vHeads]` and reads it ONCE PER
HEAD; KDA's is `[heads][head_dim]` and must be read per CHANNEL inside the k
loop. **That is a kernel on every backend, not a wiring job** -- and a SECOND
shape rather than an edit, because qwen3next still wants the per-head one. It
also needs a forget-gate kernel (`-exp(A_log) * softplus(g)` with a `g > 20`
cutoff, which RULE 8 says is generated), a LOW-RANK output gate where
`RoleSSMOutGate` is one matrix, and q/k/v JOINED at conversion where qwen3next
fuses them in the file.

**It is priced in `docs/design/model-coverage.md` 4b, against the engine as it
stands, and it is qwen3next's shape of cost rather than deepseek2's**: deepseek2
was admitted because one entry bought four product lines, and this is one model
for two new kernels. Both prior additions were made on the user's explicit
decision and this one is the same kind of call. Nothing here recommends it. Added on the
user's decision. What it cost: MLA (a latent KV cache whose key and value are
different widths), the V3 router (sigmoid gating, a selection-only bias,
grouped top-k and a routed scale), a per-layer dense/mixture split and an
UNGATED shared expert. `docs/design/model-coverage.md` is the survey it was
chosen from.

★★★ **AND IT RUNS ON THE DEVICE NOW -- ALL THREE BACKENDS, EVERY BLOCK PLACED.**
The sentence that stood here said it was "DECLINED ON THE DEVICE
BY NAME (`mlaDecline`), which is RULE 8a's honest state". That was true and is
not: `synth-deepseek` places **3 of 3 blocks** on cuda, vulkan and metal at
logit NMSE 8.0e-13 / 8.7e-13 / 2.0e-12 against the host, where it used to place
one and decline two.

    the V3 router      sigmoid gating, the selection-only bias, the grouped
                       top-k and the routed scale -- one capability, declined
                       together and landed together
    two extra kernels  ExpertGroupScore and ExpertGroupMask, because the IR has
                       NO NESTED LOOPS (every kernel in the package is depth 1)
                       and ranking an expert needs its group's rank, which needs
                       every group's score
    the ungated shared expert  a nil gate is UNGATED, not missing: sigma() can
                       never BE 1, so the multiply goes rather than being
                       neutralised
    the scratch        planUnion takes the mixture. DeepSeek leads with
                       NDenseLead dense blocks, so the first block the tier ever
                       saw was dense and every MoE buffer was unallocated

★★ **AND ALL THREE HOST ENTRY POINTS RUN MLA NOW -- `ForwardBatch` WAS THE
LAST REFUSAL AND IT IS DELETED.** `mlaProjectRows` is prefill's
batched projection with the row's slot and position taken from a function,
because **that is the only thing the two callers disagree about**: prefill's n
rows are successive positions of ONE sequence and a batch's n rows are n
DIFFERENT sequences. `model.TestMLABatchMatchesDecode` is the RULE 11c gate --
three sequences, six steps, NMSE **0.000e+00** on all three fixtures AND on the
real 11.9 GiB V2-Lite, and **1.26-1.89 against a violation that gives every row
slot 0.**

★★★ **AND WIRING IT FOUND A LIVE BUG IN THE PATH THAT WAS ALREADY BUILT:
PREFILL SOFTMAXED DeepSeek AT A SCALE 1.59x OFF DECODE's.** `forward.go` has
read `c.AttnScale` since YaRN's magnitude correction landed; `prefill.go` and
`batch.go` each derived `1/sqrt(HeadDim)` themselves. Read off the container
rather than off a config.json:

    DeepSeek-V2-Lite   HeadDim 192   AttnScale 0.114721385
                       1/sqrt(HeadDim) 0.072168784   ratio **1.589626**

**For every architecture with no log multiplier the two expressions are equal**,
which is why fourteen models and every gate in the suite agreed -- and why the
one model that disagrees is the one no gate opened. On the real V2-Lite the
pre-fix engine reads prefill-vs-decode **7.160e-03** and batch-vs-decode
**1.696e-02** against 0 after.

★★★★ **AND THE REAL MODEL WENT ON THE CARD, WHICH IS WHAT FOUND THE NEXT
ONE: `-devices cuda:0` ON DeepSeek-V2-Lite READ max|dlogit| 14.5 AGAINST A 0.94
BAND, AND THE BUG WAS ON THE HOST.** `jitllm verify -devices cuda:0`
on the 11.9 GiB container, 27 blocks, ~10 placed per token:

    before   3/8 positions disagree, max|dlogit| **14.5**
    after    0/8 disagree, max|dlogit| **0.557**, gpu 26.00 tok/s against
             the host's 7.61 -- **3.42x**

★ **RULE 11c's OWN PRESCRIPTION IS WHAT LOCATED IT, AND GUESSING WOULD HAVE
GONE TO THE DEVICE.** "Move the seam and see whether the error tracks the depth
below it" -- swept at 1/2/3/5 blocks it reads **0.24 / 0.39 / 0.54 / 0.53**, all
INSIDE the band. So the device blocks were never wrong, and the variable was
somewhere else in that command: the **f16 KV cache**, which is the amd64
DEFAULT and which every MLA gate in the tree pins to f32.

    host f32 KV vs host f16 KV          0.588   <- the recorded 0.2-0.94 band
    device + f16, 1 block, before      22.925
    device + f16, 1 block, after        0.588   <- the host's own figure

★★ **THE MECHANISM: `SetDeviceLayers` STEPS AN f16 CACHE DOWN TO f32 BEFORE
IT OFFERS A BLOCK, AND THE STEP-DOWN RE-EMITTED ATTENTION AT THE WRONG WIDTHS.**
`stepDownKVToF32` called `AddAttn(Cfg.HeadDim, ...)`, which is
`AddAttnKV(hd, hd, ...)` -- so on an MLA model it rebuilt the scores at **192**
where the cached row is **576** and the accumulate at 192 where the value is
**512**. `NewState` had derived the pair correctly since MLA landed; the second
site derived it again and got it wrong. `Config.attnWidths()` is the one
derivation now, and both call it.

★★★ **AND THE REASON NO FIXTURE GATE COULD HAVE CAUGHT IT IS STRUCTURAL,
WHICH IS THE PART WORTH KEEPING: THEY ASSERT *EVERY BLOCK PLACED*.** With 3 of 3
blocks on the card NO host attention runs, so the wrong kernels are emitted and
never called. The failure needs a PARTIAL seam -- the one configuration a gate
whose assertion is "the count is NLayer" excludes by construction.
`model.TestMLAStepsTheKVDownAtTheRightWidths` places **ONE** block on purpose
and leaves the rest home: clean 0.0018-0.0042, and **3.28-5.62 against the
violation** on the fixtures, 22.9 on the real model.

★ **THE STANDING LESSON IS RULE 11d's, ARRIVING AS A KNOB RATHER THAN AS AN
ARCHITECTURE.** Every MLA gate passes `WithKVF16(false)` for a good reason -- an
f16 cache moves logits through the same 0.2-0.94 band a seam does, so a gate
hunting a kernel bug must hold it still. The cost is that the SHIPPING default
was never executed on this architecture. **A gate that pins a knob does not
cover that knob's default, and the pin is invisible in a green line.**

★ **THE GATE FOR THE SCALE DOES NOT NEED THAT MODEL, WHICH IS THE PART WORTH
COPYING.** No fixture on this box carries `rope_scaling`, and waiting for one to
exist is how this survived. `model.TestAttnScaleReachesEveryPath` PERTURBS
`Cfg.AttnScale` by V2-Lite's own 1.5896 and demands all three paths move
together -- so an 80 MB F32 fixture becomes the model that has the feature, in a
third of a second. It reads 2.460e-02 against the two lines this replaced.
`JITLLM_MLA_MODEL=<name-or-path>` still points the two MLA gates at the real
container when someone wants the slow confirmation.

★★ **AND THE SELECTION BIAS THE ROUTER READS WAS ZERO IN THE FIXTURE, SO EVERY
GATE THAT CLAIMED TO COVER IT COMPARED TWO ARMS THAT COMPUTE THE SAME
FUNCTION.** `DeepseekV3TopkRouter` declares `e_score_correction_bias` as an
`nn.Buffer`, and `scripts/hfgold.py` randomises `named_parameters()` -- so it
stayed at the zeros transformers initialises, and a zero correction is the
IDENTITY. Found by running `internal/oracle`'s `MoEFaultBiasedWeights` against
the DEVICE: taking the weights from the BIASED scores **passed**. With the
fixture fixed it fails at **1.417e-01** against a clean 8.0e-13. RULE 10's
"check the configuration under test was actually SELECTED", arriving as a
fixture VALUE rather than as a missing flag -- the tensor was present,
converted, uploaded and read, and it did nothing.

★ **AND THE REAL MODEL CONVERTS AND RUNS, WHICH NO FIXTURE COULD HAVE TOLD
YOU.** DeepSeek-V2-Lite.Q4_K_M, 10.4 GB: 377 tensors become 404 (unfuseMLA
splitting all 27 blocks), `attn_k_b` lands q8 and `attn_v_b` keeps the source's
q4s, and it generates at 10.63 tok/s on six P-cores. Two bugs were reachable
only from a GGUF, because this architecture was built and gated entirely from
SAFETENSORS: nothing read `attention.kv_lora_rank`, and `NFFNShExp` came from a
key llama.cpp does not write for deepseek2 (`expert_shared_count` x
`expert_feed_forward_length` is the width).

★★★ **AND IT IS VERIFIED, 20 OF 20 TOKENS AGAINST llama.cpp -- AFTER THE
ORACLE SAID IT WAS WRONG.** "It generates" was committed as running-not-verified
and the caution paid: free-running greedy on the same GGUF diverged at the
FIRST generated token, both arms coherent.

    llama.cpp  The capital of France is Paris.
    jitllm     The capital of France is a must-see for any first-time visitor

    teacher-forced   18 of 20 and 20 of 20 positions agree
    free-running     20 tokens, up from the recorded 0

Two more config bugs, both invisible to every fixture because all three are
safetensors and both live on the GGUF path. **`expert_weights_norm` is NOT one
default for every architecture** -- llama.cpp's qwen3moe builder passes `true`
as a literal and its deepseek2 builder reads the key with a default of FALSE,
and V2-Lite ships no key with `norm_topk_prob: false`. **And DeepSeek does not
scale cos and sin at all**: its reference divides `yarn_get_mscale(factor,
mscale)` by `yarn_get_mscale(factor, mscale_all_dim)`, which for V2-Lite's
0.707/0.707 is exactly 1. The SQUARE in `AttnScale` was the half that was right.

★ **AND THE GATE COULD NOT REACH THE MODEL, WHICH IS WHY IT TOOK A DAY.**
`TestGreedyMatchesLlamaCpp` skips anything over 2 GiB without `JITLLM_SLOW=1`,
all-or-nothing over fourteen models, on the reasoning that "the architectures
the big two exercise are covered by smaller files in the same directory" --
true of llama and phi3, FALSE of deepseek2, whose smallest checkpoint is
9.7 GiB and every fixture of which is safetensors. `JITLLM_GREEDY_MODEL=<substring>`
names one.


## Embedding models: the BERT encoders and the pooled decoders

Ollama's embedding families (C12a/C12b in docs/design/ollama-coverage.md) run
through one API, `Model.NewEmbedder` / `jitllm embed`:

    bert, nomic-bert    engine/model/encoder.go -- a post-norm encoder built from the
                        vision tower's kernels (LayerNorm32JIT, the f32
                        attention scores/accumulate, Act32JIT); no decoder State
    qwen3-embedding,    the ordinary decoder graph through prefill, with the
    EmbeddingGemma      container's pooling bits (jlm.FlagPoolMean/CLS/Last) and
                        jlm.FlagNonCausal deciding the readout and the mask
                        (engine/model/embedstate.go)

Goldens: scripts/embedgold.py, six texts (accents, CJK, punctuation, a 62-token
paragraph) plus a ~600-token one for the long-context models. Worst cosine per
model, jitllm against each reference:

    model                         llama.cpp b10825   sentence-transformers f32
    all-minilm (Ollama f16)       0.9999998          0.9999983
    snowflake-arctic-embed:22m    0.9999998          0.9999995
    mxbai-embed-large (f16)       0.9999996          0.9999896
    nomic-embed-text (f16)        0.9999993          0.9999999
    embeddinggemma (Ollama bf16)  0.9999911 (*)      0.9999817
    qwen3-embedding 0.6b f16      0.9999990          0.9999998
    qwen3-embedding 0.6b (Ollama  0.9994639          0.9989819 -- llama.cpp on
      Q8_0)                                           the same file: 0.9989899

(*) b10825 cannot load Ollama's embeddinggemma -- it is arch `gemma3` with
`dense.0/dense.1`, and llama.cpp wants `gemma-embedding` -- so its arm is
unsloth's BF16 conversion of the same weights, which carries no dense heads; the
comparison is taken before them.

★ **A Q8_0 FILE IS BOUNDED AT ~0.999 OF THE FLOAT32 MODEL FOR EVERY ENGINE.**
Ollama ships qwen3-embedding as Q8_0 and last-token pooling makes the vector a
single row of a 28-block network: llama.cpp and jitllm both land at 0.9990 of
the float32 model, and jitllm's own f16-vs-f32 KV cache moves it by 0.99947. The
same model from Qwen's f16 GGUF reads 0.9999998, which is what says the GRAPH is
right. The gate therefore holds a quantized file to "no further from float32
than llama.cpp on the same bytes, within 2e-4" and 0.999 against llama.cpp.

★ **VIOLATIONS RUN IN-PROCESS** (`TestEmbeddingGateDiscriminates`), worst
cosine against float32: CLS pooling on a mean model 0.383, no token-type row
0.408, the post-attention LayerNorm without its weight 0.484, EmbeddingGemma
causal 0.566, its window four times too wide 0.969 (only the long text sees
it), no dense heads -0.040, mxbai mean-pooled 0.948, nomic without rotary
0.415, qwen3 mean-pooled 0.677, bidirectional 0.168, without its EOS 0.434.

WordPiece is llama.cpp's llm_tokenizer_wpm_session: token ids equal the models'
own HuggingFace tokenizers on every golden text. strip_accents needs NFD, which
Go's stdlib lacks, so scripts/genunicode now emits llama.cpp's pinned 15.1
`unicode_ranges_nfd` beside the existing tables (every existing table
byte-identical, so no token id moved).

Not done: a bidirectional decoder on a device (it embeds on the host; the
encoders run there, `docs/design/decision-models.md` section 10), the Unigram
tokenizer (C12c: bge-m3,
paraphrase-multilingual, granite-embedding, snowflake-arctic-embed2),
nomic-bert-moe, rerankers (pooling_type 4 is refused at conversion), and input
longer than the position table (refused, not truncated as sentence-transformers
does at max_seq_length).

### As it stood in AGENTS.md

★ **AN EMBEDDING MODEL IS A READOUT, NOT ALWAYS AN ARCHITECTURE.** bert and
nomic-bert are real graphs (post-norm encoder, `engine/model/encoder.go`, no decoder
State); qwen3-embedding and EmbeddingGemma are qwen3 and gemma3 whose container
carries pooling bits and `FlagNonCausal` (`jlm.FlagPool*`), run through prefill
(`engine/model/embedstate.go`). One API serves both: `Model.NewEmbedder`, `jitllm
embed`. The gate is `model.TestEmbeddingsMatchReference` against llama.cpp AND
float32 sentence-transformers goldens (`scripts/embedgold.py`), with
`TestEmbeddingGateDiscriminates` running the violations in-process.
`docs/engineering-history/model-correctness.md` has the numbers, including why
a Q8_0 file is bounded at ~0.999 of the float32 model for EVERY engine.
## ★★★ llama4 IS THE FIFTEENTH ARCHITECTURE: FIVE GRAPH FEATURES NO GGUF STATES.

Added on the user's request, text only. `docs/design/ollama-coverage.md` section
6 is the spike it was built from; llama.cpp's `src/models/llama4.cpp` and
transformers 5.17's `modeling_llama4.py` are the two references, read side by
side.

| feature | container | host | device |
|---|---|---|---|
| chunked attention, 3 layers in 4 | `FlagSWAChunked` + SWAWindow/Period | `Config.AttnWindow` (the one derivation for decode, prefill and the batch) | `kernels.ChunkWindow` (a negative window on every windowed score kernel), `FlashShape.Chunked` |
| no rotary on the 4th | `FlagNoPEGlobal` | `Config.RopeAt` in `attnPrep` | `LayerPlan.RopeAt`: the vision path one layer at a time |
| temperature on those layers | `AttnTempScale/Floor/Offset` | folded into the softmax scale (scores are linear in q) | `kernels.ScaleRowsByCount` over a table filled once (the IR has no log; floor(...) is an integer) |
| weightless q/k norm, after the rotary | `FlagQKL2Norm` | RMSNorm against ones, after RoPE | the same per-head norm BEFORE RoPE: a norm with no weight commutes with a rotation |
| top-1 sigmoid, input-weighted | `FlagExpertWeightIn` (+ Sigmoid, NoExpertNorm) | gate and up scaled by w (linear, so W(w h) = w W h), downs summed at 1 | `kernels.ActMulWeighted`, decode and the grouped prefill (index through each sorted column's pair) |
| dense/mixture interleave | `MoEStep` | `Config.MoEAt` | per-block plan, as DeepSeek's dense lead |

**No container version.** The four numbers are an optional tail of the
length-delimited config record, written only when non-zero; a container that
writes them is Arch 16, which a v26 reader refuses by code. A first draft
bumped to v27 and its test runs re-converted shared containers under other
agents -- the tail is the answer, not a private models directory alone.

**Tokenizer.** Ollama's llama4 says `pre = "default"`, which is llama3's split;
Llama 4's own tokenizer.json is o200k's (case runs, `\p{N}{1,3}`,
`ignore_merges: true`). A llama4 vocabulary carrying the fallback label is
resolved to "llama4" at conversion (RULE 7m: the model's artefact is the
authority). `pretok.Table` gained it from `unsloth/Llama-4-Scout-17B-16E-Instruct`,
which the gated meta-llama repo matches pointer blob for pointer blob
(b1fde397..., 27,948,578 bytes). llama.cpp files llama4 under gpt-4o with
ignore_merges false: a known, chosen divergence.

**Measured.**

    synth-llama4 (F32, scripts/llama4gold.py) vs transformers, decode, prefill
      at every length, batch                        worst NMSE 1.004e-11 (bound 1e-9)
    each feature removed on the host               2.4 / 4.9e-2 / 1.2 / 4.3e-2 / 2.2e-2 / 4.6e-5
    synth-llama4 on the device, every block placed
      laptop cuda  decode 5.0e-12  prefill 2.5e-13
      laptop vulkan decode 7.0e-12 prefill 2.5e-13
      V100 (the V100 host, sm_70) cuda and vulkan, the same numbers
    synth-llama4-q8 (Q8_0 twin) vs transformers    3.4e-03 (int8 activations)
    grouped batched prefill (counted, no demotion)  cuda 5.2e-4 / V100 7.3e-4, vulkan 3.1e-14;
                                                    a reversed column->pair index 2.6e-2
    Llama-4-Scout-17B-16E-Instruct Q4_K_M (unsloth, 2 parts, 66.3 GiB container)
      host vs llama.cpp (TestGreedyMatchesLlamaCpp)  20 of 20 teacher-forced, 20 greedy tokens
      "The capital of France is Paris. Paris is known as the City of Light and is famous for its art, fashion, and"
      verify, 20 of 48 blocks on two V100s          2/24 flips, both inside their own movement
                                                    (margins 0.049, 0.196), max|dlogit| 0.54-2.03
      Moonlight on the same two cards, same tool     1/24, max|dlogit| 0.18-1.45 -- the same band

**Speed, indicative only -- one sample each, two harnesses, NOT a RULE 2 row.**
the V100 host (2x E5-2680 v4, busy host), n=24/32:

    jitllm   host 2.33 tok/s   20 blocks on 2x V100 6.50 tok/s
    llama.cpp -ngl 0  3.44     -ngl 20                5.39

**Found on the way, not a llama4 bug and not fixed here.** `jitllm verify
-prefill` demands bit-identity between the device's batched prefill and its
per-token path whenever the attention is not on the matrix instruction, and it
fails on every model tried -- Llama-3.2-1B 0.81 (laptop, `JITLLM_NO_GPU_MMA=1`),
Qwen3-MoE 1.31, Moonlight 0.96 and gpt-oss-20b 1.12 on a V100 -- with the base
binary (2ab29e1), a08c29e and 389fc4e all reading the same. The tool's
premise was established earlier ("0/9 at max|dlogit| 0 on six models");
somewhere since, the two paths stopped being the same arithmetic.
Accumulate splits, the KV width and the norm partial count are not it
(each pinned, still 0.75).

**Not built.** The vision tower (34-block ViT, 2-D rotary, pixel shuffle); a KV
ring for the chunked layers (their pages hold the whole history today, so a
long context costs Scout's full cache); the NoPE-layer temperature is exercised
by the fixture's floor of 4 and never by Scout below 8191 positions; Maverick
(128 experts, no QK norm, step 2) is covered by the same code and not run.

### As it stood in AGENTS.md

★ **llama4 (text) RUNS ON THE HOST AND ON CUDA AND VULKAN, ADDED ON
THE USER'S REQUEST.** Five differences from llama, none visible in a GGUF
(llama.cpp hardcodes them against the name), so the container states them:
chunked attention on three layers in four (`FlagSWAChunked` -- a different
MASK from a sliding window, identical inside the first chunk), no rotary on
the fourth (`FlagNoPEGlobal`), attention temperature on those (`AttnTemp*`),
a weightless q/k norm after the rotary (`FlagQKL2Norm`), and a top-1 sigmoid
router whose weight scales the expert's INPUT (`FlagExpertWeightIn`), with
dense/mixture interleave (`MoEStep`). **No container version**: the four
numbers are an optional tail of the config record written only for this arch,
which a v26 reader already refuses by code. `scripts/llama4gold.py` (F32 and
`Q8=1`) against transformers at 1.0e-11; Scout Q4_K_M 20 of 20 teacher-forced
against llama.cpp on the host. `docs/engineering-history/model-correctness.md`
has the measurements; vision (the 34-block ViT) and a KV ring for the chunked
layers are not built.

## ★ FOUR "C6 DISAGREES WITH llama.cpp" FAILURES WERE THE GATE'S GOLDEN, NOT EITHER ENGINE.

`TestGreedyMatchesLlamaCpp` failed on synth-phi2 (and -q8), -nemotron,
-stablelm and -starcoder2 with CONFIDENT teacher-forced disagreements (margins
0.76-1.52) while `TestC6MatchesTransformers` held the same fixtures at ~1e-12
and real checkpoints of the same archs agreed 17-20/20. The golden was the
cause: it re-tokenized llama.cpp's printed completion with jitllm's tokenizer
and called that llama.cpp's ids. Re-tokenizing is not decoding's inverse -- a
random-weight model emits " JO" + "B" where the canonical split is " JOB" --
so the teacher-forced arm fed both engines a sequence neither had produced. On
starcoder2 the two engines' TEXTS were byte-identical and the gate reported a
confident disagreement at token 4.

Measured by reading llama.cpp's real ids out of its `--prompt-cache-all`
session file (magic, version, count, ids) and teacher-forcing jitllm on them:
all four fixtures, three prompts each, agree at every one of 60 positions
(synth-phi2-q8 has one 0.073 tie), and llama.cpp's CPU and CUDA builds emit
identical ids on all five synth models tried. No layer bisect was needed --
there was no disagreement to localise. Neither side was wrong.

Two things came out of fixing it:

- **A real jitllm bug, `tok.Vocab.Decode` rendered UNUSED tokens.** Type 5 is
  a placeholder llama.cpp's converter invents to pad a vocabulary to the
  embedding's rows ("[PAD50954]" for phi-2, "<unused_256000>" for gemma);
  llama.cpp's `token_to_piece` renders it as nothing and the model's own
  tokenizer does not contain it. synth-phi2 emits id 50954 at token 13, and
  with the ids agreeing the greedy TEXT still parted there.
  `tok.TestDecodeRendersUnusedAsNothing` (BPE and SPM) fails against the old
  Decode at `" Paris[PAD50954] Paris"` and `"  Paris<unused_256000>  Paris"`.
- **Saving a session is not free of arithmetic.** `common_prompt_batch_decode`
  decodes all but the last prompt token, saves, and decodes that token ALONE,
  which changes the first logits' reduction order: SmolVLM's greedy chain parts
  at a 0.037 tie at token 12 under it (its ordinary CUDA run and its CPU run
  part from each other too). So the gate runs llama.cpp twice: the session run
  gives ids for the teacher-forced arm, which tolerates a tie by construction,
  and the ordinary run's TEXT is the greedy chain, compared as a prefix -- the
  technique `forward_test.go`'s internal twin already used.

The gate still fails against a real violation: the host rotary pair order
flipped (`Neox: !cfg.RopeNeox`) fails every rotary fixture teacher-forced
(synth-phi2 position 13 at margin 0.547, synth-nemotron positions up to 2.55),
and the phi-2 head bias dropped from `finishLogits` fails both phi2 fixtures
on the greedy ratchet (4 and 9 tokens against the recorded 20).

## Tool calling, through the model's own template

Both HTTP shims accept tools: OpenAI `tools` / `tool_calls` / `role: "tool"`,
Anthropic `tools` / `tool_use` / `tool_result`, streaming and not. The list is
handed to the model's chat template as `tools` (`Model.ChatIDsTools`), so the
prompt is the one the model was trained on; the reply is read in the model's
own tool syntax (below), and only a call to a DECLARED name is taken, so an
answer in JSON stays an answer. A stream sends the text that cannot become a
call as it arrives, and each call as soon as its markup closes.

**The prompt is held to transformers byte for byte**
(`model.TestToolPromptMatchesTransformers`, goldens from `scripts/toolgold.py`
through `render_jinja_template` on each container's own template): Qwen2.5,
Qwen3 and Llama-3.2 match exactly. Getting there found two engine bugs:

- **`tojson(indent=N)` alphabetised keys** -- it went through a Go map --
  and Llama-3.x templates render every tool with it. It now uses the
  order-preserving encoder and re-indents (`jinja.TestToJSONKeepsKeyOrder`).
- **A schema decoded into `map[string]any` is sorted too.** `jinja.FromJSON`
  decodes a tool list into ordered Values, so `{"type", "function"}` and a
  schema's properties reach the template in the client's order, as Python's
  `json.loads` keeps them.

A template that never reads `tools` (this box's Mistral-7B-v0.3 GGUF carries
the pre-tools one) refuses the request by name rather than rendering the
conversation as if no tools were declared.

Live, `jitllmd` on six P-cores, greedy: Qwen2.5-1.5B calls
`get_weather({"city":"Paris","unit":"celsius"})`, and given the result answers
"The current temperature in Paris is 14 degrees Celsius, with light rain."; the
Anthropic route returns a `tool_use` block. That model copies its own template's
typo -- the Qwen2.5 template shows the call as `{{"name": ...}}` and the 1.5B
doubles the OPENING brace -- which the parser tolerates. Llama-3.2-1B makes the
first call correctly and then calls again instead of answering from the result,
on a prompt byte-identical to transformers'; that is the 1B model.

### Each family's syntax, and forced calls

The first parser knew two shapes, Hermes `<tool_call>{...}` and a bare JSON
call opening the reply. A model's syntax is now a property read from its chat
template (`model.ToolSyntaxOf`) -- the markup the template writes or describes
is what the weights were trained to emit -- rather than every parser tried on
every reply. Twenty-four are read (`engine/model/toolsyntax.go`): Hermes, the
Llama 3.x JSON reply, the Qwen3-Coder XML (Qwen3.5, Nemotron 3), Seed-OSS,
GLM-4.5+ and Ling, MiniMax-M2, DeepSeek V3/R1 and V3.1/V3.2 tool tokens,
DeepSeek V4's DSML, Kimi-K2's sections, Kimi-K3's XTML, harmony, Mistral v0.3
and the v11+ `[TOOL_CALLS]NAME[CALL_ID]ID[ARGS]{...}`, Granite 3, Nemotron-H,
Command R7B and Command-R, Apertus, Phi-4-mini, Llama 4 and LFM2 Python calls,
Hunyuan, Gemma 4. The source of each is the family's own template; vLLM's
tool parsers were the prior art where a template only describes the call
(Phi-4-mini, LFM2).

**The reply is read raw.** Several syntaxes are written in control tokens the
detokenizer drops (`[TOOL_CALLS]`, harmony's `<|channel|>` header that carries
`to=functions.X`, Kimi's section tokens), so the server hands the shims a
`model.ToolStream` that reads the token ids: the raw text spells every control
token out (`tok.Vocab.Literal`), and the text shown is the chat text, cut
around the calls. With the raw text replaced by Decode's,
`TestToolStreamReadsControlTokens` loses the gpt-oss and Mistral calls.

**tool_choice "required" and a named tool force a call** through the
structured-output machinery, not a second one: `Model.ToolGrammar` writes the
family's own spelling around the tool's JSON Schema (an XML family's
arguments one by one, a string's value as text up to its closing tag), the
server compiles it with the same matcher, and the mask is the same generated
`nn.Add32JIT` bias. The tokenizer the tool grammar is matched over offers the
syntax's marker control tokens by their literal text; a structured-output
grammar never sees them, so a JSON string cannot spell `<|call|>`. A model
that reasons first writes its block before the call: inside it when the
prompt opened one (DeepSeek-R1), never when the prompt wrote it closed
(thinking switched off), optional when the template writes one -- an earlier
version allowed the block whenever the template mentioned `</think>`, and
Qwen3.5 with thinking off wrote its prose answer, then `</think>`, then the
call. A schema keyword the converter does not build constrains that tool's
arguments to any JSON object rather than refusing the request; a family that
writes arguments one by one refuses it instead.

Live, greedy, six P-cores, laptop: on a prompt each answers in text
(tool_choice auto), Qwen3-0.6B, Qwen3.5-0.8B, gpt-oss-20B and
DeepSeek-R1-Distill-Qwen-1.5B call under "required" and under a named tool,
both APIs, streaming and not, arguments valid under the schema
(`server.TestToolChoiceForcesACall`). Llama-3.2-1B calls `get_weather`
whatever it is asked, so its discriminating arm is the named `get_time`.
With the constraint disabled, Qwen3 answers in text under "required" and
Llama calls the wrong tool under the named choice.

Not built: arguments streamed in pieces (a call goes out whole when its
markup closes), MiniMax-M3's argument tags, Tencent Hy3's templated tokens,
functionary, and tools on the ConnectRPC surface (a proto change).

## The format tier -- four silent dequantization traps, hermetic goldens, and the GGUF trust boundary

Moved from ARCHITECTURE.md §7 (tiers T0 and T6) when that file was retired, and
checked against `format/quant/dequant.go`, `convert/gguf/validate.go` and `convert/gguf/gguf_test.go`
at the same time. The code comments carry each line; this entry is the record of why the tier exists and
how it is gated. The design document's gemma quirk table from the same section is NOT carried: its
rotary row was the error 7a above corrects, and every other row is stated, correctly, beside the gemma
case in `convert/config.go` or in 7c.

**The failure mode of this project is fluent, grammatical, wrong text**, and the lowest place it can
start is the dequantizer. Four traps in ggml's block formats fail that way, with no error:

- **Q3_K's high-mask bit is INVERTED**: a SET bit means do NOT subtract 4. The commonest
  mis-implementation in ggml ports; `cpu-kernels.md` records the arm64 kernel folding the polarity
  into one `BIC` so it lives in an instruction.
- **Q4_0 and Q4_K nibbles are not interleaved**: the low nibbles are elements 0..15 of a Q4_0 block
  (0..31 of a Q4_K 64-element group) and the high nibbles the next run, not 2j/2j+1.
- **Q6_K's scale index strides by 2**: `sc[is+0], sc[is+2], sc[is+4], sc[is+6]` with `is = l/16`.
- **Q5_K's `qh` does not advance** across the four 64-element groups: the same 32 bytes serve all
  four, two bits each, selected by a shifting mask pair.

**The gate is bit equality against libggml itself, taken once and committed.** `scripts/gold.py`
calls `dequantize_row_*` in `libggml-base.so.0.18.0` through Python ctypes on windows of real
tensors and writes `testdata/golden/dequant/*.{in,f32}`. `convert/gguf` `TestDequantGoldens`
compares `math.Float32bits` over nine windows (Q4_0 and Q8_0 from gemma-2b and stories15M, Q8_0 from
MiniLM, Q3_K/Q4_K/Q5_K/Q6_K from tinyllama) with no tolerance, fails rather than skips when a fixture
is missing, and refuses a degenerate window -- an all-zero or constant window would match a broken
dequantizer. No `.so` is needed at test time, so a llama.cpp upgrade cannot move what "correct" means
underneath a green build.

**A GGUF is untrusted input from the internet**, and each of these is an out-of-bounds read if
unchecked (`convert/gguf/validate.go`): `version` in {2, 3}; `general.alignment` a nonzero power of
two (and at most 1 MiB); at most four dimensions; `ne[0]` a multiple of the type's block size; the
type a known enum value; header counts that the remaining bytes can hold; and every tensor's
`[off, off+nbytes)` inside the data section, with offsets strictly increasing and only alignment
padding between neighbours. The last property held EXACTLY on all three reference files the design
was written against (residual 0), so it doubles as a whole-file integrity check: a truncated or spliced
download fails at open instead of producing fluent nonsense later. `FuzzParse` covers `Open` under
RULE 9's exec budget. Since the container became the engine's only weight format this boundary guards
the CONVERTER; `format/jlm` has its own readers.

## The mixture families: glm4moe

Built on the user's decision, as compositions of what the engine already ran.
Each router was read off transformers' forward, vLLM's and SGLang's model
files and llama.cpp's builder before anything was written, and where they part
the choice is named.

### glm4moe (GLM-4.5, GLM-4.6, GLM-4.5-Air)

| piece | where it lives |
|---|---|
| biased q/k/v, unbiased o | tensors |
| per-head q/k RMSNorm (355B only) | `FlagQKNorm`, set when the file has the tensors |
| NEOX rotary on half the head | `FlagRopeNeox`, `NRot` from rope.dimension_count |
| post_attention_norm = the FFN's input norm | `retarget` |
| sigmoid, selection bias, renormalise, routed scale | DeepSeek-V3's router, unchanged |
| one ungated shared expert, one dense lead | keys, read for every mixture |
| one prediction block | `NMTP`, set aside by `nextnOf` |

The router, four implementations:

    transformers  Glm4MoeTopkRouter   router_logits.sigmoid(), hardcoded; reads no gating key
    vLLM          glm4_moe.py         scoring_func="sigmoid", hardcoded
    SGLang        glm4_moe.py         grouped top-k with the correction bias, sigmoid
    llama.cpp     glm4-moe.cpp        expert_gating_func, absent -> SIGMOID for this arch

All four agree, so the converter sets sigmoid and REFUSES a file that states
softmax. Two defaults part and neither is reachable on a published file:
`norm_topk_prob` defaults to true in the class and an absent
expert_weights_norm to false in llama.cpp's loader (every GLM-4.5 config states
true and the converter writes it; jitllm renormalises, the class's default),
and transformers/vLLM/SGLang group the top-k by `n_group` where llama.cpp does
not -- every published GLM-4.5 has `n_group` 1, which is the ungrouped top-k,
and the converter reads the group keys llama.cpp writes.

`synth-glm4moe` (scripts/moegold.py): Glm4MoeForCausalLM, 3 blocks (one dense
lead), 8 experts top-2, one shared, routed scale 2.5, q/k norm on, partial
rotary 8 of 16, F32, with the selection bias drawn at 0.5 -- transformers
initialises that buffer to zero, which is the identity -- and one prediction
block appended to the safetensors, which transformers never runs, so the
converter must set it aside. The GGUF is llama.cpp's own converter's.

    host vs transformers, decode / prefill at every length / batch   5.511e-12
    device vs host, every block and the head
      cuda:0   decode 4.010e-12, prefill 1.004e-06 (binary16 attention tiles)
      vulkan   decode 4.623e-12, prefill 3.355e-12
    step across three sessions vs each alone   cuda 4.101e-13, vulkan 5.661e-13
    batched mixture prefill vs row by row      4.715e-07, 2 grouped blocks
    TestDecodeDoesNotAllocate                  0 on the host, cuda and vulkan

Every violation lands at 3.5e-01 or above against the 1e-9 bound: softmax for
sigmoid 7.1e-01, no selection bias 1.28 (device 1.07), not renormalised 1.44
(1.31), no routed scale 3.9e-01 (3.6e-01), no shared expert 4.7e-01 (4.2e-01),
no q/k norm 1.85 (1.75), no q/k/v biases 1.35, NORM rotary 1.64 (1.63), and the
prediction block run as a fourth trunk block 3.9e-01.

★ **THE DEVICE CANNOT RUN THE SOFTMAX VIOLATION, AND THAT IS ITS OWN
FINDING.** With the selection bias present the device's router declines a
softmax gate by name -- *"a softmax router with a selection bias needs the full
softmax formed before the scores can be added or compared"* -- so the softmax
arm is host-only for glm4moe. ERNIE 4.5 is that combination for real.

★ **AND THE HOST'S ALLOCATION GATE WAS RED ON EVERY ALL-FLOAT MODEL, FOUND BY
ADDING THIS ONE.** `nn.JIT.TokenStart` built the prefetch-distance duel every
token while `newFPFTuner` returned nil for a model with no fused packed kernel,
fingerprinting an empty set and allocating 161 objects over 64 tokens. It
remembers the miss now (`fpfNone`), and the host arm reads 0.

### qwen2moe (Qwen1.5-MoE-A2.7B, Qwen2-57B-A14B)

qwen2's attention -- biased q/k/v, an unbiased o, NEOX over the whole head --
and a softmax top-k beside a shared expert scaled by its own sigmoid gate,
which is qwen3next's (`RoleShRouter`) and needed nothing new. The router:

    transformers  Qwen2MoeTopKRouter  softmax, top-k, renormalise iff norm_topk_prob (default false)
    vLLM          qwen2_moe.py        renormalize=config.norm_topk_prob; F.sigmoid(shared_expert_gate)
    llama.cpp     qwen2moe.cpp        softmax, weights norm FALSE as a literal; no key written

So an absent expert_weights_norm does NOT renormalise for this arch (as for
deepseek2); a key, if a converter ever writes one, is obeyed. transformers can
make blocks dense (`decoder_sparse_step`, `mlp_only_layers`); no GGUF carries
either, llama.cpp requires a router in every block, and the converter refuses
a block without one by name.

`synth-qwen2moe`: 3 blocks, 8 experts top-2 not renormalised, a shared expert
twice a routed one's width, GQA 4:2, F32.

    host vs transformers, decode / prefill / batch   1.977e-12
    cuda:0   decode 2.120e-12, prefill 5.805e-07
    vulkan   decode 1.763e-12, prefill 1.039e-12
    step across sessions   cuda 8.600e-13, vulkan 5.994e-13
    batched mixture prefill vs row by row   2.303e-07, 3 grouped blocks
    TestDecodeDoesNotAllocate   0 on the host, cuda and vulkan

Violations, host (device): sigmoid for softmax 7.2e-01 (7.5e-01), renormalised
1.5e-01 (1.7e-01), no shared-expert gate 1.04 (8.8e-01), no shared expert 1.28
(1.24), no q/k/v biases 9.7e-01, NORM rotary 1.65 (1.38).

### ernie4_5-moe and ernie4_5 (ERNIE 4.5)

The dense ERNIE 4.5 (0.3B) is llama's graph exactly -- llama.cpp's
ernie4-5.cpp is llama's builder with interleaved rotary and an optional bias --
so "ernie4_5" converts as `ArchLlama`, as "llama3" does. The mixtures are
`ArchErnie45MoE`, because the interleave step is the config's optional tail,
which an older reader skips. The router:

    transformers  Ernie4_5_MoeTopKRouter  softmax; top-k of softmax + e_score_correction_bias;
                                          weights gathered UNBIASED, / clamp(sum, moe_norm_min)
    llama.cpp     ernie4-5-moe.cpp        SOFTMAX and norm TRUE as literals, ffn_exp_probs_b
    vLLM          ernie45_moe.py          FusedMoE default softmax, renormalize=True, e_score_correction_bias

The bias joins the softmax PROBABILITY, not the logit -- a softmax is not
additive, so the two are different selections. The host's router already
computed that; the device's declined it by name ("a softmax router with a
selection bias needs the full softmax formed before the scores can be added or
compared"). `kernels.ExpertRank` now forms each row's max and sum in two flat
loops before the rank (the IR has no nesting) and ranks on exp(l-m)/Z + b; the
slot carries the raw logit, so `ExpertWeights` is unchanged. Every thread forms
the same max and sum in the same order, so the keys are the same numbers in
every thread and the ranks stay a permutation. A grouped softmax router is
still declined.

Two RULE 7m choices. The divisor's floor is moe_norm_min (1e-12) in the class
and DeepSeek's 1e-20 here; a softmax's top-k sums to at least k/n, so they
cannot differ. And the shared expert's width is read off ffn_gate_shexp:
llama.cpp's converter writes `intermediate_size // num_key_value_heads` where
the class builds `moe_intermediate_size * moe_num_shared_experts` -- 3072 both
ways on ERNIE-4.5-21B-A3B, and nothing ties them.

★ **llama.cpp CANNOT LOAD AN ERNIE WHOSE INTERLEAVE STEP IS ABOVE ONE.** Its
graph applies the step and its loader does not: `synth-ernie45moe` (step 2)
fails at "missing tensor 'blk.2.ffn_gate_inp.weight'". Every published ERNIE has
step 1, so no real file meets it; the fixture's oracle is transformers.

`synth-ernie45moe`: 4 blocks -- block 0 the dense lead, block 2 dense by the
step, blocks 1 and 3 mixtures -- 8 experts top-2, two shared, the bias at 0.05
(a softmax over 8 puts each near 0.125), tied head, F32. `synth-ernie45`: the
dense twin.

    host vs transformers, decode / prefill / batch   moe 8.455e-12, dense 5.621e-12
    cuda:0   moe decode 9.102e-12, prefill 3.808e-07; dense 1.783e-12 / 1.465e-07
    vulkan   moe decode 1.213e-11, prefill 1.662e-12; dense 2.407e-12 / 5.982e-13
    step across sessions, batched mixture prefill, decode allocations (0): pass

Violations, host (device): sigmoid for softmax 1.18e-01 (1.21e-01), no
selection bias 5.3e-02 (5.5e-02), not renormalised 3.0e-02 (2.8e-02), no shared
expert 1.16 (1.09), NEOX rotary 2.31 (2.35); the dense twin's NEOX 1.92 (1.82).
With the selection bias kept, glm4moe's softmax-for-sigmoid violation now runs
on the device too: 6.49e-01.

ERNIE-4.5-0.3B-PT Q4_K_M (unsloth's GGUF): **20 of 20 teacher-forced against
llama.cpp, and 20 free-running tokens identical.** ERNIE-4.5-21B-A3B was not
taken: its Q4_K_M is ~13 GB against 26 GB free on the model disk.

### The families on the M4, the principles' gates, and the real checkpoints

Every family gate above ran on the M4 (arm64 host, Metal with every block
and the head placed), each container converted there from the GGUF:

    host vs transformers   glm4moe 3.801e-12, qwen2moe 3.893e-12,
                           ernie4_5-moe 8.869e-12, ernie4_5 8.284e-12
    metal vs host          decode 3.992e-12 / 3.913e-12 / 1.150e-11 / 4.982e-12
                           prefill 1.27e-06 / 1.54e-06 / 7.68e-07 / 3.13e-07
    step across sessions   5.984e-13 / 6.110e-13 / 1.070e-12 / 3.875e-13
    every violation moves the logits as on CUDA and Vulkan, softmax+bias included

and the principles' gates on both hosts: under a one-page budget every block
and expert page is evicted and re-read (149, 207, 181 and 57 page-ins against
14, 26, 20 and 3 resident) and decode, prefill and the batch are BIT-identical
to the resident model; `TestEveryModelRunsGenerated` runs all four; decode and
every step shape allocate nothing on the host, CUDA, Vulkan and Metal.

★ **THE SHARED-EXPERT GATE WAS NEVER FREED ON THE DEVICE.** `l.shRouter`
(qwen2moe's and qwen3next's sigmoid gate vector) was uploaded at placement and
missing from `layer.auxBufs`, the one list a release and a page-out both free,
so every move of a gated block home stranded the gate's allocation. Found by
running `batchSeamMoves` on the families: synth-qwen2moe's round trip left
+261,376 B charged, and +65,536 B with the gate on the list.

★ **AND THE 64 KiB EVERY MIXTURE'S SEAM ROUND TRIP LEFT IS STAGING, NOT A
LEAK.** The same arm on Qwen3-MoE-4x0.6B and all three families read exactly
+65,536 B after shrink-then-grow (Llama-3.2-1B 0), with every logit right. With
the blocks home the host runs them, and a batch's router is a matvec per row
that the device serves on its own (`GPU.MatVec`, 63 of them over the arm):
`sizePart` grows the lone matvec's partial buffer, charged as scratch, by
4,096 B on Qwen3-MoE and 256-512 B on the fixtures, and the CUDA driver hands
each out as one 64 KiB page. Measured on cuda:0, Qwen3-MoE: scratch +4,096,
allocated +4,096, rounding +61,440, nothing else. A dense batch never offers
a block matvec alone, so it reads 0. Staging only grows to the widest shape
asked, so the gate (`batchSeamMoves`) now runs the round trip twice: the first
held exactly outside the scratch (the buffers' bytes and the charges), the
second exactly everywhere. `TestBatchSeamMovesCarryEveryRowMoE` and
`...MoEFamilies` are in the tree: clean 2.0e-02 / 3.1e-02 (Qwen3-MoE) and
6e-07..6e-06 (the fixtures), 6.9-11.5 and 0.27-2.7 against a batch that moves
row 0's history alone, and the shared-expert gate's stranding restored reads
+768 B of buffers after the first trip and +195,840 B after the second.

Real checkpoints, TestGreedyMatchesLlamaCpp on the V100 box (one GPU, sixteen
cores, the GGUFs downloaded there):

    Qwen1.5-MoE-A2.7B-Chat Q4_K_M   19 of 20 teacher-forced; free-running parts at
                                    token 5 on a 0.304 tie
    ERNIE-4.5-21B-A3B-PT Q4_K_M     18 of 20 teacher-forced; parts at token 2 on a
                                    0.198 tie
    GLM-4.5-Air Q4_K_M (106B-A12B,  20 of 20 teacher-forced, and 20 free-running
      73 GB in two GGUF parts)      tokens identical; 96 GB container, converted
                                    from the split GGUF on that box

### As it stood in AGENTS.md

★ **THE MIXTURE FAMILIES ARE COMPOSITIONS OF PIECES THE ENGINE ALREADY HAD,
AND THE WORK IS READING EACH ROUTER OFF ITS REFERENCE.** **glm4moe** (GLM-4.5,
4.6, 4.5-Air) is DeepSeek-V3's router -- sigmoid, the selection-only bias,
renormalised, a routed scale -- on GQA attention with biased q/k/v, an
optional per-head q/k norm and NEOX rotary on half the head, after a dense
lead, with a prediction block set aside (`NMTP`). Its `post_attention_norm` is
the FFN norm (`retarget`). The gate is sigmoid WHATEVER the file says:
transformers' `Glm4MoeTopkRouter`, vLLM and SGLang all hardcode it, and a file
stating softmax is refused (`convert/moefamily.go`). Gated on `synth-glm4moe`
(`scripts/moegold.py`: transformers' class, the selection bias RANDOMISED,
llama.cpp's converter) in `model.TestMoEFamiliesMatchTransformers`,
`TestMoEFamilyFeaturesAreLoadBearing` and `TestMoEFamiliesOnEveryDevice`;
GLM-4.5-Air Q4_K_M is 20 of 20 teacher-forced against llama.cpp (V100 box).
**qwen2moe** (Qwen1.5-MoE, Qwen2-57B) is qwen2's attention with a softmax
top-k that is NOT renormalised when the file says nothing (llama.cpp's literal
false, the class's default, every published config) beside qwen3next's
sigmoid-gated shared expert; a block with no router (transformers'
`mlp_only_layers`) is refused by name. `synth-qwen2moe`; Qwen1.5-MoE-A2.7B
Q4_K_M is 19 of 20 teacher-forced. **ernie4_5-moe** is
llama's attention (interleaved rotary) with a SOFTMAX router whose selection
adds a per-expert bias to the softmax PROBABILITY -- the combination the device
router declined by name until `kernels.ExpertRank` formed the softmax first --
renormalised, beside shared experts, after a dense lead and interleaved by
`MoEStep`; the shared width is read off the tensor, because llama.cpp's
converter writes `intermediate_size // num_key_value_heads` for it.
ERNIE-4.5-0.3B Q4_K_M is 20 of 20 and ERNIE-4.5-21B-A3B Q4_K_M 18 of 20
teacher-forced against llama.cpp.
All three families run every gate on the M4 too -- the arm64 host and Metal
with every block placed -- and the principles' gates: paging under a one-page
budget bit-identical (`TestMoEFamiliesPageWithTheSameAnswer`), generated code
(`TestEveryModelRunsGenerated`), zero decode and step allocations.
`docs/engineering-history/model-correctness.md` has the numbers.

## The dense llama family: smollm3, arcee, seed_oss, olmo2, exaone4, mistral3

Six architectures added together on the user's decision, each llama's block
with a few switches. Four needed no new graph feature at all; two needed one
each, and one of those (the post-norm block) is shared by two architectures.
Each is its own arch code (`jlm.ArchSmolLM3` .. `jlm.ArchMistral3`, taken from
the top of the code space down) so a reader that predates it refuses the
container rather than running llama's block with the feature missing -- every
one of those omissions is fluent. No container version moved.

| arch | what it is over llama | new in the engine |
|---|---|---|
| smollm3 | no rotary on every 4th layer, (il+1)%4==0 | `FlagNoPEGlobal` with `SWAWindow` 0: the period alone names the NoPE layers (`Config.periodLocal`) |
| arcee | ungated FFN, up -> relu^2 -> down, RMSNorm | nothing: nemotron's `FlagReLU2` and the absent gate |
| seed_oss | biased q/k/v, NEOX, head wider than hidden/heads | nothing; its `post_attention_norm` IS the FFN pre-norm (`retarget`) |
| olmo2 | NO pre-norms; q/k RMSNorm over the whole projection; post-norms | the post-norm-only block on host and device |
| exaone4 | olmo2's block with per-head q/k norm; 32B slides 3 layers in 4 with NoPE on the 4th | the same block |
| mistral3 | temperature on every layer (llama4's form, offset 0, floor = YaRN orig ctx); YaRN | the device runs the temperature on ROTATED layers |

### Read off the builders, checked against transformers and mlx-lm

- **The NoPE step is a constant everywhere.** llama.cpp's smollm3.cpp sets
  `n_no_rope_layer_step = 4`; transformers' `no_rope_layer_interval` defaults
  to 4 and SmolLM3-3B's own `no_rope_layers` list is exactly that rule; no GGUF
  key carries it. The converter writes the period.
- **OLMo 2's k norm is n_kv*head_dim wide**, not n_embd (llama.cpp, mlx-lm and
  transformers agree), so a GQA OLMo 2 (the 32B) norms q over n_head*head_dim
  and k over n_kv*head_dim: `Config.QKNormShape` gives each its own width, the
  loader refuses a norm of the wrong width, and the device takes HeadNormRows
  at each width (the shared qkPart/qkRms pair is built only where q and k are
  the same width). synth-olmo2-gqa (Olmo2ForCausalLM, GQA 4:2): host
  2.001e-12, M4 host 1.348e-12, cuda / vulkan / metal decode 2.336e-12 /
  2.441e-12 / 1.647e-12, llama.cpp 20 of 20; a per-head q/k norm reads
  5.37e-01 and k's norm read from q's weight 3.33e-01 (host), 4.76e-01 and
  3.44e-01 on every device.
- **OLMo 3's sliding layers carry no rotary scaling at all.** llama.cpp's
  olmo2 builder ropes an is_swa layer with freq_scale 1, ext_factor 0 and
  attn_factor 1; transformers builds the sliding layers a default rotary and
  the full ones YaRN. This engine's YaRN frequencies already reached the
  global table only, and the folded magnitude (AttnFactor) reached both, which
  is why a scaled OLMo 3 was refused. It is `jlm.ArchOLMo3` now -- a code of
  its own, so a reader that would scale the local layers refuses it -- with a
  second rotary at rope.freq_base_swa whose magnitude is one
  (`Config.SWARopePlain`). synth-olmo3 (Olmo3ForCausalLM: window 4 on three
  layers in four, GQA 4:2, YaRN factor 8 over an original context of 4): host
  3.259e-12, M4 host 2.971e-12, cuda / vulkan / metal decode 1.743e-12 /
  2.021e-12 / 1.344e-12, llama.cpp 20 of 20. Violations (host / device): YaRN
  on the sliding layers 2.26e-01 / 1.97e-01; its magnitude alone there
  1.64e-01 / 1.61e-01; no magnitude on the global layer 7.22e-03 (host); no
  window 1.69 / 1.83; k's norm from q's 4.97e-01 (host).
- **mistral3's YaRN magnitude is not DeepSeek's.** Its converter writes
  `mscale_all_dim` itself into `yarn_log_multiplier` (DeepSeek's writes
  0.1*mscale_all_dim), and llama.cpp's context setup scales cos and sin by
  mscale(f,1)/mscale(f,m) -- transformers' attention_factor for a config that
  carries both mscale and mscale_all_dim -- with nothing reaching the softmax.
  Read through the generic path it would have scaled cos and sin by
  1+ln(16) = 3.77 on Ministral 3. `mistral3Yarn` reads it; Ministral 3's own
  1/1 gives exactly one.
- **EXAONE 4's sliding window is keyed on the layer count in llama.cpp.**
  exaone4.cpp turns sliding on only for a 64-layer model (the 32B); transformers
  slides wherever `sliding_window` is set. They agree on every published
  checkpoint (the 1.2B carries `sliding_window: null`), and part on the
  4-layer fixture, which slides: the converter follows the file's key and
  defaults the window to 4096 for 64 layers as llama.cpp does. RULE 7m: the
  fixture is held to transformers, and llama.cpp's greedy gate skips it by name.

### The post-norm block

OLMo 2 and EXAONE 4 have no `attn_norm` and no `ffn_norm`: each branch reads
the residual as it is, and only its OUTPUT is normalised (gemma2's two
post-norms, which the engine already ran). The loader admits a missing pre-norm
only where both post-norms stand in; `State.norm` with no weight is a copy (the
parallel block's shared-norm copy is the precedent); the device copies the row
into h with a generated `Restride` kernel (`bs.copyRows`, built when the plan
says `NoPreNorm`) and keeps every fused norm path behind `l.nAttn != nil`. The
tier refuses a block with no pre-norm whose plan does not say so.

### mistral3's temperature on the device

The tier declined it by name ("attention temperature on rotated layers"). A
row's temperature and its rotation are both linear in q, so the rotated
layers scale q into a scratch `qt` with the existing `ScaleRowsByCount` kernel
before the rotary, and NoPE layers keep llama4's path.

### The numbers

Fixtures are transformers' own classes, written to GGUF by llama.cpp's
convert_hf_to_gguf.py (`scripts/c6gold.py`), all F32, bound 1e-9:

| fixture | host vs transformers | M4 host | CUDA decode / prefill | Vulkan decode / prefill | Metal (M4) decode / prefill |
|---|---|---|---|---|---|
| synth-smollm3 | 3.241e-12 | 5.383e-12 | 2.311e-12 / 7.291e-07 | 2.537e-12 / 1.617e-12 | 1.922e-12 / 1.324e-06 |
| synth-arcee | 1.207e-12 | 2.580e-12 | 6.854e-13 / 2.315e-07 | 7.612e-13 / 9.801e-13 | 5.720e-13 / 2.519e-07 |
| synth-seedoss | 2.919e-12 | 2.334e-12 | 7.729e-12 / 3.404e-07 | 6.327e-12 / 1.642e-12 | 3.080e-12 / 3.949e-07 |
| synth-olmo2 | 2.340e-12 | 2.478e-12 | 2.922e-12 / 1.324e-07 | 2.487e-12 / 1.060e-12 | 1.964e-12 / 9.070e-07 |
| synth-exaone4 | 2.618e-12 | 3.022e-12 | 3.075e-12 / 5.559e-08 | 2.369e-12 / 7.666e-13 | 3.589e-12 / 2.440e-07 |
| synth-mistral3 | 3.241e-12 | 2.949e-12 | 2.199e-12 / 7.940e-07 | 1.983e-12 / 2.199e-12 | 1.914e-12 / 6.950e-07 |

Host is decode, prefill at every length and a two-row batch; a device prefill
attends through binary16 tiles where the device has them, hence its 1e-7
class (the C6 harness bounds it at 1e-5).

Every device arm places every block and the head. The ragged step across three
sessions reads 4e-14 to 2.1e-12 on CUDA, Vulkan and Metal for all six.

Violations, each moving the logits far past the bound (host, then device;
the device arm removes the feature from the device session alone):

    smollm3   rotary on the NoPE layer          host 8.808e-02  device 9.156e-02
              NoPE every second layer           host 6.922e-01
    arcee     SiLU for relu2                    host 1.083e+00  device 9.744e-01
              GELU for relu2                    host 9.469e-01
    seed_oss  no q/k/v biases                   host 9.125e-01  device 1.010e+00
              NORM rotary for NEOX              host 1.565e+00  device 1.456e+00
              the attention's norm before FFN   host 1.129e+00
    olmo2     a per-head q/k norm               host 4.979e-01  device 5.310e-01
              no post-attention norm            host 1.560e+00
              no post-FFN norm                  host 2.622e+00
              the post-norms as pre-norms       host 1.892e+00  device 1.777e+00
              NORM rotary for NEOX              host 8.266e-01  device 9.447e-01
    exaone4   rotary on the global layer        host 2.194e-02  device 2.595e-02
              no sliding window                 host 2.128e+00  device 1.883e+00
              no q/k norm                       host 1.112e+00
              no post-FFN norm                  host 2.683e+00
              the post-norms as pre-norms       host 1.660e+00  device 1.510e+00
    mistral3  no attention temperature          host 1.387e-01  device 1.649e-01
              no YaRN magnitude on cos and sin  host 6.480e-02

and the device temperature was a decline before: the device arm took 0 of 3
blocks.

Against llama.cpp, teacher-forced through `TestGreedyMatchesLlamaCpp`: every
fixture but synth-exaone4 (above) 20 of 20; OLMo-2-0425-1B-Instruct Q4_K_M 20 of
20 and 20 greedy tokens on the Linux host and on the M4; EXAONE-4.0-1.2B Q4_K_M
8 of 8 (llama.cpp stops at EOS) and 7 greedy tokens, on both; SmolLM3-3B Q4_K_M
20 of 20 and 20 greedy tokens, on both; Ministral-3-3B-Instruct-2512 Q4_K_M 20 of
20 and 20 greedy tokens; AFM-4.5B Q4_K_M 20 of 20 and 20 greedy tokens. Every real
model is the same on the M4's arm64 host. (Ministral's temperature floor is its YaRN original
context, 16384, so no short prompt reaches a factor other than one there: the
fixture's floor of 4 is what gates the temperature.)

The real models on the RTX 3050 Ti (`jitllm verify`, JITLLM_GPU_TUNE=0,
JITLLM_KV_F16=0), every block and the head placed unless noted:

    Llama-3.2-1B (control)  0/32 disagree  max|dlogit| 0.822   4.04x the host
    OLMo-2-0425-1B          4/32           1.86                2.76x
    EXAONE-4.0-1.2B         2/32           4.19                4.78x
      with 7 of 30 blocks   2/32           4.25
    SmolLM3-3B              1/32           2.15                2.46x  (head on
                                                               the host: card full)
    Ministral-3-3B          0/32           0.308               3.15x  (head on
                                                               the host: card full)
    AFM-4.5B                0/32           0.937               1.63x  (a partial
                                                               seam: 19 blocks over
                                                               the 4 GB card)

Seed-OSS has no checkpoint under 36B (21.8 GB at Q4_K_M), which neither this
box's disk nor its card holds; it is gated on its fixture alone.

**The post-norm models sit above the 0.2-0.94 band, and it is amplification,
not a device defect: SETTLED against llama.cpp.** The same fixtures converted
to Q8_0 read max|dlogit| **2.86e-06** with 0/32 disagreeing, so the wiring is
right; what the real checkpoints add is a model that amplifies any
reduction-order difference. `model.TestPostNormDeviceErrorIsAmplification`
measures it three ways, teacher-forced over one 128-token chunk, the
log-softmax over llama.cpp's own top-16-nat window (llama-perplexity's
`--kl-divergence-base`), every device arm with an f32 KV cache against an
f32-KV host:

                                          OLMo-2-0425-1B   EXAONE-4.0-1.2B
    llama.cpp CUDA vs llama.cpp CPU       1.157 (2/63)     1.927 (6/63)
    jitllm host, f32 KV vs f16 KV         0.849 (3/63)     3.352 (7/63)
    jitllm host vs llama.cpp CPU          0.977 (4/63)     2.923 (3/63)
    jitllm cuda vs llama.cpp CPU          0.912 (3/63)     1.746 (6/63)
    jitllm cuda vs host                   0.869 (2/63)     1.367 (4/63)
    jitllm vulkan vs host                 1.106 (2/63)     1.989 (3/63)
    ONE block on cuda, alone:
      block 0                             0.824            1.333
      the middle block                    0.616            0.581
      the last block                      0.072            0.075
    ONE block on vulkan: 0 / middle / last  1.054 / 0.551 / 0.072   1.303 / 0.545 / 0.096

(max|dlogp|, argmax flips of 63 in brackets.) **The error tracks the depth
below the seam and vanishes at the last block** -- RULE 11c's falsifiable form
-- so a device block is 0.07-0.10 from the host, inside the band, and the rest
is what the blocks after it make of that. **llama.cpp's own two backends
disagree by as much** (1.16 and 1.93, with 2 and 6 flips), and the host
disagrees with ITSELF by as much when only the V cache's width changes (0.85,
3.35): a perturbation with no device in it. EXAONE's device is closer to
llama.cpp's CPU (1.75) than jitllm's own host is (2.92). The `verify` rows
above mix in the KV width (the host's default is f16, the device's f32) and a
greedy chain that compounds a flip, which is why they read higher.

The gate holds the last block alone under 0.25, block 0 at least 4x the last,
and the whole device within 1.5x of what two independent engines differ by on
the same file: the larger of llama.cpp's own CUDA-vs-CPU and jitllm's host
against llama.cpp's CPU. A 5% bend of the last block's post-FFN norm on the
host arm alone, a defect where nothing amplifies it, reads 0.646 (OLMo 2) and
0.398 (EXAONE) and fails it.

The whole-device bound was llama.cpp's cross-backend figure alone, and on the
V100 box it failed OLMo 2's Vulkan arm deterministically: 1.106 against
1.5 x 0.693. That figure is a property of llama.cpp's BUILD, not of the model:
the laptop's sm_86 CUDA build reads 1.157 against its CPU build, the box's
sm_70 build 0.693, while jitllm's arms read the same on both boxes (Vulkan
1.106 on the RTX 3050 Ti and on a V100, CUDA 0.869 and 0.785, the host
against llama.cpp's CPU 0.977 on both). By RULE 11c's measure the Vulkan
arm is not an outlier: it sits 0.952 from llama.cpp's CPU, closer than
jitllm's own host (0.977), its error tracks the depth below the seam (block 0
alone 1.054, the middle 0.551, the last 0.072), and the bent last block still
reads 0.646. So the band is what independent arithmetic differs by on this
model, and jitllm's host against llama.cpp is one such pair. With the larger
of the two, OLMo 2 is held to 1.466 on the box (1.736 on the laptop, where
llama.cpp's CUDA build is the larger) and EXAONE to 4.38 on both; the
per-block bounds, which are what see a defect in one block, are unchanged. The
whole-device bound is the coarse one: a 5% bend of EVERY block's post-FFN
norm on the host arm reads 1.796 (cuda:0), 1.791 (vulkan:0) and 1.702 (the
Iris Xe) against OLMo 2's device on the laptop, failing the first two and
passing the third by 0.03, and all three would fail the box's 1.466.

### From HuggingFace safetensors

`hfArchTable` takes the seven classes -- `SmolLM3ForCausalLM`,
`ArceeForCausalLM`, `SeedOssForCausalLM`, `Olmo2ForCausalLM`,
`Olmo3ForCausalLM` (ArchOLMo3 where it has a window, as the GGUF path converts it),
`Exaone4ForCausalLM` and `Ministral3ForCausalLM` -- in
`convert/safetensors_dense.go`. `Mistral3ForConditionalGeneration`, the
multimodal class real Ministral 3 checkpoints ship, nests the text model under
`model.language_model.` beside a vision tower and is not on the list.

**The trap is the class default, and it differs per class.** A key absent from
config.json takes its config dataclass's default, read off `from_dict` in the
reference venv rather than assumed:

    num_key_value_heads   SmolLM3 4, SeedOss 8, Ministral3 8, Exaone4 32,
                          the rest the head count
    head_dim              SeedOss 128, Ministral3 128, the rest hidden/heads
    rope_theta            SmolLM3 2e6, OLMo 3 5e5, the rest 1e4
    rms_norm_eps          SmolLM3 and SeedOss 1e-6, the rest 1e-5
    sliding_window        OLMo 3 and Exaone4 4096 on 3 layers in 4 (absent;
                          an explicit null slides nothing)
    the rotary            Ministral3 with neither rope_parameters nor
                          rope_scaling takes its OWN YaRN dict (factor 16 over
                          16384, beta 0.1, base 1e6) -- a top-level
                          rope_theta does not reach it

llama's defaults would have read a different number in every row;
`hfClassDefaults` states each class's, and `convert.TestDenseHFClassDefaults`
checks them on config.json text with the keys absent, which no fixture can do
(save_pretrained writes every key). Two more readings the reference forced:
**OLMo 3's sliding layers get base 5e5 whatever a legacy top-level rope_theta
says** (`convert_rope_params_to_dict` pops rope_theta for the full layers and
the sliding ones fall to the default), so the two bases can differ: the
sliding layers' is RopeBaseSWA, as the GGUF path reads rope.freq_base_swa; and **SmolLM3's NoPE period is
read from `no_rope_layers`**, where llama.cpp hardcodes 4 -- the fixture uses 3
over six layers so a hardcoded 4 fails. Ministral3's YaRN magnitude follows
transformers (attention_factor, else mscale(f,mscale)/mscale(f,mscale_all_dim),
else mscale(f,1)), which equals llama.cpp's mscale(f,1)/mscale(f,m) wherever
mscale is 1, as in every published checkpoint.

**The same weights through both converters give the same container.**
`convert.TestDenseSafetensorsMatchesItsGGUF` converts scripts/c6gold.py's
HuggingFace directories (`JITLLM_C6_HF`, its CV_HF) directly and through the
GGUF llama.cpp wrote from them: every `jlm.Config` field and every tensor byte
equal, for all six (38, 27, 39, 36, 47 and 30 tensors). Against violations it
reads `Config.Flags` 0 against 128 (seed_oss without NEOX), 132 against 1048708
(exaone4 without NoPE on the global layer), and q/k bytes differing in every
block (ministral3 unpermuted).

Gated against transformers by `model.TestSafetensorsMatchTransformers`
(scripts/hfgold.py fixtures, 32000-token donor, F32 cache, bound 1e-9), with
`model.TestSafetensorsDenseFamilySelected` asserting each container carries its
feature:

    synth-hf-smollm3     8 positions   1.17e-12   NoPE on layers 2 and 5
    synth-hf-arcee       8             6.61e-13
    synth-hf-seedoss     8             2.07e-13   head 32 against 16
    synth-hf-olmo2       8             9.24e-13
    synth-hf-olmo3      40             3.47e-12   window 4
    synth-hf-exaone4    40             7.89e-12   window 4, NoPE on layer 3
    synth-hf-ministral3 24             1.03e-11   YaRN 16 over 4, beta 0.5

Violations in the converter, first failing position:

    smollm3     llama.cpp's period 4 for the file's 3     1.311e-03
                no NoPE at all                            9.687e-04
    arcee       SiLU for relu2                            2.424e-01
    seed_oss    NORM rotary for NEOX                      4.984e-02
                no q/k/v biases                           9.163e-01
    olmo2       post-norms as pre-norms                   1.310e+00
    olmo3       post-norms as pre-norms                   1.135e+00
                no window                                 2.319e-01
    exaone4     rotary on the global layer                2.347e-03
                no window                                 2.038e-01
    mistral3    no temperature                            2.902e-02
                DeepSeek's default magnitude, mscale(f,1) 9.296e-03
                q/k unpermuted                            1.606e-01

A GQA OLMo 2/3 and an OLMo 3 with YaRN on its full layers convert as the GGUF
path does (synth-olmo2-gqa and synth-olmo3 through both converters: every
Config field and every tensor byte equal). What stays refused by name: an OLMo 3
with rotary scaling on its sliding layers or another kind than YaRN on its full
ones, a SmolLM3 `no_rope_layers` no period describes, an
EXAONE 4 window whose `layer_types` is not periodic or leaves no layer rotating,
a string `sliding_window_pattern` with no `layer_types`, and a Ministral3
rotary with no `llama_4_scaling_beta`.

### Found on the way

- **seed-coder's tokenizer split "):\n" as one piece.** Its punctuation
  alternative has no `[\r\n]*` tail, which `parseSplit` could not read, so the
  table gave it llama3's. `SplitParams.PunctNoNewline`, carried in an optional
  tail of the vocab record. tok/testdata/pre/seed-coder.json is the artefact.
- **A warm f32 decode allocated 162 objects per 64 tokens on the host**: the
  fused-prefetch duel was rebuilt every token on a model with no fused kernel
  (`JIT.fpfNone`).
- **SmolLM3-3B's step on Metal sat at 8.8x its order-only control**, against
  the gate's band of 8 (`denseStepBand`): 6.237e-02 against 7.108e-03, both
  at row 22, deterministic. ATTRIBUTED to that one row's sensitivity, and to
  the control sampling one order change. Per row, the step against each
  session alone and the control (split 1 against 4) are the same size
  everywhere else -- geometric mean over the 24 rows 1.27x, per-row ratios
  0.34-3.33 on 23 rows and 8.73 on row 22 -- and order changes ALONE move row
  22 by 9.2e-03, 7.1e-03 and 3.5e-02 at splits 2, 4 and 8 while every other
  row stays within 2x across them. Nothing Metal-specific is left: the step's
  matvecs are decode's (groupMV, int8 activations; the binary16 GemmTile
  count the step logs is the grain probe's compile, not a launch), and
  synth-smollm3 at Q8_0 steps at 1.066e-13. `orderControl` now takes the
  worst of splits 2, 4 and 8 against 1: SmolLM3-3B's bound is 2.83e-01 and
  its step passes; the violations still read 52x / 12x / 52x the bound on the
  M4 (session, swatab, nocap) and 31-34x / 11-13x on cuda and vulkan.
- **A sliding layer never freed a KV page behind its window**, so it took a
  fresh page every page of positions -- with synth-exaone4's window of 4, a
  page every 32 tokens inside the allocation gate's window, which skipped that
  host arm by name. Fixed on the host and every device, with the prefix cache
  and relocation agreeing: placement.md 15z.

### As it stood in AGENTS.md

★ **THE DENSE LLAMA FAMILY IS SIX SWITCHES OVER llama's BLOCK, AND TWO OF THEM
WERE NEW.** smollm3 (no rotary on every 4th layer: `FlagNoPEGlobal` with
`SWAWindow` 0, so the period alone names the NoPE layers), arcee (nemotron's
`FlagReLU2` on an RMSNorm block), seed_oss (biased q/k/v; its
`post_attention_norm` is the FFN pre-norm, `convert.retarget`), olmo2 and
exaone4 (the POST-NORM block: no pre-norms at all, each branch's output
normalised -- `State.norm` with no weight is a copy, the device's
`bs.copyRows`; olmo2's q/k norm spans the projection, exaone4's is per head and
its 32B slides 3 layers in 4 with NoPE on the 4th), and mistral3 (the attention
temperature on EVERY layer, which the device now runs on rotated layers by
scaling q before the rotary; its YaRN magnitude is `mistral3Yarn`'s, not
DeepSeek's). Arch codes are taken from the top of the code space down
(`jlm.ArchSmolLM3` = 63) and `Arch.Valid` asks the name tables. Gated by
`scripts/c6gold.py` fixtures in the C6 harness on host, CUDA, Vulkan and Metal,
the step, the five principles' gates, and SmolLM3-3B, AFM-4.5B, OLMo-2-1B,
EXAONE-4.0-1.2B and Ministral-3-3B against llama.cpp on Linux and the M4. A
GQA OLMo 2 norms k over its own projection's width (n_kv*head_dim), and OLMo 3
is `jlm.ArchOLMo3` (57): its sliding layers rotate with no scaling at all, YaRN
frequencies and magnitude reaching the global layers only (`SWARopePlain`). `docs/engineering-history/model-correctness.md`
has the numbers and the RULE 7m choice for EXAONE's window.

## The no-Go-compute sweep reported SKIP for the whole box when one file was out of scope

`model.TestEveryModelRunsGenerated` is the no-Go-compute principle's model gate,
and on a full-directory run it printed `--- SKIP` after 17 models. The cause was
one line: it converted every model through `jlmOf` on the PARENT test, and
`jlmOf` turns a scope refusal (`convert.ErrNotImplemented`) into `t.Skipf`. The
first file outside RULE 7's list -- `synth-arcee.gguf`, a fixture another tree
was adding -- skipped the gate, and every model after it in the glob never ran.
Restricting with `JITLLM_AUDIT_MODEL` hid it, because a substring never reached
the refused file. Two more silences sat beside it: a model that converted and
then failed to `Open` was `continue`d past, and the sweep globbed the top level
only, so granite (`granite/`), ERNIE-4.5-0.3B (`moefam/`), Qwen3.5-0.8B
(`qwen35/`), the encoders (`embed/`) and every safetensors fixture -- the only
deepseek2, kimilinear and HF-mixture models under 2 GiB on the box -- were never
in it.

Each model is a subtest now (`convertPair`/`jlmOfErr` return the refusal
instead of skipping), so a scope refusal is a skip NAMED in its own subtest and
every other model still runs; an `Open` failure after a successful conversion
fails; the sweep is the top-level GGUFs, the GGUFs one directory down (later
shards and projectors left out) and the safetensors fixtures with an `hf`
golden; an embedding model runs its embedder; Forward's logits must be finite;
and the parent never skips -- it fails when no model ran. Full directory on the
laptop, same box, before and after:

    before   17 models ran, then SKIP on synth-arcee.gguf
    after    76 models ran (Forward, Prefill, ForwardBatch; Embed for the
             encoders), 14 refused by name (the dense tree's six new
             architectures and gemma-embedding), 21 over 2 GiB listed by name
             (JITLLM_SLOW=1 admits them)

## q/k norm had two switches, one per tier; it has one derivation now

The host decided whether to RMSNorm q and k by whether the block HELD the
weights (`l.qNorm != nil` in `attnPrep`), the device by the plan's flag
(`Cfg.QKNorm`, which builds the kernels). They agreed only because the loader
happened to read the tensors exactly when the flag was set -- and it showed: the
mixture families' "no q/k norm" violation had to nil the TENSOR to reach the
host and clear the FLAG to reach the device (`noQKNorm`'s comment said so). A
file carrying the tensors with the flag clear was read by neither tier and
loaded silently; one with the flag and no tensors failed in the loader with
"missing attn_q_norm" rather than as a bad container.

`jlm.QKNormOn(flag, layer kind)` is the one derivation -- the flag, on every
block with attention -- and everything reads it: `jlm.Write` and `jlm.Open` run
`Config.CheckQKNorm` (a flag with no tensor, a tensor with no flag and a norm on
a linear block are each refused, naming the block), the loader and `attnPrep`
read `Config.QKNormAt`, the device offer carries the weights only when it says
so (`State.qkNorms`), and `prepLayer` declines a block whose plan and weights
disagree. Both violations are one `cfgMut` now.

`format/jlm.TestQKNormFlagAndTensorsAgree` writes the consistent model, has
Write refuse the three disagreements, and flips `FlagQKNorm` on disk in a
written container for Open to refuse; with Write's check removed the three
refusals fail, with Open's the flipped container opens. Every model on the box
reconverted through the check (the 76 of `TestEveryModelRunsGenerated`) and
none was refused. synth-glm4moe's "no q/k norm" violation through the single
switch: 1.747 on cuda:0 and vulkan, as before.

## hunyuan: the q/k norm after the rotary, run by moving a weight

Hunyuan (Hunyuan-A13B; the 0.5B/1.8B/4B/7B dense line) is llama's block with
NEOX rotary and a learned per-head q/k RMSNorm applied AFTER the rotary --
llama.cpp's `hunyuan-moe.cpp`/`hunyuan-vl.cpp` and transformers'
`HunYuanMoEV1Attention`/`HunYuanDenseV1Attention` agree on the order. No tier
had it: every q/k norm in the engine runs between the projection and the
rotary.

The algebra, checked before it was built. RMSNorm is per head: x/rms(x) times a
weight. A rotation (NEOX or not, whole or partial) preserves each head's sum of
squares, so rms(R x) = rms(x) and the UNWEIGHTED part commutes with the rotary;
the weight does not (w * R(x) != R(w * x) unless w_i = w_{i+half}). The score
is

    q'.k' = sum_d (w_q R(q)/rms(q))_d (w_k R(k)/rms(k))_d
          = sum_d (w_q w_k)_d R(q)_d R(k)_d / (rms(q) rms(k))

and both weights are per DIMENSION, one vector shared by every head, so under
GQA every (q head, kv head) pair sees the same product. So: k gets the weightless
norm (anywhere -- it commutes, so before the rotary on the existing path), and q
gets the norm weighted by w_q*w_k after the rotary. The value path is untouched.
`convert.foldPostRopeQKNorm` stores w_q*w_k as the q norm and ones as the k
norm (a precomputation, at conversion); `jlm.FlagQKNormPostRope` moves q's
launch: host `attnPrep` norms q after `RoPE32JIT`, the device launches the same
`HeadNormRows` kernel on the rotated q (`bs.qr`) into `bs.qn` -- which the
pre-rotary launch no longer writes for q, so the kernel reads one buffer and
writes another (RULE 13) -- and points the scores at it. No new kernel on any
tier. The KV cache holds the unweighted normalised rotated key on every tier,
which is what makes a seam move carry a history across tiers.

The mixture: softmax top-k, renormalised, no scale (llama.cpp passes softmax and
`true` as literals and never loads `expert_weights_scale` for the arch; the
class hardcodes `F.softmax` and the division), beside one ungated shared MLP in
every block; a file stating otherwise is refused. `jlm.ArchHunyuan` (31) for
both names, so an older reader refuses rather than norming before the rotary.

The fixtures (`scripts/moegold.py`, F32): `synth-hunyuanmoe` is A13B's shape --
head_dim 128 with the NTK-alpha base (rope_theta 10000, alpha 1000, base
11158840), which llama.cpp's converter asserts -- GQA 2:1, top-2 of 8, tied;
`synth-hunyuan` is the dense line at head_dim 16. Both built by transformers' own
classes with every weight randomised. llama.cpp's `HunYuanMoEModel.set_vocab`
rebuilds the vocabulary from Tencent's remote-code tiktoken tokenizer, which the
repos no longer ship (they ship tokenizer.json), so the fixture runs the
converter with that one method swapped for the dense class's tokenizer.json
`set_vocab` -- everything else, tensors and keys, is llama.cpp's converter.

    host vs transformers, decode/prefill at every length/batch
                         moe 2.643e-11    dense 5.733e-12
    cuda:0 vs host       moe decode 6.363e-12 prefill 1.301e-06
                         dense decode 2.871e-12 prefill 2.030e-07
    vulkan vs host       moe decode 6.997e-12 prefill 8.038e-12
                         dense decode 3.276e-12 prefill 1.027e-12
    one-page budget      214 and 57 page-ins, every logit identical

Violations, host (moe / dense): the norm before the rotary 4.26e-01 / 1.83e-01,
an unweighted norm 1.14 / 4.73e-01, no q/k norm 1.96 / 8.75e-01, sigmoid for
softmax 8.62e-01, not renormalised 1.10e-01, no shared expert 1.49, NORM rotary
8.91e-01 / 1.27. On cuda:0 and vulkan alike: 4.74e-01 / 1.92e-01, 1.20 /
5.00e-01, 1.98 / 9.22e-01, 7.89e-01, 1.23e-01, 1.53, 1.01 / 1.36.

Hunyuan-0.5B-Instruct Q4_K_M (bartowski): 20 of 20 teacher-forced against
llama.cpp; `jitllm verify` on cuda:0 and vulkan, 1/32 positions disagree,
max|dlogit| 1.07 and 1.01. A batch seam that shrinks, grows and relocates
mid-decode (`TestBatchSeamMovesCarryEveryRowHunyuan`) reads 3.0e-02..4.5e-02
before the first move and 6.2e-02..7.1e-02 after on an RTX 3050 Ti.

The same gate runs both fixtures at the families' 1e-3: their histories move
with no step (1.1e-06 and 2.1e-06 before the first move, 8.5e-07 and 1.5e-06
after). The +64 KiB a fixture's shrink-then-grow round trip left on the card
(512 B of scratch and the page it rounds to) is the lone matvec's staging
growing once, which 0377a8d0 attributed and the harness now holds by running the
round trip twice; synth-hunyuan also runs the relocation and paging gates
(`principleFixtures`), the mixture pages under
`TestMoEFamiliesPageWithTheSameAnswer` as the other families do.

★ **Hunyuan-A13B-Instruct Q4_K_M (tencent's own GGUF, V100 box): 19 of 20
teacher-forced against llama.cpp with the f16 KV cache, 20 of 20 with f32 --
and the twentieth is a ROUTER-TIE BIFURCATION, NOT A DEFECT IN EITHER
ENGINE, though its final margin is 10.5.** Prompt "The capital of France is",
llama.cpp's own continuation fed back; at position 10 ("...Tokyo.\nThe") the
model either continues the pattern (" capital") or opens a "<think>" spelt in
plain tokens ("<th"). Every arm on the same GGUF, host CPU:

    arm                                                  argmax    top     next
    transformers f32, the GGUF's own weights dequantized  <th      13.98    6.86
      + Q8_0 activations                                  <th      14.38    7.12
      + an f16-rounded K/V                                <th      13.98    6.86
      + both                                              <th      14.12    7.06
    jitllm, f32 KV cache                                  <th      14.91    6.46
    jitllm, f16 KV cache (the amd64 default)              capital  18.04    7.53
    llama.cpp, flash attention (its default), f16 or f32  <th      14.28    6.98
    llama.cpp, no flash attention, f32 KV                 capital  18.15    7.68
    llama.cpp, no flash attention, f16 KV                 capital  15.79    6.44

The reference (`HunYuanMoEV1DecoderLayer`, transformers 5.17, one layer at a
time in f32 on gguf-py's dequantization of the same file, eval mode) says "<th",
and each engine lands on both sides of it by a knob that is precision and not
graph: jitllm by its KV width, llama.cpp by its attention kernel. With an f32
cache jitllm agrees with llama.cpp's default at all twenty positions.

The mechanism is the router. At that row the 8th and 9th of 64 experts are
1e-4 apart in probability in most of the 32 blocks (0.0000 to four places in
three of them). jitllm's f32 run swaps one to three experts against
transformers in 21 of 32 blocks and stays on the reference's side; the f16 run
swaps different ones from block 4 and has a different selection outright by
block 8 (five to eight of the eight swapped per block from there), and its
residual parts from the f32 run's by NMSE 1.1e-5 at block 0, 4.8e-2 at block 7, 3.2e-1 at block 8 and
~0.7 after. Quantized activations are what make a cache rounding matter:
transformers with an f16-rounded K/V moves its residual by 8e-9..2.6e-7 at
every block and never grows, while the same rounding with Q8_0 activations
emulated (absmax/127 per 32) moves block 0 by 4.4e-6 -- each re-rounded code is
a step, not a perturbation -- and block 8 by 4e-3. Both engines quantize
activations, so each run of either samples which side of the tie the cascade
falls on.

So a final-logit margin does not bound how close a row sat to a tie
upstream: TestGreedyMatchesLlamaCpp's reading of a margin over 0.5 as "a wrong
answer" assumes the engines see one hidden state, and a mixture with 64-way
near-ties does not give them one. So the gate runs Hunyuan-A13B, by name, on
the f32 cache: the width at which jitllm agrees with the reference
(transformers on the file's own weights) at that row, and with llama.cpp's
default, which is what the gate runs llama.cpp at. On the V100 box (host tier)
that reads 20 of 20 teacher-forced and passes; the f16 cache forced in its
place fails it at position 10 (" capital" at a margin of 10.506 against
"<th", 19 of 20), so the pin is what decides the row and the gate still
parts on it. The instruments are `TestTeacherForcedTrace`
(`-tags jitllmbench`: top logits along a fed chain, every traced vector at one
position) and `scripts/hunyuanstream.py`, which streams the decoder layers over the
dequantized GGUF, which needs ~15 GB rather than the 160 GB of the whole
model.

### Its pre-tokenizer was three llama3 splits

hunyuan-dense's tokenizer.json is three Split stages, each a whole pattern
rather than the alternation skeleton `pretok.parseSplit` reads parameters off:
`\p{N}{1,3}`, the CJK/kana ranges `[一-龥぀-ゟ゠-ヿ]+`,
and Hunyuan's own word split (an ASCII mark glued to the ASCII letters after it,
a prefix that may be a digit, `[\p{P}\p{S}]` where llama3 has
`[^\s\p{L}\p{N}]`, no contractions, no digit alternative). The generated table
had parsed each into a llama3 split, so "12345" tokenized as five digits where
the model's tokenizer has "123" "45" -- the fixture's tokenizer gate caught it.
`OpDigitGroups`, `OpKanaHanRuns` and `OpHunyuanSplit` (recognised by exact
pattern, refused when not Isolated) run them, carried in a container as
`jlm.PreDigitGroups`/`PreKanaHanRuns`/`PreHunyuanSplit`.
`tok.TestHunyuanSplitsAsItsTokenizerDoes` holds 37 cases to HuggingFace
tokenizers running the model's own pre_tokenizer (`scripts/hunyuansplitgold.py`).

The byte alphabet of Hunyuan's vocabulary has no `\r`, and HuggingFace
tokenizers drops it too ("\r\n" encodes as "\n"), so `TestBPEAgainstLlamaCpp`'s
round trip now compares against the text the vocabulary can spell
(`tok.Representable`) rather than the raw text.

★ **AND THE ALLOCATION GATE WAS RED ON EVERY F32 FIXTURE IT LISTED.**
`TestDecodeDoesNotAllocate`'s `host-ksliced` arm pins k-slicing and fails when
no matvec was split -- right for a packed model, and unreachable on an all-F32
one, which has no fused packed matvec to split. synth-glm4moe, synth-qwen2moe and
synth-ernie45moe were added to the list with that arm red; it skips by name on
an all-float model now (`anyPackedBlock`), and the real Hunyuan-0.5B Q4_K_M in
the list runs it: 10752 matvecs split over k, 0 engine allocations; host, cuda
and vulkan decode and every step arm 0 on both fixtures and the real model.

### As it stood in AGENTS.md

★ **hunyuan (Hunyuan-A13B and the 0.5B-7B dense line) NORMS q AND k AFTER
THE ROTARY, AND THE ENGINE RUNS IT WITH NO NEW KERNEL: THE k WEIGHT MOVES INTO
q's AT CONVERSION.** llama.cpp's builders and transformers' classes both apply
the learned per-head RMSNorm after RoPE, an order no tier had. A rotation keeps
each head's RMS, so the unweighted part of the norm commutes with it and only
the weights do not; and the score is a sum over dimensions of (w_q R(q)/rms(q))
(w_k R(k)/rms(k)) with both weights per dimension and shared by every head, so
w_k can ride on q. `convert.foldPostRopeQKNorm` stores w_q*w_k as the q norm
and ones as the k norm; k runs FlagQKNorm's pre-rotary norm unchanged and q's
moves after the rotary (`jlm.FlagQKNormPostRope`, host `attnPrep`, device: the
same `HeadNormRows` kernel launched on the rotated q into `bs.qn`). Every tier
caches the same unweighted k, so the KV cache stays interchangeable
(`TestBatchSeamMovesCarryEveryRowHunyuan`). The mixture is a renormalised
softmax top-k beside one ungated shared expert in every block, refused when a
file states otherwise. An arch code of its own (`jlm.ArchHunyuan`) so an older
reader, which would norm before the rotary, refuses it. Gated on
`synth-hunyuanmoe` and `synth-hunyuan` (`scripts/moegold.py`: transformers' own
classes, A13B's NTK-alpha base at head_dim 128, llama.cpp's converter with the
mixture's tiktoken `set_vocab` swapped for the dense class's tokenizer.json
one); Hunyuan-0.5B-Instruct Q4_K_M is 20 of 20 teacher-forced against
llama.cpp. Hunyuan-A13B Q4_K_M is 20 of 20 with an f32 KV cache and 19 of 20
with the f16 default, and the one is a router-tie bifurcation rather than a
defect, whatever its 10.5 margin says: transformers on the GGUF's own weights
answers as llama.cpp's flash-attention default does, llama.cpp without flash
attention answers as jitllm's f16 cache does, and 64-way expert near-ties (1e-4
apart in most blocks) let each engine's quantized activations pick a side.
Its pre-tokenizer (hunyuan-dense) is three whole-pattern Splits the
table had read as three llama3 splits, so "12345" was five digits;
`pretok.OpDigitGroups`/`OpKanaHanRuns`/`OpHunyuanSplit` run them, gated against
HuggingFace tokenizers (`tok.TestHunyuanSplitsAsItsTokenizerDoes`).

## The mixture families from safetensors

`hfArchTable` gains Glm4MoeForCausalLM, Qwen2MoeForCausalLM,
Ernie4_5ForCausalLM, Ernie4_5_MoeForCausalLM, HunYuanDenseV1ForCausalLM and
HunYuanMoEV1ForCausalLM (`convert/hfmoefamily.go`), each converting to the
container its GGUF converts to. What each needed beyond a name table:

    Glm4Moe        DeepSeek's mixture keys (deepseekMoE) on GQA attention;
                   partial_rotary_factor read rather than refused; q/k norm off
                   the weights (use_qk_norm); the prediction block set aside
    Qwen2Moe       hfMoEConfig plus the shared expert and its sigmoid gate,
                   which hfMoEConfig refuses for every other class
    Ernie4_5       llama's graph with NO permutation: the class already rotates
                   adjacent pairs, the container's own layout
    Ernie4_5_Moe   ERNIE's own key names; a block routes when (il+1) %
                   moe_layer_interval == 0 past moe_layer_start_index, checked
                   against the weights; an end index before the last block is
                   refused (the container's rule has no end)
    HunYuan*       per-layer lists (moe_topk, moe_intermediate_size: hfUint,
                   refused when they differ), the NTK-alpha base, use_cla
                   refused, the after-rotary q/k norm folded as on the GGUF side

The gate per class is read off the CLASS (RULE 7m): Glm4Moe joins
`hardcodesSigmoid`; `hardcodesSoftmax` is its twin for Qwen2Moe, Ernie4_5_Moe
and HunYuanMoEV1; a `scoring_func` that contradicts the class is refused.

Two converter bugs the new fixtures found:

- `tokenizer_class` "PreTrainedTokenizerFast" does not read tokenizer_config's
  `add_bos_token`; the converter obeyed it and put a BOS before every Hunyuan
  prompt, where transformers' AutoTokenizer, HuggingFace tokenizers and
  llama.cpp all add none. The generic class now leaves the post-processor's
  answer alone.
- ERNIE keeps its correction bias and Qwen2-MoE its shared-expert gate as
  [1, n]; llama.cpp's converter writes that shape unchanged, and the
  safetensors path matches it byte for byte (ExpProbsB accepts [n, 1]).

`convert.TestMixtureFamiliesConvertAlikeFromEitherInput` converts each
family's fixture from its safetensors and from llama.cpp's GGUF of the same
weights: 21 config fields agree and every tensor is byte-identical (78, 54, 48,
29, 47 and 35 tensors). Dropping the fold on the safetensors path makes 6
tensors differ there and reads 3.3e-03 / 8.2e-03 against transformers from
position 1 -- position 0 is exact, the rotary being the identity there.
`model.TestSafetensorsMatchTransformers`, decode against each class: glm4moe
7.96e-12, qwen2moe 1.18e-12, ernie45moe 1.09e-11, ernie45 3.71e-13,
hunyuanmoe 3.82e-11, hunyuan 5.82e-12 (the goldens, testdata/golden/hf/*-hf,
are scripts/hfgold.py over moegold's saved fixtures, linked into the model
directory as <name>-hf). The real Hunyuan-0.5B-Instruct (bf16 safetensors,
tencent's repo) converts and is 20 of 20 teacher-forced against llama.cpp on
bartowski's Q4_K_M.

`model.TestSafetensorsQ8MatchTransformers` was red on synth-glm4moe-hf at
1.47e-01 against its 5e-02 bound -- at ONE position: position 7 jumps 50x while
positions 0-6 read 1.1e-03..3.1e-03, and the float path reads 7.96e-12 at all
eight. It is a top-k flip, read off the router rather than off that shape
(`TestQ8RouterMargins`, `-tags jitllmbench`, which traces the router's logits
as `moe_logits` and scores them as the selection does, sigmoid plus the
selection bias). Block 2 at position 7, top-2 of 8:

    float   selects [5 0]   0: 1.07118   2: 1.04140   margin 0.0298
    Q8_0    selects [5 2]   2: 1.06435   0: 1.06223   margin 0.0021

Q8_0 rounding moves a selection score by up to ~0.02 on this fixture (expert 2
here by +0.023), so a 0.0298 lead is within reach of it; the same block at
position 1 holds a 0.0031 lead in both containers, because the rounding
happened to move both experts the same way. Every other position selects the
same experts in both.

So the tolerance is kept and made exact: a position over the bound passes only
where some mixture block chose other experts than the float container did at
that position (the gate decodes the float container beside the Q8 one and
compares the traced selections). Against the bound pulled down to 1e-3, every
over-bound position of synth-deepseek, synth-glm4moe-hf and synth-qwen2moe-hf
that is not a flip fails by name, and position 7 alone is excused.

## YaRN's attention_factor

A YaRN config may state `attention_factor`. transformers'
`_compute_yarn_parameters` then puts it on cos and sin in place of the derived
magnitude -- mscale(f, mscale)/mscale(f, mscale_all_dim) when both are set and
non-zero, else mscale(f, 1), with mscale(f, k) = 0.1*k*ln(f) + 1 -- and the
DeepSeek family's attention classes (DeepseekV2/V3, Glm4MoeLite) multiply the
softmax scale by mscale(f, mscale_all_dim)^2 whatever the rotary did
(`yarn_apply_mscale`). llama.cpp's converter writes the stated value as
`rope.scaling.yarn_attn_factor`, and llama.cpp's runtime reads that key for grok
alone: its context setup derives the magnitude for every other architecture.
The engine read the key on neither input.

It reads it on both now (RULE 7m, chosen: transformers, the model's own
reference). The GGUF path (`convert.ropeScalingOf`) puts a stated
`yarn_attn_factor` on cos and sin for every family, mistral3 included, and
DeepSeek keeps its score correction from `yarn_log_multiplier`. The safetensors
path (`convert.hfYarnApply`) is transformers' arithmetic case for case, and
setting it down that way found a second bug: with `mscale` and `mscale_all_dim`
both present -- every shipped DeepSeek -- it put mscale(f, mscale_all_dim) on
cos and sin as well as squaring it into the score, where transformers and the
GGUF path leave cos and sin at one. DeepSeek-V2-Lite from safetensors read
AttnFactor 1.2608 against its GGUF's 1, so the score carried the correction to
the fourth power. `convert.TestYarnAgreesAcrossInputs` now holds the real
checkpoint's two inputs to one container (factor 40, AttnFactor 1, YarnLogMul
0.0707 on both), and the mscale-differs refusal is gone: the container carries
the rotary's and the score's magnitudes in two fields.

Four fixtures (`scripts/c6gold.py`: transformers' class, llama.cpp's converter;
the same weights as safetensors are `<name>-hf`), each stating or deriving the
magnitude a different way:

    fixture                  cos/sin  score             host GGUF   host HF    cuda/vulkan
    synth-olmo3-af           1.5      1/sqrt(hd)        3.5e-12     1.6e-12    <=1.2e-06
    synth-mistral3-af        0.75     1/sqrt(hd)        4.5e-12     2.1e-12    <=4.2e-07
    synth-deepseek-yarn      1        mscale(8,.707)^2  5.0e-12     1.8e-12    <=3.3e-12
    synth-deepseek-yarn-af   0.8      mscale(8,.707)^2  6.3e-12     1.9e-12    <=4.7e-12

(worst logit NMSE against transformers on the host -- decode, prefill at every
length and the batch for the GGUF arm, decode for the safetensors one -- and
against the host for every block and the head on cuda:0 and vulkan:0.)

Against the code before: the GGUF arm misreads synth-olmo3-af, synth-mistral3-af
and synth-deepseek-yarn-af (their containers carry the derived magnitude, and
`TestC6MatchesTransformers` refuses them at the configuration check); the
safetensors arm reads 3.5e-03 on synth-olmo3-af-hf, 1.39e-01 on
synth-deepseek-yarn-hf (the doubled DeepSeek correction) and 3.56e-01 on
synth-deepseek-yarn-af-hf. Taken out of the engine one at a time
(`TestC6FeaturesAreLoadBearing`): the derived magnitude for the stated one
8.6e-03 (olmo3) and 5.6e-01 (mistral3), DeepSeek's score correction on cos and
sin too 4.5e-01, no score correction 2.8e-01, the stated 0.8 ignored 5.3e-01.

synth-deepseek-yarn-af states 0.8, not the 1.5 it was built with first: at 1.5
the sharpened softmax made position 3 of the Q8_0 arm
(`TestSafetensorsQ8MatchTransformers`) read 7.0e-02 from rounding alone, with
the float arm at 1.6e-12.

Every YaRN family, and what moved: no GGUF on either Linux box states
`yarn_attn_factor` (DeepSeek-V2-Lite, DeepSeek-Coder-V2-Lite, DeepSeek-V2.5,
gpt-oss-20b/120b, both Hunyuans, the gemma3s and every synth fixture before
these), so every GGUF-made container is unchanged by construction; Qwen and
llama read YaRN from GGUF only (`hfRopeScalingOK` refuses it from safetensors).
From safetensors: Ministral3 already read it (`ministral3Rope`) and is
unchanged; OLMo 3 and every DeepSeek-family class read it now; and the
DeepSeek change above moves every DeepSeek safetensors conversion that carries
YaRN with both mscale keys -- V2, V2-Lite, V3, R1, Kimi-K2 -- onto
the GGUF's container. No safetensors fixture before these carried YaRN apart
from synth-hf-ministral3, so no existing golden moved.
eight. The reading is a top-k selection flipped by Q8_0 rounding (a sigmoid
router with a selection bias drawn at 0.5 near-ties experts); it is inferred
from that shape, not from the router scores. The gate's bound is for a wrong
layout, type or scale, each of which moves every position, so it fails at two
positions over the bound now rather than one.

### As it stood in AGENTS.md

★ **YaRN's STATED attention_factor IS transformers', AND llama.cpp NEVER READS
IT.** llama.cpp's converter writes a config's `attention_factor` as
`rope.scaling.yarn_attn_factor`; its runtime reads that key for grok alone and
derives the magnitude everywhere else. transformers' `_compute_yarn_parameters`
puts the stated value on cos and sin in place of the derived one, so the engine
does too, on both inputs (`convert.ropeScalingOf`, `convert.hfYarnApply`) --
the DeepSeek family's score correction, mscale(f, mscale_all_dim) squared, stays
beside it. Where nothing is stated, each family keeps its derivation:
0.1*ln(f)+1 for the generic classes and gpt-oss, `mistral3Yarn`'s ratio, and for
DeepSeek mscale(f, mscale)/mscale(f, mscale_all_dim) on cos and sin -- exactly
one for every shipped DeepSeek. The safetensors path had folded mscale(f,
mscale_all_dim) into cos and sin as well, so DeepSeek-V2-Lite from safetensors
scaled them by 1.2608 where its GGUF and transformers scale them by 1, and the
score took the correction to the fourth power rather than the second;
`convert.TestYarnAgreesAcrossInputs` holds the two inputs to one container.
No GGUF on either Linux box states the key, so every existing container is
unchanged by construction. `docs/engineering-history/model-correctness.md`,
"YaRN's attention_factor".

## Gemma 3 vision and InternVL 2.5 / 3: two more projectors, held to transformers

Two projector families on the one-container vision path: `gemma3` (SigLIP-so400m,
27 blocks of 1152 over 4096 patches of an 896 picture, a 4x4 average pool, an
RMSNorm, one matrix into the text width) and `internvl` (InternViT-300M, 24
blocks of 1024 with a class token and per-block layer scales, the 2x2 shuffle,
then LayerNorm, linear, GELU, linear; up to twelve 448 tiles and a thumbnail).
The text models (gemma3, qwen2) were already here. Real checkpoints:
gemma-3-4b-it Q4_K_M with ggml-org's mmproj-model-f16, InternVL3-1B-Instruct
and InternVL2_5-1B Q8_0 with their mmproj; transformers 5.17 on the CPU in
float32, with only the vision tensors fetched (vlm/fetchst.py) and the classes'
own code; llama.cpp b10825.

### The converter

- **InternViT's layer scales fold into the matrices they follow.** `ls1` scales
  the attention output and `ls2` the MLP's, per output channel, so
  `diag(ls)(Wx + b) = (diag(ls)W)x + ls*b`; the block every tier runs is then
  CLIP's unchanged and no device needed a new feature. The contraction is found
  by shape, as buildTower finds it. On a Q8_0 source the fold rescales each
  block's f16 scale and leaves the codes.
- **SigLIP's MLP is 4304 wide, 16 short of a Q8_0 block.** `padFFN` widens it to
  4320 with zero rows, a zero bias and zero columns: the same function to the
  bit (`TestPadFFNIsTheSameFunction`), and a width every packed kernel reads.
- **gemma3's input projection is stored [text, tower]** -- the reference
  computes `x @ W` and llama.cpp transposes it inside its graph on every image;
  the converter lays it out k first once.
- InternVL's projector names `mm.model.mlp.{0,1,3}` mean a LayerNorm and two
  matrices where llava's `mm.0` is a matrix, so a projector's own vocabulary is
  consulted before the shared table. InternViT-6B (26B and up: RMSNorm, a q/k
  norm) is refused by name.

### Preprocessing is the processor's, to the bit

Both processors are transformers TorchvisionBackend classes: a uint8 tensor
resized with torchvision's antialiased resampler -- PIL's filter, its support
widened by the downscale factor, weights normalised and turned into int16 at
the widest precision the largest weight allows, a horizontal pass rounded to
uint8 and then a vertical one -- and only then rescaled and normalised in
float32 (`(x - 255*mean) / (255*std)`, mean and std scaled first). The
algorithm was reconstructed in numpy and matched torchvision bit for bit on six
shapes before it was written in Go (`engine/model/imageproc.go`). InternVL's
tiling is GotOcr2's: the grid closest to the aspect ratio among every
(columns, rows) with 1..12 tiles, ties to the later (larger) grid while the
picture's area exceeds half of it, row-major crops, the thumbnail last.

    gemma3     quad-and-disc 384x384, photo 640x488       0 of 2,408,448 values differ each
    internvl   quad-and-disc (1 tile), photo (13 tiles)   0 of 602,112 and of 7,827,456
    violations CLIP's mean/std: every value; the plain bilinear resize: 579,222 to
               2,186,960 values (worst 0.0073 on gemma3); the tiles reversed: 6,955,080

It is per-image setup in integer arithmetic, not a token's work (and every
pixel of it now runs on generated code: "Image preprocessing is generated
code" below). The pictures are PNG on both sides: Go's JPEG decoder is not libjpeg-turbo, and a JPEG
differs by a few levels before any of this runs. The older projectors
(idefics3, mlp, qwen2vl_merger) keep their two-tap bilinear, which was never
held to their processors.

### The towers, node by node and end to end

Against llama-mtmd-debug's dump of the rainbow picture (raw floats, no
preprocessing in it), relative RMS over the 36 printed values per node:

    gemma3     block 0 0.33%, climbing to 4-6% by blocks 18-25; projector 3.3% (amd64 and M4)
    internvl   blocks 1-8%; projector 4.6% amd64, 6.7% M4
    violations pooling skipped 0.563; shuffle transposed 0.275

Against transformers' float32 image features (NMSE over every value):

    gemma3     rainbow 4.6e-3, photo 6.6e-3, grid 4.0e-3; pooling skipped 0.78
    internvl   rainbow 7.6e-3, photo (13 tiles) 8.7e-3, grid 1.0e-2
    projector alone (rows that are no picture): 6.2e-5 / 6.3e-5;
               pool skipped 1.85, shuffle transposed 5.0e-2

**The distance from transformers is the tower's int8 activations, measured
rather than assumed.** Transformers' own tower with every Linear's input
rounded to q8_0 per 32 and its weights to Q8_0 reads 2.1e-3 (gemma3) and
3.0e-3 (internvl) against itself; the weights alone 2.7e-4 and 2.9e-4; adding
the patch embedding's quantization takes gemma3's to 4.35e-3 against jitllm's
4.6e-3. InternVL's hidden states track that simulation layer by layer (block
24: 5.2e-3 against 4.3e-3), its projector on jitllm's own last hidden state is
exact to 8.5e-5 and its LayerNorm to 4.4e-10, and the projector amplifies the
hidden error's shape into the 7.6e-3. llama.cpp (f16 weights, f32 activations)
reads 1.1e-3 relative RMS against transformers at its 36 printed values.
**Found and not fixed**: a tower path at higher precision than int8 is the
lever.

The shuffle's ORDER is not something the end-to-end gate can see: the reference
itself moves by NMSE 1.9e-2 (rainbow) to 2.8e-2 (a four-colour patch grid) when
its 2x2 groups are transposed, because after 24 blocks neighbouring patches look
alike -- inside the gate's own precision. `TestProjectorMatchesTransformers`
runs the projector alone on rows that are no picture and catches it at 790x.

### The class token goes first, which llama.cpp gets wrong for InternVL

`clip_graph_internvl` concatenates [patches, class] and adds the position table
in row order, so patch 0 takes the class token's position and every patch the
one before its own. The reference puts the class token first. Against
transformers, llama.cpp's layout reads NMSE 0.18 (photo) and 0.25 (rainbow)
where the reference layout reads 8.7e-3 and 7.6e-3: jitllm follows the
reference (RULE 7m, chosen). `towerCLSLast` reproduces llama.cpp's layout only
so the node-by-node gate compares the same computation; in it, teacher-forcing
llama-mtmd-cli's answer on the photo agrees at 31 of 32 positions on amd64 and
32 of 32 on the M4, against 27 of 32 in the reference layout.

### gemma3's image rows are bidirectional in the text model

Gemma3's reference masks an image's soft tokens to see each other
(`token_type_ids`: OR(causal, blockwise), and on the sliding layers AND with
the window, anchored at the row's own position); llama.cpp decodes the image
chunk non-causally. `Span.Bidir` carries it: the host widens a run's rows' key
COUNT to the run's end, a prefill chunk never ends inside a run (`bidirChunk`),
and a device takes the runs through `nn.BidirDevice` -- the contiguous cache's
per-row count, or the paged cache's row descriptor end, whose start stays at
the row's window. The contiguous cache's windowed kernels anchor the window at
the count, so a run ending past the window is refused there by name.

    host, causal against bidirectional            logit NMSE 1.38e-2
    device against host, bidirectional            CUDA (RTX 3050 Ti, 20/34 blocks) 1.46e-4,
                                                  Vulkan 1.42e-4, Metal 1.57e-4, V100 CUDA 1.49e-4
    a run never cut: 64-row chunks against 512    bit-identical; cut, 1.2e-2 (M4 2.5e-2)

### The prompt and the answer

`ChatSpans` equals transformers' `apply_chat_template(tokenize=True)` id for id:
282 ids for gemma3 (the soft tokens with "\n\n" around the markers, merging
with the template's own newlines exactly as the processor's one tokenization of
the expanded string does) and 3,357 for InternVL3's 13-tile photo. Two things
found: a string-content template was handed the placeholder and a newline,
llava's convention, which put an extra newline after gemma3's image
(`imageMarkers.inline`); and transformers' processor called on rendered text
adds a second BOS to Gemma's template, which writes its own.

Teacher-forcing llama-mtmd-cli's greedy answer (same chunks, a picture already
at the tower's size): gemma3 29 of 32 on amd64 and 30 of 32 on the M4, the
first disagreement a 0.019 tie at token 14 -- the token at which llama.cpp's own
CUDA tower parts from its CPU tower on the same answer. The causal violation's
disagreements sum to 8.8 logits against 3.9. Every model describes the real
picture and not the blank one (`TestChatSpansDescribeAnImage`), and the CLI
reads the photograph's 13 tiles: InternVL2.5-1B answers "a vintage newspaper
clipping from The New York Times dated July 21, 1969, featuring a headline
about astronauts landing on the moon".

### On the device: 4096 patches did not fit, and now the attention goes in query chunks

A vision block is one submission, and its score and probability planes were
rows x heads x stride: 1.07 GB each for gemma3, 2.15 GB with nothing else, so
the 4 GB card placed none of 27 blocks. `jit_old/gpu/tier/visionattn.go` gathers a
chunk of query rows (`kernels.CopySpan`), runs the same score, softmax and
accumulate kernels built for the chunk, and scatters the output back
(`kernels.CopyAt`) whenever the planes exceed 128 MiB -- 16 chunks of 256 rows
for gemma3, none for every earlier tower. An ungated MLP's never-read gate
plane (70 MB here) is no longer allocated.

    gemma3     after block 0: CUDA 4.5e-8, Vulkan 4.5e-8, Metal 6.9e-6, V100 6.8e-6 (tensor cores)
               all 27 blocks: 3.1e-4, 3.0e-4, 2.2e-3, 2.3e-3
    internvl   after block 0: 1.9e-8, 1.7e-8, 4.4e-8; all 24 blocks: 7.4e-4, 7.4e-4, 7.5e-4

### The principles

The towers' blocks page under a two-page budget with the unbudgeted answer bit
for bit (25 and 22 evictions, amd64 and M4); they come home from every device
with the host's answer bit for bit; a warm text decode after a picture makes
zero engine allocations on the host and on CUDA, Vulkan (laptop and V100) and
Metal (`TestVisionDecodeDoesNotAllocate`).
## MiniCPM-V: a tower that reads any grid, and a picture cut into an overview and slices

MiniCPM-V 2.6, 4.0 and 4.5 share llama.cpp's `resampler` projector
(minicpmv_version 3-6) and one graph: a SigLIP-400M tower whose 70x70 position
table is read by NaViT bucketing, so a grid of any shape reads it, and a
perceiver whose 64 learned queries cross-attend over the patches with a 2-D
sin/cos on the keys. Built and measured on MiniCPM-V 4.0 (`openbmb/MiniCPM-V-4-gguf`,
text arch llama, 3.6B) -- the one generation whose reference code is not gated
on Hugging Face. 2.6's checkpoint is behind a licence wall, so its transformers
reference is not on this box; its mmproj converts by the same code.

    jitllm convert MiniCPM-V-4-Q4_K_M.gguf mmproj-MiniCPM-V-4-f16.gguf out.jlm
    27 vision blocks of 15.48 MiB; 454 tower tensors (resampler.pos_embed_k
    is loaded by llama.cpp and read by no node of its graph: dropped by name)

What it needed, and none of it is a second model:

| piece | what it took |
|---|---|
| any grid | `TowerState.EncodeGrid(px, gh, gw)`, scratch sized for `TowerConfig.MaxSeq` (a llava-uhd plan rounds a side of 1.5 patches up to 2, so 4/3 of the square grid plus a row and a column); the device's vision key count was already per call, so the tier needed only the rows of THIS grid passed (`s.x[:n*NEmbd]`) |
| bucketed table | `bucketPositions`: floor(70*i/n) on each axis |
| an FFN of 4304 | nothing new: `convert.padFFN`, built for Gemma 3's same SigLIP, pads it to a whole Q8_0 block exactly |
| the resampler | `resampler.go`, on the host: a JIT of its own on the tower's pool (128-wide heads at stride 2560 are a second attention shape) |
| the cut | `Tower.Layout` from the picture's SIZE alone, so the prompt and the State exist before a pixel is resampled; `Model.PictureSpans` lays out `<image_id>0</image_id><image> OV </image><slice> S </slice>...\n...`; `ChatSpansParts` places it |
| the resize | Pillow's BICUBIC: `imageproc.go`'s antialiased resampler, the one Gemma 3 and InternVL use, at Pillow's fixed 22-bit precision (`resizePIL`) where torchvision picks the widest an int16 holds -- the only difference between the two, and it moves samples by a step |

★ **THREE PLACES llama.cpp PARTS FROM THE PROCESSOR, EACH A CHOICE (RULE 7m),
AND jitllm FOLLOWS THE PROCESSOR IN ALL THREE.**

    mtmd drops <image_id>N</image_id>             the processor writes it (use_image_id)
    a picture with neither side over 448          the processor keeps its aspect
      is squashed to 448x448 by mtmd              (find_best_resize, upscaling)
    a long thin picture under 448^2 is sliced     the processor does not cut it
      by mtmd (it slices on a side > 448)         (ceil(area/448^2) <= 1)

and one where the reference itself is not a single answer: the bucket of
i/n. torch builds both i/n and the k/70 boundaries in float32 -- on an AVX2
host sixteen at a time from a base rounded to float32, which is what
reproduces it bit for bit -- and parts from the rational floor(70i/n) by one
bucket at 1078 coordinates of the axis lengths below 1500 (none of a 448
slice's 32). A model trained on GPUs never saw the CPU's rounding; jitllm takes
the rational, as llama.cpp does, and `TestBucketPositionsMatchTorch` counts
the divergence from torch's own output.

The pictures the gates use are the two on which all three agree: the
repository's 384 square (an overview, upscaled to 448 by both) and llama.cpp's
`test-1.jpeg` resampled by Pillow to 896x448 (two whole 448 slices, an
overview of 630x322 = 45x23 patches, and a refined picture equal to the
original, which both copy).

### The numbers

    Pillow's resize, 5 sizes up, down, unequal     0 of 2.2M samples off
      violation, torchvision's precision          259 of 2.2M off
    the processor's pixels (overview and slices)  0 of 1.8M off
    the cut, 24 picture sizes                     every one the processor's

Against transformers' own `SiglipVisionTransformer` and `Resampler`
(scripts/minicpmvgold.py, float32, the checkpoint's bf16 weights):

    piece          tower NMSE   end to end   resampler alone
    overview 45x23  1.18e-02     1.19e-03     2.6e-04
    slice 1 32x32   1.23e-02     6.96e-03     1.6e-04
    slice 2 32x32   7.91e-03     1.21e-02     1.6e-04
    384 square      1.22e-02     4.88e-03     3.0e-04

    violations     positions read without buckets  9.1e-01 .. 1.09
                   keys without their 2-D position 9.0e-04 .. 1.7e-03
                                                   (5-6x the resampler's clean
                                                   reading: a key's position
                                                   moves the answer little)
                   queries without their LayerNorm 5.2e-01 .. 6.7e-01
                   slices reversed                 9.4e-01 on both slices
                   no mean or std                  8.3e-02 .. 3.0e-01

★ **THE TOWER'S 1.2e-02 IS THE INT8 ACTIVATION ON A TOWER WITH MASSIVE
ACTIVATIONS, AND llama.cpp's f16 RUN DOES NOT PAY IT.** On llama-mtmd-debug's
own checkerboard, transformers against llama.cpp b10825 and against jitllm,
relative RMS over the dump's 36 printed values:

    node          llama.cpp vs transformers   jitllm vs llama.cpp
    pos_embed     0.0000                      0.0036
    layer_out-0   0.0005                      0.0179
    layer_out-13  0.0012                      0.0455
    layer_out-26  0.0054                      0.1916
    output        0.0010                      0.0404     (sums agree to 5e-4)

From block 13 on, nine of the residual's 1152 channels sit at ~2000 against a
median of ~0.3 (SigLIP-so400m's massive activations), so every 32-wide
activation window holding one of them quantizes its neighbours away. llama.cpp
runs the f16 mmproj's matmuls in float and does not. This is the engine's
activation quantization, not this tower's graph -- the answers below are
token-identical to llama.cpp's -- and it is a measured lever for any tower of
this family (Gemma 3's is the same SigLIP): a window that excludes the outlier
channels, or a float path for them.

On every backend, every one of the 27 blocks placed, the 45x23 overview:

    cuda:0 (RTX 3050 Ti)   tower 6.32e-03   embeddings 9.56e-04 against the host
    vulkan:0 (same card)   tower 7.39e-03   embeddings 9.65e-04
    metal (M4)             tower 5.29e-03   embeddings 6.85e-04
    after one block placed 6.7e-08 / 7.6e-08 / 7.4e-08
    violation, the device handed the scratch's every row instead of this
    grid's (the padding rows become keys)     tower 1.64e-01

End to end, the photo (the New York Times front page of 21 July 1969):

    llama.cpp b10825   "A newspaper headline that says Men Walk on Moon."
    jitllm (laptop)    "the front page of the new york times from july 21, 1969."
    blank picture      "the image is a white blank page." (an all-zero RGBA, which
                       transformers' convert_to_rgb composites over white)
    teacher-forced on llama.cpp's own prompt (BOS, no image id, 214 positions,
    the same count)    10 of 10 positions agree on the laptop; 9 of 10 on the
                       M4, the tenth a 0.049 tie at the first position

The five principles: the tower's blocks page -- budgets of 1 and 3 vision
pages, 26 and 24 evictions, every output element identical; they relocate
cuda/vulkan/metal -> host -> back, the two device encodes bit-identical; and a
decode after a three-piece picture makes 0 engine allocations over 48 tokens on
the host, CUDA, Vulkan and Metal.

★ **AND ONE DIVERGENCE IS NOT THIS FAMILY'S TO CHOOSE.** MiniCPM-V 4.0's chat
template renders no BOS and its vocabulary adds one; llama.cpp and the
processor both start the prompt with it, and jitllm's chat path, by its
documented rule (the tokenizer adds no special the template did not render,
`TestChatTemplateIsAppliedAndSpecialsAreNotDoubled`), does not. The answers above came without it. The
teacher-forced gate builds llama.cpp's prompt, BOS included.

### As it stood in AGENTS.md

★ **A TOWER READS ANY GRID NOW, AND A PICTURE MAY BE SEVERAL ENCODES.**
MiniCPM-V's tower reads a 70x70 position table by NaViT bucketing, so
`State.EncodeGrid` (a vision State) takes a grid of any shape up to `TowerConfig.MaxSeq`,
on the host and on every device (the vision block's key count was already per
call); and its processor cuts a picture into an overview and up to nine slices,
which `Tower.Layout` plans from the picture's SIZE alone, so the prompt and the
State it sizes exist before a pixel is resampled. `Model.PictureSpans` lays the
pieces out in the model's markers and `ChatSpansParts` places them. Every piece
is rows through the same blocks; nothing about a slice is a second model.
`docs/engineering-history/model-correctness.md` has the measurements and the
three places llama.cpp parts from the processor.

## Janus-Pro: llava's aligner under another name, and a letterboxed picture

Janus-Pro-1B (`mradermacher/Janus-Pro-1B-GGUF`, Q4_K_M text on llama, a Q8_0
mmproj with projector_type `janus_pro`) is the CLIP-style tower jitllm already
ran -- SigLIP-L, 24 blocks of 1024, a 576-row table for one 384 square, a
post-LayerNorm -- and llava's linear-GELU-linear aligner. Three things were
not already there, and none of them is a graph:

    the second matrix is mm.1 where llava's is mm.2   convert.towerName renames it
    the picture is letterboxed, not squashed           the longest side to 384 at
      (JanusImageProcessorPil: round(), Pillow           the picture's aspect, centred
      BICUBIC, background int(0.5*255) = 127)          on 127 (image.go letterbox)
    the markers                                        <begin_of_image> 576 <end_of_image>,
                                                       the processor's; mtmd wraps nothing

llama.cpp's port takes ceil() of each letterboxed side where the processor
rounds; on the photo (896x448 -> 384x192) the two agree.

    the processor's pixels       0 of 442,368 values off, on the photo and the square
    tower against transformers   1.47e-02 (photo), 4.15e-02 (square); aligner 1.35e-02, 2.94e-02

The tower's number is three things, measured apart on transformers' own tower:
its MLP's erf GELU is tanh here as in llama.cpp's use_gelu (7e-4); the mmproj's
weights are Q8_0 (with the tanh, 6.3e-3 and 2.7e-3); the engine's int8
activations over 24 blocks are the rest. Against llama.cpp's own run of the
same Q8_0 file on its 384 checkerboard, relative RMS over the dump's 36 values:
0.006 after the embedding, 0.008 after block 0, 0.022 after block 11, 0.062
after block 23, 0.080 out of the aligner, every sum within 7e-4 of llama.cpp's.

    every block on cuda:0 / vulkan:0 / metal   8.7e-03 / 5.6e-03 / 1.8e-02 against the host;
                                              after one block 9.4e-07 / 2.3e-08 / 4.1e-05
    llama.cpp b10825 (--jinja)  "This image is a scanned cover of The New York Times from
                                 June 20, 1960, reporting on astronauts landing on a plain,
                                 collecting rocks, ..."
    teacher-forced on its prompt (BOS, no markers, 591 positions, the same count)
                                 32-33 of 34 agree, the rest ties of 0.000-0.13 on the
                                 digits of the year, laptop and M4
    jitllm, the processor's markers  "the image is a scanned newspaper clipping from the new
                                 york times, dated september 23, 1960, featuring a headline
                                 about astronauts landing on a plain, collecting rocks, and"
    blank picture                "the image depicts a white background with a faint outline
                                 of a small, abstract figure resembling a person."

The tower pages under budgets of 1 and 3 vision pages bit-identically (23 and
21 evictions), relocates device -> host -> device with the two device encodes
identical on CUDA, Vulkan and Metal, and a decode after a picture makes 0 engine
allocations on the host, CUDA, Vulkan and Metal.

## M-RoPE, the 2-D vision rotary as one capability, and Qwen2.5-VL

### M-RoPE is a property of the row

Qwen2-VL's text rotary splits a head's 64 pairs [16, 24, 24] between the
(t, h, w) coordinates of a position. A text row has all three equal, which is
why the architecture ran verified for months with nothing consuming
`Config.RopeSections`; an image's rows do not. Row (y, x) of a still image whose
span starts at rotary position s turns at (s, s+y, s+x), and the text after it
resumes at s + max(H, W) rather than at s + H*W -- so from the first image on a
row's rotary position and its cache position part, by Qwen2-VL-2B's 392x392
picture 182 positions (transformers' rope delta -182).

Read off transformers' `get_rope_index` / `get_vision_position_ids` and
llama.cpp's mtmd n_pos, and built as rows rather than as an image pipeline:

    Span.Grid          an image span's merged grid; ChatSpans/ChatSpansImages set it
    State.spanRope     every row of a mixed prompt on the three axes
    ropeMark, ropeOf   the offset a prompt leaves for the rows after it, per slot,
                       read by decode, ForwardBatch, the ragged step, StepRuns
                       and speculation; rewound and retired with the position
    nn.Rope.TableAt    the table: the generated table kernel once per run
                       (RopeRun: pairs, axis, first frequency, stride), each over
                       its own folded planes, so a row whose coordinates agree is
                       the plain table to the bit (TestRopeTableAtIsThePlainTable...)

No rotation kernel on any tier learned M-RoPE. A device builds its table from a
row's CACHE position, which an image moves the rotary position off, so an M-RoPE
rope offers no folded planes (`ropeTabPlanes`) and the device takes the host's
table on the upload path it already had; `TestMRopeMatchesTransformers` fails if
the device builds one.

The prefix cache names an embedding row by its values AND its (t, h, w)
(`PrefillCachedMixed`, `spanKey`): the same picture under a 3x5 and a 5x3 grid
is the same 15 rows, and a key of values alone would hand one the other's
pages. The gate checks both halves -- keys of values alone are equal, keys
with positions are not -- and the transposed grid restores 0 rows where the
same prompt restores all of them.

transformers' own Qwen2VLForConditionalGeneration / Qwen2_5_VLForConditionalGeneration,
written to GGUF by llama.cpp's own converter (scripts/qwenvlgold.py), the image
rows the reference's own tower output so the text model is pinned alone:

| fixture | host | batch slot | CUDA | Vulkan | Metal (M4) | sequential image (the "before") |
|---|---|---|---|---|---|---|
| synth-qwen2vl | 2.1e-12 | 2.1e-12 | 4.1e-07 | 1.1e-12 | 2.9e-07 | 5.0e-01 |
| synth-qwen25vl | 3.5e-12 | 3.5e-12 | 4.5e-07 | 2.2e-12 | 5.9e-07 | 1.3e+00 |

(CUDA's and Metal's prompt chunk attends through f16 tiles; their decode, batch
slot and StepRuns rows are f32.) The paging (one frame), relocation (onto the
device after the image, home again) and warm-decode-after-an-image allocation
arms hold the same 1e-12 and 0 engine allocations on the host, CUDA, Vulkan
and Metal (`TestMRopeKeepsTheFivePrinciples`).

The real Qwen2-VL-2B, converted by llama.cpp's converter at F16 from the same
checkpoint transformers runs in float32 (scripts/qwenvlreal.py), on
quad-and-disc.png at its 14x14 merged grid, 12 positions teacher-forced:

    with M-RoPE (after)          worst logit NMSE 6.6e-12, 0 argmax flips
    sequential image (before)    worst logit NMSE 1.4e-02, 1 argmax flipped

★ **A GATE WOULD HAVE MISSED THE FIRST VERSION'S BUG IN EVERY OTHER MODEL.**
`spanRope` returned the prompt's START as the next rotary position on a
single-axis model, so the mark it left put every row after a SmolVLM or llava
prompt at position minus the prompt's length: fluent and wrong, and
`TestPrefillMixedMatchesPrefill` -- which compared only the prompt's last logits
-- passed it. `TestChatSpansDescribeAnImage` caught it as two descriptions that
had stopped naming the picture; the seam gate now decodes a token after the
mixed prompt too, and fails at 3.7 against the violation.

### The 2-D vision rotary, one capability

Qwen2-VL's ViT, Qwen2.5-VL's and Pixtral's rotate q and k by a patch's (row,
column), and differ only in which frequencies turn which pairs:

    Qwen2-VL / Qwen2.5-VL   two runs of HeadDim/4 pairs, each restarting at the
                            first frequency of a half-head rotary (row, then col)
    Pixtral                 the even frequencies on the row, the odd on the column

Both are `nn.Rope.Runs` (TestRopeTableAtMatchesTheTranscendentals carries all
three shapes), so the tower builds one table per image -- a row per patch, in
the order the tower runs its patches -- and both tiers rotate with the text
block's kernels over it: the host's RoPE32JIT and the device's RoPERows /
RoPERowsT, through Layers' cs. `LayerPlan.NRot` is HeadDim on a 2-D rotary
vision block and zero on one with a position table.

★ **THE DEVICE RAN QWEN2-VL's TOWER WITH NO ROTARY AND NOTHING DECLINED.** The
plan a tower offered carried no rotary, so the tier copied k into its cache
unrotated (CopyAtRowsT) and the attention ran over positionless patches: `jitllm
run -devices cuda:0 -image` encoded every picture that way. With the rotary in
the plan, one block reads 2e-5 (CUDA), 4e-8 (Vulkan), 3.7e-5 (Metal) against the
host, and the old plan 3.3e-1 (`TestRotaryTowerOnDeviceMatchesHost`).

★ **AND A TOWER OFFERED BEFORE ITS TEXT MODEL REFUSED THE TEXT MODEL.** The tier
built its decode scratch from whichever block arrived first, a vision one
included, and then refused every text block for the tower's attention scale --
Qwen2-VL's head of 80 against its text model's 128 ("an attention score scale of
0.088 on a tier built for 0.112"). SmolVLM's two heads are both 64, which is
why it never showed. The first text block now replaces a scratch built from a
vision plan.

### Qwen2.5-VL

`qwen2.5vl_merger` is Qwen2-VL's merger behind a different tower: RMSNorms with
no bias, a gated SiLU MLP with biases (v.blk.N.ffn_gate), window attention on
every block but each n_wa_pattern'th (8: blocks 7, 15, 23, 31 see the whole
image), and the merger's norm (v.post_ln) an RMSNorm. The windows are 112 pixels
-- 4x4 merge units -- and the file states only the pattern; llama.cpp's loader
defaults the size, and the container carries both (an optional tail of the
vision record, so every other tower's section is the bytes it was).

The tower runs its patches in transformers' window order: windows row-major over
the merged grid, the merge units inside a window row-major, each unit's 2x2
patches in four consecutive rows (`TowerState.windowOrder`). A windowed block's
rows then attend inside a contiguous range -- on the host the attention over
that range, on a device the whole image's scores with every key outside a row's
window set to -inf before the softmax (`kernels.WindowMaskRows`, the windows
handed over once per image through `nn.VisionWindowDevice`) -- and the merger
groups four consecutive rows and puts the merged rows back into raster order.

★ **THE 3B's MLP IS NOT WHOLE QUANTIZATION BLOCKS.** 3420 is not a multiple of
32, so its contraction cannot be Q8_0 (the published mmproj keeps that one
matrix F16) and the engine reads one packed layout. The converter pads the MLP
to 3424 with units that compute zero -- zero rows and biases on the gate and up,
zero columns on the down (`convert.padFFN`, which SigLIP-so400m's 4304 needed first) -- which is exact.

Dynamic resolution, for both Qwen towers: the grid follows the picture
(`TowerConfig.ResizeTo` is transformers' smart_resize, Python's
round-half-to-even included), resampled with PIL's BICUBIC -- antialiased by
widening its support on the way down -- and PIL's clip to 8 bits after each
pass, as a generated clamp in raw units centred on mid-grey. Against
transformers' Qwen2VLImageProcessor (PIL backend) on a textured 240x160 picture
reduced to 168x112: NMSE 7.4e-6, max |d| 0.014 -- one 8-bit level through the
normalisation, from PIL's rounding between passes, which the float resize does
not do -- against 7.1e-4 for the bilinear resize the other towers take. The
pixel ceiling defaults to the mmproj's own image_size squared (560x560, 400
tokens), the size the tower ran at before; transformers' and llama.cpp's
ceilings are far higher and leave the device's attention scratch quadratic in
a number nothing bounds. quad-and-disc.png is 384x384 and becomes 392x392 under
every one of the three.

The tower against transformers' own (scripts/qwenvlgold.py; the fixture's q/k/v
weights at 0.3, where dropping the rotary moves the tower 16-21% and int8 moves
it 5-9e-4 -- at 0.5 the attention is so sharp the int8 alone moves it 1e-2):

| | NMSE | no 2-D rotary | full attention where windowed | half the window | merged rows left in window order |
|---|---|---|---|---|---|
| synth-qwen2vl | 3.2e-4 | | | | |
| synth-qwen25vl | 5.9e-4 | 2.1e-1 | 3.1e-1 | 2.0e-1 | 3.5e-1 |

and on the devices, the whole tower against the host: CUDA 3.5e-4, Vulkan
9.2e-5, Metal 5.0e-4, with the windows ignored reading 3.1e-1 on each.

★ **A WINDOWED BLOCK ON A DEVICE ALSO RUNS IN QUERY CHUNKS, AND EACH CHUNK
NEEDS ITS OWN ROWS' WINDOWS.** At the largest picture the 3B takes (560x560,
1600 patches, 16 heads) a block's score planes pass the tier's 128 MiB vision
budget, so its attention goes four chunks of 400 queries at a time
(`tier/visionattn.go`), and the mask runs per chunk on a window buffer per
chunk. `TestQwenVLWindowsSurviveQueryChunks`, the whole tower on the device
against the host, which never chunks: 128 chunks, CUDA 1.4e-3 and Vulkan
2.6e-3 (RTX 3050 Ti; 1.5e-3 and 2.6e-3 on the V100) and Metal 1.6e-3 (M4);
windows as wide as the image 8.2e-1, and the first chunk's windows handed to
every chunk 1.9.

Real checkpoints, llama-mtmd-cli's greedy reply teacher-forced
(`TestQwenVLMatchesLlamaCppTeacherForced`):

| checkpoint (Q4_K_M, Q8_0 mmproj) | picture | host | CUDA | Vulkan | Metal (M4) |
|---|---|---|---|---|---|
| Qwen2-VL-2B | quad-and-disc.png | 17/18 | 17/18 | 17/18 | 17/18 |
| Qwen2-VL-2B | the same at 392x392 | 17/18 | 17/18 | 17/18 | 17/18 |
| Qwen2.5-VL-7B (V100) | quad-and-disc.png | 32/32 | 30/32 + 2 ties | 32/32 | |
| Qwen2.5-VL-7B (V100) | the same at 392x392 | 32/33 + 1 tie | 31/33 + 2 ties | 32/33 + 1 tie | |

Every device row places every tower block and every text block. The
392x392 picture is quad-and-disc.png already at transformers' own resize, so
neither engine resamples it and only the models differ. The one Qwen2-VL-2B
position is the first word, below.

★ **QWEN2.5-VL-3B IS HELD TO TRANSFORMERS, NOT TO llama.cpp, BECAUSE
llama.cpp LEAVES TRANSFORMERS AT THE FIRST WORD.** llama-mtmd-cli answers "A
square is divided into four equal parts" on the picture and "A yellow circle
is in the middle of a red, blue, and green square" on the same picture at
392x392 -- two answers from its own two resamplings. The engine says "The" by
3.4-3.7 logits on the host, CUDA and Vulkan, and transformers (bfloat16, on the
M4: its float32 copy is 15 GB) says "The image is a square divided into four
equal quadrants". Teacher-forced through llama.cpp's words the engine then
parts at position 2 by 1.6 -- conditioned on an "A square" the reference never
wrote, which is not evidence about either engine. Against transformers'
(`TestQwenVLRealMatchesTransformers`, the published Q4_K_M and Q8_0 mmproj):

    transformers' image rows   worst logit NMSE 1.24e-2 (amd64) / 1.14e-2 (M4), 1 flip of 12
    the engine's own tower, end to end, worst logit NMSE and the tower's rows:
      host (amd64, M4)                1.26e-2 / 1.13e-2    rows 5.8e-3 / 6.6e-3
      CUDA (V100, every block)        1.18e-2              rows 2.5e-3
      Vulkan (V100, every block)      1.08e-2              rows 6.9e-3
      Metal (M4, every block)         1.19e-2              rows 2.5e-3
    each with the same 1 flip of 12

The flip is the 11th word, " colored" for " quad(rants)". At Q4_K_M against
bfloat16 the quantization alone is 1.2e-2, so this arm cannot see M-RoPE
(sequential positions read 1.9e-2); the 2B's F16 arm is the one that does.

★ **llama.cpp's FIRST WORD ON QWEN2-VL-2B IS NOT TRANSFORMERS'.** It answers "A
yellow circle is placed ...", the engine "The yellow circle is placed ...", and
transformers in float32 on the same checkpoint "The image features a circle
...": the engine's first token is the reference's. The test names it rather
than widening its tie margin.

### Found on the way, not fixed

- **The tower's int8 activations are its precision limit on a real ViT.**
  Qwen2-VL-2B's tower, whose activations reach 1e7, reads 7.7e-2 (NMSE) against
  transformers' float32 tower -- the int8 activations alone move transformers'
  own tower by 8.8e-2 and its Q8_0 weights by 7.2e-3 -- and the end-to-end
  logits 6.4e-3 with one argmax of twelve flipped. llama.cpp's Q8_0 mmproj on
  the CPU quantizes its activations the same way. A float activation path for
  the tower's matmuls is the lever, and is unbuilt.
- **On arm64 two states can decode the same row through different kernels.**
  A packed matvec shape is timed over its first calls (nn.mvPick), so a state
  that decodes while a shape is being timed and one that decodes after it is
  chosen run different kernels: two identical Prefills then one Forward part
  by 0.2-0.7 logits on Llama-3.2-1B, OLMo-2-1B and Qwen2-1.5B on the M4, and
  are bit-identical with tuning off. The seam gate runs with tuning off; the
  size of the difference, larger than reassociation, is unexplained.
- **The tower is still an encode stage, not rows of the one sequence.**
  TowerState.Encode runs the image's patches through the tower's blocks on its
  own loop and hands the merged rows to PrefillMixed: the blocks share the
  container, the pager, the device tier and the block space, but the patches
  do not go through prefill's chunk machinery as rows. The M-RoPE positions do
  ride the rows of the one sequence.

### As it stood in AGENTS.md

★ **qwen2vl WAS ADDED AND IT COST LESS THAN THE LIST SUGGESTS,
BECAUSE ITS TEXT BLOCK IS qwen2's TENSOR FOR TENSOR** -- biased q/k/v, an
unbiased output projection, RMSNorm, SwiGLU, the gpt2/qwen2 tokenizer. The one
text difference is M-RoPE, and **for a text-only prompt M-RoPE and NEOX rope
compute the same rotation**, because the (t, h, w) triple is all the token
index until image rows occupy a span of positions. So the architecture landed
and was verified on its own, before any vision work, with nothing in `engine/model/`
changed at all -- the graph is driven by Flags.

★ **AND WITH AN IMAGE IN THE PROMPT THE TWO DIVERGE, AND M-RoPE IS BUILT AS A
PROPERTY OF THE ROW.** An image row (y, x) whose span starts at rotary position
s turns at (s, s+y, s+x), and the text after it resumes at s + max(H, W), not at
s + H*W -- so from the first image on a row's ROTARY position and its CACHE
position part. `Span.Grid` carries an image span's merged grid, `spanRope` lays
every row of a mixed prompt out on the three axes, and a per-slot `ropeMark`
carries the offset to every later row through decode, `ForwardBatch`, the ragged
step, `StepRuns` and speculation. The table is `nn.Rope.Runs` -- the generated
table kernel once per run -- so NO rotation kernel on any tier learned M-RoPE;
a device takes the host's table on the upload path it already had. The prefix
cache keys an image row on its values AND its (t, h, w) (`spanKey`), because the
same rows under a transposed grid are a different prompt.

Gated against transformers' own class on fixtures written by llama.cpp's own
converter (`TestMRopeMatchesTransformers`: 1e-12 on the host, Vulkan and a batch
slot, ~5e-7 through CUDA's and Metal's f16 prompt tiles, **0.50-1.31 against
sequential image positions**), and on the real Qwen2-VL-2B at F16 against
transformers in f32: **6.6e-12 with it, 1.4e-2 and an argmax flipped without
it**. The earlier sentence that a Qwen2-VL answer did not depend on it was true
of one picture's DESCRIPTION and false of its logits.

`Config.RopeSections` is **PAIRS, not dimensions**, which the file says:
Qwen2-VL-2B ships `[16, 24, 24, 0]` against a head_dim of 128, and
16+24+24 = 64 = 128/2. `configOf` refuses a list that does not sum to NRot/2,
because a short one leaves a tail of every head unrotated -- fluent, and
invisible to every tier, exactly as gemma's NEOX bug was.

Verified against llama.cpp on the FIRST generated token over twelve prompts, so
nothing compounds: **qwen2vl 12/12 identical, against 11/12 for Qwen2-1.5B**,
the already-verified architecture it is modelled on.

★ **qwen2.5vl_merger IS QWEN2-VL's TEXT MODEL AND MERGER BEHIND A DIFFERENT
TOWER**: RMSNorms, a gated SiLU MLP with biases, and WINDOW attention on every
block but each `n_wa_pattern`'th. The tower runs its patches in transformers'
window order, so a windowed block attends inside a contiguous range on the host
and through a -inf mask on a device (`kernels.WindowMaskRows`, the windows
handed over once per image as the call's windowed key runs, `nn.KeyRunDevice`); the merger's rows
go back to raster order. Both Qwen towers take the picture at its own
resolution (`TowerConfig.ResizeTo` is transformers' smart_resize, resampled
with PIL's antialiased BICUBIC and clipped like PIL), capped at the mmproj's
image_size squared. The 3B's 3420-wide MLP is not whole Q8_0 blocks, and the
converter pads it to 3424 with units that compute zero (`convert.padFFN`, which SigLIP-so400m's 4304 needed first).
`docs/engineering-history/model-correctness.md` has the gates and their
violations.

## Qwen3-VL: deepstack rows, a resampled position table and an interleaved M-RoPE

`qwen3vl_merger` is Qwen2-VL's merger (LayerNorm over each patch row, the 2x2
shuffle, linear, GELU, linear) behind a LayerNorm, GELU-tanh tower with a
fused q|k|v (split at conversion, `convert/visiontowers.go`), and three pieces
of its own. Each was gated on `synth-qwen3vl` and `synth-qwen3vlmoe`
(`scripts/qwenvlgold.py`: transformers' own `Qwen3VLForConditionalGeneration`
and `Qwen3VLMoeForConditionalGeneration`, every parameter randomised, written
to GGUF by llama.cpp's converter), with a violation each.

**The position table.** A learned 48x48 table (8x8 in the fixture) is added
after the patch embedding and resampled to the picture's patch grid with
transformers' bilinear, aligned-corners weights, computed in float32 as it
computes them (`posLerp`; the rows are generated axpys). The fixture's 6x10
grid resamples up one axis and down the other. The first build added the
first resampled row to every patch -- `axpyRows` with no index adds one row to
all of them -- and read 1.1e-2 against transformers' tower; split block by
block against transformers' own hidden states it was the entry alone (1e-3)
compounding to 7e-3 by block 3, and after the fix the entry reads 2.7e-5 (the
Q8_0 patch embedding's own).

**Deepstack.** After each tapped block (5, 11 and 17 on the 2B; 0 and 2 in the
fixture) a merger of its own -- the shuffle FIRST, then a LayerNorm over the
shuffled row -- emits one row per picture row, and tap k adds into the
picture's rows of the residual after text block k. The rows leave the tower
behind the merged ones in one buffer, so the image cache and the prefix cache
keep a picture whole (`Span.Deep`, `splitDeep`); in a prefill the runner adds
a span's tap into the chunk's rows after each of the first blocks
(`engine/model/deep.go`), and a device runs those blocks one submission each so
the add lands between them. The mergers are dense-region tensors at the index
of the block they tap: in the block page, at a static page size, a 2x2 merger
three times a tower block would set every vision page's size.

**The interleaved M-RoPE.** The text model is qwen3's (and qwen3moe's) block
whose rotary turns pair i by h when i%3 == 1 and i < 3*sections[1], by w when
i%3 == 2 and i < 3*sections[2], and by t otherwise (`nn.IMRopeRuns`). ggml's
IMROPE sends a pair with i%3 == 0 past 3*sections[0] to a fourth axis where
transformers keeps it on t; no published split reaches one and the converter
refuses a split that would (the fixture's [3, 3, 2] is chosen so both agree).

| gate | synth-qwen3vl | synth-qwen3vlmoe | violation |
|---|---|---|---|
| tower vs transformers at the engine's int8 | 7.8e-4 | 6.8e-4 | no 2-D rotary 3.6e-1, the table unresampled 5.9e-1 |
| deepstack rows | 2.9e-4 | 2.3e-4 | the mergers' LayerNorm skipped 5.7e+1 |
| end to end, logits, 0 argmax flips | 2.2e-3 | 4.9e-3 | taps dropped 2.0e-1 / 2.9e-1, sequential positions 3.0e-1 / 4.5e-1 |
| text model on transformers' rows | 1.7e-12 | 6.2e-13 | contiguous sections 3.3e-3 / 1.0 |
| processor (PIL BICUBIC, patch 16, mean/std 0.5) | 8.6e-12 | -- | bilinear 7.0e-4 |
| every tower block on cuda:0 / vulkan:0 vs host | 3.0e-4 / 8.5e-5 | -- | no rotary in the plan 1.4e-1 |

A Picture span runs end to end at 2.2e-3; a second prompt extending the first
restores 32 positions past the picture's end at 19 with no tower block, and a
fresh State's encode is the image cache's, bit for bit
(`TestQwen3VLPictureCarriesItsTaps`). Paging under a one-page budget,
relocation and zero warm allocations hold for both fixtures
(`TestMRopeKeepsTheFivePrinciples`, `TestWarmImageEncodeDoesNotAllocate`).

The processor's floor is transformers' (`size.shortest_edge` 65536, a 256x256
picture); llama.cpp's is 8 tokens (RULE 7m: the processor is what the weights
saw). A toy tower whose whole ceiling is under that floor takes the Qwen2-VL
floor in its gates, as its golden picture went into the processor unresized.

The fixture also found a layout trap in the oracle pair: transformers 5 saves
a Qwen3-VL mixture's packed experts as [E, out, in] and llama.cpp's converter
transposes the published [E, in, out]; the fixture is rewritten in the
published layout before conversion (`experts_as_published`).

## Qwen3.5's tower, and GLM-4.1V / 4.5V / 4.6V

**Qwen3.5** is Qwen3-VL's tower with no deepstack taps in front of qwen35's
hybrid text model, whose sections are the same interleaved split
(`jlm.Arch.InterleavedMRope` covers ArchQwen3Next: qwen3next carries no
sections and keeps no runs, so its text rows are bit-identical). On
`synth-qwen35vl` (transformers' `Qwen3_5ForConditionalGeneration`, converted
with `--no-mtp`) the text model reads transformers' image rows at 2.0e-11 on
the host, 2.5e-7 on cuda:0 and 6.8e-12 on vulkan:0, contiguous sections
1.0, sequential positions 2.0e-1; the tower 6.1e-4 against transformers'
int8; end to end 3.6e-3 with no argmax flip past a 0.1-logit tie (its
248k-wide head has two logits 0.04 apart at one row, which the tower's int8
turns either way).

**GLM-4.xV**'s tower is Qwen2.5-VL's RMSNorm, gated-SiLU block with no windows
and an entry and head of its own: the patch embedding RMSNormed, a learned
table resampled BICUBIC (PyTorch's grid_sample, align_corners False, border
padding, A = -0.75, its float32 coordinates) and added; after the post-norm a
2x2 convolution (a matrix over the (dy, dx, channel) row, laid out by the
converter), linear, LayerNorm, GELU and a gated MLP. Its processor bounds a
still image's two temporal frames together, so the tower's bounds are per
frame, half the processor's.

| gate | synth-glm4v | synth-glm4vmoe | violation |
|---|---|---|---|
| tower vs transformers at the engine's int8 | 6.8e-4 | 8.8e-4 | no rotary 2.2e-1, no patch RMSNorm 8.5e-1, table bilinear 2.8e-1, unresampled 1.8, merge group transposed 1.4 |
| end to end, logits, 0 flips | 3.5e-4 | 1.7e-3 | sequential positions 7.4e-2 / 2.0e-1 |
| text model on transformers' rows, host / cuda:0 / vulkan:0 | 4.8e-13 / 2.7e-8 / 3.6e-13 | 7.1e-13 / 1.5e-7 / 7.9e-13 | sequential positions 7.6e-2 / 2.0e-1 |
| processor (PIL BICUBIC, factor 28) | 3.0e-12 | -- | bilinear 7.1e-4 |
| every tower block on cuda:0 / vulkan:0 vs host | 5.4e-4 / 3.3e-4 | 5.0e-4 / 2.7e-4 | no rotary in the plan 4.0e-2 |

The first GLM fixture read 2.4e-2 against transformers' int8 tower and the
cause was not the engine: each tower block alone, fed the engine's own input,
matches transformers' int8 arithmetic to 1e-13, and the merge to 2.7e-5 once
the golden quantizes the convolution as the converter lays it out (along the
(dy, dx, channel) row, its input int8) -- the golden had left the 4-D kernel
float. What was left was the fixture amplifying the int8 several times more
than the Qwen ones at the same weights; GLM's fixtures take softer weights
(`scripts/qwenvlgold.py`).

**glm4** (GLM-4-0414, `synth-glm4`, transformers' `Glm4ForCausalLM`): decode,
prefill and the batch 1.2e-11 against transformers; no q/k/v biases 4.3e-1,
no post-attention norm 1.2, no post-FFN norm 1.5, NEOX for NORM 1.8; every
block and the head on cuda:0 / vulkan:0 3.1e-12 / 3.8e-12 decode; paging,
relocation and zero decode/step allocations on host, CUDA and Vulkan. A glm4
file with prediction layers (GLM-OCR) is refused by name.

## Kimi-VL: MoonViT in front of DeepSeek-V3's text model

transformers ships no Kimi-VL class, so the oracle is Moonshot's own
`modeling_kimi_vl.py` and `image_processing_kimi_vl.py`, loaded from the
checkpoint with three shims for names transformers 5 dropped (none of them on
the forward pass), and both GGUFs are llama.cpp's converter's
(`scripts/kimivlgold.py`). The fixture, `synth-kimivl`, is
KimiVLForConditionalGeneration at toy size with every parameter randomised, an
8x8 table resampled to the picture's 6x10 grid, and the prompt rendered by the
checkpoint's own template (its default system turn included).

| gate | synth-kimivl | violation |
|---|---|---|
| tower vs Moonshot's at the engine's int8 | 6.3e-4 | no rotary 4.7e-1, NEOX pairs 5.5e-1, row on the even pairs 4.2e-1, table bilinear 1.8e-1, unresampled 9.2e-1, no per-patch LayerNorm 1.6e-1, merge group transposed 5.1e-1 |
| text model on Moonshot's rows, host / cuda:0 / vulkan:0 | 8.4e-11 / 8.2e-11 / 8.4e-11 | the image's rows reversed 1.1e-2 |
| end to end, logits, 0 flips | 1.3e-3 | -- |
| processor (pad, and a resize past a lowered limit of 64 patches) | 6.0e-12 | bilinear 7.5e-1 |
| every tower block on cuda:0 / vulkan:0 vs host | 8.7e-10 / 1.5e-13 | no rotary in the plan 3.1e-1, NEOX pairs in the plan 3.3e-1 |
| the template around the picture | the processor's ids | -- |

Paging at one frame (31 page-ins), relocation onto cuda:0 and vulkan:0 and
home after the picture, and zero engine allocations on a warm encode and a
decode after the picture all hold against Moonshot's model.

The first fixture's text model, at the 0.2 every other weight takes, flipped
an argmax end to end on the tower's int8 alone (2.3e-2, the text model on
Moonshot's own rows 1.6e-11); its text weights take half that, as GLM's do.
The table at perturb's default scale was invisible beside the patch
embedding (bilinear 7.2e-4 against a clean 3.1e-4); it takes the Qwen
fixtures' x4.

**The real Kimi-VL-A3B-Instruct's tower** (`TestKimiVLRealTowerMatchesMoonshot`,
`scripts/kimivlreal.py`): the whole 16B reference does not fit the laptop, so
its MoonViT and projector are built by Moonshot's code from the first
shard's tensors alone, converted at F16 by llama.cpp's converter, and run on
quad-and-disc.png (28x28 patches, 196 rows of 2048):

| | NMSE |
|---|---|
| against Moonshot's at the engine's arithmetic (Q8_0, int8) | 1.1e-3 |
| against Moonshot's float32 (the int8 alone moves it 3.2e-3) | 3.1e-3 |
| every tower block on cuda:0 / vulkan:0, against Moonshot's | 1.2e-3 / 9.9e-4 |
| no per-patch LayerNorm (the violation) | 14 |

The tower's capacity is what any picture under the token limit pads to,
5108 rows -- four times a prime. The device's query chunking searched only
halvings that divide a tower's rows, found none, and allocated the score planes
whole: the toy tower ran a 4 GB card out of memory ("cuda took 0 of 3 tower
blocks"). `tier.visionChunk` now always chunks over budget: the largest chunk
that is a multiple of 16 and of both tiles and fits, a divisor when one is at
least half that, and otherwise the scratch rounded up to whole chunks (5108
rows at MoonViT's 16 heads: 27 chunks of 192 over 5184 rows).
`tier.TestVisionAttentionAlwaysChunksOverBudget` runs the earlier rule as its
violation, and the Kimi device gate asserts the chunks ran.

## HunyuanVL: XD-RoPE per element, and a merger with newline rows

HunyuanOCR (`HunYuanVLForConditionalGeneration`, `hunyuan_vl` and the
`hunyuanvl` projector) is hunyuan's dense block behind a LayerNorm ViT. The
oracle is transformers' own class and its PIL processor; both GGUFs are
llama.cpp's converter's (`scripts/hunyuanvlgold.py`). The fixture,
`synth-hunyuanvl`, is the class at toy size with every parameter randomised,
an 8x8 table resampled to the picture's 6x10 grid (the table at x4 and the
patch embedding at a tenth, so the table's resampling is load-bearing), and
the prompt rendered by the checkpoint's own template.

**XD-RoPE assigns a coordinate per ELEMENT, not per pair.** transformers'
`recomposition_frequencies` takes cat(freq, freq) -- the whole head -- in
chunks of 2*section[i] elements, chunk i turned by coordinate i mod 4
(sequence, column, row, picture). A NEOX pair's first element is element p
and its second p+head/2, so on a picture's rows the two halves turn by
different angles: x[p] = x0 cos(a) - x1 sin(a), x[p+h] = x0 sin(b) + x1 cos(b).
That is not a rotation and does not keep a head's RMS, which is why k's norm
moves after the rotary too (q's already did, carrying k's weight). The table
is two of NRot floats a row (A then B, `nn.XDRopeRuns`), and every tier runs
it: `RoPESplit` on AVX2, SSE and NEON, `kernels.RoPERowsSplit` in the device
IR, which turns k row-major into a scratch, norms it and copies it into the
cache, contiguous or paged. A text row's coordinates agree, so A is B and the
rotation is NEOX's. llama.cpp's `ggml_rope_multi` assigns a coordinate per
pair (RULE 7m: the engine follows transformers). A zero section still takes
its chunk index, as torch's split hands enumerate one.

A picture keeps its sequence positions -- its rows are not compressed as
M-RoPE's are -- and its grid rows, the newline column included, take
(position, column, row, picture ordinal); the begin and end rows are text.

The tower reads the patches in RASTER order (the processor flattens them
row-major, unlike Qwen-VL's merge groups) and adds the table resampled
bilinear without aligned corners. The merger is an RMSNorm, a 2x2
convolution, GELU, a 1x1 convolution, a newline row after each merged row, a
linear, begin and end rows and an RMSNorm: H*(W+1)+2 rows a picture.

| gate | synth-hunyuanvl | violation |
|---|---|---|
| layout, every row's four coordinates | transformers' get_rope_index, 30 of 30 | -- |
| tower vs transformers at the engine's int8 | 2.6e-4 | aligned corners 2.9e-2, unresampled 5.2e-1, merge group transposed 3.0e-1, no newline rows 3.2e-1 |
| text model on transformers' rows, host / batch slot / cuda:0 / vulkan:0 | 4.4e-13 / 4.4e-13 / 2.0e-8 / 4.2e-13 | coordinates per pair (llama.cpp's) 1, k normed before the rotary 1, sequential positions 7.3e-2 |
| end to end, logits, 0 flips | 5.9e-4 | -- |
| processor (PIL BICUBIC, aligned to 32) | 2.9e-12 | bilinear 5.7e-4 |
| every tower block on cuda:0 / vulkan:0 vs host | 1.9e-4 / 6.6e-6 | -- |
| the template around the picture | the processor's ids | -- |

transformers' torchvision processor parts from its PIL one by 2.9e-7; the
reference HunyuanOCR resizes with PIL.Image.resize, so the gate holds the PIL
one. Paging at one frame (12 page-ins), relocation onto cuda:0 and vulkan:0
and home after the picture, the prefix cache (30 of 30 rows restored; the
same rows under a 9x1 grid in place of 3x5 restore nothing) and zero engine
allocations on a decode after the picture all hold against transformers. The
prefix cache's hashed row name carried three coordinates in a fixed buffer;
the fourth panicked it, and it holds four now.

**The real HunyuanOCR** (`TestQwenVLRealMatchesTransformers`,
`scripts/hunyuanvlreal.py`): float32 transformers against llama.cpp's F16
conversion, quad-and-disc.png at 512^2 (a 16x16 merged grid, 274 rows of
1024), 12 tokens teacher-forced through transformers' own answer:

| | NMSE |
|---|---|
| text model on transformers' rows | 1.5e-12, 0 flips |
| the same rows at sequential positions (the violation) | 1.3e-3, 1 flip |
| the engine's own tower rows against transformers' | 8.4e-4 |
| end to end, host / 24 of 24 text blocks on cuda:0 / on vulkan:0 | 3.2e-4 / 3.2e-4 / 3.2e-4, 0 flips |
| every tower block on cuda:0 / vulkan:0, against the host (`TestHunyuanVLRealTowerOnDevice`) | 2.2e-4 / 1.9e-4 |

The checkpoint's pixel ceiling is 4096^2, a 65536-row tower whose scratch
does not fit a 4 GB card, which declines it on budget; the device gate lowers
the capacity to 1024^2. ggml-org's published mmproj states 2048^2 (the
processor's `size.longest_edge`), which transformers' `max_pixels` key
overrides; llama.cpp's current converter writes 4096^2 as transformers reads
it, and the engine takes what the file states.

`jitllm run -chat -image` on ggml-org's Q8_0 pair reads a rendered 640x200
invoice exactly ("Invoice 4721\nTotal: 98.50 EUR"), 283 prompt rows at 91.5
tok/s and decode at 72.2 on six P-cores. Its first run refused the picture:
`State.Picture` counted the merged units alone (H*W) where the prompt, sized
from the header, held H*(W+1)+2 -- a path no fixture gate reached until
`TestHunyuanVLPictureCountsItsNewlineAndEndRows`.

## minimax-m2: three pieces the engine already ran, on one block

MiniMax-M2 (and M2.1, M2.5: 230B-A10B, 62 blocks, 256 experts top-8) was read
off llama.cpp's minimax-m2.cpp, transformers' MiniMaxM2ForCausalLM, vLLM's
minimax_m2.py and mlx-lm's minimax.py before anything was written. It adds no
kernel and no graph feature: every piece is one the engine already ran on
another architecture.

| piece | where it ran before |
|---|---|
| q and k RMSNormed over the WHOLE projection, before the heads split | OLMoE, OLMo 2 (`FlagQKNormWide`; k's norm n_kv*head_dim wide under GQA) |
| NEOX rotary on 64 of 128 dimensions | glm4moe, phi2, nemotron (`NRot` from rope.dimension_count) |
| sigmoid router, selection-only bias, renormalised | DeepSeek-V3, glm4moe |
| no shared expert, no dense lead, no routed scale | qwen3moe |

The router, four implementations: transformers' `MiniMaxM2TopKRouter` applies
`sigmoid` with no key read and always divides by the top-k sum; vLLM passes
`scoring_func="sigmoid"` and renormalises; llama.cpp's builder takes
expert_gating_func from the file and passes renormalise `true` as a literal;
the published config.json states `scoring_func: "sigmoid"` and
`use_routing_bias: true`. All agree, so the converter sets sigmoid and
renormalisation and refuses a file that states softmax, no renormalisation, a
shared expert, a dense lead or a block without the selection bias.

`synth-minimaxm2` (scripts/moegold.py): MiniMaxM2ForCausalLM, 3 blocks, GQA
4:2 at head_dim 16, partial rotary 8 of 16, 8 experts top-2, F32, the
selection bias drawn at 0.5, MiniMax-M2's own tokenizer (pre-tokenizer
`minimax-m2`), llama.cpp's own converter:

    host vs transformers, decode / prefill at every length / batch   7.856e-13
    device vs host, every block and the head
      cuda:0   decode 6.712e-13, prefill 1.450e-07 (binary16 attention tiles)
      vulkan   decode 6.533e-13, prefill 4.237e-13
      metal    decode 6.543e-13, prefill 1.462e-07   (M4)
    M4 host vs transformers                     7.161e-13
    step across three sessions vs each alone   cuda 3.592e-13, vulkan 2.242e-13, metal 2.535e-13
    batched mixture prefill vs row by row      1.668e-08, 3 grouped blocks
    seam moves with three rows in flight       8.810e-07 after, 3 arms
    one-page paging                            200 page-ins, every logit identical
    TestDecodeDoesNotAllocate                  0 on the host, cuda, vulkan and metal

Every violation lands at 2.8e-01 or above against the 1e-9 bound, on the host
and in the device session alone: softmax for sigmoid 4.3e-01, no selection
bias 9.2e-01 (device 1.0), not renormalised 2.8e-01 (2.9e-01), a per-head q/k
norm (qwen3's reading of the same tensor names) 4.9e-01 (5.2e-01), k's norm
read from q's weight 6.4e-01 (6.6e-01), no q/k norm 1.11 (1.13), NORM rotary
3.2e-01 (3.0e-01).

The real model, MiniMax-M2 Q3_K_M (unsloth's three-part GGUF, 109 GB, converted
to a 140.7 GB container), on the V100 box's host tier against llama.cpp's CPU
path on the same file: 20 of 20 teacher-forced positions agree, and the greedy
chain agrees for all 20 tokens.

The safetensors path is not built for it: the published checkpoint is FP8
with 128x128 block scales (`quantization_config`), which the safetensors
converter does not read for any architecture, so the class name alone would
convert nothing anyone ships. The GGUFs are the way in.

### As it stood in AGENTS.md

★ **THE FLAGSHIPS ARE CHOSEN FROM A SURVEY THAT PRICES EACH ONE
(`docs/design/model-coverage.md` section 9), AND THEIR ARCH CODES COME FROM THE
MIDDLE OF THE GAP, COUNTING UP FROM 44 (`format/jlm/archflagship.go`).**
**minimax-m2** (MiniMax-M2/M2.1/M2.5) is a composition with no new piece:
OLMoE's whole-projection q/k norm (k's n_kv*head_dim wide under GQA), NEOX
rotary on part of the head, and DeepSeek-V3's sigmoid router with its
selection bias, renormalised, no shared expert. The gate and the
renormalisation are literals in every reference and a file stating otherwise is
refused. Gated on `synth-minimaxm2` (`scripts/moegold.py`) in the mixture
families' harness on host, CUDA, Vulkan and Metal, the step, the seam moves,
paging and allocations.

## gemma4: the first model whose layers are not one shape

Gemma 4 (E2B, E4B, 26B-A4B, 31B) is gemma3's block with its global layers at
another attention geometry: head_dim 512 against the sliding layers' 256, and
on the 31B and 26B four kv heads against sixteen. Every tier assumed one head
width per model -- the KV pages, the host attention kernels, the device
scratch, the rotary tables -- so the cost was the shape, not the arithmetic.

The container states both geometries: HeadDim/NKVHead/NRot are the global
layers', HeadDimSWA/NKVHeadSWA/NRotSWA the sliding layers' (a fourth optional
config tail, with Flags2; an older reader refuses it by arch code).

| piece | how it runs |
|---|---|
| per-layer geometry on the host | `Config.HeadDimAt/NKVHeadAt/NRotAt/KVDimAt`, a KV layout per layer (`kvlAt`), pages sized per layer, a second attention kernel set (`nn.AttnSet`, `AttnSetFor`) at the sliding width |
| per-layer geometry on the device | one scratch SET per geometry (`tier/gemma4.go`), swapped in place; `GPU.runsInto` cuts a range where the geometry changes, as it cuts where the device changes, so every submission is one geometry |
| proportional rotary on the global layers | `rope_freqs` (1 for the first quarter of the pairs, then 1e30), global table only |
| the sliding layers' rotary | `ropeSWA` at NRotSWA with its own base; on the device the sliding set holds it in the global slot |
| weightless RMSNorm on v | `Flag2VNorm`: the generated per-head RMSNorm against a ones vector (host), k's head-norm kernel against ones (device) |
| no v_proj on a global layer (`attention_k_eq_v`) | v is k's projection before k's norm (`layer.vFromK`) |
| attention scale 1 | `AttnScale` |
| `layer_scalar` | `RoleLayerOutScale` (300), after the FFN's residual add, on every path |
| the tokenizer | a fourth vocabulary kind, SentencePiece-style BPE (`jlm.VocabBPESPM`, `tok/bpespm.go`) |

The rotary is where the references could be misread: transformers' proportional
rope is `inv_freq = base^(-2i/head_dim)` for the first quarter of the pairs and
ZERO after, paired NEOX over the whole head. That is not partial rotary in
this engine's sense (`NRot`), whose exponent is `2i/NRot`; llama.cpp reaches
the same function through `rope_freqs`, and so does this engine.

`synth-gemma4` (global head 32 with one kv head and v from k, sliding head 16
with two, window 4, layers S,S,F,S,S,F) and `synth-gemma4-kv` (v_proj present
everywhere, two kv heads on both), built by `scripts/gemma4gold.py` from
transformers' own Gemma4ForCausalLM with layer_scalar randomised, written to
GGUF by llama.cpp's converter, F32:

    host vs transformers, decode / prefill at every length / batch
                                               5.898e-12, 7.326e-12
    M4 host                                    5.510e-12, 7.400e-12
    tokenizer vs HF tokenizers                 25 cases, 8438 ids, 32 byte-fallback
    device vs host, 6 blocks and the head
      cuda:0   decode 4.382e-12 / 6.316e-12, prefill 2.791e-07 / 2.470e-07
      vulkan   decode 4.690e-12 / 6.805e-12, prefill 4.717e-12 / 5.201e-12
      metal    decode 4.088e-12 / 6.029e-12, prefill 1.159e-06 / 4.623e-07  (M4)
    step across three sessions vs each alone
      cuda 3.333e-12 / 4.842e-12, vulkan 3.782e-12 / 4.462e-12, metal 3.412e-12 / 4.529e-12
    TestDecodeDoesNotAllocate                  0 on the host, cuda, vulkan and metal

and the relocation, paging, windowed-layer, forward-batch and seam-move gates
of the five principles, each history coming home at its own layer's geometry.

The real 31B (gemma-4-31B-it Q4_K_M, 60 blocks: head_dim 512 with four kv
heads on the global layers, 256 with sixteen on the sliding ones), on the V100
box's host tier against llama.cpp's CPU path on the same file: 20 of 20
teacher-forced positions agree, and the greedy chain agrees for all 20 tokens.

Host violations, against the 1e-9 bound: no v norm 1.2e-01, no layer scalar
4.9e-01, the sliding rotary on the global layers 1.7e-01, no proportional
factors 3.3e-03, a 1/sqrt(head_dim) scale 4.0e-01, NORM rotary 7.0e-01, no
window 1.32, no final softcap 2.7e-05 (a random model's logits sit far under
the cap of 30, where tanh is nearly the identity). In the device session alone:
no v norm 1.1e-01 / 2.9e-01, no layer scalar 4.4e-01; the softcap is left out
there because 2.7e-05 is inside the device's own prefill band.

### The KV relocation gate could not see this fixture's history

`TestKVPagesRelocateWithTheLayer` corrupts the relocated history and demands
the generated tokens move. On synth-gemma4 they did not: a tied head behind a
sqrt(d) embedding scale makes the argmax follow the input token whatever
attention returns. The gate sums the logits now and fails only when tokens
AND logits are unchanged; the clean arm demands both identical.

### Three device bugs the per-geometry runs found

- `SessionRows` counted a row once per geometry run, so a ragged step of 24
  rows read 96; the second run of a token is marked (`ragStep.again`).
- A recording per geometry run outgrew `maxGraphs`, so a long step evicted
  and re-recorded every token and allocated; the bound is two per block on a
  split model (`graphBound`).
- An absent `v_proj` left matvec slot 2 empty and the split tuner refused it
  ("Split=8 does not divide 4 sub-blocks"); a block with v from k skips the
  slot.

### The 26B's block: a dense MLP and a mixture side by side

Every block of the 26B-A4B runs a dense MLP and a 128-expert mixture in
parallel, read off transformers' Gemma4TextDecoderLayer and llama.cpp's
gemma4.cpp, which agree:

    d = post_ffw_norm_1(mlp(ffn_norm(x)))
    e = post_ffw_norm_2(experts(pre_ffw_norm_2(x)))      routed by
          softmax(W_r (rmsnorm(x) * router.scale * n_embd^-1/2)), top-k,
          renormalised, each weight times per_expert_scale[e]
    x = x + post_ffw_norm(d + e)

| piece | how it runs |
|---|---|
| the block | `jlm.Flag2DenseMoE`; `model.denseMoE` on every host path (prefill and the batch row by row, `denseMoERows`), `tier.emitDenseMLP` and the mixture on two inputs of its own on the device, decode and the grouped batch |
| the dense MLP | the shared expert's roles, ungated (`RoleShExpGate/Up/Down`); a mixture block's gate/up/down roles are its banks |
| the four norms | `RoleFFNNorm2`, `RolePostFFNNorm1`, `RolePostFFNNorm2`, and `RoleRouterNorm` -- router.scale over sqrt(n_embd), folded at conversion |
| per_expert_scale | `RoleExpScale` (llama.cpp's ffn_down_exps.scale), one generated multiply on every tier: `MoEGate.ExpScale` in the host router kernel (AVX2, SSE, NEON, gathered by id the way phase D gathers a score) and `kernels.ExpertWeightScale` on the device |
| the fused gate|up bank | `ffn_gate_up_exps` split into two banks at conversion, gate first (`splitGateUpBank`) |

`synth-gemma4-moe` (8 experts, top-2, expert width 32, on the dense
fixture's split geometry):

    host vs transformers, decode / prefill at every length / batch   7.323e-11
    M4 host                                                          4.063e-11
    device vs host, 6 blocks and the head
      cuda:0   decode 1.185e-11, prefill 1.474e-06
      vulkan   decode 1.612e-11, prefill 7.993e-12
      metal    decode 3.638e-11, prefill 1.255e-05  (M4; 8.388e-12 with
               tier.Config.NoFlashPrefill)
    grouped mixture prefill vs row by row   6 grouped blocks, 3.914e-06
    step, three sessions vs each alone      cuda 4.278e-12, vulkan 4.602e-12, metal 3.849e-12
    seam moves with three rows in flight    1.03e-05 after, three arms
    paging, 4/2/1 frames                    173/192/208 page-ins, ids identical
    TestDecodeDoesNotAllocate               0 on the host, cuda, vulkan and metal

The mixture's prefill band is the binary16 attention tiles' -- 1.159e-06 on
the dense fixture on the same M4 -- amplified tenfold: each branch is RMSNormed
to unit size whatever its magnitude, so an error that is small against the
residual is not small against a branch. With the tiles off the same run reads
8.388e-12; the device gate takes 5e-5 for this fixture and says why.

The real 26B-A4B (google_gemma-4-26B-A4B-it Q4_K_M: 128 experts, 8 used, the
fused gate|up bank Q4_K and the down bank Q8_0 at an expert width of 704), on
the V100 box's host tier against llama.cpp's CPU path on the same file: all 12
teacher-forced positions agree and the greedy chain agrees for 15 tokens.

Violations against the 1e-9 host bound, then the device session alone: the
per-expert scale at one 8.9e-01 (device 8.8e-01), the router on the experts'
input 1.34 (1.37), the experts on the dense MLP's input 1.46 (1.38), the dense
MLP's post-norm on the experts 8.95e-01 (8.93e-01), no layer scalar 7.6e-01
(8.2e-01). The router kernel's two faults, the scale taken by selection rank
and the scale dropped, fail the AVX2, SSE and NEON kernels bit-exactly against
`oracle.MoERouter` (52 and 78 comparisons).

### The E-models: per-layer embeddings, KV sharing, a double-wide FFN

E2B and E4B add three things to the dense block, read off transformers'
Gemma4TextModel and llama.cpp's gemma4.cpp, which agree:

| piece | how it runs |
|---|---|
| per-layer embeddings | a second token table NLayer*PLEDim wide, plus a projection of the scaled embedding normed per layer slice, `(prj + tok)/sqrt(2)` (engine/model/ple.go); each block gates its slice in after the FFN: `x += post_norm(W_out(gelu(W_gate x) * pl[li]))`. The inputs are built on the host at the embedding and handed to the device with the residual (`nn.LayerInputDevice`); the device gathers each block's slice (`kernels.PLESlice`) |
| KV sharing | the last `shared_kv_layers` blocks compute q alone and attend to the history of the last earlier block of their own kind (`Config.KVSource`, transformers' store_full_length_kv); they own no cache on either tier, and their k/v tensors are dropped at conversion |
| the double-wide FFN | those blocks' FFN is twice NFFN (`Flag2DoubleFFN`, `Config.NFFNAt`); the device keeps them in a scratch set of their own (geoKey's wide bit beside the sliding one) |

A source and its readers share one history, so they are placed together or
not at all: the device declines a reader whose source it does not hold,
offerRange offers a group only whole, and `fixKVGroups` brings a group split
by a relocation home whole (and reclaim brings it back whole). The seam-moves
gate cuts at the nearest legal seam (`kvLegalCut`).

`synth-gemma4-e` (6 blocks, the last two sharing: block 4 reads block 3's
history, block 5 block 2's; PLEDim 32; the shared blocks' FFN 256 against 128):

    host vs transformers, decode / prefill at every length / batch   5.376e-12
    M4 host                                                          7.394e-12
    device vs host, 6 blocks and the head
      cuda:0   decode 7.640e-12, prefill 3.291e-12
      vulkan   decode 8.337e-12, prefill 3.672e-12
      metal    decode 9.456e-12, prefill 3.552e-12   (M4)
    every placement 1..6 on cuda, decode   3.1e-12 .. 8.1e-12 (n=3, 4 place 2:
                                           the groups of 2 and 3 come home)
    step, three sessions vs each alone     metal 3.687e-13
    seam moves, three rows in flight       1.26e-06 after, three arms

The real checkpoints, Q4_K_M on the V100 box's host, teacher-forced against
llama.cpp (`TestGreedyMatchesLlamaCpp`): E2B 3 of 3 scored positions (llama.cpp
ends the reply at 5 tokens), E4B 14 of 14.

Violations, host then the device session alone: no per-layer embeddings 1.48
(1.45), the sliding shared block reading the first sliding block's history
rather than the last's 2.49e-01 (2.56e-01), the per-layer embedding's
post-norm unweighted 1.45e-01 (1.41e-01).

### The real checkpoints on the device: a wide band, and where it comes from

`jitllm verify` on the V100 reads max|dlogit| 1.2-4 between the device and the
host on E2B and E4B, 1.4-7.7 on the 31B with only two of sixty blocks placed,
and 1.4-13.8 on the 26B -- against the 0.2-0.94 this file records for other
models. `TestPostNormDeviceErrorIsAmplification` (RULE 11c's method, now run
on any model with a post-FFN norm and against `JITLLM_LLAMACPP_CPU`) prices it,
teacher-forced over one 128-token chunk, f32 KV on both arms, max|dlogp|:

| | E2B | 31B | Gemma 3 4B |
|---|---|---|---|
| llama.cpp's own CUDA vs its CPU | 2.79 | 3.00 | -- |
| jitllm host vs llama.cpp CPU | 3.18 | 4.29 | 0.99 |
| jitllm host, f16 KV vs f32 KV (no device at all) | 3.00 | 5.70 | 1.34 |
| jitllm device vs host, every block that fits | 3.57 | 4.69 | 1.66 |
| one block on the device, by depth | 0: 4.55, 6: 2.54, 12: 3.51 | 0: 4.95, 30: 2.93, 54: 3.00, 57: 2.34, 58: 1.43, 59: 0.64 | 0: 1.47, 17: 0.85, 33: 0.10 |

**Gemma 4 is a model that amplifies any perturbation several times more than
Gemma 3**: the host's own KV width moves the 31B by 5.70, more than the whole
device does, and llama.cpp's two backends disagree by 3.0. The device's error
tracks the depth below the seam, and jitllm's device is closer to llama.cpp's
CPU (4.01) than jitllm's own host is (4.29).

Two bounds the gate holds other models to do not hold on the 31B, and are
recorded rather than loosened: the device reads 1.56x llama.cpp's own
CUDA-vs-CPU against a 1.5x bound, and the last block alone reads 0.644 against
the 0.25 band OLMo 2 and EXAONE set (Gemma 3's last block: 0.10). The tail of
the depth profile is uniform rather than stepped -- 2.34, 1.43, 0.64 for blocks
57 (sliding), 58 (sliding) and 59 (global) -- so a sliding block contributes
what the global one does and nothing singles out the global layers' kernels;
and a 5% error bent into the last block's post-FFN norm reads 2.61, four times
the 0.644. Whether 0.644 is what one 5376-wide block's reduction order costs
under a 262144-row softcapped head or a defect smaller than 5% is UNRESOLVED.
E2B cannot place its last block alone (it reads block 13's history), so its
row has no last-block figure; its device-vs-host is 1.28x llama.cpp's.

What does settle the kernels: **`synth-gemma4-wide` runs the real head widths,
256 sliding and 512 global**, where the other fixtures run 16 and 32. Against
transformers 3.768e-11; device against host, decode / prefill:

    cuda:0   1.706e-10 / 9.079e-05   (1.760e-10 with f32 scores)
    vulkan   2.607e-11 / 3.232e-11
    metal    1.243e-10 / 9.828e-12   (M4)

CUDA's 9.079e-05 is its batched scores' binary16 operands (`Stats.AttnMMA`):
Gemma 4's attention scale is 1, so a 512-wide score is unscaled and the
operands' rounding reaches the logits. `tier.WithoutAttnMMA` takes it to
1.760e-10. On the real E2B the same switch moves `verify -prefill 96` only from
2.60 to 2.38, so the model's band, not the operands, is the real-checkpoint
term. The fixture's prefill bound is 2e-4 for that reason.

### Not built

Nothing of Gemma 4's text model is refused now. Its vision and audio towers are
another matter (section 9 of docs/design/model-coverage.md).

### As it stood in AGENTS.md

**gemma4** (31B, 26B-A4B, E2B and E4B) is the first architecture whose layers
are not one shape: its global layers attend at another head width and kv head
count than its sliding ones. The container states both (`HeadDimSWA`,
`NKVHeadSWA`, `NRotSWA`), the host reads geometry per layer (`Config.*At`,
`kvlAt`, `nn.AttnSet`), and the device keeps one scratch set per geometry
with a submission cut wherever it changes (`tier/gemma4.go`). The 26B runs a
dense MLP beside its mixture, each branch with its own norms
(`jlm.Flag2DenseMoE`), and a per-expert weight factor that is a generated
multiply in the router kernel on every tier (`MoEGate.ExpScale`,
`kernels.ExpertWeightScale`). E2B/E4B add per-layer embeddings (built on the
host at the embedding, handed to the device with the residual,
`nn.LayerInputDevice`), KV sharing (`Config.KVSource`: a reader owns no cache
and is placed only together with its source, `fixKVGroups`) and a double-wide
FFN on the shared blocks (a scratch set of its own).

## deepseek32: the lightning indexer as the tail of the latent row

DeepSeek V3.2 is deepseek2 -- MLA, the V3 router, the grouped top-k, the dense
lead -- with DeepSeek Sparse Attention: a lightning indexer that scores every
cached position against each query and lets the attention read only the
`index_topk` best. Read off transformers' `DeepseekV32Indexer`, llama.cpp's
`deepseek32` converter and the published config before anything was written.
Per block and position:

    k_idx  = NEOX-rope(LayerNorm(W_k x, weight, bias, eps 1e-6))   [D]
    q_idx  = NEOX-rope(W_qb . (normed q latent)), per head         [H x D]
    w      = W_proj x * (D*H)^-1/2                                  [H]
    score[t] = sum_h w_h * relu(q_idx,h . k_idx,t)
    keep the top_k positions (ties to the lower position); -inf elsewhere

The rotary covers the first `qk_rope_head_dim` dimensions of each indexer head,
NEOX-paired, from the model's own (YaRN) table. The reference's Hadamard
rotation is not built and need not be: it is orthogonal and applied to q and k
alike, so every dot product survives it, and transformers' own class omits it
for that reason. The reference's FP8 indexer cache is F32 here.

**The "second per-layer key cache" is not a second cache.** The indexer's key
is cached as the tail of the MLA row -- `Config.KVDim` and `LayerPlan.KVRow`
are KVLoraRank + NRot + IdxHeadDim -- so the pages that hold a position's
latent hold its indexer key. Paging, relocation between host and devices, the
prefix cache, device sessions and the batch seam moves carry it without a line
of their own; the attention kernels read the first KVLoraRank + NRot of each
row at the row's stride.

**The host** computes the selection per row (decode, prefill, the batched
path), as a -inf bias row added to the scores after the scale: the per-head
scores are the generated attention-score kernel over the row tail
(`nn.AttnSet` at the indexer's width), the ReLU a generated activation
(`ActReLU`, all three host tiers and the IR), the weighted sum an axpy and the
top-k the generated sampler order (`nn.SampleOrder`). At or below top_k
positions no mask is made and the graph is deepseek2's.

**The device** gathers rather than masks (`kernels/indexer.go`,
`tier/indexer.go`): four basic kernels, each a loop of depth one --
`IdxScoresPaged` (one thread a (row, position), the heads' dots and ReLUs in
one loop), `IdxSelect` (a position's rank is the count of positions scoring
higher, or equal and earlier; rank < k is its slot), `IdxGather` (the kept
rows' latent and rotary key into scratch pages), `IdxDesc` (their
descriptors) -- and then the unchanged MLA paged attention over the gathered
pages. The attention shape is therefore KVLoraRank + NRot wide for every MLA
model, and an indexer model has no paged prefill kernel: a chunk takes the
decode plan, row by row inside one submission. A call whose history is
partly evicted (`pagedstream.go`) is refused at preparation by name, because
the scores read every position through the table.

`synth-deepseek32` (scripts/ds32gold.py): transformers' DeepseekV32ForCausalLM,
3 blocks with a dense lead, MLA at kv_lora 16 / q_lora 32 / rope 8, 8 experts
top-2 in 2 groups with the selection bias randomised, YaRN factor 8, an
indexer of 8 heads x 16 at **top-4 against a twelve-position golden**, so from
the fifth position on the attention reads a subset. DeepSeek-V3.2-Exp's own
tokenizer, llama.cpp's own converter:

    host vs transformers, decode / prefill at every length / batch   6.485e-12
    M4 host vs transformers                                          3.958e-12
    device vs host, every block and the head
      cuda:0   decode 4.265e-12, prefill 4.025e-12
      vulkan   decode 4.983e-12, prefill 2.213e-12
      metal    decode 4.414e-12, prefill 3.687e-12   (M4)
    step across three sessions vs each alone   cuda 2.332e-13, vulkan 1.996e-13, metal 1.680e-13
    seam moves with three rows in flight       2.148e-06 after, 3 arms
    one-page paging                            68 page-ins, ids identical
    ForwardBatch under a one-page budget       NMSE 0 against decode
    TestDecodeDoesNotAllocate                  0 on the host, cuda, vulkan and metal

Every violation lands at 1.5 or above against the 1e-9 bound. On the host:
attention over every position 1.58, a top-k one short 1.92, the indexer key
unnormed 1.76, the indexer rotary on adjacent pairs 1.53, no ReLU 2.07. In the
device session alone: every position 1.76, one short 1.85, unnormed 1.83.

Three things the fixture found on the way:

- **The converter's GGUF carries no gating key**, and transformers'
  `DeepseekV32TopkRouter` hardcodes the sigmoid; the converter sets it for
  `deepseek32` and refuses a file that states softmax.
- **The tokenizer's table row was stale.** DeepSeek-V3's pre-tokenizer is
  hunyuan-dense's three whole-pattern Splits (Hunyuan's tokenizer is DeepSeek
  V3's), and `deepseek-v3`, `hy_v4` and `joyai-llm` -- whose repos serve the
  identical object -- still read them as three llama3 splits, so "12345"
  tokenized digit by digit. `tok/testdata/pre/deepseek-v3.json` now holds the
  row to tokenizer.json.
- **The older indexer keys were overwritten** until the host's KV layout took
  its row width from `Config.KVDim` rather than the attention's
  KVLoraRank + NRot: rows were written KVDim wide at the narrower stride, so
  each new position's write landed on the previous positions' indexer keys.
  Logits matched to the fourth position and parted from the fifth, the first
  the indexer selects at.

No real checkpoint: at Q4_K_M the 685B is 405 GB, past every box here, and its
smaller GGUFs are Q2_K-class, which `Dequantable` lacks.

### As it stood in AGENTS.md

**deepseek32** (DeepSeek V3.2) is deepseek2 with the lightning indexer: a
LayerNorm'd, NEOX-rotated indexer key per position, scored against each query
by sum_h w_h relu(q_h . k), and attention over the IdxTopK best positions only.
The indexer's key is the TAIL OF THE MLA CACHE ROW (`Config.KVDim`,
`LayerPlan.KVRow`), so paging, relocation, the prefix cache and device sessions
carry it with nothing of their own. The host masks the scores with a generated
top-k (`nn.SampleOrder`) and a ReLU kernel (`ActReLU`); the device scores,
selects and gathers the kept rows into scratch pages and runs the MLA paged
attention over them unchanged (`kernels/indexer.go`, `tier/indexer.go`), and
refuses a call over an evicted (streamed) history by name. Gated on
`synth-deepseek32` (`scripts/ds32gold.py`: transformers' own
DeepseekV32ForCausalLM, a top-4 under a twelve-position golden, so the sparse
path is what is compared) on host, CUDA, Vulkan and Metal and every principle
gate; no real checkpoint fits any box (685B).

## gemma3n: four residual streams carried as one

Gemma 3n (E2B, E4B; text) is gemma3's attention with the pieces Gemma 4's
E-models later kept -- per-layer inputs, KV sharing, a weightless v norm, an
attention scale of one -- inside three that nothing else here has. Read off
transformers' `Gemma3nTextModel`/`Gemma3nTextDecoderLayer`, llama.cpp's
`gemma3n.cpp` and its converter before anything was written:

    AltUp    four residual streams. At the embedding stream k = mag(P_k x0, x0);
             each block predicts every stream from all four through a router
             (m = tanh(R . rmsnorm(x0) / n_embd), c = C_pred m, pred_k = x_k +
             sum_j c_kj x_j), runs on pred_0 and corrects the others by what
             it did (x_k = pred_k + (C_corr m' + 1)_k (out - pred_0)); the
             per-layer input is gated from the corrected active stream times a
             learned scale and added to streams 1..3. The head reads the mean
             of stream 0 and each unembedded stream matched to stream 0's
             magnitude.
    LAuReL   a rank-64 branch beside the attention: (a + attn + h +
             norm(L_r L_l h)) / sqrt(2), h the attention-normed row.
    sparsity the first ten FFNs' gates through max(g - (mean + 1.645 sd), 0).

**The streams are one residual, STREAM-MAJOR** (`model/altup.go`): for `rows`
rows, stream k of row r is at (k*rows + r)*n_embd. Stream 0 is then exactly
the residual every existing n_embd kernel reads, the block runs untouched on
it, and the other streams follow it in the same slice -- so a device is handed
the whole residual (`Config.ResidW`, `LayerPlan.ResidW`), a seam move, a
relocation, the step across sessions and paging carry all four streams with no
logic of their own. The two layouts differ only where a chunk is padded, and
the tier re-strides there (`tier.streamRows`); every path that would cut a
chunk by rows refuses an AltUp chunk whole.

**The host** runs AltUp with generated kernels throughout: the router is an
F32 matvec and the tanh a softcap of one; the coefficients are a sum of
transposed rows (the converter transposes `altup_predict_coef` and
`altup_correct_coef`, so a coefficient vector is axpys weighted by the
modalities); the magnitude match and the gaussian top-k are two new row
statistics on the layer norm's passes (`cpu.EmitMagMatch`,
`cpu.EmitGaussTopK`, on AVX2, SSE and NEON); the active stream's scale is an
`ActIdentity` gated product. **The device** runs the same with seven basic IR
kernels (`kernels/altup.go`: the router, the prediction, the correction, the
finish, a row product, LAuReL's join, the gaussian's apply over the layer
norm's two partial passes). The head reads the streams' mean, which the host
takes: an AltUp head is never folded into a block submission, and a step or a
device batch projects its rows on the host after the blocks come home.

`synth-gemma3n` (scripts/gemma3ngold.py): transformers' Gemma3nForCausalLM,
ten blocks in two periods of the 5-block sliding pattern (window 4), four
streams, LAuReL at rank 32, per-layer inputs 32 wide, the gaussian on the
first three FFNs, the last two blocks reading the KV of the last earlier block
of their kind, every coefficient randomised (AltUp initialises them to zero,
the identity). Gemma 3n's own tokenizer, llama.cpp's own converter:

    host vs transformers, decode / prefill at every length / batch   2.819e-11
    M4 host vs transformers                                          2.375e-11
    device vs host, every block and the head
      cuda:0   decode 1.977e-11, prefill 6.999e-07 (binary16 attention tiles)
      vulkan   decode 3.288e-11, prefill 1.166e-11
      metal    decode 1.246e-11, prefill 7.437e-07   (M4)
    step across three sessions vs each alone   cuda 6.7e-12, vulkan 8.3e-12, metal 8.1e-12
    seam moves with three rows in flight       3.6e-06 after (laptop), 1.9e-11 (M4)
    one-frame paging                           90 page-ins, ids identical
    ForwardBatch under a one-page budget       NMSE 0 against decode
    windowed layers                            35 pages released behind the window
                                               (the KV-sharing reader holds none)
    TestDecodeDoesNotAllocate                  0 on the host, cuda, vulkan and metal

Every violation lands well past the 1e-9 bound. On the host: no prediction
8.5e-02, no correction 4.7e-03, no LAuReL 1.34, no gaussian 1.58, the
gaussian on every block 1.22, no correction scale 3.8e-02; in the device
session alone: 9.0e-02, 1.28, 3.7e-02, 1.30.

The real model, Gemma 3n E2B Q4_K_M (unsloth), against llama.cpp's CPU path on
the same file: 18 of 20 teacher-forced positions agree, the other two inside
the tie margin. On a V100 (`jitllm verify -devices cuda:0`), all 30 blocks and
the head on the card: 0 of 32 teacher-forced positions disagree with the host,
max|dlogit| 1.07-2.0, decode 68.8 tok/s against the host's 20.5 (3.36x) -- the
band Gemma 4's E-models sit in, where llama.cpp's own CUDA build reads 2.8-3.0
from its CPU build.

Where the references part (RULE 7m): llama.cpp's gaussian divides the
variance by n-1 (`ggml_sum_rows(...) / (n - 1)`), transformers by n
(`torch.std(unbiased=False)`), as the JAX reference does; the engine follows
the reference. At n = 8192 the cutoff moves by 6e-5 of a deviation. llama.cpp's
magnitude match takes no floor under the mean square, transformers' a 1e-5
one, which the engine keeps.

Found on the way:

- **The windowed-layer gate counted a KV-sharing reader's pages.** Its device
  arm expected every sliding layer to release pages behind the window, and a
  reader holds no history of its own; it counts sources now, which is what
  Gemma 4's E-models would also have needed had their fixture been in the
  window list.
- **A real GGUF stores the router F16.** The converter widens it to F32 (four
  rows; a narrow copy buys nothing), and the coefficients with it.
- **`verify` printed "kernel compilation failed" on a model that ran
  clean.** The batched matvec's split descent (`tier.dot4Split`) tries splits
  from the largest down and lets MatVec refuse one that does not divide the
  row's sub-blocks; it wrote each refusal to LastErr as it went, so LAuReL's
  k=64 projection left "Split=4 does not divide 2 sub-blocks" behind on the
  V100. Only the last refusal is recorded now, and only when no split
  compiles.
- **Spark-X2.5's pre-tokenizer row was three llama3 splits**, the same
  misreading `deepseek-v3`'s had: its third Split is a variant of Hunyuan's (no
  newline tail on the punctuation run, a newline a piece of its own), now
  `pretok.OpSparkSplit`, held to HuggingFace tokenizers on the shared corpus
  (`tok.TestSparkSplitsAsItsTokenizerDoes`).

### Not built

Gemma 3n's vision (MobileNet-V5) and audio (USM conformer) encoders.

### As it stood in AGENTS.md

**gemma3n** (Gemma 3n E2B/E4B, text) is gemma3's attention with Gemma 4's
E-model pieces (per-layer inputs, KV sharing, unweighted v norm, scale one)
inside AltUp's four parallel residual streams, with LAuReL's low-rank branch
joining the attention's residual add and a gaussian top-k on the first FFNs'
gates. The streams are ONE RESIDUAL, STREAM-MAJOR (`Config.ResidW`,
`model/altup.go`): stream 0 is where every existing kernel already reads, the
other streams follow it, and a device or a seam move carries the whole slice,
so relocation, paging, the step and the batch needed no stream logic of their
own. The head reads the streams' mean, taken on the host; an AltUp head is
never folded into a device submission (`tier/altup.go`). llama.cpp's gaussian
divides the variance by n-1 where transformers and the JAX reference divide by
n (RULE 7m): this follows the reference.
## The state-space hybrids: Mamba-2 on the delta rule's state

Built on the user's decision ("support all modern models"), priced in
`docs/design/model-coverage.md` 4c. The one fact that decided the build: ggml's
`ssm_scan` keeps a Mamba-2 head's state as head_dim rows of d_state, which is
the delta rule's `[vHeads][vDim][kDim]`, with B as the key, C as the query, x as
the value and value head h reading group `h / (heads/groups)` exactly as
qwen3next's grouped pairing does. So a Mamba-2 block is `jlm.LayerSSD`, a
recurrent kind beside `LayerLinearAttn`, and everything the hybrids already had
-- `recPair`, `MigrateRec`, the prefix cache's restore point, the ragged step's
run chaining, the one-thread scan form, the spec snapshots -- serves it
unchanged: those all ask `LayerKind.Recurrent()` now, not one kind.

### What is new

| piece | host | device |
|---|---|---|
| the selective update | `cpu.EmitGatedSSD` (AVX2, SSE, arm64): one pass a row, `S = S*decay + (x*dt)*B`, `o = S.C + D*x` | `kernels.GatedDeltaFused` with `DeltaFuse.SSD`: the conv bias and SiLU, dt and the decay, the scan and the skip in one launch, at any rows, lanes and runs |
| dt and the decay | `cpu.EmitSSDGate`: the delta gate's softplus with dt kept as an output | formed inside the scan (`ssdGateOf`, the same series) |
| the convolution | the delta rule's, plus `addBias` | the delta rule's `Conv1dRows`/`Shift` and their run forms; the bias rides the scan |
| the gated norm | `ActMul` then RMSNorm per group, Mamba's order (`ssdNorm`) | `ActMul` then `kernels.GroupNormRows`, a group's own slice of the weight |
| in_proj | split at conversion into z, xBC and dt (`convert.unfuseSSD`); xBC's rows and the conv filters and bias reordered to C, B, x so `deltaGeom` slices it as q, k, v | the same container |
| a block with no FFN | `layer.noFFN`: every entry point ends the block at the first residual | `nn.LayerPlan.NoFFN`: the residual add and `bs.copyRes` home to `bs.x` |

### Two references disagree, and the choice is written down (RULE 7m)

- **The gated norm's groups.** mamba_ssm, llama.cpp and transformers'
  Nemotron-H group it by `ngroups`; transformers' `Mamba2` and
  `GraniteMoeHybrid` normalise all of Inner. They agree at one group, which
  every Granite 4.0-H and every state-spaces Mamba-2 has. Grouped
  (`Config.SSMNormGroups`); the Mamba2 fixture is one group for that reason
  and the grouping is held by the Nemotron-H fixture, whose class groups.
- **The dt clamp.** transformers clamps dt to `time_step_limit` on a prompt
  (the chunked scan) and not on a decode step; llama.cpp never clamps and the
  GGUF does not carry the limit. Not clamped. It is not academic:
  transformers' Nemotron-H sets its limit to `(time_step_min, inf)` = 0.001,
  and the first Nemotron-H fixture parted from it at positions 6 and up by NMSE
  1e-9 to 7e-8 -- exactly the clamp, gone with `time_step_min` set below every
  dt (2.9e-12). The fixtures keep dt clear of every limit.

### synth-mamba2 (Mamba2ForCausalLM)

Three blocks, 8 heads of 16, d_state 16, one group, a biased 4-tap
convolution, no FFN, an untied head; every parameter randomised (A_log over a
decade, D around zero, the dt bias at inverse-softplus of 0.01..0.3), written
by llama.cpp's own converter.

    host vs transformers (decode, prefill at every length, the batch)  4.8e-12
    cuda:0, every block and the head      decode 5.1e-12   prefill 9.7e-13
    vulkan:0 (RTX 3050 Ti)                decode 2.7e-12   prefill 5.1e-13
    a step across three sessions, both pool forms, cuda and vulkan  3.3e-14
    three rows decoding while the seam moves under them  9.5e-13 / 3.0e-12

The violations, each on the host and on both devices, and each far past the
bound: A positive (A_log's sign lost) 2.9, no D skip 2.9, no convolution bias
1.7, no dt bias 1.8, the norm before the gate 0.13, B and C swapped 1.0, and on
the host the convolution window not carried 1.6 and the state not carried
0.87. Paging under one page, relocation to another tier and a batch faulting
its blocks in are bit-exact or inside their bounds; a warm decode token and a
warm step make zero engine allocations on the host, CUDA and Vulkan; every op
is generated (`TestEveryModelRunsGenerated`).

### granitehybrid (Granite 4.0-H: 350M, 1B, Micro, Tiny, Small)

Granite's four scales (`graniteConfig`'s, read again by
`convert.graniteHybridConfig`) over Mamba-2 blocks and attention blocks with no
rotary: llama.cpp's converter writes `rope.scaling.finetuned` false for every
hybrid but Bamba, which is `FlagNoPosEnc`. The head counts are per layer with
zeros on the recurrent ones (`attnLayerCount`). The mixture (Tiny, Small) is
the softmax top-k beside an ungated shared MLP the engine already ran for
granitemoe. Two fixtures, GraniteMoeHybridForCausalLM written with the
published files' own layer-type names ("mamba", "attention", which llama.cpp's
converter reads; transformers saves its remapped ones):

    synth-granitehybrid     host 1.8e-12   cuda decode 6.9e-13 prefill 1.1e-09
                                           vulkan decode 1.5e-12 prefill 2.0e-13
    synth-granitehybridmoe  host 7.5e-12   cuda decode 3.3e-12 prefill 4.7e-09
                                           vulkan decode 3.6e-12 prefill 3.3e-12

(a prompt's chunk attends through CUDA's binary16 tiles, hence the prefill's
1e-9.) Granite's violations beside the Mamba-2 ones: rotary on the attention
block 0.054 / 0.037, no residual scale 1.4, no logit scale 6.3, the score at
1/sqrt(head_dim) 0.14 / 0.058.

The principles, both fixtures: a one-page budget gives the resident logits bit
for bit through decode, prefill and the batch (76 and 281 page-ins against 4
and 36; `TestSSMFamiliesPageWithTheSameAnswer`, since a mixture's expert pages
are a second page size the frame-count gate cannot hold); a block relocates to
another tier and back, and the dense fixture's KV pages with it; three sessions
stepped together on CUDA and Vulkan match each alone at 5.2e-14 / 5.8e-14; a
warm token and step allocate nothing; every op is generated.

Real checkpoints, teacher-forced against llama.cpp's own greedy tokens past
EOS (`TestSSMRealModelsTeacherForced`, three prompts of 32):

    granite-4.0-h-350m Q8_0     96 of 96 against llama.cpp CUDA, 96 of 96 against its CPU
    granite-4.0-h-350m Q4_K_M   90 of 96 (CUDA), 88 of 96 (CPU)
    granite-4.0-h-1b Q4_K_M     91 of 96, every disagreement a tie (0.06 to 0.48)

The Q4_K_M disagreements are quantization, not the graph: the one confident
one (a 1.05 margin) and the two near the 0.5 tie band move between llama.cpp's
own CUDA and CPU arms, which part from each other on the same prompts, and the
Q8_0 file agrees everywhere -- which is why the gate's default list holds the
350M at Q8_0.

### nemotron_h (Nemotron-H 4B/8B, Nemotron Nano 2 9B/12B)

Every source layer is ONE mixer under its own RMSNorm: Mamba-2 (M), attention
with no rotary (*) or an ungated squared-ReLU MLP (-). llama.cpp's builder runs
them one at a time as `x = x + mixer(norm(x))`; a mixer followed by an MLP is
exactly one block of a mixer and an FFN, the MLP's norm being the FFN's. So
`convert/nemotronh.go` merges each mixer with the MLP after it at conversion
-- block indices renumbered, the MLP's `attn_norm` written as `ffn_norm` -- and
a mixer followed by another mixer becomes a block with no FFN. Every published
pattern opens with a mixer and puts every MLP after one; an MLP after an MLP is
refused by name. The prediction block and Nemotron 3 Super's latent mixture
(moe_latent_size) are refused by name.

The fixture's pattern is "M-M*-MM-" -- a mixer with its MLP, a mixer alone
before another, the attention with its MLP, two Mamba-2 layers in a row -- with
two groups, so the gated norm is grouped (the class's Zamba2RMSNormGated) and a
value head reads its own group's B and C. Two things in the way of writing it
with llama.cpp's converter, both this transformers' NemotronHConfig: it fills
in the mixture's defaults (num_experts_per_tok) for a dense model, which the
converter takes as nemotron_h_moe, and it turns the published
`hybrid_override_pattern` string into a list of names in which the converter
finds no dense MLP. A dense published config states neither; `ssmgold.py`'s
shim hands the converter exactly that.

    synth-nemotronh   host 2.9e-12
                      cuda decode 1.1e-12 prefill 3.0e-09
                      vulkan decode 9.5e-13 prefill 4.0e-13

Its violations beside the Mamba-2 ones: the gated norm over the whole of Inner
0.97, the MLP reading its mixer's norm 0.75, SiLU for the squared ReLU 1.6, and
on the host the value heads paired with key groups tiled 0.40 (a tiled Mamba-2
is declined on the device by name; no Mamba-2 is one).

The principles: paging bit for bit at one page (and under the frame gate,
which now counts a budget's bytes rather than assuming every page is
PageSize -- the mixer-only blocks are smaller, so four frames' bytes hold all
five blocks), relocation of a block and of the attention block's KV pages,
three sessions stepped together on CUDA and Vulkan at 1.3e-13 / 1.8e-13, zero
allocations a warm token and step, every op generated.

Real checkpoint, teacher-forced against llama.cpp's CPU arm:

    NVIDIA-Nemotron-Nano-9B-v2 Q4_K_M   95 of 96, the one a tie

It converts to a 7.8 GB container and is run under `scripts/cap 24G`
(`JITLLM_SSM_REAL` names it alone).

**Nemotron 3 (nemotron_h_moe) is the same arch with the MLP a mixture.** Its
experts are ungated -- up and down, no gate bank, down(relu2(up(h))) -- under
DeepSeek-V3's router (NemotronHTopkRouter: sigmoid scores, a selection-only
bias, renormalised, routed_scaling_factor) beside one ungated shared expert.
The class hardcodes the sigmoid and the GGUF carries no gating key, so the
converter sets it; one routing group is no grouping (llama.cpp never groups,
transformers does, so more than one is refused until that divergence is
chosen). The engine reads both absences off the tensors: a mixture block with
no gate bank runs the activation alone on every path (decode's batched and
per-expert arms, a prompt's expert-major pass, the shared expert), and a
mixture model's block with neither a router nor an FFN is its mixer alone
(`Config.ffnAbsent`, so `MoEAt` is false there). On the device the plan's
`UngatedFFN` now covers the mixture too: no gate slot, `kernels.Act` for the
routed and the shared activation, the grouped step's ungated arm, and a
streamed bank that sends up and down sheets alone.

The fixture is "ME*EMME" (four blocks: Mamba-2 with a mixture, the attention
with one, Mamba-2 alone, Mamba-2 with one), eight experts, two routed, scale
2.5, the selection bias randomised:

    synth-nemotronhmoe   host 3.7e-12 (M4 arm64 3.9e-12)
                         cuda decode 2.4e-12 prefill 5.9e-09
                         vulkan decode 2.7e-12 prefill 1.1e-12
                         metal (M4) decode 1.6e-12 prefill 7.8e-09
                         step, three sessions: cuda 9.3e-14, vulkan 1.1e-13,
                         every mixture block grouped

Violations beside the dense set: softmax for the sigmoid 1.5 to 1.7, no
selection bias 1.9 to 2.0, the routed weights not renormalised 0.30 to 0.36,
no routed scale 0.99 to 1.0, no shared expert 1.0. The principles' gates all
run it (it is a principle fixture): paging at one frame and one page, a batch
faulting its blocks in, relocation with the KV pages, the seam moving under
three rows, zero allocations on the host, CUDA, Vulkan and Metal, every op
generated. `TestUngatedStreamedBankMatchesTheResidentOne` holds a streamed
bank to the resident one bit for bit on the fixture's weights at Q8_0 (a float
bank has no packed layout to stream, and is declined by name), and fails at
the first logit with up's sheets left unsent.

### falcon-h1 (Falcon-H1 0.5B to 34B, Falcon-H1R)

Every block runs attention (NEOX rotary, GQA) AND a Mamba-2 mixer on the same
normed input, sums the two and adds the sum to the residual, then a SwiGLU FFN
(llama.cpp's falcon-h1.cpp, transformers' FalconH1DecoderLayer). That is a
layer kind of its own, `jlm.LayerSSDAttn`, which is both `Attends()` and
`Recurrent()`: it keeps a KV history and a recurrent state, and every consumer
of either asks the predicate rather than a kind. The mixer's x|B|C projection
cannot sit in wq, which is the attention's query, so the block carries it in
`layer.ssmIn` (`nn.SSMWeights.In` on the device, slot 29). The muP multipliers
-- nine of them, five on in_proj's segments alone -- are folded into the
weights by llama.cpp's converter; the GGUF carries none, and the fixture sets
every one far from one so a dropped fold is a wrong logit.

Every published config has the gate before the norm (`mamba_norm_before_gate`
false), which is llama.cpp's fixed order; 0.5B has no gated norm at all and
34B's is grouped by two.

    synth-falconh1 (grouped norm)   host 8.1e-12
                                    cuda decode 5.1e-12  prefill 1.6e-07
                                    vulkan decode 3.9e-12  prefill 1.1e-12
                                    Iris Xe decode 4.3e-12  prefill 1.4e-12
    synth-falconh1-nonorm           host 3.7e-12
                                    cuda decode 5.5e-12  prefill 1.7e-06
                                    vulkan decode 9.1e-12  prefill 9.7e-12

CUDA's prefill reads 1e-7 to 1e-6 where its decode reads 1e-12: the chunk's
attention scores go through binary16 tiles, and the multipliers make this
fixture's scores large. Vulkan's prefill, which does not, is exact.

Violations, host and device: NORM rotary for NEOX 1.2, the attention half
dropped 2.1 to 2.5, the mixer half dropped 0.96 to 2.5, beside the Mamba-2
set. Paging (frames and one page), relocation with the KV pages, a batch
faulting its blocks in, three sessions stepped together (3.7e-14 to 6.4e-14 on
CUDA, Vulkan and the Iris Xe; 5.8e-13 to 1.0e-12 on the scan's one-thread
form), the seam moving under three rows, zero allocations, generated code.

Real checkpoints, teacher-forced against llama.cpp:

    Falcon-H1-0.5B-Instruct Q4_K_M   94 of 96, both disagreements ties (0.026, 0.040)
    Falcon-H1-1.5B-Instruct Q8_0     94 of 96, both ties (0.074, 0.120)

The 1.5B at Q4_K_M reads 90 and 87 of 96 against llama.cpp's CUDA and CPU arms
with three confident disagreements -- and those two arms part from each other
on the same prompts, so the Q8_0 file is the one the gate holds.
    the same on cuda:0, every block  `jitllm verify` 0 of 16 positions disagree

### lfm2 (LFM2 350M to 2.6B)

LFM2's recurrent block is a gated short convolution: in_proj to B, C and x,
B*x through a depthwise causal convolution of `shortconv.l_cache` taps over
every channel of the residual, C times that, out_proj -- no bias, no
activation, and no state beside the convolution's window. In the delta rule's
terms it is a convolution of Chans = NEmbd channels with no heads, so it is
`jlm.LayerShortConv`, a fourth recurrent kind, and the window rides the same
`recPair`, migration, restore point and ragged-step run chaining as every
other recurrence. The converter splits in_proj into B, C and x
(`convert.unfuseLFM2`, three byte ranges) so each is one matvec on both tiers,
in the slots the mixed projection and the two gates already had; the two
products are the gated multiply with the identity as its activation
(`kernels.ActIdentity`, which Gemma 3n brought to every tier). Its attention
blocks carry a q/k RMSNorm per head and NEOX rotary, and its final norm is
named `token_embd_norm` in the GGUF (llama.cpp's "fix for wrong tensor name",
`convert.retarget`).

    synth-lfm2 (Lfm2ForCausalLM: 3 conv blocks, 1 attention, tied)
      host vs transformers          3.8e-12
      cuda decode 4.0e-12  prefill 4.2e-09
      vulkan decode 1.7e-12  prefill 3.1e-13

Violations on the host and both devices: B and C exchanged 2.8-3.1, x and C
exchanged 2.2-2.4, the window's taps reversed 2.5-2.7, NORM rotary 0.13, no
q/k norm 0.18, and on the host the window not carried 2.0. Real checkpoints,
teacher-forced against llama.cpp: LFM2-350M Q4_K_M 93 of 96 and LFM2-1.2B
Q4_K_M 93 of 96, every disagreement a tie under 0.32.

**lfm2moe** (LFM2-8B-A1B, LFM2-24B-A2B) is the same blocks with a sigmoid
top-k after a dense lead, renormalised, its selection-only expert bias read off
`exp_probs_b` -- every piece a mixture key the converter already reads, so it
converts as `lfm2` with no engine change. transformers' class keeps the bias
as a BUFFER, which a fixture that randomises parameters leaves at zero (the
DeepSeek fixture's lesson); `ssmgold.py` randomises it by name. Its routed
scale stays 1, as every published config states and llama.cpp's converter
writes no key for.

    synth-lfm2moe   host 4.2e-12   cuda decode 2.2e-12  prefill 7.9e-09
                                   vulkan decode 2.1e-12  prefill 1.5e-12
    violations      no selection bias 0.49-0.56, softmax for the sigmoid 0.56-0.58
    LFM2-8B-A1B Q4_K_M teacher-forced: 93 of 96, every disagreement a tie under 0.37

Both run the principles: paging at one page (229 page-ins for the mixture,
every logit identical), relocation with the KV pages, the seam moving under
three rows, three sessions stepped together on CUDA and Vulkan in both pool
forms (2.3e-13 / 2.9e-13) and on the Iris Xe, zero allocations a warm token and
step, generated code. A stateless recurrence needed four guards in the device's
recurrent pool -- a buffer, a copy and a zeroing of zero bytes are each an
error on some backend -- and the ragged step builds no scan for it.


### mamba and jamba (Mamba-1, FalconMamba, Jamba)

Mamba-1's selective scan is Mamba-2's with every channel a head of one row:
`jlm.LayerMamba1`, in the delta rule's terms Inner value heads of width one
over one shared key head of StateSize (B the key, C the query, x the value),
and the decay a VECTOR over the state, exp(dt*A) with A one row per channel,
where Mamba-2's is one scalar a head. dt does not come off in_proj: x is
convolved first, and x_proj projects the convolved x to a rank-R bottleneck,
B and C; dt_proj takes the bottleneck back up to the channels. So nothing
between the convolution and the scan can be fused the way Mamba-2's is.

What it took, besides the layer kind:

    the scan     cpu.EmitSelScan on AVX2, NEON and SSE (the decay computed in
                 the pass that updates the row, so no Rows*n plane of decays
                 is written), oracle.SelScan, nn.JIT.SelScan fanned out over
                 channels; on the device kernels.GatedDeltaFused with
                 DeltaFuse.Mamba1 -- the SSD scan's loop with B, C and x read
                 from buffers of their own and the decay per lane
    the device   kernels.BiasAct (the convolution's bias and SiLU, before
                 x_proj reads it), dt's quantization over the rank, the three
                 norms through HeadNormRows, and the four matrices on KDA's
                 slots 18..21 (nn.SSMWeights: x_proj's dt rows in FA, dt_proj
                 in FB, its B and C rows in GA and GB)
    the format   x_proj split at conversion into RoleSSMXDt, RoleSSMXB and
                 RoleSSMXC, dt_proj as RoleSSMDtProj, Jamba's three RMSNorm
                 weights as RoleSSMDtNorm/BNorm/CNorm (roles 430..436); in_proj
                 split into x (RoleAttnQKV) and z (RoleAttnGate). The rank is
                 read off x_proj; the container states none.

FalconMamba (arch "mamba", ssm.dt_b_c_rms) RMS-norms dt, B and C with no
weight. The converter writes those norms as weights of ones, so the engine
runs them exactly as Jamba's weighted ones. Their epsilon is a RULE 7m
divergence: transformers' FalconMamba uses mixer_rms_eps (1e-6 by default and
on falcon-mamba-7b), which the GGUF does not carry, and llama.cpp uses the
layer norm's epsilon (1e-5). The engine follows the file, i.e. llama.cpp; the
fixture sets the two equal, and with transformers' default the fixture reads
1.05e-9 against it -- just past the 1e-9 bound, which is the size of the
divergence.

Jamba's blocks are Mamba-1 or attention (no rotary), each followed by a
SwiGLU MLP or a softmax top-k mixture that is not renormalised
(JambaSparseMoeBlock; llama.cpp passes false as a literal). Which blocks are
mixtures is read off the tensors: a block of a mixture model with no router is
not a mixture (`Config.noRouter`, which Nemotron 3's mixer-only blocks share).
The expert width is read off the first block that has a bank, since Jamba's
block 0 is dense.

The fixtures (scripts/ssmgold.py, each the family's own transformers class,
every tensor F32, dt rank 32, Inner 128, state 16):

    synth-mamba1        host 3.0e-11   cuda 6.4e-11 / 6.4e-11
                                       vulkan 4.1e-11 / 4.1e-11
    synth-falconmamba   host 2.4e-10   cuda 3.5e-11 / 1.9e-11
                                       vulkan 1.6e-10 / 1.9e-11
    synth-jamba         host 1.3e-10   cuda 1.1e-10 / 1.6e-11
                                       vulkan 2.0e-10 / 1.5e-11
                        (device decode / prefill, every block and the head
                        placed, against the host)

These sit a decade or two above the Mamba-2 fixtures' 1e-12: the decay is an
exp per state element, and the host's degree-5 exp and the device's native
one differ in the last bits. Three sessions stepped together on CUDA read
2.5e-14 (mamba1) and 1.6e-13 (jamba, and its mixture's 16 grouped blocks).

Violations, host and device: A's sign 2.2-2.6, no D skip 1.3-2.5, no
convolution bias 2.0-2.6, no dt bias 0.17-0.51, B and C exchanged 2.1-2.5, one
decay rate a channel (Mamba-2's form) 1.3-2.8, the dt/B/C norms dropped
1.0-2.5 (FalconMamba, Jamba), Jamba's rotary 0.13 and its routed weights
renormalised 0.08-0.09. All three are principle fixtures: paging at one frame
and one page, a batch faulting its blocks in, relocation (and the KV pages on
Jamba), the seam moving under three rows (5.7e-12 / 6.5e-12 after the first
move), zero allocations a warm token and step on the host and CUDA, every op
generated. The kernel gate holds output and state at 1e-10 on AVX2 and the
M4's NEON and 1e-9 on SSE, and runs the oracle against the skip zeroed and a
scalar decay.

A dt rank that is not a multiple of 32 is declined on the device by name:
dt_proj reads a quantized activation of rank width; mamba-130m's rank is 48.

Real checkpoint, teacher-forced against llama.cpp (jamba-reasoning-3b
Q4_K_M, dense, 26 Mamba-1 blocks and 2 attention ones):

    "The capital of France is"  32 of 32 against both llama.cpp arms
    "def fibonacci(n):"         32 of 32 against both
    "Once upon a time"          24 of 32 against llama.cpp's CPU arm,
                                29 of 32 against its CUDA arm

llama.cpp's own two arms part on the story prompt from its second token
("was" against "were"), and jitllm's host and CUDA tiers agree with each other
there at max|dlogit| 0.38 (`jitllm verify`, inside the reduction band).

**Placed by transformers, and neither engine is wrong: the story prompt at
Q4_K_M is 8-bit activation noise amplified by 26 recurrences, in both
engines.** `scripts/ggufref.py` runs transformers' own JambaForCausalLM on a
GGUF's dequantized weights (gguf-py's name map, A_log and the conv squeeze
undone), so quantization is not a variable; `TestTeacherForcedAgainstReference`
(-tags jitllmbench) holds jitllm to it position by position. Along llama.cpp's
CUDA chain, V100 box, host:

    the float checkpoint as an F32 GGUF   every layer's residual 2.5e-13 to 2e-12,
                                          pos 0 logits identical to 4 digits
    the F16 GGUF, against transformers    33 of 33, worst logit NMSE 4.1e-10
      on its own F16 weights
    the Q4_K_M GGUF, against transformers 20 of 33, 6 confident disagreements,
      on its own Q4_K_M weights           NMSE 2.9e-03 at pos 0 rising to 1.06
    llama.cpp's CUDA chain against it     19 of 32; at position 17 the reference
                                          gives "the" by 4.2, where llama.cpp
                                          took "Print" and jitllm "print"

jitllm's f32 activations reproduce the reference exactly (F32 and F16 weights); its Q4_K matvecs,
like llama.cpp's, quantize the activation to 8 bits, and a hybrid of 26 Mamba
blocks carries that error forward in its state. Both engines land about as
far from the exact answer, in different directions, and llama.cpp's two arms
differ by the same mechanism. The two "confident" disagreements were never
anyone's bug. (The F16 GGUF itself is not the float model: bf16 rounds to
f16 there, embedding NMSE 7e-6, and the float model's own answers differ from
it by up to 0.5 NMSE -- which is why the reference must run the file's
weights.)

So the gate holds Jamba at F16, where llama.cpp is an exact oracle: its CPU
and CUDA arms give identical chains, transformers on the F16 weights takes
every token of that chain on the story prompt, and jitllm agrees with
llama.cpp at all 96 positions of the three prompts
(`TestSSMRealModelsTeacherForced`, 96 of 96). At Q4_K_M the same gate reads 88 of 96 with six "confident"
disagreements on the story prompt and 32 of 32 on the other two.

Plain Mamba and FalconMamba against llama.cpp (V100 box, host, the same three
prompts):

    mamba-130m-hf Q8_0          92 of 96, the other four ties (0.07-0.24)
    mamba-130m-hf F16           95 of 96, one 0.000 tie
    falcon-mamba-7B Q4_K_M      94 of 96, two ties (0.011, 0.073)
    falcon-mamba-7B F16         96 of 96
    Mamba-Codestral-7B Q4_K_M   94 of 96, two ties (0.049, 0.037) -- plain
                                Mamba-2, the first real checkpoint of it here

mamba-130m's dt rank is 48, so its blocks stay on the host on a device (the
decline above); the gate runs the host.

### The M4, the Iris Xe and the V100

Every block and the head placed, against the host, decode / prefill:

    metal (M4)          mamba2 4.6e-12 / 6.0e-13   granitehybrid 1.1e-12 / 3.9e-09
                        granitehybridmoe 3.8e-12 / 1.2e-08   nemotronh 8.9e-13 / 3.9e-09
                        falconh1 3.6e-12 / 3.1e-07   falconh1-nonorm 3.0e-12 / 9.0e-07
                        lfm2 2.8e-12 / 1.1e-08   lfm2moe 1.9e-12 / 1.2e-08
                        nemotronhmoe 1.6e-12 / 7.8e-09
    V100 (cuda:0, sm_70) falconh1 5.1e-12 / 7.6e-13   falconh1-nonorm 5.5e-12 / 8.3e-12
                        lfm2 4.0e-12 / 6.3e-13   lfm2moe 2.2e-12 / 1.2e-12
    V100 (vulkan:0)     falconh1 3.9e-12 / 1.1e-12   falconh1-nonorm 9.1e-12 / 9.7e-12
                        lfm2 1.7e-12 / 3.1e-13   lfm2moe 2.1e-12 / 1.5e-12
    vulkan:1 (Iris Xe)  mamba2 3.3e-12 / 6.6e-13   granitehybrid 1.6e-12 / 1.7e-13
                        granitehybridmoe 4.8e-12 / 4.9e-12   nemotronh 1.2e-12 / 3.0e-13

The M4 runs the transformers gate, the violations and the principles on the
arm64 host too (`EmitGatedSSD`'s NEON form against the oracle, then paging,
relocation, a batch faulting its blocks in, zero allocations and generated
code on every fixture); three sessions stepped together read 5.4e-14 to
2.8e-13 on Metal and 2.8e-14 to 1.2e-13 on the Iris Xe, and the scan's
one-thread form (`JITLLM_SCALAR_LINEAR_ROWS`) 2.3e-13 to 1.1e-12 on CUDA and
Vulkan.

### As it stood in AGENTS.md

★ **A MAMBA-2 BLOCK IS THE DELTA RULE'S STATE WITH THE KEY DOT REMOVED, SO
THE STATE-SPACE FAMILIES ARE A LAYER KIND (`jlm.LayerSSD`), NOT A SECOND
RECURRENCE.** B is the key, C the query, x the value, dt the gate; the scan is
`cpu.EmitGatedSSD` and `kernels.GatedDeltaFused`'s SSD form, and everything a
hybrid already had (`recPair`, `MigrateRec`, the restore point, the ragged
step) asks `LayerKind.Recurrent()`. Two RULE 7m choices are written down: the
gated norm is GROUPED (mamba_ssm and llama.cpp; transformers' Mamba2 and
Granite classes are not, and agree at one group) and dt is NOT clamped
(llama.cpp; transformers clamps on a prompt only). Gated against transformers
by `scripts/ssmgold.py` in `model.TestSSMFamiliesMatchTransformers`;
`docs/engineering-history/model-correctness.md` has the measurements.

## Vision as blocks: the tower on the one runner

`docs/design/vision-as-blocks.md` is the design and AGENTS.md ("THERE IS ONE
MODEL") what runs. This is what was measured moving every tower family onto it.

**The host's outputs did not move a bit.** Every family's encode of one fixed
picture, dumped by a binary built from the tree before the move and by one
after it, compared byte for byte:

    smolvlm  llava  gemma3  internvl  qwen2vl  qwen25vl  minicpmv  janus
    BIT-IDENTICAL, all eight

That is the claim the runner had to meet before anything else: the tower's
blocks run in prefill's own block body (`hostRows`) on the text State's JIT,
the attention over a picture's rows is the tiled kernel the tower always ran,
and nothing about the arithmetic changed -- only who calls it. The device
gates hold their old bounds (table below).

**A warm encode allocates nothing, every family.** The tower's loop handed the
pool a closure per region; the runner's regions are method values over fields
(`rowRegion`: norms, bias and table rows, the attention's heads, gemma3's pool,
the resampler's keys and cross-attention), as decode's are:

    the runner's first cut, still handing the pool closures
                            324 allocations a warm encode (SmolVLM)
    now                     0.0 -- smolvlm, llava, gemma3, internvl, qwen2vl,
                            qwen25vl, minicpmv, janus (TestWarmImageEncodeDoesNotAllocate)

**The prefix cache sees past a picture.** A picture is named before a block
runs (`ImageKey`: sha256 over the family's preprocessing configuration, the grid
and the pixels) and each of its rows by that name, its index and, on M-RoPE, its
(t, h, w):

    SmolVLM two-turn chat, turn 1 82 positions (picture at 6..70)
      turn 2               restored 80 -- every whole chunk of turn 1 -- with no
                           tower block and no image-cache lookup; the 8 tokens
                           after it identical to a cold run
      another picture      restored 0 past its start
      the rows unnamed     restored 0 past the picture (the old gap)
    hybrid (synth-qwen35-hybrid-mtp with synth-qwen2vl's tower)
                           restored 12 of 12 at its one exact end, no tower
                           block; unnamed rows restored 0
    jitllm run -kv-cache, SmolVLM, a second process
                           82 of 82 positions reused; prompt with the picture
                           2300 ms -> 17 ms

A picture restored whole is never encoded: `Span.Picture` carries the pixels and
the prefill encodes only a picture whose rows reach past the restore.

**An encode is a run of the step.** A picture encoded in `StepRuns` beside two
sessions decoding -- in one step, and five blocks a step -- gives its rows bit
for bit and leaves both sessions' logits bit for bit, on the host, CUDA and
Vulkan (`TestImageEncodeStepsBesideDecode`). Relocated in the middle of an encode
under a budget of two vision pages, the rows are the bits of the same placement
from the start (`TestTowerEncodeResumesAcrossARelocation`, CUDA and Vulkan).

**The processors' resize is generated code.** Pillow's and torchvision's passes
are integer dots over 8-bit samples (`cpu.EmitResample`; AVX2, SSE, NEON): exact
against the formula at every width to three vectors and a few, one to nine taps
and precisions 7, 13 and 22, and the pixel gates read zero differing values
against transformers' processors as before (gemma3, InternVL, Qwen-VL, MiniCPM-V,
Janus-Pro). The two PIL BICUBIC resamplers are one: Qwen-VL's float
`pilBicubicTaps` path is gone, and Qwen-VL takes Pillow's fixed point, 8 bits
after each pass as Pillow rounds:

    TestQwenPreprocessMatchesTheProcessor   before NMSE 7.357e-06, max|d| 0.0143
                                            after  NMSE 2.986e-12, max|d| 0.0000
                                            (bound tightened 1e-4 -> 1e-10)
    Qwen2-VL-2B f16 end to end against transformers (bf16), host
                                            worst logit NMSE 6.446e-03 -> 3.194e-03

**The SSE tier runs every tower** (`TestEveryTowerRunsSSE`, the tier forced
before the model opens, no AVX2 kernel mapped): the integer resamplers give the
AVX2 tier's bits, and against it --

    family    after block 0   whole tower
    smolvlm   1.7e-06         5.9e-04
    llava     2.6e-07         1.8e-03
    gemma3    7.5e-08         4.1e-03
    internvl  1.6e-07         1.3e-03
    qwen2vl   1.4e-07         1.3e-04
    qwen25vl  1.1e-07         2.4e-04
    minicpmv  1.5e-07         9.2e-04
    janus     3.7e-07         9.2e-03

-- the device's shape: one block before the tiers' int8 roundings compound, the
whole tower after.

**And the move does not slow an encode, which it did twice before it was
measured.** One process per sample, ABBA, the runner against the tower's own
loop (a binary of the tree before the move), A/A first; the ratio is the old
encode's time over the new one's, so above 1 is the runner faster:

    first cut               SmolVLM host 0.91, Qwen2.5-VL host 0.91,
                            SmolVLM CUDA 0.42 -- REGRESSIONS
    found                   a picture's bias rows and K/V writes ran one row at
                            a time where the tower's loop had them on the pool
                            (a tenth of SmolVLM's host encode); and a device
                            whose FIRST block is a tower's builds no one-row
                            scratch, so its batched one inherited an unprobed
                            softmax width and took the one-thread-per-head
                            kernel (2.6x on the tower's attention, located by
                            pricing each phase with tier.WithSkip)
    fixed                   A/A        SmolVLM host 1.018 (IQR/median 0.054),
                                       CUDA 0.997 (0.038)
                            SmolVLM    host 0.984 (0.033)   CUDA 1.016 (0.020)
                                       Vulkan 1.019 (0.083)
                            Qwen2.5-VL host 1.029 (0.081)
                            gemma3     host 0.995 (0.051, n=4)

Every arm is inside its A/A bias: parity, on the laptop (six P-cores, RTX 3050
Ti), ten rounds an arm unless marked.

**Two bugs the move found, both live on main before it.** A mixed prompt on a
model with per-layer embeddings (Gemma 4's E models) took token 0's per-layer
input for EVERY row, text rows included -- prefill read the row's id only where
the whole prompt was ids -- so a chat with an image in it ran the text after the
picture on the wrong per-layer embedding (`TestPrefillMixedMatchesPrefill`
262144 of 262144 logits wrong, worst 3.3; exact now). And
`TestForwardEmbdMatchesForward` ran tuned, so on arm64 its two states timed
their way to two kernels and every model read as wrong there.

### As it stood in AGENTS.md

**And the PROJECTOR list is a second list**: `idefics3`, `mlp` (llava),
`qwen2vl_merger`, `qwen2.5vl_merger` (below), `gemma3` (Gemma 3's SigLIP, average pool, RMSNorm, one
matrix), `internvl` (InternViT-300M, the 2x2 shuffle, LayerNorm and a
two-matrix MLP; the 6B ViT with a q/k norm is refused by name), `resampler`
(MiniCPM-V 2.6/4.0/4.5, minicpmv_version 3-6), `janus_pro`, `pixtral`
(Mistral Small 3.x, Ministral 3, Pixtral-12B) and `llama4` (Llama 4 Scout
and Maverick): both below. An unknown one is refused at conversion with
`ErrNotImplemented`. (An unknown `tokenizer.ggml.pre` is NOT refused at
conversion -- see below.) What the newest cost is under "THERE IS ONE MODEL"
below.

### ★ THERE IS ONE MODEL. VISION IS NOT A SECOND ONE.

**A vision tower is a transformer over patches, and every way in which this
engine treated it as a separate thing was llama.cpp's technical debt rather than
a property of the model.** llama.cpp was text-only when towers arrived, so
libmtmd bolted a second GGUF on beside the first and enforced the relation
between them at RUN time: *"mismatch between text model (n_embd = %d) and mmproj
(n_embd = %d)"*. jitllm copied that shape inward and paid for it with a second
loader, a second container, a second page budget, a second `nn.JIT`, a second
state and a duplicate of `pageIn`, `releaseLayer`, `PageSize` and `WeightBytes`.

**GGUF is the CONVERTER'S input.** A property of the input format has no business
surviving into the container, so the two-file contract dies at conversion:

    jitllm convert model.gguf mmproj.gguf out.jlm

`model.Open` builds the tower when the container carries one and `m.Tower()`
hands it over. There is no `OpenTower`, no `-mmproj` flag, and the `ProjDim ==
NEmbd` equation is refused at CONVERSION, where there is a file in front of a
person -- the same place `convert.visionOf` already refuses a projector this
engine does not implement.

★ **TWO PAGE ARRAYS, NOT TWO FILES, AND THE REASON IS ARITHMETIC.** One page
size is wrong in BOTH directions: SmolVLM-256M's tower blocks are 7.59 MiB
against its text model's 3.81, so one size costs +55% (205 -> 319 MiB); a 30B's
text blocks are 392.66 MiB against any tower's handful, so one size costs 4.5
GiB the other way. Vision block i is at `VisDataOff + i*VisPageSize` exactly as
text block i is at `DataOff + i*PageSize` -- one more multiply, still an index,
still physically padded, still no cursor. `Role.Vision()` decides which array a
tensor belongs to, so nothing else in the reader branches on it.

★ **AND THE PAGER DOES NOT KNOW A TOWER EXISTS.** Vision blocks occupy
`NBlocks..NBlocks+NVisBlocks-1` of ONE block space, so residency, the chunk
bitmap, MRU eviction and the budget are written once. A vision block competes
for a page exactly as a text block does.

★ **THE BUDGET IS BYTES NOW, BECAUSE A PAGE NO LONGER HAS ONE SIZE.**
`budget/PageSize slots` was exact while every page was the same size and is
simply wrong with two arrays -- it would charge a 7.59 MiB vision page 392.66
MiB on a 30B. `jlm.File.SetBudget(bytes)` keeps everything the count gave: no
estimate in it, a page is either allocated or not, and "the model fits" is still
"the budget holds every block, so nothing is ever evicted". Buffers are reused
through a free list keyed BY SIZE -- handing a 7.59 MiB buffer to a 392.66 MiB
page would be a bounds panic at best.

★ **AND ALL OF THE FOLLOWING DELETED THEMSELVES**, which is how you know the
separation was the problem and not a thing to manage: the division of one host
budget between two containers, the precedence rule for which got its bytes
first, the second `setPageBudget` call that handed them back after `Encode`, and
`Tower.PageSize`/`WeightBytes`/`HostBytes`, which existed only so a caller could
do that arithmetic. `Tower.Release()` went too, and the re-budget after an
encode with it: a tower block is a page like any other, and the pager evicts a
finished picture's blocks as it evicts any block.

★ **THE TOWER IS A SEGMENT OF THE MODEL, RUN BY THE ONE RUNNER.** Its blocks are
`Model.layers[NBlocks:]`, the block space the container already numbered, and a
picture is rows through a `State` over that range whose `Config` is the
segment's (`State.Vision()`, `TowerConfig.segConfig`): the text runner's own
block body (`hostRows`, prefill's), on the text State's JIT and pool -- one JIT,
which holds an attention set per geometry (`nn.JIT.AttnSetFor`, `AttnTiledFor`)
where `AddAttn` held one pair, which was the only reason a tower ever had a JIT
of its own. Two pools on six P-cores had been the worst configuration this
engine had (TWELVE worker threads spinning on the same barriers). What the
restructure deleted: `TowerState` and its block loop and helpers,
`nn.NewJITSharing`, the resampler's third JIT, `Tower.Release`, `EncodeLayout`
and the caller's encode stage, the tower's own offer path, `g.vbs`,
`visionwin.go`, `LayerPlan.Vision/VisionRMS/VisionGated/VisionWindowed`,
`nn.BidirDevice` and `nn.VisionWindowDevice` (both one `nn.KeyRunDevice`), and
the app's per-file embedding map. `docs/design/vision-as-blocks.md` is the
design.

★ **A PICTURE IS A SPAN OF THE PROMPT, NOT A STAGE BEFORE IT.** `Span.Picture`
(`State.Picture`, `Tower.PicturePieces`) carries the preprocessed pixels; the
prefill encodes it in the State's vision segment, or the model's image cache
answers with no block run. `Run.Spans` puts it in a `StepRuns` step, and
`Run.Encode` runs a picture's blocks in the step other sessions decode in, up
to `Encode.Blocks` a step: split by blocks, never by rows, whose attention
reaches every other row. A warm encode allocates nothing, every family
(`TestWarmImageEncodeDoesNotAllocate`).

★ **AND A PICTURE HAS A NAME, SO THE CACHES SEE PAST IT.** `ImageKey` is sha256
over what the tower reads -- the family's preprocessing configuration, the grid,
the preprocessed pixels -- known before a block runs. The image cache is keyed
by it; the prefix cache names each image row by the full name, its index and
(M-RoPE) its rotary position (`kvCache.nameRow`), so the pages after a picture
are named as text pages are, a second turn about it restores up to its own new
tokens, and a picture the restore covers is never encoded -- not even looked up
(`TestTwoTurnImageChatRestoresPastThePicture`; a hybrid restores at its one
exact end, `TestHybridImageChatRestoresAtItsEnd`). A name over the projector's
OUTPUT values can do neither: it exists only after the tower ran, and on a
device the split tuner picks a reduction order per process. `Encode` itself
never reads the cache -- a gate comparing two placements of one picture must
run both.

★ **AND THE TOWER RUNS ON THE DEVICE. IT IS NOT A DECLINE ANY MORE.**
RULE 8a says a block the device cannot express is declined BY THE DEVICE, not
withheld by the model -- and a tower's blocks were withheld, because while it
was a second model behind a second loader they never reached `PrepLayer` at all.
That is the absence the rule is written against, and it is the same state
gemma3's post-norms sat in for months.

A vision State's `SetDevice` -- the one placement path, `State.offerRange` --
offers them, and the plan is the text kit's: `LayerNorm` or not, `UngatedFFN` or
not, `NRot` over the per-patch table `Layers` is handed as cs (zero for a
position-table tower), and `NonCausal` -- the call's rows attend over the call's
key runs and the block keeps no history -- with `Windowed` on Qwen2.5-VL's
windowed blocks. One `Vision` flag stood for all of that until the second
family, and every family after it added a flag the text plan already had under
another name. On the tier a non-causal segment's blocks run in a scratch SET of
their own (`geoNonCausal`, Gemma 4's per-geometry sets), built for the whole
picture, and write one shared k/v pair charged to the KV budget
(`transientkv.go`): their keys live one block deep.

What each difference needed, and none of it was a new idea -- the pieces
existed and had never been wired together:

| difference | what it took |
|---|---|
| LayerNorm | `LayerNorm{Part,Var,Apply}Rows` -- the single-row kernels existed; the batched ones did not, and the per-row partial base is where a batched norm goes wrong |
| no rotary | the RoPE kernels are NOT generated, and `CopyAtRowsT` puts k in the cache's TRANSPOSED layout that `RoPERowsT` would have written |
| a 2-D rotary (Qwen2-VL, Qwen2.5-VL; Pixtral's interleave is the same table) | the TEXT block's RoPERows/RoPERowsT over a per-patch table the host builds with `nn.Rope.Runs` -- (grid row, grid column) as two runs -- and hands `Layers` as its cs. No kernel knows a position has two axes. Until it was in the plan the device ran Qwen2-VL's tower with NO rotary and declined nothing (`TestRotaryTowerOnDeviceMatchesHost`: 3.3e-1 against the host one block in) |
| bidirectional | one line: the per-row key count is `nrow` for every row instead of `pos+r+1`. The mask here is a count, not a triangle |
| ungated MLP | `kernels.Act` in place of `ActMul`, and slot 4 (the gate) skipped -- an absent weight, not a failed upload |
| biases | `BUp`/`BDown` join the existing bias upload; the LayerNorm's shift is baked into the apply kernel |

★ **AND A VISION BLOCK IS ONE SUBMISSION, WHICH IS ARITHMETIC AND NOT A LIMIT.**
Bidirectional means no row's scores exist until EVERY row's k and v are in the
cache, so the whole image goes at once -- `batchWidthFor` gives a vision plan its
patch count where a text plan gets 128. A causal chunk works because row r only
needs keys 0..r; drop that and a 128-row chunk of a 1024-patch image attends to
the 128 keys it just wrote and misses 896, which does not fault.

★ **THREE BUGS IT TOOK, AND TWO WOULD HAVE CORRUPTED THE TEXT MODEL:**
- `g.layers` is keyed by block index, so the tower offering block 0 **replaced
  the text model's block 0**. Vision blocks are offset past `NLayer` now, the
  same block space the container uses.
- `g.bs` holds ONE plan, so a ViT block submitted against the text model's
  scratch launches every kernel with the wrong element counts. The tower's
  blocks run in their own geometry's scratch set (`geoNonCausal`).
- `batchMV` read `ActWin` from `g.bs.p` unconditionally -- the text plan.

★ **AND A LAZY COMPILE INSIDE A SESSION IS A DEADLOCK, NOT A SLOWDOWN.**
`batchMV` building a twin from inside `layersOnce` posts a `Load` to the single
CUDA goroutine that is blocked running that session; the process sits at 0.4%
CPU forever. `prepBatch` already walks the layers eagerly for the text path and
says why ("safe to call from inside a Session, where Compile is not allowed") --
`prepLayer` now does the same for a vision block.

    SmolVLM-256M, 1024 patches -> 64 embeddings:
      host    ~920 ms      device   144 ms      token ids identical

`TestTowerOnDeviceMatchesHost` takes its bound **after ONE block** (NMSE
5.4e-6), because both tiers quantize activations to int8 and do not round
identically -- 5e-6 after one block is 1e-3 after twelve, which is compounding
rather than a defect. Seven violations, each of the five differences plus the
two bias sets, land at 2.1e-1 to 6.9e+1 against a 5e-5 bound.

`TestOneContainerCarriesTextAndVision` is the gate on the separation staying
dead; against a regression to two files it fails at `m.Tower() == nil`.
`TestTowerOffersItsBlocksToTheDevice` fails at 0 offers against the withholding
it replaced.
`TestVisionRunsAsBlocksOfTheModel` holds the rest: the tower's blocks are the
model's, an encode runs every one through `hostRows` on the text State's JIT,
and `PrepLayer` is called from `offerRange` and nowhere else in the package.

★ **gemma3 AND internvl: WHAT BINDS.** `docs/engineering-history/model-correctness.md`
("Gemma 3 vision and InternVL 2.5 / 3") has the measurements.
- **Their preprocessing is the reference processor's, to the bit**
  (`engine/model/imageproc.go`): torchvision's antialiased resize in int16 fixed
  point, InternVL's tile grid and order, the float32 normalisation.
  `TestPreprocessMatchesTransformers` demands zero differing values; a new
  projector family brings its processor to the same bar, held to transformers'
  pixel values on real pictures (PNG -- a JPEG's decoders disagree first).
- **A gate against transformers runs the PROJECTOR ALONE too**
  (`TestProjectorMatchesTransformers`): the tower's int8 activations put
  end-to-end features at NMSE 5e-3 to 9e-3 of the float32 reference, which is
  wider than what a shuffle's order moves the reference itself.
- **The class token goes first**, as the reference does; llama.cpp's InternVL
  puts it last and shifts every patch's position (NMSE 0.18-0.25 against
  transformers). `towerCLSLast` exists only for node-by-node oracle gates.
- **A bidirectional run in a causal prompt is a per-row key count** (`Span.Bidir`,
  `nn.KeyRunDevice`, `engine/model/bidir.go`): a prefill chunk never cuts one, a
  device that cannot honour it does not run the chunk, and the device forgets the
  runs with the prompt. One row at a time is causal by construction, so a chunk
  holding a run never takes the per-row retry.
- **A vision block whose score planes exceed 128 MiB runs its attention in query
  chunks on the device** (`jit_old/gpu/tier/visionattn.go`, `Stats.VisionAttnChunks`)
  -- the whole image is still one submission and every row still sees every key.
  And every vision block on a device writes ONE shared k/v pair
  (`jit_old/gpu/tier/transientkv.go`): a block's keys live only inside its own
  submission, so a cache per block was a gigabyte of a 4 GB card for SigLIP's 27
  blocks.
- **The processors' resize is generated code** (`nn.ResampleRowJIT`,
  `cpu.EmitResample` on AVX2, SSE and arm64): Pillow's and torchvision's passes
  are integer dots over 8-bit samples, bit-for-bit the processor's; the filter's
  taps, the slice plan and the transposes around the kernel are setup. (The
  transposes, the widening and the rest of the pixel path went to generated
  code later: "Image preprocessing is generated code".)

## Pixtral / Mistral 3 vision: Qwen2.5-VL's kit with three pieces of its own

Mistral Small 3.x, Ministral 3 and Pixtral-12B share llama.cpp's projector
`pixtral`: an RMSNorm, gated-SiLU, bias-free ViT at the picture's own size,
then -- where the file states a merge -- an RMSNorm over each patch, Mistral 3's
learned 2x2 merger, and llava's linear, GELU, linear. Every block op was the
runner's already (Qwen2.5-VL's tower is the same kit with biases). Three
pieces were not, and each is read off the reference:

    the resize      PixtralImageProcessor: ratio = longest side / image_size,
                    both sides floored by it when it is over one, each then
                    rounded UP to a merge unit (28 px); Pillow BICUBIC.
                    Qwen's smart_resize rounds to nearest -- the gate's
                    violation lands on another size
    the rotary      one sequence base^(-2i/head_dim), the even indices for the
                    grid row and the odd for the column, on INTERLEAVED pairs:
                    llama.cpp's converter permutes q and k for build_rope_2d's
                    adjacent pairs (nn.RopeRun's Freq/Step; ropeaxes.go named
                    the shape before it was used)
    the merger      torch's unfold reads each 2x2 block CHANNEL-major
                    (c*4 + dy*2 + dx); the converter permutes the merger's
                    columns into the shuffle's patch-major order
                    (convert.permuteMerger, exact), so the head shuffles as
                    every other projector does

and the prompt: PixtralProcessor writes each merged row's [IMG] tokens then
[IMG_BREAK], the last made [IMG_END]. The head writes the break's embedding
between rows (llama.cpp's v.token_embd.img_break, the reference's own row) and
the marker table writes [IMG_END]. llama.cpp's converter drops that row from a
transformers 5 checkpoint (its mmproj pass skips every language_model tensor
before it looks for the embedding), so `convert.imgBreakRow` takes it from the
text table in the same container when the mmproj lacks it.

The fixture is transformers' own Mistral3ForConditionalGeneration (Ministral
3's text model with its YaRN and temperature, a 3-block tower with an MLP of
100, a biased projector), both GGUFs written by llama.cpp's converter
(`scripts/pixtralgold.py`), a 140x196 picture -- 5x7 merged, four break rows:

    the processor's resize (300x170 -> 224x140)   NMSE 3.1e-12 (bilinear: 5.3e-04)
    the head's rows, transformers' arithmetic     NMSE 1.9e-04 at the engine's int8
                                                  (1.3e-03 against the f32 tower,
                                                  where the int8 alone is 1.2e-03)
    the four break rows                           [IMG_BREAK]'s embedding exactly
    violations                                    no rotary 2.3e-01, NEOX pairs
                                                  2.1e-01, Qwen's frequencies
                                                  1.1e-01, the merger read
                                                  channel-major 3.1e-01
    from the pixels, end to end                   worst logit NMSE 5.6e-03, 0 flips
                                                  (8.6e-03 against the f32 tower);
                                                  no break rows 1.18
    every block on cuda:0 / vulkan:0              7.0e-04 / 3.2e-04 against the host,
                                                  home again bit-identical

Two scales of the fixture were set by measurement. At the q/k init that
Qwen's fixture uses (0.3) the int8 alone moved transformers' rows by 2e-2 and
the engine sat 2e-3 from its own arithmetic; at 0.22, 6e-3 and 8e-4 with every
violation still past 0.1. And a projector at the other weights' 0.2 put the
image rows at an RMS of 6.3 against an embedding row's 0.17, which the toy
text model amplified a hundredfold (0.19 of the logits from 8e-4 on the rows);
at 0.03 the rows sit at the text model's scale. Given transformers' own rows
the text model reads 5e-12.

Under the principles: one vision page against a three-page tower evicts and
re-reads with every bit the same; a decode after the picture makes 0 engine
allocations on the host, CUDA and Vulkan; a second turn about the picture
restores 576 of 598 positions -- the picture, its breaks and the text after
it -- with no tower block and no image-cache lookup, and decodes what a cold
run does; a warm encode allocates nothing.

The real Ministral-3-3B (the published Q4_K_M text model and BF16 mmproj, one
container): the photo (640x488 -> 644x504, 46x36 patches, 414 rows and 17
breaks) encodes in 9.1 s on the host, and greedy decoding from its chat
template reads *"This image depicts the front page of The New York Times from
July 21, 1969, announcing the historic Apollo 11 moon landing with headlines
like "Men Walk"*. Against transformers' own PixtralVisionModel and
Mistral3MultiModalProjector in float32 on the checkpoint's own tower
(`scripts/pixtralreal.py`, only the tower's tensors fetched):

    the processor's pixels           NMSE 8.5e-15, max|d| 2.4e-07 (an ulp of
                                     the normalisation)
    the 414 rows, host               NMSE 8.2e-03 -- 24 blocks of Q8_0 weights
                                     and int8 activations, Janus-Pro's order
    every block on cuda:0 / vulkan:0  1.9e-03 / 1.7e-03 against the host,
                                     8.5e-03 / 7.5e-03 against transformers

The device arms run with the photo's own ceiling: at image_size 1540 the
tower asks a 4 GB card for scratch over 12,100 rows and the placement declines
every block by budget, which is what it should do with a card that small.

### As it stood in AGENTS.md

★ **pixtral IS QWEN2.5-VL's KIT WITH THREE PIECES OF ITS OWN, EACH READ OFF
THE REFERENCE.** The block is an RMSNorm, gated-SiLU, bias-free ViT at the
picture's own size (`TowerConfig.pixtralResize`: the longest side floored
under image_size, each side rounded UP to a merge unit -- Qwen's smart_resize
rounds to nearest and is the gate's violation). Its 2-D rotary is one
frequency sequence split even/odd between the row and the column on
INTERLEAVED pairs, because llama.cpp's converter permutes q and k
(`pixtralRope`, `nn.RopeRun.Step`). Mistral 3's merger reads torch's unfold,
which is channel-major, so `convert.permuteMerger` puts its columns in the
shuffle's patch-major order at conversion and the head shuffles as every
other does. The head writes the [IMG_BREAK] embedding after each merged grid
row but the last (the mmproj's copy, or the text table's row when llama.cpp's
converter dropped it, `convert.imgBreakRow`); [IMG_END] is the marker.

## Llama 4 vision: a CLIP tower over tiles, and the three details a port gets wrong

Llama 4's tower (Scout and Maverick share it: 34 blocks of 1408, head_dim 88,
336-pixel tiles of 14) is the CLIP kit -- LayerNorms with biases, biased
q/k/v/o, a GELU MLP with biases -- and its adapter is the 2x2 shuffle every
tower here runs (pixel_shuffle's two reshapes and permutes come to dy outer,
dx inner), two matrices each followed by a GELU, and the projector's matrix.
Three details are its own, each read off Llama4VisionModel:

    the class token   LAST (cat([patches, class])), with the position table's
                      last row
    the rotary        beside the table, on adjacent pairs (view_as_complex),
                      the column's half first, at (x+1, y+1), the class token
                      masked to zero: it does not turn
    the processor     Llama4ImageProcessor: the best-fitting canvas of up to
                      16 tiles (find_supported_resolutions' order and
                      get_best_fit's float32 scales), the picture resized onto
                      it without distortion with torchvision's antialiased
                      bilinear and padded black, tiles row-major, then the
                      whole picture squashed to a global tile; rescaled and
                      normalised in BFLOAT16, as the original implementation
                      does

and the prompt: <|image_start|>, each tile's rows with <|tile_x_separator|>
between two of a row and <|tile_y_separator|> after its last, <|image|>, the
global tile, <|image_end|> -- each tile a picture span of its own, so each is
named and cached as a picture.

llama.cpp's tensor map names a transformers 5 checkpoint's
`vision_model.model.layers.N.post_attention_layernorm` -- Llama 4's pre-MLP
norm -- `attn_post_norm`, the name it gives Gemma 4's genuine post-attention
norm; its own graph would apply it after the attention and run the MLP on no
norm at all. The converter reads it as ln2 for this projector.

The fixture is transformers' own Llama4ForConditionalGeneration with Scout's
text features at toy size (`scripts/llama4visiongold.py`, both GGUFs by
llama.cpp's converter), a 224x168 picture on a 2x2 canvas, not resized,
padded at the bottom:

    the processor's tiles (5 of 112)      0 of 188,160 values differ
                                          (a float32 normalisation: 24,838 of
                                          the global tile's 37,632)
    the rows at the engine's arithmetic   NMSE 3.9e-04 (f32 tower 1.1e-03)
    violations     the class token first 1.4e-01, the row before the column
                   2.0e-01, the shuffle transposed 1.8e-01, positions from
                   zero 8.2e-03 -- a rotary's relative scores cannot see a
                   shift of every patch, so only the unturned class token
                   tells, and that one is held to ten times the clean arm
    end to end from the pixels            worst logit NMSE 1.4e-03, 0 flips
                                          (1.4e-02 against the f32 tower);
                                          the global tile first 0.33
    every block on cuda:0 / vulkan:0      5.0e-05 / 6.3e-07 against the host

Both GELUs in the reference are erf (nn.GELU); the engine runs the tanh form,
as llama.cpp's ggml_gelu does, and the arithmetic-matched golden takes the
tanh form so the bound stays tight -- the f32 column is the distance to the
reference as written. Under the principles it is Pixtral's: one vision page
of 72 KiB against a 3-page tower (14 evictions) bit-identical, 0 allocations
after a picture on the host, CUDA and Vulkan, a second turn restoring 112 of
136 positions past the picture with no tower block, a warm encode allocating
nothing. `jitllm run -image` lays a picture out on its canvas; padding tiles
are pictures the image cache already holds.

### As it stood in AGENTS.md

★ **llama4 IS A CLIP TOWER OVER TILES, AND THREE OF ITS DETAILS ARE THE
ONES A PORT GETS WRONG.** The class token is LAST, with the position table's
last row (`TowerConfig.CLSLast`); the 2-D rotary beside the table turns
adjacent pairs, the column's half first, at positions starting from ONE, and
the class token does not turn (`llama4Coords`); and the processor normalises
in bfloat16 (`bf16NormLUT`), so a float32 normalisation is another picture.
The tiles are Llama4ImageProcessor's best-fitting canvas, padded black, plus
a global tile (`Tower.Llama4Pieces`), each a picture span of its own between
the processor's separators (`Model.Llama4Spans`). The adapter is the shuffle
every tower runs and two matrices each followed by a GELU, then the
projector's (`RoleVProj3`). llama.cpp's tensor map names a transformers 5
checkpoint's pre-MLP norm `attn_post_norm` (Gemma 4's name); `convert.towerName`
reads it as ln2.

The vision families' codes are taken from the top of the code space down
(`format/jlm/visionfam.go`). `docs/engineering-history/model-correctness.md`
has the numbers.

## RULE 7's list, scope and refusals as they stood in AGENTS.md

### ★ RULE 7: the architecture list is a LIST, and adding one is a decision.

Forty-seven: llama/llama3, qwen2, **qwen2vl**, qwen3, qwen3moe, olmoe, gemma1,
**gemma2**, **gemma3**, phi3/phi4, **qwen3next**, **gpt-oss**, **deepseek2**,
**kimilinear**, **llama4**, **granite** (and granitemoe), **phi2**,
**starcoder**, **starcoder2**, **command-r** (and cohere2), **stablelm**,
**falcon**, **nemotron**, **dbrx**, the mixture families **glm4moe**, **qwen2moe** and **ernie4_5-moe**
(ERNIE 4.5's dense **ernie4_5** is llama's graph and converts as llama),
**hunyuan** (hunyuan-moe and hunyuan-dense, one arch), the dense
llama family **smollm3**, **arcee**, **seed_oss**, **olmo2** (and OLMo 3),
**exaone4** and **mistral3** (text), the flagships **minimax-m2**, **gemma4**,
**deepseek32** and **gemma3n**, the state-space families **mamba2**,
**granitehybrid** (Granite 4.0-H), **nemotron_h** (Nemotron-H, Nano 2, and
Nemotron 3's nemotron_h_moe) and **falcon-h1** (attention and Mamba-2 side by
side in one block), the short convolutions **lfm2** (and lfm2moe, one arch),
Mamba-1's **mamba** (and FalconMamba) and **jamba**, and the two EMBEDDING
encoders **bert** and
**nomic-bert** —
plus `clip` for the vision tower. `convert.archOf` (convert/config.go) is the
list.

llama.cpp has 176 `LLM_ARCH_` cases and 50 graph builders. Ollama had a funded
team, covered ~21 architectures, and deleted 430,004 lines citing exactly this.
**Scope is jitllm's only defence.**

**jitllm DOES refuse to load some models, and the old text claiming otherwise was
false twice over:**

1. An architecture not in `convert.archOf` does not convert: `configOf`
   returns `architecture %q is not implemented`. That is where the refusal lives
   now that the architecture switch is the CONVERTER's and not the loader's.
2. `quant.Dequantable` is `{F32, F16, BF16, Q4_0, Q5_0, Q5_1, Q8_0, Q3_K, Q4_K,
   Q5_K, Q6_K, MXFP4}`. BF16 joined it for qwen3next's shared-expert gate, MXFP4
   for gpt-oss's expert banks and Q5_1 for Falcon's attn_qkv (every published
   K-quant of it; `kernels.MinInD` is the one question a reader asks). The
   canonical list of PACKED types, the one a gate should sweep, is
   `quant.PackedTypes` (format/quant/types.go); a hand-written copy anywhere else is a
   bug waiting -- adding MXFP4 found eight more in tests and two in shipping code
   (`SupportedPackedNative` on arm64, `WideActWindow`), and Q5_1 found five more
   in the device tests, two of which had left Q5_0 with a NaN oracle.

**So adding a quant type is TWO pieces of work and the reference dequantizer comes
FIRST**, because RULE 8 makes `internal/oracle` the reference every kernel is gated
against, and the oracle can only decode what `Dequantable` lists.

**The expensive part of a new architecture is never the arithmetic** — it is the
tokenizer, and MoE. An unknown `tokenizer.ggml.pre` is REFUSED, not defaulted
-- at LOAD (`tok/spm.go`, "pre-tokenizer %q is not implemented"). Resolution
happens at CONVERSION (`convert/vocab.go`), where an unknown name is NOT an
error: it is stored as an empty pipeline, because the legacy family splitters
still cover several names and the reader must tell "no stages" from "stages
nobody verified".

**Read the artefact, not the description of it.** Transcribe a graph from
llama.cpp's *running* output (`llama-eval-callback`), and read the TENSOR table,
not the key summary — a cost estimate made from keys alone was wrong three times
in one session about one model.

## The classic-transformer kit (C6): phi2, starcoder, command-r and cohere2, stablelm, starcoder2, falcon, nemotron, dbrx

★ **phi2 IS THE FIRST CLASSIC-TRANSFORMER BLOCK (C6), AND THE KIT IS SHARED.**
`FlagLayerNorm` (every text norm a LayerNorm, bias a tensor that may be
absent), `FlagParallel` (attention and FFN read the block INPUT; the FFN norm
when the block has one, the attention's normed row when not), an UNGATED FFN
(the absent gate tensor) with `ffn_up/down` biases, norm and head biases.
Gated by `scripts/c6gold.py` -- transformers' own class, written to GGUF by
llama.cpp's OWN `convert_hf_to_gguf.py` -- in `model.TestC6MatchesTransformers`
and `TestC6FeaturesAreLoadBearing`. It runs on CUDA and Vulkan with every
block and the head placed (`TestC6OnEveryDevice`): the vision tower's LayerNorm
and ungated-MLP kernels serve the text block (`tier.lnBlock`/`ungatedFFN`), the
parallel FFN reads `bs.h` or norms `bs.x` (the block input, untouched by the
attention half), and a LayerNorm head carries `nn.Head.NormB`. The head's OWN
bias is added by `finishLogits` on the host after either arm, so the device's
argmax is not used when a model has one.

★ **starcoder (GPTBigCode) ADDS THE ONE PIECE NO OTHER TEXT GRAPH HAS: A
LEARNED POSITION TABLE** (`RolePosEmbd`, `Model.addPos`), added to the
embedding on every entry point -- inside `forward`'s fill and prefill's
`refill`, because a device failure restarts from them -- with `FlagNoPosEnc`
for the absent rotary, and the context clamped to the table's rows. The device
needed only `copyK` for a model with no rotary anywhere.

★ **command-r/cohere2 (Cohere: Command-R, Aya, Command-R7B, Command-A) is the
kit with NO new piece**: one bias-free LayerNorm feeding a parallel block, a
SwiGLU FFN, and a `logit_scale` that MULTIPLIES -- stored as its reciprocal in
`LogitScale`, which divides and is exact for Cohere's powers of two. cohere2
adds llama4's switches (`SWAPeriod`, `FlagNoPEGlobal`). command-r-plus's
per-head q/k LayerNorm is refused by name at conversion. **stablelm** is the
kit too, with its residual read off the TENSOR as llama.cpp reads it: an
ffn_norm makes the block sequential whatever `use_parallel_residual` says.
**starcoder2** is the kit's sequential block with every matrix biased; its
config.json's sliding window is NOT applied, as llama.cpp does not apply it
(RULE 7m, chosen: the GGUF carries no window).
**falcon** is the parallel block with an ungated GELU-tanh FFN (llama.cpp's
ggml_gelu; the model trains with erf GELU -- RULE 7m, chosen) and 40B's two
input norms written as the block's attention and FFN norms at conversion
(`attn_norm_2` -> `RoleAttnNorm2`, `convert.falconNorms`). **nemotron** is
the sequential block with half-head NEOX rotary and an ungated squared-ReLU
FFN -- `ActReLU2`/`FlagReLU2`, a fourth ungated kernel on every tier. Its
LayerNorm1p's +1 is already in the GGUF (llama.cpp's converter folds it), so
the engine runs a plain LayerNorm and must not add it again. **dbrx** is the
kit's bias-free LayerNorm on a softmax top-k mixture, with q, k and v clamped
to `ClampKQV` straight after their projections (an optional config tail after
Granite's, so no container version; `jlm.ArchDBRX` = 27). Its `attn_output_norm`
is the FFN NORM (`convert.retarget`), not BERT's post-norm. Gated on a
`DbrxForCausalLM` fixture only (`synth-dbrx`, host and CUDA/Vulkan) -- every
real DBRX is 132B and none fits this box.

## qwen3next, qwen35 and qwen35moe: the first hybrid

★ **qwen35/qwen35moe** (Qwen3.5/3.6, added on the user's direction)
are qwen3next's graph under llama.cpp's names, NOT another graph: split beta/alpha fused back into ssm_ba, the value
heads TILED (`jlm.FlagDeltaKeyTiled`, reachable only at two or more value heads
per key head -- Qwen3.5-4B reads 4/20 against llama.cpp with the grouped
pairing, 20/20 with the tiled one), MTP blocks set aside. `convert/qwen35.go`.

**qwen3next was added on the user's explicit decision, to reach the
30B/70B band.** It is the first HYBRID: 36 of its 48 layers keep a recurrent
summary of the whole history instead of a KV cache, and 12 run ordinary
attention. What that cost, in the order it had to be done: a BF16 dequantizer
(the shared expert's gate is BF16 and nothing else on the box is), the container
carrying a per-layer kind map and an SSM geometry, a shared expert, a
double-width query whose gate interleaves per head, partial rotary (64 of 256),
and three new kernels on two architectures. **The tokenizer was free** -- it is
the gpt2 pipeline the table already had -- which is the exception to the rule
below rather than a counterexample to it.

## granite and granitemoe: four scaling constants

★ **granite/granitemoe** (IBM Granite 3.x/4.x dense and mixture, added
on the user's direction) are llama's block with four constants the
file states: embedding_scale, residual_scale (each block output, before its
residual add), attention.scale (the score, in place of 1/sqrt(head_dim)) and
logit_scale (DIVIDES the logits). An arch code of its own (`jlm.ArchGranite`) so
an older reader refuses the container rather than running it at scale one.
**The greedy gate cannot see two of the four** -- a positive logit divisor moves
no argmax -- so `model.TestGraniteMatchesLlamaCppValues` holds block 0's output
and the last logits to `llama-eval-callback`'s values. A mixture has no add of
its own to scale: the residual scale rides the router's routed scale
(`Config.routedScale`), and `nn.MoERouteJIT` refuses a gate whose scale differs
from the one its route was built for. On a device the residual scale is a
scaled residual add (`kernels.AddScaled`) in place of the fused ones, on every
backend and every path (decode, a prompt chunk, a step across sessions);
`model.TestGraniteScalesOnEveryDevice` holds it to the host at 1e-14 on f32
fixtures and `TestGraniteOnEveryDevice` on the real models, a violation per
scale. `docs/engineering-history/gpu-kernels.md` ("Granite on the device");
`convert/granite.go`.

## kimilinear: Kimi Delta Attention, and MLA with no rotary

★★★★ **AND kimilinear RUNS AND IS VERIFIED AGAINST transformers: 8 of 8
POSITIONS, WORST NMSE 5.08e-11 AGAINST A 1e-9 BOUND.** Kimi Delta Attention is
qwen3next's gated delta rule with a PER-CHANNEL decay, two LOW-RANK gates and a
sigmoid output gate; its full blocks are ArchDeepseek2's MLA and V3 router.
`kernels.GatedDeltaStepChan` and `cpu.EmitGatedDeltaChan` are the kernel on
four targets, `RoleSSMFA/FB/GA/GB` the gates, and `blockLinear`/`fuseOrder` the
converter machinery a safetensors hybrid needs.

★★★★ **AND THE LAST BUG WAS THE ONE THAT IS EXACT AT POSITION 0: ITS
ATTENTION IS MLA WITH *NoPE*, AND jitllm WAS ROPING IT.**
`KimiLinearAttention`'s docstring says so -- *"Multi-headed Latent Attention
(MLA) from Deepseek V2 with NoPE"* -- and its forward never calls
`apply_rotary_pos_emb`, never reads `position_embeddings`, and simply
CONCATENATES the `qk_rope_head_dim` channels into the key. The name is
DeepSeek's, inherited; the rotation is not.

**A rotation by position 0 is the IDENTITY, so the engine was bit-exact on the
first token and wrong on every one after** -- logits parted at position 1 by
NMSE 2.888e-03 while position 0 sat under 1e-9. Every first-token gate in this
tree passes such a model, and so does a greedy transcript that happens not to
flip an argmax.

★ **AND IT IS `FlagNoPosEnc` RATHER THAN `NRot = 0`, WHICH IS THE WHOLE
DISTINCTION.** NRot is load-bearing GEOMETRY for MLA: the absorbed key is
`KVLoraRank + NRot` wide and the cache row is sized from it, so zeroing NRot to
mean "no rope" would shrink the key by eight channels and silently drop them.
The flag says "do not rotate"; the geometry keeps its width.

★ **AND THE FIRST FIX FOR IT WAS WORSE THAN THE BUG.** A `continue` past the
rope in `mlaProjectRows` also skipped the absorb and **the KV cache write** --
louder, and caught immediately. Only the two `RoPE32JIT` calls are guarded.

★ **HOW IT WAS FOUND IS THE METHOD, NOT THE LUCK: BISECT THE GRAPH AGAINST THE
REFERENCE, LAYER BY LAYER AND TENSOR BY TENSOR.** Comparing per-layer residuals
put layer 0 (KDA) exact at every position and layer 1 (MLA) exact at position 0
and wrong at position 1, which is a five-minute narrowing that reading code
would not have produced. Two probe mistakes are worth keeping because each
produced a confident wrong answer first: patching
`chunk_kimi_delta_attention` globally with last-call-wins captured LAYER 2
while jitllm traced layer 0, and `from transformers import X` is LAZY, so a
smoke-test import "succeeded" without loading the module that was broken.

★★★ **AND THE DEVICE TIER HAD THE SAME ROPE BUG, WHICH IS RULE 8a's POINT
EXACTLY: A GRAPH CHANGE LANDED ON ONE TIER IS SILENTLY ABSENT FROM THE OTHER.**
Fixing `engine/model/mla.go` alone left `jit_old/gpu/tier` rotating, and `jitllm verify
-devices cuda:0` read **max|dlogit| 0.929 with 8 of 32 argmaxes flipped**
against synth-deepseek's 0.00195 on the same command -- 475x, which is what
separates a real bug from this file's own 0.2-0.94 reduction-order band. With
the copy in place it reads **0.00139, 0 of 32**.

★ **AND ON THE DEVICE IT IS A COPY RATHER THAN A SKIPPED LAUNCH, FOR THE
REASON THE HOST FIX ALREADY LEARNED.** `mlaRopeK` rotates straight into the KV
cache row's tail and `mlaRopeQ` into the absorbed query's rotary half, so
omitting the launches leaves both unwritten. `kernels.CopyAtRowsT`'s own
comment records the same trap for a vision tower, which has no rotary for the
same reason. `model.TestNoPEMLAOnEveryDevice` is the RULE 11c gate: cuda and
vulkan at worst logit NMSE 1.0e-12 and 9.4e-13, and it fails at 1.369e-02 /
0.531 against the rotation restored.

★★★ **AND KDA RUNS ON THE DEVICE: 3 OF 3 BLOCKS ON CUDA, 2 OF THEM LINEAR,
worst logit NMSE 4.993e-12 AGAINST THE HOST** (`model.TestNoPEMLAOnEveryDevice`).
`emitLinear` branches on `RecurrentPlan.ChanDecay` -- a field of its own,
because KDA's GEOMETRY is qwen3next's and its ARITHMETIC is not, so nothing in
the shape can tell them apart. **It needed no new kernel on either tier**:
`nn.SigmoidMul32JIT` and `kernels.SigmoidMul` both already existed for
qwen3next's attention out-gate, and both times the survey that called one
"owed" had read an accepted-set (`cpu.ActKind`, `kernels.ActMul`) instead of
grepping for the capability.

★ **A RANK THAT IS NOT A MULTIPLE OF 32 IS DECLINED BY NAME**: the low-rank
gates chain two matvecs through a quantized activation of rank width, and
`kernels.Quantize` works in 32-element blocks.

## The chat template, applied

★ **AND THE CHAT TEMPLATE WAS PARSED, CONVERTED, STORED AND NEVER APPLIED.
IT IS WIRED NOW.** `tok/jinja/` is a complete engine with a gate that renders every
template on the box; `convert/vocab.go` pulls `tokenizer.chat_template` (and
llama.cpp's named `tokenizer.chat_template.<name>`) out of the GGUF, or
`convert.WithChatTemplate` / `jitllm convert -chat-template` stores the
publisher's own in its place (RULE 7m: a GGUF's template is often older --
Mistral-7B-Instruct-v0.3's has no system role); `jlm.Vocab.Templates` carries it
into the container -- and nothing
outside jinja's own tests imported the package. **Every Instruct model ran as a
RAW COMPLETION**, which produces fluent output and no signal at all: RULE 10's
shape, a plausible answer from a configuration nobody selected.

`model.ChatPrompt`/`ChatIDs` render it; `jitllm run -chat [-system S]` is the
CLI, and it does NOT echo the prompt the way a completion does -- printing it is
what makes `"The capital of France is"` + `" Paris."` read as the one continuous
text a completion is, and under `-chat` it asserts two false things: the prompt
the model received is the RENDERED TEMPLATE, and a reply is a separate TURN. **Opt-in, and the default stays completion on purpose**: every board row,
`TestGreedyMatchesLlamaCpp` and the teacher-forced comparison are completions of
a bare prompt, so defaulting it on would move all of them at once and silently
make each a different measurement.

★ **THE PAIRING IS THE PART THAT GOES SILENTLY WRONG, AND COUNTING BOS TOKENS IS
THE ASSERTION THAT LOOKS RIGHT AND IS NOT.** A rendered template is a COMPLETE
prompt -- Llama-3.2's opens with `{{- bos_token }}`, Qwen3's opens with
`<|im_start|>` and emits no BOS at all -- so it must be encoded with
addSpecial=false or the first gets a doubled begin-of-text. The obvious gate
counts BOS ids and calls more than one a double; it fails on SmolLM2, whose BOS
token IS `<|im_start|>`, legitimately three times in a three-turn prompt. The
property is that the TOKENIZER must not add a special the TEMPLATE did not
render, so the gate compares against the naive pairing: Llama-3.2 36 tokens
against 37, tinyllama 18 against 19, the Qwens identical because AddBOS is
false. A box where no vocabulary set AddBOS would pass with the pairing
hardcoded either way, so the gate fails when none is reachable.

★ **AND A FLAG AFTER THE FILE IS PROMPT TEXT, WHICH IS RULE 10 ARRIVING FROM THE
COMMAND LINE.** Go's `flag` stops parsing at the first positional, so
`run <file> -chat "what is the capital of France?"` put the literal `-chat` at
the head of the PROMPT and generated a raw completion. The 80B answered *"The
capital of France is Paris."* anyway -- a strong model completes a bare question
fluently -- and the only visible trace was token 60535 leading the id list.
`misplacedFlag` refuses it in every subcommand, matching the flags the SET
DEFINES rather than a leading dash, so `"-42 degrees is"` still runs as a prompt.

## The stop set: what the file states, carried into the container

★ **THE CONVERTER DROPPED EVERY STOP ID BUT EOS, AND A REPLY RAN ON PAST ITS
TURN.** HunyuanOCR (catalogue `hunyuanocr`) closes an answer with
`<｜hy_Assistant｜>` (120007). Its GGUF states it only as
`tokenizer.ggml.eot_token_id`, beside an eos of 120020; its transformers
directory states both in generation_config.json's `eos_token_id` list. Neither
reached the container, and the tokenizer's stop set was EOS plus nine names
matched by text (`tok.eogNames`), none of which is `<｜hy_Assistant｜>`. Asked
"What is the capital of France?" under `-chat`, it answered in 57 tokens, then
wrote "The capital of France is Paris." over and over until `-n` ran out.

Every generation loop already stopped where `Vocab.IsEOG` said: `jitllm run`
(plain and speculative), the server's one-at-a-time and batched generate paths
(OpenAI, Anthropic and Connect go through them), the UI engine. The fix is the
set behind it, not the loops:

- **The container carries the stated ids** (`jlm.Vocab.Stop`), an optional tail
  of the vocab section after the per-op flags, so no version bump: a vocabulary
  with no stop ids writes the bytes it wrote before, an older reader ignores
  the tail, and a container written before it reads as no stated ids. With
  stops present the per-op flag tail is written even when no op sets a flag, so
  the reader can find where the stops start. An id outside the token table is
  refused at decode and at conversion.
- **From a GGUF**: `tokenizer.ggml.eot_token_id`, `eom_token_id` and the FIM
  pad, repo and file-separator ids, which llama.cpp's `special_eog_ids` takes
  besides EOS. A GGUF has no list key.
- **From safetensors**: generation_config.json's `eos_token_id`, one id or a
  list -- what transformers' `generate` ends on -- or config.json's (under
  `text_config` for a vision-language model) when there is no
  generation_config.json. Apertus's lists `</s>` and `<|tools_suffix|>` beside
  its tokenizer's `<|assistant_end|>`; Phi-3.5-MoE's lists `<|assistant|>`.
- **The tokenizer** unions EOS, the stated ids and the names, and the names
  are llama.cpp's current list: they had fallen behind it, so Llama-3.2's
  `<|end_of_text|>` (128001), the `</s>` a Qwen BPE vocabulary carries as an
  ordinary piece, gemma's `</s>` and Qwen's FIM pad, repo and file-separator
  tokens did not stop a generation they stop under llama.cpp. Its two
  exceptions hold: harmony's `<|end|>` is not a stop when `<|return|>` and
  `<|call|>` are (solar-open's `<|calls|>`/`<|flush|>` likewise), and `</s>` is
  not one beside gemma4's `<|tool_response>`. Where llama.cpp takes one FIM
  token of each kind -- the first its hash map yields -- every one present is
  taken.

`ignore_eos` (the server's) ignores the whole set, as vLLM's does: it skips the
`IsEOG` check rather than a single id.

Gates: `tok.TestStopSetMatchesLlamaCpp` holds the set built from a GGUF to the
one llama.cpp b10825 prints at load, on twelve vocabularies (HunyuanOCR,
Qwen2.5, stories260K-infill, gpt-oss, Phi-3.5, Llama-3.2, gemma-3, SmolLM2,
DeepSeek-V2-Lite, Mistral-7B-v0.3, Qwen3, phi-4), all equal;
`tok.TestStatedStopsEndAGeneration` drops the stated set and
`<｜hy_Assistant｜>` stops ending a generation; `convert.TestStopSetRoundTrips`
converts synth-hunyuanvl's GGUF and synth-apertus's safetensors, writes and
reopens each container and reads the set back, and reads the real HunyuanOCR's
from both inputs; `model.TestChatStopsAtTheTurnsEnd` runs HunyuanOCR-Q8_0
greedy under its chat template, stops at `<｜hy_Assistant｜>` after 57
tokens, and with the stated set dropped runs the full 64.
`TestGreedyMatchesLlamaCpp` and the forward goldens are untouched by
construction: they are completions of a bare prompt to a fixed length that
stop on EOS alone, not on `IsEOG`.

## The tokenizer authorities and the kimi-k2 pre-tokenizer (RULE 7m)

### ★ RULE 7m: llama.cpp IS A GOOD-ENOUGH ORACLE, NOT THE SPECIFICATION.

**It is runnable, it is what most people actually run a GGUF through, and it
catches gross errors instantly -- which is exactly what an oracle is for. It is
not the authority on what a model's tokenizer DOES**, and the difference has
teeth. The weights were trained against the model's own tokenizer; llama.cpp
carries a RE-IMPLEMENTATION of it, and says so in its own source:

    // original regex from tokenizer.json
    //"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|..."
    // adapted: .../pull/6920#issuecomment-2080233989
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|..."

**★ AND THE ADAPTATION IS LOSSY, MEASURABLY.** `(?i:s)` matches U+017F LATIN
SMALL LETTER LONG S under Unicode case folding; `[sS]` does not. Same for U+212A
KELVIN SIGN against `[kK]`. So `'` + U+017F is a CONTRACTION under the regex the
model was trained with and is NOT one under llama.cpp. Being bit-compatible with
llama.cpp there means being wrong in the same way it is.

**So there are THREE authorities and they disagree, and a test that only knows
about one of them cannot tell you which you are failing:**

    the model's tokenizer.json  -- what the weights were trained against
    llama.cpp's re-implementation -- what most people run, Unicode 15.1 tables
    Go's unicode package        -- whatever the TOOLCHAIN ships, 17.0.0 as written

jitllm currently uses the third by accident, which is the worst of the three: it
is nobody's intent and it changes under a `go` upgrade with nothing to catch it.

**The rule is not "always match llama.cpp" and not "always be standards-correct".
It is: KNOW which one you are matching, choose per divergence, and make the list
of divergences a reviewed artefact rather than a surprise.** Where llama.cpp
approximates and jitllm can be exact, being exact is the better engine -- but it
is a DECISION, because bug-compatibility with the thing users actually run has
real value too. What is not acceptable is not knowing.

**★ AND A FOURTH AUTHORITY OUTRANKS ALL THREE: THE CALLER.**
`tokenizer.ggml.pre` is a LABEL the converter wrote. Nothing in a GGUF ties it to
the tokenizer the weights were trained against -- so a name is an ASSERTION about
a converter, not evidence, and no fingerprint, sidecar manifest or `chkhsh`
generalisation can turn it into one. The vocabulary does corroborate it (a BPE
trained on individual digits cannot contain a `"1234"` token, and all 11 gpt2
vocabularies on this box agree with the pipeline their name selects), but
corroboration is not provenance.

So the GGUF stays the DEFAULT authority and `tok.WithTokenizer(r)` lets the
caller override it with the model's own `tokenizer.json`, parsed at RUNTIME by
`pretok.ParsePreTokenizer` -- the same parser `scripts/gentok` uses, which makes
`tok/pretok/table.go` (66 names; `go run ./scripts/gentok tok/pretok/table.go`) a CACHE
of that parse rather than a hand-maintained table
(`TestGeneratedTableMatchesArtefact` closes artefact -> ops -> checked-in source).
An override also RESCUES a `pre` name the table does not carry, which would
otherwise be refused. It replaces the pre-tokenizer and nothing else: `add_bos`
and `ignore_merges` still come from the name.

**The honest scope is CONFORMANCE TO AN IDENTIFIED ARTEFACT, not provenance.**
A manifest and a hash were both proposed as provenance; neither can supply it,
for the reason above.

★★★★ **AND THE SHARPEST DIVERGENCE YET IS kimi-k2, AS MEASURED:
llama.cpp TOKENIZES EVERY camelCase IDENTIFIER DIFFERENTLY FROM MOONSHOT'S OWN
TOKENIZER.** Moonshot ships no `tokenizer.json` at all -- Kimi-K2 and
Moonlight-16B-A3B carry a `tiktoken.model` plus a Python file whose `pat_str`
IS the authority, and the two models' patterns are byte-identical. It is o200k
with Han split out: a leading `[\p{Han}]+` alternative and `&&[^\p{Han}]`
subtracted from every letter class.

llama.cpp's `unicode_regex_split_custom_kimi_k2` implements that with ONE
`is_letter && !is_han` run -- a plain `\p{L}+` -- where the pattern specifies
o200k's two CASE-SPLIT alternatives. Run against Moonshot's own pattern under
Python's `regex` with V1 set operations:

    input           Moonshot's own          llama.cpp's kimi-k2
    getUserName     [get User Name]         [getUserName]
    iPhone          [i Phone]               [iPhone]
    helloWorld      [hello World]           [helloWorld]
    HTTPServer      [HTTPServer]            [HTTPServer]      same
    中文abc         [中文 abc]              [中文 abc]          same

★ **AND jitllm WOULD GET MOONSHOT'S ANSWER FOR FREE, WHICH IS WHAT MAKES THIS
A DECISION RATHER THAN A BUG REPORT.** `SplitParams.CaseRuns` already implements
the case split correctly -- it is what `gpt-4o` and every o200k model here uses
-- so kimi-k2 is `gpt-4o`'s entry plus ONE new property, Han. Matching llama.cpp
would mean deliberately NOT using a splitter this tree already has. RULE 7m's
"choose per divergence": the choice here is the model's own tokenizer, and the
divergence from llama.cpp is the reviewed artefact rather than a surprise.

★★ **AND IT IS BUILT, CONTAINER v25, AND Moonlight RUNS.** `SplitParams.HanRuns`
is the v19 precedent exactly (`CaseRuns`/`SuffixContract`/`PunctSlash` landed
together) and it cost the same things: `pretok`, the generated table,
`tok/bpe.go`'s splitter, `jlm` -- **a container version bump, so every `.jlm`
re-converts** -- and one new predicate. kimi-k2's row is `gpt-4o`'s flag for
flag, **plus HanRuns and minus PunctSlash** (its punctuation run ends `[\r\n]*`
where o200k's is `[\r\n/]*`).

    Moonlight-16B-A3B-Instruct Q4_K_M, six P-cores, host
    "The capital of France is Paris, which is located in the northern part
     of the country. It is one of the most visited cities in the world"
    14.25 tok/s, and 19 of 20 teacher-forced positions agree with llama.cpp
    (the twentieth is a 0.016 tie)

★ **Han IS PINNED FROM THE UCD, WHICH IS THE ONE PLACE genunicode PARTS FROM
llama.cpp -- AND BY ITS OWN ARGUMENT RATHER THAN AGAINST IT.** That generator
takes `unicode-data.cpp` because it is a VERSIONED artefact where llama.cpp's
own generator downloads `UCD/latest` unpinned; `Scripts.txt` is versioned in its
URL, and `unicode-data.cpp` carries no scripts at all. **22 merged ranges,
99,030 codepoints, Unicode 15.1.0** -- and regenerating moved NOTHING: every
existing table is byte-identical, so no token id moved.

    pinned 15.1.0          99,030 codepoints
    the `regex` module    103,351  -- newer CJK extensions, a SUPERSET
    llama.cpp's is_han     CJK Ext A-F only: no Ext G/H/I, no radicals
                           (U+2E80), no U+3005, no compatibility ideographs

The golden's generator reads its Han ranges **out of `tok/unicodedata.go`** so
the two cannot drift, and the corpus carries U+2E80, U+3005, U+F900 and U+2EBF0
-- the four that separate the three definitions.

★ **AND THE ABSENCE HAD BEEN RECORDED AS A REASON, WHICH RULE 11 REFUSES.**
`tok/pretok/table.go` listed `kimi-k2 (moonshotai/Kimi-K2-Base): http 404` among the
names it could not generate. That repo is gone -- but so is the premise: NO
Moonshot repo has ever published a `tokenizer.json`, so no repo name would have
helped. `scripts/gentok`'s `handWritten` map is the answer, and it is
deliberately NOT an escape hatch for a fetch that failed: an entry belongs there
only when there is no tokenizer.json to parse AT ALL, and it carries the
artefact that replaces it. Everything else in that table is still parsed.

## The router's renormalising divisor floor: DeepSeek's sum + 1e-20 (decided)

- ~~the renormalising divisor's floor is the user's call~~ **DECIDED AND CLOSED:
  DeepSeek's `sum + 1e-20`, NOT llama.cpp's clamp.** RULE 7m's
  instruction is to CHOOSE per divergence and write the choice down; leaving it
  meant jitllm guarded the divide with NEITHER, so a selection whose every
  sigmoid underflows -- router logits below about -104 -- divided 0 by 0 and
  took the token's logits non-finite.

  **The two guards differ in a REACHABLE range and not only at zero**, which is
  what decides it:

      sum ~= 0            clamp 0/6.1e-5 = 0    addition 0/1e-20 = 0   identical
      0 < sum < 6.1e-5    clamp DISTORTS         addition divides by ~sum

  1e-20 is below an f32 ulp of any sum above ~1e-13, so it is invisible
  everywhere the clamp would bite -- strictly the less invasive of the two, and
  the one the weights were trained against.

  ★ **IT IS FOUR DEFINITIONS AND ONE NUMBER, ASSERTED.** `internal/oracle`,
  the device IR in `jit_old/gpu/kernels`, `jit_old/cpu`'s Go constant and the WORD AT
  `moeSumFloor` in the block all three host emitters load from.
  `nn.TestRouterFloorMatchesTheKernel` compares all four -- including the block
  word rather than only the constant, because the block is what the kernel
  reads.

  ★ **AND THE FIXTURE THAT WOULD HAVE CAUGHT IT WAS ALREADY THERE, PASSING ON
  A DEGENERATE COMPARE.** `jit_old/cpu`'s top-k gate has an `allZero` case and it
  compared the kernel's NaN divisor against its own reference's NaN **by bit
  pattern**, which matched. RULE 10's "a degenerate oracle is a failure, not a
  pass", in a gate written years before the router it now covers.

  ★ **THE SOFTMAX ARM NEEDS NO FLOOR AND DELIBERATELY DOES NOT GET ONE.** Its
  divisor is a sum of `exp(x - max)` and therefore always at least 1. That is
  also why `lowertest` did not move: the two `expertweights` goldens both pinned
  the SOFTMAX path, so **the arm that cannot underflow was pinned and the arm
  that can was not pinned at all**. `expertweights_sigmoid` and
  `expertweights_sigmoid_scaled` close that, and removing the floor moves both.
## minimax-m3: block selection, with the indexer's key as one more kv head

MiniMax-M3's text model is three things the engine already ran -- MiniMax-M2's
attention (here with a per-head q/k norm), DeepSeek-V3's mixture (dense lead,
sigmoid router with the selection bias, renormalised, a routed scale, one
shared expert) and gpt-oss's clamped SwiGLU, in every FFN -- plus MiniMax
Sparse Attention on every block past the dense lead. Read off transformers'
`MiniMaxM3VLIndexer`, llama.cpp's `minimax-m3.cpp` and `conversion/minimax.py`
and the published config before anything was written. Per selecting block:

    k_idx   = NEOX-rope(RMSNorm(W_k x))                        [D], one per position
    q_idx,g = NEOX-rope(RMSNorm(W_q x)_g), one head per kv group  [G x D]
    score_g[t] = q_idx,g . k_idx,t                          (t <= p)
    block b's rank = max of its positions' scores
    forced: the local_blocks ending at the query's own (+inf)
    keep the top_k blocks; group g's heads attend to their positions only

Every norm is Gemma's (1 + w), baked by llama.cpp's converter; the rotary is
the attention's, on the first rotary_dim of each indexer head. The published
model: 60 blocks, 3 dense, 4 kv heads of 128, an indexer of 4 heads of 128,
blocks of 128, top 16, one local block.

**The key is one more kv head of the cached row.** `Config.KVRowAt` and
`LayerPlan.KVRow` are (NKVHead+1)*HeadDim on a selecting block, so the pages
that hold a position's k hold its indexer key, and paging, relocation, the
prefix cache, device sessions and the batch seam moves carry it with nothing
of their own. The value row's extra head is zero on the host and unwritten on
a device, and nothing reads it. The dense lead's row has no extra head, so the
geometry is per layer (`kvlAt`, and on the device a scratch set of its own,
`geoMSA`, the way Gemma 4's sliding layers have one). The converter refuses an
indexer head that is not one kv group's or not the head's width.

**The host ranks blocks without a max-pool.** The scores are the generated
attention-score kernel over the key head (`attnM`, the selecting blocks' own
stride); the ranking is the sampler's generated ordering (`nn.SampleOrder`)
walked best-first, keeping each new block until top_k are kept, after the
forced ones: a block's first position seen in that walk IS its maximum, so the
order of first appearance is the order by maximum. The selection is a -inf
bias row per kv group, added after the scale (`model/msa.go`).

**The device gathers** (`kernels/msa.go`, `tier/msa.go`): seven basic kernels,
each a loop of depth one -- `MSAScoresPaged` (a thread a (row, group,
position)), `MSABlockMax`, `MSASelect` (a block's rank is the count of blocks
above it, or level and lower), `MSACompact` (the kept blocks in ascending
order), `MSAGatherK`/`MSAGatherV` (each group's kept positions into its own
head of scratch pages) and `MSADesc` -- then the unchanged paged attention over
the gathered pages. Every group keeps the same count, because the query's own
block is forced in and is the only partial one (and, ascending, the last), so
one descriptor a row serves all heads; a local_blocks of 0 would break that
and is refused by name. A chunk takes the decode plan row by row, and a call
over an evicted (streamed) history is refused by name.

`synth-minimaxm3` (scripts/minimaxm3gold.py): transformers'
MiniMaxM3VLForCausalLM, 3 blocks with a dense lead, GQA 4:2 at head_dim 16,
NEOX rotary on 8, 8 experts top-2 with the selection bias randomised, one
shared expert, routed scale 2, an indexer of 2 heads of 16 with **blocks of 4,
the top 2, one local, over a 24-position golden**: from position 8 on every
query reads a subset, and each group its own. MiniMax-M3's own tokenizer;
llama.cpp's converter, reading the checkpoint in MiniMax's own layout (the
script writes it from the class's tensors):

    host vs transformers, decode / prefill at every length / batch   4.503e-12
    device vs host, every block and the head
      cuda:0   decode 3.559e-12, prefill 2.965e-13
    step across three sessions vs each alone   cuda 3.332e-13
    seam moves with three rows in flight       4.7e-12 after, 3 arms
    one-frame paging                           63 page-ins, ids identical
    ForwardBatch under a one-page budget       NMSE 0 against decode
    TestDecodeDoesNotAllocate                  0 on the host and cuda (decode and the step arms)

Every violation lands far past the 1e-9 bound. On the host: attention over
every block 0.64, a top-k one short 0.81, the own block not forced 1.54, one
selection for every group 0.098, a block ranked by its first position 0.35,
the indexer key unnormed 0.092, the query unnormed 0.047, plain SwiGLU 1.69.
In the device session alone: every block 0.67, one short 0.93, two blocks
forced 1.29, the key unnormed 0.10, the query unnormed 0.052.

Found on the way:

- **A binary16 cache flips the selection.** With amd64's default f16 KV the
  host rounded the indexer key while the device's paged K pool is f32, and
  the seam gate read 8.5e-02 at one row and step: a near tie between two
  blocks' maxima, decided the other way. The references keep the key at f32
  (llama.cpp's converter forces it), so an MSA model's host cache defaults to
  f32; `WithKVF16(true)` still forces binary16.
- **A scratch set prepared another set's blocks.** `prepBatch` and
  `prepRagged` walked every block of the device, so the dense lead's batched
  scratch was asked for the mixture blocks' grouped matvecs, refused, and
  reported a stale error from an unrelated call. They walk only the set's own
  blocks now, and the mixture refusal names itself.
- **The llama.cpp gate was comparing against dense attention.**
  `TestGreedyMatchesLlamaCpp/synth-minimaxm3` read 8 of 20 teacher-forced
  positions, with confident disagreements at 12 (margin 1.276), 14 (0.840)
  and 16 (0.516). llama.cpp runs MSA only with flash attention on and
  otherwise attends to every block; the gate offloads to CUDA at `-ngl 99`,
  where `-fa auto` finds no flash-attention kernel for head width 16, logs
  "Flash Attention not supported, set to disabled" and runs the dense graph.
  Measured on "The capital of France is" with the CUDA build: `-ngl 99` at
  the default and `-ngl 0 -fa off` produce one chain, `-ngl 0` at the default
  and `-ngl 99 -fa on` another, and the two part at the fifth generated token
  -- the first query past two blocks of four. Teacher-forced
  along each, transformers (`MiniMaxM3VLForCausalLM` on the same weights,
  rebuilt within 5e-08 of the golden) agrees with the MSA chain at 20 of 20
  and with the dense chain at 8 of 20, with the same three confident misses
  jitllm had (1.28, 0.84, 0.52); jitllm agrees with the MSA chain at 20 of
  20 teacher-forced and walks all 20 greedily. So no authority disagreed: the
  oracle was run as a different model. The gate passes `-fa on` for an MSA
  model and refuses a run whose log still says flash attention was turned
  off; removing the flag on the CUDA build fires that refusal, and removing
  both is the 8 of 20 above. The indexer key's precision played no part: the
  host cache is f32 for an MSA model, as llama.cpp's is.
- **The batched mixture gate compared the row-by-row arm with itself.**
  `TestBatchedMixtureFloatBanksDiscriminate/synth-minimaxm3` read NMSE 0
  with "floatsrc" armed (synth-minimaxm2 0.898), and the unfaulted gate read
  0.000 where every other mixture reads ~1e-8 -- while the stats showed 2
  grouped blocks. The chunk did run grouped, and its answer was thrown away:
  MSA blocks are a scratch set of their own (geoMSA), so the head never
  rides the last run and goes as a head-only call (`GPU.layersCall`'s tail,
  `[3,3)` with the head); the synth's head is float, which the fold cannot
  take, and that path ran "the blocks first" as `Layers(3, 3, nil)` -- an
  empty range with no head, a refusal. The State then re-ran the whole chunk
  a row at a time from the embeddings (prefillSrc's per-row retry), so the
  batched arm's logits were the rows arm's to the bit, every chunk, on every
  device: M3 never prefilled batched with a float head. The head-only call
  skips the blocks now (`lo < hi`), and `State.DeviceRowChunks` counts the
  retry, which the gate demands is 0 on the batched arm -- against the old
  tier it fires with "ran 1 prompt chunks again a row at a time". After:
  floatsrc reads 0.228 on M3, and on cuda:0, vulkan:0 (the RTX) and vulkan:1
  (the Iris Xe) every model in the gate reads 0.13 (kimik3) to 2.14
  (deepseek4) with the retry count at 0 throughout. "downsrc" and "wout"
  read 0 on M3 because they name paths M3's float banks do not take (a
  quantized bank's down, Llama 4's input weighting), not because they were
  thrown away.

Not yet taken: the real 428B against llama.cpp (a Q3_K_M is 197 GB, which
only the V100 box holds; llama.cpp runs MSA only with flash attention on, and
falls back to dense attention without it, which is a different model).
## The modern text group: Ling 2.0, dots.llm1, Phi-3.5-MoE and Apertus

Four families admitted together, priced in `docs/design/model-coverage.md` 4c.
Arch codes 39..42 live in `format/jlm/archmoe2.go` with their own name table;
the converters are `convert/moefamily2.go`; the fixtures are built by
`scripts/moegold.py` with each family's own reference class and every
parameter and buffer randomised, and written to GGUF by llama.cpp's converter.

### Ling 2.0 (bailingmoe2)

Nothing in the graph is new: glm4moe's router with groups switched on, qwen3's
per-head q/k norm, NEOX rotary on half the head, a dense lead, a shared expert.
What took reading:

- **The reference is remote code written against transformers 4.52.**
  `modeling_bailing_moe_v2.py` imports `is_torch_fx_available` and reads
  `ROPE_INIT_FUNCTIONS["default"]`, both gone in transformers 5. Shimming them
  one at a time was refused in favour of the version the code declares: a venv
  with transformers 4.52.3 (a second venv, `BAILING_PY`, torch from the
  main venv's site-packages through a `.pth`), and `moegold.py` re-runs itself
  under it for this fixture. The class runs eager attention with
  `use_cache=False`, its own math throughout.
- **`expert_bias` is a buffer**, zero-initialised: the identity, as DeepSeek's
  was. `perturb` draws it at 0.5, and the selection-bias violation reads
  6.5e-01 because of it.
- **The shared width is three numbers that agree at one shared expert.** The
  class builds one MLP of `moe_intermediate_size * num_shared_experts`; the
  converter writes `moe_shared_expert_intermediate_size` when the config has it;
  llama.cpp's loader multiplies whatever it reads by the count again. A fixture
  with two shared experts converts and runs here and is refused by llama.cpp
  (`blk.1.ffn_gate_shexp.weight has wrong shape; expected 64, 128, got 64, 64`).
  The width is read off the tensor (`ernieSharedWidth`), and the fixture has one
  shared expert, as every published Ling 2.0 does, so llama.cpp is a second
  oracle for it.

synth-bailingmoe2: GQA 4:2, head 16 with 8 rotary dimensions, one dense lead
block, sigmoid top-4 of 16 in 4 groups of which 2 are kept, routed scale 2.5,
F32, bound 1e-9:

    host vs transformers (decode, prefill at every length, the batch)
        amd64 2.840e-12                     M4 3.142e-12
    device vs host, every block and the head placed
        cuda:0 (RTX 3050 Ti)   decode 2.252e-12   prefill 4.732e-08
        vulkan:0               decode 2.736e-12   prefill 6.793e-13
        cuda:6 (V100, sm_70)   decode 2.252e-12   prefill 5.649e-13
        vulkan:6 (V100)        decode 2.736e-12   prefill 6.793e-13
        metal (M4)             decode 3.557e-12   prefill 8.184e-08
    the step across three sessions   2.9e-13 .. 3.8e-13 on all five
    llama.cpp, teacher-forced        20 of 20

Violations, host / device: softmax for sigmoid 1.08 / 0.74 (ungrouped on the
device, whose softmax route forms no group scores and declines a grouped one by
name), no selection bias 0.65 / 0.61, no group limit 0.63 / 0.57, not
renormalised 0.66 / 0.64, no routed scale 0.40 / 0.39, no shared expert 0.74 /
0.72, no q/k norm 1.32 / 1.47, NORM rotary 0.66 / 0.56, rotary over the whole
head 0.85 / 0.50.

The principles: paging under a one-page budget bit-identical (251 page-ins
against 28, amd64 and M4), the batch faulting its blocks in, a layer relocating to another
device, KV pages relocating with the layer, the seam moving under a batch
(4.8e-07 before the first move, 8.7e-07 after, V100), generated code only,
and zero engine allocations per warm decode token and warm step on the host,
CUDA, Vulkan (laptop and V100) and Metal. A mixture's expert pages are a second
page size, so it is not a `principleFixtures` entry
(`TestPagingDoesNotChangeTheAnswer` asks for N frames of one size and a
mixture holds more); `principleMixtures` routes it through every other gate.

Ling-mini-2.0 Q4_K_M (bartowski's GGUF of inclusionAI's checkpoint, 16B-A1.4B,
256 experts in 8 groups of which 4 are kept, top-8, V100 box host): 20 of 20
teacher-forced against llama.cpp, and the free-running chain parts at token 3
on a 0.131 tie.

### dots.llm1 (dots1)

Glm4moe's router with no groups (n_group 1) on qwen3's attention: per-head
q/k RMSNorm before NEOX rotary over the whole head, no biases, two shared
experts as one MLP of twice an expert's width, a dense lead. `Dots1TopkRouter`
hardcodes the sigmoid, so it joins `hardcodesSigmoid` and a stated softmax is
refused. Two things only the fixture found:

- **llama.cpp's dots1 loader is MHA-only.** It creates k and v at
  `n_embd_head_k * n_head` rows, so a GQA fixture is a file it refuses
  (`blk.0.attn_k.weight has wrong shape; expected 64, 64, got 64, 32`). Every
  published dots.llm1 is MHA (32/32), and the fixture is too, so llama.cpp can
  be a second oracle; jitllm runs either.
- **The class can slide a window** on the layers past `max_window_layers`
  when its config states one; every published config states null and
  llama.cpp applies none, so a stated window is refused by name on both
  inputs (the safetensors path checks `layer_types`).

synth-dots1: MHA 4 heads of 16, sigmoid top-2 of 8 with the selection bias,
routed scale 2.5, F32, bound 1e-9:

    host vs transformers (decode, prefill at every length, the batch)
        amd64 4.097e-12                     M4 4.282e-12
    from safetensors (TestSafetensorsMatchTransformers)   1.55e-12
    either input: 46 tensors byte-identical, the config agrees
    device vs host, every block and the head placed
        cuda:0 (RTX 3050 Ti)   decode 1.729e-12   prefill 2.752e-07
        vulkan:0               decode 1.789e-12   prefill 1.301e-12
        cuda:6 (V100, sm_70)   decode 1.729e-12   prefill 2.530e-12
        vulkan:6 (V100)        decode 1.789e-12   prefill 1.301e-12
        metal (M4)             decode 2.275e-12   prefill 8.546e-07
    the step across three sessions   4.5e-13 .. 8.0e-13 on all five
    llama.cpp, teacher-forced        20 of 20 (V100 box)

Violations, host / device: softmax for sigmoid 0.60 / 0.54, no selection bias
1.26 / 1.16, not renormalised 0.14 / 0.14, no routed scale 0.61 / 0.61, no
shared expert 0.77 / 0.78, no q/k norm 1.06 / 1.06, NORM rotary 0.96 / 0.94.
The principles as for Ling: one-page paging bit-identical (144 page-ins
against 11, amd64 and M4), the batch faulting its blocks in, relocation of the
layer and its KV pages, the seam moving under a batch (3.2e-06 after the move,
laptop and V100), generated code only, zero engine allocations per warm decode
token and step on the host, CUDA, Vulkan (laptop and V100) and Metal.

dots.llm1.inst Q4_K_M (bartowski's GGUF in three parts, 142B-A14B, 128 experts,
top-6, 96 GB; 108.7 GB as a container; V100 box host): 18 of 20 teacher-forced
against llama.cpp, the two others inside the tie margin, and the free-running
chain parts at token 7 on a 0.086 tie (" northern" against " north").

### Phi-3.5-MoE (phimoe): sparsemixer, and llama.cpp runs another model

The block is the C6 kit's sequential one (LayerNorm with biases, q/k/v/o
biases, a biased head behind a biased LayerNorm) with phi3's NEOX LongRoPE,
and none of that is new. The router is: transformers' `PhimoeTopKRouter` calls
`sparsemixer`, whose eval path is

    i1 = argmax(s); w1 = softmax(s with every s_j where
                                 (m1 - s_j)/max(|s_j|, m1) > 2*jitter masked)[i1]
    i2 = argmax(s without i1); w2 = the same around m2, i1 masked too

with no renormalisation (vLLM's `phimoe_routing_function` is the same function
and asserts `renormalize is False`). Each weight is near one; two experts at
weight ~1 each, where a renormalised top-2 sums to one.

**llama.cpp departs from the class twice, and the engine matches the class
(RULE 7m).** Its phimoe graph is phi3's, which builds every norm with
`LLM_NORM_RMS` and adds the LayerNorm's bias after it -- no mean is subtracted
-- and calls `build_moe_ffn(..., LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX)` with
renormalisation on. The fixture measures both as violations: the renormalised
softmax top-2 reads 0.85 against transformers, the RMSNorm 2.65. So llama.cpp
is not an oracle for this architecture at all; `TestGreedyMatchesLlamaCpp`
skips the fixture by name, and the real checkpoint is held to transformers'
own `PhimoeDecoderLayer` on the GGUF's dequantized weights
(`scripts/phimoestream.py`, which reproduces the fixture's golden argmax at all
twelve positions).

**Two numbers a GGUF does not carry.** `router_jitter_noise` (0.01 on every
published Phi-3.5-MoE and `PhimoeConfig`'s default) rides the arch code
(`jlm.PhiMoEJitter`). `short_mscale` (1.243163 on Phi-3.5-MoE) is the class's
cos/sin scale; llama.cpp's converter writes its own
`sqrt(1 + ln(max/orig)/ln(orig))` = 1.190 as `rope.scaling.attn_factor`, so the
GGUF path carries the file's number and the two engines agree there while both
part from the class by that factor on the real model. The fixture's
`short_mscale` is the formula's value so that both inputs mean the same model.

**The kernel.** `cpu.EmitSparseMixer` on AVX2, SSE and NEON and
`kernels.ExpertWeights` at `MoERoute.SparseMixer` on the device, behind the
plain top-k at Norm off -- whose weights at Norm off are the two selected
logits exactly (x/1), which is where the weights kernel reads m1 and m2. The
masks are the reference's f32 subtraction and division, so they are its masks
bit for bit; a 0/0 gap (a zero logit beside a zero maximum) is kept, as
torch's `masked_fill` keeps a NaN comparison. Against
`internal/oracle.SparseMixer`, the kernel gates (`jit_old/cpu` and
`jit_old/gpu/backend`) catch every fault the reference can carry: no mask (97 of 98
inputs), the first expert kept in the second softmax (98), the weights
renormalised (98), the clamp dropped from the factor (4 -- it only matters
where |s_j| is below the maximum, a negative maximum).

A random router puts every second expert past the 2% threshold, where each
weight is exactly one and the masks are invisible, so the fixture gives each
FFN norm a large common bias and each router row a shared component along it
(`phimoe_close_router`): the logits sit near one value with a few percent of
spread. Moving the threshold is then visible -- zero (each weight one) reads
0.93, ten times wider 1.18.

synth-phimoe (GQA 4:2, 8 experts, top-2, longrope short factors; F32; bound
1e-9):

    host vs transformers (decode, prefill at every length, the batch)
        amd64 9.163e-13                     M4 7.263e-13
    device vs host, every block and the head placed
        cuda:0 (RTX 3050 Ti)   decode 5.249e-13   prefill 1.232e-07
        vulkan:0               decode 4.182e-13   prefill 2.191e-13
        cuda:6 (V100, sm_70)   decode 5.249e-13   prefill 2.840e-13
        vulkan:6 (V100)        decode 4.182e-13   prefill 2.191e-13
        metal (M4)             decode 3.263e-13   prefill 1.295e-07
    the step across three sessions   0 (bit-identical) on all five
    the SSE tier (TestEveryModelRunsSSE)   runs, generated code only

Violations, host / device: llama.cpp's router 0.85 / 0.85, no mask 1.21 /
1.14, a zero threshold 0.93 / 1.03, a threshold ten times wider 1.18, RMSNorm
for LayerNorm 2.65 / 2.62, no norm biases 2.58, no q/k/v/o biases 1.22 / 1.29,
no head bias 3.2e-02, no LongRoPE short factors 1.18, no LongRoPE magnitude
0.41, NORM rotary 1.36 / 1.40. The principles as for the
group: one-page paging bit-identical (194 page-ins against 18, amd64 and M4),
the batch faulting its blocks in, the layer and its KV pages relocating, the
seam moving under a batch (2.1e-07 after the move), and zero engine
allocations per warm decode token and step on the host, CUDA, Vulkan (laptop
and V100) and Metal.

**The safetensors input.** `PhiMoEForCausalLM` (and transformers' own
`PhimoeForCausalLM` spelling) converts through `convert/hfphimoe.go`: mixtral's
`block_sparse_moe` names with the norm, attention and head biases, the
LongRoPE factor vectors written from config.json as the two
`rope_factors_short/long` tensors convert_hf_to_gguf.py writes, and
`short_mscale` as the cos/sin magnitude. A `router_jitter_noise` other than
the arch's 0.01, or a `num_experts_per_tok` other than two, is refused. The
fixture's two inputs convert to one container byte for byte
(`convert.TestMixtureFamiliesConvertAlikeFromEitherInput`, which now also
holds AttnFactor and the window), and `synth-phimoe-hf` matches transformers
at 4.46e-13 -- 3.5e-04 with `short_mscale` dropped, and the alike gate names
the field.

**The real checkpoints are the two small Phi MoEs, held to transformers on
their own weights** (`scripts/phimoestream.py` for a GGUF, transformers' float
model for safetensors; `TestTeacherForcedAgainstReference`, -tags
jitllmbench), teacher-forced along llama.cpp's chain for "The capital of
France is", V100 box:

    Phi-mini-MoE-instruct Q4_K_M (GGUF)     host 33 of 33, worst NMSE 3.7e-04
                                            cuda 33 of 33, worst NMSE 7.3e-03
    Phi-tiny-MoE-instruct (safetensors)     host 33 of 33, worst NMSE 2.7e-12
                                            cuda 33 of 33, worst NMSE 1.7e-08
    llama.cpp's own chain (Phi-mini)        31 of 32 tokens the reference's
                                            argmax; the other misses it by 5.5

The Q4_K_M row's NMSE is the 8-bit activation of a quantized matvec; the
safetensors row runs f32 activations and is exact. Phi-tiny-MoE has heads of
128 on a 4096-wide model (16 x 128 = 2048), stated by `head_dim` in
config.json; llama.cpp's phimoe converter writes no attention.key_length, so
its published GGUF implies heads of 256 while its q projection has 2048 rows.
llama.cpp refuses to load that file, and so does this converter now, naming the
safetensors input -- before, it converted, decoded the first token wrong
(NMSE 0.42) and panicked in the attention's output add
(`convert.TestPhimoeRefusesAHeadWidthTheFileDoesNotState`).

### Apertus: xIELU, a parameterised activation, and llama 3's rotary from safetensors

Apertus is llama's block with qwen3's per-head q/k RMSNorm (before NEOX
rotary, with llama 3's per-pair factors) and an UNGATED MLP whose activation,
transformers' `XIELUActivation`, carries four numbers per layer:

    alpha_p = softplus(a_p)       alpha_n = beta + softplus(a_n)
    x > 0:   alpha_p*x*x + beta*x
    x <= 0:  (expm1(min(x, eps)) - x)*alpha_n + beta*x

`a_p` and `a_n` are parameters; `beta` and `eps` are buffers. llama.cpp's
converter writes the raw four as per-layer arrays under bare `xielu.*` keys
(no architecture prefix), and its graph applies the softplus when it builds.
Here the converter folds the softplus once and each block carries the
effective four as `jlm.RoleXIELU` (role 500), a per-block vector: the device
uploads them with the block, relocation moves them, and the arch code
(`jlm.ArchApertus`, 39) says which activation reads them. Both inputs fold
through one function (`convert.xieluTensor`), so they write the same bytes.

**The kernel is one generated activation on every tier with its numbers in a
buffer**: `cpu.EmitXIELU` on AVX2, SSE and NEON, `kernels.XIELU` in the device
IR. `expm1(m) - x` is `g(m) + (m - x)` with `g(m) = expm1(m) - m` a degree-8
Taylor polynomial on [-1/2, 0] and `exp(m) - 1 - m` below it: computed as
`exp(m) - 1 - m` in f32 near zero, every digit of g cancels (the kernel gate
names that form as a fault and catches it on 30 inputs). Against
`internal/oracle.XIELU`, a float64 expm1, the host kernels hold 4e-7 of each
term at 13 ragged widths with a guard past the end, and the device 2e-6 on
CUDA and Vulkan (worst 0.05 of the bound); eps ignored, the alphas exchanged
and beta*x dropped each fail both.

**The safetensors input needed two pieces the HuggingFace path did not have.**
llama 3's rotary scaling was refused for every class; `hfLlama3Consume` reads
it and `llama3Freqs` writes the `rope_freqs` vector convert_hf_to_gguf.py
writes, every step in float32 as its torch tensors run it, so the factors are
the GGUF's to the bit. And the four xIELU numbers are scalars whose values
the converter needs: `hfArch.gather` reads tensors before the identify loop
and folds them (`apertusGather`). `synth-apertus` converts from either input to
the same container, every Config field and 37 tensors byte-identical
(`convert.TestDenseSafetensorsMatchesItsGGUF`).

synth-apertus (scripts/c6gold.py: transformers' own ApertusForCausalLM, GQA
4:2, llama 3 scaling at an original context of 64 so the eight pairs of a
16-wide head fall in all three bands, each layer's beta and eps drawn as
buffers; F32; bound 1e-9), laptop:

    host vs transformers (decode, prefill at every length, the batch)  2.270e-12
    safetensors vs transformers                                        1.58e-12
    cuda:0    decode 2.303e-12   prefill 1.046e-07
    vulkan:0  decode 2.913e-12   prefill 8.776e-13
    llama.cpp, teacher-forced                                          20 of 20
    the SSE tier, generated code only (TestEveryModelRunsSSE)          runs

Violations, host / device: eps ignored 2.9e-04 / 3.0e-04, the alphas
exchanged 1.3e-01, beta*x dropped 4.9e-01 / 5.2e-01, alpha_p without the
softplus 1.96 / 1.88, every block running block 0's numbers 1.3e-02 / 1.4e-02,
no llama 3 factors 4.8e-02 / 5.1e-02, NORM rotary 1.49 / 1.41, no q/k norm
1.78 / 1.65. The principles: paging under a one-frame budget, the batch
faulting its blocks in, the layer and its KV pages relocating, the step across
three sessions, generated code only, and zero engine allocations per warm
decode token and step, host and CUDA and Vulkan.

Apertus-8B-Instruct-2509 Q4_K_M: 19 of 20 teacher-forced against llama.cpp
(V100 box, host), the other a 0.185 tie.

**Token 11 of "The capital of France is" is a near-tie no two engines settle
alike, recorded in the gate (`recordedNearTies`).** After "... the country's
largest city." the top of every distribution is flat and each engine's
rounding picks a different word:

    llama.cpp, Xeon E5-2680 v4 CPU      " region"  (region -2.687, cities -2.716,
                                                    thus -2.967, urban -3.058 logprob)
    llama.cpp b10825, M4 CPU            " urban"
    transformers, the bf16 checkpoint   " not"     (not 18.40, its 18.25, as 17.00,
     read in float32                                geographically 16.77, urban 16.69;
                                                    " Paris" outside the top five)
    jitllm, Xeon (f16 cache)            " thus"    (thus 18.65, urban 18.47, Paris 18.44)
    jitllm, M4 (f32 cache)              " Paris"   (Paris 18.64, cities 17.84 -- 0.80)
    jitllm, M4, cache forced to f16     " best"    (0.87)

The spread is the model's, not one engine's: its residual stream reaches 2.6e4,
and jitllm's Xeon and M4 runs -- 2.9e-06 apart at layer 0's attention once the
M4's cache is also binary16 -- are 7.4e-04 apart after layer 0's MLP, where the
int8 activations of the two hosts' kernels round differently, and 16% apart at
the output norm. Every other position of the prompt agrees with llama.cpp or
ties within the gate's 0.5 on both hosts (18 of 20 teacher-forced on each).

## BitNet b1.58: ternary weights, int8 activations per token

BitNet b1.58 2B-4T (transformers' `BitNetForCausalLM`) is llama's block with
two more RMSNorms -- one over the attention output before wo
(`attn_sub_norm`), one over the gated product before down (`ffn_sub_norm`) --
squared ReLU on the gate, NEOX rotary and a tied embedding. Every projection
is `AutoBitLinear`: the activation is quantized to int8 by its absmax over the
whole row, per token, and the weight is ternary, {-1, 0, 1} times one scale per
tensor. The arch is `jlm.ArchBitNet` (56); the inner norms are
`jlm.RoleAttnSubNorm` and `RoleFFNSubNorm`, uploaded with the block on every
device and moved with it.

**The weight type.** One ternary type on TQ2_0's block (256 codes, one scale):
the codes sit in a 2-bit plane, four a byte a quarter sub-block apart
(`kernels.Fields`), with a bias of one, and the dot is ternary x int8 -- an
unsigned dot of the codes with the activation sum subtracted -- on AVX2, VNNI,
SSE, NEON and the device IR, never dequantized to f32. The scale's precision is
a parameter of the type, not a rounding: `TQ2_0` carries an f16 scale,
`TQ2_F32` (an engine type, `quant.EngineTypes`) an f32 one. TQ1_0 and
bitnet.cpp's I2_S (type 36: 2-bit codes in 128-element groups, one f32 scale in
a 32-byte tail) re-pack into it exactly; the decoders are held to gguf-py's
(`quant.TestTernaryMatchesGGUFPy`).

Why the f32 form exists: Microsoft's I2_S file stores each tensor's scale as an
f32 mean that is not f16-exact (worst relative error 4.6e-4). Rounding it to
f16 moved the fixture's logits by 4e-4..3.6e-3 in NMSE against the f32 golden,
and transformers fed the same rounded scales reproduced exactly that number. So
a scale that is f16-exact (the release's bf16 `weight_scale`, the F16 GGUF's)
converts to TQ2_0 and any other to TQ2_F32.

**Three inputs, one graph.** The release (uint8 codes packed four a byte plus a
bf16 `weight_scale` the output is multiplied by) converts byte for byte; the
bf16 master checkpoint is ternarized at conversion as the class does (absmean,
in bf16 under its torch_dtype: 98-99% of each projection agrees with the
release's codes, so it has its own golden); a GGUF from bitnet.cpp's converter
(I2_S, F16 or F32) folds each tensor's scale in. llama.cpp's converter cannot
write this class at all: it maps neither inner norm.

**Activations per token.** transformers quantizes each row by its own absmax.
The engine's int8 activation quantizer took its scale per 32 elements; a
session of a TokenAct arch widens the window to the widest row
(`nn.WithActWindow`), the host clamps it to the row, and the device quantizes a
window wider than one 128-thread group in a loop (`kernels.quantizeWide`).
llama.cpp's TQ2_0 dot reads its activations as q8_K, one scale per 256.

**Rounding-boundary flips are the gate's noise floor.** An activation within a
few ulps of an int8 rounding boundary rounds the other way under another
summation order, and the logits after that one step move: synth-bitnet-i2s
(F16 embedding) reads 9.4e-5 at its worst position with 17 of 44 positions
exact. The fixture gates take NMSE 2e-3 with exact positions at 1e-9, and every
violation must read at least ten times the bound.

Fixtures (scripts/bitnetgold.py: transformers' own BitNetForCausalLM with every
parameter and buffer drawn, the packed release form and the online latent
form, plus bitnet.cpp's converter's I2_S and F16 GGUFs of the latter), laptop,
host, decode and prefill at every length and the batch:

    synth-bitnet-hf (TQ2_0)            2.9e-13
    synth-bitnet-online-hf (TQ2_F32)   4.1e-13
    synth-bitnet-f16.gguf (TQ2_0)      7.0e-13
    synth-bitnet-i2s.gguf (TQ2_F32)    9.4e-5 (an int8 boundary flip)

Violations, host: no attention inner norm 0.92, no FFN inner norm 1.29, SiLU
for squared ReLU 0.51, NORM rotary 2.22, activations per 32 rather than per
token 3.5e-2, every weight's scale one 1.13, wo's scale inverted 1.74. On CUDA
and Vulkan with every block placed: decode ~1e-13; prefill with f32 attention
~1e-13; the default CUDA prefill, whose QK^T tiles are binary16, 5e-3 to
9.5e-3 (pinned to f32 attention it reads 5e-14, so the gap is the tiles, not
the ternary path); violations 0.38 to 2.2, and per-32 4e-2 to 6.7e-2.

**Not llama.cpp's graph.** llama.cpp's `bitnet` architecture is 1bitLLM's
reproduction -- a SiLU gate, NORM rotary -- and bitnet.cpp's `bitnet-b1.58`
runs SiLU too. transformers on the release reads NLL 1.70 with squared ReLU and
2.51 with SiLU, so the class is the oracle and `bitnet` is not converted. The
real checkpoints are held to transformers on their own weights
(`scripts/bitnetreal.py`), teacher-forced through three prompts of 32 greedy
tokens.

Teacher-forced on the 2B-4T, three prompts of 32 tokens (214 positions),
laptop (CUDA and Vulkan on the RTX 3050 Ti, every block placed unless said):

    the release (TQ2_0)       host 212/214, 0 confident flips, NMSE 1.5e-2
                              cuda 212/214 (5.5e-2), vulkan 214/214 (8.2e-2)
    the bf16 master (TQ2_0)   host 206/214, cuda 208/214, vulkan 205/214;
                              0 confident flips, every parting position a tie
    I2_S GGUF (TQ2_F32)       host 212/214 (2.1e-2), cuda 213/214 with one
                              block placed, 214/214 with three

The free-running continuations match the reference's token for token on two of
the three prompts; the third parts at a tie and stays fluent ("Paris. Paris is
a city in the Île-de-France region..."). Microsoft's GGUF states no
`tokenizer.ggml.pre`, so its tokenizer splits "def fibonacci(n):\n" as
llama.cpp's default does and not as the release's tokenizer.json; the gate
logs that and runs the golden's ids.

## Gemma 4 vision: clamped matrices, a pooling head, and image rows bidirectional on half the layers

Gemma 4's tower is an RMSNorm ViT at the picture's own size, and it is the
gemma kit on patches: sandwich RMSNorms, a per-head q and k norm, the
weightless v norm, scores at scale one, a gated GELU MLP. What is its own,
each read off Gemma4VisionModel and Gemma4ImageProcessor:

    the resize        get_aspect_ratio_preserving_size: the largest size at
                      the picture's aspect with whole 3x3 groups of 16-pixel
                      patches and at most 280 x 9 patches -- the processor's
                      max_soft_tokens, which no mmproj states -- Pillow
                      BICUBIC, pixels to 2x - 1
    the positions     two learned tables, the column's row of the first and
                      the row's of the second
    the rotary        each half of a head turned NEOX-paired on its own, the
                      column the first half; convert.halfNeoxToNeox swaps each
                      head's middle quarters of q and k, which makes it one
                      whole-head NEOX table (Qwen-VL's) and needs no kernel
    the clamps        every matrix's input and output clamped to bounds the
                      checkpoint carries (Gemma4ClippableLinear) on E2B and
                      E4B: 28 floats a block (RoleVClamp), a generated clamp
                      on the host (nn.ClampRange32JIT) and the device
                      (kernels.ClampAt)
    the head          the 3x3 average pool, times sqrt(NEmbd), (x - std_bias)
                      * std_scale on the 26B and 31B, an unweighted RMSNorm,
                      one matrix

and on the TEXT side, the 26B's and 31B's `use_bidirectional_attention:
"vision"`: a picture's soft tokens see each other on the SLIDING layers and
stay causal on the global ones. `Config.BidirSWA` carries it on the host
(prefill's key count), `nn.LayerPlan.BidirOff` on the device. The GGUF states
neither the flag nor its absence; the E-models (which leave it unset) are the
ones with per-layer embeddings, so the engine reads it off PLEDim.

Wiring the device arm found the tier refusing the text head: PrepHead's call
is an empty range at the block past the text model's last, which with a tower
behind it is a NON-CAUSAL block, and the non-causal check read that block's
flag and refused a head. The check now asks only of a non-empty range.

The fixture is transformers' own Gemma4ForConditionalGeneration at toy size
(`scripts/gemma4visiongold.py`, both GGUFs by llama.cpp's converter) with the
26B's text features (bidirectional "vision", sliding and global layers) and
BOTH heads' features at once -- clipped linears at the 5th and 95th
percentiles of each matrix's activations, and the standardisation -- over a
192x144 picture resized up to 57x42 patches, 266 soft tokens:

    the processor's pixels                NMSE 5.0e-16
    the rows at the engine's arithmetic   NMSE 1.1e-05 (f32 tower 1.2e-04,
                                          where the arithmetic alone moves
                                          transformers by 1.2e-04)
    violations     no clamps 4.3e-02, no pool 7.9e-02, no standardisation
                   4.3e-02, no v norm 3.6e-03, the row turning the first half
                   1.0e-01, scores at 1/sqrt(head_dim) 5.8e-02
    end to end from the pixels            worst logit NMSE 5.7e-03; one row's
                                          argmax parts, a tie of 0.081 in
                                          the toy model, named and not
                                          counted (a flip past tieMargin
                                          fails)
    the mask       causal everywhere 1.46, bidirectional on every layer 0.20
    every block (text and tower) on       3.0e-05 / 4.7e-11 against the
    cuda:0 / vulkan:0                     host; every layer bidirectional
                                          0.20 on both

Under the principles: one vision page of 68 KiB against a 3-page tower (2
evictions) bit-identical, 0 allocations after a picture on the host, CUDA and
Vulkan, a second turn restoring 288 of 316 positions past the picture with no
tower block, a warm encode allocating nothing.

The real Gemma 4 E2B (Q4_K_M text, BF16 mmproj with its audio encoder, one
container) against transformers' own Gemma4VisionModel and
Gemma4MultimodalEmbedder in float32 on the checkpoint's tower
(`scripts/gemma4real.py`, only the tower's tensors fetched), on the photo,
640x488 resized UP to 912x672 (57x42 patches, 266 soft tokens):

    the processor's pixels               NMSE 1.1e-15
    the 266 rows, host                   NMSE 9.2e-03 -- 16 blocks of Q8_0
                                         weights and int8 activations, every
                                         block's own clamps binding
    without the checkpoint's clamps      0.55
    every block on cuda:0 / vulkan:0     4.0e-03 / 4.6e-03 against the host,
                                         8.1e-03 / 6.5e-03 against transformers

and from its own chat template it reads the photo as *"a scan of an old
newspaper front page from july 21, 1969, featuring a large headline about
astronauts landing on the moon"*, and a blank picture of the same size as
*"almost entirely white"*.

The alloc gate's window had to move for this family: its 282-position prompt
put a 96-token warm-up's window across the device's 128-position score grain,
which re-captures the launch graph (15 allocations, once a grain), and across
the host's 256-position KV page. The window now starts at a fixed offset into
a page whatever the prompt's length, decodeAllocs' rule with the prompt in
front of it.

## Phi-4-reasoning-vision: SigLIP2 NaFlex at the picture's own size

llama.cpp's `phi4` projector is Phi-4-reasoning-vision-15B (NOT
Phi-4-multimodal-instruct, which no mmproj exists for): SigLIP2's NaFlex tower,
llava's two matrices and a GELU over every patch, phi3 text. What is its own,
read off the checkpoint's processing_phi4_visionr.py and transformers'
Siglip2VisionEmbeddings:

    the resize       Siglip2ImageProcessorNoUpscale: the picture's own patch
                     count clamped to the file's image_min_pixels and
                     image_max_pixels (256..3600 patches of 16), through
                     get_image_size_for_max_num_patches -- which an in-range
                     picture goes through too, so its sides become whole
                     patches -- and Pillow BILINEAR
    the positions    the 16x16 table resized to each grid: F.interpolate,
                     bilinear, align_corners=False, antialias=True, in float.
                     The taps are imageproc.go's (aaFloats, the float half of
                     aaWeights); the sums are generated axpys over whole rows
                     of channels, kept for the last grid
    the depth        hidden_states[-2]: llama.cpp's converter writes one block
                     fewer and no post-LayerNorm
    the text model   phi3 whose GGUF states attention.sliding_window 0 (the
                     config's null): a STATED zero is full attention, and only
                     an absent key falls to phi3's defaults

The pixel bounds ride the vision section as an optional tail after the
windows' (jlm.Vision.MinPixels/MaxPixels), written only when set, so every
other tower's section is the bytes it was.

The fixture is transformers' own Siglip2VisionModel read at
hidden_states[-2], the mlp2x_gelu projector and Phi3ForCausalLM -- which is
what the checkpoint's Phi4ForCausalLMV is -- every parameter randomised, both
GGUFs by llama.cpp's converter (`scripts/phi4visiongold.py`), a 200x136
picture resized UP to 19x13 patches:

    the processor's size, 14 pictures    every one the processor's
    the table resize, 7 grids            within 3e-07 of F.interpolate
                                         (two of them reductions)
    the processor's pixels               NMSE 5.3e-16; BICUBIC 4.0e-05
    the rows at the engine's arithmetic  NMSE 1.8e-04 (f32 tower 3.5e-04)
    the projector alone                  1.3e-13 on transformers' tower rows
    violations     the table at the grid's transpose 8.3e-03, the table not
                   resized 1.5e-02 -- held to ten times the clean arm, the
                   table being a small term beside the patch rows here as in
                   the checkpoint
    end to end from the pixels           worst logit NMSE 2.1e-03, 0 flips
    every block on cuda:0 / vulkan:0     text 1.8e-06 / 8.3e-12, tower
                                         1.3e-04 / 9.5e-08 against the host

Under the principles: one 56 KiB vision page against a 2-page tower (1
eviction) bit-identical, 0 allocations after a picture on the host, CUDA and
Vulkan, a second turn restoring 688 of 718 positions past the picture with no
tower block, a warm encode allocating nothing.

The real Phi-4-reasoning-vision-15B (Q4_K_M text, f16 mmproj, one container)
against the checkpoint's own tower and projector in float32:

    | picture | grid | pixels | tower | projector | cuda:0 / vulkan:0 vs transformers |
    |---|---|---|---|---|---|
    | the photo, 896x448 | 56x28 | 1.1e-15 | 1.4e-03 | 3.7e-04 | 4.2e-04 / 3.9e-04 |
    | the square | 24x24 | 1.2e-16 | 3.7e-03 | 8.7e-04 | 8.2e-04 / 7.8e-04 |

with the table at the grid's transpose at 0.77, and greedy decoding from its
own chat template reads *"a front page of the new york times newspaper
reporting on the moon landing."* -- and on a blank picture of the same size
*"a plain white background with no visible elements or features."* The device
arms run at the picture's own ceiling: 3600 patches of scratch beside the
text model's share do not fit a 4 GB card, and the placement declines by
budget, as it should.

## deepseek4: hyper-connections, and entries that outlive the window

DeepSeek V4's text model, read off transformers' `modeling_deepseek_v4.py`,
llama.cpp's `src/models/deepseek4.cpp` and its converter before anything was
written. Per block, around each of the attention and the mixture:

    flat  = the row's HCMult streams side by side, RMSNormed with no weight
    mixes = fn . flat                                       (2+H)*H of them, H = 4
    pre   = sigmoid(mixes*s0 + b) + eps                     H collapse weights
    post  = 2*sigmoid(mixes*s1 + b)                         H placements
    comb  = softmax_rows(mixes*s2 + b) + eps, divided by its column sums,
            then iters-1 times by its row sums and its column sums (each + eps)
    in    = sum_k pre_k x_k, RMSNormed by the sublayer's norm
    x'_k  = post_k * sublayer(in) + sum_j comb[j][k] x_j    (comb TRANSPOSED)

and at the head the collapse alone, from fn/base/scale of its own. The
attention: q = q_b(rmsnorm_w(q_a(in))), each head RMSNormed with no weight; ONE
key head, rmsnorm_w(kv(in)), which is also the value; both rotated on the LAST
NRot dimensions of each head in adjacent pairs; a sliding window on every
block, the heads' sinks in one softmax, the output rotated back at the query's
position, then a grouped low-rank projection (OGroups sheets of OLoraRank,
then o_b). A compressed block's query also reads ENTRIES: entry w pools
positions [w*rate, (w+1)*rate) of two projections of the block's input (kv
and a gate with a position bias added) by a softmax over the positions per
channel, is normed and rotated at w*rate on the compressed blocks' rotary
(compress_rope_freq_base with YaRN), and is visible to a query at t once
w < (t+1)/rate. An HCA block reads every visible entry; a CSA block's
projections are two heads wide, its entry pools the window before's first
head and this window's second (window 0 has no window before), and the query
reads the lightning indexer's top-k of the visible entries, scored by
sum_h w_h relu(q_h . k) against the entries of the indexer's own compressor.
The sliding blocks rotate at rope_theta, plainly.

The mixture: sqrt(softplus) gates, the selection bias, renormalised, a routed
scale and one shared expert, every SwiGLU clamped (silu(min(g, 10)) *
clamp(u, -10, 10)); the first hash_layer_count blocks select their experts
from the token's row of a frozen table (tid2eid) while the weights stay the
gate's scores.

**Where it lives.** A position's cached row is its key and, on a compressed
block, the compressor's projections of that position (the gate with its
position bias already added; on CSA the indexer compressor's too), so the
window's own pages hold every input the compressor still has to fold -- the
converter refuses a window that does not cover the HCA rate and two CSA
windows. The row is one latent head that is also the value, as MLA's, and is
float32 on the host by default. The entries are a second paged history per
compressed block, indexed by entry (`kvCache.ent`, `ds4kv.go`; on a device a
pool of its own beside the rows', `tier.entKey`), never released by the
window, since every later query may read them: relocation, the step, the
batch seam moves and the prefix cache carry them beside the rows
(`nn.EntDevice`).

**The streams are one stream-major residual**, as gemma3n's are, so the
seam, paging, the step and the batch move them as one slice. The mixer and
the compressor's pool are new generated kernels on AVX2, SSE and arm64
(`cpu.EmitHCMix`, `cpu.EmitColPool`), the clamped SwiGLU and sqrt(softplus)
are new activation kinds on all three and the device (`ActSwiGLUClamp`,
`ActSqrtSoftplus`); everything else on the host is a kernel the engine
already had (the rotary turns a head's tail by being handed the run that
starts at its nope width).

**On the device** twelve basic kernels (`kernels/ds4.go`, each a loop of depth
one at most) do what the generic block does not: the streams flattened, the
mixer (the 4x4 Sinkhorn unrolled, one thread a row), the collapse and the
placement; the tail rotary and its inverse; the position bias; the pool over
a closing window's paged rows; the entry write (a row that closes nothing
writes the dummy row); the entries' descriptors for the indexer (whose
scores and selection are DeepSeek V3.2's kernels over the entries' pool); the
gather of each row's window rows and entries into scratch pages; their
descriptors; and the router's per-row bias plane. The attention itself is the
unchanged paged attention over the gathered pages, as a latent row with the
heads' sinks. The grouped output is one indexed matvec with a slot a group
per row (its R-row twins compiled outside the session). The router gates
with sqrt(softplus) and selects through the per-row plane on every block
(`MoERoute.RowBias`): the block's selection bias copied to each row, or the
table's experts of the row's token (ids handed over by `nn.TokenDevice`). The
entries are rotated at their own positions, so a call needs the device's
rotary table, and the history is never evicted (a streamed pass cannot serve
the gathers): both are refusals by name. The head reads the streams'
collapse, which the host takes.

`synth-deepseek4` (scripts/ds4gold.py): transformers' DeepseekV4ForCausalLM,
4 blocks -- CSA (rate 4), HCA (rate 8), sliding, CSA -- under a window of 8,
4 heads of 32 rotating 16, a query latent of 32, two output groups of 32, 8
experts top-2 with a shared one, an indexer of 8 heads of 32 keeping the top
3, the first two blocks hash-routed, YaRN at the compressed blocks' own base;
every parameter and every weight buffer randomised (the selection bias, the
token table, the hyper-connections' base and scale, the position biases and
sinks). Over the golden's 32 positions the window slides, both kinds compress,
and from position 15 each CSA query reads a subset of its entries.
DeepSeek-V4-Flash's own tokenizer; llama.cpp's converter reading the
checkpoint in DeepSeek's own layout:

    host vs transformers, decode / prefill at every length / batch   3.208e-10
    device vs host, every block and the head
      cuda:0   decode 1.491e-10, prefill 7.994e-12
    the prompt as one batched submission, a decode token on the device
      cuda:0   4 attention halves, 8 mixes, 2 selections, 4 grouped mixtures
    step across three sessions vs each alone   cuda 4.832e-14
    seam moves with three rows in flight       3.3e-11 before, 1.5e-10 after (worst arm)
    one-frame paging                           128 page-ins, ids identical
    ForwardBatch under a one-page budget       NMSE 0 against decode
    batched prefill vs row by row, float banks NMSE 0
    windows: host peak 4 pages a windowed layer (bound 5, the control 48);
      cuda releases 20 pages behind the windows, logits bit-identical
    prefix cache, 700 shared of a 703/705-token pair   688 restored, host and cuda
    TestDecodeDoesNotAllocate                  0 on the host and cuda (decode and the step arms)

The 3.2e-10 is above the other flagships' ~1e-12 and under the 1e-9 bound;
which term carries it is not attributed.

Every violation lands far past the bound. On the host: the mixer softmaxed and
not Sinkhorn-normalised 1.06, untransposed 1.09, three rounds 0.67, the head's
collapse a mean 0.33, the output left rotated 2.08, the window alone 2.76, a
CSA entry without the window before 2.72, every CSA entry attended 2.41, the
top-k one short 1.73, the compressed blocks at the sliding rotary 2.08, the
hash blocks routed by score 2.92, no sinks 0.69, a window of 7 3.19, a sigmoid
gate 1.38, plain SwiGLU 2.11. In the device session alone, every one of those
but the head's (the host's): 1.05, 1.03, 0.72, 2.00, 2.70, 2.32, 2.19, 1.66,
2.24, 2.80, 0.74, 3.34, 1.30, 2.03. Two probes showed the entries' own paths
load-bearing: entries that never migrate fail the seam moves, and a restore
that claims entries it did not fault gives other tokens.

Where the references part (RULE 7m), and the choice:

- **The indexer's scale.** llama.cpp multiplies the heads' weights by
  1/sqrt(IdxHeadDim*IdxHeads) once; transformers scales the dot and the
  weights apart. A ReLU of a positively scaled dot is the scaled ReLU, so the
  two are one function; the engine takes llama.cpp's single scale.
- **Which blocks compress how.** llama.cpp names a block CSA or HCA by its
  ratio against two constants (4 and 128); transformers by layer_types, with a
  rate per type. The container carries a kind per block and a rate per kind,
  and the converter tells the kinds apart by their tensors (an indexer and a
  two-head compressor is CSA), so a V4 with other rates converts.
- **The gate.** transformers would honour softmax and sigmoid scoring through
  ACT2FN; llama.cpp's loader refuses anything but sqrt(softplus), and no
  published V4 states another. Only sqrt(softplus) converts; a file stating
  expert_weights_norm false is refused (both reference routers renormalise
  unconditionally), as is a SwiGLU limit other than 10 and a token table that
  names an expert twice for one token.
- **The fixture's experts.** The published routed experts are MXFP4, and the
  converter only repacks them; the fixture draws them as MXFP4 codes and
  scales, transformers runs their exact values, and the GGUF carries those
  values as F32 banks. MXFP4 kernels read int8 activations, which move logits
  by ~5e-3 (gpt-oss's bound) and would flip the hash and indexer selections
  the gate is about; the engine's MXFP4 path is gpt-oss's, gated there.
- **The indexer's ties.** With two indexer heads, entries whose every head's
  ReLU was zero tied at exactly zero at the selection boundary, where torch's
  top-k and the engine's order differ; the fixture has eight heads, as
  DeepSeek V3.2's found.

Found on the way:

- **A device call over several streams read one of them back.** The
  readback kept an AltUp-only case: any other model with more than one
  stream came home with nrow*n_embd floats of stream 0 and the rest of the
  residual stale. Decode never went that way (its residual is read in place),
  so only a batch step or a seam showed it, at 0.4-1.8; it is keyed on the
  stream count now. The chunk-cutting refusal beside it was AltUp-only too.
- **A prefill a device would not batch demoted every block of a model with
  streams.** The row-by-row retry skipped stream models outright, since a
  row alone is not a slice of the stream-major chunk; it gathers the row's
  streams into a residual of their own and puts them back (`rowStreams`), for
  AltUp's models as much as V4's, and hands the device each row's token.
- **The step's one-row head and the batch's head assumed AltUp was the only
  host head**; they ask `Config.streamHead` now.

★ **A HASH-ROUTED BLOCK NEEDS THE ROW'S TOKEN ID, AND THE MIXED PROMPT LOST
EVERY ONE.** `TestPrefillMixedMatchesPrefill/synth-deepseek4` and
`TestForwardEmbdMatchesForward/synth-deepseek4` were red with "block 0 routes
by token id, and a supplied embedding has none". Two things sat behind it:

- **The refusal is the references' semantics, and stays.** transformers'
  `DeepseekV4HashRouter` indexes `tid2eid[input_ids]`, which a call with
  `inputs_embeds` alone does not give it, and llama.cpp's deepseek4 builder
  gathers the table at the batch's `t_inp_tokens`, which an embedding batch
  does not have. Neither can route a row with no id, so a supplied embedding
  with none is refused by name, on the host (`ds4FFN`) and on the device
  (`ds4Stage`). V4 has no vision tower, so no picture row reaches it.
- **But a mixed prompt passed no id at all**, not even for its token spans:
  `prefillSrc` took the row ids from the prompt, which a mixed prompt does not
  have, and only Gemma 4's per-layer embeddings read the spans' tokens
  (`pleTok`). That table is now every row's id for anything that reads the
  token past the embedding (`State.spanTok`, -1 for a row with none; a
  per-layer embedding still takes token 0's for such a row), and the chunk
  loop routes V4's host and device rows by it. An embedding span may name its
  rows' ids (`Span{Embd, Tokens}`, one per row, not looked up), which is how a
  caller supplying embeddings of known tokens runs them through a hash block;
  a picture carrying tokens is still refused.

The two gates now hold V4 to both halves: an id-less embedding row is refused
naming the block (`refusesIDless`, on a State of its own), and the rows naming
their ids reach the logits bit for bit -- the mixed prompt as three spans, and
ForwardEmbd's one-row decode as one-row spans. Named by the wrong ids (each
id + 1) the rows route elsewhere and the logits move, which the mixed gate
asserts. Taking the span's ids out of `tokensOf`, or the chunk loop's read of
them, turns both gates red with the refusal. Every other model's mixed-prefill
and ForwardEmbd gate is unchanged (104 models each, all green).

Not built: V4's multi-token-prediction blocks, whose hyper-connections the
draft path's ordinary block would not run (`Speculate` refuses a V4 by name).
Not taken: no published DeepSeek V4 is held by any box here, so the real model
is neither converted nor compared against llama.cpp.

## Gemma 3n vision: MobileNet-V5, a tower made of convolutions

Gemma 3n's tower is timm's `mobilenetv5_300m_enc`, and it is the first tower
here that is not a transformer: its blocks are convolutions over a grid whose
shape changes from stage to stage. It still runs as blocks of the one model
(`m.layers[NBlocks:]`), paged, placed and relocated as any block is; what a
block IS differs. Read off timm's mobilenetv5.py, transformers'
Gemma3nVisionModel / Gemma3nMultimodalEmbedder and SiglipImageProcessorFast:

    the resize        768x768 squashed, antialiased bilinear, [0, 1]
    the stem          a 3x3 stride-2 convolution, its norm and GELU (tanh)
    the blocks        edge residuals (a full 3x3 convolution, a pointwise one),
                      universal inverted residuals (an optional depthwise
                      start, a pointwise expansion, an optional depthwise
                      middle carrying the stride, a pointwise projection) and
                      multi-query attention whose key and value may be a
                      stride-2 depthwise downsample; every norm an RMSNorm
                      over channels, every layer scale folded into the norm
                      or the projection at conversion
    the padding       TF SAME: the smaller half before the first sample
    the fusion        the last block of the second-to-last stage and the final
                      block, the coarser upsampled nearest and concatenated,
                      an expansion, a projection, an average pool to 16x16
                      and a norm
    the embedder      x sqrt(2048), its norm, one matrix, an unweighted norm

A full convolution is a matmul over an im2col tile, its weight written
tap-major (ky, kx, c) at conversion and k padded to a whole block; a
depthwise one is a generated row kernel on the host (`cpu.EmitDWConv`, AVX2,
SSE and NEON) and a generated kernel on the device (`kernels.DWConv`). On a
device a block is a PROGRAM: a list of launches over named buffers, built
and compiled when the block is placed (`jit_old/gpu/tier/conv.go`), the matrices
in the block's ordinary slots so the pager pages it as it pages any block,
and the buffers between launches one scratch every convolutional block
shares, gone with the last of them. The pieces are the elementwise set every
block uses (RMSNorm rows, GELU, add, quantize, the batched matvec) and six
kernels of their own: SAME padding, depthwise, im2col, and multi-query
attention's scores, softmax and weighted sum.

Two things on the TEXT side were wrong for a picture, both found by splicing
the golden's rows into the prompt and still reading 2e-2:

    the end-of-image marker    ids from 262144 are the vision embedder's HARD
                               tokens: embed_vision's table, its norm, the
                               projection and the unweighted norm, not the
                               text table. The converter computes the 128 rows
                               once (RoleVMNHardEmbd)
    the per-layer table        llama.cpp pads per_layer_token_embd from 262144
                               rows to 262400 with zeros, and transformers
                               maps every id past the table to row 0. The
                               converter trims the zero rows and an id past
                               the table reads row 0

With both, the golden's rows spliced in read 5.5e-12.

The fixture is transformers' own Gemma3nForConditionalGeneration at a
quarter of the encoder's channels, every parameter randomised, the audio
tower stripped, both GGUFs by llama.cpp's converter
(`scripts/gemma3nvisiongold.py`):

    the processor's pixels                NMSE 2.9e-15
    every block at the engine's arithmetic   worst 5.1e-04
    the rows                              1.6e-04 (f32 tower 7.7e-04)
    end to end from the pixels            worst logit NMSE 1.3e-04, 0 flips
    violations     symmetric padding 1.4e-02, no residual adds 1.5, the
                   coarser stage upsampled transposed 5.0e-03, each pooling
                   window's first row 5.9e-03, the end-of-image marker from
                   the text table 2.0e-02
    the embedder's sqrt(2048)             invisible: the norm after it is
                                          scale-invariant but for epsilon

On the device, every block placed:

    | | cuda:0 | vulkan:0 |
    |---|---|---|
    | each block handed the host's input | 6.5e-09 | 6.9e-09 |
    | the whole tower | 1.5e-04 | 1.7e-04 |
    | tower and text model, end to end | 6.7e-05 | 6.0e-05 |

and against a host with symmetric padding or no residual adds the device
reads 1.4e-02 and 1.5, so the comparison sees both.

Those figures are the int8 arithmetic: every matrix reads the activation
quantized as the host quantizes it. A V100 does not ship that -- with no
integer matrix instruction it takes sm_70's binary16 twin (MatVecMMA70,
GemmVolta; Metal's GemmTile is the same twin), which reads the activation
unquantized -- and on a V100 the gate read block 3 at 1.565e-04 against a
1e-6 bound, every block between 1e-6 and 1.6e-4, while the same card through
Vulkan (dp4a, int8) read 6.9e-09. The twin is not wrong: against the
reference's rows at Q8_0 weights and f32 activations it reads 6.5e-05 where
the host reads 1.4e-04 and the int8 device arms 1.3-1.4e-04. The gate had
not pinned the knob its equality depends on. It runs two arms now: int8
(`NoVolta`, held to the host at 1e-6 a block, and refusing an arm that built
a binary16 matvec) and the twin (`NoMMA`, which puts any CUDA card on it:
the RTX 3050 Ti reads block 3 at the V100's 1.565e-04), held to the host at
1e-3 and required to read closer to the f32-activation rows than the int8
arm. Its violations: the int8 arm unpinned is refused by name (234 binary16
matvecs); a 1% alternating error in the twin's activation conversion reads
1.7e-03 against the host; a 0.4% one passes both host bounds (4.1e-04) and
is caught by the f32-activation comparison alone (3.4e-04 against the int8
arm's 1.3e-04). On the way, the device's convolutional program stopped
quantizing the activation ahead of a twin that never reads it.

The six kernels are held
to their plain loops on both backends at ragged shapes
(`backend.TestConvKernelsMatchTheLoops`) and lower on all three.

Under the principles: one 168 KiB vision page against an 84-page tower (83
evictions) bit-identical; 4 of 84 blocks on the device, then home mid-encode,
bit-identical to that placement from the start on CUDA and Vulkan; 0
allocations after a picture on the host, CUDA and Vulkan; a second turn
restoring 288 of 306 positions with no tower block; a warm encode allocating
nothing; the device ledger returned to the byte after a release
(`tier.TestConvTowerReturnsEveryDeviceBuffer`, which fails at 195584 bytes
live against a scratch that is never freed).

And through a device budget below the tower: every block streamed
(Place.Stream) under the smallest budget that still places all 84 plus a
MiB, beside a host page budget of two vision pages, so every device page-in
re-reads a host page the host pager has reused
(`TestGemma3nTowerPagesThroughTheDevice`):

    | | cuda:0 | vulkan:0 |
    |---|---|---|
    | budget, against the resident tower's | 490.1 MB of 560.8 | 502.0 MB of 572.5 |
    | blocks on the card at once | 3 of 84 | 3 of 84 |
    | over the encode | 81 page-ins, 82 page-outs, 80 host evictions | the same |
    | rows against the resident tower | bit-identical | bit-identical |

and the ledger stays under its limit and returns to its figure before
placement once the blocks are released. The budget is mostly the scratch: a
768x768 picture's widest activations, not the blocks' 14 MB of weights.

The violation (`TestGemma3nTowerPagesThroughTheDeviceStale`, under the
`jitllmfault` tag: a page-in that uploads the spans its block held at
admission, without asking the host again) found a hole on the way: those
spans are EMPTY once the host has given the page back, and the upload of a
container tensor took them anyway -- four-byte placeholder planes, which a
Vulkan kernel then read past (rows at NMSE 1.0) and CUDA refused with no
reason given. `resident()` refuses planes shorter than the shape's now, by
name, so the violation reads "Q8_0 256x576: planes of 0, 0 and 0 bytes,
short of the shape's 147456, 9216 and 0" on both backends; and a page-in's
failure reports its own reason rather than one a placement's refused first
pass left in LastErr.

The real Gemma 3n E2B: no mmproj is published, so one is written by
llama.cpp's own converter (`--mmproj`, f16) from the checkpoint's vision
tensors alone, fetched by range requests beside its config and processor
(a local helper builds the directory), and converted with the
published Q4_K_M text model into one container. Against transformers' own
MobileNet-V5 and embedder in float32 (`scripts/gemma3nreal.py`), on the
photo:

    the processor's pixels       NMSE 2.8e-15
    the 256 rows, host           7.6e-03 (Q8_0 matrices, int8 activations)
    cuda:0 / vulkan:0            6.7e-03 / 7.5e-03 against transformers,
                                 3.3e-03 / 3.7e-03 against the host
    symmetric padding            0.69

and greedy decoding from its own chat template reads *"this is a black and
white newspaper front page from the new york times dated july 21, 1969,
announcing the historic event of "men walk on moon," ..."* -- and on a blank
picture of the same size *"this image is completely blank and white."*

What this does NOT cover: audio. A Gemma 3n prompt's audio tokens go through
embed_audio, which is not built; the audio tower's tensors are left out at conversion.
## kimi-k3: residual attention, a latent mixture, situ

Kimi-K3's text model, read off Moonshot's own remote code
(`modeling_kimi_linear.py` beside `modeling_kimi_k3.py`, the
KimiLinearForCausalLM that KimiK3ForConditionalGeneration wraps), fla's
`naive_kda_lowerbound_gate`, and llama.cpp's `src/models/kimi-k3.cpp` and its
converter before anything was written. It is Kimi-Linear's graph -- KDA
blocks, NoPE MLA blocks, the V3 router -- with five additions:

    residual attention  every attn_res_block_size blocks the block's RAW input
                        is banked as a checkpoint; each sublayer reads
                        softmax_j(w . v_j / rms(v_j)) over the bank and the
                        running residual (w the norm's weight times the
                        projection, fused by the converter), mixed BEFORE the
                        block banks its own; a checkpoint block's running
                        residual restarts from the attention's output; the
                        head reads one last mix
    the latent mixture  the router reads the normed row at n_embd, the experts
                        routed_down's projection of it at
                        routed_expert_hidden_size, their weighted sum is
                        RMSNormed and routed_up; the shared experts stay at
                        n_embd
    situ                beta*tanh(g/beta)*sigmoid(g) * lbeta*tanh(u/lbeta),
                        beta 4 and lbeta 25, on every FFN
    MLA's output gate   sigmoid(g_proj(normed input)) times the attention's
                        output, before o_proj
    KDA                 a full-rank output gate (one matrix where Kimi-Linear
                        has two low-rank ones) and the decay
                        exp(lb * sigmoid(exp(A_log) * (f + dt))) with lb the
                        gate_lower_bound

**Where it lives.** The checkpoints are streams of one stream-major residual,
as gemma3n's and DeepSeek V4's are (`Config.Streams`), so the seam, paging,
the step and the batch move them as one slice; the head's mix is the host's
(`streamHead`). The container gains `jlm.ArchKimiK3` (50), roles 350-355 (the
two residual scores per block and the head's, routed_down/up and the routed
norm) and a ninth config tail (the block size, the latent width, both situ
bounds, the decay bound and the latent norms' epsilon). The converter joins
llama.cpp's split q/k/v and its three convolutions back into the mixed
projection the KDA kernels read, as `blockLinear` does for a Kimi-Linear
safetensors file, and broadcasts ssm_a per channel.

**On the host** two new generated kernels on AVX2, SSE and arm64: situ as an
activation kind (`ActSitu`; tanh is Eigen's clamped rational, ~3e-7 relative,
since the exp form loses precision once multiplied by 25), and the bounded
decay (`cpu.EmitDeltaDecayBound`, 3.8e-6 against fla's formula in float64).
The mix is RMSNorms against a unit vector and one small matvec over the bank,
both kernels the engine had. **On the device** three basic kernels
(`kernels/k3.go`: the scores, the mix, the placement of every stream after a
sublayer), `DeltaDecayBoundRows` and the bound inside the fused delta rule,
situ in `ActMul`; the latent mixture is the generic one with its input and
output moved (the decode path and the grouped one read and write at
`LayerPlan.ExpWidth`).

`synth-kimik3` (scripts/k3gold.py): Moonshot's KimiLinearForCausalLM with
fla's torch references swapped in for its Triton kernels (checked against
transformers' own kimi_linear recurrent and chunked forms at 2.2e-8 and
3.8e-8), attention eager; six blocks -- KDA, KDA, MLA, KDA, KDA, MLA -- a
checkpoint every two (three banked by the head), a dense lead, then 8 experts
top-2 at a latent of 32 under n_embd 64 beside one shared expert, a query
latent of 32, the KDA gate full-rank under a bound of -5, rms_norm_eps 1e-3
so the latent norms' 1e-6 is visible; every parameter randomised (the router's
selection bias included), F32 through llama.cpp's converter. Kimi-K3's own
tiktoken tokenizer.

    host vs the reference, decode / prefill at every length / batch   2.964e-10
    device vs host, every block and the head
      cuda:0   decode 9.789e-11, prefill 4.465e-12
    the prompt as one batched submission, a decode token on the device
      cuda:0   11 mixes, 5 latent mixtures, 5 grouped, 2 MLA batched
    step across three sessions vs each alone   cuda 4.805e-14
    seam moves with three rows in flight       3.1e-7 before, 6.1e-7 after
                                               (bound 1e-3; the shipping f16 cache)
    one-frame paging                           153 page-ins, ids identical
    ForwardBatch under a one-page budget, KV pages relocating, a block
      relocating between devices, float banks batched vs row by row: pass
    TestDecodeDoesNotAllocate                  0 on the host and cuda (decode and the step arms)

Every violation lands far past the bound. On the host: no residual attention
2.402, a checkpoint block adding instead of restarting 2.782, the bank scored
on raw values 2.277, the head reading the running residual alone 2.205, a
checkpoint every three blocks 1.699, the latent sum unnormed 2.428, the MLA
output ungated 0.6123, the KDA output ungated 2.385, the decay unbounded
(Kimi-Linear's softplus) 2.611, the latent norms at rms_norm_eps 4.944e-02,
plain SwiGLU 2.845. In the device session alone, every one of those but the
head's (the host's): 2.443, 2.285, 2.291, 1.571, 2.468, 0.6116, 2.102, 2.367,
5.017e-02, 2.543.

Where the references part (RULE 7m), and the choice:

- **The MLA latent norms' epsilon.** Moonshot's code and transformers'
  KimiLinear build q_a_layernorm and kv_a_layernorm at the RMSNorm class
  default, 1e-6; llama.cpp norms them at rms_norm_eps. The engine follows the
  model's code (`jlm.Config.LatentNormEps`), for Kimi-K3 and for Kimi-Linear:
  Kimi-Linear had been norming at rms_norm_eps since it landed, which
  synth-kimilinear (rms_norm_eps 1e-5) could not see -- 3.8e-11 before, 3.2e-12
  after, both under its 1e-9 bound. The K3 fixture's 1e-3 makes it a 4.9e-02
  violation.
- **The residual scores.** llama.cpp's converter fuses the norm's weight into
  the projection; the reference keeps them apart. The score is linear in both,
  so the fused vector is the same function, and the engine reads the file's.

Found on the way:

- **A linear block of a model with a compressed MLA query never got its
  mixed projection on a device.** The tier's slot skip treated slot 0 as MLA's
  one-step query wherever the model has a query latent, and on a hybrid the
  linear blocks' slot 0 is the q|k|v the delta rule reads -- unreachable on
  Kimi-Linear, whose q_lora_rank is null, and a nil dereference on Kimi-K3's
  first placed block.
- **`releaseLayer` released none of MLA's five matrices** (q_a, q_b, kv_a and
  the two absorbed banks) nor their views, so a released MLA block held its
  weights.

The safetensors path, which reads the release's compressed-tensors MXFP4,
came after: "The release from safetensors" below. The vision tower is not
built. Kimi-K3 has no multi-token-prediction blocks to build: the release
(93 trunk layers, 497,220 tensors, none an MTP tensor) and every reachable
checkpoint state num_nextn_predict_layers 0, Moonshot's modeling code defines
no MTP module, and llama.cpp's kimi-k3 graph has none; convert/hfkimik3.go
refuses a nonzero count, which is the guard should a release add them.

### Real trained weights: Kimi-K3-0.40B

Kimi-K3 is 2.8T and mgoin/Kimi-K3-pruned75 is 442.5 GiB, but two small
checkpoints carry the architecture with trained weights:

    inference-optimization/Kimi-K3-0.40B   float32, 1.58 GB; fitted to one
        copypasta (FitnessGram, the bee movie) from scratch to a perplexity
        under 3. Eight blocks, MLA at 4 and 8 (one-indexed) and KDA elsewhere,
        a checkpoint every four blocks (two banked by the head), a dense lead
        then latent mixtures 1024 -> 512 of 8 experts top-2 beside one shared
        expert, situ, the MLA output gate, q_lora 256 / kv_lora 128, KDA's
        full-rank gate and NO gate_lower_bound: the decay is Kimi-Linear's
        softplus, the branch synth-kimik3 (bound -5) never reaches
    Eraly-ml/Kimi-K3-0.40B-Kazakh-CPT-59M   the same model continued on 59M
        tokens of Kazakh, float16

Its published GGUFs: murillo2000's F16 and MXFP4 (the latter from
inference-optimization/Kimi-K3-0.40B-MXFP4, routed experts MXFP4, the rest
F16) and mradermacher's Q8_0 and Q4_K_M. Every other "Kimi-K3" small repo on
the Hub is either a copy of this one, a 100+ GB slice of the real model, a
random-weight tiny-random, or another architecture under the name
(SepehrKerachi's KimiK3ForCausalLM, Tiny-K3).

The reference is the checkpoint's own remote code (`modeling_kimi_k3_linear.py`,
the same model as Moonshot's file with a training dispatch and transformers 5
spellings), fla swapped as for the fixture, in float32 on the file's weights
(`scripts/k3gold.py real-<name>`, run on the V100 box). Three prompts, each
with the reference's 32-token greedy continuation, every position
teacher-forced through decode and through a prefill of the prompt
(`TestKimiK3RealMatchesReference`), worst NMSE over the vocabulary's first
256 logits and the reference's top 64:

    file                           host        cuda        vulkan     flips
    F32 (llama.cpp's converter)    5.8e-13     8.2e-11     2.4e-12    0
    F16, murillo2000               9.3e-13     1.1e-11     2.1e-12    0   (against the
                                     reference on its weights rounded as the file
                                     stores them, K3_ROUND=float16)
    Q8_0, mradermacher             6.8e-03     6.8e-03     6.7e-03    0
    Q4_K_M, mradermacher           1.1e-01     1.1e-01     1.1e-01    1
    MXFP4, murillo2000             4.3e-02     4.3e-02     4.3e-02    0
    Kazakh CPT, F16 (converter)    4.9e-13     7.4e-12     4.5e-13    0

The quantized rows are the file, not the engine. The reference itself moves
5.0e-03 when its matrices are rounded to f16, which is what makes F16
against the float32 reference read 5.0e-03 and why its arm is held to the
rounded reference instead. The Q4_K_M flip is "The capital of France" ->
"ith" where the reference says " my" (margin 0.955), and llama.cpp reading
the same file answers "ith" on its CPU and CUDA builds. The greedy text is
the reference's, 32 of 32 tokens on every file and tier: "The FitnessGram
Pacer Test is a multistage aerobic capacity test that progressively gets more
difficult as it continues. The 20 meter pacer test will begin in 30 seconds.
Line up"; llama.cpp (85ca3b5, with kimi-k3) writes the same three
continuations on every file on its CPU and CUDA builds, and the same for the
Kazakh checkpoint.

The violations, on host decode over the exact arms, against a 1e-8 bound: the
decay bound applied where the file states none 1.47e-01 (Kazakh 1.29e-02), no
residual attention 34.8, a checkpoint adding instead of restarting 5.58 (3.7e-03),
the bank scored on raw values 2.44, the head reading the running residual
1.18, a checkpoint every three blocks 18.8, the latent sum unnormed 1.54, MLA
ungated 4.64, KDA ungated 0.556, plain SwiGLU 4.20. The MLA latent norms'
epsilon is inert on trained weights -- rms_norm_eps in place of 1e-6 moves the
0.40B 2.1e-7 and the Kazakh model 1.7e-12 -- so that choice keeps
synth-kimik3 as its only gate.

Found on the way: **a mixed quantization did not convert.** Q4_K_M stores a
KDA block's attn_q at Q4_K beside attn_k and attn_v at Q8_0, and
`k3JoinRows` refused parts of two types, so every mixed-quant Kimi-K3 GGUF --
the form the real model is published in -- failed at conversion. Parts of
several types are dequantized and re-stored as one Q8_0 projection, as
Qwen3.5's beta/alpha fusion does: a Q8_0 part comes back bit for bit and a
narrower one pays Q8_0's bytes (`convert.TestK3JoinsAMixedQuantization`). Not
built: running q, k and v as three projections, which would keep each part's
own bytes and needs every tier's linear block to read three matrices.

### The release from safetensors: compressed-tensors MXFP4

`jitllm convert hf://moonshotai/Kimi-K3` converts Moonshot's release where it
lives, streamed as Kimi-Linear is: `KimiK3ForConditionalGeneration` is an
entry of `hfArchTable` (`convert/hfkimik3.go`), and its routed experts, which
the release ships as compressed-tensors `mxfp4-pack-quantized`, are read by
`convert/hfmxfp4.go`.

**The config is configOf's.** text_config is turned into the GGUF keys
llama.cpp's converter (`conversion/kimi_k3.py`) writes from the same dict and
handed to `configOf`, so `kimiK3Config` is the one reader of what each key
means and the two inputs cannot drift. Every absent key takes
KimiLinearConfig's default, and each switch the engine does not run is refused
by its key, the class's default included (`mla_use_nope` and
`use_full_rank_gate` default false, `hidden_act` silu). The layer kinds are
read from `kda_layers` (what `is_kda_layer` reads) and must partition the
blocks with `full_attn_layers` (what llama.cpp reads).

**The experts are moved, not requantized.** compressed-tensors' MXFP4 is OCP
MXFP4: `weight_packed` holds e2m1 codes two to a byte (element 2i low),
`weight_scale` an E8M0 byte per 32 elements of a row worth 2^(s-127). The
container's MXFP4 is GGUF's block (the scale byte, then 16 bytes holding
element j low and j+16 high, ggml's table doubled against a 2^(s-128) scale),
so the repack is a permutation of nibbles and the scale byte copied --
llama.cpp's `repack_mxfp4_blocks`. A packed weight is any matrix the role
table places and nothing cuts (the 0.40B's shared experts are MXFP4 too); a
split, a fused cut or a gather of an MXFP4 tensor is refused. Any other
compressed-tensors format, an asymmetric or non-32 group, a non-E8M0 scale,
static input or any output activations, a KV-cache scheme, or two
quantization_configs that disagree (top level and text_config) is refused by
name. A dynamic input-activation scheme carries no tensor and is accepted
without being applied: the release states none (W4A16), and
inference-optimization's 0.40B states dynamic MXFP4 activations that its own
README calls W4A16.

**The gates.**

    convert.TestK3SafetensorsMatchesItsGGUF   synth-kimik3 and the trained
        0.40B (F32) converted from safetensors and from llama.cpp's F32 GGUF of
        the same checkpoint: every Config field equal, 144 and 191 tensors
        byte for byte; the 0.40B's three KDA decay rates part by one ulp
        (-exp(A_log): float32 torch against float64 rounded once)
    convert.TestK3MXFP4ExpertsMatchTheGGUF    inference-optimization's
        Kimi-K3-0.40B-MXFP4 against murillo2000's MXFP4 GGUF of it: every
        Config field equal, all 21 expert banks byte for byte, the shared
        experts' MXFP4 decoding exactly to the GGUF's F16, every other float
        within F16's rounding. Over the checkpoint's 168 expert matrices the
        repack equals the GGUF's on 168; nibbles swapped, the scale off by
        one, or the bytes carried with no block permutation equal on 0
    convert.TestMXFP4RepackIsLossless         the repack's blocks decode
        (quant.Dequant32) to compressed-tensors' own reading of every code
        and a sweep of scales, bit for bit
    convert.TestPlanIsTheWrite                the dry run's size and header
        are the conversion's, byte for byte

and the trained weights against Moonshot's code
(`TestKimiK3RealMatchesReference`), the golden for the MXFP4 checkpoint taken
on compressed-tensors' own decompression of its experts (`k3gold.py
real-Kimi-K3-0.40B-MXFP4`, its `real_weights`):

    container                         golden                  host      cuda      vulkan   flips
    0.40B F32 from safetensors        real-Kimi-K3-0.40B      5.8e-13   2.0e-12   2.4e-12  0
    0.40B-MXFP4 from safetensors      real-Kimi-K3-0.40B-MXFP4 3.2e-05  3.2e-05   3.1e-05  0
    the same, against the unquantized golden                  4.3e-02   4.3e-02            0

The MXFP4 container's 3.2e-5 is the engine's MXFP4 kernels rounding the
activation they multiply; against the float32 weights' golden it reads the
experts' quantization, over 400 times its bound. 32 of 32 greedy tokens are
the reference's on every prompt. The F32 arm runs the violations too, each
far past its bound as on the GGUF.

Found on the way:

- **The dry run found the release's A_log padded.** `jitllm convert -plan`
  (`convert.PlanShards`: the whole conversion up to the first tensor byte,
  every read answered by zeros of its length, so a repository is read by its
  headers alone) refused the release at its first KDA block: A_log is 128
  entries for 96 heads. Read off the Hub by range, entries 96-127 are zero in
  every block sampled; llama.cpp's converter takes `[:n_head]`, and
  Moonshot's own module (`A_log` of `num_heads`) cannot load the file
  strictly. The converter takes the heads' entries and refuses a tail that is
  not zero.
- **The trained checkpoint's tokenizer_config names specials past the
  embedding.** transformers re-saving TikTokenTokenizer appends its
  additional_special_tokens at 163840-163846 of a 163840-row model, which
  tokenization_kimi.py never reads (it takes added_tokens_decoder for its 256
  reserved rows only); the tiktoken reader refused the file. An entry past
  both is passed over, one inside the reserved rows but past the embedding
  still refused.

**The full release, planned without downloading it** (96 shards' headers by
range, 1:48 on this link, 0.43 GB resident): nothing refused; 2297 tensors,
93 blocks of 4464 MiB, 82,432 expert pages of 16.73 MiB; 168 vision tensors
named and not carried. Source 1560.9 GB; the container is 1891.2 GB: the
MXFP4 experts 1446.5 GB as they are, the 113.5 GB of BF16 carried as 227.0 GB
of F32, and 217.8 GB of page padding -- every block page is the largest
block's, and the dense lead's 33792-wide SwiGLU makes block 0 (4.68 GB at
F32) twice any mixture block. `-q8` plans 1564.6 GB. A BF16-preserving store
(the container has `TypeBF16` and the host a BF16 matvec) would carry the
BF16 as it is and halve the pages with it, about 1.67 TB by the same
arithmetic; whether to build it, or to convert with -q8, is the user's
decision.

## Image preprocessing is generated code

Only the resize's integer passes ran on generated code (`nn.ResampleRowJIT`);
a census of the image path found Go loops over every pixel or patch around
them, and two over transcendentals. Every one of those is a kernel now, on
AVX2, SSE and NEON (`jit_old/cpu/imgproc_const.go` has the contracts,
`engine/nn/image.go` the entry points), and what is still Go is geometry from
a picture's sizes and the tower's constant tables.

| site | was | is |
|---|---|---|
| MiniCPM-V's resampler sin/cos (`rsState.axes`) | `math.Pow`, `math.Sin`, `math.Cos` per (coordinate, frequency) | `SinCosTab`: the frequency's power, the float32 angle, sin and cos, in float64 lanes |
| a resampled position table's taps and weights (`posLerp`: Qwen3-VL's bilinear, GLM-4.xV's grid_sample bicubic, Kimi-VL's interpolate bicubic, HunyuanVL's interpolate bilinear) | `bilinearAxis`/`bicubicAxis`/`interpAxis`/`linearAxis`/`cubicAt` per patch, and the products per tap | `AxisTaps` per axis (the coordinate as each reference's nine float32 steps, floor, the filter's weights, the clamped indices), `LerpGrid` for the outer product, one plane per tap in raster order |
| decoding (`toRGB8`, the bilinear path's planes) | `img.At(x, y).RGBA()` per pixel through the interface | `PixelRow` on the decoder's own bytes: `*image.RGBA`, `NRGBA`, `Gray`, `RGBA64` and `YCbCr` at every subsampling, composited over white into bytes or kept as 16-bit floats |
| the horizontal pass | int32 widening, a transpose, the kernel, a transpose back, the narrowing | `ResampleH`: a pixel at a time, its three channels as lanes, bytes in and out (Pillow's own loop) |
| the vertical pass | int32 in and out | `Resample`, bytes in and out |
| the normalisation (`normalizeInto`, Llama 4's `lookupInto`) | a table lookup per sample in Go | `PixLUT`: the same per-channel table, read per sample |
| the patch gather (`encodeBegin`) | a Go loop per sample | `Copy32`, a strided copy per patch and channel |
| the bilinear path's transposes | Go loops per sample | `Copy32` |
| Gemma 4V's standardisation | two transposes around `Scale32JIT` per channel | `ActMul32JIT` per row with the identity: the same single product |

Every output is the bits it was. A dump of every preprocessing entry point of
fifteen tower families (SmolVLM, llava, Gemma 3, InternVL, Qwen2-VL, Qwen3-VL,
GLM-4.xV, Kimi-VL, Pixtral, Phi-4, Janus, MiniCPM-V, Llama 4, Gemma 4, Gemma
3n) over 25 pictures -- RGBA opaque and translucent, NRGBA translucent, Gray,
YCbCr at all six subsamplings whole and as sub-images at odd offsets, a JPEG
round trip, a palette with translucent entries, RGBA64 at an offset origin,
NRGBA64, Gray16, CMYK -- plus the bilinear path on each and a tower encode per
family (25 encodes each for Qwen3-VL, GLM-4.xV and Kimi-VL, whose position
tables change per grid), 833 SHA-256s of float bits in all, read the same
before and after; HunyuanVL, which landed meanwhile, read the same 75 (its
preprocessing and 25 encodes) against the tree it landed in. The gates read what they read before:
`TestPreprocessMatchesTransformers` 0 of 2,408,448 (Gemma 3) and 0 of
7,827,456 (InternVL) values differ, `TestLlama4PreprocessMatchesTheProcessor`
0 of 188,160, and every NMSE line of the Qwen, Pixtral, Phi-4, Gemma 4 and
Gemma 3n preprocess gates is the number it printed before.

Each kernel's gate is in jit_old/cpu, against its formula or its reference's own
code written out, on the native tier and the SSE tier beside it on amd64, and
on NEON under qemu-aarch64: ragged sizes around every vector boundary, guards
after the output (and between rows where the output is strided), the inputs
ending at a PROT_NONE page so a read past a pixel, a row or a table faults, and
a violation that must fail.

    TestPixelRowMatchesTheStdlib       132 pictures, every format and subsampling,
                                       bit-exact against At().RGBA(); a pixel's
                                       right neighbour differs
    TestResampleHMatchesTheReference   160 shapes x precisions 7/13/22; one bit
                                       less of shift, and windows a pixel late, differ
    TestAxisTapsMatchesTheReference    22 sizes x 6 table sides per filter, every
                                       weight and index; the float64 port differs
                                       at 1,650 / 4,251 / 4,514 / 1,497 of them
                                       (Qwen3-VL, GLM, Kimi, HunyuanVL)
    TestSinCosTabMatchesTheReference   774,729 entries, K = 1..9 and every d/4 a
                                       real width makes to 2048, bit-exact against
                                       the Go table; the angle folded in float64
                                       (the rotary kernel's way) differs at 338,577

**The sin/cos table is exact because the float64 is good, not because it is
the same float64.** The Go table took `math.Pow`, `math.Sin` and `math.Cos`
(Pow's Exp is assembly on amd64 and arm64, and the compiler fuses the pure-Go
parts on arm64, so their float64 already differed by host) and rounded each to
float32. The kernel computes the power as
exp of y*ln(base) -- ln(base) split so the product is exact, a Cody-Waite
reduction by ln2, the degree-13 Taylor series -- and sin and cos by a
three-part reduction by pi/2 and fdlibm's kernels, every operation its own
rounding and in the same order on x86 and NEON, so every tier computes the same
float64. Each is within a few ulp of the true value; the nearest any true value
at the real widths comes to a float32 rounding boundary is 1.5e-13 relative
(`TestSinCosMarginsHoldTheTable`, which fails below 1e-14), so every float32
lands where the Go table's did. The rotary table's kernel (`RopeTable`) does
not reproduce it: its float32 range reduction is one ulp off on 18.7% of
entries, and it folds the angle in float64 where the reference rounds the
product to float32 first.

**The position-table weights are now the same on arm64 as on amd64.** Go fuses
`x*y + z` into one FMA on arm64 and not on amd64, so the Go weights
(`((A+2)*x - (A+3))*x*x + 1`, `(i+0.5)/size*2 - 1`) were two different
functions on the two hosts. The kernels take every product and sum as its own
rounding on every tier, which is amd64's Go and the float32 arithmetic the
references are held to; the test's reference wraps each product in a float32
conversion so no host fuses it.

**The patch gather is a strided copy, not a permutation at conversion.**
Permuting the patch weight's columns to [dy][dx][channel] would make a patch
three contiguous copies, but a quantized patch weight (Q8_0 is common) groups
32 columns under one scale, so the permutation would regroup the blocks and
change the weights -- not the same tower. The container is unchanged.

**What stays Go, and why it is not per-element compute.** The resize's filter
taps (`aaFloats`, `aaWeights`) are float64 evaluations of the filter at
O(output x support) points once per picture, with a data-dependent tap count
and a normalisation by the window's sum -- not AxisTaps' shape, and geometry
from two sizes; `hWindows` lays them out for the kernel in integers;
`bilinearTaps` is two taps per output; Phi-4's position-table resize
(`phi4Table`) is `aaFloats`' taps and generated axpys over whole rows. The
normalisation tables (`normLUT`, Llama 4's `bf16NormLUT`) are 768 entries from
the tower's mean and std, read per sample by `PixLUT`. The window order, the
slice plan and the resize sizes are integer geometry. The post-tower moves
(the 2x2 merge, the unwindow, the deepstack shuffle) copy whole rows with
`copy`. The codec is file decoding and stays the standard library's: PNG's
inflate, JPEG's IDCT and colour planes (Go's decoder, not libjpeg-turbo), and
`image/draw`'s conversion of a palette, 16-bit gray, NRGBA64 or CMYK picture
into RGBA64 -- the same `At().RGBA()` the old loop read -- before `PixelRow`
reads it.

**The gate.** `TestImagePathRunsGeneratedCode` (`internal/oracle`) reads the
image-path files of engine/model and fails on a transcendental math call
anywhere in them, on float arithmetic on data inside a loop, and on a loop that
calls, per iteration, an engine/model function doing float arithmetic -- the
per-patch `cubicAt` shape, which the engine-wide `TestNoGoComputeOnTheInferencePath`
could not see because the arithmetic sat in a helper. Five functions are
listed with their reasons (the filter taps, the bilinear matrix, the two
normalisation tables, MiniCPM-V's slice grid); `TestImagePathGateDiscriminates`
runs it on the old shapes (the sin/cos loop, a cubic helper called per patch, a
per-sample normalisation, a power) and on scalar size geometry, which it leaves
alone. The engine-wide list lost `rsState.axes` and `posLerp`.

**Time, as a regression check and not a board row.** The same timing file in
the base commit's tree and this one, each binary on the laptop (i9-12900HK,
AVX2 tier, single-threaded work pinned to the six P-cores), run old, new, new,
old, median of 21 repetitions per piece, in milliseconds:

    decode, YCbCr 4:2:0 1600x1200            49.3  50.7  ->  5.1  5.5
    decode, RGBA 1600x1200                   21.7  21.4  ->  2.4  2.6
    Pillow BICUBIC, 1600x1200 -> 672x504     16.5  17.2  ->  6.6  6.8
    torchvision bilinear, -> 896x896         21.5  21.4  ->  4.6  4.8
    torchvision bicubic, 672x504 -> 1344x1008  13.7  15.9  ->  3.8  4.0
    normalise 672x504                        0.67  0.76  -> 0.30 0.32
    MiniCPM-V sin/cos, d 3584, a 32x32 grid  3.80  4.41  -> 0.33 0.34

The decode was the interface call per pixel; the resizes were the int32
widening and the two transposes, which the per-pixel horizontal kernel no
longer needs even at its basic width. Memory: the resize no longer holds two
int32 copies of the picture (12 bytes a pixel each) beside the bytes.

**Basic kernels, and what is owed.** Each image kernel but the vertical pass is
written once in legacy SSE (SSE4.1 at most) and serves both x86 tiers: the SSE
tier's declares ISATierSSE, the AVX2 tier's issues VZEROUPPER first so no
legacy instruction meets a dirty upper half. They work a pixel, four samples or
two float64 lanes at a time. The vertical pass, which carries most of a
resize's arithmetic, keeps its 256-bit AVX2 kernel (bytes in, a byte gather
out). 256-bit forms of the rest are owed, the horizontal pass first.

## Design notes moved out of source comments

Each entry below is a comment that stood in the named source file,
verbatim. The source keeps the contract, the layout and the invariants
and points here for the derivation, the transcript and the reasons.

### engine/model/ds4.go

DeepSeek V4 (transformers' DeepseekV4ForCausalLM, llama.cpp's
deepseek4.cpp; jlm.ArchDeepseek4).

The residual is Config.HCMult streams, carried stream-major as Gemma 3n's
are (Config.ResidW: stream k of row r at (k*rows + r)*n_embd), each a copy of
the embedding at the start. Each sublayer is wrapped in a hyper-connection:

	flat   = rmsnorm(the row's streams, concatenated)          no weight
	m      = fn · flat                                         (2+H)*H mixes
	pre    = sigma(m[:H]*s0 + b[:H]) + eps
	post   = 2*sigma(m[H:2H]*s1 + b[H:2H])
	comb   = sinkhorn(softmax_rows(m[2H:]*s2 + b[2H:]) + eps)  H x H
	y      = sublayer(norm_w(sum_k pre_k x_k))
	x'_k   = post_k * y + sum_j comb[j][k] x_j                 comb TRANSPOSED

sinkhorn is one column normalisation, then HCIters-1 rounds of row then
column, each dividing by (sum + eps). The head collapses the streams the same
way with its own fn and pre alone (sigma(m*s + b) + eps), then the output
norm and the projection.

The attention is one key head that is also the value, head_dim wide, with a
sliding window of SWAWindow over every block:

	qr = rmsnorm_w(q_a · h)    q = q_b · qr, each head rmsnorm'd, no weight
	k  = rmsnorm_w(kv · h)     the cached row's key and its value

both rotated, adjacent pairs, on the LAST NRot dimensions of each head; on a
compressed block (Config.CompKinds) at RopeBase with YaRN (the "compress"
rotary), on a sliding block at RopeBaseSWA, plain (the "main" one). The
attention output is rotated back at the query's position before the grouped
projection: OGroups groups of heads, each through its own OLoraRank sheet of
RoleAttnOutA, then RoleAttnOut over all of them. Each head has a sink.

A compressed block also attends to ENTRIES, one per `rate` positions: entry
w pools the window [w*rate, (w+1)*rate) of the compressor's two projections,

	e_w = rmsnorm_w(sum_j softmax_j(gate_j + ape_{j mod rate}) * kv_j)   per channel

rotated at position w*rate on the compressed rotary; a CSA block's
projections are two heads wide and its entry pools 2*rate slots, the
previous window's first half and this window's second (the overlap; window
0's previous half is empty). Entry w is visible to a query at t once its
window has closed, w < (t+1)/rate. An HCA block attends to every visible
entry; a CSA block to the lightning indexer's IdxTopK, scored as DeepSeek
V3.2's is (sum_h w_h relu(q_h . k), the query from the latent qr, the key
an entry of the indexer's own compressor at IdxHeadDim).

Where it lives: the cached row of a position is the key and, on a
compressed block, the compressor's two projections of that position (the
gate with its position bias added), so the window's own pages hold every
input the compressor still has to fold -- the window covers the HCA rate and
two CSA windows, which the converter checks -- and they page, relocate and
are stored exactly as the keys are. The entries are a second paged history
per compressed block, indexed by entry (kvCache.ent; ds4kv.go).

The FFN is the mixture: sqrt(softplus) gates, the selection bias, the
renormalised top-k, a routed scale and one shared expert, every expert's
SwiGLU clamped (ActSwiGLUClamp); the first NHashLayers blocks select their
experts from the token's row of a frozen table (the weights are still the
gate's).

### engine/model/delta.go

The gated delta net: the other half of a hybrid's block loop.

The graph is transcribed from llama.cpp's running graph; six of its steps
are ones a from-paper reading gets wrong. `llama-eval-callback` on
Qwen3-Next-80B-A3B-Instruct prints layer 0 as:

	conv_states       GET_ROWS(cache_r_l0{24576})   -> {3, 8192}
	qkv_mixed         MUL_MAT(attn_qkv.weight{2048,8192}, attn_norm)
	conv_input        CONCAT(conv_states, TRANSPOSE(qkv_mixed))   -> {4, 8192}
	conv_state_last   VIEW(conv_input)[1:]          -> CPY into cache_r_l0
	conv_output_raw   SSM_CONV(conv_input, ssm_conv1d.weight{4,8192})
	conv_output_silu  SILU(conv_output_raw)                        <- (1)
	q_conv/k_conv     VIEW{128,16} -> L2_NORM -> REPEAT to {128,32} <- (2),(3)
	v_conv            VIEW{128,32}
	mixed_ba          MUL_MAT(ssm_ba.weight{2048,64}, attn_norm) -> {4,16}
	a                 VIEW{2,16} -> +ssm_dt.bias -> SOFTPLUS -> *ssm_a  <- (4)
	b                 VIEW{2,16} -> SIGMOID                             <- (5)
	node_53           GATED_DELTA_NET(q, k, v, gate, beta, state)
	attn_output       VIEW -> RMS_NORM -> *ssm_norm.weight
	z                 MUL_MAT(attn_gate.weight{2048,4096}, attn_norm)
	final_output      node_59 * SILU(z)                                 <- (6)
	linear_attn_out   MUL_MAT(ssm_out.weight{4096,2048}, final_output)

	(1) the convolution's output is passed through SiLU before it is split.
	(2) q and k are L2-normalised, not RMS-normalised: there is no weight, and
	    the divisor is the vector's own length rather than its root mean square.
	(3) there are 16 key heads and 32 value heads, and the repeat is
	    interleaved: value head h reads key head h/2, not h%16.
	(4) the decay is softplus(a + dt_bias) * ssm_a, and ssm_a is already
	    -exp(A_log), so the product is negative and exp() of it is in (0,1).
	(5) beta is a plain sigmoid, no bias.
	(6) the output gate is SiLU(z), while the full-attention layers of the same
	    model gate with sigmoid.

The state is not a KV cache: it is the same size at every position, so
residency, paging and placement cannot treat it as KV.

### engine/model/spec.go

Speculative decoding with the model's own multi-token-prediction block.

A model that ships prediction blocks (jlm.Config.NMTP: Qwen3.5/3.6,
DeepSeek-V3-class, GLM-4.7-Flash) carries a one-block drafter trained beside
it. Row q of the draft is the token at q and the trunk's normed hidden state
at q-1 (zero at q = 0), and it predicts the token at q+1:

	z   = eh_proj([enorm(embed(x_q)) ; hnorm(h_{q-1})])
	z  -> the prediction block (an ordinary block of the architecture)
	g   = head_norm(z), logits = head(g)

which is llama.cpp's graph_mtp and vLLM's MTP layer, with llama.cpp's
pairing: the row at position q carries x_q and h_{q-1}, so the draft's KV
cache lines up with the trunk's positions and row 0 sees a zero hidden
state. vLLM masks the embedding at position 0 instead; that changes one
cached row and no output (RULE 7m: chosen, written down). A drafted token's
row carries the draft's own g as its hidden state.

A round, at trunk position P with the next token y decided and h_{P-1}:

	draft    d_1 from the draft row (y, h_{P-1}); d_i from (d_{i-1}, g)
	verify   the trunk runs [y, d_1 .. d_k] at P..P+k in ONE pass, logits and
	         hidden state at every row
	accept   greedy: the longest prefix where the trunk's argmax is the
	         draft; sampled: speculative rejection sampling, so the output is
	         distributed as the trunk's own sampler. Either way one more token
	         comes from the trunk's own logits at the first disagreement
	rollback the trunk forgets the rejected rows (attention by position; a
	         recurrence by snapshot or replay, SpecRollback) and the draft
	         re-runs the accepted rows with the trunk's hidden states, its
	         last row proposing the next round's d_1

Greedy decoding through a Speculator is greedy decoding: every emitted token
is the argmax of the trunk's logits at its position, computed over exactly
the tokens plain decode would have fed.

### engine/model/speclookup.go

Prompt lookup: speculation with no prediction block, so on every
architecture. It is a drafter of the one Speculator (`WithSpecDrafter`;
`SpecDraftAuto` takes the prediction block where the model carries one and
lookup otherwise), not a second speculator: the verification pass, the
greedy acceptance, the rollback of attention by position and of a recurrence
by `SpecRollback`, and the three breaks `specFault` names are the MTP path's,
unchanged. What differs is the draft and what is not run: no draft State, no
catch-up, and a round that finds no match drafts nothing, so it is a decode
step of one row rather than a wasted verification.

The draft: for n from `ngMax` down to `ngMin` (2..4 by default,
`WithSpecNGram`), the latest earlier occurrence of the sequence's last n
tokens (prompt and every decided token), and up to k of the tokens that
followed it (k = 5 by default, `WithSpecDraft`). This is llama.cpp's prompt
lookup and vLLM's ngram proposer; the latest occurrence, not the first, is
chosen because a copy in progress continues where it last was. n = 1 is
admitted by option and is not the default: one token matches nearly
everywhere, and each wrong match is verification rows for nothing.

RULE 8: the search compares token ids, integers already decided, and
decides only which rows the next verification carries. No value of any row's
arithmetic depends on it -- a wrong proposal changes how many tokens a pass
yields, never which -- so it is control, as the acceptance comparison beside
it is, and stays Go. It costs O(context x n) integer compares a round and
allocates nothing (`TestLookupSearch` holds it to zero allocations).

Sampling: a lookup draft is a point mass, and the rejection rule that keeps
sampling exact would accept it with probability p(x) and draw the residual
from p with x removed. That is not built: a sampled session through a lookup
Speculator drafts nothing and is plain sampled decode. Greedy only, and the
server and `-spec lookup` say so.

`WithSpecStop` marks the end-of-generation tokens: a draft in the set is
never accepted, so a round that reaches one returns it last, from the trunk's
own logits, unrun -- where plain decode stands after sampling it. `Limit`
caps a round's tokens at what the caller still wants, so the trunk runs no
row past a reply's `max_tokens`. A reply cut by a stop string inside a round
still leaves rows in the session; the server refuses `continue_session` on
such a session (`Session.specRan`) rather than run on them.

The gates (host, exact GEMM, every rollback arm on the hybrids):
`TestSpecLookupMatchesDecode` holds the output token for token to plain
greedy on stories15M, Llama-3.2-1B, the qwen35 hybrid fixture (lookup forced
over its prediction block) and the mamba2 fixture, and demands drafts
accepted; `TestSpecLookupGateDiscriminates` forces the drafts to plain
decode's tokens with every fifth corrupted, so rounds both accept and
reject, and each break -- accepting past the first mismatch, keeping the
rejected KV rows, keeping the stepped recurrent state (rows and replay) --
must part from plain decode; `TestSpecLookupOnDevice` runs the trunk on
CUDA and Vulkan, judging a flip as `TestSpecGreedyOnDevice` does.

### engine/grammar

Structured output: a reply constrained to a context-free grammar, GBNF text
(llama.cpp's format) or a JSON Schema compiled to it (`FromJSONSchema`, the
keywords llama.cpp's json-schema-to-grammar writes, each as it writes them:
type, properties, required, additionalProperties, items, prefixItems,
min/maxItems, min/maxLength, enum, const, anyOf, oneOf, $ref into $defs or
definitions). Every other validation keyword is refused by name
(`ErrUnsupported`, a 400 at the server): pattern, format, minimum and the
other numeric bounds, allOf, not, patternProperties and the rest. llama.cpp
builds pattern, format, integer bounds and allOf; those are the gap, refused
rather than ignored, since ignoring one lets through output the schema
forbids. An object with properties takes no other keys (absent
additionalProperties read as false, OpenAI's strict mode); every output stays
valid under the schema either way.

The automaton is llama.cpp's: a set of stacks of grammar positions, stepped
one code point at a time, rules entered and finished rules popped, a
right-recursive star run as a tail call so a long run does not deepen a
stack. Left recursion is refused at compile time. Two departures:

- Stacks are interned as a persistent list and a set of stacks is interned
  too, so a state is one integer and a step from a state by a code point is
  cached (`ascii` per set, `wide` beside it). JSON revisits its states every
  field, so after the first fields a step is a lookup.
- UTF-8 is checked to the byte: a code point begun by one token and finished
  by the next is carried in the state, a lead byte is admitted only where some
  stack can take a code point it could finish as, and overlong encodings,
  surrogates and bytes past U+10FFFF are refused (llama.cpp's
  partial-UTF-8 check admits some of these).

The allowed set of a state is computed once per (grammar, tokenizer, state)
and kept: the tokenizer's pieces are sorted once per tokenizer
(`grammar.Tokens`, kept beside the model: a cache keyed by the tokenizer's address served a freed tokenizer's index to the next model allocated there), a trie is ranges of that list, and the walk prunes every
piece sharing a prefix the grammar refuses. An end-of-generation token is
allowed where the grammar may end; a state the vocabulary cannot continue
allows them too, so the reply stops rather than leaves its grammar.

RULE 8. The automaton stepping, the trie walk and the allowed set are control:
they compare code points against ranges and decide which tokens may be
sampled; no value a token's arithmetic computes depends on them. The mask's
arithmetic on the logits row is generated: an additive bias of 0 and -inf
over the row, applied by `nn.Add32JIT` (the elementwise Axpy kernel on every
host tier: AVX2, SSE, NEON). What is Go is opening and closing the bias at
the allowed ids, a write per allowed token, as the sampler's history is
written for its penalty kernel. With a grammar the logits are always on the
host -- the server's constrained path reads them back for the sampler, and
never takes the device's argmax -- so the host kernel is the one tier that
holds them; a device-IR mask is not built, and would be needed only by a
constrained path that kept its logits on the card.

At the server a constrained generate decodes alone (not as a row of the step
loop) and without speculation; `ignore_eos` with a grammar is refused, since
the grammar ends the reply with an end-of-generation token.

The gates: `engine/grammar` holds the automaton to languages written by hand,
refusals, the token boundaries (a token spanning two terminals, a code point
split over byte tokens, a lead byte nothing can finish), the mask's -inf and
its zero allocations on a visited state, and random walks over the allowed
tokens validated by `internal/schemacheck`, a validator written apart from
the grammar. At the server, on stories15M and Llama-3.2-1B:
`TestStructuredOutputValidates` (24 seeds a schema at temperature 1, each
parses, validates and ends on the grammar's end), `TestAnAllowingGrammarIsPlainGreedy`
(`root ::= .*` leaves greedy token for token), `TestStructuredGateDiscriminates`
(the mask left off step 3: an invalid output among the seeds) and
`TestResponseFormatOverHTTP` (json_object, json_schema, and `pattern` refused
with a 400 naming it).

### engine/model/msa.go

MiniMax Sparse Attention: MiniMax-M3's block selection (transformers'
MiniMaxM3VLIndexer, llama.cpp's minimax-m3.cpp).

Every block past the dense lead carries an indexer. For each position it
projects one key k (IdxHeadDim wide, RMSNormed, rotated like the
attention's k), and for each query one head q_g per kv group g. The query
at position p then scores every cached position t <= p in its group:

	score_g[t] = q_g . k_t

cuts the positions into blocks of IdxBlock, ranks the blocks by their best
position (a max over the block), forces in the IdxLocal blocks ending at
its own (p/IdxBlock - l, clamped at block 0), and keeps the IdxTopK best.
Every query head of group g then attends to the positions of g's kept
blocks only (and causally, t <= p). With at most IdxTopK blocks of history
every block is kept and the block is dense.

The key is cached as one more kv head of the attention row (Config.KVRowAt,
kvlMSA): it is per position and per layer exactly as k is, so paging,
relocation, the prefix cache and every KV migration carry it with no
second cache. The attention kernels read heads 0..NKVHead-1 of the row and
the indexer's scores read head NKVHead, both at the row's stride; the value
row's extra head is zero and never read.

The selection is a mask on the scores: -inf at every position outside the
group's kept blocks, added after the scale and before the softmax, which is
exactly the reference's additive block mask.

Every operation is generated code: the scores are the attention-score
kernel over the key head, and the ranking is the sampler's segmented
ordering (nn.SampleOrder) over those scores. A block's rank is its best
position's, so walking the positions best first and keeping each new
block until IdxTopK are kept ranks the blocks by their maximum without a
max-pool: the first position seen of a block IS its maximum.

### engine/model/k3.go

Kimi-K3's text model (jlm.ArchKimiK3), transcribed from Moonshot's
modeling_kimi_linear.py (the text model KimiK3ForConditionalGeneration
wraps), which llama.cpp's kimi-k3.cpp matches. It is Kimi-Linear's hybrid --
KDA in the linear blocks (delta.go), MLA with no rotary in the full ones
(mla.go) -- with five pieces of its own:

	residual attention   the residual is banked every AttnResBlock blocks (the
	                     block's RAW input, before anything mixes it), and each
	                     sublayer reads a softmax mix of the bank and the
	                     running residual instead of the residual itself:
	                       score_j = sum(w * rmsnorm(v_j)),  in = sum_j p_j v_j
	                     the scores on normed values, the sum on raw ones. On a
	                     checkpoint block the running residual restarts from the
	                     attention's output; the head mixes the whole bank too.
	latent mixture       the routed experts read routed_down(h) at ExpertLatent;
	                     their weighted sum is RMSNormed and routed_up takes it
	                     back to n_embd; the router and the shared experts read
	                     h itself (k3MoE)
	situ                 4*tanh(g/4)*sigma(g) * 25*tanh(u/25) in every FFN
	                     (nn.ActSitu)
	MLA output gate      the attention output times sigma(g_proj(normed input))
	                     before o_proj (mlaProject, mlaOutProject)
	full-rank KDA gate   one matrix for the output gate (ssmGate, delta.go),
	                     and the decay lb*sigma(exp(A_log)*(f + dt)) when
	                     KDALowerBound is set (nn.DeltaDecayBound32JIT)

The bank is per-token state that crosses a device seam inside a token, so it
is carried as residual STREAMS, as Gemma 3n's AltUp and DeepSeek V4's
hyper-connections are (Config.ResidW): stream 0 is the running residual and
stream 1+j checkpoint j, stream-major (stream k of row r at
(k*rows + r)*n_embd). Every n_embd kernel that reads the residual reads
stream 0 untouched; a device is handed every stream and hands every stream
back, and the head's mix runs on the host (streamHead).

### engine/model/indexer.go

DeepSeek Sparse Attention: the lightning indexer of DeepSeek V3.2
(transformers' DeepseekV32Indexer, llama.cpp's deepseek32.cpp).

For a query at position p the indexer scores every cached position t <= p:

	score[t] = sum_h w_h * relu(q_h . k_t)

q_h is head h of IdxHeads, projected from the query latent (the normed q_a
output MLA already computes) and rotated NEOX on its first NRot dimensions;
k_t is one IdxHeadDim key per position, projected from the block input,
LayerNormed with a bias and rotated the same way; w is one weight per head
from the block input, times (IdxHeadDim*IdxHeads)^-1/2 (the reference's
softmax_scale and n_heads^-1/2 together). The attention then reads only the
IdxTopK highest-scoring positions. At or below IdxTopK cached positions that
is every position, and the block is exactly deepseek2's.

The key is cached in the MLA row itself, after the latent and the rotary
key (Config.KVDim): it is per position and per layer exactly as the latent
is, so paging, relocation, the prefix cache and every KV migration carry it
with no second cache. The MLA kernels read the row's first
KVLoraRank+NRot elements and the indexer's its last IdxHeadDim, both at the
row's stride.

The selection is applied as a mask on the attention scores: -inf for every
position the indexer did not keep, added after the scale and before the
softmax. exp(-inf) is exactly zero, so the result is the attention over the
kept positions alone, which is what the reference computes.

Every operation is generated code: the scores are the attention-score kernel
over the row's tail (s.idxAttn), the ReLU the ungated activation kernel, the
head sum an axpy, the top-k the sampler's segmented ordering
(nn.SampleOrder), and the mask an axpy of the bias row.

### convert/flagship.go: gemma4Config

gemma4Config is llama.cpp's gemma4.cpp and transformers' Gemma4ForCausalLM,
text only:

	gemma3's block: sqrt(n_embd) embedding, GELU-tanh,   -> EmbdScale, FlagGELU,
	  per-head q/k RMSNorm, NEOX, post-attention and         FlagQKNorm, FlagRopeNeox,
	  post-FFN norms, a final softcap                        tensors, FinalSoftcap
	sliding layers by attention.sliding_window_pattern  -> SWAWindow, SWAPeriod
	the global layers' head_dim, kv heads and rotary    -> HeadDim, NKVHead, NRot
	  (key_length, head_count_kv at a global layer,
	  rope.dimension_count) and the sliding layers'     -> HeadDimSWA, NKVHeadSWA,
	  (key_length_swa, rope.dimension_count_swa)           NRotSWA
	the global layers' "proportional" rotary            -> RoleRopeFreqs (1 for the
	  (NEOX over the whole head, base^(-2i/head_dim)        rotated pairs, 1e30 for
	  for the first quarter of the pairs, zero after)       the rest), global only
	the sliding layers' own base                        -> RopeBaseSWA
	a weightless RMSNorm on each head of v              -> Flag2VNorm
	no v_proj on a global layer (attention_k_eq_v):     -> the absent RoleAttnV
	  v is k's projection, normed without k's weight
	an attention scale of one (self.scaling = 1.0)      -> AttnScale 1
	layer_scalar on each block's output                 -> RoleLayerOutScale

The 26B's block runs a dense MLP and a mixture in parallel (jlm.Flag2DenseMoE):

	the dense MLP, ffn_norm before it, post_ffw_norm_1 after -> RoleShExpGate/Up/Down
	                                                            (ungated), RolePostFFNNorm1
	the experts, pre_ffw_norm_2 before, post_ffw_norm_2 after -> RoleFFNNorm2, RolePostFFNNorm2
	the router on rmsnorm(x) * router.scale * n_embd^-1/2,     -> RoleRouterNorm, the
	  softmax, top-k, renormalised (a literal in every            scale folded here
	  reference), times per_expert_scale                       -> RoleExpScale
	the two branches' sum, post_ffw_norm after                -> RolePostFFNNorm

The E2B/E4B add three things:

	per-layer embeddings: a second token table, NLayer x  -> Config.PLEDim, RolePLE*
	  embedding_length_per_layer_input wide, plus a
	  projection of the scaled embedding; each block
	  gates its slice in after the FFN
	the last attention.shared_kv_layers layers read the   -> Config.NKVShared; their
	  KV of the last earlier layer of their own kind         k/v tensors are dropped
	those layers' FFN twice as wide (use_double_wide_mlp) -> Flag2DoubleFFN

### convert/flagship.go: minimaxM3Config

minimaxM3Config is llama.cpp's minimax-m3.cpp and transformers'
MiniMaxM3VLForCausalLM (the text model of MiniMaxM3SparseForCausalLM):

	q and k RMSNormed per head, Gemma's (1 + w) baked  -> FlagQKNorm (llama.cpp's
	  into the weights by llama.cpp's converter            converter adds the 1)
	NEOX rotary on rope.dimension_count of each head   -> FlagRopeNeox, NRot
	a dense lead, then a sigmoid router with the       -> NDenseLead, FlagExpertSigmoid,
	  selection-only bias, renormalised, a routed         RoleExpProbsB, ExpertScale,
	  scale and one shared expert                         NFFNShExp (the shared path)
	gpt-oss's clamped SwiGLU (alpha 1.702, limit 7)    -> FlagSwiGLUOAI
	  in every FFN: the dense lead, the experts and
	  the shared expert
	MiniMax Sparse Attention on every block past the   -> IdxHeads, IdxHeadDim,
	  dense lead: per kv group the top attention.         IdxTopK, IdxBlock,
	  indexer.top_k blocks of block_size positions,       IdxLocal, RoleIdxQ/
	  each block by its best score, local_blocks          RoleIdxQNorm/RoleIdxK/
	  ending at the query's own forced in                 RoleIdxKNorm

The gate is SIGMOID and the routed weights renormalised whatever the file
says (RULE 7m): MiniMaxM3VLTopKRouter hardcodes both, the published
config states scoring_func "sigmoid", and llama.cpp's converter writes
expert_weights_norm true as a literal; a file saying otherwise describes a
model nothing runs and is refused. The swigluoai constants are literals in
llama.cpp's builder and the published config's own (1.702, 7.0); the GGUF
carries neither.

The engine caches the indexer's key as one more kv head of the attention
row, so its width must be the head's (128 and 128 on the published model);
one indexer head per kv group is the reference's own contract (llama.cpp
asserts it). A query's own block must be forced in (local_blocks >= 1, the
published 1): every kv group then reads the same number of positions,
which is what lets a device gather the groups' blocks into one key range a
row. Each of these is refused by name otherwise.

### convert/deepseek4.go: deepseek4Config

deepseek4Config is llama.cpp's deepseek4.cpp and transformers'
DeepseekV4ForCausalLM, text only:

	hyper_connection.count/sinkhorn_iterations/      -> HCMult, HCIters, HCEps
	  epsilon: the residual is that many streams,       (RoleHC*)
	  collapsed into each sublayer and mixed out of it
	q_lora_rank: q = q_b(rmsnorm_w(q_a(x))), each      -> QLoraRank (RoleAttnQA/
	  head RMSNormed without a weight                     QANorm/QB)
	one kv head, key_length wide, rmsnorm_w'd; the     -> NKVHead 1, RoleAttnK,
	  rotated row is both the key and the value           RoleAttnKVANorm
	the rotary on the LAST rope.dimension_count of     -> NRot, interleaved
	  each head, adjacent pairs (llama.cpp's NORM)        (no FlagRopeNeox)
	the sliding blocks at rope.freq_base, plain; the   -> RopeBaseSWA (plain),
	  compressed ones (and their compressor and          RopeBase + the YaRN keys
	  indexer) at compress_rope_freq_base with YaRN
	attention.sliding_window on every block            -> SWAWindow, allLocal
	per-head sinks                                     -> RoleAttnSinks
	compress_ratios per block: 0 none, the CSA rate    -> CompKinds, CompRateCSA,
	  (a block with the indexer), the HCA rate            CompRateHCA
	the indexer: head_count, key_length, top_k         -> IdxHeads, IdxHeadDim,
	                                                      IdxTopK (RoleIdxQB/Proj,
	                                                      RoleIdxComp*)
	output_group_count, output_lora_rank               -> OGroups, OLoraRank
	sqrt(softplus) gating, the selection bias,         -> Flag2ExpertSqrtSoftplus,
	  renormalised, a routed scale, one shared expert     RoleExpProbsB, ExpertScale
	hash_layer_count: the first blocks select by       -> NHashLayers,
	  ffn_gate_tid2eid[token]                             RoleHashExperts
	swiglu_clamp_exp/shexp (10 in every block)         -> Flag2SwiGLUClamp

The gate is refused unless sqrt-softplus (llama.cpp's loader refuses the
rest; transformers would honour softmax and sigmoid through ACT2FN, which no
published V4 states), and the renormalisation unless on: both routers
divide by the sum unconditionally in transformers, so a file stating
expert_weights_norm false describes a model nothing runs. The clamp is a
literal of the architecture (ds4SwiGLULimit) and another limit is refused.

The engine keeps a compressed block's pending compressor inputs in the
window's own cached rows, so the window must cover what the compressor
still reads: the HCA rate, and two CSA windows (the overlap). DeepSeek V4's
own (128 against 128 and 8) does; another is refused by name.

## Decision models

Laya (ModernBERT and its decision head), d1 and Lev (label-token readouts
over lfm2 and qwen35) answer TypeSafe's `/v1/systemone`. The research, the
readouts, the divergences and the gates with their numbers are in
`docs/design/decision-models.md` (sections 8 and 9). The bisection that
found the one bug on the way: the fixture's head runs 2 heads of 64 beside an
encoder of 4 heads of 32, and `JIT.AddAttn` keeps ONE default attention set,
so registering the head's width replaced the encoder's and every encoder
layer attended at the wrong width (layer 0 off from its first block); the
encoder and the head now each hold the set for their own geometry
(`JIT.AttnSetFor`). The real checkpoint could not have shown it: 1024 wide at
16 heads, both widths are 64.

## LFM2-VL and d1's pictures

d1-3B (LiquidAI/d1-3B, its GGUF and `mmproj-d1-3B-F16.gguf`) reads
pictures: llama.cpp's projector type `lfm2`, `jlm.ProjLFM2`. The tower is
Phi-4-reasoning-vision's SigLIP2 NaFlex with the post-LayerNorm kept
(transformers' `last_hidden_state`; all 27 blocks, `vision_feature_layer`
absent), the projector InternVL's shape: the 2x2 unshuffle, a LayerNorm
when the file has one (d1's has none), linear, GELU, linear.

What the file does not state is the class's: Lfm2VlImageProcessor's 64 to
256 tokens a piece, 2 to 10 tiles of 512 pixels past twice the budget, a
thumbnail at the smart-resized size. The converter writes them into the
container (`Vision.MinPixels/MaxPixels/MinTiles/MaxTiles` and a new tail,
`TileSize`, `Tolerance`), and the file's `image_resize_algo` as
`FlagVisionBicubic` (the class default is bilinear; d1's file says bicubic).
The tile choice is InternVL's `gridOf` unchanged; the resize is torchvision's
8-bit one (`rgb8.resize`), which differs from Pillow's in a few dozen samples
a picture (its int16 weights against Pillow's 22-bit ones).

Two places jitllm follows the model's own code and not llama.cpp
(RULE 7m): the projector's GELU is transformers' erf form (llama.cpp's lfm2
graph runs the tanh form; d1-omni's runs erf), and a picture over 1024x1024
is first resized to that many pixels with Pillow BICUBIC, as d1's
`runner.cap_pixels` does before its processor; llama.cpp hands mtmd the
picture as it came. The GELU form is under the int8 activations' noise
(projector alone, 2.443e-3 erf against 2.443e-3 tanh on quad-and-disc).

A decision request carries pictures as llama.cpp's server reads them
(`model.DecisionMedia`): `files` then `images` data URLs, then every
`image_url` part of a chat-message state, taken out of its message; at most
8; `videos` refused. The systemone template writes one marker per picture
and each becomes its LFM2 run (`<|image_start|>`, `<|img_row_R_col_C|>` and
the tile, `<|img_thumbnail|>` and the thumbnail, `<|image_end|>`). Every
question re-prefills the picture's rows from the model's image cache; the
tower runs once per request.

What the gates measured (laptop i7-12700H, AVX2 host, RTX 3050 Ti 4 GB under
CUDA and Vulkan; M4 mini, arm64 host and Metal):

| gate | result |
|---|---|
| `TestLFM2LayoutMatchesProcessor` | 21 sizes, 11 tiled, every plan the processor's |
| `TestLFM2RealMatchesTransformers`, pixels | 0 differing values on 4 pictures (1, 1, 5 and 4 pieces) |
| tower rows against transformers f32 | NMSE 4.4e-5 to 1.3e-4 |
| projector alone on the reference rows | 2.4e-3 to 5.9e-3; transformers' own projector with int8 inputs reads 2.7e-3 against itself on the photo |
| violations | bilinear resize moves 2e5-2e6 samples; no thumbnail fails by shape; transposed table NMSE 0.33-1.12; transposed unshuffle 0.10-0.37 |
| tower, all 27 blocks on CUDA / Vulkan / Metal | 1.6e-3 / 6.6e-4 to 1.9e-3 / 1.7e-3 to 3.7e-3 against the host |
| `TestD1DecidesOverPicturesAsItsReference` (d1's runner.py, f32, 8 requests of 3 questions) | worst probability 0.0263 (laptop host), 0.0183 (M4 host), 0.0160 (Metal, every text block); another picture's rows read 0.97 |
| `TestD1PicturesMatchLlamaCpp` (llama-server, same GGUF and mmproj) | positions 978 and 702, llama.cpp's exactly; worst probability 0.0083 and 0.0091 |
| `TestWarmImageEncodeDoesNotAllocate/lfm2` | 0 allocations |
| `TestEveryTowerRunsSSE/lfm2` | pixels bit-identical, encode NMSE 9.6e-4 against AVX2 |

The fixture (`scripts/lfm2visiongold.py --fixture`, `synth-lfm2vl`):
transformers' Lfm2VlForConditionalGeneration at toy size, every parameter and
buffer randomised, with the projector LayerNorm d1 lacks, both GGUFs from
llama.cpp's converter (the text model as the Lfm2ForCausalLM it is, the tower
renamed to the published checkpoints' `vision_tower.vision_model.*`, three
tower blocks because the converter names `linear_2` by a block index), and a
1100x700 picture cut into 3x2 tiles and a thumbnail. Against transformers at
the engine's arithmetic (Q8 weights and q8 activations in the reference): the
tower 3.8e-5, the projector 1.3e-4, the projector alone 5.7e-8; every prompt
position the processor's (markers, picture rows, text); logits from the pixels
7.8e-7 with no argmax flip; every text and tower block on CUDA 7.0e-10 and
Vulkan 2.0e-12 against the host. Each feature removed is seen by the projector
alone: no LayerNorm 0.23, the tanh GELU 1.5e-6 (26x the clean arm, the fixture
being where the erf form is visible), the transposed unshuffle 2.9e-2; the
transposed position table 1.9e-3 from the pixels.

Relocation and paging: `TestTowerEncodeResumesAcrossARelocation/lfm2` (the
tower's first blocks on the card, then home mid-encode, under a two-page
budget) is bit-identical on CUDA and Vulkan, and `TestD1PicturesMatchLlamaCpp`
answers both picture requests to the bit under a budget of four text pages
and two tower pages.

### Clef-Flash

Clef-Flash is qwen35 with a joint schema head (`jlm.DecisionClef`): one prompt
per request, every row's final residual kept through the prefill
(`State.rowTap`, under an embedding mode that pools nothing so no output
projection runs), and the head over them -- on the host's generated kernels,
or on the device holding the output projection as a program of
`kernels.DH*` launches (`jit_old/gpu/tier/dechead.go`). The random fixture
(`scripts/clefgold.py fixture`: the release's classes, every parameter and
persistent buffer random) is held to Clef's own code to 1.7e-6 on a logit,
and llama.cpp agrees with both to the reference's rounding.

The one bug on the way was in the fixture, not the engine: randomising
`named_buffers()` included the rotary's `inv_freq`, which is not persistent,
so the in-memory model the first goldens came from was one no checkpoint can
hold (0.16 of a logit off, every row but the first matching the reloaded
backbone to 2e-5). The goldens now come from the checkpoint as written, read
back as the release is read, and only persistent tensors are randomised.

### Clef's pictures and videos

A Clef request may carry `images` and `videos` (the request shape d1's
takes; a video is Clef's "frame array", here an array of frame data URLs or
`{"frames": [...], "fps": rate}`). Clef's own code (`encode_record`) puts
the processor's expansion of `"<|vision_start|><|image_pad|><|vision_end|>"`
per picture and `"<|vision_start|><|video_pad|><|vision_end|>"` per video,
then a newline, between `STATE:\n` and the state; the head's spans are row
indexes, so a picture's rows count as the processor's pads do
(`clefPrompt`). A picture is the Qwen3.5 tower's one Picture between its
markers. A video is Qwen3VLVideoProcessor's: frames sampled at 2 a second
(`np.linspace`, rounded half to even) -- at the 24 frames a second the
reference assumes when frames come with no rate --, every frame resized to
one size by the video smart_resize over all frames (torchvision's 8-bit
bicubic), paired by temporal_patch_size 2 with the last repeated, each pair
one Picture whose patches attend only to each other (the reference's
per-frame cu_seqlens), laid out as `<t seconds>` (the pair's two frames'
mean time, `%.1f`) and its rows between markers, all inside the
placeholder's own markers.

A pair is two different frames, which a container whose patch embedding was
folded to the planes' sum cannot read; `convert.foldPatchPlanes` now keeps
the second plane beside the sum (`RoleVPatchEmbd` index 1), and a pair's
row is x0*(W0+W1) + (x1-x0)*W1 (`State.patchPair`, generated gather,
difference and add). A container converted before carries no second plane
and refuses a video by name.

Gates (`TestClefMediaMatchesReference`, laptop host, RTX 3050 Ti under CUDA
and Vulkan): a fixture like Clef's random one but with the release's 48x48
position table (`clefgold.py fixture-vl`; the 8x8 table of the text
fixture reads a picture at 128 pixels where the processor reads 384), its
goldens with the reference's tower inputs and matrices rounded as the
container's are (`logits_a8`). Every row and span is the reference's. Two
pictures: 5.5e-4 from the reference's logits (8.1e-3 against its f32 tower),
5.2e-4 on CUDA and 5.5e-4 on Vulkan with every block; the pictures left out
move 0.126. A video of ten frames (four sampled, two pairs): 1.4e-4 (2.3e-3
against the f32 tower), 2.0e-4 on CUDA; the media left out move 0.066, and
each pair's second frame read as its first 0.081.

Two places Clef's processor and this engine's tower part, recorded rather
than built: Clef's image bounds are 65536 to 16777216 pixels where every
Qwen-VL tower here is bounded by the mmproj's image_size squared (768 for
Clef, llama.cpp's convention), so a picture over 768x768 pixels is read
smaller than Clef reads it; and a video frame larger than the tower's
bound is refused by name rather than shrunk.

### d1-omni: an LFM2 encoder under Laya's head

LiquidAI's d1-omni-600M is LFM2.5-Encoder-350M (LFM2's sixteen blocks made
bidirectional) with Laya's decision head on top: two pre-norm torch
TransformerEncoderLayers with a ReLU MLP, the question type's row added to
every row first, and a LayerNorm-Linear-GELU-Linear scorer at each option's
`<|mask|>`. llama.cpp's converter writes it as an `lfm2` file whose
block_count counts the head's two blocks and whose per-layer arrays run over
all eighteen; `convert/d1omni.go` reads the trunk through a view of the file
cut to sixteen blocks and states it as `jlm.ArchLFM2Enc`, so a reader that
knows only the decoder refuses the container rather than running it under a
causal mask. The head's fused, biased q|k|v is split in thirds of the
residual (nn.MultiheadAttention's in_proj), whatever the trunk's GQA.

What separates the trunk from LFM2's decoder, each removed by
`TestD1OmniFeaturesAreLoadBearing` (worst logit 12.3, 12.3 and 1.1 off):

- the short convolution is centred, `w0 bx[t-1] + w1 bx[t] + w2 bx[t+1]`
  with zeros past either end, where the decoder's is causal over three past
  rows. On the host it is three generated multiplies of `B*x` by each tap
  repeated over the rows, two of them added one row shifted;
- no causal mask on the attention;
- a media prefix's rows attend to the prefix alone, and the prefix's last
  row does not read the text's first through the convolution (encoder.py's
  `keep_right`); both are in the code, and the pictures' gate, which comes
  next, holds them.

`prompt.encode` is transcribed as written: each piece tokenized alone after
`<|name|>` is rewritten to `<¦name¦>`, the option block given
max(96, min(24k+32, max_len/2)) tokens shared evenly, the question cut to
that block and the state to what is left. A noul is read as [false, true],
and the answer's confidence is the chosen option's probability.

The GGUF stores every matrix in F16 and the checkpoint is F32, so a golden
from the checkpoint as it is puts the engine 1.2e-2 off on the trunk -- the
weights, not the graph. `scripts/d1omnigold.py` with `D1OMNI_F16=1` rounds
each 2-D weight but the type table to f16 first, as the GGUF does, and the
engine then sits at f32 order: 4.6e-5 on the trunk at the markers, 2.8e-5 on
a logit, 5.5e-6 on a probability (`TestD1OmniMatchesReference`).

A warm pass over every prompt makes no engine allocation, and half the
blocks' pages give the same scores to the bit (`TestD1OmniKeepsThePrinciples`).
On a device the trunk is the encoder segment's non-causal blocks
(`engine/model/encdev.go`): RMSNorm, SwiGLU, GQA with the per-head q/k norm
and NEOX rotary on the attention blocks, and on the convolution blocks a
short convolution with no state, `nn.RecurrentPlan.Centred`, run by
`kernels.Conv3Centred` -- three taps on the row before, at and after, each
read only where the row's window holds it. The windows are the ones the
attention's mask reads (`nn.RowWindowDevice`, one a row: a prefix row's the
prefix, a text row's the whole call), copied whole into one buffer for the
convolution (`blockScratch.vwinAll`), so the prefix's last row's cut and the
zeros past either end are the same fact on both. The convolution and
attention blocks share one set, built from the union of their plans. The
head attends at 16 key heads with LayerNorm after a trunk of 8 with RMSNorm,
and every kernel of a set bakes those, so the head runs in a non-causal set
of its own (`nn.LayerPlan.HeadSet`, the tier's `geoNCHead`), resized with the
trunk's and never row-chunked.

`TestD1OmniOnEveryDevice` places every block, three, and eight (a seam that
leaves the trunk's rest home) and compares the text and the picture
questions' scores with the host's: CUDA 8.6e-3, 2.8e-3 and 3.9e-3 on the
laptop's RTX 3050 Ti, Vulkan 1.7e-5. With the prefix reading the text on the
device the scores move 1.0. The kernel alone against its loop, the prefix
windows included: `backend.TestConv3CentredMatchesTheLoop`, which fails when
the right tap ignores the window. A warm question makes no allocation on
either device at a seam or with every block placed.

#### d1-omni's pictures

The mmproj's vision half (`d1omni_v`) is LFM2-VL's: SigLIP2 NaFlex with its
post-LayerNorm, the 2x2 unshuffle, mm.1, erf GELU, mm.2, no input norm, and
vision.py's plan -- LFM2-VL's tiles and thumbnail, resampled bilinear with
antialias (no image_resize_algo in the file, so the class default). It
converts as `jlm.ProjLFM2`. The mmproj also carries the audio encoder
(`d1omni_a`), which this converter does not build yet: its `a.*` and `mm.a.*`
tensors are set aside (`withoutAudio`), so a request with a voice clip finds
no audio encoder and is refused rather than run without one.

A picture is not marked in the prompt: every piece of every picture goes
through the tower, and the projected rows in order are a prefix in front of
the prompt's embeddings. The prefix rows attend to the prefix alone, the
prefix's last row does not read the text through the convolution, the head
reads the text rows, the prompt is cut to image_text_length (896) and the
noul's default wording is the reference's YES_NO ("no", "yes"). The
reference calibrates text alone: an answer over pictures takes no
temperature. The trunk's context is max_length (16384), stated at conversion
(`decision.max_length`, D1OmniConfig's default) where the GGUF says 128000.

The encoder's Decider runs the tower on its own JIT through a vision
segment (`Tower.newState`), so the encoder's JIT is built with the tower's
formats and widths, and `Model.pageIn` sends a tower block through the
layers' bind rather than the encoder's.

The tower is stored Q8_0 and runs on int8 activations (convert/vision.go,
the converter's one lossy step), and the reference reads F16 weights: the
prefix rows sit at NMSE 1.6e-3 against the reference's (an outlier channel
near -110 is 26 off), the head's logits 0.21 and the probabilities 5.2e-3
away (`TestD1OmniDecidesOverPicturesAsItsReference`). The prefix length and
the prompt after it are exact. Letting the prefix read the text moves a logit
0.91, and swapping the two pictures 1.53.

#### d1-omni's voice: the first audio tower

The mmproj's audio half (`d1omni_a`) is audio.py's: NeMo's log-mel front
end, an 8x convolutional subsampling, 17 FastConformer blocks and an adapter
with a residual correction. It is carried as tower blocks after the vision
blocks in the one tower page array (`jlm.Audio`, an optional tail of the
Vision record, so a container with no audio is the bytes it was). The
converter reads what the file states and writes what audio.py hardcodes
(16 kHz, a 512-point transform of 400-sample windows every 160); it cuts
pointwise_conv1 into the GLU's halves, makes the depthwise convolutions
tap-major, and reorders the subsampling's output matrix to read its 4096
inputs channel-last, the layout the engine's activations take.

On the host every product is the generated matmul and every elementwise step
a generated kernel:

- the |STFT|^2 is two matmuls of the centred frames (one strided copy)
  against the Hann-windowed cos and sin bases, built once at load; the mel
  filterbank is librosa's Slaney one, transcribed; the log is a new
  elementwise kernel (`nn.Ln32JIT`, the log-softmax's ln on every lane, on
  AVX2, SSE and NEON); each feature's normalising scale is one scalar from
  the generated sums;
- the subsampling's first convolution is nine strided copies and a matmul,
  its depthwise ones the tower's existing depthwise rows (`DWConvRow`), each
  layer's input cut at the frames still valid, and only the valid rows go on:
  the conformer's mask is exactly that cut, and a relative position does not
  depend on the length;
- the relative-position term of row i is the score kernel over the projected
  position table read from row t-1-i, so the reference's rel-shift is an
  offset, not a gather;
- the convolution module's depthwise convolution is its taps' shifted
  multiply-adds, BatchNorm folded into a scale and a shift by the converter.

A voice clip arrives as a data:audio WAV in "files" (16 kHz mono 16-bit PCM;
any other is refused, the engine does not resample), at most one and never
beside a picture, as llama.cpp's server takes it. The prompt after it is
written as the audio questions were trained (a choice's options as
`option_000: <description>`, a noul as plain no and yes, a null state as
{}), cut to audio_text_length (15360), with no temperature.

Against audio.py with the F16-rounded weights (`TestD1OmniHearsAsItsReference`,
on the repo's 3.3-second clip): the log-mel features 1.7e-6 off, the 41
prefix rows at NMSE 1.2e-4 (the tower at Q8_0), the prompt token for token,
the logits 0.039 and the probabilities 0.012 away. Removing the
relative-position term moves the answers 0.32, the convolution module 0.63,
writing the questions as text 0.11 (`TestD1OmniAudioFeaturesAreLoadBearing`);
a warm clip's encoding allocates nothing. llama.cpp's d1-omni server reads
the same 251 positions and answers within 0.012
(`TestD1OmniVoiceMatchesLlamaCpp`).

On a device each FastConformer block is a convolutional-tower block
(`nn.ConvConformer`), a program of generated kernels as Gemma 3n's
MobileNet blocks are (`jit_old/gpu/tier/conv.go`): LayerNorm in three passes,
the matrices' batched twins with a bias launch after each, the macaron
halves as `AddHalf`, the relative-position scores in one kernel
(`kernels.RelPosScores`: both terms, the position table's row picked by the
offset), the row softmax, the weighted sum (`AttnAccRows`), the GLU's
`SigmoidMul`, the depthwise convolution (`DWConv1dRows`), the folded
BatchNorm (`AffineRows`, the depthwise bias folded into its shift at load)
and SiLU. The program is built for the 30 seconds' 375 rows and a call
writes its own row count to a one-word buffer: the keys, the weighted sum
and the convolution stop there by SELECTING, not by multiplying by zero, since
the rows past it hold whatever the scratch held. The projected
position table is a matmul of a constant sinusoid table, 2R-1 rows, per
block per call. The audio blocks are placed through their own encoder
segment State (`audState`), the one placement path; the front end, the
subsampling, the adapter and the residual stay on the host.

`TestD1OmniHearsOnEveryDevice` places every audio block: the prefix rows sit
at NMSE 1.3e-4 against the host's on the laptop's RTX 3050 Ti under CUDA and
Vulkan, the answers 0.008-0.012 from the reference, and a warm clip makes no
allocation (the convolutional submission's closure and its run list were
made per call; they are the devTier's now). Swapping the attention's u and v
on the device moves the rows to NMSE 0.097. The kernels against their loops,
rows past the call poisoned NaN: `backend.TestConformerKernelsMatchTheLoops`.

## qwen4exp: Qwen3.8-Flash-Next (plan, before the code)

The graph, read off llama.cpp's `src/models/qwen4exp.cpp` and its converter
(`conversion/qwen4exp.py`), transformers' `modeling_qwen4_exp.py` and the
header of unsloth's UD-Q4_K_XL GGUF, before anything is written. The
`llama-eval-callback` transcription of a real file replaces this reading once
a file is converted (RULE 7: read the artefact).

48 blocks, every fourth full attention, the rest qwen35's gated delta rule
(16 key heads, 48 value heads of 128, conv 4). Per block:

    [PLE, block 2 only]  res += gated + silu(dilated_conv(norm_conv(gated)))
    in  = HC mix(res)            -> attention or delta rule
    res = res + 2*sigmoid(inject/H) * out          (each stream its own weight)
    in  = HC mix(res)            -> 512-expert softmax top-10 renormalised,
                                    one sigmoid-gated shared expert (640 wide)
    res = res + 2*sigmoid(inject/H) * out

    HC mix:  xn   = per-stream RMSNorm of the H=4 streams (gamma folded 1+w)
             gate = sigmoid(up(silu(down(xn)/H)))     low rank 320
             in   = mean over streams of gate*xn
    head:    the mix with no inject, then output (untied); no output norm.

- **Delta rule**: qwen35's, but the gated norm's gate is sigmoid, not silu
  (`output_gate_type`). Value heads grouped as qwen35 (the tiled pairing,
  `FlagDeltaKeyTiled`, is a question the fixture answers, not an assumption).
- **Attention**: qwen3next's gated double-width query, per-head q/k norm,
  interleaved M-RoPE on a quarter of the 256-wide head (sections 11/11/10),
  24 heads over 2 kv heads; plus QSA: a 4-head indexer of 128 whose raw key
  per position is cached; per complete block of 4 visible positions the mean
  key, RMSNormed and rotated at the block's first position; score =
  sum_h relu(q_h . k_block); the top budget/4 blocks plus the incomplete tail
  are the only keys the query attends. Under budget+3 visible positions
  (2051 on the real file) every key is selected and QSA is dense attention.
- **PLE**: one layer. 16 heads (2-grams and 3-grams, 8 each) hash the token
  and its two predecessors (`(t0*m0)^(t1*m1)^(t2*m2) mod p_h + off_h`, an EOS
  in the window or the sequence start reading as EOS), gather 160-wide rows of
  a 320M-row table into 2560, then key (per stream) and value projections,
  a signed-sqrt sigmoid gate per stream, and a depthwise causal conv of
  kernel 4 dilated by 3 over the normed gated value, whose 9-position history
  is per-sequence recurrent state.

Reuse map:

| feature | jitllm machinery | new |
|---|---|---|
| delta rule, conv, shared expert, gated query, M-RoPE | qwen35 / qwen3next (`nn.IMRopeRuns`) | the sigmoid gated-norm gate: a flag and a gate kind on every tier |
| four streams, stream-major residual, seam/paging/step | deepseek4 and gemma3n (`Config.streamHead`, `rowStreams`) | the mixer is not V4's Sinkhorn: a low-rank sigmoid gate and a mean, and a 2*sigmoid placement with the identity comb -- a kernel per tier beside `cpu.EmitHCMix` |
| QSA indexer and selection | minimax-m3's block selection (indexer key as one more cached head) and deepseek32/V4's relu-sum scorer | mean-pooled, normed, rotated block keys; the tail rule |
| per-layer table | gemma3n/gemma4's per-layer embedding (`pleTok`, `State.spanTok`) | the n-gram hash (int64, a generated kernel or setup integer arithmetic -- to be settled against RULE 8 with the user), the token history as recurrent state, the dilated conv state |
| the 29 GB table | the pager (O_DIRECT, static pages) | a row-granular page array: a token reads 16 rows of 90 bytes, so the table is pages of rows read on demand under the budget, never resident whole and never mapped |
| vision | qwen3vl_merger (the tower is Qwen3-VL's, per the converter) | the image token id stands in for picture rows in the hash |

Arch code: `ArchQwen4Exp = 51`, the next flagship.

Quant: UD-Q4_K_XL is not K-quant only. Per tensor: experts Q4_K/Q5_K
(gate, up) and Q5_1/Q8_0 (down), attention and projections Q8_0, the
indexer BF16, norms and router F32, and `per_layer_token_embd` IQ4_NL
(160 x 320,001,536, 28.8 GB). IQ4_NL is outside `quant.Dequantable`; only a
row gather reads it, so the work is the oracle dequantizer, then a gather
kernel per tier. The other route, repacking IQ4_NL to Q8_0 at conversion
(exact: IQ4_NL's values are a scale times an int8 table), costs 25 GB more.
Strata's IQ2_XS/IQ3_XXS/IQ3_S/Q2_0 files are a follow-up, each oracle first.

References that part (RULE 7m), to be decided as the code meets them:
the indexer's 1/sqrt(head_dim) (transformers scales, llama.cpp does not; a
positive scale on a relu sum selects the same blocks), the PLE EOS id
(transformers takes the first of a list, llama.cpp's converter the last; the
release states one), and MTP (both drop it; unsloth publishes it as a
separate GGUF).

## qwen4exp: what was built, and what the gates measured

The plan above held, with these settlements.

**The n-gram hash is generated code on every tier** (RULE 8): uint64
multiply, xor and modulo per head, the reset at an end-of-sequence id, and
the predecessors moved on (`cpu.EmitNGramHash`, AVX2/SSE/arm64; the only
setup integer work is the constants). The predecessors are recurrent state
the host keeps whichever side runs the block: the host hashes and reads the
table rows, a device gets the gathered rows (`nn.NGramDevice`).

**IQ4_NL is a native type** (`quant.IQ4_NL`, container v29): the oracle
decoder first, then a row-gather kernel per tier. The 28.8 GB table is a row
table placed after every page array and read a row at a time with O_DIRECT
(`jlm.File.ReadRow`); its page size once entered the per-block page sizing
and laid a 1.5 TB file out, which `TestRowTableIsReadARowAtATime` now holds.

**Every linear block's state slot is [delta rule | tail]**: the n-gram
block's tail is its predecessors and the dilated convolution's nine
positions of every stream, and every linear block carries the room so a
device keeps one slot width (`nn.RecurrentPlan.TailLen`; the delta kernels
take the stride). It costs the tail on the blocks that leave it unread:
on the real file 9 x 4 x 2560 floats and two words a sequence a block.

**The fixture's hyper-connection rank is 32**, not 16: a device quantizes
activations in blocks of 32, and the real rank is 320. The golden was
regenerated from transformers, not edited.

**The real file's graph**, from `llama-eval-callback` on cato (blocks 0, 2
and 3 of UD-Q4_K_XL at llama.cpp de7fa0a, `/mnt/models/qwen4exp/evalcb.log`):
the hyper-connection read is `hc_norm` (2560 x 4) -> `hc_attn_up` (320 x
10240 after the down) -> `DSV4_HC_PRE`; the placement `hc_attn_inject`
(10240 -> 4) -> `DSV4_HC_POST`; the delta rule's gate `SIGMOID(z)` times the
gated norm; 512 experts of 640 by softmax, top-10, renormalised over a
clamped sum; one 640-wide shared expert behind a sigmoid; block 1 (one-based
2) carries the PLE (`ple_gate`, `ple_gated_value`, its conv state). Each is
the transcription the host runs.

Measured, the fixture against transformers' own class:

| arm | worst logit NMSE |
| --- | --- |
| host decode, prefill at every length, batch (amd64 AVX2) | 2.4e-10 |
| host on an SSE4.2 Atom C3558, natively | within the golden's bound |
| host arm64 (M4) | within the golden's bound |
| every block and the head on CUDA (RTX 3050 Ti), decode / prefill | 2.1e-10 / 9.5e-12 |
| the same on Vulkan (RTX 3050 Ti) | 3.3e-10 / 9.8e-12 |
| the same on Metal (M4) | 3.1e-11 / 7.9e-12 |

Thirteen violations each move the logits 0.06-6.5 NMSE on the host and on
every device (`TestQwen4ExpFeaturesAreLoadBearing`,
`TestQwen4ExpOnEveryDevice`). The real checkpoint matched llama.cpp
teacher-forced 20 of 20 on a short prompt (QSA is dense there).

**Four device bugs the end-to-end gate found and the kernel gates could
not**, each green at the kernel level (`backend.TestQ4EKernelsMatchTheLoops`)
and wrong in the tier's wiring (RULE 11c):

- the placement writes every stream into `bs.x2`, which held one stream
  unless the model was DeepSeek V4 or Kimi-K3: stream 2 landed on whatever
  followed it. Bisected by reading the scratch after one block: three
  streams agreed at 1e-13 and one did not.
- an attention block read `bs.x` where its hyper-connection's mix belonged.
- QSA's row is k's heads and the raw key, copied whole; the rotary writer
  took the row's width from the heads it turns and wrote the next heads
  into the key's place (`kernels.PagedRoPERowsTIn` takes the row's width).
- the head reads its mix with no norm; the device allocated a zero-length
  norm and declined it.

**QSA on the device** is MiniMax-M3's select, compact and gather under a
forced tail (the tail block at end/blk, empty on a block boundary) and a
scorer of its own (`kernels.QSAScores`): each complete block's raw keys
pooled, normed and turned at the block's first position, scored by the
relu sum. A picture moves the positions after it (M-RoPE's marks): the
scorer composes the block start's turn with the mark's, both from the
device's rotary table (`TestQwen4ExpQSATurnsByTheMarks`, which fails at
NMSE 0.45 when the marks are not handed over). The host had the same
question wrong: `qsaMask` turned another session's blocks by this
State's slot 0.

**Gates run for it**: the five principles' lists (paging, relocation
between devices, KV pages with the layer, faulting a batch's blocks in,
zero-allocation decode on the host and every device), the batch seam
(`TestBatchSeamMovesCarryEveryRowQwen4Exp`), the prefix cache
(`TestPrefixCacheRestoresQwen4Exp`: the restored state slot carries the
hash's predecessors and the convolution's history; clearing the tail on
restore moves the continuation by 0.30 NMSE), and `TestEveryModelRunsSSE`.
A q8_0 KV cache is refused by name: QSA pools raw cached keys, which its
references hold at f32.

**Found on the M4, not this model's**: `TestKVPagesRelocateWithTheLayer`
opens no fused run there (0 of 624 asked for, and the same for
synth-kimik3), so its fused-run check is red on arm64 whatever the model.

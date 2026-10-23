# Emitter patterns: factoring the code that writes kernels

A read-only study of the kernel emitters: what repeats, how much could be shared, and the golden-output gate that keeps every generated kernel byte-identical while the emitter code is factored. Optimisations that change generated output are separate, measured follow-ups.

**Estimated net reduction: 3,000–5,300 lines of emitter source, roughly 8–13% of the 39,435 lines in kernel-family files inventoried below.** That is an engineering estimate from repeated structures, after allowing for replacement helpers; it is not a measured refactor result. The strongest opportunities are within matrix families, row reductions, and attention staging. Removing half the emitters would require much broader abstractions than the exact-output constraint currently justifies.

The refactor should preserve every specialised kernel. Its acceptance criterion is **identical emitted output for each configuration**, including instruction order, physical registers, IR operation order, parameter order, and launch metadata. Optimisations that change those outputs belong in subsequent, separately measured changes.

**What the golden gate does not govern.** The byte-identical gate guards an emitter refactor of one configuration, old against new, and nothing else. Generated kernels may and should differ per model and per hardware wherever that is faster: a different fusion, schedule, reduction order or tile per device is not a regression. Those forms are held to the reference (the oracle, transformers or llama.cpp) within a measured error band -- NMSE or the f32 reduction-order band, with a deliberate violation that must fail -- and the faster form is chosen per device by timing. Requiring a kernel to be bit-identical across hardware or across fused and unfused forms is an invented constraint (RULE 11e).

Studied checkout: `91e2ad656ebddf9ad0d5645e3f5544c8045f0bbc`. No files changed; no builds, tests, or benchmarks run.

## 1. Inventory

Counts are physical lines, including comments and blanks, excluding `_test.go`. Mixed files include their validation, shape definitions, and helpers. Consequently, these are reproducible **source-file counts**, not a claim that every counted line emits an instruction.

An important correction to the requested inventory: **this checkout has no AVX-512 emitter tier.** It probes AVX-512 capabilities, but its tiers are SSE, AVX2, and NEON. AVX-VNNI is a dot-product choice inside AVX2; it is not AVX-512. See [tier.go:5](../../jit/cpu/tier.go), [features_amd64.go:60](../../jit/cpu/features_amd64.go), and [prevnni.go:25](../../jit/cpu/prevnni.go).

### Host emitters

All filenames in this table are relative to `jit/cpu/`; numbers following filenames are their line counts.

| Tier | Family | Files and lines | Total |
|---|---|---|---:|
| AVX2, with VNNI/VEX dot variants where supported | Row-major quantised/float matvec, GEMM; packed, wide, fused, token-tiled and stationary matrices | `matvec.go` 1,368; `gemm.go` 206; `gemmk.go` 741; `packed.go` 530; `packedwide.go` 340; `packedfused.go` 510; `packedtiled.go` 518; `packedgemm.go` 707 | 4,920 |
| AVX2 | RMSNorm, LayerNorm/GaussTopK/MagMatch, activations, AXPY/scale/clamp/softcap, softmax, exp | `norm.go` 126; `layernorm.go` 200; `act.go` 222; `axpy.go` 135; `softmax.go` 100; `exp.go` 77; `elemloop.go` 52; `xielu.go` 69 | 981 |
| AVX2 | Attention scores/accumulation, paired/tiled variants; rotary application/table | `attn.go` 469; `rope.go` 191; `ropetab.go` 197 | 857 |
| AVX2 | Delta/KDA, SSD/selective scan, gates, causal/depthwise convolution, hyperconnections/pooling | `delta.go` 341; `gate.go` 207; `conv.go` 85; `dwconv.go` 72; `hc.go` 167 | 872 |
| AVX2 | Argmax, MoE routing/top-k, sparsemixer, sampling | `argmax.go` 147; `topk.go` 523; `sparsemixer.go` 83; `sample.go` 374 | 1,127 |
| AVX2 | Activation packing/quantisation, packed-row extraction/widening, image entry points/resampling | `packact.go` 235; `quantact_narrow.go` 98; `rowpacked.go` 237; `imgproc.go` 36; `resample.go` 86 | 692 |
| SSE | Float row-major and packed/fused matrices; explicit optional-form refusals | `sse_matvec.go` 221; `sse_packed.go` 853; `sse_refuse.go` 54 | 1,128 |
| SSE | Maps, norms/row statistics, packed-row/widening, rotary, exp/reduction helpers | `sse_elem.go` 445; `sse_norm.go` 716; `sse_helpers.go` 271; `sse_ropetab.go` 168; `sse_ropesplit.go` 74; `sse_xielu.go` 63 | 1,737 |
| SSE | Attention, recurrence/gates/convolution, hyperconnections/pooling, depthwise shape/emission | `sse_attn.go` 525; `sse_hybrid.go` 555; `sse_hc.go` 168; `dwshape.go` 84 | 1,332 |
| SSE | Argmax, routing/top-k, sparsemixer, sampling | `sse_argmax.go` 123; `sse_topk.go` 446; `sse_sparsemixer.go` 85; `sse_sample.go` 313 | 967 |
| SSE/shared x86 image bodies | Quantisation, image conversion/filtering/copy/position tables, resampling | `sse_quantact.go` 291; `imgproc_sse.go` 745; `sse_resample.go` 62 | 1,098 |
| NEON | Row-major matvec/GEMM and unpacking; packed/fused/tiled/stationary matrices | `matvec_a64.go` 389; `matveck_a64.go` 347; `matveck_a64_unpack.go` 95; `gemm_arm64.go` 194; `gemmk_a64.go` 691; `packed_a64.go` 1,243; `packedgemm_a64.go` 514 | 3,473 |
| NEON | Maps, norms/row statistics, softmax/exp; rotary | `elem_a64.go` 760; `rope_a64.go` 84; `ropesplit_a64.go` 70; `ropetab_a64.go` 199; `xielu_a64.go` 66 | 1,179 |
| NEON | Recurrence/gates/convolution, hyperconnections/pooling | `delta_a64.go` 262; `gate_a64.go` 159; `conv_a64.go` 78; `dwconv_a64.go` 59; `hc_a64.go` 159 | 717 |
| NEON | Attention scores/accumulation, paired/tiled variants | `attn_a64.go` 433 | 433 |
| NEON | Argmax, routing/top-k, sparsemixer, sampling | `argmax_a64.go` 124; `topk_a64.go` 429; `sparsemixer_a64.go` 83; `sample_a64.go` 310 | 946 |
| NEON | Activation packing/quantisation, row extraction/widening, image kernels/wrappers | `packact_a64.go` 115; `quantact_a64.go` 257; `rowpacked_a64.go` 214; `imgproc_neon.go` 746; `resample_a64.go` 7; `imgproc_arm64.go` 13 | 1,352 |
| AVX-512 | No separate emitter implementation | Capability probing only | 0 |

**Host family subtotal: 23,811 lines.**

The table’s dispatch contract is explicit in [emitters.go:28](../../jit/cpu/emitters.go). SSE deliberately uses existing generated fused kernels where optional wide/tiled forms are unavailable; those refusal wrappers are not additional kernel implementations ([emitters.go:119](../../jit/cpu/emitters.go)).

Additional host infrastructure, excluded from that subtotal:

| Purpose | Files and lines |
|---|---|
| Instruction encoding and ISA declaration | `asm.go` 1,012; `sse.go` 184; `sse_ops.go` 790; `a64.go` 947; `a64img.go` 108; `isa.go` 101 — **3,142** |
| Dot-product instruction selection | `prevnni.go` — **237** |
| Shared layouts, constants and shape policies | `actkind.go` 32; `elem_const.go` 124; `imgproc_const.go` 327; `resample_const.go` 88; `ropetab_const.go` 87; `rowconst.go` 46; `rowmode.go` 13; `sample_const.go` 87; `topk_const.go` 210; `sparsemixer_const.go` 63; `xielu_const.go` 45; `hc_const.go` 40; `dplane.go` 53; `scstream.go` 71; `fusedgroup.go` 72; `codes.go` 33 — **1,391** |

All non-test Go files directly under `jit/cpu` total **31,883 lines**. The remaining 3,302 lines are dispatch, mapping/execution, probes, drivers, hooks and unsupported-platform wrappers. They are not counted as template-removal opportunities.

### Device emitters

All filenames below are relative to `jit/gpu/kernels/`.

**The ordinary families produce shared IR once for all three backends.** They must not be counted three times. Matrix-unit variants differ: fragment MMA is PTX-only; collective tiles target MSL/SPIR-V, subject to supported shapes. The capability split is explicitly tested in [mmacap_test.go:14](../../jit/gpu/lowertest/mmacap_test.go).

| Family | Files and lines | Total |
|---|---|---:|
| Quantised/float matvec, split/segmented/indexed variants; integer/f16 MMA; staged GEMM | `matvec.go` 1,427; `mma.go` 656; `mma70.go` 718; `gemm70.go` 520; `gemmtile.go` 424 | 3,745 |
| Scores, softmax, accumulation, head norm; flash decode/prefill/merge; paged and staged variants | `attn.go` 1,283; `attn70.go` 323; `flash.go` 477; `flash70.go` 546; `flashkv.go` 600; `flashtile.go` 448; `mergerun.go` 69; `pagedprefill.go` 758; `pagedprefill70.go` 363; `pagedstaged.go` 557 | 5,424 |
| Quantisation, normalisation, maps, argmax, AltUp/LAuReL, XIELU, grouped norm | `elem.go` 1,052; `argmax.go` 104; `altup.go` 197; `xielu.go` 52; `ssd.go` 100 | 1,505 |
| Recurrent steps/scans and gates; DeepSeek-4/Kimi-K3 stream operations | `recurrent.go` 1,124; `ds4.go` 595; `k3.go` 144 | 1,863 |
| Expert selection, weighting, combination, grouping/sorting, sparsemixer | `moe.go` 1,022; `moegroup.go` 68; `moesort.go` 260; `sparsemixer.go` 77 | 1,427 |
| Lightning indexer and MiniMax sparse-attention selection/gather/descriptors | `indexer.go` 178; `msa.go` 336 | 514 |
| Paged writes/moves, row extraction/copy/unpack, rotary tables, per-layer embeddings, convolution/MQA | `paged.go` 297; `pagedmove.go` 29; `getrows.go` 198; `unpack.go` 231; `ropetab.go` 152; `ple.go` 30; `conv.go` 209 | 1,146 |

**Device family subtotal: 15,624 lines.**

Additional device-kernel support:

- **761 lines:** `actkind.go` 139, `config.go` 23, `decodeplan.go` 142, `divrn.go` 102, `quantof.go` 32, `ropetab_const.go` 171, `scstream.go` 85, `tile.go` 67.
- **1,318 lines:** packing/allocation and test switches: `pack.go` 649, `arena.go` 529, `arena_unix.go` 43, `arena_other.go` 27, `benchknob.go` 26, `benchknob_off.go` 7, `pagedfault.go` 29, `pagedfault_off.go` 8.

All non-test Go files directly under `jit/gpu/kernels` total **17,703 lines**.

### IR and backend lowering

| Layer/backend | Files and lines | Total |
|---|---|---:|
| Typed IR, builder, validation, fragment/collective matrix contracts | `jit/ir/ir.go` 757; `tile.go` 410 | 1,167 |
| CUDA/PTX | `jit/ptx/lower.go` 486; `buf.go` 121 | 607 |
| Vulkan/SPIR-V | `jit/spirv/lower.go` 695; `spirv.go` 141; `tile.go` 236 | 1,072 |
| Metal/MSL | `jit/msl/msl.go` 472; `tile.go` 140 | 612 |

These lowerers are already the shared backend mechanism. I would leave them outside the initial factoring effort. PTX performs an ordered walk of IR; driver compilation owns physical allocation and scheduling ([lower.go:9](../../jit/ptx/lower.go)).

### Where `engine/nn` consumes the emitters

These are orchestration/cache/dispatch files, not duplicated emitter source:

| Use | Files and lines under `engine/nn/` |
|---|---|
| JIT construction, shapes and matrix dispatch | `jit.go` 1,051; `matmul.go` 326; `packed.go` 664; `packedmm.go` 551; `packedmulti.go` 226 |
| Tuning and prefetch | `tune.go` 868; `mvpick.go` 165; `prefetch.go` 188; `prefillcores.go` 24; `options.go` 218 |
| Norms/maps, rotary and row access | `norm.go` 622; `rope.go` 67; `ropeaxes.go` 227; `ropetable.go` 318; `row.go` 102; `rowmajor.go` 35 |
| Attention, recurrence, convolution and mixing | `attn.go` 358; `delta.go` 193; `ssd.go` 121; `dwconv.go` 38; `hc.go` 93 |
| Selection/sampling/image | `argmax.go` 56; `moetopk.go` 385; `sparsemixer.go` 102; `sample.go` 341; `image.go` 386 |
| Device contract | `device.go` 1,379 |

For an output-preserving refactor, keep these call sites, selection decisions and cache keys unchanged. Relevant entry points include [norm.go:143](../../engine/nn/norm.go), [packedmm.go:322](../../engine/nn/packedmm.go), and [packedmm.go:517](../../engine/nn/packedmm.go).

## 2. Patterns and removable duplication

There is already substantial factoring:

- Host elementwise bodies use `elemLoop`/`elemLoops`, with SSE and NEON counterparts.
- LayerNorm, gaussian top-k and magnitude matching already share `emitRowStat` within each ISA.
- Float matvec already parameterises F32/F16/BF16 decoding.
- `dotEmitter` already centralises VNNI versus pre-VNNI dots.
- Device `matVec` already handles many format, tile, split and expert variants.
- `voltaDeq` is shared by multiple device matrix paths.
- Device RMSNorm variants already share `rmsNormRows`.
- Scalar/channel delta variants already share builders.

The estimate below excludes savings already realised by these abstractions.

| Pattern and remaining duplication | Concrete evidence | Estimated net lines removable |
|---|---|---:|
| **Host matrix traversal:** prologue → scale preparation → payload decode → dot chains → correction/fold → store | `EmitPackedMatVecFusedAhead`, `EmitPackedMatMulTiled`, `EmitPackedMatMulStationary`; NEON packed/tiled/stationary twins; row-major `emitGEMMQ4K`, `emitGEMMQ6K`, `emitGEMMQ3K` | **1,200–2,000** |
| **Device staged matrix traversal:** group coordinates → expert selection → dequant staging → barrier/double buffer → matrix operation → output | `GemmVolta` and `GemmTile`; repeated decode/address fragments around `MatVec`, `MatVecMMA` and existing `voltaDeq` | **350–650** |
| **Row reductions and maps:** fixed-width walk → reduction → scalar transform → apply walk | Host RMSNorm versus row-stat walkers on three ISAs; device `NormPartRows`, `LayerNormPartRows`, `LayerNormVarRows`; repeated flat-map setup | **450–800** |
| **Attention traversal and softmax fragments:** score/accumulate walks; partial-max/sum/output combination | Host single/paired/tiled score and accumulation emitters; device flash merge variants and online-softmax fragments | **450–850** |
| **Recurrent update-and-dot:** state-row traversal, update, reduction and output | Host `emitGatedDelta`, `EmitGatedSSD`, `EmitSelScan` and tier counterparts; remaining device gate/scan fragments | **200–400** |
| **Selection, gather and image traversal:** repeated index/tail/reduction scaffolding | Argmax/top-k/sample tier transcriptions; packed-row/widening and image walkers; device gather/index scaffolding | **300–600** |
| **Total** | Rounded planning range | **2,950–5,300** |

These ranges describe **net source reduction**, retaining format-specific arithmetic and ISA schedules. They are not clone-detector percentages. The upper end assumes useful sharing across ISA-specific adapters; it should be revised after the first family establishes the adapter cost.

Representative evidence:

**Packed matrices.** The fused and token-tiled emitters repeat format decomposition, masks, scale-plane handling, register assignment, argument loads and row-loop construction. Compare [packedfused.go:56](../../jit/cpu/packedfused.go) with [packedtiled.go:94](../../jit/cpu/packedtiled.go). Their different register-lifetime plans must remain separate.

The three row-major K-GEMMs repeat tile validation, scratch geometry, accumulator construction and loop scaffolding: [gemmk.go:46](../../jit/cpu/gemmk.go), [gemmk.go:286](../../jit/cpu/gemmk.go), [gemmk.go:507](../../jit/cpu/gemmk.go).

**Device staging.** `GemmVolta` and `GemmTile` independently spell closely related shape checks, dense/grouped block mapping, parameter setup and dequant staging. Both already call `newVoltaDeq`; that decoder is not additional removable duplication. See [gemm70.go:96](../../jit/gpu/kernels/gemm70.go) and [gemmtile.go:73](../../jit/gpu/kernels/gemmtile.go).

**Row reductions.** `EmitRMSNorm` and `emitRowStat` repeat the four-chain walk, vector remainder, scalar remainder and horizontal fold. Their arithmetic differs: RMS uses FMA; variance uses subtract, multiply and add. The useful common unit is the **walk and exact fold schedule**, not one interchangeable reduction expression. See [norm.go:16](../../jit/cpu/norm.go), [layernorm.go:29](../../jit/cpu/layernorm.go), [elem_a64.go:68](../../jit/cpu/elem_a64.go), and [sse_norm.go:173](../../jit/cpu/sse_norm.go).

**Attention.** SSE already has a useful factoring example, `attnScoreRowSSE`; the AVX2/NEON single/paired/tiled forms retain more independently written traversal. See [sse_attn.go:114](../../jit/cpu/sse_attn.go), [attn.go:20](../../jit/cpu/attn.go), and [attn.go:187](../../jit/cpu/attn.go).

`FlashAttentionMerge`, `FlashAttentionMergeWide` and `FlashAttentionMergeRun` share the mathematical combination of partial maxima, denominators and numerators. However, their operation ordering differs. Extract smaller ordered fragments first; merging them into one reordered builder would fail the requested gate. See [flash.go:378](../../jit/gpu/kernels/flash.go), [flash.go:435](../../jit/gpu/kernels/flash.go), and [mergerun.go:22](../../jit/gpu/kernels/mergerun.go).

## 3. Template parameters and special cases

A practical template is an **emission-time control structure with explicit instruction-emitting callbacks and schedule data**. It does not introduce runtime format switches or replace specialised kernels with generic ones.

| Pattern | Parameters that must remain explicit |
|---|---|
| Matrix | Payload layout/planes, signedness, codebook, scale/min decoding, activation window, dot instruction, accumulator type/chains, loop nesting, row/token/K tile, prefetch, scratch layout, correction order, epilogue, physical-register plan |
| Row reduction | Baked/runtime extent, vector width, unroll, accumulator assignment, exact fold tree, pass count, transform, scalar-tail neutralisation, weight/bias access, scalar versus vector instructions |
| Elementwise map | Argument bindings, constants, body, cursor order, vector/scalar load-store forms, aliasing contract, ISA prologue/epilogue |
| Attention | Q/K/V geometry, GQA mapping, contiguous/paged address generation, K transposition, KV precision, query/key tile, split, causal/window/chunk masks, sinks, softcap, max/sum update order, merge layout |
| Recurrence | State layout, scalar/channel decay, update equation, input/output state ownership, row/run ordering, reduction chains, scan lane count |
| Gather/selection/image | Index mapping, tie rules, clamp/padding policy, strides, sample/filter precision, scalar tails, output ownership |

### Special cases that must survive

1. **Register budgets are algorithmic choices.** Q5-style secondary planes reduce the fused row group from 16 to 8 because the wider allocation needs 17 vector registers. Preserve that choice and its allocation order ([fusedgroup.go:14](../../jit/cpu/fusedgroup.go)).

2. **VNNI and pre-VNNI signed dots are different sequences.** Q8 centring cannot simply use the unsigned saturating multiply-add path. `signed` and `signedFix` deliberately preserve the existing final arithmetic ([prevnni.go:87](../../jit/cpu/prevnni.go)).

3. **Integer and float folds are separate schedules.** Stationary GEMM’s integer accumulation depends on format/window/dot kind and is explicitly not universally bit-identical to the fused float path ([packedgemm.go:184](../../jit/cpu/packedgemm.go)). Preserve each variant against its own golden.

4. **Reduction tails are not ordinary map tails.** LayerNorm’s vector variance tail must clear inactive lanes after subtracting the mean; SSE uses scalar instructions instead. SSE’s eight-XMM fold mirrors four YMM chains in a specific order ([layernorm.go:37](../../jit/cpu/layernorm.go), [sse_norm.go:130](../../jit/cpu/sse_norm.go)).

5. **Host attention is not uniformly online-softmax attention.** It has separate score and accumulation kernels consumed through `nn.AttnSet`. Device flash kernels implement the online-softmax pattern; a template must not silently convert host attention to it ([engine/nn/attn.go:1](../../engine/nn/attn.go)).

6. **Fragment MMA and collective matrices have different ownership.** `OpMMA` exposes lane fragments; collective tile operations leave lane layout to the driver. Preserve separate inner policies ([ir.go:132](../../jit/ir/ir.go)).

7. **Paging and surplus-thread safety are part of the pattern.** Clamping loads, masking scores, descriptor interpretation and out-of-place writes must remain in their current positions. Multiplication by zero cannot replace masking potentially poisoned values.

### Host allocation versus device IR

On the host, the emitter owns **physical registers**. For example, the packed fused emitter allocates registers in sequence, and the tiled emitter borrows dead activation/payload registers for its epilogue. A template must accept those exact assignments and lifetimes, not recalculate a “better” allocation ([packedfused.go:99](../../jit/cpu/packedfused.go), [packedtiled.go:176](../../jit/cpu/packedtiled.go)).

On devices, the builder assigns SSA IDs by append order. Moving a constant creation earlier changes IDs and downstream text even when the computation is equivalent ([ir.go:253](../../jit/ir/ir.go)). Callbacks therefore need to preserve **when** each IR operation is emitted. Precomputing callback arguments can itself break equality.

An exact-output template has no reason to slow the generated instruction stream. Its possible costs are **emission time, Go allocation, binary size, and preparation/cache behaviour**. A template that homogenises schedules, adds runtime branches, changes memory operands or expands live ranges is a different change and should fail the refactor gate.

## 4. Optimisation opportunities after exact-output factoring

These are ranked hypotheses, not performance claims. Every item below changes generated output or selection and therefore needs a separate baseline and measured follow-up.

| Rank | Opportunity | Current-code basis and expected value |
|---:|---|---|
| **1** | **Compose producer/epilogue fragments to remove intermediate traffic and selected dispatches** | Existing `RMSNormQuantRows`, `BiasAct`, `MatVecSegments` and fused recurrent paths show concrete compositions already worth expressing. Templates could make additional combinations cheaper to implement and gate. Start where an intermediate load/store or dispatch demonstrably disappears. |
| **2** | **Tune matrix patterns through one candidate mechanism** | Tile, split, prefetch and activation-window choices currently span host tuners and device plans. A pattern can expose legal candidates plus register/shared-memory bounds, while retaining per-device measurements and caller overrides. |
| **3** | **Propagate format fixes and decode improvements across matrix variants** | Shared decode/scale fragments reduce the chance that a format-plane change reaches matvec but misses MMA or packed GEMM. This is immediate maintenance value and a route for later instruction improvements to reach more paths. |
| **4** | **Reuse attention addressing and softmax fragments across variants** | Improvements to page-address calculation, masked loads, partial merging or vector access can reach several attention forms without rewriting each. Keep separate schedules for decode, prefill, MLA and device matrix units. |
| **5** | **Explore per-ISA reduction/scheduling candidates systematically** | Common walkers make accumulator count, unroll and prefetch experiments easier to express. The benefit remains host/workload-specific; they are not portable defaults. |

Code anchors:

- Existing composition: [elem.go:188](../../jit/gpu/kernels/elem.go), [elem.go:1034](../../jit/gpu/kernels/elem.go), [matvec.go:1222](../../jit/gpu/kernels/matvec.go).
- Existing host tuning: [tune.go:379](../../engine/nn/tune.go), [prefetch.go:1](../../engine/nn/prefetch.go); device choices: [decodeplan.go:1](../../jit/gpu/kernels/decodeplan.go).
- Existing shared decoder: [mma70.go:422](../../jit/gpu/kernels/mma70.go); shared scale-stream emission: [scstream.go:1](../../jit/gpu/kernels/scstream.go).

Two historical results limit expectations: increasing decode row tiling regressed on both measured GPUs, and the Metal investigation distinguished dispatch overhead from the smaller portion fusion could recover. Templates make these experiments easier; they do not reverse the evidence ([gpu-kernels.md:4110](../../docs/engineering-history/gpu-kernels.md), [gpu-kernels.md:4921](../../docs/engineering-history/gpu-kernels.md)).

The useful connection to the composer goal is a catalogue of **legal specialised fragments, schedules, resource requirements and measured costs**, as requested in [AGENTS.md:812](../../AGENTS.md).

## 5. Golden-output gate and risks

### Existing gates are a starting point, not sufficient coverage

- `TestPrimaryTableIsTodaysEmitters` compares a table entry with the **current direct function**. Both can change together and still pass; it is a wiring gate, not a historical-output golden ([emitters_test.go:218](../../jit/cpu/emitters_test.go)).
- `TestDenseKernelsUnchanged` captures combined PTX/MSL/SPIR-V hashes for `all(t)`, using an eight-byte digest prefix. It does not cover every specialised matrix inventory or record each output independently ([golden_test.go:19](../../jit/gpu/lowertest/golden_test.go)).
- `TestContiguousPrefillIRUnchanged` records a valuable shape sweep, but only that family. Its IR hash does not include `Kernel.Comment` ([prefillgolden_test.go:20](../../jit/gpu/kernels/prefillgolden_test.go)).

### Capture before changing any emitter

1. **Freeze the baseline revision and test manifest.** Record commit, build tags, target architecture, explicit tier/dot/features, emitter entry point and every codegen argument. Capture twice in independent processes; outputs must agree before the corpus is trusted.

2. **Capture host bytes as returned by the emitter.** Include ISA declarations, constants embedded in code, branches and padding. Compare before executable mapping, without stripping “irrelevant” bytes.

3. **Capture all device layers separately:**
   - Deterministically serialised complete IR: name, comment, group, lanes, parameters and every field of every op.
   - Exact PTX text for each relevant target.
   - Exact MSL text.
   - Exact SPIR-V binary. The current API returns binary, so use that as the authoritative comparison; retain disassembly from a pinned tool for readable diagnostics.

   Do not normalise identifiers, whitespace, operation order or constant spelling to make differences disappear.

4. **Store raw artifacts plus full SHA-256 and byte lengths.** Hashes make inventories manageable; raw artifacts allow direct equality and useful diffs. A content-addressed corpus can deduplicate identical outputs without weakening comparison.

5. **Preserve the capability matrix.** Unsupported shape/backend combinations get explicit expected error outcomes. PTX-only MMA must not become an unexplained MSL/SPIR-V skip; collective-tile refusals remain explicit.

These artifacts are **test evidence**, never production kernel inputs. Production still emits at runtime under RULE 8.

### Sweep shapes from actual gates

Build the manifest from the **union of existing emitter, backend and integration-gate configurations**, then add boundary cases. A small new representative list would lose coverage.

| Family | Required sweep dimensions |
|---|---|
| Host matrices | `quant.PackedTypes`, supported floats, row-major/packed forms, VNNI/VEX/SSE/NEON, pack widths, accumulator choices, windows, prefetch modes, token/row/K tiles, main/tail forms |
| Norm/maps | Every activation and row mode; bias on/off; all existing width lists; vector/unroll boundaries and scalar-only widths |
| Attention | Existing head widths including ragged widths; unequal K/V widths; f16/f32; paired/tiled forms; GQA; contiguous/paged; transposed/row-major K; split/window/chunk/sink/softcap combinations |
| Device matrix units | Existing MMA/tile inventories, supported format/shape combinations, grouped/indexed forms, target-specific lowering and expected refusals |
| Recurrence/layout/image | Scalar/channel decay, state widths, run/slot variants, image formats/filter precision, strides and tails |

Host runtime lengths do not necessarily produce different kernels: capture each distinct emitter configuration, and exercise runtime lengths separately in execution tests.

Add a coverage check requiring every emitter-table field and registered device family to have manifest entries. Report case counts and fail if a case disappears. Examples to build from are [emitterCalls](../../jit/cpu/emitters_test.go), [lowertest.all](../../jit/gpu/lowertest/lower_test.go), and [contiguousPrefill](../../jit/gpu/kernels/prefillgolden_test.go).

### Diff and acceptance

A failure should identify:

`family / tier-or-backend / format / complete shape / schedule / first differing offset`

Provide:

- Host: byte diff and disassembly around the first change.
- IR: first changed op or metadata field, including SSA IDs.
- PTX/MSL: unified text diff.
- SPIR-V: first differing word and pinned-tool disassembly diff.
- Manifest: missing, added or changed success/refusal cases.

For the refactor, **any output difference is a failure**, even when oracle accuracy and timing pass. No golden updates should accompany emitter factoring.

Deliberately mutate one host instruction byte, one IR immediate, parameter order and workgroup metadata to prove the gate discriminates. Also test removal of a manifest case.

### Remaining risks and project rules

- **Golden equality preserves existing behaviour, not independent correctness.** Keep oracle, guard-tail, encoding and model gates. Do not replace NMSE gates with inappropriate cross-ISA bit equality: compare old versus new within the same configuration.
- **Per-ISA scheduling must remain visible.** Register reuse, FMA versus separate operations, memory operands, unroll and prefetch are preserved policy, not incidental implementation.
- **RULE 6:** emitted work, code size and traffic can be unchanged by construction; cold emission time/allocations are still unresolved until measured. No sixth `cpu.Spec` field is needed for internal template parameters ([AGENTS.md:254](../../AGENTS.md)).
- **RULE 8:** emitter callbacks run in Go while constructing code; token arithmetic remains generated. Preserve basic generated paths and tail coverage ([AGENTS.md:535](../../AGENTS.md)).
- **RULE 11c:** identical kernel bodies do not prove unchanged selection, arguments or launch geometry. Preserve call sites and run affected end-to-end device gates with placement assertions and partial seams. Pin tuning for equality tests, while separately exercising shipping defaults ([AGENTS.md:717](../../AGENTS.md)).
- **RULE 11b:** a mismatch is investigated, not blessed with `-update` ([AGENTS.md:656](../../AGENTS.md)).

## 6. Recommended first conversion

**Start with the host fixed-width row-walk family: RMSNorm plus LayerNorm/GaussTopK/MagMatch. Convert AVX2 first, then SSE and NEON in separate steps.**

This family has visible duplication, small bounded interfaces, extensive ragged-width coverage, and no need to touch packing, placement or device launch policy. It also tests the important distinction between sharing traversal and preserving arithmetic.

### Steps

1. **Land the baseline capture and comparison harness first.** Capture all affected host forms on each tier, plus the broader unchanged inventory. Freeze the golden artifacts independently of the refactor.

2. **Extract an AVX2 fixed-width walker.** Parameters should cover the baked width, cursor increment order, four-vector body, vector remainder and scalar remainder. Keep register assignments and arithmetic in the existing callers.

3. **Extract the exact four-chain fold.** Preserve the seven-instruction AVX2 fold sequence. Keep RMSNorm’s FMA and row-stat’s separate operations intact.

4. **Reuse those pieces in `EmitRMSNorm` and `emitRowStat`.** Keep prologues, scalar normalisation, mode branches and epilogues in their existing order.

5. **Repeat within SSE and NEON.** SSE keeps `normFold8SSE`; NEON keeps its four-lane geometry and current load/multiply order. Share cross-tier geometry only where it removes code after accounting for adapter overhead.

6. **Stop before changing dispatch or adding fused kernels.** Those become follow-ups with new, explicitly approved output changes and measurements.

### Exact-output and execution gates

For the first family, the golden width manifest should include the union of existing norm/row-stat lists, all widths **1–65**, and boundaries around larger unroll widths, plus the existing large widths through 8,192. Include LayerNorm bias both ways and all three row-stat modes. Existing SSE coverage already supplies widths 1–40 and many larger boundaries ([ssenorm_test.go:92](../../jit/cpu/ssenorm_test.go)).

Required checks:

- **Old/new byte equality for every case**, separately for AVX2, SSE and NEON.
- `TestRMSNormKernelMatchesReference`, `TestEmitLayerNormMatchesReference`, `TestRowStatsMatchReference`.
- SSE oracle/bit-comparison gates, `TestRowStatsSSEMatchReference`, VEX-leak checks and forced-SSE integration.
- Native arm64 execution, not merely cross-compilation.
- `TestEveryModelRunsGenerated`, `TestBatchMatchesForward`, `TestGreedyMatchesLlamaCpp`, and `TestDecodeDoesNotAllocate` on the applicable hosts.
- Gate mutation checks described above.

The existing correctness tests explicitly cover heavy-tailed inputs and output guards ([norm_test.go:27](../../jit/cpu/norm_test.go), [layernorm_test.go:14](../../jit/cpu/layernorm_test.go), [rowstat_test.go:81](../../jit/cpu/rowstat_test.go)).

### What proves “no slower”

**Identical machine code, constants, dispatch and memory layout establish unchanged generated execution work.** They do not prove unchanged startup costs or eliminate process-layout noise.

Measure cold emission time and allocations separately. For runtime verification, use the project’s A/A control first, then paired/interleaved old/new comparisons at shipping thread counts, at least 20 rounds, twice, with the stated dispersion and machine gates. Put timing checks in benchmark tooling, not correctness tests. Report unresolved effects as unresolved rather than “neutral.”

All subsequent heavy validation must use the cap, for example:

```bash
./scripts/cap 8G -- taskset -c 0,2,4,6,8,10 go test ./... -count=1
```

That command is a future validation requirement; it was not run during this read-only study.

### Rough change size

| Stage | Estimated source churn | Estimated net production reduction |
|---|---:|---:|
| AVX2 row walkers/fold | 160–260 lines added/deleted | 50–100 lines |
| Complete family across AVX2/SSE/NEON | 600–1,000 lines added/deleted | 180–300 lines |
| Golden harness and manifest plumbing | 300–500 additional test lines | Excluded from production savings |
| Raw golden corpus | Depends on final sweep; measure during capture | Excluded |

This first conversion should establish that shared emitter structure can preserve exact outputs. The larger packed-matrix family then offers the next substantial source reduction, using the same capture, coverage and diff machinery.
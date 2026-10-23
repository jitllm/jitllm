# Runtime kernel and computation inventory

Source snapshot: the commit that last changed this file. The CSV is generated, not hand-maintained:

    scripts/kernel-inventory.py > docs/kernel-inventory.csv
    scripts/kernel-inventory.py --summary      # the counts quoted below, and each GPU constructor's callers

`--root DIR` reads another checkout (a `git archive <commit>` export, as this snapshot was taken) and `--skip FILE` leaves a file out.

For an operation-by-operation view of generated execution and alternatives, see [kernel capabilities](kernel-capabilities.md). The [external comparison](kernel-comparison.md) examines ik_llama.cpp and kimi-k3-in-c against the newer working tree.

jitllm builds computation at runtime at three levels:

1. **CPU kernels:** emit native machine-code bytes for the selected instruction tier and operation shape, map them as executable memory, then call them.
2. **GPU kernels:** construct a typed kernel IR from the operation's dimensions and options, lower it to PTX, SPIR-V, or MSL, then compile/load it through the device driver.
3. **Layer execution:** construct layer plans, select and generate their constituent operations, allocate and bind working state, and assemble their execution. Where an optimized variant is unavailable, supported alternatives are generated or composed from generated operations.

These are generators, not a fixed collection of precompiled kernels. The number of generated variants depends on model shapes, formats, hardware, and tuning choices. A layer's execution is assembled from kernels; it need not be one monolithic fused kernel.

## Inventory boundaries and counts

The complete exported generator index is in [kernel-inventory.csv](kernel-inventory.csv). Its columns, each read from the source by the script:

| Column | Meaning | How it is derived |
|---|---|---|
| `kind` | `cpu`: a machine-code generator. `gpu`: an IR-kernel constructor. | `cpu` is a top-level exported `func Emit…` in `jit/cpu` whose result is `[]byte` or `([]byte, error)`; `gpu` is a top-level exported func in `jit/gpu/kernels` whose result carries `*ir.Kernel`. Test files and methods are excluded. |
| `symbol` | The function name. | As declared. A name implemented once per architecture appears once per declaration. |
| `file`, `line` | Where the declaration is. | Path from the repository root and the line of its `func`. |
| `build_constraint` | Which builds compile it. | The file's `//go:build` expression, else its GOOS/GOARCH filename suffix, else `all`. |
| `signature` | The declaration. | The text up to the body's brace, joined onto one line. |

Counts describe source entry points, not distinct mathematical operations or compiled binaries:

- The CPU dispatch table has **41 code-generator slots**, plus three capability/width queries (`RowMajorSupported`, `PackedSupported`, `MaxTiledTokens`) and the `RowMajorGGUF` flag.
- `jit/cpu` contains **164 exported byte-generator declarations**, representing **102 unique names** across architecture implementations, wrappers, and refusal stubs. This includes generators outside the dispatch table.
- `jit/gpu/kernels` contains **104 exported IR-kernel constructors**, including convenience wrappers and probes.

The index excludes individual assembler instruction methods and private helpers already reached through these generators. Those compiler building blocks are covered separately below. Host-side packing, file I/O, scheduling, and reference arithmetic are distinguished from generated execution kernels.

## CPU kernel families

The dispatch boundary is [`cpu.Emitters`](../jit/cpu/emitters.go). `primaryEmitters` selects the amd64 AVX2 or arm64 implementation; `sseEmitters` supplies the legacy x86 tier. Each generator below returns machine-code bytes, subsequently mapped by `cpu.Map` / `MapNamed`.

| Family | Generator slots / direct builders | Computation and variants |
|---|---|---|
| Matrix-vector multiplication | `RowMajor`; `Emit`, `EmitNative`, `EmitA64` | Float and supported raw quantized layouts; single-row and interleaved forms. F32/F16/BF16 use float activations. |
| Packed matrix-vector multiplication | `PackedMatVec`, `PackedWide`, `PackedFused`, `PackedFusedAhead`, `PackedFusedWin` | Container-layout weight planes; row tiles, tails, wider sweeps, and fused accumulation, with a software-prefetch distance (`Ahead`) or an activation window with the integer super-block fold (`Win`, arm64 only). `nn.MatVecPacked` selects among them. |
| Matrix-matrix / prompt prefill | `PackedTiled`, `PackedGEMM`; `EmitGEMMWindow` | Token-tiled packed matmul, the weight-stationary packed GEMM (one kernel per type, K and row count serving any token count), and raw-layout GEMM. Shapes include type, K, row count, token width, register tile, and activation window. |
| Specialized raw-layout variants | `EmitInterleaved` | Baked block counts and interleave widths; a generator/test surface, not evidence that container inference uses raw GGUF weights. |
| Activation quantization and packing | `QuantAct`, `QuantActNarrow`; `EmitPackAct` | Wide amax windows, one-block windows, int8 activation payloads, scales, bias corrections, optional half-block sums, and GEMM token packing. |
| Embedding rows and widening | `PackedRow`, `Widen` | Decode a selected packed embedding row; widen F16/BF16 to F32. F32 row copying needs no arithmetic kernel. |
| Normalization | `RMSNorm`, `LayerNorm` | Shape-specialized normalization, with optional LayerNorm bias. Head normalization also uses these CPU operations. |
| Rotary position encoding | `RoPE`, `RopeTable` | Apply adjacent-pair/NeoX rotations; generate position-dependent cosine/sine tables from model planes, including scaling configurations. |
| Attention scores | `AttnScores`, `AttnScores2`, `AttnScoresTiled` | One query, paired queries, and tiled queries; head/KV strides and F32/F16 KV layouts. |
| Attention accumulation | `AttnAcc`, `AttnAccInto`, `AttnAcc2`, `AttnAcc2Into` | Weighted value accumulation, paired heads, and accumulation into an existing result for page/window traversal. |
| Softmax | `Softmax` | Exponentiation, reduction, normalization, vector body and generated tail handling. |
| Activations and gates | `Act`, `ActMul`, `SigmoidMul` | Ungated SiLU/GELU/QuickGELU; gated SiLU/GELU/SwiGLU-OAI; sigmoid multiplication. Supported gated/ungated sets are explicit. |
| Residual / scaling / clipping | `Axpy`, `Scale`, `Softcap`, `Clamp` | Scaled addition, multiplication, soft-capping, and clamping to `[lo, hi]` (DBRX's q/k/v clip). `nn.Add32JIT` uses AXPY with alpha one. |
| MoE routing | `MoETopK`, `MoERoute` | Top-k selection, tie handling, selection and ascending-ID orders, normalization, selection bias, expert groups, and routed scaling. `MoETopK` specializes the generalized route generator. |
| Sampling | `Argmax`, `SampleSegMax`, `SampleDraw`, `SamplePenalty` | Greedy selection; segmented candidate maxima; min-p/top-p filtering and inverse-CDF selection; repetition penalty. |
| Recurrent / linear attention | `GatedDelta`, `GatedDeltaChan`, `Conv1d`, `DeltaGate` | Scalar or per-channel decay, recurrent matrix update, causal depthwise convolution, and gate transformations. |

CPU source families are split among `matvec*`, `packed*`, `packedgemm*`, `gemm*`, `packact*`, `quantact*`, `rowpacked*`, `norm*`, `layernorm*`, `rope*`, `ropetab*`, `attn*`, `act*`, `sample*`, `topk*`, `delta*`, `conv*`, and `gate*`. `sse_*` and `*_a64.go` provide their architecture implementations; the CSV lists each public generator definition.

### CPU hardware tiers

| Tier | Selection and generated code | Material distinction |
|---|---|---|
| amd64 AVX2 | AVX2, FMA, and F16C baseline; VNNI selected when usable | Packed dot-product emitters can generate VNNI or a pre-VNNI AVX2 instruction sequence. |
| amd64 SSE | SSE2, SSE3, SSSE3, SSE4.1 | Independent 128-bit generators, including normalization, attention, packed matvec, routing, sampling, recurrence, and quantization. Wide, tiled, prefetching, windowed and stationary-GEMM packed variants are deliberately declined ([`sse_refuse.go`](../jit/cpu/sse_refuse.go)), with generated alternatives used by callers. |
| arm64 NEON | ARMv8-A NEON arithmetic; per-kernel feature checks | Quantized dot kernels use SDOT where the chip has FEAT_DotProd and widen each SDOT to SMULL/SMULL2/ADDP/SADALP where it does not, the decode kernels chaining two dots into one int16 pair with SMLAL/SMLAL2 (`sdotemu.go`), so an ARMv8.0 chip (Cortex-A53/A57/A72/A73) runs every kernel with the same bits. The packed-wide entry point declines on arm64; its fused, windowed-fused, prefetching row-tile, tiled and stationary-GEMM forms come from `packed_a64.go` and `packedgemm_a64.go`. The amd64 tier has no windowed-fused form. |

Evidence: [`emitters.go`](../jit/cpu/emitters.go), [`tier.go`](../jit/cpu/tier.go), [`features_amd64.go`](../jit/cpu/features_amd64.go), [`prevnni.go`](../jit/cpu/prevnni.go), [`sse_packed.go`](../jit/cpu/sse_packed.go), [`native_arm64.go`](../jit/cpu/native_arm64.go), [`packed_other.go`](../jit/cpu/packed_other.go), [`packed_a64.go`](../jit/cpu/packed_a64.go).

The packed format inventory is Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, and MXFP4. Floating-point readers/matvecs cover F32/F16/BF16. Admission still depends on the generator's shape and ISA requirements.

## GPU kernel families

All names here are runtime IR constructors. Their complete signatures and source locations are in the CSV; aliases are included rather than counted as unrelated algorithms.

| Family | Constructors | Main specialization |
|---|---|---|
| Matrix-vector / batched projection | `MatVec`, `MatVecSegments`, `MatVecMMA`, `MatVecMMA70`, `GemmVolta`, `GemmTile` | Quant/float format, K, output rows, split reduction, row/token tiles, bias, activation window, expert bank/slot selection, subgroup strategy, and MMA tiles. `MatVecSegments` runs several decode matvecs over one activation (q, k, v) in one launch. `MatVecMMA70` and `GemmVolta` target sm_70's f16 tensor cores (`ir.MMAVolta`); `GemmTile` targets the collective tile ops Metal lowers to `simdgroup_matrix`. |
| Tensor-core activation staging | `ActF16`, `ActF16T`, `ActF16TGather` | Float activations converted to binary16 in the step-major order `MatVecMMA70` reads or the token-major order `GemmVolta`/`GemmTile` read, optionally gathering rows for a grouped mixture. |
| Split-result reduction | `Reduce` | Output row count and split count. |
| Activation quantization | `Quantize` | Input size and quantization window. |
| RMS normalization | `NormPart`, `NormPartRows`, `NormApply`, `NormApplyRows`, `RMSNormRows`, `RMSNormRowsWarp`, `RMSNormQuantRows`, `RMSNormQuantRowsWarp` | Two-kernel partitioned form, or the whole norm in one launch per row with an optional residual add; shared-memory or subgroup-shuffle reduction; the `Quant` forms also write the int8 activation `Quantize` would. |
| Layer normalization | `LayerNormPartRows`, `LayerNormVarRows`, `LayerNormApplyRows` | Mean/variance/application, row batches, epsilon, bias; used by vision blocks. |
| Activations / residuals | `Act`, `ActMul`, `ActMulWeighted`, `SigmoidMul`, `Softcap`, `Clamp`, `Add`, `Scale`, `ScaleRowsByCount`, `SplitHeadGate` | Activation kind, lengths, fused multiplication (with a routed weight on the expert input, Llama 4), softcap and clamp values, per-row scaling from a causal-count table (Llama 4's attention temperature), head/gate layout. |
| RoPE | `RoPERows`, `RoPERowsT`, `RopeTable` | Head geometry, rotated width, NeoX layout, batched rows, transposed destination, and table generation. |
| KV writes / copies | `CopyAt`, `CopyAtRows`, `CopyAtRowsT`, `CopyAtF16`, `Restride`, `CopySpan` | Position-indexed copies, batched/transposed layout, half-precision KV storage, a KV cache changing capacity in place (`Restride`), and one row copied from a runtime offset (`CopySpan`). |
| Embedding lookup | `GetRowsPacked` | Gathers token rows of a packed weight into float32, for a tied embedding held on the device. |
| Attention scores | `AttnScoresWarp`, `AttnScoresTiled`, `AttnScoresTiledW`, `AttnScoresRagged`, `AttnScoresMMA`, `AttnScoresMMAW`, `AttnScoresMMA70` | Head/KV geometry, GQA, context stride, query/key tiling, sliding window, one warp per (head, position), ragged batches of different sequences, and tensor-core computation (sm_70 in `MMA70`). |
| Attention softmax | `SoftmaxRows`, `SoftmaxRowsSink` | Scalar or subgroup reduction, batched queries, score strides, and attention sinks. |
| Attention accumulation | `AttnAcc`, `AttnAccTiled`, `AttnAccRagged`, `AttnAccSplit`, `AttnAccSplitF16`, `AttnAccMMA70` | Value geometry, query tiling, ragged batches, split accumulation, F32/F16 value cache, and sm_70 tensor cores. |
| Fused (flash) attention | `FlashAttention`, `FlashAttentionMerge`, `FlashDecodeKV`, `FlashPrefill70`, `FlashPrefillTile` | Scores, online softmax and value sum in one kernel with no score plane written; key partitions merged with sinks added once; a workgroup per KV head for decode; prompt chunks on sm_70 or on collective tile ops. |
| Head normalization | `HeadNorm`, `HeadNormRows` | Head width/count, epsilon, batched rows, scalar/subgroup implementations. |
| Greedy selection | `Argmax`, `ArgmaxRows` | Index of the largest logit, lowest index winning a tie as on the host; one row or one per sequence. |
| MoE routing / combination | `ExpertGroupScore`, `ExpertGroupMask`, `ExpertRank`, `ExpertWeights`, `ExpertRoute`, `ExpertRouteWarp`, `ExpertCombine` | Grouped selection, selection bias, sigmoid/softmax routing, top-k, normalization and scaling, weighted expert reduction. `ExpertRoute` is rank and weights in one launch; `ExpertRouteWarp` is its one-subgroup form for hundreds of experts. |
| MoE grouped dispatch | `ExpertGroupBlockCount`, `ExpertGroupCount`, `ExpertGroupEnds`, `ExpertGroupColumns`, `ExpertGroupScatter`, `ExpertGroupTiles`, `GatherRows`, `ExpertCombineGrouped` | Sorting many tokens' (token, expert) pairs into padded per-expert runs: per-block counts, offsets, run ends, column defaults, a stable scatter, per-tile expert ids, the row gather, and the combine back in selection order. |
| MoE projections / biases | `RouterMatVecOf`, `RouterMatVecTok`, `SharedRouterLogits`, `SharedExpertAdd`, `IndexedBiasAdd` | Float router projection (F32 or F16 weights, one or many tokens), shared-expert gating/addition, indexed expert biases. Expert matrix work itself is a `MatVec` specialization. |
| Recurrent / linear attention | `Conv1dRows`, `Conv1dShift`, `Conv1dRowsRuns`, `Conv1dShiftRuns`, `CopySlots`, `GatedDeltaStep`, `GatedDeltaStepTiled`, `GatedDeltaStepChan`, `GatedDeltaScan`, `GatedDeltaFused`, `DeltaGateRows`, `SplitDeltaGatesRows`, `SliceRows` | Convolution state (consecutive tokens, or a ragged step's runs -- each sequence's rows chained through its one state, a decoding sequence a run of one), scalar/per-channel recurrent decay, tiled key-head pairing (Qwen3.5), a multi-token scan with the state in registers, the scan with its SiLU/slices/norms/gates fused in, gate packing over one or many tokens, and slice geometry. |
| Raw-weight preparation | `Unpack` | Q4_K device unpacking into the packed layout. This is separate from consuming already prepared `.jlm` weights. |
| Capability / bandwidth probes | `MMAProbe`, `MMAVoltaProbe`, `TileProbe`, `StreamWide`, `StreamWideSpread` | Matrix capability tests (`MMAVoltaProbe` pins the sm_70 lane layout) and memory-access probes. They are generated kernels, but not model-forward operations. |

Four constructors have no non-test caller outside `jit/gpu/kernels`: the single-row `HeadNorm` and the probes `MMAProbe`, `MMAVoltaProbe`, `TileProbe`. `StreamWide`/`StreamWideSpread` are reached from `dev/tools/mvbench` and `SoftmaxRows` only from `dev/tools/ptxdump`; every other constructor is called from `jit/gpu/tier`. (`--summary` prints this per constructor.)

GPU source files: [`matvec.go`](../jit/gpu/kernels/matvec.go), [`mma.go`](../jit/gpu/kernels/mma.go), [`mma70.go`](../jit/gpu/kernels/mma70.go), [`gemm70.go`](../jit/gpu/kernels/gemm70.go), [`gemmtile.go`](../jit/gpu/kernels/gemmtile.go), [`elem.go`](../jit/gpu/kernels/elem.go), [`getrows.go`](../jit/gpu/kernels/getrows.go), [`attn.go`](../jit/gpu/kernels/attn.go), [`attn70.go`](../jit/gpu/kernels/attn70.go), [`flash.go`](../jit/gpu/kernels/flash.go), [`flashkv.go`](../jit/gpu/kernels/flashkv.go), [`flash70.go`](../jit/gpu/kernels/flash70.go), [`flashtile.go`](../jit/gpu/kernels/flashtile.go), [`argmax.go`](../jit/gpu/kernels/argmax.go), [`ropetab.go`](../jit/gpu/kernels/ropetab.go), [`moe.go`](../jit/gpu/kernels/moe.go), [`moesort.go`](../jit/gpu/kernels/moesort.go), [`moegroup.go`](../jit/gpu/kernels/moegroup.go), [`recurrent.go`](../jit/gpu/kernels/recurrent.go), [`unpack.go`](../jit/gpu/kernels/unpack.go), [`tile.go`](../jit/gpu/kernels/tile.go).

## When and how code is built

| Stage | Runtime construction | Reuse |
|---|---|---|
| CPU JIT/session initialization | `nn.NewJITSharing` emits raw/packed matvec, tail/wide/fused, and activation-quantize variants for the model's weight types. | Per-JIT maps keyed by format and shape. |
| CPU operation first use | Normalization, elementwise, row decoding, rotary tables, routing, and sampling emit their kernels through the selected table. Attention/recurrent builders receive the model dimensions. | Operation caches include tier and relevant dimensions/options. Some are process caches, others belong to the JIT. |
| CPU tuning / new shape | Interleave widths, GEMM tiles, and packed token tiles are generated as candidates or on demand. | Cached generated code; failed candidates can also be remembered. |
| CPU execution | `Map` validates requirements; platform mapping installs executable bytes; the assembly trampoline invokes them. | No external C compiler or assembler is needed by the runtime. |
| GPU kernel miss | `kernelMode`, `reduceKernel`, batch builders, and scratch construction call an IR constructor, then `Device.Compile`. | Per-device compiled-kernel maps and prepared layer/scratch objects. |
| CUDA | `ptx.Lower` emits PTX for the selected target; `cuda.Device.Load` calls `cuModuleLoadData`. | Driver-loaded module/function, with compatible launch sequences captured and replayed. |
| Vulkan | `spirv.Emit` writes the SPIR-V module directly; Vulkan creates a shader module and compute pipeline. | Prepared compute pipelines. No runtime dependency on `glslang` or `spirv-val`. |
| Metal | `msl.Emit` emits MSL; `newLibraryWithSource:options:error:` compiles it and the framework creates the compute pipeline. | Prepared Metal kernels/pipelines. |

Key sources: [`engine/nn/jit.go:293`](../engine/nn/jit.go#L293), [`engine/nn/norm.go`](../engine/nn/norm.go), [`engine/nn/matmul.go`](../engine/nn/matmul.go), [`engine/nn/packedmm.go`](../engine/nn/packedmm.go), [`cpu/baseline.go`](../jit/cpu/baseline.go), [`cpu/exec_linux.go`](../jit/cpu/exec_linux.go), [`cpu/exec_darwin.go`](../jit/cpu/exec_darwin.go), [`tier/tier.go:2051`](../jit/gpu/tier/tier.go#L2051), [`backend/cuda.go:628`](../backend/cuda.go#L628), [`backend/vulkan.go:103`](../backend/vulkan.go#L103), [`backend/metal.go:90`](../backend/metal.go#L90).

## Runtime layer construction

The computation layer is assembled from the loaded model's structure, weights, and hardware choices:

1. `model.State.planFor` and `layerWeightsAt` describe dimensions, attention/rotary layout, activation and normalization choices, biases, expert routing, recurrence, and MLA. `offerRange` supplies these to the device tier.
2. `PrepLayer` prepares projections and weight bindings. `initScratch` builds a list of operation-constructor jobs from the plan, generates their IR, compiles them, and binds working buffers. It is not selecting one precompiled binary for a named architecture.
3. The same machinery assembles dense attention/FFN blocks, MoE and shared-expert paths, sliding-window attention, vision blocks, hybrid linear attention, MLA, and the output head as applicable. `emitLinear` and `emitMLA` compose their specialized sequences.
4. `prepBatch` and `batchMV` build prompt-chunk variants. Context-capacity changes can rebuild scratch and the shape-dependent operations.
5. `Layers` / `layersOnce` issue the prepared operations for a run of resident blocks. Compatible CUDA sequences are captured into a runtime graph and replayed. Paging, placement, buffer moves, session switches, or relevant shape changes invalidate affected recordings.
6. On the CPU, `model.Forward`, prefill, batch, MoE, MLA, recurrence, and vision code orchestrate generated operation calls. The Go orchestration is part of the compiled program; the numerical kernels it invokes are emitted at runtime.

Sources: [`engine/nn/device.go`](../engine/nn/device.go), [`engine/model/forward.go`](../engine/model/forward.go), [`engine/model/prefill.go`](../engine/model/prefill.go), [`engine/model/vision.go`](../engine/model/vision.go), [`tier/layer.go:1067`](../jit/gpu/tier/layer.go#L1067), [`tier/layer.go:2299`](../jit/gpu/tier/layer.go#L2299), [`tier/layer.go:3417`](../jit/gpu/tier/layer.go#L3417), [`tier/session.go`](../jit/gpu/tier/session.go).

## What happens without the optimized kernel

This is where “build computation at runtime” needs concrete examples. A specialized implementation may be unavailable while the operation remains executable through another generated path.

| Missing or unsuitable optimization | Runtime alternative | Source |
|---|---|---|
| AVX-VNNI packed dot instruction | Emit AVX2 multiply/add sequences for supported packed formats. The SSE tier emits its 128-bit sequence. | [`cpu/prevnni.go:77`](../jit/cpu/prevnni.go#L77), [`cpu/sse_packed.go`](../jit/cpu/sse_packed.go) |
| CPU interleaved/shaped matvec | Use the generated one-row kernel. | [`engine/nn/jit.go:918`](../engine/nn/jit.go#L918) |
| CPU fused or wide packed matvec | Use generated packed row tiles and tails. | [`engine/nn/packed.go`](../engine/nn/packed.go) |
| Requested CPU token tile cannot be emitted | Try smaller token widths and cache a successful generated variant. | [`engine/nn/packedmm.go:309`](../engine/nn/packedmm.go#L309) |
| CPU multi-token matmul unavailable | Run generated fused kernels per token, a one-token tiled kernel, or the per-token generated matvec composition. | [`engine/nn/packedmm.go:111`](../engine/nn/packedmm.go#L111), [`engine/model/prefill.go:700`](../engine/model/prefill.go#L700), [`engine/model/vision.go:744`](../engine/model/vision.go#L744) |
| GPU matrix-instruction projection unavailable | Build and compile the dot-product `MatVec` batch variant. | [`tier/layer.go:6367`](../jit/gpu/tier/layer.go#L6367), [`tier/layer.go:6429`](../jit/gpu/tier/layer.go#L6429) |
| GPU tensor-core attention unavailable | Build and compile tiled attention-score compute kernels, including the sliding-window variant. | [`tier/layer.go:2969`](../jit/gpu/tier/layer.go#L2969) |
| GPU subgroup width unavailable or softmax check fails | Build and compile the scalar GPU softmax; head normalization also has scalar/subgroup variants. “Scalar” here is still device code. | [`tier/layer.go:3501`](../jit/gpu/tier/layer.go#L3501), [`kernels/attn.go`](../jit/gpu/kernels/attn.go) |
| Device block admission fails | Leave that block for the CPU execution path and its generated kernels. | [`engine/model/forward.go:960`](../engine/model/forward.go#L960) |
| GPU rotary-table path is unsuitable | Use/upload the host-generated rotary table. | [`tier/layer.go:4423`](../jit/gpu/tier/layer.go#L4423), [`engine/nn/ropetable.go`](../engine/nn/ropetable.go) |

These are implemented alternatives for known operations. An unknown weight representation, unsupported plan feature, invalid shape, or missing mandatory host generator can still be refused. The code does not synthesize arbitrary new model semantics from an unrecognized operation name.

## The compiler building blocks

The higher-level generators construct their kernels from lower-level operations at runtime:

- **CPU:** `Buf` encodes x86 instructions and branches; `A64` encodes ARM64; SSE helpers provide legacy encodings and vector/scalar loop bodies. Dot-product, unpack, exponentiation, half conversion, reduction, and tail helpers are reused inside generated kernels. `cpuidProbe` itself emits a small executable hardware probe. These helpers are not separately launched NN kernels.
- **GPU IR:** typed values, constants, parameters, thread/group IDs, integer/float arithmetic, shifts/masks, min/max, exponent/square root, conversions/bitcasts, loads/stores, packed dot products, FMA, comparisons/selects, loops/phis, subgroup shuffles, shared memory, barriers, fragment MMA, and collective tile operations.
- **GPU lowerers:** turn those operations into PTX, SPIR-V instructions, or MSL source. A supported operation can be expanded into several target instructions or expressions. Unsupported matrix forms are explicitly rejected so the tier can choose an alternate kernel family.

Matrix-operation coverage is deliberately specific: fragment-shaped `OpMMA` lowers through PTX; collective tile operations have SPIR-V and MSL implementations with shape/type constraints. The presence of tile support in a lowerer does not mean every production projection selects that path. The layer's `MatVecMMA` fallback above is the relevant production decision.

Sources: [`cpu/asm.go`](../jit/cpu/asm.go), [`cpu/a64.go`](../jit/cpu/a64.go), [`cpu/sse.go`](../jit/cpu/sse.go), [`cpu/sse_helpers.go`](../jit/cpu/sse_helpers.go), [`gpu/ir/ir.go`](../jit/ir/ir.go), [`gpu/ir/tile.go`](../jit/ir/tile.go), [`gpu/ptx/lower.go`](../jit/ptx/lower.go), [`gpu/spirv/lower.go`](../jit/spirv/lower.go), [`gpu/msl/msl.go`](../jit/msl/msl.go).

## Boundaries that matter for accurate wording

- “Runtime-generated kernels” is accurate for the kernel inventory. It does not mean every line of tokenization, image preprocessing, packing, paging, data movement, or orchestration is JIT-compiled.
- Core compute fallback is generally another generated implementation or composition of generated operations. It is not the float64 interpreted inference tier described by some old comments.
- In release builds, missing activation-quantize or rotary-table generation does not silently enter the historical Go loop. `jit/cpu/quantgo.go` and `engine/nn/ropego.go` reject that path; the comparison implementations are behind `jitllmtest`.
- Greedy selection calls generated `Argmax32JIT`; the old scan described in some comments is gone. Routing and sampling likewise require generated kernels.
- Raw-layout GEMM's `PackActJIT` still has a Go packing helper when no packing code is supplied. Conversion and host preparation also contain ordinary Go arithmetic. These are separate from claiming a general interpreted model execution tier.
- `.jlm` stores prepared weight layouts. It reduces paging/repacking overhead; executable CPU/GPU kernels and layer execution are built by the runtime.

## Verification in this inventory

The CSV is the script's output over the tree it was committed with. As a check on the derivation, the script was compared against the previous hand-assembled CSV: every row that names a declaration still present reproduces exactly apart from its line number, and the only rows it drops are the declarations since removed (`GEMM` and `Stream` in `jit/gpu/kernels`, the `jit/gpu/handptx` PTX text generator).

No tests were run for this refresh. The focused test run recorded with the previous snapshot (emitter-table completeness, SSE ISA separation, packed and tiled generator coverage, cross-backend lowering) belonged to that snapshot and is not carried forward as evidence for this one. The family descriptions and the prose sections above are read from the source and its doc comments, not measured; no benchmark results are part of this inventory.

# Kernel capabilities and generated alternatives

Inspected against the working tree based on `b1ce938`. This is an engineering table for the README work, not proposed README copy. See the [generator inventory](kernel-inventory.md), [symbol index](kernel-inventory.csv), and [external implementation comparison](kernel-comparison.md).

**Every execution kernel in the table is built at runtime.** “Specialized” means a generator for a particular operation, layout, or hardware strategy. “Generated alternative” means another kernel or a sequence of generated kernels that the runtime actually constructs today. It does not mean a precompiled generic library call.

## Operation coverage

| Computation | CPU generation | GPU generation | When the specialized path is unavailable |
|---|---|---|---|
| Float projections | F32/F16/BF16 row kernels; interleaved shapes; raw-layout GEMM where admitted | Float `MatVec`, with row/token/split variants | CPU shaped matvec can use generated single-row code; multi-token work can invoke generated matvec per token. GPU floats use the ordinary generated projection path without asking for integer MMA. |
| Packed quantized projections | `PackedMatVec`, `PackedWide`, `PackedFused`; generated weight decoding and activation dot products | `MatVec`, including split reduction, row tiles, bias and batched tokens | CPU wide/fused refusal uses generated base row tiles and tails. GPU creates the applicable `MatVecShape` and compiles its IR. |
| Prompt / multi-token matmul | `PackedTiled`, `EmitGEMMWindow`, `EmitPackAct` | `MatVecMMA` or batched `MatVec` | CPU tries smaller token tiles, then supported per-token generated paths. GPU MMA refusal builds and compiles the dot-product batch kernel; row tiling can shrink to two or one. |
| Activation preparation | Wide/narrow activation quantization; GEMM packing | `Quantize` | Select the model's supported activation window and emit that variant. A missing mandatory quantizer is an error; release inference does not silently switch to the historical Go reference loop. |
| Embedding rows | `PackedRow`; F16/BF16 `Widen` | Embedding preparation need not be a separate device kernel | Packed rows are decoded by generated host code; F32 rows can be copied. This is not evidence of a missing model operation. |
| RMSNorm / LayerNorm | `RMSNorm`, `LayerNorm` | Partitioned reduction and apply kernels; row-batched forms; LayerNorm variance stage | GPU normalization is already constructed as a sequence of generated stages. No universal fallback for an unsupported normalization definition is claimed. |
| Activations and gates | `Act`, `ActMul`, `SigmoidMul` | `Act`, `ActMul`, `SigmoidMul`, `SiLUMul` | Supported activation formulas are generated for the chosen backend. Existing kinds do not automatically cover new formulas such as SiTU-GLU. |
| Residual, scale, softcap | `Axpy`, `Scale`, `Softcap`; addition is AXPY with scale one | `Add`, `Scale`, `Softcap` | Composed layer execution calls these generated primitives where no larger fused operation exists. |
| RoPE and rotary tables | `RoPE`, `RopeTable` | `RoPE`, row/transposed forms, `RopeTable` | An unsuitable device table path uses a host-generated table uploaded to the device. Rotation still uses generated computation. |
| Attention scores | Single-query, paired-query and tiled-query `AttnScores` families | `AttnScores`, row/transposed/tiled variants, optional `AttnScoresMMA`, windowed variants | GPU tensor-core refusal generates tiled FMA score computation. CPU uses the attention kernel matching its head geometry and KV layout. |
| Attention softmax | `Softmax` | `SoftmaxRows`, `SoftmaxRowsSink`, scalar/subgroup variants | Missing subgroup guarantees or a failed agreement check selects a newly built scalar **GPU** softmax. It does not move this operation to interpreted host arithmetic. |
| Attention value accumulation | `AttnAcc`, paired variants and `Into` forms | Ordinary, row-batched, tiled and split accumulation; F16 split variant | Attention is composed from generated score, softmax and accumulation stages. `Into` supports adding page/window contributions. This provides attention without a fused Flash Attention kernel. |
| Head normalization | Generated normalization operations | Scalar/subgroup `HeadNorm`, `HeadNormRows` | The selected scalar or subgroup implementation is generated. |
| KV writes and layouts | Host layout/copy/conversion plus generated attention readers | `CopyAt`, row/transposed forms, `CopyAtF16` | Host KV has F32/F16 layout paths. Device `KVF16` narrows V; it is not blanket support for every cache in every precision. Paging and migration are separate from arithmetic precision. |
| MoE routing | `MoETopK`, `MoERoute`, generated router projection | `RouterMatVec`, group score/mask, rank and weight kernels | Runtime composes router projection, grouped/ungrouped selection and weighting for the supported plan. |
| MoE expert FFNs | Generated packed multi-matrix/gather execution; gate/up, activation, down | Expert-bank/slot `MatVec`; expert combination; shared-expert and indexed-bias kernels | CPU batched dispatch can decline to generated per-expert projections. Host prefill visits each selected expert once per chunk and applies it to all selecting rows. GPU builds the supported expert and slot shape. |
| Recurrent / linear attention | `GatedDelta`, `GatedDeltaChan`, `Conv1d`, `DeltaGate` | `GatedDeltaStep`, channel variant, convolution/shift, gate/slice kernels | Layer execution composes projections, convolution, gate calculation, state update and output normalization. Existing Kimi-Linear recurrence does not imply support for every later KDA gate formula. |
| MLA | Generated projections and attention at latent-specific dimensions | `emitMLA` composes generated projections, score/softmax/accumulation and output projection | Absorbed/latent MLA is available without a dedicated fused FlashMLA kernel. |
| Vision transformer blocks | Generated projections, normalization, attention and activations | The same families specialized for vision rows and plan | Shape-specific batching can use the ordinary generated projection path. Image preprocessing itself is ordinary host code. |
| Output head and sampling | Projection, `Argmax`, `SampleSegMax`, `SampleDraw`, `SamplePenalty` | Prepared output-head projection and optional norm | Sampling is generated CPU computation; a GPU sampling kernel is not required for this path. Missing mandatory CPU generation is rejected. |

Sources by family: [CPU dispatch](../jit_old/cpu/emitters.go), [CPU orchestration](../engine/nn/jit.go), [packed matvec](../engine/nn/packed.go), [packed matmul](../engine/nn/packedmm.go), [normalization/elementwise](../engine/nn/norm.go), [GPU matrix generation](../jit_old/gpu/kernels/matvec.go), [GPU attention](../jit_old/gpu/kernels/attn.go), [GPU elements](../jit_old/gpu/kernels/elem.go), [GPU MoE](../jit_old/gpu/kernels/moe.go), [GPU recurrence](../jit_old/gpu/kernels/recurrent.go), [MoE scheduling](../engine/model/moe.go), [MLA](../engine/model/mla.go), [KV layout](../engine/model/kvlayout.go).

## What the runtime can build without a specialized kernel

| Missing specialization | What is fully generated today | Where the choice is implemented |
|---|---|---|
| AVX-VNNI instruction | An AVX2 multiply/add dot-product sequence | [`HostDotKind` and dot emitter](../jit_old/cpu/prevnni.go) |
| AVX2 host tier, but the SSE floor is available | Independent legacy SSE machine code for the supported operation set | [`EmittersFor` and SSE table](../jit_old/cpu/emitters.go) |
| CPU wide/fused packed sweep | Row-tile and tail machine-code kernels | [`MatVecPacked`](../engine/nn/packed.go) |
| Requested CPU token tile | A smaller successful generated tile; applicable per-token composition | [`packedmm.go`](../engine/nn/packedmm.go), [`State.mm`](../engine/model/prefill.go) |
| GPU integer tensor-core kernel / lowering | Ordinary batched dot-product kernel generated from IR | `batchMV` → `batchMVDot4` in [batchprep.go](../jit_old/gpu/tier/batchprep.go) |
| GPU attention matrix-instruction path | Tiled FMA score kernel, including sliding-window form | `initScratch` in [scratchinit.go](../jit_old/gpu/tier/scratchinit.go) |
| Suitable GPU subgroup softmax | Scalar device kernel generated and compiled for the row shape | `pickSoftmax` in [scratchinit.go](../jit_old/gpu/tier/scratchinit.go) |
| Native packed-dot instruction in Metal's exposed compute model | Generated MSL helper arithmetic for IR `Dot4`, including bounded-input variants | [MSL lowerer](../jit_old/gpu/msl/msl.go) |
| A single fused attention kernel | Generated score + softmax + value-accumulation kernels, assembled into layer execution | `initScratch` in [scratchinit.go](../jit_old/gpu/tier/scratchinit.go) and `layersOnce` in [call.go](../jit_old/gpu/tier/call.go) |
| A single fused dense / MoE / recurrent / MLA block kernel | The model-specific sequence of generated primitives, with buffers, shapes and state bindings prepared at runtime | [`LayerPlan`](../engine/nn/device.go), [`planFor`](../engine/model/forward.go), `PrepLayer` in [prep.go](../jit_old/gpu/tier/prep.go), `emitLinear` in [mixers.go](../jit_old/gpu/tier/mixers.go), `emitMLA` in [mladecode.go](../jit_old/gpu/tier/mladecode.go) |
| Device admission for a supported block | Host execution using generated CPU kernels | `offerRange` in [forward.go](../engine/model/forward.go) |
| An unknown quant format or new model operation | **No automatic semantic fallback.** Add the format/operation definition and its generator or composition before it can run. | [Packed formats](../format/quant/types.go), [GPU format mapping](../jit_old/gpu/kernels/quantof.go), [activation definitions](../jit_old/gpu/kernels/actkind.go) |

The generated compute layer is real capability: it supplies the operation when a larger optimized implementation is absent. It does not automatically reproduce the optimized implementation's memory traffic, fusion, precision, or speed. For example, generated three-stage attention computes attention, but it does not gain Flash Attention's online-softmax memory behavior merely by being generated.

## Hardware and representation boundaries

| Target | Runtime output | Relevant boundary |
|---|---|---|
| x86-64 SSE | Native executable bytes | SSE2/SSE3/SSSE3/SSE4.1 baseline; optional generators may decline. |
| x86-64 AVX2 | Native executable bytes, optionally AVX-VNNI dot sequences | AVX2/FMA/F16C baseline. AVX-512 detection exists, but no AVX-512 compute tier is selected. |
| ARM64 | Native NEON executable bytes | Packed quantized dot kernels use SDOT where the chip has FEAT_DotProd and widen it to SMULL/SADDLP/ADDP where it does not (`jit_old/cpu/sdotemu.go`); the two give the same bits. |
| CUDA | Generated PTX loaded by the driver | Fragment MMA paths are available where their format, shape and target requirements pass. |
| Vulkan | Generated SPIR-V and a runtime compute pipeline | Ordinary compute paths work independently of fragment MMA. Collective matrix support in the lowerer is not the same as production `MatVecMMA` dispatch. |
| Metal | Generated MSL compiled into a runtime pipeline | Dot-product arithmetic and scalar/subgroup compute are generated. Collective tile support likewise does not imply every production projection uses it. |

Packed weight formats: **Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4**, plus F32/F16/BF16 float paths. Metadata recognition in `quant.Type` covers more representations than production packed execution; those are different capabilities.

The [kernel inventory](kernel-inventory.md) lists every generator by symbol, file and build constraint (`scripts/kernel-inventory.py` regenerates it). This table adds source tracing, not a new all-device numerical or performance claim.

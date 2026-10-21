# README capability audit

Every claim in the top-level [README](../README.md), mapped to the code that
implements it and the test that holds it. A README claim with no row here is a
claim with nothing behind it; add the row or remove the claim. Performance
figures map to [`docs/perf/current.md`](perf/current.md), which carries the
commands and engine versions.

Paths are relative to the repository root. "code" is where the behaviour lives;
"test" is the gate that fails if it stops being true. A test that needs a model
or a device skips without one (see [`docs/testing.md`](testing.md)).

## Get started and the CLI

| Claim | Code | Test |
|---|---|---|
| Go 1.26 or newer | `go.mod`, `server/go.mod`, `ui/go.mod` (`go 1.26`) | CI build (`.github/workflows/ci.yml`) |
| `convert -o DIR NAME out.jlm` fetches a catalog model | `cmd/jitllm/convert.go:convertCmd`, `libraryArgs`; `convert/library/models.go` (`smollm2-360m-instruct`, `llama-3.2-1b-instruct`, `smolvlm-256m-instruct`) | `convert/library/library_test.go`, `cmd/jitllm/library_test.go` |
| `run -chat -n N model.jlm prompt` | `cmd/jitllm/main.go` (`-chat`, `-system`, `-n`, `-temp`, `-top-p`) | `cmd/jitllm/cli_test.go:TestRunRefusesContradictoryFlagsBeforeLoading`, `engine/model/chat_test.go:TestChatTemplateIsAppliedAndSpecialsAreNotDoubled` |
| Inference reads `.jlm` only | `engine/model/model.go:Open` (`NotConvertedError`) | `cmd/jitllm/cli_test.go:TestSubcommandsRefuseAModelTheyCannotUse`, `server/connect_test.go:TestHandingAGGUFToLoadReturnsTheConvertCommand` |
| Flags go before the model path | `cmd/jitllm/main.go:misplacedFlag` | `cmd/jitllm/flagorder_test.go` |
| Subcommand table (`hardware`, `library`, `convert`, `run`, `embed`, `tokenize`, `info`, `speed`, `batch`, `verify`, `asm`) | `cmd/jitllm/main.go:main` dispatch and `usage` | `cmd/jitllm/cli_test.go:TestDispatchAndFlagErrorsExitWithTheirStatus`, `TestSpeedPrintsOneRatePerRepetition`, `TestVerifyAgreesWithTheHostOnARealDevice`, `TestTokenizePrintsEveryPiece`, `TestInfoDescribesTheCommittedGGUF` |

## The five principles

| Principle | Code | Test |
|---|---|---|
| JIT; only the trampolines are assembly | `jit_old/cpu/trampoline_{amd64,arm64}.s` | `jit_old/cpu/noasm_test.go:TestNoAssemblyBesidesTheTrampolines` |
| No Go compute | `internal/oracle` is test-only; `engine/nn` entry points | `internal/oracle/guard_test.go:TestNoProductionCodeImportsTheOracle`, `engine/model/nnaudit_test.go:TestEveryModelRunsGenerated` |
| Relocation | `engine/model/forward.go:SetGPULayers`, `SetDevice`; `engine/model/kvpage.go:migrateKV`, `migrateRec` | `engine/model/reloc_test.go:TestLayerRelocatesToAnotherDevice`, `engine/model/relocate_test.go:TestRelocatedBlocksComeBack`, `engine/model/kvpaged_test.go:TestKVPagesRelocateWithTheLayer`, `engine/model/migraterows_test.go:TestBatchSeamMovesCarryEveryRow`, `engine/model/mladevice_test.go:TestMLASeamMovesWithItsHistory`, `engine/model/demoterec_test.go:TestShrinkingTheSeamBringsTheSummaryHome` |
| Paging | `format/jlm/read.go`, `format/jlm/partial.go`; `engine/model/model.go:SetPageBudget`, `WithPageBudget` | `engine/model/paging_test.go:TestPagingDoesNotChangeTheAnswer`, `engine/model/pagebudget_test.go:TestOpenHonoursAPageBudget`, `TestForwardBatchFaultsItsBlocksIn`, `engine/model/offheap_test.go:TestTwoPagedModelsShareTheProcess`, `engine/model/swapages_test.go:TestWindowedLayersHoldTheirWindow` |
| Low to no Go allocation; frames off the heap | `format/jlm/offheap.go:SetOffHeap` | `engine/model/decodealloc_test.go:TestDecodeDoesNotAllocate` |

## Supported models

| Claim | Code | Test |
|---|---|---|
| 65 GGUF text architecture names, 57 graphs | `convert/config.go:archOf` (66 keys including `clip`; 58 distinct `jlm.Arch` values including `ArchCLIP`, so 57 text graphs; `configOf` also gives glm4moe with M-RoPE sections `ArchGLM4VMoE` and windowed olmo2 `ArchOLMo3`, codes over graphs already counted), `convert.Architectures` | `convert/library/arch_test.go:TestEveryArchitectureIsObtainable` |
| An architecture not on the list is refused at conversion | `convert/config.go:configOf` (`ErrNotImplemented`) | `convert/notimpl_test.go` |
| Mixtral converts as `llama` | `convert/config.go` (mixture read from the file, not the arch) | `convert/mixtralbanks_test.go` |
| 29 Hugging Face class names | `convert/safetensors.go:hfArchTable` (`PhimoeForCausalLM` is an alias of `PhiMoEForCausalLM`) | `engine/model/hfgold_test.go:TestSafetensorsMatchTransformers`, `convert/hfdense_test.go:TestDenseSafetensorsMatchesItsGGUF`, `convert/hfmoefamily_test.go:TestMixtureFamiliesConvertAlikeFromEitherInput` |
| Per-architecture correctness against llama.cpp | `engine/model` graphs | `engine/model/llamacpp_test.go:TestGreedyMatchesLlamaCpp` |
| `hunyuan_vl`: HunyuanOCR's text model, hunyuan-dense's block under XD-RoPE | `convert/config.go:archOf`; `format/jlm/archvision.go` (`ArchHunyuanVL`, `XDRope`) | `engine/model/hunyuanvl_test.go:TestHunyuanVLTextMatchesTransformers`, `TestHunyuanVLKeepsTheFivePrinciples` |
| Kimi-K3 held to its own reference code on the trained Kimi-K3-0.40B | `convert/kimik3.go`; `engine/model` Kimi-K3 graph | `engine/model/kimik3real_test.go:TestKimiK3RealMatchesReference` (F32, F16, Q8_0, Q4_K_M, MXFP4 GGUFs against `scripts/k3gold.py`); a mixed-quantization GGUF converts: `convert/kimik3_test.go:TestK3JoinsAMixedQuantization` |
| Embedding models (`bert`, `nomic-bert`, Qwen3-Embedding, EmbeddingGemma) | `engine/model/encoder.go`, `engine/model/embedstate.go`, `engine/model/embed.go:NewEmbedder`; `cmd/jitllm/embed.go` | `engine/model/embed_test.go:TestEmbeddingsMatchReference`, `TestEmbeddingGateDiscriminates` |
| Weight formats F32, F16, BF16, Q4_0, Q5_0, Q5_1, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4 | `format/quant/types.go` (`PackedTypes`, `Dequantable`) | `internal/oracle/packed_test.go:TestPackedPayloadIsNativeWidth` |
| `-q8` stores float safetensors matrices as Q8_0 | `cmd/jitllm/convert.go` (`-q8`), `convert/safetensors.go` | `engine/model/hfq8_test.go:TestSafetensorsQ8MatchTransformers` |

## Vision

| Claim | Code | Test |
|---|---|---|
| One container carries text and tower | `convert/gguf.go:FromGGUFs`; `engine/model/model.go:Tower` | `engine/model/vision_test.go:TestOneContainerCarriesTextAndVision` |
| 17 projectors | `convert/config.go:visionOf` (`idefics3`, `mlp`, `qwen2vl_merger`, `gemma3`, `internvl`, `resampler`, `janus_pro`, `qwen2.5vl_merger`, `pixtral`, `phi4`, `gemma4v`, `gemma3nv`, `llama4`, `glm4v`, `hunyuanvl`, `kimivl`, `qwen3vl_merger`; anything else `ErrNotImplemented`) | per-family gates, e.g. `TestLlavaTowerEncodes`, `TestQwenTowerMatchesTransformers`, `TestGemma3TowerMatchesLlamaCpp`, `TestGemma3nTowerMatchesTransformers`, `TestGemma4TowerMatchesTransformers`, `TestInternVLTowerMatchesLlamaCpp`, `TestMiniCPMVTowerMatchesLlamaCpp`, `TestJanusTowerMatchesLlamaCpp`, `TestPixtralTowerMatchesTransformers`, `TestPhi4TowerMatchesTransformers`, `TestLlama4TowerMatchesTransformers` (all `engine/model`) |
| Kimi-VL (`kimivl`, MoonViT; its text model is `deepseek2`) | `convert/config.go:visionOf` (`jlm.ProjKimiVL`); `engine/model/kimivit.go` | `engine/model/kimivl_test.go:TestKimiVLTextMatchesTransformers`, `TestKimiVLTowerOnDeviceMatchesHost`, `TestKimiVLRealTowerMatchesMoonshot`, `TestKimiVLKeepsTheFivePrinciples` |
| HunyuanVL and HunyuanOCR (`hunyuanvl`) | `convert/config.go:visionOf` (`jlm.ProjHunyuanVL`), `convert/hunyuanvl.go`; `engine/model/hunyuanvit.go` | `engine/model/hunyuanvl_test.go:TestHunyuanVLLaysTheImageOutAsTransformersDoes`, `TestHunyuanVLTowerOnDeviceMatchesHost`, `TestHunyuanVLRealTowerOnDevice`, `TestHunyuanVLPictureCountsItsNewlineAndEndRows` |
| Image preprocessing is generated code on AVX2, SSE and NEON, held to each family's reference processor | `jit_old/cpu/imgproc_const.go` (the kernels' contracts), `jit_old/cpu/imgproc_sse.go`, `jit_old/cpu/imgproc_neon.go`; `engine/nn/image.go`; `engine/model/imageproc.go` | `internal/oracle/imagepath_test.go:TestImagePathRunsGeneratedCode`; `jit_old/cpu/imgproc_test.go`, `imgpixel_test.go`, `imgaxis_test.go`, `imgsincos_test.go`; against the processors `engine/model/visiongold_test.go:TestPreprocessMatchesTransformers`, `TestQwenPreprocessMatchesTheProcessor`, `TestPixtralPreprocessMatchesTheProcessor`, `TestPhi4PreprocessMatchesTheProcessor`, `TestLlama4PreprocessMatchesTheProcessor`, `TestGemma4PreprocessMatchesTheProcessor`, `TestGemma3nPreprocessMatchesTheProcessor`, `TestKimiVLChatSpansAreTheProcessors`, `TestHunyuanVLChatSpansAreTheProcessors` |
| Tower blocks page and run on a GPU | `engine/model/vision.go`; `jit_old/gpu/tier` vision plans | `engine/model/vision_test.go:TestTowerEncodesUnderAPageBudget`, `TestTowerOffersItsBlocksToTheDevice`, `TestTowerOnDeviceMatchesHost` |
| `run -image` | `cmd/jitllm/picture.go`, `cmd/jitllm/main.go` (`-image`) | `cmd/jitllm/vlm_test.go:TestVLMTowerSizesTheState` |

## Devices

| Claim | Code | Test |
|---|---|---|
| AVX2/FMA/F16C tier, AVX-VNNI as a dot kind | `jit_old/cpu/tier.go` (`TierAVX2`), `jit_old/cpu/prevnni.go` | `engine/nn/prevnni_equal_test.go:TestPreVNNIMatchesVNNIBitForBit` |
| SSE4.1+SSSE3 tier; refused below it | `jit_old/cpu/tier.go` (`TierSSE`), `jit_old/cpu/sse.go`; `engine/nn/host.go:HostRefusal` | `engine/model/sse_models_test.go:TestEveryModelRunsSSE`, `jit_old/cpu/tier_test.go` |
| ARM64 NEON with SDOT | `jit_old/cpu/tier.go` (`TierNEON`), `jit_old/cpu/packed_a64.go` | `jit_old/cpu/exec_a64_test.go:TestA64ExecuteSDOT` |
| CUDA, captured decode graphs | `jit_old/gpu/backend/cuda.go`, `jit_old/gpu/cuda`, `jit_old/gpu/ptx` | `jit_old/gpu/backend/cudatiming_test.go:TestCUDAGraphTimingReportsEveryReplay` |
| Vulkan; a software device is declined | `jit_old/gpu/backend` Vulkan, `jit_old/gpu/spirv` | `jit_old/gpu/backend/vulkan_software_test.go:TestSoftwareVulkanIsDeclined` |
| Metal | `jit_old/gpu/metal`, `jit_old/gpu/msl` | `jit_old/gpu/lowertest` (MSL goldens), `engine/model/hybrid_test.go:TestHybridAttentionBlockOnDevice` (metal arm) |
| No cgo; libraries loaded at run time through goffi | `jit_old/gpu/ffi`; `go.mod` (one requirement, `github.com/go-webgpu/goffi`); `.goreleaser.yaml` (`CGO_ENABLED=0`) | `internal/srcgate/readme_test.go:TestNoCgo` (no Go file in the three modules imports `"C"`, which a `CGO_ENABLED=0` build alone would leave out rather than refuse); CI cross-builds every release target and module with `CGO_ENABLED=0` (`.github/workflows/ci.yml`, job `build`) |
| One card under two backends is one device | `jit_old/gpu/tier/choose.go`, device UUIDs in `jit_old/gpu/backend` | `jit_old/gpu/tier/devidentity_test.go:TestOnePhysicalDeviceUnderTwoBackendsIsOfferedOnce`, `jit_old/gpu/backend/identity_test.go:TestTheBackendPreferenceIsCUDAThenMetalThenVulkan`, `jit_old/gpu/tier/open_test.go:TestOneCardTwoAPIsIsOnePool` |
| `-devices`: `auto` one device by measuring; `all` or a bare `gpu` every distinct device; a bare `cuda` or `vulkan` every device of that backend, fastest first; `cuda:N`, `vulkan:SEL`, `gpu:N` one; `metal` the system default device | `jit_old/gpu/tier/open.go:ParseDevices`, `openEntries` (`everyCUDA`, `everyVulkan`, `allDevices`), `jit_old/gpu/tier/multi.go:order`, `jit_old/gpu/tier/choose.go:choose` | `jit_old/gpu/tier/devspec_test.go:TestABareBackendNameIsEveryDeviceOfIt` |
| Release targets: Linux, macOS and Windows on x86-64 and ARM64, one archive each for the CLI, the server, the desktop app and the terminal app | `.goreleaser.yaml` (`goos`, `goarch`, `archives`) | CI build loop over `linux/amd64 linux/arm64 darwin/arm64 darwin/amd64 windows/amd64 windows/arm64`; the `test` job builds, vets and runs the model-free tests natively on linux/amd64, linux/arm64, darwin/arm64 and windows/amd64 (darwin/amd64 is only cross-built) |

## Convert

| Claim | Code | Test |
|---|---|---|
| GGUF, HF directory, URL or catalog name | `cmd/jitllm/convert.go:convertCmd`, `convert/hf` | `convert/hf/fetch_test.go:TestFetchResumesInsteadOfStartingOver` |
| A Hugging Face repository is read by range, not downloaded | `cmd/jitllm/stream.go:openStreamed`, `convert/hf/remote.go` | `convert/hf/remote_test.go:TestRemoteReadsTheRangesItIsAskedFor` |
| `-chat-template FILE` (tokenizer_config.json, chat_template.json, .jinja, a directory) | `cmd/jitllm/convert.go` (`-chat-template`), `convert/chattemplate.go` | `convert/chattemplate_test.go:TestChatTemplatesFromEveryForm`, `TestConvertStoresTheChatTemplate`, `engine/model/chat_template_override_test.go:TestChatTemplateOptionReplacesTheGGUFs` |
| `HF_TOKEN` for gated repositories | `cmd/jitllm/fetch.go:hfClient`, `convert/hf/ref.go` | `convert/hf/fetch_test.go` (the refusal names `HF_TOKEN`) |
| Stop tokens: a GGUF's eot, eom and FIM ids and a safetensors model's `generation_config` eos list reach the container, and generation stops on them | `convert/vocab.go`, `convert/hfvocab.go`; `format/jlm/vocab.go` (`Vocab.Stop`), `format/jlm/sections.go`; `tok/spm.go:IsEOG` | `convert/stops_test.go:TestStopSetRoundTrips`, `TestStopSetRefusesAnIdOutsideTheVocabulary`, `tok/eog_test.go:TestStopSetMatchesLlamaCpp`, `TestStatedStopsEndAGeneration`, `engine/model/stops_test.go:TestChatStopsAtTheTurnsEnd` |
| The container carries config, tokenizer, stop tokens, templates, tower | `format/jlm` sections | `engine/model/vision_test.go:TestOneContainerCarriesTextAndVision`, `format/jlm/convert_test.go:TestContainerCarriesTheTokenizer` |

## The server

| Claim | Code | Test |
|---|---|---|
| `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/v1/messages` | `server/surface.go` (mux), `server/compat_openai.go`, `server/compat_anthropic.go`, `server/compat_embed.go` | `server/compat_test.go:TestOpenAIChatStreamArrivesTokenByToken`, `TestLegacyCompletionsCarryTextNotAMessage`, `TestOpenAIModelsListsLoadedModelsAndTheirFileNames`, `TestAnthropicStreamIsNamedEventsAndEndsWithMessageStop` |
| Embeddings as floats or base64; Connect `Embed` | `server/compat_embed.go` (`encoding_format`), `server/embed.go` | `server/embed_test.go:TestEmbedMatchesTheEmbedder`, `TestEmbedRefusesWhatCannotEmbed` |
| `/healthz` | `server/http.go` | `server/engine_test.go:TestHealthzAndTheCompatShimsAreMounted` |
| Tool calling through the model's template; `tool_calls` / `tool_use` | `server/tools.go`, `engine/model/tools.go:ParseToolCalls`, `engine/model/chat.go:ChatIDsTools` | `server/tools_test.go:TestToolRequestsReachTheTemplate`, `TestOpenAIToolCallStreams`, `TestAnthropicToolUse` |
| Connect services | `proto/jitllm/v1/*.proto`, `server/service_*.go` | `server/connect_test.go:TestConnectGenerateStreamsStartedTokensFinishedInThatOrder`, `server/e2e_test.go:TestEndToEndAgainstARealContainer` |
| `ignore_eos` on `/v1/completions`, `/v1/chat/completions` and `GenerateRequest`; `jitllmd run -ignore-eos` | `server/compat_openai.go` (`ignore_eos`), `proto/jitllm/v1/inference.proto` (field 10), `server/engine.go`, `server/batch.go`, `server/cmd/jitllmd/run.go` | `server/ignoreeos_test.go:TestIgnoreEOSRunsToMaxTokens`, `TestIgnoreEOSRunsToMaxTokensBatched` |
| Device generates decode together (`-max-batch`) | `server/batch.go`, `engine/model/step.go:StepRuns` | `server/batch_test.go:TestBatchConcurrentGreedyGeneratesShareSteps` |
| `jitllmd` verbs and `serve` flags | `server/cmd/jitllmd/main.go:verbs`, `serve` | `server/cmd/jitllmd/flagorder_test.go`, `client_test.go:TestDevicesPrintsTheSpendableTotalAndNamesAUnifiedHeap` |

## Memory and placement

| Claim | Code | Test |
|---|---|---|
| `-maxmem` host budget | `cmd/jitllm/main.go` (`bytesFlag "maxmem"`), `engine/model/model.go:SetPageBudget` | `cmd/jitllm/cli_test.go:TestATinyBudgetPagesAndSaysSo`, `engine/model/pagebudget_test.go:TestOpenHonoursAPageBudget` |
| Per-device budgets, `-devices` grammar | `jit_old/gpu/tier/open.go`, `cmd/jitllm/device.go` | `cmd/jitllm/devices_test.go:TestOpenDevicesNamesTwoDevicesAndPoolsThem` |
| `-placement` with `~` (stream) and `!` (pin) | `engine/model/options.go:ParsePlacement` (`Place.Stream`) | `engine/model/placement_test.go:TestParsePlacementStreams` |
| Expert streaming reads only the routed sheets | `engine/model/model.go:ensureExperts`, `jit_old/gpu/tier/layer.go:streamBank` | `engine/model/stream_test.go:TestStreamedExpertBankMatchesTheResidentOne` |
| `-gpu-grow`, `-relocate`, `-tune-seam` | `engine/model/forward.go` (`SetGPUTarget`, `SetRelocate`), `engine/model/seamtune.go` | `engine/model/reloc_test.go:TestSeamMovesWhileServing`, `engine/model/relocate_test.go:TestRelocateKeepsTheDeviceWhenTheContextOutgrowsIt` |
| Several devices at once | `jit_old/gpu/tier/multi.go` | `engine/model/multidev_test.go:TestTheSeamMovesAcrossTwoDevices` |
| Prefix cache, `-kv-cache-max` default 8 GiB | `engine/model/kvpage.go:PrefillCached`, `engine/model/kvfile.go`; `cmd/jitllm/main.go` (`kv-cache-max`, `8<<30`) | `engine/model/kvpaged_test.go:TestPrefixCacheReusesASharedPrefix`, `TestKVStoreRoundTripChangesNoToken` |
| Shared CPU/GPU memory counted once | `jit_old/gpu/backend` `UnifiedMemory`, `jit_old/gpu/tier` budgets | `server/connect_test.go:TestSpendableTotalSubtractsAUnifiedHeapInsteadOfAddingIt`, `engine/model/streamunified_test.go:TestStreamedContainerOnAUnifiedDevice` |

## Go library

| Claim | Code | Test |
|---|---|---|
| The example program compiles | the README's Go block; `engine/model/model.go:Open`, `Close`; `engine/model/chat.go:ChatIDs`; `engine/model/forward.go:NewState`, `Forward`, `Greedy`; `engine/model/prefill.go:Prefill`; `tok/spm.go:IsEOG`, `Decode` | `internal/srcgate/readme_test.go:TestTheREADMEGoExamplesBuild` (builds every Go block of the README against the tree); the calls themselves: `engine/model/forward_test.go`, `engine/model/chat_test.go` |
| One dependency in the core module | `go.mod` | `internal/srcgate/readme_test.go:TestTheCoreModuleHasOneDependency` |
| `SetPageBudget`, `SetGPULayers`, `SetDevice`, `SetKVStore`, `PrefillCached`, `NewEmbedder`, `Scheduler`, `Step` | `engine/model/model.go`, `forward.go`, `kvpage.go`, `embed.go`, `schedule.go`, `step.go` | `engine/model/schedule_test.go:TestSchedulerMatchesSoloSequences`, `engine/model/step_test.go:TestStepAcrossSessionsMatchesEachAlone`, and the rows above |
| Desktop app builds with `CGO_ENABLED=0` | `ui/` module | `ui` tests, e.g. `ui/screen/download_test.go:TestDiscoverListsTheWholeLibrary` |

## Performance

Every figure in the README's performance table is a row of
[`docs/perf/current.md`](perf/current.md):

| README figure | Source in `docs/perf/current.md` |
|---|---|
| V100 prefill up to 4.62x, 34 ahead and 1 at parity; decode up to 1.35x on all 35 | "V100 ... CUDA" board (Qwen3-Next-80B 4.62, Moonlight 1.35 decode, Phi-3.5-mini 1.00) |
| TTFT first on 29 of 33, up to 4.08x | "Time to first token" tables, one card and 2-5 cards, plus the pipelined rows (Qwen3-Next-80B 4506/1104 ms) |
| 8 sessions 1.49x; two 0.94x; four 1.06x | "Multi-card prompts, cold start, sessions together" |
| Batched decode up to 3.72x llama.cpp, 5.53x vLLM | "Batched decode" (olmoe batch 64; Llama-3.1-8B batch 128) |
| Xeon up to 1.67x / 1.06x; mistral.rs 294x / 1208x; ZML 69x | "Xeon" board (gpt-oss-20b prefill, olmoe decode; Qwen3-30B-A3B against mistral.rs; Llama-3.2-1B against ZML) |
| M4 Metal 1.02x / 1.53x; CPU 1.67x / 1.09x; mistral.rs 3.76x / 3.15x | "M4" board (tinyllama Metal prefill, stories15M Metal decode, Qwen2-1.5B CPU prefill, tinyllama CPU decode; tinyllama CPU and gemma-2b CPU against mistral.rs) |

## Claims removed in this revision

- "No dependencies" for the whole repository: the core module has one
  (`goffi`); `server/` and `ui/` are separate modules with their own.
- The model-family table that omitted the state-space hybrids, LFM2, MiniMax,
  Gemma 3n/4, DeepSeek V3.2/V4, Kimi-K3 and the newer mixtures, and listed
  vision as three families.
- The separate "Why .jlm?" section, folded into "Convert a model".

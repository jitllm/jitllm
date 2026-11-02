# Deleting main's code once the composer is the only build

An audit of be40b4cc. "Reached" is by a model, a default runtime condition or a shipped option once the build is flipped. M2 deletes only
what nothing reaches (the commits below, in order); the last table is what stays, what reaches it, and the composer work that frees it.

## State

- Commit 1 is done: `Config.Composed` and `composer.ComposedByDefault` are gone, every tier serves every part, the CPU's
  blocks are composed by default and `WithComposedCPU` is gone; the arms that ran `Composed=0` or a partial set were
  rebased or deleted with their subject.
- Commit 2 in part: the branches the pinned field decided (`layersSession`'s `asked`/`asks`, `promptHeld`'s
  `ComposedPrompt` test, `rowsKind`'s `asks`, `promptDeclines`' unasked case) are gone. `TierBlocks` and `RopeTableHost`
  remain, refused by Stage A when set.

## The flip is two switches

- **Cards.** `tier.go:2509` already sets `Config.Composed = ComposedStep` under `composer.ComposedByDefault`. Pinning it removes what can lower it:
  `JITLLM_GPU_COMPOSED` (`cmd/jitllm/gpuopts.go:230-232`), `verdict.go` (`admitPlan`'s `ComposedAuto` switch), `WithComposedByVerdict` and
  `Config.ComposedAuto` (`knobs.go`, `tier.go:190`), `server/engine.go:741`, `nn.LayerPlan.ComposedOn` (`device.go:894`, `forward.go:1225`).
- **CPU.** `ComposedByDefault` does not turn the CPU on: `Open` (`model.go:1453`) sets `composedCPU` only from `composedReady()&BackendCPU` or
  `WithComposedCPU`. The flip sets it whenever `!composedCPUSet`, and `WithComposedCPU(false)` (`options.go:391`) becomes test-only. Then
  `composedrecord.go`, `composedReady`, `opt.structures` and the `!composedCPU` branches (`encdev.go:74,85`, `vision.go:1148`, `embed.go:115`,
  `decide.go:246`, `forward.go:1172`, `cpuserve.go:632`) are dead. Without this every model's CPU blocks stay on main's runner.

## Edge models after the flip

| model or class | reason | part | runs on |
|---|---|---|---|
| DeepSeek-V4 | UncoveredSelect (Compressed, Streams behind it) | every block; head Streams | tier ds4 kernels; `ds4Block` on the CPU |
| Qwen3.8-Flash-Next | UncoveredNGram; UncoveredSelect with a tower | n-gram block (attention blocks with a tower); head Streams | q4e, ngram, qsa paths |
| Kimi-K3, Gemma 3n text | UncoveredStreams | head only | main's head (card and CPU) |
| Gemma 3n with its tower | UncoveredEmbedRows (`hardEmbd`); entry EmbedRows; projector Unbound | input rows, entry, projector | host |
| d1-omni | UncoveredNonCausal; no front-end statement for audio | LFM2 encoder trunk; FastConformer, scorer | `emitCentred`; `conv.go` ConvLayers; `audio.go` |
| bert, nomic-bert, qwen3-embedding, EmbeddingGemma | UncoveredPooling (`poolGraph` is stated, the verdict reads `headWhy`); token types EmbedRows | readout, input rows | main's pool and rows |
| plain ModernBERT (UNSURE) | UncoveredNonCausal (rotary at its own bases) | blocks | main |
| glm4v, kimivl, phi4, lfm2 towers; qwen3vl, glm4v, hunyuanvl, minicpmv, pixtral projectors | entry EmbedRows; projector Unbound | tower entry; projector | host: `entryRows`, `project*`, `resample`, `projectGLM`, `projectHunyuan` |
| Laya, Clef, d1-omni readouts | DeclinePick; no composed readout | label pick | `kevhead.go`, `dechead.go`, `pick.go` |
| any model with a prediction block, under `-spec mtp/auto` (opt-in) | no graph past NLayer | draft block | CPU `draftRows`, mla, moe |
| stories260K, tiny-qwen3moe | rows the tier's 32-wide int8 blocks do not divide | all blocks | the CPU |
| containers with no router score or per-expert tensors | UncoveredRouterScore, UncoveredExperts | mixture blocks | main |

**Conditions any model reaches:** a mixture whose banks do not fit the card (auto-stream puts experts on the host, DeclineExpertsHost; cached
streaming declines whole banks and every prompt chunk, DeclineExperts/ExpertsRows); history evicted to the host (DeclineStream); DeclineNoRoom;
DeclineRecurrent; every prompt chunk's rotary tables (`tableServes` is not asked at R>1, ragged or prompt steps) and Gemma 4's decode tables
(DeclineNoTable); the ragged argmax; and on the CPU, `newStateRange` provisioning main's kernels for every State (`KVWidthPaysOff` times main's
attention to pick the composed KV width). Opt-in: q8_0 KV, head-major KV, SharedStore pages on the CPU, `JITLLM_GPU_PERSIST`.

## M2: deletion commits, in order

1. **Pin the flip** (both switches above). Rebase the arms that run `Composed=0` or a partial set onto composed against the reference: `composedArms`,
   `TestComposedStepPromptRunsPrefillAttention`, `TestComposedHeadServesWhereTheBlocksDoNot`.
2. **Branches dead under ComposedStep.** `layersSession`'s `!asked`/`!asks` (PartsUnasked) block arms, the GPU sites of `partPrefill`,
   `promptDeclines`' `!asks` case, `promptHeld`'s `ComposedPrompt == 0` test. Decline reasons with no firing site or an unreachable one:
   DeclineHistoryWindow, DeclineBatch, DeclineISA (`HostRefusal` refuses first), DeclineBidir (already UncoveredNonCausal), DeclineCaller with its
   sources (`TierBlocks`, `RopeTableHost`). DeclinePrefill stays (the CPU's pool over several chunks).
3. **Main's embed gather** (any ComposedEmbed decline returns false and the host writes x): `embed.go` `EmbedRows`' non-composed half, `embedKernel`,
   `gatherSession`, `materializeLocked`'s gather half, `embIDs`; the gather branches in `layersSession` (the `gatherN` setup and its launch);
   `scratch.go`'s `headEmbeds` call; fields `embK`, `embIds`, `embOut`, `headEmbeds` (`layer.go`, `devsess.go:662`, `sesscap.go:26`);
   `kernels/getrows.go`.
4. **Probe and caller knobs that route to main:** `Config.TierBlocks`, `ComposedWidths`, `HeadNormRepeat`, `HeadMatvecRepeat`, `WithSkip`/`partSkip`,
   `shadow.go` with `ComposedShadow`/`WithComposedShadow`. Also `Config.KVReserve` and `devKVReserve`, which nothing reads (`initKVCap` takes
   `MaxSeq`): `relocate_test.go`'s `onePage` sets it and tests nothing, so it is fixed in the same commit (RULE 11b).
5. **The persistent decode path** (opt-in and reached by no default; it drops a feature, so decide before landing): `persist.go`, `persist_hook.go`,
   the launcher's persist hooks (`launch.go`), `persistSession` (`devsess.go`), `WithPersist`, `WithPersistDeps`, `Config.Persist*`,
   `gpuopts.go:193-200`, `cmd/jitllm/verify.go`'s persist arms, `engine/model/persist*_test.go`; then `ir.Persist`, `ir.PersistDeps` and the Dep ops'
   lowering (`composer/ir/persist.go`, `depsync.go`, `composer/ptx/dep.go` and the spirv/msl cases) where grep finds no other caller.
6. **Dead helpers:** `kernels.FieldsOf`. UNSURE: `kernels/unpack.go` and `arena.go` run only for `pre == nil` (`tier.matVec` behind `mvPays`,
   `PrewarmLayer`'s arena), unreachable for a `.jlm` today.

## What stays, what reaches it, what frees it

| main code | reached by | freed by |
|---|---|---|
| tier block runner: `layersSession` after the composed `continue`, its closures (`lamv`..`addRes`), `emitMLA`, `volta70Attn`, `mla70Attn`, `voltaBlock`, `flashTileAttn`, `spanKernel`, `batchMV`..`batchedMVs`; `pagedattn.go`, `pagedstream.go`, `flash.go`, `addnorm.go`, `actin.go`, `gateup2.go`, `qkvgemm.go`; kernels attn*, flash*, gemm*, mma*, `MatVec*`, fuse, decodeplan, elem, mergerun | Stream, NoRoom, experts, Recurrent, NoGraph | composed streamed history and host/paged experts at R>1; V4 and n-gram graphs |
| prompt chunks: `pagedprefill.go`, `raggedprefill.go`, `moegroup.go`, `mlabatch.go`, `declineTwins`; kernels pagedprefill*, pagedaccmma, pagedstaged, moe*, gemmint8 | declined prompt blocks; every prompt chunk of a streamed mixture | as above |
| `streamBank` host paths (`runHost`..`home`), `streamtune.go`'s host half; CPU `hostExperts` | ExpertsHost (the auto-stream default), Llama 4's top-1 | composed host experts |
| recurrent emitters (`emitLinear`, `emitSSD`, `emitShortConv`, `emitMamba1`, `emitCentred`); kernels recurrent, ssd, centredconv | DeclineRecurrent; d1-omni | composed state seats; non-causal recurrent graphs |
| family code: `k3.go`, `q4e.go`, `qsa.go`, `ds4.go`, `indexer.go`, `msa.go`, `altup.go`, `gemma4.go`'s MoE, the `clampB` sites; kernels k3, q4e, ds4, msa, indexer, altup, ple, xielu, sparsemixer | V4, qwen4exp (NoGraph); every family through the conditions above | the two graphs plus the conditions |
| head, table, sample arms (`layersSession` 10327-10524, `ropeTable`, `initRopeTable`, `initRopeAxes`, `ds4Tables`, `argmaxK`, `ragArgmax`, `sample.go`, `pick.go`'s main half) | prompt tables, Gemma 4 NoTable, ragged argmax, Streams/Pooling heads, Pick | composed prompt tables and ragged argmax |
| towers' non-causal branches, `visionattn.go`, phases; `conv.go`; `kevhead.go`, `dechead.go`; kernels conv, conformer, dechead | towerServes declines; MobileNet, FastConformer; readouts | composed conv towers and readouts |
| `declineReason` and the `*WhyNot` (layer.go 2698-2990); tuners `gemmtune.go`, `actintune.go`, `split.go`, `recipe.go`, `mvPays`; main-only knobs and Config fields | main's kernels while any run | the runner |
| CPU: `forward()` body, `rowsHost`, `hostRows`, `fuse.go`; `ds4*.go`, `q4e.go`, `ngram.go`, `qsa.go`, `mla.go`, `moe.go`; head and embed fallbacks (`State.embed` has no composed CPU arm); vision post-LN, deepstack, projectors, `mnFinish`, audio, encoders, `lfm2enc.go`, decision heads | V4, n-gram, MTP, old containers, CPU declines; always-main pieces | graphs for MTP, n-gram and V4 selection; refusing old containers; composed CPU embed and non-text |
| CPU bodies only covered models use: `altup.go` (`altPredict`, `altCorrect`, `laurel`, `laurelJoin`, `gaussTopK`), `k3.go` (`k3AttnIn`, `k3FFNIn`, `k3Resid`, `k3MoE`, `k3MoEBatch`), `ple.go` (`pleApply`, `pleRows`), `delta.go` mixers, `linrows.go`, `regions.go`; nn `delta.go`, `ssd.go`, `GaussTopK32JIT`; engine/device/cpu `delta*`, `conv*`, `sse_hybrid.go` and their emitter fields | CPU declines (q8 KV, head-major, PageOut, Cut/NoWeight, Sessions, rows-form errors); `newStateRange` provisioning | composing or refusing each CPU decline; a composed KV-width rule |

**Before the runner can go:** move out of `layer.go` what the composed path uses (`prepLayer`, `layersSession`'s served dispatch, `recDesc`
staging, `pickStage`, `pickLanes`, `initKVCap`, the streamed fill); keep kernels' `Quant` and layouts, `pack.go`, `quantof.go`, `scstream.go`,
`actkind.go`, `ropetab_const.go`, the off-heap and release files, `PRow*`, `PagedGatherPages`, `RecDescWords`, `SampleSlices`, `divrn.go`; move the
backend probes (`gridprobe.go`, `subgroup.go`, `tile.go`, `MMAProbe`, `MMAVoltaProbe`); rebuild `composer/lowertest` on the composer's forms; rebase
the composer tests that use main's kernels as arms (`MatVec`, `GemmVolta*`, `GemmTile`, `SoftmaxRows`, `Reduce`, `ActF16T`) and `choose.go`'s probe.

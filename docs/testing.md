# How tests find models

Model files (GGUFs, `.jlm` containers, HuggingFace directories) are too big to
commit, all but one: `internal/testmodels/fetch.sh` downloads the rest (see
"Getting the models" below). Every test that needs one finds it through
`internal/testmodels`, which reads one variable:

| variable | meaning | unset |
|---|---|---|
| `JITLLM_MODELS` | the model directory | `models/` at the repository root |
| `JITLLM_MODEL_FREE` | `1`: a missing model skips instead of failing (CI only, see "What CI runs") | a missing model fails |
| `JITLLM_RECIPE` | `srcgate.TestRecipesRun` runs only the recipe of this name (`docs/recipes/NAME.md`) | every recipe |
| `JITLLM_RECIPE_PYTHON` | the interpreter the recipes' Python blocks run with; one that cannot `import openai` skips them by name | `python3` |
| `JITLLM_RECIPE_NODE_MODULES` | a `node_modules` holding `openai`, linked into each recipe's directory for its JavaScript blocks | node's own resolution |

The fallback is located by walking up from the test's working directory to the
`go.mod` that declares `github.com/jitllm/jitllm` -- so it is the same
directory from every package, including the nested `ui/` and `server/` modules.
`models/` may be a symlink to a model disk; otherwise point the variable at
the models instead of faking the link:

    JITLLM_MODELS=/mnt/models/jitllm ./scripts/cap 16G -- go test ./engine/model -run TestGreedyMatchesLlamaCpp -count=1

`JITLLM_MODELS` is the same variable `jitllm fetch` uses for where a download
lands, so one setting serves both. A test binary shipped to another host has
no `go.mod` above it, so set the variable in the remote run's environment
(the arm64 and Xeon runs were driven by local helpers, not in the repo).

## A missing model is loud

A gate whose model is absent says so and names the path it looked at:

    MODEL MISSING: /nonexistent/tinyllama-1.1b-q3_K_M.gguf (set JITLLM_MODELS to the model directory) -- this gate proved nothing

Some gates FAIL rather than skip on a missing model (RULE 11: a missing
artefact is a task). They say so through one function,
`testmodels.Missing`, which fails the test -- unless `JITLLM_MODEL_FREE=1`,
where it skips with the same message. Only CI's model-free run
(`.github/scripts/test-model-free.sh`) sets it; a box meant to hold the
models leaves it unset. Neither a skip nor a model-free run is a pass; read
the skip lines of a run.

A new test that needs a model reports its absence through
`testmodels.Missing` (or skips, as `jlmOf` does), never a bare `t.Fatal`: a
bare failure is red in the model-free run, and CI does not keep a list of
names to excuse it.

## What CI runs

`.github/workflows/ci.yml` has four jobs, every one with `CGO_ENABLED=0` as
the release builds:

- **build** cross-compiles the five modules (root, server, common, ui, tui) for every release target from
  one Linux runner.
- **check**: gofmt, `go mod tidy`, buf lint and format, and that
  `server/gen` is what `buf generate` writes.
- **test**, on linux/amd64, linux/arm64, darwin/arm64 and windows/amd64:
  `scripts/vet` over the engine and `server/`, then
  `.github/scripts/test-model-free.sh` (the program is `scripts/modelfree`,
  Go so that one runner serves all four hosts). On Windows it first builds
  the five modules natively, links `jitllm.exe` and `jitllmd.exe`, and runs
  `jitllm hardware`.
- **ui**, on linux/amd64, darwin/arm64 and windows/amd64: builds the `ui`
  module and the `jitllm-ui` binary, `scripts/vet ui`, and the mock-engine
  tests, `./mock ./stage ./cmd/shots` -- `stage` plays every scenario of
  package `mock` through the mounted window and checks each frame, offscreen,
  with no display, GPU or model. On Linux it also starts `jitllm-ui -mock`
  under Xvfb with Mesa's software Vulkan (lavapipe), holds that it is still
  running after 20 seconds, and keeps a screenshot as the run's
  `jitllm-ui-mock` artifact (`.github/scripts/ui-smoke.sh`). Under Xvfb the
  window presents no frame -- the screenshot is black while vkcube on the same
  display draws -- so the picture is reported, not gated.

The model-free run is every package of the five modules but `./dev/bench`
(its perf floors want a quiet box, RULE 4), `./ui/stage` (the ui job's) and,
on macOS, `./backend` (the runner's paravirtual Metal device allows
512-thread pipelines, and the backend's gates force 1024-thread splits),
`engine/model` and `common/engine` included, plus the SSE tier under
`-tags jitllmtest` on amd64, Windows included. It sets `JITLLM_MODEL_FREE=1`
and points `JITLLM_MODELS` at a fresh directory holding only the committed
`stories260K.gguf` and the container this build converts from it, so every
gate that runs on stories260K runs, on the CPU; everything else skips with its
MODEL MISSING line. No model is downloaded, and the Linux and Windows runners
have no GPU. The job summary counts each run and lists every skip with its
reason. Run it locally the same way, with the GPUs hidden:

    CUDA_VISIBLE_DEVICES= VK_DRIVER_FILES=/nonexistent.json ./scripts/cap 8G -- .github/scripts/test-model-free.sh

On Windows the x86 encoding gate needs GNU objdump, which the job puts on the
PATH from the image's MinGW or Strawberry Perl. Windows-only test code uses
build constraints or portable calls: the cross-process GPU test lock is
`internal/testlock` (flock on Unix, LockFileEx on Windows), and
`GOOS=windows ./scripts/vet` type-checks every test file for Windows from a
Linux box, with no allowlist: a vet finding fails the job.

## Getting the models

`internal/testmodels/fetch.sh` downloads every model file a test reads into
the model directory (`$JITLLM_MODELS`, or `models/`). Its table -- name, repo,
file, bytes -- is the one record of where each test model comes from, and
every entry was checked byte for byte against the copy the goldens were taken
on:

    internal/testmodels/fetch.sh --list          # the table, sizes, what is here
    internal/testmodels/fetch.sh                 # the default set, ~40 GB
    internal/testmodels/fetch.sh llama gemma     # only names containing a word
    internal/testmodels/fetch.sh --large scout   # an opt-in large model

A file already present at the table's size is skipped, and a partial download
resumes. The default set is what a normal `go test ./...` reads: GGUFs and
their vision projectors, the embedding models under `embed/`, and the
HuggingFace directories the safetensors converter is gated on
(`tiny-random-*`, `qwen3`, `SmolLM2-360M-Instruct`, ...). `--large` adds the
files over ~10 GB and the ones only an override variable reaches (the 80B,
Kimi-Linear, Moonlight, Mixtral, Llama-4-Scout, gpt-oss-20b, DeepSeek-V2-Lite,
phi-4, falcon-7b). Two gates fail rather than skip without `--large`:
`engine/model` `TestGPTOSSMatchesLlamaCppIntermediates` and
`tok` `TestDecodeChatRendersHarmonyChannels` both need gpt-oss-20b.
`HF_TOKEN` is sent to huggingface.co when set; no entry needs it.

Committed rather than fetched: `testdata/models/stories260K.gguf` (1.1 MB), so
`go test ./convert/gguf` runs on a fresh checkout with no network. The script
copies it into the model directory, where other tests look for it.

The script fetches no `.jlm`. A test that names a container reads the one
converted from the GGUF of the same stem: the `engine/model` gates write it
beside the GGUF on first use, or convert it yourself with
`go run ./cmd/jitllm convert <dir>/<name>.gguf`.

Made rather than fetched -- the script's header has the commands:

- the `synth-*` fixtures: `scripts/hfgold.py`, `scripts/c6gold.py`,
  `scripts/llama4gold.py` and `scripts/gptossgold.py` build them with the
  reference's own classes, in the CPU torch/transformers venv each script's
  header describes (`hfgold.py` starts several from the downloaded tiny-random
  directories);
- `tiny-qwen3moe-f32.gguf`: `scripts/tiny-moe-gguf.py`, from `tiny-qwen3moe-hf/`;
- `split-test/`: `convert.TestSplitGGUFConvertsLikeTheWholeFile` needs
  tinyllama split by llama.cpp's own tool, in the model directory:

      llama-gguf-split --split --split-max-tensors 60 tinyllama-1.1b-q3_K_M.gguf split-test/tl

## Choosing WHICH model

`JITLLM_MODELS` says WHERE. The variables below say WHICH, per test. A value
that is a bare file name is looked up in the model directory; a value with a
`/` in it is a path and is used as written. "Substring" variables filter a
sweep by name and are never resolved.

| variable | tests | value | unset |
|---|---|---|---|
| `JITLLM_MODEL` | `model.TestFusionCeiling` | one model file | every model |
| | `model.TestForwardEmbdMatchesForward`, `TestPrefillMixedMatchesPrefill` | one model file | every language model under 2 GiB |
| | `model.TestPrefillRegionProbe` | one model file | `tinyllama-1.1b-q3_K_M.gguf` |
| | `model.TestDeviceDecodeRate`, `TestSamplerPerf` | one model file (a container for the sampler) | test skips |
| `JITLLM_BENCH_MODEL` | `model` gates that call `benchModel` (e.g. `TestPrefillMatchesForward`) | one model file | the smallest quantized language model in the directory |
| | `model.TestABKQuantAccsEndToEnd` | a k-quant model file | test skips |
| | `model.BenchmarkGemmaDecode` | one model file | `gemma-2b.gguf` |
| `JITLLM_ROUTE_MODEL` | `model.TestRoutingLocality`, `TestHotnessIsAProbability` | a mixture-of-experts model file | test skips |
| `JITLLM_MLA_MODEL` | `model.TestMLAPrefillMatchesDecode`, `TestMLABatchMatchesDecode` | a real MLA model (name or path) | the synth-deepseek fixtures |
| `JITLLM_DEV_MODEL` | `jit/gpu/tier.TestContainerAgreesHostAndDevice` | one file name in the directory | first present of tinyllama, Llama-3.2-1B, Qwen3-MoE, gemma-2b, olmoe |
| `JITLLM_LABELCOST_MODEL` | `model.TestRegionLabelCost` | file name in the directory | `tinyllama-1.1b-q3_K_M.gguf` |
| `JITLLM_LABELRATE_MODEL` | `model.TestRegionLabelRate` | file name in the directory | `tinyllama-1.1b-q3_K_M.gguf` |
| `JITLLM_KVPERF_MODEL` | `model.TestPagedKVCost` | file name in the directory | `tinyllama-1.1b-q3_K_M.gguf` |
| `JITLLM_KVPREFIX_MODEL` | `model.TestPrefixCacheSpeedup` | file name in the directory | `tinyllama-1.1b-q3_K_M.gguf` |
| `JITLLM_KVPREC_MODEL` | `model.TestKVPrecisionCost` | exact base name | every model under 2 GiB |
| `JITLLM_GREEDY_MODEL` | `model.TestGreedyMatchesLlamaCpp` | substring | every model; over 2 GiB only with `JITLLM_SLOW=1` |
| `JITLLM_GREEDY_PAIR` | `model.TestGreedyMatchesLlamaCpp` | `<gguf>=<container>`, each name or path | GGUFs converted by the test |
| `JITLLM_KVSWEEP_MODEL` | `model.TestPagedKVIsTokenIdenticalOnEveryModel` | substring | every model under 2 GiB |
| `JITLLM_AUDIT_MODEL` | `model.TestEveryModelRunsGenerated` | substring, any size | every model under 2 GiB (`JITLLM_SLOW=1`: all) in the directory, one level down, and the safetensors fixtures with an `hf` golden |
| `JITLLM_TOKMODEL` | `model.TestTokenizerCost` | substring | its fixed list |
| `JITLLM_ROLE_MODEL` | `convert.TestEveryTensorHasARole` | substring | every GGUF |
| `JITLLM_ST_MODEL` | `convert.TestSafetensorsArchCoverage` | substring | every HuggingFace directory |
| `JITLLM_C6_HF` | `convert.TestDenseSafetensorsMatchesItsGGUF` | the directory `scripts/c6gold.py` writes its HuggingFace fixtures to (its `CV_HF`) | `$JITLLM_MODELS/hf` |
| `JITLLM_TPL_MODEL` | `jinja.TestGGUFChatTemplatesRender` | substring | every GGUF |
| `JITLLM_SERVER_MODEL` | `server.TestEndToEndAgainstARealContainer` | a `.jlm` container | test skips |
| `JITLLM_SERVER_BATCH_MODELS` | `server.TestBatch*` | comma-separated `.jlm` names or paths, each run beside the default | `stories15M-q8_0.jlm` only |
| `JITLLM_TOOL_MODELS` | `server.TestToolChoiceForcesACall` | comma-separated `.jlm` names, in place of the default | `Qwen3-0.6B-Q8_0.jlm` and `Llama-3.2-1B-Instruct-Q4_K_M.jlm`; also run with Qwen3.5-0.8B-f16, DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M and gpt-oss-20b-Q4_K_M |
| `JITLLM_STEP_MIXTURE` | `model.TestStepAcrossSessionsMixture*` | comma-separated names or paths | Qwen3-MoE-4x0.6B and olmoe; a card that cannot place every block skips |
| `JITLLM_ALLOC_MODELS` | `model.TestDecodeDoesNotAllocate` | comma-separated names or paths, each run beside the default | Llama-3.2-1B, olmoe and Qwen3.5-0.8B only |
| `JITLLM_SLOW` | several `model` sweeps | `1` admits models over 2 GiB | small models only |
| `JITLLM_MOVE_REPLY_MODEL` | `server.TestAMoveLandsBetweenTwoTokensOfAReply` | a `.jlm` whose reply outlasts its move's preparation | `Llama-3.2-1B-Instruct-Q4_K_M.jlm` |
| `JITLLM_MOVE_PAUSE_MODEL` | `model.TestMovePause`, a measurement: the pause of a move between two devices, direct against through the host | a model name; opts in | the test skips |
| `JITLLM_POSTNORM_MODELS` | `model.TestPostNormDeviceErrorIsAmplification` | comma-separated GGUF names in the directory, any model whose last block has a post-FFN norm | OLMo 2 1B and EXAONE 4.0 1.2B |
| `JITLLM_LLAMACPP_CPU` | `model.TestPostNormDeviceErrorIsAmplification` | a directory holding a CPU-only `llama-perplexity` and its libraries (a CUDA library beside it is loaded and used) | the test skips naming it |

Every other `JITLLM_*` variable a test reads is a knob or an opt-in for a
measurement, not a model choice.

## Choosing WHICH device

`JITLLM_TEST_DEVICES` is the one device selection for every test binary in the
repository that opens a device: a comma-separated list of `cuda:N`, `vulkan:N`,
`hip:N` and `metal` (a bare backend name is refused, since "every Vulkan
device" is what a selection exists to say no to).

    JITLLM_TEST_DEVICES=cuda:2,vulkan:2 ./scripts/cap 8G -- go test ./engine/model -run TestSchedulerOnDevice

Unset, every test opens what it always opened. Set, no test opens any other
device. Ordinals are the backend's own: a CUDA ordinal is the driver's (under
`CUDA_VISIBLE_DEVICES=2` the card is `cuda:0`; without it, `cuda:2`), a Vulkan
index is the loader's. Vulkan ignores `CUDA_VISIBLE_DEVICES`, which is why
the selection exists: on an eight-card box a gate runs on the card it locked
(`flock ~/gpu<N>.lock`), and the selection is what keeps its Vulkan opens there.

How it holds (`internal/testdevices`, `backend.Restrict`):

- Every test binary that opens a device installs the selection
  (`testdevices.Install`, in a `TestMain`; `TestEveryPackageThatOpensADeviceInstallsTheSelection`
  fails for a package that does not). The backend then asks it before every
  device open, so a device outside it is not opened at all.
- A search for devices (`backend.Open`, a bare `gpu`, `auto`, `all`, `cuda`,
  `vulkan`, a loop over ordinals marked `Opts.Enumerating`) is narrowed to the
  selection and the default pick of a backend is its first selected device.
- A test that NAMES a device outside the selection (`vulkan:0` against
  `cuda:2`) is refused and recorded, and the binary's `TestMain` fails
  (`testdevices.Fail`). `TestARefusalIsAViolationOnlyWhenTheTestNamedTheDevice`
  is the violation run, and `TestOnlyTheSelectedDevicesOpen` asks every ordinal
  the host has to open and demands that only the selected ones did.
- A gate that needs more devices than are selected (blocks moved between two
  cards, four-device splits) skips naming the selection; one whose devices are
  part of what it asserts skips unless they are selected
  (`testdevices.Require`). `cuda:2,vulkan:2` is the pair for the move gates.

One thing a selection cannot do: with a Vulkan device selected, the loader's
enumeration makes the NVIDIA driver attach a graphics context to card 0 for
as long as the process lives, whichever card is selected (measured with
`nvidia-smi`: an enumeration-only process shows a `G` entry on GPU 0). A
selection with no Vulkan device enumerates nothing and touches no other card;
a selection with one (`vulkan:N`) pays that context on card 0. CUDA opens are
exact.

A test takes its device list from the helpers, never from a literal:
`Specs(defaults...)` for a sweep, `One(def)` for one device, `Spec(def)` for a
list a tier places blocks over, `Take(t, n, def)` for n distinct devices,
`Like(t, spec)` for one backend's selected device, `Ordinal(t, backend, def)`,
`Has(t, backend, ord)` for a loop over ordinals and `Within(candidates)`.

The GPU test lock (`internal/testlock`) follows the selection: unset, one file
for the whole box; set, one file per selected device, taken in sorted order, so
gates on different cards do not wait for each other and two on one card do.
`cuda:2` and `vulkan:2` are two names for one card and two lock files; two runs
that share a card under different names are not serialised, so give them the
same list.

The per-test variables this replaces are gone: `JITLLM_STEP_DEVICES`,
`JITLLM_MOVE_DEVICES`, `JITLLM_MIX_DEVICES`, `JITLLM_SPEC_DEVICES`,
`JITLLM_FOUR_DEVICES`, `JITLLM_PAGED_VK`, `JITLLM_HYBRID_DEVICES`,
`JITLLM_MOE_DEVICES`, `JITLLM_REF_DEVICES`, `JITLLM_SPLIT_DEVICE` and
`JITLLM_AB_DEVICE`. No script read them, so there is no alias. A stale one is
ignored, not an error.

`backend`'s tests open their devices through `backend.Open`, which
takes a discrete card before an integrated one -- so on a laptop with both, the
iGPU is never tested unless it is named. The test binary (not the package,
which reads no environment) maps these onto the `backend.Opts` `Open` uses:

| variable | effect | unset |
|---|---|---|
| `JITLLM_VK_DEVICE` | the Vulkan device every `backend.Open` takes, by index or name substring (`1` is the Iris Xe beside the RTX here) | the default rule's pick |
| `JITLLM_ROCM` | the ROCm library directory the HIP backend loads from (`backend`, `lowertest`, `hip` and `server` tests; `jitllm` and `jitllmd` too); with no AMD card, comgr alone runs the offline code object gate (docs/design/amd-hip.md) | the default search: the loader, `/opt/rocm/lib`, `/opt/rocm-*/lib` |
| `JITLLM_VK_MAXGROUPS` | workgroups per Vulkan dispatch, below the device's own `maxComputeWorkGroupCount`; `7` sends nearly every launch in the suite through a split dispatch | the device's own |
| `JITLLM_VK_VALIDATION` | the prefix the Khronos validation layer is installed under, for `backend.TestSplitLaunchesAreValidVulkanUsage`; `apt-get download vulkan-validationlayers && dpkg-deb -x` the `.deb` into DIR and pass `DIR/usr` | `/usr`, else the test skips |

    JITLLM_VK_DEVICE=1 JITLLM_VK_MAXGROUPS=7 ./scripts/cap 8G -- go test ./backend/ -count=1

The `model` device gates take the same limit, as `tier.WithVulkanMaxGroups`:
`JITLLM_VK_MAXGROUPS` reaches every tier the step gates
(`TestStepAcrossSessions*`) and the
default-device gates open (`TestSchedulerOnDevice*`, `TestSlidingWindowRunsOnTheDevice`
and every other caller of `swaTier`). `JITLLM_TEST_DEVICES` names their device:

    JITLLM_TEST_DEVICES=vulkan:1 JITLLM_VK_MAXGROUPS=7 ./scripts/cap 8G -- go test ./engine/model -run 'TestSchedulerOnDevice' -count=1

`cmd/jitllm` reads `JITLLM_VK_DEVICE` and `JITLLM_VK_MAXGROUPS` too, as
`tier.WithVulkanDevice` and `tier.WithVulkanMaxGroups`.

## Running the device gates composed

A test binary built with `-tags jitllmcomposed` opens every tier with the
composer serving the blocks and the head (`Config.Composed`) unless a gate
sets its own, so every device gate runs the composed path on what the front
end states and the tier's own kernels on the rest. The relocation, move and
session gates carry a composed arm in every build as well, on the models the
front end states whole, held to the composer having served
(`composedRan`: part steps served, none formed at a step, no decline outside
the arm's stated reasons):

    JITLLM_TEST_DEVICES=cuda:2,vulkan:2 ./scripts/cap 8G -- go test -tags jitllmcomposed ./engine/model -run TestBlocksMoveBetweenDevices

In a composed build the classic-kit and dense (`TestC6OnEveryDevice`), hybrid
attention (`TestHybridAttentionBlockOnDevice`) and BitNet
(`TestBitNetOnEveryDevice`) device gates hold the composer to having run
exactly the steps times the blocks the front end states, emitted or replayed
from a recording (`composedCovers`, on the counter `composedBlocksRan` uses),
and to declining every other block step for having no graph and no other
reason. A model the front end states in part runs once more with every step
issued, which counts those declines exactly.
`TestHybridOutGateViolationsOnDevice` runs the hybrid's attention block
composed with its graph rewritten (the gate left out, the halves of each
head's double-width query swapped or read whole), which only the composer
reads, and holds the hybrid gate's own verdict on it to a failure: past the
block bound with the shallow control past it too.

## The shadow arm

`tier.WithComposedShadow()` (`Config.ComposedShadow`, beside a `Config.Composed`
that asks for blocks) runs the composer's form of each block it would serve
beside the tier's own kernels, at every one-row step, and a reduction the
composer forms from generic ops (`graph.Drift`) writes the block's worst drift
from the tier's output into a slot of its own, read back as
`Stats().ShadowWorst` and `ShadowSteps`. The tier's kernels serve, so the
option changes no answer. `TestComposedShadowHoldsEveryBlock` runs the four
checkpoints in `shadowModels` on every device selected, with and without the
captured graph, and holds them to the tier's logits bit for bit, to the spread
of the tier's backends at every depth, and to flagging a rewrite of one block's
graph at that block and no other:

    JITLLM_TEST_DEVICES=cuda:2,vulkan:2 ./scripts/cap 8G -- go test -tags jitllmcomposed ./engine/model -run TestComposedShadowHoldsEveryBlock

## Reference scripts

The scripts under `scripts/` that build fixtures and record goldens
(`*gold.py`, `*real.py`, the paging and three-way harnesses) find everything
through environment variables, read by `scripts/refenv.py` for the Python
ones. None has a default that names a machine; a script stops naming the
variable it needs and did not find.

| variable | what it is | read by |
|---|---|---|
| `JITLLM_HF_PY` | the python of a venv with CPU torch, transformers, tokenizers and safetensors (`scripts/hfgold.py` has the install lines); every `*gold.py` / `*real.py` runs under it: `$JITLLM_HF_PY scripts/c6gold.py` | how the scripts are run |
| `JITLLM_MODELS` | the model directory: where GGUFs, fixtures and checkpoints are read and written | every script |
| `LCPP_SRC` | a llama.cpp source checkout (`convert_hf_to_gguf.py`, `gguf-py`, `tools/mtmd/test-1.jpeg`) | the fixture builders |
| `JITLLM_LCPP` | a llama.cpp build directory (`llama-completion`, `llama-tokenize`, `libggml-base.so`) | `gold.py`, `rungold.py`, `tokgold.py`, `embedgold.py`, `vs-llamacpp*.sh` |
| `JITLLM_LLAMACPP_CPU` | a CPU-only llama.cpp build directory (`llama-mtmd-cli`, `llama-perplexity`) | `vlmgold.py`, `qwenvlmtmd.py`, and the post-norm gate above |
| `CV_TOK`, `CV_HF` | the family tokenizers the fixtures copy, and where their HuggingFace directories are written | default `$JITLLM_MODELS/tok` and `$JITLLM_MODELS/hf` |
| `BAILING_PY`, `BAILING_REF` | the python of a second venv with transformers 4.52.3, and inclusionAI's Ling 2.0 remote code | `moegold.py`'s `synth-bailingmoe2` only |
| `L4_TOK` | the directory holding Llama 4's `tokenizer.json` | `llama4gold.py`, `llama4visiongold.py` |

The decision models' goldens: `scripts/decisiongold.sh` (llama-server's
`/v1/systemone` under `JITLLM_LCPP` on `$JITLLM_MODELS/{d1,lev,laya/gguf,clef}`; a
llama.cpp with the clef arch for Clef), `scripts/clefgold.py` (Clef's own code:
`fixture` builds the random fixture into `$JITLLM_MODELS/clef/fixture` with
llama.cpp's converter under `LCPP_SRC`, `real` runs the release's safetensors) and
`scripts/layagold.py` (Laya's own code: `real` on the checkpoint, `fixture`
builds the random fixture into `$JITLLM_MODELS/laya/fixture` with llama.cpp's
converter); `docs/design/decision-models.md` names the files.

A script-specific override (`QVL_TOK`, `PIX_HF`, `PHI4V_TOK` and the like) is
named in that script's docstring and defaults to a path under `$JITLLM_MODELS`.

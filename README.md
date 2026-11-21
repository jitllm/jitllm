<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/jitllm.svg"><img src="docs/assets/jitllm-light.svg" alt="jitllm" width="318"></picture></p>

<h3 align="center">Your models. Your hardware. Full speed.</h3>

<p align="center"><b>An operating system for LLM inference.</b><br>
JIT kernels that match your hardware · paging so big models still run · auto-tuning &amp; layer orchestration</p>


<p align="center">
<a href="https://github.com/jitllm/jitllm/releases/latest"><img src="https://img.shields.io/github/v/release/jitllm/jitllm" alt="Latest release"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 license"></a>
</p>

<p align="center">
<img src="https://img.shields.io/badge/CUDA-supported-76B900?logo=nvidia&logoColor=white" alt="CUDA supported">
<img src="https://img.shields.io/badge/Vulkan-supported-AC162C?logo=vulkan&logoColor=white" alt="Vulkan supported">
<img src="https://img.shields.io/badge/Metal-Apple%20Silicon-000000?logo=apple&logoColor=white" alt="Metal on Apple Silicon">
<img src="https://img.shields.io/badge/ROCm-untested-ED1C24?logo=amd&logoColor=white" alt="ROCm untested">
<img src="https://img.shields.io/badge/CPU-x86%20%7C%20Arm-0071C5?logo=intel&logoColor=white" alt="CPU x86 and Arm">
<br>
<img src="https://img.shields.io/badge/Linux-x86--64%20%7C%20arm64-FCC624?logo=linux&logoColor=black" alt="Linux x86-64 and arm64">
<img src="https://img.shields.io/badge/macOS-arm64%20%7C%20x86--64-000000?logo=apple&logoColor=white" alt="macOS arm64 and x86-64">
<img src="https://img.shields.io/badge/Windows-x86--64%20%7C%20arm64-0078D4" alt="Windows x86-64 and arm64">
</p>


<p align="center">
<a href="https://jitllm.org">jitllm.org</a> ·
<a href="#quick-start">Get started</a> ·
<a href="#performance">Benchmarks</a> ·
<a href="https://jitllm.org/docs/">Docs</a> ·
<a href="https://github.com/jitllm/jitllm/releases/latest">Download</a>
</p>

<p align="center">
<a href="#performance"><img src="docs/assets/readme/perf.png" width="400" alt="Performance: 4.62x llama.cpp's prefill on Qwen3-Next-80B (4x V100); 35 of 35 models decode and 34 of 35 prefill faster than on llama.cpp (V100); up to 4.08x sooner to the first token; up to 5.53x vLLM 0.18.1's batched decode; up to 1.53x llama.cpp's decode on an Apple M4"></a>
<a href="#the-five-principles"><img src="docs/assets/readme/schedule.gif" width="400" alt="A simulation of placement: model blocks page from the .jlm onto a GPU and the CPU, and relocate, KV with them, as a GPU is added, the placement changes and the GPU is removed"></a>
</p>
<p align="center">
<a href="#how-it-works"><img src="docs/assets/readme/jit.gif" width="400" alt="The JIT: the inner loop of a Q4_K matrix-vector product emitted for an AVX2+VNNI CPU, then for CUDA"></a>
<a href="#memory-and-placement"><img src="docs/assets/readme/experts.gif" width="400" alt="Expert pages: each expert of a mixture is its own page, so a token reads only the routed experts that are not resident"></a>
</p>

<p align="center">
<a href="#quick-start">Server</a> ·
<a href="docs/cli.md">CLI</a> ·
<a href="#desktop-app">Desktop</a> ·
<a href="#terminal-app">TUI</a> ·
<a href="docs/embedding-go.md">Embed</a>
</p>

## Why jitllm

Inference engines are bulky and rigid. There's a lot to set up, and even after
you get a model running, getting the most out of your hardware is another job.

I wanted to build something easier to live with: one binary you can drop onto a
machine and start running models. Something that does more of the work of
figuring out the hardware itself.

The inspiration came from operating systems and JIT compilers, two things I've
worked on a lot. An OS detects hardware, uses the right drivers, schedules
processes, and manages memory. A JIT compiler generates code for the machine
it's running on. jitllm brings those ideas together to make inference faster,
easier to run, and more flexible.

That's why I call it an operating system for LLM inference. It detects your CPUs
and GPUs, generates optimized kernels, and orchestrates model placement at
runtime. Models larger than memory page from disk. Layers can move between
devices while keeping their conversation state, so changing placement doesn't
mean starting over.

<a id="the-five-principles"></a>
<a id="how-it-works"></a>
<a id="memory-and-placement"></a>

## The engine

- **Kernels written for the machine in front of it.** Every kernel is emitted
  at load for the model's shapes and weight formats: x86 (SSE, AVX2, AVX-512,
  VNNI) and Arm (NEON, dot-product) machine code on the CPU, PTX for CUDA
  (tensor cores from Volta on), SPIR-V for Vulkan, Metal shaders, and AMDGPU
  for ROCm (not yet tested on AMD hardware). No vendor math library, no
  interpreted fallback.
- **Layers move mid-conversation.** Blocks relocate between the CPU and any
  GPU, or from one GPU to another, carrying their KV pages and recurrent state,
  and the answer stays the same.
- **Models bigger than memory.** Disk, RAM and VRAM are one paged hierarchy:
  weights page in on demand (read with O_DIRECT, never mmap), every expert of a
  mixture is its own page so a token reads only the experts it routes to, and
  KV history pages and streams through the card.
- **Broad coverage, held to the reference.** 57 graphs (65 GGUF architecture names, 30 Hugging Face classes) and 17
  vision projectors: dense, mixture-of-experts, MLA, gated-delta and
  state-space hybrids, vision-language, embeddings and decision models, each
  gated against llama.cpp or the model's own transformers class.
- **No garbage on the hot path.** A warm decode token allocates nothing on the
  Go heap, on the host and every GPU backend; weights and page frames live off
  the heap.
- **Faster tokens, same distribution.** Speculative decoding with a model's own
  prediction head or prompt lookup, an int8 KV cache, prefix caching and
  shared prefixes across sessions.

## The server

One binary, `jitllmd`, many models and many sessions at once:

- **OpenAI, Anthropic and Connect APIs**: chat, completions, messages,
  embeddings, Batches and Files, with streaming.
- **Tool calling in each model's own format** (24 families, from Hermes and
  Llama to gpt-oss Harmony, DeepSeek, Kimi and GLM), with `tool_choice`
  enforced by constrained decoding, and **structured output** from a JSON
  schema.
- **Pictures** through `image_url` and `image` blocks on vision models,
  log-probabilities, `n` choices from one prefill, and decision answers through
  `/v1/systemone`.
- **Continuous batching on the CPU and the GPU**, prompts fed in chunks beside
  running decodes, KV-aware admission, and a **fairness level**
  (`-fairness 0-100`) that time-slices sessions through the card by parking
  them to the host and back, with per-request priority.
- **A control plane**: load and unload models, create, park and resume
  sessions, move blocks between devices (`jitllmd models`, `sessions`,
  `place`, `stats`), and `/metrics` for Prometheus.

## Desktop app

Model downloads, conversion, text and image chat, and a live view of where each
block of the model sits, in one window. Get `jitllm.app` (`.dmg`) on macOS or
the setup or portable `.exe` on Windows from the
[latest release](https://github.com/jitllm/jitllm/releases/latest), or install it
from a terminal and run `jitllm-desktop`:

```sh
curl -fsSL https://jitllm.org/install.sh | sh -s -- desktop
```

```powershell
$env:JITLLM_PROGRAMS = "desktop"; irm https://jitllm.org/install.ps1 | iex
```

<p align="center">
<picture><source media="(prefers-color-scheme: light)" srcset="website/public/shots/chat-light.png"><img src="website/public/shots/chat-dark.png" width="400" alt="The desktop app's Chat screen: a reply with its thinking folded, sampling controls, and the rate and bandwidth of the reply"></picture>
<picture><source media="(prefers-color-scheme: light)" srcset="website/public/shots/machine-light.png"><img src="website/public/shots/machine-dark.png" width="400" alt="The Machine screen: which blocks run on the GPU and which on the CPU, the KV cache, and the pages held in each memory"></picture>
</p>
<p align="center">
<picture><source media="(prefers-color-scheme: light)" srcset="website/public/shots/discover-smartest-light.png"><img src="website/public/shots/discover-smartest-dark.png" width="400" alt="The Discover screen: the catalog ranked for this machine, with an expected rate and whether each model fits"></picture>
<picture><source media="(prefers-color-scheme: light)" srcset="website/public/shots/models-light.png"><img src="website/public/shots/models-dark.png" width="400" alt="The Models screen: the converted models, each with whether it fits on the GPU, fits in memory or pages from disk"></picture>
</p>

## Terminal app

The same screens in a terminal, over SSH too, sharing the desktop app's settings
and chats. The arrow keys move the model's blocks between the CPU and the GPU,
and the conversation keeps its history across the move. Install it and run
`jitllm-tui models/qwen3.jlm`:

```sh
curl -fsSL https://jitllm.org/install.sh | sh -s -- tui
```

```powershell
$env:JITLLM_PROGRAMS = "tui"; irm https://jitllm.org/install.ps1 | iex
```

<p align="center">
<img src="website/public/shots/tui-chat.png" width="400" alt="The terminal app's Chat screen: Qwen3-30B-A3B answering, the engine panel beside it">
<img src="website/public/shots/tui-engine.png" width="400" alt="The terminal app's Engine screen: every block and where it runs, the rate over time, memory and the pager">
</p>

## Install

**macOS:** download `jitllm_<version>_darwin_arm64.dmg` (Apple silicon) or
`_amd64.dmg` (Intel) from the [latest release](https://github.com/jitllm/jitllm/releases/latest),
open it and drag **jitllm** to Applications (or take the signed app alone,
`jitllm_<version>_darwin_<arch>.app.zip`, and unzip it there). The app carries `jitllm` and
`jitllmd`; on first launch it offers to put them on your `PATH`, and its
Settings can start the server at login. With Homebrew:

```sh
brew tap jitllm/jitllm
brew trust jitllm/jitllm
brew install --cask jitllm   # the app, with jitllm and jitllmd on PATH
brew install jitllm          # jitllm and jitllmd only
```

**Windows:** run `jitllm-setup_<version>_windows_amd64.exe` (or `_arm64.exe`)
from the [latest release](https://github.com/jitllm/jitllm/releases/latest).
It installs for your account only, with no administrator prompt: the CLI, the
server and the desktop app into `%LOCALAPPDATA%\Programs\jitllm`, on your
`PATH`, with Start menu shortcuts, an optional "start the server at login",
and an uninstaller under Settings > Apps. With winget (available once
published): `winget install jitllm.jitllm`. To run the desktop app without
installing anything, download the portable
`jitllm-desktop_<version>_windows_amd64.exe` (or `_arm64.exe`) and run it from
anywhere; it shares its settings and chats with an installed copy.

Until releases are signed, macOS and Windows warn the first time:
[packaging/README.md](packaging/README.md#first-launch-of-an-unsigned-build)
says what each warning means and how to open the app.

**Any platform, from a script.** Linux and macOS:

```sh
curl -fsSL https://jitllm.org/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://jitllm.org/install.ps1 | iex
```

The script installs the CLI, `jitllm`, and the server, `jitllmd`, from the latest
release, after checking each archive against the release's checksums: into
`~/.local/bin` (`/usr/local/bin` as root), or on Windows into
`%LOCALAPPDATA%\Programs\jitllm`, which it adds to your `PATH`. Add the apps with
`| sh -s -- all` (or `desktop`, `tui`), or `$env:JITLLM_PROGRAMS = "all"` before
the Windows line. Each program is one self-contained binary and its own archive
on the [Releases](https://github.com/jitllm/jitllm/releases/latest) page, for Linux,
macOS and Windows on x86-64 and arm64. GPUs need only their driver.

> **AMD GPUs:** they run through Vulkan with only the driver. With ROCm
> installed, jitllm also uses them through a native backend (`hip:0`, ...),
> which **has not yet been tested on AMD hardware**
> ([#34](https://github.com/jitllm/jitllm/issues/34)). To stay on the
> tested path, pick the Vulkan device explicitly (`-devices vulkan:0`).

## Quick start

```sh
# Fetch Qwen3-30B-A3B from the catalog and convert it (18.6 GB) into the
# models folder: a mixture of experts with 30B weights and 3B used per token.
# With less memory than it needs, it pages from disk.
jitllm convert qwen3-30b-a3b

# Chat with it. The installed service loads it on first use; with no service
# running, jitllm runs it in its own process instead and says so.
jitllm run qwen3-30b-a3b

# Or ask once.
jitllm run -chat qwen3-30b-a3b "Why is the sky blue?"
```

On Linux the install script starts `jitllmd` as a systemd user service on
`localhost:4242`; the macOS app and the Windows installer offer the same at
login, and `jitllmd service install` does it on any of them. Every model in
the models folder is available to it by name, loaded on demand:

```sh
curl http://localhost:4242/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-30b-a3b","messages":[{"role":"user","content":"Why is the sky blue?"}],"stream":true}'
```

`jitllmd service status` shows the service, `jitllmd models -alias
FILE=NAME` gives a model another name, and `jitllm run path/to/model.jlm`
runs a file directly. Flags go before the model. `jitllm <command>` with no
arguments prints the usage.

**Bring GGUF or Hugging Face (safetensors) weights**, from the catalog or your
own files. See the [CLI and conversion guide](docs/cli.md) for inputs and flags,
and the [server reference](docs/server.md) for endpoints and daemon commands.

### Docker

```sh
docker volume create jitllm-models
docker run --rm -v jitllm-models:/models --entrypoint jitllm ghcr.io/jitllm/jitllm convert qwen3-30b-a3b
docker run -d -p 4242:4242 -v jitllm-models:/models ghcr.io/jitllm/jitllm -load Qwen3-30B-A3B-Q4_K_M.jlm -id qwen3
```

Add `--gpus all` (with the NVIDIA Container Toolkit) to run on an NVIDIA GPU. The image is
`jitllmd` with the `jitllm` CLI beside it, for linux/amd64 and linux/arm64;
`docker build -t jitllm .` builds it from a checkout. See [docs/docker.md](docs/docker.md).

## Supported models

`convert` reads 65 GGUF architecture names (57 graphs), 30 Hugging Face safetensors
classes and 17 vision projectors; anything else is refused at conversion, by name.

| Kind | Families |
|---|---|
| **Text and code** | Llama 2/3, Mistral, Mixtral, SmolLM2/3, Qwen2 to Qwen3.6 (dense, MoE, Next and VL text), Gemma 1 to 4 and 3n, Phi-2/3/4 and Phi-3.5-MoE, DeepSeek V2/V3/R1/V3.2/V4, Kimi-K2, Kimi Linear, Kimi-K3, GLM-4/4.5/4.6/4.7-Flash, gpt-oss, Llama 4, Granite, OLMo 2/3, OLMoE, ERNIE 4.5, Hunyuan, MiniMax-M2/M3, Ling 2.0, dots.llm1, Apertus, EXAONE 4, Seed-OSS, Command-R/A, DBRX, Falcon, StarCoder 1/2, StableLM, Nemotron |
| **State-space and hybrid** | Mamba, Mamba-2, Jamba, Falcon-H1, Granite 4 hybrid, Nemotron-H, LFM2 |
| **Vision** | SmolVLM, LLaVA, Qwen2/2.5/3-VL, Qwen3.5, GLM-4.xV, Kimi-VL, HunyuanOCR, Gemma 3/3n/4, InternVL, MiniCPM-V, Janus-Pro, Pixtral/Mistral 3, Phi-4 vision, Llama 4 |
| **Embeddings** | BERT family, nomic-embed-text, Qwen3-Embedding, EmbeddingGemma |
| **Decision models** | Laya (ModernBERT), d1-3B, Lev 4B, answering one question against fixed labels through `/v1/systemone` |

Weights: F32, F16, BF16, Q4_0, Q5_0, Q5_1, Q8_0, Q3_K-Q6_K and MXFP4. The full list, generated
from the code, is [docs/models.md](docs/models.md); the [vision guide](docs/vision.md) covers pictures.

## Performance

Every ratio is jitllm's rate over the other engine's, measured in the same pass
on the same machine:

- **Up to 4.62x llama.cpp's prefill** (Qwen3-Next-80B on four V100s), ahead on
  34 of 35 models and faster decode on all 35, from 0.4B to 120B.
- **Up to 5.8x sooner to the first token than vLLM 0.18.1** and 1.26x its decode
  (Llama-3.1-8B, one V100), and **up to 5.53x its batched decode**.
- **Up to 3.72x llama.cpp's batched decode** at 16 to 128 sequences (V100).
- **Up to 1.53x llama.cpp's decode on an Apple M4** (Metal), up to 1.67x its
  prefill on the M4's CPU.
- **Up to 1.67x llama.cpp's prefill on a 12-core Xeon**, and up to 6x
  mistral.rs's decode.

The [performance overview](.notes/performance.md) has every row with its
hardware, backends and lengths, and where jitllm is still behind; the
[scoreboard](.notes/perf/current.md) has the commands and controls behind each.

## Why Go?

Most inference engines are written in C, C++ or Rust, and some of the best work
is. The choice here is practical:

- **It is the language the author knows best for writing a JIT**: emitting
  machine code, mapping it executable and calling into it.
- **It is portable.** One toolchain cross-compiles every target, and with no cgo
  there is no C toolchain or vendor SDK per target.
- **It compiles fast, so experiments are fast.** Most of this engine was found
  by trying an idea and measuring it.
- **It is built for servers.** Go's high-performance networking and
  synchronization let `jitllmd` serve many sessions at once, batch them in one
  step loop and time-slice them fairly, in one process with no framework. The server reaches the first token up to 5.8x sooner than vLLM
  on the same card, and a warm decode token allocates nothing, so the
  collector never touches the hot path.

The hot paths are generated machine code either way; Go carries the parts around
them (paging, placement, scheduling).

## Build

All you need is Go: **Go 1.26 or newer**, no C toolchain, no GPU SDK, no Python.

```sh
git clone https://github.com/jitllm/jitllm.git && cd jitllm
go build ./cmd/jitllm
(cd server && go build -o ../jitllmd ./cmd/jitllmd)
(cd ui && CGO_ENABLED=0 go build -o ../jitllm-desktop .)
(cd tui && go build -o ../jitllm-tui .)
```

Each is one static binary that finds CUDA, Vulkan or Metal at run time.

## Built for agents

- **For an agent using jitllm:** the OpenAI- and Anthropic-compatible APIs work
  with the SDKs and agent frameworks as they are; tool calls are read in each
  model's own format and `tool_choice` is enforced by constrained decoding, so a
  forced call is a call. `jitllm capabilities` and `GET /v1/capabilities` say
  what this build runs (architectures, quant types, backends, API fields and
  their refusals), generated from the code; `jitllm doctor -json` reports the
  host, devices and a model's compatibility, each problem with a next action;
  [tested recipes](docs/recipes/README.md) go from install to a running server,
  a connected client, tools, conversion, placement and diagnosis; and
  [`llms.txt`](llms.txt) indexes the documentation.
- **For an agent changing jitllm:** [`AGENTS.md`](AGENTS.md) is the one file
  that holds the project's binding rules, with a table that routes each
  subsystem to its evidence; the [model list](docs/models.md) is generated from
  the converter and CI fails when it drifts; `go run ./scripts/checks` maps
  changed paths to the gates that cover them; and the model-free suite
  (`go run ./scripts/modelfree`) runs every check that needs no model or GPU,
  reporting a missing model or device as a skip, never as a pass.

# jitllm roadmap

What is planned, in priority order. Each item carries its status. The
engineering history behind a status lives in `docs/engineering-history/`; this
file says only where each item stands and what is left.

## 1. Optimized kernels for MoE

**Largely done.**

- **Host dispatch is batched.** The k experts' gate/up share one activation and
  run in one pool region (`nn.MatVecPackedMulti`), and the down projections in
  one more (`MatVecPackedGather`): olmoe went from 906 to 234 regions a token,
  a dense model's density, and 10% less kernel CPU.
- **Routing is generated code.** The softmax/sigmoid, the top-k, the ascending
  sort and the renormalising divide run as kernels on the host and every device
  backend.
- **Every expert is its own page** (container v26): an over-committed mixture
  reads only the experts a token routes to, 3.10x on Moonlight-16B-A3B under a
  6 GB budget.
- **On the device** the indexed matvec splits over k (1.31x on
  Qwen3-MoE-4x0.6B), and a streamed placement (`tier.Config.StreamExperts`)
  uploads only the routed sheets, so Qwen3-Next-80B places 48 of 48 blocks on a
  2 GiB card.
- **gpt-oss is supported**, on the host and every device backend, with MXFP4 stored natively (container v24).

Where it stands: olmoe beats llama.cpp on the Xeon (1.06); the M4's CPU mixture
rows are the weakest on the board.

Left:
- a batched mixture path that pays on arm64 (it exists and is off there: the
  fused kernel's walk loses 24% to the per-expert loop at a mixture's shapes;
  it would have to drive the tiled kernel);
- fusing gate and up into one launch on the device, unmeasured.

## 2. Small models: BERT and ModernBERT

**BERT is done; ModernBERT is not started.**

`bert` and `nomic-bert` run as encoder graphs (`engine/model/encoder.go`), and
qwen3-embedding and EmbeddingGemma run as pooled readouts of their decoder
graphs. One API serves both (`Model.NewEmbedder`, `jitllm embed`), WordPiece is
in `tok/wpm.go`, and the outputs are gated against llama.cpp and float32
sentence-transformers.

Left: ModernBERT (llama.cpp's `modern-bert` pre-tokenizer is already in the
table; the architecture is not in `convert.archOf`).
ModernBERT is also the encoder most current PII taggers and routing
classifiers are built on, so it comes before items 13 and 14.

## 3. Model and request parameters, complete

**Mostly done.** `model.Sampler` has temperature, top-k, top-p, min-p, a
repetition penalty with its window, and a seed, and the sampler itself is
generated code. The server's `GenerateRequest` adds stop strings and a token
budget. Rope and YaRN parameters come from the container.

Left: frequency and presence penalties.

## 4. Concurrency

**Serving works on the host; the next piece is designed below.**

Done:
- **Ragged positions.** Each batch row has its own position, rotary table, KV
  offset and attention length, and `Retire(i)` frees a row.
- **A scheduler.** `model.Scheduler` admits queued sequences into free rows,
  steps them, and retires them on a stop or a budget.
- **Per-row prefill.** `PrefillSeq` runs an admitted row's whole prompt in one
  pass instead of one token per step.
- **Several sequences on one device tier.** `tier.GPU.Attach` gives each
  `model.State` its own session and KV history; sessions are serialised per
  device, not yet parallel.

Still refused: a *batched* State with blocks on a device (`errBatchDevice`).

**Next: fold prefill into the decode step.** The idea as put was "prefill the
next prompt while the current one finishes inference". It is not about
concurrency -- prefill and decode would contend for the same cores -- it is
about the weight pass. A decode step is memory-bound with its arithmetic idle,
so prompt tokens folded into it ride on a weight read already being paid for.
`Scheduler.admit` runs `PrefillSeq` as its own pass, so every live row
stalls for it: on tinyllama, admitting a 500-token prompt costs about 260
decode tokens of everyone else's time.

It needs a token *count* per row, not just a position: row A contributes one
decode token, row B sixteen prompt tokens, in one step over sum(n_i) tokens
with a per-token (slot, position) map. `Prefill` and `ForwardBatch` then become
one function -- they are two transcriptions of one graph. This is what
vLLM calls chunked prefill. Do not assume llama.cpp's fixed-width slots are the
right shape; jitllm's batch is a property of the State.

## 5. KV cache as an interface

**Done, apart from a remote backend and 5a.**

The cache is paged (`engine/model/kvpage.go`): pages sized per layer from the
kernels, living on whichever device holds their layer, growing a page at a time
instead of reserving the whole context. A page leaves a tier through the
user's interface -- `Get(cacheId, layer, index, io.Writer)` plus `Set`/`Drop`
-- with three backends: `NoStore` (the default), `MemStore` and `FileStore`.

Prefix reuse is built on it and keyed by content: a page is stored under a
hash of the tokens that produced it, so sessions that share a system prompt
reuse it without knowing about each other (`jitllm run -kv-cache DIR`: 5.8x on
a repeated 1181-token prompt). `SetCacheKey` names a namespace, which is also
the tenancy dial.

Left:
- a remote backend, which belongs outside core;
- restoring a cached prefix onto device blocks (`PrefillCached` refuses when
  blocks are placed, rather than attending over history the card never saw).

The backends, the adversarial audit of the cache and the prefix
design are recorded in `docs/engineering-history/placement.md`.

### 5a. The RAM prefill buffer (not started)

The one piece of the paged cache deliberately left unbuilt: when the first
layers run on a GPU and VRAM is short, a RAM prefill buffer lets the transfer be
batched rather than paid page by page.

It is not a `KVStore`: a store is where a page lives when no tier holds it. The
prefill buffer is a tier on the path between the host and the card: prefill
seals many pages per layer at once, and the point is to batch them into one
bulk transfer so the card need not hold the whole cache while the prompt runs.

Already true and not to be re-derived: sealing is the trigger
(`kvCache.seal(pos)` offers each page once as the token moves past it);
`State.migrateKV` already gathers pages into one run for
`nn.LayerDevice.MigrateKV` and scatters them back; and the device is the
constrained side, since the host commits pages lazily.

**Its design is open.** Whether the buffer is a tier with its own residency or
a batching window over the existing migrate path is undecided, and that choice
comes before any code.

## 6. Swarm orchestration over the network

**Not started.**

Distributed inference across machines. The interface belongs in core with no
dependency on libp2p or any transport; an implementation lives outside, the
dependency pointing inward -- the shape `nn.Device` already has (declared in
`engine/nn`, implemented in `jit_old/gpu`).

What that protects: `go.mod` has one require (goffi) and libp2p is about 150
modules; and there is no cgo, since goffi reaches the drivers through dlopen.

## 7. Pagination as an interface

**Half done.** The engine no longer maps weights: `format/jlm` reads pages with
`pread` (O_DIRECT where the filesystem allows) into frames it owns and evicts
itself; GGUF's mapping is the converter's alone.

Left: `jlm.Open` takes a path and reads an `*os.File`. A remote source (an
object store, a peer) needs the container to read through an interface shaped
like `io.ReaderAt` instead.

---

The common thread in 5, 6 and 7 is one architectural move: an interface in
core, implementations outside, the dependency pointing inward so `cmd/jitllm`
links none of them. `nn.Device` is the precedent.

## 8. MLX 4-bit checkpoints

Convert and run MLX's quantized safetensors (mlx-community and `mlx_lm.convert`
output). The format is affine group quantization: 4-bit codes packed eight to a
`uint32`, one scale and one bias per group of 64 weights (32 and 128 also
exist), `w = scale*q + bias`, described by config.json's `quantization` block.

- **Converter:** the HF safetensors path (`convert/safetensors.go`,
  `hfArchTable`) already reads MLX's tensor names; it needs the `quantization`
  config, each `weight`/`scales`/`biases` triple joined into one tensor, MLX's
  nibble order, and quantized embeddings and heads.
- **Format:** a native group-64 type (one scale/bias pair shared by two
  32-element activation blocks, +0% bytes) rather than a Q4_1-style repack,
  which stores every pair twice and reads ~11% more bytes on a
  bandwidth-bound decode. A new quant type: reference dequantizer first
  (RULE 8), then kernels on every tier. Scales are often bf16; storing them
  as f16 must be exact, or refused at conversion.
- **Oracle:** `mlx_lm` on the M4, teacher-forced against MLX's own logits for
  the same checkpoint (RULE 7m -- llama.cpp does not read MLX).
- **Scope:** 4-bit at group 64 first, 32/128 if cheap, then 8-bit. MLX's
  mxfp4 mode maps onto the existing MXFP4.

## 9. Ideas from Strata (single-model MoE engine for gaming GPUs)

Strata (github.com/Niko1221/Strata, C++/CUDA/HIP) runs one ~125B hybrid MoE,
Qwen3.8-Flash-Next, on a 12-16 GB card plus RAM, and claims 53-93 tok/s
decode on an RTX 5070 (not measured here). Our streamed Qwen3-Next-80B on a
4 GB card reads 0.78 tok/s; the gap is design, not kernels. In value order:

1. **The expert as the unit of placement, computed where it lives.** The GPU
   computes the experts cached in VRAM while the CPU computes the uncached
   ones in place from pinned RAM, concurrently, and a tunable share of misses
   (`pcie_frac`, 0.2-0.55) is fetched over PCIe instead
   (`src/kernels/cpu/expert.cpp`, `src/program/generate.cpp`). We place whole
   blocks on one side, or stream every routed expert to the card.
2. **An adaptive expert cache on the device.** Every 4 steps it swaps up to 96
   of the most-routed missing experts in for the least-routed resident ones,
   by decayed counts, while the CPU computes the outgoing ones so no token
   stalls (`adapt_every`/`adapt_swaps`, `resident_stage_swaps`); an offline
   routing profile seeds it. The device-side twin of our host expert LRU.
3. **Speculative decoding with the model's own MTP head:** up to 3 drafted
   tokens checked in one pass, 2.4-3.2 tokens a pass, 1.6-1.8x decode. Our
   converter used to set MTP blocks aside (`convert/qwen35.go`). Built:
   `jitllm run|speed -spec mtp`, with an adaptive draft count that stops
   speculating where no count wins (`engine/model/spec.go`).
4. **Recurrent-state checkpoints for conversations:** at turn boundaries,
   every 16k prompt tokens and at the end of the system prompt, six kept
   (~118 MB each) -- where a hybrid now restores at one point per prompt.
5. Smaller: prompt-lookup drafting enabled only where measured acceptance
   pays; the next layer's experts streamed over PCIe during this layer's
   attention in prefill; long-context KV keeping only its most-read part on
   the device; a multi-GPU layer split from a fitted cost model.
6. Reach beyond today's inputs and hardware: converting the AWQ, GPTQ and
   EXL3 safetensors checkpoints a large share of quantized releases ship as;
   an Intel AMX emitter for int8 and bf16 matrix units on recent Xeons;
   a native ROCm backend for AMD cards beside Vulkan; and audio (Whisper
   speech-to-text, then TTS) as the next modality after vision.

Not applicable: per-model kernel specialisation, the ggml kernels, a
single-user server, and its "speed projection" (a refusal-removing control
vector, not an optimisation).

## 10. Ideas from "AI as a compiler" (arXiv 2609.36800)

An LLM agent lowers Triton kernels straight to PTX, accepted by randomized
differential testing, and reaches 0.83-3.34x autotuned Triton on Ada, Hopper
and Blackwell at $1-15 of search per kernel. That search is offline and
per-target, so it cannot replace a JIT that emits at load in milliseconds
(RULE 8); what transfers is two transformations it found. Their matvec gained
0.98-1.15x and RMSNorm/RoPE at most 4-5%, which agrees with decode being
bound by bytes here. Their numbers are mostly single runs with best-of-run
headlines, and the kernels a formal checker could prove equivalent lost most
of the gain (GEMM 1.03x -> 0.24x). Neither idea below is measured here.

1. **A thread owns a whole softmax row in device attention.** Their
   FlashAttention kernel (1.37x) gives each thread one complete score row in
   the tensor-core layout, so the row's max and sum are register-local and
   no cross-lane shuffle is issued. Ours reduces across lanes, which is part
   of why some kernels need a guaranteed 32-lane subgroup. A row-per-thread
   form needs no subgroup at all, so it may be both the portable form
   (llvmpipe, Intel) and a faster one. Measure against the current kernel on
   CUDA and Metal before choosing.
2. **Quantized weights decoded straight into tensor-core operands.** Their
   1-bit BitDelta GEMM (3.34x) builds the fp16 operands in registers -- a bit
   permutation plus one `lop3` that sets sign bits of a packed constant --
   instead of unpack, convert, store to shared memory and reload. Check
   whether the device prefill GEMM (`jit_old/gpu/kernels/mma.go`) stages
   unpacked weights through shared memory; if it does, unpacking Q4_K/Q6_K
   directly into the mma fragments is a prefill lever, not a decode one.

## 11. Decision models

**Not started; scope to be defined.** Open-weight "decision" models, as opposed
to text generators. The kind is
not settled yet: candidates are JEPA-style world models (encoder plus predictor in
embedding space), rerankers and classifiers that output a score or a label,
reward and judge models, and action models (Decision Transformer, VLAs). Each
would ride the same principles as every other model here.

## 12. Ideas from ds4 (DwarfStar)

**Reviewed; not implemented or measured here.** Source:
[antirez/ds4 at 0aaea5a](https://github.com/antirez/ds4/tree/0aaea5a238fb41a35106a551e73c8409dfb751ac).

1. **Expert eviction by decaying routing frequency.** Its Metal cache counts
   selections including misses, ages the counts, and evicts by lowest frequency
   with recency as the tie-breaker. Counting misses avoids penalizing experts
   evicted before they can register a second hit. Compare this with our host
   expert LRU by replaying real routing traces at identical byte budgets,
   including prompt, decode and tool-result transitions. Measure missed bytes
   before runtime integration. Related to item 9's device-cache proposal;
   this experiment concerns the existing host pager.
2. **Checkpoint-matched continuation fixtures.** Its
   [quality tooling](https://github.com/antirez/ds4/tree/0aaea5a238fb41a35106a551e73c8409dfb751ac/gguf-tools/quality-testing)
   scores fixed hosted-model continuations by token negative log likelihood,
   including long-context cases. Extend our existing teacher-forced checks
   with independent reference coverage when adding relevant model families.
   Hosted continuations complement numerical oracles; they do not establish
   exact correctness.
3. **Reference for future architectures.** Revisit its model implementations
   and fixtures when a corresponding architecture is selected for support.
   Kernel algorithms may inform generated implementations; copied kernels,
   mapped-file overflow and its default distribution-changing speculative
   sampling do not fit our constraints. Selected-expert reads, I/O overlap,
   prefix reuse and adaptive speculation already exist here.

## 13. PII detection

**Not started.** Find and redact personal data in text as an engine feature: a
token-classification (NER-style) head on an encoder -- BERT/ModernBERT PII
taggers -- returning labelled spans, plus a redaction call that masks or replaces
them. It runs the encoder path the embedder already uses (`engine/model/encoder.go`)
with a per-token classifier readout in place of pooling, and the server can apply
it to a request before generation and to the output after. Needs ModernBERT
(item 2) for the current models; gated against transformers' token-classification
pipeline on the published checkpoints.

## 14. Semantic routing

**Not started.** Choose where a request goes by what it means: embed the request
(the embedder that exists, or a classifier head), compare it against each route's
examples or centroid, and send it to the model, instance or tool that route names
-- the server is mostly a multi-model server, so the natural target is one of its
loaded models. Routes and thresholds are the caller's (options, not built-in
policy), a request that matches nothing falls to a default, and the route taken
is reported with its score. Builds on `Model.NewEmbedder` and the server's
per-model instances; a classifier-based router shares item 13's readout.

## 15. Full Kimi-K3 from disk

**Not started.** Run the whole 2.78T Kimi-K3 on one machine,
paged off disk with a memory budget far below the model, the way
kimi-k3-in-c does (1.56 TB checkpoint, about 8 GB resident, about 26 s a token
on a laptop), and compare rate and peak memory on the same box. jitllm's K3 is
verified on trained weights at small scale (Kimi-K3-0.40B and its Kazakh
continuation, f32 rounding against Moonshot's own code, `TestKimiK3RealMatchesReference`);
the full model has never been opened.

What it needs:
- **Disk.** About 1.5 TB for the checkpoint plus the container; converting
  straight from a GGUF and dropping it keeps the peak near 1.5 TB rather than 3.
- **A format the converter reads.** The official release stores its experts as
  compressed-tensors MXFP4 safetensors, which the safetensors path does not
  decode yet; the GGUF route works for Q4_K-class files (the 2-bit quants are not
  dequantable).
- **q, k and v as three projections in a KDA block.** A mixed-quant GGUF stores
  q at Q4_K beside k and v at Q8_0; the converter joins them as one Q8_0
  projection, which grows q from about 4.5 to 8.5 bits a weight in every
  linear block. Running the three as separate projections keeps the file's
  bytes and changes every tier's linear block.
- **Scale checks** nobody has run at this size: page-table size, open time,
  conversion time, and the expert pager under a budget a small fraction of the
  model.

## 16. Application-facing contracts and serving under load

**Not started**, except where noted. Each item builds on what exists rather
than replacing it.

1. **A compatibility matrix and client-level tests.** Fields whose omission
   changes a request's meaning are refused, never ignored (non-zero
   presence/frequency penalties and `tool_choice` already are). Left: a written
   matrix of supported fields per API, and conformance cases for streaming and
   non-streaming, tool-result replay, cancellation and malformed input.
   Harmless unknown fields stay accepted.
2. **A mixed-load serving benchmark before any scheduler redesign.** Open-loop
   arrivals, mixed short and long prompts and outputs, concurrency 1 to 64,
   reporting p50/p95/p99 first-token and inter-token latency, completed and
   refused requests and throughput, beside counters that show which path ran.
3. **Shared steps for partly placed sessions and image prompts**, chosen by
   what that benchmark shows they cost. Today a session with blocks on both the
   host and a device, or a prompt carrying a picture, runs outside the shared
   step.
4. **A stateless `/v1/responses` endpoint**, when a client that needs it is
   the target; stateful behaviour is refused explicitly.
5. **Standard metrics, extended rather than rebuilt.** `/metrics` carries
   prefill and decode timings; add queue time, inter-token latency, prefix-cache
   hits, page faults, bytes transferred and shared-step refusal reasons, with
   labels of bounded cardinality (never a prompt or a session).
6. **Conversion coverage, ranked by the checkpoints people are blocked on:**
   Q2_K first for large models, then MLX 4-bit (item 8), then direct
   safetensors classes for graphs the GGUF path already runs. Other formats
   (IQ, AWQ, GPTQ, FP8) only for a named checkpoint.
7. **A capability map tied to code and tests**: graph support, source formats,
   CLI and server surface, hardware validated and real checkpoints validated,
   kept separate. Stale statements here and in source comments are corrected in
   place as part of it.
8. **Features held to compose:** speculation, constrained output, batching,
   stop strings, streaming and cancellation in combination, not only each one
   alone.

## 17. Installing from the app itself

The release ships a portable desktop app for Windows and the signed app zipped
for macOS (`packaging/README.md`); neither installs itself yet.

- **Windows:** the portable `jitllm-desktop` offers "Install jitllm": it copies
  itself (and the CLI and server, fetched from the same release and checked
  against `checksums.txt`) into `%LOCALAPPDATA%\Programs\jitllm`, adds it to
  the user `PATH`, creates the Start menu entry and the Add/Remove Programs
  entry the setup writes, and optionally starts the server at login -- the
  setup's install, reached from the app.
- **macOS:** run from Downloads or from the mounted dmg, the app offers "Move
  to Applications" and relaunches from there, so its PATH links and launchd
  agent point at a bundle that stays put.
- **Linux, if wanted later:** an AppImage, and `.deb`/`.rpm` packages.

## 18. Dense blocks streamed through a device in tiles (an idea to explore)

**Not started.** A dense block larger than every device's whole budget runs on
the host today, and so does every block a partial placement leaves home. Its
weights could be cut into page-sized tiles and streamed through a card,
uploaded double-buffered while the previous tile computes, as the device
already streams a mixture's routed expert sheets and a session's evicted
history.

It pays only where a tile is reused across many rows: prefill and wide batched
steps, where a PCIe stream feeds the tensor cores. Single-sequence decode reads
each weight once per token, and the host reads its own memory faster than PCIe
carries it, so that stays home. The choice would be made per step from the
measured transfer rate against the host's, and the block's home stays on the
host. The wider prize is GPU prompt speed for the blocks a partial placement
leaves home, the common case on consumer machines.

## 19. Coding agents on a local model, one command

**Not started.** `jitllm` starts a known coding agent (Claude Code, Codex,
opencode, aider and others) already pointed at a running `jitllmd`: it finds
the server (or a given address), picks or loads a model fit for the agent
(tool calling, context length), and launches the agent with its base URL, key
and model set in the way that agent reads them (its OpenAI- or
Anthropic-compatible endpoint, environment or config), without touching the
agent's own settings files. `jitllm agents` lists which it found installed and
which can run against the loaded model, naming what is missing (tool calling,
context, a vision projector) when one cannot.

## 20. Relieving memory pressure across models

**Not started.** The server holds several models, and today a model stays loaded
until a caller unloads it. When memory runs short -- a new model asked for, a
session's KV growing, another process taking RAM or VRAM -- the orchestrator
should make room on its own: first by paging (shrink a model's resident pages and
KV toward its floor), then by unloading a model nobody is using.

Which model gives way is a policy, chosen by the caller (a `With...()` option and
a server flag), never one fixed rule:
- **Inputs the policy weighs:** each model's footprint per tier (weights, KV,
  expert pages), how recently and how often it was used, its live sessions and
  queued requests, its reload cost (container size against the disk's measured
  read rate, plus warm-up), and a caller-given priority or pin.
- **Policies to ship:** least recently used; least used, decayed; cost-aware
  (the bytes freed against the time a reload would cost, weighted by the
  model's expected next use); and never-unload with pins.
- **Actions in order of cost:** trim pages within a model, move blocks to a
  cheaper tier (relocation), park idle sessions' KV to disk, then unload the
  whole model. A model with a running request is never unloaded mid-token; a
  model is reloaded on its next request.
- **Pressure signals:** the engine's own budgets, the device ledger, and the
  host's (PSI memory, the cgroup's limits) so another process taking memory
  counts as well.

Every decision is reported (what was trimmed, moved or unloaded, and why) in the
server's stats and logs. Builds on paging, relocation, session parking and the
server's per-model instances.

## 21. The pre-tokenizer runs the model's own pattern

**Not started.** The pre-tokenizer recognises known `tokenizer.json` Split
patterns by their text (`tok/pretok/parse.go` `parseSplit`, `isolatedSplits`),
hand-writes a matcher per family (`tok/hunyuan.go`), maps a GGUF's
`tokenizer.ggml.pre` name to stages through a generated table, and keeps legacy
name families in `tok/bpe.go` for containers that stored no stages. Every
engine that reads `tokenizer.json` (HF tokenizers and everything on it: vLLM,
SGLang, MLX, ExLlamaV3, TensorRT-LLM; mistral.rs and candle with fancy-regex;
Ollama on rewritten RE2) runs the pattern on a regex engine; only llama.cpp
hand-codes matchers, and that is what this copied.

- **First, a bug:** `parseSplit` turns any Split pattern that is not an exact
  known text into an `OpSplit` with whatever flags its substring checks set, so
  `\S+` or `[^\n]+|[\n]+` silently runs as the llama3 splitter, against its own
  comment. An unknown pattern must be refused, naming it.
- **A matcher of our own** for the constructs these patterns use (alternation,
  classes, `\p{..}`, quantifiers, `(?i)` with full case folding, lookahead such
  as `\s+(?!\S)`), with its Unicode classes from `tok/unicodedata.go` so the
  15.1.0 pin holds, refusing anything else by name; generated like the rest of
  the JIT if that measures better. It is the definition of the split.
  Oniguruma's behaviours to reproduce: Unicode `\s`, `\d`, `\w`; `$` is end of
  line; an empty match is a split point that yields no piece.
- **The container stores the pattern text**, and the GGUF name table carries
  llama.cpp's pattern per name rather than ops.
- **Fast paths** (llama3, o200k) stay only where a gate proves them equal to
  the matcher; `hunyuan.go`, the other recognisers and the legacy name families
  go.
- **Oracle:** HF tokenizers on the model's own `tokenizer.json` (tiktoken for a
  model that ships a `pat_str`) replaces llama.cpp for tokenization (RULE 7m),
  llama.cpp staying the source of a GGUF name's pattern. Golden ids per
  vocabulary from a `scripts/` generator over an adversarial corpus and real
  multi-script text, zero differing ids; a split-level gate over every pattern
  on the model disk and in llama.cpp; run against a violation (dropping
  multi-line `$` or the Unicode `\w` must fail it).
- **Invalid UTF-8** (U+FFFD per bad byte, which no reference accepts as input)
  is pinned in a test and in RULE 7m's divergence table.

A spike compared `dlclark/regexp2`, adjusted for those behaviours, with HF
tokenizers on 50 patterns and 20,000 strings each: 0 differences, but 7-17x
slower than the hand splitters, so it is a test reference, not the engine.

## 22. The apps say which engine they run, and can connect to a service

**Built:** `common/engine`'s `Remote` (the connected engine), `Choose` (Auto, Embedded, Service) and `Badge`; the desktop's Settings > Engine and header badge, the TUI's `--connect` / `--embedded` and status-bar badge; the API refused on a held address. Before it, the desktop app and the TUI always ran an engine of their own;
the desktop can also serve the API from it (`common/api`, `localhost:4242`),
and the Mac app can install a login service, a separate `jitllmd` on the same
address with its own models (`ui/install/agent.go`). A user can end up with two
engines on one machine, sharing its GPU memory and RAM and wanting one port,
and nothing in either app says which one it is talking to.

- **One engine setting:** Auto (connect to a `jitllmd` that answers at the
  configured or the login service's address, otherwise run embedded), Embedded,
  or Service at an address; a desktop setting, and `--embedded` /
  `--connect addr` in the TUI.
- **Connected mode:** the app as a client of `jitllmd`'s control plane
  (models, sessions, placement, stats, chat), as a second implementation of
  the engine interface the screens already use (`ui/screen/deps.go`), so the
  model list, memory map, block placement and chat behave the same against a
  service. A lost connection is shown as a problem, never a silent fall back to
  embedded.
- **Always visible:** a badge in the desktop header and the TUI status bar --
  `Embedded · API off`, `Embedded · API on :4242`, `Service · localhost:4242
  (jitllmd <version>)` -- that opens the setting.
- **No silent second engine:** turning the API on where the login service
  already holds the address is refused with an offer to connect; starting
  embedded beside a running service says the two share the GPU's memory.

## 23. Fusion found by search: analysis at conversion, timing on the device

**Not started.** Today each fused kernel is written by hand and a small tuner
picks among a few variants per device (the split tuner, the decode prologue's
per-card tuner). The kernel composer the project aims at searches instead:
which ops of a block to fuse, and how to tile them, chosen per shape and per
device. The work splits along what the container can know.

**At conversion (hardware-independent, stored in the `.jlm`):**
- The fusion graph of each block: which ops can fuse (norm into the next
  matvec, q/k/v and gate/up sharing an input, an activation into its
  consumer), which blocks are identical, and the model's distinct shapes, so
  the device's search starts pruned.
- Per-block weight constants the fused paths need at run time, computed once:
  e.g. the weight-side sums and minimums an int8 activation path applies as
  corrections.
- Later, layouts that suit fused kernels on every tier (q/k/v and gate/up
  rows interleaved, quantised blocks ordered for fragment loads, rows padded
  to tile multiples) — only where no tier is hurt, since one container serves
  the host and every device; a layout change is a container version.
- Optionally, a ranked shortlist of candidates per device class from a cost
  model, as the search's prior.

**On the device (hardware-dependent, never in the container):**
- A bounded search over that shortlist: generate and compile a few dozen
  candidates per distinct shape, time them by interleaved trials in the
  background after the model is ready (nothing compiles in the user's path),
  keep a winner only if it beats today's kernel, cache it per binary, device
  and shape, and adopt it when it is safe to.
- Every winner is held to the per-op kernels (token ids, logits) before it is
  kept; the per-op path stays the reference and the fallback.

First steps: the fusion graph and the per-block constants at conversion; then
the search over the existing fused variants (#87, #110, the CPU groups) before
any new kernel shapes. Prior art: Mirage (superoptimizer and MPK), TVM Ansor,
Triton autotuning.

## 24. Speculative decoding with a separate drafter model

**Not started.** Speculation today is the model's own MTP heads and prompt
lookup. A small drafter model of the same family (its tokenizer identical)
proposes several tokens; the target checks them in one batched pass and keeps
the matching prefix, so the answer is the target's alone. The drafter is a
second loaded model, so placement applies: drafter on the host or a second
device, target on the card, each paged and relocatable as any model. Open:
choosing the draft length per measured acceptance, tree drafts, and sharing a
batch's verification pass across sequences. Gate: greedy output identical
with the drafter on and off; rows with speculation off and on beside
llama.cpp's and other engines' drafters on the same pair.

## 25. Prompt reuse measured and made fast

**Partly built.** The prefix cache restores a stored prompt's KV instead of
recomputing it. What is missing is the number: time from request to first
token for a reused prompt (short and long, host and every device, resident
and paged-out), published beside other engines' figures on the same box, and
the work it points to — keeping hot prefixes on the card, restoring pages
layer by layer while the first blocks already run, and keying long system
prompts and tool definitions so agents hit them.

## 26. Hybrid and state-space chats resume at every turn

**Not started** (open-questions: "hybrid restore points during prefill").
A hybrid's recurrent state is restored at one point per prompt, its end, so a
follow-up turn recomputes the conversation. Snapshot the recurrent state with
the KV at planned prefill cuts placed on the chat template's message
boundaries, so a cold prompt and a resumed one cut identically and a resumed
turn equals a fresh one. Snapshots are budgeted and evicted least recently
used like any page. Gate: resumed equals fresh, token for token. Prior art:
TensorFold's prompt radix tree (a 9.2k-token turn resumed in about 0.3 s).

## 27. Speculation priced by measurement, then a draft tree

**Not started.** (1) Measure the verify cost per row count (1-16) and the
draft step's cost on the device at load, keep acceptance odds per stream and
depth, and verify a draft only while its expected gain beats that row's
marginal cost; this replaces the count heuristic in `spec.go`. (2) Then tree
verification: the head's second-ranked tokens and prompt-lookup copies as
extra branches, the target's picks choosing the path, KV compacted to it and
recurrent state kept per parent, on every tier (RULE 8a). Start with two
branches at depth one and measure. Small companions: a silence period for
prompt lookup after rejected runs; grammar-forced tokens as free drafts under
structured output; a reduced draft vocabulary chosen at conversion, measured
per domain.

## 28. The speculative round stays on the device

**Research.** On a model with a small active set, the host's wait between a
round's submissions can dominate. Run accept, KV compaction and the next
round's arguments as generated kernels writing a record ring the host reads
behind. Measure the host-sync share of a round first.

## 29. Sampling that speculation and batching cannot change

**Not started; opt-in.** Gumbel-max sampling with noise keyed by (seed,
position, token), so a drafted run emits exactly the undrafted sequence at
any temperature, and a batched stream its solo run. Measure acceptance
against today's rejection sampling before any default changes. A pinned
reduction-order test mode would let speculation and batch gates assert bits.

## 30. Shared prefixes survive a restart

**Not started.** Persist hot prefixes (system prompts, tool definitions) to
disk, keyed by the container's hash, the JIT's kernel hash and the host, with
a disk budget and least-recently-used eviction, so an agent's first turn
after a restart restores instead of recomputing.

## 31. Metal 4 tensor units on M5

**Not started; needs M5 hardware.** The fragment layout the IR lacked (16x16
operand fragments, the 16x32x16 operation) is now documented in working
prior art; express it as IR ops, dequantize into threadgroup memory, then
multiply fragments, for Metal prefill. Ties to the Metal MMA gap.

# Prior art worth reading, and what to take from each

Pointers, not endorsements. Each entry says what jitllm might learn and what it
should NOT copy, because the failure mode of a list like this is a project
adopting a technique whose premise it does not share.

## zml/zml — https://github.com/zml/zml

Added on the user's pointer. **Not yet read; this entry is a stub
that says what to look for, and must be rewritten by whoever reads it.**

Why it is likely relevant given where jitllm is:

- It is a Zig inference stack that targets many accelerators through one
  compilation path, which is the same bet jitllm makes with one IR lowered to
  PTX, SPIR-V and MSL. How it handles per-target divergence is directly
  applicable -- jitllm has ZERO per-target specialisation today (measured:
  no `case "ptx"` anywhere in `jit_old/gpu/kernels`), and a 1.80x gap
  between CUDA and Vulkan on the SAME card that is entirely software.
- Anything it does about device memory residency and streaming weights bears on
  `docs/design/block-paging.md`, which is a spec and not yet built.
- Its device abstraction is worth comparing against `backend.Device`, which
  earned itself by having three implementations. RULE: do not add a fourth
  abstraction in advance of a second implementation of anything.

What to be careful about:

- A different language's memory model makes some techniques untransferable.
  jitllm is Go with goffi (jit_old/gpu/ffi) and no cgo, deliberately, and a technique that needs
  cgo is not a technique jitllm can take.
- RULE 7: the architecture list is a LIST and adding to it is a decision. A
  broader stack's coverage is not an argument for matching it.
- **Read the artefact, not the description of it** (RULE 7). If a technique looks
  good, find where it is implemented and what it measures, not what the README
  claims.

## llama.cpp

The oracle, not the specification -- see RULE 7m, which has teeth: its BPE
pre-tokenizer is an adaptation of the models' own regexes and the adaptation is
LOSSY (`(?i:s)` matches U+017F, `[sS]` does not). Being bit-compatible there
means being wrong in the same way it is. Used for: greedy token ids,
`llama-tokenize --ids`, `llama-eval-callback` per-node dumps, and
`llama-bench` rows via `scripts/vs-llamacpp.sh`.

## Ollama

Covered ~21 architectures with a funded team and deleted 430,004 lines citing
scope. Quoted in RULE 7 as the reason jitllm's architecture list is a list.

## colibri — `JustVugg/colibri`

Read twice (shallow clone, read-only); moved here from ARCHITECTURE.md §5a-ter when that file was
retired. The file references are colibri's as of those readings.

A streaming-MoE memory-tiering engine in C (127,294 lines plus 50,572 of Python), with eight model
families built as eight separate binaries dispatched by `exec`. What it is NOT, checked because the
hypothesis put to it was "the model converted to code in mmap, plus a layer scheduler":

- **No run-time code generation.** A grep for `PROT_EXEC|mprotect|nvrtc|cuModuleLoadData|shaderc`
  over its 99,281 lines of C found two hits, both `glslc` at BUILD time. CPU dispatch is an if/else
  chain on an integer tag taken every call (`c/colibri.c:1218`), ISA selection is `#ifdef` against
  the build's `-march`, and its Metal backend compiles one fixed MSL source with every dimension a
  runtime argument.
- **mmap is demoted, not the default.** The default path is `pread` plus `posix_fadvise(DONTNEED)`;
  `c/st.h:3-5` says mapping charged whole-model pages to the process, and mmap survives only as
  opt-in `COLI_MMAP`. It ships no A/B of the two. jitllm measured one (`format/jlm/read.go`,
  and AGENTS.md's no-mmap section): sixteen concurrent `pread`s beat an mmap fault 4.73 to 2.22 GiB/s,
  while ONE `pread` loses to it (1.58). Queue depth is the variable, so colibri's conclusion holds and
  its stated reason (RSS) is the smaller half.
- **Nothing schedules layers.** The unit is the EXPERT: a per-layer LRU expert cache, an aged-LFU VRAM
  residency tick, and placement ranked by persisted routing counts.

What is worth taking:

- **A converter that certifies itself.** `c/cfse_pack.c` converts a safetensors shard into an
  entropy-coded container (CFSE: each tensor a CFS1 stream, `c/fse_coli.h`, with a raw fallback), and
  `cfse_pack --cert in.safetensors` round-trips compress, decompress and `memcmp` for EVERY tensor,
  per tensor via `fseek`, in about twice the largest tensor's RAM and with zero writes. Compression
  itself is the wrong direction for jitllm (it adds host-side work where the container removes it); the
  verification discipline is the part that transfers to `jitllm convert`.
- **A frozen baseline in the same binary.** `c/tests/bench_gemv_stream.c:35-52` compiles a historical
  kernel beside the live one, so its A/B denominator does not move with HEAD. jitllm's paired harnesses
  compare live arms, and a cross-commit ratio comes from building the older tree separately (the
  v6/HEAD rows in AGENTS.md RULE 1 were taken that way).

What not to copy: its anti-duplication failed measurably -- `smap_init` in `c/qwen36.c:101` lacks the
overflow and allocation checks its sibling `c/qwen38.c:97-106` has -- which is RULE 7's case for a
short architecture list made by someone else's code.

## Recurrent Looped Transformer (RLT)

alphaxiv `2609.recurrent-looped-transformer` (Yifan Zhang); moved here from
ARCHITECTURE.md §5a-ter-bis.

**It is a specification, and says so**: *"RLT is presented as an architectural and execution
specification rather than a completed empirical system... It does not report benchmark results,
measured throughput, latency, memory consumption, or comparisons with trained baselines."* A causal
encoder builds prefix-restricted key-value memory; a recurrent decoder carries a hidden state plus a
layerwise sliding-window cache bounded by a window W, and the same transition runs for prompt and
response tokens, so there is no prefill/decode split. The paper claims no arithmetic or latency win and
records that, with top-k or top-p in the behaviour policy, exact importance sampling is not recoverable
from its RL replay metadata.

What it bears on here: jitllm keeps two device kernels for one set of weights (`jit_old/gpu/kernels/matvec.go` for
decode, `jit_old/gpu/kernels/mma.go` for prefill), and changes have reached one and not the other three times
(`docs/engineering-history/gpu-kernels.md`); RLT is the architectural version of removing that seam. There are no trained
weights, no GGUF and no reference implementation, so under RULE 7 there is nothing to run or gate
against.

# Serving over HTTP: how an engine becomes a fair arm

`dev/httpbench` is the one harness for every jitllm-vs-engine row taken over
HTTP: jitllmd, llama.cpp's `llama-server`, vLLM, SGLang, mistral.rs and ZML.
A row from any other client (an engine's own `benchmark_serving.py`, `wrk`,
a script) is not on the same board, for the same reason RULE 2 keeps
`scripts/vs-llamacpp.sh` the only source of a llama.cpp row: two tools time
two different things.

This file is how to run it. It is evidence and method, not a rule; RULE 1 and
RULE 2 in AGENTS.md govern every number it produces.

## What it measures

A closed loop: at concurrency `c`, `c` workers each send a streamed request,
read it to `data: [DONE]`, and send the next. Per request:

| field | meaning |
|---|---|
| TTFB | response headers arrived |
| TTFT | the first chunk carrying text arrived |
| ITL | every gap between two chunks carrying text |
| tokens | `usage.completion_tokens` from the stream's last chunk |
| finish | the finish reason the engine sent |

The request asks for usage with `stream_options.include_usage`, which vLLM and
SGLang need and jitllm and llama-server ignore (they send it anyway). A stream
without usage falls back to counting chunks and the table says how many did:
an engine that holds text back sends one chunk for several tokens, so a chunk
count undercounts.

ITL is per chunk, not per token. That is the only thing a client can see,
and it is the same for every engine; a chunk carrying two tokens shows up
as one long gap, not two short ones.

Per level and arm: TTFT p50/p90/p99 and ITL p50/p99 (numpy's linear
percentile, as vLLM's and SGLang's scripts report), the median request's
decode rate (tokens after the first, over first-to-last chunk), aggregate
tok/s (every completion token over the level's wall time), goodput (requests/s
within `-slo-ttft` and `-slo-tpot`, the latter on the request's mean gap), the
error count, and **at max**: the share of requests that generated exactly
`max_tokens`. At max is how you know an engine honoured `ignore_eos`. The
harness measures this and does not assume it.

## The workload

- **Prompts** are generated from a fixed seed (`-seed`), from common English
  words, one length drawn per request from `-prompt-lens` (words, about a
  token each on a llama-family tokenizer; each sample records the engine's
  own `prompt_tokens`). `-prompts FILE` replaces them with one prompt per
  line, or `{"prompt": ...}` per line in a `.jsonl`.
- **Every arm of a round gets the same request list**: same prompts, same
  order, same model mix. The list changes between rounds and levels and is
  the same on a rerun.
- **`/v1/completions` by default**, so the text is the prompt and no chat
  template is involved. Two engines that render one conversation through two
  templates prefill two different prompts. `-api chat` is there for a row
  that is about the chat route, and the prompt-token column then has to
  agree between arms before the row means anything.
- **`max_tokens` with `ignore_eos: true`, always.** Otherwise each engine
  stops wherever its own sampling hits an end token, and the row compares
  generation lengths, not engines. Check the at-max column; an arm below 100%
  did not honour it, and its row is not comparable. `-no-ignore-eos` exists
  for an engine that refuses the field: it is not a fair row, so report it as
  such. An arm's `extra` in a `-config` file can rename the field or remove it
  (`"ignore_eos": null`).
- **Temperature 0 by default**, so every engine is greedy and the sampler's
  cost is the same. jitllm's shims default to the APIs' documented
  temperature 1 when a request names none (as every other engine does).
  The harness always names one, so the default does not affect a harness
  row; it matters for third-party clients.

## A fair arm

Each of these is a way a row has gone wrong before (RULE 1):

- **Same model file and quantization.** jitllm runs a `.jlm` converted from
  the GGUF llama-server runs (`jitllm convert model.gguf model.jlm`). vLLM,
  SGLang and mistral.rs run safetensors or their own quantizations: a Q4_K_M
  against a bf16 or an AWQ is a comparison of two models, so it is labelled
  with both formats or not taken. Where an engine can read the same GGUF
  (llama-server always, vLLM and mistral.rs for some architectures), use it.
- **Same context.** jitllmd's `-max-seq` (default: the model's context)
  against llama-server's `-c` (divided across `-np` slots), vLLM's
  `--max-model-len`, SGLang's `--context-length`. `/v1/models` on jitllmd and
  vLLM reports `max_model_len`.
- **Same concurrency capacity.** llama-server serves at most `-np` requests at
  once and queues the rest, so `-np` must be at least the highest level swept
  (and `-c` is shared among the slots). jitllmd's `-max-batch` bounds how many
  of one model's generates decode as rows of one step. vLLM's `--max-num-seqs`
  and SGLang's `--max-running-requests` are the same knob.
- **Prefix caching off, or labelled.** jitllm's HTTP routes have no prefix
  cache yet. vLLM (`--no-enable-prefix-caching`), SGLang
  (`--disable-radix-cache`) and llama-server (`--cache-reuse 0`; its slot
  prompt cache reuses a slot's own previous prompt) reuse prefixes across
  requests, and generated prompts share little but the harness's prompt file
  might share a lot. Turn it off, or name it in the row.
- **Same hardware, cgroup and core set.** Every arm runs under the same
  `scripts/cap` limit, with the same `taskset`. Run them one at a time on the box, never
  beside each other, unless the row is about co-tenancy. A GPU arm names its
  backend (CUDA, Vulkan, Metal) beside its number. A jitllm CUDA row against a
  llama.cpp Vulkan row is not a comparison.
- **The harness is a process too.** Run it on cores the servers are not
  pinned to (`taskset` it apart), or on another machine over a quiet link,
  and say which.
- **Warm.** `-warmup` (default 2) requests go to each arm before the sweep.
  An engine that compiles or captures graphs on first use (vLLM, SGLang,
  jitllm's tuners) has to be past that before a round is timed.

Starting each engine, one model (jitllmd on its default port 4242, the others on 8080):

    # jitllmd
    ./scripts/cap 24G -- taskset -c 0,2,4,6,8,10 \
      jitllmd serve -addr 127.0.0.1:4242 -models $JITLLM_MODELS -load model.jlm -id model -max-batch 64

    # llama.cpp
    ./scripts/cap 24G -- taskset -c 0,2,4,6,8,10 \
      llama-server -m model.gguf --port 8080 -np 64 -c $((64*4096)) -t 6 --alias model

    # vLLM
    ./scripts/cap 24G -- vllm serve <model> --port 8080 --served-model-name model \
      --max-num-seqs 64 --no-enable-prefix-caching

    # SGLang
    ./scripts/cap 24G -- python -m sglang.launch_server --model-path <model> --port 8080 \
      --served-model-name model --max-running-requests 64 --disable-radix-cache

    # mistral.rs and ZML: their OpenAI-compatible servers, on the same port, with
    # the model served as `model`; check the at-max column for ignore_eos.

## Running

A single endpoint, a sweep:

    go run ./dev/httpbench -arm jitllm=http://127.0.0.1:4242 -model model \
      -c 1,2,4,8,16,32,64 -max-tokens 128 -prompt-lens 128,512 -o jitllm.json

**The A/A self-control first (RULE 2).** One endpoint as two arms, on the
harness and box the A/B will use:

    go run ./dev/httpbench -aa -arm jitllm=http://127.0.0.1:4242 -model model \
      -rounds 8 -c 1,8,32

Every ratio must read 1.0 inside its IQR. If it does not, this harness cannot
resolve an A/B on this box at that level, whatever the A/B prints.

Then the comparison, two servers up at once on different ports (each
idle while the other is measured, since a round runs one arm, then the
other):

    go run ./dev/httpbench -arm jitllm=http://127.0.0.1:4242 -arm llama=http://127.0.0.1:8081 \
      -model model -rounds 8 -c 1,8,32

Rounds are interleaved per level (A B A B ...), each round sends the same
list to both arms, and each comparison is the median of the per-round ratios
of aggregate tok/s, TTFT p50 and ITL p50, each with IQR/median. A ratio is
**refused** at fewer than 6 rounds, at IQR/median above 0.10, or when a round
lost a request on either arm. The first arm is the base of every ratio.

Leaving two idle servers resident means both hold their memory for the whole
run. On a box where that does not fit, or where an idle engine still spins
(a polling scheduler), stop and start them between rounds by hand, and do not
treat the run as an interleaved one.

Output: the table on stdout, and `-o` (default `httpbench.json`), which has the
config, every level of every round with every sample (including each ITL gap
and the protocol it came over), and the reduced report.

`-http2` speaks HTTP/2: h2c with prior knowledge on `http://`, ALPN on
`https://`. jitllmd answers both; llama-server and vLLM speak HTTP/1.1, so
an HTTP/2 row is jitllm-only unless a proxy is in front of the others, and
then the proxy is part of the arm.

## Several models at once

`-model NAME:WEIGHT`, repeated, spreads requests across models by weight;
each request's model is drawn from the round's seed, so every arm gets the
same mix. The table prints every model's own row under the level's total,
each model's rate over the level's whole wall time (its share of what the
box produced while all of them ran).

jitllmd serves every model from one process: load the others into the
running daemon (`jitllmd models -load other.jlm -id other`) and point the
arm at it. vLLM, SGLang, llama-server and mistral.rs serve one model per
process, so their arm is **one server per model on the same box, under the
same cgroup memory limit** as the jitllm arm, each routed:

    go run ./dev/httpbench \
      -arm jitllm=http://127.0.0.1:4242 -arm vllm=http://127.0.0.1:9000 \
      -model small:3 -model big:1 \
      -route vllm/small=http://127.0.0.1:9000#Qwen3-0.6B \
      -route vllm/big=http://127.0.0.1:9001#Qwen3-8B \
      -rounds 8 -c 4,16

The harness cannot tell a multi-process arm from a single process, so the
row has to say so. Name it in the row
("vLLM, two processes, one per model, one 48G cgroup"), because it is a
different system from one process that pages blocks between its models.
The two are compared at what the box does under one memory limit, and the
row is not about how each engine runs one model. The engines that size their
KV cache as a fraction of device memory (vLLM's `--gpu-memory-utilization`,
SGLang's `--mem-fraction-static`) need that fraction divided among their
processes, or the second process does not start.

A `-config FILE` holds the same thing as JSON (`hb.Config`): arms with
per-model routes and per-arm extra body fields, the model mix, levels,
rounds, the workload and the SLOs.

## What the jitllm side still differs in

- **No prefix cache on the HTTP routes**: a request's prompt is prefilled
  whole, every time. An engine with one on is faster on a shared prefix for
  a reason that is not the kernel. Turn it off (above).
- **Admission**: a generate on a device model becomes a row of that model's
  step loop up to `-max-batch` rows, and a host-only model's sessions
  interleave on the one shared pool a step at a time. Neither is a
  continuous-batching scheduler with chunked prefill across requests in the
  way vLLM's is, so TTFT under load is where the two differ most. That is
  the engine's behaviour, not something in the harness.
- **Logprobs, n > 1 and presence/frequency penalties are refused (400)**,
  where other engines produce them. The harness asks for none of them.

## Deployment settings search

`dev/deploysearch` sweeps jitllmd serve flags (`-max-batch`, `-prompt-chunk`,
`-step-cost`, `-fairness`, `-sessions`, `-mem-cache-max`, and any other flag
given with `-knob FLAG=V1,V2` or as a bundle with `-combo`) on a named request
mix (`mixed`, `long`, `decode`, or a `-mix-file` hb.Config). Each sample
restarts the server, so every candidate is measured from a cold process, and
rounds are interleaved over the candidates (A B C, then C B A) with the
unchanged default as the control and an identical copy of it ("aa") as the
self-control. It reports TTFT p99, inter-token p99, goodput under the mix's
SLO and host plus device memory as ratios to the control, gated as RULE 2
gates (6 rounds, IQR/median at most 0.10), and the Pareto set of medians.
`-bin NAME=PATH` repeated compares builds (for example main against an
integration branch): each build's default joins the sweep. It changes no
default and prints "candidate" only for a gated win without a gated
regression; that is a reason to repeat, not to change a default. A timing row
on the shared GPU box runs under that card's lock, both arms on one card.

### First deployment search: Tesla V100-SXM2-16GB, dense and mixture

Host: the shared 2x Xeon E5-2680 v4 box, one V100 (CUDA, `-devices cuda`,
`-max-seq 3072 -kv-f16`), taken under that card's lock with both arms on the
card in one pass. Host load was not quiet: load average 19.7 at the start of
the dense pass and 6.0 at its end, 20.7 at the end of the mixture pass (other
projects' jobs; the box has 56 logical CPUs, the servers were not pinned).
Build: main at the commit the tool landed on. Mix `mixed`: concurrency 4, 16
requests, prompts of 32, 128 and 512 words, 64 tokens, goodput SLO TTFT 3 s and
mean inter-token gap 100 ms. Six interleaved rounds (A B C ... then reversed),
every sample a fresh server and 2 warm-up requests, the unchanged default as
control, an identical copy of it as the A/A. Models: Llama-3.1-8B-Instruct
Q4_K_M (dense) and OLMoE-1B-7B-0924-Instruct Q4_K_M (mixture). Candidates:
`-max-batch 8`, `-prompt-chunk 128/512`, `-step-cost 2/4`, `-fairness 0/50`,
`-sessions 4`, `-mem-cache-max 256M`.

**Result: nothing resolved, on either model.** Every candidate's every ratio
was refused by the gate (IQR/median above 0.10). The A/A is the reason: the
identical copy of the control read TTFT p99 1.10 [IQR/median 0.20] and
inter-token p99 1.12 [0.86] on the dense model, and 0.86 [0.34] and 1.40
[0.54] on the mixture. This harness at this mix cannot resolve a ratio on this
box, so no setting is a candidate and no default has a reason to change.

| median over 6 rounds, V100 | TTFT p99 (ms) | inter-token p99 (ms) | goodput (req/s) | host+device MiB |
|---|---|---|---|---|
| dense, default | 20088 | 1997 | 0.01 | 12718 |
| dense, `-sessions 4` | 20906 | 2457 | 0.01 | 12862 |
| mixture, default | 18605 | 2444 | 0.01 | 10512 |
| mixture, `-sessions 4` | 18992 | 2075 | 0.04 | 10142 |

The priority question, whether `-sessions 4` beats the default `-sessions 1`
under concurrency 4: on the dense model its TTFT p99 ratio was 0.99
[0.24], inter-token p99 1.17 [0.45], goodput 0.19 [0.00] (every one refused);
on the mixture 0.92 [0.50], 0.85 [0.36], 1.99 [0.00] (refused). Not
answered at this mix.

What the numbers do say, unpaired: at concurrency 4 every configuration, the
default included, had a TTFT p99 near 20 s and a goodput near 0.01 requests
per second against a 3 s SLO, and aggregate throughput of 12-17 tok/s, where
one stream of the same dense model on the same card decodes at about 138 tok/s
(`jitllm run`). The SLO is unreachable in this regime, so goodput is a count of
a handful of requests and carries no information; a sweep that is going to
resolve anything needs a mix the default meets its SLO on, and a warm-up that
takes a restarted server past its first-use tuning and compilation (two
requests do not). That is the next step, not a conclusion about any setting.
Per-sample logs and `results.json` were kept on the box.

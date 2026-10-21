# jitllm — the scoreboard and how it got there

> **EVIDENCE, NOT INSTRUCTIONS.** This file records what was measured, what was
> tried, and what was wrong. It is the reason a rule exists, not the rule.
> **Binding requirements live only in `AGENTS.md`** -- if something here reads as
> an instruction, `AGENTS.md` is what governs. Numbers here are dated by the
> commit that added them; treat any of them as provisional until re-measured
> (RULE 1a: a baseline pinned to a version rots silently).
>
> Split out of a single 415 KB AGENTS.md after a review
> identified the file's real failure as *unresolved authority* -- historical
> findings, current instructions and withdrawn conclusions all speaking in the
> same imperative voice.

## The prompt rows and their boundaries

`scripts/vs-llamacpp.sh` takes three prompt rows, and each engine's clock
starts and stops where its own output says, never at a guessed point:

| row | jitllm | llama.cpp |
|---|---|---|
| warm prefill, ppN, no prefix reuse (`JITLLM_MODE=prefill`, the default prompt row) | `jitllm speed -p N -n 0 -r 1`: one untimed warm-up prompt, then the timed one on a fresh State; `warmHonest` refuses a sample that restored a position, ran fewer than N rows or did not read the last row's logits back | `llama-bench -p N -n 0 -r 1`, which runs its own warm-up |
| load (`JITLLM_MODE=load`) | `speed -ttft`'s `load`: container open + device init and kernel compile + attach (placement, upload, the split tuner, the KV pool and the first prompt's kernels) | `llama-completion`'s own `load time` |
| ttft, model loaded (`JITLLM_MODE=ttft`) | `speed -ttft`'s `ttft-loaded`: ready to the first token of the process's first prompt, tokenizing excluded; refused if a kernel compiled or a tuner trial ran after ready (`readyHonest`) | `llama-completion --no-warmup`'s `prompt eval time`, its first decode after load |

llama.cpp compiles nothing at run time; jitllm compiles only before ready, and
`model.TestFirstPromptDoesNotGrowTheKVPool` holds the first prompt to zero
compiles (six without the prewarm on Llama-3.2-1B on an sm_86 card: the
embedding gather, the folded head's row copy and the fused binary16 operand
kernels). The model file is in the page cache after the harness's token-count
probe; that serves llama.cpp's mmap, while jitllm reads the container with
O_DIRECT and gets nothing from it. The split tuner's answers persist
(`tier.WithTuneCache`, keyed by device UUID, driver version and executable),
so a second process on the same card times nothing; on the RTX 3050 Ti
(Llama-3.2-1B, pp415, CUDA) that took attach from 1289 ms to 313 ms and
`speed -ttft`'s open-to-first-token from 2152 ms to 941-1115 ms.

## Decision models: `/v1/systemone`, jitllm against llama.cpp

The first latency rows for the decision models (`docs/design/decision-models.md`,
sections 5 and 6). The harness is `dev/decisionbench`: a stream of single-word
requests, each a `POST /v1/systemone` with a noul question ("Is this the name of
a centipede?") and the word as the state, against any server that speaks the
endpoint. The same labelled file and the same question go to both engines
(`dev/decisionbench/testdata/centipedes-large.tsv`: 233 words, 104 centipede
genera, species, orders and common names against 129 millipedes, other
arthropods, near-miss spellings and plain words).

**What was measured.** jitllm: `jitllmd serve` built from main at 585c3e9f
(`-load MODEL.jlm -devices cpu` or `-devices cuda:0`, every other flag its
default), the `.jlm` converted from the GGUF the other arm reads. llama.cpp:
`llama-server` b11514 (`-np 1 -c 8192 -t 12`; `-ngl 0` on the CPU rows, `-ngl 99`
on the CUDA rows), the official CPU release for the CPU rows and the same tag
built from source with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70` (CUDA 12.0)
for the CUDA rows. Both read the same file: Laya BF16 (`ggml-org/Laya-GGUF`,
ModernBERT, 421M), d1-3B Q8_0 (`LiquidAI/d1-3B-GGUF`, LFM2, 2.7B), Lev 4B Q8_0
(`ggml-org/lev-GGUF`, Qwen3.5 with the adapter merged). All three run on both
engines; none is missing.

**Hosts.** A Xeon E5-2680 v4 server (2 sockets, 14 cores each, hyper-threads on,
no other work on it, load average 0.0 before the run, `/proc/pressure/cpu` some
avg10 0.00 before every A/B run). CPU rows: one socket, cores 0-11, under
`numactl --membind=0`, 12 threads in both engines (`llama-server -t 12`; jitllm's
pool takes the taskset's width). CUDA rows: one Tesla V100-SXM2-16GB (sm_70),
both engines on that card (`CUDA_VISIBLE_DEVICES=0`, the process still pinned to
the same cores), the other seven cards idle. Benchmark hosts run bare (no
`scripts/cap`).

**Method (RULE 1, RULE 2).** Per model and backend, serially, nothing else
running: both servers loaded and warmed; then the A/A self-control on the same
harness (two `jitllmd` instances, same model); then one full pass of the file per
engine for accuracy; then two complete A/B runs of 20 rounds. A round is one
timed sample per arm, the order alternating by round (AB, BA, ...), each sample
`decisionbench -cycle` over a window after a discarded one-second burst. The
server not being timed is SIGSTOPped, so an idle pool's spinning workers share no
core with the arm under test. Reported: the median of the 20 per-round ratios and
its IQR/median (reject above 0.10; none came near). The ratios read as jitllm over
llama.cpp, above 1 meaning jitllm is faster: the p50 ratio is llama.cpp's p50 over
jitllm's, the words ratio is jitllm's words in the window over llama.cpp's.

**"Words in 32 s" is scaled, not measured over 32 s.** A 32-second window per
sample would make a 20-round A/B run an hour, so the window is 8 s (Laya, and
every CUDA row), 12 s (d1 on the CPU) or 16 s (Lev on the CPU) and the count is
scaled by 4, 2.67 or 2. The count is an integer in a short window (Lev on the CPU
answers about nine words in 16 s), so its ratio is coarse: 1.125 is 18 words
against 16. The p50 is the finer number.

### CPU: a Xeon E5-2680 v4 server, one socket, 12 threads (jitllm CPU backend, llama.cpp CPU backend)

| model | jitllm p50 | llama.cpp p50 | p50 ratio, run 1 / run 2 (IQR/median) | words in 32 s, jitllm / llama.cpp | words ratio, run 1 / run 2 | A/A p50 ratio, words ratio |
|---|---|---|---|---|---|---|
| Laya BF16 | 81.5 ms | 118.7 ms | 1.455 / 1.456 (0.007, 0.010) | 376-380 / 264 | 1.428 / 1.439 | 1.006, 1.011 |
| d1-3B Q8_0 | 323 ms | 398 ms | 1.232 / 1.234 (0.006, 0.005) | 96 / 77 | 1.241 / 1.241 | 0.997, 1.000 |
| Lev 4B Q8_0 | 1.665 s | 1.955 s | 1.174 / 1.175 (0.006, 0.004) | 18 / 16 | 1.125 / 1.125 | 0.971, 1.000 |

The Lev A/A control reads 0.971 with an IQR of 0.003: the second `jitllmd`
instance is 3% faster than the first, a property of the instance and not of the
slot, since the order alternates. The A arm of every A/B run is the first
instance, so Lev's 1.17 carries that 3% against jitllm, if the bias transfers.
The other controls are within 1.1% of 1.

### CUDA: the same server's V100, one card (jitllm CUDA backend, llama.cpp CUDA backend)

| model | jitllm p50 | llama.cpp p50 | p50 ratio, run 1 / run 2 (IQR/median) | words in 32 s, jitllm / llama.cpp | words ratio, run 1 / run 2 | A/A p50 ratio, words ratio |
|---|---|---|---|---|---|---|
| Laya BF16 | 37.0 ms | 14.0 ms | 0.378 / 0.379 (0.002, 0.002) | 860-864 / 2204-2208 | 0.390 / 0.391 | 1.002, 1.000 |
| d1-3B Q8_0 | 29.9 ms | 16.3 ms | 0.544 / 0.544 (0.005, 0.004) | 1092-1096 / 1956 | 0.560 / 0.558 | 1.002, 1.000 |
| Lev 4B Q8_0 | 70.8 ms | 53.8 ms | 0.761 / 0.755 (0.018, 0.029) | 436-438 / 584 | 0.750 / 0.747 | 1.008, 1.000 |

On the card jitllm is slower than llama.cpp on every model, by 2.6x (Laya), 1.8x
(d1) and 1.3x (Lev). Each of the six runs reproduces the other to within 1% on
the ratio. The gap shrinks as the model grows, which is what a per-request cost
that does not scale with the model looks like (30-37 ms is jitllm's floor on the
two small models, and Laya and d1 differ ten-fold in size). **This was not
diagnosed**: it was not checked how many blocks the card held, where the
request's time goes, or whether the readout (the head's logits come home whole,
`decision-models.md` section 10) is the cost. The A/A controls say the harness
sees nothing of it; the rows are a work list, RULE 6 ordering and not a verdict.

### Accuracy, caught and wrong picks (one full pass of the 233 words, threshold P(yes) 0.5)

| backend | model | jitllm accuracy | caught | wrong picks | llama.cpp accuracy | caught | wrong picks |
|---|---|---|---|---|---|---|---|
| CPU | Laya BF16 | 0.562 (131/233) | 26/104 | 24 | 0.567 (132/233) | 27/104 | 24 |
| CPU | d1-3B Q8_0 | 0.815 (190/233) | 79/104 | 18 | 0.824 (192/233) | 80/104 | 17 |
| CPU | Lev 4B Q8_0 | 0.815 (190/233) | 76/104 | 15 | 0.824 (192/233) | 78/104 | 15 |
| CUDA | Laya BF16 | 0.562 (131/233) | 26/104 | 24 | 0.562 (131/233) | 26/104 | 24 |
| CUDA | d1-3B Q8_0 | 0.824 (192/233) | 80/104 | 17 | 0.828 (193/233) | 82/104 | 18 |
| CUDA | Lev 4B Q8_0 | 0.815 (190/233) | 76/104 | 15 | 0.815 (190/233) | 76/104 | 15 |

The engines agree to within two words of 233. On the CPU rows for Laya and d1 the
probabilities were compared word by word: the largest difference is 0.025 (Laya,
mean 0.003) and 0.025 (d1, mean 0.004), and the answers that flip at 0.5 are
near-ties (Laya: Scutigera 0.490 against 0.507; d1: Julus 0.507 against 0.500 and
Parascutigera 0.496 against 0.519), which is the f32 reduction-order band, not a
different function. Lev's two-word CPU gap was not compared word by word. Laya's
accuracy is low (0.56, 26 of 104 centipedes caught) on both engines alike, so it
belongs to the model on this file and question, not to an engine; it says
nothing against the speed rows.

### Sanity of llama.cpp's absolute rate

`llama-bench` on the same d1 GGUF, same pinning: 90.2 +- 3.2 tokens/s for a
64-token prompt on the CPU (12 threads) and 2324 +- 388 on the V100. A d1 request
is 31-33 tokens, which is 355 ms and 14 ms of prompt processing; the server's p50
is 398 ms and 16.3 ms, the difference being the HTTP request, the template and the
answer arithmetic. The numbers are consistent with the host. There is no bytes per
token or roofline figure: a request is a short prefill and the engine is
compute-bound, not weight-bandwidth-bound.

### CUDA after the device fixes: where a request's time went, and the rows

The rows above were not diagnosed when they were taken. Every block was on the
card for all three models (Laya 30 of 30, the 28 encoder blocks and the 2 head
blocks; d1 30 of 30; Lev 32 of 32), so placement was not the gap. Measured with
nvprof and a Go CPU profile of `jitllm decide` on the V100, a request's time
went to:

| model | where the time went | fix |
|---|---|---|
| d1-3B Q8_0 | about 13 ms of kernels, and about 14 ms more whenever the prompt crossed a 32-row grain: the words give 31-34 token prompts, which run at 32 or 64 rows, and the tier kept one exact-width batched scratch, rebuilt (allocation, a compile-cache lookup that gob-hashes every kernel's IR, free) at each crossing | a lane keeps up to four exact widths, least recently used given back |
| Lev 4B Q8_0 | about 30 ms of a 70 ms request resetting the recurrent state: 96 pageable 2 MiB zero uploads, one per half of every delta-rule block | the zeros go up once and each half is copied on the card |
| Laya BF16 | about 32 ms in the bfloat16 matvecs: sm_70 had no tensor-core batched kernel for a float weight (the FMA tile ran the 1024x2624 projection at 12 GB/s), and the staged GEMM waited on the integer MMA's refusal, which a float-only model never asks for | GemmVolta stages a two-byte float weight as binary16; 64-row tiles for ModernBERT's 2624; an encoder block skips the int8 quantize its f16 twins never read |

Not done, and their measured size: the vocabulary head still runs whole before
the label pick (0.8 ms of a Lev request, 0.33 ms of d1's); Lev's 102-104 token
prompts run at 128 rows (no sm_70 tile is narrower than 32 tokens, so no width
between 96 and 128 exists); a CUDA graph over Laya's fixed-width encoder call
was tried, replayed with the same answers and moved the request 12.9 to 12.5-12.65
ms, inside the noise: the card is busy, not the launches, so it was not kept.

Same method, host, card, engines' versions and pinning as the rows above
(jitllmd built from this branch; llama.cpp b11514 CUDA, `-ngl 99`), the A/A
control first, then two A/B runs of 20 rounds, 8-second windows.

| model | jitllm p50 | llama.cpp p50 | p50 ratio, run 1 / run 2 (IQR/median) | words ratio, run 1 / run 2 | A/A p50 ratio, words ratio |
|---|---|---|---|---|---|
| Laya BF16 | 13.4 ms | 14.1-14.3 ms | 1.060 / 1.054 (0.021, 0.035) | 1.100 / 1.094 | 1.000, 0.998 |
| d1-3B Q8_0 | 13.5-13.9 ms | 16.3 ms | 1.215 / 1.176 (0.059, 0.091) | 1.154 / 1.153 | 1.000 (IQR/median 0.139: rejected), 0.998 |
| Lev 4B Q8_0 | 45.5-46.5 ms | 53.6 ms | 1.182 / 1.151 (0.020, 0.014) | 1.191 / 1.150 | 1.007, 1.006 |

d1's p50 A/A control is rejected (IQR/median 0.139): a d1 request runs at 32 or
64 rows by its word, so its per-window median sits on either side of a
two-valued distribution. Its words ratio (A/A 0.998, IQR/median 0.002) is the
row to read for d1. The load average read 6-11 during Laya's A/B runs (PSI
some avg10 at most 1.15); its A/A control was taken in the same pass.

Accuracy on the 233 words is unchanged on CUDA: jitllm Laya 131/233, d1
192/233, Lev 190/233, each as before; llama.cpp the same as before.

The CPU rows, taken again with this branch's binary: Laya 1.469 / 1.472, d1
1.221 / 1.220, Lev 1.174 / 1.173 (p50 ratios; A/A 1.027, 0.997, 0.988), against
1.455, 1.232, 1.174 before: the changes are CUDA-side, and the CPU rows did
not move beyond their controls.

### What this does not cover

Serial single requests only: no concurrent clients, no multi-question requests.
`scripts/corebusy.py` and the clock coherence gate were not run (the box was
otherwise idle and its load was read before each run). The CPU rows used main at
585c3e9f; main has since moved (the row-reduction emitters, among others), so a
CPU row quoted later wants a re-measure (RULE 1). Laya's CUDA placement, and
whether jitllm placed every block of d1 and Lev on the card, were not
inspected.

### The commands

On the laptop, build and ship binaries only (no source goes to the host):

    CGO_ENABLED=0 ./scripts/cap 8G -- go build -trimpath -o jitllm ./cmd/jitllm
    (cd server && CGO_ENABLED=0 ../scripts/cap 8G -- go build -trimpath -o ../jitllmd ./cmd/jitllmd)
    CGO_ENABLED=0 ./scripts/cap 8G -- go build -trimpath -o decisionbench ./dev/decisionbench
    scp jitllm jitllmd decisionbench dev/decisionbench/testdata/centipedes-large.tsv \
        scripts/decisionbench-ab.sh scripts/decisionbench-cell.sh scripts/abstats.py HOST:dbr/

On the host: the models come from Hugging Face (`ggml-org/Laya-GGUF`
`Laya-BF16.gguf`, `LiquidAI/d1-3B-GGUF` `d1-3B-Q8_0.gguf`, `ggml-org/lev-GGUF`
`lev-Q8_0.gguf`), converted by the shipped binary; llama.cpp is the b11514 release
(`llama-b11514-bin-ubuntu-x64`, unpacked to `~/dbr/llama-cpu`) and, for CUDA, the
b11514 tag built on the host:

    ./jitllm convert Laya-BF16.gguf /mnt/models/jitllm/dbr/Laya-BF16.jlm      # and d1, lev
    git clone --depth 1 --branch b11514 https://github.com/ggml-org/llama.cpp && cd llama.cpp
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
    cmake --build build -j 24 --target llama-server llama-bench

    cd ~/dbr
    ./decisionbench-cell.sh cpu  Laya-BF16 8  20      # A/A, accuracy, two A/B runs
    ./decisionbench-cell.sh cpu  d1-3B-Q8_0 12 20
    ./decisionbench-cell.sh cpu  lev-Q8_0  16 20
    ./decisionbench-cell.sh cuda Laya-BF16 8  20      # and d1-3B-Q8_0, lev-Q8_0
    python3 abstats.py res/cpu-Laya-BF16.ab1.tsv 8   # the medians and IQR above

Each server inside the cell is `numactl --membind=0 taskset -c 0-11 jitllmd serve
-models DIR -load MODEL.jlm -id MODEL -devices cpu|cuda:0` or `llama-server -m
MODEL.gguf -t 12 -c 8192 -np 1 -ngl 0|99`; one request by hand is
`decisionbench -url http://127.0.0.1:PORT -model MODEL -file centipedes-large.tsv
-cycle -window 8s`.

## ★ THE FINAL BOARD: THREE HOSTS, FIVE ENGINES, ONE PASS PER HOST

The M4 Metal 3B-7B section below has its own method (a paired harness and a
separate three-engine pass with MLX) and says so.

Every number names its host and backend. Ratios are jitllm / llama.cpp from
the same pass on the same GGUF (jitllm reads the `.jlm` converted from it).
jitllm is the better of its two `speed -r 2` runs (the second is warm: the
tuners have settled); llama.cpp is `llama-bench`'s mean over `-r 3` (which
discards its own warm-up). pp512 / tg128 unless a row says otherwise.

### V100 (an 8x V100 server: 2x Xeon E5-2680 v4, 8x Tesla V100-SXM2-16GB sm_70, CUDA)

jitllm `speed -devices cuda -p 512 -n 128 -r 2` (a bare `cuda` was cuda:0
alone when this board was taken; `-devices cuda:0` is that run) against llama.cpp built for
sm_70 (`llama-bench -ngl 99 -r 3`), back to back per model, 36 models,
one pass (board9, 5fc2765: the pipelined multi-card prompt, the cold-start
release, sessions together). Against board8 (cbe107d), row for row: jitllm's
prompt -5% to +2% and decode -1% to +1%, llama.cpp's prompt -8% to +3% and
decode -1% to 0%, so the two boards are the same engine within the box's
pass-to-pass spread. A 512-token prompt is one piece, so the pipeline does not
reach this column; its gain is in the multi-card section below.

| model | GPUs | pp512 | llama.cpp | ratio | tg128 | llama.cpp | ratio |
|---|---|---|---|---|---|---|---|
| SmolLM2-360M Q8_0 | 1 | 33613 | 24361 | 1.38 | 492.2 | 405.5 | 1.21 |
| Qwen3-0.6B Q8_0 | 1 | 23980 | 20217 | 1.19 | 422.2 | 381.1 | 1.11 |
| gemma-3-1b | 1 | 17324 | 14238 | 1.22 | 331.8 | 273.2 | 1.21 |
| Llama-3.2-1B | 1 | 21062 | 16081 | 1.31 | 530.1 | 464.6 | 1.14 |
| SmolLM2-1.7B | 1 | 13129 | 10147 | 1.29 | 376.5 | 338.8 | 1.11 |
| Qwen2.5-1.5B | 1 | 12504 | 11548 | 1.08 | 359.2 | 285.6 | 1.26 |
| gemma-2-2b | 1 | 9869 | 8177 | 1.21 | 246.3 | 208.1 | 1.18 |
| Qwen3-1.7B Q8_0 | 1 | 12468 | 11961 | 1.04 | 266.8 | 250.9 | 1.06 |
| Llama-3.2-3B | 1 | 8022 | 6279 | 1.28 | 244.6 | 220.7 | 1.11 |
| Phi-3.5-mini | 1 | 5643 | 5661 | 1.00 | 211.2 | 199.5 | 1.06 |
| Phi-4-mini | 1 | 6793 | 6051 | 1.12 | 201.1 | 198.2 | 1.01 |
| gemma-3-4b | 1 | 6924 | 5735 | 1.21 | 172.8 | 154.4 | 1.12 |
| Qwen3-4B | 1 | 6670 | 5170 | 1.29 | 190.8 | 168.5 | 1.13 |
| olmoe-1b-7b (MoE) | 1 | 10922 | 4152 | 2.63 | 459.9 | 390.5 | 1.18 |
| Mistral-7B v0.3 | 1 | 4112 | 3248 | 1.27 | 144.6 | 133.3 | 1.08 |
| Qwen2.5-7B | 1 | 4205 | 3400 | 1.24 | 144.5 | 131.5 | 1.10 |
| Llama-3.1-8B | 1 | 4102 | 3229 | 1.27 | 136.6 | 125.8 | 1.09 |
| Qwen3-8B | 1 | 3793 | 3080 | 1.23 | 128.0 | 119.8 | 1.07 |
| gemma-2-9b | 1 | 3058 | 2491 | 1.23 | 98.3 | 92.3 | 1.06 |
| gemma-3-12b | 1 | 2484 | 2010 | 1.24 | 77.9 | 75.2 | 1.03 |
| Qwen3-14B | 1 | 2315 | 1774 | 1.31 | 78.7 | 72.4 | 1.09 |
| phi-4 | 1 | 2289 | 1847 | 1.24 | 78.8 | 74.5 | 1.06 |
| gpt-oss-20b MXFP4 (MoE) | 1 | 4820 | 2714 | 1.78 | 191.1 | 170.7 | 1.12 |
| DeepSeek-V2-Lite (MoE, MLA) | 1 | 4561 | 2089 | 2.18 | 183.8 | 171.4 | 1.07 |
| DeepSeek-Coder-V2-Lite | 1 | 4518 | 2026 | 2.23 | 184.1 | 171.3 | 1.07 |
| Moonlight-16B-A3B | 1 | 4632 | 2121 | 2.18 | 176.3 | 131.0 | 1.35 |
| gemma-3-27b | 2 | 1020 | 902 | 1.13 | 39.2 | 38.7 | 1.01 |
| Qwen3-30B-A3B (MoE) | 2 | 3761 | 1076 | 3.50 | 167.4 | 153.5 | 1.09 |
| Qwen3-32B | 2 | 1051 | 796 | 1.32 | 37.3 | 34.5 | 1.08 |
| Mixtral-8x7B (MoE) | 3 | 1723 | 865 | 1.99 | 84.7 | 81.2 | 1.04 |
| Llama-3.3-70B | 4 | 457 | 371 | 1.23 | 18.3 | 17.5 | 1.05 |
| Qwen2.5-72B | 4 | 455 | 374 | 1.22 | 16.8 | 16.2 | 1.04 |
| Qwen3-Next-80B (hybrid MoE) | 4 | 2058 | 445 | 4.62 | 105.4 | 93.9 | 1.12 |
| Kimi-Linear-48B (hybrid, MLA) | 4 | 3046 | -- | -- | 111.9 | -- | -- |
| gpt-oss-120b MXFP4 (MoE) | 5 | 2257 | 1235 | 1.83 | 130.2 | 117.5 | 1.11 |

Prefill is ahead on 34 of 35 rows llama.cpp runs and at parity on the 35th
(Phi-3.5-mini, 5643 against 5661 tok/s); decode is ahead on all 35.

Kimi-Linear's prompt read 112 tok/s on cbe107d, against 2961 on board6 with
the same placement: the decode plan, built for every call at the scratch's
full width, sized its split count by the decode heuristic, and a 512-row MLA
chunk at 8 splits wanted 302 MB of partials on cards the weights fill. A
staged plan that does not fit now halves its splits
(`TestAStagedPlanThatDoesNotFitHalvesItsSplits`).

Kimi-Linear has no llama.cpp row because it exists only as HF safetensors
(jitllm converts those directly).

**mistral.rs and ZML have no V100 row, for measured reasons.** mistral.rs
v0.9.4 ships no binary below sm_80 and its source build fails on bf16 kernels
under sm_70. ZML's CUDA path is built on CUDA 13, which dropped Volta.

### Time to first token (V100, one card, same pass)

A request is text in and one token out, so the time includes tokenizing it.
The same prose, repeated to 48, 531 and 2049 tokens (Qwen3's tokenizer: 47,
530, 2048), the same Q4_K_M GGUF weights, card 0 and cores 0-13, page cache
warm:
- jitllm `speed -ttft -r 7` (8caf902): a fresh session on the loaded model --
  tokenize, attach, prefill, sample. Cold is a `jitllm run -n 1` process,
  wall time from outside.
- llama.cpp b10472 built for sm_70: `llama-server --no-cache-prompt`,
  `n_predict: 1`, timed at the client (the server's own prompt_ms is about
  1-2 ms less). Cold is a `llama-completion -n 1` process.
- vLLM 0.18.1: `LLM(...).generate(max_tokens=1)`, prefix caching off, its
  defaults otherwise. Cold is a process that loads the model and generates
  one token. On sm_70 vLLM takes Triton attention (FlashAttention-2 needs
  sm_80) and its GGUF matmuls; Ampere with its native formats is its better
  case, and this row is not that.

Warm, median of 7, ms:

| model | prompt | jitllm | llama.cpp | vLLM |
|---|---|---|---|---|
| Llama-3.2-1B | 48 | 12.6 | 11.1 | 18.9 |
| | 531 | **40.3** | 39.2 | 150.8 |
| | 2049 | **118.7** | 127.8 | 646.1 |
| Llama-3.1-8B | 48 | 40.3 | **37.4** | 97.9 |
| | 531 | **166.6** | 182.8 | 1032.2 |
| | 2049 | **556.8** | 652.8 | 4292.7 |
| Qwen3-8B | 47 | 42.4 | **39.1** | 98.6 |
| | 530 | **174.1** | 193.9 | 1035.8 |
| | 2048 | **585.4** | 690.8 | 4345.2 |

Cold, a one-token process:

| model | jitllm | llama.cpp | vLLM |
|---|---|---|---|
| Llama-3.2-1B | 5.2-6.1 s | 4.7-5.8 s | 126-133 s |
| Llama-3.1-8B | 7.5-9.3 s | 6.3-7.1 s | 173 s |
| Qwen3-8B | 7.5-9.4 s | 6.2-7.0 s | 160-170 s |

jitllm reaches the first token 7-15% before llama.cpp from ~512 tokens up,
3-14% after it on a 48-token prompt (a fixed per-request cost, unattributed),
and starts 1.2-2.4 s later on the 8B models. Both need ~5 s on a 1B model on
this box, which the two share. One pass, and the jitllm and llama.cpp arms
partly overlapped gate runs on other cards and cores: a first table, not a
gated row. The llama.cpp server stayed loaded, idle, on the card while
jitllm's arm ran; these models fit twice in 16 GB, so neither was paged. A
model that does not fit twice is measured one engine at a time (below): with
gemma-3-27b split over two cards and the server resident, jitllm paged and
read 1194 ms where it reads 112 alone.

A fresh session re-read every block's host page before this (8caf902): warm
time to first token on Llama-3.2-1B, RTX 3050 Ti, was 167 ms and is 52.

### Time to first token, the big models (V100, 2-5 cards)

The same recipe split over the cards each model needs, each engine alone on
them: llama.cpp `-ngl 99` with its default layer split, jitllm `-devices
cuda:0,...`, vLLM tensor-parallel (Mixtral on four, since its heads do not
divide by three). Warm, median of 7, ms; cold is a one-token process at 512
tokens, two runs.

| model | cards | prompt | jitllm | llama.cpp | vLLM | jitllm cold | llama.cpp cold |
|---|---|---|---|---|---|---|---|
| gemma-3-27b | 2 | 48 | **109** | 169 | refused | 16.4 s | 10.3 s |
| | | 531 | **586** | 723 | | | |
| | | 2049 | 2082 | **1969** | | | |
| Qwen3-32B | 2 | 47 | **118** | 138 | 212 | 16.3-22.3 s | 10.5 s |
| | | 530 | **591** | 688 | 2276 | | |
| | | 2048 | 2089 | **1729** | 9480 | | |
| Qwen3-30B-A3B (MoE) | 2 | 47 | **67** | 74 | 93 | 28.5-29.2 s | 10.6-10.7 s |
| | | 530 | **202** | 526 | 211 | | |
| | | 2048 | **590** | 1677 | 983 | | |
| Mixtral-8x7B (MoE) | 3 | 54 | **100** | 223 | fails to load | 18.7-21.8 s | 12.9-13.4 s |
| | | 522 | **388** | 713 | | | |
| | | 2056 | **1314** | 2555 | | | |
| Llama-3.3-70B | 4 | 48 | 261 | 291 | **254** † | 26.1-26.3 s | 15.1-17.7 s |
| | | 531 | **1292** | 1408 | 2599 † | | |
| | | 2049 | 4682 | **3015** | 10474 † | | |
| Qwen2.5-72B | 4 | 47 | **267** | 303 | 277 † | 26.4-47.3 s | 15.6-16.2 s |
| | | 530 | **1317** | 1426 | 2827 † | | |
| | | 2048 | 4699 | **3080** | 11321 † | | |
| Qwen3-Next-80B (hybrid MoE) | 4 | 47 | 170 | **166** | no GGUF support | 75.6-83.7 s | 18.5-23.5 s |
| | | 530 | **423** | 1407 | | | |
| | | 2048 | **1104** | 4506 | | | |
| gpt-oss-120b (MXFP4 MoE) | 5 | 47 | **118** | 240 | refuses MXFP4 below sm_80 | 41.9-43.5 s | 22.1-25.6 s |
| | | 530 | **358** | 588 | | | |
| | | 2048 | **1080** | 1876 | | | |

vLLM refuses gemma3 in float16 ("numerical instability"), and a V100 has no
bfloat16. Its Mixtral GGUF load dies on an uninitialized parameter
(`GGUFUninitializedParameter`), and it has no qwen3next GGUF support. † The
70B and the 72B ran out of device memory at vLLM's defaults and loaded with
`enforce_eager` and 0.92 of the memory: no CUDA graphs, so those rows are
not its defaults. vLLM cold, one-token process: Qwen3-32B 250 s,
Qwen3-30B-A3B 193 s, Llama-3.3-70B 208 s, Qwen2.5-72B 199 s. On a 48-token
prompt over four cards vLLM's tensor parallelism is the quickest of the
three on the 70B (254 ms): every card works on every layer, where a layer
split runs the cards one after another.

jitllm reached the first token sooner on 20 of 24 jitllm/llama.cpp pairs in
this table. The four it lost were the dense models split over several cards
at ~2048 tokens (gemma-3-27b 1.06, Qwen3-32B 1.21, Llama-3.3-70B 1.55,
Qwen2.5-72B 1.53); pipelining the prompt across the cards wins all four now
(the next section). The cold columns are that binary too, before the host
stopped keeping a copy of what the cards hold (also below).

Llama-3.3-70B's 2049-token row read 58.5 s before the placement plan counted
what runs beside the blocks: it put 27/26/27/0 blocks on the four cards with
the third at its budget, the prompt's history ran out of pages, eviction
sent them home, and every chunk then went a row at a time (14,116 single-row
steps). The plan now counts the history for the session's context, the
output head and each card's fixed cost, and leaves a tenth of the budget, so
the model spreads over four cards: 4.68 s. Its row above is that binary
(cards 4-7, the other socket's cores, while vLLM ran on cards 0-1); every
other jitllm row is 8caf902 on cards 0-n. Qwen2.5-72B re-taken the same way
did not move (4699 ms against 4636), so its 2048-token loss is the pipelining
question and not placement.

### Multi-card prompts, cold start, sessions together (V100)

Four changes, each measured against the binary before it in the same pass,
alternated, warm medians of 5.

**A prompt is pipelined across the cards.** A device that crosses several
cards takes a 2048-row chunk and runs it in 512-row pieces, card r taking
piece k once card r-1 has finished it, so the cards work at once where they
worked in turn (`GPU.pipeline`). It is llama.cpp's batch over micro-batches.

| model | cards | prompt | before | after | llama.cpp | |
|---|---|---|---|---|---|---|
| gemma-3-27b | 2 | 531 | 584 | 553 | 723 | |
| | | 2049 | 2075 | **1338** | 1969 | was 1.06, now 0.68 |
| Qwen3-32B | 2 | 530 | 588 | 548 | 688 | |
| | | 2048 | 2084 | **1332** | 1729 | was 1.21, now 0.77 |
| Llama-3.3-70B | 4 | 531 | 1297 | 1173 | 1408 | |
| | | 2049 | 4681 | **2222** | 3015 | was 1.55, now 0.74 |
| Qwen2.5-72B | 4 | 530 | 1319 | 1190 | 1426 | |
| | | 2048 | 4668 | **2160** | 3080 | was 1.53, now 0.70 |

llama.cpp's column is the table above, same cards, same prompts. A 530-token
prompt is two pieces and gains 6-10%; a 2048-token one is four and gains
1.6-2.2x. A vision block is never cut: its patches attend to one another.

**Cold start: the host stops holding what the card holds.** Qwen3-30B-A3B on
two cards, `speed -ttft` cold phases:

| | open | devices | attach | prompt | cold | peak RSS | minor faults |
|---|---|---|---|---|---|---|---|
| before | 0.23 s | 4.9 s | 20.3 s | 0.7 s | 26.0 s | 19.4 GB | 3.3 M |
| expert pages released | 0.23 s | 4.9 s | 11.2 s | 0.7 s | 17.0 s | 2.1 GB | 0.36 M |
| sheets read ahead, no staging copy | 0.22 s | 4.9 s | 8.1 s | 0.7 s | 13.9 s | | |

The release that drops a placed block's host page never dropped its expert
pages, which since container v26 are 18.0 of the model's 18.5 GiB: every
page went into a fresh frame (3.3 million faults, 19 s of system time) and
stayed. Released, a frame serves the next block's upload. The upload then
reads a bank's sheets 16 ahead of the copy; gathering a window into one
buffer first was tried and cost 6.7 s of a 10.4 s upload, more than the copy
calls it saved. "devices" is CUDA bringing two cards up with persistence mode
off, which llama.cpp pays too.

Against llama.cpp, one-token processes, the same prompt:

| | GGUF evicted from the page cache | cached |
|---|---|---|
| llama.cpp | 16.96 / 16.98 s | 10.19 / 10.14 s |
| jitllm, before | 27.86 / 27.06 s | (reads the disk either way) |
| jitllm, after | **14.42 / 14.52 s** | |

llama.cpp mmaps, so its earlier cold column (10.6 s) read a GGUF its own
server had just put in the page cache; jitllm reads the container with
O_DIRECT every time. From the disk jitllm now starts first.

**Sessions decode together.** `model.Step` runs one token for several States
-- separate sessions, separate users -- as rows of one ragged step on the
device (`nn.SessionStepper`), each row reading its own session's history, so
every block's weights are read once for all of them. Llama-3.1-8B, one card,
`speed -sessions N -p 32 -n 64`, against `llama-batched-bench -npp 32 -ntg 64`
on the same card:

| sessions | one after another | together | llama.cpp S_TG |
|---|---|---|---|
| 2 | 104 tok/s | 144 tok/s (1.38x) | 214 tok/s |
| 8 | 105 tok/s | **483 tok/s (4.59x)** | 324 tok/s |

Eight sessions together are 1.49x llama.cpp's eight parallel sequences; two
were 0.67x. **Up to four sessions now run decode's own matvec with every
session's token in one thread** (`tier.groupMV`), at decode's split, reduced
in the group, over the live rows only. The tiled twins it replaces read the
weights at ~530 GB/s against decode's ~730, padded two rows to eight, and
added a Reduce and an f16 conversion per matvec. Same card and model,
`speed -sessions N -p 16 -n 64`, alternated rounds (one
process per sample, 8 rounds, A/A self-control 0.9945 at IQR/median 0.016),
`JITLLM_GPU_NO_RAG_GROUP=1` the old path:

| sessions | old path | few-sequence matvec | pass 1 / pass 2 | IQR/median | llama.cpp S_TG, same pass |
|---|---|---|---|---|---|
| 2 | 143 tok/s | **198 tok/s** | 1.389 / 1.366 | 0.030 / 0.025 | 211 tok/s (**0.94x**) |
| 4 | 269 tok/s | **308 tok/s** | 1.143 / 1.141 | 0.023 / 0.016 | 289 tok/s (**1.06x**) |

At eight it loses to the tiled twins (363 against 488 tok/s): one thread's
dot products grow with the step, so `groupTokMax` is four. The llama.cpp
column is `llama-batched-bench -npp 16 -ntg 64`, a second tool, so those two
ratios are indicative; the old-path ratio is the gated one.

**A prompt over evicted history streams as a chunk.** It went a row at a
time; the decode's stream now takes a chunk's rows, its passes reaching the
furthest row. In the eviction gate a 100-token second prompt over five evicted
pages a layer takes 400 streamed passes where a row at a time takes at least
1,600, within the same bound.

### Batched decode (V100, one card, same pass)

Aggregate tok/s, a 6-token prompt then 128 generated per sequence:
- jitllm `batch -devices cuda:0` (HEAD 7dc5994).
- llama.cpp `llama-batched-bench -npp 6 -ntg 128 -ngl 99`, total S t/s.
- vLLM 0.18.1 generated tokens over its whole wall clock.

| model | batch | jitllm | llama.cpp | vs llama.cpp | vLLM | vs vLLM |
|---|---|---|---|---|---|---|
| Llama-3.1-8B Q4_K_M | 16 | 955 | 723 | 1.32 | 394 | 2.42 |
| | 32 | 1288 | 1034 | 1.25 | 415 | 3.11 |
| | 64 | 1860 | 660 | 2.82 | 436 | 4.27 |
| | 128 | 2440 | 1142 | 2.14 | 441 | 5.53 |
| olmoe-1b-7b Q4_K_M (MoE) | 16 | 1750 | 1414 | 1.24 | -- | |
| | 32 | 3042 | 1715 | 1.77 | -- | |
| | 64 | 4514 | 1214 | 3.72 | -- | |
| | 128 | 5826 | 1664 | 3.50 | -- | |
| gpt-oss-20b MXFP4 (MoE) | 16 | 764 | 541 | 1.41 | -- | |
| | 32 | 1552 | 734 | 2.12 | -- | |
| | 64 | 2535 | 932 | 2.72 | -- | |
| | 128 | 3020 | 1224 | 2.47 | -- | |
| Qwen3.5-4B Q4_K_M (hybrid) | 16 | 706 | 657 | 1.07 | 442 * | |
| | 32 | 979 | 844 | 1.16 | 565 * | |
| | 64 | 1206 | 730 | 1.65 | 473-667 * | |
| | 128 | 1322 | 934 | 1.42 | OOM * | |

jitllm leads llama.cpp at every batch size on all four models, including the
batch-32 row that read 0.93 in an earlier pass.

**vLLM gaps, each measured:**
- **Llama-3.1-8B:** vLLM ran the same Q4_K_M GGUF.
- **OLMoE:** vLLM cannot load its GGUF, and the fp16 HF checkpoint runs out
  of memory on a 16 GB card at every batch size.
- **gpt-oss-20b:** vLLM refuses MXFP4 below compute capability 8.0.
- **\* Qwen3.5-4B:** vLLM runs the fp16 HF checkpoint, about 3x the bytes of
  the Q4_K_M the other two read, so no ratio is quoted. Its two batch-64 runs
  disagree (667 against 473), and batch 128 runs out of memory.

### V100 box, big models: decode, time to first token, VRAM

Host: the 8x V100 server (2x Xeon E5-2680 v4, 8x Tesla V100-SXM2-16GB sm_70,
driver CUDA 13.0). jitllm main at 07e15cd5, CUDA backend, `-devices
cuda:0,...` over the cards named; llama.cpp 60eeeb6 built for sm_70 (ggml
0.20.1), CUDA, `-ngl 99` with its default layer split. Both engines see the
same cards through `CUDA_VISIBLE_DEVICES`. jitllm reads the v28 `.jlm`
converted from the GGUF llama.cpp reads.

**What jitllm does across cards.** A layer split: `-devices` names the cards
(and backends) and the router (`jit_old/gpu/tier/multi.go`) offers each block to
them fastest first, so the cards run a token one after another and the
residual goes home and out again at each change of card; a prompt is
pipelined across them in 512-row pieces; `-placement` pins any block or the
head to a named device. There is no tensor parallelism. A bare
`-devices cuda` was cuda:0 alone when these rows were taken: gpt-oss-120b then
put 8 of 36 blocks on the card and decoded at 1.6 tok/s, paging the rest from
the host. A bare backend name is every device of that backend since (`cuda:0`
pins one), and the same command spreads over the visible cards. A multi-card row
names every card.

**Decode, `scripts/vs-llamacpp.sh gpu`, 256 generated tokens after an
8-token prompt** (llama-bench `tg256` from an empty context), one process
per sample, ABBA, 3 rounds = 6 pairs a pass:

| model | cards | jitllm tok/s | llama.cpp tok/s | pass 1 | pass 2 | IQR/median |
|---|---|---|---|---|---|---|
| Llama-3.3-70B Q4_K_M | 4 (0-3) | 16.93-17.16 | 17.00-17.03 | 0.9994 | 0.9988 | 0.003 / 0.005 |
| gpt-oss-120b MXFP4 | 5 (0-4) | A/A only: 1.0056, IQR/median 0.018, n=6 | | | | |

The 70B is at parity, both passes. The harness refused both passes as
printed: its coherence gate discarded 5 of 6 pairs for CPU clocks that
differed by more than 1.05x, and on a row with every block on the cards the
host is idle, so those clocks are the idle governor's (1200-1550 MHz), not the
row's. The ratios above are the same samples with the CPU gates off, which is
now the harness's default for an unpinned GPU row (`JITLLM_PIN=1` keeps them).
The A/A self-control ran on this harness and host on gpt-oss-120b (jitllm in
both arms, 106.0-109.5 tok/s). gpt-oss-120b's jitllm-vs-llama.cpp passes were
not taken.

Bytes and the wall: one V100 reads 890 GB/s (torch `max` over 2 GiB on
cuda:0, this box). In a layer split the cards take a token in turn, so the
wall of a multi-card row is one card's. Llama-3.3-70B reads ~41.9 GB a
token (39.59 GiB of weights less the embedding table, of which a token reads
one row): 17.0 tok/s x 41.9 GB = 712 GB/s = 80% of the wall, for both
engines. gpt-oss-120b touches 3.49 GiB = 3.75 GB a token (jitllm's own
count): jitllm 106-109 tok/s = 398-410 GB/s = 45-46%, llama.cpp on all eight
cards 111.9 tok/s = 420 GB/s = 47%.

**llama.cpp's best configuration, all eight cards** (`llama-bench -ngl 99
-p 512 -n 256 -r 3`, absolute, not a ratio against any jitllm row):

| model | cards | pp512 tok/s | tg256 tok/s |
|---|---|---|---|
| Llama-3.3-70B Q4_K_M | 8 | 371.4 | 17.55 |
| gpt-oss-120b MXFP4 | 8 | 1102.3 | 111.88 |

**Time to first token, `scripts/ttft.py`**: one streaming `/v1/completions`
request at a time, greedy, a 513-514-token prompt (a sentence repeated behind
a per-request number, so no prefix cache answers it; llama.cpp is also told
`cache_prompt: false`), 256 tokens asked for. TTFT is send to first text;
prefill is prompt tokens over TTFT; decode is (tokens - 1) over first to last
token. Servers: `jitllmd serve -load X.jlm -devices cuda:0,... -max-seq 4096`
and `llama-server -m X.gguf -ngl 99 -c 4096 -np 1`, one at a time, each
lifetime one warm-up request then 3 timed, lifetimes jitllm, llama.cpp,
jitllm, llama.cpp (6 pairs). VRAM is `nvidia-smi` used, summed over the
cards, after the requests.

| model | engine | cards | TTFT ms | prefill tok/s | decode tok/s | VRAM |
|---|---|---|---|---|---|---|
| Llama-3.3-70B Q4_K_M | jitllm | 4 | 1351 | 380 | 17.87 | 45.8 GiB |
| | llama.cpp | 4 | 1616 | 318 | 17.31 | 44.7 GiB |
| | llama.cpp | 8 | 1585 | 324 | 17.19 | 49.2 GiB |
| gpt-oss-120b MXFP4 | jitllm | 5 | 289 | 1778 | 116.8 * | 64.8 GiB |
| | llama.cpp | 5 | 552 | 929 | 109.7 | 61.3 GiB |
| | llama.cpp | 8 | 582 | 881 | 103.4 | 63.0 GiB |

Ratios, jitllm / llama.cpp on the same cards, median of 6 per-pair ratios:
Llama-3.3-70B TTFT 0.822 (IQR/median 0.038), decode 1.032 (0.009);
gpt-oss-120b TTFT 0.513 **rejected** (IQR/median 0.115), decode 1.067 (0.042).
One pass, so these are indicative, not gated rows. The gpt-oss TTFT spread is
jitllm's own: its two lifetimes read 262-278 and 300-360 ms (lifetime A/A
0.874, IQR/median 0.118), where llama.cpp's agree within 2%. Where it comes
from is unmeasured; a lifetime that ranks the cards differently
(`tier.order` probes them at load) is the first suspect.
\* This row predates jitllmd's `ignore_eos`: gpt-oss ended its answer after
129-159 tokens, so its decode rate is over a shorter context than llama.cpp's
256. `scripts/ttft.py` sends `ignore_eos` to both servers.
The client's 70B decode ratio (1.032) and the harness's (0.999) are two
tools; the harness's is the row.

**vLLM: no row.** vLLM 0.18.1 (its existing venv on the box) ran
Llama-3.3-70B Q4_K_M as a GGUF over four cards in the earlier table above,
only with `enforce_eager` and 0.92 of the memory, and on sm_70 it has no
FlashAttention-2 (needs sm_80) and refuses MXFP4 below sm_80, so it cannot
run gpt-oss-120b on this box at all. Its 70B serving lifetime in this pass
did not start (a fault in the run script, not in vLLM) and was not re-run in
the time box.

Not yet measured: gpt-oss-120b's harness passes against llama.cpp; a
second TTFT pass; vLLM `serve` on the 70B; jitllm on all eight cards; and
MiniMax-M2 Q3_K_M, dots.llm1, Llama-4-Scout, GLM-4.5-Air, Qwen3-Next-80B,
Hunyuan-A13B and gemma-4-31B (all on the box as GGUF and v28 `.jlm`, except
Llama-4-Scout's `.jlm`, which is v26 and needs reconverting).

### V100 box, serving engines: jitllm, vLLM, SGLang, TensorRT-LLM

Host: the 8x V100 server (2x Xeon E5-2680 v4, Tesla V100-SXM2-16GB sm_70,
driver 580.173.02, CUDA 13.0 driver API). jitllm main at 7853eac5,
`jitllmd serve`, CUDA backend. Model: Llama-3.1-8B-Instruct, the headline
model of the premai.io "vLLM vs SGLang vs LMDeploy" comparison this row set
follows (its single-request workload is batch 1).

**Which engines run on Volta at all:**

| engine | version tried | on sm_70 | why |
|---|---|---|---|
| vLLM | 0.18.1 (torch 2.10.0+cu128) | runs | Triton attention: FlashAttention-2 needs sm_80 (its log: "FA2 is only supported on devices with compute capability >= 8"); MXFP4 refused below sm_80 |
| SGLang | 0.5.10 (torch 2.9.1+cu128, sglang-kernel 0.4.1, flashinfer 0.6.7.post2), its own venv | **refuses to start** | `model_runner.py` raises "SGLang only supports sm75 and above." for any capability below 7.5, with `--attention-backend triton --sampling-backend pytorch --disable-cuda-graph`. Below that check, its kernels are not built for Volta either: sglang-kernel's `common_ops` carries sm_80/89/90 cubins only (`cuobjdump`), `sgl_kernel.silu_and_mul` fails with "no kernel image is available for execution on the device", and flashinfer's `rmsnorm` raises `KeyError('sm_70')`. Newer SGLang (0.5.15 on) pins flashinfer's CUDA 13 build, and CUDA 13 dropped Volta. |
| TensorRT-LLM | 1.2.1 (current) | **not supported; not installed** | its support matrix lists Blackwell, Hopper, Ada Lovelace and Ampere only; 1.2.1 requires TensorRT 10.14 and `cuda-python>=13`, and TensorRT's support matrix states "compute capability SM 7.5 or higher" |

**Time to first token and decode, `scripts/ttft.py`**: one streaming
`/v1/completions` request at a time, greedy, `ignore_eos`, a 514-token prompt
(`-reps 30`; a per-request number in front, so no prefix cache answers it)
and 256 generated tokens; TTFT is send to first text, decode is 255 tokens
over first to last. One server at a time, each lifetime 1 warm-up and 10 timed
requests, the client on the box. Servers, all under `taskset -c 0-13`:
- `jitllmd serve -load Meta-Llama-3.1-8B-Instruct-Q4_K_M.jlm -devices cuda:0 -max-seq 4096`
- `vllm serve Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --dtype float16
  --max-model-len 4096 --gpu-memory-utilization 0.85
  --no-enable-prefix-caching` (CUDA graphs on, its default; the GGUF is the
  one jitllm's `.jlm` was converted from)
- `vllm serve unsloth/Meta-Llama-3.1-8B-Instruct --dtype float16
  --tensor-parallel-size 2`, same other flags (its native format; 16.1 GB of
  fp16 weights leave no KV cache on one 16 GB card)

Pairs are request i of one lifetime against request i of the other engine's
lifetime next to it; two lifetimes each, n = 20 a pass. Pass 1 ran jitllm,
vLLM, jitllm, vLLM; pass 2 vLLM, jitllm, vLLM, jitllm.

A/A self-control first, jitllm in both arms on this harness (two lifetimes,
n = 20): TTFT 1.027 (IQR/median 0.048), decode 0.9996 (0.002). The 871-token
A/A taken by mistake before it (`-reps 51`, the client's default): TTFT 0.994
(0.019), decode 0.999 (0.002).

Same weights, one card (cuda:0), Q4_K_M:

| engine | TTFT ms | decode tok/s | GB/token | GB/s, % of the wall | VRAM |
|---|---|---|---|---|---|
| jitllm | 172.1 / 171.9 | 128.75 / 128.57 | 4.63 | 595, 68% | 5.8 GiB |
| vLLM 0.18.1, GGUF | 1000.4 / 1000.2 | 102.03 / 101.52 | 4.63 | 471, 54% | 14.4 GiB * |

(pass 1 / pass 2 medians.) jitllm / vLLM, median of 20 per-pair ratios:

| | pass 1 | pass 2 | IQR/median |
|---|---|---|---|
| TTFT | 0.172 | 0.172 | 0.027 / 0.028 |
| decode | 1.261 | 1.267 | 0.006 / 0.001 |

Both passes pass the 0.10 gate and agree; each engine's two passes agree with
each other (jitllm 0.997 TTFT, 1.001 decode; vLLM 1.000, 1.005). jitllm reaches
the first token 5.8x sooner and decodes 1.26x faster on the same bytes. vLLM's
TTFT here (1000 ms) matches the 1032 ms its offline `LLM.generate` read at
531 tokens in the one-card TTFT table above: on sm_70 its prompt runs through
Triton attention and the GGUF matmuls. GB/token is jitllmd's own
`bytes_per_token` (4,629,371,136; the GGUF's tensors less the embedding table
are 4.617 GB), and vLLM's GGUF path reads the same quantized blocks. The wall
is 873 GB/s (torch `max` over 2 GiB on cuda:0, best of 30, this pass).
\* vLLM's VRAM is its `--gpu-memory-utilization 0.85` reservation (a 65,952-token
KV pool), not what this request needs; jitllm's is weights plus one 4096-position
session.

Two cards, different weights (no ratio: the formats differ), one pass, same
interleave, n = 20:

| engine | weights | cards used | TTFT ms | decode tok/s | GB/token | GB/s per card, % of the wall |
|---|---|---|---|---|---|---|
| vLLM 0.18.1 | fp16, tensor parallel 2 | 2 | 89.2 | 84.04 | 15.0 (7.5 a card) | 631, 72% |
| jitllm | Q4_K_M | 1 of the 2 named | 172.6 | 128.62 | 4.63 | 595, 68% |

vLLM in its native format reaches the first token sooner than jitllm on this
prompt: two cards computing every layer together against one card, with fp16
weights that need no dequantization in the prompt's matmuls. jitllm was given
`-devices cuda:0,cuda:1` and placed all 32 blocks on cuda:0 (cuda:1 held 314
MiB): the router offers each block to the fastest device first and the model
fits one card, so its two-card row is its one-card row. fp16 GB/token is the
checkpoint's 8.03 B parameters less the 0.53 B of the embedding table, at 2
bytes; vLLM's VRAM is 14.6 GiB on each card, again its reservation.

Not yet measured: the post's batched and concurrent-request workload against
vLLM `serve` (the batched-decode table above is vLLM's offline engine); vLLM
on GPTQ (its AWQ kernels need sm_75); a second pass of the two-card rows;
LMDeploy; jitllm tensor parallelism (it has none, see above).

### Xeon (a CPU-only server: Xeon E5-2680 v4, one socket, CPU)

`numactl --membind=0 taskset -c 0-11` for every engine: jitllm `speed -devices
cpu` (12 workers), llama.cpp b11222 `llama-bench -t 12`, mistral.rs v0.9.4
`bench --cpu`, same pass. The mistral.rs column is TTFT-basis
prefill / decode tok/s.

| model | jitllm pp512 | llama.cpp | ratio | jitllm tg128 | llama.cpp | ratio | mistral.rs pp / tg |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 362.2 | 330.1 | 1.10 | 61.86 | 65.01 | **0.95** | 217.5 / 10.6 |
| Llama-3.1-8B Q4_K_M | 50.5 | 47.9 | 1.05 | 11.22 | 12.15 | **0.92** | 38.0 / 3.8 |
| Qwen3-8B Q4_K_M | 49.5 | 48.0 | 1.03 | 10.84 | 11.73 | **0.92** | 37.0 / 3.5 |
| olmoe-1b-7b Q4_K_M | 241.0 | 203.9 | 1.18 | 63.19 | 59.62 | 1.06 | cannot load (`olmoe`) |
| Qwen3-30B-A3B Q4_K_M | 88.1 | 74.7 | 1.18 | 24.17 | 23.58 | 1.03 | 0.3 / 0.02 |
| gpt-oss-20b MXFP4 | 88.3 | 52.8 | 1.67 | 20.29 | 20.31 | 1.00 | 1.8 / 0.4 |

ZML (XLA CPU, bf16 safetensors, same socket): Llama-3.2-1B decodes at
0.9 tok/s and Llama-3.1-8B at 0.2 tok/s.

### M4 Metal, 3B to 7B models (Mac mini M4, 10-core GPU, Metal)

Q4_K_M GGUF converted to `.jlm`, every block on the GPU (`JITLLM_VRAM=0`),
against llama.cpp b10825 on Metal.

**Paired:** `scripts/vs-llamacpp.sh gpu`, n=40 interleaved, the A/A
self-control first (Gemma-3-4B against itself: prefill 0.9999, decode 0.9997).
Warm prefill of a ~415-token prompt; decode 128 tokens. Ratio jitllm /
llama.cpp, median of per-round ratios.

| model | prefill | decode |
|---|---|---|
| Ministral-3-3B-Instruct | 1.019 (486 vs 477 tok/s) | 1.095 (46.1 vs 42.2 tok/s) |
| Qwen3-4B | 1.012 | 1.030 |
| Llama-3.2-3B | 1.017 | 1.030 |
| Gemma-3-4B | **0.955** | 1.018 |
| Mistral-7B | **0.981** (noisy, IQR/median 0.057) | 1.030 |

**Three engines, same pass:** jitllm, llama.cpp b10825 `llama-bench`, MLX
(`mlx_lm` benchmark on the mlx-community 4-bit checkpoints), median of 3,
tok/s. Decode 128 tokens; prefill ~415 tokens (Mistral-7B 507).

| model | tg jitllm | llama.cpp | MLX | pp jitllm | llama.cpp | MLX |
|---|---|---|---|---|---|---|
| Ministral-3-3B | 46.18 | 41.70 | 49.99 | 490.4 | 509.6 | 519.3 |
| Qwen3-4B | 38.88 | 37.86 | 41.74 | 401.2 | 418.3 | 423.0 |
| Llama-3.2-3B | 48.61 | 47.09 | 52.78 | 524.8 | 548.9 | 598.6 |
| Gemma-3-4B | 38.89 | 38.07 | 43.15 | 429.9 | 479.0 | 458.9 |
| Mistral-7B | 23.69 | 23.16 | 24.69 | 229.3 | 224.9 | 243.0 |

Caveats on the three-engine table:
- Each jitllm number is a fresh process, while `llama-bench` and `mlx_lm` warm
  up in-process, which flatters them on prefill (llama.cpp's Ministral prefill
  reads 510 here and 477 in the paired harness). jitllm against llama.cpp is
  the paired table.
- MLX's 4-bit files are smaller than Q4_K_M (fewer bytes per token), so the
  MLX comparison is not same-weights. On them MLX is 4-10% ahead on decode
  (jitllm/MLX 0.90-0.96) and 5-12% on prefill (0.88-0.95).
- Converting MLX checkpoints to `.jlm`, which would make it same-weights, is
  roadmap item 8.

### M4, small models, an earlier pass (Mac mini, 4P+6E, 10-core GPU, Metal)

These rows predate the Metal changes behind the table above and have not been
re-measured.

jitllm `speed -r 2` against llama.cpp b10985 `llama-bench -r 3`, same pass,
idle box. Metal is `-devices metal` against `-ngl 99`; CPU is
`-devices cpu` against `-ngl 0 -dev none` (whose CPU prefill is Accelerate's
BLAS). stories15M runs at pp64 / tg32 because its context is 128.

| model | tier | jitllm pp | llama.cpp | ratio | jitllm tg | llama.cpp | ratio |
|---|---|---|---|---|---|---|---|
| stories15M q4_0 | Metal | 26468 | 30920 | **0.86** | 1343.2 | 876.4 | 1.53 |
| stories15M q4_0 | CPU | 16125 | 51092 | **0.32** | 7724.4 | 8089.9 | **0.95** |
| SmolLM2-360M Q8_0 | Metal | 3993 | 4352 | **0.92** | 188.5 | 180.2 | 1.05 |
| SmolLM2-360M Q8_0 | CPU | 1158 | 1345 | **0.86** | 159.1 | 172.7 | **0.92** |
| tinyllama-1.1B q3_K_M | Metal | 1534 | 1511 | 1.02 | 142.4 | 116.5 | 1.22 |
| tinyllama-1.1B q3_K_M | CPU | 447 | 504 | **0.89** | 103.3 | 94.5 | 1.09 |
| Llama-3.2-1B Q4_K_M | Metal | 1529 | 1603 | **0.95** | 112.8 | 116.1 | **0.97** |
| Llama-3.2-1B Q4_K_M | CPU | 524 | 342 | 1.53 | 83.0 | 85.3 | **0.97** |
| Qwen2-1.5B Q4_K_M | Metal | 1104 | 1159 | **0.95** | 90.5 | 91.8 | **0.99** |
| Qwen2-1.5B Q4_K_M | CPU | 417 | 250 | 1.67 | 61.8 | 68.5 | **0.90** |
| Qwen3-MoE-4x0.6B Q4_K_M | Metal | 1655 | 1821 | **0.91** | 131.6 | 135.8 | **0.97** |
| Qwen3-MoE-4x0.6B Q4_K_M | CPU | 405 | 278 | 1.46 | 105.7 | 113.6 | **0.93** |
| gemma-2b | Metal | 758 | 847 | **0.89** | 59.5 | 58.6 | 1.01 |
| gemma-2b | CPU | 251 | 297 | **0.85** | 40.9 | 42.6 | **0.96** |

mistral.rs v0.9.4 (Metal / CPU, prefill on a TTFT basis / decode tok/s) trails
both engines everywhere it runs:
- SmolLM2: Metal 3117 / 163, CPU 804 / 137.
- tinyllama: Metal 1201 / 107, CPU 119 / 86.
- Llama-1B: Metal 1272 / 106, CPU 232 / 75.
- Qwen2: CPU 170 / 60.
- Qwen3-MoE: CPU 249 / 91.
- gemma-2b: Metal 711 / 19, CPU 161 / 13.

It refuses stories15M (no `head_count_kv` in the GGUF), Qwen2 on Metal (a
dtype mismatch in add) and Qwen3-MoE on Metal ("moe experts forward"). ZML and
vLLM were not run on the M4.

### Every row below parity, and what is measured about it

**V100 decode.** ★ **The four named rows are at or above parity now**
(`kernels.FlashDecodeKV` rebuilt and made CUDA's default; see
gpu-kernels.md, "V100 decode attention as one kernel per split"). Same pass, V100
GPU 0, tg128, jitllm `speed` against `llama-bench -ngl 99 -r 3`, two rounds:

| model | new | staged (the board's path) | llama.cpp | ratio |
|---|---|---|---|---|
| Phi-4-mini | 199.3 | 189.8 | 197.8 | **1.01** (was 0.95) |
| gemma-3-12b | 77.0 | 73.6 | 75.2 | **1.02** (was 0.97) |
| gemma-2-9b | 97.3 | 92.4 | 92.3 | **1.05** (was 0.99) |
| gemma-3-27b, GPUs 0-1 | 38.77 | 37.95 | 38.70 | **1.00** (was 0.98) |

What it was: attention ran as four staged kernels (+286 µs a token on
Phi-4-mini against llama.cpp's flash-vec + combine), and the opt-in fused kernel
walked every dimension per thread in 8 workgroups. The remaining Phi-4-mini term
the old split named, the Q4_K matvecs (+177 µs), was not touched; on
gemma-3-12b the per-kernel split puts the matvecs level (10.28 against 10.28 ms)
and the rest in the norm/rope/quantize chain.
- **Qwen3-1.7B Q8_0 0.99 and Qwen2.5-72B 1.00:** inside this board's
  single-pass band. Qwen3-1.7B reads 264 against 249.5 with the new attention
  (1.06). gemma-3-4b read 1.17 in sweep5 and 1.05 here.

**V100 prefill.**
- **Phi-3.5-mini 0.95:** unattributed.
- **Qwen3-1.7B Q8_0 0.99:** inside the band.

**Xeon dense decode 0.95-0.97 (was 0.92-0.95).** Both engines are DRAM-bound at ~52 GiB/s,
counted at the memory controller (`uncore_imc cas_count_read`). jitllm moves
3.7% more bytes a token, because the container's Q4_K scale plane is 16 bytes a
super-block where GGUF packs 12, and it sustains 2.3% less read bandwidth.
- **The row-contiguous layout the goal named was built as a shadow and is
  null:** row-group-contiguous planes read 11.48 against 11.48 tok/s on 8B.
- **The packed kernel is not layout-bound:** it reads large matrices at 94% of
  a raw sequential sum on the same pool (`nn.TestLayoutRate`).
- **Per-shape k-slicing was tried:** it lost 6-7% in the model, and a picker
  for it bought +0.8% while costing a short 1B prefill 16%.
- ★ **The 12-byte scale plane is built (container v27).** Same pass,
  Xeon node 0, 12 cores, tg128, v26 against v27 binaries each on its own
  container: Llama-3.1-8B 11.11 -> 11.365 (+2.3%, llama.cpp 12.18: **0.912 ->
  0.933**), Qwen3-8B 10.90 -> 11.175 (+2.5%, llama.cpp 11.82: **0.921 ->
  0.945**), Llama-3.2-1B 59.3 -> 58.8 (-0.6%, a recorded cost). What is left
  of these rows is the 2.3% of sustained bandwidth, still unlocated.
- ★ **The fused kernel's prefetch distance is picked per shape.**
  The whole-token duel could not resolve a 4-6% difference -- two identical
  runs settled on 4 and on 64 -- and cached what it landed on. Each shape now
  times the five distances on its own first calls. Same pass, Xeon node 0, 12
  cores, tg128, medians of 12: Llama-3.1-8B 11.40 -> **11.60** (llama.cpp
  12.17: **0.953**), Qwen3-8B 11.22 -> **11.47** (11.86: **0.967**),
  Llama-3.2-1B 59.1 -> **61.0** (64.1: **0.952**), each also above the best
  single pinned distance (11.44, 11.34, 57.8). Not yet taken on the laptop.
- `docs/engineering-history/cpu-kernels.md` has the measurements.

**M4 Metal prefill 0.86-0.95 (the small models' earlier pass; on 3B-7B models it is now 0.955-1.019, above).** GemmTile's large shapes run 3.1-3.3 TFLOP/s of
a ~3.76 peak. The thin projections (1.0-1.4 TFLOP/s) and the FMA-tile attention
are what is left: Metal's simdgroup_matrix is unreachable from the IR's
fragment-shaped MMA (`docs/engineering-history/gpu-kernels.md`). tinyllama wins
at 1.02.
- **64x64 now leads GemmTile's blockings**, where llama.cpp's
  64x32 had. `TestGemmTileSpeed`, Q4_K, 512 tokens: 8192x2048 3.31 against
  3.16 TFLOP/s, 2048x8192 3.26 against 3.10, 512x2048 level. `jitllm speed
  -devices metal -p 512`, same pass, two rounds per arm: Llama-3.2-1B 1526 ->
  1587 (**0.951 -> 0.990** of llama.cpp's 1603.7), Qwen2-1.5B 1101 -> 1136
  (**0.950 -> 0.980** of 1158.6), gemma-2b 759 -> 785 (+3.4%), SmolLM2-360M
  3973 -> 4063 (+2.3%), Qwen3-MoE-4x0.6B 1655 -> 1668 (+0.8%). Nothing moved
  down.

**M4 CPU.**
- **Prefill on the small models (0.32-0.89):** llama.cpp runs Apple
  Accelerate here, and jitllm's GEMM does not block over k. The row-chunk duel
  and the integer-fold GEMM took the k-quant rows to 1.46-1.67.
- **Decode 0.90-0.97 (tinyllama wins at 1.09):** the matvec is 96.9% of a
  Qwen2 token. Its small shapes (256x1536, 1536x1536) read 31-47 GB/s against
  ~70 for large ones on four P-cores (`nn.TestShapeCost`), and adding E-cores
  lowers the rate. The integer fold in the fused decode matvec (2da5898) did
  not move it.

## Kimi-K3 2.78T from disk on the V100 box

The full release, `jitllm convert -q8 hf://moonshotai/Kimi-K3` (f831ab6),
streamed off the Hub into one container and run from the RAID. There is no
ratio: no other engine ran the model in this pass, so there is no same-pass
baseline (RULE 1), and every number below is jitllm's alone.

- **Host:** the 8x V100 server: 2x Xeon E5-2680 v4 (2 NUMA nodes, 56
  threads), 503 GB RAM, 8x Tesla V100-SXM2-16GB (sm_70, CUDA/PTX),
  `/mnt/models` a RAID0 of two NVMe (md0). Nothing else on the box (idle
  `nvidia-smi`, load under 1 before the first run).
- **Container:** `Kimi-K3-q8.jlm`, 1,564,653,101,056 B, written by 24c4857:
  attention, shared experts, the dense lead and the head Q8_0, routed experts
  the release's MXFP4 moved exactly. 1404.92 GiB of weights: 93 block pages
  of 1185.79 MiB and 82,432 expert pages of 16.73 MiB (896 experts in each of
  92 mixture blocks, 16 routed a token).
- **Run:** `jitllm run` at 7ec9eff, greedy, `-n 32`, a 5-token completion
  prompt (the container carries no chat template), the default host budget
  (392.2-393.1 GiB from MemAvailable: 1012 GiB over it), no taskset or numactl.
  Peak RSS from `/usr/bin/time -v`; VRAM from `nvidia-smi -l 5`.

| run | prompt (5 tok) | decode, 32 tok | s/token | read: prefill / decode | pages read, evictions | peak RSS | where the blocks sat |
|---|---|---|---|---|---|---|---|
| `-devices cpu`, first run | 5m55.29s | 8m34.25s | 16.07 | 31.88 GB/prompt tok / 6.01 GB/tok | 93 blocks + 16,653 experts (327.57 GiB), 0 | 330.7 GiB | 93 on the host |
| `-devices cpu`, again | 59.38s | 2m47.51s | **5.23** | 31.88 / 6.01 | the same, byte for byte | 330.7 GiB | 93 on the host |
| `-devices cpu`, "12 * 7 =" | 56.19s | 1m53.16s | 3.54 | 30.05 / 5.50 | 93 + 15,194 (303.73 GiB), 0 | 306.8 GiB | 93 on the host |
| `-devices cpu`, "def fibonacci(n):" | 56.79s | 3m7.14s | 5.85 | 30.53 / 6.58 | 93 + 17,306 (338.24 GiB), 0 | 341.3 GiB | 93 on the host |
| `-devices cuda` | 51.67s | 3m47.33s | 7.10 | 19.95 / 6.54 | 93 + 17,617 (343.33 GiB, 55.43 at load), 0 | 346.8 GiB | block 0 and the head on cuda:7 (6.93 GiB), 92 on the host |
| `-devices cuda`, `JITLLM_GPU_STREAM=1` | 1m42.18s | 9m12.54s | 17.27 | 19.96 / 6.94 | 18,445 frames (355.35 GiB, 55.43 at load), 0 | 383.4 GiB | 93/93 and the head on six cards: 23, 16, 16, 16, 15, 7 |

The time columns are the run's own `prompt` and `decode` line; the read
columns are the `pages` line's per-phase bytes.

**What it said.** Every answer is coherent and right:

    The capital of France is Paris. The Eiffel Tower is located in Paris. The
    Louvre Museum is also in Paris. The Seine River flows through Paris. Paris
    is known for its

    12 * 7 = 84
    - 84 + 12 = 96
    - 96 + 12 = 108
    - 108 + 12 = 120

    def fibonacci(n):
        if n <= 1:
            return n
        else:
            return fibonacci(n-1) + fibonacci(n-2)

The two host runs of the France prompt produced the same 32 ids. The device
runs part from the host after "Paris": `-devices cuda` at the second generated
token (`.",` and a list of quoted sentences), the streamed run at the fourth
("The city has a population of over 2 million people. It is known for its art,
culture, and cuisine. The Eiffel Tower is one"). Both are coherent; RULE 11c's
band, not a measured bug -- the split tuner was not pinned and the head ran as
sm_70 f16 tensor-core matvecs, and no equality was asked of these runs.
Correctness rests on the 0.40B gates (`TestKimiK3RealMatchesReference`: 5.8e-13
against Moonshot's own code on the trained Kimi-K3-0.40B, host, CUDA and
Vulkan, model-correctness.md "kimi-k3") plus these answers.

**What the numbers are, and are not.**

- **No run evicted.** 32 tokens touched at most 355 GiB of the 393 GiB budget,
  so every row is the fill phase: each page touched was read once and kept.
  Decode still read 5.5-6.9 GB a token (16 routed experts x 92 blocks is 24.1
  GiB a token, so about three quarters were already resident from the prompt
  and earlier tokens). The steady state, where the budget is full and the
  expert LRU evicts, needs a longer generation and is not measured here.
- **The first host run is 3.8x the second with the same ids and the same
  bytes read.** It ran with ~418 GiB of page cache left by the conversion's
  writes; its system time was 6969 s against 813 s, and the expert reads
  "blocked" 11m59s against 2m0s. Fresh frames came out of reclaim. The second
  run is the one to read.
- **No disk ceiling was measured in the same pass**, so no fraction of a wall is
  claimed. The host decode read 6.01 GB in 5.23 s a token, 1.15 GB/s of wall,
  and `iostat` during the first run showed md0 at ~0.65 GB/s and 31%
  utilised: the reads are latency-bound, not bandwidth-bound.
- **`-devices cuda` places one block.** A K3 mixture block resident with its
  896-expert bank needs 16.41 GB, above a V100's 14.46 GiB budget, so each of
  the 92 is declined ("placed resident and does not fit") and only the dense
  lead and the head reach a card. Its decode is a host decode run in a
  different machine state (after the first host run, with the placement's
  buffered reads refilling the page cache), not a device number. Placement
  took about 20 minutes (25m06s wall against 4m39s of prompt and decode):
  every block is offered to each of the eight cards twice and refused after
  its weights are read -- the process read ~510 GB through a buffered handle
  before printing the device line. Per-card VRAM: cuda:7 peaked at 11,286 MiB
  (6.93 GiB resident plus scratch), the other seven at 342-346 MiB idle and
  up to 4,174 MiB while a block was being tried.
- **`JITLLM_GPU_STREAM=1` places every block** (placement.md "Qwen3-Next-80B
  on the card"): a mixture block goes on the card without its routed bank and the
  submission uploads the 16 routed experts' sheets each step. 93/93 blocks and
  the head, 79.78 GiB resident over six cards (cuda:0 23, cuda:7 16, cuda:1 16,
  cuda:3 16, cuda:6 15, cuda:4 7; cuda:2 and cuda:5 none), five device changes
  a token, placed in 11.45 s of upload. Peak VRAM: cuda:0 15,936 MiB, cuda:6
  15,300, cuda:1 15,036, cuda:3 15,036, cuda:7 15,032, cuda:4 10,292, cuda:2
  and cuda:5 342. Its decode is slower than the host's because it is the
  upload: 641.54 s of the 649.60 s of block compute is `stream cost: upload`,
  3404 fills (92 mixture blocks x 37 steps) at 188 ms each, none overlapped
  with a read. If a fill moves the 16 routed expert pages (268 MiB) that is
  ~1.4 GiB/s over PCIe, far under the link.

### Kimi-K3 streamed on the cards (k3-streaming)

Same box, container and prompt as above, `-n 32`, one run a row, runs
interleaved device/host; no ratio (RULE 2's paired rounds were not taken).
placement.md 16c has the attribution.

| run | prompt | decode s/token | tok/s | decode B/token | peak RSS | blocks on cards |
|---|---|---|---|---|---|---|
| `-devices cpu` (x3) | 59.9-61.0 s | 3.61-3.69 | 0.27-0.28 | 6.01 GB | 330.7 GiB | 0 |
| `-devices cuda`, default (auto-stream, cache 18) (x3) | 36.9-41.5 s | 3.51-3.66 | 0.27-0.29 | 6.54-6.57 GB | 347-348 GiB | 78/93 |
| `-devices cuda`, `STREAM=1 STREAM_GROUPS=4 STREAM_CACHE=18` | 38.8 s | 3.55 | 0.28 | 6.04 GB | 333.4 GiB | 93/93 |
| `-devices cuda`, `JITLLM_GPU_NO_AUTOSTREAM=1` | 63.5 s | 3.68 | 0.27 | 6.54 GB | 347.2 GiB | 1/93 (load 1.16 GiB, 3m20s wall) |

Link ceiling measured in the same session: 8.27 GiB/s pinned to one V100.

### Kimi-K3: hybrid experts against the host (k3-streaming, later)

Same box and container, "The capital of France is", runs back to back.
placement.md 16c-2 has the attribution.

| run, 64 tokens | prompt | decode s/token | tok/s | decode B/token | peak RSS |
|---|---|---|---|---|---|
| `-devices cuda`, default (auto-stream, hybrid experts), trial off | 27.1 s | 3.11 | 0.32 | 5.58 GB | 418 GB |
| `-devices cpu` (shared-expert overlap on) | 60.1 s | 4.44 | 0.23 | 4.98 GB | 416 GB |
| `-devices cuda`, `JITLLM_GPU_HYBRID=0` (sheets sent, cache, groups tuned) | 36.2 s | 5.66 | 0.18 | 4.92 GB | 418 GB |

Paired, in one process (the stream trial, ABBA runs of 8 tokens, 10 quads, 430
tokens): hybrid / host **1.873**, IQR/median 0.057, n = 20; A/A from the same
runs 0.987 with IQR/median 0.134, over the 0.10 gate; one pass. Not a clean
RULE 2 ratio: the self-control is dispersed and the comparison was not run
twice.

### Kimi-K3: everything stacked (k3-streaming, final)

Same box, container and prompt, 64 tokens, runs back to back, one each. VRAM
is the per-card peak from `nvidia-smi -l 2` in MiB.

| run | prompt | decode s/token | tok/s | decode B/token | peak RSS | VRAM per card |
|---|---|---|---|---|---|---|
| `-devices cpu` | 59.4 s | 4.34 | 0.23 | 4.98 GB | 416 GB | -- |
| `-devices cuda`, default (auto-stream, hybrid experts), trial off | 25.9 s | **2.44** | 0.41 | 4.86 GB | 407 GB | 13.3-15.0 G on all 8 |
| `-devices cuda`, `JITLLM_EXPERTS=host` (hybrid forced) | 26.1 s | 2.85 | 0.35 | 5.38 GB | 418 GB | 13.4-15.0 G on all 8 |
| `-devices cuda`, `JITLLM_EXPERTS=card` (sheets forced) | 39.8 s | 6.64 | 0.15 | 4.93 GB | 418 GB | 0.3-15.9 G |

The cpu and sheets rows are from the binary at 0ebf9b8c; the two hybrid rows
and the trials from the one at 050ac402 (the spread fix), whose forced-hybrid
run the earlier binary could not prefill.

**Paired, in one process (the stream trial, ABBA runs of 16 tokens, 10 quads,
900 tokens, warm quads 2-10):** A/A (the placement against itself through the
same migrations) median **1.001**, IQR/median **0.111** -- over the 0.10 gate.
Hybrid / host, pass 1 median **1.935** (IQR/median 0.218), pass 2 **1.921**
(0.187). The two passes agree and the A/A sits at 1.00, but the A/A's own
dispersion fails RULE 2's gate (the hybrid arm's runs spread 0.55-0.92 tok/s
against the host's 0.37-0.41), so this is **not** a RULE 2 ratio. With 8-token
runs the A/A read 0.993 at 0.118.

### Kimi-K3: kimi-k3-in-c on the same box

kimi-k3-in-c (github.com/FareedKhan-dev/kimi-k3-in-c at 81bb6c2, built on the
box with `make -j`, `make test`: "ENGINE MATCHES THE REFERENCE EXACTLY") on the
release's safetensors at the revision jitllm's container was converted from
(moonshotai/Kimi-K3@f831ab66, 96 shards, 1,560,998,661,367 B, every file's size
checked against the Hub), its trunk packed per its README (108.8 GB). Same box
and RAID as every row above; the jitllm container was deleted to make the room
and reconverted afterwards. Command, run twice back to back under one cgroup
ceiling (`systemd-run --user --scope -p MemoryMax=420G -p MemorySwapMax=0`),
all 56 threads (its default physical-core count), no numactl:

    ./bin/k3 model --trunk trunk --trunk-gb 112 --cache-gb 260 --tok model \
        --prompt "The capital of France is" --gen 64 --incremental

| run | first token (step 0) | decode s/token (steps 1-63) | tok/s | read | peak RSS |
|---|---|---|---|---|---|
| kimi-k3-in-c, run 1 | 153.9 s | 9.55 | 0.105 | 539 GB in the run (430 GB experts, 109 GB trunk) | 376.7 GB |
| kimi-k3-in-c, run 2 | 74.9 s | 8.83 | 0.113 | 544 GB | 376.7 GB |

Then the jitllm container was reconverted with the same options
(`jitllm convert -q8 hf://moonshotai/Kimi-K3@f831ab66... Kimi-K3-q8.jlm`, 22369c5c,
4h36m streamed from the Hub, 1,564,653,101,056 B, the same size as before) and
jitllm ran back to back on the same box, disk and cgroup ceiling (MemoryMax=420G:
jitllm's self-set budget 378 GiB, kimi-k3-in-c's peak RSS 376.7 GB), all cores,
no numactl, trial off:

| run | prompt (5 tok) | decode s/token | tok/s | decode read | peak RSS |
|---|---|---|---|---|---|
| jitllm `-devices cpu`, run 1 | 302.2 s | 8.59 | 0.12 | 5.16 GB/token | 355.6 GB |
| jitllm `-devices cpu`, run 2 | 60.0 s | 4.31 | 0.23 | 5.16 GB/token | 355.6 GB |
| jitllm `-devices cuda` (hybrid), run 1 | 47.0 s | 3.46 | 0.29 | 4.91 GB/token | 357.5 GB |
| jitllm `-devices cuda` (hybrid), run 2 | 46.4 s | 2.69 | 0.37 | 4.70 GB/token | 358.0 GB |

The first CPU run came right after the conversion's 1.45 TB of writes and is the
cold-reclaim run the first table above describes (its decode read the same bytes
as run 2 at a third of the rate); the second is the one to read. The
reconverted file gives the earlier runs' answers: the CPU runs' 64 ids are the
earlier CPU runs' exactly, and the first device run's are the earlier device
runs' and kimi-k3-in-c's exactly ("Paris." and a list); the second device run
parts after 31 tokens, within the device band.

**No ratio.** RULE 1 wants both arms measured in the same pass with the same
tool, and RULE 2 paired interleaved rounds: kimi-k3-in-c and the jitllm
container could not both be on this disk (237 GiB free, a 60 GiB floor, and
1.45 TB each), so the arms are hours apart with a reconversion between them, two
runs each, not interleaved. Absolute rows only: kimi-k3-in-c decoded at
8.8-9.6 s/token with a 75-154 s first step; jitllm's host 4.31 s/token (warm run)
and its hybrid device placement 2.69-3.46 s/token with a 46-47 s prompt, under
the same memory ceiling.

Its tokens: `17374 20829 10 427 414 1008 606 142957 37092 387 7081 306 ...` --
"Paris.", then a quoted list of sentences about Paris, identical in both runs
and identical to jitllm's device runs of this prompt (the streamed rows above);
jitllm's host run parts from it at the second token (`13` against `20829`),
within RULE 11c's band of two f32 reduction orders.

## ★ THE V100 BOARD (sweep5, one pass, main at c07936d + fixes)

Host: the 8x V100 server, 2x Xeon E5-2680 v4, 8x Tesla V100-SXM2-16GB (sm_70), CUDA.
`jitllm speed -p 512 -n 128 -r 2` against `llama-bench -p 512 -n 128 -ngl 99
-r 3` (llama.cpp built on the box for sm_70), each model's two engines back to
back, 36 models, one pass. Ratio is jitllm / llama.cpp.

| model | GPUs | pp512 | llama.cpp | ratio | tg128 | llama.cpp | ratio |
|---|---|---|---|---|---|---|---|
| SmolLM2-360M Q8_0 | 1 | 31031 | 24551 | 1.26 | 456.8 | 410.0 | 1.11 |
| Qwen3-0.6B Q8_0 | 1 | 21968 | 20652 | 1.06 | 382.2 | 379.4 | 1.01 |
| gemma-3-1b | 1 | 15760 | 14621 | 1.08 | 301.5 | 273.2 | 1.10 |
| Llama-3.2-1B | 1 | 19721 | 15491 | 1.27 | 496.5 | 464.3 | 1.07 |
| SmolLM2-1.7B | 1 | 12654 | 9934 | 1.27 | 346.9 | 338.4 | 1.03 |
| Qwen2.5-1.5B | 1 | 12104 | 11239 | 1.08 | 339.0 | 285.2 | 1.19 |
| gemma-2-2b | 1 | 9390 | 8158 | 1.15 | 226.3 | 207.8 | 1.09 |
| Qwen3-1.7B Q8_0 | 1 | 11810 | 11950 | 0.99 | 244.7 | 250.8 | 0.98 |
| Llama-3.2-3B | 1 | 7704 | 6261 | 1.23 | 224.2 | 220.7 | 1.02 |
| Phi-3.5-mini | 1 | 5359 | 5717 | 0.94 | 197.1 | 199.5 | 0.99 |
| Phi-4-mini | 1 | 6589 | 6035 | 1.09 | 187.5 | 198.3 | 0.95 |
| gemma-3-4b | 1 | 6616 | 5730 | 1.15 | 157.0 | 134.4 | 1.17 |
| Qwen3-4B | 1 | 6487 | 5165 | 1.26 | 176.4 | 168.4 | 1.05 |
| olmoe-1b-7b (MoE) | 1 | 10596 | 4149 | 2.55 | 423.7 | 389.8 | 1.09 |
| Mistral-7B v0.3 | 1 | 3774 | 3253 | 1.16 | 135.6 | 133.2 | 1.02 |
| Qwen2.5-7B | 1 | 4047 | 3403 | 1.19 | 138.1 | 131.4 | 1.05 |
| Llama-3.1-8B | 1 | 3888 | 3220 | 1.21 | 129.4 | 125.7 | 1.03 |
| Qwen3-8B | 1 | 3544 | 3074 | 1.15 | 121.4 | 119.9 | 1.01 |
| gemma-2-9b | 1 | 2988 | 2500 | 1.20 | 90.7 | 92.4 | 0.98 |
| gemma-3-12b | 1 | 2439 | 2015 | 1.21 | 72.0 | 75.2 | 0.96 |
| Qwen3-14B | 1 | 2201 | 1778 | 1.24 | 75.0 | 72.5 | 1.04 |
| phi-4 | 1 | 2191 | 1828 | 1.20 | 74.9 | 74.5 | 1.01 |
| gpt-oss-20b MXFP4 (MoE) | 1 | 4332 | 2719 | 1.59 | 188.1 | 170.7 | 1.10 |
| DeepSeek-V2-Lite (MoE, MLA) | 1 | 4362 | 2087 | 2.09 | 186.8 | 171.2 | 1.09 |
| DeepSeek-Coder-V2-Lite | 1 | 4351 | 2025 | 2.15 | 185.8 | 171.1 | 1.09 |
| Moonlight-16B-A3B | 1 | 4498 | 2118 | 2.12 | 178.4 | 131.0 | 1.36 |
| gemma-3-27b | 2 | 958 * | 902 | 1.06 | 37.3 * | 38.7 | 0.96 |
| Qwen3-30B-A3B (MoE) | 2 | 3509 | 1081 | 3.25 | 150.9 | 153.5 | 0.98 |
| Qwen3-32B | 2 | 1009 | 791 | 1.28 | 35.6 | 34.5 | 1.03 |
| Mixtral-8x7B (MoE) | 2 | 1703 | 868 | 1.96 | 85.0 | 82.3 | 1.03 |
| Llama-3.3-70B | 4 | 444 | 371 | 1.20 | 17.5 | 17.4 | 1.01 |
| Qwen2.5-72B | 4 | 421 | 373 | 1.13 | 16.0 | 16.4 | 0.98 |
| Qwen3-Next-80B (hybrid MoE) | 4 | 1876 | 445 | 4.22 | 96.7 | 94.0 | 1.03 |
| gpt-oss-120b MXFP4 (MoE) | 5 | 1971 | 1222 | 1.61 | 116.1 | 118.4 | 0.98 |
| Kimi-Linear-48B (hybrid, MLA) | 4 | 2665 | -- | -- | 105.3 | -- | -- |

\* gemma-3-27b hung in the sweep build (a split-tuner deadlock, 3dc9c63) and,
once that was fixed, prefilled at 42 tok/s because fill-first placement put 59
of 62 blocks on the first card and left no room for the prompt's scratch
(a76237a, e05acdc); the row is the fixed build, ABBA'd against the old one in
one pass, with llama.cpp's 899 and 905 from the same sweep averaged.

Prefill wins 33 of 35 rows llama.cpp can run (Phi-3.5-mini 0.94, Qwen3-1.7B
Q8_0 0.99); mixtures and hybrids 1.6-4.2x. Decode is 0.95-1.36 everywhere, the
under-parity rows being Phi-4-mini 0.95, gemma-3-12b 0.96, gemma-3-27b 0.96,
and five at 0.98. What moved it since sweep4: the grouped
tensor-core expert GEMM and MLA's m8n8k4 attention (MoE prefill 0.65-0.71 ->
1.59-2.15), the chunked gated delta rule (Qwen3-Next pp 0.14 -> 4.22), the
dense prefill work (Q8_0 small-model pp 0.77-0.81 -> 0.99-1.26), and three
placement bugs: a block admitted without room for its KV paging every token
(Mixtral tg 0.05 -> 1.03, Llama-3.3-70B tg 0.80 -> 1.01), the output projection
left on the host beside an empty card (gpt-oss-120b tg 0.41 -> 0.98), and the
fill-first placement above.

## ★ THE M4 BOARD AFTER THE METAL AND arm64 LEVERS (5b04d82)

Same pass, jitllm `speed -r 1` against llama.cpp b10985 `llama-bench -r 1`
on the same GGUF, J L L J per round, two rounds (three for stories15M),
median of per-round ratios, M4 mac mini (4P+6E, 10-core GPU), boxes idle
(busy 0-19% at each model's start). pp512 / tg128; stories15M pp64 / tg32
(its context is 128). Metal: `-devices metal` against `-ngl 99`; CPU:
`-devices cpu` against `-ngl 0 -dev none` (which llama-bench still reports
as MTL,BLAS: its CPU prefill is Accelerate).

| model | Metal pp | Metal tg | CPU pp | CPU tg |
|---|---|---|---|---|
| stories15M q4_0 | 0.663 | **1.313** | 0.281 | **1.065** |
| SmolLM2-360M Q8_0 | 0.761 | **1.037** | 0.652 | 0.791 |
| tinyllama q3_K_M | 0.910 | **1.235** | 0.470 | **1.102** |
| Llama-3.2-1B Q4_K_M | 0.877 | 0.960 | 0.670 | 0.973 |
| Qwen2-1.5B Q4_K_M | 0.908 | 0.962 | 0.833 | 0.908 |
| Qwen3-MoE-4x0.6B Q4_K_M | 0.816 | 0.978 | 0.669 | 0.923 |
| gemma-2b | 0.895 | **1.000** | 0.736 | 0.893 |

Where it started the same day, before the branch: Metal prefill 0.048
(Llama) and 0.40 (Qwen3-MoE) on the dp4a tile; CPU decode Llama 0.717,
tinyllama 0.63, Qwen2 ~0.51; CPU prefill 0.36-0.62. Llama-3.2-1B decode,
816,010,912 B/token: Metal 110.9 tok/s = 90.5 GB/s, CPU 83.0 tok/s = 67.7
GB/s, against 108.9 GB/s measured on the GPU (MLX's contiguous sum reads
105.7) -- the CPU probe tops out at ~70 GB/s on four P-cores.

What is still losing, and what is known about each:

- **Metal decode, 2-4 points** (Llama, Qwen2, Qwen3-MoE): the per-dispatch and
  fixed per-token terms in docs/engineering-history/gpu-kernels.md; the
  poll-before-block Wait took ~4 points of it.
- **Metal prefill, 9-24 points**: GemmTile's big shapes run 3.1-3.3 TFLOP/s of
  a ~3.76 peak; the thin projections (1.0-1.4 TFLOP/s) and the FMA-tile
  attention are what is left. SmolLM2 (960-wide) is the thinnest.
- **CPU decode, SmolLM2 and gemma** (Q8_0/Q4_0): the matvec runs 45-70 GB/s
  per shape (nn.TestShapeCost); llama.cpp's repacked NEON reads ~71 GB/s
  whole-model on gemma against jitllm's 63.
- **CPU prefill, 17-72 points**: the k-blocking GEMM this file already names
  as the remaining structural lever, and llama.cpp runs Accelerate here.

## ★ THE MAC HAD NO PREFILL ROW AT ALL, AND IT IS THE WORST HALF OF THAT HOST

`scripts/vs-llamacpp.sh` in prefill mode, n=507, llama.cpp b10825, M4:

    tinyllama q3_K_M   0.4575 -> 0.5434   IQR 0.005    (the amax hoist below)
    gemma-2b           0.4852             IQR 0.004    CONTROL, unchanged

**THE SAME CODE WINS PREFILL ON LINUX AT 1.32 AND LOSES IT HERE AT 0.54**, which
is a 2.4x cross-host swing on one model and the largest unexplained gap on the
board. And the direction of the cause is the opposite of the obvious one: **jitllm
is FASTER in absolute terms on the Mac** (227 tok/s against ~207 on the Linux
laptop). It is llama.cpp that is 3x faster there -- 495 against ~157.

★ AND IT IS NOT i8mm, WHICH RETIRES THE STANDING PREDICTION. RULE 1d-sexies says
"what is missing is arm64's SMMLA ... that is the first thing to price for the
prefill row, and it is arm64-only". `otool -tV` on the Mac's libggml-cpu.dylib
counts **ZERO smmla and 1045 sdot**. llama.cpp is not using i8mm on this host at
all, so it cannot be where its 3x comes from. What it does have is repacked 8x8
GEMM kernels -- `ggml_gemm_q4_K_8x8_q8_K`, q5_K, q6_K, q2_K, q4_0, q8_0, iq4_nl,
mxfp4 -- and, as on amd64, **none for q3_K**.

## ★ THE amax WAS RECOMPUTED PER BLOCK AND arm64 PAID IT EIGHT TIMES: +18.8%

`JITLLM_MMPROF=1` splits MatMul into its three phases. M4, tinyllama, 232-token
prompt:

    before   pack 328 ms 39.3%   kernel 489 ms 58.6%   transpose 16 ms 1.9%
    after    pack 168 ms 24.6%   kernel 495 ms 72.7%   transpose 17 ms 2.5%

**Packing halved; the kernel and the transpose did not move**, which is the
control built into the same measurement.

Every block in an amax window shares one scale, and `packGEMM` recomputed that
maximum FOR EVERY BLOCK -- scanning the same elements `per` times. At the wide
window arm64 uses for an all-k-quant model (256 elements, eight 32-element
blocks) that is EIGHT passes where one does.

**★ IT COST NOTHING ON amd64, WHICH IS WHY IT SURVIVED.** `ActWin` stays 32
there -- RULE 5d: VPDPBUSD is unsigned x signed, so the 16-wide formats need
per-16 activation sums -- so `per` is 1 and the redundancy does not exist. The
whole defect is invisible on the host most of this project's profiling was done
on, and it is 15% of a prefill on the other one.

    M4 tinyllama prefill vs llama.cpp   0.4575 -> 0.5434   IQR 0.005
    M4 gemma-2b     prefill             0.4852             CONTROL

**gemma is Q4_0 + Q8_0, so `WideActWindow` declines it and its window is 32 --
`per` is 1 and it CANNOT gain.** It does not. A fix whose control is decided by
the model's quantization mix, and which lands exactly where the mix says it
should, is a result rather than a number.

★ THE CACHE IS KEYED ON THE WINDOW BASE AND NOT ON A CHUNK BOUNDARY, so it holds
whatever block range the pool hands a worker -- no assumption that a chunk is a
whole number of windows. Bit-identical by construction (same maximum, computed
once), and the equality gates confirm it: Prefill, ForwardBatch and the
eleven-model oracle are unchanged.

## ★ TWO MORE THINGS IN THE PACKER, AND ONE OF THEM MEASURED ZERO

**0.5434 -> 0.5518** (IQR 0.004), pack 168 -> 155 ms, from two changes that are
bit-identical and cost nothing:

  - **the half sums stopped being a second pass.** They re-read every byte back
    out of `dst` at the same SCATTERED index the store had just used -- 32 loads
    per block, one per cache line at tok=16 -- to re-derive numbers the loop was
    holding in a register. `q` is the value; `dst` is only where it goes.
  - **`roundHalfAway` stopped making four conversions to produce an integer.**
    It widened an f32 product to f64, added 0.5, truncated to int64, converted
    BACK to f64, and the caller converted to int. It is bit-identical in f32
    because the value is BOUNDED: `inv` is 127/amax and amax dominates every
    element of its window, so |v*inv| <= 127, and f32 represents every x + 0.5
    exactly below 2^23. The f64 was buying precision the quantity cannot use.

**★ AND ASSEMBLING EACH GROUP'S FOUR BYTES INTO ONE uint32 STORE MEASURED
EXACTLY ZERO -- 155 ms against 157 -- SO IT IS NOT THERE.** The packed layout
puts ki%4 in consecutive bytes, so it looked like four index computations and
four bounds checks per dword that one store would replace. The compiler was
already handling it. **The cost that remains in the packer is the per-element
FLOAT work, not the stores**, which is the thing to aim a kernel at and is
recorded here so the store idea is not re-derived from the layout.

## ★ AND THEN THE PACKER WAS GENERATED, AND IT WAS 13x: 0.5518 -> 0.6923

`cpu.EmitPackAct` on both architectures. M4, tinyllama, 232-token prompt:

    pack 155 ms 23.2%  ->  12 ms 2.2%
    MatMul total 668 ms -> 527       prompt 311 -> 384 tok/s

    M4 tinyllama prefill vs llama.cpp   0.5518 -> 0.6923   IQR 0.007
    M4 gemma-2b     prefill             0.4852 -> 0.5663   IQR 0.005
    M4 tinyllama DECODE (control)       0.9387             unchanged

**★ I ESTIMATED "AT MOST 23% OF MatMul AND REALISTICALLY HALF" AND IT DELIVERED
ESSENTIALLY ALL OF IT.** The paragraph this replaces argued the packer was the
smaller target and the GEMM kernel the bigger one. The estimate was wrong in the
direction this file usually gets wrong the other way: a Go loop doing ~10 scalar
operations an element against a kernel doing ~1.5 is not a 2x, it is a 13x, and
the share it occupied was the whole of it.

★ gemma GAINS TOO, WHICH THE PREVIOUS FIX COULD NOT. The amax hoist was worth
nothing there -- window 32, `per` 1, no redundancy -- but the per-element
quantize is the same work at any window, so the kernel helps both. **Two fixes in
the same function with DIFFERENT control models, each landing where its own
mechanism says it should.**

★ AND DECODE IS THE CONTROL THAT MATTERS. It does not go through MatMul at all
(RULE 13a: MatMul declines below eight tokens), so it must not move, and it
reads 0.9387 against the 0.9378 measured before any of this.

★ WHAT THE KERNEL DOES, AND THE TWO THINGS THAT MAKE IT BIT-IDENTICAL RATHER
THAN CLOSE:

  - **TRUNCATION, NOT ROUND-TO-NEAREST.** `VCVTTPS2DQ` / `FCVTZS` after adding a
    sign-carrying 0.5, because the Go packer rounds half AWAY from zero.
    `VCVTPS2DQ` rounds to nearest-EVEN and differs on every exact half -- a
    different quantization of the model, not a rounding detail.
  - **TWO DIVISIONS, NOT ONE MULTIPLY.** `d := amax/127` then `1/d`, reproduced
    exactly. A single 127/amax differs in the last bit and moves a quantized
    value at the boundary.

The saturating narrow (`VPACKSSDW`+`VPACKSSWB`, `SQXTN` twice) performs the
[-128,127] clamp for free AND is a no-op, because `inv` is 1/(amax/127) and amax
dominates its own window -- so the block sums may be taken from the UNSATURATED
integers and still equal the Go packer's post-clamp sums.

★ THE ZERO WINDOW IS A CASE, NOT AN EDGE. MatMul zero-fills the padding rows of a
ragged tile, so an all-zero window arrives on every prompt whose length is not a
multiple of the tile -- and 1/0 is +Inf with 0*Inf NaN. Both kernels mask `inv`
to zero there (`VCMPPS`+`VPAND`, `FCMEQ`+`BIC`), and the gate has that mode.

★ RUN AGAINST THREE VIOLATIONS, all of which fire: nearest-even instead of
half-away, no zero mask, and a single 127/amax divide.

## ★ THE LINUX DECODE BOARD, RE-TAKEN RESTED: EVERY ROW UP, MEAN +5.4%

The previous board sat stale because the box was in RULE 2d's
sustained-power clamp (765-1653 MHz, PSI 0.00, all evening) -- the open question
said "re-take them rested" twice. Taken at b31240e, `scripts/vs-llamacpp.sh`,
CPU decode n=320, six P-cores, ONE binary for all five rows:

| model | ratio | IQR | was | |
|---|---|---|---|---|
| Qwen2-1.5B-Instruct Q4_K_M | **1.0887** | 0.038 | 1.0169 | WIN |
| Qwen3-1.7B Q4_K_M | **1.0554** | 0.019 | 1.0076 | WIN |
| tinyllama-1.1B q3_K_M | 0.9923 | 0.016 | 0.9471 | |
| OLMoE-1B-7B Q4_K_M | 0.9578 | 0.025 | 0.9090 | |
| gemma-2b | 0.9294 | 0.006 | 0.8856 | 2 clamped samples discarded, n=10 |

    per-row movement   +7.1%  +4.7%  +4.8%  +5.4%  +4.9%    mean +5.4%

**★ THE MOVEMENT IS UNIFORM, WHICH IS THE INTERESTING PART.** Five models across
four architectures, two quant mixes and a mixture-of-experts all moved by
essentially the same amount. That is the signature of a change in what EVERY
token does rather than in any one kernel -- the elementwise generation (RULE 8,
independently measured at 2.6-4.5% on the M4) plus the MoE accumulation fix, the
Mixtral router kernel and the f16 KV tuning. A kernel-shaped win would have
shown up on one format and not the others, the way the arm64 fold work did.

★ AND PART OF IT IS THE CLAMP LIFTING, WHICH IS WHY BOTH BOARDS ARE PRINTED. The
old rows were taken at 2191 MHz mean against a ~3500 MHz rested peak; these were
taken with PSI avg10 under 1.3% throughout. The RATIO is normalised against
llama.cpp measured in the same seconds, so frequency cancels to first order --
but it does not cancel exactly, because the two engines do not lose rate to a
clamp at the same rate (the M4 prefill row shows jitllm shedding 7.4% to heat
where llama.cpp shed 4.2%). Treat the +5.4% as an upper bound on what the code
did and a lower bound on nothing.

★ gemma-2b DISCARDED TWO SAMPLES to the clamp and still gated at IQR 0.006 on
n=10, which is RULE 2d working as designed: discard, never average.

## ★ THE PREFILL STACK, VERIFIED END TO END: 1.5678 MEASURED AGAINST 1.5725 CLAIMED

Six commits each reported a before -> after on the M4 prefill row, and **every
step's "before" equalled the previous step's "after" to four decimals** -- five
times running. Independent re-measurements do not do that, and RULE 1 exists
because a carried baseline once moved a row across parity. So the chain was
tested rather than trusted: build the commit BEFORE the first lever, build HEAD,
and ABBA the two in ONE pass.

    claimed, compounded
      dbbb26a  (before the chain)                      0.4575
      fc4ca1a  amax hoisted out of the block loop      0.5434   x1.188
      bb65424  half sums from q, f32 rounding          0.5518   x1.015
      f7d46a3  the activation packer is GENERATED      0.6923   x1.255
      46f3fe6  unpack masks hoisted into spare regs    0.7111   x1.027
      318fd38  activation scale applied once           0.7178   x1.009
      f0b785c  the constant pool                       0.7194   x1.003
                                                                =x1.5725

    measured, dbbb26a vs HEAD, one pass, ABBA n=3, IQR/median 0.002
      0.4576 -> 0.7174   relative 1.5678
      jitllm prefill 220.5 -> 347.9 tok/s (1.578x), llama.cpp ~484 in both arms
      i.e. 214 -> 337 Gmac/s over 968.5 Mmac/token at n=507

**★ THE STACK IS REAL: 1.5678 AGAINST 1.5725, A 0.3% DISAGREEMENT INSIDE THE
ROW'S OWN DISPERSION.** And the START reproduces to 0.01% -- 0.4575 recorded,
0.4576 re-measured six commits later.

★ SO THE REPEATED BASELINES WERE NOT A BUG, AND THAT IS WORTH KNOWING FOR ITS OWN
SAKE. The M4's prefill row is stable enough ACROSS SESSIONS that a carried
baseline happened to be the right number every time. That is a property of this
host and this measurement, not a licence: the one step that re-measured its base
in the same pass (f0b785c) found 0.7103 where 0.7178 was carried, a 1.0%
session-to-session wobble that the end-to-end number absorbs. Carrying still
loses the moment the box is not this well behaved -- which is the Linux box
as it stood then, sitting in RULE 2d's power clamp.

**★ AND THE STACK IS WHAT RULE 6 ARGUES FOR, MADE OUT OF PIECES THAT AN EARLIER
SIZE THRESHOLD WOULD HAVE LEFT UNBUILT.** Two of the six levers are under 3% on their own (x1.015 and
x1.009, and the newest is x1.003). Dropping the three smallest leaves x1.528
instead of x1.5725 -- so the "too small to bother" third of the chain is 4.5
points of a 57-point win.

## ★ THE TOWER'S ELEMENTWISE OPS GO ON THE POOL: +5.1%, AND IT TOOK FIVE TRIES TO MEASURE

Jitted is not the same as parallel. act, axpy and addBiasRows were each ONE
generated-kernel call over a whole buffer, on the calling goroutine, while five
cores idled -- 226 MB of traffic an image in ~36 dispatches, every buffer 3-12 MB
against a 1.25 MB L2.

    M4, SmolVLM-256M, 512x512, 1024 patches, ABBA n=7, ONE binary
      537.2 -> 511.6 ms   1.0491x   IQR/median 0.006   gated
      A/A self-control, run FIRST    0.9987x  IQR 0.003
      so ~+5.1%

★ THE MEASUREMENT IS THE STORY HERE, NOT THE NUMBER. Four earlier attempts
produced 0.6026x ("a clear regression", reported as such and wrong), 1.4449x at
IQR 1.131, 1.2587x at IQR 0.188, and a SELF-CONTROL of 0.925 at IQR 0.328. Every
one of them was an instrument failure:

    two binaries, not one   code layout and allocator warm-up differ between
                            builds; at a few percent of a half-second encode that
                            forges the result. The knob (JITLLM_ELEM_POOL) makes
                            both arms the same binary and took the self-control
                            from 0.925/IQR 0.328 to 0.9987/IQR 0.003.
    a stray helper process  left over from a review, load 14.3, PSI 10%, CPU
                            clamped to 1000 MHz. RULE 3c, exactly: a job
                            invisible to every gate here, landing mid-round.
    packagekitd             woke up mid-pass and put two samples at 2030 and 1435
                            ms against a ~550 typical.
    no clamp rejection      the driver averaged those instead of discarding them,
                            which scripts/ab.py has known not to do for a while.
    an unbounded ssh        a dead connection made subprocess.run block FOREVER:
                            41 minutes, zero output, zero processes on the Mac,
                            indistinguishable from slow progress.

**The Linux box never managed a clean reading of this all day. The idle M4 did it
first try.** Instrument quality is not a detail at this effect size: the same
change reads -40%, +44% or +5% depending on what else the machine is doing.

★ AND THE BUG THAT MATTERED WAS NOT A PERF BUG. A review of the helper
found the fallback contract was WRONG for in-place ops: "any chunk declines
-> the Go twin redoes the whole buffer" applies the op TWICE to every chunk that
already succeeded -- GELU(GELU(x)), the residual added twice. It also found the
decline flag was written by several pool workers unsynchronised. Both are fixed
by REMOVING the flag: acceptance is a pure function of (kernel mapped, width,
source length), so nn.Act32Ready/Axpy32Ready preflight without mutating and the
decision is made once for the whole buffer. The race is then gone by
construction rather than by atomic.Bool, and the mixed state is unrepresentable.

## ★ LOWERING THE TOWER'S ATTENTION: 4.17x END TO END, AND ONLY HALF OF IT TILES

Parallelising the head loop (2.69x, below) left the SHAPE wrong: a Go loop still
issued one kernel call per (layer, head, query) -- 147,456 an image -- and each
one re-read the whole K block.

    per (layer, head) at 1024 patches, hd=64:  K is n*hd*4 = 256 KB
    re-read once per query                  -> 268 MB per (layer, head)
    x 12 heads x 12 layers                  -> 38.7 GB an image
    over the then-measured 726 ms           -> 53 GB/s, against a 55.7 GB/s wall
    arithmetic intensity                    -> 0.25 MAC/byte

**★ THE REGISTER BUDGET SAYS ONLY THE SCORES SIDE CAN BE FIXED, AND SAYS IT THE
SAME WAY ON BOTH ARCHITECTURES.** AVX2 has 16 vectors of 8 floats and NEON 32 of
4 -- 128 floats either way:

    SCORES  accumulator is qt SCALARS, one vector each, reduced per key
            qt=8 costs 10 registers      K traffic per query -> 1/8
    ACC     accumulator is qt x hd FLOATS and must stay resident to accumulate
            qt=2 hd=64   16 regs, 1 V pass    V traffic per query 0.50
            qt=4 chunk32 16 regs, 2 V passes  V traffic per query 0.50
            qt=8 chunk16 16 regs, 4 V passes  V traffic per query 0.50

Splitting hd to widen qt multiplies the V passes by exactly the factor it gains.
**0.50 for every legal shape** -- which is why AttnAcc2 exists and AttnAcc4 does
not, and why this change tiles the scores to 8 and leaves the accumulate at 2.

`EmitAttnScoresTiled` bakes hd, kvStride, qStride and scoreStride, as this file
already bakes hd and kvStride, so it needs no new Args fields. K is loaded once
per (key, vector) and feeds qt FMAs against queries that stay MEMORY OPERANDS on
amd64 -- holding them would cap hd at the register file, and gemma's hd is 256.

    SmolVLM-256M, 512x512, 1024 patches, Linux 6 P-cores, ABBA
      tiling alone      812.023 -> 545.644 ms   1.4919x  IQR 0.006  n=6
      END TO END       1992.670 -> 477.178 ms   4.1702x  IQR 0.005  n=6
      A/A self-control, run FIRST                0.9959x / 0.9991x

★ THE END-TO-END NUMBER IS ONE PASS, NOT TWO RATIOS MULTIPLIED. 2.6904 x 1.4919
= 4.01 against a measured 4.17, and the two disagree because they were taken at
different thermal states -- the same reason RULE 1 exists and the same check the
prefill stack got. Quote the measured one.

★ ONLY A BIDIRECTIONAL CALLER CAN USE THIS. Decode attends over pos+1 keys with a
different npos every token and one query at a time; there is nothing to tile. A
vision tower attends every patch to every patch with no mask and no KV cache, so
the queries arrive as a block against one K. `AddAttnTiled` is separate from
`AddAttn` for exactly that reason.

Not bit-identical to the single-query kernel, on purpose: that one runs FOUR
accumulator chains and sums them, this one runs ONE per query because qt
independent chains already cover the FMA latency. Same value to well inside f32;
`TestAttnScoresTiledMatchesReference` checks against a float64 dot product at
1e-5 and reads 0.000e+00 at qt=2/4/8, and fails at 2.75e+03 against a violation
that drops the query row from the address.

## ★ THE TOWER'S ATTENTION WAS SERIAL AND NOBODY HAD PROFILED IT: 2.69x

The +9.0% entry below closed by saying the encode was still 1.95 s, that the
attention looked like a serial double loop, and that this was **unverified by
profile**. It was verified, and the profile was blunter than the arithmetic:

    SmolVLM-256M encode, six P-cores, go tool pprof
      before   Duration 34.74s   samples 60.76s  = 174.89% of one core
      after    Duration 13.03s   samples 67.20s  = 515.84%

**1.75 of 6 cores, 29% utilisation.** `sched.Pool.await` sat at 11.57% and
`nn.JIT.MatMul` was the ONLY thing that ever reached the pool -- `Parallel`
appeared nowhere in engine/model/vision.go. The text path has run its attention on the
pool since forever; the tower never did.

★ AND THE LOOP WAS NEVER WHAT BLOCKED IT -- THE SCRATCH WAS. Every (head, patch)
pair is independent: each writes a disjoint HeadDim slice of xb and reads nothing
another writes. But `att` was ONE shared score row and `acc32` ONE shared
accumulator, so the loop had to be serial to be correct. Per-head slices cost
12*(1024+64) floats -- **52 KB** -- and 147,456 serial kernel invocations an image
become six workers' worth. The three per-patch layernorm loops (24,576 more
dispatches) went on the pool with them.

    SmolVLM-256M, 512x512, 1024 patches, Linux 6 P-cores, ABBA n=5
      1958.727 -> 725.975 ms    2.6904x    IQR/median 0.007
      A/A self-control, run FIRST           0.9991x

**Bit-identical by construction**: arithmetic order within a head is unchanged
and heads do not alias, so TestTowerMatchesLlamaCpp -- the RMS oracle against
llama.cpp's own tower output -- is untouched, and the fallback audit still reads
zero.

★ THE SESSION'S TOTAL ON THIS PATH IS 2127 -> 726 ms, 2.93x, in three steps: the
scale kernel and the deleted float32->float32 copies (+9.0%), then this. The
aggregate rate goes 54.5 -> 146.4 Gmac/s over 106.3 Gmac of work, which is
finally the same order as the ~140 Gmac/s ONE core reaches on a k-quant GEMM --
so the tower is now bound by its kernels rather than by its scheduling.

★ WHAT THIS SAYS ABOUT THE OTHER TIERS. The profile is the cheap check nobody had
run on this subsystem, and it found a 2.69x in one afternoon. RULE 5's "measure
at the thread count that SHIPS" is usually quoted about a win evaporating at six
cores; this is the same rule from the other side -- a path that never reached six
cores at all, sitting behind gates that were all green because correctness and
utilisation are different questions.

## ★ THE VISION TOWER: 2127 -> 1951 ms AN IMAGE, +9.0% FROM ONE KERNEL AND FOUR DELETED COPIES

The eleventh elementwise op (`dst *= alpha`) and the four float32->float32 copy
loops landed with NO perf claim attached, because the call-count arithmetic put
them at 0.09% of a text prefill and RULE 6 says do not quote a sub-noise change.
The TOWER is where the same two changes are not sub-noise.

    SmolVLM-256M, 512x512 image, 1024 patches, Linux 6 P-cores, ABBA n=7
      2127.121 -> 1950.777 ms    1.0837x    IQR/median 0.007
      A/A self-control, run FIRST            0.9935x  (arm B reads 0.65% slow)
      so the effect is ~1.090x

Two causes, both structural rather than clever:

  the score scale   12 layers x 12 heads x 1024 queries x 1024 keys = ~151M
                    multiplies that were a Go loop and are now one kernel call
  q32/k32/v32       three n x NEmbd buffers holding a SECOND copy of q, k and v,
                    refilled every layer by `dst[i] = float32(src[i])` on
                    operands already f32 -- 28.3M element copies and 9.4 MB an
                    image, deleted outright

★ AND THE ENCODE IS STILL 1.95 SECONDS, WHICH IS THE REAL FINDING. 106.3 Gmac of
work (12 layers x [q/k/v/o 2.42 + fc1 2.42 + fc2 2.42 + attention 1.61 Gmac]) in
1.951 s is **54.5 Gmac/s aggregate**, against ~140 Gmac/s that ONE core reaches
on a k-quant GEMM. The tower's attention is a serial double loop --

    for h := 0; h < NHead; h++ { for i := 0; i < n; i++ { scores; scale; softmax; acc } }

-- so 12 x 1024 x 12 layers = 147,456 kernel invocations run on ONE core while
five idle, and the per-patch layernorms are another 24,576 of the same shape.
Attention is 18% of the tower's MACs and the only part off the pool; at 1/6 the
throughput that is ~57% of the wall. **Unverified by profile** -- the arithmetic
is consistent with it, which is not the same as measured, and a cpuprofile is
the next step.

## ★ dareg RESERVED SIXTEEN SUB-BLOCKS OF REGISTERS TO BE READ ONCE, AND Q4_K/Q5_K
## PAID FOR IT WITH THEIR WHOLE CONSTANT POOL

The hoist above and `intFold` both apply the activation scale ONCE, in the
epilogue. The budget did not notice: `wideAct` charged `vpk` vectors to hold
those scales across all sixteen sub-blocks of a super-block, for a value nothing
inside the loop reads any more. `wideAct` implies `intFold || daHoist` -- the two
are exhaustive over `hasMin` -- so that reservation had become unconditional
waste, and it was being paid out of the ONE thing the inner loop wanted:

    arm64 k-quant GEMM, 2x2, window 256, register budget of 32
      Q3_K/Q6_K   need 30    constant pool 2 slots
      Q4_K/Q5_K   need 32    constant pool 0 slots

**An empty pool means `inlineKonst`: every unpack mask rebuilt with a MOVI at
each use, inside the sub-block loop.** Q5_K 4x1 was spending 140 MOVIs per
super-block on three distinct immediates.

**★ AND THE ANSWER IS NOT "NEVER RESERVE" -- IT IS A COUNT.** Loading the scales
in the epilogue instead puts an LDRq three instructions from its consumer where
reserving put it ~1400 away, and on the format that already had its pool that is
a LOSS. So reserve exactly when the pool can still hold every immediate the
unpack asks for: `need + vpk + len(konsts) <= 32`.

    Q3_K  2 konsts, fits either way   -> RESERVE, kernel byte-identical
    Q6_K  3 konsts, pool held 2       -> release
    Q4_K  2 konsts, pool held 0       -> release
    Q5_K  3 konsts, pool held 0       -> release

    emitted instructions at 2x2/256      MOVI per super-block
      Q3_K  1484 -> 1484  (identical)      8 ->  8
      Q4_K  1466 -> 1436                  38 ->  8
      Q5_K  1626 -> 1565                  70 ->  9
      Q6_K  1466 -> 1435                  34 ->  3
    and at 4x1/256, where the narrow tiles run
      Q5_K  1885 -> 1759                 140 -> 14

Every changed kernel is strictly SMALLER and the whole delta is MOVIs. The
narrow (window 32) kernels shrink too, because the pool no longer has to be
sized as if the window were always wide -- that fudge existed to keep
`TestA64GEMMWindowShrinksTheFold` measuring the fold, and with no wide term to
charge there is nothing to fudge.

    M4, one core, at the shapes this model runs, ABBA n=5, IQR/median 0.002-0.005
      Q5_K 5632x2048   126.7 -> 133.9   +5.8%
      Q4_K 2048x2048   143.5 -> 147.9   +3.1%
      Q6_K 2048x2048   147.8 -> 151.9   +2.8%
      Q3_K 2048x5632   144.7 -> 144.2   CONTROL, byte-identical kernel

      A/A self-control, run first       1.0028 / 0.9997 / 1.0017

    M4 prefill ROW (scripts/vs-llamacpp.sh, tinyllama q3_K_M, n=507), ABBA n=3
      0.7103 -> 0.7119   relative 1.0026  IQR/median 0.001, positive every round

★ +5.8% ON 27% OF THE MACs BECAME +0.26% OF THE ROW, AND THE OBVIOUS
EXPLANATION IS WRONG. Share-weighted by KERNEL TIME (not MACs: Q5_K is the
slowest of the three, so its time share is 30.8% against a 27.4% MAC share) the
prediction is -1.9% of the kernel phase. `JITLLM_MMPROF` reads:

      default tiles, 10 cores    837/841 -> 834/837 ms    -0.5%
      forced 2x2,    10 cores    817/819 -> 813/802 ms    -1.3%
      default tiles,  4 cores    808/817 -> 804 ms        -1.0%

One MMPROF run is ONE sample and the band it spans (-0.4% to -1.3%) is wider
than the effect, so the row -- repeated, ABBA'd, gated -- is the number.

**The cache hypothesis was tested and REFUTED.** The isolated sweep's Q5_K tile
is 7.9 MB against the M4's 16 MB L2, so it looked like it was measuring an
L2-resident regime the model is not in. Re-run at 32768 rows (46 MB, well past
L2) the gain is the SAME: 1.0614 against 1.0633. The kernel is compute-bound at
either working-set size and the loss is downstream of it.

★ AND THE E-CORES ARE WORTH NOTHING TO PREFILL ON THE M4. Falling out of the
same profiles, the MatMul kernel phase is 808-817 ms at `JITLLM_CORES=4` and
809-816 ms at 10. RULE 5 records "+34% there" for E-cores -- that measurement is
the LINUX box, and it does not transfer. Nobody has looked at why.

★ THE CONTROL IS BYTE-IDENTICAL AND STILL READS -0.4%, WHICH IS THE NUMBER TO
REMEMBER. Q3_K 2x2/256 emits the same bytes before and after; its -0.4% is what
two different PROCESSES cost, not what the change cost. Every other row is five
to fifteen times that.

### ★ THE EMITTER WAS NONDETERMINISTIC, AND FINDING IT COST AN HOUR

The MOVIs that seed the constant pool were emitted by ranging over the Go MAP
that holds them, and Go randomises map iteration on purpose. One (format, tile,
window) therefore produced a DIFFERENT byte sequence on every run.

Every sequence is correct, which is exactly why it survived: no oracle can see
it. What it breaks is everything that COMPARES emitted code -- and this project
reasons about kernels almost entirely by comparing emitted code. Here it turned
a byte-identical control into an apparent 0.3% register-allocation effect, and
sent an hour into reordering an epilogue that did not need reordering.

`TestGEMMEmissionIsDeterministic` emits each kernel sixteen times and requires
one answer. Against the violation it fails at emission 4.

## ★ THE MIN-CARRYING FORMATS GET THE ACTIVATION-SCALE HOIST: Q4_K +8.5%, Q5_K +7.7%

The decomposition below put 36.9% of a tinyllama prefill's MACs on a fold costing
seven instructions per (sub-block, row, vector) where Q3_K's costs one, and named
`intFold` as the reason: Q4_K and Q5_K carry a min term, and `emitA64KScales`'
`intSc` converts only the MAIN scales because DECODE needs the min ones float
with dmin folded in.

**★ THE INTEGER FOLD IS STILL OUT OF REACH. HALF OF IT IS NOT.** `intFold` bundles
two separable ideas -- keep the dot integer, and apply the ACTIVATION scale once.
The second needs no int32 min scales at all: at window 256 one `d_a` covers the
whole super-block, so it factors out of all sixteen sub-blocks and applies in the
epilogue. That deletes one `FMUL4s` from the main term AND one from the min term:

    per (sub-block, row, vector)    Q3_K (intFold)  1
                                    Q4_K/Q5_K       7 -> 5

**And it costs no registers, because it borrows intFold's structure exactly:**
`facc` becomes the per-SUPER-BLOCK total and the row total lives in the scratch
slot intFold already reserves.

    M4, one core, window 256, at the shapes this model runs
      Q5_K 5632x2048   113.7 -> 122.4   (+7.7%)
      Q4_K 2048x2048   131.1 -> 142.3   (+8.5%)
      Q3_K 2048x5632   145.8 -> 143.9   CONTROL, byte-identical by construction
                                        (daHoist is wideAct && hasMin; Q3_K has
                                        no min term, so its path is untouched)

      share-weighted isolated kernel   133.9 -> 137.5 per core  (+2.7%)
      M4 prefill row                   0.7111 -> 0.7178  IQR 0.002

**★ +8% ON 37% OF THE WORK PREDICTED +2.7% AND THE ROW MOVED +0.9%.** That is the
standing pattern in this file for the fourth time (RULE 5z, RULE 7n, the LDP
pairing): a kernel measured in isolation delivers a fraction of its share at the
real configuration. The row is the gated number at IQR 0.002; the single-run
phase profile reads 471 -> 470 ms of kernel and cannot resolve the difference,
which is what a single run is worth.

★ WHAT IS STILL ON THE TABLE: the other half of intFold -- keeping the dot
integer for Q4_K/Q5_K -- needs int32 min scales, which needs a SECOND flag in
`emitA64KScales` rather than widening `intSc` (gemmk_a64.go:104 warns that
applies dmin twice or not at all), AND a third accumulator array that does not
fit at 2x2. It is worth the remaining 5 -> 3 instructions, and it is a bigger
piece of work than this was.

## ★ WHY jitllm REACHES ONLY 89% OF ITS OWN arm64 KERNELS, DECOMPOSED

The retraction below left a number nobody had explained: jitllm's in-model kernel
rate against its own isolated kernels. Answering it needed two things this
project had never measured -- what prefill ACTUALLY multiplies, and what each
format costs AT THE SHAPE IT RUNS.

**★ THE CENSUS FIRST, BECAUSE BOTH PRIOR MIXES WERE WRONG.** Read off the tensor
table rather than the byte totals, one block of tinyllama q3_K_M:

    ffn_gate   k=2048 rows=5632  Q3_K   26.2% of a block
    ffn_up     k=2048 rows=5632  Q3_K   26.2%
    ffn_down   k=5632 rows=2048  Q5_K   26.2%   <- long k, and the only Q5_K matvec
    attn_q     k=2048 rows=2048  Q3_K    9.5%
    attn_output k=2048 rows=2048 Q4_K    9.5%
    attn_k     k=2048 rows= 256  Q3_K    1.2%
    attn_v     k=2048 rows= 256  Q5_K    1.2%

    by MAC:  Q3_K 63.1%   Q5_K 27.4%   Q4_K 9.5%      0.969 GMAC/token

My own 53/34/10 was FILE BYTES. a review corrected that and got the Q3_K share
right, but put Q4_K at 34.4% and Q5_K at 2.5% -- the census says the opposite,
**Q5_K 27.4% and Q4_K 9.5%**. Two independent wrong mixes on one file is a
reason to read the tensor table and not either summary.

**★ THEN EACH FORMAT AT THE SHAPE IT RUNS. THE SWEEP COULD NOT DO THIS EITHER --
IT WAS PINNED TO ONE k AND ONE ROW COUNT.** M4, one core, window 256:

              k=2048        k=5632
    Q3_K      144.1         145.4
    Q4_K      131.1         128.8
    Q5_K      123.3         113.7

**SHAPE BARELY MATTERS AND THE FORMAT DECIDES.** Long k moves Q3_K +1%, Q4_K
-2%, Q5_K -8%. The spread between FORMATS is 13-22%, and the ordering is exactly
`intFold`: **Q3_K and Q6_K get the integer fold; Q4_K and Q5_K do not**, because
they carry a min term and `emitA64KScales`' `intSc` converts only the MAIN
scales -- the DECODE kernel needs the min ones to stay float with dmin folded
in. gemmk_a64.go:104 says so in its own text.

So **36.9% of this model's prefill work runs `SCVTF` + `FMUL` + `FMLA` per
(sub-block, row, vector) where one `MLAelem` would do**, for a plumbing reason,
and the fix is a SECOND flag rather than widening `intSc` -- which that same
comment warns applies dmin twice or not at all.

**★ AND THE ACCOUNTING NOW CLOSES.** Share-weighted over the real mix and real
shapes, the isolated kernels predict **133.9 Gmac/s per core**:

    Q3_K 2048x5632   52.4% of MACs  ->  48.1% of kernel TIME
    Q5_K 5632x2048   26.2%          ->  30.8%     <- 27% of the work, 31% of the time
    Q4_K 2048x2048    9.5%          ->   9.7%

    measured in-model   224.8 GMAC / 471 ms / 4 cores = 119.3 per core
    predicted                                           133.9
    jitllm reaches                                      89.1% of its own kernels
    unexplained                                         51 ms of 471

    pool regions        6930 per prompt, 5.6 per MatMul call
    at the M4's 3.008 us round trip (RULE 6)            21 ms  = 41% of the gap

**So the 51 ms is ~21 ms of dispatch and ~30 ms of everything else**, and the
larger opportunity is not that gap at all -- it is that a quarter of the work
runs the slower fold. The earlier "84.5%" compared against a Q3_K-ONLY baseline;
weighted properly it is 89.1%, and the difference between those two numbers is
exactly the Q5_K share nobody had priced.

## ★★ RETRACTED: MOST OF THE SECTION BELOW IS WRONG, AND A REVIEW FOUND IT

Asked to attack the conclusion rather than confirm it, a review found four errors.
Three are mine and one reframes the whole comparison. The section is kept
because its METHOD (bound a lever before building it) is right and because a
wrong decomposition left standing is the most expensive thing in this file.

**1. THE MAC DENOMINATOR WAS PARAMETER COUNT, NOT WORK.** I used 1.1 GMAC/token
from "1.1B parameters". The file's actual layer projections are
`22 x (2x2048^2 + 2x2048x256 + 3x2048x5632)` = **968,884,224 MAC/token**, and
the output head runs ONCE after prefill, outside MatMul. So 232 tokens is 224.8
GMAC, not 255 -- the in-model rate is **113 Gmac/s/core, not 128**, and jitllm
reaches **84.5% of its own isolated kernel, not 96%**. "The harness is not the
problem here" does not follow.

**2. THE FORMAT MIX WAS FILE BYTES, NOT EXECUTED WORK.** 53%/34%/10% is the byte
fraction `jitllm info` prints. By projection MACs it is **Q3_K 63.1%, Q4_K 34.4%,
Q5_K 2.5%, Q6_K 0%** -- Q6_K is the output head and Q3_K's share includes the
embedding TABLE, which is bytes and no MACs. So a Q3_K-only tile sweep leaves a
third of the executed work undiagnosed, and `Q4_K` takes the float fold plus a
min correction rather than Q3_K's integer fold.

**3. THE RATIO AND THE PROFILE ARE DIFFERENT RUNS AND I COMBINED THEM.**
232/0.604/495 = 0.776, not the 0.6923 row. The profile is a 232-token prompt and
the gated row is 507 tokens; they cannot be one accounting identity.

**4. AND llama.cpp IS NOT RUNNING ONE KERNEL ON THIS HOST. IT IS RUNNING THREE
IMPLEMENTATIONS, ONE OF WHICH IS APPLE's.** `llama-bench` prints its backend as
**`MTL,BLAS`** even under `-dev none`: BLAS is an ACCEL backend and is
initialised separately from the offload device. Sampling the process during a
507-token prefill (`sample`, 5 s, 4 ggml threads):

    ggml_gemm_q4_K_8x4_q8_K     2389 samples   repacked NEON
    ggml_gemm_q5_K_8x4_q8_K      648           repacked NEON
    dequantize_row_q3_K          210   ]  under ggml_backend_blas_mul_mat
    libBLAS frames               239   ]  -> Accelerate cblas_sgemm
    ggml_vec_dot_q3_K_q8_K         4

**Q3_K -- 63% of the MACs -- has no repacked kernel, so llama.cpp DEQUANTIZES IT
TO f32 AND HANDS IT TO ACCELERATE.** The dequantize is visible at 210 samples
and the multiply itself is nearly invisible, which is what Apple's matrix
coprocessor looks like from a sampler. **And Accelerate runs its own thread pool
-- the sample shows ~221 threads -- so `-t 4` does not bound llama.cpp on macOS
the way it bounds jitllm.**

So "their kernel is ~24% faster per core" is not established. The comparison is
jitllm's NEON kernels against a MIXTURE: repacked NEON on 37% of the work and
Apple's Accelerate on 63%. That is a fair thing to lose to -- llama.cpp ships it
-- but it is not a statement about kernels, and no amount of instruction removal
in `gemmk_a64.go` closes it.

★ WHAT SURVIVES: the tile sweep and the Q8_0 comparison are real measurements of
jitllm against itself. What does NOT survive is using them to bound the GAP.
The review also notes the Q8_0 control is weaker than I claimed even as a
self-comparison -- Q8_0 differs from Q3_K in block size, scale frequency,
accumulator residency AND activation reuse (its emitter reloads activations
inside the row loop where the k-quant emitter shares them), so 153.6 vs 133.8 is
a net difference between two kernels, not an upper bound on unpacking.

★ AND THE "TILE PLATEAU" EXCLUDED TILES BLOCKED BY AN AVOIDABLE LIVE RANGE.
Under `intFold`, `dareg` is loaded before all sixteen sub-blocks and used only in
the epilogue. Moving the load there frees `vpk` registers through the dot loop --
The review puts Q3_K 1x4 at 37 registers as emitted against 29 with the shorter live
range, which would admit **32 token columns**. Not built.

## ★ THE ORIGINAL SECTION, KEPT FOR ITS METHOD

## ★ THE REMAINING arm64 PREFILL GAP IS NOT THE TILE AND NOT THE UNPACK

With the kernel at 94.4% of MatMul, the obvious suspect was the repacked 8x8
GEMM llama.cpp has and jitllm does not. Two sweeps bound both halves of that idea
before anything was built.

**★ 1. THE TILE IS ALREADY AT ITS PLATEAU, AND THE SWEEP COULD NOT SEE IT
BECAUSE IT HAD NO 1xN COLUMN.** The register budget is `2*nacc + ...` with
`nacc = mr*vpk`, so on arm64 a TOKEN costs height-times-as-much as it does
alone: at mr=1 a k-quant reaches 24 token columns, at mr=2 it caps at 16, and
4x2 is refused outright (48 vectors wanted, 32 available). A tile list without
1xN cannot see the trade it is measuring. M4, Q3_K, window 256, one core:

    1x1 ( 8 tok)   94.4 Gmac/s        1x2 (16 tok)  122.4
    2x1 ( 8 tok)  102.0               2x2 (16 tok)  133.8
    4x1 ( 8 tok)  103.1               1x3 (24 tok)  135.5

**TOKENS dominate and they SATURATE.** 8 -> 16 is +30%; 16 -> 24 is +1.3%. Rows
add a little on top (1x2 -> 2x2 is +9%). So the shipping 2x2 is within a few
percent of everything the tile dimension has to give, and making 4x2 legal --
which needs the second accumulator array out of the register file -- is worth
single digits, not the row.

**★ 2. AND THE UNPACK IS BOUNDED AT 12% BY A CONTROL THAT COSTS NOTHING TO RUN.**
Q8_0 has no unpack at all, and at the same tile on the same core it reads **153.6
Gmac/s against Q3_K's 133.8**. So a repack that removed EVERY unpack instruction
-- which is more than llama.cpp's does -- is worth at most 15%. That is the
whole prize of the structural change RULE 1d-septies debated, measured rather
than argued, and it is not 1.44x.

**★ SO WHERE THE ROW ACTUALLY GOES, IN GMAC/s PER CORE:**

    jitllm isolated kernel, Q3_K 2x2         133.8
    jitllm in the model (255 GMAC / 497 ms / 4)  128     -- 96% of its own kernel
    llama.cpp, whole prefill at 495 tok/s        166     -- if its kernel is the
                                                            same 82% share of the
                                                            token that ours is

**jitllm reaches 96% of its own kernel, so the harness is NOT the problem here --
this is the opposite of RULE 5ab's decode finding.** llama.cpp's kernel is simply
~24% faster per core, and that is above jitllm's UNPACK-FREE Q8_0 rate of 153.6,
so the unpack cannot explain it either. At 4.4 GHz, 166 Gmac/s is 2.36 SDOT per
cycle against jitllm's 1.90 on Q3_K and 2.18 on Q8_0: **instructions per SDOT, not
a missing instruction.** There is no i8mm in their build to find (zero smmla).

★ THE NEXT MEASUREMENT, NOT THE NEXT CHANGE: disassemble jitllm's arm64 GEMM inner
loop and count instructions per SDOT, the way RULE 5p did for the decode kernel.
Every estimate above is a rate; the count is what says which instructions to
remove, and this file's history is that a count taken late is a day wasted.

## ★ ONE THING A REVIEW FOUND WAS A ONE-LINE FIX WORTH 8.9% OF THE KERNEL

`inlineKonst`'s own comment has said since it was written that **48 of Q3_K's
~288 instructions per super-block are MOVI rebuilding loop-invariant masks**, and
that the DECODE kernel already hoists them. The GEMM kept calling `inlineKonst`
anyway. At the shipping 2x2 the register budget is 30 of 32, so the two values
Q3_K asks for fit exactly in what the tile left over.

    Q3_K 2x2 kernel   133.8 -> 145.7 Gmac/s   (+8.9%)
    Q3_K 4x1          103.1 -> 118.8          (+15%)
    M4 prefill row    0.6923 -> 0.7111        IQR 0.013

★ THE POOL IS SIZED AS IF THE WINDOW WERE ALWAYS WIDE, which costs the narrow
kernel a register on purpose. `wideAct` charges `vpk` more, so sizing from the
actual budget makes hoisting a function of the WINDOW -- and
`TestA64GEMMWindowShrinksTheFold` compares the two windows' emitted bytes to
prove the wide fold is cheaper. With window-dependent hoisting the narrow kernel
shrinks too: Q5_K 4x1 came out byte-identical and the gate read "the hoist did
nothing", when in fact both effects were present and cancelling. **A gate that
compares two configurations stops meaning anything the moment a change affects
both.**

## ★ THE M4 BOARD — EVERY ROW RE-TAKEN, AND FOUR OF THE FIVE MOVED

`scripts/vs-llamacpp.sh`, n=320, depth 0, 5 rounds, llama.cpp b10825, every row
gated. Taken on the M4 because the Linux laptop sat in RULE 2d's
sustained-power clamp (905-1470 MHz *idle*, 54-56 C) all evening and could not
produce a gated number at all.

| model | tier | ratio | IQR/med | previously | move |
|---|---|---|---|---|---|
| tinyllama q3_K_M | Metal | **1.0200 WIN** | 0.005 | 1.021 | flat |
| gemma-2b | Metal | 0.9509 | 0.013 | 0.9440 | +0.7% |
| tinyllama q3_K_M | CPU | 0.9378 | 0.006 | 0.8276 | **+13.3%** |
| gemma-2b | CPU | 0.9279 | 0.008 | 0.853 | **+8.8%** |
| olmoe-1b-7b | CPU | 0.8456 | 0.073 | 0.7280 | **+16.2%** |
| Qwen3-MOE-4x0.6B | CPU | 0.8392 | 0.013 | -- | first Mac row |

**THE arm64 CPU ROWS ARE WHERE THE YEAR'S WORK LANDED.** 0.8276 -> 0.9378 on
tinyllama and 0.7280 -> 0.8456 on OLMoE are the accumulated arm64 kernel work,
the paired attention kernels and the elementwise generation together, not one
change; the A/B below isolates the last of those at 2.6-4.5%.

★ THE OLMoE ROW'S IQR IS 0.073 AND EVERY OTHER ROW IS UNDER 0.013, SO TREAT IT AS
THE LOOSE ONE. llama.cpp swung 74.5-85.0 across the five rounds while jitllm held
63.1-66.0 -- the dispersion is on the OTHER arm, which is the one thing an ABBA
interleave cannot cancel (RULE 2c). It gates, and it wants a second pass before
the 16% is quoted as precise.

★ AND THE MoE ROWS ARE THE WEAK ONES ON THIS HOST, WHICH IS THE OPPOSITE OF
LINUX. RULE 1d-i records Qwen3-MOE at **1.0289 on Linux CPU** -- a win -- against
0.8392 here. Same model, same quantization, same tool: a 19-point swing that no
property of the model explains, so it is the arm64 side of the MoE path. Nobody
has looked.

★ THE LINUX BOARD IN AGENTS.md PREDATES THIS ONE AND IS NOW STALE IN THE
CONSERVATIVE DIRECTION: it predates the elementwise generation (+2.6-4.5% on
decode), and it was itself taken at 2191 MHz against a ~3500 MHz rested peak.
Re-take it when the box recovers.

## ★ GENERATING THE ELEMENTWISE OPS IS +2.6% TO +4.5% ON THE M4

`jitllm` generated `MatVec` and `MatMul` and ran everything else on Go loops.
Emitting the rest -- softmax, gated and ungated SiLU/GELU, LayerNorm, and
`dst += alpha*src` for the residual, every bias and the MoE accumulate -- on
amd64 AND arm64, and wiring every call site to them:

    M4, tinyllama q3_K_M decode, ABBA, n=160 tokens per sample, 5 rounds
      pass 1   1.0267   IQR/median 0.001
      pass 2   1.0258   IQR/median 0.001
    M4, gemma-2b decode
                        1.0449   IQR/median 0.003
    M4, SELF-CONTROL -- ONE binary in BOTH arms, and run FIRST (RULE 5ac)
                        0.9986   IQR/median 0.000

**gemma gains 1.7x what tinyllama does, and the mechanism says why.** Its
activation is GELU, which interpreted was a non-inlinable `nn.GELUTanh` per
element -- RULE 7b-bis measured that shape at ~295,000 indirect calls a token --
and its FFN is 16384 wide against tinyllama's 5632. An effect that tracks its own
mechanism across two models is a result; the same two numbers without it would be
inside a laptop's noise.

**MEASURED ON THE M4 BECAUSE THE LINUX BOX COULD NOT BE MEASURED AT ALL.** It sat
at 765-1653 MHz with PSI 0.00 and the die at 55-65 C through ten minutes of
enforced idle -- RULE 2d's sustained-power clamp, whose signature is exactly that
(cool AND slow, because a clamped package stops making heat) and whose own verdict
is to stop measuring. The M4's IQR of 0.001 is why that host is the instrument
when this one is unusable.

**THE SELF-CONTROL IS WHAT MAKES THE ROWS READABLE, AND RULE 5ac EXISTS BECAUSE
IT WAS ONCE SKIPPED.** That rule withdrew a whole result after an ad-hoc Mac
script turned out to read 3.5% faster in slot A than slot B for one identical
binary. Here the same harness reads 0.9986 -- a 0.14% slot bias -- so a 2.6%
separation is five times the instrument's own error and a 4.5% one is thirty.

**AND THE MEMORY AXIS IS FLAT TO DOWN, WHICH IS NOT THE SAME AS UNEXAMINED.** Two
f32 staging rows are deleted outright -- `batt` at `chunk*maxSeq` and `bath` at
`n*NHead*maxSeq`, both of which existed only to cross the f64 container boundary
that RULE 8b removed. What is added is the score buffers' padded stride, which is
`maxSeq` rounded up to 16 and therefore ZERO for every maxSeq a power-of-two KV
cache produces, plus a handful of 4 KB mmap'd code pages. Peak RSS on a 64-token
tinyllama decode reads 560116 KB against 560640 KB, median of three, inside a
1.4-1.9 MB run-to-run spread -- i.e. UNRESOLVED at that resolution, which the
arithmetic above predicts.

**★ AND THE WIN IS NOT THE FINDING. THE SOFTMAX PADDING BUG IS.** See
`docs/engineering-history/cpu-kernels.md`: the pad value was exp's low clamp,
which is only inert while every real score is above it, and qwen2's attention
reaches -872. A one-element softmax returned 1.7e-39 where it must return 1, the
model answered fluent repetition, and it took a model that had never been in the
oracle table to expose it.

## ★ RULE 1: the baselines. Every perf claim is a ratio against one of these.

Measured on this laptop, llama.cpp b10819, `taskset -c 0,2,4,6,8,10`:

    tinyllama-1.1B-q3_K_M  decode      68.5 tok/s   = 36.5 GB/s = 65% of the 55.7 GB/s 6P wall
    tinyllama-1.1B-q3_K_M  prefill512 141.8 tok/s   = 281 GFLOP/s
    gemma-2b               decode      27.3 tok/s   = 88.8% of wall  <- almost no headroom; never headline gemma

**★ 1a. THOSE NUMBERS ARE STALE, AND THE BASELINE MOVES. Re-measured on llama.cpp
b10825**, same box, same `-dev none -t 6`:

    tinyllama-1.1B-q3_K_M  decode  74.84 +/- 2.70 tok/s   (was 68.5, +9.3%)
    gemma-2b               decode  29.79 +/- 0.76 tok/s   (was 27.3,  +9.1%)

So **jitllm is NOT at parity on CPU; it is at ~90%** -- 68.4 against 74.84 and 26.7 against 29.79.
Earlier notes in this file and in the design document (ARCHITECTURE.md, since deleted)
claiming parity were true against b10819 and are no longer. A baseline pinned to a version is a baseline that silently rots, so re-measure it before any
claim of the form "we match llama.cpp".

**★ 1b. AND THE GPU TARGET IS FAR HIGHER THAN THE CPU ONE -- BUT SAY WHICH BACKEND.** Same box,
`-ngl 99` on the RTX 3050 Ti. llama.cpp b10825 **Vulkan**:

    tinyllama-1.1B-q3_K_M  130.39 +/- 2.20 tok/s   = 1.74x its own CPU
    gemma-2b                68.12 +/- 0.82 tok/s   = 2.29x its own CPU

and llama.cpp **CUDA**, re-measured in the same session:

    tinyllama-1.1B-q3_K_M  128.18 +/- 1.84 tok/s
    gemma-2b                79.06 +/- 0.53 tok/s   <- 16% faster than its own Vulkan

**Those two are not interchangeable and one comparison in this session treated them as if they
were.** jitllm's tier picks its API by measurement and picks PTX/CUDA on this card, so a jitllm number
belongs against the CUDA row; 68.12 was read as "the gemma target", jitllm reached 70.6, and a commit
message claimed a win that the CUDA number (79.06) does not support. Quote the backend with the
number, always.

The gap was structural rather than a tuning deficit: llama.cpp runs the WHOLE graph on the device,
while jitllm offloaded one matvec at a time and returned to the CPU between them -- so llama.cpp's
entire 14.7 ms token was shorter than jitllm's 22 ms of device time alone. That is fixed; see RULE 12.

**★ 1d. THE SCOREBOARD, COMPLETE ON BOTH HOSTS AND EVERY ROW GATED.**
`scripts/vs-llamacpp.sh`: soak to thermal steady state, ABBA, median of per-round ratios, PSI gate,
IQR/median <= 0.10 or the row is refused. llama.cpp b10825.

    LINUX (i9 6P + 8E, RTX 3050 Ti)          ratio   IQR/med
      llama-2-7b Q4_K_M   CPU                1.012    0.013   WIN
      tinyllama q3_K_M    CUDA  d=0          1.285    0.030   WIN
      tinyllama q3_K_M    CUDA  d=512        1.121            WIN
      tinyllama q3_K_M    CUDA  d=1024       0.935
      tinyllama q3_K_M    CUDA  d=1536       0.874
      tinyllama q3_K_M    CPU                0.919    0.024
      Qwen3-1.7B          CUDA               0.919
      gemma-2b            CUDA  d=0          0.927
      gemma-2b            CUDA  d=2048       0.789
      gemma-2b            CPU                0.848    0.005

    MAC (M4, 4P + 6E)                        ratio   IQR/med
      gemma-2b            CPU                0.853    0.003
      gemma-2b            Metal              0.830    0.006
      tinyllama q3_K_M    Metal              0.824    0.007
      tinyllama q3_K_M    CPU                0.763    0.010

**SUPERSEDED IN PART: MAC METAL tinyllama IS NOW A WIN AT 1.021 (RULE 12q), AND MAC METAL gemma IS
0.9440.** (This line once read 1.0238, which is the BEST of five gated passes -- RULE 12q
says in its own text to quote the median of 1.0253, 1.0211, 1.0238, 1.0121 and 1.0201. A scoreboard
that quotes a rule's best pass while the rule says to quote its median is the scoreboard rotting.) Both are `scripts/vs-llamacpp.sh` at n=320, the same tool and length as every Linux row. The rows below are otherwise current. The old headline read "three wins, all on Linux, and
no host has both tiers"; it is now four wins and the Mac has one, still on one tier per host. The Mac's IQRs are
0.003-0.010 -- an order of magnitude tighter than the Linux laptop's -- because the M4 does not
thermally thrash, so those four losses are the most solid numbers in this file and the worst.
**tinyllama on the Mac CPU at 0.763 is the single worst row anywhere**, and it is the arm64 k-quant
kernel, the same one RULE 5d already improved by 14.4%.

The GPU rows are depth-dependent and the depth is quoted for that reason: jitllm wins tinyllama on CUDA
at 0 and 512 and loses it at 1024 and beyond. Before the attention split (RULE 12l) those same rows
read 1.289 / 0.987 / 0.795 / 0.739, so the fix moved every depth but did not carry the deep ones.

**★ 1d-sexies. THE SCOREBOARD HAS NO PREFILL ROW, AND IT SHOULD: jitllm IS 0.846 OF
llama.cpp THERE.** Every row in RULE 1d is a DECODE row -- RULE 14i already
noted they are all depth-0, and they are all decode as well, so the whole board
measures one of the two things an engine does. Prefill is the other, it is the
dominant cost of a short conversation on a big model, and it had never been
compared. tinyllama q3_K_M, six P-cores, 573-token prompt against `llama-bench
-p 512`, interleaved, median of per-round ratios:

    round 1  jitllm 141.91 142.03   lcpp 166.54 163.67   0.8599
    round 2  jitllm 129.88 130.75   lcpp 157.78 150.35   0.8458
    round 3  jitllm 122.72 126.22   lcpp 151.49 152.85   0.8180
    median 0.8458, spread 0.050 over three rounds

    and after the per-format tile, same harness:
    round 1  jitllm 147.64 144.74   lcpp 163.39 162.93   0.8960
    round 2  jitllm 135.04 135.15   lcpp 159.68 158.90   0.8481
    round 3  jitllm 132.00 128.39   lcpp 151.63 151.55   0.8589
    median 0.8589

**★ THE ROW IS 0.9021 NOW, AND THE KERNEL IS WHERE IT CAME FROM.** Staging the
whole super-block's weights once per row instead of unpacking four at a time
(see the commit) moved the amd64 GEMM by 40% on Q3_K and 36% on Q6_K, and the
row with it. Same harness, four rounds, interleaved, PSI 2.8%:

    round 1  jitllm 152.74 151.22   lcpp 165.92 164.87   0.9189
    round 2  jitllm 142.09 136.22   lcpp 160.25 162.13   0.8633
    round 3  jitllm 143.15 135.96   lcpp 155.88 156.00   0.8949
    round 4  jitllm 138.24 138.87   lcpp 151.51 153.27   0.9092
    median 0.9021, IQR/median 0.050

    0.8458  where the day started
    0.8589  per-format tile (since deleted -- the kernel fix subsumed it)
    0.8860  Q3_K and Q6_K staged
    0.9021  Q4_K and Q5_K staged too

**10.9% of gap remains, and the same disassembly says where it is NOT.** After
staging, Q3_K's VNNI density is 23.8% at {4,2} against 15.0% before -- so the
kernel is still issuing three non-VNNI instructions per VPDPBUSD, and the
remaining ones are the broadcast (structural: VPDPBUSD needs the weight
replicated per lane) and the per-sub-block fold. Attacking the fold is the next
lever with a number behind it; attacking the broadcast needs a different
instruction, not a different loop.

**★ AND THE TILE FIX MOVED IT LESS THAN THE ISOLATED A/B PREDICTED, WHICH IS THE
MORE USEFUL RESULT.** The per-format prefill tile (see the commit) measures
1.0746 jitllm-against-jitllm, ABBA, tuning off on both arms, with a gemma control at
1.0127 that must not move and does not. Re-measuring the ROW against llama.cpp
afterwards reads **0.8589, up from 0.8458 -- a 1.5% move where 7.5% was
predicted.**

Do not resolve that by picking the number you prefer. What is actually
established:

  - the tile change is real (gated, controlled, and the mechanism is counted in
    the disassembly),
  - the ROW is 0.8589 and was 0.8458,
  - and the two cannot be reconciled on this box in this state, because it is
    at 1200 MHz and drifting 15% WITHIN a session -- jitllm read 147.6 in round 1
    and 128.4 in round 3 of the same three-minute comparison.

The likely difference is that the llama.cpp comparison runs the DEFAULT config,
where the tile tuner is live and was already recovering some of the gap on later
chunks, while the A/B pinned both arms with JITLLM_TUNE=0. That is a hypothesis,
not a finding. **RULE 2c's "run the whole comparison twice" caught this; RULE
2d's "when the frequencies are low, stop measuring and let the box rest" is what
to do about it.**

**TREAT IT AS PROVISIONAL: n=3 AND THE BOX WAS CLAMPED AT 1213 MHz** -- RULE 2d's
partial band, where absolute numbers are ~40% down and the rule says to stop
measuring. It is quotable only because BOTH engines drift down together across
the rounds and interleaving cancels what they share; a sequential pass would
have reported the drift as a result. Re-measure on a rested box before building
on it.

**PREFILL IS JIT-COMPILED ON BOTH ARCHITECTURES, so the gap is not a missing
tier.** `EmitGEMM` exists in `jit_old/cpu/gemm.go` (amd64) and `jit_old/cpu/gemm_arm64.go`,
reached through `nn.JIT.MatMul`, with `EmitGEMMK` for the k-quants. What is
missing is arm64's `SMMLA`: RULE 5j records that i8mm is useless for DECODE
because there is one activation vector, and the same rule notes it is a PREFILL
instruction that llama.cpp uses as one. jitllm emits none, on either path. That is
the first thing to price for the prefill row, and it is arm64-only.

**★ 1d-nonies. HALF THE PREFILL GAP WAS THE HARNESS, AND FIXING IT BEAT A DAY OF
KERNEL WORK.** RULE 1d-septies established that jitllm's prefill scales 3.85x on six
cores where llama.cpp gets 4.37x, and that gemma prefill profiles as 94.7%
matvec -- so the non-scaling time is INSIDE nn.JIT.MatMul. JITLLM_MMPROF splits that
bucket into its three phases. gemma-2b, 469-token prompt, 1 core against 6:

    phase       1 core            6 cores          scaling
    kernel      6949 ms 66.1%     1667 ms 59.3%     4.17x
    pack        2656 ms 25.3%      806 ms 28.7%     3.30x
    transpose    895 ms  8.5%      334 ms 11.9%     2.68x

**THE KERNEL SCALES BEST.** Pack and transpose scale worst and GROW as a share of
MatMul when cores are added -- 33.8% to 40.6% -- which is what a non-scaling term
looks like from the inside. Both had the same cause: the work was split along the
axis that SHARES CACHE LINES.

  - **Packing split by TOKEN.** The packed layout is k-group major, so one
    k-group's `tok` tokens are contiguous -- at tok=16 that is exactly 64 bytes,
    one line. Every worker therefore wrote a piece of every line in the output,
    and scale/sum/half are all [block][token] with the same problem. It also
    balanced badly: eight tasks for six workers is a 4x ceiling before any
    coherence cost. Split by k-BLOCK instead: 1.102x, and packing's scaling went
    3.30x -> 4.09x.
  - **The transpose wrote at stride nrows**, a line per element. Blocking eight
    rows and reading the block ONCE into a 512-byte buffer -- gout is
    [row][token], so a run of rows is one flat copy -- is 1.040x then 1.015x.

    gemma-2b   prefill vs llama.cpp   0.7290 -> 0.8555   IQR/med 0.044
    llama-2-7b                        0.8424 -> 0.8729   IQR/med 0.000

**1.163x predicted by composing the three phase A/Bs, 1.173x measured on the
row.** That agreement is the check that the phases were the right decomposition.

**★ AGAINST THE SAME DAY'S KERNEL WORK, WHICH WAS THE WRONG PLACE.** Three kernel
changes measured +40%, +4% and +1-3.4% IN ISOLATION and produced 6.7% and two
results the row could not resolve. This is RULE 11 for the second time and it
said so the first: "three analyses modelled instructions per weight byte and none
measured the thing between the kernels." **Split the profiler's biggest bucket
before optimising anything inside it.**

**★ THE OBVIOUS TRANSPOSE FIX IS A 3x REGRESSION, WHICH IS THE PART TO REMEMBER.**
Swapping to token-outer makes the writes contiguous and measures 135.6 -> 43.7
tok/s: reading gout[r*tok+i] down consecutive r pulls a new line per element, so
the whole of gout is re-read n times -- 1 MiB sixteen times per tile on gate/up.
"Make the writes contiguous" is the right instinct and the wrong trade when it
costs an n-fold re-read of the source. The question is not which side to make
contiguous; it is not re-reading either side.

**★ AND A REVIEW FOUND THE PACK ONE, HAVING BEEN TOLD I HAD SPENT THE DAY ON
KERNELS.** Its first ranked hypothesis was the cache-line ownership, above the
barrier costs I had assumed; it also corrected my sizing of the q/k/v
quantization dedup from 3/7 to 3/14 of packed elements, because `down` packs an
eight-times-wider input. RULE 12i recorded the same thing about the same tool:
"the value was not GPU expertise; it was that it had no stake in the theory being
defended."

**★ RESOLVED, AND THE HALF THAT MATTERED WAS THE HALF THAT WAS WRONG.**
A review found that `engine/sched/pool.go`'s `active` counts SPAWNED workers rather than
PARTICIPANTS, and inferred that `JITLLM_CORES=N` is therefore not an N-core pool and
that every core-scaling figure in this file is measuring something other than
what it claims. **The mechanism is real and the consequence is not.**

`Do` sets `p.active.Store(int64(len(p.parked)))` -- all spawned workers -- while
the work is split over `p.part.Load()`, so a worker the PARTICIPANT CAP excludes
still has to acknowledge the region. That is exactly what the review described, and it
is a real cost of `SetParticipants(n < len(cpus))`.

But `JITLLM_CORES` does not go through the cap. `DecodeCores` truncates the CPU LIST
(`if len(p) > n { p = p[:n] }`) BEFORE `New(cpus)` builds the pool, and `New`
spawns exactly `len(cpus)-1` goroutines and stores `part = len(cpus)`. So at
`JITLLM_CORES=N` there are N-1 workers plus the caller, spawned == participants, and
the pool is an N-core pool. **Every core-scaling figure in this file that used
`JITLLM_CORES` stands, including 3.85x against 4.37x.**

★ THE LESSON IS ABOUT THE SHAPE OF THE REPORT, NOT ABOUT THE REVIEWER. A correct
observation about one code path was carried to a conclusion about a different one,
and it sat in this file as "every number here is suspect" -- which is
the most expensive kind of open item to leave lying around, because it discounts
work that was never in doubt. **Verify the CONSEQUENCE, not just the mechanism,
before recording that a finding invalidates measurements.**

**★ 1d-octies. THE arm64 GEMM's HOISTED ACTIVATION SCALE IS WORTH 1-3.4% ON THE
M4, AND THE VARIATION IS THE PART THAT MAKES IT BELIEVABLE.** arm64 runs
WideActWindow 256 for an all-k-quant model, so every sub-block of a super-block
shares one activation scale -- and the fold reloaded it once per (sub-block,
row, vector) anyway. Loading it once per super-block, measured on the M4 at the
{2,2} tile, ABBA, three samples per arm:

    format   win 32   win 256   ratio    per-sub-block d_a loads removed
      Q3_K    108.6     109.6   1.009    vpk
      Q6_K    111.9     114.2   1.021    vpk
      Q4_K    127.6     132.0   1.034    2*vpk  <- the min term loads it twice

**Q4_K AND Q5_K CARRY A MIN TERM WHOSE FOLD LOADS d_a A SECOND TIME**, so twice
the loads are removed and the gain is three times Q3_K's. An effect that tracks
its own mechanism across three formats is a result; the same three numbers
without that explanation would be inside the noise of a machine running a
window server and an animated wallpaper.

**★ AND IT IS STILL FAR UNDER THE INSTRUCTION COUNT, WHICH IS RULE 5z's LESSON
FOR THE THIRD TIME.** The change removes 22% of the fold's instructions and 5-7%
of the emitted kernel, and delivers 1-3.4%. RULE 5q found the same thing by the
same route -- "LD1x4 removes 13 instructions and gains nothing... the activation
vector is hot in L1 and the loads were never on the critical path" -- and the
activation-scale array is smaller still. **On the M4, removing a load that hits
L1 is worth a fraction of what its instruction count suggests. Predict loads at
a discount or measure them.**

★ I ALSO CALLED THIS RESULT AT ZERO AFTER FOUR SAMPLES OF ONE FORMAT, which was
wrong and is worth recording separately: Q3_K alone reads 1.006 over its first
two pairs, and the finding only appears once Q4_K is included. **A per-format
sweep exists so that the formats disagree; reading the first one and stopping
throws away the only signal that distinguishes a result from drift.**

**★ THE CONTRACT IS EXPLICIT NOW, AND THAT IS WORTH MORE THAN THE 3%.** The
kernel is TOLD the window (`EmitGEMMWindow`) rather than inferring it, because
WideActWindow returns 256 only for an all-k-quant model and drops to 32 the
moment the model contains a Q4_0 or Q8_0 tensor -- the same kernel shape sees
both. Emitting for 256 while packing at 32 applies one sub-block's scale to all
sixteen, and on the M4 that measures NMSE 3.6e-2 to 5.1e-2 against a 1e-10 gate:
`TestGEMMWideWindowMatchesReference` is that gate, and it had to be run on the
Mac to mean anything, since amd64 ignores the window entirely.

**★ AND THE FIRST VERSION OF THAT TEST COMPARED EMITTED BYTE COUNTS.** It checked
that the wide kernel was SHORTER, which is not a claim about arithmetic -- the
wide path had never been executed at all when I first called it verified. RULE
14r's "a skipping test is a green line that means nothing", one step sideways:
a test that measures code size and is described as correctness.

**★ 1d-duodecies. THE GPU PREFILL ROW WAS 0.0298 AND IS NOW 0.2431 -- SEE 1d-terdecies. THIS RULE
IS KEPT FOR THE DIAGNOSIS, WHICH WAS RIGHT, AND ITS NUMBERS ARE RETIRED.** `scripts/vs-llamacpp.sh gpu` in the new prefill mode, matched token
counts, tinyllama q3_K_M, RTX 3050 Ti:

    round 1  jitllm 150.90 151.14   llama.cpp 4996.22 5149.87
    round 4  jitllm 149.74 152.64   llama.cpp 5081.90 5107.20
    ratio 0.0298   IQR/median 0.022, n=8

**llama.cpp does 5,100 tok/s of prompt where jitllm does 151 -- 34x.** The cause is one line:
`engine/model/prefill.go` falls back to per-token `Forward` whenever `s.gpuLayers > 0`, so a 507-token
prompt is 507 sequential single-token device submissions. jitllm's device prefill rate (151) is
therefore just its device DECODE rate (160), because it IS decode run 507 times.

**★ THIS IS RULE 12's OWN FINDING ONE LEVEL UP.** RULE 12: "serving one matvec at a time cost a
blocking round trip per matvec ... `nn.LayerDevice` moves the unit to a transformer BLOCK". The
unit for prefill is still one TOKEN. The same sentence applies unchanged.

The contrast that sizes the prize, per tier, against its OWN decode rate:

    CPU   decode  73   prefill 145   batching buys 2.0x
    GPU   decode 160   prefill 151   batching buys NOTHING -- there is no batching

**★ AND THE CPU PREFILL ROW AT MATCHED LENGTHS IS 0.8720, NOT THE 1.0370 THIS FILE REPORTED.**
Same tool, `JITLLM_MODE=prefill`, 507 tokens BOTH sides (IQR/median 0.029, n=6, two RULE 2d clamps
discarded). RULE 6b-bis explains the difference: every earlier prefill row ran jitllm at 573 tokens
against `llama-bench -p 512`, and llama.cpp's default `n_ubatch` IS 512, so 573 gave it a ragged
tail. **Quote 0.8720. Treat 1.0370 as retired.**

★ CAVEAT ON BOTH ROWS: the package sat at 1681-1936 MHz with two samples clamped to ~900, which is
the top of RULE 2d's partial band. The GPU row is unaffected (a 34x gap does not turn on 15% of CPU
clock) and the CPU row should be re-taken on a rested box.

**★ 1d-terdecies. AND IT IS 0.4151 NOW -- 13.5x IN ONE SESSION. RULES 12u THROUGH 12ac ARE THE
STEPS; THE 0.2431 BELOW IS WHERE THE BATCHING ALONE LANDED.** Same tool, same prompt, same gate:

    round 1  jitllm 1229.68 1256.41   llama.cpp 5360.15 4899.54
    round 4  jitllm 1219.24 1214.15   llama.cpp 4905.80 4894.36
    ratio 0.2431   IQR/median 0.047, n=5 (3 clamped samples discarded)

The chain, all on tinyllama q3_K_M and every step bit-identical to the per-token device path:

    152 tok/s   per-token device prefill (the bug)
    496         batched: Layers takes a chunk of 128 rows, Tok=16 token columns per thread
    607         Rowt=4 rows per thread, per-row causal bounds, no graph capture
    1229        a 2-D tile on the scores kernel, qt=4 kt=2

**THREE SEPARATE THINGS WERE THE COST AND EACH ONE ONLY BECAME VISIBLE ONCE THE ONE BEFORE IT WAS
GONE**, which is why the probe matters more than any of the fixes. `JITLLM_GPU_SKIP` takes a comma
list -- scores, softmax, acc, attn, ffn -- omits that phase from the submission, and produces wrong
tokens on purpose. There is no per-kernel profile of a device submission that does not serialise
the stream, and serialising it would measure the serialisation.

    phase              at 496 tok/s   at 607   at 1050
      scores                              326       99
      softmax                               3
      attnAcc                              38       49
      attention total                     401      163
      FFN projections                     243     ~230
      everything else                     132      132

**THE ORDER WAS: DISPATCH, THEN THE MATVEC's LOAD RATIO, THEN ATTENTION's.** And the last two are
the SAME defect in two kernels -- a thread that loads one operand per multiply is issue-bound
whatever else is true -- which is worth more than either fix: **on this device, count loads per FMA
before counting anything else.**

**★ 1d-undecies. THE PREFILL BOARD AFTER THE HARNESS WORK, AND THE 4->6 "DIP" WAS
THE TUNERS.**

    model         prefill vs llama.cpp    IQR/med
      tinyllama     0.8458 -> 0.8720       0.029   RETIRED: the 1.0370 this row used to
                                                 read compared 573 tokens against 512;
                                                 see RULE 1d-duodecies
      llama-2-7b    0.8424 -> 0.8729       0.000
      gemma-2b      0.7290 -> 0.8555       0.044

**★ AND THE 0.9751 THAT ROW USED TO READ WAS A SESSION BIAS, NOT A DIFFERENT
BUILD -- WHICH IS RULE 2c's WARNING FIRING FOR REAL AND MOVING A ROW ACROSS
PARITY.** The pre-fold binary was re-measured against llama.cpp on the SAME day
as the fold, in the same interleaved shape, and read **1.0015 at IQR/med 0.002**
where it had read 0.9751 at IQR 0.013 that morning. Same code, same script, same
PSI gate, same clamp gate. llama.cpp held 157-158 in both passes; jitllm read 153
in one and 157.7 in the other.

So a 2.7-point bias survived an ABBA interleave, a PSI gate, a clamp gate and an
IQR of 0.013 -- exactly the failure RULE 2c names ("a dispersion gate cannot see
a bias that is common to both arms"), and this time it was the difference between
"2.5% from a win" and "at parity". **The three fold numbers only agree because
the pre-fold arm was re-measured on the day**: 1.0015 -> 1.0370 is +3.5%, the
direct binary A/B is 1.0366, and the two independent fold-vs-llama.cpp passes are
1.0379 and 1.0370. **Never carry a baseline across sessions; re-measure the arm
you are comparing against, in the same pass.**

**★ AND THE CORE-COUNT DIP DOES NOT EXIST ONCE THE TUNERS ARE PINNED.** 1d-decies
recorded the ratio peaking at four cores (0.915) and falling at six (0.892) and
called it unexamined. It is not hardware. The 4-core and 6-core runs were not
comparable -- 1260 calls / 11340 pool regions against 1890 / 6201 -- because the
prefill-chunk tuner picked different widths, and `PrefillChunk` also calls
`SetParticipants(f.ptune.pick())`, so the participant count is tuned too. With
`JITLLM_TUNE=0 JITLLM_CHUNK=32 JITLLM_PART=cores`:

    cores   jitllm pinned    lcpp    ratio   jitllm scale   lcpp scale
        1        33.55   36.23    0.926       1.00x        1.00x
        2        63.36   70.87    0.894       1.89x        1.96x
        4       103.81  116.51    0.891       3.09x        3.22x
        6       144.15  158.49    0.910       4.30x        4.37x

    non-scaling term   jitllm 34%   llama.cpp 33%   -- equal

The curve is flat, not a dip, and jitllm's scaling now matches llama.cpp's. **A
scaling curve measured with three live tuners underneath it is measuring the
tuners.** Pin JITLLM_TUNE, JITLLM_CHUNK and JITLLM_PART before reading one.

**★ THE TUNERS COST 2.4% AT FOUR CORES AND 3.9% AT SIX, AND DO NOT EARN IT
BACK ON ONE PROMPT.** They are deliberately serialized -- "one at a time: batch
width is not measurable while the participant count is still moving underneath
it" -- so a single prompt settles one of three. This is the tile tuner's problem
(RULE 1d-sexies) in the other two tuners: **for a one-shot prompt the DEFAULT is
what ships, and the tuner is a cost.** The cached answers are fine; running the
tuner is what costs.

★ A THEORY THAT WAS TESTED AND DISPROVED: `tuneKey`
hashes shapes, host and core count and NOTHING about the kernel, so a cached
answer survives a kernel change that invalidates it -- the cache holds tile
choices like 8x1 and 6x2 which a fresh sweep measures at 42.4 and 108.5 Gmac/s
against {4,2}'s 112.6. That looked like it would explain the tuner cost. It does
not: the existing cache measured ~2% FASTER than a freshly-tuned one on both
models, because re-tuning costs more than a stale answer does. The missing
kernel key is a real hazard with no demonstrated cost.

**★ 1d-decies. AND THE SCALING HALF IS NOW FIXED -- jitllm SCALES BETTER THAN
llama.cpp. THE RULE BELOW IS SUPERSEDED AS A DIRECTION.** After the k-block pack
split and the buffered transpose, the same measurement reads:

    cores   jitllm was   jitllm now    lcpp   ratio was   ratio now
        1     31.83     31.22   36.23      0.879       0.862
        2     59.29     62.83   70.87      0.837       0.887
        4     93.09    106.61  116.51      0.799       0.915
        6    122.44    141.40  158.49      0.773       0.892

    scaling 1->6        3.85x -> 4.53x     llama.cpp 4.37x
    non-scaling term      43% -> 29%       llama.cpp 33%

**THE GAP NO LONGER WIDENS WITH CORES; IT NARROWS.** 1d-septies' headline -- "half
the prefill gap is not the kernel, it is scaling" -- was true when written and is
not any more, and its DIRECTION is now backwards: the remaining prefill deficit
is the SINGLE-CORE kernel, 0.862, and the harness is ahead of llama.cpp's.

So the next prefill lever is the kernel after all, which is the opposite of what
this file said in the entry before. **Keep 1d-septies for its method -- splitting the
profiler's biggest bucket is what found the pack bug -- and do not inherit its
conclusion.**

★ ONE THING IT LEAVES: the ratio peaks at FOUR cores (0.915) and falls at six
(0.892), which neither engine's scaling curve explains on its own. Unexamined.

**★ 1d-septies. HALF THE PREFILL GAP IS NOT THE KERNEL, IT IS SCALING -- AND I WAS
ONE COMMIT FROM OPTIMISING THE WRONG THING.** gemma-2b prefill, six P-cores,
`llama-bench -t 1,2,4,6` against `jitllm run`:

    cores      jitllm     lcpp   ratio   jitllm scale   lcpp scale
        1    31.83    36.23   0.879       1.00x        1.00x
        2    59.29    70.87   0.837       1.86x        1.96x
        4    93.09   116.51   0.799       2.92x        3.22x
        6   122.44   158.49   0.773       3.85x        4.37x

**THE GAP WIDENS WITH CORES.** At one core jitllm is 0.879 of llama.cpp; at six it
is 0.773. Fitting T(n) = W/n + S, the non-scaling term is **43% of jitllm's
six-core prefill against llama.cpp's 33%**. So roughly half the prefill deficit
is the kernel and half is the harness around it -- and every hour of this
session went into the kernel.

This is RULE 11 for a second time, on the other machine of the engine. There the
finding was "the bottleneck was DISPATCH, not the kernel; three separate
analyses concluded the k-quant kernels were the gap and all three were wrong",
and the fix was worth 1.37x. Prefill has never had that measurement made.

**★ AND THE KERNEL HALF IS NOT WHERE THE INSTRUCTION COUNTS POINT EITHER.**
`nm -D` on libggml-cpu-alderlake.so: llama.cpp has REPACKED 8x8 GEMM kernels for
q2_K, q4_0, q4_K, q5_K, q6_K, q8_0, iq4_nl and mxfp4 -- and **none for Q3_K**.
It repacks weights at load into a row-interleaved layout so eight rows of one
k-group are contiguous. jitllm has no repacked layout at all and reads the GGUF
rows directly. Prefill against llama.cpp, by how much of the model it can
repack:

    model         repackable   prefill ratio   IQR/med
      tinyllama       47%         0.9021        0.050    <- Q3_K is not repacked
      llama-2-7b     100%         0.8424        0.010
      gemma-2b       100%         0.7290        0.067

**jitllm's best prefill row is the one model whose dominant format llama.cpp cannot
repack**, which is the opposite of where a day of Q3_K kernel work was aimed.

**★ BUT DO NOT CONCLUDE "BUILD A REPACKED LAYOUT" FROM THAT.** The orientations
are mirror images and jitllm's is the better one on paper: VPDPBUSD does 32 MACs
either way, and jitllm puts TOKENS in the lanes and broadcasts the weight (0.5
broadcasts per VNNI at {4,2}) where a row-interleaved layout puts ROWS in the
lanes and broadcasts the activation (1.0 per VNNI). What repacking buys is that
the nibble unpack applies to eight rows at once, which only exists in the
row-lane orientation. And a PRE-UNPACKED repack loses by arithmetic: one
byte per weight is 2.7x Q3_K's bytes, the weights are read ~36 times at pp512,
and that adds ~0.86 s to a ~4.4 s prefill. llama.cpp keeps its nibbles packed
for exactly this reason.

**So the order is: measure why the GEMM scales 3.85x where llama.cpp gets 4.37x,
BEFORE any more kernel work.** gemma prefill is 94.7% matvec by profile, so the
non-scaling 43% is inside MatMul -- the activation packing, the row-group split,
the output transpose and the dispatches between them, none of which has ever
been timed separately.

**★ 1d-quinquies. tinyllama ON LINUX CPU IS 0.9790, AND THAT IS THE CLOSEST LOSING ROW ON THE BOARD.**
Measured on a RESTED box (63 C, PSI 0.00) with the RULE 2d clamp gate live: IQR/median 0.012 at n=8,
with two samples discarded from a round where both engines fell to ~900 MHz. Earlier readings of
0.9596 and a rejected 1.0068 came from a box that had been hammered for hours; this is the number.

**It is 2.1% from parity, and Linux already wins tinyllama on CUDA at 1.285.** So this single row is
what stands between the scoreboard and "both tiers on ONE host on ONE model", which is a stronger
claim than the current "both tiers, different models" (CPU on llama-2-7b, GPU on tinyllama).

**Whole-layer fusion is bigger than the gap** -- RULE 6's Linux column reads 3.03% for siblings plus
elementwise epilogues and 4.80% for everything but the reductions, against 2.1% needed. Take the
SOUND one: 3.03% is the column whose barriers can actually be removed, because a worker that
produced rows [lo,hi) of a matvec already owns those rows for the epilogue that follows. The 4.80%
column assumes barriers can also go between MATVECS, and they cannot -- a matvec reads the whole
input vector, so it needs every worker's output from the op before it. 0.979 x 1/(1-0.0303) = 1.009.

**A marginal win, and it is the only route to a same-model both-tiers result that this file can
price.** Build it for Linux tinyllama; the Mac gets 3.88% from the same change (RULE 6's M4 column)
and lands at 0.861, which is progress on the right subsystem (RULE 5ab) and not a win.

**★ AND FUSION DOES NOT CLOSE THIS ROW EITHER, BY RULE 6's OWN CORRECTION, WHICH I QUOTED PAST
FOUR TIMES.** Every fusion ceiling in this file is computed from the EMPTY-region round trip, and
RULE 6 says in its own text why that overstates the saving: "splitting real work measures 1.596 us
per region against 2.697 us idle, because dispatch overlaps with computation and an idle round trip
has nothing to hide behind". The in-context cost is 0.592 of the idle one:

      Linux tinyllama  siblings+epilogues   3.03% quoted -> 1.79% real  ->  0.9969
      Linux tinyllama  all-but-reductions   4.80%        -> 2.84%       ->  1.0076
      M4    tinyllama  siblings+epilogues   3.88%        -> 2.30%       ->  0.8470
      M4    tinyllama  all-but-reductions   6.15%        -> 3.64%       ->  0.8589

**The only row that crosses is all-but-reductions on Linux, and that column is the UNSOUND one** --
it assumes barriers can go between matvecs, and they cannot, because a matvec reads the whole input
vector and so needs every worker's output from the op before it. The sound column lands at 0.9969.

**So whole-layer fusion is not a route to any win on either host, and the ~2% it is worth on Linux
should be built for its own sake or not at all.** This is the same failure as RULE 5z's probe
coefficients one layer up: a per-unit cost measured in ISOLATION overstates what removing units
saves, because in context the thing being removed was overlapping with work. Two independent
instances in one session is enough to make it a rule -- **when a lever is priced as N x (cost of one
thing measured alone), find out what that thing costs IN SITU before believing the product.**

**★ AND THE 3.03% IS NOT SEVEN REGIONS -- TWO OF THEM ARE NOT REGIONS AT ALL.** RULE 6 describes the
epilogue column as "rope, both residual adds, the SiLU-multiply", but in `engine/model/forward.go` the
residual adds are plain SERIAL loops:

    for i := range s.x { s.x[i] += s.h[i] }

They are not dispatched, so folding them into a producing region removes no round trip -- it removes
SERIAL TIME instead, which the profile puts at 0.4% for both of them together, on one core while
five idle. Worth doing, and worth about 0.3%, not the ~0.9% two regions would be.

So the achievable decomposition against the 2.1% needed, per layer:

    q/k/v merged into one region      2 regions   0.87%   <- the hard one: needs a multi-output matvec
    gate/up merged                    1 region    0.43%
    rope folded into its producer     1 region    0.43%
    SiLU-multiply folded into `up`    1 region    0.43%
    residual adds parallelised        0 regions   0.30%   <- serial now
    --------------------------------------------------------
    all five                                     ~2.46%   ->  0.979 -> 1.003
    everything except the q/k/v merge            ~1.59%   ->  0.979 -> 0.995

**The q/k/v merge is load-bearing and it is the expensive one**, because `s.mv` issues one region per
matvec and merging three means a region that writes three different outputs from one input. Without
it the row lands at 0.995 and stays a loss. Anyone starting here should build that first, not last.

Linux is also a TIGHTER budget than the Mac: `JITLLM_PROFILE=1` reads matvec 94.0% / non-matvec 6.0%
there against 90.8% / 9.2% on the M4, which is why the same change is worth 3.03% on one host and
3.88% on the other.

**★ 1d-quater. SAMPLE LENGTH CHANGES WHICH ENGINE WINS, AND SHORT SAMPLES FLATTER jitllm.** tinyllama
CPU was reported at 1.0143, then 1.070, 1.156, 0.920, 0.996 across five passes -- four refused on
IQR, and the one that passed said WIN. Lengthening each sample from 64 to 320 tokens and soaking six
rounds instead of three, two consecutive passes read **0.9348 (IQR 0.033) and 0.9479 (IQR 0.037)**.
Those agree, they gate, and they say jitllm LOSES.

The mechanism is in the raw samples, not in the summary. Inside one interleaved pass llama.cpp held
75.2-76.8 -- under 2% -- while jitllm swung 65.6-75.3. Same box, same interleave, same thermal exposure.
Two things cause it and neither is noise:

1. **`llama-bench` warms up and `jitllm run` does not.** llama-bench defaults to one discarded warm-up
   pass, so its measured window starts with the machine already busy; jitllm's rate covers every token
   from process start. At n=64 the ramp is ~15% of the window and at n=320 it is not.
2. **jitllm degrades MORE under sustained load.** Across a long pass jitllm falls further than llama.cpp
   does, so the ratio itself is a function of how long the generation is: near parity for a short
   reply, 0.94 sustained. That is a power story, not a scheduling one, and it is a second reason to
   keep deleting instructions from the kernels rather than only counting cycles.

**So a comparison must fix the generation length and say what it was.** Short samples are not a
faster way to get the same answer; they are a different question, and they are the question that
makes jitllm look better.

The tuner was ruled out on the way: interleaved `JITLLM_TUNE=1` against `JITLLM_TUNE=0` on tinyllama reads
0.9946 at IQR/median 0.038, and both arms show the same ~15% spread. The variance is thermal and the
tuner neither causes nor cures it.

**★ 1d-bis. THE EARLIER NUMBERS IN THIS SECTION WERE FROM COLD SINGLE RUNS AND WERE WRONG BY UP TO
14%.** They came from `jitllm verify`, which runs each tier once. RULE 2c records what that costs: two
back-to-back passes of one A/B read 0.9733 and 0.8530, both passing the IQR gate. A "first CPU win"
was reported off such a pass and did not survive soaking. Quote nothing from a single run.

**★ 1d-ter. ON A 4 GB CARD llama.cpp REFUSES THE 7B AND jitllm RUNS IT.** `-ngl 99` on llama-2-7b
Q4_K_M (3.8 GiB) prints `failed to load model` -- the desktop already holds ~1.2 GiB of the 4096 MiB
(Xorg alone is 596 MiB), so the weights do not fit and there is no graceful degradation. jitllm's
`PrepLayer` declines per block, so it puts what fits on the device and runs the rest on the host: it
decodes at 17.3 tok/s with no flag tuning at all. llama.cpp's best working setting is a hand-found
`-ngl 16` at 17.6 tok/s, and `-ngl 20` is SLOWER at 12.25 because it thrashes.

Interleaved at `-ngl 16` the ratio is 0.984 at IQR/median 0.293 -- **the row fails the gate**: both
engines are partially offloading against a contended card and the samples swing 4x. The capability
difference is real and reproducible; no ratio is quoted for this row.

**★ RE-TESTED WITH THE RULE 2d CLAMP GATE ON A RESTED BOX, AND IT IS WORSE: IQR/median 0.966.** jitllm
read 8.88-16.63 and llama.cpp 9.30-16.83 in the same four rounds, with 1248 MiB of the 4096 already
held by the desktop. Two samples were discarded as CPU-clamped and the rest still swing 2x, so the
dispersion is GPU and PCIe contention rather than anything the clamp gate can see. **This row is
UNMEASURABLE on this hardware, which is not the same as jitllm losing it** -- and it is why the Linux
CPU win (llama-2-7b) and the Linux GPU win (tinyllama) cannot be made to land on the same model
here. A card that fits the 7B would settle it; this one cannot.

**★ 1e. AND THE PER-SHAPE TABLE HAS THREE DEVICE COLUMNS, NOT TWO -- I READ THE WRONG ONE.**
`jit_old/gpu/cmd/mvbench` prints one block per backend present, and on this box that is PTX first and
SPIR-V second. Quoting the second block's rows as CUDA understated CUDA by 10-18 points and made the
Metal gap look smaller than it is. Each figure is against a read wall measured in the SAME process:

    gemma's real decode shapes         CUDA/PTX   Vulkan/SPIR-V   Metal/M4
      k/v (GQA)   256 x  2048  Q4_0       48%          30%          35%
      q/o        2048 x  2048  Q4_0       85%          75%          67%
      qkv fused  2560 x  2048  Q4_0       87%          81%          70%
      gate+up   32768 x  2048  Q4_0      106%         105%          74%
      gate/up   16384 x  2048  Q4_0      108%         105%          73%
      ffn_down   2048 x 16384  Q4_0      106%         101%          73%
      lm_head  256128 x  2048  Q8_0      110%         112%         108%

CUDA's big shapes are AT the wall (over 100% means the `Stream` probe understates it -- RULE 12c).
**Metal's Q8_0 row is at the wall and every Q4_0 row is 30 points below it**, which is what is left of
the unpack after RULE 12j. That contrast, one format against another on one device, is the whole
diagnosis and needs no profiler.

**★ 1c. THE OTHER HOST. M4 Mac (4 P + 6 E cores), llama.cpp b10825 -- and the
llama.cpp numbers are UNCHANGED from the last time, so these baselines are good.**

    model        jitllm CPU  lcpp CPU   ratio    jitllm Metal  lcpp Metal  ratio
    tinyllama      54.4     99.34    0.55x       69.5      114.33    0.61x
    gemma-2b       38.6     42.57    0.91x       33.9       56.60    0.60x

Both Metal figures are with the whole-block path and the output projection resident on the device
(RULE 12), and both produce token ids identical to the CPU tier. gemma was 15.8 before that landed.

Two things this isolates, and they are different problems:

**The arm64 CPU gap is K-QUANTS ONLY.** gemma is Q4_0+Q8_0 and runs at 0.91x; tinyllama is 100%
k-quants and runs at 0.55x. It is not a missing kernel -- all six formats have an arm64 decode
kernel -- and it is not the thread count: sweeping `JITLLM_CORES` on the M4 gives 31.4 / 53.9 / 53.7 /
53.8 / 53.4 at 2/4/6/8/10, so it saturates at the four P-cores exactly as it should and llama.cpp
gets 99.34 from the same four. The arm64 k-quant decode kernel is about 1.85x off.

**gemma on Metal is SLOWER THAN THE M4's OWN CPU** (33.9 against 38.6). The GPU tier is worth having
there for tinyllama and not yet for gemma, which is the sharpest statement of where the MSL kernels
stand: `Dot4` is one instruction on NVIDIA (dp4a) and has no Metal equivalent, so MSL emulates it.

**★ THAT LAST PARAGRAPH IS STALE AND THE MSL KERNEL WORK FIXED IT. RE-MEASURED: gemma on
Metal is 54.51 against the M4's CPU at 38.69, or 1.41x.** RULES 12j, 12k and 12m are why -- the
float-carried `Dot4`, the accumulator that stops pretending to be an integer, and
`unpack_snorm4x8_to_float`. **It was quoted as a live result while building the seam tuner, as the
one case where the device loses to its own host, after it had already stopped being true.** There is
now no model on either host where the device is slower, which is worth knowing before designing
anything that exists to detect one.

**Use `-dev none`, NEVER `-ngl 0`.** ggml's scheduler op-offloads large-batch GEMMs to CUDA
regardless of `-ngl`, so `-ngl 0` reports `pp512 = 2900 tok/s` — physically impossible on six
P-cores, and a 20x error that invalidated every estimate made before it was caught.


## ★ BOTH BOARDS RE-TAKEN ON MERGED main — AND THE HARNESS HAD A FAULT THAT ONLY SHOWS OFF THE LAPTOP.

Four branches had just merged (the Q5_K MMA correctness fix, the pre-VNNI kernels,
the Metal instrumentation, llava/CLIP), so every row on record was taken on a
binary that no longer exists. RULE 1 forbids carrying those across. Both rows
below are merged main against the same llama.cpp build each host already has.

**Llama-3.2-1B-Instruct Q4_K_M, CPU decode, Xeon D-2123IT, n=128, four physical
cores, ABBA, 10 rounds, `JITLLM_SOAK=10`, both arms pinned and both given four
threads:**

| | ratio | IQR/median | n | |
|---|---|---|---|---|
| pass 1 | **0.9637** | 0.004 | 20 | loses |
| pass 2 | **0.9647** | 0.004 | 20 | loses |

Two passes agreeing to **0.10 points**, both gating, no clock trend (r=+0.13 and
+0.45 across a 14 MHz spread), cores 0.0% busy at 77 C throughout.

    jitllm     30.7 tok/s = 25.1 GB/s = 69% of this host's 36.63 GB/s wall
    llama.cpp  31.9 tok/s = 25.5 GB/s = 70%

Against the 0.9655 / 0.9690 already on record that is a reproduction to within
0.5 points, so the merges did not move it.

**Llama-3.2-1B-Instruct Q4_K_M, METAL decode, M4 mini, n=320, ABBA, 6 rounds,
`JITLLM_SOAK=10`, llama.cpp b10825 at -ngl 99:**

| | ratio | IQR/median | n | |
|---|---|---|---|---|
| pass 1 | **0.8771** | 0.010 | 12 | loses |
| pass 2 | **0.8769** | 0.018 | 12 | loses |

Agreeing to **0.02 points**. Against 0.8824 / 0.8790 on record, a reproduction.

    jitllm     101.4 tok/s x 0.816 GB = 82.7 GB/s = **76%** of the 108.9 GB/s wall
    llama.cpp  115.7 tok/s x 0.808 GB = 93.5 GB/s = **86%**

★ **AND THAT TEN POINTS OF WALL EFFICIENCY IS THE WHOLE METAL DEFICIT, MEASURED
ON THE MERGED BINARY.** It is not per-format: at equal bytes all six packed formats
read 95-99 GB/s on this box and Q4_0 -- a 4-bit format -- is the slowest. The
"28% per-byte gap on Q8_0/Q5_K/Q6_K at 71-75 GB/s" came from a single
same-shape row that read 95 and then 71 on consecutive identical runs.

★★ **AND THE HARNESS WAS GIVING llama.cpp SIX THREADS ON FOUR CORES, WHICH NO
STATISTICAL GATE CAN SEE.** `LCPP_THREADS` was hardcoded to 6 -- this laptop's
P-core count -- while jitllm sizes its pool from the taskset it is handed. Same
box, same binary, same model, same session:

    harness, -t 6 confined to 4 cores    llama.cpp 24.7 tok/s    ratio 1.2413 "WINS"
    standalone, -t 4 on 4 cores          llama.cpp 32.08         ratio 0.9637 loses

**Both gate at IQR/median 0.004 over twenty pairs and one is nonsense.** The
arms are tight, reproducible and wrong, so dispersion, a second pass and an A/A
control are all blind to it, and the error biases the ratio UP. What caught it
was llama.cpp's ABSOLUTE rate disagreeing with the 31.99 this project already
records for that host. The laptop's default coreset has six entries, so the
constant happened to be right there and every laptop row on record is
unaffected; only off-laptop rows were wrong. The thread count follows `$CORES`
now.

## ★★ olmoe CANNOT PRODUCE A GATED ROW ON THE XEON, DETERMINISTICALLY — AND THE DENSE MODEL BESIDE IT DISCARDS NOTHING.

olmoe-1b-7b-0924-instruct Q4_K_M, same box, same session, same harness, same
binary as the Llama-3.2-1B row above. Two passes, 10 rounds each, both REFUSED
by the coherence gate with the identical message:

    every pair ran its two arms at different clocks (> 1.05x):
    this box cannot produce a row right now

**And the arms' RATES are rock-steady in both passes** -- jitllm 25.18-25.87,
llama.cpp 31.62-31.92, under 3% spread each -- so this is not a noisy box
refusing a noisy row. It is the two ENGINES running the same model at different
CLOCKS:

| | jitllm's cores | llama.cpp's cores | pairs discarded |
|---|---|---|---|
| **Llama-3.2-1B (dense)** | 2600-2627 | 2600-2627 | **0 of 20** |
| **olmoe (MoE), pass 1** | 2600-2625 | **2050-2451** | **20 of 20** |
| **olmoe (MoE), pass 2** | 2600-2625 | **2069-2451** | **20 of 20** |

★ **THAT IS THE CONTROL THE OPEN QUESTION HAS BEEN MISSING, AND IT CONFIRMS THE
PREDICTION.** This file already records the coherence gate as "under suspicion
by its own data", names the candidate -- *"`ensureExperts` does synchronous
preads on the critical path where llama.cpp mmaps and faults at 4 KiB. That
would make 1.05 too tight for MoE models SPECIFICALLY rather than wrong in
general, and it predicts the dense model should discard far fewer"* -- and marks
it **unmeasured**. It is measured now, and the separation is total rather than
partial: zero discarded on the dense model against twenty of twenty on the
mixture, on one box, in one session, with one harness, an hour apart.

★★ **AND THE RECORDED olmoe ROW IS INFLATED BY THE THREAD-COUNT FAULT ABOVE.**
0.8743 / 0.8746 were taken when `LCPP_THREADS` was hardcoded to 6 on a four-core
box, which slows the BASELINE arm. With llama.cpp given its four threads the raw
median is **25.6 / 31.8 = 0.805**, both passes -- seven points below the
standing. That number is NOT a row: the gate refuses it, and it is recorded here
only so the 0.874 is not carried forward as though it survived a fixed harness.

    jitllm 25.6 tok/s x 0.757 GB/token = 19.4 GB/s = 53% of the 36.63 GB/s wall

which is the MoE shape this file already documents from the other side: a
mixture reaches ~59% of the read wall where a dense model reaches ~85%.

★ **WHAT IS STILL NOT ESTABLISHED.** Why llama.cpp's cores clock DOWN on a
mixture. Its rate is higher, not lower, so it is not stalling for want of work;
the mmap-and-fault-at-4-KiB hypothesis predicts exactly this (a thread waiting
on a page fault is descheduled and its core drops) and remains unconfirmed. The
gate stays on: a too-aggressive gate refuses rows, a too-permissive one
publishes them.

## ★★★ olmoe's DEFICIT IS ACCOUNTED FOR WITHOUT A RATIO, ON THE TRUSTED BOARD.

The coherence gate refuses every olmoe pair on the Xeon (see above), so no
jitllm-vs-llama.cpp ROW exists for it there. That is a reason and not an
accounting -- and the accounting does not need a ratio, because the confound is
a CLOCK DIFFERENCE BETWEEN TWO ENGINES and a one-arm measurement has only one
engine in it.

Same box, same binary, same session, real 64-token decodes:

| model | tok/s | bytes/token | GB/s | % of the 38.3 GB/s 4-thread wall |
|---|---|---|---|---|
| Llama-3.2-1B (dense) | 30.7 | 0.816 GB | 25.1 | **66%** |
| olmoe-1b-7b (MoE) | 25.07 | 0.752 GB | 18.85 | **49%** |

**olmoe reads 8% FEWER bytes per token and reaches 17 points less of the wall.**
That is the deficit, measured in the currency it lives in, with no second engine
and no gate involved.

★ **AND THE PROFILER SAYS THERE IS NO OTHER SUBSYSTEM TO BLAME**: on the same
run, `matvec 92.6% rmsnorm 0.6% rope 0.9% attention 3.4% residual 0.2% router
2.4%`. The router is visible and small. The missing efficiency is INSIDE the
matvec phase -- workers idle between expert matvecs -- which is what this file
already records from the laptop: the cost is the SERIAL STRETCH between regions
(the router softmax, the top-k, the ascending-id sort, `ensureExperts`'
synchronous preads), not the number of regions, which `MatVecPackedMulti` and
`MatVecPackedGather` already took from 906 to 234 a token with no rate change.

★ **SO THE 0.70 IS ACCOUNTED FOR IN THREE TERMS, EACH MEASURED:**

    it was NOT a regression        the counters already in this file: HEAD does
                                   7.1% FEWER one-thread instructions and 3.87x
                                   fewer regions than container v6 for
                                   BYTE-IDENTICAL token ids. A regression
                                   executes more work, not less.

    olmoe genuinely loses          49% of the read wall against a dense model's
                                   66%, on the trusted board, one arm, same session.

    the SPECIFIC 0.70 was the      it came from the laptop, whose harness is
    thermometer                    demonstrated to have ~10 points of
                                   irresolution -- a DOCS-ONLY commit read
                                   0.8866 against its byte-identical
                                   predecessor's 0.9853.

Nothing is left over. The magnitude of the loss is a standing property of the
mixture path on this engine; the READING of 0.70 is not reproducible on any box
under any condition tested, and the closest gated instrument refuses to publish
one at all because the two engines drive the clock differently on a mixture.

## ★★★★ olmoe IS AT PARITY ON THE XEON, GATED, AND THE COHERENCE REFUSAL WAS llama-bench's MODEL LOAD.

olmoe-1b-7b-0924-instruct Q4_K_M, CPU decode, four physical cores, ABBA, 6
rounds, `JITLLM_SOAK=10`, both arms pinned and given four threads, **n=512**:

| | ratio | IQR/median | n | clocks |
|---|---|---|---|---|
| pass 1 | **0.9918** | 0.006 | 12 | 2600-2608 BOTH arms |
| pass 2 | **0.9945** | 0.003 | 12 | 2600-2604 BOTH arms |

Two passes agreeing to **0.27 points**, both gating, **nothing discarded**, no
clock trend. This is the row the coherence gate refused twice at n=128.

★★ **THE REFUSAL WAS THE MODEL LOAD SITTING INSIDE THE MHz SAMPLING WINDOW, AND
IT SCALES WITH FILE SIZE -- SO THE GATE WAS BIASED AGAINST BIG MODELS.** At
n=128 the two arms read 2600 against 2050-2451 and every pair was thrown away.
`llama_bench` is invoked with `-r 1`, so each sample is ONE model load plus the
decode; olmoe's GGUF is 4.0 GB against the dense model's 0.77, and at 31.8 tok/s
128 tokens is ~4 s. A multi-second mmap-and-fault phase with the cores mostly
idle therefore drags llama.cpp's SAMPLED clock down by roughly its share of the
window -- ~30% for olmoe, ~5% for the dense model, which is exactly the pattern:
the dense model discarded 0 of 20 and olmoe discarded 20 of 20. At n=512 the
decode is four times longer, the load is amortised, and the arms agree to 8 MHz.

This file already recorded the same trap once, in the entry that fixed the
sampler: *"it sampled two seconds in, which for llama-bench is during MODEL LOAD
with the cores idle. It measures when I looked, not how they run."* The sampler
was fixed to read only the taskset's cores; it was never made to exclude the
load, and nothing connected that to a gate refusing MoE rows.

★★ **AND THE SPIN HYPOTHESIS IS REFUTED, MEASURED.** This file's standing
candidate is that jitllm's 250 us pool spin holds its clocks up where
llama.cpp's threads park. Run at `JITLLM_SPIN_US=0`, so jitllm's workers park
immediately: the arms still read 2600-2616 against 2150-2450 and the gate still
refused every pair -- while jitllm's own rate fell 25.6 -> 22.7 tok/s, an 11%
cost for nothing. **The spin is not what separated the clocks.**

★★★ **AND THE WHOLE olmoe STORY CLOSES, EVERY TERM MEASURED:**

    0.70    the laptop's harness, which reads 0.8866 for a DOCS-ONLY commit
            against its byte-identical predecessor's 0.9853
    0.874   the hardcoded LCPP_THREADS=6 giving llama.cpp six threads on four
            cores (see the entry above)
    0.805   REAL, but for n=128 only, and ungated: the load-phase confound
    0.992   n=512, gated twice, both arms coherent -- **PARITY**

★ **THE n-DEPENDENCE IS THE FINDING, NOT AN ARTEFACT, AND THIS FILE PREDICTED
IT.** Between n=128 and n=512 jitllm goes 25.6 -> 28.9 tok/s (+13%) while
llama.cpp goes 31.8 -> 29.2 (-8%). The MoE entry already said why: *"the harness
warms 8 tokens and measures 64, and olmoe has 64 experts over 16 layers -- 1024
expert slots -- so it is still faulting new experts throughout. A long decode
would drive that term toward zero and the 59%-of-wall figure up. The row is a
LEAD, not a standing."* It is measured now: olmoe's deficit is a SHORT-RUN
effect -- expert paging on the critical path -- and it is gone by 512 tokens.
RULE 1's "say the generation length" is worth nineteen points on this model.

## ★★★★ THE 28% PER-BYTE GAP IS MEASURED FALSE ON MERGED main, TWO INDEPENDENT WAYS.

A standing goal reads: *"Close the 28% per-byte gap on the 8-bit and two-plane
payloads (Q8_0, Q5_K, Q6_K at 71-75 GB/s vs 99-100 for the 4-bit ones) -- that
gap is the entire remaining deficit against llama.cpp."* Both halves are false,
and neither refutation is carried from a previous session.

**1. THE PER-FORMAT GAP.** `mvbench -eqbytes 384` on the M4, so every format
reads the same 384 MiB and cache pressure is matched:

| | Q4_0 | Q8_0 | Q4_K | Q6_K | Q3_K | Q5_K |
|---|---|---|---|---|---|---|
| GB/s | 96 | **97** | 98 | **97** | 98 | **96** |

**All six inside 2 GB/s of one another, and Q4_0 -- a 4-bit format -- is joint
slowest.** The three the goal calls slow read 97, 97 and 96. There is no 28%
separation and no two camps.

The 71-75 figures came from the SAME-SHAPE rows, which are a max over six split
values and drift with noise: `tll lm_head Q6_K` read 95 then 71 on consecutive
identical runs. Those rows are also printed at 102-126% of wall, which is the
probe measuring itself (see the streamWall entry) -- quote `-eqbytes`.

**2. "THAT GAP IS THE ENTIRE REMAINING DEFICIT" IS REFUTED BY THE BOARD ITSELF,
WITHOUT ANY MICROBENCHMARK.** The Metal row taken in the same pass is
**Llama-3.2-1B-Instruct Q4_K_M** -- built from Q4_K and Q6_K, and by the goal's
own premise Q4_K is one of the FAST formats. It loses:

    pass 1  0.8771   IQR/median 0.010   n=12
    pass 2  0.8769   IQR/median 0.018   n=12

    jitllm     101.4 tok/s x 0.816 GB = 82.7 GB/s = 76% of the 108.9 GB/s wall
    llama.cpp  115.7 tok/s x 0.808 GB = 93.5 GB/s = 86%

If the deficit were a per-byte penalty on the 8-bit and two-plane payloads, a
model made mostly of Q4_K would sit at parity. It is ten points of wall
efficiency short instead, on formats the premise says are fine.

★ **SO THE METAL DEFICIT IS TEN POINTS OF WALL EFFICIENCY AND IT IS NOT PER
FORMAT.** What that leaves as the lever is already recorded and unbuilt: every
tunable is at its optimum (SPLIT, ROWT, TOK, LANES, MVACC, QTILE all land
between 98.6 and 100.1), so it is a kernel DESIGN point rather than a setting --
MLX's `qmv_fast` uses two simdgroups of four results, one row per simdgroup
reduced through the free shuffle, and **no k-split or partial-sum reduction at
all**, where jitllm k-splits and reduces partials. That is the structural
difference to price, and it is the only Metal item left with evidence behind it.

## ★★ THE IN-GROUP REDUCTION IS ~0.5% AND, MORE USEFULLY, DETERMINISTIC. RE-MEASURED ON MERGED main.

`ForceGroupSplit` makes every eligible decode matvec reduce INSIDE the
threadgroup -- no partial plane, no reduction kernel -- which is the first
structural step of MLX's `qmv_fast` shape. M4, Llama-3.2-1B, GPU CLOCK rather
than wall (the tight metric), ABBA, A/A self-control run FIRST:

    A/A control   1.0298*  0.9970  1.0011  1.0023     median 1.0011
                  (*cold start, the first sample of the session)
    ABBA          1.0019  0.9958  0.9960  1.0029  0.9875  0.9904
                  median 0.9959, IQR/median 0.0115

**~0.4% of GPU time, ~0.5% corrected for the A/A bias**, reproducing the
0.9908/0.9954 measured before the merge. Two of six rounds go the wrong way and
the effect is inside the IQR, so this is a small number quoted as a small number
-- under RULE 6 its size is no reason to leave it out.

★★ **THE DETERMINISM IS THE PART WORTH HAVING, AND IT ANSWERS AN OPEN QUESTION
IN THIS FILE.** Dispatches per token, five runs each:

    tuner       477  445  445  445  469     <- varies run to run
    forcegroup  404  405  405  405  405     <- deterministic

The v14 entry above records the Metal passes agreeing to 0.85 points where they
had agreed to 0.21, blames the tuner, and says: *"that 0.85 points is the
tuner's own variance -- so pinning the best split set may be worth more than
tuning it. Unmeasured."* It is measured now: the in-group reduction removes the
run-to-run dispatch variance entirely. At the 1.09 us this file prices a Metal
dispatch at, 40-72 fewer of them is 44-78 us of a 9.0 ms token, which is where
the 0.5% comes from and closes the account on it.

★ **IT IS NOT MADE THE DEFAULT HERE.** The knob stays off: that is a shipping
decision rather than a measurement, and it was put to the user alongside three
other options and not chosen. What is recorded is the number, so the decision
can be made on one.

## ★★★ AND THAT PRICES THE LAST NAMED METAL LEVER, WHICH TURNS OUT NOT TO BE THE ANSWER.

This file names one structural difference against MLX and calls it the thing to
price: *"MLX's `qmv_fast` ... `num_simdgroups = 2`, `results_per_simdgroup = 4`
-- eight output rows per threadgroup, one row per simdgroup reduced through the
free shuffle, and NO k-split or partial-sum reduction at all. jitllm k-splits
and reduces partials; that is the structural difference to price."*

**`ForceGroupSplit` IS that reduction strategy, and it is already built.**
`tier.go:220` selects "the in-group reduction for every eligible shape", and
`layer.go`'s own comment says what that means: *"the split reduces INSIDE the
threadgroup, so this matvec writes its final row directly: no partial plane and
no red kernel."* No partial plane and no reduction kernel is exactly qmv_fast's
arrangement.

★ **AND THE THREAD COUNTS ALREADY MATCH, WHICH IS THE HALF THAT LOOKED LIKE A
DIFFERENCE AND IS NOT.** A 2048-row matvec at `Split=8` is 16,384 threads, eight
per output row. MLX's eight rows per 64-thread threadgroup is also eight threads
per row. The two kernels spend their parallelism the same way; they differed
only in where the per-row reduction happens.

**So the lever is priced at ~0.5%, not at ten points, and it is measured rather
than estimated** (see the entry above: GPU clock, ABBA, A/A control first).
Read as a ten-point opportunity, the qmv_fast note overstated it: the piece of
it jitllm was missing measured half a percent.

★ **WHAT THAT LEAVES.** The Metal deficit is ten points of wall efficiency, it
is not per format (all six read 96-98 GB/s at equal bytes), it is not the
reduction strategy (0.5%), it is not submission (3% ceiling, measured by the
tool built for it), it is not the head (2.106-2.116 ms = 79% of wall, which is
in proportion), and it is not attributable per dispatch on this hardware (four
routes, all closed). The full accounting closes at 93.7% with **no single term
over 5%** -- which is what a diffuse deficit looks like, and it is the honest
description of the remaining gap rather than a lever waiting to be pulled.

## ★★★ THE DeepSeek WORK DID NOT CHANGE WHAT AN EXISTING TOKEN COSTS, AND THE STOPWATCH COULD NOT SAY SO.

The MLA/router work touched three things every model runs: the attention
loop's scale became `Config.AttnScale` (precomputed, including YaRN's squared
correction), the loop gained a query/output width pair, and EVERY mixture's
router moved from `softmax` + `MoETopK32JIT` to the single `MoERouteJIT` entry
point. RULE 6 asks whether that regressed; this is the answer.

**★ THE TIMED A/B WAS REFUSED BY ITS OWN SELF-CONTROL, WHICH IS THE POINT OF
RUNNING ONE FIRST.** `model.TestFusionCeiling`, one test binary per tree, ABBA,
five rounds, each arm reading its OWN container so neither reconverts the
other's:

    A/A self-control (the NEW binary in BOTH arms)
      per-round ratios  0.8384  0.9003  0.9387  0.9530  1.0033
      median 0.9387   IQR/median 0.056

A harness whose same-binary control reads **0.9387** is biased by 6.1%, which
is larger than any regression worth hunting. The box said why: **88 C and every
measured core at 400 MHz**, against a 1700-2100 MHz sustained band and a ~3792
idle peak. One sample read 15.0 tok/s against the same binary's 61.0. RULE 2d's
verdict is to stop and rest the box, not to raise the round count -- and this
file already records that Linux rows belong on the Xeon for exactly this
reason. **No rate is claimed.**

**★ SO THE ANSWER COMES FROM THE COUNTERS, WHICH ARE INVARIANT TO THE
THERMOMETER.** This is the technique the v6-vs-HEAD investigation ended on:
regions per token is deterministic, and at `JITLLM_CORES=1` there are no
workers and no pool spin, so the instruction count IS the work.

| model | regions/token | one-thread instructions, n=320 | between arms |
|---|---|---|---|
| Llama-3.2-1B Q4_K_M (dense) | **218.0 / 218.0** | base 41.94/40.98/41.66 e9, new 41.08/41.53/41.43 e9 | **0.42%** |
| olmoe-1b-7b Q4_K_M (mixture) | **234.0 / 234.0** | base 82.02/81.17/81.87 e9, new 81.56/80.95/81.13 e9 | **0.58%** |

**Both gaps are SMALLER than the baseline arm's own within-arm spread** (2.31%
and 1.04%), and both region counts are identical -- and identical to the 218.0
and 234.0 this file already records for these two models. So the coordination
did not move and the work per token did not move: there is no regression to
attribute, by arithmetic rather than by a row.

★ **WHAT THIS DOES NOT ESTABLISH.** It says the existing models cost the same;
it says nothing about what DeepSeek itself costs, because there is no baseline
-- the architecture did not run before. A ratio against llama.cpp needs a
rested box and a file both engines read, and the second half of that is only
true since `convert.unfuseMLA` landed: until then every published deepseek2
GGUF was refused here for carrying the legacy fused `attn_kv_b`. That row is
takeable now and has not been taken.

## Boards that stood in AGENTS.md RULE 1

**★ THE BOARD BELOW IS THE PRE-CONTAINER ENGINE AND ITS ROWS ARE NO LONGER
REACHABLE.** They were taken on the GGUF inference path, which `model.Open`
refuses now — the engine reads one weight format and GGUF is the converter's
input. Kept because the numbers are what the CPU tier reached and therefore
what the container path has to get back to, NOT as a current scoreboard.

**★ Qwen3-30B-A3B-Q4_K_M, CPU decode, n=160 -- THE FIRST 30B ROW,
AND IT IS PROVISIONAL.** jitllm 1.0290, a WIN, **REJECTED at IQR/median 0.135**:

    jitllm     16.65 15.12 17.31 18.47 17.23 18.73 17.82 18.21 18.38 17.84 17.68 17.70
    llama.cpp  16.18 17.65 13.62 16.39  8.03 16.52 10.21 16.48 17.05 18.06  9.02 17.52

jitllm's arm is tight at 17-18 throughout; llama.cpp's is what blew the gate,
with three plainly power-clamped samples. RULE 2's verdict on that is to stop
measuring and let the box rest, NOT to raise the round count, so this row needs
a rested re-take before it is quoted as a standing.

★ **AND THE CONFIGURATION IS THE FINDING: BEING ONE PAGE SHORT OF FITTING IS A
CLIFF, NOT A GRADIENT.** The same model under `JITLLM_CAP_MEM=26G` measured
**0.5777** -- 10 tok/s against 17 -- because the budget came to 18.71 GiB
against 18.41 GiB of pages PLUS a 0.42 GiB dense region. Forty-seven of
forty-eight blocks resident is one eviction and one 392.7 MiB read EVERY TOKEN.
At 28G the whole model is resident, nothing is read, and the same binary runs
at 18. A budget heuristic that lands one page short is far worse than one that
fits, and the gap between those two rows is entirely residency.

**Container rows, CPU decode, `scripts/vs-llamacpp.sh`, n=320, six P-cores,
re-taken on the NATIVE format (container v6).** The box clamps to
1568–1880 MHz under six loaded P-cores -- it idles at 3792 of a 4900 max, so
this is the SUSTAINED-POWER clamp and not contention (PSI 0.00 throughout).
These are ratios and not absolute rates:

| model | ratio | IQR/median | n | | was (previous take) |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct Q4_K_M | **1.0037** | 0.005 | 12 | WIN | 1.0013, IQR 0.080 |
| tinyllama-1.1B q3_K_M | 0.9585 | 0.034 | 12 | loses | 0.9997, IQR 0.017 |

**★ AND THE tinyllama DROP IS NOT THE FORMAT WORK, WHICH A SAME-PASS A/B SAYS
AND A CROSS-SESSION COMPARISON CANNOT.** RULE 1's first line is that a baseline
is never carried across sessions, and this is why: the two rows moved in
OPPOSITE directions under the same change, which no property of the change
explains. Run against the pre-wiring binary, ABBA, one pass, same container:

    tinyllama q3_K_M, native format / pre-wiring    1.0091   IQR 0.028   n=6

So the format is neutral to slightly POSITIVE, and the 0.9585 is the box: the
clamp was deeper for this row (1568–1880) than when the earlier row was taken
(1800–2100), and the two engines do not shed rate to a clamp equally -- the M4
prefill row has jitllm losing 7.4% where llama.cpp lost 4.2%. **The tinyllama
row needs re-taking near 3792 MHz before it is quoted as a standing.**

The progression on the first row, one lever at a time: 0.4903 (first working
container) → 0.5860 (row loop inside the kernel) → 0.7300 (software prefetch at
the known stride) → 0.7713 (64-row tile + tail kernel) → 0.8771 (wide kernel,
accumulators in memory) → **1.0040** (fused kernel, accumulators in registers).

**★ THE GPU ROW IS A WIN AND THE HYBRID ROW IS NOT, AND THE HYBRID ONE IS THE
OPEN PROBLEM.** Same model, same box, `scripts/vs-llamacpp.sh gpu`:

| configuration | ratio | IQR/median | n | |
|---|---|---|---|---|
| every block on the card | **1.0136** | 0.031 | 6 | WIN |
| split, matched at 647 MiB of card | 0.8874 | 0.038 | 6 | LOSES |

The split row was 0.6848 until `placeHead` stopped releasing blocks it could
not use (it freed five blocks of card to fit a projection that needed six, then
gave up, keeping neither). What is left is ~10% and it is NOT the crossing:
llama.cpp's split time matches a serial model of its two halves almost exactly
(11.48 ms predicted, 87.5 measured), and jitllm's does not (11.65 predicted,
78 measured). Both engines carry 647 MiB of card; jitllm reads FEWER host bytes
(280 MiB against 302) and more device bytes, and is still slower. **Unresolved,
and it is a per-token overhead in the split configuration rather than a
placement or a bandwidth problem.**

That is placement.md 15g's crossing, priced: jitllm reads 14.1 ms/token against
llama.cpp's ~10.5 on the same split. The parts do NOT explain it -- the device
phase moves 340 MiB in 3.25 ms (102 GiB/s, matching the full-offload row) and
the host phase moves 469 MiB in 10.9 ms (42 GiB/s, matching the CPU-only row).
Each half runs at full speed and the sum is still 34% slow, so the cost is in
the seam rather than in either tier. **Unresolved.** A pinned re-take was
REJECTED at IQR/median 0.189 on a clamped box (833-905 MHz) and needs taking
rested; `JITLLM_PIN=1` pins both arms for it, which a hybrid row wants and a
full-offload row does not.

**★ AND THE LAST TWO STEPS CAME FROM A TOP-DOWN COUNTER, NOT FROM A GUESS.**
Three changes before it measured NEUTRAL, and each was aimed at the wrong
quantity. `perf stat` on the wide kernel:

    retiring 21.9%   backend-bound 73.3%   of which memory 49.1% of all slots

With only 22% of issue slots retiring, instruction count was never the
constraint -- and in the wide kernel's payload loop two of every three memory
operations are the ACCUMULATOR spilling between per-word passes, not the
weights. Running every word of a sub-block in one pass keeps it in a register;
that is worth 7-8% and it is what crossed parity.

`JITLLM_PROFILE=1` puts **97.6% of the token in the matvec** at n=320, so there
is no other subsystem to win in, and the ratio is flat across n=32/128/320 —
this is not a depth effect.

**The board lives in `docs/perf/current.md`.** CPU decode, n=320, six P-cores,
**re-taken rested at b31240e, one binary for every row:**

| model | ratio | IQR | was | |
|---|---|---|---|---|
| Qwen2-1.5B-Instruct Q4_K_M | **1.0887** | 0.038 | 1.0169 | WIN |
| Qwen3-1.7B Q4_K_M | **1.0554** | 0.019 | 1.0076 | WIN |
| tinyllama-1.1B q3_K_M | 0.9923 | 0.016 | 0.9471 | |
| OLMoE-1B-7B Q4_K_M | 0.9578 | 0.025 | 0.9090 | |
| gemma-2b | 0.9294 | 0.006 | 0.8856 | 2 samples clamped, n=10 |

**Every row moved up, by 4.7% to 7.1%, mean +5.4%** -- the elementwise generation
(RULE 8, measured 2.6-4.5% on the M4) plus everything after it. tinyllama is
0.8% from parity where it was 5.3% short. The `was` column is the previous
board, taken at 2191 MHz under a ~3500 MHz rested peak; this one was taken with
PSI avg10 under 1.3% throughout.

**★ THE M4 BOARD, RE-TAKEN ON THE CONTAINER**, `scripts/vs-llamacpp.sh`,
n=320, ABBA, six rounds each. These rows REPLACE the older ones below them,
which were the pre-container engine:

| model | tier | ratio | IQR | jitllm | llama.cpp | container vs GGUF |
|---|---|---|---|---|---|---|
| tinyllama q3_K_M | Metal | **1.0123 / 1.0240** WIN | 0.012-0.015 | 115-116 | 113.6 | **+43%** |
| gemma-2b | Metal | **0.9470 / 0.9438** | 0.004-0.011 | 53.6 | 56.8 | **-0.1%** |
| Llama-3.2-1B Q4_K_M | Metal | **0.8824 / 0.8790** | 0.008-0.011 | 98.7 | 112.0 | +6% |
| gemma-2b | CPU | **0.8463** | 0.007 | | | -0.1% |
| tinyllama q3_K_M | CPU | **0.6068** | 0.002 | | | +43% |

★★ **AND THOSE TWO ROWS ARE THE FIRST HONEST M4 CPU NUMBERS, BECAUSE arm64
DECODE WAS WRONG BEFORE THEM.** Every quantized model decoded to garbage on
that host -- prefill correct, first generated token correct, then a repeat and a
collapse to token 0:

    tinyllama  [1 450 7483 310 3444 338 3681 3681 3681 0 0 0 ...]
    gemma-2b   [2 714 6037 576 6081 603 7127 7127 0 0 0 ...]

**`nn.MatVecPacked`'s TILED branch never cleared `out`.** The kernels accumulate
and the caller owns the zero; the fused and wide branches clear and say so. The
amd64 tiled kernel zeroes Out ITSELF at each tile head and can afford to -- its
loops are tile-OUTER, super-block-INNER. `EmitA64PackedMatVec` nests them the
other way round, because the scale planes advance per super-block, so zeroing
there would wipe the previous super-block's sums; it load-accumulates and
REQUIRES a zeroed Out, which nothing gave it.

★ **IN A MODEL THAT BUFFER IS THE LOGITS**, so token 1 was right (Go zeroes fresh
memory), token 2 was logits+logits, and the argmax froze.
`model.TestBatchMatchesForward` read the same token 6342 for every sequence at
every step on the M4 -- a frozen Forward -- and that is exactly what it was.

★ **THE RATE DID NOT MOVE: 0.8454 BROKEN, 0.8463 FIXED.** A collapsed model does
the same arithmetic at the same speed, which is why no perf row ever flagged it
and why the 0.8454 row was wrong about MEANING rather than about speed.
That is the general lesson: a rate is not evidence that the engine computed the
model.

★ **AND EVERY KERNEL GATE WAS GREEN THROUGHOUT.** They all allocate a fresh
output and call ONCE -- `make` zeroes it, so the missing clear was exactly
compensated. The bug needs a SECOND call into the SAME buffer.
`nn.TestPackedMatVecTailAgrees` calls twice and asserts identity now; against
the violation it reads "row 0 went 3.632285e+07 -> 7.2648696e+07", exactly
doubling. It also gained varying scales (they were all 0x3C00, so a wrong row's
scale changed nothing), subnormal ones, and the (rows, k) PAIRS real models use
-- it had been six row counts at one k, and no model uses k=512.

★ **tinyllama's 0.6068 IS NOW THE WEAKEST ROW ON EITHER HOST**, and its +43%
container tax is most of it: the same model WINS on Metal at 1.0123 while
carrying those bytes, so the arm64 CPU kernels are what that row is about.

**The older M4 board, kept because the rows below were taken on the
PRE-CONTAINER engine and are not reachable now:**

| model | tier | ratio | IQR | was |
|---|---|---|---|---|
| tinyllama q3_K_M | Metal | 1.0200 | 0.005 | 1.021 |
| gemma-2b | Metal | 0.9509 | 0.013 | 0.9440 |
| tinyllama q3_K_M | CPU | 0.9378 | 0.006 | 0.8276 |
| gemma-2b | CPU | 0.9279 | 0.008 | 0.853 |
| olmoe-1b-7b | CPU | 0.8456 | 0.073 | 0.7280 |
| Qwen3-MOE-4x0.6B | CPU | 0.8392 | 0.013 | -- |

The Mac's MoE rows are now its weakest, and Linux wins the same Qwen3-MoE at
1.0289 — a 19-point cross-host swing on one model that nothing explains yet.

**PREFILL on the M4, measured with that older board and previously absent from the board:**
tinyllama **0.7174, and the six-commit stack is VERIFIED END TO END** -- the
commit before the chain and HEAD, ABBA'd in one pass, give 0.4576 -> 0.7174
(relative 1.5678, IQR 0.002) against 1.5725 compounded from the individual
claims, and jitllm's own prefill rate goes 220.5 -> 347.9 tok/s. Two of the six
levers are under 3% alone and the three smallest are worth 4.5 of the 57
points (was 0.4575), gemma-2b 0.5663 (was 0.4852) — the amax hoist, a
GENERATED activation packer (155 ms -> 12), and hoisting the unpack masks the
GEMM was rebuilding per sub-block. **But the row is NOT a kernel comparison:**
`llama-bench` reports `MTL,BLAS` even under `-dev none`, and sampling shows
llama.cpp runs repacked NEON for Q4_K/Q5_K and **dequantizes Q3_K — 63% of the
MACs — to f32 for Apple's Accelerate**, whose own thread pool ignores `-t 4`. The same
code WINS prefill on Linux at 1.32, and jitllm is faster in absolute terms on the
Mac (227 vs ~207 tok/s) — llama.cpp is 3x faster there. **It is not i8mm:
`otool` counts zero smmla and 1045 sdot in its arm64 build.**

★★★ **AND THE GOAL'S TWO NAMED ROWS, TAKEN ON THAT INSTRUMENT.
BOTH LEADS WERE THE THERMOMETER.** n=128, ABBA, 10 rounds, two passes each, core
busy-ness recorded at the START and the FINISH of every pass so a mid-pass
tenant cannot hide:

| model | pass 1 | pass 2 | cores |
|---|---|---|---|
| Llama-3.2-1B-Instruct Q4_K_M | **0.9655** (IQR 0.006, n=16) | **0.9690** (IQR 0.002, n=10) | 0.1% -> 0.0% both |
| olmoe-1b-7b-0924-instruct Q4_K_M | **0.8743** (IQR 0.004, n=9) | **0.8746** (IQR 0.009, n=7) | 0.1% -> 0.2%, 0.0% -> 0.0% |

Against the leads this goal was written on: **0.90 -> 0.966/0.969** and
**0.70 -> 0.871/0.869**. Neither lead survives. Llama-3.2-1B's passes agree to
0.35 points and BOTH gate; olmoe's agree to **0.14 points**, tighter than the
dense model, and its second pass is REJECTED anyway on n=5 -- the thin-row rule,
not dispersion.

★ **olmoe's ROW WAS RE-TAKEN AFTER THE CLOCK SAMPLER WAS FIXED, AND BOTH
PASSES GATE NOW.** The first take read 0.8706 (IQR 0.019, n=8) and 0.8694 (IQR
0.008, n=5 -- REJECTED on the thin-row rule), because the coherence gate was
comparing the arms on a PACKAGE average rather than on the cores the row pins
to. With the sampler reading only those cores: **0.8743 / 0.8746, IQR 0.004 and
0.009, n=9 and n=7, both gating and agreeing to 0.03 points** where they agreed
to 0.14 before. The median moved 0.4 points, which is the check that the old gate
was not deleting a real effect.

★ **WHAT THIS IS AND IS NOT.** It is jitllm against llama.cpp b11011 on the
Xeon, with the PRE-VNNI binary, at n=128. The 0.90/0.70 leads were a different
CPU, a different llama.cpp build and n=320, so this does not prove those
readings arithmetically wrong -- it establishes where the container path sits
when measured on hardware that can hold a clock. **The separate question "did
the ENGINE change between v6 and HEAD" is answered by counters, not by this
row**, and those say no: identical token ids, 218.0/218.0 regions per token,
one-thread instructions 0.22% apart, and olmoe doing 7.1% FEWER instructions and
3.87x fewer regions on HEAD. A regression executes more work, not less.

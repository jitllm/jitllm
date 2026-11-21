---
title: Benchmarks
description: jitllm against llama.cpp, vLLM, mistral.rs and ZML on V100s, a Xeon and an M4 (with MLX on the M4), with the method behind every number.
---

Every figure here names its host, its backend and the engine it is compared with, and every comparison was taken in the same pass on the same machine. Ratios are jitllm divided by the other engine, so above 1 means jitllm is faster. The full record, including every row below parity and what was measured about it, is [`.notes/perf/current.md`](https://github.com/jitllm/jitllm/blob/main/.notes/perf/current.md) in the repository.

## Method

- **jitllm:** `jitllm speed -p 512 -n 128 -r 2`, the better of the two runs (the second is warm, after the tuners have settled).
- **llama.cpp:** `llama-bench -p 512 -n 128 -r 3`, its mean (it discards its own warm-up), on the same GGUF that jitllm's `.jlm` was converted from.
- Prefill is a 512-token prompt and decode is 128 generated tokens, in tokens per second. Weights are Q4_K_M unless the row says otherwise.
- The M4 Metal section for 3B to 7B models states its own method: a paired, interleaved harness and a separate three-engine pass.
- One pass per host: each table is a single sweep, so a row within a few percent of parity is inside the box's pass-to-pass spread.

## Tesla V100, CUDA

2× Xeon E5-2680 v4 with 8× Tesla V100-SXM2-16GB (sm_70). jitllm `-devices cuda` against llama.cpp built for sm_70 at `-ngl 99`; models that need more than one card are split across the GPU count shown.

| Model | GPUs | Prefill | llama.cpp | Ratio | Decode | llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2-360M Q8_0 | 1 | 33613 | 24361 | **1.38×** | 492.2 | 405.5 | **1.21×** |
| Qwen3-0.6B Q8_0 | 1 | 23980 | 20217 | **1.19×** | 422.2 | 381.1 | **1.11×** |
| gemma-3-1b | 1 | 17324 | 14238 | **1.22×** | 331.8 | 273.2 | **1.21×** |
| Llama-3.2-1B | 1 | 21062 | 16081 | **1.31×** | 530.1 | 464.6 | **1.14×** |
| SmolLM2-1.7B | 1 | 13129 | 10147 | **1.29×** | 376.5 | 338.8 | **1.11×** |
| Qwen2.5-1.5B | 1 | 12504 | 11548 | **1.08×** | 359.2 | 285.6 | **1.26×** |
| gemma-2-2b | 1 | 9869 | 8177 | **1.21×** | 246.3 | 208.1 | **1.18×** |
| Qwen3-1.7B Q8_0 | 1 | 12468 | 11961 | **1.04×** | 266.8 | 250.9 | **1.06×** |
| Llama-3.2-3B | 1 | 8022 | 6279 | **1.28×** | 244.6 | 220.7 | **1.11×** |
| Phi-3.5-mini | 1 | 5643 | 5661 | 1.00× | 211.2 | 199.5 | **1.06×** |
| Phi-4-mini | 1 | 6793 | 6051 | **1.12×** | 201.1 | 198.2 | **1.01×** |
| gemma-3-4b | 1 | 6924 | 5735 | **1.21×** | 172.8 | 154.4 | **1.12×** |
| Qwen3-4B | 1 | 6670 | 5170 | **1.29×** | 190.8 | 168.5 | **1.13×** |
| OLMoE-1B-7B <small>MoE</small> | 1 | 10922 | 4152 | **2.63×** | 459.9 | 390.5 | **1.18×** |
| Mistral-7B v0.3 | 1 | 4112 | 3248 | **1.27×** | 144.6 | 133.3 | **1.08×** |
| Qwen2.5-7B | 1 | 4205 | 3400 | **1.24×** | 144.5 | 131.5 | **1.10×** |
| Llama-3.1-8B | 1 | 4102 | 3229 | **1.27×** | 136.6 | 125.8 | **1.09×** |
| Qwen3-8B | 1 | 3793 | 3080 | **1.23×** | 128.0 | 119.8 | **1.07×** |
| gemma-2-9b | 1 | 3058 | 2491 | **1.23×** | 98.3 | 92.3 | **1.06×** |
| gemma-3-12b | 1 | 2484 | 2010 | **1.24×** | 77.9 | 75.2 | **1.03×** |
| Qwen3-14B | 1 | 2315 | 1774 | **1.31×** | 78.7 | 72.4 | **1.09×** |
| phi-4 (14B) | 1 | 2289 | 1847 | **1.24×** | 78.8 | 74.5 | **1.06×** |
| gpt-oss-20b MXFP4 <small>MoE</small> | 1 | 4820 | 2714 | **1.78×** | 191.1 | 170.7 | **1.12×** |
| DeepSeek-V2-Lite <small>MoE · MLA</small> | 1 | 4561 | 2089 | **2.18×** | 183.8 | 171.4 | **1.07×** |
| DeepSeek-Coder-V2-Lite <small>MoE · MLA</small> | 1 | 4518 | 2026 | **2.23×** | 184.1 | 171.3 | **1.07×** |
| Moonlight-16B-A3B <small>MoE · MLA</small> | 1 | 4632 | 2121 | **2.18×** | 176.3 | 131.0 | **1.35×** |
| gemma-3-27b | 2 | 1020 | 902 | **1.13×** | 39.2 | 38.7 | **1.01×** |
| Qwen3-30B-A3B <small>MoE</small> | 2 | 3761 | 1076 | **3.50×** | 167.4 | 153.5 | **1.09×** |
| Qwen3-32B | 2 | 1051 | 796 | **1.32×** | 37.3 | 34.5 | **1.08×** |
| Mixtral-8x7B <small>MoE</small> | 3 | 1723 | 865 | **1.99×** | 84.7 | 81.2 | **1.04×** |
| Llama-3.3-70B | 4 | 457 | 371 | **1.23×** | 18.3 | 17.5 | **1.05×** |
| Qwen2.5-72B | 4 | 455 | 374 | **1.22×** | 16.8 | 16.2 | **1.04×** |
| Qwen3-Next-80B-A3B <small>hybrid MoE</small> | 4 | 2058 | 445 | **4.62×** | 105.4 | 93.9 | **1.12×** |
| Kimi-Linear-48B-A3B <small>hybrid · MLA</small> | 4 | 3046 | — | — | 111.9 | — | — |
| gpt-oss-120b MXFP4 <small>MoE</small> | 5 | 2257 | 1235 | **1.83×** | 130.2 | 117.5 | **1.11×** |

Prefill is ahead on 34 of the 35 rows llama.cpp runs and level on the 35th (Phi-3.5-mini, 5643 against 5661); decode is ahead on all 35. Mixtures gain the most: Qwen3-30B-A3B prefills 3.5× faster and Qwen3-Next-80B 4.6×. Kimi-Linear has no llama.cpp row because it exists only as Hugging Face safetensors, which jitllm converts directly. mistral.rs and ZML could not run on this card: mistral.rs ships nothing below sm_80, and ZML's CUDA path needs CUDA 13, which dropped Volta.

## Big models over several V100s

Llama-3.3-70B and gpt-oss-120b as they are usually served: jitllm's layer split over the cards named, against llama.cpp with `-ngl 99` and its own layer split, both engines given the same cards.

**Decode**, 256 generated tokens after an 8-token prompt, `scripts/vs-llamacpp.sh`, two passes:

| Model | Cards | jitllm | llama.cpp | Ratio |
|---|---:|---:|---:|---:|
| Llama-3.3-70B Q4_K_M | 4 | 16.93–17.16 | 17.00–17.03 | **0.999× / 0.999×** |

At parity in both passes, and both engines read 80% of one card's 890 GB/s: in a layer split the cards take a token in turn, so one card's bandwidth is the wall.

**Served, one request at a time**, a 514-token prompt and 256 tokens asked for, through `jitllmd serve` and `llama-server`, six warm requests to each engine over two server lifetimes:

| Model | Engine | Cards | First token | Prefill | Decode | VRAM |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.3-70B Q4_K_M | jitllm | 4 | **1351 ms** | 380 tok/s | 17.87 tok/s | 45.8 GiB |
| | llama.cpp | 4 | 1616 ms | 318 tok/s | 17.31 tok/s | 44.7 GiB |
| gpt-oss-120b MXFP4 | jitllm | 5 | **289 ms** | 1778 tok/s | 116.8 tok/s | 64.8 GiB |
| | llama.cpp | 5 | 552 ms | 929 tok/s | 109.7 tok/s | 61.3 GiB |

The 70B reaches its first token 1.20× sooner. This table is one pass, so it is indicative. gpt-oss-120b's first-token times spread more than the gate allows, from one jitllm server lifetime to the next (262–360 ms), and it stopped its answers after 129–159 tokens where llama.cpp ran to 256, so its decode rates are over different lengths.

## Time to first token, V100

A request is text in and one token out, so the time includes tokenizing it. The same prose repeated to about 48, 531 and 2049 tokens, the same Q4_K_M weights, warm, median of 7, in milliseconds. jitllm `speed -ttft -r 7` (a fresh session on a loaded model); llama.cpp `llama-server --no-cache-prompt`, timed at the client; vLLM 0.18.1 `generate(max_tokens=1)` with prefix caching off.

| Model | Cards | Prompt | jitllm | llama.cpp | vLLM |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 1 | 48 | 12.6 | **11.1** | 18.9 |
| | | 531 | 40.3 | **39.2** | 150.8 |
| | | 2049 | **118.7** | 127.8 | 646.1 |
| Llama-3.1-8B | 1 | 48 | 40.3 | **37.4** | 97.9 |
| | | 531 | **166.6** | 182.8 | 1032.2 |
| | | 2049 | **556.8** | 652.8 | 4292.7 |
| gemma-3-27b | 2 | 2049 | **1338** | 1969 | refuses float16 |
| Qwen3-32B | 2 | 2048 | **1332** | 1729 | 9480 |
| Qwen3-30B-A3B <small>MoE</small> | 2 | 47 | **67** | 74 | 93 |
| | | 2048 | **590** | 1677 | 983 |
| Mixtral-8x7B <small>MoE</small> | 3 | 2056 | **1314** | 2555 | fails to load |
| Llama-3.3-70B | 4 | 2049 | **2222** | 3015 | 10474 † |
| Qwen2.5-72B | 4 | 2048 | **2160** | 3080 | 11321 † |
| Qwen3-Next-80B <small>hybrid MoE</small> | 4 | 2048 | **1104** | 4506 | no GGUF support |
| gpt-oss-120b <small>MXFP4 MoE</small> | 5 | 2048 | **1080** | 1876 | refuses MXFP4 below sm_80 |

A long prompt over several cards is pipelined: each card takes 512-row pieces as the card before it finishes them, so the cards work at once rather than in turn. That is what takes the 70B's 2049-token prompt from 4.7 s to 2.2 s. jitllm is behind on short prompts, 3–14% on one card, a fixed per-request cost not yet attributed. † vLLM ran the 70B and 72B with `enforce_eager` and 0.92 of the memory, because its defaults ran out of device memory.

Cold, a one-token process from a GGUF that is not in the page cache: jitllm 14.4 s against llama.cpp's 17.0 s on Qwen3-30B-A3B over two cards. jitllm reads the container with direct I/O every time, so the OS cache does not help it; with the GGUF already cached, llama.cpp starts in 10.2 s.

## Sessions decoding together, one V100

Separate sessions (separate users, separate histories) decoded as rows of one step, so each block's weights are read once for all of them. Llama-3.1-8B Q4_K_M, generated tokens per second, against `llama-batched-bench` on the same card.

| Sessions | jitllm, together | llama.cpp | Ratio |
|---:|---:|---:|---:|
| 2 | 198 | 211 | 0.94× |
| 4 | 308 | 289 | **1.06×** |
| 8 | 483 | 324 | **1.49×** |

Up to four sessions run decode's own matrix product with every session's token in one thread; eight use a tiled kernel that wins there. Two sessions were 0.67× before that change, and one session at a time decodes about 105 tokens a second, so eight together are 4.6× that. The two- and four-session rows are `speed -sessions N -p 16 -n 64` against `llama-batched-bench -npp 16 -ntg 64`, a second tool, so their ratios are indicative; the eight-session row is `-p 32` against `-npp 32`. The server batches requests this way; see [Serve an API](/docs/guides/serve/#batching).

## Batched decode, one V100

Aggregate generated tokens per second: a 6-token prompt, then 128 tokens per sequence. jitllm `batch -devices cuda:0`; llama.cpp `llama-batched-bench -npp 6 -ntg 128 -ngl 99`; vLLM 0.18.1, generated tokens over its whole wall clock.

| Model | Batch | jitllm | llama.cpp | vs llama.cpp | vLLM | vs vLLM |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B Q4_K_M | 16 | 955 | 723 | 1.32× | 394 | 2.42× |
| | 32 | 1288 | 1034 | 1.25× | 415 | 3.11× |
| | 64 | 1860 | 660 | 2.82× | 436 | 4.27× |
| | 128 | 2440 | 1142 | 2.14× | 441 | 5.53× |
| OLMoE-1B-7B Q4_K_M | 16 | 1750 | 1414 | 1.24× | — | |
| | 32 | 3042 | 1715 | 1.77× | — | |
| | 64 | 4514 | 1214 | 3.72× | — | |
| | 128 | 5826 | 1664 | 3.50× | — | |
| gpt-oss-20b MXFP4 | 16 | 764 | 541 | 1.41× | — | |
| | 32 | 1552 | 734 | 2.12× | — | |
| | 64 | 2535 | 932 | 2.72× | — | |
| | 128 | 3020 | 1224 | 2.47× | — | |
| Qwen3.5-4B Q4_K_M | 16 | 706 | 657 | 1.07× | — | |
| | 32 | 979 | 844 | 1.16× | — | |
| | 64 | 1206 | 730 | 1.65× | — | |
| | 128 | 1322 | 934 | 1.42× | — | |

vLLM ran the same Q4_K_M GGUF for Llama-3.1-8B. It cannot load OLMoE's GGUF and runs out of memory on the fp16 checkpoint; it refuses MXFP4 below compute capability 8.0; and for Qwen3.5-4B it ran a different quantization, so no ratio is quoted.

## Xeon E5-2680 v4, CPU

One socket, 12 cores, every engine pinned with `numactl --membind=0 taskset -c 0-11`. jitllm `-devices cpu`; llama.cpp b11222 `-t 12`; mistral.rs v0.9.4 `bench --cpu`.

| Model | jitllm prefill | llama.cpp | Ratio | jitllm decode | llama.cpp | Ratio | mistral.rs prefill / decode |
|---|---:|---:|---:|---:|---:|---:|---|
| Llama-3.2-1B Q4_K_M | 362.2 | 330.1 | 1.10× | 61.86 | 65.01 | 0.95× | 217.5 / 10.6 |
| Llama-3.1-8B Q4_K_M | 50.5 | 47.9 | 1.05× | 11.22 | 12.15 | 0.92× | 38.0 / 3.8 |
| Qwen3-8B Q4_K_M | 49.5 | 48.0 | 1.03× | 10.84 | 11.73 | 0.92× | 37.0 / 3.5 |
| OLMoE-1B-7B Q4_K_M | 241.0 | 203.9 | 1.18× | 63.19 | 59.62 | 1.06× | cannot load |
| Qwen3-30B-A3B Q4_K_M | 88.1 | 74.7 | 1.18× | 24.17 | 23.58 | 1.03× | 0.3 / 0.02 |
| gpt-oss-20b MXFP4 | 88.3 | 52.8 | 1.67× | 20.29 | 20.31 | 1.00× | 1.8 / 0.4 |

Dense decode on this CPU was 5–8% behind llama.cpp in this pass; a 12-byte scale plane (container v27) and a per-shape prefetch distance have since brought it to 0.95–0.97. Both engines are bound by memory bandwidth there. Mixtures are at parity or ahead. ZML's XLA CPU path decoded Llama-3.2-1B at 0.9 tok/s on the same socket.

## Apple M4, Metal: 3B to 7B models

Mac mini M4, 10-core GPU. Q4_K_M GGUF converted to `.jlm`, every block on the GPU (`JITLLM_VRAM=0`), against llama.cpp b10825 on Metal.

### jitllm against llama.cpp, paired

`scripts/vs-llamacpp.sh gpu`, 40 interleaved rounds, the A/A self-control run first (Gemma-3-4B against itself: prefill 0.9999, decode 0.9997). Prefill is a warm ~415-token prompt; decode is 128 tokens. Ratio is jitllm divided by llama.cpp.

| Model | Prefill ratio | Decode ratio |
|---|---:|---:|
| Ministral-3-3B-Instruct | 1.019× (486 vs 477 tok/s) | 1.095× (46.1 vs 42.2 tok/s) |
| Qwen3-4B | 1.012× | 1.030× |
| Llama-3.2-3B | 1.017× | 1.030× |
| Gemma-3-4B | 0.955× | 1.018× |
| Mistral-7B | 0.981× (noisy: IQR/median 0.057) | 1.030× |

jitllm decodes faster than llama.cpp on all five models (1.02–1.10×). Prefill is at parity except Gemma-3-4B, 4.5% behind.

### Three engines, same pass

jitllm, llama.cpp b10825 (`llama-bench`) and MLX (`mlx_lm` benchmark on the mlx-community 4-bit checkpoints), median of 3. Decode is 128 tokens; prefill is a ~415-token prompt (507 for Mistral-7B). tok/s.

| Model | Decode jitllm | llama.cpp | MLX | Prefill jitllm | llama.cpp | MLX |
|---|---:|---:|---:|---:|---:|---:|
| Ministral-3-3B | 46.18 | 41.70 | 49.99 | 490.4 | 509.6 | 519.3 |
| Qwen3-4B | 38.88 | 37.86 | 41.74 | 401.2 | 418.3 | 423.0 |
| Llama-3.2-3B | 48.61 | 47.09 | 52.78 | 524.8 | 548.9 | 598.6 |
| Gemma-3-4B | 38.89 | 38.07 | 43.15 | 429.9 | 479.0 | 458.9 |
| Mistral-7B | 23.69 | 23.16 | 24.69 | 229.3 | 224.9 | 243.0 |

Read this table with three caveats:

- **Use the paired table above for jitllm against llama.cpp.** Each jitllm number here is a fresh process, while `llama-bench` and `mlx_lm` warm up in-process, which flatters them on prefill: llama.cpp's Ministral prefill reads 510 tok/s here and 477 in the paired harness.
- **MLX does not run the same weights.** Its 4-bit files are smaller than Q4_K_M, so it reads fewer bytes per token. On these files MLX is 4–10% ahead of jitllm on decode and 5–12% on prefill.
- Running MLX checkpoints converted to `.jlm`, which would make that a same-weights comparison, is on the roadmap.

## Apple M4, Metal and CPU: small models, an earlier pass

These rows predate the Metal changes measured above and have not been re-measured. Mac mini, 4 performance and 6 efficiency cores, 10-core GPU. jitllm against llama.cpp b10985: Metal is `-devices metal` against `-ngl 99`, CPU is `-devices cpu` against `-ngl 0 -dev none`, whose CPU prefill uses Apple's Accelerate BLAS.

| Model | Tier | jitllm prefill | llama.cpp | Ratio | jitllm decode | llama.cpp | Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-360M Q8_0 | Metal | 3993 | 4352 | 0.92× | 188.5 | 180.2 | 1.05× |
| | CPU | 1158 | 1345 | 0.86× | 159.1 | 172.7 | 0.92× |
| TinyLlama-1.1B Q3_K_M | Metal | 1534 | 1511 | 1.02× | 142.4 | 116.5 | 1.22× |
| | CPU | 447 | 504 | 0.89× | 103.3 | 94.5 | 1.09× |
| Llama-3.2-1B Q4_K_M | Metal | 1529 | 1603 | 0.95× | 112.8 | 116.1 | 0.97× |
| | CPU | 524 | 342 | 1.53× | 83.0 | 85.3 | 0.97× |
| Qwen2-1.5B Q4_K_M | Metal | 1104 | 1159 | 0.95× | 90.5 | 91.8 | 0.99× |
| | CPU | 417 | 250 | 1.67× | 61.8 | 68.5 | 0.90× |
| Qwen3-MoE-4×0.6B Q4_K_M | Metal | 1655 | 1821 | 0.91× | 131.6 | 135.8 | 0.97× |
| | CPU | 405 | 278 | 1.46× | 105.7 | 113.6 | 0.93× |
| gemma-2b | Metal | 758 | 847 | 0.89× | 59.5 | 58.6 | 1.01× |
| | CPU | 251 | 297 | 0.85× | 40.9 | 42.6 | 0.96× |

In that pass Metal prefill trailed llama.cpp by 5–11% on most of these small models, and CPU decode by up to 10%. mistral.rs v0.9.4 trailed both engines everywhere it ran.

## Measure it yourself

```sh
jitllm speed -devices auto -p 512 -n 128 -r 2 models/model.jlm
llama-bench -m model.gguf -p 512 -n 128 -ngl 99 -r 3
```

Run both on the same GGUF, on an idle machine, one after the other. The repository's `scripts/vs-llamacpp.sh` interleaves the two engines and refuses a result whose spread is too wide. [Share what you find](https://github.com/jitllm/jitllm/issues), with `jitllm hardware` output, the model, the command and the numbers.

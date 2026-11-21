# Performance overview

[Back to the README](../README.md)

## Performance

Every ratio is jitllm's rate divided by the other engine's, taken in the same
pass on the same machine; above 1.00x means jitllm is faster. Single-stream rows
are a 512-token prompt (prefill) and 128 generated tokens (decode) on the same
GGUF jitllm converted from.

| Hardware | Against | Models | Prefill | Decode |
|---|---|---|---|---|
| NVIDIA V100 (CUDA), 1-5 GPUs | llama.cpp | 35, 0.4B to 120B | up to 4.62x, ahead on 34, at parity on 1 | up to 1.35x, ahead on all 35 |
| NVIDIA V100, time to first token (47-2049-token prompts, 1-5 GPUs) | llama.cpp | 11 | first on 29 of 33, up to 4.08x sooner | |
| NVIDIA V100, 8 sessions decoding together | llama.cpp (8 parallel sequences) | Llama-3.1-8B | | 1.49x |
| NVIDIA V100, batched decode (16-128 sequences) | llama.cpp | 4 | | up to 3.72x |
| NVIDIA V100, batched decode (16-128 sequences) | vLLM 0.18.1 | Llama-3.1-8B | | up to 5.53x |
| NVIDIA V100, one card, single-stream serving (514-token prompt, 256 tokens) | vLLM 0.18.1, same GGUF | Llama-3.1-8B | first token 5.8x sooner | 1.26x |
| Xeon E5-2680 v4 (CPU, 12 cores) | llama.cpp | 6 | up to 1.67x | up to 1.06x |
| Xeon E5-2680 v4 (CPU) | mistral.rs 0.9.4 | 5 | up to 1.7x | up to 6x |
| Xeon E5-2680 v4 (CPU) | ZML (bf16 weights) | 2 | | up to 69x |
| Apple M4 (Metal), paired, ~415-token prompt | llama.cpp b10825 | 5, 3B to 7B | 0.955x to 1.019x, at parity except Gemma-3-4B | 1.018x to 1.095x, ahead on all 5 |
| Apple M4 (Metal), same pass, own 4-bit weights | MLX (mlx_lm) | 5, 3B to 7B | 0.88x to 0.95x | 0.90x to 0.96x |
| Apple M4 (Metal), small models, earlier pass | llama.cpp b10985 | 7 | up to 1.02x | up to 1.53x |
| Apple M4 (CPU) | llama.cpp | 7 | up to 1.67x | up to 1.09x |
| Apple M4 (Metal and CPU) | mistral.rs 0.9.4 | 6 | up to 3.76x | up to 3.15x |

At 70B and over several cards the engines meet on decode: Llama-3.3-70B on
four V100s decodes at 0.999x llama.cpp over 256 generated tokens (two gated
passes; 80% of one card's read bandwidth each), and reaches the first token of
a 514-token prompt 1.20x sooner (one pass).

Mixtures of experts and hybrids gain the most: Qwen3-30B-A3B prefills at 3.50x
llama.cpp on two V100s and Qwen3-Next-80B at 4.62x on four. A long prompt over
several cards is pipelined across them. The rows still behind are dense decode
on the Xeon (both engines are bound by memory bandwidth), Metal prefill on Gemma-3-4B (0.955x) and on the small models of the earlier M4 pass, MLX on the M4 (on its own, smaller 4-bit files), the
first token of a ~48-token prompt (3-14% behind on one card), and two sessions
decoding together (0.94x; four together are 1.06x). Some pairs could not run:
vLLM cannot load several of these models on a V100, SGLang refuses below sm_75 and TensorRT-LLM supports Ampere and newer, mistral.rs has no V100
build, and ZML's CUDA build dropped Volta.

The [scoreboard](../.notes/perf/current.md) has every row, the commands and engine
versions, and what is measured about each loss.
[`scripts/vs-llamacpp.sh`](../scripts/vs-llamacpp.sh) measures your own machine.

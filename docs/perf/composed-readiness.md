# Composed readiness board

Host: 2x E5-2680 v4, 8x V100-SXM2-16GB; jitllm speed, graphs on, tuning off, one process a sample, 6 rounds rotated main / main again (A/A) / composed; ratios are composed / main per round, median with IQR/median, and the A/A ratio beside it.

## Serving gate

| model | cuda (TestComposedBlocksServe) | vulkan (TestComposedBlocksServe) | Xeon CPU (TestComposedCPUBlocksServe) |
|---|---|---|---|
| Llama-3.2-1B-Instruct-Q4_K_M | pass | pass | pass |
| Llama-3.2-3B-Instruct-Q4_K_M | pass | pass | pass |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | pass | pass | pass |
| qwen2.5-1.5b-instruct-q4_k_m | pass | pass | pass |
| Qwen3-0.6B-Q8_0 | pass | pass | pass |
| Qwen3-1.7B-Q4_K_M | pass | pass | pass |
| gemma-2-2b-it-Q4_K_M | pass | pass | pass |
| gemma-3-1b-it-Q4_K_M | pass | pass | pass |
| gemma-3-4b-it-Q4_K_M | FAIL | FAIL | pass |
| Ministral-3-3B-Instruct-2512-Q4_K_M | pass | pass | pass |
| Phi-3.5-mini-instruct-Q4_K_M | FAIL | pass | pass |
| Qwen3-30B-A3B-Q4_K_M | does not fit (39 of 48 blocks) | does not fit (40 of 48 blocks) | pass |
| gpt-oss-20b-Q4_K_M | pass | pass | no run |

## Decode and prompt, cuda, V100-SXM2-16GB, composed against main

| model | tg128 comp/main | pp512 comp/main | pp2048 comp/main | main tok/s tg128 | comp tok/s tg128 | ttft512 ms main/comp | ttft2048 ms main/comp |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct-Q4_K_M | 0.444 (iqr 0.037) aa 0.982 | 0.753 (iqr 0.026) aa 0.945 | 0.783 (iqr 0.029) aa 1.010 | 463.8 | 204.4 | 25/34 | 130/168 |
| Llama-3.2-3B-Instruct-Q4_K_M | 0.635 (iqr 0.008) aa 0.990 | 0.721 (iqr 0.043) aa 0.992 | 0.702 (iqr 0.022) aa 0.997 | 216.9 | 137.8 | 65/91 | 317/452 |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 0.721 (iqr 0.018) aa 0.997 | 0.848 (iqr 0.075) aa 1.052 | 0.725 (iqr 0.047) aa 1.008 | 122.4 | 88.9 | 199/235 | 633/867 |
| qwen2.5-1.5b-instruct-q4_k_m | 0.555 (iqr 0.007) aa 1.002 | 0.853 (iqr 0.015) aa 0.994 | 0.736 (iqr 0.004) aa 0.994 | 276.7 | 153.5 | 69/82 | 196/266 |
| Qwen3-0.6B-Q8_0 | 0.462 (iqr 0.023) aa 1.013 | 0.769 (iqr 0.062) aa 1.038 | 0.695 (iqr 0.018) aa 0.999 | 360.2 | 167.9 | 24/32 | 112/160 |
| Qwen3-1.7B-Q4_K_M | 0.560 (iqr 0.003) aa 1.001 | 0.799 (iqr 0.017) aa 1.000 | 0.762 (iqr 0.010) aa 0.998 | 285.1 | 159.5 | 40/50 | 188/246 |
| gemma-2-2b-it-Q4_K_M | - | - | - | - | - | - | - |
| gemma-3-1b-it-Q4_K_M | - | - | - | - | - | - | - |
| gemma-3-4b-it-Q4_K_M | 0.485 (iqr 0.007) aa 0.998 | 0.646 (iqr 0.048) aa 0.990 | 0.321 (iqr 0.013) aa 0.990 | 153.9 | 74.8 | 74/114 | 341/1063 |
| Ministral-3-3B-Instruct-2512-Q4_K_M | 0.625 (iqr 0.011) aa 0.999 | 0.714 (iqr 0.039) aa 0.992 | 0.709 (iqr 0.047) aa 1.002 | 213.8 | 133.4 | 67/95 | 324/460 |
| Phi-3.5-mini-instruct-Q4_K_M | 0.987 (iqr 0.008) aa 1.003 | 0.449 (iqr 0.106) aa 0.997 | 0.175 (iqr 0.026) aa 1.000 | 187.6 | 184.7 | 145/322 | 530/3019 |
| Qwen3-30B-A3B-Q4_K_M | 0.826 (iqr 0.011) aa 1.006 | 0.569 (iqr 0.186) aa 0.998 | 0.736 (iqr 0.056) aa 1.001 | 51.3 | 42.2 | 1001/1679 | 6840/9344 |
| gpt-oss-20b-Q4_K_M | 0.391 (iqr 0.002) aa 0.998 | 0.601 (iqr 0.063) aa 0.964 | 0.532 (iqr 0.029) aa 0.988 | 121.2 | 47.4 | 154/255 | 459/868 |

## Decode and prompt, vulkan, V100-SXM2-16GB, composed against main

| model | tg128 comp/main | pp512 comp/main | pp2048 comp/main | main tok/s tg128 | comp tok/s tg128 | ttft512 ms main/comp | ttft2048 ms main/comp |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct-Q4_K_M | - | - | - | - | - | - | - |
| Llama-3.2-3B-Instruct-Q4_K_M | 0.605 (iqr 0.036) aa 0.966 | 0.578 (iqr 0.033) aa 1.002 | 0.569 (iqr 0.003) aa 0.999 | 93.6 | 56.6 | 291/503 | 1444/2541 |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 0.687 (iqr 0.050) aa 0.978 | 0.549 (iqr 0.010) aa 0.997 | 0.543 (iqr 0.004) aa 1.001 | 64.7 | 45.4 | 639/1160 | 2918/5380 |
| qwen2.5-1.5b-instruct-q4_k_m | - | - | - | - | - | - | - |
| Qwen3-0.6B-Q8_0 | - | - | - | - | - | - | - |
| Qwen3-1.7B-Q4_K_M | - | - | - | - | - | - | - |
| gemma-2-2b-it-Q4_K_M | - | - | - | - | - | - | - |
| gemma-3-1b-it-Q4_K_M | - | - | - | - | - | - | - |
| gemma-3-4b-it-Q4_K_M | - | - | - | 0.0 | 22.8 | -/581 | -/2667 |
| Ministral-3-3B-Instruct-2512-Q4_K_M | - | - | - | - | - | - | - |
| Phi-3.5-mini-instruct-Q4_K_M | - | - | - | - | - | - | - |
| Qwen3-30B-A3B-Q4_K_M | - | - | - | 0.0 | 2.0 | -/2342 | -/11239 |
| gpt-oss-20b-Q4_K_M | - | - | - | - | - | - | - |

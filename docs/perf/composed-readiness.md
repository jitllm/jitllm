# Composed readiness board

Host: 2x E5-2680 v4 (56 threads), 8x V100-SXM2-16GB, CUDA (sm_70, ptx), jitllm at seam-wip b867ee6e. Composed is `JITLLM_GPU_COMPOSED=1` (a `jitllmcomposed` build for the server); main is the tier's own path at its shipping default, lazy tuner on, with a tune cache warmed on that card and copied into both of its arms. The composed arm has no tuner. Every ratio is composed over main, so above 1 means composed is faster, and `aa` is main over main in the same pass.

The record is `composedStructures` (`engine/model/composedrecord.go`), read per op structure (`graph.Graph.Structure`). A structure qualifies when every model that has it reads at or above main on decode and on TTFT at every length: median ratio at least 1.0, and the IQR does not straddle 1 by more than the A/A spread. Only the CUDA backend is measured here, so a qualifying structure would be recorded for `BackendPTX` alone.

Method:

- **Decode.** `jitllm speed -devices cuda:0 -p 512 -n 128 -r 3`, graphs on, one process per sample, tg128 (the median of the three repetitions). Each model runs on two different cards, 10 rounds after a discarded round 0, arms main / composed / main rotated across the round's three slots. The ratio is composed over the mean of the round's two main arms, pooled over both cards (20 rounds), with IQR/median beside it; `aa` is the second main arm over the first. Qwen3-30B-A3B takes two cards per arm (0+1 and 2+3). Each card's process is pinned to two host cores and the cards ran side by side.
- **TTFT.** `jitllmd serve -max-seq 4608 -sessions 4 -max-batch 4 -no-mem-cache`, one server per arm alone on its card (two cards for Qwen3-30B-A3B), prompts of exactly 512, 1024 and 2048 tokens, one token out, concurrency 1, `dev/httpbench` interleaving the arms round by round (12 rounds of 4 requests, 8 for Qwen3-30B-A3B) after a discarded round 0 at every length; each server is pinned to its card's two host cores and the card pairs ran side by side. Main's server cache is warmed through three passes at every length on its card first. The A/A runs main on both cards; the two rotations swap which card holds which arm. The cell is main TTFT p50 over composed TTFT p50 as the geometric mean of the two rotations (so a card's bias cancels), the larger rotation IQR/median, the card-to-card `aa`, and the two TTFT p50 in ms (main/composed).

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

Serving-gate rows are the previous board's; this pass did not rerun the serving gates.

## Decode and TTFT, cuda, V100-SXM2-16GB, composed against main

| model | tg128 comp/main | main / comp tok/s | ttft512 comp/main | ttft1024 comp/main | ttft2048 comp/main | at or above main (decode, ttft) | structures |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct-Q4_K_M | 1.027 (iqr 0.023) aa 0.997 | 573.7 / 589.5 | 0.931 (iqr 0.008) aa 1.002; 24/26 ms | 0.956 (iqr 0.026) aa 1.008; 49/51 ms | 0.973 (iqr 0.014) aa 1.003; 103/106 ms | yes, no | `ea83bca38eb4e35d` `735a759395430315` `fffa2d042c9bcbf9` |
| Llama-3.2-3B-Instruct-Q4_K_M | 0.958 (iqr 0.009) aa 1.002 | 253.1 / 242.6 | 0.941 (iqr 0.007) aa 0.998; 63/67 ms | 0.944 (iqr 0.005) aa 1.000; 130/138 ms | 0.946 (iqr 0.006) aa 0.998; 279/294 ms | no, no | `ea83bca38eb4e35d` `735a759395430315` `fffa2d042c9bcbf9` |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 1.002 (iqr 0.006) aa 1.003 | 137.9 / 138.2 | 1.055 (iqr 0.011) aa 1.015; 134/127 ms | 1.054 (iqr 0.008) aa 1.011; 275/261 ms | 1.052 (iqr 0.011) aa 1.009; 573/544 ms | yes, yes | `7860b485e6ce6a26` `735a759395430315` `fffa2d042c9bcbf9` |
| qwen2.5-1.5b-instruct-q4_k_m | 0.938 (iqr 0.005) aa 1.001 | 407.4 / 382.6 | 1.087 (iqr 0.025) aa 1.002; 42/39 ms | 1.087 (iqr 0.018) aa 1.001; 84/77 ms | 1.079 (iqr 0.017) aa 1.001; 174/161 ms | no, yes | `7860b485e6ce6a26` `735a759395430315` `580932f75413e5a2` |
| Qwen3-1.7B-Q4_K_M | 1.010 (iqr 0.018) aa 0.999 | 363.9 / 367.1 | 0.900 (iqr 0.029) aa 0.987; 40/45 ms | 0.882 (iqr 0.021) aa 0.999; 79/89 ms | 0.885 (iqr 0.013) aa 1.001; 166/187 ms | yes, no | `ea83bca38eb4e35d` `735a759395430315` `69c0515b07b74d04` |
| gemma-3-4b-it-Q4_K_M | 1.047 (iqr 0.010) aa 1.000 | 179.5 / 188.1 | 0.936 (iqr 0.004) aa 0.997; 72/77 ms | 0.916 (iqr 0.004) aa 0.996; 149/163 ms | 0.911 (iqr 0.004) aa 0.998; 313/343 ms | yes, no | `ea83bca38eb4e35d` `abce5e2fd67293ba` `7fa1c9e0a6067621` `1a61179a30f66632` |
| Phi-3.5-mini-instruct-Q4_K_M | 0.974 (iqr 0.008) aa 1.002 | 217.1 / 211.9 | 1.066 (iqr 0.008) aa 1.002; 101/95 ms | 1.130 (iqr 0.013) aa 1.003; 207/183 ms | 1.243 (iqr 0.004) aa 0.998; 483/388 ms | no, yes | `7860b485e6ce6a26` `735a759395430315` `c27240328bba77cb` |
| Ministral-3-3B-Instruct-2512-Q4_K_M | 0.972 (iqr 0.013) aa 1.002 | 244.8 / 237.7 | 0.961 (iqr 0.010) aa 1.004; 71/74 ms | 0.960 (iqr 0.005) aa 1.006; 137/142 ms | 0.955 (iqr 0.004) aa 1.006; 290/304 ms | no, no | `ea83bca38eb4e35d` `735a759395430315` `b0d36debcbfc5412` |
| gpt-oss-20b-Q4_K_M | 0.954 (iqr 0.007) aa 1.001 | 204.5 / 195.1 | 0.846 (iqr 0.011) aa 0.998; 114/135 ms | 0.837 (iqr 0.008) aa 1.000; 225/268 ms | 0.835 (iqr 0.009) aa 0.999; 454/543 ms | no, no | `182b4fd64173470a` `5e164d400549d1e1` `7860b485e6ce6a26` `735a759395430315` |
| Qwen3-30B-A3B-Q4_K_M | 0.840 (iqr 0.006) aa 1.003 | 174.1 / 146.2 | 1.054 (iqr 0.011) aa 0.992; 141/134 ms | 1.048 (iqr 0.013) aa 0.993; 216/206 ms | 1.065 (iqr 0.005) aa 0.995; 378/356 ms | no, yes | `7860b485e6ce6a26` `735a759395430315` `c24a31becb653544` |

gpt-oss-20b's greedy 16-token answer differs between main and composed at 512 in one rotation; main against main on the two cards differs the same way, so it is a near-tie in the logits and not a composed result.

## Structures

The record is per structure, so a structure qualifies only when every model that has it does. `735a759395430315`, `7860b485e6ce6a26`, `ea83bca38eb4e35d` and `fffa2d042c9bcbf9` are shared by several models; the rest belong to one model each.

| structure | models | verdict |
|---|---|---|
| `182b4fd64173470a` | gpt-oss-20b-Q4_K_M (below) | no |
| `1a61179a30f66632` | gemma-3-4b-it-Q4_K_M (below) | no |
| `580932f75413e5a2` | qwen2.5-1.5b-instruct-q4_k_m (below) | no |
| `5e164d400549d1e1` | gpt-oss-20b-Q4_K_M (below) | no |
| `69c0515b07b74d04` | Qwen3-1.7B-Q4_K_M (below) | no |
| `735a759395430315` | Llama-3.2-1B-Instruct-Q4_K_M (below), Llama-3.2-3B-Instruct-Q4_K_M (below), Meta-Llama-3.1-8B-Instruct-Q4_K_M, qwen2.5-1.5b-instruct-q4_k_m (below), Qwen3-1.7B-Q4_K_M (below), Phi-3.5-mini-instruct-Q4_K_M (below), Ministral-3-3B-Instruct-2512-Q4_K_M (below), gpt-oss-20b-Q4_K_M (below), Qwen3-30B-A3B-Q4_K_M (below) | no |
| `7860b485e6ce6a26` | Meta-Llama-3.1-8B-Instruct-Q4_K_M, qwen2.5-1.5b-instruct-q4_k_m (below), Phi-3.5-mini-instruct-Q4_K_M (below), gpt-oss-20b-Q4_K_M (below), Qwen3-30B-A3B-Q4_K_M (below) | no |
| `7fa1c9e0a6067621` | gemma-3-4b-it-Q4_K_M (below) | no |
| `abce5e2fd67293ba` | gemma-3-4b-it-Q4_K_M (below) | no |
| `b0d36debcbfc5412` | Ministral-3-3B-Instruct-2512-Q4_K_M (below) | no |
| `c24a31becb653544` | Qwen3-30B-A3B-Q4_K_M (below) | no |
| `c27240328bba77cb` | Phi-3.5-mini-instruct-Q4_K_M (below) | no |
| `ea83bca38eb4e35d` | Llama-3.2-1B-Instruct-Q4_K_M (below), Llama-3.2-3B-Instruct-Q4_K_M (below), Qwen3-1.7B-Q4_K_M (below), gemma-3-4b-it-Q4_K_M (below), Ministral-3-3B-Instruct-2512-Q4_K_M (below) | no |
| `fffa2d042c9bcbf9` | Llama-3.2-1B-Instruct-Q4_K_M (below), Llama-3.2-3B-Instruct-Q4_K_M (below), Meta-Llama-3.1-8B-Instruct-Q4_K_M | no |

No structure qualifies, so the proposed record is empty. Of the ten models only Meta-Llama-3.1-8B-Instruct reads at or above main on both axes, and its three structures are each shared with a model that reads below (`7860b485e6ce6a26` with qwen2.5-1.5b, Phi-3.5, gpt-oss-20b and Qwen3-30B-A3B; `fffa2d042c9bcbf9` with Llama-3.2-1B and 3B; `735a759395430315` with the other eight that have it). Vulkan is not measured at this bar.

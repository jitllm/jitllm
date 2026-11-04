# Divergences from llama.cpp

AGENTS.md RULE 7m governs: know which authority each choice matches, choose
per divergence, and keep this list a reviewed artefact. Each row's measurement
is in its evidence file (MC is `docs/engineering-history/model-correctness.md`).

| where | llama.cpp | jitllm follows | evidence |
|---|---|---|---|
| contraction regex `(?i:'s...)` | `[sS]` adaptation (misses U+017F, U+212A) | the model's regex: simple case folding (`ucFold`), so `'` + U+017F is a contraction | MC "The tokenizer authorities and the kimi-k2 pre-tokenizer" |
| kimi-k2 pre-tokenizer | one `\p{L}+` run | Moonshot's `pat_str`: o200k's case split plus Han runs (`SplitParams.HanRuns`) | same |
| Han ranges | CJK Ext A-F | the UCD's Scripts.txt, Unicode 15.1.0 | same |
| llama4 tokenizer label "default" | gpt-4o with ignore_merges false | the model's tokenizer.json (o200k, ignore_merges) | MC "llama4 IS THE FIFTEENTH ARCHITECTURE" |
| YaRN stated `attention_factor` | never read | transformers: the stated value on cos and sin | MC "YaRN's attention_factor" |
| YaRN correction range | floored and ceiled | unrounded where the config says `truncate: false` (`FlagYarnExact`) | MC "gpt-oss IS THE TWELFTH ARCHITECTURE" |
| the router's renormalising divisor | a clamp | DeepSeek's `sum + 1e-20` (`nn.TestRouterFloorMatchesTheKernel`) | MC "The router's renormalising divisor floor" |
| ERNIE's divisor floor and shared width | -- | DeepSeek's floor; width read off the tensor | MC "ernie4_5-moe and ernie4_5" |
| starcoder2's config sliding window | not applied | llama.cpp (the GGUF carries no window) | MC "The classic-transformer kit (C6)" |
| falcon's GELU | tanh approximation | llama.cpp (the model trains with erf) | same |
| EXAONE 4's window | slides only at 64 layers | the file's key, fixture held to transformers | MC "The dense llama family" |
| the SSM gated norm and dt | grouped norm, dt unclamped | llama.cpp and mamba_ssm (transformers clamps dt on a prompt) | MC "The state-space hybrids" |
| FalconMamba's dt/B/C norm epsilon | the layer norm's | the file, i.e. llama.cpp | MC "mamba and jamba" |
| phimoe's norms and router | phi3's graph: an RMSNorm plus a bias, a renormalised softmax top-2 | transformers' PhimoeForCausalLM: LayerNorm, sparsemixer not renormalised; the real checkpoints held to it on their own weights | MC "Phi-3.5-MoE (phimoe)" |
| phimoe's cos/sin scale | its converter's `attn_factor` | the file's on the GGUF input, `short_mscale` on safetensors (the two differ on Phi-3.5-MoE) | same |
| Jamba at Q4_K_M ("Once upon a time") | its CPU and CUDA arms part at the second token | neither engine: transformers on the file's own weights parts from both (8-bit activations through 26 recurrences); the gate holds Jamba at F16, where all agree at every position | MC "mamba and jamba" |
| gemma3n's gaussian top-k variance | divides by n-1 | the reference: divides by n | MC "gemma3n" |
| MiniMax-M3's indexer key in an f16 cache | f32 (its converter forces it) | f32: an MSA model's host cache defaults to f32, since a rounded key flips a near-tied block | MC "minimax-m3" |
| DeepSeek V4's CSA and HCA blocks | named by ratio against the constants 4 and 128 | a kind per block and a rate per kind in the container, told apart at conversion by their tensors | MC "deepseek4" |
| Kimi-Linear's and Kimi-K3's MLA latent norm epsilon | rms_norm_eps | the model's code: the RMSNorm class default, 1e-6 (`jlm.Config.LatentNormEps`) | MC "kimi-k3" |
| HunyuanVL's XD-RoPE | a coordinate per rotated PAIR (`ggml_rope_multi`) | transformers: a coordinate per ELEMENT, so a pair's halves turn by different angles on a picture's rows | MC "HunyuanVL" |
| QSA's indexer score | unscaled | llama.cpp: transformers scales by 1/sqrt(head_dim), and a positive scale on a relu sum selects the same blocks | MC "qwen4exp: Qwen3.8-Flash-Next" |
| QSA's cache precision | the converter's f16/f32 choice | f32: QSA pools raw cached keys, which its references hold at f32; a q8_0 cache is refused by name | MC "qwen4exp: what was built" |
| the n-gram reset id | its converter writes the last of the config's end-of-sequence list as `ple.eos_token_id` | llama.cpp: the file's key (transformers' module takes the first of the list; the two agree where the list has one id) | MC "qwen4exp: Qwen3.8-Flash-Next" |
| LFM2-VL's projector GELU | tanh form (FFN_GELU) | transformers' erf form (ACT2FN["gelu"]) | MC "LFM2-VL and d1's pictures" |
| d1's pictures over 1024x1024 | handed to mtmd as they came | d1's runner.py: resized to 1 MiPx with Pillow BICUBIC first (cap_pixels) | same |
| InternVL's class token | last | first, as the reference | MC "Gemma 3 vision and InternVL" |
| MiniCPM-V's image ids, resize and slicing | three departures from the processor | the processor | MC "MiniCPM-V" |
| Kimi-K3-0.40B-MXFP4's dynamic MXFP4 input activations | -- (its converter moves the weights alone) | the release's W4A16: the experts' weights moved, their activations the engine's MXFP4 kernels' own | MC "The release from safetensors" |
| BitNet b1.58 2B-4T's graph | llama.cpp's `bitnet` is 1bitLLM's reproduction (SiLU gate, NORM rotary) and its converter maps neither inner norm; bitnet.cpp's `bitnet-b1.58` runs SiLU too | transformers' BitNetForCausalLM: squared ReLU, NEOX (on the release, NLL 1.70 against SiLU's 2.51); `bitnet` is refused by name | MC "BitNet b1.58" |
| BitNet's activation quantization | per 256-element block (q8_K, TQ2_0's dot) | transformers: per token, int8 absmax over the whole row | same |
| I2_S's scales | f32 per tensor | carried exactly (`TQ2_F32`); rounding them to f16 moves the logits 4e-4..3.6e-3 in NMSE on the fixture | same |
| the chat template | the GGUF's | the publisher's when given (`-chat-template`) | MC "The chat template, applied" |
| a decision request without `instructions` | refused | TypeSafe's protocol: optional, asked by the question's id (Clef's code, lev's template) | `docs/design/decision-models.md` |
| Laya's confidence | TypeSafe's formula | Laya's code: one minus normalised entropy | same |
| Laya's sequence | the template rendered whole, then re-cut | `rl_common.build_sequence`: each piece tokenized alone (token for token equal on the gates' requests) | same |
| Laya's GELU (GeGLU and the scorer) | tanh in the GeGLU, erf in the scorer | the reference: erf in both (`ActGELUErf`, generated on every tier; worst logit 1e-6 on the f32 fixture, where tanh was 3e-4) | same |
| Laya's state cut | the context | the context: the GGUF does not carry the reference's max_len (512) | same |
| Clef's confidence | TypeSafe's formula | Clef's code (`systemone_answer`): the chosen option's probability, a score's largest | same |
| the stop set | EOS, a GGUF's eot, eom and FIM ids, and its name list | llama.cpp's from a GGUF (`tok.TestStopSetMatchesLlamaCpp`); from safetensors transformers' generation_config eos list beside the names | MC "The stop set: what the file states, carried into the container" |
| the sampler's order | penalty, top-k, top-p, min-p, then the temperature | the same order; transformers tempers first, so above or below temperature 1 its candidates differ | MC "The sampler's order of the cuts and the temperature" |
| SmolVLM's picture (idefics3) | one square, two-tap bilinear (mtmd's overview) | llama.cpp: transformers' Idefics3ImageProcessor also cuts a large picture into tiles with a Lanczos resize; the container states the square alone (`jlm.Tower`) | MC "RULE 7j" |

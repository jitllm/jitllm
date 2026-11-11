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
| the router's renormalising divisor | a clamp | DeepSeek's `sum + 1e-20` (`model.TestRouterFloorMatchesTheOracle`) | MC "The router's renormalising divisor floor" |
| ERNIE's divisor floor and shared width | -- | DeepSeek's floor; width read off the tensor | MC "ernie4_5-moe and ernie4_5" |
| starcoder2's config sliding window | not applied | llama.cpp (the GGUF carries no window) | MC "The classic-transformer kit (C6)" |
| falcon's GELU | tanh approximation | llama.cpp (the model trains with erf) | same |
| EXAONE 4's window | slides only at 64 layers | the file's key, fixture held to transformers | MC "The dense llama family" |
| the SSM gated norm and dt | grouped norm, dt unclamped | llama.cpp and mamba_ssm (transformers clamps dt on a prompt) | MC "The state-space hybrids" |
| FalconMamba's dt/B/C norm epsilon | the layer norm's | the file, i.e. llama.cpp | MC "mamba and jamba" |
| phimoe's norms and router | phi3's graph: an RMSNorm plus a bias, a renormalised softmax top-2 | transformers' PhimoeForCausalLM: LayerNorm, sparsemixer not renormalised; the real checkpoints held to it on their own weights | MC "Phi-3.5-MoE (phimoe)" |
| phi3's and phimoe's LongRoPE past `original_max_position_embeddings` | `rope_factors_long` for every position when the context (`-c`) passes it | llama.cpp's: `rope_factors_long` for every position when the instance's context (`maxSeq`) passes it, `rope_factors_short` within it (`model.RopeFor`), one set for the instance's whole history; transformers takes the long factors for every position once a forward's last position passes it, so a prompt past it under an instance within it parts from transformers, and a short prompt under an instance past it parts from transformers too | MC "Phi-3.5-MoE (phimoe)" |
| phimoe's cos/sin scale | its converter's `attn_factor` | the file's on the GGUF input, `short_mscale` on safetensors (the two differ on Phi-3.5-MoE) | same |
| Jamba at Q4_K_M ("Once upon a time") | its CPU and CUDA arms part at the second token | neither engine: transformers on the file's own weights parts from both (8-bit activations through 26 recurrences); the gate holds Jamba at F16, where all agree at every position | MC "mamba and jamba" |
| gemma3n's gaussian top-k variance | divides by n-1 | the reference: divides by n | MC "gemma3n" |
| MiniMax-M3's indexer key in an f16 cache | f32 (its converter forces it) | f32: an MSA model's host cache defaults to f32, since a rounded key flips a near-tied block | MC "minimax-m3" |
| DeepSeek V4's CSA and HCA blocks | named by ratio against the constants 4 and 128 | a kind per block and a rate per kind in the container, told apart at conversion by their tensors | MC "deepseek4" |
| Kimi-Linear's and Kimi-K3's MLA latent norm epsilon | rms_norm_eps | the model's code: the RMSNorm class default, 1e-6 (`jlm.Config.LatentNormEps`) | MC "kimi-k3" |
| HunyuanVL's XD-RoPE | a coordinate per rotated PAIR (`ggml_rope_multi`) | transformers: a coordinate per ELEMENT, so a pair's halves turn by different angles on a picture's rows | MC "HunyuanVL" |
| QSA's indexer score | unscaled | llama.cpp: transformers scales by 1/sqrt(head_dim), and a positive scale on a relu sum selects the same blocks | MC "qwen4exp: Qwen3.8-Flash-Next" |
| Clef's confidence | not the chosen option's probability (tone 0.6849 where its own largest probability is 0.7637) | Clef's systemone_answer: a choice's and a score's confidence is the largest probability (`jlm.JointHead.Confidence`) | MC "Clef's joint head as the model's readout" |
| QSA's cache precision | the converter's f16/f32 choice | f32: QSA pools raw cached keys, which its references hold at f32; a q8_0 cache the caller forces rounds them too, and its pooling is the rounded keys' | MC "qwen4exp: what was built" |
| the n-gram reset id | its converter writes the last of the config's end-of-sequence list as `ple.eos_token_id` | llama.cpp: the file's key (transformers' module takes the first of the list; the two agree where the list has one id) | MC "qwen4exp: Qwen3.8-Flash-Next" |
| a voice clip's mix and resample | mtmd's miniaudio decoder: mono at the tower's rate, miniaudio's default channel mix, its linear resampler (low-pass order MA_MAX_FILTER_ORDER) | ffmpeg's: libswresample's default matrix and polyphase Kaiser filter (`graph.OpSound`, `composer.NewSWR`), within 5e-6 of ffmpeg on every clip (the user's decision) | MC "d1-omni's voice from any WAV" |
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
| d1-omni's confidence | TypeSafe's formula (calm 0.5102 on the voice request) | d1-omni's code: the largest probability (`jlm.ConfidencePMax`; calm 0.6734 there) | `engine.TestD1OmniDecidesOverItsVoice` |
| the stop set | EOS, a GGUF's eot, eom and FIM ids, and its name list | llama.cpp's from a GGUF (`tok.TestStopSetMatchesLlamaCpp`); from safetensors transformers' generation_config eos list beside the names | MC "The stop set: what the file states, carried into the container" |
| the sampler's order | penalty, top-k, top-p, min-p, then the temperature | the same order; transformers tempers first, so above or below temperature 1 its candidates differ | MC "The sampler's order of the cuts and the temperature" |
| Qwen2-VL's picture area bounds | its own token limits | transformers' Qwen2VLImageProcessor class defaults, 56^2 to 28^2*1280 pixels: the mmproj states none, and the published preprocessor_config.json's 12845056 ceiling is not in the file (`convert.qwen2vlTower`) | MC "Vision as blocks: the tower on the one runner" |
| Qwen3-VL's picture area bounds (Qwen3.5-VL's too) | its own token limits | the published preprocessor_config.json's size, 65536 to 16777216 pixels: the mmproj states none and the processor class's defaults are Qwen2-VL's (`convert.qwen3vlTower`); its fast processor's torchvision resize is not Pillow's, the Pil processor's is the one matched | MC "Vision as blocks: the tower on the one runner" |
| Qwen3-VL's position table at the grid | the same bilinear over align-corners points | transformers' fast_pos_embed_interpolate in f32, each tap pair's weight applied separably (row taps over the column taps' sums) and the points as one division (i*(side-1)/(n-1)) where torch.linspace steps; both within a rounding (`composer.TestComposedCPUPosResampleFollowsTheReference`); a bf16 checkpoint's reference rounds the weights to bf16, the engine does not | same |
| the Qwen-VL mergers' GELU (Qwen3-VL's taps' too) | tanh form (FFN_GELU) | transformers' erf form (nn.GELU), every merger the converter states (`convert.qwenVLTower`) | MC "Vision as blocks: the tower on the one runner" |
| SmolVLM's picture (idefics3) | one square, two-tap bilinear (mtmd's overview) | llama.cpp: transformers' Idefics3ImageProcessor also cuts a large picture into tiles with a Lanczos resize; the container states the square alone (`jlm.Tower`) | MC "RULE 7j" |
| a JPEG's (and lossy WebP's) pixels | stb_image's decode | Go's decoder: its YCbCr planes, the chroma at the sample `image.YCbCr.COffset` names (no fancy upsampling), converted as `image/color.YCbCrToRGB` in the picture op (`graph.PixYCbCr8`); PIL's libjpeg upsamples chroma, so a JPEG's pixels are not transformers' -- a PNG's are | `engine.TestPictureFormatsEncodeAsTheirPixels` |
| a 16-bit picture's samples (a 16-bit PNG or TIFF) | stb_image's: each sample's high byte | each sample's high byte, read in place by the picture op (the request's sample width), as Pillow reads a 16-bit RGB picture too; Pillow opens a 16-bit gray one as mode `I` and `convert("RGB")` clips it at 255 (Pillow 10.2, measured), so a 16-bit gray picture's pixels are not transformers' | `engine.TestPictureFormatsEncodeAsTheirPixels` |
| Gemma 3n's `<end_of_image>` after a picture | the token as text: the text table's row | transformers: the vision embedder's hard row (the head appends it); a prompt's own `<end_of_image>` outside a picture still reads the text table's row | MC "Gemma 3n vision" |

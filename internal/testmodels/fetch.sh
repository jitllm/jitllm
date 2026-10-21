#!/bin/sh
# fetch.sh -- download the model files the tests read, into the model directory.
#
#   internal/testmodels/fetch.sh                 the default set (what `go test ./...` reads)
#   internal/testmodels/fetch.sh llama gemma     only entries whose name contains a word
#   internal/testmodels/fetch.sh --large         the default set plus the opt-in large files
#   internal/testmodels/fetch.sh --large scout   one large model (a split set comes whole)
#   internal/testmodels/fetch.sh --list [...]    print the table, sizes and what is here
#
# The directory is $JITLLM_MODELS, or models/ at the repository root -- the
# same answer testmodels.Dir gives a test. A file already there at the table's
# size is skipped. A download goes to <name>.part (resumed if present) and is
# renamed only once its size matches, so a half file never reads as a model.
# HF_TOKEN, when set, is sent to huggingface.co; nothing here needs it.
#
# THIS TABLE IS THE SOURCE OF TRUTH for where a test model comes from. A test
# that names a new model file adds its line here, verified: the repo and file
# against https://huggingface.co/api/models/<repo>?blobs=true, the size against
# the file itself. Every entry below was checked byte for byte (sha256) against
# the copy the goldens were taken on.
#
# Columns: name (path inside the model directory), repo, file, bytes, set.
#   repo "ollama:library/<m>" is the Ollama registry and file is the blob's
#   digest: those four GGUFs are Ollama's own conversions, and a HuggingFace
#   file of the same model is different bytes (tinyllama's has the same size
#   and another header, which convert/gguf's header test pins).
#   set "default" is what a normal test run reads. "large" is opt-in
#   (--large): over ~10 GB, or reached only through an override variable
#   (JITLLM_GREEDY_PAIR, JITLLM_MLA_MODEL, JITLLM_ROUTE_MODEL, ...).
#   An alias -- a second name for the same repo and file -- is a hard link to
#   the first name when that is already here.
#
# Committed rather than fetched: testdata/models/stories260K.gguf (1.1 MB). Its
# line below is copied from there when the size matches.
#
# NOT DOWNLOADS -- listed so nobody goes looking for them:
#
#   *.jlm containers. Tests that name one (Llama-3.2-1B-Instruct-Q4_K_M.jlm,
#     stories260K.jlm, qwen35/Qwen3.5-0.8B-Q4_K_M.jlm, ...) read the container
#     of the GGUF with the same stem. The engine/model gates write it beside
#     the GGUF on first use; or convert it yourself:
#         go run ./cmd/jitllm convert <dir>/<name>.gguf
#     A VLM container takes its projector as the second argument:
#     <name>-vlm.jlm is made by the gates; ui's qwen2vl-vlm.jlm is
#         go run ./cmd/jitllm convert Qwen2-VL-2B-Instruct-Q4_K_M.gguf \
#             mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf qwen2vl-vlm.jlm
#     gemma-2b-v13.jlm (ui's stale-container case) is a container in an old
#     format version; the current build cannot write one.
#
#   synth-* fixtures, built by the reference's own classes in a CPU
#     torch/transformers venv (hfgold.py starts several from the tiny-random
#     directories above). Each script's header says what it needs and how to
#     run it:
#         scripts/hfgold.py      synth-deepseek, -deepseek-dense, -deepseek-lite,
#                                synth-gemma, -glm4moelite, -kimilinear, -mixtral,
#                                synth-phi3, -qwen2 (directories)
#         scripts/c6gold.py      synth-phi2, -phi2-q8, -starcoder, -starcoder2,
#                                synth-commandr, -cohere2, -stablelm, -stablelm-par,
#                                synth-falcon, -falcon40, -nemotron, -dbrx (GGUFs)
#         scripts/llama4gold.py  synth-llama4, synth-llama4-q8
#         scripts/gptossgold.py  synth-gptoss
#
#   tiny-qwen3moe-f32.gguf -- scripts/tiny-moe-gguf.py, from tiny-qwen3moe-hf/.
#
#   kimik3/Kimi-K3-0.40B-F32.gguf and kimik3/Kimi-K3-0.40B-Kazakh-CPT-59M-F16.gguf
#     -- llama.cpp's own converter (a build with conversion/kimi_k3.py) on the
#     safetensors checkpoints inference-optimization/Kimi-K3-0.40B (model.safetensors
#     and its remote code) and Eraly-ml/Kimi-K3-0.40B-Kazakh-CPT-59M:
#         convert_hf_to_gguf.py <checkpoint> --outtype f32 (f16 for the Kazakh one)
#     The same checkpoints are what scripts/k3gold.py real-<name> reads (K3_REAL).
#     The safetensors gates read the float32 checkpoint as kimik3/Kimi-K3-0.40B-hf
#     and scripts/k3gold.py's fixture directory as hf/synth-kimik3 (a link to
#     where CV_HF put them is enough).
#
#   gemma3n/mmproj-gemma-3n-E2B-it-f16.gguf -- llama.cpp's converter on a
#     directory holding unsloth/gemma-3n-E2B-it's config, processor and
#     tokenizer beside its vision tensors only.
#
#   HunyuanOCR-f16.gguf and mmproj-HunyuanOCR-f16.gguf -- scripts/hunyuanvlreal.py
#     with HYVL_GGUF=1, from hunyuanvl/hf/HunyuanOCR.
#
#   The real vision goldens under vlmb/hf/<model>/goldens -- the reference
#     scripts each real-model gate names.
#
#   split-test/tl-0000N-of-00004.gguf -- tinyllama split by llama.cpp's tool:
#         llama-gguf-split --split --split-max-tensors 60 \
#             tinyllama-1.1b-q3_K_M.gguf split-test/tl
#
#   Fixtures a test writes into its own temp dir (a.gguf, broken.gguf,
#   model.gguf, m.jlm, out.jlm and the like).
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
dir=${JITLLM_MODELS:-$root/models}

table() {
	sed -e 's/#.*//' -e '/^[[:space:]]*$/d' <<'EOF'
# name                                             repo                                                   file                                                                            bytes        set
# --- reference models: small, llama-architecture ---
stories260K.gguf                                   ggml-org/models                                        tinyllamas/stories260K.gguf                                                     1185376      default
stories15M-q4_0.gguf                               ggml-org/models                                        tinyllamas/stories15M-q4_0.gguf                                                 19077344     default
stories15M-q8_0.gguf                               ggml-org/models                                        tinyllamas/stories15M-q8_0.gguf                                                 26671328     default
stories260K-infill.gguf                            ggml-org/models                                        tinyllamas/stories260K-infill.gguf                                              1185760      default
tinyllama-1.1b-q3_K_M.gguf                         ollama:library/tinyllama                               sha256:e6a691bef0258fb979ff3b1feae3002ccc6bb7f5ac5c551af6bd4bd0276213a7         550819200    default
gemma-2b.gguf                                      ollama:library/gemma                                   sha256:c1864a5eb19305c40519da12cc543519e48a0697ecd30e15d5ac228644957d12         1678447520   default
# --- language models the gates name ---
Llama-3.2-1B-Instruct-Q4_K_M.gguf                  bartowski/Llama-3.2-1B-Instruct-GGUF                   Llama-3.2-1B-Instruct-Q4_K_M.gguf                                               807694464    default
SmolLM2-360M-Instruct-Q8_0.gguf                    bartowski/SmolLM2-360M-Instruct-GGUF                   SmolLM2-360M-Instruct-Q8_0.gguf                                                 386405280    default
Qwen2-1.5B-Instruct-Q4_K_M.gguf                    Qwen/Qwen2-1.5B-Instruct-GGUF                          qwen2-1_5b-instruct-q4_k_m.gguf                                                 986045824    default
qwen2.5-1.5b-instruct-q4_k_m.gguf                  Qwen/Qwen2.5-1.5B-Instruct-GGUF                        qwen2.5-1.5b-instruct-q4_k_m.gguf                                               1117320736   default
Qwen3-0.6B-Q8_0.gguf                               Qwen/Qwen3-0.6B-GGUF                                   Qwen3-0.6B-Q8_0.gguf                                                            639446688    default
Qwen3-1.7B-Q4_K_M.gguf                             unsloth/Qwen3-1.7B-GGUF                                Qwen3-1.7B-Q4_K_M.gguf                                                          1107409472   default
Qwen3-MOE-4x0.6B-Q4_K_M.gguf                       mradermacher/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-GGUF Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder.Q4_K_M.gguf                              964652864    default
gemma-2-2b-it-Q4_K_M.gguf                          bartowski/gemma-2-2b-it-GGUF                           gemma-2-2b-it-Q4_K_M.gguf                                                       1708582752   default
gemma-3-1b-it-Q4_K_M.gguf                          unsloth/gemma-3-1b-it-GGUF                             gemma-3-1b-it-Q4_K_M.gguf                                                       806058272    default
google_gemma-3-1b-it-Q4_K_M.gguf                   bartowski/google_gemma-3-1b-it-GGUF                    google_gemma-3-1b-it-Q4_K_M.gguf                                                806058496    default
gemma-3-4b-it-Q4_K_M.gguf                          ggml-org/gemma-3-4b-it-GGUF                            gemma-3-4b-it-Q4_K_M.gguf                                                       2489757856   default
Phi-3.5-mini-instruct-Q4_K_M.gguf                  bartowski/Phi-3.5-mini-instruct-GGUF                   Phi-3.5-mini-instruct-Q4_K_M.gguf                                               2393232672   default
DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf          bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF           DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf                                       1117320800   default
Mistral-7B-Instruct-v0.3-Q4_K_M.gguf               bartowski/Mistral-7B-Instruct-v0.3-GGUF                Mistral-7B-Instruct-v0.3-Q4_K_M.gguf                                            4372812000   default
Mistral-7B-Instruct-v0.3/tokenizer_config.json     mistralai/Mistral-7B-Instruct-v0.3                     tokenizer_config.json                                                           140874       default
olmoe-1b-7b-0924-instruct-Q4_K_M.gguf              allenai/OLMoE-1B-7B-0924-Instruct-GGUF                 olmoe-1b-7b-0924-instruct-q4_k_m.gguf                                           4213512672   default
# the same file under the name format/jlm and jit_old/gpu/tier ask for (alias)
olmoe-1b-7b-Q4_K_M.gguf                            allenai/OLMoE-1B-7B-0924-Instruct-GGUF                 olmoe-1b-7b-0924-instruct-q4_k_m.gguf                                           4213512672   default
granite/granite-3.3-2b-instruct-Q4_K_M.gguf        ibm-granite/granite-3.3-2b-instruct-GGUF               granite-3.3-2b-instruct-Q4_K_M.gguf                                             1545303328   default
granite/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf  bartowski/granite-3.1-1b-a400m-instruct-GGUF           granite-3.1-1b-a400m-instruct-Q4_K_M.gguf                                       821847360    default
granite/granite-4.0-micro-Q4_K_M.gguf              ibm-granite/granite-4.0-micro-GGUF                     granite-4.0-micro-Q4_K_M.gguf                                                   2099502528   default
qwen35/Qwen3.5-0.8B-Q4_K_M.gguf                    unsloth/Qwen3.5-0.8B-GGUF                              Qwen3.5-0.8B-Q4_K_M.gguf                                                        532517120    default
hunyuan/tencent_Hunyuan-0.5B-Instruct-Q4_K_M.gguf  bartowski/tencent_Hunyuan-0.5B-Instruct-GGUF           tencent_Hunyuan-0.5B-Instruct-Q4_K_M.gguf                                       354970464    default
newarch/OLMo-2-0425-1B-Instruct-Q4_K_M.gguf        allenai/OLMo-2-0425-1B-Instruct-GGUF                   OLMo-2-0425-1B-Instruct-Q4_K_M.gguf                                             935515296    default
newarch/EXAONE-4.0-1.2B-Q4_K_M.gguf                LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF                       EXAONE-4.0-1.2B-Q4_K_M.gguf                                                     812437792    default
newarch/SmolLM3-3B-Q4_K_M.gguf                     unsloth/SmolLM3-3B-GGUF                                SmolLM3-3B-Q4_K_M.gguf                                                          1915306528   default
newarch/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf   mistralai/Ministral-3-3B-Instruct-2512-GGUF            Ministral-3-3B-Instruct-2512-Q4_K_M.gguf                                        2147023008   default
newarch/AFM-4.5B-Q4_K_M.gguf                       arcee-ai/AFM-4.5B-GGUF                                 AFM-4.5B-Q4_K_M.gguf                                                            2916301824   default
# --- vision: text model and projector ---
SmolVLM-256M-Instruct-Q8_0.gguf                    ggml-org/SmolVLM-256M-Instruct-GGUF                    SmolVLM-256M-Instruct-Q8_0.gguf                                                 175054528    default
mmproj-SmolVLM-256M-Instruct-Q8_0.gguf             ggml-org/SmolVLM-256M-Instruct-GGUF                    mmproj-SmolVLM-256M-Instruct-Q8_0.gguf                                          103769856    default
Qwen2-VL-2B-Instruct-Q4_K_M.gguf                   ggml-org/Qwen2-VL-2B-Instruct-GGUF                     Qwen2-VL-2B-Instruct-Q4_K_M.gguf                                                986046944    default
mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf              ggml-org/Qwen2-VL-2B-Instruct-GGUF                     mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf                                           709883360    default
llava-phi-3-mini-int4.gguf                         xtuner/llava-phi-3-mini-gguf                           llava-phi-3-mini-int4.gguf                                                      2318919200   default
llava-phi-3-mini-mmproj-f16.gguf                   xtuner/llava-phi-3-mini-gguf                           llava-phi-3-mini-mmproj-f16.gguf                                                607648960    default
gemma3n/gemma-3n-E2B-it-Q4_K_M.gguf                unsloth/gemma-3n-E2B-it-GGUF                           gemma-3n-E2B-it-Q4_K_M.gguf                                                     3026881888   default
vlmb/gemma-4-E2B-it-Q4_K_M.gguf                    unsloth/gemma-4-E2B-it-GGUF                            gemma-4-E2B-it-Q4_K_M.gguf                                                      3106738272   default
vlmb/mmproj-gemma-4-E2B-it-BF16.gguf               ggml-org/gemma-4-E2B-it-GGUF                           mmproj-gemma-4-E2B-it-BF16.gguf                                                 986833664    default
vlmb/mmproj-Ministral-3-3B-BF16.gguf               mistralai/Ministral-3-3B-Instruct-2512-GGUF            Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf                                   841501856    default
# --- embedding models, in embed/ so no decoder sweep picks them up ---
# (names and goldens: scripts/embedgold.py, testdata/golden/embed)
embed/all-minilm-l6.gguf                           second-state/All-MiniLM-L6-v2-Embedding-GGUF           all-MiniLM-L6-v2-ggml-model-f16.gguf                                            45949216     default
embed/snowflake-arctic-embed-22m.gguf              ollama:library/snowflake-arctic-embed                  sha256:a83b0493f894772b41e5cfe3e6effc288a94d300cfbbeb900891f8377f358bbf         45827008     default
embed/mxbai-embed-large.gguf                       ChristianAzinn/mxbai-embed-large-v1-gguf               mxbai-embed-large-v1_fp16.gguf                                                  669603712    default
embed/nomic-embed-text.gguf                        ollama:library/nomic-embed-text                        sha256:970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6         274290656    default
embed/qwen3-embedding-0.6b.gguf                    Qwen/Qwen3-Embedding-0.6B-GGUF                         Qwen3-Embedding-0.6B-Q8_0.gguf                                                  639150592    default
embed/qwen3-embedding-0.6b-f16.gguf                Qwen/Qwen3-Embedding-0.6B-GGUF                         Qwen3-Embedding-0.6B-f16.gguf                                                   1197629632   default
embed/embeddinggemma-300m.gguf                     ollama:library/embeddinggemma                          sha256:0800cbac9c2064dde519420e75e512a83cb360de3ad5df176185dc69652fc515         621867104    default
embed/embeddinggemma-300M-BF16-lcpp.gguf           unsloth/embeddinggemma-300m-GGUF                       embeddinggemma-300M-BF16.gguf                                                   612429792    default
# convert/gguf's header test pins this file (it still reads it from a fixed path)
embed/MiniLM-L6-v2.Q8_0.gguf                       leliuga/all-MiniLM-L6-v2-GGUF                          all-MiniLM-L6-v2.Q8_0.gguf                                                      25008064     default
# --- HuggingFace directories: the safetensors converter's fixtures ---
# (goldens: scripts/hfgold.py, testdata/golden/hf; the synth-* ones are built from these)
SmolLM2-360M-Instruct/config.json                  HuggingFaceTB/SmolLM2-360M-Instruct                    config.json                                                                     846          default
SmolLM2-360M-Instruct/generation_config.json       HuggingFaceTB/SmolLM2-360M-Instruct                    generation_config.json                                                          132          default
SmolLM2-360M-Instruct/model.safetensors            HuggingFaceTB/SmolLM2-360M-Instruct                    model.safetensors                                                               723674912    default
SmolLM2-360M-Instruct/special_tokens_map.json      HuggingFaceTB/SmolLM2-360M-Instruct                    special_tokens_map.json                                                         655          default
SmolLM2-360M-Instruct/tokenizer_config.json        HuggingFaceTB/SmolLM2-360M-Instruct                    tokenizer_config.json                                                           3764         default
SmolLM2-360M-Instruct/tokenizer.json               HuggingFaceTB/SmolLM2-360M-Instruct                    tokenizer.json                                                                  2104556      default
qwen3/added_tokens.json                            yujiepan/qwen3-tiny-random                             added_tokens.json                                                               707          default
qwen3/chat_template.jinja                          yujiepan/qwen3-tiny-random                             chat_template.jinja                                                             4116         default
qwen3/config.json                                  yujiepan/qwen3-tiny-random                             config.json                                                                     725          default
qwen3/generation_config.json                       yujiepan/qwen3-tiny-random                             generation_config.json                                                          248          default
qwen3/merges.txt                                   yujiepan/qwen3-tiny-random                             merges.txt                                                                      1671853      default
qwen3/model.safetensors                            yujiepan/qwen3-tiny-random                             model.safetensors                                                               19598752     default
qwen3/special_tokens_map.json                      yujiepan/qwen3-tiny-random                             special_tokens_map.json                                                         613          default
qwen3/tokenizer_config.json                        yujiepan/qwen3-tiny-random                             tokenizer_config.json                                                           5404         default
qwen3/tokenizer.json                               yujiepan/qwen3-tiny-random                             tokenizer.json                                                                  11422654     default
qwen3/vocab.json                                   yujiepan/qwen3-tiny-random                             vocab.json                                                                      2776833      default
qwen3-moe/added_tokens.json                        yujiepan/qwen3-moe-tiny-random                         added_tokens.json                                                               707          default
qwen3-moe/chat_template.jinja                      yujiepan/qwen3-moe-tiny-random                         chat_template.jinja                                                             4116         default
qwen3-moe/config.json                              yujiepan/qwen3-moe-tiny-random                         config.json                                                                     959          default
qwen3-moe/generation_config.json                   yujiepan/qwen3-moe-tiny-random                         generation_config.json                                                          248          default
qwen3-moe/merges.txt                               yujiepan/qwen3-moe-tiny-random                         merges.txt                                                                      1671853      default
qwen3-moe/model.safetensors                        yujiepan/qwen3-moe-tiny-random                         model.safetensors                                                               19946416     default
qwen3-moe/special_tokens_map.json                  yujiepan/qwen3-moe-tiny-random                         special_tokens_map.json                                                         613          default
qwen3-moe/tokenizer_config.json                    yujiepan/qwen3-moe-tiny-random                         tokenizer_config.json                                                           5404         default
qwen3-moe/tokenizer.json                           yujiepan/qwen3-moe-tiny-random                         tokenizer.json                                                                  11422654     default
qwen3-moe/vocab.json                               yujiepan/qwen3-moe-tiny-random                         vocab.json                                                                      2776833      default
qwen3-next-moe/added_tokens.json                   yujiepan/qwen3-next-moe-tiny-random                    added_tokens.json                                                               707          default
qwen3-next-moe/chat_template.jinja                 yujiepan/qwen3-next-moe-tiny-random                    chat_template.jinja                                                             4040         default
qwen3-next-moe/config.json                         yujiepan/qwen3-next-moe-tiny-random                    config.json                                                                     1281         default
qwen3-next-moe/generation_config.json              yujiepan/qwen3-next-moe-tiny-random                    generation_config.json                                                          247          default
qwen3-next-moe/merges.txt                          yujiepan/qwen3-next-moe-tiny-random                    merges.txt                                                                      1671853      default
qwen3-next-moe/model.safetensors                   yujiepan/qwen3-next-moe-tiny-random                    model.safetensors                                                               5740400      default
qwen3-next-moe/special_tokens_map.json             yujiepan/qwen3-next-moe-tiny-random                    special_tokens_map.json                                                         613          default
qwen3-next-moe/tokenizer_config.json               yujiepan/qwen3-next-moe-tiny-random                    tokenizer_config.json                                                           5405         default
qwen3-next-moe/tokenizer.json                      yujiepan/qwen3-next-moe-tiny-random                    tokenizer.json                                                                  11422654     default
qwen3-next-moe/vocab.json                          yujiepan/qwen3-next-moe-tiny-random                    vocab.json                                                                      2776833      default
qwen2.5-tiny-random/added_tokens.json              yujiepan/qwen2.5-tiny-random                           added_tokens.json                                                               605          default
qwen2.5-tiny-random/config.json                    yujiepan/qwen2.5-tiny-random                           config.json                                                                     702          default
qwen2.5-tiny-random/generation_config.json         yujiepan/qwen2.5-tiny-random                           generation_config.json                                                          243          default
qwen2.5-tiny-random/merges.txt                     yujiepan/qwen2.5-tiny-random                           merges.txt                                                                      1671853      default
qwen2.5-tiny-random/model.safetensors              yujiepan/qwen2.5-tiny-random                           model.safetensors                                                               4871288      default
qwen2.5-tiny-random/special_tokens_map.json        yujiepan/qwen2.5-tiny-random                           special_tokens_map.json                                                         613          default
qwen2.5-tiny-random/tokenizer_config.json          yujiepan/qwen2.5-tiny-random                           tokenizer_config.json                                                           7306         default
qwen2.5-tiny-random/tokenizer.json                 yujiepan/qwen2.5-tiny-random                           tokenizer.json                                                                  7031645      default
qwen2.5-tiny-random/vocab.json                     yujiepan/qwen2.5-tiny-random                           vocab.json                                                                      2776833      default
tiny-qwen3moe-hf/added_tokens.json                 trl-internal-testing/tiny-Qwen3MoeForCausalLM          added_tokens.json                                                               707          default
tiny-qwen3moe-hf/config.json                       trl-internal-testing/tiny-Qwen3MoeForCausalLM          config.json                                                                     949          default
tiny-qwen3moe-hf/generation_config.json            trl-internal-testing/tiny-Qwen3MoeForCausalLM          generation_config.json                                                          214          default
tiny-qwen3moe-hf/merges.txt                        trl-internal-testing/tiny-Qwen3MoeForCausalLM          merges.txt                                                                      1671853      default
tiny-qwen3moe-hf/model.safetensors                 trl-internal-testing/tiny-Qwen3MoeForCausalLM          model.safetensors                                                               5212168      default
tiny-qwen3moe-hf/special_tokens_map.json           trl-internal-testing/tiny-Qwen3MoeForCausalLM          special_tokens_map.json                                                         613          default
tiny-qwen3moe-hf/tokenizer_config.json             trl-internal-testing/tiny-Qwen3MoeForCausalLM          tokenizer_config.json                                                           5404         default
tiny-qwen3moe-hf/tokenizer.json                    trl-internal-testing/tiny-Qwen3MoeForCausalLM          tokenizer.json                                                                  11422654     default
tiny-qwen3moe-hf/vocab.json                        trl-internal-testing/tiny-Qwen3MoeForCausalLM          vocab.json                                                                      2776833      default
tiny-random-LlamaForCausalLM/config.json           trl-internal-testing/tiny-random-LlamaForCausalLM      config.json                                                                     531          default
tiny-random-LlamaForCausalLM/generation_config.json trl-internal-testing/tiny-random-LlamaForCausalLM     generation_config.json                                                          137          default
tiny-random-LlamaForCausalLM/model.safetensors     trl-internal-testing/tiny-random-LlamaForCausalLM      model.safetensors                                                               4131512      default
tiny-random-LlamaForCausalLM/special_tokens_map.json trl-internal-testing/tiny-random-LlamaForCausalLM    special_tokens_map.json                                                         552          default
tiny-random-LlamaForCausalLM/tokenizer_config.json trl-internal-testing/tiny-random-LlamaForCausalLM      tokenizer_config.json                                                           822          default
tiny-random-LlamaForCausalLM/tokenizer.json        trl-internal-testing/tiny-random-LlamaForCausalLM      tokenizer.json                                                                  1842792      default
tiny-random-LlamaForCausalLM/tokenizer.model       trl-internal-testing/tiny-random-LlamaForCausalLM      tokenizer.model                                                                 499723       default
tiny-random-GemmaForCausalLM/config.json           Xenova/tiny-random-GemmaForCausalLM                    config.json                                                                     691          default
tiny-random-GemmaForCausalLM/generation_config.json Xenova/tiny-random-GemmaForCausalLM                   generation_config.json                                                          137          default
tiny-random-GemmaForCausalLM/model.safetensors     Xenova/tiny-random-GemmaForCausalLM                    model.safetensors                                                               8198848      default
tiny-random-GemmaForCausalLM/special_tokens_map.json Xenova/tiny-random-GemmaForCausalLM                  special_tokens_map.json                                                         636          default
tiny-random-GemmaForCausalLM/tokenizer_config.json Xenova/tiny-random-GemmaForCausalLM                    tokenizer_config.json                                                           2147         default
tiny-random-GemmaForCausalLM/tokenizer.json        Xenova/tiny-random-GemmaForCausalLM                    tokenizer.json                                                                  17477957     default
tiny-random-GemmaForCausalLM/tokenizer.model       Xenova/tiny-random-GemmaForCausalLM                    tokenizer.model                                                                 4241003      default
# the weights have no tokenizer.json beside them; Mixtral-Instruct's own is used
tiny-random-MixtralForCausalLM/config.json         hf-tiny-v2/tiny-random-MixtralForCausalLM              config.json                                                                     850          default
tiny-random-MixtralForCausalLM/generation_config.json hf-tiny-v2/tiny-random-MixtralForCausalLM           generation_config.json                                                          221          default
tiny-random-MixtralForCausalLM/model.safetensors   hf-tiny-v2/tiny-random-MixtralForCausalLM              model.safetensors                                                               264736       default
tiny-random-MixtralForCausalLM/tokenizer.model     hf-tiny-v2/tiny-random-MixtralForCausalLM              tokenizer.model                                                                 493443       default
tiny-random-MixtralForCausalLM/tokenizer_config.json mistralai/Mixtral-8x7B-Instruct-v0.1                 tokenizer_config.json                                                           2103         default
tiny-random-MixtralForCausalLM/tokenizer.json      mistralai/Mixtral-8x7B-Instruct-v0.1                   tokenizer.json                                                                  1795188      default
tiny-random-OlmoeForCausalLM/config.json           hf-tiny-v2/tiny-random-OlmoeForCausalLM                config.json                                                                     942          default
tiny-random-OlmoeForCausalLM/generation_config.json hf-tiny-v2/tiny-random-OlmoeForCausalLM               generation_config.json                                                          204          default
tiny-random-OlmoeForCausalLM/model.safetensors     hf-tiny-v2/tiny-random-OlmoeForCausalLM                model.safetensors                                                               13013656     default
tiny-random-OlmoeForCausalLM/tokenizer_config.json hf-tiny-v2/tiny-random-OlmoeForCausalLM                tokenizer_config.json                                                           367          default
tiny-random-OlmoeForCausalLM/tokenizer.json        hf-tiny-v2/tiny-random-OlmoeForCausalLM                tokenizer.json                                                                  3565673      default
tiny-random-phi3/added_tokens.json                 katuni4ka/tiny-random-phi3                             added_tokens.json                                                               940          default
tiny-random-phi3/config.json                       katuni4ka/tiny-random-phi3                             config.json                                                                     935          default
tiny-random-phi3/generation_config.json            katuni4ka/tiny-random-phi3                             generation_config.json                                                          140          default
tiny-random-phi3/model.safetensors                 katuni4ka/tiny-random-phi3                             model.safetensors                                                               12425728     default
tiny-random-phi3/special_tokens_map.json           katuni4ka/tiny-random-phi3                             special_tokens_map.json                                                         623          default
tiny-random-phi3/tokenizer_config.json             katuni4ka/tiny-random-phi3                             tokenizer_config.json                                                           8075         default
tiny-random-phi3/tokenizer.json                    katuni4ka/tiny-random-phi3                             tokenizer.json                                                                  1849861      default
tiny-random-phi3/tokenizer.model                   katuni4ka/tiny-random-phi3                             tokenizer.model                                                                 499723       default
tiny-random-Phi3ForCausalLM/added_tokens.json      Xenova/tiny-random-Phi3ForCausalLM                     added_tokens.json                                                               293          default
tiny-random-Phi3ForCausalLM/config.json            Xenova/tiny-random-Phi3ForCausalLM                     config.json                                                                     724          default
tiny-random-Phi3ForCausalLM/generation_config.json Xenova/tiny-random-Phi3ForCausalLM                     generation_config.json                                                          145          default
tiny-random-Phi3ForCausalLM/model.safetensors      Xenova/tiny-random-Phi3ForCausalLM                     model.safetensors                                                               8292520      default
tiny-random-Phi3ForCausalLM/special_tokens_map.json Xenova/tiny-random-Phi3ForCausalLM                    special_tokens_map.json                                                         569          default
tiny-random-Phi3ForCausalLM/tokenizer_config.json  Xenova/tiny-random-Phi3ForCausalLM                     tokenizer_config.json                                                           3150         default
tiny-random-Phi3ForCausalLM/tokenizer.json         Xenova/tiny-random-Phi3ForCausalLM                     tokenizer.json                                                                  1844868      default
tiny-random-Phi3ForCausalLM/tokenizer.model        Xenova/tiny-random-Phi3ForCausalLM                     tokenizer.model                                                                 499723       default
# Kimi-Linear's tokenizer and config, not its weights: the tiktoken gates read these
Kimi-Linear-48B-A3B-Instruct/chat_template.jinja   moonshotai/Kimi-Linear-48B-A3B-Instruct                chat_template.jinja                                                             1850         default
Kimi-Linear-48B-A3B-Instruct/config.json           moonshotai/Kimi-Linear-48B-A3B-Instruct                config.json                                                                     1751         default
Kimi-Linear-48B-A3B-Instruct/configuration_kimi.py moonshotai/Kimi-Linear-48B-A3B-Instruct                configuration_kimi.py                                                           4948         default
Kimi-Linear-48B-A3B-Instruct/generation_config.json moonshotai/Kimi-Linear-48B-A3B-Instruct               generation_config.json                                                          147          default
Kimi-Linear-48B-A3B-Instruct/model.safetensors.index.json moonshotai/Kimi-Linear-48B-A3B-Instruct         model.safetensors.index.json                                                    1987740      default
Kimi-Linear-48B-A3B-Instruct/special_tokens_map.json moonshotai/Kimi-Linear-48B-A3B-Instruct              special_tokens_map.json                                                         5583         default
Kimi-Linear-48B-A3B-Instruct/tiktoken.model        moonshotai/Kimi-Linear-48B-A3B-Instruct                tiktoken.model                                                                  2795286      default
Kimi-Linear-48B-A3B-Instruct/tokenization_kimi.py  moonshotai/Kimi-Linear-48B-A3B-Instruct                tokenization_kimi.py                                                            12534        default
Kimi-Linear-48B-A3B-Instruct/tokenizer_config.json moonshotai/Kimi-Linear-48B-A3B-Instruct                tokenizer_config.json                                                           3695         default
# Kimi-K3-0.40B, a trained K3 test model, as published GGUFs: TestKimiK3RealMatchesReference
kimik3/Kimi-K3-0.40B-F16.gguf                      murillo2000/Kimi-K3-0.40B-GGUF                         Kimi-K3-0.40B-F16.gguf                                                          784318432    default
kimik3/Kimi-K3-0.40B-MXFP4.gguf                    murillo2000/Kimi-K3-0.40B-GGUF                         Kimi-K3-0.40B-MXFP4.gguf                                                        751976576    default
kimik3/Kimi-K3-0.40B.Q8_0.gguf                     mradermacher/Kimi-K3-0.40B-GGUF                        Kimi-K3-0.40B.Q8_0.gguf                                                         420148096    default
kimik3/Kimi-K3-0.40B.Q4_K_M.gguf                   mradermacher/Kimi-K3-0.40B-GGUF                        Kimi-K3-0.40B.Q4_K_M.gguf                                                       272663424    default
# ... and in the release's format, compressed-tensors MXFP4 experts, as safetensors with its remote code:
# TestKimiK3RealMatchesReference, convert.TestK3MXFP4ExpertsMatchTheGGUF
kimik3/Kimi-K3-0.40B-MXFP4-hf/config.json          inference-optimization/Kimi-K3-0.40B-MXFP4 config.json                                                                     11130        default
kimik3/Kimi-K3-0.40B-MXFP4-hf/generation_config.json inference-optimization/Kimi-K3-0.40B-MXFP4 generation_config.json                                                          222          default
kimik3/Kimi-K3-0.40B-MXFP4-hf/tokenizer_config.json inference-optimization/Kimi-K3-0.40B-MXFP4 tokenizer_config.json                                                           4856         default
kimik3/Kimi-K3-0.40B-MXFP4-hf/tiktoken.model       inference-optimization/Kimi-K3-0.40B-MXFP4 tiktoken.model                                                                  2795286      default
kimik3/Kimi-K3-0.40B-MXFP4-hf/tokenization_kimi.py inference-optimization/Kimi-K3-0.40B-MXFP4 tokenization_kimi.py                                                            16145        default
kimik3/Kimi-K3-0.40B-MXFP4-hf/configuration_kimi_k3.py inference-optimization/Kimi-K3-0.40B-MXFP4 configuration_kimi_k3.py                                                        11215        default
kimik3/Kimi-K3-0.40B-MXFP4-hf/modeling_kimi_linear.py inference-optimization/Kimi-K3-0.40B-MXFP4 modeling_kimi_linear.py                                                         39           default
kimik3/Kimi-K3-0.40B-MXFP4-hf/modeling_kimi_k3_linear.py inference-optimization/Kimi-K3-0.40B-MXFP4 modeling_kimi_k3_linear.py                                                      54294        default
kimik3/Kimi-K3-0.40B-MXFP4-hf/modeling_kimi_k3.py  inference-optimization/Kimi-K3-0.40B-MXFP4 modeling_kimi_k3.py                                                             52097        default
kimik3/Kimi-K3-0.40B-MXFP4-hf/model.safetensors    inference-optimization/Kimi-K3-0.40B-MXFP4 model.safetensors                                                               1472194984   default
# --- large: opt-in with --large ---
falcon/falcon-7b-instruct.Q4_K_M.gguf              mradermacher/falcon-7b-instruct-GGUF                   falcon-7b-instruct.Q4_K_M.gguf                                                  4975127552   large
phi-4-Q4_K_M.gguf                                  bartowski/phi-4-GGUF                                   phi-4-Q4_K_M.gguf                                                               9053114816   large
DeepSeek-V2-Lite.Q4_K_M.gguf                       mradermacher/DeepSeek-V2-Lite-GGUF                     DeepSeek-V2-Lite.Q4_K_M.gguf                                                    10364416736  large
Moonlight-16B-A3B-Instruct-Q4_K_M.gguf             gabriellarson/Moonlight-16B-A3B-Instruct-GGUF          Moonlight-16B-A3B-Instruct-Q4_K_M.gguf                                          10540747360  large
gpt-oss-20b-Q4_K_M.gguf                            unsloth/gpt-oss-20b-GGUF                               gpt-oss-20b-Q4_K_M.gguf                                                         11624759488  large
Mixtral-8x7B-Instruct-Q3_K_M.gguf                  TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF               mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf                                          20363356096  large
Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf            unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF               Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf                                         48506100512  large
Kimi-Linear-Q8_0-GGUF/moonshotai_Kimi-Linear-48B-A3B-Instruct-Q8_0-00001-of-00002.gguf bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF moonshotai_Kimi-Linear-48B-A3B-Instruct-Q8_0/moonshotai_Kimi-Linear-48B-A3B-Instruct-Q8_0-00001-of-00002.gguf 39761166432 large
Kimi-Linear-Q8_0-GGUF/moonshotai_Kimi-Linear-48B-A3B-Instruct-Q8_0-00002-of-00002.gguf bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF moonshotai_Kimi-Linear-48B-A3B-Instruct-Q8_0/moonshotai_Kimi-Linear-48B-A3B-Instruct-Q8_0-00002-of-00002.gguf 12487220000 large
llama4/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF Q4_K_M/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf 49848379744 large
llama4/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00002-of-00002.gguf unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF Q4_K_M/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00002-of-00002.gguf 15511520608 large
# --- the state-space hybrids and short convolutions: TestSSMRealModelsTeacherForced's list ---
granite/granite-4.0-h-350m-Q8_0.gguf               ibm-granite/granite-4.0-h-350m-GGUF                    granite-4.0-h-350m-Q8_0.gguf                                                    366195616    default
granite/granite-4.0-h-1b-Q4_K_M.gguf               ibm-granite/granite-4.0-h-1b-GGUF                      granite-4.0-h-1b-Q4_K_M.gguf                                                    901162208    default
granite/granite-4.0-h-tiny-Q4_K_M.gguf             ibm-granite/granite-4.0-h-tiny-GGUF                    granite-4.0-h-tiny-Q4_K_M.gguf                                                  4230976352   default
nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF    nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf                                   6525629280   default
falconh1/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf       tiiuae/Falcon-H1-0.5B-Instruct-GGUF                    Falcon-H1-0.5B-Instruct-Q4_K_M.gguf                                             314806560    default
falconh1/Falcon-H1-1.5B-Instruct-Q8_0.gguf         tiiuae/Falcon-H1-1.5B-Instruct-GGUF                    Falcon-H1-1.5B-Instruct-Q8_0.gguf                                               1656492736   default
lfm2/LFM2-350M-Q4_K_M.gguf                         LiquidAI/LFM2-350M-GGUF                                LFM2-350M-Q4_K_M.gguf                                                           229309376    default
lfm2/LFM2-1.2B-Q4_K_M.gguf                         LiquidAI/LFM2-1.2B-GGUF                                LFM2-1.2B-Q4_K_M.gguf                                                           730893248    default
lfm2/LFM2-8B-A1B-Q4_K_M.gguf                       LiquidAI/LFM2-8B-A1B-GGUF                              LFM2-8B-A1B-Q4_K_M.gguf                                                         5044779712   default
jamba/jamba-reasoning-3b-F16.gguf                  ai21labs/AI21-Jamba-Reasoning-3B-GGUF                  jamba-reasoning-3b-F16.gguf                                                     6402230208   default
mamba/mamba-130m-hf.Q8_0.gguf                      Felladrin/gguf-mamba-130m-hf                           mamba-130m-hf.Q8_0.gguf                                                         155395936    default
falconmamba/falcon-mamba-7B-Q4_K_M.gguf            tiiuae/falcon-mamba-7b-Q4_K_M-GGUF                     falcon-mamba-7B-Q4_K_M.gguf                                                     4204230432   default
mamba2/mamba-codestral-7b-v0.1-q4_k_m.gguf         viniciusianni/Mamba-Codestral-7B-v0.1-Q4_K_M-GGUF      mamba-codestral-7b-v0.1-q4_k_m.gguf                                             4147481568   default
nemotron/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF       nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf                                      24655899872  large
# --- the modern text group, reached through JITLLM_GREEDY_MODEL or JITLLM_REF_MODEL ---
inclusionAI_Ling-mini-2.0-Q4_K_M.gguf              bartowski/inclusionAI_Ling-mini-2.0-GGUF               inclusionAI_Ling-mini-2.0-Q4_K_M.gguf                                           9941894336   large
Phi-mini-MoE-instruct-Q4_K_M.gguf                  gabriellarson/Phi-mini-MoE-instruct-GGUF               Phi-mini-MoE-instruct-Q4_K_M.gguf                                               4993133376   large
swiss-ai_Apertus-8B-Instruct-2509-Q4_K_M.gguf      bartowski/swiss-ai_Apertus-8B-Instruct-2509-GGUF       swiss-ai_Apertus-8B-Instruct-2509-Q4_K_M.gguf                                   5057885568   large
dots1/rednote-hilab_dots.llm1.inst-Q4_K_M-00001-of-00003.gguf bartowski/rednote-hilab_dots.llm1.inst-GGUF rednote-hilab_dots.llm1.inst-Q4_K_M/rednote-hilab_dots.llm1.inst-Q4_K_M-00001-of-00003.gguf 39768344544 large
dots1/rednote-hilab_dots.llm1.inst-Q4_K_M-00002-of-00003.gguf bartowski/rednote-hilab_dots.llm1.inst-GGUF rednote-hilab_dots.llm1.inst-Q4_K_M/rednote-hilab_dots.llm1.inst-Q4_K_M-00002-of-00003.gguf 39631424480 large
dots1/rednote-hilab_dots.llm1.inst-Q4_K_M-00003-of-00003.gguf bartowski/rednote-hilab_dots.llm1.inst-GGUF rednote-hilab_dots.llm1.inst-Q4_K_M/rednote-hilab_dots.llm1.inst-Q4_K_M-00003-of-00003.gguf 16720542784 large
Qwen3VL-2B-Instruct-Q4_K_M.gguf                    Qwen/Qwen3-VL-2B-Instruct-GGUF                         Qwen3VL-2B-Instruct-Q4_K_M.gguf                                                 1107409952   large
mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf               Qwen/Qwen3-VL-2B-Instruct-GGUF                         mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf                                            445053216    large
Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf               Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF                    Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf                                            18556687168  large
mmproj-Qwen3VL-30B-A3B-Instruct-Q8_0.gguf          Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF                    mmproj-Qwen3VL-30B-A3B-Instruct-Q8_0.gguf                                       712148928    large
GLM-4.6V-Flash-Q4_K_M.gguf                         ggml-org/GLM-4.6V-Flash-GGUF                           GLM-4.6V-Flash-Q4_K_M.gguf                                                      6166577888   large
mmproj-GLM-4.6V-Flash-Q8_0.gguf                    ggml-org/GLM-4.6V-Flash-GGUF                           mmproj-GLM-4.6V-Flash-Q8_0.gguf                                                 980055680    large
GLM-4-9B-0414-Q4_K_M.gguf                          unsloth/GLM-4-9B-0414-GGUF                             GLM-4-9B-0414-Q4_K_M.gguf                                                       6166574944   large
Kimi-VL-A3B-Thinking-2506-Q4_K_M.gguf              ggml-org/Kimi-VL-A3B-Thinking-2506-GGUF                Kimi-VL-A3B-Thinking-2506-Q4_K_M.gguf                                           10540747680  large
mmproj-Kimi-VL-A3B-Thinking-2506-Q8_0.gguf         ggml-org/Kimi-VL-A3B-Thinking-2506-GGUF                mmproj-Kimi-VL-A3B-Thinking-2506-Q8_0.gguf                                      618098624    large
hunyuanvl/HunyuanOCR-Q8_0.gguf                     ggml-org/HunyuanOCR-GGUF                               HunyuanOCR-Q8_0.gguf                                                            577949408    default
hunyuanvl/mmproj-HunyuanOCR-Q8_0.gguf              ggml-org/HunyuanOCR-GGUF                               mmproj-HunyuanOCR-Q8_0.gguf                                                     732938240    large
hunyuanvl/hf/HunyuanOCR/config.json                tencent/HunyuanOCR                                     config.json                                                                     2233         default
hunyuanvl/hf/HunyuanOCR/generation_config.json     tencent/HunyuanOCR                                     generation_config.json                                                          139          default
hunyuanvl/hf/HunyuanOCR/tokenizer.json             tencent/HunyuanOCR                                     tokenizer.json                                                                  9527297      default
hunyuanvl/hf/HunyuanOCR/tokenizer_config.json      tencent/HunyuanOCR                                     tokenizer_config.json                                                           166592       default
hunyuanvl/hf/HunyuanOCR/special_tokens_map.json    tencent/HunyuanOCR                                     special_tokens_map.json                                                         836          default
hunyuanvl/hf/HunyuanOCR/chat_template.jinja        tencent/HunyuanOCR                                     chat_template.jinja                                                             994          default
hunyuanvl/hf/HunyuanOCR/preprocessor_config.json   tencent/HunyuanOCR                                     preprocessor_config.json                                                        579          default
hunyuanvl/hf/HunyuanOCR/model.safetensors          tencent/HunyuanOCR                                     model.safetensors                                                               2239932512   large
EOF
}

usage() {
	sed -n '2,/^set -eu/p' "$0" | sed -e '/^set -eu/d' -e 's/^# \{0,1\}//'
}

list=0 large=0 filters=
for a in "$@"; do
	case $a in
	--list | -l) list=1 ;;
	--large) large=1 ;;
	-h | --help) usage; exit 0 ;;
	-*) echo "fetch.sh: unknown flag $a (try --help)" >&2; exit 2 ;;
	*) filters="$filters $a" ;;
	esac
done

# size prints a file's length in bytes, following a symlink (wc -c answers
# from the inode on GNU and BSD alike).
size() { wc -c <"$1" | tr -d ' '; }

# select prints the chosen table rows. A filter is a case-insensitive
# substring of the name; a large row needs --large; a split GGUF set is taken
# whole when any of its parts is chosen.
select_rows() {
	table | awk -v filters="$filters" -v large="$large" '
	function key(n) { sub(/-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]\.gguf$/, "", n); return n }
	{
		row[NR] = $0; name[NR] = $1; set[NR] = $5
		ok = 1
		if (filters != "") {
			ok = 0; nf = split(filters, f, " ")
			for (i = 1; i <= nf; i++) if (index(tolower($1), tolower(f[i]))) { ok = 1; hit[f[i]] = 1; if ($5 == "large") bigonly[f[i]]++; else small[f[i]]++ }
		}
		if (ok && ($5 == "default" || large)) pick[key($1)] = 1
	}
	END {
		for (i = 1; i <= NR; i++) if (key(name[i]) in pick && (set[i] == "default" || large)) print row[i]
		if (filters != "") {
			nf = split(filters, f, " ")
			for (i = 1; i <= nf; i++) {
				if (!(f[i] in hit)) printf "fetch.sh: nothing in the table matches %s\n", f[i] > "/dev/stderr"
				else if (!large && !(f[i] in small)) printf "fetch.sh: %s matches only large files; add --large\n", f[i] > "/dev/stderr"
			}
		}
	}'
}

human() { awk -v b="$1" 'BEGIN { split("B KB MB GB TB", u, " "); i = 1; while (b >= 1000 && i < 5) { b /= 1000; i++ } printf(i == 1 ? "%d %s" : "%.1f %s", b, u[i]) }'; }

url() {
	case $1 in
	ollama:*) echo "https://registry.ollama.ai/v2/${1#ollama:}/blobs/$2" ;;
	*) echo "https://huggingface.co/$1/resolve/main/$2" ;;
	esac
}

# get downloads url into out, resuming a partial one. HF_TOKEN goes only to
# huggingface.co: curl does not carry a -H header across the redirect to the
# file's storage host.
get() {
	if [ -t 2 ]; then meter=--progress-bar; else meter=--silent; fi
	case $1 in
	https://huggingface.co/*)
		if [ -n "${HF_TOKEN:-}" ]; then
			curl -fL --show-error $meter --retry 5 --retry-delay 5 -C - \
				-H "Authorization: Bearer $HF_TOKEN" -o "$2" "$1"
			return
		fi ;;
	esac
	curl -fL --show-error $meter --retry 5 --retry-delay 5 -C - -o "$2" "$1"
}

fetch_one() { # name repo file bytes
	dst=$dir/$1
	if [ -e "$dst" ]; then
		have=$(size "$dst")
		if [ "$have" = "$4" ]; then
			echo "have   $1"
			return 0
		fi
		echo "SIZE   $1 is $have bytes, the table says $4; left alone -- move it aside to fetch it" >&2
		return 1
	fi
	mkdir -p "$(dirname "$dst")"

	# committed copy
	if [ -f "$root/testdata/models/$1" ] && [ "$(size "$root/testdata/models/$1")" = "$4" ]; then
		cp "$root/testdata/models/$1" "$dst"
		echo "copy   $1 (testdata/models)"
		return 0
	fi
	# alias: the same repo and file already here under another name
	for other in $(table | awk -v r="$2" -v f="$3" -v n="$1" '$2 == r && $3 == f && $1 != n { print $1 }'); do
		if [ -f "$dir/$other" ] && [ "$(size "$dir/$other")" = "$4" ]; then
			ln "$dir/$other" "$dst" 2>/dev/null || cp "$dir/$other" "$dst"
			echo "link   $1 -> $other"
			return 0
		fi
	done

	part=$dst.part
	if [ -f "$part" ] && [ "$(size "$part")" -gt "$4" ]; then
		rm -f "$part"
	fi
	if ! [ -f "$part" ] || [ "$(size "$part")" != "$4" ]; then
		echo "fetch  $1  ($(human "$4"), $2)"
		if ! get "$(url "$2" "$3")" "$part"; then
			echo "FAIL   $1: download failed; $part is kept and a rerun resumes it" >&2
			return 1
		fi
	fi
	got=$(size "$part")
	if [ "$got" != "$4" ]; then
		echo "FAIL   $1: got $got bytes, the table says $4" >&2
		[ "$got" -gt "$4" ] && rm -f "$part"
		return 1
	fi
	mv "$part" "$dst"
	echo "done   $1"
}

rows=$(select_rows)

if [ "$list" = 1 ]; then
	echo "model directory: $dir"
	printf '%s\n' "$rows" | while read -r name repo file bytes set; do
		[ -n "$name" ] || continue
		st=-
		[ -e "$dir/$name" ] && st=have
		[ "$st" = have ] && [ "$(size "$dir/$name")" != "$bytes" ] && st=SIZE
		printf '%-4s %9s  %-7s %-58s %s\n' "$st" "$(human "$bytes")" "$set" "$name" "$repo"
	done
	printf '%s\n' "$rows" | awk -v d="$dir" 'NF { n++; t += $4 } END { printf "%d files, %.1f GB\n", n, t / 1e9 }'
	exit 0
fi

[ -n "$rows" ] || exit 1
mkdir -p "$dir"
fail=0
# a here-document, not a pipe, so fail survives the loop
while read -r name repo file bytes set; do
	[ -n "$name" ] || continue
	fetch_one "$name" "$repo" "$file" "$bytes" || fail=$((fail + 1))
done <<EOF
$rows
EOF
if [ "$fail" -gt 0 ]; then
	echo "fetch.sh: $fail file(s) not fetched" >&2
	exit 1
fi

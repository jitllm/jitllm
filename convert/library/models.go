package library

// Models is the list, smallest first within each family. Sizes are the Hub's.
//
// Sized for desktop-class machines: 31 GiB of RAM with a 4 GB card, and a
// 16 GB arm64 host. The last few do not fit either in memory and run
// paged off the disk; their Note says so.
var Models = []Model{
	// --- llama (llama 2 and 3, SmolLM2 and Mistral are this architecture) ---
	{Name: "smollm2-360m-instruct", Title: "SmolLM2 360M Instruct", Arch: "llama", Params: "360M", Quant: "Q8_0",
		Repo: "bartowski/SmolLM2-360M-Instruct-GGUF", File: "SmolLM2-360M-Instruct-Q8_0.gguf", Bytes: 386405280,
		Note: "Tiny and fast; good for trying the app out."},
	{Name: "llama-3.2-1b-instruct", Title: "Llama 3.2 1B Instruct", Arch: "llama", Params: "1B", Quant: "Q4_K_M",
		Repo: "bartowski/Llama-3.2-1B-Instruct-GGUF", File: "Llama-3.2-1B-Instruct-Q4_K_M.gguf", Bytes: 807694464},
	{Name: "smollm2-1.7b-instruct", Title: "SmolLM2 1.7B Instruct", Arch: "llama", Params: "1.7B", Quant: "Q4_K_M",
		Repo: "bartowski/SmolLM2-1.7B-Instruct-GGUF", File: "SmolLM2-1.7B-Instruct-Q4_K_M.gguf", Bytes: 1055609824},
	{Name: "llama-3.2-3b-instruct", Title: "Llama 3.2 3B Instruct", Arch: "llama", Params: "3B", Quant: "Q4_K_M",
		Repo: "bartowski/Llama-3.2-3B-Instruct-GGUF", File: "Llama-3.2-3B-Instruct-Q4_K_M.gguf", Bytes: 2019377696},
	{Name: "mistral-7b-instruct-v0.3", Title: "Mistral 7B Instruct v0.3", Arch: "llama", Params: "7B", Quant: "Q4_K_M",
		Repo: "bartowski/Mistral-7B-Instruct-v0.3-GGUF", File: "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", Bytes: 4372812000},
	{Name: "llama-3.1-8b-instruct", Title: "Llama 3.1 8B Instruct", Arch: "llama", Params: "8B", Quant: "Q4_K_M",
		Repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", File: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", Bytes: 4920739232},
	{Name: "llama-3.3-70b-instruct", Title: "Llama 3.3 70B Instruct", Arch: "llama", Params: "70B", Quant: "Q4_K_M",
		Repo: "bartowski/Llama-3.3-70B-Instruct-GGUF", File: "Llama-3.3-70B-Instruct-Q4_K_M.gguf", Bytes: 42520398816,
		Note: "Larger than memory on these machines: runs paged off the disk, slowly."},

	// --- qwen2 / qwen2vl ---
	{Name: "qwen2.5-1.5b-instruct", Title: "Qwen2.5 1.5B Instruct", Arch: "qwen2", Params: "1.5B", Quant: "Q4_K_M",
		Repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF", File: "qwen2.5-1.5b-instruct-q4_k_m.gguf", Bytes: 1117320736},
	{Name: "qwen2.5-7b-instruct", Title: "Qwen2.5 7B Instruct", Arch: "qwen2", Params: "7B", Quant: "Q4_K_M",
		Repo: "bartowski/Qwen2.5-7B-Instruct-GGUF", File: "Qwen2.5-7B-Instruct-Q4_K_M.gguf", Bytes: 4683074240},
	{Name: "qwen2-vl-2b-instruct", Title: "Qwen2-VL 2B Instruct", Arch: "qwen2vl", Params: "2B", Quant: "Q4_K_M",
		Repo: "ggml-org/Qwen2-VL-2B-Instruct-GGUF", File: "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
		MMProj: "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf",
		Bytes:  986046944 + 709883360, Note: "Sees images."},

	// --- qwen3 / qwen3moe / qwen3next ---
	{Name: "qwen3-0.6b", Title: "Qwen3 0.6B", Arch: "qwen3", Params: "0.6B", Quant: "Q8_0",
		Repo: "Qwen/Qwen3-0.6B-GGUF", File: "Qwen3-0.6B-Q8_0.gguf", Bytes: 639446688, Note: "Reasons before it answers."},
	{Name: "qwen3-1.7b", Title: "Qwen3 1.7B", Arch: "qwen3", Params: "1.7B", Quant: "Q8_0",
		Repo: "Qwen/Qwen3-1.7B-GGUF", File: "Qwen3-1.7B-Q8_0.gguf", Bytes: 1834426016, Note: "Reasons before it answers."},
	{Name: "qwen3-4b", Title: "Qwen3 4B", Arch: "qwen3", Params: "4B", Quant: "Q4_K_M",
		Repo: "Qwen/Qwen3-4B-GGUF", File: "Qwen3-4B-Q4_K_M.gguf", Bytes: 2497280256, Note: "Reasons before it answers."},
	{Name: "qwen3-8b", Title: "Qwen3 8B", Arch: "qwen3", Params: "8B", Quant: "Q4_K_M",
		Repo: "Qwen/Qwen3-8B-GGUF", File: "Qwen3-8B-Q4_K_M.gguf", Bytes: 5027783488, Note: "Reasons before it answers."},
	{Name: "qwen3-14b", Title: "Qwen3 14B", Arch: "qwen3", Params: "14B", Quant: "Q4_K_M",
		Repo: "Qwen/Qwen3-14B-GGUF", File: "Qwen3-14B-Q4_K_M.gguf", Bytes: 9001752960, Note: "Reasons before it answers."},
	{Name: "qwen3-30b-a3b", Title: "Qwen3 30B-A3B", Arch: "qwen3moe", Params: "30B-A3B", Quant: "Q4_K_M",
		Repo: "Qwen/Qwen3-30B-A3B-GGUF", File: "Qwen3-30B-A3B-Q4_K_M.gguf", Bytes: 18556685824,
		Note: "A mixture of experts: 30B of weights, 3B used per token, so it is fast for its size. Needs about 20 GB of memory."},
	{Name: "qwen3-next-80b-a3b-instruct", Title: "Qwen3-Next 80B-A3B Instruct", Arch: "qwen3next", Params: "80B-A3B", Quant: "Q4_K_M",
		Repo: "unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF", File: "Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf", Bytes: 48506100512,
		Note: "Larger than memory on these machines: runs paged off the disk, slowly."},
	// Qwen3.5 is qwen3next's hybrid block with a dense feed-forward; the 4B has
	// two value heads per key head, which is the case the tiled pairing is for.
	{Name: "qwen3.5-2b", Title: "Qwen3.5 2B", Arch: "qwen35", Params: "2B", Quant: "Q4_K_M",
		Repo: "unsloth/Qwen3.5-2B-GGUF", File: "Qwen3.5-2B-Q4_K_M.gguf", Bytes: 1280835840,
		Note: "A hybrid: three of every four layers keep a fixed-size recurrent state instead of a KV cache."},
	{Name: "qwen3.5-4b", Title: "Qwen3.5 4B", Arch: "qwen35", Params: "4B", Quant: "Q4_K_M",
		Repo: "unsloth/Qwen3.5-4B-GGUF", File: "Qwen3.5-4B-Q4_K_M.gguf", Bytes: 2740937888,
		Note: "A hybrid: three of every four layers keep a fixed-size recurrent state instead of a KV cache."},
	{Name: "qwen3.6-35b-a3b", Title: "Qwen3.6 35B-A3B", Arch: "qwen35moe", Params: "35B-A3B", Quant: "Q4_K_M",
		Repo: "unsloth/Qwen3.6-35B-A3B-GGUF", File: "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", Bytes: 22134528992,
		Note: "A hybrid mixture of experts: 3B of 35B used per token. Needs about 24 GB of memory."},

	// --- phi2 ---
	{Name: "phi-2", Title: "Phi-2", Arch: "phi2", Params: "2.7B", Quant: "Q4_K_M",
		Repo: "TheBloke/phi-2-GGUF", File: "phi-2.Q4_K_M.gguf", Bytes: 1789239136,
		Note: "A base model, not an instruct one: it completes text and code."},

	// --- starcoder ---
	{Name: "starcoderbase-1b", Title: "StarCoderBase 1B", Arch: "starcoder", Params: "1B", Quant: "Q4_K_M",
		Repo: "mradermacher/starcoderbase-1b-GGUF", File: "starcoderbase-1b.Q4_K_M.gguf", Bytes: 793436000,
		Note: "A code completion model, not a chat one."},

	// --- command-r, cohere2 ---
	{Name: "aya-expanse-8b", Title: "Aya Expanse 8B", Arch: "command-r", Params: "8B", Quant: "Q4_K_M",
		Repo: "bartowski/aya-expanse-8b-GGUF", File: "aya-expanse-8b-Q4_K_M.gguf", Bytes: 5056982720},
	{Name: "command-r7b", Title: "Command R7B", Arch: "cohere2", Params: "7B", Quant: "Q4_K_M",
		Repo: "bartowski/c4ai-command-r7b-12-2024-GGUF", File: "c4ai-command-r7b-12-2024-Q4_K_M.gguf", Bytes: 5057009792},

	// --- starcoder2 ---
	{Name: "starcoder2-3b", Title: "StarCoder2 3B", Arch: "starcoder2", Params: "3B", Quant: "Q4_K_M",
		Repo: "mradermacher/starcoder2-3b-GGUF", File: "starcoder2-3b.Q4_K_M.gguf", Bytes: 1887905664,
		Note: "A code completion model, not a chat one."},

	// --- falcon ---
	// Q4_K_M, whose attn_qkv is Q5_1 (Falcon-7B's 4544-wide rows are not a
	// multiple of 256, so every K-quant of it carries Q5_1 there): verified 11
	// of 11 teacher-forced against llama.cpp.
	{Name: "falcon-7b-instruct", Title: "Falcon 7B Instruct", Arch: "falcon", Params: "7B", Quant: "Q4_K_M",
		Repo: "mradermacher/falcon-7b-instruct-GGUF", File: "falcon-7b-instruct.Q4_K_M.gguf", Bytes: 4975127552},

	// --- qwen2moe ---
	// 19 of 20 teacher-forced against llama.cpp (a 0.304 tie at token 5).
	{Name: "qwen1.5-moe-a2.7b-chat", Title: "Qwen1.5-MoE A2.7B Chat", Arch: "qwen2moe", Params: "14.3B-A2.7B",
		Quant: "Q4_K_M", Repo: "duyntnet/Qwen1.5-MoE-A2.7B-Chat-imatrix-GGUF",
		File: "Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf", Bytes: 9496238048},

	// --- ernie4_5-moe ---
	// The base model: 18 of 20 teacher-forced against llama.cpp (a 0.198 tie
	// at token 2).
	{Name: "ernie-4.5-21b-a3b", Title: "ERNIE 4.5 21B-A3B (base)", Arch: "ernie4_5-moe", Params: "21B-A3B",
		Quant: "Q4_K_M", Repo: "unsloth/ERNIE-4.5-21B-A3B-PT-GGUF",
		File: "ERNIE-4.5-21B-A3B-PT-Q4_K_M.gguf", Bytes: 13331015456},

	// --- ernie4_5 ---
	// The base (pre-trained) model: 20 of 20 teacher-forced against llama.cpp.
	{Name: "ernie-4.5-0.3b", Title: "ERNIE 4.5 0.3B (base)", Arch: "ernie4_5", Params: "0.3B", Quant: "Q4_K_M",
		Repo: "unsloth/ERNIE-4.5-0.3B-PT-GGUF", File: "ERNIE-4.5-0.3B-PT-Q4_K_M.gguf", Bytes: 240645600},

	// --- hunyuan-dense ---
	// 20 of 20 teacher-forced against llama.cpp.
	{Name: "hunyuan-0.5b-instruct", Title: "Hunyuan 0.5B Instruct", Arch: "hunyuan-dense", Params: "0.5B",
		Quant: "Q4_K_M", Repo: "bartowski/tencent_Hunyuan-0.5B-Instruct-GGUF",
		File: "tencent_Hunyuan-0.5B-Instruct-Q4_K_M.gguf", Bytes: 354970464},

	// --- bailingmoe2 ---
	// 20 of 20 teacher-forced against llama.cpp.
	{Name: "ling-mini-2.0", Title: "Ling mini 2.0", Arch: "bailingmoe2", Params: "16B-A1.4B",
		Quant: "Q4_K_M", Repo: "bartowski/inclusionAI_Ling-mini-2.0-GGUF",
		File: "inclusionAI_Ling-mini-2.0-Q4_K_M.gguf", Bytes: 9941894336},

	// --- phimoe ---
	// llama.cpp runs another graph for this architecture (RULE 7m): 33 of 33
	// teacher-forced against transformers' PhimoeForCausalLM on this file's
	// own weights, host and CUDA.
	{Name: "phi-mini-moe-instruct", Title: "Phi mini MoE Instruct", Arch: "phimoe", Params: "7.6B-A2.4B",
		Quant: "Q4_K_M", Repo: "gabriellarson/Phi-mini-MoE-instruct-GGUF",
		File: "Phi-mini-MoE-instruct-Q4_K_M.gguf", Bytes: 4993133376},

	// --- apertus ---
	// 19 of 20 teacher-forced against llama.cpp, the other a tie.
	{Name: "apertus-8b-instruct", Title: "Apertus 8B Instruct", Arch: "apertus", Params: "8B", Quant: "Q4_K_M",
		Repo: "bartowski/swiss-ai_Apertus-8B-Instruct-2509-GGUF",
		File: "swiss-ai_Apertus-8B-Instruct-2509-Q4_K_M.gguf", Bytes: 5057885568},

	// --- bitnet-b1.58 ---
	// Microsoft's own conversion; 212 of 214 teacher-forced positions take
	// transformers' token on the release's golden, the other two ties.
	{Name: "bitnet-b1.58-2b-4t", Title: "BitNet b1.58 2B-4T", Arch: "bitnet-b1.58", Params: "2B", Quant: "I2_S",
		Repo: "microsoft/bitnet-b1.58-2B-4T-gguf", File: "ggml-model-i2_s.gguf", Bytes: 1187801280,
		Note: "Ternary weights: every projection is -1, 0 or 1 times one scale."},

	// --- the state-space hybrids and the short convolutions ---
	// Each teacher-forced against llama.cpp (TestSSMRealModelsTeacherForced).
	{Name: "granite-4.0-h-350m", Title: "Granite 4.0 H 350M", Arch: "granitehybrid", Params: "350M",
		Quant: "Q8_0", Repo: "ibm-granite/granite-4.0-h-350m-GGUF", File: "granite-4.0-h-350m-Q8_0.gguf",
		Bytes: 366195616},
	{Name: "nemotron-nano-9b-v2", Title: "NVIDIA Nemotron Nano 9B v2", Arch: "nemotron_h", Params: "9B",
		Quant: "Q4_K_M", Repo: "bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF",
		File: "nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf", Bytes: 6525629280},
	{Name: "nemotron-3-nano-30b-a3b", Title: "NVIDIA Nemotron 3 Nano 30B-A3B", Arch: "nemotron_h_moe",
		Params: "30B-A3B", Quant: "Q4_K_M", Repo: "bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF",
		File: "nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf", Bytes: 24655899872},
	{Name: "falcon-h1-0.5b-instruct", Title: "Falcon-H1 0.5B Instruct", Arch: "falcon-h1", Params: "0.5B",
		Quant: "Q4_K_M", Repo: "tiiuae/Falcon-H1-0.5B-Instruct-GGUF", File: "Falcon-H1-0.5B-Instruct-Q4_K_M.gguf",
		Bytes: 314806560},
	{Name: "lfm2-350m", Title: "LFM2 350M", Arch: "lfm2", Params: "350M", Quant: "Q4_K_M",
		Repo: "LiquidAI/LFM2-350M-GGUF", File: "LFM2-350M-Q4_K_M.gguf", Bytes: 229309376},
	{Name: "lfm2-8b-a1b", Title: "LFM2 8B-A1B", Arch: "lfm2moe", Params: "8B-A1B", Quant: "Q4_K_M",
		Repo: "LiquidAI/LFM2-8B-A1B-GGUF", File: "LFM2-8B-A1B-Q4_K_M.gguf", Bytes: 5044779712},
	{Name: "mamba-codestral-7b", Title: "Mamba Codestral 7B", Arch: "mamba2", Params: "7B", Quant: "Q4_K_M",
		Repo: "viniciusianni/Mamba-Codestral-7B-v0.1-Q4_K_M-GGUF", File: "mamba-codestral-7b-v0.1-q4_k_m.gguf",
		Bytes: 4147481568},
	{Name: "falcon-mamba-7b", Title: "Falcon Mamba 7B", Arch: "mamba", Params: "7B", Quant: "Q4_K_M",
		Repo: "tiiuae/falcon-mamba-7b-Q4_K_M-GGUF", File: "falcon-mamba-7B-Q4_K_M.gguf", Bytes: 4204230432},
	{Name: "jamba-reasoning-3b", Title: "AI21 Jamba Reasoning 3B", Arch: "jamba", Params: "3B", Quant: "F16",
		Repo: "ai21labs/AI21-Jamba-Reasoning-3B-GGUF", File: "jamba-reasoning-3b-F16.gguf", Bytes: 6402230208,
		Note: "F16: the quantized files part from every reference on a long story, llama.cpp included."},

	// --- nemotron ---
	{Name: "nemotron-mini-4b", Title: "Nemotron-Mini 4B Instruct", Arch: "nemotron", Params: "4B", Quant: "Q4_K_M",
		Repo: "bartowski/Nemotron-Mini-4B-Instruct-GGUF", File: "Nemotron-Mini-4B-Instruct-Q4_K_M.gguf", Bytes: 2697387072},

	// --- the dense llama family: smollm3, arcee, seed_oss, olmo2, exaone4, mistral3 ---
	{Name: "smollm3-3b", Title: "SmolLM3 3B", Arch: "smollm3", Params: "3B", Quant: "Q4_K_M",
		Repo: "unsloth/SmolLM3-3B-GGUF", File: "SmolLM3-3B-Q4_K_M.gguf", Bytes: 1915306528},
	{Name: "afm-4.5b", Title: "Arcee AFM 4.5B", Arch: "arcee", Params: "4.5B", Quant: "Q4_K_M",
		Repo: "arcee-ai/AFM-4.5B-GGUF", File: "AFM-4.5B-Q4_K_M.gguf", Bytes: 2916301824},
	{Name: "seed-oss-36b", Title: "Seed-OSS 36B Instruct", Arch: "seed_oss", Params: "36B", Quant: "Q4_K_M",
		Repo: "bartowski/ByteDance-Seed_Seed-OSS-36B-Instruct-GGUF",
		File: "ByteDance-Seed_Seed-OSS-36B-Instruct-Q4_K_M.gguf", Bytes: 21762149024},
	{Name: "olmo-2-1b", Title: "OLMo 2 1B Instruct", Arch: "olmo2", Params: "1B", Quant: "Q4_K_M",
		Repo: "allenai/OLMo-2-0425-1B-Instruct-GGUF", File: "OLMo-2-0425-1B-Instruct-Q4_K_M.gguf", Bytes: 935515296},
	{Name: "exaone-4.0-1.2b", Title: "EXAONE 4.0 1.2B", Arch: "exaone4", Params: "1.2B", Quant: "Q4_K_M",
		Repo: "LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF", File: "EXAONE-4.0-1.2B-Q4_K_M.gguf", Bytes: 812437792},
	{Name: "ministral-3-3b", Title: "Ministral 3 3B Instruct", Arch: "mistral3", Params: "3B", Quant: "Q4_K_M",
		Repo: "mistralai/Ministral-3-3B-Instruct-2512-GGUF", File: "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
		Bytes: 2147023008, Note: "Text only: the vision tower ships as a separate mmproj."},

	// --- stablelm ---
	{Name: "stablelm-2-1.6b", Title: "StableLM 2 1.6B", Arch: "stablelm", Params: "1.6B", Quant: "Q4_K_M",
		Repo: "mradermacher/stablelm-2-1_6b-GGUF", File: "stablelm-2-1_6b.Q4_K_M.gguf", Bytes: 1031442496,
		Note: "A base model, not an instruct one."},

	// --- granite, granitemoe ---
	{Name: "granite-3.3-2b", Title: "Granite 3.3 2B Instruct", Arch: "granite", Params: "2B", Quant: "Q4_K_M",
		Repo: "ibm-granite/granite-3.3-2b-instruct-GGUF", File: "granite-3.3-2b-instruct-Q4_K_M.gguf", Bytes: 1545303328},
	{Name: "granite-4.0-micro", Title: "Granite 4.0 Micro", Arch: "granite", Params: "3B", Quant: "Q4_K_M",
		Repo: "ibm-granite/granite-4.0-micro-GGUF", File: "granite-4.0-micro-Q4_K_M.gguf", Bytes: 2099502528},
	{Name: "granite-3.1-1b-a400m", Title: "Granite 3.1 1B-A400M Instruct", Arch: "granitemoe", Params: "1B-A400M", Quant: "Q4_K_M",
		Repo: "bartowski/granite-3.1-1b-a400m-instruct-GGUF", File: "granite-3.1-1b-a400m-instruct-Q4_K_M.gguf", Bytes: 821847360,
		Note: "A mixture of experts: 400M of 1B used per token."},

	// --- deepseek2 ---
	//
	// One architecture, several product lines (DeepSeek-V2/V3/R1, Kimi-K2,
	// Moonlight, GLM-4.7-Flash). Both shapes are listed on purpose: V2-Lite is
	// the V2 shape (no q_lora_rank, greedy top-k), and Moonlight declares
	// DeepseekV3ForCausalLM (sigmoid router with selection bias, grouped
	// top-k, routed scale), the Kimi shape at a size this hardware can hold.
	{Name: "deepseek-v2-lite", Title: "DeepSeek-V2-Lite", Arch: "deepseek2", Params: "16B-A2.4B", Quant: "Q4_K_M",
		Repo: "mradermacher/DeepSeek-V2-Lite-GGUF", File: "DeepSeek-V2-Lite.Q4_K_M.gguf", Bytes: 10364416736,
		Note: "Multi-head latent attention: the KV cache is one row per position for the whole layer, so it is small for its size."},
	{Name: "deepseek-coder-v2-lite-instruct", Title: "DeepSeek-Coder-V2-Lite Instruct", Arch: "deepseek2", Params: "16B-A2.4B", Quant: "Q4_K_M",
		Repo: "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF", File: "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf", Bytes: 10364416768,
		Note: "The same architecture, trained for code."},
	{Name: "moonlight-16b-a3b-instruct", Title: "Moonlight 16B-A3B Instruct", Arch: "deepseek2", Params: "16B-A3B", Quant: "Q4_K_M",
		Repo: "gabriellarson/Moonlight-16B-A3B-Instruct-GGUF", File: "Moonlight-16B-A3B-Instruct-Q4_K_M.gguf", Bytes: 10540747360,
		Note: "Moonshot AI's own model, and the DeepSeek-V3 graph Kimi is built on, at a size that fits."},

	// --- olmoe ---
	{Name: "olmoe-1b-7b-instruct", Title: "OLMoE 1B-7B Instruct", Arch: "olmoe", Params: "7B-A1B", Quant: "Q4_K_M",
		Repo: "allenai/OLMoE-1B-7B-0924-Instruct-GGUF", File: "olmoe-1b-7b-0924-instruct-q4_k_m.gguf", Bytes: 4213512672,
		Note: "A mixture of experts: 7B of weights, 1B used per token."},

	// --- gemma2 / gemma3 ---
	{Name: "gemma-2-2b-it", Title: "Gemma 2 2B Instruct", Arch: "gemma2", Params: "2B", Quant: "Q4_K_M",
		Repo: "bartowski/gemma-2-2b-it-GGUF", File: "gemma-2-2b-it-Q4_K_M.gguf", Bytes: 1708582752,
		Note: "Runs on the CPU: the GPU does not run Gemma 2 yet."},
	{Name: "gemma-2-9b-it", Title: "Gemma 2 9B Instruct", Arch: "gemma2", Params: "9B", Quant: "Q4_K_M",
		Repo: "bartowski/gemma-2-9b-it-GGUF", File: "gemma-2-9b-it-Q4_K_M.gguf", Bytes: 5761057728,
		Note: "Runs on the CPU: the GPU does not run Gemma 2 yet."},
	{Name: "gemma-3-1b-it", Title: "Gemma 3 1B Instruct", Arch: "gemma3", Params: "1B", Quant: "Q4_K_M",
		Repo: "bartowski/google_gemma-3-1b-it-GGUF", File: "google_gemma-3-1b-it-Q4_K_M.gguf", Bytes: 806058496},
	{Name: "gemma-3-4b-it", Title: "Gemma 3 4B Instruct", Arch: "gemma3", Params: "4B", Quant: "Q4_K_M",
		Repo: "bartowski/google_gemma-3-4b-it-GGUF", File: "google_gemma-3-4b-it-Q4_K_M.gguf", Bytes: 2489758112,
		Note: "Text only here: its vision tower is not supported yet."},
	{Name: "gemma-3-12b-it", Title: "Gemma 3 12B Instruct", Arch: "gemma3", Params: "12B", Quant: "Q4_K_M",
		Repo: "bartowski/google_gemma-3-12b-it-GGUF", File: "google_gemma-3-12b-it-Q4_K_M.gguf", Bytes: 7300575264,
		Note: "Text only here: its vision tower is not supported yet."},
	{Name: "gemma-3n-E2B-it", Title: "Gemma 3n E2B Instruct", Arch: "gemma3n", Params: "5B (E2B)", Quant: "Q4_K_M",
		Repo: "unsloth/gemma-3n-E2B-it-GGUF", File: "gemma-3n-E2B-it-Q4_K_M.gguf", Bytes: 3026881888,
		Note: "Text only here: its vision and audio encoders are not supported yet."},

	// --- phi3 (Phi-3, Phi-3.5 and Phi-4 are this architecture) ---
	{Name: "phi-3.5-mini-instruct", Title: "Phi-3.5 mini Instruct", Arch: "phi3", Params: "3.8B", Quant: "Q4_K_M",
		Repo: "bartowski/Phi-3.5-mini-instruct-GGUF", File: "Phi-3.5-mini-instruct-Q4_K_M.gguf", Bytes: 2393232672},
	{Name: "phi-4-mini-instruct", Title: "Phi-4 mini Instruct", Arch: "phi3", Params: "3.8B", Quant: "Q4_K_M",
		Repo: "bartowski/microsoft_Phi-4-mini-instruct-GGUF", File: "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf", Bytes: 2491874688},
	{Name: "phi-4", Title: "Phi-4", Arch: "phi3", Params: "14B", Quant: "Q4_K_M",
		Repo: "bartowski/phi-4-GGUF", File: "phi-4-Q4_K_M.gguf", Bytes: 9053114816},

	// --- llama4 ---
	//
	// Q3_K_S, not Q4_K_M, because the catalogue downloads one file: every
	// Scout quant from Q3_K_M up is split by the Hub's 50 GB cap. `jitllm
	// convert` takes a split GGUF by its first part.
	{Name: "llama-4-scout-17b-16e-instruct", Title: "Llama 4 Scout 17B-16E Instruct", Arch: "llama4",
		Params: "109B-A17B", Quant: "Q3_K_S",
		Repo: "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF", File: "Llama-4-Scout-17B-16E-Instruct-Q3_K_S.gguf",
		Bytes:    46740372480,
		Note:     "A mixture of experts, one of 16 per token, text only here. Larger than memory on these machines: runs paged off the disk, slowly.",
		Verified: "2026-09-27 linux/amd64 jlm v26"},

	// --- gpt-oss ---
	//
	// A mixture whose expert banks are MXFP4 whatever the file is called, so
	// these are the ggml-org conversions rather than a re-quantized copy: the
	// experts are the format either way and a "Q4_K_M" build of this
	// architecture differs only in the dense tensors.
	{Name: "gpt-oss-20b", Title: "gpt-oss 20B", Arch: "gpt-oss", Params: "21B-A3.6B", Quant: "MXFP4",
		Repo: "ggml-org/gpt-oss-20b-GGUF", File: "gpt-oss-20b-MXFP4.gguf", Bytes: 12109566624,
		Note: "Mixture of experts: 21B stored, ~3.6B active per token."},
	{Name: "gpt-oss-120b", Title: "gpt-oss 120B", Arch: "gpt-oss", Params: "117B-A5.1B", Quant: "MXFP4",
		Repo: "ggml-org/gpt-oss-120b-GGUF", File: "gpt-oss-120b-MXFP4.gguf", Bytes: 63387346208,
		Note: "Larger than memory on these machines: runs paged off the disk, slowly."},

	// --- embeddings: a vector per text, through `jitllm embed`, not `run` ---
	//
	// The f16 files are byte-for-byte the size of the Ollama registry's blobs
	// for all-minilm and mxbai-embed-large, and within 96 bytes for
	// nomic-embed-text (its header), so they are the models Ollama serves.
	{Name: "all-minilm-l6-v2", Title: "all-MiniLM-L6-v2 (embeddings)", Arch: "bert", Params: "22M", Quant: "F16",
		Repo: "second-state/All-MiniLM-L6-v2-Embedding-GGUF", File: "all-MiniLM-L6-v2-ggml-model-f16.gguf",
		Bytes: 45949216, Note: "A sentence embedding model: `jitllm embed`, 384 dims."},
	{Name: "nomic-embed-text-v1.5", Title: "nomic-embed-text v1.5 (embeddings)", Arch: "nomic-bert", Params: "137M",
		Quant: "F16", Repo: "nomic-ai/nomic-embed-text-v1.5-GGUF", File: "nomic-embed-text-v1.5.f16.gguf",
		Bytes: 274290560, Note: "A text embedding model: `jitllm embed`, 768 dims. Prefix search_query: / search_document:."},
	{Name: "mxbai-embed-large-v1", Title: "mxbai-embed-large v1 (embeddings)", Arch: "bert", Params: "335M",
		Quant: "F16", Repo: "ChristianAzinn/mxbai-embed-large-v1-gguf", File: "mxbai-embed-large-v1_fp16.gguf",
		Bytes: 669603712, Note: "A text embedding model: `jitllm embed`, 1024 dims."},
	// --- decision models: typed answers through `jitllm decide` and
	// /v1/systemone, not `run` (docs/design/decision-models.md) ---
	{Name: "laya", Title: "Laya (decision model)", Arch: "modern-bert", Params: "421M", Quant: "BF16",
		Repo: "ggml-org/Laya-GGUF", File: "Laya-BF16.gguf", Bytes: 844026720,
		Note: "A decision model: `jitllm decide` or POST /v1/systemone, English."},
	{Name: "d1-3b", Title: "d1-3B (decision model)", Arch: "lfm2", Params: "3B", Quant: "Q8_0",
		Repo: "LiquidAI/d1-3B-GGUF", File: "d1-3B-Q8_0.gguf", Bytes: 2874781280,
		Note: "A decision model: `jitllm decide` or POST /v1/systemone; text states only."},
	{Name: "lev", Title: "Lev 4B (decision model)", Arch: "qwen35", Params: "4B", Quant: "Q8_0",
		Repo: "ggml-org/lev-GGUF", File: "lev-Q8_0.gguf", Bytes: 4482405280,
		Note: "A decision model: `jitllm decide` or POST /v1/systemone; the adapter merged into Qwen3.5-4B."},
	{Name: "clef-flash", Title: "Clef-Flash (decision model)", Arch: "clef", Params: "9.1B", Quant: "Q8_0",
		Repo: "ggml-org/Clef-Flash-GGUF", File: "Clef-Flash-Q8_0.gguf", Bytes: 9657260192,
		Note: "A decision model: `jitllm decide` or POST /v1/systemone; every question answered jointly in one pass, text states only."},
	{Name: "qwen3-embedding-0.6b", Title: "Qwen3-Embedding 0.6B (embeddings)", Arch: "qwen3", Params: "0.6B",
		Quant: "F16", Repo: "Qwen/Qwen3-Embedding-0.6B-GGUF", File: "Qwen3-Embedding-0.6B-f16.gguf",
		Bytes: 1197629632, Note: "A text embedding model: `jitllm embed`, 1024 dims, last-token pooled."},

	// --- vision: a text model and its tower in one container ---
	{Name: "smolvlm-256m-instruct", Title: "SmolVLM 256M Instruct", Arch: "llama", Params: "256M", Quant: "Q8_0",
		Repo: "ggml-org/SmolVLM-256M-Instruct-GGUF", File: "SmolVLM-256M-Instruct-Q8_0.gguf",
		MMProj: "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf", Bytes: 175054528 + 103769856,
		Note: "Sees images; tiny."},
	{Name: "smolvlm-500m-instruct", Title: "SmolVLM 500M Instruct", Arch: "llama", Params: "500M", Quant: "Q8_0",
		Repo: "ggml-org/SmolVLM-500M-Instruct-GGUF", File: "SmolVLM-500M-Instruct-Q8_0.gguf",
		MMProj: "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf", Bytes: 436806912 + 108783360,
		Note: "Sees images."},
	{Name: "llava-phi-3-mini", Title: "LLaVA Phi-3 mini", Arch: "llama", Params: "3.8B", Quant: "int4",
		Repo: "xtuner/llava-phi-3-mini-gguf", File: "llava-phi-3-mini-int4.gguf",
		MMProj: "llava-phi-3-mini-mmproj-f16.gguf", Bytes: 2318919200 + 607648960,
		Note: "Sees images."},
	// Qwen3-VL's text models are qwen3 and qwen3moe under an interleaved
	// M-RoPE (qwen3vl, qwen3vlmoe); GLM-4.6V-Flash's is GLM-4-0414's block
	// (glm4) under an M-RoPE; Kimi-VL's is DeepSeek-V3's (deepseek2).
	{Name: "qwen3-vl-2b-instruct", Title: "Qwen3-VL 2B Instruct", Arch: "qwen3vl", Params: "2B", Quant: "Q4_K_M",
		Repo: "Qwen/Qwen3-VL-2B-Instruct-GGUF", File: "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
		MMProj: "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf", Bytes: 1107409952 + 445053216,
		Note: "Sees images."},
	{Name: "qwen3-vl-30b-a3b-instruct", Title: "Qwen3-VL 30B-A3B Instruct", Arch: "qwen3vlmoe", Params: "30B-A3B",
		Quant: "Q4_K_M", Repo: "Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF", File: "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf",
		MMProj: "mmproj-Qwen3VL-30B-A3B-Instruct-Q8_0.gguf", Bytes: 18556687168 + 712148928,
		Note: "Sees images; a mixture of experts, 3B of 30B used per token."},
	{Name: "glm-4.6v-flash", Title: "GLM-4.6V Flash", Arch: "glm4", Params: "9B", Quant: "Q4_K_M",
		Repo: "ggml-org/GLM-4.6V-Flash-GGUF", File: "GLM-4.6V-Flash-Q4_K_M.gguf",
		MMProj: "mmproj-GLM-4.6V-Flash-Q8_0.gguf", Bytes: 6166577888 + 980055680,
		Note: "Sees images."},
	{Name: "glm-4-9b-0414", Title: "GLM-4 9B 0414", Arch: "glm4", Params: "9B", Quant: "Q4_K_M",
		Repo: "unsloth/GLM-4-9B-0414-GGUF", File: "GLM-4-9B-0414-Q4_K_M.gguf", Bytes: 6166574944},
	{Name: "kimi-vl-a3b-thinking-2506", Title: "Kimi-VL A3B Thinking 2506", Arch: "deepseek2", Params: "16B-A3B",
		Quant: "Q4_K_M", Repo: "ggml-org/Kimi-VL-A3B-Thinking-2506-GGUF", File: "Kimi-VL-A3B-Thinking-2506-Q4_K_M.gguf",
		MMProj: "mmproj-Kimi-VL-A3B-Thinking-2506-Q8_0.gguf", Bytes: 10540747680 + 618098624,
		Note: "Sees images; reasons before it answers."},
	{Name: "hunyuanocr", Title: "HunyuanOCR", Arch: "hunyuan_vl", Params: "1B", Quant: "Q8_0",
		Repo: "ggml-org/HunyuanOCR-GGUF", File: "HunyuanOCR-Q8_0.gguf", MMProj: "mmproj-HunyuanOCR-Q8_0.gguf",
		Bytes: 577949408 + 732938240, Note: "Reads text out of documents and pictures."},
}

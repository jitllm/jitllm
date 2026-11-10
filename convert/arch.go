package convert

import "github.com/jitllm/jitllm/format/jlm"

// Config is a model's config as the converter builds it: the container's
// (jlm.Config) and the architecture its source declares, which the converter
// alone reads -- the container states the graph, and nothing past the
// converter names an architecture.
type Config struct {
	*jlm.Config
	Arch Arch
}

// The architectures the converter reads, its names for the graphs source files declare.
// Nothing past the converter names one: the container states the graph.

// ArchBitNet is Microsoft's BitNet b1.58 (BitNetForCausalLM; llama.cpp's
// "bitnet", bitnet.cpp's "bitnet-b1.58"): llama's block with ternary
// projections (TypeTQ2), every projection input quantized to int8 per token
// by its absolute maximum (the engine's activation window is the whole row),
// an RMSNorm over the attention output before wo (RoleAttnSubNorm) and over
// act(gate)*up before down (RoleFFNSubNorm), and a squared-ReLU gate. The code
// carries the per-token quantization and the sub-norms' places, which no file
// states. A dense-group code, from the top down (archdense.go).
const ArchBitNet Arch = 56

// The dense llama-family architectures added as one group. Their codes are
// taken from the top of the 6-bit range down, so a concurrent addition counting
// up from ArchDBRX does not collide with them; Arch.Valid asks the name tables
// rather than a maximum, so the gap between the two ends is not a code.
//
// Each is a code of its own so a reader that predates it refuses the container
// rather than running it as llama's block with a feature missing -- every one
// of those omissions is fluent.
const (
	// ArchSmolLM3 is HuggingFace's SmolLM3: llama's block with no rotary on
	// every SWAPeriod-th layer, (il+1) % SWAPeriod == 0 (FlagNoPEGlobal with
	// SWAWindow 0: the period alone names the NoPE layers, and no layer slides).
	// llama.cpp hardcodes the period at 4 (n_no_rope_layer_step).
	ArchSmolLM3 Arch = 63

	// ArchArcee is Arcee's AFM-4.5B: llama's block with an ungated squared-ReLU
	// FFN (up, relu^2, down; FlagReLU2 and no gate tensor).
	ArchArcee Arch = 62

	// ArchSeedOSS is ByteDance's Seed-OSS: llama's block with biased q, k and v
	// (tensors) and NEOX rotary. Its post_attention_norm is the FFN's pre-norm,
	// which the converter writes as RoleFFNNorm.
	ArchSeedOSS Arch = 61

	// ArchOLMo2 is AI2's OLMo 2 and OLMo 3: no pre-norms at all. q and k are
	// RMSNormed over the whole projection (FlagQKNormWide) before NEOX rotary,
	// and the attention and FFN outputs are RMSNormed (RolePostAttnNorm,
	// RolePostFFNNorm) before their residual adds. OLMo 3 adds a sliding window
	// on three layers in four.
	ArchOLMo2 Arch = 60

	// ArchExaone4 is LG's EXAONE 4.0: OLMo 2's post-norm block with qwen3's
	// per-head q/k RMSNorm, NEOX rotary, and on the 32B a sliding window on
	// three layers in four with no rotary on the global one (FlagNoPEGlobal).
	ArchExaone4 Arch = 59

	// ArchMistral3 is Mistral Small 3.1/3.2, Magistral, Devstral and Ministral
	// 3, text only: llama's block with an attention temperature on every layer
	// (AttnTemp*, offset 0, floor the YaRN original context) and YaRN.
	ArchMistral3 Arch = 58

	// ArchOLMo3 is AI2's OLMo 3: OLMo 2's block with a sliding window on
	// three layers in four, whose rotary carries no scaling -- neither YaRN's
	// frequencies nor its magnitude (AttnFactor), which reach the global
	// layers only. A code of its own so a reader that would scale the local
	// layers refuses the container.
	ArchOLMo3 Arch = 57
)

func denseArchName(a Arch) (string, bool) {
	switch a {
	case ArchBitNet:
		return "bitnet", true
	case ArchSmolLM3:
		return "smollm3", true
	case ArchArcee:
		return "arcee", true
	case ArchSeedOSS:
		return "seed_oss", true
	case ArchOLMo2:
		return "olmo2", true
	case ArchExaone4:
		return "exaone4", true
	case ArchMistral3:
		return "mistral3", true
	case ArchOLMo3:
		return "olmo3", true
	}
	return "", false
}

// The encoders after BERT, counting up from 150: clear of the state-space
// group (100..) and of the vision-language text models (200..).
const (
	// ArchModernBERT is ModernBERT (answerdotai/ModernBERT, mmBERT): a
	// pre-norm encoder with no biases, LayerNorms without bias, a fused q|k|v
	// unfused at conversion, NEOX rotary on every layer, and GeGLU. Every
	// SWAPeriod-th layer from the first attends globally at RopeBase; the
	// rest attend within a symmetric window (SWAWindow/2 positions either
	// side) at RopeBaseSWA. Layer 0 has no attention norm; a final norm ends
	// the stack. Laya's decision head (Config.DecisionBlocks pre-norm blocks
	// with biases, a token-type row per question type, a scorer) follows it.
	ArchModernBERT Arch = 150
)

func encoderArchName(a Arch) (string, bool) {
	switch a {
	case ArchModernBERT:
		return "modern-bert", true
	}
	return "", false
}

// The flagship architectures added as one group (MiniMax, Gemma 4, DeepSeek
// V3.2, ...). The mixture families count up from ArchDBRX and the dense
// llama family down from the top of the 6-bit range, so these are taken from
// the middle of the gap, counting up from 44: a concurrent addition at either
// end does not collide with them. Arch.Valid asks the name tables.
const (
	// ArchMiniMaxM2 is MiniMax-M2 and M2.1/M2.5 (MiniMaxM2ForCausalLM): a GQA
	// block whose q and k are RMSNormed over the WHOLE projection
	// (FlagQKNormWide, OLMoE's) before NEOX rotary on part of each head, and
	// in every block a sigmoid top-k with a selection-only bias, renormalised,
	// with no shared expert and no routed scale. An arch of its own so an
	// older reader refuses it by code.
	ArchMiniMaxM2 Arch = 44

	// ArchGemma4 is Google's Gemma 4 text model (E2B, E4B, 26B-A4B, 31B):
	// gemma3's block with its global layers at another attention geometry
	// (HeadDim/NKVHead/NRot are the global layers', Config.Geoms each
	// block's), a weightless RMSNorm on v (Flag2VNorm), v
	// projected by k's weights where a block has no RoleAttnV, an attention
	// scale of one (AttnScale), a "proportional" rotary on the global layers
	// carried as RoleRopeFreqs, and a scalar on each block's output
	// (RoleLayerOutScale). A code of its own so an older reader, which would
	// run every layer at one geometry, refuses it.
	ArchGemma4 Arch = 45

	// ArchDeepseek32 is DeepSeek-V3.2 (DeepseekV32ForCausalLM): ArchDeepseek2's
	// MLA and V3 router with DeepSeek Sparse Attention. A lightning indexer
	// scores every cached position for each query (Config.IdxHeads heads of
	// IdxHeadDim, its own key cached beside the latent) and the attention reads
	// only the IdxTopK best. A code of its own so an older reader, which would
	// attend to every position, refuses it.
	ArchDeepseek32 Arch = 46

	// ArchGemma3n is Google's Gemma 3n text model (E2B, E4B): gemma3's
	// attention with Gemma 4's per-layer embeddings, KV sharing, unweighted v
	// norm and attention scale of one, inside AltUp's Config.PredStreams parallel
	// residual streams (each block predicts all of them from a router, runs on
	// the active one and corrects the rest), with a LAuReL low-rank branch
	// beside the attention and a gaussian top-k on the first NSparse FFNs'
	// gates. A code of its own so an older reader, which would run one stream,
	// refuses it.
	ArchGemma3n Arch = 47

	// ArchMiniMaxM3 is MiniMax-M3's text model (MiniMaxM3SparseForCausalLM):
	// MiniMax-M2's attention with a per-head Gemma q/k norm (the +1 baked by
	// the converter), DeepSeek-V3's dense lead, shared expert and routed scale
	// on a sigmoid router, gpt-oss's clamped SwiGLU everywhere, and MiniMax
	// Sparse Attention on every block after the dense lead: an indexer key per
	// position, cached as one more kv head, and per kv group a selection of
	// Config.IdxBlock-position blocks (IdxTopK by their best score, the
	// IdxLocal ending at the query's own forced in) that attention reads. A
	// code of its own so an older reader, which would attend to every
	// position, refuses it.
	ArchMiniMaxM3 Arch = 48

	// ArchDeepseek4 is DeepSeek V4's text model (DeepseekV4ForCausalLM):
	// Config.HCMult residual streams mixed into and out of each block by
	// Sinkhorn-normalised hyper-connections (RoleHC*), single-head MQA whose
	// key is its value, rotated on the tail of each head, with per-head sinks,
	// a sliding window on every block and, on the compressed blocks
	// (Config.CompKinds), the entries a gated compressor folds every
	// CompRateCSA or CompRateHCA positions into -- all of them on an HCA
	// block, the lightning indexer's top IdxTopK on a CSA block -- the
	// attention output rotated back and projected through OGroups grouped
	// low-rank matrices (RoleAttnOutA, then RoleAttnOut), and a sqrt-softplus
	// router (Flag2ExpertSqrtSoftplus) whose first NHashLayers blocks select
	// by token id (RoleHashExperts) under a clamped SwiGLU (Flag2SwiGLUClamp).
	// A code of its own so an older reader, which would run one stream and no
	// compressed history, refuses it.
	ArchDeepseek4 Arch = 49

	// ArchKimiK3 is Kimi-K3's text model (KimiK3ForConditionalGeneration's
	// KimiLinearForCausalLM): ArchKimiLinear's hybrid -- Kimi Delta Attention
	// in the linear blocks, MLA with no positional encoding in the full ones,
	// a V3 sigmoid router with a dense lead and shared experts -- with five
	// pieces of its own: attention over the residual's checkpoints, one banked
	// every AttnResBlock blocks and each sublayer's input a softmax mix of the
	// bank and the running residual by RoleAttnResScore/RoleFFNResScore (the
	// head's by RoleOutputResScore); a latent mixture whose routed experts run
	// at ExpertLatent behind RoleFFNRoutedDown, RoleFFNRoutedNorm and
	// RoleFFNRoutedUp; the situ activation (SituBeta, SituLinearBeta) in every
	// FFN; a sigmoid gate on the MLA output before its projection (RoleAttnGate
	// in a full block); and a full-rank KDA output gate (RoleAttnGate in a
	// linear block) whose decay is KDALowerBound * sigmoid(.) when the bound
	// is set. A code of its own so an older reader, which would run none of
	// the five, refuses it.
	ArchKimiK3 Arch = 50

	// ArchQwen4Exp is Qwen3.8-Flash-Next's text model (Qwen4ExpForCausalLM):
	// ArchQwen3Next's hybrid -- the gated delta rule with its output norm
	// gated by a sigmoid (Flag2DeltaSigmoidGate), every fourth block qwen3next's
	// gated attention, a softmax top-k mixture beside a sigmoid-gated shared
	// expert -- inside Config.HCMult residual streams mixed by low-rank gated
	// hyper-connections (Config.HCLowRank, RoleHC*Norm/Down/Up/Inject): each
	// sublayer reads the mean of its streams' grouped-RMSNormed values times a
	// sigmoid gate, and adds its output into every stream scaled by
	// 2*sigmoid(inject/HCMult); the head reads the same mix, with no output
	// norm. The attention blocks select their keys by QSA: an indexer key per
	// position, cached as one more kv head, scores each complete block of
	// QSARatio positions by its mean key, normed and rotated at the block's
	// first position, and attention reads the IdxTopK/QSARatio best blocks and
	// the incomplete tail. One linear block (NGramBlock) adds an n-gram
	// embedding: the token and its predecessors hashed per head into a shared
	// table (RoleNGramTable, read a row at a time), gated per stream and
	// convolved over the previous positions. A code of its own so an older
	// reader, which would run one stream, refuses it.
	ArchQwen4Exp Arch = 51
)

func flagshipArchName(a Arch) (string, bool) {
	switch a {
	case ArchMiniMaxM2:
		return "minimax-m2", true
	case ArchGemma4:
		return "gemma4", true
	case ArchDeepseek32:
		return "deepseek32", true
	case ArchGemma3n:
		return "gemma3n", true
	case ArchMiniMaxM3:
		return "minimax-m3", true
	case ArchDeepseek4:
		return "deepseek4", true
	case ArchKimiK3:
		return "kimi-k3", true
	case ArchQwen4Exp:
		return "qwen4exp", true
	}
	return "", false
}

// The modern text families added as one group: Ling 2.0, dots.llm1,
// Phi-3.5-MoE and Apertus. Their codes are 39..42 of the 6-bit range, between
// the families counting up from ArchDBRX and the dense group counting down
// from 63; Arch.Valid asks the name tables, so the gaps are not codes.
//
// Each is a code of its own so a reader that predates it refuses the container
// rather than running it as a neighbouring graph with a feature missing.
const (
	// ArchBailingMoE2 is inclusionAI's Ling and Ring 2.0 (Ling-mini-2.0,
	// Ling-flash-2.0): DeepSeek-V3's router -- sigmoid gating, a selection-only
	// bias (expert_bias), the group-limited top-k, renormalised, a routed scale
	// -- beside shared experts after a dense lead, on GQA attention with a
	// per-head q/k RMSNorm and NEOX rotary on part of each head. Its fused
	// query_key_value is unfused at conversion.
	ArchBailingMoE2 Arch = 42

	// ArchDots1 is rednote-hilab's dots.llm1: DeepSeek-V3's router (sigmoid,
	// a selection-only bias, renormalised, a routed scale) beside shared
	// experts after a dense lead, on attention (MHA in every published
	// checkpoint) with qwen3's per-head q/k RMSNorm and NEOX rotary over the
	// whole head.
	ArchDots1 Arch = 41

	// ArchPhiMoE is Microsoft's Phi-3.5-MoE: the classic sequential block --
	// LayerNorm with biases, biased q/k/v/o, a biased head behind a biased
	// LayerNorm -- with phi3's NEOX LongRoPE, and a mixture routed by
	// SPARSEMIXER's inference path: the top two logits, each weighted by a
	// softmax over the logits within a relative 2*PhiMoEJitter of it, not
	// renormalised. The code carries the router: a GGUF states no gating key
	// for it, and llama.cpp runs a renormalised softmax top-2 there.
	ArchPhiMoE Arch = 40

	// ArchApertus is the Swiss AI Initiative's Apertus: llama's block with
	// qwen3's per-head q/k RMSNorm, NEOX rotary with llama 3's per-pair
	// factors (rope_freqs), and an UNGATED MLP -- up, xIELU, down -- whose
	// activation carries four numbers per block (RoleXIELU). The code carries
	// the activation, since a GGUF states it only through those numbers.
	ArchApertus Arch = 39
)

func moe2ArchName(a Arch) (string, bool) {
	switch a {
	case ArchBailingMoE2:
		return "bailingmoe2", true
	case ArchDots1:
		return "dots1", true
	case ArchPhiMoE:
		return "phimoe", true
	case ArchApertus:
		return "apertus", true
	}
	return "", false
}

// The state-space hybrids and the short convolutions. Their codes sit in the
// middle of the 6-bit range (43..49), clear of the dense group counting down
// from 63 and of additions counting up from ArchHunyuan; Arch.Valid asks the
// name tables, so the gaps are not codes.
//
// Each is a code of its own so a reader that predates it refuses the
// container: every one of their layers would otherwise run as a gated delta
// rule, fluently.
const (
	// ArchMamba2 is Mamba-2 (state-spaces/mamba2, Mamba-Codestral): every
	// block a Mamba-2 mixer (LayerSSD) under one RMSNorm, and no FFN.
	ArchMamba2 Arch = 100

	// ArchGraniteHybrid is IBM Granite 4.0-H: Granite's four scales
	// (ArchGranite's) over Mamba-2 blocks and attention blocks with no rotary,
	// each followed by a SwiGLU FFN or a softmax mixture beside a shared MLP.
	ArchGraniteHybrid Arch = 101

	// ArchNemotronH is NVIDIA's Nemotron-H and Nemotron Nano 2, and its
	// mixture Nemotron 3 Nano (nemotron_h_moe): every source layer is ONE
	// mixer under its own RMSNorm -- Mamba-2, attention with no rotary, or a
	// squared-ReLU MLP (a mixture of them in the MoE) -- merged at conversion
	// into blocks of a mixer and an optional FFN.
	ArchNemotronH Arch = 102

	// ArchFalconH1 is TII's Falcon-H1: every block runs attention (NEOX
	// rotary) and a Mamba-2 mixer in parallel on one normed input
	// (jlm.LayerSSDAttn), then a SwiGLU FFN. Its muP multipliers are folded
	// into the weights by the converter that writes the GGUF.
	ArchFalconH1 Arch = 103

	// ArchLFM2 is Liquid's LFM2: gated short-convolution blocks
	// (jlm.LayerShortConv) with attention blocks among them (NEOX rotary, a
	// q/k RMSNorm per head), each followed by a SwiGLU FFN.
	ArchLFM2 Arch = 104

	// ArchMamba is Mamba-1 (state-spaces/mamba, FalconMamba): every block a
	// Mamba-1 mixer (LayerScan) under one RMSNorm, and no FFN.
	ArchMamba Arch = 105

	// ArchJamba is AI21's Jamba: Mamba-1 mixers with weighted dt/B/C norms
	// and attention blocks with no rotary among them, each followed by a
	// SwiGLU FFN or a softmax top-k mixture (on the blocks with a router).
	ArchJamba Arch = 106

	// ArchLFM2Enc is LFM2 made an encoder (LiquidAI's LFM2.5-Encoder, d1-omni's
	// trunk): LFM2's blocks with no causal mask, a centred 3-tap convolution
	// where LFM2's is causal, and the final norm over every row. A code of its
	// own so a reader that knows only the decoder refuses the container rather
	// than running it with a causal mask. A media prefix's rows attend only to
	// each other and the convolution never lets them read the text.
	ArchLFM2Enc Arch = 107
)

func ssmArchName(a Arch) (string, bool) {
	switch a {
	case ArchMamba2:
		return "mamba2", true
	case ArchGraniteHybrid:
		return "granitehybrid", true
	case ArchNemotronH:
		return "nemotron_h", true
	case ArchFalconH1:
		return "falcon-h1", true
	case ArchLFM2:
		return "lfm2", true
	case ArchMamba:
		return "mamba", true
	case ArchJamba:
		return "jamba", true
	case ArchLFM2Enc:
		return "lfm2enc", true
	}
	return "", false
}

// The vision-language families added as one group (Qwen3-VL, GLM-4.xV,
// Kimi-VL, HunyuanVL): the text architectures they need, their projectors and
// their towers' own roles. Each range is its own, far from every other group's,
// because several groups were added at once and a code is the container's,
// forever: architectures count up from 200 (clear of the moe2 group counting
// down from 42, the flagships up from 44, the dense llama family down from 63
// and the state-space families up from 100), projectors up from 48, roles up
// from 700. Arch.Valid, Role.Valid and the String methods ask the tables, so a
// gap between two ranges is not a code.
const (
	// ArchQwen3VL is Qwen3-VL's text model: qwen3's block (per-head q/k
	// RMSNorm, NEOX rotary) whose rotary is INTERLEAVED M-RoPE -- a head's
	// pairs turn by (t, h, w) in rotation, pair i by h when i%3 == 1 and
	// i < 3*RopeSections[1], by w when i%3 == 2 and i < 3*RopeSections[2], by
	// t otherwise (transformers' recomposition_frequencies). A code of its own
	// so an older reader, which would lay the sections out contiguously,
	// refuses it. A picture's deepstack rows (the tower's) add into the image
	// rows after each of the first text blocks; the text model needs no field
	// for that, since the tower says how many there are.
	ArchQwen3VL Arch = 200

	// ArchQwen3VLMoE is Qwen3-VL's mixture text model (30B-A3B, 235B-A22B):
	// qwen3moe's block under ArchQwen3VL's rotary.
	ArchQwen3VLMoE Arch = 201

	// ArchGLM4 is Zhipu's GLM-4-0414 block (GLM-4-9B/32B-0414, GLM-Z1, and the
	// text model of GLM-4.1V and GLM-4.6V-Flash): llama's block with biased
	// q, k and v, a norm after the attention and after the MLP before each
	// residual add (RolePostAttnNorm, RolePostFFNNorm, beside the pre-norms),
	// and a rotary on the first part of each head -- NORM pairs, or with
	// RopeSections the NEOX M-RoPE of Qwen2-VL, which llama.cpp's converter
	// permutes q and k into.
	ArchGLM4 Arch = 202

	// ArchGLM4VMoE is GLM-4.5V's and GLM-4.6V's text model: ArchGLM4MoE
	// under Qwen2-VL's contiguous M-RoPE. A code of its own so an older
	// reader, which would rotate an image's rows by their sequence index,
	// refuses it.
	ArchGLM4VMoE Arch = 203

	// ArchHunyuanVL is HunyuanVL's text model (HunyuanOCR): hunyuan-dense's
	// block -- the per-head q/k RMSNorm after the rotary, k's weight folded
	// into q's (FlagQKNormPostRope) -- under XD-RoPE: four position axes
	// (sequence, width, height, image index) assigned per ELEMENT of
	// cat(freq, freq) in RopeSections' order, so a NEOX pair's two halves turn
	// by different axes on an image's rows (transformers'
	// recomposition_frequencies). That rotation does not keep a head's RMS,
	// so k's norm goes after it too. An image's rows keep their sequence
	// positions: nothing after an image is shifted. A code of its own so an
	// older reader, which would rotate the pair as one, refuses it.
	ArchHunyuanVL Arch = 204
)

func visionArchName(a Arch) (string, bool) {
	switch a {
	case ArchQwen3VL:
		return "qwen3vl", true
	case ArchQwen3VLMoE:
		return "qwen3vlmoe", true
	case ArchGLM4:
		return "glm4", true
	case ArchGLM4VMoE:
		return "glm4vmoe", true
	case ArchHunyuanVL:
		return "hunyuan_vl", true
	}
	return "", false
}

// Arch is the converter's name for the graph a source file declares: what it
// reads the file as, and so what it states in the container. No container
// records it (v31): a reader runs what the config record and the tensors
// state.
type Arch uint16

const (
	ArchNone Arch = 0

	ArchLlama    Arch = 1 // llama, llama2, llama3
	ArchQwen2    Arch = 2
	ArchQwen3    Arch = 3
	ArchQwen3MoE Arch = 4
	ArchOLMoE    Arch = 5
	ArchGemma    Arch = 6
	ArchPhi3     Arch = 7 // phi3, phi4
	ArchCLIP     Arch = 8 // a vision tower, not a language model
	// ArchQwen3Next is the hybrid: most layers run a gated delta rule over a
	// recurrent state and the rest run ordinary attention. See LayerKinds.
	ArchQwen3Next Arch = 9
	// ArchQwen2VL is qwen2's block with multimodal rope: position is a
	// (t, h, w) triple split across RopeSections of the head's rotary pairs.
	// For a text-only prompt all three are the token index, so M-RoPE equals
	// NEOX until image rows occupy a span of positions.
	ArchQwen2VL Arch = 10
	// ArchGemma2 is gemma1's block plus two extra RMSNorms per block (after
	// attention and after the FFN: RolePostAttnNorm/RolePostFFNNorm),
	// sliding-window attention on alternating layers, and logit softcapping,
	// x = c*tanh(x/c), on the attention scores and the output logits. Leaving
	// the softcap out is fluent, since only extremes are pulled in.
	ArchGemma2 Arch = 11
	// ArchGemma3 is gemma2's block with the softcaps removed, plus QK-norm on
	// each head before the rotary, a 5-local-1-global window (SWAPeriod 6), and
	// two rotary bases: local layers at 10000, global at the file's
	// rope.freq_base (1e6 on gemma-3-1b). The 10000 is an architecture constant
	// llama.cpp hardcodes, not a key; falling back to the global base is fluent
	// and wrong in five layers of six.
	ArchGemma3 Arch = 12
	// ArchGPTOSS is OpenAI's gpt-oss mixture. Attention carries a learned
	// per-head sink (RoleAttnSinks) that joins each softmax's denominator; q, k,
	// v and o are biased; alternate layers slide over 128 positions (SWAPeriod
	// 2); the rotary is YaRN-scaled (Config.Yarn*). The router and every expert
	// projection are biased, and the activation is FlagSwiGLUOAI.
	ArchGPTOSS Arch = 13

	// ArchDeepseek2 is DeepSeek-V2, V3 and R1, and the models that declare
	// DeepseekV3ForCausalLM under another name: Kimi-K2, Moonlight-16B-A3B, and
	// GLM-4.7-Flash (Glm4MoeLiteForCausalLM, same tensors). GLM-4.5-Air
	// (Glm4MoeForCausalLM) has no kv_lora_rank, is not MLA, and is
	// ArchGLM4MoE.
	//
	// MLA: the attention is MQA over a latent. kv_a projects the residual to
	// KVLoraRank + NRot; the cache stores that row once per position for the
	// layer, and the up-projections are absorbed into q (RoleAttnKB) and out
	// (RoleAttnVB), so K and V have different head widths (see
	// Config.HeadDimV).
	//
	// The router is sigmoid rather than softmax, carries a selection-only bias,
	// scores experts in groups, and scales the surviving weights. See
	// FlagExpertSigmoid, NExpertGroup and ExpertScale.
	ArchDeepseek2 Arch = 14

	// ArchKimiLinear is the second hybrid: ArchDeepseek2's MLA attention and
	// router in its full layers, Kimi Delta Attention in its linear ones. It is
	// an arch of its own rather than a deepseek2 flag because most blocks do not
	// run attention at all (Config.LayerKinds says which). KDA is not
	// qwen3next's delta rule: the decay is per channel, both gates come off a
	// low-rank bottleneck, and beta arrives alone (kernels.GatedDeltaStepChan).
	ArchKimiLinear Arch = 15

	// ArchBERT is the BERT encoder, an embedding model. Token, token-type and
	// learned position embeddings are summed and LayerNormed; every block is
	// post-norm (LN(x + attn(x)), then LN(x + ffn(x))) with biased q/k/v/o, an
	// ungated GELU MLP with biases, and bidirectional attention. There is no
	// head: the last residual is pooled (Flags' pooling bits) and L2-normalised.
	// all-minilm, mxbai-embed-large, bge-large and others declare this graph.
	ArchBERT Arch = 16
	// ArchNomicBERT is nomic-embed-text's encoder: ArchBERT's post-norm block
	// with no position table (NEOX rotary on q and k), a SwiGLU MLP, and no
	// projection biases. The converter unfuses its q|k|v as it does phi3's.
	ArchNomicBERT Arch = 17

	// ArchLlama4 is Meta's Llama 4 (Scout 16E, Maverick 128E): llama's block
	// with five differences, each fluent when missed.
	//
	//	iRoPE          three layers in four rotate and attend within an
	//	               8192-position chunk (FlagSWAChunked); the fourth has no
	//	               positional encoding and attends to everything
	//	               (FlagNoPEGlobal)
	//	temperature    the NoPE layers scale q by 1 + s*ln(1 + floor((p+o)/f))
	//	               (AttnTempScale, AttnTempFloor, AttnTempOffset)
	//	QK L2 norm     a weightless RMSNorm of each q and k head after the
	//	               rotary, on the rotating layers only (FlagQKL2Norm)
	//	interleave     dense and mixture blocks alternate by a step (MoEStep)
	//	router         top-1 sigmoid, not renormalised, weighting the expert's
	//	               input (FlagExpertWeightIn)
	//
	// plus an ungated shared expert as wide as a routed one. None of the five
	// is visible in a GGUF (llama.cpp hardcodes them against the name), so the
	// container states them.
	ArchLlama4 Arch = 18

	// ArchGranite is IBM's Granite 3.x and 4.0 dense and mixture models
	// (llama.cpp's "granite" and "granitemoe"): llama's block with four scaling
	// constants:
	//
	//	embedding_scale    multiplies the embedding (EmbdScale)
	//	residual_scale     multiplies every block's attention and FFN output
	//	                   before its residual add (ResidualScale)
	//	attention.scale    the score multiplier, in place of 1/sqrt(HeadDim)
	//	                   (AttnScale)
	//	logit_scale        divides the logits (LogitScale)
	//
	// An arch of its own so an older reader, which never reads the config's
	// optional tail, refuses it by code rather than running every constant at
	// one.
	ArchGranite Arch = 19

	// ArchPhi2 is Microsoft's phi-2 (and phi-1.5), the first classic
	// transformer block (C6), GPT-2's shape rather than llama's:
	//
	//	LayerNorm with a bias      in place of RMSNorm (FlagLayerNorm)
	//	one norm, two branches     attention and the FFN both read the same
	//	                           normed input and both add into the
	//	                           residual (FlagParallel)
	//	an ungated FFN             up, GELU-tanh, down; the absent gate
	//	                           tensor says so
	//	biases on every matrix     q, k, v, o, up, down and the head
	//	partial NEOX rotary        32 of 80 dimensions on phi-2
	//
	// An arch of its own so an older reader refuses it rather than running it
	// as llama's block.
	ArchPhi2 Arch = 20

	// ArchStarcoder is BigCode's StarCoder / SantaCoder (GPTBigCode): GPT-2's
	// block plus a learned absolute position table (RolePosEmbd) added to the
	// embedding and no rotary (FlagNoPosEnc). Attention is multi-query, q|k|v
	// is split at conversion, the FFN is ungated GELU-tanh, every matrix is
	// biased, and the residual is sequential. An older reader would run it with
	// no positional information.
	ArchStarcoder Arch = 21

	// ArchCommandR is Cohere's Command-R and Aya ("command-r") and Command-R7B /
	// Command-A ("cohere2"): the C6 parallel block from one bias-free
	// LayerNorm, a gated SwiGLU FFN, adjacent-pair rotary, tied embeddings, and
	// a logit_scale that multiplies (stored as its reciprocal in LogitScale).
	// cohere2 adds a sliding window on three layers in four (SWAPeriod) and no
	// rotary on the global layer (FlagNoPEGlobal).
	ArchCommandR Arch = 22

	// ArchStarcoder2 is BigCode's StarCoder2: the classic sequential block with
	// biased LayerNorms, biases on q/k/v/o and the ungated GELU-tanh FFN, full
	// NEOX rotary and grouped kv heads. The code exists so an older reader
	// refuses it. Its config.json's sliding window is not applied, following
	// llama.cpp (RULE 7m): the GGUF carries no window key.
	ArchStarcoder2 Arch = 23

	// ArchStableLM is Stability's StableLM 2 and StableLM 3B: LayerNorm with
	// biases, biased q/k/v and unbiased o, a gated SwiGLU FFN, NEOX rotary on a
	// quarter of each head, and a residual that is sequential when the block
	// carries an FFN norm and parallel when it does not (decided by the tensor,
	// as llama.cpp does). StableLM 2 12B's per-head q/k LayerNorm is refused.
	ArchStableLM Arch = 24

	// ArchFalcon is TII's Falcon 7B/40B/180B: the parallel block with an
	// ungated GELU FFN, MQA or GQA over NEOX rotary, LayerNorm with biases and
	// no matrix biases. 7B normalises once; 40B and 180B normalise twice, which
	// the converter writes as the block's attention and FFN norms.
	ArchFalcon Arch = 25

	// ArchNemotron is NVIDIA's Nemotron-4 family at the sizes llama.cpp runs
	// (Minitron, Nemotron-Mini): the classic sequential block with LayerNorm
	// (+bias), NEOX rotary on half of each head, and an ungated squared-ReLU FFN
	// (FlagReLU2). Its LayerNorm1p's +1 is already in the file (llama.cpp's
	// converter adds it), so a plain LayerNorm runs; adding it again is fluent.
	ArchNemotron Arch = 26

	// ArchDBRX is Databricks' DBRX (132B, 16 experts, 4 routed): the sequential
	// block with bias-free LayerNorms (norm_2, the GGUF's attn_output_norm, is
	// the FFN norm), a fused q|k|v split at conversion, NEOX rotary, a softmax
	// top-k mixture of SwiGLU experts, an untied head, and CLIP_QKV: q, k and v
	// clamped to [-ClampKQV, ClampKQV] after the projection. An arch of its own
	// so an older reader refuses it instead of skipping the clamp.
	ArchDBRX Arch = 27

	// ArchGLM4MoE is Zhipu's GLM-4.5, GLM-4.6 and GLM-4.5-Air
	// (Glm4MoeForCausalLM): a GQA block with biased q/k/v and an unbiased o,
	// an optional per-head q/k RMSNorm, NEOX rotary on part of each head, and
	// DeepSeek-V3's router without MLA -- sigmoid gating, a selection-only
	// bias, renormalisation and a routed scale -- beside an ungated shared
	// expert, after a dense lead. Its post_attention_norm is the FFN's input
	// norm. It is not ArchDeepseek2: there is no kv_lora_rank. An arch of its
	// own so an older reader refuses it by code.
	ArchGLM4MoE Arch = 28

	// ArchQwen2MoE is Qwen1.5-MoE and Qwen2-57B-A14B: qwen2's block (biased
	// q/k/v, NEOX rotary) with a softmax top-k mixture that is not
	// renormalised, beside a shared expert scaled by its own sigmoid gate
	// (RoleShRouter, qwen3next's). An arch of its own so an older reader
	// refuses it by code.
	ArchQwen2MoE Arch = 29

	// ArchErnie45MoE is Baidu's ERNIE 4.5 mixtures (21B-A3B, 300B-A47B):
	// llama's block (interleaved rotary, a tied head) with a softmax top-k
	// whose SELECTION adds a per-expert bias to the softmax probability
	// (RoleExpProbsB), renormalised, beside shared experts, after a dense lead
	// and interleaved by a step (MoEStep). An arch of its own because the step
	// is the config's optional tail, which an older reader skips.
	ArchErnie45MoE Arch = 30

	// ArchHunyuan is Tencent's Hunyuan, dense (0.5B-7B) and mixture
	// (Hunyuan-A13B): llama's block with NEOX rotary and a per-head q/k
	// RMSNorm applied AFTER the rotary (FlagQKNormPostRope); the mixture is a
	// renormalised softmax top-k beside one ungated shared expert in every
	// block. An arch of its own so an older reader, which would norm before the
	// rotary, refuses it by code.
	ArchHunyuan Arch = 31

	archMax = ArchHunyuan
)

func coreArchName(a Arch) (string, bool) {
	switch a {
	case ArchLlama:
		return "llama", true
	case ArchQwen2:
		return "qwen2", true
	case ArchQwen3:
		return "qwen3", true
	case ArchQwen3MoE:
		return "qwen3moe", true
	case ArchOLMoE:
		return "olmoe", true
	case ArchGemma:
		return "gemma", true
	case ArchPhi3:
		return "phi3", true
	case ArchCLIP:
		return "clip", true
	case ArchQwen3Next:
		return "qwen3next", true
	case ArchQwen2VL:
		return "qwen2vl", true
	case ArchGemma2:
		return "gemma2", true
	case ArchGemma3:
		return "gemma3", true
	case ArchGPTOSS:
		return "gptoss", true
	case ArchDeepseek2:
		return "deepseek2", true
	case ArchKimiLinear:
		return "kimilinear", true
	case ArchBERT:
		return "bert", true
	case ArchNomicBERT:
		return "nomic-bert", true
	case ArchLlama4:
		return "llama4", true
	case ArchGranite:
		return "granite", true
	case ArchPhi2:
		return "phi2", true
	case ArchStarcoder:
		return "starcoder", true
	case ArchCommandR:
		return "command-r", true
	case ArchStableLM:
		return "stablelm", true
	case ArchStarcoder2:
		return "starcoder2", true
	case ArchFalcon:
		return "falcon", true
	case ArchNemotron:
		return "nemotron", true
	case ArchDBRX:
		return "dbrx", true
	case ArchGLM4MoE:
		return "glm4moe", true
	case ArchQwen2MoE:
		return "qwen2moe", true
	case ArchErnie45MoE:
		return "ernie4_5-moe", true
	case ArchHunyuan:
		return "hunyuan", true
	}
	return "", false
}

func (a Arch) String() string {
	if n := a.name(); n != "" {
		return n
	}
	return "arch(" + itoa(int(a)) + ")"
}

func (a Arch) name() string {
	if n, ok := coreArchName(a); ok {
		return n
	}
	for _, group := range archGroups() {
		if n, ok := group(a); ok {
			return n
		}
	}
	return ""
}

func archGroups() []func(Arch) (string, bool) {
	return []func(Arch) (string, bool){denseArchName, ssmArchName, moe2ArchName, flagshipArchName, encoderArchName, visionArchName}
}

// Valid reports whether a is an architecture this format defines: a code
// with a name, so a gap between two ranges of codes is not one.
func (a Arch) Valid() bool { return a != ArchNone && (a <= archMax || a.name() != "") }

// PhiMoEJitter is sparsemixer's jitter_eps: router_jitter_noise, which every
// published Phi-3.5-MoE (and PhimoeConfig's default) sets to 0.01. A GGUF does
// not carry it, so the architecture does; the safetensors converter refuses a
// config that states another value.
const PhiMoEJitter = 0.01

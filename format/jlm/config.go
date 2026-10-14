package jlm

// The container's model description: typed fields, not a key-value bag. The
// source format's keys are namespaced by architecture, so reading one needs
// the architecture and another project's vocabulary, and a key nobody wrote
// looks like a key nobody read. Here the fields are fixed and their zero
// values documented. The architecture is a code selecting a graph this engine
// implements, so an unimplemented one is refused at conversion.

// Arch selects the block graph.
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

var archNames = [...]string{
	ArchLlama: "llama", ArchQwen2: "qwen2", ArchQwen3: "qwen3", ArchQwen3MoE: "qwen3moe",
	ArchOLMoE: "olmoe", ArchGemma: "gemma", ArchPhi3: "phi3", ArchCLIP: "clip",
	ArchQwen3Next: "qwen3next", ArchQwen2VL: "qwen2vl", ArchGemma2: "gemma2",
	ArchGemma3: "gemma3", ArchGPTOSS: "gptoss", ArchDeepseek2: "deepseek2",
	ArchKimiLinear: "kimilinear", ArchBERT: "bert", ArchNomicBERT: "nomic-bert", ArchLlama4: "llama4",
	ArchGranite: "granite", ArchPhi2: "phi2", ArchStarcoder: "starcoder",
	ArchCommandR: "command-r", ArchStableLM: "stablelm", ArchStarcoder2: "starcoder2", ArchFalcon: "falcon",
	ArchNemotron: "nemotron", ArchDBRX: "dbrx", ArchGLM4MoE: "glm4moe",
	ArchQwen2MoE: "qwen2moe", ArchErnie45MoE: "ernie4_5-moe", ArchHunyuan: "hunyuan",
}

// LayerKind says what one repeating unit is. The container stores one per
// layer rather than a rule: the source carries no per-layer key, and engines
// hardcode "every fourth layer is full attention" against the architecture
// name. The converter writes down which layers carry a recurrent block.
type LayerKind uint8

const (
	LayerNone LayerKind = 0
	// LayerFullAttn is softmax attention over a KV cache.
	LayerFullAttn LayerKind = 1
	// LayerLinearAttn is a causal convolution and a gated delta rule over a
	// recurrent state whose size does not depend on the context.
	LayerLinearAttn LayerKind = 2
	// LayerSSD is a Mamba-2 mixer: the same causal convolution (with a bias)
	// over the same per-head state, stepped by the selective update -- the
	// delta rule with no key dot, dt in the gate's place and a D skip -- and
	// a gated RMSNorm of y*silu(z) grouped by SSM.Groups. Its code is apart
	// from the low ones so a concurrent addition of a kind does not collide.
	LayerSSD LayerKind = 8
	// LayerSSDAttn is Falcon-H1's block: softmax attention AND a Mamba-2
	// mixer side by side over the same normed input, their outputs summed
	// into one residual add before the FFN. It keeps a KV cache and a
	// recurrent state both.
	LayerSSDAttn LayerKind = 9
	// LayerShortConv is LFM2's gated short convolution: in_proj to B, C and
	// x, B*x through a depthwise causal convolution of SSM.ConvKernel taps
	// over NEmbd channels, C times that, out_proj. Its only state is the
	// convolution's window.
	LayerShortConv LayerKind = 10
	// LayerMamba1 is Mamba-1's selective scan (Mamba, FalconMamba, Jamba): in
	// the delta rule's terms every one of SSM.Inner channels is a value head
	// of one row, with B and C shared by all of them and a decay that is a
	// vector over the state (exp(dt*A), A one row of StateSize per channel).
	// dt comes off a low-rank bottleneck of x (RoleSSMXDt, RoleSSMDtProj), B
	// and C off x too, each optionally RMS-normed (Jamba, FalconMamba).
	LayerMamba1 LayerKind = 11
)

// Recurrent reports whether a layer keeps a recurrent state in place of a KV
// cache: a convolution window and a per-head state, both constant in the
// context. Every consumer of that state -- its allocation, migration, the
// prefix cache's restore point, a session's reset -- asks this, not one kind.
func (k LayerKind) Recurrent() bool {
	return k == LayerLinearAttn || k == LayerSSD || k == LayerSSDAttn || k == LayerShortConv ||
		k == LayerMamba1
}

// Attends reports whether a layer runs softmax attention over a KV cache.
func (k LayerKind) Attends() bool { return k == LayerFullAttn || k == LayerSSDAttn }

// QKNormOn is the one derivation of whether a block RMSNorms q and k: the
// model's flag, on every block that has attention. The converter's check, the
// reader's check, the host and the device all read it, so a container whose flag
// and tensors disagree is refused rather than run one way on one tier and
// another way on the other.
func QKNormOn(flag bool, kind LayerKind) bool {
	return flag && kind.Attends()
}

// QKNormAt is QKNormOn for text block b of this model. A block past
// LayerKinds (a prediction block) is attention.
func (c *Config) QKNormAt(b int) bool {
	return QKNormOn(c.Flags.Has(FlagQKNorm), c.layerKind(b))
}

func (c *Config) layerKind(b int) LayerKind {
	if b >= 0 && b < len(c.LayerKinds) {
		return c.LayerKinds[b]
	}
	return LayerFullAttn
}

func (k LayerKind) String() string {
	switch k {
	case LayerFullAttn:
		return "full"
	case LayerLinearAttn:
		return "linear"
	case LayerSSD:
		return "ssd"
	case LayerSSDAttn:
		return "ssd+attn"
	case LayerShortConv:
		return "shortconv"
	case LayerMamba1:
		return "mamba1"
	}
	return "layer(" + itoa(uint64(k)) + ")"
}

func (a Arch) String() string {
	if n := a.name(); n != "" {
		return n
	}
	return "arch(" + itoa(uint64(a)) + ")"
}

func (a Arch) name() string {
	if int(a) < len(archNames) && archNames[a] != "" {
		return archNames[a]
	}
	if n := denseArchNames[a]; n != "" {
		return n
	}
	if n := ssmArchNames[a]; n != "" {
		return n
	}
	if n := moe2ArchNames[a]; n != "" {
		return n
	}
	if n := flagshipArchNames[a]; n != "" {
		return n
	}
	if n := encoderArchNames[a]; n != "" {
		return n
	}
	return visionArchNames[a]
}

// Valid reports whether a is an architecture this format defines: a code
// with a name, so a gap between two ranges of codes is not one.
func (a Arch) Valid() bool { return a != ArchNone && (a <= archMax || a.name() != "") }

// Encoder reports whether a is an embedding encoder (the BERT family): run by
// the engine's encoder rather than its decoder graph, with no KV cache and no
// logits. A decoder that pools (FlagPoolMean and its siblings) is not one.
func (a Arch) Encoder() bool {
	return a == ArchBERT || a == ArchNomicBERT || a == ArchModernBERT || a == ArchLFM2Enc
}

// Config is the model description. Every field is written by the converter and
// read by the engine; nothing is derived at load from a string.
type Config struct {
	Arch Arch

	NLayer  uint32
	NEmbd   uint32
	NHead   uint32
	NKVHead uint32
	HeadDim uint32
	NRot    uint32 // rotary dimensions; equals HeadDim unless the model says otherwise
	NFFN    uint32
	NVocab  uint32
	NCtx    uint32

	RMSEps   float32
	RopeBase float32
	// AttnFactor scales cos and sin (phi3 LongRoPE). Zero means one.
	AttnFactor float32
	// EmbdScale multiplies the looked-up embedding. Zero means one.
	EmbdScale float32

	// Mixture of experts. NExpert 0 is a dense feed-forward.
	NExpert     uint32
	NExpertUsed uint32
	NFFNExp     uint32 // the PER-EXPERT width, which is not NFFN

	// LayerKinds is one entry per layer, or empty for a model whose every layer
	// is ordinary attention. A reader that finds it empty treats every layer as
	// LayerFullAttn, which is what every non-hybrid architecture is.
	LayerKinds []LayerKind

	// SSM is the linear-attention geometry, present only when some layer is
	// LayerLinearAttn.
	SSM SSMConfig

	// NFFNShExp is the shared expert's feed-forward width. A shared expert runs
	// for every token beside the routed ones and has its own gate; zero means
	// the model has none.
	NFFNShExp uint32

	// RopeSections splits a head's rotary pairs (not dimensions) between the
	// components of a multimodal position (t, h, w); the fourth entry is unused.
	// It is empty for every architecture whose position is a plain index, and
	// the sections must sum to NRot/2 (Qwen2-VL-2B: [16, 24, 24, 0], head_dim
	// 128). Read as dimensions they rotate a quarter of each head, fluently.
	RopeSections [4]uint32

	// Sliding-window attention. Zero for either means every layer sees the
	// whole history.
	SWAWindow   uint32
	SWAPeriod   uint32
	RopeBaseSWA float32

	// Logit softcapping: x = c*tanh(x/c), with c the cap. Zero means none.
	// gemma2 caps the attention scores at 50 and the output logits at 30, so
	// they are two fields.
	AttnSoftcap  float32
	FinalSoftcap float32

	// YaRN rotary scaling. YarnFactor 0 means none. The per-pair frequency is
	// base^(-2p/NRot) mixed toward that divided by YarnFactor by a linear ramp
	// over pairs, from full extrapolation below the pair corr(YarnBetaFast) to
	// full interpolation above corr(YarnBetaSlow), where
	//
	//	corr(b) = NRot * ln(YarnOrigCtx / (2*pi*b)) / (2 * ln(RopeBase))
	//
	// floored and ceiled respectively unless FlagYarnExact is set, and clamped
	// to [0, NRot-1]. The magnitude correction is not here: the converter folds
	// it into AttnFactor, which already scales cos and sin.
	YarnFactor   float32
	YarnOrigCtx  uint32
	YarnBetaFast float32
	YarnBetaSlow float32

	// RopeLinear divides every rotary angle of the global layers by itself
	// (linear position interpolation, gemma-3-4b's factor 8). Zero means none.
	// Layers on the local rotary (RopeBaseSWA) are not scaled.
	RopeLinear float32

	// HeadDimV is the value head's width when it differs from HeadDim. Zero
	// means they are equal, which is every architecture but MLA: its absorbed
	// form attends over a KVLoraRank+NRot key and returns a KVLoraRank value.
	HeadDimV uint32

	// NDenseLead is how many leading blocks have an ordinary dense
	// feed-forward before the mixture begins. Zero means every block is
	// whatever NExpert says. A count, because every model that has one ships a
	// prefix (source key "leading_dense_block_count").
	NDenseLead uint32

	// Multi-head Latent Attention. KVLoraRank 0 means the model has none and
	// every field in this block is unread.
	//
	// The block's attention is MQA over a compressed latent:
	//
	//	kv_a(x)          -> KVLoraRank + NRot            (RoleAttnKVA)
	//	                    RMSNormed over the first KVLoraRank (RoleAttnKVANorm)
	//	q                -> NHead * HeadDim              (RoleAttnQB, or
	//	                    RoleAttnQ directly when QLoraRank is 0)
	//	                    split HeadDim into
	//	                      HeadDim-NRot  "nope", not rotated
	//	                      NRot          "pe",   rotated
	//
	// The nope half of q is multiplied by RoleAttnKB into KVLoraRank and
	// concatenated with the rotated pe half, so the query is KVLoraRank + NRot
	// wide and matches the cached row. RoleAttnVB takes the attention's
	// KVLoraRank-wide result back out to HeadDimV per head before RoleAttnOut.
	//
	// NRot is the rotary half and there is no second field for it: the source
	// writes qk_rope_head_dim into rope.dimension_count, so a separate field
	// could only disagree.
	//
	// HeadDim is the natural width (qk_nope+qk_rope, 192 on DeepSeek), not the
	// absorbed KVLoraRank+NRot (576); the converter reads key_length_mla and
	// derives the absorbed width. The softmax is scaled by sqrt(HeadDim).
	//
	// The cache holds one row per position for the whole layer, not one per
	// head, which is the point of the mechanism.
	QLoraRank  uint32 // 0 means q projects from the residual in one step
	KVLoraRank uint32

	// YarnLogMul is the multiplier on YaRN's magnitude correction, ln(factor).
	// Zero means the standard 0.1.
	//
	// It is 0.1*mscale_all_dim, the convention llama.cpp writes into
	// rope.scaling.yarn_log_multiplier (and divides by 0.1 at load). It reaches
	// two places: AttnFactor scales cos and sin, and the attention softmax is
	// scaled by the square of the same correction.
	YarnLogMul float32

	// ExpertScale multiplies the routed mixture weights after any
	// renormalisation -- DeepSeek's routed_scaling_factor. Zero means one.
	ExpertScale float32

	// Grouped expert selection. NExpertGroup 0 or 1 means the top-k is taken
	// over every expert at once, which is every mixture here before DeepSeek.
	//
	// Otherwise the experts divide into NExpertGroup contiguous groups, each
	// group scores as the sum of its top two biased weights, the best
	// NExpertGroupUsed groups survive, and the top-k runs over those alone.
	// Kimi-K2 sets both to 1, so the grouped path must reduce exactly to the
	// ungrouped one.
	NExpertGroup     uint32
	NExpertGroupUsed uint32

	// NMTP is how many multi-token-prediction blocks follow the trunk: blocks
	// NLayer..NLayer+NMTP-1 of the text page array. Zero means none. A block
	// of that range is an ordinary attention block of this architecture (its
	// layer kind is LayerFullAttn whatever LayerKinds says of the trunk, and a
	// mixture where MoEAt says so) plus the roles only a prediction block has:
	//
	//	enorm(embedding of the next token)   RoleNextnENorm
	//	hnorm(the trunk's normed hidden)     RoleNextnHNorm
	//	eh_proj([enorm ; hnorm])             RoleNextnEHProj, 2*NEmbd -> NEmbd
	//	the block, then its own head norm    RoleNextnHeadNorm (optional:
	//	                                     the trunk's output norm otherwise)
	//
	// and the head is the trunk's unless the block carries RoleNextnHead. The
	// blocks serve speculative drafting only; nothing in the trunk reads them.
	NMTP uint32

	// MoEStep interleaves dense and mixture blocks: block il has a mixture
	// feed-forward when (il+1) % MoEStep == 0. Zero or one means every block
	// past NDenseLead is whatever NExpert says. Maverick is 2 (odd blocks are
	// mixtures), Scout 1. It is a period, where NDenseLead is a prefix; a block
	// is a mixture only when both allow it.
	MoEStep uint32

	// Attention temperature: the query is scaled by
	//
	//	1 + AttnTempScale * ln(1 + floor((pos + AttnTempOffset) / AttnTempFloor))
	//
	// before the scores, on the NoPE layers when FlagNoPEGlobal is set
	// (Llama 4) and on every layer otherwise (mistral3's form). AttnTempScale 0
	// means none. Llama 4's constants are 0.1, 8192 and 1. The factor is 1
	// below AttnTempFloor - AttnTempOffset, so a gate needs a small floor.
	AttnTempScale  float32
	AttnTempFloor  uint32
	AttnTempOffset float32

	// AttnScale is the attention score multiplier when the model states one:
	// Granite's attention.scale, and gemma2/gemma3 27B's 1/sqrt(NEmbd/NHead)
	// (query_pre_attn_scalar, not the head width). Zero means 1/sqrt(HeadDim).
	// YaRN's magnitude correction still multiplies whatever this is.
	//
	// ResidualScale multiplies each block's attention and FFN output before
	// its residual add; zero means one. LogitScale divides the logits; zero
	// means one. The three are a second optional tail; see encodeConfig.
	AttnScale     float32
	ResidualScale float32
	LogitScale    float32

	// ClampKQV clamps q, k and v to [-ClampKQV, ClampKQV] after their
	// projections (and biases) and before any norm or rotary: DBRX's
	// clip_qkv. Zero means none. A third optional tail, written only for
	// ArchDBRX; see encodeConfig.
	ClampKQV float32

	// The sliding layers' own attention geometry, when it is not the global
	// layers': Gemma 4 runs its sliding layers at head_dim 256 and its global
	// ones at 512, with different kv head counts on the 31B and 26B and a
	// different rotary width. HeadDim, NKVHead and NRot above are the GLOBAL
	// layers' (the source's key_length and rope.dimension_count); these apply
	// where Config.SWA says a layer slides. Zero means the same as the global
	// field. A fourth optional tail, written only for ArchGemma4; see
	// encodeConfig.
	HeadDimSWA, NKVHeadSWA, NRotSWA uint32
	// Flags2 are the quirks past the 32 bits of Flags, carried in the same
	// tail. See Flag2VNorm.
	Flags2 Flags2
	// Gemma 4's E2B/E4B, in the same tail. NKVShared is how many of the last
	// layers compute no k and v and attend to the history of the last earlier
	// layer of their own kind (KVSource). PLEDim is the width of each layer's
	// slice of the per-layer embeddings (RolePLETokEmbd and friends), zero
	// for none.
	NKVShared, PLEDim uint32

	// DeepSeek V3.2's lightning indexer (ArchDeepseek32): IdxHeads heads of
	// IdxHeadDim score every cached position for each query, and attention
	// reads the IdxTopK best. Zero heads means none. A fifth optional tail,
	// written only for ArchDeepseek32; see encodeConfig.
	IdxHeads, IdxHeadDim, IdxTopK uint32

	// Gemma 3n (ArchGemma3n): AltUp parallel residual streams (zero for one,
	// the ordinary residual), and a gaussian top-k on the gate of the first
	// NSparse blocks' FFNs, which keeps what lies SparseStd standard
	// deviations above the gate row's mean. A sixth optional tail, written
	// only for ArchGemma3n; see encodeConfig.
	AltUp, NSparse uint32
	SparseStd      float32

	// MiniMax Sparse Attention (ArchMiniMaxM3): the indexer scores BLOCKS of
	// IdxBlock positions, each block by its best position, one selection per
	// kv group (IdxHeads == NKVHead), and attention reads the IdxTopK best
	// blocks with the IdxLocal blocks ending at the query's own forced in.
	// Zero is DeepSeek V3.2's per-position selection. A seventh optional
	// tail, written only for ArchMiniMaxM3; see encodeConfig.
	IdxBlock, IdxLocal uint32

	// DeepSeek V4 (ArchDeepseek4): HCMult residual streams mixed by
	// hyper-connections whose stream mixer is HCIters Sinkhorn rounds at
	// HCEps; the attention output projected through OGroups groups of
	// OLoraRank; CompKinds says per block whether it compresses (CompCSA every
	// CompRateCSA positions, with the indexer; CompHCA every CompRateHCA,
	// attending to every entry; CompNone a sliding window alone); the first
	// NHashLayers blocks route by token id. An eighth optional tail, written
	// only for ArchDeepseek4; see encodeConfig.
	HCMult, HCIters          uint32
	HCEps                    float32
	OGroups, OLoraRank       uint32
	CompRateCSA, CompRateHCA uint32
	NHashLayers              uint32
	CompKinds                []CompKind

	// Kimi-K3 (ArchKimiK3): a residual checkpoint every AttnResBlock blocks
	// (zero for none), the routed experts at ExpertLatent (zero for n_embd),
	// the situ activation's two bounds (SituBeta on the gate, SituLinearBeta
	// on up; zero SituBeta is no situ), KDA's decay bound (zero for the
	// softplus decay; a set bound is negative) and the MLA latent norms'
	// epsilon (zero for RMSEps). A ninth optional tail; see encodeConfig.
	AttnResBlock, ExpertLatent uint32
	SituBeta, SituLinearBeta   float32
	KDALowerBound              float32
	LatentNormEps              float32

	// Qwen3.8-Flash-Next (ArchQwen4Exp), an eleventh optional tail; see
	// encodeConfig. HCMult (above) streams mixed through a HCLowRank gate.
	// QSARatio is the QSA block size (zero: the attention blocks attend every
	// position), with IdxHeads/IdxHeadDim/IdxTopK the indexer's heads, width
	// and token budget. NGramBlock is the one-based block that adds the
	// n-gram embedding (zero for none): NGram-grams for n = 2..NGram,
	// NGramHeadsPer heads each, so (NGram-1)*NGramHeadsPer heads of
	// NGramRowDim; head h of a token reads table row
	//
	//	(t0*NGramMul[0] ^ ... ^ t(n-1)*NGramMul[n-1]) mod NGramVocab[h] + NGramOff[h]
	//
	// over the token and its predecessors, unsigned 64-bit, where a
	// predecessor at or before an NGramEOS (or before the sequence) reads as
	// NGramEOS. NGramImage stands in for a picture's rows. NGramConv is the
	// convolution's taps, at a dilation of NGram.
	HCLowRank                    uint32
	QSARatio                     uint32
	NGramBlock, NGram            uint32
	NGramHeadsPer, NGramRowDim   uint32
	NGramConv                    uint32
	NGramEOS, NGramImage         uint32
	NGramMul, NGramOff, NGramVoc []uint64

	// A decision model (decision.go): its readout, the head blocks that end
	// the stack and the token budget of the question and options, and its
	// fitted temperatures. A tenth optional tail; see encodeConfig.
	Decision           DecisionKind
	DecisionBlocks     uint32
	DecisionHeadTokens uint32
	DecisionTemps      []DecisionTemp
	// DecisionHeads is the head blocks' attention head count, which Laya's
	// reference sets as max(1, n_embd/64) whatever the encoder's is.
	DecisionHeads uint32
	// DecisionRouting is Clef's routing blocks, before its DecisionBlocks
	// joint ones, and DecisionEps the head's LayerNorm epsilon (torch's
	// default, which the reference never sets). An eleventh optional tail;
	// see encodeConfig.
	DecisionRouting uint32
	DecisionEps     float32

	Flags Flags
}

// CompKind is a DeepSeek V4 block's attention: a sliding window alone, or the
// window and a compressed history.
type CompKind uint8

const (
	// CompNone is a sliding-window block with no compressor.
	CompNone CompKind = iota
	// CompCSA compresses overlapping windows of 2*CompRateCSA positions every
	// CompRateCSA and keeps the lightning indexer's IdxTopK entries per query.
	CompCSA
	// CompHCA compresses every CompRateHCA positions and attends to every
	// entry.
	CompHCA
)

// Flags2 are boolean quirks past Flags' 32 bits, in the fourth optional tail.
type Flags2 uint32

const (
	// Flag2VNorm RMSNorms each head of v with no weight before it is cached
	// (Gemma 4's v_norm, with_scale=False).
	Flag2VNorm Flags2 = 1 << 0
	// Flag2DenseMoE is Gemma 4's mixture block (26B-A4B): every mixture block
	// also runs a dense MLP, in parallel, and each branch has its own norms.
	// The dense MLP reads RoleFFNNorm's output and is post-normed by
	// RolePostFFNNorm1; the mixture's experts read RoleFFNNorm2's and are
	// post-normed by RolePostFFNNorm2; its router reads an RMSNorm of the
	// residual weighted by RoleRouterNorm; the two branches' sum is post-normed
	// by RolePostFFNNorm before the residual add. The dense MLP is carried as
	// the shared expert's roles (RoleShExpGate/Up/Down), ungated.
	Flag2DenseMoE Flags2 = 1 << 1
	// Flag2DoubleFFN makes the feed-forward of every KV-sharing layer twice
	// NFFN wide (Gemma 4's use_double_wide_mlp).
	Flag2DoubleFFN Flags2 = 1 << 2
	// Flag2SwiGLUClamp makes the gated activation DeepSeek V4's clamped
	// SwiGLU: silu(min(gate, 10)) * clamp(up, -10, 10), in the experts and
	// the shared expert. The limit is the architecture's literal (every
	// reference's default and the published config's).
	Flag2SwiGLUClamp Flags2 = 1 << 3
	// Flag2ExpertSqrtSoftplus makes the router's gate sqrt(softplus(logit))
	// over each expert (DeepSeek V4's scoring_func), in place of the softmax
	// or FlagExpertSigmoid's sigmoid.
	Flag2ExpertSqrtSoftplus Flags2 = 1 << 4
	// Flag2DeltaSigmoidGate gates the delta rule's output norm by sigmoid(z)
	// where qwen3next's gate is silu(z) (Qwen3.8-Flash-Next's
	// output_gate_type).
	Flag2DeltaSigmoidGate Flags2 = 1 << 5
)

// Has reports whether every bit of f2 is set.
func (f Flags2) Has(f2 Flags2) bool { return f&f2 == f2 }

// HeadDimAt, NKVHeadAt and NRotAt are text block b's attention geometry:
// the sliding layers' own where the model has one (HeadDimSWA and friends),
// the model's otherwise.
func (c *Config) HeadDimAt(b int) uint32 { return c.swaOr(b, c.HeadDimSWA, c.HeadDim) }

func (c *Config) NKVHeadAt(b int) uint32 { return c.swaOr(b, c.NKVHeadSWA, c.NKVHead) }

func (c *Config) NRotAt(b int) uint32 { return c.swaOr(b, c.NRotSWA, c.NRot) }

// KVShared reports that text block b computes no k and v of its own and
// attends to KVSource(b)'s history.
func (c *Config) KVShared(b int) bool {
	return c.NKVShared != 0 && uint32(b) >= c.NLayer-c.NKVShared
}

// KVSource is the block whose history block b attends to: b itself, or for a
// KV-sharing block the last block before the shared run of b's own kind
// (sliding or global), as transformers' store_full_length_kv picks it. -1
// when no such block exists, which a reader refuses.
func (c *Config) KVSource(b int) int {
	if !c.KVShared(b) {
		return b
	}
	sliding := c.swaOr(b, 1, 0) == 1
	for j := int(c.NLayer-c.NKVShared) - 1; j >= 0; j-- {
		if (c.swaOr(j, 1, 0) == 1) == sliding {
			return j
		}
	}
	return -1
}

// NFFNAt is text block b's feed-forward width: twice NFFN on a KV-sharing
// block under Flag2DoubleFFN.
func (c *Config) NFFNAt(b int) uint32 {
	if c.Flags2.Has(Flag2DoubleFFN) && c.KVShared(b) {
		return 2 * c.NFFN
	}
	return c.NFFN
}

func (c *Config) swaOr(b int, swa, global uint32) uint32 {
	if swa != 0 && c.SWAWindow > 0 && c.SWAPeriod > 0 && uint32(b)%c.SWAPeriod < c.SWAPeriod-1 {
		return swa
	}
	return global
}

// Flags are the architecture's boolean quirks. Named bits, so a reader that
// does not know a flag skips a bit rather than mis-parsing a record.
type Flags uint32

const (
	// FlagGELU makes the feed-forward GELU-tanh instead of SiLU.
	FlagGELU Flags = 1 << 0
	// FlagTiedEmbd means there is no RoleOutput: the head IS RoleTokenEmbd.
	FlagTiedEmbd Flags = 1 << 1
	// FlagQKNorm RMSNorms q and k before the rotary.
	FlagQKNorm Flags = 1 << 2
	// FlagQKNormWide makes that one norm over the whole projection rather than
	// one per head. Same tensors, same position, different arithmetic.
	FlagQKNormWide Flags = 1 << 3
	// FlagNoExpertNorm skips dividing the top-k mixture weights by their sum.
	// Stated as a skip so the zero value is what most architectures want.
	FlagNoExpertNorm Flags = 1 << 4
	// FlagFusedQKV is retired and must not be set: convert.unfuse splits a
	// fused attn_qkv into RoleAttnQ/K/V (a packed row range has no byte range
	// the device tier can read). The bit stays reserved rather than reused,
	// since a value that changes meaning between versions is what the version
	// number exists to prevent.
	FlagFusedQKV Flags = 1 << 5
	// FlagFusedGateUp is retired for the same reason; see FlagFusedQKV.
	FlagFusedGateUp Flags = 1 << 6
	// FlagRopeNeox pairs (i, i+NRot/2) instead of (2i, 2i+1).
	FlagRopeNeox Flags = 1 << 7
	// FlagAttnOutGate means the q projection is double width and its second
	// half is a gate: out = attn_out * sigmoid(gate). The gate is not a
	// separate tensor on a full-attention layer, hence a flag, not a role.
	FlagAttnOutGate Flags = 1 << 8
	// FlagQuickGELU makes the activation x*sigma(1.702x) (CLIP's) rather than
	// GELU-tanh or SiLU; the three activations are three kernels. Setting both
	// this and FlagGELU is refused at conversion.
	FlagQuickGELU Flags = 1 << 9

	// FlagVisionRope says the tower carries no learned position table and
	// rotates q and k by the patch's (row, column) instead (Qwen2-VL's ViT). It
	// is a flag rather than inferred from a missing table, so a truncated CLIP
	// file is not silently accepted as a position-free tower.
	FlagVisionRope Flags = 1 << 10

	// FlagSwiGLUOAI makes the gated activation gpt-oss's: with g the gate and u
	// the up projection, x = min(g, 7), y = clamp(u, -7, 7), and the output is
	// x*sigma(1.702x)*(y+1). Both constants are the architecture's. Setting
	// both this and FlagGELU is refused at load.
	FlagSwiGLUOAI Flags = 1 << 11

	// FlagYarnExact uses YaRN's correction range as computed rather than
	// floored and ceiled. gpt-oss's own configuration says so ("truncate":
	// false), which matches OpenAI's reference implementation; llama.cpp always
	// rounds. The source GGUF does not carry the choice.
	FlagYarnExact Flags = 1 << 12

	// FlagExpertSigmoid makes the router's gate sigmoid over each expert's
	// logit instead of a softmax over all of them. A sigmoid's top-k weights
	// sum to nothing in particular, so whether to renormalise
	// (FlagNoExpertNorm) and what to scale by (ExpertScale) are separate,
	// orthogonal choices; DeepSeek uses all three.
	FlagExpertSigmoid Flags = 1 << 13

	// FlagNoPosEnc says the attention blocks apply no positional encoding at
	// all. Position still orders the causal mask; it never reaches q or k.
	//
	// It is not NRot == 0: Kimi-Linear's MLA keeps qk_rope_head_dim channels
	// that it concatenates unrotated, and NRot sizes the absorbed key and the
	// cache row. A rotation by position 0 is the identity, so wrongly roping a
	// NoPE model is exact on the first token and wrong after.
	FlagNoPosEnc Flags = 1 << 14

	// FlagNonCausal says every position attends to every position of its
	// input: an encoder's attention (BERT, EmbeddingGemma). With a sliding
	// window it is symmetric, |i - j| <= SWAWindow/2, as llama.cpp's
	// LLAMA_SWA_TYPE_SYMMETRIC and transformers compute.
	FlagNonCausal Flags = 1 << 15

	// The pooling bits say how an embedding is read out of the last residual,
	// and their presence makes a container an embedding model. At most one is
	// set: the mean over every position, the first position (a BERT [CLS]), or
	// the last (qwen3-embedding, whose tokenizer appends the EOS it pools). The
	// pooled vector is then L2-normalised. Bits rather than a field, so the
	// config record's layout does not change.
	FlagPoolMean Flags = 1 << 16
	FlagPoolCLS  Flags = 1 << 17
	FlagPoolLast Flags = 1 << 18

	// FlagSWAChunked makes the sliding-window layers attend within a chunk
	// rather than a window: position p sees keys floor(p/SWAWindow)*SWAWindow
	// through p (Llama 4's iRoPE). The two masks agree only inside the first
	// chunk, so a wrong one is exact on every short prompt.
	FlagSWAChunked Flags = 1 << 19

	// FlagNoPEGlobal says the global layers (those Config.SWA leaves
	// unwindowed) carry no positional encoding, and only the windowed ones
	// rotate: Llama 4's NoPE layers, and cohere2's "rope only on the sliding
	// layers". Per layer, unlike FlagNoPosEnc.
	FlagNoPEGlobal Flags = 1 << 20

	// FlagQKL2Norm RMSNorms each head of q and k with no weight, after the
	// rotary, on the layers that rotate (Llama 4's Llama4TextL2Norm, eps
	// RMSEps). FlagQKNorm instead carries a learned weight and runs before the
	// rotary.
	FlagQKL2Norm Flags = 1 << 21

	// FlagExpertWeightIn applies the routed weight to the expert's input rather
	// than its output (Llama 4): silu(w*g)*(w*u) is not w*silu(g)*u. Since gate
	// and up are linear, the engine scales their outputs by w and sums the down
	// projections at weight one.
	FlagExpertWeightIn Flags = 1 << 22
	// FlagDeltaKeyTiled pairs value head vh of a gated delta rule with key head
	// vh % n_k_heads (tiled) instead of vh / (n_v_heads/n_k_heads) (grouped).
	// It records how the file orders its value heads: llama.cpp's qwen35
	// converter reorders them to tiled (so ggml_repeat can broadcast the keys)
	// and its qwen3next converter does not. Undoing the reorder would split
	// Q4_K super-blocks of ssm_out, so the container states the pairing. With
	// one value head per key head the pairings coincide and the flag is unset.
	FlagDeltaKeyTiled Flags = 1 << 23
	// FlagLayerNorm makes every norm of the text graph a LayerNorm,
	// (x-mean)/sqrt(var+eps)*w + b, where earlier architectures use RMSNorm.
	// The bias is a tensor (RoleAttnNormBias and siblings) and may be absent:
	// a bias-free LayerNorm (command-r) still centres every row, so the flag is
	// independent of the bias.
	FlagLayerNorm Flags = 1 << 24
	// FlagParallel is the parallel residual: attention and the FFN both read
	// the block's input, and both outputs add into it,
	//
	//	x += attn(norm_a(x)) + ffn(norm_f(x))
	//
	// where norm_f is the FFN norm when the block carries one and the same
	// normed row as attention when it does not (phi-2, command-r, falcon-7b).
	// Running it sequentially is fluent.
	FlagParallel Flags = 1 << 25
	// FlagReLU2 makes the feed-forward's activation max(x, 0)^2 (Nemotron's
	// relu2), a fourth kernel beside SiLU, FlagGELU and FlagSwiGLUOAI. Setting
	// it with either other flag is refused at load.
	FlagReLU2 Flags = 1 << 26
	// FlagQKNormPostRope moves the q/k norm after the rotary (Hunyuan). A
	// weighted norm does not commute with a rotation, but the unweighted part
	// does -- a rotation keeps each head's RMS -- so the converter folds the k
	// weight into q's: the k norm is stored as ones and applied before the
	// rotary as FlagQKNorm's is, and q's norm, weighted by w_q*w_k, runs after
	// it. The scores are the model's (q.k is a sum over dimensions, and both
	// weights are per dimension, shared by every head), and the KV cache holds
	// the unweighted normed key on every tier. See convert.foldPostRopeQKNorm.
	FlagQKNormPostRope Flags = 1 << 27
	// FlagVisionBicubic says a tower's pictures are resampled bicubic where
	// its family's processor class defaults to bilinear: LFM2-VL's
	// image_resize_algo, which d1's file states and LFM2-VL's older files do
	// not. A vision-section flag; no text model sets it.
	FlagVisionBicubic Flags = 1 << 28
)

// Pooling is the embedding readout the flags describe, or PoolNone for a model
// that is not an embedding model.
type Pooling uint8

const (
	PoolNone Pooling = iota
	PoolMean
	PoolCLS
	PoolLast
)

func (p Pooling) String() string {
	switch p {
	case PoolMean:
		return "mean"
	case PoolCLS:
		return "cls"
	case PoolLast:
		return "last"
	}
	return "none"
}

// Pooling decodes the pooling bits. More than one is refused by the reader.
func (f Flags) Pooling() Pooling {
	switch {
	case f.Has(FlagPoolMean):
		return PoolMean
	case f.Has(FlagPoolCLS):
		return PoolCLS
	case f.Has(FlagPoolLast):
		return PoolLast
	}
	return PoolNone
}

func (f Flags) Has(b Flags) bool { return f&b != 0 }

// SSMConfig is the recurrent block's geometry.
//
// The names are the container's, not the source's "ssm": what this drives is
// a gated delta rule, and the fields say what each number does.
type SSMConfig struct {
	ConvKernel uint32 // causal convolution width, in positions
	Groups     uint32 // key/query head groups
	Inner      uint32 // the convolved channel count: q + k + v
	StateSize  uint32 // the recurrent state's width per head
	NHeadV     uint32 // value heads, which is also how many decay rates there are
}

// Vision is the tower's description. Present only when Arch is ArchCLIP.
type Vision struct {
	NLayer    uint32
	NEmbd     uint32
	NHead     uint32
	NFFN      uint32
	ImageSize uint32
	PatchSize uint32
	ProjDim   uint32
	Scale     uint32 // pixel-shuffle factor: Scale*Scale patches become one token
	Eps       float32
	Projector Projector
	Mean, Std [3]float32
	Flags     Flags // FlagGELU applies here too
	// MinTiles and MaxTiles bound a tiling preprocessor's crop count: InternVL
	// cuts an image into between MinTiles and MaxTiles ImageSize squares on
	// the grid closest to its aspect ratio, plus a thumbnail. Zero for a tower
	// that takes one square.
	MinTiles, MaxTiles uint32

	// MinPixels and MaxPixels bound a picture a tower reads at its own size
	// when the file states the bounds (Phi-4-reasoning-vision:
	// clip.vision.image_min_pixels and image_max_pixels). Zero otherwise. An
	// optional tail after the windows', written only when set.
	MinPixels, MaxPixels uint32

	// WinPattern and WinSize are Qwen2.5-VL's window attention: every block
	// attends inside windows of WinSize x WinSize pixels except each
	// WinPattern'th, which attends to the whole image (llama.cpp's n_wa_pattern;
	// transformers' fullatt_block_indexes are its multiples, less one). Zero
	// for a tower whose every block sees every patch. An optional tail of the
	// record, written only when set, so every other tower's section is the
	// bytes it was.
	WinPattern, WinSize uint32

	// TileSize and Tolerance are LFM2-VL's tiling (Lfm2VlImageProcessor's
	// tile_size and max_pixels_tolerance): a picture whose rounded size holds
	// more than Tolerance times MaxPixels is cut into TileSize squares. Zero
	// for every other tower; an optional tail after the pixel bounds, written
	// only when set.
	TileSize  uint32
	Tolerance float32

	// Audio is the container's audio tower, nil for none: its blocks follow
	// the vision blocks in the tower's page array (see Audio). The last
	// optional tail of the record.
	Audio *Audio
}

// Projector is how tower embeddings become language-model tokens.
type Projector uint16

const (
	ProjNone     Projector = 0
	ProjIdefics3 Projector = 1
	// ProjMLP is llava's: one linear, a GELU, one linear, and no pixel
	// shuffle. idefics3 groups Scale x Scale patches and runs one matrix;
	// this runs two over each patch on its own.
	ProjMLP Projector = 2
	// ProjQwen2VL is the "qwen2vl_merger": the pixel shuffle and the
	// two-matrix MLP. Scale x Scale neighbouring patches are concatenated into
	// one row of NEmbd*Scale^2, then linear, GELU, linear.
	//
	// llama.cpp permutes the patch grid before block 0 so each 2x2 group lands
	// in consecutive rows; this keeps raster order and groups spatially at the
	// merger. Attention is permutation-equivariant and the rotary takes each
	// patch's own (row, column), so the embeddings are the same with no
	// permutation kernels. The group's internal order must still match:
	// row-major, dy outer and dx inner, as llama.cpp's permute chain
	// {1280,8,8} -> {2560,4,8} -> {2560,4,2,4} -> {2560,2,4,4} spells out.
	ProjQwen2VL Projector = 3
	// ProjGemma3 is Gemma 3's: the SigLIP grid average-pooled Scale x Scale
	// (64x64 patches to 16x16 tokens), an RMSNorm, and one matrix with no
	// bias. Pooling AVERAGES a group where the shuffles above CONCATENATE it,
	// so its matrix reads NEmbd and not NEmbd*Scale^2.
	ProjGemma3 Projector = 4
	// ProjInternVL is InternVL's: the shuffle (the same spatial grouping as
	// qwen2vl_merger, dy outer and dx inner), a LayerNorm with a bias over the
	// grouped row, then linear, GELU, linear.
	ProjInternVL Projector = 5
	// ProjQwen25VL is the "qwen2.5vl_merger": Qwen2-VL's merger behind a
	// different tower -- RMSNorm for LayerNorm, a gated SiLU MLP with biases,
	// and window attention (Vision.WinPattern) -- with the merger's norm an
	// RMSNorm too. The tower runs its patches in window order, each 2x2 merge
	// group in four consecutive rows (transformers' window_index), so the merge
	// groups CONSECUTIVE rows where qwen2vl_merger groups spatial ones, and the
	// merged rows are put back into raster order at the end.
	ProjQwen25VL Projector = 6
	// ProjResampler is MiniCPM-V's (llama.cpp's "resampler"): a fixed set of
	// learned queries cross-attends over the tower's patches, keys carrying a
	// 2-D sin/cos position, so every image or slice becomes the same number of
	// tokens whatever its grid. The tower's own positions are a learned table
	// of PosBuckets x PosBuckets rows indexed by bucketing each patch's
	// fractional (row, column), so a grid of any shape reads it. 32 and not 4:
	// three families were added at once and a projector code is the
	// container's, forever.
	ProjResampler Projector = 32
	// ProjJanus is Janus-Pro's aligner: llava's linear, GELU, linear, with no
	// pixel shuffle -- the same graph as ProjMLP under its own code, because
	// the file names its second matrix mm.1 where llava's is mm.2, its
	// picture is letterboxed rather than squashed, and its markers differ.
	ProjJanus Projector = 33
)

func (p Projector) String() string {
	switch p {
	case ProjIdefics3:
		return "idefics3"
	case ProjMLP:
		return "mlp"
	case ProjQwen2VL:
		return "qwen2vl_merger"
	case ProjGemma3:
		return "gemma3"
	case ProjInternVL:
		return "internvl"
	case ProjQwen25VL:
		return "qwen2.5vl_merger"
	case ProjResampler:
		return "resampler"
	case ProjJanus:
		return "janus_pro"
	}
	if n, ok := visionFamProjNames[p]; ok {
		return n
	}
	if n := visionProjNames[p]; n != "" {
		return n
	}
	return "projector(" + itoa(uint64(p)) + ")"
}

package jlm

import "github.com/jitllm/jitllm/format/quant"

// This file is the container's own vocabulary: the codes that appear in a .jlm
// file and what they mean, which is the whole of what a .jlm parser has to
// read. A container that stored another format's type numbers, tensor names
// and key-value metadata would be renamed, not decoupled; a converter
// translates into these codes, and nothing on the reading side asks what a
// source format called anything.

// Type is how a weight is stored in a container. The codes are this format's
// and the layouts are described here in full, because a reader must not need a
// second document.
//
// Every quantised type is stored transposed and in planes. A tensor of
// nrows x k is written as nsub = k/Sub sub-blocks; within a sub-block the
// payload is Planes[0] bits per weight followed, for the wide types, by
// Planes[1] more; and word w of sub-block s for row r is at word index
// (s*W + w)*nrows + r, W = Sub*(Planes[0]+Planes[1])/32 payload words per
// sub-block per row. Scales follow in their own arrays at the same row
// stride. See Layout for the numbers and write.go for the writer.
type Type uint8

const (
	TypeNone Type = 0

	// Verbatim types: the bytes are the values, row-major, no transpose. A norm,
	// a bias and a router are stored this way.
	TypeF32 Type = 1 // 4 bytes per element, IEEE binary32, little-endian
	TypeF16 Type = 2 // 2 bytes per element, IEEE binary16, little-endian

	// Packed types. Each is: a payload in planes, one f16 super-scale per
	// SuperSub sub-blocks per row, and -- where SuperSub > 1 -- one 8-bit
	// per-sub-block scale (and minimum, for the Min types) packed several to a
	// word. Layout() gives every number a reader needs.
	TypeQ4  Type = 3 // 4-bit, 32-wide sub-block, one scale, a constant bias of 8
	TypeQ8  Type = 4 // 8-bit signed, 32-wide sub-block, one scale
	TypeQ3S Type = 5 // 4-bit payload, 16-wide, 16 sub-blocks per super, bias 4
	TypeQ4S Type = 6 // 4-bit payload, 32-wide, 8 per super, per-sub-block minimum
	TypeQ5S Type = 7 // 8-bit payload, 32-wide, 8 per super, per-sub-block minimum
	TypeQ6S Type = 8 // 4+2-bit planes, 16-wide, 16 per super, bias 32

	// TypeBF16 is verbatim like the two above: the top 16 bits of an IEEE
	// binary32, little-endian. Same 8 exponent bits as f32, so widening is a
	// shift and every value is exactly representable.
	TypeBF16 Type = 9

	// TypeQ5 is Q4's scales with Q6S's second plane: 4+1-bit planes, 32-wide,
	// no per-sub-block scale, a constant bias of 16. It is GGUF's Q5_0, which
	// Unsloth's gemma-3-1b quantization uses for attn_k and attn_v.
	TypeQ5 Type = 10

	// TypeMX4 is OCP MXFP4, GGUF's type 39: a 4-bit plane of e2m1 codes, 32-wide,
	// one scale per sub-block. A code is a float, not an integer, so the weight
	// is scale*(Codes[q] - Bias) -- see TypeLayout.Codes. The scale is the
	// block's E8M0 exponent, 2^(e-128), stored as a byte (see kernels.DSlots).
	TypeMX4 Type = 11

	// TypeQ51 is TypeQ5's planes with a per-sub-block minimum in place of the
	// constant bias: 4+1-bit planes, 32-wide, no per-sub-block scale, and the
	// d word holds the scale in its low half and the negated minimum in its
	// high half, so weight = d*q - (-m). It is GGUF's Q5_1, which published
	// K-quants of Falcon use for attn_qkv (a 4544-wide row is not a multiple of
	// 256).
	TypeQ51 Type = 12

	// TypeIQ4NL is GGUF's IQ4_NL, carried verbatim in its own 18-byte blocks
	// (an f16 scale and 32 nibbles indexing quant.IQ4NLValues). It is not
	// packed: the only tensor that may be this type is a row table
	// (RoleNGramTable), read a row at a time and gathered, never multiplied.
	TypeIQ4NL Type = 13

	typeMax = TypeIQ4NL
)

// Valid reports whether t is a code this format defines. A file carrying
// anything else is refused rather than guessed at.
func (t Type) Valid() bool { return t != TypeNone && t <= typeMax }

func (t Type) String() string {
	switch t {
	case TypeF32:
		return "f32"
	case TypeF16:
		return "f16"
	case TypeQ4:
		return "q4"
	case TypeQ8:
		return "q8"
	case TypeQ3S:
		return "q3s"
	case TypeQ4S:
		return "q4s"
	case TypeQ5S:
		return "q5s"
	case TypeQ6S:
		return "q6s"
	case TypeQ5:
		return "q5"
	case TypeMX4:
		return "mx4"
	case TypeQ51:
		return "q51"
	case TypeBF16:
		return "bf16"
	case TypeIQ4NL:
		return "iq4nl"
	}
	return "type(" + itoa(uint64(t)) + ")"
}

// TypeLayout is everything a reader needs to decode one packed type, so that
// this file plus write.go is a complete specification.
type TypeLayout struct {
	Sub      int     // elements sharing one 8-bit scale
	Planes   [2]int  // payload bits per weight: primary, then secondary (0 if none)
	SuperSub int     // sub-blocks sharing one f16 super-scale; 1 means none
	ScaleOff int     // added to the stored 8-bit scale before use
	Bias     float32 // subtracted as Bias*scale, when HasMin is false
	// HasMin: the format stores its own per-sub-block minimum instead. At
	// SuperSub 1 there is no minimum array, and the d word's high half,
	// negated, is the minimum itself (TypeQ51).
	HasMin bool
	Signed bool // payload bytes are signed
	// Codes, when set, is what a primary code means: the weight is
	// scale*(Codes[q] - Bias) rather than scale*(q - Bias).
	Codes *[16]uint8
}

// Layout returns the on-disk description of a packed type, and ok=false for a
// verbatim one.
func Layout(t Type) (TypeLayout, bool) {
	switch t {
	case TypeQ4:
		return TypeLayout{Sub: 32, Planes: [2]int{4, 0}, SuperSub: 1, Bias: 8}, true
	case TypeQ8:
		return TypeLayout{Sub: 32, Planes: [2]int{8, 0}, SuperSub: 1, Signed: true}, true
	case TypeQ3S:
		return TypeLayout{Sub: 16, Planes: [2]int{4, 0}, SuperSub: 16, ScaleOff: -32, Bias: 4}, true
	case TypeQ4S:
		return TypeLayout{Sub: 32, Planes: [2]int{4, 0}, SuperSub: 8, HasMin: true}, true
	case TypeQ5S:
		// 4+1 two-plane since v15; TestLayoutMatchesThePacker
		// (layout_test.go) holds every field here to the packer.
		return TypeLayout{Sub: 32, Planes: [2]int{4, 1}, SuperSub: 8, HasMin: true}, true
	case TypeQ5:
		return TypeLayout{Sub: 32, Planes: [2]int{4, 1}, SuperSub: 1, Bias: 16}, true
	case TypeQ6S:
		return TypeLayout{Sub: 16, Planes: [2]int{4, 2}, SuperSub: 16, ScaleOff: -128, Bias: 32}, true
	case TypeMX4:
		return TypeLayout{Sub: 32, Planes: [2]int{4, 0}, SuperSub: 1, Bias: 12, Codes: &mx4Codes}, true
	case TypeQ51:
		// HasMin at SuperSub 1: there is no per-sub-block minimum to read, so
		// the minimum is the d word's high half, stored negated.
		return TypeLayout{Sub: 32, Planes: [2]int{4, 1}, SuperSub: 1, HasMin: true}, true
	}
	return TypeLayout{}, false
}

// mx4Codes is the doubled e2m1 values plus 12: {0, 1, 2, 3, 4, 6, 8, 12} for
// codes 0-7 and their negatives for 8-15, biased so every entry is unsigned.
var mx4Codes = [16]uint8{12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0}

// Role says what a tensor is. A container identifies every tensor by (Role,
// Block, Index) and the name it carries is a label nothing reads. A code
// rather than a string, because a string vocabulary ("blk.3.attn_q.weight")
// is a specification written somewhere else.
type Role uint16

const (
	RoleNone Role = 0

	// Model-level.
	RoleTokenEmbd  Role = 1 // [k=n_embd, rows=n_vocab]
	RoleOutput     Role = 2 // the head; absent when the embedding is tied
	RoleOutputNorm Role = 3

	// An encoder's embedding stage (ArchBERT): the token-type table, the
	// learned absolute position table, and the LayerNorm over their sum.
	RoleTokenTypes        Role = 4 // [k=n_embd, rows=n_token_types]; row 0 is "sentence A"
	RolePosEmbd           Role = 5 // [k=n_embd, rows=n_ctx]
	RoleTokenEmbdNorm     Role = 6
	RoleTokenEmbdNormBias Role = 7
	// An embedding model's projection heads after pooling: EmbeddingGemma's
	// two sentence-transformers Dense layers, n_embd -> 3072 -> n_embd, no
	// bias and no activation between them.
	RoleEmbdDense1 Role = 8
	RoleEmbdDense2 Role = 9
	// A decision model's scorer (Laya's): a LayerNorm with bias, a square
	// projection with bias, GELU, and a projection to one score per row with
	// bias, read at the option markers.
	RoleScorerNorm     Role = 10
	RoleScorerNormBias Role = 11
	RoleScorer         Role = 12 // [k=n_embd, rows=n_embd]
	RoleScorerBias     Role = 13
	RoleScorerOut      Role = 14 // [k=n_embd, rows=1]
	RoleScorerOutBias  Role = 15

	// Attention, per block.
	RoleAttnNorm     Role = 16
	RoleAttnQ        Role = 17
	RoleAttnK        Role = 18
	RoleAttnV        Role = 19
	RoleAttnOut      Role = 20
	RoleAttnQKV      Role = 21 // q, k and v as row ranges of one tensor
	RoleAttnQBias    Role = 22
	RoleAttnKBias    Role = 23
	RoleAttnVBias    Role = 24
	RoleAttnOutBias  Role = 25
	RoleAttnQNorm    Role = 26
	RoleAttnKNorm    Role = 27
	RolePostAttnNorm Role = 28
	RoleAttnGate     Role = 29
	// RoleAttnSinks is one learned logit per query head, F32 [NHead]. It joins
	// the head's softmax -- the maximum and the denominator -- after the score
	// scale, and carries no value, so the probabilities over real positions sum
	// to less than one.
	RoleAttnSinks Role = 30

	// Feed-forward, per block.
	RoleFFNNorm     Role = 32
	RoleFFNGate     Role = 33
	RoleFFNUp       Role = 34 // holds gate and up concatenated when fused
	RoleFFNDown     Role = 35
	RolePostFFNNorm Role = 36

	// A post-norm encoder block (ArchBERT): the LayerNorm over x + attention
	// and the one over x + ffn, and the MLP's two biases.
	// Not RolePostAttnNorm/RolePostFFNNorm: gemma's post-norms apply to the
	// sublayer output before it joins the residual, BERT's to the residual
	// sum, replacing it. One role for both would make the reader consult the
	// architecture to learn what a tensor is.
	RoleAttnOutNorm      Role = 37
	RoleAttnOutNormBias  Role = 38
	RoleLayerOutNorm     Role = 39
	RoleLayerOutNormBias Role = 40
	RoleFFNUpBias        Role = 41
	RoleFFNDownBias      Role = 42

	// Mixture-of-experts biases (gpt-oss): the router's, one per expert, added
	// to its logits before the softmax; and each bank's, one row per expert --
	// F32 [NFFNExp, NExpert] for gate and up, [NEmbd, NExpert] for down --
	// added to that expert's projection.
	RoleRouterBias  Role = 44
	RoleExpGateBias Role = 45
	RoleExpUpBias   Role = 46
	RoleExpDownBias Role = 47

	// Mixture of experts, per block. The bank roles hold every expert in one
	// tensor; the indexed roles are one expert each and carry Entry.Index.
	RoleRouter      Role = 48
	RoleExpGateBank Role = 49
	RoleExpUpBank   Role = 50
	RoleExpDownBank Role = 51
	RoleExpGate     Role = 52
	RoleExpUp       Role = 53
	RoleExpDown     Role = 54
	// Shared experts: an always-on feed-forward beside the routed ones. One
	// matrix each, not a bank; the shared gate is a vector whose sigmoid
	// scales the whole contribution.
	RoleShRouter  Role = 55
	RoleShExpGate Role = 56
	RoleShExpUp   Role = 57
	RoleShExpDown Role = 58

	// Rotary tables a model ships rather than derives.
	RoleRopeFactorsLong  Role = 59
	RoleRopeFactorsShort Role = 60
	RoleRopeFreqs        Role = 61

	// A linear-attention block: a causal convolution and a gated delta rule
	// over a recurrent state. See LayerLinearAttn. The codes start at 96
	// because 64..95 are the vision tower's; a role code is an on-disk
	// identifier, and two roles sharing one resolve tensors to the wrong node.
	RoleSSMInProj  Role = 96  // q, k and v in one projection
	RoleSSMBA      Role = 97  // the b and a streams, concatenated
	RoleSSMConv1d  Role = 98  // [kernel, channels], causal, one filter per channel
	RoleSSMA       Role = 99  // per-head log decay
	RoleSSMDtBias  Role = 100 // per-head delta-t bias
	RoleSSMNorm    Role = 101 // the gated RMSNorm over one head
	RoleSSMOut     Role = 102 // the out projection
	RoleSSMOutGate Role = 103 // the output gate

	// Multi-head Latent Attention. See jlm.Config's MLA block for the graph
	// these make; the shapes below are per block. They are new roles, not
	// RoleAttnQ/K/V: MLA has no key projection, and every consumer that reasons
	// about k's shape (KV layout, device upload, row splitter) would be
	// quietly wrong.
	RoleAttnQA      Role = 104 // q down-projection; absent when QLoraRank is 0
	RoleAttnQANorm  Role = 105 // RMSNorm over the q latent
	RoleAttnQB      Role = 106 // q up-projection, to NHead*HeadDim
	RoleAttnKVA     Role = 107 // kv down-projection, to KVLoraRank+QKRopeHeadDim
	RoleAttnKVANorm Role = 108 // RMSNorm over the kv latent's first KVLoraRank
	RoleAttnKB      Role = 109 // absorbed into q's nope half: HeadDim-rope -> KVLoraRank
	RoleAttnVB      Role = 110 // absorbed out of attention: KVLoraRank -> HeadDim

	// RoleExpProbsB is DeepSeek's e_score_correction_bias: one value per
	// expert, added to the gated scores and moving the selection only. It is
	// not RoleRouterBias: gpt-oss's ffn_gate_inp.bias biases the router
	// logits before gating and reaches the weights. llama.cpp keeps two tensors
	// for the same reason; one role would give each model the other's mixture.
	RoleExpProbsB Role = 111

	// RoleAttnKVB is MLA's fused up-projection, which no container carries:
	// convert.unfuseMLA splits it into RoleAttnKB and RoleAttnVB, as
	// RoleAttnQKV is split for phi3. It has a code so identify() can hand the
	// splitter something.
	RoleAttnKVB Role = 112

	// Kimi Delta Attention's two low-rank gates. KDA is the gated delta rule
	// with a per-channel decay, and both driving vectors come off a rank-r
	// bottleneck. They are four roles rather than two folded matrices because
	// the fold would multiply the bytes (NEmbd*qkv against NEmbd*r + r*qkv) on
	// a bandwidth-bound decode. RoleSSMOutGate is not reused for the g pair: it
	// is one whole matrix in qwen3next.
	RoleSSMFA Role = 113 // forget gate down-projection, NEmbd -> r
	RoleSSMFB Role = 114 // forget gate up-projection,   r -> vHeads*kDim
	RoleSSMGA Role = 115 // output gate down-projection, NEmbd -> r
	RoleSSMGB Role = 116 // output gate up-projection,   r -> Inner

	// Vision tower. Its blocks are a second page array (see Role.Vision).
	RoleVPatchEmbd    Role = 64
	RoleVPatchBias    Role = 65
	RoleVPosEmbd      Role = 66
	RoleVPostNorm     Role = 67
	RoleVPostNormBias Role = 68
	RoleVProj         Role = 69
	RoleVProjBias     Role = 70
	RoleVAttnNorm     Role = 71
	RoleVAttnNormBias Role = 72
	RoleVAttnQ        Role = 73
	RoleVAttnK        Role = 74
	RoleVAttnV        Role = 75
	RoleVAttnOut      Role = 76
	RoleVAttnQBias    Role = 77
	RoleVAttnKBias    Role = 78
	RoleVAttnVBias    Role = 79
	RoleVAttnOutBias  Role = 80
	RoleVFFNNorm      Role = 81
	RoleVFFNNormBias  Role = 82
	RoleVFC1          Role = 83
	RoleVFC1Bias      Role = 84
	RoleVFC2          Role = 85
	RoleVFC2Bias      Role = 86
	// 87-91 are CLIP's, and the range stops at 95 because RoleSSMInProj is
	// 96. These are the four things a CLIP tower has and an idefics3 one does
	// not, plus the projector's second matrix.
	//
	// RoleVClassEmbd is one embedding prepended to the patch sequence, so the
	// tower's sequence is Patches()+1 rows and its position table has one more
	// row than it has patches.
	RoleVClassEmbd Role = 87
	// RoleVPreNorm/Bias is a LayerNorm over the whole sequence before block 0.
	// idefics3 has none; CLIP has it and dropping it is a silent wrong answer.
	RoleVPreNorm     Role = 88
	RoleVPreNormBias Role = 89
	// RoleVProj2/Bias is the second matrix of the two-linear MLP projector.
	// The first is RoleVProj, which idefics3 also uses: llava's projector is
	// RoleVProj, a GELU, then this.
	RoleVProj2     Role = 90
	RoleVProj2Bias Role = 91
	// RoleVProjNorm/Bias is a norm the projector applies before its first
	// matrix: gemma3's soft_emb_norm (an RMSNorm, no bias, with Gemma's +1
	// already folded in by llama.cpp's converter) and InternVL's mlp1.0 (a
	// LayerNorm with a bias over the pixel-shuffled row).
	RoleVProjNorm     Role = 92
	RoleVProjNormBias Role = 93
	// RoleVFFNGate/Bias is a gated tower MLP's gate (Qwen2.5-VL's SwiGLU):
	// silu(gate x + b) * (up x + b), then down. Past the text range, since the
	// tower's own range is full; Role.Vision names both ranges.
	RoleVFFNGate     Role = 200
	RoleVFFNGateBias Role = 201

	// The classic block's norm biases and the head's bias (C6). All four
	// are vectors, read once at load (Expanded).
	RoleAttnNormBias   Role = 117
	RoleFFNNormBias    Role = 118
	RoleOutputNormBias Role = 119
	RoleOutputBias     Role = 120
	// RoleAttnQKVBias is a fused q|k|v bias, and like RoleAttnKVB no container
	// carries one: convert.unfuse splits it into the three biases exactly as
	// it splits the fused matrix beside it.
	RoleAttnQKVBias Role = 121
	// RoleAttnNorm2 and its bias are Falcon-40B's second input norm, and like
	// RoleAttnQKVBias no container carries them: convert.falconNorms resolves
	// the pair onto the attention and FFN norms.
	RoleAttnNorm2     Role = 122
	RoleAttnNorm2Bias Role = 123

	// A multi-token-prediction block's own tensors (Config.NMTP). eh_proj takes
	// the normed embedding of the next token concatenated with the normed
	// hidden state, in that order, to one residual row; the two norms are
	// vectors. The head norm and the head are optional: absent, the trunk's
	// output norm and output projection serve, which is what every Qwen3.5
	// file means. RoleNextnEmbd is a prediction block's own embedding table,
	// absent where the trunk's serves.
	RoleNextnEHProj   Role = 124 // [k=2*n_embd, rows=n_embd]
	RoleNextnENorm    Role = 125
	RoleNextnHNorm    Role = 126
	RoleNextnHeadNorm Role = 127
	RoleNextnHead     Role = 128 // [k=n_embd, rows=n_vocab]
	RoleNextnEmbd     Role = 129 // [k=n_embd, rows=n_vocab]

	// MiniCPM-V's resampler projector: a perceiver with one cross-attention
	// layer whose Queries learned queries attend over the tower's patches. Its
	// own range, far from every other, because three vision families were
	// added at once and role codes are the container's, forever. Every one is
	// a dense-block tensor (see Role.Vision).
	RoleVRsQuery      Role = 600 // [k=D, rows=Queries], read once
	RoleVRsKV         Role = 601 // the kv projection, NEmbd -> D, no bias
	RoleVRsLnQ        Role = 602
	RoleVRsLnQBias    Role = 603
	RoleVRsLnKV       Role = 604
	RoleVRsLnKVBias   Role = 605
	RoleVRsQ          Role = 606
	RoleVRsQBias      Role = 607
	RoleVRsK          Role = 608
	RoleVRsKBias      Role = 609
	RoleVRsV          Role = 610
	RoleVRsVBias      Role = 611
	RoleVRsOut        Role = 612
	RoleVRsOutBias    Role = 613
	RoleVRsLnPost     Role = 614
	RoleVRsLnPostBias Role = 615
	RoleVRsProj       Role = 616 // the last matrix, D -> D, no bias
	// Mamba-2's two per-block vectors beside the delta rule's (LayerSSD): the
	// skip D, one per head, and the convolution's bias, one per channel.
	// Their codes are apart from the run above so a concurrent addition
	// counting up from it does not collide.
	RoleSSMD        Role = 400
	RoleSSMConvBias Role = 401
	// LFM2's in_proj, split at conversion into its three outputs: B and C,
	// the two gates around the short convolution, and x, what it convolves.
	RoleSCB Role = 402
	RoleSCC Role = 403
	RoleSCX Role = 404
	// Mamba-1's (LayerMamba1): x_proj split at conversion into its three
	// outputs -- the dt bottleneck, B and C -- dt_proj back up to the
	// channels, and Jamba's three RMSNorm weights (FalconMamba's weightless
	// norms are written as ones). Apart from the run above for the same
	// reason as it.
	RoleSSMXDt    Role = 430
	RoleSSMXB     Role = 431
	RoleSSMXC     Role = 432
	RoleSSMDtProj Role = 433
	RoleSSMDtNorm Role = 434
	RoleSSMBNorm  Role = 435
	RoleSSMCNorm  Role = 436
)

// roleNames is for diagnostics only. Nothing resolves a tensor through it.
var roleNames = map[Role]string{
	RoleTokenEmbd: "token_embd", RoleOutput: "output", RoleOutputNorm: "output_norm",
	RoleAttnNorm: "attn_norm", RoleAttnQ: "attn_q", RoleAttnK: "attn_k", RoleAttnV: "attn_v",
	RoleAttnOut: "attn_out", RoleAttnQKV: "attn_qkv", RoleAttnQBias: "attn_q.bias",
	RoleAttnKBias: "attn_k.bias", RoleAttnVBias: "attn_v.bias", RoleAttnOutBias: "attn_out.bias",
	RoleAttnQNorm: "attn_q_norm", RoleAttnKNorm: "attn_k_norm", RolePostAttnNorm: "post_attn_norm",
	RoleFFNNorm: "ffn_norm", RoleFFNGate: "ffn_gate", RoleFFNUp: "ffn_up", RoleFFNDown: "ffn_down",
	RolePostFFNNorm: "post_ffn_norm", RoleRouter: "router",
	RoleExpGateBank: "exp_gate_bank", RoleExpUpBank: "exp_up_bank", RoleExpDownBank: "exp_down_bank",
	RoleExpGate: "exp_gate", RoleExpUp: "exp_up", RoleExpDown: "exp_down",
	RoleAttnGate: "attn_gate", RoleShRouter: "sh_router", RoleAttnSinks: "attn_sinks",
	RoleRouterBias: "router.bias", RoleExpGateBias: "exp_gate_bank.bias",
	RoleExpUpBias: "exp_up_bank.bias", RoleExpDownBias: "exp_down_bank.bias",
	RoleShExpGate: "sh_exp_gate", RoleShExpUp: "sh_exp_up", RoleShExpDown: "sh_exp_down",
	RoleSSMBA: "ssm_ba", RoleSSMConv1d: "ssm_conv1d",
	RoleSSMA: "ssm_a", RoleSSMDtBias: "ssm_dt_bias", RoleSSMNorm: "ssm_norm",
	RoleSSMOut: "ssm_out",
	RoleSSMFA:  "ssm_f_a", RoleSSMFB: "ssm_f_b",
	RoleSSMGA: "ssm_g_a", RoleSSMGB: "ssm_g_b",
	RoleAttnQA: "attn_q_a", RoleAttnQANorm: "attn_q_a_norm", RoleAttnQB: "attn_q_b",
	RoleAttnKVA: "attn_kv_a", RoleAttnKVANorm: "attn_kv_a_norm",
	RoleAttnKB: "attn_k_b", RoleAttnVB: "attn_v_b",
	// Named although no container carries it: every diagnostic between
	// identify() and the split prints it.
	RoleAttnKVB:         "attn_kv_b",
	RoleExpProbsB:       "exp_probs_b",
	RoleRopeFactorsLong: "rope_factors_long", RoleRopeFactorsShort: "rope_factors_short",
	RoleRopeFreqs:  "rope_freqs",
	RoleVPatchEmbd: "v_patch_embd", RoleVPatchBias: "v_patch_bias", RoleVPosEmbd: "v_pos_embd",
	RoleVPostNorm: "v_post_norm", RoleVPostNormBias: "v_post_norm.bias",
	RoleVProj: "v_proj", RoleVProjBias: "v_proj.bias",
	RoleVAttnNorm: "v_attn_norm", RoleVAttnNormBias: "v_attn_norm.bias",
	RoleVAttnQ: "v_attn_q", RoleVAttnK: "v_attn_k", RoleVAttnV: "v_attn_v", RoleVAttnOut: "v_attn_out",
	RoleVAttnQBias: "v_attn_q.bias", RoleVAttnKBias: "v_attn_k.bias",
	RoleVAttnVBias: "v_attn_v.bias", RoleVAttnOutBias: "v_attn_out.bias",
	RoleVFFNNorm: "v_ffn_norm", RoleVFFNNormBias: "v_ffn_norm.bias",
	RoleVFC1: "v_fc1", RoleVFC1Bias: "v_fc1.bias", RoleVFC2: "v_fc2", RoleVFC2Bias: "v_fc2.bias",
	RoleVClassEmbd: "v_class_embd", RoleVPreNorm: "v_pre_norm",
	RoleVPreNormBias: "v_pre_norm.bias",
	RoleVProj2:       "v_proj2", RoleVProj2Bias: "v_proj2.bias",
	RoleVProjNorm: "v_proj_norm", RoleVProjNormBias: "v_proj_norm.bias",
	RoleVFFNGate: "v_ffn_gate", RoleVFFNGateBias: "v_ffn_gate.bias",
	RoleTokenTypes: "token_types", RolePosEmbd: "position_embd",
	RoleTokenEmbdNorm: "token_embd_norm", RoleTokenEmbdNormBias: "token_embd_norm.bias",
	RoleEmbdDense1: "embd_dense1", RoleEmbdDense2: "embd_dense2",
	RoleScorerNorm: "scorer_norm", RoleScorerNormBias: "scorer_norm.bias",
	RoleScorer: "scorer", RoleScorerBias: "scorer.bias",
	RoleScorerOut: "scorer_out", RoleScorerOutBias: "scorer_out.bias",
	RoleAttnOutNorm: "attn_out_norm", RoleAttnOutNormBias: "attn_out_norm.bias",
	RoleLayerOutNorm: "layer_out_norm", RoleLayerOutNormBias: "layer_out_norm.bias",
	RoleFFNUpBias: "ffn_up.bias", RoleFFNDownBias: "ffn_down.bias",
	RoleAttnNormBias: "attn_norm.bias", RoleFFNNormBias: "ffn_norm.bias",
	RoleOutputNormBias: "output_norm.bias", RoleOutputBias: "output.bias",
	RoleAttnQKVBias: "attn_qkv.bias",
	RoleAttnNorm2:   "attn_norm_2", RoleAttnNorm2Bias: "attn_norm_2.bias",
	RoleNextnEHProj: "nextn.eh_proj", RoleNextnENorm: "nextn.enorm", RoleNextnHNorm: "nextn.hnorm",
	RoleNextnHeadNorm: "nextn.shared_head_norm", RoleNextnHead: "nextn.shared_head_head",
	RoleNextnEmbd: "nextn.embed_tokens",
	RoleVRsQuery:  "v_rs_query", RoleVRsKV: "v_rs_kv",
	RoleVRsLnQ: "v_rs_ln_q", RoleVRsLnQBias: "v_rs_ln_q.bias",
	RoleVRsLnKV: "v_rs_ln_kv", RoleVRsLnKVBias: "v_rs_ln_kv.bias",
	RoleVRsQ: "v_rs_q", RoleVRsQBias: "v_rs_q.bias", RoleVRsK: "v_rs_k", RoleVRsKBias: "v_rs_k.bias",
	RoleVRsV: "v_rs_v", RoleVRsVBias: "v_rs_v.bias", RoleVRsOut: "v_rs_out", RoleVRsOutBias: "v_rs_out.bias",
	RoleVRsLnPost: "v_rs_ln_post", RoleVRsLnPostBias: "v_rs_ln_post.bias", RoleVRsProj: "v_rs_proj",
	RoleSSMD: "ssm_d", RoleSSMConvBias: "ssm_conv1d.bias",
	RoleSCB: "shortconv.in_proj.b", RoleSCC: "shortconv.in_proj.c", RoleSCX: "shortconv.in_proj.x",
	RoleSSMXDt: "ssm_x.dt", RoleSSMXB: "ssm_x.b", RoleSSMXC: "ssm_x.c", RoleSSMDtProj: "ssm_dt",
	RoleSSMDtNorm: "ssm_dt_norm", RoleSSMBNorm: "ssm_b_norm", RoleSSMCNorm: "ssm_c_norm",
}

func (r Role) String() string {
	if s, ok := roleNames[r]; ok {
		return s
	}
	return "role(" + itoa(uint64(r)) + ")"
}

// Valid reports whether r is a role this format defines.
func (r Role) Valid() bool { _, ok := roleNames[r]; return ok }

func itoa(v uint64) string {
	if v == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}

// SourceType maps a container type onto the engine's quantisation enum.
//
// It is a mapping, not an identity: the container's codes are its own and
// quant.Type is the engine's internal enum, so either can renumber or grow
// without breaking the other.
func SourceType(t Type) (quant.Type, bool) {
	switch t {
	case TypeF32:
		return quant.F32, true
	case TypeF16:
		return quant.F16, true
	case TypeBF16:
		return quant.BF16, true
	case TypeQ4:
		return quant.Q4_0, true
	case TypeQ5:
		return quant.Q5_0, true
	case TypeQ51:
		return quant.Q5_1, true
	case TypeMX4:
		return quant.MXFP4, true
	case TypeQ8:
		return quant.Q8_0, true
	case TypeQ3S:
		return quant.Q3_K, true
	case TypeQ4S:
		return quant.Q4_K, true
	case TypeQ5S:
		return quant.Q5_K, true
	case TypeQ6S:
		return quant.Q6_K, true
	case TypeIQ4NL:
		return quant.IQ4_NL, true
	}
	return 0, false
}

// TypeOf is SourceType's inverse, for a converter.
func TypeOf(t quant.Type) (Type, bool) {
	switch t {
	case quant.F32:
		return TypeF32, true
	case quant.F16:
		return TypeF16, true
	case quant.BF16:
		return TypeBF16, true
	case quant.Q4_0:
		return TypeQ4, true
	case quant.Q5_0:
		return TypeQ5, true
	case quant.Q5_1:
		return TypeQ51, true
	case quant.MXFP4:
		return TypeMX4, true
	case quant.Q8_0:
		return TypeQ8, true
	case quant.Q3_K:
		return TypeQ3S, true
	case quant.Q4_K:
		return TypeQ4S, true
	case quant.Q5_K:
		return TypeQ5S, true
	case quant.Q6_K:
		return TypeQ6S, true
	case quant.IQ4_NL:
		return TypeIQ4NL, true
	}
	return 0, false
}

// Vision reports whether this role belongs to the vision tower, which decides
// which page array the tensor lives in: one container, two arrays. There are
// two arrays because one page size is wrong in both directions (SmolVLM's tower
// blocks are twice its text blocks; a 30B's text blocks are a hundred times
// any tower's), and two static sizes keep addressing an index.
func (r Role) Vision() bool {
	return r >= RoleVPatchEmbd && r <= RoleVProjNormBias || r >= RoleVRsQuery && r <= RoleVRsProj ||
		r == RoleVFFNGate || r == RoleVFFNGateBias || visionFamRole(r) || visionFamilyRole(r)
}

// Expanded says a tensor is read once at load and copied out, rather than
// driven in place as a matrix-vector product. Only these may leave a block's
// page: a reader dequantizes them into its own slice (model.vec), so where
// they live constrains nothing later. A weight is read in place and must stay
// a page, the unit that is placed and relocated.
//
// The MoE router is a weight and must not be here: routing data must not be
// evicted, but must stay relocatable, which is a residency policy over pages,
// not a layout decision. ssm_conv1d is here although 2-D: a per-channel
// window by use. Shape is not the test.
func (r Role) Expanded() bool {
	switch r {
	case RoleOutputNorm, RoleAttnNorm, RoleFFNNorm,
		RoleAttnQNorm, RoleAttnKNorm, RolePostAttnNorm, RolePostFFNNorm,
		RoleAttnQBias, RoleAttnKBias, RoleAttnVBias, RoleAttnOutBias,
		RoleRopeFreqs, RoleRopeFactorsLong, RoleRopeFactorsShort,
		RoleShRouter,
		RoleAttnSinks, RoleRouterBias, RoleExpGateBias, RoleExpUpBias, RoleExpDownBias,
		RoleSSMConv1d, RoleSSMA, RoleSSMDtBias, RoleSSMNorm, RoleSSMD, RoleSSMConvBias,
		RoleSSMDtNorm, RoleSSMBNorm, RoleSSMCNorm,
		// MLA's two latent norms are per-block vectors like every other norm;
		// the five MLA matrices are the block's weight and belong in its page.
		RoleAttnQANorm, RoleAttnKVANorm,
		// One float per expert, read once at load like every other bias here.
		RoleExpProbsB,
		// The tower's per-block vectors too: buildTower reads every norm and
		// bias at load, so leaving them in the pages forced the whole
		// container to be read before one LayerNorm.
		RoleVAttnNorm, RoleVAttnNormBias, RoleVFFNNorm, RoleVFFNNormBias,
		RoleVAttnQBias, RoleVAttnKBias, RoleVAttnVBias, RoleVAttnOutBias,
		RoleVFC1Bias, RoleVFC2Bias, RoleVFFNGateBias,
		// CLIP's three extra vectors, read once and copied out. All are
		// DenseBlock anyway; stating their kind keeps the predicate honest.
		RoleVClassEmbd, RoleVPreNorm, RoleVPreNormBias,
		// The projector's norm, a dense-block vector like the three above.
		RoleVProjNorm, RoleVProjNormBias,
		// The resampler's learned queries, norms and biases: read once at
		// load. Its five matrices are weights and stay in place.
		RoleVRsQuery, RoleVRsLnQ, RoleVRsLnQBias, RoleVRsLnKV, RoleVRsLnKVBias,
		RoleVRsQBias, RoleVRsKBias, RoleVRsVBias, RoleVRsOutBias,
		RoleVRsLnPost, RoleVRsLnPostBias,
		// An encoder block's post-norms and MLP biases: vectors, read once.
		RoleAttnOutNorm, RoleAttnOutNormBias, RoleLayerOutNorm, RoleLayerOutNormBias,
		RoleFFNUpBias, RoleFFNDownBias,
		// And its embedding stage's vectors. They are dense-block tensors and
		// always resident anyway; stating their kind keeps the predicate honest.
		RoleTokenTypes, RoleTokenEmbdNorm, RoleTokenEmbdNormBias,
		// A decision scorer's norm and biases.
		RoleScorerNorm, RoleScorerNormBias, RoleScorerBias, RoleScorerOutBias,
		// The classic block's norm biases (C6), vectors like the norms beside
		// them. The head's bias is a dense-block tensor and resident anyway.
		RoleAttnNormBias, RoleFFNNormBias, RoleOutputNormBias, RoleOutputBias,
		// A prediction block's three norms, vectors like every norm here.
		RoleNextnENorm, RoleNextnHNorm, RoleNextnHeadNorm:
		return true
	}
	return r == RoleXIELU || flagshipExpanded(r) || visionFamExpanded(r) || visionFamilyExpanded(r)
}

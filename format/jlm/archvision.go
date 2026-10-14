package jlm

import "fmt"

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

var visionArchNames = map[Arch]string{
	ArchQwen3VL: "qwen3vl", ArchQwen3VLMoE: "qwen3vlmoe", ArchGLM4: "glm4", ArchGLM4VMoE: "glm4vmoe",
	ArchHunyuanVL: "hunyuan_vl",
}

// XDRope reports whether a's M-RoPE assigns its axes per element of the
// rotary rather than per pair (ArchHunyuanVL).
func (a Arch) XDRope() bool { return a == ArchHunyuanVL }

// InterleavedMRope reports whether a's M-RoPE sections are interleaved over
// a head's pairs rather than laid out as contiguous runs (ArchQwen2VL's).
// Qwen3.5 (ArchQwen3Next with sections) is Qwen3-VL's split under its own
// tower; qwen3next itself carries no sections and has no runs.
func (a Arch) InterleavedMRope() bool {
	return a == ArchQwen3VL || a == ArchQwen3VLMoE || a == ArchQwen3Next || a == ArchQwen4Exp
}

const (
	// ProjQwen3VL is the "qwen3vl_merger": Qwen2-VL's 2x2 merger (a
	// LayerNorm with a bias over each patch row, the shuffle, linear, GELU,
	// linear) behind a tower that adds a LEARNED position table, bilinearly
	// resampled to the picture's grid (align_corners), on top of the 2-D
	// rotary; and deepstack: after each tapped block (RoleVDs* at that index)
	// a merger of its own -- the shuffle first, then a LayerNorm over the
	// shuffled row -- emits rows that add into the text model's image rows
	// after its first blocks, tap k into block k.
	ProjQwen3VL Projector = 48

	// ProjGLM4V is GLM-4.1V's, 4.5V's and 4.6V's ("glm4v"): an RMSNorm,
	// gated-SiLU tower with the 2-D rotary and a learned table resampled
	// BICUBIC to the picture's grid, an RMSNorm after the patch embedding
	// (RoleVEmbdNorm); then a 2x2 convolution merges (RoleVMergeConv, laid
	// out k first over the shuffled row), and linear, LayerNorm, GELU and a
	// gated SiLU MLP (RoleVProjGate/Up/Down) make the text model's rows.
	ProjGLM4V Projector = 49

	// ProjKimiVL is Kimi-VL's ("kimivl"): MoonViT -- CLIP's LayerNorm,
	// GELU-tanh block with biased q/k/v, a learned table resampled bicubic
	// to the grid, and the 2-D rotary in complex pairs (pair 2j by the
	// column, 2j+1 by the row) -- then a LayerNorm on each patch
	// (RoleVProjInNorm) BEFORE the 2x2 shuffle, linear, GELU, linear.
	ProjKimiVL Projector = 50

	// ProjHunyuanVL is HunyuanVL's ("hunyuanvl"): a LayerNorm, GELU ViT with
	// biased q/k/v and no rotary, a learned table resampled bilinear
	// (align_corners False) to the grid and added by ROW -- the processor
	// hands the patches over in merge-group order and the table and the
	// merger both read the rows as a raster, as the reference does; then an
	// RMSNorm (RoleVProjInNorm, no bias), a 2x2 convolution (RoleVMergeConv),
	// GELU, a 1x1 convolution (RoleVProj), a newline row after every merged
	// row (RoleVImgNewline), a linear (RoleVProj2), the begin and end rows
	// (RoleVImgBegin/End) around the picture, and an RMSNorm
	// (RoleVProjNorm). A picture is H*(W+1)+2 rows.
	ProjHunyuanVL Projector = 51
)

var visionProjNames = map[Projector]string{
	ProjQwen3VL: "qwen3vl_merger", ProjGLM4V: "glm4v", ProjKimiVL: "kimivl", ProjHunyuanVL: "hunyuanvl",
}

// The group's roles. A deepstack merger is DenseBlock with Index the tower
// block it taps: the block page would be the right home for it, but at a
// static page size it would set every vision page's size (a 2x2 merger is
// three times a Qwen3-VL-2B tower block), so it lives with the main merger.
const (
	RoleVDsNorm     Role = 700 // [4*NEmbd], over the shuffled row
	RoleVDsNormBias Role = 701
	RoleVDsFC1      Role = 702 // [k=4*NEmbd, rows=4*NEmbd]
	RoleVDsFC1Bias  Role = 703
	RoleVDsFC2      Role = 704 // [k=4*NEmbd, rows=ProjDim]
	RoleVDsFC2Bias  Role = 705

	// GLM-4.xV's (ProjGLM4V). RoleVEmbdNorm is the RMSNorm after the patch
	// embedding (v.norm_embd); RoleVMergeConv the 2x2 merging convolution as a
	// matrix [k=Scale^2*NEmbd over (dy, dx, channel), rows=ProjDim];
	// RoleVProjGate/Up/Down the projector's gated MLP, ProjDim to its
	// context width and back.
	RoleVEmbdNorm      Role = 706
	RoleVMergeConv     Role = 707
	RoleVMergeConvBias Role = 708
	RoleVProjGate      Role = 709
	RoleVProjGateBias  Role = 710
	RoleVProjUp        Role = 711
	RoleVProjUpBias    Role = 712
	RoleVProjDown      Role = 713
	RoleVProjDownBias  Role = 714

	// Kimi-VL's projector LayerNorm, over each patch's NEmbd before the
	// shuffle (InternVL's RoleVProjNorm is over the shuffled row).
	RoleVProjInNorm     Role = 715
	RoleVProjInNormBias Role = 716

	// HunyuanVL's merger rows: the newline after each merged row (the 1x1
	// convolution's width), and the begin and end rows around the picture
	// (the text model's width).
	RoleVImgNewline Role = 717
	RoleVImgBegin   Role = 718
	RoleVImgEnd     Role = 719
)

var visionRoleNames = map[Role]string{
	RoleVDsNorm: "v_ds_norm", RoleVDsNormBias: "v_ds_norm.bias",
	RoleVDsFC1: "v_ds_fc1", RoleVDsFC1Bias: "v_ds_fc1.bias",
	RoleVDsFC2: "v_ds_fc2", RoleVDsFC2Bias: "v_ds_fc2.bias",
	RoleVEmbdNorm: "v_embd_norm", RoleVMergeConv: "v_merge_conv", RoleVMergeConvBias: "v_merge_conv.bias",
	RoleVProjGate: "v_proj_gate", RoleVProjGateBias: "v_proj_gate.bias",
	RoleVProjUp: "v_proj_up", RoleVProjUpBias: "v_proj_up.bias",
	RoleVProjDown: "v_proj_down", RoleVProjDownBias: "v_proj_down.bias",
	RoleVProjInNorm: "v_proj_in_norm", RoleVProjInNormBias: "v_proj_in_norm.bias",
	RoleVImgNewline: "v_img_newline", RoleVImgBegin: "v_img_begin", RoleVImgEnd: "v_img_end",
}

func init() {
	for r, n := range visionRoleNames {
		// Another group taking the same number would rename its role here and
		// read one tensor as the other.
		if was, ok := roleNames[r]; ok {
			panic(fmt.Sprintf("jlm: role %d is both %q and %q", r, was, n))
		}
		roleNames[r] = n
	}
}

// visionFamilyRole is Role.Vision for the group's roles: every one is a
// tower's.
func visionFamilyRole(r Role) bool { return r >= RoleVDsNorm && r <= RoleVImgEnd }

// visionFamilyExpanded is Role.Expanded for the group's roles: the vectors a
// reader copies out at load.
func visionFamilyExpanded(r Role) bool {
	switch r {
	case RoleVDsNorm, RoleVDsNormBias, RoleVDsFC1Bias, RoleVDsFC2Bias,
		RoleVEmbdNorm, RoleVMergeConvBias, RoleVProjGateBias, RoleVProjUpBias, RoleVProjDownBias,
		RoleVProjInNorm, RoleVProjInNormBias, RoleVImgNewline, RoleVImgBegin, RoleVImgEnd:
		return true
	}
	return false
}

package jlm

// The vision families whose codes are taken from the TOP of the code space
// down: Pixtral / Mistral 3, Llama 4, Gemma 4, Gemma 3n and
// Phi-4-reasoning-vision. Another set of families takes the next free codes
// up from the existing ones at the same time, and a code is the container's
// forever, so the two sets grow towards each other and cannot meet by
// accident: a shared number would read one family's tensor as the other's.

const (
	// ProjPixtral is llama.cpp's "pixtral": Mistral's ViT (RMSNorm, a gated
	// SiLU MLP, no biases, a 2-D rotary whose two halves take the even and the
	// odd frequencies of one sequence, interleaved pairs) at the picture's own
	// size, then -- for Mistral Small 3.1 and Ministral 3 -- an RMSNorm over
	// each patch, a learned merge of Scale x Scale patches and llava's linear,
	// GELU, linear; Pixtral-12B has no merge (Scale 1). The rows of each
	// merged grid row are followed by the [IMG_BREAK] token's embedding
	// (RoleVImgBreak), but the last.
	ProjPixtral Projector = 65535
	// ProjLlama4 is Llama 4's: a CLIP-style ViT (LayerNorms with biases,
	// biased q/k/v/o, a GELU MLP) whose class token goes LAST with the
	// position table's last row, a 2-D rotary on adjacent pairs beside the
	// table, then the 2x2 pixel shuffle, two matrices each followed by a GELU
	// (Llama4VisionMLP2) and the projector's matrix (RoleVProj3). A picture is
	// cut into up to MaxTiles ImageSize tiles on the best-fitting canvas and a
	// global one.
	ProjLlama4 Projector = 65534
	// ProjGemma4V is Gemma 4's: an RMSNorm ViT with sandwich norms, a
	// per-head q/k norm and a weightless v norm, attention at scale one, a
	// gated GELU MLP, a 2-D rotary per half of a head, two learned position
	// tables (column and row), every matrix's input and output clamped
	// (RoleVClamp); then a Scale x Scale average pool, sqrt(NEmbd), the
	// standardisation (RoleVStdBias/Scale) where present, an unweighted
	// RMSNorm and one matrix. A picture is read at its own size within a soft
	// token budget.
	ProjGemma4V Projector = 65533
	// ProjPhi4 is llama.cpp's "phi4": Phi-4-reasoning-vision-15B (NOT
	// Phi-4-multimodal-instruct, which no mmproj exists for). SigLIP2's NaFlex
	// tower at the picture's own size, between MinPixels and MaxPixels, its
	// square position table resized to each grid; the reference reads
	// hidden_states[-2], so the file carries one block fewer and no
	// post-LayerNorm; then llava's two matrices and a GELU over every patch.
	ProjPhi4 Projector = 65532
	// ProjGemma3nV is Gemma 3n's: timm's MobileNet-V5 encoder, a
	// convolutional tower whose blocks change shape from one to the next --
	// a stride-2 stem, edge residuals (a full 3x3 convolution), universal
	// inverted residuals (depthwise convolutions around two pointwise ones)
	// and multi-query attention whose keys and values are a stride-2
	// depthwise downsample -- then the multi-scale fusion adapter (the last
	// two stages' outputs at one resolution, an inverted-residual FFN, an
	// average pool to 16x16 and an RMSNorm) and the embedder (sqrt(NEmbd), an
	// RMSNorm, one matrix and an unweighted RMSNorm). Rows are positions and
	// channels are a row; every convolution weight is tap-major, (ky, kx, c).
	ProjGemma3nV Projector = 65531
	// ProjLFM2 is llama.cpp's "lfm2": LFM2-VL's and d1's. SigLIP2's NaFlex
	// tower (Phi-4-reasoning-vision's, with its post-LayerNorm) at each
	// piece's own grid, its square position table resized to the grid; then
	// the Scale x Scale pixel unshuffle (InternVL's grouping, dy outer and dx
	// inner), a LayerNorm over the grouped row when the file carries one
	// (RoleVProjNorm; d1's does not), and linear, GELU, linear. A picture
	// past the token budget is cut into between MinTiles and MaxTiles
	// TileSize squares plus a thumbnail (Vision.TileSize, Vision.Tolerance).
	ProjLFM2 Projector = 65530
)

const (
	// RoleVMerge is Mistral 3's patch merger: one matrix, NEmbd*Scale^2 ->
	// NEmbd, no bias, over the Scale x Scale patches the shuffle groups. The
	// source reads them channel-major (torch's unfold); the converter permutes
	// its columns into the shuffle's order, patch-major, dy outer and dx inner.
	RoleVMerge Role = 65535
	// RoleVImgBreak is the [IMG_BREAK] token's embedding, one text-width row
	// the head writes after every merged grid row but the last (Pixtral).
	// llama.cpp's converter copies it out of the text model's table into the
	// mmproj, unquantized, which is the row the reference looks up.
	RoleVImgBreak Role = 65534
	// RoleVProj3 is a projector's third matrix, after RoleVProj and
	// RoleVProj2: Llama 4's multi_modal_projector, after its adapter's two.
	RoleVProj3 Role = 65533
	// Gemma 4's tower block: the norms after the attention and after the
	// MLP, and the per-head q and k norms, one HeadDim vector each.
	RoleVPostAttnNorm Role = 65532
	RoleVPostFFNNorm  Role = 65531
	RoleVAttnQNorm    Role = 65530
	RoleVAttnKNorm    Role = 65529
	// RoleVClamp is a block's clipped linears' bounds, 28 floats: for q, k,
	// v, the attention output, the MLP's gate, up and down in that order,
	// the input's minimum and maximum and the output's (Gemma 4's
	// Gemma4ClippableLinear). A bound the file does not state is infinite.
	RoleVClamp Role = 65528
	// RoleVStdBias and RoleVStdScale standardise the pooled rows,
	// (x - bias) * scale (Gemma 4's standardize).
	RoleVStdBias  Role = 65527
	RoleVStdScale Role = 65526

	// Gemma 3n's MobileNet-V5 (ProjGemma3nV). Each convolution's weight is
	// tap-major: a full or pointwise one a matrix whose k is (ky, kx, c), a
	// depthwise filter K*K rows of channels. Every norm is an RMSNorm over a
	// position's channels (timm's RmsNorm2d); a block's layer scale is folded
	// into the norm or matrix it follows.
	RoleVMNStem     Role = 65525 // the stem's 3x3 stride-2 convolution
	RoleVMNStemBias Role = 65524
	RoleVMNStemNorm Role = 65523
	// An edge residual: a full convolution, a norm and GELU, a pointwise
	// one, a norm.
	RoleVMNConvExp Role = 65522
	RoleVMNNorm1   Role = 65521
	RoleVMNConvPwl Role = 65520
	RoleVMNNorm2   Role = 65519
	// A universal inverted residual: an optional depthwise start and its
	// norm, a pointwise expansion, a norm and GELU, an optional depthwise
	// middle with its norm and GELU, a pointwise projection and its norm.
	RoleVMNDwStart     Role = 65518
	RoleVMNDwStartNorm Role = 65517
	RoleVMNPwExp       Role = 65516
	RoleVMNPwExpNorm   Role = 65515
	RoleVMNDwMid       Role = 65514
	RoleVMNDwMidNorm   Role = 65513
	RoleVMNPwProj      Role = 65512
	RoleVMNPwProjNorm  Role = 65511
	// Multi-query attention's key and value downsample, a depthwise filter
	// and a norm each; its projections are RoleVAttnQ/K/V/Out and its block
	// norm RoleVAttnNorm.
	RoleVMNKDown     Role = 65510
	RoleVMNKDownNorm Role = 65509
	RoleVMNVDown     Role = 65508
	RoleVMNVDownNorm Role = 65507
	// RoleVMNGeom is a block's place, {stage, index in the stage}: the first
	// block of a stage downsamples, and the fusion adapter reads the last
	// blocks of the last two stages.
	RoleVMNGeom Role = 65506
	// The fusion adapter's FFN and its closing norm. The embedder's norm
	// before its matrix is gemma3's (RoleVProjNorm, then RoleVProj).
	RoleVMNFusionExp      Role = 65505
	RoleVMNFusionExpNorm  Role = 65504
	RoleVMNFusionProj     Role = 65503
	RoleVMNFusionProjNorm Role = 65502
	RoleVMNFusionNorm     Role = 65501
	// RoleVMNLayerScale is a block's layer scale as the source carries it.
	// The converter folds it into the norm or the matrix it follows, so no
	// container carries one; the code names the source tensor and stays
	// reserved.
	RoleVMNLayerScale Role = 65500
	// RoleVMNHardEmbd is the text embedding of Gemma 3n's hard vision tokens
	// (its end-of-image marker and the image placeholder), [128][text width]:
	// the embedder's own table through its norm, its matrix and its
	// unweighted norm, which transformers puts in place of the text table's
	// rows for every id from RoleVMNHardOffset on. The converter computes it
	// once; the reader takes those ids' rows from it, unscaled.
	RoleVMNHardEmbd   Role = 65499
	RoleVMNHardOffset Role = 65498
	// RoleVMNHardTable and RoleVMNHardNorm are the embedder's table and norm
	// as the source carries them, folded into RoleVMNHardEmbd at conversion:
	// no container carries one.
	RoleVMNHardTable Role = 65497
	// RoleVTileMarks is a tiled picture's marker rows, [rows][text width]:
	// the text table's rows of the tokens that stand before each tile and
	// before its thumbnail (TOpMarkRow), copied at conversion.
	RoleVTileMarks Role = 65495
	// RoleVRsPosAxes is the resampler's key positions as two tables of the
	// same length, [2*length][D]: row x of the first the column's sin and
	// cos over the first half, row y of the second the row's over the
	// second (TOpPosAxes), written at conversion (resampler.py's
	// get_2d_sincos_pos_embed, for any grid a picture's piece reaches).
	RoleVRsPosAxes Role = 65494
	// RoleVPatchEmbd2 is a two-frame patch embedding's second plane, in
	// RoleVPatchEmbd's layout, which holds the planes' sum: a video's pair of
	// frames x0, x1 runs as x0*(W0+W1) + (x1-x0)*W1 (Video.Entry).
	RoleVPatchEmbd2 Role = 65493
	RoleVMNHardNorm Role = 65496
)

func visionFamRoleName(r Role) (string, bool) {
	switch r {
	case RoleVMerge:
		return "mm.patch_merger", true
	case RoleVImgBreak:
		return "v.token_embd.img_break", true
	case RoleVTileMarks:
		return "v.token_embd.tile_marks", true
	case RoleVRsPosAxes:
		return "resampler.pos_axes", true
	case RoleVPatchEmbd2:
		return "v.patch_embd.weight.1", true
	case RoleVProj3:
		return "mm.model.fc", true
	case RoleVPostAttnNorm:
		return "v.attn_post_norm", true
	case RoleVPostFFNNorm:
		return "v.ffn_post_norm", true
	case RoleVAttnQNorm:
		return "v.attn_q_norm", true
	case RoleVAttnKNorm:
		return "v.attn_k_norm", true
	case RoleVClamp:
		return "v.clamp", true
	case RoleVStdBias:
		return "v.std_bias", true
	case RoleVStdScale:
		return "v.std_scale", true
	case RoleVMNStem:
		return "v.conv_stem.conv", true
	case RoleVMNStemBias:
		return "v.conv_stem.conv.bias", true
	case RoleVMNStemNorm:
		return "v.conv_stem.bn", true
	case RoleVMNConvExp:
		return "v.conv_exp", true
	case RoleVMNNorm1:
		return "v.bn1", true
	case RoleVMNConvPwl:
		return "v.conv_pwl", true
	case RoleVMNNorm2:
		return "v.bn2", true
	case RoleVMNDwStart:
		return "v.dw_start.conv", true
	case RoleVMNDwStartNorm:
		return "v.dw_start.bn", true
	case RoleVMNPwExp:
		return "v.pw_exp.conv", true
	case RoleVMNPwExpNorm:
		return "v.pw_exp.bn", true
	case RoleVMNDwMid:
		return "v.dw_mid.conv", true
	case RoleVMNDwMidNorm:
		return "v.dw_mid.bn", true
	case RoleVMNPwProj:
		return "v.pw_proj.conv", true
	case RoleVMNPwProjNorm:
		return "v.pw_proj.bn", true
	case RoleVMNKDown:
		return "v.attn.key.down_conv", true
	case RoleVMNKDownNorm:
		return "v.attn.key.norm", true
	case RoleVMNVDown:
		return "v.attn.value.down_conv", true
	case RoleVMNVDownNorm:
		return "v.attn.value.norm", true
	case RoleVMNGeom:
		return "v.mn_geom", true
	case RoleVMNFusionExp:
		return "v.msfa.ffn.pw_exp.conv", true
	case RoleVMNFusionExpNorm:
		return "v.msfa.ffn.pw_exp.bn", true
	case RoleVMNFusionProj:
		return "v.msfa.ffn.pw_proj.conv", true
	case RoleVMNFusionProjNorm:
		return "v.msfa.ffn.pw_proj.bn", true
	case RoleVMNFusionNorm:
		return "v.msfa.norm", true
	case RoleVMNLayerScale:
		return "v.layer_scale", true
	case RoleVMNHardEmbd:
		return "mm.hard_embd", true
	case RoleVMNHardOffset:
		return "mm.hard_offset", true
	case RoleVMNHardTable:
		return "mm.embedding", true
	case RoleVMNHardNorm:
		return "mm.hard_emb_norm", true
	}
	return "", false
}

func visionFamProjName(p Projector) (string, bool) {
	switch p {
	case ProjD1OmniA:
		return "d1omni_a", true
	case ProjPixtral:
		return "pixtral", true
	case ProjLlama4:
		return "llama4", true
	case ProjGemma4V:
		return "gemma4v", true
	case ProjPhi4:
		return "phi4", true
	case ProjGemma3nV:
		return "gemma3nv", true
	case ProjLFM2:
		return "lfm2", true
	}
	return "", false
}

// visionFamRole reports whether r is one of the roles above, every one a
// tower tensor (Role.Vision).
func visionFamRole(r Role) bool {
	_, ok := visionFamRoleName(r)
	return ok
}

// visionFamExpanded is Role.Expanded for the roles above: the vectors a
// reader copies out at load.
func visionFamExpanded(r Role) bool {
	switch r {
	case RoleVImgBreak, RoleVPostAttnNorm, RoleVPostFFNNorm, RoleVAttnQNorm, RoleVAttnKNorm,
		RoleVClamp, RoleVStdBias, RoleVStdScale,
		RoleVMNStemBias, RoleVMNStemNorm, RoleVMNNorm1, RoleVMNNorm2, RoleVMNDwStart, RoleVMNDwStartNorm,
		RoleVMNPwExpNorm, RoleVMNDwMid, RoleVMNDwMidNorm, RoleVMNPwProjNorm, RoleVMNKDown,
		RoleVMNKDownNorm, RoleVMNVDown, RoleVMNVDownNorm, RoleVMNGeom, RoleVMNFusionExpNorm,
		RoleVMNFusionProjNorm, RoleVMNFusionNorm, RoleVMNHardEmbd, RoleVMNHardOffset:
		return true
	}
	return false
}

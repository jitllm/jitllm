package jlm

// The roles of a block with an attention of relative score biases, row sinks,
// short convolutions and inputs that read the residual streams apart, and
// of a looped block's message gates. Their codes are apart from every other
// group's so a concurrent addition does not collide.
const (
	// RoleAttnKInNorm is the key input's norm, where a block norms its key and value inputs apart from the query's.
	RoleAttnKInNorm Role = 900
	// RoleAttnVInNorm is the value input's norm.
	RoleAttnVInNorm Role = 901
	// RoleAttnSinksProj is each head's sink logit from the row: a sink a row's input (graph.Attrs.RowSinks).
	RoleAttnSinksProj Role = 902
	// RoleAttnQClipScale is the query's learned scale before its L2 clip, one a channel.
	RoleAttnQClipScale Role = 903
	// RoleAttnRel is the relative states' projection, d_rel channels a head.
	RoleAttnRel Role = 904
	// RoleAttnRelClipScale is the relative states' learned scale before their L2 clip.
	RoleAttnRelClipScale Role = 905
	// RoleAttnRelProj is a head's relative state to its score bias at each backward distance (graph.Attrs.Extent).
	RoleAttnRelProj Role = 906
	// RoleAttnKConv is the key's short convolution, its static taps.
	RoleAttnKConv Role = 907
	// RoleAttnVConv is the value's short convolution, its static taps.
	RoleAttnVConv Role = 908
	// RoleAttnKConvProj is the key convolution's per-row taps from the row, one set a key head.
	RoleAttnKConvProj Role = 909
	// RoleAttnVConvProj is the value convolution's per-row taps from the row.
	RoleAttnVConvProj Role = 910
	// RoleAttnPostConv is the short convolution over the attention's output.
	RoleAttnPostConv Role = 911
	// RoleFFNPostConv is the short convolution over the feed-forward's output.
	RoleFFNPostConv Role = 912
	// RoleAttnResScoreQ is the query input's read of the residual streams, where q, k and v read them apart.
	RoleAttnResScoreQ Role = 913
	// RoleAttnResScoreK is the key input's read of the residual streams.
	RoleAttnResScoreK Role = 914
	// RoleAttnResScoreV is the value input's read of the residual streams.
	RoleAttnResScoreV Role = 915
	// RoleLoopMsgQ is a looped block's query of the previous pass's states, for its query input's message.
	RoleLoopMsgQ Role = 916
	// RoleLoopMsgK is the same for its key input's message.
	RoleLoopMsgK Role = 917
	// RoleLoopMsgV is the same for its value input's message.
	RoleLoopMsgV Role = 918
	// RoleLoopMsgFFN is the same for its feed-forward input's message.
	RoleLoopMsgFFN Role = 919
	// RoleLoopQDown is the q input's message gate: [norm(input), norm(message)] down to the gate's rank.
	RoleLoopQDown Role = 920
	// RoleLoopQForget is the q input's forget gate from the rank.
	RoleLoopQForget Role = 921
	// RoleLoopQGain is the q input's message gain from the rank.
	RoleLoopQGain Role = 922
	// RoleLoopKDown is the k input's message gate: [norm(input), norm(message)] down to the gate's rank.
	RoleLoopKDown Role = 923
	// RoleLoopKForget is the k input's forget gate from the rank.
	RoleLoopKForget Role = 924
	// RoleLoopKGain is the k input's message gain from the rank.
	RoleLoopKGain Role = 925
	// RoleLoopVDown is the v input's message gate: [norm(input), norm(message)] down to the gate's rank.
	RoleLoopVDown Role = 926
	// RoleLoopVForget is the v input's forget gate from the rank.
	RoleLoopVForget Role = 927
	// RoleLoopVGain is the v input's message gain from the rank.
	RoleLoopVGain Role = 928
	// RoleLoopFFNDown is the ffn input's message gate: [norm(input), norm(message)] down to the gate's rank.
	RoleLoopFFNDown Role = 929
	// RoleLoopFFNForget is the ffn input's forget gate from the rank.
	RoleLoopFFNForget Role = 930
	// RoleLoopFFNGain is the ffn input's message gain from the rank.
	RoleLoopFFNGain Role = 931
)

func loopRoleName(r Role) (string, bool) {
	switch r {
	case RoleAttnKInNorm:
		return "attn_k_in_norm", true
	case RoleAttnVInNorm:
		return "attn_v_in_norm", true
	case RoleAttnSinksProj:
		return "attn_sinks_proj", true
	case RoleAttnQClipScale:
		return "attn_q_clip_scale", true
	case RoleAttnRel:
		return "attn_rel", true
	case RoleAttnRelClipScale:
		return "attn_rel_clip_scale", true
	case RoleAttnRelProj:
		return "attn_rel_proj", true
	case RoleAttnKConv:
		return "attn_k_conv", true
	case RoleAttnVConv:
		return "attn_v_conv", true
	case RoleAttnKConvProj:
		return "attn_k_conv_proj", true
	case RoleAttnVConvProj:
		return "attn_v_conv_proj", true
	case RoleAttnPostConv:
		return "attn_post_conv", true
	case RoleFFNPostConv:
		return "ffn_post_conv", true
	case RoleAttnResScoreQ:
		return "attn_res_score_q", true
	case RoleAttnResScoreK:
		return "attn_res_score_k", true
	case RoleAttnResScoreV:
		return "attn_res_score_v", true
	case RoleLoopMsgQ:
		return "loop_msg_q", true
	case RoleLoopMsgK:
		return "loop_msg_k", true
	case RoleLoopMsgV:
		return "loop_msg_v", true
	case RoleLoopMsgFFN:
		return "loop_msg_ffn", true
	case RoleLoopQDown:
		return "loop_q_down", true
	case RoleLoopQForget:
		return "loop_q_forget", true
	case RoleLoopQGain:
		return "loop_q_gain", true
	case RoleLoopKDown:
		return "loop_k_down", true
	case RoleLoopKForget:
		return "loop_k_forget", true
	case RoleLoopKGain:
		return "loop_k_gain", true
	case RoleLoopVDown:
		return "loop_v_down", true
	case RoleLoopVForget:
		return "loop_v_forget", true
	case RoleLoopVGain:
		return "loop_v_gain", true
	case RoleLoopFFNDown:
		return "loop_ffn_down", true
	case RoleLoopFFNForget:
		return "loop_ffn_forget", true
	case RoleLoopFFNGain:
		return "loop_ffn_gain", true
	}
	return "", false
}

// loopExpanded says a role of this group is a vector read once at load
// (Role.Expanded): the norms, scales, static taps and queries.
func loopExpanded(r Role) bool {
	switch r {
	case RoleAttnKInNorm, RoleAttnVInNorm, RoleAttnQClipScale, RoleAttnRelClipScale, RoleAttnKConv, RoleAttnVConv, RoleAttnPostConv, RoleFFNPostConv, RoleAttnResScoreQ, RoleAttnResScoreK, RoleAttnResScoreV, RoleLoopMsgQ, RoleLoopMsgK, RoleLoopMsgV, RoleLoopMsgFFN:
		return true
	}
	return false
}

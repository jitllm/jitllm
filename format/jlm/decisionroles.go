package jlm

// Clef's joint schema head (Cloudflare/clef-flash, joint_schema_model.py's
// JointSchemaHead), as llama.cpp's converter names it (conversion/clef.py):
// the decision.* tensors are model-level, and the head's blocks, dec.blk.N,
// are stored at DenseBlock with Index N -- the routing blocks first
// (Config.DecisionRouting of them), then the joint ones (Config.DecisionBlocks)
// -- so they never meet a backbone block's number. A group of its own, counted
// up from 800.
const (
	// The LayerNorm over the backbone's final hidden states, and the six
	// projections out of the backbone's width into the head's.
	RoleDecHiddenNorm     Role = 800
	RoleDecHiddenNormBias Role = 801
	RoleDecProjMemory     Role = 802
	RoleDecProjQuestion   Role = 803
	RoleDecProjOptQ       Role = 804 // option_question
	RoleDecProjGlobal     Role = 805
	RoleDecProjOptCtx     Role = 806 // option_context
	RoleDecProjOptLex     Role = 807 // option_lexical
	// The three LayerNorms of the head's tail, and the learned scalars stored
	// as the values used at inference: exp(min(prior, log 100)),
	// exp(min(joint, log 100)) and sigmoid(residual_gate).
	RoleDecOptSumNorm     Role = 808
	RoleDecOptSumNormBias Role = 809
	RoleDecFieldNorm      Role = 810
	RoleDecFieldNormBias  Role = 811
	RoleDecOptNorm        Role = 812
	RoleDecOptNormBias    Role = 813
	RoleDecScales         Role = 814
	// A head block's: the joint blocks' self-attention over the questions,
	// every block's cross-attention to the memory (a routing block norms the
	// memory first, CrossKVNorm), and every block's MLP.
	RoleDecSelfNorm        Role = 820
	RoleDecSelfNormBias    Role = 821
	RoleDecSelfQ           Role = 822
	RoleDecSelfK           Role = 823
	RoleDecSelfV           Role = 824
	RoleDecSelfOut         Role = 825
	RoleDecSelfQBias       Role = 826
	RoleDecSelfKBias       Role = 827
	RoleDecSelfVBias       Role = 828
	RoleDecSelfOutBias     Role = 829
	RoleDecCrossNorm       Role = 830
	RoleDecCrossNormBias   Role = 831
	RoleDecCrossKVNorm     Role = 832
	RoleDecCrossKVNormBias Role = 833
	RoleDecCrossQ          Role = 834
	RoleDecCrossK          Role = 835
	RoleDecCrossV          Role = 836
	RoleDecCrossOut        Role = 837
	RoleDecCrossQBias      Role = 838
	RoleDecCrossKBias      Role = 839
	RoleDecCrossVBias      Role = 840
	RoleDecCrossOutBias    Role = 841
	RoleDecFFNNorm         Role = 842
	RoleDecFFNNormBias     Role = 843
	RoleDecFFNUp           Role = 844
	RoleDecFFNUpBias       Role = 845
	RoleDecFFNDown         Role = 846
	RoleDecFFNDownBias     Role = 847
)

func decisionRoleName(r Role) (string, bool) {
	switch r {
	case RoleDecHiddenNorm:
		return "decision.hidden_norm", true
	case RoleDecHiddenNormBias:
		return "decision.hidden_norm.bias", true
	case RoleDecProjMemory:
		return "decision.proj_memory", true
	case RoleDecProjQuestion:
		return "decision.proj_question", true
	case RoleDecProjOptQ:
		return "decision.proj_option_question", true
	case RoleDecProjGlobal:
		return "decision.proj_global", true
	case RoleDecProjOptCtx:
		return "decision.proj_option_context", true
	case RoleDecProjOptLex:
		return "decision.proj_option_lexical", true
	case RoleDecOptSumNorm:
		return "decision.option_summary_norm", true
	case RoleDecOptSumNormBias:
		return "decision.option_summary_norm.bias", true
	case RoleDecFieldNorm:
		return "decision.field_norm", true
	case RoleDecFieldNormBias:
		return "decision.field_norm.bias", true
	case RoleDecOptNorm:
		return "decision.option_norm", true
	case RoleDecOptNormBias:
		return "decision.option_norm.bias", true
	case RoleDecScales:
		return "decision.scales", true
	case RoleDecSelfNorm:
		return "dec.attn_norm", true
	case RoleDecSelfNormBias:
		return "dec.attn_norm.bias", true
	case RoleDecSelfQ:
		return "dec.attn_q", true
	case RoleDecSelfK:
		return "dec.attn_k", true
	case RoleDecSelfV:
		return "dec.attn_v", true
	case RoleDecSelfOut:
		return "dec.attn_o", true
	case RoleDecSelfQBias:
		return "dec.attn_q.bias", true
	case RoleDecSelfKBias:
		return "dec.attn_k.bias", true
	case RoleDecSelfVBias:
		return "dec.attn_v.bias", true
	case RoleDecSelfOutBias:
		return "dec.attn_o.bias", true
	case RoleDecCrossNorm:
		return "dec.cross_attn_norm", true
	case RoleDecCrossNormBias:
		return "dec.cross_attn_norm.bias", true
	case RoleDecCrossKVNorm:
		return "dec.cross_attn_norm_kv", true
	case RoleDecCrossKVNormBias:
		return "dec.cross_attn_norm_kv.bias", true
	case RoleDecCrossQ:
		return "dec.cross_attn_q", true
	case RoleDecCrossK:
		return "dec.cross_attn_k", true
	case RoleDecCrossV:
		return "dec.cross_attn_v", true
	case RoleDecCrossOut:
		return "dec.cross_attn_o", true
	case RoleDecCrossQBias:
		return "dec.cross_attn_q.bias", true
	case RoleDecCrossKBias:
		return "dec.cross_attn_k.bias", true
	case RoleDecCrossVBias:
		return "dec.cross_attn_v.bias", true
	case RoleDecCrossOutBias:
		return "dec.cross_attn_o.bias", true
	case RoleDecFFNNorm:
		return "dec.ffn_norm", true
	case RoleDecFFNNormBias:
		return "dec.ffn_norm.bias", true
	case RoleDecFFNUp:
		return "dec.ffn_up", true
	case RoleDecFFNUpBias:
		return "dec.ffn_up.bias", true
	case RoleDecFFNDown:
		return "dec.ffn_down", true
	case RoleDecFFNDownBias:
		return "dec.ffn_down.bias", true
	}
	return "", false
}

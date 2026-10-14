package jlm

import "fmt"

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

var decisionRoleNames = map[Role]string{
	RoleDecHiddenNorm: "decision.hidden_norm", RoleDecHiddenNormBias: "decision.hidden_norm.bias",
	RoleDecProjMemory: "decision.proj_memory", RoleDecProjQuestion: "decision.proj_question",
	RoleDecProjOptQ: "decision.proj_option_question", RoleDecProjGlobal: "decision.proj_global",
	RoleDecProjOptCtx: "decision.proj_option_context", RoleDecProjOptLex: "decision.proj_option_lexical",
	RoleDecOptSumNorm: "decision.option_summary_norm", RoleDecOptSumNormBias: "decision.option_summary_norm.bias",
	RoleDecFieldNorm: "decision.field_norm", RoleDecFieldNormBias: "decision.field_norm.bias",
	RoleDecOptNorm: "decision.option_norm", RoleDecOptNormBias: "decision.option_norm.bias",
	RoleDecScales:   "decision.scales",
	RoleDecSelfNorm: "dec.attn_norm", RoleDecSelfNormBias: "dec.attn_norm.bias",
	RoleDecSelfQ: "dec.attn_q", RoleDecSelfK: "dec.attn_k", RoleDecSelfV: "dec.attn_v", RoleDecSelfOut: "dec.attn_o",
	RoleDecSelfQBias: "dec.attn_q.bias", RoleDecSelfKBias: "dec.attn_k.bias",
	RoleDecSelfVBias: "dec.attn_v.bias", RoleDecSelfOutBias: "dec.attn_o.bias",
	RoleDecCrossNorm: "dec.cross_attn_norm", RoleDecCrossNormBias: "dec.cross_attn_norm.bias",
	RoleDecCrossKVNorm: "dec.cross_attn_norm_kv", RoleDecCrossKVNormBias: "dec.cross_attn_norm_kv.bias",
	RoleDecCrossQ: "dec.cross_attn_q", RoleDecCrossK: "dec.cross_attn_k",
	RoleDecCrossV: "dec.cross_attn_v", RoleDecCrossOut: "dec.cross_attn_o",
	RoleDecCrossQBias: "dec.cross_attn_q.bias", RoleDecCrossKBias: "dec.cross_attn_k.bias",
	RoleDecCrossVBias: "dec.cross_attn_v.bias", RoleDecCrossOutBias: "dec.cross_attn_o.bias",
	RoleDecFFNNorm: "dec.ffn_norm", RoleDecFFNNormBias: "dec.ffn_norm.bias",
	RoleDecFFNUp: "dec.ffn_up", RoleDecFFNUpBias: "dec.ffn_up.bias",
	RoleDecFFNDown: "dec.ffn_down", RoleDecFFNDownBias: "dec.ffn_down.bias",
}

func init() {
	for r, n := range decisionRoleNames {
		// Another group taking the same number would rename its role here
		// and read one tensor as the other.
		if was, ok := roleNames[r]; ok {
			panic(fmt.Sprintf("jlm: role %d is both %q and %q", r, was, n))
		}
		roleNames[r] = n
	}
}

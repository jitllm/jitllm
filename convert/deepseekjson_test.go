package convert

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestDeepseekThroughTheLibrary: the MLA geometry, the router and the query
// form the two deepseek files state. V2 gates with a softmax and V3 (and
// GLM-4.7-Flash) with a sigmoid whatever scoring_func says; a null rank
// states the one-step query and a rank the two-step, each held to the tensors
// the blocks ship; a key-value head count that is not the attention heads is
// refused.
func TestDeepseekThroughTheLibrary(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	js := `{"architectures": ["DeepseekV3ForCausalLM"], "vocab_size": 100, "hidden_size": 64, "intermediate_size": 96,
	  "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4, "kv_lora_rank": 16, "q_lora_rank": 32,
	  "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 12, "n_routed_experts": 8, "num_experts_per_tok": 2,
	  "moe_intermediate_size": 32, "n_shared_experts": 1, "first_k_dense_replace": 1}`
	blk := func(b int, name string) string { return "model.layers." + itoa(b) + "." + name }
	twoStep := []string{"lm_head.weight", blk(0, "mlp.gate_proj.weight"), blk(0, "self_attn.q_a_proj.weight"),
		blk(1, "self_attn.q_a_proj.weight"), blk(1, "mlp.gate.weight"), blk(1, "mlp.experts.0.gate_proj.weight")}
	oneStep := []string{"lm_head.weight", blk(0, "mlp.gate_proj.weight"), blk(0, "self_attn.q_proj.weight"),
		blk(1, "self_attn.q_proj.weight"), blk(1, "mlp.gate.weight"), blk(1, "mlp.experts.0.gate_proj.weight")}
	v3, v2 := lib["DeepseekV3ForCausalLM"], lib["DeepseekV2ForCausalLM"]
	if v3 == nil || v2 == nil || lib["Glm4MoeLiteForCausalLM"] == nil {
		t.Fatal("the library lacks a deepseek class")
	}
	cfg, err := v3.config(hfConfigOf(t, js), twoStep)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.HeadDim != 24 || cfg.HeadDimV != 12 || cfg.NRot != 8 || cfg.KVLoraRank != 16 || cfg.QLoraRank != 32 ||
		cfg.NKVHead != 4 || cfg.NDenseLead != 1 || cfg.NFFNShExp != 32 || !cfg.Flags.Has(jlm.FlagExpertSigmoid) {
		t.Errorf("head %d/%d rot %d latent %d/%d kv heads %d lead %d shared %d flags %v", cfg.HeadDim, cfg.HeadDimV,
			cfg.NRot, cfg.KVLoraRank, cfg.QLoraRank, cfg.NKVHead, cfg.NDenseLead, cfg.NFFNShExp, cfg.Flags)
	}
	soft, err := v2.config(hfConfigOf(t, strings.Replace(js, "DeepseekV3", "DeepseekV2", 1)), twoStep)
	if err != nil || soft.Flags.Has(jlm.FlagExpertSigmoid) {
		t.Errorf("a V2 router: %v", err)
	}
	nullRank := strings.Replace(js, `"q_lora_rank": 32`, `"q_lora_rank": null`, 1)
	if c, err := v3.config(hfConfigOf(t, nullRank), oneStep); err != nil || c.QLoraRank != 0 {
		t.Errorf("a null rank with q_proj: %v", err)
	}
	asV2 := strings.Replace(js, "DeepseekV3", "DeepseekV2", 1)
	for _, r := range []struct {
		name, js string
		class    *modelJSON
		names    []string
		err      string
	}{
		{"a rank and q_proj", js, v3, oneStep, `block 0 carries no self_attn.q_a_proj.weight, and "q_lora_rank != null" is true`},
		{"a null rank and q_a_proj", nullRank, v3, twoStep, `block 0 carries self_attn.q_a_proj.weight, and "q_lora_rank != null" is false`},
		{"grouped key-value heads", strings.Replace(js, `"num_key_value_heads": 4`, `"num_key_value_heads": 1`, 1), v3, twoStep,
			"latent attention shares one key-value latent across every head"},
		{"a softmax on V3", strings.Replace(js, `"first_k_dense_replace"`, `"scoring_func": "softmax", "first_k_dense_replace"`, 1), v3, twoStep,
			"states the class's sigmoid router"},
		{"a sigmoid on V2", strings.Replace(asV2, `"first_k_dense_replace"`, `"scoring_func": "sigmoid", "first_k_dense_replace"`, 1), v2, twoStep,
			"states the class's softmax router"},
	} {
		if _, err := r.class.config(hfConfigOf(t, r.js), r.names); err == nil || !strings.Contains(err.Error(), r.err) {
			t.Errorf("%s: %v, want an error saying %q", r.name, err, r.err)
		}
	}
}

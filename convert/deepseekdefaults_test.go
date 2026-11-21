package convert

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestDeepseekClassDefaults: a config.json that omits the keys a class's own
// config dataclass defaults takes that class's value, not another's, read from
// transformers' configuration_deepseek_v2, _v3 and _glm4_moe_lite.
func TestDeepseekClassDefaults(t *testing.T) {
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	const base = `"vocab_size": 100, "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 4,
	  "num_attention_heads": 4, "kv_lora_rank": 16, "qk_nope_head_dim": 16, "qk_rope_head_dim": 8,
	  "v_head_dim": 12, "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32`
	blk := func(b int, name string) string { return "model.layers." + itoa(b) + "." + name }
	namesWith := func(lead int) []string {
		names := []string{"lm_head.weight"}
		for b := 0; b < 4; b++ {
			names = append(names, blk(b, "self_attn.q_a_proj.weight"))
			if b < lead {
				names = append(names, blk(b, "mlp.gate_proj.weight"))
			} else {
				names = append(names, blk(b, "mlp.gate.weight"), blk(b, "mlp.experts.0.gate_proj.weight"))
			}
		}
		return names
	}
	for _, c := range []struct {
		class        string
		ctx          uint32
		lead, shared uint32
		scale        float32
		groups, used uint32
		eps          float32
		q            uint32
		noExpertNorm bool
	}{
		{"DeepseekV3ForCausalLM", 4096, 3, 1, 2.5, 8, 4, 1e-6, 1536, false},
		{"Glm4MoeLiteForCausalLM", 202752, 1, 1, 1.8, 1, 1, 1e-5, 768, false},
		{"DeepseekV2ForCausalLM", 2048, 0, 2, 1, 0, 0, 1e-6, 1536, true},
	} {
		cfg, err := lib[c.class].config(hfConfigOf(t, `{"architectures": ["`+c.class+`"], `+base+`}`), namesWith(int(c.lead)))
		if err != nil {
			t.Errorf("%s: %v", c.class, err)
			continue
		}
		if cfg.NCtx != c.ctx || cfg.NDenseLead != c.lead || cfg.NFFNShExp != c.shared*32 || cfg.ExpertScale != c.scale ||
			cfg.NExpertGroup != c.groups || cfg.NExpertGroupUsed != c.used || cfg.RMSEps != c.eps ||
			cfg.QLoraRank != c.q || cfg.Flags.Has(jlm.FlagNoExpertNorm) != c.noExpertNorm {
			t.Errorf("%s: ctx %d lead %d shared %d scale %v groups %d/%d eps %v q %d flags %v", c.class, cfg.NCtx,
				cfg.NDenseLead, cfg.NFFNShExp, cfg.ExpertScale, cfg.NExpertGroup, cfg.NExpertGroupUsed, cfg.RMSEps,
				cfg.QLoraRank, cfg.Flags)
		}
	}
}

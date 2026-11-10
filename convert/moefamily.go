package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// The mixture families added beside DeepSeek's: GLM-4.5 (glm4moe), Qwen1.5-MoE
// (qwen2moe), ERNIE 4.5 (ernie4_5-moe). Each reads its architecture's constants here; everything a
// mixture states in keys
// (expert counts, the shared expert, the dense lead, the routed scale, the
// prediction blocks) is read for every architecture in configOf.

// glm4moeConfig is llama.cpp's glm4-moe.cpp and transformers'
// Glm4MoeForCausalLM:
//
//	q, k, v biased, o not                    -> tensors
//	per-head q/k RMSNorm when the file has it -> FlagQKNorm (the 355B only)
//	NEOX rotary on rope.dimension_count       -> FlagRopeNeox, NRot (half a head)
//	post_attention_norm on the residual       -> the FFN norm (retarget)
//	sigmoid router, selection-only bias,      -> FlagExpertSigmoid, RoleExpProbsB,
//	renormalised, routed_scaling_factor          expert_weights_norm/scale
//	one ungated shared expert, a dense lead   -> keys, read in configOf
//
// The gate is SIGMOID whatever expert_gating_func says (RULE 7m): transformers'
// Glm4MoeTopkRouter hardcodes router_logits.sigmoid() and reads no gating key,
// and llama.cpp defaults an absent key to sigmoid for this arch. A file that
// states softmax is refused rather than obeyed or overridden.
//
// An absent expert_weights_norm renormalises, as Glm4MoeConfig's
// norm_topk_prob defaults to true (llama.cpp's loader would default it to
// false); every published GLM-4.5 config states true and llama.cpp's converter
// writes it, so the two never meet on a real file.
//
// Grouped selection is read off the keys as for any mixture. Every published
// GLM-4.5 has n_group 1 and topk_group 1, which is the ungrouped top-k.
func glm4moeConfig(f *meta.File, c *Config) error {
	c.Flags |= jlm.FlagRopeNeox | jlm.FlagExpertSigmoid
	if err := partialRotary(c); err != nil {
		return err
	}
	if kv, ok := f.Key("expert_gating_func"); ok {
		if v, _ := kv.Uint(); v != 2 {
			return fmt.Errorf("expert_gating_func %d, and Glm4MoeTopkRouter is sigmoid: %w",
				v, ErrNotImplemented)
		}
	}
	// The q/k norm is optional per checkpoint (use_qk_norm): GLM-4.5 355B has
	// it and GLM-4.5-Air does not. Both or neither.
	_, q := f.Get("blk.0.attn_q_norm.weight")
	_, k := f.Get("blk.0.attn_k_norm.weight")
	if q != k {
		return fmt.Errorf("blk.0 carries a q norm %v and a k norm %v", q, k)
	}
	if q {
		c.Flags |= jlm.FlagQKNorm
	}
	return nil
}

// qwen2moeConfig is llama.cpp's qwen2moe.cpp and transformers'
// Qwen2MoeForCausalLM: qwen2's attention (biased q/k/v, NEOX rotary over the
// whole head) and a softmax top-k beside a shared expert whose output is
// scaled by sigmoid(ffn_gate_inp_shexp . x).
//
// The routed weights are not renormalised when the file says nothing
// (configOf): llama.cpp's builder passes false as a literal and its converter
// writes no key, Qwen2MoeConfig's norm_topk_prob defaults to false, and every
// published Qwen1.5-MoE and Qwen2-57B config states false. A file that writes
// expert_weights_norm is obeyed.
//
// transformers can make some blocks dense (decoder_sparse_step,
// mlp_only_layers); no GGUF key carries either, llama.cpp requires every block
// to be a mixture, and no published checkpoint uses them. A block with no
// router is refused by name rather than run as a mixture with no router.
func qwen2moeConfig(f *meta.File, c *Config) error {
	c.Flags |= jlm.FlagRopeNeox
	if f.UintKey("expert_count", 0) == 0 {
		return fmt.Errorf("no expert_count: qwen2moe is a mixture")
	}
	if kv, ok := f.Key("expert_gating_func"); ok {
		if v, _ := kv.Uint(); v != 1 {
			return fmt.Errorf("expert_gating_func %d, and Qwen2MoeTopKRouter is softmax: %w",
				v, ErrNotImplemented)
		}
	}
	for i := 0; i < int(c.NLayer); i++ {
		if _, ok := f.Get("blk." + itoa(i) + ".ffn_gate_inp.weight"); !ok {
			return fmt.Errorf("block %d has no router (decoder_sparse_step or mlp_only_layers): %w",
				i, ErrNotImplemented)
		}
	}
	return nil
}

// ernie45moeConfig is llama.cpp's ernie4-5-moe.cpp and transformers'
// Ernie4_5_MoeForCausalLM: llama's attention (interleaved rotary, no biases on
// a published checkpoint, a tied head) and a softmax top-k whose selection key
// is the softmax probability plus a per-expert bias (Ernie4_5_MoeStatics),
// renormalised, beside shared experts. A block is a mixture past the dense
// lead (moe_layer_start_index) when (il+1) % interleave_moe_layer_step == 0.
//
// The gate is softmax and the weights are renormalised, and neither is read
// from a key: llama.cpp's builder passes both as literals and its converter
// writes neither, and the class always divides by the clamped sum. A file that
// states otherwise is refused. The divisor's floor is moe_norm_min (1e-12) in
// the class and DeepSeek's 1e-20 here; a softmax's top-k sums to at least k/n,
// so the two never differ.
//
// moe_layer_end_index has no key and llama.cpp takes every block past the lead;
// every published ERNIE ends at its last block, which is that.
func ernie45moeConfig(f *meta.File, c *Config) error {
	if kv, ok := f.Key("expert_gating_func"); ok {
		if v, _ := kv.Uint(); v != 1 {
			return fmt.Errorf("expert_gating_func %d, and Ernie4_5_MoeTopKRouter is softmax: %w",
				v, ErrNotImplemented)
		}
	}
	if kv, ok := f.Key("expert_weights_norm"); ok {
		if v, _ := kv.Uint(); v == 0 {
			return fmt.Errorf("expert_weights_norm false, and Ernie4_5_MoeTopKRouter always "+
				"renormalises: %w", ErrNotImplemented)
		}
	}
	step := f.UintKey("interleave_moe_layer_step", 0)
	if step == 0 {
		return fmt.Errorf("no interleave_moe_layer_step, which llama.cpp requires of this arch")
	}
	c.MoEStep = uint32(step)
	return nil
}

// ernieSharedWidth takes the shared expert's width off its tensor. llama.cpp's
// converter writes intermediate_size // num_key_value_heads, where the class
// builds moe_intermediate_size * moe_num_shared_experts; the two agree on
// every published ERNIE and nothing ties them, and the tensor is the fact.
func ernieSharedWidth(f *meta.File, c *Config) error {
	for i := 0; i < int(c.NLayer); i++ {
		t, ok := f.Get("blk." + itoa(i) + ".ffn_gate_shexp.weight")
		if !ok {
			continue
		}
		if len(t.Dims) < 2 {
			return fmt.Errorf("blk.%d.ffn_gate_shexp.weight is %v", i, t.Dims)
		}
		c.NFFNShExp = uint32(t.Dims[1])
		return nil
	}
	c.NFFNShExp = 0
	return nil
}

package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// The modern text families added as one group (ArchBailingMoE2 and its
// siblings in format/jlm/archmoe2.go). Each reads its architecture's constants
// here; what a mixture states in keys (expert counts, groups, the shared
// expert, the dense lead, the routed scale, the prediction blocks) is read for
// every architecture in configOf.

// bailingmoe2Config is llama.cpp's bailingmoe2.cpp and inclusionAI's
// BailingMoeV2ForCausalLM (the remote code Ling-mini-2.0 ships):
//
//	fused query_key_value, no biases          -> unfuse (fusesQKV)
//	per-head q/k RMSNorm before the rotary    -> FlagQKNorm
//	NEOX rotary on rope.dimension_count       -> FlagRopeNeox, NRot (half a head)
//	sigmoid router, expert_bias for selection -> FlagExpertSigmoid, RoleExpProbsB
//	group-limited top-k (top-2 group score)   -> expert_group_count/_used_count
//	renormalised + 1e-20, routed_scaling      -> expert_weights_norm/scale
//	shared experts, a dense lead               -> keys, read in configOf
//
// The gate is SIGMOID whatever the file says (RULE 7m): BailingMoeV2Gate
// computes torch.sigmoid(logits) and reads no gating key, so a file that
// states softmax describes weights the class never ran and is refused.
//
// The shared expert's width is read off its tensor (ernieSharedWidth): the
// class builds one MLP of moe_intermediate_size * num_shared_experts, the
// converter writes moe_shared_expert_intermediate_size when the config has it,
// and llama.cpp's loader multiplies whatever it reads by the count again. The
// three agree at one shared expert, which every published Ling 2.0 has.
//
// The per-head norm is not optional in the class's published configs
// (use_qk_norm true) and llama.cpp creates both tensors as required, so a
// block missing either is refused rather than run without it.
func bailingmoe2Config(f *meta.File, c *Config) error {
	c.Flags |= jlm.FlagRopeNeox | jlm.FlagExpertSigmoid | jlm.FlagQKNorm
	if err := partialRotary(c); err != nil {
		return err
	}
	if f.UintKey("expert_count", 0) == 0 {
		return fmt.Errorf("no expert_count: bailingmoe2 is a mixture")
	}
	if kv, ok := f.Key("expert_gating_func"); ok {
		if v, _ := kv.Uint(); v != 2 {
			return fmt.Errorf("expert_gating_func %d, and BailingMoeV2Gate is sigmoid: %w",
				v, ErrNotImplemented)
		}
	}
	return requireQKNorm(f, "Ling 2.0")
}

// requireQKNorm refuses a file whose block 0 lacks either per-head norm, for
// an architecture whose class always builds both.
func requireQKNorm(f *meta.File, who string) error {
	for _, n := range []string{"blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight"} {
		if _, ok := f.Get(n); !ok {
			return fmt.Errorf("no %s: %s norms each head of q and k", n, who)
		}
	}
	return nil
}

// phimoeConfig is transformers' PhimoeForCausalLM, which llama.cpp's phimoe
// (phi3.cpp's graph) departs from twice (RULE 7m, chosen: the model's own
// class):
//
//	nn.LayerNorm with a bias, every norm      -> FlagLayerNorm (llama.cpp runs
//	                                             an RMSNorm and adds the bias)
//	biased q/k/v/o, a biased head             -> tensors
//	NEOX LongRoPE, short factors below the    -> phi3's path: rope_factors_short,
//	original context                             AttnFactor
//	sparsemixer top-2, not renormalised       -> the arch's code (ArchPhiMoE),
//	                                             FlagNoExpertNorm (llama.cpp runs
//	                                             a renormalised softmax top-2)
//
// The LayerNorm's epsilon is rms_norm_eps, which the converter writes as
// layer_norm_rms_epsilon and configOf already read.
//
// sparsemixer selects two experts whatever num_experts_per_tok says; a file
// stating another count, or a gating function, is refused rather than run as
// something its class never computed. A window no shorter than the context
// (Phi-3.5-MoE's 131072 at 131072) masks nothing and is dropped; a shorter
// one slides every layer, as phi3's does.
//
// The cos/sin scale is the file's: transformers multiplies by short_mscale
// from config.json (1.243 on Phi-3.5-MoE), which no GGUF carries, and
// llama.cpp's converter writes its own attn_factor (1.190) instead. The two
// inputs therefore differ by that number on the real model; the safetensors
// path reads the class's.
func phimoeConfig(f *meta.File, c *Config) error {
	c.Flags |= jlm.FlagLayerNorm | jlm.FlagRopeNeox | jlm.FlagNoExpertNorm
	// A head narrower than embedding/heads (Phi-tiny-MoE's 128 against
	// 4096/16) is stated only by head_dim in config.json, and llama.cpp's
	// phimoe converter writes no attention.key_length: such a file says 256
	// where its q projection says 128, and llama.cpp refuses to load it. The
	// projection is the evidence; the file is refused, and the safetensors
	// input, which reads head_dim, converts the model.
	if q, ok := f.Layer(0, "attn_q.weight"); ok && len(q.Dims) == 2 &&
		q.Dims[1] != uint64(c.NHead)*uint64(c.HeadDim) {
		return fmt.Errorf("blk.0.attn_q.weight has %d rows for %d heads of %d: the file states no "+
			"attention.key_length for its head width; convert the model from its safetensors",
			q.Dims[1], c.NHead, c.HeadDim)
	}
	if err := partialRotary(c); err != nil {
		return err
	}
	if n := f.UintKey("expert_count", 0); n < 2 {
		return fmt.Errorf("expert_count %d: phimoe is a mixture", n)
	}
	if k := f.UintKey("expert_used_count", 0); k != 2 {
		return fmt.Errorf("expert_used_count %d, and sparsemixer selects 2: %w", k, ErrNotImplemented)
	}
	if kv, ok := f.Key("expert_gating_func"); ok {
		v, _ := kv.Uint()
		return fmt.Errorf("expert_gating_func %d, and PhimoeTopKRouter is sparsemixer: %w", v, ErrNotImplemented)
	}
	if win := uint32(f.UintKey("attention.sliding_window", 0)); win > 0 && win < c.NCtx {
		c.SWAWindow, c.SWAPeriod = win, allLocal(c.NLayer)
	}
	return nil
}

// dots1Config is llama.cpp's dots1.cpp and transformers' Dots1ForCausalLM:
//
//	q, k, v, o unbiased                      -> tensors
//	per-head q/k RMSNorm before the rotary    -> FlagQKNorm
//	NEOX rotary over the whole head           -> FlagRopeNeox
//	sigmoid router, selection-only bias,      -> FlagExpertSigmoid, RoleExpProbsB,
//	renormalised + 1e-20, routed scale           expert_weights_norm/scale
//	shared experts, a dense lead              -> keys, read in configOf
//
// The gate is SIGMOID whatever the file says (RULE 7m): Dots1TopkRouter
// computes router_logits.sigmoid() and reads no gating key. llama.cpp reads
// expert_gating_func with no default for this arch and its converter writes
// it from scoring_func, so an absent key is the class's sigmoid and a stated
// softmax is refused.
//
// The class slides a window on the layers past max_window_layers when its
// config carries one; every published dots.llm1 states sliding_window null
// and llama.cpp applies none, so a file stating a window is refused by name
// rather than run with every layer global.
func dots1Config(f *meta.File, c *Config) error {
	c.Flags |= jlm.FlagRopeNeox | jlm.FlagExpertSigmoid | jlm.FlagQKNorm
	if f.UintKey("expert_count", 0) == 0 {
		return fmt.Errorf("no expert_count: dots1 is a mixture")
	}
	if kv, ok := f.Key("expert_gating_func"); ok {
		if v, _ := kv.Uint(); v != 2 {
			return fmt.Errorf("expert_gating_func %d, and Dots1TopkRouter is sigmoid: %w",
				v, ErrNotImplemented)
		}
	}
	if w := f.UintKey("attention.sliding_window", 0); w != 0 {
		return fmt.Errorf("attention.sliding_window %d: dots.llm1's sliding layers are not "+
			"implemented: %w", w, ErrNotImplemented)
	}
	if c.NRot != c.HeadDim {
		return fmt.Errorf("rotary width %d on a %d-wide head: dots1 rotates the whole head", c.NRot, c.HeadDim)
	}
	return requireQKNorm(f, "dots.llm1")
}

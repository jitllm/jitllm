package convert

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// phimoeHF is Phi-3.5-MoE, Phi-mini-MoE and Phi-tiny-MoE (PhiMoEForCausalLM,
// transformers' PhimoeForCausalLM): mixtral's tensor names with every norm a
// LayerNorm with a bias, biased q/k/v/o and a biased head, NEOX rotary (so q
// and k are not permuted, as convert_hf_to_gguf.py does not permute them), and
// phi3's LongRoPE where the config states it. The router is sparsemixer, which
// the arch code carries (PhiMoEJitter); phimoeConfig is the GGUF half.
func phimoeHF() *hfArch {
	return &hfArch{
		arch:        ArchPhiMoE,
		blockPrefix: llamaHF().blockPrefix,
		model: map[string]jlm.Role{
			"model.embed_tokens.weight": jlm.RoleTokenEmbd,
			"model.norm.weight":         jlm.RoleOutputNorm,
			"model.norm.bias":           jlm.RoleOutputNormBias,
			"lm_head.weight":            jlm.RoleOutput,
			"lm_head.bias":              jlm.RoleOutputBias,
		},
		block: map[string]jlm.Role{
			"input_layernorm.weight":          jlm.RoleAttnNorm,
			"input_layernorm.bias":            jlm.RoleAttnNormBias,
			"post_attention_layernorm.weight": jlm.RoleFFNNorm,
			"post_attention_layernorm.bias":   jlm.RoleFFNNormBias,
			"self_attn.q_proj.weight":         jlm.RoleAttnQ,
			"self_attn.k_proj.weight":         jlm.RoleAttnK,
			"self_attn.v_proj.weight":         jlm.RoleAttnV,
			"self_attn.o_proj.weight":         jlm.RoleAttnOut,
			"self_attn.q_proj.bias":           jlm.RoleAttnQBias,
			"self_attn.k_proj.bias":           jlm.RoleAttnKBias,
			"self_attn.v_proj.bias":           jlm.RoleAttnVBias,
			"self_attn.o_proj.bias":           jlm.RoleAttnOutBias,
			"block_sparse_moe.gate.weight":    jlm.RoleRouter,
		},
		expertPrefix: "block_sparse_moe.experts.",
		expert: map[string]jlm.Role{
			"w1.weight": jlm.RoleExpGateBank,
			"w3.weight": jlm.RoleExpUpBank,
			"w2.weight": jlm.RoleExpDownBank,
		},
		ignore: llamaHF().ignore,
		config: phimoeHFConfig,
	}
}

// phimoeHFConfig reads PhimoeConfig. sparsemixer keeps two experts and weighs
// each by a softmax masked at 2*router_jitter_noise; the arch code carries the
// 0.01 every published checkpoint states, so another value is refused rather
// than run at the wrong threshold. input_jitter_noise perturbs training only
// and is not read. A window shorter than the context slides every layer, as
// phi3's does on both inputs.
func phimoeHFConfig(c *hfConfig, names []string) (*Config, error) {
	if raw, ok := c.keys["router_jitter_noise"]; ok {
		var j float64
		if err := json.Unmarshal(raw, &j); err != nil {
			return nil, fmt.Errorf("convert: %s: router_jitter_noise: %w", c.path, err)
		}
		if float32(j) != PhiMoEJitter {
			return nil, fmt.Errorf("convert: %s: router_jitter_noise %v, and phimoe's sparsemixer "+
				"threshold is the arch's %v: %w", c.path, j, PhiMoEJitter, ErrNotImplemented)
		}
	}
	if c.NumExpertsPerTok != 2 {
		return nil, fmt.Errorf("convert: %s: num_experts_per_tok %d, and sparsemixer selects 2: %w",
			c.path, c.NumExpertsPerTok, ErrNotImplemented)
	}
	lr, err := hfLongRopeConsume(c)
	if err != nil {
		return nil, err
	}
	win := uint32(0)
	if c.SlidingWindow != nil {
		win, c.SlidingWindow = *c.SlidingWindow, nil
	}
	cfg, err := hfConfigWith(ArchPhiMoE, jlm.FlagLayerNorm|jlm.FlagRopeNeox, "silu")(c, names)
	if err != nil {
		return nil, err
	}
	if cfg.NExpert < 2 {
		return nil, fmt.Errorf("convert: %s: num_local_experts %d: phimoe is a mixture", c.path, cfg.NExpert)
	}
	// sparsemixer's weights are not renormalised, whatever norm_topk_prob says.
	cfg.Flags |= jlm.FlagNoExpertNorm
	if win > 0 && win < cfg.NCtx {
		cfg.SWAWindow, cfg.SWAPeriod = win, allLocal(cfg.NLayer)
	}
	if lr != nil {
		if err := lr.apply(c, cfg); err != nil {
			return nil, err
		}
	}
	return cfg, nil
}

// hfLongRope is phi3's LongRoPE as config.json states it: a factor per rotary
// pair for positions below original_max_position_embeddings and another past
// it, and the cos/sin magnitude.
type hfLongRope struct {
	Type       string    `json:"type"`
	RopeType   string    `json:"rope_type"`
	Short      []float64 `json:"short_factor"`
	Long       []float64 `json:"long_factor"`
	ShortScale *float64  `json:"short_mscale"`
	AttnFactor *float64  `json:"attention_factor"`
	OrigCtx    uint32    `json:"original_max_position_embeddings"`
}

// hfLongRopeConsume reads a LongRoPE rope_scaling and removes it from the
// config, so llamaHFConfig's refusal of every scaling does not fire on it.
// Any other scaling is left for that refusal.
func hfLongRopeConsume(c *hfConfig) (*hfLongRope, error) {
	if len(c.RopeScaling) == 0 || string(c.RopeScaling) == "null" {
		return nil, nil
	}
	var lr hfLongRope
	if err := json.Unmarshal(c.RopeScaling, &lr); err != nil {
		return nil, fmt.Errorf("convert: %s: rope_scaling: %w", c.path, err)
	}
	kind := lr.RopeType
	if kind == "" {
		kind = lr.Type
	}
	if kind != "longrope" && kind != "su" {
		return nil, nil
	}
	if lr.OrigCtx == 0 {
		if raw, ok := c.keys["original_max_position_embeddings"]; ok {
			if err := json.Unmarshal(raw, &lr.OrigCtx); err != nil {
				return nil, fmt.Errorf("convert: %s: original_max_position_embeddings: %w", c.path, err)
			}
		}
	}
	c.RopeScaling = nil
	return &lr, nil
}

// apply writes the factors as the rope_factors_short/long tensors the GGUF
// carries (convert_hf_to_gguf.py writes the same two vectors), the original
// context the engine keys the long factors on (model.RopeFor), and the
// magnitude into AttnFactor. The magnitude is the class's: PhimoeRotaryEmbedding multiplies
// cos and sin by short_mscale, and transformers' LongRoPE by attention_factor,
// or sqrt(1 + ln(ctx/orig)/ln(orig)) where neither is stated.
func (lr *hfLongRope) apply(c *hfConfig, cfg *Config) error {
	ts, err := lr.state(c.path, cfg)
	if err != nil {
		return err
	}
	c.extra = append(c.extra, ts...)
	return nil
}

// state is apply's statement for a config read from path: the two factor
// vectors, with the original context and the magnitude set on cfg.
func (lr *hfLongRope) state(path string, cfg *Config) ([]jlm.Tensor, error) {
	pairs := int(cfg.NRot / 2)
	if len(lr.Short) != pairs || len(lr.Long) != pairs {
		return nil, fmt.Errorf("convert: %s: LongRoPE factors %d short and %d long for %d rotary pairs",
			path, len(lr.Short), len(lr.Long), pairs)
	}
	if lr.OrigCtx == 0 {
		return nil, fmt.Errorf("convert: %s: LongRoPE without original_max_position_embeddings", path)
	}
	cfg.YarnOrigCtx = lr.OrigCtx
	switch {
	case lr.ShortScale != nil:
		cfg.AttnFactor = float32(*lr.ShortScale)
	case lr.AttnFactor != nil:
		cfg.AttnFactor = float32(*lr.AttnFactor)
	case cfg.NCtx > lr.OrigCtx:
		s := float64(cfg.NCtx) / float64(lr.OrigCtx)
		cfg.AttnFactor = float32(math.Sqrt(1 + math.Log(s)/math.Log(float64(lr.OrigCtx))))
	}
	vec := func(role jlm.Role, name string, f []float64) jlm.Tensor {
		b := make([]byte, 4*len(f))
		for i, x := range f {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(float32(x)))
		}
		return jlm.Tensor{Role: role, Block: jlm.DenseBlock, Index: -1, Type: jlm.TypeF32, NDim: 1,
			Dims: [4]uint64{uint64(len(f))}, Data: b, Name: name}
	}
	return []jlm.Tensor{
		vec(jlm.RoleRopeFactorsShort, "rope_factors_short.weight", lr.Short),
		vec(jlm.RoleRopeFactorsLong, "rope_factors_long.weight", lr.Long)}, nil
}

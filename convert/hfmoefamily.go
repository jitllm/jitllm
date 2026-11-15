package convert

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
)

// The mixture family from safetensors: Hunyuan (HunYuanDenseV1ForCausalLM,
// HunYuanMoEV1ForCausalLM). Each converts to the container its GGUF converts
// to -- the same arch, flags and tensors -- so the two inputs are one model;
// convert/moefamily.go and convert/hunyuan.go are the GGUF half and say why
// each graph is what it is. Gated by testdata/golden/hf in
// model.TestSafetensorsMatchTransformers on the fixtures scripts/moegold.py
// builds with each family's own class.

// hfUint is a count config.json writes as a number or as a list of one value
// per layer (Hunyuan's moe_topk and moe_intermediate_size). A list whose
// entries differ is refused: the container has one value per model.
type hfUint uint32

func (u *hfUint) UnmarshalJSON(b []byte) error {
	var n uint32
	if err := json.Unmarshal(b, &n); err == nil {
		*u = hfUint(n)
		return nil
	}
	var l []uint32
	if err := json.Unmarshal(b, &l); err != nil {
		return err
	}
	for i := range l {
		if l[i] != l[0] {
			return fmt.Errorf("a per-layer list %v whose entries differ, and the container "+
				"carries one value: %w", l, ErrNotImplemented)
		}
	}
	if len(l) > 0 {
		*u = hfUint(l[0])
	}
	return nil
}

// hardcodesSoftmax is hardcodesSigmoid's twin (RULE 7m): these classes route
// with a softmax whatever config.json says, so a file stating another gate
// describes weights the class never ran and is refused.
func hardcodesSoftmax(archs []string) bool {
	for _, a := range archs {
		switch a {
		case "HunYuanMoEV1ForCausalLM":
			return true
		}
	}
	return false
}

// hfGateAgrees refuses a scoring_func that contradicts the class's own gate.
func hfGateAgrees(c *hfConfig) error {
	want := ""
	switch {
	case hardcodesSigmoid(c.Architectures):
		want = "sigmoid"
	case hardcodesSoftmax(c.Architectures):
		want = "softmax"
	}
	if want != "" && c.ScoringFunc != "" && c.ScoringFunc != want {
		return fmt.Errorf("convert: %s: scoring_func %q, and %s gates with a %s whatever "+
			"the file says: %w", c.path, c.ScoringFunc, strings.Join(c.Architectures, ", "), want,
			ErrNotImplemented)
	}
	return nil
}

// hfQKNormPresent reports whether block 0 ships a per-head q/k norm under
// these two names, refusing a block that ships one of the two.
func hfQKNormPresent(c *hfConfig, names []string, q, k string) (bool, error) {
	var hq, hk bool
	for _, n := range names {
		switch n {
		case "model.layers.0." + q:
			hq = true
		case "model.layers.0." + k:
			hk = true
		}
	}
	if hq != hk {
		return false, fmt.Errorf("convert: %s: block 0 carries a q norm %v and a k norm %v", c.path, hq, hk)
	}
	return hq, nil
}

// --- Hunyuan ----------------------------------------------------------------

// hunyuanHF is both Hunyuan classes; a mixture is read off num_experts. The
// q/k norm follows the rotary and is folded at conversion (see convert/hunyuan.go
// and jlm.FlagQKNormPostRope); the checkpoint is NEOX and unpermuted.
func hunyuanHF() *hfArch {
	return &hfArch{
		arch:        ArchHunyuan,
		blockPrefix: llamaHF().blockPrefix,
		model:       llamaHF().model,
		block: blockPlus(llamaHF().block, map[string]jlm.Role{
			"self_attn.query_layernorm.weight": jlm.RoleAttnQNorm,
			"self_attn.key_layernorm.weight":   jlm.RoleAttnKNorm,
			"mlp.gate.wg.weight":               jlm.RoleRouter,
			"mlp.shared_mlp.gate_proj.weight":  jlm.RoleShExpGate,
			"mlp.shared_mlp.up_proj.weight":    jlm.RoleShExpUp,
			"mlp.shared_mlp.down_proj.weight":  jlm.RoleShExpDown,
		}),
		expertPrefix: "mlp.experts.",
		expert:       hfExperts(),
		ignore:       llamaHF().ignore,
		config:       hunyuanHFConfig,
	}
}

// hunyuanHFConfig reads Tencent's keys. rope_scaling "dynamic" with an alpha
// is a base, rope_theta * alpha^(d/(d-2)) over head_dim -- the class's
// DynamicNTKAlphaRotary and llama.cpp's converter both -- and its dynamic
// update fires only past max_position_embeddings, which is the container's
// context. The mixture: num_experts, moe_topk, one shared MLP as wide as
// intermediate_size, a renormalised softmax. transformers builds the experts
// from intermediate_size and llama.cpp from moe_intermediate_size; a file
// where they differ is refused (RULE 7m: the references would disagree).
func hunyuanHFConfig(c *hfConfig, names []string) (*Config, error) {
	bad := func(f string, a ...any) error { return fmt.Errorf("convert: "+c.path+": "+f, a...) }
	if err := hfGateAgrees(c); err != nil {
		return nil, err
	}
	if c.UseCLA != nil && *c.UseCLA {
		return nil, fmt.Errorf("convert: %s: use_cla (cross-layer attention) is not "+
			"implemented: %w", c.path, ErrNotImplemented)
	}
	head := c.HiddenSize / max(c.NumAttentionHeads, 1)
	if c.HeadDim != nil && *c.HeadDim != 0 {
		head = *c.HeadDim
	}
	base, err := hunyuanRopeBase(c, head)
	if err != nil {
		return nil, err
	}
	nexp := c.NumExperts
	moe := nexp > 1
	topk, expW, shared := c.MoETopK, c.MoEIntermediateSize, c.NumSharedExpert
	// hfMoEConfig reads none of Hunyuan's spellings; keep it from seeing a
	// mixture it would misread.
	c.NumExperts, c.MoEIntermediateSize = 0, 0
	cfg, err := hfConfigWith(ArchHunyuan, jlm.FlagRopeNeox, "silu")(c, names)
	if err != nil {
		return nil, err
	}
	cfg.RopeBase = base
	qk, err := hfQKNormPresent(c, names, "self_attn.query_layernorm.weight", "self_attn.key_layernorm.weight")
	if err != nil {
		return nil, err
	}
	if qk {
		cfg.Flags |= jlm.FlagQKNorm | jlm.FlagQKNormPostRope
	}
	if !moe {
		return cfg, nil
	}
	if expW != 0 && uint32(expW) != c.IntermediateSize {
		return nil, fmt.Errorf("convert: %s: moe_intermediate_size %d against intermediate_size %d: "+
			"transformers builds the experts from the second and llama.cpp from the first: %w",
			c.path, expW, c.IntermediateSize, ErrNotImplemented)
	}
	if shared != 0 && shared != 1 {
		return nil, fmt.Errorf("convert: %s: num_shared_expert %d, and the class builds one "+
			"shared MLP: %w", c.path, shared, ErrNotImplemented)
	}
	cfg.NExpert = nexp
	cfg.NExpertUsed, cfg.NFFNExp, cfg.NFFNShExp = uint32(topk), c.IntermediateSize, c.IntermediateSize
	if cfg.NExpert == 0 || cfg.NExpertUsed == 0 || cfg.NExpertUsed > cfg.NExpert {
		return nil, bad("moe_topk %d of %d experts", topk, cfg.NExpert)
	}
	return cfg, nil
}

// hunyuanRopeBase is the rotary base, and consumes the rope_scaling it reads
// so llamaHFConfig's refusal does not see it. Anything but none, default or
// dynamic-with-alpha is left for that refusal.
func hunyuanRopeBase(c *hfConfig, head uint32) (float32, error) {
	theta := c.RopeTheta
	if theta == 0 {
		theta = 10000
	}
	if len(c.RopeScaling) == 0 || string(c.RopeScaling) == "null" {
		return float32(theta), nil
	}
	var m map[string]any
	if err := json.Unmarshal(c.RopeScaling, &m); err != nil {
		return 0, fmt.Errorf("convert: %s: rope_scaling does not parse: %w", c.path, err)
	}
	kind, _ := m["rope_type"].(string)
	if kind == "" {
		kind, _ = m["type"].(string)
	}
	alpha, _ := m["alpha"].(float64)
	if kind != "dynamic" || alpha == 0 {
		return float32(theta), nil
	}
	if head <= 2 {
		return 0, fmt.Errorf("convert: %s: head_dim %d", c.path, head)
	}
	c.RopeScaling = nil
	d := float64(head)
	return float32(theta * math.Pow(alpha, d/(d-2))), nil
}

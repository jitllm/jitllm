package convert

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
)

// The mixture families from safetensors: GLM-4.5 (Glm4MoeForCausalLM),
// Qwen1.5-MoE (Qwen2MoeForCausalLM), ERNIE 4.5 (Ernie4_5ForCausalLM,
// Ernie4_5_MoeForCausalLM) and Hunyuan (HunYuanDenseV1ForCausalLM,
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
		case "Qwen2MoeForCausalLM", "Ernie4_5_MoeForCausalLM", "HunYuanMoEV1ForCausalLM":
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

// hfRotaryFraction takes partial_rotary_factor out of the config for a class
// that rotates part of each head, returning how many dimensions turn.
// llamaHFConfig refuses the key on every other class.
func hfRotaryFraction(c *hfConfig, headDim uint32) (uint32, error) {
	f := 1.0
	if c.PartialRotaryFactor != nil {
		f = *c.PartialRotaryFactor
	}
	c.PartialRotaryFactor = nil
	n := float64(headDim) * f
	if n <= 0 || n != math.Trunc(n) || uint32(n)%2 != 0 || uint32(n) > headDim {
		return 0, fmt.Errorf("convert: %s: partial_rotary_factor %v of head_dim %d is not "+
			"an even count of dimensions", c.path, f, headDim)
	}
	return uint32(n), nil
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

// --- GLM-4.5 -------------------------------------------------------------

// glm4moeHF is GLM-4.5/4.6/4.5-Air: DeepSeek-V3's router and mixture keys on
// GQA attention with biased q/k/v, an optional per-head q/k norm (use_qk_norm)
// and NEOX rotary over partial_rotary_factor of each head, with the prediction
// block (num_nextn_predict_layers) carried and set aside as DeepSeek's is.
func glm4moeHF() *hfArch {
	return &hfArch{
		arch:        jlm.ArchGLM4MoE,
		blockPrefix: llamaHF().blockPrefix,
		model:       llamaHF().model,
		block: blockPlus(qwen3Block(), map[string]jlm.Role{
			"mlp.gate.weight":                     jlm.RoleRouter,
			"mlp.gate.e_score_correction_bias":    jlm.RoleExpProbsB,
			"mlp.shared_experts.gate_proj.weight": jlm.RoleShExpGate,
			"mlp.shared_experts.up_proj.weight":   jlm.RoleShExpUp,
			"mlp.shared_experts.down_proj.weight": jlm.RoleShExpDown,
			"eh_proj.weight":                      jlm.RoleNextnEHProj,
			"enorm.weight":                        jlm.RoleNextnENorm,
			"hnorm.weight":                        jlm.RoleNextnHNorm,
			"shared_head.norm.weight":             jlm.RoleNextnHeadNorm,
			"shared_head.head.weight":             jlm.RoleNextnHead,
			"embed_tokens.weight":                 jlm.RoleNextnEmbd,
		}),
		expertPrefix: "mlp.experts.",
		expert:       hfExperts(),
		ignore:       llamaHF().ignore,
		config:       glm4moeHFConfig,
	}
}

// glm4moeHFConfig reads GLM's keys. The gate is a sigmoid whatever the file
// says (Glm4MoeTopkRouter hardcodes it; hardcodesSigmoid), as on the GGUF side.
func glm4moeHFConfig(c *hfConfig, names []string) (*jlm.Config, error) {
	if err := hfGateAgrees(c); err != nil {
		return nil, err
	}
	head := c.HiddenSize / max(c.NumAttentionHeads, 1)
	if c.HeadDim != nil && *c.HeadDim != 0 {
		head = *c.HeadDim
	}
	rot, err := hfRotaryFraction(c, head)
	if err != nil {
		return nil, err
	}
	if c.NumNextnPredictLayers > 0 {
		first := fmt.Sprintf("model.layers.%d.", c.NumHiddenLayers)
		for _, n := range names {
			if strings.HasPrefix(n, first) {
				c.nextn = c.NumNextnPredictLayers
				break
			}
		}
	}
	cfg, err := hfConfigWith(jlm.ArchGLM4MoE, jlm.FlagRopeNeox, "silu")(c, names)
	if err != nil {
		return nil, err
	}
	cfg.NRot = rot
	qk, err := hfQKNormPresent(c, names, "self_attn.q_norm.weight", "self_attn.k_norm.weight")
	if err != nil {
		return nil, err
	}
	if qk {
		cfg.Flags |= jlm.FlagQKNorm
	}
	cfg.NMTP = c.nextn
	if err := deepseekMoE(c, cfg, names, "mlp."); err != nil {
		return nil, err
	}
	if !cfg.Flags.Has(jlm.FlagExpertSigmoid) {
		return nil, fmt.Errorf("convert: %s: a GLM-4.5 router that is not a sigmoid", c.path)
	}
	return cfg, nil
}

// --- Qwen1.5-MoE -----------------------------------------------------------

// qwen2moeHF is qwen2's attention with a softmax top-k beside a shared expert
// scaled by its own sigmoid gate (RoleShRouter, qwen3next's).
func qwen2moeHF() *hfArch {
	return &hfArch{
		arch:        jlm.ArchQwen2MoE,
		blockPrefix: llamaHF().blockPrefix,
		model:       llamaHF().model,
		block: blockPlus(llamaHF().block, map[string]jlm.Role{
			"mlp.gate.weight":                    jlm.RoleRouter,
			"mlp.shared_expert.gate_proj.weight": jlm.RoleShExpGate,
			"mlp.shared_expert.up_proj.weight":   jlm.RoleShExpUp,
			"mlp.shared_expert.down_proj.weight": jlm.RoleShExpDown,
			"mlp.shared_expert_gate.weight":      jlm.RoleShRouter,
		}),
		expertPrefix: "mlp.experts.",
		expert:       hfExperts(),
		ignore:       llamaHF().ignore,
		config:       qwen2moeHFConfig,
	}
}

// qwen2moeHFConfig is hfMoEConfig's mixture plus the shared expert, whose
// width hfMoEConfig refuses for every other class. The weights are not
// renormalised unless norm_topk_prob says so (Qwen2MoeConfig's default, every
// published config, llama.cpp's literal).
func qwen2moeHFConfig(c *hfConfig, names []string) (*jlm.Config, error) {
	if err := hfGateAgrees(c); err != nil {
		return nil, err
	}
	sh := c.SharedExpertIntermediateSize
	c.SharedExpertIntermediateSize = 0
	cfg, err := hfConfigWith(jlm.ArchQwen2MoE, jlm.FlagRopeNeox, "silu")(c, names)
	if err != nil {
		return nil, err
	}
	if cfg.NExpert == 0 {
		return nil, fmt.Errorf("convert: %s: no num_experts: qwen2moe is a mixture", c.path)
	}
	if sh == 0 {
		return nil, fmt.Errorf("convert: %s: no shared_expert_intermediate_size", c.path)
	}
	cfg.NFFNShExp = sh
	return cfg, nil
}

// --- ERNIE 4.5 -------------------------------------------------------------

// ernie45HF is ERNIE 4.5's dense line: llama's graph exactly, as its GGUF
// converts. Its checkpoint rotates adjacent pairs already (the class's
// rotate_half interleaves), which is the container's own layout, so unlike
// llama nothing is permuted.
func ernie45HF() *hfArch {
	return &hfArch{
		arch:        jlm.ArchLlama,
		blockPrefix: llamaHF().blockPrefix,
		model:       llamaHF().model,
		block:       llamaHF().block,
		ignore:      llamaHF().ignore,
		config:      hfConfigWith(jlm.ArchLlama, 0, "silu"),
	}
}

// ernie45moeHF is the mixture: a softmax top-k whose selection adds a
// per-expert bias to the softmax probability (moe_statics), renormalised,
// beside shared experts, after a dense lead and interleaved by a step.
func ernie45moeHF() *hfArch {
	return &hfArch{
		arch:        jlm.ArchErnie45MoE,
		blockPrefix: llamaHF().blockPrefix,
		model:       llamaHF().model,
		block: blockPlus(llamaHF().block, map[string]jlm.Role{
			"mlp.gate.weight":                         jlm.RoleRouter,
			"mlp.moe_statics.e_score_correction_bias": jlm.RoleExpProbsB,
			"mlp.shared_experts.gate_proj.weight":     jlm.RoleShExpGate,
			"mlp.shared_experts.up_proj.weight":       jlm.RoleShExpUp,
			"mlp.shared_experts.down_proj.weight":     jlm.RoleShExpDown,
		}),
		expertPrefix: "mlp.experts.",
		expert:       hfExperts(),
		ignore:       llamaHF().ignore,
		config:       ernie45moeHFConfig,
	}
}

// ernie45moeHFConfig reads ERNIE's own key names. A block is a mixture when
// (il+1) % moe_layer_interval == 0 and il lies in [moe_layer_start_index,
// moe_layer_end_index]; the container's rule is the first two (NDenseLead,
// MoEStep), so an end before the last block is refused.
func ernie45moeHFConfig(c *hfConfig, names []string) (*jlm.Config, error) {
	if err := hfGateAgrees(c); err != nil {
		return nil, err
	}
	bad := func(f string, a ...any) error { return fmt.Errorf("convert: "+c.path+": "+f, a...) }
	cfg, err := hfConfigWith(jlm.ArchErnie45MoE, 0, "silu")(c, names)
	if err != nil {
		return nil, err
	}
	n, k, w := c.MoENumExperts, c.MoEK, uint32(c.MoEIntermediateSize)
	if n == 0 || k == 0 || k > n || w == 0 {
		return nil, bad("moe_num_experts %d, moe_k %d, moe_intermediate_size %d", n, k, w)
	}
	cfg.NExpert, cfg.NExpertUsed, cfg.NFFNExp = n, k, w
	cfg.NFFNShExp = c.MoENumSharedExperts * w
	step := max(c.MoELayerInterval, 1)
	start := uint32(0)
	if c.MoELayerStartIndex != nil {
		if *c.MoELayerStartIndex < 0 || uint32(*c.MoELayerStartIndex) > cfg.NLayer {
			return nil, bad("moe_layer_start_index %d", *c.MoELayerStartIndex)
		}
		start = uint32(*c.MoELayerStartIndex)
	}
	if c.MoELayerEndIndex != nil && *c.MoELayerEndIndex >= 0 && uint32(*c.MoELayerEndIndex)+1 < cfg.NLayer {
		return nil, fmt.Errorf("convert: %s: moe_layer_end_index %d ends the mixtures before "+
			"block %d: %w", c.path, *c.MoELayerEndIndex, cfg.NLayer-1, ErrNotImplemented)
	}
	cfg.NDenseLead, cfg.MoEStep = start, step
	// Which blocks route is read off the names as well as computed: a key that
	// disagrees with the weights is a model nothing would read correctly.
	for li := uint32(0); li < cfg.NLayer; li++ {
		want := li >= start && (li+1)%step == 0
		has := false
		for _, nm := range names {
			if nm == fmt.Sprintf("model.layers.%d.mlp.gate.weight", li) {
				has = true
				break
			}
		}
		if has != want {
			return nil, bad("block %d has a router %v and the step says %v", li, has, want)
		}
	}
	return cfg, nil
}

// --- Hunyuan ----------------------------------------------------------------

// hunyuanHF is both Hunyuan classes; a mixture is read off num_experts. The
// q/k norm follows the rotary and is folded at conversion (see convert/hunyuan.go
// and jlm.FlagQKNormPostRope); the checkpoint is NEOX and unpermuted.
func hunyuanHF() *hfArch {
	return &hfArch{
		arch:        jlm.ArchHunyuan,
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
func hunyuanHFConfig(c *hfConfig, names []string) (*jlm.Config, error) {
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
	cfg, err := hfConfigWith(jlm.ArchHunyuan, jlm.FlagRopeNeox, "silu")(c, names)
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

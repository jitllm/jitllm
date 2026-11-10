package convert

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
)

// The modern text group from safetensors (convert/moefamily2.go is the GGUF
// half and says why each graph is what it is). Each converts to the container
// its GGUF converts to: the same arch, flags and tensors.

// dots1HF is dots.llm1: DeepSeek-V3's mixture keys (deepseekMoE) on qwen3's
// attention, a per-head q/k norm and NEOX rotary over the whole head. The gate
// is a sigmoid whatever the file says (Dots1TopkRouter; hardcodesSigmoid).
func dots1HF() *hfArch {
	return &hfArch{
		arch:        ArchDots1,
		blockPrefix: llamaHF().blockPrefix,
		model:       llamaHF().model,
		block: blockPlus(qwen3Block(), map[string]jlm.Role{
			"mlp.gate.weight":                     jlm.RoleRouter,
			"mlp.gate.e_score_correction_bias":    jlm.RoleExpProbsB,
			"mlp.shared_experts.gate_proj.weight": jlm.RoleShExpGate,
			"mlp.shared_experts.up_proj.weight":   jlm.RoleShExpUp,
			"mlp.shared_experts.down_proj.weight": jlm.RoleShExpDown,
		}),
		expertPrefix: "mlp.experts.",
		expert:       hfExperts(),
		ignore:       llamaHF().ignore,
		config:       dots1HFConfig,
	}
}

// dots1HFConfig reads dots.llm1's keys. Its config carries layer_types even
// with no window: every entry must be full attention, the sliding layers being
// what the GGUF side refuses too.
func dots1HFConfig(c *hfConfig, names []string) (*Config, error) {
	if err := hfGateAgrees(c); err != nil {
		return nil, err
	}
	for i, k := range c.LayerTypes {
		if k != "full_attention" {
			return nil, fmt.Errorf("convert: %s: layer %d is %q: dots.llm1's sliding layers are "+
				"not implemented: %w", c.path, i, k, ErrNotImplemented)
		}
	}
	c.LayerTypes = nil
	cfg, err := hfConfigWith(ArchDots1, jlm.FlagRopeNeox|jlm.FlagQKNorm, "silu")(c, names)
	if err != nil {
		return nil, err
	}
	qk, err := hfQKNormPresent(c, names, "self_attn.q_norm.weight", "self_attn.k_norm.weight")
	if err != nil {
		return nil, err
	}
	if !qk {
		return nil, fmt.Errorf("convert: %s: dots.llm1 norms each head of q and k, and block 0 has no norm",
			c.path)
	}
	if err := deepseekMoE(c, cfg, names, "mlp."); err != nil {
		return nil, err
	}
	if !cfg.Flags.Has(jlm.FlagExpertSigmoid) {
		return nil, fmt.Errorf("convert: %s: a dots.llm1 router that is not a sigmoid", c.path)
	}
	return cfg, nil
}

// moe2SigmoidClasses are the group's classes whose router hardcodes the
// sigmoid; hardcodesSigmoid consults them.
func moe2SigmoidClasses() []string {
	return []string{"Dots1ForCausalLM"}
}

// moe2HardcodesSigmoid reports whether any of archs is one of them.
func moe2HardcodesSigmoid(archs []string) bool {
	for _, a := range archs {
		if slices.Contains(moe2SigmoidClasses(), a) {
			return true
		}
	}
	return false
}

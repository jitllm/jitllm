package engine

import (
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
)

// GLM-4-0414's text block (jlm.ArchGLM4), the text model of GLM-4.1V and
// GLM-4.6V-Flash, is gated by the classic-transformer (C6) harness as the
// dense llama family is (densearch_test.go): transformers' own Glm4ForCausalLM
// written to GGUF by llama.cpp's converter (scripts/c6gold.py synth-glm4),
// on decode, prefill and the batch, against the host and every device, and
// through the five principles' gates. Its image side is gated with the Qwen-VL
// family (synth-glm4v, mrope_test.go).
func glm4Fixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{{"synth-glm4", func(m *model.Model) bool {
		c, l := m.Cfg, &m.Layers[0]
		return c.Arch == "glm4" && !c.RopeNeox && c.NRot < c.HeadDim && l.Vector(jlm.RoleAttnQBias) != nil && l.Vector(jlm.RoleAttnKBias) != nil &&
			l.Vector(jlm.RoleAttnVBias) != nil && l.Vector(jlm.RoleAttnOutBias) == nil && l.AttnNorm != nil && l.FfnNorm != nil && l.Vector(jlm.RolePostAttnNorm) != nil &&
			l.Vector(jlm.RolePostFFNNorm) != nil && l.Gate.E != nil && c.NKVHead < c.NHead && !c.TiedEmbd
	}}}
}

func glm4Violations() []c6Violation {
	return []c6Violation{
		{"no q/k/v biases", noQKVBias},
		{"no post-attention norm", noPostAttnNorm},
		{"no post-FFN norm", noPostFFNNorm},
		{"NEOX rotary for NORM", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.RopeNeox = true }) }},
	}
}

func glm4DevViolations() []c6Violation {
	return []c6Violation{
		{"no post-FFN norm", noPostFFNNorm},
		{"NEOX rotary for NORM", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.RopeNeox = true }) }},
	}
}

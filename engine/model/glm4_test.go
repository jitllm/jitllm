package model

// GLM-4-0414's text block (jlm.ArchGLM4), the text model of GLM-4.1V and
// GLM-4.6V-Flash, is gated by the classic-transformer (C6) harness as the
// dense llama family is (densearch_test.go): transformers' own Glm4ForCausalLM
// written to GGUF by llama.cpp's converter (scripts/c6gold.py synth-glm4),
// on decode, prefill and the batch, against the host and every device, and
// through the five principles' gates. Its image side is gated with the Qwen-VL
// family (synth-glm4v, mrope_test.go).
func init() {
	c6Fixtures = append(c6Fixtures, struct {
		name string
		sel  func(m *Model) bool
	}{"synth-glm4", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "glm4" && !c.RopeNeox && c.NRot < c.HeadDim && l.bq != nil && l.bk != nil &&
			l.bv != nil && l.bo == nil && l.attnNorm != nil && l.ffnNorm != nil && l.postAttnNorm != nil &&
			l.postFFNNorm != nil && l.gate.e != nil && c.NKVHead < c.NHead && !c.TiedEmbd
	}})
	c6Violations["synth-glm4"] = []c6Violation{
		{"no q/k/v biases", noQKVBias},
		{"no post-attention norm", noPostAttnNorm},
		{"no post-FFN norm", noPostFFNNorm},
		{"NEOX rotary for NORM", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = true }) }},
	}
	c6DevViolations["synth-glm4"] = []c6Violation{
		{"no post-FFN norm", noPostFFNNorm},
		{"NEOX rotary for NORM", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = true }) }},
	}
	principleFixtures = append(principleFixtures, "synth-glm4.gguf")
}

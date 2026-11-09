package engine

import (
	"math"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
)

// Granite 4.0-H (jlm.ArchGraniteHybrid) on the Mamba-2 harness
// (ssmfamily_test.go): its dense and mixture fixtures, built by
// GraniteMoeHybridForCausalLM (scripts/ssmgold.py), join every gate there and
// the dense one the five principles' through principleFixtures.
func granitehybridFixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{
		{"synth-granitehybrid", func(m *model.Model) bool {
			c := m.Cfg
			return c.SSD() && c.NoPosEnc && c.LogitScale == float64(float32(3.5)) &&
				c.ResidualScale == float64(float32(0.35)) && c.EmbdScale == float64(float32(6)) &&
				c.AttnScale == float64(float32(0.09)) && c.NExpert == 0 && c.NKVHead == 2 && c.NHead == 4 &&
				c.LayerKind(0).Recurrent() && !c.LayerKind(1).Recurrent() && c.LayerKind(2).Recurrent() &&
				!m.Layers[0].NoFFN && !m.Layers[1].NoFFN && m.Layers[1].Gate.E != nil && len(m.Layers[0].Vector(jlm.RoleSSMD)) == 8 &&
				!c.TiedEmbd
		}},
		{"synth-granitehybridmoe", func(m *model.Model) bool {
			c := m.Cfg
			return c.SSD() && c.NoPosEnc && c.NExpert == 8 && c.NExpertUsed == 2 &&
				c.NFFNShExp > 0 && !c.LayerKind(2).Recurrent() && c.LayerKind(3).Recurrent() &&
				m.Layers[0].Router.E != nil && m.Layers[0].ShGate.E != nil && m.Layers[0].Vector(jlm.RoleShRouter) == nil &&
				c.TiedEmbd
		}},
	}
}

// granitehybridViolations are Granite 4.0-H's own features beside its Mamba-2
// blocks: the four scales and the attention's missing rotary.
func granitehybridViolations() []c6Violation {
	return []c6Violation{
		{"rotary on the attention block", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) { c.NoPosEnc = false })
		}},
		{"no residual scale", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.ResidualScale = 1 }) }},
		{"no logit scale", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.LogitScale = 1 }) }},
		{"the score at 1/sqrt(head_dim)", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) { c.AttnScale = 1 / math.Sqrt(float64(c.HeadDim)) })
		}},
	}
}

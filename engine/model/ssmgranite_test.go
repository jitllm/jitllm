package model

import "math"

// Granite 4.0-H (jlm.ArchGraniteHybrid) on the Mamba-2 harness
// (ssmfamily_test.go): its dense and mixture fixtures, built by
// GraniteMoeHybridForCausalLM (scripts/ssmgold.py), join every gate there and
// the dense one the five principles' through principleFixtures.
func init() {
	ssmFixtures = append(ssmFixtures, granitehybridFixtures()...)
	for _, fx := range granitehybridFixtures() {
		ssmFamilyViolations[fx.name] = func(bool) []c6Violation { return granitehybridViolations() }
	}
	// The mixture pages under TestSSMFamiliesPageWithTheSameAnswer:
	// TestPagingDoesNotChangeTheAnswer counts frames of one page size, and its
	// expert pages are another.
	principleFixtures = append(principleFixtures, "synth-granitehybrid.gguf")
}

func granitehybridFixtures() []struct {
	name string
	sel  func(m *Model) bool
} {
	return []struct {
		name string
		sel  func(m *Model) bool
	}{
		{"synth-granitehybrid", func(m *Model) bool {
			c := m.Cfg
			return c.Arch == "granitehybrid" && c.SSD() && c.NoPosEnc && c.LogitScale == float64(float32(3.5)) &&
				c.ResidualScale == float64(float32(0.35)) && c.EmbdScale == float64(float32(6)) &&
				c.AttnScale == float64(float32(0.09)) && c.NExpert == 0 && c.NKVHead == 2 && c.NHead == 4 &&
				c.LayerKind(0).Recurrent() && !c.LayerKind(1).Recurrent() && c.LayerKind(2).Recurrent() &&
				!m.layers[0].noFFN && !m.layers[1].noFFN && m.layers[1].gate.e != nil && len(m.layers[0].ssmD) == 8 &&
				!c.TiedEmbd
		}},
		{"synth-granitehybridmoe", func(m *Model) bool {
			c := m.Cfg
			return c.Arch == "granitehybrid" && c.SSD() && c.NoPosEnc && c.NExpert == 8 && c.NExpertUsed == 2 &&
				c.NFFNShExp > 0 && !c.LayerKind(2).Recurrent() && c.LayerKind(3).Recurrent() &&
				m.layers[0].router.e != nil && m.layers[0].shGate.e != nil && m.layers[0].shRouter == nil &&
				c.TiedEmbd
		}},
	}
}

// granitehybridViolations are Granite 4.0-H's own features beside its Mamba-2
// blocks: the four scales and the attention's missing rotary.
func granitehybridViolations() []c6Violation {
	return []c6Violation{
		{"rotary on the attention block", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoPosEnc = false })
		}},
		{"no residual scale", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ResidualScale = 1 }) }},
		{"no logit scale", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.LogitScale = 1 }) }},
		{"the score at 1/sqrt(head_dim)", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.AttnScale = 1 / math.Sqrt(float64(c.HeadDim)) })
		}},
	}
}

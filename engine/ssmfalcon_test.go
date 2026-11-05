package engine

import (
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
)

// Falcon-H1 (jlm.ArchFalconH1) on the Mamba-2 harness (ssmfamily_test.go):
// FalconH1ForCausalLM with every muP multiplier far from one
// (scripts/ssmgold.py), with the grouped gated norm every published size but
// 0.5B carries and without it, as 0.5B ships.
// falconh1Block is what every Falcon-H1 block must carry: attention and a
// Mamba-2 mixer both, the mixer's projection beside the attention's four.
func falconh1Block(m *model.Model, norm bool) bool {
	c := m.Cfg
	ok := c.Arch == "falcon-h1" && c.SSD() && c.SSDAttn() && c.RopeNeox && !c.NoPosEnc &&
		c.NHead == 4 && c.NKVHead == 2 && !c.TiedEmbd
	for i := range m.Layers {
		l := &m.Layers[i]
		ok = ok && c.LayerKind(i) == jlm.LayerSSDAttn && !l.NoFFN && l.SsmIn.Rows == 128+2*int(c.SSM.Groups)*16 &&
			l.Wq.Rows == 64 && l.Wk.Rows == 32 && len(l.Vector(jlm.RoleSSMD)) == 8 && (l.Vector(jlm.RoleSSMNorm) != nil) == norm
	}
	return ok
}

func falconh1Fixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{
		{"synth-falconh1", func(m *model.Model) bool {
			return falconh1Block(m, true) && m.Cfg.SSM.Groups == 2 && m.Cfg.SSMNormGroups == 2
		}},
		{"synth-falconh1-nonorm", func(m *model.Model) bool { return falconh1Block(m, false) && m.Cfg.SSM.Groups == 1 }},
	}
}

// falconh1Violations are Falcon-H1's own: each half of the parallel block
// dropped from the sum, and the rotary's pairing.
func falconh1Violations(bool) []c6Violation {
	return []c6Violation{
		{"NORM rotary for NEOX", func(m *model.Model) func() {
			was := m.Rope.Neox
			m.Rope.Neox = false
			restore := cfgMut(m, func(c *model.Config) { c.RopeNeox = false })
			return func() { m.Rope.Neox = was; restore() }
		}},
		{"the attention half dropped", func(m *model.Model) func() { return halfDropped(m, true) }},
		{"the mixer half dropped", func(m *model.Model) func() { return halfDropped(m, false) }},
	}
}

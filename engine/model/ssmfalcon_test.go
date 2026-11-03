package model

import (
	"github.com/jitllm/jitllm/format/jlm"
)

// Falcon-H1 (jlm.ArchFalconH1) on the Mamba-2 harness (ssmfamily_test.go):
// FalconH1ForCausalLM with every muP multiplier far from one
// (scripts/ssmgold.py), with the grouped gated norm every published size but
// 0.5B carries and without it, as 0.5B ships.
func init() {
	ssmFixtures = append(ssmFixtures, falconh1Fixtures()...)
	for _, fx := range falconh1Fixtures() {
		ssmFamilyViolations[fx.name] = falconh1Violations
	}
	ssmNoNorm["synth-falconh1-nonorm"] = true
	principleFixtures = append(principleFixtures, "synth-falconh1.gguf")
}

// falconh1Block is what every Falcon-H1 block must carry: attention and a
// Mamba-2 mixer both, the mixer's projection beside the attention's four.
func falconh1Block(m *Model, norm bool) bool {
	c := m.Cfg
	ok := c.Arch == "falcon-h1" && c.SSD() && c.SSDAttn() && c.RopeNeox && !c.NoPosEnc &&
		c.NHead == 4 && c.NKVHead == 2 && !c.TiedEmbd
	for i := range m.layers {
		l := &m.layers[i]
		ok = ok && c.LayerKind(i) == jlm.LayerSSDAttn && !l.noFFN && l.ssmIn.rows == 128+2*int(c.SSM.Groups)*16 &&
			l.wq.rows == 64 && l.wk.rows == 32 && len(l.ssmD) == 8 && (l.ssmNorm != nil) == norm
	}
	return ok
}

func falconh1Fixtures() []struct {
	name string
	sel  func(m *Model) bool
} {
	return []struct {
		name string
		sel  func(m *Model) bool
	}{
		{"synth-falconh1", func(m *Model) bool {
			return falconh1Block(m, true) && m.Cfg.SSM.Groups == 2 && m.Cfg.SSMNormGroups == 2
		}},
		{"synth-falconh1-nonorm", func(m *Model) bool { return falconh1Block(m, false) && m.Cfg.SSM.Groups == 1 }},
	}
}

// falconh1Violations are Falcon-H1's own: each half of the parallel block
// dropped from the sum, and the rotary's pairing.
func falconh1Violations(bool) []c6Violation {
	return []c6Violation{
		{"NORM rotary for NEOX", func(m *Model) func() {
			was := m.rope.Neox
			m.rope.Neox = false
			restore := cfgMut(m, func(c *Config) { c.RopeNeox = false })
			return func() { m.rope.Neox = was; restore() }
		}},
		{"the attention half dropped", falconHalf(halfFaultNoAttn)},
		{"the mixer half dropped", falconHalf(halfFaultNoMixer)},
	}
}

// falconKinds runs every block as kind alone: one half of the sum dropped.
// falconHalf drops one half of every block (halfFault): computed, bound and
// weighed by zero, as the container stores both.
func falconHalf(f halfFault) func(m *Model) func() {
	return func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.halfFault = f })
	}
}

package model

import (
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
)

// Mamba-1 (jlm.LayerMamba1) on the Mamba-2 harness (ssmfamily_test.go): plain
// Mamba, FalconMamba and Jamba, each built by its own transformers class
// (scripts/ssmgold.py), join every gate there and the five principles'
// through principleFixtures. Mamba-2's own violations do not apply (its
// norm, its grouped B/C in the convolved channels); the scan's are below.
func init() {
	ssmFixtures = append(ssmFixtures, mamba1Fixtures()...)
	for _, fx := range mamba1Fixtures() {
		ssmNotMamba[fx.name] = true
		principleFixtures = append(principleFixtures, fx.name+".gguf")
	}
	ssmFamilyViolations["synth-mamba1"] = func(bool) []c6Violation { return mamba1Violations() }
	ssmFamilyViolations["synth-mamba1-ragged"] = func(bool) []c6Violation { return mamba1Violations() }
	ssmFamilyViolations["synth-falconmamba"] = func(bool) []c6Violation {
		return append(slices.Clone(mamba1Violations()), mamba1NormViolations()...)
	}
	ssmFamilyViolations["synth-jamba"] = func(bool) []c6Violation {
		return append(append(slices.Clone(mamba1Violations()), mamba1NormViolations()...), jambaViolations()...)
	}
}

func mamba1Fixtures() []struct {
	name string
	sel  func(m *Model) bool
} {
	return []struct {
		name string
		sel  func(m *Model) bool
	}{
		{"synth-mamba1", func(m *Model) bool {
			c := m.Cfg
			ok := c.Arch == "mamba" && c.Mamba1() && c.SSM.Inner == 128 && c.SSM.StateSize == 16 &&
				c.dtRank == 32 && c.NLayer == 3 && !c.TiedEmbd
			for i := range m.layers {
				l := &m.layers[i]
				ok = ok && l.noFFN && c.LayerKind(i).Recurrent() && len(l.ssmA) == 128*16 &&
					len(l.ssmD) == 128 && len(l.ssmConvB) == 128 && l.ssmDtNorm == nil
			}
			return ok
		}},
		// A dt rank of 40, not a whole number of the activation quantizer's
		// blocks (mamba-130m's is 48): every tier quantizes dt's bottleneck with
		// a ragged tail.
		{"synth-mamba1-ragged", func(m *Model) bool {
			c := m.Cfg
			return c.Arch == "mamba" && c.Mamba1() && c.dtRank == 40 && c.NLayer == 3
		}},
		{"synth-falconmamba", func(m *Model) bool {
			c := m.Cfg
			ok := c.Arch == "mamba" && c.Mamba1() && c.NLayer == 3
			for i := range m.layers {
				l := &m.layers[i]
				ok = ok && l.noFFN && len(l.ssmDtNorm) == 32 && len(l.ssmBNorm) == 16 && len(l.ssmCNorm) == 16
			}
			return ok
		}},
		{"synth-jamba", func(m *Model) bool {
			// Four blocks: Mamba-1 with a dense MLP, Mamba-1 with a mixture,
			// attention with a dense MLP, Mamba-1 with a mixture.
			c := m.Cfg
			ok := c.Arch == "jamba" && c.Mamba1() && c.NLayer == 4 && c.NExpert == 4 && c.NExpertUsed == 2 &&
				c.NoExpertNorm && !c.ExpertSigmoid && c.NoPosEnc && c.NKVHead == 2 && c.NFFNExp == 96
			for i, want := range []struct{ rec, moe bool }{{true, false}, {true, true}, {false, false}, {true, true}} {
				l := &m.layers[i]
				ok = ok && c.LayerKind(i).Recurrent() == want.rec && c.MoEAt(i) == want.moe && !l.noFFN
				if want.rec {
					ok = ok && len(l.ssmDtNorm) == 32
				}
			}
			return ok
		}},
	}
}

// mamba1Violations are the scan's own: the decay's sign, the skip, the
// convolution's bias, dt's bias, and B and C exchanged.
func mamba1Violations() []c6Violation {
	return []c6Violation{
		{"A positive (A_log's sign lost)", func(m *Model) func() {
			return ssdScaled(m, func(l *layer) *[]float32 { return &l.ssmA }, func(v []float32) {
				for i := range v {
					v[i] = -v[i]
				}
			})
		}},
		{"no D skip", func(m *Model) func() {
			return ssdScaled(m, func(l *layer) *[]float32 { return &l.ssmD }, func(v []float32) { clear(v) })
		}},
		{"no convolution bias", func(m *Model) func() {
			return ssdScaled(m, func(l *layer) *[]float32 { return &l.ssmConvB }, func(v []float32) { clear(v) })
		}},
		{"no dt bias", func(m *Model) func() {
			return ssdScaled(m, func(l *layer) *[]float32 { return &l.ssmDtBias }, func(v []float32) { clear(v) })
		}},
		{"B and C exchanged", func(m *Model) func() {
			return eachLayer(m, func(l *layer) { l.ssmXB, l.ssmXC = l.ssmXC, l.ssmXB })
		}},
		{"one decay rate a channel (Mamba-2's form)", func(m *Model) func() {
			n := m.Cfg.delta().kDim
			return ssdScaled(m, func(l *layer) *[]float32 { return &l.ssmA }, func(v []float32) {
				for j := 0; j < len(v); j += n {
					for i := 1; i < n; i++ {
						v[j+i] = v[j]
					}
				}
			})
		}},
	}
}

// mamba1NormViolations are the dt, B and C RMSNorms dropped (Jamba's weighted
// ones, FalconMamba's of ones).
func mamba1NormViolations() []c6Violation {
	return []c6Violation{
		{"no dt/B/C norms", func(m *Model) func() {
			undoL := eachLayer(m, func(l *layer) { l.ssmDtNorm, l.ssmBNorm, l.ssmCNorm = nil, nil, nil })
			undoR := dropStored(m, jlm.RoleSSMDtNorm, jlm.RoleSSMBNorm, jlm.RoleSSMCNorm)
			return func() { undoR(); undoL() }
		}},
	}
}

// jambaViolations are Jamba's beside its mixers: rotary on the attention,
// and the routed weights renormalised.
func jambaViolations() []c6Violation {
	return []c6Violation{
		{"rotary on the attention block", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoPosEnc = false })
		}},
		{"the routed weights renormalised", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoExpertNorm = false })
		}},
	}
}

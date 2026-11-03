package model

import (
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
)

// LFM2 (jlm.ArchLFM2) on the state-space harness (ssmfamily_test.go):
// Lfm2ForCausalLM with three gated short convolutions and one attention block
// (scripts/ssmgold.py). Its recurrence is a convolution window alone, which
// TestSSDFeaturesAreLoadBearing's window violation covers.
func lfm2Fixtures() []struct {
	name string
	sel  func(m *Model) bool
} {
	return []struct {
		name string
		sel  func(m *Model) bool
	}{
		{"synth-lfm2", func(m *Model) bool {
			c := m.Cfg
			ok := c.Arch == "lfm2" && c.ShortConv() && !c.SSD() && c.RopeNeox && c.QKNorm && c.NHead == 4 &&
				c.NKVHead == 2 && c.TiedEmbd && c.SSM.ConvKernel == 3
			for i, conv := range []bool{true, false, true, true} {
				l := &m.layers[i]
				ok = ok && c.LayerKind(i).Recurrent() == conv && !l.noFFN && l.gate.e != nil
				if conv {
					ok = ok && c.LayerKind(i) == jlm.LayerShortConv && l.wq.rows == 64 && l.ssmGate.rows == 64 &&
						l.ssmBA.rows == 64 && l.ssmOut.rows == 64 && len(l.ssmConv1d) == 3*64 && l.vector(jlm.RoleSSMA) == nil
				}
			}
			return ok
		}},
		{"synth-lfm2moe", func(m *Model) bool {
			c := m.Cfg
			ok := c.Arch == "lfm2" && c.ShortConv() && c.NExpert == 8 && c.NExpertUsed == 2 && c.ExpertSigmoid &&
				c.NDenseLead == 1 && m.expSelBias() != nil && c.TiedEmbd
			for i, conv := range []bool{true, false, true, true} {
				ok = ok && c.LayerKind(i).Recurrent() == conv && (i == 0) != (m.layers[i].router.e != nil)
			}
			return ok
		}},
	}
}

// lfm2Violations are LFM2's own: the two gates exchanged, x and the outer
// gate exchanged, the window reversed, and the attention block's pieces.
func lfm2Violations(bool) []c6Violation {
	return []c6Violation{
		{"B and C exchanged", func(m *Model) func() {
			return eachLayer(m, func(l *layer) {
				if l.ssmConv1d != nil {
					l.ssmGate, l.ssmBA = l.ssmBA, l.ssmGate
				}
			})
		}},
		{"x and C exchanged (x gating after the convolution)", func(m *Model) func() {
			return eachLayer(m, func(l *layer) {
				if l.ssmConv1d != nil {
					l.wq, l.ssmBA = l.ssmBA, l.wq
				}
			})
		}},
		{"the window's taps reversed", func(m *Model) func() {
			n := m.Cfg.NEmbd
			return eachLayer(m, func(l *layer) {
				if l.ssmConv1d == nil {
					return
				}
				w := slices.Clone(l.ssmConv1d)
				taps := len(w) / n
				for t := 0; t < taps; t++ {
					copy(w[t*n:(t+1)*n], l.ssmConv1d[(taps-1-t)*n:(taps-t)*n])
				}
				l.ssmConv1d = w
				// The composer reads the file's layout, each channel's taps
				// in a row (ssmConvTaps): reversed there too.
				if ct := l.ssmConvTaps; ct.data != nil && ct.rows == n && ct.k == taps {
					rv := make([]float32, n*taps)
					for c := range n {
						for t := range taps {
							rv[c*taps+t] = w[t*n+c]
						}
					}
					l.ssmConvTaps = f32Tensor(rv, n, taps)
				}
			})
		}},
		{"NORM rotary for NEOX", func(m *Model) func() {
			was := m.rope.Neox
			m.rope.Neox = false
			restore := cfgMut(m, func(c *Config) { c.RopeNeox = false })
			return func() { m.rope.Neox = was; restore() }
		}},
		{"no q/k norm", noQKNorm},
	}
}

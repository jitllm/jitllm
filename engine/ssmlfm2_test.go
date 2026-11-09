package engine

import (
	"slices"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
)

// LFM2 (jlm.ArchLFM2) on the state-space harness (ssmfamily_test.go):
// Lfm2ForCausalLM with three gated short convolutions and one attention block
// (scripts/ssmgold.py). Its recurrence is a convolution window alone, which
// TestSSDFeaturesAreLoadBearing's window violation covers.
func lfm2Fixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{
		{"synth-lfm2", func(m *model.Model) bool {
			c := m.Cfg
			ok := c.ShortConv() && !c.SSD() && c.RopeNeox && c.QKNorm && c.NHead == 4 &&
				c.NKVHead == 2 && c.TiedEmbd && c.SSM.ConvKernel == 3
			for i, conv := range []bool{true, false, true, true} {
				l := &m.Layers[i]
				ok = ok && c.LayerKind(i).Recurrent() == conv && !l.NoFFN && l.Gate.E != nil
				if conv {
					ok = ok && c.LayerKind(i) == jlm.LayerShortConv && l.Wq.Rows == 64 && l.SsmGate.Rows == 64 &&
						l.SsmBA.Rows == 64 && l.SsmOut.Rows == 64 && len(l.SsmConv1d) == 3*64 && l.Vector(jlm.RoleSSMA) == nil
				}
			}
			return ok
		}},
		{"synth-lfm2moe", func(m *model.Model) bool {
			c := m.Cfg
			ok := c.ShortConv() && c.NExpert == 8 && c.NExpertUsed == 2 && c.ExpertSigmoid &&
				c.NDenseLead == 1 && m.ExpSelBias() != nil && c.TiedEmbd
			for i, conv := range []bool{true, false, true, true} {
				ok = ok && c.LayerKind(i).Recurrent() == conv && (i == 0) != (m.Layers[i].Router.E != nil)
			}
			return ok
		}},
	}
}

// lfm2Violations are LFM2's own: the two gates exchanged, x and the outer
// gate exchanged, the window reversed, and the attention block's pieces.
func lfm2Violations(bool) []c6Violation {
	return []c6Violation{
		{"B and C exchanged", func(m *model.Model) func() {
			return eachLayer(m, func(l *model.Layer) {
				if l.SsmConv1d != nil {
					l.SsmGate, l.SsmBA = l.SsmBA, l.SsmGate
				}
			})
		}},
		{"x and C exchanged (x gating after the convolution)", func(m *model.Model) func() {
			return eachLayer(m, func(l *model.Layer) {
				if l.SsmConv1d != nil {
					l.Wq, l.SsmBA = l.SsmBA, l.Wq
				}
			})
		}},
		{"the window's taps reversed", func(m *model.Model) func() {
			n := m.Cfg.NEmbd
			return eachLayer(m, func(l *model.Layer) {
				if l.SsmConv1d == nil {
					return
				}
				w := slices.Clone(l.SsmConv1d)
				taps := len(w) / n
				for t := 0; t < taps; t++ {
					copy(w[t*n:(t+1)*n], l.SsmConv1d[(taps-1-t)*n:(taps-t)*n])
				}
				l.SsmConv1d = w
				// The composer reads the file's layout, each channel's taps
				// in a row (ssmConvTaps): reversed there too.
				if ct := l.SsmConvTaps; ct.Data != nil && ct.Rows == n && ct.K == taps {
					rv := make([]float32, n*taps)
					for c := range n {
						for t := range taps {
							rv[c*taps+t] = w[t*n+c]
						}
					}
					l.SsmConvTaps = model.F32Tensor(rv, n, taps)
				}
			})
		}},
		{"NORM rotary for NEOX", func(m *model.Model) func() {
			was := m.Rope.Neox
			m.Rope.Neox = false
			restore := cfgMut(m, func(c *model.Config) { c.RopeNeox = false })
			return func() { m.Rope.Neox = was; restore() }
		}},
		{"no q/k norm", noQKNorm},
	}
}

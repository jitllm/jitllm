package engine

import (
	"slices"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
)

// LFM2 (lfm2) on the state-space harness (ssmfamily_test.go):
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
					ok = ok && c.LayerKind(i) == jlm.LayerShortConv && matrixRows(l, jlm.RoleSCX) == 64 && matrixRows(l, jlm.RoleSCB) == 64 &&
						matrixRows(l, jlm.RoleSCC) == 64 && matrixRows(l, jlm.RoleSSMOut) == 64 && convTaps(l) == 3*64 && l.Vector(jlm.RoleSSMA) == nil
				}
			}
			return ok
		}},
		{"synth-lfm2moe", func(m *model.Model) bool {
			c := m.Cfg
			ok := c.ShortConv() && c.NExpert == 8 && c.NExpertUsed == 2 && c.ExpertSigmoid &&
				c.NDenseLead == 1 && m.ExpSelBias() != nil && c.TiedEmbd
			for i, conv := range []bool{true, false, true, true} {
				ok = ok && c.LayerKind(i).Recurrent() == conv && (i == 0) != (m.Layers[i].Matrix(jlm.RoleRouter) != nil)
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
			return eachSwap(m, jlm.RoleSCB, jlm.RoleSCC)
		}},
		{"x and C exchanged (x gating after the convolution)", func(m *model.Model) func() {
			return eachSwap(m, jlm.RoleSCX, jlm.RoleSCC)
		}},
		{"the window's taps reversed", func(m *model.Model) func() {
			var undo []func()
			undoL := eachLayer(m, func(l *model.Layer) {
				undo = append(undo, setConvTaps(l, func(v []float32, rows, taps int) {
					for c := range rows {
						slices.Reverse(v[c*taps : (c+1)*taps])
					}
				}))
			})
			return func() {
				for _, u := range undo {
					u()
				}
				undoL()
			}
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

// convTaps is the count of block l's convolution taps, every channel's.
func convTaps(l *model.Layer) int {
	if mt := l.Matrix(jlm.RoleSSMConv1d); mt != nil {
		return mt.Rows * mt.K
	}
	return 0
}

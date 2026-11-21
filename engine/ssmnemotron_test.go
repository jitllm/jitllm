package engine

import (
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
)

// Nemotron-H (nemotron_h) on the Mamba-2 harness (ssmfamily_test.go):
// its fixture, built by NemotronHForCausalLM (scripts/ssmgold.py) with one
// mixer a source layer merged at conversion, joins every gate there and the
// five principles' through principleFixtures.
func nemotronhFixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{
		{"synth-nemotronh", func(m *model.Model) bool {
			// "M-M*-MM-": a mixer with its MLP, a mixer alone, the attention with
			// its MLP, a mixer alone, a mixer with its MLP -- five blocks.
			c := m.Cfg
			ok := hasKind(c, jlm.LayerSSD, jlm.LayerSSDAttn) && c.NLayer == 5 && c.SSM.Groups == 2 &&
				c.NoPosEnc && m.Container.Config().Flags.Has(jlm.FlagReLU2) && c.NKVHead == 2 && !c.TiedEmbd
			for i, want := range []struct {
				rec, noFFN bool
			}{{true, false}, {true, true}, {false, false}, {true, true}, {true, false}} {
				l := &m.Layers[i]
				ok = ok && c.LayerKind(i).Recurrent() == want.rec && l.NoFFN == want.noFFN &&
					(l.NoFFN || (l.Gate.E == nil && l.Up.E != nil && l.Vector(jlm.RoleFFNNorm) != nil))
			}
			return ok
		}},
		{"synth-nemotronhmoe", func(m *model.Model) bool {
			// "ME*EMME": a mixer with its mixture, the attention with its mixture,
			// a mixer alone, a mixer with its mixture -- four blocks, the mixtures
			// ungated (up and down, no gate bank) beside an ungated shared expert.
			c := m.Cfg
			ok := hasKind(c, jlm.LayerSSD, jlm.LayerSSDAttn) && c.NLayer == 4 && c.NExpert == 8 && c.NExpertUsed == 2 &&
				c.ExpertSigmoid && !c.NoExpertNorm && c.ExpertScale == 2.5 && c.NExpertGroup == 0 &&
				int(m.Container.Config().NFFNExp) == 64 && c.NFFNShExp == 96 && m.Container.Config().Flags.Has(jlm.FlagReLU2) && c.NoPosEnc
			for i, want := range []struct {
				rec, moe bool
			}{{true, true}, {false, true}, {true, false}, {true, true}} {
				l := &m.Layers[i]
				ok = ok && c.LayerKind(i).Recurrent() == want.rec && c.MoEAt(i) == want.moe && l.NoFFN == !want.moe
				if want.moe {
					ok = ok && l.UngatedExp && l.Gate.E == nil && l.Up.E != nil && l.Vector(jlm.RoleExpProbsB) != nil &&
						l.Matrix(jlm.RoleShExpGate) == nil && l.Matrix(jlm.RoleShExpUp) != nil && l.Vector(jlm.RoleShRouter) == nil
				}
			}
			return ok
		}},
	}
}

// nemotronhMoEViolations are Nemotron 3's router and shared expert: the V3
// gate's four pieces and the always-on expert dropped. The experts' missing
// gate is held by the dense set's "SiLU for the squared ReLU", which an
// ungated activation runs through every expert too.
func nemotronhMoEViolations() []c6Violation {
	return []c6Violation{
		{"softmax for sigmoid", routerSwap(false)},
		{"no selection bias", noSelectionBias},
		{"the routed weights not renormalised", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = true })
		}},
		{"no routed scale", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.ExpertScale = 1 }) }},
		{"no shared expert", noSharedExpert},
	}
}

// nemotronhViolations are Nemotron-H's own: the grouped gated norm, the merge
// (an MLP's norm becoming its block's FFN norm), the squared ReLU, and -- on
// the host, since the device declines a tiled Mamba-2 by name and no Mamba-2
// is one -- the value heads paired with the wrong key group.
func nemotronhViolations(device bool) []c6Violation {
	vs := []c6Violation{
		{"the gated norm over the whole of Inner", func(m *model.Model) func() {
			// The norm's groups are the stated count that groups B and C too,
			// so the violation is the norm op's alone.
			return restated(m, func(g *graph.Graph) *graph.Graph {
				return editing(opsNamed("ssm.norm"), func(o *graph.Op) { o.Attr.HeadDim = g.Values[o.Out[0]].Width })(g)
			})
		}},
		{"the MLP reading its mixer's norm", func(m *model.Model) func() {
			return eachLayer(m, func(l *model.Layer) {
				if l.Vector(jlm.RoleFFNNorm) != nil {
					addVec(l, jlm.RoleFFNNorm, l.Vector(jlm.RoleAttnNorm))
				}
			})
		}},
		{"SiLU for the squared ReLU", actSwap(0)},
	}
	if !device {
		vs = append(vs, c6Violation{"value heads paired with key groups tiled", func(m *model.Model) func() {
			// The scan reads B and C at each head's place in a cycle of the
			// groups, where Mamba-2 reads them at its group.
			return restated(m, editing(opsOfKind(graph.OpStateScan), func(o *graph.Op) {
				o.Attr.Words = slices.Clone(o.Attr.Words)
				for k := range min(2, len(o.Attr.Words)) {
					if w := graph.ScanAxis(o.Attr.Words[k]); w&graph.ScanGroup != 0 {
						o.Attr.Words[k] = uint64(w&^graph.ScanGroup | graph.ScanCycle)
					}
				}
			}))
		}})
	}
	return vs
}

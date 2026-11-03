package model

import "github.com/jitllm/jitllm/engine/nn"

// Nemotron-H (jlm.ArchNemotronH) on the Mamba-2 harness (ssmfamily_test.go):
// its fixture, built by NemotronHForCausalLM (scripts/ssmgold.py) with one
// mixer a source layer merged at conversion, joins every gate there and the
// five principles' through principleFixtures.
func nemotronhFixtures() []struct {
	name string
	sel  func(m *Model) bool
} {
	return []struct {
		name string
		sel  func(m *Model) bool
	}{
		{"synth-nemotronh", func(m *Model) bool {
			// "M-M*-MM-": a mixer with its MLP, a mixer alone, the attention with
			// its MLP, a mixer alone, a mixer with its MLP -- five blocks.
			c := m.Cfg
			ok := c.Arch == "nemotron_h" && c.SSD() && c.NLayer == 5 && c.SSM.Groups == 2 &&
				c.SSMNormGroups == 2 && c.NoPosEnc && c.Act == nn.ActReLU2 && c.NKVHead == 2 && !c.TiedEmbd
			for i, want := range []struct {
				rec, noFFN bool
			}{{true, false}, {true, true}, {false, false}, {true, true}, {true, false}} {
				l := &m.layers[i]
				ok = ok && c.LayerKind(i).Recurrent() == want.rec && l.noFFN == want.noFFN &&
					(l.noFFN || (l.gate.e == nil && l.up.e != nil && l.ffnNorm != nil))
			}
			return ok
		}},
		{"synth-nemotronhmoe", func(m *Model) bool {
			// "ME*EMME": a mixer with its mixture, the attention with its mixture,
			// a mixer alone, a mixer with its mixture -- four blocks, the mixtures
			// ungated (up and down, no gate bank) beside an ungated shared expert.
			c := m.Cfg
			ok := c.Arch == "nemotron_h" && c.SSD() && c.NLayer == 4 && c.NExpert == 8 && c.NExpertUsed == 2 &&
				c.ExpertSigmoid && !c.NoExpertNorm && c.ExpertScale == 2.5 && c.NExpertGroup == 0 &&
				c.NFFNExp == 64 && c.NFFNShExp == 96 && c.Act == nn.ActReLU2 && c.NoPosEnc
			for i, want := range []struct {
				rec, moe bool
			}{{true, true}, {false, true}, {true, false}, {true, true}} {
				l := &m.layers[i]
				ok = ok && c.LayerKind(i).Recurrent() == want.rec && c.MoEAt(i) == want.moe && l.noFFN == !want.moe
				if want.moe {
					ok = ok && l.ungatedExp && l.gate.e == nil && l.up.e != nil && l.expProbsB != nil &&
						l.shGate.e == nil && l.shUp.e != nil && l.shRouter == nil
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
		{"the routed weights not renormalised", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoExpertNorm = true })
		}},
		{"no routed scale", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertScale = 1 }) }},
		{"no shared expert", noSharedExpert},
	}
}

// nemotronhViolations are Nemotron-H's own: the grouped gated norm, the merge
// (an MLP's norm becoming its block's FFN norm), the squared ReLU, and -- on
// the host, since the device declines a tiled Mamba-2 by name and no Mamba-2
// is one -- the value heads paired with the wrong key group.
func nemotronhViolations(device bool) []c6Violation {
	vs := []c6Violation{
		{"the gated norm over the whole of Inner", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.SSMNormGroups = 1 })
		}},
		{"the MLP reading its mixer's norm", func(m *Model) func() {
			return eachLayer(m, func(l *layer) {
				if l.ffnNorm != nil {
					l.ffnNorm = l.attnNorm
				}
			})
		}},
		{"SiLU for the squared ReLU", actSwap(nn.ActSiLU)},
	}
	if !device {
		vs = append(vs, c6Violation{"value heads paired with key groups tiled", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.DeltaKeyTiled = true })
		}})
	}
	return vs
}

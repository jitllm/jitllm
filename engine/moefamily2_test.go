package engine

import (
	"math"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// The modern text families (bailingmoe2 and its siblings) join the
// mixture-family harness: scripts/moegold.py builds each fixture with the
// family's own reference class -- inclusionAI's remote code for Ling 2.0 --
// with every parameter and the router's selection bias randomised, and writes
// it to GGUF with convert_hf_to_gguf.py. They are appended here rather than in
// moefamily_test.go, so the gates are TestMoEFamiliesMatchTransformers,
// TestMoEFamilyFeaturesAreLoadBearing, TestMoEFamiliesOnEveryDevice and
// TestMoEFamiliesPageWithTheSameAnswer under each fixture's name.
func moe2Fixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{
		{"synth-bailingmoe2", func(m *model.Model) bool {
			c := m.Cfg
			dense, moe := &m.Layers[0], &m.Layers[1]
			return c.RopeNeox && 2*c.NRot == c.HeadDim && c.QKNorm &&
				!c.QKNormWide && c.ExpertSigmoid && !c.NoExpertNorm && c.ExpertScale == 2.5 &&
				c.NExpert == 16 && c.NExpertUsed == 4 && c.NExpertGroup == 4 && c.NExpertGroupUsed == 2 &&
				c.NFFNShExp == int(m.Container.Config().NFFNExp) && c.NDenseLead == 1 && c.NLayer == 3 && c.NPredict == 0 &&
				c.NKVHead < c.NHead && !c.TiedEmbd &&
				dense.Matrix(jlm.RoleRouter) == nil && dense.Gate.E != nil &&
				moe.Matrix(jlm.RoleRouter) != nil && len(moe.Vector(jlm.RoleExpProbsB)) == c.NExpert && moe.Matrix(jlm.RoleShExpGate) != nil &&
				moe.Vector(jlm.RoleShRouter) == nil && moe.Vector(jlm.RoleAttnQNorm) != nil && moe.Vector(jlm.RoleAttnKNorm) != nil &&
				moe.Vector(jlm.RoleAttnQBias) == nil && moe.Vector(jlm.RoleAttnKBias) == nil && moe.Vector(jlm.RoleAttnVBias) == nil && moe.Vector(jlm.RoleAttnOutBias) == nil &&
				moe.Wq.E != nil && moe.Wk.E != nil && moe.Wv.E != nil
		}},
		{"synth-dots1", func(m *model.Model) bool {
			c := m.Cfg
			dense, moe := &m.Layers[0], &m.Layers[1]
			return c.RopeNeox && c.NRot == c.HeadDim && c.QKNorm && !c.QKNormWide &&
				c.ExpertSigmoid && !c.NoExpertNorm && c.ExpertScale == 2.5 && c.NExpert == 8 &&
				c.NExpertUsed == 2 && c.NExpertGroup <= 1 && c.NFFNShExp == 2*int(m.Container.Config().NFFNExp) &&
				c.NDenseLead == 1 && c.NLayer == 3 && c.NPredict == 0 && c.NKVHead == c.NHead && !c.TiedEmbd &&
				windowOf(m) == 0 && dense.Matrix(jlm.RoleRouter) == nil && dense.Gate.E != nil &&
				moe.Matrix(jlm.RoleRouter) != nil && len(moe.Vector(jlm.RoleExpProbsB)) == c.NExpert && moe.Matrix(jlm.RoleShExpGate) != nil &&
				moe.Vector(jlm.RoleShRouter) == nil && moe.Vector(jlm.RoleAttnQNorm) != nil && moe.Vector(jlm.RoleAttnKNorm) != nil &&
				moe.Vector(jlm.RoleAttnQBias) == nil && moe.Vector(jlm.RoleAttnKBias) == nil && moe.Vector(jlm.RoleAttnVBias) == nil && moe.Vector(jlm.RoleAttnOutBias) == nil
		}},
		{"synth-phimoe", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			ok := c.LayerNorm && !c.Parallel && c.RopeNeox && c.NRot == c.HeadDim &&
				len(m.Container.Config().RouterWeight.Terms) != 0 && c.NoExpertNorm && !c.ExpertSigmoid && c.NExpertGroup <= 1 &&
				c.NExpert == 8 && c.NExpertUsed == 2 && c.NFFNShExp == 0 && c.NDenseLead == 0 &&
				c.NKVHead < c.NHead && !c.TiedEmbd && windowOf(m) == 0 &&
				math.Abs(c.AttnFactor-math.Sqrt(1+math.Log(8)/math.Log(512))) < 1e-6 &&
				len(m.Rope.Freqs) == c.NRot/2 && m.Vector(jlm.RoleOutputBias) != nil && m.Vector(jlm.RoleOutputNormBias) != nil
			for i := range m.Layers {
				l = &m.Layers[i]
				ok = ok && l.Matrix(jlm.RoleRouter) != nil && l.Vector(jlm.RoleExpProbsB) == nil && l.Matrix(jlm.RoleShExpGate) == nil &&
					l.Vector(jlm.RoleAttnNormBias) != nil && l.Vector(jlm.RoleFFNNormBias) != nil && l.Vector(jlm.RoleAttnQBias) != nil && l.Vector(jlm.RoleAttnKBias) != nil && l.Vector(jlm.RoleAttnVBias) != nil && l.Vector(jlm.RoleAttnOutBias) != nil
			}
			return ok
		}},
	}
}

// ungrouped runs a grouped router as a plain top-k over every expert.
func ungrouped(m *model.Model) func() {
	return cfgMut(m, func(c *model.Config) { c.NExpertGroup, c.NExpertGroupUsed = 0, 0 })
}

// hostFullRotary turns every dimension of the head where the model turns
// NRot: the host's rotary table is built at Open, so the config alone does
// not reach it.
func hostFullRotary(m *model.Model) func() {
	was := m.Rope.NRot
	m.Rope.NRot = m.Cfg.HeadDim
	undo := cfgMut(m, func(c *model.Config) { c.NRot = c.HeadDim })
	return func() { undo(); m.Rope.NRot = was }
}

func moe2Violations() map[string][]c6Violation {
	return map[string][]c6Violation{
		"synth-bailingmoe2": {
			{"softmax for sigmoid", routerSwap(false)},
			{"no selection bias", noSelectionBias},
			{"no group limit", ungrouped},
			{"the routed weights not renormalised", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = true })
			}},
			{"no routed scale", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.ExpertScale = 1 }) }},
			{"no shared expert", noSharedExpert},
			{"no q/k norm", noQKNorm},
			{"NORM rotary for NEOX", hostNormRotary},
			{"rotary over the whole head", hostFullRotary},
		},
		"synth-dots1": {
			{"softmax for sigmoid", routerSwap(false)},
			{"no selection bias", noSelectionBias},
			{"the routed weights not renormalised", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = true })
			}},
			{"no routed scale", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.ExpertScale = 1 }) }},
			{"no shared expert", noSharedExpert},
			{"no q/k norm", noQKNorm},
			{"NORM rotary for NEOX", hostNormRotary},
		},
		"synth-phimoe": {
			{"the GGUF oracle's router (a renormalised softmax top-2)", llamaCppPhiMoERouter},
			{"no sparsemixer mask (a softmax over every expert)", unmaskedSoftmax},
			// The threshold moved, so the fixture's logits are known to sit near
			// it: a router whose every other expert fell outside would weight each
			// choice exactly one, and neither of these could move it.
			{"a threshold of zero (each weight one)", mixerEps(1e-12)},
			{"a threshold ten times wider", mixerEps(0.1)},
			{"RMSNorm for LayerNorm", rmsForLayerNorm},
			{"no norm biases", func(m *model.Model) func() {
				undoL := eachLayer(m, func(l *model.Layer) { addVec(l, jlm.RoleAttnNormBias, nil); addVec(l, jlm.RoleFFNNormBias, nil) })
				undoR := dropStored(m, jlm.RoleAttnNormBias, jlm.RoleFFNNormBias)
				return func() { undoR(); undoL() }
			}},
			{"no q/k/v/o biases", func(m *model.Model) func() {
				return dropVecs(m, jlm.RoleAttnQBias, jlm.RoleAttnKBias, jlm.RoleAttnVBias, jlm.RoleAttnOutBias)
			}},
			{"no head bias", func(m *model.Model) func() {
				undoV := setDenseVec(m, jlm.RoleOutputBias, func([]float32) []float32 { return nil })
				undoR := dropStoredDense(m, jlm.RoleOutputBias)
				return func() { undoR(); undoV() }
			}},
			{"no LongRoPE short factors", func(m *model.Model) func() {
				undo := relowered(m)
				was := m.Rope.Freqs
				m.Rope.Freqs = nil
				return func() { m.Rope.Freqs = was; undo() }
			}},
			{"no LongRoPE magnitude", func(m *model.Model) func() {
				undo := relowered(m)
				was := m.Rope.Scale
				m.Rope.Scale = 1
				return func() { m.Rope.Scale = was; undo() }
			}},
			{"NORM rotary for NEOX", hostNormRotary},
		},
	}
}

func moe2DevViolations() map[string][]c6Violation {
	return map[string][]c6Violation{
		"synth-bailingmoe2": {
			// Ungrouped too: the device's softmax route forms no group scores, and
			// declines a grouped one by name rather than running it.
			{"softmax for sigmoid", func(m *model.Model) func() {
				undoS := routerSwap(false)(m)
				undoG := ungrouped(m)
				return func() { undoG(); undoS() }
			}},
			{"no selection bias", noSelectionBias},
			{"no group limit", ungrouped},
			{"the routed weights not renormalised", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = true })
			}},
			{"no routed scale", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.ExpertScale = 1 }) }},
			{"no shared expert", noSharedExpert},
			{"no q/k norm", noQKNorm},
			{"NORM rotary for NEOX", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.RopeNeox = false }) }},
			{"rotary over the whole head", hostFullRotary},
		},
		"synth-dots1": {
			{"softmax for sigmoid", routerSwap(false)},
			{"no selection bias", noSelectionBias},
			{"the routed weights not renormalised", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = true })
			}},
			{"no routed scale", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.ExpertScale = 1 }) }},
			{"no shared expert", noSharedExpert},
			{"no q/k norm", noQKNorm},
			{"NORM rotary for NEOX", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.RopeNeox = false }) }},
		},
		"synth-phimoe": {
			{"the GGUF oracle's router (a renormalised softmax top-2)", llamaCppPhiMoERouter},
			{"no sparsemixer mask (a softmax over every expert)", unmaskedSoftmax},
			{"a threshold of zero (each weight one)", mixerEps(1e-12)},
			{"RMSNorm for LayerNorm", rmsForLayerNorm},
			{"no q/k/v/o biases", func(m *model.Model) func() {
				return dropVecs(m, jlm.RoleAttnQBias, jlm.RoleAttnKBias, jlm.RoleAttnVBias, jlm.RoleAttnOutBias)
			}},
			{"NORM rotary for NEOX", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.RopeNeox = false }) }},
		},
	}
}

// llamaCppPhiMoERouter routes Phi-3.5-MoE as the GGUF oracle does: a softmax
// top-2, renormalised.
func llamaCppPhiMoERouter(m *model.Model) func() {
	undo := cfgMut(m, func(c *model.Config) { c.NoExpertNorm = false })
	undoR := statedRouter(m, unstatedScore(&jlm.Config{NExpert: 2}), jlm.Expr{})
	return func() { undoR(); undo() }
}

// unstatedScore is the router score the converter states for c, a config
// no library file states anything for.
func unstatedScore(c *jlm.Config) jlm.Expr {
	e, err := convert.RouterScore(convert.Unstated(c))
	if err != nil {
		panic(err)
	}
	return e
}

// statedRouter states the router's score and weight to the front end as a
// container of that router states them, the mutation of main's config beside
// it reaching the composed arm.
func statedRouter(m *model.Model, score, weight jlm.Expr) func() {
	c := m.Container.Config()
	ws, ww := c.RouterScore, c.RouterWeight
	c.RouterScore, c.RouterWeight = score, weight
	undo := restate(m, func(map[jlm.Role]graph.Shape) {})
	return func() {
		c.RouterScore, c.RouterWeight = ws, ww
		undo()
	}
}

// unmaskedSoftmax weights each of the two by a softmax over every expert, not
// renormalised: sparsemixer with no mask, and both of its experts' shared
// denominator.
func unmaskedSoftmax(m *model.Model) func() {
	return statedRouter(m, unstatedScore(&jlm.Config{NExpert: 2}), jlm.Expr{})
}

// mixerEps runs sparsemixer at another jitter_eps.
func mixerEps(eps float32) func(m *model.Model) func() {
	return func(m *model.Model) func() {
		return statedRouter(m, m.Container.Config().RouterScore, convert.SparseMixerWeight(eps))
	}
}

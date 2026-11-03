package model

import (
	"math"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// The modern text families (jlm.ArchBailingMoE2 and its siblings) join the
// mixture-family harness: scripts/moegold.py builds each fixture with the
// family's own reference class -- inclusionAI's remote code for Ling 2.0 --
// with every parameter and the router's selection bias randomised, and writes
// it to GGUF with llama.cpp's converter. They are appended here rather than in
// moefamily_test.go, so the gates are TestMoEFamiliesMatchTransformers,
// TestMoEFamilyFeaturesAreLoadBearing, TestMoEFamiliesOnEveryDevice and
// TestMoEFamiliesPageWithTheSameAnswer under each fixture's name.
func moe2Fixtures() []struct {
	name string
	sel  func(m *Model) bool
} {
	return []struct {
		name string
		sel  func(m *Model) bool
	}{
		{"synth-bailingmoe2", func(m *Model) bool {
			c := m.Cfg
			dense, moe := &m.layers[0], &m.layers[1]
			return c.Arch == "bailingmoe2" && c.RopeNeox && 2*c.NRot == c.HeadDim && c.QKNorm &&
				!c.QKNormWide && c.ExpertSigmoid && !c.NoExpertNorm && c.ExpertScale == 2.5 &&
				c.NExpert == 16 && c.NExpertUsed == 4 && c.NExpertGroup == 4 && c.NExpertGroupUsed == 2 &&
				c.NFFNShExp == c.NFFNExp && c.NDenseLead == 1 && c.NLayer == 3 && c.NMTP == 0 &&
				c.NKVHead < c.NHead && !c.TiedEmbd &&
				dense.router.e == nil && dense.gate.e != nil &&
				moe.router.e != nil && len(moe.expProbsB) == c.NExpert && moe.shGate.e != nil &&
				moe.shRouter == nil && moe.qNorm != nil && moe.kNorm != nil &&
				moe.bq == nil && moe.bk == nil && moe.bv == nil && moe.bo == nil &&
				moe.wq.e != nil && moe.wk.e != nil && moe.wv.e != nil
		}},
		{"synth-dots1", func(m *Model) bool {
			c := m.Cfg
			dense, moe := &m.layers[0], &m.layers[1]
			return c.Arch == "dots1" && c.RopeNeox && c.NRot == c.HeadDim && c.QKNorm && !c.QKNormWide &&
				c.ExpertSigmoid && !c.NoExpertNorm && c.ExpertScale == 2.5 && c.NExpert == 8 &&
				c.NExpertUsed == 2 && c.NExpertGroup <= 1 && c.NFFNShExp == 2*c.NFFNExp &&
				c.NDenseLead == 1 && c.NLayer == 3 && c.NMTP == 0 && c.NKVHead == c.NHead && !c.TiedEmbd &&
				c.SWAWindow == 0 && dense.router.e == nil && dense.gate.e != nil &&
				moe.router.e != nil && len(moe.expProbsB) == c.NExpert && moe.shGate.e != nil &&
				moe.shRouter == nil && moe.qNorm != nil && moe.kNorm != nil &&
				moe.bq == nil && moe.bk == nil && moe.bv == nil && moe.bo == nil
		}},
		{"synth-phimoe", func(m *Model) bool {
			c, l := m.Cfg, &m.layers[0]
			ok := c.Arch == "phimoe" && c.LayerNorm && !c.Parallel && c.RopeNeox && c.NRot == c.HeadDim &&
				c.SparseMixer == 0.01 && c.NoExpertNorm && !c.ExpertSigmoid && c.NExpertGroup <= 1 &&
				c.NExpert == 8 && c.NExpertUsed == 2 && c.NFFNShExp == 0 && c.NDenseLead == 0 &&
				c.NKVHead < c.NHead && !c.TiedEmbd && c.SWAWindow == 0 &&
				math.Abs(c.AttnFactor-math.Sqrt(1+math.Log(8)/math.Log(512))) < 1e-6 &&
				len(m.rope.Freqs) == c.NRot/2 && m.outB != nil && m.outNormB != nil
			for i := range m.layers {
				l = &m.layers[i]
				ok = ok && l.router.e != nil && l.expProbsB == nil && l.shGate.e == nil &&
					l.attnNormB != nil && l.ffnNormB != nil && l.bq != nil && l.bk != nil && l.bv != nil && l.bo != nil
			}
			return ok
		}},
	}
}

// ungrouped runs a grouped router as a plain top-k over every expert.
func ungrouped(m *Model) func() {
	return cfgMut(m, func(c *Config) { c.NExpertGroup, c.NExpertGroupUsed = 0, 0 })
}

// hostFullRotary turns every dimension of the head where the model turns
// NRot: the host's rotary table is built at Open, so the config alone does
// not reach it.
func hostFullRotary(m *Model) func() {
	was := m.rope.NRot
	m.rope.NRot = m.Cfg.HeadDim
	undo := cfgMut(m, func(c *Config) { c.NRot = c.HeadDim })
	return func() { undo(); m.rope.NRot = was }
}

func moe2Violations() map[string][]c6Violation {
	return map[string][]c6Violation{
		"synth-bailingmoe2": {
			{"softmax for sigmoid", routerSwap(false)},
			{"no selection bias", noSelectionBias},
			{"no group limit", ungrouped},
			{"the routed weights not renormalised", func(m *Model) func() {
				return cfgMut(m, func(c *Config) { c.NoExpertNorm = true })
			}},
			{"no routed scale", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertScale = 1 }) }},
			{"no shared expert", noSharedExpert},
			{"no q/k norm", noQKNorm},
			{"NORM rotary for NEOX", hostNormRotary},
			{"rotary over the whole head", hostFullRotary},
		},
		"synth-dots1": {
			{"softmax for sigmoid", routerSwap(false)},
			{"no selection bias", noSelectionBias},
			{"the routed weights not renormalised", func(m *Model) func() {
				return cfgMut(m, func(c *Config) { c.NoExpertNorm = true })
			}},
			{"no routed scale", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertScale = 1 }) }},
			{"no shared expert", noSharedExpert},
			{"no q/k norm", noQKNorm},
			{"NORM rotary for NEOX", hostNormRotary},
		},
		"synth-phimoe": {
			{"llama.cpp's router (a renormalised softmax top-2)", llamaCppPhiMoERouter},
			{"no sparsemixer mask (a softmax over every expert)", unmaskedSoftmax},
			// The threshold moved, so the fixture's logits are known to sit near
			// it: a router whose every other expert fell outside would weight each
			// choice exactly one, and neither of these could move it.
			{"a threshold of zero (each weight one)", mixerEps(1e-12)},
			{"a threshold ten times wider", mixerEps(0.1)},
			{"RMSNorm for LayerNorm", rmsForLayerNorm},
			{"no norm biases", func(m *Model) func() {
				undoL := eachLayer(m, func(l *layer) { l.attnNormB, l.ffnNormB = nil, nil })
				undoR := dropStored(m, jlm.RoleAttnNormBias, jlm.RoleFFNNormBias)
				return func() { undoR(); undoL() }
			}},
			{"no q/k/v/o biases", func(m *Model) func() {
				undoL := eachLayer(m, func(l *layer) { l.bq, l.bk, l.bv, l.bo = nil, nil, nil, nil })
				undoR := dropStored(m, jlm.RoleAttnQBias, jlm.RoleAttnKBias, jlm.RoleAttnVBias, jlm.RoleAttnOutBias)
				return func() { undoR(); undoL() }
			}},
			{"no head bias", func(m *Model) func() {
				was := m.outB
				m.outB = nil
				undoR := dropStoredDense(m, jlm.RoleOutputBias)
				return func() { undoR(); m.outB = was }
			}},
			{"no LongRoPE short factors", func(m *Model) func() {
				was := m.rope.Freqs
				m.rope.Freqs = nil
				return func() { m.rope.Freqs = was }
			}},
			{"no LongRoPE magnitude", func(m *Model) func() {
				was := m.rope.Scale
				m.rope.Scale = 1
				return func() { m.rope.Scale = was }
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
			{"softmax for sigmoid", func(m *Model) func() {
				undoS := routerSwap(false)(m)
				undoG := ungrouped(m)
				return func() { undoG(); undoS() }
			}},
			{"no selection bias", noSelectionBias},
			{"no group limit", ungrouped},
			{"the routed weights not renormalised", func(m *Model) func() {
				return cfgMut(m, func(c *Config) { c.NoExpertNorm = true })
			}},
			{"no routed scale", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertScale = 1 }) }},
			{"no shared expert", noSharedExpert},
			{"no q/k norm", noQKNorm},
			{"NORM rotary for NEOX", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = false }) }},
			{"rotary over the whole head", hostFullRotary},
		},
		"synth-dots1": {
			{"softmax for sigmoid", routerSwap(false)},
			{"no selection bias", noSelectionBias},
			{"the routed weights not renormalised", func(m *Model) func() {
				return cfgMut(m, func(c *Config) { c.NoExpertNorm = true })
			}},
			{"no routed scale", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertScale = 1 }) }},
			{"no shared expert", noSharedExpert},
			{"no q/k norm", noQKNorm},
			{"NORM rotary for NEOX", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = false }) }},
		},
		"synth-phimoe": {
			{"llama.cpp's router (a renormalised softmax top-2)", llamaCppPhiMoERouter},
			{"no sparsemixer mask (a softmax over every expert)", unmaskedSoftmax},
			{"a threshold of zero (each weight one)", mixerEps(1e-12)},
			{"RMSNorm for LayerNorm", rmsForLayerNorm},
			{"no q/k/v/o biases", func(m *Model) func() {
				undoL := eachLayer(m, func(l *layer) { l.bq, l.bk, l.bv, l.bo = nil, nil, nil, nil })
				undoR := dropStored(m, jlm.RoleAttnQBias, jlm.RoleAttnKBias, jlm.RoleAttnVBias, jlm.RoleAttnOutBias)
				return func() { undoR(); undoL() }
			}},
			{"NORM rotary for NEOX", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = false }) }},
		},
	}
}

// llamaCppPhiMoERouter routes Phi-3.5-MoE as llama.cpp does: a softmax top-2,
// renormalised.
func llamaCppPhiMoERouter(m *Model) func() {
	undo := cfgMut(m, func(c *Config) { c.SparseMixer, c.NoExpertNorm = 0, false })
	undoR := statedRouter(m, convert.RouterScore(&jlm.Config{NExpert: 2}), jlm.Expr{})
	return func() { undoR(); undo() }
}

// statedRouter states the router's score and weight to the front end as a
// container of that router states them, the mutation of main's config beside
// it reaching the composed arm.
func statedRouter(m *Model, score, weight jlm.Expr) func() {
	c := m.container.Config()
	ws, ww := c.RouterScore, c.RouterWeight
	c.RouterScore, c.RouterWeight = score, weight
	undo := m.restate(func(map[jlm.Role]graph.Shape) {})
	return func() {
		c.RouterScore, c.RouterWeight = ws, ww
		undo()
	}
}

// unmaskedSoftmax weights each of the two by a softmax over every expert, not
// renormalised: sparsemixer with no mask, and both of its experts' shared
// denominator.
func unmaskedSoftmax(m *Model) func() {
	undo := cfgMut(m, func(c *Config) { c.SparseMixer = 0 })
	undoR := statedRouter(m, convert.RouterScore(&jlm.Config{NExpert: 2}), jlm.Expr{})
	return func() { undoR(); undo() }
}

// mixerEps runs sparsemixer at another jitter_eps.
func mixerEps(eps float32) func(m *Model) func() {
	return func(m *Model) func() {
		undo := cfgMut(m, func(c *Config) { c.SparseMixer = eps })
		undoR := statedRouter(m, m.container.Config().RouterScore, convert.SparseMixerWeight(eps))
		return func() { undoR(); undo() }
	}
}

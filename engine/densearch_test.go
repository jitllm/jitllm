package engine

import (
	"math"
	"slices"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// The dense llama-family architectures (jlm.ArchSmolLM3 and its siblings) are
// gated by the classic-transformer (C6) harness: transformers' own class,
// written to GGUF by llama.cpp's convert_hf_to_gguf.py (scripts/c6gold.py), on
// decode, prefill at every length and the batch, against the host and against
// every device. They join its lists here rather than in c6_test.go, so the
// gates are TestC6MatchesTransformers, TestC6FeaturesAreLoadBearing and
// TestC6OnEveryDevice under each fixture's name, and the relocation and
// paging gates through principleFixtures.
func denseFixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{
		{"synth-apertus", func(m *model.Model) bool {
			c, l, l1 := m.Cfg, &m.Layers[0], &m.Layers[1]
			return c.Arch == "apertus" && !c.LayerNorm && c.RopeNeox && c.QKNorm && !c.QKNormWide &&
				len(l.QNorm) == c.HeadDim && m.Container.Config().Arch == jlm.ArchApertus && l.Gate.E == nil && l.Vector(jlm.RoleFFNUpBias) == nil &&
				len(l.Vector(jlm.RoleXIELU)) == 4 && len(l1.Vector(jlm.RoleXIELU)) == 4 && l.Vector(jlm.RoleXIELU)[0] != l1.Vector(jlm.RoleXIELU)[0] &&
				l.Vector(jlm.RoleXIELU)[2] != l1.Vector(jlm.RoleXIELU)[2] && l.Vector(jlm.RoleXIELU)[3] < -0.01 && len(m.Rope.Freqs) == c.NRot/2 && c.NKVHead < c.NHead && !c.TiedEmbd
		}},
		{"synth-smollm3", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "smollm3" && !c.LayerNorm && !c.RopeNeox && c.NoPEGlobal &&
				c.SWAWindow == 0 && c.SWAPeriod == 4 && c.RopeAt(0) && c.RopeAt(2) && !c.RopeAt(3) &&
				!c.SWA(0) && c.NKVHead < c.NHead && l.Gate.E != nil && c.TiedEmbd
		}},
		{"synth-arcee", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "arcee" && !c.LayerNorm && m.Container.Config().Flags.Has(jlm.FlagReLU2) && l.Gate.E == nil &&
				l.FfnNorm != nil && l.Vector(jlm.RoleFFNUpBias) == nil && c.NKVHead < c.NHead && !c.TiedEmbd
		}},
		{"synth-seedoss", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "seed_oss" && c.RopeNeox && c.HeadDim*c.NHead != c.NEmbd &&
				l.Vector(jlm.RoleAttnQBias) != nil && l.Vector(jlm.RoleAttnKBias) != nil && l.Vector(jlm.RoleAttnVBias) != nil && l.Vector(jlm.RoleAttnOutBias) == nil && l.FfnNorm != nil &&
				l.Vector(jlm.RolePostAttnNorm) == nil && c.NKVHead < c.NHead && !c.TiedEmbd
		}},
		{"synth-olmo2", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "olmo2" && c.RopeNeox && c.QKNorm && c.QKNormWide &&
				len(l.QNorm) == c.NHead*c.HeadDim && l.AttnNorm == nil && l.FfnNorm == nil &&
				l.Vector(jlm.RolePostAttnNorm) != nil && l.Vector(jlm.RolePostFFNNorm) != nil && m.HasNoPreNorm() && !c.TiedEmbd
		}},
		{"synth-exaone4", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "exaone4" && c.RopeNeox && c.QKNorm && !c.QKNormWide &&
				len(l.QNorm) == c.HeadDim && l.AttnNorm == nil && l.FfnNorm == nil &&
				l.Vector(jlm.RolePostAttnNorm) != nil && l.Vector(jlm.RolePostFFNNorm) != nil && c.NoPEGlobal &&
				c.SWAWindow == 4 && c.SWAPeriod == 4 && c.SWA(0) && !c.SWA(3) && c.RopeAt(2) && !c.RopeAt(3) &&
				c.NKVHead < c.NHead
		}},
		{"synth-mistral3", func(m *model.Model) bool {
			c := m.Cfg
			return c.Arch == "mistral3" && !c.RopeNeox && !c.NoPEGlobal && c.AttnTempScale == 0.5 &&
				c.AttnTempFloor == 4 && c.AttnTempOffs == 0 && c.YarnFactor == 16 &&
				c.AttnTemp(0, 4) != 1 && c.AttnTemp(2, 8) != 1 && c.AttnTemp(1, 3) == 1 &&
				m.Rope.Scale > 1.1 && m.Rope.Scale < 1.13
		}},
		{"synth-olmo2-gqa", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "olmo2" && c.QKNormWide && c.NKVHead < c.NHead &&
				len(l.QNorm) == c.NHead*c.HeadDim && len(l.KNorm) == c.NKVHead*c.HeadDim && m.HasNoPreNorm()
		}},
		{"synth-olmo3", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "olmo3" && c.QKNormWide && c.NKVHead < c.NHead && len(l.KNorm) == c.NKVHead*c.HeadDim &&
				c.SWAWindow == 4 && c.SWAPeriod == 4 && c.SWA(0) && !c.SWA(3) && c.YarnFactor == 8 &&
				m.Rope.Scale > 1.2 && m.Rope.Scale < 1.22 && m.Rope.Freqs != nil &&
				m.RopeSWA != nil && m.RopeSWA.Scale == 1 && m.RopeSWA.Freqs == nil && m.HasNoPreNorm()
		}},
	}
}

// neoxOff and noTemp are the violations several fixtures share.
func neoxOff(m *model.Model) func() { return hostNormRotary(m) }

func noTemp(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.AttnTempScale = 0 }) }

// unitRopeScale takes mistral3's YaRN magnitude off cos and sin.
func unitRopeScale(m *model.Model) func() {
	was := m.Rope.Scale
	m.Rope.Scale = 1
	return func() { m.Rope.Scale = was }
}

// kFromQ gives each block's k norm the first n_kv*head_dim of q's weight: the
// two whole-projection norms read as one, which is what an engine that sized
// k's norm from q's would compute.
func kFromQ(m *model.Model) func() {
	return eachLayer(m, func(l *model.Layer) { l.KNorm = l.QNorm[:len(l.KNorm)] })
}

// localYarn gives OLMo 3's sliding layers the global rotary -- YaRN's
// frequencies and its magnitude -- and localScaled its magnitude alone.
func localYarn(m *model.Model) func() {
	was := m.RopeSWA
	r := m.Rope
	r.Base = was.Base
	m.RopeSWA = &r
	return func() { m.RopeSWA = was }
}

// windowBy is the model with its sliding window d keys wider: a window off
// by one position, which the device arm must see past the window.
func windowBy(d int) func(m *model.Model) func() {
	return func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.SWAWindow += d }) }
}

func localScaled(m *model.Model) func() {
	was := m.RopeSWA.Scale
	m.RopeSWA.Scale = m.Rope.Scale
	return func() { m.RopeSWA.Scale = was }
}

func denseViolations() map[string][]c6Violation {
	return map[string][]c6Violation{
		"synth-olmo2-gqa": {
			{"a per-head q/k norm", perHeadQK},
			{"k's norm read from q's weight", kFromQ},
		},
		"synth-olmo3": {
			{"YaRN on the sliding layers", localYarn},
			{"YaRN's magnitude on the sliding layers", localScaled},
			{"no YaRN magnitude on the global layers", unitRopeScale},
			{"no sliding window", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.SWAWindow = 0 }) }},
			{"k's norm read from q's weight", kFromQ},
		},
		"synth-smollm3": {
			{"rotary on the NoPE layer", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.NoPEGlobal = false }) }},
			{"NoPE every second layer", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.SWAPeriod = 2 }) }},
		},
		"synth-arcee": {
			{"SiLU for relu2", actSwap(0)},
			{"GELU for relu2", actSwap(jlm.FlagGELU)},
		},
		"synth-seedoss": {
			{"no q/k/v biases", noQKVBias},
			{"NORM rotary for NEOX", neoxOff},
			{"the attention's norm before the FFN", func(m *model.Model) func() {
				return eachLayer(m, func(l *model.Layer) { l.FfnNorm = l.AttnNorm })
			}},
		},
		"synth-olmo2": {
			{"a per-head q/k norm", perHeadQK},
			{"no post-attention norm", noPostAttnNorm},
			{"no post-FFN norm", noPostFFNNorm},
			{"the post-norms as pre-norms", postAsPre},
			{"NORM rotary for NEOX", neoxOff},
		},
		"synth-exaone4": {
			{"rotary on the global layer", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.NoPEGlobal = false }) }},
			{"no sliding window", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.SWAWindow = 0 }) }},
			{"no q/k norm", noQKNorm},
			{"no post-FFN norm", noPostFFNNorm},
			{"the post-norms as pre-norms", postAsPre},
		},
		"synth-mistral3": {
			{"no attention temperature", noTemp},
			{"no YaRN magnitude on cos and sin", unitRopeScale},
		},
		"synth-apertus": apertusViolations(),
	}
}

// apertusViolations take each piece of Apertus's xIELU and rotary out. They
// run on the host and on every device: the numbers ride with each block's
// weights, and the rotary factors are folded into the plan's table.
func apertusViolations() []c6Violation {
	return []c6Violation{
		{"xIELU's eps ignored", xieluMut(func(p []float32, _ int) { p[3] = float32(math.Inf(1)) })},
		{"xIELU's alphas exchanged", xieluMut(func(p []float32, _ int) { p[0], p[1] = p[1], p[0] })},
		{"xIELU's beta*x dropped", xieluMut(func(p []float32, _ int) { p[2] = 0 })},
		{"xIELU's alpha_p without the softplus", xieluMut(func(p []float32, _ int) {
			p[0] = float32(math.Log(math.Expm1(float64(p[0]))))
		})},
		{"every block running block 0's xIELU", func(m *model.Model) func() {
			p0 := slices.Clone(m.Layers[0].Vector(jlm.RoleXIELU))
			return eachLayer(m, func(l *model.Layer) {
				setVec(l, jlm.RoleXIELU, func([]float32) []float32 { return p0 })
			})
		}},
		{"no llama 3 rotary factors", noRopeFreqs},
		{"NORM rotary for NEOX", neoxOff},
		{"no q/k norm", noQKNorm},
	}
}

// apertusDevViolations are apertusViolations with the rotary pairing flipped
// in the config, which is what the device reads; the host's table is built
// at Open, so the host list flips that instead.
func apertusDevViolations() []c6Violation {
	return append(slices.Clone(apertusViolations()[:len(apertusViolations())-2]),
		c6Violation{"NORM rotary for NEOX", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) { c.RopeNeox = false })
		}},
		c6Violation{"no q/k norm", noQKNorm})
}

// xieluMut runs f on a copy of every block's xIELU numbers.
func xieluMut(f func(p []float32, li int)) func(m *model.Model) func() {
	return func(m *model.Model) func() {
		li := 0
		return eachLayer(m, func(l *model.Layer) {
			setVec(l, jlm.RoleXIELU, func(v []float32) []float32 {
				p := slices.Clone(v)
				f(p, li)
				return p
			})
			li++
		})
	}
}

// perHeadQK runs OLMo 2's whole-projection q/k norm one head at a time, with
// each head reading the first head_dim of the weight -- what an engine that
// took the qwen3 reading of the same tensor names would compute.
func perHeadQK(m *model.Model) func() {
	undoC := cfgMut(m, func(c *model.Config) { c.QKNormWide = false })
	hd := m.Cfg.HeadDim
	undoL := eachLayer(m, func(l *model.Layer) { l.QNorm, l.KNorm = l.QNorm[:hd], l.KNorm[:hd] })
	undoR := restate(m, func(inv map[jlm.Role]graph.Shape) {
		for _, r := range []jlm.Role{jlm.RoleAttnQNorm, jlm.RoleAttnKNorm} {
			if _, ok := inv[r]; ok {
				inv[r] = graph.Shape{K: hd}
			}
		}
	})
	return func() { undoR(); undoL(); undoC() }
}

// postAsPre moves each post-norm to the pre-norm slot of its branch, which is
// llama's block with OLMo 2's weights.
func postAsPre(m *model.Model) func() {
	undoL := eachLayer(m, func(l *model.Layer) {
		l.AttnNorm, l.FfnNorm = l.Vector(jlm.RolePostAttnNorm), l.Vector(jlm.RolePostFFNNorm)
		setVec(l, jlm.RolePostAttnNorm, func([]float32) []float32 { return nil })
		setVec(l, jlm.RolePostFFNNorm, func([]float32) []float32 { return nil })
	})
	undoR := restate(m, func(inv map[jlm.Role]graph.Shape) {
		for _, r := range [][2]jlm.Role{{jlm.RolePostAttnNorm, jlm.RoleAttnNorm}, {jlm.RolePostFFNNorm, jlm.RoleFFNNorm}} {
			if s, ok := inv[r[0]]; ok {
				inv[r[1]] = s
				delete(inv, r[0])
			}
		}
	})
	return func() { undoR(); undoL() }
}

func denseDevViolations() map[string][]c6Violation {
	return map[string][]c6Violation{
		"synth-apertus": apertusDevViolations(),
		"synth-olmo2-gqa": {
			{"a per-head q/k norm", perHeadQK},
			{"k's norm read from q's weight", kFromQ},
		},
		"synth-olmo3": {
			{"YaRN on the sliding layers", localYarn},
			{"YaRN's magnitude on the sliding layers", localScaled},
			{"no sliding window", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.SWAWindow = 0 }) }},
			{"a window one key wider", windowBy(1)},
			{"a window one key narrower", windowBy(-1)},
		},
		"synth-smollm3": {
			{"rotary on the NoPE layer", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.NoPEGlobal = false }) }},
		},
		"synth-arcee": {
			{"SiLU for relu2", actSwap(0)},
		},
		"synth-seedoss": {
			{"no q/k/v biases", noQKVBias},
			{"NORM rotary for NEOX", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.RopeNeox = false }) }},
		},
		"synth-olmo2": {
			{"a per-head q/k norm", perHeadQK},
			{"the post-norms as pre-norms", postAsPre},
			{"NORM rotary for NEOX", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.RopeNeox = false }) }},
		},
		"synth-exaone4": {
			{"rotary on the global layer", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.NoPEGlobal = false }) }},
			{"no sliding window", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.SWAWindow = 0 }) }},
			{"a window one key wider", windowBy(1)},
			{"a window one key narrower", windowBy(-1)},
			{"the post-norms as pre-norms", postAsPre},
		},
		"synth-mistral3": {
			{"no attention temperature", noTemp},
		},
	}
}

package model

import (
	"math"
	"slices"

	"github.com/jitllm/jitllm/engine/nn"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// The dense llama-family architectures (jlm.ArchSmolLM3 and its siblings) are
// gated by the classic-transformer (C6) harness: transformers' own class,
// written to GGUF by llama.cpp's convert_hf_to_gguf.py (scripts/c6gold.py), on
// decode, prefill at every length and the batch, against the host and against
// every device. They join its lists here rather than in c6_test.go, so the
// gates are TestC6MatchesTransformers, TestC6FeaturesAreLoadBearing and
// TestC6OnEveryDevice under each fixture's name, and the relocation and
// paging gates through principleFixtures.
func init() {
	c6Fixtures = append(c6Fixtures, denseFixtures...)
	for k, v := range denseViolations {
		c6Violations[k] = v
	}
	for k, v := range denseDevViolations {
		c6DevViolations[k] = v
	}
	// And the gates of the five principles: relocation, paging and a batch
	// that faults its blocks in, each on every fixture of the group.
	for _, fx := range denseFixtures {
		principleFixtures = append(principleFixtures, fx.name+".gguf")
	}
}

var denseFixtures = []struct {
	name string
	sel  func(m *Model) bool
}{
	{"synth-apertus", func(m *Model) bool {
		c, l, l1 := m.Cfg, &m.layers[0], &m.layers[1]
		return c.Arch == "apertus" && !c.LayerNorm && c.RopeNeox && c.QKNorm && !c.QKNormWide &&
			len(l.qNorm) == c.HeadDim && c.Act == nn.ActXIELU && l.gate.e == nil && l.upB == nil &&
			len(l.xielu) == 4 && len(l1.xielu) == 4 && l.xielu[0] != l1.xielu[0] && l.xielu[2] != l1.xielu[2] &&
			l.xielu[3] < -0.01 && len(m.rope.Freqs) == c.NRot/2 && c.NKVHead < c.NHead && !c.TiedEmbd
	}},
	{"synth-smollm3", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "smollm3" && !c.LayerNorm && !c.RopeNeox && c.NoPEGlobal &&
			c.SWAWindow == 0 && c.SWAPeriod == 4 && c.RopeAt(0) && c.RopeAt(2) && !c.RopeAt(3) &&
			!c.SWA(0) && c.NKVHead < c.NHead && l.gate.e != nil && c.TiedEmbd
	}},
	{"synth-arcee", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "arcee" && !c.LayerNorm && c.Act == nn.ActReLU2 && l.gate.e == nil &&
			l.ffnNorm != nil && l.upB == nil && c.NKVHead < c.NHead && !c.TiedEmbd
	}},
	{"synth-seedoss", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "seed_oss" && c.RopeNeox && c.HeadDim*c.NHead != c.NEmbd &&
			l.bq != nil && l.bk != nil && l.bv != nil && l.bo == nil && l.ffnNorm != nil &&
			l.postAttnNorm == nil && c.NKVHead < c.NHead && !c.TiedEmbd
	}},
	{"synth-olmo2", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "olmo2" && c.RopeNeox && c.QKNorm && c.QKNormWide &&
			len(l.qNorm) == c.NHead*c.HeadDim && l.attnNorm == nil && l.ffnNorm == nil &&
			l.postAttnNorm != nil && l.postFFNNorm != nil && m.hasNoPreNorm() && !c.TiedEmbd
	}},
	{"synth-exaone4", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "exaone4" && c.RopeNeox && c.QKNorm && !c.QKNormWide &&
			len(l.qNorm) == c.HeadDim && l.attnNorm == nil && l.ffnNorm == nil &&
			l.postAttnNorm != nil && l.postFFNNorm != nil && c.NoPEGlobal &&
			c.SWAWindow == 4 && c.SWAPeriod == 4 && c.SWA(0) && !c.SWA(3) && c.RopeAt(2) && !c.RopeAt(3) &&
			c.NKVHead < c.NHead
	}},
	{"synth-mistral3", func(m *Model) bool {
		c := m.Cfg
		return c.Arch == "mistral3" && !c.RopeNeox && !c.NoPEGlobal && c.AttnTempScale == 0.5 &&
			c.AttnTempFloor == 4 && c.AttnTempOffs == 0 && c.YarnFactor == 16 &&
			c.AttnTemp(0, 4) != 1 && c.AttnTemp(2, 8) != 1 && c.AttnTemp(1, 3) == 1 &&
			m.rope.Scale > 1.1 && m.rope.Scale < 1.13
	}},
	{"synth-olmo2-gqa", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "olmo2" && c.QKNormWide && c.NKVHead < c.NHead &&
			len(l.qNorm) == c.NHead*c.HeadDim && len(l.kNorm) == c.NKVHead*c.HeadDim && m.hasNoPreNorm()
	}},
	{"synth-olmo3", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "olmo3" && c.QKNormWide && c.NKVHead < c.NHead && len(l.kNorm) == c.NKVHead*c.HeadDim &&
			c.SWAWindow == 4 && c.SWAPeriod == 4 && c.SWA(0) && !c.SWA(3) && c.YarnFactor == 8 &&
			m.rope.Scale > 1.2 && m.rope.Scale < 1.22 && m.rope.Freqs != nil &&
			m.ropeSWA != nil && m.ropeSWA.Scale == 1 && m.ropeSWA.Freqs == nil && m.hasNoPreNorm()
	}},
}

// neoxOff and noTemp are the violations several fixtures share.
func neoxOff(m *Model) func() { return hostNormRotary(m) }

func noTemp(m *Model) func() { return cfgMut(m, func(c *Config) { c.AttnTempScale = 0 }) }

// unitRopeScale takes mistral3's YaRN magnitude off cos and sin.
func unitRopeScale(m *Model) func() {
	was := m.rope.Scale
	m.rope.Scale = 1
	return func() { m.rope.Scale = was }
}

// kFromQ gives each block's k norm the first n_kv*head_dim of q's weight: the
// two whole-projection norms read as one, which is what an engine that sized
// k's norm from q's would compute.
func kFromQ(m *Model) func() {
	return eachLayer(m, func(l *layer) { l.kNorm = l.qNorm[:len(l.kNorm)] })
}

// localYarn gives OLMo 3's sliding layers the global rotary -- YaRN's
// frequencies and its magnitude -- and localScaled its magnitude alone.
func localYarn(m *Model) func() {
	was := m.ropeSWA
	r := m.rope
	r.Base = was.Base
	m.ropeSWA = &r
	return func() { m.ropeSWA = was }
}

func localScaled(m *Model) func() {
	was := m.ropeSWA.Scale
	m.ropeSWA.Scale = m.rope.Scale
	return func() { m.ropeSWA.Scale = was }
}

var denseViolations = map[string][]c6Violation{
	"synth-olmo2-gqa": {
		{"a per-head q/k norm", perHeadQK},
		{"k's norm read from q's weight", kFromQ},
	},
	"synth-olmo3": {
		{"YaRN on the sliding layers", localYarn},
		{"YaRN's magnitude on the sliding layers", localScaled},
		{"no YaRN magnitude on the global layers", unitRopeScale},
		{"no sliding window", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.SWAWindow = 0 }) }},
		{"k's norm read from q's weight", kFromQ},
	},
	"synth-smollm3": {
		{"rotary on the NoPE layer", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.NoPEGlobal = false }) }},
		{"NoPE every second layer", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.SWAPeriod = 2 }) }},
	},
	"synth-arcee": {
		{"SiLU for relu2", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.Act = nn.ActSiLU }) }},
		{"GELU for relu2", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.Act = nn.ActGELU }) }},
	},
	"synth-seedoss": {
		{"no q/k/v biases", noQKVBias},
		{"NORM rotary for NEOX", neoxOff},
		{"the attention's norm before the FFN", func(m *Model) func() {
			return eachLayer(m, func(l *layer) { l.ffnNorm = l.attnNorm })
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
		{"rotary on the global layer", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.NoPEGlobal = false }) }},
		{"no sliding window", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.SWAWindow = 0 }) }},
		{"no q/k norm", noQKNorm},
		{"no post-FFN norm", noPostFFNNorm},
		{"the post-norms as pre-norms", postAsPre},
	},
	"synth-mistral3": {
		{"no attention temperature", noTemp},
		{"no YaRN magnitude on cos and sin", unitRopeScale},
	},
	"synth-apertus": apertusViolations,
}

// apertusViolations take each piece of Apertus's xIELU and rotary out. They
// run on the host and on every device: the numbers ride with each block's
// weights, and the rotary factors are folded into the plan's table.
var apertusViolations = []c6Violation{
	{"xIELU's eps ignored", xieluMut(func(p []float32, _ int) { p[3] = float32(math.Inf(1)) })},
	{"xIELU's alphas exchanged", xieluMut(func(p []float32, _ int) { p[0], p[1] = p[1], p[0] })},
	{"xIELU's beta*x dropped", xieluMut(func(p []float32, _ int) { p[2] = 0 })},
	{"xIELU's alpha_p without the softplus", xieluMut(func(p []float32, _ int) {
		p[0] = float32(math.Log(math.Expm1(float64(p[0]))))
	})},
	{"every block running block 0's xIELU", func(m *Model) func() {
		p0 := slices.Clone(m.layers[0].xielu)
		return eachLayer(m, func(l *layer) { l.xielu = p0 })
	}},
	{"no llama 3 rotary factors", noRopeFreqs},
	{"NORM rotary for NEOX", neoxOff},
	{"no q/k norm", noQKNorm},
}

// apertusDevViolations are apertusViolations with the rotary pairing flipped
// in the config, which is what the device reads; the host's table is built
// at Open, so the host list flips that instead.
var apertusDevViolations = append(slices.Clone(apertusViolations[:len(apertusViolations)-2]),
	c6Violation{"NORM rotary for NEOX", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.RopeNeox = false })
	}},
	c6Violation{"no q/k norm", noQKNorm})

// xieluMut runs f on a copy of every block's xIELU numbers.
func xieluMut(f func(p []float32, li int)) func(m *Model) func() {
	return func(m *Model) func() {
		li := 0
		return eachLayer(m, func(l *layer) {
			p := slices.Clone(l.xielu)
			f(p, li)
			l.xielu, li = p, li+1
		})
	}
}

// perHeadQK runs OLMo 2's whole-projection q/k norm one head at a time, with
// each head reading the first head_dim of the weight -- what an engine that
// took the qwen3 reading of the same tensor names would compute.
func perHeadQK(m *Model) func() {
	undoC := cfgMut(m, func(c *Config) { c.QKNormWide = false })
	hd := m.Cfg.HeadDim
	undoL := eachLayer(m, func(l *layer) { l.qNorm, l.kNorm = l.qNorm[:hd], l.kNorm[:hd] })
	undoR := m.restate(func(inv map[jlm.Role]graph.Shape) {
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
func postAsPre(m *Model) func() {
	undoL := eachLayer(m, func(l *layer) {
		l.attnNorm, l.ffnNorm = l.postAttnNorm, l.postFFNNorm
		l.postAttnNorm, l.postFFNNorm = nil, nil
	})
	undoR := m.restate(func(inv map[jlm.Role]graph.Shape) {
		for _, r := range [][2]jlm.Role{{jlm.RolePostAttnNorm, jlm.RoleAttnNorm}, {jlm.RolePostFFNNorm, jlm.RoleFFNNorm}} {
			if s, ok := inv[r[0]]; ok {
				inv[r[1]] = s
				delete(inv, r[0])
			}
		}
	})
	return func() { undoR(); undoL() }
}

var denseDevViolations = map[string][]c6Violation{
	"synth-apertus": apertusDevViolations,
	"synth-olmo2-gqa": {
		{"a per-head q/k norm", perHeadQK},
		{"k's norm read from q's weight", kFromQ},
	},
	"synth-olmo3": {
		{"YaRN on the sliding layers", localYarn},
		{"YaRN's magnitude on the sliding layers", localScaled},
		{"no sliding window", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.SWAWindow = 0 }) }},
	},
	"synth-smollm3": {
		{"rotary on the NoPE layer", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.NoPEGlobal = false }) }},
	},
	"synth-arcee": {
		{"SiLU for relu2", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.Act = nn.ActSiLU }) }},
	},
	"synth-seedoss": {
		{"no q/k/v biases", noQKVBias},
		{"NORM rotary for NEOX", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = false }) }},
	},
	"synth-olmo2": {
		{"a per-head q/k norm", perHeadQK},
		{"the post-norms as pre-norms", postAsPre},
		{"NORM rotary for NEOX", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = false }) }},
	},
	"synth-exaone4": {
		{"rotary on the global layer", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.NoPEGlobal = false }) }},
		{"no sliding window", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.SWAWindow = 0 }) }},
		{"the post-norms as pre-norms", postAsPre},
	},
	"synth-mistral3": {
		{"no attention temperature", noTemp},
	},
}

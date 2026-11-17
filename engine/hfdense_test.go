package engine

import (
	"testing"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
)

// hfDenseSelected is what each dense llama-family safetensors fixture's
// container must carry, so TestSafetensorsMatchTransformers' agreement is
// about the feature and not about a model that lost it on the way in (RULE
// 10: check the configuration under test was SELECTED). Each fixture sets the
// feature at a value that bites (scripts/hfgold.py).
func hfDenseSelected() map[string]func(m *model.Model) bool {
	return map[string]func(m *model.Model) bool{
		// NoPE every THIRD layer, read off no_rope_layers rather than llama.cpp's 4.
		"synth-hf-smollm3": func(m *model.Model) bool {
			c := m.Cfg
			return !c.RopeNeox && c.NoPEGlobal && windowOf(m) == 0 && m.Container.Config().SWAPeriod == 3 &&
				ropesAt(m, 0) && ropesAt(m, 1) && !ropesAt(m, 2) && ropesAt(m, 3) && !ropesAt(m, 5) && !slides(m, 0) &&
				c.TiedEmbd && c.NKVHead < c.NHead
		},
		"synth-hf-arcee": func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return m.Container.Config().Flags.Has(jlm.FlagReLU2) && l.Gate.E == nil && l.Vector(jlm.RoleFFNNorm) != nil && !c.TiedEmbd
		},
		"synth-hf-seedoss": func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.RopeNeox && c.HeadDim*c.NHead != c.NEmbd &&
				l.Vector(jlm.RoleAttnQBias) != nil && l.Vector(jlm.RoleAttnKBias) != nil && l.Vector(jlm.RoleAttnVBias) != nil && l.Vector(jlm.RoleAttnOutBias) == nil && l.Vector(jlm.RoleFFNNorm) != nil && l.Vector(jlm.RolePostAttnNorm) == nil
		},
		"synth-hf-olmo2": func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.RopeNeox && c.QKNorm && c.QKNormWide &&
				len(l.Vector(jlm.RoleAttnQNorm)) == c.NHead*c.HeadDim && l.Vector(jlm.RoleAttnNorm) == nil && l.Vector(jlm.RoleFFNNorm) == nil &&
				l.Vector(jlm.RolePostAttnNorm) != nil && l.Vector(jlm.RolePostFFNNorm) != nil && windowOf(m) == 0
		},
		// OLMo 3: OLMo 2's block with a window on three layers in four, every
		// layer rotating, the sliding ones on a plain rotary of their own.
		"synth-hf-olmo3": func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.QKNormWide && l.Vector(jlm.RoleAttnNorm) == nil && l.Vector(jlm.RolePostFFNNorm) != nil &&
				windowOf(m) == 4 && m.Container.Config().SWAPeriod == 4 && slides(m, 0) && slides(m, 2) && !slides(m, 3) &&
				!c.NoPEGlobal && ropesAt(m, 3) && slidingRotaryPlain(m)
		},
		"synth-hf-exaone4": func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.RopeNeox && c.QKNorm && !c.QKNormWide && len(l.Vector(jlm.RoleAttnQNorm)) == c.HeadDim &&
				l.Vector(jlm.RoleAttnNorm) == nil && l.Vector(jlm.RolePostAttnNorm) != nil && c.NoPEGlobal && windowOf(m) == 4 &&
				m.Container.Config().SWAPeriod == 4 && slides(m, 0) && !slides(m, 3) && ropesAt(m, 2) && !ropesAt(m, 3) && c.NKVHead < c.NHead
		},
		"synth-hf-ministral3": func(m *model.Model) bool {
			c := m.Cfg
			return !c.RopeNeox && c.AttnTempScale == 0.5 && c.AttnTempFloor == 4 &&
				c.AttnTempOffs == 0 && c.YarnFactor == 16 &&
				m.Rope.Scale > 1.1 && m.Rope.Scale < 1.13
		},
	}
}

// TestSafetensorsDenseFamilySelected opens each fixture the way
// TestSafetensorsMatchTransformers does and asserts its container carries the
// feature that fixture exists to gate.
func TestSafetensorsDenseFamilySelected(t *testing.T) {
	for name, sel := range hfDenseSelected() {
		t.Run(name, func(t *testing.T) {
			m, err := model.Open(hfContainer(t, name))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if !sel(m) {
				t.Fatalf("the container does not carry the feature this fixture gates: %+v", *m.Cfg)
			}
		})
	}
}

// slidingRotaryPlain reports that every sliding block's stated rotary turns
// at a magnitude of one and divides by no stored factors, and some block is
// sliding.
func slidingRotaryPlain(m *model.Model) bool {
	n := 0
	for li := range m.Cfg.NLayer {
		g := m.Composing().BlockGraph(li)
		if g == nil {
			return false
		}
		if w, _ := g.AttendWindow(); w == 0 {
			continue
		}
		for _, o := range g.Ops {
			if o.Kind != graph.OpRope {
				continue
			}
			if o.Attr.Freqs || o.Attr.Mscale != 0 && o.Attr.Mscale != 1 {
				return false
			}
			n++
		}
	}
	return n > 0
}

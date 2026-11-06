package engine

import (
	"testing"

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
			return c.Arch == "smollm3" && !c.RopeNeox && c.NoPEGlobal && c.SWAWindow == 0 && c.SWAPeriod == 3 &&
				c.RopeAt(0) && c.RopeAt(1) && !c.RopeAt(2) && c.RopeAt(3) && !c.RopeAt(5) && !c.SWA(0) &&
				c.TiedEmbd && c.NKVHead < c.NHead
		},
		"synth-hf-arcee": func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "arcee" && m.Container.Config().Flags.Has(jlm.FlagReLU2) && l.Gate.E == nil && l.FfnNorm != nil && !c.TiedEmbd
		},
		"synth-hf-seedoss": func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "seed_oss" && c.RopeNeox && c.HeadDim*c.NHead != c.NEmbd &&
				l.Vector(jlm.RoleAttnQBias) != nil && l.Vector(jlm.RoleAttnKBias) != nil && l.Vector(jlm.RoleAttnVBias) != nil && l.Vector(jlm.RoleAttnOutBias) == nil && l.FfnNorm != nil && l.Vector(jlm.RolePostAttnNorm) == nil
		},
		"synth-hf-olmo2": func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "olmo2" && c.RopeNeox && c.QKNorm && c.QKNormWide &&
				len(l.QNorm) == c.NHead*c.HeadDim && l.AttnNorm == nil && l.FfnNorm == nil &&
				l.Vector(jlm.RolePostAttnNorm) != nil && l.Vector(jlm.RolePostFFNNorm) != nil && c.SWAWindow == 0
		},
		// OLMo 3: OLMo 2's block with a window on three layers in four, every
		// layer rotating, the sliding ones on a plain rotary of their own.
		"synth-hf-olmo3": func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "olmo3" && c.QKNormWide && l.AttnNorm == nil && l.Vector(jlm.RolePostFFNNorm) != nil &&
				c.SWAWindow == 4 && c.SWAPeriod == 4 && c.SWA(0) && c.SWA(2) && !c.SWA(3) &&
				!c.NoPEGlobal && c.RopeAt(3) && c.SWARopePlain && m.RopeSWA != nil
		},
		"synth-hf-exaone4": func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "exaone4" && c.RopeNeox && c.QKNorm && !c.QKNormWide && len(l.QNorm) == c.HeadDim &&
				l.AttnNorm == nil && l.Vector(jlm.RolePostAttnNorm) != nil && c.NoPEGlobal && c.SWAWindow == 4 &&
				c.SWAPeriod == 4 && c.SWA(0) && !c.SWA(3) && c.RopeAt(2) && !c.RopeAt(3) && c.NKVHead < c.NHead
		},
		"synth-hf-ministral3": func(m *model.Model) bool {
			c := m.Cfg
			return c.Arch == "mistral3" && !c.RopeNeox && c.AttnTempScale == 0.5 && c.AttnTempFloor == 4 &&
				c.AttnTempOffs == 0 && c.YarnFactor == 16 && c.AttnTemp(1, 3) == 1 && c.AttnTemp(0, 4) != 1 &&
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

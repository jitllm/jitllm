package model

import "math"

// YaRN's magnitude, read from what each family's reference reads
// (convert.ropeScalingOf, convert.hfYarnApply): a config's stated
// attention_factor on cos and sin, else the derived one, and for the DeepSeek
// family the score's own mscale(f, mscale_all_dim)^2. Each fixture is
// transformers' class written to GGUF by llama.cpp's converter
// (scripts/c6gold.py), so the gates are TestC6MatchesTransformers,
// TestC6FeaturesAreLoadBearing and TestC6OnEveryDevice under its name; the same
// weights as safetensors (<name>-hf) run TestSafetensorsMatchTransformers.
// mscale is YaRN's derived magnitude, 0.1*k*ln(factor) + 1.
func mscale(factor, k float64) float64 { return 0.1*k*math.Log(factor) + 1 }

func yarnFixtures() []struct {
	name string
	sel  func(m *Model) bool
} {
	return []struct {
		name string
		sel  func(m *Model) bool
	}{
		// attention_factor 1.5 where the formula gives 1.208, on the global layers
		// only.
		{"synth-olmo3-af", func(m *Model) bool {
			c := m.Cfg
			return c.Arch == "olmo3" && c.YarnFactor == 8 && m.rope.Scale == 1.5 &&
				m.ropeSWA != nil && m.ropeSWA.Scale == 1 && c.AttnScale == 1/math.Sqrt(float64(c.HeadDim))
		}},
		// attention_factor 0.75, outranking the mscale/mscale_all_dim ratio.
		{"synth-mistral3-af", func(m *Model) bool {
			c := m.Cfg
			return c.Arch == "mistral3" && c.YarnFactor == 16 && m.rope.Scale == 0.75 && c.AttnScale == 1/math.Sqrt(float64(c.HeadDim))
		}},
		// mscale == mscale_all_dim: cos and sin at exactly one, the score at
		// mscale(8, 0.707)^2.
		{"synth-deepseek-yarn", func(m *Model) bool {
			c := m.Cfg
			return c.MLA() && c.YarnFactor == 8 && m.rope.Scale == 1 && yarnScoreScaled(c)
		}},
		// And a stated 0.8 on cos and sin beside the same score correction.
		{"synth-deepseek-yarn-af", func(m *Model) bool {
			c := m.Cfg
			return c.MLA() && c.YarnFactor == 8 && float32(m.rope.Scale) == 0.8 && yarnScoreScaled(c)
		}},
	}
}

// yarnScoreScaled reports whether the score carries mscale(8, 0.707)^2 over
// 1/sqrt(head_dim).
func yarnScoreScaled(c *Config) bool {
	want := mscale(8, 0.707) * mscale(8, 0.707) / math.Sqrt(float64(c.HeadDim))
	return math.Abs(c.AttnScale-want) < 1e-6*want
}

// ropeScale sets the magnitude on cos and sin.
func ropeScale(s float64) func(m *Model) func() {
	return func(m *Model) func() {
		was := m.rope.Scale
		m.rope.Scale = s
		return func() { m.rope.Scale = was }
	}
}

func yarnViolations() map[string][]c6Violation {
	return map[string][]c6Violation{
		"synth-olmo3-af": {
			{"the derived magnitude for the stated one", ropeScale(mscale(8, 1))},
		},
		"synth-mistral3-af": {
			{"the mscale ratio for the stated magnitude", ropeScale(mscale(16, 1) / mscale(16, 0.5))},
		},
		"synth-deepseek-yarn": {
			{"the score's correction on cos and sin too", ropeScale(mscale(8, 0.707))},
			{"no correction on the score", func(m *Model) func() {
				return cfgMut(m, func(c *Config) { c.AttnScale = 1 / math.Sqrt(float64(c.HeadDim)) })
			}},
		},
		"synth-deepseek-yarn-af": {
			{"the stated magnitude ignored", ropeScale(1)},
		},
	}
}

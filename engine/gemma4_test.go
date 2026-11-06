package engine

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
)

// Gemma 4 held to transformers' own Gemma4ForCausalLM: scripts/gemma4gold.py
// builds each fixture with random weights (layer_scalar included, which
// transformers initialises to the identity) and writes it to GGUF with
// llama.cpp's own converter. Every tensor is F32, so the bound is c6NMSE.

const gemma4GoldScript = "scripts/gemma4gold.py"

func openGemma4(t *testing.T, name string) (*model.Model, *c6Golden, []Option) {
	t.Helper()
	return openGolden(t, "gemma4", gemma4GoldScript, name)
}

// gemma4Fixtures names each fixture and what its container must carry: the
// global layers' geometry apart from the sliding layers' on every axis.
func gemma4Fixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{
		{"synth-gemma4", func(m *model.Model) bool {
			c := m.Cfg
			ok := c.Arch == "gemma4" && c.MixedGeom() && c.HeadDim == 32 && c.HeadDimAt(0) == 16 &&
				c.NKVHead == 1 && c.NKVHeadAt(0) == 2 && c.NRot == 32 && c.NRotAt(0) == 16 &&
				c.VNorm && c.AttnScale == 1 && c.RopeNeox && c.QKNorm && c.SWAWindow == 4 &&
				c.SWAPeriod == 3 && c.FinalSoftcap == 30 && c.TiedEmbd && m.RopeSWA != nil &&
				len(m.Rope.Freqs) == c.HeadDim/2 && m.RopeSWA.Freqs == nil
			for i := range m.Layers {
				l := &m.Layers[i]
				ok = ok && l.VFromK == !c.SWA(i) && l.OutScale != 0 && l.OutScale != 1 &&
					len(l.QNorm) == c.HeadDimAt(i) && l.Vector(jlm.RolePostAttnNorm) != nil && l.Vector(jlm.RolePostFFNNorm) != nil
			}
			return ok
		}},
		// The real head widths, 256 and 512: the attention kernels at the shape a
		// checkpoint runs them at.
		{"synth-gemma4-wide", func(m *model.Model) bool {
			c := m.Cfg
			return c.Arch == "gemma4" && c.MixedGeom() && c.HeadDim == 512 && c.HeadDimAt(0) == 256 &&
				c.NRot == 512 && c.NRotAt(0) == 256 && c.NKVHead == 1 && c.NKVHeadAt(0) == 1 &&
				m.Layers[2].VFromK && !m.Layers[1].VFromK
		}},
		{"synth-gemma4-kv", func(m *model.Model) bool {
			c := m.Cfg
			ok := c.Arch == "gemma4" && c.MixedGeom() && c.HeadDim == 32 && c.HeadDimAt(0) == 16 &&
				c.NKVHead == 2 && c.NKVHeadAt(0) == 2 && c.VNorm
			for i := range m.Layers {
				ok = ok && !m.Layers[i].VFromK
			}
			return ok
		}},
		{"synth-gemma4-e", func(m *model.Model) bool {
			c := m.Cfg
			ok := c.Arch == "gemma4" && c.MixedGeom() && c.PLEDim == 32 && c.NKVShared == 2 &&
				c.NFFNAt(4) == 2*c.NFFN && c.NFFNAt(3) == c.NFFN &&
				c.KVSource(4) == 3 && c.KVSource(5) == 2 && c.KVSource(1) == 1 &&
				m.PleTok.K == 6*32 && m.PleProj.Rows == 6*32 && len(m.PleNorm) == 32
			for i := range m.Layers {
				l := &m.Layers[i]
				ok = ok && matRows(l, jlm.RolePLEGate) == 32 && matK(l, jlm.RolePLEProj) == 32 && len(l.Vector(jlm.RolePLEPostNorm)) == c.NEmbd &&
					(l.Wk.Rows == 0) == c.KVShared(i) && (l.KNorm == nil) == c.KVShared(i) && !l.VFromK
			}
			return ok
		}},
		// The E-model block at a per-layer input 40 wide: not a whole number of
		// the activation quantizer's blocks, so every tier's quantizer takes its
		// ragged tail (nn's on the host, kernels.QuantizePadded on a device).
		{"synth-gemma4-e-ragged", func(m *model.Model) bool {
			c := m.Cfg
			ok := c.Arch == "gemma4" && c.PLEDim == 40 && c.NKVShared == 2 && m.PleProj.Rows == 6*40
			for i := range m.Layers {
				ok = ok && matRows(&m.Layers[i], jlm.RolePLEGate) == 40 && matK(&m.Layers[i], jlm.RolePLEProj) == 40
			}
			return ok
		}},
		{"synth-gemma4-moe", func(m *model.Model) bool {
			c := m.Cfg
			ok := c.Arch == "gemma4" && c.MixedGeom() && m.Layers[0].Vector(jlm.RoleFFNNorm2) != nil && c.NExpert == 8 &&
				c.NExpertUsed == 2 && c.NFFNExp == 32 && c.NFFNShExp == c.NFFN && !c.NoExpertNorm
			for i := range m.Layers {
				l := &m.Layers[i]
				ok = ok && c.MoEAt(i) && l.Router.Rows == c.NExpert && l.ShGate.Rows == c.NFFN &&
					l.Vector(jlm.RoleShRouter) == nil && len(l.Vector(jlm.RoleExpScale)) == c.NExpert && len(l.Vector(jlm.RoleRouterNorm)) == c.NEmbd &&
					l.Vector(jlm.RoleFFNNorm2) != nil && l.Vector(jlm.RolePostFFNNorm1) != nil && l.Vector(jlm.RolePostFFNNorm2) != nil
			}
			return ok
		}},
	}
}

// gemma4Violations are the features Gemma 4 adds, taken out one at a time on
// the host. Each must move the logits far past c6NMSE.
func gemma4Violations() map[string][]c6Violation {
	return map[string][]c6Violation{
		"synth-gemma4": {
			{"no v norm", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.VNorm = false }) }},
			{"no layer scalar", noLayerScalar},
			{"the sliding rotary on the global layers", globalRopeIsLocal},
			{"the global layers' rotary without its proportional factors", noRopeFreqs},
			{"an attention scale of 1/sqrt(head_dim)", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.AttnScale = 1 / math.Sqrt(float64(c.HeadDim)) })
			}},
			// A window no prompt reaches: the sliding layers keep their geometry
			// (Config.SWA) and attend to everything.
			{"no sliding window", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.SWAWindow = 1 << 20 }) }},
			{"no final softcap", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.FinalSoftcap = 0 }) }},
			{"NORM rotary for NEOX", hostNormRotary},
		},
		"synth-gemma4-kv": {
			{"no v norm", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.VNorm = false }) }},
			{"the sliding rotary on the global layers", globalRopeIsLocal},
		},
		"synth-gemma4-wide": {
			{"no v norm", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.VNorm = false }) }},
			{"the global layers' rotary without its proportional factors", noRopeFreqs},
		},
		"synth-gemma4-moe":      gemma4MoEViolations(),
		"synth-gemma4-e":        gemma4EViolations(),
		"synth-gemma4-e-ragged": gemma4EViolations(),
	}
}

// gemma4EViolations are the E-models' features, one at a time.
func gemma4EViolations() []c6Violation {
	return []c6Violation{
		{"no per-layer embeddings", func(m *model.Model) func() {
			undoC := cfgMut(m, func(c *model.Config) { c.PLEDim = 0 })
			undoR := dropStored(m, jlm.RolePLEGate, jlm.RolePLEProj, jlm.RolePLEPostNorm)
			return func() { undoR(); undoC() }
		}},
		// The sliding shared layer reads the first sliding layer's history rather
		// than the last one's (transformers' store_full_length_kv); the global
		// one has no other source of its kind.
		{"a shared layer reading an earlier source", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) {
				c.KvSrcAt = make([]int, c.NLayer)
				for li := range c.KvSrcAt {
					c.KvSrcAt[li] = 2
				}
				c.KvSrcAt[4] = 1
			})
		}},
		{"the per-layer embedding's post-norm unweighted", func(m *model.Model) func() {
			return eachLayer(m, func(l *model.Layer) { setVec(l, jlm.RolePLEPostNorm, onesLike) })
		}},
	}
}

// gemma4MoEViolations are the 26B block's features, one at a time.
func gemma4MoEViolations() []c6Violation {
	return []c6Violation{
		// Ones rather than nil, so the device runs it rather than declining a
		// missing vector.
		{"no per-expert scale", func(m *model.Model) func() {
			return eachLayer(m, func(l *model.Layer) { setVec(l, jlm.RoleExpScale, onesLike) })
		}},
		{"the router on the experts' input", func(m *model.Model) func() {
			return eachLayer(m, func(l *model.Layer) {
				setVec(l, jlm.RoleRouterNorm, func([]float32) []float32 { return l.Vector(jlm.RoleFFNNorm2) })
			})
		}},
		{"the experts on the dense MLP's input", func(m *model.Model) func() {
			return eachLayer(m, func(l *model.Layer) { setVec(l, jlm.RoleFFNNorm2, func([]float32) []float32 { return l.FfnNorm }) })
		}},
		{"the dense MLP's post-norm on the experts", func(m *model.Model) func() {
			return eachLayer(m, func(l *model.Layer) {
				setVec(l, jlm.RolePostFFNNorm2, func([]float32) []float32 { return l.Vector(jlm.RolePostFFNNorm1) })
			})
		}},
		{"no layer scalar", noLayerScalar},
	}
}

// noLayerScalar runs every block at scale one.
func noLayerScalar(m *model.Model) func() {
	undoL := eachLayer(m, func(l *model.Layer) { l.OutScale = 0 })
	undoR := dropStored(m, jlm.RoleLayerOutScale)
	return func() { undoR(); undoL() }
}

// globalRopeIsLocal gives the global layers the sliding layers' base with no
// factors: one rotary for every layer, at each layer's own width.
func globalRopeIsLocal(m *model.Model) func() {
	was := m.Rope
	m.Rope.Base, m.Rope.Freqs = m.RopeSWA.Base, nil
	return func() { m.Rope = was }
}

// noRopeFreqs turns every pair of the global head (the plain rotary over 32
// dimensions), where the proportional rotary turns the first quarter.
func noRopeFreqs(m *model.Model) func() {
	was := m.Rope.Freqs
	m.Rope.Freqs = nil
	return func() { m.Rope.Freqs = was }
}

// TestGemma4MatchesTransformers holds each fixture to transformers' own class
// on decode, prefill at every length and the batched path, and the container's
// tokenizer to Gemma 4's tokenizer.json.
func TestGemma4MatchesTransformers(t *testing.T) {
	for _, fx := range gemma4Fixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openGemma4(t, fx.name)
			defer m.Close()
			if !fx.sel(m) {
				t.Fatalf("the container does not carry the fixture's graph: %+v", m.Cfg)
			}
			if m.Vocab == nil {
				t.Fatalf("no tokenizer: %v", m.TokErr)
			}
			for _, tk := range g.Texts {
				if got := m.Vocab.Encode(tk.Text, false); !slices.Equal(got, tk.IDs) {
					t.Errorf("Encode(%q)\n  ours           %v\n  tokenizer.json %v", tk.Text, got, tk.IDs)
				}
			}
			worst := c6Worst(t, m, g, func(what string, p int, lg []float32) float64 {
				t.Helper()
				nmse := llama4Cmp(g.Pos[p].Head, lg)
				if math.IsNaN(nmse) || math.IsInf(nmse, 0) || nmse > c6NMSE {
					t.Errorf("%s pos %d: NMSE %.3e against transformers (bound %.0e)", what, p, nmse, c6NMSE)
				}
				if am := Greedy(lg); am != g.Pos[p].Argmax {
					t.Errorf("%s pos %d: argmax %d, transformers %d", what, p, am, g.Pos[p].Argmax)
				}
				return nmse
			}, openOpts...)
			t.Logf("decode, prefill at every length and the batch: worst NMSE %.3e", worst)
		})
	}
}

// gemma4DevViolations are the same features taken out while only the device
// session runs.
func gemma4DevViolations() map[string][]c6Violation {
	return map[string][]c6Violation{
		"synth-gemma4": {
			{"no v norm", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.VNorm = false }) }},
			{"no layer scalar", noLayerScalar},
		},
		"synth-gemma4-kv": {
			{"no v norm", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.VNorm = false }) }},
		},
		"synth-gemma4-wide": {
			{"no v norm", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.VNorm = false }) }},
		},
		"synth-gemma4-moe":      gemma4MoEViolations(),
		"synth-gemma4-e":        gemma4EViolations(),
		"synth-gemma4-e-ragged": gemma4EViolations(),
	}
}

// TestGemma4OnEveryDevice runs each fixture with every block and the head on
// each device present against the host: the sliding and the global layers in
// their own scratch sets (tier/gemma4.go), cut into one submission per
// geometry. Then each feature taken out of the device session alone.
//
// The mixture block's prefill takes a wider bound: it reads 1.255e-05 on Metal
// and 1.5e-06 on CUDA, and 8.388e-12 on Metal with the binary16 attention
// tiles off (gpu.Config.NoFlashPrefill), so the band is the tiles' -- 1.159e-06
// on the dense fixture -- amplified by the post-norms that rescale each
// branch to unit RMS whatever its size.
func TestGemma4OnEveryDevice(t *testing.T) {
	for _, fx := range gemma4Fixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openGemma4(t, fx.name)
			defer m.Close()
			pre := 1e-5
			switch {
			case m.Layers[0].Vector(jlm.RoleFFNNorm2) != nil:
				pre = 5e-5
			case m.Cfg.HeadDim == 512:
				// CUDA's batched scores take binary16 operands (Stats.AttnMMA),
				// and Gemma 4's attention scale of 1 leaves a 512-wide score
				// unscaled: 9.079e-05 on CUDA, 1.760e-10 with f32 scores
				// (gpu.WithoutAttnMMA), 3.2e-11 on Vulkan.
				pre = 2e-4
			}
			fixtureOnEveryDeviceWithin(t, m, g, gemma4DevViolations()[fx.name], pre, openOpts...)
		})
	}
}

// TestGemma4FeaturesAreLoadBearing runs each violation through decode.
func TestGemma4FeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range gemma4Fixtures() {
		vs := gemma4Violations()[fx.name]
		t.Run(fx.name, func(t *testing.T) {
			if len(vs) == 0 {
				t.Fatal("no violations for this fixture")
			}
			m, g, openOpts := openGemma4(t, fx.name)
			defer m.Close()
			for _, v := range vs {
				t.Run(v.name, func(t *testing.T) {
					undo := v.mut(m)
					defer undo()
					st := testInstance(t, m, len(g.IDs)+1, openOpts...)
					defer st.Close()
					worst := 0.0
					for p, id := range g.IDs {
						lg, err := st.Forward(id)
						if err != nil {
							t.Fatal(err)
						}
						worst = max(worst, llama4Cmp(g.Pos[p].Head, lg))
					}
					if !(worst > 1e3*c6NMSE) {
						t.Errorf("the gate cannot see it: worst NMSE %.3e", worst)
					}
					t.Logf("worst NMSE %.3e", worst)
				})
			}
		})
	}
}

// Gemma 4's fixtures run the five principles' relocation and paging gates
// (principleFixtures) and the windowed-layer gate: a relocated history must
// come home at its own layer's geometry, and the sliding layers' pages are a
// window's.
func init() {
}

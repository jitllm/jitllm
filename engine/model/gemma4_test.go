package model

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// Gemma 4 held to transformers' own Gemma4ForCausalLM: scripts/gemma4gold.py
// builds each fixture with random weights (layer_scalar included, which
// transformers initialises to the identity) and writes it to GGUF with
// llama.cpp's own converter. Every tensor is F32, so the bound is c6NMSE.

const gemma4GoldScript = "scripts/gemma4gold.py"

func openGemma4(t *testing.T, name string) (*Model, *c6Golden) {
	t.Helper()
	return openGolden(t, "gemma4", gemma4GoldScript, name)
}

// gemma4Fixtures names each fixture and what its container must carry: the
// global layers' geometry apart from the sliding layers' on every axis.
var gemma4Fixtures = []struct {
	name string
	sel  func(m *Model) bool
}{
	{"synth-gemma4", func(m *Model) bool {
		c := m.Cfg
		ok := c.Arch == "gemma4" && c.GeomSplit() && c.HeadDim == 32 && c.HeadDimSWA == 16 &&
			c.NKVHead == 1 && c.NKVHeadSWA == 2 && c.NRot == 32 && c.NRotSWA == 16 &&
			c.VNorm && c.AttnScale == 1 && c.RopeNeox && c.QKNorm && c.SWAWindow == 4 &&
			c.SWAPeriod == 3 && c.FinalSoftcap == 30 && c.TiedEmbd && m.ropeSWA != nil &&
			len(m.rope.Freqs) == c.HeadDim/2 && m.ropeSWA.Freqs == nil
		for i := range m.layers {
			l := &m.layers[i]
			ok = ok && l.vFromK == !c.SWA(i) && l.outScale != 0 && l.outScale != 1 &&
				len(l.qNorm) == c.HeadDimAt(i) && l.postAttnNorm != nil && l.postFFNNorm != nil
		}
		return ok
	}},
	// The real head widths, 256 and 512: the attention kernels at the shape a
	// checkpoint runs them at.
	{"synth-gemma4-wide", func(m *Model) bool {
		c := m.Cfg
		return c.Arch == "gemma4" && c.GeomSplit() && c.HeadDim == 512 && c.HeadDimSWA == 256 &&
			c.NRot == 512 && c.NRotSWA == 256 && c.NKVHead == 1 && c.NKVHeadSWA == 1 &&
			m.layers[2].vFromK && !m.layers[1].vFromK
	}},
	{"synth-gemma4-kv", func(m *Model) bool {
		c := m.Cfg
		ok := c.Arch == "gemma4" && c.GeomSplit() && c.HeadDim == 32 && c.HeadDimSWA == 16 &&
			c.NKVHead == 2 && c.NKVHeadSWA == 2 && c.VNorm
		for i := range m.layers {
			ok = ok && !m.layers[i].vFromK
		}
		return ok
	}},
	{"synth-gemma4-e", func(m *Model) bool {
		c := m.Cfg
		ok := c.Arch == "gemma4" && c.GeomSplit() && c.PLEDim == 32 && c.NKVShared == 2 &&
			c.DoubleFFN && c.NFFNAt(4) == 2*c.NFFN && c.NFFNAt(3) == c.NFFN &&
			c.KVSource(4) == 3 && c.KVSource(5) == 2 && c.KVSource(1) == 1 &&
			m.pleTok.k == 6*32 && m.pleProj.rows == 6*32 && len(m.pleNorm) == 32
		for i := range m.layers {
			l := &m.layers[i]
			ok = ok && l.pleGate.rows == 32 && l.pleProj.k == 32 && len(l.plePost) == c.NEmbd &&
				(l.wk.rows == 0) == c.KVShared(i) && (l.kNorm == nil) == c.KVShared(i) && !l.vFromK
		}
		return ok
	}},
	// The E-model block at a per-layer input 40 wide: not a whole number of
	// the activation quantizer's blocks, so every tier's quantizer takes its
	// ragged tail (nn's on the host, kernels.QuantizePadded on a device).
	{"synth-gemma4-e-ragged", func(m *Model) bool {
		c := m.Cfg
		ok := c.Arch == "gemma4" && c.PLEDim == 40 && c.NKVShared == 2 && m.pleProj.rows == 6*40
		for i := range m.layers {
			ok = ok && m.layers[i].pleGate.rows == 40 && m.layers[i].pleProj.k == 40
		}
		return ok
	}},
	{"synth-gemma4-moe", func(m *Model) bool {
		c := m.Cfg
		ok := c.Arch == "gemma4" && c.GeomSplit() && c.DenseMoE && c.NExpert == 8 &&
			c.NExpertUsed == 2 && c.NFFNExp == 32 && c.NFFNShExp == c.NFFN && !c.NoExpertNorm
		for i := range m.layers {
			l := &m.layers[i]
			ok = ok && c.MoEAt(i) && l.router.rows == c.NExpert && l.shGate.rows == c.NFFN &&
				l.shRouter == nil && len(l.expScale) == c.NExpert && len(l.routerNorm) == c.NEmbd &&
				l.ffnNorm2 != nil && l.postFFNNorm1 != nil && l.postFFNNorm2 != nil
		}
		return ok
	}},
}

// gemma4Violations are the features Gemma 4 adds, taken out one at a time on
// the host. Each must move the logits far past c6NMSE.
var gemma4Violations = map[string][]c6Violation{
	"synth-gemma4": {
		{"no v norm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.VNorm = false }) }},
		{"no layer scalar", noLayerScalar},
		{"the sliding rotary on the global layers", globalRopeIsLocal},
		{"the global layers' rotary without its proportional factors", noRopeFreqs},
		{"an attention scale of 1/sqrt(head_dim)", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.AttnScale = 1 / math.Sqrt(float64(c.HeadDim)) })
		}},
		// A window no prompt reaches: the sliding layers keep their geometry
		// (Config.SWA) and attend to everything.
		{"no sliding window", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.SWAWindow = 1 << 20 }) }},
		{"no final softcap", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.FinalSoftcap = 0 }) }},
		{"NORM rotary for NEOX", hostNormRotary},
	},
	"synth-gemma4-kv": {
		{"no v norm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.VNorm = false }) }},
		{"the sliding rotary on the global layers", globalRopeIsLocal},
	},
	"synth-gemma4-wide": {
		{"no v norm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.VNorm = false }) }},
		{"the global layers' rotary without its proportional factors", noRopeFreqs},
	},
	"synth-gemma4-moe":      gemma4MoEViolations,
	"synth-gemma4-e":        gemma4EViolations,
	"synth-gemma4-e-ragged": gemma4EViolations,
}

// gemma4EViolations are the E-models' features, one at a time.
var gemma4EViolations = []c6Violation{
	{"no per-layer embeddings", func(m *Model) func() {
		undoC := cfgMut(m, func(c *Config) { c.PLEDim = 0 })
		undoR := dropStored(m, jlm.RolePLEGate, jlm.RolePLEProj, jlm.RolePLEPostNorm)
		return func() { undoR(); undoC() }
	}},
	// The sliding shared layer reads the first sliding layer's history rather
	// than the last one's (transformers' store_full_length_kv); the global
	// one has no other source of its kind.
	{"a shared layer reading an earlier source", func(m *Model) func() {
		return cfgMut(m, func(c *Config) {
			c.kvSrcOf = func(li int) int {
				if li == 4 {
					return 1
				}
				return 2
			}
		})
	}},
	{"the per-layer embedding's post-norm unweighted", func(m *Model) func() {
		ones := make([]float32, m.Cfg.NEmbd)
		for i := range ones {
			ones[i] = 1
		}
		return eachLayer(m, func(l *layer) { l.plePost = ones })
	}},
}

// gemma4MoEViolations are the 26B block's features, one at a time.
var gemma4MoEViolations = []c6Violation{
	// Ones rather than nil, so the device runs it rather than declining a
	// missing vector.
	{"no per-expert scale", func(m *Model) func() {
		ones := make([]float32, m.Cfg.NExpert)
		for i := range ones {
			ones[i] = 1
		}
		return eachLayer(m, func(l *layer) { l.expScale = ones })
	}},
	{"the router on the experts' input", func(m *Model) func() {
		return eachLayer(m, func(l *layer) { l.routerNorm = l.ffnNorm2 })
	}},
	{"the experts on the dense MLP's input", func(m *Model) func() {
		return eachLayer(m, func(l *layer) { l.ffnNorm2 = l.ffnNorm })
	}},
	{"the dense MLP's post-norm on the experts", func(m *Model) func() {
		return eachLayer(m, func(l *layer) { l.postFFNNorm2 = l.postFFNNorm1 })
	}},
	{"no layer scalar", noLayerScalar},
}

// noLayerScalar runs every block at scale one.
func noLayerScalar(m *Model) func() {
	undoL := eachLayer(m, func(l *layer) { l.outScale = 0 })
	undoR := dropStored(m, jlm.RoleLayerOutScale)
	return func() { undoR(); undoL() }
}

// globalRopeIsLocal gives the global layers the sliding layers' base with no
// factors: one rotary for every layer, at each layer's own width.
func globalRopeIsLocal(m *Model) func() {
	was := m.rope
	m.rope.Base, m.rope.Freqs = m.ropeSWA.Base, nil
	return func() { m.rope = was }
}

// noRopeFreqs turns every pair of the global head (the plain rotary over 32
// dimensions), where the proportional rotary turns the first quarter.
func noRopeFreqs(m *Model) func() {
	was := m.rope.Freqs
	m.rope.Freqs = nil
	return func() { m.rope.Freqs = was }
}

// TestGemma4MatchesTransformers holds each fixture to transformers' own class
// on decode, prefill at every length and the batched path, and the container's
// tokenizer to Gemma 4's tokenizer.json.
func TestGemma4MatchesTransformers(t *testing.T) {
	for _, fx := range gemma4Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openGemma4(t, fx.name)
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
			})
			t.Logf("decode, prefill at every length and the batch: worst NMSE %.3e", worst)
		})
	}
}

// gemma4DevViolations are the same features taken out while only the device
// session runs.
var gemma4DevViolations = map[string][]c6Violation{
	"synth-gemma4": {
		{"no v norm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.VNorm = false }) }},
		{"no layer scalar", noLayerScalar},
	},
	"synth-gemma4-kv": {
		{"no v norm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.VNorm = false }) }},
	},
	"synth-gemma4-wide": {
		{"no v norm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.VNorm = false }) }},
	},
	"synth-gemma4-moe":      gemma4MoEViolations,
	"synth-gemma4-e":        gemma4EViolations,
	"synth-gemma4-e-ragged": gemma4EViolations,
}

// TestGemma4OnEveryDevice runs each fixture with every block and the head on
// each device present against the host: the sliding and the global layers in
// their own scratch sets (tier/gemma4.go), cut into one submission per
// geometry. Then each feature taken out of the device session alone.
//
// The mixture block's prefill takes a wider bound: it reads 1.255e-05 on Metal
// and 1.5e-06 on CUDA, and 8.388e-12 on Metal with the binary16 attention
// tiles off (tier.Config.NoFlashPrefill), so the band is the tiles' -- 1.159e-06
// on the dense fixture -- amplified by the post-norms that rescale each
// branch to unit RMS whatever its size.
func TestGemma4OnEveryDevice(t *testing.T) {
	for _, fx := range gemma4Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openGemma4(t, fx.name)
			defer m.Close()
			pre := 1e-5
			switch {
			case m.Cfg.DenseMoE:
				pre = 5e-5
			case m.Cfg.HeadDim == 512:
				// CUDA's batched scores take binary16 operands (Stats.AttnMMA),
				// and Gemma 4's attention scale of 1 leaves a 512-wide score
				// unscaled: 9.079e-05 on CUDA, 1.760e-10 with f32 scores
				// (tier.WithoutAttnMMA), 3.2e-11 on Vulkan.
				pre = 2e-4
			}
			fixtureOnEveryDeviceWithin(t, m, g, gemma4DevViolations[fx.name], pre)
		})
	}
}

// TestGemma4FeaturesAreLoadBearing runs each violation through decode.
func TestGemma4FeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range gemma4Fixtures {
		vs := gemma4Violations[fx.name]
		t.Run(fx.name, func(t *testing.T) {
			if len(vs) == 0 {
				t.Fatal("no violations for this fixture")
			}
			m, g := openGemma4(t, fx.name)
			defer m.Close()
			for _, v := range vs {
				t.Run(v.name, func(t *testing.T) {
					undo := v.mut(m)
					defer undo()
					st := m.NewState(len(g.IDs) + 1)
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
	principleFixtures = append(principleFixtures, "synth-gemma4.gguf", "synth-gemma4-kv.gguf", "synth-gemma4-moe.gguf",
		"synth-gemma4-e.gguf", "synth-gemma4-wide.gguf")
	windowFixtures = append(windowFixtures, "synth-gemma4.gguf")
}

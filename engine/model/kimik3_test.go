package model

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

const k3GoldScript = "scripts/k3gold.py"

func openK3(t *testing.T, name string) (*Model, *c6Golden) {
	t.Helper()
	return openGolden(t, "kimik3", k3GoldScript, name)
}

// k3Fixtures names each fixture and what its container must carry: six
// blocks, MLA at 2 and 5 and KDA elsewhere, a checkpoint every two blocks
// (three in the bank by the head), a dense lead, then latent mixtures (64 ->
// 32) beside one shared expert, situ everywhere, the MLA output gate, the
// full-rank KDA gate and the decay bound, and the latent norms' own epsilon.
var k3Fixtures = []struct {
	name string
	sel  func(m *Model) bool
}{
	{"synth-kimik3", func(m *Model) bool {
		c := m.Cfg
		ok := c.Arch == "kimi-k3" && c.ResAttn() && c.AttnResBlock == 2 && c.Streams() == 4 &&
			c.ExpertLatent == 32 && c.Act == nn.ActSitu && c.KDALowerBound == -5 &&
			c.LatentNormEps == float64(float32(1e-6)) && c.RMSEps == float64(float32(1e-3)) &&
			c.ChanDecay() && c.MLA() && c.NoPosEnc && c.QLoraRank == 32 && c.KVLoraRank == 32 &&
			c.NDenseLead == 1 && c.NFFNShExp == 32 && c.ExpertSigmoid && c.ExpertScale == 2 &&
			len(m.resOut) == 64 &&
			slices.Equal(c.LayerKinds, []jlm.LayerKind{jlm.LayerLinearAttn, jlm.LayerLinearAttn,
				jlm.LayerFullAttn, jlm.LayerLinearAttn, jlm.LayerLinearAttn, jlm.LayerFullAttn})
		for i := range m.layers {
			l := &m.layers[i]
			lin := c.LayerKind(i).Recurrent()
			ok = ok && len(l.resAttn) == 64 && len(l.resFFN) == 64 &&
				(l.mlaGate.e != nil) == !lin && (lin == (l.ssmGate.e != nil)) && l.ssmGA.e == nil &&
				(l.routedDown.e != nil) == (i >= 1) && (l.routedNorm != nil) == (i >= 1)
		}
		return ok
	}},
	// The latent and KDA's head 40 wide: not a whole number of the activation
	// quantizer's blocks, so every tier quantizes the latent and the low-rank
	// gates' rank (the head width) with a ragged tail.
	{"synth-kimik3-ragged", func(m *Model) bool {
		c := m.Cfg
		return c.Arch == "kimi-k3" && c.ExpertLatent == 40 && c.ChanDecay()
	}},
}

// TestKimiK3MatchesReference holds each fixture to Moonshot's own remote code
// (KimiLinearForCausalLM from modeling_kimi_linear.py, fla's kernels as their
// torch references; scripts/k3gold.py) on decode, prefill at every length and
// the batched path, and the container's tokenizer to Kimi-K3's tiktoken.
func TestKimiK3MatchesReference(t *testing.T) {
	for _, fx := range k3Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openK3(t, fx.name)
			defer m.Close()
			if !fx.sel(m) {
				t.Fatalf("the container does not carry the fixture's graph: %+v", m.Cfg)
			}
			if m.Vocab == nil {
				t.Fatalf("no tokenizer: %v", m.TokErr)
			}
			for _, tk := range g.Texts {
				if got := m.Vocab.Encode(tk.Text, false); !slices.Equal(got, tk.IDs) {
					t.Errorf("Encode(%q)\n  ours      %v\n  tiktoken  %v", tk.Text, got, tk.IDs)
				}
			}
			worst := c6Worst(t, m, g, func(what string, p int, lg []float32) float64 {
				t.Helper()
				nmse := llama4Cmp(g.Pos[p].Head, lg)
				if math.IsNaN(nmse) || math.IsInf(nmse, 0) || nmse > c6NMSE {
					t.Errorf("%s pos %d: NMSE %.3e against the reference (bound %.0e)", what, p, nmse, c6NMSE)
				}
				if am := Greedy(lg); am != g.Pos[p].Argmax {
					t.Errorf("%s pos %d: argmax %d, the reference %d", what, p, am, g.Pos[p].Argmax)
				}
				return nmse
			})
			t.Logf("decode, prefill at every length and the batch: worst NMSE %.3e", worst)
		})
	}
}

// k3Violations take one piece of the graph out at a time. Each must move the
// logits far past c6NMSE.
var k3Violations = []c6Violation{
	{"no residual attention: every sublayer reads the running residual", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.k3Fault = k3FaultNoBank })
	}},
	{"a checkpoint block adds its attention instead of restarting", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.k3Fault = k3FaultNoRestart })
	}},
	{"the bank scored on raw values", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.k3Fault = k3FaultRawScores })
	}},
	{"the head reads the running residual alone", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.k3Fault = k3FaultNoHeadMix })
	}},
	{"a checkpoint every three blocks", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.AttnResBlock = 3 })
	}},
	{"the latent sum unnormed", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.k3Fault = k3FaultNoLatentNorm })
	}},
	{"the MLA output ungated", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.k3Fault = k3FaultNoMLAGate })
	}},
	{"the KDA output ungated", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.k3Fault = k3FaultNoKDAGate })
	}},
	{"the KDA decay unbounded (Kimi-Linear's softplus)", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.KDALowerBound = 0 })
	}},
	{"the MLA latent norms at rms_norm_eps (llama.cpp's)", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.LatentNormEps = c.RMSEps })
	}},
	{"plain SwiGLU", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.Act = nn.ActSiLU })
	}},
}

// TestKimiK3FeaturesAreLoadBearing runs each violation through decode.
func TestKimiK3FeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range k3Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openK3(t, fx.name)
			defer m.Close()
			for _, v := range k3Violations {
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

// k3DevViolations are every violation the device runs itself: all of them but
// the head's mix, which the host takes.
var k3DevViolations = func() []c6Violation {
	var out []c6Violation
	for _, v := range k3Violations {
		if v.name != "the head reads the running residual alone" {
			out = append(out, v)
		}
	}
	return out
}()

// TestKimiK3OnEveryDevice runs each fixture with every block and the head on
// each device present against the host -- the residual attention's mixes and
// checkpoints, KDA's full-rank gate and bounded decay, MLA's output gate and
// latent norms, the latent mixture and situ (tier's k3.go) -- on decode and on
// a prefill chunk. Then each piece taken out of the device session alone.
func TestKimiK3OnEveryDevice(t *testing.T) {
	for _, fx := range k3Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openK3(t, fx.name)
			defer m.Close()
			fixtureOnEveryDeviceWithin(t, m, g, k3DevViolations, 1e-5)
		})
	}
}

// TestKimiK3DeviceRunsTheChunkWhole is the device gate's selection check: a
// prompt goes to the device as one batched submission -- every mix that has a
// checkpoint to read, the latent mixture grouped -- and a decode token runs
// every mix and latent mixture there. Without it, a chunk refused and
// replayed a row at a time would pass TestKimiK3OnEveryDevice as well.
func TestKimiK3DeviceRunsTheChunkWhole(t *testing.T) {
	m, g := openK3(t, "synth-kimik3")
	defer m.Close()
	c := m.Cfg
	nl, mixes, latent := c.NLayer, 0, 0
	for li := 0; li < nl; li++ {
		for _, ffn := range []bool{false, true} {
			if c.resBank(li, ffn) > 0 {
				mixes++
			}
		}
		if li >= c.NDenseLead {
			latent++
		}
	}
	ran := 0
	for _, spec := range stepDevices() {
		gpu, err := openTier(t, append([]tier.Option{tier.WithDevices(spec),
			tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			st := m.NewState(len(g.IDs) + 2)
			defer st.Close()
			st.SetDeviceLayers(gpu, -1)
			if st.GPULayers() != nl || !st.HeadOnDevice() {
				t.Fatalf("placed %d of %d blocks, head %v: %v", st.GPULayers(), nl, st.HeadOnDevice(), gpu.Err())
			}
			s0 := gpu.Stats()
			lg, err := st.Prefill(g.IDs)
			if err != nil {
				t.Fatal(err)
			}
			s1 := gpu.Stats()
			if _, err := st.Forward(Greedy(lg)); err != nil {
				t.Fatal(err)
			}
			s2 := gpu.Stats()
			t.Logf("prefill of %d: %d mixes, %d latent mixtures, %d grouped, %d MLA batched; a decode "+
				"token: %d mixes, %d latent mixtures", len(g.IDs), s1.K3Mixes-s0.K3Mixes,
				s1.K3Latent-s0.K3Latent, s1.GroupedMoE-s0.GroupedMoE, s1.MLABatched-s0.MLABatched,
				s2.K3Mixes-s1.K3Mixes, s2.K3Latent-s1.K3Latent)
			if s1.K3Mixes-s0.K3Mixes != mixes || s1.K3Latent-s0.K3Latent != latent ||
				s1.GroupedMoE-s0.GroupedMoE != latent || s1.MLABatched == s0.MLABatched {
				t.Fatalf("the prompt did not run as one batched submission of every block")
			}
			if s2.K3Mixes-s1.K3Mixes != mixes || s2.K3Latent-s1.K3Latent != latent {
				t.Fatalf("a decode token did not run every block on the device")
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

// Kimi-K3 runs the five principles' relocation and paging gates
// (principleFixtures): its KDA blocks carry a recurrent summary and its MLA
// blocks a latent history, both of which page and move with the block.
//
// Its safetensors inputs (KimiK3ForConditionalGeneration, the release's
// compressed-tensors MXFP4 experts in the second) run the generated-code
// sweep, TestEveryModelRunsGenerated, as safetensors.
func init() {
	principleFixtures = append(principleFixtures, "synth-kimik3.gguf")
	auditHFDirs = append(auditHFDirs, "hf/synth-kimik3", "kimik3/Kimi-K3-0.40B-MXFP4-hf")
}

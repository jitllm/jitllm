package engine

import (
	"math"
	"reflect"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/nn"
)

const k3GoldScript = "scripts/k3gold.py"

func openK3(t *testing.T, name string) (*model.Model, *c6Golden, []Option) {
	t.Helper()
	return openGolden(t, "kimik3", k3GoldScript, name)
}

// k3Fixtures names each fixture and what its container must carry: six
// blocks, MLA at 2 and 5 and KDA elsewhere, a checkpoint every two blocks
// (three in the bank by the head), a dense lead, then latent mixtures (64 ->
// 32) beside one shared expert, situ everywhere, the MLA output gate, the
// full-rank KDA gate and the decay bound, and the latent norms' own epsilon.
func k3Fixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{
		{"synth-kimik3", func(m *model.Model) bool {
			c := m.Cfg
			ok := m.Container.Config().AttnResBlock == 2 && c.Streams() == 4 &&
				c.ExpWidth() == 32 && m.Container.Config().SituBeta != 0 && reflect.DeepEqual(c.DeltaDecay, jlm.DeltaDecayBounded(-5)) &&
				c.LatentNormEps == float64(float32(1e-6)) && c.RMSEps == float64(float32(1e-3)) &&
				c.ChanDecay() && c.MLA() && c.NoPosEnc && c.QLoraRank == 32 && c.KVLoraRank == 32 &&
				c.NDenseLead == 1 && c.NFFNShExp == 32 && c.ExpertSigmoid && c.ExpertScale == 2 &&
				len(nn.VectorOf(m.Vecs, jlm.RoleOutputResScore)) == 64 &&
				slices.Equal(c.LayerKinds, []jlm.LayerKind{jlm.LayerLinearAttn, jlm.LayerLinearAttn,
					jlm.LayerFullAttn, jlm.LayerLinearAttn, jlm.LayerLinearAttn, jlm.LayerFullAttn})
			for i := range m.Layers {
				l := &m.Layers[i]
				lin := c.LayerKind(i).Recurrent()
				ok = ok && len(l.Vector(jlm.RoleAttnResScore)) == 64 && len(l.Vector(jlm.RoleFFNResScore)) == 64 &&
					(l.Matrix(jlm.RoleAttnGate) != nil && l.SsmGate.E == nil) == !lin && (lin == (l.SsmGate.E != nil)) && l.SsmGA.E == nil &&
					(l.Matrix(jlm.RoleFFNRoutedDown) != nil) == (i >= 1) && (l.Vector(jlm.RoleFFNRoutedNorm) != nil) == (i >= 1)
			}
			return ok
		}},
		// The latent and KDA's head 40 wide: not a whole number of the activation
		// quantizer's blocks, so every tier quantizes the latent and the low-rank
		// gates' rank (the head width) with a ragged tail.
		{"synth-kimik3-ragged", func(m *model.Model) bool {
			c := m.Cfg
			return c.ExpWidth() == 40 && c.ChanDecay()
		}},
	}
}

// k3NMSE is the bound against Moonshot's reference. Its kernels' torch
// references reduce in an order of their own, so the fixtures sit near
// 1e-9 where c6NMSE's sit at ~1e-13: 9.6e-10 through the bank's own
// kernels, 1.0e-9 through its stated reads, which score with a multiply and
// an add. Every violation lands at 3e-3 or far above
// (TestKimiK3FeaturesAreLoadBearing).
const k3NMSE = 2e-9

// TestKimiK3MatchesReference holds each fixture to Moonshot's own remote code
// (KimiLinearForCausalLM from modeling_kimi_linear.py, fla's kernels as their
// torch references; scripts/k3gold.py) on decode, prefill at every length and
// the batched path, and the container's tokenizer to Kimi-K3's tiktoken.
func TestKimiK3MatchesReference(t *testing.T) {
	for _, fx := range k3Fixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openK3(t, fx.name)
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
				if math.IsNaN(nmse) || math.IsInf(nmse, 0) || nmse > k3NMSE {
					t.Errorf("%s pos %d: NMSE %.3e against the reference (bound %.0e)", what, p, nmse, k3NMSE)
				}
				if am := Greedy(lg); am != g.Pos[p].Argmax {
					t.Errorf("%s pos %d: argmax %d, the reference %d", what, p, am, g.Pos[p].Argmax)
				}
				return nmse
			}, openOpts...)
			t.Logf("decode, prefill at every length and the batch: worst NMSE %.3e", worst)
		})
	}
}

// k3Violations take one piece of the graph out at a time. Each must move the
// logits far past c6NMSE.
func k3Violations() []c6Violation {
	return []c6Violation{
		{"no residual attention: every sublayer reads the running residual", unbanked},
		{"a checkpoint block adds its attention instead of restarting", unrestarted},
		{"the bank scored on raw values", rawBankScores},
		{"the head reads the running residual alone", headUnbanked},
		{"a checkpoint every three blocks", func(m *model.Model) func() {
			return streamsMut(m, func(s *jlm.Streams) {
				c := *m.Container.Config()
				c.AttnResBlock = 3
				convert.StateStreams(&c, func(r jlm.Role, li int32) bool { return m.Stores(int(li), r) })
				*s = c.Streams
			})
		}},
		{"the latent sum unnormed", latentUnnormed},
		{"the MLA output ungated", latentAttnUngated},
		{"the KDA output ungated", recurrentUngated},
		{"the KDA decay unbounded (Kimi-Linear's softplus)", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) { c.DeltaDecay = jlm.DeltaDecaySoftplus() })
		}},
		{"the MLA latent norms at rms_norm_eps (llama.cpp's)", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) { c.LatentNormEps = c.RMSEps })
		}},
		{"plain SwiGLU", actSwap(0)},
	}
}

// TestKimiK3FeaturesAreLoadBearing runs each violation through decode.
func TestKimiK3FeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range k3Fixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openK3(t, fx.name)
			defer m.Close()
			for _, v := range k3Violations() {
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

// TestKimiK3OnEveryDevice runs each fixture with every block and the head on
// each device present against the host -- the residual attention's mixes and
// checkpoints, KDA's full-rank gate and bounded decay, MLA's output gate and
// latent norms, the latent mixture and situ (tier's k3.go) -- on decode and on
// a prefill chunk. Then each piece taken out of the device session alone.
func TestKimiK3OnEveryDevice(t *testing.T) {
	for _, fx := range k3Fixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openK3(t, fx.name)
			defer m.Close()
			// A composed build's blocks read the statement, which each of
			// these takes the piece out of as well.
			fixtureOnEveryDeviceWithin(t, m, g, k3ComposedViolations(t), 1e-5, openOpts...)
		})
	}
}

// TestKimiK3DeviceRunsTheChunkWhole is the device gate's selection check: a
// prompt goes to the device as one batched submission -- every mix that has a
// checkpoint to read, the latent mixture grouped -- and a decode token runs
// every mix and latent mixture there. Without it, a chunk refused and
// replayed a row at a time would pass TestKimiK3OnEveryDevice as well.
func TestKimiK3DeviceRunsTheChunkWhole(t *testing.T) {
	m, g, openOpts := openK3(t, "synth-kimik3")
	defer m.Close()
	c := m.Cfg
	nl := c.NLayer
	ran := 0
	for _, spec := range stepDevices() {
		gpux, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(spec),
			gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)...)
		if err != nil || gpux == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gpux.Close()
			st := testInstance(t, m, len(g.IDs)+2, openOpts...)
			defer st.Close()
			st.SetDeviceLayers(gpux, -1)
			if st.GPULayers() != nl || !st.HeadOnDevice() {
				t.Fatalf("placed %d of %d blocks, head %v: %v", st.GPULayers(), nl, st.HeadOnDevice(), gpux.Err())
			}
			s0 := gpux.Stats()
			lg, err := st.Prefill(g.IDs)
			if err != nil {
				t.Fatal(err)
			}
			s1 := gpux.Stats()
			if _, err := st.Forward(Greedy(lg)); err != nil {
				t.Fatal(err)
			}
			s2 := gpux.Stats()
			// The composer runs each block whole: the prompt's chunk one
			// served prompt part a block, a decode token one block step a
			// block, none declined.
			p0, pd0 := s0.ServedPart(nn.ComposedPrompt)
			p1, pd1 := s1.ServedPart(nn.ComposedPrompt)
			b1, bd1 := s1.ServedPart(nn.ComposedBlock)
			b2, bd2 := s2.ServedPart(nn.ComposedBlock)
			t.Logf("prefill of %d: %d prompt blocks served, %d declined; a decode token: %d block steps served, %d declined",
				len(g.IDs), p1-p0, pd1-pd0, b2-b1, bd2-bd1)
			if p1-p0 != nl || pd1 != pd0 {
				t.Fatalf("the prompt did not run as one composed submission of every block (%s)", declinesOf(s1))
			}
			if b2-b1 != nl || bd2 != bd1 {
				t.Fatalf("a decode token did not run every block composed on the device (%s)", declinesOf(s2))
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

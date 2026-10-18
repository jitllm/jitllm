package model

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

const ds4GoldScript = "scripts/ds4gold.py"

func openDS4(t *testing.T, name string) (*Model, *c6Golden) {
	t.Helper()
	return openGolden(t, "deepseek4", ds4GoldScript, name)
}

// ds4Fixtures names each fixture and what its container must carry: four
// streams; a CSA block (rate 4, the indexer's top 3), an HCA block (rate 8), a
// sliding block and a second CSA block, all under a window of 8, so over the
// golden's 32 positions the window slides, both kinds compress, and from
// position 15 each CSA query reads a subset of its entries; the first two
// blocks route by token id.
var ds4Fixtures = []struct {
	name string
	sel  func(m *Model) bool
}{
	{"synth-deepseek4", func(m *Model) bool {
		c := m.Cfg
		ok := c.Arch == "deepseek4" && c.DSV4() && c.HCMult == 4 && c.HCIters == 20 && c.NKVHead == 1 &&
			c.HeadDim == 32 && c.NRot == 16 && !c.RopeNeox && c.QLoraRank == 32 && c.OGroups == 2 &&
			c.OLoraRank == 32 && c.SWAWindow == 8 && c.CompRateCSA == 4 && c.CompRateHCA == 8 &&
			c.IdxHeads == 8 && c.IdxHeadDim == 32 && c.IdxTopK == 3 && c.NHashLayers == 2 &&
			c.ExpertSqrtSoftplus && c.Act == nn.ActSwiGLUClamp && c.NFFNShExp == 32 && !c.Indexer() &&
			slices.Equal(c.CompKinds, []jlm.CompKind{jlm.CompCSA, jlm.CompHCA, jlm.CompNone, jlm.CompCSA}) &&
			c.YarnFactor == 16 && c.RopeBase == 160000 && c.RopeBaseSWA == 10000 && m.ropeSWA != nil &&
			m.ropeSWA.Freqs == nil && m.rope.Freqs != nil && m.ds4Head != nil
		for i := range m.layers {
			l := &m.layers[i]
			ok = ok && l.ds4 != nil && len(l.ds4.woAG) == 2 && (l.ds4.hash != nil) == (i < 2) &&
				(l.ds4.compKV.e != nil) == (c.CompAt(i) != jlm.CompNone) &&
				(l.idxQB.e != nil) == (c.CompAt(i) == jlm.CompCSA) && len(l.sinks) == 4
		}
		return ok
	}},
}

// TestDeepseek4MatchesTransformers holds each fixture to transformers' own
// DeepseekV4ForCausalLM on decode, prefill at every length and the batched
// path, and the container's tokenizer to DeepSeek-V4's tokenizer.json.
func TestDeepseek4MatchesTransformers(t *testing.T) {
	for _, fx := range ds4Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openDS4(t, fx.name)
			defer m.Close()
			if !fx.sel(m) {
				t.Fatalf("the container does not carry the fixture's graph: %+v", m.Cfg)
			}
			if ent := (len(g.IDs)) / m.Cfg.CompRateCSA; ent <= m.Cfg.IdxTopK {
				t.Fatalf("%d golden positions are %d CSA entries against a top-%d: the indexer keeps "+
					"everything", len(g.IDs), ent, m.Cfg.IdxTopK)
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

// ds4Violations take one piece of the graph out at a time. Each must move the
// logits far past c6NMSE.
var ds4Violations = []c6Violation{
	{"the stream mixer softmaxed and not Sinkhorn-normalised", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.ds4Fault = ds4FaultCombSoftmax })
	}},
	{"the stream mixer untransposed", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.ds4Fault = ds4FaultCombRows })
	}},
	{"three Sinkhorn rounds", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.HCIters = 3 })
	}},
	{"the head's collapse a mean", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.ds4Fault = ds4FaultHeadMean })
	}},
	{"the attention output left rotated", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.ds4Fault = ds4FaultNoDerope })
	}},
	{"the window alone", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.ds4Fault = ds4FaultWindowOnly })
	}},
	{"a CSA entry without the window before", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.ds4Fault = ds4FaultNoOverlap })
	}},
	{"every CSA entry attended", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.ds4Fault = ds4FaultAllEntries })
	}},
	{"the indexer's top-k one short", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.IdxTopK-- })
	}},
	{"the compressed blocks at the sliding rotary", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.ds4Fault = ds4FaultMainRope })
	}},
	{"the hash blocks routed by score", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.ds4Fault = ds4FaultScoreRoute })
	}},
	{"no sinks", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.ds4Fault = ds4FaultNoSinks })
	}},
	{"a window of 7", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.SWAWindow-- })
	}},
	{"a sigmoid gate", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.ExpertSqrtSoftplus, c.ExpertSigmoid = false, true })
	}},
	{"plain SwiGLU", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.Act = nn.ActSiLU })
	}},
}

// ds4DevViolations are every violation the device runs itself: all of them
// but the head's collapse, which the host takes.
var ds4DevViolations = func() []c6Violation {
	var out []c6Violation
	for _, v := range ds4Violations {
		if v.name != "the head's collapse a mean" {
			out = append(out, v)
		}
	}
	return out
}()

// TestDeepseek4OnEveryDevice runs each fixture with every block and the head
// on each device present against the host -- the hyper-connections, the
// window and the entries through the paged pools, the compressors, the
// indexer's selection, the grouped output and the mixture's routing (tier's
// ds4.go) -- on decode and on a prefill chunk. Then each piece taken out of
// the device session alone.
func TestDeepseek4OnEveryDevice(t *testing.T) {
	for _, fx := range ds4Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openDS4(t, fx.name)
			defer m.Close()
			fixtureOnEveryDeviceWithin(t, m, g, ds4DevViolations, 1e-5)
		})
	}
}

// DeepSeek V4 runs the five principles' relocation and paging gates
// (principleFixtures) and the window's (windowFixtures): its cached row
// carries the compressor's pending inputs, so it pages, relocates and is
// restored with the key, and the entries are a second paged history beside it.
func init() {
	principleFixtures = append(principleFixtures, "synth-deepseek4.gguf")
	windowFixtures = append(windowFixtures, "synth-deepseek4.gguf")
	prefixWindowFixtures = append(prefixWindowFixtures, "synth-deepseek4.gguf")
}

// TestDeepseek4FeaturesAreLoadBearing runs each violation through decode.
func TestDeepseek4FeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range ds4Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openDS4(t, fx.name)
			defer m.Close()
			for _, v := range ds4Violations {
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

// TestDeepseek4MixedPromptOnEveryDevice is TestPrefillMixedMatchesPrefill's
// hash-routed half on each device, every block and the head placed: a prompt
// whose middle rows are supplied embeddings naming their token ids routes them
// on the device as the token prompt does (the chunk's ids reach the device,
// SetTokenIDs) and runs as one submission, and the same rows without their
// ids are refused naming the routing block, not routed anywhere.
func TestDeepseek4MixedPromptOnEveryDevice(t *testing.T) {
	m, g := openDS4(t, "synth-deepseek4")
	defer m.Close()
	ids := g.IDs
	cut1, cut2 := 5, 19
	e := embdRows(t, m, ids[cut1:cut2])
	nl := m.Cfg.NLayer
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
			run := func(spans ...Span) ([]float32, error) {
				st := m.NewState(len(ids) + 2)
				defer st.Close()
				st.SetDeviceLayers(gpu, -1)
				if st.GPULayers() != nl || !st.HeadOnDevice() {
					t.Fatalf("placed %d of %d blocks, head %v: %v", st.GPULayers(), nl, st.HeadOnDevice(), gpu.Err())
				}
				s0 := gpu.Stats()
				lg, err := st.PrefillMixed(spans...)
				if err == nil && gpu.Stats().DS4Blocks-s0.DS4Blocks != nl {
					t.Fatalf("the prompt did not run as one submission of every block (%d attention halves)",
						gpu.Stats().DS4Blocks-s0.DS4Blocks)
				}
				return slices.Clone(lg), err
			}
			want, err := run(Span{Tokens: ids})
			if err != nil {
				t.Fatal(err)
			}
			got, err := run(Span{Tokens: ids[:cut1]}, Span{Embd: e, Tokens: ids[cut1:cut2]}, Span{Tokens: ids[cut2:]})
			if err != nil {
				t.Fatal(err)
			}
			nm, _ := nmse32(want, got)
			t.Logf("embedding rows naming their ids against the token prompt: NMSE %.3e", nm)
			if math.IsNaN(nm) || nm > 1e-9 {
				t.Errorf("embedding rows naming their ids: NMSE %.3e against the token prompt on the device", nm)
			}
			if _, err := run(Span{Tokens: ids[:cut1]}, Span{Embd: e}, Span{Tokens: ids[cut2:]}); err == nil ||
				!strings.Contains(err.Error(), "routes by token id") {
				t.Errorf("embedding rows with no ids on the device: %v, want a refusal naming the routing block", err)
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

// TestDeepseek4DeviceRunsTheChunkWhole is the device gate's selection check:
// a prompt goes to the device as one batched submission -- every block's
// attention once, the CSA blocks' indexer selecting, the mixture grouped --
// and each decode token runs every block's attention and both
// hyper-connections there. Without it, a chunk refused and replayed a row at
// a time would pass TestDeepseek4OnEveryDevice as well.
func TestDeepseek4DeviceRunsTheChunkWhole(t *testing.T) {
	m, g := openDS4(t, "synth-deepseek4")
	defer m.Close()
	nl, csa := m.Cfg.NLayer, 0
	for li := 0; li < nl; li++ {
		if m.Cfg.CompAt(li) == jlm.CompCSA {
			csa++
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
			t.Logf("prefill of %d: %d attention halves, %d mixes, %d selections, %d grouped mixtures; a "+
				"decode token: %d attention halves, %d mixes", len(g.IDs), s1.DS4Blocks-s0.DS4Blocks,
				s1.DS4HC-s0.DS4HC, s1.IdxSelects-s0.IdxSelects, s1.GroupedMoE-s0.GroupedMoE,
				s2.DS4Blocks-s1.DS4Blocks, s2.DS4HC-s1.DS4HC)
			if d := s1.DS4Blocks - s0.DS4Blocks; d != nl || s1.DS4HC-s0.DS4HC != 2*nl ||
				s1.IdxSelects-s0.IdxSelects != csa || s1.GroupedMoE == s0.GroupedMoE {
				t.Fatalf("the prompt did not run as one batched submission of every block")
			}
			if s2.DS4Blocks-s1.DS4Blocks != nl || s2.DS4HC-s1.DS4HC != 2*nl {
				t.Fatalf("a decode token did not run every block on the device")
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

// TestDeepseek4LongChunksOnEveryDevice prefills past the golden's 32 rows on
// each device, every block and the head placed, against the host: chunks of
// 96 and 320 rows and a prompt of two chunks (512 and 193). The
// hyper-connection mixer was a 64-wide workgroup launched in groups of 128,
// which Vulkan runs at the width the kernel declares, so every row past the
// first 64 of each 128 kept a stale mix: a 32-row golden never reaches one,
// and from 65 rows the last row's logits came back zero there while CUDA,
// which takes the launch's width, matched.
func TestDeepseek4LongChunksOnEveryDevice(t *testing.T) {
	m, _ := openDS4(t, "synth-deepseek4")
	defer m.Close()
	nl := m.Cfg.NLayer
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
			for _, n := range []int{96, 300, 705} {
				ids := make([]int32, n)
				for i := range ids {
					ids[i] = int32(3 + (i*11)%61)
				}
				run := func(dev bool) []float32 {
					st := m.NewState(n + 1)
					defer st.Close()
					chunks := (n + nn.MaxDevicePrefillChunk - 1) / nn.MaxDevicePrefillChunk
					s0 := gpu.Stats()
					if dev {
						st.SetDeviceLayers(gpu, -1)
						if st.GPULayers() != nl || !st.HeadOnDevice() {
							t.Fatalf("placed %d of %d blocks, head %v: %v", st.GPULayers(), nl, st.HeadOnDevice(),
								gpu.Err())
						}
					}
					lg, err := st.Prefill(ids)
					if err != nil {
						t.Fatal(err)
					}
					if d := gpu.Stats().DS4Blocks - s0.DS4Blocks; dev && d != nl*chunks {
						t.Fatalf("%d rows: %d attention halves on the device, want %d chunk(s) of every block (the "+
							"device: %q)", n, d, chunks, gpu.Err())
					}
					return slices.Clone(lg)
				}
				want, got := run(false), run(true)
				nm, _ := nmse32(want, got)
				t.Logf("%d rows: NMSE %.3e against the host, argmax %d / %d", n, nm, Greedy(got), Greedy(want))
				if math.IsNaN(nm) || nm > 1e-5 || Greedy(got) != Greedy(want) {
					t.Errorf("%d rows: NMSE %.3e against the host, argmax %d, want %d", n, nm, Greedy(got),
						Greedy(want))
				}
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

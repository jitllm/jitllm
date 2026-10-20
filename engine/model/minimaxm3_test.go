package model

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"

	"github.com/jitllm/jitllm/engine/nn"
)

const m3GoldScript = "scripts/minimaxm3gold.py"

func openM3(t *testing.T, name string) (*Model, *c6Golden) {
	t.Helper()
	return openGolden(t, "minimaxm3", m3GoldScript, name)
}

// m3Fixtures names each fixture and what its container must carry: a dense
// lead block, then blocks whose selection keeps two of six 4-position blocks
// over the golden's 24 positions, one per kv group, so each group's sparse
// selection is what is compared.
var m3Fixtures = []struct {
	name string
	sel  func(m *Model) bool
}{
	{"synth-minimaxm3", func(m *Model) bool {
		c := m.Cfg
		ok := c.Arch == "minimax-m3" && c.MSA() && !c.Indexer() && !c.MLA() && c.NKVHead == 2 &&
			c.HeadDim == 16 && c.NRot == 8 && c.RopeNeox && c.QKNorm && !c.QKNormWide &&
			c.IdxHeads == 2 && c.IdxHeadDim == 16 && c.IdxTopK == 2 && c.IdxBlock == 4 &&
			c.IdxLocal == 1 && c.NDenseLead == 1 && c.ExpertSigmoid && c.ExpertScale == 2 &&
			c.NFFNShExp == 32 && c.Act == nn.ActSwiGLUOAI && c.KVRowAt(0) == 32 && c.KVRowAt(1) == 48
		for i := range m.layers {
			l := &m.layers[i]
			if c.MSAAt(i) {
				ok = ok && l.idxQ.rows == 2*16 && l.idxK.rows == 16 && len(l.idxQNorm) == 16 &&
					len(l.idxKNorm) == 16
			} else {
				ok = ok && l.idxQ.e == nil && l.idxK.e == nil
			}
		}
		return ok
	}},
}

// m3Violations take one piece of the block selection out at a time. Each must
// move the logits far past c6NMSE at the positions past the top-k.
var m3Violations = []c6Violation{
	{"attention over every block", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.IdxTopK = 1 << 20 })
	}},
	{"a top-k one short", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.msaFault = msaFaultOneShort })
	}},
	{"the query's own block not forced in", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.msaFault = msaFaultNoLocal })
	}},
	{"one selection for every kv group", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.msaFault = msaFaultSharedGroups })
	}},
	{"a block ranked by its first position", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.msaFault = msaFaultFirstPosition })
	}},
	{"the indexer key unnormed", func(m *Model) func() {
		return eachLayer(m, func(l *layer) {
			if l.idxKNorm != nil {
				w := make([]float32, len(l.idxKNorm))
				for i := range w {
					w[i] = 1
				}
				l.idxKNorm = w
			}
		})
	}},
	{"the indexer query unnormed", func(m *Model) func() {
		return eachLayer(m, func(l *layer) {
			if l.idxQNorm != nil {
				w := make([]float32, len(l.idxQNorm))
				for i := range w {
					w[i] = 1
				}
				l.idxQNorm = w
			}
		})
	}},
	{"plain SwiGLU", func(m *Model) func() {
		// The activation and its bounds: the device composes the unit from
		// the bounds (Config.GLU), so plain SwiGLU is both gone.
		return cfgMut(m, func(c *Config) { c.Act, c.GLU = nn.ActSiLU, jlm.GLU{} })
	}},
}

// TestMiniMaxM3MatchesTransformers holds each fixture to transformers' own
// MiniMaxM3VLForCausalLM on decode, prefill at every length and the batched
// path, and the container's tokenizer to MiniMax-M3's tokenizer.json.
func TestMiniMaxM3MatchesTransformers(t *testing.T) {
	for _, fx := range m3Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openM3(t, fx.name)
			defer m.Close()
			if !fx.sel(m) {
				t.Fatalf("the container does not carry the fixture's graph: %+v", m.Cfg)
			}
			if blocks := (len(g.IDs) + m.Cfg.IdxBlock - 1) / m.Cfg.IdxBlock; blocks <= m.Cfg.IdxTopK {
				t.Fatalf("%d golden positions are %d blocks against a top-%d: the selection keeps "+
					"everything", len(g.IDs), blocks, m.Cfg.IdxTopK)
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

// TestMiniMaxM3FeaturesAreLoadBearing runs each violation through decode.
func TestMiniMaxM3FeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range m3Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openM3(t, fx.name)
			defer m.Close()
			for _, v := range m3Violations {
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

// m3DevViolations are the selection's pieces taken out while only the device
// session runs: the plan's top-k and forced blocks, and the key norm's
// weights, which the tier reads at placement.
var m3DevViolations = []c6Violation{
	{"attention over every block", m3Violations[0].mut},
	{"a top-k one short", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.IdxTopK-- })
	}},
	{"two blocks forced in", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.IdxLocal = 2 })
	}},
	{"the indexer key unnormed", m3Violations[5].mut},
	{"the indexer query unnormed", m3Violations[6].mut},
}

// TestMiniMaxM3OnEveryDevice runs each fixture with every block and the head
// on each device present against the host: the indexer's key as the row's
// last head, the scores, the block ranking, the selection and the gather
// (tier/msa.go), on decode and on a prefill chunk. Then each piece taken out
// of the device session alone.
func TestMiniMaxM3OnEveryDevice(t *testing.T) {
	for _, fx := range m3Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openM3(t, fx.name)
			defer m.Close()
			fixtureOnEveryDeviceWithin(t, m, g, m3DevViolations, 1e-5)
		})
	}
}

// MiniMax-M3 runs the five principles' relocation and paging gates
// (principleFixtures): the indexer's key is a head of the cached row, so it
// pages, relocates and is restored with it.
func init() {
	principleFixtures = append(principleFixtures, "synth-minimaxm3.gguf")
}

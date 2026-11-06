package engine

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

const m3GoldScript = "scripts/minimaxm3gold.py"

func openM3(t *testing.T, name string) (*model.Model, *c6Golden, []Option) {
	t.Helper()
	return openGolden(t, "minimaxm3", m3GoldScript, name)
}

// m3Fixtures names each fixture and what its container must carry: a dense
// lead block, then blocks whose selection keeps two of six 4-position blocks
// over the golden's 24 positions, one per kv group, so each group's sparse
// selection is what is compared.
func m3Fixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{
		{"synth-minimaxm3", func(m *model.Model) bool {
			c := m.Cfg
			s := c.Selection()
			ok := c.Arch == "minimax-m3" && s != nil && s.Rank == jlm.RankBest && !c.MLA() && c.NKVHead == 2 &&
				c.HeadDim == 16 && c.NRot == 8 && c.RopeNeox && c.QKNorm && !c.QKNormWide &&
				s.Heads == 2 && s.Groups == 2 && s.HeadDim == 16 && s.KeyWidth == 16 && s.Keep == 2 && s.Block == 4 &&
				s.Local == 1 && c.NDenseLead == 1 && c.ExpertSigmoid && c.ExpertScale == 2 &&
				c.NFFNShExp == 32 && m.Container.Config().Flags.Has(jlm.FlagSwiGLUOAI) && c.KVRowAt(0) == 32 && c.KVRowAt(1) == 48
			for i := range m.Layers {
				l := &m.Layers[i]
				if c.SelectAt(i) {
					ok = ok && matRows(l, jlm.RoleIdxQ) == 2*16 && matRows(l, jlm.RoleIdxK) == 16 && len(l.Vector(jlm.RoleIdxQNorm)) == 16 &&
						len(l.Vector(jlm.RoleIdxKNorm)) == 16
				} else {
					ok = ok && l.Matrix(jlm.RoleIdxQ) == nil && l.Matrix(jlm.RoleIdxK) == nil
				}
			}
			return ok
		}},
	}
}

// m3Violations take one piece of the block selection out at a time. Each must
// move the logits far past c6NMSE at the positions past the top-k.
func m3Violations() []c6Violation {
	return []c6Violation{
		{"attention over every block", func(m *model.Model) func() {
			return selectionMut(m, func(s *jlm.Selection) { s.Keep = 1 << 20 })
		}},
		{"a top-k one short", func(m *model.Model) func() { return selection(m, func(a *graph.Attrs) { a.TopK-- }) }},
		{"the query's own block not forced in", func(m *model.Model) func() { return selection(m, func(a *graph.Attrs) { a.Local = 0 }) }},
		{"one selection for every kv group", sharedSelection},
		{"a block ranked by its first position", func(m *model.Model) func() { return selection(m, func(a *graph.Attrs) { a.Firsts = true }) }},
		{"the indexer key unnormed", func(m *model.Model) func() {
			return eachLayer(m, func(l *model.Layer) {
				setVec(l, jlm.RoleIdxKNorm, onesLike)
			})
		}},
		{"the indexer query unnormed", func(m *model.Model) func() {
			return eachLayer(m, func(l *model.Layer) {
				setVec(l, jlm.RoleIdxQNorm, onesLike)
			})
		}},
		{"plain SwiGLU", actSwap(0)},
	}
}

// TestMiniMaxM3MatchesTransformers holds each fixture to transformers' own
// MiniMaxM3VLForCausalLM on decode, prefill at every length and the batched
// path, and the container's tokenizer to MiniMax-M3's tokenizer.json.
func TestMiniMaxM3MatchesTransformers(t *testing.T) {
	for _, fx := range m3Fixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openM3(t, fx.name)
			defer m.Close()
			if !fx.sel(m) {
				t.Fatalf("the container does not carry the fixture's graph: %+v", m.Cfg)
			}
			s := m.Cfg.Selection()
			if blocks := (len(g.IDs) + int(s.Block) - 1) / int(s.Block); blocks <= int(s.Keep) {
				t.Fatalf("%d golden positions are %d blocks against a top-%d: the selection keeps "+
					"everything", len(g.IDs), blocks, s.Keep)
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

// TestMiniMaxM3FeaturesAreLoadBearing runs each violation through decode.
func TestMiniMaxM3FeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range m3Fixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openM3(t, fx.name)
			defer m.Close()
			for _, v := range m3Violations() {
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

// m3DevViolations are the selection's pieces taken out while only the device
// session runs: the plan's top-k and forced blocks, and the key norm's
// weights, which the tier reads at placement.
func m3DevViolations() []c6Violation {
	return []c6Violation{
		{"attention over every block", m3Violations()[0].mut},
		{"a top-k one short", func(m *model.Model) func() {
			return selectionMut(m, func(s *jlm.Selection) { s.Keep-- })
		}},
		{"two blocks forced in", func(m *model.Model) func() {
			return selectionMut(m, func(s *jlm.Selection) { s.Local = 2 })
		}},
		{"the indexer key unnormed", m3Violations()[5].mut},
		{"the indexer query unnormed", m3Violations()[6].mut},
	}
}

// TestMiniMaxM3OnEveryDevice runs each fixture with every block and the head
// on each device present against the host: the indexer's key as the row's
// last head, the scores, the block ranking, the selection and the gather
// (tier/msa.go), on decode and on a prefill chunk. Then each piece taken out
// of the device session alone.
func TestMiniMaxM3OnEveryDevice(t *testing.T) {
	for _, fx := range m3Fixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openM3(t, fx.name)
			defer m.Close()
			fixtureOnEveryDeviceCounted(t, m, g, m3DevViolations(), 1e-5, true, openOpts...)
		})
	}
}

// MiniMax-M3 runs the five principles' relocation and paging gates
// (principleFixtures): the indexer's key is a head of the cached row, so it
// pages, relocates and is restored with it.
func init() {
}

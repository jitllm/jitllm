package engine

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
)

const ds32GoldScript = "scripts/ds32gold.py"

func openDS32(t *testing.T, name string) (*model.Model, *c6Golden, []Option) {
	t.Helper()
	return openGolden(t, "deepseek32", ds32GoldScript, name)
}

// ds32Fixtures names each fixture and what its container must carry: MLA with
// the V3 router and an indexer whose top-k is below the golden's twelve
// positions, so the sparse path is what is compared.
func ds32Fixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{
		{"synth-deepseek32", func(m *model.Model) bool {
			c := m.Cfg
			s := c.Selection()
			ok := c.MLA() && s != nil && s.Rank == jlm.RankBest && s.Block == 1 && s.Relu &&
				s.Weight != nil && c.QLoraRank == 32 && c.KVLoraRank == 16 && c.NRot == 8 && s.Heads == 8 && s.HeadDim == 16 &&
				s.Keep == 4 && c.KVDim() == 16+8+16 && c.ExpertSigmoid && c.NExpertGroup == 2 &&
				c.NDenseLead == 1 && c.YarnFactor == 8
			for i := range m.Layers {
				l := &m.Layers[i]
				ok = ok && matRows(l, jlm.RoleIdxQB) == 8*16 && matRows(l, jlm.RoleIdxK) == 16 && matRows(l, jlm.RoleIdxProj) == 8 &&
					len(l.Vector(jlm.RoleIdxKNorm)) == 16 && len(l.Vector(jlm.RoleIdxKNormB)) == 16
			}
			return ok
		}},
	}
}

// ds32Violations take one piece of the indexer out at a time. Each must move
// the logits far past c6NMSE at the positions past the top-k.
func ds32Violations() []c6Violation {
	return []c6Violation{
		{"attention over every position", func(m *model.Model) func() {
			return selectionMut(m, func(s *jlm.Selection) { s.Keep = 1 << 20 })
		}},
		{"a top-k one short", func(m *model.Model) func() {
			return selectionMut(m, func(s *jlm.Selection) { s.Keep-- })
		}},
		{"the indexer key unnormed", func(m *model.Model) func() {
			return eachLayer(m, func(l *model.Layer) {
				setVec(l, jlm.RoleIdxKNorm, onesLike)
				setVec(l, jlm.RoleIdxKNormB, func(v []float32) []float32 { return make([]float32, len(v)) })
			})
		}},
		{"the indexer rotary on adjacent pairs", func(m *model.Model) func() {
			return restated(m, editing(func(g *graph.Graph) func(int, *graph.Op) bool {
				return func(_ int, o *graph.Op) bool {
					if o.Kind != graph.OpRope || len(o.Out) == 0 {
						return false
					}
					// The indexer's key and query rotaries, its stated
					// programs' (Model.blockOps, "rope" and the builder's
					// suffix); the attention's are named for their operand.
					return strings.HasPrefix(g.Values[o.Out[0]].Name, "rope.")
				}
			}, func(o *graph.Op) { o.Attr.Layout = graph.RopeNorm }))
		}},
		{"no ReLU on the heads' scores", func(m *model.Model) func() {
			return restated(m, editing(func(*graph.Graph) func(int, *graph.Op) bool {
				return func(_ int, o *graph.Op) bool { return o.Kind == graph.OpSelect && o.Attr.Fn == graph.MapMax }
			}, func(o *graph.Op) { o.Attr.Fn = 0 }))
		}},
	}
}

// TestDeepseek32MatchesTransformers holds each fixture to transformers' own
// DeepseekV32ForCausalLM on decode, prefill at every length and the batched
// path, and the container's tokenizer to DeepSeek-V3.2's tokenizer.json.
func TestDeepseek32MatchesTransformers(t *testing.T) {
	for _, fx := range ds32Fixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openDS32(t, fx.name)
			defer m.Close()
			if !fx.sel(m) {
				t.Fatalf("the container does not carry the fixture's graph: %+v", m.Cfg)
			}
			if keep := int(m.Cfg.Selection().Keep); len(g.IDs) <= keep {
				t.Fatalf("%d golden positions against a top-%d: the indexer selects nothing",
					len(g.IDs), keep)
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

// TestDeepseek32FeaturesAreLoadBearing runs each violation through decode.
func TestDeepseek32FeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range ds32Fixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openDS32(t, fx.name)
			defer m.Close()
			for _, v := range ds32Violations() {
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

// ds32DevViolations are the indexer's pieces taken out while only the device
// session runs: the plan's top-k and the key norm's weights, which the tier
// reads at placement.
func ds32DevViolations() []c6Violation {
	return []c6Violation{
		{"attention over every position", func(m *model.Model) func() {
			return selectionMut(m, func(s *jlm.Selection) { s.Keep = 1 << 20 })
		}},
		{"a top-k one short", func(m *model.Model) func() {
			return selectionMut(m, func(s *jlm.Selection) { s.Keep-- })
		}},
		{"the indexer key unnormed", ds32Violations()[2].mut},
	}
}

// TestDeepseek32OnEveryDevice runs each fixture with every block and the head
// on each device present against the host: the indexer's key in the paged
// row's tail, its scores, the selection and the gather (tier/indexer.go), on
// decode and on a prefill chunk. Then each piece taken out of the device
// session alone.
func TestDeepseek32OnEveryDevice(t *testing.T) {
	for _, fx := range ds32Fixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openDS32(t, fx.name)
			defer m.Close()
			fixtureOnEveryDeviceCounted(t, m, g, ds32DevViolations(), 1e-5, true, openOpts...)
		})
	}
}

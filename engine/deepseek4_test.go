package engine

import (
	"github.com/jitllm/jitllm/convert"
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
)

// ds4NMSE is the bound against transformers' DeepseekV4ForCausalLM: its
// Sinkhorn rounds and the hyper-connections' float32 mixes sit above the
// other flagships' ~1e-12 (3.2e-10 on the full fixture, recorded in
// model-correctness.md).
const ds4NMSE = 1e-9

// ds4Fixtures are the DeepSeek V4 fixtures the engine composes, and what
// each container must carry: the hyper-connected streams stated, and the
// first two blocks routed by their token tables.
func ds4Fixtures() []string { return []string{"synth-deepseek4-sliding"} }

// TestDeepseek4MatchesTransformers holds the composed DeepSeek V4 to
// transformers' own class (scripts/ds4gold.py) on decode, prefill at every
// length and the batch: four hyper-connected streams mixed by Sinkhorn
// rounds, read and written through the converter's stated programs, and
// mixtures whose first blocks route by the token's row of a stored table.
func TestDeepseek4MatchesTransformers(t *testing.T) {
	for _, name := range ds4Fixtures() {
		t.Run(name, func(t *testing.T) {
			m, g, openOpts := openGolden(t, "deepseek4", "scripts/ds4gold.py", name)
			defer m.Close()
			if !ds4Carries(m) {
				t.Fatalf("the container does not carry the fixture's graph: %+v", m.Cfg)
			}
			if m.Vocab == nil {
				t.Fatalf("no tokenizer: %v", m.TokErr)
			}
			for _, tk := range g.Texts {
				if got := m.Vocab.Encode(tk.Text, false); !slices.Equal(got, tk.IDs) {
					t.Errorf("Encode(%q)\n  ours          %v\n  transformers  %v", tk.Text, got, tk.IDs)
				}
			}
			worst := c6Worst(t, m, g, func(what string, p int, lg []float32) float64 {
				t.Helper()
				nmse := llama4Cmp(g.Pos[p].Head, lg)
				if math.IsNaN(nmse) || math.IsInf(nmse, 0) || nmse > ds4NMSE {
					t.Errorf("%s pos %d: NMSE %.3e against transformers (bound %.0e)", what, p, nmse, ds4NMSE)
				}
				if am := Greedy(lg); am != g.Pos[p].Argmax {
					t.Errorf("%s pos %d: argmax %d, transformers' %d", what, p, am, g.Pos[p].Argmax)
				}
				return nmse
			}, openOpts...)
			t.Logf("decode, prefill at every length and the batch: worst NMSE %.3e", worst)
		})
	}
}

// ds4Carries says m states its hyper-connected streams and its first two
// blocks store a token table.
func ds4Carries(m *model.Model) bool {
	if m.Cfg.Streams() != 4 || !m.Cfg.StatesStreams() || len(m.Layers) < 3 {
		return false
	}
	return m.Layers[0].Held(jlm.RoleHashExperts) != nil && m.Layers[1].Held(jlm.RoleHashExperts) != nil &&
		m.Layers[2].Held(jlm.RoleHashExperts) == nil
}

// ds4Violations take one stated piece of the graph out at a time; each must
// move the logits far past ds4NMSE.
func ds4Violations() []c6Violation {
	rounds := func(n uint32) func(m *model.Model) func() {
		return func(m *model.Model) func() {
			return streamsMut(m, func(s *jlm.Streams) {
				c := *m.Container.Config()
				c.HCIters = n
				convert.StateStreams(&c, func(r jlm.Role, li int32) bool { return m.Stores(int(li), r) })
				*s = c.Streams
			})
		}
	}
	return []c6Violation{
		{"the mixer softmaxed and not Sinkhorn-normalised", rounds(1)},
		{"three Sinkhorn rounds", rounds(3)},
		{"q not normed", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.QUnitNorm = false }) }},
		{"the rotary on each head's first elements", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) { c.RopeTail = false })
		}},
		{"the output left rotated", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) { c.OutUnrotated = false })
		}},
		{"plain SwiGLU", actSwap(0)},
	}
}

// TestDeepseek4FeaturesAreLoadBearing runs each violation through decode.
func TestDeepseek4FeaturesAreLoadBearing(t *testing.T) {
	for _, name := range ds4Fixtures() {
		t.Run(name, func(t *testing.T) {
			m, g, openOpts := openGolden(t, "deepseek4", "scripts/ds4gold.py", name)
			defer m.Close()
			for _, v := range ds4Violations() {
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
					if !(worst > 1e3*ds4NMSE) {
						t.Errorf("the gate cannot see it: worst NMSE %.3e", worst)
					}
					t.Logf("worst NMSE %.3e", worst)
				})
			}
		})
	}
}

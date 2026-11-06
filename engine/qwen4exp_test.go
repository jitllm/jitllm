package engine

import (
	"math"
	"reflect"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
)

const q4eGoldScript = "scripts/qwen4expgold.py"

func init() {
}

func openQ4E(t *testing.T, name string) (*model.Model, *c6Golden, []Option) {
	t.Helper()
	return openGolden(t, "qwen4exp", q4eGoldScript, name)
}

// q4eFixtures names each fixture and what its container must carry: four
// hyper-connection streams at rank 32 (40 on the ragged fixture); linear, attention, linear, attention;
// QSA on both attention blocks (budget 8 at a ratio of 4, so from position 12
// a query reads a subset of its keys); the n-gram block on block 2 (3-grams,
// two heads each, rows of 16); the delta rule's gate a sigmoid.
func q4eFixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{
		{"synth-qwen4exp", func(m *model.Model) bool {
			c := m.Cfg
			s := c.Selection()
			return c.Arch == "qwen4exp" && c.HCMult == 4 && c.HCLowRank == 32 && s != nil && s.Rank == jlm.RankPooled &&
				s.Block == 4 && s.Keep == 2 && s.Heads == 8 && s.HeadDim == 32 && c.NGramBlock == 3 &&
				c.NGram == 3 && c.NGramHeadsPer == 2 && c.NGramRowDim == 16 && c.NGramConv == 4 &&
				reflect.DeepEqual(c.DeltaGate, jlm.DeltaGateSigmoid()) && c.NRot == 16 && c.RopeNeox && c.AttnOutGate &&
				c.LayerKind(0).Recurrent() && !c.LayerKind(1).Recurrent() &&
				c.LayerKind(2).Recurrent() && !c.LayerKind(3).Recurrent() && m.Ngram != nil
		}},
		// The hyper-connections' rank 40: not a whole number of the activation
		// quantizer's blocks, so every tier quantizes it with a ragged tail.
		{"synth-qwen4exp-ragged", func(m *model.Model) bool {
			c := m.Cfg
			return c.Arch == "qwen4exp" && c.HCMult == 4 && c.HCLowRank == 40 && m.Ngram != nil
		}},
	}
}

// TestQwen4ExpMatchesTransformers holds each fixture to transformers' own
// Qwen4ExpForCausalLM on decode, prefill at every length and the batched
// path, and the container's tokenizer to Qwen3.8-Flash-Next's tokenizer.json.
func TestQwen4ExpMatchesTransformers(t *testing.T) {
	for _, fx := range q4eFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openQ4E(t, fx.name)
			defer m.Close()
			if !fx.sel(m) {
				t.Fatalf("the container does not carry the fixture's graph: %+v", m.Cfg)
			}
			s := m.Cfg.Selection()
			if blocks := len(g.IDs) / int(s.Block); blocks <= int(s.Keep) {
				t.Fatalf("%d golden positions are %d QSA blocks against a budget of %d: every key is read",
					len(g.IDs), blocks, s.Keep)
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

// q4eViolations take one piece of the graph out at a time, the
// hyper-connection's and QSA's pooling as rewrites of the composed statement
// (q4eComposedViolations). Each must move the logits far past c6NMSE.
func q4eViolations() []c6Violation {
	return []c6Violation{
		{"no n-gram embedding", func(m *model.Model) func() {
			return streamsMut(m, func(s *jlm.Streams) {
				for i := range s.Blocks {
					s.Blocks[i].Pre = jlm.SubStreams{}
				}
			})
		}},
		{"the n-gram embedding without its convolution", func(m *model.Model) func() {
			return streamsMut(m, func(s *jlm.Streams) {
				for i := range s.Blocks {
					p := s.Blocks[i].Pre.Read
					if p == nil {
						continue
					}
					// The sum of the gated values and the convolution's silu
					// keeps the gated values alone.
					q := *p
					q.Ops = slices.Clone(p.Ops)
					q.Out = q.Ops[int(q.Out)-int(q.NIn)].In[0]
					s.Blocks[i].Pre.Read = &q
				}
			})
		}},
		{"the n-gram window not reset at the end of a sequence", func(m *model.Model) func() {
			return restated(m, editing(opsOfKind(graph.OpNGramHash), func(o *graph.Op) { o.Attr.Reset = false }))
		}},
		{"another 2-gram multiplier", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) {
				c.NGramMul = slices.Clone(c.NGramMul)
				c.NGramMul[1] += 2
			})
		}},
		{"the QSA budget a block short", func(m *model.Model) func() {
			return selectionMut(m, func(s *jlm.Selection) { s.Keep-- })
		}},
		{"the delta rule's output gated by silu", recurrentSiLUGated},
	}
}

// TestQwen4ExpFeaturesAreLoadBearing runs each violation through decode: the
// fixture must see every one.
func TestQwen4ExpFeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range q4eFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openQ4E(t, fx.name)
			defer m.Close()
			for _, v := range append(slices.Clone(q4eViolations()), q4eComposedViolations(t)...) {
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

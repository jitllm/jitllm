package engine

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

const gemma3nGoldScript = "scripts/gemma3ngold.py"

func openGemma3n(t *testing.T, name string) (*model.Model, *c6Golden, []Option) {
	t.Helper()
	return openGolden(t, "gemma3n", gemma3nGoldScript, name)
}

// gemma3nFixtures names each fixture and what its container must carry: four
// AltUp streams, LAuReL, per-layer inputs, KV sharing and a sparse lead.
func gemma3nFixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{
		{"synth-gemma3n", func(m *model.Model) bool {
			c, jc := m.Cfg, m.Container.Config()
			ok := c.Arch == "gemma3n" && c.Streams() == 4 && len(jc.Streams.Expand) == 3 && c.PLEDim == 32 &&
				c.NKVShared == 2 && jc.NSparse == 3 && jc.SparseStd > 1.64 && jc.SparseStd < 1.65 && c.VNorm && c.SWAPeriod == 5
			for i := range m.Layers {
				l := &m.Layers[i]
				ok = ok && matRows(l, jlm.RoleLowRankIn) == 32 && matRows(l, jlm.RoleStreamRouter) == 4 &&
					matRows(l, jlm.RoleStreamPredictCoef) == 16
			}
			return ok
		}},
	}
}

// gemma3nActs is a violation restating every block's activation as the
// container states it (ActLayers), which is what the front end builds the
// blocks from: to picks block li's from the blocks' own.
func gemma3nActs(name string, to func(c *jlm.Config, acts []uint8) func(li int) uint8) c6Violation {
	return c6Violation{name, func(m *model.Model) func() {
		jc := m.Container.Config()
		was := slices.Clone(jc.ActLayers)
		pick := to(jc, was)
		for li := range jc.ActLayers {
			jc.ActLayers[li] = pick(li)
		}
		undo := cfgMut(m, func(*model.Config) {})
		return func() {
			copy(jc.ActLayers, was)
			undo()
		}
	}}
}

// gemma3nViolations take one piece of Gemma 3n out at a time.
func gemma3nViolations() []c6Violation {
	return []c6Violation{
		{"no AltUp prediction", func(m *model.Model) func() {
			return streamsMut(m, func(s *jlm.Streams) {
				for i := range s.Blocks {
					s.Blocks[i].Pre.Writes = make([]*jlm.Program, len(s.Blocks[i].Pre.Writes))
				}
			})
		}},
		{"no AltUp correction", func(m *model.Model) func() {
			return streamsMut(m, func(s *jlm.Streams) {
				for i := range s.Blocks {
					s.Blocks[i].FFN.Post = uncorrected(s.Blocks[i].FFN.Post)
				}
			})
		}},
		{"no LAuReL", func(m *model.Model) func() {
			return eachLayer(m, func(l *model.Layer) {
				setVec(l, jlm.RoleLowRankNorm, func(v []float32) []float32 { return make([]float32, len(v)) })
			})
		}},
		gemma3nActs("no gaussian top-k", func(c *jlm.Config, a []uint8) func(int) uint8 {
			return func(int) uint8 { return a[c.NSparse] }
		}),
		gemma3nActs("the gaussian on every block", func(c *jlm.Config, a []uint8) func(int) uint8 {
			return func(int) uint8 { return a[0] }
		}),
		{"no correction scale", func(m *model.Model) func() {
			return eachLayer(m, func(l *model.Layer) {
				setVec(l, jlm.RoleStreamCorrectScale, onesLike)
			})
		}},
	}
}

// TestGemma3nMatchesTransformers holds each fixture to transformers' own
// Gemma3nForCausalLM on decode, prefill at every length and the batched path,
// and the container's tokenizer to Gemma 3n's tokenizer.json.
func TestGemma3nMatchesTransformers(t *testing.T) {
	for _, fx := range gemma3nFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openGemma3n(t, fx.name)
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

// TestGemma3nFeaturesAreLoadBearing runs each violation through decode.
func TestGemma3nFeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range gemma3nFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openGemma3n(t, fx.name)
			defer m.Close()
			for _, v := range gemma3nViolations() {
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

// gemma3nDevViolations take a piece out while only the device session runs.
func gemma3nDevViolations() []c6Violation {
	return []c6Violation{gemma3nViolations()[0], gemma3nViolations()[2], gemma3nViolations()[5],
		gemma3nViolations()[4]}
}

// TestGemma3nOnEveryDevice runs each fixture with every block and the head on
// each device present against the host: the streams stream-major in the
// device's residual, the prediction, LAuReL, the gaussian top-k and the
// correction in the session (tier/altup.go), on decode and a prefill chunk.
func TestGemma3nOnEveryDevice(t *testing.T) {
	for _, fx := range gemma3nFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openGemma3n(t, fx.name)
			defer m.Close()
			fixtureOnEveryDeviceWithin(t, m, g, gemma3nDevViolations(), 1e-5, openOpts...)
		})
	}
}

// TestBatchSeamMovesCarryEveryRowAltUp is the seam-move gate on Gemma 3n:
// a batch's rows carry four residual streams across every move, stream-major
// in a chunk and in a device's residual alike, beside a KV-sharing pair whose
// source and reader must move together.
func TestBatchSeamMovesCarryEveryRowAltUp(t *testing.T) {
	w := fusedRunsCheck(t, false)
	defer w.done()
	p, ok := existingModel(testmodels.Path("synth-gemma3n.gguf"))
	if !ok {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path("synth-gemma3n.gguf")+
			" (set JITLLM_MODELS to the model directory) -- run "+gemma3nGoldScript+" (RULE 11)")
	}
	batchSeamMoves(t, p, 1e-3, true, w)
}

// uncorrected is the shared program p of a block's correction with its
// coefficients read as zero: every stream's innovation weighed by one.
func uncorrected(p *jlm.Program) *jlm.Program {
	q := *p
	q.Ops = slices.Clone(p.Ops)
	for j, o := range q.Ops {
		if o.Code != jlm.TOpLinear || len(o.Bind) == 0 || o.Bind[0] != jlm.RoleStreamCorrectCoef || j+1 >= len(q.Ops) {
			continue
		}
		var e jlm.ExprBuilder
		e.Fn(jlm.FnAdd, e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Num(0)), jlm.Num(1))
		next := q.Ops[j+1]
		next.Expr = &jlm.Expr{Terms: e.T}
		q.Ops[j+1] = next
		return &q
	}
	// A correction that states no coefficients is left as it is: the gate
	// then reports a violation it cannot see.
	return p
}

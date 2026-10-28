package model

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

const gemma3nGoldScript = "scripts/gemma3ngold.py"

func openGemma3n(t *testing.T, name string) (*Model, *c6Golden) {
	t.Helper()
	return openGolden(t, "gemma3n", gemma3nGoldScript, name)
}

// gemma3nFixtures names each fixture and what its container must carry: four
// AltUp streams, LAuReL, per-layer inputs, KV sharing and a sparse lead.
var gemma3nFixtures = []struct {
	name string
	sel  func(m *Model) bool
}{
	{"synth-gemma3n", func(m *Model) bool {
		c := m.Cfg
		ok := c.Arch == "gemma3n" && c.PredStreams == 4 && c.PLEDim == 32 && c.NKVShared == 2 && c.NSparse == 3 &&
			c.SparseStd > 1.64 && c.SparseStd < 1.65 && c.VNorm && c.SWAPeriod == 5 && len(m.altUnembd) == 3
		for i := range m.layers {
			l := &m.layers[i]
			ok = ok && l.laurelL.rows == 32 && l.altRouter.rows == 4 && len(l.altPredT) == 64
		}
		return ok
	}},
}

// gemma3nViolations take one piece of Gemma 3n out at a time.
var gemma3nViolations = []c6Violation{
	{"no AltUp prediction", func(m *Model) func() {
		return eachLayer(m, func(l *layer) { l.altPredT = make([]float32, len(l.altPredT)) })
	}},
	{"no AltUp correction", func(m *Model) func() {
		return eachLayer(m, func(l *layer) {
			l.altCorrT = make([]float32, len(l.altCorrT))
		})
	}},
	{"no LAuReL", func(m *Model) func() {
		return eachLayer(m, func(l *layer) { l.laurelPost = make([]float32, len(l.laurelPost)) })
	}},
	{"no gaussian top-k", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.NSparse = 0 })
	}},
	{"the gaussian on every block", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.NSparse = c.NLayer })
	}},
	{"no correction scale", func(m *Model) func() {
		return eachLayer(m, func(l *layer) {
			ones := make([]float32, len(l.altCorrScale))
			for i := range ones {
				ones[i] = 1
			}
			l.altCorrScale = ones
		})
	}},
}

// TestGemma3nMatchesTransformers holds each fixture to transformers' own
// Gemma3nForCausalLM on decode, prefill at every length and the batched path,
// and the container's tokenizer to Gemma 3n's tokenizer.json.
func TestGemma3nMatchesTransformers(t *testing.T) {
	for _, fx := range gemma3nFixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openGemma3n(t, fx.name)
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

// TestGemma3nFeaturesAreLoadBearing runs each violation through decode.
func TestGemma3nFeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range gemma3nFixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openGemma3n(t, fx.name)
			defer m.Close()
			for _, v := range gemma3nViolations {
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

// gemma3nDevViolations take a piece out while only the device session runs.
var gemma3nDevViolations = []c6Violation{gemma3nViolations[0], gemma3nViolations[2], gemma3nViolations[5],
	gemma3nViolations[4]}

// TestGemma3nOnEveryDevice runs each fixture with every block and the head on
// each device present against the host: the streams stream-major in the
// device's residual, the prediction, LAuReL, the gaussian top-k and the
// correction in the session (tier/altup.go), on decode and a prefill chunk.
func TestGemma3nOnEveryDevice(t *testing.T) {
	for _, fx := range gemma3nFixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openGemma3n(t, fx.name)
			defer m.Close()
			fixtureOnEveryDeviceWithin(t, m, g, gemma3nDevViolations, 1e-5)
		})
	}
}

// TestBatchSeamMovesCarryEveryRowAltUp is the seam-move gate on Gemma 3n:
// a batch's rows carry four residual streams across every move, stream-major
// in a chunk and in a device's residual alike, beside a KV-sharing pair whose
// source and reader must move together.
func TestBatchSeamMovesCarryEveryRowAltUp(t *testing.T) {
	defer fusedRunsCheck(t, false)()
	p, ok := existingModel(testmodels.Path("synth-gemma3n.gguf"))
	if !ok {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path("synth-gemma3n.gguf")+
			" (set JITLLM_MODELS to the model directory) -- run "+gemma3nGoldScript+" (RULE 11)")
	}
	batchSeamMoves(t, p, 1e-3, true)
}

// Gemma 3n runs the five principles' relocation and paging gates
// (principleFixtures) and the windowed-layer gate: its sliding layers' pages
// are a window's.
func init() {
	principleFixtures = append(principleFixtures, "synth-gemma3n.gguf")
	windowFixtures = append(windowFixtures, "synth-gemma3n.gguf")
}

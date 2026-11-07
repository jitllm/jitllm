package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestMixedGeometriesTakeTheirOwnScale runs a model whose blocks have heads
// of two widths (synth-gemma4-wide) under a derived attention scale, which
// was refused: each block's attention scales by one over the root of its
// own head width, every block's graph states so, the scale moves the
// answer (the stated one is not the derived one), and the CPU and a card
// holding every block agree as closely as under the stated scale.
func TestMixedGeometriesTakeTheirOwnScale(t *testing.T) {
	const gen = 6
	name := "synth-gemma4-wide.gguf"
	src, ok := existingModel(testmodels.Path(name))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s", testmodels.Path(name))
	}
	path := jlmOf(t, src)
	run := func(derived bool, g *gpu.GPU) [][]float32 {
		m, err := model.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		c := m.Cfg
		if !c.MixedGeom() {
			t.Fatalf("%s states one geometry", name)
		}
		if derived {
			m.Container.Config().AttnScale = 0
			c.AttnScale = 0
			m.Front.Set = nil
			widths, attends := map[int]bool{}, 0
			for li, bg := range m.FrontGraphs().Blocks {
				if bg == nil {
					continue
				}
				want := 1 / math.Sqrt(float64(c.HeadDimAt(li)))
				for _, op := range bg.Ops {
					if op.Kind != graph.OpAttend {
						continue
					}
					attends++
					if math.Abs(op.Attr.Scale-want) > 1e-12 {
						t.Fatalf("block %d (head %d) attends at scale %v, want %v", li, c.HeadDimAt(li), op.Attr.Scale, want)
					}
				}
				widths[c.HeadDimAt(li)] = true
			}
			if len(widths) < 2 || attends < c.NLayer {
				t.Fatalf("the blocks' graphs cover %d head widths and %d attentions of %d blocks", len(widths), attends, c.NLayer)
			}
		}
		ids := m.Vocab.Encode("The capital of France is", true)
		s := testInstance(t, m, len(ids)+gen+1, WithKVF16(false))
		defer s.Close()
		if g != nil {
			if err := s.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if s.GPULayers() != c.NLayer {
				t.Fatalf("the card took %d of %d blocks: %s", s.GPULayers(), c.NLayer, g.Err())
			}
		}
		lg, err := s.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		out := [][]float32{append([]float32(nil), lg...)}
		for range gen {
			if lg, err = s.Forward(Greedy(lg)); err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	worst := func(a, b [][]float32) float64 {
		w := 0.0
		for i := range a {
			w = max(w, logitNMSE(a[i], b[i]))
		}
		return w
	}
	stated, derived := run(false, nil), run(true, nil)
	if worst(derived, stated) == 0 {
		t.Fatal("the derived scale's logits are the stated scale's: the gate cannot see the scale")
	}
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || g == nil {
				noDevice(t, spec, err)
			}
			defer g.Close()
			band := worst(run(false, g), stated)
			got := worst(run(true, g), derived)
			t.Logf("derived: %.3e from the CPU's, stated %.3e", got, band)
			if got > 10*max(band, 1e-9) {
				t.Fatalf("the derived scale's card parts from its CPU by %.3e, the stated scale's by %.3e", got, band)
			}
		})
	}
}

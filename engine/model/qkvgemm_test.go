package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// qkvGemmPrefill prefills ids on one fresh CUDA tier twice, the q/k/v grid
// off (three GEMMs) and on (GemmVoltaSegments), the switch thrown between
// the arms on the same tier, then decodes toks after each; it returns both
// arms' logits (the prompt's, then each step's) and the on arm's stats. With
// unsplit the three GEMMs are held to one k each (BatchShape.Split 1), which
// is the grid's arithmetic. layers < 0 places every block.
func qkvGemmPrefill(t *testing.T, m *Model, ids, toks []int32, unsplit, noGraph bool, layers int, extra ...tier.Option) (off, on [][]float32, st tier.Stats, placed int) {
	t.Helper()
	opts := []tier.Option{tier.WithDevices("cuda"), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(c *tier.Config) { c.NoGraph = noGraph })}
	if unsplit {
		opts = append(opts, tier.WithBatchShape(tier.BatchShape{Split: 1}))
	}
	g, err := openTier(t, append(opts, extra...)...)
	if err != nil || g == nil {
		return nil, nil, tier.Stats{}, 0
	}
	defer g.Close()
	arm := func(grid bool) [][]float32 {
		g.QKVGemm = grid
		s := m.NewState(len(ids) + len(toks) + 4)
		defer s.Close()
		if err := s.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if layers >= 0 {
			s.SetGPULayers(layers)
		}
		placed = s.GPULayers()
		l, err := s.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		out := [][]float32{append([]float32(nil), l...)}
		for _, tok := range toks {
			if l, err = s.Forward(tok); err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), l...))
		}
		s.SetGPULayers(0)
		return out
	}
	off = arm(false)
	if n := g.Stats().QKVGemmRuns; n != 0 {
		t.Fatalf("the three-GEMM arm ran %d q/k/v grids", n)
	}
	on = arm(true)
	return off, on, g.Stats(), placed
}

// TestQKVGemmMatchesThreeGemms prefills a real dense model on CUDA with each
// block's q, k and v as one GemmVoltaSegments grid and as three GEMMs, and
// holds the prompt's and a few decoded steps' logits: bit for bit against
// three GEMMs held to one k each (the grid's arithmetic), and inside the
// step gate's order band (denseStepBand times orderControl) against the
// shipping three, whose k and v split k and reduce. Every block placed with
// and without the graph, and a half seam. Stats().QKVGemmRuns is the selection
// check; on a card with no binary16 GEMM nothing fuses, which fails here.
func TestQKVGemmMatchesThreeGemms(t *testing.T) {
	m := gateUp2Model(t, "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if m == nil {
		testmodels.Missing(t, "MODEL MISSING: Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	}
	defer m.Close()
	ids := make([]int32, 192)
	for i := range ids {
		ids[i] = int32(1 + (i*7919)%(m.Cfg.NVocab-1))
	}
	toks := []int32{100, 137, 174, 211}
	bound := 0.0
	for _, unsplit := range []bool{true, false} {
		for _, c := range []struct {
			seam    int
			noGraph bool
		}{{-1, false}, {-1, true}, {m.Cfg.NLayer / 2, false}} {
			off, on, st, placed := qkvGemmPrefill(t, m, ids, toks, unsplit, c.noGraph, c.seam)
			if off == nil {
				t.Skip("no CUDA device")
			}
			want := m.Cfg.NLayer
			if c.seam >= 0 {
				want = c.seam
			}
			if placed != want {
				t.Fatalf("seam %d: the device took %d blocks, want %d", c.seam, placed, want)
			}
			if st.QKVGemmRuns == 0 {
				t.Fatalf("seam %d: no block ran q/k/v as one grid (%s)", c.seam, st.LastErr)
			}
			nmse, diff := gateUp2NMSE(on, off)
			worst := gateUp2Worst(on, off)
			if unsplit && diff != 0 || math.IsNaN(worst) {
				t.Fatalf("unsplit seam %d graph %v: grid against three GEMMs: %d logits differ (NMSE %.3g)", c.seam, !c.noGraph, diff, nmse)
			}
			if !unsplit {
				if bound == 0 {
					bound = denseStepBand * orderControl(t, m, "cuda")
				}
				if worst > bound {
					t.Fatalf("seam %d graph %v: grid against the shipping three: worst row NMSE %.3g, bound %.3g", c.seam, !c.noGraph, worst, bound)
				}
			}
			t.Logf("unsplit %v seam %d graph %v: %d grids, worst row NMSE %.3g (bound %.3g), %d logits differ",
				unsplit, c.seam, !c.noGraph, st.QKVGemmRuns, worst, bound, diff)
		}
	}
}

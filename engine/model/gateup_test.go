package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// gateUpModels are the dense gated FFNs the fused gate/up gate decodes.
var gateUpModels = []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf"}

// gateUpDecode decodes steps teacher-forced tokens after a prompt on a fresh
// tier with Config.GateUpFuse set to fuse and returns every step's logits and
// the tier's stats. layers < 0 places every block.
//
// The tuner runs, because the fusion needs the in-group split it picks on a
// card (the fitted table's partial-writing split declines it), and every arm
// reads its answers from one cache directory: the first arm measures, the
// rest run the same splits, so the reduction orders are the same and the
// logits can be compared bit for bit.
func gateUpDecode(t *testing.T, m *Model, ids, toks []int32, fuse, noGraph bool, layers int, cache string, extra ...tier.Option) ([][]float32, tier.Stats, int) {
	t.Helper()
	g, err := tier.OpenWith(append([]tier.Option{tier.WithTuneCache(cache), tier.WithGateUpFuse(fuse),
		tier.WithConfig(func(c *tier.Config) { c.NoGraph = noGraph })}, extra...)...)
	if err != nil || g == nil {
		t.Skipf("no GPU: %v", err)
	}
	defer g.Close()
	s := m.NewState(len(ids) + len(toks) + 4)
	defer s.Close()
	if err := s.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if layers >= 0 {
		s.SetGPULayers(layers)
	}
	placed := s.GPULayers()
	l, err := s.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	var out [][]float32
	for _, tok := range toks {
		out = append(out, append([]float32(nil), l...))
		if l, err = s.Forward(tok); err != nil {
			t.Fatal(err)
		}
	}
	s.SetGPULayers(0)
	return out, g.Stats(), placed
}

// TestGateUpFuseMatchesThreeLaunches decodes real dense models with the FFN's
// gate and up as one launch and down forming act(gate)*up in its prologue
// (Config.GateUpFuse), and without, and holds the two to the same logits bit
// for bit: every row keeps its own reduction and the element and its rounding
// are QuantizeAct's in both. Every block placed with and without the captured
// graph, and a partial seam. Stats().GateUpFused is the selection check.
func TestGateUpFuseMatchesThreeLaunches(t *testing.T) {
	ran := 0
	for _, name := range gateUpModels {
		p := testmodels.Path(name)
		if _, err := os.Stat(p); err != nil {
			if _, err := os.Stat(strings.TrimSuffix(p, ".gguf") + ".jlm"); err != nil {
				t.Logf("MODEL MISSING: %s", name)
				continue
			}
		}
		t.Run(name, func(t *testing.T) {
			m, err := Open(jlmOf(t, p), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The river rose every spring until the bridges were islands, and", true)
			toks := make([]int32, 16)
			for i := range toks {
				toks[i] = int32(100 + 37*i)
			}
			cache := t.TempDir()
			for _, c := range []struct {
				seam    int
				noGraph bool
			}{{-1, false}, {-1, true}, {m.Cfg.NLayer / 2, false}} {
				off, so, _ := gateUpDecode(t, m, ids, toks, false, c.noGraph, c.seam, cache)
				on, st, placed := gateUpDecode(t, m, ids, toks, true, c.noGraph, c.seam, cache)
				want := m.Cfg.NLayer
				if c.seam >= 0 {
					want = c.seam
				}
				if placed != want {
					t.Fatalf("seam %d: the device took %d blocks, want %d", c.seam, placed, want)
				}
				if st.GateUpFused == 0 {
					t.Fatalf("seam %d: no FFN ran gate and up as one launch (%s; %s)", c.seam, st.GateUpWhy, st.LastErr)
				}
				if so.GateUpFused != 0 {
					t.Fatalf("seam %d: the three-launch arm fused %d FFNs", c.seam, so.GateUpFused)
				}
				for i := range on {
					for j := range on[i] {
						if math.Float32bits(on[i][j]) != math.Float32bits(off[i][j]) {
							t.Fatalf("seam %d graph %v step %d logit %d: fused %v, three launches %v",
								c.seam, !c.noGraph, i, j, on[i][j], off[i][j])
						}
					}
				}
				t.Logf("seam %d graph %v: %d fused FFNs, %d steps bit-identical", c.seam, !c.noGraph, st.GateUpFused, len(on))
			}
			ran++
		})
	}
	if ran == 0 {
		testmodels.Missing(t, "no model ran: this gate proved nothing (set JITLLM_MODELS)")
	}
}

package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// gateUp2Models are the dense gated FFNs the Gate2 gate decodes: gate and up
// one format in both.
var gateUp2Models = []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf"}

// gateUp2Decode decodes steps teacher-forced tokens after a prompt on a fresh
// tier with Config.GateUp2 set to on and returns every step's logits, the
// tier's stats and the blocks placed. layers < 0 places every block.
//
// The tuner runs, because Gate2 needs a final-row split (in-group, or 1),
// and every arm reads its answers from one cache directory: the first arm
// measures, the rest run the same splits, so the reduction orders are the
// same and the logits can be compared bit for bit.
func gateUp2Decode(t *testing.T, m *Model, ids, toks []int32, on, noGraph bool, layers int, cache string) ([][]float32, tier.Stats, int) {
	t.Helper()
	g, err := tier.OpenWith(tier.WithTuneCache(cache), tier.WithGateUp2(on),
		tier.WithConfig(func(c *tier.Config) { c.NoGraph = noGraph }))
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

// gateUp2Model opens name, or reports it missing.
func gateUp2Model(t *testing.T, name string) *Model {
	t.Helper()
	p := testmodels.Path(name)
	if _, err := os.Stat(p); err != nil {
		if _, err := os.Stat(strings.TrimSuffix(p, ".gguf") + ".jlm"); err != nil {
			return nil
		}
	}
	m, err := Open(jlmOf(t, p), noTune)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// TestGateUp2MatchesTwoLaunches decodes real dense models with the FFN's
// gate and up as one Gate2 launch (Config.GateUp2) and without, and holds
// the two to the same logits bit for bit: each row is summed in the order
// the gate's own launch sums it, and act(gate)*up is the same multiply on
// the same two floats. Every block placed with and without the captured
// graph, and a partial seam. Stats().GateUp2Runs is the selection check.
func TestGateUp2MatchesTwoLaunches(t *testing.T) {
	ran := 0
	for _, name := range gateUp2Models {
		m := gateUp2Model(t, name)
		if m == nil {
			t.Logf("MODEL MISSING: %s", name)
			continue
		}
		t.Run(name, func(t *testing.T) {
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
				off, so, _ := gateUp2Decode(t, m, ids, toks, false, c.noGraph, c.seam, cache)
				on, st, placed := gateUp2Decode(t, m, ids, toks, true, c.noGraph, c.seam, cache)
				want := m.Cfg.NLayer
				if c.seam >= 0 {
					want = c.seam
				}
				if placed != want {
					t.Fatalf("seam %d: the device took %d blocks, want %d", c.seam, placed, want)
				}
				if st.GateUp2Runs == 0 {
					t.Fatalf("seam %d: no FFN ran gate and up as one Gate2 launch (%s)", c.seam, st.LastErr)
				}
				if so.GateUp2Runs != 0 {
					t.Fatalf("seam %d: the two-launch arm ran %d Gate2 FFNs", c.seam, so.GateUp2Runs)
				}
				for i := range on {
					for j := range on[i] {
						if math.Float32bits(on[i][j]) != math.Float32bits(off[i][j]) {
							t.Fatalf("seam %d graph %v step %d logit %d: Gate2 %v, two launches %v",
								c.seam, !c.noGraph, i, j, on[i][j], off[i][j])
						}
					}
				}
				t.Logf("seam %d graph %v: %d Gate2 FFNs, %d steps bit-identical", c.seam, !c.noGraph, st.GateUp2Runs, len(on))
			}
			ran++
		})
	}
	if ran == 0 {
		testmodels.Missing(t, "no model ran: this gate proved nothing (set JITLLM_MODELS)")
	}
}

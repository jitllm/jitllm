package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// openSynthLlama4Q8 is the Q8_0 twin of the fixture (Q8=1 scripts/llama4gold.py):
// every matrix quantized, and the golden taken from a transformers model whose
// weights were dequantized back from the same bytes.
func openSynthLlama4Q8(t *testing.T) (*Model, *synthLlama4) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "llama4", "synth-llama4-q8.json"))
	if err != nil {
		t.Fatal(err)
	}
	g := &synthLlama4{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	path, ok := existingModel(testmodels.Path("synth-llama4-q8.gguf"))
	if !ok {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path("synth-llama4-q8.gguf")+
			" (set JITLLM_MODELS to the model directory) -- run Q8=1 scripts/llama4gold.py (RULE 11)")
	}
	m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	return m, g
}

// TestSynthLlama4Q8MatchesTransformers is the quantized fixture against the
// reference. The weights are identical on both sides, so what remains is the
// engine's int8 activation quantization: hence 5e-3 rather than the F32
// fixture's 1e-9, as in TestSynthGPTOSSMatchesTransformers.
func TestSynthLlama4Q8MatchesTransformers(t *testing.T) {
	m, g := openSynthLlama4Q8(t)
	defer m.Close()
	worst, agree := 0.0, 0
	st := m.NewState(len(g.IDs) + 1)
	defer st.Close()
	for p, id := range g.IDs {
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		nmse := llama4Cmp(g.Pos[p].Head, lg)
		if math.IsNaN(nmse) || math.IsInf(nmse, 0) || !(nmse < 5e-3) {
			t.Errorf("pos %d: NMSE %.3e against transformers", p, nmse)
		}
		worst = math.Max(worst, nmse)
		if Greedy(lg) == g.Pos[p].Argmax {
			agree++
		}
	}
	if agree < len(g.IDs)-1 {
		t.Errorf("argmax agrees at %d of %d positions", agree, len(g.IDs))
	}
	t.Logf("Q8_0 fixture: worst NMSE %.3e, argmax %d of %d", worst, agree, len(g.IDs))
}

// TestSynthLlama4Q8BatchedPrefillOnDevice holds the device's BATCHED prefill on
// the quantized fixture -- the one path the F32 fixture cannot reach, because a
// float bank has no grouped mixture kernel -- against the host, and checks the
// grouped mixture actually ran (by counter). The input-weighted mixture reads
// each sorted column's weight through its (token, slot) pair; a wrong index is
// a finite, plausible, different model.
func TestSynthLlama4Q8BatchedPrefillOnDevice(t *testing.T) {
	m, g := openSynthLlama4Q8(t)
	defer m.Close()
	host := m.NewState(len(g.IDs) + 1)
	want, err := host.Prefill(g.IDs)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)
	host.Close()
	ran := 0
	for _, spec := range stepDevices() {
		gpu, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			st := m.NewState(len(g.IDs) + 1)
			defer st.Close()
			st.SetDeviceLayers(gpu, -1)
			if n := st.GPULayers(); n != m.Cfg.NLayer {
				t.Fatalf("%s took %d of %d blocks: %v %v", spec, n, m.Cfg.NLayer, st.DeviceDeclines(), gpu.Err())
			}
			before := gpu.Stats().GroupedMoE
			got, err := st.Prefill(g.IDs)
			if err != nil {
				t.Fatal(err)
			}
			if n := gpu.Stats().GroupedMoE - before; n == 0 {
				t.Fatalf("the grouped mixture never ran (%s), so this compared the per-row path", gpu.Err())
			}
			// The chunk must have stayed on the device: a demotion restarts it
			// on the host, whose answer matches exactly.
			if n, d := st.GPULayers(), st.DeviceDemotions(); n != m.Cfg.NLayer || d != 0 {
				t.Fatalf("the prefill fell to the host: %d blocks placed, %d demotions (%s)", n, d, gpu.Err())
			}
			nmse := llama4Cmp(f64s(want), got)
			if math.IsNaN(nmse) || !(nmse < 1e-2) || Greedy(got) != Greedy(want) {
				t.Fatalf("batched prefill on %s: NMSE %.3e, argmax %d against the host's %d",
					spec, nmse, Greedy(got), Greedy(want))
			}
			t.Logf("batched prefill on %s, grouped mixture ran: NMSE %.3e against the host", spec, nmse)
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

func f64s(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

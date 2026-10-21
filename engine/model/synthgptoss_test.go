package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestSynthGPTOSSMatchesTransformers holds the whole gpt-oss graph to
// transformers' GptOssForCausalLM, position by position, on a small model that
// carries every feature: sinks, a sliding window of 4 inside 12 positions, YaRN
// untruncated, a biased router, biased experts, swiglu-oai past its clamp and
// MXFP4 expert banks (scripts/gptossgold.py, which also writes the golden).
//
// The weights are identical on both sides (the banks were quantized to MXFP4
// and dequantized back into transformers), so what remains is the engine's int8
// activation quantization; the bounds are sized for that, allowing an argmax
// flip or two on a random 201k-entry head. It is also the arm64 gate for
// gpt-oss, since the real model does not fit a small arm64 host.
func TestSynthGPTOSSMatchesTransformers(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "gptoss", "synth-gptoss.json"))
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		IDs []int32 `json:"ids"`
		Pos []struct {
			Argmax int32
			Head   []float32
		} `json:"pos"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	path, ok := existingModel(testmodels.Path("synth-gptoss.gguf"))
	if !ok {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path("synth-gptoss.gguf")+" (set JITLLM_MODELS to the model directory) -- run scripts/gptossgold.py (RULE 11)")
	}
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	agree, total := 0, 0
	cmp := func(what string, p int, got []float32) {
		t.Helper()
		want := g.Pos[p]
		var num, den float64
		for i, w := range want.Head {
			d := float64(got[i] - w)
			num, den = num+d*d, den+float64(w)*float64(w)
		}
		nmse := num / den
		if math.IsNaN(nmse) || den == 0 {
			t.Fatalf("%s pos %d: degenerate comparison", what, p)
		}
		am := Greedy(got)
		total++
		if am == want.Argmax {
			agree++
		}
		if !(nmse < 5e-3) {
			t.Errorf("%s pos %d: NMSE %.2e against transformers", what, p, nmse)
			return
		}
		t.Logf("%s pos %2d: NMSE %.2e, argmax %d (transformers %d)", what, p, nmse, am, want.Argmax)
	}
	st := m.NewState(len(g.IDs) + 1)
	defer st.Close()
	for p, id := range g.IDs {
		logits, err := st.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		cmp("forward", p, logits)
	}
	pf := m.NewState(len(g.IDs) + 1)
	defer pf.Close()
	logits, err := pf.Prefill(g.IDs)
	if err != nil {
		t.Fatal(err)
	}
	cmp("prefill", len(g.IDs)-1, logits)
	if agree < total-2 {
		t.Errorf("argmax agrees at %d of %d positions", agree, total)
	}
}

// TestSynthGPTOSSOnEveryDevice runs the model with every block and the head on
// each device present -- cuda, vulkan, metal -- against the host, position by
// position. Sinks, the sliding window, biased q/k/v/o, the biased router and
// experts, swiglu-oai and the MXFP4 banks all run as device kernels, and a block
// the device declines, or a session that falls back, fails the gate.
//
// The routing is pinned because a random router has near-ties that the
// device's int8 activations can flip: a large router bias fixes a different
// pair of experts per layer (so the bank is indexed past expert 0) while the
// router still weights them. The remaining arithmetic is bounded at 1e-2;
// ignoring sinks, router bias or expert biases lands far above.
//
// Host exactness against transformers is TestSynthGPTOSSMatchesTransformers.
func TestSynthGPTOSSOnEveryDevice(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "gptoss", "synth-gptoss.json"))
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		IDs []int32 `json:"ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	path, ok := existingModel(testmodels.Path("synth-gptoss.gguf"))
	if !ok {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path("synth-gptoss.gguf")+" (set JITLLM_MODELS to the model directory) -- run scripts/gptossgold.py (RULE 11)")
	}
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.layers[0].sinks == nil || m.layers[0].routerB == nil || m.layers[0].expDownB == nil {
		t.Fatal("synth-gptoss carries no sinks or mixture biases: this gate would test nothing")
	}
	for li := range m.layers {
		l := &m.layers[li]
		pin := append([]float32(nil), l.routerB...)
		for j := 0; j < m.Cfg.NExpertUsed; j++ {
			pin[(li*m.Cfg.NExpertUsed+j)%m.Cfg.NExpert] += 100
		}
		l.routerB = pin
		// Scale the down and up biases so a wrong bias is visible above the
		// int8 noise (swiglu-oai clamps the up side at 7). Both tiers run the
		// scaled values.
		db := append([]float32(nil), l.expDownB...)
		for i := range db {
			db[i] *= 50
		}
		ub := append([]float32(nil), l.expUpB...)
		for i := range ub {
			ub[i] *= 10
		}
		l.expDownB, l.expUpB = db, ub
	}
	run := func(st *State) [][]float32 {
		var out [][]float32
		for _, id := range g.IDs {
			lg, err := st.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	host := m.NewState(len(g.IDs) + 1)
	want := run(host)
	host.Close()
	ran := 0
	for _, spec := range stepDevices() {
		gpu, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
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
				t.Fatalf("%s took %d of %d blocks: %v", spec, n, m.Cfg.NLayer, gpu.Err())
			}
			got := run(st)
			if n := st.GPULayers(); n != m.Cfg.NLayer {
				t.Fatalf("the session fell to the host (%d blocks left): %v", n, gpu.Err())
			}
			worst := 0.0
			for p := range want {
				var num, den float64
				for i := range want[p] {
					d := float64(got[p][i] - want[p][i])
					num, den = num+d*d, den+float64(want[p][i])*float64(want[p][i])
				}
				nmse := num / den
				if math.IsNaN(nmse) || den == 0 || !(nmse < 1e-2) {
					t.Fatalf("pos %d: logit NMSE %.3e against the host", p, nmse)
				}
				worst = math.Max(worst, nmse)
			}
			t.Logf("%d positions, %d blocks and the head on %s, worst logit NMSE %.3e",
				len(want), m.Cfg.NLayer, spec, worst)
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

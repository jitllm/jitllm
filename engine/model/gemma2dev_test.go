package model

import (
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestGemma2SoftcapOnDevice runs gemma-2-2b with every block on the device
// against the host over a short prompt, comparing the logits at every position.
//
// Position 0 proves nothing about a softcap: with one key the softmax is 1
// whatever the scores are.
func TestGemma2SoftcapOnDevice(t *testing.T) {
	path := testmodels.Path("gemma-2-2b-it-Q4_K_M.gguf")
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS to the model directory)", path)
	}
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Cfg.AttnSoftcap == 0 {
		t.Fatal("gemma-2-2b carries no attention softcap: this gate would test nothing")
	}
	// At the model's own cap of 50 real scores sit where c*tanh(x/c) is nearly
	// x, so skipping the cap stays inside the band. Both tiers run the same
	// config, so capping at 2, which bends nearly every score, is still
	// like-for-like: working ~5e-3, the cap skipped ~6e-2, the bound between.
	m.Cfg.AttnSoftcap = 2
	// The block graph states the cap its attend op carries, and the device
	// plan is lowered from it: rebuilt, the graph states the cap of 2.
	if err := m.buildGraphs(); err != nil {
		t.Fatal(err)
	}
	ids := m.Vocab.Encode("The capital of France is Paris, and the capital of Germany is", true)
	run := func(st *State) [][]float32 {
		var out [][]float32
		for _, id := range ids {
			lg, err := st.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	host := m.NewState(64)
	want := run(host)
	host.Close()

	g, err := openTier(t, tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	dev := m.NewState(64)
	defer dev.Close()
	dev.SetDeviceLayers(g, -1)
	if n := dev.GPULayers(); n != m.Cfg.NLayer {
		t.Fatalf("the device took %d of %d blocks: %v", n, m.Cfg.NLayer, g.Err())
	}
	got := run(dev)
	if n := dev.GPULayers(); n != m.Cfg.NLayer {
		t.Fatalf("the device session fell to the host (%d blocks left): %v", n, g.Err())
	}
	worst := 0.0
	for p := range want {
		var num, den float64
		for i := range want[p] {
			d := float64(got[p][i] - want[p][i])
			num += d * d
			den += float64(want[p][i]) * float64(want[p][i])
		}
		nmse := num / den
		worst = math.Max(worst, nmse)
		if math.IsNaN(nmse) || nmse > 2e-2 {
			t.Fatalf("pos %d: logit NMSE %.3e with every block on the device", p, nmse)
		}
	}
	t.Logf("%d positions, %d blocks on the device, worst logit NMSE %.3e", len(want), m.Cfg.NLayer, worst)
}

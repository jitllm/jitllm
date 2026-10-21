package model

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestDeviceMemoryHoldsSteadyAcrossMoves walks a decoding model's seam home and
// back over the card, cycle after cycle -- the weights released and re-placed,
// the history migrated both ways -- and reads the card's free memory from the
// DRIVER after each cycle. The tier's own ledger covers what it charges; the
// driver sees every buffer, charged or not. After the first cycle has built
// whatever scratch the moves need, the figure must come back: a leak lowers it
// every cycle, so two cycles in a row below the first fail. One cycle below is
// not enough, because the figure is the whole card's and another process
// allocating on it moves it too.
func TestDeviceMemoryHoldsSteadyAcrossMoves(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS) -- this gate proved nothing", err)
	}
	m, err := Open(jlmOf(t, p), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	g, err := openTier(t, tier.WithDevices(testdevices.Like(t, "cuda:0")), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	const cycles, steps = 6, 6
	st := m.NewState(16 + cycles*steps + 1)
	defer st.Close()
	st.SetDevice(g)
	full := st.GPULayers()
	if full != m.Cfg.NLayer {
		t.Skipf("CARD TOO SMALL: %d of %d blocks -- this gate proved nothing", full, m.Cfg.NLayer)
	}
	lg, err := st.Prefill(m.Vocab.Encode("The capital of France is", true))
	if err != nil {
		t.Fatal(err)
	}
	var steady uint64
	below := 0
	for c := 0; c < cycles; c++ {
		for i, n := range []int{full / 2, 0, full / 3, full} {
			if got := st.SetGPULayers(n); got != n {
				t.Fatalf("cycle %d: asked for %d blocks, got %d", c, n, got)
			}
			if i < steps {
				if lg, err = st.Forward(Greedy(lg)); err != nil {
					t.Fatal(err)
				}
			}
		}
		free := g.DeviceFree()[0]
		if free == 0 {
			t.Skip("the driver reports no free memory figure -- this gate proved nothing")
		}
		t.Logf("cycle %d: %d MiB free on the card", c, free>>20)
		switch {
		case c == 0:
			steady = free
		case free < steady:
			below++
			if below == 2 {
				t.Fatalf("cycles %d and %d: %d MiB free, after cycle 0 %d: %d MiB more held for the same "+
					"placement, two cycles running (the figure is the whole card's: check nothing else "+
					"was using it)", c-1, c, free>>20, steady>>20, (steady-free)>>20)
			}
		default:
			below = 0
		}
	}
}

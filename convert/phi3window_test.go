package convert

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// phi3 slides every layer, and a converted phi3 must say so (llama.cpp applies
// phi3.attention.sliding_window to every layer). Phi-3.5's 262,144 window over
// a 131,072 context cannot bite, but it is the file's statement and is carried.
func TestPhi3SlidesEveryLayer(t *testing.T) {
	p := testmodels.Path("Phi-3.5-mini-instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	c := ggufSource(t, p).Config
	if c.SWAWindow != 262144 {
		t.Errorf("SWAWindow %d, the file says 262144", c.SWAWindow)
	}
	// Config.SWA's own rule, il%P < P-1, must hold for every layer.
	for il := uint32(0); il < c.NLayer; il++ {
		if c.SWAPeriod == 0 || il%c.SWAPeriod >= c.SWAPeriod-1 {
			t.Fatalf("layer %d is not local under period %d", il, c.SWAPeriod)
		}
	}
}

// phi-4 has no window at all, and the absent key must convert as that.
//
// Its HF config says sliding_window: null and llama.cpp disables phi3's SWA
// for it, so the file carries no phi3.attention.sliding_window.
func TestPhi4AttendsFully(t *testing.T) {
	p := testmodels.Path("phi-4-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	c := ggufSource(t, p).Config
	if c.SWAWindow != 0 || c.SWAPeriod != 0 {
		t.Errorf("SWAWindow %d period %d, phi-4 attends every position", c.SWAWindow, c.SWAPeriod)
	}
}

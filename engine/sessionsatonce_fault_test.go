//go:build jitllmfault

package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestSessionsEvictAtOnceDiscriminates runs TestSessionsEvictAtOnce's arm in
// one call with each streamed pass's upload skipped, so a pass reads whatever
// the free ids held, and demands the arm fall outside its bound: the band
// its control widens it to must still see a page read stale.
func TestSessionsEvictAtOnceDiscriminates(t *testing.T) {
	m, err := model.Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, spec := range testdevices.Specs("cuda:0", "vulkan:0", "metal") {
		t.Run(spec, func(t *testing.T) {
			var f gpu.Faults
			f.SetPagedFault("stream")
			a := evictAtOnce(t, m, atOncePrompts(m), spec, true, true, gpu.WithFaults(&f))
			t.Logf("violation: %d page(s) home, %d streamed passes, worst NMSE %.3e against each alone, bound %.3e (%.3e with no budget in one call)",
				a.ev, a.passes, a.worst, a.bound(), a.ctl)
			if a.ev == 0 || a.passes == 0 {
				t.Fatalf("%d evictions and %d streamed passes: the violation was never exercised", a.ev, a.passes)
			}
			if a.held() {
				t.Fatalf("the gate passed a stream that uploaded nothing (NMSE %.3e, bound %.3e)", a.worst, a.bound())
			}
		})
	}
}

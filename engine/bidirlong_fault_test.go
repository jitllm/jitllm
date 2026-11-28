//go:build jitllmfault && (amd64 || arm64)

package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// TestLongBidirRunCutIsSeen is TestLongBidirRunMatchesTheCPU's violation:
// the run's second phase reads only the keys of its own submission's rows
// (the "runcut" fault), the run cut at the submission: in float forms the
// card's logits fail the clean gate's bound and stand more than ten times
// as far from the CPU's as the clean run's on the same card.
func TestLongBidirRunCutIsSeen(t *testing.T) {
	m := longBidirModel(t)
	defer m.Close()
	a := newLongBidirArm(t, m)
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			bands := formBands(t, spec, func(g *gpu.GPU) float64 {
				v, _ := nmseOf(longBidirLogits(t, m, g, m.Cfg.NLayer, longBidirSpans(t, m, false)), a.cpuCau)
				return v
			})
			for _, f := range windowForms {
				band := bands[f.name]
				var faults gpu.Faults
				g := openCentredCard(t, spec, 0, append(f.opts(), gpu.WithFaults(&faults))...)
				clean, _ := nmseOf(longBidirLogits(t, m, g, m.Cfg.NLayer, longBidirSpans(t, m, true)), a.cpuRun)
				faults.SetRowsFault("runcut")
				bad, _ := nmseOf(longBidirLogits(t, m, g, m.Cfg.NLayer, longBidirSpans(t, m, true)), a.cpuRun)
				faults.SetRowsFault("")
				g.Close()
				seen := bad > longBidirBound*max(band, 1e-6) && bad > 10*clean
				t.Logf("violation on %s %s, the run cut at the submission: NMSE %.3e, %.1fx the causal band %.3e, %.0fx the clean run's %.3e; seen %v",
					spec, f.name, bad, bad/max(band, 1e-6), band, bad/max(clean, 1e-9), clean, seen)
				// As shipped the cut can fall inside the int8 class's band
				// (CUDA's 1.2e-2); the float forms are what see it.
				if f.float && !seen {
					t.Fatalf("a run cut at the submission reads NMSE %.3e on %s %s (bound %.3e, clean %.3e): the gate cannot see the cut",
						bad, spec, f.name, longBidirBound*max(band, 1e-6), clean)
				}
			}
		})
	}
}

//go:build jitllmfault && (amd64 || arm64)

package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// TestLongBidirRunCutIsSeen is TestLongBidirRunMatchesTheCPU's violation:
// the run's second phase reads only the keys of its own submission's rows
// (the "runcut" fault), the run cut at the submission: the card's logits
// fail the clean gate's bound and stand more than ten times as far from
// the CPU's as the clean run's on the same card.
func TestLongBidirRunCutIsSeen(t *testing.T) {
	m := longBidirModel(t)
	defer m.Close()
	a := newLongBidirArm(t, m)
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			var f gpu.Faults
			g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff), gpu.WithFaults(&f)}, testTierOpts(t)...)...)
			if err != nil || g == nil {
				noDevice(t, spec, err)
			}
			defer g.Close()
			band, _ := nmseOf(longBidirLogits(t, m, g, m.Cfg.NLayer, longBidirSpans(t, m, false)), a.cpuCau)
			clean, _ := nmseOf(longBidirLogits(t, m, g, m.Cfg.NLayer, longBidirSpans(t, m, true)), a.cpuRun)
			f.SetRowsFault("runcut")
			bad, _ := nmseOf(longBidirLogits(t, m, g, m.Cfg.NLayer, longBidirSpans(t, m, true)), a.cpuRun)
			f.SetRowsFault("")
			t.Logf("violation on %s, the run cut at the submission: NMSE %.3e, %.1fx the causal band %.3e, %.0fx the clean run's %.3e",
				spec, bad, bad/max(band, 1e-6), band, bad/max(clean, 1e-9), clean)
			if !(bad > longBidirBound*max(band, 1e-6)) || !(bad > 10*clean) {
				t.Fatalf("a run cut at the submission reads NMSE %.3e on %s (bound %.3e, clean %.3e): the gate cannot see the cut",
					bad, spec, longBidirBound*max(band, 1e-6), clean)
			}
		})
	}
}

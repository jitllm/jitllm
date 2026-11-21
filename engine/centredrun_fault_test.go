//go:build jitllmfault && (amd64 || arm64)

package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// TestCentredRunHaloIsSeen is TestCentredRunMatchesTheCPU's violation: the
// pieces of a centred block read no rows past their edges (the "nohalo"
// fault), each piece's convolution cut at it, at the reserved width as
// placed and narrowed: the card's logits fail the clean gate's bound and
// stand more than five times as far from the CPU's as the clean run's (on
// Vulkan, whose band is forty times CUDA's, the cut read seven times).
func TestCentredRunHaloIsSeen(t *testing.T) {
	m := centredModel(t)
	defer m.Close()
	a := newCentredArm(t, m)
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			g := openCentredCard(t, spec, 0)
			band, _ := nmseOf(centredLogits(t, m, g, m.Cfg.NLayer, a.short), a.cpuShort)
			g.Close()
			for _, rows := range []int{0, 64} {
				var f gpu.Faults
				g := openCentredCard(t, spec, rows, gpu.WithFaults(&f))
				clean, _ := nmseOf(centredLogits(t, m, g, m.Cfg.NLayer, a.long), a.cpuLong)
				f.SetRowsFault("nohalo")
				bad, _ := nmseOf(centredLogits(t, m, g, m.Cfg.NLayer, a.long), a.cpuLong)
				f.SetRowsFault("")
				g.Close()
				t.Logf("violation on %s, reserved %d, no halo: NMSE %.3e, %.1fx the band %.3e, %.0fx the clean run's %.3e",
					spec, rows, bad, bad/max(band, 1e-6), band, bad/max(clean, 1e-9), clean)
				if !(bad > centredBound*max(band, 1e-6)) || !(bad > 5*clean) {
					t.Fatalf("pieces with no halo read NMSE %.3e on %s, reserved %d (bound %.3e, clean %.3e): the gate cannot see the halo",
						bad, spec, rows, centredBound*max(band, 1e-6), clean)
				}
			}
		})
	}
}

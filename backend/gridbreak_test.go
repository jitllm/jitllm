//go:build jitllmtest

package backend_test

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestGridBarrierProbeSeesAnEarlyCrossing is the grid barrier gate's violation
// (RULE 10): with every waiter let through without waiting
// (backend.Opts.BreakGridBarrier; the barrier still counts, so the launch ends),
// kernels.GridProbe must find wrong words. If it does not, it cannot see an
// early crossing and TestGridBarrierHoldsEveryWorkgroup certifies nothing.
func TestGridBarrierProbeSeesAnEarlyCrossing(t *testing.T) {
	ran := 0
	broken := func() []backend.Device {
		return backend.OpenWith(backend.Opts{Admit: testdevices.Admit(), BreakGridBarrier: true})
	}
	eachGridDeviceWith(t, broken, func(d backend.Device, width, limit int) {
		for _, groups := range dedupe(2, limit/2, limit) {
			t.Run(fmt.Sprintf("%s/w%d/g%d", d.API(), width, groups), func(t *testing.T) {
				r := newGridRig(t, d, groups, width)
				r.poison()
				if err := r.k.Launch(groups, width, r.bufs...); err != nil {
					t.Fatal(err)
				}
				bad, first := r.wrong()
				if bad == 0 {
					t.Fatalf("a grid barrier that waits for nothing passed the probe")
				}
				t.Logf("seen: %d words wrong; first %s", bad, first)
			})
			ran++
		}
	})
	if ran == 0 {
		t.Skip("no device runs a grid barrier")
	}
}

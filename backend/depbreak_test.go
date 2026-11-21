//go:build jitllmtest

package backend_test

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestDepCountersSeeABrokenWait is TestDepCountersOrderTheChain's violation
// (RULE 10): with every wait returning at once (backend.Opts.BreakDepWait), the
// chain must read words before they are written at the co-residency limit.
// If it does not, the gate's agreement says nothing about the counters.
func TestDepCountersSeeABrokenWait(t *testing.T) {
	ran := 0
	broken := func() []backend.Device {
		return backend.OpenWith(backend.Opts{Admit: testdevices.Admit(), BreakDepWait: true})
	}
	eachGridDeviceWith(t, broken, func(d backend.Device, width, limit int) {
		if d.API() != "ptx" {
			return
		}
		t.Run(fmt.Sprintf("%s/w%d/g%d", d.API(), width, limit), func(t *testing.T) {
			bad := depChain(t, d, width, limit)
			t.Logf("%d words wrong with every wait broken", bad)
			if bad == 0 {
				t.Fatal("every wait broken and the chain still agrees with the host")
			}
		})
		ran++
	})
	if ran == 0 {
		t.Skip("no device lowers the dependency counters")
	}
}

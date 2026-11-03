package gpu

import (
	"testing"

	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
)

// TestEveryPackedFormatReachesTheDevice sweeps quant.PackedTypes() -- the
// CANONICAL list -- rather than naming formats here.
//
// A hand-written copy of that list went stale once (a new format answered
// false, so a whole model's blocks were declined). quantOf forwards to
// layout.QuantOf, and this gate covers a format added to quant and to no
// layout at all. Against a violation it names the format.
func TestEveryPackedFormatReachesTheDevice(t *testing.T) {
	if len(quant.PackedTypes()) == 0 {
		t.Fatal("quant.PackedTypes() is empty; this gate would pass by sweeping nothing")
	}
	for _, q := range quant.PackedTypes() {
		got, ok := quantOf(q)
		if !ok {
			t.Errorf("%s is a packed format and the device holds no layout for it; "+
				"a model holding one declines every block that uses it", q)
			continue
		}
		want, _ := layout.QuantOf(q)
		if got != want {
			t.Errorf("%s maps to %v here and %v in the layout; the device has a second list again",
				q, got, want)
		}
	}
	t.Logf("%d packed formats, every one reaching the device", len(quant.PackedTypes()))
}

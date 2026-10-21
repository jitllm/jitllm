package tier

import (
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestEveryPackedFormatReachesTheDevice sweeps quant.PackedTypes -- the
// CANONICAL list -- rather than naming formats here.
//
// A hand-written copy of that list went stale once (a new format answered
// false, so a whole model's blocks were declined). quantOf forwards to
// kernels.QuantOf, and this gate covers a format added to quant and to no
// kernel table at all. Against a violation it names the format.
func TestEveryPackedFormatReachesTheDevice(t *testing.T) {
	if len(quant.PackedTypes) == 0 {
		t.Fatal("quant.PackedTypes is empty; this gate would pass by sweeping nothing")
	}
	declined := 0
	for _, q := range quant.PackedTypes {
		// A format the device DECLINES by name is not a stale list: the
		// declining is kernels.DeviceWhyNot's, one place, and it must say why.
		if kq, packed := kernels.QuantOf(q); packed && kernels.DeviceWhyNot(kq) != "" {
			if _, ok := quantOf(q); ok {
				t.Errorf("%s: the device kernels decline it (%s) and quantOf answers true",
					q, kernels.DeviceWhyNot(kq))
			}
			t.Logf("%s declined: %s", q, kernels.DeviceWhyNot(kq))
			declined++
			continue
		}
		got, ok := quantOf(q)
		if !ok {
			t.Errorf("%s is a packed format and the device tier has no kernel for it; "+
				"a model holding one declines every block that uses it", q)
			continue
		}
		want, _ := kernels.QuantOf(q)
		if got != want {
			t.Errorf("%s maps to %v here and %v in kernels; the tier has a second list again",
				q, got, want)
		}
	}
	t.Logf("%d packed formats, %d reaching a device kernel, %d declined by name",
		len(quant.PackedTypes), len(quant.PackedTypes)-declined, declined)
}

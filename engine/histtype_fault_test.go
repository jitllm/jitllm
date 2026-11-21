//go:build jitllmfault

package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// TestTwoHistoryTypesShareABlockDiscriminates runs TestTwoHistoryTypesShareABlock
// with the card's pool keys dropping the history type
// (gpu.Faults.SetPagedFault "histtype"): the second instance is seated in
// the first's pool layers and runs its forms. The gate must find it, or it
// cannot see two types sharing one pool.
func TestTwoHistoryTypesShareABlockDiscriminates(t *testing.T) {
	devs := stepDevices()
	if len(devs) == 0 {
		t.Fatal("no card listed (JITLLM_TEST_DEVICES): the gate holds two instances' histories on a card")
	}
	for _, dev := range devs {
		for _, kts := range twoTypePairs() {
			t.Run(dev+"/"+kts[0].String()+"+"+kts[1].String(), func(t *testing.T) {
				var fs gpu.Faults
				fs.SetPagedFault("histtype")
				defer fs.SetPagedFault("")
				cpu, alone, both := twoHistoryTypes(t, dev, kts, gpu.WithFaults(&fs))
				bad := twoTypesApart(t, dev, kts, cpu, alone, both)
				for _, b := range bad {
					t.Logf("found: %s", b)
				}
				if len(bad) == 0 {
					t.Fatal("two history types in one pool layer stood inside the gate: it cannot see a block's pool keyed without the type")
				}
			})
		}
	}
}

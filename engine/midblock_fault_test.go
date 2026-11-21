//go:build jitllmfault

package engine

import (
	"errors"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// TestAMidBlockFailureNeverAppliesARecurrenceTwice: a hybrid on the card
// whose token fails at a block after placed linear blocks advanced their
// recurrence (gpu.Faults.SetBlockFault) refuses the host restart
// (errRecurRetry): the host would apply those blocks a second time. One that
// fails at its first block, before anything advanced, restarts the token on
// the CPU device and answers as the host does. The fault reached its block
// in both arms, or the gate saw nothing.
//
//	go test -tags jitllmfault ./engine -run TestAMidBlockFailureNeverAppliesARecurrenceTwice
func TestAMidBlockFailureNeverAppliesARecurrenceTwice(t *testing.T) {
	m, openOpts := hybridModelOpt(t, hyOpt{layers: 8})
	defer m.Close()
	ids := []int32{1, 2, 3, 4, 5}
	host := testInstance(t, m, 16, openOpts...)
	defer host.Close()
	var want []float32
	for _, id := range ids {
		l, err := host.Forward(id)
		if err != nil {
			t.Fatalf("host: %v", err)
		}
		want = append(want[:0], l...)
	}
	// The first linear block's index and the first block past a linear one.
	first, past := -1, -1
	for li := range m.Cfg.NLayer {
		if m.Cfg.LayerKind(li).Recurrent() && first < 0 {
			first = li
		}
		if first >= 0 && li > first && past < 0 {
			past = li
		}
	}
	if first != 0 || past < 0 {
		t.Fatalf("the fixture's first linear block is %d, the block past it %d: this gate wants block 0 linear", first, past)
	}
	for _, c := range []struct {
		name     string
		block    int
		advanced bool
	}{{"at the first block", 0, false}, {"past a linear block", past, true}} {
		t.Run(c.name, func(t *testing.T) {
			var f gpu.Faults
			g, err := openGPU(t, gpu.WithDevices("gpu:0"), gpu.WithDeviceTune(gpu.TuneOff), gpu.WithFaults(&f))
			if err != nil || g == nil {
				noDevice(t, "device", err)
			}
			defer g.Close()
			dev := testInstance(t, m, 16, openOpts...)
			defer dev.Close()
			dev.SetDeviceLayers(g, m.Cfg.NLayer)
			if n := dev.GPULayers(); n != m.Cfg.NLayer {
				t.Fatalf("the card took %d of %d blocks: %s", n, m.Cfg.NLayer, g.Err())
			}
			for _, id := range ids[:len(ids)-1] {
				if _, err := dev.Forward(id); err != nil {
					t.Fatalf("on the card: %v", err)
				}
			}
			f.SetBlockFault(c.block, 1)
			got, err := dev.Forward(ids[len(ids)-1])
			if f.BlockFaultReached() == 0 {
				t.Fatalf("no submission reached block %d: the fault never ran", c.block)
			}
			if c.advanced {
				var rr errRecurRetry
				if !errors.As(err, &rr) {
					t.Fatalf("a token failing at block %d after linear blocks advanced answered (err %v): the host "+
						"restart applied their recurrence twice", c.block, err)
				}
				if !dev.RecurrenceAdvanced() {
					t.Fatal("RecurrenceAdvanced is false after linear blocks stepped")
				}
				t.Logf("block %d: %v", c.block, err)
				return
			}
			if err != nil {
				t.Fatalf("a token failing at its first block, before anything advanced, did not restart on the host: %v", err)
			}
			if n := dev.GPULayers(); n != 0 {
				t.Fatalf("after the restart %d blocks are still on the card", n)
			}
			e, worst := nmseOf(got, want)
			t.Logf("restarted on the host: %.3e NMSE (worst %.3g) from the host's", e, worst)
			if !(e <= hostStepNMSE) {
				t.Fatalf("the restarted token is %.3e NMSE from the host's (bound %.0e)", e, hostStepNMSE)
			}
		})
	}
}

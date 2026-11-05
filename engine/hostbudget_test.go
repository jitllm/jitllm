package engine

import (
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/nn"
)

// hostHungryDevice is a device whose memory is the host's, as an integrated GPU
// or Apple Silicon has. It takes no blocks: the point is the budget, which a
// device that declines every block still holds.
type hostHungryDevice struct{ reserved uint64 }

func (d *hostHungryDevice) HostReserved() uint64 { return d.reserved }

func (d *hostHungryDevice) MatVec([]float32, quant.Type, []byte, []float32, int, int) bool {
	return false
}
func (d *hostHungryDevice) PrepLayer(int, *nn.LayerPlan, *nn.LayerWeights) bool { return false }
func (d *hostHungryDevice) Layers(int, int, int, int, []float32, []float32, []float32, *nn.Head) bool {
	return false
}
func (d *hostHungryDevice) PrepHead(*nn.Head) bool { return false }
func (d *hostHungryDevice) HeadResident() bool     { return false }
func (d *hostHungryDevice) MigrateKV(int, []float32, []float32, int, bool) bool {
	return false
}

// MigrateRec: this fake keeps no recurrent state, so nothing to move is success.
func (d *hostHungryDevice) MigrateRec(int, []float32, []float32, bool) bool { return true }
func (d *hostHungryDevice) ReserveKV(int) bool                              { return true }
func (d *hostHungryDevice) ReleaseLayers(int, int)                          {}
func (d *hostHungryDevice) Reserve(int, int) bool                           { return true }
func (d *hostHungryDevice) PrewarmLayer(int, *nn.LayerPlan, *nn.LayerWeights) bool {
	return false
}

// TestHostBudgetLosesWhatAnIntegratedDeviceHolds: a device on host memory must
// be subtracted from the host weight budget, or the host and the device spend
// the same bytes. Asserted through the public calls (SetMemBudget,
// SetDeviceLayers, HostBudget) in the order cmd/jitllm and an embedder make
// them: the budget is set before the device exists.
func TestHostBudgetLosesWhatAnIntegratedDeviceHolds(t *testing.T) {
	m, err := model.Open(jlmOf(t, models()["stories260K"]))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer m.Close()
	st := testInstance(t, m, 64)
	defer st.Close()

	const budget = 24 << 30
	const held = 6 << 30
	st.SetMemBudget(budget)
	if got := st.HostBudget(); got != budget {
		t.Fatalf("with no device attached the host budget is %d, want the %d it was given",
			got, uint64(budget))
	}

	// The budget was set first and the device arrives after, as run() does.
	st.SetDeviceLayers(&hostHungryDevice{reserved: held}, -1)
	got := st.HostBudget()
	t.Logf("budget %d, device on host memory holds %d, host keeps %d", uint64(budget), uint64(held), got)
	if got != budget-held {
		t.Fatalf("host budget %d after attaching a device that holds %d of the same memory; "+
			"want %d. The host and the device are spending the same bytes",
			got, uint64(held), uint64(budget-held))
	}

	// Detaching gives it back: a session that parks its accelerator must get
	// the whole budget back.
	st.SetDeviceLayers(nil, -1)
	if got := st.HostBudget(); got != budget {
		t.Fatalf("host budget %d after detaching the device, want the full %d back",
			got, uint64(budget))
	}

	// A device with no claim on host memory (every discrete card) leaves the
	// budget as it was.
	st.SetDeviceLayers(&hostHungryDevice{reserved: 0}, -1)
	if got := st.HostBudget(); got != budget {
		t.Fatalf("a device reserving nothing moved the host budget to %d from %d", got, uint64(budget))
	}
}

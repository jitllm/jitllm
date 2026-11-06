package engine

import (
	"testing"

	"github.com/jitllm/jitllm/model"
)

// ownerFake is a device that counts the instances holding a model on it and
// the times it gave the model back.
type ownerFake struct {
	unifiedFake
	holds, gave int
}

func (d *ownerFake) HoldModel(uint64) { d.holds++ }

func (d *ownerFake) ReleaseModel(uint64) {
	if d.holds--; d.holds <= 0 {
		d.holds = 0
		d.gave++
	}
}

// TestADeviceKeepsAModelForItsInstances holds the design's shape (instance
// paging, step 5): a device keeps what a model placed while an instance
// holds it there, the last instance's release gives it back -- never one
// instance's Close while another still runs the model -- and the model's
// Close reaches no device. On one CPU device two instances share the model's
// frames; the first instance's Close leaves them, the second's lets them go.
//
// VIOLATION SIGNATURE. A model whose Close released its devices (the owner
// registry this replaced) gives the fake one more release after both
// instances have closed; an instance whose Close released the CPU device's
// hold outright makes the second instance read new frames.
func TestADeviceKeepsAModelForItsInstances(t *testing.T) {
	m, err := model.Open(jlmOf(t, models()["stories260K"]))
	if err != nil {
		t.Fatal(err)
	}
	cd, err := OpenCPU()
	if err != nil {
		t.Fatal(err)
	}
	defer cd.Close()
	dev := &ownerFake{unifiedFake: unifiedFake{held: map[int]bool{}}}
	a := testInstance(t, m, 32, WithCPU(cd))
	b := testInstance(t, m, 32, WithCPU(cd))
	for _, s := range []*Instance{a, b} {
		if err := s.SetDeviceLayers(dev, 0); err != nil {
			t.Fatal(err)
		}
	}
	if dev.holds != 2 || a.fr != b.fr {
		t.Fatalf("two instances attached: the device counts %d holds, and they share frames %v", dev.holds, a.fr == b.fr)
	}
	fr := a.fr
	a.Close()
	if dev.gave != 0 {
		t.Errorf("one of two instances closed and the device gave the model back %d time(s)", dev.gave)
	}
	if got, err := cd.Frames(m.Id, m.NewFrames); err != nil || got != fr {
		t.Errorf("one of two instances closed and the CPU device's frames for the model are new (%v)", err)
	}
	b.Close()
	if dev.gave != 1 {
		t.Errorf("both instances closed and the device gave the model back %d time(s), want once", dev.gave)
	}
	got, err := cd.Frames(m.Id, m.NewFrames)
	if err != nil {
		t.Fatal(err)
	}
	if got == fr {
		t.Error("both instances closed and the CPU device still holds the model's frames")
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if dev.gave != 1 {
		t.Errorf("the model's Close reached the device: it gave the model back %d time(s)", dev.gave)
	}
}

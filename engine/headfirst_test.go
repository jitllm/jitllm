package engine

import (
	"testing"

	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// roomFake is a device of room for cap units: each block takes one, the head
// head units.
type roomFake struct {
	unifiedFake
	cap, head int
	headOn    bool
}

func (d *roomFake) used() int {
	n := len(d.held)
	if d.headOn {
		n += d.head
	}
	return n
}

func (d *roomFake) PrepLayer(li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	if d.held[li] {
		return true
	}
	if d.used()+1 > d.cap {
		return false
	}
	return d.unifiedFake.PrepLayer(li, p, w)
}

func (d *roomFake) PrepHead(*nn.Head) bool {
	if d.headOn || d.used()+d.head <= d.cap {
		d.headOn = true
		return true
	}
	return false
}

func (d *roomFake) HeadResident() bool { return d.headOn }

// TestTheHeadBuysItsRoomOnAFullDevice: a device with room for every block and
// not the output projection beside them gives the projection the room of the
// last blocks whose reads it is worth -- it is read every token, a block once
// -- as a partial device does, and keeps every block when the projection is
// worth less than the block it would displace.
func TestTheHeadBuysItsRoomOnAFullDevice(t *testing.T) {
	m, err := model.Open(jlmOf(t, models()["stories15M"]))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	n := m.Cfg.NLayer
	head, block := m.HeadReadBytes(), m.BlockReadBytes(n-1)
	if head <= block+2*seamCrossBytes {
		t.Fatalf("the projection reads %d bytes and the last block %d: this model cannot show the trade", head, block)
	}
	for _, c := range []struct {
		name       string
		headUnits  int
		wantBlocks int
		wantHead   bool
	}{
		{"the head worth a block", 1, n - 1, true},
		{"the head worth no block it would need", n, n, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			st := testInstance(t, m, 64)
			defer st.Close()
			dev := &roomFake{unifiedFake: unifiedFake{held: map[int]bool{}}, cap: n, head: c.headUnits}
			if err := st.SetDeviceLayers(dev, -1); err != nil {
				t.Fatal(err)
			}
			if got := st.GPULayers(); got != c.wantBlocks || st.HeadOnDevice() != c.wantHead {
				t.Fatalf("%d of %d blocks and the head placed %v; want %d and %v", got, n, st.HeadOnDevice(), c.wantBlocks, c.wantHead)
			}
		})
	}
}

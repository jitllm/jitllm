package engine

import (
	"reflect"
	"testing"

	"github.com/jitllm/jitllm/common/session"
)

const (
	h = byte(session.PlaceCPU)
	d = byte(session.PlaceDevice)
)

func onZero(int) int { return 0 }
func paged(int) bool { return false }

// The strip is built from the runs, so a hole in the placement shows up: a
// declined block is skipped, not the end of the offer sequence.
func TestTheStripShowsAHoleInThePlacement(t *testing.T) {
	for _, c := range []struct {
		name string
		runs [][2]int
		want []byte
	}{
		{"a prefix, which still works", [][2]int{{0, 3}}, []byte{d, d, d, h, h, h}},
		{"a hole in the middle", [][2]int{{0, 2}, {4, 6}}, []byte{d, d, h, h, d, d}},
		{"nothing placed", nil, []byte{h, h, h, h, h, h}},
		{"a run that is not at the start", [][2]int{{3, 6}}, []byte{h, h, h, d, d, d}},
		{"a run past the end is clamped", [][2]int{{4, 99}}, []byte{h, h, h, h, d, d}},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := blockMap(6, c.runs, onZero, paged); !reflect.DeepEqual(got, c.want) {
				t.Errorf("blockMap(6, %v)\n got %v\nwant %v", c.runs, got, c.want)
			}
		})
	}
	if blockMap(0, [][2]int{{0, 3}}, onZero, paged) != nil {
		t.Error("a model with no blocks produced a strip")
	}
}

// Each byte carries the device that runs the block and the host pager's
// residency, independently: a device block can have its page in host memory
// too, and a host block can be paged out.
func TestTheMapCarriesTheDeviceAndTheResidency(t *testing.T) {
	devOf := func(li int) int { return li % 2 }
	resident := func(li int) bool { return li < 2 }
	got := blockMap(4, [][2]int{{0, 2}}, devOf, resident)
	r := byte(session.PlaceResident)
	want := []byte{byte(session.OnDevice(0)) | r, byte(session.OnDevice(1)) | r, h, h}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("blockMap = %v, want %v", got, want)
	}
	for i, b := range got {
		p := session.Placement(b)
		if wantDev := []int{0, 1, -1, -1}[i]; p.Device() != wantDev {
			t.Errorf("block %d: Device() = %d, want %d", i, p.Device(), wantDev)
		}
		if p.Resident() != (i < 2) {
			t.Errorf("block %d: Resident() = %v", i, p.Resident())
		}
	}
}

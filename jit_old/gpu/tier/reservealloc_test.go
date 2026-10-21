package tier

import "testing"

// TestReservingKVAcrossDevicesAllocatesNothing: once a model's blocks span two
// devices the state reserves the next position on each of them every token
// (model.State.relocateFor), so a warm reservation is on the decode path and
// must make no heap allocation. The vision decode allocation gates caught it
// on an eight-card box, where a bare "vulkan" opens every card: 64 objects
// over 64 tokens, every one the device list ReserveKV built per call.
func TestReservingKVAcrossDevicesAllocatesNothing(t *testing.T) {
	p := fakePlan()
	const nblocks = 4
	ws := blocks(p, nblocks)
	two := twoBlockBytes(t, p)
	g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: two}, {Dev: &fakeDev{}, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	for li := 0; li < nblocks; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("PrepLayer(%d): %s", li, g.Err())
		}
	}
	if per := g.Placed(); len(per) != 2 || per[0] == 0 || per[1] == 0 {
		t.Fatalf("blocks per device %v: the reservation would ask one device and prove nothing", per)
	}
	const pos = 9
	if !g.ReserveKV(pos) || !g.ReserveKVSeqs([]int{0}, []int{pos}) {
		t.Fatalf("the warm-up reservation was refused: %s", g.Err())
	}
	if n := testing.AllocsPerRun(100, func() { g.ReserveKV(pos) }); n != 0 {
		t.Errorf("a warm ReserveKV over two devices allocates %.1f times", n)
	}
	if n := testing.AllocsPerRun(100, func() { g.ReserveKVSeqs([]int{0}, []int{pos}) }); n != 0 {
		t.Errorf("a warm ReserveKVSeqs over two devices allocates %.1f times", n)
	}
}

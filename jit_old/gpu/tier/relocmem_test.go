package tier

import (
	"fmt"
	"testing"
)

// TestRelocationReturnsEveryDeviceBuffer runs the moves a serving process
// makes again and again -- a block spilled onto the next device with its
// history, sent home, reclaimed, and the whole seam walked home and back --
// and requires each device's live buffers to come back to the same level
// after every cycle, not only at Close, and the device a block leaves must free
// it at the move. A buffer held until the tier closes (a spilled block's old
// copy, a moved block's history pages) passes a Close-time ledger and still
// holds memory for the life of a server; and a level that only comes back at
// the cycle's end can hide a copy the block's return happens to reuse.
func TestRelocationReturnsEveryDeviceBuffer(t *testing.T) {
	p := fakePlan()
	sz, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 1 << 40}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !sz.PrepLayer(0, p, blocks(p, 1)[0]) {
		t.Fatalf("sizing run: %s", sz.Err())
	}
	blk := sz.devs[0].used
	sz.Close()

	const n, pos = 6, 100
	devs := []*fakeDev{{}, {}}
	// Four blocks and their history on the first, room for the rest and two
	// more on the second, so a spill has somewhere to go.
	g, err := New([]Slot{{Dev: devs[0], Bytes: 4*blk + blk/2}, {Dev: devs[1], Bytes: 5*blk + blk/2}},
		WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ws := blocks(p, n)
	kvDim := p.NKVHead * p.HeadDim
	hist := map[int][2][]float32{}
	for li := range ws {
		hist[li] = [2][]float32{make([]float32, pos*kvDim), make([]float32, pos*kvDim)}
		for i := range hist[li][0] {
			hist[li][0][i], hist[li][1][i] = float32(li+i%7), float32(li-i%5)
		}
	}
	place := func(li int) {
		t.Helper()
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("block %d declined: %s", li, g.Err())
		}
		if !g.ReserveKV(pos) || !g.MigrateKV(li, hist[li][0], hist[li][1], pos, true) {
			t.Fatalf("block %d's history did not go up: %s", li, g.Err())
		}
	}
	home := func(li int) {
		t.Helper()
		if !g.MigrateKV(li, hist[li][0], hist[li][1], pos, false) {
			t.Fatalf("block %d's history did not come home: %s", li, g.Err())
		}
	}
	for li := range ws {
		place(li)
	}
	live := func() string {
		out := ""
		for _, d := range devs {
			out += fmt.Sprintf("[%d buffers, %d bytes]", d.bufs-d.freed, d.allocBytes-d.freedBytes)
		}
		return out
	}
	liveOf := func(d *fakeDev) int { return d.allocBytes - d.freedBytes }
	start := placement(g, n)
	var steady string
	for cycle := 0; cycle < 4; cycle++ {
		// Spill the first device's last block onto the second, history and all.
		li := -1
		for i, d := range placement(g, n) {
			if d == 0 {
				li = i
			}
		}
		if li < 0 {
			t.Fatalf("cycle %d: nothing on the first device: %v", cycle, placement(g, n))
		}
		home(li)
		if !g.SpillAfter(li) {
			t.Fatalf("cycle %d: block %d may not spill", cycle, li)
		}
		was := liveOf(devs[0])
		place(li)
		if got := placement(g, n)[li]; got != 1 {
			t.Fatalf("cycle %d: block %d spilled to device %d, want 1", cycle, li, got)
		}
		if now := liveOf(devs[0]); now >= was {
			t.Fatalf("cycle %d: block %d spilled to device 1 and device 0 still holds %d of %d bytes: the old copy was not freed",
				cycle, li, now, was)
		}
		// Home and reclaimed.
		home(li)
		was = liveOf(devs[1])
		g.ReleaseLayers(li, li+1)
		if now := liveOf(devs[1]); now >= was {
			t.Fatalf("cycle %d: block %d went home and device 1 still holds %d of %d bytes", cycle, li, now, was)
		}
		place(li)
		// The whole seam home and back up.
		for i := range ws {
			home(i)
		}
		g.ReleaseLayers(0, n)
		for i, d := range devs {
			if w := liveOf(d); w >= len(ws)*int(blk) {
				t.Fatalf("cycle %d: every block home and device %d still holds %d bytes", cycle, i, w)
			}
		}
		for i := range ws {
			place(i)
		}
		now := live()
		t.Logf("cycle %d: placement %v, live %s", cycle, placement(g, n), now)
		if cycle == 0 {
			steady = now
		} else if now != steady {
			t.Fatalf("cycle %d: live device memory %s, after cycle 0 %s: the moves leave buffers behind",
				cycle, now, steady)
		}
	}
	t.Logf("first placement %v", start)
	g.Close()
	for i, d := range devs {
		if d.doubleFree != 0 {
			t.Errorf("device %d: %d buffer(s) freed twice", i, d.doubleFree)
		}
		if d.freed != d.bufs || d.freedBytes != d.allocBytes {
			t.Errorf("device %d after Close: %d of %d buffers and %d of %d bytes returned",
				i, d.freed, d.bufs, d.freedBytes, d.allocBytes)
		}
	}
}

//go:build linux

package tier

import "testing"

// TestATowerIsBuiltForItsPicture places a tower whose largest grid is 65536
// rows -- HunyuanOCR's -- on a 256 MB card and encodes pictures of three sizes
// on it. Its set must be built for the picture, not the grid: placed at
// visStart rows, grown for a larger picture, refused with nothing allocated
// for a picture the card cannot hold, shrunk for a small one, and returned
// whole when the blocks leave. A first block the budget refuses leaves nothing
// behind either. Built for the grid, as it was, the set is the card several
// times over and no block is placed.
func TestATowerIsBuiltForItsPicture(t *testing.T) {
	p := visionPlan()
	p.MaxSeq = 65536
	const budget = 256 << 20
	const blocks = 4
	ws := visionBlocks(p, blocks)
	// open is a card of card bytes with a budget of bytes.
	open := func(bytes, card int) (*GPU, *fakeDev) {
		d := &fakeDev{capBytes: card}
		g, err := New([]Slot{{Dev: d, Bytes: uint64(bytes)}}, WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		return g, d
	}
	g, d := open(budget, budget)
	defer g.Close()
	dt := g.devs[0]
	live := func() int { return d.allocBytes - d.freedBytes }
	for li := range blocks {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("tower block %d declined on a %d MB card: %s", li, budget>>20, g.Err())
		}
	}
	placed := g.Stats().ScratchBytes
	if dt.visCap != visStart || placed == 0 {
		t.Fatalf("placement built the set for %d rows and charged %d bytes, want %d rows and a charge",
			dt.visCap, placed, visStart)
	}
	cs := make([]float32, p.NEmbd)
	run := func(n int) {
		t.Helper()
		if !g.ReserveRows(n) {
			t.Fatalf("a picture of %d rows refused: %s", n, g.Err())
		}
		if !g.Layers(0, blocks, 0, n, make([]float32, n*p.NEmbd), cs, nil, nil) {
			t.Fatalf("a picture of %d rows did not run: %s", n, g.Err())
		}
	}
	run(3000)
	grown := g.Stats().ScratchBytes
	if dt.visCap != visRowsFor(3000, p.MaxSeq) || grown <= placed {
		t.Fatalf("a 3000-row picture: the set is %d rows and %d bytes (placed at %d)", dt.visCap, grown, placed)
	}
	// The grid itself: asked of the budget, not of the card.
	bufs, held := d.bufs, live()
	if g.ReserveRows(p.MaxSeq) {
		t.Fatalf("a %d-row set fit a %d MB card", p.MaxSeq, budget>>20)
	}
	if d.bufs != bufs || live() != held || dt.visCap != visRowsFor(3000, p.MaxSeq) {
		t.Fatalf("the refused picture allocated %d buffers (%d -> %d bytes live) and left the set at %d rows",
			d.bufs-bufs, held, live(), dt.visCap)
	}
	// Sizes that alternate keep the set: the shrink waits for visShrinkCalls
	// calls in a row that need half of it or less.
	bufs = d.bufs
	for range visShrinkCalls - 1 {
		run(3000)
		run(500)
	}
	if d.bufs != bufs || dt.visCap != visRowsFor(3000, p.MaxSeq) {
		t.Fatalf("pictures of 3000 and 500 rows in turn rebuilt the set: %d buffers allocated, %d rows",
			d.bufs-bufs, dt.visCap)
	}
	run(3000)
	for range visShrinkCalls - 1 {
		run(500)
	}
	if dt.visCap != visRowsFor(3000, p.MaxSeq) {
		t.Fatalf("%d pictures of 500 rows in a row shrank the set to %d rows", visShrinkCalls-1, dt.visCap)
	}
	run(500)
	small := g.Stats().ScratchBytes
	if dt.visCap != visRowsFor(500, p.MaxSeq) || small >= placed {
		t.Fatalf("a 500-row picture after a 3000-row one: the set is %d rows and %d bytes", dt.visCap, small)
	}
	t.Logf("a %d-row grid on a %d MB card: set %d bytes at %d rows placed, %d at 3072, %d at 512; "+
		"the grid refused with nothing allocated", p.MaxSeq, budget>>20, placed, visStart, grown, small)
	g.ReleaseLayers(0, blocks)
	if dt.scratch != 0 || dt.visCap != 0 || live() > 4096 {
		t.Fatalf("every tower block released: scratch %d bytes, set at %d rows, %d bytes live",
			dt.scratch, dt.visCap, live())
	}

	// A first block the budget refuses: its set was built, and the card
	// held it, before the budget said no to the block's weights -- and the
	// set goes with the block.
	one, od := open(budget, budget)
	if !one.PrepLayer(0, p, ws[0]) {
		t.Fatalf("one tower block: %s", one.Err())
	}
	need := int(one.Stats().BudgetUsed)
	one.Close()
	w := ws[0]
	half := (len(w.Wq.Data) + len(w.Wk.Data) + len(w.Wv.Data) + len(w.Wo.Data) + len(w.Up.Data) + len(w.Down.Data)) / 2
	g2, d2 := open(need-half, budget)
	defer g2.Close()
	if g2.PrepLayer(0, p, ws[0]) {
		t.Fatalf("a block of %d bytes placed on a budget of %d", need, need-half)
	}
	if left := d2.allocBytes - d2.freedBytes; left > 4096 || g2.devs[0].scratch != 0 || g2.devs[0].visCap != 0 {
		t.Fatalf("a refused tower block left %d bytes live (%d charged, set at %d rows) of the %d its set took",
			left, g2.devs[0].scratch, g2.devs[0].visCap, od.allocBytes)
	}
}

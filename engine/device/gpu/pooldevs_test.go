package gpu

import (
	"testing"
)

// noPromptReserve builds no prompt width at placement: the paging pairs are
// about the pool, and their budgets (cardFor) are sized in blocks.
func noPromptReserve() Option {
	return WithConfig(func(c *Config) { c.NoScratchReserve = true })
}

// pooledPair is two fake devices spending one pool of poolBytes, as one card
// under two backends does (planSlots), with budgets a and b.
func pooledPair(t *testing.T, a, b, poolBytes uint64, opt ...Option) (*GPU, *Pool, *fakeDev, *fakeDev) {
	t.Helper()
	pool := NewPool("one card", poolBytes)
	fa, fb := &fakeDev{name: "a"}, &fakeDev{name: "b"}
	g, err := New([]Slot{{Dev: fa, Name: "a", Bytes: a, Pool: pool}, {Dev: fb, Name: "b", Bytes: b, Pool: pool}}, append(opt, WithDeviceTune(TuneOff))...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(g.Close)
	return g, pool, fa, fb
}

// historyOn gives d one attention layer of history at one page.
func historyOn(t *testing.T, d *devTier) (*kvPool, *kvLayerPool) {
	t.Helper()
	kp := newKVPool(64, 0)
	d.mu.Lock()
	err := d.addKVLayer(kp, 0, kvGeom{kvRow: 16}, 1)
	d.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	return kp, kp.layers[0]
}

// growBeside is growAll from inside an allocation of another device of the
// pool, on that device's goroutine with its lock held: a stand-in for d
// growing at that moment, which asks none of its peers (fromPeers), since a
// d running beside it would wait for that lock to go first.
func growBeside(d *devTier, kp *kvPool, l *kvLayerPool) int {
	d.mu.Lock()
	peers := d.peers
	d.peers = nil
	d.mu.Unlock()
	n := growAll(d, kp, l)
	d.mu.Lock()
	d.peers = peers
	d.mu.Unlock()
	return n
}

// growAll grows d's history a page at a time until the budget or the pool
// refuses, and returns the pages it took.
func growAll(d *devTier, kp *kvPool, l *kvLayerPool) int {
	d.mu.Lock()
	defer d.mu.Unlock()
	n := l.n
	for d.growKVLayer(kp, l, l.n+1) == nil {
	}
	return l.n - n
}

// poolHeld fails when the pool was charged past its limit (Stats.PoolOver,
// Stats.ScratchOver), a hold was left open, or the pool holds other than its
// devices' charges and kept slots.
func poolHeld(t *testing.T, g *GPU, pool *Pool) {
	t.Helper()
	var held uint64
	for _, d := range g.devs {
		d.mu.Lock()
		if d.claimed != 0 {
			t.Errorf("%s holds %d bytes of its pool no operation is open for", label(d.dev, d.ord), d.claimed)
		}
		if d.winHeld != 0 {
			t.Errorf("%s holds %d bytes of its pool for a scratch window no longer open", label(d.dev, d.ord), d.winHeld)
		}
		held += d.used + d.swapHeld
		d.mu.Unlock()
	}
	if over := g.Stats().PoolOver; over != 0 || pool.Used() > pool.Limit() {
		t.Fatalf("%d charge(s) left the pool past its limit: %d of %d bytes used", over, pool.Used(), pool.Limit())
	}
	if over := g.Stats().ScratchOver; over != 0 {
		t.Fatalf("%d scratch window(s) left the pool past its limit: %d of %d bytes used", over, pool.Used(), pool.Limit())
	}
	if pool.Used() != held {
		t.Fatalf("the pool holds %d bytes and its devices %d", pool.Used(), held)
	}
}

// TestAHistoryGrowsInsideItsPool: a history's growth on one device of a pool
// holds its room before it allocates, so the other device, growing at that
// very moment (the allocation's), finds the room taken rather than spending
// it under the first. Checked and charged after, both spent the same bytes
// and the pool went past its limit (Stats.PoolOver).
func TestAHistoryGrowsInsideItsPool(t *testing.T) {
	g, pool, fa, _ := pooledPair(t, 1<<20, 1<<20, 1<<20)
	da, db := g.devs[0], g.devs[1]
	kpa, la := historyOn(t, da)
	kpb, lb := historyOn(t, db)
	grown := 0
	fa.onAlloc = func() {
		fa.onAlloc = nil
		grown = growBeside(db, kpb, lb)
	}
	da.mu.Lock()
	err := da.growKVLayer(kpa, la, 40)
	da.mu.Unlock()
	if grown == 0 {
		t.Fatal("the other device grew nothing while this one allocated: the gate proves nothing")
	}
	if err != nil {
		t.Fatalf("a growth that found room lost it to the other device: %v", err)
	}
	poolHeld(t, g, pool)
	t.Logf("one device grew to %d pages, the other by %d pages beside it; the pool %d of %d", la.n, grown, pool.Used(), pool.Limit())
}

// streamedOn offers n streamed blocks of p to g, which a pooledPair places on
// its first device alone, and fails unless they are all there.
func streamedOn(t *testing.T, g *GPU, n int) {
	t.Helper()
	p := pagePlan()
	streamBlocks(g, n)
	ws := pageBlocks(p, n)
	for li := 0; li < n; li++ {
		if ok, err := g.PrepLayerOn("a", li, p, ws[li]); !ok || err != nil {
			t.Fatalf("block %d declined: %v %s", li, err, g.Err())
		}
	}
	if pl := g.Placed(); pl[0] != n {
		t.Fatalf("placed %v: the gate wants every block on the first device", pl)
	}
}

// inBlocks counts d's streamed blocks paged in.
func inBlocks(d *devTier) int {
	d.mu.Lock()
	defer d.mu.Unlock()
	n := 0
	for li := range d.stream {
		if l := d.layers[li]; l != nil && l.pg != nil && l.pg.in {
			n++
		}
	}
	return n
}

// TestASwapSlotIsKeptFromTheOtherDevice: a device paging through one slot of
// a pool the other device grows history in keeps the slot. Inside a page-in
// the bytes a page-out frees are the page-in's (openSlot); between two
// page-ins they are held as the slot (keepOf). Given back to the pool, the
// other device grew into them and the next page-in found nothing to page
// out.
func TestASwapSlotIsKeptFromTheOtherDevice(t *testing.T) {
	const n = 4
	p := pagePlan()
	card := cardFor(t, p, 2, noPromptReserve())
	g, pool, fa, _ := pooledPair(t, card, card, card+1<<16, noPromptReserve())
	da, db := g.devs[0], g.devs[1]
	streamedOn(t, g, n)
	kpb, lb := historyOn(t, db)
	if !token(t, g, p, n, 0) {
		t.Fatalf("token 0: %s", g.Err())
	}
	poolHeld(t, g, pool)
	outs := da.stats().PageOuts
	fills := 0
	fa.onFree = func() {
		fills++
		growBeside(db, kpb, lb)
	}
	if !token(t, g, p, n, 1) {
		t.Fatalf("a page-in lost its slot to the other device's history: %s", g.Err())
	}
	fa.onFree = nil
	if fills == 0 || da.stats().PageOuts == outs {
		t.Fatalf("%d page-out(s) and %d growth(s) beside them: the gate proves nothing", da.stats().PageOuts-outs, fills)
	}
	poolHeld(t, g, pool)
	// Between two page-ins: the block goes out for something else, the other
	// device grows, the next block comes in.
	da.mu.Lock()
	for li := range da.stream {
		if l := da.layers[li]; l != nil && l.pg != nil && l.pg.in {
			da.pageOut(li)
		}
	}
	da.mu.Unlock()
	if inBlocks(da) != 0 {
		t.Fatal("a streamed block is still in: the gate proves nothing")
	}
	growAll(db, kpb, lb)
	if !token(t, g, p, n, 2) {
		t.Fatalf("the slot went to the other device's history between two page-ins: %s", g.Err())
	}
	poolHeld(t, g, pool)
	t.Logf("%d page-out(s) beside the other device's growth; its history %d pages", da.stats().PageOuts-outs, lb.n)
}

// TestAHistoryPagesOutTheOtherDevicesBlock: a device whose history finds its
// pool spent pages out the other device's idle streamed blocks (fromPeers),
// as it pages out its own, and never the other's last one: the other keeps a
// slot and still runs its token.
func TestAHistoryPagesOutTheOtherDevicesBlock(t *testing.T) {
	const n = 4
	p := pagePlan()
	block, _ := blockAndScratch(t, p, noPromptReserve())
	card := cardFor(t, p, 3, noPromptReserve())
	g, pool, _, _ := pooledPair(t, card, card, card+block/2, noPromptReserve())
	da, db := g.devs[0], g.devs[1]
	streamedOn(t, g, n)
	if !token(t, g, p, n, 0) {
		t.Fatalf("token 0: %s", g.Err())
	}
	poolHeld(t, g, pool)
	if in := inBlocks(da); in < 2 {
		t.Fatalf("%d streamed block(s) in after a token on two slots: the gate proves nothing", in)
	}
	outs := da.stats().PageOuts
	kpb, lb := historyOn(t, db)
	growAll(db, kpb, lb)
	took := da.stats().PageOuts - outs
	if got := kpb.charged(); took == 0 || got < block {
		t.Fatalf("the history grew to %d bytes beside an idle %d-byte block of the other device, "+
			"which paged out %d block(s): a device short of its pool pages out only its own", got, block, took)
	}
	if inBlocks(da) == 0 {
		t.Fatal("the other device's last block went out: its slot is gone")
	}
	if !token(t, g, p, n, 1) {
		t.Fatalf("the device whose block went out does not run its token: %s", g.Err())
	}
	poolHeld(t, g, pool)
	t.Logf("history %d bytes, %d block(s) of the other device paged out for it", kpb.charged(), took)
}

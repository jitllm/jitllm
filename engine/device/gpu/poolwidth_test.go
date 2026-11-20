package gpu

import (
	"testing"
	"time"
)

// TestAPeerInFlightIsWaitedFor: a block placed resident on a device whose
// pool a peer's idle streamed blocks fill, while a submission of that peer
// is in flight, waits for the submission and pages the idle blocks out,
// where the peer is after it in the lock order. Passed over, the block was
// declined to the host beside blocks nothing was reading. A peer before it
// is still passed over, and counted (Stats.PeerPassed).
func TestAPeerInFlightIsWaitedFor(t *testing.T) {
	const n = 4
	p := pagePlan()
	block, _ := blockAndScratch(t, p, noPromptReserve())
	card := cardFor(t, p, 3, noPromptReserve())
	for _, tc := range []struct {
		name, streams, admits string
		locked                bool
	}{{"after", "b", "a", false}, {"before", "a", "b", false}, {"before-locked", "a", "b", true}} {
		t.Run(tc.name, func(t *testing.T) {
			g, pool, _, _ := pooledPair(t, card, card, 1<<40, noPromptReserve())
			by := map[string]*devTier{"a": g.devs[0], "b": g.devs[1]}
			ds, da := by[tc.streams], by[tc.admits]
			streamBlocks(g, n)
			ws := pageBlocks(p, n+1)
			for li := 0; li < n; li++ {
				if ok, err := g.PrepLayerOn(tc.streams, li, p, ws[li]); !ok || err != nil {
					t.Fatalf("block %d declined: %v %s", li, err, g.Err())
				}
			}
			if !token(t, g, p, n, 0) {
				t.Fatalf("token 0: %s", g.Err())
			}
			if in := inBlocks(ds); in < 2 {
				t.Fatalf("%d streamed block(s) in after a token on three slots: the gate proves nothing", in)
			}
			pool.SetLimit(pool.Used() + block/2)
			ds.mu.Lock()
			tk := ds.beginSub()
			ds.mu.Unlock()
			held := make(chan struct{})
			done := make(chan struct{})
			go func() {
				ds.mu.Lock()
				if tc.locked {
					// A pair taken in lockPair's order: the peer's lock held
					// while it waits for the admitting device's.
					close(held)
					time.Sleep(50 * time.Millisecond)
					da.mu.Lock()
					da.mu.Unlock()
				} else {
					ds.mu.Unlock()
					close(held)
					time.Sleep(50 * time.Millisecond)
					ds.mu.Lock()
				}
				ds.endSub(tk)
				ds.mu.Unlock()
				close(done)
			}()
			<-held
			type res struct {
				ok  bool
				err error
			}
			placed := make(chan res, 1)
			go func() {
				ok, err := g.PrepLayerOn(tc.admits, n, p, ws[n])
				placed <- res{ok, err}
			}()
			var r res
			select {
			case r = <-placed:
			case <-time.After(30 * time.Second):
				t.Fatal("the admission and the peer's holder wait for each other: a deadlock")
			}
			<-done
			if r.err != nil {
				t.Fatal(r.err)
			}
			ok := r.ok
			st := da.stats()
			t.Logf("placed %v: %d block(s) of the other device paged out, %d pass(es) over it, %d wait(s) with its own lock let go: %s",
				ok, st.PeerPageOuts, st.PeerPassed, st.PeerLetGo, g.Err())
			if da.ord > ds.ord && st.PeerLetGo == 0 {
				t.Fatalf("a busy peer before the device in the lock order: its own lock never let go (%d pass(es) over it)", st.PeerPassed)
			}
			if !ok || st.PeerPageOuts == 0 {
				t.Fatalf("a block placed resident beside a peer's idle streamed blocks, a submission of the peer in flight: placed %v, %d paged out: %s", ok, st.PeerPageOuts, g.Err())
			}
			if inBlocks(ds) == 0 {
				t.Fatal("the other device's last block went out: its slot is gone")
			}
			if st.PeerPassed != 0 {
				t.Fatalf("a busy peer passed over %d time(s)", st.PeerPassed)
			}
			poolHeld(t, g, pool)
		})
	}
}

// TestASharedPoolReservesTheWidthADeviceAloneDoes: a block placed on a device
// whose pool another device spends reserves the prompt width the same budget
// reserves on a device alone. A prompt arena's buffer made beside the one it
// replaces held both in the window (the peak), so the shared pool settled a
// narrower width than the room the growth was priced at.
func TestASharedPoolReservesTheWidthADeviceAloneDoes(t *testing.T) {
	p := pagePlan()
	widthOn := func(g *GPU) (int, uint64) {
		t.Helper()
		if ok, err := g.PrepLayerOn(g.devs[0].dev.Name(), 0, p, pageBlocks(p, 1)[0]); !ok || err != nil {
			t.Fatalf("block 0 declined: %v %s", err, g.Err())
		}
		d := g.devs[0]
		d.mu.Lock()
		defer d.mu.Unlock()
		return d.promptW, d.used
	}
	alone := func(bytes uint64) (int, uint64) {
		g, err := New([]Slot{{Dev: &fakeDev{name: "a"}, Name: "a", Bytes: bytes}}, WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatal(err)
		}
		defer g.Close()
		return widthOn(g)
	}
	full, used := alone(1 << 40)
	if full < 2*batchGrain {
		t.Fatalf("a device with room reserved a %d-row prompt width: nothing narrower to settle at, the gate proves nothing", full)
	}
	w, _ := alone(used)
	if w != full {
		t.Fatalf("a device alone at the %d bytes a %d-row width placed in reserved %d rows", used, full, w)
	}
	g, pool, _, _ := pooledPair(t, used, used, used)
	sw, _ := widthOn(g)
	st := g.devs[0].stats()
	t.Logf("%d bytes: %d rows alone, %d in a shared pool (%d of %d bytes used)", used, w, sw, pool.Used(), pool.Limit())
	if sw != w {
		t.Fatalf("the same %d-byte budget reserved %d rows on a device alone and %d in a shared pool", used, w, sw)
	}
	if st.PoolOver != 0 || st.ScratchOver != 0 {
		t.Fatalf("%d charge(s) and %d window(s) past the pool", st.PoolOver, st.ScratchOver)
	}
}

// TestAReservedWidthPagesOutTheOtherDevicesBlock: a placement's reserved
// prompt width (reserveScratch) in a pool the other device's idle streamed
// blocks fill pages them out for its buffers, as the width's instances and
// the block's admission do, and as a device alone pages out its own: the
// width a device alone reserves at that budget. Passed over, the probe
// settled a narrower width than the room the other device was idling in.
func TestAReservedWidthPagesOutTheOtherDevicesBlock(t *testing.T) {
	const n = 4
	p := pagePlan()
	block, _ := blockAndScratch(t, p, noPromptReserve())
	one, err := New([]Slot{{Dev: &fakeDev{name: "a"}, Name: "a", Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatal(err)
	}
	if ok, err := one.PrepLayerOn("a", n, p, pageBlocks(p, 1)[0]); !ok || err != nil {
		t.Fatalf("block %d declined alone: %v %s", n, err, one.Err())
	}
	one.devs[0].mu.Lock()
	full, need := one.devs[0].promptW, one.devs[0].used
	one.devs[0].mu.Unlock()
	one.Close()
	g, pool, _, _ := pooledPair(t, 1<<40, 1<<40, 1<<40)
	da, db := g.devs[0], g.devs[1]
	streamBlocks(g, n)
	ws := pageBlocks(p, n+1)
	for li := 0; li < n; li++ {
		if ok, err := g.PrepLayerOn("b", li, p, ws[li]); !ok || err != nil {
			t.Fatalf("block %d declined: %v %s", li, err, g.Err())
		}
	}
	if !token(t, g, p, n, 0) {
		t.Fatalf("token 0: %s", g.Err())
	}
	if in := inBlocks(db); in < 3 {
		t.Fatalf("%d streamed block(s) in: the gate proves nothing", in)
	}
	pool.SetLimit(pool.Used() + need - block)
	if ok, err := g.PrepLayerOn("a", n, p, ws[n]); !ok || err != nil {
		t.Fatalf("block %d declined: %v %s", n, err, g.Err())
	}
	da.mu.Lock()
	w := da.promptW
	da.mu.Unlock()
	st := da.stats()
	t.Logf("%d rows alone at %d bytes, %d beside the other device's idle blocks, %d of them paged out", full, need, w, st.PeerPageOuts)
	if w != full {
		t.Fatalf("the reserved width settled at %d rows beside the other device's idle streamed blocks, %d alone", w, full)
	}
	if inBlocks(db) == 0 {
		t.Fatal("the other device's last block went out: its slot is gone")
	}
	poolHeld(t, g, pool)
}

// TestAWindowGivesItsOperationBackWhatItTook: a scratch window that took
// its buffers' room from the operation it runs in (the pool full, no idle
// block of the other device to page out) gives what it did not spend back
// to that operation at its close. Given to the pool, the other device grew
// into it and the operation's own charge went past the pool (PoolOver).
func TestAWindowGivesItsOperationBackWhatItTook(t *testing.T) {
	g, pool, _, _ := pooledPair(t, 1<<30, 1<<30, 1<<30, noPromptReserve())
	da, db := g.devs[0], g.devs[1]
	kpb, lb := historyOn(t, db)
	const op = 4 << 20
	pool.SetLimit(pool.Used() + op)
	da.mu.Lock()
	before, held := da.claim(op)
	if !held {
		da.mu.Unlock()
		t.Fatal("the operation's hold did not fit the pool it was sized to")
	}
	m := da.scratchWin()
	b, err := da.alloc(1 << 20)
	if err != nil {
		da.mu.Unlock()
		t.Fatalf("a window inside an operation holding %d bytes: %v", op, err)
	}
	took := op - da.claimed
	b.Free()
	m.close()
	da.mu.Unlock()
	if took == 0 {
		t.Fatal("the window took nothing from the operation: the gate proves nothing")
	}
	grew := growAll(db, kpb, lb)
	da.mu.Lock()
	da.charge(op)
	da.refund(op)
	da.unclaim(before)
	da.mu.Unlock()
	t.Logf("the window took %d bytes from the operation; the other device grew %d page(s) after its close", took, grew)
	if s := da.stats(); s.PoolOver != 0 {
		t.Fatalf("the operation's charge went past the pool %d time(s): the window gave the pool what it took from the operation", s.PoolOver)
	}
	poolHeld(t, g, pool)
}

// TestAPairIsNotLetGo: a hold taken with another device's (lockPair, a
// block's move) is not let go in the middle. A history the later device
// grows under it, short of its pool, passes over the earlier device, whose
// lock is its own caller's, rather than waiting for it.
func TestAPairIsNotLetGo(t *testing.T) {
	const n = 4
	p := pagePlan()
	block, _ := blockAndScratch(t, p, noPromptReserve())
	card := cardFor(t, p, 3, noPromptReserve())
	g, pool, _, _ := pooledPair(t, card, card, 1<<40, noPromptReserve())
	ds, da := g.devs[0], g.devs[1]
	streamBlocks(g, n)
	ws := pageBlocks(p, n)
	for li := range n {
		if ok, err := g.PrepLayerOn("a", li, p, ws[li]); !ok || err != nil {
			t.Fatalf("block %d declined: %v %s", li, err, g.Err())
		}
	}
	if !token(t, g, p, n, 0) {
		t.Fatalf("token 0: %s", g.Err())
	}
	if in := inBlocks(ds); in < 2 {
		t.Fatalf("%d streamed block(s) in after a token on three slots: the gate proves nothing", in)
	}
	kp, l := historyOn(t, da)
	pool.SetLimit(pool.Used() + block/2)
	grew := make(chan int, 1)
	go func() {
		lockPair(ds, da)
		from := l.n
		for da.growKVLayer(kp, l, l.n+1) == nil {
		}
		n := l.n - from
		unlockPair(ds, da)
		grew <- n
	}()
	select {
	case n := <-grew:
		t.Logf("the history grew %d page(s) under the pair", n)
	case <-time.After(30 * time.Second):
		t.Fatal("a pair's hold let go in the middle waits for itself: a deadlock")
	}
	st := da.stats()
	if st.PeerPassed == 0 {
		t.Fatal("the history never asked the device before it: the gate proves nothing")
	}
	if st.PeerLetGo != 0 {
		t.Fatalf("a pair's hold let go %d time(s)", st.PeerLetGo)
	}
	poolHeld(t, g, pool)
}

package tier

import (
	"slices"
	"testing"
)

// TestRefusedNamesTheDeviceThatSaidNo places blocks on two devices, leaves the
// first with no room, and asks both for a longer history.
//
// The answer is the first device's blocks and only those: naming other blocks
// would send them to the host without freeing a byte where growth needs it.
func TestRefusedNamesTheDeviceThatSaidNo(t *testing.T) {
	p := fakePlan()
	two := twoBlockBytes(t, p)
	g, err := New([]Slot{
		{Dev: &fakeDev{}, Bytes: two},
		{Dev: &fakeDev{}, Bytes: 1 << 40},
	}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(g.Close)
	const n = 5
	for li, w := range blocks(p, n) {
		if !g.PrepLayer(li, p, w) {
			t.Fatalf("block %d declined on every device: %s", li, g.Err())
		}
	}
	at := placement(g, n)
	var want []int
	for li, d := range at {
		if d == 0 {
			want = append(want, li)
		}
	}
	if len(want) == 0 || len(want) == n {
		t.Fatalf("placement %v does not split the blocks across the two devices", at)
	}
	if g.ReserveKV(1000) {
		t.Fatalf("a device with no room to spare grew its history (placement %v)", at)
	}
	got := g.Refused()
	slices.Sort(got)
	if !slices.Equal(got, want) {
		t.Fatalf("Refused() = %v, want the full device's blocks %v (placement %v)", got, want, at)
	}
	g.SetBudget(1 << 40)
	if !g.ReserveKV(1000) {
		t.Fatalf("refused with room restored: %s", g.Err())
	}
	if got := g.Refused(); got != nil {
		t.Fatalf("Refused() = %v after a reservation that succeeded", got)
	}
	t.Logf("placement %v: the refusal named %v", at, want)
}

// TestTrimAndDetachGiveRoomBack runs two sessions on one card, and holds the
// two ways room comes back: a session's history shrinking to a shorter
// sequence, and a session detaching.
//
// Each session's pages are its own, so a trim goes ahead under another session
// and leaves that session's history exactly as it was. The pages stay in the
// pool as free room for the next sequence (kvCompact gives them to the budget
// when something needs it), and every return of room moves the report a caller
// holding relocated blocks waits on.
func TestTrimAndDetachGiveRoomBack(t *testing.T) {
	p := fakePlan()
	g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(g.Close)
	ws := blocks(p, 2)
	a := g.Attach().(*gpuSession)
	for li, w := range ws {
		if !a.PrepLayer(li, p, w) {
			t.Fatalf("block %d declined: %s", li, g.Err())
		}
	}
	if !a.ReserveKV(400) {
		t.Fatalf("400 positions refused: %s", g.Err())
	}
	b := g.Attach().(*gpuSession)
	for li, w := range ws {
		if !b.PrepLayer(li, p, w) {
			t.Fatalf("second session: block %d declined: %s", li, g.Err())
		}
	}
	if !b.ReserveKV(200) {
		t.Fatalf("the second session's 200 positions refused: %s", g.Err())
	}
	aHeld, bHeld, gen := a.HeldBytes(), b.HeldBytes(), g.RoomGen()
	if !a.TrimKV(100) || a.HeldBytes() >= aHeld {
		t.Fatalf("a session asking for 100 positions kept %d of its %d bytes", a.HeldBytes(), aHeld)
	}
	if b.HeldBytes() != bHeld {
		t.Fatalf("trimming the first session moved the second's history: %d -> %d bytes",
			bHeld, b.HeldBytes())
	}
	if g.RoomGen() == gen {
		t.Fatal("trimming gave pages back and the room report did not move")
	}
	gen = g.RoomGen()
	b.Detach()
	if g.RoomGen() == gen {
		t.Fatal("detaching a session gave its pages back and the room report did not move")
	}
	for _, l := range g.devs[0].layers {
		if l != nil && (l.kv[b.sid] != nil || l.kv[a.sid] == nil) {
			t.Fatalf("after the detach the block holds history for %d session(s), want only the first's", len(l.kv))
		}
	}
	if st := g.Stats(); a.HeldBytes() != st.KVBytes {
		t.Fatalf("the ledger says %d bytes of history and the one session left holds %d",
			st.KVBytes, a.HeldBytes())
	}
	t.Logf("trim %d -> %d bytes, the other session's %d untouched; detach gave the rest back",
		aHeld, a.HeldBytes(), bHeld)
}

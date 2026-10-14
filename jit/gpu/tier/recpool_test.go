package tier

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
)

// TestRecurrentPoolReturnsEveryByte walks the recurrent pools (recpool.go)
// through what a serving process does with hybrid sessions -- sessions
// arriving, one leaving from the middle of the seats and one from the top, a
// seat reused, a batch growing its seat, a whole block released -- and requires
// the ledger, the device's live buffers and the seats to agree at every step,
// and every byte and buffer back when the last session has gone. A pool sized
// to the most sessions it ever held, or a seat outliving its session, passes a
// Close-time ledger and holds memory for the life of a server.
//
// Every resize carries the seated sessions' states across on the device, so
// the fake's ledger also counts what crossed: exactly the seats held, one
// device copy per run of them, and not one read to the host anywhere in the
// walk -- arrivals, departures, a batch's growth or a release.
func TestRecurrentPoolReturnsEveryByte(t *testing.T) {
	d := &fakeDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(g.Close)
	dt := g.devs[0]
	p := pagePlan()
	p.Recurrent = nn.RecurrentPlan{
		Conv: 4, Chans: 96, KHeads: 2, VHeads: 4, KDim: 8, VDim: 8,
		StateLen: 4 * 8 * 8, ConvState: 3 * 96,
	}
	slot := uint64(2 * (p.Recurrent.StateLen + p.Recurrent.ConvState) * 4)
	l0, l1 := &layer{linear: true, ok: true}, &layer{linear: true, ok: true}
	dt.layers = map[int]*layer{0: l0, 1: l1}
	used0, live0 := dt.used, len(d.liveBufs)
	// carried asks for the device copies so far to have carried exactly
	// seats slots' worth of state, in both pools, and the host to have read
	// nothing.
	carried := func(when string, seats int) {
		t.Helper()
		want := 2 * seats * int(slot)
		switch {
		case d.reads != 0:
			t.Fatalf("%s: %d read(s), %d bytes, to the host: a pool's resize must carry its states "+
				"on the device", when, d.reads, d.readBytes)
		case d.copyBytes != want+int(dt.RecZeroBytes) || dt.RecCarryBytes != uint64(want):
			t.Fatalf("%s: %d bytes copied on the device (RecCarryBytes %d, RecZeroBytes %d), want %d "+
				"carried -- %d seat(s) in two pools -- beside the zeroing", when, d.copyBytes,
				dt.RecCarryBytes, dt.RecZeroBytes, want, seats)
		}
	}

	attach := func(sid uint64) {
		t.Helper()
		for _, l := range []*layer{l0, l1} {
			if !dt.addSessionRec(sid, l, p) {
				t.Fatalf("session %d declined: %s", sid, dt.LastErr)
			}
		}
	}
	// zeros is the zeroing's source the pools keep (recZeros): one buffer
	// once a seat has been zeroed.
	zeros := func() int {
		if dt.recZeros != nil {
			return 1
		}
		return 0
	}
	// check asks for slots slots in each of the two pools, and the seats.
	check := func(when string, slots int, seats map[uint64]int) {
		t.Helper()
		want := 2 * uint64(slots) * slot
		switch {
		case dt.recSlots != slots:
			t.Fatalf("%s: the pools hold %d slots, want %d", when, dt.recSlots, slots)
		case dt.used-used0 != want || dt.RecBytes != want:
			t.Fatalf("%s: charged %d and RecBytes %d, want %d -- two pools of %d slots",
				when, dt.used-used0, dt.RecBytes, want, slots)
		case l0.recBytes+l1.recBytes != want:
			t.Fatalf("%s: the blocks hold %d, want %d: the release walks refund l.recBytes",
				when, l0.recBytes+l1.recBytes, want)
		case len(d.liveBufs)-live0 != 8+zeros():
			t.Fatalf("%s: %d live buffers, want 8 and the zeros kept (%d) -- two pools, two halves of state and window",
				when, len(d.liveBufs)-live0, zeros())
		case len(dt.recSeats) != len(seats):
			t.Fatalf("%s: %d seats, want %d", when, len(dt.recSeats), len(seats))
		}
		for sid, base := range seats {
			if st, ok := dt.recSeats[sid]; !ok || st.base != base {
				t.Fatalf("%s: session %d is at slot %d (seated %v), want %d", when, sid, st.base, ok, base)
			}
			if got := dt.heldBy(sid); got != 2*slot*uint64(dt.recSeats[sid].rows) {
				t.Fatalf("%s: session %d holds %d, want its seat in both blocks", when, sid, got)
			}
		}
	}

	attach(1)
	attach(2)
	attach(3)
	check("three sessions", 3, map[uint64]int{1: 0, 2: 1, 3: 2})
	// The second arrival carried one seat across, the third two (one run).
	carried("three sessions", 1+2)
	if d.copies-int(dt.RecZeroCopies) != 2*4*2 {
		t.Fatalf("%d device copies beside %d zeroings, want 16: two resizes of two pools, four buffers each, "+
			"each seat run one copy", d.copies, dt.RecZeroCopies)
	}
	// The top seat leaving shrinks every pool; one from the middle frees its
	// slot for the next arrival.
	// Both seats left are carried, so the shrunk pools upload nothing.
	w0 := d.writeBytes
	dt.dropSession(3)
	if up := d.writeBytes - w0; up != 0 {
		t.Fatalf("a shrink that carries every slot it keeps uploaded %d bytes from the host", up)
	}
	check("the top seat left", 2, map[uint64]int{1: 0, 2: 1})
	carried("the top seat left", 3+2)
	dt.dropSession(1)
	check("the bottom seat left", 2, map[uint64]int{2: 1})
	attach(4)
	check("a seat reused", 2, map[uint64]int{2: 1, 4: 0})
	// A batch of three grows its seat into a run of its own.
	w0, z0, s0 := d.writeBytes, dt.RecZeroBytes, dt.RecZeroSource
	if !dt.growSeat(4, 3) {
		t.Fatalf("growSeat: %s", dt.LastErr)
	}
	if st := dt.recSeats[4]; st.rows != 3 || st.base != 2 || dt.recSlots != 5 {
		t.Fatalf("a batch of three is at %+v of %d slots, want three slots from 2 of 5", st, dt.recSlots)
	}
	if want := 2 * 5 * slot; dt.used-used0 != want {
		t.Fatalf("charged %d for 5 slots, want %d", dt.used-used0, want)
	}
	// Session 2's seat is carried, and the growing seat's own one slot is set
	// aside on the device and put back at its new place (a seat may grow
	// while it holds a live state), so three seats' worth moves. The host
	// zeroes the four slots the carry does not fill, and the new seat once
	// more as it is taken: seven slots in each pool, not the whole of either.
	// A seat's zeroing copies from zeros uploaded once a call
	// (recZeroLayers) where the resize writes its slots from the host: the
	// two together are the seven slots.
	carried("a batch grown", 5+1+2)
	host := d.writeBytes - w0 - int(dt.RecZeroSource-s0)
	if zb, want := host+int(dt.RecZeroBytes-z0), 2*7*int(slot); zb != want {
		t.Fatalf("growing a seat to a batch of three zeroed %d bytes (%d from the host, %d on the "+
			"device), want %d -- the slots no carry fills and the new seat, not the pools",
			zb, host, dt.RecZeroBytes-z0, want)
	}
	dt.dropSession(4)
	check("the batch left", 2, map[uint64]int{2: 1})
	carried("the batch left", 8+1)
	dt.dropSession(2)
	if dt.used != used0 || dt.RecBytes != 0 || l0.pool != nil || l1.pool != nil || len(dt.recSeats) != 0 {
		t.Fatalf("every session gone and %d still charged, RecBytes %d, pools %v %v, %d seats",
			dt.used-used0, dt.RecBytes, l0.pool != nil, l1.pool != nil, len(dt.recSeats))
	}
	if len(d.liveBufs) != live0 || d.doubleFree != 0 {
		t.Fatalf("%d buffers still live, %d freed twice", len(d.liveBufs)-live0, d.doubleFree)
	}

	// A released block takes its pool; the other keeps the seats it holds.
	attach(5)
	attach(6)
	dt.ReleaseLayers(1, 2)
	if l1.pool != nil || l0.pool == nil || dt.used-used0 != 2*slot || len(d.liveBufs)-live0 != 4+zeros() {
		t.Fatalf("after releasing a block: its pool %v, the other's %v, %d charged, %d live",
			l1.pool != nil, l0.pool != nil, dt.used-used0, len(d.liveBufs)-live0)
	}
	dt.ReleaseLayers(0, 1)
	carried("every block released", 9+1)
	if dt.used != used0 || len(dt.recSeats) != 0 || len(d.liveBufs) != live0 || d.doubleFree != 0 {
		t.Fatalf("every block released and %d charged, %d seats, %d live, %d freed twice",
			dt.used-used0, len(dt.recSeats), len(d.liveBufs)-live0, d.doubleFree)
	}
}

// TestMigrateRecBringsOnlyItsSeatHome: MigrateRec reads a session's recurrent
// state home (the -kv-cache seal, a seam moving), and a pool holds every
// session's state for the block. A buffer reads from its start, so a session
// seated above slot 0 used to bring every state below it home too -- other
// sessions' summaries. Three sessions each send a state of their own up and
// read it back: each must get exactly its own, and the host must read exactly
// its seat's bytes, in both forms of the pool, on every real device. The
// sizes are not multiples of 16 floats, so every seat's offset is ragged.
func TestMigrateRecBringsOnlyItsSeatHome(t *testing.T) {
	devs := realDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		// The tier owns the device and closes it, so one tier serves both
		// forms: the last session out returns the pools to no form at all.
		g, err := New([]Slot{{Dev: d, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		for _, shared := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/%s/shared=%v", d.API(), d.Name(), shared), func(t *testing.T) {
				migrateRecCase(t, g.devs[0], shared)
			})
		}
		g.Close()
	}
}

func migrateRecCase(t *testing.T, dt *devTier, shared bool) {
	dt.SharedRec = shared
	p := pagePlan()
	p.Recurrent = nn.RecurrentPlan{
		Conv: 4, Chans: 97, KHeads: 1, VHeads: 3, KDim: 7, VDim: 5,
		StateLen: 3 * 5 * 7, ConvState: 3 * 97,
	}
	S, C := p.Recurrent.StateLen, p.Recurrent.ConvState
	l := &layer{linear: true, ok: true}
	dt.layers = map[int]*layer{0: l}
	mine := func(sid uint64, n, salt int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(int(sid)*100000 + salt*10000 + i)
		}
		return v
	}
	const sessions = 3
	for sid := uint64(1); sid <= sessions; sid++ {
		if !dt.addSessionRec(sid, l, p) {
			t.Fatalf("session %d declined: %s", sid, dt.LastErr)
		}
		if !dt.migrateRec(sid, 0, mine(sid, C, 1), mine(sid, S, 2), true) {
			t.Fatalf("session %d: sending its state up: %s", sid, dt.LastErr)
		}
	}
	for sid := uint64(1); sid <= sessions; sid++ {
		conv, state := make([]float32, C), make([]float32, S)
		_, b0 := backend.HostReads()
		if !dt.migrateRec(sid, 0, conv, state, false) {
			t.Fatalf("session %d: bringing its state home: %s", sid, dt.LastErr)
		}
		_, b1 := backend.HostReads()
		if got, want := b1-b0, uint64(4*(C+S)); got != want {
			t.Fatalf("session %d at slot %d: %d bytes read to the host, want its own seat's %d",
				sid, dt.recSeats[sid].base, got, want)
		}
		for _, c := range []struct {
			what      string
			got, want []float32
		}{{"window", conv, mine(sid, C, 1)}, {"state", state, mine(sid, S, 2)}} {
			for i := range c.want {
				if c.got[i] != c.want[i] {
					t.Fatalf("session %d %s[%d] = %g, want %g: another seat's state came home",
						sid, c.what, i, c.got[i], c.want[i])
				}
			}
		}
	}
	for sid := uint64(1); sid <= sessions; sid++ {
		dt.dropSession(sid)
	}
	if dt.recSlots != 0 || len(dt.recSeats) != 0 || l.pool != nil {
		t.Fatalf("every session gone and %d slots, %d seats, pool %v", dt.recSlots, len(dt.recSeats), l.pool != nil)
	}
}

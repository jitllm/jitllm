package tier

import (
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
)

// TestRecurrentStateIsChargedAndReturned: a linear block's four state buffers
// must be allocated, zeroed, charged, and given back on BOTH free walks.
//
// The ledger is the gate because a leak here is otherwise invisible: four
// buffers on a path with two free walks. Zeroed is part of it: a summary of
// garbage is a history the model never saw, and later tokens stay fluent.
func TestRecurrentStateIsChargedAndReturned(t *testing.T) {
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
	want := uint64(2 * (p.Recurrent.StateLen + p.Recurrent.ConvState) * 4)

	l := &layer{}
	used0, bufs0 := dt.used, d.bufs
	if !dt.addSessionRec(0, l, p) {
		t.Fatalf("addSessionRec declined: %s", dt.LastErr)
	}
	if l.recOf(0) == nil {
		t.Fatal("no recurrent pair was recorded for the session")
	}
	if l.pool == nil {
		t.Fatal("the block has no recurrent pool")
	}
	for i := 0; i < 2; i++ {
		if l.pool.s[i] == nil || l.pool.c[i] == nil {
			t.Fatalf("half %d is incomplete: a swap would read a nil buffer", i)
		}
	}
	if d.bufs-bufs0 != 4 {
		t.Fatalf("%d buffer(s) allocated, want 4 -- two halves of the delta state and "+
			"two of the conv window", d.bufs-bufs0)
	}
	if dt.used-used0 != want {
		t.Fatalf("charged %d bytes, want %d", dt.used-used0, want)
	}
	if dt.RecBytes != want || l.recBytes != want {
		t.Fatalf("RecBytes %d, l.recBytes %d, want %d in both -- the release path refunds "+
			"l.recBytes, so the two must agree or the budget leaks per relocation",
			dt.RecBytes, l.recBytes, want)
	}

	// It comes back through the real walk, not through freeRec: a gate that
	// does the freeing cannot notice a walk that forgot to.
	freed0, freedBytes0 := d.freed, d.freedBytes
	dt.dropLayerLocal(l)
	if d.freed-freed0 != 4 {
		t.Fatalf("%d buffer(s) freed, want 4", d.freed-freed0)
	}
	if uint64(d.freedBytes-freedBytes0) != want {
		t.Fatalf("freed %d bytes, want %d", d.freedBytes-freedBytes0, want)
	}
	if dt.used != used0 {
		t.Fatalf("budget is %d after the release, was %d before the allocation -- the walk "+
			"freed the buffers and left the ledger high, which removes a card's budget one "+
			"migration at a time", dt.used, used0)
	}
	if d.doubleFree != 0 {
		t.Fatalf("%d double free(s): freeRec ran over a pair twice", d.doubleFree)
	}
	// A second pass must be a no-op, because both walks can reach one layer.
	dt.dropLayerLocal(l)
	if d.doubleFree != 0 {
		t.Fatalf("%d double free(s) after a second freeRec -- the two walks would corrupt "+
			"the allocator's ledger between them", d.doubleFree)
	}

	// A block with no recurrence takes nothing.
	l2 := &layer{}
	used1 := dt.used
	if !dt.addSessionRec(0, l2, pagePlan()) {
		t.Fatalf("addSessionRec declined a dense block: %s", dt.LastErr)
	}
	if l2.recOf(0) != nil || dt.used != used1 {
		t.Fatal("a dense block was given recurrent state")
	}
}

// TestGraphKeySeparatesTheRecurrentParity: two tokens that read different halves
// of the recurrent state must not share a recorded launch sequence.
//
// A linear block's state reads one half and writes the other, then flips. A
// recording made at parity 0 replayed at parity 1 would read the stale half
// again, so the state never advances and every token stays fluent. The gate is
// on the key, which decides whether a recording is reused.
func TestGraphKeySeparatesTheRecurrentParity(t *testing.T) {
	a := graphKey{lo: 0, hi: 4, scoreN: 256, rows: 1}
	b := a
	b.recParity[0] ^= 1
	if a == b {
		t.Fatal("graphKey ignores the recurrent parity: a token reading the stale half of " +
			"the state would replay the other half's recording, and the summary would " +
			"never advance past one step")
	}
	// A map keyed on it must hold both recordings.
	m := map[graphKey]int{a: 1, b: 2}
	if len(m) != 2 {
		t.Fatalf("a graph map holds %d recording(s) for two parities", len(m))
	}
	// And nothing else moved: the parity must not alias an existing axis.
	c := a
	c.kvF16 = !a.kvF16
	if c == b {
		t.Fatal("the parity and the f16 arm are the same bit")
	}
}

// TestRangeParityIsTheRangesOwn: a recording is keyed by the halves its own
// linear blocks read next, not by a device-wide flip, and by every block's
// half rather than the first's.
//
// A device holding two runs of one model (an explicit placement) submits twice
// a token. A flip per submission comes back to the same value every token while
// each block's halves alternate, so the same recording replays and reads the
// stale half. rangeParity reads each block's own half; a range with no linear
// block keys as zero and never alternates. And two blocks of one range can
// disagree -- a block placed later, or a session that stepped apart from the
// company of a step across sessions -- which a key on the first block alone
// cannot see.
func TestRangeParityIsTheRangesOwn(t *testing.T) {
	d := &fakeDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(g.Close)
	dt := g.devs[0]
	a, b, c := &recPair{}, &recPair{}, &recPair{}
	dt.layers = map[int]*layer{
		1: {linear: true, rec: map[uint64]*recPair{0: a}},
		2: {},
		3: {linear: true, rec: map[uint64]*recPair{0: c}},
		5: {linear: true, rec: map[uint64]*recPair{0: b}},
	}
	par := func(lo, hi int) parityMask {
		t.Helper()
		m, ok := dt.rangeParity(0, lo, hi)
		if !ok {
			t.Fatalf("[%d,%d) is too long to key", lo, hi)
		}
		return m
	}
	if !dt.hasRecurrent(0, 4) || dt.hasRecurrent(2, 3) {
		t.Fatal("hasRecurrent does not follow the linear blocks")
	}
	if par(0, 4) != (parityMask{}) || par(4, 6) != (parityMask{}) {
		t.Fatal("a fresh pair reads half 0")
	}
	// One token: both runs step, each block's halves flip once.
	a.cur, b.cur, c.cur = 1, 1, 1
	if par(0, 4) != (parityMask{0b11}) || par(4, 6) != (parityMask{1}) {
		t.Fatal("after a step each run keys on its own flipped halves")
	}
	// Only the first run stepped (the second was refused): they now differ,
	// which one device-wide bit cannot express.
	a.cur, c.cur = 0, 0
	if par(0, 4) != (parityMask{}) || par(4, 6) != (parityMask{1}) {
		t.Fatal("two runs on one device share a parity")
	}
	// Two blocks of one range at different halves key apart from both at
	// either half.
	c.cur = 1
	if m := par(0, 4); m == (parityMask{}) || m == (parityMask{0b11}) {
		t.Fatal("a range keys on its first block's half alone")
	}
	if par(2, 3) != (parityMask{}) {
		t.Fatal("a range with no linear block keys as half 1")
	}
}

package tier

import (
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
)

// TestHistoryComesHomeFromAFullCard is the small card a server parks rows on:
// a budget past what the driver has free (a 4 GiB RTX 3050 Ti beside a
// desktop, its budget set 1 GiB past placement), so the card is full while
// the tier's arithmetic still has room. Two sessions hold history on the one
// block; then the card fills. A park reads a session's history home, and the
// read used to ask the card for a gather buffer first: refused, the history
// stayed and the park failed ("park: N block(s) stay on the device"), which
// failed the request (server.TestSixtyFourRequestsWhoseHistoryStreams). It
// now comes home through the pool's reserve. And the driver's refusal of a
// growth the budget allowed marks the card tight: KVRoom counts no growth
// past what the driver reports free (this fake reports nothing, so none), and
// a scheduler holds rows back rather than admitting history the card cannot
// take.
func TestHistoryComesHomeFromAFullCard(t *testing.T) {
	p := fakePlan()
	d := &fakeDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff),
		WithConfig(func(c *Config) { c.KVPage = 64 }))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(g.Close)
	w := blocks(p, 1)[0]
	const pos = 200
	kvDim := p.NKVHead * p.HeadDim
	k, v := make([]float32, pos*kvDim), make([]float32, pos*kvDim)
	a, b := g.Attach(), g.Attach()
	for _, s := range []nn.LayerDevice{a, b} {
		if !s.PrepLayer(0, p, w) {
			t.Fatalf("PrepLayer: %s", g.Err())
		}
		if !s.ReserveKV(pos) || !s.MigrateKV(0, k, v, pos, true) {
			t.Fatalf("a session's history did not go up: %s", g.Err())
		}
	}
	room0, page := g.KVRoom()
	// Something else takes the rest of the card.
	d.capBytes = d.allocBytes - d.freedBytes
	// a grows: the budget has room, the driver has none.
	a.ReserveKV(pos + 4*page)
	room1, _ := g.KVRoom()
	kp := g.devs[0].kvp
	free := 0
	for _, l := range kp.layers {
		free = len(l.free) + len(l.fenced)
	}
	t.Logf("room %d positions before the card filled, %d after the driver refused a growth (%d free pages of %d positions)",
		room0, room1, free, page)
	if room1 > max(free-1, 0)*page {
		t.Errorf("KVRoom reports %d positions on a card the driver refused to grow, past its %d free page(s): "+
			"a scheduler admits history the card cannot take", room1, free)
	}
	// b parks: its history comes home with no buffer to spare for the read.
	reads := d.reads
	if !b.MigrateKV(0, k, v, pos, false) {
		t.Fatalf("on a full card a session's history would not come home: %s", g.Err())
	}
	st := g.Stats()
	if st.KVStageReads == 0 || d.reads == reads {
		t.Fatalf("the history came home with %d reserve read(s) and %d readback(s): the full card was not what it read through",
			st.KVStageReads, d.reads-reads)
	}
	if live := d.allocBytes - d.freedBytes; live > d.capBytes {
		t.Fatalf("the card holds %d bytes past its %d", live, d.capBytes)
	}
}

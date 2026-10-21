package tier

import (
	"testing"
)

// TestPagedStagingAllocatesNothing holds a token's paged bookkeeping to zero Go
// heap allocations once the pool is warm: building the call's rows, taking the
// pages its positions need, readying the scratch and staging the descriptors.
// Every per-token allocation is garbage the collector has to find, and a
// serving process pays for it in pauses; a capacity change (a new page, a
// longer table, a new split count) may allocate, a token must not.
func TestPagedStagingAllocatesNothing(t *testing.T) {
	p := fakePlan()
	g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	ws := blocks(p, 2)
	for li := range ws {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("PrepLayer(%d): %s", li, g.Err())
		}
	}
	d := g.devs[0]
	if !d.paged || d.bs == nil || d.bs.pkv == nil {
		t.Fatalf("the device is not paged (paged %v, scratch %v): this test would prove nothing", d.paged, d.bs != nil)
	}
	bs := d.bs
	pos := 5
	token := func() {
		d.pgRows = d.pagedRows(0, d.pgRows, 1, 1, pos, nil)
		if !d.pagedAppendRows(0, d.pgRows) {
			t.Fatal(d.LastErr)
		}
		d.mu.Lock()
		bs.pkv.rows = d.pagedRows(0, bs.pkv.rows, 1, 1, pos, nil)
		if err := d.pagedPrep(0, bs, bs.pkv.rows); err != nil {
			d.mu.Unlock()
			t.Fatal(err)
		}
		d.pagedStage(bs.pkv, &bs.p, bs.pkv.rows)
		d.mu.Unlock()
	}
	token() // warm: the first page, the table run, the variant
	if len(bs.pkv.desc) == 0 {
		t.Fatal("nothing was staged: the measurement below would be of nothing")
	}
	if n := testing.AllocsPerRun(100, token); n != 0 {
		t.Fatalf("a warm token's paged staging allocates %.1f times", n)
	}
}

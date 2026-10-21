package tier

import (
	"encoding/binary"
	"errors"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// TestKVPoolPages holds the device page pool to its contract
// (docs/design/device-kv-paging.md): ids are each layer's own, never alias
// between sequences in a layer and never hand out the dummy; a sequence's run
// has one offset in every layer's table; a released page is not reused before
// the next completed submission; growth is all or nothing, refused with
// ErrKVCapacity by the budget or the backend's buffer limit with nothing
// changed; a shrink compacts live pages whatever ids they held, so a hole
// below live pages is room again; releasing a sequence in one layer leaves the
// others alone; the charge is always the layers' sizes, their read reserves
// with them; and every buffer comes back.
func TestKVPoolPages(t *testing.T) {
	dev := &fakeDev{}
	g, err := New([]Slot{{Dev: dev, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatal(err)
	}
	d := g.devs[0]
	d.mu.Lock()
	defer d.mu.Unlock()

	const p = 64
	kp := newKVPool(p, 0)
	used0 := d.used
	for li, geom := range []kvGeom{{kvRow: 16}, {kvRow: 16, f16: true}, {kvRow: 24, mla: true}} {
		if err := d.addKVLayer(kp, li, geom, 1); err != nil {
			t.Fatal(err)
		}
	}
	charged := func() {
		t.Helper()
		var want uint64
		for _, l := range kp.layers {
			want += kp.pageBytes(l)*uint64(l.n) + uint64(kp.tabWords)*4 + kvStageBytes(l.geom, p)
		}
		if kp.charged() != want || d.used-used0 != want {
			t.Fatalf("the layers say %d charged, the device %d, want %d", kp.charged(), d.used-used0, want)
		}
	}
	distinct := func(li int) {
		t.Helper()
		l := kp.layers[li]
		seen := map[uint32]seqID{}
		for s, ids := range l.owned {
			for _, id := range ids {
				if id == 0 || int(id) >= l.n {
					t.Fatalf("layer %d gave %v page %d of %d", li, s, id, l.n)
				}
				if o, ok := seen[id]; ok {
					t.Fatalf("layer %d page %d given to both %v and %v", li, id, o, s)
				}
				seen[id] = s
			}
		}
		for _, id := range l.free {
			if _, ok := seen[id]; ok || id == 0 || int(id) >= l.n {
				t.Fatalf("layer %d page %d is free and owned, the dummy, or out of range", li, id)
			}
		}
	}
	charged()
	a, b, c := seqID{1, 0}, seqID{2, 0}, seqID{2, 64}
	// Interleaved, so a's and b's ids alternate: a hole in the middle is what
	// a tail-only shrink cannot give back.
	for i := 0; i < 3; i++ {
		for _, s := range []seqID{a, b} {
			for _, li := range []int{0, 1} {
				if err := d.allocPages(0, kp, kp.layers[li], s, 1); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	for li := range kp.layers {
		distinct(li)
	}
	if len(kp.layers[2].owned) != 0 {
		t.Fatal("layer 2 was never asked and holds pages")
	}
	if kp.ranges[a].off == kp.ranges[b].off || kp.ranges[a].off == 0 {
		t.Fatalf("runs a %+v b %+v: the dummy entry or one run for two sequences", kp.ranges[a], kp.ranges[b])
	}
	charged()

	// A released page is fenced: the next allocation cannot reuse it.
	l0 := kp.layers[0]
	aPages := slices.Clone(l0.owned[a])
	l0.releasePages(a)
	if err := d.allocPages(0, kp, l0, c, len(aPages)); err != nil {
		t.Fatal(err)
	}
	for _, id := range l0.owned[c] {
		if slices.Contains(aPages, id) {
			t.Fatalf("page %d reused before a completed submission: a step in flight may still read it", id)
		}
	}
	l0.fence()
	for _, id := range aPages {
		if !slices.Contains(l0.free, id) {
			t.Fatalf("page %d released and fenced, not free", id)
		}
	}
	// Layer 1 was not asked: a's pages there are untouched.
	if len(kp.layers[1].owned[a]) != 3 {
		t.Fatalf("releasing a in layer 0 took its pages in layer 1: %v", kp.layers[1].owned[a])
	}
	charged()

	// The backend's limit: nothing changes when the layer cannot grow.
	n, free, owned := l0.n, slices.Clone(l0.free), len(l0.owned)
	kp.maxBytes = uint64(l0.n * p * 16 * 4)
	err = d.allocPages(0, kp, l0, a, len(l0.free)+1)
	if !errors.Is(err, ErrKVCapacity) || l0.n != n || !slices.Equal(l0.free, free) || len(l0.owned) != owned {
		t.Fatalf("a layer at the backend's limit: err %v, pages %d -> %d, free %d -> %d",
			err, n, l0.n, len(free), len(l0.free))
	}
	kp.maxBytes = 0
	// The budget: likewise.
	d.limit = d.used + 1
	err = d.allocPages(0, kp, l0, a, len(l0.free)+1)
	if !errors.Is(err, ErrKVCapacity) || l0.n != n || !slices.Equal(l0.free, free) {
		t.Fatalf("a layer over the budget: err %v, pages %d -> %d", err, n, l0.n)
	}
	d.limit = 1 << 30
	charged()

	// A shrink compacts: b goes, its pages sat between c's and the free ones,
	// and the layer comes down to exactly what c holds and the dummy.
	l0.releasePages(b)
	l0.fence()
	cPages := len(l0.owned[c])
	if err := d.shrinkKVLayer(kp, l0); err != nil {
		t.Fatal(err)
	}
	if l0.n != 1+cPages {
		t.Fatalf("compacted to %d pages with %d live", l0.n, cPages)
	}
	distinct(0)
	charged()
	// And never below the working set.
	l0.floor = l0.n + 2
	if err := d.growKVLayer(kp, l0, l0.floor); err != nil {
		t.Fatal(err)
	}
	if err := d.shrinkKVLayer(kp, l0); err != nil || l0.n != l0.floor {
		t.Fatalf("shrunk to %d pages under a floor of %d (%v)", l0.n, l0.floor, err)
	}
	charged()

	d.freeKVPool(kp)
	if d.used != used0 {
		t.Fatalf("the pool freed and %d bytes are still charged", d.used-used0)
	}
	d.mu.Unlock()
	g.Close()
	d.mu.Lock()
	if len(dev.liveBufs) != 0 || dev.doubleFree != 0 {
		t.Fatalf("after Close: %d buffer(s) live, %d freed twice", len(dev.liveBufs), dev.doubleFree)
	}
}

// TestKVPoolKeepsPagesAcrossAResize writes distinct words into every owned page
// of three layer shapes -- f32, packed f16 and MLA -- on every real device,
// grows each layer past its size and compacts it back after a sequence whose
// pages sit between the others' is released, and demands both that each page
// reads back word for word through its sequence's run in the layer's table,
// and that the table holds exactly the ids the host has. A compaction that
// copied pages in some other order, copied short, or left a table naming the
// old ids has a page to lose.
func TestKVPoolKeepsPagesAcrossAResize(t *testing.T) {
	devs := realDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host -- this gate proved nothing")
	}
	const p = 64
	for _, dv := range devs {
		t.Run(dv.API(), func(t *testing.T) {
			g, err := New([]Slot{{Dev: dv, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
			if err != nil {
				t.Fatal(err)
			}
			defer g.Close()
			d := g.devs[0]
			d.mu.Lock()
			defer d.mu.Unlock()
			kp := newKVPool(p, 0)
			defer d.freeKVPool(kp)
			geoms := []kvGeom{{kvRow: 96}, {kvRow: 96, f16: true}, {kvRow: 72, mla: true}}
			for li, gm := range geoms {
				if err := d.addKVLayer(kp, li, gm, 1); err != nil {
					t.Fatal(err)
				}
			}
			a, b, c := seqID{1, 0}, seqID{2, 0}, seqID{3, 0}
			// Interleaved one page at a time, so c's pages end up between a's
			// and b's in every layer.
			for i := 0; i < 4; i++ {
				for _, s := range []seqID{a, c, b} {
					for li := range geoms {
						if err := d.allocPages(0, kp, kp.layers[li], s, 1); err != nil {
							t.Fatal(err)
						}
					}
				}
			}
			// A word is keyed by the sequence and its page's position in it, not
			// by the id, which a compaction changes.
			word := func(li int, s seqID, j, half, i int) uint32 {
				return uint32((li*7919+int(s.sid)*104729+j*31+half*17)*1_000_003+i) * 2654435761
			}
			type bufWords struct {
				li, half, words int
			}
			var halves []bufWords
			for li, gm := range geoms {
				halves = append(halves, bufWords{li, 0, gm.kWords(p)})
				if gm.vWords(p) > 0 {
					halves = append(halves, bufWords{li, 1, gm.vWords(p)})
				}
			}
			buf := func(h bufWords) backend.Buf {
				if h.half == 0 {
					return kp.layers[h.li].k
				}
				return kp.layers[h.li].v
			}
			for _, h := range halves {
				for _, s := range []seqID{a, b} {
					for j, id := range kp.layers[h.li].owned[s] {
						w := make([]byte, h.words*4)
						for i := 0; i < h.words; i++ {
							binary.LittleEndian.PutUint32(w[4*i:], word(h.li, s, j, h.half, i))
						}
						if err := buf(h).WriteAt(int(id)*h.words*4, w); err != nil {
							t.Fatal(err)
						}
					}
				}
			}
			check := func(when string) {
				t.Helper()
				// allocPages queues its table writes (flushTabs); a reader of
				// the tables flushes first, as every one in the tier does.
				if err := d.flushTabs(); err != nil {
					t.Fatal(err)
				}
				for _, h := range halves {
					l := kp.layers[h.li]
					all := make([]byte, l.n*h.words*4)
					if err := buf(h).Read(all); err != nil {
						t.Fatal(err)
					}
					tab := make([]byte, kp.tabWords*4)
					if err := l.tab.Read(tab); err != nil {
						t.Fatal(err)
					}
					for _, s := range []seqID{a, b} {
						r := kp.ranges[s]
						for j, id := range l.owned[s] {
							if got := binary.LittleEndian.Uint32(tab[(r.off+j)*4:]); got != id {
								t.Fatalf("%s: layer %d's table says %v page %d is %d, the host says %d",
									when, h.li, s, j, got, id)
							}
							for i := 0; i < h.words; i++ {
								got := binary.LittleEndian.Uint32(all[(int(id)*h.words+i)*4:])
								if want := word(h.li, s, j, h.half, i); got != want {
									t.Fatalf("%s: layer %d %v page %d (id %d) half %d word %d reads %#x, want %#x",
										when, h.li, s, j, id, h.half, i, got, want)
								}
							}
						}
					}
				}
			}
			check("written")
			n := kp.layers[0].n
			for li := range geoms {
				if err := d.allocPages(0, kp, kp.layers[li], c, n+8); err != nil {
					t.Fatal(err)
				}
			}
			check("after growing")
			big := kp.layers[0].n
			for li := range geoms {
				l := kp.layers[li]
				l.releasePages(c)
				l.fence()
				if err := d.shrinkKVLayer(kp, l); err != nil {
					t.Fatal(err)
				}
				if want := 1 + len(l.owned[a]) + len(l.owned[b]); l.n != want {
					t.Fatalf("layer %d compacted to %d pages, want %d", li, l.n, want)
				}
			}
			check("after compacting")
			t.Logf("%d -> %d -> %d pages a layer, every live page intact through its table", n, big, kp.layers[0].n)
		})
	}
}

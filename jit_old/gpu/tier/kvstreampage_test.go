package tier

import "testing"

// TestKVShrinkKeepsTheStreamPage: a compaction forced by another layer's
// growth leaves a layer whose sequence has pages at home one free id to
// stream them through. Compacted to exactly what is live, the layer had none
// and the next call refused with "no free page to stream" (a 4096-token
// prompt on a card holding five of Qwen3.8-Flash-Next's blocks). Without
// pages at home the same layer compacts to what it holds, so the second arm
// shows the reserve is not kept for nothing and the first fails when the
// reserve is removed.
func TestKVShrinkKeepsTheStreamPage(t *testing.T) {
	for _, tc := range []struct {
		atHome bool
		free   int
	}{{true, 1}, {false, 0}} {
		dev := &fakeDev{}
		g, err := New([]Slot{{Dev: dev, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatal(err)
		}
		d := g.devs[0]
		d.mu.Lock()
		kp := newKVPool(64, 0)
		if err := d.addKVLayer(kp, 0, kvGeom{kvRow: 16}, 1); err != nil {
			t.Fatal(err)
		}
		l := kp.layers[0]
		s := seqID{1, 0}
		if err := d.allocPages(0, kp, l, s, 6); err != nil {
			t.Fatal(err)
		}
		if err := d.growKVLayer(kp, l, 12); err != nil {
			t.Fatal(err)
		}
		if tc.atHome {
			kp.evicted = map[seqID]int{s: 2}
		}
		l.fence()
		if err := d.shrinkKVLayer(kp, l); err != nil {
			t.Fatal(err)
		}
		if got := len(l.free); got != tc.free {
			t.Fatalf("pages at home %v: %d free after the compaction, want %d (pool %d pages)", tc.atHome, got, tc.free, l.n)
		}
		d.mu.Unlock()
		g.Close()
	}
}

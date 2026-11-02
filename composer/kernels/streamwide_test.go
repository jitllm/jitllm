package kernels

import "testing"

// TestStreamWideRefusesARaggedWidth is the gate on the read-wall probe's own
// arithmetic.
//
// StreamWide loops per/w times and the caller divides by `per` words, so a
// ragged width would report more bandwidth than it measured (1.5x at per=3
// w=2), and per < w is Loop(0), an unguarded countdown on CUDA.
func TestStreamWideRefusesARaggedWidth(t *testing.T) {
	for _, c := range []struct {
		per, w int
		ok     bool
	}{
		{1024, 1, true}, {1024, 8, true}, // what mvbench actually runs at 1 GiB
		{3, 2, false},  // under-reads by a third and says nothing
		{1, 2, false},  // Loop(0)
		{12, 8, false}, // 8 of 12 words
		{8, 8, true},
	} {
		_, err := StreamWide(c.per, c.w)
		if c.ok && err != nil {
			t.Errorf("StreamWide(%d,%d) refused a width that divides: %v", c.per, c.w, err)
		}
		if !c.ok && err == nil {
			t.Errorf("StreamWide(%d,%d) was ACCEPTED -- it loops %d times and reads "+
				"%d of %d words while the caller times all of them",
				c.per, c.w, c.per/c.w, (c.per/c.w)*c.w, c.per)
		}
	}
	// The spread arm shares the arithmetic and must share the refusal.
	if _, err := StreamWideSpread(3, 2, 262144); err == nil {
		t.Error("StreamWideSpread accepted per=3 w=2 -- the two arms must refuse alike, " +
			"or the contiguity comparison is between a full read and a partial one")
	}
}

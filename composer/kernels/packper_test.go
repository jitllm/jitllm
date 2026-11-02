package kernels

import "testing"

// TestPackWorkersNeverShareADWord: every worker boundary packPer produces falls
// on a d-word boundary for every packed format, at every worker count. Two
// workers OR-ing into one word lose an update about one run in fifteen, so the
// arithmetic is gated rather than the lottery.
func TestPackWorkersNeverShareADWord(t *testing.T) {
	for q := Quant(0); q < Quant(len(qtab)); q++ {
		if IsFloat(q) || qtab[q].blockE == 0 {
			continue
		}
		g := DSlots(q)
		for _, nrows := range []int{256, 300, 1000, 2880, 4096, 14336} {
			for n := 2; n <= 64; n++ {
				if per := packPer(q, nrows, n); per%g != 0 {
					t.Fatalf("%v rows=%d workers=%d: %d rows a worker cuts a %d-row d word", q, nrows, n, per, g)
				}
			}
		}
	}
}

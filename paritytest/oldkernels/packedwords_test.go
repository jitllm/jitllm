package oldkernels_test

import (
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestPackedWordsMatchesPack holds the size predicate to the packer itself.
//
// PackedWords lets the tier refuse a tensor without packing it; if it
// disagreed with PackWeights the tier would silently decline work that fits
// or accept work that does not.
func TestPackedWordsMatchesPack(t *testing.T) {
	for _, f := range packedRefFormats {
		q := f.k
		for _, s := range []struct{ rows, k int }{
			{1, 256}, {7, 512}, {64, 2048}, {300, 1024}, {98304, 256},
		} {
			if s.k%int(q.Elems()) != 0 {
				continue
			}
			raw := make([]byte, s.rows*(s.k/q.Elems())*q.BlockBytes())
			wq, wd, wsc, err := kernels.PackWeights(q, raw, s.rows, s.k)
			if err != nil {
				t.Fatalf("%s %dx%d: %v", q, s.rows, s.k, err)
			}
			gq, gd, gsc, err := kernels.PackedWords(q, s.rows, s.k)
			if err != nil {
				t.Fatalf("%s %dx%d: %v", q, s.rows, s.k, err)
			}
			if gq != len(wq) || gd != len(wd) || gsc != len(wsc) {
				t.Errorf("%s %dx%d: predicted (%d,%d,%d) packed (%d,%d,%d)",
					q, s.rows, s.k, gq, gd, gsc, len(wq), len(wd), len(wsc))
			}
		}
	}
}

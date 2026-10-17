package jlm

import (
	"testing"
)

// HoldPages reads several pages in one plan and leaves each whole, pinned
// and holding its own bytes: pages cold, half filled (a range read before)
// and already whole, over several calls, so a plan reused from an earlier
// call carries nothing into the next. A warm call (every page whole)
// allocates nothing.
func TestHoldPagesReadsEveryPageWhole(t *testing.T) {
	const (
		nblocks = 6
		chunk   = uint64(4096)
		page    = 8 * chunk
	)
	f := newReal(t, nblocks, page, chunk)
	defer f.stopIO()
	base, _ := f.pageAt(2)
	if err := f.EnsureRanges(2, []Range{{Off: base + 3*chunk, N: chunk}}); err != nil {
		t.Fatal(err)
	}
	check := func(bs []int) {
		t.Helper()
		out, errs := make([]Lease, len(bs)), make([]error, len(bs))
		f.HoldPages(bs, out, errs)
		for i, b := range bs {
			if errs[i] != nil {
				t.Fatalf("block %d: %v", b, errs[i])
			}
			p := f.pages[b]
			for j := range p {
				if p[j] != byte(b+1) {
					t.Fatalf("block %d byte %d is %d, want %d", b, j, p[j], b+1)
				}
			}
			if !f.pinned(b) {
				t.Fatalf("block %d is not pinned under its lease", b)
			}
			out[i].Release()
		}
	}
	check([]int{0, 2, 4})
	check([]int{1, 3})
	check([]int{0, 1, 2, 3, 4, 5})
	bs, out, errs := []int{0, 1, 2}, make([]Lease, 3), make([]error, 3)
	if n := testing.AllocsPerRun(10, func() {
		f.HoldPages(bs, out, errs)
		for i := range out {
			out[i].Release()
		}
	}); n != 0 {
		t.Fatalf("a warm HoldPages allocates %.1f times", n)
	}
}

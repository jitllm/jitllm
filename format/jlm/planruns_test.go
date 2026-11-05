package jlm

import (
	"math/rand"
	"runtime"
	"testing"
)

// TestPlanRunsMatchesTheWindowSweep gates the run planner against the
// implementation it replaces.
//
// It is an equality gate, not a bound: the runs decide which bytes the pager
// reads, and one chunk too few leaves a matvec reading stale frame bytes as
// numbers. The oracle is the old algorithm verbatim (a bool window swept over
// the ranges' whole span), the only thing that can say "same answer".
func TestPlanRunsMatchesTheWindowSweep(t *testing.T) {
	const (
		page   = 1 << 20
		chunk  = 4096
		blocks = 4
	)
	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 400; trial++ {
		f := newFake(blocks, page, 0, Align)
		f.chunk = chunk
		b := rng.Intn(blocks)
		base, _ := f.file.pageAt(b)
		nc := int(page / chunk)

		// A partially filled page, which is the state a warm mixture is in.
		have := make([]bool, nc)
		for i := range have {
			have[i] = rng.Intn(4) == 0
		}
		f.pages[b] = make([]byte, page)
		f.filled = make([][]bool, len(f.pages))
		f.filled[b] = have

		// Ranges the way ensureExperts builds them: a handful of scattered
		// spans of a few chunks each, deliberately including zero-length ones,
		// exactly-touching pairs and duplicates.
		rs := make([]Range, 0, 16)
		for i := 0; i < 1+rng.Intn(12); i++ {
			off := uint64(rng.Intn(nc)) * chunk
			n := uint64(rng.Intn(3*chunk) + 1)
			if off+n > page {
				n = page - off
			}
			rs = append(rs, Range{Off: base + off, N: n})
			switch rng.Intn(4) {
			case 0:
				rs = append(rs, Range{Off: base + off, N: 0})
			case 1: // a span that starts exactly where the last one ended
				e := off + n
				if e < page {
					rs = append(rs, Range{Off: base + e, N: min64(chunk, page-e)})
				}
			case 2:
				rs = append(rs, rs[len(rs)-1])
			}
		}

		want := windowSweep(f, b, base, page, nc, have, rs)
		_, gotBase, got, err := f.planRuns(b, rs, nil)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		if len(want) > 0 && gotBase != base {
			t.Fatalf("trial %d: base %d, want %d", trial, gotBase, base)
		}
		if len(got) != len(want) {
			t.Fatalf("trial %d: %d run(s) %v, want %d %v", trial, len(got), got, len(want), want)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d run %d: %v, want %v (all: %v vs %v)",
					trial, i, got[i], want[i], got, want)
			}
		}
	}
}

func min64(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}

// windowSweep is planRuns' body as it stood before the sort, kept here as the
// oracle and nowhere else.
func windowSweep(f *Frames, b int, base, psize uint64, nc int, have []bool, rs []Range) []run {
	lo0, hi0 := nc, 0
	for _, r := range rs {
		if r.N == 0 {
			continue
		}
		lo, hi := f.chunkSpan(r, base, nc)
		if lo < lo0 {
			lo0 = lo
		}
		if hi > hi0 {
			hi0 = hi
		}
	}
	if lo0 >= hi0 {
		return nil
	}
	want := make([]bool, hi0-lo0)
	for _, r := range rs {
		if r.N == 0 {
			continue
		}
		lo, hi := f.chunkSpan(r, base, nc)
		for i := lo; i < hi; i++ {
			want[i-lo0] = true
		}
	}
	var runs []run
	for i := lo0; i < hi0; {
		if !want[i-lo0] || have[i] {
			i++
			continue
		}
		j := i
		for j < hi0 && want[j-lo0] && !have[j] {
			j++
		}
		runs = append(runs, run{i, j})
		i = j
	}
	return runs
}

// TestPlanRunsCostsWhatWasAskedForAndNotThePage is the gate on the change's
// whole point, and it is a cost gate rather than a correctness one.
//
// The same plan must cost the same on a bigger page: a routed selection
// brackets nearly the whole page, so a planner bounded by the ranges' span
// grows with the page. It measures bytes, not allocation count, since
// make([]bool, n) is one allocation whatever n is; against the window sweep
// restored it fails by name.
func TestPlanRunsCostsWhatWasAskedForAndNotThePage(t *testing.T) {
	const chunk = 4096
	ask := func(page uint64) uint64 {
		f := newFake(1, page, 0, Align)
		f.chunk = chunk
		nc := int(page / chunk)
		f.pages[0] = make([]byte, page)
		f.filled = [][]bool{make([]bool, nc)}
		base, _ := f.file.pageAt(0)
		// Two spans at opposite ends: the shape a routed selection has, and the
		// one that defeats a window bound.
		rs := []Range{{Off: base, N: chunk}, {Off: base + uint64(nc-1)*chunk, N: chunk}}

		const iters = 2000
		f.planRuns(0, rs, nil) // warm the scratch; the first call may grow it
		f.unpin(0)
		var a, b runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&a)
		for i := 0; i < iters; i++ {
			for c := range f.filled[0] {
				f.filled[0][c] = false
			}
			f.planRuns(0, rs, nil)
			f.unpin(0)
		}
		runtime.ReadMemStats(&b)
		return (b.TotalAlloc - a.TotalAlloc) / iters
	}
	small, large := ask(1<<20), ask(64<<20) // 256 chunks against 16384
	if large > small+128 {
		t.Fatalf("a plan for two spans costs %d B on a 1 MiB page and %d B on a "+
			"64 MiB one: the planner is sized by the PAGE, not by what was asked for",
			small, large)
	}
	t.Logf("per plan: %d B at 256 chunks, %d B at 16384 chunks", small, large)
}

//go:build linux

package sched

import (
	"slices"
	"sync/atomic"
	"testing"
)

func TestDetectCores(t *testing.T) {
	p := PCores()
	if len(p) == 0 {
		t.Skip("no CPU topology exposed")
	}
	t.Logf("physical P-cores: %v; decode pool: %v", p, DecodeCores())
	// One entry per physical core: no two siblings of the same core, or two
	// workers would share load ports and one would look like a straggler.
	seen := map[int]bool{}
	for _, c := range p {
		if seen[c] {
			t.Errorf("cpu %d listed twice", c)
		}
		seen[c] = true
	}
	// The pool takes every P-core the process may run on (RULE 5's six are
	// the taskset's to choose, not the pool's), and WithCores narrows it to
	// the first n.
	if d := DecodeCores(); !slices.Equal(d, p) {
		t.Errorf("decode pool %v is not the P-cores %v", d, p)
	}
	for _, n := range []int{1, 2, len(p), len(p) + 1} {
		k := min(n, len(p))
		if d := DecodeCores(WithCores(n)); !slices.Equal(d, p[:k]) {
			t.Errorf("WithCores(%d): decode pool %v, want the first %d of %v", n, d, k, p)
		}
	}
}

// TestDoCoversExactlyOnce is the property that matters: dynamic chunk claiming
// must cover every index exactly once. A double-claim silently doubles a row of
// the output and a missed claim silently zeroes one, and both produce fluent
// text rather than an error.
func TestDoCoversExactlyOnce(t *testing.T) {
	p := New(DecodeCores())
	defer p.Close()
	for _, tc := range []struct{ total, chunk int }{
		{1, 1}, {7, 1}, {100, 7}, {1000, 13}, {4096, 64}, {5, 64}, {0, 4},
	} {
		hits := make([]int32, tc.total)
		p.Do(tc.total, tc.chunk, func(_, lo, hi int) {
			for i := lo; i < hi; i++ {
				atomic.AddInt32(&hits[i], 1)
			}
		})
		for i, h := range hits {
			if h != 1 {
				t.Fatalf("total=%d chunk=%d: index %d claimed %d times, want 1",
					tc.total, tc.chunk, i, h)
			}
		}
	}
}

// TestDoIsReusable: the pool must survive many regions, since decode runs
// hundreds per token.
func TestDoIsReusable(t *testing.T) {
	p := New(DecodeCores())
	defer p.Close()
	var sum atomic.Int64
	for r := 0; r < 200; r++ {
		p.Do(512, 16, func(_, lo, hi int) {
			var s int64
			for i := lo; i < hi; i++ {
				s += int64(i)
			}
			sum.Add(s)
		})
	}
	want := int64(200) * (511 * 512 / 2)
	if sum.Load() != want {
		t.Errorf("sum = %d, want %d", sum.Load(), want)
	}
}

func TestNilPool(t *testing.T) {
	var p *Pool
	got := 0
	p.Do(10, 4, func(_, lo, hi int) { got += hi - lo })
	if got != 10 {
		t.Errorf("nil pool ran %d of 10 indices; it must still do the work serially", got)
	}
	p.Close()
	if p.N() != 1 {
		t.Errorf("nil pool reports %d workers, want 1", p.N())
	}
}

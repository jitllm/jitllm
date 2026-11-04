package cpu

import (
	"sync/atomic"
	"testing"

	"github.com/jitllm/jitllm/engine/sched"
)

// TestPoolCoversEveryRangeOnce runs a region over the CPU device's pool,
// unpinned and pinned to one participant, and holds every index to one visit
// by a worker below Workers; a prefill on cores wider than the decode pool's
// runs its regions there and gives GOMAXPROCS back.
func TestPoolCoversEveryRangeOnce(t *testing.T) {
	cover := func(t *testing.T, p *Pool) {
		t.Helper()
		const n = 1000
		var seen [n]atomic.Int32
		var bad atomic.Int32
		w := p.Workers()
		p.Do(n, 7, func(worker, lo, hi int) {
			if worker < 0 || worker >= w {
				bad.Add(1)
			}
			for i := lo; i < hi; i++ {
				seen[i].Add(1)
			}
		})
		if bad.Load() != 0 {
			t.Fatalf("%d ranges ran on a worker outside [0, %d)", bad.Load(), w)
		}
		for i := range seen {
			if v := seen[i].Load(); v != 1 {
				t.Fatalf("index %d visited %d times on %d workers", i, v, w)
			}
		}
	}
	p := NewPool()
	defer p.Close()
	if p.Workers() < 1 || p.Max() < p.Workers() {
		t.Fatalf("%d workers, at most %d", p.Workers(), p.Max())
	}
	cover(t, p)

	one := NewPool(WithParticipants(1))
	defer one.Close()
	if one.Workers() != 1 {
		t.Fatalf("pinned to one participant, the pool drains with %d", one.Workers())
	}
	cover(t, one)

	if p.BeginPrefillE() {
		t.Fatal("the E-cores' prefill ran on a pool without the serving split")
	}
	if all := append(sched.SMTSiblings(), sched.ECores()...); len(all) > len(sched.DecodeCores()) {
		wide := NewPool(WithPrefillCPUs(all))
		defer wide.Close()
		if wide.prefill == nil {
			t.Fatalf("a prefill on %d cores beside a decode pool of %d made no prefill pool", len(all), wide.decode.Max())
		}
		wide.BeginPrefill()
		if wide.run != wide.prefill || wide.Max() < len(all) {
			t.Fatal("BeginPrefill left the regions on the decode pool")
		}
		cover(t, wide)
		wide.EndPrefill()
		if wide.run != wide.decode || wide.procs.Held() {
			t.Fatal("EndPrefill did not return to the decode pool and GOMAXPROCS")
		}
	}
}

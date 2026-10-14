package sched

import (
	"os"
	"sort"
	"sync/atomic"
	"testing"
	"time"
)

// TestFusedPhasesCoverExactlyOnce: inside a fused region every Do still covers
// [0,total) exactly once, in order -- a phase is complete before the next is
// published -- and the region counter sees one region however many phases ran.
func TestFusedPhasesCoverExactlyOnce(t *testing.T) {
	for _, shared := range []bool{false, true} {
		var p *Pool
		if shared {
			p = Shared([]int{0, 1, 2, 3})
		} else {
			p = New([]int{0, 1, 2, 3})
		}
		p.ResetRegions()
		ph0 := p.Phases()
		if !p.Fuse() {
			t.Fatalf("shared=%v: Fuse refused a pool with one user", shared)
		}
		if !p.Fused() {
			t.Fatal("Fused is false inside the region")
		}
		const phases = 200
		hits := make([]atomic.Int32, 1000)
		for ph := 0; ph < phases; ph++ {
			total := 1 + (ph*37)%len(hits)
			p.Do(total, 1+ph%7, func(_, lo, hi int) {
				for i := lo; i < hi; i++ {
					hits[i].Add(1)
				}
			})
			for i := 0; i < len(hits); i++ {
				want := int32(0)
				if i < total {
					want = 1
				}
				if got := hits[i].Swap(0); got != want {
					t.Fatalf("shared=%v phase %d: element %d ran %d times, want %d", shared, ph, i, got, want)
				}
			}
		}
		p.Unfuse()
		if p.Fused() {
			t.Fatal("Fused is true after Unfuse")
		}
		par, _ := p.Regions()
		// Phases of total <= chunk run inline and are not phases.
		if par != 1 {
			t.Fatalf("shared=%v: %d parallel regions, want the one fused region", shared, par)
		}
		if got := p.Phases() - ph0; got < phases/2 {
			t.Fatalf("shared=%v: %d phases counted, want most of %d", shared, got, phases)
		}
		// The pool still runs ordinary regions afterwards.
		var n atomic.Int64
		p.Do(1000, 10, func(_, lo, hi int) { n.Add(int64(hi - lo)) })
		if n.Load() != 1000 {
			t.Fatalf("after Unfuse a region covered %d of 1000", n.Load())
		}
		p.Close()
	}
}

// TestFusedPhaseOrder: phase k+1 reads what every chunk of phase k wrote. A
// phase published before the last one finished would read a stale value.
func TestFusedPhaseOrder(t *testing.T) {
	p := New([]int{0, 1, 2, 3, 4, 5})
	defer p.Close()
	const n = 4096
	a := make([]int64, n)
	if !p.Fuse() {
		t.Fatal("Fuse refused")
	}
	bad := atomic.Int64{}
	for ph := 1; ph <= 300; ph++ {
		p.Do(n, 64, func(_, lo, hi int) {
			// Read a row another worker wrote in the previous phase.
			for i := lo; i < hi; i++ {
				if a[(i+n/2)%n] != int64(ph-1) {
					bad.Add(1)
				}
			}
		})
		p.Do(n, 64, func(_, lo, hi int) {
			for i := lo; i < hi; i++ {
				a[i] = int64(ph)
			}
		})
	}
	p.Unfuse()
	if bad.Load() != 0 {
		t.Fatalf("%d reads saw the phase before last: a phase began before its predecessor ended", bad.Load())
	}
}

// TestFuseRefusesASharedCrew: a crew with two users never fuses, since one
// user would hold the cores across the other's regions.
func TestFuseRefusesASharedCrew(t *testing.T) {
	a := Shared([]int{0, 1, 2})
	b := Shared([]int{0, 1, 2})
	defer a.Close()
	defer b.Close()
	if a.Fuse() {
		a.Unfuse()
		t.Fatal("Fuse opened a region on a crew with two views")
	}
}

// TestFuseBesideItsOwnWideView: a caller's prefill view on a wider core set
// (SharedBeside) takes turns with its decode view and leaves the decode view
// free to fuse -- which a host whose prefill set is wider than its decode set
// (the M4's P and E cores) never did -- while another caller's view on the
// wider set still refuses it, and closing that view gives fusion back.
func TestFuseBesideItsOwnWideView(t *testing.T) {
	narrow := Shared([]int{0, 1})
	defer narrow.Close()
	wide := SharedBeside([]int{0, 1, 2, 3}, narrow)
	defer wide.Close()
	if narrow.crew.mu != wide.crew.mu {
		t.Fatal("the two views do not take turns on one lock: this gate is not on the case it names")
	}
	if !narrow.Fuse() {
		t.Fatal("Fuse refused beside the caller's own wider view")
	}
	narrow.Unfuse()
	other := Shared([]int{0, 1, 2, 3})
	if narrow.Fuse() {
		narrow.Unfuse()
		other.Close()
		t.Fatal("Fuse opened a region while another caller holds a view on cores it takes turns with")
	}
	other.Close()
	if !narrow.Fuse() {
		t.Fatal("Fuse refused after the other caller's view closed")
	}
	narrow.Unfuse()
}

// TestFusedPhaseIsCheaperThanARegion measures what fusion is for: a phase
// against a region of the same empty body. Opt-in (JITLLM_FUSE_BENCH=1): it
// logs a measurement and asserts nothing about it (RULE 4).
func TestFusedPhaseIsCheaperThanARegion(t *testing.T) {
	if os.Getenv("JITLLM_FUSE_BENCH") == "" {
		t.Skip("set JITLLM_FUSE_BENCH=1 to measure a phase against a region")
	}
	p := New(DecodeCores())
	defer p.Close()
	if p.Max() < 2 {
		t.Skip("one core")
	}
	body := func(_, lo, hi int) {}
	const per = 2000
	region := func() time.Duration {
		st := time.Now()
		for i := 0; i < per; i++ {
			p.Do(p.N()*2, 1, body)
		}
		return time.Since(st) / per
	}
	phase := func() time.Duration {
		st := time.Now()
		p.Fuse()
		for i := 0; i < per; i++ {
			p.Do(p.N()*2, 1, body)
		}
		p.Unfuse()
		return time.Since(st) / per
	}
	region()
	phase()
	var rs []float64
	var r0, f0 time.Duration
	for i := 0; i < 21; i++ {
		r, f := region(), phase()
		r0, f0 = r0+r, f0+f
		rs = append(rs, float64(f)/float64(r))
	}
	sort.Float64s(rs)
	t.Logf("%d participants: region %v, phase %v (means); median phase/region %.3f, IQR %.3f",
		p.N(), r0/21, f0/21, rs[10], rs[15]-rs[5])
}

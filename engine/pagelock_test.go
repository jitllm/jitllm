//go:build linux

package engine

import (
	"sync"
	"testing"
)

// TestConcurrentDecodeOnAnEvictingContainer runs several instances decoding at
// once on a container whose budget evicts (three frames for eight blocks), so
// frames change hands between goroutines. Run it under -race; the token
// comparison also catches a page evicted under a live reader. It fails rather
// than skips if the budget does not evict.
func TestConcurrentDecodeOnAnEvictingContainer(t *testing.T) {
	probe, _ := hybridModelOpt(t, hyOpt{moe: true, layers: 8, manyExp: true})
	page := probe.Container.H.PageSize
	probe.Close()
	m, openOpts2 := hybridModelOpt(t, hyOpt{moe: true, layers: 8, manyExp: true, budget: 3 * page})
	defer m.Close()
	if !m.Frames.CanEvict() {
		t.Fatalf("the budget is %d bytes over 8 pages of %d and nothing is evicted: "+
			"this gate would pass on a tree with no lock", m.Frames.Budget(), page)
	}

	// Built serially, driven concurrently: building JITs concurrently is not
	// what this test is about.
	const workers = 3
	ss := make([]*Instance, workers)
	for i := range ss {
		ss[i] = testInstance(t, m, 16, openOpts2...)
		defer ss[i].Close()
	}
	// The single-threaded answer for every worker, taken first, so a page
	// evicted under a live reader shows as a wrong logit, not only a race.
	want := make([][]float32, workers)
	for w := range ss {
		s := testInstance(t, m, 16, openOpts2...)
		for k := 0; k < 4; k++ {
			l, err := s.Forward(int32(1 + w*4 + k))
			if err != nil {
				t.Fatalf("serial worker %d: %v", w, err)
			}
			want[w] = append([]float32(nil), l...)
		}
		s.Close()
	}

	got := make([][]float32, workers)
	errs := make([]error, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for k := 0; k < 4; k++ {
				l, err := ss[w].Forward(int32(1 + w*4 + k))
				if err != nil {
					errs[w] = err
					return
				}
				got[w] = append(got[w][:0], l...)
			}
		}(w)
	}
	wg.Wait()
	for w, err := range errs {
		if err != nil {
			t.Fatalf("worker %d: %v", w, err)
		}
	}
	for w := range got {
		for i := range want[w] {
			if got[w][i] != want[w][i] {
				t.Fatalf("worker %d logit %d: %v concurrently, %v alone -- a page "+
					"changed hands under a live reader", w, i, got[w][i], want[w][i])
			}
		}
	}
}

// TestConcurrentDecodeOnAFittingMixture is the other half: a mixture whose
// budget holds everything, so CanEvict is false and NOTHING serialises the
// sessions. An expert's view is bound per token against its expert page, and a
// block page-in must not blank a view another session's MoE is reading. Run
// under -race.
func TestConcurrentDecodeOnAFittingMixture(t *testing.T) {
	m, openOpts := hybridModelOpt(t, hyOpt{moe: true, layers: 8, manyExp: true})
	defer m.Close()
	if m.Frames.CanEvict() {
		t.Fatal("the budget evicts: this is the serialised case, not the one this gate is for")
	}
	if l := &m.Layers[0]; l.Gate.E == nil || !m.Container.ExpertPaged(l.Gate.E) {
		t.Fatal("the fixture's banks are not in expert pages: this gate proved nothing")
	}
	const workers = 3
	ss := make([]*Instance, workers)
	for i := range ss {
		ss[i] = testInstance(t, m, 16, openOpts...)
		defer ss[i].Close()
	}
	want := make([][]float32, workers)
	for w := range ss {
		s := testInstance(t, m, 16, openOpts...)
		for k := 0; k < 4; k++ {
			l, err := s.Forward(int32(1 + w*4 + k))
			if err != nil {
				t.Fatalf("serial worker %d: %v", w, err)
			}
			want[w] = append([]float32(nil), l...)
		}
		s.Close()
	}
	got := make([][]float32, workers)
	errs := make([]error, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for k := 0; k < 4; k++ {
				l, err := ss[w].Forward(int32(1 + w*4 + k))
				if err != nil {
					errs[w] = err
					return
				}
				got[w] = append(got[w][:0], l...)
			}
		}(w)
	}
	wg.Wait()
	for w, err := range errs {
		if err != nil {
			t.Fatalf("worker %d: %v", w, err)
		}
	}
	for w := range got {
		for i := range want[w] {
			if got[w][i] != want[w][i] {
				t.Fatalf("worker %d logit %d: %v concurrently, %v alone", w, i, got[w][i], want[w][i])
			}
		}
	}
}

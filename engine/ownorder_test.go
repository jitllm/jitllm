package engine

import (
	"sync"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestInstancesPageInTheirOwnOrder holds two instances of one model on one
// CPU device, each under a quota of half its text pages, to paging in an
// order of their own: each decodes as one alone under the same quota, bit
// for bit, through a view of its own over the frames they share
// (jlm.Frames.View), and each holds no more than its quota. Two instances
// paging through one view -- the frames' one order -- is the violation.
func TestInstancesPageInTheirOwnOrder(t *testing.T) {
	for _, name := range transitionModels() {
		t.Run(name, func(t *testing.T) {
			path := jlmOf(t, models()[name])
			ids := []int32{1, 7, 42, 3, 99, 5, 17, 2}
			m, err := model.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			h := m.Container.H
			quota := uint64(h.PageSize) * uint64(max(h.NBlocks/2, 1))
			solo := testInstance(t, m, len(ids)+1, WithPageBudget(quota))
			want := teacherForced(t, solo, ids)
			solo.Close()
			cpu := sharedCPU(t)
			x := testInstance(t, m, len(ids)+1, cpu, WithPageBudget(quota))
			defer x.Close()
			y := testInstance(t, m, len(ids)+1, cpu, WithPageBudget(quota))
			defer y.Close()
			var got [2][][]float32
			var wg sync.WaitGroup
			for i, in := range []*Instance{x, y} {
				wg.Add(1)
				go func() { defer wg.Done(); got[i] = teacherForced(t, in, ids) }()
			}
			wg.Wait()
			for i := range got {
				if at := sameRows(got[i], want); at >= 0 {
					t.Errorf("instance %d beside the other parts from one alone at position %d", i, at)
				}
			}
			if x.fr == y.fr {
				t.Fatal("VIOLATION: two instances on one CPU device page through one view of the frames")
			}
			hx, hy := x.fr.Held(), y.fr.Held()
			t.Logf("two instances under %d bytes each: views of their own holding %d and %d bytes", quota, hx, hy)
			if hx > quota || hy > quota {
				t.Fatalf("an instance holds past its quota of %d bytes: %d and %d", quota, hx, hy)
			}
		})
	}
}

package model

import (
	"fmt"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// derivedTable is the table block history pg's pages are in, laid out from
// the pages themselves: page j of sequence q names the arena slot pg.k[j]
// starts, or slot 0.
func derivedTable(pg *kvPages, nseq int) []uint32 {
	slots, a := pg.t.Slots(), pg.t.Arena()
	t := make([]uint32, slots*nseq)
	for j := range slots {
		i := 0
		if j < len(pg.k) {
			if si, ok := a.SlotOf(pg.k[j]); ok {
				i = si
			}
		}
		for q := range nseq {
			t[q*slots+j] = uint32(i*nseq + q)
		}
	}
	return t
}

// cpuTablesWrong is the first CPU block of s whose view's table names another
// slot than its pages are in, and how many blocks it compared.
func cpuTablesWrong(s *Instance) (error, int) {
	nseq := max(s.nseq, 1)
	n := 0
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		pg := &s.kv.layers[li]
		if pg.t == nil {
			continue
		}
		if s.Session.view.At(li) != pg.t {
			return fmt.Errorf("block %d's table is not its session's view's", li), n
		}
		if got, want := pg.t.Pages(nseq), derivedTable(pg, nseq); !slices.Equal(got, want) {
			return fmt.Errorf("block %d's table is %v, its pages are in %v", li, got, want), n
		}
		n++
	}
	return nil, n
}

func checkCPUTables(t *testing.T, s *Instance, when string) {
	t.Helper()
	err, n := cpuTablesWrong(s)
	if err != nil {
		t.Fatalf("%s: %v", when, err)
	}
	if n == 0 {
		t.Fatalf("%s: no block keeps a table in the CPU device's view, so nothing was compared", when)
	}
}

// TestCPUPageTablesFollowThePages: the table a composed CPU block reads its
// history through is its session's view's (cpudev.View), kept as each page
// changes (cpudev.Table.Set) rather than laid out again for every call. After
// every way a page changes it names the slot each page is in: pages grown by
// a prompt and a decode, a prefix restored from the shared store, a shared
// page made private before a write, pages evicted under a budget and faulted
// back from a store. A page swapped behind the view's back is seen.
func TestCPUPageTablesFollowThePages(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	defer m.setKVPageForTest(sharePage)()

	st := NewSharedStore()
	a := shareSession(t, m, 0, st, true)
	defer a.s.Close()
	checkCPUTables(t, a.s, "the first session, its pages grown and published")
	b := shareSession(t, m, 1, st, true)
	defer b.s.Close()
	if b.s.KVRestored() == 0 {
		t.Fatal("the second session restored nothing from the shared store")
	}
	checkCPUTables(t, b.s, "a session restoring the shared prefix")

	li := firstAttn(b.s)
	pg := &b.s.kv.layers[li]
	if !pg.isShared(0) {
		t.Fatal("the second session's first page is not shared: nothing is made private")
	}
	b.s.cpuWritable(li, 3)
	if pg.isShared(0) {
		t.Fatal("a write's page stayed shared")
	}
	checkCPUTables(t, b.s, "a shared page made private")

	ids := sharePrompt(m, 2)
	s := m.NewInstance(len(ids) + 16)
	defer s.Close()
	s.SetKVStore(NewMemStore())
	pp := 0
	for li := range s.kv.layers {
		pp += 2 * 4 * 2 * s.kv.layers[li].pp
	}
	mustBudget(t, s, uint64(pp))
	lg, err := s.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	checkCPUTables(t, s, "a prompt under a two-page budget")
	for i := range 8 {
		if lg, err = s.Forward(int32(argmax(lg))); err != nil {
			t.Fatal(err)
		}
		checkCPUTables(t, s, fmt.Sprintf("decode step %d under the budget", i))
	}
	if s.kv.evicted == 0 || s.kv.fetched == 0 {
		t.Fatalf("under a two-page budget %d pages were evicted and %d faulted back: nothing paged", s.kv.evicted, s.kv.fetched)
	}

	// The violation: a page made another slot without the view told.
	pg = &s.kv.layers[firstAttn(s)]
	j := slices.IndexFunc(pg.k, func(k []float32) bool { return k != nil && pg.inArena(0, k) })
	if j < 0 {
		t.Fatal("no page of the first attending block is in its arena")
	}
	k2, _ := pg.t.Take()
	if k2 == nil {
		t.Fatal("the arena had no slot to swap in")
	}
	old := pg.k[j]
	pg.k[j] = k2
	err, _ = cpuTablesWrong(s)
	pg.k[j] = old
	pg.t.Let(k2)
	if err == nil {
		t.Fatal("a page swapped behind the view's back was not seen")
	}
	t.Logf("%d pages evicted and %d faulted back, every table where its pages are; a swapped page seen: %v", s.kv.evicted, s.kv.fetched, err)
}

package engine

import (
	"fmt"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// derivedTable is the table block history pg's pages are in, laid out from
// the pages themselves: page j of sequence q names the arena slot page j's
// key starts, or slot 0.
func derivedTable(pg *kvPages, nseq int) []uint32 {
	slots, a := pg.arena().Slots(), pg.arena().Arena()
	t := make([]uint32, slots*nseq)
	for j := range slots {
		i := 0
		if si, ok := a.SlotOf(pg.mem.K(j)); ok {
			i = si
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
		if pg.arena() == nil {
			continue
		}
		if s.Session.view.At(li) != pg.arena() {
			return fmt.Errorf("block %d's table is not its session's view's", li), n
		}
		if got, want := pg.arena().Pages(nseq), derivedTable(pg, nseq); !slices.Equal(got, want) {
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
// back from a store. A table told another slot than its page is in is seen.
func TestCPUPageTablesFollowThePages(t *testing.T) {
	m, err := model.Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	// Both sessions on one CPU device: the second restores the first's
	// pages, slots of that device's arena.
	cpu := sharedCPU(t)
	st := NewSharedStore()
	a := shareSession(t, m, 0, st, true, withKVPage(sharePage), cpu)
	defer a.s.Close()
	checkCPUTables(t, a.s, "the first session, its pages grown and published")
	b := shareSession(t, m, 1, st, true, withKVPage(sharePage), cpu)
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
	s := testInstance(t, m, len(ids)+16, withKVPage(sharePage))
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

	// The violation: the table told another slot than the page is in.
	pg = &s.kv.layers[firstAttn(s)]
	j := -1
	for n := range pg.mem.Len() {
		if pg.mem.ArenaPage(n) {
			j = n
			break
		}
	}
	if j < 0 {
		t.Fatal("no page of the first attending block is in its arena")
	}
	k2, _ := pg.mem.Take()
	if k2 == nil {
		t.Fatal("the arena had no slot to name instead")
	}
	pg.arena().Set(j, k2)
	err, _ = cpuTablesWrong(s)
	pg.arena().Set(j, pg.mem.K(j))
	pg.mem.Give(k2)
	if err == nil {
		t.Fatal("a table naming another slot than its page is in was not seen")
	}
	t.Logf("%d pages evicted and %d faulted back, every table where its pages are; a wrong table seen: %v", s.kv.evicted, s.kv.fetched, err)
}

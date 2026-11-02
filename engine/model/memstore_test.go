package model

import (
	"bytes"
	"errors"
	"io"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestBoundedMemStoreEvictsTheOldestTailFirst holds a MemStore to its limit:
// it never holds more than it was given, it drops from the pass used longest
// ago (a Get counts as use) and, within it, the page furthest into the
// prompt, and its byte count goes back to zero when everything is dropped.
func TestBoundedMemStoreEvictsTheOldestTailFirst(t *testing.T) {
	page := bytes.Repeat([]byte{7}, 4096)
	m := NewBoundedMemStore(3 * 4096)
	defer m.Close()
	for i := 0; i < 3; i++ {
		if err := m.Set("c", 0, i, bytes.NewReader(page)); err != nil {
			t.Fatal(err)
		}
	}
	var sink bytes.Buffer
	// Page 0 is used again in a new pass, so pages 1 and 2 are the oldest
	// pass, and 2 is its tail.
	if err := m.Get("c", 0, 0, &sink); err != nil {
		t.Fatal(err)
	}
	if err := m.Set("c", 0, 3, bytes.NewReader(page)); err != nil {
		t.Fatal(err)
	}
	if got := m.Bytes(); got > 3*4096 {
		t.Fatalf("the store holds %d bytes past its %d limit", got, 3*4096)
	}
	if _, ok := m.pages[kvStoreKey{"c", 0, 2}]; ok {
		t.Fatal("page 2, the oldest pass's tail, is still held")
	}
	for _, i := range []int{0, 1, 3} {
		sink.Reset()
		if err := m.Get("c", 0, i, &sink); err != nil {
			t.Fatalf("page %d was evicted instead of the oldest pass's tail: %v", i, err)
		}
		if !bytes.Equal(sink.Bytes(), page) {
			t.Fatalf("page %d came back changed", i)
		}
	}
	st := m.Stats()
	if st.Evicted != 1 || st.Pages != 3 {
		t.Fatalf("stats %+v: want one eviction and three pages", st)
	}
	m.SetLimit(4096)
	if st := m.Stats(); st.Pages != 1 || st.Bytes != 4096 {
		t.Fatalf("lowering the limit left %+v", st)
	}
	if _, ok := m.pages[kvStoreKey{"c", 0, 0}]; !ok {
		t.Fatal("lowering the limit kept a page other than the prompt's head")
	}
	if err := m.Drop("c"); err != nil {
		t.Fatal(err)
	}
	if m.Bytes() != 0 {
		t.Fatalf("an emptied store still counts %d bytes", m.Bytes())
	}
	// A page bigger than the whole limit is not held, rather than evicting
	// everything for nothing.
	if err := m.Set("c", 0, 9, bytes.NewReader(bytes.Repeat([]byte{1}, 8192))); err != nil {
		t.Fatal(err)
	}
	if m.Bytes() != 0 {
		t.Fatal("a page larger than the limit was held")
	}
}

// TestABoundedStoreChangesNoToken runs a prompt twice through a store bounded
// below what the first pass wrote: the second pass restores less than an
// unbounded store would (so the bound bit), and generates the same tokens as
// a run with no store at all.
func TestABoundedStoreChangesNoToken(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const maxSeq, gen, page = 256, 12, 8
	prompt := make([]int32, 0, 70)
	for i := 0; i < 70; i++ {
		prompt = append(prompt, int32(300+i*7))
	}
	run := func(st KVStore) ([]int32, int) {
		defer m.setKVPageForTest(page)()
		s := m.NewInstance(maxSeq)
		defer s.Close()
		var lg []float32
		var err error
		if st != nil {
			s.SetKVStore(st)
			mustKey(t, s, "tinyllama/bounded")
			lg, err = s.PrefillCached(prompt)
		} else {
			lg, err = s.Prefill(prompt)
		}
		if err != nil {
			t.Fatal(err)
		}
		out := make([]int32, 0, gen)
		for i := 0; i < gen; i++ {
			id := int32(argmax(lg))
			out = append(out, id)
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return out, s.KVRestored()
	}
	want, _ := run(nil)

	whole := NewMemStore()
	defer whole.Close()
	run(whole)
	_, all := run(whole)
	if all == 0 {
		t.Fatal("an unbounded store restored nothing: the gate below would compare two cold runs")
	}

	limit := whole.Bytes() / 3
	bounded := NewBoundedMemStore(limit)
	defer bounded.Close()
	run(bounded)
	got, some := run(bounded)
	if some >= all {
		t.Fatalf("a store bounded to a third restored %d positions, an unbounded one %d: the bound never bit", some, all)
	}
	if some == 0 {
		t.Fatal("the bounded store restored nothing: it evicted the prompt's head and kept pages no restore can reach")
	}
	if bounded.Stats().Evicted == 0 {
		t.Fatal("the bounded store evicted nothing")
	}
	if bounded.Bytes() > limit {
		t.Fatalf("the bounded store holds %d bytes past its %d limit", bounded.Bytes(), limit)
	}
	sameTokens(t, "bounded store", want, got)
	t.Logf("unbounded restored %d, bounded %d of %d positions; tokens identical", all, some, len(prompt))
}

// TestAnInterruptedPrefillStopsBetweenChunks: an interrupt that answers true
// after the first chunk stops the prefill there with ErrPrefillInterrupted,
// and the instance, reset, answers as a fresh one does.
func TestAnInterruptedPrefillStopsBetweenChunks(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	s := m.NewInstance(2048)
	defer s.Close()
	prompt := make([]int32, 0, 3*MaxPrefillChunk)
	for i := 0; i < cap(prompt); i++ {
		prompt = append(prompt, int32(300+i%900))
	}
	want, err := s.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)
	s.Reset()

	asked := 0
	s.SetPrefillInterrupt(func() bool { asked++; return asked > 1 })
	if _, err := s.Prefill(prompt); !errors.Is(err, ErrPrefillInterrupted) {
		t.Fatalf("an interrupted prefill returned %v", err)
	}
	if asked != 2 {
		t.Fatalf("the interrupt was asked %d times; want once per chunk until it said stop (2)", asked)
	}
	if p := s.Pos(); p == 0 || p >= len(prompt) {
		t.Fatalf("the interrupted prefill stopped at position %d of %d: not between chunks", p, len(prompt))
	}
	s.SetPrefillInterrupt(nil)
	s.Reset()
	got, err := s.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	if argmax(got) != argmax(want) {
		t.Fatalf("after an interrupt and a reset the prefill picks %d, a clean one %d", argmax(got), argmax(want))
	}
}

// TestPinnedPagesOutliveTheBound: pages set through a Pinned view stay through
// any limit until the view releases them, a page two views pin stays until
// both have released it, and a released page is evicted like any other.
func TestPinnedPagesOutliveTheBound(t *testing.T) {
	page := bytes.Repeat([]byte{5}, 4096)
	m := NewBoundedMemStore(4096)
	defer m.Close()
	a, b := m.Pinned(), m.Pinned()
	for _, v := range []*PinnedStore{a, b} {
		if err := v.Set("p", 0, 0, bytes.NewReader(page)); err != nil {
			t.Fatal(err)
		}
	}
	if err := a.Set("p", 0, 1, bytes.NewReader(page)); err != nil {
		t.Fatal(err)
	}
	for i := 2; i < 5; i++ {
		if err := m.Set("q", 0, i, bytes.NewReader(page)); err != nil {
			t.Fatal(err)
		}
	}
	if st := m.Stats(); st.Pinned != 2 {
		t.Fatalf("%+v: want both pinned pages held past the limit", st)
	}
	a.Release()
	if err := m.Get("p", 0, 0, io.Discard); err != nil {
		t.Fatalf("a page the second view still pins went with the first's release: %v", err)
	}
	if err := m.Get("p", 0, 1, io.Discard); !errors.Is(err, ErrNoPage) {
		t.Fatalf("a released page outlived the limit (%v)", err)
	}
	b.Release()
	if st := m.Stats(); st.Pinned != 0 || st.Bytes > 4096 {
		t.Fatalf("after both releases: %+v", st)
	}
}

// TestSetFillsHoldsEveryPrefixOverOneCopy: the fills of one partial page read
// back as the bytes Set would have held, the shared copy is charged once, and
// it is given back with its last page, whichever order the pages go in.
func TestSetFillsHoldsEveryPrefixOverOneCopy(t *testing.T) {
	src := make([]byte, 3*4096)
	for i := range src {
		src[i] = byte(i*7 + i>>8)
	}
	keys := []string{"a16", "a32", "a48"}
	lens := []int{4096, 2 * 4096, 3 * 4096}
	m := NewBoundedMemStore(8 * 4096)
	defer m.Close()
	if err := m.SetFills(4, 2, src, keys, lens); err != nil {
		t.Fatal(err)
	}
	if got := m.Bytes(); got != 3*4096 {
		t.Fatalf("three fills charged %d bytes, want one copy of the longest, %d", got, 3*4096)
	}
	for i, k := range keys {
		var sink bytes.Buffer
		if err := m.Get(k, 4, 2, &sink); err != nil {
			t.Fatalf("fill %s: %v", k, err)
		}
		if !bytes.Equal(sink.Bytes(), src[:lens[i]]) {
			t.Fatalf("fill %s came back as other bytes than src[:%d]", k, lens[i])
		}
	}
	// The longest fill alone going leaves the copy held for the others.
	m.mu.Lock()
	m.remove(m.pages[kvStoreKey{"a48", 4, 2}])
	held := m.bytes
	m.mu.Unlock()
	if held != 3*4096 {
		t.Fatalf("dropping the longest fill freed memory its shorter fills still point into: %d held", held)
	}
	var sink bytes.Buffer
	if err := m.Get("a32", 4, 2, &sink); err != nil || !bytes.Equal(sink.Bytes(), src[:2*4096]) {
		t.Fatalf("a shorter fill after the longest went: %v", err)
	}
	m.SetLimit(1)
	if st := m.Stats(); st.Pages != 0 || st.Bytes != 0 {
		t.Fatalf("every fill evicted left %+v", st)
	}
	if err := m.SetFills(0, 0, src, []string{"x"}, []int{len(src) + 1}); err == nil {
		t.Fatal("a length past the bytes offered was accepted")
	}
}

// TestSetFillsRespectsTheBound: fills too large for the limit are not held, the
// rest are, and a fill set that all misses the limit leaks nothing.
func TestSetFillsRespectsTheBound(t *testing.T) {
	src := bytes.Repeat([]byte{9}, 4*4096)
	m := NewBoundedMemStore(2 * 4096)
	defer m.Close()
	if err := m.SetFills(0, 0, src, []string{"s", "m", "l"}, []int{4096, 2 * 4096, 4 * 4096}); err != nil {
		t.Fatal(err)
	}
	if st := m.Stats(); st.Pages != 2 {
		t.Fatalf("%+v: want the two fills inside the limit", st)
	}
	if got := m.Bytes(); got > 2*4096 {
		t.Fatalf("held %d past a %d limit", got, 2*4096)
	}
	if err := m.SetFills(1, 1, src, []string{"big"}, []int{4 * 4096}); err != nil {
		t.Fatal(err)
	}
	if st := m.Stats(); st.Pages != 2 {
		t.Fatalf("%+v: a fill larger than the limit was held", st)
	}
}

// TestPinnedFillsOutliveTheBound: fills offered through a pinned view stay out
// of eviction until Release, as pinned pages from Set do.
func TestPinnedFillsOutliveTheBound(t *testing.T) {
	src := bytes.Repeat([]byte{3}, 2*4096)
	m := NewBoundedMemStore(4096)
	defer m.Close()
	pv := m.Pinned()
	if err := pv.SetFills(0, 0, src, []string{"p1", "p2"}, []int{4096, 2 * 4096}); err != nil {
		t.Fatal(err)
	}
	if st := m.Stats(); st.Pages != 2 || st.Pinned != 2 {
		t.Fatalf("%+v: both pinned fills must stay past the limit", st)
	}
	pv.Release()
	if st := m.Stats(); st.Pages > 1 || st.Bytes > 4096 {
		t.Fatalf("%+v: after Release the bound applies", st)
	}
}

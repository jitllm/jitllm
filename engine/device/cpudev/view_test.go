package cpudev

import (
	"slices"
	"testing"
)

// TestTableFollowsItsPages sets, moves and clears a table's pages across a
// change of its sequences and holds it, after each, to the table laid out
// from the slots each page is in.
func TestTableFollowsItsPages(t *testing.T) {
	const pp, slots = 64, 6
	a, err := MapArena(pp, 2*slots, false)
	if err != nil {
		t.Fatal(err)
	}
	defer a.unmap()
	v := NewView(3)
	tb := v.Table(1, a, slots)
	if v.At(1) != tb || v.At(0) != nil || v.At(7) != nil {
		t.Fatal("the view does not hold the table it made")
	}
	pages := make([][]float32, slots)
	want := func(nseq int) []uint32 {
		w := make([]uint32, slots*nseq)
		for j, k := range pages {
			i, _ := a.SlotOf(k)
			for q := range nseq {
				w[q*slots+j] = uint32(i*nseq + q)
			}
		}
		return w
	}
	check := func(when string, nseq int) {
		t.Helper()
		if got, w := tb.Pages(nseq), want(nseq); !slices.Equal(got, w) {
			t.Fatalf("%s, %d sequences: %v, want %v", when, nseq, got, w)
		}
	}
	check("no page", 1)
	for _, j := range []int{0, 2, 5} {
		k, _ := tb.Take()
		pages[j] = k
		tb.Set(j, k)
	}
	check("three pages", 1)
	check("three pages", 3)
	k, _ := tb.Take()
	tb.Let(pages[2])
	pages[2] = k
	tb.Set(2, k)
	check("a page moved to another slot", 3)
	elsewhere := make([]float32, pp)
	pages[5] = elsewhere
	tb.Set(5, elsewhere)
	pages[5] = nil
	check("a page not the arena's", 3)
	tb.Set(slots+3, k)
	check("a page past the table", 3)
	tb.Clear()
	clear(pages)
	check("cleared", 3)
	check("cleared", 1)
	if !tb.Holds(k) || tb.Holds(elsewhere) || tb.Holds(nil) {
		t.Fatal("Holds does not tell the arena's slots from other memory")
	}
}

// TestViewHoldsRecurrentStates makes a block's state, reads it back through a
// view grown to hold it, and reads a nil view and a block without one as
// holding nothing.
func TestViewHoldsRecurrentStates(t *testing.T) {
	var nilView *View
	if nilView.Recurrent() || nilView.Conv(0) != nil || nilView.State(0) != nil || nilView.Blocks() != 0 {
		t.Fatal("a nil view holds something")
	}
	v := NewView(2)
	if v.Recurrent() {
		t.Fatal("a fresh view holds a recurrent state")
	}
	v.MakeRec(4, 3, 5)
	if !v.Recurrent() || v.Blocks() != 5 {
		t.Fatalf("after a state for block 4: recurrent %v, %d blocks", v.Recurrent(), v.Blocks())
	}
	if len(v.Conv(4)) != 3 || len(v.State(4)) != 5 || v.Conv(1) != nil || v.State(9) != nil {
		t.Fatalf("block 4 holds %d and %d, block 1 %v", len(v.Conv(4)), len(v.State(4)), v.Conv(1))
	}
	v.State(4)[2] = 7
	v.MakeRec(4, 3, 5)
	if v.State(4)[2] != 0 {
		t.Fatal("a state made again kept what the old one held")
	}
}

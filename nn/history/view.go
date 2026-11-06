package history

// View is a session's hold on the CPU device's history, as a card's session
// view (gpu.GPU.Attach) is on the card's: for each block keeping one, its
// pages' memory (Pages) and the table its forms read them through
// (SrcPages), and a linear block's recurrent state and convolution rows
// (SrcRecState, SrcRecConv). The session's history decides what each holds;
// the view is where it is. The zero View holds nothing, and a nil one reads
// as one.
type View struct {
	pages []*Pages
	rec   []recState
}

// recState is a linear block's state and convolution rows, every sequence's
// back to back.
type recState struct{ conv, state []float32 }

// NewView is a view of n blocks' history, holding none yet.
func NewView(n int) *View { return &View{pages: make([]*Pages, n)} }

// Blocks is how many blocks the view spans: one past the last that holds
// anything.
func (v *View) Blocks() int {
	if v == nil {
		return 0
	}
	return max(len(v.pages), len(v.rec))
}

// At is block li's table, nil where it has none.
func (v *View) At(li int) *Table {
	if v == nil || li >= len(v.pages) || v.pages[li] == nil {
		return nil
	}
	return v.pages[li].t
}

// MakeRec gives block li a recurrent state of state floats and convolution
// rows of conv, zeroed, in place of any it held.
func (v *View) MakeRec(li, conv, state int) {
	for len(v.rec) <= li {
		v.rec = append(v.rec, recState{})
	}
	v.rec[li] = recState{conv: make([]float32, conv), state: make([]float32, state)}
}

// Recurrent says some block holds a recurrent state.
func (v *View) Recurrent() bool { return v != nil && len(v.rec) > 0 }

// Conv is block li's convolution rows, nil where it holds none.
func (v *View) Conv(li int) []float32 {
	if v == nil || li >= len(v.rec) {
		return nil
	}
	return v.rec[li].conv
}

// State is block li's recurrent state, nil where it holds none.
func (v *View) State(li int) []float32 {
	if v == nil || li >= len(v.rec) {
		return nil
	}
	return v.rec[li].state
}

// Table is one block's pages in its arena: the slot each page is held in and
// the page table its forms read. Page j of sequence q is at q*slots+j and
// names slot*nseq+q, a slot holding every sequence's positions in order; a
// page the arena does not hold names slot 0's, which no row's range reaches.
// The table follows Set, so a call binds it as it is.
type Table struct {
	a     *Arena
	slots int
	of    []uint32
	nseq  int
	tab   []uint32
}

// Arena is the arena the table's slots are of.
func (t *Table) Arena() *Arena { return t.a }

// Slots is the pages the table spans.
func (t *Table) Slots() int { return t.slots }

// Take is a fresh slot's key and value, held by the table's history, or nil
// where every slot is taken.
func (t *Table) Take() (k, v []float32) {
	i, ok := t.a.Take()
	if !ok {
		return nil, nil
	}
	return t.a.Page(i)
}

// Holds says b is one of the arena's slots.
func (t *Table) Holds(b []float32) bool {
	_, ok := t.a.SlotOf(b)
	return ok
}

// Let lets the history's hold on the slot b starts go, where b is one.
func (t *Table) Let(b []float32) {
	if i, ok := t.a.SlotOf(b); ok {
		t.a.Drop(i)
	}
}

// Set says page n is now k: a slot of the arena, or memory it does not hold
// (nil, or a page elsewhere), which the table names slot 0 for.
func (t *Table) Set(n int, k []float32) {
	if n >= t.slots {
		return
	}
	i := 0
	if k != nil {
		if si, ok := t.a.SlotOf(k); ok {
			i = si
		}
	}
	t.of[n] = uint32(i)
	if t.nseq > 0 {
		for q := range t.nseq {
			t.tab[q*t.slots+n] = uint32(i*t.nseq + q)
		}
	}
}

// Clear says no page is held: every page names slot 0, sequence q's slot 0
// being entry q.
func (t *Table) Clear() {
	clear(t.of)
	for q := range t.nseq {
		for j := range t.slots {
			t.tab[q*t.slots+j] = uint32(q)
		}
	}
}

// Pages is the table for nseq sequences: kept by Set, and laid out again
// only when the sequences change.
func (t *Table) Pages(nseq int) []uint32 {
	nseq = max(nseq, 1)
	if t.nseq != nseq {
		n := t.slots * nseq
		if cap(t.tab) < n {
			t.tab = make([]uint32, n)
		}
		t.tab, t.nseq = t.tab[:n], nseq
		for j, i := range t.of {
			for q := range nseq {
				t.tab[q*t.slots+j] = i*uint32(nseq) + uint32(q)
			}
		}
	}
	return t.tab
}

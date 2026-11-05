package jlm

import "testing"

// newFake is frames over a File with no file behind it, so these gates need
// no container on disk. fakeIn is EnsurePage's residency half with the read
// left out.
func newFake(text, textPage, vis, visPage uint64) *Frames {
	f := &Frames{file: &File{H: &Header{
		PageSize: textPage, NBlocks: uint32(text),
		VisPageSize: visPage, NVisBlocks: uint32(vis),
		DataOff: 1 << 20,
	}}}
	f.file.H.VisDataOff = f.file.H.DataOff + text*textPage
	f.pages = make([][]byte, int(text)+int(vis))
	f.free = map[uint64][][]byte{}
	f.recent = -1
	return f
}

func (f *Frames) fakeIn(i int, mark byte) {
	if f.pages[i] != nil {
		f.recent = i
		return
	}
	_, size := f.file.pageAt(i)
	f.makeRoom(i, size)
	b := f.take(size)
	b[0] = mark
	f.pages[i], f.recent = b, i
	f.resident += size
}

// TestBudgetReusesItsBuffers is a memory gate. A page can be over a gigabyte,
// so a page-in that evicts must land in the evicted page's frame rather than a
// fresh one: a fresh frame costs its page faults, and on the heap a dropped one
// was garbage the collector need not reclaim promptly. Under a budget of four
// pages, faulting eight blocks in turn must reuse the first four frames, and
// ReleasePages, which drops every page, must give every frame back.
func TestBudgetReusesItsBuffers(t *testing.T) {
	const page = 4096
	f := newFake(8, page, 0, Align)
	f.SetBudget(4 * page)
	first := map[*byte]bool{}
	for i := 0; i < 4; i++ {
		f.fakeIn(i, byte(i+1))
		first[&f.pages[i][0]] = true
	}
	for i := 4; i < 8; i++ {
		f.fakeIn(i, byte(i+1))
		if !first[&f.pages[i][0]] {
			t.Errorf("block %d got a new frame where an evicted one was free: %d bytes REALLOCATED", i, page)
		}
	}
	f.ReleasePages()
	if r, fb := f.ResidentBytes(), f.FreeBytes(); r != 0 || fb != 0 {
		t.Fatalf("ReleasePages left %d bytes resident and %d free", r, fb)
	}
}

// And the free list must not hand a vision buffer to a text page: a pool that
// ignored size would return a short buffer, a bounds panic at best.
func TestBudgetKeepsTheTwoPageSizesApart(t *testing.T) {
	const textPage, visPage = 8192, 4096
	f := newFake(4, textPage, 3, visPage)
	f.SetBudget(textPage + visPage)

	f.fakeIn(0, 1)                     // a text page
	f.fakeIn(int(f.file.H.NBlocks), 2) // a vision page
	if n := len(f.pages[0]); uint64(n) != textPage {
		t.Fatalf("text block 0 got a %d-byte page, want %d", n, textPage)
	}
	if n := len(f.pages[f.file.H.NBlocks]); uint64(n) != visPage {
		t.Fatalf("vision block 0 got a %d-byte page, want %d", n, visPage)
	}
	// Evict both, then fault them in the opposite order: if the free list were
	// keyed by anything but size, the text page would come back short.
	f.ReleasePages()
	f.fakeIn(int(f.file.H.NBlocks), 3)
	f.fakeIn(1, 4)
	if n := len(f.pages[1]); uint64(n) != textPage {
		t.Fatalf("text block 1 came back %d bytes, want %d -- the free list handed "+
			"back a vision buffer", n, textPage)
	}
	if n := len(f.pages[f.file.H.NBlocks]); uint64(n) != visPage {
		t.Fatalf("vision block 0 came back %d bytes, want %d", n, visPage)
	}
}

// TestBudgetKeepsResidency is the gate on "why is it reading a block that is
// already in memory".
//
// Re-budgeting must not nil every page: with the budget set twice on a
// normal load a model would be read in, thrown away, and read again by the
// first token. Widening must be free and narrowing must evict exactly the excess.
func TestBudgetKeepsResidency(t *testing.T) {
	const blocks, page = 8, 4096
	f := newFake(blocks, page, 0, Align)
	f.SetBudget(blocks * page)
	for i := 0; i < 4; i++ {
		f.fakeIn(i, byte(i+1))
	}

	f.SetBudget(blocks * page) // same budget: nothing should be evicted
	for i := 0; i < 4; i++ {
		if !f.Resident(i) {
			t.Errorf("block %d stopped being resident across an identical re-budget", i)
		}
		if f.pages[i] == nil || f.pages[i][0] != byte(i+1) {
			t.Errorf("block %d lost its contents across an identical re-budget", i)
		}
	}

	// Widening is free too -- and it is what a tower's Release hands back.
	f.SetBudget(blocks * page * 2)
	for i := 0; i < 4; i++ {
		if !f.Resident(i) {
			t.Errorf("block %d was evicted by WIDENING the budget", i)
		}
	}

	// A page-in with room to spare must not evict anything.
	f.fakeIn(4, 5)
	for i := 0; i < 4; i++ {
		if !f.Resident(i) {
			t.Fatalf("block %d was evicted to serve a page-in that had room", i)
		}
	}

	// Narrowing past the resident set does evict, which is the ask.
	f.SetBudget(2 * page)
	if f.ResidentBytes() > 2*page {
		t.Errorf("%d bytes resident under a %d-byte budget", f.ResidentBytes(), 2*page)
	}
	if n := f.ResidentPages(); n > 2 {
		t.Errorf("%d pages resident under a 2-page budget", n)
	}
}

// A budget below one page still serves the page: it is a configuration error
// the caller cannot recover from mid-token, and thrashing one page is better
// than failing a read.
func TestBudgetBelowOnePageStillServesIt(t *testing.T) {
	const page = 4096
	f := newFake(4, page, 0, Align)
	f.SetBudget(page / 2)
	f.fakeIn(0, 1)
	if !f.Resident(0) {
		t.Fatal("a budget below one page served no page at all")
	}
}

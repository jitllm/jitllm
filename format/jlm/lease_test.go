package jlm

import "testing"

// A pinned page is never the victim, and the pin is what keeps it.
//
// A lease is a public primitive nothing in the tree relies on yet, which is
// when it needs a gate: a pin that silently does nothing looks identical to a
// working one. The fixture puts the held page in the victim's chair (one
// frame, f.recent set to the held block before every fault); with more
// frames the MRU victim is never the held block and the gate passes against
// its own violation.
func TestAPinnedPageIsNeverTheVictim(t *testing.T) {
	const page = 4096
	f := newFake(4, page, 0, Align)
	f.SetBudget(page) // ONE frame for four blocks

	f.fakeIn(0, 1)
	l, err := f.Hold(0, nil)
	if err != nil {
		t.Fatalf("Hold: %v", err)
	}
	if n := f.pinCount(0); n != 1 {
		t.Fatalf("block 0 has %d pins after Hold, want 1", n)
	}
	for i := 1; i < 4; i++ {
		f.recent = 0 // block 0 IS the page the pager is about to choose
		f.fakeIn(i, byte(i+1))
		if !f.Resident(0) {
			t.Fatalf("block 0 was evicted while pinned, faulting block %d", i)
		}
	}
	if f.pages[0][0] != 1 {
		t.Errorf("block 0's frame holds %d, not its own mark: the pin kept the "+
			"entry and not the memory", f.pages[0][0])
	}

	// And it must become evictable again, or a lease is a leak: a pin never
	// dropped fills the pool with unevictable pages.
	l.Release()
	if n := f.pinCount(0); n != 0 {
		t.Fatalf("block 0 has %d pins after Release, want 0", n)
	}
	f.DropPage(1)
	f.DropPage(2)
	f.DropPage(3)
	f.recent = 0
	f.fakeIn(1, 9)
	if f.Resident(0) {
		t.Errorf("block 0 survived a fault after its lease was released: the pin " +
			"is not what was keeping it")
	}

	// Release is idempotent so a deferred one is safe beside an early one; a
	// second decrement would let the next holder's pin read as zero.
	l.Release()
	if n := f.pinCount(0); n != 0 {
		t.Errorf("a second Release took the count to %d", n)
	}
}

// Two leases on one block, because a pin is a count and not a flag: the second
// holder must keep the page after the first releases.
func TestTwoLeasesOnOneBlock(t *testing.T) {
	const page = 4096
	f := newFake(4, page, 0, Align)
	f.SetBudget(page)
	f.fakeIn(0, 1)

	a, err := f.Hold(0, nil)
	if err != nil {
		t.Fatalf("Hold: %v", err)
	}
	b, err := f.Hold(0, nil)
	if err != nil {
		t.Fatalf("Hold: %v", err)
	}
	if n := f.pinCount(0); n != 2 {
		t.Fatalf("%d pins after two Holds, want 2", n)
	}
	a.Release()
	if n := f.pinCount(0); n != 1 {
		t.Fatalf("%d pins after one Release, want 1", n)
	}
	for i := 1; i < 4; i++ {
		f.recent = 0
		f.fakeIn(i, byte(i+1))
		if !f.Resident(0) {
			t.Fatalf("block 0 went while the second lease still held it")
		}
	}
	b.Release()
	if n := f.pinCount(0); n != 0 {
		t.Errorf("%d pins after both Releases", n)
	}
	f.DropPage(1)
	f.DropPage(2)
	f.DropPage(3)
	f.recent = 0
	f.fakeIn(1, 9)
	if f.Resident(0) {
		t.Errorf("block 0 survived once both leases were released")
	}
}

// pinCount reads the pin under the lock. A test helper rather than a File
// method, since nothing outside this package needs to ask.
func (f *Frames) pinCount(i int) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	if i < 0 || i >= len(f.pins) {
		return 0
	}
	return f.pins[i]
}

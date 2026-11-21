package jlm

import (
	"os"
	"path/filepath"
	"testing"
	"unsafe"
)

// TestRecycleNamesEveryFrameItRefills is the pager's half of placement.md 15v.
// A device keeps copies of weights keyed on their address, and a frame is one
// address for block after block, so the pager must name every frame whose bytes
// it is about to change: one handed to a page (fresh, or an evicted page's,
// reused), one given back, and everything at Close -- and it must do so
// without f.mu held, since the device it reaches may be in the middle of a
// page-in of its own.
func TestRecycleNamesEveryFrameItRefills(t *testing.T) {
	const page = 4096
	f := newFake(2, page, 0, Align)
	f.SetBudget(page) // one frame: block 1 can only take block 0's
	var got []uintptr
	unlocked := true
	f.OnRecycle(func(p unsafe.Pointer, n uintptr) {
		if !f.mu.TryLock() {
			unlocked = false
		} else {
			f.mu.Unlock()
		}
		if n < page {
			t.Errorf("a %d-byte range for a %d-byte frame", n, page)
		}
		got = append(got, uintptr(p))
	})
	named := func(b []byte) bool {
		at := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
		for _, p := range got {
			if p == at {
				return true
			}
		}
		return false
	}

	l, err := f.Hold(0, nil)
	if err != nil {
		t.Fatal(err)
	}
	frame := f.pages[0]
	l.Release()
	if !named(frame) {
		t.Fatal("a fresh frame handed to block 0 was not named")
	}

	got = nil
	if l, err = f.Hold(1, nil); err != nil {
		t.Fatal(err)
	}
	if unsafe.SliceData(f.pages[1]) != unsafe.SliceData(frame) {
		t.Fatal("block 1 did not reuse block 0's frame, so this gate measures nothing")
	}
	l.Release()
	if !named(frame) {
		t.Fatal("block 0's frame was refilled with block 1 and not named")
	}

	got = nil
	f.ReleasePages()
	if !named(frame) {
		t.Fatal("ReleasePages gave a frame back without naming it")
	}

	// Close names the dense region and every frame, held or free.
	h, err := os.Create(filepath.Join(t.TempDir(), "close"))
	if err != nil {
		t.Fatal(err)
	}
	f.file.f, f.dense = h, alignedBuf(page)
	dense := f.dense
	if l, err = f.Hold(0, nil); err != nil {
		t.Fatal(err)
	}
	held := f.pages[0]
	l.Release()
	got = nil
	f.Close()
	if err := f.file.Close(); err != nil {
		t.Fatal(err)
	}
	if !named(dense) || !named(held) {
		t.Fatalf("Close named the dense region %v and a held frame %v", named(dense), named(held))
	}
	if !unlocked {
		t.Fatal("the callback ran with f.mu held")
	}
}

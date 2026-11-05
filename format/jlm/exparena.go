package jlm

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/internal/offheap"
)

// The expert pages' frames, where the host reserves address space: one
// reservation the size of the file's expert array, each expert page's frame
// at that page's own place in it, committed when the page is read in and
// handed back when it is evicted. An expert's sheet of a bank is then at one
// address whichever page-in brought it there, so a bank is one view of its
// experts at a stride of the expert page size (BankView): a device reading
// the host's memory binds the bank whole and reads the experts it routed to,
// each made resident first (EnsureExperts). The budget, the LRU order, the
// page-ins and the evictions are the pager's own; only where a frame lives
// changes. Without a reservation (no anonymous mmap, or a page size the
// O_DIRECT alignment does not divide) expert pages take frames as block pages
// do.

// expFrame is expert page li's frame of size bytes in the reservation, made
// at the first expert page-in, or nil where there is none. The reservation is
// made with the frames on the heap too (OnHeap): it is address space, and a
// bank is one view of its experts only there. Callers hold f.mu.
func (f *Frames) expFrame(li int, size uint64) ([]byte, error) {
	if f.exp == nil && !f.expNone {
		n := uint64(f.file.H.NExpPages) * f.file.H.ExpPageSize
		if n == 0 || f.file.H.ExpPageSize%Align != 0 {
			f.expNone = true
			return nil, nil
		}
		b, mapped, err := offheap.ReserveOffHeap(int(n))
		if err != nil || !mapped {
			f.expNone = true
			return nil, nil
		}
		f.exp = b
	}
	if f.exp == nil {
		return nil, nil
	}
	at := uint64(li-int(f.file.H.NBlocks)-int(f.file.H.NVisBlocks)) * f.file.H.ExpPageSize
	if size > f.file.H.ExpPageSize || at+size > uint64(len(f.exp)) {
		return nil, nil
	}
	b := f.exp[at : at+size : at+size]
	// A bank's view reads this frame and no other, so a frame it cannot commit
	// fails the page-in.
	if err := offheap.CommitOffHeap(b); err != nil {
		return nil, fmt.Errorf("jlm: expert page %d: %w", li, err)
	}
	placeMemory(b, f.noHuge, f.nodeMask)
	f.expHeld += size
	return b, nil
}

// inExp reports whether b is a frame of the expert reservation.
func (f *Frames) inExp(b []byte) bool {
	if len(f.exp) == 0 || cap(b) == 0 {
		return false
	}
	p, lo := uintptr(unsafe.Pointer(unsafe.SliceData(b))), uintptr(unsafe.Pointer(unsafe.SliceData(f.exp)))
	return p >= lo && p < lo+uintptr(len(f.exp))
}

// dropExp hands an evicted expert page's frame back to the system. Callers
// hold f.mu.
func (f *Frames) dropExp(b []byte) {
	f.retire(b)
	offheap.DropOffHeap(b[:cap(b)])
	f.expHeld -= uint64(cap(b))
}

// unmapExp gives the reservation back at Close. Callers hold f.mu and hold no
// frame of it.
func (f *Frames) unmapExp() {
	if f.exp != nil {
		offheap.UnmapOffHeap(f.exp)
		f.exp, f.expHeld = nil, 0
	}
}

// BankView is bank entry e's planes as one view of every expert's sheet,
// in the expert reservation: each plane's address for expert 0 (nil for a
// plane the format has none of) and the bytes from one expert's sheet to the
// next's, the expert page size. ok is false where e is not in expert pages
// or there is no reservation. A sheet is readable only while its page is
// resident (EnsureExperts) and held (Hold); the addresses hold until Close.
func (f *Frames) BankView(e *Entry) (planes [3]unsafe.Pointer, stride uint64, ok bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.file.ExpertPaged(e) || f.file.ExpertPage(e, 0) < 0 {
		return planes, 0, false
	}
	if f.exp == nil && !f.expNone {
		if _, err := f.expFrame(f.file.ExpertPage(e, 0), 0); err != nil {
			return planes, 0, false
		}
	}
	if f.exp == nil {
		return planes, 0, false
	}
	base := unsafe.Pointer(unsafe.SliceData(f.exp))
	qs, d, sc, _, err := sheetPlanes(e)
	if err != nil {
		return planes, 0, false
	}
	for i, pl := range [3]struct{ off, n uint64 }{{e.QSOff, qs}, {e.DOff, d}, {e.SCOff, sc}} {
		if pl.n > 0 {
			planes[i] = unsafe.Add(base, pl.off-f.file.H.ExpDataOff)
		}
	}
	return planes, f.file.H.ExpPageSize, true
}

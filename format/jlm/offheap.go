package jlm

import (
	"sync/atomic"
	"unsafe"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// A File's weight memory -- its page frames and its dense region -- lives off
// the Go heap: anonymous mappings (kernels.MapOffHeap, the allocator the
// device arena uses) the runtime does not know exist. They are not in the
// collector's heap goal, not in its memory-limit arithmetic and not marked,
// and the File unmaps them at Close. On the heap, a frame holds no pointers and
// is never scanned, but it is permanently live heap: frames set the
// collector's goal, and a weight budget near a memory limit left the collector
// running back to back (AGENTS.md RULE 2f). Off the heap the same budget runs
// with a handful of cycles. This is anonymous memory the File owns and frees;
// the model FILE is still read, never mapped.
//
// The cost is the collector's safety net: a span into a frame that outlives
// its File's Close faults instead of keeping the frame alive. Where the
// platform has no anonymous mmap, buf falls back to the heap.
var offHeap atomic.Bool

func init() { offHeap.Store(true) }

// SetOffHeap sets whether Files opened after the call keep their weight memory
// off the Go heap (the default), and reports the previous setting. Off is the
// control arm for a measurement; a caller turning it off should also call
// sched.SetWeightsOffHeap(false), so the weight budget keeps its reserve
// against the collector's goal.
func SetOffHeap(on bool) bool { return offHeap.Swap(on) }

// offHeapBytes is every live off-heap mapping in the process, all Files.
var offHeapBytes atomic.Int64

// OffHeapBytes is how much weight memory this process holds outside the Go
// heap.
func OffHeapBytes() int64 { return offHeapBytes.Load() }

// buf is an n-byte weight buffer, Align-aligned for O_DIRECT, on the heap or
// mapped as the File was opened to do. A mapped one is recorded for Close.
func (f *File) buf(n int) []byte {
	if f.offHeap && n > 0 {
		if b, mapped, err := kernels.MapOffHeap(n); err == nil && mapped {
			if f.mapped == nil {
				f.mapped = map[*byte][]byte{}
			}
			f.mapped[unsafe.SliceData(b)] = b
			offHeapBytes.Add(int64(cap(b)))
			return b[:n:n]
		}
	}
	return alignedBuf(n)
}

// OnRecycle sets fn to be told the address range of every buffer whose bytes
// stop being the ones a reader may have copied: a frame handed to a page (a
// fresh one or one the pager reuses), a frame or the dense region given back,
// and everything at Close. A device that keeps copies keyed on a weight's
// address (nn.Device) has to drop the ones inside that range, or it answers for
// the next block's bytes from the last block's copy (placement.md 15v).
//
// fn runs after f.mu is released, on whichever goroutine paged, and before
// that goroutine reads anything into the frame. nil turns it off.
func (f *File) OnRecycle(fn func(p unsafe.Pointer, n uintptr)) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.recycle = fn
}

// retire records b for the recycle callback: the whole buffer, since a frame
// handed to a smaller page (take) still holds the tail of what it held before.
// Callers hold f.mu.
func (f *File) retire(b []byte) {
	if f.recycle == nil || cap(b) == 0 {
		return
	}
	f.recycled = append(f.recycled, b[:cap(b)])
}

// flushRecycled hands what retire recorded to the callback. Callers do not hold
// f.mu: the callback reaches a device's own lock, and the device reaches this
// File's (a page-in from inside an upload).
func (f *File) flushRecycled() {
	f.mu.Lock()
	fn, bs := f.recycle, f.recycled
	f.recycled, f.recycledSpare = f.recycledSpare, nil
	f.mu.Unlock()
	if fn != nil {
		for _, b := range bs {
			fn(unsafe.Pointer(unsafe.SliceData(b)), uintptr(len(b)))
		}
	}
	if cap(bs) == 0 {
		return
	}
	// The array goes back for the next retire: a paged decode recycles a frame
	// per block per token, and a fresh array each time was an allocation each.
	clear(bs)
	f.mu.Lock()
	if f.recycledSpare == nil {
		f.recycledSpare = bs[:0]
	}
	f.mu.Unlock()
}

// release gives a buffer this File no longer holds back: a mapping is
// unmapped at once, a heap buffer is left to the collector. Callers hold f.mu and
// hold no span into it.
func (f *File) release(b []byte) {
	f.retire(b)
	if m, ok := f.mapped[unsafe.SliceData(b)]; ok {
		delete(f.mapped, unsafe.SliceData(b))
		offHeapBytes.Add(-int64(cap(m)))
		kernels.UnmapOffHeap(m)
	}
}

// putFree keeps a frame of size bytes for the next page-in. Callers hold f.mu.
func (f *File) putFree(size uint64, b []byte) {
	f.free[size] = append(f.free[size], b)
	f.freeBytes += size
}

// trimFree releases free frames until at most keep bytes of them are left.
// Callers hold f.mu.
func (f *File) trimFree(keep uint64) {
	for size, l := range f.free {
		for len(l) > 0 && f.freeBytes > keep {
			f.release(l[len(l)-1])
			l[len(l)-1] = nil
			l = l[:len(l)-1]
			f.freeBytes -= size
		}
		// An emptied list keeps its array: page sizes are few and fixed at
		// conversion, and dropping it made the next putFree of that size
		// allocate one again (a paged mixture token, every eviction).
		f.free[size] = l
	}
}

// TrimFree hands back every free frame. A frame on the free list waits for the
// next page-in, which is what makes a burst of them cheap (placement releases
// each block's pages and the next block's land in the same frames), and once
// the burst is over nothing will take it: the caller ends one by trimming.
func (f *File) TrimFree() {
	if f == nil {
		return
	}
	defer f.flushRecycled()
	f.mu.Lock()
	defer f.mu.Unlock()
	f.trimFree(0)
}

// MappedBytes is how much off-heap memory this File holds, and
// FreeBytes how much of its frame memory is on the free list.
func (f *File) MappedBytes() uint64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	var n uint64
	for _, m := range f.mapped {
		n += uint64(cap(m))
	}
	return n
}

func (f *File) FreeBytes() uint64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.freeBytes
}

// unmapAll returns every mapping this File still holds. Callers hold f.mu and
// have already dropped every reference the File holds.
func (f *File) unmapAll() {
	for k, m := range f.mapped {
		offHeapBytes.Add(-int64(cap(m)))
		kernels.UnmapOffHeap(m)
		delete(f.mapped, k)
	}
}

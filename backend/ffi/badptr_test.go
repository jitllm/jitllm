//go:build linux || darwin

package ffi

import (
	"testing"
	"unsafe"
)

// TestHeapPointerGoesStaleOnAStackCopy demonstrates that the object shape this
// package used to build (a heap array holding stack addresses) is genuinely
// broken, not only illegal on paper. TestCallArgumentsStayOnTheStack shows the
// shipped code no longer builds it. One observed "found bad pointer in Go heap"
// crash is attributed to it only by inference.
//
// It tests the deterministic failure mode. The collector reporting a bad
// pointer depends on the old stack's span being freed and reissued, a race;
// stale arguments do not: stack copying adjusts pointers in frames, not in heap
// objects, so the heap copy keeps pointing at the old stack every time.
func TestHeapPointerGoesStaleOnAStackCopy(t *testing.T) {
	var sink [1]unsafe.Pointer
	esc = &sink // make the array escape, exactly as passing ps[:N] did

	before, after := fillThenGrow(esc)
	held := uintptr(esc[0])

	if before == after {
		t.Skipf("the stack did not move (local stayed at %#x), so this run "+
			"exercised nothing; raise the recursion depth", before)
	}
	if held != before {
		t.Fatalf("the heap object holds %#x and the local was at %#x before the "+
			"copy; this test is not measuring what it claims", held, before)
	}
	// The finding: the frame's reference to local was adjusted by the
	// runtime and the heap object's was not.
	t.Logf("local moved %#x -> %#x on a stack copy; the heap object still holds %#x",
		before, after, held)
	if held == after {
		t.Fatal("the heap-held address tracked the stack copy, which would mean the " +
			"runtime adjusts pointers in heap objects -- it does not, and if this " +
			"ever passes the reasoning behind prepared.call needs re-deriving")
	}
}

// esc is package-level so the array it points at is heap-allocated, which is
// what turns the legal version of prepared.call into the illegal one.
var esc *[1]unsafe.Pointer

// fillThenGrow writes a frame-local's address into a heap object exactly as the
// old wrappers did, forces the stack to be copied, and reports where the local
// was before and after.
//
//go:noinline
func fillThenGrow(dst *[1]unsafe.Pointer) (before, after uintptr) {
	var local [4]uint64
	// Stored as an integer so the compiler keeps local in the frame, as the
	// old wrappers' noescape did.
	*(*uintptr)(unsafe.Pointer(&dst[0])) = uintptr(unsafe.Pointer(&local[0]))
	before = uintptr(unsafe.Pointer(&local[0]))
	// Deep enough to exceed the starting stack and force a copy. The result is
	// consumed so the recursion cannot be optimised away.
	local[0] = uint64(deep(2048)[0])
	after = uintptr(unsafe.Pointer(&local[0]))
	return before, after
}

//go:noinline
func deep(n int) [96]byte {
	var pad [96]byte
	if n > 0 {
		p := deep(n - 1)
		pad[0] = p[0] + 1
	}
	return pad
}

//go:build linux || darwin

package ffi

import (
	"testing"
	"unsafe"
)

// TestCallArgumentsStayOnTheStack is the gate on the one invariant this package
// can break silently: a heap object must never hold a pointer into a goroutine
// stack.
//
// A call copies its arguments into one of its binding's heap frames, so no
// address of a stack slot reaches goffi (see frames). The count is the gate
// that the frames are reused: building the arguments per call, in the
// wrapper's frame or a new object, reads one allocation or more.
//
// The count is only meaningful uninstrumented: -race reads 0, but
// -gcflags=all=-d=checkptr=2 instruments every unsafe.Pointer conversion and
// reads 6-8 with nothing escaping.
func TestCallArgumentsStayOnTheStack(t *testing.T) {
	lib, err := Open("libc.so.6", "libc.so", "libSystem.B.dylib")
	if err != nil {
		// A libc is not a missing artefact: if one cannot be opened the gate
		// must fail, not skip.
		t.Fatalf("no libc to bind against: %v", err)
	}
	defer lib.Close()

	// One argument and several: each width is a different size class, and
	// only the one-argument array is under the 16-byte scan-exemption floor.
	abs := Fn1[int32, int32](lib, "abs")
	if got := abs(-7); got != 7 {
		t.Fatalf("abs(-7) = %d, so this is measuring a call that does not work", got)
	}
	memcmp := Fn3[int32, unsafe.Pointer, unsafe.Pointer, uint64](lib, "memcmp")
	a, b := []byte("jitllm"), []byte("jitllm")
	if got := memcmp(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), uint64(len(a))); got != 0 {
		t.Fatalf("memcmp of equal bytes = %d", got)
	}

	for _, c := range []struct {
		name string
		f    func()
	}{
		{"Fn1", func() { abs(-1) }},
		{"Fn3", func() { memcmp(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), uint64(len(a))) }},
	} {
		if n := testing.AllocsPerRun(2000, c.f); n != 0 {
			t.Errorf("%s: %.3f allocations per call -- the argument array left the "+
				"frame, so a heap object now holds pointers into a goroutine stack",
				c.name, n)
		}
	}
}

//go:build linux || darwin

package ffi

import (
	"runtime"
	"sync"
	"testing"
	"unsafe"

	gffi "github.com/go-webgpu/goffi/ffi"
)

// TestCallSurvivesStackGrowthAndGC calls memcpy into a fresh buffer from
// goroutines at every depth up to a few stack sizes, so that at some depths
// goffi's own frames grow the stack after it has turned the argument
// addresses into integers and before the C function runs, while a collector
// runs beside it.
//
// The destination is the detector: if C is handed an address the stack copy
// left behind, memcpy writes the old stack and the live buffer stays zero.
// The violation arm is the design this package shipped before: the argument
// array, the arguments and the buffer in the frame, hidden from escape
// analysis. It must be caught at some depth; if it is not, the test exercised
// nothing.
func TestCallSurvivesStackGrowthAndGC(t *testing.T) {
	lib, err := Open("libc.so.6", "libc.so", "libSystem.B.dylib")
	if err != nil {
		t.Fatalf("no libc to bind against: %v", err)
	}
	defer lib.Close()
	ptr := descOf[unsafe.Pointer]()
	c := &calls{
		bound: Fn3[unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uint64](lib, "memcpy"),
		raw:   prep(lib, "memcpy", ptr, ptr, ptr, descOf[uint64]()),
	}

	const depths = 1200
	stop := make(chan struct{})
	var gcs sync.WaitGroup
	gcs.Add(1)
	go func() {
		defer gcs.Done()
		for {
			select {
			case <-stop:
				return
			default:
				runtime.GC()
			}
		}
	}()
	for d := range depths {
		if !atDepth(c, d, false) {
			t.Errorf("depth %d: memcpy wrote somewhere other than the live buffer", d)
		}
	}
	close(stop)
	gcs.Wait()

	caught := 0
	for d := range depths {
		if !atDepth(c, d, true) {
			caught++
		}
	}
	if caught == 0 {
		t.Fatal("the stack-frame violation was never caught: no depth moved the " +
			"stack inside the call, so this gate proves nothing")
	}
	t.Logf("the stack-frame violation wrote the old stack at %d of %d depths", caught, depths)
}

type calls struct {
	bound func(unsafe.Pointer, unsafe.Pointer, uint64) unsafe.Pointer
	raw   *prepared
}

// atDepth runs one memcpy on a fresh goroutine below d frames of padding and
// reports whether it landed in the live buffer.
func atDepth(c *calls, d int, violate bool) bool {
	ok := make(chan bool)
	go func() { ok <- pad(c, d, violate) }()
	return <-ok
}

//go:noinline
func pad(c *calls, d int, violate bool) bool {
	var b [32]byte
	if d > 0 {
		r := pad(c, d-1, violate)
		b[0] = 1
		return r && b[0] == 1
	}
	return copyHere(c, violate)
}

//go:noinline
func copyHere(c *calls, violate bool) bool {
	if violate {
		return copyInFrame(c)
	}
	src := [8]byte{1, 2, 3, 4, 5, 6, 7, 8}
	var dst [8]byte
	c.bound(unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), uint64(len(dst)))
	return dst == src
}

// copyInFrame is the violation, in a function of its own: the bound call
// makes its buffer escape, and escape analysis does not tell branches apart.
//
//go:noinline
func copyInFrame(c *calls) bool {
	src := [8]byte{1, 2, 3, 4, 5, 6, 7, 8}
	var dst [8]byte
	dp, sp, n := unsafe.Pointer(&dst[0]), unsafe.Pointer(&src[0]), uint64(len(dst))
	var r unsafe.Pointer
	ps := [3]unsafe.Pointer{hide(unsafe.Pointer(&dp)), hide(unsafe.Pointer(&sp)), hide(unsafe.Pointer(&n))}
	gffi.CallFunction(&c.raw.cif, c.raw.fn, hide(unsafe.Pointer(&r)),
		unsafe.Slice((*unsafe.Pointer)(hide(unsafe.Pointer(&ps[0]))), len(ps)))
	return dst == src
}

// hide is the old noescape, for the violation only: the value comes back
// through an integer, which escape analysis does not follow.
func hide(p unsafe.Pointer) unsafe.Pointer {
	u := uintptr(p)
	return *(*unsafe.Pointer)(unsafe.Pointer(&u))
}

package cuda

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/gpu/ffi"
)

// A queue is a non-blocking stream that carries one session's whole sequence:
// its copies, launches, captures and waits. Two queues on one context run at
// once on the device, which the legacy stream -- implicitly synchronised with
// every blocking stream -- cannot. Because the legacy stream does not order
// against a non-blocking one, nothing a queue's session depends on may travel
// on the legacy stream: the copies below take the queue's stream, and a read
// waits on that stream alone.

// Bound only if the driver has all of them, like the graph calls.
var (
	cuMemcpyHtoDAsync   func(CUdevptr, unsafe.Pointer, uint64, CUstream) CUresult
	cuMemcpyDtoHAsync   func(unsafe.Pointer, CUdevptr, uint64, CUstream) CUresult
	cuStreamSynchronize func(CUstream) CUresult
)

func loadQueues(lib *ffi.Lib) {
	for _, n := range []string{"cuMemcpyHtoDAsync_v2", "cuMemcpyDtoHAsync_v2", "cuStreamSynchronize"} {
		if !lib.Has(n) {
			return
		}
	}
	cuMemcpyHtoDAsync = ffi.Fn4[CUresult, CUdevptr, unsafe.Pointer, uint64, CUstream](lib, "cuMemcpyHtoDAsync_v2")
	cuMemcpyDtoHAsync = ffi.Fn4[CUresult, unsafe.Pointer, CUdevptr, uint64, CUstream](lib, "cuMemcpyDtoHAsync_v2")
	cuStreamSynchronize = ffi.Fn1[CUresult, CUstream](lib, "cuStreamSynchronize")
}

// QueuesAvailable reports whether the driver can run sessions on queues of
// their own: non-blocking streams and copies on them.
func QueuesAvailable() bool { return cuStreamSynchronize != nil && cuStreamCreate != nil }

// streamNonBlocking is CU_STREAM_NON_BLOCKING.
const streamNonBlocking = 1

// NewQueue creates a non-blocking stream. The calling thread must have the
// context current.
func (d *Device) NewQueue() (*Stream, error) {
	if !QueuesAvailable() {
		return nil, fmt.Errorf("cuda: this driver has no non-blocking streams")
	}
	st := &Stream{}
	if err := call(cuStreamCreate(&st.s, streamNonBlocking), "cuStreamCreate"); err != nil {
		return nil, err
	}
	return st, nil
}

// Sync waits for every launch and copy on the stream.
func (s *Stream) Sync() error { return call(cuStreamSynchronize(s.s), "cuStreamSynchronize") }

// WriteAtOn copies host memory in at a byte offset on st. A pageable source is
// staged by the driver before the call returns, so p may be reused at once.
func (b *Buffer) WriteAtOn(st *Stream, off int, p unsafe.Pointer, n int) error {
	if off < 0 || uint64(off)+uint64(n) > b.n {
		return fmt.Errorf("cuda: writing %d bytes at offset %d of a %d-byte buffer", n, off, b.n)
	}
	return call(cuMemcpyHtoDAsync(b.p+CUdevptr(off), p, uint64(n), st.s), "cuMemcpyHtoDAsync")
}

// ReadOn copies the buffer out on st, after everything queued on st before
// it, and returns once the bytes have landed.
func (b *Buffer) ReadOn(st *Stream, p unsafe.Pointer, n int) error {
	if err := call(cuMemcpyDtoHAsync(p, b.p, uint64(n), st.s), "cuMemcpyDtoHAsync"); err != nil {
		return err
	}
	return st.Sync()
}

// SyncLegacy waits for every launch and copy on the context's legacy stream,
// and not for any other stream's.
func SyncLegacy() error { return call(cuStreamSynchronize(0), "cuStreamSynchronize") }

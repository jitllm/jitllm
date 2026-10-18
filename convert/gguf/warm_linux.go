//go:build linux

package gguf

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/jitllm/jitllm/engine/sched"
)

// touch faults every page of b in, on the calling goroutine, and stops early
// when stopped reports true. Reading one byte per page is the whole job: each
// goroutine is a queue of one I/O. The stop is checked per page, since each
// fault may take milliseconds.
//
// It returns the bytes' xor, which the caller stores where the compiler cannot
// see it unused, so the loads are not optimised away.
func touch(b []byte, stopped func() bool) byte {
	var acc byte
	for off := 0; off < len(b); off += 4096 {
		if stopped() {
			break
		}
		acc ^= b[off]
	}
	return acc
}

// touchParallel faults b in from n goroutines at once, in the background, and
// returns a stop function the unmapper MUST call. warm (whole file, at load) is
// its caller.
//
// The parallelism is the point: an NVMe needs queue depth, a page fault is a
// queue of one, and madvise(WILLNEED) does not build enough. Eight goroutines
// fault a cold file at about twice the rate of one (see
// docs/design/block-paging.md).
//
// The stop function is not optional: touching a mapping after munmap is a
// use-after-free, and a Close during a warm would segfault.
func touchParallel(b []byte, n int) func() {
	if len(b) == 0 || n <= 0 {
		return func() {}
	}
	var stop atomic.Bool
	var wg sync.WaitGroup
	stopped := stop.Load
	// One slot per goroutine for touch's result; they escape into the
	// goroutines, so the stores stay.
	sums := make([]byte, n)
	per := (len(b) + n - 1) / n
	for w := 0; w < n; w++ {
		lo, hi := w*per, min(w*per+per, len(b))
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(w int, part []byte) {
			defer wg.Done()
			sums[w] = touch(part, stopped)
		}(w, b[lo:hi])
	}
	return func() {
		stop.Store(true)
		wg.Wait()
		runtime.KeepAlive(sums)
	}
}

// warm faults the mapping in from several goroutines at once, in the
// background, and returns a stop function the unmapper MUST call.
//
// Parallel faulting of a mapping reaches parallel pread's throughput, so the
// mapping keeps zero-copy reads and its exemption from RLIMIT_DATA.
func warm(b []byte) func() {
	if len(b) == 0 {
		return func() {}
	}
	// It must not fault past the budget, or it is a reclaim storm on a model
	// larger than the cgroup. The prefix is warmed; see warmSpan.
	b = b[:warmSpan(uint64(len(b)), sched.MemBudget())]
	return touchParallel(b, min(max(runtime.NumCPU(), 1), 8))
}

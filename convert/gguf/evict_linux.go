//go:build linux

package gguf

import (
	"syscall"

	"github.com/jitllm/jitllm/convert/meta"
)

// Evict drops a range of f's mapping from the page cache, so the memory it
// occupied is available for something jitllm would rather keep. It returns the
// bytes it actually handed back, which is zero when the range holds no whole
// page or the kernel refused.
//
// It is the one lever user space has over the page cache: jitllm cannot pin a
// page (mlock needs privilege), but by dropping cold pages itself it keeps the
// resident set under budget with knowledge the kernel's LRU lacks.
//
// MADV_DONTNEED on a read-only MAP_PRIVATE file mapping discards the pages and
// nothing else: the next access re-faults them from the file. There is no
// dirty data to lose, which is what makes this safe to do to weights and would
// not be safe to do to anything jitllm had written.
//
// madvise refuses the whole call on an unaligned start (an unaligned length is
// fine), and tensor ranges start on the GGUF's 32-byte alignment, so the range
// is narrowed to whole pages first (pageAlignIn). The returned byte count lets
// a gate assert that eviction actually happened.
func Evict(f *meta.File, b []byte) int {
	// It refuses anything that is not one of this package's mappings: on
	// anonymous memory (the Go heap) MADV_DONTNEED silently zeroes the page.
	// See mapped.go.
	if !Mapped(f, b) {
		return 0
	}
	p := pageAlignIn(b)
	if len(p) == 0 {
		return 0
	}
	// Advisory, once the request is well formed: a refusal then costs a page
	// that stays cached, not a wrong answer.
	if err := syscall.Madvise(p, syscall.MADV_DONTNEED); err != nil {
		return 0
	}
	return len(p)
}

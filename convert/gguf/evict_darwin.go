//go:build darwin

package gguf

import (
	"syscall"
	"unsafe"

	"github.com/jitllm/jitllm/convert/meta"
)

// Evict is the darwin half of evict_linux.go's madvise: MADV_DONTNEED has the
// same semantics on a read-only MAP_PRIVATE file mapping. It matters more on
// Apple Silicon, where the packed arena is imported by the GPU rather than
// uploaded, so the GGUF pages are a redundant copy in the one shared pool.
//
// syscall.Madvise is not bound on darwin, so this goes through syscall.Syscall
// with SYS_MADVISE. A refusal returns 0, as on Linux: a page that stays cached,
// not a wrong answer.
func Evict(f *meta.File, b []byte) int {
	// Mapped, for the reason evict_linux.go gives: MADV_DONTNEED on anonymous
	// memory discards the page and the next read returns zeros, with no error
	// and no fault, and a []byte does not say which kind it is.
	if !Mapped(f, b) {
		return 0
	}
	p := pageAlignIn(b)
	if len(p) == 0 {
		return 0
	}
	if _, _, errno := syscall.Syscall(syscall.SYS_MADVISE,
		uintptr(unsafe.Pointer(&p[0])), uintptr(len(p)), uintptr(syscall.MADV_DONTNEED)); errno != 0 {
		return 0
	}
	return len(p)
}

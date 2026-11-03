package gguf

import "github.com/jitllm/jitllm/convert/meta"

// Mapped reports whether b lies entirely inside f's file mapping. It is what
// makes Evict safe to offer a []byte of unknown provenance.
//
// MADV_DONTNEED is safe on a read-only private file mapping (clean pages
// re-fault from disk) and destructive on anonymous memory (the Go heap: the
// next read silently returns zeros), and the two are the same []byte to a
// caller. Callers such as the packed arena are handed opaque slices, so the
// check lives here, where the mapping was made: a range outside it gets zero
// and no syscall.
func Mapped(f *meta.File, b []byte) bool {
	if len(b) == 0 || f == nil || len(f.Data) == 0 {
		return false
	}
	lo := uintptr(ptr(b))
	hi := lo + uintptr(len(b))
	flo := uintptr(ptr(f.Data))
	return lo >= flo && hi <= flo+uintptr(len(f.Data))
}

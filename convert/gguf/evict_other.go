//go:build !linux && !darwin

package gguf

import "github.com/jitllm/jitllm/convert/meta"

// Evict is a no-op where Go's syscall package has no Madvise, and reports zero
// bytes handed back. Residency then falls back to the kernel's own policy.
func Evict(f *meta.File, b []byte) int { return 0 }

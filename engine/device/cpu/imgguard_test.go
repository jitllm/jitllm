//go:build (amd64 || arm64) && (linux || darwin)

package cpu

import (
	"syscall"
	"testing"
)

// imgGuard places src so its last byte is the last before a PROT_NONE page:
// an image kernel that reads one byte past a pixel, a row or a table faults
// here, where on a Go heap slice it would read the next allocation and every
// numeric check would stay green.
func imgGuard(t *testing.T, src []byte) []byte {
	t.Helper()
	pg := syscall.Getpagesize()
	n := len(src)
	body := (n + pg - 1) / pg * pg
	if body == 0 {
		body = pg
	}
	m, err := syscall.Mmap(-1, 0, body+pg, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	if err := syscall.Mprotect(m[body:], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { syscall.Munmap(m) })
	b := m[body-n : body : body]
	copy(b, src)
	return b
}

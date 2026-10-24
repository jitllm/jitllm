//go:build linux

package probe

import (
	"syscall"
	"unsafe"
)

const madvHugepage = 14 // MADV_HUGEPAGE

// hugeBuffer is n bytes of anonymous memory on huge pages where the kernel
// grants them, as the engine's frames ask for, and its release: mapped 2 MB
// over and cut to a 2 MB boundary, since the kernel backs only aligned
// ranges with huge pages.
func hugeBuffer(n int) ([]byte, func(), error) {
	const huge = 2 << 20
	b, err := syscall.Mmap(-1, 0, n+huge, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		return nil, nil, err
	}
	skip := int((huge - uintptr(unsafe.Pointer(&b[0]))%huge) % huge)
	out := b[skip : skip+n]
	syscall.Madvise(out, madvHugepage)
	return out, func() { syscall.Munmap(b) }, nil
}

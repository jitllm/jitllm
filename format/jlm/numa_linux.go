//go:build linux

package jlm

import (
	"syscall"
	"unsafe"
)

const (
	mpolBind       = 2
	mpolInterleave = 3
	mpolMFMove     = 1 << 1
)

const madvHugepage = 14 // MADV_HUGEPAGE

// placeMemory applies a File's Placement to b: huge-page advice on its 2 MB
// interior, and the interleave policy on its whole pages. MPOL_MF_MOVE migrates
// any already present -- a buffer the allocator reused, or the dense region read
// at Open -- so memory is spread even when it is not fresh. Best effort: a
// refusal leaves the kernel's default, which is correct and only slower.
//
// It asks for huge pages because the packed layout walks a plane at a time. A
// payload word of every row is one plane, nrows*4 bytes, so the decode
// kernel's streams jump tens of KB on a wide matrix and nearly every access
// crosses a 4 KB page. With the frames on 4 KB pages (a kernel THP mode of
// `madvise` leaves them there unless asked) page-table walks cost several
// percent of every cycle.
func placeMemory(b []byte, noHuge bool, mask []uint64) {
	if len(b) == 0 {
		return
	}
	if !noHuge {
		const huge = 2 << 20
		start := (uintptr(unsafe.Pointer(&b[0])) + huge - 1) &^ (huge - 1)
		end := (uintptr(unsafe.Pointer(&b[0])) + uintptr(len(b))) &^ (huge - 1)
		if end > start {
			syscall.Syscall(syscall.SYS_MADVISE, start, end-start, madvHugepage)
		}
	}
	if mask == nil {
		return
	}
	const page = 4096
	start := (uintptr(unsafe.Pointer(&b[0])) + page - 1) &^ (page - 1)
	end := (uintptr(unsafe.Pointer(&b[0])) + uintptr(len(b))) &^ (page - 1)
	if end <= start {
		return
	}
	syscall.Syscall6(syscall.SYS_MBIND, start, end-start, mpolInterleave,
		uintptr(unsafe.Pointer(&mask[0])), uintptr(len(mask)*64+1), mpolMFMove)
}

// bindNode places b's whole pages on one node (MPOL_BIND), migrating any
// already there: an expert page is read whole by the expert it holds, so it
// lives where that expert is computed (File.claim). Best effort, as
// placeMemory.
func bindNode(b []byte, mask []uint64) {
	if len(b) == 0 || len(mask) == 0 {
		return
	}
	const page = 4096
	start := (uintptr(unsafe.Pointer(&b[0])) + page - 1) &^ (page - 1)
	end := (uintptr(unsafe.Pointer(&b[0])) + uintptr(len(b))) &^ (page - 1)
	if end <= start {
		return
	}
	syscall.Syscall6(syscall.SYS_MBIND, start, end-start, mpolBind,
		uintptr(unsafe.Pointer(&mask[0])), uintptr(len(mask)*64+1), mpolMFMove)
}

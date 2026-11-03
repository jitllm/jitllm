//go:build linux || darwin

// The hosts where Evict calls madvise; elsewhere it is evict_other.go's no-op
// and there is no page narrowing to gate.

package gguf

import (
	"os"
	"testing"
	"unsafe"
)

// TestEvictNarrowsToWholePages: madvise refuses the whole call on an unaligned
// start, and a GGUF tensor is only 32-byte aligned. The narrowed range must be
// page-aligned at both ends, inside the original, and the syscall must succeed.
func TestEvictNarrowsToWholePages(t *testing.T) {
	f := open(t, stories260K)
	ps := uint64(os.Getpagesize())

	// A deliberately unaligned span in the middle of the mapping, which is what
	// every LayerBytes and every expert bank looks like.
	base := uintptr(unsafe.Pointer(&f.Data[0]))
	for _, skew := range []uint64{0, 1, 32, 4095, ps, ps + 32} {
		lo := skew
		hi := lo + 3*ps
		if hi > uint64(len(f.Data)) {
			t.Fatalf("model too small for this test: %d bytes", len(f.Data))
		}
		b := f.Data[lo:hi]
		p := pageAlignIn(b)
		if len(p) == 0 {
			t.Fatalf("skew %d: pageAlignIn gave nothing for a %d byte span", skew, len(b))
		}
		pl := uint64(uintptr(unsafe.Pointer(&p[0])) - base)
		if (pl+uint64(base))%ps != 0 {
			t.Fatalf("skew %d: narrowed start is not page-aligned", skew)
		}
		if uint64(uintptr(unsafe.Pointer(&p[0]))+uintptr(len(p)))%ps != 0 {
			t.Fatalf("skew %d: narrowed end is not page-aligned", skew)
		}
		if pl < lo || pl+uint64(len(p)) > hi {
			t.Fatalf("skew %d: narrowed range [%d,%d) escapes [%d,%d)",
				skew, pl, pl+uint64(len(p)), lo, hi)
		}
		if n := Evict(f, b); n != len(p) {
			t.Fatalf("skew %d: Evict handed back %d bytes, want %d -- "+
				"a zero here is madvise refusing the call, not the page cache declining",
				skew, n, len(p))
		}
	}

	// A span too short to hold a whole page evicts nothing and says so, rather
	// than making a call that would be refused.
	if n := Evict(f, f.Data[33:35]); n != 0 {
		t.Fatalf("Evict of a 2-byte span handed back %d bytes", n)
	}
	if n := Evict(f, nil); n != 0 {
		t.Fatalf("Evict(f, nil) handed back %d bytes", n)
	}
}

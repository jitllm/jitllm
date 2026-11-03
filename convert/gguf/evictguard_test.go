//go:build linux

package gguf

import (
	"os"
	"syscall"
	"testing"
)

// TestEvictRefusesMemoryItDidNotMap is the gate on the guard that makes Evict
// safe to hand an opaque []byte.
//
// The first half demonstrates the hazard on memory of its own: MADV_DONTNEED
// on anonymous memory silently zeroes the page (see mapped.go). The second half
// is the guard: a range this package did not map gets zero bytes and no
// syscall.
func TestEvictRefusesMemoryItDidNotMap(t *testing.T) {
	ps := os.Getpagesize()
	anon, err := syscall.Mmap(-1, 0, 4*ps, syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Skipf("no anonymous mmap on this host: %v", err)
	}
	defer syscall.Munmap(anon)
	for i := range anon {
		anon[i] = 0xAB
	}

	// The hazard, on memory owned by this test and nothing else.
	if err := syscall.Madvise(anon[ps:2*ps], syscall.MADV_DONTNEED); err != nil {
		t.Fatalf("madvise on anonymous memory: %v", err)
	}
	zeroed := 0
	for _, b := range anon[ps : 2*ps] {
		if b == 0 {
			zeroed++
		}
	}
	if zeroed != ps {
		t.Fatalf("MADV_DONTNEED left %d of %d anonymous bytes intact: this kernel does not "+
			"have the behaviour the guard exists for, so re-derive the guard rather than "+
			"trusting this test", ps-zeroed, ps)
	}

	// The guard: the same call through Evict does nothing at all, because this
	// package did not map it.
	for i := range anon {
		anon[i] = 0xCD
	}
	if n := Evict(open(t, stories260K), anon); n != 0 {
		t.Fatalf("Evict handed back %d bytes of memory gguf never mapped -- on a Go heap "+
			"slice that is a hole in the heap, not a page cache saving", n)
	}
	for i, b := range anon {
		if b != 0xCD {
			t.Fatalf("Evict zeroed byte %d of memory it should have refused", i)
		}
	}

	// And a range this package did map is still evicted, or the guard has
	// simply turned the lever off.
	f := open(t, stories260K)
	if n := Evict(f, f.Data); n == 0 {
		t.Fatalf("Evict refused a live gguf mapping of %d bytes: the guard rejects the "+
			"ranges it exists to allow", len(f.Data))
	}
	if !Mapped(f, f.Data[7:99]) {
		t.Fatalf("a subslice of a live mapping reads as unmapped")
	}
	f.Close()
	if Mapped(f, f.Data) {
		t.Fatalf("a CLOSED file's range still reads as mapped: the registry leaks, and the " +
			"address will eventually be reused by something that must not be evicted")
	}
}

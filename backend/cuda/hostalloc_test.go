package cuda

import (
	"testing"
	"unsafe"
)

// TestHostAllocIsWritableAndTransfers: page-locked memory comes back the size
// asked, the host can write all of it, and a transfer out of it lands the same
// bytes on the card as one out of the Go heap.
func TestHostAllocIsWritableAndTransfers(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skip("no CUDA device:", err)
	}
	defer d.Close()
	const n = 3<<20 + 5
	pin, err := HostAlloc(n)
	if err != nil {
		t.Fatalf("HostAlloc(%d): %v", n, err)
	}
	defer FreeHost(pin)
	if len(pin) != n {
		t.Fatalf("HostAlloc(%d) is %d bytes", n, len(pin))
	}
	for i := range pin {
		pin[i] = byte(i*7 + i>>9)
	}
	b, err := d.Alloc(n)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Free()
	if err := b.WriteAt(0, unsafe.Pointer(&pin[0]), n); err != nil {
		t.Fatalf("a transfer out of page-locked memory: %v", err)
	}
	back := make([]byte, n)
	if err := b.Read(unsafe.Pointer(&back[0]), n); err != nil {
		t.Fatal(err)
	}
	for i := range back {
		if back[i] != pin[i] {
			t.Fatalf("byte %d is %d on the card, %d on the host", i, back[i], pin[i])
		}
	}
	if _, err := HostAlloc(0); err == nil {
		t.Error("HostAlloc(0) succeeded")
	}
}

// BenchmarkHostToDevicePinned and BenchmarkHostToDevicePageable are the
// link's rate for one 64 MiB transfer out of page-locked memory and out of the
// Go heap: the ceiling a streamed expert fill is measured against
// (placement.md 16c). Separate benchmarks, not sub-benchmarks: the context is
// current on the thread Open locked, and a sub-benchmark runs on another.
func BenchmarkHostToDevicePinned(b *testing.B)   { benchH2D(b, true) }
func BenchmarkHostToDevicePageable(b *testing.B) { benchH2D(b, false) }

func benchH2D(b *testing.B, pinned bool) {
	d, err := Open()
	if err != nil {
		b.Skip("no CUDA device:", err)
	}
	defer d.Close()
	const n = 64 << 20
	buf, err := d.Alloc(n)
	if err != nil {
		b.Fatal(err)
	}
	defer buf.Free()
	src := make([]byte, n)
	if pinned {
		if src, err = HostAlloc(n); err != nil {
			b.Fatal(err)
		}
		defer FreeHost(src)
	}
	b.SetBytes(n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := buf.WriteAt(0, unsafe.Pointer(&src[0]), n); err != nil {
			b.Fatal(err)
		}
	}
}

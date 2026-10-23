package backend_test

import (
	"bytes"
	"testing"

	"github.com/jitllm/jitllm/backend"
)

// TestWriteAtAssemblesAndRefusesAnOverrun builds a device buffer out of runs
// that are not adjacent on the host, as a routed mixture's compact expert bank
// (k scattered sheets laid back to back) needs.
//
// The bound is the other half of the gate. Without the checks the three
// backends fail three different ways:
//
//	ptx    the driver refuses: CUDA_ERROR_INVALID_VALUE.
//	spirv  the staged path loses the device (VK_ERROR_DEVICE_LOST).
//	mapped the mapped paths (integrated GPUs, Apple) are a plain Go copy that
//	       truncates silently.
//
// So every backend must refuse rather than write, and leave the buffer
// unchanged.
func TestWriteAtAssemblesAndRefusesAnOverrun(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}

	// Three runs of different lengths, deliberately not written in order: an
	// ignored dstOffset would leave the last write at 0, which ascending order
	// would partly hide.
	const n = 4096
	runs := []struct {
		off int
		b   []byte
	}{
		{off: 1024, b: bytes.Repeat([]byte{0xA1}, 512)},
		{off: 0, b: bytes.Repeat([]byte{0xB2}, 1024)},
		{off: 1536, b: bytes.Repeat([]byte{0xC3}, n-1536)},
	}
	want := make([]byte, n)
	for _, r := range runs {
		copy(want[r.off:], r.b)
	}

	for _, d := range devs {
		// Both entry points, which are different code on CUDA: Buf's hops to
		// the device-owning goroutine per call, Session's is already on it.
		for _, arm := range []struct {
			name  string
			write func(b backend.Buf, off int, p []byte) error
		}{
			{"buf", func(b backend.Buf, off int, p []byte) error { return b.WriteAt(off, p) }},
			{"session", func(b backend.Buf, off int, p []byte) error {
				var err error
				d.Session(func(s backend.Session) { err = s.WriteAt(b, off, p) })
				return err
			}},
		} {
			t.Run(d.API()+"/"+arm.name, func(t *testing.T) {
				b, err := d.Alloc(n)
				if err != nil {
					t.Fatalf("alloc: %v", err)
				}
				defer b.Free()
				// Poison, so a run that never lands is a wrong byte rather than
				// a zero the allocator happened to supply (RULE 13).
				if err := b.Write(bytes.Repeat([]byte{0x5A}, n)); err != nil {
					t.Fatalf("poison: %v", err)
				}
				for _, r := range runs {
					if err := arm.write(b, r.off, r.b); err != nil {
						t.Fatalf("WriteAt(%d, %d bytes): %v", r.off, len(r.b), err)
					}
				}
				got := make([]byte, n)
				if err := b.Read(got); err != nil {
					t.Fatalf("read: %v", err)
				}
				if !bytes.Equal(got, want) {
					for i := range got {
						if got[i] != want[i] {
							t.Fatalf("byte %d is %#x, want %#x: a run landed at the "+
								"wrong offset", i, got[i], want[i])
						}
					}
				}

				// One byte past the end, and an offset past the end entirely.
				if err := b.WriteAt(n-3, []byte{1, 2, 3, 4}); err == nil {
					t.Errorf("WriteAt wrote 4 bytes at offset %d of a %d-byte "+
						"buffer without refusing", n-3, n)
				}
				if err := b.WriteAt(n+8, []byte{1}); err == nil {
					t.Errorf("WriteAt accepted an offset past the end of the buffer")
				}
				if err := b.WriteAt(-1, []byte{1}); err == nil {
					t.Errorf("WriteAt accepted a negative offset")
				}
				// The refusals must not have written anything.
				if err := b.Read(got); err != nil {
					t.Fatalf("read: %v", err)
				}
				if !bytes.Equal(got, want) {
					t.Errorf("a refused WriteAt still changed the buffer")
				}
			})
		}
	}
}

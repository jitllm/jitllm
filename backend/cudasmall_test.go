package backend_test

import (
	"bytes"
	"testing"

	"github.com/jitllm/jitllm/backend"
)

// TestSmallBuffersShareAPiece: on CUDA a buffer of 16 KiB or less is a slot
// of a 64 KiB piece shared with buffers of its size, so forty small buffers
// cost the card a few pieces, not forty, and every one holds its own bytes.
// Freed, they give every piece back. The violation is a buffer a piece of its
// own: forty pieces.
func TestSmallBuffersShareAPiece(t *testing.T) {
	gpuLock(t)
	ran, devs := 0, everyDevice(t)
	for _, d := range devs {
		ac, ok := d.(backend.AllocCounter)
		if d.API() != "ptx" || !ok {
			continue
		}
		ran++
		t.Run(d.Name(), func(t *testing.T) {
			sizes := []int{4, 100, 1000, 4096, 9000, 16384, 256, 3}
			f0, a0 := ac.Footprint(), ac.Allocated()
			var bufs []backend.Buf
			var want [][]byte
			asked := 0
			for i := range 40 {
				n := sizes[i%len(sizes)]
				b, err := d.Alloc(n)
				if err != nil {
					t.Fatal(err)
				}
				p := bytes.Repeat([]byte{byte(i + 1)}, n)
				if err := b.Write(p); err != nil {
					t.Fatal(err)
				}
				bufs, want, asked = append(bufs, b), append(want, p), asked+n
			}
			for i, b := range bufs {
				got := make([]byte, len(want[i]))
				if err := b.Read(got); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got, want[i]) {
					t.Fatalf("buffer %d of %d bytes does not hold its own bytes: another's slot overlaps it", i, len(want[i]))
				}
			}
			if got := ac.Allocated() - a0; got != uint64(asked) {
				t.Errorf("allocated %d bytes for %d asked", got, asked)
			}
			// Seven classes: no more than one piece each for these sizes.
			if got := ac.Footprint() - f0; got > 7*64<<10 {
				t.Errorf("forty small buffers cost the card %d bytes (%d pieces): a piece a buffer", got, got/(64<<10))
			}
			t.Logf("forty small buffers, %d bytes asked: %d bytes of pieces", asked, ac.Footprint()-f0)
			for _, b := range bufs {
				b.Free()
			}
			if f, a := ac.Footprint(), ac.Allocated(); f != f0 || a != a0 {
				t.Errorf("freed, the card holds %d bytes of pieces for %d allocated, %d and %d before", f, a, f0, a0)
			}
		})
	}
	if ran == 0 {
		t.Fatalf("no CUDA device in %d devices: the gate ran nothing", len(devs))
	}
}

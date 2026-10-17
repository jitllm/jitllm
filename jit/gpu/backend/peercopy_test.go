package backend

import (
	"bytes"
	"fmt"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestCopyPeerMovesTheBytes copies a pattern between two CUDA cards through
// PeerCopier at an offset on each side and reads it back: the bytes arrive
// where they were sent and the bytes around them are untouched. It logs
// whether the cards addressed each other (direct) and the rate, which is the
// number an expert card's cache misses ride on; no rate is asserted.
func TestCopyPeerMovesTheBytes(t *testing.T) {
	specs := testdevices.Take(t, 2, "cuda:0,cuda:1")
	var devs [2]Device
	for i := range devs {
		ord := 0
		if _, err := fmt.Sscanf(specs[i], "cuda:%d", &ord); err != nil {
			t.Fatalf("%q is not a cuda:N spec", specs[i])
		}
		d, err := OpenCUDA(ord)
		if err != nil {
			t.Skipf("%s: %v", specs[i], err)
		}
		defer d.Close()
		devs[i] = d
	}
	pc, ok := devs[1].(PeerCopier)
	if !ok {
		t.Skip("this backend has no PeerCopier")
	}
	const n = 6 << 20
	src, err := devs[0].Alloc(n + 8)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Free()
	dst, err := devs[1].Alloc(n + 16)
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Free()
	pat := make([]byte, n+8)
	for i := range pat {
		pat[i] = byte(i*7 + i>>8)
	}
	if err := src.Write(pat); err != nil {
		t.Fatal(err)
	}
	if err := dst.Write(make([]byte, n+16)); err != nil {
		t.Fatal(err)
	}
	direct, err := pc.CopyPeer(dst, 8, devs[0], src, 4, n)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]byte, n+16)
	if err := dst.Read(got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got[8:8+n], pat[4:4+n]) || !bytes.Equal(got[:8], make([]byte, 8)) || !bytes.Equal(got[8+n:], make([]byte, 8)) {
		t.Fatal("the copy landed wrong bytes, or wrote outside its range")
	}
	const reps = 20
	t0 := time.Now()
	for range reps {
		if _, err := pc.CopyPeer(dst, 8, devs[0], src, 4, n); err != nil {
			t.Fatal(err)
		}
	}
	el := time.Since(t0)
	t.Logf("%s -> %s: direct=%v, %d MiB x %d in %v = %.2f GiB/s per copy call", specs[0], specs[1], direct, n>>20, reps, el,
		float64(n)*reps/el.Seconds()/(1<<30))
}

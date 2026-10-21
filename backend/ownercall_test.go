package backend_test

import (
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/backend"
)

// TestADeviceCallFromInsideASessionCompletes holds the CUDA backend to never
// deadlocking on a Free, Alloc, Write or Read issued from inside a Session,
// whose callback runs on the owner goroutine. The backend runs such a call
// inline; the test times out rather than hanging the suite if that regresses.
func TestADeviceCallFromInsideASessionCompletes(t *testing.T) {
	gpuLock(t)
	if n, err := backend.CUDACount(); err != nil || n < 1 {
		t.Skip("no CUDA device here:", err)
	}
	d, err := backend.OpenCUDA(testdevices.Ordinal(t, "cuda", 0))
	if err != nil {
		t.Fatalf("OpenCUDA: %v", err)
	}
	defer d.Close()
	victim, err := d.Alloc(256)
	if err != nil {
		t.Fatal(err)
	}
	before := backend.OwnerCalls()
	done := make(chan error, 1)
	go func() {
		var err error
		d.Session(func(s backend.Session) {
			victim.Free()
			var b backend.Buf
			if b, err = d.Alloc(64); err != nil {
				return
			}
			if err = b.Write(make([]byte, 64)); err != nil {
				return
			}
			b.Free()
		})
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("a device call inside a Session failed: %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("a Free/Alloc/Write issued inside a Session did not return in 30 s: the owner goroutine is " +
			"waiting on itself (the deadlock that hung gemma-3-27b in the split tuner)")
	}
	if n := backend.OwnerCalls() - before; n < 4 {
		t.Fatalf("%d owner-thread call(s) counted, want 4: the calls did not take the inline path", n)
	}
	// And the device still works from outside, so the inline calls left the
	// owner goroutine alive and the context sound.
	b, err := d.Alloc(16)
	if err != nil {
		t.Fatalf("Alloc after the session: %v", err)
	}
	want := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	got := make([]byte, 16)
	if err := b.Write(want); err != nil {
		t.Fatal(err)
	}
	if err := b.Read(got); err != nil || string(got) != string(want) {
		t.Fatalf("round trip after the session: %v %v", got, err)
	}
	b.Free()
}

// TestCUDASessionRunsOnTheCallersThread: a Session from an ordinary goroutine
// runs on that goroutine's thread under the device lock (cudaDev.inline), with
// no hop to the owner goroutine. Opened with Opts.CUDAPosted the same Session
// posts, and this fails.
func TestCUDASessionRunsOnTheCallersThread(t *testing.T) {
	if n, err := backend.CUDACount(); err != nil || n < 1 {
		t.Skip("no CUDA device here:", err)
	}
	d, err := backend.OpenCUDA(testdevices.Ordinal(t, "cuda", 0))
	if err != nil {
		t.Fatalf("OpenCUDA: %v", err)
	}
	defer d.Close()
	b, err := d.Alloc(64)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Free()
	before := backend.PostedCalls()
	for i := 0; i < 8; i++ {
		d.Session(func(s backend.Session) {
			if err := s.Write(b, make([]byte, 64)); err != nil {
				t.Error(err)
			}
		})
	}
	if n := backend.PostedCalls() - before; n != 0 {
		t.Fatalf("%d of 8 Sessions were posted to the owner goroutine: the inline path was not taken", n)
	}
}

// TestCUDAPostingIsPerDevice: two contexts in one process, one opened posted
// and one inline, each keep their own mode whichever was opened last.
func TestCUDAPostingIsPerDevice(t *testing.T) {
	if n, err := backend.CUDACount(); err != nil || n < 1 {
		t.Skip("no CUDA device here:", err)
	}
	session := func(d backend.Device) int64 {
		b, err := d.Alloc(64)
		if err != nil {
			t.Fatal(err)
		}
		defer b.Free()
		before := backend.PostedCalls()
		for i := 0; i < 8; i++ {
			d.Session(func(s backend.Session) {
				if err := s.Write(b, make([]byte, 64)); err != nil {
					t.Error(err)
				}
			})
		}
		return backend.PostedCalls() - before
	}
	for _, postedFirst := range []bool{true, false} {
		open := func(posted bool) backend.Device {
			d, err := backend.OpenCUDAWith(testdevices.Ordinal(t, "cuda", 0), backend.Opts{CUDAPosted: posted})
			if err != nil {
				t.Fatalf("OpenCUDAWith(0): %v", err)
			}
			return d
		}
		a, b := open(postedFirst), open(!postedFirst)
		posted, inline := a, b
		if !postedFirst {
			posted, inline = b, a
		}
		if n := session(inline); n != 0 {
			t.Errorf("posted opened first=%v: the inline device posted %d of 8 Sessions", postedFirst, n)
		}
		if n := session(posted); n < 8 {
			t.Errorf("posted opened first=%v: the posted device posted %d calls, want at least 8", postedFirst, n)
		}
		a.Close()
		b.Close()
	}
}

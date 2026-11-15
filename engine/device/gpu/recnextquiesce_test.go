package gpu

import (
	"testing"
	"time"
)

// TestRecNextWaitsForASubmissionInFlight: ensureRecNext replacing the
// next-state buffer every session's linear blocks share frees the old one
// only once no submission is in flight -- one run alone with the device's
// lock let go (aloneUnlocked) may still write it -- and then replaces it; a
// buffer big enough is kept.
func TestRecNextWaitsForASubmissionInFlight(t *testing.T) {
	fd := &fakeDev{}
	g, err := New([]Slot{{Dev: fd, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(g.Close)
	d := g.dev(0)
	d.mu.Lock()
	d.recS = 8
	if !d.ensureRecNext(4) {
		d.mu.Unlock()
		t.Fatalf("the first buffer: %s", d.LastErr)
	}
	old, freed := d.recNext, fd.freed
	if !d.ensureRecNext(2) || d.recNext != old {
		d.mu.Unlock()
		t.Fatal("a buffer big enough was replaced")
	}
	tk := d.beginSub()
	d.mu.Unlock()
	done := make(chan bool, 1)
	go func() {
		d.mu.Lock()
		defer d.mu.Unlock()
		done <- d.ensureRecNext(64)
	}()
	select {
	case <-done:
		// The submission ends either way, or Close waits on it for ever.
		d.mu.Lock()
		d.endSub(tk)
		d.mu.Unlock()
		t.Fatal("ensureRecNext replaced the buffer while a submission was in flight")
	case <-time.After(300 * time.Millisecond):
	}
	d.mu.Lock()
	if replaced, n := d.recNext != old, fd.freed-freed; replaced || n != 0 {
		d.endSub(tk)
		d.mu.Unlock()
		t.Fatalf("under the submission in flight the buffer was replaced (%v) or freed (%d buffers)", replaced, n)
	}
	d.endSub(tk)
	d.mu.Unlock()
	select {
	case ok := <-done:
		if !ok {
			t.Fatalf("the replacement failed: %s", d.LastErr)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("ensureRecNext still waits with no submission in flight")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.recNext == old || fd.freed == freed || d.recNextBytes < uint64(64*d.recS*4) {
		t.Fatalf("the buffer was not replaced: %d bytes, %d freed", d.recNextBytes, fd.freed-freed)
	}
}

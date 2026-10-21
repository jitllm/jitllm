package cuda

import (
	"testing"
	"time"
)

// TestFFIOverhead measures what one FFI call into the driver costs, and what
// the device-owner channel hop adds on top: microseconds would be a real share
// of a token, nanoseconds would put the tier's overhead elsewhere.
func TestFFIOverhead(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skip("no CUDA device:", err)
	}
	// Close is required: Open locks this goroutine to its thread with a live
	// context, and exiting locked destroys that thread under the driver
	// (see api.go's Close).
	defer d.Close()

	// cuDriverGetVersion is the cheapest call the driver has: it reads a global
	// and returns. Whatever it costs IS the FFI overhead.
	var v int32
	const n = 200000
	for i := 0; i < 1000; i++ {
		cuDriverGetVersion(&v)
	}
	start := time.Now()
	for i := 0; i < n; i++ {
		cuDriverGetVersion(&v)
	}
	per := time.Since(start) / n
	t.Logf("ffi call (cuDriverGetVersion): %v", per)

	// A Go call of the same shape, for the floor.
	sink := int32(0)
	goCall := func(p *int32) { *p = 1 }
	start = time.Now()
	for i := 0; i < n; i++ {
		goCall(&sink)
	}
	t.Logf("plain Go indirect call:           %v", time.Since(start)/n)

	// The channel hop backend.cudaDev.do() adds.
	reqs := make(chan func())
	go func() {
		for f := range reqs {
			f()
		}
	}()
	do := func(f func()) {
		done := make(chan struct{})
		reqs <- func() { f(); close(done) }
		<-done
	}
	for i := 0; i < 1000; i++ {
		do(func() {})
	}
	start = time.Now()
	for i := 0; i < n/10; i++ {
		do(func() { cuDriverGetVersion(&v) })
	}
	t.Logf("same call through the owner hop:  %v", time.Since(start)/(n/10))
	close(reqs)
}

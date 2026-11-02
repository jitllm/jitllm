package kernels

import (
	"math/rand"
	"testing"
)

// TestScStreamRoundTrips holds ScField/ScValue to scPut for all sixteen fields
// with every value, and to the stream's own definition: field f is stream bits
// [6f, 6f+6), words low first, so the three words are exactly full.
func TestScStreamRoundTrips(t *testing.T) {
	rng := rand.New(rand.NewSource(27))
	for trial := 0; trial < 2000; trial++ {
		var ws [ScStreamWords]uint32
		var want [16]uint8
		for f := range want {
			want[f] = uint8(rng.Intn(64))
			scPut(&ws, f/2, f%2 == 1, want[f])
		}
		for f, v := range want {
			if got := ScValue(ws, f/2, f%2 == 1); got != v {
				t.Fatalf("field %d: got %d want %d (words %08x)", f, got, v, ws)
			}
			var bits uint8
			for i := 0; i < 6; i++ {
				pos := 6*f + i
				bits |= uint8(ws[pos/32]>>uint(pos%32)&1) << uint(i)
			}
			if bits != v {
				t.Fatalf("field %d is not stream bits [%d,%d)", f, 6*f, 6*f+6)
			}
		}
	}
	// Exactly two fields straddle, and only ever into the next word.
	n := 0
	for f := 0; f < 16; f++ {
		if w, _, hi := ScField(f/2, f%2 == 1); hi > 0 {
			n++
			if w+1 >= ScStreamWords {
				t.Fatalf("field %d straddles past the last word", f)
			}
		}
	}
	if n != 2 {
		t.Fatalf("%d straddling fields, want 2", n)
	}
}

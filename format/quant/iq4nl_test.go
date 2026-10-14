package quant

import (
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestIQ4NLMatchesGGUFPy holds the IQ4_NL decoder to an implementation that
// is not this one: the gguf package's IQ4_NL.dequantize_blocks, written out by
// scripts/iq4nlgold.py over every code in every slot at a sweep of d that
// reaches subnormal halves, the largest finite one and negatives.
//
// Bits, not values: d times an int8 is exact in f32, so anything but equality
// is a bug -- the table misread, the nibbles interleaved, the sign dropped.
func TestIQ4NLMatchesGGUFPy(t *testing.T) {
	raw, err := os.ReadFile("testdata/iq4_nl.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Source string
		Blocks string
		Bits   []uint32
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	src, err := hex.DecodeString(g.Blocks)
	if err != nil {
		t.Fatal(err)
	}
	if len(src)%18 != 0 || len(g.Bits) != len(src)/18*32 || len(g.Bits) == 0 {
		t.Fatalf("golden is %d bytes and %d values", len(src), len(g.Bits))
	}
	got32 := make([]float32, len(g.Bits))
	if err := Dequant32(IQ4_NL, src, got32); err != nil {
		t.Fatal(err)
	}
	got64 := make([]float64, len(g.Bits))
	if err := Dequant(IQ4_NL, src, got64); err != nil {
		t.Fatal(err)
	}
	nonzero, negative := 0, 0
	for i, want := range g.Bits {
		w := math.Float32frombits(want)
		if math.Float32bits(got32[i]) != want {
			t.Fatalf("element %d (block %d): got %#08x (%g), gguf-py %#08x (%g)",
				i, i/32, math.Float32bits(got32[i]), got32[i], want, w)
		}
		if !math.IsInf(float64(w), 0) && float32(got64[i]) != w {
			t.Fatalf("element %d: f64 %v rounds to %v, want %v", i, got64[i], float32(got64[i]), w)
		}
		if want&0x7fffffff != 0 {
			nonzero++
		}
		if w < 0 {
			negative++
		}
	}
	if nonzero < len(g.Bits)/2 || negative < len(g.Bits)/8 {
		t.Fatalf("%d nonzero and %d negative of %d golden values -- the golden is degenerate",
			nonzero, negative, len(g.Bits))
	}
	if !Dequantable(IQ4_NL) {
		t.Fatal("IQ4_NL decodes and Dequantable says otherwise")
	}
	t.Logf("%d blocks bit-identical to gguf-py (%s)", len(src)/18, g.Source)
}

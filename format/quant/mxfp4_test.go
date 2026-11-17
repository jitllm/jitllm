package quant

import (
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestMXFP4MatchesGGUFPy holds the decoder to an implementation that is not
// this one: the pip gguf package's MXFP4.dequantize_blocks, written out by
// scripts/mxfp4gold.py over every code at seventeen exponents -- the two
// subnormal ones and the top of the range included -- plus the first 64 blocks
// of gpt-oss-20b's blk.0.ffn_down_exps.
//
// Bits, not values: the decoding is exact, so anything but equality is a bug.
func TestMXFP4MatchesGGUFPy(t *testing.T) {
	raw, err := os.ReadFile("testdata/mxfp4.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
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
	if len(src)%17 != 0 || len(g.Bits) != len(src)/17*32 || len(g.Bits) == 0 {
		t.Fatalf("golden is %d bytes and %d values", len(src), len(g.Bits))
	}
	got32 := make([]float32, len(g.Bits))
	if err := Dequant32(MXFP4, src, got32); err != nil {
		t.Fatal(err)
	}
	got64 := make([]float64, len(g.Bits))
	if err := Dequant(MXFP4, src, got64); err != nil {
		t.Fatal(err)
	}
	nonzero := 0
	for i, want := range g.Bits {
		if math.Float32bits(got32[i]) != want {
			t.Fatalf("element %d (block %d, e=%d): got %#08x (%g), golden %#08x (%g)",
				i, i/32, src[i/32*17], math.Float32bits(got32[i]), got32[i], want, math.Float32frombits(want))
		}
		// The f64 destination must be the same value wherever f32 holds it.
		if w := math.Float32frombits(want); !math.IsInf(float64(w), 0) && got64[i] != float64(w) {
			t.Fatalf("element %d: f64 %v, want %v", i, got64[i], w)
		}
		if want&0x7fffffff != 0 {
			nonzero++
		}
	}
	if nonzero < len(g.Bits)/2 {
		t.Fatalf("only %d of %d golden values are nonzero -- the golden is degenerate", nonzero, len(g.Bits))
	}
	if !Dequantable(MXFP4) {
		t.Fatal("MXFP4 decodes and Dequantable says otherwise")
	}
	t.Logf("%d blocks bit-identical to the golden", len(src)/17)
}

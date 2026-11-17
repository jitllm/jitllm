package quant

import (
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestQ51MatchesGGUFPy holds the Q5_1 decoder to an implementation that is not
// this one: the pip gguf package's Q5_1.dequantize_blocks, written out by
// scripts/q51gold.py over every quant in every slot at a sweep of d and m that
// reaches subnormal halves, the largest finite one and negative minimums, plus
// real falcon-7b attn_qkv blocks when the golden was built with a GGUF.
//
// Bits, not values. d*q is exact in f32 (a half times five bits) and the one
// rounding is the add of m, which both sides do once -- so anything but
// equality is a bug: the fifth bit read from the wrong qh bit, the minimum
// read from the scale's slot, or the minimum's sign dropped.
func TestQ51MatchesGGUFPy(t *testing.T) {
	raw, err := os.ReadFile("testdata/q5_1.json")
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
	if len(src)%24 != 0 || len(g.Bits) != len(src)/24*32 || len(g.Bits) == 0 {
		t.Fatalf("golden is %d bytes and %d values", len(src), len(g.Bits))
	}
	got32 := make([]float32, len(g.Bits))
	if err := Dequant32(Q5_1, src, got32); err != nil {
		t.Fatal(err)
	}
	got64 := make([]float64, len(g.Bits))
	if err := Dequant(Q5_1, src, got64); err != nil {
		t.Fatal(err)
	}
	nonzero, negative := 0, 0
	for i, want := range g.Bits {
		w := math.Float32frombits(want)
		if math.Float32bits(got32[i]) != want {
			t.Fatalf("element %d (block %d): got %#08x (%g), golden %#08x (%g)",
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
	if !Dequantable(Q5_1) {
		t.Fatal("Q5_1 decodes and Dequantable says otherwise")
	}
	t.Logf("%d blocks bit-identical to the golden (%s)", len(src)/24, g.Source)
}

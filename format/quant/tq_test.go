package quant

import (
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestTernaryMatchesGGUFPy holds the TQ1_0 and TQ2_0 decoders to an
// implementation that is not this one: the pip gguf package's
// dequantize_blocks, written out by scripts/tqgold.py over every byte value in
// every position at a sweep of d reaching subnormal halves, the largest finite
// one and negatives, each format's own quantizer output, and real blocks when
// the golden was built with a GGUF.
//
// Bits, not values: a code times an f16 is exact in f32, so anything but
// equality is a bug -- a digit read in the wrong order, TQ1_0's mod-256
// multiply done wide, or TQ2_0's element order transposed.
func TestTernaryMatchesGGUFPy(t *testing.T) {
	raw, err := os.ReadFile("testdata/tq.json")
	if err != nil {
		t.Fatal(err)
	}
	var g map[string]struct {
		Source string
		Blocks string
		Bits   []uint32
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	for _, ty := range []Type{TQ1_0, TQ2_0} {
		t.Run(ty.String(), func(t *testing.T) {
			e, ok := g[ty.String()]
			if !ok {
				t.Fatalf("no %s golden", ty)
			}
			src, err := hex.DecodeString(e.Blocks)
			if err != nil {
				t.Fatal(err)
			}
			bb := int(ty.BlockBytes())
			if len(src)%bb != 0 || len(e.Bits) != len(src)/bb*256 || len(e.Bits) == 0 {
				t.Fatalf("golden is %d bytes and %d values", len(src), len(e.Bits))
			}
			got32 := make([]float32, len(e.Bits))
			if err := Dequant32(ty, src, got32); err != nil {
				t.Fatal(err)
			}
			got64 := make([]float64, len(e.Bits))
			if err := Dequant(ty, src, got64); err != nil {
				t.Fatal(err)
			}
			var pos, neg, zero int
			for i, want := range e.Bits {
				w := math.Float32frombits(want)
				if math.Float32bits(got32[i]) != want {
					t.Fatalf("element %d (block %d): got %#08x (%g), gguf-py %#08x (%g)",
						i, i/256, math.Float32bits(got32[i]), got32[i], want, w)
				}
				if !math.IsInf(float64(w), 0) && float32(got64[i]) != w {
					t.Fatalf("element %d: f64 %v rounds to %v, want %v", i, got64[i], float32(got64[i]), w)
				}
				switch {
				case w > 0:
					pos++
				case w < 0:
					neg++
				default:
					zero++
				}
			}
			n := len(e.Bits)
			if pos < n/8 || neg < n/8 || zero < n/16 {
				t.Fatalf("%d positive, %d negative and %d zero of %d golden values -- the golden is degenerate",
					pos, neg, zero, n)
			}
			if !Dequantable(ty) {
				t.Fatalf("%s decodes and Dequantable says otherwise", ty)
			}
			t.Logf("%d blocks bit-identical to gguf-py (%s)", len(src)/bb, e.Source)
		})
	}
}

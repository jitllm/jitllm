package oracle

import (
	"math"
	"testing"
)

// TestFP8ByteValues holds the element decoders to values worked by hand from
// the formats' definitions: E4M3's largest normal, its smallest normal and
// subnormal, a sign; every E2M1 code; E8M0 around its bias.
func TestFP8ByteValues(t *testing.T) {
	e4 := map[byte]float32{
		0x00: 0, 0x80: 0, 0x38: 1, 0xB8: -1, 0x3C: 1.5, 0x40: 2, 0x7E: 448, 0xFE: -448,
		0x08: 0.015625, 0x01: 0.001953125, 0x07: 0.013671875, 0x30: 0.5, 0x5D: 26,
	}
	for b, want := range e4 {
		if got := E4M3(b); got != want {
			t.Errorf("E4M3(%#02x) = %v, want %v", b, got, want)
		}
	}
	if !math.IsNaN(float64(E4M3(0x7F))) || !math.IsNaN(float64(E4M3(0xFF))) {
		t.Error("E4M3's NaN pattern decodes to a number")
	}
	e2 := []float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}
	for n, want := range e2 {
		if got := E2M1(byte(n)); got != want {
			t.Errorf("E2M1(%d) = %v, want %v", n, got, want)
		}
	}
	for b, want := range map[byte]float32{127: 1, 128: 2, 126: 0.5, 113: 1.0 / 16384, 135: 256} {
		if got := E8M0(b); got != want {
			t.Errorf("E8M0(%d) = %v, want %v", b, got, want)
		}
	}
}

// TestDequantFP8Blocks decodes two rows of each format laid out by hand and
// compares every element, past the first block, with its worked value.
func TestDequantFP8Blocks(t *testing.T) {
	const k = 32
	codes := make([]byte, 2*k/2)
	scales := []byte{0x38, 0x40, 0x30, 0x7E}
	want := make([]float32, 2*k)
	for i := range 2 * k {
		n := byte(i*7+3) & 15
		codes[i/2] |= n << (4 * (i % 2))
		s := []float32{1, 2, 0.5, 448}[i/16]
		want[i] = []float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}[n] * s * 0.25
	}
	got := make([]float32, 2*k)
	DequantNVFP4(got, codes, scales, 0.25, 2, k)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("NVFP4 element %d: %v, want %v", i, got[i], want[i])
		}
	}
	// A nibble order swapped is the violation this gate exists for.
	if E2M1(codes[0]>>4) == E2M1(codes[0]&15) {
		t.Fatal("the first byte's nibbles decode alike: the order is not checked")
	}

	vals := make([]byte, 2*64)
	mx := []byte{127, 120, 130, 113}
	wantM := make([]float32, len(vals))
	for i := range vals {
		vals[i] = byte(i*37 + 5)
		if vals[i]&0x7F == 0x7F {
			vals[i] = 0x38
		}
		wantM[i] = E4M3(vals[i]) * float32(math.Ldexp(1, int(mx[i/32])-127))
	}
	gotM := make([]float32, len(vals))
	DequantMXFP8(gotM, vals, mx, 2, 64)
	for i := range wantM {
		if gotM[i] != wantM[i] {
			t.Fatalf("MXFP8 element %d: %v, want %v", i, gotM[i], wantM[i])
		}
	}
	if gotM[40] != E4M3(vals[40])/128 {
		t.Fatalf("MXFP8 element 40 takes 2^-7 from its block: %v", gotM[40])
	}
}

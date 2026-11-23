package jlm

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
)

// fp8Scale is a random E4M3 byte that is not the fn variant's NaN.
func fp8Scale(rng *rand.Rand) byte {
	b := byte(rng.Uint32())
	if b&0x7F == 0x7F {
		b ^= 1
	}
	return b
}

// TestDecodeNV4AndMX8MatchTheOracle holds DecodeVerbatim, evaluated in Go,
// to the oracle's dequantizers of the checkpoint's own planes, element by
// element: rows laid out by NV4Rows and MX8Rows from random codes, every
// scale byte but NaN (subnormal E4M3 and E8M0 0 among them), sheets of
// their own globals, and ragged widths -- one block, a padded row, and a
// checkpoint's 1152. Then the violations: one block's scale changed, or the
// codes read high nibble first, must move the decoded values.
func TestDecodeNV4AndMX8MatchTheOracle(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	const sheets, rows = 2, 3
	for _, k := range []int{16, 32, 48, 80, 1152} {
		codes := make([]byte, sheets*rows*k/2)
		for i := range codes {
			codes[i] = byte(rng.Uint32())
		}
		scales := make([]byte, sheets*rows*k/16)
		for i := range scales {
			scales[i] = fp8Scale(rng)
		}
		scales[0], scales[len(scales)-1] = 0x01, 0x7E
		global := []float32{0.33105251 / 2688, 1.0 / 3}
		raw, err := NV4Rows(codes, scales, global, sheets, rows, k)
		if err != nil {
			t.Fatal(err)
		}
		words, err := VerbatimWords(TypeNV4, sheets*rows, k)
		if err != nil || words*4 != len(raw) {
			t.Fatalf("nv4 k %d: %d words for %d bytes (%v)", k, words, len(raw), err)
		}
		want := make([]float32, sheets*rows*k)
		for s := range sheets {
			oracle.DequantNVFP4(want[s*rows*k:], codes[s*rows*k/2:], scales[s*rows*k/16:], global[s], rows, k)
		}
		dec := func(raw []byte, r, e int) float32 {
			return DecodeVerbatim[gv](planes{qs: raw}, TypeNV4, k, gv{u: uint32(r)}, gv{u: uint32(e)}).f
		}
		nonzero := 0
		for r := range sheets * rows {
			for e := range k {
				got := dec(raw, r, e)
				if got != want[r*k+e] {
					t.Fatalf("nv4 k %d: element %d of row %d decodes to %v, the oracle %v", k, e, r, got, want[r*k+e])
				}
				if got != 0 {
					nonzero++
				}
			}
		}
		if nonzero < len(want)/2 {
			t.Fatalf("nv4 k %d: %d of %d elements nonzero: the comparison is degenerate", k, nonzero, len(want))
		}

		// The violations, past the first row and block.
		r, e := sheets*rows-1, k-1
		mut := append([]byte(nil), raw...)
		mut[r*int(NV4RowBytes(k))+4+(e/16)*9] ^= 0x08
		if dec(mut, r, e) == want[r*k+e] && want[r*k+e] != 0 {
			t.Fatalf("nv4 k %d: a block's scale changed and its element did not", k)
		}
		swapped := append([]byte(nil), codes...)
		for i := range swapped {
			swapped[i] = swapped[i]<<4 | swapped[i]>>4
		}
		bad := make([]float32, len(want))
		for s := range sheets {
			oracle.DequantNVFP4(bad[s*rows*k:], swapped[s*rows*k/2:], scales[s*rows*k/16:], global[s], rows, k)
		}
		differ := 0
		for i := range bad {
			if bad[i] != want[i] {
				differ++
			}
		}
		if differ == 0 {
			t.Fatalf("nv4 k %d: the nibble order is not observed", k)
		}

		vals := make([]byte, rows*k)
		for i := range vals {
			vals[i] = fp8Scale(rng)
		}
		mx := make([]byte, rows*k/32)
		for i := range mx {
			mx[i] = byte(100 + rng.IntN(50))
		}
		mx[0] = 0
		raw, err = MX8Rows(vals, mx, rows, k/32*32)
		if k%32 != 0 {
			if err == nil {
				t.Fatalf("mx8 k %d: a row of part of a block was laid out", k)
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		wantM := make([]float32, rows*k)
		oracle.DequantMXFP8(wantM, vals, mx, rows, k)
		for r := range rows {
			for e := range k {
				got := DecodeVerbatim[gv](planes{qs: raw}, TypeMX8, k, gv{u: uint32(r)}, gv{u: uint32(e)}).f
				if got != wantM[r*k+e] && !(math.IsNaN(float64(got)) && math.IsNaN(float64(wantM[r*k+e]))) {
					t.Fatalf("mx8 k %d: element %d of row %d decodes to %v, the oracle %v", k, e, r, got, wantM[r*k+e])
				}
			}
		}
		r, e = rows-1, k-1
		mut = append([]byte(nil), raw...)
		mut[r*int(MX8RowBytes(k))+(e/32)*33] ^= 1
		if got := DecodeVerbatim[gv](planes{qs: mut}, TypeMX8, k, gv{u: uint32(r)}, gv{u: uint32(e)}).f; got == wantM[r*k+e] && got != 0 {
			t.Fatalf("mx8 k %d: a block's exponent changed and its element did not", k)
		}
	}
}

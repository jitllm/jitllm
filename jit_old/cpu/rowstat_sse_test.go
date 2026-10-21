//go:build amd64

package cpu

import (
	"fmt"
	"testing"
)

// TestRowStatsSSEMatchReference is TestRowStatsMatchReference on the SSE
// tier, and holds each kernel to the AVX2 one's bits where this host can run
// both: the passes and the fold are the same, with no FMA on either.
func TestRowStatsSSEMatchReference(t *testing.T) {
	const c, eps = 0.3, 1e-5
	em := EmittersFor(TierSSE)
	for _, n := range append(f2Widths(), 2048, 8192) {
		x, ref := rowStatInputs(n)
		for _, mode := range []RowMode{RowGauss, RowMag, RowStd} {
			name := fmt.Sprintf("rowstat%d_sse_%d", mode, n)
			b, err := em.RowStat(n, mode)
			code := f2SSEKernel(t, name, b, err)
			kc, r, avx := []float32{1 / float32(n), 0, c}, []float32(nil), EmitGaussTopK(n)
			switch mode {
			case RowMag:
				kc, r, avx = []float32{1 / float32(n), eps, 1}, ref, EmitMagMatch(n)
			case RowStd:
				kc, avx = rowStdKons(n, eps), EmitRowStd(n)
			}
			got := runRowStat(t, code, name, x, r, kc)
			code.Close()
			want := rowStatRef(mode, x, ref, c, eps)
			zero := true
			for _, v := range want {
				zero = zero && v == 0
			}
			if zero {
				for i, v := range got {
					if v != 0 {
						t.Errorf("%s: element %d is %v where the reference is exactly 0", name, i, v)
					}
				}
			} else if nmse := f2NMSE(t, name, got, want); !(nmse <= 1e-12) {
				t.Errorf("%s: NMSE %.3e against float64", name, nmse)
			}
			if tw := f2AVX2Twin(t, avx); tw != nil {
				twin := runRowStat(t, tw, name, x, r, kc)
				tw.Close()
				for i := range got {
					if got[i] != twin[i] {
						t.Errorf("%s: element %d is %v on SSE and %v on AVX2", name, i, got[i], twin[i])
						break
					}
				}
			}
		}
	}
}

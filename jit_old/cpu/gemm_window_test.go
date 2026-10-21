package cpu

import (
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// The arm64 k-quant GEMM must emit valid code at both activation windows, and
// the wide one must actually be shorter -- if it is not, the hoist did nothing.
func TestA64GEMMWindowShrinksTheFold(t *testing.T) {
	for _, q := range []quant.Type{quant.Q3_K, quant.Q4_K, quant.Q5_K, quant.Q6_K} {
		for _, tl := range [][2]int{{4, 1}, {2, 2}, {2, 1}} {
			narrow, err1 := emitGEMMA64K(q, tl[0], tl[1], 8, Q8Block)
			wide, err2 := emitGEMMA64K(q, tl[0], tl[1], 8, 256)
			if err1 != nil || err2 != nil {
				t.Logf("%v %dx%d: narrow=%v wide=%v", q, tl[0], tl[1], err1, err2)
				continue
			}
			if len(wide) >= len(narrow) {
				t.Errorf("%v %dx%d: wide window emitted %d bytes, narrow %d -- the hoist did nothing",
					q, tl[0], tl[1], len(wide), len(narrow))
			}
			t.Logf("%v %dx%d: %d -> %d bytes (%.0f%% smaller)", q, tl[0], tl[1],
				len(narrow), len(wide), 100*(1-float64(len(wide))/float64(len(narrow))))
		}
	}
}

// TestGEMMWideWindowMatchesReference executes the wide-window kernel against
// the float64 reference. Hoisting the activation scale out of the sub-block
// loop is correct only because at window 256 every sub-block shares one scale;
// emitting at 256 while packing at 32 (the violation) fails at NMSE ~1e-2.
func TestGEMMWideWindowMatchesReference(t *testing.T) {
	for _, wt := range []quant.Type{quant.Q3_K, quant.Q4_K, quant.Q5_K, quant.Q6_K} {
		for _, tile := range [][2]int{{1, 1}, {2, 1}, {4, 1}, {2, 2}, {3, 1}, {4, 2}, {6, 2}} {
			for _, k := range []int{256, 512, 2048} {
				t.Run(wt.String()+"/"+name(tile[0], tile[1], k), func(t *testing.T) {
					gemmVsReferenceWin(t, wt, tile[0], tile[1], k, 256)
				})
			}
		}
	}
}

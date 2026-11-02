//go:build amd64

package cpu

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// TestAVX2MatrixEmittersAreByteStable pins the amd64 quantised matrix family
// (emitgolden_matrix_test.go): the fused matvec at every prefetch, the token
// tile and the weight-stationary GEMM under both dot sequences (dotEmitter's
// VNNI and pre-VNNI forms, chosen by the DotKind argument), and the row-major
// GEMM, which is VPDPBUSD only and takes no DotKind.
func TestAVX2MatrixEmittersAreByteStable(t *testing.T) {
	m := &matrixCases{t: t}
	for _, dk := range []DotKind{DotVNNI, DotVEX} {
		for _, q := range quant.PackedTypes {
			p := fmt.Sprintf("dk=%d/%s", dk, q)
			for _, ah := range fusedAheads {
				c, err := EmitPackedMatVecFusedAhead(q, dk, ah)
				m.add(fmt.Sprintf("fused/%s/ahead=%d", p, ah), c, err)
			}
			for _, s := range matrixShapes(q, PackedFusedGroupOf(q)) {
				for tok := 1; tok <= PackedTiledTokens+1; tok++ {
					c, err := EmitPackedMatMulTiled(q, s[0], s[1], tok, dk)
					m.add(fmt.Sprintf("tiled/%s/k=%d/nrows=%d/tok=%d", p, s[0], s[1], tok), c, err)
				}
			}
			for _, s := range matrixShapes(q, 8) {
				for _, win := range []int{0, Q8Block, 256} {
					c, err := EmitPackedMatMulStationary(q, s[0], s[1], win, dk)
					m.add(fmt.Sprintf("stationary/%s/k=%d/nrows=%d/win=%d", p, s[0], s[1], win), c, err)
				}
			}
		}
	}
	for _, q := range quant.PackedTypes {
		gemmSweep(func(mr, nr, nb, win int) {
			c, err := EmitGEMMWindow(q, mr, nr, nb, win)
			m.add(fmt.Sprintf("gemm/%s/tile=%dx%d/nb=%d/win=%d", q, mr, nr, nb, win), c, err)
		})
	}
	m.check("avx2_matrix")
}

// BenchmarkMatrixEmission is the cold emission cost of the amd64 packed
// family: every format's fused kernel, the tiled kernel at a real shape and
// the stationary GEMM, once each per iteration, under the host's dot
// sequence. A measurement, not a gate (RULE 4).
func BenchmarkMatrixEmission(b *testing.B) {
	for b.Loop() {
		for _, q := range quant.PackedTypes {
			EmitPackedMatVecFusedAhead(q, DotVNNI, 2)
			EmitPackedMatMulTiled(q, 4096, 4096, 2, DotVNNI)
			EmitPackedMatMulStationary(q, 4096, 4096, 256, DotVEX)
			EmitGEMMWindow(q, 4, 2, 16, 256)
		}
	}
}

// BenchmarkNEONMatrixEmission is BenchmarkMatrixEmission's arm64 twin.
func BenchmarkNEONMatrixEmission(b *testing.B) {
	neonDot(true, func() {
		for b.Loop() {
			for _, q := range quant.PackedTypes {
				EmitA64PackedMatVecFused(q, PackedFusedGroupOf(q), 2)
				EmitA64PackedMatMulTiled(q, 4096, 4096, 2)
				EmitA64PackedMatMulStationary(q, 4096, 4096, 256)
				emitGEMMA64K(q, 4, 2, 16, 256)
			}
		}
	})
}

// TestSSEMatrixEmittersAreByteStable pins the SSE tier's side: its packed and
// fused kernels, and the refusals standing for the prefetch, tile and
// stationary forms (sse_refuse.go), so a port lands as a recorded change.
func TestSSEMatrixEmittersAreByteStable(t *testing.T) {
	m := &matrixCases{t: t}
	for _, q := range quant.PackedTypes {
		c, err := EmitPackedMatVecFusedSSE(q)
		m.add(fmt.Sprintf("fused/%s", q), c, err)
		for rows := 0; rows <= 68; rows += 4 {
			c, err := EmitPackedMatVecSSE(q, rows)
			m.add(fmt.Sprintf("matvec/%s/rows=%d", q, rows), c, err)
		}
		for _, ah := range fusedAheads {
			c, err := EmitPackedMatVecFusedAheadSSE(q, ah)
			m.add(fmt.Sprintf("fusedahead/%s/ahead=%d", q, ah), c, err)
		}
		c, err = EmitPackedMatMulTiledSSE(q, 2048, 512, 2)
		m.add(fmt.Sprintf("tiled/%s", q), c, err)
		c, err = EmitPackedGEMMSSE(q, 2048, 512, 256)
		m.add(fmt.Sprintf("stationary/%s", q), c, err)
	}
	m.check("sse_matrix")
}

package cpu

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// The host quantised matrix family under the emitter golden gate
// (emitgolden_test.go): the packed matvec, the token tile, the
// weight-stationary GEMM and the row-major GEMM, on every tier and dot
// sequence, at every format quant.PackedTypes lists.

// matrixCases collects one family's cases. A refusal is a case too: its
// recorded bytes are the error text, so an emitter that starts emitting where
// it refused, or refuses where it emitted, fails like a changed byte.
type matrixCases struct {
	t     *testing.T
	cases []goldenCase
	seen  map[string]bool
	n, nr int // emitted, refused
}

func (m *matrixCases) add(name string, code []byte, err error) {
	m.t.Helper()
	if m.seen == nil {
		m.seen = map[string]bool{}
	}
	if m.seen[name] {
		m.t.Fatalf("%s: case named twice", name)
	}
	m.seen[name] = true
	if err != nil {
		m.cases = append(m.cases, goldenCase{name, []byte("refused: " + err.Error())})
		m.nr++
		return
	}
	if len(code) == 0 {
		m.t.Fatalf("%s: emitted nothing", name)
	}
	m.cases = append(m.cases, goldenCase{name, code})
	m.n++
}

func (m *matrixCases) check(family string) {
	m.t.Helper()
	if m.n == 0 {
		m.t.Fatalf("%s: every case refused", family)
	}
	m.t.Logf("%s: %d kernels, %d refusals", family, m.n, m.nr)
	checkGolden(m.t, family, m.cases)
}

// packedUnit is the k granularity of a packed format: one super-block.
func packedUnit(t quant.Type) int {
	q, ok := kernels.QuantOf(t)
	if !ok {
		return 0
	}
	sub, _, _, _ := kernels.Layout(q)
	perSuper, _ := kernels.ScaleLayout(q)
	return sub * perSuper
}

// matrixShapes is the (k, nrows) set a baked-shape matrix emitter is swept
// at. k and nrows reach the code only as immediates (token strides), so the
// sweep crosses each list with the other's two ends rather than taking the
// full product: ragged multiples of the format's super-block and row group
// (1, 2, 3, 5 and 7 of them), the shapes the gates bake (k=256 and 2048,
// nrows=16, 64 and 512) and widths real models carry, plus a k that is not a
// whole super-block and an nrows that is not a whole group, which the
// emitters refuse.
func matrixShapes(t quant.Type, group int) [][2]int {
	u := packedUnit(t)
	ks := []int{u, 2 * u, 3 * u, 5 * u}
	for _, k := range []int{256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 8192, 11008, 14336, 18944} {
		if k%u == 0 {
			ks = append(ks, k)
		}
	}
	rs := []int{group, 2 * group, 3 * group, 7 * group}
	for _, r := range []int{16, 64, 128, 512, 896, 2048, 4096, 4864, 11008, 14336} {
		if r%group == 0 {
			rs = append(rs, r)
		}
	}
	var out [][2]int
	seen := map[[2]int]bool{}
	put := func(k, r int) {
		if !seen[[2]int{k, r}] {
			seen[[2]int{k, r}] = true
			out = append(out, [2]int{k, r})
		}
	}
	for _, k := range ks {
		put(k, group)
		put(k, 4096)
	}
	for _, r := range rs {
		put(u, r)
		put(4096, r)
	}
	put(u/2, group)
	put(u, group+group/2)
	return out
}

// fusedAheads is engine/nn/prefetch.go's prefetchDistances (the duel's
// candidates, FusedAheadWords first) and the neighbours of its rungs.
var fusedAheads = []int{FusedAheadWords, 0, 1, 2, 3, 4, 8, 16}

// gemmTiles is every register tile the row-major GEMM accepts (mr <= 8,
// mr*nr <= 12; engine/nn/tune.go's gemmCandidates are among them) and one
// past each bound.
func gemmTiles() [][2]int {
	var out [][2]int
	for mr := 1; mr <= 9; mr++ {
		for nr := 1; mr*nr <= 13; nr++ {
			out = append(out, [2]int{mr, nr})
		}
	}
	return out
}

// gemmNBs is the super-block counts a row-major GEMM is swept at. nb reaches
// the code as the row stride's immediate: every tile runs at the first two,
// engine/nn/tune.go's candidate tiles at all of them.
var gemmNBs = []int{1, 8, 2, 3, 16, 56, 64}

func gemmCandidateTile(tl [2]int) bool {
	for _, c := range [][2]int{{4, 2}, {6, 2}, {2, 2}, {3, 4}, {2, 4}, {8, 1}, {4, 1}, {1, 1}} {
		if c == tl {
			return true
		}
	}
	return false
}

// gemmSweep calls f at every (tile, nb, window) the row-major GEMM is pinned at.
func gemmSweep(f func(mr, nr, nb, win int)) {
	for _, tl := range gemmTiles() {
		for i, nb := range gemmNBs {
			if i >= 2 && !gemmCandidateTile(tl) {
				continue
			}
			for _, win := range []int{Q8Block, 256} {
				f(tl[0], tl[1], nb, win)
			}
		}
	}
}

// neonDot runs f with the A64 emitters encoding SDOT (dot) or widening it to
// the ARMv8.0 sequence (!dot), whatever the host.
func neonDot(dot bool, f func()) {
	oldEnc, oldNo := forceDotEncoding, forceNoDotProd
	forceDotEncoding, forceNoDotProd = dot, !dot
	defer func() { forceDotEncoding, forceNoDotProd = oldEnc, oldNo }()
	f()
}

// TestNEONMatrixEmittersAreByteStable pins the arm64 family: the packed
// matvec through all three of emitA64Packed's entry points, the token tile,
// the weight-stationary GEMM and the row-major k-quant GEMM, each with and
// without FEAT_DotProd.
func TestNEONMatrixEmittersAreByteStable(t *testing.T) {
	m := &matrixCases{t: t}
	for _, dot := range []bool{true, false} {
		neonDot(dot, func() {
			for _, q := range quant.PackedTypes {
				p := fmt.Sprintf("dot=%v/%s", dot, q)
				for rows := 0; rows <= 68; rows += 4 {
					for _, pf := range []int{-1, 0, 64, 256} {
						c, err := EmitA64PackedMatVecPF(q, rows, pf)
						m.add(fmt.Sprintf("matvec/%s/rows=%d/pf=%d", p, rows, pf), c, err)
					}
					for _, ah := range fusedAheads {
						c, err := EmitA64PackedMatVecFused(q, rows, ah)
						m.add(fmt.Sprintf("fused/%s/rows=%d/ahead=%d", p, rows, ah), c, err)
					}
					for _, win := range []int{Q8Block, 256} {
						for _, ah := range []int{0, 2, FusedAheadWords} {
							c, err := EmitA64PackedMatVecFusedWin(q, rows, ah, win)
							m.add(fmt.Sprintf("fusedwin/%s/rows=%d/ahead=%d/win=%d", p, rows, ah, win), c, err)
						}
					}
				}
				for _, s := range matrixShapes(q, PackedFusedGroupOf(q)) {
					for tok := 0; tok <= PackedTiledTokens+1; tok++ {
						c, err := EmitA64PackedMatMulTiled(q, s[0], s[1], tok)
						m.add(fmt.Sprintf("tiled/%s/k=%d/nrows=%d/tok=%d", p, s[0], s[1], tok), c, err)
					}
				}
				for _, s := range matrixShapes(q, 8) {
					for _, win := range []int{0, Q8Block, 256} {
						c, err := EmitA64PackedMatMulStationary(q, s[0], s[1], win)
						m.add(fmt.Sprintf("stationary/%s/k=%d/nrows=%d/win=%d", p, s[0], s[1], win), c, err)
					}
				}
				gemmSweep(func(mr, nr, nb, win int) {
					c, err := emitGEMMA64K(q, mr, nr, nb, win)
					m.add(fmt.Sprintf("gemmk/%s/tile=%dx%d/nb=%d/win=%d", p, mr, nr, nb, win), c, err)
				})
			}
		})
	}
	m.check("neon_matrix")
}

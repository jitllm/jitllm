package paritytest

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// fused is program p lowered and through the fusion pass, its one group.
func fused(t *testing.T, p *compose.Program) *compose.Row {
	t.Helper()
	gs, err := p.Lower()
	if err != nil {
		t.Fatal(err)
	}
	var f compose.FuseParams
	f.Allow(compose.PrimAdd, compose.PrimNorm)
	f.Allow(compose.PrimNorm, compose.PrimQuant)
	f.Allow(compose.PrimActMul, compose.PrimQuant)
	gs = compose.Fuse(p, gs, f)
	if len(gs) != 1 {
		t.Fatalf("%d groups after the fusion pass, want 1", len(gs))
	}
	return &gs[0].Row
}

// TestFusedNormsAreMains holds the fusion pass over a residual add, a norm
// and a quantize, each lowered alone, to main's fused row kernels.
func TestFusedNormsAreMains(t *testing.T) {
	const eps = 1e-6
	for _, c := range []struct {
		w, rows    int
		add, quant bool
	}{{256, 8, true, true}, {256, 8, false, true}, {256, 8, true, false}, {1024, 1, true, true}, {1024, 1, false, true}} {
		name := fmt.Sprintf("w%d_r%d_add%v_q%v", c.w, c.rows, c.add, c.quant)
		t.Run(name, func(t *testing.T) {
			var mk *ir.Kernel
			var err error
			if c.quant {
				mk, err = mainkernels.RMSNormQuantRowsWarp(c.w, c.rows, eps, false, c.add, 32)
			} else {
				mk, err = mainkernels.RMSNormRowsWarp(c.w, c.rows, eps, false, c.add)
			}
			if err != nil {
				t.Fatal(err)
			}
			p := &compose.Program{Rows: c.rows}
			val := func(e compose.ValElem) int {
				p.Vals = append(p.Vals, compose.Value{Elem: e, Width: c.w, Buf: -1})
				return len(p.Vals) - 1
			}
			x := val(compose.ValF32)
			if c.add {
				r, s := val(compose.ValF32), val(compose.ValF32)
				p.Prims = append(p.Prims, compose.Prim{Kind: compose.PrimAdd, In: []int{x, r}, Out: []int{s}})
				x = s
			}
			y := val(compose.ValF32)
			p.Prims = append(p.Prims, compose.Prim{Kind: compose.PrimNorm, In: []int{x}, Out: []int{y}, Eps: eps})
			if c.quant {
				p.Prims = append(p.Prims, compose.Prim{Kind: compose.PrimQuant, In: []int{y}, Out: []int{val(compose.ValQ8)}})
			}
			same(t, name, []*ir.Kernel{mk}, planned(t, fused(t, p), compose.Facts{Target: sm70}))
		})
	}
}

// TestFusedActQuantIsMains holds the gated activation and its quantize,
// each lowered alone, through the fusion pass to main's QuantizeAct.
func TestFusedActQuantIsMains(t *testing.T) {
	for _, n := range []int{1024, 8 * 2048} {
		name := fmt.Sprint(n)
		t.Run(name, func(t *testing.T) {
			mk, err := mainkernels.QuantizeAct(n, 32, mainkernels.ActSiLU)
			if err != nil {
				t.Fatal(err)
			}
			p := &compose.Program{Rows: 1, Vals: []compose.Value{{Width: n, Buf: -1}, {Width: n, Buf: -1}, {Width: n, Buf: -1}, {Elem: compose.ValQ8, Width: n, Buf: -1}}}
			p.Prims = []compose.Prim{{Kind: compose.PrimActMul, Act: kernels.ActSiLU, In: []int{0, 1}, Out: []int{2}},
				{Kind: compose.PrimQuant, In: []int{2}, Out: []int{3}}}
			same(t, name, []*ir.Kernel{mk}, planned(t, fused(t, p), compose.Facts{Target: sm70}))
		})
	}
}

// TestFusedMatVecEpiloguesAreMains holds a matvec and the add or gated
// activation reading it, each lowered alone, through the fusion and
// dead-store passes to main's matvec with the residual or gate epilogue.
func TestFusedMatVecEpiloguesAreMains(t *testing.T) {
	for _, q := range []format{{kernels.Q8_0, mainkernels.Q8_0}, {kernels.Q4_K, mainkernels.Q4_K}, {kernels.Float32, mainkernels.Float32}} {
		for _, sp := range []int{1, 2, 4} {
			for _, epi := range []int{1, 2} {
				const rows, k = 256, 512
				name := fmt.Sprintf("%v_s%d_epi%d", q.c, sp, epi)
				t.Run(name, func(t *testing.T) {
					if epi == 2 && sp == 2 {
						t.Skip("the gate rides a final row")
					}
					mk, err := mainkernels.MatVec(mainkernels.MatVecShape{T: q.m, K: k, Rows: rows, NTok: 1,
						Split: sp, GroupSplit: sp == 4, Bias: epi == 1, Gate: epi == 2, GateAct: mainkernels.ActSiLU})
					if err != nil {
						t.Fatal(err)
					}
					p := &compose.Program{Rows: 1, Vals: []compose.Value{{Elem: compose.ValQ8, Width: k, Buf: -1},
						{Width: rows, Buf: -1}, {Width: rows, Buf: -1}, {Width: rows, Buf: -1}, {Width: k, Buf: -1}}}
					f := compose.Facts{Target: sm70, Step: compose.StepDecode, DecodeSplit: sp, DecodeGroup: sp == 4}
					mv := compose.Prim{Kind: compose.PrimMatVec, In: []int{0, 4}, Float: 1, Out: []int{1}, Facts: f,
						W: compose.MatVecWeight{T: q.c, K: k, Rows: rows}}
					ep := compose.Prim{Kind: compose.PrimAdd, In: []int{1, 2}, Out: []int{3}}
					if epi == 2 {
						ep = compose.Prim{Kind: compose.PrimActMul, Act: kernels.ActSiLU, In: []int{2, 1}, Out: []int{3}}
					}
					p.Prims = []compose.Prim{mv, ep}
					gs, err := p.Lower()
					if err != nil {
						t.Fatal(err)
					}
					var fp compose.FuseParams
					fp.Allow(compose.PrimMatVec, compose.PrimAdd)
					fp.Allow(compose.PrimMatVec, compose.PrimActMul)
					gs = compose.Fuse(p, gs, fp)
					var pp compose.PruneParams
					pp.Drop[compose.PrimMatVec] = true
					gs = compose.Prune(p, gs, pp)
					if len(gs) != 1 {
						t.Fatalf("%d groups", len(gs))
					}
					ks := []*ir.Kernel{mk}
					if sp == 2 {
						r, err := mainkernels.Reduce(rows, sp)
						if err != nil {
							t.Fatal(err)
						}
						ks = append(ks, r)
					}
					same(t, name, ks, plannedAll(t, &gs[0].Row, gs[0].Facts))
				})
			}
		}
	}
}

// TestMatVecPrimIsMains holds the matvec prim alone, a float weight's
// over its float row and a quantized one's over its int8 operand, to
// main's decode matvec at its split.
func TestMatVecPrimIsMains(t *testing.T) {
	for _, q := range []format{{kernels.Float32, mainkernels.Float32}, {kernels.Float16, mainkernels.Float16}, {kernels.Q8_0, mainkernels.Q8_0}} {
		for _, sp := range []int{1, 2} {
			name := fmt.Sprintf("%v_s%d", q.c, sp)
			t.Run(name, func(t *testing.T) {
				const rows, k = 64, 64
				mk, err := mainkernels.MatVec(mainkernels.MatVecShape{T: q.m, K: k, Rows: rows, NTok: 1, Split: sp})
				if err != nil {
					t.Fatal(err)
				}
				p := &compose.Program{Rows: 1, Vals: []compose.Value{{Elem: compose.ValQ8, Width: k, Buf: -1}, {Width: rows, Buf: -1}, {Width: k, Buf: -1}}}
				mv := compose.Prim{Kind: compose.PrimMatVec, In: []int{0, 2}, Out: []int{1}, Float: 1,
					Facts: compose.Facts{Target: sm70, Step: compose.StepDecode, DecodeSplit: sp}, W: compose.MatVecWeight{T: q.c, K: k, Rows: rows}}
				p.Prims = []compose.Prim{mv}
				gs, err := p.Lower()
				if err != nil {
					t.Fatal(err)
				}
				ks := []*ir.Kernel{mk}
				if sp > 1 {
					r, err := mainkernels.Reduce(rows, sp)
					if err != nil {
						t.Fatal(err)
					}
					ks = append(ks, r)
				}
				same(t, name, ks, plannedAll(t, &gs[0].Row, gs[0].Facts))
			})
		}
	}
}

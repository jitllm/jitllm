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
	gs = compose.Fuse(gs)
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
				p.Vals = append(p.Vals, compose.Value{Elem: e, Width: c.w})
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
			p := &compose.Program{Rows: 1, Vals: []compose.Value{{Width: n}, {Width: n}, {Width: n}, {Elem: compose.ValQ8, Width: n}}}
			p.Prims = []compose.Prim{{Kind: compose.PrimActMul, Act: kernels.ActSiLU, In: []int{0, 1}, Out: []int{2}},
				{Kind: compose.PrimQuant, In: []int{2}, Out: []int{3}}}
			same(t, name, []*ir.Kernel{mk}, planned(t, fused(t, p), compose.Facts{Target: sm70}))
		})
	}
}

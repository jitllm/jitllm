package paritytest

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestTwoPassNormIsMains holds the composer's two-pass RMSNorm -- the sums
// of squares in parts, then the apply -- to main's NormPartRows and
// NormApplyRows, at one row and over rows, with and without the weight's
// added one.
func TestTwoPassNormIsMains(t *testing.T) {
	for _, c := range []struct{ k, parts, rows int }{{64, 8, 1}, {2048, 64, 1}, {256, 16, 32}, {4096, 64, 4}} {
		for _, addOne := range []bool{false, true} {
			name := fmt.Sprintf("k%d_p%d_r%d_one%v", c.k, c.parts, c.rows, addOne)
			t.Run(name, func(t *testing.T) {
				const eps = 1e-6
				mp, err := mainkernels.NormPartRows(c.k, c.parts, c.rows)
				if err != nil {
					t.Fatal(err)
				}
				ma, err := mainkernels.NormApplyRows(c.k, c.parts, eps, addOne, c.rows)
				if err != nil {
					t.Fatal(err)
				}
				gp := &compose.Row{Width: c.parts * c.rows, Rows: 1}
				x, out := gp.Buf(compose.BufRow), gp.Buf(compose.BufOut)
				gp.Store(out, gp.SquareSumParts(x, c.k, c.parts))
				same(t, name+"/part", []*ir.Kernel{mp}, emitAt(t, gp, compose.Schedule{Group: 128, Flat: true, Target: sm70}))
				ga := &compose.Row{Width: c.k, Rows: c.rows}
				ax, w, part := ga.Buf(compose.BufRow), ga.Buf(compose.BufVec), ga.Buf(compose.BufRow)
				ga.Store(ga.Buf(compose.BufOut), ga.NormApplyParts(ax, w, part, c.k, c.parts, eps, addOne))
				same(t, name+"/apply", []*ir.Kernel{ma}, emitAt(t, ga, compose.Schedule{Group: 128, Flat: true, Target: sm70}))
			})
		}
	}
}

// TestArgmaxIsMains holds the composer's argmax to main's Argmax and
// ArgmaxRows: a vocabulary at one row and over rows.
func TestArgmaxIsMains(t *testing.T) {
	for _, c := range []struct{ n, rows int }{{1000, 1}, {32000, 1}, {151936, 1}, {4096, 4}, {32000, 8}} {
		name := fmt.Sprintf("n%d_r%d", c.n, c.rows)
		t.Run(name, func(t *testing.T) {
			var mk *ir.Kernel
			var err error
			if c.rows == 1 {
				mk, err = mainkernels.Argmax(c.n)
			} else {
				mk, err = mainkernels.ArgmaxRows(c.n, c.rows)
			}
			if err != nil {
				t.Fatal(err)
			}
			g := &compose.Row{Width: c.n, Rows: c.rows}
			g.Argmax(g.Load(g.Buf(compose.BufRow)), g.Buf(compose.BufOutU32))
			ps, err := g.Plan(compose.Facts{Target: sm70})
			if err != nil {
				t.Fatal(err)
			}
			same(t, name, []*ir.Kernel{mk}, emitAt(t, g, ps[0].S))
		})
	}
}

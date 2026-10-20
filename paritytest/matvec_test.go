package paritytest

import (
	"flag"
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

var sm70 = compose.Target{API: compose.APIPTX, ISA: ir.PTX(70), Lanes: ir.SubgroupLanes}

// formed is every kernel g forms under schedule s.
func formed(t *testing.T, g *compose.Row, s compose.Schedule) []*ir.Kernel {
	t.Helper()
	ps, err := g.PlanAt(s)
	if err != nil {
		t.Fatal(err)
	}
	var ks []*ir.Kernel
	for _, p := range ps {
		k, err := p.G.Emit(p.S)
		if err != nil {
			t.Fatal(err)
		}
		ks = append(ks, k)
	}
	return ks
}

var explain = flag.Bool("explain", false, "write out every missing instruction of main's")

// same holds the composed kernels to main's instruction by instruction.
func same(t *testing.T, name string, main, composed []*ir.Kernel) {
	t.Helper()
	total, n, miss := Compare(main, composed)
	if n == total {
		return
	}
	in := NewInterner()
	if *explain {
		for _, m := range miss {
			t.Logf("missing %%%d %s", m.Op, Explain(in, m.K, m.Op, 4))
		}
		for _, k := range composed {
			for i := range k.Ops {
				t.Logf("composed %%%d %s", i, Explain(in, k, i, 4))
			}
		}
		for _, k := range main {
			for i, o := range k.Ops {
				t.Logf("main raw %%%d %v %v args %v imm %d", i, o.Kind, o.Type, o.Args, o.Imm)
			}
		}
	}
	first := ""
	if len(miss) > 0 {
		first = Explain(in, miss[0].K, miss[0].Op, 3)
	}
	t.Errorf("%s: %d of %d of main's instructions (%.1f%%); first missing %%%d %s",
		name, n, total, 100*float64(n)/float64(total), miss[0].Op, first)
}

// format is one weight format in both vocabularies: the composer's and
// main's own, named apart so a renumbering cannot pair two formats.
type format struct {
	c kernels.Quant
	m mainkernels.Quant
}

type mvCase struct {
	t                        format
	k, rows, ntok, tok, rowt int
	split                    int
	bias                     bool
}

func (c mvCase) String() string {
	return fmt.Sprintf("%v_%dx%d_t%d_tok%d_rowt%d_s%d_b%v", c.t.c, c.rows, c.k, c.ntok, c.tok, c.rowt, c.split, c.bias)
}

// mainMatVec is main's matvec at c and, under a split, its reduce.
func mainMatVec(t *testing.T, c mvCase) []*ir.Kernel {
	t.Helper()
	s := mainkernels.MatVecShape{T: c.t.m, K: c.k, Rows: c.rows, Split: c.split, Bias: c.bias, Rowt: c.rowt}
	if c.ntok > 1 {
		s.NTok, s.Tok = c.ntok, c.tok
	}
	k, err := mainkernels.MatVec(s)
	if err != nil {
		t.Fatal(err)
	}
	ks := []*ir.Kernel{k}
	if c.split > 1 {
		r, err := mainkernels.Reduce(c.rows*max(c.ntok, 1), c.split)
		if err != nil {
			t.Fatal(err)
		}
		ks = append(ks, r)
	}
	return ks
}

// composedMatVec is the composer's group for c, as the tier builds it, and
// the schedule c pins.
func composedMatVec(t *testing.T, c mvCase) []*ir.Kernel {
	t.Helper()
	ntok := max(c.ntok, 1)
	g := &compose.Row{Width: c.rows, Rows: ntok}
	w := compose.MatVecWeight{T: c.t.c, K: c.k, Rows: c.rows, QS: g.Buf(compose.BufU32), D: -1, SC: -1}
	if !kernels.IsFloat(c.t.c) {
		w.D = g.Buf(compose.BufU32)
	}
	a := compose.MatVecOperand{Toks: ntok}
	if kernels.IsFloat(c.t.c) {
		a.X = g.Buf(compose.BufRow)
	} else {
		a.A, a.AX = g.Buf(compose.BufU32), g.Buf(compose.BufRow)
	}
	v, err := g.MatVec(w, a)
	if err != nil {
		t.Fatal(err)
	}
	if c.bias {
		v = g.Bin(compose.OpAdd, v, g.Load(g.Buf(compose.BufVec)))
	}
	g.Store(g.Buf(compose.BufOut), v)
	s := compose.Schedule{Group: 128, Flat: true, Target: sm70, TileCols: max(c.rowt, 1), Split: max(c.split, 1)}
	if c.ntok > 1 {
		s.Tile = c.tok
	}
	return formed(t, g, s)
}

// TestMatVecIsMains holds the composer's matvec to main's at the same
// format, shape and schedule: the schedule main's shape states, so what
// differs is the former's, not the plan's.
func TestMatVecIsMains(t *testing.T) {
	var cases []mvCase
	for _, q := range []format{{kernels.Float32, mainkernels.Float32}, {kernels.Q8_0, mainkernels.Q8_0}} {
		for _, rk := range [][2]int{{64, 64}, {96, 64}, {64, 256}} {
			for _, sp := range []int{1, 2} {
				for _, rowt := range []int{1, 2} {
					cases = append(cases, mvCase{t: q, rows: rk[0], k: rk[1], ntok: 1, rowt: rowt, split: sp})
				}
				for _, tr := range [][2]int{{8, 2}, {16, 4}, {8, 1}} {
					cases = append(cases, mvCase{t: q, rows: rk[0], k: rk[1], ntok: 32, tok: tr[0], rowt: tr[1], split: sp})
				}
			}
		}
	}
	for _, c := range cases {
		t.Run(c.String(), func(t *testing.T) {
			same(t, c.String(), mainMatVec(t, c), composedMatVec(t, c))
		})
	}
}

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

// plannedAll is every kernel g's plan forms for facts f.
func plannedAll(t *testing.T, g *compose.Row, f compose.Facts) []*ir.Kernel {
	t.Helper()
	ps, err := g.Plan(f)
	if err != nil {
		t.Fatal(err)
	}
	var ks []*ir.Kernel
	for _, p := range ps {
		ks = append(ks, emitPlanned(t, p)...)
	}
	return ks
}

var explain = flag.Bool("explain", false, "write out every missing instruction of main's")

// same holds the composed kernels to main's instruction by instruction.
func same(t *testing.T, name string, main, composed []*ir.Kernel) {
	t.Helper()
	total, n, miss := Compare(main, composed)
	ctotal, cn, extra := Compare(composed, main)
	in := NewInterner()
	if *explain {
		for _, m := range extra {
			t.Logf("extra %%%d %s", m.Op, Explain(in, m.K, m.Op, 4))
		}
		if len(extra) > 0 && n == total {
			for _, k := range composed {
				for i := range k.Ops {
					t.Logf("composed %%%d %s", i, Explain(in, k, i, 4))
				}
			}
		}
	}
	if n == total && cn != ctotal {
		t.Errorf("%s: every value of main's, and %d of the composer's %d more: first %%%d %s", name, ctotal-cn, ctotal,
			extra[0].Op, Explain(in, extra[0].K, extra[0].Op, 3))
	}
	if n == total {
		return
	}
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
	f := compose.Facts{Target: sm70, Step: compose.StepDecode, DecodeSplit: max(c.split, 1), DecodeRowt: max(c.rowt, 1)}
	if c.ntok > 1 {
		f = compose.Facts{Target: sm70, Step: compose.StepChunk, Tok: c.tok, RowTile: c.rowt, Split: max(c.split, 1), NoMMA: true}
	}
	return plannedAll(t, g, f)
}

// TestMatVecIsMains holds the composer's matvec to main's at the same
// format, shape and schedule: the schedule main's shape states, so what
// differs is the former's, not the plan's.
func TestMatVecIsMains(t *testing.T) {
	var cases []mvCase
	for _, q := range []format{{kernels.Float32, mainkernels.Float32}, {kernels.Float16, mainkernels.Float16}, {kernels.BFloat16, mainkernels.BFloat16}, {kernels.Q8_0, mainkernels.Q8_0}} {
		for _, rk := range [][2]int{{64, 64}, {96, 64}, {64, 256}} {
			for _, sp := range []int{1, 2} {
				for _, rowt := range []int{1, 2} {
					cases = append(cases, mvCase{t: q, rows: rk[0], k: rk[1], ntok: 1, rowt: rowt, split: sp})
				}
				for _, tr := range [][2]int{{8, 2}, {16, 4}, {16, 2}, {4, 2}} {
					cases = append(cases, mvCase{t: q, rows: rk[0], k: rk[1], ntok: 32, tok: tr[0], rowt: tr[1], split: sp})
					cases = append(cases, mvCase{t: q, rows: rk[0], k: rk[1], ntok: 32, tok: tr[0], rowt: tr[1], split: sp, bias: true})
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

// TestExpertMatVecIsMains holds the composer's matvec over a bank's
// selected sheets to main's indexed matvec: the slots as the group's rows,
// each reading its expert's sheet, the activation shared (gate and up) or a
// slot's own (down), split or not.
func TestExpertMatVecIsMains(t *testing.T) {
	for _, q := range []format{{kernels.Float32, mainkernels.Float32}, {kernels.Float16, mainkernels.Float16}, {kernels.Q8_0, mainkernels.Q8_0}} {
		for _, rk := range [][2]int{{64, 64}, {96, 128}} {
			for _, sp := range []int{1, 2} {
				for _, own := range []bool{false, true} {
					for _, ek := range [][2]int{{8, 2}, {16, 4}} {
						name := fmt.Sprintf("%v_%dx%d_e%dx%d_s%d_own%v", q.c, rk[0], rk[1], ek[0], ek[1], sp, own)
						t.Run(name, func(t *testing.T) {
							s := mainkernels.MatVecShape{T: q.m, K: rk[1], Rows: rk[0], Split: sp, Experts: ek[0], Slots: ek[1], SlotAct: own}
							mk, err := mainkernels.MatVec(s)
							if err != nil {
								t.Fatal(err)
							}
							ks := []*ir.Kernel{mk}
							if sp > 1 {
								r, err := mainkernels.Reduce(rk[0]*ek[1], sp)
								if err != nil {
									t.Fatal(err)
								}
								ks = append(ks, r)
							}
							g := &compose.Row{Width: rk[0], Rows: ek[1]}
							w := compose.MatVecWeight{T: q.c, K: rk[1], Rows: rk[0], QS: g.Buf(compose.BufU32), D: -1, SC: -1, Experts: ek[0]}
							if !kernels.IsFloat(q.c) {
								w.D = g.Buf(compose.BufU32)
							}
							a := compose.MatVecOperand{Toks: ek[1], One: !own}
							if !own {
								a.Toks = 1
							}
							if kernels.IsFloat(q.c) {
								a.X = g.Buf(compose.BufRow)
							} else {
								a.A, a.AX = g.Buf(compose.BufU32), g.Buf(compose.BufRow)
							}
							w.Sel = g.Buf(compose.BufU32)
							v, err := g.MatVec(w, a)
							if err != nil {
								t.Fatal(err)
							}
							g.Store(g.Buf(compose.BufOut), v)
							same(t, name, ks, plannedAll(t, g, compose.Facts{Target: sm70, Step: compose.StepDecode, DecodeSplit: sp}))
						})
					}
				}
			}
		}
	}
}

// TestGroupedMatVecIsMains holds the composer's matvec over a mixture's
// sorted columns -- a token group of four a thread, each group reading the
// sheet its tile's expert names -- to main's grouped matvec.
func TestGroupedMatVecIsMains(t *testing.T) {
	for _, q := range []format{{kernels.Float32, mainkernels.Float32}, {kernels.Float16, mainkernels.Float16}, {kernels.Q8_0, mainkernels.Q8_0}} {
		for _, rk := range [][2]int{{64, 64}, {96, 128}} {
			for _, ep := range [][2]int{{8, 64}, {16, 128}} {
				name := fmt.Sprintf("%v_%dx%d_e%d_np%d", q.c, rk[0], rk[1], ep[0], ep[1])
				t.Run(name, func(t *testing.T) {
					mk, err := mainkernels.MatVec(mainkernels.MatVecShape{T: q.m, K: rk[1], Rows: rk[0], Experts: ep[0], NTok: ep[1], Tok: 4, ActWin: 32})
					if err != nil {
						t.Fatal(err)
					}
					g := &compose.Row{Width: rk[0], Rows: ep[1]}
					w := compose.MatVecWeight{T: q.c, K: rk[1], Rows: rk[0], QS: g.Buf(compose.BufU32), D: -1, SC: -1, Experts: ep[0], SelPer: 4}
					if !kernels.IsFloat(q.c) {
						w.D = g.Buf(compose.BufU32)
					}
					a := compose.MatVecOperand{Toks: ep[1]}
					if kernels.IsFloat(q.c) {
						a.X = g.Buf(compose.BufRow)
					} else {
						a.A, a.AX = g.Buf(compose.BufU32), g.Buf(compose.BufRow)
					}
					w.Sel = g.Buf(compose.BufU32)
					v, err := g.MatVec(w, a)
					if err != nil {
						t.Fatal(err)
					}
					g.Store(g.Buf(compose.BufOut), v)
					same(t, name, []*ir.Kernel{mk}, plannedAll(t, g, compose.Facts{Target: sm70, Step: compose.StepChunk, NoMMA: true}))
				})
			}
		}
	}
}

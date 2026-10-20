package paritytest

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// routes are the router shapes the route tests sweep: a plain softmax
// renormalised and not, a sigmoid with a shared bias, grouped, a softmax
// with a bias on the probability, a sqrt-softplus with a bias a row; one row
// and four.
func routes() []mainkernels.MoERoute {
	var rs []mainkernels.MoERoute
	for _, rows := range []int{1, 4, 8} {
		rs = append(rs,
			mainkernels.MoERoute{NExpert: 8, K: 2, Norm: true, Rows: rows},
			mainkernels.MoERoute{NExpert: 4, K: 2, Norm: true, Rows: rows},
			mainkernels.MoERoute{NExpert: 8, K: 2, Sigmoid: true, Bias: true, Norm: true, Rows: rows},
			mainkernels.MoERoute{NExpert: 8, K: 2, Sigmoid: true, Bias: true, Norm: true, Scale: 2.5, Rows: rows},
			mainkernels.MoERoute{NExpert: 8, K: 2, Sigmoid: true, Bias: true, NGroup: 2, NGroupUsed: 1, Norm: true, Rows: rows},
			mainkernels.MoERoute{NExpert: 64, K: 6, Norm: true, Scale: 2.5, Rows: rows},
			mainkernels.MoERoute{NExpert: 60, K: 4, Rows: rows},
			mainkernels.MoERoute{NExpert: 64, K: 8, Sigmoid: true, Bias: true, Norm: true, Scale: 2.5, Rows: rows},
			mainkernels.MoERoute{NExpert: 64, K: 8, Sigmoid: true, Bias: true, NGroup: 8, NGroupUsed: 4, Norm: true, Rows: rows},
			mainkernels.MoERoute{NExpert: 64, K: 8, Bias: true, Norm: true, Rows: rows},
			mainkernels.MoERoute{NExpert: 32, K: 4, SqrtSoftplus: true, Bias: true, RowBias: true, Norm: true, Rows: rows},
		)
	}
	return rs
}

func routeOf(m mainkernels.MoERoute) compose.Route {
	return compose.Route{NExpert: m.NExpert, K: m.K, Sigmoid: m.Sigmoid, SqrtSoftplus: m.SqrtSoftplus, Bias: m.Bias,
		RowBias: m.RowBias, NGroup: m.NGroup, Norm: m.Norm, Scale: m.Scale, Rows: m.Rows}
}

func routeName(m mainkernels.MoERoute) string {
	return fmt.Sprintf("e%d_k%d_sig%v_ssp%v_b%v_rb%v_g%d_n%v_s%v_r%d", m.NExpert, m.K, m.Sigmoid, m.SqrtSoftplus,
		m.Bias, m.RowBias, m.NGroup, m.Norm, m.Scale, m.Rows)
}

func planned(t *testing.T, g *compose.Row, f compose.Facts) []*ir.Kernel {
	t.Helper()
	if f.Target.API == compose.APINone {
		f.Target = sm70
	}
	pl, err := g.Plan(f)
	if err != nil {
		t.Fatal(err)
	}
	if len(pl) != 1 {
		t.Fatalf("planned %d kernels", len(pl))
	}
	return emitPlanned(t, pl[0])
}

// TestRouteRankIsMains holds the composer's rank to main's ExpertRank.
func TestRouteRankIsMains(t *testing.T) {
	for _, m := range routes() {
		name := routeName(m)
		t.Run(name, func(t *testing.T) {
			mk, err := mainkernels.ExpertRank(m)
			if err != nil {
				t.Fatal(err)
			}
			r := routeOf(m)
			g := &compose.Row{Width: m.NExpert, Rows: max(m.Rows, 1)}
			l, sel, top := g.Buf(compose.BufRow), g.Buf(compose.BufOutU32), g.Buf(compose.BufOut)
			bm, bias := -1, -1
			switch {
			case m.NGroup > 1:
				bm = g.Buf(compose.BufRow)
			case m.Bias:
				bias = g.Buf(compose.BufRow)
			}
			g.RouteRank(r, l, sel, top, bm, bias)
			same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{}))
		})
	}
}

// TestRouteWeightsIsMains holds the composer's weights to main's ExpertWeights.
func TestRouteWeightsIsMains(t *testing.T) {
	for _, m := range routes() {
		name := routeName(m)
		t.Run(name, func(t *testing.T) {
			mk, err := mainkernels.ExpertWeights(m)
			if err != nil {
				t.Fatal(err)
			}
			r := routeOf(m)
			g := &compose.Row{Width: 1, Rows: max(m.Rows, 1)}
			top, w := g.Buf(compose.BufRow), g.Buf(compose.BufOut)
			l := -1
			if !m.Norm && !(m.Sigmoid || m.SqrtSoftplus) {
				l = g.Buf(compose.BufRow)
			}
			g.RouteWeights(r, top, w, l)
			same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{}))
		})
	}
}

// TestRouteFusedIsMains holds the composer's one-row routes to main's
// ExpertRouteWarp (a subgroup) and ExpertRoute (a workgroup, an expert a
// thread).
func TestRouteFusedIsMains(t *testing.T) {
	for _, m := range []mainkernels.MoERoute{
		{NExpert: 8, K: 2, Norm: true}, {NExpert: 64, K: 8, Norm: true, Scale: 2.5}, {NExpert: 40, K: 4, Norm: true},
		{NExpert: 128, K: 8, Norm: true}, {NExpert: 60, K: 4}, {NExpert: 8, K: 2},
	} {
		name := routeName(m)
		r := routeOf(m)
		if m.Norm {
			t.Run("lanes_"+name, func(t *testing.T) {
				mk, err := mainkernels.ExpertRouteWarp(m)
				if err != nil {
					t.Fatal(err)
				}
				g := &compose.Row{Width: ir.SubgroupLanes, Rows: 1}
				l, sel, top, w := g.Buf(compose.BufRow), g.Buf(compose.BufOutU32), g.Buf(compose.BufOut), g.Buf(compose.BufOut)
				g.RouteLanes(r, l, sel, top, w)
				same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{}))
			})
		}
		t.Run("shared_"+name, func(t *testing.T) {
			mk, err := mainkernels.ExpertRoute(m)
			if err != nil {
				t.Fatal(err)
			}
			g := &compose.Row{Width: m.NExpert, Rows: 1}
			l, sel, top, w := g.Buf(compose.BufRow), g.Buf(compose.BufOutU32), g.Buf(compose.BufOut), g.Buf(compose.BufOut)
			g.RouteShared(r, l, sel, top, w)
			same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{}))
		})
	}
}

// TestCombineIsMains holds the composer's weighted sum of the slots to
// main's ExpertCombine.
func TestCombineIsMains(t *testing.T) {
	for _, c := range []struct{ n, slots int }{{64, 2}, {4096, 8}, {96, 6}} {
		name := fmt.Sprintf("n%d_s%d", c.n, c.slots)
		t.Run(name, func(t *testing.T) {
			mk, err := mainkernels.ExpertCombine(c.n, c.slots)
			if err != nil {
				t.Fatal(err)
			}
			g := &compose.Row{Width: c.n, Rows: 1}
			e, w, out := g.Buf(compose.BufRow), g.Buf(compose.BufRow), g.Buf(compose.BufOut)
			g.Store(out, g.Combine(e, w, c.slots))
			same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{}))
		})
	}
}

// TestGroupingIsMains holds the composer's grouping of a mixture's rows by
// expert, the gather, the grouped combine and the indexed bias to main's
// ExpertGroup*, GatherRows, ExpertCombineGrouped and IndexedBiasAdd.
func TestGroupingIsMains(t *testing.T) {
	for _, c := range []struct{ rows, k, nExp, unit int }{{32, 2, 8, 4}, {128, 8, 64, 4}, {16, 4, 128, 32}, {7, 3, 5, 4}} {
		name := fmt.Sprintf("r%d_k%d_e%d_u%d", c.rows, c.k, c.nExp, c.unit)
		np := (c.rows*c.k + c.nExp*(c.unit-1) + c.unit - 1) / c.unit * c.unit
		nb := compose.GroupBlocks(c.rows, c.k)
		check := func(t *testing.T, sub string, mk func() (*ir.Kernel, error), build func(g *compose.Row), width int) {
			t.Helper()
			k, err := mk()
			if err != nil {
				t.Fatal(err)
			}
			g := &compose.Row{Width: width, Rows: 1}
			build(g)
			same(t, name+"/"+sub, []*ir.Kernel{k}, planned(t, g, compose.Facts{}))
		}
		u32, out := compose.BufU32, compose.BufOutU32
		t.Run(name, func(t *testing.T) {
			check(t, "blockcount", func() (*ir.Kernel, error) { return mainkernels.ExpertGroupBlockCount(c.rows, c.k, c.nExp) }, func(g *compose.Row) {
				sel, bc := g.Buf(u32), g.Buf(out)
				g.Store(bc, g.GroupBlockCount(sel, c.rows, c.k, c.nExp))
			}, nb*c.nExp)
			check(t, "count", func() (*ir.Kernel, error) { return mainkernels.ExpertGroupCount(c.rows, c.k, c.nExp) }, func(g *compose.Row) {
				bc, off, cnt := g.Buf(u32), g.Buf(out), g.Buf(out)
				g.GroupCount(bc, off, cnt, c.rows, c.k, c.nExp)
			}, nb*c.nExp)
			check(t, "ends", func() (*ir.Kernel, error) { return mainkernels.ExpertGroupEnds(c.nExp, c.unit) }, func(g *compose.Row) {
				cnt, ends := g.Buf(u32), g.Buf(out)
				g.Store(ends, g.GroupEnds(cnt, c.nExp, c.unit))
			}, c.nExp)
			for _, pair := range []bool{false, true} {
				check(t, fmt.Sprintf("columns_%v", pair), func() (*ir.Kernel, error) { return mainkernels.ExpertGroupColumns(c.rows, c.nExp, np, pair) }, func(g *compose.Row) {
					ends, colE, perm, pp := g.Buf(u32), g.Buf(out), g.Buf(out), -1
					if pair {
						pp = g.Buf(out)
					}
					g.GroupColumns(ends, colE, perm, pp, c.rows, c.nExp)
				}, np)
				check(t, fmt.Sprintf("scatter_%v", pair), func() (*ir.Kernel, error) {
					return mainkernels.ExpertGroupScatter(c.rows, c.k, c.nExp, c.unit, pair)
				}, func(g *compose.Row) {
					sel, cnt, ends, off, perm, pos, pp := g.Buf(u32), g.Buf(u32), g.Buf(u32), g.Buf(u32), g.Buf(out), g.Buf(out), -1
					if pair {
						pp = g.Buf(out)
					}
					g.GroupScatter(sel, cnt, ends, off, perm, pos, pp, c.rows, c.k, c.nExp, c.unit)
				}, c.rows*c.k)
			}
			check(t, "tiles", func() (*ir.Kernel, error) { return mainkernels.ExpertGroupTiles(c.nExp, np/4, 4) }, func(g *compose.Row) {
				ends, tsel := g.Buf(u32), g.Buf(out)
				g.Store(tsel, g.GroupTiles(ends, c.nExp, 4))
			}, np/4)
			check(t, "gather", func() (*ir.Kernel, error) { return mainkernels.GatherRows(64, np, c.rows) }, func(g *compose.Row) {
				src, perm, dst := g.Buf(compose.BufRow), g.Buf(u32), g.Buf(compose.BufOut)
				g.Store(dst, g.GatherRows(src, perm, 64, c.rows))
			}, np*64)
			check(t, "combine", func() (*ir.Kernel, error) { return mainkernels.ExpertCombineGrouped(64, c.k, c.rows) }, func(g *compose.Row) {
				e, w, pos, o := g.Buf(compose.BufRow), g.Buf(compose.BufRow), g.Buf(u32), g.Buf(compose.BufOut)
				g.Store(o, g.CombineGrouped(e, w, pos, 64, c.k))
			}, c.rows*64)
			check(t, "bias", func() (*ir.Kernel, error) { return mainkernels.IndexedBiasAdd(c.k, 96) }, func(g *compose.Row) {
				in, sel, bias, o := g.Buf(compose.BufRow), g.Buf(u32), g.Buf(compose.BufRow), g.Buf(compose.BufOut)
				g.Store(o, g.IndexedBias(in, sel, bias, 96))
			}, c.k*96)
		})
	}
}

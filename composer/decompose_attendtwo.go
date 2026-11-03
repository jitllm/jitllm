package composer

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/engine/graph"
)

// attendSeg is one history an attention over two reads: its descriptors and
// page table, its pools and their row width, the attended range's attributes
// (the window), and a selection's keep mask where one selects in it.
type attendSeg struct {
	name               string
	hs                 history
	desc, pages        Operand
	keys, values       Operand
	keyPool, valuePool Pool
	kw                 int
	mask               *Operand
	maskGroups         int
}

// attendTwo is decomposeAttend over two histories (Reads[1] the entry
// history, graph.StateEnt): for each, every attended key's dot with the
// query, scaled, a selection's dropped positions and what the row does not
// attend at the least float, and their maximum; the larger of the two
// maxima; each history's weights by it, their sums and weighted values; and
// the row the two make together divided by the two sums and the sink's
// exp(sink - max): one softmax over both, with the head's sink.
func attendTwo(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.Reads) != 2 || o.Reads[1].State != graph.StateEnt || a.Softcap != 0 || a.Bidir || a.RowWindows || a.EntWidth < a.NKVHead*a.HeadDim {
		return nil, fmt.Errorf("an attention over %v, entries %d wide, softcap %v", o.Reads, a.EntWidth, a.Softcap)
	}
	kw := a.NKVHead * a.HeadDim
	if a.Width != 0 {
		kw = a.Width
	}
	values, vl := c.History.values()
	for _, l := range []Pool{c.History.Keys, vl} {
		if err := l.check(kw); err != nil {
			return nil, err
		}
	}
	desc := TableRows.op()
	if a.Window > 0 {
		desc = TableWindows.op()
	}
	ea := a
	ea.Window, ea.Chunked = 0, false
	segs := [2]attendSeg{
		{name: "kv", hs: history{a: a, pageSize: c.Step.Page}, desc: desc, pages: TablePages.op(), keys: c.History.Keys.operand(TableKeys.op()),
			values: values, keyPool: c.History.Keys, valuePool: vl, kw: kw},
		{name: "ent", hs: history{a: ea, pageSize: c.Step.Page}, desc: TableEntRows.op(), pages: TableEntPages.op(), keys: TableEntKeys.op(),
			values: TableEntKeys.op(), kw: a.EntWidth},
	}
	if len(o.In) == 2 {
		sel := c.producer(o.In[1])
		if sel == nil || sel.Kind != graph.OpSelect || a.NHead%sel.Attr.NKVHead != 0 || len(sel.Reads) != 1 || sel.Reads[0] != o.Reads[1] {
			return nil, fmt.Errorf("an attention of %d heads over a selection that is not of its entries", a.NHead)
		}
		m := maskOf(c, o.In[1])
		segs[1].mask, segs[1].maskGroups = &m, sel.Attr.NKVHead
	}
	U, F := ir.U32, ir.F32
	R, H, D, C := c.Step.Rows, a.NHead, a.HeadDim, c.Step.Keys
	gqa := a.NHead / a.NKVHead
	q, out := c.value(o.In[0]), c.value(o.Out[0])
	tmp := func(name string) Operand {
		return Operand{Kind: OperandScratch, Name: out.Name + "." + name, Temp: true}
	}
	minus := math.Float32frombits(0xff7fffff)
	keys, heads, dims := []int{R, H, C}, []int{R, H}, []int{R, H, D}
	gmax := tmp("max")

	var prims []Prim
	var maxes, weights, sums, vals [2]Operand
	tablesOf := func(s attendSeg) func(...Operand) []Operand {
		return func(ops ...Operand) []Operand { return append([]Operand{ops[0], s.desc, s.pages}, ops[1:]...) }
	}
	rngOf := func(s attendSeg, axis int) Facts {
		return Facts{Range: s.hs.span(axis), Extents: map[int]int{1: R * rowWords, 2: c.pages()}}
	}
	for i, s := range segs {
		tables := tablesOf(s)
		dots, scores := tmp(s.name+".dots"), tmp(s.name+".scores")
		maxes[i], weights[i], sums[i], vals[i] = tmp(s.name+".max"), tmp(s.name+".weights"), tmp(s.name+".sum"), tmp(s.name+".values")

		var dx exprs
		h, d := dx.coord(1), dx.coord(3)
		pos, first := s.hs.position(&dx)
		pid, off := s.hs.page(&dx, s.hs.held(&dx, pos, first))
		kd := dx.op(OpAdd, U, dx.op(OpMul, U, dx.op(OpDiv, U, h, dx.u(uint32(gqa))), dx.u(uint32(D))), d)
		qat := dx.op(OpAdd, U, dx.op(OpMul, U, dx.op(OpAdd, U, dx.op(OpMul, U, dx.coord(0), dx.u(uint32(H))), h), dx.u(uint32(D))), d)
		dx.op(OpMul, F, dx.load(0, qat), c.poolRead(&dx, 3, s.keyPool, s.kw, pid, off, kd))

		var sx exprs
		spos, sfirst := s.hs.position(&sx)
		valid := sx.op(OpLess, ir.Pred, sx.op(OpSub, U, spos, sfirst), sx.op(OpSub, U, sx.live(), sfirst))
		score := sx.op(OpMul, F, sx.at(0, keys, sx.coord(0), sx.coord(1), sx.coord(2)), sx.f(float32(a.Scale)))
		scoreIn := tables(dots)
		if s.mask != nil {
			G := max(1, s.maskGroups)
			grp := sx.op(OpDiv, U, sx.coord(1), sx.u(uint32(H/G)))
			keep := sx.at(3, []int{R, G, C}, sx.coord(0), grp, sx.coord(2))
			score = sx.sel(F, sx.op(OpLess, ir.Pred, sx.f(0.5), keep), score, sx.f(minus))
			scoreIn = tables(dots, *s.mask)
		}
		sx.sel(F, valid, score, sx.f(minus))

		var mx exprs
		mx.at(0, keys, mx.coord(0), mx.coord(1), mx.coord(2))

		scored := rngOf(s, 2)
		scored.Extents[0] = R * H * D
		prims = append(prims,
			Prim{Kind: PrimContract, Shape: []int{R, H, C, D}, In: tables(q, s.keys), Out: dots, Expr: dx.e, Facts: scored},
			scoreMap(keys, scoreIn, scores, sx.e, rngOf(s, 2)),
			Prim{Kind: PrimReduce, Shape: keys, In: tables(scores), Out: maxes[i], Combine: OpMax, Init: int64(math.Float32bits(minus)),
				Expr: mx.e, Facts: rngOf(s, 2)})
		// scores is the weights' input below.
		weights[i] = scores
	}

	var gx exprs
	gx.op(OpMax, F, gx.at(0, heads, gx.coord(0), gx.coord(1)), gx.at(1, heads, gx.coord(0), gx.coord(1)))
	prims = append(prims, Prim{Kind: PrimMap, Shape: heads, In: []Operand{maxes[0], maxes[1]}, Out: gmax, Expr: gx.e})

	for i, s := range segs {
		tables := tablesOf(s)
		scores, w := weights[i], tmp(s.name+".w")
		weights[i] = w

		var wx exprs
		wpos, _ := s.hs.position(&wx)
		below := wx.op(OpLess, ir.Pred, wpos, wx.live())
		sv := wx.at(0, keys, wx.coord(0), wx.coord(1), wx.coord(2))
		m := wx.at(3, heads, wx.coord(0), wx.coord(1))
		wx.sel(F, below, wx.un(OpExp, F, wx.op(OpSub, F, sv, m)), wx.f(0))

		var lx exprs
		lx.at(0, keys, lx.coord(0), lx.coord(1), lx.coord(2))

		var vx exprs
		vh, vd, vk := vx.coord(1), vx.coord(2), vx.coord(3)
		vpos, vfirst := s.hs.position(&vx)
		vpid, voff := s.hs.page(&vx, s.hs.held(&vx, vpos, vfirst))
		vd2 := vx.op(OpAdd, U, vx.op(OpMul, U, vx.op(OpDiv, U, vh, vx.u(uint32(gqa))), vx.u(uint32(D))), vd)
		vx.op(OpMul, F, vx.at(0, keys, vx.coord(0), vh, vk), c.poolRead(&vx, 3, s.valuePool, s.kw, vpid, voff, vd2))

		prims = append(prims,
			Prim{Kind: PrimMap, Shape: keys, In: tables(scores, gmax), Out: w, Expr: wx.e, Facts: rngOf(s, 2)},
			Prim{Kind: PrimReduce, Shape: keys, In: tables(w), Out: sums[i], Combine: OpAdd, Expr: lx.e, Facts: rngOf(s, 2)},
			Prim{Kind: PrimContract, Shape: []int{R, H, D, C}, In: tables(w, s.values), Out: vals[i], Expr: vx.e, Facts: rngOf(s, 3)})
	}

	var ox exprs
	r, h := ox.coord(0), ox.coord(1)
	nv := ox.op(OpAdd, F, ox.at(0, dims, r, h, ox.coord(2)), ox.at(1, dims, r, h, ox.coord(2)))
	l := ox.op(OpAdd, F, ox.at(2, heads, r, h), ox.at(3, heads, r, h))
	outIn := []Operand{vals[0], vals[1], sums[0], sums[1]}
	if len(o.Weight) == 1 {
		l = ox.op(OpAdd, F, l, ox.un(OpExp, F, ox.op(OpSub, F, ox.at(5, []int{H}, h), ox.at(4, heads, r, h))))
		outIn = append(outIn, gmax, sinkVec(o))
	}
	ox.op(OpMul, F, nv, ox.op(OpDiv, F, ox.f(1), ox.op(OpMax, F, l, ox.f(1e-30))))
	return append(prims, Prim{Kind: PrimMap, Shape: dims, In: outIn, Out: out, Expr: ox.e}), nil
}

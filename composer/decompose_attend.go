package composer

import (
	"fmt"
	"math"
	"slices"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// sinkVec is an attention's per-head sinks.
func sinkVec(o graph.Op) Operand {
	return Operand{Kind: OperandVector, Role: o.Weight[0].Role, Name: o.Weight[0].Role.String()}
}

// scoreMap is the score primitive.
func scoreMap(shape []int, in []Operand, out Operand, e []Expr, f Facts) Prim {
	return Prim{Kind: PrimMap, Shape: shape, In: in, Out: out, Expr: e, Facts: f}
}

// decomposeAttend is decode attention over the paged history: per row and
// query head, each key's dot with the query, scaled and softcapped where the
// op is, the scores masked outside what the row attends, their maximum, the
// weights and their sum (at least one, its largest term exp 0), the weighted
// values and the row they make divided by the sum. A causal windowed op
// attends what the rows' windowed descriptors (the step's window table)
// say, as the engine's own windowed kernels do: the engine stages each
// row's first position as its window's start, the Window positions ending
// at the query's or the query's aligned chunk.
func decomposeAttend(o graph.Op, c *Context) ([]Prim, error) {
	undo, err := c.onHistory(o)
	if err != nil {
		return nil, err
	}
	defer undo()
	a := o.Attr
	if len(o.In) != 1 && len(o.In) != 2 || len(o.Out) != 1 || len(o.Weight) > 1 || a.Window < 0 || a.Softcap < 0 ||
		a.NHead <= 0 || a.NKVHead <= 0 || a.NHead%a.NKVHead != 0 || c.Step.Page <= 0 || c.Step.Keys <= 0 {
		return nil, fmt.Errorf("an attention of %d heads over %d, window %d, softcap %v", a.NHead, a.NKVHead, a.Window, a.Softcap)
	}
	kw := a.NKVHead * a.HeadDim
	if a.Width != 0 {
		if a.Width < kw {
			return nil, fmt.Errorf("an attention of %d heads of %d over history rows of %d", a.NKVHead, a.HeadDim, a.Width)
		}
		kw = a.Width
	}
	if len(o.Reads) == 2 {
		return attendBoth(o, c, kw)
	}
	g, err := c.attendGeomOf(o.Reads[0], 1, kw, a)
	if err != nil {
		return nil, err
	}
	if len(o.In) == 2 {
		// A block selection: the positions each kv group's mask keeps.
		sel := c.producer(o.In[1])
		if c.History.Passes {
			return nil, fmt.Errorf("a block selection over a history read in passes")
		}
		if sel == nil || sel.Kind != graph.OpSelect || a.NHead%sel.Attr.NKVHead != 0 || a.Window != 0 || a.Bidir {
			return nil, fmt.Errorf("an attention of %d heads, window %d, over a selection that is not its own history's", a.NHead, a.Window)
		}
		m := maskOf(c, o.In[1])
		g.Mask, g.MaskGroups = &m, sel.Attr.NKVHead
	}
	return attend(o, c, c.value(o.In[0]), c.value(o.Out[0]), g), nil
}

// attendGeomOf is an attention's geometry over the history r, an entry each
// period positions, its rows kw wide.
func (c *Context) attendGeomOf(r graph.StateRef, period, kw int, a graph.Attrs) (attendGeom, error) {
	l, err := c.layoutOf(r.Hist)
	if err != nil {
		return attendGeom{}, err
	}
	if max(1, l.Period) != max(1, period) {
		return attendGeom{}, fmt.Errorf("an attention over an entry each %d positions of a history of one each %d", period, l.Period)
	}
	values, vl := l.values(r.Hist)
	for _, p := range []Pool{l.Keys, vl} {
		if err := p.check(kw); err != nil {
			return attendGeom{}, err
		}
	}
	g := attendGeom{D: a.HeadDim, KW: kw, Keys: l.Keys.operand(TableKeys.of(int(r.Hist))), Values: values, ValuePool: vl,
		KeyPool: l.Keys, Slots: l.slots(c.Step.Page), Pages: TablePages.of(int(r.Hist)), Period: period}
	// A block sharing another's history (graph.SharedHistory) reads that
	// block's pools as it left them.
	if r.From > 0 {
		n := r.From - 1
		if c.Graph.Block < 0 {
			return attendGeom{}, fmt.Errorf("an attention over block %d's history outside a block", n)
		}
		g.Keys.Of, g.Values.Of = c.blockPart(n), c.blockPart(n)
	}
	return g, nil
}

// attendBoth is an attention over two histories in one softmax: each read
// as a partial of its own (attendPartial), the second's entries masked by a
// selection where the op has one, then merged as a pass is folded into the
// running partial (foldPasses): each side's weighted values and sum scaled
// by exp of its maximum less the larger, a side that attended nothing
// dropped, a sink's exp(sink - max) in the sum once.
func attendBoth(o graph.Op, c *Context, kw int) ([]Prim, error) {
	a := o.Attr
	if c.History.Passes {
		return nil, fmt.Errorf("an attention over two histories read in passes")
	}
	rows, err := c.attendGeomOf(o.Reads[0], 1, kw, a)
	if err != nil {
		return nil, err
	}
	ents, err := c.attendGeomOf(o.Reads[1], a.Span, int(a.Words[0]), a)
	if err != nil {
		return nil, err
	}
	ents.Tag = "e."
	if len(o.In) == 2 {
		sel := c.producer(o.In[1])
		if sel == nil || sel.Kind != graph.OpSelect || a.NHead%sel.Attr.NKVHead != 0 {
			return nil, fmt.Errorf("an attention of %d heads over a selection that is not its entries' groups", a.NHead)
		}
		m := maskOf(c, o.In[1])
		ents.Mask, ents.MaskGroups = &m, sel.Attr.NKVHead
	}
	q, out := c.value(o.In[0]), c.value(o.Out[0])
	ea := a
	ea.Window, ea.Chunked, ea.RowWindows = 0, false, false
	pr, vr, lr, mr := attendPartial(o, a, c, q, out, rows)
	pe, ve, le, me := attendPartial(o, ea, c, q, out, ents)
	R, H, D := c.Step.Rows, a.NHead, a.HeadDim
	F := ir.F32
	heads, dims := []int{R, H}, []int{R, H, D}
	minus := math.Float32frombits(0xff7fffff)

	var x exprs
	r, h, d := x.coord(0), x.coord(1), x.coord(2)
	mo, lo, mj, lj := x.at(2, heads, r, h), x.at(1, heads, r, h), x.at(5, heads, r, h), x.at(4, heads, r, h)
	held := x.op(OpLess, ir.Pred, x.f(0), lo)
	seen := x.op(OpLess, ir.Pred, x.f(minus/2), mj)
	mn := x.sel(F, held, x.sel(F, seen, x.op(OpMax, F, mo, mj), mo), mj)
	ex := x.sel(F, held, x.un(OpExp, F, x.op(OpSub, F, mo, mn)), x.f(0))
	ey := x.sel(F, seen, x.un(OpExp, F, x.op(OpSub, F, mj, mn)), x.f(0))
	acc := x.op(OpAdd, F, x.op(OpMul, F, x.at(0, dims, r, h, d), ex), x.op(OpMul, F, x.at(3, dims, r, h, d), ey))
	l := x.op(OpAdd, F, x.op(OpMul, F, lo, ex), x.op(OpMul, F, lj, ey))
	in := []Operand{vr, lr, mr, ve, le, me}
	if len(o.Weight) == 1 {
		l = x.op(OpAdd, F, l, x.un(OpExp, F, x.op(OpSub, F, x.at(6, []int{H}, h), mn)))
		in = append(in, sinkVec(o))
	}
	x.op(OpMul, F, acc, x.op(OpDiv, F, x.f(1), x.op(OpMax, F, l, x.f(1e-30))))
	prims := append(pr, pe...)
	return append(prims, Prim{Kind: PrimMap, Shape: dims, In: in, Out: out, Expr: x.e,
		Facts: Facts{Heads: &Heads{Rows: R, Query: H, Key: a.NKVHead, Dim: D, Seqs: c.Step.Seqs}}}), nil
}

// attendGeom is an attention's shape over the history's pools: D a key
// head's width and a value head's (the work passes take one head width), KW
// a pool row's, the keys read from Keys and the values from Values.
type attendGeom struct {
	D, KW        int
	Keys, Values Operand
	// Mask, where set, is a selection's keep mask, the attention's second
	// input, MaskGroups groups of the query heads each a row of it: a
	// position it does not keep scores as the least float.
	Mask       *Operand
	MaskGroups int
	// ValuePool is how the values' pool lays out a page, KeyPool the keys',
	// Slots a page's slots in either (History.slots) and Pages the page
	// table they are read through.
	ValuePool, KeyPool Pool
	Slots              int
	Pages              Operand
	// Period, past 1, reads a history of an entry each Period positions.
	Period int
	// Tag sets this history's scratch apart from another's the op reads.
	Tag string
}

// attend is decodeAttend's primitives for the query q (rows by heads by
// g.D) into out (the same), o's attributes its heads, scale, window,
// softcap and sinks.
func attend(o graph.Op, c *Context, q, out Operand, g attendGeom) []Prim {
	a := o.Attr
	R, H, D := c.Step.Rows, a.NHead, g.D
	prims, values, sums, maxes := attendPartial(o, a, c, q, out, g)
	if c.History.Passes {
		return append(prims, foldPasses(o, R, H, D, values, sums, maxes, out)...)
	}
	var ox exprs
	nv := ox.at(0, []int{R, H, D}, ox.coord(0), ox.coord(1), ox.coord(2))
	l := ox.at(1, []int{R, H}, ox.coord(0), ox.coord(1))
	outIn := []Operand{values, sums}
	if len(o.Weight) == 1 {
		// A sink is one more logit a head in the denominator alone: the
		// sum gains exp(sink - max), the max the keys' alone, so the
		// attention keeps the form of one without (the split's partitions
		// and their combine). A sink past the max by more than a float's
		// exponent makes the sum infinite and the row zero, its limit.
		m := ox.at(2, []int{R, H}, ox.coord(0), ox.coord(1))
		l = ox.op(OpAdd, ir.F32, l, ox.un(OpExp, ir.F32, ox.op(OpSub, ir.F32, ox.at(3, []int{H}, ox.coord(1)), m)))
		outIn = append(outIn, maxes, sinkVec(o))
	}
	ox.op(OpMul, ir.F32, nv, ox.op(OpDiv, ir.F32, ox.f(1), ox.op(OpMax, ir.F32, l, ox.f(1e-30))))
	return append(prims, Prim{Kind: PrimMap, Shape: []int{R, H, D}, In: outIn, Out: out, Expr: ox.e, Facts: Facts{Heads: &Heads{Rows: R, Query: H, Key: a.NKVHead, Dim: D, Seqs: c.Step.Seqs}}})
}

// attendPartial is attend's primitives up to its partial over the history
// g reads, a's heads, scale, window and softcap: the weighted values, their
// sum and the maximum score, each a row's head's.
func attendPartial(o graph.Op, a graph.Attrs, c *Context, q, out Operand, g attendGeom) (prims []Prim, values, sums, maxes Operand) {
	U, F := ir.U32, ir.F32
	R, H, D := c.Step.Rows, a.NHead, g.D
	C := c.Step.Keys
	if g.Period > 1 {
		C = max(1, C/g.Period)
	}
	gqa, kw, pg := a.NHead/a.NKVHead, g.KW, c.Step.Page
	tmp := func(name string) Operand {
		return Operand{Kind: OperandScratch, Name: out.Name + "." + g.Tag + name, Temp: true}
	}
	dots, scores, weights := tmp("dots"), tmp("scores"), tmp("weights")
	maxes, sums, values = tmp("max"), tmp("sum"), tmp("values")
	minus := math.Float32frombits(0xff7fffff)
	keys, heads := []int{R, H, C}, []int{R, H}

	hs := history{a: a, pageSize: pg, period: g.Period}
	position, held, page := hs.position, hs.held, hs.page
	rng := func(axis int) Facts {
		return Facts{Range: hs.span(axis), Heads: &Heads{Rows: R, Query: H, Key: a.NKVHead, Dim: D, Seqs: c.Step.Seqs},
			Extents: map[int]int{1: R * rowWords, 2: c.pages()}}
	}

	var dx exprs
	h, d := dx.coord(1), dx.coord(3)
	pos, first := position(&dx)
	pid, off := page(&dx, held(&dx, pos, first))
	kh := dx.op(OpDiv, U, h, dx.u(uint32(gqa)))
	kd := dx.op(OpAdd, U, dx.op(OpMul, U, kh, dx.u(uint32(D))), d)
	qat := dx.op(OpAdd, U, dx.op(OpMul, U, dx.op(OpAdd, U, dx.op(OpMul, U, dx.coord(0), dx.u(uint32(H))), h), dx.u(uint32(D))), d)
	dx.op(OpMul, F, dx.load(0, qat), c.poolReadIn(&dx, 3, g.KeyPool, g.Slots, kw, pid, off, kd))

	var sx exprs
	spos, sfirst := position(&sx)
	valid := sx.op(OpLess, ir.Pred, sx.op(OpSub, U, spos, sfirst), sx.op(OpSub, U, sx.live(), sfirst))
	dot := sx.at(0, keys, sx.coord(0), sx.coord(1), sx.coord(2))
	score := sx.op(OpMul, F, dot, sx.f(float32(a.Scale)))
	if c := float32(a.Softcap); c > 0 {
		e := sx.un(OpExp, F, sx.op(OpMul, F, sx.f(2/c), score))
		score = sx.op(OpMul, F, sx.f(c), sx.op(OpSub, F, sx.f(1), sx.op(OpDiv, F, sx.f(2), sx.op(OpAdd, F, e, sx.f(1)))))
	}
	if g.Mask != nil {
		G := max(1, g.MaskGroups)
		grp := sx.op(OpDiv, U, sx.coord(1), sx.u(uint32(H/G)))
		keep := sx.at(3, []int{R, G, C}, sx.coord(0), grp, sx.coord(2))
		score = sx.sel(F, sx.op(OpLess, ir.Pred, sx.f(0.5), keep), score, sx.f(minus))
	}
	sx.sel(F, valid, score, sx.f(minus))

	var mx exprs
	mx.at(0, keys, mx.coord(0), mx.coord(1), mx.coord(2))

	var wx exprs
	wpos, _ := position(&wx)
	below := wx.op(OpLess, ir.Pred, wpos, wx.live())
	s := wx.at(0, keys, wx.coord(0), wx.coord(1), wx.coord(2))
	m := wx.at(3, heads, wx.coord(0), wx.coord(1))
	wx.sel(F, below, wx.un(OpExp, F, wx.op(OpSub, F, s, m)), wx.f(0))

	var lx exprs
	lx.at(0, keys, lx.coord(0), lx.coord(1), lx.coord(2))

	var vx exprs
	vh, vd, vk := vx.coord(1), vx.coord(2), vx.coord(3)
	vpos, vfirst := position(&vx)
	vpid, voff := page(&vx, held(&vx, vpos, vfirst))
	vkh := vx.op(OpDiv, U, vh, vx.u(uint32(gqa)))
	vd2 := vx.op(OpAdd, U, vx.op(OpMul, U, vkh, vx.u(uint32(D))), vd)
	w := vx.at(0, keys, vx.coord(0), vh, vk)
	v := c.poolReadIn(&vx, 3, g.ValuePool, g.Slots, kw, vpid, voff, vd2)
	if c.History.Passes {
		// A pass the row attends nothing of reads a page outside the pass.
		v = vx.sel(F, vx.op(OpLess, ir.Pred, vpos, vx.live()), v, vx.f(0))
	}
	vx.op(OpMul, F, w, v)

	sum := rng(2)
	if !c.History.Passes {
		sum.Least = 1
	}
	scored := rng(2)
	scored.Extents[0] = R * H * D
	desc := TableRows.op()
	if g.Period <= 1 && (a.RowWindows || a.Window > 0 && !a.Bidir) {
		desc = TableWindows.op()
	}
	tables := func(ops ...Operand) []Operand { return append([]Operand{ops[0], desc, g.Pages}, ops[1:]...) }
	// The score map reads the mask as its fourth operand.
	withMask := func(d Operand) []Operand {
		if g.Mask != nil {
			return tables(d, *g.Mask)
		}
		return tables(d)
	}
	prims = []Prim{
		{Kind: PrimContract, Shape: []int{R, H, C, D}, In: tables(q, g.Keys), Out: dots, Expr: dx.e, Facts: scored},
		scoreMap(keys, withMask(dots), scores, sx.e, rng(2)),
		{Kind: PrimReduce, Shape: keys, In: tables(scores), Out: maxes, Combine: OpMax, Init: int64(math.Float32bits(minus)),
			Expr: mx.e, Facts: rng(2)},
		{Kind: PrimMap, Shape: keys, In: tables(scores, maxes), Out: weights, Expr: wx.e, Facts: rng(2)},
		{Kind: PrimReduce, Shape: keys, In: tables(weights), Out: sums, Combine: OpAdd, Expr: lx.e, Facts: sum},
		{Kind: PrimContract, Shape: []int{R, H, D, C}, In: tables(weights, g.Values), Out: values, Expr: vx.e, Facts: rng(3)},
	}
	return prims, values, sums, maxes
}

// The running partial of an attention read in passes: a row's heads, each
// its weighted values, then their maximum, then their sum.

// foldsAPass reports whether p folds a pass into the running partial: a
// pass ends at its kernel (Kernel.Folds), and what reads the partial runs
// once, after the last pass.
func foldsAPass(p Prim) bool {
	return p.Out.Kind == TableRunOut.op().Kind && p.Out.Name == TableRunOut.op().Name
}

// foldPasses folds a pass's weighted values, sums and maxima into the
// running partial, each side scaled by exp of its maximum less the larger,
// and finishes the row from the fold: the values over the sum, a sink's
// exp(sink - max) in the sum alone. A partial of sum zero has attended
// nothing, whatever its maximum: the first pass folds into zeros.
func foldPasses(o graph.Op, R, H, D int, values, sums, maxes, out Operand) []Prim {
	U, F := ir.U32, ir.F32
	runs, heads, dims := []int{R, H, D + 2}, []int{R, H}, []int{R, H, D}

	var fx exprs
	r, h, d := fx.coord(0), fx.coord(1), fx.coord(2)
	mo, lo := fx.at(3, runs, r, h, fx.u(uint32(D))), fx.at(3, runs, r, h, fx.u(uint32(D+1)))
	mj, lj := fx.at(2, heads, r, h), fx.at(1, heads, r, h)
	held := fx.op(OpLess, ir.Pred, fx.f(0), lo)
	// A pass whose every key the row may not see has the mask's maximum,
	// its weights exp(0) each: it attended nothing.
	seen := fx.op(OpLess, ir.Pred, fx.f(math.Float32frombits(0xff7fffff)/2), mj)
	mn := fx.sel(F, held, fx.sel(F, seen, fx.op(OpMax, F, mo, mj), mo), mj)
	ea := fx.sel(F, held, fx.un(OpExp, F, fx.op(OpSub, F, mo, mn)), fx.f(0))
	eb := fx.sel(F, seen, fx.un(OpExp, F, fx.op(OpSub, F, mj, mn)), fx.f(0))
	vd := fx.op(OpMin, U, d, fx.u(uint32(D-1)))
	acc := fx.op(OpAdd, F, fx.op(OpMul, F, fx.at(3, runs, r, h, d), ea), fx.op(OpMul, F, fx.at(0, dims, r, h, vd), eb))
	sum := fx.op(OpAdd, F, fx.op(OpMul, F, lo, ea), fx.op(OpMul, F, lj, eb))
	tail := fx.sel(F, fx.op(OpLess, ir.Pred, d, fx.u(uint32(D+1))), mn, sum)
	fx.sel(F, fx.op(OpLess, ir.Pred, d, fx.u(uint32(D))), acc, tail)

	var ex exprs
	er, eh := ex.coord(0), ex.coord(1)
	l := ex.at(0, runs, er, eh, ex.u(uint32(D+1)))
	in := []Operand{TableRunOut.op()}
	if len(o.Weight) == 1 {
		m := ex.at(0, runs, er, eh, ex.u(uint32(D)))
		l = ex.op(OpAdd, F, l, ex.un(OpExp, F, ex.op(OpSub, F, ex.at(1, []int{H}, eh), m)))
		in = append(in, sinkVec(o))
	}
	ex.op(OpMul, F, ex.at(0, runs, er, eh, ex.coord(2)), ex.op(OpDiv, F, ex.f(1), ex.op(OpMax, F, l, ex.f(1e-30))))
	return []Prim{
		{Kind: PrimMap, Shape: runs, In: []Operand{values, sums, maxes, TableRunIn.op()}, Out: TableRunOut.op(), Expr: fx.e},
		{Kind: PrimMap, Shape: dims, In: in, Out: out, Expr: ex.e},
	}
}

// history is how a primitive reads the paged history of a row: the rows'
// descriptors its operand 1 and the page table its operand 2, row r its
// coordinate 0, the attended positions those the descriptor states -- a
// causal window's descriptors carry its start -- or for a bidirectional op
// the symmetric window of Window/2 either side of the row's own position.
type history struct {
	a        graph.Attrs
	pageSize int
	// period, past 1, reads entries (History.Period): the row attends
	// entries 0 to its range's end over period, entry j held at the page of
	// the position closing it, clamped to the row's own position so an entry
	// the row cannot see yet is read where it holds a page.
	period int
}

// row is word w of the row's descriptor.
func (hs history) row(x *exprs, w int) int {
	U := ir.U32
	return x.load(1, x.op(OpAdd, U, x.op(OpMul, U, x.coord(0), x.u(rowWords)), x.u(uint32(w))))
}

// start is the first position the row attends: its first, or past it a
// bidirectional window's. A causal window's first is its descriptor's: a
// start computed here as well would hide a descriptor that lost the window.
func (hs history) start(x *exprs) int {
	U, a := ir.U32, hs.a
	if hs.period > 1 {
		return x.u(0)
	}
	first := hs.row(x, rowStart)
	if a.Window == 0 || !a.Bidir {
		return first
	}
	half := x.u(uint32(a.Window / 2))
	return x.op(OpMax, U, first, x.op(OpSub, U, x.op(OpMax, U, hs.row(x, rowWrite), half), half))
}

// span is the attended range along axis, its bounds the row's.
func (hs history) span(axis int) *Range {
	bound := func(w int) []Expr {
		var x exprs
		if w == rowStart {
			hs.start(&x)
		} else {
			hs.end(&x)
		}
		return x.e
	}
	return &Range{Axis: axis, First: bound(rowStart), End: bound(rowEnd)}
}

// end is one past the last position the row attends: its end, or within
// a symmetric window the window's.
func (hs history) end(x *exprs) int {
	end := hs.row(x, rowEnd)
	if hs.period > 1 {
		return x.op(OpDiv, ir.U32, end, x.u(uint32(hs.period)))
	}
	if a := hs.a; a.Window > 0 && a.Bidir {
		U := ir.U32
		return x.op(OpMin, U, end, x.op(OpAdd, U, hs.row(x, rowWrite), x.u(uint32(a.Window/2+1))))
	}
	return end
}

// position is the key's position and the row's first.
func (hs history) position(x *exprs) (pos, first int) {
	return x.pos(), hs.start(x)
}

// held is a position kept inside what the row attends.
func (hs history) held(x *exprs, pos, first int) int {
	U := ir.U32
	last := x.op(OpSub, U, x.op(OpMax, U, x.live(), x.u(1)), x.u(1))
	return x.op(OpMin, U, x.op(OpMax, U, pos, first), last)
}

// page is position t's page and its offset in the page.
func (hs history) page(x *exprs, t int) (pid, off int) {
	U := ir.U32
	if P := hs.period; P > 1 {
		t = x.op(OpMin, U, x.op(OpSub, U, x.op(OpMul, U, x.op(OpAdd, U, t, x.u(1)), x.u(uint32(P))), x.u(1)), hs.row(x, rowWrite))
	}
	pid = x.load(2, x.op(OpAdd, U, hs.row(x, rowTab), x.op(OpDiv, U, t, x.u(uint32(hs.pageSize)))))
	off = x.op(OpRem, U, t, x.u(uint32(hs.pageSize)))
	if hs.period > 1 {
		off = x.op(OpDiv, U, off, x.u(uint32(hs.period)))
	}
	return pid, off
}

// producer is the op of the graph that writes v, nil for none.
func (c *Context) producer(v graph.ValueID) *graph.Op {
	for i := range c.Graph.Ops {
		if slices.Contains(c.Graph.Ops[i].Out, v) {
			return &c.Graph.Ops[i]
		}
	}
	return nil
}

// maskOf is the keep mask value sel names, which an attention reads: one a
// row and attended position, 1 kept and 0 not.
func maskOf(c *Context, sel graph.ValueID) Operand {
	return Operand{Kind: OperandScratch, Name: c.value(sel).Name + ".mask"}
}

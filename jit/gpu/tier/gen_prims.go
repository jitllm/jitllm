package tier

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// The graph's primitives (graph.OpMap, OpReduce, OpTopK, OpRank, OpTake),
// each a row group formed from its op's attributes and its values' widths
// alone: a value of the program is a row of its width a row of the step, an
// operand narrower than the output is broadcast (an element of it a group of
// the output's), and a bound vector is one row for every row. Nothing here
// knows which composite op a primitive came from.

// genPrim is primitive op o's launch at rows rows, or nil (o.Why says why).
func (g *devTier) genPrim(o *devOp, prog *devProgram, rows int) []genLaunch {
	if len(o.Out) == 0 || len(o.In) == 0 {
		o.Why = whyShape
		return nil
	}
	w := func(v devVal) int { return prog.vals[v].Width }
	// st is value v's row stride: its width and any padding past it.
	st := prog.stride
	index := func(v devVal) bool { return prog.vals[v].Elem == elemU32 }
	rb := &rowBuild{}
	r := &rb.r
	u := func(n int) int { return r.UConst(n) }
	// at is the element of value v (input j's buffer b) the output's column
	// col reads: its row's, broadcast where v is narrower.
	at := func(b int, v devVal, out int, col int) (int, bool) {
		n := w(v)
		if n == 0 || out%n != 0 {
			return 0, false
		}
		ix := col
		if n != out {
			ix = r.UBin(compose.OpDiv, col, u(out/n))
		}
		return r.LoadAt(b, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(st(v))), ix)), true
	}
	inBuf := func(j int) int {
		k := compose.BufRow
		if index(o.In[j]) {
			k = compose.BufU32
		}
		return rb.buf(k, gaIn0+genArg(j))
	}
	switch o.Kind {
	case dkMap:
		out := w(o.Out[0])
		*r = compose.Row{Width: out, Rows: rows}
		col := r.Column()
		var ops []int
		for j := range o.In {
			x, ok := at(inBuf(j), o.In[j], out, col)
			if !ok {
				o.Why = whyShape
				return nil
			}
			if index(o.In[j]) {
				x = rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpFloat, A: x})
			}
			ops = append(ops, x)
		}
		if o.V != nil {
			// A bound vector: one row, read by every row at the column.
			ops = append(ops, r.LoadAt(rb.buf(compose.BufVec, gaV), col))
		}
		if len(ops) < o.A.Fn.Arity() {
			ops = append(ops, r.Const(float32(o.A.Const)))
		}
		if len(ops) != o.A.Fn.Arity() {
			o.Why = whyShape
			return nil
		}
		var y int
		switch fn := o.A.Fn; fn {
		case graph.MapAdd, graph.MapSub, graph.MapMul, graph.MapDiv, graph.MapMax, graph.MapMin:
			y = r.Bin(map[graph.MapFn]compose.Op{graph.MapAdd: compose.OpAdd, graph.MapSub: compose.OpSub,
				graph.MapMul: compose.OpMul, graph.MapDiv: compose.OpDiv, graph.MapMax: compose.OpMax,
				graph.MapMin: compose.OpMin}[fn], ops[0], ops[1])
		case graph.MapExp:
			y = rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpExp, A: ops[0]})
		case graph.MapSqrt:
			y = rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpSqrt, A: ops[0]})
		case graph.MapSigmoid:
			y = rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpAct, Act: kernels.ActSigmoid, A: ops[0]})
		case graph.MapSoftplus:
			y = rb.node(compose.Node{Kind: compose.NUnary, Op: compose.OpAct, Act: kernels.ActSoftplus, A: ops[0]})
		case graph.MapWhere:
			y = r.Select(r.Less(r.Const(0), ops[0]), ops[1], ops[2])
		case graph.MapLess:
			y = r.Select(r.Less(ops[0], ops[1]), r.Const(1), r.Const(0))
		case graph.MapFma:
			y = rb.node(compose.Node{Kind: compose.NBinary, Op: compose.OpFma, A: ops[0], B: ops[1], C: ops[2]})
		default:
			o.Why = whyShape
			return nil
		}
		r.StoreAt(rb.buf(compose.BufOut, gaOut0), y, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(st(o.Out[0]))), col))
	case dkReduce:
		n, out := w(o.In[0]), w(o.Out[0])
		span := o.A.Span
		if span == 0 {
			span = n
		}
		if o.A.Order != graph.OrderSeq || span < 1 || n%span != 0 || out != n/span || index(o.In[0]) {
			o.Why = whyShape
			return nil
		}
		*r = compose.Row{Width: out, Rows: rows}
		b := inBuf(0)
		base := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(st(o.In[0]))), r.UBin(compose.OpMul, r.Column(), u(span)))
		x := func(f int) int { return r.LoadAt(b, r.UBin(compose.OpAdd, base, f)) }
		var y int
		switch o.A.Fn {
		case graph.MapAdd:
			y = r.FoldSum(span, x)
		case graph.MapMax:
			y = r.FoldMax(span, x)
		case graph.MapMin:
			c := r.Loop(span)
			m := r.Fold(c, x(u(0)))
			r.SetFold(m, r.Bin(compose.OpMin, m, x(r.Iter(c))))
			y = m
		default:
			o.Why = whyShape
			return nil
		}
		r.StoreAt(rb.buf(compose.BufOut, gaOut0), y, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(st(o.Out[0]))), r.Column()))
	case dkRank, dkSelectK:
		n := w(o.In[0])
		span := o.A.Span
		if span == 0 {
			span = n
		}
		if span < 1 || n%span != 0 || index(o.In[0]) {
			o.Why = whyShape
			return nil
		}
		// The group's width first: the input's buffer is the row's first.
		k := o.A.TopK
		groups := n / span
		stride := st(o.Out[0]) // select_k: a row's places, groups*k and a bin where the value carries one
		width := groups * k
		if o.Kind == dkRank {
			width = n
		}
		*r = compose.Row{Width: width, Rows: rows}
		b := inBuf(0)
		// key is element f of the group starting at base.
		key := func(base, f int) int { return r.LoadAt(b, r.UBin(compose.OpAdd, base, f)) }
		rowBase := func() int { return r.UBin(compose.OpMul, r.RowIdx(), u(st(o.In[0]))) }
		if o.Kind == dkRank {
			if w(o.Out[0]) != n {
				o.Why = whyShape
				return nil
			}
			col := r.Column()
			e := r.UBin(compose.OpRem, col, u(span))
			base := r.UBin(compose.OpAdd, rowBase(), r.UBin(compose.OpSub, col, e))
			rank := r.Rank(span, e, key(base, e), func(f int) int { return key(base, f) })
			r.StoreAt(rb.buf(compose.BufOutU32, gaOut0), rank, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(st(o.Out[0]))), col))
			break
		}
		// select_k: output slot j of group gi holds the element ranked j: a
		// fold over the group's elements keeping the one whose rank is j.
		if k < 1 || k > span || w(o.Out[0]) != groups*k || st(o.Out[1]) != stride {
			o.Why = whyShape
			return nil
		}
		col := r.Column()
		j := r.UBin(compose.OpRem, col, u(k))
		base := r.UBin(compose.OpAdd, rowBase(), r.UBin(compose.OpMul, r.UBin(compose.OpDiv, col, u(k)), u(span)))
		c := r.Loop(span)
		f := r.Iter(c)
		idx := r.Fold(c, u(0))
		val := r.Fold(c, key(base, u(0)))
		kf := key(base, f)
		rank := r.Rank(span, f, kf, func(h int) int { return key(base, h) })
		// rank == j: neither below nor above it.
		keep := func(now, then int) int {
			return r.Select(r.Less(rank, j), now, r.Select(r.Less(j, rank), now, then))
		}
		r.SetFold(idx, keep(idx, f))
		r.SetFold(val, keep(val, kf))
		at := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(stride)), col)
		r.StoreAt(rb.buf(compose.BufOutU32, gaOut0), idx, at)
		r.StoreAt(rb.buf(compose.BufOut, gaOut1), val, at)
	case dkTake:
		// From a value (In[0], indices In[1]) or from the op's bound vector
		// (indices In[0]): one source row for every row.
		ixv := o.In[len(o.In)-1]
		out, iw := w(o.Out[0]), st(ixv)
		vec := len(o.In) == 1
		if vec && o.V == nil || !vec && (len(o.In) != 2 || index(o.In[0])) || !index(ixv) || iw < out {
			o.Why = whyShape
			return nil
		}
		*r = compose.Row{Width: out, Rows: rows}
		col := r.Column()
		ix := r.LoadAt(inBuf(len(o.In)-1), r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(iw)), col))
		var x int
		if vec {
			x = r.LoadAt(rb.buf(compose.BufVec, gaV), ix)
		} else {
			n := w(o.In[0])
			x = r.LoadAt(inBuf(0), r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(st(o.In[0]))), r.UBin(compose.OpMin, ix, u(n-1))))
		}
		r.StoreAt(rb.buf(compose.BufOut, gaOut0), x, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(st(o.Out[0]))), col))
	case dkConcat:
		// Each input's span of the output in order; a value listed twice is
		// one buffer read at two spans.
		out := w(o.Out[0])
		*r = compose.Row{Width: out, Rows: rows}
		col := r.Column()
		bufOf := map[devVal]int{}
		var off []int
		total := 0
		for j, v := range o.In {
			if index(v) {
				o.Why = whyShape
				return nil
			}
			if _, ok := bufOf[v]; !ok {
				if j > int(gaIn3-gaIn0) {
					o.Why = whyShape // a launch reads its inputs from its first four arguments
					return nil
				}
				bufOf[v] = rb.buf(compose.BufRow, gaIn0+genArg(j))
			}
			off = append(off, total)
			total += w(v)
		}
		if total != out {
			o.Why = whyShape
			return nil
		}
		load := func(j int) int {
			v := o.In[j]
			ix := r.UBin(compose.OpMin, r.UBin(compose.OpSub, col, u(off[j])), u(w(v)-1))
			return r.LoadAt(bufOf[v], r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(st(v))), ix))
		}
		x := load(len(o.In) - 1)
		for j := len(o.In) - 2; j >= 0; j-- {
			x = r.Select(r.Less(col, u(off[j+1])), load(j), x)
		}
		r.StoreAt(rb.buf(compose.BufOut, gaOut0), x, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(st(o.Out[0]))), col))
	case dkRelayout:
		// Between a row of the value's width a row (row-major) and blocks of
		// HD stream-major across the rows: block k of row r at
		// (k*rows + r)*HD, rows the variant's.
		n, b := w(o.In[0]), o.HD
		if b < 1 || n%b != 0 || w(o.Out[0]) != n || len(o.In) != 1 {
			o.Why = whyShape
			return nil
		}
		*r = compose.Row{Width: n, Rows: rows}
		col := r.Column()
		row := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(n)), col)
		blocked := r.UBin(compose.OpAdd, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.UBin(compose.OpDiv, col, u(b)), u(rows*b)),
			r.UBin(compose.OpMul, r.RowIdx(), u(b))), r.UBin(compose.OpRem, col, u(b)))
		from, to := blocked, row
		if o.N == 1 {
			from, to = row, blocked
		}
		r.StoreAt(rb.buf(compose.BufOut, gaOut0), r.LoadAt(inBuf(0), from), to)
	case dkTile:
		out, n := w(o.Out[0]), w(o.In[0])
		span := o.A.Span
		if span == 0 {
			span = n
		}
		if span < 1 || span > n || out%span != 0 {
			o.Why = whyShape
			return nil
		}
		*r = compose.Row{Width: out, Rows: rows}
		col := r.Column()
		x := r.LoadAt(inBuf(0), r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(st(o.In[0]))), r.UBin(compose.OpRem, col, u(span))))
		r.StoreAt(rb.buf(compose.BufOut, gaOut0), x, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), u(st(o.Out[0]))), col))
	default:
		return nil
	}
	k := g.rowKern(r)
	if k.k == nil {
		o.Why = whyNoKernel
		return nil
	}
	l := genLaunch{k: k.k, groups: rows, width: k.group, n: uint8(len(rb.args))}
	copy(l.args[:], rb.args)
	return []genLaunch{l}
}

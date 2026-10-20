package tier

import (
	"math"

	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// The n-gram embedding into a residual of h streams (graph.OpNGram): each
// stream of the projected key and of the residual RMSNormed against its own
// row of a weight, their scaled dot through the signed square root's
// sigmoid the stream's gate, the gate times the projected value, that normed
// and convolved over the history the block's recurrent tail keeps, and both
// added into every stream.

// genStreams forms the n-gram block's ops over rows rows.
func (g *devTier) genStreams(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	p := &bs.p
	d := p.NEmbd
	if d < 1 || len(o.In) == 0 {
		return nil
	}
	w := prog.vals[o.In[0]].Width
	switch o.Kind {
	case dkStreamNorm:
		if o.V == nil || w%d != 0 {
			return nil
		}
		return g.launches(g.streamNorm(prog, o, rows, w/d, d, float32(p.RMSEps)))
	case dkStreamDot:
		if len(o.In) != 2 || w%d != 0 || prog.vals[o.Out[0]].Width != w/d {
			return nil
		}
		h := w / d
		s := newSelRow(d, rows*h, 0)
		k, q := s.buf(compose.BufRow, gaIn0), s.buf(compose.BufRow, gaIn1)
		out := s.buf(compose.BufOut, gaOut0)
		sum := s.r.Reduce(compose.RedSum, s.r.Bin(compose.OpMul, s.r.Load(k), s.r.Load(q)))
		v := s.r.Act(kernels.ActSqrtSigmoid, s.r.Bin(compose.OpMul, sum, s.r.Const(float32(1/math.Sqrt(float64(d))))))
		s.r.StoreAt(out, v, s.r.RowIdx())
		return g.launches(s.rowBuild)
	case dkGatedVal:
		if len(o.In) != 2 || prog.vals[o.In[1]].Width != d {
			return nil
		}
		h := w
		s := newSelRow(d, rows*h, 0)
		gate, val := s.buf(compose.BufRow, gaIn0), s.buf(compose.BufRow, gaIn1)
		out := s.buf(compose.BufOut, gaOut0)
		rk := s.r.RowIdx()
		v := s.r.LoadAt(val, s.ub(compose.OpAdd, s.ub(compose.OpMul, s.ub(compose.OpDiv, rk, s.u(h)), s.u(d)), s.r.Column()))
		s.r.Store(out, s.r.Bin(compose.OpMul, s.r.LoadAt(gate, rk), v))
		return g.launches(s.rowBuild)
	case dkNGramConv:
		return g.ngramConv(o, prog, bs, rows)
	case dkNGramAdd:
		if len(o.In) != 3 || w%d != 0 {
			return nil
		}
		h := w / d
		s := newSelRow(d, h*rows, 0)
		x, gv, cv := s.buf(compose.BufRow, gaIn0), s.buf(compose.BufRow, gaIn1), s.buf(compose.BufRow, gaIn2)
		out := s.buf(compose.BufOut, gaOut0)
		kr := s.r.RowIdx()
		k, r := s.ub(compose.OpDiv, kr, s.u(rows)), s.ub(compose.OpRem, kr, s.u(rows))
		rk := s.ub(compose.OpAdd, s.ub(compose.OpMul, s.ub(compose.OpAdd, s.ub(compose.OpMul, r, s.u(h)), k), s.u(d)), s.r.Column())
		sum := s.r.Bin(compose.OpAdd, s.r.LoadAt(gv, rk), s.r.LoadAt(cv, rk))
		s.r.Store(out, s.r.Bin(compose.OpAdd, s.r.Load(x), sum))
		return g.launches(s.rowBuild)
	}
	return nil
}

// streamNorm is each of rows*h stream rows of d RMSNormed against its own
// row of the op's weight, row-major out: read stream-major (stream k of row
// r at (k*rows + r)*d) where the input is the residual.
func (g *devTier) streamNorm(prog *devProgram, o *devOp, rows, h, d int, eps float32) *rowBuild {
	s := newSelRow(d, rows*h, 0)
	x, wt := s.buf(compose.BufRow, gaIn0), s.buf(compose.BufVec, gaV)
	out := s.buf(compose.BufOut, gaOut0)
	rk, col := s.r.RowIdx(), s.r.Column()
	k := s.ub(compose.OpRem, rk, s.u(h))
	in := s.r.Load(x)
	if sl := prog.slots[o.In[0]]; (sl == slotX || sl == slotX2) && h > 1 {
		r := s.ub(compose.OpDiv, rk, s.u(h))
		in = s.r.LoadAt(x, s.ub(compose.OpAdd, s.ub(compose.OpMul, s.ub(compose.OpAdd, s.ub(compose.OpMul, k, s.u(rows)), r), s.u(d)), col))
	}
	ss := s.r.Reduce(compose.RedSum, s.r.Bin(compose.OpMul, in, in))
	mean := s.r.Bin(compose.OpAdd, s.r.Bin(compose.OpMul, ss, s.r.Const(1/float32(d))), s.r.Const(eps))
	scale := s.r.Bin(compose.OpDiv, s.r.Const(1), s.r.Add(compose.Node{Kind: compose.NUnary, Op: compose.OpSqrt, A: mean}))
	wv := s.r.LoadAt(wt, s.ub(compose.OpAdd, s.ub(compose.OpMul, k, s.u(d)), col))
	s.r.Store(out, s.r.Bin(compose.OpMul, s.r.Bin(compose.OpMul, in, scale), wv))
	return s.rowBuild
}

// ngramConv is the dilated convolution over a call's rows, a thread a
// channel: each row in order reads its value and, tap t of taps reading
// (taps-1-t)*dil positions back, the history -- the in slot's, or the
// previous row's when that was the same sequence's -- writes
// silu(sum_t w[t][c] * value), and the history moved on is written at the
// out slot. The history is the last (taps-1)*dil positions' values, oldest
// first, at the end of the recurrent tail.
func (g *devTier) ngramConv(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	p := &bs.p
	hd := prog.vals[o.In[0]].Width
	taps, dil, stride := o.A.Taps, o.A.Dil, p.Recurrent.StateLen
	hist := (taps - 1) * dil
	off := stride - hd*hist
	if o.V == nil || taps < 2 || dil < 1 || hd < 1 || off < 0 || g.genRag {
		return nil
	}
	s := newSelRow(hd, 1, 0)
	st, cur, wt, desc := s.buf(compose.BufRow, gaStateIn), s.buf(compose.BufRow, gaIn0), s.buf(compose.BufVec, gaV), s.buf(compose.BufU32, gaRecDesc)
	out, sout := s.buf(compose.BufOut, gaOut0), s.buf(compose.BufOut, gaStateOut)
	R := &s.r
	u, ub := s.u, s.ub
	c := R.Column()
	w := make([]int, taps)
	for t := range w {
		w[t] = R.LoadAt(wt, ub(compose.OpAdd, c, u(t*hd)))
	}
	count := R.LoadAt(desc, u(0))
	in, outS := R.LoadAt(desc, u(1)), R.LoadAt(desc, u(2))
	lp := R.Loop(rows)
	e := R.Iter(lp)
	prevJ := R.Fold(lp, u(0xFFFFFFFF))
	h := make([]int, hist)
	for i := range h {
		h[i] = R.Fold(lp, R.Const(0))
	}
	j := u(0)
	valid := R.Less(e, count)
	same := R.Less(ub(compose.OpXor, j, prevJ), u(1))
	base := ub(compose.OpAdd, ub(compose.OpMul, in, u(stride)), u(off))
	cur0 := make([]int, hist)
	for i := range cur0 {
		ld := R.LoadAt(st, ub(compose.OpAdd, base, ub(compose.OpAdd, u(i*hd), c)))
		cur0[i] = R.Select(same, h[i], ld)
	}
	rowAt := ub(compose.OpAdd, ub(compose.OpMul, ub(compose.OpMin, e, u(rows-1)), u(hd)), c)
	cv := R.LoadAt(cur, rowAt)
	acc := R.Const(0)
	for t := 0; t < taps; t++ {
		v := cv
		if back := (taps - 1 - t) * dil; back > 0 {
			v = cur0[hist-back]
		}
		acc = s.fma(w[t], v, acc)
	}
	y := R.Bin(compose.OpDiv, acc, R.Bin(compose.OpAdd, R.Const(1),
		R.Add(compose.Node{Kind: compose.NUnary, Op: compose.OpExp, A: R.Bin(compose.OpSub, R.Const(0), acc)})))
	R.StoreAt(out, y, rowAt)
	ob := ub(compose.OpAdd, ub(compose.OpMul, outS, u(stride)), u(off))
	for i := range h {
		nv := cv
		if i < hist-1 {
			nv = cur0[i+1]
		}
		next := R.Select(valid, nv, cur0[i])
		R.StoreAt(sout, next, ub(compose.OpAdd, ob, ub(compose.OpAdd, u(i*hd), c)))
		R.SetFold(h[i], next)
	}
	R.SetFold(prevJ, j)
	return g.launches(s.rowBuild)
}

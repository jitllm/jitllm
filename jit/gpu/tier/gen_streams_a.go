package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// Stream ops as composed row groups: a residual that is several streams
// (stream-major: stream k of row r at (k*rows + r)*d) is read and written by
// ops over a row of d, one group a (stream, row). Two mixing mechanisms have
// their ops here, both stated by structure and not by name: a router over the
// active stream whose coefficients predict and correct every stream, and a
// gated mean over the streams with a placement of the sublayer's output into
// each. The tables the router and the predictions read are the op's bound
// vector (devOp.V); the stream count, the activation's input scale and the
// placement's output scale are the op's attributes and values.

// saLaunch is row group rb as one generated launch, nil where the backend
// refused its kernel.
func (g *devTier) saLaunch(rb *rowBuild) []genLaunch {
	k := g.rowKern(&rb.r)
	if k.k == nil {
		return nil
	}
	l := genLaunch{k: k.k, groups: rb.r.Rows, width: k.group, n: uint8(len(rb.args))}
	copy(l.args[:], rb.args)
	return []genLaunch{l}
}

// saStreamAt is the row-group nodes of the (stream, row) a group of a
// stream-major residual holds: the stream k and the row r its index names.
func saStreamAt(r *compose.Row, rows int) (k, row int) {
	return r.UBin(compose.OpDiv, r.RowIdx(), r.UConst(rows)), r.UBin(compose.OpRem, r.RowIdx(), r.UConst(rows))
}

// saDecline records why op o has no generated launches, for the executor's
// error and the program dump to name.
func (g *devTier) saDecline(o *devOp, why string, a ...any) []genLaunch {
	g.LastErr = fmt.Sprintf("op %s: %s", o.Kind, fmt.Sprintf(why, a...))
	return nil
}

// genStreamsA is a stream op as a composed row group, nil where o is not one
// of them or cannot be formed. Callers hold g.mu.
func (g *devTier) genStreamsA(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	switch o.Kind {
	case dkStreamRoute, dkStreamPredict, dkStreamCorrect, dkStreamScale, dkStreamFinish,
		dkStreamNorm, dkStreamGateMix, dkPlaceWeights, dkStreamPlace:
	default:
		return nil
	}
	if o.Kind == dkStreamPlace && len(o.In) != 3 {
		return nil // a banked placement (one input fewer) is another mechanism's
	}
	need := 1
	switch o.Kind {
	case dkStreamPredict, dkStreamFinish:
		need = 2
	case dkStreamCorrect, dkStreamPlace:
		need = 3
	case dkPlaceWeights:
		need = 0
	}
	if len(o.Out) != 1 || len(o.In) < need {
		return g.saDecline(o, "%d inputs and %d outputs", len(o.In), len(o.Out))
	}
	eps := float32(bs.p.RMSEps)
	if o.A.Eps != 0 {
		eps = float32(o.A.Eps)
	}
	vw := func(v devVal) int { return prog.vals[v].Width }
	rb := &rowBuild{}
	switch o.Kind {
	case dkStreamRoute:
		// m[r][j] = tanh((h_r . W_j)/d): one group a row over the streams, each
		// element a serial dot product over the row's d, W the bound router.
		s, d := vw(o.Out[0]), vw(o.In[0])
		if o.V == nil || len(o.In) != 1 || s < 2 || s > 256 || d < 1 {
			return g.saDecline(o, "a router of %d streams over %d with no table", s, d)
		}
		rb.r = compose.Row{Width: s, Rows: rows}
		h, w := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufVec, gaV)
		r := &rb.r
		ctr := r.LoopIdx()
		hat := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), r.UConst(d)), ctr)
		wat := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.Column(), r.UConst(d)), ctr)
		dot := r.Fold(r.LoadAt(h, hat), r.LoadAt(w, wat), d)
		x := r.Bin(compose.OpMul, dot, r.Const(1/float32(d)))
		// tanh as 1 - 2/(exp(2x) + 1).
		e := r.Add(compose.Node{Kind: compose.NUnary, Op: compose.OpExp, A: r.Bin(compose.OpMul, r.Const(2), x)})
		one := r.Const(1)
		rb.store(r.Bin(compose.OpSub, one, r.Bin(compose.OpDiv, r.Const(2), r.Bin(compose.OpAdd, e, one))), gaOut0)
	case dkStreamPredict:
		// out_k = x_k + sum_j c_kj x_j, c_kj = sum_i m_i T[i][k*s + j] (T the
		// bound table of s rows of s*s), a group a (stream, row).
		s := vw(o.In[1])
		d := saRowWidth(prog, o.Out[0], s)
		if o.V == nil || len(o.In) != 2 || s < 2 || d < 1 {
			return g.saDecline(o, "a prediction over %d streams of %d with no table", s, d)
		}
		rb.r = compose.Row{Width: d, Rows: s * rows}
		x, m, t := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufRow, gaIn1), rb.buf(compose.BufVec, gaV)
		r := &rb.r
		k, row := saStreamAt(r, rows)
		mb, kb := r.UBin(compose.OpMul, row, r.UConst(s)), r.UBin(compose.OpMul, k, r.UConst(s))
		acc := r.Load(x)
		for j := 0; j < s; j++ {
			c := r.Const(0)
			for i := 0; i < s; i++ {
				ti := r.LoadAt(t, r.UBin(compose.OpAdd, kb, r.UConst(i*s*s+j)))
				mi := r.LoadAt(m, r.UBin(compose.OpAdd, mb, r.UConst(i)))
				c = r.Add(compose.Node{Kind: compose.NBinary, Op: compose.OpFma, A: mi, B: ti, C: c})
			}
			at := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.UBin(compose.OpAdd, r.UConst(j*rows), row), r.UConst(d)), r.Column())
			acc = r.Add(compose.Node{Kind: compose.NBinary, Op: compose.OpFma, A: c, B: r.LoadAt(x, at), C: acc})
		}
		rb.store(acc, gaOut0)
	case dkStreamCorrect:
		// out_k = pred_k + (1 + sum_i m_i T[i][k]) * (o - pred_0): o the block's
		// output row, T the bound table of s rows of s.
		s := vw(o.In[2])
		d := vw(o.In[1])
		if o.V == nil || len(o.In) != 3 || s < 2 || d < 1 || vw(o.Out[0]) != s*d {
			return g.saDecline(o, "a correction over %d streams of %d with no table", s, d)
		}
		rb.r = compose.Row{Width: d, Rows: s * rows}
		pred, ob, m, t := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufRow, gaIn1), rb.buf(compose.BufRow, gaIn2), rb.buf(compose.BufVec, gaV)
		r := &rb.r
		k, row := saStreamAt(r, rows)
		c := r.Const(0)
		for i := 0; i < s; i++ {
			ti := r.LoadAt(t, r.UBin(compose.OpAdd, k, r.UConst(i*s)))
			mi := r.LoadAt(m, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, row, r.UConst(s)), r.UConst(i)))
			c = r.Add(compose.Node{Kind: compose.NBinary, Op: compose.OpFma, A: mi, B: ti, C: c})
		}
		c = r.Bin(compose.OpAdd, c, r.Const(1))
		rem := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, row, r.UConst(d)), r.Column())
		in := r.Bin(compose.OpSub, r.LoadAt(ob, rem), r.LoadAt(pred, rem))
		rb.store(r.Add(compose.Node{Kind: compose.NBinary, Op: compose.OpFma, A: c, B: in, C: r.Load(pred)}), gaOut0)
	case dkStreamScale:
		// The active stream's row times the bound per-channel scale.
		d := vw(o.Out[0])
		if o.V == nil || len(o.In) != 1 || d < 1 {
			return g.saDecline(o, "a scale of %d with no table", d)
		}
		rb.r = compose.Row{Width: d, Rows: rows}
		rb.store(rb.bin(compose.OpMul, rb.load(compose.BufRow, gaIn0), rb.load(compose.BufVec, gaV)), gaOut0)
	case dkStreamFinish:
		// out = src + (stream > 0 ? p : 0): the row p added to every stream
		// but the first.
		d := vw(o.In[1])
		if len(o.In) != 2 || d < 1 || vw(o.Out[0])%d != 0 || vw(o.Out[0]) < 2*d {
			return g.saDecline(o, "a finish of %d over %d", vw(o.Out[0]), d)
		}
		s := vw(o.Out[0]) / d
		rb.r = compose.Row{Width: d, Rows: s * rows}
		src, p := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufRow, gaIn1)
		r := &rb.r
		_, row := saStreamAt(r, rows)
		pat := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, row, r.UConst(d)), r.Column())
		rest := r.Less(r.UConst(rows-1), r.RowIdx())
		rb.store(r.Bin(compose.OpAdd, r.Load(src), r.Select(rest, r.LoadAt(p, pat), r.Const(0))), gaOut0)
	case dkStreamNorm:
		// Each stream of each row normed against its own row of the bound
		// weight: a group a (row, stream), its input stream-major (the
		// residual's) or each row's streams together (ByRow), its output
		// row-major.
		h := o.A.Streams
		if h < 1 || o.V == nil || len(o.In) != 1 {
			return g.saDecline(o, "a stream norm over %d streams with no weight", h)
		}
		d := saRowWidth(prog, o.Out[0], h)
		if d < 1 {
			return g.saDecline(o, "a stream norm of width %d over %d streams", vw(o.Out[0]), h)
		}
		rb.r = compose.Row{Width: d, Rows: rows * h}
		in, w := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufVec, gaV)
		r := &rb.r
		k := r.UBin(compose.OpRem, r.RowIdx(), r.UConst(h))
		var x int
		if o.ByRow {
			x = r.Load(in)
		} else {
			rr := r.UBin(compose.OpDiv, r.RowIdx(), r.UConst(h))
			at := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, k, r.UConst(rows)), rr), r.UConst(d)), r.Column())
			x = r.LoadAt(in, at)
		}
		wt := r.LoadAt(w, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, k, r.UConst(d)), r.Column()))
		rb.store(r.Bin(compose.OpMul, r.Bin(compose.OpMul, x, r.RMSScale(x, d, eps)), wt), gaOut0)
	case dkStreamGateMix:
		// The mean over the streams of sigmoid(gate) * normed; a mix with no
		// gate input is the plain mean.
		h := o.A.Streams
		d := vw(o.Out[0])
		if h < 1 || d < 1 || len(o.In) < 1 || len(o.In) > 2 {
			return g.saDecline(o, "a mix over %d streams of %d", h, d)
		}
		rb.r = compose.Row{Width: d, Rows: rows}
		xn := rb.buf(compose.BufRow, gaIn0)
		gate := -1
		if len(o.In) == 2 {
			gate = rb.buf(compose.BufRow, gaIn1)
		}
		r := &rb.r
		acc := r.Const(0)
		for k := 0; k < h; k++ {
			at := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), r.UConst(h*d)), r.UBin(compose.OpAdd, r.Column(), r.UConst(k*d)))
			x := r.LoadAt(xn, at)
			if gate >= 0 {
				x = r.Bin(compose.OpMul, r.Act(kernels.ActSigmoid, r.LoadAt(gate, at)), x)
			}
			acc = r.Bin(compose.OpAdd, acc, x)
		}
		rb.store(r.Bin(compose.OpMul, acc, r.Const(1/float32(h))), gaOut0)
	case dkPlaceWeights:
		// A placement's weights from its projection: OutScale * sigmoid(x *
		// Alpha); with no projection, the scale itself.
		h := vw(o.Out[0])
		if h < 1 || o.OutScale == 0 || len(o.In) > 1 {
			return g.saDecline(o, "placement weights of %d scaled by %g", h, o.OutScale)
		}
		rb.r = compose.Row{Width: h, Rows: rows}
		r := &rb.r
		w := r.Const(float32(o.OutScale))
		if len(o.In) == 1 {
			x := rb.load(compose.BufRow, gaIn0)
			if o.Alpha != 0 && o.Alpha != 1 {
				x = r.Bin(compose.OpMul, x, r.Const(float32(o.Alpha)))
			}
			w = r.Act(kernels.ActSigmoid, x)
			if o.OutScale != 1 {
				w = r.Bin(compose.OpMul, r.Const(float32(o.OutScale)), w)
			}
		}
		rb.store(w, gaOut0)
	case dkStreamPlace:
		// A sublayer's output added into every stream at its weight:
		// out[k][r] = x[k][r] + w[r][k] * y[r], one weight a (row, stream).
		// A placement whose third input is not one weight a stream is another
		// mechanism's.
		h := vw(o.In[2])
		d := vw(o.In[1])
		if h < 1 || d < 1 || o.A.Streams != h || vw(o.Out[0]) != h*d {
			return nil
		}
		rb.r = compose.Row{Width: d, Rows: h * rows}
		x, y, w := rb.buf(compose.BufRow, gaIn0), rb.buf(compose.BufRow, gaIn1), rb.buf(compose.BufRow, gaIn2)
		r := &rb.r
		k, row := saStreamAt(r, rows)
		wt := r.LoadAt(w, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, row, r.UConst(h)), k))
		yt := r.LoadAt(y, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, row, r.UConst(d)), r.Column()))
		rb.store(r.Add(compose.Node{Kind: compose.NBinary, Op: compose.OpFma, A: wt, B: yt, C: r.Load(x)}), gaOut0)
	}
	return g.saLaunch(rb)
}

// saRowWidth is the width of one stream's row in value v of s streams: its
// width over the streams where it holds them all, else its own.
func saRowWidth(prog *devProgram, v devVal, s int) int {
	w := prog.vals[v].Width
	if s > 0 && w%s == 0 && w/s > 0 {
		return w / s
	}
	return w
}

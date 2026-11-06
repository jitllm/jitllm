package convert

import (
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// stateAltUp states Gemma 3n's AltUp and LAuReL as the residual's streams
// (transformers' Gemma3nTextModel and Gemma3nTextDecoderLayer): s streams
// of n_embd, stream 0 the active one.
//
//	x_k    = mag(E_k x_0, x_0)                          the input rows' streams
//	w      = tanh(R . rmsnorm(x_0) / n_embd)            a block's router
//	pred_k = x_k + sum_j c_kj x_j,  c = C_pred w         before the attention
//	a      = (pred_0 + attn + n + norm(U D n)) / sqrt(2), n = attn_norm(pred_0)
//	o      = a + ffn                                    after the FFN
//	x_k    = pred_k + ((C_corr w') + 1)_k (o - pred_0)  w' the router over o
//	x_k   += norm(P (gelu(G (x_0 * scale)) * ple))      for k >= 1
//	y      = (x_0 + sum_k mag(U_k x_k, x_0)) / s        the head's row
//
// mag(p, r) = p * sqrt(mean(r^2)) / sqrt(max(mean(p^2), 1e-5)).
func stateAltUp(c *jlm.Config) {
	n := int(c.PredStreams)
	st := jlm.Streams{N: uint8(n), Expand: make([]*jlm.Program, n-1), Blocks: make([]jlm.BlockStreams, c.NLayer)}
	for k := 1; k < n; k++ {
		st.Expand[k-1] = altExpand(c, k)
	}
	for li := range st.Blocks {
		st.Blocks[li] = jlm.BlockStreams{Pre: altPredict(c), Attn: jlm.SubStreams{Join: altLaurel(c)}, FFN: altCorrect(c, li)}
	}
	st.Head = altCollapse(c)
	c.Streams = st
}

// altProgram builds a program over nin inputs.
type altProgram struct {
	p *jlm.Program
	d int32
	c *jlm.Config
}

func newAltProgram(c *jlm.Config, nin int) *altProgram {
	return &altProgram{p: &jlm.Program{NIn: uint16(nin)}, d: int32(c.NEmbd), c: c}
}

func (a *altProgram) op(o jlm.TowerOp) uint16 {
	a.p.Ops = append(a.p.Ops, o)
	return uint16(int(a.p.NIn) + len(a.p.Ops) - 1)
}

func (a *altProgram) expr(w int32, f func(e *jlm.ExprBuilder), in ...uint16) uint16 {
	var e jlm.ExprBuilder
	f(&e)
	return a.op(jlm.TowerOp{Code: jlm.TOpStated, In: in, Width: w, Expr: &jlm.Expr{Terms: e.T}})
}

func (a *altProgram) mul(w int32, x, y uint16) uint16 {
	return a.expr(w, func(e *jlm.ExprBuilder) { e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Operand(1)) }, x, y)
}

func (a *altProgram) scaled(w int32, k float32, x uint16) uint16 {
	return a.expr(w, func(e *jlm.ExprBuilder) { e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Num(k)) }, x)
}

func (a *altProgram) add(x, y uint16) uint16 {
	return a.op(jlm.TowerOp{Code: jlm.TOpAdd, In: []uint16{x, y}})
}

func (a *altProgram) slice(x uint16, off, w int32) uint16 {
	return a.op(jlm.TowerOp{Code: jlm.TOpSlice, In: []uint16{x}, Offset: off, Width: w})
}

func (a *altProgram) linear(x uint16, r jlm.Role) uint16 {
	return a.op(jlm.TowerOp{Code: jlm.TOpLinear, In: []uint16{x}, Bind: []jlm.Role{r}})
}

// norm is x RMS-normed and scaled by r.
func (a *altProgram) norm(x uint16, r jlm.Role) uint16 {
	v := a.op(jlm.TowerOp{Code: jlm.TOpRMSNorm, In: []uint16{x}, Span: a.d, Eps: a.c.RMSEps})
	return a.op(jlm.TowerOp{Code: jlm.TOpScale, In: []uint16{v}, Bind: []jlm.Role{r}})
}

// mag is p matched to the magnitude of r: p * sqrt(mean(r^2)) /
// sqrt(max(mean(p^2), 1e-5)).
func (a *altProgram) mag(p, r uint16) uint16 {
	inv := float32(1 / float64(a.d))
	sr := a.op(jlm.TowerOp{Code: jlm.TOpSum, In: []uint16{a.mul(a.d, r, r)}})
	sp := a.op(jlm.TowerOp{Code: jlm.TOpSum, In: []uint16{a.mul(a.d, p, p)}})
	k := a.expr(1, func(e *jlm.ExprBuilder) {
		e.Fn(jlm.FnDiv, e.Fn(jlm.FnSqrt, e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Num(inv))),
			e.Fn(jlm.FnSqrt, e.Fn(jlm.FnMax, e.Fn(jlm.FnMul, jlm.Operand(1), jlm.Num(inv)), jlm.Num(1e-5))))
	}, sr, sp)
	return a.mul(a.d, k, p)
}

// router is the router's weight for each stream over the row x.
func (a *altProgram) router(x uint16) uint16 {
	h := a.scaled(a.d, float32(1/float64(a.d)), a.norm(x, jlm.RoleStreamRouterNorm))
	r := a.linear(h, jlm.RoleStreamRouter)
	return a.expr(int32(a.c.PredStreams), func(e *jlm.ExprBuilder) { e.Fn(jlm.FnTanh, jlm.Operand(0)) }, r)
}

// altExpand is stream k of the input rows from stream 0.
func altExpand(c *jlm.Config, k int) *jlm.Program {
	a := newAltProgram(c, 1)
	e := a.linear(0, jlm.RoleStreamExpand)
	a.p.Out = a.mag(a.slice(e, int32(k-1)*a.d, a.d), 0)
	return a.p
}

// altPredict is a block's predictions: the read over every stream is the
// coefficients, and each stream's write its prediction from every stream.
func altPredict(c *jlm.Config) jlm.SubStreams {
	n := int(c.PredStreams)
	a := newAltProgram(c, n)
	a.p.Out = a.linear(a.router(0), jlm.RoleStreamPredictCoef)
	u := jlm.SubStreams{Read: a.p}
	for k := range n {
		u.From = append(u.From, int8(k))
		u.Exports = append(u.Exports, uint16(k))
	}
	for k := range n {
		w := newAltProgram(c, 2+n)
		acc := uint16(0)
		for j := range n {
			cj := w.slice(1, int32(k*n+j), 1)
			acc = w.add(acc, w.mul(w.d, cj, uint16(2+j)))
		}
		w.p.Out = acc
		u.Writes = append(u.Writes, w.p)
	}
	return u
}

// altLaurel is the attention's join over the running residual and its
// output: the two and the side branch over the attention's normed input,
// over sqrt(2).
func altLaurel(c *jlm.Config) *jlm.Program {
	a := newAltProgram(c, 2)
	n := a.norm(0, jlm.RoleAttnNorm)
	side := a.add(n, a.norm(a.linear(a.linear(n, jlm.RoleLowRankIn), jlm.RoleLowRankOut), jlm.RoleLowRankNorm))
	a.p.Out = a.scaled(a.d, float32(1/math.Sqrt2), a.add(a.add(0, 1), side))
	return a.p
}

// altCorrect is a block's FFN over the streams: its read the running
// residual, the prediction of the active stream and the row's per-layer
// inputs, exported; what its writes share the router's corrections, the
// innovation and the per-layer input gated from the corrected active
// stream; each write the stream corrected, the per-layer input added past
// the active one.
func altCorrect(c *jlm.Config, li int) jlm.SubStreams {
	n, p := int(c.PredStreams), int32(c.PLEDim)
	u := jlm.SubStreams{From: []int8{jlm.StreamRun, jlm.StreamIn, jlm.StreamSide}, Read: &jlm.Program{NIn: 3, Out: 0},
		Exports: []uint16{0, 1, 2}}
	// The shared program's inputs: the FFN's output, the running residual,
	// the active stream's prediction, the per-layer inputs.
	a := newAltProgram(c, 4)
	o := a.add(1, 0)
	cc := a.expr(int32(n), func(e *jlm.ExprBuilder) { e.Fn(jlm.FnAdd, jlm.Operand(0), jlm.Num(1)) },
		a.linear(a.router(o), jlm.RoleStreamCorrectCoef))
	innov := a.expr(a.d, func(e *jlm.ExprBuilder) { e.Fn(jlm.FnSub, jlm.Operand(0), jlm.Operand(1)) }, o, 2)
	x0 := a.add(2, a.mul(a.d, a.slice(cc, 0, 1), innov))
	g := a.op(jlm.TowerOp{Code: jlm.TOpScale, In: []uint16{x0}, Bind: []jlm.Role{jlm.RoleStreamCorrectScale}})
	gate := a.linear(g, jlm.RolePLEGate)
	in := a.slice(3, int32(li)*p, p)
	h := a.expr(p, func(e *jlm.ExprBuilder) { e.Fn(jlm.FnMul, e.GELUTanh(jlm.Operand(0)), jlm.Operand(1)) }, gate, in)
	ple := a.norm(a.linear(h, jlm.RolePLEProj), jlm.RolePLEPostNorm)
	a.p.Out = a.op(jlm.TowerOp{Code: jlm.TOpConcat, In: []uint16{cc, innov, ple}})
	u.Post = a.p
	// A write's inputs: the stream's prediction, the FFN's output, the three
	// exports, the shared value.
	for k := range n {
		w := newAltProgram(c, 6)
		x := w.add(0, w.mul(w.d, w.slice(5, int32(k), 1), w.slice(5, int32(n), w.d)))
		if k > 0 {
			x = w.add(x, w.slice(5, int32(n)+w.d, w.d))
		}
		w.p.Out = x
		u.Writes = append(u.Writes, w.p)
	}
	return u
}

// altCollapse is the head's row: stream 0 and each other stream's
// unembedding matched to it, averaged.
func altCollapse(c *jlm.Config) jlm.SubStreams {
	n := int(c.PredStreams)
	a := newAltProgram(c, n)
	acc := uint16(0)
	for k := 1; k < n; k++ {
		acc = a.add(acc, a.mag(a.linear(uint16(k), jlm.RoleStreamCollapse1+jlm.Role(k-1)), 0))
	}
	a.p.Out = a.scaled(a.d, float32(1/float64(n)), acc)
	u := jlm.SubStreams{Read: a.p}
	for k := range n {
		u.From = append(u.From, int8(k))
	}
	return u
}

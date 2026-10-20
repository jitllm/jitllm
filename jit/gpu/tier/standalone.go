package tier

import (
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// MatVec is nn.Device's single matvec on the card.
func (g *devTier) MatVec(out []float32, t quant.Type, w []byte, x []float32, nrows, k int) bool {
	return g.matVec(0, out, t, w, x, nrows, k) == served
}

// mvLaunch is one composed launch of a standalone matvec: its kernel, grid
// and the call's buffers by argument.
type mvLaunch struct {
	k             backend.Kernel
	groups, width int
	args          [8]mvArg
	n             int
}

// mvArg names a standalone matvec's buffer.
type mvArg uint8

const (
	mvQS mvArg = iota
	mvD
	mvSC
	mvX
	mvA
	mvAX
	mvOut
	mvPart
)

// mvShape is a standalone matvec's formed launches' key.
type mvShape struct {
	q            kernels.Quant
	rows, k, spl int
	win          int
}

// mvPlan is a standalone matvec's launches, formed once a shape.
type mvPlan struct {
	ls [4]mvLaunch
	n  int
}

// mvCall is one matVec's device side, handed to matVecSession through the
// devTier rather than captured by a closure. g.mu is held across the call.
type mvCall struct {
	r   *resident
	x   []float32
	k   int
	f   *mvPlan
	out int
	err error
}

// matVec serves one matvec of a weight outside any block: the float row up,
// its int8 operand quantized on the card for a packed weight, the composed
// decode matvec at the weight's split and the split's planes summed, the
// row back.
func (g *devTier) matVec(sid uint64, out []float32, t quant.Type, w []byte, x []float32, nrows, k int) decline {
	q, ok := quantOf(t)
	if !ok || k%q.Elems() != 0 || k%32 != 0 || nrows <= 0 || len(w) == 0 {
		g.mu.Lock()
		g.NoKernel++
		g.mu.Unlock()
		return noKernel
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.mvPays(sid, len(w)) {
		return tooSmall
	}
	r := g.resident(-1, 0, q, w, nrows, k, nil)
	if r == nil || !r.ok {
		return noRoom
	}
	split := chooseSplit(nrows, k/32, g.kb.split)
	f := g.mvFormOf(mvShape{q: q, rows: nrows, k: k, spl: split, win: min(max(g.actWinFor(1), 32), k)})
	if f == nil {
		return failed
	}
	c := mvCall{r: r, x: x, k: k, out: nrows, f: f}
	if !g.sizeOut(nrows) || !g.sizeXF(k) || !g.sizeAct(k/4, 3*k/32) || !g.sizeMVPart(nrows*split) {
		return noRoom
	}
	g.mvc = c
	if g.mvcFn == nil {
		g.mvcFn = g.matVecSession
	}
	g.dev.Session(g.mvcFn)
	err := g.mvc.err
	g.mvc = mvCall{}
	if err != nil {
		return failed
	}
	copy(out[:nrows], g.hostOut[:nrows])
	g.Served++
	return served
}

// matVecSession is matVec's Session: the float row up, the launches, the
// row back.
func (g *devTier) matVecSession(sess backend.Session) {
	c := &g.mvc
	if c.err = sess.Write(g.xfBuf, f32b(c.x[:c.k])); c.err != nil {
		return
	}
	lc := launcher{to: &g.mvTo}
	g.mvTo.s = sess
	defer func() { g.mvTo.s = nil }()
	for _, l := range c.f.ls[:c.f.n] {
		var bufs [8]backend.Buf
		for i, a := range l.args[:l.n] {
			bufs[i] = [...]backend.Buf{c.r.qs, c.r.d, c.r.sc, g.xfBuf, g.aBuf, g.axBuf, g.outBuf, g.mvPart}[a]
		}
		if c.err = lc.launch(l.k, l.groups, l.width, bufs[:l.n]...); c.err != nil {
			return
		}
	}
	c.err = sess.Read(g.outBuf, g.hostRaw[:c.out*4])
}

// mvFormOf is shape s's launches, formed on first use and kept, nil where
// the composer has no form for them. Callers hold g.mu.
func (g *devTier) mvFormOf(s mvShape) *mvPlan {
	if f, ok := g.mvForms[s]; ok {
		return f
	}
	f := g.formMV(s)
	if g.mvForms == nil {
		g.mvForms = map[mvShape]*mvPlan{}
	}
	g.mvForms[s] = f
	return f
}

// formMV composes shape s: for a packed weight the float row's int8 operand,
// then the decode matvec at s's split and its planes' sum.
func (g *devTier) formMV(s mvShape) *mvPlan {
	q, nrows, k := s.q, s.rows, s.k
	float := kernels.IsFloat(q)
	f := &mvPlan{}
	add := func(rb *compose.Row, args []mvArg, fc compose.Facts) bool {
		fc.Target = g.composeTarget()
		ps, err := rb.Plan(fc)
		if err != nil {
			g.LastErr = err.Error()
			return false
		}
		for _, p := range ps {
			sc := p.S
			kk, _ := g.genKern(append([]uint32{encRow}, p.G.Encode(sc)...), func() (*ir.Kernel, error) { return p.G.Emit(sc) })
			if kk == nil || f.n >= len(f.ls) || len(p.Bufs) > len(f.ls[0].args) {
				return false
			}
			l := mvLaunch{k: kk, groups: (p.Threads + sc.Group - 1) / sc.Group, width: sc.Group, n: len(p.Bufs)}
			for i, b := range p.Bufs {
				l.args[i] = mvPart
				if b >= 0 {
					l.args[i] = args[b]
				}
			}
			f.ls[f.n] = l
			f.n++
		}
		return true
	}
	if !float {
		qr := &compose.Row{Width: k / 4, Rows: 1}
		qr.Quantize8(qr.Buf(compose.BufRow), qr.Buf(compose.BufOutU32), qr.Buf(compose.BufOut), s.win)
		if !add(qr, []mvArg{mvX, mvA, mvAX}, compose.Facts{}) {
			return nil
		}
	}
	mr := &compose.Row{Width: nrows, Rows: 1}
	args := []mvArg{mvQS}
	wt := compose.MatVecWeight{T: q, K: k, Rows: nrows, QS: mr.Buf(compose.BufU32), D: -1, SC: -1}
	if !float {
		wt.D = mr.Buf(compose.BufU32)
		args = append(args, mvD)
	}
	if q.Info().PerSuper > 1 {
		wt.SC = mr.Buf(compose.BufU32)
		args = append(args, mvSC)
	}
	op := compose.MatVecOperand{Toks: 1}
	if float {
		op.X = mr.Buf(compose.BufRow)
		args = append(args, mvX)
	} else {
		op.A, op.AX = mr.Buf(compose.BufU32), mr.Buf(compose.BufRow)
		args = append(args, mvA, mvAX)
	}
	v, err := mr.MatVec(wt, op)
	if err != nil {
		g.LastErr = err.Error()
		return nil
	}
	mr.Stores = append(mr.Stores, compose.Store{Buf: mr.Buf(compose.BufOut), Val: v})
	args = append(args, mvOut)
	if !add(mr, args, compose.Facts{Step: compose.StepDecode, DecodeSplit: s.spl, DecodeRowt: 1}) {
		return nil
	}
	return f
}

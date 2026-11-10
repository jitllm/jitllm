package convert

import "github.com/jitllm/jitllm/format/jlm"

// stateMHC states DeepSeek V4's manifold-constrained hyper-connections
// (transformers' DeepseekV4HyperConnection and DeepseekV4HyperHead): HCMult
// streams, each the embedding at the input rows. Each sublayer reads the
// streams side by side, RMS-normed without a weight over the whole row, as
//
//	m    = fn xn                 (2+H)*H wide, H the streams
//	pre  = sigmoid(m[:H] s0 + base[:H]) + eps
//	post = 2 sigmoid(m[H:2H] s1 + base[H:2H])
//	comb = Sinkhorn(m[2H:] s2 + base[2H:])   H x H, each row softmaxed + eps,
//	                                         then the columns, and HCIters-1
//	                                         rounds of rows then columns, each
//	                                         divided by its sum + eps
//	in   = sum_k pre_k x_k
//	x_k  = post_k sublayer(in) + sum_j comb[j][k] x_j
//
// s the block's three stored scales, base its stored bias; the head reads
// sum_k pre_k x_k through a gate of its own (one scale, no post or comb).
func stateMHC(c *Config) {
	n := int(c.HCMult)
	s := jlm.Streams{N: uint8(n), Expand: make([]*jlm.Program, n-1), Blocks: make([]jlm.BlockStreams, c.NLayer+c.NMTP)}
	for li := range s.Blocks {
		s.Blocks[li] = jlm.BlockStreams{
			Attn: mhcSub(c, jlm.RoleHCAttnFn, jlm.RoleHCAttnScale, jlm.RoleHCAttnBase),
			FFN:  mhcSub(c, jlm.RoleHCFFNFn, jlm.RoleHCFFNScale, jlm.RoleHCFFNBase),
		}
	}
	s.Head = mhcHead(c)
	c.Streams = s
}

// mhcProg builds one of the hyper-connection's programs over the n streams.
type mhcProg struct {
	p *jlm.Program
	n int
}

func newMHCProg(n int) *mhcProg { return &mhcProg{p: &jlm.Program{NIn: uint16(n)}, n: n} }

func (g *mhcProg) op(o jlm.TowerOp) uint16 {
	g.p.Ops = append(g.p.Ops, o)
	return uint16(g.n + len(g.p.Ops) - 1)
}

func (g *mhcProg) expr(w int32, f func(e *jlm.ExprBuilder), in ...uint16) uint16 {
	var e jlm.ExprBuilder
	f(&e)
	return g.op(jlm.TowerOp{Code: jlm.TOpStated, In: in, Width: w, Expr: &jlm.Expr{Terms: e.T}})
}

func (g *mhcProg) slice(in uint16, off, w int32) uint16 {
	return g.op(jlm.TowerOp{Code: jlm.TOpSlice, In: []uint16{in}, Offset: off, Width: w})
}

func (g *mhcProg) sum(in uint16) uint16 { return g.op(jlm.TowerOp{Code: jlm.TOpSum, In: []uint16{in}}) }

func (g *mhcProg) add(a, b uint16) uint16 {
	return g.op(jlm.TowerOp{Code: jlm.TOpAdd, In: []uint16{a, b}})
}

// mul is a times b, either one element wide.
func (g *mhcProg) mul(w int32, a, b uint16) uint16 {
	return g.expr(w, func(e *jlm.ExprBuilder) { e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Operand(1)) }, a, b)
}

// div is a / (b + eps), b one element wide or a's width.
func (g *mhcProg) div(w int32, a, b uint16, eps float32) uint16 {
	return g.expr(w, func(e *jlm.ExprBuilder) {
		e.Fn(jlm.FnDiv, jlm.Operand(0), e.Fn(jlm.FnAdd, jlm.Operand(1), jlm.Num(eps)))
	}, a, b)
}

// vector is the stored vector r, w wide, as a value: zeros shifted by it.
func (g *mhcProg) vector(of uint16, r jlm.Role, w int32) uint16 {
	z := g.expr(w, func(e *jlm.ExprBuilder) { e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Num(0)) }, g.slice(of, 0, w))
	return g.op(jlm.TowerOp{Code: jlm.TOpShift, In: []uint16{z}, Bind: []jlm.Role{r}})
}

// mix is the streams' normed row through fn, each of the segments (their
// widths ws) times its stored scale, plus the stored base.
func (g *mhcProg) mix(c *Config, fn, scale, base jlm.Role, ws ...int32) uint16 {
	streams := make([]uint16, g.n)
	for k := range streams {
		streams[k] = uint16(k)
	}
	cat := g.op(jlm.TowerOp{Code: jlm.TOpConcat, In: streams})
	xn := g.op(jlm.TowerOp{Code: jlm.TOpRMSNorm, In: []uint16{cat}, Eps: c.RMSEps})
	m := g.op(jlm.TowerOp{Code: jlm.TOpLinear, In: []uint16{xn}, Bind: []jlm.Role{fn}})
	sc := g.vector(m, scale, int32(len(ws)))
	var segs []uint16
	off := int32(0)
	for i, w := range ws {
		segs = append(segs, g.mul(w, g.slice(m, off, w), g.slice(sc, int32(i), 1)))
		off += w
	}
	all := segs[0]
	if len(segs) > 1 {
		all = g.op(jlm.TowerOp{Code: jlm.TOpConcat, In: segs})
	}
	return g.op(jlm.TowerOp{Code: jlm.TOpShift, In: []uint16{all}, Bind: []jlm.Role{base}})
}

// collapse is sum_k w_k x_k, w the n weights.
func (g *mhcProg) collapse(d int32, w uint16) uint16 {
	acc := g.mul(d, g.slice(w, 0, 1), 0)
	for k := 1; k < g.n; k++ {
		acc = g.add(acc, g.mul(d, g.slice(w, int32(k), 1), uint16(k)))
	}
	return acc
}

func mhcSub(c *Config, fn, scale, base jlm.Role) jlm.SubStreams {
	n, d, eps := int32(c.HCMult), int32(c.NEmbd), c.HCEps
	g := newMHCProg(int(n))
	m := g.mix(c, fn, scale, base, n, n, n*n)
	pre := g.expr(n, func(e *jlm.ExprBuilder) {
		e.Fn(jlm.FnAdd, e.Fn(jlm.FnSigmoid, jlm.Operand(0)), jlm.Num(eps))
	}, g.slice(m, 0, n))
	post := g.expr(n, func(e *jlm.ExprBuilder) {
		e.Fn(jlm.FnMul, e.Fn(jlm.FnSigmoid, jlm.Operand(0)), jlm.Num(2))
	}, g.slice(m, n, n))
	// comb's rows, each softmaxed plus eps.
	rows := make([]uint16, n)
	for i := range n {
		r := g.slice(m, 2*n+i*n, n)
		mx := g.op(jlm.TowerOp{Code: jlm.TOpMax, In: []uint16{r}})
		ex := g.expr(n, func(e *jlm.ExprBuilder) {
			e.Fn(jlm.FnExp, e.Fn(jlm.FnSub, jlm.Operand(0), jlm.Operand(1)))
		}, r, mx)
		rows[i] = g.expr(n, func(e *jlm.ExprBuilder) {
			e.Fn(jlm.FnAdd, e.Fn(jlm.FnDiv, jlm.Operand(0), jlm.Operand(1)), jlm.Num(eps))
		}, ex, g.sum(ex))
	}
	cols := func() {
		cs := rows[0]
		for i := int32(1); i < n; i++ {
			cs = g.add(cs, rows[i])
		}
		for i := range rows {
			rows[i] = g.div(n, rows[i], cs, eps)
		}
	}
	cols()
	for range int(c.HCIters) - 1 {
		for i := range rows {
			rows[i] = g.div(n, rows[i], g.sum(rows[i]), eps)
		}
		cols()
	}
	g.p.Out = g.collapse(d, pre)
	// Each stream's mix of the others: sum_j comb[j][k] x_j.
	exports := make([]uint16, 0, n+1)
	for k := range n {
		acc := g.mul(d, g.slice(rows[0], k, 1), 0)
		for j := int32(1); j < n; j++ {
			acc = g.add(acc, g.mul(d, g.slice(rows[j], k, 1), uint16(j)))
		}
		exports = append(exports, acc)
	}
	exports = append(exports, post)
	from := make([]int8, n)
	for k := range from {
		from[k] = int8(k)
	}
	u := jlm.SubStreams{From: from, Read: g.p, Exports: exports}
	// Write k reads the stream, the sublayer's output, then the exports.
	for k := range n {
		wp := &jlm.Program{NIn: uint16(2 + n + 1)}
		postK := uint16(2 + n)
		wp.Ops = append(wp.Ops, jlm.TowerOp{Code: jlm.TOpSlice, In: []uint16{postK}, Offset: k, Width: 1})
		var e jlm.ExprBuilder
		e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Operand(1))
		at := wp.NIn
		wp.Ops = append(wp.Ops, jlm.TowerOp{Code: jlm.TOpStated, In: []uint16{at, 1}, Width: d, Expr: &jlm.Expr{Terms: e.T}},
			jlm.TowerOp{Code: jlm.TOpAdd, In: []uint16{uint16(2 + k), at + 1}})
		wp.Out = at + 2
		u.Writes = append(u.Writes, wp)
	}
	return u
}

func mhcHead(c *Config) jlm.SubStreams {
	n, d, eps := int32(c.HCMult), int32(c.NEmbd), c.HCEps
	g := newMHCProg(int(n))
	m := g.mix(c, jlm.RoleHCHeadFn, jlm.RoleHCHeadScale, jlm.RoleHCHeadBase, n)
	pre := g.expr(n, func(e *jlm.ExprBuilder) {
		e.Fn(jlm.FnAdd, e.Fn(jlm.FnSigmoid, jlm.Operand(0)), jlm.Num(eps))
	}, m)
	g.p.Out = g.collapse(d, pre)
	from := make([]int8, n)
	for k := range from {
		from[k] = int8(k)
	}
	return jlm.SubStreams{From: from, Read: g.p}
}

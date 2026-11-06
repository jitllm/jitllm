package convert

import (
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// StateStreams states the residual's streams in c where the model carries
// several: Kimi-K3's bank of checkpoints, a stream for the running residual
// and one per block of AttnResBlock (modeling_kimi_linear.py's block
// attention residuals). Each block's sublayer reads the bank banked before
// it and the running residual, mixed by the sublayer's score vector; the
// first block of a group banks its input into the group's stream, and its
// attention's output is the running residual rather than added to it.
// stores says block li (-1 the dense tensors) stores role.
func StateStreams(c *jlm.Config, stores func(r jlm.Role, li int32) bool) {
	if c.PredStreams > 1 && c.NLayer > 0 {
		stateAltUp(c)
		c.PredStreams = 0
		return
	}
	if c.HCLowRank > 0 && c.HCMult > 1 && c.NLayer > 0 {
		stateHyper(c)
		return
	}
	if c.AttnResBlock == 0 || c.NLayer == 0 {
		return
	}
	n, blk := int(c.NLayer), int(c.AttnResBlock)
	final := (n + blk - 1) / blk
	s := jlm.Streams{N: uint8(1 + final), Blocks: make([]jlm.BlockStreams, n)}
	for li := range n {
		b := &s.Blocks[li]
		bank := (li + blk - 1) / blk
		checkpoint := li%blk == 0
		if checkpoint {
			b.Bank = uint8(1 + li/blk)
			b.Attn.Join = &jlm.Program{NIn: 2, Out: 1}
		}
		b.Attn = bankRead(c, b.Attn, bankStreams(bank), jlm.RoleAttnResScore, stores(jlm.RoleAttnResScore, int32(li)))
		ffn := bankStreams(li/blk + 1)
		if checkpoint {
			ffn[b.Bank-1] = jlm.StreamIn
		}
		b.FFN = bankRead(c, b.FFN, ffn, jlm.RoleFFNResScore, stores(jlm.RoleFFNResScore, int32(li)))
	}
	s.Head = bankRead(c, s.Head, bankStreams(final), jlm.RoleOutputResScore, stores(jlm.RoleOutputResScore, jlm.DenseBlock))
	c.Streams = s
}

// sourceStores says s holds role for block li (-1 the dense tensors).
func sourceStores(s *jlm.Source) func(jlm.Role, int32) bool {
	held := map[[2]int32]bool{}
	for _, t := range s.Tensors {
		held[[2]int32{int32(t.Role), t.Block}] = true
	}
	return func(r jlm.Role, li int32) bool { return held[[2]int32{int32(r), li}] }
}

// bankStreams is streams 1 to n, each read as itself.
func bankStreams(n int) []int8 {
	from := make([]int8, n)
	for k := range from {
		from[k] = int8(1 + k)
	}
	return from
}

// bankRead is u reading the bank from and the running residual through a
// mix scored by role, where the block stores it: each row RMS-normed with
// no weight, scored by the sum of role's vector times it, the scores'
// softmax weighting the rows. A block that stores no score reads the
// running residual.
func bankRead(c *jlm.Config, u jlm.SubStreams, from []int8, role jlm.Role, stored bool) jlm.SubStreams {
	if !stored {
		return u
	}
	u.From = append(from, jlm.StreamRun)
	ns := len(u.From)
	p := &jlm.Program{NIn: uint16(ns)}
	op := func(o jlm.TowerOp) uint16 {
		p.Ops = append(p.Ops, o)
		return uint16(ns + len(p.Ops) - 1)
	}
	scores := make([]uint16, ns)
	for j := range ns {
		nv := op(jlm.TowerOp{Code: jlm.TOpRMSNorm, In: []uint16{uint16(j)}, Eps: c.RMSEps})
		sv := op(jlm.TowerOp{Code: jlm.TOpScale, In: []uint16{nv}, Bind: []jlm.Role{role}})
		scores[j] = op(jlm.TowerOp{Code: jlm.TOpSum, In: []uint16{sv}})
	}
	sc := op(jlm.TowerOp{Code: jlm.TOpConcat, In: scores})
	mx := op(jlm.TowerOp{Code: jlm.TOpMax, In: []uint16{sc}})
	var e jlm.ExprBuilder
	e.Fn(jlm.FnExp, e.Fn(jlm.FnSub, jlm.Operand(0), jlm.Operand(1)))
	ex := op(jlm.TowerOp{Code: jlm.TOpStated, In: []uint16{sc, mx}, Width: int32(ns), Expr: &jlm.Expr{Terms: e.T}})
	sum := op(jlm.TowerOp{Code: jlm.TOpSum, In: []uint16{ex}})
	var d jlm.ExprBuilder
	d.Fn(jlm.FnDiv, jlm.Operand(0), jlm.Operand(1))
	pr := op(jlm.TowerOp{Code: jlm.TOpStated, In: []uint16{ex, sum}, Width: int32(ns), Expr: &jlm.Expr{Terms: d.T}})
	w := int32(c.NEmbd)
	var acc uint16
	for j := range ns {
		pj := op(jlm.TowerOp{Code: jlm.TOpSlice, In: []uint16{pr}, Offset: int32(j), Width: 1})
		var m jlm.ExprBuilder
		if j == 0 {
			m.Fn(jlm.FnMul, jlm.Operand(0), jlm.Operand(1))
			acc = op(jlm.TowerOp{Code: jlm.TOpStated, In: []uint16{pj, uint16(j)}, Width: w, Expr: &jlm.Expr{Terms: m.T}})
			continue
		}
		m.Fn(jlm.FnMul, jlm.Operand(0), jlm.Operand(1))
		part := op(jlm.TowerOp{Code: jlm.TOpStated, In: []uint16{pj, uint16(j)}, Width: w, Expr: &jlm.Expr{Terms: m.T}})
		acc = op(jlm.TowerOp{Code: jlm.TOpAdd, In: []uint16{acc, part}})
	}
	p.Out = acc
	u.Read = p
	return u
}

// stateHyper states Qwen3.8-Flash-Next's hyper-connections (transformers'
// Qwen4ExpForCausalLM): HCMult streams, the input rows' copies of the
// embedding; each sublayer reads, with xn the streams side by side each
// RMS-normed against its own row of the stored weight,
//
//	g   = sigmoid(U silu(D xn / S)) * xn
//	in  = the mean over the streams k of g_k
//	w   = 2 sigmoid(J xn / S)                  one weight a stream
//	x_k = x_k + w_k sublayer(in)
//
// and the head reads the mix through a gate of its own, with no placement.
func stateHyper(c *jlm.Config) {
	n := int(c.HCMult)
	s := jlm.Streams{N: uint8(n), Expand: make([]*jlm.Program, n-1), Blocks: make([]jlm.BlockStreams, c.NLayer+c.NMTP)}
	for li := range s.Blocks {
		s.Blocks[li] = jlm.BlockStreams{
			Attn: hyperSub(c, jlm.RoleHCAttnNorm, jlm.RoleHCAttnDown, jlm.RoleHCAttnUp, jlm.RoleHCAttnInject),
			FFN:  hyperSub(c, jlm.RoleHCFFNNorm, jlm.RoleHCFFNDown, jlm.RoleHCFFNUp, jlm.RoleHCFFNInject),
		}
	}
	s.Head = hyperSub(c, jlm.RoleHCHeadNorm, jlm.RoleHCHeadDown, jlm.RoleHCHeadUp, 0)
	if b := int(c.NGramBlock); b != 0 && b <= int(c.NLayer) {
		s.Blocks[b-1].Pre = ngramPre(c)
	}
	c.Streams = s
}

// hyperSub is a sublayer reading the streams through the gate the roles
// bind, and with an inject role placing its output into every stream.
func hyperSub(c *jlm.Config, norm, down, up, inject jlm.Role) jlm.SubStreams {
	n, d := int(c.HCMult), int32(c.NEmbd)
	sd, inv := int32(n)*d, 1/float32(n)
	p := &jlm.Program{NIn: uint16(n)}
	op := func(o jlm.TowerOp) uint16 {
		p.Ops = append(p.Ops, o)
		return uint16(n + len(p.Ops) - 1)
	}
	expr := func(w int32, f func(e *jlm.ExprBuilder), in ...uint16) uint16 {
		var e jlm.ExprBuilder
		f(&e)
		return op(jlm.TowerOp{Code: jlm.TOpStated, In: in, Width: w, Expr: &jlm.Expr{Terms: e.T}})
	}
	scaled := func(w int32, k float32, in uint16) uint16 {
		return expr(w, func(e *jlm.ExprBuilder) { e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Num(k)) }, in)
	}
	streams := make([]uint16, n)
	from := make([]int8, n)
	for k := range n {
		streams[k], from[k] = uint16(k), int8(k)
	}
	cat := op(jlm.TowerOp{Code: jlm.TOpConcat, In: streams})
	normed := op(jlm.TowerOp{Code: jlm.TOpRMSNorm, In: []uint16{cat}, Span: d, Eps: c.RMSEps})
	xn := op(jlm.TowerOp{Code: jlm.TOpScale, In: []uint16{normed}, Bind: []jlm.Role{norm}})
	lo := scaled(int32(c.HCLowRank), inv, op(jlm.TowerOp{Code: jlm.TOpLinear, In: []uint16{xn}, Bind: []jlm.Role{down}}))
	act := expr(int32(c.HCLowRank), func(e *jlm.ExprBuilder) {
		e.Fn(jlm.FnMul, jlm.Operand(0), e.Fn(jlm.FnSigmoid, jlm.Operand(0)))
	}, lo)
	u := op(jlm.TowerOp{Code: jlm.TOpLinear, In: []uint16{act}, Bind: []jlm.Role{up}})
	g := expr(sd, func(e *jlm.ExprBuilder) { e.Fn(jlm.FnMul, e.Fn(jlm.FnSigmoid, jlm.Operand(0)), jlm.Operand(1)) }, u, xn)
	acc := op(jlm.TowerOp{Code: jlm.TOpSlice, In: []uint16{g}, Width: d})
	for k := 1; k < n; k++ {
		gk := op(jlm.TowerOp{Code: jlm.TOpSlice, In: []uint16{g}, Offset: int32(k) * d, Width: d})
		acc = op(jlm.TowerOp{Code: jlm.TOpAdd, In: []uint16{acc, gk}})
	}
	mix := scaled(d, inv, acc)
	u2 := jlm.SubStreams{From: from, Read: p}
	if inject != 0 {
		wv := scaled(int32(n), inv, op(jlm.TowerOp{Code: jlm.TOpLinear, In: []uint16{xn}, Bind: []jlm.Role{inject}}))
		ws := expr(int32(n), func(e *jlm.ExprBuilder) { e.Fn(jlm.FnSigmoid, jlm.Operand(0)) }, wv)
		w := scaled(int32(n), 2, ws)
		u2.Exports = []uint16{w}
		for k := range n {
			wp := &jlm.Program{NIn: 3}
			wp.Ops = append(wp.Ops, jlm.TowerOp{Code: jlm.TOpSlice, In: []uint16{2}, Offset: int32(k), Width: 1})
			var e jlm.ExprBuilder
			e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Operand(1))
			wp.Ops = append(wp.Ops, jlm.TowerOp{Code: jlm.TOpStated, In: []uint16{3, 1}, Width: d, Expr: &jlm.Expr{Terms: e.T}},
				jlm.TowerOp{Code: jlm.TOpAdd, In: []uint16{0, 4}})
			wp.Out = 5
			u2.Writes = append(u2.Writes, wp)
		}
	}
	p.Out = mix
	return u2
}

// ngramPre states Qwen3.8-Flash-Next's n-gram embedding (transformers'
// Qwen4ExpTextPLELayer), added into every stream before its block's
// attention reads them: each row's n-gram table rows project to a key per
// stream and one value; each stream's normed key against its normed
// residual, scaled by 1/sqrt(n_embd), gates the value by the sigmoid of its
// signed square root, the magnitude floored at the reference's
// clamp_min(1e-6); the gated values and their normed depthwise convolution
// through silu, dilated by the n-gram, are added to the streams.
func ngramPre(c *jlm.Config) jlm.SubStreams {
	n, d := int(c.HCMult), int32(c.NEmbd)
	hd := int32(n) * d
	p := &jlm.Program{NIn: uint16(n)}
	op := func(o jlm.TowerOp) uint16 {
		p.Ops = append(p.Ops, o)
		return uint16(n + len(p.Ops) - 1)
	}
	expr := func(w int32, f func(e *jlm.ExprBuilder), in ...uint16) uint16 {
		var e jlm.ExprBuilder
		f(&e)
		return op(jlm.TowerOp{Code: jlm.TOpStated, In: in, Width: w, Expr: &jlm.Expr{Terms: e.T}})
	}
	norm := func(v uint16, r jlm.Role) uint16 {
		normed := op(jlm.TowerOp{Code: jlm.TOpRMSNorm, In: []uint16{v}, Span: d, Eps: c.RMSEps})
		return op(jlm.TowerOp{Code: jlm.TOpScale, In: []uint16{normed}, Bind: []jlm.Role{r}})
	}
	mul := func(e *jlm.ExprBuilder) { e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Operand(1)) }
	heads := int32((c.NGram - 1) * c.NGramHeadsPer)
	emb := op(jlm.TowerOp{Code: jlm.TOpNGramRows, Bind: []jlm.Role{jlm.RoleNGramTable}, Span: int32(c.NGram),
		Width: heads * int32(c.NGramRowDim)})
	kn := norm(op(jlm.TowerOp{Code: jlm.TOpLinear, In: []uint16{emb}, Bind: []jlm.Role{jlm.RoleNGramKey}}), jlm.RoleNGramNormKey)
	val := op(jlm.TowerOp{Code: jlm.TOpLinear, In: []uint16{emb}, Bind: []jlm.Role{jlm.RoleNGramValue}})
	streams := make([]uint16, n)
	from := make([]int8, n)
	for k := range n {
		streams[k], from[k] = uint16(k), int8(k)
	}
	qn := norm(op(jlm.TowerOp{Code: jlm.TOpConcat, In: streams}), jlm.RoleNGramNormQuery)
	dot := op(jlm.TowerOp{Code: jlm.TOpSum, In: []uint16{expr(hd, mul, kn, qn)}, Span: d})
	gate := expr(int32(n), func(e *jlm.ExprBuilder) {
		s := e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Num(float32(1/math.Sqrt(float64(d)))))
		mag := e.Fn(jlm.FnSqrt, e.Fn(jlm.FnMax, e.Fn(jlm.FnAbs, s), jlm.Num(1e-6)))
		sign := e.Fn(jlm.FnSub, e.Fn(jlm.FnLess, jlm.Num(0), s), e.Fn(jlm.FnLess, s, jlm.Num(0)))
		e.Fn(jlm.FnSigmoid, e.Fn(jlm.FnMul, sign, mag))
	}, dot)
	gv := make([]uint16, n)
	for k := range n {
		gv[k] = expr(d, mul, op(jlm.TowerOp{Code: jlm.TOpSlice, In: []uint16{gate}, Offset: int32(k), Width: 1}), val)
	}
	gvs := op(jlm.TowerOp{Code: jlm.TOpConcat, In: gv})
	conv := op(jlm.TowerOp{Code: jlm.TOpTailConv, In: []uint16{norm(gvs, jlm.RoleNGramNormConv)},
		Bind: []jlm.Role{jlm.RoleNGramConv}, Span: int32(c.NGramConv), Offset: int32(c.NGram), Width: hd})
	act := expr(hd, func(e *jlm.ExprBuilder) {
		e.Fn(jlm.FnMul, jlm.Operand(0), e.Fn(jlm.FnSigmoid, jlm.Operand(0)))
	}, conv)
	p.Out = op(jlm.TowerOp{Code: jlm.TOpAdd, In: []uint16{gvs, act}})
	u := jlm.SubStreams{From: from, Read: p}
	for k := range n {
		u.Writes = append(u.Writes, &jlm.Program{NIn: 2, Ops: []jlm.TowerOp{
			{Code: jlm.TOpSlice, In: []uint16{1}, Offset: int32(k) * d, Width: d},
			{Code: jlm.TOpAdd, In: []uint16{0, 2}},
		}, Out: 3})
	}
	return u
}

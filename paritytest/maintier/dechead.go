package maintier

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	kernels "github.com/jitllm/jitllm/paritytest/mainkernels"
)

// Clef's joint schema head on the device (nn.ClefHeadDevice): the request's
// final residual rows go up, the head runs as a program of kernels.DH*
// launches on the device that holds the model's output projection -- where
// the backbone's last rows came from -- and one logit per option comes back.
// The head's matrices are uploaded once per head (f32, as the model hands
// them) and charged against the device's budget; the kernels are compiled per
// request shape, before the submission, and kept.

// dhHead is a head's buffers on one device.
type dhHead struct {
	h     *nn.ClefHead
	bufs  map[*float32]backend.Buf // by the first element of the host slice
	bytes uint64
}

// dhState is the device's Clef state: the resident heads, the compiled
// kernels and the programs of the request shapes seen last.
type dhState struct {
	heads map[*nn.ClefHead]*dhHead
	kerns map[dhKey]backend.Kernel
	progs []*dhProg
}

type dhKey struct {
	name string
	a    [10]int
	f    [3]float32
	b    [2]bool
}

// ClefHead runs the head on the device holding the output projection.
func (g *GPU) ClefHead(h *nn.ClefHead, in *nn.ClefHeadInput, out []float32) bool {
	g.mu.Lock()
	d := g.head
	g.mu.Unlock()
	if d == nil {
		return false
	}
	return d.clefHead(h, in, out)
}

// ClefHead is the session's device's.
func (s *gpuSession) ClefHead(h *nn.ClefHead, in *nn.ClefHeadInput, out []float32) bool {
	defer s.leave(s.step())
	return s.g.ClefHead(h, in, out)
}

// dhUpload puts h's matrices and vectors on the device, once.
func (g *devTier) dhUpload(h *nn.ClefHead) (*dhHead, bool) {
	if g.dh.heads == nil {
		g.dh.heads = map[*nn.ClefHead]*dhHead{}
	}
	if r, ok := g.dh.heads[h]; ok {
		return r, true
	}
	var all [][]float32
	add := func(vs ...[]float32) {
		for _, v := range vs {
			if len(v) != 0 {
				all = append(all, v)
			}
		}
	}
	add(h.OutNorm, h.OutNormB, h.HidW, h.HidB, h.PMem, h.PQ, h.POQ, h.PG, h.POC, h.POL, h.SumW, h.SumB, h.FieldW, h.FieldB,
		h.OptW, h.OptB, h.Types, h.Sc, h.ScB, h.ScOut, h.ScOutB)
	for i := range h.Blocks {
		b := &h.Blocks[i]
		add(b.SelfW, b.SelfB, b.SQ, b.SK, b.SV, b.SO, b.SBQ, b.SBK, b.SBV, b.SBO, b.CrossW, b.CrossB, b.KVW, b.KVB,
			b.CQ, b.CK, b.CV, b.CO, b.CBQ, b.CBK, b.CBV, b.CBO, b.FFNW, b.FFNB, b.Up, b.BUp, b.Down, b.BDown)
	}
	var need uint64
	for _, v := range all {
		need += uint64(4 * len(v))
	}
	if !g.room(need) {
		g.LastErr = fmt.Sprintf("the decision head needs %d bytes, %d of %d used", need, g.used, g.limit)
		return nil, false
	}
	r := &dhHead{h: h, bufs: map[*float32]backend.Buf{}}
	for _, v := range all {
		if _, ok := r.bufs[&v[0]]; ok {
			continue
		}
		b, err := g.dev.Alloc(4 * len(v))
		if err == nil {
			err = b.Write(f32b(v))
		}
		if err != nil {
			if b != nil {
				b.Free()
			}
			r.free()
			g.LastErr = fmt.Sprintf("the decision head: %v", err)
			return nil, false
		}
		r.bufs[&v[0]] = b
		r.bytes += uint64(4 * len(v))
	}
	g.charge(r.bytes)
	g.dh.heads[h] = r
	return r, true
}

func (r *dhHead) free() {
	for _, b := range r.bufs {
		b.Free()
	}
	r.bufs = nil
}

// dhClose gives back every head and kernel. Callers hold g.mu.
func (g *devTier) dhClose() {
	for _, r := range g.dh.heads {
		g.refund(r.bytes)
		r.free()
	}
	for _, k := range g.dh.kerns {
		if k != nil {
			k.Close()
		}
	}
	for _, p := range g.dh.progs {
		p.free()
	}
	g.dh = dhState{}
}

// dhKern is the compiled kernel for key. Callers hold g.mu.
func (g *devTier) dhKern(key dhKey, build func() (*ir.Kernel, error)) (backend.Kernel, error) {
	if k, ok := g.dh.kerns[key]; ok {
		if k == nil {
			return nil, fmt.Errorf("%s does not compile here", key.name)
		}
		return k, nil
	}
	if g.dh.kerns == nil {
		g.dh.kerns = map[dhKey]backend.Kernel{}
	}
	ker, err := build()
	var k backend.Kernel
	if err == nil {
		k, err = g.dev.Compile(ker)
	}
	if err != nil {
		g.dh.kerns[key] = nil
		return nil, fmt.Errorf("%s: %w", key.name, err)
	}
	g.dh.kerns[key] = k
	return k, nil
}

// dhNewBuf is a buffer of n floats, poisoned (RULE 13). Callers hold g.mu.
func (g *devTier) dhNewBuf(n int) (backend.Buf, error) {
	n = max(n, 1)
	b, err := g.dev.Alloc(4 * n)
	if err != nil {
		return nil, err
	}
	poison := make([]float32, n)
	for i := range poison {
		poison[i] = float32(math.NaN())
	}
	if err := b.Write(f32b(poison)); err != nil {
		b.Free()
		return nil, err
	}
	return b, nil
}

// dhStep is one launch of the head's program.
type dhStep struct {
	k       backend.Kernel
	threads int
	bufs    []backend.Buf
}

// dhProg is one request shape's program: its launches, the buffers they
// run against (its own, so another shape's growth never moves them), and
// the inputs' constant parts. Kept per shape (dhProgs), so a warm request
// writes its inputs, launches and reads back, allocating nothing.
type dhProg struct {
	key                            dhShape
	prog                           []dhStep
	own                            []backend.Buf
	resid, pool, lex, owner, qtype backend.Buf
	zero, mask, logits             backend.Buf
	sel                            [3]backend.Buf
	selB                           [3][]byte
	zeroB                          []byte
	in                             *nn.ClefHeadInput
	out                            []float32
	err                            error
	run                            func(backend.Session)
}

// dhShape keys a program: the head and the request's rows, questions and
// options.
type dhShape struct {
	h         *nn.ClefHead
	n, nq, no int
}

// dhProgs is how many request shapes a device keeps programs for; the oldest
// goes first.
const dhProgs = 8

func (p *dhProg) free() {
	for _, b := range p.own {
		b.Free()
	}
	p.own = nil
}

// clefHead runs the request's program, building it on the first request of
// its shape, in one submission, and reads the logits back.
func (g *devTier) clefHead(h *nn.ClefHead, in *nn.ClefHeadInput, out []float32) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return false
	}
	hd, okUp := g.dhUpload(h)
	if !okUp {
		return false
	}
	n, nq, no := in.N, in.NQ, in.NO
	if len(in.Resid) != n*h.D || len(in.Pool) != (nq+1+no)*n || len(in.Lex) != no*h.D || len(out) < no ||
		len(in.Owner) != no || len(in.QType) != nq || len(in.Mask) != nq*no {
		g.LastErr = fmt.Sprintf("the decision head: an input of the wrong shape (%d rows, %d questions, %d options)", n, nq, no)
		return false
	}
	key := dhShape{h, n, nq, no}
	var pr *dhProg
	for i, p := range g.dh.progs {
		if p.key == key {
			pr = p
			// Most recent last.
			copy(g.dh.progs[i:], g.dh.progs[i+1:])
			g.dh.progs[len(g.dh.progs)-1] = p
			break
		}
	}
	if pr == nil {
		var err error
		if pr, err = g.dhBuild(h, hd, n, nq, no); err != nil {
			g.LastErr = "the decision head: " + err.Error()
			return false
		}
		if len(g.dh.progs) == dhProgs {
			g.dh.progs[0].free()
			g.dh.progs = append(g.dh.progs[:0], g.dh.progs[1:]...)
		}
		g.dh.progs = append(g.dh.progs, pr)
	}
	pr.in, pr.out, pr.err = in, out[:no], nil
	g.dev.Session(pr.run)
	pr.in, pr.out = nil, nil
	if pr.err != nil {
		g.LastErr = "the decision head: " + pr.err.Error()
		return false
	}
	g.ClefHeads++
	return true
}

// dhRun is a program's submission: the inputs up, the launches, the logits
// back.
func (p *dhProg) dhRun(s backend.Session) {
	in := p.in
	w := func(b backend.Buf, x []byte) {
		if p.err == nil {
			p.err = s.Write(b, x)
		}
	}
	w(p.resid, f32b(in.Resid))
	w(p.pool, f32b(in.Pool))
	w(p.lex, f32b(in.Lex))
	w(p.owner, u32b(in.Owner))
	w(p.qtype, u32b(in.QType))
	w(p.mask, f32b(in.Mask))
	w(p.zero, p.zeroB)
	for i := range p.sel {
		w(p.sel[i], p.selB[i])
	}
	for i := range p.prog {
		if p.err != nil {
			return
		}
		st := &p.prog[i]
		p.err = s.Launch(st.k, (st.threads+127)/128, 128, st.bufs...)
	}
	if p.err == nil {
		p.err = s.Read(p.logits, f32b(p.out))
	}
}

// dhBuild compiles a request shape's program. Callers hold g.mu.
func (g *devTier) dhBuild(h *nn.ClefHead, hd *dhHead, n, nq, no int) (*dhProg, error) {
	pr := &dhProg{key: dhShape{h, n, nq, no}}
	pr.run = pr.dhRun
	named := map[string]backend.Buf{}
	w := func(v []float32) backend.Buf { return hd.bufs[&v[0]] }
	D, W, H, F := h.D, h.W, h.Heads, h.FFN
	var prog []dhStep
	var err error
	launch := func(k backend.Kernel, threads int, bufs ...backend.Buf) {
		prog = append(prog, dhStep{k, threads, bufs})
	}
	buf := func(name string, floats int) backend.Buf {
		if err != nil {
			return nil
		}
		if b, ok := named[name]; ok {
			return b
		}
		var b backend.Buf
		if b, err = g.dhNewBuf(floats); err != nil {
			return nil
		}
		named[name] = b
		pr.own = append(pr.own, b)
		return b
	}
	kern := func(key dhKey, build func() (*ir.Kernel, error)) backend.Kernel {
		if err != nil {
			return nil
		}
		var k backend.Kernel
		k, err = g.dhKern(key, build)
		return k
	}
	// linear is dst = x W^T (+ b), rows of x of width k into width nOut.
	linear := func(dst, x backend.Buf, rows, k, nOut int, wt, bias []float32) {
		hasB := len(bias) != 0
		kk := kern(dhKey{name: "matabt", a: [10]int{1, rows, nOut, k, k, k, nOut, 0, 0, 0}, f: [3]float32{1}, b: [2]bool{hasB}},
			func() (*ir.Kernel, error) { return kernels.DHMatABt(1, rows, nOut, k, k, k, nOut, 0, 0, 0, 1, hasB) })
		bb := w(wt)
		if hasB {
			bb = w(bias)
		}
		launch(kk, rows*nOut, x, w(wt), bb, dst)
	}
	norm := func(dst, x backend.Buf, rows, width int, wt, bias []float32, eps float32, rms bool) {
		hasB := len(bias) != 0
		kk := kern(dhKey{name: "norm", a: [10]int{width, rows}, f: [3]float32{eps}, b: [2]bool{rms, hasB}},
			func() (*ir.Kernel, error) { return kernels.DHNormRows(width, rows, eps, rms, hasB) })
		bb := w(wt)
		if hasB {
			bb = w(bias)
		}
		launch(kk, rows, x, w(wt), bb, dst)
	}
	add := func(dst, a, b backend.Buf, nf int) {
		launch(kern(dhKey{name: "add", a: [10]int{nf}}, func() (*ir.Kernel, error) { return kernels.Add(nf) }), nf, a, b, dst)
	}
	gather := func(dst, x, idx, t backend.Buf, rows, width int) {
		launch(kern(dhKey{name: "gather", a: [10]int{width, rows}}, func() (*ir.Kernel, error) { return kernels.DHGatherAdd(width, rows) }),
			rows*width, x, idx, t, dst)
	}
	act := func(dst, x backend.Buf, nf int) {
		launch(kern(dhKey{name: "gelu", a: [10]int{nf}}, func() (*ir.Kernel, error) { return kernels.Act(nf, kernels.ActGELUErf) }), nf, x, dst)
	}
	l2 := func(dst, x backend.Buf, rows, width int) {
		launch(kern(dhKey{name: "l2", a: [10]int{width, rows}}, func() (*ir.Kernel, error) { return kernels.DHL2Rows(width, rows) }), rows, x, dst)
	}
	dot := func(dst, a, idx, b backend.Buf, rows, width int) {
		launch(kern(dhKey{name: "dot", a: [10]int{width, rows}}, func() (*ir.Kernel, error) { return kernels.DHRowDot(width, rows) }), rows, a, idx, b, dst)
	}
	softmax := func(dst, x backend.Buf, rows, m int) {
		launch(kern(dhKey{name: "softmax", a: [10]int{rows, m}}, func() (*ir.Kernel, error) { return kernels.RowSoftmax(rows, m) }), rows, x, dst)
	}
	// attn is torch's MultiheadAttention, mq query rows over nk key rows.
	attn := func(dst, xq, xkv backend.Buf, mq, nk int, wq, wk, wv, wo, bq, bk, bv, bo []float32) {
		q, k, v := buf("aq", mq*W), buf("ak", nk*W), buf("av", nk*W)
		linear(q, xq, mq, W, W, wq, bq)
		linear(k, xkv, nk, W, W, wk, bk)
		linear(v, xkv, nk, W, W, wv, bv)
		hd := W / H
		s, p, o := buf("as", H*mq*nk), buf("ap", H*mq*nk), buf("ao", mq*W)
		sc := float32(1 / math.Sqrt(float64(hd)))
		launch(kern(dhKey{name: "matabt", a: [10]int{H, mq, nk, hd, W, W, nk, hd, hd, mq * nk}, f: [3]float32{sc}},
			func() (*ir.Kernel, error) { return kernels.DHMatABt(H, mq, nk, hd, W, W, nk, hd, hd, mq*nk, sc, false) }),
			H*mq*nk, q, k, k, s)
		softmax(p, s, H*mq, nk)
		launch(kern(dhKey{name: "matab", a: [10]int{H, mq, hd, nk, nk, W, W, mq * nk, hd, hd}},
			func() (*ir.Kernel, error) { return kernels.DHMatAB(H, mq, hd, nk, nk, W, W, mq*nk, hd, hd) }),
			H*mq*hd, p, v, o)
		linear(dst, o, mq, W, W, wo, bo)
	}
	// ffn is x' = x + down(gelu(up(LN(x)))); the result lands in dst.
	ffn := func(dst, x backend.Buf, rows int, b *nn.ClefHeadBlock) {
		ln, u, ua, dn := buf("fl", rows*W), buf("fu", rows*F), buf("fa", rows*F), buf("fd", rows*W)
		norm(ln, x, rows, W, b.FFNW, b.FFNB, h.Eps, false)
		linear(u, ln, rows, W, F, b.Up, b.BUp)
		act(ua, u, rows*F)
		linear(dn, ua, rows, F, W, b.Down, b.BDown)
		add(dst, x, dn, rows*W)
	}

	// The request's inputs.
	resid, pool, lex := buf("resid", n*D), buf("pool", (nq+1+no)*n), buf("lex", no*D)
	owner, qtype, zero, mask := buf("owner", no), buf("qtype", nq), buf("zero", max(nq, no)), buf("mask", nq*no)
	// 1. The output norm, the head's norm, the memory, the pooled rows.
	hn, hid, mem := buf("hn", n*D), buf("hid", n*D), buf("mem", n*W)
	if h.OutLayerNorm {
		norm(hn, resid, n, D, h.OutNorm, h.OutNormB, h.RMSEps, false)
	} else {
		norm(hn, resid, n, D, h.OutNorm, nil, h.RMSEps, true)
	}
	norm(hid, hn, n, D, h.HidW, h.HidB, h.Eps, false)
	linear(mem, hid, n, D, W, h.PMem, nil)
	pooled := buf("pooled", (nq+1+no)*D)
	launch(kern(dhKey{name: "matab", a: [10]int{1, nq + 1 + no, D, n, n, D, D}},
		func() (*ir.Kernel, error) { return kernels.DHMatAB(1, nq+1+no, D, n, n, D, D, 0, 0, 0) }),
		(nq+1+no)*D, pool, hid, pooled)
	// pooled rows: the questions, the last position, the options.
	qv, glob, ctx := buf("qv", nq*D), buf("glob", D), buf("ctx", no*D)
	copyRows := func(dst backend.Buf, from, rows int) {
		// A copy as a pooling product: the identity rows are in the pool's
		// tail, so a slice is a gather with idx (rows of the pooled table).
		launch(kern(dhKey{name: "slice", a: [10]int{1, rows, D, nq + 1 + no, nq + 1 + no, D, D}},
			func() (*ir.Kernel, error) { return kernels.DHMatAB(1, rows, D, nq+1+no, nq+1+no, D, D, 0, 0, 0) }),
			rows*D, buf("sel"+fmt.Sprint(from), rows*(nq+1+no)), pooled, dst)
	}
	copyRows(qv, 0, nq)
	copyRows(glob, nq, 1)
	copyRows(ctx, nq+1, no)
	// 2. One query per option.
	o1, o2, o3, gq := buf("o1", no*W), buf("o2", no*W), buf("o3", no*W), buf("gq", nq*W)
	linear(o1, ctx, no, D, W, h.POC, nil)
	linear(o2, lex, no, D, W, h.POL, nil)
	add(o3, o1, o2, no*W)
	linear(gq, qv, nq, D, W, h.POQ, nil)
	opts := buf("opts0", no*W)
	gather(opts, o3, owner, gq, no, W)
	// 3. The routing blocks.
	cur, nxt := "opts0", "opts1"
	for bi := 0; bi < h.Routing; bi++ {
		b := &h.Blocks[bi]
		kv, qn, a, x1 := buf("kv", n*W), buf("qn", no*W), buf("ra", no*W), buf("rx", no*W)
		norm(kv, mem, n, W, b.KVW, b.KVB, h.Eps, false)
		norm(qn, buf(cur, no*W), no, W, b.CrossW, b.CrossB, h.Eps, false)
		attn(a, qn, kv, no, n, b.CQ, b.CK, b.CV, b.CO, b.CBQ, b.CBK, b.CBV, b.CBO)
		add(x1, buf(cur, no*W), a, no*W)
		ffn(buf(nxt, no*W), x1, no, b)
		cur, nxt = nxt, cur
	}
	opts = buf(cur, no*W)
	// 4. The fields.
	f0, sw, sp, sm, sn := buf("f0", nq*W), buf("sw", nq*no), buf("sp", nq*no), buf("sm", nq*W), buf("sn", nq*W)
	linear(f0, qv, nq, D, W, h.PQ, nil)
	sc := float32(1 / math.Sqrt(float64(W)))
	launch(kern(dhKey{name: "matabt", a: [10]int{1, nq, no, W, W, W, no}, f: [3]float32{sc}},
		func() (*ir.Kernel, error) { return kernels.DHMatABt(1, nq, no, W, W, W, no, 0, 0, 0, sc, false) }),
		nq*no, f0, opts, opts, sw)
	swm := buf("swm", nq*no)
	add(swm, sw, mask, nq*no)
	softmax(sp, swm, nq, no)
	launch(kern(dhKey{name: "matab", a: [10]int{1, nq, W, no, no, W, W}},
		func() (*ir.Kernel, error) { return kernels.DHMatAB(1, nq, W, no, no, W, W, 0, 0, 0) }),
		nq*W, sp, opts, sm)
	norm(sn, sm, nq, W, h.SumW, h.SumB, h.Eps, false)
	f1, gw, f2, f3 := buf("f1", nq*W), buf("gw", W), buf("f2", nq*W), buf("fa0", nq*W)
	add(f1, f0, sn, nq*W)
	linear(gw, glob, 1, D, W, h.PG, nil)
	gather(f2, f1, zero, gw, nq, W)
	launch(kern(dhKey{name: "gather", a: [10]int{W, nq}}, func() (*ir.Kernel, error) { return kernels.DHGatherAdd(W, nq) }),
		nq*W, f2, qtype, w(h.Types), f3)
	// 5. The joint blocks.
	cur, nxt = "fa0", "fa1"
	for bi := h.Routing; bi < len(h.Blocks); bi++ {
		b := &h.Blocks[bi]
		x, a, x1, a2, x2 := buf("jx", nq*W), buf("ja", nq*W), buf("jx1", nq*W), buf("ja2", nq*W), buf("jx2", nq*W)
		norm(x, buf(cur, nq*W), nq, W, b.SelfW, b.SelfB, h.Eps, false)
		attn(a, x, x, nq, nq, b.SQ, b.SK, b.SV, b.SO, b.SBQ, b.SBK, b.SBV, b.SBO)
		add(x1, buf(cur, nq*W), a, nq*W)
		xn := buf("jxn", nq*W)
		norm(xn, x1, nq, W, b.CrossW, b.CrossB, h.Eps, false)
		attn(a2, xn, mem, nq, n, b.CQ, b.CK, b.CV, b.CO, b.CBQ, b.CBK, b.CBV, b.CBO)
		add(x2, x1, a2, nq*W)
		ffn(buf(nxt, nq*W), x2, nq, b)
		cur, nxt = nxt, cur
	}
	fields := buf("fields", nq*W)
	norm(fields, buf(cur, nq*W), nq, W, h.FieldW, h.FieldB, h.Eps, false)
	// 6. The prior, the cosine, the scorer.
	anc, ancN, lexN, prior := buf("anc", nq*D), buf("ancn", nq*D), buf("lexn", no*D), buf("prior", no)
	gather(anc, qv, zero, glob, nq, D)
	l2(ancN, anc, nq, D)
	l2(lexN, lex, no, D)
	dot(prior, lexN, owner, ancN, no, D)
	on, onl, fl, cosv := buf("on", no*W), buf("onl", no*W), buf("fl2", nq*W), buf("cos", no)
	norm(on, opts, no, W, h.OptW, h.OptB, h.Eps, false)
	l2(onl, on, no, W)
	l2(fl, fields, nq, W)
	dot(cosv, onl, owner, fl, no, W)
	feat, hs, ha, res, logits := buf("feat", no*4*W), buf("hs", no*W), buf("ha", no*W), buf("res", no), buf("logits", no)
	launch(kern(dhKey{name: "features", a: [10]int{W, no}}, func() (*ir.Kernel, error) { return kernels.DHFeatures(W, no) }),
		no*W, fields, owner, on, feat)
	linear(hs, feat, no, 4*W, W, h.Sc, h.ScB)
	act(ha, hs, no*W)
	linear(res, ha, no, W, 1, h.ScOut, h.ScOutB)
	launch(kern(dhKey{name: "combine", a: [10]int{no}, f: h.Scales},
		func() (*ir.Kernel, error) {
			return kernels.DHCombine(no, h.Scales[0], h.Scales[1], h.Scales[2])
		}), no, prior, cosv, res, logits)
	if err != nil {
		pr.free()
		return nil, err
	}
	// The selection rows: identity rows picking pooled rows from..from+rows.
	sel := func(from, rows int) []byte {
		m := make([]float32, rows*(nq+1+no))
		for i := 0; i < rows; i++ {
			m[i*(nq+1+no)+from+i] = 1
		}
		return f32b(m)
	}
	pr.prog = prog
	pr.resid, pr.pool, pr.lex, pr.owner, pr.qtype = resid, pool, lex, owner, qtype
	pr.zero, pr.mask, pr.logits = zero, mask, logits
	pr.sel = [3]backend.Buf{named["sel0"], named[fmt.Sprint("sel", nq)], named[fmt.Sprint("sel", nq+1)]}
	pr.selB = [3][]byte{sel(0, nq), sel(nq, 1), sel(nq+1, no)}
	pr.zeroB = u32b(make([]uint32, max(nq, no)))
	return pr, nil
}

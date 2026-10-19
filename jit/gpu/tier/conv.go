package tier

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// A convolutional tower's blocks on the device (nn.ConvPlan): Gemma 3n's
// MobileNet-V5. A block is a PROGRAM -- a list of launches over named
// buffers, built and compiled at admission, since a Session may not compile
// -- and ConvLayers runs the programs of a range in one submission, the
// activation staying on the card from block to block. The matrices are the
// block's ordinary slots (Up and Down, or Wq..Wo), so the pager pages a
// convolutional block exactly as it pages any other; the buffers between
// launches are one scratch the device's convolutional blocks share, sized
// for the widest of them.

// cbuf names a buffer a program launches against: the scratch's, the
// block's own vectors, and the block's input and output, which are the two
// activation buffers in turn.
type cbuf uint8

const (
	cX cbuf = iota // the block's input
	cY             // its output
	cT1
	cT2
	cE1
	cE2
	cPad
	cCol
	cQ
	cK
	cV
	cAO
	cSc
	cSc2
	cA  // the int8 activation, bytes
	cAX // its scales and sums
	// A FastConformer block's (ConvConformer): the call's row count (one
	// u32, written per call), the residual's three stages and a LayerNorm's
	// two partial planes.
	cTN
	cX1
	cX2
	cX3
	cLP
	cLV
	nConvScratch
)

const (
	cN1 cbuf = 32 + iota
	cN2
	cDwS
	cDwSN
	cDwM
	cDwMN
	cKD
	cKDN
	cVD
	cVDN
	cVecEnd
)

// A FastConformer block's vectors (nn.ConformerVecs.List order), from cC0.
const (
	cC0 cbuf = 64 + iota
	cCFF1NB
	cCFF1UB
	cCFF1DB
	cCAttN
	cCAttNB
	cCQB
	cCKB
	cCVB
	cCOB
	cCU
	cCV
	cCConvN
	cCConvNB
	cCGLUVB
	cCGLUGB
	cCDW
	cCBNS
	cCBNH
	cCPW2B
	cCFF2N
	cCFF2NB
	cCFF2UB
	cCFF2DB
	cCOutN
	cCOutNB
	cCPE
	cCEnd
)

// convStep is one launch. A matmul (mm set) is its batched twin over the
// block's weight in slot, reading the int8 activation quantized from src (and
// src itself where the twin or a float weight wants floats).
type convStep struct {
	k             backend.Kernel
	threads       int
	groups, width int // a launch of groups x width (the norms); 0 is la's 128
	bufs          []cbuf
	mm            *mv
	slot          int
	src, dst      cbuf
}

// convLayer is a convolutional block on the device.
type convLayer struct {
	p    nn.ConvPlan
	vecs [cVecEnd - cN1]backend.Buf
	// cvecs is a FastConformer block's vectors, from cC0.
	cvecs [cCEnd - cC0]backend.Buf
	// ts is the layer's weight slots (layer.tensors), taken once: the slice
	// is made on every call of tensors.
	ts   []**resident
	prog []convStep
	// graph is the block's verified graph where the plan carries one: its
	// launches are lowered from it (graphConvProgram) and dev is the
	// program they were recorded from. vecRole names the vectors by role.
	graph   *graph.Graph
	dev     *devProgram
	vecRole []vecAt
	// out is the block's output length in floats.
	out int
}

// convScratch is the buffers every convolutional block's program runs
// against; act holds the activation between blocks, act[cur] the current.
type convScratch struct {
	b    [nConvScratch]backend.Buf
	n    [nConvScratch]int // bytes
	act  [2]backend.Buf
	actN int
	cur  int
}

// shapeKern is the compiled kernel for key, built once per device: the one
// shape-keyed cache the convolution path and the program's generated ops share. nil is a
// shape the backend refused, and the reason is in LastErr.
func (g *devTier) shapeKern(key convKey, build func() (*ir.Kernel, error)) backend.Kernel {
	enc := []uint32{encTemplate, uint32(key.id), uint32(key.act), math.Float32bits(key.eps)}
	for _, a := range key.a {
		enc = append(enc, uint32(a))
	}
	k, _ := g.genKern(enc, build)
	return k
}

// same is TF SAME's padding along one side: the output length and the zeros
// before the first sample (the smaller half).
func same(n, k, s int) (out, before int) {
	out = (n + s - 1) / s
	return out, max((out-1)*s+k-n, 0) / 2
}

// convProgram builds block l's program against plan p and the weights ws (in
// weightList order), and the scratch bytes it needs. A shape no kernel
// expresses is a decline, named.
func (g *devTier) convProgram(cl *convLayer, ws []nn.Weight) ([nConvScratch]int, string) {
	p := cl.p
	var need [nConvScratch]int
	var prog []convStep
	why := ""
	want := func(b cbuf, floats int) { need[b] = max(need[b], 4*floats) }
	kern := func(key convKey, build func() (*ir.Kernel, error)) backend.Kernel {
		k := g.shapeKern(key, build)
		if k == nil && why == "" {
			why = fmt.Sprintf("the device has no %s kernel: %s", key.id, g.LastErr)
		}
		return k
	}
	launch := func(k backend.Kernel, threads int, bufs ...cbuf) {
		prog = append(prog, convStep{k: k, threads: threads, bufs: bufs})
	}
	norm := func(src, vec, dst cbuf, c, rows int) {
		eps := float32(p.Eps)
		k := kern(convKey{id: kidRmsnormrows, a: [8]int{c, rows}, eps: eps}, func() (*ir.Kernel, error) {
			return kernels.RMSNormRows(c, rows, eps, false, false)
		})
		prog = append(prog, convStep{k: k, groups: rows, width: kernels.RMSNormGroup, bufs: []cbuf{src, vec, dst}})
		want(dst, c*rows)
	}
	gelu := func(src, dst cbuf, n int) {
		launch(kern(convKey{id: kidAct, act: kernels.ActGELU, a: [8]int{n}}, func() (*ir.Kernel, error) {
			return kernels.Act(n, kernels.ActGELU)
		}), n, src, dst)
		want(dst, n)
	}
	add := func(a, b, dst cbuf, n int) {
		launch(kern(convKey{id: kidAdd, a: [8]int{n}}, func() (*ir.Kernel, error) { return kernels.Add(n) }), n, a, b, dst)
	}
	mm := func(slot int, src, dst cbuf, n, k int) {
		x := ws[slot]
		q, ok := quantOf(x.T)
		if !ok || x.K != k || len(x.Data) == 0 {
			if why == "" {
				why = fmt.Sprintf("%s is %v %dx%d, and the block reads %d", weightRole(slot), x.T, x.Rows, x.K, k)
			}
			return
		}
		tok := BatchTok
		for tok > 1 && n%tok != 0 {
			tok /= 2
		}
		bm, ok := g.batchMV(mv{q: q, k: k, rows: x.Rows}, n, tok)
		if !ok {
			if why == "" {
				why = fmt.Sprintf("%s has no batched matvec at %d rows: %s", weightRole(slot), n, g.LastErr)
			}
			return
		}
		// A binary16 twin (sm_70's, Metal's) converts src itself (bm.act)
		// and never reads the int8 activation.
		if bm.act == nil {
			nk := n * k
			launch(kern(convKey{id: kidQuantize, a: [8]int{nk, p.ActWin}}, func() (*ir.Kernel, error) {
				return kernels.Quantize(nk, p.ActWin)
			}), kernels.QuantizeThreads(nk/32), src, cA, cAX)
			need[cA] = max(need[cA], nk)
			need[cAX] = max(need[cAX], 3*(nk/32)*4)
		}
		prog = append(prog, convStep{mm: &bm, slot: slot, src: src, dst: dst})
		want(dst, n*x.Rows)
	}
	// pad copies src (h x w of c) into the padded grid for a k x k window at
	// stride s, and returns the padded width and the output grid.
	pad := func(src cbuf, h, w, c, k, s int) (wp, ho, wo int) {
		ho, top := same(h, k, s)
		wo, left := same(w, k, s)
		hp := (ho-1)*s + k
		wp = (wo-1)*s + k
		launch(kern(convKey{id: kidPadrows, a: [8]int{h, w, c, hp, wp, top, left}}, func() (*ir.Kernel, error) {
			return kernels.PadRows(h, w, c, hp, wp, top, left)
		}), hp*wp*c, src, cPad)
		want(cPad, hp*wp*c)
		return wp, ho, wo
	}
	dw := func(filt, dst cbuf, k, s, wp, c, ho, wo int) {
		launch(kern(convKey{id: kidDwconv, a: [8]int{k, s, wp, c, ho, wo}}, func() (*ir.Kernel, error) {
			return kernels.DWConv(k, s, wp, c, ho, wo)
		}), ho*wo*c, cPad, filt, dst)
		want(dst, ho*wo*c)
	}
	// finish is the projection's norm and the residual: the norm writes the
	// output, or a temporary the input is added to.
	finish := func(src cbuf, n int) {
		if !p.Residual {
			norm(src, cN2, cY, p.COut, n)
			return
		}
		norm(src, cN2, cT2, p.COut, n)
		add(cT2, cX, cY, n*p.COut)
	}
	want(cX, p.HIn*p.WIn*p.CIn)
	want(cY, p.HOut*p.WOut*p.COut)
	n := p.HOut * p.WOut
	switch p.Kind {
	case nn.ConvEdge:
		wp, ho, wo := pad(cX, p.HIn, p.WIn, p.CIn, 3, p.Stride)
		if ho*wo != n {
			return need, fmt.Sprintf("an edge residual writes %dx%d and its plan says %dx%d", ho, wo, p.HOut, p.WOut)
		}
		kpad := ws[5].K
		padLen := ((ho-1)*p.Stride + 3) * wp * p.CIn
		launch(kern(convKey{id: kidIm2col, a: [8]int{3, p.Stride, wp, p.CIn, wo, kpad, n, padLen}}, func() (*ir.Kernel, error) {
			return kernels.Im2col(3, p.Stride, wp, p.CIn, wo, kpad, n, padLen)
		}), n*kpad, cPad, cCol)
		want(cCol, n*kpad)
		mm(5, cCol, cE1, n, kpad)
		norm(cE1, cN1, cE2, p.CExp, n)
		gelu(cE2, cE1, n*p.CExp)
		mm(6, cE1, cT1, n, p.CExp)
		finish(cT1, n)
	case nn.ConvInverted:
		in, h, w := cX, p.HIn, p.WIn
		if p.KStart > 0 {
			wp, ho, wo := pad(cX, h, w, p.CIn, p.KStart, p.SStart)
			dw(cDwS, cT2, p.KStart, p.SStart, wp, p.CIn, ho, wo)
			norm(cT2, cDwSN, cT1, p.CIn, ho*wo)
			in, h, w = cT1, ho, wo
		}
		nm := h * w
		mm(5, in, cE1, nm, p.CIn)
		norm(cE1, cN1, cE2, p.CExp, nm)
		gelu(cE2, cE1, nm*p.CExp)
		e := cE1
		if p.KMid > 0 {
			wp, ho, wo := pad(cE1, h, w, p.CExp, p.KMid, p.SMid)
			dw(cDwM, cE2, p.KMid, p.SMid, wp, p.CExp, ho, wo)
			norm(cE2, cDwMN, cE1, p.CExp, ho*wo)
			gelu(cE1, cE2, ho*wo*p.CExp)
			e, h, w = cE2, ho, wo
		}
		if h*w != n {
			return need, fmt.Sprintf("an inverted residual writes %dx%d and its plan says %dx%d", h, w, p.HOut, p.WOut)
		}
		mm(6, e, cT1, n, p.CExp)
		finish(cT1, n)
	case nn.ConvAttention:
		nin, m, qd := p.HIn*p.WIn, p.HKV*p.WKV, p.Heads*p.KD
		norm(cX, cN1, cT1, p.CIn, nin)
		mm(0, cT1, cQ, nin, p.CIn)
		for _, side := range []struct {
			slot      int
			down, nrm cbuf
			out       cbuf
		}{{1, cKD, cKDN, cK}, {2, cVD, cVDN, cV}} {
			src := cT1
			if p.KVK > 0 {
				wp, ho, wo := pad(cT1, p.HIn, p.WIn, p.CIn, p.KVK, 2)
				if ho*wo != m {
					return need, fmt.Sprintf("the key downsample writes %dx%d and its plan says %dx%d", ho, wo, p.HKV, p.WKV)
				}
				dw(side.down, cT2, p.KVK, 2, wp, p.CIn, ho, wo)
				norm(cT2, side.nrm, cE1, p.CIn, m)
				src = cE1
			}
			mm(side.slot, src, side.out, m, p.CIn)
		}
		launch(kern(convKey{id: kidMqascores, a: [8]int{nin, m, p.Heads, p.KD}}, func() (*ir.Kernel, error) {
			return kernels.MQAScores(nin, m, p.Heads, p.KD)
		}), nin*p.Heads*m, cQ, cK, cSc)
		want(cSc, nin*p.Heads*m)
		launch(kern(convKey{id: kidRowsoftmax, a: [8]int{nin * p.Heads, m}}, func() (*ir.Kernel, error) {
			return kernels.RowSoftmax(nin*p.Heads, m)
		}), nin*p.Heads, cSc, cSc2)
		want(cSc2, nin*p.Heads*m)
		launch(kern(convKey{id: kidMqaacc, a: [8]int{nin, m, p.Heads, p.KD}}, func() (*ir.Kernel, error) {
			return kernels.MQAAcc(nin, m, p.Heads, p.KD)
		}), nin*qd, cSc2, cV, cAO)
		want(cAO, nin*qd)
		if p.Residual {
			mm(3, cAO, cT2, nin, qd)
			add(cT2, cX, cY, nin*p.COut)
		} else {
			mm(3, cAO, cY, nin, qd)
		}
	case nn.ConvConformer:
		if cl.graph != nil {
			dev, why2 := g.graphConvProgram(cl, cl.graph, ws, kern, launch, mm, want, need[:])
			cl.dev = dev
			if why == "" {
				why = why2
			}
			break
		}
		why2 := g.conformerProgram(p, kern, launch, mm, want, need[:])
		if why == "" {
			why = why2
		}
	default:
		return need, fmt.Sprintf("a convolutional block of kind %d", p.Kind)
	}
	cl.prog, cl.out = prog, n*p.COut
	// The two activations are one pair (act), sized for the widest side.
	need[cX] = max(need[cX], need[cY])
	need[cY] = 0
	return need, why
}

// convVecs is the block's vectors in cbuf order from cN1, with the lengths a
// plan wants (0 for one the block has none of).
func convVecs(p *nn.ConvPlan, w *nn.ConvWeights) ([cVecEnd - cN1][]float32, [cVecEnd - cN1]int) {
	have := [cVecEnd - cN1][]float32{w.Norm1, w.Norm2, w.DwStart, w.DwStartNorm, w.DwMid, w.DwMidNorm,
		w.KDown, w.KDownNorm, w.VDown, w.VDownNorm}
	var want [cVecEnd - cN1]int
	switch p.Kind {
	case nn.ConvEdge:
		want[0], want[1] = p.CExp, p.COut
	case nn.ConvInverted:
		want[0], want[1] = p.CExp, p.COut
		if p.KStart > 0 {
			want[2], want[3] = p.KStart*p.KStart*p.CIn, p.CIn
		}
		if p.KMid > 0 {
			want[4], want[5] = p.KMid*p.KMid*p.CExp, p.CExp
		}
	case nn.ConvAttention:
		want[0] = p.CIn
		if p.KVK > 0 {
			k := p.KVK * p.KVK * p.CIn
			want[6], want[7], want[8], want[9] = k, p.CIn, k, p.CIn
		}
	}
	return have, want
}

// prepConv is PrepLayer for a convolutional block. Callers hold g.mu.
func (g *devTier) prepConv(li int, p *nn.LayerPlan, w *nn.LayerWeights, mayPage bool) bool {
	c, cw := p.Conv, w.Conv
	if cw == nil {
		g.LastErr = fmt.Sprintf("block %d: a convolutional plan offered no convolutional vectors", li)
		return false
	}
	// No history: a block already here is shared whole.
	if l := g.layers[li]; l != nil && l.ok {
		return true
	}
	have, want := convVecs(c, cw)
	var chave [cCEnd - cC0][]float32
	if c.Kind == nn.ConvConformer {
		if cw.Conformer == nil {
			g.LastErr = fmt.Sprintf("block %d: a conformer plan offered no conformer vectors", li)
			return false
		}
		chave = cw.Conformer.List()
		cwant := conformerVecLens(c)
		for i, v := range chave {
			if len(v) != cwant[i] {
				g.LastErr = fmt.Sprintf("block %d: conformer vector %d is %d floats, want %d", li, i, len(v), cwant[i])
				return false
			}
		}
	}
	var aux uint64
	for _, v := range chave {
		aux += uint64(4 * len(v))
	}
	for i, n := range want {
		if len(have[i]) != n {
			g.LastErr = fmt.Sprintf("block %d: convolutional vector %d is %d floats, want %d", li, i, len(have[i]), n)
			return false
		}
		aux += uint64(4 * n)
	}
	g.dropGraph()
	ws := weightList(nil, w, nil)
	cl := &convLayer{p: *c, graph: p.Graph}
	// The block's vectors by role: each the list entry over the same memory,
	// uploaded once in the list's place.
	for _, v := range w.Vecs {
		for j, x := range chave {
			if len(x) != 0 && len(v.V) == len(x) && &v.V[0] == &x[0] {
				cl.vecRole = append(cl.vecRole, vecAt{v.Role, cC0 + cbuf(j)})
				break
			}
		}
	}
	sz, why := g.convProgram(cl, ws)
	if why != "" {
		g.LastErr = fmt.Sprintf("block %d: %s", li, why)
		return false
	}
	full, why, ok := g.pageSizeWhy(ws)
	if !ok {
		g.LastErr = fmt.Sprintf("block %d: %s", li, why)
		return false
	}
	if full > g.widest {
		g.widest = full
	}
	need, _ := g.blockBytes(ws)
	need += aux + g.convGrowth(sz)
	g.kvCompact(need, nil)
	if !g.room(need) {
		if n := g.slotsIfPerm(aux); !mayPage || !g.stream[li] || n < minSlots {
			if mayPage {
				g.Declined++
			}
			g.LastErr = fmt.Sprintf("block %d needs %d bytes, %d of %d used", li, need, g.used, g.limit)
			return false
		}
		if !g.reclaim(need, li, 0, li+1) {
			g.Declined++
			g.LastErr = fmt.Sprintf("block %d needs %d bytes and nothing can be paged out (%d of %d used)",
				li, need, g.used, g.limit)
			return false
		}
	}
	if w.Ensure != nil {
		if err := w.Ensure(w); err != nil {
			g.LastErr = fmt.Sprintf("block %d: %v", li, err)
			return false
		}
		ws = weightList(nil, w, nil)
	}
	if !g.convGrow(sz) {
		return false
	}
	g.ensureRaw(ws, need)
	l := &layer{nonCausal: true, conv: cl}
	ts := l.tensors()
	cl.ts = ts
	nmat := 7
	if c.Kind == nn.ConvConformer {
		nmat = 12
	}
	for i, x := range ws[:nmat] {
		if len(x.Data) == 0 {
			continue
		}
		q, _ := quantOf(x.T)
		r := g.resident(li, i, q, x.Data, x.Rows, x.K, x.Packed)
		if r == nil || !r.ok {
			if g.LastErr == "" {
				g.LastErr = fmt.Sprintf("block %d: %s not resident", li, weightRole(i))
			}
			g.dropTensors(l)
			return false
		}
		*ts[i] = r
	}
	for i, v := range have {
		if want[i] == 0 {
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
			for _, o := range cl.vecs {
				if o != nil {
					o.Free()
				}
			}
			g.dropTensors(l)
			g.LastErr = fmt.Sprintf("block %d: %v", li, err)
			return false
		}
		cl.vecs[i] = b
	}
	for i, v := range chave {
		if len(v) == 0 {
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
			for _, o := range append(cl.vecs[:], cl.cvecs[:]...) {
				if o != nil {
					o.Free()
				}
			}
			g.dropTensors(l)
			g.LastErr = fmt.Sprintf("block %d: %v", li, err)
			return false
		}
		cl.cvecs[i] = b
	}
	l.auxBytes = aux
	g.charge(aux)
	l.ok = true
	var pb, pr uint64
	for _, rp := range ts {
		r := *rp
		pb += r.bytes()
		if n := r.bytes(); n > 0 {
			if f, err := g.planesFootprint(r.t, r.nrows, r.k); err == nil && f > n {
				pr += f - n
			}
		}
	}
	l.pg = &page{ws: append([]nn.Weight(nil), ws...), bytes: pb, round: pr, in: true, ensure: w.Ensure}
	g.pagesIn++
	g.pageBytes += pb + pr
	if old := g.layers[li]; old != nil {
		g.dropLayerLocal(old)
	}
	if g.layers == nil {
		g.layers = map[int]*layer{}
	}
	// Its own program of generated kernels (convProg), outside the device
	// program's lowering: named, so the compiler's coverage counts it.
	l.progWhy = "a convolution block, run by its own program of kernels"
	if cl.dev != nil {
		// Lowered from its graph: the program it was recorded from.
		l.prog, l.progWhy = cl.dev, ""
	}
	g.layers[li] = l
	g.layerGen++
	g.convN++
	g.trim()
	return true
}

// convGrowth is what growing the scratch to sz costs.
func (g *devTier) convGrowth(sz [nConvScratch]int) uint64 {
	cs := &g.conv
	var n uint64
	for i, b := range sz {
		if cbuf(i) == cX {
			if b > cs.actN {
				n += 2 * uint64(b-cs.actN)
			}
			continue
		}
		if b > cs.n[i] {
			n += uint64(b - cs.n[i])
		}
	}
	return n
}

// convGrow grows the shared scratch to at least sz, every new buffer
// poisoned (RULE 13) and charged as scratch (regrow). Callers hold g.mu.
func (g *devTier) convGrow(sz [nConvScratch]int) bool {
	cs := &g.conv
	if sz[cX] > cs.actN {
		for i := range cs.act {
			n := cs.actN
			if !g.regrow(&cs.act[i], &n, sz[cX], 1, true, true) {
				return false
			}
		}
		cs.actN = sz[cX]
	}
	for i := range sz {
		if c := cbuf(i); c != cX && c != cY && sz[i] > 0 && !g.regrow(&cs.b[i], &cs.n[i], sz[i], 1, true, true) {
			return false
		}
	}
	return true
}

// convLeft is a convolutional block leaving the device; the last one takes
// the scratch with it. Callers hold g.mu.
func (g *devTier) convLeft(l *layer) {
	if l.conv == nil {
		return
	}
	l.conv = nil
	if g.convN--; g.convN > 0 {
		return
	}
	defer g.scratchWin().close()
	cs := &g.conv
	for _, b := range append(cs.b[:], cs.act[:]...) {
		if b != nil {
			b.Free()
		}
	}
	g.conv = convScratch{}
}

// convBufs is the block's own vectors, for auxBufs.
func (l *layer) convBufs() []backend.Buf {
	if l.conv == nil {
		return nil
	}
	return append(l.conv.vecs[:], l.conv.cvecs[:]...)
}

// convOut is block li's output length in floats, 0 where this device does
// not hold it as a convolutional block.
func (g *devTier) convOut(li int) int {
	g.mu.Lock()
	defer g.mu.Unlock()
	if l := g.layers[li]; l != nil && l.conv != nil {
		return l.conv.out
	}
	return 0
}

// ConvLayers runs convolutional blocks [lo, hi) (nn.ConvDevice), paging them
// in as it goes: one submission per resident run, the activation on the card
// between them.
func (g *devTier) ConvLayers(lo, hi int, x, out []float32, tap int, tapOut []float32) bool {
	if g.injectedFail() {
		return false
	}
	for li := lo; li < hi; {
		if !g.pageInHold(li, lo, hi) {
			return false
		}
		// Held as submit holds its range: no page-in of another session's
		// sends a block away before convOnce reads it.
		end := li + 1
		g.mu.Lock()
		for end < hi && g.holdPages(end, end+1) {
			end++
		}
		g.mu.Unlock()
		ok := g.convOnce(li, end, li == lo, end == hi, x, out, tap, tapOut)
		g.mu.Lock()
		g.dropPages(li, end)
		g.mu.Unlock()
		if !ok {
			return false
		}
		li = end
	}
	return true
}

// convOnce is one submission of resident blocks [lo, hi): x goes up first
// when the call starts here, out comes back when it ends here, and the tap
// block's output comes back as it is written.
func (g *devTier) convOnce(lo, hi int, first, last bool, x, out []float32, tap int, tapOut []float32) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	for li := lo; li < hi; li++ {
		if l := g.layers[li]; l == nil || l.conv == nil || !l.ok {
			g.LastErr = fmt.Sprintf("block %d is not a convolutional block on this device", li)
			return false
		}
	}
	cs := &g.conv
	p0 := &g.layers[lo].conv.p
	// A conformer call is the clip's rows, at most the plan's.
	conformer := p0.Kind == nn.ConvConformer
	if first {
		bad := len(x) != p0.HIn*p0.WIn*p0.CIn
		if conformer {
			bad = len(x) == 0 || len(x)%p0.CIn != 0 || len(x) > p0.HIn*p0.CIn
		}
		if bad || 4*len(x) > cs.actN {
			g.LastErr = fmt.Sprintf("block %d reads %d floats, handed %d", lo, p0.HIn*p0.WIn*p0.CIn, len(x))
			return false
		}
	}
	if l := g.layers[hi-1].conv; last && len(out) != l.out && !(conformer && len(out) == len(x)) {
		g.LastErr = fmt.Sprintf("block %d writes %d floats, handed %d", hi-1, l.out, len(out))
		return false
	}
	// The call's arguments ride the devTier and the submission is a method
	// value made once, so a warm call allocates no closure (RULE 2f).
	c := &g.convCall
	*c = convCallArgs{lo: lo, hi: hi, first: first, last: last, x: x, out: out, tap: tap, tapOut: tapOut}
	if conformer {
		c.rows = len(x) / p0.CIn
	}
	if g.convFn == nil {
		g.convFn = g.convSession
	}
	g.dev.Session(g.convFn)
	err := c.err
	*c = convCallArgs{}
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	g.ConvBlocks += int64(hi - lo)
	return true
}

// convCallArgs is one convOnce call's arguments and its submission's state.
type convCallArgs struct {
	lo, hi      int
	first, last bool
	x, out      []float32
	tap         int
	tapOut      []float32
	rows        int // a conformer call's rows, 0 for another block's
	err         error
	fsrc, msrc  backend.Buf
	memoK       backend.Kernel
	lc          launcher
	tn          [4]byte
}

// convSession is convOnce's submission, inside the backend's Session.
func (g *devTier) convSession(s backend.Session) {
	c, cs := &g.convCall, &g.conv
	g.convTo.s = s
	defer func() { g.convTo.s = nil }()
	c.lc = launcher{to: &g.convTo, err: &c.err, fsrc: &c.fsrc, memoK: &c.memoK, memoSrc: &c.msrc}
	if c.first {
		c.err = s.Write(cs.act[cs.cur], f32b(c.x))
		if c.rows > 0 && c.err == nil {
			binary.LittleEndian.PutUint32(c.tn[:], uint32(c.rows))
			c.err = s.Write(cs.b[cTN], c.tn[:])
		}
	}
	for li := c.lo; li < c.hi && c.err == nil; li++ {
		l := g.layers[li]
		g.convRun(&c.lc, l, cs)
		cs.cur = 1 - cs.cur
		if li == c.tap && c.err == nil {
			c.err = s.Read(cs.act[cs.cur], f32b(c.tapOut[:l.conv.out]))
		}
	}
	if c.last && c.err == nil {
		c.err = s.Read(cs.act[cs.cur], f32b(c.out))
	}
}

// convRun launches one block's program. Callers hold g.mu, inside a Session.
func (g *devTier) convRun(lc *launcher, l *layer, cs *convScratch) {
	cl := l.conv
	var bufs [12]backend.Buf
	buf := func(b cbuf) backend.Buf {
		switch {
		case b == cX:
			return cs.act[cs.cur]
		case b == cY:
			return cs.act[1-cs.cur]
		case b >= cC0:
			return cl.cvecs[b-cC0]
		case b >= cN1:
			return cl.vecs[b-cN1]
		}
		return cs.b[b]
	}
	ts := cl.ts
	for i := range cl.prog {
		st := &cl.prog[i]
		if st.mm == nil {
			bs := bufs[:len(st.bufs)]
			for j, b := range st.bufs {
				bs[j] = buf(b)
			}
			if st.width != 0 {
				if *lc.err == nil {
					*lc.err = lc.launch(st.k, st.groups, st.width, bs...)
				}
				continue
			}
			lc.la(st.k, st.threads, bs...)
			continue
		}
		bm, r := st.mm, *ts[st.slot]
		src, dst := buf(st.src), buf(st.dst)
		a, ax := cs.b[cA], cs.b[cAX]
		d := r.d
		if kernels.IsFloat(r.t) {
			d = src
		}
		out := dst
		if bm.split > 1 && !bm.group {
			out = g.partBuf
		}
		if bm.act != nil {
			lc.la(bm.act, bm.actN, src, g.f16Buf)
			lc.law(bm.kern, bm.threads, bm.width, r.qs, d, r.sc, g.f16Buf, out)
			if t := bm.tail; t != nil {
				lc.la(t.kern, t.threads, r.qs, d, r.sc, g.f16Buf, out)
			}
		} else {
			n := bm.threads
			if n == 0 {
				n = bm.rows * bm.groups * max(bm.split, 1)
			}
			w := 128
			if bm.group {
				w = kernels.GroupSplitWidth(bm.split)
			}
			if *lc.err == nil {
				*lc.err = lc.launch(bm.kern, (n+w-1)/w, w, r.qs, d, r.sc, a, ax, out)
			}
		}
		if bm.split > 1 && !bm.group {
			lc.la(bm.red, bm.redN, g.partBuf, dst)
		}
	}
}

// ConvLayers runs convolutional blocks [lo, hi) device by device, the
// activation crossing the host between devices.
func (g *GPU) ConvLayers(lo, hi int, x, out []float32, tap int, tapOut []float32) bool {
	g.convMu.Lock()
	defer g.convMu.Unlock()
	g.mu.Lock()
	rs, ok := g.runsInto(g.convRuns[:0], lo, hi)
	g.convRuns = rs
	g.mu.Unlock()
	if !ok || len(rs) == 0 {
		return false
	}
	in := x
	for i, r := range rs {
		dst := out
		if i < len(rs)-1 {
			n := r.dev.convOut(r.hi - 1)
			if n == 0 {
				return false
			}
			k := i % 2
			if cap(g.convTmp[k]) < n {
				g.convTmp[k] = make([]float32, n)
			}
			dst = g.convTmp[k][:n]
		}
		if !r.dev.ConvLayers(r.lo, r.hi, in, dst, tap, tapOut) {
			return false
		}
		in = dst
	}
	return true
}

// ConvLayers forwards to the tier (nn.ConvDevice).
func (s *gpuSession) ConvLayers(lo, hi int, x, out []float32, tap int, tapOut []float32) bool {
	defer s.leave(s.step())
	return s.g.ConvLayers(lo, hi, x, out, tap, tapOut)
}

// conformerVecLens is the lengths nn.ConformerVecs.List's vectors must be.
func conformerVecLens(p *nn.ConvPlan) [cCEnd - cC0]int {
	d, ff, r := p.CIn, p.CExp, p.HIn
	return [cCEnd - cC0]int{d, d, ff, d, d, d, d, d, d, d, d, d, d, d, d, d,
		p.KMid * d, d, d, d, d, d, ff, d, d, d, (2*r - 1) * d}
}

// conformerProgram is convProgram's FastConformer block: over R = p.HIn rows
// of width d, the first t (cTN, per call) the clip's.
func (g *devTier) conformerProgram(p nn.ConvPlan, kern func(convKey, func() (*ir.Kernel, error)) backend.Kernel,
	launch func(backend.Kernel, int, ...cbuf), mm func(int, cbuf, cbuf, int, int), want func(cbuf, int),
	need []int) string {
	r, d, ff, heads, kd := p.HIn, p.CIn, p.CExp, p.Heads, p.KD
	if p.WIn != 1 || p.COut != d || heads*kd != d || p.KMid < 1 || p.KMid%2 != 1 {
		return fmt.Sprintf("a conformer block of %d x %d into %d, %d heads of %d, %d taps", p.HIn, p.WIn, p.COut,
			heads, kd, p.KMid)
	}
	need[cTN] = max(need[cTN], 4)
	eps := float32(p.Eps)
	parts := partsFor(d)
	ln := func(src, w, b, dst cbuf) {
		want(cLP, parts*r)
		want(cLV, parts*r)
		launch(kern(convKey{id: kidLnpart, a: [8]int{d, parts, r}}, func() (*ir.Kernel, error) {
			return kernels.LayerNormPartRows(d, parts, r)
		}), parts*r, src, cLP)
		launch(kern(convKey{id: kidLnvar, a: [8]int{d, parts, r}}, func() (*ir.Kernel, error) {
			return kernels.LayerNormVarRows(d, parts, r)
		}), parts*r, src, cLP, cLV)
		launch(kern(convKey{id: kidLnapply, a: [8]int{d, parts, r}, eps: eps}, func() (*ir.Kernel, error) {
			return kernels.LayerNormApplyRows(d, parts, eps, true, r)
		}), d*r, src, w, cLP, cLV, b, dst)
		want(dst, d*r)
	}
	bias := func(src, b, dst cbuf, width int, act kernels.ActKind) {
		launch(kern(convKey{id: kidBiasAct, act: act, a: [8]int{width, r}}, func() (*ir.Kernel, error) {
			return kernels.BiasAct(width, r, act)
		}), width*r, src, b, dst)
		want(dst, width*r)
	}
	add := func(a, b, dst cbuf) {
		launch(kern(convKey{id: kidAdd, a: [8]int{d * r}}, func() (*ir.Kernel, error) { return kernels.Add(d * r) }),
			d*r, a, b, dst)
		want(dst, d*r)
	}
	half := func(a, b, dst cbuf) {
		// x + b/2: the half feed-forward's residual, alpha from cCHalf.
		launch(kern(convKey{id: kidAddhalf, a: [8]int{d * r}}, func() (*ir.Kernel, error) {
			return kernels.AddHalf(d * r)
		}), d*r, a, b, dst)
		want(dst, d*r)
	}
	ffn := func(in, norm, normB, upB, downB, out cbuf, up, down int) {
		ln(in, norm, normB, cT1)
		mm(up, cT1, cE1, r, d)
		bias(cE1, upB, cE2, ff, kernels.ActSiLU)
		mm(down, cE2, cT2, r, ff)
		bias(cT2, downB, cT1, d, kernels.ActIdentity)
		half(in, cT1, out)
	}
	// x1 = x + FF1(LN x)/2
	ffn(cX, cC0, cCFF1NB, cCFF1UB, cCFF1DB, cX1, 5, 6)
	// x2 = x1 + attention
	ln(cX1, cCAttN, cCAttNB, cT1)
	for _, s := range []struct {
		slot int
		b    cbuf
		dst  cbuf
	}{{0, cCQB, cQ}, {1, cCKB, cK}, {2, cCVB, cV}} {
		mm(s.slot, cT1, cE1, r, d)
		bias(cE1, s.b, s.dst, d, kernels.ActIdentity)
	}
	mm(4, cCPE, cCol, 2*r-1, d)
	launch(kern(convKey{id: kidRelposscores, a: [8]int{r, heads, kd}}, func() (*ir.Kernel, error) {
		return kernels.RelPosScores(r, heads, kd)
	}), r*heads*r, cQ, cK, cCol, cCU, cCV, cTN, cSc)
	want(cSc, r*heads*r)
	launch(kern(convKey{id: kidRowsoftmax, a: [8]int{r * heads, r}}, func() (*ir.Kernel, error) {
		return kernels.RowSoftmax(r*heads, r)
	}), r*heads, cSc, cSc2)
	want(cSc2, r*heads*r)
	launch(kern(convKey{id: kidAttnaccrows, a: [8]int{r, heads, kd}}, func() (*ir.Kernel, error) {
		return kernels.AttnAccRows(r, heads, kd)
	}), r*d, cSc2, cV, cTN, cAO)
	want(cAO, r*d)
	mm(3, cAO, cE1, r, d)
	bias(cE1, cCOB, cT2, d, kernels.ActIdentity)
	add(cX1, cT2, cX2)
	// x3 = x2 + the convolution module
	ln(cX2, cCConvN, cCConvNB, cT1)
	mm(7, cT1, cE1, r, d)
	bias(cE1, cCGLUVB, cT2, d, kernels.ActIdentity)
	mm(10, cT1, cE1, r, d)
	bias(cE1, cCGLUGB, cE2, d, kernels.ActIdentity)
	launch(kern(convKey{id: kidSigmoidmul, a: [8]int{d * r}}, func() (*ir.Kernel, error) {
		return kernels.SigmoidMul(d * r)
	}), d*r, cE2, cT2, cE1)
	launch(kern(convKey{id: kidDwconv1drows, a: [8]int{p.KMid, d, r}}, func() (*ir.Kernel, error) {
		return kernels.DWConv1dRows(p.KMid, d, r)
	}), d*r, cE1, cCDW, cTN, cT2)
	launch(kern(convKey{id: kidAffinerows, a: [8]int{d, r}}, func() (*ir.Kernel, error) {
		return kernels.AffineRows(d, r)
	}), d*r, cT2, cCBNS, cCBNH, cE2)
	launch(kern(convKey{id: kidAct, act: kernels.ActSiLU, a: [8]int{d * r}}, func() (*ir.Kernel, error) {
		return kernels.Act(d*r, kernels.ActSiLU)
	}), d*r, cE2, cE1)
	mm(11, cE1, cT2, r, d)
	bias(cT2, cCPW2B, cT1, d, kernels.ActIdentity)
	add(cX2, cT1, cX3)
	// out = LN(x3 + FF2(LN x3)/2)
	ffn(cX3, cCFF2N, cCFF2NB, cCFF2UB, cCFF2DB, cX1, 8, 9)
	ln(cX1, cCOutN, cCOutNB, cY)
	return ""
}

package model

import (
	"sync"

	"github.com/jitllm/jitllm/engine/nn"
)

// regions is the per-State form of the pool regions a decode token runs every
// layer, with the function each hands the pool built once -- engine/nn's
// hotRegions, for the graph's own regions. A closure literal over a call's
// locals escapes into the pool's job and is a heap object per region, which is
// how a warm token came to allocate per region
// (TestDecodeDoesNotAllocate). A State runs one region at a time, so
// one set of fields serves every call; each is cleared of its slices
// afterwards so the State pins no caller's buffer.
type regions struct {
	elem   elemRegion
	attn   attnRegion
	delta  deltaRegion
	mamba1 mamba1Region
	rows   rowRegion
	embed  embedRegion
	pattn  prefAttnRegion
}

// mamba1Region is Mamba-1's selective scan fanned out over channels
// (State.mamba1Scan).
type mamba1Region struct {
	st, a, d []float32
	fn       func(lo, hi int)
}

// mamba1Channels is one chunk of the scan: channels lo..hi-1.
func (s *State) mamba1Channels(lo, hi int) {
	j, n := &s.rg.mamba1, s.dg.kDim
	s.jit.SelScan(s.dOut[lo:hi], j.st[lo*n:hi*n], s.dSB, s.dSC, s.dMixed[lo:hi], s.dBeta[lo:hi],
		j.a[lo*n:hi*n], j.d[lo:hi], hi-lo, n)
}

// elemOp is which elementwise op an elemRegion runs.
type elemOp uint8

const (
	elemActMul elemOp = iota // a = act(a) * b: the gated FFN's activation
	elemAct                  // a = act(a): an ungated FFN's
	elemAxpy                 // a += alpha*b: the residual add
	elemReduce               // a += sum_j w[j]*moeDown[j]: a mixture's routed sum
	elemXIELU                // a = xielu(a; b): Apertus's ungated FFN, b its four numbers
)

// elemRegion is an elementwise op split by vector across the pool. Every op
// but the residual add carries the ragged remainder with the last chunk, as
// the closures it replaces did.
type elemRegion struct {
	op    elemOp
	a, b  []float32 // the operands; b is the routed weights for elemReduce
	act   nn.ActKind
	alpha float32
	n, v  int // the logical length (elemReduce's row width) and its whole vectors
	fn    func(lo, hi int)
}

// elemRun dispatches the elementwise region described by s.rg.elem over its v
// vectors, and clears its slices afterwards.
func (s *State) elemRun() {
	j := &s.rg.elem
	if j.fn == nil {
		j.fn = s.elemChunk
	}
	s.jit.Parallel(j.v, 64, j.fn)
	j.a, j.b = nil, nil
}

// elemChunk is one chunk of the elementwise region.
func (s *State) elemChunk(lo, hi int) {
	j := &s.rg.elem
	lanes := nn.ElemLanes
	a, b := lo*lanes, hi*lanes
	if hi == j.v && j.op != elemAxpy {
		b = j.n // the ragged remainder rides with the last chunk
	}
	switch j.op {
	case elemActMul:
		s.actmul(j.a[a:b], j.b[a:b], j.act)
	case elemAct:
		nn.Act32JIT(j.a[a:b], j.act)
	case elemXIELU:
		nn.XIELU32JIT(j.a[a:b], j.b)
	case elemAxpy:
		s.axpy(j.a[a:b], j.b[a:b], j.alpha)
	case elemReduce:
		n := j.n
		for e, w := range j.b {
			nn.Axpy32JIT(j.a[a:b], s.moeDown[e*n+a:e*n+b], w)
		}
	}
}

// attnRegion is decode's attention fan-out over heads: the arguments the
// region reads, filled by forward before each dispatch (see State.attnHeads).
type attnRegion struct {
	l                     *layer
	li, gqa, qw, ow, hd   int
	astride, pair, w0, an int
	mla                   bool
	scale, softcap        float32
	qbuf, obuf, xb, atf   []float32
	// bias is the lightning indexer's selection over the window (indexer.go),
	// nil where every position is read. bstride is the distance between kv
	// groups' rows of it: zero for DeepSeek V3.2's one selection, the scores
	// stride for MiniMax-M3's one per group (msa.go).
	bias    []float32
	bstride int
	fast    *nn.AttnSet
	fn      func(lo, hi int)
}

// attnHeads is one chunk of decode's attention: heads lo..hi-1. Heads are
// independent -- each writes its own slice of the output and its own scores
// row -- so the split needs no coordination. With GQA, query head hh reads kv
// head hh/gqa.
func (s *State) attnHeads(lo, hi int) {
	j := &s.rg.attn
	for hh := lo; hh < hi; hh++ {
		// A pair needs both heads in this task and the same kv head.
		if j.pair != 0 && j.gqa%2 == 0 && hh%2 == 0 && hh+1 < hi {
			q0, af0, kvh := j.head(hh)
			q1, af1, _ := j.head(hh + 1)
			okS := j.pair&1 != 0 && s.kvScores2(j.fast, j.li, 0, kvh, q0, q1, af0, af1, j.w0, j.an)
			if !okS {
				s.kvScores(j.fast, j.li, 0, kvh, q0, af0, j.w0, j.an)
				s.kvScores(j.fast, j.li, 0, kvh, q1, af1, j.w0, j.an)
			}
			s.attnSoftmax(af0, hh)
			s.attnSoftmax(af1, hh+1)
			o0, o1 := j.out(hh), j.out(hh+1)
			okA := j.pair&2 != 0 && s.kvAcc2(j.fast, j.li, 0, kvh, o0, o1, af0, af1, j.w0, j.an)
			if !okA {
				s.kvAcc(j.fast, j.li, 0, kvh, o0, af0, j.w0, j.an)
				s.kvAcc(j.fast, j.li, 0, kvh, o1, af1, j.w0, j.an)
			}
			j.store(hh, o0)
			j.store(hh+1, o1)
			// Counted only when a paired kernel actually ran, not when the
			// branch fell back to two single calls.
			if okS || okA {
				s.attnPaired.Add(1)
			}
			hh++
			continue
		}
		qh, af, kvh := j.head(hh)
		s.kvScores(j.fast, j.li, 0, kvh, qh, af, j.w0, j.an)
		s.attnSoftmax(af, hh)
		out := j.out(hh)
		s.kvAcc(j.fast, j.li, 0, kvh, out, af, j.w0, j.an)
		j.store(hh, out)
	}
}

// head is query head hh's query, its scores row and its kv head.
func (j *attnRegion) head(hh int) (qh, af []float32, kvh int) {
	return j.qbuf[hh*j.qw : (hh+1)*j.qw], j.atf[hh*j.astride : hh*j.astride+j.an], hh / j.gqa
}

// out is head hh's output row.
func (j *attnRegion) out(hh int) []float32 { return j.obuf[hh*j.ow : (hh+1)*j.ow] }

// store copies a head's output into xb; a no-op under MLA, whose result stays
// in s.mlaAcc for the un-absorb.
func (j *attnRegion) store(hh int, out []float32) {
	if !j.mla {
		copy(j.xb[hh*j.hd:], out)
	}
}

// attnSoftmax scales a head's scores, caps them (gemma2, before the softmax)
// and normalises them with the layer's sink.
func (s *State) attnSoftmax(af []float32, hh int) {
	j := &s.rg.attn
	s.scale(af, j.scale)
	softcap(af, j.softcap)
	if j.bias != nil {
		s.idxApply(af, j.bias[(hh/j.gqa)*j.bstride:], j.w0)
	}
	s.softmaxSink(af, j.l.sinks, hh)
}

// deltaRegion is the gated delta rule's fan-out over value heads (State.delta).
type deltaRegion struct {
	st, q, k, v []float32
	// d is a Mamba-2 block's skip, one per value head; nil on the delta rule.
	d  []float32
	fn func(lo, hi int)
}

// deltaHeads is one chunk of the gated delta rule: value heads lo..hi-1.
func (s *State) deltaHeads(lo, hi int) {
	j, g := &s.rg.delta, s.dg
	for vh := lo; vh < hi; vh++ {
		kh := g.keyHead(vh)
		kv := j.k[kh*g.kDim : (kh+1)*g.kDim]
		qv := j.q[kh*g.kDim : (kh+1)*g.kDim]
		vv := j.v[vh*g.vDim : (vh+1)*g.vDim]
		o := s.dOut[vh*g.vDim : (vh+1)*g.vDim]
		mat := j.st[vh*g.vDim*g.kDim:][:g.vDim*g.kDim]
		if g.ssd {
			// k is B, q is C and v is x; dBeta holds dt (ssdAttn).
			s.jit.GatedSSD(o, mat, kv, qv, vv, s.dDecay[vh], s.dBeta[vh], j.d[vh], g.vDim, g.kDim)
			continue
		}
		if g.chanDecay {
			// The decay vector is this value head's, where k and q are the
			// key head's: rep value heads share a key and each owns its own
			// forget gate.
			dv := s.dDecay[vh*g.kDim : (vh+1)*g.kDim]
			s.jit.GatedDeltaChan(o, mat, kv, qv, vv, dv, s.dBeta[vh], g.kDim)
			continue
		}
		s.jit.GatedDelta(o, mat, kv, qv, vv, s.dDecay[vh], s.dBeta[vh], g.kDim)
	}
}

// rowOp is which per-row op a rowRegion runs.
type rowOp uint8

const (
	rowNorm   rowOp = iota // dst row = norm(src row): a LayerNorm or an RMSNorm
	rowAxpy                // dst row += alpha * b, or the table row idx[r] of b
	rowHeads               // a vision block's attention, heads lo..hi-1 (visAttendHead)
	rowPool                // gemma3's average pool of a Scale x Scale block, then its RMSNorm
	rowKeyPos              // the resampler's keys: dst row = src row + its 2-D position
	rowCross               // the resampler's cross-attention, heads lo..hi-1 (crossHeads)
	rowPrep                // a vision block's rotary and K/V write, rows lo..hi-1 (visPrepRows)
	rowLerp                // a resampled position table's rows (lerpRows)
)

// rowRegion is a region over rows -- or, for rowHeads, over heads -- whose
// arguments the caller fills before the dispatch (rowRun): the prompt's and an
// encode's per-row ops, without a closure per call.
type rowRegion struct {
	op             rowOp
	dst, src, w, b []float32
	idx            []int32
	dim            int
	ln             bool
	eps            float64
	alpha          float32
	// rowHeads: the block and its key runs; rowPool and rowKeyPos: the grid.
	li       int
	segs     []int
	side, gw int
	fn       func(lo, hi int)
}

// rowRun dispatches the row region described by s.rg.rows over n items in
// chunks of chunk, and clears its slices afterwards.
func (s *State) rowRun(n, chunk int) {
	j := &s.rg.rows
	if j.fn == nil {
		j.fn = s.rowItems
	}
	s.jit.Parallel(n, chunk, j.fn)
	j.dst, j.src, j.w, j.b, j.idx, j.segs = nil, nil, nil, nil, nil, nil
}

// rowItems is one chunk of the row region.
func (s *State) rowItems(lo, hi int) {
	j := &s.rg.rows
	d := j.dim
	switch j.op {
	case rowNorm:
		for i := lo; i < hi; i++ {
			y, x := j.dst[i*d:(i+1)*d], j.src[i*d:(i+1)*d]
			switch {
			case j.w == nil:
				copy(y, x)
			case j.ln:
				nn.LayerNorm32JIT(y, x, j.w, j.b, j.eps)
			default:
				nn.RMSNorm32JIT(y, x, j.w, j.eps)
			}
		}
	case rowAxpy:
		for i := lo; i < hi; i++ {
			b := j.b
			if j.idx != nil {
				r := int(j.idx[i])
				b = j.b[r*d : (r+1)*d]
			}
			nn.Axpy32JIT(j.dst[i*d:(i+1)*d], b, j.alpha)
		}
	case rowHeads:
		for h := lo; h < hi; h++ {
			for w := 0; w+1 < len(j.segs); w += 2 {
				s.visAttendHead(j.li, h, j.segs[w], j.segs[w+1], j.alpha)
			}
		}
	case rowPool:
		s.poolRows(lo, hi)
	case rowKeyPos:
		s.keyPosRows(lo, hi)
	case rowCross:
		s.crossHeads(lo, hi)
	case rowPrep:
		s.visPrepRows(lo, hi)
	case rowLerp:
		s.lerpRows(lo, hi)
	}
}

// embedRegion is a prompt chunk's embedding rows over the pool (State.embedRun):
// row base+i of the prompt into row i of s.bx, from src[base+i] or, where src
// is nil, ids[base+i]'s vocabulary row. err is the first row's failure.
type embedRegion struct {
	src  []func(dst []float32) error
	ids  []int32
	base int
	mu   sync.Mutex
	err  error
	fn   func(lo, hi int)
}

// embedRun fills the chunk's n rows from prompt row base on, and returns the
// first failure.
func (s *State) embedRun(src []func(dst []float32) error, ids []int32, base, n int) error {
	j := &s.rg.embed
	if j.fn == nil {
		j.fn = s.embedItems
	}
	j.src, j.ids, j.base, j.err = src, ids, base, nil
	s.jit.Parallel(n, max(1, n/(4*s.jit.Workers())), j.fn)
	err := j.err
	j.src, j.ids, j.err = nil, nil, nil
	return err
}

// embedItems is one chunk of the embedding region: rows lo..hi-1.
func (s *State) embedItems(lo, hi int) {
	j, w := &s.rg.embed, s.c.NEmbd
	for i := lo; i < hi; i++ {
		if err := s.fillRow(j.src, j.ids, j.base+i, s.bx[i*w:(i+1)*w]); err != nil {
			j.mu.Lock()
			if j.err == nil {
				j.err = err
			}
			j.mu.Unlock()
			return
		}
	}
}

// fillRow writes prompt row i into dst: src[i]'s, or ids[i]'s vocabulary row
// where src is nil.
func (s *State) fillRow(src []func(dst []float32) error, ids []int32, i int, dst []float32) error {
	if src == nil {
		return s.embedInto(dst, ids[i])
	}
	return src[i](dst)
}

// prefAttnRegion is a prompt chunk's attention fan-out over its rows
// (State.hostRows): the arguments each row reads, filled before the dispatch.
type prefAttnRegion struct {
	l                         *layer
	li, kvli, slot, p0, n     int
	qDim, qw, ow, hd, gqa     int
	astride, mstride          int
	mla, bidir                bool
	scale                     float64
	bq, bxb, qsrc, odst, attf []float32
	masks                     [][]float32
	fast                      *nn.AttnSet
	fn                        func(lo, hi int)
}

// prefAttnRows is one chunk of a prompt chunk's attention: rows lo..hi-1.
// Each row owns its own scores buffer, so heads can share it without any
// coordination, and the causal bound p0+i is what keeps this equal to the
// decode path.
func (s *State) prefAttnRows(lo, hi int) {
	j, c := &s.rg.pattn, s.c
	l, li, kvli, slot := j.l, j.li, j.kvli, j.slot
	for i := lo; i < hi; i++ {
		pos := j.p0 + i
		out := j.bxb[i*j.qDim : (i+1)*j.qDim]
		if !j.mla {
			clear(out)
		}
		qfi := j.qsrc[i*c.NHead*j.qw : (i+1)*c.NHead*j.qw]
		ofi := j.odst[i*c.NHead*j.ow : (i+1)*c.NHead*j.ow]
		for hh := 0; hh < c.NHead; hh++ {
			qh := qfi[hh*j.qw : (hh+1)*j.qw]
			if !j.mla {
				copy(qh, j.bq[i*j.qDim+hh*j.hd:])
			}
			kvh := hh / j.gqa
			// The same generated kernels and sliding window decode uses, so
			// Prefill and Forward compute the same attention.
			w0, an := c.AttnWindow(li, pos)
			// A row of a bidirectional run sees keys to the run's end; the
			// window still starts where the row's own position puts it.
			if e := s.bidirEnd(pos); e > 0 && (!c.BidirSWA || c.SWA(li)) {
				an = e - w0
			}
			if j.bidir {
				// Every position of the sequence, before this one and after it.
				w0, an = 0, j.p0+j.n
				if c.SWA(li) {
					w0, an = symmetricWindow(pos, j.p0+j.n, c.SWAWindow)
				}
			}
			af := j.attf[i*j.astride : i*j.astride+an]
			s.kvScores(j.fast, kvli, slot, kvh, qh, af, w0, an)
			// Llama 4's temperature is per row: it steps with the position.
			s.scale(af, float32(j.scale*c.AttnTemp(li, pos)))
			softcap(af, c.AttnSoftcap)
			if j.masks != nil && j.masks[i] != nil {
				s.idxApply(af, j.masks[i][kvh*j.mstride:], w0)
			}
			s.softmaxSink(af, l.sinks, hh)
			oh := ofi[hh*j.ow : (hh+1)*j.ow]
			s.kvAcc(j.fast, kvli, slot, kvh, oh, af, w0, an)
			// MLA's result stays in bmlaAcc, where the un-absorb reads it.
			if !j.mla {
				copy(out[hh*j.hd:], oh)
			}
		}
	}
}

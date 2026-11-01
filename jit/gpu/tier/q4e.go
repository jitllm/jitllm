package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// Qwen3.8-Flash-Next on the device (nn.LayerPlan.Q4E; engine/model/q4e.go is
// the graph and kernels/q4e.go the kernels it needed).
//
// The residual is HCMult streams, stream-major in bs.x as DeepSeek V4's and
// Kimi-K3's are, and bs.x2 is every stream too. A sublayer reads its
// hyper-connection's mix: each stream RMSNormed against its own row of the
// norm (x.xn, row-major), a low-rank gate over the normed streams (down,
// silu(./h), up), and the mean of sigmoid(gate) times them (x.mix), normed as
// any block's input is. Its output y goes into every stream at
// 2*sigmoid(inject(xn)/h) (Q4EPlace): bs.x into bs.x2 after the attention or
// the delta rule, bs.x2 into bs.x after the mixture.
//
// The n-gram block adds its embedding into the streams first, into x.xg,
// which the attention's mix and placement then read in place of bs.x: the
// host hashes each row's n-gram and reads the table rows (SetNGramRows), and
// the projections, the per-stream gate, the norm and the dilated convolution
// over the history in the block's recurrent tail run here. The head's mix is
// the host's, as for every model with streams.

// q4eWhyNot names a Qwen3.8-Flash-Next shape this tier declines, or "".
func q4eWhyNot(p *nn.LayerPlan) string {
	q := p.Q4E
	if q == nil {
		return ""
	}
	hd := q.HCMult * p.NEmbd
	switch {
	case q.HCMult < 1 || q.HCLowRank < 1:
		return fmt.Sprintf("hyper-connections of %d streams at rank %d", q.HCMult, q.HCLowRank)
	case q.QSARatio != 0 && qsaWhyNot(p) != "":
		return qsaWhyNot(p)
	case p.Parallel || p.AltUp != 0 || p.DS4 != nil || p.K3 != nil:
		return "hyper-connections beside another residual structure"
	case q.NGramHere && (p.Recurrent.Conv == 0 || p.Recurrent.TailLen < hd*q.NGramDil*(q.NGramConv-1)):
		return "an n-gram block with no recurrent tail to hold its convolution's history"
	}
	return ""
}

// q4eBlock is one Qwen3.8-Flash-Next block's vectors; its matrices are the
// layer's residents (slots 39..46).
type q4eBlock struct {
	// The two hyper-connections' grouped norms, HCMult rows of n_embd.
	normA, normF backend.Buf
	// The n-gram block's three grouped norms and its convolution, tap-major
	// (nil on every other block).
	ngK, ngQ, ngC, ngConv backend.Buf
	ngram                 bool
}

func (b *q4eBlock) bufs() []backend.Buf {
	if b == nil {
		return nil
	}
	return []backend.Buf{b.normA, b.normF, b.ngK, b.ngQ, b.ngC, b.ngConv}
}

// q4eSlot says whether a Qwen3.8-Flash-Next block carries slot i (39..46):
// the two hyper-connections' down, up and placement always, the n-gram
// key and value on the n-gram block.
func q4eSlot(p *nn.LayerPlan, i int) bool {
	if p.Q4E == nil {
		return false
	}
	if i >= 45 {
		return p.Q4E.NGramHere
	}
	return i >= 39
}

// prepQ4E uploads block li's Qwen3.8-Flash-Next vectors.
func (g *devTier) prepQ4E(l *layer, li int, p *nn.LayerPlan, w *nn.LayerWeights,
	up func(*backend.Buf, []float32)) string {
	q := p.Q4E
	if q == nil {
		return ""
	}
	hd := q.HCMult * p.NEmbd
	b := &q4eBlock{ngram: q.NGramHere}
	vecs := []struct {
		name string
		have []float32
		want int
		dst  *backend.Buf
	}{{"attention hyper-connection norm", w.HCAttn.Norm, hd, &b.normA},
		{"mixture hyper-connection norm", w.HCFFN.Norm, hd, &b.normF}}
	if b.ngram {
		vecs = append(vecs, []struct {
			name string
			have []float32
			want int
			dst  *backend.Buf
		}{{"n-gram key norm", w.NGram.NormKey, hd, &b.ngK},
			{"n-gram query norm", w.NGram.NormQuery, hd, &b.ngQ},
			{"n-gram convolution norm", w.NGram.NormC, hd, &b.ngC},
			{"n-gram convolution", w.NGram.Conv, hd * q.NGramConv, &b.ngConv}}...)
	}
	for _, v := range vecs {
		if len(v.have) != v.want {
			return fmt.Sprintf("block %d: the %s is %d wide, want %d", li, v.name, len(v.have), v.want)
		}
		up(v.dst, v.have)
		l.auxBytes += uint64(4 * v.want)
	}
	l.q4e = b
	return ""
}

// q4eScratch is a scratch's Qwen3.8-Flash-Next buffers and kernels (bs.q4e),
// built for its rows.
type q4eScratch struct {
	h, rank, width int
	// The hyper-connections.
	xn, gate, lo, lo2, mix  backend.Buf
	inj, w                  backend.Buf
	aHD, axHD, aLR, axLR    backend.Buf
	normS, quantHD, quantLR backend.Kernel
	silu, gateMix, injK     backend.Kernel
	place                   backend.Kernel
	// The n-gram block.
	emb, aW, axW, key, val     backend.Buf
	kn, qn, dot, gv, gvn, conv backend.Buf
	xg                         backend.Buf
	quantW, normR, streamDot   backend.Kernel
	gatedVal, ngConv, ngAdd    backend.Kernel
	// stage is the n-gram rows' padded staging.
	stage []float32
}

// initQ4E builds bs's Qwen3.8-Flash-Next set. Callers hold g.mu.
func (g *devTier) initQ4E(bs *blockScratch) error {
	p := &bs.p
	q := p.Q4E
	if q == nil {
		return nil
	}
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	R, d, h := bs.rows, p.NEmbd, q.HCMult
	hd, rank := h*d, q.HCLowRank
	actWin := p.ActWin
	if actWin <= 0 {
		actWin = 32
	}
	x := &q4eScratch{h: h, rank: rank}
	bs.q4e = x
	var err error
	comp := func(dst *backend.Kernel, mk func() (*ir.Kernel, error)) {
		if err != nil {
			return
		}
		var kk *ir.Kernel
		if kk, err = mk(); err != nil {
			return
		}
		*dst, err = g.compile(kk)
	}
	al := func(dst *backend.Buf, n int) {
		if err == nil && n > 0 {
			*dst, err = g.dev.Alloc(n * 4)
		}
	}
	eps := float32(p.RMSEps)
	place := kernels.Q4EPlaceGated
	switch q.Fault {
	case nn.Q4EFaultPlainPlace:
		place = kernels.Q4EPlacePlain
	case nn.Q4EFaultHalfPlace:
		place = kernels.Q4EPlaceHalf
	}
	comp(&x.normS, func() (*ir.Kernel, error) { return kernels.Q4EGroupNorm(d, h, R, eps, true) })
	comp(&x.quantHD, func() (*ir.Kernel, error) { return quantizeRows(R, hd, actWin) })
	comp(&x.quantLR, func() (*ir.Kernel, error) { return quantizeRows(R, rank, actWin) })
	comp(&x.silu, func() (*ir.Kernel, error) {
		return kernels.Q4EScaleSiLU(R*rank, h, q.Fault == nn.Q4EFaultUnscaledGate)
	})
	comp(&x.gateMix, func() (*ir.Kernel, error) {
		return kernels.Q4EGateMix(d, h, R, q.Fault == nn.Q4EFaultNoGate)
	})
	comp(&x.injK, func() (*ir.Kernel, error) { return kernels.Q4EInject(R*h, h, place) })
	comp(&x.place, func() (*ir.Kernel, error) { return kernels.Q4EPlace(d, h, R) })
	al(&x.xn, R*hd)
	al(&x.gate, R*hd)
	al(&x.lo, R*rank)
	al(&x.lo2, R*rank)
	al(&x.mix, R*d)
	al(&x.inj, R*h)
	al(&x.w, R*h)
	al(&x.aHD, R*pad32(hd))
	al(&x.axHD, 3*R*(pad32(hd)/32))
	al(&x.aLR, R*pad32(rank))
	al(&x.axLR, 3*R*(pad32(rank)/32))
	// The n-gram set is the model's: a scratch serves every block of its
	// shape, the n-gram block among them.
	if W := q.NGramWidth; W != 0 && p.Recurrent.TailLen != 0 {
		x.width = W
		r := p.Recurrent
		taps, dil := q.NGramConv, q.NGramDil
		// The history is the tail's last positions, after the predecessors
		// the host keeps.
		off := r.StateLen - hd*dil*(taps-1)
		comp(&x.quantW, func() (*ir.Kernel, error) { return quantizeRows(R, W, actWin) })
		comp(&x.normR, func() (*ir.Kernel, error) { return kernels.Q4EGroupNorm(d, h, R, eps, false) })
		comp(&x.streamDot, func() (*ir.Kernel, error) { return kernels.Q4EStreamDot(d, h, R) })
		comp(&x.gatedVal, func() (*ir.Kernel, error) { return kernels.Q4EGatedValue(d, h, R) })
		comp(&x.ngConv, func() (*ir.Kernel, error) {
			return kernels.Q4ENGramConv(hd, taps, dil, R, r.StateLen, off, false, q.Fault == nn.Q4EFaultNoNGramConv)
		})
		comp(&x.ngAdd, func() (*ir.Kernel, error) { return kernels.Q4ENGramAdd(d, h, R) })
		al(&x.emb, R*W)
		al(&x.aW, R*pad32(W))
		al(&x.axW, 3*R*(pad32(W)/32))
		al(&x.key, R*hd)
		al(&x.val, R*d)
		al(&x.kn, R*hd)
		al(&x.qn, R*hd)
		al(&x.dot, R*h)
		al(&x.gv, R*hd)
		al(&x.gvn, R*hd)
		al(&x.conv, R*hd)
		al(&x.xg, R*hd)
		x.stage = make([]float32, R*W)
	}
	return err
}

// free releases the set's kernels and buffers.
func (x *q4eScratch) free() {
	if x == nil {
		return
	}
	for _, k := range []backend.Kernel{x.normS, x.quantHD, x.quantLR, x.silu, x.gateMix, x.injK, x.place,
		x.quantW, x.normR, x.streamDot, x.gatedVal, x.ngConv, x.ngAdd} {
		if k != nil {
			closeKern(k)
		}
	}
	for _, b := range []backend.Buf{x.xn, x.gate, x.lo, x.lo2, x.mix, x.inj, x.w, x.aHD, x.axHD, x.aLR, x.axLR,
		x.emb, x.aW, x.axW, x.key, x.val, x.kn, x.qn, x.dot, x.gv, x.gvn, x.conv, x.xg} {
		if b != nil {
			b.Free()
		}
	}
}

// SetNGramRows takes the rows' n-gram table rows for the next call
// (nn.NGramDevice).
func (g *devTier) SetNGramRows(rows []float32) {
	g.mu.Lock()
	g.ngHost = rows
	g.mu.Unlock()
}

// SetNGramRows keeps the rows' n-gram table rows for this session's next
// call, which hands them to every device it holds (inputsTo).
func (s *gpuSession) SetNGramRows(rows []float32) { s.ng = rows }

// q4eWrites writes the call's n-gram rows, zeros past the caller's. Callers
// hold g.mu.
func (g *devTier) q4eWrites(x *q4eScratch, w func(backend.Buf, []byte)) {
	if x == nil || x.emb == nil {
		return
	}
	m := copy(x.stage, g.ngHost)
	clear(x.stage[m:])
	w(x.emb, f32b(x.stage))
}

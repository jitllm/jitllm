package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// Gemma 3n's AltUp and LAuReL on the device (nn.LayerPlan.AltUp; the graph is
// engine/model/altup.go, the kernels kernels/altup.go).
//
// bs.x holds every residual stream, stream-major (stream k of row r at
// (k*R + r)*n_embd), so the block's kernels, which read and write R*n_embd,
// run on stream 0 untouched. A block:
//
//	predict  the router on stream 0, then every stream's prediction into
//	         bs.altPred and the active one copied into stream 0
//	LAuReL   after the attention norm: L_r · L_l · h, normed, plus h, into
//	         bs.altLaur; the attention's residual add is (x + attn +
//	         laurel)/sqrt(2) (LaurelJoin) into bs.x2, never fused
//	FFN      a sparse block's gate through the gaussian top-k
//	correct  the router on the block's output, every stream corrected into
//	         bs.altB; the per-layer input gated from its active stream times
//	         the scale; AltUpFinish adds it to every other stream into bs.x
//
// The head reads the streams' mean, which the host takes: Layers never folds
// the head into an AltUp submission (the State calls the head alone).

// streamRows copies a residual of `from` rows into one of `to` rows: the
// first min(to, from) rows of every stream, each stream at its own row count
// (stream-major). With one stream it is a prefix copy.
func streamRows(dst, src []float32, p *nn.LayerPlan, to, from int) {
	if p.Streams() == 1 {
		n := min(len(dst), len(src))
		copy(dst[:n], src[:n])
		return
	}
	d, rows := p.NEmbd, min(to, from)
	for k := 0; k < p.Streams(); k++ {
		copy(dst[k*to*d:(k*to+rows)*d], src[k*from*d:(k*from+rows)*d])
	}
}

// streams is how many residual streams this device's blocks carry: AltUp's,
// or one.
func (d *devTier) streams() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.bs == nil {
		return 1
	}
	return d.bs.p.Streams()
}

// kernJob is one of a scratch's kernels and how to build it.
type kernJob struct {
	dst  *backend.Kernel
	make func() (*ir.Kernel, error)
}

// altJobs are an AltUp scratch's kernels, none where the plan has no AltUp.
func altJobs(bs *blockScratch, p *nn.LayerPlan, rows, actWin int) []kernJob {
	if p.AltUp == 0 {
		return nil
	}
	s, d, rank := p.AltUp, p.NEmbd, p.LaurelRank
	gparts := bs.gParts
	jobs := []kernJob{
		{&bs.altRoute, func() (*ir.Kernel, error) { return mainkernels.AltUpRoute(rows, d, s) }},
		{&bs.altPredictK, func() (*ir.Kernel, error) { return mainkernels.AltUpPredict(rows, d, s) }},
		{&bs.altCorrectK, func() (*ir.Kernel, error) { return mainkernels.AltUpCorrect(rows, d, s) }},
		{&bs.altFinish, func() (*ir.Kernel, error) { return mainkernels.AltUpFinish(rows, d, s) }},
		{&bs.altMul, func() (*ir.Kernel, error) { return mainkernels.MulRows(rows, d) }},
		{&bs.altCopy, func() (*ir.Kernel, error) { return mainkernels.Restride(rows, d, d, d) }},
		{&bs.laurelJoin, func() (*ir.Kernel, error) { return mainkernels.LaurelJoin(rows * d) }},
		{&bs.laurelQuant, func() (*ir.Kernel, error) { return mainkernels.Quantize(rows*rank, actWin) }},
	}
	if p.Sparse {
		jobs = append(jobs,
			kernJob{&bs.gPart, func() (*ir.Kernel, error) { return mainkernels.LayerNormPartRows(p.NFFN, gparts, rows) }},
			kernJob{&bs.gVar, func() (*ir.Kernel, error) { return mainkernels.LayerNormVarRows(p.NFFN, gparts, rows) }},
			kernJob{&bs.gApply, func() (*ir.Kernel, error) {
				return mainkernels.GaussTopKApplyRows(p.NFFN, gparts, p.SparseStd, rows)
			}})
	}
	return jobs
}

// allocAltUp allocates an AltUp scratch's buffers through al.
func allocAltUp(bs *blockScratch, p *nn.LayerPlan, rows int, al func(*backend.Buf, int)) {
	if p.AltUp == 0 {
		return
	}
	s, d := p.AltUp, p.NEmbd
	bs.gParts = partsFor(p.NFFN)
	al(&bs.altM, rows*s*4)
	al(&bs.altPred, rows*s*d*4)
	al(&bs.altB, rows*s*d*4)
	al(&bs.altLaur, rows*d*4)
	al(&bs.altLT, rows*p.LaurelRank*4)
	if p.Sparse {
		al(&bs.gPartB, rows*bs.gParts*4)
		al(&bs.gVarB, rows*bs.gParts*4)
		al(&bs.gg, rows*p.NFFN*4)
	}
}

// prepAltUp uploads block li's AltUp vectors and router. Its two LAuReL
// matrices go through prepLayer's weight list (slots 27 and 28).
func (g *devTier) prepAltUp(l *layer, li int, p *nn.LayerPlan, w *nn.LayerWeights,
	up func(*backend.Buf, []float32)) string {
	if p.AltUp == 0 {
		return ""
	}
	s, d := p.AltUp, p.NEmbd
	switch {
	case w.AltRouter.T != quant.F32 || w.AltRouter.Rows != s || w.AltRouter.K != d || len(w.AltRouter.Data) < 4*s*d:
		return fmt.Sprintf("block %d: the AltUp router is %v %dx%d, want an F32 %dx%d", li, w.AltRouter.T,
			w.AltRouter.Rows, w.AltRouter.K, s, d)
	case len(w.AltRouterNorm) != d || len(w.AltPredT) != s*s*s || len(w.AltCorrT) != s*s ||
		len(w.AltCorrScale) != d || len(w.LaurelPost) != d:
		return fmt.Sprintf("block %d: AltUp's vectors are %d, %d, %d, %d and LAuReL's norm %d", li,
			len(w.AltRouterNorm), len(w.AltPredT), len(w.AltCorrT), len(w.AltCorrScale), len(w.LaurelPost))
	}
	up(&l.altRouterNorm, w.AltRouterNorm)
	up(&l.altPredT, w.AltPredT)
	up(&l.altCorrT, w.AltCorrT)
	up(&l.altCorrScale, w.AltCorrScale)
	up(&l.nLaurel, w.LaurelPost)
	up(&l.altRouter, f32of(w.AltRouter.Data[:4*s*d]))
	l.sparse = p.Sparse
	l.auxBytes += uint64(4 * (3*d + s*s*s + s*s + s*d))
	return ""
}

// emitAltPredict is a block's prologue: the predictions into bs.altPred and
// the active one into stream 0 of bs.x.
func (g *devTier) emitAltPredict(lc *launcher, bs *blockScratch, l *layer, R int,
	norm func(src, w, wb backend.Buf)) {
	p := &bs.p
	norm(bs.x, l.altRouterNorm, nil)
	lc.la(bs.altRoute, R*p.AltUp, bs.h, l.altRouter, bs.altM)
	lc.la(bs.altPredictK, R*p.AltUp*p.NEmbd, bs.x, bs.altM, l.altPredT, bs.altPred)
	lc.la(bs.altCopy, R*p.NEmbd, bs.altPred, bs.x)
	g.AltUpBlocks++
}

// emitLaurel writes LAuReL's branch of bs.h into bs.altLaur and leaves bs.a
// holding bs.h's quantization again, as the attention expects. fsrc is the
// float row a float weight's matvec reads in place of the int8 one: set
// before each of the branch's two, and left at bs.h.
func (g *devTier) emitLaurel(lc *launcher, bs *blockScratch, l *layer, R int,
	mvrun func(mv, *resident, backend.Buf), norm func(src, w, wb backend.Buf), fsrc *backend.Buf) {
	p := &bs.p
	*fsrc = bs.h
	mvrun(l.mvLL, l.laurelL, bs.altLT)
	lc.la(bs.laurelQuant, mainkernels.QuantizeThreads(R*p.LaurelRank/32), bs.altLT, bs.a, bs.ax)
	*fsrc = bs.altLT
	mvrun(l.mvLR, l.laurelR, bs.mvOut)
	// norm writes bs.h, which the attention reads: the branch's norm goes
	// through bs.altB's first plane and comes back after.
	lc.la(bs.altCopy, R*p.NEmbd, bs.h, bs.altB)
	norm(bs.mvOut, l.nLaurel, nil)
	lc.la(bs.addE, R*p.NEmbd, bs.h, bs.altB, bs.altLaur)
	lc.la(bs.altCopy, R*p.NEmbd, bs.altB, bs.h)
	lc.la(bs.quantE, mainkernels.QuantizeThreads(R*ePad(p)/32), bs.h, bs.a, bs.ax)
	*fsrc = bs.h
}

// emitGauss runs a sparse block's gaussian top-k on its gate bs.g, into
// bs.gg, and returns the buffer the FFN's activation reads.
func (g *devTier) emitGauss(lc *launcher, bs *blockScratch, l *layer, R int) backend.Buf {
	if !l.sparse || bs.gApply == nil {
		return bs.g
	}
	p := &bs.p
	lc.la(bs.gPart, R*bs.gParts, bs.g, bs.gPartB)
	lc.la(bs.gVar, R*bs.gParts, bs.g, bs.gPartB, bs.gVarB)
	lc.la(bs.gApply, R*p.NFFN, bs.g, bs.gPartB, bs.gVarB, bs.gg)
	return bs.gg
}

// emitAltCorrect is a block's epilogue, with the block's output in stream 0
// of bs.x: every stream corrected, and the per-layer input gated into the
// others, back into bs.x.
func (g *devTier) emitAltCorrect(lc *launcher, bs *blockScratch, l *layer, R int,
	mvrun func(mv, *resident, backend.Buf), norm func(src, w, wb backend.Buf), fsrc *backend.Buf) {
	p := &bs.p
	norm(bs.x, l.altRouterNorm, nil)
	lc.la(bs.altRoute, R*p.AltUp, bs.h, l.altRouter, bs.altM)
	lc.la(bs.altCorrectK, R*p.AltUp*p.NEmbd, bs.altPred, bs.x, bs.altM, l.altCorrT, bs.altB)
	lc.la(bs.altMul, R*p.NEmbd, bs.altB, l.altCorrScale, bs.h)
	lc.la(bs.quantE, mainkernels.QuantizeThreads(R*ePad(p)/32), bs.h, bs.a, bs.ax)
	*fsrc = bs.h
	mvrun(l.mvpg, l.pleGate, bs.pleG)
	lc.la(bs.pleSlice, R*p.PLEDim, bs.pleIn, l.pleOff, bs.pleS)
	lc.la(bs.pleAct, R*p.PLEDim, bs.pleG, bs.pleS, bs.pleA)
	lc.la(bs.pleQuant, mainkernels.QuantizeThreads(R*pad32(p.PLEDim)/32), bs.pleA, bs.a, bs.ax)
	*fsrc = bs.pleA
	mvrun(l.mvpp, l.pleProj, bs.pleO)
	norm(bs.pleO, l.nPLE, nil)
	lc.la(bs.altFinish, R*p.AltUp*p.NEmbd, bs.altB, bs.h, bs.x)
}

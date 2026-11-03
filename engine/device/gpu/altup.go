package gpu

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/quant"
)

// Gemma 3n's AltUp and LAuReL on the device (nn.LayerPlan.AltUp; the graph is
// engine/model/altup.go).
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
// the head into an AltUp submission (the instance calls the head alone).

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

// prepAltUp uploads block li's AltUp router. Its two LAuReL matrices go
// through prepLayer's weight list (slots 27 and 28).
func (g *devTier) prepAltUp(l *layer, li int, p *nn.LayerPlan, w *nn.LayerWeights,
	up func(*backend.Buf, []float32)) string {
	if p.AltUp == 0 {
		return ""
	}
	s, d := p.AltUp, p.NEmbd
	if w.AltRouter.T != quant.F32 || w.AltRouter.Rows != s || w.AltRouter.K != d || len(w.AltRouter.Data) < 4*s*d {
		return fmt.Sprintf("block %d: the AltUp router is %v %dx%d, want an F32 %dx%d", li, w.AltRouter.T,
			w.AltRouter.Rows, w.AltRouter.K, s, d)
	}
	up(&l.altRouter, f32of(w.AltRouter.Data[:4*s*d]))
	// The composer reads the router as a plain float matrix (roleWeight).
	if q, ok := quantOf(quant.F32); ok && l.altRouter != nil {
		l.altRouterW = &resident{qs: l.altRouter, nrows: s, k: d, t: q, ok: true}
	}
	l.sparse = p.Sparse
	l.auxBytes += uint64(4 * s * d)
	return ""
}

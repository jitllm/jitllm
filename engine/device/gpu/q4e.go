package gpu

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/nn"
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
// which the attention's mix and placement then read in place of bs.x: its
// graph gathers each row's rows of the n-gram table, which no card holds
// (28.8 GB on the release), so the block runs where the table's rows are
// read; the projections, the per-stream gate, the norm and the dilated
// convolution over the history in the block's recurrent tail are its
// card form's once a card gathers them. The head's mix is
// the host's, as for every model with streams.

// q4eBlock is one Qwen3.8-Flash-Next block's n-gram convolution; its
// matrices are the layer's residents (slots 34..41) and its vectors are in
// the layer's vecs.
type q4eBlock struct {
	// ngConv is the n-gram block's convolution, a channel's taps contiguous
	// as the file stores them (nil on every other block).
	ngConv backend.Buf
	// ngTaps is ngConv as the composer binds it, a row a channel (the
	// buffer is ngConv's).
	ngTaps resident
	ngram  bool
}

func (b *q4eBlock) bufs() []backend.Buf {
	if b == nil {
		return nil
	}
	return []backend.Buf{b.ngConv}
}

// q4eSlot says whether a Qwen3.8-Flash-Next block carries slot i (34..41):
// the two hyper-connections' down, up and placement always, the n-gram
// key and value on the n-gram block.
func q4eSlot(p *nn.LayerPlan, i int) bool {
	if p.Q4E == nil {
		return false
	}
	if i >= 40 {
		return p.Q4E.NGramHere
	}
	return i >= 34
}

// prepQ4E uploads block li's n-gram convolution, which the composer binds
// as a matrix.
func (g *devTier) prepQ4E(l *layer, li int, p *nn.LayerPlan, w *nn.LayerWeights,
	up func(*backend.Buf, []float32)) string {
	q := p.Q4E
	if q == nil {
		return ""
	}
	b := &q4eBlock{ngram: q.NGramHere}
	if b.ngram {
		hd := q.HCMult * p.NEmbd
		if len(w.NGram.Conv) != hd*q.NGramConv {
			return fmt.Sprintf("block %d: the n-gram convolution is %d wide, want %d", li, len(w.NGram.Conv),
				hd*q.NGramConv)
		}
		up(&b.ngConv, w.NGram.Conv)
		l.auxBytes += uint64(4 * len(w.NGram.Conv))
		b.ngTaps = resident{qs: b.ngConv, nrows: hd, k: q.NGramConv, t: layout.Float32, ok: true}
	}
	l.q4e = b
	return ""
}

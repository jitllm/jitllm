package gpu

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer/kernels"
	"github.com/jitllm/jitllm/engine/nn"
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

// q4eBlock is one Qwen3.8-Flash-Next block's vectors; its matrices are the
// layer's residents (slots 39..46).
type q4eBlock struct {
	// The two hyper-connections' grouped norms, HCMult rows of n_embd.
	normA, normF backend.Buf
	// The n-gram block's three grouped norms and its convolution, tap-major
	// (nil on every other block).
	ngK, ngQ, ngC, ngConv backend.Buf
	// ngTaps is ngConv as the composer binds it: a channel's taps
	// contiguous, as the file stores them (the buffer is ngConv's).
	ngTaps resident
	ngram  bool
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
	if b.ngram {
		b.ngTaps = resident{qs: b.ngConv, nrows: hd, k: q.NGramConv, t: kernels.Float32, ok: true}
	}
	l.q4e = b
	return ""
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

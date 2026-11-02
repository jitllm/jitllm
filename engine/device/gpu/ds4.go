package gpu

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer/kernels"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/quant"
)

// DeepSeek V4 on the device (nn.LayerPlan.DS4; engine/model/ds4.go is the
// graph and kernels/ds4.go the kernels it needed).
//
// The residual is the hyper-connections' streams, stream-major in bs.x as
// Gemma 3n's are (altup.go), and each sublayer is wrapped: ds4Pre turns the
// streams into the sublayer's normed input in bs.h (quantized for the
// matvecs), ds4Post the sublayer's output and the streams before it into the
// streams after -- bs.x into bs.x2 around the attention, bs.x2 into bs.x around
// the mixture. The head reads the streams' collapse, which the host takes:
// Layers never folds the head into these blocks, as for AltUp.
//
// The attention (ds4Attn) caches each position's row -- the key and, on a
// compressed block, the compressor's projections of the position -- in the
// paged pool as a latent row (one head that is also the value), folds an
// entry wherever a row closes one into the block's second pool (entKey), and
// gathers each row's keys -- its window, then the entries it reads -- into
// scratch pages the paged attention reads with the heads' sinks, as DeepSeek
// V3.2's indexer gathers its selection (indexer.go). A CSA block's entries
// are chosen by the indexer's scores over the entries' own keys. A history
// with evicted pages is refused by the call: the window and the entries are
// read through their tables. The pools never evict (kvLayerPool.noEvict).
//
// The mixture is the generic one with two changes: the router gates with
// sqrt(softplus) and every block selects with a per-row bias plane
// (kernels.MoERoute.RowBias) -- the block's selection bias copied to each row,
// or on a block that routes by token, zero at the experts of the row's token
// (SetTokenIDs) and -inf elsewhere.

// entKey is the pool key of block li's entries: a second history beside the
// block's rows, keyed apart from every block index.
func entKey(li int) int { return -1 - li }

// ds4Block is one DeepSeek V4 block's vectors beyond the common ones; its
// matrices are the layer's residents (woA, compKV, ...).
type ds4Block struct {
	comp, rate  int
	row, entRow int
	hash        bool
	// The two hyper-connections' fn (F32) and per-mix scale and base.
	fnA, fnF, scA, bA, scF, bF backend.Buf
	// The compressors' position biases and norms.
	ape, norm, iape, inorm backend.Buf
	// table is a hash block's token table, ids as floats.
	table backend.Buf
	// The F32 tables as a composed block binds them: the hyper-connections'
	// fn, the position biases (a row a place in the window) and the token
	// table (a row a token).
	fnAR, fnFR, apeR, iapeR, tableR resident
}

func (b *ds4Block) bufs() []backend.Buf {
	if b == nil {
		return nil
	}
	return []backend.Buf{b.fnA, b.fnF, b.scA, b.bA, b.scF, b.bF, b.ape, b.norm, b.iape, b.inorm, b.table}
}

// prepDS4 uploads block li's DeepSeek V4 vectors and fn. Its matrices go
// through prepLayer's weight list (slots 31..35).
func (g *devTier) prepDS4(l *layer, li int, p *nn.LayerPlan, w *nn.LayerWeights,
	up func(*backend.Buf, []float32)) string {
	d := p.DS4
	if d == nil {
		return ""
	}
	if !g.pagedPlan(p) {
		return fmt.Sprintf("block %d: DeepSeek V4's window and entries are read through the paged pool, "+
			"which this device does not keep", li)
	}
	s, n := d.HCMult, p.NEmbd
	mix := (2 + s) * s
	b := &ds4Block{comp: p.Comp, rate: d.RateAt(p.Comp), row: p.KVRow(), entRow: d.EntRowAt(p.Comp, p.HeadDim),
		hash: p.HashExperts}
	fn := func(x nn.Weight, what string) ([]float32, string) {
		if x.T != quant.F32 || x.Rows != mix || x.K != s*n || len(x.Data) < 4*mix*s*n {
			return nil, fmt.Sprintf("block %d: the %s hyper-connection is %v %dx%d, want an F32 %dx%d", li, what,
				x.T, x.Rows, x.K, mix, s*n)
		}
		return f32of(x.Data[:4*mix*s*n]), ""
	}
	fa, why := fn(w.HCAttnFn, "attention's")
	if why != "" {
		return why
	}
	ff, why := fn(w.HCFFNFn, "mixture's")
	if why != "" {
		return why
	}
	hd, ihd := p.HeadDim, d.IdxHeadDim
	coff := map[int]int{nn.DS4CompHCA: 1, nn.DS4CompCSA: 2}[p.Comp]
	vecs := []struct {
		name string
		have []float32
		want int
		dst  *backend.Buf
	}{
		{"attention scale", w.HCAttnScale, 3, &b.scA}, {"attention base", w.HCAttnBase, mix, &b.bA},
		{"mixture scale", w.HCFFNScale, 3, &b.scF}, {"mixture base", w.HCFFNBase, mix, &b.bF},
		{"query latent norm", w.QANorm, p.QLoraRank, &l.nQA}, {"key norm", w.KVANorm, hd, &l.nKVA},
	}
	if p.Comp != nn.DS4CompNone {
		vecs = append(vecs, struct {
			name string
			have []float32
			want int
			dst  *backend.Buf
		}{"compressor position bias", w.CompAPE, b.rate * coff * hd, &b.ape}, struct {
			name string
			have []float32
			want int
			dst  *backend.Buf
		}{"compressor norm", w.CompNorm, hd, &b.norm})
	}
	if p.Comp == nn.DS4CompCSA {
		vecs = append(vecs, struct {
			name string
			have []float32
			want int
			dst  *backend.Buf
		}{"indexer compressor position bias", w.IdxCompAPE, b.rate * 2 * ihd, &b.iape}, struct {
			name string
			have []float32
			want int
			dst  *backend.Buf
		}{"indexer compressor norm", w.IdxCompNorm, ihd, &b.inorm})
	}
	if p.HashExperts {
		if p.Vocab <= 0 || len(w.HashExperts) != p.Vocab*p.NExpertUsed {
			return fmt.Sprintf("block %d: the token table holds %d ids, want %d tokens of %d", li,
				len(w.HashExperts), p.Vocab, p.NExpertUsed)
		}
		vecs = append(vecs, struct {
			name string
			have []float32
			want int
			dst  *backend.Buf
		}{"token table", w.HashExperts, len(w.HashExperts), &b.table})
	}
	for _, v := range vecs {
		if len(v.have) != v.want {
			return fmt.Sprintf("block %d: the %s is %d wide, want %d", li, v.name, len(v.have), v.want)
		}
		up(v.dst, v.have)
		l.auxBytes += uint64(4 * v.want)
	}
	up(&b.fnA, fa)
	up(&b.fnF, ff)
	l.auxBytes += uint64(2 * 4 * len(fa))
	f32r := func(qs backend.Buf, rows, k int) resident {
		return resident{qs: qs, nrows: rows, k: k, t: kernels.Float32, ok: qs != nil}
	}
	b.fnAR, b.fnFR = f32r(b.fnA, mix, s*n), f32r(b.fnF, mix, s*n)
	b.apeR, b.iapeR = f32r(b.ape, b.rate, coff*hd), f32r(b.iape, b.rate, 2*ihd)
	b.tableR = f32r(b.table, p.Vocab, p.NExpertUsed)
	l.ds4 = b
	return ""
}

// ds4Scratch is a scratch's DeepSeek V4 buffers and kernels (bs.ds4), built
// for its rows and every kind of block the model has.
type ds4Scratch struct {
	kmax int
}

// ds4KeyCap is the paged attention's key bound for a DeepSeek V4 scratch: a
// row reads at most kmax gathered keys whatever its position.
func ds4KeyCap(bs *blockScratch, keys int) int {
	if bs.ds4 == nil {
		return keys
	}
	return min(keys, (bs.ds4.kmax+scoreGrain-1)/scoreGrain*scoreGrain)
}

// ds4Slot reports whether slot i of the weight list (weightList) is one a
// DeepSeek V4 block carries: the key, o_b, the mixture and shared expert, the
// query latent's two halves, the grouped output's bank, and per compression
// kind the compressors (and on CSA the indexer's query and weights).
func ds4Slot(p *nn.LayerPlan, i int) bool {
	switch i {
	case 1, 3, 4, 5, 6, 7, 8, 9, 13, 14, 31:
		return true
	case 32, 33:
		return p.Comp != nn.DS4CompNone
	case 24, 26, 34, 35:
		return p.Comp == nn.DS4CompCSA
	}
	return false
}

// ds4Twins compiles block l's grouped output at R rows -- the indexed
// matvec with a slot a group per row -- outside any session, so the session
// looks it up (ds4WoA). Callers hold g.mu.
func (g *devTier) ds4Twins(l *layer, R int) bool {
	if l.ds4 == nil || R <= 1 {
		return true
	}
	if _, ok := l.woAW[R]; ok {
		return true
	}
	m := l.mvWoA
	c, err := g.idKernel(kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, Split: 1,
		Experts: m.experts, Slots: R * m.experts, SlotAct: true, Sub: subFor(m.q, m.k)})
	if err != nil {
		g.LastErr = fmt.Sprintf("tier: DeepSeek V4's grouped output at %d rows: %v", R, err)
		return false
	}
	if l.woAW == nil {
		l.woAW = map[int]mv{}
	}
	l.woAW[R] = mv{kern: c, rows: m.rows, slots: R * m.experts, split: 1, q: m.q, k: m.k,
		experts: m.experts, slotAct: true}
	return true
}

// SetTokenIDs takes the rows' token ids for the next call (nn.TokenDevice).
func (g *devTier) SetTokenIDs(ids []int32) {
	g.mu.Lock()
	g.tokIDs = ids
	g.mu.Unlock()
}

// SetTokenIDs keeps the rows' token ids for this session's next call, which
// hands them to every device it holds (inputsTo).
func (s *gpuSession) SetTokenIDs(ids []int32) { s.ids = ids }

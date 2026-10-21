package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
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
// (mainkernels.MoERoute.RowBias) -- the block's selection bias copied to each row,
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
		{"attention scale", w.HCAttnScale, mix, &b.scA}, {"attention base", w.HCAttnBase, mix, &b.bA},
		{"mixture scale", w.HCFFNScale, mix, &b.scF}, {"mixture base", w.HCFFNBase, mix, &b.bF},
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
	l.ds4 = b
	return ""
}

// ds4Scratch is a scratch's DeepSeek V4 buffers and kernels (bs.ds4), built
// for its rows and every kind of block the model has.
type ds4Scratch struct {
	rows, kmax, ppr int
	// The hyper-connections.
	flat, flatn, mix, hc, coll backend.Buf
	onesFlat, onesHD           backend.Buf
	hcFlat, hcRms, hcFn        backend.Kernel
	hcMix, hcColl, hcPost      backend.Kernel
	// The attention's projections, norms and rotations.
	qa, qan, q, qn, qr  backend.Buf
	kraw, kn, kr        backend.Buf
	ckv, cg, cgA        backend.Buf
	ikv, ig, igA, iw    backend.Buf
	iq, iqr             backend.Buf
	pe, pen, per        backend.Buf
	ipe, ipen, iper     backend.Buf
	rmsQ, rmsHeads      backend.Kernel
	rmsHD, rmsIHD       backend.Kernel
	quantQA, quantOA    backend.Kernel
	ropeQ, ropeK, ropeI backend.Kernel
	ropeIQ, ropeIE      backend.Kernel
	// Per compression kind (nn.DS4Comp*): the row's writers, the bias adds,
	// the pools, the entry writers, the gather and its descriptors, and each
	// kind's entry rotary table and its positions.
	kinds [3]*ds4Kind
	// The indexer: the entries' descriptors, the scores and the selection.
	edesc, score, sel     backend.Buf
	entDesc, idxSc, idxSl backend.Kernel
	// The gathered keys, their table (the identity) and descriptors, the
	// attention's output, it rotated back, the groups' outputs and their
	// selection (the identity, per row).
	gk, gtab, gdesc backend.Buf
	ao, aor, oa     backend.Buf
	oSel            backend.Buf
	// The router's per-row selection bias, the zeros a score-routed hash
	// block reads under the fault, and the rows' token ids.
	rbias, zeroB, ids    backend.Buf
	routeHash, routeCopy backend.Kernel
	noSink               backend.Buf
	// idsStage is the token ids' per-call staging.
	idsStage []uint32
}

// ds4Kind is one compression kind's kernels in a scratch.
type ds4Kind struct {
	rate, row, ent, coff   int
	wK, wKV, wG, wIKV, wIG backend.Kernel
	ape, apeI              backend.Kernel
	pool, poolI            backend.Kernel
	entW, entWI            backend.Kernel
	gather, desc           backend.Kernel
	csE, erpos             backend.Buf
	posStage               []uint32
}

// free releases the set's kernels and buffers.
func (x *ds4Scratch) free(g *devShared) {
	if x == nil {
		return
	}
	ks := []backend.Kernel{x.hcFlat, x.hcRms, x.hcFn, x.hcMix, x.hcColl, x.hcPost, x.rmsQ, x.rmsHeads,
		x.rmsHD, x.rmsIHD, x.quantQA, x.quantOA, x.ropeQ, x.ropeK, x.ropeI, x.ropeIQ, x.ropeIE,
		x.entDesc, x.idxSc, x.idxSl, x.routeHash, x.routeCopy}
	bs := []backend.Buf{x.flat, x.flatn, x.mix, x.hc, x.coll, x.onesFlat, x.onesHD, x.qa, x.qan, x.q, x.qn,
		x.qr, x.kraw, x.kn, x.kr, x.ckv, x.cg, x.cgA, x.ikv, x.ig, x.igA, x.iw, x.iq, x.iqr, x.pe, x.pen,
		x.per, x.ipe, x.ipen, x.iper, x.edesc, x.score, x.sel, x.gk, x.gtab, x.gdesc, x.ao, x.aor, x.oa,
		x.oSel, x.rbias, x.zeroB, x.ids, x.noSink}
	for _, k := range x.kinds {
		if k == nil {
			continue
		}
		ks = append(ks, k.wK, k.wKV, k.wG, k.wIKV, k.wIG, k.ape, k.apeI, k.pool, k.poolI, k.entW, k.entWI,
			k.gather, k.desc)
		bs = append(bs, k.csE, k.erpos)
	}
	for _, k := range ks {
		if k != nil {
			g.closeKern(k)
		}
	}
	for _, b := range bs {
		if b != nil {
			b.Free()
		}
	}
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

// ds4WoA is block l's grouped output at R rows, or false where ds4Twins has
// not built it.
func (l *layer) ds4WoA(R int) (mv, bool) {
	if R == 1 {
		return l.mvWoA, true
	}
	m, ok := l.woAW[R]
	return m, ok
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

// ds4Stage is the call's DeepSeek V4 staging: each row's token id and, per
// compressed kind, the position of the entry the row closes (its rotary
// table's), written with the call's other per-token writes. A padded row
// stages zeros. It refuses a supplied embedding on a block that routes by
// token. Callers hold g.mu.
func (g *devTier) ds4Stage(bs *blockScratch, lo, hi int, rows []pagedRow) error {
	x := bs.ds4
	if x == nil {
		return nil
	}
	hash := false
	for li := lo; li < hi; li++ {
		if l := g.layers[li]; l != nil && l.ds4 != nil && l.ds4.hash {
			hash = true
		}
	}
	for r := range x.idsStage {
		x.idsStage[r] = 0
		if r >= len(rows) || rows[r].pad {
			continue
		}
		if hash {
			if r >= len(g.tokIDs) || g.tokIDs[r] < 0 {
				return fmt.Errorf("row %d has no token id, and a block routes by token (SetTokenIDs)", r)
			}
			x.idsStage[r] = uint32(g.tokIDs[r])
		}
	}
	for _, k := range x.kinds {
		if k == nil || k.rate == 0 {
			continue
		}
		k.posStage[0] = uint32(len(rows))
		for r := range rows {
			at := 0
			if !rows[r].pad {
				at = max((rows[r].pos+1)/k.rate-1, 0) * k.rate
			}
			k.posStage[r+1] = uint32(at)
		}
	}
	return nil
}

// ds4Writes writes the staging ds4Stage filled.
func (x *ds4Scratch) ds4Writes(w func(backend.Buf, []byte)) {
	if x == nil {
		return
	}
	w(x.ids, u32view(x.idsStage))
	for _, k := range x.kinds {
		if k != nil && k.erpos != nil {
			w(k.erpos, u32view(k.posStage))
		}
	}
}

// ds4Tables builds the call's entry rotary tables on the device, one per
// compressed kind, from the positions ds4Writes staged.
func (g *devTier) ds4Tables(lc *launcher, bs *blockScratch, R int) {
	x := bs.ds4
	if x == nil {
		return
	}
	for _, k := range x.kinds {
		if k != nil && k.csE != nil {
			lc.la(bs.ropeTable, R*bs.ropeNPairs, k.csE, bs.ropeTab, bs.ropeConst, k.erpos)
		}
	}
}

// ds4Post is a hyper-connection's placement: the streams after the sublayer,
// from its output y and the streams before it (src), into dst.
func (g *devTier) ds4Post(lc *launcher, bs *blockScratch, R int, src, y, dst backend.Buf) {
	lc.la(bs.ds4.hcPost, bs.p.DS4.HCMult*R*bs.p.NEmbd, src, y, bs.ds4.hc, dst)
}

// ds4RouteBias writes the R rows' selection bias for block l's router and
// returns it.
func (g *devTier) ds4RouteBias(lc *launcher, bs *blockScratch, l *layer, R int) backend.Buf {
	x, p := bs.ds4, &bs.p
	switch {
	case l.ds4.hash && p.DS4.Fault == nn.DS4FaultScoreRoute:
		lc.la(x.routeCopy, R*p.NExpert, x.zeroB, x.rbias)
	case l.ds4.hash:
		lc.la(x.routeHash, R*p.NExpert, l.ds4.table, x.ids, x.rbias)
	default:
		lc.la(x.routeCopy, R*p.NExpert, l.expSelB, x.rbias)
	}
	return x.rbias
}

// ds4KMax is the most keys one row of kind comp reads at a plan of maxSeq.
func ds4KMax(d *nn.DS4Plan, comp, maxSeq int) int {
	n := d.Window
	switch {
	case comp == nn.DS4CompNone || d.Fault == nn.DS4FaultWindowOnly:
	case comp == nn.DS4CompHCA || d.Fault == nn.DS4FaultAllEntries:
		n += maxSeq/d.RateAt(comp) + 1
	default:
		n += d.IdxTopK
	}
	return n
}

package maintier

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	kernels "github.com/jitllm/jitllm/paritytest/mainkernels"
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

// ds4WhyNot names a DeepSeek V4 shape this tier declines, or "".
func ds4WhyNot(p *nn.LayerPlan) string {
	d := p.DS4
	if d == nil {
		return ""
	}
	gw := 0
	if d.OGroups > 0 {
		gw = p.NHead / d.OGroups * p.HeadDim
	}
	switch {
	case d.HCMult != kernels.DS4Streams:
		return fmt.Sprintf("DeepSeek V4 with %d hyper-connection streams (the mixer is unrolled for %d)",
			d.HCMult, kernels.DS4Streams)
	case p.NKVHead != 1 || p.QLoraRank == 0 || p.MLA() || d.Window <= 0 || p.RopeNeox ||
		p.NRot <= 0 || p.NRot%2 != 0 || p.NRot > p.HeadDim:
		return "DeepSeek V4 attention of a shape the kernels are not built for"
	case d.OGroups <= 0 || p.NHead%d.OGroups != 0 || p.QLoraRank%32 != 0 || gw%32 != 0 ||
		(d.OGroups*d.OLoraRank)%32 != 0:
		return fmt.Sprintf("DeepSeek V4's query latent of %d and output groups of %d into %d: the "+
			"activation quantizer works in 32-element blocks", p.QLoraRank, gw, d.OLoraRank)
	case p.Comp == nn.DS4CompCSA && (d.IdxHeads == 0 || d.IdxTopK == 0 || d.IdxHeadDim < p.NRot ||
		d.RateAt(p.Comp) == 0):
		return "a CSA block with no indexer"
	case p.Comp == nn.DS4CompHCA && d.RateAt(p.Comp) == 0:
		return "an HCA block with no rate"
	case p.NExpert == 0 || p.NFFNShExp == 0:
		return "a DeepSeek V4 block without its mixture"
	}
	return ""
}

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

// ds4Keys is a kind's key choice under the plan's fault.
func ds4Keys(d *nn.DS4Plan, comp int) kernels.DS4Keys {
	k := kernels.DS4Keys{Window: d.Window, Rate: d.RateAt(comp)}
	switch {
	case comp == nn.DS4CompNone || d.Fault == nn.DS4FaultWindowOnly:
		k.None = true
	case comp == nn.DS4CompHCA || d.Fault == nn.DS4FaultAllEntries:
		k.All = true
	default:
		k.TopK = d.IdxTopK
	}
	return k
}

// ds4KMax is the most keys one row of kind comp reads at a plan of maxSeq.
func ds4KMax(d *nn.DS4Plan, comp, maxSeq int) int {
	k := ds4Keys(d, comp)
	n := d.Window
	switch {
	case k.None:
	case k.All:
		n += maxSeq/k.Rate + 1
	default:
		n += k.TopK
	}
	return n
}

// initDS4 builds bs's DeepSeek V4 set. Callers hold g.mu.
func (g *devTier) initDS4(bs *blockScratch) error {
	p := &bs.p
	d := p.DS4
	if d == nil {
		return nil
	}
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	R, P := bs.rows, g.kvPage()
	s, n := d.HCMult, p.NEmbd
	mix := (2 + s) * s
	H, hd, nr := p.NHead, p.HeadDim, p.NRot
	ih, ihd := d.IdxHeads, d.IdxHeadDim
	actWin := p.ActWin
	if actWin <= 0 {
		actWin = 32
	}
	x := &ds4Scratch{rows: R}
	bs.ds4 = x
	for _, comp := range []int{nn.DS4CompNone, nn.DS4CompCSA, nn.DS4CompHCA} {
		x.kmax = max(x.kmax, ds4KMax(d, comp, p.MaxSeq))
	}
	x.ppr = kernels.DS4GatherPages(x.kmax, P)
	var err error
	comp := func(dst *backend.Kernel, mk func() (*ir.Kernel, error)) {
		if err != nil {
			return
		}
		var k *ir.Kernel
		if k, err = mk(); err != nil {
			return
		}
		*dst, err = g.dev.Compile(k)
	}
	al := func(dst *backend.Buf, n int) {
		if err == nil && n > 0 {
			*dst, err = g.dev.Alloc(n * 4)
		}
	}
	iters := d.HCIters
	if d.Fault == nn.DS4FaultCombSoftmax {
		iters = 0
	}
	comp(&x.hcFlat, func() (*ir.Kernel, error) { return kernels.HCFlat(n, R) })
	comp(&x.hcRms, func() (*ir.Kernel, error) { return kernels.RMSNormRows(s*n, R, float32(p.RMSEps), false, false) })
	comp(&x.hcFn, func() (*ir.Kernel, error) { return kernels.RouterMatVecTok(mix, s*n, R, false) })
	comp(&x.hcMix, func() (*ir.Kernel, error) { return kernels.HCMix(R, iters, d.HCEps) })
	comp(&x.hcColl, func() (*ir.Kernel, error) { return kernels.HCCollapse(n, R) })
	comp(&x.hcPost, func() (*ir.Kernel, error) { return kernels.HCPost(n, R, d.Fault == nn.DS4FaultCombRows) })
	eps := float32(p.RMSEps)
	comp(&x.rmsQ, func() (*ir.Kernel, error) { return kernels.RMSNormRows(p.QLoraRank, R, eps, false, false) })
	comp(&x.rmsHeads, func() (*ir.Kernel, error) { return kernels.RMSNormRows(hd, R*H, eps, false, false) })
	comp(&x.rmsHD, func() (*ir.Kernel, error) { return kernels.RMSNormRows(hd, R, eps, false, false) })
	comp(&x.quantQA, func() (*ir.Kernel, error) { return kernels.Quantize(R*p.QLoraRank, actWin) })
	comp(&x.quantOA, func() (*ir.Kernel, error) { return kernels.Quantize(R*d.OGroups*d.OLoraRank, actWin) })
	comp(&x.ropeQ, func() (*ir.Kernel, error) { return kernels.RoPETail(H, hd, nr, R, false) })
	comp(&x.ropeK, func() (*ir.Kernel, error) { return kernels.RoPETail(1, hd, nr, R, false) })
	comp(&x.ropeI, func() (*ir.Kernel, error) { return kernels.RoPETail(H, hd, nr, R, true) })
	if ih > 0 {
		comp(&x.rmsIHD, func() (*ir.Kernel, error) { return kernels.RMSNormRows(ihd, R, eps, false, false) })
		comp(&x.ropeIQ, func() (*ir.Kernel, error) { return kernels.RoPETail(ih, ihd, nr, R, false) })
		comp(&x.ropeIE, func() (*ir.Kernel, error) { return kernels.RoPETail(1, ihd, nr, R, false) })
	}
	// The indexer's score scale, a load-time constant (1/sqrt of its heads
	// times their width, as the host's).
	idxScale := float32(1 / math.Sqrt(float64(max(ihd*ih, 1))))
	for _, c := range []int{nn.DS4CompNone, nn.DS4CompCSA, nn.DS4CompHCA} {
		rate := d.RateAt(c)
		if c != nn.DS4CompNone && rate == 0 {
			continue
		}
		k := &ds4Kind{rate: rate, row: d.RowAt(c, hd), ent: d.EntRowAt(c, hd)}
		k.coff = map[int]int{nn.DS4CompHCA: 1, nn.DS4CompCSA: 2}[c]
		x.kinds[c] = k
		kv, gt := hd, hd+k.coff*hd
		ikv, ig := gt+k.coff*hd, gt+k.coff*hd+2*ihd
		comp(&k.wK, func() (*ir.Kernel, error) { return kernels.PagedCopyRows(hd, R, P, k.row, 0, false) })
		keys := ds4Keys(d, c)
		comp(&k.gather, func() (*ir.Kernel, error) {
			re := k.ent
			if re == 0 {
				re = hd // bound, never read
			}
			return kernels.DS4Gather(R, x.kmax, hd, P, k.row, re, keys)
		})
		comp(&k.desc, func() (*ir.Kernel, error) { return kernels.DS4Desc(R, x.kmax, P, keys) })
		if c == nn.DS4CompNone {
			continue
		}
		cw := k.coff * hd
		noOv := d.Fault == nn.DS4FaultNoOverlap
		comp(&k.wKV, func() (*ir.Kernel, error) { return kernels.PagedCopyRows(cw, R, P, k.row, kv, false) })
		comp(&k.wG, func() (*ir.Kernel, error) { return kernels.PagedCopyRows(cw, R, P, k.row, gt, false) })
		comp(&k.ape, func() (*ir.Kernel, error) { return kernels.DS4ApeAdd(cw, rate, R) })
		comp(&k.pool, func() (*ir.Kernel, error) {
			return kernels.DS4Pool(R, P, k.row, kv, gt, hd, rate, c == nn.DS4CompCSA, noOv)
		})
		comp(&k.entW, func() (*ir.Kernel, error) { return kernels.DS4EntWrite(hd, R, P, k.ent, 0, rate) })
		al(&k.csE, R*nr)
		al(&k.erpos, R+1)
		k.posStage = make([]uint32, R+1)
		if c != nn.DS4CompCSA {
			continue
		}
		comp(&k.wIKV, func() (*ir.Kernel, error) { return kernels.PagedCopyRows(2*ihd, R, P, k.row, ikv, false) })
		comp(&k.wIG, func() (*ir.Kernel, error) { return kernels.PagedCopyRows(2*ihd, R, P, k.row, ig, false) })
		comp(&k.apeI, func() (*ir.Kernel, error) { return kernels.DS4ApeAdd(2*ihd, rate, R) })
		comp(&k.poolI, func() (*ir.Kernel, error) {
			return kernels.DS4Pool(R, P, k.row, ikv, ig, ihd, rate, true, noOv)
		})
		comp(&k.entWI, func() (*ir.Kernel, error) { return kernels.DS4EntWrite(ihd, R, P, k.ent, hd, rate) })
		tcap := p.MaxSeq/rate + 1
		comp(&x.entDesc, func() (*ir.Kernel, error) { return kernels.DS4EntDesc(R, rate) })
		comp(&x.idxSc, func() (*ir.Kernel, error) {
			return kernels.IdxScoresPaged(R, ih, ihd, tcap, P, k.ent, hd, idxScale)
		})
		comp(&x.idxSl, func() (*ir.Kernel, error) { return kernels.IdxSelect(R, tcap, d.IdxTopK) })
		al(&x.edesc, R*kernels.PRowWords)
		al(&x.score, R*tcap)
		al(&x.sel, R*(d.IdxTopK+1))
	}
	if x.sel == nil {
		al(&x.sel, R) // bound by the gathers of a model with no CSA block, never read
	}
	comp(&x.routeCopy, func() (*ir.Kernel, error) { return kernels.DS4RouteBias(p.NExpert, 0, 0, R, false) })
	if p.Vocab > 0 {
		comp(&x.routeHash, func() (*ir.Kernel, error) {
			return kernels.DS4RouteBias(p.NExpert, p.NExpertUsed, p.Vocab, R, true)
		})
	}
	al(&x.flat, R*s*n)
	al(&x.flatn, R*s*n)
	al(&x.mix, R*mix)
	al(&x.hc, R*mix)
	al(&x.coll, R*n)
	al(&x.onesFlat, s*n)
	al(&x.onesHD, max(hd, ihd))
	al(&x.qa, R*p.QLoraRank)
	al(&x.qan, R*p.QLoraRank)
	al(&x.q, R*H*hd)
	al(&x.qn, R*H*hd)
	al(&x.qr, R*H*hd)
	al(&x.kraw, R*hd)
	al(&x.kn, R*hd)
	al(&x.kr, R*hd)
	al(&x.ckv, R*2*hd)
	al(&x.cg, R*2*hd)
	al(&x.cgA, R*2*hd)
	al(&x.pe, R*hd)
	al(&x.pen, R*hd)
	al(&x.per, R*hd)
	if ih > 0 {
		al(&x.ikv, R*2*ihd)
		al(&x.ig, R*2*ihd)
		al(&x.igA, R*2*ihd)
		al(&x.iw, R*ih)
		al(&x.iq, R*ih*ihd)
		al(&x.iqr, R*ih*ihd)
		al(&x.ipe, R*ihd)
		al(&x.ipen, R*ihd)
		al(&x.iper, R*ihd)
	}
	al(&x.gk, R*x.ppr*P*hd)
	al(&x.gtab, R*x.ppr)
	al(&x.gdesc, R*kernels.PRowWords)
	al(&x.ao, R*H*hd)
	al(&x.aor, R*H*hd)
	al(&x.oa, R*d.OGroups*d.OLoraRank)
	al(&x.oSel, R*d.OGroups)
	al(&x.rbias, R*p.NExpert)
	al(&x.zeroB, p.NExpert)
	al(&x.ids, R)
	al(&x.noSink, H)
	if err != nil {
		return err
	}
	x.idsStage = make([]uint32, R)
	ones := make([]float32, max(s*n, hd, ihd))
	for i := range ones {
		ones[i] = 1
	}
	tab := make([]uint32, R*x.ppr)
	for i := range tab {
		tab[i] = uint32(i)
	}
	sel := make([]uint32, R*d.OGroups)
	for i := range sel {
		sel[i] = uint32(i % d.OGroups)
	}
	ninf := make([]float32, H)
	for i := range ninf {
		ninf[i] = float32(math.Inf(-1))
	}
	for _, w := range []struct {
		b backend.Buf
		v []byte
	}{{x.onesFlat, f32b(ones[:s*n])}, {x.onesHD, f32b(ones[:max(hd, ihd)])}, {x.gtab, u32view(tab)},
		{x.oSel, u32view(sel)}, {x.zeroB, make([]byte, 4*p.NExpert)}, {x.noSink, f32b(ninf)}} {
		if err := w.b.Write(w.v); err != nil {
			return err
		}
	}
	return nil
}

// free releases the set's kernels and buffers.
func (x *ds4Scratch) free() {
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
			closeKern(k)
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

// ds4Rows launches k over rows groups of RMSNormGroup.
func ds4Rows(lc *launcher, k backend.Kernel, rows int, bufs ...backend.Buf) {
	*lc.memoK, *lc.memoSrc = nil, nil
	if *lc.err == nil {
		*lc.err = lc.launch(k, rows, kernels.RMSNormGroup, bufs...)
	}
}

// ds4Pre is a hyper-connection's mix for R rows of src (the streams): the
// mixer's outputs into x.hc and the collapsed row normed by nw into bs.h,
// quantized into bs.a/bs.ax. attn picks the attention's hyper-connection.
func (g *devTier) ds4Pre(lc *launcher, bs *blockScratch, l *layer, R int, src backend.Buf, attn bool,
	norm func(src, w, wb backend.Buf)) {
	x, b, p := bs.ds4, l.ds4, &bs.p
	fn, sc, bb, nw := b.fnF, b.scF, b.bF, l.nFFN
	if attn {
		fn, sc, bb, nw = b.fnA, b.scA, b.bA, l.nAttn
	}
	n, s := p.NEmbd, p.DS4.HCMult
	mix := (2 + s) * s
	lc.la(x.hcFlat, R*s*n, src, x.flat)
	ds4Rows(lc, x.hcRms, R, x.flat, x.onesFlat, x.flatn)
	if *lc.err == nil {
		*lc.err = lc.launch(x.hcFn, mix*R, kernels.RouterThreads(1), fn, x.flatn, x.mix)
	}
	lc.la(x.hcMix, R, x.mix, sc, bb, x.hc)
	lc.la(x.hcColl, R*n, src, x.hc, x.coll)
	norm(x.coll, nw, nil)
	lc.la(bs.quantE, kernels.QuantizeThreads(R*n/32), bs.h, bs.a, bs.ax)
	g.DS4HC++
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

// ds4Attn is block li's attention for R rows: their normed input in bs.h
// (quantized in bs.a/bs.ax), its output in bs.mvOut. pc is the block's rows
// in the paged pool and ent its entries'.
func (g *devTier) ds4Attn(s backend.Session, lc *launcher, bs *blockScratch, l *layer, R int,
	pc *pagedCall, ent *kvLayerPool, mvrun func(mv, *resident, backend.Buf),
	mvid func(mv, *resident, backend.Buf, backend.Buf)) {
	x, b, p := bs.ds4, l.ds4, &bs.p
	d := p.DS4
	k := x.kinds[b.comp]
	H, hd := p.NHead, p.HeadDim
	cs := bs.csSWA
	if b.comp != nn.DS4CompNone && d.Fault != nn.DS4FaultMainRope {
		cs = bs.cs
	}
	pRow := pc.pk.pRow
	// The projections of the normed input: the query latent, the key, and
	// on a compressed block the compressor's (and on CSA the indexer's).
	mvrun(l.mvQA, l.wqa, x.qa)
	mvrun(l.mvk, l.wk, x.kraw)
	if b.comp != nn.DS4CompNone {
		mvrun(l.mvCKV, l.compKV, x.ckv)
		mvrun(l.mvCG, l.compGate, x.cg)
	}
	csa := b.comp == nn.DS4CompCSA
	if csa {
		mvrun(l.mvICKV, l.idxCompKV, x.ikv)
		mvrun(l.mvICG, l.idxCompGate, x.ig)
		mvrun(l.mvIW, l.idxProj, x.iw)
	}
	// The query: the latent normed, its quantization through q_b (and the
	// indexer's q), each head normed with no weight and rotated.
	ds4Rows(lc, x.rmsQ, R, x.qa, l.nQA, x.qan)
	lc.la(x.quantQA, kernels.QuantizeThreads(R*p.QLoraRank/32), x.qan, bs.a, bs.ax)
	mvrun(l.mvQB, l.wqb, x.q)
	if csa {
		mvrun(l.mvIQ, l.idxQB, x.iq)
		lc.la(x.ropeIQ, R*d.IdxHeads*d.IdxHeadDim, x.iq, cs, x.iqr)
	}
	ds4Rows(lc, x.rmsHeads, R*H, x.q, x.onesHD, x.qn)
	lc.la(x.ropeQ, R*H*hd, x.qn, cs, x.qr)
	// The key: normed and rotated, cached at the row's position with the
	// compressor's inputs beside it.
	ds4Rows(lc, x.rmsHD, R, x.kraw, l.nKVA, x.kn)
	lc.la(x.ropeK, R*hd, x.kn, cs, x.kr)
	lc.la(k.wK, R*hd, x.kr, pc.k, bs.zero, pc.tab, pRow)
	if b.comp != nn.DS4CompNone {
		cw := k.coff * hd
		lc.la(k.ape, R*cw, x.cg, b.ape, pRow, x.cgA)
		lc.la(k.wKV, R*cw, x.ckv, pc.k, bs.zero, pc.tab, pRow)
		lc.la(k.wG, R*cw, x.cgA, pc.k, bs.zero, pc.tab, pRow)
		if csa {
			iw := 2 * d.IdxHeadDim
			lc.la(k.apeI, R*iw, x.ig, b.iape, pRow, x.igA)
			lc.la(k.wIKV, R*iw, x.ikv, pc.k, bs.zero, pc.tab, pRow)
			lc.la(k.wIG, R*iw, x.igA, pc.k, bs.zero, pc.tab, pRow)
		}
		// Every row's entry, folded where the row closes one: pooled,
		// normed, rotated at the entry's position and cached (a row that
		// closes nothing writes the dummy row).
		lc.la(k.pool, R*hd, pc.k, pc.tab, pRow, x.pe)
		ds4Rows(lc, x.rmsHD, R, x.pe, b.norm, x.pen)
		lc.la(x.ropeK, R*hd, x.pen, k.csE, x.per)
		lc.la(k.entW, R*hd, x.per, ent.k, ent.tab, pRow)
		if csa {
			ihd := d.IdxHeadDim
			lc.la(k.poolI, R*ihd, pc.k, pc.tab, pRow, x.ipe)
			ds4Rows(lc, x.rmsIHD, R, x.ipe, b.inorm, x.ipen)
			lc.la(x.ropeIE, R*ihd, x.ipen, k.csE, x.iper)
			lc.la(k.entWI, R*ihd, x.iper, ent.k, ent.tab, pRow)
		}
	}
	// The keys each row reads: its window, then its entries -- every
	// visible one on HCA, the indexer's top k on CSA.
	ek, et := pc.k, pc.tab
	if ent != nil {
		ek, et = ent.k, ent.tab
	}
	if csa && !ds4Keys(d, b.comp).All && !ds4Keys(d, b.comp).None {
		tcap := p.MaxSeq/k.rate + 1
		lc.la(x.entDesc, R*kernels.PRowWords, pRow, x.edesc)
		lc.la(x.idxSc, R*tcap, x.iqr, x.iw, ent.k, ent.tab, x.edesc, x.score)
		lc.la(x.idxSl, R*tcap, x.score, x.edesc, x.sel)
		g.IdxSelects++
	}
	lc.la(k.gather, R*x.kmax*hd, pc.k, pc.tab, ek, et, pRow, x.sel, x.gk)
	lc.la(k.desc, R*kernels.PRowWords, pRow, x.gdesc)
	sink := l.sinks
	if d.Fault == nn.DS4FaultNoSinks {
		sink = x.noSink
	}
	if *lc.err == nil {
		*lc.err = pc.pk.pagedDecode(s, lc, x.qr, x.gk, x.gk, bs.n, x.ao, x.gtab, x.gdesc, sink)
	}
	// The output rotated back at the query's position, through its groups'
	// sheets (one indexed matvec, a slot a group per row) and o_b.
	out := x.ao
	if d.Fault != nn.DS4FaultNoDerope {
		lc.la(x.ropeI, R*H*hd, x.ao, cs, x.aor)
		out = x.aor
	}
	lc.la(bs.quantQ, kernels.QuantizeThreads(R*H*hd/32), out, bs.a, bs.ax)
	wa, ok := l.ds4WoA(R)
	if !ok {
		if *lc.err == nil {
			*lc.err = errNoBatch
		}
		return
	}
	mvid(wa, l.woA, x.oa, x.oSel)
	lc.la(x.quantOA, kernels.QuantizeThreads(R*d.OGroups*d.OLoraRank/32), x.oa, bs.a, bs.ax)
	mvrun(l.mvo, l.wo, bs.mvOut)
	g.DS4Blocks++
}

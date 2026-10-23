package tier

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// DeepSeek V3.2's lightning indexer on the device (nn.LayerPlan.IdxHeads;
// engine/model/indexer.go has the graph).
//
// The indexer's key is cached in the MLA row of the paged pool, after the
// latent and the rotary key (nn.LayerPlan.KVRow), so it pages, migrates and
// is stored with them. Each call:
//
//	the key: idx_k on the block input, its LayerNorm, the NEOX rotary on its
//	  first NRot dimensions, into the row's tail at the row's position
//	the query: idx_q_b on the normed query latent, rotated the same way per head
//	the weights: idx_proj on the block input
//	the scores, the selection and the gather (kernels/indexer.go): each row's
//	  IdxTopK best positions' latent rows copied into scratch pages
//
// and the MLA attention then runs unchanged over the gathered pages, whose
// descriptors give each row exactly its kept positions. Gathering every call,
// including below IdxTopK positions where every position is kept, keeps one
// submission shape for every token. A history with evicted pages (streamed)
// is refused by the call: the scores read every position through the table.

// idxScratch is one batch width's indexer buffers and kernels, on the paged
// scratch (pagedScratch.idx).
type idxScratch struct {
	h, d, k, tcap, w, ppr  int
	lnPart, lnVar, lnApply backend.Kernel
	copyK, ropeK, writeK   backend.Kernel
	copyQ, ropeQ           backend.Kernel
	scores, sel, gather    backend.Kernel
	desc                   backend.Kernel
	// kRaw/kn/kRot the key through its norm and rotary, kPart/kVar the norm's
	// partials; qRaw/qRot the query; w the heads' weights.
	kRaw, kPart, kVar, kn, kRot backend.Buf
	qRaw, qRot, wts             backend.Buf
	kOff, qOff                  backend.Buf
	// score each row's per-position scores, selBuf its kept positions (k+1
	// slots, the last a spare), gk the gathered pages, gtab their table and
	// gdesc their descriptors.
	score, selBuf, gk, gtab, gdesc backend.Buf
	parts                          int
}

// initIdx builds bs's indexer: rows rows of P-position pages whose rows are
// row wide, the latent and rotary key the first w of them. Callers hold g.mu.
func (g *devTier) initIdx(bs *blockScratch, rows, P, row, w int,
	comp func(*backend.Kernel, *ir.Kernel, error) error) error {
	p := &bs.p
	// A top-k past the context keeps every position; the gather is sized
	// for what a row can hold.
	tcap := max(p.MaxSeq, 1)
	kk := min(p.IdxTopK, tcap)
	ix := &idxScratch{h: p.IdxHeads, d: p.IdxHeadDim, k: kk, tcap: tcap, w: w,
		ppr: kernels.IdxGatherPages(kk, P), parts: partsFor(p.IdxHeadDim)}
	bs.pkv.idx = ix
	h, d, k, rot := ix.h, ix.d, ix.k, p.NRot
	scale := float32(1 / math.Sqrt(float64(d*h)))
	for _, b := range []struct {
		dst *backend.Kernel
		mk  func() (*ir.Kernel, error)
	}{
		{&ix.lnPart, func() (*ir.Kernel, error) { return kernels.LayerNormPartRows(d, ix.parts, rows) }},
		{&ix.lnVar, func() (*ir.Kernel, error) { return kernels.LayerNormVarRows(d, ix.parts, rows) }},
		{&ix.lnApply, func() (*ir.Kernel, error) { return kernels.LayerNormApplyRows(d, ix.parts, 1e-6, true, rows) }},
		{&ix.copyK, func() (*ir.Kernel, error) { return kernels.CopyAtRows(d, rows) }},
		{&ix.ropeK, func() (*ir.Kernel, error) { return kernels.RoPERows(1, d, rot, true, rows) }},
		{&ix.writeK, func() (*ir.Kernel, error) { return kernels.PagedCopyRows(d, rows, P, row, w, false) }},
		{&ix.copyQ, func() (*ir.Kernel, error) { return kernels.CopyAtRows(h*d, rows) }},
		{&ix.ropeQ, func() (*ir.Kernel, error) { return kernels.RoPERows(h, d, rot, true, rows) }},
		{&ix.scores, func() (*ir.Kernel, error) {
			return kernels.IdxScoresPaged(rows, h, d, ix.tcap, P, row, w, scale)
		}},
		{&ix.sel, func() (*ir.Kernel, error) { return kernels.IdxSelect(rows, ix.tcap, k) }},
		{&ix.gather, func() (*ir.Kernel, error) { return kernels.IdxGather(rows, k, w, P, row) }},
		{&ix.desc, func() (*ir.Kernel, error) { return kernels.IdxDesc(rows, k, P) }},
	} {
		kk, err := b.mk()
		if err := comp(b.dst, kk, err); err != nil {
			return fmt.Errorf("the indexer: %w", err)
		}
	}
	var err error
	al := func(dst *backend.Buf, n int) {
		if err == nil {
			*dst, err = g.dev.Alloc(n * 4)
		}
	}
	al(&ix.kRaw, rows*d)
	al(&ix.kPart, rows*ix.parts)
	al(&ix.kVar, rows*ix.parts)
	al(&ix.kn, rows*d)
	al(&ix.kRot, rows*d)
	al(&ix.qRaw, rows*h*d)
	al(&ix.qRot, rows*h*d)
	al(&ix.wts, rows*h)
	al(&ix.kOff, rows)
	al(&ix.qOff, rows)
	al(&ix.score, rows*ix.tcap)
	al(&ix.selBuf, rows*(k+1))
	al(&ix.gk, rows*ix.ppr*P*w)
	al(&ix.gtab, rows*ix.ppr)
	al(&ix.gdesc, rows*kernels.PRowWords)
	if err != nil {
		return err
	}
	ko, qo, tab := make([]uint32, rows), make([]uint32, rows), make([]uint32, rows*ix.ppr)
	for r := range ko {
		ko[r], qo[r] = uint32(r*d), uint32(r*h*d)
	}
	for i := range tab {
		tab[i] = uint32(i)
	}
	for _, x := range []struct {
		b backend.Buf
		v []uint32
	}{{ix.kOff, ko}, {ix.qOff, qo}, {ix.gtab, tab}} {
		if err := x.b.Write(u32view(x.v)); err != nil {
			return err
		}
	}
	return nil
}

// free releases the indexer's kernels and buffers.
func (ix *idxScratch) free() {
	if ix == nil {
		return
	}
	for _, k := range []backend.Kernel{ix.lnPart, ix.lnVar, ix.lnApply, ix.copyK, ix.ropeK, ix.writeK,
		ix.copyQ, ix.ropeQ, ix.scores, ix.sel, ix.gather, ix.desc} {
		if k != nil {
			closeKern(k)
		}
	}
	for _, b := range []backend.Buf{ix.kRaw, ix.kPart, ix.kVar, ix.kn, ix.kRot, ix.qRaw, ix.qRot, ix.wts,
		ix.kOff, ix.qOff, ix.score, ix.selBuf, ix.gk, ix.gtab, ix.gdesc} {
		if b != nil {
			b.Free()
		}
	}
}

// idxKeyAndWeights runs the indexer's projections of the block input -- the
// key and the heads' weights -- while bs.a/bs.ax hold its quantization, then
// the key's norm and rotary, and writes the key into the pool row's tail.
func (g *devTier) idxKeyAndWeights(lc *launcher, bs *blockScratch, l *layer, R int, pc *pagedCall,
	mvrun func(mv, *resident, backend.Buf)) {
	ix := pc.pk.idx
	mvrun(l.mvIK, l.idxK, ix.kRaw)
	mvrun(l.mvIW, l.idxProj, ix.wts)
	lc.la(ix.lnPart, R*ix.parts, ix.kRaw, ix.kPart)
	lc.la(ix.lnVar, R*ix.parts, ix.kRaw, ix.kPart, ix.kVar)
	lc.la(ix.lnApply, R*ix.d, ix.kRaw, l.nIdxK, ix.kPart, ix.kVar, l.nIdxKB, ix.kn)
	lc.la(ix.copyK, R*ix.d, ix.kn, ix.kRot, ix.kOff)
	lc.la(ix.ropeK, R*bs.p.NRot/2, ix.kn, bs.cs, ix.kOff, ix.kRot)
	lc.la(ix.writeK, R*ix.d, ix.kRot, pc.k, bs.zero, pc.tab, pc.pk.pRow)
}

// idxQuery runs the indexer's query projection while bs.a/bs.ax hold the
// normed query latent's quantization, and its rotary.
func (g *devTier) idxQuery(lc *launcher, bs *blockScratch, l *layer, R int, pc *pagedCall,
	mvrun func(mv, *resident, backend.Buf)) {
	ix := pc.pk.idx
	mvrun(l.mvIQ, l.idxQB, ix.qRaw)
	lc.la(ix.copyQ, R*ix.h*ix.d, ix.qRaw, ix.qRot, ix.qOff)
	lc.la(ix.ropeQ, R*ix.h*bs.p.NRot/2, ix.qRaw, bs.cs, ix.qOff, ix.qRot)
}

// idxAttend selects each row's positions, gathers them and runs the MLA
// attention over the gathered pages: q the row-wide absorbed queries, out
// the latent-wide results. A call over evicted history was refused at its
// preparation (the scores read every position through the table).
func (g *devTier) idxAttend(s backend.Session, lc *launcher, R int, pc *pagedCall, q, n, out backend.Buf) error {
	pk := pc.pk
	ix := pk.idx
	lc.la(ix.scores, R*ix.tcap, ix.qRot, ix.wts, pc.k, pc.tab, pk.pRow, ix.score)
	lc.la(ix.sel, R*ix.tcap, ix.score, pk.pRow, ix.selBuf)
	lc.la(ix.gather, R*ix.k*ix.w, pc.k, pc.tab, pk.pRow, ix.selBuf, ix.gk)
	lc.la(ix.desc, R*kernels.PRowWords, pk.pRow, ix.gdesc)
	g.IdxSelects++
	return pk.pagedDecode(s, lc, q, ix.gk, ix.gk, n, out, ix.gtab, ix.gdesc, nil)
}

// idxKeys is the most keys a gathered row attends: the call's, at most the
// top-k.
func idxKeys(p *nn.LayerPlan, keys int) int {
	if p.IdxHeads == 0 {
		return keys
	}
	return min(keys, (min(p.IdxTopK, max(p.MaxSeq, 1))+scoreGrain-1)/scoreGrain*scoreGrain)
}

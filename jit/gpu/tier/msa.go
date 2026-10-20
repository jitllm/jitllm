package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// MiniMax Sparse Attention on the device (nn.LayerPlan.MSA;
// engine/model/msa.go has the graph).
//
// A selecting block's cache row carries the indexer's key as one more kv
// head (nn.LayerPlan.KVRow), so it pages, migrates and is stored with K and V.
// The block runs in a scratch set of its own (geoMSA, gemma4.go): every K/V
// writer bakes the row. Each call:
//
//	the key: idx_k on the block's normed input, its per-head RMSNorm, then
//	  assembled after k's heads into one (NKVHead+1)-head row, which the
//	  pool's own writers rotate and store (the indexer's key turns with the
//	  attention's: same table, same width)
//	the query: idx_q on the same input, each head's RMSNorm, the rotary
//	the scores, the block ranking, the selection, the compaction and the
//	  gather (kernels/msa.go): each row's kept positions, per kv group, copied
//	  into scratch pages
//
// and the attention then runs unchanged over the gathered pages, whose
// descriptors give each row exactly its kept count. Gathering every call,
// including where every block is kept, keeps one submission shape for every
// token. A history with evicted pages (streamed) is refused by the call: the
// scores read every position through the table.

// msaScratch is one batch width's selection buffers and kernels, on the paged
// scratch (pagedScratch.msa).
type msaScratch struct {
	h, d, k, blk, tcap, nb, ppr, kvRow, attnW int
	qNorm, kNorm, qLanes, kLanes              int
	normQ, normK                              backend.Kernel
	copyQ, ropeQ, asmK, asmI                  backend.Kernel
	scores, bmax, sel, compact                backend.Kernel
	gatherK, gatherV, desc                    backend.Kernel
	// qRaw/qn/qRot the query through its norm and rotary; kRaw/kn the key
	// through its norm; kAll the assembled row, k's heads then the key.
	qRaw, qn, qRot, kRaw, kn, kAll backend.Buf
	qOff, allOff, idxOff           backend.Buf
	// score each (row, group)'s per-position scores, bmx its blocks' ranks,
	// flag the kept blocks, list their ascending order (k+1 slots, the last a
	// spare), gk/gv the gathered pages, gtab their table and gdesc their
	// descriptors.
	score, bmx, flag, list, gk, gv, gtab, gdesc backend.Buf
}

// msaWhyNot names a block-selection shape this tier declines, or "".
func msaWhyNot(p *nn.LayerPlan) string {
	switch {
	case !p.MSA():
		return ""
	case p.MLA() || p.Recurrent.Conv > 0 || p.GeomSplit || p.KVShared || p.AttnOutGate:
		return "MiniMax Sparse Attention with latent attention, a recurrence, a split geometry, " +
			"KV sharing or an out-gate"
	case p.IdxHeads != p.NKVHead || p.IdxHeadDim != p.HeadDim:
		return fmt.Sprintf("MiniMax Sparse Attention with %d indexer heads of %d on %d kv heads of %d",
			p.IdxHeads, p.IdxHeadDim, p.NKVHead, p.HeadDim)
	case p.IdxLocal < 1 || p.IdxLocal > p.IdxTopK:
		return fmt.Sprintf("MiniMax Sparse Attention forcing %d of %d blocks", p.IdxLocal, p.IdxTopK)
	case p.NoPosEnc || p.NRot == 0 || p.NoPEGlobal:
		return "MiniMax Sparse Attention with no rotary"
	}
	return ""
}

// initMSA builds bs's selection: rows rows of P-position pages whose K row is
// kvRow wide, the indexer's key its last head. Callers hold g.mu.
func (g *devTier) initMSA(bs *blockScratch, rows, P int,
	comp func(*backend.Kernel, *ir.Kernel, error) error) error {
	p := &bs.p
	tcap := max(p.MaxSeq, 1)
	kk := min(p.IdxTopK, oldkernels.MSABlocks(tcap, p.IdxBlock))
	m := &msaScratch{h: p.IdxHeads, d: p.IdxHeadDim, k: kk, blk: p.IdxBlock, tcap: tcap,
		nb: oldkernels.MSABlocks(tcap, p.IdxBlock), ppr: oldkernels.MSAGatherPages(kk, p.IdxBlock, P),
		kvRow: p.KVRow(), attnW: p.NKVHead * p.HeadDim}
	bs.pkv.msa = m
	m.qLanes, m.kLanes = bs.kNormL, bs.kNormL
	h, d, rot, kv := m.h, m.d, p.NRot, p.NKVHead*p.HeadDim
	for _, b := range []struct {
		dst *backend.Kernel
		mk  func() (*ir.Kernel, error)
	}{
		{&m.normQ, func() (*ir.Kernel, error) { return oldkernels.HeadNormRows(h, d, p.RMSEps, rows, m.qLanes) }},
		{&m.normK, func() (*ir.Kernel, error) { return oldkernels.HeadNormRows(1, d, p.RMSEps, rows, m.kLanes) }},
		{&m.copyQ, func() (*ir.Kernel, error) { return oldkernels.CopyAtRows(h*d, rows) }},
		{&m.ropeQ, func() (*ir.Kernel, error) { return oldkernels.RoPERows(h, d, rot, p.RopeNeox, rows) }},
		{&m.asmK, func() (*ir.Kernel, error) { return oldkernels.CopyAtRows(kv, rows) }},
		{&m.asmI, func() (*ir.Kernel, error) { return oldkernels.CopyAtRows(d, rows) }},
		{&m.scores, func() (*ir.Kernel, error) {
			return oldkernels.MSAScoresPaged(rows, h, d, tcap, P, m.kvRow, kv)
		}},
		{&m.bmax, func() (*ir.Kernel, error) { return oldkernels.MSABlockMax(rows, h, tcap, m.blk, p.IdxLocal) }},
		{&m.sel, func() (*ir.Kernel, error) { return oldkernels.MSASelect(rows, h, tcap, m.blk, kk) }},
		{&m.compact, func() (*ir.Kernel, error) { return oldkernels.MSACompact(rows, h, tcap, m.blk, kk) }},
		{&m.gatherK, func() (*ir.Kernel, error) {
			return oldkernels.MSAGatherK(rows, p.NKVHead, p.HeadDim, m.blk, kk, P, m.kvRow)
		}},
		{&m.gatherV, func() (*ir.Kernel, error) {
			return oldkernels.MSAGatherV(rows, p.NKVHead, p.HeadDim, m.blk, kk, P, m.kvRow, g.KVF16)
		}},
		{&m.desc, func() (*ir.Kernel, error) { return oldkernels.MSADesc(rows, m.blk, kk, P) }},
	} {
		kn, err := b.mk()
		if err := comp(b.dst, kn, err); err != nil {
			return fmt.Errorf("the block selection: %w", err)
		}
	}
	vw := m.ppr * P * kv
	if g.KVF16 {
		vw /= 2
	}
	var err error
	al := func(dst *backend.Buf, n int) {
		if err == nil {
			*dst, err = g.dev.Alloc(n * 4)
		}
	}
	al(&m.qRaw, rows*h*d)
	al(&m.qn, rows*h*d)
	al(&m.qRot, rows*h*d)
	al(&m.kRaw, rows*d)
	al(&m.kn, rows*d)
	al(&m.kAll, rows*m.kvRow)
	al(&m.qOff, rows)
	al(&m.allOff, rows)
	al(&m.idxOff, rows)
	al(&m.score, rows*h*tcap)
	al(&m.bmx, rows*h*m.nb)
	al(&m.flag, rows*h*m.nb)
	al(&m.list, rows*h*(kk+1))
	al(&m.gk, rows*m.ppr*P*kv)
	al(&m.gv, rows*vw)
	al(&m.gtab, rows*m.ppr)
	al(&m.gdesc, rows*oldkernels.PRowWords)
	if err != nil {
		return err
	}
	qo, ao, io, tab := make([]uint32, rows), make([]uint32, rows), make([]uint32, rows),
		make([]uint32, rows*m.ppr)
	for r := range qo {
		qo[r], ao[r], io[r] = uint32(r*h*d), uint32(r*m.kvRow), uint32(r*m.kvRow+kv)
	}
	for i := range tab {
		tab[i] = uint32(i)
	}
	for _, x := range []struct {
		b backend.Buf
		v []uint32
	}{{m.qOff, qo}, {m.allOff, ao}, {m.idxOff, io}, {m.gtab, tab}} {
		if err := x.b.Write(u32view(x.v)); err != nil {
			return err
		}
	}
	return nil
}

// free releases the selection's kernels and buffers.
func (m *msaScratch) free(g *devShared) {
	if m == nil {
		return
	}
	for _, k := range []backend.Kernel{m.normQ, m.normK, m.copyQ, m.ropeQ, m.asmK, m.asmI, m.scores,
		m.bmax, m.sel, m.compact, m.gatherK, m.gatherV, m.desc} {
		if k != nil {
			g.closeKern(k)
		}
	}
	for _, b := range []backend.Buf{m.qRaw, m.qn, m.qRot, m.kRaw, m.kn, m.kAll, m.qOff, m.allOff,
		m.idxOff, m.score, m.bmx, m.flag, m.list, m.gk, m.gv, m.gtab, m.gdesc} {
		if b != nil {
			b.Free()
		}
	}
}

// msaKeyQuery runs the indexer's projections while bs.a/bs.ax hold the
// block's normed input, their norms and the query's rotary, and assembles
// k's normed heads kn and the key into one row each (m.kAll), which the
// pool's writers then rotate and store.
func (g *devTier) msaKeyQuery(lc *launcher, bs *blockScratch, l *layer, R int, kn, cs backend.Buf,
	mvrun func(mv, *resident, backend.Buf)) error {
	m := bs.pkv.msa
	mvrun(l.mvIK, l.idxK, m.kRaw)
	mvrun(l.mvMQ, l.idxQ, m.qRaw)
	qg, qw := headNormGeom(m.qLanes, R*m.h)
	if err := lc.launch(m.normQ, qg, qw, m.qRaw, l.nIdxQ, m.qn); err != nil {
		return err
	}
	kg, kw := headNormGeom(m.kLanes, R)
	if err := lc.launch(m.normK, kg, kw, m.kRaw, l.nIdxK, m.kn); err != nil {
		return err
	}
	lc.la(m.copyQ, R*m.h*m.d, m.qn, m.qRot, m.qOff)
	lc.la(m.ropeQ, R*m.h*bs.p.NRot/2, m.qn, cs, m.qOff, m.qRot)
	lc.la(m.asmK, R*m.attnW, kn, m.kAll, m.allOff)
	lc.la(m.asmI, R*m.d, m.kn, m.kAll, m.idxOff)
	return nil
}

// msaAttend selects each row's blocks per kv group, gathers them and runs
// the attention over the gathered pages: q the rotated queries, out the
// heads' results. pl is the layer's pool, tab its table arena.
func (g *devTier) msaAttend(s backend.Session, lc *launcher, R int, pk *pagedScratch, pl *kvLayerPool,
	q, n, out backend.Buf) error {
	m := pk.msa
	lc.la(m.scores, R*m.h*m.tcap, m.qRot, pl.k, pl.tab, pk.pRow, m.score)
	lc.la(m.bmax, R*m.h*m.nb, m.score, pk.pRow, m.bmx)
	lc.la(m.sel, R*m.h*m.nb, m.bmx, pk.pRow, m.flag)
	lc.la(m.compact, R*m.h*m.nb, m.flag, m.list)
	lc.la(m.gatherK, R*m.attnW*m.k*m.blk, pl.k, pl.tab, pk.pRow, m.list, m.gk)
	vw := m.attnW
	if g.KVF16 {
		vw /= 2
	}
	lc.la(m.gatherV, R*m.k*m.blk*vw, pl.v, pl.tab, pk.pRow, m.list, m.gv)
	lc.la(m.desc, R*oldkernels.PRowWords, pk.pRow, m.gdesc)
	g.MSASelects++
	return pk.pagedDecode(s, lc, q, m.gk, m.gv, n, out, m.gtab, m.gdesc, nil)
}

// msaKeys is the most keys a gathered row attends: the call's, at most the
// kept blocks.
func msaKeys(p *nn.LayerPlan, keys int) int {
	if !p.MSA() {
		return keys
	}
	return min(keys, (p.IdxTopK*p.IdxBlock+scoreGrain-1)/scoreGrain*scoreGrain)
}

package tier

import (
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// mlaBatch is the scratch for multi-head latent attention over a batched
// chunk. emitMLA is decode's sequence; this is the same sequence over R rows,
// and the two differ in exactly three places.
//
//  1. Every elementwise launch is its Rows twin (SliceRows over R*NHead heads,
//     RoPERows over R positions, NormPartRows/NormApplyRows over R latents,
//     CopyAtRows with a per-row destination).
//  2. The two absorbs are GROUPED matvecs rather than indexed ones. Decode's
//     indexed matvec runs one sheet per head against one token; a chunk has R
//     tokens per head, which is exactly the grouped mixture's shape with the
//     head as the expert -- every head's run is R columns, so it needs no
//     padding once R is a multiple of groupTok, and one sheet read serves all R.
//     The activations are gathered HEAD-major for it and the results come back
//     token-major.
//  3. The accumulate is AttnAccTiled at the latent's width over the row-wide
//     cache, the batched twin of decode's AttnAcc with the same (headDim,
//     kvDim, gqa) substitution; the scores kernel initScratch builds for a
//     latent plan is already the Rows one.
//
// Nothing here is a new kernel. The decode path (emitMLA, rows == 1) is not
// touched: this scratch exists only on a batched blockScratch.
type mlaBatch struct {
	rows, nh, lat, rot, nope, hdv, row, qlr int
	parts, qaParts                          int

	sliceLat, normPart, normApply, copyLat, sliceKPE, ropeK, noPEK,
	qaNormPart, qaNormApply, qaQuant,
	sliceNope, slicePE, ropeQ, gatherNope, quantNope, copyAbs, copyPE,
	acc, gatherAcc, quantAcc, gatherOut, quantOut backend.Kernel
	// scores70 is the scores on sm_70's m8n8k4 over the row-major latent
	// cache (kernels.AttnScoresMMA70 at kStride 0), nil where the FMA tile
	// the block scratch built (bs.scores) serves; acc70 the accumulate
	// (kernels.AttnAccMMA70), nil where acc serves.
	scores70, acc70 backend.Kernel

	kv, kvLat, kvn, kvPart, kpe, pOff, qa, qan, qaPart,
	qNope, nopeHM, pe, pErot, absRaw, abs, accOut, accHM, outHM, out,
	permHM, permTM, tsel, absOff, peOff, qOff backend.Buf
}

// newMLABatch builds the batched latent scratch for a rows-wide blockScratch,
// or nil with g.LastErr naming why. bs must already carry its attention tiles.
func (g *devTier) newMLABatch(bs *blockScratch, p *nn.LayerPlan, rows int) *mlaBatch {
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	if rows%groupTok != 0 {
		g.LastErr = "tier: batched latent attention: the chunk width is not a whole number of matvec groups"
		return nil
	}
	nh, lat, rot := p.NHead, p.KVLoraRank, p.NRot
	m := &mlaBatch{rows: rows, nh: nh, lat: lat, rot: rot, nope: p.HeadDim - rot,
		hdv: p.HeadDimV, row: p.KVLoraRank + p.NRot, qlr: p.QLoraRank,
		parts: partsFor(lat), qaParts: partsFor(p.QLoraRank)}
	actWin := p.ActWin
	if actWin == 0 {
		actWin = 32
	}
	eps := latentEps(p) // the latent norms', which Kimi's code sets apart
	var err error
	kern := func(dst *backend.Kernel, mk func() (*ir.Kernel, error)) {
		if err != nil {
			return
		}
		var kk *ir.Kernel
		if kk, err = mk(); err == nil {
			*dst, err = g.dev.Compile(kk)
		}
	}
	buf := func(dst *backend.Buf, n int) {
		if err == nil {
			*dst, err = g.dev.Alloc(n)
		}
		if err == nil && g.kb.poisonScratch {
			(*dst).Write(nanFill(n))
		}
	}
	R, nope, row, hdv := rows, m.nope, m.row, m.hdv
	// The shared latent and its rotary key.
	kern(&m.sliceLat, func() (*ir.Kernel, error) { return kernels.SliceRows(lat, row, 0, R) })
	kern(&m.normPart, func() (*ir.Kernel, error) { return kernels.NormPartRows(lat, m.parts, R) })
	kern(&m.normApply, func() (*ir.Kernel, error) { return kernels.NormApplyRows(lat, m.parts, eps, false, R) })
	kern(&m.copyLat, func() (*ir.Kernel, error) { return kernels.CopyAtRows(lat, R) })
	kern(&m.sliceKPE, func() (*ir.Kernel, error) { return kernels.SliceRows(rot, row, lat, R) })
	if p.NoPosEnc {
		kern(&m.noPEK, func() (*ir.Kernel, error) { return kernels.CopyAtRows(rot, R) })
	}
	// The fault arm's rotations on a NoPE model (MLAFaultRotateNoPE).
	rotate := !p.NoPosEnc || g.kb.mlaFault == MLAFaultRotateNoPE
	if rotate {
		kern(&m.ropeK, func() (*ir.Kernel, error) { return kernels.RoPERows(1, rot, rot, p.RopeNeox, R) })
	}
	if m.qlr != 0 {
		kern(&m.qaNormPart, func() (*ir.Kernel, error) { return kernels.NormPartRows(m.qlr, m.qaParts, R) })
		kern(&m.qaNormApply, func() (*ir.Kernel, error) {
			return kernels.NormApplyRows(m.qlr, m.qaParts, eps, false, R)
		})
		kern(&m.qaQuant, func() (*ir.Kernel, error) { return kernels.Quantize(R*m.qlr, actWin) })
	}
	// The query's two halves, the absorb and the assembled row-wide query.
	kern(&m.sliceNope, func() (*ir.Kernel, error) { return kernels.SliceRows(nope, p.HeadDim, 0, R*nh) })
	kern(&m.slicePE, func() (*ir.Kernel, error) { return kernels.SliceRows(rot, p.HeadDim, nope, R*nh) })
	if rotate {
		kern(&m.ropeQ, func() (*ir.Kernel, error) { return kernels.RoPERows(nh, rot, rot, p.RopeNeox, R) })
	}
	kern(&m.gatherNope, func() (*ir.Kernel, error) { return kernels.GatherRows(nope, nh*R, R*nh) })
	kern(&m.quantNope, func() (*ir.Kernel, error) { return kernels.Quantize(nh*R*nope, actWin) })
	kern(&m.copyAbs, func() (*ir.Kernel, error) { return kernels.CopyAtRows(lat, nh*R) })
	kern(&m.copyPE, func() (*ir.Kernel, error) { return kernels.CopyAtRows(rot, R*nh) })
	// The accumulate at the latent's width over the row-wide cache, and the
	// un-absorb's two reorderings.
	kern(&m.acc, func() (*ir.Kernel, error) {
		// gqa is NHead so that every head's value column is the row's own
		// prefix -- decode's substitution (see the attnAcc builder), and its
		// fault arm too.
		gqa := nh
		if g.kb.mlaFault == MLAFaultValueHeadMajor {
			gqa = 1
		}
		return kernels.AttnAccTiled(nh, lat, row, gqa, bs.sstride, R, bs.atile)
	})
	// Scores and accumulate on sm_70's tensor cores. The FMA tile walks the
	// row in 4-byte loads a row apart (the layout is the cache's and cannot
	// be transposed, since the same row is the value); m8n8k4 reads a key's
	// four dims as one 16-byte load. AttnAccMMA70 reads V row-major, so the
	// latent's first KVLoraRank floats are its V as they stand; its 32-row
	// query group is why initScratch fixes the softmax atile at 32
	// (bs.mla70). A refusal (another backend, an f16 cache) keeps the FMA
	// tile. See docs/engineering-history/gpu-kernels.md, "A mixture's prompt on
	// sm_70's tensor cores".
	if err == nil && bs.mla70 {
		comp := func(kk *ir.Kernel, e error) backend.Kernel {
			if e != nil {
				return nil
			}
			c, e := g.dev.Compile(kk)
			if e != nil {
				return nil
			}
			return c
		}
		m.scores70 = comp(kernels.AttnScoresMMA70(nh, row, row, nh, bs.sstride, g.scoreScale(p), R, 0,
			volta70AttnMT, volta70AttnNT, 0))
		gqa := nh
		if g.kb.mlaFault == MLAFaultValueHeadMajor {
			gqa = 1 // the tiled accumulate's fault arm, for the same gate
		}
		m.acc70 = comp(kernels.AttnAccMMA70(nh, lat, row, gqa, bs.sstride, R, volta70LatMT(lat), volta70AttnNT))
	}
	kern(&m.gatherAcc, func() (*ir.Kernel, error) { return kernels.GatherRows(lat, nh*R, R*nh) })
	kern(&m.quantAcc, func() (*ir.Kernel, error) { return kernels.Quantize(nh*R*lat, actWin) })
	kern(&m.gatherOut, func() (*ir.Kernel, error) { return kernels.GatherRows(hdv, R*nh, nh*R) })
	kern(&m.quantOut, func() (*ir.Kernel, error) { return kernels.Quantize(R*nh*hdv, actWin) })

	for _, b := range []struct {
		dst *backend.Buf
		n   int
	}{{&m.kv, R * row * 4}, {&m.kvLat, R * lat * 4}, {&m.kvn, R * lat * 4}, {&m.kvPart, R * m.parts * 4},
		{&m.kpe, R * rot * 4}, {&m.pOff, R * 4},
		{&m.qNope, R * nh * nope * 4}, {&m.nopeHM, R * nh * nope * 4},
		{&m.pe, R * nh * rot * 4}, {&m.pErot, R * nh * rot * 4},
		{&m.absRaw, R * nh * lat * 4}, {&m.abs, R * nh * row * 4},
		{&m.accOut, R * nh * lat * 4}, {&m.accHM, R * nh * lat * 4},
		{&m.outHM, R * nh * hdv * 4}, {&m.out, R * nh * hdv * 4},
		{&m.permHM, nh * R * 4}, {&m.permTM, nh * R * 4}, {&m.tsel, nh * R / groupTok * 4},
		{&m.absOff, nh * R * 4}, {&m.peOff, nh * R * 4}, {&m.qOff, R * 4}} {
		buf(b.dst, b.n)
	}
	if m.qlr != 0 {
		buf(&m.qa, R*m.qlr*4)
		buf(&m.qan, R*m.qlr*4)
		buf(&m.qaPart, R*m.qaParts*4)
	}
	if err == nil {
		// The geometry, written once. Head-major column j = h*R+r is token r's
		// head h; token-major j = r*NHead+h is the same pair the other way.
		permHM, permTM := make([]uint32, nh*R), make([]uint32, nh*R)
		absOff, peOff := make([]uint32, nh*R), make([]uint32, nh*R)
		tsel := make([]uint32, nh*R/groupTok)
		qOff := make([]uint32, R)
		for h := 0; h < nh; h++ {
			for r := 0; r < R; r++ {
				hm, tm := h*R+r, r*nh+h
				permHM[hm], permTM[tm] = uint32(tm), uint32(hm)
				// The absorbed latent of (h, r) lands at the head's slot of
				// token r's row-wide query; the rotated key follows it.
				absOff[hm] = uint32(tm * row)
				peOff[tm] = uint32(tm*row + lat)
			}
		}
		for gi := range tsel {
			h := gi * groupTok / R
			if g.kb.mlaFault == MLAFaultSheetOff {
				h = (h + 1) % nh // decode's fault arm, for the same gate
			}
			tsel[gi] = uint32(h)
		}
		for r := range qOff {
			qOff[r] = uint32(r * nh * rot)
		}
		for _, w := range []struct {
			b backend.Buf
			v []uint32
		}{{m.permHM, permHM}, {m.permTM, permTM}, {m.absOff, absOff}, {m.peOff, peOff},
			{m.tsel, tsel}, {m.qOff, qOff}} {
			if err == nil {
				err = w.b.Write(u32b(w.v))
			}
		}
	}
	if err != nil {
		g.LastErr = "tier: batched latent attention: " + err.Error()
		m.free(g.devShared)
		return nil
	}
	return m
}

func (m *mlaBatch) free(g *devShared) {
	if m == nil {
		return
	}
	for _, k := range []backend.Kernel{m.sliceLat, m.normPart, m.normApply, m.copyLat, m.sliceKPE,
		m.ropeK, m.noPEK, m.qaNormPart, m.qaNormApply, m.qaQuant, m.sliceNope, m.slicePE, m.ropeQ,
		m.gatherNope, m.quantNope, m.copyAbs, m.copyPE, m.acc, m.gatherAcc, m.quantAcc,
		m.gatherOut, m.quantOut, m.scores70, m.acc70} {
		if k != nil {
			g.closeKern(k)
		}
	}
	for _, b := range []backend.Buf{m.kv, m.kvLat, m.kvn, m.kvPart, m.kpe, m.pOff, m.qa, m.qan,
		m.qaPart, m.qNope, m.nopeHM, m.pe, m.pErot, m.absRaw, m.abs, m.accOut, m.accHM, m.outHM,
		m.out, m.permHM, m.permTM, m.tsel, m.absOff, m.peOff, m.qOff} {
		if b != nil {
			b.Free()
		}
	}
}

// mlaBanks is the (absorb, un-absorb) pair of matvecs and weights, with the
// swap fault applied exactly as emitMLA applies it.
func mlaBanks(l *layer, fault MLAFault) (absMV mv, absW *resident, unMV mv, unW *resident) {
	absMV, absW, unMV, unW = l.mvKB, l.wkb, l.mvVB, l.wvb
	if fault == MLAFaultSwapBanks {
		absMV, absW, unMV, unW = l.mvVB, l.wvb, l.mvKB, l.wkb
	}
	return
}

// emitMLARows is emitMLA over a batched chunk of bs.rows rows: from the
// quantized pre-norm (bs.a/bs.ax, float bs.h) to bs.mvOut. Its order is
// emitMLA's, including kv_a running before the query latent's quantize
// overwrites the block's activation.
func (g *devTier) emitMLARows(s backend.Session, bs *blockScratch, l *layer, kvp *kvPair,
	p *nn.LayerPlan, errp *error,
	lc *launcher,
	mvrun func(mv, *resident, backend.Buf),
	sm backend.Kernel, smG, smW, nCap int, pc *pagedCall) {
	m := bs.mlab
	R, nh, lat, rot, nope := m.rows, m.nh, m.lat, m.rot, m.nope
	// grouped runs a bank over src, quantized into (bs.a, bs.ax) just before;
	// a float bank reads src itself, in the scale plane's slot.
	grouped := func(x mv, r *resident, src, dst backend.Buf) {
		c, ok := g.groupedMV(x, nh, nh*R)
		if !ok {
			if *errp == nil {
				*errp = errNoBatch
			}
			return
		}
		d := r.d
		if kernels.IsFloat(r.t) {
			d = src
		}
		// Every group is in use: a head's run is R columns, a whole number of
		// groups, so nothing is padded and nothing is skipped.
		lc.la(c, x.rows*(nh*R/groupTok), r.qs, d, r.sc, bs.a, bs.ax, dst, m.tsel)
	}

	// ── the query, in one step or two, and the shared latent ─────────────
	if m.qlr != 0 {
		mvrun(l.mvQA, l.wqa, m.qa)
	} else {
		mvrun(l.mvq, l.wq, bs.q)
	}
	mvrun(l.mvKVA, l.wkva, m.kv)
	idx := pc != nil && pc.pk.idx != nil
	if idx {
		g.idxKeyAndWeights(lc, bs, l, R, pc, mvrun)
	}
	if m.qlr != 0 {
		lc.la(m.qaNormPart, R*m.qaParts, m.qa, m.qaPart)
		lc.la(m.qaNormApply, R*m.qlr, m.qa, l.nQA, m.qaPart, m.qan)
		lc.la(m.qaQuant, kernels.QuantizeThreads(R*m.qlr/32), m.qan, bs.a, bs.ax)
		mvrun(l.mvQB, l.wqb, bs.q)
		if idx {
			g.idxQuery(lc, bs, l, R, pc, mvrun)
		}
	}

	// ── the latent norm over each row's first KVLoraRank floats, then the
	// cache row: the normed latent at pos*row, the rotated key after it ────
	lc.la(m.sliceLat, R*lat, m.kv, m.kvLat)
	lc.la(m.normPart, R*m.parts, m.kvLat, m.kvPart)
	lc.la(m.normApply, R*lat, m.kvLat, l.nKVA, m.kvPart, m.kvn)
	lc.la(m.sliceKPE, R*rot, m.kv, m.kpe)
	// A NoPE model copies its rotary channels as they are, in a chunk or a
	// ragged step as in decode; the fault arm rotates them here and not in
	// decode (MLAFaultRotateNoPE).
	rotate := !p.NoPosEnc || g.kb.mlaFault == MLAFaultRotateNoPE
	if pc != nil {
		pc.writeLatent(lc, R, m.kvn, m.kpe, pc.pk.rotOff, bs, rotate)
	} else {
		lc.la(m.copyLat, R*lat, m.kvn, kvp.kc, bs.koff)
		if p.NoPosEnc {
			lc.la(m.noPEK, R*rot, m.kpe, kvp.kc, m.pOff)
		} else {
			lc.la(m.ropeK, R*rot/2, m.kpe, bs.cs, m.pOff, kvp.kc)
		}
	}

	// ── the query's halves: nope head-major for the absorb, pe rotated ───
	lc.la(m.sliceNope, R*nh*nope, bs.q, m.qNope)
	lc.la(m.slicePE, R*nh*rot, bs.q, m.pe)
	pe := m.pe
	if rotate {
		lc.la(m.ropeQ, R*nh*rot/2, m.pe, bs.cs, m.qOff, m.pErot)
		pe = m.pErot
	}
	lc.la(m.gatherNope, nh*R*nope, m.qNope, m.permHM, m.nopeHM)
	lc.la(m.quantNope, kernels.QuantizeThreads(nh*R*nope/32), m.nopeHM, bs.a, bs.ax)
	absMV, absW, unMV, unW := mlaBanks(l, g.kb.mlaFault)
	grouped(absMV, absW, m.nopeHM, m.absRaw)
	lc.la(m.copyAbs, nh*R*lat, m.absRaw, m.abs, m.absOff)
	lc.la(m.copyPE, R*nh*rot, pe, m.abs, m.peOff)

	// ── attention, at two widths over one buffer ─────────────────────────
	if pc != nil {
		// A prefill chunk through the paged prefill kernels, a batch's rows
		// through the decode plan, the value the row's prefix either way.
		if *errp == nil {
			if idx {
				*errp = g.idxAttend(s, lc, R, pc, m.abs, bs.n, m.accOut)
			} else if pc.pk.prefill {
				*errp = pc.pk.pf.prefillAttn(s, lc, m.abs, pc.k, pc.k, bs.n, m.accOut, pc.tab, false, nil)
				g.PagedPrefillLaunches++
				if pc.pk.pf.on70() {
					g.PagedPrefill70++
				}
			} else {
				*errp = pc.pk.attend(s, lc, pc.pl, m.abs, pc.k, pc.k, bs.n, m.accOut, pc.tab, pc.desc, nil)
			}
		}
	} else {
		g.emitMLARowsContiguous(s, bs, kvp, errp, lc, sm, smG, smW, nCap)
	}

	// ── un-absorb W_v, head-major, and project out token-major ──────────
	lc.la(m.gatherAcc, nh*R*lat, m.accOut, m.permHM, m.accHM)
	lc.la(m.quantAcc, kernels.QuantizeThreads(nh*R*lat/32), m.accHM, bs.a, bs.ax)
	grouped(unMV, unW, m.accHM, m.outHM)
	lc.la(m.gatherOut, R*nh*m.hdv, m.outHM, m.permTM, m.out)
	lc.la(m.quantOut, kernels.QuantizeThreads(R*nh*m.hdv/32), g.k3GatedMLA(lc, bs, l, R, m.out), bs.a, bs.ax)
	mvrun(l.mvo, l.wo, bs.mvOut)
	g.MLABatched++
}

// pagedCall is one layer's paged history as an MLA block's emitter needs it:
// the pool's buffer (the latent rows; the value is their prefix), the table
// arena and the call's descriptors.
type pagedCall struct {
	pk        *pagedScratch
	pl        *kvLayerPool
	k         backend.Buf
	tab, desc backend.Buf
}

// writeLatent puts R rows' latents at the head of their positions' rows in the
// pool and their rotary keys after them: the rotation (or, on a NoPE model,
// nothing: rotate false) into the scratch's staging at rotOff's per-row
// offsets, then one paged copy per part. off is the rotation's offset buffer:
// bs.zero for a single row, pk.rotOff for a batch.
func (pc *pagedCall) writeLatent(lc *launcher, R int,
	lat, kpe, off backend.Buf, bs *blockScratch, rotate bool) {
	pk, p := pc.pk, &bs.p
	rot := p.NRot
	lc.la(pk.copyLat, R*p.KVLoraRank, lat, pc.k, bs.zero, pc.tab, pk.pRow)
	src := kpe
	if rotate {
		rope := bs.mlaRopeK
		if R > 1 {
			rope = bs.mlab.ropeK
		}
		lc.la(rope, R*rot/2, kpe, bs.cs, off, pk.kpeRot)
		src = pk.kpeRot
	}
	lc.la(pk.copyPE, R*rot, src, pc.k, bs.zero, pc.tab, pk.pRow)
}

// emitMLARowsContiguous is emitMLARows' attention over the contiguous latent
// cache: the scores, the softmax and the accumulate at two widths over one
// buffer.
func (g *devTier) emitMLARowsContiguous(s backend.Session, bs *blockScratch, kvp *kvPair, errp *error,
	lc *launcher, sm backend.Kernel, smG, smW, nCap int) {
	m := bs.mlab
	R, nh, lat := m.rows, m.nh, m.lat
	if m.scores70 != nil {
		lc.la(m.scores70, kernels.AttnScoresMMA70Warps(nh, R, nCap, volta70AttnMT, volta70AttnNT)*32,
			m.abs, kvp.kc, bs.n, bs.att)
		g.MLAScores70++
	} else {
		kg := (nCap + bs.ktile - 1) / bs.ktile
		lc.la(bs.scores, (R/bs.qtile)*nh*kg, m.abs, kvp.kc, bs.n, bs.att)
	}
	if *errp == nil {
		// Not la(): the softmax group is not 128 wide. MLA carries no sinks.
		*errp = lc.launch(sm, smG, smW, bs.att, bs.n, bs.prob)
	}
	if m.acc70 != nil {
		lc.la(m.acc70, kernels.AttnAccMMA70Warps(nh, lat, R, volta70LatMT(lat), volta70AttnNT)*32,
			bs.prob, kvp.value(), bs.n, m.accOut)
		g.MLAAcc70++
	} else {
		lc.la(m.acc, R*nh*lat/bs.atile, bs.prob, kvp.value(), bs.n, m.accOut)
	}
}

package tier

import (
	"github.com/jitllm/jitllm/jit/gpu/backend"
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
	// cache (mainkernels.AttnScoresMMA70 at kStride 0), nil where the FMA tile
	// the block scratch built (bs.scores) serves; acc70 the accumulate
	// (mainkernels.AttnAccMMA70), nil where acc serves.
	scores70, acc70 backend.Kernel

	kv, kvLat, kvn, kvPart, kpe, pOff, qa, qan, qaPart,
	qNope, nopeHM, pe, pErot, absRaw, abs, accOut, accHM, outHM, out,
	permHM, permTM, tsel, absOff, peOff, qOff backend.Buf
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

// pagedCall is one layer's paged history as an MLA block's emitter needs it:
// the pool's buffer (the latent rows; the value is their prefix), the table
// arena and the call's descriptors.
type pagedCall struct {
	pk        *pagedScratch
	pl        *kvLayerPool
	k         backend.Buf
	tab, desc backend.Buf
}

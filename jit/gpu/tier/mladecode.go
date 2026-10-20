package tier

import (
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// emitMLA issues one multi-head latent attention block's launches, from the
// quantized pre-norm to bs.mvOut, the span emitLinear and the ordinary attention
// half occupy.
//
// The cache holds one row per position for the layer. engine/model/mla.go has the
// derivation; the two identities are
//
//	score_h = q_nope_h . (W_k_h . c)  =  (W_k_hᵀ . q_nope_h) . c
//	out_h   = Σ a_j (W_v_h . c_j)     =  W_v_h . (Σ a_j c_j)
//
// so W_kᵀ folds into the query before the scores and W_v into the output after,
// and no key or value is materialised: a row-wide absorbed query against a
// row-wide cache for the scores, a latent-wide result from the same rows for the
// accumulation. Every launch reuses an existing kernel (the absorb is the
// indexed matvec with the head as the index).
//
// The order is engine/model/mla.go's; in particular kv_a runs before the query's
// second step, since both read bs.a/bs.ax and quantizing the query latent
// overwrites it.
func (g *devTier) emitMLA(s backend.Session, bs *blockScratch, l *layer, kvp *kvPair,
	p *nn.LayerPlan, errp *error,
	lc *launcher,
	mvrun func(mv, *resident, backend.Buf),
	mvid func(mv, *resident, backend.Buf, backend.Buf),
	sm backend.Kernel, smG, smW, nCap int, skip map[string]bool, pc *pagedCall) {
	lat, rot, nope := bs.mlaLat, bs.mlaRot, bs.mlaNopeW
	nh := p.NHead

	// ── the query, in one step or two ────────────────────────────────────
	//
	// One step or two is a property of the model, asked of the plan: V3
	// compresses the query through a latent, V2-Lite's q_lora_rank is null.
	// The query projection and kv_a go in one launch where fuseSegs built it
	// (decode only).
	qkFused := !skip["qkvo"] && l.mlaQK != nil && bs.rows == 1 && !g.TableSplit
	if qkFused {
		qw, qo := l.wq, bs.q
		if p.QLoraRank != 0 {
			qw, qo = l.wqa, bs.mlaQA
		}
		if *errp == nil {
			*errp = lc.launch(l.mlaQK.kern, l.mlaQK.blocks, l.mlaQK.w, bs.a, bs.ax,
				qw.qs, qw.d, qw.sc, qo, l.wkva.qs, l.wkva.d, l.wkva.sc, bs.mlaKV)
		}
		g.SegFused++
	} else if !skip["qkvo"] {
		if p.QLoraRank != 0 {
			mvrun(l.mvQA, l.wqa, bs.mlaQA)
		} else {
			mvrun(l.mvq, l.wq, bs.q)
		}
	}

	// ── the shared latent and the shared rotary key ──────────────────────
	//
	// One projection for both k and v: kv_a emits KVLoraRank+NRot floats, one
	// for the whole layer. It runs here, off the residual's quantization,
	// before the query's second step replaces it.
	if !skip["qkvo"] && !qkFused {
		mvrun(l.mvKVA, l.wkva, bs.mlaKV)
	}
	// The lightning indexer's key and weights read the same quantization.
	idx := pc != nil && pc.pk.idx != nil
	if idx {
		g.idxKeyAndWeights(lc, bs, l, 1, pc, mvrun)
	}
	if p.QLoraRank != 0 {
		// The norm covers the query latent only, out of place (RULE 13).
		if !skip["norm"] {
			lc.la(bs.mlaNormQAPart, bs.mlaQAParts, bs.mlaQA, bs.mlaQAPart)
			lc.la(bs.mlaNormQAApply, p.QLoraRank, bs.mlaQA, l.nQA, bs.mlaQAPart, bs.mlaQAn)
		}
		if !skip["quant"] {
			lc.la(bs.mlaQuantQA, oldkernels.QuantizeThreads(p.QLoraRank/32), bs.mlaQAn, bs.a, bs.ax)
		}
		if !skip["qkvo"] {
			mvrun(l.mvQB, l.wqb, bs.q)
		}
		if idx {
			g.idxQuery(lc, bs, l, 1, pc, mvrun)
		}
	}

	// ── the latent norm, over the first KVLoraRank floats only ───────────
	//
	// The rotary tail is not normed; the kernel is built for lat elements, so
	// it is untouched by construction.
	if !skip["norm"] {
		lc.la(bs.mlaNormKVPart, bs.mlaParts, bs.mlaKV, bs.mlaKVPart)
		lc.la(bs.mlaNormKVApply, lat, bs.mlaKV, l.nKVA, bs.mlaKVPart, bs.mlaKVn)
	}

	// ── the cache row: the normed latent, then the rotated key ───────────
	//
	// Two writes into one row at two offsets: bs.koff holds pos*row and
	// bs.mlaPOff pos*row+lat, so the rotation lands straight in the cache. The
	// row is the key and its prefix the value (kvPair.value).
	if !skip["rope"] && pc != nil {
		// Paged: the latent at the row's head of this position's page, and
		// the rotated key after it, rotated into staging first.
		lc.la(bs.mlaSliceKPE, rot, bs.mlaKV, bs.mlaKPE)
		pc.writeLatent(lc, 1, bs.mlaKVn, bs.mlaKPE, bs.zero, bs, !p.NoPosEnc)
	} else if !skip["rope"] {
		lc.la(bs.mlaCopyLat, lat, bs.mlaKVn, kvp.kc, bs.koff)
		lc.la(bs.mlaSliceKPE, rot, bs.mlaKV, bs.mlaKPE)
		if p.NoPosEnc {
			lc.la(bs.mlaNoPEK, rot, bs.mlaKPE, kvp.kc, bs.mlaPOff)
		} else {
			lc.la(bs.mlaRopeK, rot/2, bs.mlaKPE, bs.cs, bs.mlaPOff, kvp.kc)
		}
	}

	// ── the query's two halves, compacted ────────────────────────────────
	//
	// The nope half is the head of each head and the rotary half its tail, at
	// stride HeadDim, which neither the rotation nor the absorb can address, so
	// both are compacted (as the host does).
	lc.la(bs.mlaSliceNope, nh*nope, bs.q, bs.mlaNope)
	lc.la(bs.mlaSlicePE, nh*rot, bs.q, bs.mlaPE)
	if !skip["rope"] {
		if p.NoPosEnc {
			lc.la(bs.mlaNoPEQ, nh*rot, bs.mlaPE, bs.mlaPErot, bs.zero)
		} else {
			lc.la(bs.mlaRopeQ, nh*rot/2, bs.mlaPE, bs.cs, bs.zero, bs.mlaPErot)
		}
	}

	// ── absorb W_kᵀ into the query ───────────────────────────────────────
	//
	// One indexed launch, not NHead matvecs: slot h reads bs.mlaNope at h*nope
	// and the bank's sheet h (SlotAct), with the identity selection.
	if !skip["quant"] {
		lc.la(bs.mlaQuantNope, oldkernels.QuantizeThreads(nh*nope/32), bs.mlaNope, bs.a, bs.ax)
	}
	// Which bank is which is a container fact: attn_k_b is stored already
	// transposed and attn_v_b verbatim, and they can be the same shape. The
	// fault arm swaps them.
	absMV, absW, unMV, unW := l.mvKB, l.wkb, l.mvVB, l.wvb
	if g.kb.mlaFault == MLAFaultSwapBanks {
		absMV, absW, unMV, unW = l.mvVB, l.wvb, l.mvKB, l.wkb
	}
	if !skip["qkvo"] {
		mvid(absMV, absW, bs.mlaAbsRaw, bs.mlaIdent)
	}
	// The row-wide query is assembled by copies at the per-head offsets
	// initScratch wrote, since an indexed matvec writes slot h at h*Rows.
	lc.la(bs.mlaCopyAbs, nh*lat, bs.mlaAbsRaw, bs.mlaAbs, bs.mlaAbsOff)
	lc.la(bs.mlaCopyPE, nh*rot, bs.mlaPErot, bs.mlaAbs, bs.mlaPEOff)

	// ── attention, at two widths over one buffer ─────────────────────────
	if !skip["attn"] && idx {
		if *errp == nil {
			*errp = g.idxAttend(s, lc, 1, pc, bs.mlaAbs, bs.n, bs.mlaAcc)
		}
	} else if !skip["attn"] && pc != nil {
		if *errp == nil {
			*errp = pc.pk.attend(s, lc, pc.pl, bs.mlaAbs, pc.k, pc.k, bs.n, bs.mlaAcc, pc.tab, pc.desc, nil)
		}
	} else if !skip["attn"] {
		if !skip["scores"] {
			kg := (nCap + bs.ktile - 1) / bs.ktile
			if bs.scoresWarp {
				// One warp-wide workgroup per (head, position).
				if *errp == nil {
					*errp = lc.launch(bs.scores, p.NHead*nCap, ir.SubgroupLanes, bs.mlaAbs, kvp.kc, bs.n, bs.att)
				}
			} else {
				lc.la(bs.scores, p.NHead*kg, bs.mlaAbs, kvp.kc, bs.n, bs.att)
			}
		}
		if *errp == nil && !skip["softmax"] {
			// Not la(): the softmax group is not 128 wide. MLA carries no
			// attention sinks -- that is gpt-oss -- so this is the plain arm.
			*errp = lc.launch(sm, smG, smW, bs.att, bs.n, bs.prob)
		}
		// kvp.value() (which is kvp.kc) so the accumulate reads the value by
		// name.
		if !skip["acc"] {
			lc.la(bs.attnAcc, nh*lat, bs.prob, kvp.value(), bs.n, bs.mlaAcc)
		}
	}

	// ── un-absorb W_v and project out ────────────────────────────────────
	//
	// The second gather, with the first's contract: head h consumes its own
	// attention result (bs.mlaAcc at h*KVLoraRank) against its own sheet of
	// attn_v_b.
	if !skip["quant"] {
		lc.la(bs.mlaQuantAcc, oldkernels.QuantizeThreads(nh*lat/32), bs.mlaAcc, bs.a, bs.ax)
	}
	if !skip["qkvo"] {
		mvid(unMV, unW, bs.mlaOut, bs.mlaIdent)
	}
	// Kimi-K3's output gate (k3GatedMLA) and no bias: the o bias is added by
	// the caller on bs.mvOut.
	if !skip["quant"] {
		lc.la(bs.mlaQuantOut, oldkernels.QuantizeThreads(nh*p.HeadDimV/32), g.k3GatedMLA(lc, bs, l, 1, bs.mlaOut),
			bs.a, bs.ax)
	}
	if !skip["qkvo"] {
		mvrun(l.mvo, l.wo, bs.mvOut)
	}
}

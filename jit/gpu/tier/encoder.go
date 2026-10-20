package tier

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// layersSession is the device side of layersOnce: the staging writes, the
// launches (or a graph replay) and the readback, in one Session. Its arguments
// are g.sub's; see submitArgs.
func (g *devTier) layersSession(s backend.Session) {
	// The persistent path (persist.go) runs the submission through psess,
	// which issues what the launcher holds before any other call on the
	// session. Entered again rather than s reassigned below: a reassigned
	// parameter that closures capture moves to the heap, five objects a token.
	if g.sub.persist && g.sub.R == 1 && g.sub.rag == nil && g.launchTo.ps != nil &&
		g.launchTo.ps.sync != nil && g.launchTo.psess.Session == nil {
		g.launchTo.psess = persistSession{Session: s, t: &g.launchTo, g: g}
		g.layersSession(&g.launchTo.psess)
		g.launchTo.psess = persistSession{}
		return
	}
	a := &g.sub
	// The session the submission runs for, carried in its recording's key.
	sid := a.key.sid
	bs, hb, head, lo, hi, pos, x := a.bs, a.hb, a.head, a.lo, a.hi, a.pos, a.x
	hx, hcs, hcsSWA, pn, koffs, kposs, rposs := a.hx, a.hcs, a.hcsSWA, a.pn, a.koffs, a.kposs, a.rposs
	p, rag, pe, key := a.p, a.rag, a.pe, a.key
	R, nrow, kvDim, kst, gatherN, rrows := a.R, a.nrow, a.kvDim, a.kst, a.gatherN, a.rrows
	fold, ropeDev, argmax, sampleK, samplePen, pick := a.fold, a.ropeDev, a.argmax, a.sampleK, a.samplePen, a.pick
	ragHead, parOK, allRows, hNorm := a.ragHead, a.parOK, a.allRows, a.hNorm
	var err error
	direct := false
	defer func() { a.err, a.direct = err, direct }()
	// Runs first: no admitted sheet is in flight when the call returns.
	defer func() {
		if len(g.adm) > 0 || len(g.admPub) > 0 {
			if aerr := g.finishAdmits(s); err == nil {
				err = aerr
			}
		}
	}()
	w := func(b backend.Buf, p []byte) {
		if err == nil {
			t := time.Now()
			err = s.Write(b, p)
			g.TSubStage += time.Since(t)
		}
	}
	// fsrc is the float buffer the current int8 activation was quantized
	// from: a launch writing (bs.a, bs.ax) quantizes its first buffer, and
	// that shape is the record. A float weight reads it in place of a
	// scale plane (see mainkernels.MatVec), so it keeps its precision.
	var fsrc backend.Buf
	// actMemo is the ActF16 conversion the last batched matvec left in
	// g.f16Buf: which kernel (its layout and shape) and from which float
	// source. A launch through la or laRows may rewrite any buffer, so each
	// one forgets it; runBatched sets it after its own.
	var actMemoK backend.Kernel
	var actMemoSrc backend.Buf
	// lc is the submission's launch path; its la is the ordinary launch
	// (see launcher).
	g.launchTo.s = s
	if ps, ok := s.(*persistSession); ok {
		g.launchTo.s = ps.Session
	}
	persist := g.launchTo.psess.Session != nil
	defer func() { g.launchTo.s = nil }()
	lc := &launcher{to: &g.launchTo, err: &err, fsrc: &fsrc,
		memoK: &actMemoK, memoSrc: &actMemoSrc, a: bs.a, ax: bs.ax, g: g}

	// lamv launches a decode matvec at its own workgroup width: an in-group
	// split is 32 rows by Split segments (mainkernels.GroupSplitWidth), every
	// other matvec 128. The thread count is the same either way.
	lamv := func(m mv, k backend.Kernel, threads int, bufs ...backend.Buf) {
		w := 128
		if m.group {
			w = mainkernels.GroupSplitWidth(m.split)
		} else if m.width > 0 && k == m.kern {
			w = m.width
		}
		if err == nil {
			err = lc.launch(k, (threads+w-1)/w, w, bufs...)
		}
	}
	// dOf is what a matvec takes in the scale-plane slot.
	dOf := func(r *resident) backend.Buf {
		if !kernels.IsFloat(r.t) {
			return r.d
		}
		if fsrc == nil && err == nil {
			err = fmt.Errorf("tier: a %v matvec ran before any activation was quantized", r.t)
		}
		return fsrc
	}
	// runBatched launches a batched twin: straight into dst, or, when it
	// splits k (a ragged step on a card with no matrix instruction; see
	// ragMV), into the partials and a Reduce over every row of every token.
	runBatched := func(bm mv, r *resident, dst, a, ax, d, f backend.Buf) {
		if bm.act != nil {
			// sm_70's tensor-core twin reads the float activation as f16.
			//
			// Once per source and layout, not once per matvec: q, k and v read
			// one normed row, as do gate and up. A layout is the kernel (Q6_K's
			// step order is not Q4_K's).
			if bm.act != actMemoK || f != actMemoSrc {
				lc.la(bm.act, bm.actN, f, g.f16Buf)
				g.ActF16Launches++
			}
			memoK, memoSrc := bm.act, f
			defer func() { actMemoK, actMemoSrc = memoK, memoSrc }()
			out := dst
			if bm.split > 1 {
				out = g.partBuf
			}
			if bm.bias != nil {
				lc.law(bm.kern, bm.threads, bm.width, r.qs, d, r.sc, g.f16Buf, out, bm.bias)
			} else {
				lc.law(bm.kern, bm.threads, bm.width, r.qs, d, r.sc, g.f16Buf, out)
			}
			if t := bm.tail; t != nil {
				if bm.bias != nil {
					lc.law(t.kern, t.threads, t.width, r.qs, d, r.sc, g.f16Buf, out, bm.bias)
				} else {
					lc.law(t.kern, t.threads, t.width, r.qs, d, r.sc, g.f16Buf, out)
				}
			}
			if bm.split > 1 {
				lc.la(bm.red, bm.redN, g.partBuf, dst)
			}
			return
		}
		n := bm.threads
		if n == 0 {
			n = bm.rows * bm.groups * max(bm.split, 1)
		}
		out := dst
		if bm.split > 1 && !bm.group {
			out = g.partBuf
		}
		if bm.bias != nil {
			lamv(bm, bm.kern, n, r.qs, d, r.sc, a, ax, out, bm.bias)
		} else {
			lamv(bm, bm.kern, n, r.qs, d, r.sc, a, ax, out)
		}
		if bm.split > 1 && !bm.group {
			lc.la(bm.red, bm.redN, g.partBuf, dst)
		}
	}
	mvrun := func(m mv, r *resident, dst backend.Buf) {
		if R > 1 {
			// The batched twin has no split arm: a prefill chunk already has token
			// parallelism, so MatVec refuses NTok > 1 with Split > 1.
			bm, ok := g.batchMV(m, R, bs.tok)
			if rag != nil {
				bm, ok = g.ragStepMV(m, R, len(rag.pos))
			}
			if !ok {
				if err == nil {
					err = errNoBatch
				}
				return
			}
			runBatched(bm, r, dst, bs.a, bs.ax, dOf(r), fsrc)
			return
		}
		if g.TableSplit && m.alt != nil {
			m = *m.alt
		}
		out := dst
		if m.split > 1 && !m.group {
			out = g.partBuf
		}
		// The grid divides by the row tile; an oversized one is silent, since
		// surplus threads clamp to the last work item.
		n := m.rows * m.split
		if m.rowt > 1 {
			n /= m.rowt
		}
		if m.bias != nil {
			lamv(m, m.kern, n, r.qs, dOf(r), r.sc, bs.a, bs.ax, out, m.bias)
		} else {
			lamv(m, m.kern, n, r.qs, dOf(r), r.sc, bs.a, bs.ax, out)
		}
		if m.split > 1 && !m.group {
			lc.la(m.red, m.rows, g.partBuf, dst)
		}
	}
	// skip holds WithSkip's bisection probes (scores, softmax, acc, attn, ffn,
	// ...), which produce wrong tokens on purpose: skipping a phase and
	// re-timing the prompt prices it without serialising the stream.
	skip := map[string]bool{}
	if R > 1 {
		for k := range g.kb.skip {
			skip[k] = true
		}
	}
	// laRows launches one RMSNormGroup-thread group per row.
	laRows := func(k backend.Kernel, bufs ...backend.Buf) {
		actMemoK, actMemoSrc = nil, nil
		if err == nil {
			err = lc.launch(k, R, mainkernels.RMSNormGroup, bufs...)
		}
	}
	// norm is RMSNorm or LayerNorm, decided by the block once; LayerNorm has
	// a pass in between, a bias, and an apply reading two partial arrays.
	norm := func(src, w, wb backend.Buf) {
		// No pre-norm (OLMo 2): the branch reads the residual as it is.
		if w == nil {
			lc.la(bs.copyRows, R*p.NEmbd, src, bs.h)
			return
		}
		if bs.rms != nil && bs.normVar == nil && wb == nil {
			laRows(bs.rms, src, w, bs.h)
			return
		}
		lc.la(bs.normPart, R*bs.parts, src, bs.part)
		if bs.normVar != nil {
			lc.la(bs.normVar, R*bs.parts, src, bs.part, bs.part2)
			lc.la(bs.normApply, R*p.NEmbd, src, w, bs.part, bs.part2, wb, bs.h)
			return
		}
		lc.la(bs.normApply, R*p.NEmbd, src, w, bs.part, bs.h)
	}
	// Scanned rather than counted on the tier, so no release path can let a
	// counter drift.
	streamed := false
	for li := lo; li < hi && !streamed; li++ {
		// A batched mixture reads its selection home too (moegroup.go).
		if l := g.layers[li]; l != nil && (l.stream != nil || (R > 1 && l.router != nil)) {
			streamed = true
		}
	}
	// addRes is a block's residual add, out = x + y, with y scaled first where
	// the model scales its block outputs (residScaled): never fused into a
	// projection or a norm, which carry no scale.
	scaled := residScaled(p)
	addRes := func(x, y, out backend.Buf) {
		if scaled {
			if g.kb.scaleFault == ScaleFaultResidual {
				lc.la(bs.addE, R*p.NEmbd, x, y, out)
				return
			}
			lc.la(bs.addS, R*p.NEmbd, x, y, bs.rscale, out)
			g.ResidScaled++
			return
		}
		lc.la(bs.addE, R*p.NEmbd, x, y, out)
	}
	// A block-end residual add held for the next block's attention norm to do
	// in its own launch (foldAddNorm): x = pendX + pendY. The encoder and
	// the device program (runProg) both hold and take it.
	var pendX, pendY backend.Buf
	var pendNorm, pendConv backend.Kernel
	// runProg runs block li's device program (devprog.go): each op one launch
	// of the kernel the encoder's unfused path takes for it, over the scratch
	// slot the program assigned its values. A program that attends reads its
	// history through the block's pages in the device's pool.
	// the device program's executor begins here (progratchet_test.go)
	runProg := func(l *layer, li int, prog *devProgram) {
		pk := bs.pkv
		var pl *kvLayerPool
		var kvp *kvPair
		var kvBuf backend.Buf
		switch {
		case prog.attends && pk != nil:
			// The paged history: the pool's layer, each row's page by its
			// descriptor.
			if g.kvp != nil {
				pl = g.kvp.layers[l.kvSrc]
			}
			if pl == nil {
				if err == nil {
					err = fmt.Errorf("tier: block %d has no pages in this device's pool", li)
				}
				return
			}
			kvBuf = pl.k
		case prog.attends:
			// A contiguous history (a non-causal segment's): this session's
			// cache at each row's offset.
			if kvp = g.layers[l.kvSrc].kvOf(sid); kvp == nil {
				if err == nil {
					err = fmt.Errorf("tier: block %d has no KV cache for session %d", li, sid)
				}
				return
			}
			kvBuf = kvp.kc
		}
		rcs := bs.cs
		if bs.csSWA != nil && p.LocalRope(li) {
			rcs = bs.csSWA
		}
		buf := [numSlots]backend.Buf{slotX: bs.x, slotX2: bs.x2, slotH: bs.h, slotQ8: bs.a, slotQ: bs.q, slotK: bs.k,
			slotV: bs.v, slotQR: bs.qr, slotKV: kvBuf, slotXB: bs.xb, slotOut: bs.mvOut, slotG: bs.g, slotU: bs.u, slotAct: bs.act,
			slotQN: bs.qn, slotKN: bs.kn, slotF16: g.f16Buf, slotQT: bs.qt, slotQG: bs.qg, slotOGate: bs.ogate,
			slotOGated: bs.ogated, slotVN: bs.vn, slotQC: bs.qClamp, slotKC: bs.kClamp, slotVC: bs.vClamp, slotSub: bs.sub, slotSide: bs.side,
			slotRLogits: bs.rlogits, slotRLogitsB: bs.rlogitsB, slotRBM: bs.rbm, slotSel: bs.rsel, slotTop: bs.rtop,
			slotRW: bs.rw, slotRWS: bs.rwS, slotEG: bs.eg, slotEU: bs.eu, slotEGB: bs.egB, slotEUB: bs.euB,
			slotEAct: bs.eact, slotEDown: bs.edown, slotEDownB: bs.edownB, slotShG: bs.shg, slotShU: bs.shu,
			slotShAct: bs.shact, slotShOut: bs.shout, slotShLogit: bs.shlogit, slotShSum: bs.shsum,
			slotSelW: bs.ident, slotOnes: bs.wOnes,
			slotMQA: bs.mlaQA, slotMQAn: bs.mlaQAn, slotMKV: bs.mlaKV, slotMKVn: bs.mlaKVn, slotMNope: bs.mlaNope,
			slotMPE: bs.mlaPE, slotMPErot: bs.mlaPErot, slotMKPE: bs.mlaKPE, slotMAbsRaw: bs.mlaAbsRaw, slotMAbs: bs.mlaAbs,
			slotMAcc: bs.mlaAcc, slotMOut: bs.mlaOut, slotMIdent: bs.mlaIdent, slotMKVLat: bs.mlaKV, slotHR: bs.hr,
			slotDMixed: bs.dMixed, slotDZ: bs.dZ, slotDBA: bs.dBA, slotDBRaw: bs.dBRaw, slotDConv: bs.dConv,
			slotDOut: bs.dOut, slotDAct: bs.dAct, slotScB: bs.scB, slotScC: bs.scC, slotScBx: bs.scBx, slotScY: bs.scY,
			slotLRQ8: bs.dLRa, slotDLow: bs.dLowRank, slotDAlpha: bs.dAlpha, slotM1B: bs.m1B, slotM1C: bs.m1C,
			slotM1LowN: bs.m1LowN, slotM1BN: bs.m1BN, slotM1CN: bs.m1CN,
			slotPLEG: bs.pleG, slotPLES: bs.pleS, slotPLEA: bs.pleA, slotPLEO: bs.pleO, slotKRot: bs.krot,
			slotPred: bs.altPred, slotAltM: bs.altM, slotAltB: bs.altB, slotLT: bs.altLT, slotLaur: bs.altLaur, slotGG: bs.gg}
		// A block-selecting indexer's buffers (msa.go, qsa.go).
		var ms *msaScratch
		var qs *qsaScratch
		var ix *idxScratch
		if pk != nil {
			ms, qs, ix = pk.msa, pk.qsa, pk.idx
		}
		switch {
		case ix != nil:
			buf[slotIKRaw], buf[slotIdxW], buf[slotIKn], buf[slotIKRot], buf[slotIQRaw], buf[slotIQRot] = ix.kRaw, ix.wts, ix.kn, ix.kRot, ix.qRaw, ix.qRot
		case ms != nil:
			buf[slotIKRaw], buf[slotIQRaw], buf[slotIQn], buf[slotIKn], buf[slotIQRot], buf[slotKAll] = ms.kRaw, ms.qRaw, ms.qn, ms.kn, ms.qRot, ms.kAll
		case qs != nil:
			buf[slotIKRaw], buf[slotIQRaw], buf[slotIQn], buf[slotIQRot], buf[slotKAll] = qs.kRaw, qs.qRaw, qs.qn, qs.qRot, qs.kAll
		}
		if x := bs.q4e; x != nil {
			buf[slotSXN], buf[slotSQ8], buf[slotSQ8LR], buf[slotSLo], buf[slotSLo2] = x.xn, x.aHD, x.aLR, x.lo, x.lo2
			buf[slotSGate], buf[slotSMix], buf[slotSInj], buf[slotSW] = x.gate, x.mix, x.inj, x.w
			buf[slotNEmb], buf[slotSQ8W], buf[slotNKey], buf[slotNVal], buf[slotNKn], buf[slotNQn] = x.emb, x.aW, x.key, x.val, x.kn, x.qn
			buf[slotNDot], buf[slotNGv], buf[slotNGvn], buf[slotNConv], buf[slotNXg] = x.dot, x.gv, x.gvn, x.conv, x.xg
		}
		buf[slotCIn], buf[slotCOut], buf[slotGC], buf[slotUC] = bs.cin, bs.cout, bs.gClamp, bs.uClamp
		copy(buf[slotVal0:slotVal0+numVals], g.valBufs[:])
		if x := bs.ds4; x != nil {
			buf[slotRBias], buf[slotHFlat], buf[slotHFlatN], buf[slotHMix], buf[slotHC], buf[slotHColl] = x.rbias, x.flat, x.flatn, x.mix, x.hc, x.coll
			buf[slotCQA], buf[slotCQAN], buf[slotCQ], buf[slotCQN], buf[slotCQR] = x.qa, x.qan, x.q, x.qn, x.qr
			buf[slotCKRaw], buf[slotCKN], buf[slotCKR], buf[slotCKV], buf[slotCG] = x.kraw, x.kn, x.kr, x.ckv, x.cg
			buf[slotCIKV], buf[slotCIG], buf[slotCIW], buf[slotCIQ], buf[slotCIQR] = x.ikv, x.ig, x.iw, x.iq, x.iqr
			buf[slotCPE], buf[slotCPEN], buf[slotCPER], buf[slotCIPE], buf[slotCIPEN], buf[slotCIPER] = x.pe, x.pen, x.per, x.ipe, x.ipen, x.iper
			buf[slotCSel], buf[slotCGK], buf[slotCAO], buf[slotCAOR], buf[slotCOA] = x.sel, x.gk, x.ao, x.aor, x.oa
		}
		if x := bs.k3; x != nil {
			buf[slotBScores], buf[slotBMix], buf[slotMGate], buf[slotMGated] = x.scores, x.mix, x.mgate, x.mgated
			buf[slotLat], buf[slotLatAcc], buf[slotLatN] = x.lat, x.latAcc, x.latN
		}
		// The rewrites legal at one row, on a prompt chunk and on a ragged
		// step differ: each class keeps its own fused program.
		fc := l.fusedFor(R, rag)
		if fc.bs != bs || fc.fault != faultSig(l, &bs.p) {
			// The prologue builds every block's variant before the
			// submission (ensureFused); one built here is a step that
			// compiled inside a submission, counted so a gate holds it to
			// zero. Released by defer: a generator that panics must surface,
			// not leave g.mu held for the next lock to deadlock on.
			g.GenInStep++
			func() {
				took := g.subLock()
				defer g.subUnlock(took)
				g.genInSub = true
				defer func() { g.genInSub = false }()
				g.ensureFusedLocked(l, prog, bs, R, rag)
			}()
		}
		prog = fc.prog
		if fc.valErr != nil {
			if err == nil {
				err = fmt.Errorf("block %d: %w", li, fc.valErr)
			}
			return
		}
		l.ran.note(formQKV, uint8(fc.qkv))
		if fc.dense {
			l.ran.note(formGateUp, uint8(fc.gu))
			l.ran.note(formDown, uint8(fc.down))
		}
		// A residual add the previous block held (the fold, below) runs in
		// this block's first launch where that is the attention norm writing
		// the first GEMM's operand: the kernel foldAddNorm built for it.
		first := 0
		if pendX != nil {
			if o := prog.ops[0]; o.Kind == dkNormF16 && o.Role == jlm.RoleAttnNorm && prog.slots[o.In[0]] == slotX {
				y := pendY
				if g.addNormBroken() {
					y = pendX
				}
				laRows(pendNorm, pendX, y, l.nAttn, bs.x, bs.h, g.f16Buf)
				actMemoK, actMemoSrc = pendConv, bs.h
				g.NormF16++
				g.AddNormFolded++
				first = 1
			} else {
				addRes(pendX, pendY, bs.x)
			}
			pendX, pendY = nil, nil
		}
		qOff, kro := bs.zero, bs.zero
		if R > 1 {
			qOff, kro = bs.roff, bs.kroff
		}
		gst := genState{bs: bs, slots: prog.slots}
		var sk backend.Buf
		if p.AttnSinks {
			if sk = l.sinks; sk == nil {
				sk = bs.noSink
			}
		}
		// Over rows a mixture's values live in the grouped scratch (bs.mg),
		// and its ops launch that scratch's kernels (groupRun for the
		// experts).
		mg := bs.mg
		rowsMoE := R > 1 && prog.experts
		if rowsMoE {
			buf[slotOnes] = mg.ones
			buf[slotRLogits], buf[slotRLogitsB], buf[slotRBM], buf[slotSel], buf[slotTop], buf[slotRW], buf[slotRWS] =
				mg.rlog, mg.rlogB, mg.rbm, mg.rsel, mg.rtop, mg.rw, mg.rwS
			buf[slotEG], buf[slotEU], buf[slotEGB], buf[slotEUB], buf[slotEAct], buf[slotEDown], buf[slotEDownB] =
				mg.eg, mg.eu, mg.egB, mg.euB, mg.eact, mg.edown, mg.edownB
		}
		// Over rows latent attention's values live in the batched latent
		// scratch (bs.mlab), its absorbs grouped by head.
		ml := bs.mlab
		rowsMLA := R > 1 && prog.latent
		if rowsMLA {
			buf[slotMQA], buf[slotMQAn], buf[slotMKV], buf[slotMKVLat], buf[slotMKVn] = ml.qa, ml.qan, ml.kv, ml.kvLat, ml.kvn
			buf[slotMNope], buf[slotMPE], buf[slotMPErot], buf[slotMKPE] = ml.qNope, ml.pe, ml.pErot, ml.kpe
			buf[slotMAbsRaw], buf[slotMAbs], buf[slotMAcc], buf[slotMOut] = ml.absRaw, ml.abs, ml.accOut, ml.out
			// The head-major rows headMajor's gathers write, in the order
			// they come: storage the latent scratch sizes for R.
			buf[slotPool0], buf[slotPool1] = ml.nopeHM, ml.accHM
		}
		var gr groupRun
		skipNext := false
		at := func(v devVal) backend.Buf { return buf[prog.slots[v]] }
		// A traced run reads each graph op's values back (Config.OpTrace).
		var tr *opTracer
		if g.OpTrace != nil && prog.graph != nil && err == nil {
			tr = &opTracer{g: g, s: s, prog: prog, rows: bs.rows, R: R, li: li, ne: p.NEmbd, at: at, src: -1}
		}
		for oi, o := range prog.ops[first:] {
			if tr != nil && err == nil {
				tr.begin(o.Src)
			}
			if skipNext {
				skipNext = false
				continue
			}
			if R > 1 && o.OneRow {
				continue // what reads it over rows produces it itself
			}
			// An op about to write the row the last binary16 conversion read
			// voids it: the next GEMM converts the row's new value. An op
			// that makes the conversion itself sets it after writing.
			for _, v := range o.Out {
				if actMemoSrc != nil && at(v) == actMemoSrc {
					actMemoK, actMemoSrc = nil, nil
				}
			}
			g.ProgOps++
			// An op with generated launches runs them and nothing else; the
			// block's last add stays with the executor where the next
			// block's norm may fold it (foldAddNorm).
			if len(o.Gen) != 0 && o.Gen[0].former == 0 {
				{
					gst.cs, gst.qOff, gst.kOff = rcs, qOff, kro
					gst.moe, gst.cols = rowsMoE, gr.cols
					if mg != nil {
						gst.colPair = mg.colPair
					}
					// Only a matvec reads a weight (devOp.W is 0, a real index,
					// on every other op).
					isMat := o.Kind == dkMatVec || o.Kind == dkMatVecRes || o.Kind == dkMatVecGate || o.Kind == dkExpertMatVec
					if isMat && prog.f16 && len(o.In) > 0 {
						// A binary16 GEMM reads the float row (a float weight
						// through the scale slot).
						fsrc = at(o.In[0])
					}
					if isMat && o.W >= 0 && o.W < len(l.rs) && *l.rs[o.W] != nil {
						r := *l.rs[o.W]
						gst.qs, gst.d, gst.sc, gst.mbias = r.qs, dOf(r), r.sc, (*l.ms[o.W]).bias
					}
					gst.a, gst.ax = bs.a, bs.ax
					gst.f16, gst.memoK, gst.memoSrc = g.f16Buf, actMemoK, actMemoSrc
					gst.split = g.partBuf
					if pl != nil && pk != nil {
						gst.kpool, gst.vpool, gst.tab, gst.prow = pl.k, pl.v, pl.tab, pk.pRow
					}
					if o.Kind == dkGatherHeads {
						gst.perm = g.perms[[2]int{o.Heads, R}]
					}
					runGen(lc, &o, R, bs.rows, &buf, &gst)
					actMemoK, actMemoSrc = gst.memoK, gst.memoSrc
				}
				switch o.Kind {
				case dkQuant:
					// A binary16 GEMM converts the float row the quantize read.
					fsrc = at(o.In[0])
				case dkNormQuant, dkActQuant:
					fsrc = at(o.Out[0])
				case dkAddNormQuant:
					fsrc = at(o.Out[1])
				}
				g.GenOps++
				continue
			}
			if R > 1 && o.OneRow {
				continue
			}
			// No generated kernel for the op: nothing else runs one.
			if err == nil {
				err = fmt.Errorf("tier: block %d: op %d (%v) has no generated kernel: %v", li, first+oi, o.Kind, o.Why)
			}
			return
		}
		if tr != nil && err == nil {
			tr.flush()
		}
		// A recurrence's state parity flips once its launches are queued, and
		// only while the submission is good: a block past a failure must not
		// flip, or the next token reads a half nothing wrote.
		if prog.scans && err == nil {
			g.stepFlip(sid, l)
			g.recSteps++
		}
		g.ProgBlocks++
	} // the device program's executor ends here (progratchet_test.go)
	// emit issues the whole token's launches. It is a closure so the same code
	// both runs the sequence and records it: one source for both means a graph
	// cannot differ from the launches it replaces.
	emit := func() {
		// The rotary table first: every block reads it and nothing writes it again.
		// Both launches are pure functions of buffers, so a replay rebuilds it from
		// whatever bs.rpos holds this token.
		if ropeDev {
			lc.la(bs.ropeTable, rrows*bs.ropeNPairs, bs.cs, bs.ropeTab, bs.ropeConst, bs.rpos)
			if p.SWAPeriod > 0 && bs.csSWA != nil {
				lc.la(bs.ropeTable, rrows*bs.ropeNPairs, bs.csSWA, bs.ropeTabSWA, bs.ropeConst, bs.rpos)
			}
			g.ds4Tables(lc, bs, R)
		}
		// No add is held across submissions: each emit starts clean.
		pendX, pendY, pendNorm, pendConv = nil, nil, nil, nil
		for li := lo; li < hi; li++ {
			l := g.layers[li]
			// The previous block's held launches go out; this block's are
			// held when it is one the path takes.
			lc.persistBlock(persist && !streamed && persistable(l, bs))
			// A block whose every batched matvec is on sm_70's f16 path never reads
			// the int8 activation, so it skips the quantize: the norm runs alone and
			// fsrc names the float row the ActF16 conversion reads.
			// Not an AltUp block, whose LAuReL branch quantizes between the norm
			// and the attention.
			voltaOnly := R > 1 && rag == nil && g.voltaBlock(l, R, bs.tok) && l.altRouter == nil && l.ds4 == nil &&
				l.k3 == nil && l.q4e == nil
			// A held add this block's norm will not take runs alone first.
			if pendX != nil && !g.foldTakes(l, R, voltaOnly, skip) {
				addRes(pendX, pendY, bs.x)
				pendX, pendY = nil, nil
			}
			// The device program, where it lowers the block and the step is
			// one decode row over the paged history (devprog.go).
			why := progRanProgram
			switch {
			case l.prog == nil:
				why = progNotLowered
			case g.NoProgram:
				why = progNoProgram
			case l.prog.experts && R > 1 && (bs.mg == nil || bs.mg.rows != R || l.prog.streamed):
				why = progRowsExperts
			case l.prog.latent && R > 1 && (bs.mlab == nil || bs.mlab.rows != R):
				why = progRowsLatent
			case l.prog.streamed && (g.StreamProbe || g.StreamPrefetch || g.StreamFixedSel):
				why = progStreamTuning
			case l.prog.attends && R > 1 && rag == nil && !p.NonCausal && (bs.pkv == nil || !bs.pkv.prefill):
				// A causal block's rows attend through the paged prefill; a
				// non-causal one's over its contiguous history (ncAttend).
				why = progRowsNotPrefill
			case len(skip) != 0 || g.blockFaulted():
				why = progFault
			case l.prog.attends && bs.pkv != nil && ((R == 1 || rag != nil) && (bs.pkv.cur == nil || bs.pkv.prefill) ||
				R > 1 && rag == nil && bs.pkv.pf == nil),
				l.prog.attends && bs.pkv == nil && !p.NonCausal:
				// A paged block whose call lacks its plan, or a causal block
				// with no paged history.
				why = progNotPaged
			case l.prog.ropes && (bs.ropeQ == nil || l.noRope):
				why = progRotary
			case g.TableSplit:
				why = progTableSplit
			}
			if R > 1 {
				l.progRunRows = why
			} else {
				l.progRun = why
			}
			if why == progRanProgram {
				runProg(l, li, l.prog)
				l.ran.note(formBlock, uint8(blockProgram))
				continue
			}
			// The program could not run this step, and nothing else runs a
			// block: the step fails, naming why.
			if err == nil {
				err = fmt.Errorf("tier: block %d: the program cannot run this step: %v", li, why)
			}
			return
		}
		if pendX != nil {
			addRes(pendX, pendY, bs.x)
		}
		if head != nil && (rag != nil || allRows) {
			// The output norm, the projection of the rows that want logits
			// (they lead) and a per-row argmax over them: their tokens and
			// logits come home. A chunk asking for every row's logits
			// (nn.Head.RowLogits on Layers) takes the same tail.
			hr := g.bs.head
			norm(bs.x, hNorm, g.bs.hNormB)
			lc.la(bs.quantE, mainkernels.QuantizeThreads(R*ePad(p)/32), bs.h, bs.a, bs.ax)
			one, oneOK := mv{}, false
			if ragHead == 1 {
				one, oneOK = g.headOneMV(R) // compiled by prepRagged
			}
			if m := one; oneOK {
				// One row: decode's head matvec at decode's split, reading
				// the activation's first row -- the wanted row's -- and
				// writing that row's logits, the first of ragLogits. A
				// batched twin would project every row of the step to read
				// back one.
				if g.rowsFaulted(rowsHeadone) {
					// Decode's own kernel, which takes the sums to follow
					// the first row's scales: they follow all R rows'.
					m.kern = g.bs.mvHead.kern
				}
				out := bs.ragLogits
				if m.split > 1 && !m.group {
					out = g.partBuf
				}
				nt := m.rows * m.split
				if m.rowt > 1 {
					nt /= m.rowt
				}
				lamv(m, m.kern, nt, hr.qs, dOf(hr), hr.sc, bs.a, bs.ax, out)
				if m.split > 1 && !m.group {
					lc.la(m.red, m.rows, g.partBuf, bs.ragLogits)
				}
			} else {
				hm := bs.ragHead
				if gm, ok := g.groupMV(g.headMV(), ragHead, R); ok {
					hm = gm // the wanted rows only
				}
				runBatched(hm, hr, bs.ragLogits, bs.a, bs.ax, dOf(hr), fsrc)
			}
			// The final softcap, then the argmax over the capped rows: over
			// the wanted rows only, as the one-row and few-row heads project
			// nothing past them, and no row past them is read.
			if bs.ragCapK != nil && !g.rowsFaulted(rowsNocap) {
				lc.la(bs.ragCapK, ragHead*hr.nrows, bs.ragLogits, bs.ragCap)
			}
			if err == nil {
				err = lc.launch(bs.ragArgmax, ragHead, mainkernels.ArgmaxGroup, bs.ragOut(g.Config), bs.ragTok)
			}
		} else if fold {
			// The chunk's last row, into the decode scratch, and the decode
			// scratch's head sequence on it -- mvrun's one-row arm, spelled
			// out because mvrun's R is the chunk's.
			hr := hb.head
			lc.la(hb.spanK, p.NEmbd, bs.x, hb.x, hb.spanOff)
			if hNorm == nil {
				lc.la(hb.copyRows, p.NEmbd, hb.x, hb.h)
			} else {
				lc.la(hb.normPart, hb.parts, hb.x, hb.part)
				lc.la(hb.normApply, p.NEmbd, hb.x, hNorm, hb.part, hb.h)
			}
			lc.la(hb.quantE, mainkernels.QuantizeThreads(p.NEmbd/32), hb.h, hb.a, hb.ax)
			// The label pick projects the picked rows alone (pickMV).
			pickOnly := pick && g.pickMV != nil && hb.headCap == nil
			if pickOnly {
				lc.la(g.pickMV, pickMax, hr.qs, hr.d, hr.sc, hb.a, hb.ax, g.pickOut, g.pickIDs)
				g.PickRows++
			}
			m := hb.mvHead
			if g.TableSplit && m.alt != nil {
				m = *m.alt
			}
			out := hb.logits
			if m.split > 1 && !m.group {
				out = g.partBuf
			}
			n := m.rows * m.split
			if m.rowt > 1 {
				n /= m.rowt
			}
			if !pickOnly {
				lamv(m, m.kern, n, hr.qs, hr.d, hr.sc, hb.a, hb.ax, out)
				if m.split > 1 && !m.group {
					lc.la(m.red, m.rows, g.partBuf, hb.logits)
				}
				if hb.headCap != nil {
					lc.la(hb.headCap, m.rows, hb.logits, hb.logitsCap)
				}
				if pick {
					lc.la(g.pickK, pickMax, hb.headOut(), g.pickIDs, g.pickOut)
				}
			}
			g.HeadFolds++
		} else if head != nil {
			// The output norm and the vocabulary projection, in the same
			// submission: the residual stream never comes home.
			if hNorm == nil {
				// No output norm: the row as it is.
				lc.la(bs.copyRows, R*p.NEmbd, bs.x, bs.h)
			} else if bs.normVar != nil {
				// A LayerNorm head (C6): its mean, variance and apply, with
				// the output norm's bias (zeros where the file has none).
				norm(bs.x, hNorm, bs.hNormB)
			} else {
				lc.la(bs.normPart, R*bs.parts, bs.x, bs.part)
			}
			// HeadNormRepeat is a probe, not a feature: normApply is idempotent, so
			// each repeat costs exactly one extra tiny dispatch, which prices whether
			// the head's fixed per-token cost is dispatch structure or work (see the
			// Metal entries in AGENTS.md).
			if bs.normVar == nil && hNorm != nil {
				for r := 0; r < g.HeadNormRepeat; r++ {
					lc.la(bs.normApply, p.NEmbd, bs.x, hNorm, bs.part, bs.h)
				}
				lc.la(bs.normApply, p.NEmbd, bs.x, hNorm, bs.part, bs.h)
			}
			lc.la(bs.quantE, mainkernels.QuantizeThreads(R*ePad(p)/32), bs.h, bs.a, bs.ax)
			// HeadMatvecRepeat is the same idempotent-repeat probe on the head
			// matvec, pricing it by repetition.
			pickOnly := pick && g.pickMV != nil && bs.headCap == nil && R == 1
			if pickOnly {
				// The label pick projects the picked rows alone (pickMV).
				lc.la(g.pickMV, pickMax, bs.head.qs, bs.head.d, bs.head.sc, bs.a, bs.ax, g.pickOut, g.pickIDs)
				g.PickRows++
			} else {
				for r := 0; r < g.HeadMatvecRepeat; r++ {
					mvrun(bs.mvHead, bs.head, bs.logits)
				}
				mvrun(bs.mvHead, bs.head, bs.logits)
			}
			if bs.headCap != nil {
				lc.la(bs.headCap, bs.mvHead.rows, bs.logits, bs.logitsCap)
			}
			if argmax && err == nil {
				err = lc.launch(g.argmaxK, 1, mainkernels.ArgmaxGroup, bs.headOut(), g.argmaxOut)
			}
			if sampleK > 0 && err == nil {
				err = g.launchSample(lc, bs.headOut(), bs.mvHead.rows, sampleK, samplePen)
			}
			if pick && !pickOnly {
				lc.la(g.pickK, pickMax, bs.headOut(), g.pickIDs, g.pickOut)
			}
		}
		// What the last block held goes out inside emit, so a recording
		// captures it.
		lc.persistFlush()
	}
	// submit gets the sequence onto the device, by replaying a captured graph
	// where the backend has one and by issuing the launches where it does not.
	// The staging writes and the readback stay outside it: a captured memcpy
	// would bake in a Go heap address.
	submit := func() {
		if err != nil {
			return
		}
		tl := time.Now()
		defer func() { g.TSubLaunch += time.Since(tl) }()
		rec, ok := g.recorderOf(s)
		if !ok {
			g.graphOff = true
		}
		// No recording when the key would not hold still or a replay cannot be
		// right:
		// - PerLayerSubmit changes the key every block, so -ab submit with a
		//   resident head compares two changes at once unless NoGraph is set on
		//   both arms.
		// - A prefill chunk's key steps with its position, so it would be
		//   captured every chunk and replayed never. A ragged step is recorded
		//   like decode.
		// - A paging device: a graph holds device addresses, and a page-in
		//   allocates the buffers afresh.
		// - A streamed block: the mid-FFN host readback and upload have no place
		//   in a replay.
		// - History streamed from home: its uploads and waits are per call.
		if (R > 1 && rag == nil) || !parOK || g.NoGraph || g.graphOff || g.PerLayerSubmit ||
			a.paging || streamed || bs.pkv != nil && bs.pkv.st != nil {
			te := time.Now()
			emit()
			g.TEmit += time.Since(te)
			return
		}
		cur, have := g.recs[key]
		if !have {
			tc := time.Now()
			r, e := backend.Record(rec, emit)
			g.TCapture += time.Since(tc)
			if e != nil || err != nil {
				// A capture executes nothing, so the work has still to be
				// issued -- the ordinary way, since a capture that failed
				// once will fail again. A recording that came back around a
				// failed launch is complete as far as the driver knows and
				// wrong as far as jitllm does, so it is retired unused.
				g.graphOff, err = true, nil
				if e != nil {
					g.LastErr = "graphs disabled: " + e.Error()
				}
				if r != nil {
					g.stale = append(g.stale, r)
				}
				emit()
				return
			}
			if g.recs == nil {
				g.recs = make(map[graphKey]backend.Recording)
			}
			g.recs[key], cur = r, r
			g.Captures++
		}
		if e := rec.Replay(cur); e != nil {
			g.graphOff, g.LastErr = true, "graphs disabled: "+e.Error()
			g.stale = append(g.stale, cur)
			delete(g.recs, key)
			emit()
		} else {
			if g.recUsed == nil {
				g.recUsed = map[graphKey]uint64{}
			}
			g.recTick++
			g.recUsed[key] = g.recTick
			if have {
				g.replayRecurrent(sid, lo, hi)
			}
		}
	}

	// Position state is per token, not per block: written once here rather
	// than 54 times across 18 blocks.
	if fold {
		w(hb.spanOff, g.u32b(uint32((nrow-1)*p.NEmbd)))
	}
	if pe != nil {
		hr := g.bs.head
		g.subBytes.embIDs = embIDs(g.subBytes.embIDs, pe.ids, R, hr.nrows)
		w(g.bs.embIds, g.subBytes.embIDs)
		if err == nil {
			err = lc.launch(g.bs.embK, gatherN/128, 128, hr.qs, hr.d, hr.sc, g.bs.embIds, bs.x)
		}
		g.EmbedLaunches++
	} else {
		w(bs.x, f32b(hx))
	}
	if bs.pleIn != nil {
		w(bs.pleIn, f32b(g.pleRows(R*p.PLEWidth)))
	}
	if ropeDev {
		// Four bytes a row in place of NRot*4; the table never crosses the bus.
		w(bs.rpos, rposs)
	} else {
		if bs.cs != nil {
			// nil for a block with no rotary; nothing reads it and there is
			// no buffer to write into.
			w(bs.cs, f32b(hcs))
		}
		if p.SWAPeriod > 0 && bs.csSWA != nil {
			w(bs.csSWA, f32b(hcsSWA))
		}
	}
	w(bs.n, pn)
	if samplePen {
		// The history changes every token; the recording reads its count.
		w(g.sampleArgs, u32b(head.SampleArgs))
	}
	if pick {
		// The ids change every question; the recording reads them.
		w(g.pickIDs, g.pickStage(head))
	}
	if bs.dRows != nil {
		// The recurrent descriptor (see emitLinear), staged here outside any
		// recording: written inside emit it was a memcpy mid-capture, which
		// invalidates a CUDA stream capture. Its row count is nrow, the
		// caller's rows, not the grid R: padding rows must not go through
		// the convolution's shift and the delta rule as tokens. Its slots
		// are where each sequence's state is in the pools, which a
		// recording therefore does not bake.
		w(bs.dRows, g.recDesc(sid, nrow))
	}
	w(bs.koff, koffs)
	if kst > 0 {
		w(bs.kpos, kposs)
	}
	bs.ds4.ds4Writes(w)
	if bs.pkv != nil {
		bs.pkv.qsa.qsaWrites(w)
	}
	g.q4eWrites(bs.q4e, w)
	if pk := bs.pkv; pk != nil && pk.cur != nil {
		w(pk.pRow, u32view(pk.desc))
		if len(pk.descW) > 0 {
			w(pk.pRowW, u32view(pk.descW))
		}
		if pk.prefill {
			pk.pf.prefillWrites(w)
		}
		pk.ragWrites(w)
		if pk.st != nil {
			pk.st.writes(w)
		}
	}
	if bs.mlab != nil && R > 1 {
		// The batched twin of mlaPOff below: each row's rotary key sits at
		// its own cache row plus KVLoraRank -- a padded row's in the trash
		// slot, where its latent went.
		po := g.subBytes.mlaPOff[:0]
		for r := 0; r < R; r++ {
			po = binary.LittleEndian.AppendUint32(po, binary.LittleEndian.Uint32(koffs[4*r:])+uint32(bs.mlab.lat))
		}
		g.subBytes.mlaPOff = po
		w(bs.mlab.pOff, po)
	}
	if bs.mlaPOff != nil {
		// The rotary key's slot inside this position's row: bs.koff holds
		// pos*row (the latent's slot) and the key follows at +KVLoraRank, so the
		// rotation writes straight into the cache.
		w(bs.mlaPOff, g.u32b(uint32(pos*kvDim+bs.mlaLat)))
	}

	submit()

	if err != nil {
		return
	}
	// Read straight into the caller's slice when the lengths agree: copying
	// out of bs.hRaw was a measurable cost for a large vocabulary, and
	// cuMemcpyDtoH takes any host pointer.
	tr := time.Now()
	defer func() { g.TSubRead += time.Since(tr) }()
	// The head's input, before the reads below consume the session's
	// attention: the normed rows a prediction block reads (nn.Head.Hidden).
	if head != nil && head.Hidden != nil && err == nil {
		src, rows := bs.h, 1
		switch {
		case rag != nil || allRows:
			rows = ragHead
		case fold:
			src = hb.h
		}
		if len(head.Hidden) < rows*p.NEmbd {
			err = fmt.Errorf("a %d-float hidden destination for %d rows of %d", len(head.Hidden), rows, p.NEmbd)
		} else {
			err = s.Read(src, f32b(head.Hidden[:rows*p.NEmbd]))
		}
	}
	switch {
	case (rag != nil || allRows) && head != nil:
		tb := slices.Grow(g.subBytes.tok[:0], 4*R)[:4*R]
		g.subBytes.tok = tb
		if err = s.Read(bs.ragTok, tb); err == nil {
			for r := range ragHead {
				head.Tokens[r] = int32(binary.LittleEndian.Uint32(tb[4*r:]))
			}
		}
		// The rows' logits are already in ragLogits, [R][vocab]; the first
		// ragHead rows are the wanted ones.
		if need := ragHead * g.bs.head.nrows; err == nil && head.RowLogits {
			if len(head.Logits) < need {
				err = fmt.Errorf("rows: %d logits for %d rows of %d", len(head.Logits), ragHead, g.bs.head.nrows)
			} else {
				err = s.Read(bs.ragOut(g.Config), f32b(head.Logits[:need]))
			}
		}
		direct = true
	case sampleK > 0:
		if err = s.Read(g.sampleV, f32b(head.SampleVals[:sampleK])); err == nil {
			err = s.Read(g.sampleI, u32b(head.SampleIDs[:sampleK]))
		}
		head.Sampled = err == nil
		// Counted here, not at the launches, which a replayed recording skips.
		g.SampleReads++
		g.SampleLaunches += 2
		if samplePen {
			g.SampleLaunches++
		}
		direct = true
	case pick:
		err = s.Read(g.pickOut, f32b(head.PickVals[:len(head.Pick)]))
		head.Picked = err == nil
		// Counted here, not at the launch, which a replayed recording skips.
		g.PickReads++
		direct = true
	case argmax:
		tok := g.subBytes.one[:]
		if err = s.Read(g.argmaxOut, tok); err == nil {
			head.Token = int32(binary.LittleEndian.Uint32(tok))
		}
		direct = true
	case head != nil && len(head.Logits) == len(hb.hOut):
		err = s.Read(hb.headOut(), f32b(head.Logits))
		direct = true
	case head != nil:
		err = s.Read(hb.headOut(), hb.hRaw)
	case len(x) == nrow*p.ResidW() && (p.Streams() == 1 || nrow == R):
		err = s.Read(bs.x, f32b(x))
		direct = true
	case p.Streams() > 1:
		// Every stream comes home, stream-major at R rows (streamRows cuts it).
		err = s.Read(bs.x, bs.hraw[:R*p.ResidW()*4])
	default:
		err = s.Read(bs.x, bs.hraw[:nrow*p.NEmbd*4])
	}
}

// contigAttend is attention over a contiguous history (kvp: a non-causal
// segment's, or a block whose history is not paged) into bs.xb: the flash
// kernel where built, the one-kernel prefill on sm_70 or Metal's matrix
// unit, else scores, a softcap, a window mask, the softmax and the weighted
// sum, per query chunk where the score planes are sized for one. The block
// encoder and the device program's attention both run it.
func (g *devTier) contigAttend(lc *launcher, bs *blockScratch, l *layer, li, R int, kvp *kvPair, qAtt backend.Buf,
	nCap int, sm backend.Kernel, smG, smW int, skip map[string]bool, fv *flashVariant, flash backend.Kernel, useFlash bool) error {
	p := &bs.p
	qdim := p.NHead * p.HeadDim
	var err error
	if useFlash {
		dst := bs.xb
		if fv.splits > 1 {
			dst = bs.flashPart
		}
		args := []backend.Buf{qAtt, kvp.kc, kvp.vc, bs.n, dst}
		if p.AttnSinks {
			sk := l.sinks
			if sk == nil {
				sk = bs.noSink
			}
			args = append(args, sk)
		}
		if err == nil {
			err = lc.launch(flash, fv.groups, bs.flashWidth, args...)
			if err == nil && fv.splits > 1 {
				ma := []backend.Buf{bs.flashPart, bs.xb}
				if p.AttnSinks {
					ma = append(ma, args[5])
				}
				err = lc.launch(fv.merge, R*p.NHead, 32, ma...)
			}
			g.FlashLaunches++
		}
	} else if bs.fp70 != nil && !skip["scores"] && !skip["softmax"] && !skip["acc"] && !l.windowed {
		// The chunk's attention as one kernel: scores,
		// softmax and the weighted V sum per key tile, the
		// scores never written (mainkernels.FlashPrefill70).
		fk := bs.fp70
		if p.Window(li) > 0 {
			fk = bs.fp70W
		}
		if err == nil {
			err = lc.launch(fk, mainkernels.FlashPrefill70Groups(mainkernels.FlashPrefill70Shape{Heads: p.NHead, Rows: R}),
				mainkernels.FlashPrefill70Threads, qAtt, kvp.kc, kvp.vc, bs.n, bs.xb)
		}
		g.FlashPrefills++
	} else if bs.fpT != nil && !skip["scores"] && !skip["softmax"] && !skip["acc"] && !l.windowed {
		// The same, on Metal's matrix unit (mainkernels.FlashPrefillTile).
		fk := bs.fpT
		if p.Window(li) > 0 {
			fk = bs.fpTW
		}
		if err == nil {
			err = lc.launch(fk, mainkernels.FlashTileGroups(p.NHead, R, bs.fpTile),
				bs.fpTile.Threads(), qAtt, kvp.kc, kvp.vc, bs.n, bs.xb)
		}
		g.FlashPrefills++
	} else {
		// One pass over the rows, or one per query chunk of a vision
		// block whose score planes are sized for a chunk (arows):
		// the chunk's queries gathered into qc, its output scattered
		// back from xc.
		run := func(qIn, out, win backend.Buf, R int) {
			if !skip["scores"] {
				// A local layer takes the windowed pair: bs.n's key counts are
				// causal and shared, while the window is per layer and baked in.
				scores, scoresMMA := bs.scores, bs.scoresMMA
				if p.Window(li) > 0 {
					scores, scoresMMA = bs.scoresW, bs.scoresMMAW
				}
				scores70 := bs.scores70
				if p.Window(li) > 0 {
					scores70 = bs.scores70W
				}
				if scoresMMA != nil {
					// One warp per 16 query rows x 8*attnNT key positions.
					kg := (nCap + 8*g.tiles.attnNT - 1) / (8 * g.tiles.attnNT)
					lc.la(scoresMMA, (R/16)*p.NHead*kg*32, qIn, kvp.kc, bs.n, bs.att)
				} else if scores70 != nil {
					lc.la(scores70, mainkernels.AttnScoresMMA70Warps(p.NHead, R, nCap, volta70AttnMT,
						volta70AttnNT)*32, qIn, kvp.kc, bs.n, bs.att)
				} else {
					// The key axis is covered by ceil(nCap/ktile)
					// threads, which is the same G the kernel derives
					// from pN.
					kg := (nCap + bs.ktile - 1) / bs.ktile
					lc.la(scores, (R/bs.qtile)*p.NHead*kg, qIn, kvp.kc, bs.n, bs.att)
				}
			}
			att := bs.att
			if bs.softcap != nil {
				lc.la(bs.softcap, R*p.NHead*bs.sstride, bs.att, bs.attCap)
				att = bs.attCap
			}
			// Qwen2.5-VL's windowed blocks: every key outside a
			// row's window to -inf before the softmax, the windows
			// of this pass's rows in win.
			// Packed sequences: every key outside a row's own
			// sequence, the same mask over the runs (SetSeqRuns).
			if win != nil {
				lc.la(bs.winMask, R*p.NHead*bs.sstride, att, win, bs.attWin)
				att = bs.attWin
			}
			if err == nil && !skip["softmax"] {
				// Not la(): softmax is the one kernel whose group is
				// not 128 wide. With 32 lanes to a head the group IS
				// the warp, which is what lets the shuffle assume
				// lane == tid.
				if bs.p.AttnSinks {
					sk := l.sinks
					if sk == nil {
						sk = bs.noSink
					}
					err = lc.launch(sm, smG, smW, att, bs.n, bs.prob, sk)
				} else {
					err = lc.launch(sm, smG, smW, att, bs.n, bs.prob)
				}
			}
			if bs.accSplit > 1 {
				lc.la(bs.attnAcc, qdim*bs.accSplit, bs.prob, kvp.vc, bs.n, g.partBuf)
				lc.la(bs.attnRed, qdim, g.partBuf, out)
			} else if !skip["acc"] && bs.acc70 != nil {
				lc.la(bs.acc70, mainkernels.AttnAccMMA70Warps(p.NHead, p.HeadDim, R,
					volta70AccMT, volta70AttnNT)*32, bs.prob, kvp.vc, bs.n, out)
			} else if !skip["acc"] {
				lc.la(bs.attnAcc, R*qdim/bs.atile, bs.prob, kvp.vc, bs.n, out)
			}
		}
		if bs.arows > 0 && bs.arows < R {
			for c := range bs.qoffs {
				lc.la(bs.qGather, bs.arows*qdim, qAtt, bs.qc, bs.qoffs[c])
				run(bs.qc, bs.xc, bs.winOf(l, c), bs.arows)
				lc.la(bs.xScatter, bs.arows*qdim, bs.xc, bs.xb, bs.qoffs[c])
			}
			g.VisionAttnChunks += len(bs.qoffs)
		} else {
			run(qAtt, bs.xb, bs.winOf(l, 0), R)
		}
	}
	return err
}

// q8Pair is the int8 operand buffers of slot s where it is one beside the
// block's own (bs.a, bs.ax): a low-rank pair's, a stream mix's.
func q8Pair(bs *blockScratch, s devSlot) (a, ax backend.Buf, ok bool) {
	switch s {
	case slotLRQ8:
		return bs.dLRa, bs.dLRax, true
	case slotSQ8:
		if x := bs.q4e; x != nil {
			return x.aHD, x.axHD, true
		}
	case slotSQ8LR:
		if x := bs.q4e; x != nil {
			return x.aLR, x.axLR, true
		}
	case slotSQ8W:
		if x := bs.q4e; x != nil {
			return x.aW, x.axW, true
		}
	}
	return nil, nil, false
}

// blockAct is block l's activation kernel: a non-causal block's own where its
// MLP is not the scratch set's (layer.actK), else the set's.
func blockAct(bs *blockScratch, l *layer) backend.Kernel {
	if l.nonCausal && l.actK != nil {
		return l.actK
	}
	return bs.actMul
}

// opTracer reads a block's graph-op values back as its program runs
// (Config.OpTrace): each graph op's inputs before its first device op, its
// outputs after its last. Built only for a traced run.
type opTracer struct {
	g       *devTier
	s       backend.Session
	prog    *devProgram
	rows, R int
	li, ne  int
	at      func(devVal) backend.Buf
	src     int
	ops     []int
	vals    map[graph.ValueID][]float32
}

// begin starts the graph op a device op lowers, flushing the one before.
func (t *opTracer) begin(src []int) {
	if len(src) == 0 || src[0] == t.src || src[0] >= len(t.prog.graph.Ops) {
		return
	}
	t.flush()
	t.src, t.ops, t.vals = src[0], append([]int(nil), src...), map[graph.ValueID][]float32{}
	for _, gi := range t.ops {
		t.read(t.prog.graph.Ops[gi].In)
	}
}

// flush reads the current graph op's outputs and hands the op to the trace.
func (t *opTracer) flush() {
	if t.src < 0 {
		return
	}
	for _, gi := range t.ops {
		t.read(t.prog.graph.Ops[gi].Out)
	}
	t.g.OpTrace(t.li, t.prog.graph, t.ops, t.vals, t.R)
	t.src = -1
}

// read reads the device values of graph values ids, row-major: a residual of
// several streams is stream-major across the rows on the device.
func (t *opTracer) read(ids []graph.ValueID) {
	for _, id := range ids {
		dv, ok := t.prog.gvals[id]
		if !ok {
			continue
		}
		n, st := t.prog.vals[dv].Width, t.prog.stride(dv)
		raw := make([]byte, 4*st*t.rows)
		if t.s.Read(t.at(dv), raw) != nil {
			continue
		}
		sl := t.prog.slots[dv]
		streams := (sl == slotX || sl == slotX2) && n > t.ne && n%t.ne == 0
		f := make([]float32, n*t.R)
		for r := 0; r < t.R; r++ {
			for c := 0; c < n; c++ {
				i := r*st + c
				if streams {
					i = (c/t.ne)*t.rows*t.ne + r*t.ne + c%t.ne
				}
				u := binary.LittleEndian.Uint32(raw[4*i:])
				if t.prog.vals[dv].Elem == elemU32 {
					f[r*n+c] = float32(u) // an index row, as the golden holds it
				} else {
					f[r*n+c] = math.Float32frombits(u)
				}
			}
		}
		t.vals[id] = f
	}
}

package tier

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
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
	p, rag, key := a.p, a.rag, a.key
	R, nrow, kvDim, kst, rrows := a.R, a.nrow, a.kvDim, a.kst, a.rrows
	fold, ropeDev, argmax, sampleK, samplePen, pick := a.fold, a.ropeDev, a.argmax, a.sampleK, a.samplePen, a.pick
	ragHead, parOK, allRows := a.ragHead, a.parOK, a.allRows
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
	// skip holds WithSkip's bisection probes (scores, softmax, acc, attn, ffn,
	// ...), which produce wrong tokens on purpose: skipping a phase and
	// re-timing the prompt prices it without serialising the stream.
	skip := map[string]bool{}
	if R > 1 {
		for k := range g.kb.skip {
			skip[k] = true
		}
	}
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
		first := 0
		qOff, kro := bs.zero, bs.zero
		if R > 1 {
			qOff, kro = bs.roff, bs.kroff
		}
		gst := genState{bs: bs, slots: prog.slots, ops: prog.ops}
		var sk backend.Buf
		if p.AttnSinks {
			if sk = l.sinks; sk == nil {
				sk = bs.noSink
			}
		}
		gst.sink = sk
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
			if len(o.Gen) == 0 && o.Why == whyNothing {
				continue
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
			if len(o.Gen) == 1 && o.Gen[0].former == formAttend {
				// The attention of this call's keys, or its run table's.
				switch {
				case o.Kind == dkAttendGathered:
					o.Gen = g.gatheredAttendAt(&o, prog, bs, fc.rows, pagedKeys(pk.rows))
				case pk.rag.n > 0:
					o.Gen = g.ragAttendAt(&o, prog, bs)
				case pk.pre.on:
					o.Gen = g.preAttendAt(&o, prog, bs, fc.rows)
				default:
					o.Gen = g.attendAt(&o, prog, bs, fc.rows, pagedKeys(pk.rows))
				}
				if o.Gen == nil && err == nil {
					err = fmt.Errorf("tier: block %d: no attention for the call's keys", li)
				}
			}
			if o.Why == whyFused || len(o.Gen) != 0 && o.Gen[0].former == 0 {
				if len(o.Gen) != 0 {
					gst.cs, gst.qOff, gst.kOff = rcs, qOff, kro
					gst.moe = rowsMoE
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
					for si, w := range o.Ws {
						if r := *l.rs[w]; r != nil {
							gst.segs[si] = [4]backend.Buf{r.qs, dOf(r), r.sc, (*l.ms[w]).bias}
						}
					}
					if isMat && o.W >= 0 && o.W < len(l.rs) && *l.rs[o.W] != nil {
						r := *l.rs[o.W]
						gst.qs, gst.d, gst.sc, gst.mbias = r.qs, nil, r.sc, (*l.ms[o.W]).bias
						if !kernels.IsFloat(r.t) {
							// A float weight's product reads its float row as
							// an input of its own; a quantized one its scales.
							gst.d = dOf(r)
						}
					}
					gst.a, gst.ax = bs.a, bs.ax
					gst.f16, gst.memoK, gst.memoSrc = g.f16Buf, actMemoK, actMemoSrc
					gst.split = g.partBuf
					if pl != nil && pk != nil {
						gst.kpool, gst.vpool, gst.tab, gst.prow = pl.k, pl.v, pl.tab, pk.pRow
						gst.prowW, gst.scores, gst.wts, gst.parts = pk.pRowW, pk.sc, pk.wt, pk.part
					}
					if l.pool != nil {
						// The step reads the session's half and writes the other.
						par, _ := g.stepPar(sid, l)
						gst.convIn, gst.convOut = l.pool.c[par], l.pool.c[1-par]
						gst.convW, gst.recDesc = l.ssmConv1d, bs.dRows
						gst.stateIn, gst.stateOut = l.pool.s[par], l.pool.s[1-par]
						if g.recShared {
							gst.stateIn, gst.stateOut = l.pool.s[0], g.recNext
						}
						gst.recDt, gst.recA, gst.recD, gst.recCB = l.ssmDt, l.ssmA, l.ssmD, l.ssmCB
						gst.recNorm, gst.recMid = l.ssmNorm, bs.dOutN
					}
					gst.router, gst.routerB, gst.selB, gst.routeW, gst.shRouter = l.router, l.routerB, l.expSelB, bs.rw, l.shRouter
					if o.Kind == dkGatherHeads {
						gst.perm = g.perms[headOrder(&o, R)]
					}
					runGen(lc, &o, R, fc.rows, &buf, &gst)
					if o.Kind == dkScan && g.recShared && err == nil {
						// The shared form's next state goes home.
						st := bs.p.Recurrent.StateLen
						if cp := g.recCopies[[2]int{st, 1}]; cp != nil {
							lc.la(cp, st, g.recNext, l.pool.s[0], bs.dRows)
						} else {
							err = fmt.Errorf("tier: block %d: no copy home for a shared recurrent state", li)
						}
					}
					actMemoK, actMemoSrc = gst.memoK, gst.memoSrc
				}
				switch o.Kind {
				case dkQuant:
					// A binary16 GEMM converts the float row the quantize read.
					fsrc = at(o.In[0])
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
		if ropeDev {
			lc.la(bs.ropeTable, rrows*bs.ropeNPairs, bs.cs, bs.ropeTab, bs.ropeConst, bs.rpos)
			if p.SWAPeriod > 0 && bs.csSWA != nil {
				lc.la(bs.ropeTable, rrows*bs.ropeNPairs, bs.csSWA, bs.ropeTabSWA, bs.ropeConst, bs.rpos)
			}
		}
		for li := lo; li < hi; li++ {
			l := g.layers[li]
			lc.persistBlock(persist && persistable(l, bs))
			why := progRanProgram
			switch {
			case l.prog == nil:
				why = progNotLowered
			case g.NoProgram:
				why = progNoProgram
			case l.prog.experts && R > 1 && (bs.mg == nil || bs.mg.rows != R || l.prog.streamed):
				why = progRowsExperts
			case len(skip) != 0 || g.blockFaulted():
				why = progFault
			case l.prog.attends && bs.pkv == nil && !p.NonCausal:
				why = progNotPaged
			case l.prog.ropes && l.noRope:
				why = progRotary
			case g.TableSplit:
				why = progTableSplit
			}
			if R > 1 {
				l.progRunRows = why
			} else {
				l.progRun = why
			}
			if why != progRanProgram {
				if err == nil {
					err = fmt.Errorf("tier: block %d: the program cannot run this step: %v", li, why)
				}
				return
			}
			runProg(l, li, l.prog)
			l.ran.note(formBlock, uint8(blockProgram))
		}
		switch {
		case head == nil || err != nil:
		case rag == nil && !allRows && !fold && R == 1 && bs.headProg != nil:
			g.runHead(lc, bs.headProg, decodeHeadBufs(bs))
		case rag != nil && !allRows && ragHead == 1 && bs.ragHeadProg != nil:
			g.runHead(lc, bs.ragHeadProg, rowsHeadBufs(bs, g.bs))
		case fold && hb.spanProg != nil && hb.headProg != nil:
			hbufs := decodeHeadBufs(hb)
			hbufs.src, hbufs.off = bs.x, hb.spanOff
			g.runHead(lc, hb.spanProg, hbufs)
			g.runHead(lc, hb.headProg, hbufs)
		default:
			err = fmt.Errorf("tier: the head has no composed form for this step")
		}
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
			a.paging {
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

	if a.pe != nil {
		hr := g.bs.head
		g.subBytes.embIDs = embIDs(g.subBytes.embIDs, a.pe.ids, R, hr.nrows)
		w(g.bs.embIds, g.subBytes.embIDs)
		if err == nil {
			err = lc.launch(g.bs.emb.k, a.gatherN/g.bs.emb.group, g.bs.emb.group, hr.qs, hr.d, hr.sc, g.bs.embIds, bs.x)
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
	if pk := bs.pkv; pk != nil && len(pk.desc) > 0 {
		w(pk.pRow, u32view(pk.desc))
		if len(pk.descW) > 0 {
			w(pk.pRowW, u32view(pk.descW))
		}
		if pp := &pk.pre; pp.on {
			for i := range pp.desc {
				if len(pp.desc[i]) > 0 {
					w(pp.pDesc[i], u32view(pp.desc[i]))
				}
			}
		}
		if rr := &pk.rag; rr.n > 0 {
			w(rr.pMap, u32view(rr.slots))
			for i := range rr.base {
				if len(rr.base[i]) > 0 {
					w(rr.pBase[i], u32view(rr.base[i]))
				}
			}
		}
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

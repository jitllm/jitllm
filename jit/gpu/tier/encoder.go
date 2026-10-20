package tier

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
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
	p, rag, pe, key := a.p, a.rag, a.pe, a.key
	R, nrow, kvDim, qdim, kst, nCap, gatherN, rrows := a.R, a.nrow, a.kvDim, a.qdim, a.kst, a.nCap, a.gatherN, a.rrows
	sm, smG, smW := a.sm, a.smG, a.smW
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
	// scale plane (see kernels.MatVec), so it keeps its precision.
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
	// split is 32 rows by Split segments (kernels.GroupSplitWidth), every
	// other matvec 128. The thread count is the same either way.
	lamv := func(m mv, k backend.Kernel, threads int, bufs ...backend.Buf) {
		w := 128
		if m.group {
			w = kernels.GroupSplitWidth(m.split)
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
	// mvrunAct is mvrun with the activation named rather than assumed, for a
	// chained matvec such as KDA's low-rank gates, whose second half reads the
	// first half's output. It takes the float source f too, because a float
	// weight reads the float activation (dOf) rather than a/ax.
	mvrunAct := func(m mv, r *resident, dst, a, ax, f backend.Buf) {
		dOfAct := func(r *resident) backend.Buf {
			if !kernels.IsFloat(r.t) {
				return r.d
			}
			return f
		}
		if R > 1 {
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
			runBatched(bm, r, dst, a, ax, dOfAct(r), f)
			return
		}
		if g.TableSplit && m.alt != nil {
			m = *m.alt
		}
		out := dst
		if m.split > 1 && !m.group {
			out = g.partBuf
		}
		n := m.rows * m.split
		if m.rowt > 1 {
			n /= m.rowt
		}
		if m.bias != nil {
			lamv(m, m.kern, n, r.qs, dOfAct(r), r.sc, a, ax, out, m.bias)
		} else {
			lamv(m, m.kern, n, r.qs, dOfAct(r), r.sc, a, ax, out)
		}
		if m.split > 1 && !m.group {
			lc.la(m.red, m.rows, g.partBuf, dst)
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
	// mvres is mvrun with the residual in the bias slot: dst = resid + W*a,
	// replacing the Add that followed. Only a one-row decode with the
	// variant built; false means the caller runs mvrun and the Add.
	mvres := func(m mv, r *resident, dst, resid backend.Buf) bool {
		if rag != nil {
			gm, ok := g.groupKern(m, len(rag.pos), R, groupRes, 0)
			if ok {
				lamv(gm, gm.kern, gm.rows*gm.split, r.qs, dOf(r), r.sc, bs.a, bs.ax, dst, resid)
				g.RagResFused++
			}
			return ok
		}
		if m.res == nil || R > 1 || (g.TableSplit && m.alt != nil) {
			return false
		}
		out := dst
		if m.split > 1 && !m.group {
			out = g.partBuf
		}
		n := m.rows * m.split
		if m.rowt > 1 {
			n /= m.rowt
		}
		lamv(m, m.res, n, r.qs, dOf(r), r.sc, bs.a, bs.ax, out, resid)
		if m.split > 1 && !m.group {
			lc.la(m.red, m.rows, g.partBuf, dst)
		}
		g.ResidualFused++
		return true
	}
	// mvgate launches the up projection's Gate variant, writing
	// act(gate)*up into dst; false where there is none, and the caller then
	// runs the matvec and the ActMul.
	mvgate := func(m mv, r *resident, dst, gate backend.Buf) bool {
		if rag != nil {
			gm, ok := g.groupKern(m, len(rag.pos), R, groupGate, p.Act)
			if ok {
				lamv(gm, gm.kern, gm.rows*gm.split, r.qs, dOf(r), r.sc, bs.a, bs.ax, dst, gate)
				g.RagGateFused++
			}
			return ok
		}
		if m.gated == nil || R > 1 || (g.TableSplit && m.alt != nil) {
			return false
		}
		lamv(m, m.gated, m.rows*m.split, r.qs, dOf(r), r.sc, bs.a, bs.ax, dst, gate)
		return true
	}
	// mvIn launches one of m's operand-prologue kernels (fuseActIn): the
	// float row src, and the norm weight nw where a norm leads, in the
	// operand's place; extra is the residual (res) or the gate (gated).
	mvIn := func(m mv, k backend.Kernel, r *resident, dst, src, nw, extra backend.Buf) {
		out := dst
		if m.split > 1 && !m.group {
			out = g.partBuf
		}
		n := m.rows * m.split
		if m.rowt > 1 && k != m.gatedIn {
			n /= m.rowt
		}
		var arr [8]backend.Buf
		bufs := append(arr[:0], r.qs, r.d, r.sc, src)
		if nw != nil {
			bufs = append(bufs, nw)
		}
		bufs = append(bufs, out)
		if extra != nil {
			bufs = append(bufs, extra)
		}
		lamv(m, k, n, bufs...)
		if m.split > 1 && !m.group {
			lc.la(m.red, m.rows, g.partBuf, dst)
		}
		g.ActInLaunches++
	}
	// mvid launches an indexed matvec: the same buffers plus the selection,
	// and rows*slots*split threads rather than rows. The kernel takes pSel
	// after pOut, so the argument order here is the parameter order there.
	// The reduction is over Rows*Slots rows, which is the launch's own
	// output geometry and not the bank's.
	// An in-group split (retuneIndexed) writes the final rows itself, at its
	// own workgroup width.
	mvid := func(m mv, r *resident, dst, sel backend.Buf) {
		out := dst
		if m.split > 1 && !m.group {
			out = g.partBuf
		}
		lamv(m, m.kern, m.rows*m.slots*m.split, r.qs, dOf(r), r.sc, bs.a, bs.ax, out, sel)
		if m.split > 1 && !m.group {
			lc.la(m.red, m.rows*m.slots, g.partBuf, dst)
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
	skipAttn, skipFFN := skip["attn"], skip["ffn"]
	// laRows launches one RMSNormGroup-thread group per row.
	laRows := func(k backend.Kernel, bufs ...backend.Buf) {
		actMemoK, actMemoSrc = nil, nil
		if err == nil {
			err = lc.launch(k, R, kernels.RMSNormGroup, bufs...)
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
	// resScaled keeps the projections from fusing the residual; only
	// ScaleFaultStepResidual lets a ragged step's fuse it, unscaled.
	resScaled := scaled && !(rag != nil && g.kb.scaleFault == ScaleFaultStepResidual)
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
	// mixFFN runs block li's routed mixture (and its shared expert) over the
	// normed row in bs.h and its quantize, and returns the buffer holding the
	// FFN's output: the block encoder and the device program's mixture op
	// (devprog.go) both call it.
	mixFFN := func(l *layer, li int) backend.Buf {
		ffnOut := bs.mvOut
		// Gemma 4's block runs its dense MLP first, on the ffn_norm row
		// in bs.h, and then the mixture on two norms of its own of x2:
		// the router's into bs.hr, the experts' into bs.h. See
		// emitDenseMLP.
		rin := bs.h
		if p.DenseMoE {
			g.emitDenseMLP(lc, bs, l, R, bs.x2, mvrun, &err)
			rin = bs.hr
		}
		// The whole mixture with nothing crossing the bus: the router reads
		// the normed f32 vector (the block's one unquantized matrix) and its
		// output feeds ExpertRank, whose output indexes the expert matvecs.
		// Kimi-K3's experts run at the latent (k3.go): read from its
		// projection of the normed row, summed in x.latAcc and taken
		// out of the latent below.
		ein, eout := bs.h, bs.mvOut
		latent := l.k3 != nil && l.k3.latent != 0
		if latent {
			ein, eout = bs.k3.lat, bs.k3.latAcc
		}
		hybrid, split := false, false
		// late is host expert work left to run after the card's next
		// launches (the hits, the shared expert) are queued, so the
		// card computes while the host does: the misses' sum goes to
		// lateDst, and lateAdd adds it to the card's hits into eout.
		late, lateAdd := false, false
		var lateDst backend.Buf
		finishLate := func() {
			if !late {
				return
			}
			late = false
			if err == nil {
				err = l.stream.hostLate(sid, g, s, lateDst, split)
			}
			if lateAdd {
				lc.la(bs.addE, p.ExpWidth(), bs.esplit, bs.hsum, eout)
			}
			if err == nil && l.stream.cached() {
				err = l.stream.admit(g, s, l, p.NExpert, g.splitAdmit())
			}
		}
		if R > 1 {
			if latent {
				g.k3LatentIn(lc, bs, l, R, mvrun)
			}
			g.emitGroupedMoE(sid, s, lc, bs, l, R, nrow, rin, ein, eout, &err)
			if latent {
				g.k3LatentOut(lc, bs, l, R, mvrun)
			}
		} else {
			// route ranks one block's router over rin into bs.rsel, bs.rtop and,
			// with weights, bs.rw. It is a closure so the streamed path's
			// cross-layer probe can run the NEXT block's router over this block's
			// row through exactly the launches that block will run itself.
			route := func(l *layer, weights bool) bool {
				// A prediction writes its selection apart (rselPred), so the
				// block's own can be read after it.
				selDst := bs.rsel
				if !weights && g.splitPredict() > 0 {
					selDst = bs.rselPred
				}
				lc.la(l.mvRouter, kernels.RouterThreads(bs.nExpert), l.router, rin, bs.rlogits)
				logits := bs.rlogits
				if l.routerB != nil {
					// gpt-oss biases the logits before the rank, so the
					// bias decides which experts are chosen.
					lc.la(bs.rbias, bs.nExpert, bs.rlogits, l.routerB, bs.rlogitsB)
					logits = bs.rlogitsB
				}
				// The V3 family's two extra passes exist because the IR has no nested
				// loops: ranking needs each group's mask, which needs the group's rank
				// among all groups, which needs a sum over members. Materialising the
				// group scores and then the masked plane keeps each pass flat.
				plane := logits
				if bs.gscore != nil {
					if l.expSelB != nil {
						lc.la(bs.gscore, p.ExpertGroups, logits, bs.rgs, l.expSelB)
						lc.la(bs.gmask, bs.nExpert, logits, bs.rgs, bs.rbm, l.expSelB)
					} else {
						lc.la(bs.gscore, p.ExpertGroups, logits, bs.rgs)
						lc.la(bs.gmask, bs.nExpert, logits, bs.rgs, bs.rbm)
					}
					plane = bs.rbm
				}
				// The arity follows the route. Vulkan refuses an undeclared buffer count;
				// CUDA ignores a surplus pointer and silently never reads the plane.
				// DeepSeek V4 selects through a plane per row (ds4.go).
				selB := l.expSelB
				if l.ds4 != nil {
					selB = g.ds4RouteBias(lc, bs, l, 1)
				}
				fusedRoute := bs.route != nil && bs.gscore == nil && selB == nil
				switch {
				case fusedRoute:
					// Rank and weights in one workgroup (kernels.ExpertRoute).
					// The probe's weights go to a scratch of their own:
					// this block's are already in bs.rw and still to
					// be read.
					rwDst := bs.rw
					if !weights {
						rwDst = bs.rwProbe
					}
					if err == nil {
						err = lc.launch(bs.route, 1, bs.routeWidth,
							logits, selDst, bs.rtop, rwDst)
					}
				case bs.gscore != nil:
					lc.la(bs.rank, bs.nExpert, logits, selDst, bs.rtop, plane)
				case selB != nil:
					lc.la(bs.rank, bs.nExpert, logits, selDst, bs.rtop, selB)
				default:
					lc.la(bs.rank, bs.nExpert, logits, selDst, bs.rtop)
				}
				if err == nil && !fusedRoute && weights {
					// One thread: k is 2 to 8 and this is three short loops.
					// The un-renormalised softmax kernel reads the full
					// logit array; the renormalised one does not declare
					// it, and the sigmoid one reads neither -- sigma()
					// already ran inside the rank.
					if p.NoExpertNorm && !p.ExpertSigmoid {
						err = lc.launch(bs.weights, 1, 1, bs.rtop, bs.rw, logits)
					} else {
						err = lc.launch(bs.weights, 1, 1, bs.rtop, bs.rw)
					}
				}
				return false
			}
			route(l, true)
			// The next hybrid block's router over this block's row, queued
			// before the wait so its selection comes home with this one's:
			// a prediction, which only moves sheets (predictAdmit).
			var predNx *layer
			if g.splitPredict() > 0 && l.stream != nil && l.stream.hybrid && li+1 < hi {
				if nx := g.layers[li+1]; nx != nil && nx.stream != nil && nx.stream.hybrid &&
					nx.stream.cached() && nx.router != nil {
					route(nx, false)
					predNx = nx
				}
			}
			// The suspension, for a streamed bank: the selection comes home, k
			// sheets of gate/up/down are uploaded at their compact offsets, and the
			// submission resumes indexing 0..k-1, because the bank it reads is the
			// selection. The Sync is required: Launch does not wait, and the uploads
			// that follow must precede the launches that read them.
			sel := bs.rsel
			if l.stream != nil {
				sel = bs.ident
				if l.stream.cached() {
					// fill writes each selected expert's cache slot over
					// the selection, in selection order.
					sel = bs.rsel
				}
				if g.StreamFixedSel {
					// The probe spreads per block on purpose: a fixed 0..k-1 would keep
					// one small working set resident and measure that along with the
					// round trip. A block-dependent stride removes only the readback.
					for i := range l.stream.sel {
						l.stream.sel[i] = uint32((li*31 + i*7) % p.NExpert)
					}
				} else if err == nil {
					// Admissions queued by the blocks before go out
					// while the card runs this one up to its router.
					if len(g.adm) > 0 {
						err = g.flushAdmits(s)
					}
					tw := time.Now()
					if err == nil {
						err = s.Sync()
					}
					g.TStreamWait += time.Since(tw)
				}
				if err == nil && !g.StreamFixedSel {
					err = s.Read(bs.rsel, u32b(l.stream.sel))
				}
				// Sheets sent while the card ran up to this router
				// (flushAdmits) have landed or are about to: in use from
				// this block on, a predicted one for this very block.
				if err == nil && len(g.admPub) > 0 {
					err = g.publishAdmits(s)
				}
				if err == nil && predNx != nil {
					predNx.stream.pred = growU32(predNx.stream.pred, len(predNx.stream.sel))
					if err = s.Read(bs.rselPred, u32b(predNx.stream.pred)); err == nil {
						predNx.stream.havePred = true
						predNx.stream.predictAdmit(g, predNx, p.NExpert, g.splitPredict())
					}
				}
				if err == nil && !g.StreamFixedSel {
					l.stream.score(g, li)
				}
				hybrid = l.stream.hybrid && !g.StreamFixedSel
				// Every selection the card routes is the session's
				// count, a hybrid block's or a streamed one's.
				if !g.StreamFixedSel {
					l.stream.count(g.hostFns[sid][li].uses)
				}
				// The split: a hybrid block with a cache runs the experts the
				// cache holds on the card and the rest on the host, at once.
				split = hybrid && l.stream.cached()
				if err == nil && split {
					if l.down != l.stream.bank {
						l.stream.forget()
						l.stream.bank = l.down
					}
					split = l.stream.split(g) > 0
				}
				if err == nil && split {
					if latent {
						g.k3LatentIn(lc, bs, l, 1, mvrun)
						err = s.Sync()
					}
					if err == nil {
						err = l.stream.splitIn(s, ein, bs.rw, p.ExpWidth(), bs.nUsed)
					}
					if err == nil {
						err = s.WriteAt(bs.rwHit, 0, f32b(l.stream.dw))
					}
				}
				if err == nil && hybrid && !split {
					// Hybrid: the experts' input and the routing weights come
					// home, the host runs the selected experts, and their sum
					// goes back where the combine would have written it.
					if latent {
						g.k3LatentIn(lc, bs, l, 1, mvrun)
						err = l.stream.runHost(sid, g, s, ein, bs.rw, eout, p.ExpWidth(), bs.nUsed)
						if err == nil && l.stream.cached() {
							// No hit: the cache still learns from the selection.
							err = l.stream.admit(g, s, l, p.NExpert, g.splitAdmit())
						}
					} else {
						// The input and weights come home now; the host runs
						// the experts once the shared expert is queued.
						err = l.stream.hybridIn(s, ein, bs.rw, p.ExpWidth(), bs.nUsed)
						late, lateDst = true, eout
					}
				}
				// A hybrid block runs the probe too when asked (its recall
				// is the measure a predicted upload is priced on), and
				// never the host-page prefetch, which its warm pages need
				// not.
				if err == nil && (g.StreamProbe || !hybrid && g.StreamPrefetch) && li+1 < hi {
					// The cross-layer probe: the next block's router over this
					// block's normed row, ranked by that block's own launches.
					// It clobbers rsel and rtop, which nothing below reads on a
					// streamed block (it indexes bs.ident and the weights are
					// already in bs.rw).
					if nx := g.layers[li+1]; nx != nil && nx.stream != nil && nx.router != nil {
						if route(nx, false) {
							g.ProbeFused++
						} else if err = s.Sync(); err == nil {
							nx.stream.pred = growU32(nx.stream.pred, len(nx.stream.sel))
							err = s.Read(bs.rsel, u32b(nx.stream.pred))
							nx.stream.havePred = err == nil
							if err == nil && g.StreamPrefetch && !hybrid {
								nx.stream.prefetch(g)
							}
						}
					}
				}
				if err == nil && !hybrid {
					tf := time.Now()
					if err = l.stream.fill(g, s, l, p.NExpert); err != nil && l.stream.cached() {
						l.stream.forget()
					}
					g.fillDone(time.Since(tf))
				}
				if err == nil && l.stream.cached() {
					err = s.WriteAt(bs.rsel, 0, u32b(l.stream.slotSel))
				}
				if err != nil {
					return ffnOut
				}
				if hybrid {
					g.HybridRuns++
				} else {
					g.StreamFills++
				}
			}
			if hybrid && !split && latent {
				// The routed sum is already in eout.
				g.k3LatentOut(lc, bs, l, 1, mvrun)
			}
			if !hybrid || split {
				// gate and up read the one normed vector, so they are
				// indexed on the weight side only. bs.a/bs.ax still hold
				// its quantization from the launch above -- or, at
				// Kimi-K3's latent, its projection's.
				if latent && !split {
					g.k3LatentIn(lc, bs, l, 1, mvrun)
				}
				ffnW := bs.nUsed * bs.nFFNExp
				// mvidE launches an expert matvec with fused epilogue
				// operands between its output and the selection -- the
				// kernel's parameter order (pOut, [pBias], [pGate], pSel).
				mvidE := func(k backend.Kernel, m mv, r *resident, dst backend.Buf, extra ...backend.Buf) {
					// Joined in a stack array: appending to a literal of
					// its own length grew it on the heap every launch.
					var all [12]backend.Buf
					args := append(all[:0], r.qs, dOf(r), r.sc, bs.a, bs.ax, dst)
					args = append(append(args, extra...), sel)
					lamv(m, k, m.rows*m.slots*m.split, args...)
					g.MoEFused++
				}
				rw := bs.rw
				if split {
					rw = bs.rwHit
				}
				if p.DenseMoE {
					// Each routed weight times its expert's own factor.
					lc.la(bs.ewScale, bs.nUsed, bs.rw, bs.rsel, l.expScale, bs.rwS)
					rw = bs.rwS
				}
				if ungatedFFN(p) {
					// Ungated experts (Nemotron 3): up, then the activation
					// alone. No bias exists on any such architecture.
					mvid(l.mvu, l.up, bs.eu, sel)
					lc.la(bs.actMulE, ffnW, bs.eu, bs.eact)
				} else if l.moeUp != nil {
					// act(gate)*up in the up projection's own epilogue, the
					// biases in each matvec's (fuseMoE).
					if l.moeGate != nil {
						mvidE(l.moeGate, l.mvg, l.gate, bs.eg, l.expGateB)
					} else {
						mvid(l.mvg, l.gate, bs.eg, sel)
					}
					if l.expUpB != nil {
						mvidE(l.moeUp, l.mvu, l.up, bs.eact, l.expUpB, bs.eg)
					} else {
						mvidE(l.moeUp, l.mvu, l.up, bs.eact, bs.eg)
					}
				} else {
					mvid(l.mvg, l.gate, bs.eg, sel)
					mvid(l.mvu, l.up, bs.eu, sel)
					eg, eu := bs.eg, bs.eu
					// The experts' biases are indexed by the true ids in rsel,
					// even where the weights are a compacted bank read 0..k-1.
					if l.expGateB != nil {
						lc.la(bs.ebiasFF, ffnW, bs.eg, bs.rsel, l.expGateB, bs.egB)
						eg = bs.egB
					}
					if l.expUpB != nil {
						lc.la(bs.ebiasFF, ffnW, bs.eu, bs.rsel, l.expUpB, bs.euB)
						eu = bs.euB
					}
					if bs.actMulW != nil {
						// Llama 4: the weight scales gate and up, and the
						// downs are summed at one.
						lc.la(bs.actMulW, ffnW, eg, eu, bs.rw, bs.eact)
						rw = bs.wOnes
					} else {
						lc.la(bs.actMulE, ffnW, eg, eu, bs.eact)
					}
				}
				// One Quantize over all the slots, writing a flat [scales][sums] pAX,
				// which is the layout down's SlotAct per-slot bases expect.
				lc.la(bs.quantX, kernels.QuantizeThreads(bs.nUsed*pad32(p.NFFNExp)/32), bs.eact, bs.a, bs.ax)
				edown := bs.edown
				if l.moeDown != nil {
					mvidE(l.moeDown, l.mvd, l.down, bs.edown, l.expDownB)
				} else {
					mvid(l.mvd, l.down, bs.edown, sel)
					if l.expDownB != nil {
						lc.la(bs.ebiasD, bs.nUsed*p.NEmbd, bs.edown, bs.rsel, l.expDownB, bs.edownB)
						edown = bs.edownB
					}
				}
				if !split {
					lc.la(bs.combine, p.ExpWidth(), edown, rw, eout)
				} else {
					// The card's hits sum into esplit while the host runs
					// the misses; the two sums are added in that order.
					lc.la(bs.combine, p.ExpWidth(), edown, rw, bs.esplit)
					late, lateAdd, lateDst = true, true, bs.hsum
					if latent {
						// The latent's out projection reads the sum: no
						// shared expert runs between.
						finishLate()
					}
				}
				if latent {
					g.k3LatentOut(lc, bs, l, 1, mvrun)
				}
			}
		}
		shared := bs.shAdd != nil && !p.DenseMoE &&
			(l.shGate != nil || ungatedFFN(p)) && l.shUp != nil && l.shDown != nil
		if !shared {
			finishLate()
		}
		if p.DenseMoE {
			// The experts' post-norm, then the dense MLP's normed output
			// (bs.shsum) added: the two branches' sum, which the post-FFN
			// norm below takes like any block's FFN output.
			lc.la(bs.normPart, R*bs.parts, bs.mvOut, bs.part)
			lc.la(bs.normApply, R*p.NEmbd, bs.mvOut, l.nPost2, bs.part, bs.h)
			lc.la(bs.addE, R*p.NEmbd, bs.shsum, bs.h, bs.mvOut)
		}
		// The always-on expert is added after the routed sum, not into it: the
		// routed weights may be renormalised by their own sum (see engine/model/moe.go).
		if shared {
			if l.shRouter != nil {
				lc.la(bs.shRouterK, kernels.SharedRouterThreads(R), l.shRouter, bs.h, bs.shlogit)
			}
			// Re-quantize h: quantX overwrote bs.a/bs.ax with the expert
			// activations.
			lc.la(bs.quantE, kernels.QuantizeThreads(R*ePad(p)/32), bs.h, bs.a, bs.ax)
			switch {
			case ungatedFFN(p):
				// Ungated (Nemotron 3): up and the activation alone.
				mvrun(l.mvsu, l.shUp, bs.shu)
				lc.la(bs.shActMul, R*bs.nShExp, bs.shu, bs.shact)
			case l.shGU != nil && R == 1 && !g.TableSplit:
				// gate and up in one launch (fuseSegs).
				if err == nil {
					err = lc.launch(l.shGU.kern, l.shGU.blocks, l.shGU.w, bs.a, bs.ax,
						l.shGate.qs, l.shGate.d, l.shGate.sc, bs.shg,
						l.shUp.qs, l.shUp.d, l.shUp.sc, bs.shu)
				}
				g.SegFused++
			default:
				mvrun(l.mvsg, l.shGate, bs.shg)
				mvrun(l.mvsu, l.shUp, bs.shu)
			}
			if !ungatedFFN(p) {
				lc.la(bs.shActMul, R*bs.nShExp, bs.shg, bs.shu, bs.shact)
			}
			lc.la(bs.shQuant, kernels.QuantizeThreads(R*bs.nShExp/32), bs.shact, bs.a, bs.ax)
			mvrun(l.mvsd, l.shDown, bs.shout)
			// The host's experts run now, the shared expert queued
			// ahead of them on the card.
			finishLate()
			// An ungated shared expert is a plain add: no logit makes sigma()
			// exactly one, so the multiply goes rather than being neutralised.
			if l.shRouter != nil {
				lc.la(bs.shAdd, R*p.NEmbd, bs.mvOut, bs.shout, bs.shlogit, bs.shsum)
			} else {
				lc.la(bs.addE, R*p.NEmbd, bs.mvOut, bs.shout, bs.shsum)
			}
			ffnOut = bs.shsum
		}
		return ffnOut
	}
	// perLayerEmbed adds block l's per-layer embedding to the row src, into
	// bs.x: x = src + post_norm(W_out . (act(W_gate . src) * pl[li])), the
	// sum through x2 and moved home. The encoder and the device program's
	// op (devprog.go) both call it.
	perLayerEmbed := func(l *layer, src backend.Buf) {
		lc.la(bs.quantE, kernels.QuantizeThreads(R*ePad(p)/32), src, bs.a, bs.ax)
		mvrun(l.mvpg, l.pleGate, bs.pleG)
		lc.la(bs.pleSlice, R*p.PLEDim, bs.pleIn, l.pleOff, bs.pleS)
		lc.la(bs.pleAct, R*p.PLEDim, bs.pleG, bs.pleS, bs.pleA)
		lc.la(bs.pleQuant, kernels.QuantizeThreads(R*pad32(p.PLEDim)/32), bs.pleA, bs.a, bs.ax)
		mvrun(l.mvpp, l.pleProj, bs.pleO)
		lc.la(bs.normPart, R*bs.parts, bs.pleO, bs.part)
		lc.la(bs.normApply, R*p.NEmbd, bs.pleO, l.nPLE, bs.part, bs.h)
		addRes(src, bs.h, bs.x2)
		lc.la(bs.scaleX, R*p.NEmbd, bs.x2, bs.one, bs.x)
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
		// A non-causal call in two phases (a row-chunked segment): the first
		// writes its chunk's keys and attends nothing, the second attends
		// every chunk's and writes none.
		kOff := bs.koff
		if kst > 0 {
			kOff = bs.kpos
		}
		ncWrite := !(p.NonCausal && g.ncPhase == 2)
		ncAttend := !(p.NonCausal && g.ncPhase == 1)
		qdim := p.NHead * p.HeadDim
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
		genIdx := 0 // the op a former is running, which it may read ahead of
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
		lg := func(k backend.Kernel, threads, group int, bufs ...backend.Buf) {
			if err == nil {
				err = lc.launch(k, (threads+group-1)/group, group, bufs...)
			}
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
		// grouped runs an absorb bank over the heads' rows, head-major, its
		// int8 operand just quantized (emitMLARows' grouped).
		latentGrouped := func(m mv, r *resident, src, dst backend.Buf) {
			c, ok := g.groupedMV(m, ml.nh, ml.nh*R)
			if !ok {
				if err == nil {
					err = errNoBatch
				}
				return
			}
			d := r.d
			if kernels.IsFloat(r.t) {
				d = src
			}
			lc.la(c, m.rows*(ml.nh*R/groupTok), r.qs, d, r.sc, bs.a, bs.ax, dst, ml.tsel)
		}
		var gr groupRun
		skipNext := false
		at := func(v devVal) backend.Buf { return buf[prog.slots[v]] }
		// The matvec former's launcher: the op's weight's kernel, emitted from
		// its MatVecShape (format, width, rows, split, epilogue) and cached
		// by it, at the step's rows -- the decode kernel, a batched or ragged
		// twin, the split with its reduction, the residual or gate epilogue.
		// form runs a former's launch for op: called directly, never stored,
		// so it costs a warm step nothing.
		form := func(op *devOp) {
			o := *op
			var m mv
			var r *resident
			if o.W >= 0 && o.W < len(l.ms) {
				m, r = *l.ms[o.W], *l.rs[o.W]
			}
			switch o.Kind {
			case dkMatVec:
				if a, ax, ok := q8Pair(bs, prog.slots[o.In[0]]); ok {
					// A matvec reading an int8 operand of its own (a low-rank
					// pair's, a stream mix's), its float source beside it.
					mvrunAct(m, r, at(o.Out[0]), a, ax, at(o.In[1]))
					return
				}
				if prog.f16 {
					// The GEMM converts this row, or finds it written.
					fsrc = at(o.In[0])
				}
				mvrun(m, r, at(o.Out[0]))
			case dkMatVecRes:
				if !mvres(m, r, at(o.Out[0]), at(o.In[1])) && err == nil {
					err = fmt.Errorf("tier: block %d: matvec-residual has no variant for %s", li, o.Role)
				}
			case dkMatVecGate:
				if !mvgate(m, r, at(o.Out[0]), at(o.In[1])) && err == nil {
					err = fmt.Errorf("tier: block %d: matvec-gate has no variant for %s", li, o.Role)
				}
			case dkExpertMatVec:
				if rowsMLA && prog.slots[o.Out[0]] == slotMAbsRaw {
					latentGrouped(*l.ms[o.W], *l.rs[o.W], at(o.In[2]), at(o.Out[0]))
					break
				}
				if rowsMLA && prog.slots[o.Out[0]] == slotMOut {
					// Head-major out of the grouped un-absorb, token-major after.
					latentGrouped(*l.ms[o.W], *l.rs[o.W], at(o.In[2]), ml.outHM)
					lc.la(ml.gatherOut, R*ml.nh*ml.hdv, ml.outHM, ml.permTM, at(o.Out[0]))
					break
				}
				if rowsMoE {
					// A bias the grouped form folds into its epilogue (the
					// tensor-core one) takes the bias op's place.
					var bias backend.Buf
					var next *devOp
					if j := genIdx + 1; j < len(prog.ops) && prog.ops[j].Kind == dkExpertBias && prog.ops[j].In[0] == o.Out[0] {
						next = &prog.ops[j]
						bias = next.V
					}
					gr.ein = at(o.In[2])
					if gr.grouped(groupLaunch{lc, &err}, &err, l.ms[o.W], *l.rs[o.W], bias, at(o.Out[0])) && next != nil {
						buf[prog.slots[next.Out[0]]] = at(o.Out[0])
						skipNext = true
					}
					break
				}
				mvid(*l.ms[o.W], *l.rs[o.W], at(o.Out[0]), at(o.In[1]))
			case dkExpertMatVecE:
				// The bank's fused kernel: (pOut, [pBias], [pGate], pSel).
				m, r := *l.ms[o.W], *l.rs[o.W]
				var all [12]backend.Buf
				args := append(all[:0], r.qs, dOf(r), r.sc, bs.a, bs.ax, at(o.Out[0]))
				if o.V != nil {
					args = append(args, o.V)
				}
				if len(o.In) > 2 {
					args = append(args, at(o.In[2]))
				}
				lamv(m, o.K, m.rows*m.slots*m.split, append(args, at(o.In[1]))...)
				g.MoEFused++
			case dkAttend:
				if pk == nil {
					if !ncAttend {
						break
					}
					var fv *flashVariant
					var flash backend.Kernel
					if len(bs.flashV) > 0 {
						fv = bs.flashFor(nCap)
						flash = fv.k
						if p.Window(li) > 0 {
							flash = fv.kW
						}
					}
					if err == nil {
						err = g.contigAttend(lc, bs, l, li, R, kvp, at(o.In[0]), nCap, sm, smG, smW, nil, fv, flash,
							g.flashOn() && flash != nil && !l.windowed)
					}
					break
				}
				if pk.prefill {
					// A prefill chunk: one sequence's rows in order, a K or V
					// load serving a whole query tile.
					win := p.Window(li) > 0
					if err == nil {
						err = pk.pf.prefillAttn(s, lc, at(o.In[0]), pl.k, pl.v, bs.n, at(o.Out[0]), pl.tab, win, sk)
						g.PagedPrefillLaunches++
						if pk.pf.flash != nil {
							g.FlashPrefills++
						}
						g.PagedPrefillPasses += pk.pf.passes(win)
						if pk.pf.on70() {
							g.PagedPrefill70++
						}
						if pk.pf.accOnMMA() {
							g.PagedAccMMA++
						}
						if pk.pf.on80() {
							g.FlashPrefill80++
						}
					}
					return
				}
				// A decode row or a step's rows: each a descriptor, a window
				// its keyStart, through the decode plan. A ragged step with a
				// run table empties the table's rows from the decode plan and
				// attends them with the table's prefill (raggedprefill.go).
				win := p.Window(li) > 0
				desc := pk.pRow
				if win {
					desc = pk.pRowW
				}
				ragOn := rag != nil && pk.ragRows > 0
				dec := desc
				if ragOn {
					dec = pk.pRowDec
					if win {
						dec = pk.pRowDecW
					}
				}
				if err == nil && !(ragOn && pk.ragAlone) {
					err = pk.attend(s, lc, pl, at(o.In[0]), pl.k, pl.v, bs.n, at(o.Out[0]), pl.tab, dec, sk)
					g.PagedLaunches++
					if pk.cur.plan.Path == kernels.PathStaged {
						g.PagedStagedLaunches++
					}
				}
				if ragOn && err == nil {
					if pk.ragStaged {
						err = pk.ragStagedAttend(lc, at(o.In[0]), pl.k, pl.v, bs.n, at(o.Out[0]), pl.tab, desc, sk, win)
					} else {
						err = pk.ragAttend(lc, at(o.In[0]), pl.k, pl.v, bs.n, at(o.Out[0]), pl.tab, desc, sk)
					}
					g.RagPrefills++
				}
			case dkKVWrite:
				if pk != nil && prog.slots[o.In[0]] == slotKAll {
					// The assembled row: the selecting form turns the
					// indexer's head with k's; the pooled form keeps it raw,
					// copied whole first.
					kHeads := p.NKVHead
					if ms != nil {
						kHeads++
					}
					if partialRotary(p) || qs != nil {
						lc.la(pk.copyK, R*kvDim, at(o.In[0]), pl.k, bs.zero, pl.tab, pk.pRow)
					}
					lc.la(pk.ropeK, R*kHeads*p.NRot/2, at(o.In[0]), rcs, bs.zero, pl.k, pl.tab, pk.pRow)
					lc.la(pk.copyV, R*p.NKVHead*p.HeadDim, at(o.In[1]), pl.v, bs.zero, pl.tab, pk.pRow)
					break
				}
				if pk == nil {
					if ncWrite {
						if partialRotary(p) {
							lc.la(bs.copyK, R*kvDim, at(o.In[0]), kvp.kc, kOff)
						}
						lc.la(bs.ropeK, R*p.NKVHead*p.NRot/2, at(o.In[0]), rcs, kOff, kvp.kc)
						lc.la(bs.copyV, R*p.NKVHead*p.HeadDim, at(o.In[1]), kvp.vc, bs.koff)
					}
					break
				}
				if partialRotary(p) {
					lc.la(pk.copyK, R*p.KVRow(), at(o.In[0]), pl.k, bs.zero, pl.tab, pk.pRow)
				}
				lc.la(pk.ropeK, R*p.NKVHead*p.NRot/2, at(o.In[0]), rcs, bs.zero, pl.k, pl.tab, pk.pRow)
				lc.la(pk.copyV, R*p.NKVHead*p.HeadDim, at(o.In[1]), pl.v, bs.zero, pl.tab, pk.pRow)
			case dkKVCopy:
				if pk == nil {
					if ncWrite {
						lc.la(bs.copyK, R*kvDim, at(o.In[0]), kvp.kc, kOff)
						lc.la(bs.copyV, R*p.NKVHead*p.HeadDim, at(o.In[1]), kvp.vc, bs.koff)
					}
					break
				}
				lc.la(pk.copyK, R*p.KVRow(), at(o.In[0]), pl.k, bs.zero, pl.tab, pk.pRow)
				lc.la(pk.copyV, R*p.NKVHead*p.HeadDim, at(o.In[1]), pl.v, bs.zero, pl.tab, pk.pRow)
			case dkLatentAttend:
				// The latent row is the key and its prefix the value: a
				// prefill chunk through the paged prefill kernels, a step's
				// rows through the decode plan.
				g.PagedLaunches++
				if rowsMLA {
					g.MLABatched++
				}
				if !pk.prefill && pk.cur != nil && pk.cur.plan.Path == kernels.PathStaged {
					g.PagedStagedLaunches++
				}
				if err == nil && rowsMLA && pk.prefill {
					err = pk.pf.prefillAttn(s, lc, at(o.In[0]), pl.k, pl.k, bs.n, at(o.Out[0]), pl.tab, false, nil)
					g.PagedPrefillLaunches++
					if pk.pf.on70() {
						g.PagedPrefill70++
					}
					break
				}
				if err == nil {
					err = pk.attend(s, lc, pl, at(o.In[0]), pl.k, pl.k, bs.n, at(o.Out[0]), pl.tab, pk.pRow, nil)
				}
			case dkQKV:
				kern, blocks, w := l.qkv, l.qkvBlocks, l.qkvW
				if rag != nil {
					// A ragged step's segments (groupQKV).
					seg, _ := g.groupQKV(l, len(rag.pos), R)
					kern, blocks, w = seg.kern, seg.blocks, seg.w
					g.RagQKVFused++
				}
				var bufArr [17]backend.Buf
				bufs := append(bufArr[:0], bs.a, bs.ax)
				for i, sg := range [3]struct {
					m mv
					r *resident
				}{{l.mvq, l.wq}, {l.mvk, l.wk}, {l.mvv, l.wv}} {
					bufs = append(bufs, sg.r.qs, sg.r.d, sg.r.sc, at(o.Out[i]))
					if sg.m.bias != nil {
						bufs = append(bufs, sg.m.bias)
					}
				}
				if err == nil {
					err = lc.launch(kern, blocks, w, bufs...)
				}
			case dkQKVGemm:
				qs := o.QS
				fsrc = at(o.In[0])
				if qs.act != actMemoK || fsrc != actMemoSrc {
					lc.la(qs.act, qs.actN, fsrc, g.f16Buf)
					g.ActF16Launches++
				}
				actMemoK, actMemoSrc = qs.act, fsrc
				all := [3]struct {
					m mv
					r *resident
				}{{l.mvq, l.wq}, {l.mvk, l.wk}, {l.mvv, l.wv}}
				var arr [16]backend.Buf
				bufs := append(arr[:0], g.f16Buf)
				for _, i := range qs.segs {
					sg := all[i]
					w := sg.r
					if g.qkvGemmSwapped() && i > 0 && qs.has(3-i) && all[3-i].r.t == w.t && all[3-i].m.rows == sg.m.rows {
						w = all[3-i].r
					}
					bufs = append(bufs, w.qs, w.d, w.sc, at(o.Out[i]))
					if sg.m.bias != nil {
						bufs = append(bufs, sg.m.bias)
					}
				}
				lc.law(qs.kern, qs.threads, qs.width, bufs...)
				g.QKVGemmRuns++
				for i, sg := range all {
					if !qs.has(i) {
						mvrun(sg.m, sg.r, at(o.Out[i]))
					}
				}
			case dkRouterLogits:
				if rowsMoE {
					router := mg.router
					if l.routerF16 {
						router = mg.router16
					}
					lg(router, kernels.RouterThreads(mg.nExp*R), kernels.RouterThreads(1), l.router, at(o.In[0]), at(o.Out[0]))
					break
				}
				lc.la(l.mvRouter, kernels.RouterThreads(bs.nExpert), l.router, at(o.In[0]), at(o.Out[0]))
			case dkResIn:
				m, r := *l.ms[o.W], *l.rs[o.W]
				mvIn(m, m.resIn, r, at(o.Out[0]), at(o.In[0]), nil, at(o.In[1]))
				g.ActInSaved++
			case dkQKVSep:
				for i, sg := range [3]struct {
					m mv
					r *resident
				}{{l.mvq, l.wq}, {l.mvk, l.wk}, {l.mvv, l.wv}} {
					mvIn(sg.m, sg.m.kernIn, sg.r, at(o.Out[i]), at(o.In[0]), o.V, sg.m.bias)
				}
				g.ActInSaved++
			case dkRopeQKV:
				lc.la(pk.ropeQKV, kernels.RoPEQKVPagedThreads(p.NHead, p.NKVHead, p.HeadDim, p.NRot, R, g.KVF16),
					at(o.In[0]), rcs, qOff, at(o.Out[0]), at(o.In[1]), pl.k, at(o.In[2]), pl.v, pl.tab, pk.pRow)
				g.FusedLaunches++
			case dkSlice:
				if rowsMLA {
					switch prog.slots[o.Out[0]] {
					case slotMNope:
						lc.la(ml.sliceNope, R*ml.nh*ml.nope, at(o.In[0]), at(o.Out[0]))
					case slotMPE:
						lc.la(ml.slicePE, R*ml.nh*ml.rot, at(o.In[0]), at(o.Out[0]))
					case slotMKPE:
						lc.la(ml.sliceKPE, R*ml.rot, at(o.In[0]), at(o.Out[0]))
					}
					break
				}
				switch prog.slots[o.Out[0]] {
				case slotMNope:
					lc.la(bs.mlaSliceNope, p.NHead*bs.mlaNopeW, at(o.In[0]), at(o.Out[0]))
				case slotMPE:
					lc.la(bs.mlaSlicePE, p.NHead*bs.mlaRot, at(o.In[0]), at(o.Out[0]))
				case slotMKPE:
					lc.la(bs.mlaSliceKPE, bs.mlaRot, at(o.In[0]), at(o.Out[0]))
				}
			case dkView:
				// One row's prefix is read in place; each of several rows'
				// is copied out.
				if rowsMLA {
					lc.la(ml.sliceLat, R*ml.lat, at(o.In[0]), at(o.Out[0]))
				}
			case dkConcatHeads:
				if rowsMLA {
					lc.la(ml.copyAbs, ml.nh*R*ml.lat, at(o.In[0]), at(o.Out[0]), ml.absOff)
					lc.la(ml.copyPE, R*ml.nh*ml.rot, at(o.In[1]), at(o.Out[0]), ml.peOff)
					break
				}
				lc.la(bs.mlaCopyAbs, p.NHead*bs.mlaLat, at(o.In[0]), at(o.Out[0]), bs.mlaAbsOff)
				lc.la(bs.mlaCopyPE, p.NHead*bs.mlaRot, at(o.In[1]), at(o.Out[0]), bs.mlaPEOff)
			case dkRopeTail:
				if rowsMLA {
					lc.la(ml.ropeQ, R*ml.nh*ml.rot/2, at(o.In[0]), bs.cs, ml.qOff, at(o.Out[0]))
					break
				}
				lc.la(bs.mlaRopeQ, p.NHead*bs.mlaRot/2, at(o.In[0]), bs.cs, bs.zero, at(o.Out[0]))
			case dkTopK:
				// The arity follows the inputs: a masked plane, a selection
				// bias, or the logits alone.
				// The rank kernel's arity is the plan's route's (routeOf): a
				// masked plane where it groups, else a selection bias where it
				// carries one. Vulkan refuses a surplus buffer.
				rt := routeOf(p)
				plane, sb := rt.NGroup > 1 && len(o.In) > 1, !(rt.NGroup > 1) && rt.Bias && o.V != nil
				if rt.RowBias && !(rt.NGroup > 1) && len(o.In) > 1 {
					// A bias per row (dkRowBias's), read where the block's would be.
					o.V, sb = at(o.In[len(o.In)-1]), true
				}
				switch {
				case rowsMoE && plane:
					lg(mg.rank, R*mg.nExp, 128, at(o.In[0]), at(o.Out[0]), at(o.Out[1]), at(o.In[1]))
				case rowsMoE && sb:
					lg(mg.rank, R*mg.nExp, 128, at(o.In[0]), at(o.Out[0]), at(o.Out[1]), o.V)
				case rowsMoE:
					lg(mg.rank, R*mg.nExp, 128, at(o.In[0]), at(o.Out[0]), at(o.Out[1]))
				case plane:
					lc.la(bs.rank, bs.nExpert, at(o.In[0]), at(o.Out[0]), at(o.Out[1]), at(o.In[1]))
				case sb:
					lc.la(bs.rank, bs.nExpert, at(o.In[0]), at(o.Out[0]), at(o.Out[1]), o.V)
				default:
					lc.la(bs.rank, bs.nExpert, at(o.In[0]), at(o.Out[0]), at(o.Out[1]))
				}
			case dkRouteWeights:
				// One thread: k is a handful. The unrenormalised softmax reads
				// every logit; the others do not declare them.
				if rowsMoE {
					if p.NoExpertNorm && !p.ExpertSigmoid {
						lg(mg.weights, R, 64, at(o.In[0]), at(o.Out[0]), at(o.In[1]))
					} else {
						lg(mg.weights, R, 64, at(o.In[0]), at(o.Out[0]))
					}
					break
				}
				if err == nil {
					if p.NoExpertNorm && !p.ExpertSigmoid {
						err = lc.launch(bs.weights, 1, 1, at(o.In[0]), at(o.Out[0]), at(o.In[1]))
					} else {
						err = lc.launch(bs.weights, 1, 1, at(o.In[0]), at(o.Out[0]))
					}
				}
			case dkCombine:
				if prog.slots[o.Out[0]] == slotLatAcc {
					g.K3Latent++ // experts in a latent
				}
				if rowsMoE {
					lg(mg.combine, R*p.ExpWidth(), 128, at(o.In[0]), at(o.In[1]), mg.pos, at(o.Out[0]))
					g.GroupedMoE++
					break
				}
				lc.la(bs.combine, p.ExpWidth(), at(o.In[0]), at(o.In[1]), at(o.Out[0]))
			case dkGroupByExpert:
				if rowsMoE {
					gr = groupRun{g: g, m: mg, l: l, p: p, rows: R, e: mg.nExp, ew: p.ExpWidth()}
					if !gr.sort(groupLaunch{lc, &err}, &err, s) {
						return
					}
				}
			case dkGateLogit:
				lc.la(bs.shRouterK, kernels.SharedRouterThreads(R), o.V, at(o.In[0]), at(o.Out[0]))
			case dkGroupMask:
				// The selection bias rides the group kernels where the route
				// carries one (routeOf), as the rank's does.
				sb := routeOf(p).Bias && o.V != nil
				switch {
				case rowsMoE && sb:
					lg(mg.gscore, R*p.ExpertGroups, 128, at(o.In[0]), mg.rgs, o.V)
					lg(mg.gmask, R*mg.nExp, 128, at(o.In[0]), mg.rgs, at(o.Out[0]), o.V)
				case rowsMoE:
					lg(mg.gscore, R*p.ExpertGroups, 128, at(o.In[0]), mg.rgs)
					lg(mg.gmask, R*mg.nExp, 128, at(o.In[0]), mg.rgs, at(o.Out[0]))
				case sb:
					lc.la(bs.gscore, p.ExpertGroups, at(o.In[0]), bs.rgs, o.V)
					lc.la(bs.gmask, bs.nExpert, at(o.In[0]), bs.rgs, at(o.Out[0]), o.V)
				default:
					lc.la(bs.gscore, p.ExpertGroups, at(o.In[0]), bs.rgs)
					lc.la(bs.gmask, bs.nExpert, at(o.In[0]), bs.rgs, at(o.Out[0]))
				}
			case dkRouterBias:
				if rowsMoE {
					lg(mg.biasR, R*mg.nExp, 128, at(o.In[0]), mg.zsel, o.V, at(o.Out[0]))
					break
				}
				lc.la(bs.rbias, bs.nExpert, at(o.In[0]), o.V, at(o.Out[0]))
			case dkExpertBias:
				if rowsMoE {
					k, n := mg.biasFF, gr.cols*p.NFFNExp
					if prog.slots[o.Out[0]] == slotEDownB {
						k, n = mg.biasD, gr.cols*p.ExpWidth()
					}
					lg(k, n, 128, at(o.In[0]), mg.colE, o.V, at(o.Out[0]))
					break
				}
				k, n := bs.ebiasFF, bs.nUsed*bs.nFFNExp
				if prog.slots[o.Out[0]] == slotEDownB {
					k, n = bs.ebiasD, bs.nUsed*p.NEmbd
				}
				lc.la(k, n, at(o.In[0]), at(o.In[1]), o.V, at(o.Out[0]))
			case dkExpertFetch:
				// The paging runtime, by property of the bank: the selection
				// home, the sheets filled, the matvecs' index chosen.
				st := l.stream
				if err == nil && len(g.adm) > 0 {
					err = g.flushAdmits(s)
				}
				tw := time.Now()
				if err == nil {
					err = s.Sync()
				}
				g.TStreamWait += time.Since(tw)
				if err == nil {
					err = s.Read(at(o.In[0]), u32b(st.sel))
				}
				if err == nil && len(g.admPub) > 0 {
					err = g.publishAdmits(s)
				}
				if err != nil {
					return
				}
				st.score(g, li)
				st.count(g.hostFns[sid][li].uses)
				tf := time.Now()
				if err = st.fill(g, s, l, p.NExpert); err != nil && st.cached() {
					st.forget()
				}
				g.fillDone(time.Since(tf))
				buf[slotSelW] = bs.ident
				if err == nil && st.cached() {
					// The cache's slots over the selection, in its order.
					err = s.WriteAt(at(o.In[0]), 0, u32b(st.slotSel))
					buf[slotSelW] = at(o.In[0])
				}
				if err != nil {
					return
				}
				g.StreamFills++
			case dkConv1d, dkScan:
				// The session's state, double-buffered: this step reads one
				// half and writes the other (the shared-state mode writes the
				// device's scratch and copies it home).
				par, has := g.stepPar(sid, l)
				pool := l.pool
				if !has || pool == nil {
					if err == nil {
						err = fmt.Errorf("tier: block %d has no recurrent state for session %d", li, sid)
					}
					return
				}
				rr := bs.rec
				var rk *ragLinear
				if rag != nil {
					if rk = bs.ragLin[len(rag.pos)]; rk == nil {
						if err == nil {
							err = fmt.Errorf("tier: block %d: no batched recurrent step for %d sequences", li, len(rag.pos))
						}
						return
					}
				}
				if o.Kind == dkConv1d {
					inC, outC := pool.c[par], pool.c[1-par]
					if rk != nil {
						lc.la(rk.conv, len(rag.pos)*rr.Chans, inC, at(o.In[0]), l.ssmConv1d, at(o.Out[0]), bs.dRows)
						lc.la(rk.shift, len(rag.runs)*(rr.Conv-1)*rr.Chans, inC, at(o.In[0]), outC, bs.dRows)
					} else {
						lc.la(bs.convRows, R*rr.Chans, inC, at(o.In[0]), l.ssmConv1d, at(o.Out[0]), bs.dRows)
						lc.la(bs.convShift, (rr.Conv-1)*rr.Chans, inC, at(o.In[0]), outC, bs.dRows)
					}
					break
				}
				inS, outS := pool.s[par], pool.s[1-par]
				if g.recShared {
					inS, outS = pool.s[0], g.recNext
				}
				seqs := 1
				switch {
				case rr.Mamba1 && rk != nil:
					seqs = len(rag.pos)
					lc.la(rk.delta, len(rag.runs)*rk.perRun, inS, at(o.In[0]), at(o.In[1]), at(o.In[2]), at(o.In[3]),
						l.ssmDt, l.ssmA, l.ssmD, at(o.Out[0]), outS, bs.dRows)
					g.LinearFused++
				case rr.Mamba1:
					lc.la(bs.m1Scan, bs.scanThreads, inS, at(o.In[0]), at(o.In[1]), at(o.In[2]), at(o.In[3]),
						l.ssmDt, l.ssmA, l.ssmD, at(o.Out[0]), outS, bs.dRows)
					g.LinearFused++
				case rr.ChanDecay && rk != nil:
					seqs = len(rag.pos)
					lc.la(rk.delta, len(rag.runs)*rk.perRun, inS, at(o.In[0]), at(o.In[1]), at(o.In[2]),
						l.ssmDt, l.ssmA, at(o.Out[0]), outS, bs.dRows)
					g.LinearFused++
				case rr.ChanDecay && bs.deltaFused != nil:
					lc.la(bs.deltaFused, bs.scanThreads, inS, at(o.In[0]), at(o.In[1]), at(o.In[2]),
						l.ssmDt, l.ssmA, at(o.Out[0]), outS, bs.dRows)
					g.LinearFused++
				case rk != nil:
					seqs = len(rag.pos)
					if rr.SSD {
						lc.la(rk.delta, len(rag.runs)*rk.perRun, inS, at(o.In[0]), at(o.In[1]), l.ssmDt, l.ssmA, l.ssmCB, l.ssmD,
							at(o.Out[0]), outS, bs.dRows)
					} else {
						lc.la(rk.delta, len(rag.runs)*rk.perRun, inS, at(o.In[0]), at(o.In[1]), l.ssmDt, l.ssmA,
							at(o.Out[0]), outS, bs.dRows)
					}
					g.LinearFused++
					if rk.lanes == 1 {
						g.LinearRowsScalar++
					}
					if g.rag.sid != nil {
						g.SessionLinear++
					}
				case rr.SSD:
					lc.la(bs.ssdScan, bs.scanThreads, inS, at(o.In[0]), at(o.In[1]), l.ssmDt, l.ssmA, l.ssmCB, l.ssmD,
						at(o.Out[0]), outS, bs.dRows)
					g.LinearFused++
				case bs.deltaFused != nil:
					lc.la(bs.deltaFused, bs.scanThreads, inS, at(o.In[0]), at(o.In[1]), l.ssmDt, l.ssmA,
						at(o.Out[0]), outS, bs.dRows)
					g.LinearFused++
				default:
					g.deltaUnfused(lc, bs, l, li, R, inS, outS, &err)
				}
				if rk == nil && R > 1 {
					g.LinearBatched++
				}
				if g.recShared {
					cp := g.recCopies[[2]int{rr.StateLen, seqs}]
					if cp == nil && err == nil {
						err = fmt.Errorf("tier: block %d: no copy home for %d shared recurrent state(s)", li, seqs)
					}
					lc.la(cp, seqs*rr.StateLen, g.recNext, pool.s[0], bs.dRows)
				}
			case dkGatedNorm:
				rr := bs.rec
				inner := rr.VHeads * rr.VDim
				if rr.Mamba1 {
					// y*silu(z): the selective scan has no output norm.
					lc.la(bs.outActMul, R*inner, at(o.In[1]), at(o.In[0]), at(o.Out[0]))
					break
				}
				if rr.SSD {
					ng, nwid := headNormGeom(bs.ssdNormL, R*rr.NormGroups)
					switch {
					case rr.NoNorm:
						lc.la(bs.outActMul, R*inner, at(o.In[1]), at(o.In[0]), at(o.Out[0]))
					case rr.NormBeforeGate:
						if err == nil {
							err = lc.launch(bs.ssdNorm, ng, nwid, at(o.In[0]), l.ssmNorm, bs.dOutN)
						}
						lc.la(bs.outActMul, R*inner, at(o.In[1]), bs.dOutN, at(o.Out[0]))
					default:
						lc.la(bs.outActMul, R*inner, at(o.In[1]), at(o.In[0]), bs.dOutN)
						if err == nil {
							err = lc.launch(bs.ssdNorm, ng, nwid, bs.dOutN, l.ssmNorm, at(o.Out[0]))
						}
					}
					break
				}
				vg, vwid := headNormGeom(bs.outNormL, R*rr.VHeads)
				if rr.ChanDecay && l.k3 != nil && l.k3.fault == nn.K3FaultNoKDAGate {
					// The gate left out: the normed row is the output.
					if err == nil {
						err = lc.launch(bs.outNorm, vg, vwid, at(o.In[0]), l.ssmNorm, bs.dOutN)
					}
					buf[prog.slots[o.Out[0]]] = bs.dOutN
					break
				}
				if bs.outNormGate != nil && g.normGateOn(l, R == 1) {
					// The norm's epilogue gates it: the normalised row is never
					// written.
					in, z := at(o.In[0]), at(o.In[1])
					if g.normGateBroken() {
						in, z = z, in
					}
					if err == nil {
						err = lc.launch(bs.outNormGate, vg, vwid, in, l.ssmNorm, z, at(o.Out[0]))
					}
					g.NormGateRuns++
					break
				}
				if err == nil {
					err = lc.launch(bs.outNorm, vg, vwid, at(o.In[0]), l.ssmNorm, bs.dOutN)
				}
				if rr.ChanDecay || rr.SigmoidGate {
					lc.la(bs.outSigMul, R*inner, at(o.In[1]), bs.dOutN, at(o.Out[0]))
				} else {
					lc.la(bs.outActMul, R*inner, at(o.In[1]), bs.dOutN, at(o.Out[0]))
				}
			case dkSSMNorm:
				switch prog.slots[o.Out[0]] {
				case slotM1LowN:
					dg, dw := headNormGeom(bs.m1NormDtL, R)
					if err == nil {
						err = lc.launch(bs.m1NormDt, dg, dw, at(o.In[0]), l.m1DtN, at(o.Out[0]))
					}
				default:
					w := l.m1BN
					if prog.slots[o.Out[0]] == slotM1CN {
						w = l.m1CN
					}
					cg, cw := headNormGeom(bs.m1NormBCL, R)
					if err == nil {
						err = lc.launch(bs.m1NormBC, cg, cw, at(o.In[0]), w, at(o.Out[0]))
					}
				}
			case dkConvBias:
				lc.la(bs.m1Bias, R*bs.rec.Chans, at(o.In[0]), l.ssmCB, at(o.Out[0]))
			}
		}
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
			if len(o.Gen) != 0 && !(o.Kind == dkAdd && first+oi == len(prog.ops)-1 && prog.f16 && prog.slots[o.Out[0]] == slotX) {
				if o.Gen[0].former != 0 {
					genIdx = first + oi
					if o.FIn > 0 {
						// The float row the program states behind the op's
						// int8 operand, not whichever row the step last
						// quantized.
						fsrc = at(o.In[o.FIn-1])
					}
					form(&o)
				} else {
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
				if o.Gen[0].former != 0 {
					g.FormerOps++
				} else {
					g.GenOps++
				}
				continue
			}
			if R > 1 && o.OneRow {
				continue
			}
			switch o.Kind {
			case dkQuant:
				if x := bs.q4e; x != nil && prog.slots[o.Out[0]] == slotSQ8W {
					lc.la(x.quantW, kernels.QuantizeThreads(R*pad32(x.width)/32), at(o.In[0]), x.aW, x.axW)
					break
				}
				if x := bs.q4e; x != nil && (prog.slots[o.Out[0]] == slotSQ8 || prog.slots[o.Out[0]] == slotSQ8LR) {
					k, n, a, ax := x.quantHD, x.h*p.NEmbd, x.aHD, x.axHD
					if prog.slots[o.Out[0]] == slotSQ8LR {
						k, n, a, ax = x.quantLR, x.rank, x.aLR, x.axLR
					}
					lc.la(k, kernels.QuantizeThreads(R*pad32(n)/32), at(o.In[0]), a, ax)
					break
				}
				if prog.slots[o.Out[0]] == slotLRQ8 {
					// The low-rank projections' own operand (bs.dLRa, dLRax).
					k, n := bs.quantLR, bs.rec.KDim
					if bs.rec.Mamba1 {
						k, n = bs.m1Quant, bs.rec.Rank
					}
					lc.la(k, kernels.QuantizeThreads(R*pad32(n)/32), at(o.In[0]), bs.dLRa, bs.dLRax)
					break
				}
				src := at(o.In[0])
				fsrc = src
				switch w := prog.vals[o.In[0]].Width; {
				case prog.slots[o.In[0]] == slotEAct:
					// Every selected expert's activation, one block of scales
					// a slot: the layout the down bank's slots read.
					lc.la(bs.quantX, kernels.QuantizeThreads(bs.nUsed*pad32(p.NFFNExp)/32), src, bs.a, bs.ax)
				case prog.slots[o.In[0]] == slotShAct:
					lc.la(bs.shQuant, kernels.QuantizeThreads(R*bs.nShExp/32), src, bs.a, bs.ax)
				case prog.slots[o.In[0]] == slotLT:
					lc.la(bs.laurelQuant, kernels.QuantizeThreads(R*p.LaurelRank/32), src, bs.a, bs.ax)
				case prog.slots[o.In[0]] == slotPLEA:
					lc.la(bs.pleQuant, kernels.QuantizeThreads(R*pad32(p.PLEDim)/32), src, bs.a, bs.ax)
				case prog.slots[o.In[0]] == slotCQAN:
					lc.la(bs.ds4.quantQA, kernels.QuantizeThreads(R*p.QLoraRank/32), src, bs.a, bs.ax)
				case prog.slots[o.In[0]] == slotCAOR || prog.slots[o.In[0]] == slotCAO:
					lc.la(bs.quantQ, kernels.QuantizeThreads(R*p.NHead*p.HeadDim/32), src, bs.a, bs.ax)
				case prog.slots[o.In[0]] == slotCOA:
					lc.la(bs.ds4.quantOA, kernels.QuantizeThreads(R*p.DS4.OGroups*p.DS4.OLoraRank/32), src, bs.a, bs.ax)
				case prog.slots[o.In[0]] == slotLat || prog.slots[o.In[0]] == slotLatN || prog.slots[o.In[0]] == slotLatAcc:
					lc.la(bs.k3.quantL, kernels.QuantizeThreads(R*pad32(bs.k3.latent)/32), src, bs.a, bs.ax)
				case prog.slots[o.In[0]] == slotMGated && rowsMLA:
					lc.la(ml.quantOut, kernels.QuantizeThreads(R*ml.nh*ml.hdv/32), src, bs.a, bs.ax)
				case prog.slots[o.In[0]] == slotMGated:
					lc.la(bs.mlaQuantOut, kernels.QuantizeThreads(p.NHead*p.HeadDimV/32), src, bs.a, bs.ax)
				case prog.slots[o.In[0]] == slotDAct || prog.slots[o.In[0]] == slotDMixed:
					lc.la(bs.quantD, kernels.QuantizeThreads(R*pad32(bs.rec.VHeads*bs.rec.VDim)/32), src, bs.a, bs.ax)
				case rowsMLA && prog.slots[o.In[0]] == slotMQAn:
					lc.la(ml.qaQuant, kernels.QuantizeThreads(R*ml.qlr/32), src, bs.a, bs.ax)
				case rowsMLA && prog.slots[o.In[0]] == slotMNope:
					// Head-major for the grouped absorb.
					lc.la(ml.gatherNope, ml.nh*R*ml.nope, src, ml.permHM, ml.nopeHM)
					lc.la(ml.quantNope, kernels.QuantizeThreads(ml.nh*R*ml.nope/32), ml.nopeHM, bs.a, bs.ax)
				case rowsMLA && prog.slots[o.In[0]] == slotMAcc:
					lc.la(ml.gatherAcc, ml.nh*R*ml.lat, src, ml.permHM, ml.accHM)
					lc.la(ml.quantAcc, kernels.QuantizeThreads(ml.nh*R*ml.lat/32), ml.accHM, bs.a, bs.ax)
				case rowsMLA && prog.slots[o.In[0]] == slotMOut:
					lc.la(ml.quantOut, kernels.QuantizeThreads(R*ml.nh*ml.hdv/32), src, bs.a, bs.ax)
				case prog.slots[o.In[0]] == slotMQAn:
					lc.la(bs.mlaQuantQA, kernels.QuantizeThreads(p.QLoraRank/32), src, bs.a, bs.ax)
				case prog.slots[o.In[0]] == slotMNope:
					lc.la(bs.mlaQuantNope, kernels.QuantizeThreads(p.NHead*bs.mlaNopeW/32), src, bs.a, bs.ax)
				case prog.slots[o.In[0]] == slotMAcc:
					lc.la(bs.mlaQuantAcc, kernels.QuantizeThreads(p.NHead*bs.mlaLat/32), src, bs.a, bs.ax)
				case prog.slots[o.In[0]] == slotMOut:
					lc.la(bs.mlaQuantOut, kernels.QuantizeThreads(p.NHead*p.HeadDimV/32), src, bs.a, bs.ax)
				case w == p.NEmbd:
					lc.la(bs.quantE, kernels.QuantizeThreads(R*ePad(p)/32), src, bs.a, bs.ax)
				case w == qdim:
					lc.la(bs.quantQ, kernels.QuantizeThreads(R*qPad(p, qdim)/32), src, bs.a, bs.ax)
				default:
					lc.la(bs.quantF, kernels.QuantizeThreads(R*p.NFFN/32), src, bs.a, bs.ax)
				}
			case dkRopeQ:
				// The unrotated tail first: the rotary writes only the NRot
				// dimensions it turns, and the scores read the whole head.
				if bs.copyQ != nil {
					lc.la(bs.copyQ, R*qdim, at(o.In[0]), at(o.Out[0]), qOff)
				}
				lc.la(bs.ropeQ, R*p.NHead*p.NRot/2, at(o.In[0]), rcs, qOff, at(o.Out[0]))
			case dkAdd:
				// The block's last add, held for the next block's attention
				// norm to do where its program or the encoder folds it in
				// (foldAddNorm): one launch where a prompt chunk took two.
				if first+oi == len(prog.ops)-1 && prog.f16 && prog.slots[o.Out[0]] == slotX {
					if nk, conv := g.foldAddNorm(bs, l, li, hi, R, true, scaled, persist, at(o.In[1]), skip); nk != nil {
						pendX, pendY, pendNorm, pendConv = at(o.In[0]), at(o.In[1]), nk, conv
						break
					}
				}
				// A residual add reads the block's stream and takes its scale;
				// a sum of two branches (the attention's and a mixer's beside
				// it) is a plain add.
				if r0, r1 := prog.slots[o.In[0]], prog.slots[o.In[1]]; r0 != slotX && r0 != slotX2 && r1 != slotX && r1 != slotX2 {
					lc.la(bs.addE, R*p.NEmbd, at(o.In[0]), at(o.In[1]), at(o.Out[0]))
					break
				}
				addRes(at(o.In[0]), at(o.In[1]), at(o.Out[0]))
			case dkSplitQG:
				lc.la(bs.splitQG, R*qdim, at(o.In[0]), at(o.Out[0]), at(o.Out[1]))
			case dkRopeKRow:
				kro := bs.zero
				if R > 1 {
					kro = bs.kroff
				}
				lc.la(bs.ropeKRow, R*p.NKVHead*p.NRot/2, at(o.In[0]), rcs, kro, at(o.Out[0]))
			case dkIdxNorm:
				switch {
				case ix != nil:
					// The lightning indexer's key: a LayerNorm with its bias.
					lc.la(ix.lnPart, R*ix.parts, at(o.In[0]), ix.kPart)
					lc.la(ix.lnVar, R*ix.parts, at(o.In[0]), ix.kPart, ix.kVar)
					lc.la(ix.lnApply, R*ix.d, at(o.In[0]), l.nIdxK, ix.kPart, ix.kVar, l.nIdxKB, at(o.Out[0]))
				case prog.slots[o.Out[0]] == slotIKn && ms != nil:
					kg, kw := headNormGeom(ms.kLanes, R)
					if err == nil {
						err = lc.launch(ms.normK, kg, kw, at(o.In[0]), l.nIdxK, at(o.Out[0]))
					}
				case ms != nil:
					qg, qw := headNormGeom(ms.qLanes, R*ms.h)
					if err == nil {
						err = lc.launch(ms.normQ, qg, qw, at(o.In[0]), l.nIdxQ, at(o.Out[0]))
					}
				case qs != nil:
					qg, qw := headNormGeom(qs.qLanes, R*qs.h)
					if err == nil {
						err = lc.launch(qs.normQ, qg, qw, at(o.In[0]), l.nIdxQ, at(o.Out[0]))
					}
				}
			case dkIdxRope:
				if ix != nil {
					if prog.slots[o.Out[0]] == slotIKRot {
						lc.la(ix.copyK, R*ix.d, at(o.In[0]), at(o.Out[0]), ix.kOff)
						lc.la(ix.ropeK, R*p.NRot/2, at(o.In[0]), bs.cs, ix.kOff, at(o.Out[0]))
					} else {
						lc.la(ix.copyQ, R*ix.h*ix.d, at(o.In[0]), at(o.Out[0]), ix.qOff)
						lc.la(ix.ropeQ, R*ix.h*p.NRot/2, at(o.In[0]), bs.cs, ix.qOff, at(o.Out[0]))
					}
				} else if ms != nil {
					lc.la(ms.copyQ, R*ms.h*ms.d, at(o.In[0]), at(o.Out[0]), ms.qOff)
					lc.la(ms.ropeQ, R*ms.h*p.NRot/2, at(o.In[0]), bs.cs, ms.qOff, at(o.Out[0]))
				} else if qs != nil {
					lc.la(qs.copyQ, R*qs.h*qs.d, at(o.In[0]), at(o.Out[0]), qs.qOff)
					lc.la(qs.ropeQ, R*qs.h*p.NRot/2, at(o.In[0]), bs.cs, qs.qOff, at(o.Out[0]))
				}
			case dkIdxAssemble:
				if ms != nil {
					lc.la(ms.asmK, R*ms.attnW, at(o.In[0]), at(o.Out[0]), ms.allOff)
					lc.la(ms.asmI, R*ms.d, at(o.In[1]), at(o.Out[0]), ms.idxOff)
				} else if qs != nil {
					lc.la(qs.asmK, R*qs.attnW, at(o.In[0]), at(o.Out[0]), qs.allOff)
					lc.la(qs.asmI, R*qs.d, at(o.In[1]), at(o.Out[0]), qs.idxOff)
				}
			case dkIdxScore:
				if ms != nil {
					lc.la(ms.scores, R*ms.h*ms.tcap, at(o.In[0]), pl.k, pl.tab, pk.pRow, ms.score)
					lc.la(ms.bmax, R*ms.h*ms.nb, ms.score, pk.pRow, ms.bmx)
				} else if qs != nil {
					lc.la(qs.csTab, qs.nb*p.NRot/2, qs.cs, qs.planes, qs.consts, qs.csPos)
					lc.la(qs.dTab, qsaMarkCap*p.NRot/2, qs.dcs, qs.planes, qs.consts, qs.dpos)
					lc.la(qs.scores, R*qs.nb, at(o.In[0]), pl.k, pl.tab, pk.pRow, qs.cs, l.nIdxK, qs.marks, qs.dcs, qs.score)
				}
			case dkSelectBlocks:
				if ms != nil {
					lc.la(ms.sel, R*ms.h*ms.nb, ms.bmx, pk.pRow, ms.flag)
					lc.la(ms.compact, R*ms.h*ms.nb, ms.flag, ms.list)
				} else if qs != nil {
					lc.la(qs.sel, R*qs.groups*qs.nb, qs.score, pk.pRow, qs.flag)
					lc.la(qs.compact, R*qs.groups*qs.nb, qs.flag, qs.list)
				}
			case dkGatherKV:
				vw := 0
				switch {
				case ms != nil:
					if vw = ms.attnW; g.KVF16 {
						vw /= 2
					}
					lc.la(ms.gatherK, R*ms.attnW*ms.k*ms.blk, pl.k, pl.tab, pk.pRow, ms.list, ms.gk)
					lc.la(ms.gatherV, R*ms.k*ms.blk*vw, pl.v, pl.tab, pk.pRow, ms.list, ms.gv)
					lc.la(ms.desc, R*kernels.PRowWords, pk.pRow, ms.gdesc)
					g.MSASelects++
				case qs != nil:
					if vw = qs.attnW; g.KVF16 {
						vw /= 2
					}
					lc.la(qs.gatherK, R*qs.attnW*qs.k*qs.blk, pl.k, pl.tab, pk.pRow, qs.list, qs.gk)
					lc.la(qs.gatherV, R*qs.k*qs.blk*vw, pl.v, pl.tab, pk.pRow, qs.list, qs.gv)
					lc.la(qs.desc, R*kernels.PRowWords, pk.pRow, qs.gdesc)
					g.QSASelects++
				}
			case dkAttendGathered:
				var gk, gv, gtab, gdesc, sink backend.Buf
				switch {
				case l.ds4 != nil:
					// One key head that is the value, with sinks.
					x := bs.ds4
					gk, gv, gtab, gdesc, sink = x.gk, x.gk, x.gtab, x.gdesc, l.sinks
					if p.DS4.Fault == nn.DS4FaultNoSinks {
						sink = x.noSink
					}
				case ms != nil:
					gk, gv, gtab, gdesc = ms.gk, ms.gv, ms.gtab, ms.gdesc
				case qs != nil:
					gk, gv, gtab, gdesc = qs.gk, qs.gv, qs.gtab, qs.gdesc
				default:
					if err == nil {
						err = fmt.Errorf("tier: block %d selects its keys and has no selecting scratch here", li)
					}
					return
				}
				if err == nil {
					err = pk.pagedDecode(s, lc, at(o.In[0]), gk, gv, bs.n, at(o.Out[0]), gtab, gdesc, sink)
					g.PagedLaunches++
				}
			case dkStreamNorm:
				// A row a stream: the streams themselves through the stream
				// norm, a per-stream row of the n-gram's through the row norm.
				// The n-gram's norms are the tier's uploads (grouped, as its
				// kernels read them), not the container's vectors.
				k, w := bs.q4e.normS, o.V
				switch prog.slots[o.Out[0]] {
				case slotNKn:
					k, w = bs.q4e.normR, l.q4e.ngK
				case slotNQn:
					w = l.q4e.ngQ
				case slotNGvn:
					k, w = bs.q4e.normR, l.q4e.ngC
				}
				ds4Rows(lc, k, R*bs.q4e.h, at(o.In[0]), w, at(o.Out[0]))
			case dkIdxWrite:
				lc.la(ix.writeK, R*ix.d, at(o.In[0]), pl.k, bs.zero, pl.tab, pk.pRow)
			case dkIdxAttendLatent:
				lc.la(ix.scores, R*ix.tcap, at(o.In[1]), ix.wts, pl.k, pl.tab, pk.pRow, ix.score)
				lc.la(ix.sel, R*ix.tcap, ix.score, pk.pRow, ix.selBuf)
				lc.la(ix.gather, R*ix.k*ix.w, pl.k, pl.tab, pk.pRow, ix.selBuf, ix.gk)
				lc.la(ix.desc, R*kernels.PRowWords, pk.pRow, ix.gdesc)
				g.IdxSelects++
				if err == nil {
					err = pk.pagedDecode(s, lc, at(o.In[0]), ix.gk, ix.gk, bs.n, at(o.Out[0]), ix.gtab, ix.gdesc, nil)
					g.PagedLaunches++
				}
			case dkRowBias:
				g.ds4RouteBias(lc, bs, l, R)
			case dkStreamFlatten:
				lc.la(bs.ds4.hcFlat, R*p.DS4.HCMult*p.NEmbd, at(o.In[0]), at(o.Out[0]))
			case dkFlatNorm:
				ds4Rows(lc, bs.ds4.hcRms, R, at(o.In[0]), bs.ds4.onesFlat, at(o.Out[0]))
			case dkMixLogits:
				fn := l.ds4.fnF
				if o.N == 1 {
					fn = l.ds4.fnA
				}
				n := p.DS4.HCMult
				if err == nil {
					err = lc.launch(bs.ds4.hcFn, (2+n)*n*R, kernels.RouterThreads(1), fn, at(o.In[0]), at(o.Out[0]))
				}
			case dkSinkhorn:
				sc, bb := l.ds4.scF, l.ds4.bF
				if o.N == 1 {
					sc, bb = l.ds4.scA, l.ds4.bA
				}
				lc.la(bs.ds4.hcMix, R, at(o.In[0]), sc, bb, at(o.Out[0]))
			case dkStreamCollect:
				lc.la(bs.ds4.hcColl, R*p.NEmbd, at(o.In[0]), at(o.In[1]), at(o.Out[0]))
				g.DS4HC++
			case dkRowRMS:
				x, b := bs.ds4, l.ds4
				switch prog.slots[o.Out[0]] {
				case slotCQAN:
					ds4Rows(lc, x.rmsQ, R, at(o.In[0]), l.nQA, at(o.Out[0]))
				case slotCQN:
					ds4Rows(lc, x.rmsHeads, R*p.NHead, at(o.In[0]), x.onesHD, at(o.Out[0]))
				case slotCKN:
					ds4Rows(lc, x.rmsHD, R, at(o.In[0]), l.nKVA, at(o.Out[0]))
				case slotCPEN:
					ds4Rows(lc, x.rmsHD, R, at(o.In[0]), b.norm, at(o.Out[0]))
				case slotCIPEN:
					ds4Rows(lc, x.rmsIHD, R, at(o.In[0]), b.inorm, at(o.Out[0]))
				}
			case dkTurn:
				x, d := bs.ds4, p.DS4
				cs := bs.csSWA
				if l.ds4.comp != nn.DS4CompNone && d.Fault != nn.DS4FaultMainRope {
					cs = bs.cs
				}
				kd := x.kinds[l.ds4.comp]
				switch prog.slots[o.Out[0]] {
				case slotCIQR:
					lc.la(x.ropeIQ, R*d.IdxHeads*d.IdxHeadDim, at(o.In[0]), cs, at(o.Out[0]))
				case slotCQR:
					lc.la(x.ropeQ, R*p.NHead*p.HeadDim, at(o.In[0]), cs, at(o.Out[0]))
				case slotCKR:
					lc.la(x.ropeK, R*p.HeadDim, at(o.In[0]), cs, at(o.Out[0]))
				case slotCPER:
					lc.la(x.ropeK, R*p.HeadDim, at(o.In[0]), kd.csE, at(o.Out[0]))
				case slotCIPER:
					lc.la(x.ropeIE, R*d.IdxHeadDim, at(o.In[0]), kd.csE, at(o.Out[0]))
				case slotCAOR:
					lc.la(x.ropeI, R*p.NHead*p.HeadDim, at(o.In[0]), cs, at(o.Out[0]))
				}
			case dkCacheWrite:
				x, kd, d := bs.ds4, bs.ds4.kinds[l.ds4.comp], p.DS4
				switch o.N {
				case 0:
					lc.la(kd.wK, R*p.HeadDim, at(o.In[0]), pl.k, bs.zero, pl.tab, pk.pRow)
				case 1:
					cw := kd.coff * p.HeadDim
					lc.la(kd.ape, R*cw, at(o.In[1]), l.ds4.ape, pk.pRow, x.cgA)
					lc.la(kd.wKV, R*cw, at(o.In[0]), pl.k, bs.zero, pl.tab, pk.pRow)
					lc.la(kd.wG, R*cw, x.cgA, pl.k, bs.zero, pl.tab, pk.pRow)
				case 2:
					iw := 2 * d.IdxHeadDim
					lc.la(kd.apeI, R*iw, at(o.In[1]), l.ds4.iape, pk.pRow, x.igA)
					lc.la(kd.wIKV, R*iw, at(o.In[0]), pl.k, bs.zero, pl.tab, pk.pRow)
					lc.la(kd.wIG, R*iw, x.igA, pl.k, bs.zero, pl.tab, pk.pRow)
				}
			case dkPoolEntries:
				kd := bs.ds4.kinds[l.ds4.comp]
				if o.N == 0 {
					lc.la(kd.pool, R*p.HeadDim, pl.k, pl.tab, pk.pRow, at(o.Out[0]))
				} else {
					lc.la(kd.poolI, R*p.DS4.IdxHeadDim, pl.k, pl.tab, pk.pRow, at(o.Out[0]))
				}
			case dkEntryWrite:
				kd, ent := bs.ds4.kinds[l.ds4.comp], g.kvp.layers[entKey(li)]
				if ent == nil {
					if err == nil {
						err = fmt.Errorf("tier: block %d has no entry pages in this device's pool", li)
					}
					return
				}
				if o.N == 0 {
					lc.la(kd.entW, R*p.HeadDim, at(o.In[0]), ent.k, ent.tab, pk.pRow)
				} else {
					lc.la(kd.entWI, R*p.DS4.IdxHeadDim, at(o.In[0]), ent.k, ent.tab, pk.pRow)
				}
			case dkEntrySelect:
				x, d, kd, ent := bs.ds4, p.DS4, bs.ds4.kinds[l.ds4.comp], g.kvp.layers[entKey(li)]
				if keys := ds4Keys(d, l.ds4.comp); keys.All || keys.None || ent == nil {
					break // every visible entry, or none: nothing to select
				}
				tcap := p.MaxSeq/kd.rate + 1
				lc.la(x.entDesc, R*kernels.PRowWords, pk.pRow, x.edesc)
				lc.la(x.idxSc, R*tcap, at(o.In[0]), at(o.In[1]), ent.k, ent.tab, x.edesc, x.score)
				lc.la(x.idxSl, R*tcap, x.score, x.edesc, at(o.Out[0]))
				g.IdxSelects++
			case dkGatherEntries:
				x, kd := bs.ds4, bs.ds4.kinds[l.ds4.comp]
				ek, et := pl.k, pl.tab
				if ent := g.kvp.layers[entKey(li)]; ent != nil {
					ek, et = ent.k, ent.tab
				}
				lc.la(kd.gather, R*x.kmax*p.HeadDim, pl.k, pl.tab, ek, et, pk.pRow, x.sel, at(o.Out[0]))
				lc.la(kd.desc, R*kernels.PRowWords, pk.pRow, x.gdesc)
			case dkGroupedMatVec:
				wa, ok := l.ds4WoA(R)
				if !ok {
					if err == nil {
						err = errNoBatch
					}
					return
				}
				mvid(wa, l.woA, at(o.Out[0]), bs.ds4.oSel)
				g.DS4Blocks++
			case dkStreamScore:
				lc.la(bs.k3.score[o.N], R*(o.N+1), at(o.In[0]), o.V, at(o.Out[0]))
			case dkStreamWeighted:
				lc.la(bs.k3.mixK[o.N], R*p.NEmbd, at(o.In[0]), at(o.In[1]), at(o.Out[0]))
				g.K3Mixes++
			case dkStreamDot:
				ds4Rows(lc, bs.q4e.streamDot, R*bs.q4e.h, at(o.In[0]), at(o.In[1]), at(o.Out[0]))
			case dkGatedVal:
				lc.la(bs.q4e.gatedVal, R*bs.q4e.h*p.NEmbd, at(o.In[0]), at(o.In[1]), at(o.Out[0]))
			case dkNGramConv:
				// The block's recurrent state carries the convolution's
				// history, at the step's parity.
				par, has := g.stepPar(sid, l)
				pool := l.pool
				if !has || pool == nil {
					if err == nil {
						err = fmt.Errorf("tier: block %d has no recurrent state for session %d", li, sid)
					}
					return
				}
				inS, outS := pool.s[par], pool.s[1-par]
				if g.recShared {
					inS, outS = pool.s[0], g.recNext
				}
				conv := bs.q4e.ngConv
				if rag != nil {
					if rk := bs.ragLin[len(rag.pos)]; rk != nil {
						conv = rk.ngConv
					}
				}
				// The convolution's taps as the tier uploaded them (tap-major).
				lc.la(conv, bs.q4e.h*p.NEmbd, inS, at(o.In[0]), l.q4e.ngConv, bs.dRows, at(o.Out[0]), outS)
			case dkNGramAdd:
				lc.la(bs.q4e.ngAdd, bs.q4e.h*R*p.NEmbd, at(o.In[0]), at(o.In[1]), at(o.In[2]), at(o.Out[0]))
				g.Q4ENGram++
			case dkStreamGateMix:
				lc.la(bs.q4e.gateMix, R*p.NEmbd, at(o.In[0]), at(o.In[1]), at(o.Out[0]))
				g.Q4EMixes++
			case dkPlaceWeights:
				lc.la(bs.q4e.injK, R*bs.q4e.h, at(o.In[0]), at(o.Out[0]))
			case dkStreamPlace:
				if prog.slots[o.In[2%len(o.In)]] == slotHC {
					lc.la(bs.ds4.hcPost, p.DS4.HCMult*R*p.NEmbd, at(o.In[0]), at(o.In[1]), at(o.In[2]), at(o.Out[0]))
					break
				}
				if len(o.In) == 2 {
					// The banked placement: the checkpoint's kernel banks the
					// block's input as it places.
					lc.la(bs.k3.post[o.N], bs.k3.streams*R*p.NEmbd, at(o.In[0]), at(o.In[1]), at(o.Out[0]))
					break
				}
				lc.la(bs.q4e.place, bs.q4e.h*R*p.NEmbd, at(o.In[0]), at(o.In[1]), at(o.In[2]), at(o.Out[0]))
			case dkStreamRoute:
				lc.la(bs.altRoute, R*p.AltUp, at(o.In[0]), l.altRouter, at(o.Out[0]))
			case dkStreamPredict:
				lc.la(bs.altPredictK, R*p.AltUp*p.NEmbd, at(o.In[0]), at(o.In[1]), l.altPredT, at(o.Out[0]))
				g.AltUpBlocks++
			case dkStreamCorrect:
				lc.la(bs.altCorrectK, R*p.AltUp*p.NEmbd, at(o.In[0]), at(o.In[1]), at(o.In[2]), l.altCorrT, at(o.Out[0]))
			case dkStreamScale:
				lc.la(bs.altMul, R*p.NEmbd, at(o.In[0]), l.altCorrScale, at(o.Out[0]))
			case dkStreamFinish:
				lc.la(bs.altFinish, R*p.AltUp*p.NEmbd, at(o.In[0]), at(o.In[1]), at(o.Out[0]))
			case dkGaussTopK:
				if g.emitGauss(lc, bs, l, R) != at(o.Out[0]) {
					err = fmt.Errorf("tier: block %d: the gaussian top-k has no kernels on this device", li)
					return
				}
			case dkPLESlice:
				if bs.pleIn == nil {
					if err == nil {
						err = fmt.Errorf("tier: block %d: no per-layer inputs on this step", li)
					}
					return
				}
				lc.la(bs.pleSlice, R*p.PLEDim, bs.pleIn, l.pleOff, at(o.Out[0]))
			case dkQTemp:
				if at(o.Out[0]) == nil {
					if err == nil {
						err = fmt.Errorf("tier: block %d: no buffer for a scaled rotated query", li)
					}
					return
				}
				lc.la(bs.qTemp, R*qdim, at(o.In[0]), bs.n, bs.tempTab, at(o.Out[0]))
			case dkAct:
				// An ungated FFN's activation; Apertus' xIELU reads its four
				// numbers as a buffer.
				nf := prog.vals[o.In[0]].Width
				switch {
				case prog.slots[o.Out[0]] == slotEAct && rowsMoE:
					lg(mg.actMul, gr.cols*p.NFFNExp, 128, at(o.In[0]), at(o.Out[0]))
				case prog.slots[o.Out[0]] == slotEAct:
					lc.la(bs.actMulE, bs.nUsed*bs.nFFNExp, at(o.In[0]), at(o.Out[0]))
				case prog.slots[o.Out[0]] == slotShAct:
					lc.la(bs.shActMul, R*bs.nShExp, at(o.In[0]), at(o.Out[0]))
				case l.xielu != nil:
					lc.la(bs.actMul, R*nf, at(o.In[0]), l.xielu, at(o.Out[0]))
				default:
					lc.la(blockAct(bs, l), R*nf, at(o.In[0]), at(o.Out[0]))
				}
			case dkRecurrent:
				// The recurrent mixer with its pre-norm and quantize, as the
				// encoder runs it, ending in bs.mvOut (emitLinear).
				g.emitLinear(sid, s, bs, l, li, R, rag != nil, p, &err, lc, mvrun, mvrunAct)
				if err != nil {
					return
				}
			case dkGate2, dkGate2In:
				wu, wg := l.up, l.gate
				if g.gateUp2Swapped() {
					wu, wg = wg, wu
				}
				n := l.mvu.rows * l.mvu.split
				if o.Kind == dkGate2In {
					lamv(l.mvu, l.mvu.gate2In, n, wu.qs, wu.d, wu.sc, at(o.In[0]), o.V, at(o.Out[0]), wg.qs, wg.d, wg.sc)
					g.Gate2InRuns++
					g.ActInLaunches++
				} else {
					lamv(l.mvu, l.mvu.gate2, n, wu.qs, wu.d, wu.sc, bs.a, bs.ax, at(o.Out[0]), wg.qs, wg.d, wg.sc)
					g.GateUp2Runs++
				}
			case dkQKVIn:
				var bufArr [17]backend.Buf
				bufs := append(bufArr[:0], at(o.In[0]), o.V)
				for i, sg := range [3]struct {
					m mv
					r *resident
				}{{l.mvq, l.wq}, {l.mvk, l.wk}, {l.mvv, l.wv}} {
					bufs = append(bufs, sg.r.qs, sg.r.d, sg.r.sc, at(o.Out[i]))
					if sg.m.bias != nil {
						bufs = append(bufs, sg.m.bias)
					}
				}
				if err == nil {
					err = lc.launch(l.qkvIn, l.qkvBlocks, l.qkvW, bufs...)
				}
				g.ActInLaunches++
				g.ActInSaved++
			case dkFFNIn:
				mvIn(l.mvg, l.mvg.kernIn, l.gate, at(o.Out[0]), at(o.In[0]), o.V, nil)
				if o.Role == jlm.RoleFFNUp {
					mvIn(l.mvu, l.mvu.gatedIn, l.up, at(o.Out[1]), at(o.In[0]), o.V, at(o.Out[0]))
				} else {
					mvIn(l.mvu, l.mvu.kernIn, l.up, at(o.Out[1]), at(o.In[0]), o.V, nil)
				}
				g.ActInSaved++
			case dkDownAct:
				mvIn(l.mvd, l.mvd.resActIn, l.down, at(o.Out[0]), at(o.In[0]), at(o.In[1]), at(o.In[2]))
				g.ActInSaved++
			case dkNormF16:
				laRows(o.K, at(o.In[0]), o.V, at(o.Out[0]), g.f16Buf)
				actMemoK, actMemoSrc = o.Conv, at(o.Out[0])
				g.NormF16++
			case dkAddNormF16:
				laRows(o.K, at(o.In[0]), at(o.In[1]), o.V, at(o.Out[0]), at(o.Out[1]), g.f16Buf)
				actMemoK, actMemoSrc = o.Conv, at(o.Out[1])
				g.NormF16++
				g.AddNormFolded++
			case dkActF16:
				// The product goes to down's operand only; its float row is
				// named in the memo, which down's GEMM finds written.
				lc.la(o.K, o.N, at(o.In[0]), at(o.In[1]), g.f16Buf)
				actMemoK, actMemoSrc = o.Conv, at(o.Out[0])
				g.ActF16Fused++
			default:
				// Every op runs generated launches or a former's; a kind
				// with neither has nothing that runs it, which is a
				// compiler bug, not a step to skip.
				if err == nil {
					err = fmt.Errorf("tier: block %d: op %s has no generated kernel (%v; last: %s)", li, o.Kind, o.Why, g.LastErr)
				}
				return
			}
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
			attnFused, ffnFused := false, false
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
			l.ran.note(formBlock, uint8(blockEncoder))
			// Gemma 3n: every stream's prediction, the block run on the active
			// one (altup.go).
			if l.altRouter != nil {
				g.emitAltPredict(lc, bs, l, R, norm)
			}
			// A linear block replaces the attention half and ends where it ends, in
			// bs.mvOut, so everything below is shared (RULE 8a's block as a unit).
			if l.linear {
				g.emitLinear(sid, s, bs, l, li, R, rag != nil, p, &err, lc, mvrun, mvrunAct)
			}
			if l.ds4 != nil {
				// DeepSeek V4: the hyper-connection's mix, then its own
				// attention, ending in bs.mvOut (ds4.go).
				pl, ent := g.kvp.layers[li], g.kvp.layers[entKey(li)]
				if pl == nil || bs.pkv == nil || bs.pkv.cur == nil {
					if err == nil {
						err = fmt.Errorf("tier: block %d has no pages in this device's pool", li)
					}
					return
				}
				g.ds4Pre(lc, bs, l, R, bs.x, true, norm)
				pc := pagedCall{pk: bs.pkv, pl: pl, k: pl.k, tab: pl.tab, desc: bs.pkv.pRow}
				g.PagedLaunches++
				g.ds4Attn(s, lc, bs, l, R, &pc, ent, mvrun, mvid)
			}
			// A block that runs attention beside its mixer (Falcon-H1) runs the
			// attention half too, from the same norm, and the two outputs are
			// summed below.
			if (!l.linear || l.withAttn) && l.ds4 == nil {
				// This session's attention history for the block -- its source's,
				// for a KV-sharing block -- while the weights are the model's and
				// shared. See layer.kv.
				// A row-chunked block's second phase reads the k and v its
				// first phase wrote for every chunk, as a KV-sharing block
				// reads its source's: q alone.
				shared := l.kvSrc != li || p.NonCausal && g.ncPhase == 2
				kvp := g.layers[l.kvSrc].kvOf(sid)
				if kvp == nil {
					// A missing cache fails the submission rather than launching against
					// nil buffers: PrepLayer returns false when a second session's KV does
					// not fit, so this is reachable on a nearly full card.
					if err == nil {
						err = fmt.Errorf("tier: block %d has no KV cache for session %d",
							li, sid)
					}
					return
				}
				// --- attention
				// Kimi-K3's block reads the mix of its banked checkpoints and
				// the running residual (k3.go); every other block bs.x.
				ain := bs.x
				if l.k3 != nil {
					ain = g.k3AttnIn(lc, bs, l, R)
				}
				if l.q4e != nil {
					// The hyper-connection's mix (q4e.go); an attention
					// block carries no n-gram embedding, so no state.
					ain = g.q4eAttnIn(lc, bs, l, R, mvrunAct, nil, nil, nil)
				}
				// The operand prologue (fuseActIn): q/k/v normalise and quantize
				// the block input themselves, and the norm launch goes. Only
				// where q/k/v run as the fused launch below (qkvIn) or as
				// their three launches (qkvSep), and nothing else reads the
				// normed row (bs.h).
				qkvPro := R == 1 && rag == nil && g.actInOn() && l.k3 == nil && l.q4e == nil && !l.mla && l.clampB == nil &&
					l.kvSrc == li && !l.vFromK && !g.TableSplit && l.altRouter == nil && bs.normVar == nil &&
					l.nAttnB == nil && l.nAttn != nil && !voltaOnly && !skip["norm"] && !skip["quant"] && !skip["qkvo"]
				qkvIn := qkvPro && l.qkv != nil && l.qkvIn != nil &&
					l.qkvBias == [3]bool{l.mvq.bias != nil, l.mvk.bias != nil, l.mvv.bias != nil}
				qkvSep := qkvPro && l.qkv == nil && l.mvq.kernIn != nil
				if qkvIn || qkvSep {
					g.ActInSaved++
				} else if voltaOnly && !skip["norm"] {
					// The norm writes the first projection's binary16 operand
					// too, and the memo tells its GEMM so.
					if pendX != nil {
						// The previous block's residual add and this norm, one
						// launch: x = pendX + pendY and h = RMSNorm(x).
						y := pendY
						if g.addNormBroken() {
							y = pendX
						}
						laRows(pendNorm, pendX, y, l.nAttn, bs.x, bs.h, g.f16Buf)
						actMemoK, actMemoSrc = pendConv, bs.h
						pendX, pendY = nil, nil
						g.NormF16++
						g.AddNormFolded++
					} else {
						nk, conv := backend.Kernel(nil), backend.Kernel(nil)
						if l.nAttnB == nil && l.nAttn != nil && l.altRouter == nil {
							nk, conv = g.rmsF16For(bs, l.mvq, R, false)
						}
						if nk != nil {
							laRows(nk, ain, l.nAttn, bs.h, g.f16Buf)
							actMemoK, actMemoSrc = conv, bs.h
							g.NormF16++
						} else {
							norm(ain, l.nAttn, l.nAttnB)
						}
					}
					fsrc = bs.h
					g.QuantSkipped++
				} else if !skip["norm"] && !skip["quant"] && bs.rmsQ != nil && bs.normVar == nil && l.nAttnB == nil &&
					l.nAttn != nil && l.altRouter == nil {
					laRows(bs.rmsQ, ain, l.nAttn, bs.h, bs.a, bs.ax)
					fsrc = bs.h // what la records for a quantize: float weights read it
				} else {
					if !skip["norm"] {
						norm(ain, l.nAttn, l.nAttnB)
					}
					if !skip["quant"] {
						lc.la(bs.quantE, kernels.QuantizeThreads(R*ePad(p)/32), bs.h, bs.a, bs.ax)
					}
					if l.altRouter != nil {
						g.emitLaurel(lc, bs, l, R, mvrun, norm, &fsrc)
					}
				}
				// MLA takes the rest of the attention half and ends in bs.mvOut, the
				// same seam emitLinear sits on.
				var pc *pagedCall
				if bs.pkv != nil && l.mla {
					pl := g.kvp.layers[li]
					if pl == nil {
						if err == nil {
							err = fmt.Errorf("tier: block %d has no pages in this device's pool", li)
						}
						return
					}
					pc = &pagedCall{pk: bs.pkv, pl: pl, k: pl.k, tab: pl.tab, desc: bs.pkv.pRow}
					g.PagedLaunches++
					if !bs.pkv.prefill && bs.pkv.cur.plan.Path == kernels.PathStaged {
						g.PagedStagedLaunches++
					}
				}
				// Kimi-K3's MLA output gate reads the normed input's
				// quantization, which the query's second step replaces.
				if l.mla && l.mlaGate != nil {
					g.k3GateMLA(bs, l, mvrun)
				}
				if l.mla && R > 1 {
					g.emitMLARows(s, bs, l, kvp, p, &err, lc, mvrun, sm, smG, smW, nCap, pc)
				} else if l.mla {
					g.emitMLA(s, bs, l, kvp, p, &err, lc, mvrun, mvid, sm, smG, smW, nCap, skip, pc)
				} else {
					// A few-sequence step takes decode's one launch over its rows.
					var seg segLaunch
					segOK := false
					// A block with no v projection (Gemma 4's global layers:
					// v is k's) takes neither fused launch, both of which write
					// three outputs.
					vFromK := l.vFromK
					if rag != nil && !g.TableSplit && !vFromK && !shared {
						seg, segOK = g.groupQKV(l, len(rag.pos), R)
					}
					qf, qs := g.qkvLowering(l, R, rag != nil, shared, vFromK, segOK, qkvSep)
					l.ran.note(formQKV, uint8(qf))
					if !skip["qkvo"] {
						if qf == qkvQOnly {
							// q alone: k and v are the source's, already cached.
							mvrun(l.mvq, l.wq, bs.q)
						} else if qf == qkvQK {
							mvrun(l.mvq, l.wq, bs.q)
							mvrun(l.mvk, l.wk, bs.k)
						} else if qf == qkvGrouped {
							kern, blocks, w := l.qkv, l.qkvBlocks, l.qkvW
							if segOK {
								kern, blocks, w = seg.kern, seg.blocks, seg.w
								g.RagQKVFused++
							}
							// Built in a stack array: two, then up to five a
							// projection; appending past a literal's length
							// grew it on the heap every block.
							var bufArr [17]backend.Buf
							bufs := append(bufArr[:0], bs.a, bs.ax)
							if qkvIn {
								kern, bufs = l.qkvIn, append(bufArr[:0], ain, l.nAttn)
								g.ActInLaunches++
							}
							for _, sg := range []struct {
								m   mv
								r   *resident
								out backend.Buf
							}{{l.mvq, l.wq, bs.q}, {l.mvk, l.wk, bs.k}, {l.mvv, l.wv, bs.v}} {
								bufs = append(bufs, sg.r.qs, sg.r.d, sg.r.sc, sg.out)
								if sg.m.bias != nil {
									bufs = append(bufs, sg.m.bias)
								}
							}
							if err == nil {
								err = lc.launch(kern, blocks, w, bufs...)
							}
						} else if qf == qkvClamped {
							// Gemma 4's clipped linears: each projection reads its
							// own clamped copy of the normed rows, quantized for it,
							// and writes into a buffer its output clamp reads.
							for s, pr := range [3]struct {
								m   mv
								r   *resident
								out backend.Buf
								w   int
							}{{l.mvq, l.wq, bs.q, qdim}, {l.mvk, l.wk, bs.k, kvDim}, {l.mvv, l.wv, bs.v, kvDim}} {
								lc.la(bs.clampK[2*s], R*p.NEmbd, bs.h, l.clampB, bs.cin)
								lc.la(bs.quantE, kernels.QuantizeThreads(R*ePad(p)/32), bs.cin, bs.a, bs.ax)
								fsrc = bs.cin
								mvrun(pr.m, pr.r, pr.out)
							}
						} else if qf == qkvSeparate {
							mvIn(l.mvq, l.mvq.kernIn, l.wq, bs.q, ain, l.nAttn, l.mvq.bias)
							mvIn(l.mvk, l.mvk.kernIn, l.wk, bs.k, ain, l.nAttn, l.mvk.bias)
							mvIn(l.mvv, l.mvv.kernIn, l.wv, bs.v, ain, l.nAttn, l.mvv.bias)
						} else if qf == qkvGEMM {
							// q, k and v as one grid over one binary16 operand
							// (GemmVoltaSegments); a projection left out runs alone.
							if qs.act != actMemoK || fsrc != actMemoSrc {
								lc.la(qs.act, qs.actN, fsrc, g.f16Buf)
								g.ActF16Launches++
							}
							actMemoK, actMemoSrc = qs.act, fsrc
							all := [3]struct {
								m   mv
								r   *resident
								out backend.Buf
							}{{l.mvq, l.wq, bs.q}, {l.mvk, l.wk, bs.k}, {l.mvv, l.wv, bs.v}}
							var arr [16]backend.Buf
							bufs := append(arr[:0], g.f16Buf)
							for _, i := range qs.segs {
								sg := all[i]
								w := sg.r
								if g.qkvGemmSwapped() && i > 0 && qs.has(3-i) && all[3-i].r.t == w.t && all[3-i].m.rows == sg.m.rows {
									w = all[3-i].r
								}
								bufs = append(bufs, w.qs, w.d, w.sc, sg.out)
								if sg.m.bias != nil {
									bufs = append(bufs, sg.m.bias)
								}
							}
							lc.law(qs.kern, qs.threads, qs.width, bufs...)
							g.QKVGemmRuns++
							for i, sg := range all {
								if !qs.has(i) {
									mvrun(sg.m, sg.r, sg.out)
								}
							}
						} else {
							mvrun(l.mvq, l.wq, bs.q)
							mvrun(l.mvk, l.wk, bs.k)
							mvrun(l.mvv, l.wv, bs.v)
						}
					}
					// The per-head norm goes between the projection and RoPE, and only
					// there: a rotation preserves each pair's norm but not the per-head
					// scaling, so norming afterwards is a different function.
					qSrc, kSrc, vSrc := bs.q, bs.k, bs.v
					// DBRX clamps all three before anything reads them, as llama.cpp and
					// transformers both do.
					if bs.clampQ != nil {
						lc.la(bs.clampQ, R*qdim, bs.q, bs.qClamp)
						lc.la(bs.clampKV, R*kvDim, bs.k, bs.kClamp)
						lc.la(bs.clampKV, R*kvDim, bs.v, bs.vClamp)
						qSrc, kSrc, vSrc = bs.qClamp, bs.kClamp, bs.vClamp
					}
					if l.clampB != nil {
						lc.la(bs.clampK[1], R*qdim, bs.q, l.clampB, bs.qClamp)
						lc.la(bs.clampK[3], R*kvDim, bs.k, l.clampB, bs.kClamp)
						lc.la(bs.clampK[5], R*kvDim, bs.v, l.clampB, bs.vClamp)
						qSrc, kSrc, vSrc = bs.qClamp, bs.kClamp, bs.vClamp
					}
					if bs.splitQG != nil {
						// Deinterleave before anything reads q: the norm, RoPE and
						// the scores all want the query alone, qdim wide.
						lc.la(bs.splitQG, R*qdim, bs.q, bs.qg, bs.ogate)
						qSrc = bs.qg
					}
					// Gemma 4's global layers: v is k's projection, read before
					// k's norm writes bs.kn and before the rotary.
					if vFromK {
						vSrc = kSrc
					}
					kOff := bs.koff
					if kst > 0 {
						kOff = bs.kpos
					}
					qOff := bs.zero
					if R > 1 {
						qOff = bs.roff
					}
					// Llama 4's weightless norm runs on the rotating layers
					// only, against ones (see bs.l4Ones for why before RoPE).
					nQ, nK := l.nQ, l.nK
					if nQ == nil && p.QKL2Norm && p.RopeAt(li) {
						nQ, nK = bs.l4Ones, bs.l4Ones
					}
					if bs.qNorm != nil && nQ != nil {
						// The geometry follows the kernel's lanes (headNormGeom). A wide norm
						// is one head per row, whatever NHead is.
						qh, kh := R*p.NHead, R*p.NKVHead
						if p.QKNorm && p.QKNormWide {
							qh, kh = R, R
						}
						qg, qw := headNormGeom(bs.qNormL, qh)
						kg, kw := headNormGeom(bs.kNormL, kh)
						if bs.qkRms != nil {
							// One whole-row norm per vector: llama.cpp's
							// rms_norm_f32 shape, two launches not four.
							laRows(bs.qkRms, qSrc, l.nQ, bs.qn)
							laRows(bs.qkRms, kSrc, l.nK, bs.kn)
							g.QKRms++
						} else if bs.qkPart != nil && bs.qkApply != nil {
							w := p.NHead * p.HeadDim
							lc.la(bs.qkPart, R*bs.parts, qSrc, bs.part)
							lc.la(bs.qkApply, R*w, qSrc, nQ, bs.part, bs.qn)
							lc.la(bs.qkPart, R*bs.parts, kSrc, bs.part)
							lc.la(bs.qkApply, R*w, kSrc, nK, bs.part, bs.kn)
						} else {
							if err == nil && !p.QKNormPost {
								// qSrc, not bs.q: with an attention out-gate bs.q is
								// the double-width interleaved projection and the norm
								// wants the query alone.
								err = lc.launch(bs.qNorm, qg, qw, qSrc, nQ, bs.qn)
							}
							if err == nil && !shared && !p.RopeSplit {
								err = lc.launch(bs.kNorm, kg, kw, kSrc, nK, bs.kn)
							}
						}
						// Hunyuan's q is normed after the rotary instead (below),
						// and under XD-RoPE its k too.
						if !p.QKNormPost {
							qSrc = bs.qn
						}
						if !p.RopeSplit {
							kSrc = bs.kn
						}
						// Gemma 4's v: each head RMSNormed with no weight, on k's
						// head-norm kernel (the same heads and width) against ones.
						if p.VNorm && bs.vn != nil && err == nil && !shared {
							err = lc.launch(bs.kNorm, kg, kw, vSrc, bs.hdOnes, bs.vn)
							vSrc = bs.vn
						}
					}
					// Per layer: Llama 4 leaves every SWAPeriod'th layer unrotated.
					roped := bs.ropeQ != nil && !l.noRope && (p.RopeAt(li) || rag != nil && g.rowsFaulted(rowsNope))
					// mistral3's temperature reaches the rotated layers too. A row's
					// factor and its rotation are both linear in q, so q is scaled
					// first, into qt, and rotated from there.
					if roped && !p.NoPEGlobal && bs.qTemp != nil && bs.qt != nil &&
						!(rag != nil && g.rowsFaulted(rowsNotemp)) {
						lc.la(bs.qTemp, R*qdim, qSrc, bs.n, bs.tempTab, bs.qt)
						qSrc = bs.qt
					}
					// Where K and V go: this session's contiguous cache at the
					// row's offset, or the pool's layer at the page each row's
					// descriptor names (pagedattn.go), the offset unread.
					copyK, ropeK, copyV := bs.copyK, bs.ropeK, bs.copyV
					kDst, vDst, vOff := kvp.kc, kvp.vc, bs.koff
					var pgArgs []backend.Buf
					var pgTab backend.Buf // this layer's table arena
					pk := bs.pkv
					if pk != nil {
						pl := g.kvp.layers[l.kvSrc]
						if pl == nil {
							if err == nil {
								err = fmt.Errorf("tier: block %d has no pages in this device's pool", li)
							}
							return
						}
						copyK, ropeK, copyV = pk.copyK, pk.ropeK, pk.copyV
						kDst, vDst, kOff, vOff = pl.k, pl.v, bs.zero, bs.zero
						pgTab = pl.tab
						pgArgs = []backend.Buf{pl.tab, pk.pRow}
					}
					laKV := func(k backend.Kernel, threads int, bufs ...backend.Buf) {
						// Joined in a stack array: append to bufs itself would
						// grow it on the heap every launch.
						var all [8]backend.Buf
						lc.la(k, threads, append(append(all[:0], bufs...), pgArgs...)...)
					}
					// MiniMax-M3's selecting block: the indexer's key joins k's
					// heads in one row, which the pool's writers turn and store
					// together; its query waits for the selection (msa.go).
					kHeads, vThreads := p.NKVHead, R*p.NKVHead*p.HeadDim
					if l.msa {
						if pk == nil || pk.msa == nil {
							if err == nil {
								err = fmt.Errorf("tier: block %d selects its keys and has no paged scratch here", li)
							}
							return
						}
						if err == nil {
							err = g.msaKeyQuery(lc, bs, l, R, kSrc, bs.cs, mvrun)
						}
						kSrc, kHeads = pk.msa.kAll, kHeads+1
					}
					// Qwen3.8-Flash-Next's: the raw key joins the row,
					// which is copied whole; only k's heads turn (qsa.go).
					if l.qsa {
						if pk == nil || pk.qsa == nil {
							if err == nil {
								err = fmt.Errorf("tier: block %d selects its keys and has no paged scratch here", li)
							}
							return
						}
						if err == nil {
							err = g.qsaKeyQuery(lc, bs, l, R, kSrc, bs.cs, mvrun)
						}
						kSrc = pk.qsa.kAll
					}
					fusedKV := false
					if !skip["rope"] {
						// An unrotated layer still needs K and V in the cache: q is read where
						// it is (qAtt below) and k is copied where RoPERowsT would have
						// written it. A KV-sharing block writes neither: its source did.
						if !roped && !shared {
							laKV(copyK, R*kvDim, kSrc, kDst, kOff)
						} else if roped {
							// The table layer li was trained with: gemma3 puts five layers in
							// six on the local base (see nn.LayerPlan.SWAPeriod), ModernBERT
							// all but every third from the first (LocalRope).
							rcs := bs.cs
							if bs.csSWA != nil && p.LocalRope(li) {
								rcs = bs.csSWA
							}
							// The unrotated tail first: RoPERows writes only the nRot
							// dimensions it rotates, and the scores read the whole head. It
							// copies the normalized q and k, as the host passes the tail
							// through, rather than zeros.
							if bs.copyQ != nil {
								lc.la(bs.copyQ, R*qdim, qSrc, bs.qr, qOff)
							}
							if copyK != nil && (partialRotary(p) || l.qsa) && !shared {
								laKV(copyK, R*kvDim, kSrc, kDst, kOff)
							}
							if pk != nil && pk.ropeQKV != nil && !shared && bs.copyQ == nil && kHeads == p.NKVHead &&
								!g.NoFuse && !skip["fuse"] {
								// q's rotation, k's into its page and v's copy
								// in one launch (kernels.RoPEQKVPaged), the
								// three kernels below being its parts.
								lc.la(pk.ropeQKV, kernels.RoPEQKVPagedThreads(p.NHead, p.NKVHead, p.HeadDim, p.NRot, R, g.KVF16),
									qSrc, rcs, qOff, bs.qr, kSrc, kDst, vSrc, vDst, pgTab, pk.pRow)
								g.FusedLaunches++
								fusedKV = true
							} else {
								lc.la(bs.ropeQ, R*p.NHead*p.NRot/2, qSrc, rcs, qOff, bs.qr)
							}
							if fusedKV {
							} else if !shared && p.RopeSplit {
								// XD-RoPE: k turns row-major, is normed after
								// the rotary (the rotation does not keep a head's
								// RMS), and is copied into the cache.
								kro := bs.zero
								if R > 1 {
									kro = bs.kroff
								}
								lc.la(bs.ropeKRow, R*p.NKVHead*p.NRot/2, kSrc, rcs, kro, bs.krot)
								kg, kw := headNormGeom(bs.kNormL, R*p.NKVHead)
								if err == nil {
									err = lc.launch(bs.kNorm, kg, kw, bs.krot, l.nK, bs.kn)
								}
								laKV(copyK, R*kvDim, bs.kn, kDst, kOff)
							} else if !shared {
								laKV(ropeK, R*kHeads*p.NRot/2, kSrc, rcs, kOff, kDst)
							}
						}
						if !shared && !fusedKV {
							laKV(copyV, vThreads, vSrc, vDst, vOff)
						}
					}
					// Where the score kernel reads q: the rotated buffer, or q itself on
					// an unrotated layer (bs.qr would hold the last block's).
					qAtt := bs.qr
					if !roped {
						qAtt = qSrc
						// Llama 4's temperature, on exactly these layers:
						// q times a factor of its position. bs.qr is free
						// here, since nothing rotated into it.
						if bs.qTemp != nil && !(rag != nil && g.rowsFaulted(rowsNotemp)) {
							lc.la(bs.qTemp, R*qdim, qSrc, bs.n, bs.tempTab, bs.qr)
							qAtt = bs.qr
						}
					}
					// Hunyuan's q norm runs after the rotary, weighted by w_q*w_k
					// (jlm.FlagQKNormPostRope), into bs.qn -- which the pre-rotary
					// norm left unwritten for q, so it reads one buffer and writes
					// another (RULE 13).
					if p.QKNormPost && bs.qNorm != nil && l.nQ != nil {
						qg, qw := headNormGeom(bs.qNormL, R*p.NHead)
						if err == nil {
							err = lc.launch(bs.qNorm, qg, qw, qAtt, l.nQ, bs.qn)
						}
						qAtt = bs.qn
					}
					// A row-chunked block's first phase ends with its chunk's
					// k and v in the cache: the attention reads every chunk's,
					// so it runs in the second phase.
					if p.NonCausal && g.ncPhase == 1 {
						continue
					}
					var fv *flashVariant
					var flash backend.Kernel
					if len(bs.flashV) > 0 {
						fv = bs.flashFor(nCap)
						flash = fv.k
						if p.Window(li) > 0 {
							flash = fv.kW
						}
					}
					// A windowed vision block masks its scores between the score
					// kernel and the softmax, so it takes the three-kernel path.
					useFlash := g.flashOn() && flash != nil && !skip["scores"] && !skip["softmax"] && !skip["acc"] &&
						!l.windowed
					if !skipAttn && l.qsa {
						if err == nil {
							err = g.qsaAttend(s, lc, bs, R, pk, g.kvp.layers[l.kvSrc], l, qAtt, bs.n, bs.xb)
							g.PagedLaunches++
						}
					} else if !skipAttn && l.msa {
						// Each kv group's kept blocks, gathered, then the decode
						// plan over them, a chunk's rows included (msa.go).
						if err == nil {
							err = g.msaAttend(s, lc, R, pk, g.kvp.layers[l.kvSrc], qAtt, bs.n, bs.xb)
							g.PagedLaunches++
						}
					} else if !skipAttn && pk != nil && pk.prefill {
						// A prefill chunk: one sequence's rows in order, a K
						// or V load serving a whole query tile.
						var sk backend.Buf
						if p.AttnSinks {
							if sk = l.sinks; sk == nil {
								sk = bs.noSink
							}
						}
						win := p.Window(li) > 0 && !g.pagedFaulted(pagedWindow)
						if err == nil {
							err = pk.pf.prefillAttn(s, lc, qAtt, kDst, vDst, bs.n, bs.xb, pgTab, win, sk)
							g.PagedPrefillLaunches++
							if pk.pf.flash != nil {
								g.FlashPrefills++ // the paged one-kernel form, on any backend
							}
							g.PagedPrefillPasses += pk.pf.passes(win)
							if pk.pf.on70() {
								g.PagedPrefill70++
							}
							if pk.pf.accOnMMA() {
								g.PagedAccMMA++
							}
							if pk.pf.on80() {
								g.FlashPrefill80++
							}
						}
					} else if !skipAttn && pk != nil {
						// A decode row or a batch's rows: each a descriptor, a
						// window its keyStart (pRowW), through the decode plan
						// chosen before the submission (pagedPlanFor).
						desc := pk.pRow
						win := p.Window(li) > 0 && !g.pagedFaulted(pagedWindow) && !(rag != nil && g.rowsFaulted(rowsNowin))
						if win {
							desc = pk.pRowW
						}
						// A ragged call with a run table: the decode plan over
						// the descriptors with the table's rows emptied, then
						// the table's prefill over those rows
						// (raggedprefill.go).
						ragOn := rag != nil && pk.ragRows > 0
						dec := desc
						if ragOn {
							dec = pk.pRowDec
							if win {
								dec = pk.pRowDecW
							}
						}
						var sk backend.Buf
						if p.AttnSinks {
							if sk = l.sinks; sk == nil {
								sk = bs.noSink
							}
						}
						if err == nil && !(ragOn && pk.ragAlone) {
							err = pk.attend(s, lc, g.kvp.layers[l.kvSrc], qAtt, kDst, vDst, bs.n, bs.xb, pgTab, dec, sk)
							g.PagedLaunches++
							if pk.cur.plan.Path == kernels.PathStaged {
								g.PagedStagedLaunches++
							}
						}
						if ragOn && err == nil {
							tabRag := desc
							if g.rowsFaulted(rowsRagtab) {
								// The violation: the table's rows read their
								// keys through the emptied set, so they attend
								// nothing of their own.
								tabRag = dec
							}
							if pk.ragStaged {
								err = pk.ragStagedAttend(lc, qAtt, kDst, vDst, bs.n, bs.xb, pgTab, tabRag, sk, win)
							} else {
								err = pk.ragAttend(lc, qAtt, kDst, vDst, bs.n, bs.xb, pgTab, tabRag, sk)
							}
							g.RagPrefills++
						}
					} else if !skipAttn && err == nil {
						err = g.contigAttend(lc, bs, l, li, R, kvp, qAtt, nCap, sm, smG, smW, skip, fv, flash, useFlash)
					}
					// The out-gate goes before wo: gating after the projection would scale
					// the residual contribution instead (as engine/model/forward.go notes).
					aSrc := bs.xb
					if bs.sigMul != nil {
						lc.la(bs.sigMul, R*qdim, bs.ogate, bs.xb, bs.ogated)
						aSrc = bs.ogated
					}
					// The attention output projection quantizing its own
					// operand (fuseActIn), with the residual add it already
					// carries.
					oIn := R == 1 && rag == nil && g.actInOn() && l.mvo.resIn != nil && !voltaOnly && l.clampB == nil && l.nSubAttn == nil &&
						!skip["quant"] && !skip["qkvo"] && l.nPostAttn == nil && !l.withAttn && !resScaled && !skip["add"] &&
						l.altRouter == nil && l.k3 == nil && l.q4e == nil && !(g.TableSplit && l.mvo.alt != nil) && l.mvo.k == qdim
					if oIn {
						g.ActInSaved++
					} else if l.nSubAttn != nil {
						// BitNet: the norm over the attention output and the
						// quantize wo reads, one launch a row.
						laRows(bs.subQA, aSrc, l.nSubAttn, bs.sub, bs.a, bs.ax)
						fsrc = bs.sub
					} else if voltaOnly {
						fsrc = aSrc
						g.QuantSkipped++
					} else if l.clampB != nil {
						lc.la(bs.clampK[6], R*qdim, aSrc, l.clampB, bs.cin)
						lc.la(bs.quantQ, kernels.QuantizeThreads(R*qPad(p, qdim)/32), bs.cin, bs.a, bs.ax)
						fsrc = bs.cin
					} else if !skip["quant"] {
						lc.la(bs.quantQ, kernels.QuantizeThreads(R*qPad(p, qdim)/32), aSrc, bs.a, bs.ax)
					}
					if oIn {
						mvIn(l.mvo, l.mvo.resIn, l.wo, bs.x2, aSrc, nil, bs.x)
						attnFused = true
					} else if !skip["qkvo"] {
						// x2 = x + Wo*attn in one launch where nothing sits
						// between the projection and the residual add.
						if !(l.nPostAttn == nil && !l.withAttn && !resScaled && !skip["add"] && l.altRouter == nil && l.k3 == nil && l.q4e == nil &&
							mvres(l.mvo, l.wo, bs.x2, bs.x)) {
							mvrun(l.mvo, l.wo, bs.mvOut)
						} else {
							attnFused = true
						}
					}
				}
			}

			// gemma2/gemma3's post-attention norm, before the residual add: it
			// normalises the attention output, not the stream. It writes bs.h rather
			// than in place (RULE 13); bs.h is free here, consumed by q/k/v and
			// rewritten by the FFN pre-norm.
			aout := bs.mvOut
			if l.withAttn {
				// attention + mixer; bs.h is free here (see below).
				lc.la(bs.addE, R*p.NEmbd, bs.mvOut, bs.side, bs.h)
				aout = bs.h
			}
			if l.clampB != nil && !attnFused {
				lc.la(bs.clampK[7], R*p.NEmbd, aout, l.clampB, bs.cout)
				aout = bs.cout
			}
			if l.nPostAttn != nil && !skip["norm"] {
				lc.la(bs.normPart, R*bs.parts, aout, bs.part)
				lc.la(bs.normApply, R*p.NEmbd, aout, l.nPostAttn, bs.part, bs.h)
				aout = bs.h
			}

			// A block that is its mixer alone ends here: the residual add, and
			// the sum moved home to bs.x, where the next block reads it.
			if l.noFFN {
				if !skip["add"] && !attnFused {
					addRes(bs.x, aout, bs.x2)
					lc.la(bs.copyRes, R*p.NEmbd, bs.x2, bs.x)
				}
				continue
			}

			// --- feed-forward
			quantDone := false
			// The operand prologue (fuseActIn): gate and up normalise and
			// quantize the stream themselves, and the down projection
			// quantizes act(gate)*up; the FFN norm and its quantize go. Only
			// on a plain gated FFN, where nothing else reads the normed row.
			ffnPlain := R == 1 && rag == nil && g.actInOn() &&
				!voltaOnly && !skipFFN && !skip["norm"] && !skip["quant"] && !skip["actmul"] && bs.normVar == nil &&
				l.nFFNB == nil && l.nFFN != nil && l.nResid == nil && l.router == nil && l.clampB == nil &&
				!l.nonCausal && !l.sparse && l.k3 == nil && l.q4e == nil && l.ds4 == nil && l.altRouter == nil && !p.Parallel &&
				!ungatedFFN(p) && !(g.TableSplit && (l.mvg.alt != nil || l.mvu.alt != nil))
			ffnIn := ffnPlain && l.mvg.kernIn != nil && (l.mvu.gatedIn != nil || l.mvu.kernIn != nil)
			// The FFN site's gate2in recipe (recipe.go): Gate2 building its
			// own operand from the residual the attention output's launch
			// wrote, so the norm launch goes. Only behind a fused residual:
			// otherwise the norm's launch is the add too, and stays.
			g2in := ffnPlain && !ffnIn && attnFused && g.ffnFormOf(l) == FFNGate2In && l.mvu.gate2In != nil &&
				g.gateUp2On(l, R, false, voltaOnly, false, skip["actmul"])
			// The FFN norm's fused form where the block runs binary16 GEMMs:
			// it writes the first FFN projection's operand too.
			var ffnNorm, ffnConv backend.Kernel
			if voltaOnly && l.nFFN != nil {
				first := l.mvg
				if first.kern == nil {
					first = l.mvu
				}
				ffnNorm, ffnConv = g.rmsF16For(bs, first, R, true)
			}
			switch {
			case l.q4e != nil:
				// Qwen3.8-Flash-Next: the output placed into every stream,
				// then the mixture's input is the mix over them (q4e.go).
				g.q4ePostAttn(lc, bs, l, R, aout)
				norm(g.q4eFFNIn(lc, bs, l, R, mvrunAct), l.nFFN, l.nFFNB)
			case l.k3 != nil:
				// Kimi-K3: every stream after the attention, then the FFN's
				// input is the mix over them (k3.go).
				g.k3PostAttn(lc, bs, l, R, aout)
				if !skip["norm"] {
					norm(g.k3FFNIn(lc, bs, l, R), l.nFFN, l.nFFNB)
				}
			case l.ds4 != nil:
				// The hyper-connection around the attention, then the
				// mixture's: its input normed into bs.h and quantized (ds4.go).
				g.ds4Post(lc, bs, R, bs.x, aout, bs.x2)
				g.ds4Pre(lc, bs, l, R, bs.x2, false, norm)
				quantDone = true
			case l.altRouter != nil:
				// LAuReL's branch joins the attention's residual add (altup.go),
				// and the FFN's norm takes the joined row.
				if !skip["add"] {
					lc.la(bs.laurelJoin, R*p.NEmbd, bs.x, aout, bs.altLaur, bs.x2)
				}
				if !skip["norm"] {
					norm(bs.x2, l.nFFN, l.nFFNB)
				}
			case p.Parallel:
				// The parallel residual (C6): the FFN reads the block input. bs.x is
				// still that input (attention wrote bs.x2); a block sharing the
				// attention's norm (phi-2) reads bs.h, which nothing rewrote.
				if !skip["add"] && !attnFused {
					addRes(bs.x, aout, bs.x2)
				}
				if l.nFFN != nil && !skip["norm"] {
					norm(bs.x, l.nFFN, l.nFFNB)
				}
			case !voltaOnly && !attnFused && !scaled && !skip["add"] && !skip["norm"] && !skip["quant"] && bs.addRmsQ != nil &&
				bs.normVar == nil && l.nFFNB == nil && aout != bs.h && l.nFFN != nil:
				// Not when aout is bs.h (a post-attention norm wrote it): the
				// quantize phase re-reads the input after the norm phase wrote h
				// (RULE 13). x2 = x + attention, h = RMSNorm(x2) and its quantized
				// form, one launch.
				laRows(bs.addRmsQ, bs.x, aout, l.nFFN, bs.x2, bs.h, bs.a, bs.ax)
				fsrc = bs.h // what la records for a quantize: float weights read it
				quantDone = true
			case voltaOnly && !attnFused && !scaled && !skip["add"] && !skip["norm"] && bs.addRms != nil &&
				bs.normVar == nil && l.nFFNB == nil && aout != bs.h && l.nFFN != nil && ffnNorm != nil:
				// The same launch, writing the gate's binary16 operand too.
				laRows(ffnNorm, bs.x, aout, l.nFFN, bs.x2, bs.h, g.f16Buf)
				actMemoK, actMemoSrc = ffnConv, bs.h
				g.NormF16++
			case !attnFused && !scaled && !skip["add"] && !skip["norm"] && bs.addRms != nil &&
				bs.normVar == nil && l.nFFNB == nil && aout != bs.h && l.nFFN != nil:
				// x2 = x + attention and h = RMSNorm(x2), one launch; not when aout
				// is bs.h either, for the reason above.
				laRows(bs.addRms, bs.x, aout, l.nFFN, bs.x2, bs.h)
			default:
				if !skip["add"] && !attnFused {
					addRes(bs.x, aout, bs.x2)
				}
				if ffnIn || g2in {
					g.ActInSaved++
					quantDone = true
				} else if !voltaOnly && !skip["norm"] && !skip["quant"] && bs.rmsQ != nil && bs.normVar == nil && l.nFFNB == nil &&
					l.nFFN != nil {
					laRows(bs.rmsQ, bs.x2, l.nFFN, bs.h, bs.a, bs.ax)
					fsrc = bs.h // what la records for a quantize: float weights read it
					quantDone = true
				} else if !skip["norm"] {
					norm(bs.x2, l.nFFN, l.nFFNB)
				}
			}
			// A post-norm block's residual is the norm of x + attention, the
			// row the FFN reads (nn.LayerPlan.PostResidNorm).
			if l.nResid != nil && !skip["norm"] {
				lc.la(bs.copyRows, R*p.NEmbd, bs.h, bs.x2)
			}
			if voltaOnly && !quantDone {
				fsrc = bs.h
				g.QuantSkipped++
			} else if !skip["quant"] && !quantDone {
				lc.la(bs.quantE, kernels.QuantizeThreads(R*ePad(p)/32), bs.h, bs.a, bs.ax)
			}
			// ffnOut is what the FFN produced: bs.mvOut for every model, and
			// the shared-expert sum where one exists.
			ffnOut := bs.mvOut
			if l.router != nil {
				ffnOut = mixFFN(l, li)
				if err != nil {
					return
				}
			} else if !skipFFN {
				// A non-causal block's own MLP, which may not be the set's
				// (layer.nffn): its width, gating and activation.
				ungated, nffn, actK := ungatedFFN(p), p.NFFN, bs.actMul
				if l.nonCausal {
					ungated, nffn = l.ungated, l.nffn
					if l.actK != nil {
						actK = l.actK
					}
				}
				gIn, uIn := bs.g, bs.u
				guDone, gu2 := false, false
				form := g.ffnGateUp(l, R, rag != nil, voltaOnly, ungated, resScaled, skip, g2in, ffnIn, nffn)
				l.ran.note(formGateUp, uint8(form))
				if form == gateUpClamped {
					// Gemma 4's clipped linears: gate and up each read their own
					// clamped copy of the normed rows, and the activation their
					// clamped outputs.
					for s, pr := range [2]struct {
						m       mv
						r       *resident
						out, cl backend.Buf
					}{{l.mvg, l.gate, bs.g, bs.gClamp}, {l.mvu, l.up, bs.u, bs.uClamp}} {
						lc.la(bs.clampK[8+2*s], R*p.NEmbd, bs.h, l.clampB, bs.cin)
						lc.la(bs.quantE, kernels.QuantizeThreads(R*ePad(p)/32), bs.cin, bs.a, bs.ax)
						fsrc = bs.cin
						mvrun(pr.m, pr.r, pr.out)
						lc.la(bs.clampK[9+2*s], R*p.NFFN, pr.out, l.clampB, pr.cl)
					}
					gIn, uIn = bs.gClamp, bs.uClamp
				} else if form == gateUpGate2In {
					// The norm, gate and up in one launch: up's weight, the
					// residual and the norm weight, the product's row, then the
					// gate's weight (Gate2 with ActIn's norm).
					wu, wg := l.up, l.gate
					if g.gateUp2Swapped() {
						wu, wg = wg, wu
					}
					lamv(l.mvu, l.mvu.gate2In, l.mvu.rows*l.mvu.split, wu.qs, wu.d, wu.sc, bs.x2, l.nFFN, bs.act,
						wg.qs, wg.d, wg.sc)
					g.Gate2InRuns++
					g.ActInLaunches++
					gu2 = true
				} else if form == gateUpIn {
					mvIn(l.mvg, l.mvg.kernIn, l.gate, bs.g, bs.x2, l.nFFN, nil)
				} else if form == gateUpGU {
					// Gate and up in one launch over one operand, and down
					// forming act(gate)*up as it quantizes, with the residual
					// add (fuseGateUp). Every row keeps its own reduction.
					if err == nil {
						err = lc.launch(l.gu.kern, l.gu.blocks, l.gu.w, bs.a, bs.ax,
							l.gate.qs, l.gate.d, l.gate.sc, bs.g, l.up.qs, l.up.d, l.up.sc, bs.u)
					}
					if g.gateUpSwapped() {
						mvIn(l.mvd, l.guDown, l.down, bs.x, bs.u, bs.g, bs.x2)
					} else {
						mvIn(l.mvd, l.guDown, l.down, bs.x, bs.g, bs.u, bs.x2)
					}
					g.GateUpFused++
					ffnFused = true
					guDone = true
				} else if gu2 = form == gateUpGate2; gu2 {
					// Gate and up in one launch: up's weight, the operand, the
					// product's row, then the gate's weight (Gate2).
					wu, wg := l.up, l.gate
					if g.gateUp2Swapped() {
						wu, wg = wg, wu
					}
					lamv(l.mvu, l.mvu.gate2, l.mvu.rows*l.mvu.split, wu.qs, wu.d, wu.sc, bs.a, bs.ax, bs.act,
						wg.qs, wg.d, wg.sc)
					g.GateUp2Runs++
				} else if form == gateUpTwo {
					mvrun(l.mvg, l.gate, bs.g)
					// Gemma 3n's sparse lead: the gate's gaussian top-k.
					gIn = g.emitGauss(lc, bs, l, R)
				}
				if !guDone {
					gated := gu2
					switch {
					case gu2:
					case ffnIn && l.mvu.gatedIn != nil:
						mvIn(l.mvu, l.mvu.gatedIn, l.up, bs.act, bs.x2, l.nFFN, gIn)
						gated = true
					case ffnIn:
						mvIn(l.mvu, l.mvu.kernIn, l.up, bs.u, bs.x2, l.nFFN, nil)
					default:
						// BitNet normalises the product before down, so up does not
						// form it in its own store.
						gated = !ungated && l.clampB == nil && l.nSubFFN == nil && !skip["actmul"] && mvgate(l.mvu, l.up, bs.act, gIn)
						if !gated && l.clampB == nil {
							mvrun(l.mvu, l.up, bs.u)
						}
					}
					// The down projection building its own operand (fuseActIn):
					// quantizing act(gate)*up where up wrote it (downIn), or
					// forming it from gate and up as it quantizes (downAct).
					down := g.ffnDown(l, R, rag != nil, voltaOnly, resScaled, skip, gated, ffnIn, nffn)
					l.ran.note(formDown, uint8(down))
					downIn, downAct := down == downFormIn, down == downFormAct
					// The activation writes the down projection's operand where it
					// can: binary16 for a binary16 GEMM, int8 for the rest.
					var actConv backend.Kernel
					quantDoneF := downAct
					if !skip["actmul"] && !gated && !ungated && l.clampB == nil && !downAct {
						if voltaOnly {
							if ak, conv, n := g.actF16For(bs, l.mvd, R); ak != nil {
								lc.la(ak, n, gIn, uIn, g.f16Buf)
								actConv = conv
								g.ActF16Fused++
							}
						} else if bs.quantAct != nil && !skip["quantf"] && !g.NoOperandFuse && l.nSubFFN == nil {
							lc.la(bs.quantAct, kernels.QuantizeThreads(R*p.NFFN/32), gIn, uIn, bs.act, bs.a, bs.ax)
							fsrc = bs.act
							quantDoneF = true
							g.QuantActs++
						}
					}
					if !skip["actmul"] && !gated && actConv == nil && !quantDoneF {
						if ungated && l.xielu != nil {
							// Apertus: the block's four numbers ride as a buffer.
							lc.la(actK, R*nffn, bs.u, l.xielu, bs.act)
						} else if ungated {
							// One operand: the gate does not exist.
							lc.la(actK, R*nffn, bs.u, bs.act)
						} else {
							lc.la(actK, R*nffn, gIn, uIn, bs.act)
						}
					}
					if voltaOnly {
						fsrc = bs.act
						g.QuantSkipped++
						if actConv != nil {
							actMemoK, actMemoSrc = actConv, bs.act
						}
					} else if l.clampB != nil && !quantDoneF {
						lc.la(bs.clampK[12], R*p.NFFN, bs.act, l.clampB, bs.cin)
						lc.la(bs.quantF, kernels.QuantizeThreads(R*p.NFFN/32), bs.cin, bs.a, bs.ax)
						fsrc = bs.cin
					}
					if downIn || downAct {
						g.ActInSaved++
					} else if l.nSubFFN != nil {
						// BitNet: the norm over act(gate)*up and the quantize down
						// reads, one launch a row.
						laRows(bs.subQF, bs.act, l.nSubFFN, bs.sub, bs.a, bs.ax)
						fsrc = bs.sub
					} else if !voltaOnly && l.clampB == nil && !skip["quantf"] && !quantDoneF {
						lc.la(bs.quantF, kernels.QuantizeThreads(R*nffn/32), bs.act, bs.a, bs.ax)
					}
					if downIn {
						mvIn(l.mvd, l.mvd.resIn, l.down, bs.x, bs.act, nil, bs.x2)
						ffnFused = true
					} else if downAct {
						// pIn is the gate's row and the norm-weight slot carries up's.
						mvIn(l.mvd, l.mvd.resActIn, l.down, bs.x, gIn, uIn, bs.x2)
						ffnFused = true
					} else if l.clampB == nil && l.nPostFFN == nil && !resScaled && !skip["add"] && l.k3 == nil && l.q4e == nil && mvres(l.mvd, l.down, bs.x, bs.x2) {
						ffnFused = true
					} else {
						mvrun(l.mvd, l.down, bs.mvOut)
						if l.clampB != nil {
							lc.la(bs.clampK[13], R*p.NEmbd, bs.mvOut, l.clampB, bs.cout)
							ffnOut = bs.cout
						}
					}
				}
			}
			// The post-FFN norm, same shape and the same reasoning: bs.h
			// held the FFN pre-norm and the gate/up matvecs have read it.
			fout := ffnOut
			if l.nPostFFN != nil && !skip["norm"] {
				lc.la(bs.normPart, R*bs.parts, ffnOut, bs.part)
				lc.la(bs.normApply, R*p.NEmbd, ffnOut, l.nPostFFN, bs.part, bs.h)
				fout = bs.h
			}
			if l.ds4 != nil {
				g.ds4Post(lc, bs, R, bs.x2, fout, bs.x)
			} else if l.q4e != nil {
				g.q4ePostFFN(lc, bs, R, fout)
			} else if l.k3 != nil {
				g.k3PostFFN(lc, bs, R, fout)
			} else if !skip["add"] && !ffnFused {
				if nk, conv := g.foldAddNorm(bs, l, li, hi, R, voltaOnly, scaled, persist, fout, skip); nk != nil {
					pendX, pendY, pendNorm, pendConv = bs.x2, fout, nk, conv
				} else {
					addRes(bs.x2, fout, bs.x)
				}
			}
			// BERT's output norm, over x + mlp, out of place and back.
			if l.nResid != nil && !skip["norm"] {
				norm(bs.x, l.nResid, l.nResidB)
				lc.la(bs.copyRows, R*p.NEmbd, bs.h, bs.x)
			}
			// Gemma 4's per-layer embedding: x += post_norm(W_out ·
			// (gelu(W_gate · x) * pl[li])), the sum through x2 and back.
			if l.altRouter != nil {
				// Gemma 3n: every stream corrected and the per-layer input
				// gated into the rest (altup.go).
				g.emitAltCorrect(lc, bs, l, R, mvrun, norm, &fsrc)
			} else if l.pleGate != nil && bs.pleIn != nil {
				perLayerEmbed(l, bs.x)
			}
			// Gemma 4's layer_scalar on the whole output row, out of place:
			// x times the block's scalar into h (dead past the add), and back.
			if l.outScale != nil && bs.scaleX != nil {
				lc.la(bs.scaleX, R*p.NEmbd, bs.x, l.outScale, bs.h)
				lc.la(bs.scaleX, R*p.NEmbd, bs.h, bs.one, bs.x)
			}
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
			lc.la(bs.quantE, kernels.QuantizeThreads(R*ePad(p)/32), bs.h, bs.a, bs.ax)
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
				err = lc.launch(bs.ragArgmax, ragHead, kernels.ArgmaxGroup, bs.ragOut(g.Config), bs.ragTok)
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
			lc.la(hb.quantE, kernels.QuantizeThreads(p.NEmbd/32), hb.h, hb.a, hb.ax)
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
			lc.la(bs.quantE, kernels.QuantizeThreads(R*ePad(p)/32), bs.h, bs.a, bs.ax)
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
				err = lc.launch(g.argmaxK, 1, kernels.ArgmaxGroup, bs.headOut(), g.argmaxOut)
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
		// scores never written (kernels.FlashPrefill70).
		fk := bs.fp70
		if p.Window(li) > 0 {
			fk = bs.fp70W
		}
		if err == nil {
			err = lc.launch(fk, kernels.FlashPrefill70Groups(kernels.FlashPrefill70Shape{Heads: p.NHead, Rows: R}),
				kernels.FlashPrefill70Threads, qAtt, kvp.kc, kvp.vc, bs.n, bs.xb)
		}
		g.FlashPrefills++
	} else if bs.fpT != nil && !skip["scores"] && !skip["softmax"] && !skip["acc"] && !l.windowed {
		// The same, on Metal's matrix unit (kernels.FlashPrefillTile).
		fk := bs.fpT
		if p.Window(li) > 0 {
			fk = bs.fpTW
		}
		if err == nil {
			err = lc.launch(fk, kernels.FlashTileGroups(p.NHead, R, bs.fpTile),
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
					lc.la(scores70, kernels.AttnScoresMMA70Warps(p.NHead, R, nCap, volta70AttnMT,
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
				lc.la(bs.acc70, kernels.AttnAccMMA70Warps(p.NHead, p.HeadDim, R,
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

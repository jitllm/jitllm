package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// deltaScanOf is the chunked delta rule's geometry for a linear block's plan:
// the key pairing, the decay's axis and the subgroup the device guarantees are
// the plan's and the device's, and nothing else about the scan varies.
func deltaScanOf(r nn.RecurrentPlan, rows, lanes int) kernels.DeltaScan {
	d := kernels.DeltaScan{VHeads: r.VHeads, VDim: r.VDim, KDim: r.KDim,
		Rep: r.VHeads / r.KHeads, PerChan: r.ChanDecay, Rows: rows, Lanes: lanes}
	if r.KeyTiled {
		d.TiledK = r.KHeads
	}
	if r.TailLen > 0 {
		d.Stride = r.StateLen
	}
	return d
}

// emitLinear is the gated delta rule's half of a block submission: everything
// between the pre-norm and bs.mvOut, which is exactly what the attention half
// spans.
//
// The order is engine/model/delta.go's, transcribed rather than re-derived. Where the
// host works in place this works out of place (ir.Validate), so q and k
// ping-pong between two buffers. recPair.cur is the half of the block's pool a
// submission reads; it writes the other, and the flip happens after the
// launches are issued so a captured graph replays the pointers it recorded.
// Every kernel that touches a state finds its sequence's slots in bs.dRows,
// the recurrent descriptor (recDesc). It takes the caller's la and
// mvrun so a linear block shares the launch accounting and batched-twin
// selection.
func (g *devTier) emitLinear(sid uint64, s backend.Session, bs *blockScratch, l *layer,
	li, R int, rag bool, p *nn.LayerPlan, errp *error,
	lc *launcher,
	mvrun func(mv, *resident, backend.Buf),
	mvrunAct func(mv, *resident, backend.Buf, backend.Buf, backend.Buf, backend.Buf)) {
	r := bs.rec
	if l.centred {
		g.emitCentred(bs, l, li, R, p, errp, lc, mvrun)
		return
	}
	// The real row count (bs.dRows) matters here where attention would mask:
	// a zero padding column shifted into the state stays there. layersOnce
	// stages it with the other per-call writes, outside emit, through the
	// session's Write (a Buf.Write would deadlock on the CUDA goroutine, and a
	// copy inside a recording invalidates the capture).
	par, has := g.stepPar(sid, l)
	pool := l.pool
	if !has || pool == nil {
		if *errp == nil {
			*errp = fmt.Errorf("tier: block %d has no recurrent state for session %d", li, sid)
		}
		return
	}
	inS, inC := pool.s[par], pool.c[par]
	outS, outC := pool.s[1-par], pool.c[1-par]
	if g.recShared {
		// One persistent state; the step writes the device's scratch.
		inS, outS = pool.s[0], g.recNext
	}
	seqs := 1
	qkDim := r.KHeads * r.KDim
	inner := r.VHeads * r.VDim

	// The pre-norm and its quantization, the same two the attention half runs,
	// over Kimi-K3's residual mix where the block has one.
	in := bs.x
	if l.k3 != nil {
		in = g.k3AttnIn(lc, bs, l, R)
	}
	if l.q4e != nil {
		conv := bs.q4e.ngConv
		if rag && l.q4e.ngram {
			if rk := bs.ragLin[len(g.rag.pos)]; rk != nil {
				conv = rk.ngConv
			}
		}
		in = g.q4eAttnIn(lc, bs, l, R, mvrunAct, inS, outS, conv)
	}
	if l.nAttn == nil {
		// No pre-norm (Qwen3.8-Flash-Next's mix is normed already).
		lc.la(bs.copyRows, R*p.NEmbd, in, bs.h)
	} else {
		lc.la(bs.normPart, R*bs.parts, in, bs.part)
		lc.la(bs.normApply, R*p.NEmbd, in, l.nAttn, bs.part, bs.h)
	}
	lc.la(bs.quantE, kernels.QuantizeThreads(R*ePad(p)/32), bs.h, bs.a, bs.ax)

	if r.SSD {
		g.emitSSD(sid, bs, l, li, R, rag, p, errp, lc, mvrun, pool, par, seqs)
		return
	}
	if r.ShortConv {
		g.emitShortConv(sid, bs, l, li, R, rag, p, errp, lc, mvrun, pool, par)
		return
	}
	if r.Mamba1 {
		g.emitMamba1(sid, bs, l, li, R, rag, p, errp, lc, mvrun, mvrunAct, pool, par, seqs)
		return
	}
	// The three projections. Wq carries the fused q|k|v of a linear block (the
	// layer kind decides the consumer; see nn.SSMWeights).
	mvrun(l.mvq, l.wq, bs.dMixed)
	if r.ChanDecay {
		// KDA's two gates are low-rank: two matvecs with a quantize between, the
		// second reading the rank-r temporary (hence mvrunAct). The output gate
		// goes first because both use dLowRank. Kimi-K3's output gate is one
		// full-rank matrix.
		if l.ssmGA == nil {
			mvrun(l.mvSG, l.ssmGate, bs.dZ)
		} else {
			mvrun(l.mvGA, l.ssmGA, bs.dLowRank)
			lc.la(bs.quantLR, kernels.QuantizeThreads(R*pad32(r.KDim)/32), bs.dLowRank, bs.dLRa, bs.dLRax)
			mvrunAct(l.mvGB, l.ssmGB, bs.dZ, bs.dLRa, bs.dLRax, bs.dLowRank)
		}
		mvrun(l.mvFA, l.ssmFA, bs.dLowRank)
		lc.la(bs.quantLR, kernels.QuantizeThreads(R*pad32(r.KDim)/32), bs.dLowRank, bs.dLRa, bs.dLRax)
		mvrunAct(l.mvFB, l.ssmFB, bs.dAlpha, bs.dLRa, bs.dLRax, bs.dLowRank)
		// ssmBA is b_proj alone here: beta's logits, one per value head, with
		// no alpha stream in it to deinterleave.
		mvrun(l.mvBA, l.ssmBA, bs.dBRaw)
	} else {
		mvrun(l.mvSG, l.ssmGate, bs.dZ)
		mvrun(l.mvBA, l.ssmBA, bs.dBA)
	}

	// The convolution and its shift are two kernels because they write two
	// buffers; folding the shift in would load and store the state (refused by
	// ir.Validate). A batched decode's rows are sequences, each with its own
	// window and state at its own seat (growSeat, recDesc), and every row
	// steps once.
	if rag {
		n, runs := len(g.rag.pos), len(g.rag.runs)
		seqs = n
		rk := bs.ragLin[n]
		if rk == nil {
			if *errp == nil {
				*errp = fmt.Errorf("tier: block %d: no batched linear step for %d sequences", li, n)
			}
			return
		}
		lc.la(rk.conv, n*r.Chans, inC, bs.dMixed, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(rk.shift, runs*(r.Conv-1)*r.Chans, inC, bs.dMixed, outC, bs.dRows)
		if r.ChanDecay {
			lc.la(rk.delta, runs*rk.perRun, inS, bs.dConv, bs.dAlpha, bs.dBRaw,
				l.ssmDt, l.ssmA, bs.dOut, outS, bs.dRows)
		} else {
			lc.la(rk.delta, runs*rk.perRun, inS, bs.dConv, bs.dBA,
				l.ssmDt, l.ssmA, bs.dOut, outS, bs.dRows)
		}
		g.LinearFused++
		if rk.lanes == 1 {
			g.LinearRowsScalar++
		}
		if g.rag.sid != nil {
			g.SessionLinear++
		}
	} else {
		lc.la(bs.convRows, R*r.Chans, inC, bs.dMixed, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(bs.convShift, (r.Conv-1)*r.Chans, inC, bs.dMixed, outC, bs.dRows)
	}
	// One launch from the convolution to the state where the device has a
	// subgroup (kernels.GatedDeltaFused): the SiLU, slices, L2 norms, scales and
	// gates are formed inside the scan, replacing eleven launches.
	if rag {
		// stepped above, one launch over every sequence
	} else if bs.deltaFused != nil {
		if r.ChanDecay {
			lc.la(bs.deltaFused, bs.scanThreads, inS, bs.dConv, bs.dAlpha, bs.dBRaw,
				l.ssmDt, l.ssmA, bs.dOut, outS, bs.dRows)
		} else {
			lc.la(bs.deltaFused, bs.scanThreads, inS, bs.dConv, bs.dBA,
				l.ssmDt, l.ssmA, bs.dOut, outS, bs.dRows)
		}
		g.LinearFused++
		if R > 1 {
			g.LinearBatched++
		}
	} else {
		lc.la(bs.ssmSiLU, R*r.Chans, bs.dConv, bs.dMixed)

		// The gates, from the projection the convolution did not touch.
		if r.ChanDecay {
			// Two launches at two widths: the decay over VHeads*KDim channels and
			// beta over VHeads heads, each discarding the other's half. The
			// arithmetic is qwen3next's unchanged -- the reference computes
			// exp(-exp(A_log)*softplus(g)) and ssm_a already holds -exp(A_log),
			// negated and broadcast per channel at conversion.
			if r.DecayBound != 0 {
				lc.la(bs.deltaGateChan, R*r.VHeads*r.KDim, bs.dAlpha, l.ssmDt, l.ssmA, bs.dDecay)
			} else {
				lc.la(bs.deltaGateChan, R*r.VHeads*r.KDim, bs.dAlpha, l.ssmDt, l.ssmA,
					bs.dAlpha, bs.dDecay, bs.dGateWaste)
			}
			lc.la(bs.deltaGate, R*r.VHeads, bs.dAlpha, l.ssmDt, l.ssmA, bs.dBRaw,
				bs.dGateWaste, bs.dBeta)
		} else {
			lc.la(bs.splitGates, R*r.VHeads, bs.dBA, bs.dBRaw, bs.dAlpha)
			lc.la(bs.deltaGate, R*r.VHeads, bs.dAlpha, l.ssmDt, l.ssmA, bs.dBRaw, bs.dDecay, bs.dBeta)
		}

		// q, k and v as three buffers, because a launch has no views.
		lc.la(bs.sliceQ, R*qkDim, bs.dMixed, bs.dQ)
		lc.la(bs.sliceK, R*qkDim, bs.dMixed, bs.dK)
		lc.la(bs.sliceV, R*inner, bs.dMixed, bs.dV)

		// The per-key-head L2 norm, then one scale each, then a second on q alone:
		// the host scales all 2*kHeads groups by 1/sqrt(kDim) and the query again,
		// so q ends at scale squared and k at scale.
		gw, gwid := headNormGeom(bs.qkNormL, R*r.KHeads)
		if *errp == nil {
			*errp = lc.launch(bs.qkNorm, gw, gwid, bs.dQ, l.ssmOnes, bs.dQn)
		}
		if *errp == nil {
			*errp = lc.launch(bs.qkNorm, gw, gwid, bs.dK, l.ssmOnes, bs.dKn)
		}
		lc.la(bs.qkScale, R*qkDim, bs.dKn, bs.dScale, bs.dK)
		lc.la(bs.qkScale, R*qkDim, bs.dQn, bs.dScale, bs.dQ)
		lc.la(bs.qkScale, R*qkDim, bs.dQ, bs.dScale, bs.dQn)

		// The rule, out of place: the state arrives through inS and leaves through
		// outS. The scan steps every one of the R real rows (bs.dRows) in order
		// inside one launch; a decode scratch on a device with no subgroup steps
		// its one row with GatedDeltaStep.
		if R > 1 || bs.deltaScan != nil {
			if bs.deltaScan == nil {
				if *errp == nil {
					*errp = fmt.Errorf("tier: block %d: %d rows and no chunked delta rule", li, R)
				}
				return
			}
			lc.la(bs.deltaScan, bs.scanThreads, inS, bs.dK, bs.dQn, bs.dV, bs.dDecay, bs.dBeta,
				bs.dOut, outS, bs.dRows)
			if R > 1 {
				g.LinearBatched++
			}
		} else {
			step := bs.deltaStep
			if r.ChanDecay {
				step = bs.deltaStepChan
			}
			lc.la(step, inner, inS, bs.dK, bs.dQn, bs.dV, bs.dDecay, bs.dBeta,
				bs.dOut, outS, bs.dRows)
		}
	}

	if g.recShared {
		// The new state home, before the next block reuses the scratch: each
		// sequence's row of recNext (its out slot) to its slot of the pool.
		// Compiled before the session (ensureRecNext): Compile inside one
		// deadlocks CUDA's owner goroutine.
		cp := g.recCopies[[2]int{r.StateLen, seqs}]
		if cp == nil && *errp == nil {
			*errp = fmt.Errorf("tier: block %d: no copy home for %d shared recurrent state(s)", li, seqs)
		}
		lc.la(cp, seqs*r.StateLen, g.recNext, pool.s[0], bs.dRows)
	}

	// The gated output norm: RMSNorm per value head against ssm_norm, times
	// SiLU of the z projection.
	vg, vwid := headNormGeom(bs.outNormL, R*r.VHeads)
	gated := bs.dAct
	noGate := r.ChanDecay && l.k3 != nil && l.k3.fault == nn.K3FaultNoKDAGate
	fused := !noGate && bs.outNormGate != nil && g.normGateOn(l, R == 1)
	if fused {
		// graph.NormGate: the norm's epilogue gates it, and the normalised
		// row is never written.
		in, z := bs.dOut, bs.dZ
		if g.normGateBroken() {
			in, z = z, in
		}
		if *errp == nil {
			*errp = lc.launch(bs.outNormGate, vg, vwid, in, l.ssmNorm, z, bs.dAct)
		}
		g.NormGateRuns++
	} else if *errp == nil {
		*errp = lc.launch(bs.outNorm, vg, vwid, bs.dOut, l.ssmNorm, bs.dOutN)
	}
	switch {
	case fused:
	case noGate:
		gated = bs.dOutN
	case r.ChanDecay || r.SigmoidGate:
		// KDA's gate and Qwen3.8-Flash-Next's are sigmoid(z).
		lc.la(bs.outSigMul, R*inner, bs.dZ, bs.dOutN, bs.dAct)
	default:
		lc.la(bs.outActMul, R*inner, bs.dZ, bs.dOutN, bs.dAct)
	}
	lc.la(bs.quantD, kernels.QuantizeThreads(R*pad32(inner)/32), gated, bs.a, bs.ax)
	mvrun(l.mvSO, l.ssmOut, bs.mvOut)

	// The flip is after the launches, so a captured graph replays the pointers
	// it recorded. It happens only while the submission is still good: la
	// latches the first error and skips later launches, so a block past a
	// failure must not flip, or the next token reads a half nothing wrote. This
	// does not make a failed submission resumable (blocks before the failure did
	// advance, which is why model.State.forward refuses a host restart).
	if errp == nil || *errp == nil {
		g.stepFlip(sid, l)
		// graphKey reads this block's half (rangeParity), so the next token
		// replays the other recording; with a constant parity one recording
		// would serve every token and the recurrence would freeze.
		g.recSteps++
	}
}

// emitSSD is a Mamba-2 block's half of the submission, emitLinear's twin from
// the projections on: engine/model/delta.go's ssdAttn, transcribed. Three
// projections (C|B|x, z and dt), the convolution and its shift as the delta
// rule runs them, one launch for the whole update -- the bias and SiLU, dt and
// the decay, the state over every row (or every run of a ragged step) and the
// D skip (kernels.GatedDeltaFused with SSD) -- then y*silu(z), its grouped
// norm, and ssm_out into bs.mvOut. The pre-norm and its quantization ran in
// emitLinear; the state is out of place exactly as there.
func (g *devTier) emitSSD(sid uint64, bs *blockScratch, l *layer, li, R int, rag bool, p *nn.LayerPlan,
	errp *error, lc *launcher, mvrun func(mv, *resident, backend.Buf), pool *recPool, par int, seqs int) {
	r := bs.rec
	inS, inC := pool.s[par], pool.c[par]
	outS, outC := pool.s[1-par], pool.c[1-par]
	if g.recShared {
		inS, outS = pool.s[0], g.recNext
	}
	inner := r.VHeads * r.VDim
	// A block that attends too keeps its mixer's projection apart from the
	// attention's query, and leaves its output in bs.side for the sum.
	in, inW, out := l.mvq, l.wq, bs.mvOut
	if l.withAttn {
		in, inW, out = l.mvSI, l.ssmIn, bs.side
	}
	mvrun(in, inW, bs.dMixed)
	mvrun(l.mvSG, l.ssmGate, bs.dZ)
	mvrun(l.mvBA, l.ssmBA, bs.dBRaw)
	if rag {
		n, runs := len(g.rag.pos), len(g.rag.runs)
		seqs = n
		rk := bs.ragLin[n]
		if rk == nil {
			if *errp == nil {
				*errp = fmt.Errorf("tier: block %d: no batched Mamba-2 step for %d sequences", li, n)
			}
			return
		}
		lc.la(rk.conv, n*r.Chans, inC, bs.dMixed, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(rk.shift, runs*(r.Conv-1)*r.Chans, inC, bs.dMixed, outC, bs.dRows)
		lc.la(rk.delta, runs*rk.perRun, inS, bs.dConv, bs.dBRaw, l.ssmDt, l.ssmA, l.ssmCB, l.ssmD,
			bs.dOut, outS, bs.dRows)
		g.LinearFused++
		if rk.lanes == 1 {
			g.LinearRowsScalar++
		}
		if g.rag.sid != nil {
			g.SessionLinear++
		}
	} else {
		lc.la(bs.convRows, R*r.Chans, inC, bs.dMixed, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(bs.convShift, (r.Conv-1)*r.Chans, inC, bs.dMixed, outC, bs.dRows)
		lc.la(bs.ssdScan, bs.scanThreads, inS, bs.dConv, bs.dBRaw, l.ssmDt, l.ssmA, l.ssmCB, l.ssmD,
			bs.dOut, outS, bs.dRows)
		g.LinearFused++
		if R > 1 {
			g.LinearBatched++
		}
	}
	if g.recShared {
		cp := g.recCopies[[2]int{r.StateLen, seqs}]
		if cp == nil && *errp == nil {
			*errp = fmt.Errorf("tier: block %d: no copy home for %d shared recurrent state(s)", li, seqs)
		}
		lc.la(cp, seqs*r.StateLen, g.recNext, pool.s[0], bs.dRows)
	}
	// The gated norm: y*silu(z) then the grouped RMSNorm, Mamba's order; or
	// norm(y)*silu(z) where the plan says so; or the gate alone with no norm.
	ng, nwid := headNormGeom(bs.ssdNormL, R*r.NormGroups)
	act := bs.dAct
	switch {
	case r.NoNorm:
		lc.la(bs.outActMul, R*inner, bs.dZ, bs.dOut, bs.dAct)
	case r.NormBeforeGate:
		if *errp == nil {
			*errp = lc.launch(bs.ssdNorm, ng, nwid, bs.dOut, l.ssmNorm, bs.dOutN)
		}
		lc.la(bs.outActMul, R*inner, bs.dZ, bs.dOutN, bs.dAct)
	default:
		lc.la(bs.outActMul, R*inner, bs.dZ, bs.dOut, bs.dOutN)
		if *errp == nil {
			*errp = lc.launch(bs.ssdNorm, ng, nwid, bs.dOutN, l.ssmNorm, bs.dAct)
		}
	}
	lc.la(bs.quantD, kernels.QuantizeThreads(R*pad32(inner)/32), act, bs.a, bs.ax)
	mvrun(l.mvSO, l.ssmOut, out)
	if errp == nil || *errp == nil {
		g.stepFlip(sid, l)
		g.recSteps++
	}
}

// emitCentred issues one LFM2 encoder convolution block (d1-omni's trunk):
// the pre-norm and its quantization, x, B and C, B*x through the centred
// convolution over the call's rows -- each tap inside the row's window,
// bs.vwinAll -- C times that, and out_proj into bs.mvOut. No state: the call
// is the whole sequence.
func (g *devTier) emitCentred(bs *blockScratch, l *layer, li, R int, p *nn.LayerPlan, errp *error,
	lc *launcher, mvrun func(mv, *resident, backend.Buf)) {
	r := bs.rec
	// The windows cover this call's rows: a windowed block's submission is
	// refused otherwise (layersOnce), and the padded rows past them read the
	// whole call's window, whose output nothing copies out.
	if bs.convCentred == nil || bs.vwinAll == nil {
		if *errp == nil {
			*errp = fmt.Errorf("tier: block %d convolves inside row windows and the device holds none", li)
		}
		return
	}
	lc.la(bs.normPart, R*bs.parts, bs.x, bs.part)
	lc.la(bs.normApply, R*p.NEmbd, bs.x, l.nAttn, bs.part, bs.h)
	lc.la(bs.quantE, kernels.QuantizeThreads(R*ePad(p)/32), bs.h, bs.a, bs.ax)
	mvrun(l.mvq, l.wq, bs.dMixed)
	mvrun(l.mvSG, l.ssmGate, bs.scB)
	mvrun(l.mvBA, l.ssmBA, bs.scC)
	lc.la(bs.scMul, R*r.Chans, bs.scB, bs.dMixed, bs.scBx)
	lc.la(bs.convCentred, R*r.Chans, bs.scBx, l.ssmConv1d, bs.vwinAll, bs.dConv)
	lc.la(bs.scMul, R*r.Chans, bs.scC, bs.dConv, bs.scY)
	lc.la(bs.quantE, kernels.QuantizeThreads(R*ePad(p)/32), bs.scY, bs.a, bs.ax)
	mvrun(l.mvSO, l.ssmOut, bs.mvOut)
}

// emitShortConv issues one LFM2 short-convolution block, from the quantized
// pre-norm to bs.mvOut: x, B and C, B*x through the convolution (its window is
// the block's whole recurrent state, in pool.c), C times that, out_proj. The
// two products are kernels.ActMul with the identity.
func (g *devTier) emitShortConv(sid uint64, bs *blockScratch, l *layer, li, R int, rag bool, p *nn.LayerPlan,
	errp *error, lc *launcher, mvrun func(mv, *resident, backend.Buf), pool *recPool, par int) {
	r := bs.rec
	inC, outC := pool.c[par], pool.c[1-par]
	mvrun(l.mvq, l.wq, bs.dMixed)
	mvrun(l.mvSG, l.ssmGate, bs.scB)
	mvrun(l.mvBA, l.ssmBA, bs.scC)
	lc.la(bs.scMul, R*r.Chans, bs.scB, bs.dMixed, bs.scBx)
	if rag {
		n, runs := len(g.rag.pos), len(g.rag.runs)
		rk := bs.ragLin[n]
		if rk == nil {
			if *errp == nil {
				*errp = fmt.Errorf("tier: block %d: no batched short convolution for %d sequences", li, n)
			}
			return
		}
		lc.la(rk.conv, n*r.Chans, inC, bs.scBx, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(rk.shift, runs*(r.Conv-1)*r.Chans, inC, bs.scBx, outC, bs.dRows)
		if g.rag.sid != nil {
			g.SessionLinear++
		}
	} else {
		lc.la(bs.convRows, R*r.Chans, inC, bs.scBx, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(bs.convShift, (r.Conv-1)*r.Chans, inC, bs.scBx, outC, bs.dRows)
		if R > 1 {
			g.LinearBatched++
		}
	}
	lc.la(bs.scMul, R*r.Chans, bs.scC, bs.dConv, bs.scY)
	lc.la(bs.quantE, kernels.QuantizeThreads(R*ePad(p)/32), bs.scY, bs.a, bs.ax)
	mvrun(l.mvSO, l.ssmOut, bs.mvOut)
	if errp == nil || *errp == nil {
		g.stepFlip(sid, l)
		g.recSteps++
	}
}

// emitMamba1 is a Mamba-1 block's half of the submission, engine/model/delta.go's
// mamba1Attn transcribed: x and z, the convolution and its shift as the delta
// rule runs them, x's bias and SiLU, x_proj's three outputs and their norms,
// dt_proj through a quantized bottleneck, the whole scan in one launch
// (kernels.GatedDeltaFused with Mamba1, every run of a ragged step), then
// y*silu(z) and ssm_out into bs.mvOut. The pre-norm and its quantization ran
// in emitLinear; the state is out of place exactly as there. Its four
// matrices ride KDA's slots (nn.SSMWeights): x_proj's dt rows in FA, dt_proj
// in FB, its B and C rows in GA and GB.
func (g *devTier) emitMamba1(sid uint64, bs *blockScratch, l *layer, li, R int, rag bool, p *nn.LayerPlan,
	errp *error, lc *launcher, mvrun func(mv, *resident, backend.Buf),
	mvrunAct func(mv, *resident, backend.Buf, backend.Buf, backend.Buf, backend.Buf),
	pool *recPool, par int, seqs int) {
	r := bs.rec
	inS, inC := pool.s[par], pool.c[par]
	outS, outC := pool.s[1-par], pool.c[1-par]
	if g.recShared {
		inS, outS = pool.s[0], g.recNext
	}
	inner := r.VHeads * r.VDim
	mvrun(l.mvq, l.wq, bs.dMixed)
	mvrun(l.mvSG, l.ssmGate, bs.dZ)
	var rk *ragLinear
	runs := 0
	if rag {
		n := len(g.rag.pos)
		runs, seqs = len(g.rag.runs), n
		if rk = bs.ragLin[n]; rk == nil {
			if *errp == nil {
				*errp = fmt.Errorf("tier: block %d: no batched Mamba-1 step for %d sequences", li, n)
			}
			return
		}
		lc.la(rk.conv, n*r.Chans, inC, bs.dMixed, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(rk.shift, runs*(r.Conv-1)*r.Chans, inC, bs.dMixed, outC, bs.dRows)
	} else {
		lc.la(bs.convRows, R*r.Chans, inC, bs.dMixed, l.ssmConv1d, bs.dConv, bs.dRows)
		lc.la(bs.convShift, (r.Conv-1)*r.Chans, inC, bs.dMixed, outC, bs.dRows)
	}
	// x through its bias and SiLU, into dMixed (dead past the convolution),
	// and quantized for x_proj's three matvecs.
	lc.la(bs.m1Bias, R*r.Chans, bs.dConv, l.ssmCB, bs.dMixed)
	lc.la(bs.quantD, kernels.QuantizeThreads(R*pad32(inner)/32), bs.dMixed, bs.a, bs.ax)
	mvrun(l.mvFA, l.ssmFA, bs.dLowRank)
	mvrun(l.mvGA, l.ssmGA, bs.m1B)
	mvrun(l.mvGB, l.ssmGB, bs.m1C)
	low, bv, cv := bs.dLowRank, bs.m1B, bs.m1C
	if l.m1DtN != nil {
		// Jamba's three RMSNorms (FalconMamba's of ones), out of place.
		dg, dw := headNormGeom(bs.m1NormDtL, R)
		if *errp == nil {
			*errp = lc.launch(bs.m1NormDt, dg, dw, bs.dLowRank, l.m1DtN, bs.m1LowN)
		}
		cg, cw := headNormGeom(bs.m1NormBCL, R)
		if *errp == nil {
			*errp = lc.launch(bs.m1NormBC, cg, cw, bs.m1B, l.m1BN, bs.m1BN)
		}
		if *errp == nil {
			*errp = lc.launch(bs.m1NormBC, cg, cw, bs.m1C, l.m1CN, bs.m1CN)
		}
		low, bv, cv = bs.m1LowN, bs.m1BN, bs.m1CN
	}
	// dt_proj reads the bottleneck's own quantization.
	lc.la(bs.m1Quant, kernels.QuantizeThreads(R*pad32(r.Rank)/32), low, bs.dLRa, bs.dLRax)
	mvrunAct(l.mvFB, l.ssmFB, bs.dBRaw, bs.dLRa, bs.dLRax, low)
	if rag {
		lc.la(rk.delta, runs*rk.perRun, inS, bs.dMixed, bv, cv, bs.dBRaw, l.ssmDt, l.ssmA, l.ssmD,
			bs.dOut, outS, bs.dRows)
		g.LinearFused++
		if rk.lanes == 1 {
			g.LinearRowsScalar++
		}
		if g.rag.sid != nil {
			g.SessionLinear++
		}
	} else {
		lc.la(bs.m1Scan, bs.scanThreads, inS, bs.dMixed, bv, cv, bs.dBRaw, l.ssmDt, l.ssmA, l.ssmD,
			bs.dOut, outS, bs.dRows)
		g.LinearFused++
		if R > 1 {
			g.LinearBatched++
		}
	}
	if g.recShared {
		cp := g.recCopies[[2]int{r.StateLen, seqs}]
		if cp == nil && *errp == nil {
			*errp = fmt.Errorf("tier: block %d: no copy home for %d shared recurrent state(s)", li, seqs)
		}
		lc.la(cp, seqs*r.StateLen, g.recNext, pool.s[0], bs.dRows)
	}
	lc.la(bs.outActMul, R*inner, bs.dZ, bs.dOut, bs.dAct)
	lc.la(bs.quantD, kernels.QuantizeThreads(R*pad32(inner)/32), bs.dAct, bs.a, bs.ax)
	mvrun(l.mvSO, l.ssmOut, bs.mvOut)
	if errp == nil || *errp == nil {
		g.stepFlip(sid, l)
		g.recSteps++
	}
}

package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/backend"
)

// deltaUnfused is the delta rule's state update as separate launches, where
// the device has no subgroup for mainkernels.GatedDeltaFused: the SiLU, the
// gates, q, k and v sliced, normed and scaled, then the scan or one row's
// step. The block encoder and the device program's scan both run it.
func (g *devTier) deltaUnfused(lc *launcher, bs *blockScratch, l *layer, li, R int, inS, outS backend.Buf, errp *error) {
	r := bs.rec
	qkDim := r.KHeads * r.KDim
	inner := r.VHeads * r.VDim
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

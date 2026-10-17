package tier

import (
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// fuseActIn builds block l's operand-prologue kernels (kernels.ActIn): q/k/v
// normalising and quantizing the residual themselves, gate and up the FFN's
// normed row, the attention output and FFN down projections quantizing their
// float input. Each takes the place of a RMSNormQuantRows or Quantize launch
// on a one-token decode, and the operand never reaches global memory.
//
// Built beside the kernels they shadow, after fuseQKV and after every
// retune, so the split each one carries is the shipping one. The launch
// decides per block whether the block's graph lets it run (actInBlock).
func (g *devTier) fuseActIn(l *layer) {
	l.qkvIn = nil
	for _, m := range []*mv{&l.mvq, &l.mvk, &l.mvv, &l.mvo, &l.mvg, &l.mvu, &l.mvd} {
		m.resIn, m.gatedIn, m.kernIn, m.resActIn = nil, nil, nil, nil
	}
	l.mvu.gate2In = nil
	if g.bs == nil {
		return
	}
	p := &g.bs.p
	// The prologue meets its window and sums by subgroup shuffles.
	// A window wider than one group (a per-token model's, kernels.WideWindow)
	// is not one the shuffles meet.
	if lnBlock(p) || l.mla || l.linear || l.nonCausal || g.pickLanes() != ir.SubgroupLanes || kernels.WideWindow(p.ActWin) {
		return
	}
	win := max(p.ActWin, 32)
	norm := kernels.ActIn{On: true, Norm: true, Eps: float32(p.RMSEps), Warp: g.warpNorm(), Window: win}
	quant := kernels.ActIn{On: true, Window: win}
	pf := g.ActInPrefetch
	switch {
	case pf == 0:
		// Measured on a V100 (Llama-3.2-1B): 4 to 8 words beat none and
		// every word; all of them thrash L2 behind the prologue.
		pf = 8
	case pf < 0:
		pf = 0
	}
	norm.Prefetch, quant.Prefetch = pf, pf
	// The FFN site's gate2in recipe (recipe.go): Gate2 with the FFN norm's
	// prologue, the norm, gate and up in one launch. Built wherever Gate2 is,
	// whatever the verdict, so a forced recipe and the trial both find it.
	if mu := l.mvu; mu.gate2 != nil && l.nFFN != nil && !ungatedFFN(p) {
		s := kernels.MatVecShape{Center: g.center, T: mu.q, K: mu.k, Rows: mu.rows, Split: mu.split,
			GroupSplit: mu.group, Rowt: 1, Gate2: true, GateAct: p.Act, In: norm}
		l.mvu.gate2In = g.inKernel(s)
	}
	// A site runs the prologue where the caller forced it (ActInSites) or
	// where tuneActIn measured it faster than the two launches on this card;
	// actInTrial builds every candidate for that measurement.
	sites := g.ActInSites
	on := func(site int, key string) bool {
		if sites != 0 {
			return sites&site != 0
		}
		return site&actInMeasured != 0 && (g.actInTrial || g.actInWin[key])
	}
	plain := func(m mv) bool {
		return m.kern != nil && m.slots == 0 && m.groups == 0 && m.threads == 0 && m.act == nil
	}
	shape := func(m mv, in kernels.ActIn) kernels.MatVecShape {
		return kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, Split: m.split,
			GroupSplit: m.group, Rowt: max(m.rowt, 1), Bias: m.bias != nil, In: in}
	}
	l.actInKeys = [4]string{}
	// A graphed block offers a site only where its rewrite matched, is
	// legal and pays at this device's Target (graph.ActInQKV, ActInO,
	// ActInDown); its key carries the Target's. A block without a graph
	// takes the structural checks below alone.
	sfx := [3]string{}
	offered := [3]bool{true, true, true}
	for i, rw := range []*graph.Rewrite{graph.ActInQKV, graph.ActInO, graph.ActInDown} {
		graphed, tkey, why := g.offer(l, rw.Name, graph.RowsDecode)
		l.actInWhy[i] = why
		offered[i], sfx[i] = !graphed || why == nil, targetSuffix(tkey)
	}
	if l.qkv != nil && offered[0] {
		var segs []kernels.MatVecShape
		for _, m := range []mv{l.mvq, l.mvk, l.mvv} {
			segs = append(segs, kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, Bias: m.bias != nil})
		}
		key := fmt.Sprintf("qkv/%v/%d/%v", segs, l.mvq.split, l.mvq.group) + sfx[0]
		if on(actInQKV, key) {
			// A key only beside a kernel: tuneActIn times every keyed site's
			// prologue, and a declined one (ErrGroupTooWide) has none.
			if l.qkvIn = g.segInKernel(segs, l.mvq.split, l.mvq.group, norm); l.qkvIn != nil {
				l.actInKeys[0] = key
			}
		}
	}
	// q, k and v one launch each, where they are not one launch together.
	if l.qkv == nil && offered[0] && plain(l.mvq) && plain(l.mvk) && plain(l.mvv) {
		key := fmt.Sprintf("qkvs/%v/%v/%v", shape(l.mvq, norm), shape(l.mvk, norm), shape(l.mvv, norm)) + sfx[0]
		if on(actInQKV, key) {
			l.mvq.kernIn, l.mvk.kernIn, l.mvv.kernIn = g.inKernel(shape(l.mvq, norm)), g.inKernel(shape(l.mvk, norm)),
				g.inKernel(shape(l.mvv, norm))
			l.actInKeys[0] = key
			if l.mvq.kernIn == nil || l.mvk.kernIn == nil || l.mvv.kernIn == nil {
				l.mvq.kernIn, l.mvk.kernIn, l.mvv.kernIn, l.actInKeys[0] = nil, nil, nil, ""
			}
		}
	}
	res := func(m mv, in kernels.ActIn) (kernels.MatVecShape, bool) {
		if !plain(m) || m.res == nil || m.bias != nil {
			return kernels.MatVecShape{}, false
		}
		s := shape(m, in)
		s.Bias = true
		return s, true
	}
	if s, ok := res(l.mvo, quant); ok && offered[1] {
		if key := fmt.Sprintf("o/%v", s) + sfx[1]; on(actInO, key) {
			if l.mvo.resIn = g.inKernel(s); l.mvo.resIn != nil {
				l.actInKeys[1] = key
			}
		}
	}
	if s, ok := res(l.mvd, quant); ok && l.mvu.gated != nil && offered[2] {
		if key := fmt.Sprintf("d/%v", s) + sfx[2]; on(actInDown, key) {
			if l.mvd.resIn = g.inKernel(s); l.mvd.resIn != nil {
				l.actInKeys[2] = key
			}
		}
	}
	if sites&actInGateUp == 0 || ungatedFFN(p) || !plain(l.mvg) || !plain(l.mvu) || l.mvg.bias != nil || l.mvu.bias != nil {
		return
	}
	l.mvg.kernIn = g.inKernel(shape(l.mvg, norm))
	if l.mvu.gated != nil {
		s := shape(l.mvu, norm)
		s.Rowt, s.Gate, s.GateAct = 1, true, p.Act
		l.mvu.gatedIn = g.inKernel(s)
	}
	if l.mvu.gatedIn == nil && sites&actInDown != 0 {
		// No gated up: up writes its row, and down forms act(gate)*up as it
		// quantizes (QuantizeAct's arithmetic).
		l.mvu.kernIn = g.inKernel(shape(l.mvu, norm))
		if s, ok := res(l.mvd, kernels.ActIn{On: true, Gated: true, Act: p.Act, Window: win, Prefetch: pf}); ok {
			l.mvd.resActIn = g.inKernel(s)
		}
	}
	if l.mvg.kernIn == nil || (l.mvu.gatedIn == nil && l.mvu.kernIn == nil) {
		l.mvg.kernIn, l.mvu.gatedIn, l.mvu.kernIn = nil, nil, nil
	}
}

// The operand-prologue sites (Config.ActInSites).
const (
	actInQKV = 1 << iota
	actInO
	actInGateUp
	actInDown
	// actInMeasured is the sites tuneActIn decides; gate and up only run
	// the prologue where the caller forces them (two launches replace one
	// norm, which lost on every card measured).
	actInMeasured = actInQKV | actInO | actInDown
)

// inKernel is an ActIn matvec from the device-wide cache (idKerns, keyed by
// the whole shape), nil where the kernels refuse the shape or it does not
// compile -- recorded in LastErr, and the block keeps its two launches.
func (g *devTier) inKernel(s kernels.MatVecShape) backend.Kernel {
	k, err := g.idKernel(s)
	if err != nil {
		g.actInDeclined(err)
		return nil
	}
	return k
}

func (g *devTier) segInKernel(segs []kernels.MatVecShape, split int, grp bool, in kernels.ActIn) backend.Kernel {
	key := fmt.Sprintf("in/%v/%d/%v/%v", segs, split, grp, in)
	if k, ok := g.segKerns[key]; ok {
		return k
	}
	ker, err := kernels.MatVecSegmentsIn(segs, split, grp, in)
	var k backend.Kernel
	if err == nil {
		k, err = g.dev.Compile(ker)
	}
	if err != nil {
		g.actInDeclined(err)
		k = nil
	}
	if g.segKerns == nil {
		g.segKerns = map[string]backend.Kernel{}
	}
	g.segKerns[key] = k
	return k
}

// actInOn is whether this submission may take the operand prologue at all.
func (g *devTier) actInOn() bool { return !g.NoFuse && !g.NoActFuse }

// actInDeclined records why a prologue kernel was not built. A width the
// kernel's registers cannot launch at (backend.ErrGroupTooWide) is a shape
// this device takes in two launches, not an error.
func (g *devTier) actInDeclined(err error) {
	if errors.Is(err, backend.ErrGroupTooWide) {
		g.ActInDeclined++
		return
	}
	g.LastErr = err.Error()
}

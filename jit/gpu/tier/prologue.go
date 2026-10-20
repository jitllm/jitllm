package tier

import "fmt"

// The device program's prologue: what a submission builds before it runs,
// under g.mu and outside the submission -- each block's fused variant for the
// step's row class, its kernels generated from its ops, and the tables they
// read (headPerm). The step itself only looks them up.

// fusedFor is the block's fused variant for a step of R rows: the rewrites
// legal at one row, on a prompt chunk and on a ragged step differ, so each
// class keeps its own.
func (l *layer) fusedFor(R int, rag *ragStep) *progFused {
	switch {
	case rag != nil && ragFew(len(rag.pos)):
		return &l.progRagFew[len(rag.pos)-2]
	case rag != nil:
		return &l.progRag
	case R > 1:
		return &l.progRows
	}
	return &l.progOne
}

// ensureFusedLocked builds block l's variant of program prog for a step of R
// rows over scratch bs -- its rewrites, then every op's kernels generated
// from the op -- where it is not built for bs already. Callers hold g.mu.
func (g *devTier) ensureFusedLocked(l *layer, prog *devProgram, bs *blockScratch, R int, rag *ragStep) {
	fc := l.fusedFor(R, rag)
	if fc.bs == bs && fc.fault == faultSig(l, &bs.p) {
		return
	}
	nseq := 0
	if rag != nil {
		nseq = len(rag.pos)
	}
	fc.prog, fc.took, fc.err = prog.fused(l, bs, g, R, nseq)
	if fc.prog == prog {
		fc.prog = prog.clone() // a refused fusion hands back the base program, which the rewrite must not change
	}
	faultRewrite(fc.prog, l, &bs.p)
	if g.kb.mlaFault == MLAFaultSwapBanks {
		swapLatentBanks(fc.prog)
	}
	fc.fault = faultSig(l, &bs.p)
	// The values the builder left unslotted, coloured by liveness now the
	// rewrites are done, and their buffers grown to this step's rows.
	// Inside a submission they are never grown: a launch already recorded
	// may read the buffers a growth would free.
	fc.valErr = nil
	fc.prog.floatSources()
	need, err := colourVals(fc.prog)
	switch {
	case err != nil:
		fc.valErr = err
	case g.genInSub && !g.valsFit(need, max(R, bs.rows)):
		fc.valErr = fmt.Errorf("tier: a block's program values outgrow their buffers inside a submission")
	case !g.genInSub && !g.sizeVals(need, max(R, bs.rows)):
		fc.valErr = fmt.Errorf("tier: a block's program values do not fit")
	}
	// The int8 operand: as wide as the widest row the program quantizes.
	if a, ax := operandNeed(fc.prog, max(R, bs.rows)); fc.valErr == nil && (a > bs.aCap || ax > bs.axCap) {
		switch {
		case g.genInSub:
			fc.valErr = fmt.Errorf("tier: a block's program operand outgrows its buffer inside a submission")
		case !g.regrow(&bs.a, &bs.aCap, a, 1, true, true) || !g.regrow(&bs.ax, &bs.axCap, ax, 1, true, true):
			fc.valErr = fmt.Errorf("tier: a block's program operand does not fit")
		}
	}
	fc.qkv, fc.gu, fc.down, fc.dense = fc.prog.forms()
	fc.bs = bs
	// A decode step's kernels are formed for its one row, a chunk's for the
	// scratch's rows: the shape each step runs.
	fc.rows = bs.rows
	if R == 1 && rag == nil {
		fc.rows = 1
	}
	g.curLayer, g.genRag = l, rag != nil
	if rag != nil && ragFew(len(rag.pos)) {
		g.genRagLive = len(rag.pos)
	}
	g.genProgram(fc.prog, bs, fc.rows)
	g.curLayer, g.genRag, g.genRagLive = nil, false, 0
}

// ensureOrders builds the head-major orders block l's variant for R rows
// gathers through, outside any submission. Callers hold g.mu.
func (g *devTier) ensureOrders(l *layer, R int, rag *ragStep) {
	fc := l.fusedFor(R, rag)
	if fc.prog == nil {
		return
	}
	for i := range fc.prog.ops {
		if o := &fc.prog.ops[i]; o.Kind == dkGatherHeads {
			k := headOrder(o, R)
			g.headPerm(k[0], k[1])
		}
	}
}

// ensureFused builds, in the prologue and outside any submission, the fused
// variant and generated kernels of every block of [lo, hi) that runs its
// program on a step of R rows over bs. Callers hold g.mu.
func (g *devTier) ensureFused(lo, hi, R int, bs *blockScratch, rag *ragStep) {
	for li := lo; li < hi && li < len(g.layers); li++ {
		if l := g.layers[li]; l != nil && l.ok && l.prog != nil && l.conv == nil {
			g.ensureFusedLocked(l, l.prog, bs, R, rag)
			g.ensureOrders(l, R, rag)
		}
	}
}

// ragGroupTok is the widest ragged step whose matvecs carry every live row
// in one thread: its dot products grow with the step, and past four rows the
// tiled form is faster (gpu-kernels.md, "Measurements once cited in
// jit/gpu/tier's comments").
const ragGroupTok = 4

func ragFew(n int) bool { return n >= 2 && n <= ragGroupTok }

// headOrder is the order a head gather reads its rows through: the chunk's
// rows head-major, or with N 1 back to the tokens' (the same order with
// heads and rows swapped).
func headOrder(o *devOp, R int) [2]int {
	if o.N == 1 {
		return [2]int{R, o.Heads}
	}
	return [2]int{o.Heads, R}
}

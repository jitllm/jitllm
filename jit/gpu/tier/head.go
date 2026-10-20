package tier

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// headArg is a buffer a head launch reads or writes, resolved at the launch.
type headArg uint8

const (
	haX      headArg = iota // the residual row
	haPart                  // the norm's partial sums
	haNorm                  // the output norm's weight
	haH                     // the normed row
	haA                     // the int8 operand, values
	haAX                    // and scales
	haQS                    // the projection's payload
	haD                     // its scale plane (a float format's: the normed row)
	haSC                    // its scale-of-scale plane
	haSplit                 // a k split's partial plane
	haLogits                // the logits
	haCapped                // the softcapped logits
	haTok                   // each row's greatest logit's index
	haSrc                   // a folded chunk's residual rows
	haPart2                 // a LayerNorm's variance partials
	haNormB                 // its bias
	haOff                   // the word naming the folded row's place in them
)

// headLaunch is one launch of the decode head: a composed kernel and its
// buffers by role.
type headLaunch struct {
	k             backend.Kernel
	groups, width int
	n             uint8
	args          [8]headArg
}

// headBuild is a head kernel's group and the roles of its buffers in order.
type headBuild struct {
	r    compose.Row
	args []headArg
}

func (hb *headBuild) buf(k compose.BufKind, a headArg) int {
	hb.r.Bufs = append(hb.r.Bufs, compose.Buf{Kind: k})
	hb.args = append(hb.args, a)
	return len(hb.r.Bufs) - 1
}

// headKern is hb's kernels under the plan facts f gives it: one launch, or a
// split's two.
func (g *devTier) headKern(hb *headBuild, f compose.Facts) ([]headLaunch, error) {
	ps, err := hb.r.Plan(f)
	if err != nil {
		return nil, err
	}
	var out []headLaunch
	for _, p := range ps {
		s := p.S
		pg := p.G
		k, _ := g.genKern(append([]uint32{encRow}, pg.Encode(s)...), func() (*ir.Kernel, error) { return pg.Emit(s) })
		if k == nil {
			return nil, fmt.Errorf("%s", g.LastErr)
		}
		groups := pg.Rows
		if s.Flat {
			groups = (p.Threads + s.Group - 1) / s.Group
		}
		l := headLaunch{k: k, groups: groups, width: s.Group}
		for i, b := range p.Bufs {
			switch {
			case b >= 0:
				l.args[i] = hb.args[b]
			default:
				l.args[i] = haSplit
			}
		}
		l.n = uint8(len(p.Bufs))
		out = append(out, l)
	}
	return out, nil
}

// PrepHead is nn.LayerDevice's placement of the output norm and projection:
// the projection resident, and the decode head composed -- the norm's
// partial sums and its apply, the int8 operand, the projection at main's
// split, and the final softcap.
func (g *devTier) PrepHead(h *nn.Head) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	bs := g.bs
	if bs == nil || h == nil {
		return false
	}
	if bs.head != nil && bs.head.ok && bs.head.nrows == h.W.Rows && bs.headProg != nil {
		return true
	}
	p := &bs.p
	q, ok := quantOf(h.W.T)
	switch {
	case !ok || h.W.K != p.NEmbd || h.W.Rows <= 0 || len(h.W.Data) == 0:
		return false
	case len(h.Norm) == 0 || p.NEmbd%bs.parts != 0 || lnBlock(p) && (len(h.NormB) != len(h.Norm) || bs.part2 == nil):
		g.LastErr = "the head's norm has no composed form"
		return false
	}
	r := g.resident(-1, 0, q, h.W.Data, h.W.Rows, h.W.K, h.W.Packed)
	if r == nil || !r.ok {
		return false
	}
	split := chooseSplit(h.W.Rows, h.W.K/32, g.kb.split)
	if g.HeadSplit > 0 {
		split = g.HeadSplit
	}
	if !g.sizePart(h.W.Rows * split) {
		return false
	}
	var err error
	f := make([]float32, len(h.Norm))
	for i, v := range h.Norm {
		f[i] = float32(v)
	}
	if bs.hNorm, err = g.dev.Alloc(len(f) * 4); err != nil || bs.hNorm.Write(f32b(f)) != nil {
		return false
	}
	if lnBlock(p) {
		if bs.hNormB, err = g.dev.Alloc(len(h.NormB) * 4); err != nil || bs.hNormB.Write(f32b(h.NormB)) != nil {
			return false
		}
		g.charge(uint64(len(h.NormB) * 4))
	}
	if bs.logits, err = g.dev.Alloc(h.W.Rows * 4); err != nil {
		return false
	}
	if h.Softcap != 0 {
		if bs.logitsCap, err = g.dev.Alloc(h.W.Rows * 4); err != nil {
			return false
		}
	}
	bs.hRaw = make([]byte, h.W.Rows*4)
	bs.hOut = unsafe.Slice((*float32)(unsafe.Pointer(&bs.hRaw[0])), h.W.Rows)
	prog, err := g.headProgram(p, r, h, split)
	if err != nil {
		g.LastErr = fmt.Sprintf("the head: %v", err)
		return false
	}
	bs.head, bs.headProg = r, prog
	bs.mvHead = mv{rows: h.W.Rows, split: split}
	g.charge(uint64(len(h.Norm)*4 + h.W.Rows*4))
	bs.headEmbeds, bs.embScale = h.Embeds, h.EmbdScale
	if bs.embScale == 0 {
		bs.embScale = 1
	}
	g.shapeChanged()
	return true
}

// headProgram is the decode head's launches over projection r.
func (g *devTier) headProgram(p *nn.LayerPlan, r *resident, h *nn.Head, split int) ([]headLaunch, error) {
	t := g.composeTarget()
	flat := compose.Facts{Target: t}
	parts, n := g.bs.parts, p.NEmbd
	var prog []headLaunch
	add := func(hb *headBuild, f compose.Facts) error {
		ls, err := g.headKern(hb, f)
		prog = append(prog, ls...)
		return err
	}
	if lnBlock(p) {
		for _, b := range layerNormBuilds(n, 1, parts, float32(p.RMSEps)) {
			if err := add(b, flat); err != nil {
				return nil, err
			}
		}
	} else {
		ps := &headBuild{r: compose.Row{Width: parts, Rows: 1}}
		x := ps.buf(compose.BufRow, haX)
		ps.r.Store(ps.buf(compose.BufOut, haPart), ps.r.SquareSumParts(x, n, parts))
		if err := add(ps, flat); err != nil {
			return nil, err
		}
		na := &headBuild{r: compose.Row{Width: n, Rows: 1}}
		ax, w, part := na.buf(compose.BufRow, haX), na.buf(compose.BufVec, haNorm), na.buf(compose.BufRow, haPart)
		na.r.Store(na.buf(compose.BufOut, haH), na.r.NormApplyParts(ax, w, part, n, parts, float32(p.RMSEps), false))
		if err := add(na, flat); err != nil {
			return nil, err
		}
	}
	float := kernels.IsFloat(r.t)
	if !float {
		qb := &headBuild{r: compose.Row{Width: n / 4, Rows: 1}}
		qb.r.Quantize8(qb.buf(compose.BufRow, haH), qb.buf(compose.BufOutU32, haA), qb.buf(compose.BufOut, haAX), min(g.actWinFor(1), n))
		if err := add(qb, flat); err != nil {
			return nil, err
		}
	}
	mb := &headBuild{r: compose.Row{Width: h.W.Rows, Rows: 1}}
	wt := compose.MatVecWeight{T: r.t, K: n, Rows: h.W.Rows, QS: mb.buf(compose.BufU32, haQS), D: -1, SC: -1}
	op := compose.MatVecOperand{Toks: 1}
	if float {
		op.X = mb.buf(compose.BufRow, haD)
	} else {
		wt.D = mb.buf(compose.BufU32, haD)
		wt.SC = mb.buf(compose.BufU32, haSC)
		op.A, op.AX = mb.buf(compose.BufU32, haA), mb.buf(compose.BufRow, haAX)
	}
	v, err := mb.r.MatVec(wt, op)
	if err != nil {
		return nil, err
	}
	mb.r.Store(mb.buf(compose.BufOut, haLogits), v)
	if err := add(mb, compose.Facts{Target: t, Step: compose.StepDecode, DecodeSplit: split}); err != nil {
		return nil, err
	}
	if h.Softcap != 0 {
		cb := &headBuild{r: compose.Row{Width: h.W.Rows, Rows: 1}}
		lx := cb.r.Load(cb.buf(compose.BufRow, haLogits))
		cb.r.Store(cb.buf(compose.BufOut, haCapped), cb.r.Softcap(lx, h.Softcap))
		if err := add(cb, flat); err != nil {
			return nil, err
		}
	}
	return prog, nil
}

// headBufs is what a head program's roles name on one launch.
type headBufs struct {
	x, part, norm, h, a, ax, logits, capped, tok, src, off, part2, normB backend.Buf
	r                                                                    *resident
}

// runHead launches prog over bufs, the first groups of each launch where
// groups is above 0 (the rows that want a head).
func (g *devTier) runHead(lc *launcher, prog []headLaunch, hb headBufs) {
	var bufs [8]backend.Buf
	for i := range prog {
		l := &prog[i]
		for j := 0; j < int(l.n); j++ {
			switch l.args[j] {
			case haX:
				bufs[j] = hb.x
			case haPart:
				bufs[j] = hb.part
			case haNorm:
				bufs[j] = hb.norm
			case haH:
				bufs[j] = hb.h
			case haA:
				bufs[j] = hb.a
			case haAX:
				bufs[j] = hb.ax
			case haQS:
				bufs[j] = hb.r.qs
			case haD:
				if kernels.IsFloat(hb.r.t) {
					bufs[j] = hb.h
				} else {
					bufs[j] = hb.r.d
				}
			case haSC:
				bufs[j] = hb.r.sc
			case haSplit:
				bufs[j] = g.partBuf
			case haLogits:
				bufs[j] = hb.logits
			case haCapped:
				bufs[j] = hb.capped
			case haTok:
				bufs[j] = hb.tok
			case haSrc:
				bufs[j] = hb.src
			case haOff:
				bufs[j] = hb.off
			case haPart2:
				bufs[j] = hb.part2
			case haNormB:
				bufs[j] = hb.normB
			}
		}
		if err := lc.launch(l.k, l.groups, l.width, bufs[:l.n]...); err != nil {
			return
		}
	}
}

// decodeHeadBufs is the decode head's binding on bs.
func decodeHeadBufs(bs *blockScratch) headBufs {
	return headBufs{x: bs.x, part: bs.part, norm: bs.hNorm, h: bs.h, a: bs.a, ax: bs.ax, logits: bs.logits,
		capped: bs.logitsCap, r: bs.head, part2: bs.part2, normB: bs.hNormB}
}

// rowsHeadBufs is the per-row head's binding: rows scratch bs, the head the
// decode scratch hb holds.
func rowsHeadBufs(bs, hb *blockScratch) headBufs {
	return headBufs{x: bs.x, part: bs.part, norm: hb.hNorm, h: bs.h, a: bs.a, ax: bs.ax, logits: bs.ragLogits,
		capped: bs.ragCap, tok: bs.ragTok, r: hb.head, part2: bs.part2, normB: hb.hNormB}
}

// ensureRowsHead composes the per-row head for rows scratch bs, the decode
// scratch's head over its rows: their RMSNorm, the int8 operand, the first
// row's projection at decode's split (the operand's sums after every row's
// scales), its softcap, and the argmax of each wanted row. Callers hold
// g.mu, outside any submission.
func (g *devTier) ensureRowsHead(bs *blockScratch, h *nn.Head) bool {
	hd := g.bs
	if bs.ragHeadProg != nil {
		return true
	}
	if hd == nil || hd.headProg == nil || h == nil || !g.reserveRowsHead(bs, &bs.p) {
		return false
	}
	p, r, R := &bs.p, hd.head, bs.rows
	n, vocab := p.NEmbd, hd.mvHead.rows
	t := g.composeTarget()
	flat := compose.Facts{Target: t}
	var prog []headLaunch
	add := func(b *headBuild, f compose.Facts) bool {
		ls, err := g.headKern(b, f)
		prog = append(prog, ls...)
		if err != nil {
			g.LastErr = fmt.Sprintf("rows: the head: %v", err)
		}
		return err == nil
	}
	if lnBlock(p) {
		if bs.part2 == nil {
			return false
		}
		for _, b := range layerNormBuilds(n, R, bs.parts, float32(p.RMSEps)) {
			if !add(b, flat) {
				return false
			}
		}
	} else {
		nb := &headBuild{r: compose.Row{Width: n, Rows: R}}
		x := nb.r.Load(nb.buf(compose.BufRow, haX))
		w := nb.r.Load(nb.buf(compose.BufVec, haNorm))
		ss := nb.r.Add(compose.Node{Kind: compose.NReduce, Red: compose.RedSum, A: nb.r.Bin(compose.OpMul, x, x)})
		rms := nb.r.Add(compose.Node{Kind: compose.NUnary, Op: compose.OpRsqrt,
			A: nb.r.Bin(compose.OpAdd, nb.r.Bin(compose.OpMul, ss, nb.r.Const(1/float32(n))), nb.r.Const(float32(p.RMSEps)))})
		nb.r.Store(nb.buf(compose.BufOut, haH), nb.r.Bin(compose.OpMul, nb.r.Bin(compose.OpMul, x, rms), w))
		if !add(nb, flat) {
			return false
		}
	}
	float := kernels.IsFloat(r.t)
	if !float {
		qb := &headBuild{r: compose.Row{Width: R * n / 4, Rows: 1}}
		qb.r.Quantize8(qb.buf(compose.BufRow, haH), qb.buf(compose.BufOutU32, haA), qb.buf(compose.BufOut, haAX), min(g.actWinFor(R), n))
		if !add(qb, flat) {
			return false
		}
	}
	mb := &headBuild{r: compose.Row{Width: vocab, Rows: 1}}
	wt := compose.MatVecWeight{T: r.t, K: n, Rows: vocab, QS: mb.buf(compose.BufU32, haQS), D: -1, SC: -1}
	op := compose.MatVecOperand{Toks: R}
	if float {
		op.X = mb.buf(compose.BufRow, haD)
	} else {
		wt.D, wt.SC = mb.buf(compose.BufU32, haD), mb.buf(compose.BufU32, haSC)
		op.A, op.AX = mb.buf(compose.BufU32, haA), mb.buf(compose.BufRow, haAX)
	}
	v, err := mb.r.MatVec(wt, op)
	if err != nil {
		g.LastErr = fmt.Sprintf("rows: the head: %v", err)
		return false
	}
	mb.r.Store(mb.buf(compose.BufOut, haLogits), v)
	if !add(mb, compose.Facts{Target: t, Step: compose.StepDecode, DecodeSplit: hd.mvHead.split}) {
		return false
	}
	in := haLogits
	if h.Softcap != 0 {
		cb := &headBuild{r: compose.Row{Width: vocab, Rows: 1}}
		lx := cb.r.Load(cb.buf(compose.BufRow, haLogits))
		cb.r.Store(cb.buf(compose.BufOut, haCapped), cb.r.Softcap(lx, h.Softcap))
		if !add(cb, flat) {
			return false
		}
		in = haCapped
	}
	ab := &headBuild{r: compose.Row{Width: vocab, Rows: R}}
	ab.r.Argmax(ab.r.Load(ab.buf(compose.BufRow, in)), ab.buf(compose.BufOutU32, haTok))
	if !add(ab, flat) {
		return false
	}
	// The argmax runs over the wanted rows: one group each, the first.
	prog[len(prog)-1].groups = 1
	bs.ragHeadProg = prog
	return true
}

// ensureSpan composes the decode scratch hb's row copy a folded chunk's head
// reads: the chunk's last residual row, at the offset its word holds, into
// hb's row. Callers hold g.mu.
func (g *devTier) ensureSpan(hb *blockScratch) bool {
	if hb.spanProg != nil {
		return true
	}
	if hb.spanOff == nil {
		w := g.scratchWin() // the scratch's own word (scratch.go)
		b, err := g.dev.Alloc(4)
		w.close()
		if err != nil {
			return false
		}
		hb.spanOff = b
	}
	sb := &headBuild{r: compose.Row{Width: hb.p.NEmbd, Rows: 1}}
	src, dst, off := sb.buf(compose.BufRow, haSrc), sb.buf(compose.BufOut, haX), sb.buf(compose.BufU32, haOff)
	sb.r.Store(dst, sb.r.LoadAt(src, sb.r.UBin(compose.OpAdd, sb.r.LoadAt(off, sb.r.UConst(0)), sb.r.Column())))
	ls, err := g.headKern(sb, compose.Facts{Target: g.composeTarget()})
	if err != nil {
		g.LastErr = fmt.Sprintf("the folded head's row copy: %v", err)
		return false
	}
	hb.spanProg = ls
	return true
}

// layerNormBuilds is a LayerNorm head's norm over rows of n in three
// launches: the parts' sums, their deviations, the apply with the norm's
// weight and bias.
func layerNormBuilds(n, rows, parts int, eps float32) []*headBuild {
	sb := &headBuild{r: compose.Row{Width: parts * rows, Rows: 1}}
	sx := sb.buf(compose.BufRow, haX)
	sb.r.Store(sb.buf(compose.BufOut, haPart), sb.r.SumParts(sx, n, parts))
	vb := &headBuild{r: compose.Row{Width: parts * rows, Rows: 1}}
	vx, vs := vb.buf(compose.BufRow, haX), vb.buf(compose.BufRow, haPart)
	vb.r.Store(vb.buf(compose.BufOut, haPart2), vb.r.VarParts(vx, vs, n, parts))
	ab := &headBuild{r: compose.Row{Width: n, Rows: rows}}
	ax, w := ab.buf(compose.BufRow, haX), ab.buf(compose.BufVec, haNorm)
	sum, vr := ab.buf(compose.BufRow, haPart), ab.buf(compose.BufRow, haPart2)
	bias := ab.buf(compose.BufVec, haNormB)
	ab.r.Store(ab.buf(compose.BufOut, haH), ab.r.LayerNormApplyParts(ax, w, bias, sum, vr, n, parts, eps))
	return []*headBuild{sb, vb, ab}
}

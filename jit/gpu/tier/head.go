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
	case len(h.Norm) == 0 || lnBlock(p) || p.NEmbd%bs.parts != 0:
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

// runHead launches the decode head on bs's row.
func (g *devTier) runHead(lc *launcher, bs *blockScratch) {
	r := bs.head
	var bufs [8]backend.Buf
	for i := range bs.headProg {
		l := &bs.headProg[i]
		for j := 0; j < int(l.n); j++ {
			switch l.args[j] {
			case haX:
				bufs[j] = bs.x
			case haPart:
				bufs[j] = bs.part
			case haNorm:
				bufs[j] = bs.hNorm
			case haH:
				bufs[j] = bs.h
			case haA:
				bufs[j] = bs.a
			case haAX:
				bufs[j] = bs.ax
			case haQS:
				bufs[j] = r.qs
			case haD:
				if kernels.IsFloat(r.t) {
					bufs[j] = bs.h
				} else {
					bufs[j] = r.d
				}
			case haSC:
				bufs[j] = r.sc
			case haSplit:
				bufs[j] = g.partBuf
			case haLogits:
				bufs[j] = bs.logits
			case haCapped:
				bufs[j] = bs.logitsCap
			}
		}
		if err := lc.launch(l.k, l.groups, l.width, bufs[:l.n]...); err != nil {
			return
		}
	}
}

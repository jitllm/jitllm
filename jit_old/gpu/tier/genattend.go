package tier

import (
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/gpu/compose"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// stagedPlan is a paged attention's split count and chunk for keys keys:
// the power of two at or below sqrt(keys/8), raised until the accumulate
// launches a quarter of the device's resident threads, and no chunk under
// 32 keys.
func stagedPlan(p compose.Paged, keys, slots int) compose.Paged {
	if slots <= 0 {
		slots = 16384
	}
	keys = max(1, keys)
	for sp := 1 << int(math.Log2(math.Sqrt(max(1, float64(keys)/8)))); ; sp *= 2 {
		p.Splits = sp
		p.Chunk = max(32, ((keys+sp-1)/sp+31)/32*32)
		if 4*p.Rows*(p.Heads/p.HeadGroup())*sp*p.Dim >= slots || p.Chunk == 32 {
			return p
		}
	}
}

// genAttend is the paged decode attention as the staged composition: the
// scores of every split's slots, each split's softmax, its weighted sum of
// V, and the splits' merge, planned for the scratch's whole context so one
// variant serves every call. The planes are grown here, outside any
// submission.
func (g *devTier) genAttend(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	// Formed for the whole context, which sizes the planes every shorter
	// call's plan fits in (the splits only grow with the keys); a call runs
	// the plan of its own keys (attendAt).
	if g.attendKeys(o, prog, bs, rows, (max(g.kvCap, 1)+scoreGrain-1)/scoreGrain*scoreGrain) == nil {
		return nil
	}
	return []genLaunch{{former: formAttend}}
}

// attendAt is the attention's launches for a call whose rows attend up to
// keys keys, rounded to the recording grain: generated once a plan and kept.
func (g *devTier) attendAt(o *devOp, prog *devProgram, bs *blockScratch, rows, keys int) []genLaunch {
	keys = (max(keys, 1) + scoreGrain - 1) / scoreGrain * scoreGrain
	k := attendKey{op: o.Src[0], rows: rows, keys: keys}
	if ls, ok := prog.attend[k]; ok {
		return ls
	}
	ls := g.attendKeys(o, prog, bs, rows, keys)
	if ls != nil {
		if prog.attend == nil {
			prog.attend = map[attendKey][]genLaunch{}
		}
		prog.attend[k] = ls
	}
	return ls
}

// attendKey names an attention variant: its op, its rows, its keys.
type attendKey struct {
	op, rows, keys int
	gathered       bool
}

// attendKeys is the paged attention planned for keys keys: the flash
// decode where its plan takes it, else the staged composition.
func (g *devTier) attendKeys(o *devOp, prog *devProgram, bs *blockScratch, rows, keys int) []genLaunch {
	p := &bs.p
	pg := bs.pkv
	// A latent row is one KV head of the latent and its key, the value its
	// prefix, both read from the one pool.
	latent := o.Kind == dkLatentAttend
	kvh, hd, vd, vpool := p.NKVHead, p.HeadDim, p.HeadDim, gaVPool
	if latent {
		kvh, hd, vd, vpool = 1, p.KVLoraRank+p.NRot, p.KVLoraRank, gaKPool
	}
	if pg == nil || g.KVF16 || p.MLA() != latent || p.NonCausal || len(o.In) != 1 ||
		prog.vals[o.In[0]].Width != p.NHead*hd {
		return nil
	}
	if !latent {
		if fl, ok := g.flashPlan(p, rows, keys); ok {
			return g.genFlash(fl, bs)
		}
	}
	if p.AttnSinks {
		return nil
	}
	pp := compose.Paged{Rows: rows, Heads: p.NHead, KVHeads: kvh, Dim: hd, Page: g.kvPage()}
	if latent {
		pp.MLA = vd
	}
	pa := stagedPlan(pp, keys, g.dev.Slots())
	plane := pa.Rows * pa.Heads * pa.Splits * pa.Chunk
	parts := pa.Rows * pa.Heads * pa.Splits * ((vd+31)/32*32 + 64)
	if g.genInSub && (pg.scCap < plane || pg.partCap < parts) ||
		!g.regrow(&pg.sc, &pg.scCap, plane, 4, true, true) || !g.regrow(&pg.wt, &pg.wtCap, plane, 4, true, true) ||
		!g.regrow(&pg.part, &pg.partCap, parts, 4, true, true) {
		return nil
	}
	prow := gaPRow
	if l := g.curLayer; l != nil && l.windowed {
		prow = gaPRowW
	}
	scale, cap := g.scoreScale(p), p.AttnSoftcap
	// Where the device guarantees a subgroup, a split's softmax takes one,
	// and so does a latent row's slot, hundreds of floats a key.
	lanes := g.composeTarget().Lanes == ir.SubgroupLanes
	ps, ss := pa, pa
	if lanes {
		ss.Lanes = ir.SubgroupLanes
		if latent {
			ps.Lanes = ir.SubgroupLanes
		}
	}

	// The staged chain as the composer's program: scores, softmax,
	// accumulation and merge, each lowered alone and planned by the shape.
	args := []genArg{gaIn0, gaKPool, vpool, gaTab, prow, gaScores, gaWts, gaParts, gaOut0}
	vals := make([]compose.Value, len(args))
	for i := range vals {
		vals[i] = compose.Value{Buf: -1}
	}
	vals[3].Elem, vals[4].Elem = compose.ValU32, compose.ValU32
	pr := &compose.Program{Rows: rows, Vals: vals, Prims: []compose.Prim{
		{Kind: compose.PrimScores, In: []int{0, 1, 3, 4}, Out: []int{5}, Paged: ps, Scale: scale, Softcap: cap},
		{Kind: compose.PrimSoftmax, In: []int{5, 4}, Out: []int{6, 7}, Paged: pa},
		{Kind: compose.PrimAcc, In: []int{6, 2, 3, 4}, Out: []int{7}, Paged: pa},
		{Kind: compose.PrimMerge, In: []int{7}, Out: []int{8}, Paged: pa},
	}}
	if ss.Lanes > 1 {
		pr.Prims[1].Facts = compose.Facts{RowGroup: 64, GroupRows: true}
	}
	gs, err := pr.Lower()
	if err != nil {
		g.LastErr = err.Error()
		return nil
	}
	var ls []genLaunch
	for gi := range gs {
		cls, err := gs[gi].Launches(g.composeTarget())
		if err != nil {
			g.LastErr = err.Error()
			return nil
		}
		for _, cl := range cls {
			k, _ := g.genKern(append([]uint32{encRow}, cl.Key...), cl.Emit)
			if k == nil || len(cl.Binds) > len(genLaunch{}.args) {
				return nil
			}
			l := genLaunch{k: k, groups: cl.Groups, width: cl.Width, n: uint8(len(cl.Binds))}
			for j, bd := range cl.Binds {
				l.args[j] = args[bd.Index]
			}
			ls = append(ls, l)
		}
	}
	return ls
}

// flashPlan is the paged decode attention as one workgroup a (row, query
// group, split) where the device promises 32 lanes and the history is short
// against it: on PTX and Metal while keys*Dim stays under 24 resident
// threads' worth of the device, on Vulkan under 16384 keys. A head wider
// than 128 at 4096 keys or more takes four warps; the splits are the power
// of two nearest sqrt(keys/8), capped at about 1.6 workgroups a 1280
// threads, floored by what the chunk holds.
func (g *devTier) flashPlan(p *nn.LayerPlan, rows, keys int) (compose.Flash, bool) {
	t := g.composeTarget()
	if t.Lanes != ir.SubgroupLanes {
		return compose.Flash{}, false
	}
	slots := g.dev.Slots()
	if slots <= 0 {
		slots = 16384
	}
	s := compose.Flash{Heads: p.NHead, KVHeads: p.NKVHead, Dim: p.HeadDim, Rows: rows, Page: g.kvPage(),
		Scale: float32(p.Scale()), Softcap: p.AttnSoftcap, Sink: p.AttnSinks,
		KImm: t.API == compose.APIMSL, VecV: t.API == compose.APIPTX}
	switch t.API {
	case compose.APIPTX, compose.APIMSL:
		if keys*s.Dim >= 24*slots {
			return s, false
		}
	case compose.APISPIRV:
		if keys >= 16384 {
			return s, false
		}
	}
	if t.API != compose.APISPIRV && t.API != compose.APIMSL && s.Dim > 128 && keys >= 4096 {
		s.Warps = 4
	}
	s.Splits = 1
	floor := (keys + s.Chunk() - 1) / s.Chunk()
	capped := max(1, (slots+1280*s.Groups()-1)/(1280*s.Groups()))
	sq := 1 << int(math.Round(math.Log2(math.Sqrt(max(1, float64(keys)/8)))))
	s.Splits = max(1, floor, min(sq, capped))
	return s, true
}

// genFlash is the flash decode's launch and, over splits, the merge of the
// splits' partials.
func (g *devTier) genFlash(s compose.Flash, bs *blockScratch) []genLaunch {
	pg := bs.pkv
	parts := s.Rows * s.Heads * s.Splits * ((s.Dim+31)/32*32 + 64)
	if s.Splits > 1 && (g.genInSub && pg.partCap < parts || !g.genInSub && !g.regrow(&pg.part, &pg.partCap, parts, 4, true, true)) {
		return nil
	}
	prow := gaPRow
	if l := g.curLayer; l != nil && l.windowed {
		prow = gaPRowW
	}
	fb := &rowBuild{r: compose.Row{Width: s.Width(), Rows: s.Groups()}}
	q, k, v := fb.buf(compose.BufRow, gaIn0), fb.buf(compose.BufRow, gaKPool), fb.buf(compose.BufRow, gaVPool)
	out := gaOut0
	if s.Splits > 1 {
		out = gaParts
	}
	o := fb.buf(compose.BufOut, out)
	tab, desc := fb.buf(compose.BufU32, gaTab), fb.buf(compose.BufU32, prow)
	sink := -1
	if s.Sink && s.Splits == 1 {
		sink = fb.buf(compose.BufRow, gaSink)
	}
	fb.r.PagedFlash(s, q, k, v, tab, desc, o, sink)
	rbs := []*rowBuild{fb}
	if s.Splits > 1 {
		pa := compose.Paged{Rows: s.Rows, Heads: s.Heads, KVHeads: s.KVHeads, Dim: s.Dim, Page: s.Page, Splits: s.Splits}
		mg := &rowBuild{r: compose.Row{Width: s.Rows * s.Heads * s.Dim, Rows: 1}}
		mpart, mout := mg.buf(compose.BufRow, gaParts), mg.buf(compose.BufOut, gaOut0)
		msink := -1
		if s.Sink {
			msink = mg.buf(compose.BufRow, gaSink)
		}
		mg.r.PagedMerge(mpart, mout, msink, pa)
		rbs = append(rbs, mg)
	}
	var ls []genLaunch
	for _, rb := range rbs {
		k := g.rowKern(&rb.r)
		if k.k == nil {
			return nil
		}
		l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args)), whole: true}
		copy(l.args[:], rb.args)
		ls = append(ls, l)
	}
	return ls
}

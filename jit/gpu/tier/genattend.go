package tier

import (
	"math"

	"github.com/jitllm/jitllm/jit/gpu/compose"
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
	p := &bs.p
	pg := bs.pkv
	if pg == nil || g.KVF16 || p.MLA() || p.AttnSinks || p.NonCausal || len(o.In) != 1 ||
		prog.vals[o.In[0]].Width != p.NHead*p.HeadDim {
		return nil
	}
	pa := stagedPlan(compose.Paged{Rows: rows, Heads: p.NHead, KVHeads: p.NKVHead, Dim: p.HeadDim, Page: g.kvPage()},
		g.kvCap, g.dev.Slots())
	plane := pa.Rows * pa.Heads * pa.Splits * pa.Chunk
	parts := pa.Rows * pa.Heads * pa.Splits * ((pa.Dim+31)/32*32 + 64)
	if g.genInSub && (pg.scCap < plane || pg.partCap < parts) ||
		!g.regrow(&pg.sc, &pg.scCap, plane, 4, true, true) || !g.regrow(&pg.wt, &pg.wtCap, plane, 4, true, true) ||
		!g.regrow(&pg.part, &pg.partCap, parts, 4, true, true) {
		return nil
	}
	prow := gaPRow
	if l := g.curLayer; l != nil && l.windowed {
		prow = gaPRowW
	}
	scale, cap := float32(p.Scale()), p.AttnSoftcap
	G := pa.HeadGroup()

	sc := &rowBuild{r: compose.Row{Width: rows * (pa.Heads / G) * pa.Splits * pa.Chunk, Rows: 1}}
	q, k := sc.buf(compose.BufRow, gaIn0), sc.buf(compose.BufRow, gaKPool)
	sc.buf(compose.BufU32, prow)
	out := sc.buf(compose.BufOut, gaScores)
	tab, desc := sc.buf(compose.BufU32, gaTab), sc.buf(compose.BufU32, prow)
	sc.r.PagedScores(q, k, out, tab, desc, pa, scale, cap)

	sm := &rowBuild{r: compose.Row{Width: rows * pa.Heads * pa.Splits, Rows: 1}}
	scores := sm.buf(compose.BufRow, gaScores)
	sm.buf(compose.BufU32, prow)
	wts, part := sm.buf(compose.BufOut, gaWts), sm.buf(compose.BufOut, gaParts)
	sm.buf(compose.BufU32, gaTab)
	sm.r.PagedSoftmax(scores, wts, part, sm.buf(compose.BufU32, prow), pa)

	ac := &rowBuild{r: compose.Row{Width: rows * (pa.Heads / G) * pa.Splits * pa.Dim, Rows: 1}}
	w, v := ac.buf(compose.BufRow, gaWts), ac.buf(compose.BufRow, gaVPool)
	ac.buf(compose.BufU32, prow)
	apart := ac.buf(compose.BufOut, gaParts)
	atab, adesc := ac.buf(compose.BufU32, gaTab), ac.buf(compose.BufU32, prow)
	ac.r.PagedAcc(w, v, apart, atab, adesc, pa)

	mg := &rowBuild{r: compose.Row{Width: rows * pa.Heads * pa.Dim, Rows: 1}}
	mpart, mout := mg.buf(compose.BufRow, gaParts), mg.buf(compose.BufOut, gaOut0)
	mg.r.PagedMerge(mpart, mout, -1, pa)

	var ls []genLaunch
	for _, rb := range []*rowBuild{sc, sm, ac, mg} {
		k := g.rowKern(&rb.r)
		if k.k == nil {
			return nil
		}
		l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args))}
		copy(l.args[:], rb.args)
		ls = append(ls, l)
	}
	return ls
}

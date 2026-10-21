package tier

import (
	"github.com/jitllm/jitllm/jit_old/gpu/compose"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// genMoERows is a mixture's ops over a chunk's rows, through the grouped
// scratch: the pairs grouped by expert on the device and the used column
// count read home, the rows gathered in that order and quantized once for
// gate and up, each bank a grouped matvec over the used columns, the outputs
// combined back in token order. handled is false for an op that is not a
// mixture's over rows.
func (g *devTier) genMoERows(o *devOp, prog *devProgram, bs *blockScratch, rows int) (ls []genLaunch, handled bool) {
	m := bs.mg
	if rows < 2 || !prog.experts || m == nil || m.rows != rows || o.Heads > 0 {
		return nil, false // a bank over heads selects by head, not by the route's grouping
	}
	launch := func(rb *rowBuild, colScale, colDiv int, once uint8) bool {
		k := g.rowKern(&rb.r)
		if k.k == nil {
			return false
		}
		l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args)), whole: true,
			colScale: colScale, colDiv: colDiv, once: once}
		copy(l.args[:], rb.args)
		ls = append(ls, l)
		return true
	}
	p := &bs.p
	E, K, np := m.nExp, m.k, m.p
	ew := p.ExpWidth()
	win := p.ActWin
	if win <= 0 {
		win = 32
	}
	u32, out32 := compose.BufU32, compose.BufOutU32
	switch o.Kind {
	case dkGroupByExpert:
		nb := compose.GroupBlocks(rows, K)
		bc := &rowBuild{r: compose.Row{Width: nb * E, Rows: 1}}
		sel := bc.buf(u32, gaIn0)
		bc.r.Store(bc.buf(out32, gaMgBC), bc.r.GroupBlockCount(sel, rows, K, E))
		ct := &rowBuild{r: compose.Row{Width: nb * E, Rows: 1}}
		ct.r.GroupCount(ct.buf(u32, gaMgBC), ct.buf(out32, gaMgOff), ct.buf(out32, gaMgCnt), rows, K, E)
		en := &rowBuild{r: compose.Row{Width: E, Rows: 1}}
		cnt := en.buf(u32, gaMgCnt)
		en.r.Store(en.buf(out32, gaMgEnds), en.r.GroupEnds(cnt, E, m.unit))
		co := &rowBuild{r: compose.Row{Width: np, Rows: 1}}
		cp := -1
		if m.colPair != nil {
			cp = co.buf(out32, gaColPair)
		}
		co.r.GroupColumns(co.buf(u32, gaMgEnds), co.buf(out32, gaMgColE), co.buf(out32, gaMgPerm), cp, rows, E)
		sc := &rowBuild{r: compose.Row{Width: rows * K, Rows: 1}}
		ssel, scnt, sends, soff := sc.buf(u32, gaIn0), sc.buf(u32, gaMgCnt), sc.buf(u32, gaMgEnds), sc.buf(u32, gaMgOff)
		sp := -1
		if m.colPair != nil {
			sp = sc.buf(out32, gaColPair)
		}
		sc.r.GroupScatter(ssel, scnt, sends, soff, sc.buf(out32, gaMgPerm), sc.buf(out32, gaMgPos), sp, rows, K, E, m.unit)
		ti := &rowBuild{r: compose.Row{Width: np / groupTok, Rows: 1}}
		ends := ti.buf(u32, gaMgEnds)
		ti.r.Store(ti.buf(out32, gaMgTsel), ti.r.GroupTiles(ends, E, groupTok))
		builds := []*rowBuild{bc, ct, en, co, sc, ti}
		if m.tile {
			tu := &rowBuild{r: compose.Row{Width: np / m.unit, Rows: 1}}
			tends := tu.buf(u32, gaMgEnds)
			tu.r.Store(tu.buf(out32, gaMgTselU), tu.r.GroupTiles(tends, E, m.unit))
			builds = append(builds, tu)
		}
		for _, rb := range builds {
			if !launch(rb, 0, 0, 0) {
				return nil, true
			}
		}
		ls = append(ls, genLaunch{readEnds: true})
	case dkExpertMatVec:
		l := g.curLayer
		if l == nil || o.W < 0 || o.W >= len(l.ms) || *l.rs[o.W] == nil || len(o.In) < 3 {
			return nil, true
		}
		mv := *l.ms[o.W]
		if mv.experts <= 1 || mv.pad {
			return nil, true
		}
		float := kernels.IsFloat(mv.q)
		if m.tile && !float {
			if tl := g.genTileBank(o, prog, mv, m, rows); tl != nil {
				ls = append(ls, tl...)
				break
			}
		}
		src := gaIn2
		if prog.vals[o.In[2]].Layout != layoutPairs {
			// Gate and up read the rows gathered into sorted order.
			gb := &rowBuild{r: compose.Row{Width: np * ew, Rows: 1}}
			x, perm := gb.buf(compose.BufRow, gaIn2), gb.buf(u32, gaMgPerm)
			gb.r.Store(gb.buf(compose.BufOut, gaMgHG), gb.r.GatherRows(x, perm, ew, rows))
			if !launch(gb, ew, 1, 1) {
				return nil, true
			}
			src = gaMgHG
		}
		if !float {
			qb := &rowBuild{r: compose.Row{Width: np * mv.k / 4, Rows: 1}}
			qb.r.Quantize8(qb.buf(compose.BufRow, src), qb.buf(out32, gaMgGA), qb.buf(compose.BufOut, gaMgGAX), win)
			if !launch(qb, mv.k/4, 1, 2) {
				return nil, true
			}
		}
		rb := &rowBuild{r: compose.Row{Width: mv.rows, Rows: np}}
		w := compose.MatVecWeight{T: mv.q, K: mv.k, Rows: mv.rows, QS: rb.buf(u32, gaWQS), D: -1, SC: -1, Experts: mv.experts, SelPer: groupTok}
		if !float {
			w.D = rb.buf(u32, gaWD)
		}
		if mv.q.Info().PerSuper > 1 {
			w.SC = rb.buf(u32, gaWSC)
		}
		a := compose.MatVecOperand{Toks: np}
		if float {
			a.X = rb.buf(compose.BufRow, src)
		} else {
			a.A, a.AX = rb.buf(u32, gaMgGA), rb.buf(compose.BufRow, gaMgGAX)
		}
		w.Sel = rb.buf(u32, gaMgTsel)
		v, err := rb.r.MatVec(w, a)
		if err != nil {
			g.LastErr = err.Error()
			return nil, true
		}
		rb.store(v, gaOut0)
		ps, err := rb.r.Plan(compose.Facts{Target: g.composeTarget(), Step: compose.StepChunk})
		if err != nil || len(ps) != 1 {
			return nil, true
		}
		s := ps[0].S
		k, _ := g.genKern(append([]uint32{encRow}, ps[0].G.Encode(s)...), func() (*ir.Kernel, error) { return ps[0].G.Emit(s) })
		if k == nil {
			return nil, true
		}
		gl := genLaunch{k: k, width: s.Group, n: uint8(len(rb.args)), whole: true, colScale: mv.rows, colDiv: groupTok}
		copy(gl.args[:], rb.args)
		ls = append(ls, gl)
	case dkCombine:
		if len(o.In) != 2 {
			return nil, true
		}
		cb := &rowBuild{r: compose.Row{Width: rows * ew, Rows: 1}}
		e, w, pos := cb.buf(compose.BufRow, gaIn0), cb.buf(compose.BufRow, gaIn1), cb.buf(u32, gaMgPos)
		cb.store(cb.r.CombineGrouped(e, w, pos, ew, K), gaOut0)
		if !launch(cb, 0, 0, 0) {
			return nil, true
		}
	default:
		return nil, false
	}
	o.Why = whyNone
	return ls, true
}

// genTileBank is a bank over a chunk's sorted columns as the grouped tile
// GEMM: the source made the binary16 operand (gate and up gather the rows
// in sorted order as they convert, shared where their formats agree), then
// the widest blocking the bank's rows divide over the used token blocks,
// the row block fastest. Nil where the format or the shape has no form.
func (g *devTier) genTileBank(o *devOp, prog *devProgram, mv mv, m *moeGroup, rows int) []genLaunch {
	sub := mv.q.Info().Sub
	if sub <= 0 || mv.k%sub != 0 || m.unit%16 != 0 {
		return nil
	}
	var ls []genLaunch
	launch := func(rb *rowBuild, colScale, colDiv int, once uint8) bool {
		k := g.rowKern(&rb.r)
		if k.k == nil {
			return false
		}
		l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(rb.args)), whole: true,
			colScale: colScale, colDiv: colDiv, once: once}
		copy(l.args[:], rb.args)
		ls = append(ls, l)
		return true
	}
	np := m.p
	cv := &rowBuild{r: compose.Row{Width: np * mv.k / sub, Rows: 1}}
	x, h16 := cv.buf(compose.BufRow, gaIn2), cv.buf(compose.BufOutU32, gaMgH16)
	var err error
	if prog.vals[o.In[2]].Layout != layoutPairs {
		err = cv.r.ActF16TGather(mv.q, mv.k, x, cv.buf(compose.BufU32, gaMgPerm), rows, h16)
	} else {
		err = cv.r.ActF16T(mv.q, mv.k, x, h16)
	}
	if err != nil || !launch(cv, mv.k/sub, 1, 3) {
		return nil
	}
	for _, mt := range []int{4, 2, 1} {
		s := compose.GemmTile{T: mv.q, Rows: mv.rows, K: mv.k, NTok: np, MT: mt, NT: m.unit / 16, WM: 2, WN: 2,
			KB: max(32/sub, 1), Tall: g.tileTall() && mt%2 == 0, Experts: mv.experts}
		if mv.rows%s.BlockRows() != 0 || np%s.Toks() != 0 {
			continue
		}
		blocks := mv.rows / s.BlockRows()
		rb := &rowBuild{r: compose.Row{Width: s.Threads(), Rows: s.Groups()}}
		qs, d, sc := rb.buf(compose.BufU32, gaWQS), rb.buf(compose.BufU32, gaWD), rb.buf(compose.BufU32, gaWSC)
		b, out := rb.buf(compose.BufU32, gaMgH16), rb.buf(compose.BufTile, gaOut0)
		if rb.r.GemmTile(s, qs, d, sc, b, out, -1, rb.buf(compose.BufU32, gaMgTselU)) != nil {
			continue
		}
		if !launch(rb, blocks*s.Threads(), m.unit, 0) {
			return nil
		}
		return ls
	}
	return nil
}

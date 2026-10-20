package tier

import (
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// A block-selecting attention (an OpIndex with Block set, read by the
// attention): each row's history cut into blocks of Block positions, each
// block scored against the row's indexer query per kv group, the best kept
// with the blocks the selection forces in, their keys and values gathered
// into pages of the row's own, and the attention run over those. Two
// scorings: a block's best position (each position's key the cached
// indexer head, the Local blocks ending at the row's own forced in), or,
// Pooled, the block's keys averaged, normed and turned at its first
// position, every complete block a candidate and the incomplete tail
// forced in.

// selMarkCap is how many rotary marks (a picture moving the positions after
// it) one call's rows carry between them.
const selMarkCap = 64

// selSize is a block selection's sizes.
type selSize struct {
	pooled                          bool
	blk, local, tcap, nb, keep, ppr int
	heads, groups, hd, attnW, kvRow int
	// scores is the scores value's width a row, list the selection's, gk
	// and gv the gathered pages'.
	scores, list, gk, gv int
}

func selSizes(p *nn.LayerPlan, a graph.Attrs, page int) selSize {
	s := selSize{pooled: a.Pooled, blk: max(a.Block, 1), local: a.Local, tcap: max(p.MaxSeq, 1),
		heads: p.IdxHeads, groups: p.NKVHead, hd: p.HeadDim, attnW: p.NKVHead * p.HeadDim, kvRow: p.KVRow()}
	if s.pooled {
		s.nb = s.tcap/s.blk + 1
		s.keep = min(p.IdxTopK/s.blk+1, s.nb)
	} else {
		s.nb = (s.tcap + s.blk - 1) / s.blk
		s.keep = min(p.IdxTopK, s.nb)
	}
	s.ppr = (s.keep*s.blk + page - 1) / max(page, 1)
	s.scores, s.list = s.groups*s.nb, s.groups*(s.keep+1)
	s.gk, s.gv = s.ppr*page*s.attnW, s.ppr*page*s.attnW
	return s
}

// ownBuf is op o's own buffer slot of words words in bs, made once (filled
// from init where given) and kept with the scratch. Nil inside a submission
// where it would have to be made.
func (g *devTier) ownBuf(bs *blockScratch, o *devOp, slot, words int, init []uint32) backend.Buf {
	k := ownKey{src: o.Src[0], kind: o.Kind, slot: slot}
	if e, ok := bs.own[k]; ok && e.n >= words {
		return e.b
	}
	if g.genInSub {
		return nil
	}
	b, err := g.dev.Alloc(max(words, 1) * 4)
	if err == nil && init != nil {
		err = b.Write(u32view(init))
	}
	if err != nil {
		if b != nil {
			b.Free()
		}
		g.LastErr = "an op's own buffer: " + err.Error()
		return nil
	}
	if e, ok := bs.own[k]; ok {
		e.b.Free()
	}
	if bs.own == nil {
		bs.own = map[ownKey]ownEntry{}
	}
	bs.own[k] = ownEntry{b: b, n: words}
	return b
}

// ownKey names an op's own buffer in a scratch: the graph op it lowers, its
// kind and its slot.
type ownKey struct {
	src  int
	kind devKind
	slot int
}

type ownEntry struct {
	b backend.Buf
	n int
}

// f32u is v's bits.
func f32u(v []float32) []uint32 {
	u := make([]uint32, len(v))
	for i, f := range v {
		u[i] = math.Float32bits(f)
	}
	return u
}

// rowTable is rows words r*stride + base.
func rowTable(rows, stride, base int) []uint32 {
	t := make([]uint32, rows)
	for r := range t {
		t[r] = uint32(r*stride + base)
	}
	return t
}

// launches forms each group into its launch, nil if one does not form.
func (g *devTier) launches(rbs ...*rowBuild) []genLaunch {
	var ls []genLaunch
	for _, rb := range rbs {
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

// selRow is a flat group of width columns by rows rows with the helpers the
// selection's kernels share.
type selRow struct {
	*rowBuild
	page int
}

func newSelRow(width, rows, page int) selRow {
	return selRow{rowBuild: &rowBuild{r: compose.Row{Width: width, Rows: rows}}, page: page}
}

func (s selRow) u(n int) int                    { return s.r.UConst(n) }
func (s selRow) ub(op compose.Op, a, b int) int { return s.r.UBin(op, a, b) }

// desc is word w of row r's descriptor.
func (s selRow) desc(d, r, w int) int {
	return s.r.LoadAt(d, s.ub(compose.OpAdd, s.ub(compose.OpMul, r, s.u(kernels.PRowWords)), s.u(w)))
}

// pageOf is the page holding position t of the row whose table starts at tab.
func (s selRow) pageOf(tbl, tab, t int) int {
	return s.r.LoadAt(tbl, s.ub(compose.OpAdd, tab, s.ub(compose.OpDiv, t, s.u(s.page))))
}

// own is the row's end (at least one) and its own block: the block of its
// last position, or the pooled form's tail.
func (s selRow) own(d, r int, sz selSize) (end, own int) {
	end = s.ub(compose.OpMax, s.desc(d, r, kernels.PRowEnd), s.u(1))
	if sz.pooled {
		return end, s.ub(compose.OpDiv, end, s.u(sz.blk))
	}
	return end, s.ub(compose.OpDiv, s.ub(compose.OpSub, end, s.u(1)), s.u(sz.blk))
}

// kept is a row's kept positions: every kept block whole but its own, which
// ends at the history's end.
func (s selRow) kept(end, own int, sz selSize) int {
	nsel := s.ub(compose.OpMin, s.ub(compose.OpAdd, own, s.u(1)), s.u(sz.keep))
	whole := s.ub(compose.OpMul, s.ub(compose.OpSub, nsel, s.u(1)), s.u(sz.blk))
	return s.ub(compose.OpAdd, whole, s.ub(compose.OpSub, end, s.ub(compose.OpMul, own, s.u(sz.blk))))
}

func (s selRow) fma(a, b, c int) int {
	return s.r.Add(compose.Node{Kind: compose.NBinary, Op: compose.OpFma, A: a, B: b, C: c})
}

// genSelect forms a block selection's ops: its scores, its kept blocks, the
// gather and the attention over the gathered pages.
func (g *devTier) genSelect(o *devOp, prog *devProgram, bs *blockScratch, rows int) []genLaunch {
	x := selectOfProg(prog)
	p := &bs.p
	if x == nil || bs.pkv == nil || g.KVF16 {
		return nil
	}
	page := g.kvPage()
	sz := selSizes(p, x.Attr, page)
	switch o.Kind {
	case dkIdxAssemble:
		return g.genAssemble(o, prog, bs, rows, sz)
	case dkIdxScore:
		if sz.pooled {
			return g.genPooledScores(o, prog, bs, rows, sz, page)
		}
		return g.genBestScores(o, prog, bs, rows, sz, page)
	case dkSelectBlocks:
		return g.genKeep(o, prog, bs, rows, sz)
	case dkGatherKV:
		if prog.vals[o.Out[0]].Width != sz.gk || prog.vals[o.Out[1]].Width != sz.gv {
			return nil // the page size moved since the program was built
		}
		return g.genGather(o, prog, rows, sz, page)
	case dkAttendGathered:
		tab := g.ownBuf(bs, o, 0, rows*sz.ppr, rowTable(rows*sz.ppr, 1, 0))
		if tab == nil {
			return nil
		}
		o.Own[0] = tab
		q := *o
		q.In = o.In[:1]
		if g.attendKeys(&q, prog, bs, rows, (max(g.kvCap, 1)+scoreGrain-1)/scoreGrain*scoreGrain) == nil {
			return nil
		}
		return []genLaunch{{former: formAttend}}
	}
	return nil
}

// selectOfProg is the program's block-selecting indexer, nil without one.
func selectOfProg(prog *devProgram) *graph.Op {
	if prog.graph == nil {
		return nil
	}
	return selectOf(prog.graph)
}

// gatheredAttendAt is the attention over the gathered pages for a call
// whose rows attend up to keys keys, at most the kept positions: the decode
// plan at every row count, its pool arguments the gathered pages'.
func (g *devTier) gatheredAttendAt(o *devOp, prog *devProgram, bs *blockScratch, rows, keys int) []genLaunch {
	x := selectOfProg(prog)
	if x == nil {
		return nil
	}
	sz := selSizes(&bs.p, x.Attr, g.kvPage())
	keys = (min(max(keys, 1), sz.keep*sz.blk) + scoreGrain - 1) / scoreGrain * scoreGrain
	k := attendKey{op: o.Src[0], rows: rows, keys: keys, gathered: true}
	if ls, ok := prog.attend[k]; ok {
		return ls
	}
	q := *o
	q.In = o.In[:1]
	ls := gatheredArgs(g.attendKeys(&q, prog, bs, rows, keys))
	if len(ls) == 0 {
		return nil
	}
	if prog.attend == nil {
		prog.attend = map[attendKey][]genLaunch{}
	}
	prog.attend[k] = ls
	return ls
}

// gatheredArgs points an attention's pool arguments at the gathered pages:
// the keys In[1], the values In[2], the descriptors In[3], the table the
// op's own.
func gatheredArgs(ls []genLaunch) []genLaunch {
	out := make([]genLaunch, len(ls))
	for i, l := range ls {
		for j := range l.args[:l.n] {
			switch l.args[j] {
			case gaKPool:
				l.args[j] = gaIn1
			case gaVPool:
				l.args[j] = gaIn2
			case gaPRow, gaPRowW:
				l.args[j] = gaIn3
			case gaTab:
				l.args[j] = gaOwn0
			}
		}
		out[i] = l
	}
	return out
}

// genAssemble is the cached row with the indexer's key: the attention's
// key heads at the row's start, the indexer's key after them.
func (g *devTier) genAssemble(o *devOp, prog *devProgram, bs *blockScratch, rows int, sz selSize) []genLaunch {
	kw, iw := prog.vals[o.In[0]].Width, prog.vals[o.In[1]].Width
	if kw != sz.attnW || kw+iw > prog.vals[o.Out[0]].Width {
		return nil
	}
	ow := prog.stride(o.Out[0])
	at, ix := g.ownBuf(bs, o, 0, rows, rowTable(rows, ow, 0)), g.ownBuf(bs, o, 1, rows, rowTable(rows, ow, kw))
	if at == nil || ix == nil {
		return nil
	}
	o.Own[0], o.Own[1] = at, ix
	copyAt := func(w int, src, off genArg) *rowBuild {
		cb := &rowBuild{r: compose.Row{Width: w, Rows: rows}}
		x, t := cb.buf(compose.BufRow, src), cb.buf(compose.BufU32, off)
		cb.r.StoreAt(cb.buf(compose.BufOut, gaOut0), cb.r.Load(x), cb.r.UBin(compose.OpAdd, cb.r.Load(t), cb.r.Column()))
		return cb
	}
	return g.launches(copyAt(kw, gaIn0, gaOwn0), copyAt(iw, gaIn1, gaOwn1))
}

// genBestScores scores each block by its best position: every position of
// each row's history against the row's query heads (one a kv group), then
// each block's maximum, the Local blocks ending at the row's own +inf and a
// block past it -inf.
func (g *devTier) genBestScores(o *devOp, prog *devProgram, bs *blockScratch, rows int, sz selSize, page int) []genLaunch {
	p := &bs.p
	d := p.IdxHeadDim
	if sz.heads != sz.groups || prog.vals[o.In[0]].Width != sz.heads*d || sz.local < 1 {
		return nil
	}
	raw := g.ownBuf(bs, o, 0, rows*sz.heads*sz.tcap, nil)
	if raw == nil {
		return nil
	}
	o.Own[0] = raw
	koff := sz.attnW

	s := newSelRow(sz.tcap, rows*sz.heads, page)
	q, k, tbl, dsc := s.buf(compose.BufRow, gaIn0), s.buf(compose.BufRow, gaKPool), s.buf(compose.BufU32, gaTab), s.buf(compose.BufU32, gaPRow)
	out := s.buf(compose.BufOut, gaOwn0)
	rh := s.r.RowIdx()
	r := s.ub(compose.OpDiv, rh, s.u(sz.heads))
	end := s.ub(compose.OpMax, s.desc(dsc, r, kernels.PRowEnd), s.u(1))
	t := s.ub(compose.OpMin, s.r.Column(), s.ub(compose.OpSub, end, s.u(1)))
	pid := s.pageOf(tbl, s.desc(dsc, r, kernels.PRowTab), t)
	kb := s.ub(compose.OpAdd, s.ub(compose.OpAdd, s.ub(compose.OpMul, pid, s.u(page*sz.kvRow)), s.u(koff*page)),
		s.ub(compose.OpRem, t, s.u(page)))
	qb := s.ub(compose.OpMul, rh, s.u(d))
	c := s.r.Loop(d)
	i := s.r.Iter(c)
	acc := s.r.Fold(c, s.r.Const(0))
	kv := s.r.LoadAt(k, s.ub(compose.OpAdd, kb, s.ub(compose.OpMul, i, s.u(page))))
	s.r.SetFold(acc, s.fma(s.r.LoadAt(q, s.ub(compose.OpAdd, qb, i)), kv, acc))
	s.r.Store(out, acc)

	m := newSelRow(sz.nb, rows*sz.heads, page)
	sc, mdsc := m.buf(compose.BufRow, gaOwn0), m.buf(compose.BufU32, gaPRow)
	mout := m.buf(compose.BufOut, gaOut0)
	bi, mrh := m.r.Column(), m.r.RowIdx()
	mr := m.ub(compose.OpDiv, mrh, m.u(sz.heads))
	mend, own := m.own(mdsc, mr, sz)
	negInf, posInf := m.r.Const(float32(math.Inf(-1))), m.r.Const(float32(math.Inf(1)))
	t0 := m.ub(compose.OpMul, bi, m.u(sz.blk))
	sb := m.ub(compose.OpMul, mrh, m.u(sz.tcap))
	mc := m.r.Loop(sz.blk)
	mi := m.r.Iter(mc)
	mx := m.r.Fold(mc, negInf)
	tt := m.ub(compose.OpAdd, t0, mi)
	sv := m.r.LoadAt(sc, m.ub(compose.OpAdd, sb, m.ub(compose.OpMin, tt, m.u(sz.tcap-1))))
	m.r.SetFold(mx, m.r.Select(m.r.Less(tt, mend), m.r.Bin(compose.OpMax, mx, sv), mx))
	forced := m.r.Less(own, m.ub(compose.OpAdd, bi, m.u(sz.local)))
	v := m.r.Select(forced, posInf, mx)
	m.r.Store(mout, m.r.Select(m.r.Less(own, bi), negInf, v))
	return g.launches(s.rowBuild, m.rowBuild)
}

// genPooledScores scores each complete block of each row's history by its
// keys' mean (the cached indexer head), normed by the indexer's key norm,
// turned at the block's first position (a picture's mark moving it) and
// held to the row's query heads: sum_h relu(q_h . key). The tail block
// scores +inf and a block past it -inf. The turns are two tables built per
// call: the blocks' first positions, and the marks' deltas.
func (g *devTier) genPooledScores(o *devOp, prog *devProgram, bs *blockScratch, rows int, sz selSize, page int) []genLaunch {
	p := &bs.p
	d, rot, h := p.IdxHeadDim, p.NRot, sz.heads
	planes := p.RopeTabScalar
	if prog.vals[o.In[0]].Width != h*d || rot <= 0 || rot%2 != 0 || rot > d ||
		len(planes) != kernels.RopeTabBlock(rot/2) {
		return nil
	}
	if g.curLayer == nil {
		return nil
	}
	kn := g.curLayer.vecOf(jlm.RoleIdxKNorm)
	if kn == nil {
		return nil
	}
	pos := make([]uint32, sz.nb+1)
	pos[0] = uint32(sz.nb)
	for b := range sz.nb {
		pos[b+1] = uint32(b * sz.blk)
	}
	marks := make([]uint32, rows+1+selMarkCap)
	dpos := make([]uint32, selMarkCap+1)
	dpos[0] = 1
	bufs := [...]backend.Buf{
		g.ownBuf(bs, o, 0, sz.nb*rot, nil),
		g.ownBuf(bs, o, 1, sz.nb+1, pos),
		g.ownBuf(bs, o, 2, len(planes), f32u(planes)),
		g.ownBuf(bs, o, 3, len(kernels.RopeTabConsts()), f32u(kernels.RopeTabConsts())),
		g.ownBuf(bs, o, 4, len(marks), marks),
		g.ownBuf(bs, o, 5, len(dpos), dpos),
	}
	dcs := g.ownBuf(bs, o, 6, selMarkCap*rot, nil)
	for _, b := range bufs {
		if b == nil {
			return nil
		}
	}
	if dcs == nil {
		return nil
	}
	copy(o.Own[:], bufs[:])
	o.Own[6], o.V = dcs, kn

	table := func(n int, out, at genArg) *rowBuild {
		rb := &rowBuild{r: compose.Row{Width: rot / 2, Rows: n}}
		o, pl, cn, ps := rb.buf(compose.BufOut, out), rb.buf(compose.BufRow, gaOwn2), rb.buf(compose.BufU32, gaOwn3), rb.buf(compose.BufU32, at)
		rb.r.RopeTable(o, pl, cn, ps)
		return rb
	}

	s := newSelRow(sz.nb, rows, page)
	q, k, tbl, dsc := s.buf(compose.BufRow, gaIn0), s.buf(compose.BufRow, gaKPool), s.buf(compose.BufU32, gaTab), s.buf(compose.BufU32, gaPRow)
	cs, w, mk, dc := s.buf(compose.BufRow, gaOwn0), s.buf(compose.BufVec, gaV), s.buf(compose.BufU32, gaOwn4), s.buf(compose.BufRow, gaOwn6)
	out := s.buf(compose.BufOut, gaOut0)
	R := &s.r
	bi, r := R.Column(), R.RowIdx()
	end := s.ub(compose.OpMax, s.desc(dsc, r, kernels.PRowEnd), s.u(1))
	tail := s.ub(compose.OpDiv, end, s.u(sz.blk))
	tab := s.desc(dsc, r, kernels.PRowTab)
	kbase := make([]int, sz.blk)
	for j := range kbase {
		t := s.ub(compose.OpMin, s.ub(compose.OpAdd, s.ub(compose.OpMul, bi, s.u(sz.blk)), s.u(j)), s.ub(compose.OpSub, end, s.u(1)))
		pid := s.pageOf(tbl, tab, t)
		kbase[j] = s.ub(compose.OpAdd, s.ub(compose.OpAdd, s.ub(compose.OpMul, pid, s.u(page*sz.kvRow)), s.u(sz.attnW*page)),
			s.ub(compose.OpRem, t, s.u(page)))
	}
	mean := func(i int) int {
		acc := R.Const(0)
		for j := range kbase {
			acc = R.Bin(compose.OpAdd, acc, R.LoadAt(k, s.ub(compose.OpAdd, kbase[j], s.ub(compose.OpMul, i, s.u(page)))))
		}
		if len(kbase) > 1 {
			acc = R.Bin(compose.OpMul, acc, R.Const(1/float32(len(kbase))))
		}
		return acc
	}
	// The row's last mark at or before the block's start.
	m0 := R.LoadAt(mk, r)
	nm := s.ub(compose.OpSub, R.LoadAt(mk, s.ub(compose.OpAdd, r, s.u(1))), m0)
	start := s.ub(compose.OpMul, bi, s.u(sz.blk))
	none := s.u(0x7FFFFFFF)
	mc := R.LoopN(nm)
	mi := R.Iter(mc)
	sel := R.Fold(mc, none)
	mw := R.LoadAt(mk, s.ub(compose.OpAdd, s.ub(compose.OpAdd, m0, mi), s.u(rows+1)))
	at := s.ub(compose.OpAnd, mw, s.u(0x7FFFFFFF))
	R.SetFold(sel, R.Select(R.Less(start, at), sel, s.ub(compose.OpAdd, m0, mi)))
	marked := R.Less(sel, none)
	selC := R.Select(marked, sel, s.u(0))
	neg := R.Less(s.u(0), s.ub(compose.OpAnd, R.LoadAt(mk, s.ub(compose.OpAdd, selC, s.u(rows+1))), s.u(0x80000000)))
	dsign := R.Select(neg, R.Const(-1), R.Const(1))
	// The norm's sum of squares.
	nc := R.Loop(d)
	ni := R.Iter(nc)
	ss := R.Fold(nc, R.Const(0))
	mv := mean(ni)
	R.SetFold(ss, s.fma(mv, mv, ss))
	inv := R.Bin(compose.OpDiv, R.Const(1), R.Add(compose.Node{Kind: compose.NUnary, Op: compose.OpSqrt,
		A: R.Bin(compose.OpAdd, R.Bin(compose.OpMul, ss, R.Const(1/float32(d))), R.Const(float32(p.RMSEps)))}))
	qb := s.ub(compose.OpMul, r, s.u(h*d))
	nv := func(i int) int { return R.Bin(compose.OpMul, R.Bin(compose.OpMul, mean(i), inv), R.LoadAt(w, i)) }
	qat := func(i, hh int) int {
		return R.LoadAt(q, s.ub(compose.OpAdd, s.ub(compose.OpAdd, qb, i), s.u(hh*d)))
	}
	acc := make([]int, h)
	np := rot / 2
	pc := R.Loop(np)
	pp := R.Iter(pc)
	for hh := range acc {
		acc[hh] = R.Fold(pc, R.Const(0))
	}
	ia, ib := s.ub(compose.OpAdd, pp, pp), s.ub(compose.OpAdd, s.ub(compose.OpAdd, pp, pp), s.u(1))
	if p.RopeNeox {
		ia, ib = pp, s.ub(compose.OpAdd, pp, s.u(np))
	}
	va, vb := nv(ia), nv(ib)
	csb := s.ub(compose.OpAdd, s.ub(compose.OpMul, bi, s.u(rot)), s.ub(compose.OpAdd, pp, pp))
	c1, s1 := R.LoadAt(cs, csb), R.LoadAt(cs, s.ub(compose.OpAdd, csb, s.u(1)))
	db := s.ub(compose.OpAdd, s.ub(compose.OpMul, selC, s.u(rot)), s.ub(compose.OpAdd, pp, pp))
	c2 := R.Select(marked, R.LoadAt(dc, db), R.Const(1))
	s2 := R.Select(marked, R.Bin(compose.OpMul, dsign, R.LoadAt(dc, s.ub(compose.OpAdd, db, s.u(1)))), R.Const(0))
	cc := R.Bin(compose.OpSub, R.Bin(compose.OpMul, c1, c2), R.Bin(compose.OpMul, s1, s2))
	sn := R.Bin(compose.OpAdd, R.Bin(compose.OpMul, s1, c2), R.Bin(compose.OpMul, c1, s2))
	ra := R.Bin(compose.OpSub, R.Bin(compose.OpMul, va, cc), R.Bin(compose.OpMul, vb, sn))
	rb := R.Bin(compose.OpAdd, R.Bin(compose.OpMul, va, sn), R.Bin(compose.OpMul, vb, cc))
	for hh := range acc {
		R.SetFold(acc[hh], s.fma(qat(ib, hh), rb, s.fma(qat(ia, hh), ra, acc[hh])))
	}
	if rot < d {
		tc := R.Loop(d - rot)
		j := s.ub(compose.OpAdd, R.Iter(tc), s.u(rot))
		v := nv(j)
		for hh := range acc {
			f := R.Fold(tc, acc[hh])
			R.SetFold(f, s.fma(qat(j, hh), v, f))
			acc[hh] = f
		}
	}
	score := R.Const(0)
	for hh := range acc {
		score = R.Bin(compose.OpAdd, score, R.Bin(compose.OpMax, acc[hh], R.Const(0)))
	}
	negInf, posInf := R.Const(float32(math.Inf(-1))), R.Const(float32(math.Inf(1)))
	score = R.Select(R.Less(bi, tail), score, R.Select(R.Less(tail, bi), negInf, posInf))
	ob := s.ub(compose.OpAdd, s.ub(compose.OpMul, r, s.u(sz.groups*sz.nb)), bi)
	for gg := range sz.groups {
		R.StoreAt(out, score, s.ub(compose.OpAdd, ob, s.u(gg*sz.nb)))
	}
	return g.launches(table(sz.nb, gaOwn0, gaOwn1), table(selMarkCap, gaOwn6, gaOwn5), s.rowBuild)
}

// genKeep flags each row's and group's kept blocks -- at or below its own,
// fewer than min(keep, own+1) ranking above it (a higher score, or an equal
// one at a lower block) -- then lists the kept blocks in ascending order,
// a block not kept in the spare slot.
func (g *devTier) genKeep(o *devOp, prog *devProgram, bs *blockScratch, rows int, sz selSize) []genLaunch {
	if prog.vals[o.In[0]].Width != sz.scores || prog.vals[o.Out[0]].Width != sz.list {
		return nil
	}
	flag := g.ownBuf(bs, o, 0, rows*sz.groups*sz.nb, nil)
	if flag == nil {
		return nil
	}
	o.Own[0] = flag
	s := newSelRow(sz.nb, rows*sz.groups, 0)
	sc, dsc := s.buf(compose.BufRow, gaIn0), s.buf(compose.BufU32, gaPRow)
	out := s.buf(compose.BufOutU32, gaOwn0)
	R := &s.r
	bi, rh := R.Column(), R.RowIdx()
	r := s.ub(compose.OpDiv, rh, s.u(sz.groups))
	_, own := s.own(dsc, r, sz)
	n := s.ub(compose.OpMin, s.ub(compose.OpAdd, own, s.u(1)), s.u(sz.nb))
	keep := s.ub(compose.OpMin, n, s.u(sz.keep))
	base := s.ub(compose.OpMul, rh, s.u(sz.nb))
	sv := R.Load(sc)
	c := R.LoopN(n)
	uu := R.Iter(c)
	rank := R.Fold(c, s.u(0))
	su := R.LoadAt(sc, s.ub(compose.OpAdd, base, uu))
	before := R.Select(R.Less(sv, su), s.u(1), R.Select(R.Less(su, sv), s.u(0), R.Select(R.Less(uu, bi), s.u(1), s.u(0))))
	R.SetFold(rank, s.ub(compose.OpAdd, rank, before))
	R.Store(out, R.Select(R.Less(rank, keep), R.Select(R.Less(bi, n), s.u(1), s.u(0)), s.u(0)))

	cp := newSelRow(sz.nb, rows*sz.groups, 0)
	fl := cp.buf(compose.BufU32, gaOwn0)
	list := cp.buf(compose.BufOutU32, gaOut0)
	C := &cp.r
	cbi, crh := C.Column(), C.RowIdx()
	cbase := cp.ub(compose.OpMul, crh, cp.u(sz.nb))
	cc := C.LoopN(cbi)
	cu := C.Iter(cc)
	slot := C.Fold(cc, cp.u(0))
	C.SetFold(slot, cp.ub(compose.OpAdd, slot, C.LoadAt(fl, cp.ub(compose.OpAdd, cbase, cu))))
	kk := cp.u(sz.keep)
	at := C.Select(C.Less(cp.u(0), C.Load(fl)), cp.ub(compose.OpMin, slot, kk), kk)
	C.StoreAt(list, cbi, cp.ub(compose.OpAdd, cp.ub(compose.OpMul, crh, cp.u(sz.keep+1)), at))
	return g.launches(s.rowBuild, cp.rowBuild)
}

// genGather copies each row's kept keys and values into its gathered pages
// (kv head h of gathered position j the kept position j of group h) and
// writes their descriptors.
func (g *devTier) genGather(o *devOp, prog *devProgram, rows int, sz selSize, page int) []genLaunch {
	kc, w := sz.keep*sz.blk, sz.attnW
	pos := func(s selRow, tbl, dsc, sel, r, hh, j int) (t, pid int) {
		end, own := s.own(dsc, r, sz)
		kept := s.kept(end, own, sz)
		jj := s.ub(compose.OpMin, j, s.ub(compose.OpSub, kept, s.u(1)))
		rh := s.ub(compose.OpAdd, s.ub(compose.OpMul, r, s.u(sz.groups)), hh)
		blk := s.r.LoadAt(sel, s.ub(compose.OpAdd, s.ub(compose.OpMul, rh, s.u(sz.keep+1)), s.ub(compose.OpDiv, jj, s.u(sz.blk))))
		t = s.ub(compose.OpMin, s.ub(compose.OpAdd, s.ub(compose.OpMul, blk, s.u(sz.blk)), s.ub(compose.OpRem, jj, s.u(sz.blk))),
			s.ub(compose.OpSub, end, s.u(1)))
		return t, s.pageOf(tbl, s.desc(dsc, r, kernels.PRowTab), t)
	}
	gk := newSelRow(kc, rows*w, page)
	src, tbl, dsc, sel := gk.buf(compose.BufRow, gaKPool), gk.buf(compose.BufU32, gaTab), gk.buf(compose.BufU32, gaPRow), gk.buf(compose.BufU32, gaIn0)
	dk := gk.buf(compose.BufOut, gaOut0)
	j, er := gk.r.Column(), gk.r.RowIdx()
	e, r := gk.ub(compose.OpRem, er, gk.u(w)), gk.ub(compose.OpDiv, er, gk.u(w))
	t, pid := pos(gk, tbl, dsc, sel, r, gk.ub(compose.OpDiv, e, gk.u(sz.hd)), j)
	from := gk.ub(compose.OpAdd, gk.ub(compose.OpAdd, gk.ub(compose.OpMul, pid, gk.u(page*sz.kvRow)), gk.ub(compose.OpMul, e, gk.u(page))),
		gk.ub(compose.OpRem, t, gk.u(page)))
	gp := gk.ub(compose.OpAdd, gk.ub(compose.OpMul, r, gk.u(sz.ppr)), gk.ub(compose.OpDiv, j, gk.u(page)))
	to := gk.ub(compose.OpAdd, gk.ub(compose.OpAdd, gk.ub(compose.OpMul, gp, gk.u(page*w)), gk.ub(compose.OpMul, e, gk.u(page))),
		gk.ub(compose.OpRem, j, gk.u(page)))
	gk.r.StoreAt(dk, gk.r.LoadAt(src, from), to)

	gv := newSelRow(w, rows*kc, page)
	vsrc, vtbl, vdsc, vsel := gv.buf(compose.BufRow, gaVPool), gv.buf(compose.BufU32, gaTab), gv.buf(compose.BufU32, gaPRow), gv.buf(compose.BufU32, gaIn0)
	dv := gv.buf(compose.BufOut, gaOut1)
	uu, rj := gv.r.Column(), gv.r.RowIdx()
	vj, vr := gv.ub(compose.OpRem, rj, gv.u(kc)), gv.ub(compose.OpDiv, rj, gv.u(kc))
	vt, vpid := pos(gv, vtbl, vdsc, vsel, vr, gv.ub(compose.OpDiv, uu, gv.u(sz.hd)), vj)
	vfrom := gv.ub(compose.OpAdd, gv.ub(compose.OpMul, gv.ub(compose.OpAdd, gv.ub(compose.OpMul, vpid, gv.u(page)), gv.ub(compose.OpRem, vt, gv.u(page))),
		gv.u(sz.kvRow)), uu)
	vgp := gv.ub(compose.OpAdd, gv.ub(compose.OpMul, vr, gv.u(sz.ppr)), gv.ub(compose.OpDiv, vj, gv.u(page)))
	vto := gv.ub(compose.OpAdd, gv.ub(compose.OpMul, gv.ub(compose.OpAdd, gv.ub(compose.OpMul, vgp, gv.u(page)), gv.ub(compose.OpRem, vj, gv.u(page))),
		gv.u(w)), uu)
	gv.r.StoreAt(dv, gv.r.LoadAt(vsrc, vfrom), vto)

	ds := newSelRow(kernels.PRowWords, rows, page)
	ddsc := ds.buf(compose.BufU32, gaPRow)
	dout := ds.buf(compose.BufOutU32, gaOut2)
	ww, dr := ds.r.Column(), ds.r.RowIdx()
	dend, down := ds.own(ddsc, dr, sz)
	kept := ds.kept(dend, down, sz)
	tab := ds.ub(compose.OpMul, dr, ds.u(sz.ppr))
	isTab := ds.r.Less(ww, ds.u(kernels.PRowTab+1))
	isEnd := ds.r.Select(ds.r.Less(ww, ds.u(kernels.PRowEnd)), ds.u(0),
		ds.r.Select(ds.r.Less(ww, ds.u(kernels.PRowEnd+1)), ds.u(1), ds.u(0)))
	ds.r.Store(dout, ds.r.Select(isTab, tab, ds.ub(compose.OpMul, isEnd, kept)))
	return g.launches(gk.rowBuild, gv.rowBuild, ds.rowBuild)
}

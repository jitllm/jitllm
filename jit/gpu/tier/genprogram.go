package tier

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// progBuild is prog being stated as the composer's program: past prog's
// values, the step's tables an op reads (a rotary's table, the history's
// descriptors, pages and pools), each a value bound to its argument.
type progBuild struct {
	cp  *compose.Program
	ext []genArg
	at  map[genArg]int
}

// table is the program value standing for the step's buffer at argument a.
func (pb *progBuild) table(a genArg, e compose.ValElem) int {
	if v, ok := pb.at[a]; ok {
		return v
	}
	pb.cp.Vals = append(pb.cp.Vals, compose.Value{Elem: e, Buf: -1})
	pb.cp.Out = append(pb.cp.Out, true)
	pb.ext = append(pb.ext, a)
	v := len(pb.cp.Vals) - 1
	pb.at[a] = v
	return v
}

// composeProgram is prog's ops as the composer's program at rows, op i its
// prim i: an op of a kind the composer forms, with operands it takes, a
// prim; every other an extern.
func (g *devTier) composeProgram(prog *devProgram, bs *blockScratch, rows int) *progBuild {
	cp := &compose.Program{Rows: rows, Vals: make([]compose.Value, len(prog.vals)), Out: make([]bool, len(prog.vals))}
	pb := &progBuild{cp: cp, at: map[genArg]int{}}
	for v, t := range prog.vals {
		cp.Out[v] = prog.slots[v] == slotX // the block's row, read past it
		e := compose.ValF32
		if t.Elem == elemQ8 {
			e = compose.ValQ8
		}
		b := int(prog.slots[v])
		if s := prog.slots[v]; s == slotKV || s == slotAuto {
			b = -1
		}
		cp.Vals[v] = compose.Value{Elem: e, Width: t.Width, Buf: b}
	}
	cp.Prims = make([]compose.Prim, len(prog.ops))
	for i := range prog.ops {
		cp.Prims[i] = g.primOf(&prog.ops[i], prog, pb, bs, rows)
	}
	return pb
}

// primOf is op o as a prim of the composer's program.
func (g *devTier) primOf(o *devOp, prog *devProgram, pb *progBuild, bs *blockScratch, rows int) compose.Prim {
	vals := func(vs []devVal) []int {
		out := make([]int, len(vs))
		for i, v := range vs {
			out[i] = int(v)
		}
		return out
	}
	ext := compose.Prim{Kind: compose.PrimExtern, In: vals(o.In), Out: vals(o.Out)}
	switch o.Kind {
	case dkKVWrite, dkKVCopy:
		return g.kvPrim(o, prog, pb, bs, rows, ext)
	case dkExpertBias:
		// Each selected pair's row plus its expert's bias row.
		if o.V == nil || len(o.In) != 2 || len(o.Out) != 1 || prog.vals[o.In[1]].Elem != elemU32 {
			return ext
		}
		return compose.Prim{Kind: compose.PrimGatherAdd, In: ext.In, Out: ext.Out}
	}
	for _, v := range append(append([]devVal(nil), o.In...), o.Out...) {
		if prog.vals[v].Layout != layoutRow {
			return ext
		}
	}
	if len(o.Out) != 1 {
		return ext
	}
	if o.Kind == dkMatVec {
		return g.matvecPrim(o, prog, ext, bs, rows)
	}
	p := &bs.p
	pr := compose.Prim{In: ext.In, Out: ext.Out, Alpha: float32(o.Alpha)}
	switch o.Kind {
	case dkRouterBias:
		if o.V == nil || len(o.In) != 1 {
			return ext
		}
		pr.Kind = compose.PrimAdd
	case dkMap:
		// Rows at their width, no padding past it, each operand dividing
		// the result.
		fn, ok := mapFnOf(o.A.Fn)
		out := prog.vals[o.Out[0]].Width
		if !ok || prog.stride(o.Out[0]) != out {
			return ext
		}
		for _, v := range o.In {
			if n := prog.vals[v].Width; n == 0 || out%n != 0 || prog.stride(v) != n {
				return ext
			}
		}
		pr.Kind, pr.Map, pr.Bound, pr.Alpha = compose.PrimMap, fn, o.V != nil, float32(o.A.Const)
	case dkAdd:
		if len(o.In) != 2 || o.Alpha != 0 && o.Alpha != 1 && o.GC == nil {
			return ext
		}
		pr.Kind = compose.PrimAdd
	case dkMul:
		if len(o.In) != 2 {
			return ext
		}
		pr.Kind = compose.PrimMul
	case dkCopy:
		pr.Kind = compose.PrimCopy
	case dkAct:
		a, ok := kernAct(o.Act)
		if !ok || o.Act == graph.ActXIELU || len(o.In) != 1 {
			return ext
		}
		pr.Kind, pr.Act = compose.PrimAct, a
	case dkActMul:
		a, ok := kernAct(o.Act)
		if len(o.In) != 2 || !ok && !o.A.GLU.Bounded() {
			return ext
		}
		pr.Kind, pr.Act, pr.GLU = compose.PrimActMul, a, o.A.GLU
	case dkClamp:
		if o.Alpha <= 0 {
			return ext
		}
		pr.Kind = compose.PrimClamp
	case dkScale:
		if o.V == nil {
			return ext
		}
		pr.Kind = compose.PrimScale
	case dkNorm:
		w := prog.vals[o.Out[0]].Width
		if o.V == nil || prog.vals[o.In[0]].Width != w || p.LayerNorm ||
			prog.scans && o.Role == jlm.RoleAttnNorm && w%bs.parts == 0 ||
			o.Role == jlm.RoleAttnQANorm || o.Role == jlm.RoleAttnKVANorm {
			return ext
		}
		pr.Kind, pr.Eps = compose.PrimNorm, float32(p.RMSEps)
		if o.A.Eps != 0 {
			pr.Eps = float32(o.A.Eps)
		}
	case dkQuant:
		pr.Kind, pr.Win = compose.PrimQuant, p.ActWin
	case dkRopeQ, dkRopeKRow:
		ra := o.A
		iw := prog.vals[o.In[0]].Width
		if ra.HeadDim <= 0 || iw%ra.HeadDim != 0 || ra.NRot <= 0 || ra.NRot%2 != 0 || ra.NRot > ra.HeadDim {
			return ext
		}
		off := pb.table(gaQOff, compose.ValU32)
		if o.Kind == dkRopeKRow {
			off = pb.table(gaKOff, compose.ValU32)
		}
		if o.N > 0 {
			// Rows landing N apart in a table of the op's own.
			if o.Own[0] = g.ownBuf(bs, o, 0, rows, rowTable(rows, o.N, 0)); o.Own[0] == nil {
				return ext
			}
			off = pb.table(gaOwn0, compose.ValU32)
		}
		pr.Kind = compose.PrimRope
		pr.In = []int{int(o.In[0]), pb.table(gaCS, compose.ValF32), off}
		pr.Rope = compose.RopeAttrs{Heads: iw / ra.HeadDim, HD: ra.HeadDim, NRot: ra.NRot,
			Neox: ra.Layout == graph.RopeNeox || ra.Layout == graph.RopeXD, Split: ra.Layout == graph.RopeXD}
	default:
		return ext
	}
	return pr
}

// kvPrim is a history write as a prim: k's row turned into its page (or
// copied, kvCopy), v's row beside it; ext where the history is not paged,
// a row is wider than its heads in a way the prim does not state, or the
// rotary is one it does not turn.
func (g *devTier) kvPrim(o *devOp, prog *devProgram, pb *progBuild, bs *blockScratch, rows int, ext compose.Prim) compose.Prim {
	p := &bs.p
	turn := o.Kind == dkKVWrite
	kw, vw := p.KVRow(), p.NKVHead*p.HeadDim
	if bs.pkv == nil || p.NonCausal || len(o.In) < 2 || prog.vals[o.In[0]].Width != kw || prog.vals[o.In[1]].Width != vw {
		return ext
	}
	heads, whole := p.NKVHead, false
	if kw != vw {
		x := selectOfProg(prog)
		if x == nil || kw != vw+p.HeadDim {
			return ext
		}
		if x.Attr.Pooled {
			whole = true
		} else {
			heads++
		}
	}
	if turn && (p.NRot <= 0 || p.NoPosEnc || p.RopeSplit || p.NRot%2 != 0 || p.NRot > p.HeadDim) {
		return ext
	}
	pr := compose.Prim{Kind: compose.PrimKVWrite,
		In:  []int{int(o.In[0]), int(o.In[1]), pb.table(gaCS, compose.ValF32), pb.table(gaPRow, compose.ValU32), pb.table(gaTab, compose.ValU32)},
		Out: append([]int{pb.table(gaKPool, compose.ValF32), pb.table(gaVPool, compose.ValF32)}, ext.Out...),
		KV: compose.KVAttrs{KW: kw, VW: vw, Page: g.kvPage(), Heads: heads, HD: p.HeadDim, NRot: p.NRot,
			Turn: turn, Neox: p.RopeNeox, Whole: whole, F16: g.KVF16}}
	return pr
}

// matvecPrim is matvec o as a prim: a decode step's or a chunk's
// projection of a resident dense weight, planned under the step's facts and
// the weight's tuned split; ext where it takes another form.
func (g *devTier) matvecPrim(o *devOp, prog *devProgram, ext compose.Prim, bs *blockScratch, rows int) compose.Prim {
	l := g.curLayer
	if g.genRag || o.Heads > 0 || l == nil || o.W < 0 || o.W >= len(l.ms) || *l.rs[o.W] == nil {
		return ext
	}
	// Over a chunk's rows, where the device has a binary16 GEMM or the
	// collective tiles, the staged forms (gemm16Of, genGemmTile) serve.
	if rows > 1 && (g.composeTarget().API == compose.APIPTX && g.f16MMA() != 0 || g.tilesOK() || g.PreferTiles || g.PreferGemmInt8) {
		return ext
	}
	m := *l.ms[o.W]
	if m.experts > 1 || m.pad || m.slots > 0 || kernels.IsFloat(m.q) && o.FIn == 0 {
		return ext
	}
	pr := compose.Prim{Kind: compose.PrimMatVec, In: ext.In, Out: ext.Out, Bias: m.bias != nil, Float: int(o.FIn) - 1,
		W: compose.MatVecWeight{T: m.q, K: m.k, Rows: m.rows}}
	// A latent attention's projection runs alone, as main's: its residual
	// add goes with the norm after it.
	pr.Alone = o.Role == jlm.RoleAttnOut && prog.latent
	// Which epilogue each projection carries, as main's matvec does: the
	// residual on the attention's and the FFN's output projections, the gate
	// on up; none on any other.
	switch o.Role {
	case jlm.RoleAttnOut, jlm.RoleFFNDown:
		pr.Epi = 1 << compose.PrimAdd
	case jlm.RoleFFNUp:
		pr.Epi = 1 << compose.PrimActMul
	default:
		pr.Alone = true
	}
	step := compose.StepDecode
	if rows > 1 {
		step = compose.StepChunk
	}
	pr.Facts = compose.Facts{Step: step, Tok: bs.tok, RowTile: pinned(g.tiles.rowt),
		Split: g.kb.batch.Split, DecodeSplit: max(m.split, 1), DecodeRowt: max(m.rowt, 1),
		RagTok: g.kb.batch.RagTok, RagRowT: g.kb.batch.RagRowT, RagSplit: g.kb.batch.RagSplit,
		MMAMT: g.tiles.mmaMT, MMANT: g.tiles.mmaNT, NoMMA: g.NoMMA, DecodeGroup: m.group}
	return pr
}

// fuseParams is the fusion pass's parameters on this device, main's: a
// residual's sum into its norm, a norm's row into its quantize, and the
// dense FFN's gated activation into its quantize but under NoOperandFuse.
func (g *devTier) fuseParams(rows int) compose.FuseParams {
	var f compose.FuseParams
	f.Allow(compose.PrimAdd, compose.PrimNorm)
	f.Allow(compose.PrimNorm, compose.PrimQuant)
	if g.Config == nil || !g.NoOperandFuse {
		f.AllowIf(compose.PrimActMul, compose.PrimQuant, compose.PairRule{Buf: int(slotAct) + 1})
	}
	// At decode, a projection's residual add and up's gate, as main's matvec
	// carries them: its sole reader, no bias; the gate on a final row, up its
	// second operand.
	if rows == 1 {
		f.AllowIf(compose.PrimMatVec, compose.PrimAdd, compose.PairRule{Sole: true, NoBias: true, Plain: true})
		f.AllowIf(compose.PrimMatVec, compose.PrimActMul, compose.PairRule{Sole: true, NoBias: true, Arg: 2, Final: true})
	}
	return f
}

// siblingParams is the sibling pass's: at decode, up to maxSegs
// contractions over one int8 operand as one segmented launch, a gate and an
// up its product reads apart, as main's; none under NoFuse.
func (g *devTier) siblingParams(rows int) compose.SiblingParams {
	return compose.SiblingParams{Off: rows != 1 || g.Config != nil && g.NoFuse, Max: maxSegs, NoProduct: true}
}

// pruneParams is the dead-store pass's: a projection's row its fused
// epilogue consumed is not stored, as main's matvec stores only the sum.
func (g *devTier) pruneParams() compose.PruneParams {
	var f compose.PruneParams
	f.Drop[compose.PrimMatVec] = true
	return f
}

// groupLaunch is composed group gr's launches as the composer plans them,
// each buffer bound by what it binds to; ok false where the composer cannot
// plan it, a split's planes do not fit, or the backend refused a kernel.
func (g *devTier) groupLaunch(pb *progBuild, nvals int, gr *compose.Group) ([]genLaunch, bool) {
	ls, err := gr.Launches(g.composeTarget())
	if err != nil {
		g.LastErr = err.Error()
		return nil, false
	}
	var out []genLaunch
	for _, cl := range ls {
		if cl.Planes > 0 && (g.genInSub && g.partCap < cl.Planes || !g.genInSub && !g.sizePart(cl.Planes)) {
			return nil, false
		}
		tag := uint32(encRow)
		if cl.Segmented {
			tag = encSegments
		}
		k, _ := g.genKern(append([]uint32{tag}, cl.Key...), cl.Emit)
		if k == nil || len(cl.Binds) > len(genLaunch{}.args) {
			return nil, false
		}
		l := genLaunch{k: k, groups: cl.Groups, width: cl.Width, n: uint8(len(cl.Binds))}
		for j, b := range cl.Binds {
			switch {
			case b.Kind == compose.BindScratch:
				l.args[j] = gaSplit
			case b.Kind == compose.BindVal && b.Index >= nvals:
				l.args[j] = pb.ext[b.Index-nvals] // one of the step's tables
			default:
				l.args[j] = [...]genArg{compose.BindVal: gaVal, compose.BindQ8: gaValQ8A, compose.BindQ8X: gaValQ8X,
					compose.BindVec: gaOpV, compose.BindVecB: gaOpVB, compose.BindConst: gaOpGC,
					compose.BindWQS: gaOpWQS, compose.BindWD: gaOpWD, compose.BindWSC: gaOpWSC, compose.BindWBias: gaOpWBias}[b.Kind]
				l.refs[j] = int32(b.Index)
			}
		}
		out = append(out, l)
	}
	return out, true
}

// mapFnOf is the graph's map as the composer's.
func mapFnOf(f graph.MapFn) (compose.MapFn, bool) {
	switch f {
	case graph.MapAdd:
		return compose.MapAdd, true
	case graph.MapSub:
		return compose.MapSub, true
	case graph.MapMul:
		return compose.MapMul, true
	case graph.MapDiv:
		return compose.MapDiv, true
	case graph.MapMax:
		return compose.MapMax, true
	case graph.MapMin:
		return compose.MapMin, true
	case graph.MapExp:
		return compose.MapExp, true
	case graph.MapSigmoid:
		return compose.MapSigmoid, true
	case graph.MapSoftplus:
		return compose.MapSoftplus, true
	case graph.MapSqrt:
		return compose.MapSqrt, true
	case graph.MapWhere:
		return compose.MapWhere, true
	case graph.MapLess:
		return compose.MapLess, true
	case graph.MapFma:
		return compose.MapFma, true
	}
	return 0, false
}

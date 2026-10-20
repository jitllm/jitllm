package tier

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// composeProgram is prog's ops as the composer's program at rows, op i its
// prim i: an op of a kind the composer forms, with operands it takes, a
// prim; every other an extern.
func (g *devTier) composeProgram(prog *devProgram, bs *blockScratch, rows int) *compose.Program {
	cp := &compose.Program{Rows: rows, Vals: make([]compose.Value, len(prog.vals)), Out: make([]bool, len(prog.vals))}
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
		cp.Prims[i] = g.primOf(&prog.ops[i], prog, bs, rows)
	}
	return cp
}

// primOf is op o as a prim of the composer's program.
func (g *devTier) primOf(o *devOp, prog *devProgram, bs *blockScratch, rows int) compose.Prim {
	vals := func(vs []devVal) []int {
		out := make([]int, len(vs))
		for i, v := range vs {
			out[i] = int(v)
		}
		return out
	}
	ext := compose.Prim{Kind: compose.PrimExtern, In: vals(o.In), Out: vals(o.Out)}
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
	default:
		return ext
	}
	return pr
}

// matvecPrim is matvec o as a prim: a decode step's projection of a
// resident dense weight, planned under the step's facts and the weight's
// tuned split; ext where it takes another form.
func (g *devTier) matvecPrim(o *devOp, prog *devProgram, ext compose.Prim, bs *blockScratch, rows int) compose.Prim {
	l := g.curLayer
	if rows != 1 || g.genRag || o.Heads > 0 || l == nil || o.W < 0 || o.W >= len(l.ms) || *l.rs[o.W] == nil {
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
	pr.Facts = compose.Facts{Step: compose.StepDecode, Tok: bs.tok, RowTile: pinned(g.tiles.rowt),
		Split: g.kb.batch.Split, DecodeSplit: max(m.split, 1), DecodeRowt: max(m.rowt, 1),
		RagTok: g.kb.batch.RagTok, RagRowT: g.kb.batch.RagRowT, RagSplit: g.kb.batch.RagSplit,
		MMAMT: g.tiles.mmaMT, MMANT: g.tiles.mmaNT, NoMMA: g.NoMMA, DecodeGroup: m.group}
	return pr
}

// fuseParams is the fusion pass's parameters on this device, main's: a
// residual's sum into its norm, a norm's row into its quantize, and the
// dense FFN's gated activation into its quantize but under NoOperandFuse.
func (g *devTier) fuseParams() compose.FuseParams {
	var f compose.FuseParams
	f.Allow(compose.PrimAdd, compose.PrimNorm)
	f.Allow(compose.PrimNorm, compose.PrimQuant)
	if g.Config == nil || !g.NoOperandFuse {
		f.AllowIf(compose.PrimActMul, compose.PrimQuant, compose.PairRule{Buf: int(slotAct) + 1})
	}
	// A projection's residual add and up's gate, as main's matvec carries
	// them: its sole reader, no bias; the gate on a final row, up its second
	// operand.
	f.AllowIf(compose.PrimMatVec, compose.PrimAdd, compose.PairRule{Sole: true, NoBias: true, Plain: true})
	f.AllowIf(compose.PrimMatVec, compose.PrimActMul, compose.PairRule{Sole: true, NoBias: true, Arg: 2, Final: true})
	return f
}

// pruneParams is the dead-store pass's: a projection's row its fused
// epilogue consumed is not stored, as main's matvec stores only the sum.
func (g *devTier) pruneParams() compose.PruneParams {
	var f compose.PruneParams
	f.Drop[compose.PrimMatVec] = true
	return f
}

// groupLaunch is composed group gr's launch, its buffers bound by what
// they bind to; ok false where the composer cannot plan it or the backend
// refused it.
func (g *devTier) groupLaunch(gr *compose.Group) ([]genLaunch, bool) {
	bind := func(l *genLaunch, j int, b compose.Bind) {
		l.args[j] = [...]genArg{compose.BindVal: gaVal, compose.BindQ8: gaValQ8A, compose.BindQ8X: gaValQ8X,
			compose.BindVec: gaOpV, compose.BindVecB: gaOpVB, compose.BindConst: gaOpGC,
			compose.BindWQS: gaWQS, compose.BindWD: gaWD, compose.BindWSC: gaWSC, compose.BindWBias: gaMBias}[b.Kind]
		l.refs[j] = int32(b.Index)
	}
	if len(gr.Binds) > len(genLaunch{}.args) {
		return nil, false
	}
	if !gr.Contract {
		k := g.rowKern(&gr.Row)
		if k.k == nil {
			return nil, false
		}
		l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(gr.Binds))}
		for j, b := range gr.Binds {
			bind(&l, j, b)
		}
		return []genLaunch{l}, true
	}
	// A contraction: its plan's launches, a k split's planes in the split
	// scratch and their sum.
	f := gr.Facts
	f.Target = g.composeTarget()
	ps, err := gr.Row.Plan(f)
	if err != nil {
		g.LastErr = err.Error()
		return nil, false
	}
	var out []genLaunch
	for _, p := range ps {
		s := p.S
		if s.Split > 1 {
			need := gr.Row.Width * gr.Row.Rows * s.Split
			if g.genInSub && g.partCap < need || !g.genInSub && !g.sizePart(need) {
				return nil, false
			}
		}
		k, _ := g.genKern(append([]uint32{encRow}, p.G.Encode(s)...), func() (*ir.Kernel, error) { return p.G.Emit(s) })
		if k == nil || len(p.Bufs) > len(genLaunch{}.args) {
			return nil, false
		}
		l := genLaunch{k: k, groups: (p.Threads + s.Group - 1) / s.Group, width: s.Group, n: uint8(len(p.Bufs))}
		for i, b := range p.Bufs {
			if b < 0 {
				l.args[i] = gaSplit
				continue
			}
			bind(&l, i, gr.Binds[b])
		}
		out = append(out, l)
	}
	return out, true
}

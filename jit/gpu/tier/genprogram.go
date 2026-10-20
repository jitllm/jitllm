package tier

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/format/jlm"
)

// composeProgram is prog's ops as the composer's program at rows, op i its
// prim i: an op of a kind the composer forms, with operands it takes, a
// prim; every other an extern.
func (g *devTier) composeProgram(prog *devProgram, bs *blockScratch, rows int) *compose.Program {
	cp := &compose.Program{Rows: rows, Vals: make([]compose.Value, len(prog.vals))}
	for v, t := range prog.vals {
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

// fuseParams is the fusion pass's parameters on this device, main's: a
// residual's sum into its norm, a norm's row into its quantize, and the
// dense FFN's gated activation into its quantize but under NoOperandFuse.
func (g *devTier) fuseParams() compose.FuseParams {
	var f compose.FuseParams
	f.Allow(compose.PrimAdd, compose.PrimNorm)
	f.Allow(compose.PrimNorm, compose.PrimQuant)
	if g.Config == nil || !g.NoOperandFuse {
		f.AllowIn(compose.PrimActMul, compose.PrimQuant, int(slotAct))
	}
	return f
}

// groupLaunch is composed group gr's launch, its buffers bound by what
// they bind to; ok false where the composer cannot plan it or the backend
// refused it.
func (g *devTier) groupLaunch(gr *compose.Group) (genLaunch, bool) {
	k := g.rowKern(&gr.Row)
	if k.k == nil || len(gr.Binds) > len(genLaunch{}.args) {
		return genLaunch{}, false
	}
	l := genLaunch{k: k.k, groups: k.groups, width: k.group, n: uint8(len(gr.Binds))}
	for j, b := range gr.Binds {
		l.args[j] = [...]genArg{compose.BindVal: gaVal, compose.BindQ8: gaValQ8A, compose.BindQ8X: gaValQ8X,
			compose.BindVec: gaOpV, compose.BindVecB: gaOpVB, compose.BindConst: gaOpGC}[b.Kind]
		l.refs[j] = int32(b.Index)
	}
	return l, true
}

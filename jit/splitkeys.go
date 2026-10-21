package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// SplitKeys splits a normalized exponential reduction -- a maximum over
// terms, the terms' exp less it, their sum and their weighted sum, the
// latter divided by the former -- into partitions of each output's live
// span: each partition's maximum, sum and weighted sum into a partial, a
// partition's span its share of the live span rounded up to Align, the
// partial the weighted sums padded to Pad then Copies copies of the maximum
// and of the sum; and a merge weighting each partition by exp of its
// maximum less the greatest.
type SplitKeys struct {
	Align, Pad, Copies int
}

func (SplitKeys) Name() string { return "splitkeys" }

func (s SplitKeys) Apply(prog []*Nest, t Target) ([]*Nest, error) {
	var out []*Nest
	for i := 0; i < len(prog); i++ {
		if i+5 <= len(prog) {
			if ns, ok := s.split(prog[i:i+5], t); ok {
				for _, n := range ns {
					n.Passes = append(n.Passes, s.Name())
				}
				out = append(out, ns...)
				i += 4
				continue
			}
		}
		out = append(out, prog[i])
	}
	return out, nil
}

// builder is a primitive's operands and function being written.
type builder struct {
	x   exprs
	ins []Operand
}

func (b *builder) operand(o Operand) int {
	for i, x := range b.ins {
		if x == o {
			return i
		}
	}
	b.ins = append(b.ins, o)
	return len(b.ins) - 1
}

func (b *builder) load(o Operand, ix int) int {
	return b.x.add(Expr{Op: ExprLoad, In: b.operand(o), A: ix})
}

// access is index ix through a.
func (b *builder) access(ix int, a Access) int {
	x := &b.x
	if a.Mul > 1 {
		ix = x.op(OpMul, ir.U32, ix, x.u(uint32(a.Mul)))
	}
	if a.Div > 1 {
		ix = x.op(OpDiv, ir.U32, ix, x.u(uint32(a.Div)))
	}
	if a.Mod > 0 {
		ix = x.op(OpRem, ir.U32, ix, x.u(uint32(a.Mod)))
	}
	if a.Off != 0 {
		ix = x.op(OpAdd, ir.U32, ix, x.u(uint32(a.Off)))
	}
	return ix
}

// emit is p's function at element el, a read of an operand in sub its
// node at the read's index.
func (b *builder) emit(p Prim, exprs []Expr, el int, sub map[Operand]func(ix int) int) int {
	vals := make([]int, len(exprs))
	read := func(j, ix int) int {
		if f, ok := sub[p.In[j]]; ok {
			return f(ix)
		}
		return b.load(p.In[j], ix)
	}
	for i, x := range exprs {
		switch x.Op {
		case ExprElem:
			vals[i] = el
		case ExprIn:
			vals[i] = read(x.In, b.access(el, p.At[x.In]))
		case ExprLoad:
			vals[i] = read(x.In, vals[x.A])
		default:
			e := x
			e.A, e.B, e.C = vals[x.A], vals[x.B], vals[x.C]
			vals[i] = b.x.add(e)
		}
	}
	return vals[len(vals)-1]
}

func (s SplitKeys) split(ch []*Nest, t Target) ([]*Nest, bool) {
	for _, n := range ch {
		if n.Prim == nil {
			return nil, false
		}
	}
	mx, wt, sm, vl, out := ch[0].Prim, ch[1].Prim, ch[2].Prim, ch[3].Prim, ch[4].Prim
	if mx.Kind != PrimReduce || mx.Combine != OpMax || mx.Heads == nil || mx.Span == nil ||
		wt.Kind != PrimMap || !reads(wt, mx.Out) || !hasOp(wt.Expr, OpExp) ||
		sm.Kind != PrimReduce || sm.Combine != OpAdd || !reads(sm, wt.Out) || sm.Span == nil ||
		vl.Kind != PrimReduce || vl.Combine != OpAdd || !reads(vl, wt.Out) || vl.Span == nil ||
		out.Kind != PrimMap || !reads(out, vl.Out) || !reads(out, sm.Out) {
		return nil, false
	}
	h := *mx.Heads
	C, O := mx.K, mx.N
	D := vl.N / O
	S := flashGeometry(h).splits(h, C, t.Slots)
	if S < 2 || sm.N != O || sm.K != C || vl.K != C || wt.N != O*C || out.N != O*D {
		return nil, false
	}
	U, F := ir.U32, ir.F32
	stride := (D+s.Pad-1)/s.Pad*s.Pad + 2*s.Copies
	part := Operand{Kind: OperandScratch, Name: out.Out.Name + ".part"}
	maxS := Operand{Kind: OperandScratch, Name: out.Out.Name + ".max.s", Temp: true}
	sumS := Operand{Kind: OperandScratch, Name: out.Out.Name + ".sum.s", Temp: true}
	minus := int64(math.Float32bits(math.Float32frombits(0xff7fffff)))

	// per is output o's partition span: its live span over S, rounded up.
	per := func(b *builder, p Prim, o int) int {
		x := &b.x
		span := b.emit(p, p.Span, o, nil)
		share := x.op(OpDiv, U, x.op(OpAdd, U, span, x.u(uint32(S-1))), x.u(uint32(S)))
		return x.op(OpMul, U, x.op(OpDiv, U, x.op(OpAdd, U, share, x.u(uint32(s.Align-1))), x.u(uint32(s.Align))), x.u(uint32(s.Align)))
	}
	// partial is reduction p over partition sp of output o's terms: each
	// term outside it its start.
	partial := func(p *Prim, inner int, out Operand, at func(b *builder, o, sp int) int) Prim {
		var b builder
		x := &b.x
		el := x.add(Expr{Op: ExprElem})
		o2, k := x.op(OpDiv, U, el, x.u(uint32(C))), x.op(OpRem, U, el, x.u(uint32(C)))
		o := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpDiv, U, o2, x.u(uint32(S*inner))), x.u(uint32(inner))), x.op(OpRem, U, o2, x.u(uint32(inner))))
		sp := x.op(OpRem, U, x.op(OpDiv, U, o2, x.u(uint32(inner))), x.u(uint32(S)))
		pr := per(&b, *p, o)
		lo := x.op(OpMul, U, sp, pr)
		in := x.op(OpLess, ir.Pred, x.op(OpSub, U, k, lo), pr)
		term := b.emit(*p, p.Expr, x.op(OpAdd, U, x.op(OpMul, U, o, x.u(uint32(C))), k), nil)
		start := x.add(Expr{Op: OpConst, Type: elemOf(p.Type), Imm: p.Init})
		x.sel(elemOf(p.Type), in, term, start)
		q := Prim{Kind: PrimReduce, N: p.N * S, K: C, Out: out, Combine: p.Combine, Init: p.Init, Type: p.Type, Expr: x.e, In: b.ins}
		if at != nil {
			var sb builder
			e := sb.x.add(Expr{Op: ExprElem})
			o3 := sb.x.op(OpAdd, U, sb.x.op(OpMul, U, sb.x.op(OpDiv, U, e, sb.x.u(uint32(S*inner))), sb.x.u(uint32(inner))), sb.x.op(OpRem, U, e, sb.x.u(uint32(inner))))
			sp3 := sb.x.op(OpRem, U, sb.x.op(OpDiv, U, e, sb.x.u(uint32(inner))), sb.x.u(uint32(S)))
			at(&sb, o3, sp3)
			q.Index = sb.x.e
		}
		return q
	}
	m := partial(mx, 1, maxS, nil)

	// The weights: each term's exp less its partition's maximum.
	var wb builder
	we := wb.x.add(Expr{Op: ExprElem})
	wo, wk := wb.x.op(OpDiv, U, we, wb.x.u(uint32(C))), wb.x.op(OpRem, U, we, wb.x.u(uint32(C)))
	wsp := wb.x.op(OpDiv, U, wk, per(&wb, *mx, wo))
	ms := func(ix int) int {
		return wb.load(maxS, wb.x.op(OpAdd, U, wb.x.op(OpMul, U, wo, wb.x.u(uint32(S))), wsp))
	}
	wb.emit(*wt, wt.Expr, we, map[Operand]func(int) int{mx.Out: ms})
	weights := Prim{Kind: PrimMap, N: O * C, Out: wt.Out, In: wb.ins, Expr: wb.x.e}

	l := partial(sm, 1, sumS, nil)
	at := func(b *builder, o, sp, off int) int {
		x := &b.x
		return x.op(OpAdd, U, x.op(OpMul, U, x.op(OpAdd, U, x.op(OpMul, U, o, x.u(uint32(S))), sp), x.u(uint32(stride))), x.u(uint32(off)))
	}
	n := partial(vl, D, part, func(b *builder, o, sp int) int {
		x := &b.x
		rh, d := x.op(OpDiv, U, o, x.u(uint32(D))), x.op(OpRem, U, o, x.u(uint32(D)))
		return x.op(OpAdd, U, at(b, rh, sp, 0), d)
	})

	// The copies: each partition's maximum and sum, Copies times each.
	var cb builder
	ce := cb.x.add(Expr{Op: ExprElem})
	osp, c := cb.x.op(OpDiv, U, ce, cb.x.u(uint32(s.Copies))), cb.x.op(OpRem, U, ce, cb.x.u(uint32(s.Copies)))
	ro, rsp := cb.x.op(OpDiv, U, osp, cb.x.u(uint32(S))), cb.x.op(OpRem, U, osp, cb.x.u(uint32(S)))
	mv, lv := cb.load(maxS, osp), cb.load(sumS, osp)
	mat := cb.x.op(OpAdd, U, at(&cb, ro, rsp, stride-2*s.Copies), c)
	lat := cb.x.op(OpAdd, U, at(&cb, ro, rsp, stride-s.Copies), c)
	copies := Prim{Kind: PrimMap, N: O * S * s.Copies, In: cb.ins, Expr: cb.x.e,
		Stores: []PrimStore{{Out: part, Index: mat, Value: mv}, {Out: part, Index: lat, Value: lv}}}

	// The merge: the partitions weighted by exp of their maximum less the
	// greatest, the original normalization over the weighted sums.
	var mb builder
	me := mb.x.add(Expr{Op: ExprElem})
	mrh, md := mb.x.op(OpDiv, U, me, mb.x.u(uint32(D))), mb.x.op(OpRem, U, me, mb.x.u(uint32(D)))
	big := mb.x.add(Expr{Op: OpConst, Type: F, Imm: minus})
	maxes, sums, nums := make([]int, S), make([]int, S), make([]int, S)
	for sp := 0; sp < S; sp++ {
		spn := mb.x.u(uint32(sp))
		maxes[sp] = mb.load(part, at(&mb, mrh, spn, stride-2*s.Copies))
		sums[sp] = mb.load(part, at(&mb, mrh, spn, stride-s.Copies))
		big = mb.x.op(OpMax, F, big, maxes[sp])
	}
	L, N := mb.x.f(0), mb.x.f(0)
	for sp := 0; sp < S; sp++ {
		e := mb.x.un(OpExp, F, mb.x.op(OpSub, F, maxes[sp], big))
		L = mb.x.add(Expr{Op: OpFma, Type: F, A: sums[sp], B: e, C: L})
		nums[sp] = mb.load(part, mb.x.op(OpAdd, U, at(&mb, mrh, mb.x.u(uint32(sp)), 0), md))
		N = mb.x.add(Expr{Op: OpFma, Type: F, A: nums[sp], B: e, C: N})
	}
	mb.emit(*out, out.Expr, me, map[Operand]func(int) int{vl.Out: func(int) int { return N }, sm.Out: func(int) int { return L }})
	merge := Prim{Kind: PrimMap, N: O * D, Out: out.Out, In: mb.ins, Expr: mb.x.e}

	var ns []*Nest
	for _, p := range []Prim{m, weights, l, n, copies, merge} {
		lowered, err := lower(p)
		if err != nil {
			return nil, false
		}
		for _, x := range lowered {
			x.Kind = p.Kind
			q := p
			x.Prim = &q
			ns = append(ns, x)
		}
	}
	return ns, true
}

func reads(p *Prim, o Operand) bool {
	for _, x := range p.In {
		if x == o {
			return true
		}
	}
	return false
}

func hasOp(es []Expr, op Op) bool {
	for _, e := range es {
		if e.Op == op {
			return true
		}
	}
	return false
}

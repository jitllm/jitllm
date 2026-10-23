package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// SplitExtreme cuts a loop finding the largest of its terms and the value
// carried with it -- an argmax -- into contiguous segments, each element's
// segments threads of their own writing a plane of their largest terms and
// a plane of the values carried with them, and a combine that runs the
// planes in order the way the loop ran its terms: a later segment wins only
// where it is strictly larger, so the first of equal terms keeps its value.
// The segments are the most, at most Parts and a power of two, that divide
// the trips.
type SplitExtreme struct {
	Parts int
	Kinds []PrimKind
}

func (SplitExtreme) Name() string { return "splitextreme" }

func (s SplitExtreme) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(s.Name(), ks, s.apply)
}

func (s SplitExtreme) apply(n *Kernel) ([]*Kernel, bool, error) {
	if n.Work || !covers(s.Kinds, n.Kind) {
		return nil, false, nil
	}
	c, ext, car, vs, cs, ok := extremeSplittable(n)
	if !ok {
		return nil, false, nil
	}
	T := n.Loops[c].Trip
	S := s.Parts
	for S > 1 && T%S != 0 {
		S /= 2
	}
	if S <= 1 {
		return nil, false, nil
	}
	vst, cst := n.Stores[vs], n.Stores[cs]
	vt, ct := n.Nodes[ext].Type, n.Nodes[car].Type
	vout, cout := n.Bufs[vst.Buf].Operand, n.Bufs[cst.Buf].Operand
	vplanes := Operand{Kind: OperandScratch, Name: n.Name + "." + vout.Name + ".planes", Of: vout.Of}
	cplanes := Operand{Kind: OperandScratch, Name: n.Name + "." + cout.Name + ".planes", Of: cout.Of, Elem: ct}
	shape := append([]int{S}, n.Space...)

	p := n.clone()
	p.Name = n.Name + ".part"
	p.Space = shape
	for i := range p.Nodes {
		if p.Nodes[i].Op == OpCoord {
			p.Nodes[i].A++
		}
	}
	seg := p.add(Node{Op: OpCoord, A: 0, Type: ir.U32})
	p.Loops[c].Trip = T / S
	if it := p.iterOf(c); it >= 0 {
		k := p.bin(OpAdd, ir.U32, p.bin(OpMul, ir.U32, seg, p.u32(T/S)), it)
		p.redirect(it, k, k)
	}
	vb := p.buf(Buf{Kind: BufOut, Elem: vt, Operand: vplanes, Shape: shape})
	cb := p.buf(Buf{Kind: BufOut, Elem: ct, Operand: cplanes, Shape: shape})
	p.Stores[vs] = Store{Buf: vb, Val: ext, At: -1, Ax: append([]int{seg}, vst.Ax...)}
	p.Stores[cs] = Store{Buf: cb, Val: car, At: -1, Ax: append([]int{seg}, cst.Ax...)}

	m := &Kernel{Name: n.Name + ".combine", Kind: n.Kind, Space: append([]int(nil), n.Space...), Group: n.Group, Part: n.Part, Of: n.Of}
	coords := make([]int, len(m.Space))
	for i := range coords {
		coords[i] = m.add(Node{Op: OpCoord, A: i, Type: ir.U32})
	}
	vin := m.buf(Buf{Kind: BufIn, Elem: vt, Operand: vplanes, Shape: shape})
	cin := m.buf(Buf{Kind: BufIn, Elem: ct, Operand: cplanes, Shape: shape})
	l := m.loop(S, -1)
	it := m.add(Node{Op: OpIter, At: l, Type: ir.U32})
	at := append([]int{it}, coords...)
	x := m.add(Node{Op: OpLoad, Type: vt, Buf: vin, Ax: at})
	y := m.add(Node{Op: OpLoad, Type: ct, Buf: cin, Ax: append([]int(nil), at...)})
	acc := m.add(Node{Op: OpFold, At: l, Type: vt, B: m.f32(math.Float32bits(float32(math.Inf(-1))))})
	m.Nodes[acc].A = m.bin(OpMax, vt, x, acc)
	carried := m.add(Node{Op: OpFold, At: l, Type: ct, B: m.add(Node{Op: OpConst, Type: ct})})
	won := m.add(Node{Op: OpLess, Type: ir.Pred, A: acc, B: x})
	m.Nodes[carried].A = m.add(Node{Op: OpSelect, Type: ct, A: won, B: y, C: carried})
	out := make([]Store, len(n.Stores))
	out[vs] = Store{Buf: m.buf(n.Bufs[vst.Buf]), Val: acc, At: -1, Ax: coords}
	out[cs] = Store{Buf: m.buf(n.Bufs[cst.Buf]), Val: carried, At: -1, Ax: append([]int(nil), coords...)}
	m.Stores = out
	return []*Kernel{p, m}, true, nil
}

// extremeSplittable finds an element's argmax: a top-level loop whose only
// folds are a largest term and a value carried with it, each stored at the
// element's own place, the kernel's only two stores. vs and cs are the
// stores of the largest term and of the carried value.
func extremeSplittable(n *Kernel) (c, ext, car, vs, cs int, ok bool) {
	if len(n.Stores) != 2 {
		return
	}
	for _, s := range n.Stores {
		if s.At != -1 || !n.storesOwn(s) {
			return
		}
	}
	car, ext = n.Stores[1].Val, n.Stores[0].Val
	vs, cs = 0, 1
	if n.carriedBy(car) != ext {
		car, ext, vs, cs = ext, car, 1, 0
	}
	if n.Nodes[car].Op != OpFold || n.Nodes[ext].Op != OpFold || n.carriedBy(car) != ext ||
		n.Nodes[n.Nodes[ext].A].Op != OpMax || n.Nodes[n.Nodes[car].A].Op != OpSelect {
		return
	}
	c = n.Nodes[ext].At
	if n.Nodes[car].At != c || n.Loops[c].Parent != -1 || n.Loops[c].Count != 0 || len(n.foldsOf(c)) != 2 {
		return
	}
	return c, ext, car, vs, cs, true
}

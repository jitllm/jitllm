package jit

import "github.com/jitllm/jitllm/jit/ir"

// Split cuts a sum over a loop into contiguous segments, each element's
// segments threads of their own writing partial planes, and a combine that
// sums the planes. The segments double while they stay under Max, the
// threads under twice Slots (the device's where Slots is 0), and the trips
// divide into Block-wide blocks a segment. Where Parts is set the segments
// are instead the most, at most Parts and a power of two, that divide the
// trips.
type Split struct {
	Slots, Max, Block int
	Parts             int
	Kinds             []PrimKind
}

func (Split) Name() string { return "split" }

func (s Split) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(s.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) { return s.apply(k, d) })
}

func (s Split) apply(n *Kernel, d Device) ([]*Kernel, bool, error) {
	c, f, term, st, ok := splittable(n)
	if !ok || n.Work || !covers(s.Kinds, n.Kind) {
		return nil, false, nil
	}
	T, E := n.Loops[c].Trip, n.Elements()
	slots := s.Slots
	if slots == 0 {
		slots = d.Slots
	}
	S := 1
	if s.Parts > 0 {
		S = s.Parts
		for S > 1 && T%S != 0 {
			S /= 2
		}
	}
	for s.Parts == 0 && S < s.Max && E*S < 2*slots && T%(s.Block*S*2) == 0 {
		S *= 2
	}
	if S == 1 {
		return nil, false, nil
	}
	out := n.Bufs[n.Stores[st].Buf].Operand
	planes := Operand{Kind: OperandScratch, Name: out.Name + ".planes", Of: out.Of}
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
	// The planes hold the sum's own type: an integer dot's sum is summed
	// as integers.
	t := n.Nodes[f].Type
	zero := func(k *Kernel) int { return k.add(Node{Op: OpConst, Type: t}) }
	b := p.buf(Buf{Kind: BufOut, Elem: t, Operand: planes, Shape: shape})
	val := f
	if term >= 0 {
		first := p.add(Node{Op: OpLess, Type: ir.Pred, A: seg, B: p.u32(1)})
		val = p.bin(OpAdd, t, f, p.add(Node{Op: OpSelect, Type: t, A: first, B: term, C: zero(p)}))
	}
	ax := append([]int{seg}, p.Stores[st].Ax...)
	p.Stores[st] = Store{Buf: b, Val: val, At: -1, Ax: ax}

	m := &Kernel{Name: n.Name + ".combine", Kind: n.Kind, Space: append([]int(nil), n.Space...), Group: n.Group, Part: n.Part, Of: n.Of}
	coords := make([]int, len(m.Space))
	for i := range coords {
		coords[i] = m.add(Node{Op: OpCoord, A: i, Type: ir.U32})
	}
	in := m.buf(Buf{Kind: BufIn, Elem: t, Operand: planes, Shape: shape})
	ob := m.buf(n.Bufs[n.Stores[st].Buf])
	l := m.loop(S, -1)
	it := m.add(Node{Op: OpIter, At: l, Type: ir.U32})
	acc := m.add(Node{Op: OpFold, At: l, Type: t, B: zero(m)})
	x := m.add(Node{Op: OpLoad, Type: t, Buf: in, Ax: append([]int{it}, coords...)})
	m.Nodes[acc].A = m.bin(OpAdd, t, acc, x)
	m.Stores = append(m.Stores, Store{Buf: ob, Val: acc, At: -1, Ax: coords})
	return []*Kernel{p, m}, true, nil
}

// splittable finds the element's sum: a top-level loop with one fold, a sum,
// which the element's one store writes at its own place, alone or with a
// term beside it the loop does not change (-1 for none).
func splittable(n *Kernel) (c, f, term, st int, ok bool) {
	if len(n.Stores) != 1 || n.Stores[0].At != -1 || !n.storesOwn(n.Stores[0]) {
		return 0, 0, 0, 0, false
	}
	f, term = n.Stores[0].Val, -1
	if v := n.Nodes[f]; v.Op == OpAdd {
		switch {
		case n.Nodes[v.A].Op == OpFold:
			f, term = v.A, v.B
		case n.Nodes[v.B].Op == OpFold:
			f, term = v.B, v.A
		}
	}
	nd := n.Nodes[f]
	if nd.Op != OpFold || n.Loops[nd.At].Parent != -1 || len(n.foldsOf(nd.At)) != 1 || !n.sumFold(f) ||
		term >= 0 && n.Varies(term, nd.At) {
		return 0, 0, 0, 0, false
	}
	return nd.At, f, term, 0, true
}

// Blocks runs a sum loop Size trips at a time: each block's trips unrolled
// as one chain from zero, the chain added into the sum once a block.
type Blocks struct {
	Size  int
	Kinds []PrimKind
}

func (Blocks) Name() string { return "blocks" }

func (b Blocks) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(b.Name(), ks, b.apply)
}

func (b Blocks) apply(n *Kernel) ([]*Kernel, bool, error) {
	B := b.Size
	if !covers(b.Kinds, n.Kind) {
		return nil, false, nil
	}
	var r *Kernel
	for c := range n.Loops {
		folds := n.foldsOf(c)
		it := n.iterOf(c)
		if B <= 1 || n.Loops[c].Count > 0 || n.Loops[c].Trip%B != 0 || len(folds) != 1 || it < 0 || !n.sumFold(folds[0]) {
			continue
		}
		if r == nil {
			r = n.clone()
		}
		f := folds[0]
		step := r.Nodes[f].A
		r.Loops[c].Trip /= B
		base := r.bin(OpMul, ir.U32, it, r.u32(B))
		chain := r.f32(0)
		for j := 0; j < B; j++ {
			k := base
			if j > 0 {
				k = r.bin(OpAdd, ir.U32, base, r.u32(j))
			}
			chain = r.copyOf(step, map[int]int{f: chain, it: k})
		}
		r.Nodes[f].A = r.bin(OpAdd, r.Nodes[f].Type, f, chain)
	}
	if r == nil {
		return nil, false, nil
	}
	return []*Kernel{r}, true, nil
}

func covers(kinds []PrimKind, k PrimKind) bool {
	for _, x := range kinds {
		if x == k {
			return true
		}
	}
	return false
}

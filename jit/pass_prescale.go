package jit

import "github.com/jitllm/jitllm/jit/ir"

// Prescale scales a contraction's terms rather than its results where the
// one map reading them multiplies each by a factor the contracted axis does
// not move: a map over the weights the contraction reads, over its range,
// stores them times the factor, and the contraction reads those and writes
// what the map wrote. The sum distributes over the factor.
type Prescale struct{}

func (Prescale) Name() string { return "prescale" }

func (p Prescale) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	out := append([]*Kernel(nil), ks...)
	for ki := 0; ki < len(out); ki++ {
		k := out[ki]
		w, vals, ok := k.contractsWeights()
		if !ok || readers(out, vals) != 1 {
			continue
		}
		mi := -1
		for j := ki + 1; j < len(out); j++ {
			if usesOperand(out[j], vals) {
				mi = j
				break
			}
		}
		if mi < 0 {
			continue
		}
		m := out[mi]
		g, dst, ok := m.scalesBy(vals)
		if !ok || !sameSpace(k, m) {
			continue
		}
		n, scaled := scaleMap(k, m, w, g)
		kc := k.clone()
		for b := range kc.Bufs {
			switch kc.Bufs[b].Operand {
			case w.Operand:
				kc.Bufs[b].Operand = scaled
			case vals:
				kc.Bufs[b].Operand = dst
			}
		}
		n.Passes = append(append([]string(nil), m.Passes...), p.Name())
		kc.Passes = append(kc.Passes, p.Name())
		rest := append(append([]*Kernel(nil), out[ki+1:mi]...), out[mi+1:]...)
		out = append(append(append([]*Kernel(nil), out[:ki]...), n, kc), rest...)
		ki++
	}
	return out, nil
}

// readers is how many kernels of ks load operand o.
func readers(ks []*Kernel, o Operand) int {
	n := 0
	for _, k := range ks {
		for _, nd := range k.Nodes {
			if nd.Op == OpLoad && k.Bufs[nd.Buf].Operand == o {
				n++
				break
			}
		}
	}
	return n
}

// contractsWeights is, for a contraction of one loop summing the product of
// a load at its leading coordinates and the loop's iteration with another,
// that load's buffer and the operand the sum is stored to.
func (k *Kernel) contractsWeights() (Buf, Operand, bool) {
	f, ok := k.fold()
	if k.Kind != PrimContract || !ok || len(k.Stores) != 1 || k.Stores[0].Val != f || k.Stores[0].At != -1 {
		return Buf{}, Operand{}, false
	}
	st := k.Nodes[k.Nodes[f].A]
	if st.Op != OpFma || st.C != f {
		return Buf{}, Operand{}, false
	}
	a := k.Nodes[st.A]
	it := k.iterOf(0)
	if a.Op != OpLoad || len(a.Ax) != 3 || a.Ax[2] != it {
		return Buf{}, Operand{}, false
	}
	for i, x := range a.Ax[:2] {
		if nd := k.Nodes[x]; nd.Op != OpCoord || nd.A != i {
			return Buf{}, Operand{}, false
		}
	}
	return k.Bufs[a.Buf], k.Bufs[k.Stores[0].Buf].Operand, true
}

// scalesBy is, for a map storing each element of v at its own place times a
// factor that reads neither v nor its last coordinate, the factor and the
// operand stored.
func (m *Kernel) scalesBy(v Operand) (int, Operand, bool) {
	if m.Kind != PrimMap || len(m.Loops) > 0 || len(m.Stores) != 1 || !m.readsOwn(v) {
		return 0, Operand{}, false
	}
	s := m.Stores[0]
	val := m.Nodes[s.Val]
	if val.Op != OpMul {
		return 0, Operand{}, false
	}
	last := len(m.Space) - 1
	for _, pr := range [][2]int{{val.A, val.B}, {val.B, val.A}} {
		x := m.Nodes[pr[0]]
		if x.Op != OpLoad || m.Bufs[x.Buf].Operand != v {
			continue
		}
		moves := false
		var walk func(i int)
		walk = func(i int) {
			nd := m.Nodes[i]
			if nd.Op == OpCoord && nd.A == last || nd.Op == OpLoad && m.Bufs[nd.Buf].Operand == v {
				moves = true
			}
			for _, a := range m.Operands(&nd) {
				walk(*a)
			}
		}
		walk(pr[1])
		if !moves {
			return pr[1], m.Bufs[s.Buf].Operand, true
		}
	}
	return 0, Operand{}, false
}

// scaleMap is the map storing each weight of w, over k's range of its last
// axis, times m's factor g: the map and the operand it stores.
func scaleMap(k, m *Kernel, w Buf, g int) (*Kernel, Operand) {
	space := append(append([]int(nil), k.Space[:2]...), k.Loops[0].Trip)
	n := &Kernel{Name: "map", Kind: PrimMap, Space: space, Heads: k.Heads, Part: k.Part, Of: k.Of}
	coords := make([]int, 3)
	for i := range coords {
		coords[i] = n.add(Node{Op: OpCoord, A: i, Type: ir.U32})
	}
	leaf := func(i int, nd Node) (int, bool) {
		if nd.Op == OpCoord {
			return coords[nd.A], true
		}
		return 0, false
	}
	x := n.add(Node{Op: OpLoad, Type: w.Elem, Buf: n.bufLike(w), Ax: append([]int(nil), coords...)})
	val := n.bin(OpMul, w.Elem, x, copyInto(n, m, g, leaf))
	scaled := w.Operand
	scaled.Name += ".scaled"
	ob := n.buf(Buf{Kind: BufOut, Elem: w.Elem, Operand: scaled, Shape: append([]int(nil), space...)})
	n.Stores = append(n.Stores, Store{Buf: ob, Val: val, At: -1, Ax: append([]int(nil), coords...)})
	if pos := k.Positions; pos != nil {
		n.Positions = &Positions{Axis: 2, Loop: -1, Align: 1, First: copyInto(n, k, pos.First, leaf), End: copyInto(n, k, pos.End, leaf)}
	}
	return n, scaled
}

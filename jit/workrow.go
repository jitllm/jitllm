package jit

import "github.com/jitllm/jitllm/jit/ir"

// WorkRow runs a row's reduction and the map that reads it as one workgroup
// a row: each of their loops over the row spread over the Group threads,
// a thread's places clamped to the row and its sum masked past it, the sum
// combined over the workgroup.
type WorkRow struct {
	Group int
}

func (WorkRow) Name() string { return "workrow" }

func (w WorkRow) Apply(prog []*Nest, t Target) ([]*Nest, error) {
	out := append([]*Nest(nil), prog...)
	for ri := 0; ri < len(out); ri++ {
		r := out[ri]
		c, f, sum, ok := rowSum(r)
		if !ok {
			continue
		}
		for mi := ri + 1; mi < len(out); mi++ {
			m := out[mi]
			if m.Kind == PrimMap && len(m.Space) == 1 && m.Space[0] == r.Space[0]*r.Loops[c].Trip && m.reads(sum) && soleReader(out, m, sum) {
				out[ri] = w.join(r, c, f, m, sum)
				out[ri].Passes = append(append([]string(nil), r.Passes...), w.Name())
				out = append(out[:mi], out[mi+1:]...)
				break
			}
			if conflicts(r, m) {
				break
			}
		}
	}
	return out, nil
}

// rowSum finds a reduction's nest: a thread a row, one loop over the row with
// one sum, stored at the row into scratch.
func rowSum(n *Nest) (c, f int, sum Operand, ok bool) {
	if n.Kind != PrimReduce || len(n.Space) != 1 || len(n.Loops) != 1 || len(n.Stores) != 1 {
		return 0, 0, Operand{}, false
	}
	st := n.Stores[0]
	f = st.Val
	if nd := n.Nodes[f]; nd.Op != OpFold || !n.sumFold(f) || !n.atElement(st.Ix) || n.Bufs[st.Buf].Operand.Kind != OperandScratch {
		return 0, 0, Operand{}, false
	}
	return n.Nodes[f].At, f, n.Bufs[st.Buf].Operand, true
}

func soleReader(prog []*Nest, m *Nest, o Operand) bool {
	for _, n := range prog {
		if n != m && n.reads(o) {
			return false
		}
	}
	return true
}

func (w WorkRow) join(r *Nest, c, f int, m *Nest, sum Operand) *Nest {
	K := r.Loops[c].Trip
	n := r.clone()
	n.Name = r.Name + "+" + m.Name
	n.Work, n.Group = true, w.Group
	n.Stores = nil
	n.Loops[c].Spread = true
	if it := n.iterOf(c); it >= 0 {
		at := n.bin(OpMin, ir.U32, it, n.u32(K-1))
		n.redirect(it, at, at)
		live := n.add(Node{Op: OpLess, Type: ir.Pred, A: it, B: n.u32(K)})
		step := &n.Nodes[n.Nodes[f].A]
		other := step.B
		if other == f {
			other = step.A
		}
		masked := n.add(Node{Op: OpSelect, Type: ir.F32, A: live, B: other, C: n.f32(0)})
		step = &n.Nodes[n.Nodes[f].A]
		step.A, step.B = f, masked
	}
	total := n.un(OpGroupSum, ir.F32, f)

	l := n.loop(K, -1)
	n.Loops[l].Spread = true
	j := n.add(Node{Op: OpIter, At: l, Type: ir.U32})
	jc := n.bin(OpMin, ir.U32, j, n.u32(K-1))
	row := n.add(Node{Op: OpCoord, A: 0, Type: ir.U32})
	el := n.bin(OpAdd, ir.U32, n.bin(OpMul, ir.U32, row, n.u32(K)), jc)
	bufOf := map[int]int{}
	for i, b := range m.Bufs {
		if b.Operand == sum {
			continue
		}
		bufOf[i] = n.buf(b)
	}
	nodeOf := make([]int, len(m.Nodes))
	for i, nd := range m.Nodes {
		switch {
		case nd.Op == OpCoord:
			nodeOf[i] = el
			continue
		case nd.Op == OpLoad && m.Bufs[nd.Buf].Operand == sum:
			nodeOf[i] = total
			continue
		case nd.Op == OpLoad:
			nd.Buf = bufOf[nd.Buf]
		}
		for _, a := range m.args(&nd) {
			*a = nodeOf[*a]
		}
		nodeOf[i] = n.add(nd)
	}
	for _, s := range m.Stores {
		n.Stores = append(n.Stores, Store{Buf: bufOf[s.Buf], Ix: nodeOf[s.Ix], Val: nodeOf[s.Val], At: l})
	}
	return n
}

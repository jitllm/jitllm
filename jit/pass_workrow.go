package jit

import "github.com/jitllm/jitllm/jit/ir"

// WorkRow runs a row's reduction and the map that reads it as one workgroup
// a row where the map's row is reduced again after it: each of their loops
// over the row spread over the Group threads, a thread's places clamped to
// the row and its sum masked past it, the sum combined over the workgroup.
type WorkRow struct {
	Group int
}

func (WorkRow) Name() string { return "workrow" }

func (w WorkRow) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	out := append([]*Kernel(nil), ks...)
	for ri := 0; ri < len(out); ri++ {
		r := out[ri]
		c, f, sum, ok := rowSum(r)
		if !ok {
			continue
		}
		for mi := ri + 1; mi < len(out); mi++ {
			m := out[mi]
			if m.Kind == PrimMap && len(m.Loops) == 0 && len(m.Space) == 2 && m.Space[0] == r.Space[0] &&
				m.Space[1] == r.Loops[c].Trip && readsRow(m, sum) && soleReader(out, m, sum) && reducedAgain(out[mi+1:], m) {
				j := w.join(r, c, f, m, sum)
				j.Passes = append(append([]string(nil), r.Passes...), w.Name())
				out[ri] = j
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

// reducedAgain reports whether a later reduction reads what m stores.
func reducedAgain(ks []*Kernel, m *Kernel) bool {
	for _, s := range m.Stores {
		if s.Barrier || s.Run {
			continue
		}
		for _, k := range ks {
			if k.Kind == PrimReduce && k.reads(m.Bufs[s.Buf].Operand) {
				return true
			}
		}
	}
	return false
}

// rowSum finds a reduction over a row: a thread a row, one loop with one
// sum, stored at the row into scratch.
func rowSum(n *Kernel) (c, f int, sum Operand, ok bool) {
	if n.Kind != PrimReduce || n.Work || len(n.Space) != 1 || len(n.Loops) != 1 || len(n.Stores) != 1 {
		return 0, 0, Operand{}, false
	}
	st := n.Stores[0]
	f = st.Val
	if nd := n.Nodes[f]; nd.Op != OpFold || !n.sumFold(f) || !n.storesOwn(st) || n.Bufs[st.Buf].Operand.Kind != OperandScratch {
		return 0, 0, Operand{}, false
	}
	return n.Nodes[f].At, f, n.Bufs[st.Buf].Operand, true
}

// readsRow reports whether a map over [rows][width] reads o only at its row.
func readsRow(m *Kernel, o Operand) bool {
	any := false
	for _, nd := range m.Nodes {
		if nd.Op != OpLoad || m.Bufs[nd.Buf].Operand != o {
			continue
		}
		if len(nd.Ax) != 1 || m.Nodes[nd.Ax[0]].Op != OpCoord || m.Nodes[nd.Ax[0]].A != 0 {
			return false
		}
		any = true
	}
	return any
}

func (w WorkRow) join(r *Kernel, c, f int, m *Kernel, sum Operand) *Kernel {
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
		step := n.Nodes[n.Nodes[f].A]
		other := step.B
		if other == f {
			other = step.A
		}
		masked := n.add(Node{Op: OpSelect, Type: ir.F32, A: live, B: other, C: n.f32(0)})
		n.Nodes[n.Nodes[f].A].A, n.Nodes[n.Nodes[f].A].B = f, masked
	}
	total := n.un(OpGroupSum, ir.F32, f)

	l := n.loop(K, -1)
	n.Loops[l].Spread = true
	j := n.add(Node{Op: OpIter, At: l, Type: ir.U32})
	jc := n.bin(OpMin, ir.U32, j, n.u32(K-1))
	n.Loops[l].Elem = jc + 1
	row := n.add(Node{Op: OpCoord, A: 0, Type: ir.U32})
	leaf := func(i int, nd Node) (int, bool) {
		switch {
		case nd.Op == OpCoord && nd.A == 0:
			return row, true
		case nd.Op == OpCoord:
			return jc, true
		case nd.Op == OpLoad && m.Bufs[nd.Buf].Operand == sum:
			return total, true
		}
		return 0, false
	}
	for _, s := range m.Stores {
		ns := Store{Buf: n.bufLike(m.Bufs[s.Buf]), At: l}
		ns.Val = copyInto(n, m, s.Val, leaf)
		if s.Ax != nil {
			for _, a := range s.Ax {
				ns.Ax = append(ns.Ax, copyInto(n, m, a, leaf))
			}
		} else {
			ns.Ix = copyInto(n, m, s.Ix, leaf)
		}
		n.Stores = append(n.Stores, ns)
	}
	return n
}

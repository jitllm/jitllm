package composer

import (
	"slices"

	"github.com/jitllm/jitllm/composer/ir"
)

// WorkRow runs a row's reduction and the map that reads it as one workgroup
// a row where the map's row is reduced again after it, on the matrix unit
// or not: each of their loops over the row spread over the Group threads, a
// thread's places clamped to the row and a sum masked past it, the sum or
// the largest combined over the workgroup, and stored still where another
// kernel reads it.
// Where Heads is set, a reduction over the last axis of a space of several
// -- a row's heads -- and the map reading it run as one workgroup of Heads
// threads an element of that space, whatever reads the map after it.
type WorkRow struct {
	Group, Heads int
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
		group := min(w.Group, d.groupMost())
		if len(r.Space) > 1 {
			group = min(w.Heads, d.groupMost())
		}
		if group == 0 {
			continue
		}
		for mi := ri + 1; mi < len(out); mi++ {
			m := out[mi]
			if m.Kind == PrimMap && len(m.Loops) == 0 && len(m.Space) == len(r.Space)+1 && slices.Equal(m.Space[:len(r.Space)], r.Space) &&
				m.Space[len(r.Space)] == r.Loops[c].Trip && readsRow(m, sum, len(r.Space)) &&
				(len(r.Space) > 1 || reducedAgain(out[mi+1:], m)) && !movesPast(out[ri+1:mi], m) {
				j := w.join(r, c, f, m, sum, group, !soleReader(out, m, sum))
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

// movesPast reports whether m may not move before any of ks, the kernels
// between it and the place a join puts it.
func movesPast(ks []*Kernel, m *Kernel) bool {
	for _, x := range ks {
		if conflicts(x, m) {
			return true
		}
	}
	return false
}

// reducedAgain reports whether a later reduction, on the matrix unit or
// not, reads what m stores.
func reducedAgain(ks []*Kernel, m *Kernel) bool {
	for _, s := range m.Stores {
		if s.Barrier || s.Run {
			continue
		}
		for _, k := range ks {
			if (k.Kind == PrimReduce || k.Kind == PrimMatrix) && k.reads(m.Bufs[s.Buf].Operand) {
				return true
			}
		}
	}
	return false
}

// rowSum finds a reduction over a row: a thread a row, one loop with one
// sum or one largest, stored at the row into scratch.
func rowSum(n *Kernel) (c, f int, sum Operand, ok bool) {
	if n.Kind != PrimReduce || n.Work || len(n.Space) < 1 || len(n.Loops) != 1 || len(n.Stores) != 1 {
		return 0, 0, Operand{}, false
	}
	st := n.Stores[0]
	f = st.Val
	if nd := n.Nodes[f]; nd.Op != OpFold || !n.sumFold(f) && !n.largestFold(f) || !n.storesOwn(st) || n.Bufs[st.Buf].Operand.Kind != OperandScratch {
		return 0, 0, Operand{}, false
	}
	return n.Nodes[f].At, f, n.Bufs[st.Buf].Operand, true
}

// readsRow reports whether a map over its outer axes and a row reads o
// only at its outer coordinates.
func readsRow(m *Kernel, o Operand, outer int) bool {
	any := false
	for _, nd := range m.Nodes {
		if nd.Op != OpLoad || m.Bufs[nd.Buf].Operand != o {
			continue
		}
		if len(nd.Ax) != outer {
			return false
		}
		for a, x := range nd.Ax {
			if m.Nodes[x].Op != OpCoord || m.Nodes[x].A != a {
				return false
			}
		}
		any = true
	}
	return any
}

// largestFold reports whether fold f keeps the largest of its terms as
// float32, none of them reading it: a thread taking any of the terms, and
// any of them again, keeps the loop's largest.
func (n *Kernel) largestFold(f int) bool {
	term := n.extremeTerm(f)
	return term >= 0 && n.Nodes[n.Nodes[f].A].Op == OpMax && n.Nodes[f].Type == ir.F32 && !n.reaches(term)[f]
}

// join is r and m as one workgroup a row, r's store kept where keep is set.
func (w WorkRow) join(r *Kernel, c, f int, m *Kernel, sum Operand, group int, keep bool) *Kernel {
	K := r.Loops[c].Trip
	n := r.clone()
	n.Name = r.Name + "+" + m.Name
	n.Work, n.Group = true, group
	n.Stores = nil
	n.Loops[c].Spread = true
	largest := !n.sumFold(f)
	if it := n.iterOf(c); it >= 0 {
		at := n.bin(OpMin, ir.U32, it, n.u32(K-1))
		n.redirect(it, at, at)
		if !largest {
			live := n.add(Node{Op: OpLess, Type: ir.Pred, A: it, B: n.u32(K)})
			step := n.Nodes[n.Nodes[f].A]
			other := step.B
			if other == f {
				other = step.A
			}
			masked := n.add(Node{Op: OpSelect, Type: ir.F32, A: live, B: other, C: n.f32(0)})
			n.Nodes[n.Nodes[f].A].A, n.Nodes[n.Nodes[f].A].B = f, masked
		}
	}
	op := OpGroupSum
	if largest {
		op = OpGroupMax
	}
	total := n.un(op, ir.F32, f)
	if keep {
		st := r.Stores[0]
		st.Val = total
		n.Stores = append(n.Stores, st)
	}

	l := n.loop(K, -1)
	n.Loops[l].Spread = true
	j := n.add(Node{Op: OpIter, At: l, Type: ir.U32})
	jc := n.bin(OpMin, ir.U32, j, n.u32(K-1))
	n.Loops[l].Elem = jc + 1
	outer := make([]int, len(r.Space))
	for a := range outer {
		outer[a] = n.add(Node{Op: OpCoord, A: a, Type: ir.U32})
	}
	leaf := func(i int, nd Node) (int, bool) {
		switch {
		case nd.Op == OpCoord && nd.A < len(outer):
			return outer[nd.A], true
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

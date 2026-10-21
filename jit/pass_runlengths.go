package jit

import "github.com/jitllm/jitllm/jit/ir"

// RunLengths makes a range of positions a static class and its length a run
// time one: the range's extent padded up to a multiple of Grain, so one
// kernel serves every step of the class, its positions counted from the
// first rounded down to Align. What the range attends stays the live
// length's, a run-time value. A Grain of 0 is the exact length: a range on a
// loop counts its live positions, the end less the first, at run time.
//
// Where Widths is set, a kernel that consumes none of its reductions itself
// takes the width it runs over at run time too, so one kernel serves every
// width: its outermost loops, and its reductions' loops, that run over a
// row of a value, a head's own width aside.
type RunLengths struct {
	Grain, Align int
	Widths       bool
}

func (RunLengths) Name() string { return "runlengths" }

func (p RunLengths) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	padded := map[Operand]int{}
	out := make([]*Kernel, len(ks))
	for i, k := range ks {
		out[i] = k
		pos := k.Positions
		if pos == nil || pos.Align != 1 {
			continue
		}
		m := k.clone()
		mp := *pos
		m.Positions = &mp
		if p.Grain == 0 {
			if mp.Loop < 0 {
				continue
			}
			m.Loops[mp.Loop].Count = m.bin(OpSub, ir.U32, mp.End, mp.First) + 1
			m.Passes = append(m.Passes, p.Name())
			out[i] = m
			continue
		}
		if mp.Axis >= 0 {
			C := m.Space[mp.Axis]
			m.Space[mp.Axis] = (C + p.Grain - 1) / p.Grain * p.Grain
			for _, s := range m.Stores {
				if !s.Barrier && !s.Run && m.storesOwn(s) {
					padded[m.Bufs[s.Buf].Operand] = mp.Axis
				}
			}
		} else {
			C := m.Loops[mp.Loop].Trip
			m.Loops[mp.Loop].Trip = (C + p.Grain - 1) / p.Grain * p.Grain
		}
		for _, n := range mp.Pos {
			first := m.Nodes[n].A
			m.Nodes[n].A = m.bin(OpMul, ir.U32, m.bin(OpDiv, ir.U32, first, m.u32(p.Align)), m.u32(p.Align))
		}
		mp.Align = p.Align
		m.Passes = append(m.Passes, p.Name())
		out[i] = m
	}
	if p.Widths {
		for i, k := range out {
			if m := k.runWidths(); m != nil {
				m.Passes = append(m.Passes, p.Name())
				out[i] = m
			}
		}
	}
	for i, k := range out {
		var m *Kernel
		for b, buf := range k.Bufs {
			axis, ok := padded[buf.Operand]
			if !ok || axis >= len(buf.Shape) {
				continue
			}
			if m == nil {
				m = k
				if m == ks[i] {
					m = k.clone()
				}
			}
			sh := append([]int(nil), buf.Shape...)
			sh[axis] = (sh[axis] + p.Grain - 1) / p.Grain * p.Grain
			m.Bufs[b].Shape = sh
		}
		if m != nil {
			out[i] = m
		}
	}
	return out, nil
}

// runWidths is k with its widths taken at run time, nil where none is.
func (k *Kernel) runWidths() *Kernel {
	for i, nd := range k.Nodes {
		if nd.Op == OpFold && k.readOutside(i) {
			return nil
		}
	}
	var m *Kernel
	for c, l := range k.Loops {
		if l.Range || l.Count > 0 || l.Lanes > 0 {
			continue
		}
		if l.Parent >= 0 && len(k.foldsOf(c)) == 0 {
			continue
		}
		if h := k.Heads; h != nil && l.Trip == h.Dim {
			continue
		}
		if !k.rowOfValue(c) {
			continue
		}
		if m == nil {
			m = k.clone()
		}
		ln := m.add(Node{Op: OpLength, Type: ir.U32, A: c, Imm: int64(l.Trip)})
		m.Loops[c].Count = ln + 1
		m.sizeBy(c, ln)
	}
	return m
}

// sizeBy makes each extent of a buffer's shape that loop c's iteration
// runs over, of the loop's trips, the run-time length ln.
func (k *Kernel) sizeBy(c, ln int) {
	it := k.iterOf(c)
	mark := func(b int, ax []int) {
		bf := &k.Bufs[b]
		for j, a := range ax {
			if a == it && j < len(bf.Shape) && bf.Shape[j] == k.Loops[c].Trip {
				if bf.Dims == nil {
					bf.Dims = make([]int, len(bf.Shape))
				}
				bf.Dims[j] = ln + 1
			}
		}
	}
	for _, nd := range k.Nodes {
		if nd.Op == OpLoad && nd.Ax != nil {
			mark(nd.Buf, nd.Ax)
		}
	}
	for _, s := range k.Stores {
		if s.Ax != nil && !s.Barrier && !s.Run {
			mark(s.Buf, s.Ax)
		}
	}
	for b := range k.Bufs {
		for o := range k.Bufs {
			if k.Bufs[o].Operand == k.Bufs[b].Operand && k.Bufs[o].Dims != nil && k.Bufs[b].Dims == nil && sameShape(k.Bufs[o].Shape, k.Bufs[b].Shape) {
				k.Bufs[b].Dims = append([]int(nil), k.Bufs[o].Dims...)
			}
		}
	}
}

// readOutside reports whether a node other than its own step, or a loop's
// count, reads fold f: the reduction consumed in its kernel.
func (k *Kernel) readOutside(f int) bool {
	step := k.Nodes[f].A
	for i, nd := range k.Nodes {
		if i == step || i == f {
			continue
		}
		for _, a := range k.args(&nd) {
			if *a == f {
				return true
			}
		}
	}
	return false
}

// rowOfValue reports whether loop c's iteration is a coordinate of a
// value's row the kernel reads or writes.
func (k *Kernel) rowOfValue(c int) bool {
	it := k.iterOf(c)
	if it < 0 {
		return false
	}
	isValue := func(b int) bool {
		o := k.Bufs[b].Operand
		return o.Kind == OperandValue || o.Kind == OperandScratch
	}
	for _, nd := range k.Nodes {
		if nd.Op == OpLoad && nd.Ax != nil && isValue(nd.Buf) {
			for _, a := range nd.Ax {
				if a == it {
					return true
				}
			}
		}
	}
	for _, s := range k.Stores {
		if s.Ax != nil && !s.Barrier && !s.Run && isValue(s.Buf) {
			for _, a := range s.Ax {
				if a == it {
					return true
				}
			}
		}
	}
	return false
}

package jit

import "github.com/jitllm/jitllm/jit/ir"

// RunLengths makes a range of positions a static class and its length a run
// time one: the range's extent padded up to a multiple of Grain, so one
// kernel serves every step of the class, its positions counted from the
// first rounded down to Align. What the range attends stays the live
// length's, a run-time value. A Grain of 0 is the exact length: a range on a
// loop counts its live positions, the end less the first, at run time.
type RunLengths struct {
	Grain, Align int
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

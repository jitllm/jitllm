package jit

import "github.com/jitllm/jitllm/jit/ir"

// Offsets places a range's query -- a value a kernel over a range of
// positions reads -- at a run-time offset of the step's query region, the
// offset the first word of Table: its writer stores past the offset, so one
// launch serves the step's rows wherever the engine places them, and the
// range reads the region.
type Offsets struct {
	Table Operand
}

func (Offsets) Name() string { return "offsets" }

func (p Offsets) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	query := map[Operand]bool{}
	for _, k := range ks {
		if k.Positions == nil {
			continue
		}
		for _, b := range k.Bufs {
			if b.Kind == BufIn && b.Operand.Kind == OperandValue && !k.writesAny(b.Operand, ks) {
				query[b.Operand] = true
			}
		}
	}
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		for i, s := range k.Stores {
			if s.Barrier || s.Run || !query[k.Bufs[s.Buf].Operand] {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			t := m.buf(Buf{Kind: BufIn, Elem: ir.U32, Operand: p.Table, Extent: 1})
			off := m.load(t, m.u32(0))
			ix := s.Ix
			if s.Ax != nil {
				ix = m.flatAt(m.Bufs[s.Buf].Shape, s.Ax)
			}
			m.Stores[i].Ax, m.Stores[i].Ix = nil, m.bin(OpAdd, ir.U32, off, ix)
			m.Bufs[s.Buf].Shape = nil
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// writesAny reports whether a kernel with positions writes o.
func (k *Kernel) writesAny(o Operand, ks []*Kernel) bool {
	for _, x := range ks {
		if x.Positions != nil && x.writes(o) {
			return true
		}
	}
	return false
}

// flatAt is coordinates ax of a space of extents sh as a flat index.
func (k *Kernel) flatAt(sh, ax []int) int {
	ix := ax[0]
	for d := 1; d < len(ax); d++ {
		ix = k.bin(OpAdd, ir.U32, k.bin(OpMul, ir.U32, ix, k.u32(sh[d])), ax[d])
	}
	return ix
}

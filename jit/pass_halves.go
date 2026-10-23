package jit

import "github.com/jitllm/jitllm/jit/ir"

// Halves stores the state a part keeps from step to step, its history, as
// binary16: each store narrows its float, rounded to nearest even and a NaN
// keeping its sign alone, each load widens. Where the device's conversion
// keeps a NaN's payload, as x86's does, the payload is cleared after it:
// h &^= (h&0x7FFF > 0x7C00) & 0x01FF in each half.
type Halves struct {
	KeepsPayload bool
}

func (Halves) Name() string { return "halves" }

func (p Halves) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		any := false
		for _, b := range k.Bufs {
			any = any || b.Operand.Kind == OperandTable && b.Operand.PerPart
		}
		if !any {
			return nil, false, nil
		}
		m := k.clone()
		U := ir.U32
		for b := range m.Bufs {
			if o := m.Bufs[b].Operand; o.Kind == OperandTable && o.PerPart {
				m.Bufs[b].Half = true
			}
		}
		for i := range m.Stores {
			s := &m.Stores[i]
			if s.Barrier || s.Run || !m.Bufs[s.Buf].Half {
				continue
			}
			h := m.add(Node{Op: OpHalf, Type: U, A: s.Val})
			if p.KeepsPayload {
				w := func(v uint32) int { return m.add(Node{Op: OpConst, Type: U, Imm: int64(v)}) }
				t := m.bin(OpAnd, U, h, w(0x7FFF7FFF))
				t = m.bin(OpGtW, U, t, w(0x7C007C00))
				t = m.bin(OpAnd, U, t, w(0x01FF01FF))
				h = m.bin(OpAndNot, U, h, t)
			}
			s.Val = h
		}
		return []*Kernel{m}, true, nil
	})
}

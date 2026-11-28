package composer

import (
	"github.com/jitllm/jitllm/composer/ir"
)

// Halves stores the state a part keeps from step to step, its history, as
// binary16: each store narrows its float, rounded to nearest even and a NaN
// keeping its sign alone, each load widens. Where the device's conversion
// keeps a NaN's payload, as x86's does, the payload is cleared after it:
// h &^= (h&0x7FFF > 0x7C00) & 0x01FF in each half.
type Halves struct {
	KeepsPayload bool
}

func (Halves) Name() string { return "halves" }

// history reports whether o is a part's history: a table of its own that
// is not a recurrent state ("rec."), which every device keeps at f32.
func (o Operand) history() bool {
	return o.Kind == OperandTable && o.PerPart && !o.Table.recurrent()
}

// entries reports the histories past the rows ks read whose values are
// their own: those every device keeps at f32, keys and values alike.
func entries(ks []*Kernel) map[int]bool {
	var own map[int]bool
	for _, k := range ks {
		for _, b := range k.Bufs {
			if o := b.Operand; o.Kind == OperandTable && o.Table == TableValues && o.Instance > 0 {
				if own == nil {
					own = map[int]bool{}
				}
				own[o.Instance] = true
			}
		}
	}
	return own
}

func (p Halves) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	own := entries(ks)
	half := func(o Operand) bool {
		return o.history() && !(o.Instance > 0 && own[o.Instance] && (o.Table == TableKeys || o.Table == TableValues))
	}
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		any := false
		for _, b := range k.Bufs {
			any = any || half(b.Operand)
		}
		if !any {
			return nil, false, nil
		}
		m := k.clone()
		U := ir.U32
		for b := range m.Bufs {
			if half(m.Bufs[b].Operand) {
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

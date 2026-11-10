package composer

import "github.com/jitllm/jitllm/composer/ir"

// Unsigned keeps an unsigned compare's order on a device whose integer
// compare is signed (the x86 one): a less of two words, either of which may
// reach past 2^31, compares them with their top bits flipped. A compare of
// indices, lengths and the constants below 2^31 they are made with is the
// same either way and is left.
type Unsigned struct{}

func (Unsigned) Name() string { return "unsigned" }

func (p Unsigned) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		small := k.smallWords()
		var at []int
		for i, nd := range k.Nodes {
			if nd.Op == OpLess && k.Nodes[nd.A].Type == ir.U32 && (!small[nd.A] || !small[nd.B]) {
				at = append(at, i)
			}
		}
		if len(at) == 0 {
			return nil, false, nil
		}
		m := k.clone()
		top := m.u32(1 << 31)
		flip := func(a int) int {
			if c := m.Nodes[a]; c.Op == OpConst {
				return m.u32(int(uint32(c.Imm) ^ 1<<31))
			}
			return m.bin(OpXor, ir.U32, a, top)
		}
		for _, i := range at {
			a, b := flip(m.Nodes[i].A), flip(m.Nodes[i].B)
			m.Nodes[i].A, m.Nodes[i].B = a, b
		}
		return []*Kernel{m}, true, nil
	})
}

// smallWords is the integer nodes below 2^31 by construction: indices,
// lengths, coordinates, argument words, the words of the step's tables,
// constants below it, and what sums, products, shifts, minima, maxima and
// quotients make of them. A difference is not one: it wraps past 2^31 below
// zero.
func (k *Kernel) smallWords() []bool {
	small := make([]bool, len(k.Nodes))
	for changed := true; changed; {
		changed = false
		for i, nd := range k.Nodes {
			if small[i] || nd.Type == ir.F32 || nd.Type == ir.Pred {
				continue
			}
			s := false
			switch nd.Op {
			case OpIter, OpLength, OpSpan, OpCoord, OpParam:
				s = true
			case OpLoad:
				// The step's tables hold positions, rows and pages; its tokens
				// hold -1 for a row that names none.
				o := k.Bufs[nd.Buf].Operand
				s = o.Kind == OperandTable && o.Table != TableTokens
			case OpConst:
				s = nd.Imm >= 0 && nd.Imm < 1<<31
			case OpAdd, OpMul, OpShl, OpMin, OpMax, OpDiv:
				s = small[nd.A] && small[nd.B]
			case OpShr, OpRem, OpAnd:
				s = small[nd.A] || nd.Op != OpShr && small[nd.B] ||
					nd.Op == OpShr && k.Nodes[nd.B].Op == OpConst && k.Nodes[nd.B].Imm > 0
			}
			if s {
				small[i], changed = true, true
			}
		}
	}
	return small
}

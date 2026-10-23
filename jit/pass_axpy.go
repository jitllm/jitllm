package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// Axpy writes a map that adds a row into the row it stores over as the add
// family's one form, fma(c, y, x) with c a constant of the call: so the
// kernel that adds serves every scaled add.
type Axpy struct{}

func (Axpy) Name() string { return "axpy" }

func (p Axpy) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Kind != PrimMap || len(k.Stores) != 1 {
			return nil, false, nil
		}
		st := k.Stores[0]
		v := k.Nodes[st.Val]
		if v.Op != OpAdd || v.Type != ir.F32 {
			return nil, false, nil
		}
		x, y := v.A, v.B
		if k.inPlaceLoad(y, st) {
			x, y = y, x
		}
		if !k.inPlaceLoad(x, st) || k.Nodes[y].Op != OpLoad {
			return nil, false, nil
		}
		m := k.clone()
		one := m.add(Node{Op: OpConst, Type: ir.F32, Imm: int64(math.Float32bits(1))})
		m.Nodes[st.Val] = Node{Op: OpFma, Type: ir.F32, A: one, B: y, C: x}
		return []*Kernel{m}, true, nil
	})
}

// inPlaceLoad reports whether node i loads the operand store s writes, at
// the element it writes.
func (k *Kernel) inPlaceLoad(i int, s Store) bool {
	nd := k.Nodes[i]
	return nd.Op == OpLoad && k.Bufs[nd.Buf].Operand == k.Bufs[s.Buf].Operand &&
		(nd.Ax != nil && sameInts(nd.Ax, s.Ax) || nd.Ax == nil && nd.A == s.Ix)
}

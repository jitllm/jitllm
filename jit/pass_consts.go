package jit

import "github.com/jitllm/jitllm/jit/ir"

// Consts places a kernel's float constants, zero aside, in a block of words
// its call points to, each read from its word where it was used: for a
// device whose instructions take no float immediate. The words are in the
// order the kernel's nodes first hold them.
type Consts struct{}

func (Consts) Name() string { return "consts" }

func (p Consts) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		live := k.live()
		var at []int
		for i, nd := range k.Nodes {
			if nd.Op == OpConst && nd.Type == ir.F32 && uint32(nd.Imm) != 0 && live[i] {
				at = append(at, i)
			}
		}
		if len(at) == 0 {
			return nil, false, nil
		}
		m := k.clone()
		b := m.buf(Buf{Kind: BufIn, Elem: ir.F32, Operand: Operand{Kind: OperandConsts, Name: k.Name + ".consts"}})
		slot := map[uint32]int{}
		for _, i := range at {
			v := uint32(m.Nodes[i].Imm)
			w, ok := slot[v]
			if !ok {
				w = len(m.Bufs[b].Words)
				slot[v] = w
				m.Bufs[b].Words = append(m.Bufs[b].Words, v)
			}
			m.Nodes[i] = Node{Op: OpLoad, Type: ir.F32, Buf: b, A: m.u32(w)}
		}
		return []*Kernel{m}, true, nil
	})
}

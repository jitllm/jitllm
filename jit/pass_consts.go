package jit

import "github.com/jitllm/jitllm/jit/ir"

// Consts places a kernel's float constants, zero aside, in a block of words
// its call points to, each read from its word where it was used: for a
// device whose instructions take no float immediate. The words are in the
// order the code first reads them, each store's value evaluated its deeper
// operand first. A kernel whose constants, the integers held in vectors
// among them, are the words of Shared reads Shared instead.
type Consts struct {
	Shared []uint32
}

func (Consts) Name() string { return "consts" }

func (p Consts) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var at []int
		seen := map[int]bool{}
		memo := map[int]int{}
		var walk func(i int)
		walk = func(i int) {
			if seen[i] {
				return
			}
			seen[i] = true
			nd := k.Nodes[i]
			if nd.Op == OpConst && nd.Type == ir.F32 && uint32(nd.Imm) != 0 {
				at = append(at, i)
				return
			}
			for _, a := range k.evalArgs(&nd, memo) {
				walk(*a)
			}
		}
		for _, s := range k.Stores {
			if !s.Barrier && !s.Run {
				walk(s.Val)
			}
		}
		if len(at) == 0 {
			return nil, false, nil
		}
		if m, ok := k.shared(p.Shared, at); ok {
			return []*Kernel{m}, true, nil
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

// shared reads floats at and the integer constants held in vectors from
// block, where they are block's words.
func (k *Kernel) shared(block []uint32, at []int) (*Kernel, bool) {
	if block == nil {
		return nil, false
	}
	word := map[uint32]int{}
	for w := len(block) - 1; w >= 0; w-- {
		word[block[w]] = w
	}
	all := append([]int(nil), at...)
	for i := range k.vectorWords() {
		if nd := k.Nodes[i]; nd.Op == OpConst && nd.Type != ir.F32 {
			all = append(all, i)
		}
	}
	read := map[uint32]bool{}
	for _, i := range all {
		v := uint32(k.Nodes[i].Imm)
		if _, ok := word[v]; !ok {
			return nil, false
		}
		read[v] = true
	}
	if len(read) != len(word) {
		return nil, false
	}
	m := k.clone()
	b := m.buf(Buf{Kind: BufIn, Elem: ir.F32, Shared: true, Words: append([]uint32(nil), block...), Operand: Operand{Kind: OperandConsts, Name: "shared.consts"}})
	for _, i := range all {
		nd := m.Nodes[i]
		m.Nodes[i] = Node{Op: OpLoad, Type: nd.Type, Buf: b, A: m.u32(word[uint32(nd.Imm)])}
	}
	return m, true
}

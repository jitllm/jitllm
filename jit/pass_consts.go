package jit

import "github.com/jitllm/jitllm/jit/ir"

// Consts places a kernel's float constants, zero aside, in a block of words
// its call points to, each read from its word where it was used: for a
// device whose instructions take no float immediate. A kernel that has the
// block places its integer constants held in vectors there too, read from
// it where a vector reads them, and each lane's index as that many words. The words are in the order the code
// first reads them, each store's value evaluated its deeper operand first.
// A kernel whose constants are the words of Shared reads Shared instead,
// and a constant a block of fixed words the kernel reads holds is read
// from that block.
type Consts struct {
	Shared []uint32
}

func (Consts) Name() string { return "consts" }

func (p Consts) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		vw := k.vectorWords()
		var at []int
		floats := false
		seen := map[int]bool{}
		memo := map[int]int{}
		var walk func(i int)
		walk = func(i int) {
			if seen[i] {
				return
			}
			seen[i] = true
			nd := k.Nodes[i]
			switch {
			case nd.Op == OpConst && nd.Type == ir.F32 && uint32(nd.Imm) != 0:
				at = append(at, i)
				floats = true
				return
			case nd.Op == OpConst && vw[i] && nd.Imm != 0, nd.Op == OpLanes:
				at = append(at, i)
				floats = floats || nd.Op == OpLanes
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
		m, at := k.fromTables(at)
		floats = false
		for _, i := range at {
			floats = floats || m.Nodes[i].Type == ir.F32 || m.Nodes[i].Op == OpLanes
		}
		if !floats {
			if m == k {
				return nil, false, nil
			}
			return []*Kernel{m}, true, nil
		}
		if s, ok := m.shared(p.Shared, at); ok {
			return []*Kernel{s}, true, nil
		}
		if m == k {
			m = k.clone()
		}
		b := m.buf(Buf{Kind: BufIn, Elem: ir.F32, Operand: Operand{Kind: OperandConsts, Name: k.Name + ".consts"}})
		slot := map[uint32]int{}
		for _, i := range at {
			nd := m.Nodes[i]
			if nd.Op == OpLanes {
				w := len(m.Bufs[b].Words)
				for j := range nd.Imm {
					m.Bufs[b].Words = append(m.Bufs[b].Words, uint32(j))
					if _, ok := slot[uint32(j)]; !ok {
						slot[uint32(j)] = w + int(j)
					}
				}
				m.Nodes[i] = Node{Op: OpLoad, Type: nd.Type, Buf: b, A: m.u32(w), Imm: nd.Imm, Hint: nd.Hint, Share: nd.Share}
				continue
			}
			v := uint32(nd.Imm)
			w, ok := slot[v]
			if !ok {
				w = len(m.Bufs[b].Words)
				slot[v] = w
				m.Bufs[b].Words = append(m.Bufs[b].Words, v)
			}
			ld := Node{Op: OpLoad, Type: nd.Type, Buf: b, A: m.u32(w), Hint: nd.Hint, Share: nd.Share}
			if nd.Type == ir.F32 {
				m.Nodes[i] = ld
				continue
			}
			l := m.add(ld)
			for r := range k.Nodes {
				if k.Nodes[r].Op == OpLoad || !vw[r] && k.Nodes[r].Type != ir.F32 && k.Nodes[r].Type != ir.Pred {
					continue
				}
				for _, a := range m.args(&m.Nodes[r]) {
					if *a == i {
						*a = l
					}
				}
			}
		}
		return []*Kernel{m}, true, nil
	})
}

// fromTables reads each constant of at that an input of fixed words holds
// from that input: the kernel, a copy where it read one, and the constants
// left.
func (k *Kernel) fromTables(at []int) (*Kernel, []int) {
	m, left := k, []int(nil)
	for _, i := range at {
		nd := k.Nodes[i]
		b, w := -1, -1
		if nd.Op == OpConst {
			for x, bf := range k.Bufs {
				if bf.Kind != BufIn || bf.Words == nil || bf.Operand.Kind == OperandConsts {
					continue
				}
				for j, v := range bf.Words {
					if v == uint32(nd.Imm) {
						b, w = x, j
						break
					}
				}
				if b >= 0 {
					break
				}
			}
		}
		if b < 0 {
			left = append(left, i)
			continue
		}
		if m == k {
			m = k.clone()
		}
		ld := Node{Op: OpLoad, Type: nd.Type, Buf: b, A: m.u32(w), Hint: nd.Hint, Share: nd.Share}
		if nd.Type == ir.F32 {
			m.Nodes[i] = ld
			continue
		}
		l := m.add(ld)
		vw := k.vectorWords()
		for r := range k.Nodes {
			if k.Nodes[r].Op == OpLoad || !vw[r] && k.Nodes[r].Type != ir.F32 && k.Nodes[r].Type != ir.Pred {
				continue
			}
			for _, a := range m.args(&m.Nodes[r]) {
				if *a == i {
					*a = l
				}
			}
		}
	}
	return m, left
}

// shared reads constants at from block, where they are block's words.
func (k *Kernel) shared(block []uint32, at []int) (*Kernel, bool) {
	if block == nil {
		return nil, false
	}
	word := map[uint32]int{}
	for w := len(block) - 1; w >= 0; w-- {
		word[block[w]] = w
	}
	read := map[uint32]bool{}
	for _, i := range at {
		v := uint32(k.Nodes[i].Imm)
		if _, ok := word[v]; !ok || k.Nodes[i].Op != OpConst {
			return nil, false
		}
		read[v] = true
	}
	if len(read) != len(word) {
		return nil, false
	}
	m := k.clone()
	b := m.buf(Buf{Kind: BufIn, Elem: ir.F32, Shared: true, Words: append([]uint32(nil), block...), Operand: Operand{Kind: OperandConsts, Name: "shared.consts"}})
	for _, i := range at {
		nd := m.Nodes[i]
		m.Nodes[i] = Node{Op: OpLoad, Type: nd.Type, Buf: b, A: m.u32(word[uint32(nd.Imm)]), Hint: nd.Hint, Share: nd.Share}
	}
	return m, true
}

package jit

import "github.com/jitllm/jitllm/jit/ir"

// HeadPairs runs N query heads that share their kv head in one call of a
// kernel that reads the history for each: the history read once, each head's
// own rows its own buffers. The second head's values copy the first's
// where its head is the first's plus one.
type HeadPairs struct {
	N int
}

func (HeadPairs) Name() string { return "pairs" }

func (p HeadPairs) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		h := k.Heads
		if p.N < 2 || h == nil || len(k.Space) < 2 || k.Space[1] != h.Query || h.Key <= 0 || (h.Query/h.Key)%p.N != 0 || !k.readsHistory() {
			return nil, false, nil
		}
		m := k.clone()
		U := ir.U32
		coord := -1
		for i, nd := range m.Nodes {
			if nd.Op == OpCoord && nd.A == 1 {
				coord = i
			}
		}
		if coord < 0 {
			return nil, false, nil
		}
		base := m.bin(OpMul, U, coord, m.u32(p.N))
		m.redirect(coord, base, base)
		m.Space[1] = h.Query / p.N
		stores := append([]Store(nil), m.Stores...)
		for j := 1; j < p.N; j++ {
			hj := m.bin(OpAdd, U, base, m.u32(j))
			r := replica{k: m, sub: map[int]int{base: hj}, memo: map[int]int{}}
			for _, s := range stores {
				if s.Barrier || s.Run || !r.reaches(s.Val, map[int]bool{}) && !m.storeReaches(&r, s) {
					continue
				}
				n := s
				n.Val = r.copy(s.Val)
				n.Ix = r.copy(s.Ix)
				if s.Ax != nil {
					n.Ax = make([]int, len(s.Ax))
					for x, a := range s.Ax {
						n.Ax[x] = r.copy(a)
					}
				}
				n.Copy = j
				m.Stores = append(m.Stores, n)
			}
			for orig, c := range r.memo {
				if c != orig {
					m.Nodes[c].Copy = j
				}
			}
		}
		for {
			step := m.cse()
			step = m.foldDecided() || step
			if !step {
				break
			}
			m.compact()
		}
		m.ownRows()
		return []*Kernel{m}, true, nil
	})
}

func (k *Kernel) storeReaches(r *replica, s Store) bool {
	for _, a := range k.storeArgs(&s) {
		if r.reaches(*a, map[int]bool{}) {
			return true
		}
	}
	return false
}

// readsHistory reports whether k reads a part's state across steps.
func (k *Kernel) readsHistory() bool {
	for _, nd := range k.Nodes {
		if nd.Op == OpLoad {
			if o := k.Bufs[nd.Buf].Operand; o.Kind == OperandTable && o.PerPart {
				return true
			}
		}
	}
	return false
}

// ownRows gives each copy's loads and stores that are not the first
// head's buffers of their own.
func (k *Kernel) ownRows() {
	bufOf := map[[2]int]int{}
	own := func(b, j int) int {
		if x, ok := bufOf[[2]int{b, j}]; ok {
			return x
		}
		nb := k.Bufs[b]
		nb.Operand.Copy = j
		x := k.buf(nb)
		bufOf[[2]int{b, j}] = x
		return x
	}
	for i := range k.Nodes {
		nd := &k.Nodes[i]
		if nd.Op == OpLoad && nd.Copy > 0 && k.Bufs[nd.Buf].Operand.Kind != OperandTable {
			nd.Buf = own(nd.Buf, nd.Copy)
		}
	}
	for i := range k.Stores {
		s := &k.Stores[i]
		if s.Copy > 0 && !s.Barrier && !s.Run {
			s.Buf = own(s.Buf, s.Copy)
		}
	}
}

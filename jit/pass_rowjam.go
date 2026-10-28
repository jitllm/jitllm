package jit

import "github.com/jitllm/jitllm/jit/ir"

// RowJam runs N of a step's rows in each call of a kernel tiling a weight
// over the step's rows (Workers): the leading axis counts units of N rows,
// and each row of a unit is a copy of the values its row reaches -- its
// activation's loads, its sums, its stores -- beside one reading of the
// weight's values, so a call decodes a tile of the weight once for N rows.
// A call of rows rows runs the units they reach (Code.LiveUnitsOf); a
// unit's rows past them read and write rows of the form's capacity no live
// row reads.
type RowJam struct {
	N int
}

func (RowJam) Name() string { return "rowjam" }

func (p RowJam) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if p.N < 2 || k.Unjammed || !k.TileMajor || k.Calls != 2 || len(k.Space) < 3 || k.Capacity != k.Space[0] || k.Space[0]%p.N != 0 || k.writesWhatItReads() {
			return nil, false, nil
		}
		m := k.clone()
		coord := -1
		for i, nd := range m.Nodes {
			if nd.Op != OpCoord || nd.A != 0 {
				continue
			}
			if coord < 0 {
				coord = i
				continue
			}
			m.redirect(i, coord, -1)
		}
		if coord < 0 {
			return nil, false, nil
		}
		U := ir.U32
		base := m.bin(OpMul, U, coord, m.u32(p.N))
		m.redirect(coord, base, base)
		m.Space[0] /= p.N
		m.Jam = p.N
		stores := append([]Store(nil), m.Stores...)
		for j := 1; j < p.N; j++ {
			r := replica{k: m, sub: map[int]int{base: m.bin(OpAdd, U, base, m.u32(j))}, memo: map[int]int{}}
			for _, s := range stores {
				if s.Barrier || s.Run || !m.storeReaches(&r, s) {
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
				m.Stores = append(m.Stores, n)
			}
		}
		for {
			step := m.cse(false)
			step = m.foldDecided() || step
			if !step {
				break
			}
			m.compact()
		}
		return []*Kernel{m}, true, nil
	})
}

// writesWhatItReads reports whether k stores to an operand it loads: a
// copy's row past a call's rows would read what another wrote.
func (k *Kernel) writesWhatItReads() bool {
	read := map[Operand]bool{}
	for _, nd := range k.Nodes {
		if nd.Op == OpLoad {
			read[k.Bufs[nd.Buf].Operand] = true
		}
	}
	for _, s := range k.Stores {
		if !s.Barrier && !s.Run && read[k.Bufs[s.Buf].Operand] {
			return true
		}
	}
	return false
}

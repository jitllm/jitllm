package composer

import "math/bits"

// Periods runs a loop of sums a whole period of its iteration a trip,
// where the loop indexes by the iteration's low bits and the rest apart --
// a block's sub-blocks, each with its own scale picked from words the block
// shares: the iteration masked to under 2^n, shifted right by n, or its
// remainder or quotient by 2^n, makes the period 2^n, at most Most. Each
// trip then steps the sums once for each position of the period, in order
// (Unroll's unrollBy), and the period's position is a constant in each, so
// what was picked by the low bits is a fixed word and offset.
type Periods struct {
	Most int
}

func (Periods) Name() string { return "periods" }

func (p Periods) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		for c := range k.Loops {
			l := k.Loops[c]
			if l.Count > 0 || l.Lanes > 0 || l.Range || l.Trip < 2 || k.hasInner(c) || k.storesAt(c) || len(k.foldsOf(c)) == 0 {
				continue
			}
			u := k.period(c, p.Most)
			if u < 2 || l.Trip%u != 0 {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			m.unrollBy(c, u)
		}
		if m == nil {
			return nil, false, nil
		}
		m.compact()
		return []*Kernel{m}, true, nil
	})
}

// period is the longest period, a power of two at most most, that loop c's
// iteration is read by: masked, shifted right, or divided by a power of
// two; 1 for none.
func (k *Kernel) period(c, most int) int {
	its := map[int]bool{}
	for _, it := range k.itersOf(c) {
		its[it] = true
	}
	u := 1
	for _, nd := range k.Nodes {
		if !its[nd.A] {
			continue
		}
		b := k.Nodes[nd.B]
		if b.Op != OpConst || b.Imm <= 0 {
			continue
		}
		var n int64
		switch nd.Op {
		case OpAnd:
			if b.Imm&(b.Imm+1) == 0 {
				n = b.Imm + 1
			}
		case OpShr:
			if b.Imm < 31 {
				n = 1 << b.Imm
			}
		case OpRem, OpDiv:
			if b.Imm&(b.Imm-1) == 0 {
				n = b.Imm
			}
		}
		if n > 1 && n <= int64(most) && int(n) > u {
			u = int(n)
		}
	}
	return 1 << (bits.Len(uint(u)) - 1)
}

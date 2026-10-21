package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// Quarters reduces an angle to quarter turns by the magic sum: Sum less
// Sum + M - M, M the float whose ulp is four, is Sum mod 4 in [-2, 2]
// exactly where |Sum| is under M/2; the turn is that plus Rest rounded to
// nearest, and the part the two's difference. Regs, where set, are the
// registers of M and of the part's multipliers (the one applied last, the
// large term added last, and the one before it); and where also set, of
// Sum and the values made in its place (Sum mod 4, the part), of Rest, of
// the turn, and of the temporaries: the sums' later terms, Sum + M and its
// difference, Sum mod 4 plus Rest, the turn as a float and the part's
// multiples.
type Quarters struct {
	Regs []int
}

func (Quarters) Name() string { return "quarters" }

func (p Quarters) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	const magic = 4 << 23
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		q := k.Quarter
		if q == nil || q.Bound >= magic/2 {
			return nil, false, nil
		}
		m := k.clone()
		F := ir.F32
		mg := m.add(Node{Op: OpConst, Type: F, Imm: int64(math.Float32bits(magic))})
		t1 := m.bin(OpAdd, F, q.Sum, mg)
		t2 := m.bin(OpSub, F, t1, mg)
		m4 := m.bin(OpSub, F, q.Sum, t2)
		sr := m.bin(OpAdd, F, m4, q.Rest)
		turn := m.add(Node{Op: OpRoundI32, Type: ir.I32, A: sr})
		tf := m.un(OpCvtF32, F, turn)
		part := m.bin(OpSub, F, m4, tf)
		m.redirect(q.Turn, turn, -1)
		m.redirect(q.Part, part, -1)
		if len(p.Regs) >= 3 {
			m.Nodes[mg].Hint = p.Regs[0] + 1
			m.hintMultipliers(part, p.Regs[1:3])
		}
		if len(p.Regs) == 7 {
			sum, rest, tn, tmp := p.Regs[3]+1, p.Regs[4]+1, p.Regs[5]+1, p.Regs[6]+1
			m.hintSum(q.Sum, sum, tmp)
			m.hintSum(q.Rest, rest, tmp)
			for _, x := range []int{t1, t2, sr, tf} {
				m.Nodes[x].Hint = tmp
			}
			m.Nodes[m4].Hint, m.Nodes[part].Hint, m.Nodes[turn].Hint = sum, sum, tn
			m.hintMultiples(part, tmp)
		}
		m.Quarter = nil
		m.compact()
		return []*Kernel{m}, true, nil
	})
}

// hintMultipliers gives the multiplier of the fma reading part that is
// applied last regs[0], and that of the fma it adds to regs[1], each node
// loading the same word alike.
func (k *Kernel) hintMultipliers(part int, regs []int) {
	other := func(i int) int {
		nd := k.Nodes[i]
		if nd.Op != OpFma {
			return -1
		}
		switch part {
		case nd.A:
			return nd.B
		case nd.B:
			return nd.A
		}
		return -1
	}
	for i, nd := range k.Nodes {
		c := nd.C
		if other(i) < 0 || other(c) < 0 {
			continue
		}
		for j, w := range []int{other(i), other(c)} {
			for x := range k.Nodes {
				if k.sameWord(x, w) {
					k.Nodes[x].Hint = regs[j] + 1
				}
			}
		}
	}
}

// sameWord reports whether nodes x and w read one word of one table, or
// are one node.
func (k *Kernel) sameWord(x, w int) bool {
	if x == w {
		return true
	}
	a, b := k.Nodes[x], k.Nodes[w]
	if a.Op != b.Op || a.Type != b.Type {
		return false
	}
	switch a.Op {
	case OpBitcast:
		return k.sameWord(a.A, b.A)
	case OpLoad:
		return a.Buf == b.Buf && a.Ptr == 0 && b.Ptr == 0 && sameExpr(k, a.A, k, b.A)
	}
	return false
}

// hintSum gives the add chain at i and its first term register acc, the
// chain's other terms tmp.
func (k *Kernel) hintSum(i, acc, tmp int) {
	for k.Nodes[i].Op == OpAdd {
		k.Nodes[i].Hint = acc
		k.Nodes[k.Nodes[i].B].Hint = tmp
		i = k.Nodes[i].A
	}
	k.Nodes[i].Hint = acc
}

// hintMultiples gives the fmas multiplying part, and the product each adds
// to, register tmp.
func (k *Kernel) hintMultiples(part, tmp int) {
	for i, nd := range k.Nodes {
		if nd.Op != OpFma || nd.A != part && nd.B != part {
			continue
		}
		k.Nodes[i].Hint = tmp
		if c := k.Nodes[nd.C]; c.Op == OpMul {
			k.Nodes[nd.C].Hint = tmp
		}
	}
}

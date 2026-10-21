package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// Quarters reduces an angle to quarter turns by the magic sum: Sum less
// Sum + M - M, M the float whose ulp is four, is Sum mod 4 in [-2, 2]
// exactly where |Sum| is under M/2; the turn is that plus Rest rounded to
// nearest, and the part the two's difference. Regs, where set, are the
// registers of M and of the part's multipliers: the one applied last (the
// large term, added last) and the one before it.
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
		m4 := m.bin(OpSub, F, q.Sum, m.bin(OpSub, F, m.bin(OpAdd, F, q.Sum, mg), mg))
		turn := m.add(Node{Op: OpRoundI32, Type: ir.I32, A: m.bin(OpAdd, F, m4, q.Rest)})
		part := m.bin(OpSub, F, m4, m.un(OpCvtF32, F, turn))
		m.redirect(q.Turn, turn, -1)
		m.redirect(q.Part, part, -1)
		if len(p.Regs) == 3 {
			m.Nodes[mg].Hint = p.Regs[0] + 1
			m.hintMultipliers(part, p.Regs[1:])
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

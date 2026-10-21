package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// Exp writes exp out for a device with no instruction for it: x clamped to
// [-87.3, 88], n = round(x*log2e), r = x - n*ln2 with ln2 in two parts so
// n*hi is exact, exp(r) by Horner on ((((c5 r + c4) r + c3) r + c2) r + 1) r
// + 1, and 2^n made in the exponent field as (n + 127) << 23. Where
// Scratch is set, the value is worked in Scratch[0], the register of the
// value it is made from, and the result returned there, and the
// reduction's and the polynomial's values take Scratch[1], the exponent's
// Scratch[2].
type Exp struct {
	Scratch [3]int
}

func (Exp) Name() string { return "exp" }

func (p Exp) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		for i, nd := range k.Nodes {
			if nd.Op != OpExp || nd.Type != ir.F32 {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			m.expand(i, p.Scratch)
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

func (k *Kernel) expand(i int, scratch [3]int) {
	F, I := ir.F32, ir.I32
	hint := func(n, s int) int {
		if scratch != [3]int{} {
			k.Nodes[n].Hint = scratch[s+1] + 1
		}
		return n
	}
	c := func(v float32) int { return k.f32(math.Float32bits(v)) }
	fma := func(a, b, z int) int { return k.add(Node{Op: OpFma, Type: F, A: a, B: b, C: z}) }
	x := k.Nodes[i].A
	share := i + 1
	if in := k.Nodes[x]; scratch != [3]int{} && arity(in.Op) == 2 {
		for _, o := range []int{in.A, in.B} {
			if k.Nodes[o].Op == OpLoad && k.Nodes[o].Type == F {
				k.Nodes[o].Share, k.Nodes[x].Share = share, share
				k.Nodes[o].Hint = scratch[0] + 1
				break
			}
		}
	}
	lo, hi := k.add(Node{Op: OpConst, Type: F, Imm: int64(math.Float32bits(-87.3))}), k.add(Node{Op: OpConst, Type: F, Imm: int64(math.Float32bits(88))})
	hint(lo, 0)
	hint(hi, 0)
	x = k.bin(OpMin, F, k.bin(OpMax, F, x, lo), hi)
	n := hint(k.add(Node{Op: OpRoundI32, Type: I, A: hint(k.bin(OpMul, F, x, c(float32(math.Log2E))), 0)}), 1)
	nf := hint(k.un(OpCvtF32, F, n), 0)
	r := fma(nf, c(-0.693359375), x)
	r = fma(nf, c(2.12194440e-4), r)
	q := c(float32(1.0 / 120))
	for _, v := range []float32{float32(1.0 / 24), float32(1.0 / 6), 0.5, 1, 1} {
		q = hint(fma(q, r, c(v)), 0)
	}
	e := hint(k.bin(OpShl, I, hint(k.bin(OpAdd, I, n, k.add(Node{Op: OpConst, Type: I, Imm: 127})), 1), k.add(Node{Op: OpConst, Type: I, Imm: 23})), 1)
	k.Nodes[i] = Node{Op: OpMul, Type: F, A: q, B: k.un(OpBitcast, F, e)}
	if scratch != [3]int{} {
		k.Nodes[i].Share = share
	}
}

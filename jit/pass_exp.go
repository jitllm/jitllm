package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// Exp writes exp out for a device with no instruction for it: x clamped to
// [-87.3, 88], n = round(x*log2e), r = x - n*ln2 with ln2 in two parts so
// n*hi is exact, exp(r) by Horner on ((((c5 r + c4) r + c3) r + c2) r + 1) r
// + 1, and 2^n made in the exponent field as (n + 127) << 23.
type Exp struct{}

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
			m.expand(i)
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

func (k *Kernel) expand(i int) {
	F, I := ir.F32, ir.I32
	c := func(v float32) int { return k.f32(math.Float32bits(v)) }
	fma := func(a, b, z int) int { return k.add(Node{Op: OpFma, Type: F, A: a, B: b, C: z}) }
	x := k.Nodes[i].A
	x = k.bin(OpMin, F, k.bin(OpMax, F, x, c(-87.3)), c(88))
	n := k.add(Node{Op: OpRoundI32, Type: I, A: k.bin(OpMul, F, x, c(float32(math.Log2E)))})
	nf := k.un(OpCvtF32, F, n)
	r := fma(nf, c(-0.693359375), x)
	r = fma(nf, c(2.12194440e-4), r)
	q := c(float32(1.0 / 120))
	for _, v := range []float32{float32(1.0 / 24), float32(1.0 / 6), 0.5, 1, 1} {
		q = fma(q, r, c(v))
	}
	e := k.bin(OpShl, I, k.bin(OpAdd, I, n, k.add(Node{Op: OpConst, Type: I, Imm: 127})), k.add(Node{Op: OpConst, Type: I, Imm: 23}))
	k.Nodes[i] = Node{Op: OpMul, Type: F, A: q, B: k.un(OpBitcast, F, e)}
}

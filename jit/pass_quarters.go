package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// Quarters reduces an angle to quarter turns by the magic sum: Sum less
// Sum + M - M, M the float whose ulp is four, is Sum mod 4 in [-2, 2]
// exactly where |Sum| is under M/2; the turn is that plus Rest rounded to
// nearest, and the part the two's difference.
type Quarters struct{}

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
		m.Quarter = nil
		m.compact()
		return []*Kernel{m}, true, nil
	})
}

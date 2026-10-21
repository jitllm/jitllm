package jit

import (
	"math"

	"github.com/jitllm/jitllm/jit/ir"
)

// Bounds takes the larger of a load and a constant as the load where the
// kernel that stores the loaded value states every element of it is at
// least the constant.
type Bounds struct{}

func (Bounds) Name() string { return "bounds" }

func (p Bounds) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	least := map[Operand]float32{}
	for _, k := range ks {
		if k.Least == 0 {
			continue
		}
		for _, s := range k.Stores {
			if !s.Barrier && !s.Run {
				least[k.Bufs[s.Buf].Operand] = k.Least
			}
		}
	}
	if len(least) == 0 {
		return ks, nil
	}
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		for i, nd := range k.Nodes {
			if nd.Op != OpMax || nd.Type != ir.F32 {
				continue
			}
			for _, pr := range [][2]int{{nd.A, nd.B}, {nd.B, nd.A}} {
				x, c := k.Nodes[pr[0]], k.Nodes[pr[1]]
				if x.Op != OpLoad || c.Op != OpConst {
					continue
				}
				l, ok := least[k.Bufs[x.Buf].Operand]
				if !ok || math.Float32frombits(uint32(c.Imm)) > l {
					continue
				}
				if m == nil {
					m = k.clone()
				}
				m.redirect(i, pr[0], -1)
				break
			}
		}
		if m == nil {
			return nil, false, nil
		}
		m.compact()
		return []*Kernel{m}, true, nil
	})
}

package jit

import "github.com/jitllm/jitllm/jit/ir"

// Trips runs a loop over positions as many trips as its live positions
// need, a run-time count: the live span over the positions one trip covers,
// rounded up -- a binding's unit, or one where a loop no binding dealt over
// threads walks a position a trip.
type Trips struct{}

func (Trips) Name() string { return "trips" }

func (p Trips) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		for c, l := range k.Loops {
			unit := l.Unit
			if unit == 0 && !l.Spread {
				unit = 1
			}
			if !l.Range || l.Span == 0 || unit == 0 || l.Count != 0 {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			span := l.Span - 1
			n := span
			if unit > 1 {
				n = m.bin(OpDiv, ir.U32, m.bin(OpAdd, ir.U32, span, m.u32(unit-1)), m.u32(unit))
			}
			m.Loops[c].Count = n + 1
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

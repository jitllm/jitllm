package jit

import "github.com/jitllm/jitllm/jit/ir"

// LiveRows holds a launch's threads to the elements of its live rows, where
// the step's tables say how many of its leading axis's rows are live: the
// live rows' elements, at least a row's and at most the space's.
type LiveRows struct{}

func (LiveRows) Name() string { return "liverows" }

func (p LiveRows) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Rows == 0 || k.Clamp != 0 || k.Work || len(k.Space) == 0 || k.Space[0] == 0 {
			return nil, false, nil
		}
		m := k.clone()
		E := m.elements()
		inner := E / m.Space[0]
		live := m.bin(OpMin, ir.U32, m.bin(OpMul, ir.U32, m.Rows-1, m.u32(inner)), m.u32(E))
		m.Clamp = m.bin(OpMax, ir.U32, m.u32(inner), live) + 1
		return []*Kernel{m}, true, nil
	})
}

package jit

// InPlace stores a map's output over an input it reads at the element it
// writes, of the same shape and type, that no later kernel reads: the
// output takes the input's storage, in the kernel and in every later
// kernel that reads it.
type InPlace struct{}

func (InPlace) Name() string { return "inplace" }

func (p InPlace) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	out := append([]*Kernel(nil), ks...)
	for i, k := range out {
		if k.Kind != PrimMap || len(k.Stores) != 1 || len(k.Bufs) == 0 {
			continue
		}
		st := k.Stores[0]
		if st.Ax == nil || st.Barrier || st.Run {
			continue
		}
		ob := k.Bufs[st.Buf]
		if ob.Operand.Kind == OperandTable {
			continue
		}
		in := -1
		for _, nd := range k.Nodes {
			if nd.Op != OpLoad || nd.Ax == nil || !sameInts(nd.Ax, st.Ax) {
				continue
			}
			b := k.Bufs[nd.Buf]
			if b.Elem != ob.Elem || !sameShape(b.Shape, ob.Shape) || b.Operand.Kind == OperandTable || b.Operand.Kind == OperandWeight {
				continue
			}
			if !readLater(out[i+1:], b.Operand) {
				in = nd.Buf
				break
			}
		}
		if in < 0 {
			continue
		}
		from, to := ob.Operand, k.Bufs[in].Operand
		for j := i; j < len(out); j++ {
			if !usesOperand(out[j], from) {
				continue
			}
			m := out[j].clone()
			for b := range m.Bufs {
				if m.Bufs[b].Operand == from {
					m.Bufs[b].Operand = to
				}
			}
			m.Passes = append(m.Passes, p.Name())
			out[j] = m
		}
	}
	return out, nil
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// readLater reports whether any of ks loads operand o.
func readLater(ks []*Kernel, o Operand) bool {
	for _, k := range ks {
		for _, nd := range k.Nodes {
			if nd.Op == OpLoad && k.Bufs[nd.Buf].Operand == o {
				return true
			}
		}
	}
	return false
}

func usesOperand(k *Kernel, o Operand) bool {
	for _, b := range k.Bufs {
		if b.Operand == o {
			return true
		}
	}
	return false
}

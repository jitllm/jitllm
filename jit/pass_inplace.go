package jit

// InPlace stores a value not already over its own input over an input the
// kernel reads at the element it writes, of the same shape and type, that nothing reads after: no later
// kernel, and no later loop of this one nor this loop at another element.
// The output takes the input's storage, in the kernel and in every later
// kernel that reads it, each head's copy of it too.
type InPlace struct{}

func (InPlace) Name() string { return "inplace" }

func (p InPlace) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	out := append([]*Kernel(nil), ks...)
	for i := range out {
		for si := range out[i].Stores {
			k := out[i]
			st := k.Stores[si]
			if st.Ax == nil || st.Barrier || st.Run || len(k.Stores) == 1 && k.Kind != PrimMap || len(k.Stores) > 1 && st.At < 0 {
				continue
			}
			ob := k.Bufs[st.Buf]
			if ob.Operand.Kind == OperandTable || k.readsAt(ob.Operand, st) {
				continue
			}
			in := -1
			for _, nd := range k.Nodes {
				if nd.Op != OpLoad || nd.Ax == nil || !sameInts(nd.Ax, st.Ax) {
					continue
				}
				b := k.Bufs[nd.Buf]
				if b.Elem != ob.Elem || !sameShape(b.Shape, ob.Shape) || b.Operand.Kind == OperandTable || b.Operand.Kind == OperandWeight || b.Operand == ob.Operand {
					continue
				}
				if !readLater(out[i+1:], b.Operand) && (len(k.Stores) == 1 || !k.readsAfter(b.Operand, st)) {
					in = nd.Buf
					break
				}
			}
			if in < 0 {
				continue
			}
			from, to := ob.Operand, k.Bufs[in].Operand
			same := func(o Operand) (int, bool) {
				c := o.Copy
				o.Copy = from.Copy
				return c, o == from
			}
			for j := i; j < len(out); j++ {
				touched := false
				for _, b := range out[j].Bufs {
					_, ok := same(b.Operand)
					touched = touched || ok
				}
				if !touched {
					continue
				}
				m := out[j].clone()
				for b := range m.Bufs {
					if c, ok := same(m.Bufs[b].Operand); ok {
						m.Bufs[b].Operand = to
						m.Bufs[b].Operand.Copy = c
					}
				}
				m.Passes = append(m.Passes, p.Name())
				out[j] = m
			}
		}
	}
	return out, nil
}

// readsAt reports whether k reads operand o where store st writes it: the
// store is over its own input.
func (k *Kernel) readsAt(o Operand, st Store) bool {
	hs := k.homes()
	for i, nd := range k.Nodes {
		if nd.Op == OpLoad && k.Bufs[nd.Buf].Operand == o && sameInts(nd.Ax, st.Ax) && (st.At < 0 || hs[i] >= 0 && k.within(hs[i], st.At)) {
			return true
		}
	}
	return false
}

// readsAfter reports whether k reads operand o other than where store st
// writes it: in a loop after st's, or in st's at another element.
func (k *Kernel) readsAfter(o Operand, st Store) bool {
	hs := k.homes()
	for i, nd := range k.Nodes {
		if nd.Op != OpLoad || k.Bufs[nd.Buf].Operand != o {
			continue
		}
		h := hs[i]
		switch {
		case st.At < 0:
		case h > st.At && !k.within(h, st.At):
			return true
		case h >= 0 && k.within(h, st.At) && !sameInts(nd.Ax, st.Ax):
			return true
		}
	}
	return false
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

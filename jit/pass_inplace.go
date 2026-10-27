package jit

// InPlace stores a value not already over its own input over an input the
// kernel reads at the element it writes, of the same shape and type, that nothing reads after: no later
// kernel, and no later loop of this one nor this loop at another element;
// a graph's output only a graph input's. The output takes the input's storage,
// in the kernel and in every later kernel that reads it, each head's copy of it
// too. In a graph whose inputs and outputs are apart (Request.Apart), an output
// keeps its own storage and no value takes an input's: each is its caller's
// buffer.
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
			// An earlier kernel writing the output keeps its storage: the
			// value's elements it stores would be apart from these.
			if ob.Operand.Kind == OperandTable || k.Apart && ob.Operand.Out || k.readsAt(ob.Operand, st) || writtenBy(out[:i], ob.Operand) {
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
				if k.Apart && b.Operand.Kind == OperandValue && !writtenBy(out[:i], b.Operand) && !k.writes(b.Operand) {
					continue
				}
				// A graph's output is stored where its caller binds it: it
				// takes only an input's storage, which the call site may bind
				// to the same place (a block's residual), never a value the
				// form made (a head's logits after its projection).
				if ob.Operand.Out && (b.Operand.Temp || !callerHeld(out, b.Operand)) {
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
	hs := k.Homes()
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
	hs := k.Homes()
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

// callerHeld reports whether operand o is the caller's: the first kernel of
// ks to touch it reads it, so no kernel made it.
func callerHeld(ks []*Kernel, o Operand) bool {
	for _, k := range ks {
		one := []*Kernel{k}
		if readLater(one, o) {
			return true
		}
		if writtenBy(one, o) {
			return false
		}
	}
	return false
}

// writtenBy reports whether a kernel of ks stores operand o, any head's
// copy of it.
func writtenBy(ks []*Kernel, o Operand) bool {
	o.Copy = 0
	for _, k := range ks {
		for _, s := range k.Stores {
			w := k.Bufs[s.Buf].Operand
			w.Copy = 0
			if !s.Barrier && w == o {
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

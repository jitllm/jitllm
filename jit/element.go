package jit

// The element's own place: a kernel's buffer of its space's shape, read or
// written at the element's coordinates in order.

// own reports whether coordinates ax of buffer b are the element's own.
func (k *Kernel) own(b int, ax []int) bool {
	sh := k.Bufs[b].Shape
	if ax == nil || len(ax) != len(k.Space) || len(sh) != len(k.Space) {
		return false
	}
	for i, a := range ax {
		if nd := k.Nodes[a]; nd.Op != OpCoord || nd.A != i || sh[i] != k.Space[i] {
			return false
		}
	}
	return true
}

func (k *Kernel) storesOwn(s Store) bool { return !s.Barrier && !s.Run && k.own(s.Buf, s.Ax) }

func (k *Kernel) reads(o Operand) bool {
	for _, nd := range k.Nodes {
		if nd.Op == OpLoad && k.Bufs[nd.Buf].Operand == o {
			return true
		}
	}
	return false
}

func (k *Kernel) writes(o Operand) bool {
	for _, s := range k.Stores {
		if !s.Barrier && !s.Run && k.Bufs[s.Buf].Operand == o {
			return true
		}
	}
	return false
}

// readsOwn reports whether k reads o, and only at its element's own place.
func (k *Kernel) readsOwn(o Operand) bool {
	any := false
	for _, nd := range k.Nodes {
		if nd.Op == OpLoad && k.Bufs[nd.Buf].Operand == o {
			if !k.own(nd.Buf, nd.Ax) {
				return false
			}
			any = true
		}
	}
	return any
}

func sameSpace(a, b *Kernel) bool {
	if len(a.Space) != len(b.Space) {
		return false
	}
	for i := range a.Space {
		if a.Space[i] != b.Space[i] {
			return false
		}
	}
	return true
}

// soleReader reports whether c is the only kernel of ks reading o. A value
// another part reads is a part's output, which the engine reads too (the
// logits, a residual at a device seam): never solely c's.
func soleReader(ks []*Kernel, c *Kernel, o Operand) bool {
	if o.Kind == OperandValue && (o.Of != c.Of || o.Out) {
		return false
	}
	for _, k := range ks {
		if k != c && k.reads(o) {
			return false
		}
	}
	return true
}

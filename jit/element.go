package jit

import (
	"slices"

	"github.com/jitllm/jitllm/jit/ir"
)

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

// ownWrites is the operands every store of k writes at its element's own
// place.
func (k *Kernel) ownWrites() []Operand {
	var own []Operand
	for _, s := range k.Stores {
		if s.Barrier || s.Run {
			continue
		}
		if o := withCopy(k.Bufs[s.Buf].Operand, 0); !slices.Contains(own, o) && k.writesOwnAlone(o) {
			own = append(own, o)
		}
	}
	return own
}

func (k *Kernel) writesOwnAlone(o Operand) bool {
	var of []Store
	for _, s := range k.Stores {
		if !s.Barrier && !s.Run && withCopy(k.Bufs[s.Buf].Operand, 0) == o {
			of = append(of, s)
		}
	}
	if len(of) == 1 && of[0].Ax == nil && of[0].At < 0 {
		return k.injective(of[0].Ix)
	}
	for _, s := range of {
		if !k.storesOwn(s) {
			return false
		}
	}
	return true
}

// injective reports whether flat index node i is an affine sum of the
// space's coordinates that no two of its points share.
func (k *Kernel) injective(i int) bool {
	coef, _, ok := k.affine(i)
	if !ok {
		return false
	}
	var axes []int
	for a, n := range k.Space {
		if n > 1 {
			if coef[a] <= 0 {
				return false
			}
			axes = append(axes, a)
		}
	}
	slices.SortFunc(axes, func(a, b int) int { return int(coef[a] - coef[b]) })
	span := int64(0)
	for _, a := range axes {
		if coef[a] <= span {
			return false
		}
		span += coef[a] * int64(k.Space[a]-1)
	}
	return true
}

// affine is node i as the sum of its coordinates' coefficients times them
// and a constant, in unsigned 32-bit words.
func (k *Kernel) affine(i int) (map[int]int64, int64, bool) {
	nd := k.Nodes[i]
	if nd.Type != ir.U32 {
		return nil, 0, false
	}
	switch nd.Op {
	case OpConst:
		return map[int]int64{}, nd.Imm, true
	case OpCoord:
		return map[int]int64{nd.A: 1}, 0, true
	case OpAdd:
		a, x, ok := k.affine(nd.A)
		b, y, ok2 := k.affine(nd.B)
		if !ok || !ok2 {
			return nil, 0, false
		}
		for c, v := range b {
			a[c] += v
		}
		return a, x + y, true
	case OpMul:
		a, x, ok := k.affine(nd.A)
		b, y, ok2 := k.affine(nd.B)
		if !ok || !ok2 || len(a) > 0 && len(b) > 0 {
			return nil, 0, false
		}
		if len(a) == 0 {
			a, x, b, y = b, y, a, x
		}
		for c := range a {
			a[c] *= y
		}
		return a, x * y, true
	}
	return nil, 0, false
}

// ownApart is p's Own that c, joined to it at other places than p's own,
// does not write.
func ownApart(p, c *Kernel) []Operand {
	var own []Operand
	for _, o := range p.Own {
		if !c.writesCopy(o) {
			own = append(own, o)
		}
	}
	return own
}

// ownRenamed is k's Own once operand from is bound as to, k as it was
// before.
func (k *Kernel) ownRenamed(from, to Operand) []Operand {
	from.Copy, to.Copy = 0, 0
	writes := k.writesCopy
	var own []Operand
	for _, o := range k.Own {
		if o != from && o != to {
			own = append(own, o)
		}
	}
	ok := func(o Operand) bool { return !writes(o) || slices.Contains(k.Own, o) }
	if (writes(from) || writes(to)) && ok(from) && ok(to) {
		own = append(own, to)
	}
	return own
}

// writesCopy reports whether k writes operand o or any copy of it.
func (k *Kernel) writesCopy(o Operand) bool {
	for _, s := range k.Stores {
		if !s.Barrier && !s.Run && withCopy(k.Bufs[s.Buf].Operand, 0) == withCopy(o, 0) {
			return true
		}
	}
	return false
}

func withCopy(o Operand, c int) Operand {
	o.Copy = c
	return o
}

// ownJoined is the Own of p and c joined over one space.
func ownJoined(p, c *Kernel) []Operand {
	own := ownApart(p, c)
	for _, o := range c.Own {
		if !p.writesCopy(o) || slices.Contains(p.Own, o) {
			own = append(own, o)
		}
	}
	return own
}

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

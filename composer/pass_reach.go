package composer

import "slices"

// Reach is the leading rows of operand Operand each point of a kernel's
// space reaches alone: along space axis i, the Per[i] rows from its
// coordinate times Per[i], of an operand Space[i]*Per[i] rows long there.
type Reach struct {
	Operand Operand
	Per     []int
}

// Reaches states, for each operand a kernel reads or writes only at its
// points' own leading rows, the rows a point reaches (Kernel.Reach): every
// access's leading coordinates the point's times a multiple and plus less
// than it, the operand's leading extents the space's times the same.
type Reaches struct{}

func (Reaches) Name() string { return "reaches" }

func (Reaches) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	for _, k := range ks {
		k.Reach = k.ownReach()
	}
	return ks, nil
}

func (k *Kernel) ownReach() []Reach {
	if len(k.Space) == 0 {
		return nil
	}
	var out []Reach
	var bad []Operand
	see := func(b int, ax []int) {
		o := withCopy(k.Bufs[b].Operand, 0)
		if slices.Contains(bad, o) {
			return
		}
		p := k.rowsReached(b, ax)
		i := slices.IndexFunc(out, func(r Reach) bool { return r.Operand == o })
		switch {
		case p != nil && i < 0:
			out = append(out, Reach{o, p})
		case p == nil || !slices.Equal(p, out[i].Per):
			bad = append(bad, o)
			if i >= 0 {
				out = slices.Delete(out, i, i+1)
			}
		}
	}
	for _, nd := range k.Nodes {
		if nd.Op == OpLoad {
			see(nd.Buf, nd.Ax)
		}
	}
	for _, s := range k.Stores {
		if !s.Barrier && !s.Run {
			see(s.Buf, s.Ax)
		}
	}
	return out
}

// rowsReached is the rows of buffer b's leading axes coordinates ax reach
// at a point: along each, the point's coordinate times m plus a constant
// under m, of an axis m times the space's; nil where they are not.
func (k *Kernel) rowsReached(b int, ax []int) []int {
	sh := k.Bufs[b].Shape
	if len(ax) < len(k.Space) || len(sh) != len(ax) {
		return nil
	}
	per := make([]int, len(k.Space))
	for i := range k.Space {
		m, ok := k.rowOf(ax[i], i)
		if !ok || sh[i] != k.Space[i]*m {
			return nil
		}
		per[i] = m
	}
	return per
}

// rowOf is node x as coordinate i times m plus a constant under m.
func (k *Kernel) rowOf(x, i int) (int, bool) {
	nd := k.Nodes[x]
	switch nd.Op {
	case OpCoord:
		return 1, nd.A == i
	case OpConst:
		return 1, k.Space[i] == 1 && nd.Imm == 0
	case OpMul:
		a, b := nd.A, nd.B
		if k.Nodes[a].Op == OpConst {
			a, b = b, a
		}
		if c := k.Nodes[a]; c.Op == OpCoord && c.A == i && k.Nodes[b].Op == OpConst && k.Nodes[b].Imm > 0 {
			return int(k.Nodes[b].Imm), true
		}
	case OpAdd:
		a, b := nd.A, nd.B
		if k.Nodes[a].Op == OpConst {
			a, b = b, a
		}
		if c := k.Nodes[b]; c.Op == OpConst && c.Imm >= 0 {
			if m, ok := k.rowOf(a, i); ok && k.Nodes[a].Op == OpMul && c.Imm < int64(m) {
				return m, true
			}
		}
	}
	return 0, false
}

package composer

import (
	"reflect"
	"slices"
)

// ContractPairs runs twin reductions of one input, read only by the map after
// them at its own element, as one kernel taking both and the map.
type ContractPairs struct{}

func (ContractPairs) Name() string { return "contractpairs" }

func (p ContractPairs) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	out := append([]*Kernel(nil), ks...)
	for i := 0; i+2 < len(out); i++ {
		a, b, m := out[i], out[i+1], out[i+2]
		w := twinWeights(a, b)
		if w == nil || m.Work || m.Kind != PrimMap || len(m.Loops) != 0 || m.Part != a.Part || !sameSpace(a, m) {
			continue
		}
		oa, ob := a.Bufs[a.Stores[0].Buf].Operand, b.Bufs[b.Stores[0].Buf].Operand
		if oa == ob || !readsOwn(m, oa) || !readsOwn(m, ob) || !sharesActivation(a, b) {
			continue
		}
		out[i] = p.merge(out, a, b, m, w)
		out = append(out[:i+1], out[i+3:]...)
	}
	return out, nil
}

func (p ContractPairs) merge(ks []*Kernel, a, b, m *Kernel, w []bool) *Kernel {
	f := a.clone()
	sa, sb := a.Stores[0], b.Stores[0]
	vb := copyInto(f, b, sb.Val, func(i int, _ Node) (int, bool) { return i, !w[i] })
	oa, ob := a.Bufs[sa.Buf].Operand, b.Bufs[sb.Buf].Operand
	var st []Store
	if !soleReader(ks, m, oa) {
		st = append(st, f.Stores[0])
	}
	if !soleReader(ks, m, ob) {
		st = append(st, Store{Buf: f.bufLike(b.Bufs[sb.Buf]), At: -1, Val: vb, Ax: slices.Clone(sa.Ax)})
	}
	leaf := func(_ int, nd Node) (int, bool) {
		switch {
		case nd.Op == OpCoord:
			return sa.Ax[nd.A], true
		case nd.Op == OpLoad && m.Bufs[nd.Buf].Operand == oa:
			return sa.Val, true
		case nd.Op == OpLoad && m.Bufs[nd.Buf].Operand == ob:
			return vb, true
		}
		return 0, false
	}
	for _, s := range m.Stores {
		ns := Store{Buf: f.bufLike(m.Bufs[s.Buf]), At: -1, Val: copyInto(f, m, s.Val, leaf)}
		if s.Ax != nil {
			for _, x := range s.Ax {
				ns.Ax = append(ns.Ax, copyInto(f, m, x, leaf))
			}
		} else {
			ns.Ix = copyInto(f, m, s.Ix, leaf)
		}
		st = append(st, ns)
	}
	f.Stores = st
	f.Name = a.Name + "+" + m.Name
	f.Passes = append(f.Passes, p.Name())
	f.compact()
	return f
}

// twinWeights marks b's nodes that reach a weight of its own, where a and b
// are one reduction node for node but for their weights; nil where not.
func twinWeights(a, b *Kernel) []bool {
	if a.Work || b.Work || a.Kind != b.Kind || a.Kind != PrimReduce && a.Kind != PrimContract || a.Unfused || b.Unfused || len(a.Stores) != 1 || len(b.Stores) != 1 ||
		a.Positions != nil || a.Heads != nil || a.Quarter != nil || a.Rows != 0 || a.Clamp != 0 || a.Capacity != 0 || a.Seq != nil ||
		len(a.Args) > 0 || len(a.Ptrs) > 0 || a.Part != b.Part || a.Of != b.Of || a.Group != b.Group || !sameSpace(a, b) ||
		len(a.Nodes) != len(b.Nodes) || len(a.Bufs) != len(b.Bufs) || !reflect.DeepEqual(a.Loops, b.Loops) {
		return nil
	}
	for _, k := range []*Kernel{a, b} {
		s := k.Stores[0]
		if s.Barrier || s.Run || s.At != -1 || s.Ax == nil || k.Bufs[s.Buf].Kind != BufOut {
			return nil
		}
	}
	if sa, sb := a.Stores[0], b.Stores[0]; sa.Val != sb.Val || !slices.Equal(sa.Ax, sb.Ax) {
		return nil
	}
	own := make([]bool, len(b.Bufs))
	weights := false
	for i, x := range a.Bufs {
		y := b.Bufs[i]
		switch {
		case i == a.Stores[0].Buf:
			if x.Kind != y.Kind || x.Elem != y.Elem || !slices.Equal(x.Shape, y.Shape) {
				return nil
			}
		case x.Operand == y.Operand && x.Kind == y.Kind && slices.Equal(x.Shape, y.Shape):
		case x.Kind == BufIn && y.Kind == BufIn && x.Operand.Kind == OperandWeight && y.Operand.Kind == OperandWeight &&
			x.Operand.Format == y.Operand.Format && x.Operand.Region == y.Operand.Region && x.Elem == y.Elem && slices.Equal(x.Shape, y.Shape):
			own[i], weights = true, true
		case x.Operand.Temp && y.Operand.Temp && x.Kind == y.Kind && x.Elem == y.Elem && slices.Equal(x.Shape, y.Shape):
			own[i] = true
		default:
			return nil
		}
	}
	if !weights {
		return nil
	}
	w := make([]bool, len(b.Nodes))
	for i := range a.Nodes {
		x, y := a.Nodes[i], b.Nodes[i]
		if x.Op != y.Op || x.Type != y.Type || x.A != y.A || x.B != y.B || x.C != y.C || x.At != y.At || x.Buf != y.Buf || x.Imm != y.Imm ||
			!slices.Equal(x.Ax, y.Ax) || x.Tile != nil || y.Tile != nil {
			return nil
		}
		w[i] = y.Op == OpLoad && own[y.Buf]
	}
	for changed := true; changed; {
		changed = false
		for i := range b.Nodes {
			if w[i] {
				continue
			}
			nd := b.Nodes[i]
			for _, o := range b.Operands(&nd) {
				if w[*o] {
					w[i], changed = true, true
					break
				}
			}
		}
	}
	return w
}

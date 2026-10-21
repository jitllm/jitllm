package jit

import "github.com/jitllm/jitllm/jit/ir"

// FoldLoads takes a load its one reader broadcasts as the broadcast's, and
// a load of a whole vector, one its loop moves, that its one reader can
// read from memory as that reader's operand: the second multiplicand of a sum's step in a vector
// loop -- the first where the second is a load that must be widened -- or
// the second operand of a map's operation, an integer and's among them, in
// a vector loop no scalar tail follows.
type FoldLoads struct{}

func (FoldLoads) Name() string { return "foldloads" }

func (p FoldLoads) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		hs := k.homes()
		for i, nd := range k.Nodes {
			var b int
			if nd.Op == OpBroadcast {
				if ld := k.Nodes[nd.A]; ld.Op == OpLoad && !ld.Mem && !k.Bufs[ld.Buf].Half && k.uses(nd.A) == 1 {
					if m == nil {
						m = k.clone()
					}
					m.Nodes[nd.A].Mem = true
				}
				continue
			}
			switch nd.Op {
			case OpFma, OpMul, OpAdd:
				b = nd.B
			case OpAnd:
				if nd.Type == ir.F32 {
					continue
				}
				b = nd.B
			default:
				continue
			}
			foldable := func(x int) bool {
				ld := k.Nodes[x]
				return ld.Op == OpLoad && !ld.Mem && !k.Bufs[ld.Buf].Half && k.uses(x) == 1
			}
			if !foldable(b) {
				if nd.Op != OpFma || !foldable(nd.A) || k.Nodes[nd.B].Op != OpLoad {
					continue
				}
				b = nd.A
			}
			c := hs[i]
			if c < 0 || k.Loops[c].Lanes == 0 || !k.varies(b, c) {
				continue
			}
			sum := nd.Op == OpFma && k.Nodes[nd.C].Op == OpFold
			if !sum && (nd.Op == OpFma || k.hasTail(c)) {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			m.Nodes[b].Mem = true
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// uses is how many nodes and stores read node i.
func (k *Kernel) uses(i int) int {
	n := 0
	for _, nd := range k.Nodes {
		for _, a := range k.args(&nd) {
			if *a == i {
				n++
			}
		}
	}
	for _, s := range k.Stores {
		if !s.Barrier && !s.Run {
			for _, a := range k.storeArgs(&s) {
				if *a == i {
					n++
				}
			}
		}
	}
	return n
}

// hasTail reports whether a scalar loop takes vector loop c's last trips.
func (k *Kernel) hasTail(c int) bool {
	t := k.tailOf(c)
	return t >= 0 && k.Loops[t].Lanes == 0
}

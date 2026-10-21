package jit

import "github.com/jitllm/jitllm/jit/ir"

// Blocks runs a sum loop Size trips at a time: each block's trips unrolled
// as one chain from zero, the chain added into the sum once a block.
type Blocks struct {
	Size  int
	Kinds []PrimKind
}

func (Blocks) Name() string { return "blocks" }

func (b Blocks) Apply(prog []*Nest, t Target) ([]*Nest, error) {
	return eachNest(b.Name(), prog, b.apply)
}

func (b Blocks) apply(n *Nest) ([]*Nest, bool, error) {
	B := b.Size
	if !covers(b.Kinds, n.Kind) {
		return nil, false, nil
	}
	var r *Nest
	for c := range n.Loops {
		folds := n.foldsOf(c)
		it := n.iterOf(c)
		if B <= 1 || n.Loops[c].Trip%B != 0 || len(folds) != 1 || it < 0 || !n.sumFold(folds[0]) {
			continue
		}
		if r == nil {
			r = n.clone()
		}
		f := folds[0]
		step := r.Nodes[f].A
		r.Loops[c].Trip /= B
		base := r.bin(OpMul, ir.U32, it, r.u32(B))
		chain := r.f32(0)
		for j := 0; j < B; j++ {
			k := base
			if j > 0 {
				k = r.bin(OpAdd, ir.U32, base, r.u32(j))
			}
			chain = r.copyOf(step, map[int]int{f: chain, it: k})
		}
		r.Nodes[f].A = r.bin(OpAdd, r.Nodes[f].Type, f, chain)
	}
	if r == nil {
		return nil, false, nil
	}
	return []*Nest{r}, true, nil
}

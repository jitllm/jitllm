package jit

import "testing"

func (k *Kernel) explainFuse(t *testing.T, a, b int) {
	ia, ib := k.iterOf(a), k.iterOf(b)
	t.Logf("fuse %d %d: range %v %v trips %d %d inner %v %v storesAt(b) %v iters %d %d folds(b) %v",
		a, b, k.Loops[a].Range, k.Loops[b].Range, k.Loops[a].Trip, k.Loops[b].Trip, k.hasInner(a), k.hasInner(b), k.storesAt(b), ia, ib, k.foldsOf(b))
	for _, s := range k.Stores {
		if s.At == a && !s.Barrier && !s.Run {
			t.Logf("  store buf %d %v ax %v", s.Buf, k.Bufs[s.Buf].Operand.Name, s.Ax)
		}
	}
	for _, f := range k.foldsOf(b) {
		for i, nd := range k.Nodes {
			if nd.Op == OpLoad && k.reachesNode(k.Nodes[f].A, i) {
				t.Logf("  fold %d load %d buf %d %s ax %v", f, i, nd.Buf, k.Bufs[nd.Buf].Operand.Name, nd.Ax)
			}
		}
	}
}

func (k *Kernel) reachesNode(root, x int) bool {
	seen := map[int]bool{}
	var walk func(i int) bool
	walk = func(i int) bool {
		if i == x {
			return true
		}
		if seen[i] {
			return false
		}
		seen[i] = true
		nd := k.Nodes[i]
		for _, a := range k.Operands(&nd) {
			if walk(*a) {
				return true
			}
		}
		return false
	}
	return walk(root)
}

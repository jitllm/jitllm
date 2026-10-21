package jit

// copies is, for each node, the head of a pair whose rows it is made from:
// a load of a pair's second head's buffer that head, any node the largest
// of its operands', and a node made of no buffer but constants -- a sum of
// nothing and what adds it -- the head of the nodes that read it, where
// they agree.
func (k *Kernel) copies() []int {
	out := make([]int, len(k.Nodes))
	paired := false
	for _, b := range k.Bufs {
		paired = paired || b.Operand.Copy > 0
	}
	if !paired {
		return out
	}
	for changed := true; changed; {
		changed = false
		for i, nd := range k.Nodes {
			c := 0
			if nd.Op == OpLoad {
				c = k.Bufs[nd.Buf].Operand.Copy
			}
			for _, a := range k.args(&nd) {
				if *a != i {
					c = max(c, out[*a])
				}
			}
			if c > out[i] {
				out[i], changed = c, true
			}
		}
		for i := range k.Nodes {
			if out[i] != 0 || k.Nodes[i].Op == OpConst || !k.headless(i) {
				continue
			}
			readers, c := 0, -1
			for j, nd := range k.Nodes {
				if j == i {
					continue
				}
				for _, a := range k.args(&nd) {
					if *a == i {
						readers++
						if c < 0 || out[j] == c {
							c = out[j]
						} else {
							c = 0
						}
					}
				}
			}
			if readers > 0 && c > 0 {
				out[i], changed = c, true
			}
		}
	}
	return out
}

// headless reports whether node i is made from no buffer: constants and
// sums of nothing.
func (k *Kernel) headless(i int) bool {
	seen := map[int]bool{}
	var walk func(i int) bool
	walk = func(i int) bool {
		if seen[i] {
			return true
		}
		seen[i] = true
		nd := k.Nodes[i]
		if nd.Op == OpLoad || nd.Op == OpParam {
			return false
		}
		for _, a := range k.args(&nd) {
			if !walk(*a) {
				return false
			}
		}
		return true
	}
	return walk(i)
}

// twins is, for each node of a pair's later head, the first head's node it
// copies: a sum the first head's sum at its place among its loop's sums,
// any other node the same operation on the twins of its operands, a load of
// the first head's buffer of that load's operand at the same place.
func (k *Kernel) twins(cp []int) map[int]int {
	out := map[int]int{}
	var twin func(i int) (int, bool)
	twin = func(i int) (int, bool) {
		if cp[i] == 0 {
			return i, true
		}
		if t, ok := out[i]; ok {
			return t, t >= 0
		}
		for j := range k.Nodes {
			if cp[j] != 0 || k.Nodes[j].Op != k.Nodes[i].Op {
				continue
			}
			before := map[int]bool{}
			for x := range out {
				before[x] = true
			}
			out[i] = j
			if k.sameShapeAs(i, j, twin) {
				return j, true
			}
			for x := range out {
				if !before[x] {
					delete(out, x)
				}
			}
		}
		out[i] = -1
		return -1, false
	}
	for c := range k.Loops {
		byCopy := map[int][]int{}
		for _, f := range k.foldsOf(c) {
			byCopy[cp[f]] = append(byCopy[cp[f]], f)
		}
		for j, fs := range byCopy {
			if j == 0 {
				continue
			}
			for x, f := range fs {
				if x < len(byCopy[0]) {
					out[f] = byCopy[0][x]
				}
			}
		}
	}
	for i := range k.Nodes {
		if cp[i] > 0 {
			twin(i)
		}
	}
	return out
}

// sameShapeAs reports whether node i is node j for another head: the same
// operation, its operands j's operands' copies, a load the same place of
// the same operand's other head's buffer.
func (k *Kernel) sameShapeAs(i, j int, twin func(int) (int, bool)) bool {
	a, b := k.Nodes[i], k.Nodes[j]
	if a.Op != b.Op || a.Type != b.Type || a.Imm != b.Imm || a.At != b.At || a.Mem != b.Mem || a.Ptr != b.Ptr && a.Op != OpLoad {
		return false
	}
	if a.Op == OpLoad {
		oa, ob := k.Bufs[a.Buf].Operand, k.Bufs[b.Buf].Operand
		oa.Copy, ob.Copy = 0, 0
		if oa != ob {
			return false
		}
	}
	x, y := k.args(&a), k.args(&b)
	if len(x) != len(y) {
		return false
	}
	for n := range x {
		if *x[n] == i && *y[n] == j {
			continue
		}
		if p, q := k.Nodes[*x[n]], k.Nodes[*y[n]]; p.Op == OpConst && q.Op == OpConst && p.Imm == q.Imm && p.Type == q.Type {
			continue
		}
		t, ok := twin(*x[n])
		if !ok || t != *y[n] {
			return false
		}
	}
	return true
}

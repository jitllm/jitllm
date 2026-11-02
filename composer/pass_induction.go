package composer

import "github.com/jitllm/jitllm/composer/ir"

// Induction carries a loop's iteration as a value where the code reads it
// as data rather than as an address: a sum of the loop from 0, one more
// each trip; in a vector loop from each lane's index, the lanes more each
// trip; and in the loop that takes a vector loop's remaining trips, where
// it reads the trips done and its own, on from the vector loop's. Tail,
// where set, is 1 + the register of that loop's step.
type Induction struct {
	Tail int
}

func (Induction) Name() string { return "induction" }

func (p Induction) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		live, addr := k.Live(), k.addressNodes()
		vec := map[int]int{}
		for c := range k.Loops {
			it := k.iterOf(c)
			if it < 0 {
				continue
			}
			l := k.Loops[c]
			from, data := it, k.dataReaders(it, live, addr)
			mixed := k.mixedReaders(it, live, addr)
			v, cont := vec[l.Of-1]
			if l.Lanes == 0 && l.Of > 0 && cont {
				for d, nd := range k.Nodes {
					if nd.Op == OpAdd && nd.B == it && live[d] {
						if dr := k.dataReaders(d, live, addr); len(dr) > 0 {
							from, data = d, dr
						}
					}
				}
			}
			if l.Lanes == 0 || from != it {
				mixed = nil
			}
			if len(data) == 0 && len(mixed) == 0 {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			start, step := m.add(Node{Op: OpConst, Type: ir.I32}), int64(1)
			switch {
			case l.Lanes > 0:
				start, step = m.add(Node{Op: OpLanes, Type: ir.I32, Imm: int64(l.Lanes)}), int64(l.Lanes*max(1, l.Copies))
			case from != it:
				start = v
			}
			f := m.add(Node{Op: OpFold, At: c, Type: ir.I32, B: start, Running: true})
			for _, r := range data {
				for _, a := range m.Operands(&m.Nodes[r]) {
					if *a == from {
						*a = f
					}
				}
			}
			// A value computed from the iteration that also feeds an address
			// keeps the iteration there and reads the lanes' where it is data:
			// its chain from the iteration is made again over the fold.
			made := map[int]int{it: f}
			var over func(i int) int
			over = func(i int) int {
				if j, ok := made[i]; ok {
					return j
				}
				if !k.readsNode(i, it) {
					return i
				}
				nd := m.Nodes[i]
				for _, a := range m.Operands(&nd) {
					*a = over(*a)
				}
				made[i] = m.add(nd)
				return made[i]
			}
			for _, r := range mixed {
				if r.store {
					m.Stores[r.at].Val = over(m.Stores[r.at].Val)
					continue
				}
				nd := &m.Nodes[r.at]
				for _, a := range m.Operands(nd) {
					if *a != it && addr[*a] && k.readsNode(*a, it) {
						*a = over(*a)
					}
				}
			}
			st := m.add(Node{Op: OpConst, Type: ir.I32, Imm: step})
			if from != it {
				m.Nodes[st].Hint = p.Tail
			}
			m.Nodes[f].A = m.bin(OpAdd, ir.I32, f, st)
			if l.Lanes > 0 {
				vec[c] = f
				// An unrolled copy's iteration, the trip's plus a whole
				// number of vectors, read as data: the lanes' plus as many.
				for d, nd := range k.Nodes {
					if nd.Op != OpAdd || nd.A != it || !live[d] || k.Nodes[nd.B].Op != OpConst {
						continue
					}
					dr := k.dataReaders(d, live, addr)
					if len(dr) == 0 {
						continue
					}
					off := m.bin(OpAdd, ir.I32, f, m.add(Node{Op: OpConst, Type: ir.I32, Imm: k.Nodes[nd.B].Imm}))
					for _, r := range dr {
						for _, a := range m.Operands(&m.Nodes[r]) {
							if *a == d {
								*a = off
							}
						}
					}
				}
			}
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// addressNodes is the nodes that feed an index a load or a store reads:
// every node an index is computed from, a fold's reads not followed, as
// readsNode walks them.
func (k *Kernel) addressNodes() map[int]bool {
	seen := map[int]bool{}
	var stack []int
	push := func(i int) {
		if !seen[i] {
			seen[i] = true
			stack = append(stack, i)
		}
	}
	for _, nd := range k.Nodes {
		if nd.Op == OpLoad {
			for _, a := range k.Operands(&nd) {
				push(*a)
			}
		}
	}
	for _, s := range k.Stores {
		if s.Barrier || s.Run {
			continue
		}
		push(s.Ix)
		for _, a := range s.Ax {
			push(a)
		}
	}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if nd := k.Nodes[i]; nd.Op != OpFold {
			for _, a := range k.Operands(&nd) {
				push(*a)
			}
		}
	}
	return seen
}

// readsNode reports whether node i is computed from node x.
func (k *Kernel) readsNode(i, x int) bool {
	if i == x {
		return true
	}
	nd := k.Nodes[i]
	if nd.Op == OpFold {
		return false
	}
	for _, a := range k.Operands(&nd) {
		if k.readsNode(*a, x) {
			return true
		}
	}
	return false
}

// mixedReader is a data reader of a value that also feeds an address: node
// at, or where store is set the value Stores[at] writes.
type mixedReader struct {
	at    int
	store bool
}

// mixedReaders is what reads, as data, a value computed from node i that
// feeds an address: a live node that is not a load and feeds no address,
// and a store's value, the iteration itself among those.
func (k *Kernel) mixedReaders(i int, live, addr map[int]bool) []mixedReader {
	var out []mixedReader
	for j, nd := range k.Nodes {
		if !live[j] || nd.Op == OpLoad || addr[j] {
			continue
		}
		for _, a := range k.Operands(&nd) {
			if *a != i && addr[*a] && k.readsNode(*a, i) {
				out = append(out, mixedReader{at: j})
				break
			}
		}
	}
	for j, st := range k.Stores {
		if !st.Barrier && !st.Run && (st.Val == i || addr[st.Val] && k.readsNode(st.Val, i)) {
			out = append(out, mixedReader{at: j, store: true})
		}
	}
	return out
}

// dataReaders is the live nodes that read node i as data: not a load, and
// feeding no address.
func (k *Kernel) dataReaders(i int, live, addr map[int]bool) []int {
	var out []int
	for j, nd := range k.Nodes {
		if !live[j] || nd.Op == OpLoad || addr[j] {
			continue
		}
		for _, a := range k.Operands(&nd) {
			if *a == i {
				out = append(out, j)
				break
			}
		}
	}
	return out
}

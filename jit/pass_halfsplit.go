package jit

import "github.com/jitllm/jitllm/jit/ir"

// HalfSplit cuts a chain of byte dots of unsigned bytes where its sixteen-bit
// run (halfRuns) would pass an int16 into further runs, each from zero and
// added to the sum before it once widened, on a device whose ISA holds Tier
// and none of Unless: a dot past the bound otherwise widens alone. The
// sums are integers, so the reassociation is exact.
type HalfSplit struct {
	Tier, Unless uint64
}

func (HalfSplit) Name() string { return "halfsplit" }

func (p HalfSplit) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	if d.Kind != DeviceCPU || d.ISA&p.Tier != p.Tier || d.ISA&p.Unless != 0 {
		return ks, nil
	}
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Unsplit {
			return nil, false, nil
		}
		var m *Kernel
		readers, reader := k.readerCounts()
		for i := range k.Nodes {
			if k.dotBound(i) < 0 || !k.dotZero(k.Nodes[i].C) {
				continue
			}
			chain := []int{i}
			for cur := i; readers[cur] == 1; {
				j := reader[cur]
				nd := &k.Nodes[j]
				if k.dotBound(j) < 0 || nd.C != cur || nd.A == cur || nd.B == cur {
					break
				}
				chain = append(chain, j)
				cur = j
			}
			cuts := k.halfCuts(chain)
			if len(cuts) == 0 {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			zero, acc, at := k.Nodes[i].C, -1, -1
			for _, c := range cuts {
				prev, last := chain[c.start-1], chain[c.end]
				if at == c.start-1 {
					prev = acc
				}
				m.Nodes[chain[c.start]].C = zero
				s := m.add(Node{Op: OpAdd, Type: ir.I32, A: prev, B: last})
				acc, at = s, c.end
				if c.end+1 < len(chain) {
					m.Nodes[chain[c.end+1]].C = s
				} else {
					m.redirect(last, s, s)
				}
			}
		}
		if m == nil {
			return nil, false, nil
		}
		m.Split = true
		return []*Kernel{m}, true, nil
	})
}

type halfCut struct{ start, end int }

// halfCuts is the runs after the first that chain's dots, in order, fall
// into at the int16 bound, each of two dots or more; a dot left alone past
// one stays on the sum before it.
func (k *Kernel) halfCuts(chain []int) []halfCut {
	var cuts []halfCut
	used, start := 0, 0
	for x := 0; x < len(chain); x++ {
		b := k.dotBound(chain[x])
		if x == start || used+b <= 32767 {
			used += b
			continue
		}
		if start > 0 && x-1 > start {
			cuts = append(cuts, halfCut{start, x - 1})
		}
		start, used = x, b
	}
	if start > 0 && len(chain)-1 > start {
		cuts = append(cuts, halfCut{start, len(chain) - 1})
	}
	return cuts
}

// dotBound is twice the largest byte of node i's unsigned operand times
// 127, a bound on its pairs' sums; -1 where node i is no byte dot of
// unsigned bytes.
func (k *Kernel) dotBound(i int) int {
	nd := &k.Nodes[i]
	if !nd.Op.byteDot() || nd.Op != OpDot4U && !k.unsignedBytes(nd.A) {
		return -1
	}
	w, most := k.wordBits(nd.A, 6), 0
	for b := range 4 {
		most = max(most, int(w>>(8*b)&0xFF))
	}
	return 2 * most * 127
}

// readerCounts is how many times each node is read, stores twice, and
// one node that reads it.
func (k *Kernel) readerCounts() (map[int]int, map[int]int) {
	readers, reader := map[int]int{}, map[int]int{}
	for j := range k.Nodes {
		for _, a := range k.Operands(&k.Nodes[j]) {
			readers[*a]++
			reader[*a] = j
		}
	}
	for x := range k.Stores {
		for _, a := range k.storeArgs(&k.Stores[x]) {
			readers[*a] += 2
		}
	}
	return readers, reader
}

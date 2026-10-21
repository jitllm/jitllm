package jit

import "github.com/jitllm/jitllm/jit/ir"

// Parts runs consecutive independent primitives of Kinds as parts of one
// launch: the launch as wide as the widest part, each part's element the
// thread's index clamped to its own.
type Parts struct {
	Kinds []PrimKind
}

func (Parts) Name() string { return "parts" }

func (p Parts) Apply(prog []*Nest, t Target) ([]*Nest, error) {
	var out []*Nest
	for i := 0; i < len(prog); {
		j := i + 1
		for j < len(prog) && p.joins(prog[i:j], prog[j]) {
			j++
		}
		if j-i < 2 || !p.joins(nil, prog[i]) {
			out = append(out, prog[i])
			i++
			continue
		}
		m := p.merge(prog[i:j])
		for _, n := range prog[i:j] {
			for _, ps := range n.Passes {
				if !m.ran(ps) {
					m.Passes = append(m.Passes, ps)
				}
			}
		}
		m.Passes = append(m.Passes, p.Name())
		out = append(out, m)
		i = j
	}
	return out, nil
}

func (p Parts) joins(group []*Nest, n *Nest) bool {
	if !covers(p.Kinds, n.Kind) || n.Work || len(n.Space) != 1 || len(n.Loops) != 0 {
		return false
	}
	for _, g := range group {
		if g.Group != n.Group || conflicts(g, n) || conflicts(n, g) {
			return false
		}
	}
	return true
}

func (p Parts) merge(parts []*Nest) *Nest {
	N := 0
	for _, n := range parts {
		N = max(N, n.Space[0])
	}
	m := &Nest{Name: "parts", Space: []int{N}, Group: parts[0].Group, Kind: parts[0].Kind}
	flat := m.add(Node{Op: OpFlat, Type: ir.U32})
	for _, n := range parts {
		el := m.bin(OpMin, ir.U32, flat, m.u32(n.Space[0]-1))
		bufOf := map[int]int{}
		for i, b := range n.Bufs {
			j := -1
			for k, mb := range m.Bufs {
				if mb.Operand == b.Operand && mb.Kind == b.Kind {
					j = k
				}
			}
			if j < 0 {
				j = m.buf(b)
			}
			bufOf[i] = j
		}
		nodeOf := make([]int, len(n.Nodes))
		for i, nd := range n.Nodes {
			if nd.Op == OpCoord {
				nodeOf[i] = el
				continue
			}
			if nd.Op == OpLoad {
				nd.Buf = bufOf[nd.Buf]
			}
			for _, a := range n.args(&nd) {
				*a = nodeOf[*a]
			}
			nodeOf[i] = m.add(nd)
		}
		for _, s := range n.Stores {
			m.Stores = append(m.Stores, Store{Buf: bufOf[s.Buf], Ix: nodeOf[s.Ix], Val: nodeOf[s.Val], At: -1})
		}
	}
	return m
}

func (n *Nest) ran(pass string) bool {
	for _, p := range n.Passes {
		if p == pass {
			return true
		}
	}
	return false
}

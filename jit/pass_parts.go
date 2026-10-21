package jit

import "github.com/jitllm/jitllm/jit/ir"

// Parts runs consecutive independent maps of Kinds as parts of one launch:
// the launch as wide as the widest part, each part's element the thread's
// index held to its own.
type Parts struct {
	Kinds []PrimKind
}

func (Parts) Name() string { return "parts" }

func (p Parts) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	var out []*Kernel
	for i := 0; i < len(ks); {
		j := i + 1
		for j < len(ks) && p.joins(ks[i:j], ks[j]) {
			j++
		}
		if j-i < 2 || !p.joins(nil, ks[i]) {
			out = append(out, ks[i])
			i++
			continue
		}
		m := p.merge(ks[i:j])
		for _, n := range ks[i:j] {
			for _, ps := range n.Passes {
				if !m.ran(ps) {
					m.Passes = append(m.Passes, ps)
				}
			}
		}
		m.Passes = append(m.Passes, p.Name())
		m.compact()
		out = append(out, m)
		i = j
	}
	return out, nil
}

func (p Parts) joins(group []*Kernel, n *Kernel) bool {
	if !covers(p.Kinds, n.Kind) || n.Work || len(n.Loops) != 0 {
		return false
	}
	for _, g := range group {
		if g.Group != n.Group || conflicts(g, n) || conflicts(n, g) {
			return false
		}
	}
	return true
}

func (p Parts) merge(parts []*Kernel) *Kernel {
	N := 0
	for _, n := range parts {
		N = max(N, n.elements())
	}
	m := &Kernel{Name: "parts", Space: []int{N}, Group: parts[0].Group, Kind: parts[0].Kind, Part: parts[0].Part}
	flat := m.add(Node{Op: OpFlat, Type: ir.U32})
	for _, n := range parts {
		el := m.bin(OpMin, ir.U32, flat, m.u32(n.elements()-1))
		coords := m.decode(el, n.Space)
		leaf := func(i int, nd Node) (int, bool) {
			if nd.Op == OpCoord {
				return coords[nd.A], true
			}
			return 0, false
		}
		for _, s := range n.Stores {
			ns := Store{Buf: m.bufLike(n.Bufs[s.Buf]), Val: copyInto(m, n, s.Val, leaf), At: -1}
			if s.Ax != nil {
				for _, a := range s.Ax {
					ns.Ax = append(ns.Ax, copyInto(m, n, a, leaf))
				}
			} else {
				ns.Ix = copyInto(m, n, s.Ix, leaf)
			}
			m.Stores = append(m.Stores, ns)
		}
	}
	return m
}

func (k *Kernel) ran(pass string) bool {
	for _, p := range k.Passes {
		if p == pass {
			return true
		}
	}
	return false
}

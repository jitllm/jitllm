package jit

import "github.com/jitllm/jitllm/jit/ir"

// HalfWords reads a binary16 picked from a word of halves as the half
// itself where the half's index is one value: the low half of word W
// shifted right by S, where S is 16 times the low bit of a value whose
// upper bits W counts, is the half at that value's index of the same
// storage read as halves. Rows that share a word their scales are packed
// in then read their halves side by side; a word whose two halves the
// same row holds is left as it is.
type HalfWords struct{}

func (HalfWords) Name() string { return "halfwords" }

func (p HalfWords) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		var g *alg
		for i := range k.Nodes {
			nd := k.Nodes[i]
			if nd.Op != OpWidenHalf {
				continue
			}
			x := k.Nodes[nd.A]
			if x.Op != OpShr {
				continue
			}
			word, shift := x.A, x.B
			ld := k.Nodes[word]
			if ld.Op != OpLoad || ld.Type != ir.U32 || ld.Ax != nil || ld.At > 0 || ld.Ptr > 0 || k.Bufs[ld.Buf].Half || k.Bufs[ld.Buf].Kind != BufIn {
				continue
			}
			if m == nil {
				m = k.clone()
				g = newAlg(m)
			}
			idx := g.times(g.of(ld.A), lin{"": 2})
			half, ok := g.bit16(g.of(shift))
			if !ok {
				continue
			}
			idx.add(half, 1)
			// The bit must merge into the word's index: its value the one
			// the word counts pairs of.
			merged := g.merged(idx)
			kept := false
			for m := range half {
				_, kept = merged[m]
			}
			if kept {
				continue
			}
			b := m.halfView(ld.Buf)
			m.Nodes[i] = Node{Op: OpLoad, Type: ir.F32, Buf: b, A: g.build(merged)}
		}
		if m == nil {
			return nil, false, nil
		}
		m.compact()
		return []*Kernel{m}, true, nil
	})
}

// bit16 is s/16 where s is 16 times one bit: a constant 0 or 16, or a
// field of one bit times 16.
func (g *alg) bit16(s lin) (lin, bool) {
	if c, ok := s.constant(); ok {
		return lin{"": c / 16}, c == 0 || c == 16
	}
	if len(s) != 1 {
		return nil, false
	}
	for m, v := range s {
		ids := monoIDs(m)
		if v != 16 || len(ids) != 1 {
			return nil, false
		}
		if a := g.atoms[ids[0]]; !a.field || a.mask != 1 {
			return nil, false
		}
		return lin{m: 1}, true
	}
	return nil, false
}

// halfView is buffer b's storage read as binary16 elements.
func (k *Kernel) halfView(b int) int {
	src := k.Bufs[b]
	for i, x := range k.Bufs {
		if x.Operand == src.Operand && x.Kind == src.Kind && x.Half {
			return i
		}
	}
	v := src
	v.Half, v.Elem, v.Shape = true, ir.F32, nil
	return k.buf(v)
}

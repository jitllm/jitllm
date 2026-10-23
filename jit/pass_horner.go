package jit

import "slices"

// Horner gives each chain of fmas that multiply the one before -- p' = x*p
// + c, a polynomial in Horner's form -- its registers: chain k, in the
// order the chains were made, its steps and its leading constant Acc[k];
// a step's constant addend, and a multiplier made for that step alone,
// Coef; the multiplier the chains share, Var. A chain whose steps are
// hinted already is left as it is.
type Horner struct {
	Acc       []int
	Coef, Var int
}

func (Horner) Name() string { return "horner" }

func (p Horner) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		chains := k.hornerChains()
		if len(chains) == 0 || len(p.Acc) == 0 {
			return nil, false, nil
		}
		m := k.clone()
		for ci, ch := range chains {
			if ci >= len(p.Acc) {
				break
			}
			acc := p.Acc[ci] + 1
			for si, s := range ch {
				nd := m.Nodes[s]
				m.Nodes[s].Hint = acc
				prev, mult := nd.A, nd.B
				if si == 0 || ch[si-1] != nd.A {
					prev, mult = nd.B, nd.A
				}
				if si == 0 {
					if m.constLoad(prev) {
						m.Nodes[prev].Hint = acc
					}
					if m.constLoad(mult) {
						m.Nodes[mult].Hint, mult = acc, prev
					}
				}
				if m.constLoad(nd.C) && m.uses(nd.C) == 1 {
					m.Nodes[nd.C].Hint = p.Coef + 1
				}
				switch {
				case m.uses(mult) == 1 && !m.constLoad(mult):
					m.Nodes[mult].Hint = p.Coef + 1
				case m.Nodes[mult].Hint == 0:
					m.Nodes[mult].Hint = p.Var + 1
				}
			}
		}
		return []*Kernel{m}, true, nil
	})
}

// constLoad reports whether node i loads one fixed word.
func (k *Kernel) constLoad(i int) bool {
	nd := k.Nodes[i]
	return nd.Op == OpLoad && (k.Bufs[nd.Buf].Operand.Kind == OperandConsts || k.Bufs[nd.Buf].Words != nil)
}

// hornerChains is the kernel's chains of two or more fmas each multiplying
// the one before, which only it reads, in the order they were made; none
// with a step hinted.
func (k *Kernel) hornerChains() [][]int {
	next := map[int]int{}
	prev := map[int]bool{}
	for i, nd := range k.Nodes {
		if nd.Op != OpFma {
			continue
		}
		for _, x := range []int{nd.A, nd.B} {
			if k.Nodes[x].Op == OpFma && k.uses(x) == 1 {
				next[x] = i
				prev[i] = true
				break
			}
		}
	}
	var out [][]int
	for i, nd := range k.Nodes {
		if nd.Op != OpFma || prev[i] {
			continue
		}
		if _, ok := next[i]; !ok {
			continue
		}
		ch := []int{i}
		for x, ok := next[i]; ok; x, ok = next[x] {
			ch = append(ch, x)
		}
		if !slices.ContainsFunc(ch, func(s int) bool { return k.Nodes[s].Hint > 0 }) {
			out = append(out, ch)
		}
	}
	return out
}

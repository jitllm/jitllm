package jit

import (
	"slices"

	"github.com/jitllm/jitllm/jit/ir"
)

// ShiftChain shifts a word a vector holds, read right by several
// constants, each time from its previous shift by the difference: (x >> a) >> (b-a) is x >> b for
// b < 32. Regs, where set, are the registers of the word and its shifts,
// of a mask every shift is anded with, and of each masked shift in turn.
type ShiftChain struct {
	Regs []int
}

func (ShiftChain) Name() string { return "shiftchain" }

func (p ShiftChain) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		vw := k.VectorWords()
		for x := range k.Nodes {
			if !vw[x] {
				continue
			}
			var shifts []int
			for i, nd := range k.Nodes {
				if nd.Op == OpShr && nd.A == x && nd.Type == ir.U32 && k.Nodes[nd.B].Op == OpConst {
					shifts = append(shifts, i)
				}
			}
			if len(shifts) < 2 {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			amount := func(i int) int64 { return m.Nodes[m.Nodes[i].B].Imm }
			slices.SortFunc(shifts, func(a, b int) int { return int(amount(a) - amount(b)) })
			prev, at := x, int64(0)
			for _, i := range shifts {
				a := amount(i)
				m.Nodes[i].A, m.Nodes[i].B = prev, m.u32(int(a-at))
				prev, at = i, a
			}
			if len(p.Regs) >= 2 {
				m.chainHints(x, shifts, p.Regs)
			}
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// chainHints gives word x and its shifts one register and the mask each
// is anded with another.
func (k *Kernel) chainHints(x int, shifts []int, regs []int) {
	label := 1
	for _, nd := range k.Nodes {
		label = max(label, nd.Share+1)
	}
	for _, i := range append([]int{x}, shifts...) {
		k.Nodes[i].Hint, k.Nodes[i].Share = regs[0]+1, label
	}
	for j, nd := range k.Nodes {
		if nd.Op != OpAnd {
			continue
		}
		mask := func(j int) bool {
			b := k.Nodes[j]
			if b.Op == OpBroadcast {
				b = k.Nodes[b.A]
			}
			return b.Op == OpConst || b.Op == OpLoad && (k.Bufs[b.Buf].Operand.Kind == OperandConsts || k.Bufs[b.Buf].Words != nil)
		}
		for d, i := range append([]int{x}, shifts...) {
			m := -1
			switch {
			case nd.A == i && mask(nd.B):
				m = nd.B
			case nd.B == i && mask(nd.A):
				m = nd.A
			}
			if m < 0 {
				continue
			}
			k.Nodes[m].Hint = regs[1] + 1
			if d+2 < len(regs) {
				k.Nodes[j].Hint = regs[d+2] + 1
			}
		}
	}
}

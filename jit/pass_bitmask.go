package jit

import (
	"math/bits"

	"github.com/jitllm/jitllm/jit/ir"
)

// BitMask holds a predicate that bit k of a word is set, where it is read
// only as a mask of a word's bits, as that word with the bit shifted to the
// sign and the sign spread over the word.
type BitMask struct{}

func (BitMask) Name() string { return "bitmask" }

func (p BitMask) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		var m *Kernel
		for i, nd := range k.Nodes {
			if nd.Op != OpLess || !k.isConst(nd.A, 0) {
				continue
			}
			and := k.Nodes[nd.B]
			if and.Op != OpAnd || and.Type == ir.F32 || k.Nodes[and.B].Op != OpConst {
				continue
			}
			bit := uint32(k.Nodes[and.B].Imm)
			if bits.OnesCount32(bit) != 1 || !k.onlyMasks(i) {
				continue
			}
			if m == nil {
				m = k.clone()
			}
			up := m.bin(OpShl, and.Type, and.A, m.u32(31-bits.TrailingZeros32(bit)))
			m.Nodes[up].Hint = nd.Hint
			m.Nodes[i] = Node{Op: OpSar, Type: and.Type, A: up, B: m.u32(31), Hint: nd.Hint}
		}
		if m == nil {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// onlyMasks reports whether every reader of node i takes it as a mask: an
// and of bits, or a select's predicate.
func (k *Kernel) onlyMasks(i int) bool {
	n := 0
	for _, nd := range k.Nodes {
		for x, a := range k.args(&nd) {
			if *a != i {
				continue
			}
			n++
			if !(nd.Op == OpAnd || nd.Op == OpSelect && x == 0) {
				return false
			}
		}
	}
	for _, s := range k.Stores {
		for _, a := range k.storeArgs(&s) {
			if *a == i {
				return false
			}
		}
	}
	return n > 0
}

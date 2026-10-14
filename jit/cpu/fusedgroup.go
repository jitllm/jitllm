package cpu

import (
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// PackedFusedGroup is how many rows one iteration of the fused kernel covers,
// for most formats. Ask PackedFusedGroupOf instead: Args.Rows counts groups,
// so a caller assuming 16 against a kernel emitted at 8 walks twice the rows it
// was handed and reads off the end of the payload.
const PackedFusedGroup = 16

// PackedFusedGroupOf is the group size for t.
//
// It is eight for a 4-bit primary plane with a secondary plane (Q5_K, Q5_0)
// because of the register file: at sixteen rows the amd64 budget reads
//
//	3 (mask, da, bscale) + 2 (accumulators) + 3 (pay, tmp, t2) + 8 (act)
//	  + 1 (the pre-shifted secondary mask)  =  17, and there are 16.
//
// Halving the group lands on exactly 16. Spilling the secondary mask instead
// would put a load in the payload loop, which does not amortise.
//
// It lives in an untagged file because a group size is a property of the
// layout, not the host: amd64's fused and token-tiled kernels and arm64's tiled
// one all divide by it.
func PackedFusedGroupOf(t quant.Type) int {
	q, ok := kernels.QuantOf(t)
	if !ok {
		return PackedFusedGroup
	}
	return packedFusedGroupK(q)
}

// packedFusedGroupK is PackedFusedGroupOf keyed by the DEVICE-layout quant,
// which is what the emitters already carry.
func packedFusedGroupK(q kernels.Quant) int {
	// Derived from the budget, not from a format's name, so a new format with a
	// secondary plane cannot silently get the default 16 and be refused by the
	// emitter. This is EmitPackedMatVecFused's own budget line.
	for _, group := range []int{PackedFusedGroup, 8} {
		if fusedRegisters(q, group) <= 16 {
			return group
		}
	}
	return 8 // let the emitter refuse by name rather than returning 0 here
}

// fusedRegisters is EmitPackedMatVecFused's register budget for one row group.
// It is the one place that arithmetic is written down for a caller that has to
// predict it; the emitter re-derives it and refuses if it is wrong.
func fusedRegisters(q kernels.Quant, group int) int {
	sub, bits, _, _ := kernels.Layout(q)
	_, scOff := kernels.ScaleLayout(q)
	pw := sub * bits / 32
	nact := pw
	if bits < 8 {
		nact = (8 / bits) * pw // one broadcast a field a word
	}
	need := 3 + group/8 + 3 + nact // mask, da, bscale + accumulators + pay/tmp/t2
	if scOff != 0 {
		need++
	}
	if kernels.HiPlane(q) != 0 {
		need++ // the pre-shifted secondary mask
	}
	if kernels.Codes(q) != nil {
		need++ // the code table
	}
	return need
}

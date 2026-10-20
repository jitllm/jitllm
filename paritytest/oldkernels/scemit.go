package oldkernels

import (
	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// scStreamAt locates sub-block subIdx's scale/min pair at run time, where the
// sub-block is not known when the kernel is built. A super-block is exactly
// kernels.ScStreamWords words (96 bits, eight 12-bit pairs), so a row's whole plane
// is one contiguous bit stream and sub-block i's pair starts at bit 12*i: word
// lo = 12*i/32 of the row, from bit `bit`, spilling into word hi = lo+1 only
// when it straddles (bit > 20). A pair that does not straddle reads hi = lo,
// whose bits scStreamPair shifts entirely away, and the tensor's last pair
// never straddles, so no index runs past the plane. (Splitting into
// super-block and word costs extra ops the issue-bound device matvec feels.)
func scStreamAt(b *ir.Builder, subIdx ir.Value) (lo, hi, bit ir.Value) {
	c := func(x int64) ir.Value { return b.Const(ir.U32, x) }
	pos := b.Mul(ir.U32, subIdx, c(12))
	lo, bit = b.Shr(ir.U32, pos, c(5)), b.And(ir.U32, pos, c(31))
	return lo, b.Add(ir.U32, lo, b.Select(ir.U32, b.Lt(ir.U32, c(20), bit), c(1), c(0))), bit
}

// scStreamPair is the funnel over the two words scStreamAt named: one shf.r on
// PTX (ir.OpFunnel).
func scStreamPair(b *ir.Builder, lo, hi, bit ir.Value) (sc, m ir.Value) {
	c := func(x int64) ir.Value { return b.Const(ir.U32, x) }
	pair := b.Funnel(lo, hi, bit)
	return b.And(ir.U32, pair, c(63)), b.And(ir.U32, b.Shr(ir.U32, pair, c(6)), c(63))
}

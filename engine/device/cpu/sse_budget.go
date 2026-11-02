package cpu

import "github.com/jitllm/jitllm/format/quant"

// The SSE tier's share of invariant I3 (budget.go): no single entry into
// generated code may run longer than maxCallNanos, because a goroutine inside
// JIT code cannot be async-preempted and every runtime.GC() waits for it.
//
// RowsPerCall prices a block (MaxBlocksPerCall() blocks a call whatever a
// block holds), which is the AVX2 tier's model. The SSE fused kernel on a
// low-power core costs 0.19-0.41 ns per element (Q4_0 cheapest, Q5_K dearest),
// so that chunk would run up to twenty times the budget on the k-quants.
//
// This tier's budget is therefore per element, at the worst format plus a
// margin: 0.5 ns an element puts a k=2048 call at 96 rows. Subnormal f16
// scales cost a microcode assist, still well under the budget.
const sseCallNanosPerElem = 0.5

// RowsPerCallFor is how many rows of a k-element packed weight of type t one
// kernel call may serve on tier t and stay inside maxCallNanos.
//
// For every tier but SSE it is RowsPerCall(k / t.BlockElems()) exactly, so a
// caller that switches to it changes nothing on those tiers. On the SSE tier it
// is the per-element budget above, rounded down to OutLine and never below it
// (RowsPerCall's own rule: a chunk smaller than a cache line of output costs
// more in false sharing than a longer call costs in GC latency).
func RowsPerCallFor(tier Tier, t quant.Type, k int) int {
	if tier != TierSSE {
		be := int(t.BlockElems())
		if be < 1 {
			be = 1
		}
		return RowsPerCall(k / be)
	}
	if k < 1 {
		return OutLine
	}
	r := int(float64(maxCallNanos) / sseCallNanosPerElem / float64(k))
	if r < OutLine {
		return OutLine
	}
	return r / OutLine * OutLine
}

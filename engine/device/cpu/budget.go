package cpu

import "fmt"

// maxCallNanos is invariant I3: no single entry into generated code may run
// longer than this.
//
// A goroutine inside generated code cannot be async-preempted (runtime.findfunc
// fails for a PC outside Go's text), so every runtime.GC() waits for it. The
// defence is a budget checked at generation time: a kernel that would run too
// long must fail to build rather than stall production with no stack trace.
const maxCallNanos = 100_000

// cyclesPerBlock is the cost of one block iteration: measured at roughly 10
// cycles per Q4_0 block, with a safety factor. Over-pessimism has a real cost:
// a larger value once capped ffn_down at fewer rows per call than a cache line
// of output, and adjacent workers ping-ponged the line.
const cyclesPerBlock = 16

// clockHz is the conservative sustained clock. Under-estimating it over-states
// the predicted time, which fails safe.
const clockHz = 3.0e9

// checkBudget refuses a Spec whose worst-case call would exceed the budget.
//
// It bounds the work per call, which the caller controls by chunking rows, so
// it is not a limit on model size. Entry costs about 10 ns, so chunking is
// nearly free.
func checkBudget(s Spec, codeLen int) error {
	if codeLen == 0 {
		return fmt.Errorf("jit: Emit produced no code")
	}
	// The generated loop is bounded by its runtime arguments, so the budget is
	// expressed as the largest (rows x blocks) a caller may pass.
	if n := MaxBlocksPerCall(); n < 1 {
		return fmt.Errorf("jit: budget of %d ns cannot fit a single block", maxCallNanos)
	}
	return nil
}

// MaxBlocksPerCall is how many weight blocks one call may process and stay
// inside the budget. Callers chunk rows against it.
func MaxBlocksPerCall() int {
	return int(maxCallNanos * clockHz / 1e9 / cyclesPerBlock)
}

// OutLine is how many float32 outputs share one 64-byte cache line.
//
// A parallel region must never hand two workers row ranges that land in the same
// line: they are disjoint in the program and adjacent in the hardware, so the
// line ping-pongs between cores and the kernel stalls on coherence traffic
// instead of memory. Chunks are therefore rounded to this.
const OutLine = 16

// RowsPerCall is how many rows of blocksPerRow blocks fit in one budgeted call.
//
// Always at least OutLine, even when the budget would prefer fewer: a chunk
// smaller than a cache line of output costs far more in false sharing than a
// slightly longer call costs in GC latency. And always at least one row, because
// refusing would make a model unrunnable rather than slow.
func RowsPerCall(blocksPerRow int) int {
	if blocksPerRow < 1 {
		return OutLine
	}
	r := MaxBlocksPerCall() / blocksPerRow
	if r < OutLine {
		return OutLine
	}
	return r / OutLine * OutLine
}

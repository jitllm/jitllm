package backend

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// GridLimited is a Device that can say how many workgroups of a kernel with a
// grid barrier (ir.OpGridBarrier) it holds resident at once, which is the most
// one launch of that kernel may carry: a barrier waiting on a workgroup that
// cannot start until a waiting one finishes never returns. A Device without it
// cannot run such a kernel.
type GridLimited interface {
	GridLimit(k Kernel) (int, error)
}

// GridLimit asks d for k's co-residency limit. A device that cannot answer
// refuses, which is the safe direction: a persistent kernel is never the only
// way to run its work.
func GridLimit(d Device, k Kernel) (int, error) {
	g, ok := d.(GridLimited)
	if !ok {
		return 0, fmt.Errorf("backend: %s states no co-residency limit, so a grid barrier cannot run on it", d.API())
	}
	return g.GridLimit(k)
}

// IRSource is a compiled Kernel that keeps the IR it was compiled from, so a
// launch sequence can be composed into one persistent kernel (ir.Persist).
type IRSource interface{ IR() *ir.Kernel }

// IROf is k's IR, or nil when its backend does not keep it.
func IROf(k Kernel) *ir.Kernel {
	if s, ok := k.(IRSource); ok {
		return s.IR()
	}
	return nil
}

// checkGrid refuses a launch of a grid-barrier kernel wider than the device
// holds resident -- RULE 13's kind of invariant, enforced where the grid is
// known rather than trusted to the caller.
func checkGrid(api, name string, grid bool, groups, limit int, limitErr error) error {
	if !grid {
		return nil
	}
	if limitErr != nil {
		return fmt.Errorf("backend: %s: %s has a grid barrier and this device cannot bound its residency: %w",
			api, name, limitErr)
	}
	if groups > limit {
		return fmt.Errorf("backend: %s: %s has a grid barrier and was launched %d groups wide; "+
			"this device holds %d resident at once", api, name, groups, limit)
	}
	return nil
}

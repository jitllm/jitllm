package probe

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/device/cpu"
)

// hostLanes is a vector's floats.
func (e emitter) hostLanes() int {
	if e.sse {
		return 4
	}
	return 8
}

// formLanes is the words one vector instruction the estimator counts in a
// form covers: a vector's.
func (e emitter) formLanes() int { return e.hostLanes() }

// hostCheck refuses an x86 host below the SSE tier.
func hostCheck(h *cpu.Target) error {
	if t := h.Tier(); t != cpu.TierAVX2 && t != cpu.TierSSE {
		return fmt.Errorf("probe: the host's measuring kernels need SSE4.1 and SSSE3 at least; this host is %v", t)
	}
	return nil
}

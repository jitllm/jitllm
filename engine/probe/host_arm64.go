package probe

import (
	"github.com/jitllm/jitllm/engine/device/cpu"
)

// hostLanes is a NEON vector's floats.
func (e emitter) hostLanes() int { return 4 }

// formLanes is the words one vector instruction the estimator counts in a
// form covers: the composer's arm64 vector is two NEON registers, eight
// lanes, counted as one.
func (e emitter) formLanes() int { return 8 }

// hostCheck accepts every arm64 host: the measuring kernels are base NEON.
func hostCheck(*cpu.Target) error { return nil }

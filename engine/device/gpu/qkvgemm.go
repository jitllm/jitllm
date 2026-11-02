package gpu

import (
	"github.com/jitllm/jitllm/backend"
)

// qkvSeg is a prompt chunk's q/k/v grid for one set of shapes: the
// GemmVoltaSegments kernel, its launch, which of q (0), k (1) and v (2) it
// carries, and the binary16 conversion they all read.
type qkvSeg struct {
	kern backend.Kernel
}

// qkvGemmSwapped hands the grid's k segment v's weights and v's k's (where
// they are one format and shape): the violation the q/k/v gate must see
// (SwapQKVGemmForTest, jitllmtest builds only).
var qkvGemmSwapped bool

package tier

import (
	"github.com/jitllm/jitllm/engine/nn"
)

// PrepHead is nn.LayerDevice's placement of the output norm and projection.
// The composer forms no head yet, so the host runs it.
func (g *devTier) PrepHead(h *nn.Head) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.LastErr = "the head has no composed form"
	return false
}

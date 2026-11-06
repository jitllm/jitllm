package gpu

import (
	"github.com/jitllm/jitllm/nn"
)

// A selecting block on the device (nn.LayerPlan.Select): the indexer's key
// rides the cached row after the attention's own (nn.LayerPlan.KeyW), so it
// pages, migrates and is stored with it, and the attention reads the
// positions its composed selection keeps.

// selectKeys is the most keys a gathered row attends: the call's, at most
// the blocks the selection keeps.
func selectKeys(p *nn.LayerPlan, keys int) int {
	if p.Select == nil {
		return keys
	}
	return min(keys, (min(p.Select.Keep*p.Select.Block, max(p.MaxSeq, 1))+scoreGrain-1)/scoreGrain*scoreGrain)
}

package composer

import (
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
)

// ActWindow is how many activation elements share one int8 scale for a model
// whose matrices are of types: decided once for the model, so every device
// quantizes a row alike and the devices agree on token ids. A k-quant's
// weights carry a scale per 256-element super-block, so a model of k-quants
// alone takes the super-block; a 32-wide weight block wants a 32-wide
// scale, and a model holding one keeps the narrow window. The window is the
// Step's ActWindow every composition of the model is asked at.
func ActWindow(types []quant.Type) int {
	const narrow, super = 32, 256
	any := false
	for _, t := range types {
		q, ok := layout.QuantOf(t)
		if !ok {
			continue
		}
		if layout.NarrowScales(q) {
			return narrow
		}
		any = true
	}
	if !any {
		return narrow
	}
	return super
}

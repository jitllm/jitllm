package layout_test

import (
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
)

// packedRefFormats is every packed format with its device-layout id, derived
// from quant.PackedTypes so a new format cannot be left out.
var packedRefFormats = func() (fs []struct {
	name string
	g    quant.Type
	k    layout.Quant
}) {
	for _, g := range quant.PackedTypes {
		k, ok := layout.QuantOf(g)
		if !ok {
			panic(g.String() + " is in quant.PackedTypes and has no device layout")
		}
		fs = append(fs, struct {
			name string
			g    quant.Type
			k    layout.Quant
		}{g.String(), g, k})
	}
	return fs
}()

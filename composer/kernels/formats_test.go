package kernels_test

import (
	"github.com/jitllm/jitllm/composer/kernels"
	"github.com/jitllm/jitllm/format/quant"
)

// packedRefFormats is every packed format with its device-layout id, derived
// from quant.PackedTypes so a new format cannot be left out.
var packedRefFormats = func() (fs []struct {
	name string
	g    quant.Type
	k    kernels.Quant
}) {
	for _, g := range quant.PackedTypes {
		k, ok := kernels.QuantOf(g)
		if !ok {
			panic(g.String() + " is in quant.PackedTypes and has no device layout")
		}
		fs = append(fs, struct {
			name string
			g    quant.Type
			k    kernels.Quant
		}{g.String(), g, k})
	}
	return fs
}()

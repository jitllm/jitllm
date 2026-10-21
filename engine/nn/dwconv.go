//go:build amd64 || arm64

package nn

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// AddDWConv generates the depthwise convolution row of shape s (cpu.DWShape)
// once. A convolutional tower provisions every shape it runs at load, so
// DWConvRow, which may run on every worker at once, only reads the map.
func (f *JIT) AddDWConv(s cpu.DWShape) {
	if _, ok := f.dw[s]; ok {
		return
	}
	if f.dw == nil {
		f.dw = map[cpu.DWShape]*cpu.Code{}
	}
	f.dw[s] = mustEmit(s.Key())(f.em.DWConv(s))
}

// DWConvRow runs one output row of the depthwise convolution s: out is
// WOut*Chans floats, in the padded input from the row's window's first sample
// (K rows of WP*Chans floats), w the tap-major filter.
func (f *JIT) DWConvRow(out, in, w []float32, s cpu.DWShape) {
	code := f.dw[s]
	if code == nil {
		panic(fmt.Sprintf("jit: DWConvRow at %+v, which AddDWConv never provisioned", s))
	}
	need := ((s.K-1)*s.WP + (s.WOut-1)*s.Stride + s.K) * s.Chans
	if len(out) < s.WOut*s.Chans || len(in) < need || len(w) < s.K*s.K*s.Chans {
		panic(fmt.Sprintf("jit: DWConvRow at %+v: out %d, in %d (want %d), w %d", s, len(out), len(in), need, len(w)))
	}
	args := cpu.Args{Out: &out[0], AScale: &w[0], Q32: &in[0]}
	code.Call(&args)
}

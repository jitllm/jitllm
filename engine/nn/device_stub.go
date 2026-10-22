//go:build !(amd64 || arm64)

package nn

import "github.com/jitllm/jitllm/format/quant"

// Device on a platform with no generated code at all. The interface still
// exists so callers compile everywhere; nothing ever calls it.
type Device interface {
	MatVec(out []float64, t quant.Type, w []byte, x []float64, nrows, k int) bool
}

func (f *JIT) SetDevice(Device) {}
func (f *JIT) Device() Device    { return nil }

package probe

import (
	"github.com/jitllm/jitllm/composer/x86"
)

// chaseKernel is the hand-built chase the composed one replaced, the
// control arm of TestComposedHostChaseMatchesTheBuilt: it follows [1]
// pointers from the address in [0], each read waiting for the one before.
func (e emitter) chaseKernel() ([]byte, error) {
	a := e.buf()
	a.MOVLoad(x86.RSI, x86.At(x86.RDI, 0))
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 8))
	top := a.Label()
	a.Bind(top)
	a.MOVLoad(x86.RSI, x86.At(x86.RSI, 0))
	a.DEC(x86.RCX)
	a.JNZ(top)
	a.RET()
	return a.Bytes()
}

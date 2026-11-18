package probe

import (
	"github.com/jitllm/jitllm/composer/a64"
)

// chaseKernel is the hand-built chase the composed one replaced, the
// control arm of TestComposedHostChaseMatchesTheBuilt: it follows [1]
// pointers from the address in [0], each read waiting for the one before.
func (e emitter) chaseKernel() ([]byte, error) {
	var a a64.A64
	a.LDRx(a64.X1, a64.X0, 0)
	a.LDRx(a64.X2, a64.X0, 8)
	top := a.Label()
	a.Bind(top)
	a.LDRx(a64.X1, a64.X1, 0)
	a.SUBimm(a64.X2, a64.X2, 1)
	a.CBNZ(a64.X2, top)
	a.RET()
	return a.Bytes()
}

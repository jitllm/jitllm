package compose

import (
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// A unary activation with no formula (a gated kind asked for alone) fails to
// emit, by an error, never emitted as another function; one with a formula
// emits.
func TestEmitRefusesAnActivationWithNoFormula(t *testing.T) {
	for _, k := range []kernels.ActKind{kernels.ActSwiGLUClamp, kernels.ActSwiGLUOAI, kernels.ActSitu, kernels.ActXIELU} {
		g := &Row{Width: 32, Rows: 1}
		g.Store(g.Buf(BufOut), g.Act(k, g.Load(g.Buf(BufRow))))
		if _, err := g.Emit(Schedule{Group: 32}); err == nil {
			t.Errorf("%v: a unary %v emitted", k, k)
		}
	}
	for _, k := range []kernels.ActKind{kernels.ActSigmoid, kernels.ActSiLU, kernels.ActGELU, kernels.ActIdentity,
		kernels.ActGELUErf, kernels.ActSoftplus, kernels.ActSqrtSigmoid, kernels.ActTanh} {
		g := &Row{Width: 32, Rows: 1}
		g.Store(g.Buf(BufOut), g.Act(k, g.Load(g.Buf(BufRow))))
		if _, err := g.Emit(Schedule{Group: 32}); err != nil {
			t.Errorf("%v: refused: %v", k, err)
		}
	}
}

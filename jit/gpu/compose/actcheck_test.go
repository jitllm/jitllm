package compose

import (
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// A unary activation with no element form is refused when the group is
// checked, by an error, never emitted as another function.
func TestCheckRefusesAnActivationWithNoElementForm(t *testing.T) {
	for _, k := range []kernels.ActKind{kernels.ActSwiGLUClamp, kernels.ActSwiGLUOAI, kernels.ActSitu, kernels.ActXIELU} {
		g := &Row{Width: 32, Rows: 1}
		g.Store(g.Buf(BufOut), g.Act(k, g.Load(g.Buf(BufRow))))
		if err := g.Check(); err == nil {
			t.Errorf("%v: a unary %v was admitted", k, k)
		}
		if _, err := g.Emit(Schedule{Group: 32}); err == nil {
			t.Errorf("%v: emitted", k)
		}
	}
	for _, k := range []kernels.ActKind{kernels.ActSigmoid, kernels.ActSiLU, kernels.ActGELU, kernels.ActIdentity} {
		g := &Row{Width: 32, Rows: 1}
		g.Store(g.Buf(BufOut), g.Act(k, g.Load(g.Buf(BufRow))))
		if err := g.Check(); err != nil {
			t.Errorf("%v: refused: %v", k, err)
		}
	}
}

package jit_test

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestGuardedPhaseIsTheWarpsOwn holds a norm's int8 phase to its guard.
// Where its units come in whole warps, the window's combine -- lane shuffles
// -- is inside the guard's loop of a run-time count, so the warps past the
// last unit skip the phase. Where they do not, no shuffle is inside one, as a
// lane that skipped would be read; and no barrier is, as a warp that skipped
// would never meet it. (A window whose combine is an exchange through
// workgroup slots re-forms clamped; the windows a matvec takes today, 32 and
// a k-quant's 256, combine by shuffles.)
func TestGuardedPhaseIsTheWarpsOwn(t *testing.T) {
	for _, c := range []struct {
		width, window int
		inside        bool
	}{
		{2048, 32, true},
		{1120, 32, false},
	} {
		h := graph.New()
		h.G.Block = -1
		h.Out(h.MatVec(h.Norm(h.In("x", c.width), jlm.RoleOutputNorm, 1e-5), jlm.RoleOutput, 256))
		_, codes, err := jit.Default().Form(jit.Request{Graph: h.G,
			Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jlm.TypeQ8},
			Step:    jit.Step{Rows: 1, ActWindow: c.window},
			Device:  jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80}})
		if err != nil {
			t.Fatalf("%d wide, window %d: %v", c.width, c.window, err)
		}
		phases := 0
		for _, code := range codes {
			if !slices.Contains(code.Passes, "coarsen") {
				continue
			}
			phases++
			shuffles, barriers := guarded(code.GPU)
			if c.inside && shuffles == 0 {
				t.Errorf("%d wide, window %d: %s computes its phase outside its guard", c.width, c.window, code.Name)
			}
			if !c.inside && shuffles > 0 {
				t.Errorf("%d wide, window %d: %s shuffles %d times inside a guard its warps do not take whole",
					c.width, c.window, code.Name, shuffles)
			}
			if barriers > 0 {
				t.Errorf("%d wide, window %d: %s meets a barrier %d times inside a guard", c.width, c.window, code.Name, barriers)
			}
		}
		if phases == 0 {
			t.Fatalf("%d wide, window %d: no coarsened phase formed", c.width, c.window)
		}
	}
}

// guarded counts k's shuffles and barriers inside a loop of a run-time count.
func guarded(k *ir.Kernel) (shuffles, barriers int) {
	var open []bool
	for _, o := range k.Ops {
		switch {
		case o.Kind == ir.OpLoop:
			open = append(open, o.Imm < 0)
		case o.Kind == ir.OpEndLoop:
			open = open[:len(open)-1]
		case !slices.Contains(open, true):
		case o.Kind == ir.OpShuffleXor:
			shuffles++
		case o.Kind == ir.OpBarrier:
			barriers++
		}
	}
	return shuffles, barriers
}

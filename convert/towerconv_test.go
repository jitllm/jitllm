package convert

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestEveryProjectorStatesItsConventions: every projector the converter
// takes leaves a tower whose conventions are stated, and a tower that turns
// its heads states a rotary over every pair of a head.
func TestEveryProjectorStatesItsConventions(t *testing.T) {
	for name, p := range projectorOf {
		for _, rope := range []bool{false, true} {
			v := &jlm.Vision{Projector: p, NEmbd: 1280, NHead: 16}
			if rope {
				v.Flags |= jlm.FlagVisionRope
			}
			c, err := jlm.ConventionsOf(v)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if !c.Stated() || c.ProjAct == 0 || c.Class == 0 || (c.PosChain == 0) != (c.PosFilter == 0) {
				t.Errorf("%s: conventions %+v", name, c)
			}
			if rope != (c.Rope != nil) {
				t.Fatalf("%s: rotary %v for a tower that turns %v", name, c.Rope, rope)
			}
			if c.Rope != nil {
				pairs := 0
				for _, u := range c.Rope.Runs {
					pairs += int(u.Pairs)
				}
				if 2*pairs != 1280/16 || c.Rope.FreqDim == 0 || c.Rope.Base == 0 {
					t.Errorf("%s: a rotary of %d pairs over heads of %d: %+v", name, pairs, 1280/16, c.Rope)
				}
			}
		}
	}
}

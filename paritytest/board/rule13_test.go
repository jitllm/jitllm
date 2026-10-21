package board

import (
	"testing"

	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestNoGPUKernelReadsWhatItWrites holds every GPU kernel the composer forms
// of the board's fixture with no barrier in it -- one phase -- at the
// storage the estimate binds it to, to RULE 13: such a kernel that reads a
// buffer it writes races its own threads, and the IR cannot see two
// parameters bound to one buffer. A kernel with a barrier orders its phases
// itself.
func TestNoGPUKernelReadsWhatItWrites(t *testing.T) {
	f, err := jlm.Open(testmodels.Path(*boardModel + ".jlm"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	d := cost.Device{Device: sm70}
	ccs, err := cost.EstimateCodes(jit.Default(), f, jit.Step{Rows: 1, Page: 64, Keys: 9}, func(jit.Part) cost.Device { return d }, true)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, cc := range ccs {
		if cc.Code.GPU == nil || phased(cc.Code.GPU) {
			continue
		}
		n++
		read, wrote := map[jit.Operand]bool{}, map[jit.Operand]bool{}
		for j, b := range cc.Code.Binds {
			if b.Writes {
				wrote[cc.Binds[j].Storage] = true
			} else {
				read[cc.Binds[j].Storage] = true
			}
		}
		for at := range read {
			if wrote[at] {
				t.Errorf("%s reads and writes %q", cc.Code.Name, at.Name)
			}
		}
	}
	if n == 0 {
		t.Fatal("the composer formed no GPU kernel of one phase")
	}
}

func phased(k *ir.Kernel) bool {
	for _, o := range k.Ops {
		if o.Kind == ir.OpBarrier || o.Kind == ir.OpGridBarrier {
			return true
		}
	}
	return false
}

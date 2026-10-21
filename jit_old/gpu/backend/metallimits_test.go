package backend_test

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// TestMetalRefusesAThreadgroupThePipelineCannotRun: Metal accepts a dispatch
// wider than the compiled pipeline's maxTotalThreadsPerThreadgroup and then
// does not run the kernel, leaving its output unwritten. A 2048-thread group
// is past every Apple GPU's limit, so Compile must refuse it by name; a
// 128-thread one must compile. Violation (no limit check): the wide kernel
// compiles.
func TestMetalRefusesAThreadgroupThePipelineCannotRun(t *testing.T) {
	gpuLock(t)
	ran := 0
	for _, d := range backend.Open() {
		defer d.Close()
		if d.API() != "msl" {
			continue
		}
		mk := func(width int) *ir.Kernel {
			b := ir.New("k", [3]int{width, 1, 1})
			pIn := b.Param("pIn", ir.F32)
			pOut := b.Param("pOut", ir.F32)
			i := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
			b.Store(pOut, i, b.Load(ir.F32, pIn, i, 0), 0)
			return b.Done()
		}
		k, err := d.Compile(mk(128))
		if err != nil {
			t.Fatalf("%s: a 128-thread kernel was refused: %v", d.Name(), err)
		}
		k.Close()
		if k, err := d.Compile(mk(2048)); err == nil {
			k.Close()
			t.Fatalf("%s: a 2048-thread threadgroup compiled; it would not run", d.Name())
		} else if !strings.Contains(err.Error(), "threadgroup") {
			t.Fatalf("%s: refused for the wrong reason: %v", d.Name(), err)
		} else {
			t.Logf("%s: %v", d.Name(), err)
		}
		ran++
	}
	if ran == 0 {
		t.Skip("no Metal device on this host")
	}
}

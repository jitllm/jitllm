package board

import (
	"bytes"
	"testing"

	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/amdgpu"
	"github.com/jitllm/jitllm/jit/msl"
	"github.com/jitllm/jitllm/jit/ptx"
	"github.com/jitllm/jitllm/jit/spirv"
	oldamdgpu "github.com/jitllm/jitllm/jit_old/gpu/amdgpu"
	oldmsl "github.com/jitllm/jitllm/jit_old/gpu/msl"
	oldptx "github.com/jitllm/jitllm/jit_old/gpu/ptx"
	oldspirv "github.com/jitllm/jitllm/jit_old/gpu/spirv"
	"github.com/jitllm/jitllm/paritytest"
)

// TestLowerersMatchTheirOrigin holds the lowerers jit carries, copied from
// jit_old to read the composer's IR, to the originals: every GPU kernel the
// composer forms of the board's fixture lowers to the same PTX, MSL, SPIR-V
// and AMDGPU code through either.
func TestLowerersMatchTheirOrigin(t *testing.T) {
	plan, err := composed(t, *boardModel, func(jit.Part) jit.Device { return sm70 })
	if plan == nil {
		t.Fatal(err)
	}
	gfx, err := amdgpu.TargetFor("gfx1100")
	if err != nil {
		t.Fatal(err)
	}
	oldgfx, err := oldamdgpu.TargetFor("gfx1100")
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, pp := range plan.Parts {
		for _, c := range pp.Codes {
			k := c.GPU
			if k == nil {
				continue
			}
			o := paritytest.Old(k)
			n++
			for _, sm := range []string{"sm_70", "sm_86"} {
				same(t, k.Name+" ptx "+sm, func() (string, error) { return ptx.Lower(k, sm) }, func() (string, error) { return oldptx.Lower(o, sm) })
			}
			same(t, k.Name+" msl", func() (string, error) { return msl.Emit(k) }, func() (string, error) { return oldmsl.Emit(o) })
			same(t, k.Name+" amdgpu", func() (string, error) { return amdgpu.Lower(k, gfx) }, func() (string, error) { return oldamdgpu.Lower(o, oldgfx) })
			a, aerr := spirv.Emit(k)
			b, berr := oldspirv.Emit(o)
			if (aerr == nil) != (berr == nil) || !bytes.Equal(a, b) {
				t.Errorf("%s spirv: copy %d words (%v), origin %d words (%v)", k.Name, len(a)/4, aerr, len(b)/4, berr)
			}
		}
	}
	if n == 0 {
		t.Fatal("the composer formed no GPU kernel")
	}
	t.Logf("%d kernels lower alike through the copies and their origins", n)
}

func same(t *testing.T, what string, copy, origin func() (string, error)) {
	t.Helper()
	a, aerr := copy()
	b, berr := origin()
	if (aerr == nil) != (berr == nil) || a != b {
		t.Errorf("%s: copy %d bytes (%v), origin %d bytes (%v)", what, len(a), aerr, len(b), berr)
	}
}

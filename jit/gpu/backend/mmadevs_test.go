package backend_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// mmaDevices is the devices that can compile an ir.OpMMA, asked once each, so a
// backend without MMA does not produce a skip for every cell of a sweep.
// lowertest's TestMMACoversExactlyWhatItClaims asserts which backends lower MMA
// and why the others decline.
//
// The probe is one tile with the results stored, so nothing is dead-coded away,
// and it compiles the same shape the sweeps use. pOut takes the accumulator's
// type: an i32 parameter stored as f32 is refused by ir.Validate, and that
// refusal once made every MMA sweep skip on every CUDA card.
//
// A PTX device whose target has the instruction and still declines FAILS: a
// decline there is a broken probe or a broken lowering, never "not swept".
func mmaDevices(t *testing.T, devs []backend.Device, sh ir.MMAShape) []backend.Device {
	t.Helper()
	b := ir.New("mmacap", [3]int{32, 1, 1})
	pA, pB := b.Param("pA", ir.U32), b.Param("pB", ir.U32)
	pOut := b.Param("pOut", sh.Elem())
	na, nb, nc := sh.Frags()
	a := make([]ir.Value, na)
	for i := range a {
		a[i] = b.Load(ir.U32, pA, b.Const(ir.U32, int64(i)), 0)
	}
	bb := make([]ir.Value, nb)
	for i := range bb {
		bb[i] = b.Load(ir.U32, pB, b.Const(ir.U32, int64(i)), 0)
	}
	c := make([]ir.Value, nc)
	for i := range c {
		c[i] = b.Const(sh.Elem(), 0)
	}
	for i, v := range b.MMA(sh, a, bb, c) {
		b.Store(pOut, b.Const(ir.U32, int64(i)), v, 0)
	}
	probe := b.Done()

	out := make([]backend.Device, 0, len(devs))
	for _, d := range devs {
		k, err := d.Compile(probe)
		if err != nil {
			if need := mmaMinSM(sh); d.API() == "ptx" && ptxSM(d.Name()) >= need {
				t.Fatalf("%s: sm_%d and later have the %dx%dx%d MMA, and the probe was refused: %v",
					d.Name(), need, sh.M, sh.N, sh.K, err)
			}
			t.Logf("%s: no MMA lowering, not swept: %v", d.API(), err)
			continue
		}
		k.Close()
		out = append(out, d)
	}
	return out
}

// mmaMinSM is the first PTX target with sh's mma.sync (PTX ISA, "Matrix
// shape"): m8n8k4 f16 at sm_70, m16n8k8 f16 at sm_75, the m16n8k16 and
// m16n8k32 tiles at sm_80.
func mmaMinSM(sh ir.MMAShape) int {
	switch {
	case sh == ir.MMAVolta():
		return 70
	case sh.Kind == ir.MMAF16 && sh.K == 8:
		return 75
	}
	return 80
}

// ptxSM reads the target out of a CUDA device's name, "#0 <card> (sm_86)", and
// is 0 when there is none.
func ptxSM(name string) int {
	i := strings.LastIndex(name, "(sm_")
	if i < 0 {
		return 0
	}
	var sm int
	if _, err := fmt.Sscanf(name[i:], "(sm_%d)", &sm); err != nil {
		return 0
	}
	return sm
}

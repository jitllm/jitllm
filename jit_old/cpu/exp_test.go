//go:build amd64

package cpu

import (
	"math"
	"testing"
	"unsafe"
)

// TestEmitExpPS checks the vectorised exp against math.Exp before anything is
// built on it.
//
// It is tested alone because a softmax built on a broken exp still sums to 1,
// hiding the error.
func TestEmitExpPS(t *testing.T) {
	// A kernel that exps 8 floats: load, exp, store. Out = y, A = x, Scr = the
	// constant block, exactly as the real callers wire it.
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RSI, At(RDI, 16)) // A -> x
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> consts
	loadExpConsts(&a)
	a.VMOVDQULoad(Y0, At(RSI, 0))
	emitExpPS(&a, Y0, Y1, Y2)
	a.VMOVDQUStore(At(RCX, 0), Y0)
	a.VZEROUPPER()
	a.RET()

	code := avx2Primitive(t, a.Bytes(), "TestEmitExpSSE")
	if code == nil {
		return
	}
	defer code.Close()
	consts := expConsts()

	// The range a softmax actually feeds it: x - max, so at most 0, down to
	// where f32 underflows. Plus 0 itself, where exp must be exactly 1.
	for _, base := range [][]float32{
		{0, -0.5, -1, -2, -5, -10, -20, -40},
		{-0.001, -0.125, -0.347, -0.694, -1.386, -50, -80, -87},
		{-88, -100, -200, -1000, -3, -7, -15, -30},
	} {
		x := append([]float32(nil), base...)
		got := make([]float32, 8)
		args := Args{
			Out: &got[0],
			A:   (*int8)(unsafe.Pointer(&x[0])),
			Scr: (*byte)(unsafe.Pointer(&consts[0])),
		}
		code.Call(&args)
		for i, v := range x {
			want := math.Exp(float64(v))
			if want < 1e-38 {
				// Under f32's normal range the clamp makes it zero, which is
				// the correct answer for a softmax term and is what the Go
				// version does too.
				if got[i] > 1e-30 {
					t.Errorf("exp(%g) = %g, want ~0", v, got[i])
				}
				continue
			}
			if rel := math.Abs(float64(got[i])-want) / want; rel > 3e-6 {
				t.Errorf("exp(%g) = %g, want %g (rel %.2e)", v, got[i], want, rel)
			}
		}
	}
}

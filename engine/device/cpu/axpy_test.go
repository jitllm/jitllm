package cpu

import (
	"math"
	"testing"
	"unsafe"
)

// TestEmitAxpyMatchesReference gates dst += alpha*src against float64.
//
// alpha != 1 is tested separately from alpha == 1, because the add path
// passes a shared constant and the MoE path passes a routing probability --
// a kernel that ignored Scr entirely would pass the first and fail the second,
// and a kernel that read the wrong offset would do the reverse.
func TestEmitAxpyMatchesReference(t *testing.T) {
	code := onHost(t)(hostTable().Axpy())
	defer code.Close()

	for _, alpha := range []float32{1, 0.37, -2.5} {
		for _, n := range []int{ElemLanes, 64, 2048} {
			dst := make([]float32, n)
			src := make([]float32, n)
			want := make([]float64, n)
			for i := range dst {
				dst[i] = float32(math.Sin(float64(i) * 0.7))
				src[i] = float32(math.Cos(float64(i)*0.3) * 3)
				want[i] = float64(dst[i]) + float64(alpha)*float64(src[i])
			}
			al := alpha
			args := Args{
				Out:    &dst[0],
				AScale: &src[0],
				Scr:    (*byte)(unsafe.Pointer(&al)),
				K:      int64(n / ElemLanes),
			}
			code.Call(&args)

			var sse, sy2 float64
			for i := range want {
				d := float64(dst[i]) - want[i]
				sse, sy2 = sse+d*d, sy2+want[i]*want[i]
			}
			if nmse := sse / sy2; nmse > 1e-12 {
				t.Errorf("alpha=%v n=%d: NMSE %.3e", alpha, n, nmse)
			}
		}
	}
}

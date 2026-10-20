package oldkernels_test

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestHeadNormAtEveryRealHeadDim runs the per-head QK norm against nn.RMSNorm32
// at the head widths models actually use.
//
// 256 is the case to cover: the 32-lane kernel gives each lane headDim/32
// elements, 4 at headDim 128 and 8 at 256, and qwen3next is the first model
// at 256.
func TestHeadNormAtEveryRealHeadDim(t *testing.T) {
	d := dev(t)
	const eps = 1e-6
	for _, hd := range []int{64, 96, 128, 256} {
		for _, nh := range []int{1, 4} {
			t.Run(itoa(hd)+"x"+itoa(nh), func(t *testing.T) {
				k, err := kernels.HeadNorm(nh, hd, eps, 32)
				if err != nil {
					t.Skipf("no 32-lane head norm for headDim %d: %v", hd, err)
				}
				src := ramp(nh*hd, func(i int) float32 {
					return float32(math.Sin(float64(i)*0.37))*3 + 0.5
				})
				w := ramp(hd, func(i int) float32 {
					return float32(math.Cos(float64(i)*0.11)) + 1.5
				})
				// The host: one RMSNorm per head, the weight shared by all.
				want := make([]float32, nh*hd)
				for h := 0; h < nh; h++ {
					oracle.RMSNorm32(want[h*hd:(h+1)*hd], src[h*hd:(h+1)*hd], w, eps)
				}

				kern, err := d.Compile(k)
				if err != nil {
					t.Fatalf("compile: %v", err)
				}
				defer kern.Close()
				bs, _ := d.Alloc(nh * hd * 4)
				bw, _ := d.Alloc(hd * 4)
				bo, _ := d.Alloc(nh * hd * 4)
				defer func() { bs.Free(); bw.Free(); bo.Free() }()
				poison := make([]float32, nh*hd)
				for i := range poison {
					poison[i] = float32(math.NaN())
				}
				bo.Write(f32b(poison))
				bs.Write(f32b(src))
				bw.Write(f32b(w))
				if err := kern.Launch(nh, 32, bs, bw, bo); err != nil {
					t.Fatalf("launch: %v", err)
				}
				raw := make([]byte, nh*hd*4)
				bo.Read(raw)
				got := make([]float32, nh*hd)
				b2f(raw, got)
				for i := range got {
					if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
						t.Fatalf("headDim=%d head %d dim %d is %g -- never written; "+
							"the per-lane loop does not cover the head",
							hd, i/hd, i%hd, got[i])
					}
					if math.Abs(float64(got[i])-float64(want[i])) > 2e-5 {
						t.Fatalf("headDim=%d head %d dim %d = %g, host says %g",
							hd, i/hd, i%hd, got[i], want[i])
					}
				}
			})
		}
	}
}

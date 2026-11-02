package cpu

import (
	"math"
	"testing"
	"unsafe"
)

// refRMSNorm is the float64 oracle, written here rather than imported so the
// test does not depend on the package it is checking.
func refRMSNorm(y, x, w []float32, eps float64) {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	scale := 1 / math.Sqrt(sum/float64(len(x))+eps)
	for i, v := range x {
		y[i] = float32(float64(v) * scale * float64(w[i]))
	}
}

// TestRMSNormKernelMatchesReference gates the emitted norm against float64.
//
// The bar is NMSE: the kernel sums squares in four f32 chains, the reference
// in one f64 chain. 1e-10 is far from what a wrong stride or a missed tail
// produces (the violations land at 1e-2 and above).
func TestRMSNormKernelMatchesReference(t *testing.T) {
	// No feature skip: this kernel needs only baseline AVX2/FMA.
	//
	// The second half of the list is ragged widths: 1 through 7 have no vector
	// at all, 33 and 35 have a vector tail and a scalar one, and the output
	// carries a guard so a write past n fails by name.
	for _, n := range []int{8, 32, 64, 128, 256, 576, 1152, 2048, 2304, 4096, 8192,
		1, 2, 3, 5, 7, 9, 13, 31, 33, 35, 63, 100, 2047} {
		code := onHost(t)(hostTable().RMSNorm(n))
		x := make([]float32, n)
		w := make([]float32, n)
		for i := range x {
			// Heavy-tailed on purpose: a uniform vector hides a dropped tail,
			// because every element contributes the same amount to the sum.
			x[i] = float32(math.Sin(float64(i)*0.7+0.3)) * float32(1+i%13)
			w[i] = 1 + float32(i%7)*0.03
		}
		want := make([]float32, n)
		refRMSNorm(want, x, w, 1e-5)

		got := sentinelRow(n)
		consts := []float32{1 / float32(n), 1e-5, 1}
		args := Args{
			Out:    &got[0],
			A:      (*int8)(unsafe.Pointer(&x[0])),
			AScale: &w[0],
			Scr:    (*byte)(unsafe.Pointer(&consts[0])),
			K:      int64(n),
		}
		code.Call(&args)
		code.Close()
		for i := n; i < len(got); i++ {
			if math.Float32bits(got[i]) != softmaxSentinel {
				t.Fatalf("n=%d: wrote element %d past the end", n, i)
			}
		}

		var sse, sy2 float64
		for i := range want {
			d := float64(got[i]) - float64(want[i])
			sse, sy2 = sse+d*d, sy2+float64(want[i])*float64(want[i])
		}
		if sy2 == 0 {
			t.Fatalf("n=%d: the reference is all zero, so this proves nothing", n)
		}
		if nmse := sse / sy2; nmse > 1e-10 {
			t.Errorf("n=%d: NMSE %.3e against the float64 reference", n, nmse)
		}
	}
}

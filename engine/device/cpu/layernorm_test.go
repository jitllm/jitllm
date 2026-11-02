package cpu

import (
	"math"
	"testing"
	"unsafe"
)

// TestEmitLayerNormMatchesReference gates the generated LayerNorm against a
// float64 reference at the widths a vision tower really uses.
//
// The input carries a large mean, as a ViT residual does, which is where the
// one-pass E[x^2]-E[x]^2 identity cancels away in f32.
func TestEmitLayerNormMatchesReference(t *testing.T) {
	// The second half of the list is ragged on purpose -- see the RMSNorm gate.
	// The variance tail is the one that can go wrong where the mean's cannot: a
	// zeroed lane contributes (0-mean)^2 unless it is cleared.
	for _, n := range []int{8, 24, 64, 768, 1024, 3072, 1, 2, 3, 5, 7, 9, 13, 33, 35, 100} {
		for _, bias := range []bool{false, true} {
			code := onHost(t)(hostTable().LayerNorm(n, bias))
			x := make([]float32, n)
			w := make([]float32, n)
			b := make([]float32, n)
			for i := range x {
				x[i] = float32(40 + math.Sin(float64(i)*0.37)*3)
				w[i] = float32(0.5 + float64(i%11)*0.07)
				b[i] = float32(math.Cos(float64(i) * 0.11))
			}
			eps := 1e-6
			var mean float64
			for _, v := range x {
				mean += float64(v)
			}
			mean /= float64(n)
			var varr float64
			for _, v := range x {
				d := float64(v) - mean
				varr += d * d
			}
			varr /= float64(n)
			inv := 1 / math.Sqrt(varr+eps)
			want := make([]float64, n)
			for i, v := range x {
				want[i] = (float64(v) - mean) * inv * float64(w[i])
				if bias {
					want[i] += float64(b[i])
				}
			}

			y := sentinelRow(n)
			kc := []float32{1 / float32(n), float32(eps), 1}
			args := Args{
				Out:    &y[0],
				A:      (*int8)(unsafe.Pointer(&x[0])),
				AScale: &w[0],
				Scr:    (*byte)(unsafe.Pointer(&kc[0])),
				K:      int64(n),
			}
			if bias {
				args.W = (*byte)(unsafe.Pointer(&b[0]))
			}
			code.Call(&args)
			code.Close()
			for i := n; i < len(y); i++ {
				if math.Float32bits(y[i]) != softmaxSentinel {
					t.Fatalf("n=%d bias=%v: wrote element %d past the end", n, bias, i)
				}
			}

			var sse, sy2 float64
			for i := range want {
				d := float64(y[i]) - want[i]
				sse, sy2 = sse+d*d, sy2+want[i]*want[i]
			}
			if nmse := sse / sy2; nmse > 1e-10 {
				t.Errorf("n=%d bias=%v: NMSE %.3e against the float64 reference", n, bias, nmse)
			}
		}
	}
}

// TestEmitActMatchesActMul checks the ungated activation against the gated one
// with an `up` vector of ones.
//
// EmitAct's likely failure is reading `up` anyway, which ones would hide, so
// the second arm uses a real `up` and asserts the two disagree.
func TestEmitActMatchesActMul(t *testing.T) {
	consts := actConsts()
	n := 1024
	for _, k := range []ActKind{ActSiLU, ActGELU} {
		// ReLU2 has no gated twin to compare with; its values are held to
		// max(x,0)^2 by TestElementwiseSSEEveryWidth on both tiers.
		ref := onHost(t)(hostTable().ActMul(k))
		got := onHost(t)(hostTable().Act(k))
		mk := func() ([]float32, []float32) {
			d := make([]float32, n)
			u := make([]float32, n)
			for i := range d {
				d[i] = float32(math.Sin(float64(i)*0.31) * float64(1+i%17))
				u[i] = 1
			}
			return d, u
		}
		a, ones := mk()
		bb, _ := mk()
		args := Args{Out: &a[0], AScale: &ones[0],
			Scr: (*byte)(unsafe.Pointer(&consts[0])), K: int64(n / ElemLanes)}
		ref.Call(&args)
		args2 := Args{Out: &bb[0],
			Scr: (*byte)(unsafe.Pointer(&consts[0])), K: int64(n / ElemLanes)}
		got.Call(&args2)
		for i := range a {
			if a[i] != bb[i] {
				t.Fatalf("%s: ungated differs at %d: %v vs %v", k, i, a[i], bb[i])
			}
		}

		// The control: with a real `up`, the two must not agree.
		c, u := mk()
		for i := range u {
			u[i] = float32(0.5 + float64(i%9)*0.1)
		}
		args.Out, args.AScale = &c[0], &u[0]
		ref.Call(&args)
		same := true
		for i := range c {
			if c[i] != bb[i] {
				same = false
				break
			}
		}
		if same {
			t.Fatalf("%s: gated and ungated agree on a non-unit `up`; "+
				"this test cannot tell them apart and proves nothing", k)
		}
		ref.Close()
		got.Close()
	}
}

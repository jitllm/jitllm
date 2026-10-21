package cpu

import (
	"math"
	"testing"
	"unsafe"
)

func refSiLU(x float64) float64 { return x / (1 + math.Exp(-x)) }

func refGELU(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(0.79788456*(x+0.044715*x*x*x)))
}

// TestEmitActMulMatchesReference gates every gated activation against float64.
//
// SiLU and GELU are close over much of the range, so a control asserts they
// differ. swiglu-oai's clamps must be reached: the inputs span |gate| to 17
// and |up| to 13, and the test counts elements past each limit.
func TestEmitActMulMatchesReference(t *testing.T) {
	consts := actConsts()
	ref := func(k ActKind, g, u float64) float64 {
		switch k {
		case ActGELU:
			return refGELU(g) * u
		case ActGELUErf:
			return 0.5 * g * (1 + math.Erf(g/math.Sqrt2)) * u
		case ActIdentity:
			return g * u
		case ActSwiGLUOAI:
			x := math.Min(g, 7)
			return x / (1 + math.Exp(-1.702*x)) * (math.Max(math.Min(u, 7), -7) + 1)
		case ActSwiGLUClamp:
			return refSiLU(math.Min(g, 10)) * math.Max(math.Min(u, 10), -10)
		case ActSitu:
			return 4 * math.Tanh(g/4) / (1 + math.Exp(-g)) * 25 * math.Tanh(u/25)
		case ActReLU2:
			r := math.Max(g, 0)
			return r * r * u
		}
		return refSiLU(g) * u
	}
	for _, k := range Gated {
		code := onHost(t)(hostTable().ActMul(k))
		n := 1024
		dst := make([]float32, n)
		up := make([]float32, n)
		want := make([]float64, n)
		gHigh, uOut := 0, 0
		for i := range dst {
			// The range an FFN gate really spans, both signs, including the
			// region where GELU and SiLU part company.
			v := math.Sin(float64(i)*0.31) * float64(1+i%17)
			u := math.Sin(float64(i)*0.13) * float64(1+i%13)
			dst[i], up[i] = float32(v), float32(u)
			want[i] = ref(k, float64(dst[i]), float64(up[i]))
			if v > 7 {
				gHigh++
			}
			if math.Abs(u) > 7 {
				uOut++
			}
		}
		if gHigh == 0 || uOut == 0 {
			t.Fatalf("%s: %d gates above 7 and %d ups outside +-7 -- the clamps are not exercised", k, gHigh, uOut)
		}
		args := Args{
			Out:    &dst[0],
			AScale: &up[0],
			Scr:    (*byte)(unsafe.Pointer(&consts[0])),
			K:      int64(n / ElemLanes),
		}
		code.Call(&args)
		code.Close()

		var sse, sy2 float64
		for i := range want {
			d := float64(dst[i]) - want[i]
			sse, sy2 = sse+d*d, sy2+want[i]*want[i]
		}
		if nmse := sse / sy2; nmse > 1e-9 {
			t.Errorf("%s: NMSE %.3e against the float64 reference", k, nmse)
		}
	}

	// swiglu-oai is not SiLU times up: the control for its three differences.
	var d2 float64
	for i := -40; i <= 40; i++ {
		g, u := float64(i)*0.25, float64(-i)*0.3
		d2 += math.Abs(ref(ActSwiGLUOAI, g, u) - ref(ActSiLU, g, u))
	}
	if d2 < 1e-3 {
		t.Fatalf("swiglu-oai and SiLU*up agree to %g; this test cannot tell them apart", d2)
	}

	// The control: the two activations must NOT be the same function.
	var diff float64
	for i := -40; i <= 40; i++ {
		x := float64(i) * 0.25
		diff += math.Abs(refSiLU(x) - refGELU(x))
	}
	if diff < 1e-3 {
		t.Fatalf("SiLU and GELU agree to %g over the sampled range; "+
			"this test cannot tell them apart and proves nothing", diff)
	}
}

// TestEmitSigmoidMulMatchesReference gates the gate, with the control that it
// is not SiLU: sigma(g)*v and SiLU(g)*v differ by a factor of g, so they agree
// where |g| is near 1 and a SiLU kernel would otherwise pass.
func TestEmitSigmoidMulMatchesReference(t *testing.T) {
	consts := actConsts()
	code := onHost(t)(hostTable().SigmoidMul())
	defer code.Close()

	n := 1024
	gate := make([]float32, n)
	val := make([]float32, n)
	want := make([]float64, n)
	for i := range gate {
		// Both signs and well past where sigma saturates, because a gate is a
		// projection output and is not bounded.
		g := math.Sin(float64(i)*0.37) * float64(1+i%23)
		gate[i] = float32(g)
		val[i] = float32(0.5 + float64(i%11)*0.2)
		want[i] = refSigmoid(g) * float64(val[i])
	}
	args := Args{
		Out:    &gate[0], // the GATE is the in-place argument
		AScale: &val[0],  // what it gates
		Scr:    (*byte)(unsafe.Pointer(&consts[0])),
		K:      int64(n / ElemLanes),
	}
	code.Call(&args)

	var sse, sy2 float64
	for i := range want {
		d := float64(gate[i]) - want[i]
		sse, sy2 = sse+d*d, sy2+want[i]*want[i]
	}
	if nmse := sse / sy2; nmse > 1e-9 {
		t.Errorf("NMSE %.3e against the float64 reference", nmse)
	}

	// The control: sigma and SiLU must not be the same function. They differ
	// by a factor of x, so this also fails if the kernel silently became SiLU.
	var diff float64
	for i := -40; i <= 40; i++ {
		x := float64(i) * 0.25
		diff += math.Abs(refSigmoid(x) - refSiLU(x))
	}
	if diff < 1e-3 {
		t.Fatalf("sigmoid and SiLU agree to %g over the sampled range; "+
			"the control cannot tell the two kernels apart", diff)
	}
	// And sigma is bounded where SiLU is not -- a second, independent way to
	// tell them apart, on the saturating tail where a gate actually lives.
	if s := refSigmoid(30); s <= 0.99 || s > 1 {
		t.Fatalf("sigmoid(30) = %g, want just under 1", s)
	}
}

func refSigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

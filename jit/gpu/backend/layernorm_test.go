package backend_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestLayerNormAndAct gates the vision tower's two device kernels on every
// backend present, against a float64 reference. It is in backend/ because
// kernels/ only ever tests one device.
//
// The input carries a large mean on purpose: that is why LayerNorm is three
// kernels rather than two. A one-pass E[x^2]-E[x]^2 cancels away most of the
// f32 mantissa when the mean dwarfs the variance; centred data cannot tell the
// two forms apart.
func TestLayerNormAndAct(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			layerNormCase(t, d)
			actCase(t, d)
		})
	}
}

func layerNormCase(t *testing.T, d backend.Device) {
	const k, parts = 768, 64
	const eps = 1e-6
	x := make([]float32, k)
	w := make([]float32, k)
	bias := make([]float32, k)
	for i := range x {
		x[i] = float32(40 + math.Sin(float64(i)*0.37)*3)
		w[i] = float32(0.5 + float64(i%11)*0.07)
		bias[i] = float32(math.Cos(float64(i) * 0.11))
	}
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= k
	var varr float64
	for _, v := range x {
		dd := float64(v) - mean
		varr += dd * dd
	}
	varr /= k
	inv := 1 / math.Sqrt(varr+eps)

	g := newGPU(t, d)
	defer g.free()
	bX := g.up(f32bytes(x))
	bW := g.up(f32bytes(w))
	bB := g.up(f32bytes(bias))
	bS := g.up(make([]byte, parts*4))
	bV := g.up(make([]byte, parts*4))
	bO := g.up(make([]byte, k*4))

	comp := func(kk *ir.Kernel, err error) backend.Kernel {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		c, err := d.Compile(kk)
		if err != nil {
			t.Fatalf("compile %s: %v", kk.Name, err)
		}
		return c
	}
	kp := comp(mainkernels.LayerNormPartRows(k, parts, 1))
	defer kp.Close()
	kv := comp(mainkernels.LayerNormVarRows(k, parts, 1))
	defer kv.Close()

	raw := make([]byte, k*4)
	for _, withBias := range []bool{false, true} {
		ka := comp(mainkernels.LayerNormApplyRows(k, parts, eps, withBias, 1))
		if err := kp.Launch((parts+127)/128, 128, bX, bS); err != nil {
			t.Fatal(err)
		}
		if err := kv.Launch((parts+127)/128, 128, bX, bS, bV); err != nil {
			t.Fatal(err)
		}
		var err error
		if withBias {
			err = ka.Launch((k+127)/128, 128, bX, bW, bS, bV, bB, bO)
		} else {
			err = ka.Launch((k+127)/128, 128, bX, bW, bS, bV, bO)
		}
		if err != nil {
			t.Fatal(err)
		}
		ka.Close()
		if err := bO.Read(raw); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < k; i++ {
			want := (float64(x[i]) - mean) * inv * float64(w[i])
			if withBias {
				want += float64(bias[i])
			}
			got := float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])))
			if math.Abs(got-want) > 1e-4*math.Abs(want)+1e-5 {
				t.Fatalf("bias=%v layernorm[%d] = %v, want %v", withBias, i, got, want)
			}
		}
	}
}

// actCase gates the ungated activation -- the one a ViT's FFN needs, where
// ActMul cannot help because there is no `up` to multiply by.
//
// The control is that the two activations disagree: SiLU and GELU are close
// over much of the range, so a kernel that emitted SiLU for both would pass a
// loose bar on the GELU case and "GELU works" would be a claim about SiLU.
func actCase(t *testing.T, d backend.Device) {
	const n = 1024
	in := make([]float32, n)
	up := make([]float32, n)
	for i := range in {
		in[i] = float32(math.Sin(float64(i)*0.31) * float64(1+i%17))
		up[i] = float32(math.Cos(float64(i)*0.13) * float64(1+i%13))
	}
	g := newGPU(t, d)
	defer g.free()
	bIn := g.up(f32bytes(in))
	bUp := g.up(f32bytes(up))
	bOut := g.up(make([]byte, n*4))
	raw := make([]byte, n*4)
	ungated := func(k mainkernels.ActKind, v float64) float64 {
		switch k {
		case mainkernels.ActGELU:
			return 0.5 * v * (1 + math.Tanh(0.7978845608028654*(v+0.044715*v*v*v)))
		case mainkernels.ActQuickGELU:
			return v / (1 + math.Exp(-1.702*v))
		case mainkernels.ActReLU2:
			r := math.Max(v, 0)
			return r * r
		case mainkernels.ActReLU:
			return math.Max(v, 0)
		case mainkernels.ActSqrtSoftplus:
			return math.Sqrt(math.Log1p(math.Exp(v)))
		case mainkernels.ActGELUErf:
			return 0.5 * v * (1 + math.Erf(v/math.Sqrt2))
		case mainkernels.ActSqrtSigmoid:
			m := math.Sqrt(math.Max(math.Abs(v), 1e-6))
			if v < 0 {
				m = -m
			}
			return 1 / (1 + math.Exp(-m))
		}
		return v / (1 + math.Exp(-v))
	}
	gated := func(k mainkernels.ActKind, v, u float64) float64 {
		// Gemma 3n's identity is the product alone: there is no activation
		// for the ungated arm to stand in.
		if k == mainkernels.ActIdentity {
			return v * u
		}
		if k == mainkernels.ActSwiGLUOAI {
			x := math.Min(v, 7)
			return x / (1 + math.Exp(-1.702*x)) * (math.Max(math.Min(u, 7), -7) + 1)
		}
		if k == mainkernels.ActSwiGLUClamp {
			x := math.Min(v, 10)
			return x / (1 + math.Exp(-x)) * math.Max(math.Min(u, 10), -10)
		}
		if k == mainkernels.ActSitu {
			b, lb := float64(mainkernels.SituBeta), float64(mainkernels.SituLinearBeta)
			return b * math.Tanh(v/b) / (1 + math.Exp(-v)) * lb * math.Tanh(u/lb)
		}
		return ungated(k, v) * u
	}
	run := func(name string, kk interface{}, err error, bufs ...backend.Buf) []float64 {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		kern, err := d.Compile(kk.(*ir.Kernel))
		if err != nil {
			t.Fatalf("compile %s: %v", name, err)
		}
		defer kern.Close()
		if err := kern.Launch((n+127)/128, 128, bufs...); err != nil {
			t.Fatal(err)
		}
		if err := bOut.Read(raw); err != nil {
			t.Fatal(err)
		}
		out := make([]float64, n)
		for i := range out {
			out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])))
		}
		return out
	}
	check := func(name string, got []float64, want func(i int) float64) {
		t.Helper()
		for i, g := range got {
			w := want(i)
			if math.Abs(g-w) > 2e-4*math.Abs(w)+1e-5 {
				t.Fatalf("%s[%d] = %v, want %v", name, i, g, w)
			}
		}
	}
	var outs [][]float64
	for _, k := range mainkernels.Ungated {
		kk, err := mainkernels.Act(n, k)
		o := run("Act "+k.String(), kk, err, bIn, bOut)
		check("Act "+k.String(), o, func(i int) float64 { return ungated(k, float64(in[i])) })
		outs = append(outs, o)
	}
	for _, k := range mainkernels.Gated {
		kk, err := mainkernels.ActMul(n, k)
		o := run("ActMul "+k.String(), kk, err, bIn, bUp, bOut)
		check("ActMul "+k.String(), o, func(i int) float64 { return gated(k, float64(in[i]), float64(up[i])) })
	}
	// swiglu-oai's clamps (7) and swiglu-clamp's (10) must be reached or the
	// cases are their unclamped forms: the inputs span |x| to 17 and |up| to 13.
	hi, out := 0, 0
	for i := range in {
		if in[i] > 10 {
			hi++
		}
		if math.Abs(float64(up[i])) > 10 {
			out++
		}
	}
	if hi == 0 || out == 0 {
		t.Fatalf("%d gates above 10 and %d ups outside +-10: the clamps are not exercised", hi, out)
	}
	// The kinds must disagree, or "GELU works" would be a claim about SiLU.
	for a := 0; a < len(outs); a++ {
		for b := a + 1; b < len(outs); b++ {
			same := true
			for i := range outs[a] {
				if outs[a][i] != outs[b][i] {
					same = false
					break
				}
			}
			if same {
				t.Fatalf("%v and %v produced identical output", mainkernels.Ungated[a], mainkernels.Ungated[b])
			}
		}
	}
}

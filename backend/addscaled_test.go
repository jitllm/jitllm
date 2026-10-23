package backend_test

import (
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestAddScaledRoundsOnce holds every device's scaled residual add to the
// host's BIT FOR BIT: a + alpha*b rounded once, as the host's axpy computes it
// (VFMADD231PS, FMLA). The reference is the exact value rounded to f32 by
// math/big, not a Go expression, which the arm64 compiler fuses on its own.
// Granite's residual_scale (0.22) is the alpha: at 1 the fused and the
// two-rounding forms agree, so a llama could never see the difference. The
// input is checked to discriminate: the two-rounding form must part from the
// reference somewhere.
func TestAddScaledRoundsOnce(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const n = 1 << 16
	rng := rand.New(rand.NewSource(5))
	a, b := make([]float32, n), make([]float32, n)
	for i := range a {
		a[i], b[i] = float32(rng.NormFloat64()*12), float32(rng.NormFloat64())
	}
	const alpha = float32(0.22)
	want := make([]float32, n)
	twice := 0
	for i := range want {
		var x big.Float
		x.SetPrec(200).SetFloat64(float64(alpha))
		x.Mul(&x, new(big.Float).SetPrec(200).SetFloat64(float64(b[i])))
		x.Add(&x, new(big.Float).SetPrec(200).SetFloat64(float64(a[i])))
		want[i], _ = x.Float32()
		p := float32(alpha * b[i]) // an explicit conversion forbids the fusion
		if a[i]+p != want[i] {
			twice++
		}
	}
	if twice == 0 {
		t.Fatal("the two-rounding form agrees with the fused one on every element: the input cannot tell them apart")
	}
	k, err := kernels.AddScaled(n)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			kk, err := d.Compile(k)
			if err != nil {
				t.Fatal(err)
			}
			defer kk.Close()
			g := newGPU(t, d)
			defer g.free()
			nan := make([]float32, n)
			for i := range nan {
				nan[i] = float32(math.NaN())
			}
			bo := g.up(f32bytes(nan))
			if err := kk.Launch(n/128, 128, g.up(f32bytes(a)), g.up(f32bytes(b)), g.up(f32bytes([]float32{alpha})),
				bo); err != nil {
				t.Fatal(err)
			}
			raw := make([]byte, 4*n)
			if err := bo.Read(raw); err != nil {
				t.Fatal(err)
			}
			bad := 0
			for i := range want {
				got := math.Float32frombits(uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 |
					uint32(raw[4*i+3])<<24)
				if math.Float32bits(got) != math.Float32bits(want[i]) {
					bad++
				}
			}
			if bad > 0 {
				t.Fatalf("%s: %d of %d scaled adds differ from a + alpha*b rounded once (the two-rounding form "+
					"differs in %d)", d.API(), bad, n, twice)
			}
			t.Logf("%s: %d scaled adds rounded once, as the host's axpy; two roundings would differ in %d",
				d.API(), n, twice)
		})
	}
}

package backend_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestRouterMatVecMatchesADotProduct: the router's logits against a float64
// dot product, with a ragged k (the clamped, selected-away tail), row counts
// on and off a group, an output poisoned with NaN and a guard word after it
// that must survive. Non-finite output is a failure before any tolerance.
//
// Over both weight types: Mixtral's router is F16, and a kernel reading it as
// F32 would route on bytes past its buffer. The odd shapes (1x7, 3x129) are the
// ones whose last element shares a word with nothing, which is why the tier
// pads the upload and this does too.
func TestRouterMatVecMatchesADotProduct(t *testing.T) {
	// Subtests, because each takes gpuLock and releases it at its own end.
	t.Run("f32", func(t *testing.T) { routerMatVecCase(t, false) })
	t.Run("f16", func(t *testing.T) { routerMatVecCase(t, true) })
}

func routerMatVecCase(t *testing.T, f16 bool) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(7))
	for _, d := range devs {
		defer d.Close()
		for _, c := range []struct{ rows, k int }{{64, 2048}, {128, 2048}, {1, 7}, {3, 129}, {60, 2304}, {512, 1000}} {
			w := make([]float32, c.rows*c.k)
			x := make([]float32, c.k)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
			}
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			wb := f32bytes(w)
			if f16 {
				wb = make([]byte, (2*len(w)+3)&^3)
				for i, v := range w {
					h := f32ToF16(v)
					w[i] = float32(quant.DecodeHalf(h))
					binary.LittleEndian.PutUint16(wb[2*i:], h)
				}
			}
			k, err := mainkernels.RouterMatVecOf(c.rows, c.k, f16)
			if err != nil {
				t.Fatal(err)
			}
			if err := k.Validate(); err != nil {
				t.Fatal(err)
			}
			kern, err := d.Compile(k)
			if err != nil {
				t.Fatalf("%s: compile: %v", d.API(), err)
			}
			g := newGPU(t, d)
			bw, bx := g.up(wb), g.up(f32bytes(x))
			poison := make([]float32, c.rows+1)
			for i := range poison {
				poison[i] = float32(math.NaN())
			}
			const guard = 0x5EED5EED
			pb := f32bytes(poison)
			binary.LittleEndian.PutUint32(pb[c.rows*4:], guard)
			bo := g.up(pb)
			if err := kern.Launch((mainkernels.RouterThreads(c.rows)+127)/128, 128, bw, bx, bo); err != nil {
				t.Fatalf("%s: launch: %v", d.API(), err)
			}
			raw := make([]byte, len(pb))
			if err := bo.Read(raw); err != nil {
				t.Fatal(err)
			}
			kern.Close()
			g.free()
			if gw := binary.LittleEndian.Uint32(raw[c.rows*4:]); gw != guard {
				t.Fatalf("%s f16=%v %dx%d: wrote past the output (guard %#x)", d.API(), f16, c.rows, c.k, gw)
			}
			for r := 0; r < c.rows; r++ {
				got := float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[r*4:])))
				var want float64
				for i := 0; i < c.k; i++ {
					want += float64(w[r*c.k+i]) * float64(x[i])
				}
				if math.IsNaN(got) || math.IsInf(got, 0) {
					t.Fatalf("%s f16=%v %dx%d: row %d is %v", d.API(), f16, c.rows, c.k, r, got)
				}
				if diff := math.Abs(got - want); diff > 1e-3*(1+math.Sqrt(float64(c.k))) {
					t.Fatalf("%s f16=%v %dx%d: row %d %v, want %v", d.API(), f16, c.rows, c.k, r, got, want)
				}
			}
		}
	}
}

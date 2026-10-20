package backend_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// TestSoftcapMatchesTheHost runs gemma2's attention cap on every backend
// against float64 tanh, over scores that saturate both ways, and requires a
// masked key (-inf) to come back -inf rather than -c.
func TestSoftcapMatchesTheHost(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const n, c = 1000, float32(50)
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(math.Sin(float64(i)*0.37) * float64(i))
	}
	in[7], in[500] = float32(math.Inf(-1)), float32(math.Inf(-1))
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := newGPU(t, d)
			defer g.free()
			bIn, bOut := g.up(f32bytes(in)), g.up(make([]byte, n*4))
			kk, err := oldkernels.Softcap(n, c)
			if err != nil {
				t.Fatal(err)
			}
			kern, err := d.Compile(kk)
			if err != nil {
				t.Fatal(err)
			}
			defer kern.Close()
			if err := kern.Launch((n+127)/128, 128, bIn, bOut); err != nil {
				t.Fatal(err)
			}
			raw := make([]byte, n*4)
			if err := bOut.Read(raw); err != nil {
				t.Fatal(err)
			}
			sat := 0
			for i := 0; i < n; i++ {
				got := float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])))
				if math.IsInf(float64(in[i]), -1) {
					if !math.IsInf(got, -1) {
						t.Fatalf("masked key %d came back %v, want -inf", i, got)
					}
					continue
				}
				want := float64(c) * math.Tanh(float64(in[i])/float64(c))
				if math.Abs(float64(in[i])) > 5*float64(c) {
					sat++
				}
				if math.Abs(got-want) > 2e-5*float64(c) {
					t.Fatalf("softcap[%d] of %v = %v, want %v", i, in[i], got, want)
				}
			}
			if sat == 0 {
				t.Fatal("no score saturates: the clamps are untested")
			}
		})
	}
}

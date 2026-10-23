package backend_test

import (
	"fmt"
	"github.com/jitllm/jitllm/internal/oracle"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestQuantizePaddedMatchesTheHost holds kernels.QuantizePadded -- rows of k
// floats quantized as rows of kp whose columns past k are zero, the
// activation of a matrix packed at kp (a tower 144 wide) -- to the host's
// quantizer over the same rows padded with zeros, bit for bit, on ragged
// widths either side of a block and both window kinds. The violation is the
// rows quantized as if they were dense at kp, reading the next row into each
// pad, which must move the int8s.
func TestQuantizePaddedMatchesTheHost(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	for _, c := range []struct{ rows, k, window int }{
		{1, 1, 32}, {3, 7, 32}, {5, 31, 32}, {4, 33, 32}, {3, 65, 32},
		{7, 144, 32}, {2, 143, 32}, {3, 145, 32}, {9, 144, 128}, {3, 200, 64},
	} {
		kp := (c.k + 31) / 32 * 32
		window := min(c.window, kp)
		rng := rand.New(rand.NewSource(int64(c.rows*1000 + c.k)))
		x := make([]float32, c.rows*c.k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		padded := make([]float32, c.rows*kp)
		for r := 0; r < c.rows; r++ {
			copy(padded[r*kp:], x[r*c.k:(r+1)*c.k])
		}
		n := c.rows * kp
		wantA, wantD, wantS := make([]uint32, n/4), make([]float32, n/32), make([]float32, n/16)
		if err := oracle.PackActivationsInto(wantA, wantD, wantS, padded, window); err != nil {
			t.Fatal(err)
		}
		for _, d := range devs {
			t.Run(fmt.Sprintf("%s/%dx%d/w%d", d.API(), c.rows, c.k, window), func(t *testing.T) {
				got := quantPadded(t, d, x, c.rows, c.k, kp, window)
				diff := 0
				for i := 0; i < n; i++ {
					if got[i] != byte(wantA[i/4]>>(8*(i%4))) {
						diff++
					}
				}
				if diff != 0 {
					t.Fatalf("%d of %d int8s differ from the host's over zero-padded rows", diff, n)
				}
			})
		}
	}
	// The violation: 144-wide rows read as dense 160-wide ones.
	const rows, k, kp = 3, 144, 160
	x := make([]float32, rows*kp)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)))
	}
	for _, d := range devs {
		good := quantPadded(t, d, x[:rows*k], rows, k, kp, 32)
		bad := quantDense(t, d, x[:rows*kp], 32)
		moved := 0
		for i := range good {
			if good[i] != bad[i] {
				moved++
			}
		}
		if moved == 0 {
			t.Errorf("%s: rows read dense at %d move no int8: the gate does not see the padding", d.API(), kp)
		}
	}
}

func quantPadded(t *testing.T, d backend.Device, x []float32, rows, k, kp, window int) []byte {
	t.Helper()
	kern, err := kernels.QuantizePadded(rows, k, kp, window)
	if err != nil {
		t.Fatal(err)
	}
	return quantRun(t, d, kern, x, rows*kp)
}

func quantDense(t *testing.T, d backend.Device, x []float32, window int) []byte {
	t.Helper()
	kern, err := kernels.Quantize(len(x), window)
	if err != nil {
		t.Fatal(err)
	}
	return quantRun(t, d, kern, x, len(x))
}

// quantRun launches a quantizer over n padded elements and returns its int8s;
// the int8 buffer starts as 0xFF bytes, so a word nothing wrote shows.
func quantRun(t *testing.T, d backend.Device, kern *ir.Kernel, x []float32, n int) []byte {
	t.Helper()
	kk, err := d.Compile(kern)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer kk.Close()
	g := newGPU(t, d)
	defer g.free()
	poison := make([]byte, n)
	for i := range poison {
		poison[i] = 0xFF
	}
	nan := make([]float32, n/32+n/16)
	for i := range nan {
		nan[i] = float32(math.NaN())
	}
	bx, ba, bax := g.up(f32bytes(x)), g.up(poison), g.up(f32bytes(nan))
	if err := kk.Launch((kernels.QuantizeThreads(n/32)+127)/128, 128, bx, ba, bax); err != nil {
		t.Fatalf("launch: %v", err)
	}
	got := make([]byte, n)
	if ba.Read(got) != nil {
		t.Fatal("read")
	}
	return got
}

package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// TestQuantizePaddedMatchesTheHost holds oldkernels.QuantizePadded -- rows of k
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
		{1, 1, 32}, {3, 7, 32}, {3, 8, 32}, {2, 69, 32}, {5, 31, 32}, {4, 33, 32}, {3, 65, 32},
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
		if err := kernels.PackActivationsInto(wantA, wantD, wantS, padded, window); err != nil {
			t.Fatal(err)
		}
		for _, d := range devs {
			t.Run(fmt.Sprintf("%s/%dx%d/w%d", d.API(), c.rows, c.k, window), func(t *testing.T) {
				got, ax := quantPadded(t, d, x, c.rows, c.k, kp, window)
				// The scales and the 16-sums the matvecs read beside the int8s: a
				// wrong tail scale leaves the int8s of a row right and its dot wrong.
				for i := range wantD {
					if ax[i] != wantD[i] {
						t.Fatalf("scale %d: %g, the host's %g", i, ax[i], wantD[i])
					}
				}
				for i := range wantS {
					if ax[n/32+i] != wantS[i] {
						t.Fatalf("sum %d: %g, the host's %g", i, ax[n/32+i], wantS[i])
					}
				}
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
		good, _ := quantPadded(t, d, x[:rows*k], rows, k, kp, 32)
		bad, _ := quantDense(t, d, x[:rows*kp], 32)
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

func quantPadded(t *testing.T, d backend.Device, x []float32, rows, k, kp, window int) ([]byte, []float32) {
	t.Helper()
	kern, err := oldkernels.QuantizePadded(rows, k, kp, window)
	if err != nil {
		t.Fatal(err)
	}
	return quantRun(t, d, kern, x, rows*kp)
}

func quantDense(t *testing.T, d backend.Device, x []float32, window int) ([]byte, []float32) {
	t.Helper()
	kern, err := oldkernels.Quantize(len(x), window)
	if err != nil {
		t.Fatal(err)
	}
	return quantRun(t, d, kern, x, len(x))
}

// quantRun launches a quantizer over n padded elements and returns its int8s
// and its [scales | sums]. Both buffers start poisoned (0xFF bytes, NaNs) and
// carry a guard word past their end, which must come back untouched.
func quantRun(t *testing.T, d backend.Device, kern *ir.Kernel, x []float32, n int) ([]byte, []float32) {
	t.Helper()
	const guard = 0xA5
	kk, err := d.Compile(kern)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer kk.Close()
	g := newGPU(t, d)
	defer g.free()
	poison := make([]byte, n+4)
	for i := range poison {
		poison[i] = 0xFF
	}
	copy(poison[n:], []byte{guard, guard, guard, guard})
	nax := n/32 + n/16
	nan := make([]float32, nax+1)
	for i := range nan {
		nan[i] = float32(math.NaN())
	}
	nan[nax] = math.Float32frombits(0xA5A5A5A5)
	bx, ba, bax := g.up(f32bytes(x)), g.up(poison), g.up(f32bytes(nan))
	if err := kk.Launch((oldkernels.QuantizeThreads(n/32)+127)/128, 128, bx, ba, bax); err != nil {
		t.Fatalf("launch: %v", err)
	}
	got := make([]byte, n+4)
	if ba.Read(got) != nil {
		t.Fatal("read")
	}
	for _, b := range got[n:] {
		if b != guard {
			t.Fatalf("the guard word past the int8 rows was written: % x", got[n:])
		}
	}
	axb := make([]byte, 4*(nax+1))
	if bax.Read(axb) != nil {
		t.Fatal("read")
	}
	ax := make([]float32, nax+1)
	for i := range ax {
		ax[i] = math.Float32frombits(binary.LittleEndian.Uint32(axb[4*i:]))
	}
	if math.Float32bits(ax[nax]) != 0xA5A5A5A5 {
		t.Fatal("the guard word past the scales was written")
	}
	return got[:n], ax[:nax]
}

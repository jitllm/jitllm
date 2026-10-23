package backend_test

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestRMSNormQuantIsNormThenQuantize: one RMSNormQuantRows launch writes, BIT
// FOR BIT, what RMSNormRows followed by Quantize writes -- the norm output, the
// packed int8, the scales and the per-16 sums -- over widths that take one and
// two passes of the group, windows of 32 and 256, one and three rows, with and
// without the residual add and the +1 weight. Every output is NaN-poisoned.
func TestRMSNormQuantIsNormThenQuantize(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(3))
	for _, d := range devs {
		defer d.Close()
		for _, k := range []int{4096, 2048, 5120, 1024, 512, 256, 128, 64, 1536, 1152, 2304, 2560, 3072, 896} {
			// k and 8192 are the row windows a per-token model asks for (BitNet):
			// the fused kernel takes one scale a row, Quantize the clamped window.
			for _, win := range []int{32, 256, k, 8192} {
				for _, rows := range []int{1, 3, 8, 16} {
					for _, add := range []bool{false, true} {
						for _, one := range []bool{false, true} {
							name := fmt.Sprintf("%s/k%d/w%d/r%d/add%v/one%v", d.API(), k, win, rows, add, one)
							rmsQuantCase(t, d, rng, name, k, win, rows, add, one, false)
							if d.API() == "ptx" {
								rmsQuantCase(t, d, rng, name+"/warp", k, win, rows, add, one, true)
							}
						}
					}
				}
			}
		}
	}
}

func rmsQuantCase(t *testing.T, d backend.Device, rng *rand.Rand, name string, k, win, rows int, add, one, warp bool) {
	t.Helper()
	mkq, mkn := kernels.RMSNormQuantRows, kernels.RMSNormRows
	if warp {
		mkq, mkn = kernels.RMSNormQuantRowsWarp, kernels.RMSNormRowsWarp
	}
	fused, err := mkq(k, rows, 1e-5, one, add, win)
	if err != nil {
		t.Logf("%s: not a shape (%v)", name, err)
		return
	}
	norm, err := mkn(k, rows, 1e-5, one, add)
	if err != nil {
		t.Fatal(err)
	}
	quant, err := kernels.Quantize(rows*k, kernels.ClampWindow(win, k))
	if err != nil {
		t.Fatal(err)
	}
	kf, err := d.Compile(fused)
	if err != nil {
		t.Fatalf("%s: compile: %v", name, err)
	}
	defer kf.Close()
	kn, err := d.Compile(norm)
	if err != nil {
		t.Fatal(err)
	}
	defer kn.Close()
	kq, err := d.Compile(quant)
	if err != nil {
		t.Fatal(err)
	}
	defer kq.Close()
	g := newGPU(t, d)
	defer g.free()
	x, y, w := make([]float32, k*rows), make([]float32, k*rows), make([]float32, k)
	for i := range x {
		x[i], y[i] = float32(rng.NormFloat64()*3), float32(rng.NormFloat64())
	}
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	nan := func(n int) []byte {
		p := make([]float32, n)
		for i := range p {
			p[i] = float32(math.NaN())
		}
		return f32bytes(p)
	}
	nb := rows * k / 32
	bx, by, bw := g.up(f32bytes(x)), g.up(f32bytes(y)), g.up(f32bytes(w))
	type outs struct{ sum, h, a, ax backend.Buf }
	mk := func() outs {
		return outs{g.up(nan(k * rows)), g.up(nan(k * rows)), g.up(nan(k * rows / 4)), g.up(nan(3 * nb))}
	}
	ref, got := mk(), mk()
	nbufs := func(o outs) []backend.Buf {
		if add {
			return []backend.Buf{bx, by, bw, o.sum, o.h}
		}
		return []backend.Buf{bx, bw, o.h}
	}
	if err := kn.Launch(rows, kernels.RMSNormGroup, nbufs(ref)...); err != nil {
		t.Fatal(err)
	}
	qt := kernels.QuantizeThreads(nb)
	if err := kq.Launch((qt+127)/128, 128, ref.h, ref.a, ref.ax); err != nil {
		t.Fatal(err)
	}
	if err := kf.Launch(rows, kernels.RMSNormGroup, append(nbufs(got), got.a, got.ax)...); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	for _, c := range []struct {
		what   string
		r, o   backend.Buf
		n      int
		exists bool
	}{{"sum", ref.sum, got.sum, k * rows, add}, {"h", ref.h, got.h, k * rows, true},
		{"packed", ref.a, got.a, k * rows / 4, true}, {"scales+sums", ref.ax, got.ax, 3 * nb, true}} {
		if !c.exists {
			continue
		}
		rb, gb := make([]byte, c.n*4), make([]byte, c.n*4)
		if c.r.Read(rb) != nil || c.o.Read(gb) != nil {
			t.Fatalf("%s: read", name)
		}
		if !bytes.Equal(rb, gb) {
			diff := 0
			for i := 0; i < len(rb); i += 4 {
				if !bytes.Equal(rb[i:i+4], gb[i:i+4]) {
					diff++
				}
			}
			t.Fatalf("%s: %s: %d of %d words differ", name, c.what, diff, c.n)
		}
		if bytes.Equal(rb, nan(c.n)) {
			t.Fatalf("%s: %s: the reference never wrote -- a degenerate oracle", name, c.what)
		}
	}
}

package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// deviceTopK runs the device sampler's half -- the penalty when hist is
// non-empty, then both top-k passes -- and returns the k candidates.
func deviceTopK(t *testing.T, d backend.Device, x []float32, hist []int32, pen float32, k int) ([]float32, []int32) {
	t.Helper()
	n := len(x)
	g := newGPU(t, d)
	defer g.free()
	launch := func(name string, groups int, kk func() (*ir.Kernel, error), bufs ...backend.Buf) {
		t.Helper()
		kern, err := kk()
		if err != nil {
			t.Fatal(err)
		}
		if err := kern.Validate(); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		c, err := d.Compile(kern)
		if err != nil {
			t.Fatalf("%s: compile: %v", name, err)
		}
		defer c.Close()
		if err := c.Launch(groups, kernels.SampleGroup, bufs...); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
	in := g.up(f32bytes(x))
	src := in
	if len(hist) > 0 {
		args := make([]byte, 4*kernels.SampleArgsWords(len(hist)))
		binary.LittleEndian.PutUint32(args, uint32(len(hist)))
		binary.LittleEndian.PutUint32(args[4:], math.Float32bits(pen))
		for i, h := range hist {
			binary.LittleEndian.PutUint32(args[8+4*i:], uint32(h))
		}
		src = g.up(seedPoison(4 * n))
		launch("penalty", kernels.SampleGroups(n), func() (*ir.Kernel, error) { return kernels.SamplePenalty(n) },
			in, g.up(args), src)
	}
	groups := kernels.SampleSlices(n)
	v1, i1 := g.up(seedPoison(4*groups*k)), g.up(seedPoison(4*groups*k))
	launch("pass 1", groups, func() (*ir.Kernel, error) {
		return kernels.SampleTopK(n, kernels.SampleSlice, k, false)
	}, src, src, v1, i1)
	v2, i2 := g.up(seedPoison(4*k+4)), g.up(seedPoison(4*k+4))
	launch("pass 2", 1, func() (*ir.Kernel, error) {
		return kernels.SampleTopK(groups*k, groups*k, k, true)
	}, v1, i1, v2, i2)
	bv, bi := make([]byte, 4*k+4), make([]byte, 4*k+4)
	if err := v2.Read(bv); err != nil {
		t.Fatal(err)
	}
	if err := i2.Read(bi); err != nil {
		t.Fatal(err)
	}
	if binary.LittleEndian.Uint32(bv[4*k:]) != 0x5EED5EED || binary.LittleEndian.Uint32(bi[4*k:]) != 0x5EED5EED {
		t.Fatalf("the second pass wrote past its %d candidates", k)
	}
	vs, ids := make([]float32, k), make([]int32, k)
	for j := range k {
		vs[j] = math.Float32frombits(binary.LittleEndian.Uint32(bv[4*j:]))
		ids[j] = int32(binary.LittleEndian.Uint32(bi[4*j:]))
	}
	return vs, ids
}

func seedPoison(n int) []byte {
	b := make([]byte, n)
	for i := 0; i+4 <= n; i += 4 {
		binary.LittleEndian.PutUint32(b[i:], 0x5EED5EED)
	}
	return b
}

// hostTopK is the host sampler's own selection: its penalty kernel over a
// copy, then nn.SampleOrder's first k.
func hostTopK(x []float32, hist []int32, pen float32, k int) ([]float32, []int32) {
	vals := append([]float32(nil), x...)
	if len(hist) > 0 {
		off := make([]int64, len(hist))
		for i, h := range hist {
			off[i] = 4 * int64(h)
		}
		nn.SamplePenalty32JIT(vals, off, pen)
	}
	var o nn.SampleOrder
	o.Begin(vals)
	vs, ids := make([]float32, k), make([]int32, k)
	for j := range k {
		vs[j], ids[j], _ = o.Next()
	}
	return vs, ids
}

// TestDeviceSamplerSelectsTheHostsCandidates: the device penalty and top-k
// return the host sampler's candidates bit for bit -- the same values, the
// same ids, in the same order -- over ragged vocabularies, ties planted
// across slices, a flat row, negative logits under the penalty and a history
// that repeats a token.
func TestDeviceSamplerSelectsTheHostsCandidates(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(11))
	ran := 0
	for _, d := range devs {
		defer d.Close()
		for _, n := range []int{300, 2049, 32000, 151936} {
			for _, k := range []int{1, 7, 40, 256} {
				if k >= n {
					continue
				}
				for _, shape := range []string{"random", "ties", "flat", "penalty"} {
					x := make([]float32, n)
					for i := range x {
						x[i] = float32(rng.NormFloat64() * 4)
					}
					var hist []int32
					pen := float32(1)
					switch shape {
					case "ties":
						for range 20 {
							x[rng.Intn(n)] = 30
						}
					case "flat":
						for i := range x {
							x[i] = 1.5
						}
					case "penalty":
						pen = 1.3
						for range 64 {
							hist = append(hist, int32(rng.Intn(n)))
						}
						hist = append(hist, hist[3], hist[3]) // a repeat is penalized again
						x[hist[3]] = 25
					}
					name := fmt.Sprintf("%s/n%d/k%d/%s", d.API(), n, k, shape)
					wv, wi := hostTopK(x, hist, pen, k)
					gv, gi := deviceTopK(t, d, x, hist, pen, k)
					for j := range k {
						if gi[j] != wi[j] || math.Float32bits(gv[j]) != math.Float32bits(wv[j]) {
							t.Fatalf("%s: candidate %d is (%d, %v), the host's (%d, %v)",
								name, j, gi[j], gv[j], wi[j], wv[j])
						}
					}
					ran++
				}
			}
		}
	}
	if ran == 0 {
		t.Fatal("no configuration ran")
	}
	for _, d := range devs {
		t.Logf("ran on %s", d.API())
	}
	t.Logf("%d configurations", ran)
}

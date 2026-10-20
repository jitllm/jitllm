package backend_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// The convolutional tower's kernels (kernels/conv.go) on every device against
// the plain loops they are, at ragged shapes: a padded grid at stride 1 and 2,
// odd channel counts, a 1-wide filter, and attention whose keys are fewer
// than its queries. Every output buffer starts NaN, so an element no thread
// writes fails.

func convRun(t *testing.T, d backend.Device, k *ir.Kernel, err error, threads int, out int, in ...[]float32) []float32 {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	kk, err := d.Compile(k)
	if err != nil {
		t.Fatal(err)
	}
	defer kk.Close()
	g := newGPU(t, d)
	defer g.free()
	bufs := make([]backend.Buf, 0, len(in)+1)
	for _, x := range in {
		bufs = append(bufs, g.up(f32bytes(x)))
	}
	nan := make([]float32, out)
	for i := range nan {
		nan[i] = float32(math.NaN())
	}
	o := g.up(f32bytes(nan))
	bufs = append(bufs, o)
	if err := kk.Launch((threads+127)/128, 128, bufs...); err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, 4*out)
	if err := o.Read(raw); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, out)
	for i := range got {
		got[i] = math.Float32frombits(uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 |
			uint32(raw[4*i+3])<<24)
	}
	return got
}

func convCheck(t *testing.T, what string, got []float32, want []float64, tol float64) {
	t.Helper()
	for i, w := range want {
		if d := math.Abs(float64(got[i]) - w); !(d <= tol*(1+math.Abs(w))) {
			t.Fatalf("%s: [%d] = %v, want %v", what, i, got[i], w)
		}
	}
}

func convRand(rng *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	return x
}

func TestConvKernelsMatchTheLoops(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(11))
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, sh := range []struct{ h, w, c, k, s int }{
				{7, 9, 5, 3, 2}, {8, 6, 13, 3, 1}, {5, 5, 3, 5, 2}, {4, 7, 9, 1, 2}, {6, 6, 32, 3, 2},
			} {
				ho, wo := (sh.h+sh.s-1)/sh.s, (sh.w+sh.s-1)/sh.s
				ph := max((ho-1)*sh.s+sh.k-sh.h, 0)
				pw := max((wo-1)*sh.s+sh.k-sh.w, 0)
				top, left := ph/2, pw/2
				hp, wp := (ho-1)*sh.s+sh.k, (wo-1)*sh.s+sh.k
				x := convRand(rng, sh.h*sh.w*sh.c)
				k, err := oldkernels.PadRows(sh.h, sh.w, sh.c, hp, wp, top, left)
				pad := convRun(t, d, k, err, hp*wp*sh.c, hp*wp*sh.c, x)
				want := make([]float64, hp*wp*sh.c)
				for y := 0; y < hp; y++ {
					for xx := 0; xx < wp; xx++ {
						sy, sx := y-top, xx-left
						if sy < 0 || sy >= sh.h || sx < 0 || sx >= sh.w {
							continue
						}
						for c := 0; c < sh.c; c++ {
							want[(y*wp+xx)*sh.c+c] = float64(x[(sy*sh.w+sx)*sh.c+c])
						}
					}
				}
				convCheck(t, "PadRows", pad, want, 0)
				w := convRand(rng, sh.k*sh.k*sh.c)
				k, err = oldkernels.DWConv(sh.k, sh.s, wp, sh.c, ho, wo)
				dw := convRun(t, d, k, err, ho*wo*sh.c, ho*wo*sh.c, pad, w)
				want = make([]float64, ho*wo*sh.c)
				for y := 0; y < ho; y++ {
					for xx := 0; xx < wo; xx++ {
						for c := 0; c < sh.c; c++ {
							acc := 0.0
							for ky := 0; ky < sh.k; ky++ {
								for kx := 0; kx < sh.k; kx++ {
									acc += float64(pad[((y*sh.s+ky)*wp+xx*sh.s+kx)*sh.c+c]) * float64(w[(ky*sh.k+kx)*sh.c+c])
								}
							}
							want[(y*wo+xx)*sh.c+c] = acc
						}
					}
				}
				convCheck(t, "DWConv", dw, want, 1e-5)
				kk := sh.k * sh.k * sh.c
				kpad := (kk + 31) / 32 * 32
				n := ho * wo
				k, err = oldkernels.Im2col(sh.k, sh.s, wp, sh.c, wo, kpad, n, len(pad))
				col := convRun(t, d, k, err, n*kpad, n*kpad, pad)
				want = make([]float64, n*kpad)
				for p := 0; p < n; p++ {
					y, xx := p/wo, p%wo
					for ky := 0; ky < sh.k; ky++ {
						for kx := 0; kx < sh.k; kx++ {
							for c := 0; c < sh.c; c++ {
								want[p*kpad+(ky*sh.k+kx)*sh.c+c] = float64(pad[((y*sh.s+ky)*wp+xx*sh.s+kx)*sh.c+c])
							}
						}
					}
				}
				convCheck(t, "Im2col", col, want, 0)
			}
			for _, sh := range []struct{ n, m, heads, kd int }{{12, 4, 3, 8}, {9, 9, 1, 5}, {16, 4, 2, 32}} {
				q := convRand(rng, sh.n*sh.heads*sh.kd)
				kv := convRand(rng, sh.m*sh.kd)
				v := convRand(rng, sh.m*sh.kd)
				k, err := oldkernels.MQAScores(sh.n, sh.m, sh.heads, sh.kd)
				sc := convRun(t, d, k, err, sh.n*sh.heads*sh.m, sh.n*sh.heads*sh.m, q, kv)
				want := make([]float64, len(sc))
				for i := range want {
					j, rh := i%sh.m, i/sh.m
					acc := 0.0
					for e := 0; e < sh.kd; e++ {
						acc += float64(q[rh*sh.kd+e]) * float64(kv[j*sh.kd+e])
					}
					want[i] = acc / math.Sqrt(float64(sh.kd))
				}
				convCheck(t, "MQAScores", sc, want, 1e-5)
				rows := sh.n * sh.heads
				k, err = oldkernels.RowSoftmax(rows, sh.m)
				pr := convRun(t, d, k, err, rows, rows*sh.m, sc)
				for r := 0; r < rows; r++ {
					mx := math.Inf(-1)
					for j := 0; j < sh.m; j++ {
						mx = math.Max(mx, float64(sc[r*sh.m+j]))
					}
					s := 0.0
					for j := 0; j < sh.m; j++ {
						s += math.Exp(float64(sc[r*sh.m+j]) - mx)
					}
					for j := 0; j < sh.m; j++ {
						want[r*sh.m+j] = math.Exp(float64(sc[r*sh.m+j])-mx) / s
					}
				}
				convCheck(t, "RowSoftmax", pr, want[:rows*sh.m], 1e-5)
				k, err = oldkernels.MQAAcc(sh.n, sh.m, sh.heads, sh.kd)
				ao := convRun(t, d, k, err, rows*sh.kd, rows*sh.kd, pr, v)
				want = make([]float64, rows*sh.kd)
				for i := range want {
					e, rh := i%sh.kd, i/sh.kd
					for j := 0; j < sh.m; j++ {
						want[i] += float64(pr[rh*sh.m+j]) * float64(v[j*sh.kd+e])
					}
				}
				convCheck(t, "MQAAcc", ao, want, 1e-5)
			}
		})
	}
}

package backend_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestConformerKernelsMatchTheLoops runs a FastConformer block's kernels on
// every device against the loops they are, over a scratch of R rows of which
// the first t are real -- the rest NaN, so a kernel that reads one into a
// real row fails (a multiply by zero would not hide it).
func TestConformerKernelsMatchTheLoops(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(5))
	nan := float32(math.NaN())
	for _, d := range devs {
		for _, tc := range []struct{ r, tn, heads, kd, k int }{{8, 5, 2, 4, 3}, {16, 16, 3, 8, 9}, {12, 1, 1, 5, 5}} {
			r, tn, heads, kd := tc.r, tc.tn, tc.heads, tc.kd
			dm := heads * kd
			poison := func(x []float32, valid int) []float32 {
				for i := valid; i < len(x); i++ {
					x[i] = nan
				}
				return x
			}
			nbuf := []float32{math.Float32frombits(uint32(tn))}
			q := poison(convRand(rng, r*dm), tn*dm)
			k := poison(convRand(rng, r*dm), tn*dm)
			v := poison(convRand(rng, r*dm), tn*dm)
			p := convRand(rng, (2*r-1)*dm)
			u, vb := convRand(rng, dm), convRand(rng, dm)

			// Scores: the valid rows' valid keys.
			ker, err := mainkernels.RelPosScores(r, heads, kd)
			got := convRun(t, d, ker, err, r*heads*r, r*heads*r, q, k, p, u, vb, nbuf)
			for i := 0; i < tn; i++ {
				for h := 0; h < heads; h++ {
					for j := 0; j < r; j++ {
						at := (i*heads+h)*r + j
						if j >= tn {
							if got[at] > -1e38 {
								t.Fatalf("%s: score (%d,%d,%d) past t is %v, want masked", d.Name(), i, h, j, got[at])
							}
							continue
						}
						var s float64
						for e := 0; e < kd; e++ {
							c := h*kd + e
							s += float64(q[i*dm+c]+u[c])*float64(k[j*dm+c]) +
								float64(q[i*dm+c]+vb[c])*float64(p[(r-1-i+j)*dm+c])
						}
						s /= math.Sqrt(float64(kd))
						if math.Abs(float64(got[at])-s) > 1e-4*(1+math.Abs(s)) {
							t.Fatalf("%s: score (%d,%d,%d) is %v, want %v", d.Name(), i, h, j, got[at], s)
						}
					}
				}
			}

			// The weighted sum over the valid keys only.
			probs := convRand(rng, r*heads*r)
			ker, err = mainkernels.AttnAccRows(r, heads, kd)
			got = convRun(t, d, ker, err, r*dm, r*dm, probs, v, nbuf)
			for i := 0; i < tn; i++ {
				for c := 0; c < dm; c++ {
					h := c / kd
					var s float64
					for j := 0; j < tn; j++ {
						s += float64(probs[(i*heads+h)*r+j]) * float64(v[j*dm+c])
					}
					if math.Abs(float64(got[i*dm+c])-s) > 1e-4*(1+math.Abs(s)) {
						t.Fatalf("%s: acc (%d,%d) is %v, want %v", d.Name(), i, c, got[i*dm+c], s)
					}
				}
			}

			// The centred depthwise convolution inside the valid rows.
			w := convRand(rng, tc.k*dm)
			ker, err = mainkernels.DWConv1dRows(tc.k, dm, r)
			got = convRun(t, d, ker, err, r*dm, r*dm, v, w, nbuf)
			half := tc.k / 2
			for i := 0; i < tn; i++ {
				for c := 0; c < dm; c++ {
					var s float64
					for uu := 0; uu < tc.k; uu++ {
						if src := i + uu - half; src >= 0 && src < tn {
							s += float64(w[uu*dm+c]) * float64(v[src*dm+c])
						}
					}
					if math.Abs(float64(got[i*dm+c])-s) > 1e-4*(1+math.Abs(s)) {
						t.Fatalf("%s: dwconv (%d,%d) is %v, want %v", d.Name(), i, c, got[i*dm+c], s)
					}
				}
			}

			// The folded BatchNorm.
			x := convRand(rng, r*dm)
			sc, sh := convRand(rng, dm), convRand(rng, dm)
			ker, err = mainkernels.AffineRows(dm, r)
			got = convRun(t, d, ker, err, r*dm, r*dm, x, sc, sh)
			want := make([]float64, r*dm)
			for i := range want {
				want[i] = float64(x[i])*float64(sc[i%dm]) + float64(sh[i%dm])
			}
			convCheck(t, d.Name()+" affine", got, want, 1e-5)
		}
	}
}

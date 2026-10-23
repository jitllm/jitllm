package backend_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestExpertRouter holds the device router to the host router kernel, nn.MoETopK32JIT.
//
// The tie cases are constructed: the host kernel gives equal probabilities to
// the lower index (ggml_argsort descending), and random floats never tie. A
// kernel ranking by "strictly greater" alone would give tied experts the same
// output slot and silently drop one.
func TestExpertRouter(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	cases := []struct {
		name string
		l    []float32
		k    int
	}{
		{"random8", nil, 2},
		{"random128", nil, 8},
		{"all-equal", []float32{1, 1, 1, 1, 1, 1, 1, 1}, 3},
		{"tie-at-the-cut", []float32{5, 3, 3, 1}, 2},
		{"tie-at-the-top", []float32{9, 9, 2, 1, 0, 0, 0, 0}, 2},
		{"negatives", []float32{-8, -2, -30, -2, -1, -9, -2, -3}, 3},
		{"k-equals-n", []float32{1, 4, 2, 3}, 4},
		{"k-is-one", []float32{1, 4, 2, 3}, 1},
	}
	rng := rand.New(rand.NewSource(20260907))
	for i := range cases {
		if cases[i].l == nil {
			n := 8
			if i == 1 {
				n = 128
			}
			cases[i].l = make([]float32, n)
			for j := range cases[i].l {
				cases[i].l[j] = float32(rng.NormFloat64() * 3)
			}
		}
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, c := range cases {
				t.Run(c.name, func(t *testing.T) {
					routerCase(t, d, c.l, c.k, true)
				})
				// olmoe does not renormalise, so its weight is the full
				// softmax over every expert rather than over the selected k.
				// The two agree only when k == n, which is why this runs the
				// same cases rather than trusting the renormalised ones.
				t.Run(c.name+"/nonorm", func(t *testing.T) {
					routerCase(t, d, c.l, c.k, false)
				})
				// Scaled: the host's MoERouteJIT applies routed_scaling_factor to
				// every route, softmax arm included.
				t.Run(c.name+"/scaled", func(t *testing.T) { routerCaseScaled(t, d, c.l, c.k, true, 2.5) })
				t.Run(c.name+"/nonorm-scaled", func(t *testing.T) { routerCaseScaled(t, d, c.l, c.k, false, 2.5) })
			}
		})
	}
}

func routerCase(t *testing.T, d backend.Device, l []float32, k int, norm bool) {
	routerCaseScaled(t, d, l, k, norm, 1)
}

// routerCaseScaled is routerCase with a routed_scaling_factor: the host's
// MoERouteJIT multiplies every selected weight by it after the normalisation,
// softmax or sigmoid alike.
func routerCaseScaled(t *testing.T, d backend.Device, l []float32, k int, norm bool, scale float32) {
	n := len(l)
	g := newGPU(t, d)
	defer g.free()
	bL := g.up(f32bytes(l))
	bSel := g.up(make([]byte, (k+1)*4))
	bTop := g.up(make([]byte, (k+1)*4))
	bW := g.up(make([]byte, k*4))

	run := func(mk func() (*ir.Kernel, error), grid, width int, args ...backend.Buf) {
		t.Helper()
		kk, err := mk()
		if err != nil {
			t.Fatal(err)
		}
		kern, err := d.Compile(kk)
		if err != nil {
			t.Fatal(err)
		}
		defer kern.Close()
		if err := kern.Launch(grid, width, args...); err != nil {
			t.Fatal(err)
		}
	}
	run(func() (*ir.Kernel, error) { return kernels.ExpertRank(kernels.MoERoute{NExpert: n, K: k}) },
		(n+127)/128, 128, bL, bSel, bTop)
	if norm {
		run(func() (*ir.Kernel, error) {
			return kernels.ExpertWeights(kernels.MoERoute{NExpert: n, K: k, Norm: true, Scale: scale})
		},
			1, 1, bTop, bW)
	} else {
		run(func() (*ir.Kernel, error) {
			return kernels.ExpertWeights(kernels.MoERoute{NExpert: n, K: k, Scale: scale})
		},
			1, 1, bTop, bW, bL)
	}

	selRaw, wRaw := make([]byte, (k+1)*4), make([]byte, k*4)
	if err := bSel.Read(selRaw); err != nil {
		t.Fatal(err)
	}
	if err := bW.Read(wRaw); err != nil {
		t.Fatal(err)
	}

	// Reference: the host kernel's rule, largest first, ties to the lower index.
	want := make([]int, 0, k)
	for len(want) < k {
		best := -1
		for e := 0; e < n; e++ {
			if used(want, e) {
				continue
			}
			if best < 0 || l[e] > l[best] {
				best = e
			}
		}
		want = append(want, best)
	}
	for j := 0; j < k; j++ {
		if got := int(binary.LittleEndian.Uint32(selRaw[j*4:])); got != want[j] {
			t.Fatalf("slot %d: expert %d, want %d (logits %v)", j, got, want[j], l)
		}
	}
	// Weights: softmax over the selected logits, which is the renormalised
	// full softmax with Z cancelled -- or, un-renormalised, the full softmax
	// itself, whose denominator runs over every expert.
	m := l[want[0]]
	var sum float64
	if norm {
		for _, e := range want {
			sum += math.Exp(float64(l[e] - m))
		}
	} else {
		for e := 0; e < n; e++ {
			sum += math.Exp(float64(l[e] - m))
		}
	}
	for j, e := range want {
		wf := math.Exp(float64(l[e]-m)) / sum * float64(scale)
		got := float64(math.Float32frombits(binary.LittleEndian.Uint32(wRaw[j*4:])))
		if math.Abs(got-wf) > 1e-6 {
			t.Errorf("slot %d weight: got %v want %v", j, got, wf)
		}
	}
}

func used(s []int, e int) bool {
	for _, v := range s {
		if v == e {
			return true
		}
	}
	return false
}

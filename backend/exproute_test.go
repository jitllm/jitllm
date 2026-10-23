package backend_test

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestExpertRouteMatchesRankAndWeights holds the fused softmax router to the
// two kernels it replaces, bit for bit: the same selection in the same slot
// order, the same logits in pTop, and weights that are the same floats --
// which they must be, since the maximum is order-free and both sums run in the
// same order. TestExpertRouter holds the pair to the host rule, so this closes
// the chain. The cases carry ties (at the top, at the cut, all equal), both
// renormalisations, and expert counts from 4 to 512 (qwen3next's), so a
// workgroup wider than a warp -- where the threadgroup-memory hand-off and its
// barrier are what the renormalised arm depends on -- is covered.
func TestExpertRouteMatchesRankAndWeights(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(20260927))
	randL := func(n int) []float32 {
		l := make([]float32, n)
		for i := range l {
			l[i] = float32(rng.NormFloat64() * 3)
		}
		return l
	}
	cases := []struct {
		name string
		l    []float32
		k    int
	}{
		{"olmoe", randL(64), 8},
		{"deepseek-v2-lite", randL(64), 6},
		{"gpt-oss", randL(32), 4},
		{"qwen3next", randL(512), 10},
		{"all-equal", []float32{1, 1, 1, 1, 1, 1, 1, 1}, 3},
		{"tie-at-the-cut", []float32{5, 3, 3, 1}, 2},
		{"tie-at-the-top", []float32{9, 9, 2, 1, 0, 0, 0, 0}, 2},
		{"k-equals-n", []float32{1, 4, 2, 3}, 4},
		{"k-is-one", []float32{1, 4, 2, 3}, 1},
		// A ragged last column for the warp route: 60 is not a whole number
		// of lanes, so four lanes read past the end and must stay -inf.
		{"ragged-60", randL(60), 8},
		{"tie-across-lanes", []float32{1, 7, 3, 7, 0, 7, 2, 7, 5, 6, 7, 1, 0, 2, 3, 4,
			7, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7}, 6},
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, c := range cases {
				for _, norm := range []bool{true, false} {
					for _, scale := range []float32{1, 2.5} {
						name := c.name
						if !norm {
							name += "/nonorm"
						}
						if scale != 1 {
							name += "/scaled"
						}
						t.Run(name, func(t *testing.T) { exprouteCase(t, d, c.l, c.k, norm, scale) })
					}
				}
			}
		})
	}
}

func exprouteCase(t *testing.T, d backend.Device, l []float32, k int, norm bool, scale float32) {
	n := len(l)
	r := kernels.MoERoute{NExpert: n, K: k, Norm: norm, Scale: scale}
	g := newGPU(t, d)
	defer g.free()
	bL := g.up(f32bytes(l))
	launch := func(kk *ir.Kernel, err error, grid, width int, args ...backend.Buf) {
		t.Helper()
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
	read := func(b backend.Buf, n int) []byte {
		p := make([]byte, n)
		if err := b.Read(p); err != nil {
			t.Fatal(err)
		}
		return p
	}
	// The pair.
	sel0, top0, w0 := g.up(make([]byte, (k+1)*4)), g.up(make([]byte, (k+1)*4)), g.up(make([]byte, (k+1)*4))
	kr, err := kernels.ExpertRank(r)
	launch(kr, err, (n+127)/128, 128, bL, sel0, top0)
	kw, err := kernels.ExpertWeights(r)
	if norm {
		launch(kw, err, 1, 1, top0, w0)
	} else {
		launch(kw, err, 1, 1, top0, w0, bL)
	}
	// The fused kernel, into buffers poisoned so an unwritten slot shows.
	poison := bytes.Repeat([]byte{0xFF}, (k+1)*4)
	sel1, top1, w1 := g.up(poison), g.up(poison), g.up(poison)
	kf, err := kernels.ExpertRoute(r)
	launch(kf, err, 1, kernels.ExpertRouteWidth(n), bL, sel1, top1, w1)

	if a, b := read(sel0, k*4), read(sel1, k*4); !bytes.Equal(a, b) {
		t.Fatalf("selection differs: pair %v fused %v", a, b)
	}
	if a, b := read(top0, k*4), read(top1, k*4); !bytes.Equal(a, b) {
		t.Fatalf("pTop differs: pair %v fused %v", a, b)
	}
	if a, b := read(w0, k*4), read(w1, k*4); !bytes.Equal(a, b) {
		t.Fatalf("weights differ: pair %v fused %v", a, b)
	}
	// And the one-warp form, to the same bits. It exists for the
	// renormalised route only (its sums keep ExpertWeights' order there and
	// cannot over every expert), and it needs the 32-lane subgroup.
	if !norm {
		return
	}
	if ok, why := backend.GuaranteedLanes(d, ir.SubgroupLanes); !ok {
		t.Logf("%s: no guaranteed 32-lane subgroup (%s); the warp route is not built here", d.API(), why)
		return
	}
	sel2, top2, w2 := g.up(poison), g.up(poison), g.up(poison)
	kw2, err := kernels.ExpertRouteWarp(r)
	launch(kw2, err, 1, ir.SubgroupLanes, bL, sel2, top2, w2)
	if a, b := read(sel0, k*4), read(sel2, k*4); !bytes.Equal(a, b) {
		t.Fatalf("warp selection differs: pair %v warp %v", a, b)
	}
	if a, b := read(top0, k*4), read(top2, k*4); !bytes.Equal(a, b) {
		t.Fatalf("warp pTop differs: pair %v warp %v", a, b)
	}
	if a, b := read(w0, k*4), read(w2, k*4); !bytes.Equal(a, b) {
		t.Fatalf("warp weights differ: pair %v warp %v", a, b)
	}
}

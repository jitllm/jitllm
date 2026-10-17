package backend_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestHeadNormGatedIsTheTwoLaunches holds the norm-gate recipe's numerical
// contract on every backend this host opens: kernels.HeadNormGated is the
// bits of HeadNormRows writing the normalised row and then ActMul (SiLU) or
// SigmoidMul reading it, at every shape, row count and lane width the tier
// launches, with a poisoned guard after the output that must survive. The
// gate is shown to discriminate: the other gate's kernel, and a fused launch
// against a weight off by one ulp, must both differ.
func TestHeadNormGatedIsTheTwoLaunches(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	cases := []struct{ heads, dim, rows int }{
		{16, 128, 1}, // qwen3next's value heads at decode
		{32, 128, 1}, // qwen35's
		{16, 128, 5}, // a prompt chunk's rows
		{8, 64, 1},
		{4, 256, 3},
		{1, 32, 1}, // the narrowest 32-lane width
		{3, 96, 2}, // three passes per lane
	}
	const eps = 1e-6
	ran := 0
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			wide, _ := backend.GuaranteedLanes(d, 32)
			lanes := []int{1}
			if wide {
				lanes = append([]int{32}, lanes...)
			}
			for _, c := range cases {
				for _, l := range lanes {
					for _, gate := range []kernels.NormGate{kernels.GateSiLU, kernels.GateSigmoid} {
						t.Run(name(c.heads, c.dim)+"/r"+itoa32(c.rows)+"/lanes"+itoa32(l)+"/g"+itoa32(int(gate)), func(t *testing.T) {
							normGateCase(t, d, c.heads, c.dim, c.rows, l, eps, gate)
							ran++
						})
					}
				}
			}
		})
	}
	if ran == 0 {
		t.Fatal("no case ran")
	}
}

func normGateCase(t *testing.T, d backend.Device, heads, dim, rows, lanes int, eps float64, gate kernels.NormGate) {
	n := heads * dim * rows
	rng := rand.New(rand.NewSource(int64(n)*31 + int64(gate)))
	in, z, w := make([]float32, n), make([]float32, n), make([]float32, dim)
	for i := range in {
		in[i] = float32(rng.NormFloat64())
		z[i] = float32(3 * rng.NormFloat64())
	}
	for i := range w {
		w[i] = float32(0.5 + rng.Float64())
	}
	// The guard after the output: a NaN pattern the kernels must not touch.
	const guard = 64
	poison := make([]byte, (n+guard)*4)
	for i := 0; i < n+guard; i++ {
		binary.LittleEndian.PutUint32(poison[i*4:], 0x7fc0dead)
	}
	g := newGPU(t, d)
	defer g.free()
	bIn, bW, bZ := g.up(f32bytes(in)), g.up(f32bytes(w)), g.up(f32bytes(z))
	bTmp, bTwo, bOne := g.up(poison), g.up(poison), g.up(poison)

	compile := func(k *ir.Kernel, err error) backend.Kernel {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		c, err := d.Compile(k)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { c.Close() })
		return c
	}
	groups, width := heads*rows, 32
	if lanes == 1 {
		groups, width = (heads*rows+63)/64, 64
	}
	norm := compile(kernels.HeadNormRows(heads, dim, eps, rows, lanes))
	var mul backend.Kernel
	if gate == kernels.GateSiLU {
		mul = compile(kernels.ActMul(n, kernels.ActSiLU))
	} else {
		mul = compile(kernels.SigmoidMul(n))
	}
	fused := compile(kernels.HeadNormGated(heads, dim, eps, rows, lanes, gate))
	if err := norm.Launch(groups, width, bIn, bW, bTmp); err != nil {
		t.Fatal(err)
	}
	if err := mul.Launch((n+127)/128, 128, bZ, bTmp, bTwo); err != nil {
		t.Fatal(err)
	}
	if err := fused.Launch(groups, width, bIn, bW, bZ, bOne); err != nil {
		t.Fatal(err)
	}
	read := func(b backend.Buf) []byte {
		raw := make([]byte, (n+guard)*4)
		if err := b.Read(raw); err != nil {
			t.Fatal(err)
		}
		return raw
	}
	two, one := read(bTwo), read(bOne)
	if !bytes.Equal(one[n*4:], poison[n*4:]) {
		t.Fatal("the fused launch wrote past its output")
	}
	for i := 0; i < n; i++ {
		a, b := binary.LittleEndian.Uint32(two[i*4:]), binary.LittleEndian.Uint32(one[i*4:])
		if a != b {
			t.Fatalf("element %d: two launches %v (%#x), fused %v (%#x)", i,
				math.Float32frombits(a), a, math.Float32frombits(b), b)
		}
		if f := math.Float32frombits(a); f != f || math.IsInf(float64(f), 0) {
			t.Fatalf("element %d is %v: a degenerate reference", i, f)
		}
	}
	// The violations: the other gate, and a scale one ulp off, each differ.
	other := kernels.GateSigmoid
	if gate == kernels.GateSigmoid {
		other = kernels.GateSiLU
	}
	wrong := compile(kernels.HeadNormGated(heads, dim, eps, rows, lanes, other))
	bBad := g.up(poison)
	if err := wrong.Launch(groups, width, bIn, bW, bZ, bBad); err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(read(bBad)[:n*4], two[:n*4]) {
		t.Fatal("the other gate's kernel matched: the comparison cannot see the gate")
	}
	w2 := append([]float32(nil), w...)
	w2[dim-1] = math.Nextafter32(w2[dim-1], 2)
	bW2, bBad2 := g.up(f32bytes(w2)), g.up(poison)
	if err := fused.Launch(groups, width, bIn, bW2, bZ, bBad2); err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(read(bBad2)[:n*4], two[:n*4]) {
		t.Fatal("a scale one ulp off matched: the comparison cannot see the last element of a head")
	}
}

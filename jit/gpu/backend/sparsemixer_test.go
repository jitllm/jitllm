package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// TestSparseMixerWeightsMatchTheOracle holds the device's sparsemixer route --
// ExpertRank's plain logit rank, then ExpertWeights at MoERoute.SparseMixer --
// to internal/oracle.SparseMixer on every backend present, over the expert
// counts either side of a subgroup and the inputs that make the masks matter:
// clusters within the threshold of the top two, a tie for the maximum, a
// negative maximum and zeros. Ids exactly; weights to the device exp's
// relative error. Each defect the reference can carry must also fail here.
func TestSparseMixerWeightsMatchTheOracle(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const eps = 0.01
	type input struct {
		name string
		s    []float32
	}
	inputs := func(n int) []input {
		rng := rand.New(rand.NewSource(int64(n)*31 + 1))
		mk := func(f func(i int) float32) []float32 {
			s := make([]float32, n)
			for i := range s {
				s[i] = f(i)
			}
			return s
		}
		rnd := func(int) float32 { return float32(rng.NormFloat64()) }
		near := mk(rnd)
		for i := 0; i < n; i += 3 {
			near[i] = 5 * (1 - float32(rng.Float64())*0.03)
		}
		behind := mk(func(int) float32 { return float32(rng.NormFloat64()) - 4 })
		behind[n-1] = 3
		for i := 0; i < n-1; i += 2 {
			behind[i] = 1.5 * (1 - float32(rng.Float64())*0.025)
		}
		tie := mk(rnd)
		tie[0], tie[n-1] = 4, 4
		return []input{{"random", mk(rnd)}, {"cluster", near}, {"clusterBehind", behind}, {"tie", tie},
			{"negative", mk(func(int) float32 { return -1 - float32(rng.Float64()) })},
			{"zeros", mk(func(i int) float32 { return -float32(i % 3) })}}
	}
	faults := []oracle.SparseMixerFault{oracle.SparseMixerFaultNoMask, oracle.SparseMixerFaultKeepFirst,
		oracle.SparseMixerFaultRenormalised, oracle.SparseMixerFaultAbsFactor}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			caught := make([]int, len(faults))
			for _, n := range []int{2, 3, 8, 16, 31, 33, 64} {
				r := oldkernels.MoERoute{NExpert: n, K: 2, SparseMixer: eps}
				for _, in := range inputs(n) {
					sel, w := sparseMixerRun(t, d, r, in.s)
					want := oracle.SparseMixer(in.s, eps, oracle.SparseMixerFaultNone)
					if d := sparseMixerDiff(want, sel, w); d != "" {
						t.Errorf("n=%d %s: %s", n, in.name, d)
					}
					for i, f := range faults {
						if sparseMixerDiff(oracle.SparseMixer(in.s, eps, f), sel, w) != "" {
							caught[i]++
						}
					}
				}
			}
			for i, c := range caught {
				if c == 0 {
					t.Errorf("fault %d: the gate cannot see it", faults[i])
				}
			}
		})
	}
}

// sparseMixerRun launches the route on one token and returns the two ids and
// the two weights.
func sparseMixerRun(t *testing.T, d backend.Device, r oldkernels.MoERoute, s []float32) ([2]int32, [2]float32) {
	t.Helper()
	g := newGPU(t, d)
	defer g.free()
	k := r.K
	bL := g.up(f32bytes(s))
	bSel := g.up(make([]byte, (k+1)*4))
	bTop := g.up(make([]byte, (k+1)*4))
	bW := g.up(make([]byte, k*4))
	run := func(mk func() (*ir.Kernel, error), grid, width int, args ...backend.Buf) {
		t.Helper()
		kk, err := mk()
		if err != nil {
			t.Fatal(err)
		}
		c, err := d.Compile(kk)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		if err := c.Launch(grid, width, args...); err != nil {
			t.Fatal(err)
		}
	}
	run(func() (*ir.Kernel, error) { return oldkernels.ExpertRank(r) }, (r.NExpert+127)/128, 128, bL, bSel, bTop)
	run(func() (*ir.Kernel, error) { return oldkernels.ExpertWeights(r) }, 1, 1, bTop, bW, bL)
	selRaw, wRaw := make([]byte, (k+1)*4), make([]byte, k*4)
	if err := bSel.Read(selRaw); err != nil {
		t.Fatal(err)
	}
	if err := bW.Read(wRaw); err != nil {
		t.Fatal(err)
	}
	var sel [2]int32
	var w [2]float32
	for i := 0; i < 2; i++ {
		sel[i] = int32(binary.LittleEndian.Uint32(selRaw[4*i:]))
		w[i] = math.Float32frombits(binary.LittleEndian.Uint32(wRaw[4*i:]))
	}
	return sel, w
}

// sparseMixerDiff compares the device's selection-order ids and weights with
// a reference route. The device's exp is the backend's own (PTX ex2.approx and
// its kin), a few ulp from libm.
func sparseMixerDiff(want oracle.MoERoute, sel [2]int32, w [2]float32) string {
	for i := 0; i < 2; i++ {
		if sel[i] != want.Sel[i] {
			return fmt.Sprintf("ids %v, want %v", sel, want.Sel)
		}
		if d := math.Abs(float64(w[i]-want.Wt[i])) / math.Abs(float64(want.Wt[i])); !(d <= 1e-5) {
			return fmt.Sprintf("weights %v, want %v", w, want.Wt)
		}
	}
	return ""
}

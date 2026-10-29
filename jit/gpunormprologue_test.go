package jit_test

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// normQKV is a row's norm read by q, k and v, each Q4_K, or by q alone
// where KV is 0.
func normQKV(K, Q, KV int) (*graph.Builder, graph.ValueID, []graph.ValueID, map[jlm.Role]jlm.Type) {
	b := graph.New()
	in := b.In("x", K)
	n := b.Norm(in, jlm.RoleAttnNorm, 1e-5)
	outs := []graph.ValueID{b.MatVec(n, jlm.RoleAttnQ, Q)}
	if KV > 0 {
		outs = append(outs, b.MatVec(n, jlm.RoleAttnK, KV), b.MatVec(n, jlm.RoleAttnV, KV))
	}
	b.Out(outs...)
	formats := map[jlm.Role]jlm.Type{jlm.RoleAttnQ: jlm.TypeQ4S, jlm.RoleAttnK: jlm.TypeQ4S, jlm.RoleAttnV: jlm.TypeQ4S, jlm.RoleAttnNorm: jlm.TypeF32}
	return b, in, outs, formats
}

func prefetches(codes []jit.Code) int {
	n := 0
	for _, c := range codes {
		for _, o := range c.GPU.Ops {
			if o.Kind == ir.OpPrefetch {
				n++
			}
		}
	}
	return n
}

// TestComposedGPUNormInProjections forms a row's norm with q, k and v on a
// V100's statement: in their one launch, behind touches of their first
// weight words, where its workgroups run in one wave; its own launch where
// they take several or a thread takes many of its trips. The violation:
// no touches.
func TestComposedGPUNormInProjections(t *testing.T) {
	v100 := jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, ExactDiv: true, ExactArith: true, Regs: 32, Shared: 96 << 10, VectorLoads: true}
	for _, c := range []struct {
		K, Q, KV, want int
	}{{1536, 8960, 0, 1}, {2560, 10240, 0, 1}, {4096, 14336, 0, 2}, {4096, 4096, 1024, 2}, {3072, 3072, 3072, 2}} {
		b, _, _, formats := normQKV(c.K, c.Q, c.KV)
		codes := formWith(t, jit.Default(), b.G, formats, v100)
		if len(codes) != c.want {
			t.Errorf("%d/%d/%d: %d launches, want %d", c.K, c.Q, c.KV, len(codes), c.want)
		}
		if pf := prefetches(codes); c.want == 1 && pf == 0 || c.want != 1 && pf != 0 {
			t.Errorf("%d/%d/%d: %d touches in %d launches", c.K, c.Q, c.KV, pf, len(codes))
		}
		r := jit.Default()
		p := r.Profiles[jit.DeviceGPU]
		p.Passes = slices.Clone(p.Passes)
		for i, ps := range p.Passes {
			if pr, ok := ps.(jit.Prologue); ok {
				pr.Lead = 0
				p.Passes[i] = pr
			}
		}
		r.Profiles[jit.DeviceGPU] = p
		if pf := prefetches(formWith(t, r, b.G, formats, v100)); pf != 0 {
			t.Errorf("%d/%d/%d: %d touches with none asked", c.K, c.Q, c.KV, pf)
		}
	}
}

// TestComposedGPUNormPrologueRuns runs a row's norm in its projection's
// launch on every GPU device present, the projection its launch-apart form's. The
// violation: the norm weight scaled must part.
func TestComposedGPUNormPrologueRuns(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	const K, Q, KV = 2048, 1024, 0
	rng := rand.New(rand.NewSource(5))
	weights := map[jlm.Role][3][]uint32{}
	weights[jlm.RoleAttnQ] = packed(t, quant.Q4_K, kernels.Q4_K, Q, K, rng)
	w := make([]float32, K)
	x := make([]float32, K)
	for i := range w {
		w[i], x[i] = float32(1+0.3*rng.NormFloat64()), float32(rng.NormFloat64())
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			b, in, outs, formats := normQKV(K, Q, KV)
			codes := formWith(t, jit.Default(), b.G, formats, dev)
			apart := formWith(t, without("prologue"), b.G, formats, dev)
			if len(apart) != 2 {
				t.Fatalf("the norm and its projection formed apart as %d launches, want two", len(apart))
			}
			// One launch where the device states its units and holds every
			// workgroup of the projections at once.
			want, g := 2, apart[1].Groups
			if dev.Units > 0 && dev.Slots > 0 && (g+dev.Units-1)/dev.Units*apart[1].GPU.Group[0] < dev.Slots/dev.Units {
				want = 1
			} else if dev.Units > 0 && dev.Slots > 0 {
				t.Fatalf("%d workgroups on %d units: the shape does not run the norm in the projections", g, dev.Units)
			}
			if len(codes) != want {
				t.Fatalf("the norm and its projection formed as %d launches (%d workgroups, %d units): want %d", len(codes), g, dev.Units, want)
			}
			vecs := map[jlm.Role][]float32{jlm.RoleAttnNorm: w}
			run := func(codes []jit.Code, o graph.ValueID, n int) []float64 {
				res, err := runRoles(d, codes, map[graph.ValueID][]float32{in: x}, weights, vecs, o, n)
				if err != nil {
					t.Fatal(err)
				}
				return res
			}
			if e := nmseOf(run(codes, outs[0], Q), run(apart, outs[0], Q)); !(e < 1e-12) {
				t.Errorf("the projection parts from its launch apart: NMSE %.3g", e)
			}
			scaled := slices.Clone(w)
			for i := range scaled {
				scaled[i] *= 1 + 0.5*float32(i%3)
			}
			vecs[jlm.RoleAttnNorm] = scaled
			got := run(codes, outs[0], Q)
			vecs[jlm.RoleAttnNorm] = w
			if e := nmseOf(got, run(apart, outs[0], Q)); !(e > 1e-3) {
				t.Errorf("the norm weight scaled reads NMSE %.3g: the gate cannot see the norm", e)
			}
			t.Logf("%s: the norm and its projection in %d launch(es), %d touches; %d workgroups, %d units", d.API(), len(codes), prefetches(codes), g, dev.Units)
		})
	}
}

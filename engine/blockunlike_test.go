package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
)

// TestABlockUnlikeItsSetRunsOnTheCard places blocks whose plans differ from
// the plan their card's scratch set was built from -- in a scale, a norm, the
// parallel residual, the temperature -- beside that set, and holds each to
// the host. One warm tier: a first instance places block 0 and builds the
// set; a second instance of the model with one feature changed is offered
// every block. Block 0 is the first's (another graph at that index, which it
// does not share); every other block is placed in the first's set and runs
// its own graph, within the clean arm's band of the host under the changed
// config, which itself reads far from the unchanged one.
func TestABlockUnlikeItsSetRunsOnTheCard(t *testing.T) {
	m, g, openOpts := openSynthLlama4(t)
	defer m.Close()
	cfg := *m.Cfg
	decode := func(st *Instance) [][]float32 {
		var out [][]float32
		for _, id := range g.IDs {
			lg, err := st.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	onHost := func(violate func() func()) [][]float32 {
		defer violate()()
		st := testInstance(t, m, len(g.IDs)+1, openOpts...)
		defer st.Close()
		return decode(st)
	}
	cmp := func(want, got [][]float32) (float64, int) {
		worst, flips := 0.0, 0
		for p := range want {
			var num, den float64
			for i := range want[p] {
				d := float64(got[p][i] - want[p][i])
				num, den = num+d*d, den+float64(want[p][i])*float64(want[p][i])
			}
			nmse := num / den
			if math.IsNaN(nmse) || math.IsInf(nmse, 0) || den == 0 {
				return math.Inf(1), len(want)
			}
			worst = math.Max(worst, nmse)
			if Greedy(got[p]) != Greedy(want[p]) {
				flips++
			}
		}
		return worst, flips
	}
	base := onHost(func() func() { return func() {} })
	ran := 0
	edit := func(f func(c *model.Config)) func() func() { return func() func() { return cfgMut(m, f) } }
	for _, v := range []struct {
		name    string
		violate func() func()
	}{
		{"no QK L2 norm", edit(func(c *model.Config) { c.QKL2Norm = false })},
		{"no attention temperature", edit(func(c *model.Config) { c.AttnTempScale = 0 })},
		{"another score scale", edit(func(c *model.Config) { c.AttnScale = 0.5 / math.Sqrt(float64(c.HeadDim)) })},
		{"a residual scale", edit(func(c *model.Config) { c.ResidualScale = 0.5 })},
		{"a parallel block", func() func() { return restated(m, parallelFFN) }},
	} {
		want := onHost(v.violate)
		if d, _ := cmp(base, want); !(d > 1e-3) {
			t.Fatalf("%s: the fixture cannot see it -- the host reads %.3e from the unchanged config", v.name, d)
		}
		for _, spec := range stepDevices() {
			probe, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || probe == nil {
				t.Logf("%s: not present (%v)", spec, err)
				continue
			}
			probe.Close()
			ran++
			func() {
				gpux, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
				if err != nil {
					t.Fatal(err)
				}
				defer gpux.Close()
				first := testInstance(t, m, len(g.IDs)+1, openOpts...)
				defer first.Close()
				first.SetDeviceLayers(gpux, 1)
				if n := first.GPULayers(); n != 1 {
					t.Fatalf("%s on %s: the first instance placed %d blocks: %v %v", v.name, spec, n, first.DeviceDeclines(), gpux.Err())
				}
				defer v.violate()()
				second := testInstance(t, m, len(g.IDs)+1, openOpts...)
				defer second.Close()
				second.SetDeviceLayers(gpux, -1)
				if n := second.GPULayers(); n != cfg.NLayer-1 {
					t.Fatalf("%s on %s: the second instance placed %d of blocks 1..%d beside the first's set: %v %v",
						v.name, spec, n, cfg.NLayer-1, second.DeviceDeclines(), gpux.Err())
				}
				got := decode(second)
				if n := second.GPULayers(); n != cfg.NLayer-1 {
					t.Fatalf("%s on %s: the session fell to the host (%d blocks left): %v", v.name, spec, n, gpux.Err())
				}
				worst, flips := cmp(want, got)
				if !(worst < 1e-2) || flips > 1 {
					t.Fatalf("%s on %s: worst logit NMSE %.3e, %d argmax flips, against the host", v.name, spec, worst, flips)
				}
				far, _ := cmp(base, got)
				t.Logf("%s on %s: blocks 1..%d beside the first's set, worst logit NMSE %.3e, %d flips; %.3e from the unchanged config",
					v.name, spec, cfg.NLayer-1, worst, flips, far)
			}()
		}
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

// parallelFFN has each block's feed-forward read the block's input where it
// reads the attention's residual, as a parallel block's does; the residual
// sums the block's input, its attention and its feed-forward either way.
func parallelFFN(g *graph.Graph) *graph.Graph {
	var in graph.ValueID
	for _, o := range g.Ops {
		if binding(jlm.RoleAttnNorm)(0, &o) {
			in = o.In[0]
			break
		}
	}
	return editing(opsBinding(jlm.RoleFFNNorm), func(o *graph.Op) { o.In[0] = in })(g)
}

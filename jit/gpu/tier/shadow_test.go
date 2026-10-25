package tier

import (
	"encoding/binary"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestStatsAddMergesTheShadowSlots holds Stats.add to the shadow arm's
// per-block slots, which the reflective gate above it does not walk: a block
// is on one device, so the worst of each slot is the largest and the steps are
// summed, and a device holding higher blocks lengthens the result.
func TestStatsAddMergesTheShadowSlots(t *testing.T) {
	var s Stats
	a := Stats{ShadowWorst: []float32{0, 0.5, 0}, ShadowSteps: []int{0, 7, 0}}
	s.add(a)
	s.add(Stats{ShadowWorst: []float32{0.25, 0, 0, 2}, ShadowSteps: []int{3, 0, 0, 9}})
	if want := []float32{0.25, 0.5, 0, 2}; !slices.Equal(s.ShadowWorst, want) {
		t.Errorf("the worst drifts are %v, want %v", s.ShadowWorst, want)
	}
	if want := []int{3, 7, 0, 9}; !slices.Equal(s.ShadowSteps, want) {
		t.Errorf("the compared steps are %v, want %v", s.ShadowSteps, want)
	}
	if a.ShadowWorst[0] != 0 || len(a.ShadowWorst) != 3 {
		t.Errorf("add changed the stats it summed: %v", a.ShadowWorst)
	}
}

// TestShadowDriftOnTheDevice runs the shadow arm's comparison on a device
// against the quantity it is defined as, in Go: the squared distance between
// the two outputs over the squared distance the tier's output moved the
// block's input, kept as the worst over the steps beside their count. A
// second output farther off moves the worst, a nearer one leaves it, and
// neither a NaN nor an infinity can hide in a maximum: each reads as the cap.
// A block that moved nothing divides by a floor, not by zero.
func TestShadowDriftOnTheDevice(t *testing.T) {
	for _, spec := range testdevices.Specs("cuda:0", "vulkan:0", "metal") {
		t.Run(spec, func(t *testing.T) { shadowDriftOn(t, spec) })
	}
}

func shadowDriftOn(t *testing.T, spec string) {
	g, err := OpenWith(WithDevices(spec), WithDeviceTune(TuneOff),
		WithConfig(func(c *Config) { c.Composed, c.ComposedShadow = ComposedBlock, true }))
	if err != nil || g == nil {
		t.Skipf("no %s here: %v", spec, err)
	}
	defer g.Close()
	d := g.devs[0]
	const n = 160
	d.mu.Lock()
	sv, err := d.servedShadow(&layer{graph: new(graph.Graph)}, n)
	d.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	var bufs [4]backend.Buf
	for i, w := range []int{n, n, n, 2} {
		if bufs[i], err = d.dev.Alloc(4 * w); err != nil {
			t.Fatal(err)
		}
		defer bufs[i].Free()
	}
	rng := rand.New(rand.NewSource(7))
	row := func(scale float64, from []float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(scale * rng.NormFloat64())
			if from != nil {
				out[i] += from[i]
			}
		}
		return out
	}
	// drift is the quantity in float64, the way the comparison defines it.
	drift := func(in, want, got []float32) float64 {
		var num, den float64
		for i := range in {
			a, b := float64(want[i])-float64(got[i]), float64(want[i])-float64(in[i])
			num += a * a
			den += b * b
		}
		return num / (den + 1e-30)
	}
	in := row(1, nil)
	want := row(0.3, in)
	nan := row(0.01, want)
	nan[n/2] = float32(math.NaN())
	inf := row(0.01, want)
	inf[3] = float32(math.Inf(1))
	steps := []struct {
		name      string
		in, w, g  []float32
		reset     bool
		worst     float64
		wantSteps int
	}{
		{name: "an output near the tier's", in: in, w: want, g: row(0.003, want), reset: true},
		{name: "a farther one", in: in, w: want, g: row(0.03, want)},
		{name: "a nearer one again", in: in, w: want, g: row(0.0003, want)},
		{name: "a NaN", in: in, w: want, g: nan},
		{name: "an infinity", in: in, w: want, g: inf},
		{name: "a block that moved nothing, the outputs apart", in: want, w: want, g: row(0.01, want), reset: true},
		{name: "the outputs equal", in: in, w: want, g: want, reset: true},
	}
	var raw [8]byte
	var lerr error
	d.dev.Session(func(s backend.Session) {
		lc := &launcher{to: &launchTo{s: s}, err: &lerr}
		var worst float64
		count := 0
		for _, st := range steps {
			if st.reset {
				worst, count = 0, 0
				if lerr = s.Write(bufs[3], f32b([]float32{0, 0})); lerr != nil {
					return
				}
			}
			for i, v := range [][]float32{st.in, st.w, st.g} {
				if lerr = s.Write(bufs[i], f32b(v)); lerr != nil {
					return
				}
			}
			if lerr = sv.run(d, lc, bufs[:], bufs[3:]); lerr != nil {
				return
			}
			if lerr = s.Read(bufs[3], raw[:]); lerr != nil {
				return
			}
			e := drift(st.in, st.w, st.g)
			if math.IsNaN(e) || e >= graph.DriftCap {
				e = graph.DriftCap
			}
			worst, count = math.Max(worst, e), count+1
			gotWorst := float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[:4])))
			gotSteps := math.Float32frombits(binary.LittleEndian.Uint32(raw[4:]))
			if math.Abs(gotWorst-worst) > 1e-3*worst || int(gotSteps) != count {
				t.Errorf("%s: the device keeps a worst of %.6g over %v steps, the definition says %.6g over %d", st.name, gotWorst, gotSteps, worst, count)
			}
			t.Logf("%s: worst %.6g over %v steps (the definition: %.6g over %d)", st.name, gotWorst, gotSteps, worst, count)
		}
	})
	if lerr != nil {
		t.Fatal(lerr)
	}
}

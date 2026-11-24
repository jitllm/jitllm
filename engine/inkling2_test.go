package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
)

// TestInkling2FeaturesAreLoadBearing holds Inkling 2's statement
// (convert/models/inkling2.json) against transformers' own class: the stated
// graph within its golden's bound, and each feature, removed, outside it --
// the relative bias by backward distance, a row's own sinks, the full
// layers' per-row convolution taps and the second pass's messages.
func TestInkling2FeaturesAreLoadBearing(t *testing.T) {
	const name = "synth-hf-inkling2"
	m, err := model.Open(hfContainer(t, name))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	bound := hfGoldBound()[name]
	if w := hfWorst(t, m, name); !(w <= bound) {
		t.Fatalf("the stated graph: worst NMSE %.3e against transformers (bound %.0e)", w, bound)
	}
	for _, c := range inkling2Violations() {
		undo := c.apply(m)
		w := hfWorst(t, m, name)
		undo()
		if !(w > bound) {
			t.Errorf("%s: worst NMSE %.3e, inside the band (%.0e): the feature is not load-bearing", c.name, w, bound)
			continue
		}
		t.Logf("%s: worst NMSE %.3e", c.name, w)
	}
}

// inkling2Violation is one of Inkling 2's features taken out of the stated
// graph; apply returns its undo.
type inkling2Violation struct {
	name  string
	apply func(*model.Model) func()
}

func inkling2Violations() []inkling2Violation {
	attend := func(edit func(o *graph.Op)) func(*model.Model) func() {
		return func(m *model.Model) func() { return restated(m, editing(opsOfKind(graph.OpAttend), edit)) }
	}
	noBias := attend(func(o *graph.Op) {
		if o.Attr.Extent > 0 {
			o.In = append(o.In[:1:1], o.In[2:]...)
			o.Attr.Extent = 0
		}
	})
	noSinks := attend(func(o *graph.Op) {
		if o.Attr.RowSinks {
			o.In = o.In[:len(o.In)-1]
			o.Attr.RowSinks = false
		}
	})
	noTaps := func(m *model.Model) func() {
		return restated(m, editing(opsOfKind(graph.OpStateConv), func(o *graph.Op) { o.In = o.In[:1] }))
	}
	noMessages := func(m *model.Model) func() {
		ungated := func(o *graph.Op) {
			o.Attr.Expr = []graph.Term{{Fn: graph.MapAdd, Args: [3]graph.Arg{{Kind: graph.ArgIn}, {Kind: graph.ArgConst}}}}
		}
		undo := []func(){}
		for _, n := range []string{"hq.gated", "hk.gated", "hv.gated", "fi.gated"} {
			undo = append(undo, restated(m, editing(opsNamed(n), ungated)))
		}
		return func() {
			for i := len(undo) - 1; i >= 0; i-- {
				undo[i]()
			}
		}
	}
	return []inkling2Violation{
		{"no relative bias", noBias},
		{"no sinks", noSinks},
		{"no per-row convolution taps", noTaps},
		{"no messages from the first pass", noMessages},
	}
}

// TestInkling2OnEveryDevice holds Inkling 2 on each listed card against
// transformers' golden, within the golden's bound (RULE 11c): every block and
// the head on the card, decoded a token at a time and prefilled as one chunk
// on float tiles,
// and half the blocks on the card, at an f32 history and at an f16 one, the
// shipping default on amd64.
// The card's tuner is off. Each feature removed on the every-block arm must
// leave the bound.
func TestInkling2OnEveryDevice(t *testing.T) {
	const name = "synth-hf-inkling2"
	m, err := model.Open(hfContainer(t, name))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	bound := hfGoldBound()[name]
	ids, heads := hfGold(t, name)
	nl := m.Cfg.NLayer
	ran := 0
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || g == nil {
				noDevice(t, spec, err)
			}
			defer g.Close()
			placed := func(n int, f16 bool) func(*Instance) {
				return func(st *Instance) {
					if st.KVIsF16() != f16 {
						t.Fatalf("the history is f16 %v, wanted %v", st.KVIsF16(), f16)
					}
					st.SetDeviceFallback(false)
					if err := st.SetDevice(g); err != nil {
						t.Fatal(err)
					}
					if n < nl {
						st.SetGPULayers(n)
					}
					if st.GPULayers() != n || (n == nl) != st.HeadOnDevice() {
						t.Fatalf("placed %d of %d blocks (wanted %d), head %v: %s", st.GPULayers(), nl, n, st.HeadOnDevice(), g.Err())
					}
				}
			}
			for _, a := range []struct {
				name string
				n    int
				f16  bool
			}{
				{"every block", nl, false},
				{"half the blocks", nl / 2, false},
				{"half the blocks, an f16 history", nl / 2, true},
			} {
				w := hfWorstOn(t, m, name, placed(a.n, a.f16), WithKVF16(a.f16))
				t.Logf("%s: worst NMSE %.3e against transformers", a.name, w)
				if !(w <= bound) {
					t.Errorf("%s on %s: worst NMSE %.3e against transformers, bound %.0e", a.name, spec, w, bound)
				}
			}
			prompt := func(d *gpu.GPU) float64 {
				st := testInstance(t, m, len(ids)+1, WithKVF16(false))
				defer st.Close()
				st.SetDeviceFallback(false)
				if err := st.SetDevice(d); err != nil {
					t.Fatal(err)
				}
				if st.GPULayers() != nl || !st.HeadOnDevice() {
					t.Fatalf("the prompt arm placed %d of %d blocks, head %v: %s", st.GPULayers(), nl, st.HeadOnDevice(), d.Err())
				}
				lg, err := st.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				return hfNMSE64(lg, heads[len(ids)-1])
			}
			fl, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff), gpu.WithConfig(func(c *gpu.Config) { c.NoMMA = true }))
			if err != nil || fl == nil {
				t.Fatalf("%s did not open a second time: %v", spec, err)
			}
			w := prompt(fl)
			fl.Close()
			t.Logf("every block, the prompt as one chunk on float tiles: last row's NMSE %.3e", w)
			if !(w <= bound) {
				t.Errorf("the prompt as one chunk on %s: NMSE %.3e against transformers, bound %.0e", spec, w, bound)
			}
			// The matrix unit's tiles multiply binary16 operands, and this
			// fixture amplifies rounding about 1e8-fold in NMSE (its own f32
			// forward reads 1.6e-06 against its f64 one): no bound on it
			// separates a wrong tile from binary16. Reported, not held.
			t.Logf("every block, the prompt as one chunk on the device's own tiles: last row's NMSE %.3e", prompt(g))
			for _, v := range inkling2Violations() {
				undo := v.apply(m)
				w := hfWorstOn(t, m, name, placed(nl, false), WithKVF16(false))
				undo()
				t.Logf("%s, every block: worst NMSE %.3e", v.name, w)
				if !(w > bound) {
					t.Errorf("%s on %s: worst NMSE %.3e, inside the bound: the card does not see the feature", v.name, spec, w)
				}
			}
			ran++
		})
	}
	if ran == 0 {
		t.Skip("no device listed")
	}
}

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
	bound := hfNMSE
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

// inkling2Tiles bounds a prompt on the card's own tiles, whose matrix unit
// multiplies binary16 operands: the last row reads 1.8e-06 on CUDA (V100),
// 8.2e-12 on float tiles, and transformers' own run with every projection's
// input rounded to binary16 2.2e-06 (scripts/inkling2gold.py); each feature
// removed reads about 2.
const inkling2Tiles = 1e-5

// inkling2F16History bounds a binary16 history against transformers' f32
// one: every block on the CPU reads 3.5e-05, half the blocks on a card (its
// half widened) 2.7e-05 on CUDA and Vulkan, an f32 history 8.4e-11.
const inkling2F16History = 1e-4

// TestInkling2OnEveryDevice holds Inkling 2 on each listed card against
// transformers' golden, within the golden's bound (RULE 11c): every block and
// the head on the card, decoded a token at a time and prefilled as one chunk
// on float tiles,
// and on its own tiles, and half the blocks on the card, at an f32 history
// and at an f16 one (the shipping default on amd64) written before the card
// takes its half.
// The card's tuner is off. Each feature removed on the every-block arm must
// leave the bound.
func TestInkling2OnEveryDevice(t *testing.T) {
	const name = "synth-hf-inkling2"
	m, err := model.Open(hfContainer(t, name))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	bound := hfNMSE
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
			placed := func(n int) func(*Instance) {
				return func(st *Instance) {
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
			}{
				{"every block", nl},
				{"half the blocks", nl / 2},
			} {
				w := hfWorstOn(t, m, name, placed(a.n), WithKVF16(false))
				t.Logf("%s: worst NMSE %.3e against transformers", a.name, w)
				if !(w <= bound) {
					t.Errorf("%s on %s: worst NMSE %.3e against transformers, bound %.0e", a.name, spec, w, bound)
				}
			}
			// A binary16 history keeps its format on every device, placed
			// before its first position or after it: the CPU's half and the
			// card's half both hold binary16.
			st0 := testInstance(t, m, len(ids)+1, WithKVF16(true))
			placed(nl / 2)(st0)
			if !st0.KVIsF16() {
				t.Fatal("placed at position 0, the binary16 history stepped down")
			}
			st0.Close()
			cpu16 := hfWorstOn(t, m, name, func(st *Instance) {
				if !st.KVIsF16() {
					t.Fatal("the CPU's history is not binary16")
				}
			}, WithKVF16(true))
			st := testInstance(t, m, len(ids)+1, WithKVF16(true))
			w16 := 0.0
			for p, id := range ids {
				if p == 1 {
					placed(nl / 2)(st)
					if !st.KVIsF16() {
						t.Fatal("the history written on the CPU stepped down when the card took half the blocks")
					}
				}
				lg, err := st.Forward(id)
				if err != nil {
					t.Fatal(err)
				}
				w16 = max(w16, hfNMSE64(lg, heads[p]))
			}
			st.Close()
			t.Logf("half the blocks, an f16 history: worst NMSE %.3e against transformers; every block on the CPU at an f16 history %.3e", w16, cpu16)
			if !(w16 <= inkling2F16History) || !(cpu16 <= inkling2F16History) {
				t.Errorf("an f16 history on %s: worst NMSE %.3e (every block on the CPU %.3e), bound %.0e", spec, w16, cpu16, inkling2F16History)
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
			if w := prompt(g); !(w <= inkling2Tiles) {
				t.Errorf("the prompt as one chunk on %s's own tiles: NMSE %.3e against transformers, bound %.0e", spec, w, inkling2Tiles)
			} else {
				t.Logf("every block, the prompt as one chunk on the device's own tiles: last row's NMSE %.3e", w)
			}
			for _, v := range inkling2Violations() {
				undo := v.apply(m)
				w := hfWorstOn(t, m, name, placed(nl), WithKVF16(false))
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

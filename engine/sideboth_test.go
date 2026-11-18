package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestTapsBesidePerLayerInputs runs a model of per-layer embeddings given two
// tower taps as well, both its blocks' side input: one side row a token, the
// tap rows first and the per-layer inputs after them. A text prompt's tap
// rows are zeros, so its logits are the plain model's bit for bit, on the
// CPU and with every block on each card -- which they are only where every
// reader finds its own part of the row: a per-layer input read where the
// taps are reads zeros and moves them.
func TestTapsBesidePerLayerInputs(t *testing.T) {
	const taps, gen = 2, 6
	ran := 0
	for _, name := range []string{"synth-gemma3n.gguf", "synth-gemma4-e.gguf"} {
		t.Run(name, func(t *testing.T) {
			src, ok := existingModel(testmodels.Path(name))
			if !ok {
				testmodels.Missing(t, "MODEL MISSING: %s", testmodels.Path(name))
			}
			path := jlmOf(t, src)
			run := func(withTaps bool, g *gpu.GPU) [][]float32 {
				m, err := model.Open(path)
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				if m.Cfg.PLEDim == 0 {
					t.Fatalf("%s has no per-layer embeddings", name)
				}
				if withTaps {
					m.Cfg.DeepTaps = taps
					defer restated(m, tapsFirst(taps*m.Cfg.NEmbd))()
				defer rowsTapsFirst(m, taps*m.Cfg.NEmbd)()
					if w := m.Cfg.SideWidth(); w != taps*m.Cfg.NEmbd+m.Cfg.PleWidth() {
						t.Fatalf("a side row of %d, want the taps' %d and the per-layer inputs' %d", w,
							taps*m.Cfg.NEmbd, m.Cfg.PleWidth())
					}
				}
				ids := m.Vocab.Encode("The capital of France is", true)
				s := testInstance(t, m, len(ids)+gen+1, WithKVF16(false))
				defer s.Close()
				if g != nil {
					if err := s.SetDevice(g); err != nil {
						t.Fatal(err)
					}
					if s.GPULayers() != m.Cfg.NLayer {
						t.Fatalf("the card took %d of %d blocks: %s", s.GPULayers(), m.Cfg.NLayer, g.Err())
					}
				}
				lg, err := s.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				out := [][]float32{append([]float32(nil), lg...)}
				for range gen {
					if lg, err = s.Forward(Greedy(lg)); err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), lg...))
				}
				return out
			}
			// The CPU's forms are the same with taps and without, bit for bit;
			// a card's wider side row may take another reduction order, the
			// prompt's gather on the card among them (1.1e-8 on synth-gemma3n),
			// held to a band a per-layer input read where the taps are -- a
			// read of zeros -- would leave.
			same := func(arm string, want, got [][]float32, band float64) {
				for i := range want {
					if e := logitNMSE(got[i], want[i]); e > band || band == 0 && e != 0 {
						t.Fatalf("%s: step %d parts by NMSE %.3e with taps", arm, i, e)
					}
					for j := range want[i] {
						if band == 0 && want[i][j] != got[i][j] {
							t.Fatalf("%s: step %d logit %d is %v with taps, %v without", arm, i, j, got[i][j], want[i][j])
						}
					}
				}
				ran++
			}
			same("cpu", run(false, nil), run(true, nil), 0)
			for _, spec := range stepDevices() {
				t.Run(spec, func(t *testing.T) {
					g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
					if err != nil || g == nil {
						noDevice(t, spec, err)
					}
					defer g.Close()
					same(spec, run(false, g), run(true, g), 1e-7)
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no arm ran")
	}
}

// tapsFirst states a block's side row with n elements of tap rows before its
// per-layer inputs, as the converter states the blocks of a model whose
// tower taps them: the side input n wider, each slice of it n further on.
func tapsFirst(n int) func(*graph.Graph) *graph.Graph {
	return func(g *graph.Graph) *graph.Graph {
		side := -1
		for _, v := range g.In {
			if isNamed(g, v, "ple") {
				side = int(v)
			}
		}
		if side < 0 {
			return g
		}
		out := cloned(g)
		out.Values[side].Width += n
		for i := range out.Ops {
			if o := &out.Ops[i]; o.Kind == graph.OpSlice && len(o.In) > 0 && int(o.In[0]) == side {
				o.Attr.Offset += n
			}
		}
		return out
	}
}

// rowsTapsFirst states the input rows' side row, and a given row's, with n
// zeros before the per-layer inputs, until the returned undo: front.json's
// taps.zero and side, which the converter states for a model whose tower
// taps its blocks. Without it the rows write the per-layer inputs where the
// taps are.
func rowsTapsFirst(m *model.Model, n int) func() {
	s := m.FrontGraphs()
	set := *s
	for _, g := range []**graph.Graph{&set.CpuEmbed, &set.Embed, &set.Given} {
		if *g != nil {
			*g = sideTapsFirst(*g, n)
		}
	}
	m.Front.Mu.Lock()
	was := m.Front.Set
	m.Front.Set = &set
	m.Front.Mu.Unlock()
	return func() {
		m.Front.Mu.Lock()
		m.Front.Set = was
		m.Front.Mu.Unlock()
	}
}

// sideTapsFirst is g with its side output, the per-layer inputs, after n
// zeros of the row's own.
func sideTapsFirst(g *graph.Graph, n int) *graph.Graph {
	last := len(g.Out) - 1
	if last < 1 || !isNamed(g, g.Out[last], "ple.in") {
		return g
	}
	out := cloned(g)
	pl := out.Out[last]
	tz, side := graph.ValueID(len(out.Values)), graph.ValueID(len(out.Values)+1)
	out.Values = append(out.Values, graph.Value{Name: "taps.zero", Width: n}, graph.Value{Name: "side", Width: n + out.Values[pl].Width})
	out.Ops = append(out.Ops,
		graph.Op{Kind: graph.OpMap, In: []graph.ValueID{out.Out[0]}, Out: []graph.ValueID{tz}, Attr: graph.Attrs{Fn: graph.MapMul}},
		graph.Op{Kind: graph.OpConcat, In: []graph.ValueID{tz, pl}, Out: []graph.ValueID{side}})
	out.Out[last] = side
	return out
}

package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestRefusedInputRowsStayOnTheCPU holds the front end's refusals of the
// input rows to every device, which is what lets the State promise a device
// the rows by the front end's graph alone (embedOnDevice, no family test): on
// a model with each refused feature -- Gemma 3n's four AltUp streams beside
// its per-layer embeddings, DeepSeek V4's and Qwen3.8-Flash-Next's
// hyper-connection streams, and Gemma 4's per-layer embeddings on a single
// stream -- the front end states no input rows by its named reason, the head
// asks no device to gather them (nn.Head.GatherRow), the composer is asked
// for none, and a decode with the whole step composed stays within the
// device band of the CPU's. With the rows' plain graph stated in place of the
// refusal (the violation), a single-stream model's rows are promised without
// their per-layer inputs and the decode parts from the CPU's, and a
// stream-major residual's are declined by name, every one.
func TestRefusedInputRowsStayOnTheCPU(t *testing.T) {
	const bound = 1e-5
	cases := []struct {
		name string
		open func(*testing.T, string) (*Model, *c6Golden)
		why  Uncovered
	}{
		{"synth-gemma3n", openGemma3n, UncoveredStreams},
		{"synth-deepseek4", openDS4, UncoveredStreams},
		{"synth-qwen4exp", openQ4E, UncoveredStreams},
		{"synth-gemma4-e", openGemma4, UncoveredPLE},
	}
	ran := 0
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, g := tc.open(t, tc.name)
			defer m.Close()
			f := m.frontGraphs()
			if f.embed != nil || f.embedWhy != tc.why {
				t.Fatalf("the front end states the input rows as %v (%s), where %s refuses them", f.embed != nil, f.embedWhy, tc.why)
			}
			host := m.NewState(len(g.IDs) + 1)
			var want [][]float32
			for _, id := range g.IDs {
				l, err := host.Forward(id)
				if err != nil {
					t.Fatal(err)
				}
				want = append(want, append([]float32(nil), l...))
			}
			host.Close()
			prompt, toks := g.IDs[:1], g.IDs[1:]
			decode := func(t *testing.T, dev string) (worst float64, st tier.Stats, gather bool) {
				var got [][]float32
				got, st, _, _ = composedDecodeWith(t, m, dev, prompt, toks, tier.ComposedStep, true, false, func(s *State) {
					gather = s.head != nil && s.head.GatherRow
				})
				for i := range got {
					e := logitNMSE(got[i], want[i])
					if math.IsNaN(e) {
						t.Fatalf("pos %d: the logits are not finite", i)
					}
					worst = max(worst, e)
				}
				return worst, st, gather
			}
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					worst, st, gather := decode(t, dev)
					ne, de, nu := st.ServedPart(tier.ComposedEmbed)
					np, dp, _ := st.ServedPart(tier.PromptEmbed)
					switch {
					case gather:
						t.Fatalf("the head asks the device to gather input rows the front end refuses")
					case ne+de+nu+np+dp != 0:
						t.Fatalf("the composer was asked for refused input rows: %d served, %d declined, the prompt's %d and %d", ne, de, np, dp)
					case !(worst < bound):
						t.Fatalf("worst logit NMSE %.3e against the CPU, bound %.0e", worst, bound)
					}
					t.Logf("%s: the rows filled by the CPU, worst logit NMSE %.3e", dev, worst)
					// The violation: the token's row alone stated as the input
					// rows.
					f.embed = plainEmbed(m)
					defer func() { f.embed = nil }()
					bad, bst, gather := decode(t, dev)
					ne, de, _ = bst.ServedPart(tier.ComposedEmbed)
					switch {
					case !gather:
						t.Fatalf("with the rows' plain graph stated the head asks no device to gather them")
					case m.Cfg.Streams() > 1 && (ne != 0 || de != len(toks)):
						t.Fatalf("with the plain graph over %d streams the composer gathered %d rows and declined %d of %d (%s)",
							m.Cfg.Streams(), ne, de, len(toks), declinesOf(bst))
					case m.Cfg.Streams() == 1 && (!(bad > 100*worst) || !(bad > 1e-4)):
						t.Fatalf("with the plain graph the composer gathered %d of %d rows (%d declined, %d replayed: %s) and the decode is %.3e from the CPU (clean %.3e): the gate cannot see it",
							ne, len(toks), de, bst.ReplayedPart(tier.ComposedEmbed), declinesOf(bst), bad, worst)
					}
					t.Logf("violation, the plain graph stated: %d rows gathered, %d declined (%s), worst NMSE %.3e", ne, de, declinesOf(bst), bad)
					ran++
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no device ran")
	}
}

// plainEmbed is the input rows the front end states for a model with none
// of its refusals: the token's row, scaled.
func plainEmbed(m *Model) *graph.Graph {
	c := m.Cfg
	b := graph.NewBlock(-1)
	x := b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "embd", Width: c.NEmbd}},
		[]graph.Binding{{Role: jlm.RoleTokenEmbd, K: c.NEmbd, Rows: c.NVocab}}, graph.Attrs{}, nil, nil)[0]
	if s := c.EmbdScale; s != 0 && s != 1 {
		x = b.Emit(graph.OpMap, []graph.ValueID{x}, []graph.Result{{Name: "scaled", Width: c.NEmbd}}, nil,
			graph.Attrs{Fn: graph.MapMul, Const: s}, nil, nil)[0]
	}
	b.Out(x)
	return b.G
}

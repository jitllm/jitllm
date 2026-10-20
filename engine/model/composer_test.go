package model

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle/graphref"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestComposerMatchesTheGolden runs every fixture's blocks on the device as
// composed programs and holds each graph op, on the values the device fed it,
// to the graph's golden (graphref: each op by its definition): the composer
// generates kernels for any graph, and this says, per model, the first op
// whose kernels do not compute what the op states. Ops that read a matrix are
// the matvec kernels' own gates' and are passed over; every other op -- the
// primitives a composite decomposes into, norms, activations, adds -- is
// compared, at every row of a prompt chunk and of decode steps.
//
// VIOLATION SIGNATURE. Break a primitive's generator or the builder's wiring
// of one (a value in the wrong slot, a layout read wrong) and this names the
// op and the block.
func TestComposerMatchesTheGolden(t *testing.T) {
	for _, name := range []string{"stories15M-q8_0", "synth-qwen2", "synth-glm4", "synth-olmo2", "tiny-qwen3moe-f32",
		"synth-qwen2moe", "synth-mamba2", "synth-lfm2", "synth-granitehybrid", "synth-falconmamba", "synth-falcon", "synth-phi2",
		"synth-starcoder", "synth-starcoder2", "synth-stablelm", "synth-commandr", "synth-nemotron", "synth-deepseek",
		"synth-deepseek-lite", "synth-deepseek-dense", "synth-glm4moelite", "synth-llama4", "synth-smollm3", "synth-hunyuan",
		"synth-cohere2", "synth-exaone4", "synth-mistral3", "synth-qwen35-hybrid-mtp", "synth-qwen35moe-hybrid-mtp",
		"synth-olmo3", "synth-gemma4", "synth-gemma4-e", "synth-gemma4-kv", "synth-gemma4-wide", "synth-gemma4-moe",
		"synth-gemma3n", "synth-minimaxm3", "synth-qwen4exp", "synth-kimik3", "synth-deepseek32", "synth-deepseek4",
		"synth-dbrx", "synth-bitnet-f16", "synth-falconh1", "synth-falconh1-nonorm"} {
		t.Run(name, func(t *testing.T) {
			m, err := Open(graphModel(t, name), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			s := m.NewState(64)
			defer s.Close()
			checked, first := 0, ""
			trace := func(li int, g *graph.Graph, ops []int, vals map[graph.ValueID][]float32, rows int) {
				if first != "" {
					return
				}
				w := blockWeights(t, m, s, li)
				for _, gi := range ops {
					o := g.Ops[gi]
					if readsBank(o) {
						return
					}
					for _, b := range o.Weight {
						if _, ok := w[b.Role]; !ok {
							return // a weight the host does not hold in f32 here
						}
					}
				}
				for r := 0; r < rows; r++ {
					// One row of every value the ops read and wrote.
					in := map[graph.ValueID][]float32{}
					for id, v := range vals {
						n := g.Values[id].Width
						if len(v) >= (r+1)*n {
							in[id] = v[r*n : (r+1)*n]
						}
					}
					sub := &graph.Graph{Version: g.Version, Values: g.Values}
					for _, gi := range ops {
						sub.Ops = append(sub.Ops, g.Ops[gi])
					}
					inputs := map[graph.ValueID][]float32{}
					for _, o := range sub.Ops {
						for _, id := range o.In {
							if x, ok := in[id]; ok {
								inputs[id] = x
							}
						}
					}
					ref, err := graphref.EvalGraph(sub, inputs, w)
					if err != nil {
						return // an op the golden does not evaluate (a stateful one)
					}
					for _, o := range sub.Ops {
						for _, id := range o.Out {
							dev, gold := in[id], ref[id]
							if len(dev) != len(gold) || len(gold) == 0 {
								continue
							}
							checked++
							scale := 1.0
							for _, x := range gold {
								scale = math.Max(scale, math.Abs(float64(x)))
							}
							for i := range gold {
								if d := math.Abs(float64(dev[i] - gold[i])); d > 1e-3*scale || math.IsNaN(float64(dev[i])) {
									first = fmt.Sprintf("block %d, row %d: %s -> %%%s[%d] is %g, the golden's %g (ops %v)",
										li, r, o.Kind, g.Values[id].Name, i, dev[i], gold[i], ops)
									return
								}
							}
						}
					}
				}
			}
			opts := append([]tier.Option{tier.WithDevices("cuda"),
				tier.WithConfig(func(c *tier.Config) {
					c.NoGraph = true
					c.OpTrace = trace
				})}, testTierOpts(t)...)
			g, err := openTier(t, opts...)
			if err != nil || g == nil {
				noDevice(t, "cuda", err)
			}
			defer g.Close()
			if err := s.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if s.GPULayers() != m.Cfg.NLayer {
				t.Skipf("the device took %d of %d blocks: %v", s.GPULayers(), m.Cfg.NLayer, s.DeviceDeclines())
			}
			lg, err := s.Prefill(m.Vocab.Encode("Once upon a time", true))
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < 3 && first == ""; i++ {
				if lg, err = s.Forward(Greedy(slices.Clone(lg))); err != nil {
					t.Fatal(err)
				}
			}
			if first != "" {
				t.Fatalf("the composer parts from the golden: %s", first)
			}
			if checked == 0 {
				t.Fatalf("no op was compared: the trace reached nothing")
			}
			t.Logf("%d op outputs held to the golden", checked)
		})
	}
}

// readsBank is an op bound to a bank of experts: the banked kernels' own
// gates hold those.
func readsBank(o graph.Op) bool {
	for _, b := range o.Weight {
		if b.Sheets > 0 {
			return true
		}
	}
	return false
}

// blockWeights is block li's vectors and matrices by role, in f32: a
// matrix's quantized rows decoded (quant.Dequant32), rows of K.
func blockWeights(t *testing.T, m *Model, s *State, li int) map[jlm.Role][]float32 {
	w := map[jlm.Role][]float32{}
	for _, v := range s.vectors(&m.layers[li]) {
		w[v.Role] = v.V
	}
	var buf [maxMatrices]*tensor
	for _, tn := range m.layers[li].appendMatrices(buf[:0]) {
		if tn.e == nil || len(tn.data) == 0 || tn.e.Role == jlm.RoleNone {
			continue
		}
		out := make([]float32, tn.rows*tn.k)
		if err := quant.Dequant32(tn.typ, tn.data, out); err != nil {
			continue
		}
		w[tn.e.Role] = out
	}
	return w
}

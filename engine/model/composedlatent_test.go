package model

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// latentRewrite is a violation of a latent attention block's statement,
// written over a copy of its graph: false where the graph has nothing it
// rewrites.
type latentRewrite struct {
	what    string
	rewrite func(g *graph.Graph) bool
}

// latentRewrites drop the latent row's norm, turn the rotated parts one pair
// short, swap the absorb's bank with the un-absorb's (where the two share a
// shape), and read each head's value as the last KVLoraRank of the key's
// width rather than its latent prefix.
func latentRewrites() []latentRewrite {
	return []latentRewrite{
		{"the latent row's norm dropped", func(g *graph.Graph) bool {
			for i, o := range g.Ops {
				if o.Kind == graph.OpNorm && o.Weight[0].Role == jlm.RoleAttnKVANorm {
					g.Ops[i] = graph.Op{Kind: graph.OpSlice, In: o.In, Out: o.Out, Attr: graph.Attrs{Width: g.Values[o.Out[0]].Width}}
					return true
				}
			}
			return false
		}},
		{"the rotated parts a pair short", func(g *graph.Graph) bool {
			n := 0
			for i, o := range g.Ops {
				if o.Kind == graph.OpRope {
					g.Ops[i].Attr.NRot -= 2
					n++
				}
			}
			return n == 2
		}},
		{"the absorb's and the un-absorb's banks swapped", func(g *graph.Graph) bool {
			var kb, vb *graph.Binding
			for i, o := range g.Ops {
				if o.Kind != graph.OpExpertMatVec {
					continue
				}
				switch w := &g.Ops[i].Weight[0]; w.Role {
				case jlm.RoleAttnKB:
					kb = w
				case jlm.RoleAttnVB:
					vb = w
				}
			}
			if kb == nil || vb == nil || kb.K != vb.K || kb.Rows != vb.Rows {
				return false
			}
			kb.Role, vb.Role = vb.Role, kb.Role
			return true
		}},
		{"the value read at the key's width, its last KVLoraRank", func(g *graph.Graph) bool {
			for i, o := range g.Ops {
				if o.Kind == graph.OpSlice && len(o.In) == 1 && slices.ContainsFunc(g.Ops, func(a graph.Op) bool {
					return a.Kind == graph.OpAttend && a.Out[0] == o.In[0]
				}) {
					g.Ops[i].Attr.Offset = o.Attr.HeadDim - o.Attr.Width
					return true
				}
			}
			return false
		}},
	}
}

// latentGateModels are the models the latent attention gate decodes: the
// fixture whose every block attends through a latent, which the violations
// run on, the same fixture stated with no position encoding (the rotary
// part's key and query unturned, on the tier and the composer alike), the
// YaRN fixtures, and a real checkpoint whose first block is a dense latent
// block and the rest mixtures, which the front end refuses by name and the
// tier runs.
func latentGateModels() []struct {
	file          string
	violate, nope bool
} {
	return []struct {
		file          string
		violate, nope bool
	}{{"synth-deepseek-dense", true, false}, {"synth-deepseek-dense", false, true}, {"synth-deepseek-yarn-af.gguf", false, false},
		{"DeepSeek-V2-Lite.Q4_K_M.gguf", false, false}}
}

// TestComposedLatentViolations holds the composed latent attention blocks to
// the float reference as TestComposedBlocksServe holds every arm, every block
// placed, the composer serving every latent block the front end states at
// every step and declining no other block but for no graph; then, on the
// fixture, runs each latentRewrite over every block's statement and requires
// the arm to leave the band: the gate sees a dropped latent norm, a rotation
// a pair short and a value read at the wrong place in the key's width.
func TestComposedLatentViolations(t *testing.T) {
	ran := 0
	for _, lm := range latentGateModels() {
		name := lm.file
		if lm.nope {
			name += "/nope"
		}
		t.Run(name, func(t *testing.T) {
			m := frontModel(t, lm.file)
			if lm.nope {
				m.Cfg.NoPosEnc = true
				m.front.mu.Lock()
				m.front.set = nil
				m.front.mu.Unlock()
				for li := range m.Cfg.NLayer {
					for _, o := range m.Composing().BlockGraph(li).Ops {
						if o.Kind == graph.OpRope {
							t.Fatalf("block %d turns by the rotary with no position encoding stated", li)
						}
					}
				}
			}
			covered := 0
			for li := range m.Cfg.NLayer {
				why := m.Uncovered(li)
				switch {
				case why == Covered && m.stores(li, jlm.RoleAttnKVA):
					covered++
				case why == Covered || !namedReason(why):
					t.Fatalf("block %d is neither a latent block the front end states nor one it refuses by name (%s %v)", li, why, m.UncoveredErr(li))
				case lm.violate:
					t.Fatalf("block %d of the fixture is refused (%s): the violations would not reach it", li, why)
				}
			}
			if covered == 0 {
				t.Fatalf("no block attends through a latent the front end states: this gate proves nothing")
			}
			ids := []int32{5, 11, 23, 41, 67, 89, 101, 127}
			toks := make([]int32, 48)
			for i := range toks {
				toks[i] = int32((int(ids[i%len(ids)])*7 + i) % m.Vocab.Size())
			}
			parts := gpu.ComposedBlock
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					want, _, _ := composedDecode(t, m, dev, ids, toks, 0, false, false)
					ref, _, _ := composedDecode(t, m, dev, ids, toks, gpu.ComposedBlock|gpu.ComposedHead, false, true)
					widest := 0.0
					for i := 1; i < len(ref); i++ {
						widest = max(widest, logitNMSE(want[i], ref[i]))
					}
					band := func(got [][]float32) (med, worst float64) {
						var ratios []float64
						for i := 1; i < len(ref); i++ {
							et, eg := logitNMSE(want[i], ref[i]), logitNMSE(got[i], ref[i])
							if math.IsNaN(et) || math.IsNaN(eg) || !(et > 0) {
								t.Fatalf("step %d: the float reference is %.3g from the tier and %.3g from the arm: a degenerate reference proves nothing", i, et, eg)
							}
							ratios = append(ratios, eg/et)
							worst = max(worst, eg)
						}
						slices.Sort(ratios)
						return ratios[len(ratios)/2], worst
					}
					got, st, _ := composedDecode(t, m, dev, ids, toks, parts, false, false)
					// Every step runs every latent block, emitted or replayed in a
					// recording, and declines the rest for no graph alone.
					n, _ := st.ServedPart(parts)
					n += st.ReplayedPart(parts)
					declined := 0
					for i, k := range st.BlockDeclines {
						if k > 0 && gpu.DeclineName(i) != composedNoGraph {
							t.Fatalf("the composer declined %d block step(s): %s", k, gpu.DeclineName(i))
						}
						declined += k
					}
					if n != len(toks)*covered || declined > len(toks)*(m.Cfg.NLayer-covered) {
						t.Fatalf("the composer ran %d block steps and declined %d, where %d steps of the %d blocks it states are %d (%s)",
							n, declined, len(toks), covered, len(toks)*covered, declinesOf(st))
					}
					med, worst := band(got)
					if !(med <= bandRatio) || !(worst <= bandWorst*widest) {
						t.Fatalf("the composed blocks are %.3g at the median of the tier's distance and %.3g at their worst, where the tier reaches %.3g (bounds %.1f and %.1fx)",
							med, worst, widest, bandRatio, bandWorst)
					}
					t.Logf("composed latent attention, %d of %d blocks: median ratio %.3f, worst %.3g against the tier's %.3g", covered, m.Cfg.NLayer, med, worst, widest)
					if !lm.violate {
						ran++
						return
					}
					for _, rw := range latentRewrites() {
						var undo []func()
						for li := range m.Cfg.NLayer {
							u := rewriteBlock(t, m, li, rw.rewrite)
							if u == nil {
								t.Fatalf("%s: block %d has nothing to rewrite", rw.what, li)
							}
							undo = append(undo, u)
						}
						bad, bst, _ := composedDecode(t, m, dev, ids, toks, parts, false, false)
						for _, u := range undo {
							u()
						}
						if n, d := bst.ServedPart(parts); n == 0 || d != 0 {
							t.Fatalf("with %s the composer served %d block steps and declined %d (%s; %q)", rw.what, n, d, declinesOf(bst), bst.LastErr)
						}
						bmed, bworst := band(bad)
						if bmed <= bandRatio && bworst <= bandWorst*widest {
							t.Fatalf("with %s the arm is %.3g at the median of the tier's distance and %.3g at its worst, inside the band (%.1f and %.1fx of %.3g): the gate cannot see it",
								rw.what, bmed, bworst, bandRatio, bandWorst, widest)
						}
						t.Logf("with %s: median ratio %.3g, worst %.3g against the band's %.3g", rw.what, bmed, bworst, bandWorst*widest)
					}
					ran++
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no device ran")
	}
}

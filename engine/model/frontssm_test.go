package model

import (
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestFrontStatesTheStateSpaceMixers holds every recurrent block of each
// state-space fixture (ssmFixtures: a decay a head with grouped norms
// before and after the gate or none, beside attention, mixer-only blocks
// and mixtures; a decay a state element of each channel, with and without
// the dt/B/C norms; the gated short convolution) to its statement: one state convolution over the mixer's
// input, then -- but for a short convolution -- one additive delta rule at
// the container's geometry (one decay a value head, or one a state element
// where the channels are the value heads), stepping its own state, its
// output projected by the stored projection; a block that attends as well
// states its attention beside it. Every block is covered, and two of each
// kind form on a card and on the CPU.
func TestFrontStatesTheStateSpaceMixers(t *testing.T) {
	for _, fx := range ssmFixtures {
		t.Run(fx.name, func(t *testing.T) {
			m := frontModel(t, fx.name+".gguf")
			dg := m.Cfg.delta()
			byKind := map[jlm.LayerKind][]int{}
			for li := range m.Cfg.NLayer {
				k := m.Cfg.LayerKind(li)
				if why := m.Uncovered(li); why != Covered {
					t.Fatalf("block %d (%v) is refused by %q (%v)", li, k, why, m.UncoveredErr(li))
				}
				byKind[k] = append(byKind[k], li)
				if !k.Recurrent() {
					continue
				}
				g := m.Composing().BlockGraph(li)
				convs, rules := opsOf(g, graph.OpStateConv), opsOf(g, graph.OpDeltaRule)
				if len(convs) != 1 || convs[0].Attr.Taps != dg.conv || convs[0].Writes[0].Name != "rec.conv" {
					t.Fatalf("block %d states %d state convolutions, want one of %d taps", li, len(convs), dg.conv)
				}
				if k.Attends() != (len(opsOf(g, graph.OpAttend)) == 1) {
					t.Fatalf("block %d (%v) states %d attentions", li, k, len(opsOf(g, graph.OpAttend)))
				}
				readerOf(t, g, jlm.RoleSSMOut)
				if k == jlm.LayerShortConv {
					if len(rules) != 0 {
						t.Fatalf("block %d, a short convolution, states %d delta rules", li, len(rules))
					}
					continue
				}
				if len(rules) != 1 {
					t.Fatalf("block %d states %d delta rules, want one", li, len(rules))
				}
				a := rules[0].Attr
				heads, kHeads, decay := dg.vHeads, dg.kHeads, dg.vHeads
				if k == jlm.LayerMamba1 {
					kHeads, decay = 1, dg.vHeads*dg.kDim
				}
				if !a.Additive || a.Tiled || a.NHead != heads || a.NKVHead != kHeads || a.HeadDim != dg.kDim ||
					g.Values[rules[0].In[3]].Width != decay || rules[0].Writes[0].Name != "rec.state" {
					t.Fatalf("block %d: a delta rule of %+v, a decay of %d, want an additive one of %d heads over %d of %d, a decay of %d",
						li, a, g.Values[rules[0].In[3]].Width, heads, kHeads, dg.kDim, decay)
				}
			}
			for k, bs := range byKind {
				t.Logf("%v: %d blocks", k, len(bs))
				formBlocks(t, m, bs[:min(2, len(bs))])
			}
		})
	}
}

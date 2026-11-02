package model

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/engine/graph"
)

// kdaUngated is a linear block's graph with its output read as it is: the
// sigmoid gate over the delta rule's normed output a copy of that output.
// A graph of another block comes back as it was.
func kdaUngated(t *testing.T, m *Model) func(*graph.Graph) *graph.Graph {
	return func(g *graph.Graph) *graph.Graph {
		if !m.Cfg.LayerKind(g.Block).Recurrent() {
			return g
		}
		return rewriteOps(t, g, func(_ int, o *graph.Op) bool {
			if o.Kind != graph.OpExpr || !strings.HasPrefix(g.Values[o.Out[0]].Name, "gated.") || len(o.In) != 2 {
				return false
			}
			*o = graph.Op{Kind: graph.OpMap, In: o.In[1:2], Out: o.Out, Attr: graph.Attrs{Fn: graph.MapMul, Const: 1}}
			return true
		})
	}
}

// TestKimiLinearKDAGateIsLoadBearing holds Kimi-Linear's composed blocks on
// every device present to the host's logits, every block served, and its
// Kimi Delta Attention's output gate taken out of the statement alone moving
// them past 100 times the clean worst and past 1e-4.
func TestKimiLinearKDAGateIsLoadBearing(t *testing.T) {
	m := frontModel(t, "synth-kimilinear")
	if coveredBlocks(m) != m.Cfg.NLayer || !m.Cfg.ChanDecay() {
		t.Fatalf("the front end states %d of %d blocks, channel decay %v: the gate selects nothing",
			coveredBlocks(m), m.Cfg.NLayer, m.Cfg.ChanDecay())
	}
	ids := []int32{5, 11, 23, 41}
	toks := make([]int32, 24)
	for i := range toks {
		toks[i] = int32((int(ids[i%len(ids)])*7 + i) % m.Vocab.Size())
	}
	ran := 0
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			ran++
			want, _, _, _ := composedDecode(t, m, dev, ids, toks, 0, true, false)
			run := func() float64 {
				got, st, _, _ := composedDecode(t, m, dev, ids, toks, gpu.ComposedBlock, true, false)
				served, declined := st.ServedPart(gpu.ComposedBlock)
				if n := len(toks) * m.Cfg.NLayer; served != n || declined != 0 {
					t.Fatalf("%d block steps served and %d declined, want %d served (%s)", served, declined, n, declinesOf(st))
				}
				worst := 0.0
				for i := range got {
					e := logitNMSE(got[i], want[i])
					if math.IsNaN(e) {
						t.Fatalf("step %d: the logits are not finite", i)
					}
					worst = max(worst, e)
				}
				return worst
			}
			worst := run()
			undo := m.rewriteEveryBlock(t, kdaUngated(t, m))
			bad := run()
			undo()
			t.Logf("%s: clean worst NMSE %.3g, the KDA output ungated %.3g", dev, worst, bad)
			if !(worst < 1e-5) {
				t.Fatalf("worst logit NMSE %.3g against the host, bound 1e-5", worst)
			}
			if !(bad > 100*worst) || !(bad > 1e-4) {
				t.Errorf("the KDA output ungated: the composed arm cannot see it -- worst NMSE %.3g against a clean %.3g", bad, worst)
			}
		})
	}
	if ran == 0 {
		t.Fatal("no device on this host")
	}
}

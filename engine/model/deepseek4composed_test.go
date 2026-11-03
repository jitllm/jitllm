package model

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestDeepseek4ServesComposedOnCPU runs each DeepSeek V4 fixture on the CPU
// device with its input rows' copies of the embedding into every stream and
// the head's collapse of the streams composed, teacher-forced on the
// fixture's ids against main's kernels within 1e-5: every row of both and
// every head step served, every block step declined for having no graph
// (each block selects its keys, UncoveredSelect, before its Sinkhorn-mixed
// streams are reached). The copies left empty, the collapse's weights
// unscaled and a stream read twice where the collapse reads them move the
// logits past 100 times the clean worst and past 1e-4.
func TestDeepseek4ServesComposedOnCPU(t *testing.T) {
	const bound = 1e-5
	for _, fx := range ds4Fixtures() {
		t.Run(fx.name, func(t *testing.T) {
			gm, g := openDS4(t, fx.name)
			ok := fx.sel(gm)
			gm.Close()
			if !ok {
				t.Fatal("the container does not carry the fixture's graph")
			}
			path := jlmOf(t, testmodels.Path(fx.name+".gguf"))
			ref, err := Open(path, noTune())
			if err != nil {
				t.Fatal(err)
			}
			defer ref.Close()
			want := partCPULogits(t, ref, g.IDs, cpuFaultNone, false, 0, 0)
			m, err := Open(path, noTune())
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if s := m.frontGraphs(); s.expand == nil || s.collapse == nil {
				t.Fatal("the streams' copies or the head's collapse are not stated")
			}
			for li := range m.Cfg.NLayer {
				if why := m.Uncovered(li); why != UncoveredSelect {
					t.Fatalf("block %d is refused by %q, want %q", li, why, UncoveredSelect)
				}
			}
			run := func(f cpuFault) float64 {
				return worstFrom(want, partCPULogits(t, m, g.IDs, f, true, 0, composer.DeclineNoGraph))
			}
			worst := run(cpuFaultNone)
			t.Logf("the composed CPU copies and collapse from main: worst NMSE %.3g over %d steps", worst, len(g.IDs))
			if !(worst < bound) {
				t.Fatalf("the composed CPU is %.3g from main, bound %g", worst, bound)
			}
			check := func(name string, bad float64) {
				t.Helper()
				t.Logf("violation %q: worst NMSE %.3g", name, bad)
				if !(bad > 100*worst) || !(bad > 1e-4) {
					t.Errorf("%s: the composed CPU cannot see it -- worst NMSE %.3g against a clean %.3g", name, bad, worst)
				}
			}
			check("a stream read twice where the collapse reads them", run(cpuFaultStreams))
			undo := m.rewriteExpansion(t, func(g *graph.Graph) *graph.Graph {
				return rewriteOps(t, g, func(_ int, o *graph.Op) bool {
					if o.Kind != graph.OpSlice {
						return false
					}
					*o = graph.Op{Kind: graph.OpMap, In: o.In, Out: o.Out, Attr: graph.Attrs{Fn: graph.MapMul, Const: 0}}
					return true
				})
			})
			bad := run(cpuFaultNone)
			undo()
			check("the streams past the first start empty", bad)
			undo = m.rewriteCollapse(t, func(g *graph.Graph) *graph.Graph {
				return rewriteOps(t, g, func(_ int, o *graph.Op) bool {
					if o.Kind != graph.OpMap || !strings.HasPrefix(g.Values[o.Out[0]].Name, "hc.pre.scaled") {
						return false
					}
					*o = graph.Op{Kind: graph.OpMap, In: o.In, Out: o.Out, Attr: graph.Attrs{Fn: graph.MapMul, Const: 1}}
					return true
				})
			})
			bad = run(cpuFaultNone)
			undo()
			check("the collapse's weights unscaled", bad)
		})
	}
}

package engine

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestRecurrentStateIsTheGraphs: every block's recurrent state, as its
// stated graph lays it out (graph.Graph.RecurrentState), is the state the
// model's recurrent geometry sizes (Config.Delta), on every relocation
// model; a block that keeps none states none.
func TestRecurrentStateIsTheGraphs(t *testing.T) {
	blocks := 0
	for _, name := range relocModels() {
		// A restated fixture's first recurrent block departs from the
		// geometry on purpose (twoStateSuffix): its graph is not the
		// converter's.
		if _, _, ok := restatedFixture(name); ok {
			continue
		}
		t.Run(name, func(t *testing.T) {
			m, err := model.Open(jlmOf(t, testmodels.Path(name)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			c, dg := m.Cfg, deltaOf(m)
			for li := range c.NLayer {
				conv, state := m.Composing().BlockGraph(li).RecurrentState()
				wantConv, wantState := 0, 0
				if c.LayerKind(li).Recurrent() {
					wantConv, wantState = dg.ConvStateLen(), dg.StateStride()
					blocks++
				}
				if conv != wantConv || state != wantState {
					t.Errorf("block %d (%v): its graph keeps %d window and %d state floats a sequence, the geometry %d and %d",
						li, c.LayerKind(li), conv, state, wantConv, wantState)
				}
			}
		})
	}
	if blocks == 0 {
		t.Fatal("no recurrent block was checked")
	}
	t.Logf("%d recurrent blocks", blocks)
}

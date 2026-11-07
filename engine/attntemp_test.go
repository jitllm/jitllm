package engine

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
)

// TestATemperatureOffsetOfAnyValue: the attention temperature 1 + s ln(1 +
// floor((pos + o) / f)) is stated for any offset o, not only a whole one.
// For a whole position m, floor((m + 0.5) / f) is floor(m / f), so the Llama
// 4 fixture at offset 1.5 -- its floor taken in the expression -- must
// read offset 1's logits bit for bit, on the composed CPU and with every
// block on a card, while offset 2 moves them (the temperature steps a
// position sooner). The fractional arm is held to having taken the
// expression's floor.
func TestATemperatureOffsetOfAnyValue(t *testing.T) {
	m, g, opts := openSynthLlama4(t)
	defer m.Close()
	rounds := func() int {
		n := 0
		for li := range m.Cfg.NLayer {
			if why := m.Uncovered(li); why != model.Covered {
				t.Fatalf("block %d at offset %v is refused by %q (%v)", li, m.Cfg.AttnTempOffs, why, m.UncoveredErr(li))
			}
			for _, o := range m.Composing().BlockGraph(li).Ops {
				for _, e := range o.Attr.Expr {
					if o.Kind == graph.OpExpr && e.Fn == graph.MapRound {
						n++
					}
				}
			}
		}
		return n
	}
	gc, _ := openGPU(t)
	if gc != nil {
		defer gc.Close()
	}
	type arm struct {
		name string
		run  func() *Instance
	}
	arms := []arm{{"composed CPU", func() *Instance { return testInstance(t, m, len(g.IDs)+1, opts...) }}}
	if gc != nil {
		arms = append(arms, arm{"card", func() *Instance { return cardInstance(t, m, gc) }})
	}
	for _, a := range arms {
		t.Run(a.name, func(t *testing.T) {
			decode := func(offs float64) ([][]float32, int) {
				defer cfgMut(m, func(c *model.Config) { c.AttnTempOffs = offs })()
				n := rounds()
				st := a.run()
				defer st.Close()
				out := make([][]float32, 0, len(g.IDs))
				for _, id := range g.IDs {
					lg, err := st.Forward(id)
					if err != nil {
						t.Fatalf("offset %v: %v", offs, err)
					}
					out = append(out, slices.Clone(lg))
				}
				return out, n
			}
			whole, n1 := decode(1)
			half, nh := decode(1.5)
			further, _ := decode(2)
			if n1 != 0 || nh == 0 {
				t.Fatalf("offset 1 states %d rounding(s) and offset 1.5 %d: the arms did not take their own paths", n1, nh)
			}
			for p := range whole {
				if !slices.Equal(whole[p], half[p]) {
					t.Fatalf("position %d: offset 1.5 reads other logits than offset 1", p)
				}
			}
			moved := false
			for p := range whole {
				moved = moved || !slices.Equal(whole[p], further[p])
			}
			if !moved {
				t.Fatal("offset 2 reads offset 1's logits at every position: the gate cannot see the temperature")
			}
			t.Logf("%d positions: offset 1.5 equals offset 1 bit for bit through %d rounding(s); offset 2 moves them", len(whole), nh)
		})
	}
}

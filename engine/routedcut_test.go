package engine

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestRoutedCutSplitsEveryMixture cuts each mixture fixture's blocks at the
// routed experts their front end states (graph.Cut): both halves verify, the
// second binds every expert bank the block binds and the first none, and
// together they bind every weight the block binds.
func TestRoutedCutSplitsEveryMixture(t *testing.T) {
	total := 0
	for _, fx := range moeFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, _, _ := openMoE(t, fx.name)
			defer m.Close()
			roles := func(g *graph.Graph) map[jlm.Role]bool {
				rs := map[jlm.Role]bool{}
				for _, o := range g.Ops {
					for _, w := range o.Weight {
						rs[w.Role] = true
					}
				}
				return rs
			}
			cut := 0
			for li := range m.Cfg.NLayer {
				if !m.Cfg.MoEAt(li) {
					continue
				}
				g := m.Composing().BlockGraph(li)
				if g.Routed == (graph.Region{}) {
					t.Fatalf("block %d states no routed experts", li)
				}
				pre, rest, err := graph.Cut(g, g.Routed)
				if err != nil {
					t.Fatalf("block %d: %v", li, err)
				}
				f, _ := m.Inventory(int32(li))
				if err := graph.Verify(pre, f); err != nil {
					t.Fatalf("block %d's first half: %v", li, err)
				}
				if err := graph.Verify(rest, f); err != nil {
					t.Fatalf("block %d's second half: %v", li, err)
				}
				whole, a, b := roles(g), roles(pre), roles(rest)
				banks := 0
				for r := range whole {
					switch {
					case !a[r] && !b[r]:
						t.Errorf("block %d: neither half binds %v", li, r)
					case jlm.ExpertBank(r) && a[r]:
						t.Errorf("block %d: the first half binds the bank %v", li, r)
					case jlm.ExpertBank(r):
						banks++
					}
				}
				if banks == 0 {
					t.Errorf("block %d: the second half binds no bank", li)
				}
				cut++
			}
			total += cut
			t.Logf("%d blocks cut (a family's dense fixture has none)", cut)
		})
	}
	if total == 0 {
		t.Fatal("no block was cut")
	}
}

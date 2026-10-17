package tier

import (
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
)

// TestVoltaMoEWidthClass holds the class boundary: a scratch below the pair
// count builds the dp4a half, at or above it the staged GEMM. The violation is
// the boundary moved: the same shape must change class with it.
func TestVoltaMoEWidthClass(t *testing.T) {
	g := &devTier{devShared: &devShared{}}
	g.Config = &Config{}
	g.VoltaMoEMinPairs = 64
	for _, c := range []struct {
		rows, k int
		wide    bool
	}{{3, 10, false}, {6, 10, false}, {7, 10, true}, {128, 10, true}} {
		if got := g.voltaMoEWide(c.rows, c.k, 512); got != c.wide {
			t.Errorf("rows %d k %d: wide %v, want %v", c.rows, c.k, got, c.wide)
		}
	}
	g.VoltaMoEMinPairs = 1 << 30
	if g.voltaMoEWide(128, 10, 512) {
		t.Error("a raised boundary left a 128-row scratch wide: the gate cannot fire")
	}
}

// TestVoltaMoEDefaultClass: with no boundary stated a scratch is wide once its
// pairs reach the bank's expert count.
func TestVoltaMoEDefaultClass(t *testing.T) {
	g := &devTier{devShared: &devShared{}}
	g.Config = &Config{}
	if g.voltaMoEWide(8, 10, 512) || !g.voltaMoEWide(52, 10, 512) {
		t.Error("the default boundary is not the expert count")
	}
}

// TestGemmDeepVerdictIsPerWidthClass: the deep-tile verdict was timed at the
// prompt width and holds for that class only. The violation, a verdict applied
// at every width, makes the narrow launch win.
func TestGemmDeepVerdictIsPerWidthClass(t *testing.T) {
	g := &devTier{devShared: &devShared{}, lane: &lane{}}
	g.promptW = 256
	m := mv{rows: 4096, k: 4096}
	g.gemmDeep = map[splitKey]bool{{m.q, m.rows, m.k}: true}
	for _, c := range []struct {
		ntok int
		want bool
	}{{256, true}, {128, true}, {127, false}, {3, false}} {
		if got := g.gemmDeepWon(m, c.ntok); got != c.want {
			t.Errorf("ntok %d: %v, want %v", c.ntok, got, c.want)
		}
	}
	// The class is the rewrite's guard (graph.DeepTile.Pays), not a test at
	// the launch: with the guard taken away the narrow launch wins.
	pays := graph.DeepTile.Pays
	defer func() { graph.DeepTile.Pays = pays }()
	graph.DeepTile.Pays = func(*graph.Graph, graph.Region, graph.Target) error { return nil }
	if !g.gemmDeepWon(m, 3) {
		t.Error("with DeepTile's guard removed a 3-row launch still refused: the class is decided elsewhere")
	}
}

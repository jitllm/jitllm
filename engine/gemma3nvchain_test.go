package engine

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// openGemma3nV is the synth-gemma3nv fixture (scripts/gemma3nvisiongold.py)
// converted, its text model and its MobileNet-V5 tower.
func openGemma3nV(t *testing.T) *model.Model {
	t.Helper()
	text, proj := testmodels.Path("synth-gemma3nv.gguf"), testmodels.Path("mmproj-synth-gemma3nv.gguf")
	if !sourcePresent(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (scripts/gemma3nvisiongold.py writes them) (RULE 11)", text)
	}
	m, err := model.Open(jlmOfPair(t, text, proj))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// gemma3nvShapes is the golden's shape of each tap, channels then grid.
func gemma3nvShapes(t *testing.T) map[string][3]int {
	t.Helper()
	b, err := os.ReadFile(testmodels.Path("synth-gemma3nv.gold/shapes.json"))
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v (scripts/gemma3nvisiongold.py writes it)", err)
	}
	var s map[string][3]int
	if err := json.Unmarshal(b, &s); err != nil {
		t.Fatal(err)
	}
	return s
}

// TestGemma3nStatesItsChain holds Gemma 3n's stated MobileNet-V5 tower to
// the reference's own shapes (the golden's taps, transformers' hooks on the
// stem and every block): the converter states each block as a program of its
// own (jlm.Tower.Chain) over the grid the block before it left, and the
// model's graphs write every block's rows and channels as the reference's
// block does; the head reads the last block and the last of the stage before
// it, and writes the 256 soft rows and the end-of-image marker's. The first
// block stated without its downsample is the violation: every block after it
// reads the wrong grid.
func TestGemma3nStatesItsChain(t *testing.T) {
	m := openGemma3nV(t)
	defer m.Close()
	want := gemma3nvShapes(t)
	tg := m.Tower
	v := m.Container.Vision()
	if !tg.Chain || len(tg.Blocks) != int(v.NLayer) || len(tg.Blocks) == 0 {
		t.Fatalf("a chain of %d blocks (chained %v) for the tower's %d", len(tg.Blocks), tg.Chain, v.NLayer)
	}
	e := tg.Classes[0].Entry[graph.PixRGBA8]
	ev := e.Values[e.Out[0]]
	if s := want["stem"]; ev.Rows != s[1]*s[2] || ev.Width != s[0] {
		t.Fatalf("the stem writes %d rows of %d, the reference %dx%d of %d", ev.Rows, ev.Width, s[1], s[2], s[0])
	}
	kinds := map[jlm.OpCode]int{}
	for i, g := range tg.Blocks {
		o := g.Values[g.Out[0]]
		s, ok := want[fmt.Sprintf("blk%d", i)]
		if !ok {
			t.Fatalf("the golden has no block %d", i)
		}
		if o.Rows != s[1]*s[2] || o.Width != s[0] {
			t.Fatalf("block %d writes %d rows of %d, the reference %dx%d of %d", i, o.Rows, o.Width, s[1], s[2], s[0])
		}
		ops := v.Tower.Chain[i].Prog.Ops
		for _, op := range ops {
			if op.Code == jlm.TOpUnfold || op.Code == jlm.TOpRowsAttend {
				kinds[op.Code]++
			}
		}
	}
	if tg.Out != 257 || tg.Fixed != 257 || tg.Head.Values[tg.Head.Out[0]].Width != int(v.ProjDim) {
		t.Fatalf("the head writes %d rows (fixed %d) of %d, want 256 soft rows and the marker's of %d", tg.Out, tg.Fixed,
			tg.Head.Values[tg.Head.Out[0]].Width, v.ProjDim)
	}
	if len(tg.HeadFrom) != 1 {
		t.Fatalf("the head reads %d earlier blocks, want the last of the stage before the last", len(tg.HeadFrom))
	}
	from := tg.Blocks[tg.HeadFrom[0]]
	fo := from.Values[from.Out[0]]
	last := tg.Blocks[len(tg.Blocks)-1]
	lo := last.Values[last.Out[0]]
	t.Logf("%d blocks (%d edge residuals, %d attention), every one the reference's shape; the head fuses block %d's %d rows of %d "+
		"with the last's %d of %d into 257 rows of %d", len(tg.Blocks), kinds[jlm.TOpUnfold], kinds[jlm.TOpRowsAttend],
		tg.HeadFrom[0], fo.Rows, fo.Width, lo.Rows, lo.Width, v.ProjDim)
	if kinds[jlm.TOpUnfold] == 0 || kinds[jlm.TOpRowsAttend] == 0 {
		t.Fatal("the chain states no edge residual or no attention block")
	}
	if c := m.Container; c.Has(jlm.RoleVMNGeom, 0, -1) || c.Has(jlm.RoleVMNHardOffset, jlm.DenseBlock, -1) {
		t.Fatal("the container keeps the blocks' places or the hard tokens' offset, which the stated tower holds")
	}

	// The violation: the first block stated without its downsample.
	op := &v.Tower.Chain[0].Prog.Ops[0]
	st := op.Stride
	op.Stride = 1
	bad, err := m.BuildTowerGraphs()
	op.Stride = st
	if err == nil {
		g := bad.Blocks[0]
		o := g.Values[g.Out[0]]
		if s := want["blk0"]; o.Rows == s[1]*s[2] {
			t.Fatalf("VIOLATION: the first block without its downsample still writes the reference's %d rows", o.Rows)
		}
	}
	t.Logf("violation (the first block without its downsample): %v", err)
}

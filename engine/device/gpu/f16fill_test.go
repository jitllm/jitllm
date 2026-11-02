package gpu

import "testing"

// TestF16GemmGridFillsTheCard: the m16n8 GEMM's tile and k-split are chosen
// for a grid of about a workgroup an SM (a sixteenth of the resident
// threads). Llama-3.2-1B's projections keep the widest tile unsplit on a 20-SM
// card and on a 142-SM one; Qwen3-0.6B's 1024-row ones and a 32-row step take
// a narrower tile or a split there, and a shape no tile divides takes none.
func TestF16GemmGridFillsTheCard(t *testing.T) {
	const laptop, big = 20 * 1536 / 16, 142 * 1536 / 16
	for _, c := range []struct {
		rows, ntok, k, fill int
		wantFirst           bool // the widest tile, unsplit
	}{
		{8192, 512, 2048, laptop, true},
		{2048, 512, 8192, laptop, true},
		{2048, 512, 2048, laptop, true},
		{8192, 512, 2048, big, true},
		{2048, 512, 2048, big, true},
		{1024, 512, 1024, big, false},
		{512, 32, 2048, big, false},
	} {
		i := fillTile(f16Tiles, 16, c.rows, c.ntok, c.fill)
		if i < 0 {
			t.Fatalf("%+v: no tile divides", c)
		}
		tl := f16Tiles[i]
		tl.F16K = 16
		groups := (c.rows / tl.Rows()) * (c.ntok / tl.Toks())
		trips := c.k / 32
		split := fillSplit(trips, groups*tl.Threads(), c.fill)
		threads := groups * tl.Threads() * split
		t.Logf("%dx%d t%d fill %d: tile %dx%d, %d groups, split %d, %d threads", c.rows, c.k, c.ntok, c.fill,
			tl.Rows(), tl.Toks(), groups, split, threads)
		if c.wantFirst && (i != 0 || split != 1) {
			t.Fatalf("%+v: tile %d split %d, want the widest unsplit", c, i, split)
		}
		if threads < c.fill && split < 8 && trips%(2*split) == 0 {
			t.Fatalf("%+v: %d threads under the fill %d with the split still able to double", c, threads, c.fill)
		}
		if !c.wantFirst && i == 0 && split == 1 {
			t.Fatalf("%+v: the widest tile unsplit leaves %d threads of a %d fill", c, threads, c.fill)
		}
	}
	if fillTile(f16Tiles, 16, 100, 512, big) != -1 {
		t.Fatal("a 100-row matrix took a tile")
	}
}

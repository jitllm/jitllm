package convert

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestLongTileGeometryFollowsTheProcessor holds PreLongTiles' geometry to
// transformers' Idefics3ImageProcessor on every picture size
// scripts/idefics3sizes.py records, under SmolVLM-256M's config and the
// 2.2B's: the squares' count, each tile's resize to the grid and its window
// row-major, the global view last at one tile; and the marker runs
// Idefics3Processor writes around each. A geometry of another longest edge
// is the violation it must see.
func TestLongTileGeometryFollowsTheProcessor(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "idefics3", "sizes.json"))
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Transformers string `json:"transformers"`
		Cases        []struct {
			Longest, Tile, W, H, W1, H1, Rows, Cols, Squares int
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Cases) == 0 {
		t.Fatal("no cases")
	}
	const doubled = 1
	parts := func(longest, tile, w, h, rows, cols, squares int) error {
		geo := longTileGeometry(longest, tile, tile/16, doubled, idefics3Grid, true)
		_, n, err := model.GeometrySquare(geo, w, h, 0)
		if err != nil {
			return err
		}
		if n != squares {
			return fmt.Errorf("%d squares, the processor's %d", n, squares)
		}
		for i := range n {
			sq, _, err := model.GeometrySquare(geo, w, h, i)
			if err != nil {
				return err
			}
			want := model.Square{Ow: tile, Oh: tile, Pw: tile, Ph: tile, Mark: 2 + doubled, MarkN: 2, Trail: 2 + doubled, TrailN: 1}
			switch {
			case i < n-1:
				r, c := i/cols, i%cols
				want.Ow, want.Oh, want.X0, want.Y0 = cols*tile, rows*tile, c*tile, r*tile
				want.Mark, want.Trail, want.TrailN = 4+doubled+2*(r*idefics3Grid+c), 0, 0
				if c == cols-1 && r < rows-1 {
					want.Trail, want.TrailN = 1, 1
				}
			case n > 1:
				want.Mark, want.MarkN = 2, doubled+2
			}
			got := sq
			got.Class, got.Gw, got.Gh, got.Cw, got.Ch, got.Cx, got.Cy = 0, 0, 0, 0, 0, 0, 0
			if got != want {
				return fmt.Errorf("square %d is %+v, want %+v", i, got, want)
			}
		}
		return nil
	}
	seen := 0
	for _, c := range g.Cases {
		if err := parts(c.Longest, c.Tile, c.W, c.H, c.Rows, c.Cols, c.Squares); err != nil {
			t.Errorf("%dx%d at %d over %d: %v", c.W, c.H, c.Longest, c.Tile, err)
		}
		if parts(c.Longest*3/4, c.Tile, c.W, c.H, c.Rows, c.Cols, c.Squares) != nil {
			seen++
		}
	}
	if seen == 0 {
		t.Error("a longest edge of three quarters cuts every picture as the processor does: the gate cannot see the edge")
	}
	t.Logf("%d sizes cut as transformers %s's processor cuts them; a shorter edge parts on %d", len(g.Cases), g.Transformers, seen)
}

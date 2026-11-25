package composer_test

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestMatrixTileFollowsTheModel forms projections at a 512-row prompt chunk
// for a V100's stated units, threads and workgroup memory, and holds each
// matrix kernel to the tile and the cut of k the device's model costs least:
// a grid too thin to fill the units cuts k, a wide one takes the profile's
// widest warp tile whole, at the depth of k its registers hold, and a forced
// tile (Device.Tile) is the one formed whatever the model says, its cut the
// model's unless it states one.
func TestMatrixTileFollowsTheModel(t *testing.T) {
	dev := composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, Shared: 96 << 10, Matrix: [3]int{16, 16, 16}}
	tiles := gpuMatrixTiles()
	own, narrow := tiles[0], composer.MatrixTile{Rows: 128, Cols: 32, Depth: 32, Warps: [2]int{4, 1}, Resident: 2}
	cut := func(t composer.MatrixTile, s int) composer.MatrixTile { t.Split = s; return t }
	wide := composer.MatrixTile{Rows: 128, Cols: 128, Depth: 32, Warps: [2]int{2, 2}, Resident: 2}
	for _, c := range []struct {
		n, k  int
		force composer.MatrixTile
		want  composer.MatrixTile
	}{
		{512, 2048, composer.MatrixTile{}, cut(own, 4)},
		{128, 2048, composer.MatrixTile{}, cut(own, 8)},
		{4096, 4096, composer.MatrixTile{}, wide},
		{14336, 4096, composer.MatrixTile{}, wide},
		{512, 2048, own, cut(own, 4)},
		{512, 2048, cut(narrow, 1), cut(narrow, 1)},
		{4096, 4096, narrow, narrow},
	} {
		d := dev
		d.Tile = c.force
		b := graph.New()
		in := b.In("x", c.k)
		b.Out(b.MatVec(in, jlm.RoleAttnQ, c.n))
		_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: jlm.TypeQ4S},
			Step: composer.Step{Rows: 512, Page: 64, Keys: 1, ActWindow: 32}, Device: d})
		if err != nil {
			t.Fatal(err)
		}
		found, halves := false, false
		for _, cd := range codes {
			halves = halves || strings.HasSuffix(cd.Name, ".halves")
			if cd.Matrix == [2]int{} || cd.Band.Upto > 0 {
				continue
			}
			found = true
			if !formedAtTile(cd, c.want, 512, c.n) {
				t.Fatalf("%d outputs of %d, forced %v: %d workgroups of %d threads, not %v's", c.n, c.k, c.force, cd.Groups, cd.GPU.Group[0], c.want)
			}
			shared := 0
			for _, o := range cd.GPU.Ops {
				if o.Kind == ir.OpShared {
					shared += int(o.Imm)
				}
			}
			if c.want == wide && shared != 5632 {
				t.Fatalf("%d outputs of %d at %v: a stage of %d words, not the 5632 of 16 elements of k a trip its registers hold", c.n, c.k, c.want, shared)
			}
		}
		if !found {
			t.Fatalf("%d outputs of %d: no kernel was formed on the matrix unit", c.n, c.k)
		}
		if convert := c.n <= 128; halves == convert {
			t.Fatalf("%d outputs of %d: a halves kernel %v, the stage converting the row's factor %v", c.n, c.k, halves, convert)
		}
	}
}

// TestMatrixBandsFollowTheModel forms a projection at a 512-row step for a
// V100's stated units, threads and workgroup memory, and holds its codes to
// the bands the device's model prices: the vector form at the fewest rows,
// a narrower tile of the profile's at the calls past it, and the form of the
// step's capacity past those; every call of 1 to 512 rows launches exactly
// one form's projection. A forced tile is the form at every call.
func TestMatrixBandsFollowTheModel(t *testing.T) {
	dev := composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, Shared: 96 << 10, Matrix: [3]int{16, 16, 16}}
	form := func(d composer.Device, n, k int) []composer.Code {
		b := graph.New()
		in := b.In("x", k)
		b.Out(b.MatVec(in, jlm.RoleAttnQ, n))
		_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: jlm.TypeQ4S},
			Step: composer.Step{Rows: 512, Page: 64, Keys: 1, ActWindow: 32}, Device: d})
		if err != nil {
			t.Fatal(err)
		}
		return codes
	}
	projections := func(codes []composer.Code, rows int) (vec, mat []composer.Code) {
		for _, c := range codes {
			if !c.Runs(rows, 0) || !strings.Contains(c.Name, "reduce") || strings.Contains(c.Name, "combine") ||
				strings.Contains(c.Name, "halves") || strings.Contains(c.Name, "rowscale") {
				continue
			}
			if c.Matrix == [2]int{} {
				vec = append(vec, c)
			} else {
				mat = append(mat, c)
			}
		}
		return vec, mat
	}
	for _, sh := range [][2]int{{4096, 4096}, {14336, 4096}, {4096, 14336}} {
		codes := form(dev, sh[0], sh[1])
		narrow := 0
		for _, c := range codes {
			if c.Matrix != [2]int{} && c.Band.Upto > 0 {
				narrow++
			}
		}
		if narrow == 0 {
			t.Fatalf("%v: no matrix code runs in a band below the capacity's form: the model's narrower tiles were never formed", sh)
		}
		groups := map[int]bool{}
		for rows := 1; rows <= 512; rows++ {
			vec, mat := projections(codes, rows)
			if len(vec) > 0 && len(mat) > 0 || len(vec)+len(mat) == 0 {
				t.Fatalf("%v at %d rows: %d vector and %d matrix projection codes launch, want one form's", sh, rows, len(vec), len(mat))
			}
			if len(mat) > 0 {
				groups[mat[0].Groups] = true
			}
		}
		if len(groups) < 2 {
			t.Fatalf("%v: every call's matrix projection is one tile's (%v)", sh, groups)
		}
		d := dev
		d.Tile = composer.MatrixTile{Rows: 128, Cols: 128, Depth: 32, Warps: [2]int{2, 2}, Resident: 2}
		for _, c := range form(d, sh[0], sh[1]) {
			if c.Matrix != [2]int{} && c.Band.Upto > 0 {
				t.Fatalf("%v: a forced tile's form runs %s in a band of its own", sh, c.Name)
			}
		}
	}
}

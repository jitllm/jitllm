package jit_test

import (
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// TestMatrixTileFollowsTheModel forms projections at a 512-row prompt chunk
// for a V100's stated units, threads and workgroup memory, and holds each
// matrix kernel to the tile the device's model costs least: a grid too thin
// to fill the units takes narrower workgroups, a wide one keeps the
// profile's, and a forced tile (Device.Tile) is the one formed whatever the
// model says.
func TestMatrixTileFollowsTheModel(t *testing.T) {
	dev := jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, Shared: 96 << 10, Matrix: [3]int{16, 16, 16}}
	tiles := gpuMatrixTiles()
	own, narrow := tiles[0], jit.MatrixTile{Rows: 128, Cols: 32, Depth: 32, Warps: [2]int{4, 1}, Resident: 2}
	for _, c := range []struct {
		n, k  int
		force jit.MatrixTile
		want  jit.MatrixTile
	}{
		{512, 2048, jit.MatrixTile{}, narrow},
		{128, 2048, jit.MatrixTile{}, narrow},
		{4096, 4096, jit.MatrixTile{}, own},
		{14336, 4096, jit.MatrixTile{}, own},
		{512, 2048, own, own},
		{4096, 4096, narrow, narrow},
	} {
		d := dev
		d.Tile = c.force
		b := graph.New()
		in := b.In("x", c.k)
		b.Out(b.MatVec(in, jlm.RoleAttnQ, c.n))
		_, codes, err := jit.Default().Form(jit.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: jlm.TypeQ4S},
			Step: jit.Step{Rows: 512, Page: 64, Keys: 1, ActWindow: 32}, Device: d})
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, cd := range codes {
			if cd.Matrix == [2]int{} {
				continue
			}
			found = true
			if !formedAtTile(cd, c.want, 512, c.n) {
				t.Fatalf("%d outputs of %d, forced %v: %d workgroups of %d threads, not %v's", c.n, c.k, c.force, cd.Groups, cd.GPU.Group[0], c.want)
			}
		}
		if !found {
			t.Fatalf("%d outputs of %d: no kernel was formed on the matrix unit", c.n, c.k)
		}
	}
}

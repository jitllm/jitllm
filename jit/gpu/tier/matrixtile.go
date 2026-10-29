package tier

import "github.com/jitllm/jitllm/jit"

// matrixTile is the matrix tile Config.MatrixTile forces at a step of rows:
// the profile's k'th form (its own, then jit.Matrix.Tiles), or none, where
// the composer's model chooses per projection.
func (g *devTier) matrixTile(r *jit.Registry, dev jit.Device, rows int) jit.MatrixTile {
	f := g.Config.MatrixTile
	if f <= 0 || dev.Matrix[0] == 0 || rows < dev.Matrix[0] {
		return jit.MatrixTile{}
	}
	for _, p := range r.Profiles[jit.DeviceGPU].Passes {
		if m, ok := p.(jit.Matrix); ok {
			if tiles := append([]jit.MatrixTile{m.Tile()}, m.Tiles...); f <= len(tiles) {
				return tiles[f-1]
			}
		}
	}
	return jit.MatrixTile{}
}

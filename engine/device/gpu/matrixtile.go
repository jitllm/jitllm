package gpu

import "github.com/jitllm/jitllm/composer"

// matrixTile is the matrix tile Config.MatrixTile forces at a step of rows:
// the profile's k'th form (its own, then composer.Matrix.Tiles), or none, where
// the composer's model chooses per projection.
func (g *devTier) matrixTile(r *composer.Registry, dev composer.Device, rows int) composer.MatrixTile {
	f := g.Config.MatrixTile
	if f <= 0 || dev.Matrix[0] == 0 || rows < dev.Matrix[0] {
		return composer.MatrixTile{}
	}
	for _, p := range r.Profiles[composer.DeviceGPU].Passes {
		if m, ok := p.(composer.Matrix); ok {
			if tiles := append([]composer.MatrixTile{m.Tile()}, m.Tiles...); f <= len(tiles) {
				return tiles[f-1]
			}
		}
	}
	return composer.MatrixTile{}
}

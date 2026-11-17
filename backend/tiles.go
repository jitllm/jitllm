package backend

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/composer/ptx"
)

// TileShape is one collective multiply-accumulate: an M x K by K x N product
// of Elem operands into an Acc accumulator, its rows scaled first where
// Scale is set (ir.OpTileScale).
type TileShape struct {
	M, N, K   int
	Elem, Acc ir.TileElem
	Scale     bool
}

// types is the shape's A, B and accumulator tile types.
func (s TileShape) types() (a, b, acc ir.TileType) {
	return ir.TileType{Rows: s.M, Cols: s.K, Elem: s.Elem, Use: ir.TileA},
		ir.TileType{Rows: s.K, Cols: s.N, Elem: s.Elem, Use: ir.TileB},
		ir.TileType{Rows: s.M, Cols: s.N, Elem: s.Acc, Use: ir.TileAcc}
}

// tileStater is a device that states which collective tiles it multiplies.
type tileStater interface {
	lowersTile(s TileShape) error
}

// LowersTile reports whether d multiplies collective tiles of shape s,
// staged from workgroup memory and carried across a loop -- the form a
// tiled GEMM uses -- and the reason when it does not.
//
// The answer is what the device's API states (CUDA's compute capability,
// the cooperative-matrix combinations a Vulkan device enumerated and the
// features it enabled, a Metal device's GPU family) read against what its
// lowering emits for that statement (ptx.LowersTile, spirv.LowersStagedTile,
// msl.LowersTile, amdgpu.LowersTile). Nothing is compiled.
func LowersTile(d Device, s TileShape) (bool, string) {
	a, b, acc := s.types()
	for _, t := range [3]ir.TileType{a, b, acc} {
		if !t.Valid() {
			return false, fmt.Sprintf("backend: %v is not a well-formed tile type", t)
		}
	}
	if s.Scale && s.Acc != ir.TileF32 {
		return false, fmt.Sprintf("backend: a row scale of a %s accumulator: tile.scale scales a float32 one", s.Acc)
	}
	ts, ok := d.(tileStater)
	if !ok {
		return false, fmt.Sprintf("backend: %s states no collective tile", d.API())
	}
	if err := ts.lowersTile(s); err != nil {
		return false, err.Error()
	}
	return true, ""
}

// TileQuadPairs reports whether d lowers the collective tiles in sm_70's
// quad-pair layout (composer/ptx/tile.go): one m8n8k4 a pair, four pairs a warp
// each on its own 4 of k, so a tile multiplies at the unit's full rate only
// where its k is a multiple of 16, and a k of 8 runs half the pairs on zeros.
func TileQuadPairs(d Device) bool {
	c, ok := d.(*cudaDev)
	return ok && ptx.QuadPairTiles(c.d.Target)
}

// Fragment is the matrix instruction d runs alone, its lanes' fragments
// laid as ir states them (ir.MMAVolta() where TileQuadPairs), else none.
func Fragment(d Device) ir.MMAShape {
	if TileQuadPairs(d) {
		return ir.MMAVolta()
	}
	return ir.MMAShape{}
}

package backend

import (
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

// LowersTile reports whether d compiles a kernel that multiplies collective
// tiles of shape s, staged from workgroup memory and carried across a loop --
// the form a tiled GEMM uses -- and the device's own words when it does not.
//
// It is the capability question the tier asks before it builds a tile kernel,
// in place of asking for a backend by name: Metal lowers 8x8 tiles to
// simdgroup_matrix, PTX lowers any f16 tile whose sides are multiples of 8 to
// mma.sync (sm_70 and later; see TileQuadPairs), and Vulkan lowers the shapes
// the device enumerated. The probe is compiled, never launched; asking costs a compile,
// so a caller keeps the answer for the device it holds.
func LowersTile(d Device, s TileShape) (bool, string) {
	k, err := tileProbe(s)
	if err != nil {
		return false, err.Error()
	}
	kern, err := d.Compile(k)
	if err != nil {
		return false, err.Error()
	}
	kern.Close()
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

// tileProbe is the smallest kernel with every part of a staged tile GEMM: A
// and B staged through workgroup memory behind a barrier, the accumulator a
// loop-carried tile, the result stored column-major to a buffer.
func tileProbe(s TileShape) (*ir.Kernel, error) {
	aT := ir.TileType{Rows: s.M, Cols: s.K, Elem: s.Elem, Use: ir.TileA}
	bT := ir.TileType{Rows: s.K, Cols: s.N, Elem: s.Elem, Use: ir.TileB}
	cT := ir.TileType{Rows: s.M, Cols: s.N, Elem: s.Acc, Use: ir.TileAcc}
	b := ir.New("tilecap", [3]int{ir.SubgroupLanes, 1, 1})
	pIn := b.Param("pIn", ir.U32)
	pOut := b.ParamTile("pOut", s.Acc)
	words := (s.M*s.K + s.K*s.N) * s.Elem.Bytes() / 4
	sh := b.Shared("tilecap", ir.U32, max(words, 1))
	tid := b.TID()
	b.Store(sh, tid, b.Load(ir.U32, pIn, tid, 0), 0)
	b.Barrier()
	zero := b.Const(ir.U32, 0)
	seed := b.ConstF32(0)
	if s.Acc == ir.TileI32 {
		seed = b.Const(ir.I32, 0)
	}
	init := b.TileSplat(cT, seed)
	b.Loop(2)
	acc := b.TilePhi(init)
	a := b.TileLoad(aT, sh, zero, b.Const(ir.U32, int64(s.K)), false)
	bb := b.TileLoad(bT, sh, b.Const(ir.U32, int64(s.M*s.K)), b.Const(ir.U32, int64(s.K)), true)
	c := acc
	if s.Scale {
		c = b.TileScaleRows(acc, sh, zero)
	}
	b.SetPhi(acc, b.TileMMA(a, bb, c))
	b.EndLoop()
	b.TileStore(pOut, zero, b.Const(ir.U32, int64(s.M)), acc, true)
	k, err := b.Done()
	if err != nil {
		return nil, err
	}
	return k, k.Validate()
}

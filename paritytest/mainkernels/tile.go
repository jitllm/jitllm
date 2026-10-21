package mainkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// TileProbe is the smallest kernel that exercises a collective matrix end to
// end: load A and B, zero an accumulator, chain several multiply-accumulates,
// store the result.
//
// The chain is what matters: a cooperative value must survive from one MulAdd
// to the next without a memory round trip, which a single MMA would not test.
// The device's shapes are a list it enumerates
// (jit_old/gpu/vulkan.CooperativeMatrixShapes) and operand, accumulator and
// storage types must all match, so one numeric probe checks them all.
//
// A and B are row-major, k tiles of them laid end to end. The reference is
// plain: out = sum over t of A[t] * B[t].
func TileProbe(m, n, k, chain int, elem ir.TileElem) (*ir.Kernel, error) {
	if chain < 1 {
		return nil, fmt.Errorf("kernels: TileProbe: chain=%d, want at least one MulAdd", chain)
	}
	acc := ir.TileF32
	if elem == ir.TileS8 {
		acc = ir.TileI32
	}
	aT := ir.TileType{Rows: m, Cols: k, Elem: elem, Use: ir.TileA}
	bT := ir.TileType{Rows: k, Cols: n, Elem: elem, Use: ir.TileB}
	cT := ir.TileType{Rows: m, Cols: n, Elem: acc, Use: ir.TileAcc}
	for _, t := range []ir.TileType{aT, bT, cT} {
		if !t.Valid() {
			return nil, fmt.Errorf("kernels: TileProbe: %v is not a well-formed tile type", t)
		}
	}

	b := ir.New("tileprobe", [3]int{32, 1, 1})
	pA := b.ParamTile("pA", elem)
	pB := b.ParamTile("pB", elem)
	pOut := b.ParamTile("pOut", acc)

	// Strides are in elements of the component type: Vulkan counts them in
	// the matrix's own components, not bytes and not the buffer's scalars.
	aStride := b.Const(ir.U32, int64(k))
	bStride := b.Const(ir.U32, int64(n))
	outStride := b.Const(ir.U32, int64(n))

	c := b.TileSplat(cT, b.Const(acc2scalar(acc), 0))
	for t := 0; t < chain; t++ {
		av := b.TileLoad(aT, pA, b.Const(ir.U32, int64(t*m*k)), aStride, false)
		bv := b.TileLoad(bT, pB, b.Const(ir.U32, int64(t*k*n)), bStride, false)
		c = b.TileMMA(av, bv, c)
	}
	b.TileStore(pOut, b.Const(ir.U32, 0), outStride, c, false)
	return b.Done(), nil
}

// acc2scalar is the scalar type a splat's seed carries. The accumulator's
// component type is the tile's; the value handed to the splat is an ordinary
// register, and those are still only 32-bit here.
func acc2scalar(e ir.TileElem) ir.Type {
	if e == ir.TileI32 {
		return ir.I32
	}
	return ir.F32
}

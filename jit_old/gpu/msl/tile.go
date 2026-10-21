package msl

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// Metal's matrix unit (simdgroup_matrix). What it admits was measured by
// metal.TestSimdgroupMatrixForms:
//
//	f32 8x8 load/mac/store                     available
//	f16 8x8, half accumulator                  available
//	half operands, float accumulator           available
//	f32 8x16, f32 16x16                        not available
//	int8 8x8 (char), int32 8x8 accumulator     not available
//	transposed load (the B operand's layout)   available
//
// The mixed form (f16*f16+f32 -> f32) is attention's shape, despite the
// single-template-parameter documentation. 8x8 is the only shape (tileT refuses
// any other), and Vulkan's cooperative matrices here have no 8x8x8
// (vulkan.CooperativeMatrixShapes), so no tile shape is common to both;
// composing a larger tile from 8x8 ones is the kernel's decision, not the lowering's. There is no
// integer form, so s8 tiles are declined as hardware.

// tileT is the MSL spelling of a tile type, or an error naming why this target
// has no such matrix. Every refusal here is a compile-time refusal of the whole
// kernel: there is no scalar twin to fall back to within a tile op.
func tileT(t ir.TileType) (string, error) {
	if t.Rows != 8 || t.Cols != 8 {
		return "", fmt.Errorf("msl: simdgroup_matrix is 8x8 on this target and %s is not; "+
			"compose the shape in the kernel (simdgroup_matrix<float,8,16> and "+
			"<float,16,16> both fail to compile here -- metal.TestSimdgroupMatrixForms)", t)
	}
	switch t.Elem {
	case ir.TileF32:
		return "simdgroup_matrix<float,8,8>", nil
	case ir.TileF16:
		if t.Use == ir.TileAcc {
			// Unreachable through Valid(), which refuses a narrow accumulator.
			// Stated anyway: half accumulation is a silent precision
			// substitution and must never be reached by falling through.
			return "", fmt.Errorf("msl: a half accumulator is refused, not substituted")
		}
		return "simdgroup_matrix<half,8,8>", nil
	}
	return "", fmt.Errorf("msl: no %s matrix: simdgroup_matrix has no integer form on this "+
		"target (char and int both fail to compile -- metal.TestSimdgroupMatrixForms)", t.Elem)
}

// tile lowers the four collective-tile ops.
func (l *lowerer) tile(i int, o ir.Op) error {
	ty, err := tileT(*o.Tile)
	if err != nil {
		return err
	}
	v := fmt.Sprintf("v%d", i+1)

	switch o.Kind {
	case ir.OpTileSplat:
		l.name[i+1] = v
		l.line("%s %s = %s(%s);", ty, v, ty, l.n(o.Args[0]))

	case ir.OpTileMMA:
		// D and C are separate declarations: the IR is SSA, and writing
		// through C would make the next MulAdd in a chain read an overwritten
		// value.
		l.name[i+1] = v
		l.line("%s %s = %s;", ty, v, l.n(o.Args[2]))
		l.line("simdgroup_multiply_accumulate(%s, %s, %s, %s);",
			v, l.n(o.Args[0]), l.n(o.Args[1]), v)

	case ir.OpTileLoad, ir.OpTileStore:
		p, err := l.tileParam(o)
		if err != nil {
			return err
		}
		// The layout is simdgroup_load's transpose flag, not a property
		// of the type. ulong2(0,0) is the origin within the tile; the
		// view's offset is already in the pointer.
		trans := "false"
		if ir.ColMajor(o) {
			trans = "true"
		}
		ptr := fmt.Sprintf("%s + %s", p, l.n(o.Args[1]))
		if o.Kind == ir.OpTileStore {
			l.line("simdgroup_store(%s, %s, %s, ulong2(0, 0), %s);",
				l.n(o.Vargs[0]), ptr, l.n(o.Args[2]), trans)
			return nil
		}
		l.name[i+1] = v
		l.line("%s %s;", ty, v)
		l.line("simdgroup_load(%s, %s, %s, ulong2(0, 0), %s);", v, ptr, l.n(o.Args[2]), trans)

	default:
		return fmt.Errorf("msl: %s is not a tile op", o.Kind)
	}
	return nil
}

// tileParam is the buffer a collective access starts from, refusing a buffer
// whose element type is not the tile's.
//
// It is the same refusal spirv/tile.go makes: simdgroup_load reads its
// component type off the pointer, so a half matrix over a `device float*`
// would silently read the wrong elements. ir.ParamTile declares what a buffer
// holds.
func (l *lowerer) tileParam(o ir.Op) (string, error) {
	pd := int(o.Args[0])
	if pd <= 0 || pd > len(l.k.Ops) {
		return "", fmt.Errorf("msl: %s: operand 0 is not a parameter", o.Kind)
	}
	if sh := l.k.Ops[pd-1]; sh.Kind == ir.OpShared {
		// A shared array is declared in words and read as the tile's
		// component: offset and stride count components
		// (ir.validateTiles), so the cast comes before the offset.
		return fmt.Sprintf("((threadgroup %s*)%s)", tileElemName(o.Tile.Elem), l.n(o.Args[0])), nil
	}
	p := l.k.Params[l.k.Ops[pd-1].Imm]
	if !p.IsTile {
		return "", fmt.Errorf("msl: %s reads %q, which is not a tile parameter", o.Kind, p.Name)
	}
	if p.Tile != o.Tile.Elem {
		return "", fmt.Errorf("msl: %s wants %s and %q holds %s", o.Kind, o.Tile.Elem, p.Name, p.Tile)
	}
	return p.Name, nil
}

// tileElemName is the MSL element type a tile PARAMETER is declared with.
func tileElemName(e ir.TileElem) string {
	switch e {
	case ir.TileF16:
		return "half"
	case ir.TileS8:
		return "char"
	case ir.TileI32:
		return "int"
	}
	return "float"
}

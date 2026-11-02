package msl

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/composer/ir"
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
// single-template-parameter documentation. 8x8 is the only matrix shape, so
// a larger tile -- a multiple of 8 each way, as ir.TileType.Valid holds it --
// is lowered as a struct of 8x8 matrices, row-major by block (tileStructs):
// a load or store of each block at its own offset, a product the blocks'
// sum over the depth. There is no integer form, so s8 tiles are declined as
// hardware.

// tileElem is the MSL simdgroup_matrix of one 8x8 block of t, or an error
// naming why this target has no such matrix. Every refusal here is a
// compile-time refusal of the whole kernel: there is no scalar twin to fall
// back to within a tile op.
func tileElem(t ir.TileType) (string, error) {
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

// tileT is the MSL spelling of a tile type: one simdgroup_matrix at 8x8,
// else the struct of its blocks (tileStructs).
func tileT(t ir.TileType) (string, error) {
	if t.Rows <= 0 || t.Cols <= 0 || t.Rows%8 != 0 || t.Cols%8 != 0 {
		return "", fmt.Errorf("msl: %s is not a whole number of 8x8 simdgroup matrices", t)
	}
	e, err := tileElem(t)
	if err != nil {
		return "", err
	}
	if t.Rows == 8 && t.Cols == 8 {
		return e, nil
	}
	return fmt.Sprintf("jitllm_tile_%s_%dx%d", tileElemName(t.Elem), t.Rows, t.Cols), nil
}

// tileStructs declares the struct of every tile type of k past 8x8, each
// once: its 8x8 blocks, row-major by block. A struct and not an array, so a
// loop-carried tile is reassigned whole as a scalar is.
func tileStructs(k *ir.Kernel) (string, error) {
	seen := map[string]bool{}
	var b strings.Builder
	for _, o := range k.Ops {
		if o.Tile == nil || o.Tile.Rows == 8 && o.Tile.Cols == 8 {
			continue
		}
		n, err := tileT(*o.Tile)
		if err != nil {
			return "", err
		}
		if seen[n] {
			continue
		}
		seen[n] = true
		e, _ := tileElem(*o.Tile)
		fmt.Fprintf(&b, "struct %s { %s m[%d]; };\n", n, e, o.Tile.Rows/8*(o.Tile.Cols/8))
	}
	return b.String(), nil
}

// tile lowers the four collective-tile ops.
func (l *lowerer) tile(i int, o ir.Op) error {
	t := *o.Tile
	ty, err := tileT(t)
	if err != nil {
		return err
	}
	v := fmt.Sprintf("v%d", i+1)
	whole := t.Rows == 8 && t.Cols == 8
	br, bc := t.Rows/8, t.Cols/8
	// blk is block (r, c) of a tile of rows x cols blocks named name.
	blk := func(name string, rows, cols, r, c int) string {
		if rows == 1 && cols == 1 {
			return name
		}
		return fmt.Sprintf("%s.m[%d]", name, r*cols+c)
	}

	switch o.Kind {
	case ir.OpTileSplat:
		// make_filled_simdgroup_matrix, not the scalar constructor, which
		// makes a diagonal matrix.
		l.name[i+1] = v
		fill := fmt.Sprintf("make_filled_simdgroup_matrix<%s, 8, 8>(%s)", tileElemName(t.Elem), l.n(o.Args[0]))
		if whole {
			l.line("%s %s = %s;", ty, v, fill)
			return nil
		}
		l.line("%s %s;", ty, v)
		for q := range br * bc {
			l.line("%s.m[%d] = %s;", v, q, fill)
		}

	case ir.OpTileMMA:
		// D and C are separate declarations: the IR is SSA, and writing
		// through C would make the next MulAdd in a chain read an overwritten
		// value.
		l.name[i+1] = v
		l.line("%s %s = %s;", ty, v, l.n(o.Args[2]))
		at := l.k.Ops[o.Args[0]-1].Tile
		if at == nil || at.Cols <= 0 || at.Cols%8 != 0 {
			return fmt.Errorf("msl: op %d: a tile product whose A is not a tile of whole 8x8 blocks", i+1)
		}
		bk := at.Cols / 8
		for r := range br {
			for c := range bc {
				d := blk(v, br, bc, r, c)
				for k := range bk {
					l.line("simdgroup_multiply_accumulate(%s, %s, %s, %s);", d,
						blk(l.n(o.Args[0]), br, bk, r, k), blk(l.n(o.Args[1]), bk, bc, k, c), d)
				}
			}
		}

	case ir.OpTileLoad, ir.OpTileStore:
		p, err := l.tileParam(o)
		if err != nil {
			return err
		}
		// The layout is simdgroup_load's transpose flag, not a property
		// of the type. ulong2(0,0) is the origin within the block; the
		// view's offset and the block's are in the pointer: block (r, c)
		// starts 8r rows and 8c columns in, a row a stride apart where the
		// layout is row-major and a column a stride apart where it is not.
		trans := "false"
		if ir.ColMajor(o) {
			trans = "true"
		}
		stride := l.n(o.Args[2])
		ptr := func(r, c int) string {
			if ir.ColMajor(o) {
				r, c = c, r
			}
			if r == 0 && c == 0 {
				return fmt.Sprintf("%s + %s", p, l.n(o.Args[1]))
			}
			return fmt.Sprintf("%s + %s + %du * %s + %du", p, l.n(o.Args[1]), 8*r, stride, 8*c)
		}
		name := v
		if o.Kind == ir.OpTileStore {
			name = l.n(o.Vargs[0])
		} else {
			l.name[i+1] = v
			l.line("%s %s;", ty, v)
		}
		for r := range br {
			for c := range bc {
				if o.Kind == ir.OpTileStore {
					l.line("simdgroup_store(%s, %s, %s, ulong2(0, 0), %s);", blk(name, br, bc, r, c), ptr(r, c), stride, trans)
				} else {
					l.line("simdgroup_load(%s, %s, %s, ulong2(0, 0), %s);", blk(name, br, bc, r, c), ptr(r, c), stride, trans)
				}
			}
		}

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

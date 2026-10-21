package spirv

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
)

// Cooperative matrices (SPV_KHR_cooperative_matrix), Vulkan's only matrix
// instruction. Unlike PTX's mma.sync, the value is opaque, loaded from memory
// with a stride, and the lane layout is undefined, which is why it is a
// separate IR op (the tile ops) rather than a lowering of ir.OpMMA.
//
// The shapes are the driver's and are enumerated, not assumed (see
// jit_old/gpu/vulkan/coopmat.go): the feature bit says nothing about (M,N,K) or
// component types.
const (
	opTypeCooperativeMatrixKHR   = 4456
	opCooperativeMatrixLoadKHR   = 4457
	opCooperativeMatrixStoreKHR  = 4458
	opCooperativeMatrixMulAddKHR = 4459

	// CooperativeMatrixOperandsMask. A literal on the MulAdd, not an id.
	coopASigned      = 0x0001 // MatrixASignedComponentsKHR
	coopBSigned      = 0x0002 // MatrixBSignedComponentsKHR
	coopCSigned      = 0x0004 // MatrixCSignedComponentsKHR
	coopResultSigned = 0x0008 // MatrixResultSignedComponentsKHR

	capFloat16                  = 9
	capStorageBuffer16BitAccess = 4433
	capStorageBuffer8BitAccess  = 4448
	capCooperativeMatrixKHR     = 6022

	// VkComponentTypeKHR / CooperativeMatrixUse / layout values. In the
	// instruction they are <id>s of u32 constants, not literals.
	coopUseA   = 0
	coopUseB   = 1
	coopUseAcc = 2

	coopRowMajor = 0
	coopColMajor = 1

	scopeSubgroupID = 3
)

// hasTiles reports whether the kernel contains a collective value, which is
// what decides the capabilities and extensions the module must declare.
func hasTiles(k *ir.Kernel) bool {
	for _, p := range k.Params {
		if p.IsTile {
			return true
		}
	}
	for _, o := range k.Ops {
		if ir.IsTileOp(o.Kind) {
			return true
		}
	}
	return false
}

// tileCaps declares what a cooperative-matrix module needs. It is called from
// the same place as the dot-product and subgroup capabilities.
//
// 16-bit and 8-bit storage are capabilities separate from the arithmetic ones:
// an f16 buffer needs StorageBuffer16BitAccess whatever the matrix extension
// says, or the module validates on one driver and is rejected by another.
func (l *lowerer) tileCaps() {
	m := &l.m
	inst(&m.caps, opCapability, capCooperativeMatrixKHR)
	inst(&m.exts, opExtension, str("SPV_KHR_cooperative_matrix")...)
	f16, i8 := false, false
	for _, p := range l.k.Params {
		if !p.IsTile {
			continue
		}
		switch p.Tile {
		case ir.TileF16:
			f16 = true
		case ir.TileS8:
			i8 = true
		}
	}
	for _, o := range l.k.Ops {
		if o.Tile == nil {
			continue
		}
		switch o.Tile.Elem {
		case ir.TileF16:
			f16 = true
		case ir.TileS8:
			i8 = true
		}
	}
	if f16 {
		inst(&m.caps, opCapability, capFloat16)
		inst(&m.caps, opCapability, capStorageBuffer16BitAccess)
		inst(&m.exts, opExtension, str("SPV_KHR_16bit_storage")...)
	}
	if i8 {
		inst(&m.caps, opCapability, capInt8)
		inst(&m.caps, opCapability, capStorageBuffer8BitAccess)
		inst(&m.exts, opExtension, str("SPV_KHR_8bit_storage")...)
	}
}

// compType is the SPIR-V type id for a tile's component type, interned.
func (l *lowerer) compType(e ir.TileElem) uint32 {
	if l.comps == nil {
		l.comps = map[ir.TileElem]uint32{}
	}
	if id, ok := l.comps[e]; ok {
		return id
	}
	id := l.m.id()
	switch e {
	case ir.TileF16:
		inst(&l.m.types, opTypeFloat, id, 16)
	case ir.TileF32:
		id = l.tf32
	case ir.TileS8:
		inst(&l.m.types, opTypeInt, id, 8, 1)
	case ir.TileI32:
		id = l.ti32
	}
	l.comps[e] = id
	return id
}

// tileType is the OpTypeCooperativeMatrixKHR id for a tile type, interned.
//
// Every operand of the type instruction (scope, rows, columns, use) is an <id>
// of a u32 constant, not a literal; literals give an operand-kind error that
// does not name the cause.
func (l *lowerer) tileType(t ir.TileType) uint32 {
	if l.tiles == nil {
		l.tiles = map[ir.TileType]uint32{}
	}
	if id, ok := l.tiles[t]; ok {
		return id
	}
	use := uint32(coopUseA)
	switch t.Use {
	case ir.TileB:
		use = coopUseB
	case ir.TileAcc:
		use = coopUseAcc
	}
	id := l.m.id()
	inst(&l.m.types, opTypeCooperativeMatrixKHR, id, l.compType(t.Elem),
		l.u32c(scopeSubgroupID), l.u32c(uint32(t.Rows)), l.u32c(uint32(t.Cols)), l.u32c(use))
	l.tiles[t] = id
	return id
}

// tileOp lowers one collective op.
func (l *lowerer) tileOp(o ir.Op, v ir.Value) error {
	m := &l.m
	switch o.Kind {
	case ir.OpTileSplat:
		// A cooperative matrix constructed from one scalar: how an
		// accumulator is zeroed without a memory round trip.
		id := m.id()
		inst(&m.funcs, opCompositeConstruct, l.tileType(*o.Tile), id, l.val[o.Args[0]])
		l.val[v] = id

	case ir.OpTileLoad, ir.OpTileStore:
		ptr, err := l.tilePtr(o)
		if err != nil {
			return err
		}
		layout := uint32(coopRowMajor)
		if ir.ColMajor(o) {
			layout = coopColMajor
		}
		if o.Kind == ir.OpTileStore {
			inst(&m.funcs, opCooperativeMatrixStoreKHR, ptr, l.val[o.Vargs[0]],
				l.u32c(layout), l.val[o.Args[2]])
			return nil
		}
		id := m.id()
		inst(&m.funcs, opCooperativeMatrixLoadKHR, l.tileType(*o.Tile), id, ptr,
			l.u32c(layout), l.val[o.Args[2]])
		l.val[v] = id

	case ir.OpTileMMA:
		id := m.id()
		// The operands mask is required for integers: absent, every operand
		// is read as unsigned. It is derived from each operand's own element
		// type rather than guessed from the result's.
		ops := uint32(0)
		for i, bit := range [3]uint32{coopASigned, coopBSigned, coopCSigned} {
			if t := ir.TileTypeOf(l.k, o.Args[i]); t != nil && t.Elem.Signed() {
				ops |= bit
			}
		}
		if o.Tile.Elem.Signed() {
			ops |= coopResultSigned
		}
		if ops != 0 {
			inst(&m.funcs, opCooperativeMatrixMulAddKHR, l.tileType(*o.Tile), id,
				l.val[o.Args[0]], l.val[o.Args[1]], l.val[o.Args[2]], ops)
		} else {
			inst(&m.funcs, opCooperativeMatrixMulAddKHR, l.tileType(*o.Tile), id,
				l.val[o.Args[0]], l.val[o.Args[1]], l.val[o.Args[2]])
		}
		l.val[v] = id

	default:
		return fmt.Errorf("spirv: %s is not a tile op", o.Kind)
	}
	return nil
}

// tilePtr is the pointer a cooperative access starts from: the element of the
// buffer at the view's offset.
//
// The buffer must be a tile parameter holding the tile's component type: a
// cooperative load does not convert, so a mismatch is refused at compile time
// rather than reading the wrong elements.
func (l *lowerer) tilePtr(o ir.Op) (uint32, error) {
	pd := int(o.Args[0])
	p := l.k.Params[l.k.Ops[pd-1].Imm]
	if !p.IsTile {
		return 0, fmt.Errorf("spirv: %s through %q, which is not a tile buffer "+
			"(declare it with ir.ParamTile)", o.Kind, p.Name)
	}
	if p.Tile != o.Tile.Elem {
		return 0, fmt.Errorf("spirv: %s: buffer %q holds %s and the tile is %s; a "+
			"cooperative access does not convert", o.Kind, p.Name, p.Tile, o.Tile.Elem)
	}
	ptr := l.m.id()
	inst(&l.m.funcs, opAccessChain, l.ptrElem(l.compType(p.Tile)), ptr,
		l.val[o.Args[0]], l.u32c(0), l.val[o.Args[1]])
	return ptr, nil
}

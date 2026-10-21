package ir

import "fmt"

// Collective tile values: one logical matrix distributed across a subgroup,
// whose lane mapping belongs to the driver rather than the kernel.
//
// OpMMA's operands are per-lane registers the kernel builds itself. A
// cooperative matrix (SPV_KHR_cooperative_matrix, MSL's simdgroup_matrix) is
// instead an opaque value loaded from memory with a stride, and the specs do
// not say which lane holds which element. So load, splat, multiply-accumulate
// and store are four ops, and a tile never appears in scalar arithmetic.
//
// The shapes are the device's, not PTX's; see jit_old/gpu/vulkan/coopmat.go.

// TileUse is an operand's role. A cooperative matrix carries its role in its
// type, because the driver lays each out for the multiply it feeds.
type TileUse uint8

const (
	// TileA is the left operand, Rows x Cols = M x K.
	TileA TileUse = iota
	// TileB is the right operand, K x N.
	TileB
	// TileAcc is the accumulator and the result, M x N. Its element type is
	// wider than the operands' (f32 for f16, i32 for int8).
	TileAcc
)

func (u TileUse) String() string {
	switch u {
	case TileA:
		return "A"
	case TileB:
		return "B"
	case TileAcc:
		return "Acc"
	}
	return fmt.Sprintf("use(%d)", uint8(u))
}

// TileElem is a tile's component type, deliberately separate from the scalar
// Type: registers carry binary16 packed in a u32 and int8 four to a word, so
// every scalar op stays 32-bit. The component type is a property of the matrix.
type TileElem uint8

const (
	TileF16 TileElem = iota
	TileF32
	TileS8
	TileI32
)

func (e TileElem) String() string {
	switch e {
	case TileF16:
		return "f16"
	case TileF32:
		return "f32"
	case TileS8:
		return "s8"
	case TileI32:
		return "i32"
	}
	return fmt.Sprintf("elem(%d)", uint8(e))
}

// TileType is a collective value's complete type.
//
// Scope is absent: every shape this hardware offers is subgroup scope.
type TileType struct {
	Rows, Cols int
	Elem       TileElem
	Use        TileUse
}

func (t TileType) String() string {
	return fmt.Sprintf("tile<%dx%d %s %s>", t.Rows, t.Cols, t.Elem, t.Use)
}

// Valid reports whether the type is well formed. Whether the device has it is
// a capability question, answered by a list.
func (t TileType) Valid() bool {
	if t.Rows <= 0 || t.Cols <= 0 || t.Rows%8 != 0 || t.Cols%8 != 0 {
		return false
	}
	switch t.Use {
	case TileA, TileB, TileAcc:
	default:
		return false
	}
	switch t.Elem {
	case TileF16, TileF32, TileS8, TileI32:
	default:
		return false
	}
	// An accumulator must be wider than an operand; a narrow accumulator is a
	// different instruction.
	if t.Use == TileAcc && (t.Elem == TileF16 || t.Elem == TileS8) {
		return false
	}
	return true
}

// TileLoad reads a tile from buf at an element offset, with a row stride in
// elements of the buffer (Vulkan counts the matrix's component type; bytes
// would validate and read the wrong memory). colMajor is a property of the
// access, not the type.
func (b *Builder) TileLoad(t TileType, buf, offset, stride Value, colMajor bool) Value {
	if !t.Valid() {
		panic(fmt.Sprintf("ir: TileLoad: %v is not a well-formed tile type", t))
	}
	b.k.Lanes = SubgroupLanes
	tt := t
	return b.emit(Op{
		Kind: OpTileLoad, Type: Void, Tile: &tt,
		Args: [3]Value{buf, offset, stride}, Imm: boolImm(colMajor),
	})
}

// TileSplat is a tile every one of whose elements is v. It is how an
// accumulator is zeroed; the value is collective, so it is an op.
func (b *Builder) TileSplat(t TileType, v Value) Value {
	if !t.Valid() {
		panic(fmt.Sprintf("ir: TileSplat: %v is not a well-formed tile type", t))
	}
	b.k.Lanes = SubgroupLanes
	tt := t
	return b.emit(Op{Kind: OpTileSplat, Type: Void, Tile: &tt, Args: [3]Value{v}})
}

// TilePhi is a loop-carried collective value: an accumulator that survives
// from one trip of a Loop to the next, closed with SetPhi like any Phi. Without
// it a GEMM's k-loop would have to be unrolled or spill the accumulator. The
// init and the update must be tiles of the same type; validateTiles holds both.
func (b *Builder) TilePhi(init Value) Value {
	t := b.tileTypeOf(init)
	if t == nil {
		panic("ir: TilePhi: the initial value is not a tile")
	}
	tt := *t
	return b.emit(Op{Kind: OpPhi, Type: Void, Tile: &tt, Args: [3]Value{init}})
}

// SubgroupIndex is the index of the thread's subgroup within its workgroup,
// tid / SubgroupLanes. It is subgroup-uniform, so a collective tile's address
// may depend on it. laneDependent recognises exactly this shape (a Shr of the
// thread id by at least log2(SubgroupLanes)), so any other tid-derived offset
// is still refused.
func (b *Builder) SubgroupIndex() Value {
	b.k.Lanes = SubgroupLanes
	return b.Shr(U32, b.TID(), b.Const(U32, subgroupShift))
}

// subgroupShift is log2(SubgroupLanes).
const subgroupShift = 5

// TileMMA is D = A*B + C over collective tiles. The result has C's type, so a
// chain accumulates without a memory round trip.
func (b *Builder) TileMMA(a, bb, c Value) Value {
	ta, tbb, tc := b.tileTypeOf(a), b.tileTypeOf(bb), b.tileTypeOf(c)
	switch {
	case ta == nil || tbb == nil || tc == nil:
		panic("ir: TileMMA: every operand must be a tile value")
	case ta.Use != TileA || tbb.Use != TileB || tc.Use != TileAcc:
		panic(fmt.Sprintf("ir: TileMMA: roles are %s/%s/%s, want A/B/Acc", ta.Use, tbb.Use, tc.Use))
	case ta.Cols != tbb.Rows:
		panic(fmt.Sprintf("ir: TileMMA: A is %dx%d and B is %dx%d; K must agree",
			ta.Rows, ta.Cols, tbb.Rows, tbb.Cols))
	case ta.Rows != tc.Rows || tbb.Cols != tc.Cols:
		panic(fmt.Sprintf("ir: TileMMA: A*B is %dx%d and C is %dx%d",
			ta.Rows, tbb.Cols, tc.Rows, tc.Cols))
	}
	tt := *tc
	return b.emit(Op{Kind: OpTileMMA, Type: Void, Tile: &tt, Args: [3]Value{a, bb, c}})
}

// TileStore writes a tile back to memory. It is the only way out of a tile
// value: there is no element accessor, since the lane mapping is undefined.
func (b *Builder) TileStore(buf, offset, stride, tile Value, colMajor bool) {
	t := b.tileTypeOf(tile)
	if t == nil {
		panic("ir: TileStore: the stored value is not a tile")
	}
	tt := *t
	b.emit(Op{
		Kind: OpTileStore, Type: Void, Tile: &tt,
		Args: [3]Value{buf, offset, stride}, Vargs: []Value{tile}, Imm: boolImm(colMajor),
	})
}

// tileTypeOf is the type of a value if it is a tile, and nil if it is scalar.
func (b *Builder) tileTypeOf(v Value) *TileType {
	if v <= 0 || int(v) > len(b.k.Ops) {
		return nil
	}
	return b.k.Ops[v-1].Tile
}

// TileTypeOf is tileTypeOf for a finished kernel.
func TileTypeOf(k *Kernel, v Value) *TileType {
	if v <= 0 || int(v) > len(k.Ops) {
		return nil
	}
	return k.Ops[v-1].Tile
}

// IsTileOp reports whether a kind produces or consumes a collective value.
func IsTileOp(k Kind) bool {
	switch k {
	case OpTileLoad, OpTileSplat, OpTileMMA, OpTileStore:
		return true
	}
	return false
}

// ColMajor reports the layout an access op carries.
func ColMajor(o Op) bool { return o.Imm != 0 }

func boolImm(b bool) int64 {
	if b {
		return 1
	}
	return 0
}

// validateTiles enforces what a collective value may do. Above all, a tile must
// never reach a scalar op: its lane mapping is undefined, so that would lower
// and run and be silently wrong.
func (k *Kernel) validateTiles() error {
	tiles := 0
	for i, o := range k.Ops {
		if o.Kind == OpPhi && o.Tile != nil {
			// A loop-carried tile (TilePhi): its init and update are tiles of
			// exactly its type, and it is nothing else.
			if !o.Tile.Valid() {
				return fmt.Errorf("ir: op %d (phi): %v is not a well-formed tile type", i, *o.Tile)
			}
			for n := 0; n < 2; n++ {
				if a := TileTypeOf(k, o.Args[n]); a == nil || *a != *o.Tile {
					return fmt.Errorf("ir: op %d: a tile phi's %s (value %d) is not a %v",
						i, []string{"init", "update"}[n], o.Args[n], *o.Tile)
				}
			}
			tiles++
			continue
		}
		if o.Tile != nil {
			if !IsTileOp(o.Kind) {
				return fmt.Errorf("ir: op %d (%s) carries a tile type and is not a tile op", i, o.Kind)
			}
			if !o.Tile.Valid() {
				return fmt.Errorf("ir: op %d (%s): %v is not a well-formed tile type", i, o.Kind, *o.Tile)
			}
			tiles++
		} else if IsTileOp(o.Kind) {
			return fmt.Errorf("ir: op %d (%s) is a tile op with no tile type", i, o.Kind)
		}

		// A tile value may be used only by a tile op, in the position that
		// expects one.
		if !IsTileOp(o.Kind) {
			for n, a := range o.Args {
				if TileTypeOf(k, a) != nil {
					return fmt.Errorf("ir: op %d (%s) takes the collective value %d as scalar "+
						"argument %d; a tile has no lane mapping a scalar op could read",
						i, o.Kind, a, n)
				}
			}
			for n, a := range o.Vargs {
				if TileTypeOf(k, a) != nil {
					return fmt.Errorf("ir: op %d (%s) takes the collective value %d as scalar "+
						"operand %d", i, o.Kind, a, n)
				}
			}
			continue
		}

		switch o.Kind {
		case OpTileLoad, OpTileStore:
			// Args are {buffer, offset, stride}, none a tile. The buffer is a
			// parameter or a shared array. A shared array is declared in
			// 32-bit words and the view's offset and stride count the tile's
			// components, so an f16 tile reads two per word; a backend that
			// cannot address workgroup memory cooperatively refuses the op.
			pd := int(o.Args[0])
			if pd <= 0 || pd > len(k.Ops) || (k.Ops[pd-1].Kind != OpParam && k.Ops[pd-1].Kind != OpShared) {
				return fmt.Errorf("ir: op %d (%s) loads through value %d, which is neither a "+
					"parameter nor a shared array", i, o.Kind, o.Args[0])
			}
			for n, a := range o.Args {
				if TileTypeOf(k, a) != nil {
					return fmt.Errorf("ir: op %d (%s): the view's argument %d is a tile", i, o.Kind, n)
				}
			}
			// The address must be subgroup-uniform: every lane participates
			// and must agree on pointer and stride, so a lane-dependent
			// address is undefined behaviour and is refused.
			for n := 1; n < 3; n++ {
				if lane, why := k.laneDependent(o.Args[n], map[Value]bool{}); lane {
					return fmt.Errorf("ir: op %d (%s): the view's %s is lane-dependent (%s); "+
						"a collective access needs a subgroup-uniform address",
						i, o.Kind, []string{"", "offset", "stride"}[n], why)
				}
			}
			if o.Kind == OpTileStore {
				if len(o.Vargs) != 1 || TileTypeOf(k, o.Vargs[0]) == nil {
					return fmt.Errorf("ir: op %d (tile.store) stores something that is not a tile", i)
				}
			}
		case OpTileSplat:
			if TileTypeOf(k, o.Args[0]) != nil {
				return fmt.Errorf("ir: op %d (tile.splat) splats a tile", i)
			}
		case OpTileMMA:
			a, b, c := TileTypeOf(k, o.Args[0]), TileTypeOf(k, o.Args[1]), TileTypeOf(k, o.Args[2])
			if a == nil || b == nil || c == nil {
				return fmt.Errorf("ir: op %d (tile.mma) has a scalar operand", i)
			}
			if a.Use != TileA || b.Use != TileB || c.Use != TileAcc {
				return fmt.Errorf("ir: op %d (tile.mma) roles are %s/%s/%s, want A/B/Acc",
					i, a.Use, b.Use, c.Use)
			}
			if a.Cols != b.Rows || a.Rows != c.Rows || b.Cols != c.Cols {
				return fmt.Errorf("ir: op %d (tile.mma): A %dx%d, B %dx%d, C %dx%d do not compose",
					i, a.Rows, a.Cols, b.Rows, b.Cols, c.Rows, c.Cols)
			}
			if a.Elem != b.Elem {
				return fmt.Errorf("ir: op %d (tile.mma): operands are %s and %s", i, a.Elem, b.Elem)
			}
			if *o.Tile != *c {
				return fmt.Errorf("ir: op %d (tile.mma) result %v is not the accumulator's %v",
					i, *o.Tile, *c)
			}
		}
	}
	// A tile op implies a subgroup, so the width must be declared.
	if tiles > 0 && k.Lanes != SubgroupLanes {
		return fmt.Errorf("ir: %d tile op(s) and Lanes=%d; a collective value needs %d",
			tiles, k.Lanes, SubgroupLanes)
	}
	return nil
}

// LaneDependent reports whether v can differ between the lanes of a subgroup.
// False is a proof, by construction from the ops, that every lane holds the
// same v; true may be conservative.
func (k *Kernel) LaneDependent(v Value) bool {
	d, _ := k.laneDependent(v, map[Value]bool{})
	return d
}

// laneDependent reports whether v is derived from anything that differs between
// lanes, and names the first such thing. seen guards the back edge of a phi.
func (k *Kernel) laneDependent(v Value, seen map[Value]bool) (bool, string) {
	if v <= 0 || int(v) > len(k.Ops) || seen[v] {
		return false, ""
	}
	seen[v] = true
	o := k.Ops[v-1]
	if o.Kind == OpShr && o.Args[0] > 0 && k.Ops[o.Args[0]-1].Kind == OpTID &&
		o.Args[1] > 0 && k.Ops[o.Args[1]-1].Kind == OpConst && k.Ops[o.Args[1]-1].Imm >= subgroupShift {
		return false, "" // SubgroupIndex or coarser: one value per subgroup
	}
	switch o.Kind {
	case OpTID:
		return true, "tid"
	case OpShuffleXor:
		return true, "shuffle.xor"
	case OpShuffle:
		return true, "shuffle"
	case OpLaneID:
		return true, "laneid"
	case OpSubgroupReduce, OpSubgroupBroadcast:
		// Every lane of the subgroup receives the same value, whatever the
		// lanes held going in: that is what the two collectives are for.
		return false, ""
	}
	for _, a := range o.Args {
		if d, why := k.laneDependent(a, seen); d {
			return true, why
		}
	}
	for _, a := range o.Vargs {
		if d, why := k.laneDependent(a, seen); d {
			return true, why
		}
	}
	return false, ""
}

// ParamTile declares a buffer whose bytes are a tile's component type. A
// cooperative load reads its own component type from memory and does not
// convert, so storage and tile must agree.
func (b *Builder) ParamTile(name string, e TileElem) Value {
	b.k.Params = append(b.k.Params, Param{Name: name, Elem: U32, Tile: e, IsTile: true})
	return b.emit(Op{Kind: OpParam, Type: Ptr, Imm: int64(len(b.k.Params) - 1)})
}

// Bytes is the component's size in memory.
func (e TileElem) Bytes() int {
	switch e {
	case TileF16:
		return 2
	case TileS8:
		return 1
	}
	return 4
}

// Signed reports whether this element type carries a sign that a matrix
// instruction has to be told about. SPIR-V's OpCooperativeMatrixMulAddKHR
// reads every operand as unsigned when its operands mask is absent, which is
// harmless for floats and wrong for s8.
func (e TileElem) Signed() bool { return e == TileS8 || e == TileI32 }

// UsesTiles reports whether this kernel contains any cooperative-matrix op, so
// a backend can decline it before compiling rather than after computing.
func UsesTiles(k *Kernel) bool {
	for _, o := range k.Ops {
		if IsTileOp(o.Kind) {
			return true
		}
	}
	return false
}

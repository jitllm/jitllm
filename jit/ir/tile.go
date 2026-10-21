package ir

import "fmt"

type TileUse uint8

const (
	TileA TileUse = iota
	TileB
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

type TileType struct {
	Rows, Cols int
	Elem       TileElem
	Use        TileUse
}

func (t TileType) String() string {
	return fmt.Sprintf("tile<%dx%d %s %s>", t.Rows, t.Cols, t.Elem, t.Use)
}

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
	if t.Use == TileAcc && (t.Elem == TileF16 || t.Elem == TileS8) {
		return false
	}
	return true
}

func (b *Builder) TileLoad(t TileType, buf, offset, stride Value, colMajor bool) Value {
	if !t.Valid() {
		b.fail("ir: TileLoad: %v is not a well-formed tile type", t)
	}
	b.k.Lanes = SubgroupLanes
	tt := t
	return b.emit(Op{
		Kind: OpTileLoad, Type: Void, Tile: &tt,
		Args: [3]Value{buf, offset, stride}, Imm: boolImm(colMajor),
	})
}

func (b *Builder) TileSplat(t TileType, v Value) Value {
	if !t.Valid() {
		b.fail("ir: TileSplat: %v is not a well-formed tile type", t)
	}
	b.k.Lanes = SubgroupLanes
	tt := t
	return b.emit(Op{Kind: OpTileSplat, Type: Void, Tile: &tt, Args: [3]Value{v}})
}

func (b *Builder) TilePhi(init Value) Value {
	t := b.tileTypeOf(init)
	if t == nil {
		b.fail("ir: TilePhi: the initial value is not a tile")
	}
	tt := *t
	return b.emit(Op{Kind: OpPhi, Type: Void, Tile: &tt, Args: [3]Value{init}})
}

func (b *Builder) SubgroupIndex() Value {
	b.k.Lanes = SubgroupLanes
	return b.Shr(U32, b.TID(), b.Const(U32, subgroupShift))
}

const subgroupShift = 5

func (b *Builder) TileMMA(a, bb, c Value) Value {
	ta, tbb, tc := b.tileTypeOf(a), b.tileTypeOf(bb), b.tileTypeOf(c)
	switch {
	case ta == nil || tbb == nil || tc == nil:
		b.fail("ir: TileMMA: every operand must be a tile value")
	case ta.Use != TileA || tbb.Use != TileB || tc.Use != TileAcc:
		b.fail("ir: TileMMA: roles are %s/%s/%s, want A/B/Acc", ta.Use, tbb.Use, tc.Use)
	case ta.Cols != tbb.Rows:
		b.fail("ir: TileMMA: A is %dx%d and B is %dx%d; K must agree",
			ta.Rows, ta.Cols, tbb.Rows, tbb.Cols)
	case ta.Rows != tc.Rows || tbb.Cols != tc.Cols:
		b.fail("ir: TileMMA: A*B is %dx%d and C is %dx%d",
			ta.Rows, tbb.Cols, tc.Rows, tc.Cols)
	}
	tt := *tc
	return b.emit(Op{Kind: OpTileMMA, Type: Void, Tile: &tt, Args: [3]Value{a, bb, c}})
}

func (b *Builder) TileStore(buf, offset, stride, tile Value, colMajor bool) {
	t := b.tileTypeOf(tile)
	if t == nil {
		b.fail("ir: TileStore: the stored value is not a tile")
	}
	tt := *t
	b.emit(Op{
		Kind: OpTileStore, Type: Void, Tile: &tt,
		Args: [3]Value{buf, offset, stride}, Vargs: []Value{tile}, Imm: boolImm(colMajor),
	})
}

func (b *Builder) tileTypeOf(v Value) *TileType {
	if v <= 0 || int(v) > len(b.k.Ops) {
		return nil
	}
	return b.k.Ops[v-1].Tile
}

func TileTypeOf(k *Kernel, v Value) *TileType {
	if v <= 0 || int(v) > len(k.Ops) {
		return nil
	}
	return k.Ops[v-1].Tile
}

func IsTileOp(k Kind) bool {
	switch k {
	case OpTileLoad, OpTileSplat, OpTileMMA, OpTileStore:
		return true
	}
	return false
}

func ColMajor(o Op) bool { return o.Imm != 0 }

func boolImm(b bool) int64 {
	if b {
		return 1
	}
	return 0
}

func (k *Kernel) validateTiles() error {
	tiles := 0
	for i, o := range k.Ops {
		if o.Kind == OpPhi && o.Tile != nil {
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
	if tiles > 0 && k.Lanes != SubgroupLanes {
		return fmt.Errorf("ir: %d tile op(s) and Lanes=%d; a collective value needs %d",
			tiles, k.Lanes, SubgroupLanes)
	}
	return nil
}

func (k *Kernel) LaneDependent(v Value) bool {
	d, _ := k.laneDependent(v, map[Value]bool{})
	return d
}

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

func (b *Builder) ParamTile(name string, e TileElem) Value {
	b.k.Params = append(b.k.Params, Param{Name: name, Elem: U32, Tile: e, IsTile: true})
	return b.emit(Op{Kind: OpParam, Type: Ptr, Imm: int64(len(b.k.Params) - 1)})
}

func (e TileElem) Bytes() int {
	switch e {
	case TileF16:
		return 2
	case TileS8:
		return 1
	}
	return 4
}

func (e TileElem) Signed() bool { return e == TileS8 || e == TileI32 }

func UsesTiles(k *Kernel) bool {
	for _, o := range k.Ops {
		if IsTileOp(o.Kind) {
			return true
		}
	}
	return false
}

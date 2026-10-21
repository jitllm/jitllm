package ir

import "fmt"

type ReduceOp uint8

const (
	ReduceAdd ReduceOp = iota
	ReduceMax
	ReduceMin
	nReduce
)

func (r ReduceOp) String() string {
	switch r {
	case ReduceAdd:
		return "add"
	case ReduceMax:
		return "max"
	case ReduceMin:
		return "min"
	}
	return fmt.Sprintf("reduce(%d)", uint8(r))
}

func ReduceOf(o Op) ReduceOp { return ReduceOp(o.Imm) }

func (b *Builder) SubgroupReduce(t Type, x Value, op ReduceOp) Value {
	b.k.Lanes = SubgroupLanes
	return b.emit(Op{Kind: OpSubgroupReduce, Type: t, Args: [3]Value{x}, Imm: int64(op)})
}

func (b *Builder) SubgroupBroadcast(t Type, x, lane Value) Value {
	b.k.Lanes = SubgroupLanes
	return b.emit(Op{Kind: OpSubgroupBroadcast, Type: t, Args: [3]Value{x, lane}})
}

func (b *Builder) Shuffle(t Type, x, lane Value) Value {
	b.k.Lanes = SubgroupLanes
	return b.emit(Op{Kind: OpShuffle, Type: t, Args: [3]Value{x, lane}})
}

func (b *Builder) LaneID() Value {
	b.k.Lanes = SubgroupLanes
	return b.emit(Op{Kind: OpLaneID, Type: U32})
}

func (b *Builder) SubgroupSize() Value {
	return b.emit(Op{Kind: OpSubgroupSize, Type: U32})
}

func (k *Kernel) validateSubgroup(i int, o Op) error {
	switch o.Kind {
	case OpLaneID, OpSubgroupSize:
		if o.Type != U32 {
			return fmt.Errorf("ir: op %d (%s) yields u32, not %s", i, o.Kind, o.Type)
		}
		if o.Kind == OpSubgroupSize {
			return nil
		}
	default:
		switch o.Type {
		case F32, I32, U32:
		default:
			return fmt.Errorf("ir: op %d (%s) on a %s; the collectives move 32-bit scalars", i, o.Kind, o.Type)
		}
		if t := k.typeOf(o.Args[0]); t != o.Type {
			return fmt.Errorf("ir: op %d (%s) is %s and its operand (value %d) is %s",
				i, o.Kind, o.Type, o.Args[0], t)
		}
	}
	switch o.Kind {
	case OpSubgroupReduce:
		if o.Imm < 0 || o.Imm >= int64(nReduce) {
			return fmt.Errorf("ir: op %d: reduction %d is not a ReduceOp", i, o.Imm)
		}
	case OpSubgroupBroadcast, OpShuffle:
		if t := k.typeOf(o.Args[1]); t != U32 && t != I32 {
			return fmt.Errorf("ir: op %d (%s): the source lane (value %d) is %s, not an integer",
				i, o.Kind, o.Args[1], t)
		}
		if o.Kind == OpSubgroupBroadcast {
			if lane, why := k.laneDependent(o.Args[1], map[Value]bool{}); lane {
				return fmt.Errorf("ir: op %d (subgroup.broadcast): the source lane is lane-dependent "+
					"(%s); a per-lane source is a Shuffle", i, why)
			}
		}
	}
	if k.Group[0]%SubgroupLanes != 0 || k.Group[1] != 1 || k.Group[2] != 1 {
		return fmt.Errorf("ir: op %d (%s) needs a workgroup of 32n x 1 x 1 so that lane == "+
			"tid mod 32; this one is %v", i, o.Kind, k.Group)
	}
	return nil
}

func (k *Kernel) typeOf(v Value) Type {
	if v <= 0 || int(v) > len(k.Ops) {
		return Void
	}
	return k.Ops[v-1].Type
}

func (b *Builder) Prefetch(buf, idx Value) {
	b.emit(Op{Kind: OpPrefetch, Type: Void, Args: [3]Value{buf, idx}})
}

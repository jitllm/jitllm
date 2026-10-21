package ir

import "fmt"

// The subgroup collectives: a reduction, a broadcast, a shuffle by lane index,
// and the two queries that name a lane and the subgroup's width.
//
// Before these the IR had one cross-lane op, OpShuffleXor, and every kernel
// that reduced across a warp spelled the butterfly out itself. A reduction as
// one op lets each target use what it has -- SPIR-V's
// OpGroupNonUniform*(Reduce), MSL's simd_sum/simd_max, PTX's shfl butterfly --
// and a broadcast or an indexed shuffle is what a fused kernel needs to hand
// one lane's scalar (a row's scale, a block's maximum) to the rest without a
// round trip through workgroup memory.
//
// They are defined over the aligned block of SubgroupLanes (32) lanes a
// ShuffleXor is, so a kernel using any of them but OpSubgroupSize declares
// Lanes and runs only where a backend guarantees that width
// (backend.GuaranteedLanes). The order a reduction combines its lanes in is the
// target's: a float sum is a tree whose shape differs between backends, the
// same reduction-order band RULE 11c already allows between host and device.

// ReduceOp is what an OpSubgroupReduce combines with.
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

// ReduceOf is the ReduceOp an OpSubgroupReduce carries.
func ReduceOf(o Op) ReduceOp { return ReduceOp(o.Imm) }

// SubgroupReduce combines x across the 32 lanes of the thread's subgroup and
// returns the result in every lane.
func (b *Builder) SubgroupReduce(t Type, x Value, op ReduceOp) Value {
	b.k.Lanes = SubgroupLanes
	return b.emit(Op{Kind: OpSubgroupReduce, Type: t, Args: [3]Value{x}, Imm: int64(op)})
}

// SubgroupBroadcast returns x as it stands in lane `lane`, in every lane. The
// lane must be the same in every lane of the subgroup (a constant, or derived
// from the group index or another collective); Validate refuses one that is
// not, since a per-lane source is a Shuffle and some targets' broadcasts
// require uniformity. A lane past 31 reads lane & 31, as Shuffle does.
func (b *Builder) SubgroupBroadcast(t Type, x, lane Value) Value {
	b.k.Lanes = SubgroupLanes
	return b.emit(Op{Kind: OpSubgroupBroadcast, Type: t, Args: [3]Value{x, lane}})
}

// Shuffle returns x as it stands in lane (lane & 31) of the thread's subgroup.
// The source may differ from lane to lane. The mask is part of the op, so a
// source past the subgroup reads a lane inside it on every target rather than
// whatever each one leaves undefined.
func (b *Builder) Shuffle(t Type, x, lane Value) Value {
	b.k.Lanes = SubgroupLanes
	return b.emit(Op{Kind: OpShuffle, Type: t, Args: [3]Value{x, lane}})
}

// LaneID is the thread's index within its subgroup, 0..31.
func (b *Builder) LaneID() Value {
	b.k.Lanes = SubgroupLanes
	return b.emit(Op{Kind: OpLaneID, Type: U32})
}

// SubgroupSize is the width of the subgroup the thread's collectives act on.
// In a kernel that declares Lanes it is SubgroupLanes; in one that does not it
// is whatever width the device ran it at, which is the only way such a kernel
// can learn it. It reports the width and does not depend on it, so it does not
// by itself make a kernel width-dependent.
func (b *Builder) SubgroupSize() Value {
	return b.emit(Op{Kind: OpSubgroupSize, Type: U32})
}

// validateSubgroup checks one collective's operands and the workgroup it needs.
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
	// lane == tid mod 32 holds only for a workgroup of whole subgroups along x.
	if k.Group[0]%SubgroupLanes != 0 || k.Group[1] != 1 || k.Group[2] != 1 {
		return fmt.Errorf("ir: op %d (%s) needs a workgroup of 32n x 1 x 1 so that lane == "+
			"tid mod 32; this one is %v", i, o.Kind, k.Group)
	}
	return nil
}

// typeOf is a value's scalar type, Void for an invalid id.
func (k *Kernel) typeOf(v Value) Type {
	if v <= 0 || int(v) > len(k.Ops) {
		return Void
	}
	return k.Ops[v-1].Type
}

// Prefetch hints that element idx of the global buffer buf will be read
// soon: see OpPrefetch.
func (b *Builder) Prefetch(buf, idx Value) {
	b.emit(Op{Kind: OpPrefetch, Type: Void, Args: [3]Value{buf, idx}})
}

// Package ir is the composer's GPU kernel IR.
package ir

import (
	"fmt"
	"math"
)

type Type uint8

const (
	Void Type = iota
	I32
	U32
	F32
	Pred
	Ptr
)

func (t Type) String() string {
	return [...]string{"void", "i32", "u32", "f32", "pred", "ptr"}[t]
}

type Value uint32

type Kind uint8

const (
	OpInvalid Kind = iota
	OpConst
	OpParam
	OpTID
	OpCTAID
	OpNTID
	OpAdd
	OpSub
	OpMul
	OpDiv
	OpRem
	OpShl
	OpShr
	OpAnd
	OpXor
	OpMin
	OpMax
	OpExp
	OpSqrt
	OpCvtF32
	OpCvtF16H
	OpPackF16
	OpCvtI32
	OpBitcast
	OpLoad
	OpStore
	OpDot4
	OpFma
	OpLt
	OpSelect
	OpLoop
	OpEndLoop
	OpPhi
	OpShuffleXor
	OpShared
	OpBarrier
	OpMMA
	OpMMAGet
	OpTileLoad
	OpTileSplat
	OpTileMMA
	OpTileStore
	OpSubF16x2
	OpFmaF16x2
	OpLoadV
	OpLoadVGet
	OpStoreV
	OpFunnel
	OpSubgroupReduce
	OpSubgroupBroadcast
	OpShuffle
	OpLaneID
	OpSubgroupSize
	OpPrefetch
	OpGridBarrier
	OpDepTask
	OpDepEpoch
	OpDepWait
	OpDepSignal
	OpDepExit
	nKind
)

var kindName = [nKind]string{
	"invalid", "const", "param", "tid", "ctaid", "ntid",
	"add", "sub", "mul", "div", "rem", "shl", "shr", "and", "xor", "min", "max", "exp", "sqrt",
	"cvt.f32", "cvt.f16h", "packf16", "cvt.i32", "bitcast", "load", "store", "dot4", "fma", "lt",
	"select", "loop", "endloop", "phi", "shuffle.xor", "shared", "barrier", "mma", "mma.get",
	"tile.load", "tile.splat", "tile.mma", "tile.store", "sub.f16x2", "fma.f16x2",
	"loadv", "loadv.get", "storev", "funnel",
	"subgroup.reduce", "subgroup.broadcast", "shuffle", "laneid", "subgroup.size", "prefetch",
	"grid.barrier", "dep.task", "dep.epoch", "dep.wait", "dep.signal", "dep.exit",
}

func (k Kind) String() string { return kindName[k] }

type Op struct {
	Kind  Kind
	Type  Type
	Args  [3]Value
	Imm   int64
	Vargs []Value
	Tile  *TileType
}

type Param struct {
	Name   string
	Elem   Type
	Tile   TileElem
	IsTile bool
}

type Kernel struct {
	Name    string
	Params  []Param
	Ops     []Op
	Group   [3]int
	Lanes   int
	Comment string
}

const SubgroupLanes = 32

func WidthDependent(k Kind) bool {
	switch k {
	case OpShuffleXor, OpMMA, OpSubgroupReduce, OpSubgroupBroadcast, OpShuffle, OpLaneID:
		return true
	}
	return IsTileOp(k)
}

type Builder struct {
	k   Kernel
	err error
}

func (b *Builder) fail(format string, args ...any) {
	if b.err == nil {
		b.err = fmt.Errorf(format, args...)
	}
}

func New(name string, group [3]int) *Builder {
	return &Builder{k: Kernel{Name: name, Group: group}}
}

func (b *Builder) emit(o Op) Value {
	b.k.Ops = append(b.k.Ops, o)
	return Value(len(b.k.Ops))
}

func (b *Builder) Shared(name string, elem Type, n int) Value {
	_ = name
	return b.emit(Op{Kind: OpShared, Type: elem, Imm: int64(n)})
}

func (b *Builder) Barrier() {
	b.emit(Op{Kind: OpBarrier})
}

func (b *Builder) Param(name string, elem Type) Value {
	b.k.Params = append(b.k.Params, Param{Name: name, Elem: elem})
	return b.emit(Op{Kind: OpParam, Type: Ptr, Imm: int64(len(b.k.Params) - 1)})
}

func (b *Builder) Const(t Type, v int64) Value { return b.emit(Op{Kind: OpConst, Type: t, Imm: v}) }

func (b *Builder) IntConst(v Value) (int64, bool) {
	if v == 0 || int(v) > len(b.k.Ops) {
		return 0, false
	}
	o := b.k.Ops[v-1]
	return o.Imm, o.Kind == OpConst && (o.Type == U32 || o.Type == I32)
}

func (b *Builder) ConstF32(v float32) Value {
	return b.Const(F32, int64(math.Float32bits(v)))
}
func (b *Builder) TID() Value   { return b.emit(Op{Kind: OpTID, Type: U32}) }
func (b *Builder) CTAID() Value { return b.emit(Op{Kind: OpCTAID, Type: U32}) }
func (b *Builder) NTID() Value  { return b.emit(Op{Kind: OpNTID, Type: U32}) }

func (b *Builder) bin(k Kind, t Type, x, y Value) Value {
	return b.emit(Op{Kind: k, Type: t, Args: [3]Value{x, y}})
}

func (b *Builder) Add(t Type, x, y Value) Value { return b.bin(OpAdd, t, x, y) }
func (b *Builder) Min(t Type, x, y Value) Value { return b.bin(OpMin, t, x, y) }
func (b *Builder) Max(t Type, x, y Value) Value { return b.bin(OpMax, t, x, y) }

func (b *Builder) Exp(x Value) Value  { return b.un(OpExp, F32, x) }
func (b *Builder) Sqrt(x Value) Value { return b.un(OpSqrt, F32, x) }

func (b *Builder) un(k Kind, t Type, x Value) Value {
	return b.emit(Op{Kind: k, Type: t, Args: [3]Value{x}})
}
func (b *Builder) Shl(t Type, x, y Value) Value { return b.bin(OpShl, t, x, y) }
func (b *Builder) Shr(t Type, x, y Value) Value { return b.bin(OpShr, t, x, y) }
func (b *Builder) And(t Type, x, y Value) Value { return b.bin(OpAnd, t, x, y) }
func (b *Builder) Xor(t Type, x, y Value) Value { return b.bin(OpXor, t, x, y) }

func (b *Builder) Funnel(lo, hi, n Value) Value {
	return b.emit(Op{Kind: OpFunnel, Type: U32, Args: [3]Value{lo, hi, n}})
}
func (b *Builder) Sub(t Type, x, y Value) Value { return b.bin(OpSub, t, x, y) }
func (b *Builder) Mul(t Type, x, y Value) Value { return b.bin(OpMul, t, x, y) }
func (b *Builder) Div(t Type, x, y Value) Value { return b.bin(OpDiv, t, x, y) }
func (b *Builder) Rem(t Type, x, y Value) Value { return b.bin(OpRem, t, x, y) }
func (b *Builder) Lt(t Type, x, y Value) Value  { return b.bin(OpLt, Pred, x, y) }

func (b *Builder) Select(t Type, p, x, y Value) Value {
	return b.emit(Op{Kind: OpSelect, Type: t, Args: [3]Value{p, x, y}})
}

func (b *Builder) CvtF32(x Value) Value {
	return b.emit(Op{Kind: OpCvtF32, Type: F32, Args: [3]Value{x}})
}

func (b *Builder) CvtF16H(x Value) Value {
	return b.emit(Op{Kind: OpCvtF16H, Type: F32, Args: [3]Value{x}})
}

func (b *Builder) PackF16(lo, hi Value) Value {
	return b.emit(Op{Kind: OpPackF16, Type: U32, Args: [3]Value{lo, hi}})
}

func (b *Builder) SubF16x2(x, y Value) Value {
	return b.emit(Op{Kind: OpSubF16x2, Type: U32, Args: [3]Value{x, y}})
}

func (b *Builder) FmaF16x2(x, y, z Value) Value {
	return b.emit(Op{Kind: OpFmaF16x2, Type: U32, Args: [3]Value{x, y, z}})
}

func (b *Builder) CvtI32(x Value) Value {
	return b.emit(Op{Kind: OpCvtI32, Type: I32, Args: [3]Value{x}})
}

func (b *Builder) Bitcast(t Type, x Value) Value {
	return b.emit(Op{Kind: OpBitcast, Type: t, Args: [3]Value{x}})
}

func (b *Builder) Load(t Type, buf, idx Value, off int64) Value {
	return b.emit(Op{Kind: OpLoad, Type: t, Args: [3]Value{buf, idx}, Imm: off})
}

func (b *Builder) Store(buf, idx, v Value, off int64) {
	b.emit(Op{Kind: OpStore, Type: Void, Args: [3]Value{buf, idx, v}, Imm: off})
}

func (b *Builder) LoadV(t Type, buf, idx Value, off int64, n int) []Value {
	if n != 2 && n != 4 {
		b.fail("ir: LoadV of %d elements; 2 or 4", n)
	}
	v := b.emit(Op{Kind: OpLoadV, Type: t, Args: [3]Value{buf, idx}, Imm: off<<3 | int64(n)})
	out := []Value{v}
	for i := 1; i < n; i++ {
		out = append(out, b.emit(Op{Kind: OpLoadVGet, Type: t, Args: [3]Value{v}, Imm: int64(i)}))
	}
	return out
}

func (b *Builder) LdMatrix(buf, idx Value, off int64, n int) []Value {
	if n != 2 && n != 4 {
		b.fail("ir: LdMatrix of %d matrices; 2 or 4", n)
	}
	b.k.Lanes = SubgroupLanes
	v := b.emit(Op{Kind: OpLoadV, Type: U32, Args: [3]Value{buf, idx}, Imm: ldmMark | off<<3 | int64(n)})
	out := []Value{v}
	for i := 1; i < n; i++ {
		out = append(out, b.emit(Op{Kind: OpLoadVGet, Type: U32, Args: [3]Value{v}, Imm: int64(i)}))
	}
	return out
}

const ldmMark = int64(1) << 60

func IsLdMatrix(o Op) bool { return o.Kind == OpLoadV && o.Imm&ldmMark != 0 }

func (b *Builder) StoreV(buf, idx Value, off int64, vs ...Value) {
	if len(vs) != 2 && len(vs) != 4 {
		b.fail("ir: StoreV of %d elements; 2 or 4", len(vs))
	}
	b.emit(Op{Kind: OpStoreV, Type: Void, Args: [3]Value{buf, idx}, Imm: off<<3 | int64(len(vs)),
		Vargs: append([]Value{}, vs...)})
}

func VecOf(o Op) (off int64, n int) { return (o.Imm &^ ldmMark) >> 3, int(o.Imm & 7) }

func (b *Builder) Dot4(a, x, acc Value) Value {
	return b.emit(Op{Kind: OpDot4, Type: I32, Args: [3]Value{a, x, acc}})
}

func (b *Builder) Dot4Bounded(a, x, acc Value) Value {
	return b.emit(Op{Kind: OpDot4, Type: I32, Args: [3]Value{a, x, acc}, Imm: 1})
}

func (b *Builder) Fma(a, x, c Value) Value {
	return b.emit(Op{Kind: OpFma, Type: F32, Args: [3]Value{a, x, c}})
}

func (b *Builder) Loop(trip int64) { b.emit(Op{Kind: OpLoop, Type: Void, Imm: trip}) }

func (b *Builder) LoopN(n Value) {
	b.emit(Op{Kind: OpLoop, Type: Void, Imm: -1, Args: [3]Value{n}})
}

func (b *Builder) Phi(t Type, init Value) Value {
	return b.emit(Op{Kind: OpPhi, Type: t, Args: [3]Value{init}})
}

func (b *Builder) SetPhi(phi, next Value) {
	b.k.Ops[phi-1].Args[1] = next
}

func (b *Builder) EndLoop() { b.emit(Op{Kind: OpEndLoop, Type: Void}) }

func (b *Builder) ShuffleXor(t Type, x Value, mask int64) Value {
	b.k.Lanes = SubgroupLanes
	return b.emit(Op{Kind: OpShuffleXor, Type: t, Args: [3]Value{x}, Imm: mask})
}

func (b *Builder) Done() (*Kernel, error) {
	if b.err != nil {
		return nil, b.err
	}
	return &b.k, nil
}

func (k *Kernel) Validate() error {
	if err := k.validateTiles(); err != nil {
		return err
	}
	if err := k.validateGrid(); err != nil {
		return err
	}
	if err := k.validateDeps(); err != nil {
		return err
	}
	depth := 0
	var loops []int
	for i, o := range k.Ops {
		for n, a := range o.Args {
			if o.Kind == OpPhi && n == 1 {
				continue
			}
			if a != 0 && int(a) > i {
				return fmt.Errorf("ir: op %d (%s) uses value %d defined later", i, o.Kind, a)
			}
		}
		for _, a := range o.Vargs {
			if a != 0 && int(a) > i {
				return fmt.Errorf("ir: op %d (%s) uses value %d defined later", i, o.Kind, a)
			}
		}
		switch o.Kind {
		case OpMMA:
			sh := MMAShapeOf(o)
			na, nb, nc := sh.Frags()
			if !sh.Valid() || len(o.Vargs) != na+nb+nc {
				return fmt.Errorf("ir: op %d: MMA %dx%dx%d with %d fragments", i, sh.M, sh.N, sh.K, len(o.Vargs))
			}
			if k.Group[0]%32 != 0 || k.Group[1] != 1 || k.Group[2] != 1 {
				return fmt.Errorf("ir: op %d: an MMA kernel needs a workgroup of 32n x 1 x 1 "+
					"so that lane == tid mod 32; this one is %v", i, k.Group)
			}
			if depth > 0 {
				_ = depth
			}
		case OpLoadV, OpStoreV:
			if _, n := VecOf(o); n != 2 && n != 4 || (o.Kind == OpStoreV && len(o.Vargs) != n) {
				return fmt.Errorf("ir: op %d: a %s of %d elements", i, o.Kind, n)
			}
			if o.Kind == OpStoreV && o.Imm&ldmMark != 0 {
				return fmt.Errorf("ir: op %d: a StoreV marked ldmatrix", i)
			}
			if IsLdMatrix(o) && (o.Args[0] == 0 || k.Ops[o.Args[0]-1].Kind != OpShared || o.Type != U32) {
				return fmt.Errorf("ir: op %d: ldmatrix reads a u32 shared buffer", i)
			}
		case OpLoadVGet:
			if o.Args[0] == 0 || k.Ops[o.Args[0]-1].Kind != OpLoadV {
				return fmt.Errorf("ir: op %d: LoadVGet does not name a LoadV", i)
			}
			if _, n := VecOf(k.Ops[o.Args[0]-1]); o.Imm < 1 || int(o.Imm) >= n {
				return fmt.Errorf("ir: op %d: LoadVGet component %d out of range", i, o.Imm)
			}
		case OpMMAGet:
			if o.Args[0] == 0 || k.Ops[o.Args[0]-1].Kind != OpMMA {
				return fmt.Errorf("ir: op %d: MMAGet does not name an MMA", i)
			}
			_, _, nc := MMAShapeOf(k.Ops[o.Args[0]-1]).Frags()
			if o.Imm < 1 || int(o.Imm) >= nc {
				return fmt.Errorf("ir: op %d: MMAGet component %d out of range", i, o.Imm)
			}
		case OpShared:
			if depth > 0 {
				return fmt.Errorf("ir: op %d: Shared declared inside a loop; it must precede every Loop", i)
			}
		case OpLoop:
			if o.Imm < 0 && o.Args[0] == 0 {
				return fmt.Errorf("ir: op %d: LoopN has no count value", i)
			}
			depth++
			loops = append(loops, i)
		case OpEndLoop:
			depth--
			if len(loops) > 0 {
				loops = loops[:len(loops)-1]
			}
			if depth < 0 {
				return fmt.Errorf("ir: op %d: EndLoop without Loop", i)
			}
		case OpSubgroupReduce, OpSubgroupBroadcast, OpShuffle, OpLaneID, OpSubgroupSize:
			if err := k.validateSubgroup(i, o); err != nil {
				return err
			}
		case OpShuffleXor:
			if o.Imm < 1 || o.Imm > 31 {
				return fmt.Errorf("ir: op %d: shuffle mask %d is outside [1,31]; a wider mask "+
					"reads a lane that may be in another subgroup", i, o.Imm)
			}
			if k.Group[0]%32 != 0 || k.Group[1] != 1 || k.Group[2] != 1 {
				return fmt.Errorf("ir: op %d: a shuffling kernel needs a workgroup of "+
					"32n x 1 x 1 so that lane == tid mod 32; this one is %v", i, k.Group)
			}
		case OpPhi:
			if o.Args[1] == 0 {
				return fmt.Errorf("ir: op %d: Phi has no update -- SetPhi was never called", i)
			}
			if len(loops) > 0 && int(o.Args[0]) > loops[len(loops)-1] {
				return fmt.Errorf("ir: op %d: Phi's init (value %d) is defined inside the loop "+
					"opened at op %d; hoist it before the Loop", i, o.Args[0], loops[len(loops)-1])
			}
		}
	}
	for i, o := range k.Ops {
		var t Type
		switch o.Kind {
		case OpLoad, OpLoadV:
			t = o.Type
		case OpStore, OpStoreV:
			if o.Args[2] == 0 {
				continue
			}
			t = k.Ops[o.Args[2]-1].Type
		default:
			continue
		}
		if p := k.paramOf(o.Args[0]); p >= 0 && k.Params[p].Elem != t {
			return fmt.Errorf("ir: %s: op %d accesses %s parameter %q as %s; bitcast the value instead",
				k.Name, i+1, k.Params[p].Elem, k.Params[p].Name, t)
		}
	}
	var loaded, stored [maxParams]bool
	for _, o := range k.Ops {
		switch o.Kind {
		case OpGridBarrier, OpDepTask:
			for i := range loaded {
				if loaded[i] && stored[i] {
					return fmt.Errorf("ir: param %d is both read and written; surplus threads clamp to "+
						"the last item and would apply the update twice -- make the kernel out of place", i)
				}
			}
			loaded, stored = [maxParams]bool{}, [maxParams]bool{}
		case OpLoad, OpLoadV:
			if p := k.paramOf(o.Args[0]); p >= 0 {
				loaded[p] = true
			}
		case OpStore, OpStoreV:
			if p := k.paramOf(o.Args[0]); p >= 0 {
				stored[p] = true
			}
		}
	}
	for i := range loaded {
		if loaded[i] && stored[i] {
			return fmt.Errorf("ir: param %d is both read and written; surplus threads clamp to "+
				"the last item and would apply the update twice -- make the kernel out of place", i)
		}
	}

	for i, o := range k.Ops {
		if WidthDependent(o.Kind) && k.Lanes != SubgroupLanes {
			return fmt.Errorf("ir: op %d (%s) is defined over %d lanes and the kernel declares "+
				"Lanes=%d; a kernel whose answer depends on the subgroup width must say so, or "+
				"no backend can refuse to run it on a device that will not guarantee one",
				i, o.Kind, SubgroupLanes, k.Lanes)
		}
	}
	if k.Lanes != 0 {
		used := false
		for _, o := range k.Ops {
			used = used || WidthDependent(o.Kind)
		}
		if !used {
			return fmt.Errorf("ir: the kernel declares Lanes=%d and contains no op whose result "+
				"depends on the subgroup width; a stale declaration costs a device that could "+
				"run it", k.Lanes)
		}
	}

	if depth != 0 {
		return fmt.Errorf("ir: %d loop(s) left open", depth)
	}
	return nil
}

const maxParams = 64

func (k *Kernel) paramOf(v Value) int {
	if v == 0 || int(v) > len(k.Ops) {
		return -1
	}
	o := k.Ops[v-1]
	if o.Kind != OpParam || o.Imm < 0 || o.Imm >= maxParams {
		return -1
	}
	return int(o.Imm)
}

type MMAShape struct {
	M, N, K int
	Kind    MMAKind
}

type MMAKind uint8

const (
	MMAS8 MMAKind = iota
	MMAF16
)

var mmaShapes = []MMAShape{
	{16, 8, 16, MMAS8}, {16, 8, 32, MMAS8},
	{16, 8, 8, MMAF16}, {16, 8, 16, MMAF16},
	MMAVolta(), MMAS8Turing(),
}

func MMAS8Turing() MMAShape { return MMAShape{8, 8, 16, MMAS8} }

func MMAVolta() MMAShape { return MMAShape{8, 8, 4, MMAF16} }

func (s MMAShape) Frags() (a, bb, c int) {
	if s == MMAVolta() {
		return 2, 2, 8
	}
	per := 4
	if s.Kind == MMAF16 {
		per = 2
	}
	return s.M * s.K / (32 * per), s.K * s.N / (32 * per), s.M * s.N / 32
}

func (s MMAShape) Elem() Type {
	if s.Kind == MMAF16 {
		return F32
	}
	return I32
}

func (s MMAShape) Valid() bool {
	for _, x := range mmaShapes {
		if x == s {
			return true
		}
	}
	return false
}

func (b *Builder) MMA(sh MMAShape, a, bb, c []Value) []Value {
	na, nb, nc := sh.Frags()
	if !sh.Valid() || len(a) != na || len(bb) != nb || len(c) != nc {
		b.fail("ir: MMA %dx%dx%d wants %d/%d/%d fragments, got %d/%d/%d",
			sh.M, sh.N, sh.K, na, nb, nc, len(a), len(bb), len(c))
	}
	b.k.Lanes = SubgroupLanes
	v := b.emit(Op{
		Kind:  OpMMA,
		Type:  sh.Elem(),
		Imm:   int64(sh.Kind)<<24 | int64(sh.M)<<16 | int64(sh.N)<<8 | int64(sh.K),
		Vargs: append(append(append([]Value{}, a...), bb...), c...),
	})
	out := make([]Value, nc)
	out[0] = v
	for i := 1; i < nc; i++ {
		out[i] = b.emit(Op{Kind: OpMMAGet, Type: sh.Elem(), Args: [3]Value{v}, Imm: int64(i)})
	}
	return out
}

func MMAShapeOf(o Op) MMAShape {
	return MMAShape{M: int(o.Imm>>16) & 0xFF, N: int(o.Imm>>8) & 0xFF, K: int(o.Imm) & 0xFF,
		Kind: MMAKind(o.Imm >> 24)}
}

func (k *Kernel) WrittenParams() uint64 {
	if len(k.Params) > 64 {
		return ^uint64(0)
	}
	var m uint64
	for _, o := range k.Ops {
		if o.Kind != OpStore && o.Kind != OpStoreV && o.Kind != OpTileStore {
			continue
		}
		b := o.Args[0]
		if b == 0 || int(b) > len(k.Ops) || k.Ops[b-1].Kind != OpParam {
			if b != 0 && int(b) <= len(k.Ops) && k.Ops[b-1].Kind == OpShared {
				continue
			}
			return ^uint64(0)
		}
		m |= 1 << uint(k.Ops[b-1].Imm)
	}
	return m
}

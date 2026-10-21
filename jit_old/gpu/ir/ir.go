// Package ir is a very small typed kernel IR for NN compute.
//
// It exists for the same reason the CPU emitters do: six quantization formats
// times several kernel shapes would otherwise be a dozen hand-written PTX
// strings to keep in step. It is not a shader compiler: no parser, no general
// CFG, no optimizer. The device driver owns instruction selection, register
// allocation and scheduling on every target.
//
// What a generator does own is source-level structure such as the register
// tile, which swings int8 GEMM throughput by more than an order of magnitude. Anything
// not needed to express those kernels is absent on purpose.
package ir

import (
	"fmt"
	"math"
)

// Type is the value type. Kernels load packed bytes as u32, accumulate in i32,
// scale in f32 and branch on a predicate. There is no i8 type: four packed
// bytes in a u32 is what dp4a, OpSDotKHR and SDOT all consume.
type Type uint8

const (
	Void Type = iota
	I32
	U32
	F32
	Pred
	// Ptr is a device address. It is distinct from a wide integer because
	// walking cursors are loop-carried pointers, and a Phi given a 32-bit
	// register would truncate them.
	Ptr
)

func (t Type) String() string {
	return [...]string{"void", "i32", "u32", "f32", "pred", "ptr"}[t]
}

// Value is an SSA value id. Zero is invalid, so a zero-valued struct field is
// caught rather than silently aliasing value 0.
type Value uint32

// Kind is the operation. About thirty, which is the whole instruction set.
type Kind uint8

const (
	OpInvalid Kind = iota
	OpConst        // Imm -> Type
	OpParam        // Imm = parameter index, yields a pointer
	OpTID          // thread index within the group
	OpCTAID        // group index
	OpNTID         // group size
	OpAdd
	OpSub
	OpMul
	OpDiv
	OpRem
	OpShl
	// OpShr is a logical right shift, never arithmetic: it is used on packed
	// quantized weights, where sign extension would silently corrupt the top
	// byte.
	OpShr
	OpAnd
	// OpXor centers quantized weights: a nibble q in 0..15 becomes the signed
	// byte q-biasK via (q + (0x80-biasK)) ^ 0x80808080, lane-safe because no
	// byte carries.
	OpXor
	OpMin
	OpMax
	// OpExp and OpSqrt are the backends' own transcendentals (SiLU and softmax
	// want exp, RMSNorm wants sqrt). A float32 device path cannot match the
	// host bit for bit anyway, so the gate is NMSE on the logits.
	OpExp
	OpSqrt
	OpCvtF32 // i32 -> f32
	// OpCvtF16H decodes the low 16 bits of a u32 as binary16 and yields f32.
	// The high half is reached by shifting first, which keeps the op unary.
	// It lets f16 scales stay f16 on the device instead of being widened.
	OpCvtF16H
	// OpPackF16 rounds two f32 to binary16 and packs them into one u32, Args[0]
	// low and Args[1] high: the inverse of OpCvtF16H. It serves the f16 KV
	// cache; an op rather than an f16 type keeps the change to three lowerings.
	OpPackF16
	// OpCvtI32 truncates an f32 toward zero. OpBitcast reinterprets 32 bits
	// between u32 and f32 without converting. Together they let a kernel build
	// an exact exp (range reduction, polynomial, 2^n written into the exponent
	// field) instead of a backend's approximate one.
	OpCvtI32
	OpBitcast
	// Access is by element index, not byte pointer: SPIR-V's Logical
	// addressing model and Metal have no raw pointers, and the cost to PTX is
	// one mad.wide.u32 per access.
	OpLoad  // Args[0] = buffer, Args[1] = element index, Imm = extra elements
	OpStore // Args[0] = buffer, Args[1] = index, Args[2] = value, Imm = extra
	OpDot4  // Args[0], Args[1] packed 4x int8; Args[2] accumulator -> i32.
	// Imm == 1 asserts both operands lie in [-127,127]; see Dot4Bounded.
	OpFma // Args[0]*Args[1] + Args[2] -> f32
	OpLt  // -> Pred
	// OpSelect picks Args[1] where Args[0] (a Pred) holds and Args[2]
	// otherwise. It is data flow, not a branch: both arms are evaluated and
	// nothing diverges, which is why it is allowed in an IR with no
	// conditional execution outside a loop.
	OpSelect
	OpLoop // loop head; Imm = trip count, or -1 with Args[0] = a runtime count
	OpEndLoop
	OpPhi // loop-carried value: Args[0] = initial, Args[1] = updated
	// OpShuffleXor reads Args[0] from the lane whose index is (lane ^ Imm),
	// within this thread's group of 32 consecutive lanes. It is the IR's
	// register-level cross-thread primitive (reductions without a second
	// kernel).
	//
	// The mask is an immediate in [1,31], so lane ^ mask cannot leave the
	// aligned 32-lane block; that makes one lowering correct on a 32-wide warp,
	// subgroup or simdgroup. It is not portable to a narrower subgroup, so the
	// width is declared (Kernel.Lanes) and a backend must guarantee it
	// (GuaranteedLanes(32)) before the kernel is selected; Vulkan pins the
	// width at pipeline creation. VkPhysicalDeviceSubgroupProperties.subgroupSize
	// is only the default width, not a promise: an Intel integrated GPU reports 32 and runs
	// some shaders at 16, and an out-of-subgroup shuffle is undefined rather
	// than reliably wrong. See docs/engineering-history/gpu-kernels.md.
	OpShuffleXor
	// OpShared declares a workgroup-local array and yields a buffer value that
	// Load and Store take exactly as they take a Param. Imm is the element
	// count; Type is the element type. It exists for tiled GEMMs, which stage
	// a weight tile for the whole thread block.
	OpShared
	// OpBarrier synchronises the workgroup and orders shared accesses across
	// it. Every thread of the group must reach it, so it may not appear under
	// any construct some threads skip. The IR has no conditional execution
	// outside a loop, so that holds by construction.
	OpBarrier
	// OpMMA is a warp-level integer matrix multiply-accumulate: the 32 lanes
	// of one warp compute D = A*B + C over an MxNxK int8 tile, each lane
	// holding a fixed slice of every fragment. The fragment-to-lane contract
	// is the kernel's to honour, so the shape is IR and the encoding is the
	// lowerer's. Because the packing is chosen at load time, the fragments can
	// be the packed layout itself (kernels.PackWeights) with no shared-memory
	// staging.
	//
	// Imm packs the shape as m<<16 | n<<8 | k. Vargs is A fragments, then B,
	// then C; the result is component 0 and OpMMAGet extracts the rest.
	OpMMA
	// OpMMAGet extracts component Imm of an OpMMA result (Args[0]).
	OpMMAGet
	// OpTileLoad, OpTileSplat, OpTileMMA and OpTileStore are the collective
	// tile form of a matrix multiply, beside OpMMA rather than replacing it.
	// With OpMMA the kernel owns the lane layout (MMAShape.Frags); with a
	// cooperative matrix the driver owns it and the value is loaded from
	// memory with a stride. The two are not expressible as each other. The
	// shapes also differ per device, so support is a capability list rather
	// than a feature bit.
	OpTileLoad
	OpTileSplat
	OpTileMMA
	OpTileStore
	// OpSubF16x2 and OpFmaF16x2 are packed binary16 arithmetic on two halves
	// in one u32: a-b, and a*b+c rounded once. They serve sm_70's tensor-core
	// matvec (kernels.MatVecMMA70). Only PTX lowers them.
	OpSubF16x2
	OpFmaF16x2
	// OpLoadV reads n (2 or 4) consecutive elements (idx+off .. idx+off+n-1)
	// in one instruction; OpLoadVGet extracts component Imm (1..n-1), the op
	// itself being component 0, as with OpMMA/OpMMAGet. OpStoreV writes its
	// Vargs to the same span. Imm packs off<<3 | n, and ldmMark marks an
	// OpLoadV that is a warp's ldmatrix (Builder.LdMatrix).
	//
	// The address must be aligned to the whole vector (8 or 16 bytes), and
	// that is the kernel's to guarantee: PTX faults on a misaligned vector
	// access. Only PTX lowers them.
	OpLoadV
	OpLoadVGet
	OpStoreV
	// OpFunnel is (Args[1]:Args[0]) >> Args[2], the low 32 bits of a 64-bit
	// right shift of the pair, Args[2] in [0,31]. PTX has it as shf.r; SPIR-V
	// and MSL spell it as shifts. It serves the SC stream's 12-bit pairs that
	// straddle two words (kernels.scStreamPair).
	OpFunnel
	// The subgroup collectives: see subgroup.go. OpSubgroupReduce combines
	// Args[0] across the 32 lanes and leaves the result in every lane (Imm is
	// the ReduceOp); OpSubgroupBroadcast reads Args[0] from lane Args[1], a
	// subgroup-uniform value; OpShuffle reads it from lane Args[1] & 31, which
	// may differ per lane; OpLaneID is the lane's index in its subgroup and
	// OpSubgroupSize the subgroup's width.
	OpSubgroupReduce
	OpSubgroupBroadcast
	OpShuffle
	OpLaneID
	OpSubgroupSize
	// OpPrefetch asks for the line holding element Args[1] of the global
	// buffer Args[0] to be brought into the cache ahead of a load: a hint
	// with no result and no effect on any value, so a backend with no such
	// instruction emits nothing. PTX lowers it as prefetch.global.L2. It lets
	// a kernel start its weight stream before a prologue whose barriers would
	// otherwise hold it back (kernels.ActIn).
	OpPrefetch
	// OpGridBarrier synchronises every workgroup of the launch through the
	// sync buffer Args[0]; see gridsync.go.
	OpGridBarrier
	// The dependency counters (depsync.go): a task's start, the epoch read,
	// a wait on a counter, a signal on one and the launch's exit.
	OpDepTask
	OpDepEpoch
	OpDepWait
	OpDepSignal
	OpDepExit
	nKind
)

// kindName has one name per Kind, in enum order. A name missing from the
// middle would shift every later name, so TestKindNamesLineUp checks the
// alignment and not just the length.
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

// Op is one instruction.
type Op struct {
	Kind Kind
	Type Type
	Args [3]Value
	Imm  int64
	// Vargs is the operand list for ops that do not fit in three (OpMMA's
	// fragments, OpStoreV). Every other op leaves it nil.
	Vargs []Value
	// Tile is the complete type of the collective value this op produces or
	// stores, nil for every other op. It is a field rather than bits in Imm so
	// no lowerer has to reconstruct it.
	Tile *TileType
}

// Param is a kernel parameter. Only buffers: every scalar the kernels need is
// a shape constant, and shape constants are baked, not passed.
type Param struct {
	Name string
	Elem Type // element type of the buffer
	// Tile is the buffer's component type when it holds tile storage, and
	// IsTile says whether it does. A cooperative load reads f16 or int8
	// straight from memory, which Elem cannot name. Scalar ops stay 32-bit.
	Tile   TileElem
	IsTile bool
}

// Kernel is a complete compute kernel.
type Kernel struct {
	Name   string
	Params []Param
	Ops    []Op
	Group  [3]int // workgroup size
	// Lanes is the subgroup width this kernel's arithmetic depends on, or 0
	// when its result is independent of subgroup grouping. The Builder sets it
	// for ShuffleXor, MMA and the tile ops; Validate checks it in both
	// directions, and a kernel with Lanes != 0 only runs on a device that
	// guarantees that width (backend.GuaranteedLanes).
	Lanes   int
	Comment string
}

// SubgroupLanes is the one width this IR's cross-lane ops are defined at: a
// CUDA warp, an Apple simdgroup, and the aligned 32-lane block a ShuffleXor
// mask in [1,31] cannot leave.
const SubgroupLanes = 32

// WidthDependent reports whether an op's result depends on which lanes share
// a subgroup. It is a function so the class has one definition a test can
// sweep. The tile ops belong to it because a cooperative matrix is distributed
// across exactly one subgroup. OpShared and OpBarrier are workgroup-scoped and
// do not.
func WidthDependent(k Kind) bool {
	switch k {
	case OpShuffleXor, OpMMA, OpSubgroupReduce, OpSubgroupBroadcast, OpShuffle, OpLaneID:
		return true
	}
	return IsTileOp(k)
}

// Builder constructs a Kernel. Values are handed out, never named by the
// caller, so register aliasing mistakes cannot be written.
type Builder struct {
	k Kernel
}

func New(name string, group [3]int) *Builder {
	return &Builder{k: Kernel{Name: name, Group: group}}
}

func (b *Builder) emit(o Op) Value {
	b.k.Ops = append(b.k.Ops, o)
	return Value(len(b.k.Ops)) // 1-based: 0 is the invalid value
}

// Shared declares a workgroup-local array of n elements and returns a buffer
// value usable with Load and Store. The size is a compile-time constant on all
// three targets.
func (b *Builder) Shared(name string, elem Type, n int) Value {
	_ = name // the lowerers name it by value index; see each backend's emit
	return b.emit(Op{Kind: OpShared, Type: elem, Imm: int64(n)})
}

// Barrier synchronises the workgroup. Every thread must reach it.
func (b *Builder) Barrier() {
	b.emit(Op{Kind: OpBarrier})
}

// Param declares a buffer parameter and returns its pointer value.
func (b *Builder) Param(name string, elem Type) Value {
	b.k.Params = append(b.k.Params, Param{Name: name, Elem: elem})
	return b.emit(Op{Kind: OpParam, Type: Ptr, Imm: int64(len(b.k.Params) - 1)})
}

// Const makes a constant. For F32, Imm is the IEEE-754 BIT PATTERN, not the
// value -- use ConstF32 rather than writing one by hand.
func (b *Builder) Const(t Type, v int64) Value { return b.emit(Op{Kind: OpConst, Type: t, Imm: v}) }

// IntConst is v's value where v is an integer constant.
func (b *Builder) IntConst(v Value) (int64, bool) {
	if v == 0 || int(v) > len(b.k.Ops) {
		return 0, false
	}
	o := b.k.Ops[v-1]
	return o.Imm, o.Kind == OpConst && (o.Type == U32 || o.Type == I32)
}

// ConstF32 makes a float constant from its value. An F32 Const's Imm is the
// bit pattern on every backend; use this rather than writing one by hand.
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

// Funnel is (hi:lo) >> n, n in [0,31]: see OpFunnel.
func (b *Builder) Funnel(lo, hi, n Value) Value {
	return b.emit(Op{Kind: OpFunnel, Type: U32, Args: [3]Value{lo, hi, n}})
}
func (b *Builder) Sub(t Type, x, y Value) Value { return b.bin(OpSub, t, x, y) }
func (b *Builder) Mul(t Type, x, y Value) Value { return b.bin(OpMul, t, x, y) }
func (b *Builder) Div(t Type, x, y Value) Value { return b.bin(OpDiv, t, x, y) }
func (b *Builder) Rem(t Type, x, y Value) Value { return b.bin(OpRem, t, x, y) }
func (b *Builder) Lt(t Type, x, y Value) Value  { return b.bin(OpLt, Pred, x, y) }

// Select yields x where p holds and y otherwise. Both are evaluated.
func (b *Builder) Select(t Type, p, x, y Value) Value {
	return b.emit(Op{Kind: OpSelect, Type: t, Args: [3]Value{p, x, y}})
}

func (b *Builder) CvtF32(x Value) Value {
	return b.emit(Op{Kind: OpCvtF32, Type: F32, Args: [3]Value{x}})
}

// CvtF16H decodes the low half of x as a binary16.
func (b *Builder) CvtF16H(x Value) Value {
	return b.emit(Op{Kind: OpCvtF16H, Type: F32, Args: [3]Value{x}})
}

// PackF16 rounds a and b to binary16 and packs them into one u32, a low.
func (b *Builder) PackF16(lo, hi Value) Value {
	return b.emit(Op{Kind: OpPackF16, Type: U32, Args: [3]Value{lo, hi}})
}

// SubF16x2 is a-b on two packed binary16 halves.
func (b *Builder) SubF16x2(x, y Value) Value {
	return b.emit(Op{Kind: OpSubF16x2, Type: U32, Args: [3]Value{x, y}})
}

// FmaF16x2 is a*b+c on two packed binary16 halves, rounded once.
func (b *Builder) FmaF16x2(x, y, z Value) Value {
	return b.emit(Op{Kind: OpFmaF16x2, Type: U32, Args: [3]Value{x, y, z}})
}

// CvtI32 truncates an f32 toward zero.
func (b *Builder) CvtI32(x Value) Value {
	return b.emit(Op{Kind: OpCvtI32, Type: I32, Args: [3]Value{x}})
}

// Bitcast reinterprets 32 bits as type t without converting.
func (b *Builder) Bitcast(t Type, x Value) Value {
	return b.emit(Op{Kind: OpBitcast, Type: t, Args: [3]Value{x}})
}

// Load reads element (idx + off) of a buffer.
func (b *Builder) Load(t Type, buf, idx Value, off int64) Value {
	return b.emit(Op{Kind: OpLoad, Type: t, Args: [3]Value{buf, idx}, Imm: off})
}

// Store writes v to element (idx + off) of a buffer.
func (b *Builder) Store(buf, idx, v Value, off int64) {
	b.emit(Op{Kind: OpStore, Type: Void, Args: [3]Value{buf, idx, v}, Imm: off})
}

// LoadV reads elements idx+off .. idx+off+n-1 of a buffer in one access, n 2
// or 4. The address must be aligned to 4n bytes; see OpLoadV.
func (b *Builder) LoadV(t Type, buf, idx Value, off int64, n int) []Value {
	if n != 2 && n != 4 {
		panic(fmt.Sprintf("ir: LoadV of %d elements; 2 or 4", n))
	}
	v := b.emit(Op{Kind: OpLoadV, Type: t, Args: [3]Value{buf, idx}, Imm: off<<3 | int64(n)})
	out := []Value{v}
	for i := 1; i < n; i++ {
		out = append(out, b.emit(Op{Kind: OpLoadVGet, Type: t, Args: [3]Value{v}, Imm: int64(i)}))
	}
	return out
}

// LdMatrix is a warp's collective load of n (2 or 4) 8x8 matrices of
// 16-bit elements (8 rows of 16 bytes) from shared memory: PTX's ldmatrix.
// Lane l names, in idx+off, the u32 element where row l%8 of matrix l/8
// starts (16-byte aligned; lanes past 8n are ignored), and receives as
// component i the 32-bit word l%4 of row l/4 of matrix i -- the A or B
// fragment of an mma whose operand rows are those 16 bytes (for int8,
// m8n8k16's A at k 4(l%4)..+3 of row l/4). Every lane of the warp must
// reach it. Only PTX lowers it; the others refuse it by name.
func (b *Builder) LdMatrix(buf, idx Value, off int64, n int) []Value {
	if n != 2 && n != 4 {
		panic(fmt.Sprintf("ir: LdMatrix of %d matrices; 2 or 4", n))
	}
	b.k.Lanes = SubgroupLanes
	v := b.emit(Op{Kind: OpLoadV, Type: U32, Args: [3]Value{buf, idx}, Imm: ldmMark | off<<3 | int64(n)})
	out := []Value{v}
	for i := 1; i < n; i++ {
		out = append(out, b.emit(Op{Kind: OpLoadVGet, Type: U32, Args: [3]Value{v}, Imm: int64(i)}))
	}
	return out
}

// ldmMark is the Imm bit that makes an OpLoadV LdMatrix's: high, so the
// encoding of every other vector access (and every IR fingerprint) is as it
// was before ldmatrix existed.
const ldmMark = int64(1) << 60

// IsLdMatrix reports whether an OpLoadV is LdMatrix's collective load.
func IsLdMatrix(o Op) bool { return o.Kind == OpLoadV && o.Imm&ldmMark != 0 }

// StoreV writes vs to elements idx+off .. idx+off+len(vs)-1 of a buffer in
// one access, len(vs) 2 or 4. The address must be aligned to 4*len(vs) bytes.
func (b *Builder) StoreV(buf, idx Value, off int64, vs ...Value) {
	if len(vs) != 2 && len(vs) != 4 {
		panic(fmt.Sprintf("ir: StoreV of %d elements; 2 or 4", len(vs)))
	}
	b.emit(Op{Kind: OpStoreV, Type: Void, Args: [3]Value{buf, idx}, Imm: off<<3 | int64(len(vs)),
		Vargs: append([]Value{}, vs...)})
}

// VecOf unpacks a LoadV/StoreV Imm into its element offset and width.
func VecOf(o Op) (off int64, n int) { return (o.Imm &^ ldmMark) >> 3, int(o.Imm & 7) }

// Dot4 is the four-way packed int8 dot product with accumulate: dp4a on PTX,
// OpSDotKHR on SPIR-V, four multiplies on Metal.
func (b *Builder) Dot4(a, x, acc Value) Value {
	return b.emit(Op{Kind: OpDot4, Type: I32, Args: [3]Value{a, x, acc}})
}

// Dot4Bounded is Dot4 where both operands are known to lie in [-127, 127],
// i.e. no byte is -128. It computes the same thing; a lowerer may ignore the
// fact. MSL uses it to widen with unpack_snorm4x8_to_float, which is exact on
// that range and wrong by one step outside it.
func (b *Builder) Dot4Bounded(a, x, acc Value) Value {
	return b.emit(Op{Kind: OpDot4, Type: I32, Args: [3]Value{a, x, acc}, Imm: 1})
}

// Fma is a*b+c in f32.
func (b *Builder) Fma(a, x, c Value) Value {
	return b.emit(Op{Kind: OpFma, Type: F32, Args: [3]Value{a, x, c}})
}

// Loop opens a counted loop of trip iterations. Everything emitted until
// EndLoop is the body. Values that change across iterations are declared with
// Phi.
func (b *Builder) Loop(trip int64) { b.emit(Op{Kind: OpLoop, Type: Void, Imm: trip}) }

// LoopN opens a loop whose trip count is a runtime value. It exists for
// attention, whose score loop runs pos+1 times; every other shape is baked.
// n == 0 executes the body zero times on every backend (PTX pays an entry
// guard for it).
func (b *Builder) LoopN(n Value) {
	b.emit(Op{Kind: OpLoop, Type: Void, Imm: -1, Args: [3]Value{n}})
}

// Phi declares a loop-carried value: init before the loop, update at the end.
func (b *Builder) Phi(t Type, init Value) Value {
	return b.emit(Op{Kind: OpPhi, Type: t, Args: [3]Value{init}})
}

// SetPhi records the value a Phi takes on the next iteration.
func (b *Builder) SetPhi(phi, next Value) {
	b.k.Ops[phi-1].Args[1] = next
}

func (b *Builder) EndLoop() { b.emit(Op{Kind: OpEndLoop, Type: Void}) }

// ShuffleXor returns x as it stands in lane (lane ^ mask). Every lane gets a
// value, so a five-step butterfly leaves the reduction in all of them, which
// keeps clamped surplus threads storing the same number.
func (b *Builder) ShuffleXor(t Type, x Value, mask int64) Value {
	b.k.Lanes = SubgroupLanes
	return b.emit(Op{Kind: OpShuffleXor, Type: t, Args: [3]Value{x}, Imm: mask})
}

// Done returns the finished kernel.
func (b *Builder) Done() *Kernel { return &b.k }

// Validate checks the invariants a backend relies on that the device compilers
// cannot see, such as a dangling value or an unclosed loop, which would
// otherwise produce well-formed but wrong output.
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
	var loops []int // indices of the OpLoop ops currently open
	for i, o := range k.Ops {
		for n, a := range o.Args {
			// A Phi's second argument is the back edge and is defined later.
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
			// A matrix instruction is a contract between the 32 lanes of a
			// warp, so the workgroup must be whole warps.
			if k.Group[0]%32 != 0 || k.Group[1] != 1 || k.Group[2] != 1 {
				return fmt.Errorf("ir: op %d: an MMA kernel needs a workgroup of 32n x 1 x 1 "+
					"so that lane == tid mod 32; this one is %v", i, k.Group)
			}
			if depth > 0 {
				// An MMA is warp-synchronous like a shuffle: every lane must
				// execute it. Loops here are uniform-trip by construction, so
				// this is allowed, unlike Shared.
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
			// Shared arrays must be declared outside every loop: the targets
			// disagree on what an in-loop declaration means (a fresh object
			// per iteration versus one hoisted object), with no diagnostic.
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
			// A mask below 32 cannot leave the aligned 32-lane block.
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
			// The init must be defined outside the loop: it is the value
			// arriving from the preamble, and backends fail obscurely
			// otherwise.
			if len(loops) > 0 && int(o.Args[0]) > loops[len(loops)-1] {
				return fmt.Errorf("ir: op %d: Phi's init (value %d) is defined inside the loop "+
					"opened at op %d; hoist it before the Loop", i, o.Args[0], loops[len(loops)-1])
			}
		}
	}
	// No kernel may read a buffer it writes. Surplus threads are clamped to
	// the last work item rather than branched off, so they recompute and store
	// it again; that is harmless only when the result is a pure function of
	// read-only inputs. An in-place kernel silently corrupts the last item.
	// A load or store reads its buffer as the type the parameter declares.
	// The backends disagree on anything else: PTX and SPIR-V reinterpret the
	// bits, MSL converts the value, so a u32 read from an f32 buffer is a
	// number on two of them and zero on the third.
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
	// Per phase: a grid barrier orders a write before every later read, so
	// the sets start again after one (gridsync.go); per task, likewise, where
	// validateDeps has already required the wait that orders them
	// (depsync.go).
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

	// The declared width must match the ops, in both directions: a missing
	// declaration lets the kernel reach a device that guarantees no width, and
	// a stale one refuses a device that could run it.
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

// maxParams bounds the alias check's bookkeeping. Kernels here take a handful
// of buffers; a persistent kernel composed of a decode block's launches
// (Persist) takes every buffer they bind.
const maxParams = 64

// paramOf resolves a value to the parameter index it names, or -1.
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

// MMAShape is one matrix tile: M rows of A by N columns of B over K elements.
// K is chosen by the quant format's sub-block, because the per-sub-block scale
// must apply to a complete integer dot product.
type MMAShape struct {
	M, N, K int
	Kind    MMAKind
}

// MMAKind is the operand type. The accumulator is int32 for S8 and float32
// for F16, wider than the operands in both cases.
type MMAKind uint8

const (
	// MMAS8 is int8 x int8 -> int32, the quantized matvec's shape.
	MMAS8 MMAKind = iota
	// MMAF16 is binary16 x binary16 -> float32, for attention. The operands
	// lose mantissa (11 bits against 24) while the accumulator stays float32,
	// so gates use NMSE plus token ids rather than bit equality.
	MMAF16
)

var mmaShapes = []MMAShape{
	{16, 8, 16, MMAS8}, {16, 8, 32, MMAS8},
	{16, 8, 8, MMAF16}, {16, 8, 16, MMAF16},
	MMAVolta(), MMAS8Turing(),
}

// MMAS8Turing is m8n8k16 int8 -> int32, the integer matrix instruction sm_75
// has (the m16n8 integer shapes are sm_80's). Lane (g, q) = (lane/4, lane%4)
// holds A's row g at k 4q..4q+3, B's column g at the same k, and D's row g at
// columns 2q and 2q+1. A kernel asks for it by name: it is not how the m16n8
// shapes lower, so a device without them still refuses those (and the tier's
// integer-matvec probe keeps its answer).
func MMAS8Turing() MMAShape { return MMAShape{8, 8, 16, MMAS8} }

// MMAVolta is sm_70's only matrix instruction, mma.sync m8n8k4 f16 -> f32. It
// is not a warp-wide tile: the warp is four quad-pairs (lanes 4q..4q+3 with
// 16+4q..16+4q+3), each computing its own 8x8x4 product. A lane holds one row
// of A and one column of B, four k each (two packed registers), and eight
// accumulators. Layout, as measured on sm_70 hardware:
//
//	A, B   lane l holds row/column (l&3) + 4*(l>>4) of its quad-pair, k 0..3
//	D[i]   row (l&1) + 2*((i>>1)&1) + 4*(l>>4), column (i&1) + 2*((l>>1)&1) + 4*(i>>2)
//
// sm_70 has no integer mma.sync. Only PTX lowers it.
func MMAVolta() MMAShape { return MMAShape{8, 8, 4, MMAF16} }

// Frags is how many u32 registers one lane holds of A, B and C for this shape.
// A lane holds M*K/32 operands of A and K*N/32 of B, packed four int8 or two
// binary16 to a register, and M*N/32 accumulators.
func (s MMAShape) Frags() (a, bb, c int) {
	if s == MMAVolta() {
		// Per quad-pair, not per warp: a lane's row of four halves, its column
		// of four, and eight of the quad-pair's sixty-four accumulators.
		return 2, 2, 8
	}
	per := 4
	if s.Kind == MMAF16 {
		per = 2
	}
	return s.M * s.K / (32 * per), s.K * s.N / (32 * per), s.M * s.N / 32
}

// Elem is the accumulator type this shape produces.
func (s MMAShape) Elem() Type {
	if s.Kind == MMAF16 {
		return F32
	}
	return I32
}

// Valid reports whether the hardware has this tile.
func (s MMAShape) Valid() bool {
	for _, x := range mmaShapes {
		if x == s {
			return true
		}
	}
	return false
}

// MMA emits a warp-level D = A*B + C. a, bb and c are the per-lane fragment
// registers (four int8 or two binary16 per u32 for a and bb, one accumulator
// each for c). It returns the components of D. The caller owns the fragment
// layout, which is why this takes registers rather than pointers.
func (b *Builder) MMA(sh MMAShape, a, bb, c []Value) []Value {
	na, nb, nc := sh.Frags()
	if !sh.Valid() || len(a) != na || len(bb) != nb || len(c) != nc {
		panic(fmt.Sprintf("ir: MMA %dx%dx%d wants %d/%d/%d fragments, got %d/%d/%d",
			sh.M, sh.N, sh.K, na, nb, nc, len(a), len(bb), len(c)))
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

// MMAShapeOf returns the shape whose Imm this op carries.
func MMAShapeOf(o Op) MMAShape {
	return MMAShape{M: int(o.Imm>>16) & 0xFF, N: int(o.Imm>>8) & 0xFF, K: int(o.Imm) & 0xFF,
		Kind: MMAKind(o.Imm >> 24)}
}

// WrittenParams is a bit per parameter the kernel stores into (OpStore,
// OpStoreV, OpTileStore), parameter i at bit i; a kernel with more than 64
// parameters reports every bit set. A backend that runs dispatches
// concurrently orders them by it: a dispatch reading what an earlier one
// writes, or writing what an earlier one reads or writes, waits for it.
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
			// Not a parameter by name (shared memory is OpShared and is not
			// a store target here): say everything is written.
			if b != 0 && int(b) <= len(k.Ops) && k.Ops[b-1].Kind == OpShared {
				continue
			}
			return ^uint64(0)
		}
		m |= 1 << uint(k.Ops[b-1].Imm)
	}
	return m
}

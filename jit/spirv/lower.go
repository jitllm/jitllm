package spirv

import (
	"encoding/binary"
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
)

// glsl.std.450 instruction numbers, only the ones used.
const (
	glslUnpackHalf2x16 = 62
	glslPackHalf2x16   = 58
	glslExp            = 27
	glslSqrt           = 31
	glslFMin           = 37
	glslFMax           = 40
	glslSMax           = 42
	glslUMax           = 41
	glslUMin           = 38
	glslSMin           = 39
)

type lowerer struct {
	m     mod
	k     *ir.Kernel
	val   []uint32 // ir.Value -> spirv id
	tid   uint32   // loaded LocalInvocationId.x
	ctaid uint32
	glsl  uint32

	tvoid, tbool, tu32, ti32, tf32, tv3u32, tv2f32 uint32
	consts                                         map[[2]uint32]uint32 // (type, bits) -> id
	ptrTo                                          map[uint32]uint32    // elem type -> ptr StorageBuffer
	bufs                                           []uint32             // param index -> variable id
	shared                                         map[ir.Value]uint32  // OpShared value -> variable id
	ptrWG                                          map[uint32]uint32    // elem type -> ptr Workgroup
	needDot                                        bool
	// comps and tiles intern the cooperative-matrix component and matrix types.
	// A type instruction may appear once, and a tile kernel asks for the same
	// three or four types on every op.
	comps       map[ir.TileElem]uint32
	tiles       map[ir.TileType]uint32
	needShuffle bool
	// needArith and the two builtins serve the subgroup collectives: the
	// reductions' capability, and the lane index and width as Input
	// variables, declared only for a kernel that reads them.
	needArith        bool
	laneVar, sizeVar uint32
	laneID, sgSize   uint32
	cur              uint32 // label of the block being filled
	// anchorLoop is the op index of the loop whose count takes the descriptor
	// anchor, anchorBuf the parameter it reads, and anchorNever the entry's
	// always-false predicate; -1 when the module needs none. See anchorFor.
	anchorLoop, anchorBuf int
	anchorNever           uint32
	// grid is whether the kernel has a grid barrier, and nGroups the loaded
	// NumWorkgroups.x it counts arrivals against (grid.go).
	grid    bool
	nGroups uint32
}

// anchorFor finds where a module's first buffer access sits. When it is inside
// a loop with a runtime count, it returns the outermost such loop around it and
// a parameter the anchor can measure; otherwise -1.
//
// The anchor exists for llvmpipe. Its SoA backend runs a loop's body under an
// empty execution mask once every lane has left, and a workgroup whose count
// is zero gets exactly that pass and nothing else. Lavapipe resolves a buffer
// through its descriptor set's base address, defined where the module first
// touches a buffer; defined inside such a loop, the per-lane copy of it that
// the code after a nested loop reads is zero on that pass, and Mesa before
// 63b89856 ("gallivm: don't deref a NULL buffer descriptor with an empty exec
// mask") reads the descriptor through lane 0 of it when no lane is active: a
// load at the binding's offset from a null pointer. kernels.MatVecSegments,
// whose every access sits in a per-segment loop that the other segments'
// workgroups run zero times, faulted on the first launch.
//
// One buffer access at the entry, ahead of every loop, gives the base a
// definition that dominates every other use, and the rest fold into it (a
// buffer access after the loops does not: it dominates nothing). An
// OpArrayLength is that access: it reads the descriptor, not memory. It is kept
// alive by feeding the loop's count through a select on a length no buffer can
// have, so no compiler can drop it and no launch can see it.
func anchorFor(k *ir.Kernel) (loop, buf int) {
	buf = -1
	for i, p := range k.Params {
		if !p.IsTile { // stride 4: a length of at most 2^30
			buf = i
			break
		}
	}
	var open []int // runtime-count loops enclosing the walk, outermost first
	var all []int  // every enclosing loop
	for i, o := range k.Ops {
		switch o.Kind {
		case ir.OpLoop:
			all = append(all, i)
			if o.Imm < 0 {
				open = append(open, i)
			}
		case ir.OpEndLoop:
			if n := len(all) - 1; n >= 0 {
				if len(open) > 0 && open[len(open)-1] == all[n] {
					open = open[:len(open)-1]
				}
				all = all[:n]
			}
		case ir.OpLoad, ir.OpStore, ir.OpLoadV, ir.OpStoreV, ir.OpTileLoad, ir.OpTileStore:
			a := o.Args[0]
			if a == 0 || k.Ops[a-1].Kind != ir.OpParam {
				continue
			}
			if len(open) == 0 || buf < 0 {
				return -1, -1
			}
			return open[0], buf
		}
	}
	return -1, -1
}

// Emit lowers a kernel to a SPIR-V binary module.
func Emit(k *ir.Kernel) ([]byte, error) {
	if err := k.Validate(); err != nil {
		return nil, err
	}
	for i, o := range k.Ops {
		if ir.IsLdMatrix(o) {
			// ldmatrix's lane contract is NVIDIA's warp of 32 with a fixed
			// row-to-lane map; there is no instruction here that keeps it.
			return nil, fmt.Errorf("spirv: op %d: no lowering for an ldmatrix load", i+1)
		}
	}
	l := &lowerer{k: k, val: make([]uint32, len(k.Ops)+1),
		consts: map[[2]uint32]uint32{}, ptrTo: map[uint32]uint32{}}
	for i, o := range k.Ops {
		// Two tile forms not lowered here yet, refused by name: a loop-carried
		// cooperative matrix and a cooperative access to workgroup memory.
		// SPV_KHR_cooperative_matrix can express both, but neither is gated.
		// They exist for Metal's staged GEMM (kernels.GemmTile).
		if o.Kind == ir.OpPhi && o.Tile != nil {
			return nil, fmt.Errorf("spirv: op %d: a loop-carried cooperative matrix is not lowered", i)
		}
		if (o.Kind == ir.OpTileLoad || o.Kind == ir.OpTileStore) && o.Args[0] > 0 &&
			k.Ops[o.Args[0]-1].Kind == ir.OpShared {
			return nil, fmt.Errorf("spirv: op %d: a cooperative matrix access to workgroup memory is not lowered", i)
		}
		switch o.Kind {
		case ir.OpDot4:
			l.needDot = true
		case ir.OpShuffleXor, ir.OpShuffle, ir.OpSubgroupBroadcast:
			l.needShuffle = true
		case ir.OpSubgroupReduce:
			l.needArith = true
		case ir.OpLaneID:
			l.laneVar = 1 // allocated in build
		case ir.OpSubgroupSize:
			l.sizeVar = 1
		}
	}
	l.grid = k.GridSync()
	l.anchorLoop, l.anchorBuf = anchorFor(k)
	if err := l.build(); err != nil {
		return nil, err
	}
	return l.serialize(), nil
}

func (l *lowerer) build() error {
	m := &l.m

	inst(&m.caps, opCapability, capShader)
	if l.needDot {
		inst(&m.caps, opCapability, capDotProduct)
		inst(&m.caps, opCapability, capDotProductInput4x8BitPacked)
		inst(&m.exts, opExtension, str("SPV_KHR_integer_dot_product")...)
	}
	if l.needShuffle {
		// GroupNonUniform is declared alongside the Shuffle capability it
		// enables, which depends on it. Both are core in SPIR-V 1.3.
		inst(&m.caps, opCapability, capGroupNonUniform)
		inst(&m.caps, opCapability, capGroupNonUniformShuffle)
	}
	if l.needArith || l.laneVar != 0 || l.sizeVar != 0 {
		// GroupNonUniform is what the builtins and every GroupNonUniform
		// instruction stand on; declared once, as spirv-val wants.
		if !l.needShuffle {
			inst(&m.caps, opCapability, capGroupNonUniform)
		}
		if l.needArith {
			inst(&m.caps, opCapability, capGroupNonUniformArithmetic)
		}
	}
	if hasTiles(l.k) {
		l.tileCaps()
	}
	l.glsl = m.id()
	inst(&m.exts, opExtInstImport, append([]uint32{l.glsl}, str("GLSL.std.450")...)...)

	// Types. Order matters only in that a type must precede its uses, which
	// materialise-on-demand gives for free.
	l.tvoid = m.id()
	inst(&m.types, opTypeVoid, l.tvoid)
	l.tbool = m.id()
	inst(&m.types, opTypeBool, l.tbool)
	l.tu32 = m.id()
	inst(&m.types, opTypeInt, l.tu32, 32, 0)
	l.ti32 = m.id()
	inst(&m.types, opTypeInt, l.ti32, 32, 1)
	l.tf32 = m.id()
	inst(&m.types, opTypeFloat, l.tf32, 32)
	l.tv3u32 = m.id()
	inst(&m.types, opTypeVector, l.tv3u32, l.tu32, 3)
	l.tv2f32 = m.id()
	inst(&m.types, opTypeVector, l.tv2f32, l.tf32, 2)

	tfn := m.id()
	inst(&m.types, opTypeFunction, tfn, l.tvoid)

	// Buffers: one StorageBuffer struct { runtime array } per parameter, at
	// descriptor set 0, binding i.
	for i, p := range l.k.Params {
		et := l.spvType(p.Elem)
		stride := uint32(4)
		if p.IsTile {
			// The array stride is the component's size: an f16 buffer
			// with stride 4 would read every other element.
			et = l.compType(p.Tile)
			stride = uint32(p.Tile.Bytes())
		}
		arr, st := m.id(), m.id()
		inst(&m.types, opTypeRuntimeArray, arr, et)
		inst(&m.types, opTypeStruct, st, arr)
		pst := m.id()
		inst(&m.types, opTypePointer, pst, storageStorageBuffer, st)
		v := m.id()
		inst(&m.types, opVariable, pst, v, storageStorageBuffer)
		inst(&m.deco, opDecorate, arr, decoArrayStride, stride)
		inst(&m.deco, opMemberDecorate, st, 0, decoOffset, 0)
		inst(&m.deco, opDecorate, st, decoBlock)
		inst(&m.deco, opDecorate, v, decoDescriptorSet, 0)
		inst(&m.deco, opDecorate, v, decoBinding, uint32(i))
		if l.grid {
			inst(&m.deco, opDecorate, v, decoCoherent)
		}
		l.bufs = append(l.bufs, v)
	}

	// Workgroup arrays are module-scope OpVariables with a fixed-length
	// OpTypeArray, collected in a pre-pass because their uses come after the
	// type section closes.
	l.shared = map[ir.Value]uint32{}
	for i, o := range l.k.Ops {
		if o.Kind != ir.OpShared {
			continue
		}
		et := l.spvType(o.Type)
		arr := m.id()
		inst(&m.types, opTypeArray, arr, et, l.u32c(uint32(o.Imm)))
		pa := m.id()
		inst(&m.types, opTypePointer, pa, storageWorkgroup, arr)
		v := m.id()
		inst(&m.types, opVariable, pa, v, storageWorkgroup)
		l.shared[ir.Value(i+1)] = v
	}

	// Builtin input vectors.
	pin := m.id()
	inst(&m.types, opTypePointer, pin, storageInput, l.tv3u32)
	vLocal, vGroup := m.id(), m.id()
	inst(&m.types, opVariable, pin, vLocal, storageInput)
	inst(&m.types, opVariable, pin, vGroup, storageInput)
	inst(&m.deco, opDecorate, vLocal, decoBuiltIn, builtInLocalInvocationId)
	inst(&m.deco, opDecorate, vGroup, decoBuiltIn, builtInWorkgroupId)

	// The group base: a push-constant block of one uint32 that the runtime
	// sets per dispatch (vulkan.Ctx.dispatch). A device bounds the workgroups
	// of one dispatch -- 65535 on an Intel integrated GPU -- so a larger launch is several
	// dispatches, and the IR's CTAID is WorkgroupId.x plus the index of the
	// dispatch's first workgroup. Every launch pushes it, 0 for one that fits,
	// so this is the one path every kernel takes rather than a variant only an
	// oversized launch would reach. The block is not an entry-point interface
	// variable: SPIR-V 1.3 lists only Input and Output there.
	baseSt := m.id()
	inst(&m.types, opTypeStruct, baseSt, l.tu32)
	inst(&m.deco, opDecorate, baseSt, decoBlock)
	inst(&m.deco, opMemberDecorate, baseSt, 0, decoOffset, 0)
	pBaseSt, pBase := m.id(), m.id()
	inst(&m.types, opTypePointer, pBaseSt, storagePushConstant, baseSt)
	inst(&m.types, opTypePointer, pBase, storagePushConstant, l.tu32)
	vBase := m.id()
	inst(&m.types, opVariable, pBaseSt, vBase, storagePushConstant)

	// The lane index and the subgroup width, scalar u32 Input builtins.
	iface := []uint32{vLocal, vGroup}
	var vNum uint32
	if l.grid {
		vNum = m.id()
		inst(&m.types, opVariable, pin, vNum, storageInput)
		inst(&m.deco, opDecorate, vNum, decoBuiltIn, builtInNumGroups)
		iface = append(iface, vNum)
	}
	if l.laneVar != 0 || l.sizeVar != 0 {
		pu := m.id()
		inst(&m.types, opTypePointer, pu, storageInput, l.tu32)
		if l.laneVar != 0 {
			l.laneVar = m.id()
			inst(&m.types, opVariable, pu, l.laneVar, storageInput)
			inst(&m.deco, opDecorate, l.laneVar, decoBuiltIn, builtInSubgroupLocalInvocation)
			iface = append(iface, l.laneVar)
		}
		if l.sizeVar != 0 {
			l.sizeVar = m.id()
			inst(&m.types, opVariable, pu, l.sizeVar, storageInput)
			inst(&m.deco, opDecorate, l.sizeVar, decoBuiltIn, builtInSubgroupSize)
			iface = append(iface, l.sizeVar)
		}
	}

	fn, entry := m.id(), m.id()
	name := append([]uint32{execModelGLCompute, fn}, str(l.k.Name)...)
	inst(&m.entry, opEntryPoint, append(name, iface...)...)
	inst(&m.exec, opExecutionMode, fn, execModeLocalSize,
		uint32(l.k.Group[0]), uint32(l.k.Group[1]), uint32(l.k.Group[2]))

	inst(&m.funcs, opFunction, l.tvoid, fn, 0, tfn)
	inst(&m.funcs, opLabel, entry)

	// Extract .x once. Every kernel uses these and re-loading per use would
	// leave the driver to CSE something we never had to emit.
	ld1, ld2 := m.id(), m.id()
	inst(&m.funcs, opLoad, l.tv3u32, ld1, vLocal)
	inst(&m.funcs, opLoad, l.tv3u32, ld2, vGroup)
	l.tid = m.id()
	inst(&m.funcs, opCompositeExtract, l.tu32, l.tid, ld1, 0)
	gx, acBase, base := m.id(), m.id(), m.id()
	inst(&m.funcs, opCompositeExtract, l.tu32, gx, ld2, 0)
	inst(&m.funcs, opAccessChain, pBase, acBase, vBase, l.u32c(0))
	inst(&m.funcs, opLoad, l.tu32, base, acBase)
	l.ctaid = m.id()
	inst(&m.funcs, opIAdd, l.tu32, l.ctaid, gx, base)

	if l.anchorLoop >= 0 {
		n := m.id()
		inst(&m.funcs, opArrayLength, l.tu32, n, l.bufs[l.anchorBuf], 0)
		l.anchorNever = m.id()
		inst(&m.funcs, opIEqual, l.tbool, l.anchorNever, n, l.u32c(^uint32(0)))
	}

	if l.grid {
		ld := m.id()
		inst(&m.funcs, opLoad, l.tv3u32, ld, vNum)
		l.nGroups = m.id()
		inst(&m.funcs, opCompositeExtract, l.tu32, l.nGroups, ld, 0)
	}
	if l.laneVar != 0 {
		l.laneID = m.id()
		inst(&m.funcs, opLoad, l.tu32, l.laneID, l.laneVar)
	}
	if l.sizeVar != 0 {
		l.sgSize = m.id()
		inst(&m.funcs, opLoad, l.tu32, l.sgSize, l.sizeVar)
	}

	l.cur = entry
	if err := l.block(0, len(l.k.Ops)); err != nil {
		return err
	}
	inst(&m.funcs, opReturn)
	inst(&m.funcs, opFunctionEnd)
	return nil
}

func (l *lowerer) spvType(t ir.Type) uint32 {
	switch t {
	case ir.I32:
		return l.ti32
	case ir.U32:
		return l.tu32
	case ir.F32:
		return l.tf32
	case ir.Pred:
		return l.tbool
	case ir.Void:
		return l.tvoid
	}
	return l.tu32
}

func (l *lowerer) constID(t uint32, bits uint32) uint32 {
	key := [2]uint32{t, bits}
	if id, ok := l.consts[key]; ok {
		return id
	}
	id := l.m.id()
	inst(&l.m.types, opConstant, t, id, bits)
	l.consts[key] = id
	return id
}

func (l *lowerer) u32c(v uint32) uint32 { return l.constID(l.tu32, v) }

// ptrElem returns the pointer-to-element type for a buffer element type.
func (l *lowerer) ptrElem(et uint32) uint32 {
	if id, ok := l.ptrTo[et]; ok {
		return id
	}
	id := l.m.id()
	inst(&l.m.types, opTypePointer, id, storageStorageBuffer, et)
	l.ptrTo[et] = id
	return id
}

// ptrElemWG is ptrElem in the Workgroup storage class. Shared memory needs its
// own pointer types: a pointer's storage class is part of its TYPE in SPIR-V,
// so reusing the StorageBuffer one is not a mismatch spirv-val forgives.
func (l *lowerer) ptrElemWG(et uint32) uint32 {
	if l.ptrWG == nil {
		l.ptrWG = map[uint32]uint32{}
	}
	if id, ok := l.ptrWG[et]; ok {
		return id
	}
	id := l.m.id()
	inst(&l.m.types, opTypePointer, id, storageWorkgroup, et)
	l.ptrWG[et] = id
	return id
}

// matchEnd finds the OpEndLoop matching the OpLoop at i.
func (l *lowerer) matchEnd(i int) int {
	d := 0
	for j := i; j < len(l.k.Ops); j++ {
		switch l.k.Ops[j].Kind {
		case ir.OpLoop:
			d++
		case ir.OpEndLoop:
			d--
			if d == 0 {
				return j
			}
		}
	}
	return -1
}

// block lowers ops[lo:hi), recursing on loops.
//
// A loop-carried value is strict SSA here: OpPhi must be the first instruction
// of the loop header and name its predecessor blocks, so the IR's phis are
// hoisted to the header and their back-edge operands patched in once the body
// has been walked.
func (l *lowerer) block(lo, hi int) error {
	m := &l.m
	for i := lo; i < hi; i++ {
		o := l.k.Ops[i]
		v := ir.Value(i + 1)
		switch o.Kind {
		case ir.OpLoop:
			end := l.matchEnd(i)
			if end < 0 || end >= hi {
				return fmt.Errorf("spirv: unmatched loop at op %d", i)
			}
			if err := l.loop(i, end); err != nil {
				return err
			}
			i = end
			continue
		case ir.OpPhi:
			// Already materialised in the loop header by loop(); the IR's
			// position for it is just where the builder happened to emit it.
			if l.val[v] == 0 {
				return fmt.Errorf("spirv: op %d: phi outside a loop", i)
			}
		case ir.OpEndLoop:
			return fmt.Errorf("spirv: op %d: EndLoop outside a loop", i)
		case ir.OpParam:
			l.val[v] = l.bufs[o.Imm]
		case ir.OpShared:
			l.val[v] = l.shared[v]
		case ir.OpBarrier:
			// A kernel reads after a barrier what its workgroup wrote to a
			// storage buffer before it, as well as to shared memory.
			inst(&l.m.funcs, opControlBarrier, l.u32c(scopeWorkgroup), l.u32c(scopeWorkgroup),
				l.u32c(semAcqRel|semUniform|semWorkgroupMem))
		case ir.OpGridBarrier:
			l.gridBarrier(l.val[o.Args[0]])
		case ir.OpDepTask, ir.OpDepEpoch, ir.OpDepWait, ir.OpDepSignal, ir.OpDepExit:
			// A workgroup spinning on another's counter needs that one to
			// run, and Vulkan promises no forward progress between
			// workgroups (docs/design/dep-block.md).
			return fmt.Errorf("spirv: op %d: %s: dependency counters are not lowered for SPIR-V", i, o.Kind)
		case ir.OpConst:
			l.val[v] = l.constID(l.spvType(o.Type), uint32(o.Imm))
		case ir.OpTID:
			l.val[v] = l.tid
		case ir.OpCTAID:
			l.val[v] = l.ctaid
		case ir.OpNTID:
			l.val[v] = l.u32c(uint32(l.k.Group[0]))
		case ir.OpLaneID:
			l.val[v] = l.laneID
		case ir.OpSubgroupSize:
			l.val[v] = l.sgSize
		case ir.OpPrefetch:
			// SPIR-V has no prefetch; the hint lowers to nothing.
		case ir.OpTileLoad, ir.OpTileSplat, ir.OpTileMMA, ir.OpTileStore:
			if err := l.tileOp(o, v); err != nil {
				return fmt.Errorf("spirv: op %d: %w", i, err)
			}
		case ir.OpLoad:
			l.val[v] = l.access(o, opLoad)
		case ir.OpStore:
			l.access(o, opStore)
		default:
			id, err := l.arith(o)
			if err != nil {
				return fmt.Errorf("spirv: op %d: %w", i, err)
			}
			l.val[v] = id
		}
		_ = m
	}
	return nil
}

// access emits the OpAccessChain that both Load and Store need. The chain is
// (buffer, member 0, index) because every buffer is a Block struct wrapping one
// runtime array -- the shape Vulkan requires of a storage buffer.
func (l *lowerer) access(o ir.Op, which uint32) uint32 {
	idx := l.val[o.Args[1]]
	if o.Imm != 0 {
		n := l.m.id()
		inst(&l.m.funcs, opIAdd, l.tu32, n, idx, l.u32c(uint32(o.Imm)))
		idx = n
	}
	et := l.spvType(o.Type)
	if which == opStore {
		et = l.spvType(l.k.Ops[o.Args[2]-1].Type)
	}
	ptr := l.m.id()
	// One index for shared, two for a buffer: a StorageBuffer parameter is a
	// struct wrapping a runtime array, a Workgroup array is the array itself.
	if _, isShared := l.shared[o.Args[0]]; isShared {
		inst(&l.m.funcs, opAccessChain, l.ptrElemWG(et), ptr, l.val[o.Args[0]], idx)
	} else {
		inst(&l.m.funcs, opAccessChain, l.ptrElem(et), ptr, l.val[o.Args[0]], l.u32c(0), idx)
	}
	if which == opStore {
		inst(&l.m.funcs, opStore, ptr, l.val[o.Args[2]])
		return 0
	}
	id := l.m.id()
	inst(&l.m.funcs, opLoad, et, id, ptr)
	return id
}

func (l *lowerer) arith(o ir.Op) (uint32, error) {
	a, b, c := l.val[o.Args[0]], l.val[o.Args[1]], l.val[o.Args[2]]
	t := l.spvType(o.Type)
	id := l.m.id()
	f := o.Type == ir.F32
	sign := o.Type == ir.I32

	pick := func(iop, fop uint32) uint32 {
		if f {
			return fop
		}
		return iop
	}
	_ = pick
	switch o.Kind {
	case ir.OpAdd:
		inst(&l.m.funcs, pick(opIAdd, opFAdd), t, id, a, b)
	case ir.OpSub:
		inst(&l.m.funcs, pick(opISub, opFSub), t, id, a, b)
	case ir.OpMul:
		inst(&l.m.funcs, pick(opIMul, opFMul), t, id, a, b)
	case ir.OpDiv:
		inst(&l.m.funcs, sel(f, sign, opFDiv, opSDiv, opUDiv), t, id, a, b)
	case ir.OpRem:
		inst(&l.m.funcs, sel(f, sign, opFRem, opSRem, opUMod), t, id, a, b)
	case ir.OpShl:
		inst(&l.m.funcs, opShiftLeftLogical, t, id, a, b)
	case ir.OpShr:
		// OpShiftRightLogical, never OpShiftRightArithmetic.
		inst(&l.m.funcs, opShiftRightLogical, t, id, a, b)
	case ir.OpAnd:
		inst(&l.m.funcs, opBitwiseAnd, t, id, a, b)
	case ir.OpXor:
		inst(&l.m.funcs, opBitwiseXor, t, id, a, b)
	case ir.OpFunnel:
		// (hi:lo) >> n as shifts: the left one split (<<1, then <<31-n) so n
		// == 0 never shifts by 32, which SPIR-V leaves undefined. The halves
		// are disjoint, so IAdd is the OR.
		one, k31 := l.constID(t, 1), l.constID(t, 31)
		lo, h1, sh, hi := l.m.id(), l.m.id(), l.m.id(), l.m.id()
		inst(&l.m.funcs, opShiftRightLogical, t, lo, a, c)
		inst(&l.m.funcs, opShiftLeftLogical, t, h1, b, one)
		inst(&l.m.funcs, opISub, t, sh, k31, c)
		inst(&l.m.funcs, opShiftLeftLogical, t, hi, h1, sh)
		inst(&l.m.funcs, opIAdd, t, id, lo, hi)
	case ir.OpSelect:
		inst(&l.m.funcs, opSelect, t, id, a, b, c)
	case ir.OpMin:
		inst(&l.m.funcs, opExtInst, t, id, l.glsl, sel(f, sign, glslFMin, glslSMin, glslUMin), a, b)
	case ir.OpMax:
		inst(&l.m.funcs, opExtInst, t, id, l.glsl, sel(f, sign, glslFMax, glslSMax, glslUMax), a, b)
	case ir.OpExp:
		inst(&l.m.funcs, opExtInst, t, id, l.glsl, glslExp, a)
	case ir.OpSqrt:
		inst(&l.m.funcs, opExtInst, t, id, l.glsl, glslSqrt, a)
	case ir.OpCvtF32:
		inst(&l.m.funcs, opConvertSToF, t, id, a)
	case ir.OpCvtI32:
		inst(&l.m.funcs, opConvertFToS, t, id, a)
	case ir.OpBitcast:
		inst(&l.m.funcs, opBitcast, t, id, a)
	case ir.OpPackF16:
		// The mirror of OpCvtF16H: build a vec2 and let GLSL.std.450 do the
		// rounding, so no Float16 capability is needed here either.
		pair := l.m.id()
		inst(&l.m.funcs, opCompositeConstruct, l.tv2f32, pair, a, b)
		inst(&l.m.funcs, opExtInst, l.tu32, id, l.glsl, glslPackHalf2x16, pair)
	case ir.OpCvtF16H:
		// GLSL.std.450 UnpackHalf2x16 yields a vec2 of 32-bit floats, so no
		// Float16 capability is needed -- the extension set is already imported
		// unconditionally.
		pair := l.m.id()
		inst(&l.m.funcs, opExtInst, l.tv2f32, pair, l.glsl, glslUnpackHalf2x16, a)
		inst(&l.m.funcs, opCompositeExtract, t, id, pair, 0)
	case ir.OpLt:
		// The comparison's type comes from the OPERANDS, not from Pred.
		ot := l.k.Ops[o.Args[0]-1].Type
		inst(&l.m.funcs, sel(ot == ir.F32, ot == ir.I32, opFOrdLessThan, opSLessThan, opULessThan),
			l.tbool, id, a, b)
	case ir.OpFma:
		// OpFMul + OpFAdd rather than the GLSL Fma: these kernels scale an
		// already-exact int32 dot product, so contraction buys nothing and an
		// unfused pair is what every target agrees on.
		mul := id
		inst(&l.m.funcs, opFMul, t, mul, a, b)
		id = l.m.id()
		inst(&l.m.funcs, opFAdd, t, id, mul, c)
	case ir.OpDot4:
		d := id
		inst(&l.m.funcs, opSDot, l.ti32, d, a, b, packedFormat4x8Bit)
		id = l.m.id()
		inst(&l.m.funcs, opIAdd, l.ti32, id, d, c)
	case ir.OpShuffleXor:
		// The scope and the mask are both <id>s here rather than literals --
		// SPIR-V has no immediate operands -- so both come from the constant
		// pool. Scope is Subgroup, which is the only scope Vulkan allows a
		// non-uniform shuffle.
		inst(&l.m.funcs, opGroupNonUniformShuffleXor, t, id,
			l.u32c(scopeSubgroup), a, l.u32c(uint32(o.Imm)))
	case ir.OpSubgroupReduce:
		var op uint32
		switch ir.ReduceOf(o) {
		case ir.ReduceAdd:
			op = pick(opGroupNonUniformIAdd, opGroupNonUniformFAdd)
		case ir.ReduceMax:
			op = sel(f, sign, opGroupNonUniformFMax, opGroupNonUniformSMax, opGroupNonUniformUMax)
		case ir.ReduceMin:
			op = sel(f, sign, opGroupNonUniformFMin, opGroupNonUniformSMin, opGroupNonUniformUMin)
		}
		inst(&l.m.funcs, op, t, id, l.u32c(scopeSubgroup), groupOperationReduce, a)
	case ir.OpSubgroupBroadcast, ir.OpShuffle:
		// OpGroupNonUniformShuffle for both: OpGroupNonUniformBroadcast wants
		// a constant lane before SPIR-V 1.5, and this module is 1.3. The lane
		// is masked to the subgroup first, as the IR defines it.
		// The lane operand must be an unsigned integer.
		src := b
		if l.k.Ops[o.Args[1]-1].Type == ir.I32 {
			src = l.m.id()
			inst(&l.m.funcs, opBitcast, l.tu32, src, b)
		}
		lane := l.m.id()
		inst(&l.m.funcs, opBitwiseAnd, l.tu32, lane, src, l.u32c(31))
		inst(&l.m.funcs, opGroupNonUniformShuffle, t, id, l.u32c(scopeSubgroup), a, lane)
	case ir.OpMMA, ir.OpMMAGet:
		// Declined: VK_KHR_cooperative_matrix is an extension not required of
		// a device, and its fragment layout is opaque, so it needs a different
		// kernel rather than a lowering. The tier falls back to the dp4a
		// matvec.
		return 0, fmt.Errorf("spirv: no integer matrix instruction: %s needs "+
			"VK_KHR_cooperative_matrix and a fragment-opaque kernel", o.Kind)
	default:
		return 0, fmt.Errorf("no lowering for %s", o.Kind)
	}
	return id, nil
}

// loop lowers ops[start:end] where start is the OpLoop and end the OpEndLoop.
func (l *lowerer) loop(start, end int) error {
	m := &l.m

	// Phis belonging to THIS loop, not to a nested one.
	var phis []int
	for j, d := start+1, 0; j < end; j++ {
		switch l.k.Ops[j].Kind {
		case ir.OpLoop:
			d++
		case ir.OpEndLoop:
			d--
		case ir.OpPhi:
			if d == 0 {
				phis = append(phis, j)
			}
		}
	}

	pre := l.cur
	header, body, cont, merge := m.id(), m.id(), m.id(), m.id()
	inst(&m.funcs, opBranch, header)
	inst(&m.funcs, opLabel, header)

	// The counter is ours: the IR states a trip count, not a bound.
	ctr := m.id()
	ctrAt := len(m.funcs)
	inst(&m.funcs, opPhi, l.tu32, ctr, l.u32c(0), pre, 0, cont)

	at := make([]int, len(phis))
	for n, j := range phis {
		o := l.k.Ops[j]
		id := m.id()
		at[n] = len(m.funcs)
		inst(&m.funcs, opPhi, l.spvType(o.Type), id, l.val[o.Args[0]], pre, 0, cont)
		l.val[ir.Value(j+1)] = id
	}

	cmp := m.id()
	bound := l.u32c(uint32(l.k.Ops[start].Imm))
	if l.k.Ops[start].Imm < 0 {
		bound = l.val[l.k.Ops[start].Args[0]] // LoopN: the count is a value
	}
	if start == l.anchorLoop {
		// No buffer is 2^32-1 elements long, so this is the count; see anchorFor.
		sel := m.id()
		inst(&m.funcs, opSelect, l.tu32, sel, l.anchorNever, l.u32c(0), bound)
		bound = sel
	}
	inst(&m.funcs, opULessThan, l.tbool, cmp, ctr, bound)
	inst(&m.funcs, opLoopMerge, merge, cont, loopControlNone)
	inst(&m.funcs, opBranchConditional, cmp, body, merge)

	inst(&m.funcs, opLabel, body)
	l.cur = body
	if err := l.block(start+1, end); err != nil {
		return err
	}
	inst(&m.funcs, opBranch, cont)

	inst(&m.funcs, opLabel, cont)
	next := m.id()
	inst(&m.funcs, opIAdd, l.tu32, next, ctr, l.u32c(1))
	inst(&m.funcs, opBranch, header)

	inst(&m.funcs, opLabel, merge)
	l.cur = merge

	// Back edges, now that the body's value ids exist. Word 5 of an OpPhi is
	// the second incoming value.
	m.funcs[ctrAt+5] = next
	for n, j := range phis {
		u := l.k.Ops[j].Args[1]
		if l.val[u] == 0 {
			return fmt.Errorf("spirv: op %d: phi update %d was never lowered", j, u)
		}
		m.funcs[at[n]+5] = l.val[u]
	}
	return nil
}

func (l *lowerer) serialize() []byte {
	m := &l.m
	var w []uint32
	w = append(w, 0x07230203, 0x00010300, 0, m.next+1, 0)
	w = append(w, m.caps...)
	w = append(w, m.exts...)
	inst(&w, opMemoryModel, addrModelLogical, memModelGLSL450)
	w = append(w, m.entry...)
	w = append(w, m.exec...)
	w = append(w, m.deco...)
	w = append(w, m.types...)
	w = append(w, m.funcs...)

	b := make([]byte, 4*len(w))
	for i, x := range w {
		binary.LittleEndian.PutUint32(b[4*i:], x)
	}
	return b
}

// sel picks the float, signed-integer or unsigned-integer form of an opcode.
//
// sel picks the float, signed-integer or unsigned-integer form of an opcode. A
// type-blind choice is often still a valid instruction on the wrong operands,
// which compiles and computes wrong numbers.
func sel(isFloat, isSigned bool, fop, sop, uop uint32) uint32 {
	switch {
	case isFloat:
		return fop
	case isSigned:
		return sop
	default:
		return uop
	}
}

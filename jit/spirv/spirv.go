// Package spirv emits SPIR-V compute modules from the IR.
//
// Unlike PTX text, SPIR-V is a binary word stream with a fixed section order
// (capabilities, extensions, memory model, entry points, execution modes,
// debug, decorations, types and constants, functions), numbered ids defined
// before use, and types deduplicated by structure. So the sections are built
// separately and serialised at the end; the lowering is otherwise the same
// forward walk as ptx/lower.go.
//
// The int8 dot product is OpSDot with PackedVectorFormat4x8Bit, which needs
// SPV_KHR_integer_dot_product. That extension is core in SPIR-V 1.6 and widely
// available before it; a device without it cannot run these kernels at full
// speed and the runtime checks for it rather than emitting a slow polyfill.
package spirv

import "encoding/binary"

// SPIR-V opcodes, only the ones used.
const (
	opNop                = 0
	opExtension          = 10
	opMemoryModel        = 14
	opEntryPoint         = 15
	opExecutionMode      = 16
	opCapability         = 17
	opTypeVoid           = 19
	opTypeBool           = 20
	opTypeInt            = 21
	opTypeFloat          = 22
	opTypeVector         = 23
	opTypeRuntimeArray   = 29
	opTypeStruct         = 30
	opTypePointer        = 32
	opTypeFunction       = 33
	opConstant           = 43
	opFunction           = 54
	opFunctionEnd        = 56
	opVariable           = 59
	opLoad               = 61
	opStore              = 62
	opAccessChain        = 65
	opTypeArray          = 28
	opControlBarrier     = 224
	opDecorate           = 71
	opMemberDecorate     = 72
	opCompositeConstruct = 80
	opCompositeExtract   = 81
	opConvertSToF        = 111
	opIAdd               = 128
	opISub               = 130
	opIMul               = 132
	opUDiv               = 134
	opUMod               = 137
	opShiftLeftLogical   = 196
	opShiftRightLogical  = 194
	opConvertFToS        = 110
	opBitcast            = 124
	opFSub               = 131
	opSDiv               = 135
	opFDiv               = 136
	opSRem               = 138
	opFRem               = 140
	opSLessThan          = 177
	opFOrdLessThan       = 184
	opSelect             = 169
	opBitwiseAnd         = 199
	opBitwiseXor         = 198
	opULessThan          = 176
	opFAdd               = 129
	opFMul               = 133
	opLabel              = 248
	opBranch             = 249
	opBranchConditional  = 250
	opReturn             = 253
	opLoopMerge          = 246
	opPhi                = 245
	opSDot               = 4450
	opArrayLength        = 68
	opIEqual             = 170
	// OpGroupNonUniformShuffleXor. Core in SPIR-V 1.3, which is the version
	// this module already declares, so no extension is needed -- only the
	// capability.
	opGroupNonUniformShuffleXor = 346
	// The rest of the subgroup collectives (ir/subgroup.go), all core in
	// SPIR-V 1.3: an indexed shuffle, and the arithmetic reductions, which
	// take a GroupOperation literal (Reduce) before the value.
	opGroupNonUniformShuffle = 345
	opGroupNonUniformIAdd    = 349
	opGroupNonUniformFAdd    = 350
	opGroupNonUniformSMin    = 353
	opGroupNonUniformUMin    = 354
	opGroupNonUniformFMin    = 355
	opGroupNonUniformSMax    = 356
	opGroupNonUniformUMax    = 357
	opGroupNonUniformFMax    = 358
	groupOperationReduce     = 0
	opExtInstImport          = 11
	opExtInst                = 12
)

const (
	capShader                      = 1
	capInt8                        = 39
	capDotProduct                  = 6019
	capDotProductInput4x8BitPacked = 6017
	capGroupNonUniform             = 61
	capGroupNonUniformShuffle      = 65
	capGroupNonUniformArithmetic   = 63
	builtInSubgroupSize            = 36
	builtInSubgroupLocalInvocation = 41
	scopeSubgroup                  = 3
	execModeLocalSize              = 17
	execModelGLCompute             = 5
	storageStorageBuffer           = 12
	// Workgroup is shared memory (ir.OpShared).
	storageWorkgroup         = 4
	storageInput             = 1
	storagePushConstant      = 9 // the workgroup base a split launch sets per dispatch
	storageFunction          = 7
	decoBlock                = 2
	decoArrayStride          = 6
	decoBuiltIn              = 11
	decoBinding              = 33
	decoDescriptorSet        = 34
	decoOffset               = 35
	builtInWorkgroupId       = 26
	builtInLocalInvocationId = 27
	builtInWorkgroupSize     = 25
	packedFormat4x8Bit       = 0
	loopControlNone          = 0
	memModelGLSL450          = 1
	addrModelLogical         = 0
)

type mod struct {
	caps, exts, entry, exec, deco, types, funcs []uint32
	next                                        uint32
}

func (m *mod) id() uint32 { m.next++; return m.next }

func inst(dst *[]uint32, op uint32, words ...uint32) {
	*dst = append(*dst, (uint32(len(words)+1)<<16)|op)
	*dst = append(*dst, words...)
}

func str(s string) []uint32 {
	b := append([]byte(s), 0)
	for len(b)%4 != 0 {
		b = append(b, 0)
	}
	out := make([]uint32, len(b)/4)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(b[i*4:])
	}
	return out
}

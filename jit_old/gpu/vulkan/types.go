package vulkan

// C struct layouts for the subset of Vulkan this runtime uses.
//
// The padding is written out: these structs cross a C ABI with no header to
// check them against, so a wrong offset is a driver misreading a field, not a
// compile error. Go's alignment agrees with C's here, but explicit padding in
// the u32-then-pointer pattern makes each offset checkable against
// vulkan_core.h by eye.
//
// Handles: dispatchable ones (instance, device, queue, command buffer) are
// pointers; non-dispatchable ones (buffer, memory, pipeline, ...) are always
// 64-bit even on 32-bit hosts, so they are uint64 and not uintptr.

type (
	Instance   uintptr
	PhysDevice uintptr
	Device     uintptr
	VkQueue    uintptr
	CmdBuffer  uintptr
	Buffer     uint64
	Memory     uint64
	ShaderMod  uint64
	DescLayout uint64
	PipeLayout uint64
	Pipeline   uint64
	DescPool   uint64
	DescSet    uint64
	CmdPool    uint64
	Fence      uint64
	Result     int32
)

const (
	stAppInfo        = 0
	stInstanceCI     = 1
	stDeviceQueueCI  = 2
	stDeviceCI       = 3
	stSubmitInfo     = 4
	stMemAllocInfo   = 5
	stBufferCI       = 12
	stShaderModuleCI = 16
	stComputePipeCI  = 29
	stPipeShaderCI   = 18
	stPipeLayoutCI   = 30
	stDescPoolCI     = 33
	stDescSetAlloc   = 34
	stWriteDescSet   = 35
	stDescLayoutCI   = 32
	stCmdPoolCI      = 39
	stCmdBufAlloc    = 40
	stCmdBufBegin    = 42
	stDotProdFeat    = 1000280000

	descStorageBuffer  = 7
	stageCompute       = 0x20
	usageStorageBuffer = 0x20
	usageTransferSrc   = 0x1
	usageTransferDst   = 0x2
	memDeviceLocal     = 0x1
	memHostVisible     = 0x2
	memHostCoherent    = 0x8
	queueGraphics      = 0x1
	queueCompute       = 0x2
	cmdBufOneTime      = 0x1
	pipeBindCompute    = 1
)

type appInfo struct {
	sType   uint32
	_       uint32
	pNext   uintptr
	pName   uintptr
	appVer  uint32
	_       uint32
	pEngine uintptr
	engVer  uint32
	apiVer  uint32
}

type instanceCI struct {
	sType   uint32
	_       uint32
	pNext   uintptr
	flags   uint32
	_       uint32
	pApp    uintptr
	nLayers uint32
	_       uint32
	pLayers uintptr
	nExts   uint32
	_       uint32
	pExts   uintptr
}

type queueCI struct {
	sType  uint32
	_      uint32
	pNext  uintptr
	flags  uint32
	family uint32
	count  uint32
	_      uint32
	prio   uintptr
}

type deviceCI struct {
	sType   uint32
	_       uint32
	pNext   uintptr
	flags   uint32
	nQueues uint32
	pQueues uintptr
	nLayers uint32
	_       uint32
	pLayers uintptr
	nExts   uint32
	_       uint32
	pExts   uintptr
	pFeats  uintptr
}

// dotFeat is VkPhysicalDeviceShaderIntegerDotProductFeatures. The standalone
// struct rather than VkPhysicalDeviceVulkan13Features: three fields whose
// offsets are unmistakable, against fifteen VkBool32 in a fixed order that is
// exactly the kind of thing to get subtly wrong from memory.
type dotFeat struct {
	sType uint32
	_     uint32
	pNext uintptr
	on    uint32
	_     uint32
}

type queueFamily struct {
	flags, count, tsBits uint32
	granularity          [3]uint32
}

type memType struct{ flags, heap uint32 }
type memHeap struct {
	size  uint64
	flags uint32
	_     uint32
}
type memProps struct {
	typeCount uint32
	types     [32]memType
	heapCount uint32
	heaps     [16]memHeap
}

type bufferCI struct {
	sType   uint32
	_       uint32
	pNext   uintptr
	flags   uint32
	_       uint32
	size    uint64
	usage   uint32
	sharing uint32
	nFamily uint32
	_       uint32
	pFamily uintptr
}

type memReq struct {
	size, align uint64
	typeBits    uint32
	_           uint32
}

type memAlloc struct {
	sType   uint32
	_       uint32
	pNext   uintptr
	size    uint64
	typeIdx uint32
	_       uint32
}

type descBinding struct {
	binding, kind, count, stages uint32
	pSamplers                    uintptr
}

type descLayoutCI struct {
	sType uint32
	_     uint32
	pNext uintptr
	flags uint32
	nBind uint32
	pBind uintptr
}

// pushRange is VkPushConstantRange: 12 bytes, three uint32.
type pushRange struct {
	stages uint32
	offset uint32
	size   uint32
}

type pipeLayoutCI struct {
	sType uint32
	_     uint32
	pNext uintptr
	flags uint32
	nSets uint32
	pSets uintptr
	nPush uint32
	_     uint32
	pPush uintptr
}

type shaderCI struct {
	sType uint32
	_     uint32
	pNext uintptr
	flags uint32
	_     uint32
	size  uint64
	pCode uintptr
}

type stageCI struct {
	sType  uint32
	_      uint32
	pNext  uintptr
	flags  uint32
	stage  uint32
	module uint64
	pName  uintptr
	pSpec  uintptr
}

type computeCI struct {
	sType   uint32
	_       uint32
	pNext   uintptr
	flags   uint32
	_       uint32
	stage   stageCI
	layout  uint64
	baseHdl uint64
	baseIdx int32
	_       uint32
}

type poolSize struct{ kind, count uint32 }

type descPoolCI struct {
	sType   uint32
	_       uint32
	pNext   uintptr
	flags   uint32
	maxSets uint32
	nSizes  uint32
	_       uint32
	pSizes  uintptr
}

type descSetAI struct {
	sType uint32
	_     uint32
	pNext uintptr
	pool  uint64
	n     uint32
	_     uint32
	pSets uintptr
}

type descBufInfo struct{ buffer, offset, rng uint64 }

// VkBufferCopy
type bufCopy struct{ srcOff, dstOff, size uint64 }

type writeDesc struct {
	sType   uint32
	_       uint32
	pNext   uintptr
	dstSet  uint64
	dstBind uint32
	dstElem uint32
	count   uint32
	kind    uint32
	pImage  uintptr
	pBuffer uintptr
	pTexel  uintptr
}

type cmdPoolCI struct {
	sType  uint32
	_      uint32
	pNext  uintptr
	flags  uint32
	family uint32
}

type cmdBufAI struct {
	sType uint32
	_     uint32
	pNext uintptr
	pool  uint64
	level uint32
	count uint32
}

type cmdBeginI struct {
	sType    uint32
	_        uint32
	pNext    uintptr
	flags    uint32
	_        uint32
	pInherit uintptr
}

type submitInfo struct {
	sType   uint32
	_       uint32
	pNext   uintptr
	nWait   uint32
	_       uint32
	pWait   uintptr
	pStages uintptr
	nCmd    uint32
	_       uint32
	pCmd    uintptr
	nSignal uint32
	_       uint32
	pSignal uintptr
}

// memBarrier is VkMemoryBarrier: a whole-pipeline dependency, which is what one
// compute dispatch reading another's output needs.
type memBarrier struct {
	sType uint32
	_     uint32
	pNext uintptr
	src   uint32
	dst   uint32
}

const (
	stMemBarrier       = 46
	accessShaderRead   = 0x20
	accessShaderWrite  = 0x40
	stageComputeShader = 0x800
	// batchSets bounds one command buffer's dispatches. A decode token issues
	// ~450, so the batch flushes rather than failing when it runs out.
	batchSets = 1024
	// batchBufs is the largest buffer count any kernel takes; the k-quant
	// matvec is six.
	batchBufs = 8
)

// ---------------------------------------------------------------------------
// Device selection and memory budget.
//
// Every offset below is offsetof() from Vulkan-Headers vulkan_core.h
// (VK_HEADER_VERSION 362), and layout_test.go asserts each one.

const (
	stPhysDevMemProps2 = 1000059006 // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2
	stMemBudgetEXT     = 1000237000 // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT

	// VkPhysicalDeviceType. The numeric order is NOT the preference order --
	// integrated is 1 and discrete is 2 -- so devTypeRank does the ranking and
	// these are only names for the wire values.
	devTypeOther      = 0
	devTypeIntegrated = 1
	devTypeDiscrete   = 2
	devTypeVirtual    = 3
	devTypeCPU        = 4

	heapDeviceLocal = 0x1 // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT

	maxMemoryHeaps = 16  // VK_MAX_MEMORY_HEAPS
	maxDevNameLen  = 256 // VK_MAX_PHYSICAL_DEVICE_NAME_SIZE
	maxExtNameLen  = 256 // VK_MAX_EXTENSION_NAME_SIZE

	extMemoryBudget = "VK_EXT_memory_budget"
)

// physProps is VkPhysicalDeviceProperties, 824 bytes.
//
// It is a typed struct so layout_test.go can check the fields read
// (deviceType, deviceName). The tail -- limits (504 bytes) and
// sparseProperties (20) -- is sized rather than transcribed: the driver writes
// all 824 bytes, and limits holds VkDeviceSize members, so the array is
// []uint64 and the struct is 8-aligned like C's.
type physProps struct {
	apiVer    uint32
	driverVer uint32
	vendorID  uint32
	deviceID  uint32
	devType   uint32
	name      [maxDevNameLen]byte
	cacheUUID [16]byte
	_         uint32 // C pads here too: limits is 8-aligned and 292 is not
	limits    [63]uint64
	sparse    [5]uint32
}

// maxStorageBufferRange is VkPhysicalDeviceLimits.maxStorageBufferRange, the
// eighth uint32 of the limits (after four image dimensions, the array layers,
// the texel-buffer elements and the uniform range): byte 28, the high half of
// the fourth word.
func (p *physProps) maxStorageBufferRange() uint32 { return uint32(p.limits[3] >> 32) }

// maxComputeWorkGroupCount is VkPhysicalDeviceLimits.maxComputeWorkGroupCount[0],
// the most workgroups one vkCmdDispatch may ask for along x. It sits at byte
// 220 of the limits: the high half of the 28th word, whose low half is
// maxComputeSharedMemorySize. Byte 224 starts the [1] and [2] entries.
func (p *physProps) maxComputeWorkGroupCount() uint32 { return uint32(p.limits[27] >> 32) }

// maxComputeSharedMemorySize is the low half of the same word, read only by
// the layout gate: the specification's floor on it is a second witness that
// the word is the one maxComputeWorkGroupCount is read from.
func (p *physProps) maxComputeSharedMemorySize() uint32 { return uint32(p.limits[27]) }

// memProps2 is VkPhysicalDeviceMemoryProperties2: the same 520-byte payload as
// vkGetPhysicalDeviceMemoryProperties returns, behind an sType/pNext head so
// an extension struct can be chained onto the query.
type memProps2 struct {
	sType uint32
	_     uint32
	pNext uintptr
	props memProps
}

// memBudgetEXT is VkPhysicalDeviceMemoryBudgetPropertiesEXT, chained into
// memProps2.pNext.
//
// heapBudget is how much of the heap this process may allocate, already net of
// what other processes hold; heapUsage is what this process holds. Both are
// indexed by heap.
type memBudgetEXT struct {
	sType  uint32
	_      uint32
	pNext  uintptr
	budget [maxMemoryHeaps]uint64
	usage  [maxMemoryHeaps]uint64
}

// extProps is VkExtensionProperties, 260 bytes -- the element type of
// vkEnumerateDeviceExtensionProperties' array.
type extProps struct {
	name    [maxExtNameLen]byte
	specVer uint32
}

// ---------------------------------------------------------------------------
// Device identity.
//
// deviceUUID answers "is this the same card as that one"; pipelineCacheUUID,
// also sixteen bytes, does not (it changes with the driver). The Vulkan and
// CUDA specifications both promise deviceUUID is the same for the same physical
// device, which makes the cross-backend match evidence.

const (
	stPhysDevIDProps = 1000071004 // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES

	uuidSize = 16 // VK_UUID_SIZE
	luidSize = 8  // VK_LUID_SIZE
)

// idProps is VkPhysicalDeviceIDProperties, 64 bytes, chained into
// physProps2.pNext. Core since Vulkan 1.1; on an older device the query itself
// is invalid usage, which is why readDeviceUUID refuses rather than guesses.
type idProps struct {
	sType      uint32
	_          uint32
	pNext      uintptr
	deviceUUID [uuidSize]byte
	driverUUID [uuidSize]byte
	deviceLUID [luidSize]byte
	nodeMask   uint32
	luidValid  uint32
}

// ---------------------------------------------------------------------------
// Subgroup width.
//
// A reported width is not a promised one. VkPhysicalDeviceSubgroupProperties
// .subgroupSize is the default width; a device with a range (Intel's 8..32) may
// report 32 and run a compute shader at 16. The promise is minSubgroupSize ==
// maxSubgroupSize, or a width pinned at pipeline creation through
// VK_EXT_subgroup_size_control; subgroup.go reads both.
//
// Every offset below is offsetof() from Vulkan-Headers vulkan_core.h
// (VK_HEADER_VERSION 362), and layout_test.go asserts each one.

const (
	stPhysDevProps2 = 1000059001 // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2
	stSubgroupProps = 1000094000 // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES
	// The three VK_EXT_subgroup_size_control structures, promoted to core in
	// Vulkan 1.3 and numbered identically either way.
	stSizeCtlProps = 1000225000 // ..._PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES
	stReqSizeCI    = 1000225001 // ..._PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO
	stSizeCtlFeat  = 1000225002 // ..._PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES

	// VkSubgroupFeatureFlagBits. A device can have the right WIDTH and no
	// shuffle at all, which is a different refusal with the same consequence.
	subgroupBasic   = 0x00000001 // VK_SUBGROUP_FEATURE_BASIC_BIT
	subgroupShuffle = 0x00000010 // VK_SUBGROUP_FEATURE_SHUFFLE_BIT

	// VkPipelineShaderStageCreateFlagBits. REQUIRE_FULL_SUBGROUPS makes the
	// driver launch only whole subgroups, so a workgroup can never contain a
	// partial one.
	pipeStageRequireFullSubgroups = 0x00000002

	extSubgroupSizeControl = "VK_EXT_subgroup_size_control"

	// VK_API_VERSION_1_1, the version that makes vkGetPhysicalDeviceProperties2
	// and VkPhysicalDeviceSubgroupProperties core. A device below it answers
	// nothing here, which is a refusal and not a crash.
	apiVersion11 = 1<<22 | 1<<12
)

// physProps2 is VkPhysicalDeviceProperties2, 840 bytes: the same payload the
// core query returns, behind an sType/pNext head so the subgroup structs can be
// chained onto one call.
type physProps2 struct {
	sType uint32
	_     uint32
	pNext uintptr
	props physProps
}

// subgroupProps is VkPhysicalDeviceSubgroupProperties (core since 1.1).
//
// size is the default subgroup width and not a guarantee -- see the section
// header. ops is the VkSubgroupFeatureFlags mask: shuffles are one bit of it,
// and a device may support subgroups and not them.
type subgroupProps struct {
	sType   uint32
	_       uint32
	pNext   uintptr
	size    uint32
	stages  uint32
	ops     uint32
	quadAll uint32
}

// sizeCtlProps is VkPhysicalDeviceSubgroupSizeControlProperties: the RANGE the
// driver may choose from, and which shader stages will accept a pinned width.
// reqStages must contain VK_SHADER_STAGE_COMPUTE_BIT for any of this to reach a
// compute pipeline.
type sizeCtlProps struct {
	sType     uint32
	_         uint32
	pNext     uintptr
	min       uint32
	max       uint32
	maxGroups uint32
	reqStages uint32
}

// sizeCtlFeat is VkPhysicalDeviceSubgroupSizeControlFeatures, chained into
// VkDeviceCreateInfo. Pinning a width needs subgroupSizeControl enabled and the
// REQUIRE_FULL_SUBGROUPS flag needs computeFullSubgroups: a supported extension
// that is not enabled buys nothing.
type sizeCtlFeat struct {
	sType   uint32
	_       uint32
	pNext   uintptr
	control uint32
	full    uint32
}

// reqSizeCI is VkPipelineShaderStageRequiredSubgroupSizeCreateInfo, chained into
// the compute pipeline's stage. This is the whole guarantee: with it the driver
// must run the shader at `size` lanes or fail to create the pipeline.
type reqSizeCI struct {
	sType uint32
	_     uint32
	pNext uintptr
	size  uint32
	_     uint32
}

// ---------------------------------------------------------------------------
// VK_EXT_external_memory_host: host memory used as a buffer, without a copy.
//
// An integrated GPU's heap is system memory, so uploading a weight copies host
// memory to host memory. With this extension the driver takes the host pointer
// (the packed arena's, kernels.Arena) and a page-in becomes a descriptor
// update. A discrete card may advertise it too, where imported weights would be
// re-read across PCIe; the caller's guard is UnifiedMemory().
//
// Every offset below is offsetof() from vulkan_core.h; layout_test.go asserts
// each one.

const (
	extExternalMemoryHost = "VK_EXT_external_memory_host"

	stExtMemBufferCI   = 1000072002 // VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO
	stImportMemHostPtr = 1000178000 // ..._IMPORT_MEMORY_HOST_POINTER_INFO_EXT
	stMemHostPtrProps  = 1000178001 // ..._MEMORY_HOST_POINTER_PROPERTIES_EXT
	stExtMemHostProps  = 1000178002 // ..._PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT

	// VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT: memory the
	// application allocated itself, as opposed to a mapped file or another
	// API's allocation.
	handleTypeHostAlloc = 0x00000080
)

// extMemBufCI is VkExternalMemoryBufferCreateInfo, chained onto the buffer's
// create info so the driver knows the memory it will be bound to is imported.
type extMemBufCI struct {
	sType   uint32
	_       uint32
	pNext   uintptr
	handles uint32
	_       uint32
}

// importHostPtrInfo is VkImportMemoryHostPointerInfoEXT, chained onto
// VkMemoryAllocateInfo. The allocation then wraps the pointer instead of
// producing memory of its own, and vkFreeMemory releases the wrapper without
// touching the host bytes.
type importHostPtrInfo struct {
	sType      uint32
	_          uint32
	pNext      uintptr
	handleType uint32
	_          uint32
	p          uintptr
}

// memHostPtrProps is VkMemoryHostPointerPropertiesEXT: which memory types may
// be used to import this pointer. The query is required: the type index handed
// to vkAllocateMemory must be in its mask, and guessing is invalid usage.
type memHostPtrProps struct {
	sType    uint32
	_        uint32
	pNext    uintptr
	typeBits uint32
	_        uint32
}

// extMemHostProps is VkPhysicalDeviceExternalMemoryHostPropertiesEXT: the
// alignment an imported pointer and its length must both satisfy (typically
// 4096); kernels.Arena lays its arrays out on at least that.
type extMemHostProps struct {
	sType    uint32
	_        uint32
	pNext    uintptr
	minAlign uint64
}
